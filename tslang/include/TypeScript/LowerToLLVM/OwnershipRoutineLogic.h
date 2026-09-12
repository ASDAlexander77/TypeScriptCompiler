#ifndef MLIR_TYPESCRIPT_LOWERTOLLVMLOGIC_OWNERSHIPROUTINELOGIC_H_
#define MLIR_TYPESCRIPT_LOWERTOLLVMLOGIC_OWNERSHIPROUTINELOGIC_H_

#include "TypeScript/Config.h"
#include "TypeScript/Defines.h"
#include "TypeScript/TypeScriptOps.h"

#include "TypeScript/MLIRLogic/MLIRTypeHelper.h"

#include "TypeScript/LowerToLLVM/TypeHelper.h"
#include "TypeScript/LowerToLLVM/TypeConverterHelper.h"
#include "TypeScript/LowerToLLVM/CodeLogicHelper.h"
#include "TypeScript/LowerToLLVM/LLVMCodeHelperBase.h"
#include "TypeScript/LowerToLLVM/TypeDescriptorLogic.h"

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"

using namespace mlir;
namespace mlir_ts = mlir::typescript;

namespace typescript
{

// Generates, once per type, the routine that drops one reference to a value of that type:
// each heap block it owns loses a reference, and the ones that lose their last are destroyed -
// their fields released in turn, then freed. The routine's address goes in the type's
// descriptor (TYPE_DESCR_RELEASE), which is the only thing that references it -- nothing calls
// these yet. See docs/reference-counting-evaluation.md sections 9.4 and 9.6.
//
// The routines are reference-counting shaped in every memory model, because they are dead code
// in all but `-mm=rc`, and one shape is simpler than two. Only `-mm=rc` initialises the count
// they read (LLVMCodeHelperBase::_MemoryAlloc).
//
// Calling convention: the routine takes a pointer to the *storage holding* a value of the
// type, not the value. That is uniform across value categories - a class field, an "any"
// payload slot and a local variable are all addressed the same way - and it is what lets a
// field's release be a plain call with a GEP.
class OwnershipRoutineLogic
{
    Operation *op;
    PatternRewriter &rewriter;
    const TypeConverter *typeConverter;
    CompileOptions &compileOptions;

  public:
    OwnershipRoutineLogic(Operation *op, PatternRewriter &rewriter, const TypeConverter *typeConverter,
                        CompileOptions &compileOptions)
        : op(op), rewriter(rewriter), typeConverter(typeConverter), compileOptions(compileOptions)
    {
    }

    // Symbol name of the routine for `type`, generating it if needed. Empty when the type
    // owns no heap memory, in which case the descriptor's release slot stays null - a null
    // slot means "nothing to release", not "unknown".
    // A generated ownership routine belongs to no user function, so it must not carry that
    // function's DISubprogram - but it must still carry a location.
    //
    // Every op in one of these routines is built with `op->getLoc()`, the location of whatever
    // op triggered the generation, because that is the only location in scope. Under `--di` that
    // is a FusedLoc carrying the *enclosing* function's DISubprogram, and the translation
    // attaches it to whatever function it lands on - so `main`'s subprogram was attached to
    // `tsrel_...`, `tsret_...`, `__tslang_inc_ref` and `__tslang_free_block` as well, and
    // "DISubprogram attached to more than one function" failed the module. No reference-counted
    // program could be built with debug info at all, in any tier.
    //
    // Dropping the location entirely does not work, and the way it fails is worth recording:
    // `DIScopeForLLVMFuncOpPass` then *gives* the routine a fresh subprogram of its own, and its
    // body ops - now at unknown locations - trip the opposite check, "inlinable function call in
    // a function with a DISubprogram location must have a debug location".
    //
    // So the subprogram is peeled off and the file/line underneath it kept. The routine reaches
    // that pass with no subprogram and a real location, gets one of its own, and every op in it
    // still has somewhere to hang. Section 9.69.
    // Reduces a location to its plain file/line, discarding every layer of debug scope wrapped
    // around it - the fused DISubprogram, a fused DILocalVariable, a call-site chain.
    //
    // Peeling only the subprogram is not enough, and how that fails is worth recording: an op
    // whose location is fused with a DILocalVariable still names a scope inside the *user*
    // function, so the routine gets "!dbg attachment points at wrong subprogram for function"
    // instead. These routines have no source variables and no call sites of their own, so
    // there is nothing here worth keeping above the file and line.
    static mlir::Location plainLocation(mlir::Location loc)
    {
        if (auto fused = mlir::dyn_cast<mlir::FusedLoc>(loc))
        {
            auto nested = fused.getLocations();
            return nested.empty() ? mlir::UnknownLoc::get(loc.getContext()) : plainLocation(nested.front());
        }

        if (auto callSite = mlir::dyn_cast<mlir::CallSiteLoc>(loc))
        {
            return plainLocation(callSite.getCallee());
        }

        // A NameLoc wraps the location it names rather than replacing it, so a scope can hide
        // one layer further in: `loc("sAny"(fused<#di_subprogram<main>>[...]))` is what a block
        // argument of a generated routine carries, and unwrapping only the fused and call-site
        // forms walks straight past it.
        if (auto named = mlir::dyn_cast<mlir::NameLoc>(loc))
        {
            return plainLocation(named.getChildLoc());
        }

        return loc;
    }

    static void dropDebugLocations(mlir::LLVM::LLVMFuncOp funcOp)
    {
        funcOp->setLoc(plainLocation(funcOp->getLoc()));
        funcOp->walk([](mlir::Operation *nested) { nested->setLoc(plainLocation(nested->getLoc())); });

        // Block arguments carry locations of their own, and walking operations does not reach
        // them. They become phis, so leaving them alone leaves the routine with
        // `%11 = phi i64 ..., !dbg !39` naming the user function's scope - the same "wrong
        // subprogram" complaint, arriving from the one place the operation walk cannot see.
        funcOp->walk([](mlir::Block *block) {
            for (auto arg : block->getArguments())
            {
                arg.setLoc(plainLocation(arg.getLoc()));
            }
        });
    }

    std::string getOrCreateReleaseRoutine(mlir::Type type)
    {
        if (!ownsHeapMemory(type))
        {
            return {};
        }

        auto name = getRoutineName(type);
        auto parentModule = op->getParentOfType<ModuleOp>();
        if (parentModule.lookupSymbol<LLVM::LLVMFuncOp>(name))
        {
            return name;
        }

        TypeHelper th(rewriter);
        auto loc = op->getLoc();

        OpBuilder::InsertionGuard insertGuard(rewriter);
        rewriter.setInsertionPointToStart(parentModule.getBody());

        auto funcOp = rewriter.create<LLVM::LLVMFuncOp>(
            loc, name, th.getFunctionType(th.getVoidType(), {th.getPtrType()}), LLVM::Linkage::Internal);

        // the symbol must exist before the body is built: a recursive type (`class Node {
        // next: Node }`) reaches its own routine while generating it
        auto *entryBlock = funcOp.addEntryBlock(rewriter);
        rewriter.setInsertionPointToStart(entryBlock);

        buildBody(type, entryBlock->getArgument(0));

        rewriter.create<LLVM::ReturnOp>(loc, ValueRange{});

        dropDebugLocations(funcOp);
        return name;
    }

    // Symbol name of the retain routine for `type`, generating it if needed. Empty when the
    // type owns no heap memory, in which case the descriptor's retain slot stays null.
    //
    // Like release, the routine addresses the storage holding a value rather than the value.
    std::string getOrCreateRetainRoutine(mlir::Type type)
    {
        if (!ownsHeapMemory(type))
        {
            return {};
        }

        auto name = getRetainRoutineName(type);
        auto parentModule = op->getParentOfType<ModuleOp>();
        if (parentModule.lookupSymbol<LLVM::LLVMFuncOp>(name))
        {
            return name;
        }

        TypeHelper th(rewriter);
        auto loc = op->getLoc();

        OpBuilder::InsertionGuard insertGuard(rewriter);
        rewriter.setInsertionPointToStart(parentModule.getBody());

        auto funcOp = rewriter.create<LLVM::LLVMFuncOp>(
            loc, name, th.getFunctionType(th.getVoidType(), {th.getPtrType()}), LLVM::Linkage::Internal);

        // as with release, the symbol must exist before the body is built, so that a
        // recursive type reaches its own routine while generating it
        auto *entryBlock = funcOp.addEntryBlock(rewriter);
        rewriter.setInsertionPointToStart(entryBlock);

        buildRetainBody(type, entryBlock->getArgument(0));

        rewriter.create<LLVM::ReturnOp>(loc, ValueRange{});

        dropDebugLocations(funcOp);
        return name;
    }

    // Takes one reference to `value`, whose TypeScript type is `type`. Emits nothing when the
    // type owns no heap memory. Wrapped for the same reason as emitReleaseValue.
    void emitRetainValue(mlir::Type type, mlir::Value value)
    {
        auto wrapperName = getOrCreateRetainValueRoutine(type);
        if (wrapperName.empty())
        {
            return;
        }

        rewriter.create<LLVM::CallOp>(op->getLoc(), TypeRange{},
                                      FlatSymbolRefAttr::get(rewriter.getContext(), wrapperName), ValueRange{value});
    }

    // Drops one reference held by `value`, whose TypeScript type is `type`. Emits nothing
    // when the type owns no heap memory.
    //
    // The per-type routines address storage rather than values, so this goes through a small
    // value-taking wrapper. Its alloca sits in the wrapper's own entry block, which keeps
    // every caller from having to find a safe place for one - a release inside a loop must not
    // grow the frame - and LLVM inlines and promotes the whole thing away.
    void emitReleaseValue(mlir::Type type, mlir::Value value)
    {
        auto wrapperName = getOrCreateReleaseValueRoutine(type);
        if (wrapperName.empty())
        {
            return;
        }

        rewriter.create<LLVM::CallOp>(op->getLoc(), TypeRange{},
                                      FlatSymbolRefAttr::get(rewriter.getContext(), wrapperName), ValueRange{value});
    }

    // Drops one reference held by the value in `slotPtr`, whose TypeScript type is `type`.
    // Emits nothing when the type owns no heap memory.
    //
    // The slot-taking form is what `ts.ReleaseSlot` lowers to: MLIRGen already has the
    // variable's storage in hand, so going through emitReleaseValue's alloca wrapper would
    // only spill a value that was already in memory.
    void emitReleaseSlot(mlir::Type type, mlir::Value slotPtr)
    {
        releaseSlot(type, slotPtr);
    }

    // Takes one reference on the value in `slotPtr`. The mirror of emitReleaseSlot.
    void emitRetainSlot(mlir::Type type, mlir::Value slotPtr)
    {
        retainSlot(type, slotPtr);
    }

    // === Capture cells ===
    //
    // A variable captured by reference does not live in its frame: its storage is a heap block
    // of its own, so that the frame and every closure over it read and write the same value.
    // `cellPtr` addresses that block, and these two count owners *of the block* - which is a
    // different question from emitRetainSlot/emitReleaseSlot, who count owners of the value in
    // it. See docs/reference-counting-evaluation.md section 9.34.

    void emitRetainCell(mlir::Value cellPtr)
    {
        emitIncRef(cellPtr);
    }

    // The value goes back before the block does, and in that order: releasing it reads the
    // cell.
    void emitReleaseCell(mlir::Type contentsType, mlir::Value cellPtr)
    {
        emitIfLastReference(cellPtr, [&]() {
            releaseSlot(contentsType, cellPtr);
            emitFreeBlock(cellPtr);
        });
    }

    // === Capture boxes ===
    //
    // A closure's `this` is a heap block with one field per captured variable, and it owns two
    // different things at once: the cell of each variable captured by reference, and whatever
    // the inline copy of each variable captured by value owns. The generic record routines
    // handle only the second - a RefType field is storage a value does not own, everywhere
    // else in the compiler - so a capture box gets routines of its own.
    //
    // They are keyed by the capture's own `ref<tuple<..>>` type rather than by the box's
    // `object<tuple<..>>`, so that the descriptor and the routines cannot collide with the
    // generic ones for an object of the same shape.
    std::string getOrCreateCaptureBoxReleaseRoutine(mlir_ts::RefType captureRefType)
    {
        return buildCaptureBoxRoutine(captureRefType, "tsrelcb_", /*retaining=*/false);
    }

    // Copying a closure duplicates its one reference to the box and nothing else - what the
    // box holds is not duplicated - so this stops at the block, as an object's retain does.
    std::string getOrCreateCaptureBoxRetainRoutine(mlir_ts::RefType captureRefType)
    {
        return buildCaptureBoxRoutine(captureRefType, "tsretcb_", /*retaining=*/true);
    }

    // Does a value of this type own heap memory, directly or through its fields? The same
    // question decides both directions: a type with nothing to release has nothing to retain.
    //
    // The answer lives in MLIRTypeHelper because MLIRGen has to ask it too - it is what
    // decides whether a local is an owner - and the two sides disagreeing would place retains
    // that never pair with a release.
    bool ownsHeapMemory(mlir::Type type)
    {
        MLIRTypeHelper mth(rewriter.getContext(), compileOptions);
        return mth.ownsHeapMemory(op->getLoc(), type);
    }

    // Gives back the references held by `count` elements starting at `startIndex`, in an array
    // whose data begins at `dataPtr`.
    //
    // `splice` is what needs this: the elements it removes are memmoved over, so the references
    // in those slots are dropped on the floor rather than released. Every other insertion point
    // in this arc sits in MLIRGen, where the number of elements involved is a compile-time
    // matter; here it is a runtime value known only at this level, which is why this one release
    // is emitted from the lowering. See §9.74.
    //
    // Emitted **only under `-mm=rc`**, and that check cannot be skipped. The release routines are
    // reference-counting shaped in every memory model and are dead weight under `gc` (§9.4); what
    // keeps them dead there is that `ts.Release` erases on the way to LLVM (§9.10). A direct call
    // planted by a lowering has no such eraser in front of it, so without this guard `gc` would
    // start freeing objects it is still tracing.
    void emitReleaseArrayElements(mlir::Type elementType, mlir::Value dataPtr, mlir::Value startIndex,
                                  mlir::Value count)
    {
        if (!compileOptions.isRefCounted())
        {
            return;
        }

        auto routineName = getOrCreateReleaseRoutine(elementType);
        if (routineName.empty())
        {
            return;
        }

        TypeHelper th(rewriter);
        TypeConverterHelper tch(typeConverter);

        auto loc = op->getLoc();
        auto ptrTy = th.getPtrType();
        auto llvmElementType = tch.convertType(elementType);

        emitCountedLoop(count, [&](mlir::Value index) {
            auto offset = rewriter.create<LLVM::AddOp>(loc, index.getType(), index, startIndex);
            auto elementPtr =
                rewriter.create<LLVM::GEPOp>(loc, ptrTy, llvmElementType, dataPtr, ValueRange{offset});
            rewriter.create<LLVM::CallOp>(loc, TypeRange{},
                                          FlatSymbolRefAttr::get(rewriter.getContext(), routineName),
                                          ValueRange{elementPtr});
        });
    }

  private:
    // Field types of a record-shaped type, empty for anything else.
    llvm::SmallVector<mlir::Type> getFieldTypes(mlir::Type type)
    {
        MLIRTypeHelper mth(rewriter.getContext(), compileOptions);
        return mth.getOwnershipFieldTypes(type);
    }

    std::string getOrCreateReleaseValueRoutine(mlir::Type type)
    {
        return getOrCreateValueWrapper(type, getOrCreateReleaseRoutine(type), "tsrelv_");
    }

    std::string getRoutineName(mlir::Type type)
    {
        std::stringstream ss;
        ss << "tsrel_" << (size_t)hash_value(type);
        return ss.str();
    }

    std::string getRetainRoutineName(mlir::Type type)
    {
        std::stringstream ss;
        ss << "tsret_" << (size_t)hash_value(type);
        return ss.str();
    }

    std::string getOrCreateRetainValueRoutine(mlir::Type type)
    {
        return getOrCreateValueWrapper(type, getOrCreateRetainRoutine(type), "tsretv_");
    }

    // A value-taking wrapper around a storage-taking routine: stores the value into an alloca
    // in the wrapper's own entry block and calls through. Keeping the alloca here rather than
    // at each call site means a retain or release inside a loop never grows the caller's
    // frame, and LLVM inlines and promotes the whole thing away.
    std::string getOrCreateValueWrapper(mlir::Type type, std::string slotRoutine, StringRef prefix)
    {
        if (slotRoutine.empty())
        {
            return {};
        }

        std::stringstream nameStream;
        nameStream << prefix.str() << (size_t)hash_value(type);
        auto name = nameStream.str();

        auto parentModule = op->getParentOfType<ModuleOp>();
        if (parentModule.lookupSymbol<LLVM::LLVMFuncOp>(name))
        {
            return name;
        }

        TypeHelper th(rewriter);
        TypeConverterHelper tch(typeConverter);
        CodeLogicHelper clh(op, rewriter);

        auto loc = op->getLoc();
        auto llvmType = tch.convertType(type);

        OpBuilder::InsertionGuard insertGuard(rewriter);
        rewriter.setInsertionPointToStart(parentModule.getBody());

        auto funcOp = rewriter.create<LLVM::LLVMFuncOp>(loc, name, th.getFunctionType(th.getVoidType(), {llvmType}),
                                                        LLVM::Linkage::Internal);

        auto *entryBlock = funcOp.addEntryBlock(rewriter);
        rewriter.setInsertionPointToStart(entryBlock);

        auto slot = rewriter.create<LLVM::AllocaOp>(loc, th.getPtrType(), llvmType, clh.createI32ConstantOf(1));
        rewriter.create<LLVM::StoreOp>(loc, entryBlock->getArgument(0), slot);
        rewriter.create<LLVM::CallOp>(loc, TypeRange{}, FlatSymbolRefAttr::get(rewriter.getContext(), slotRoutine),
                                      ValueRange{slot});
        rewriter.create<LLVM::ReturnOp>(loc, ValueRange{});

        dropDebugLocations(funcOp);
        return name;
    }

    // free(payload - headerSize). One generated helper rather than an inline free at every
    // site, so there is a single place for the allocator to change under `-mm=rc`.
    //
    // Only ever reached from inside emitIfLastReference, so the block is known to be mortal
    // and to have just lost its last reference.
    void emitFreeBlock(mlir::Value payloadPtr)
    {
        TypeHelper th(rewriter);
        auto loc = op->getLoc();
        auto parentModule = op->getParentOfType<ModuleOp>();

        const char *helperName = "__tslang_free_block";
        if (!parentModule.lookupSymbol<LLVM::LLVMFuncOp>(helperName))
        {
            OpBuilder::InsertionGuard insertGuard(rewriter);
            rewriter.setInsertionPointToStart(parentModule.getBody());

            auto helper = rewriter.create<LLVM::LLVMFuncOp>(
                loc, helperName, th.getFunctionType(th.getVoidType(), {th.getPtrType()}), LLVM::Linkage::Internal);

            auto *entryBlock = helper.addEntryBlock(rewriter);
            rewriter.setInsertionPointToStart(entryBlock);

            LLVMCodeHelperBase ch(op, rewriter, typeConverter, compileOptions);
            ch.MemoryFree(entryBlock->getArgument(0));

            rewriter.create<LLVM::ReturnOp>(loc, ValueRange{});

            dropDebugLocations(helper);
        }

        rewriter.create<LLVM::CallOp>(loc, TypeRange{}, FlatSymbolRefAttr::get(rewriter.getContext(), helperName),
                                      ValueRange{payloadPtr});
    }

    // Drops one reference to a block, answering "was that the last one?" - that is, should the
    // caller now destroy the value and free the block.
    //
    // A block marked HEAP_BLOCK_IMMORTAL is neither decremented nor ever the last. That is what
    // lets a string literal, or a `typeof` result pointing into a descriptor, be released like
    // any other string without writing to read-only memory or freeing a static block.
    mlir::Value emitDecRef(mlir::Value payloadPtr)
    {
        TypeHelper th(rewriter);
        auto loc = op->getLoc();
        auto parentModule = op->getParentOfType<ModuleOp>();

        const char *helperName = "__tslang_dec_ref";
        if (!parentModule.lookupSymbol<LLVM::LLVMFuncOp>(helperName))
        {
            OpBuilder::InsertionGuard insertGuard(rewriter);
            rewriter.setInsertionPointToStart(parentModule.getBody());

            auto helper = rewriter.create<LLVM::LLVMFuncOp>(
                loc, helperName, th.getFunctionType(th.getLLVMBoolType(), {th.getPtrType()}), LLVM::Linkage::Internal);

            auto *entryBlock = helper.addEntryBlock(rewriter);
            rewriter.setInsertionPointToStart(entryBlock);

            TypeConverterHelper tch(typeConverter);
            LLVMCodeHelperBase ch(op, rewriter, typeConverter, compileOptions);

            auto llvmIndexType = tch.convertType(th.getIndexType());
            auto blockPtr = ch.getBlockPtrFromPayloadPtr(loc, entryBlock->getArgument(0), llvmIndexType);
            auto count = rewriter.create<LLVM::LoadOp>(loc, llvmIndexType, blockPtr);
            auto immortal = rewriter.create<LLVM::ConstantOp>(
                loc, llvmIndexType, rewriter.getIntegerAttr(llvmIndexType, HEAP_BLOCK_IMMORTAL));
            auto isMortal = rewriter.create<LLVM::ICmpOp>(loc, LLVM::ICmpPredicate::ne, count, immortal);

            auto *decBlock = rewriter.createBlock(&helper.getBody(), helper.getBody().end());
            auto *immortalBlock = rewriter.createBlock(&helper.getBody(), helper.getBody().end());

            rewriter.setInsertionPointToEnd(entryBlock);
            rewriter.create<LLVM::CondBrOp>(loc, isMortal, decBlock, immortalBlock);

            rewriter.setInsertionPointToStart(decBlock);
            auto one = rewriter.create<LLVM::ConstantOp>(loc, llvmIndexType, rewriter.getIntegerAttr(llvmIndexType, 1));
            auto newCount = rewriter.create<LLVM::SubOp>(loc, llvmIndexType, count, one);
            rewriter.create<LLVM::StoreOp>(loc, newCount, blockPtr);
            auto zero = rewriter.create<LLVM::ConstantOp>(loc, llvmIndexType, rewriter.getIntegerAttr(llvmIndexType, 0));
            auto wasLast = rewriter.create<LLVM::ICmpOp>(loc, LLVM::ICmpPredicate::eq, newCount, zero);
            rewriter.create<LLVM::ReturnOp>(loc, ValueRange{wasLast});

            rewriter.setInsertionPointToStart(immortalBlock);
            rewriter.create<LLVM::ReturnOp>(
                loc, ValueRange{rewriter.create<LLVM::ConstantOp>(loc, th.getLLVMBoolType(),
                                                                  rewriter.getIntegerAttr(th.getLLVMBoolType(), 0))});

            dropDebugLocations(helper);
        }

        auto callOp = rewriter.create<LLVM::CallOp>(loc, TypeRange{th.getLLVMBoolType()},
                                                    FlatSymbolRefAttr::get(rewriter.getContext(), helperName),
                                                    ValueRange{payloadPtr});
        return callOp.getResult();
    }

    // Takes one more reference to a block, when there is a block and it is mortal.
    //
    // Skipping an immortal block is not an optimisation: incrementing HEAP_BLOCK_IMMORTAL
    // would turn all-ones into zero, and the next release would read that as "last reference"
    // and free a string literal.
    void emitIncRef(mlir::Value payloadPtr)
    {
        TypeHelper th(rewriter);
        auto loc = op->getLoc();
        auto parentModule = op->getParentOfType<ModuleOp>();

        const char *helperName = "__tslang_inc_ref";
        if (!parentModule.lookupSymbol<LLVM::LLVMFuncOp>(helperName))
        {
            OpBuilder::InsertionGuard insertGuard(rewriter);
            rewriter.setInsertionPointToStart(parentModule.getBody());

            auto helper = rewriter.create<LLVM::LLVMFuncOp>(
                loc, helperName, th.getFunctionType(th.getVoidType(), {th.getPtrType()}), LLVM::Linkage::Internal);

            auto *entryBlock = helper.addEntryBlock(rewriter);
            rewriter.setInsertionPointToStart(entryBlock);

            TypeConverterHelper tch(typeConverter);
            LLVMCodeHelperBase ch(op, rewriter, typeConverter, compileOptions);

            auto llvmIndexType = tch.convertType(th.getIndexType());
            auto payload = entryBlock->getArgument(0);

            auto nullPtr = rewriter.create<LLVM::ZeroOp>(loc, th.getPtrType());
            auto isNotNull = rewriter.create<LLVM::ICmpOp>(loc, LLVM::ICmpPredicate::ne, payload, nullPtr);

            auto *loadBlock = rewriter.createBlock(&helper.getBody(), helper.getBody().end());
            auto *incBlock = rewriter.createBlock(&helper.getBody(), helper.getBody().end());
            auto *returnBlock = rewriter.createBlock(&helper.getBody(), helper.getBody().end());

            rewriter.setInsertionPointToEnd(entryBlock);
            rewriter.create<LLVM::CondBrOp>(loc, isNotNull, loadBlock, returnBlock);

            rewriter.setInsertionPointToStart(loadBlock);
            auto blockPtr = ch.getBlockPtrFromPayloadPtr(loc, payload, llvmIndexType);
            auto count = rewriter.create<LLVM::LoadOp>(loc, llvmIndexType, blockPtr);
            auto immortal = rewriter.create<LLVM::ConstantOp>(
                loc, llvmIndexType, rewriter.getIntegerAttr(llvmIndexType, HEAP_BLOCK_IMMORTAL));
            auto isMortal = rewriter.create<LLVM::ICmpOp>(loc, LLVM::ICmpPredicate::ne, count, immortal);
            rewriter.create<LLVM::CondBrOp>(loc, isMortal, incBlock, returnBlock);

            rewriter.setInsertionPointToStart(incBlock);
            auto one = rewriter.create<LLVM::ConstantOp>(loc, llvmIndexType, rewriter.getIntegerAttr(llvmIndexType, 1));
            rewriter.create<LLVM::StoreOp>(loc, rewriter.create<LLVM::AddOp>(loc, llvmIndexType, count, one), blockPtr);
            rewriter.create<LLVM::BrOp>(loc, ValueRange{}, returnBlock);

            rewriter.setInsertionPointToStart(returnBlock);
            rewriter.create<LLVM::ReturnOp>(loc, ValueRange{});

            dropDebugLocations(helper);
        }

        rewriter.create<LLVM::CallOp>(loc, TypeRange{}, FlatSymbolRefAttr::get(rewriter.getContext(), helperName),
                                      ValueRange{payloadPtr});
    }

    // Runs `thenBody` -- the destroy half: release what the value owns, then free it -- only
    // when `payloadPtr` is non-null and the reference being dropped was the last one.
    void emitIfLastReference(mlir::Value payloadPtr, llvm::function_ref<void()> thenBody)
    {
        TypeHelper th(rewriter);
        auto loc = op->getLoc();

        auto *currentBlock = rewriter.getInsertionBlock();
        auto *continuationBlock = rewriter.splitBlock(currentBlock, rewriter.getInsertionPoint());
        auto *thenBlock = rewriter.createBlock(continuationBlock);
        auto *decBlock = rewriter.createBlock(thenBlock);

        rewriter.setInsertionPointToEnd(thenBlock);
        thenBody();
        rewriter.create<LLVM::BrOp>(loc, ValueRange{}, continuationBlock);

        rewriter.setInsertionPointToEnd(decBlock);
        auto wasLast = emitDecRef(payloadPtr);
        rewriter.create<LLVM::CondBrOp>(loc, wasLast, thenBlock, continuationBlock);

        rewriter.setInsertionPointToEnd(currentBlock);
        auto nullPtr = rewriter.create<LLVM::ZeroOp>(loc, th.getPtrType());
        auto isNotNull = rewriter.create<LLVM::ICmpOp>(loc, LLVM::ICmpPredicate::ne, payloadPtr, nullPtr);
        rewriter.create<LLVM::CondBrOp>(loc, isNotNull, decBlock, continuationBlock);

        rewriter.setInsertionPointToStart(continuationBlock);
    }

    // Runs `thenBody` only when `ptrValue` is not null, and leaves the insertion point on the
    // continuation.
    void emitIfNonNull(mlir::Value ptrValue, llvm::function_ref<void()> thenBody)
    {
        TypeHelper th(rewriter);
        auto loc = op->getLoc();

        auto *currentBlock = rewriter.getInsertionBlock();
        auto *continuationBlock = rewriter.splitBlock(currentBlock, rewriter.getInsertionPoint());
        auto *thenBlock = rewriter.createBlock(continuationBlock);

        rewriter.setInsertionPointToEnd(thenBlock);
        thenBody();
        rewriter.create<LLVM::BrOp>(loc, ValueRange{}, continuationBlock);

        rewriter.setInsertionPointToEnd(currentBlock);
        auto nullPtr = rewriter.create<LLVM::ZeroOp>(loc, th.getPtrType());
        auto isNotNull = rewriter.create<LLVM::ICmpOp>(loc, LLVM::ICmpPredicate::ne, ptrValue, nullPtr);
        rewriter.create<LLVM::CondBrOp>(loc, isNotNull, thenBlock, continuationBlock);

        rewriter.setInsertionPointToStart(continuationBlock);
    }

    // Calls the release routine of `type` on `slotPtr`, if it has one.
    void releaseSlot(mlir::Type type, mlir::Value slotPtr)
    {
        auto routineName = getOrCreateReleaseRoutine(type);
        if (routineName.empty())
        {
            return;
        }

        rewriter.create<LLVM::CallOp>(op->getLoc(), TypeRange{},
                                      FlatSymbolRefAttr::get(rewriter.getContext(), routineName), ValueRange{slotPtr});
    }

    // Releases each field a record-shaped value owns. `basePtr` addresses the record itself.
    void releaseFields(mlir::Type recordType, mlir::Value basePtr)
    {
        TypeHelper th(rewriter);
        TypeConverterHelper tch(typeConverter);

        auto loc = op->getLoc();
        auto llvmRecordType = tch.convertType(recordType);

        for (auto [index, fieldType] : llvm::enumerate(getFieldTypes(recordType)))
        {
            // A field holding a `ref` to a tuple is a capture box - the `.captured` field of an
            // object literal with methods, or of the state object a generator becomes - and it is
            // the one field an object owns that the generic routines cannot see, because a
            // reference into storage is not ownership anywhere else in the compiler. It gets the
            // same treatment a closure's `this` gets (§9.33): give back the cells it holds, then
            // free the box. Without it the box and every cell under it outlive the object that was
            // their only owner, which is a generator's parameter leaking once per call (§9.52).
            //
            // Nothing else produces a `ref<tuple<..>>` field: `ref` is not spellable in the
            // language, so this shape is the compiler's own and always means a capture box.
            auto routineName = std::string();
            if (auto refFieldType = dyn_cast<mlir_ts::RefType>(fieldType))
            {
                if (isa<mlir_ts::TupleType>(refFieldType.getElementType()))
                {
                    routineName = getOrCreateCaptureBoxReleaseRoutine(refFieldType);
                }
            }
            else
            {
                routineName = getOrCreateReleaseRoutine(fieldType);
            }

            if (routineName.empty())
            {
                continue;
            }

            auto fieldPtr = rewriter.create<LLVM::GEPOp>(loc, th.getPtrType(), llvmRecordType, basePtr,
                                                         ArrayRef<LLVM::GEPArg>{0, (int32_t)index});
            rewriter.create<LLVM::CallOp>(loc, TypeRange{},
                                          FlatSymbolRefAttr::get(rewriter.getContext(), routineName),
                                          ValueRange{fieldPtr});
        }
    }

    // Reads the release routine out of the descriptor `tagValue` names and calls it on
    // `valueSlotPtr`, when the descriptor has one. This is the payoff of tags pointing into
    // descriptors: a value whose type is only known at run time can still be released.
    void releaseViaDescriptor(mlir::Value tagValue, mlir::Value valueSlotPtr)
    {
        TypeHelper th(rewriter);
        TypeConverterHelper tch(typeConverter);
        TypeDescriptorLogic tdl(rewriter, tch, op->getLoc());

        auto loc = op->getLoc();

        auto recordPtr = tdl.getRecordPtrFromTag(tagValue);
        auto releasePtrSlot =
            rewriter.create<LLVM::GEPOp>(loc, th.getPtrType(),
                                         TypeDescriptorLogic::getRecordType(rewriter, tch.convertType(th.getIndexType())),
                                         recordPtr, ArrayRef<LLVM::GEPArg>{0, TYPE_DESCR_RELEASE});
        auto releaseFn = rewriter.create<LLVM::LoadOp>(loc, th.getPtrType(), releasePtrSlot);

        emitIfNonNull(releaseFn, [&]() {
            // indirect call: callee pointer is operand #0, the rest are call arguments, and
            // AttrSizedOperandSegments needs that split set explicitly
            mlir::SmallVector<mlir::Value> ops{releaseFn, valueSlotPtr};
            auto callOp = rewriter.create<LLVM::CallOp>(loc, TypeRange{}, ops);
            callOp.getProperties().setOperandSegmentSizes({static_cast<int32_t>(ops.size()), 0});
            callOp.setOpBundleSizes({});
        });
    }

    // Both directions for a value shaped { .., this, type } - an interface, and a bound or
    // hybrid function. They differ only in which descriptor slot they read: load the tag beside
    // `this`, and hand the address of the `this` field to the concrete type's own routine. That
    // address is what the routine wants either way - a release or retain routine takes the
    // storage holding a value, and the second field is exactly the storage holding the class,
    // object or capture-box reference.
    //
    // The tag is checked before anything reads through it: getRecordPtrFromTag walks backwards
    // from the tag to the record, so a null tag would be dereferenced, not skipped, by the
    // null check inside releaseViaDescriptor. A null tag is the ordinary case for a function
    // value that is not a closure, so this is not a corner.
    void releaseViaTagBesideThis(mlir::Type type, mlir::Value slotPtr, int32_t tagIndex, bool retaining)
    {
        TypeHelper th(rewriter);
        TypeConverterHelper tch(typeConverter);

        auto loc = op->getLoc();
        auto ptrTy = th.getPtrType();
        auto llvmType = tch.convertType(type);

        auto tagSlot = rewriter.create<LLVM::GEPOp>(loc, ptrTy, llvmType, slotPtr,
                                                    ArrayRef<LLVM::GEPArg>{0, tagIndex});
        auto tagValue = rewriter.create<LLVM::LoadOp>(loc, ptrTy, tagSlot);
        auto thisSlot = rewriter.create<LLVM::GEPOp>(loc, ptrTy, llvmType, slotPtr,
                                                     ArrayRef<LLVM::GEPArg>{0, THIS_VALUE_INDEX});

        emitIfNonNull(tagValue, [&]() {
            if (retaining)
            {
                retainViaDescriptor(tagValue, thisSlot);
            }
            else
            {
                releaseViaDescriptor(tagValue, thisSlot);
            }
        });
    }

    void releaseViaInterfaceTag(mlir::Type type, mlir::Value slotPtr, bool retaining)
    {
        releaseViaTagBesideThis(type, slotPtr, INTERFACE_TYPE_INDEX, retaining);
    }

    // Body of both capture-box routines: they take the storage holding the box pointer, like
    // every other routine, so each begins by loading the box out of it.
    std::string buildCaptureBoxRoutine(mlir_ts::RefType captureRefType, StringRef prefix, bool retaining)
    {
        std::stringstream nameStream;
        nameStream << prefix.str() << (size_t)hash_value(captureRefType);
        auto name = nameStream.str();

        auto parentModule = op->getParentOfType<ModuleOp>();
        if (parentModule.lookupSymbol<LLVM::LLVMFuncOp>(name))
        {
            return name;
        }

        TypeHelper th(rewriter);
        auto loc = op->getLoc();
        auto ptrTy = th.getPtrType();

        OpBuilder::InsertionGuard insertGuard(rewriter);
        rewriter.setInsertionPointToStart(parentModule.getBody());

        auto funcOp = rewriter.create<LLVM::LLVMFuncOp>(
            loc, name, th.getFunctionType(th.getVoidType(), {ptrTy}), LLVM::Linkage::Internal);

        auto *entryBlock = funcOp.addEntryBlock(rewriter);
        rewriter.setInsertionPointToStart(entryBlock);

        auto boxValue = rewriter.create<LLVM::LoadOp>(loc, ptrTy, entryBlock->getArgument(0));
        if (retaining)
        {
            emitIncRef(boxValue);
        }
        else
        {
            emitIfLastReference(boxValue, [&]() {
                releaseCapturedFields(captureRefType.getElementType(), boxValue);
                emitFreeBlock(boxValue);
            });
        }

        rewriter.create<LLVM::ReturnOp>(loc, ValueRange{});

        dropDebugLocations(funcOp);
        return name;
    }

    // Gives back what a dying box holds: one owner of each captured cell, and the contents of
    // each field captured by value.
    void releaseCapturedFields(mlir::Type tupleType, mlir::Value boxPtr)
    {
        TypeHelper th(rewriter);
        TypeConverterHelper tch(typeConverter);

        auto loc = op->getLoc();
        auto ptrTy = th.getPtrType();
        auto llvmTupleType = tch.convertType(tupleType);

        for (auto [index, fieldType] : llvm::enumerate(getFieldTypes(tupleType)))
        {
            auto refFieldType = dyn_cast<mlir_ts::RefType>(fieldType);
            if (!refFieldType && !ownsHeapMemory(fieldType))
            {
                continue;
            }

            auto fieldPtr = rewriter.create<LLVM::GEPOp>(loc, ptrTy, llvmTupleType, boxPtr,
                                                         ArrayRef<LLVM::GEPArg>{0, (int32_t)index});
            if (refFieldType)
            {
                // the field holds the cell's address, so the cell is one load further in
                emitReleaseCell(refFieldType.getElementType(), rewriter.create<LLVM::LoadOp>(loc, ptrTy, fieldPtr));
            }
            else
            {
                releaseSlot(fieldType, fieldPtr);
            }
        }
    }

    void buildBody(mlir::Type type, mlir::Value slotPtr)
    {
        TypeHelper th(rewriter);
        TypeConverterHelper tch(typeConverter);

        auto loc = op->getLoc();
        auto ptrTy = th.getPtrType();

        // a string is its own block
        if (isa<mlir_ts::StringType>(type))
        {
            auto strValue = rewriter.create<LLVM::LoadOp>(loc, ptrTy, slotPtr);
            emitIfLastReference(strValue, [&]() { emitFreeBlock(strValue); });
            return;
        }

        // an array value is { data, length }; it owns the data block and, through it, the
        // elements
        if (auto arrayType = dyn_cast<mlir_ts::ArrayType>(type))
        {
            buildArrayBody(arrayType, slotPtr);
            return;
        }

        // a class or object reference owns the instance block
        if (isa<mlir_ts::ClassType>(type) || isa<mlir_ts::ObjectType>(type))
        {
            auto storageType = isa<mlir_ts::ClassType>(type) ? cast<mlir_ts::ClassType>(type).getStorageType()
                                                             : cast<mlir_ts::ObjectType>(type).getStorageType();

            auto instanceValue = rewriter.create<LLVM::LoadOp>(loc, ptrTy, slotPtr);
            emitIfLastReference(instanceValue, [&]() {
                releaseFields(storageType, instanceValue);
                emitFreeBlock(instanceValue);
            });
            return;
        }

        // an "any" box owns its own block, and its payload's type is only known through the
        // tag
        if (isa<mlir_ts::AnyType>(type))
        {
            auto boxValue = rewriter.create<LLVM::LoadOp>(loc, ptrTy, slotPtr);
            emitIfLastReference(boxValue, [&]() {
                auto anyStructType = LLVM::LLVMStructType::getLiteral(
                    rewriter.getContext(), {tch.convertType(th.getIndexType()), ptrTy, th.getI8Type()}, false);

                auto tagSlot = rewriter.create<LLVM::GEPOp>(loc, ptrTy, anyStructType, boxValue,
                                                            ArrayRef<LLVM::GEPArg>{0, ANY_TYPE});
                auto tagValue = rewriter.create<LLVM::LoadOp>(loc, ptrTy, tagSlot);
                auto dataSlot = rewriter.create<LLVM::GEPOp>(loc, ptrTy, anyStructType, boxValue,
                                                             ArrayRef<LLVM::GEPArg>{0, ANY_DATA});

                releaseViaDescriptor(tagValue, dataSlot);
                emitFreeBlock(boxValue);
            });
            return;
        }

        // an interface is { vtable, this, type }: the reference it holds is `this`, and what
        // that points at is only known through the tag beside it. There is no block of the
        // interface's own to free - the value is a pair of pointers held wherever it sits.
        if (isa<mlir_ts::InterfaceType>(type))
        {
            releaseViaInterfaceTag(type, slotPtr, /*retaining=*/false);
            return;
        }

        // a closure is { func, this, type } and owns its `this` when that `this` is a capture
        // box - which is what the tag says, and says nothing where it is a bound method
        if (isa<mlir_ts::BoundFunctionType>(type) || isa<mlir_ts::HybridFunctionType>(type))
        {
            releaseViaTagBesideThis(type, slotPtr, CLOSURE_TYPE_INDEX, /*retaining=*/false);
            return;
        }

        // a tagged union carries its payload inline, so there is no block of its own to
        // free - only the payload to release, again through the tag
        if (auto unionType = dyn_cast<mlir_ts::UnionType>(type))
        {
            MLIRTypeHelper mth(rewriter.getContext(), compileOptions);
            mlir::Type baseType;
            if (mth.isUnionTypeNeedsTag(loc, unionType, baseType))
            {
                auto llvmUnionType = tch.convertType(unionType);
                auto tagSlot = rewriter.create<LLVM::GEPOp>(loc, ptrTy, llvmUnionType, slotPtr,
                                                            ArrayRef<LLVM::GEPArg>{0, UNION_TAG_INDEX});
                auto tagValue = rewriter.create<LLVM::LoadOp>(loc, ptrTy, tagSlot);
                auto valueSlot = rewriter.create<LLVM::GEPOp>(loc, ptrTy, llvmUnionType, slotPtr,
                                                              ArrayRef<LLVM::GEPArg>{0, UNION_VALUE_INDEX});

                // A null tag is a union that holds nothing yet, and that is the ordinary case
                // rather than a corner: a class instance comes out of calloc zeroed, and the
                // first assignment to a union field releases the old value before storing the
                // new one. The check has to be here, because getRecordPtrFromTag walks
                // backwards from the tag to find the descriptor - it would read through the
                // null long before the null check inside releaseViaDescriptor.
                emitIfNonNull(tagValue, [&]() { releaseViaDescriptor(tagValue, valueSlot); });
            }
            else
            {
                releaseSlot(baseType, slotPtr);
            }

            return;
        }

        // an optional carries its value inline behind a flag
        if (auto optionalType = dyn_cast<mlir_ts::OptionalType>(type))
        {
            buildOptionalBody(optionalType, slotPtr);
            return;
        }

        // everything left is record-shaped and inline: release what the fields own, free
        // nothing, because this value's storage belongs to whoever holds it
        releaseFields(type, slotPtr);
    }

    // Calls the retain routine of `type` on `slotPtr`, if it has one.
    void retainSlot(mlir::Type type, mlir::Value slotPtr)
    {
        auto routineName = getOrCreateRetainRoutine(type);
        if (routineName.empty())
        {
            return;
        }

        rewriter.create<LLVM::CallOp>(op->getLoc(), TypeRange{},
                                      FlatSymbolRefAttr::get(rewriter.getContext(), routineName), ValueRange{slotPtr});
    }

    // Retains each field an inline record-shaped value owns. `basePtr` addresses the record.
    void retainFields(mlir::Type recordType, mlir::Value basePtr)
    {
        TypeHelper th(rewriter);
        TypeConverterHelper tch(typeConverter);

        auto loc = op->getLoc();
        auto llvmRecordType = tch.convertType(recordType);

        for (auto [index, fieldType] : llvm::enumerate(getFieldTypes(recordType)))
        {
            auto routineName = getOrCreateRetainRoutine(fieldType);
            if (routineName.empty())
            {
                continue;
            }

            auto fieldPtr = rewriter.create<LLVM::GEPOp>(loc, th.getPtrType(), llvmRecordType, basePtr,
                                                         ArrayRef<LLVM::GEPArg>{0, (int32_t)index});
            rewriter.create<LLVM::CallOp>(loc, TypeRange{},
                                          FlatSymbolRefAttr::get(rewriter.getContext(), routineName),
                                          ValueRange{fieldPtr});
        }
    }

    // The retain counterpart of releaseViaDescriptor, reading the descriptor's retain slot.
    void retainViaDescriptor(mlir::Value tagValue, mlir::Value valueSlotPtr)
    {
        TypeHelper th(rewriter);
        TypeConverterHelper tch(typeConverter);
        TypeDescriptorLogic tdl(rewriter, tch, op->getLoc());

        auto loc = op->getLoc();

        auto recordPtr = tdl.getRecordPtrFromTag(tagValue);
        auto retainPtrSlot =
            rewriter.create<LLVM::GEPOp>(loc, th.getPtrType(),
                                         TypeDescriptorLogic::getRecordType(rewriter, tch.convertType(th.getIndexType())),
                                         recordPtr, ArrayRef<LLVM::GEPArg>{0, TYPE_DESCR_RETAIN});
        auto retainFn = rewriter.create<LLVM::LoadOp>(loc, th.getPtrType(), retainPtrSlot);

        emitIfNonNull(retainFn, [&]() {
            mlir::SmallVector<mlir::Value> ops{retainFn, valueSlotPtr};
            auto callOp = rewriter.create<LLVM::CallOp>(loc, TypeRange{}, ops);
            callOp.getProperties().setOperandSegmentSizes({static_cast<int32_t>(ops.size()), 0});
            callOp.setOpBundleSizes({});
        });
    }

    // The mirror of buildBody, and it is shorter for one reason worth stating plainly:
    // retaining a *reference* stops at the block. Release recurses into an object's fields,
    // but only inside emitIfLastReference - that is, only when the block is about to die and
    // its fields' references die with it. A second reference to the same object does not
    // duplicate the object's own references to its fields, so retain must not touch them.
    // Only values held *inline* - tuples, optionals, tagged unions - propagate a retain
    // inwards, because copying one really does duplicate every reference it holds.
    void buildRetainBody(mlir::Type type, mlir::Value slotPtr)
    {
        TypeHelper th(rewriter);
        TypeConverterHelper tch(typeConverter);

        auto loc = op->getLoc();
        auto ptrTy = th.getPtrType();

        // each of these is a reference to a block of its own: string, class and object
        // instances, and an "any" box
        if (isa<mlir_ts::StringType>(type) || isa<mlir_ts::ClassType>(type) || isa<mlir_ts::ObjectType>(type) ||
            isa<mlir_ts::AnyType>(type))
        {
            emitIncRef(rewriter.create<LLVM::LoadOp>(loc, ptrTy, slotPtr));
            return;
        }

        // an array value is { data, length }: the copy shares the data block, and the block
        // already holds whatever the elements own
        if (auto arrayType = dyn_cast<mlir_ts::ArrayType>(type))
        {
            auto llvmArrayType = tch.convertType(arrayType);
            auto dataSlot = rewriter.create<LLVM::GEPOp>(loc, ptrTy, llvmArrayType, slotPtr,
                                                         ArrayRef<LLVM::GEPArg>{0, ARRAY_DATA_INDEX});
            emitIncRef(rewriter.create<LLVM::LoadOp>(loc, ptrTy, dataSlot));
            return;
        }

        // an interface holds one reference, to its `this`; copying the pair duplicates that
        // reference and nothing else, so this stops at the block like the cases above
        if (isa<mlir_ts::InterfaceType>(type))
        {
            releaseViaInterfaceTag(type, slotPtr, /*retaining=*/true);
            return;
        }

        // copying a closure duplicates its one reference to the capture box and nothing else
        if (isa<mlir_ts::BoundFunctionType>(type) || isa<mlir_ts::HybridFunctionType>(type))
        {
            releaseViaTagBesideThis(type, slotPtr, CLOSURE_TYPE_INDEX, /*retaining=*/true);
            return;
        }

        // a tagged union carries its payload inline, so what it holds is copied with it
        if (auto unionType = dyn_cast<mlir_ts::UnionType>(type))
        {
            MLIRTypeHelper mth(rewriter.getContext(), compileOptions);
            mlir::Type baseType;
            if (mth.isUnionTypeNeedsTag(loc, unionType, baseType))
            {
                auto llvmUnionType = tch.convertType(unionType);
                auto tagSlot = rewriter.create<LLVM::GEPOp>(loc, ptrTy, llvmUnionType, slotPtr,
                                                            ArrayRef<LLVM::GEPArg>{0, UNION_TAG_INDEX});
                auto tagValue = rewriter.create<LLVM::LoadOp>(loc, ptrTy, tagSlot);
                auto valueSlot = rewriter.create<LLVM::GEPOp>(loc, ptrTy, llvmUnionType, slotPtr,
                                                              ArrayRef<LLVM::GEPArg>{0, UNION_VALUE_INDEX});

                // see the release side: a zeroed union holds nothing, and the tag has to be
                // checked here rather than inside retainViaDescriptor, which reads through it
                emitIfNonNull(tagValue, [&]() { retainViaDescriptor(tagValue, valueSlot); });
            }
            else
            {
                retainSlot(baseType, slotPtr);
            }

            return;
        }

        if (auto optionalType = dyn_cast<mlir_ts::OptionalType>(type))
        {
            buildRetainOptionalBody(optionalType, slotPtr);
            return;
        }

        // everything left is record-shaped and inline
        retainFields(type, slotPtr);
    }

    void buildRetainOptionalBody(mlir_ts::OptionalType optionalType, mlir::Value slotPtr)
    {
        TypeHelper th(rewriter);
        TypeConverterHelper tch(typeConverter);

        auto loc = op->getLoc();
        auto ptrTy = th.getPtrType();
        auto llvmOptionalType = tch.convertType(optionalType);

        auto hasValueSlot = rewriter.create<LLVM::GEPOp>(loc, ptrTy, llvmOptionalType, slotPtr,
                                                         ArrayRef<LLVM::GEPArg>{0, OPTIONAL_HASVALUE_INDEX});
        auto hasValue = rewriter.create<LLVM::LoadOp>(loc, th.getLLVMBoolType(), hasValueSlot);

        auto *currentBlock = rewriter.getInsertionBlock();
        auto *continuationBlock = rewriter.splitBlock(currentBlock, rewriter.getInsertionPoint());
        auto *thenBlock = rewriter.createBlock(continuationBlock);

        rewriter.setInsertionPointToEnd(thenBlock);
        auto valueSlot = rewriter.create<LLVM::GEPOp>(loc, ptrTy, llvmOptionalType, slotPtr,
                                                      ArrayRef<LLVM::GEPArg>{0, OPTIONAL_VALUE_INDEX});
        retainSlot(optionalType.getElementType(), valueSlot);
        rewriter.create<LLVM::BrOp>(loc, ValueRange{}, continuationBlock);

        rewriter.setInsertionPointToEnd(currentBlock);
        rewriter.create<LLVM::CondBrOp>(loc, hasValue, thenBlock, continuationBlock);

        rewriter.setInsertionPointToStart(continuationBlock);
    }

    void buildOptionalBody(mlir_ts::OptionalType optionalType, mlir::Value slotPtr)
    {
        TypeHelper th(rewriter);
        TypeConverterHelper tch(typeConverter);

        auto loc = op->getLoc();
        auto ptrTy = th.getPtrType();
        auto llvmOptionalType = tch.convertType(optionalType);

        auto hasValueSlot = rewriter.create<LLVM::GEPOp>(loc, ptrTy, llvmOptionalType, slotPtr,
                                                         ArrayRef<LLVM::GEPArg>{0, OPTIONAL_HASVALUE_INDEX});
        auto hasValue = rewriter.create<LLVM::LoadOp>(loc, th.getLLVMBoolType(), hasValueSlot);

        auto *currentBlock = rewriter.getInsertionBlock();
        auto *continuationBlock = rewriter.splitBlock(currentBlock, rewriter.getInsertionPoint());
        auto *thenBlock = rewriter.createBlock(continuationBlock);

        rewriter.setInsertionPointToEnd(thenBlock);
        auto valueSlot = rewriter.create<LLVM::GEPOp>(loc, ptrTy, llvmOptionalType, slotPtr,
                                                      ArrayRef<LLVM::GEPArg>{0, OPTIONAL_VALUE_INDEX});
        releaseSlot(optionalType.getElementType(), valueSlot);
        rewriter.create<LLVM::BrOp>(loc, ValueRange{}, continuationBlock);

        rewriter.setInsertionPointToEnd(currentBlock);
        rewriter.create<LLVM::CondBrOp>(loc, hasValue, thenBlock, continuationBlock);

        rewriter.setInsertionPointToStart(continuationBlock);
    }

    void buildArrayBody(mlir_ts::ArrayType arrayType, mlir::Value slotPtr)
    {
        TypeHelper th(rewriter);
        TypeConverterHelper tch(typeConverter);

        auto loc = op->getLoc();
        auto ptrTy = th.getPtrType();
        auto llvmIndexType = tch.convertType(th.getIndexType());
        auto llvmArrayType = tch.convertType(arrayType);

        auto dataSlot = rewriter.create<LLVM::GEPOp>(loc, ptrTy, llvmArrayType, slotPtr,
                                                     ArrayRef<LLVM::GEPArg>{0, ARRAY_DATA_INDEX});
        auto dataValue = rewriter.create<LLVM::LoadOp>(loc, ptrTy, dataSlot);

        emitIfLastReference(dataValue, [&]() {
            auto elementRoutine = getOrCreateReleaseRoutine(arrayType.getElementType());
            if (!elementRoutine.empty())
            {
                auto sizeSlot = rewriter.create<LLVM::GEPOp>(loc, ptrTy, llvmArrayType, slotPtr,
                                                             ArrayRef<LLVM::GEPArg>{0, ARRAY_SIZE_INDEX});
                auto sizeValue = rewriter.create<LLVM::LoadOp>(loc, llvmIndexType, sizeSlot);

                emitCountedLoop(sizeValue, [&](mlir::Value index) {
                    auto llvmElementType = tch.convertType(arrayType.getElementType());
                    auto elementPtr = rewriter.create<LLVM::GEPOp>(loc, ptrTy, llvmElementType, dataValue,
                                                                   ValueRange{index});
                    rewriter.create<LLVM::CallOp>(loc, TypeRange{},
                                                  FlatSymbolRefAttr::get(rewriter.getContext(), elementRoutine),
                                                  ValueRange{elementPtr});
                });
            }

            emitFreeBlock(dataValue);
        });
    }

    // for (index = 0; index < count; index++) body(index)
    void emitCountedLoop(mlir::Value count, llvm::function_ref<void(mlir::Value)> body)
    {
        TypeHelper th(rewriter);
        TypeConverterHelper tch(typeConverter);

        auto loc = op->getLoc();
        auto llvmIndexType = tch.convertType(th.getIndexType());

        auto *currentBlock = rewriter.getInsertionBlock();
        auto *continuationBlock = rewriter.splitBlock(currentBlock, rewriter.getInsertionPoint());

        auto *conditionBlock = rewriter.createBlock(continuationBlock, {llvmIndexType}, {loc});
        auto *bodyBlock = rewriter.createBlock(continuationBlock);

        rewriter.setInsertionPointToEnd(currentBlock);
        auto zero = rewriter.create<LLVM::ConstantOp>(loc, llvmIndexType, rewriter.getIntegerAttr(llvmIndexType, 0));
        rewriter.create<LLVM::BrOp>(loc, ValueRange{zero}, conditionBlock);

        rewriter.setInsertionPointToEnd(conditionBlock);
        auto index = conditionBlock->getArgument(0);
        auto keepGoing = rewriter.create<LLVM::ICmpOp>(loc, LLVM::ICmpPredicate::slt, index, count);
        rewriter.create<LLVM::CondBrOp>(loc, keepGoing, bodyBlock, continuationBlock);

        rewriter.setInsertionPointToEnd(bodyBlock);
        body(index);
        auto one = rewriter.create<LLVM::ConstantOp>(loc, llvmIndexType, rewriter.getIntegerAttr(llvmIndexType, 1));
        auto nextIndex = rewriter.create<LLVM::AddOp>(loc, llvmIndexType, index, one);
        rewriter.create<LLVM::BrOp>(loc, ValueRange{nextIndex}, conditionBlock);

        rewriter.setInsertionPointToStart(continuationBlock);
    }
};

} // namespace typescript

#endif // MLIR_TYPESCRIPT_LOWERTOLLVMLOGIC_OWNERSHIPROUTINELOGIC_H_
