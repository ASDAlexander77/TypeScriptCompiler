#ifndef MLIR_TYPESCRIPT_LOWERTOLLVMLOGIC_LLVMCODEHELPERWRAP_H_
#define MLIR_TYPESCRIPT_LOWERTOLLVMLOGIC_LLVMCODEHELPERWRAP_H_

#include "TypeScript/LowerToLLVM/TypeConverterHelper.h"
#include "TypeScript/LowerToLLVM/TypeHelper.h"

#include "mlir/Dialect/Arith/IR/Arith.h"

using namespace mlir;
namespace mlir_ts = mlir::typescript;

namespace typescript
{

enum class MemoryAllocSet
{
    None,
    Zero,
    Atomic
};

template <typename T>
mlir::Value castLogic(mlir::Value, mlir::Type, mlir::Operation *, PatternRewriter &, TypeConverterHelper, CompileOptions&);

class LLVMCodeHelperBase
{
  protected:
    mlir::Operation *op;
    PatternRewriter &rewriter;
    const TypeConverter *typeConverter;
    CompileOptions &compileOptions;

  public:
    LLVMCodeHelperBase(mlir::Operation *op, PatternRewriter &rewriter, const TypeConverter *typeConverter, CompileOptions &compileOptions)
        : op(op), rewriter(rewriter), typeConverter(typeConverter), compileOptions(compileOptions)
    {
    }

    template <typename T> 
    void seekLast(mlir::Block *block)
    {
        // find last string
        auto lastUse = [&](Operation *op) {
            if (auto globalOp = dyn_cast_or_null<LLVM::GlobalOp>(op))
            {
                if (globalOp.getValueAttr() && isa<T>(globalOp.getValueAttr()))
                {
                    rewriter.setInsertionPointAfter(globalOp);
                }
            }
        };

        block->walk(lastUse);
    }

    void seekLast(mlir::Block *block)
    {
        // find last string
        auto lastUse = [&](mlir::Operation *op) {
            if (auto globalOp = dyn_cast_or_null<LLVM::GlobalOp>(op))
            {
                rewriter.setInsertionPointAfter(globalOp);
            }
        };

        block->walk(lastUse);
    }

    void seekLastWithBody(mlir::Block *block)
    {
        // find last string
        auto lastUse = [&](Operation *op) {
            if (auto globalOp = dyn_cast_or_null<LLVM::GlobalOp>(op))
            {
                if (globalOp.getInitializerBlock())
                {
                    rewriter.setInsertionPointAfter(globalOp);
                }
            }
        };

        block->walk(lastUse);
    }

    template <typename T> void seekLastOp(mlir::Block *block)
    {
        // find last string
        auto lastUse = [&](Operation *op) {
            if (auto opT = dyn_cast_or_null<T>(op))
            {
                rewriter.setInsertionPointAfter(opT);
            }
        };

        block->walk(lastUse);
    }

    template <typename T> Operation *seekFirstNonConstantOp(T funcOp)
    {
        auto found = false;
        Operation *foundOp;
        // find last string
        auto lastUse = [&](Operation *op) {
            if (found)
            {
                return;
            }

            auto constantOp = dyn_cast_or_null<mlir_ts::ConstantOp>(op);
            if (!constantOp)
            {
                auto constOp = dyn_cast_or_null<mlir::arith::ConstantOp>(op);
                if (!constOp)
                {
                    found = true;
                    foundOp = op;
                }
            }
        };

        funcOp.walk(lastUse);

        return foundOp;
    }

    std::string getStorageStringName(std::string value)
    {
        auto opHash = std::hash<std::string>{}(value);

        std::stringstream strVarName;
        strVarName << "s_" << opHash;

        return strVarName.str();
    }

  private:
    /// Return a value representing an access into a global string with the given
    /// name, creating the string if necessary.
    mlir::Value getOrCreateGlobalString_(StringRef name, StringRef value)
    {
        auto loc = op->getLoc();
        auto parentModule = op->getParentOfType<ModuleOp>();

        TypeHelper th(rewriter);
        TypeConverterHelper tch(typeConverter);

        auto llvmIndexType = tch.convertType(th.getIndexType());

        // A static string is a block like any other: the same header word sits in front of
        // the characters, marked immortal, so a pointer to a string literal and a pointer to
        // a heap string are the same shape and a release can tell them apart. All-ones bytes
        // encode HEAP_BLOCK_IMMORTAL whatever the word size or endianness.
        auto headerSize = getHeapBlockHeaderSize();
        std::string blockBytes(headerSize, (char)0xFF);
        blockBytes.append(value.data(), value.size());

        // Create the global at the entry of the module.
        LLVM::GlobalOp global;
        if (!(global = parentModule.lookupSymbol<LLVM::GlobalOp>(name)))
        {
            OpBuilder::InsertionGuard insertGuard(rewriter);
            rewriter.setInsertionPointToStart(parentModule.getBody());

            seekLast<StringAttr>(parentModule.getBody());

            auto type = th.getArrayType(th.getI8Type(), blockBytes.size());
            global = rewriter.create<LLVM::GlobalOp>(loc, type, true, LLVM::Linkage::Internal, name, rewriter.getStringAttr(blockBytes));
            // the header is read as a whole word, so the block base has to be word-aligned
            global.setAlignment(headerSize);
        }

        // Get the pointer to the first character in the global string - past the header.
        mlir::Value globalPtr = rewriter.create<LLVM::AddressOfOp>(loc, global);
        return rewriter.create<LLVM::GEPOp>(loc, th.getPtrType(), global.getType(), globalPtr,
                                            ArrayRef<LLVM::GEPArg>{0, (int32_t)headerSize});
    }

  public:
    mlir::Value getOrCreateGlobalString(std::string value)
    {
        return getOrCreateGlobalString(getStorageStringName(value), value);
    }

    mlir::Value getOrCreateGlobalString(StringRef name, std::string value)
    {
        return getOrCreateGlobalString_(name, StringRef(value.data(), value.length() + 1));
    }

    LLVM::LLVMFuncOp getOrInsertFunction(mlir::Location loc, ModuleOp parentModule, const StringRef &name, const LLVM::LLVMFunctionType &llvmFnType)
    {
        if (auto funcOp = parentModule.lookupSymbol<LLVM::LLVMFuncOp>(name))
        {
            return funcOp;
        }

        PatternRewriter::InsertionGuard insertGuard(rewriter);
        rewriter.setInsertionPointToStart(parentModule.getBody());
        return rewriter.create<LLVM::LLVMFuncOp>(loc, name, llvmFnType);
    }    

    LLVM::LLVMFuncOp getOrInsertFunction(const StringRef &name, const LLVM::LLVMFunctionType &llvmFnType)
    {
        return getOrInsertFunction(mlir::UnknownLoc::get(op->getContext()) /*op->getLoc()*/, op->getParentOfType<ModuleOp>(), name, llvmFnType);
    }

    mlir::Value MemoryAlloc(mlir::Value sizeOfAlloc, MemoryAllocSet zero = MemoryAllocSet::None)
    {
        return _MemoryAlloc<int>(sizeOfAlloc, zero);
    }

    mlir::Value MemoryAlloc(mlir::Type storageType, MemoryAllocSet zero = MemoryAllocSet::None)
    {
        TypeHelper th(rewriter);
        TypeConverterHelper tch(typeConverter);

        auto llvmIndexType = tch.convertType(th.getIndexType());

        auto loc = op->getLoc();

        auto sizeOfTypeValueMLIR = rewriter.create<mlir_ts::SizeOfOp>(loc, th.getIndexType(), storageType);
        auto sizeOfTypeValue = rewriter.create<mlir_ts::DialectCastOp>(loc, llvmIndexType, sizeOfTypeValueMLIR);
        return MemoryAlloc(sizeOfTypeValue, zero);
    }

    mlir::Value MemoryRealloc(mlir::Value ptrValue, mlir::Value sizeOfAlloc)
    {
        return _MemoryRealloc<int>(ptrValue, sizeOfAlloc);
    }

    LogicalResult MemoryFree(mlir::Value ptrValue)
    {
        return _MemoryFree<int>(ptrValue);
    }

    mlir::Value Alloca(mlir::Type elementType, int count, bool inalloca = false)
    {
        auto location = op->getLoc();

        mlir::OpBuilder::InsertionGuard insertGuard(rewriter);

        // put all allocs at 'func' top
        auto parentFuncOp = op->getParentOfType<LLVM::LLVMFuncOp>();
        if (parentFuncOp)
        {
            // if inside function (not in global op)
            rewriter.setInsertionPoint(&parentFuncOp.getBody().front().front());
        }

        CodeLogicHelper clh(op, rewriter);
        TypeHelper th(rewriter);
        auto allocated = rewriter.create<LLVM::AllocaOp>(location, th.getPtrType(), elementType, clh.createI32ConstantOf(count), inalloca);
        return allocated;
    }

    mlir::Value Alloca(mlir::Type elementType, mlir::Value count, bool inalloca = false)
    {
        auto location = op->getLoc();
        CodeLogicHelper clh(op, rewriter);
        TypeHelper th(rewriter);
        auto allocated = rewriter.create<LLVM::AllocaOp>(location, th.getPtrType(), elementType, count, inalloca);
        return allocated;
    }

    // === Heap block header (memory-model groundwork) ===
    //
    // Every heap block allocated through _MemoryAlloc reserves a leading pointer-sized word,
    // and the pointer handed back to the rest of the compiler addresses the payload just past
    // it. Under GC that word is never read; under `-mm=rc` it is the reference count. Keeping
    // the layout identical in both memory models is what makes a GC-built module and an
    // RC-built module safe to link together; see docs/reference-counting-evaluation.md,
    // sections 4 and 9.1.
    //
    // This covers every heap block that the compiler still emits, class instances included.
    // The one path it would not have covered - GC_malloc_explicitly_typed, whose Boehm type
    // descriptor indexes bits relative to the object base and so could not tolerate the base
    // moving - sits behind ENABLE_TYPED_GC and was retired in §9.2, before the header existed.
    unsigned getHeapBlockHeaderSize()
    {
        return compileOptions.sizeBits / 8;
    }

    mlir::Value createHeapBlockHeaderSizeConstant(mlir::Location loc, mlir::Type llvmIndexType, bool negated = false)
    {
        auto bytes = static_cast<int64_t>(getHeapBlockHeaderSize());
        return rewriter.create<LLVM::ConstantOp>(loc, llvmIndexType,
                                                 rewriter.getIntegerAttr(llvmIndexType, negated ? -bytes : bytes));
    }

    // payload = block + headerSize
    mlir::Value getPayloadPtrFromBlockPtr(mlir::Location loc, mlir::Value blockPtr, mlir::Type llvmIndexType)
    {
        TypeHelper th(rewriter);
        auto offset = createHeapBlockHeaderSizeConstant(loc, llvmIndexType);
        return rewriter.create<LLVM::GEPOp>(loc, th.getPtrType(), th.getI8Type(), blockPtr, ValueRange{offset});
    }

    // block = payload - headerSize
    mlir::Value getBlockPtrFromPayloadPtr(mlir::Location loc, mlir::Value payloadPtr, mlir::Type llvmIndexType)
    {
        TypeHelper th(rewriter);
        auto offset = createHeapBlockHeaderSizeConstant(loc, llvmIndexType, /*negated=*/true);
        return rewriter.create<LLVM::GEPOp>(loc, th.getPtrType(), th.getI8Type(), payloadPtr, ValueRange{offset});
    }

    template <typename T> mlir::Value _MemoryAlloc(mlir::Value sizeOfAlloc, MemoryAllocSet memAllocMode)
    {
        TypeHelper th(rewriter);
        TypeConverterHelper tch(typeConverter);
        CodeLogicHelper clh(op, rewriter);

        auto llvmIndexType = tch.convertType(th.getIndexType());

        auto loc = op->getLoc();

        auto i8PtrTy = th.getPtrType();
        auto mallocFuncOp = getOrInsertFunction(
            compileOptions.isWasm ? "ts_malloc" : "malloc", 
            th.getFunctionType(i8PtrTy, {llvmIndexType}));

        auto effectiveSize = sizeOfAlloc;

        if (effectiveSize.getType() != th.getIndexType() && effectiveSize.getType() != llvmIndexType)
        {
            effectiveSize = castLogic<int>(effectiveSize, th.getIndexType(), op, rewriter, tch, compileOptions);
        }

        if (effectiveSize.getType() == th.getIndexType())
        {
            effectiveSize = rewriter.create<mlir_ts::DialectCastOp>(loc, llvmIndexType, effectiveSize);
        }

        // reserve the block header in front of the payload
        auto headerSizeValue = createHeapBlockHeaderSizeConstant(loc, llvmIndexType);
        mlir::Value paddedSize = rewriter.create<LLVM::AddOp>(loc, llvmIndexType, ValueRange{effectiveSize, headerSizeValue});

        auto callResults = rewriter.create<LLVM::CallOp>(loc, mallocFuncOp, ValueRange{paddedSize});
        if (memAllocMode == MemoryAllocSet::Atomic)
        {
            callResults->setAttr("mode", rewriter.getStringAttr("atomic"));
        }

        auto blockPtr = callResults.getResult();

        if (memAllocMode == MemoryAllocSet::Zero)
        {
            // NOTE: zero the whole block, header included, rather than just the payload. That keeps
            // this memset's first operand the raw allocation call itself, which is what GCPass's
            // removeRedundantMemSet matches on in order to drop it when GC_malloc already zeroed.
            // TODO: replace with @llvm.memset.p0.i64 & @llvm.memset.p0.i32
            auto memsetFuncOp = getOrInsertFunction("memset", th.getFunctionType(i8PtrTy, {i8PtrTy, th.getI32Type(), llvmIndexType}));
            auto const0 = clh.createI32ConstantOf(0);
            rewriter.create<LLVM::CallOp>(loc, memsetFuncOp, ValueRange{blockPtr, const0, paddedSize});
        }

        if (compileOptions.isRefCounted())
        {
            // The block starts *unowned*. Whoever first takes it - a local's declaration, a
            // field or element store, a literal capturing it, a push - is what brings the count
            // to one, and that owner's release is what takes it back to zero and frees it.
            //
            // It was born at one until §9.24: the reference an allocation came with was never
            // consumed, so every count sat one above the truth and nothing was ever freed. That
            // was deliberate while the insertion points were being built one at a time, because
            // it made a missing retain an inert leak rather than a premature free. All of them
            // are in place now (§9.19 through §9.23), so the slack comes out here.
            //
            // Being born at zero also gives the remaining mistakes a benign shape at the
            // boundary: a release of a block nobody ever took underflows to all-ones, which is
            // HEAP_BLOCK_IMMORTAL, so the block leaks instead of being freed out from under a
            // live reference.
            //
            // Written after any memset above, which zeroes the header along with the payload.
            // Only under `-mm=rc` -- under `gc` nothing reads the word, and a store per
            // allocation on the hot path is not worth paying for dead code.
            rewriter.create<LLVM::StoreOp>(
                loc, rewriter.create<LLVM::ConstantOp>(loc, llvmIndexType, rewriter.getIntegerAttr(llvmIndexType, 0)),
                blockPtr);
        }

        return getPayloadPtrFromBlockPtr(loc, blockPtr, llvmIndexType);
    }

    template <typename T> mlir::Value _MemoryRealloc(mlir::Value ptrValue, mlir::Value sizeOfAlloc)
    {
        TypeHelper th(rewriter);
        TypeConverterHelper tch(typeConverter);

        auto loc = op->getLoc();

        auto i8PtrTy = th.getPtrType();
        assert (ptrValue.getType() == i8PtrTy);

        auto llvmIndexType = tch.convertType(th.getIndexType());

        auto mallocFuncOp = getOrInsertFunction(
            compileOptions.isWasm ? "ts_realloc" : "realloc", 
            th.getFunctionType(i8PtrTy, {i8PtrTy, llvmIndexType}));

        auto effectiveSize = sizeOfAlloc;
        if (effectiveSize.getType() != th.getIndexType() && effectiveSize.getType() != llvmIndexType)
        {
            effectiveSize = castLogic<int>(effectiveSize, th.getIndexType(), op, rewriter, tch, compileOptions);
        }

        if (effectiveSize.getType() == th.getIndexType())
        {
            effectiveSize = rewriter.create<mlir_ts::DialectCastOp>(loc, llvmIndexType, effectiveSize);
        }

        // the incoming pointer addresses the payload; realloc must see the block base, and the
        // block must stay large enough for the header it carries
        auto headerSizeValue = createHeapBlockHeaderSizeConstant(loc, llvmIndexType);
        mlir::Value paddedSize = rewriter.create<LLVM::AddOp>(loc, llvmIndexType, ValueRange{effectiveSize, headerSizeValue});
        auto blockPtrValue = getBlockPtrFromPayloadPtr(loc, ptrValue, llvmIndexType);

        auto callResults = rewriter.create<LLVM::CallOp>(loc, mallocFuncOp, ValueRange{blockPtrValue, paddedSize});
        return getPayloadPtrFromBlockPtr(loc, callResults.getResult(), llvmIndexType);
    }

    template <typename T> mlir::LogicalResult _MemoryFree(mlir::Value ptrValue)
    {
        TypeHelper th(rewriter);
        TypeConverterHelper tch(typeConverter);

        auto loc = op->getLoc();

        auto i8PtrTy = th.getPtrType();

        auto freeFuncOp = getOrInsertFunction(
            compileOptions.isWasm ? "ts_free" : "free", 
            th.getFunctionType(th.getVoidType(), {i8PtrTy}));

        auto casted = rewriter.create<LLVM::BitcastOp>(loc, i8PtrTy, ptrValue);

        // the incoming pointer addresses the payload; free must see the block base
        auto llvmIndexType = tch.convertType(th.getIndexType());
        auto blockPtrValue = getBlockPtrFromPayloadPtr(loc, casted, llvmIndexType);

        rewriter.create<LLVM::CallOp>(loc, freeFuncOp, ValueRange{blockPtrValue});

        return mlir::success();
    }
};

} // namespace typescript

#endif // MLIR_TYPESCRIPT_LOWERTOLLVMLOGIC_LLVMCODEHELPERWRAP_H_
