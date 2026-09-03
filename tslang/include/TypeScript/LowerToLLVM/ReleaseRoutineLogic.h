#ifndef MLIR_TYPESCRIPT_LOWERTOLLVMLOGIC_RELEASEROUTINELOGIC_H_
#define MLIR_TYPESCRIPT_LOWERTOLLVMLOGIC_RELEASEROUTINELOGIC_H_

#include "TypeScript/Config.h"
#include "TypeScript/Defines.h"
#include "TypeScript/TypeScriptOps.h"

#include "TypeScript/MLIRLogic/MLIRTypeHelper.h"

#include "TypeScript/LowerToLLVM/TypeHelper.h"
#include "TypeScript/LowerToLLVM/TypeConverterHelper.h"
#include "TypeScript/LowerToLLVM/LLVMCodeHelperBase.h"
#include "TypeScript/LowerToLLVM/TypeDescriptorLogic.h"

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"

using namespace mlir;
namespace mlir_ts = mlir::typescript;

namespace typescript
{

// Generates, once per type, the routine that releases everything a value of that type owns:
// the heap blocks it is the sole owner of, and, recursively, whatever its fields own. The
// routine's address goes in the type's descriptor (TYPE_DESCR_RELEASE), which is the only
// thing that references it -- nothing calls these yet. See
// docs/reference-counting-evaluation.md section 9.4.
//
// Calling convention: the routine takes a pointer to the *storage holding* a value of the
// type, not the value. That is uniform across value categories - a class field, an "any"
// payload slot and a local variable are all addressed the same way - and it is what lets a
// field's release be a plain call with a GEP.
class ReleaseRoutineLogic
{
    Operation *op;
    PatternRewriter &rewriter;
    const TypeConverter *typeConverter;
    CompileOptions &compileOptions;

  public:
    ReleaseRoutineLogic(Operation *op, PatternRewriter &rewriter, const TypeConverter *typeConverter,
                        CompileOptions &compileOptions)
        : op(op), rewriter(rewriter), typeConverter(typeConverter), compileOptions(compileOptions)
    {
    }

    // Symbol name of the routine for `type`, generating it if needed. Empty when the type
    // owns no heap memory, in which case the descriptor's release slot stays null - a null
    // slot means "nothing to release", not "unknown".
    std::string getOrCreateReleaseRoutine(mlir::Type type)
    {
        if (!needsRelease(type))
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

        return name;
    }

    // Does a value of this type own heap memory, directly or through its fields?
    bool needsRelease(mlir::Type type)
    {
        llvm::SmallPtrSet<mlir::Type, 8> visiting;
        return needsRelease(type, visiting);
    }

  private:
    bool needsRelease(mlir::Type type, llvm::SmallPtrSetImpl<mlir::Type> &visiting)
    {
        if (!visiting.insert(type).second)
        {
            return false;
        }

        // owns its own block
        if (isa<mlir_ts::StringType>(type) || isa<mlir_ts::ArrayType>(type) || isa<mlir_ts::ClassType>(type) ||
            isa<mlir_ts::ObjectType>(type) || isa<mlir_ts::AnyType>(type))
        {
            return true;
        }

        if (auto unionType = dyn_cast<mlir_ts::UnionType>(type))
        {
            MLIRTypeHelper mth(rewriter.getContext(), compileOptions);
            mlir::Type baseType;
            if (mth.isUnionTypeNeedsTag(op->getLoc(), unionType, baseType))
            {
                // which member it holds is only known at run time, so the tag's descriptor
                // decides - assume it may own something
                return true;
            }

            return needsRelease(baseType, visiting);
        }

        if (auto optionalType = dyn_cast<mlir_ts::OptionalType>(type))
        {
            return needsRelease(optionalType.getElementType(), visiting);
        }

        for (auto fieldType : getFieldTypes(type))
        {
            if (needsRelease(fieldType, visiting))
            {
                return true;
            }
        }

        // Deliberately not released, each for its own reason:
        //  - InterfaceType carries only a name, so the concrete layout behind its `this`
        //    pointer is not recoverable from the type. Needs an RTTI lookup, not a static
        //    walk.
        //  - Function/BoundFunction/HybridFunction: the capture box is heap-allocated
        //    (ALLOC_CAPTURE_IN_HEAP) but its type does not appear in the function type, so
        //    there is nothing here to walk.
        //  - RefType/ValueRefType point at storage this value does not own.
        //  - ConstArrayType and ConstTupleType are static data.
        return false;
    }

    // Field types of a record-shaped type, empty for anything else.
    llvm::SmallVector<mlir::Type> getFieldTypes(mlir::Type type)
    {
        llvm::SmallVector<mlir::Type> result;

        auto addFields = [&](auto fields) {
            for (auto &field : fields)
            {
                result.push_back(field.type);
            }
        };

        if (auto tupleType = dyn_cast<mlir_ts::TupleType>(type))
        {
            addFields(tupleType.getFields());
        }
        else if (auto classStorageType = dyn_cast<mlir_ts::ClassStorageType>(type))
        {
            addFields(classStorageType.getFields());
        }
        else if (auto objectStorageType = dyn_cast<mlir_ts::ObjectStorageType>(type))
        {
            addFields(objectStorageType.getFields());
        }

        return result;
    }

    std::string getRoutineName(mlir::Type type)
    {
        std::stringstream ss;
        ss << "tsrel_" << (size_t)hash_value(type);
        return ss.str();
    }

    // free(payload - headerSize), unless the block says it is immortal.
    //
    // The check is not decoration: a string literal compiles to `store ptr @s_..., ...`, so a
    // `string` field can hold a pointer into a read-only global that no allocator produced.
    // Static blocks carry the header too, marked HEAP_BLOCK_IMMORTAL, which is what lets this
    // tell them apart.
    //
    // Routed through one generated helper so the other condition a release will grow -- a
    // reference count reaching zero -- has a single place to land.
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

            TypeConverterHelper tch(typeConverter);
            LLVMCodeHelperBase ch(op, rewriter, typeConverter, compileOptions);

            auto llvmIndexType = tch.convertType(th.getIndexType());
            auto payloadPtr = entryBlock->getArgument(0);
            auto blockPtr = ch.getBlockPtrFromPayloadPtr(loc, payloadPtr, llvmIndexType);
            auto headerWord = rewriter.create<LLVM::LoadOp>(loc, llvmIndexType, blockPtr);
            auto immortal = rewriter.create<LLVM::ConstantOp>(
                loc, llvmIndexType, rewriter.getIntegerAttr(llvmIndexType, HEAP_BLOCK_IMMORTAL));
            auto isMortal = rewriter.create<LLVM::ICmpOp>(loc, LLVM::ICmpPredicate::ne, headerWord, immortal);

            auto *freeBlock = rewriter.createBlock(&helper.getBody(), helper.getBody().end());
            auto *returnBlock = rewriter.createBlock(&helper.getBody(), helper.getBody().end());

            rewriter.setInsertionPointToEnd(entryBlock);

            rewriter.create<LLVM::CondBrOp>(loc, isMortal, freeBlock, returnBlock);

            rewriter.setInsertionPointToStart(freeBlock);
            ch.MemoryFree(payloadPtr);
            rewriter.create<LLVM::BrOp>(loc, ValueRange{}, returnBlock);

            rewriter.setInsertionPointToStart(returnBlock);
            rewriter.create<LLVM::ReturnOp>(loc, ValueRange{});
        }

        rewriter.create<LLVM::CallOp>(loc, TypeRange{}, FlatSymbolRefAttr::get(rewriter.getContext(), helperName),
                                      ValueRange{payloadPtr});
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
            auto routineName = getOrCreateReleaseRoutine(fieldType);
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
            emitIfNonNull(strValue, [&]() { emitFreeBlock(strValue); });
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
            emitIfNonNull(instanceValue, [&]() {
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
            emitIfNonNull(boxValue, [&]() {
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

                releaseViaDescriptor(tagValue, valueSlot);
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

        emitIfNonNull(dataValue, [&]() {
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

#endif // MLIR_TYPESCRIPT_LOWERTOLLVMLOGIC_RELEASEROUTINELOGIC_H_
