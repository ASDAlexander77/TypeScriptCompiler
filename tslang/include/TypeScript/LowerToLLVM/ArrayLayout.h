#ifndef MLIR_TYPESCRIPT_LOWERTOLLVM_ARRAYLAYOUT_H_
#define MLIR_TYPESCRIPT_LOWERTOLLVM_ARRAYLAYOUT_H_

#include "TypeScript/Defines.h"
#include "TypeScript/TypeScriptOps.h"
#include "TypeScript/MLIRLogic/MLIRHelper.h"
#include "TypeScript/LowerToLLVM/LLVMCodeHelperBase.h"
#include "TypeScript/LowerToLLVM/CodeLogicHelper.h"

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"

namespace typescript
{

// The one place that knows how an array value is laid out
// (docs/superpowers/specs/2026-10-03-array-reference-design.md, section 3). An array value is one
// pointer to a heap header { data, length, capacity } that never moves; only `data` is
// reallocated. Copying an array copies the pointer, so a change of length through one name is
// seen through every other (#453). A `slot` is a pointer to where an array value is stored.
//
// A null header - zeroed memory: an element of an array grown by `length =`, a field with no
// initializer, `undefined` or `null` as an array - reads as an empty array, and an op that
// changes an array through its slot gives the slot a fresh empty header first (ruling R1).
class ArrayLayout : public LLVMCodeHelperBase
{
  public:
    ArrayLayout(mlir::Operation *op, PatternRewriter &rewriter, const TypeConverter *typeConverter,
                CompileOptions &compileOptions)
        : LLVMCodeHelperBase(op, rewriter, typeConverter, compileOptions)
    {
    }

    // { data, length, capacity }: what an array value points to
    LLVM::LLVMStructType headerType()
    {
        TypeHelper th(rewriter);
        auto llvmIndexType = typeConverter->convertType(th.getIndexType());
        return LLVM::LLVMStructType::getLiteral(rewriter.getContext(), {th.getPtrType(), llvmIndexType, llvmIndexType},
                                                false);
    }

    // the header an op that changes the array held in `slot` works on. A slot holding null - a
    // zeroed element or field, `undefined` as an array - gets an empty header first, so a change
    // made through the slot is kept (R1)
    mlir::Value headerForUpdate(mlir_ts::ArrayType arrayType, mlir::Value slot)
    {
        TypeHelper th(rewriter);
        CodeLogicHelper clh(op, rewriter);
        auto loc = op->getLoc();
        auto ptrType = th.getPtrType();
        mlir::Value header = rewriter.create<LLVM::LoadOp>(loc, ptrType, slot);
        auto isNull = rewriter.create<LLVM::ICmpOp>(loc, LLVM::ICmpPredicate::eq, header,
                                                    rewriter.create<LLVM::ZeroOp>(loc, ptrType));
        return clh.conditionalExpressionLowering(
            loc, ptrType, isNull,
            [&](OpBuilder &, Location) -> mlir::Value {
                auto fresh = MemoryAlloc(headerType(), MemoryAllocSet::Zero);
                if (compileOptions.isRefCounted())
                {
                    // the slot is the header's one reference, and nothing above lowering knows
                    // the header exists to take it: born at zero, the owner's release would
                    // underflow the count to immortal and leak it
                    auto llvmIndexType = typeConverter->convertType(th.getIndexType());
                    rewriter.create<LLVM::StoreOp>(
                        loc,
                        rewriter.create<LLVM::ConstantOp>(loc, llvmIndexType, rewriter.getIntegerAttr(llvmIndexType, 1)),
                        getBlockPtrFromPayloadPtr(loc, fresh, llvmIndexType));
                }

                rewriter.create<LLVM::StoreOp>(loc, fresh, slot);
                return fresh;
            },
            [&](OpBuilder &, Location) -> mlir::Value { return header; });
    }

    // the same, for a routine that only reads: the header held in `slot`, possibly null (never
    // materialised: R1)
    mlir::Value headerForRead(mlir_ts::ArrayType, mlir::Value slot)
    {
        TypeHelper th(rewriter);
        return rewriter.create<LLVM::LoadOp>(op->getLoc(), th.getPtrType(), slot);
    }

    // the field addresses of a non-null header
    mlir::Value dataAddress(mlir_ts::ArrayType arrayType, mlir::Value header)
    {
        return fieldAddress(arrayType, header, ARRAY_DATA_INDEX);
    }

    mlir::Value lengthAddress(mlir_ts::ArrayType arrayType, mlir::Value header)
    {
        return fieldAddress(arrayType, header, ARRAY_SIZE_INDEX);
    }

    mlir::Value capacityAddress(mlir_ts::ArrayType arrayType, mlir::Value header)
    {
        return fieldAddress(arrayType, header, ARRAY_CAPACITY_INDEX);
    }

    // of an array value; a null header reads as an empty array (R1)
    mlir::Value data(mlir_ts::ArrayType arrayType, mlir::Value array)
    {
        TypeHelper th(rewriter);
        return loadOrZero(array, th.getPtrType(), ARRAY_DATA_INDEX);
    }

    mlir::Value length(mlir_ts::ArrayType arrayType, mlir::Value array)
    {
        TypeHelper th(rewriter);
        return loadOrZero(array, typeConverter->convertType(th.getIndexType()), ARRAY_SIZE_INDEX);
    }

    // what `===` and truthiness compare: the header pointer
    mlir::Value identity(mlir_ts::ArrayType, mlir::Value array)
    {
        return array;
    }

    // a new array over `data` (a block just allocated, or null for an empty array) with `length`
    // elements
    mlir::Value make(mlir_ts::ArrayType arrayType, mlir::Value data, mlir::Value length)
    {
        auto loc = op->getLoc();
        if (compileOptions.isRefCounted() && !data.getDefiningOp<LLVM::ZeroOp>())
        {
            // under rc the header holds a counted reference to its data block, as every copy of
            // the array did before the switch: a string made over the block
            // (`<string><Opaque>Ref(buffer[0])`, the default library's convertNumber) takes a
            // reference of its own, and the block is freed by whichever of the two lets go last
            TypeHelper th(rewriter);
            auto llvmIndexType = typeConverter->convertType(th.getIndexType());
            rewriter.create<LLVM::StoreOp>(
                loc, rewriter.create<LLVM::ConstantOp>(loc, llvmIndexType, rewriter.getIntegerAttr(llvmIndexType, 1)),
                getBlockPtrFromPayloadPtr(loc, data, llvmIndexType));
        }

        auto header = MemoryAlloc(headerType());
        rewriter.create<LLVM::StoreOp>(loc, data, fieldAddress(arrayType, header, ARRAY_DATA_INDEX));
        rewriter.create<LLVM::StoreOp>(loc, length, fieldAddress(arrayType, header, ARRAY_SIZE_INDEX));
        rewriter.create<LLVM::StoreOp>(loc, length, fieldAddress(arrayType, header, ARRAY_CAPACITY_INDEX));
        return header;
    }

    // a constant array in global data: a header global "ah_<data global>" holding
    // { data, length, capacity = length }, preceded under rc and own by an immortal block word (as
    // getOrCreateGlobalArray does for the data). Nothing is allocated, so this works inside
    // another global's initializer. Returns the header's address.
    mlir::Value makeStatic(mlir_ts::ArrayType arrayType, StringRef dataGlobalName, int64_t dataOffsetBytes,
                           int64_t length)
    {
        auto loc = op->getLoc();
        auto parentModule = op->getParentOfType<mlir::ModuleOp>();
        TypeHelper th(rewriter);
        auto llvmIndexType = typeConverter->convertType(th.getIndexType());
        auto withWord = compileOptions.tracksOwnership();
        auto headerName = ("ah_" + dataGlobalName).str();

        auto global = parentModule.lookupSymbol<LLVM::GlobalOp>(headerName);
        if (!global)
        {
            OpBuilder::InsertionGuard guard(rewriter);
            rewriter.setInsertionPointToStart(parentModule.getBody());
            mlir::Type globalType = headerType();
            if (withWord)
            {
                globalType = LLVM::LLVMStructType::getLiteral(rewriter.getContext(), {llvmIndexType, headerType()},
                                                              /*isPacked=*/true);
            }

            global = rewriter.create<LLVM::GlobalOp>(loc, globalType, /*isConstant=*/true, LLVM::Linkage::Internal,
                                                     headerName, mlir::Attribute{});
            global.setAlignment(getHeapBlockHeaderSize());

            auto &region = global.getInitializerRegion();
            rewriter.setInsertionPointToStart(rewriter.createBlock(&region));
            auto llvmLength =
                rewriter.create<LLVM::ConstantOp>(loc, llvmIndexType, rewriter.getIntegerAttr(llvmIndexType, length));
            mlir::Value dataPtr = rewriter.create<LLVM::AddressOfOp>(loc, th.getPtrType(), dataGlobalName);
            if (dataOffsetBytes != 0)
            {
                dataPtr = rewriter.create<LLVM::GEPOp>(
                    loc, th.getPtrType(), th.getI8Type(), dataPtr,
                    ValueRange{rewriter.create<LLVM::ConstantOp>(
                        loc, llvmIndexType, rewriter.getIntegerAttr(llvmIndexType, dataOffsetBytes))});
            }

            mlir::Value headerVal = rewriter.create<LLVM::UndefOp>(loc, headerType());
            headerVal = rewriter.create<LLVM::InsertValueOp>(loc, headerVal, dataPtr,
                                                             MLIRHelper::getStructIndex(rewriter, ARRAY_DATA_INDEX));
            headerVal = rewriter.create<LLVM::InsertValueOp>(loc, headerVal, llvmLength,
                                                             MLIRHelper::getStructIndex(rewriter, ARRAY_SIZE_INDEX));
            headerVal = rewriter.create<LLVM::InsertValueOp>(loc, headerVal, llvmLength,
                                                             MLIRHelper::getStructIndex(rewriter, ARRAY_CAPACITY_INDEX));
            mlir::Value globalVal = headerVal;
            if (withWord)
            {
                globalVal = rewriter.create<LLVM::UndefOp>(loc, globalType);
                globalVal = rewriter.create<LLVM::InsertValueOp>(
                    loc, globalVal,
                    rewriter.create<LLVM::ConstantOp>(loc, llvmIndexType,
                                                      rewriter.getIntegerAttr(llvmIndexType, HEAP_BLOCK_IMMORTAL)),
                    MLIRHelper::getStructIndex(rewriter, 0));
                globalVal = rewriter.create<LLVM::InsertValueOp>(loc, globalVal, headerVal,
                                                                 MLIRHelper::getStructIndex(rewriter, 1));
            }

            rewriter.create<LLVM::ReturnOp>(loc, ValueRange{globalVal});
        }

        mlir::Value address = rewriter.create<LLVM::AddressOfOp>(loc, global);
        return withWord ? getPayloadPtrFromBlockPtr(loc, address, llvmIndexType) : address;
    }

    // no array at all: a null header, which reads as empty (R1)
    mlir::Value zero(mlir_ts::ArrayType)
    {
        TypeHelper th(rewriter);
        return rewriter.create<LLVM::ZeroOp>(op->getLoc(), th.getPtrType());
    }

    mlir::Value undef(mlir_ts::ArrayType arrayType)
    {
        return zero(arrayType);
    }

  private:
    mlir::Value fieldAddress(mlir_ts::ArrayType, mlir::Value header, int32_t index)
    {
        TypeHelper th(rewriter);
        return rewriter.create<LLVM::GEPOp>(op->getLoc(), th.getPtrType(), headerType(), header,
                                            ArrayRef<LLVM::GEPArg>{0, index});
    }

    mlir::Value loadOrZero(mlir::Value header, mlir::Type fieldType, int32_t index)
    {
        TypeHelper th(rewriter);
        CodeLogicHelper clh(op, rewriter);
        auto loc = op->getLoc();
        auto isSet = rewriter.create<LLVM::ICmpOp>(loc, LLVM::ICmpPredicate::ne, header,
                                                   rewriter.create<LLVM::ZeroOp>(loc, th.getPtrType()));
        return clh.conditionalExpressionLowering(
            loc, fieldType, isSet,
            [&](OpBuilder &, Location) -> mlir::Value {
                return rewriter.create<LLVM::LoadOp>(
                    loc, fieldType,
                    rewriter.create<LLVM::GEPOp>(loc, th.getPtrType(), headerType(), header,
                                                 ArrayRef<LLVM::GEPArg>{0, index}));
            },
            [&](OpBuilder &, Location) -> mlir::Value { return rewriter.create<LLVM::ZeroOp>(loc, fieldType); });
    }
};

} // namespace typescript

#endif // MLIR_TYPESCRIPT_LOWERTOLLVM_ARRAYLAYOUT_H_
