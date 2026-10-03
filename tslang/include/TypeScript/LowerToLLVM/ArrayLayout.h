#ifndef MLIR_TYPESCRIPT_LOWERTOLLVM_ARRAYLAYOUT_H_
#define MLIR_TYPESCRIPT_LOWERTOLLVM_ARRAYLAYOUT_H_

#include "TypeScript/Defines.h"
#include "TypeScript/TypeScriptOps.h"
#include "TypeScript/MLIRLogic/MLIRHelper.h"
#include "TypeScript/LowerToLLVM/LLVMCodeHelperBase.h"

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"

namespace typescript
{

// The one place that knows how an array value is laid out
// (docs/superpowers/specs/2026-10-03-array-reference-design.md). An array value is a
// { data, length } struct; a `slot` is a pointer to one.
class ArrayLayout : public LLVMCodeHelperBase
{
  public:
    ArrayLayout(mlir::Operation *op, PatternRewriter &rewriter, const TypeConverter *typeConverter,
                CompileOptions &compileOptions)
        : LLVMCodeHelperBase(op, rewriter, typeConverter, compileOptions)
    {
    }

    // what the field GEPs are based on, for an op that changes the array held in `slot`
    mlir::Value headerForUpdate(mlir_ts::ArrayType, mlir::Value slot)
    {
        return slot;
    }

    // the same, for a routine that only reads (never materialises: R1)
    mlir::Value headerForRead(mlir_ts::ArrayType, mlir::Value slot)
    {
        return slot;
    }

    mlir::Value dataAddress(mlir_ts::ArrayType arrayType, mlir::Value header)
    {
        return fieldAddress(arrayType, header, ARRAY_DATA_INDEX);
    }

    mlir::Value lengthAddress(mlir_ts::ArrayType arrayType, mlir::Value header)
    {
        return fieldAddress(arrayType, header, ARRAY_SIZE_INDEX);
    }

    // of an array value
    mlir::Value data(mlir_ts::ArrayType, mlir::Value array)
    {
        TypeHelper th(rewriter);
        return rewriter.create<LLVM::ExtractValueOp>(op->getLoc(), th.getPtrType(), array,
                                                     MLIRHelper::getStructIndex(rewriter, ARRAY_DATA_INDEX));
    }

    mlir::Value length(mlir_ts::ArrayType, mlir::Value array)
    {
        TypeHelper th(rewriter);
        return rewriter.create<LLVM::ExtractValueOp>(op->getLoc(), typeConverter->convertType(th.getIndexType()), array,
                                                     MLIRHelper::getStructIndex(rewriter, ARRAY_SIZE_INDEX));
    }

    // what `===` and truthiness compare
    mlir::Value identity(mlir_ts::ArrayType arrayType, mlir::Value array)
    {
        return data(arrayType, array);
    }

    // a new array over `data` (owned by the array) with `length` elements
    mlir::Value make(mlir_ts::ArrayType arrayType, mlir::Value data, mlir::Value length)
    {
        auto loc = op->getLoc();
        auto llvmArrayType = typeConverter->convertType(arrayType);
        mlir::Value value = rewriter.create<LLVM::UndefOp>(loc, llvmArrayType);
        value = rewriter.create<LLVM::InsertValueOp>(loc, llvmArrayType, value, data,
                                                     MLIRHelper::getStructIndex(rewriter, ARRAY_DATA_INDEX));
        return rewriter.create<LLVM::InsertValueOp>(loc, llvmArrayType, value, length,
                                                    MLIRHelper::getStructIndex(rewriter, ARRAY_SIZE_INDEX));
    }

    // a constant array inside a global's initializer: `data` is static, nothing is allocated
    // (Task 4 replaces this with makeStatic(arrayType, dataGlobalName, dataOffsetBytes, length))
    mlir::Value makeStatic(mlir_ts::ArrayType arrayType, mlir::Value data, mlir::Value length)
    {
        return make(arrayType, data, length);
    }

    mlir::Value zero(mlir_ts::ArrayType arrayType)
    {
        return rewriter.create<LLVM::ZeroOp>(op->getLoc(), typeConverter->convertType(arrayType));
    }

    mlir::Value undef(mlir_ts::ArrayType arrayType)
    {
        return rewriter.create<LLVM::UndefOp>(op->getLoc(), typeConverter->convertType(arrayType));
    }

  private:
    mlir::Value fieldAddress(mlir_ts::ArrayType arrayType, mlir::Value header, int32_t index)
    {
        TypeHelper th(rewriter);
        return rewriter.create<LLVM::GEPOp>(op->getLoc(), th.getPtrType(), typeConverter->convertType(arrayType), header,
                                            ArrayRef<LLVM::GEPArg>{0, index});
    }
};

} // namespace typescript

#endif // MLIR_TYPESCRIPT_LOWERTOLLVM_ARRAYLAYOUT_H_
