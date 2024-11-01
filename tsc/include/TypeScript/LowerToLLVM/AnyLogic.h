#ifndef MLIR_TYPESCRIPT_LOWERTOLLVMLOGIC_ANYLOGIC_H_
#define MLIR_TYPESCRIPT_LOWERTOLLVMLOGIC_ANYLOGIC_H_

#include "TypeScript/Config.h"
#include "TypeScript/Defines.h"
#include "TypeScript/Passes.h"
#include "TypeScript/TypeScriptDialect.h"
#include "TypeScript/TypeScriptOps.h"

#include "TypeScript/LowerToLLVM/TypeHelper.h"
#include "TypeScript/LowerToLLVM/TypeConverterHelper.h"
#include "TypeScript/LowerToLLVM/LLVMTypeConverterHelper.h"
#include "TypeScript/LowerToLLVM/CodeLogicHelper.h"
#include "TypeScript/LowerToLLVM/LLVMCodeHelperBase.h"
#include "TypeScript/LowerToLLVM/TypeOfOpHelper.h"

using namespace mlir;
namespace mlir_ts = mlir::typescript;

namespace typescript
{

class AnyLogic
{
    Operation *op;
    PatternRewriter &rewriter;
    TypeConverterHelper &tch;
    TypeHelper th;
    LLVMCodeHelperBase ch;
    CodeLogicHelper clh;
    Location loc;

  protected:
    mlir::Type indexType;
    mlir::Type llvmIndexType;
    mlir::Type valuePtrType;

  public:
    AnyLogic(Operation *op, PatternRewriter &rewriter, TypeConverterHelper &tch, Location loc, CompileOptions &compileOptions)
        : op(op), rewriter(rewriter), tch(tch), th(rewriter), ch(op, rewriter, tch.typeConverter, compileOptions), clh(op, rewriter), loc(loc)
    {
        indexType = th.getIndexType();
        llvmIndexType = tch.convertType(indexType);
        valuePtrType = th.getPtrType();
    }

    LLVM::LLVMStructType getStorageAnyType(mlir::Type llvmStorageType)
    {
        return LLVM::LLVMStructType::getLiteral(rewriter.getContext(), {llvmIndexType, valuePtrType, llvmStorageType}, false);
    }

    mlir::Value castToAny(mlir::Value in, mlir::Type inType, mlir::Type inLLVMType)
    {
        // get typeof value
        // auto typeOfValue = rewriter.create<mlir_ts::TypeOfOp>(loc, mlir_ts::StringType::get(rewriter.getContext()), in);
        TypeOfOpHelper toh(rewriter);
        auto typeOfValue = toh.typeOfLogic(loc, in, inType);
        return castToAny(in, typeOfValue, inLLVMType);
    }

    mlir::Value castToAny(mlir::Value in, mlir::Value typeOfValue, mlir::Type inLLVMType)
    {
        // TODO: add type id to track data type
        auto ptrTy = th.getPtrType();
        auto llvmStorageType = inLLVMType;
        auto dataWithSizeType = getStorageAnyType(llvmStorageType);

        auto memValue = ch.MemoryAlloc(dataWithSizeType);

        // set value size
        auto sizeMLIR = rewriter.create<mlir_ts::SizeOfOp>(loc, indexType, llvmStorageType);
        auto size = rewriter.create<mlir_ts::DialectCastOp>(loc, llvmIndexType, sizeMLIR);

        auto zero = clh.createI32ConstantOf(0);
        auto one = clh.createI32ConstantOf(1);
        auto two = clh.createI32ConstantOf(2);

        auto ptrSize = rewriter.create<LLVM::GEPOp>(loc, ptrTy, dataWithSizeType, memValue, ValueRange{zero, zero});
        rewriter.create<LLVM::StoreOp>(loc, size, ptrSize);

        auto typeOfStr = rewriter.create<LLVM::GEPOp>(loc, ptrTy, dataWithSizeType, memValue, ValueRange{zero, one});
        rewriter.create<LLVM::StoreOp>(loc, typeOfValue, typeOfStr);

        // set actual value
        auto ptrValue = rewriter.create<LLVM::GEPOp>(loc, ptrTy, dataWithSizeType, memValue, ValueRange{zero, two});
        rewriter.create<LLVM::StoreOp>(loc, in, ptrValue);

        return memValue;
    }

    mlir::Value UnboxAny(mlir::Value in, mlir::Type resLLVMType)
    {
        // TODO: add type id to track data type
        // TODO: add data size check
        auto ptrTy = th.getPtrType();
        auto llvmStorageType = resLLVMType;
        auto dataWithSizeType = getStorageAnyType(llvmStorageType);

        auto inDataWithSizeTypedValue = rewriter.create<LLVM::BitcastOp>(loc, ptrTy, in);

        auto zero = clh.createI32ConstantOf(0);
        // auto one = clh.createI32ConstantOf(1);
        auto two = clh.createI32ConstantOf(2);

        // set actual value
        auto ptrValue =
            rewriter.create<LLVM::GEPOp>(loc, ptrTy, dataWithSizeType, inDataWithSizeTypedValue, ValueRange{zero, two});
        return rewriter.create<LLVM::LoadOp>(loc, resLLVMType, ptrValue);
    }

    mlir::Value getTypeOfAny(mlir::Value in)
    {
        // TODO: add type id to track data type
        // TODO: add data size check
        // any random type
        auto ptrTy = th.getPtrType();
        auto llvmStorageType = th.getI8Type();
        auto dataWithSizeType = getStorageAnyType(llvmStorageType);

        auto inDataWithSizeTypedValue = rewriter.create<LLVM::BitcastOp>(loc, ptrTy, in);

        auto zero = clh.createI32ConstantOf(0);
        auto one = clh.createI32ConstantOf(1);
        // auto two = clh.createI32ConstantOf(2);

        // set actual value
        auto ptrValue = rewriter.create<LLVM::GEPOp>(loc, ptrTy, dataWithSizeType, inDataWithSizeTypedValue,
                                                     ValueRange{zero, one});
        return rewriter.create<LLVM::LoadOp>(loc, valuePtrType, ptrValue);
    }
};
} // namespace typescript

#endif // MLIR_TYPESCRIPT_LOWERTOLLVMLOGIC_ANYLOGIC_H_
