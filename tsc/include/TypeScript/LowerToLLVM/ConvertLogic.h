#ifndef MLIR_TYPESCRIPT_LOWERTOLLVMLOGIC_CONVERTERLOGIC_H_
#define MLIR_TYPESCRIPT_LOWERTOLLVMLOGIC_CONVERTERLOGIC_H_

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

using namespace mlir;
namespace mlir_ts = mlir::typescript;

namespace typescript
{

class ConvertLogic
{
    Operation *op;
    PatternRewriter &rewriter;
    TypeConverterHelper &tch;
    TypeHelper th;
    LLVMCodeHelperBase ch;
    CodeLogicHelper clh;
    Location loc;

  protected:
    mlir::Type sizeType;
    mlir::Type typeOfValueType;

  public:
    ConvertLogic(Operation *op, PatternRewriter &rewriter, TypeConverterHelper &tch, Location loc)
        : op(op), rewriter(rewriter), tch(tch), th(rewriter), ch(op, rewriter, &tch.typeConverter), clh(op, rewriter), loc(loc)
    {
        sizeType = th.getIndexType();
        typeOfValueType = th.getI8PtrType();
    }

    mlir::Value itoa(mlir::Value value)
    {
        auto i8PtrTy = th.getI8PtrType();

        auto _itoaFuncOp = ch.getOrInsertFunction(
            "_itoa", th.getFunctionType(th.getI8PtrType(),
                                        ArrayRef<mlir::Type>{rewriter.getI32Type(), th.getI8PtrType(), rewriter.getI32Type()}, true));

        auto bufferSizeValue = clh.createI32ConstantOf(50);
        // auto newStringValue = rewriter.create<LLVM::AllocaOp>(loc, i8PtrTy, bufferSizeValue, true);
        auto newStringValue = ch.MemoryAllocBitcast(i8PtrTy, bufferSizeValue);
        auto base = clh.createI32ConstantOf(10);

        return rewriter.create<LLVM::CallOp>(loc, _itoaFuncOp, ValueRange{value, newStringValue, base}).getResult(0);
    }

    mlir::Value i64toa(mlir::Value value)
    {
        auto i8PtrTy = th.getI8PtrType();

        auto _i64toaFuncOp = ch.getOrInsertFunction(
            "_i64toa", th.getFunctionType(th.getI8PtrType(),
                                          ArrayRef<mlir::Type>{rewriter.getI64Type(), th.getI8PtrType(), rewriter.getI32Type()}, true));

        auto bufferSizeValue = clh.createI32ConstantOf(50);
        // auto newStringValue = rewriter.create<LLVM::AllocaOp>(loc, i8PtrTy, bufferSizeValue, true);
        auto newStringValue = ch.MemoryAllocBitcast(i8PtrTy, bufferSizeValue);
        auto base = clh.createI32ConstantOf(10);

        return rewriter.create<LLVM::CallOp>(loc, _i64toaFuncOp, ValueRange{value, newStringValue, base}).getResult(0);
    }

    mlir::Value gcvt(mlir::Value in)
    {
        auto i8PtrTy = th.getI8PtrType();

        auto _gcvtFuncOp = ch.getOrInsertFunction(
            "_gcvt", th.getFunctionType(th.getI8PtrType(),
                                        ArrayRef<mlir::Type>{rewriter.getF64Type(), rewriter.getI32Type(), th.getI8PtrType()}, true));

        auto bufferSizeValue = clh.createI32ConstantOf(50);
        // auto newStringValue = rewriter.create<LLVM::AllocaOp>(loc, i8PtrTy, bufferSizeValue, true);
        auto newStringValue = ch.MemoryAllocBitcast(i8PtrTy, bufferSizeValue);
        auto doubleValue = rewriter.create<LLVM::FPExtOp>(loc, rewriter.getF64Type(), in);
        auto precision = clh.createI32ConstantOf(16);

        return rewriter.create<LLVM::CallOp>(loc, _gcvtFuncOp, ValueRange{doubleValue, precision, newStringValue}).getResult(0);
    }

    mlir::Value sprintf(int buffSize, std::string format, mlir::Value value)
    {
        auto i8PtrTy = th.getI8PtrType();

#ifdef WIN32
        auto sprintfFuncOp = ch.getOrInsertFunction(
            "sprintf_s", th.getFunctionType(rewriter.getI32Type(), {th.getI8PtrType(), rewriter.getI32Type(), th.getI8PtrType()}, true));
#else
        auto sprintfFuncOp = ch.getOrInsertFunction(
            "snprintf", th.getFunctionType(rewriter.getI32Type(), {th.getI8PtrType(), rewriter.getI32Type(), th.getI8PtrType()}, true));
#endif

        auto bufferSizeValue = clh.createI32ConstantOf(buffSize);
        // auto newStringValue = rewriter.create<LLVM::AllocaOp>(loc, i8PtrTy, bufferSizeValue, true);
        auto newStringValue = ch.MemoryAllocBitcast(i8PtrTy, bufferSizeValue);

        auto opHash = std::hash<std::string>{}(format);

        std::stringstream formatVarName;
        formatVarName << "frmt_" << opHash;

        auto formatSpecifierCst = ch.getOrCreateGlobalString(formatVarName.str(), format);

        rewriter.create<LLVM::CallOp>(loc, sprintfFuncOp, ValueRange{newStringValue, bufferSizeValue, formatSpecifierCst, value});

        return newStringValue;
    }

    mlir::Value sprintfOfInt(mlir::Value value)
    {
        return sprintf(50, "%d", value);
    }

    mlir::Value sprintfOfF32orF64(mlir::Value value)
    {
        auto doubleValue = rewriter.create<LLVM::FPExtOp>(loc, rewriter.getF64Type(), value);
        return sprintf(50, "%g", doubleValue);
    }

    mlir::Value sprintfOfI64(mlir::Value value)
    {
        return sprintf(50, "%llu", value);
    }

    mlir::Value intToString(mlir::Value value)
    {
#ifndef USE_SPRINTF
        return itoa(value);
#else
        return sprintfOfInt(value);
#endif
    }

    mlir::Value int64ToString(mlir::Value value)
    {
#ifndef USE_SPRINTF
        return i64toa(value);
#else
        return sprintfOfI64(value);
#endif
    }

    mlir::Value f32OrF64ToString(mlir::Value value)
    {
#ifndef USE_SPRINTF
        return gcvt(value);
#else
        return sprintfOfF32orF64(value);
#endif
    }
};
} // namespace typescript

#endif // MLIR_TYPESCRIPT_LOWERTOLLVMLOGIC_CONVERTERLOGIC_H_
