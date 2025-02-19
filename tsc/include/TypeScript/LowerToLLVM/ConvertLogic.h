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

  public:
    ConvertLogic(Operation *op, PatternRewriter &rewriter, TypeConverterHelper &tch, Location loc, CompileOptions &compileOptions)
        : op(op), rewriter(rewriter), tch(tch), th(rewriter), ch(op, rewriter, &tch.typeConverter, compileOptions), clh(op, rewriter), loc(loc)
    {
    }

    mlir::Value itoa(mlir::Value value)
    {
        auto i8PtrTy = th.getI8PtrType();

        auto _itoaFuncOp = ch.getOrInsertFunction(
            "_itoa", th.getFunctionType(th.getI8PtrType(),
                                        ArrayRef<mlir::Type>{rewriter.getI32Type(), th.getI8PtrType(), rewriter.getI32Type()}, true));

        auto bufferSizeValue = clh.createI32ConstantOf(50);
        // auto newStringValue = ch.Alloca(i8PtrTy, bufferSizeValue, true);
        auto newStringValue = ch.MemoryAllocBitcast(i8PtrTy, bufferSizeValue, MemoryAllocSet::Atomic);
        auto base = clh.createI32ConstantOf(10);

        return rewriter.create<LLVM::CallOp>(loc, _itoaFuncOp, ValueRange{value, newStringValue, base}).getResult();
    }

    mlir::Value i64toa(mlir::Value value)
    {
        auto i8PtrTy = th.getI8PtrType();

        auto _i64toaFuncOp = ch.getOrInsertFunction(
            "_i64toa", th.getFunctionType(th.getI8PtrType(),
                                          ArrayRef<mlir::Type>{rewriter.getI64Type(), th.getI8PtrType(), rewriter.getI32Type()}, true));

        auto bufferSizeValue = clh.createI32ConstantOf(50);
        // auto newStringValue = ch.Alloca(i8PtrTy, bufferSizeValue, true);
        auto newStringValue = ch.MemoryAllocBitcast(i8PtrTy, bufferSizeValue, MemoryAllocSet::Atomic);
        auto base = clh.createI32ConstantOf(10);

        return rewriter.create<LLVM::CallOp>(loc, _i64toaFuncOp, ValueRange{value, newStringValue, base}).getResult();
    }

    mlir::Value gcvt(mlir::Value in)
    {
        auto i8PtrTy = th.getI8PtrType();

        auto _gcvtFuncOp = ch.getOrInsertFunction(
            "_gcvt", th.getFunctionType(th.getI8PtrType(),
                                        ArrayRef<mlir::Type>{rewriter.getF64Type(), rewriter.getI32Type(), th.getI8PtrType()}, true));

        auto bufferSizeValue = clh.createI32ConstantOf(50);
        // auto newStringValue = ch.Alloca(i8PtrTy, bufferSizeValue, true);
        auto newStringValue = ch.MemoryAllocBitcast(i8PtrTy, bufferSizeValue, MemoryAllocSet::Atomic);
        auto doubleValue = rewriter.create<LLVM::FPExtOp>(loc, rewriter.getF64Type(), in);
        auto precision = clh.createI32ConstantOf(16);

        return rewriter.create<LLVM::CallOp>(loc, _gcvtFuncOp, ValueRange{doubleValue, precision, newStringValue}).getResult();
    }

    mlir::Value sprintf(int buffSize, std::string format, mlir::Value value)
    {
        auto i8PtrTy = th.getI8PtrType();

        auto llvmIndexType = tch.convertType(th.getIndexType());
        auto bufferSizeValue = clh.createIndexConstantOf(llvmIndexType, buffSize);

        auto opHash = std::hash<std::string>{}(format);

        std::stringstream formatVarName;
        formatVarName << "frmt_" << opHash;

        auto formatSpecifierCst = ch.getOrCreateGlobalString(formatVarName.str(), format);

        auto newVal = rewriter.create<mlir_ts::ConvertFOp>(loc, th.getStringType(), bufferSizeValue, formatSpecifierCst, ValueRange{value});
        return newVal;
    }

    mlir::Value sprintfOfF64(mlir::Value value)
    {
#ifdef NUMBER_F64
        auto doubleValue = value;
#else
        auto doubleValue = rewriter.create<LLVM::FPExtOp>(loc, rewriter.getF64Type(), value);
#endif
        return sprintf(50, "%g", doubleValue);
    }

    mlir::Value sprintfOfInt(mlir::Value valueIn, int width, bool isSigned)
    {
        mlir::Value value = valueIn;

        std::string frm = "%";

        if (isSigned)
        {
            frm += "-";
        }

        switch (width)
        {
            case 8: 
                frm += "hh";
                break;
            case 16: 
                frm += "h";
                break;
            case 64: 
                frm += "ll";
                break;
            default:
                break;
        }

        if (isSigned)
        {
            frm += "i";
        }
        else
        {
            frm += "u";
        }

        if (width < 32)
        {
            if (isSigned)
            {
                value = rewriter.create<LLVM::SExtOp>(loc, tch.convertType(rewriter.getIntegerType(32)), value);
            }
            else
            {
                value = rewriter.create<LLVM::ZExtOp>(loc, tch.convertType(rewriter.getIntegerType(32)), value);
            }
        }

        return sprintf(50, frm, value);
    }

    mlir::Value intToString(mlir::Value value, int width, bool isSigned)
    {
#ifndef USE_SPRINTF
        return itoa(value);
#else
        return sprintfOfInt(value, width, isSigned);
#endif
    }

    mlir::Value f64ToString(mlir::Value value)
    {
#ifndef USE_SPRINTF
        return gcvt(value);
#else
        return sprintfOfF64(value);
#endif
    }
};
} // namespace typescript

#endif // MLIR_TYPESCRIPT_LOWERTOLLVMLOGIC_CONVERTERLOGIC_H_
