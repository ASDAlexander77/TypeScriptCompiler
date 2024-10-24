#ifndef MLIR_TYPESCRIPT_LOWERTOLLVMLOGIC_DEFAULTLOGIC_H_
#define MLIR_TYPESCRIPT_LOWERTOLLVMLOGIC_DEFAULTLOGIC_H_

#include "TypeScript/Config.h"
#include "TypeScript/Defines.h"
#include "TypeScript/Passes.h"
#include "TypeScript/TypeScriptDialect.h"
#include "TypeScript/TypeScriptOps.h"

#include "TypeScript/LowerToLLVM/CodeLogicHelper.h"
#include "TypeScript/LowerToLLVM/LLVMCodeHelper.h"
#include "TypeScript/LowerToLLVM/TypeConverterHelper.h"
#include "TypeScript/LowerToLLVM/TypeHelper.h"

#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"

#include "mlir/IR/PatternMatch.h"

using namespace mlir;
namespace mlir_ts = mlir::typescript;

namespace typescript
{

class DefaultLogic
{
    Operation *op;
    PatternRewriter &rewriter;
    TypeHelper th;
    LLVMCodeHelper ch;
    CodeLogicHelper clh;
    Location loc;

  public:
    DefaultLogic(Operation *op, PatternRewriter &rewriter, TypeConverterHelper &tch, Location loc, CompileOptions &compileOptions)
        : op(op), rewriter(rewriter), th(rewriter), ch(op, rewriter, tch.typeConverter, compileOptions), clh(op, rewriter), loc(loc)
    {
    }

    mlir::Value getDefaultValueForOrUndef(mlir::Type dataType)
    {
        mlir::Value defValue = getDefaultValueFor(dataType);
        if (!defValue)
        {
            defValue = rewriter.create<LLVM::UndefOp>(loc, dataType);
        }

        return defValue;
    }

    mlir::Value getDefaultValueFor(mlir::Type dataType)
    {
        // default value
        mlir::Value defaultValue;

        // TODO: finish for all types

        if (dataType.isa<LLVM::LLVMPointerType>())
        {
            defaultValue = rewriter.create<LLVM::ConstantOp>(loc, dataType, 0);
        }
        else if (dataType.isa<mlir::IntegerType>())
        {
            dataType.cast<mlir::IntegerType>().getWidth();
            defaultValue = clh.createIConstantOf(dataType.cast<mlir::IntegerType>().getWidth(), 0);
        }
        else if (dataType.isa<mlir::FloatType>())
        {
            dataType.cast<mlir::FloatType>().getWidth();
            defaultValue = clh.createFConstantOf(dataType.cast<mlir::FloatType>().getWidth(), 0.0);
        }

        return defaultValue;
    }
};
} // namespace typescript

#endif // MLIR_TYPESCRIPT_LOWERTOLLVMLOGIC_DEFAULTLOGIC_H_
