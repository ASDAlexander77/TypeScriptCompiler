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
        auto defValue = rewriter.create<LLVM::ZeroOp>(loc, dataType);
        return defValue;
    }
};
} // namespace typescript

#endif // MLIR_TYPESCRIPT_LOWERTOLLVMLOGIC_DEFAULTLOGIC_H_
