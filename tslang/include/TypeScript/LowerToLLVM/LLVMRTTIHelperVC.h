#ifndef MLIR_TYPESCRIPT_LOWERTOLLVMLOGIC_LLVMRTTIHELPERVC_H_
#define MLIR_TYPESCRIPT_LOWERTOLLVMLOGIC_LLVMRTTIHELPERVC_H_

#include "TypeScript/LowerToLLVM/LLVMRTTIHelperVCWin32.h"
#include "TypeScript/LowerToLLVM/LLVMRTTIHelperVCLinux.h"

using namespace mlir;
namespace mlir_ts = mlir::typescript;

namespace typescript
{

class LLVMRTTIHelperVC
{
    Operation *op;
    PatternRewriter &rewriter;
    const TypeConverter *typeConverter;
    CompileOptions &compileOptions;
    bool isWasm;
    bool isWindows;

  public:
    LLVMRTTIHelperVC(Operation *op, PatternRewriter &rewriter, const TypeConverter *typeConverter, CompileOptions &compileOptions)
        : op(op), rewriter(rewriter), typeConverter(typeConverter), compileOptions(compileOptions), isWasm(compileOptions.isWasm), isWindows(compileOptions.isWindows)
    {
    }

    LogicalResult setPersonality(mlir::func::FuncOp newFuncOp)
    {
        if (isWindows)
        {
            LLVMRTTIHelperVCWin32 rtti(op, rewriter, typeConverter, compileOptions);
            return rtti.setPersonality(newFuncOp);
        }
        else
        {
            LLVMRTTIHelperVCLinux rtti(op, rewriter, typeConverter, compileOptions);
            return rtti.setPersonality(newFuncOp);
        }

        return failure();
    }
};

} // namespace typescript

#endif // MLIR_TYPESCRIPT_LOWERTOLLVMLOGIC_LLVMRTTIHELPERVCWIN32_H_
