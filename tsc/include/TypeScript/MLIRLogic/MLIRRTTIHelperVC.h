#ifndef MLIR_TYPESCRIPT_LOWERTOLLVMLOGIC_MLIRRTTIHELPERVC_H_
#define MLIR_TYPESCRIPT_LOWERTOLLVMLOGIC_MLIRRTTIHELPERVC_H_

#include "TypeScript/MLIRLogic/MLIRRTTIHelperVCWin32.h"
#include "TypeScript/MLIRLogic/MLIRRTTIHelperVCLinux.h"

namespace typescript
{

class MLIRRTTIHelperVC
{
  private: 
    mlir::OpBuilder &rewriter;
    mlir::ModuleOp &parentModule;    
    bool isWasm;
    bool isWindows;

  public:
    MLIRRTTIHelperVC(mlir::OpBuilder &rewriter, mlir::ModuleOp &parentModule, CompileOptions &compileOptions) 
        : rewriter(rewriter), parentModule(parentModule), isWasm(compileOptions.isWasm), isWindows(compileOptions.isWindows)
    {
    }

    bool setRTTIForType(mlir::Location loc, mlir::Type type, std::function<ClassInfo::TypePtr(StringRef fullClassName)> resolveClassInfo)
    {
        if (isWasm)
        {
            llvm_unreachable("not implemented");
        }
        else if (isWindows)
        {
            MLIRRTTIHelperVCWin32 rtti(rewriter, parentModule);
            return rtti.setRTTIForType(loc, type, resolveClassInfo);
        }
        else
        {
            MLIRRTTIHelperVCLinux rtti(rewriter, parentModule);
            return rtti.setRTTIForType(loc, type, resolveClassInfo);
        }

        return false;
    }
};
} // namespace typescript

#endif // MLIR_TYPESCRIPT_LOWERTOLLVMLOGIC_LLVMRTTIHELPERVC_H_
