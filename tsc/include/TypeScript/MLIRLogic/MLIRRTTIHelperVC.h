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
    MLIRRTTIHelperVCWin32 rttiWin;
    MLIRRTTIHelperVCLinux rttiLinux;

  public:
    MLIRRTTIHelperVC(mlir::OpBuilder &rewriter, mlir::ModuleOp &parentModule, CompileOptions &compileOptions) 
        : rewriter(rewriter), parentModule(parentModule), isWasm(compileOptions.isWasm), isWindows(compileOptions.isWindows), 
          rttiWin(rewriter, parentModule, compileOptions), rttiLinux(rewriter, parentModule, compileOptions)
    {
    }

    bool setRTTIForType(mlir::Location loc, mlir::Type type, std::function<ClassInfo::TypePtr(StringRef fullClassName)> resolveClassInfo)
    {
        if (isWindows)
        {
            return rttiWin.setRTTIForType(loc, type, resolveClassInfo);
        }
        else
        {
            return rttiLinux.setRTTIForType(loc, type, resolveClassInfo);
        }

        return false;
    }

    bool hasType()
    {
        if (isWindows)
        {
            return rttiWin.hasType();
        }
        else
        {
            return rttiLinux.hasType();
        }

        return false;
    } 

    bool setType(mlir::Type type)
    {
        if (isWindows)
        {
            return rttiWin.setType(type);
        }
        else
        {
            return rttiLinux.setType(type);
        }

        return false;
    }    

    mlir::Value throwInfoPtrValue(mlir::Location loc)
    {
        if (isWindows)
        {
            return rttiWin.throwInfoPtrValue(loc);
        }
        else
        {
            return rttiLinux.throwInfoPtrValue(loc);
        }

        return mlir::Value();
    }

    mlir::Value typeInfoPtrValue(mlir::Location loc)
    {
        if (isWindows)
        {
            return rttiWin.typeInfoPtrValue(loc);
        }
        else
        {
            return rttiLinux.typeInfoPtrValue(loc);
        }

        return mlir::Value();
    } 

    mlir_ts::TupleType getLandingPadType()
    {
        if (isWindows)
        {
            return rttiWin.getLandingPadType();
        }
        else
        {
            return rttiLinux.getLandingPadType();
        }

        llvm_unreachable("not implemented");
    } 
};
} // namespace typescript

#endif // MLIR_TYPESCRIPT_LOWERTOLLVMLOGIC_LLVMRTTIHELPERVC_H_
