#ifndef MLIR_TYPESCRIPT_LOWERTOLLVMLOGIC_LLVMDEBUGINFOFIXER_H_
#define MLIR_TYPESCRIPT_LOWERTOLLVMLOGIC_LLVMDEBUGINFOFIXER_H_

#include "TypeScript/LowerToLLVM/LocationHelper.h"
#include "TypeScript/LowerToLLVM/LLVMDebugInfo.h"
#include "TypeScript/LowerToLLVM/LLVMTypeConverterHelper.h"

#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Builders.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"

#include "llvm/ADT/TypeSwitch.h"
#include "llvm/DebugInfo/DWARF/DWARFUnit.h"

#define DEBUG_TYPE "llvm"

using namespace mlir;
namespace mlir_ts = mlir::typescript;

namespace typescript
{

class LLVMDebugInfoHelperFixer
{
    PatternRewriter &rewriter;
    LLVMTypeConverter &typeConverter;
    CompileOptions &compileOptions;

  public:
    LLVMDebugInfoHelperFixer(PatternRewriter &rewriter, LLVMTypeConverter &typeConverter, CompileOptions &compileOptions)
        : rewriter(rewriter), typeConverter(typeConverter), compileOptions(compileOptions)
    {
    }

    void fixFuncOp(mlir::func::FuncOp newFuncOp) {
        auto location = newFuncOp->getLoc();
        if (auto funcLocWithSubprog = dyn_cast<mlir::FusedLocWith<mlir::LLVM::DISubprogramAttr>>(location))
        {
            LocationHelper lh(rewriter.getContext());
            LLVMTypeConverterHelper llvmtch(typeConverter);
            LLVMDebugInfoHelper di(rewriter.getContext(), llvmtch);        

            auto oldMetadata = funcLocWithSubprog.getMetadata();
            auto funcNameAttr = newFuncOp.getNameAttr();

            // return type
            auto [file, lineAndColumn] = lh.getLineAndColumnAndFile(location);
            auto [line, column] = lineAndColumn;

            SmallVector <mlir::LLVM::DITypeAttr> resultTypes;
            for (auto resType : newFuncOp.getResultTypes())
            {
                auto diType = di.getDIType({}, resType, file, line, file);
                resultTypes.push_back(diType);
            }

            auto subroutineTypeAttr = mlir::LLVM::DISubroutineTypeAttr::get(rewriter.getContext(), llvm::dwarf::DW_CC_normal, resultTypes);
            auto subprogramAttr = mlir::LLVM::DISubprogramAttr::get(rewriter.getContext(), oldMetadata.getCompileUnit(), oldMetadata.getScope(), 
                oldMetadata.getName(), oldMetadata.getLinkageName(), oldMetadata.getFile(), oldMetadata.getLine(), oldMetadata.getScopeLine(), 
                oldMetadata.getSubprogramFlags(), subroutineTypeAttr);

            LLVM_DEBUG(llvm::dbgs() << "\n!! new prog attr: " << subprogramAttr << "\n");

            auto newLocation = mlir::FusedLoc::get(rewriter.getContext(), funcLocWithSubprog.getLocations(), subprogramAttr);
            newFuncOp->setLoc(newLocation);

            newFuncOp.walk([&, newLocation, subprogramAttr](Operation *op) {
                auto opLocation = op->getLoc();

                LLVM_DEBUG(llvm::dbgs() << "\n!! operator: " << *op << " location: " << opLocation << "\n");

                if (auto scopeFusedLoc = opLocation.dyn_cast<mlir::FusedLocWith<LLVM::DIScopeAttr>>())
                {
                    auto metadata = scopeFusedLoc.getMetadata();
                    if (auto subprogramAttr = dyn_cast<mlir::LLVM::DISubprogramAttr>(metadata))
                    {
                        auto newOpLocation = mlir::FusedLoc::get(rewriter.getContext(), scopeFusedLoc.getLocations(), subprogramAttr);
                        op->setLoc(newOpLocation);
                    }
                }
            });

        }
    }

    void removeScope(mlir::func::FuncOp newFuncOp) {
        auto location = newFuncOp->getLoc();
        if (auto funcLocWithSubprog = dyn_cast<mlir::FusedLocWith<mlir::LLVM::DISubprogramAttr>>(location))
        {
            auto newLocation = mlir::FusedLoc::get(rewriter.getContext(), funcLocWithSubprog.getLocations());

            LLVM_DEBUG(llvm::dbgs() << "\n!! new location: " << newLocation << "\n");

            newFuncOp->setLoc(newLocation);

            newFuncOp.walk([&](Operation *op) {
                auto opLocation = op->getLoc();

                LLVM_DEBUG(llvm::dbgs() << "\n!! operator: " << *op << " location: " << opLocation << "\n");

                if (auto scopeFusedLoc = opLocation.dyn_cast<mlir::FusedLocWith<LLVM::DIScopeAttr>>())
                {
                    auto metadata = scopeFusedLoc.getMetadata();
                    if (auto subprogramAttr = dyn_cast<mlir::LLVM::DISubprogramAttr>(metadata))
                    {
                        auto newOpLocation = mlir::FusedLoc::get(rewriter.getContext(), scopeFusedLoc.getLocations());
                        //op->setLoc(newOpLocation);
                    }
                }
            });

        }
    }    

};

}

#undef DEBUG_TYPE

#endif // MLIR_TYPESCRIPT_LOWERTOLLVMLOGIC_LLVMDEBUGINFOFIXER_H_