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

    static mlir::Location stripMetadata(mlir::Location loc)
    {
        mlir::Location ret = UnknownLoc::get(loc.getContext());
        mlir::TypeSwitch<mlir::Location>(loc)
            .Case<mlir::FileLineColLoc>([&](mlir::FileLineColLoc loc) {
                // nothing todo
                ret = loc;
            })
            .Case<mlir::NameLoc>([&](mlir::NameLoc loc) {
                auto newChildLoc = stripMetadata(loc.getChildLoc());
                ret = NameLoc::get(loc.getName(), newChildLoc);
            })
            .Case<mlir::OpaqueLoc>([&](mlir::OpaqueLoc loc) {
                auto newFallbackLoc = stripMetadata(loc.getFallbackLocation());
                ret = OpaqueLoc::get(loc.getContext(), newFallbackLoc);
            })
            .Case<mlir::CallSiteLoc>([&](mlir::CallSiteLoc loc) {
                auto newCallerLoc = stripMetadata(loc.getCaller());
                ret = mlir::CallSiteLoc::get(loc.getCallee(), newCallerLoc);
            })        
            .Case<mlir::FusedLoc>([&](mlir::FusedLoc loc) {
                SmallVector<mlir::Location> newLocs;
                for (auto subLoc : loc.getLocations())
                {
                    newLocs.push_back(stripMetadata(subLoc));
                }

                ret = mlir::FusedLoc::get(loc.getContext(), newLocs);
            })
            .Default([&](mlir::Location loc) { 
                llvm_unreachable("not implemented");
            });     

        return ret;   
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

    void removeAllMetadata(mlir::func::FuncOp newFuncOp) {
        newFuncOp->walk([&](Operation *op) {
            op->setLoc(stripMetadata(op->getLoc()));

            // Strip block arguments debug info.
            // for (auto &region : op->getRegions()) {
            //     for (auto &block : region.getBlocks()) {
            //         for (auto &arg : block.getArguments()) {
            //             arg.setLoc(stripMetadata(arg.getLoc()));
            //         }
            //     }
            // }
        });
    }    
};

}

#undef DEBUG_TYPE

#endif // MLIR_TYPESCRIPT_LOWERTOLLVMLOGIC_LLVMDEBUGINFOFIXER_H_