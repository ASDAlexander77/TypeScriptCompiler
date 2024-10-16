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

#include <functional>

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
        mlir::Location ret = loc;
        mlir::TypeSwitch<mlir::Location>(loc)
            .Case<mlir::FileLineColLoc>([&](mlir::FileLineColLoc loc) {
                // nothing todo
            })
            .Case<mlir::NameLoc>([&](mlir::NameLoc loc) {
                auto newChildLoc = stripMetadata(loc.getChildLoc());
                if (newChildLoc != loc.getChildLoc()) {
                    ret = NameLoc::get(loc.getName(), newChildLoc);
                }
            })
            .Case<mlir::OpaqueLoc>([&](mlir::OpaqueLoc loc) {
                auto newFallbackLoc = stripMetadata(loc.getFallbackLocation());
                if (newFallbackLoc != loc.getFallbackLocation()) {
                    ret = OpaqueLoc::get(loc.getContext(), newFallbackLoc);
                }
            })
            .Case<mlir::CallSiteLoc>([&](mlir::CallSiteLoc loc) {
                auto newCallerLoc = stripMetadata(loc.getCaller());
                if (newCallerLoc != loc.getCaller()) {
                    ret = mlir::CallSiteLoc::get(loc.getCallee(), newCallerLoc);
                }
            })        
            .Case<mlir::FusedLoc>([&](mlir::FusedLoc loc) {
                SmallVector<mlir::Location> newLocs;
                auto newLoc = false;
                for (auto subLoc : loc.getLocations())
                {
                    auto newSubLoc = stripMetadata(subLoc);
                    newLocs.push_back(newSubLoc);
                    newLoc |= newSubLoc != subLoc;
                }

                if (loc.getMetadata() || newLoc) {
                    ret = mlir::FusedLoc::get(loc.getContext(), newLocs);
                }
            })
            .Default([&](mlir::Location loc) { 
                llvm_unreachable("not implemented");
            });     

        return ret;   
    }

    static mlir::Location replaceMetadata(mlir::Location loc, mlir::Attribute newMetadata)
    {
        return replaceMetadata(loc, [&](mlir::Attribute currentMetadata) {
            return newMetadata;
        });   
    }

    static mlir::Location replaceMetadataIfEquals(mlir::Location loc, mlir::Attribute newMetadata, mlir::Attribute oldMetadata)
    {
        return replaceMetadata(loc, [&](mlir::Attribute currentMetadata) {
            return (currentMetadata == oldMetadata) ? newMetadata : currentMetadata;
        });
    }    

    static mlir::Location replaceScope(mlir::Location loc, mlir::Attribute newScope, mlir::Attribute oldScope)
    {
        return replaceMetadata(loc, [&](mlir::Attribute currentMetadata) -> mlir::Attribute {
            if (currentMetadata == oldScope) 
            {
                return newScope;
            }

            if (auto lexicalBlockAttr = currentMetadata.dyn_cast_or_null<mlir::LLVM::DILexicalBlockAttr>()) {
                if (lexicalBlockAttr.getScope() == oldScope) {
                    auto newLexicalBlockAttr = 
                        mlir::LLVM::DILexicalBlockAttr::get(
                            currentMetadata.getContext(), 
                            newScope.cast<mlir::LLVM::DIScopeAttr>(), 
                            lexicalBlockAttr.getFile(), 
                            lexicalBlockAttr.getLine(), 
                            lexicalBlockAttr.getColumn());   
                    return newLexicalBlockAttr;
                }
            }

            if (auto localVarScope = currentMetadata.dyn_cast_or_null<mlir::LLVM::DILocalVariableAttr>()) {
                if (localVarScope.getScope() == oldScope) {
                    auto newLocalVar = mlir::LLVM::DILocalVariableAttr::get(
                        localVarScope.getContext(), 
                        newScope.cast<mlir::LLVM::DIScopeAttr>(), 
                        localVarScope.getName(), 
                        localVarScope.getFile(), 
                        localVarScope.getLine(), 
                        localVarScope.getArg(), 
                        localVarScope.getAlignInBits(), 
                        localVarScope.getType());
                    return newLocalVar;
                }
            }

            if (auto subprogScope = currentMetadata.dyn_cast_or_null<mlir::LLVM::DISubprogramAttr>()) {
                auto newSubprogramAttr = mlir::LLVM::DISubprogramAttr::get(
                    subprogScope.getContext(), 
                    subprogScope.getCompileUnit(), 
                    newScope.cast<mlir::LLVM::DIScopeAttr>(), 
                    subprogScope.getName(), 
                    subprogScope.getLinkageName(), 
                    subprogScope.getFile(), 
                    subprogScope.getLine(), 
                    subprogScope.getScopeLine(), 
                    subprogScope.getSubprogramFlags(), 
                    subprogScope.getType());
                return newSubprogramAttr;
            }

            // default case
            return currentMetadata;
        });
    }        

    static mlir::Location replaceMetadata(mlir::Location loc, std::function<mlir::Attribute(mlir::Attribute)> f)
    {
        mlir::Location ret = loc;        
        mlir::TypeSwitch<mlir::Location>(loc)
            .Case<mlir::FileLineColLoc>([&](mlir::FileLineColLoc loc) {
                // nothing todo
            })
            .Case<mlir::NameLoc>([&](mlir::NameLoc loc) {
                auto newChildLoc = replaceMetadata(loc.getChildLoc(), f);
                if (newChildLoc != loc.getChildLoc()) {
                    ret = NameLoc::get(loc.getName(), newChildLoc);
                }
            })
            .Case<mlir::OpaqueLoc>([&](mlir::OpaqueLoc loc) {
                auto newFallbackLoc = replaceMetadata(loc.getFallbackLocation(), f);
                if (newFallbackLoc != loc.getFallbackLocation()) {
                    ret = OpaqueLoc::get(loc.getContext(), newFallbackLoc);
                }
            })
            .Case<mlir::CallSiteLoc>([&](mlir::CallSiteLoc loc) {
                auto newCallerLoc = replaceMetadata(loc.getCaller(), f);
                if (newCallerLoc != loc.getCallee()) {
                    ret = mlir::CallSiteLoc::get(loc.getCallee(), newCallerLoc);
                }
            })        
            .Case<mlir::FusedLoc>([&](mlir::FusedLoc loc) {
                auto newMetadata = f(loc.getMetadata());
                if (loc.getMetadata() != newMetadata)
                {
                    SmallVector<mlir::Location> newLocs;
                    for (auto subLoc : loc.getLocations())
                    {
                        newLocs.push_back(replaceMetadata(subLoc, f));
                    }

                    ret = mlir::FusedLoc::get(loc.getContext(), newLocs, newMetadata);
                }
            })
            .Default([&](mlir::Location loc) { 
                llvm_unreachable("not implemented");
            });     

        return ret;   
    }  
  
    static void walkMetadata(mlir::Location loc, std::function<void(mlir::Attribute)> f)
    {
        mlir::TypeSwitch<mlir::Location>(loc)
            .Case<mlir::FileLineColLoc>([&](mlir::FileLineColLoc loc) {
                // nothing todo
            })
            .Case<mlir::NameLoc>([&](mlir::NameLoc loc) {
                walkMetadata(loc.getChildLoc(), f);
            })
            .Case<mlir::OpaqueLoc>([&](mlir::OpaqueLoc loc) {
                walkMetadata(loc.getFallbackLocation(), f);
            })
            .Case<mlir::CallSiteLoc>([&](mlir::CallSiteLoc loc) {
                walkMetadata(loc.getCaller(), f);
            })        
            .Case<mlir::FusedLoc>([&](mlir::FusedLoc loc) {
                f(loc.getMetadata());
                for (auto subLoc : loc.getLocations())
                {
                    walkMetadata(subLoc, f);
                }
            })
            .Default([&](mlir::Location loc) { 
                llvm_unreachable("not implemented");
            });     
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
            auto subprogramAttr = mlir::LLVM::DISubprogramAttr::get(
                rewriter.getContext(), 
                oldMetadata.getCompileUnit(), 
                oldMetadata.getScope(), 
                oldMetadata.getName(), 
                oldMetadata.getLinkageName(), 
                oldMetadata.getFile(), 
                oldMetadata.getLine(), 
                oldMetadata.getScopeLine(), 
                oldMetadata.getSubprogramFlags(), 
                subroutineTypeAttr);

            LLVM_DEBUG(llvm::dbgs() << "\n!! new prog attr: " << subprogramAttr << "\n");

            newFuncOp.walk([&, subprogramAttr](Operation *op) {

                LLVM_DEBUG(llvm::dbgs() << "\n!! replacing for: " << *op << "\n");

                op->setLoc(replaceScope(op->getLoc(), subprogramAttr, oldMetadata));

                // Strip block arguments debug info.
                for (auto &region : op->getRegions()) {
                    for (auto &block : region.getBlocks()) {
                        for (auto &arg : block.getArguments()) {
                            arg.setLoc(replaceScope(arg.getLoc(), subprogramAttr, oldMetadata));
                        }
                    }
                }                
            });

            //debugPrint(newFuncOp);
        }
    }

    void removeAllMetadata(mlir::func::FuncOp newFuncOp) {
        newFuncOp->walk([&](Operation *op) {
            op->setLoc(stripMetadata(op->getLoc()));

            // Strip block arguments debug info.
            for (auto &region : op->getRegions()) {
                for (auto &block : region.getBlocks()) {
                    for (auto &arg : block.getArguments()) {
                        arg.setLoc(stripMetadata(arg.getLoc()));
                    }
                }
            }
        });
    }    

    void debugPrint(mlir::func::FuncOp newFuncOp) {

        // debug
        walkMetadata(newFuncOp->getLoc(), [&](mlir::Attribute metadata) {
            LLVM_DEBUG(llvm::dbgs() << "\n!! metadata: " << metadata << "\n");
        });

        newFuncOp.walk([&](Operation *op) {

            walkMetadata(op->getLoc(), [&](mlir::Attribute metadata) {
                LLVM_DEBUG(llvm::dbgs() << "\n!! metadata: " << metadata << "\n");
            });

            // Strip block arguments debug info.
            for (auto &region : op->getRegions()) {
                for (auto &block : region.getBlocks()) {
                    for (auto &arg : block.getArguments()) {
                        walkMetadata(arg.getLoc(), [&](mlir::Attribute metadata) {
                            LLVM_DEBUG(llvm::dbgs() << "\n!! metadata: " << metadata << "\n");
                        });
                    }
                }
            }                
        });

    }
};

}

#undef DEBUG_TYPE

#endif // MLIR_TYPESCRIPT_LOWERTOLLVMLOGIC_LLVMDEBUGINFOFIXER_H_