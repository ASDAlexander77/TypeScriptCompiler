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

class LLVMDebugInfoHelperCreator
{
    LLVM::LLVMFuncOp llvmFuncOp;
    const LLVMTypeConverter *typeConverter;
    MLIRContext *context;

public:
    LLVMDebugInfoHelperCreator(LLVM::LLVMFuncOp llvmFuncOp, const LLVMTypeConverter *typeConverter)
        : llvmFuncOp(llvmFuncOp), typeConverter(typeConverter)
    {
        context = llvmFuncOp.getContext();
    }

    auto createSubProgramAttr() -> void
    {
        auto location = llvmFuncOp->getLoc();
        if (auto funcLocWithSubprog = dyn_cast<mlir::FusedLocWith<mlir::LLVM::DISubprogramAttr>>(location))
        {
            return;
        }

        auto moduleOp = llvmFuncOp->getParentOp();
        auto moduleLocation = moduleOp->getLoc();
        if (auto funcLocWithCompileUnit = dyn_cast<mlir::FusedLocWith<mlir::LLVM::DICompileUnitAttr>>(moduleLocation))
        {
            auto compileUnit = funcLocWithCompileUnit.getMetadata();

            LocationHelper lh(context);
            LLVMTypeConverterHelper llvmtch(typeConverter);
            LLVMDebugInfoHelper di(context, llvmtch);        

            auto funcNameAttr = llvmFuncOp.getNameAttr();

            // return type
            auto [file, lineAndColumn] = lh.getLineAndColumnAndFile(location);
            auto [line, column] = lineAndColumn;

            SmallVector <mlir::LLVM::DITypeAttr> resultTypes;
            for (auto resType : llvmFuncOp.getResultTypes())
            {
                auto diType = di.getDIType(location, {}, resType, file, line, file);
                resultTypes.push_back(diType);
            }

            if (llvmFuncOp.getArgumentTypes().size() > 0 &&  llvmFuncOp.getResultTypes().size() == 0)
            {
                // return type is null
                resultTypes.push_back(mlir::LLVM::DINullTypeAttr());
            }

            for (auto argType : llvmFuncOp.getArgumentTypes())
            {
                auto diType = di.getDIType(location, {}, argType, file, line, file);
                resultTypes.push_back(diType);
            }

            auto subroutineTypeAttr = mlir::LLVM::DISubroutineTypeAttr::get(context, llvm::dwarf::DW_CC_normal, resultTypes);
            auto subprogramAttr = mlir::LLVM::DISubprogramAttr::get(
                context, 
                DistinctAttr::create(mlir::UnitAttr::get(context)),
                compileUnit, 
                compileUnit, 
                funcNameAttr, 
                funcNameAttr, 
                compileUnit.getFile(), 
                line, 
                line, 
                LLVM::DISubprogramFlags::Pure, 
                subroutineTypeAttr);

            LLVM_DEBUG(llvm::dbgs() << "\n!! new prog attr: " << subprogramAttr << "\n");        

            // we do not use replaceScope here as we are fixing metadata
            auto newLoc = mlir::FusedLoc::get(location.getContext(), {location}, subprogramAttr);
            llvmFuncOp->setLoc(newLoc);            
        }        
    }

    void clear() {
        llvmFuncOp->setLoc(mlir::UnknownLoc::get(llvmFuncOp.getContext()));
    }
};

class LLVMDebugInfoHelperFixer
{
    mlir_ts::FuncOp funcOp;
    const LLVMTypeConverter *typeConverter;
    MLIRContext *context;

public:
    LLVMDebugInfoHelperFixer(mlir_ts::FuncOp funcOp, const LLVMTypeConverter *typeConverter)
        : funcOp(funcOp), typeConverter(typeConverter)
    {
        context = funcOp.getContext();
    }

    void fix() {
        auto location = funcOp->getLoc();
        if (auto funcLocWithSubprog = dyn_cast<mlir::FusedLocWith<mlir::LLVM::DISubprogramAttr>>(location))
        {
            LocationHelper lh(context);
            LLVMTypeConverterHelper llvmtch(typeConverter);
            LLVMDebugInfoHelper di(context, llvmtch);        

            auto oldMetadata = funcLocWithSubprog.getMetadata();
            auto funcNameAttr = funcOp.getNameAttr();

            // return type
            auto [file, lineAndColumn] = lh.getLineAndColumnAndFile(location);
            auto [line, column] = lineAndColumn;

            SmallVector <mlir::LLVM::DITypeAttr> resultTypes;
            for (auto resType : funcOp.getResultTypes())
            {
                auto diType = di.getDIType(location, {}, resType, file, line, file);
                resultTypes.push_back(diType);
            }

            if (funcOp.getArgumentTypes().size() > 0 &&  funcOp.getResultTypes().size() == 0)
            {
                // return type is null
                resultTypes.push_back(mlir::LLVM::DINullTypeAttr());
            }

            for (auto argType : funcOp.getArgumentTypes())
            {
                auto diType = di.getDIType(location, {}, argType, file, line, file);
                resultTypes.push_back(diType);
            }

            auto subroutineTypeAttr = mlir::LLVM::DISubroutineTypeAttr::get(context, llvm::dwarf::DW_CC_normal, resultTypes);
            auto subprogramAttr = mlir::LLVM::DISubprogramAttr::get(
                context, 
                DistinctAttr::create(mlir::UnitAttr::get(context)),
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

            // we do not use replaceScope here as we are fixing metadata
            funcOp->setLoc(replaceScope(location, subprogramAttr, oldMetadata));

            fixNestedOpScope(subprogramAttr, oldMetadata);
            //debugPrint(funcOp);
        }
    }

private:
    static void removeAllMetadata(mlir::func::FuncOp newFuncOp) {
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

    static mlir::Location stripMetadata(mlir::Location loc)
    {
        mlir::Location ret = loc;
        mlir::TypeSwitch<mlir::Location>(loc)
            .Case<mlir::FileLineColLoc>([&](mlir::FileLineColLoc loc) {
                // nothing todo
            })
            .Case<mlir::FusedLoc>([&](mlir::FusedLoc loc) {
                SmallVector<mlir::Location> newLocs;
                auto anyNewLoc = false;
                for (auto subLoc : loc.getLocations())
                {
                    auto newSubLoc = stripMetadata(subLoc);
                    newLocs.push_back(newSubLoc);
                    anyNewLoc |= newSubLoc != subLoc;
                }

                if (loc.getMetadata() || anyNewLoc) {
                    ret = mlir::FusedLoc::get(loc.getContext(), anyNewLoc ? newLocs : loc.getLocations());
                }
            })
            .Case<mlir::UnknownLoc>([&](mlir::UnknownLoc loc) {
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
            .Default([&](mlir::Location loc) { 
                llvm_unreachable("not implemented");
            });     

        return ret;   
    }

    mlir::LLVM::DILexicalBlockAttr recreateLexicalBlockForNewScope(mlir::LLVM::DILexicalBlockAttr lexicalBlockAttr, mlir::Attribute newScope, mlir::Attribute oldScope) {
        if (lexicalBlockAttr.getScope() == oldScope) {
            auto newLexicalBlockAttr = 
                mlir::LLVM::DILexicalBlockAttr::get(
                    lexicalBlockAttr.getContext(), 
                    mlir::cast<mlir::LLVM::DIScopeAttr>(newScope), 
                    lexicalBlockAttr.getFile(), 
                    lexicalBlockAttr.getLine(), 
                    lexicalBlockAttr.getColumn());   

            // we need to fix nested scope again
            fixNestedOpScope(newLexicalBlockAttr, lexicalBlockAttr);
            
            return newLexicalBlockAttr;
        }

        return lexicalBlockAttr;
    }
    
    static mlir::LLVM::DILocalVariableAttr recreateLocalVariableForNewScope(mlir::LLVM::DILocalVariableAttr localVarScope, mlir::Attribute newScope, mlir::Attribute oldScope) {
        if (localVarScope.getScope() == oldScope) {
            auto newLocalVar = mlir::LLVM::DILocalVariableAttr::get(
                localVarScope.getContext(), 
                mlir::cast<mlir::LLVM::DIScopeAttr>(newScope), 
                localVarScope.getName(), 
                localVarScope.getFile(), 
                localVarScope.getLine(), 
                localVarScope.getArg(), 
                localVarScope.getAlignInBits(), 
                localVarScope.getType());
            return newLocalVar;
        }

        return localVarScope;
    }

    mlir::LLVM::DISubprogramAttr recreateSubprogramForNewScope(mlir::LLVM::DISubprogramAttr subprogScope, mlir::Attribute newScope, mlir::Attribute oldScope) {
        if (subprogScope.getScope() == oldScope) {
            auto newSubprogramAttr = mlir::LLVM::DISubprogramAttr::get(
                subprogScope.getContext(), 
                DistinctAttr::create(mlir::UnitAttr::get(subprogScope.getContext())),
                subprogScope.getCompileUnit(), 
                mlir::cast<mlir::LLVM::DIScopeAttr>(newScope), 
                subprogScope.getName(), 
                subprogScope.getLinkageName(), 
                subprogScope.getFile(), 
                subprogScope.getLine(), 
                subprogScope.getScopeLine(), 
                subprogScope.getSubprogramFlags(), 
                subprogScope.getType());

            // we need to fix nested scope again
            fixNestedOpScope(newSubprogramAttr, subprogScope);

            return newSubprogramAttr;
        }

        return subprogScope;
    }

    static mlir::LLVM::DILabelAttr recreateLabelForNewScope(mlir::LLVM::DILabelAttr labelScope, mlir::Attribute newScope, mlir::Attribute oldScope) {
        if (labelScope.getScope() == oldScope) {
            auto newLabel = mlir::LLVM::DILabelAttr::get(
                labelScope.getContext(), 
                mlir::cast<mlir::LLVM::DIScopeAttr>(newScope), 
                labelScope.getName(), 
                labelScope.getFile(), 
                labelScope.getLine());
            return newLabel;
        }

        return labelScope;
    }

    mlir::Attribute recreateMetadataForNewScope(mlir::Attribute currentMetadata, mlir::Attribute newScope, mlir::Attribute oldScope) {
        if (auto lexicalBlockAttr = dyn_cast_or_null<mlir::LLVM::DILexicalBlockAttr>(currentMetadata)) {
            return recreateLexicalBlockForNewScope(lexicalBlockAttr, newScope, oldScope);
        }

        if (auto localVarScope = dyn_cast_or_null<mlir::LLVM::DILocalVariableAttr>(currentMetadata)) {
            return recreateLocalVariableForNewScope(localVarScope, newScope, oldScope);
        }

        if (auto subprogScope = dyn_cast_or_null<mlir::LLVM::DISubprogramAttr>(currentMetadata)) {
            return recreateSubprogramForNewScope(subprogScope, newScope, oldScope);
        }

        if (auto labelScope = dyn_cast_or_null<mlir::LLVM::DILabelAttr>(currentMetadata)) {
            return recreateLabelForNewScope(labelScope, newScope, oldScope);
        }

        return currentMetadata;
    }

    mlir::Location replaceScope(mlir::Location loc, mlir::Attribute newScope, mlir::Attribute oldScope)
    {
        return replaceMetadata(loc, [&](mlir::Attribute currentMetadata) -> mlir::Attribute {
            if (currentMetadata == oldScope) 
            {
                return newScope;
            }

            return recreateMetadataForNewScope(currentMetadata, newScope, oldScope);
        });
    }        

    static mlir::Location replaceMetadata(mlir::Location loc, std::function<mlir::Attribute(mlir::Attribute)> f)
    {
        mlir::Location ret = loc;        
        mlir::TypeSwitch<mlir::Location>(loc)
            .Case<mlir::FileLineColLoc>([&](mlir::FileLineColLoc loc) {
                // nothing todo
            })
            .Case<mlir::FusedLoc>([&](mlir::FusedLoc loc) {              
                SmallVector<mlir::Location> newLocs;
                auto anyNew = false;
                for (auto subLoc : loc.getLocations())
                {
                    auto newSubLoc = replaceMetadata(subLoc, f);
                    newLocs.push_back(newSubLoc);
                    anyNew |= newSubLoc != subLoc;
                }

                auto newMetadata = f(loc.getMetadata());
                if (loc.getMetadata() != newMetadata || anyNew)
                {
                    ret = mlir::FusedLoc::get(loc.getContext(), anyNew ? newLocs : loc.getLocations(), newMetadata);
                }
            })
            .Case<mlir::UnknownLoc>([&](mlir::UnknownLoc loc) {
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
            .Default([&](mlir::Location loc) { 
                LLVM_DEBUG(llvm::dbgs() << "\n!! location: " << loc << "\n");
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
            .Case<mlir::FusedLoc>([&](mlir::FusedLoc loc) {
                f(loc.getMetadata());
                for (auto subLoc : loc.getLocations())
                {
                    walkMetadata(subLoc, f);
                }
            })
            .Case<mlir::UnknownLoc>([&](mlir::UnknownLoc loc) {
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
            .Default([&](mlir::Location loc) { 
                llvm_unreachable("not implemented");
            });     
    }   

    void fixNestedOpScope(mlir::Attribute newMetadata, mlir::Attribute oldMetadata) {
        funcOp.walk([&, newMetadata, oldMetadata](Operation *op) {

            LLVM_DEBUG(llvm::dbgs() << "\n!! replacing for: " << *op << "\n");

            op->setLoc(replaceScope(op->getLoc(), newMetadata, oldMetadata));

            // Strip block arguments debug info.
            for (auto &region : op->getRegions()) {
                for (auto &block : region.getBlocks()) {
                    for (auto &arg : block.getArguments()) {
                        arg.setLoc(replaceScope(arg.getLoc(), newMetadata, oldMetadata));
                    }
                }
            }                

            // fix metadata for llvm.dbg.declare & llvm.dbg.value
            if (auto dbgDeclare = dyn_cast<mlir::LLVM::DbgDeclareOp>(op)) {
                auto newLocVar = recreateLocalVariableForNewScope(dbgDeclare.getVarInfo(), newMetadata, oldMetadata);
                if (newLocVar != dbgDeclare.getVarInfo()) {
                    dbgDeclare.setVarInfoAttr(newLocVar);
                }
            } else if (auto dbgValue = dyn_cast<mlir::LLVM::DbgValueOp>(op)) {
                auto newLocVar = recreateLocalVariableForNewScope(dbgValue.getVarInfo(), newMetadata, oldMetadata);
                if (newLocVar != dbgDeclare.getVarInfo()) {
                    dbgValue.setVarInfoAttr(newLocVar);
                }
            } else if (auto dbgLabel = dyn_cast<mlir::LLVM::DbgLabelOp>(op)) {
                auto newLabel = recreateLabelForNewScope(dbgLabel.getLabel(), newMetadata, oldMetadata);
                if (newLabel != dbgLabel.getLabel()) {
                    dbgLabel.setLabelAttr(newLabel);
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