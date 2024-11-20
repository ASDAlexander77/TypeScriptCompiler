#include "TypeScript/Pass/DebugInfoPatchPass.h"

#include "llvm/IR/PatternMatch.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/DIBuilder.h"

using namespace llvm;
using namespace PatternMatch;

#define DEBUG_TYPE "pass"

struct DebugInfoPatchPassCode
{
    DebugInfoPatchPassCode()
    {
    }

    bool runOnFunction(Function &F)
    {
        auto MadeChange = false;

        LLVM_DEBUG(llvm::dbgs() << "\nDI PATCH Function: " << F.getName());
        LLVM_DEBUG(llvm::dbgs() << "\nDI PATCH Dump Before: ...\n" << F << "\n";);

        llvm::SmallVector<llvm::DbgDeclareInst *> workSet;
        for (auto &I : instructions(F))
        {
            if (auto *DDI = dyn_cast<DbgDeclareInst>(&I))
            {
                workSet.push_back(DDI);
                continue;
            }
        }        

        DIBuilder DBuilder(*F.getParent());

        for (auto &DDI : workSet)
        {
            auto diVar = DDI->getVariable();
            LLVM_DEBUG(llvm::dbgs() << "\nDI VAR: " << *diVar);

            auto diType = diVar->getType();
            LLVM_DEBUG(llvm::dbgs() << "\nDI TYPE: " << *diType);

            if (auto diCompositeType = dyn_cast<llvm::DICompositeType>(diType))
            {
                if (diCompositeType->getTag() == dwarf::DW_TAG_array_type)
                {
                    LLVM_DEBUG(llvm::dbgs() << "\n!! DI found array";);      

                    /*
                        Metadata *getRawDataLocation() const { return getOperand(9); }
                        DIVariable *getDataLocation() const {
                            return dyn_cast_or_null<DIVariable>(getRawDataLocation());
                        }

                        void replaceTemplateParams(DITemplateParameterArray TemplateParams) {
                            replaceOperandWith(6, TemplateParams.get());
                        }                        
                    */

                    // replace dataLocation;
                    auto loadAddrFrom0 = DBuilder.createExpression({dwarf::DW_OP_push_object_address, dwarf::DW_OP_deref});
                    diCompositeType->replaceOperandWith(9, loadAddrFrom0);

                    // replace array size
                    auto elements = diCompositeType->getElements();
                    if (auto diSubrange = dyn_cast<llvm::DISubrange>(*elements.begin()))
                    {
                        LLVM_DEBUG(llvm::dbgs() << "\n!! DI subrange found";);      
                        /*
                          Metadata *getRawCountNode() const { return getOperand(0).get(); }
                        */

                        auto loadAddrFrom1 = DBuilder.createExpression({dwarf::DW_OP_push_object_address, dwarf::DW_OP_plus_uconst, 8/*32*/, dwarf::DW_OP_deref});
                        diSubrange->replaceOperandWith(0, loadAddrFrom1);                       
                    }

                    LLVM_DEBUG(llvm::dbgs() << "\nDI TYPE MODIFIED: " << *diCompositeType);

                    MadeChange = true;
                }
            }
        }

        LLVM_DEBUG(llvm::dbgs() << "\n!! DI PATCH Change: " << MadeChange;);
        LLVM_DEBUG(llvm::dbgs() << "\n!! DI PATCH Dump After: ...\n" << F << "\n";);

        return MadeChange;
    }
};

namespace ts
{
    llvm::PreservedAnalyses DebugInfoPatchPass::run(llvm::Function &F, llvm::FunctionAnalysisManager &AM)
    {
        DebugInfoPatchPassCode LPF{};
        if (!LPF.runOnFunction(F))
        {
            return llvm::PreservedAnalyses::all();
        }

        return llvm::PreservedAnalyses::none();
    }
}