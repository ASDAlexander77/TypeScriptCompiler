#include "TypeScript/Pass/MemAllocFixPass.h"

#include "llvm/IR/PatternMatch.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Module.h"

using namespace llvm;
using namespace PatternMatch;

#define DEBUG_TYPE "pass"

struct MemAllocFixPassCode
{
    MemAllocFixPassCode()
    {
    }

    bool runOnFunction(Function &F)
    {
        auto MadeChange = false;

        LLVM_DEBUG(llvm::dbgs() << "\nMEM ALLOC PATCH Function: " << F.getName());
        LLVM_DEBUG(llvm::dbgs() << "\nMEM ALLOC Dump Before: ...\n" << F << "\n";);

        if (F.empty())
        {
            StringRef newName;
            if (mapName(F.getName(), newName))
            {
                F.setName(newName);
                return true;
            }
        }

        // llvm::SmallVector<llvm::CallBase *> workSet;
        // for (auto &I : instructions(F))
        // {
        //     if (auto *CI = dyn_cast<CallBase>(&I))
        //     {
        //         workSet.push_back(CI);
        //         continue;
        //     }
        // }        

        // for (auto &CI : workSet)
        // {
        //     // TODO: ...
        //     LLVM_DEBUG(llvm::dbgs() << "\n!! call called func name: " << CI->getCalledFunction()->getName(););
        // }

        LLVM_DEBUG(llvm::dbgs() << "\n!! MEM ALLOC Change: " << MadeChange;);
        LLVM_DEBUG(llvm::dbgs() << "\n!! MEM ALLOC Dump After: ...\n" << F << "\n";);

        return MadeChange;
    }

    bool mapName(StringRef name, StringRef &newName)
    {
        if (name == "ts_malloc")
        {
            newName = "malloc";
        }
        else if (name == "ts_realloc")
        {
            newName = "realloc";
        }
        else if (name == "ts_free")
        {
            newName = "free";
        }
        else
        {
            return false;
        }

        return true;
    }    
};

namespace ts
{
    inline bool isToBeRemoved(StringRef name)
    {
        return (name == "malloc" || name == "realloc" || name == "free");
    }      
    
    llvm::PreservedAnalyses MemAllocFixPass::run(llvm::Module &M, llvm::ModuleAnalysisManager &MAM)
    {
        MemAllocFixPassCode MAFP{};
        bool MadeChange = false;

        // remove first
        llvm::SmallVector<Function *> removeSet;
        for (auto &F : M)
        {
            if (F.empty() && isToBeRemoved(F.getName()))
            {
                removeSet.push_back(&F);
            }
        }

        for (auto f : removeSet)
        {
            LLVM_DEBUG(llvm::dbgs() << "\n!! MEM ALLOC removing: " << *f;);
            MadeChange = true;
            f->eraseFromParent();
        }

        // process not removed
        for (auto &F : M)
        {
            MadeChange |= MAFP.runOnFunction(F);
        }

        return MadeChange ? llvm::PreservedAnalyses::none() : llvm::PreservedAnalyses::all();     
    }
}