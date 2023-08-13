#include "TypeScript/MemAllocFixPass.h"

#include "llvm/IR/PatternMatch.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/IRBuilder.h"

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
            if (!mapName(F.getName(), newName))
            {
                F.setName("malloc");
                return true;
            }
        }

        llvm::SmallVector<llvm::CallInst *> workSet;
        for (auto &I : instructions(F))
        {
            if (auto *CI = dyn_cast<CallInst>(&I))
            {
                workSet.push_back(CI);
                continue;
            }
        }        

        for (auto &CI : workSet)
        {
            // TODO: ...
            LLVM_DEBUG(llvm::dbgs() << "\n!! call name: " << CI->getValueName(););
        }

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
    
    llvm::PreservedAnalyses MemAllocFixPass::run(llvm::Module &M, llvm::ModuleAnalysisManager &MAM)
    {
        MemAllocFixPassCode MAFP{};
        bool MadeChange = false;

        for (auto &F : M)
        {
            MadeChange |= MAFP.runOnFunction(F);
        }

        return MadeChange ? llvm::PreservedAnalyses::none() : llvm::PreservedAnalyses::all();     
    }
}