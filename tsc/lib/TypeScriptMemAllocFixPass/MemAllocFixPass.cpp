#include "TypeScript/MemAllocFixPass.h"

#include "llvm/IR/PatternMatch.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/IRBuilder.h"

using namespace llvm;
using namespace PatternMatch;

#define DEBUG_TYPE "pass"

struct MemAllocFixPassCode
{
    DebugInfoPatchPassCode()
    {
    }

    bool runOnFunction(Function &F)
    {
        auto MadeChange = false;

        LLVM_DEBUG(llvm::dbgs() << "\nMEM ALLOC PATCH Function: " << F.getName());
        LLVM_DEBUG(llvm::dbgs() << "\nMEM ALLOC Dump Before: ...\n" << F << "\n";);

        llvm::SmallVector<llvm::CallInst *> workSet;
        for (auto &I : instructions(F))
        {
            if (auto *DDI = dyn_cast<CallInst>(&I))
            {
                workSet.push_back(DDI);
                continue;
            }
        }        

        for (auto &DDI : workSet)
        {
            // TODO: ...
        }

        LLVM_DEBUG(llvm::dbgs() << "\n!! DI PATCH Change: " << MadeChange;);
        LLVM_DEBUG(llvm::dbgs() << "\n!! DI PATCH Dump After: ...\n" << F << "\n";);

        return MadeChange;
    }
};

namespace ts
{
    llvm::PreservedAnalyses MemAllocFixPass::run(llvm::Function &F, llvm::FunctionAnalysisManager &AM)
    {
        MemAllocFixPassCode LPF{};
        if (!LPF.runOnFunction(F))
        {
            return llvm::PreservedAnalyses::all();
        }

        return llvm::PreservedAnalyses::none();
    }
}