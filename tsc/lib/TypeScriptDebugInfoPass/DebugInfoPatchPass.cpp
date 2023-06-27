#include "TypeScript/DebugInfoPatchPass.h"

#include "llvm/IR/PatternMatch.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/IRBuilder.h"

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

        // TODO: ...

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