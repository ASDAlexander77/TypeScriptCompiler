#ifndef DEBUGINFO_PATCH_PASS__H
#define DEBUGINFO_PATCH_PASS__H

#include "llvm/Pass.h"
#include "llvm/IR/PassManager.h"

namespace ts
{
    class DebugInfoPatchPass : public llvm::PassInfoMixin<DebugInfoPatchPass>
    {
    public:
        llvm::PreservedAnalyses run(llvm::Function &F, llvm::FunctionAnalysisManager &AM);

        static bool isRequired()
        {
            return true;
        }
    };
}

#endif // DEBUGINFO_PATCH_PASS__H
