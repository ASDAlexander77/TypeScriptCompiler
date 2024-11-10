#ifndef LANDINGPAD_FIX_PASS__H
#define LANDINGPAD_FIX_PASS__H

#include "llvm/Pass.h"
#include "llvm/IR/PassManager.h"

namespace ts
{
    class LandingPadFixPass : public llvm::PassInfoMixin<LandingPadFixPass>
    {
    public:
        llvm::PreservedAnalyses run(llvm::Function &F, llvm::FunctionAnalysisManager &AM);

        static bool isRequired()
        {
            return true;
        }
    };
}

#endif // LANDINGPAD_FIX_PASS__H
