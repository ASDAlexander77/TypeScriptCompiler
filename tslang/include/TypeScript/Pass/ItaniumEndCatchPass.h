#ifndef ITANIUM_END_CATCH_PASS__H
#define ITANIUM_END_CATCH_PASS__H

#include "llvm/Pass.h"
#include "llvm/IR/PassManager.h"

namespace ts
{
    // Itanium EH: a catch is ended (__cxa_end_catch) on every path out of it, unwind paths
    // included. See ItaniumEndCatchPass.cpp.
    class ItaniumEndCatchPass : public llvm::PassInfoMixin<ItaniumEndCatchPass>
    {
    public:
        llvm::PreservedAnalyses run(llvm::Function &F, llvm::FunctionAnalysisManager &AM);

        static bool isRequired()
        {
            return true;
        }
    };
}

#endif // ITANIUM_END_CATCH_PASS__H
