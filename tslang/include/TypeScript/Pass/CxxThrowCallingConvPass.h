#ifndef CXXTHROWCALLINGCONVPASS__H
#define CXXTHROWCALLINGCONVPASS__H

#include "llvm/Pass.h"
#include "llvm/IR/PassManager.h"

namespace ts
{
    // _CxxThrowException is __stdcall on 32-bit x86 MSVC; its symbol is then __CxxThrowException@8.
    // Sets that convention on the declaration and on every call and invoke of it. A call whose
    // convention differs from its callee's is undefined behaviour, which InstCombine turns into
    // `unreachable`, so this runs before the optimization pipeline.
    class CxxThrowCallingConvPass : public llvm::PassInfoMixin<CxxThrowCallingConvPass>
    {
    public:
        llvm::PreservedAnalyses run(llvm::Module &M, llvm::ModuleAnalysisManager &AM);

        static bool isRequired()
        {
            return true;
        }
    };
}

#endif // CXXTHROWCALLINGCONVPASS__H
