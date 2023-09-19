#ifndef WIN32EXCEPTIONPASS__H
#define WIN32EXCEPTIONPASS__H

#include "llvm/Pass.h"
#include "llvm/IR/PassManager.h"

namespace ts
{
    class Win32ExceptionPass : public llvm::PassInfoMixin<Win32ExceptionPass>
    {
    public:
        llvm::PreservedAnalyses run(llvm::Function &F, llvm::FunctionAnalysisManager &AM);

        static bool isRequired()
        {
            return true;
        }
    };
}

#endif // WIN32EXCEPTIONPASS__H
