#ifndef TYPESCRIPTEXCEPTIONPASS__H
#define TYPESCRIPTEXCEPTIONPASS__H

#include "llvm/Pass.h"
#include "llvm/IR/PassManager.h"

namespace ts
{
    class TypeScriptExceptionPass : public llvm::PassInfoMixin<TypeScriptExceptionPass>
    {
    public:
        llvm::PreservedAnalyses run(llvm::Function &F, llvm::FunctionAnalysisManager &AM);

        static bool isRequired()
        {
            return true;
        }
    };
}

#endif // TYPESCRIPTEXCEPTIONPASS__H
