#ifndef ALIAS_PASS__H
#define ALIAS_PASS__H

#include "llvm/Pass.h"
#include "llvm/IR/PassManager.h"

namespace ts
{
    class AliasPass : public llvm::PassInfoMixin<AliasPass>
    {
    private:
        bool isWasm;
        size_t intSize;

    public:
        AliasPass(bool isWasm, size_t intSize) : isWasm(isWasm), intSize(intSize) {
        }

        llvm::PreservedAnalyses run(llvm::Module &M, llvm::ModuleAnalysisManager &MAM);

        static bool isRequired()
        {
            return true;
        }
    };  
}

#endif // ALIAS_PASS__H
