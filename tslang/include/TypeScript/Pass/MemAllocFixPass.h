#ifndef MEMALLOCFIX_PASS__H
#define MEMALLOCFIX_PASS__H

#include "llvm/Pass.h"
#include "llvm/IR/PassManager.h"

namespace ts
{
    class MemAllocFixPass : public llvm::PassInfoMixin<MemAllocFixPass>
    {
    private:
        int intSize;

    public:
        MemAllocFixPass(int intSize) : intSize(intSize) {}

        llvm::PreservedAnalyses run(llvm::Module &M, llvm::ModuleAnalysisManager &MAM);

        static bool isRequired()
        {
            return true;
        }
    };
}

#endif // MEMALLOCFIX_PASS__H
