#ifndef EXPORT_FIX_PASS__H
#define EXPORT_FIX_PASS__H

#include "llvm/Pass.h"
#include "llvm/IR/PassManager.h"

namespace ts
{
    class ExportFixPass : public llvm::PassInfoMixin<ExportFixPass>
    {
    private:
        bool isWindowsMSVCEnvironment;
        bool isWasm;

    public:
        ExportFixPass(bool isWindowsMSVCEnvironment, bool isWasm = false)
            : isWindowsMSVCEnvironment(isWindowsMSVCEnvironment), isWasm(isWasm) {
        }

        llvm::PreservedAnalyses run(llvm::Module &M, llvm::ModuleAnalysisManager &MAM);

        static bool isRequired()
        {
            return true;
        }
    };
}

#endif // EXPORT_FIX_PASS__H
