#include "TypeScript/ExportFixPass.h"

#include "llvm/IR/PatternMatch.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/IRBuilder.h"

using namespace llvm;
using namespace PatternMatch;

#define DEBUG_TYPE "pass"

struct ExportFixPassCode
{
    bool isWindowsMSVCEnvironment;

    ExportFixPassCode(bool isWindowsMSVCEnvironment) : isWindowsMSVCEnvironment(isWindowsMSVCEnvironment)
    {
    }

    bool runOnFunction(Function &F)
    {
        auto MadeChange = false;

        LLVM_DEBUG(llvm::dbgs() << "\nEXPORT Function: " << F.getName());
        LLVM_DEBUG(llvm::dbgs() << "\nEXPORT Dump Before: ...\n" << F << "\n";);
        
        if (F.hasFnAttribute("export"))
        {
            F.removeFnAttr("export");

            // set DLLExport
            if (isWindowsMSVCEnvironment)
                F.setDLLStorageClass(llvm::GlobalVariable::DLLExportStorageClass);
            MadeChange = true;
        }

        if (F.hasFnAttribute("import"))
        {
            F.removeFnAttr("import");

            // set DLLExport
            if (isWindowsMSVCEnvironment)
                F.setDLLStorageClass(llvm::GlobalVariable::DLLImportStorageClass);
            MadeChange = true;
        }

        LLVM_DEBUG(llvm::dbgs() << "\n!! EXPORT Change: " << MadeChange;);
        LLVM_DEBUG(llvm::dbgs() << "\n!! EXPORT Dump After: ...\n" << F << "\n";);

        return MadeChange;
    }
};

namespace ts
{
    llvm::PreservedAnalyses ExportFixPass::run(llvm::Function &F, llvm::FunctionAnalysisManager &AM)
    {
        ExportFixPassCode LPF{isWindowsMSVCEnvironment};
        if (!LPF.runOnFunction(F))
        {
            return llvm::PreservedAnalyses::all();
        }

        return llvm::PreservedAnalyses::none();
    }
}