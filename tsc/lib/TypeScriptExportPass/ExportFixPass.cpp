#include "TypeScript/Defines.h"
#include "TypeScript/ExportFixPass.h"

#include "llvm/IR/PatternMatch.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Module.h"

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

        if (F.hasFnAttribute(DLL_EXPORT))
        {
            F.removeFnAttr(DLL_EXPORT);

            // set DLLExport
            if (isWindowsMSVCEnvironment)
                F.setDLLStorageClass(llvm::GlobalVariable::DLLExportStorageClass);
            MadeChange = true;
        }

        if (F.hasFnAttribute(DLL_IMPORT))
        {
            F.removeFnAttr(DLL_IMPORT);

            // set DLLExport
            if (isWindowsMSVCEnvironment)
                F.setDLLStorageClass(llvm::GlobalVariable::DLLImportStorageClass);
            MadeChange = true;
        }

        if (MadeChange)
        {
            LLVM_DEBUG(llvm::dbgs() << "\n!! EXPORT Change: " << MadeChange;);
            LLVM_DEBUG(llvm::dbgs() << "\n!! EXPORT Dump After: ...\n" << F << "\n";);
        }

        return MadeChange;
    }

    bool runOnGlobal(llvm::GlobalVariable &G)
    {
        auto MadeChange = false;

        LLVM_DEBUG(llvm::dbgs() << "\nEXPORT Global: " << G.getName());
        LLVM_DEBUG(llvm::dbgs() << "\nEXPORT Dump Before: ...\n" << G << "\n";);

        // if (!G.hasLocalLinkage())
        //     G.setDLLStorageClass(llvm::GlobalVariable::DLLExportStorageClass);
        
        if (G.hasSection())
        {
            // set DLLExport
            if (G.getSection() == "export" || G.getSection() == DLL_EXPORT)
            {
                if (isWindowsMSVCEnvironment && !G.hasDLLExportStorageClass())
                    G.setDLLStorageClass(llvm::GlobalVariable::DLLExportStorageClass);
                G.setSection("");
                MadeChange = true;
            }
            else if (G.getSection() == "import" || G.getSection() == DLL_IMPORT)
            {
                if (isWindowsMSVCEnvironment && !G.hasDLLImportStorageClass())
                    G.setDLLStorageClass(llvm::GlobalVariable::DLLImportStorageClass);
                G.setSection("");
                MadeChange = true;
            }
        }

        if (MadeChange)
        {
            LLVM_DEBUG(llvm::dbgs() << "\n!! EXPORT Change: " << MadeChange;);
            LLVM_DEBUG(llvm::dbgs() << "\n!! EXPORT Dump After: ...\n" << G << "\n";);
        }

        return MadeChange;
    }    
};

namespace ts
{
    llvm::PreservedAnalyses ExportFixPass::run(llvm::Module &M, llvm::ModuleAnalysisManager &MAM)
    {
        ExportFixPassCode LPF{isWindowsMSVCEnvironment};
        bool MadeChange = false;

        for (auto &G : M.globals())
        {
            MadeChange |= LPF.runOnGlobal(G);
        }

        for (auto &F : M)
        {
            MadeChange |= LPF.runOnFunction(F);
        }

        return MadeChange ? llvm::PreservedAnalyses::none() : llvm::PreservedAnalyses::all();
    }
}