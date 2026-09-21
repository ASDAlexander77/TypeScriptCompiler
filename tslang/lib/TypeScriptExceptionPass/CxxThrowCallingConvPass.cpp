#include "TypeScript/Pass/CxxThrowCallingConvPass.h"

#include "llvm/IR/InstrTypes.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/ErrorHandling.h"

using namespace llvm;

#define DEBUG_TYPE "pass"

namespace ts
{
    llvm::PreservedAnalyses CxxThrowCallingConvPass::run(llvm::Module &M, llvm::ModuleAnalysisManager &AM)
    {
        auto *throwFn = M.getFunction("_CxxThrowException");
        if (!throwFn)
        {
            return llvm::PreservedAnalyses::all();
        }

        throwFn->setCallingConv(CallingConv::X86_StdCall);
        for (auto *user : throwFn->users())
        {
            auto *call = dyn_cast<CallBase>(user);
            if (!call || call->getCalledOperand() != throwFn)
            {
                // Only direct calls and invokes are expected. Any other use (the function taken as
                // a value) could reach an indirect call this pass cannot give the convention to.
                report_fatal_error("_CxxThrowException is used other than as a direct callee; "
                                   "its x86 stdcall convention cannot be applied to that use");
            }

            call->setCallingConv(CallingConv::X86_StdCall);
        }

        return llvm::PreservedAnalyses::none();
    }
}
