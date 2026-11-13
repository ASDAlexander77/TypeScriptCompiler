#include "TypeScript/Pass/CxxThrowCallingConvPass.h"
#include "TypeScript/Defines.h"
#include "TypeScript/LowerToLLVM/LLVMRTTIHelperVCWin32Const.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstrTypes.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/ErrorHandling.h"

using namespace llvm;

#define DEBUG_TYPE "pass"

namespace ts
{
    // `@stdcall declare function` (STDCALL_ATTR_NAME): the convention on the declaration - which also
    // gives its symbol the @N suffix the import library names - and on every direct call of it. Any other
    // use, the function taken as a value, is called as any function value is, the C way: it gets a C
    // function that calls this one instead. A variadic function is the C convention, as in MSVC.
    static bool applyDeclaredStdcall(Module &M)
    {
        SmallVector<Function *> declared;
        for (auto &F : M)
        {
            if (F.hasFnAttribute(STDCALL_ATTR_NAME))
            {
                declared.push_back(&F);
            }
        }

        for (auto *F : declared)
        {
            F->removeFnAttr(STDCALL_ATTR_NAME);
            if (F->isVarArg())
            {
                continue;
            }

            F->setCallingConv(CallingConv::X86_StdCall);

            auto isCallOfIt = [](Use &use) {
                auto *call = dyn_cast<CallBase>(use.getUser());
                return call && call->isCallee(&use);
            };

            auto takenAsValue = false;
            for (auto &use : F->uses())
            {
                if (auto *call = dyn_cast<CallBase>(use.getUser()); call && call->isCallee(&use))
                {
                    call->setCallingConv(CallingConv::X86_StdCall);
                }
                else
                {
                    takenAsValue = true;
                }
            }

            if (!takenAsValue)
            {
                continue;
            }

            auto *callee = Function::Create(F->getFunctionType(), GlobalValue::InternalLinkage, F->getName() + ".cdecl", M);
            callee->setAttributes(F->getAttributes());
            F->replaceUsesWithIf(callee, [&](Use &use) { return !isCallOfIt(use); });

            IRBuilder<> builder(BasicBlock::Create(M.getContext(), "", callee));
            SmallVector<Value *> args;
            for (auto &arg : callee->args())
            {
                args.push_back(&arg);
            }

            auto *call = builder.CreateCall(F, args);
            call->setCallingConv(CallingConv::X86_StdCall);
            call->setAttributes(F->getAttributes());
            if (call->getType()->isVoidTy())
            {
                builder.CreateRetVoid();
            }
            else
            {
                builder.CreateRet(call);
            }
        }

        return !declared.empty();
    }

    llvm::PreservedAnalyses CxxThrowCallingConvPass::run(llvm::Module &M, llvm::ModuleAnalysisManager &AM)
    {
        auto changed = applyDeclaredStdcall(M);

        // The CRT calls a CatchableType's copy function as a member function - `__thiscall` on
        // x86: the catch variable in ECX, the thrown object on the stack. Nothing in the module
        // calls these thunks; the EH tables only hold their addresses.
        for (auto &F : M)
        {
            if (F.getName().starts_with(typescript::windows::copyThunkPrefix))
            {
                F.setCallingConv(CallingConv::X86_ThisCall);
                changed = true;
            }
        }

        auto *throwFn = M.getFunction("_CxxThrowException");
        if (!throwFn)
        {
            return changed ? llvm::PreservedAnalyses::none() : llvm::PreservedAnalyses::all();
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
