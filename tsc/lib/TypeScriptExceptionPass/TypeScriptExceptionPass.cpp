#include "TypeScript/TypeScriptExceptionPass.h"

#include "llvm/Analysis/DomTreeUpdater.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/PassManager.h"
#include "llvm/IR/PatternMatch.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/InitializePasses.h"
#include "llvm/Support/CommandLine.h"

#include "llvm/ADT/SmallPtrSet.h"

using namespace llvm;
using namespace PatternMatch;

#define DEBUG_TYPE "pass"

namespace
{
struct TypeScriptExceptionPass : public FunctionPass
{
    static char ID;
    TypeScriptExceptionPass() : FunctionPass(ID)
    {
    }

    bool runOnFunction(Function &F) override
    {
        auto MadeChange = false;

        llvm::SmallPtrSet<LandingPadInst *, 16> landingPadInstWorkSet;
        llvm::SmallPtrSet<ResumeInst *, 16> resumeInstWorkSet;

        LLVM_DEBUG(llvm::dbgs() << "\nFunction: " << F.getName() << "\n\n";);

        auto &DL = F.getParent()->getDataLayout();

        for (auto &I : instructions(F))
        {
            if (auto *LPI = dyn_cast<LandingPadInst>(&I))
            {
                landingPadInstWorkSet.insert(LPI);
                continue;
            }

            if (auto *RI = dyn_cast<ResumeInst>(&I))
            {
                resumeInstWorkSet.insert(RI);
                continue;
            }
        }

        for (auto *LPI : landingPadInstWorkSet)
        {
            LLVM_DEBUG(llvm::dbgs() << "\nProcessing: " << *LPI << " isKnownSentinel: " << (LPI->isKnownSentinel() ? "true" : "false")
                                    << "\n\n";);

            // add catchswitch & catchpad
            llvm::IRBuilder<> Builder(LPI);
            llvm::LLVMContext &Ctx = Builder.getContext();

            // split
            BasicBlock *CurrentBB = LPI->getParent();
            BasicBlock *ContinuationBB = CurrentBB->splitBasicBlock(LPI->getIterator(), "catch");

            CurrentBB->getTerminator()->eraseFromParent();

            auto *CSI = CatchSwitchInst::Create(LPI, nullptr /*unwind to caller*/, 1, "catch.switch", CurrentBB);
            CSI->addHandler(ContinuationBB);

            auto nullI8Ptr = ConstantPointerNull::get(PointerType::get(IntegerType::get(Ctx, 8), 0));
            auto iVal64 = ConstantInt::get(IntegerType::get(Ctx, 32), 32);

            auto *CPI = CatchPadInst::Create(CSI, {nullI8Ptr, iVal64, nullI8Ptr}, "catchpad", LPI);

            LPI->replaceAllUsesWith(CPI);
            LPI->eraseFromParent();

            // LLVM_DEBUG(llvm::dbgs() << "\nLanding Pad - Done. Function Dump: " << F << "\n\n";);

            MadeChange = true;
        }

        for (auto *RI : resumeInstWorkSet)
        {
            llvm::IRBuilder<> Builder(RI);
            llvm::LLVMContext &Ctx = Builder.getContext();

            // LLVM_DEBUG(llvm::dbgs() << "\nTerminator before: " << *RI->getParent()->getTerminator() << "\n\n";);
            // auto *UI = new UnreachableInst(Ctx, RI->getParent());

            auto CR = CatchReturnInst::Create(RI->getOperand(0), RI->getParent()->getNextNode(), RI->getParent());

            RI->replaceAllUsesWith(CR);
            RI->eraseFromParent();

            // LLVM_DEBUG(llvm::dbgs() << "\nTerminator after: " << *RI->getParent()->getTerminator() << "\n\n";);
            // LLVM_DEBUG(llvm::dbgs() << "\nResume - Done. Function Dump: " << F << "\n\n";);

            MadeChange = true;
        }

        LLVM_DEBUG(llvm::dbgs() << "\nDone. Function Dump: " << F << "\n\n";);

        return MadeChange;
    }
};
} // namespace

char TypeScriptExceptionPass::ID = 0;

#define CONFIG false
#define ANALYSIS false

INITIALIZE_PASS(TypeScriptExceptionPass, DEBUG_TYPE, TYPESCRIPT_EXCEPTION_PASS_NAME, CONFIG, ANALYSIS)

static RegisterPass<TypeScriptExceptionPass> X(DEBUG_TYPE, TYPESCRIPT_EXCEPTION_PASS_NAME, CONFIG, ANALYSIS);

const void *llvm::getTypeScriptExceptionPassID()
{
    return &TypeScriptExceptionPass::ID;
}