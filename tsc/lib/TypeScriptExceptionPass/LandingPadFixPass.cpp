#include "TypeScript/LandingPadFixPass.h"

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

struct LandingPadFixPass : public FunctionPass
{
    static char ID;
    LandingPadFixPass() : FunctionPass(ID)
    {
    }

    bool runOnFunction(Function &F) override
    {
        auto MadeChange = false;

        LLVM_DEBUG(llvm::dbgs() << "\nLANDFIX Function: " << F.getName());
        LLVM_DEBUG(llvm::dbgs() << "\nLANDFIX Dump Before: ...\n" << F << "\n";);

        llvm::SmallVector<llvm::LandingPadInst *> workSet;
        for (auto &I : instructions(F))
        {
            if (auto *LPI = dyn_cast<LandingPadInst>(&I))
            {
                workSet.push_back(LPI);
                continue;
            }
        }

        // create begin of catch block
        llvm::IRBuilder<> Builder(F.getContext());

        for (auto &LPI : workSet)
        {
            // auto hasFilter = LPI->getNumClauses() == 1 && LPI->isFilter(0);
            auto hasFilter = false;
            for (unsigned int i = 0; i < LPI->getNumClauses(); i++)
            {
                hasFilter |= LPI->isFilter(i);
            }

            if (hasFilter)
            {
                Builder.SetInsertPoint(LPI);
                auto newLandingPad = Builder.CreateLandingPad(LPI->getType(), 0);
                newLandingPad->setCleanup(true);
                LPI->replaceAllUsesWith(newLandingPad);
                LPI->eraseFromParent();
            }

            MadeChange = true;
        }

        LLVM_DEBUG(llvm::dbgs() << "\n!! LANDFIX Change: " << MadeChange;);
        LLVM_DEBUG(llvm::dbgs() << "\n!! LANDFIX Dump After: ...\n" << F << "\n";);

        return MadeChange;
    }
};
} // namespace

char LandingPadFixPass::ID = 0;

#define CONFIG false
#define ANALYSIS false

INITIALIZE_PASS(LandingPadFixPass, DEBUG_TYPE, LANDINGPAD_FIX_PASS_NAME, CONFIG, ANALYSIS)

static RegisterPass<LandingPadFixPass> X(DEBUG_TYPE, LANDINGPAD_FIX_PASS_NAME, CONFIG, ANALYSIS);

const void *llvm::getLandingPadFixPassID()
{
    return &LandingPadFixPass::ID;
}