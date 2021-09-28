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

        llvm::SmallVector<LandingPadInst *> landingPadInstWorkSet;
        llvm::SmallVector<ResumeInst *> resumeInstWorkSet;
        llvm::SmallDenseMap<LandingPadInst *, llvm::SmallVector<CallBase *> *> calls;
        llvm::SmallDenseMap<LandingPadInst *, CatchPadInst *> landingPadNewOps;
        llvm::SmallDenseMap<LandingPadInst *, StoreInst *> landingPadStoreOps;
        llvm::SmallDenseMap<LandingPadInst *, InvokeInst *> landingPadUnwindOps;
        llvm::SmallDenseMap<LandingPadInst *, Value *> landingPadStack;
        llvm::SmallDenseMap<LandingPadInst *, bool> landingPadHasAlloca;

        LLVM_DEBUG(llvm::dbgs() << "\nFunction: " << F.getName() << "\n\n";);

        llvm::SmallVector<CallBase *> *currentCalls = nullptr;
        LandingPadInst *currentLPI = nullptr;
        for (auto &I : instructions(F))
        {
            if (auto *LPI = dyn_cast<LandingPadInst>(&I))
            {
                landingPadInstWorkSet.push_back(LPI);
                currentLPI = LPI;
                currentCalls = new llvm::SmallVector<CallBase *>();
                calls[LPI] = currentCalls;
                continue;
            }

            if (auto *RI = dyn_cast<ResumeInst>(&I))
            {
                currentCalls = nullptr;
                currentLPI = nullptr;
                resumeInstWorkSet.push_back(RI);
                continue;
            }

            // saving StoreInst to set response
            if (currentLPI)
            {
                if (auto *II = dyn_cast<InvokeInst>(&I))
                {
                    if (!landingPadUnwindOps[currentLPI])
                    {
                        landingPadUnwindOps[currentLPI] = II;
                    }
                }

                if (auto *SI = dyn_cast<StoreInst>(&I))
                {
                    if (!landingPadStoreOps[currentLPI])
                    {
                        landingPadStoreOps[currentLPI] = SI;
                    }
                }

                if (auto *AI = dyn_cast<AllocaInst>(&I))
                {
                    landingPadHasAlloca[currentLPI] = true;
                }
            }

            if (currentCalls)
            {
                if (auto *CB = dyn_cast<CallBase>(&I))
                {
                    currentCalls->push_back(CB);
                    continue;
                }
            }
        }

        llvm::SmallVector<LandingPadInst *> toRemoveLandingPad;
        llvm::SmallVector<ResumeInst *> toRemoveResumeInstWorkSet;

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

            auto *II = landingPadUnwindOps[LPI];
            auto *CSI = CatchSwitchInst::Create(ConstantTokenNone::get(Ctx),
                                                II ? II->getUnwindDest() : nullptr
                                                /*unwind to caller if null*/,
                                                1, "catch.switch", CurrentBB);
            CSI->addHandler(ContinuationBB);

            CatchPadInst *CPI = nullptr;
            if (LPI->getNumClauses() > 0 && LPI->isCatch(0))
            {
                // check what is type of catch
                auto value = LPI->getOperand(0);
                auto isNullInst = isa<ConstantPointerNull>(value);
                if (isNullInst)
                {
                    // catch (...) as catch value is null
                    auto nullI8Ptr = ConstantPointerNull::get(PointerType::get(IntegerType::get(Ctx, 8), 0));
                    auto iVal64 = ConstantInt::get(IntegerType::get(Ctx, 32), 64);
                    CPI = CatchPadInst::Create(CSI, {nullI8Ptr, iVal64, nullI8Ptr}, "catchpad", LPI);
                }
                else
                {
                    auto varRef = landingPadStoreOps[LPI];
                    assert(varRef);
                    auto iValTypeId = ConstantInt::get(IntegerType::get(Ctx, 32), getTypeNumber(varRef->getPointerOperandType()));
                    CPI = CatchPadInst::Create(CSI, {value, iValTypeId, varRef->getPointerOperand()}, "catchpad", LPI);
                    varRef->eraseFromParent();
                }
            }
            else
            {
                llvm_unreachable("not implemented");
            }

            // save stack
            Value *SP = nullptr;
            auto hasAlloca = landingPadHasAlloca[LPI];
            if (hasAlloca)
            {
                SP = Builder.CreateCall(Intrinsic::getDeclaration(F.getParent(), Intrinsic::stacksave), {});
                landingPadStack[LPI] = SP;
            }

            toRemoveLandingPad.push_back(LPI);
            landingPadNewOps[LPI] = CPI;

            // set funcset
            llvm::SmallVector<CallBase *> *callsByLandingPad = calls[LPI];
            if (callsByLandingPad)
            {
                for (auto callBase : *callsByLandingPad)
                {
                    llvm::SmallVector<OperandBundleDef> opBundle;
                    opBundle.emplace_back(OperandBundleDef("funclet", CPI));
                    auto *newCallBase = CallBase::Create(callBase, opBundle, callBase);
                    callBase->replaceAllUsesWith(newCallBase);
                    callBase->eraseFromParent();
                }
            }

            // LLVM_DEBUG(llvm::dbgs() << "\nLanding Pad - Done. Function Dump: " << F << "\n\n";);

            MadeChange = true;
        }

        for (auto *RI : resumeInstWorkSet)
        {
            llvm::IRBuilder<> Builder(RI);
            // auto *UI = new UnreachableInst(Ctx, RI->getParent());

            auto *LPI = (llvm::LandingPadInst *)RI->getOperand(0);

            auto hasAlloca = landingPadHasAlloca[LPI];
            if (hasAlloca)
            {
                assert(landingPadStack[LPI]);
                // restore stack
                Builder.CreateCall(Intrinsic::getDeclaration(F.getParent(), Intrinsic::stackrestore), {landingPadStack[LPI]});
            }

            assert(landingPadNewOps[LPI]);

            auto CR = CatchReturnInst::Create(landingPadNewOps[LPI], RI->getParent()->getNextNode(), RI->getParent());

            RI->replaceAllUsesWith(CR);

            toRemoveResumeInstWorkSet.push_back(RI);

            // LLVM_DEBUG(llvm::dbgs() << "\nTerminator after: " << *RI->getParent()->getTerminator() << "\n\n";);
            // LLVM_DEBUG(llvm::dbgs() << "\nResume - Done. Function Dump: " << F << "\n\n";);

            MadeChange = true;
        }

        // remove
        for (auto RI : toRemoveResumeInstWorkSet)
        {
            RI->eraseFromParent();
        }

        for (auto LPI : toRemoveLandingPad)
        {
            LPI->eraseFromParent();
        }

        // LLVM_DEBUG(llvm::dbgs() << "\nDone. Function Dump: " << F << "\n\n";);

        // cleaups
        for (auto p : calls)
        {
            delete p.second;
        }

        return MadeChange;
    }

    int getTypeNumber(Type *catchValType)
    {
        if (catchValType->isIntegerTy() || catchValType->isFloatTy())
        {
            return 0;
        }

        // default if char*, class etc
        return 1;
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