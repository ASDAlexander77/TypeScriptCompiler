#include "TypeScript/TypeScriptExceptionPass.h"

#include "TypeScript/LowerToLLVM/LLVMRTTIHelperVCWin32Const.h"

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

        LLVM_DEBUG(llvm::dbgs() << "\nFunction: " << F.getName() << "\n\n";);

        llvm::SmallVector<CallBase *> *currentCalls = nullptr;
        for (auto &I : instructions(F))
        {
            if (auto *LPI = dyn_cast<LandingPadInst>(&I))
            {
                landingPadInstWorkSet.push_back(LPI);
                currentCalls = new llvm::SmallVector<CallBase *>();
                calls[LPI] = currentCalls;
                continue;
            }

            if (auto *RI = dyn_cast<ResumeInst>(&I))
            {
                currentCalls = nullptr;
                resumeInstWorkSet.push_back(RI);
                continue;
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

            auto *CSI = CatchSwitchInst::Create(ConstantTokenNone::get(Ctx), nullptr /*unwind to caller*/, 1, "catch.switch", CurrentBB);
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
                    auto type = value->getType();
                    if (type->isPointerTy())
                    {
                        type = type->getPointerElementType();
                    }

                    if (type->isIntegerTy())
                    {
                        auto nullI8Ptr = ConstantPointerNull::get(PointerType::get(IntegerType::get(Ctx, 8), 0));
                        auto iVal0 = ConstantInt::get(IntegerType::get(Ctx, 32), 0);
                        auto foundIt = llvm::find_if(F.getParent()->getGlobalList(),
                                                     [&](auto &item) { return item.getName() == typescript::I32Type::typeInfoRef; });
                        if (foundIt != F.getParent()->getGlobalList().end())
                        {
                            auto &globalValue = *foundIt;

                            Constant *zero_32 = Constant::getNullValue(IntegerType::getInt32Ty(Ctx));
                            Constant *gep_params[] = {zero_32};

                            Constant *throwInfoPtr = ConstantExpr::getGetElementPtr(globalValue.getValueType(), &globalValue, gep_params);
                            CPI = CatchPadInst::Create(CSI, {throwInfoPtr, iVal0, nullI8Ptr}, "catchpad", LPI);
                        }
                        else
                        {
                            llvm_unreachable("not implemented, can't find ??_R0H@8");
                        }
                    }
                    else
                    {
                        llvm_unreachable("not implemented");
                    }
                }
            }
            else
            {
                llvm_unreachable("not implemented");
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

            auto CR = CatchReturnInst::Create(landingPadNewOps[(llvm::LandingPadInst *)RI->getOperand(0)], RI->getParent()->getNextNode(),
                                              RI->getParent());

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