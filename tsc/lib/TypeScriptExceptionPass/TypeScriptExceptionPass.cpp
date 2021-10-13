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

struct CatchRegion
{
    CatchRegion() = default;

    LandingPadInst *landingPad;
    llvm::SmallVector<CallBase *> calls;
    CatchPadInst *catchPad;
    CleanupPadInst *cleanupPad;
    StoreInst *store;
    InvokeInst *unwindInfoOp;
    Value *stack;
    bool hasAlloca;
    llvm::Instruction *cxaEndCatch;
    llvm::Instruction *end;

    bool isCatch()
    {
        return landingPad->getNumClauses() > 0 && landingPad->isCatch(0) && !landingPad->isCleanup();
    }

    bool isCleanup()
    {
        if (landingPad && landingPad->isCleanup())
        {
            return true;
        }

        if (landingPad->getNumClauses() == 0)
        {
            landingPad->setCleanup(true);
            return true;
        }

        // BUG: HACK. using filter[] as cleanup landingpad
        if ((landingPad->getNumClauses() > 0 && !landingPad->isCatch(0)))
        {
            // this is filter[], treat it as cleanup
            landingPad->setCleanup(true);
            return true;
        }

        return false;
    }
};

struct TypeScriptExceptionPass : public FunctionPass
{
    llvm::StructType *ThrowInfoType;

    static char ID;
    TypeScriptExceptionPass() : FunctionPass(ID), ThrowInfoType{nullptr}
    {
    }

    bool runOnFunction(Function &F) override
    {
        auto MadeChange = false;

        llvm::SmallVector<CatchRegion> catchRegionsWorkSet;

        LLVM_DEBUG(llvm::dbgs() << "\nFunction: " << F.getName() << "\n\n";);
        LLVM_DEBUG(llvm::dbgs() << "\nDump Before: " << F << "\n\n";);

        CatchRegion *catchRegion = nullptr;
        auto endOfCatch = false;
        auto endOfCatchIfResume = false;
        llvm::SmallVector<llvm::Instruction *> toRemoveWorkSet;
        for (auto &I : instructions(F))
        {
            if (auto *LPI = dyn_cast<LandingPadInst>(&I))
            {
                catchRegionsWorkSet.push_back(CatchRegion());
                catchRegion = &catchRegionsWorkSet.back();

                catchRegion->landingPad = LPI;

                endOfCatch = false;
                continue;
            }

            // it is outsize of catch/finally region
            if (!catchRegion)
            {
                continue;
            }

            if (endOfCatchIfResume && dyn_cast<ResumeInst>(&I))
            {
                endOfCatch = true;
            }

            endOfCatchIfResume = false;

            if (endOfCatch)
            {
                // BR, or instraction without BR
                catchRegion->end = &I;
                endOfCatch = false;
                catchRegion = nullptr;
                continue;
            }

            if (catchRegion->unwindInfoOp == nullptr)
            {
                if (auto *II = dyn_cast<InvokeInst>(&I))
                {
                    catchRegion->unwindInfoOp = II;
                }
            }

            if (!catchRegion->store)
            {
                if (auto *SI = dyn_cast<StoreInst>(&I))
                {
                    assert(!catchRegion->store);
                    catchRegion->store = SI;
                }
            }

            if (dyn_cast<AllocaInst>(&I))
            {
                catchRegion->hasAlloca = true;
            }

            if (auto *CI = dyn_cast<CallInst>(&I))
            {
                LLVM_DEBUG(llvm::dbgs() << "\nCall: " << CI->getCalledFunction()->getName() << "");

                if (CI->getCalledFunction()->getName() == "__cxa_end_catch")
                {
                    toRemoveWorkSet.push_back(&I);
                    catchRegion->cxaEndCatch = &I;
                    endOfCatch = true;
                    continue;
                }

                // possible end
                if (CI->getCalledFunction()->getName() == "_CxxThrowException")
                {
                    // do not put continue, we need to add facelet
                    catchRegion->end = &I;
                }
            }

            if (auto *II = dyn_cast<InvokeInst>(&I))
            {
                LLVM_DEBUG(llvm::dbgs() << "\nInvoke: " << II->getCalledFunction()->getName() << "");

                if (II->getCalledFunction()->getName() == "__cxa_end_catch")
                {
                    toRemoveWorkSet.push_back(&I);
                    catchRegion->cxaEndCatch = &I;
                    catchRegion->end = &I;

                    endOfCatchIfResume = true;

                    continue;
                }

                // possible end
                if (II->getCalledFunction()->getName() == "_CxxThrowException")
                {
                    // do not put continue, we need to add facelet
                    catchRegion->end = &I;
                }
            }

            if (auto *CB = dyn_cast<CallBase>(&I))
            {
                catchRegion->calls.push_back(CB);
                continue;
            }
        }

        // create begin of catch block
        for (auto &catchRegion : catchRegionsWorkSet)
        {
            auto *LPI = catchRegion.landingPad;

            LLVM_DEBUG(llvm::dbgs() << "\nProcessing: " << *LPI << " isKnownSentinel: " << (LPI->isKnownSentinel() ? "true" : "false")
                                    << "\n\n";);

            // add catchswitch & catchpad
            llvm::IRBuilder<> Builder(LPI);
            llvm::LLVMContext &Ctx = Builder.getContext();

            if (catchRegion.isCatch())
            {
                // split
                BasicBlock *CurrentBB = LPI->getParent();
                BasicBlock *ContinuationBB = CurrentBB->splitBasicBlock(LPI->getIterator(), "catch");

                CurrentBB->getTerminator()->eraseFromParent();

                auto *II = catchRegion.unwindInfoOp;
                auto *CSI = CatchSwitchInst::Create(ConstantTokenNone::get(Ctx),
                                                    II ? II->getUnwindDest() : nullptr
                                                    /*unwind to caller if null*/,
                                                    1, "catch.switch", CurrentBB);
                CSI->addHandler(ContinuationBB);

                // check what is type of catch
                auto value = LPI->getOperand(0);
                auto isNullInst = isa<ConstantPointerNull>(value);
                if (isNullInst)
                {
                    // catch (...) as catch value is null
                    auto nullI8Ptr = ConstantPointerNull::get(IntegerType::get(Ctx, 8)->getPointerTo());
                    auto iVal64 = ConstantInt::get(IntegerType::get(Ctx, 32), 64);
                    catchRegion.catchPad = CatchPadInst::Create(CSI, {nullI8Ptr, iVal64, nullI8Ptr}, "catchpad", LPI);
                }
                else
                {
                    auto varRef = catchRegion.store;
                    assert(varRef);
                    auto iValTypeId = ConstantInt::get(IntegerType::get(Ctx, 32), getTypeNumber(varRef->getPointerOperandType()));
                    catchRegion.catchPad = CatchPadInst::Create(CSI, {value, iValTypeId, varRef->getPointerOperand()}, "catchpad", LPI);
                    varRef->eraseFromParent();
                }
            }
            else
            {
                assert(catchRegion.isCleanup());

                catchRegion.cleanupPad = CleanupPadInst::Create(ConstantTokenNone::get(Ctx), None, "cleanuppad", LPI);
            }

            // save stack
            if (catchRegion.hasAlloca)
            {
                catchRegion.stack = Builder.CreateCall(Intrinsic::getDeclaration(F.getParent(), Intrinsic::stacksave), {});
            }

            // set funcset
            for (auto callBase : catchRegion.calls)
            {
                llvm::SmallVector<OperandBundleDef> opBundle;
                if (catchRegion.catchPad)
                {
                    opBundle.emplace_back(OperandBundleDef("funclet", catchRegion.catchPad));
                }
                else if (catchRegion.cleanupPad)
                {
                    opBundle.emplace_back(OperandBundleDef("funclet", catchRegion.cleanupPad));
                }
                else
                {
                    llvm_unreachable("not implemented");
                }

                auto replaceEndData = catchRegion.end == callBase;

                auto *newCallBase = CallBase::Create(callBase, opBundle, callBase);
                callBase->replaceAllUsesWith(newCallBase);
                callBase->eraseFromParent();

                if (replaceEndData)
                {
                    catchRegion.end = newCallBase;
                }
            }

            // LLVM_DEBUG(llvm::dbgs() << "\nLanding Pad - Done. Function Dump: " << F << "\n\n";);

            MadeChange = true;
        }

        // create end of catch block
        for (auto &catchRegion : catchRegionsWorkSet)
        {
            auto *I = catchRegion.end;
            assert(I);
            auto *LPI = catchRegion.landingPad;
            assert(LPI);

            llvm::BasicBlock *retBlock = nullptr;

            llvm::IRBuilder<> Builder(I);
            llvm::LLVMContext &Ctx = Builder.getContext();

            auto *BI = dyn_cast<BranchInst>(I);
            if (BI)
            {
                retBlock = BI->getSuccessor(0);
            }
            else if (auto *II = dyn_cast<InvokeInst>(I))
            {
                retBlock = II->getNormalDest();
            }
            else if (dyn_cast<ResumeInst>(I))
            {
                // nothing todo
                toRemoveWorkSet.push_back(I);
            }
            else
            {
                retBlock = Builder.GetInsertBlock()->splitBasicBlock(I, "end.of.exception");
                BI = dyn_cast<BranchInst>(&retBlock->getPrevNode()->back());
                Builder.SetInsertPoint(BI);
            }

            toRemoveWorkSet.push_back(&*LPI);

            if (catchRegion.hasAlloca)
            {
                assert(catchRegion.stack);
                // restore stack
                Builder.CreateCall(Intrinsic::getDeclaration(F.getParent(), Intrinsic::stackrestore), {catchRegion.stack});
            }

            if (catchRegion.isCatch())
            {
                if (!dyn_cast<InvokeInst>(I))
                {
                    assert(catchRegion.catchPad);
                    auto CR = CatchReturnInst::Create(catchRegion.catchPad, retBlock, BI ? BI->getParent() : I->getParent());
                    if (BI)
                    {
                        // remove BranchInst
                        BI->replaceAllUsesWith(CR);
                        toRemoveWorkSet.push_back(&*BI);
                    }
                }
            }
            else
            {
                // cleanup
                assert(catchRegion.cleanupPad);

                BasicBlock *emptyBlockBefore = nullptr;
                if (I->getPrevNode() == nullptr)
                {
                    emptyBlockBefore = Builder.GetInsertBlock();
                }

                BasicBlock *CurrentBB = I->getParent();
                BasicBlock *ContinuationBB = CurrentBB->splitBasicBlock(CurrentBB->getTerminator(), "catch.pad");
                BasicBlock *CSIBlock =
                    emptyBlockBefore ? emptyBlockBefore : BasicBlock::Create(Ctx, "catch.dispatch", CurrentBB->getParent(), ContinuationBB);

                CurrentBB->getTerminator()->eraseFromParent();

                if (emptyBlockBefore)
                {
                    CurrentBB = emptyBlockBefore->getPrevNode();
                }

                CleanupReturnInst::Create(catchRegion.cleanupPad, CSIBlock, CurrentBB);

                // add rethrow code
                auto *II = catchRegion.unwindInfoOp;
                auto *CSI = CatchSwitchInst::Create(ConstantTokenNone::get(Ctx),
                                                    II ? II->getUnwindDest() : nullptr
                                                    /*unwind to caller if null*/,
                                                    1, "catchswitch", CSIBlock);

                CSI->addHandler(ContinuationBB);

                // catch (...) as catch value is null
                auto nullI8Ptr = ConstantPointerNull::get(IntegerType::get(Ctx, 8)->getPointerTo());
                auto iVal64 = ConstantInt::get(IntegerType::get(Ctx, 32), 64);
                auto *CPI = CatchPadInst::Create(CSI, {nullI8Ptr, iVal64, nullI8Ptr}, "catchpad", ContinuationBB);

                // rethrow
                llvm::SmallVector<OperandBundleDef> opBundle;
                opBundle.emplace_back(OperandBundleDef("funclet", CPI));

                auto throwFunc = getThrowFn(Ctx, F.getParent());

                auto nullTI = ConstantPointerNull::get(cast<llvm::PointerType>(throwFunc.getFunctionType()->params()[1]));

                auto *UI = new UnreachableInst(Ctx, ContinuationBB);

                Builder.SetInsertPoint(UI);

                Builder.CreateCall(throwFunc, {nullI8Ptr, nullTI}, opBundle);

                // end
            }

            // LLVM_DEBUG(llvm::dbgs() << "\nTerminator after: " << *RI->getParent()->getTerminator() << "\n\n";);
            // LLVM_DEBUG(llvm::dbgs() << "\nResume - Done. Function Dump: " << F << "\n\n";);

            MadeChange = true;
        }

        LLVM_DEBUG(llvm::dbgs() << "\nDump Before deleting: " << F << "\n\n";);

        // remove
        for (auto CI : toRemoveWorkSet)
        {
            CI->eraseFromParent();
        }

        // LLVM_DEBUG(llvm::dbgs() << "\nDone. Function Dump: " << F << "\n\n";);

        LLVM_DEBUG(llvm::dbgs() << "\nChange: " << MadeChange << "\n\n";);
        LLVM_DEBUG(llvm::dbgs() << "\nDump After: " << F << "\n\n";);

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

    llvm::StructType *getThrowInfoType(LLVMContext &Ctx)
    {
        if (ThrowInfoType)
        {
            return ThrowInfoType;
        }

        llvm::Type *FieldTypes[] = {
            IntegerType::get(Ctx, 32),                // Flags
            IntegerType::get(Ctx, 8)->getPointerTo(), // CleanupFn
            IntegerType::get(Ctx, 8)->getPointerTo(), // ForwardCompat
            IntegerType::get(Ctx, 8)->getPointerTo()  // CatchableTypeArray
        };
        ThrowInfoType = llvm::StructType::create(Ctx, FieldTypes, "eh.ThrowInfo");
        return ThrowInfoType;
    }

    llvm::FunctionCallee getThrowFn(LLVMContext &Ctx, llvm::Module *module)
    {
        auto globalFunc = module->getNamedValue("_CxxThrowException");
        if (globalFunc)
        {
            return cast<llvm::Function>(globalFunc);
        }

        // _CxxThrowException is passed an exception object and a ThrowInfo object
        // which describes the exception.
        llvm::Type *Args[] = {IntegerType::get(Ctx, 8)->getPointerTo(), getThrowInfoType(Ctx)->getPointerTo()};
        auto *FTy = llvm::FunctionType::get(Type::getVoidTy(Ctx), Args, /*isVarArg=*/false);
        auto Throw = Function::Create(FTy, llvm::GlobalValue::LinkageTypes::InternalLinkage, "_CxxThrowException");
        /*
        // _CxxThrowException is stdcall on 32-bit x86 platforms.
        if (CGM.getTarget().getTriple().getArch() == llvm::Triple::x86)
        {
            if (auto *Fn = dyn_cast<llvm::Function>(Throw.getCallee()))
                Fn->setCallingConv(llvm::CallingConv::X86_StdCall);
        }
        */

        return Throw;
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