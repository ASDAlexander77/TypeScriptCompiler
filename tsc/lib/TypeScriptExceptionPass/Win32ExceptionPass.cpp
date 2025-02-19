#include "TypeScript/Win32ExceptionPass.h"

#include "llvm/IR/PatternMatch.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/IRBuilder.h"

#include "llvm/ADT/PostOrderIterator.h"

using namespace llvm;
using namespace PatternMatch;

#define DEBUG_TYPE "pass"

//#define SAVE_STACK true

struct CatchRegion
{
    CatchRegion() = default;

    LandingPadInst *landingPad;
    llvm::SmallVector<CallBase *> calls;
    CatchPadInst *catchPad;
    CleanupPadInst *cleanupPad;
    InvokeInst *unwindInfoOp;
    Value *stack;
    bool hasAlloca;
    llvm::Instruction *cxaBeginCatch;
    llvm::Instruction *saveCatch;
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

struct Win32ExceptionPassCode
{
    llvm::StructType *ThrowInfoType;

    Win32ExceptionPassCode() : ThrowInfoType{nullptr}
    {
    }

    bool runOnFunction(Function &F)
    {
        auto MadeChange = false;

        llvm::SmallVector<CatchRegion> catchRegionsWorkSet;

        LLVM_DEBUG(llvm::dbgs() << "\n!! Function: " << F.getName(););
        LLVM_DEBUG(llvm::dbgs() << "\n!! Dump Before: ...\n" << F << "\n";);

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
                    LLVM_DEBUG(llvm::dbgs() << "\n!! set (unwindInfoOp) : " << *II << "\n";);
                    catchRegion->unwindInfoOp = II;
                }
            }

            if (dyn_cast<AllocaInst>(&I))
            {
                catchRegion->hasAlloca = true;
            }

            if (auto *CI = dyn_cast<CallInst>(&I))
            {
                if (CI->getCalledFunction() != nullptr && CI->getCalledFunction()->hasName())
                {
                    LLVM_DEBUG(llvm::dbgs() << "\n!! Call: " << CI->getCalledFunction()->getName() << "");

                    if (CI->getCalledFunction()->getName() == "__cxa_begin_catch")
                    {
                        LLVM_DEBUG(llvm::dbgs() << "\n!! __cxa_begin_catch : " << *CI << "\n";);
                        LLVM_DEBUG(llvm::dbgs() << "\n!! __cxa_begin_catch op 0 : " << *CI->getOperand(0) << "\n";);

                        toRemoveWorkSet.push_back(&I);
                        auto extractOp = cast<llvm::ExtractValueInst>(CI->getOperand(0));
                        toRemoveWorkSet.push_back(extractOp);
                        catchRegion->cxaBeginCatch = &I;
                        continue;
                    }

                    if (CI->getCalledFunction()->getName() == "ts.internal.save_catch_var")
                    {
                        catchRegion->saveCatch = &I;
                        continue;
                    }

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
                    else
                    {
                        LLVM_DEBUG(llvm::dbgs() << "\n!! WARNING Must be Invoke: " << I << "\n");

                        // TODO: uncomment me
                        /*
                        llvm_unreachable("CallInst must not be used in Try/Catch/Finally block as it will cause issue with incorrect unwind "
                                        "destination when Inliner inlines body of method");
                        */
                    }
                }
            }

            if (auto *II = dyn_cast<InvokeInst>(&I))
            {
                if (II->getCalledFunction() != nullptr && II->getCalledFunction()->hasName())
                {
                    LLVM_DEBUG(llvm::dbgs() << "\n!! Invoke: " << II->getCalledFunction()->getName() << "");

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

            LLVM_DEBUG(llvm::dbgs() << "\n!! Processing: " << *LPI << " isKnownSentinel: " << (LPI->isKnownSentinel() ? "true" : "false")
                                    << "\n";);

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
                if (isNullInst || !catchRegion.saveCatch)
                {
                    // catch (...) as catch value is null
                    auto nullI8Ptr = ConstantPointerNull::get(IntegerType::get(Ctx, 8)->getPointerTo());
                    auto iVal64 = ConstantInt::get(IntegerType::get(Ctx, 32), 64);
                    catchRegion.catchPad = CatchPadInst::Create(CSI, {nullI8Ptr, iVal64, nullI8Ptr}, "catchpad", LPI);
                }
                else
                {
                    auto varRef = catchRegion.saveCatch->getOperand(1)->stripPointerCasts();
                    assert(varRef);
                    auto iValTypeId = ConstantInt::get(IntegerType::get(Ctx, 32), getTypeNumber(varRef->getType()));
                    catchRegion.catchPad = CatchPadInst::Create(CSI, {value, iValTypeId, varRef}, "catchpad", LPI);
                    catchRegion.saveCatch->eraseFromParent();
                }
            }
            else
            {
                assert(catchRegion.isCleanup());

                catchRegion.cleanupPad = CleanupPadInst::Create(ConstantTokenNone::get(Ctx), std::nullopt, "cleanuppad", LPI);
            }

            auto opBundle = getCallBundleFromCatchRegion(catchRegion);

#ifdef SAVE_STACK            
            // save stack
            if (catchRegion.hasAlloca)
            {
                // TODO: it seems I don't need opBundle here
                auto stackSaveFuncCallee = Intrinsic::getDeclaration(F.getParent(), Intrinsic::stacksave);
                auto callInst = Builder.CreateCall(stackSaveFuncCallee->getFunctionType(), stackSaveFuncCallee, {}, opBundle);
                catchRegion.stack = callInst;
            }
#endif

            // set funcset
            llvm::SmallVector<CallBase *> newCalls;
            for (auto callBase : catchRegion.calls)
            {
                auto replaceEndData = catchRegion.end == callBase;
                auto replaceUnwindInfoOp = catchRegion.unwindInfoOp == callBase;

                CallBase *newCallBase = nullptr;
                /*
                if (catchRegion.unwindInfoOp)
                {
                    if (isa<CallInst>(callBase))
                    {
                        if (auto *CI = cast<CallInst>(callBase))
                        {
                            newCallBase = ToInvoke(CI, catchRegion.unwindInfoOp->getUnwindDest(), opBundle);
                        }
                    }
                }
                */

                // default case
                if (!newCallBase)
                {
                    newCallBase = CallBase::Create(callBase, opBundle, callBase);
                }

                callBase->replaceAllUsesWith(newCallBase);
                callBase->eraseFromParent();

                newCalls.push_back(newCallBase);

                if (replaceEndData)
                {
                    catchRegion.end = newCallBase;
                }

                if (replaceUnwindInfoOp)
                {
                    catchRegion.unwindInfoOp = cast<InvokeInst>(newCallBase);
                }
            }

            catchRegion.calls.clear();
            catchRegion.calls.append(newCalls);

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

#ifdef SAVE_STACK
            if (catchRegion.hasAlloca)
            {
                // TODO: if we already have stackrestore before we do not need this stack restore, it will cause the issue for optimization and thus will not be compiled
                // TODO: it seems I don't need opBundle here

                assert(catchRegion.stack);

                // restore stack
                auto opBundle = getCallBundleFromCatchRegion(catchRegion);
                auto stackRestoreFuncCallee = Intrinsic::getDeclaration(F.getParent(), Intrinsic::stackrestore);
                Builder.CreateCall(stackRestoreFuncCallee->getFunctionType(), stackRestoreFuncCallee, {catchRegion.stack}, opBundle);
            }
#endif            

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

                // convert call to Invoke with the same unwind
                /*
                llvm::SmallVector<OperandBundleDef> opBundle;
                opBundle.emplace_back(OperandBundleDef("funclet", catchRegion.catchPad));
                for (auto callBase : catchRegion.calls)
                {
                    if (isa<CallInst>(callBase))
                    {
                        if (auto *CI = cast<CallInst>(callBase))
                        {
                            LLVM_DEBUG(llvm::dbgs() << "\n!! CONVERT CALL TO INVOKE(catchpad): " << *callBase << "\n");

                            auto newCallBase = ToInvoke(CI, retBlock, opBundle);
                            callBase->replaceAllUsesWith(newCallBase);
                            callBase->eraseFromParent();
                        }
                    }
                }
                */
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

                // TODO: we need to find block with unreachable first
                if (auto *unreachBlock = findUnreachableBlock(F))
                {
                    // TODO: I don't know why but we need it here
                    unreachBlock->setName("unreachable");
                    auto *BI = BranchInst::Create(unreachBlock, ContinuationBB);
                    Builder.SetInsertPoint(BI);
                }
                else
                {
                    auto *UI = new UnreachableInst(Ctx, ContinuationBB);
                    Builder.SetInsertPoint(UI);
                }

                Builder.CreateCall(throwFunc, {nullI8Ptr, nullTI}, opBundle);

                // end

                // convert call to Invoke with the same unwind
                /*
                llvm::SmallVector<OperandBundleDef> opBundleCleanup;
                opBundleCleanup.emplace_back(OperandBundleDef("funclet", catchRegion.cleanupPad));
                for (auto callBase : catchRegion.calls)
                {
                    if (isa<CallInst>(callBase))
                    {
                        if (auto *CI = cast<CallInst>(callBase))
                        {
                            LLVM_DEBUG(llvm::dbgs() << "\n!! CONVERT CALL TO INVOKE(cleanup): " << *callBase << "\n");

                            auto newCallBase = ToInvoke(CI, CSIBlock, opBundleCleanup);
                            callBase->replaceAllUsesWith(newCallBase);
                            callBase->eraseFromParent();
                        }
                    }
                }
                */

                // fix incorrect landing pad
                // inlineing messing up with landingpad, it may create landing pad with filter and catch clauses, so we are fixing
                // consequences
                llvm::SmallVector<OperandBundleDef> opBundleCleanup;
                opBundleCleanup.emplace_back(OperandBundleDef("funclet", catchRegion.cleanupPad));
                for (auto callBase : catchRegion.calls)
                {
                    if (isa<InvokeInst>(callBase))
                    {
                        if (auto *II = cast<InvokeInst>(callBase))
                        {
                            if (II->getUnwindDest() != CSIBlock)
                            {
                                LLVM_DEBUG(llvm::dbgs() << "\n!! FIX INVOKE(cleanup): " << *callBase << "\n");

                                auto newCallBase = ToInvoke(callBase, CSIBlock, opBundleCleanup);
                                callBase->replaceAllUsesWith(newCallBase);
                                callBase->eraseFromParent();
                            }
                        }
                    }
                }
            }

            // LLVM_DEBUG(llvm::dbgs() << "\nTerminator after: " << *RI->getParent()->getTerminator() << "\n\n";);
            // LLVM_DEBUG(llvm::dbgs() << "\nResume - Done. Function Dump: " << F << "\n\n";);

            MadeChange = true;
        }

        LLVM_DEBUG(llvm::dbgs() << "\n!! Dump Before cleanup process: ...\n" << F << "\n\n";);

        // remove
        for (auto CI : toRemoveWorkSet)
        {
            // TODO: we need to fix issue wit PHI node after inline works
            if (CI->getNumUses() > 0)
            {
                for (auto &U : CI->uses())
                {
                    if (U.getUser() && isa<PHINode>(U.getUser()))
                    {
                        // Instruction *UserI = cast<Instruction>(U.getUser());
                        PHINode *UserPHI = cast<PHINode>(U.getUser());
                        if (UserPHI)
                        {
                            UserPHI->eraseFromParent();
                            break;
                        }
                    }
                }
            }

            CI->eraseFromParent();
        }

        cleanupEmptyBlocksWithoutPredecessors(F);

        // LLVM_DEBUG(llvm::dbgs() << "\nDone. Function Dump: " << F << "\n\n";);

        LLVM_DEBUG(llvm::dbgs() << "\n!! Change: " << MadeChange;);
        LLVM_DEBUG(llvm::dbgs() << "\n!! Dump After: ...\n" << F << "\n\n";);

        return MadeChange;
    }

    llvm::SmallVector<OperandBundleDef> getCallBundleFromCatchRegion(CatchRegion &catchRegion)
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

        return opBundle;
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

    InvokeInst *ToInvoke(CallBase *CB, BasicBlock *unwind, llvm::SmallVector<OperandBundleDef> &opBundle)
    {
        BasicBlock *CurrentBB = CB->getParent();
        BasicBlock *ContinuationBB = CurrentBB->splitBasicBlock(CB->getIterator(), "invoke.cont");

        CurrentBB->getTerminator()->eraseFromParent();

        SmallVector<Value *> args;
        for (auto &arg : CB->args())
        {
            args.push_back(CB->getArgOperand(arg.getOperandNo()));
        }

        auto newInvoke =
            InvokeInst::Create(CB->getFunctionType(), CB->getCalledOperand(), ContinuationBB, unwind, args, opBundle, "invoke", CurrentBB);

        return newInvoke;
    }

    void cleanupEmptyBlocksWithoutPredecessors(Function &F)
    {
        auto any = false;
        do
        {
            any = false;
            SmallPtrSet<BasicBlock *, 16> workSet;
            for (auto &regionBlock : F)
            {

                if (regionBlock.isEntryBlock())
                {
                    continue;
                }

                if (regionBlock.hasNPredecessors(0))
                {
                    auto count = std::distance(regionBlock.begin(), regionBlock.end());
                    if (count == 0 || (count == 1 && (isa<BranchInst>(regionBlock.begin()) || isa<UnreachableInst>(regionBlock.begin()))))
                    {
                        LLVM_DEBUG(llvm::dbgs() << "\n!! REMOVING EMPTY BLOCK: ..." << regionBlock << "\n";);
                        workSet.insert(&regionBlock);
                    }
                }
            }

            for (auto blockPtr : workSet)
            {
                blockPtr->eraseFromParent();
                any = true;
            }
        } while (any);
    }

    BasicBlock *findUnreachableBlock(Function &F)
    {
        SmallPtrSet<BasicBlock *, 16> workSet;
        for (auto &regionBlock : F)
        {
            auto count = std::distance(regionBlock.begin(), regionBlock.end());
            if (count == 1 && isa<UnreachableInst>(regionBlock.begin()))
            {
                return &regionBlock;
            }
        }

        return nullptr;
    }
};

namespace ts
{
    bool verifyFunction(llvm::Function &F) 
    {
        llvm::ReversePostOrderTraversal<llvm::Function *> RPOT(&F);

        for (llvm::BasicBlock *BI : RPOT) 
        {
            for (BasicBlock::iterator II = BI->begin(), IE = BI->end(); II != IE;)
            {
                assert(II->getParent() == &*BI && "Moved to a different block!");
                ++II;
            }
        }

        return true;            
    }

    llvm::PreservedAnalyses Win32ExceptionPass::run(llvm::Function &F, llvm::FunctionAnalysisManager &AM)
    {
        Win32ExceptionPassCode TSEP{};
        if (!TSEP.runOnFunction(F))
        {
            LLVM_DEBUG(verifyFunction(F););

            return llvm::PreservedAnalyses::all();
        }

        return llvm::PreservedAnalyses::none();
    }
}