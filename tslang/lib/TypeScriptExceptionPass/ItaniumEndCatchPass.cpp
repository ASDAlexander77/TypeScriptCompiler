#include "TypeScript/Pass/ItaniumEndCatchPass.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/Debug.h"
#include "llvm/Transforms/Utils/Local.h"

#include <cstdlib>

using namespace llvm;

#define DEBUG_TYPE "pass"

// Itanium EH ends a catch with __cxa_end_catch, which pops the caught-exceptions stack and frees
// the exception once nothing holds it. TryOpLowering ends a catch on the ways out of it it can
// see - the end of the clause, a `return`, a `break`, a `throw` - but not on an unwind edge: a
// call in a catch clause that throws, or the rethrow a finally, a `using` or a typed catch that
// does not match ends with. The exception that catch began was then never ended: one leak per
// throw, and the caught-exceptions stack grew without bound (2.7M throws out of a catch clause
// took 480 MB, in every memory model). A rethrown exception was never freed either: after
// __cxa_rethrow its count stays one too high unless the catch that rethrew it is ended, so the
// catch that took it next could not free it.
//
// This pass ends them where C++ compilers do, on the unwind edge:
// - TryOpLowering marks each try's entry (ts.internal.eh_try(id)) and each of its landing pads
//   (ts.internal.eh_try_pad(id) right after the landingpad). A pad expects the catches active at
//   its try's entry - not the ones active where an exception reaches it, which a pad cannot tell
//   from its predecessors: the pad of a try around a catch clause may be reached only from a
//   call in that clause.
// - The number of catches active at each point is counted forward from the function's entry
//   (__cxa_begin_catch + 1, __cxa_end_catch - 1).
// - An invoke that unwinds to a pad with more catches active than the pad expects is pointed at
//   a copy of the pad that ends the difference first, then joins the pad.
// - A call that may throw, with catches active and no pad to unwind to, is given one that ends
//   them and resumes the unwind; a `resume` with catches active ends them first.
//
// The rethrown exception is ended this way too: __cxa_end_catch on a rethrown exception pops it
// without freeing it, and the catch that takes it next ends it for good.
//
// A function where the count does not come out the same on every path into a block (a `yield`
// in a catch clause, say) is left as it was, markers aside.
namespace
{
constexpr const char *TRY_MARKER_NAME = "ts.internal.eh_try";
constexpr const char *PAD_MARKER_NAME = "ts.internal.eh_try_pad";

struct ItaniumEndCatchPassCode
{
    Function *beginCatch = nullptr;
    Function *endCatch = nullptr;
    Function *tryMarker = nullptr;
    Function *padMarker = nullptr;

    static Function *calleeOf(Instruction &I)
    {
        if (auto *CB = dyn_cast<CallBase>(&I))
        {
            return CB->getCalledFunction();
        }

        return nullptr;
    }

    static int markerId(Instruction &I)
    {
        return (int)cast<ConstantInt>(cast<CallBase>(&I)->getArgOperand(0))->getSExtValue();
    }

    static bool report()
    {
        static auto on = std::getenv("TSLANG_EH_END_CATCH_REPORT") != nullptr;
        return on;
    }

    bool fail(Function &F, const char *why)
    {
        LLVM_DEBUG(dbgs() << "\n!! ITANIUM END CATCH: " << F.getName() << " left as it is: " << why << "\n";);
        if (report())
        {
            errs() << "ItaniumEndCatchPass: " << F.getName() << " left as it is: " << why << "\n";
        }

        return false;
    }

    // the change in the number of active catches an instruction makes
    int delta(Instruction &I)
    {
        auto *callee = calleeOf(I);
        if (callee == nullptr)
        {
            return 0;
        }

        if (callee == beginCatch)
        {
            return 1;
        }

        if (callee == endCatch)
        {
            return -1;
        }

        return 0;
    }

    // nothing runs after it on the normal path: what follows a throw is an `unreachable` that
    // throws made at different depths may share
    static bool noReturn(Instruction &I)
    {
        auto *CB = dyn_cast<CallBase>(&I);
        if (CB == nullptr)
        {
            return false;
        }

        if (CB->doesNotReturn())
        {
            return true;
        }

        auto *callee = CB->getCalledFunction();
        return callee != nullptr &&
               (callee->getName() == "__cxa_throw" || callee->getName() == "__cxa_rethrow" ||
                callee->getName() == "_Unwind_Resume");
    }

    bool mayThrowOut(CallInst *CI)
    {
        if (CI->doesNotThrow() || CI->isInlineAsm() || isa<IntrinsicInst>(CI))
        {
            return false;
        }

        auto *callee = CI->getCalledFunction();
        return callee != beginCatch && callee != endCatch && callee != tryMarker && callee != padMarker;
    }

    // the number of catches active on entry to each block reachable from the function's entry
    bool countActiveCatches(Function &F, DenseMap<BasicBlock *, int> &padTry, DenseMap<BasicBlock *, int> &depthIn)
    {
        DenseMap<int, int> tryDepth;
        DenseMap<int, SmallVector<BasicBlock *>> padsOfTry;
        for (auto [pad, id] : padTry)
        {
            padsOfTry[id].push_back(pad);
        }

        SmallVector<BasicBlock *> worklist;
        auto consistent = true;
        auto enter = [&](BasicBlock *BB, int depth) {
            auto [it, inserted] = depthIn.try_emplace(BB, depth);
            if (inserted)
            {
                worklist.push_back(BB);
            }
            // the block every failed assert in a function shares, say: nothing leaves it
            else if (it->second != depth && !isa<UnreachableInst>(BB->getTerminator()))
            {
                if (consistent && report())
                {
                    errs() << "ItaniumEndCatchPass: " << BB->getName() << " entered with " << it->second << " and " << depth
                           << "\n";
                    if (std::getenv("TSLANG_EH_END_CATCH_REPORT")[0] == '2')
                    {
                        errs() << F << "\n";
                    }
                }

                consistent = false;
            }
        };

        enter(&F.getEntryBlock(), 0);
        while (!worklist.empty() && consistent)
        {
            auto *BB = worklist.pop_back_val();
            auto depth = depthIn[BB];
            auto returns = true;
            for (auto &I : *BB)
            {
                if (!returns)
                {
                    break;
                }

                returns = !noReturn(I);

                if (calleeOf(I) == tryMarker && tryMarker)
                {
                    auto id = markerId(I);
                    auto [it, inserted] = tryDepth.try_emplace(id, depth);
                    if (!inserted && it->second != depth)
                    {
                        return fail(F, "a try entered with different catches active");
                    }

                    for (auto *pad : padsOfTry.lookup(id))
                    {
                        enter(pad, depth);
                    }
                }

                if (auto *II = dyn_cast<InvokeInst>(&I))
                {
                    if (!padTry.contains(II->getUnwindDest()))
                    {
                        return fail(F, "an invoke unwinds to a pad of no marked try");
                    }
                }

                depth += delta(I);
                if (depth < 0)
                {
                    return fail(F, "more catches ended than begun");
                }
            }

            auto *terminator = BB->getTerminator();
            if (!returns)
            {
                continue;
            }

            if (auto *II = dyn_cast<InvokeInst>(terminator))
            {
                enter(II->getNormalDest(), depth);
            }
            else
            {
                for (auto *successor : successors(BB))
                {
                    enter(successor, depth);
                }
            }
        }

        if (!consistent)
        {
            return fail(F, "a block entered with different catches active");
        }

        return true;
    }

    bool runOnFunction(Function &F)
    {
        auto *M = F.getParent();
        tryMarker = M->getFunction(TRY_MARKER_NAME);
        padMarker = M->getFunction(PAD_MARKER_NAME);
        if (!tryMarker && !padMarker)
        {
            return false;
        }

        SmallVector<CallInst *> markers;
        DenseMap<BasicBlock *, int> padTry;
        auto padsMarked = true;
        for (auto &I : instructions(F))
        {
            auto *callee = calleeOf(I);
            if (callee == nullptr || (callee != tryMarker && callee != padMarker))
            {
                continue;
            }

            markers.push_back(cast<CallInst>(&I));
            if (callee == padMarker)
            {
                auto *BB = I.getParent();
                padsMarked &= BB->isLandingPad();
                padTry[BB] = markerId(I);
            }
        }

        if (markers.empty())
        {
            return false;
        }

        beginCatch = M->getFunction("__cxa_begin_catch");
        endCatch = M->getFunction("__cxa_end_catch");

        auto erasedMarkers = [&]() {
            for (auto *marker : markers)
            {
                marker->eraseFromParent();
            }

            return true;
        };

        // with no catch begun there is nothing to end
        if (!beginCatch || !F.hasPersonalityFn())
        {
            return erasedMarkers();
        }

        if (!padsMarked)
        {
            fail(F, "a pad marker outside a landing pad");
            return erasedMarkers();
        }

        DenseMap<BasicBlock *, int> depthIn;
        if (!countActiveCatches(F, padTry, depthIn))
        {
            return erasedMarkers();
        }

        if (!endCatch)
        {
            endCatch = cast<Function>(
                M->getOrInsertFunction("__cxa_end_catch", FunctionType::get(Type::getVoidTy(F.getContext()), false))
                    .getCallee());
        }

        // what to change, collected before any of it is: the blocks are split as it is changed
        SmallVector<std::pair<InvokeInst *, int>> invokes;
        SmallVector<std::pair<CallInst *, int>> calls;
        SmallVector<std::pair<ResumeInst *, int>> resumes;
        for (auto [BB, depthOnEntry] : depthIn)
        {
            auto depth = depthOnEntry;
            for (auto &I : *BB)
            {
                if (auto *II = dyn_cast<InvokeInst>(&I))
                {
                    if (!depthIn.contains(II->getUnwindDest()))
                    {
                        fail(F, "an invoke unwinds to a pad whose try is never entered");
                        return erasedMarkers();
                    }

                    auto excess = depth - depthIn.lookup(II->getUnwindDest());
                    if (excess < 0)
                    {
                        fail(F, "an invoke unwinds to a pad that expects more catches than are active");
                        return erasedMarkers();
                    }

                    if (excess > 0)
                    {
                        invokes.push_back({II, excess});
                    }
                }
                else if (auto *CI = dyn_cast<CallInst>(&I))
                {
                    if (depth > 0 && mayThrowOut(CI))
                    {
                        calls.push_back({CI, depth});
                    }
                }
                else if (auto *RI = dyn_cast<ResumeInst>(&I))
                {
                    if (depth > 0)
                    {
                        resumes.push_back({RI, depth});
                    }
                }

                depth += delta(I);
                if (noReturn(I))
                {
                    break;
                }
            }
        }

        erasedMarkers();

        auto endCatches = [&](IRBuilder<> &builder, int count) {
            for (auto i = 0; i < count; i++)
            {
                builder.CreateCall(endCatch);
            }
        };

        // a copy of the pad that ends `excess` catches first, then joins it
        DenseMap<std::pair<BasicBlock *, int>, BasicBlock *> endingPads;
        DenseMap<BasicBlock *, PHINode *> joins;
        auto endingPadOf = [&](BasicBlock *pad, int excess) {
            auto &endingPad = endingPads[{pad, excess}];
            if (endingPad)
            {
                return endingPad;
            }

            auto *landingPad = pad->getLandingPadInst();
            auto *&join = joins[pad];
            if (!join)
            {
                auto *rest = pad->splitBasicBlock(landingPad->getNextNode(), pad->getName() + ".caught");
                join = PHINode::Create(landingPad->getType(), 2, "", rest->begin());
                landingPad->replaceUsesWithIf(join, [&](Use &use) { return use.getUser() != join; });
                join->addIncoming(landingPad, pad);
            }

            endingPad = BasicBlock::Create(F.getContext(), "endcatch.pad", &F, pad);
            auto *copy = landingPad->clone();
            copy->insertInto(endingPad, endingPad->end());
            IRBuilder<> builder(endingPad);
            endCatches(builder, excess);
            builder.CreateBr(join->getParent());
            join->addIncoming(copy, endingPad);
            return endingPad;
        };

        for (auto [II, excess] : invokes)
        {
            II->setUnwindDest(endingPadOf(II->getUnwindDest(), excess));
        }

        // a pad of its own that ends `count` catches and carries on unwinding
        Type *landingPadType = nullptr;
        for (auto [pad, id] : padTry)
        {
            landingPadType = pad->getLandingPadInst()->getType();
            break;
        }

        DenseMap<int, BasicBlock *> exitPads;
        auto exitPadOf = [&](int count) {
            auto &exitPad = exitPads[count];
            if (!exitPad)
            {
                exitPad = BasicBlock::Create(F.getContext(), "endcatch.resume", &F);
                IRBuilder<> builder(exitPad);
                auto *landingPad = builder.CreateLandingPad(landingPadType, 0);
                landingPad->setCleanup(true);
                endCatches(builder, count);
                builder.CreateResume(landingPad);
            }

            return exitPad;
        };

        for (auto [CI, count] : landingPadType ? calls : decltype(calls){})
        {
            changeToInvokeAndSplitBasicBlock(CI, exitPadOf(count));
        }

        for (auto [RI, count] : resumes)
        {
            IRBuilder<> builder(RI);
            endCatches(builder, count);
        }

        return true;
    }
};
} // namespace

namespace ts
{
llvm::PreservedAnalyses ItaniumEndCatchPass::run(llvm::Function &F, llvm::FunctionAnalysisManager &AM)
{
    ItaniumEndCatchPassCode pass{};
    if (!pass.runOnFunction(F))
    {
        return llvm::PreservedAnalyses::all();
    }

    return llvm::PreservedAnalyses::none();
}
} // namespace ts

#undef DEBUG_TYPE
