#include "mlir/Pass/Pass.h"

#include "TypeScript/TypeScriptDialect.h"
#include "TypeScript/TypeScriptOps.h"
#include "TypeScript/Passes.h"
#include "TypeScript/Defines.h"
#include "TypeScript/MLIRLogic/MLIRTypeHelper.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "pass"

namespace mlir_ts = mlir::typescript;

namespace
{

// Lets a call take over the reference its callee returned, instead of retaining a second one.
//
// Every function retains its result on the way out (§9.24), so a receiver that retains it again
// - `let y = f()`, `h.item = f()` - is one owner above the truth and nothing is ever freed. The
// receiving sites already know how to consume such a value (§9.25); what they cannot know, at
// the point MLIRGen builds the call, is whether *this* callee is one that retains. Three things
// stop that being answerable there: the retain lives in the return statement, so other paths to
// a return reach it differently; the callee's FuncOp need not exist yet, which would make a
// lookup depend on declaration order; and a declared, imported or runtime callee looks identical
// to a local one while having no retaining return at all.
//
// All three dissolve once every function is present, which is why this is a pass rather than
// another marking site. It does not predict which callees retain - it looks. A function counts
// as returning owned only when every `ts.ReturnVal` of a heap-owning value in it is preceded by
// a `ts.Retain` of that same value. A callee with no body, a generator whose yields return
// without retaining, and anything else that does not match are simply not marked, and their
// callers go on retaining - which leaks rather than frees something live. The whole design puts
// the uncertain case on the leaking side.
//
// It runs in every memory model. The ops it removes erase on the way to LLVM under `gc` and
// `none` anyway, so removing them early keeps the IR the same shape in all three and keeps the
// ownership verifier checking one thing rather than two.
class OwnedReturnConsumptionPass
    : public mlir::PassWrapper<OwnedReturnConsumptionPass, mlir::OperationPass<mlir::ModuleOp>>
{
    CompileOptions compileOptions;

  public:
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(OwnedReturnConsumptionPass)

    OwnedReturnConsumptionPass(CompileOptions &compileOptions) : compileOptions(compileOptions)
    {
    }

    void runOnOperation() override
    {
        auto module = getOperation();
        MLIRTypeHelper mth(module->getContext(), compileOptions);

        llvm::DenseSet<mlir::StringRef> returnsOwned;
        module.walk([&](mlir_ts::FuncOp funcOp) {
            if (functionReturnsOwned(mth, funcOp))
            {
                returnsOwned.insert(funcOp.getName());
            }
        });

        // `new C()` is marked where it is built, so there can be discarded temporaries to give
        // back even when no function here is classified as returning owned.
        if (returnsOwned.empty())
        {
            releaseDiscardedTemporaries(mth, module);
            return;
        }

        llvm::SmallVector<mlir::Operation *> toErase;
        module.walk([&](mlir_ts::CallIndirectOp callOp) {
            if (callOp.getNumResults() != 1)
            {
                return;
            }

            auto result = callOp.getResult(0);
            if (!mth.ownsHeapMemory(callOp.getLoc(), result.getType()))
            {
                return;
            }

            auto callee = calleeNameOf(callOp);
            if (callee.empty() || !returnsOwned.contains(callee))
            {
                return;
            }

            // Already settled at the point the call was built - `new C()` is marked there,
            // where the callee is known outright. Nothing left to remove.
            if (callOp->hasAttr(OWNED_RESULT_ATTR_NAME))
            {
                return;
            }

            if (auto *retain = findReceiverRetain(result))
            {
                callOp->setAttr(OWNED_RESULT_ATTR_NAME, mlir::UnitAttr::get(&getContext()));
                callOp->setAttr(OWNED_RESULT_CONSUMED_ATTR_NAME, mlir::UnitAttr::get(&getContext()));
                toErase.push_back(retain);
                return;
            }

            // Nobody took it. The +1 stands with no owner, which is the leak §9.30 closes -
            // marked here, released below once every consumer has had its say.
            callOp->setAttr(OWNED_RESULT_ATTR_NAME, mlir::UnitAttr::get(&getContext()));
        });

        for (auto *op : toErase)
        {
            op->erase();
        }

        releaseDiscardedTemporaries(mth, module);
    }

  private:
    // Gives back the +1 on a produced reference that no receiver ever took.
    //
    // Every function retains its result on the way out (§9.24), so a call hands back a reference
    // whether or not the caller does anything with it. Where a receiver takes it over the pair is
    // balanced (§9.25, §9.27); where nothing does, the reference stands with no owner and the
    // value is never freed. That is not a corner case: `raytrace.ts` is built out of
    // `Vector.plus(Vector.times(k, a), b)`, so nearly every allocation it makes is an
    // intermediate passed straight as an argument, and it reclaimed nothing at all before this.
    //
    // WHERE the release goes is the whole difficulty - "after the last use" is not something
    // MLIRGen can see while it is still building the expression. The answer here is the END OF
    // THE PRODUCER'S OWN BLOCK, which is a temporary's natural lifetime (the enclosing statement,
    // or one iteration of a loop body) and, more importantly, is unconditionally after every use
    // in that block. Placing it after the last *user* instead looks tighter and is wrong: the
    // receiver of `let x = <T>f()` retains the result of the CAST, not of the call, so the call's
    // last user is the cast and a release put there would run before that retain and free the
    // value out from under it. End-of-block cannot get that ordering wrong.
    //
    // Two things disqualify a value, and both simply leave it leaking as before:
    //
    //  - a user outside the producer's block, so the value outlives the block or is used on a
    //    path this cannot see;
    //  - a user that is a terminator, since a value handed to a successor as a block argument is
    //    still live after the point this would release it;
    //  - a `ts.StateLabel` after the definition, which is a generator's resume point: the state
    //    machine re-enters the block THERE, so the end of the block is reachable on a path that
    //    never ran the op that produced the value. Nothing about that is visible while the
    //    generator is still one block - it only becomes a use before definition once the state
    //    machine is expanded, and it surfaces as a dominance failure in the affine lowering
    //    rather than as anything this pass could notice.
    //
    // That bias is the same one the rest of this arc takes: an unreleased reference is invisible,
    // a released one that was still owned is a use-after-free.
    void releaseDiscardedTemporaries(MLIRTypeHelper &mth, mlir::ModuleOp module)
    {
        llvm::SmallVector<mlir::Operation *> discarded;
        module.walk([&](mlir::Operation *op) {
            if (!op->hasAttr(OWNED_RESULT_ATTR_NAME) || op->hasAttr(OWNED_RESULT_CONSUMED_ATTR_NAME))
            {
                return;
            }

            if (op->getNumResults() != 1 || !mth.ownsHeapMemory(op->getLoc(), op->getResult(0).getType()))
            {
                return;
            }

            discarded.push_back(op);
        });

        mlir::OpBuilder builder(&getContext());
        for (auto *op : discarded)
        {
            if (!allUsesReleasableInOwnBlock(op))
            {
                continue;
            }

            auto *block = op->getBlock();
            auto *terminator = block->getTerminator();
            if (terminator)
            {
                builder.setInsertionPoint(terminator);
            }
            else
            {
                builder.setInsertionPointToEnd(block);
            }

            builder.create<mlir_ts::ReleaseOp>(op->getLoc(), op->getResult(0));
        }
    }

    // Can a release at the end of this value's own block give its reference back safely? See
    // releaseDiscardedTemporaries for what disqualifies a use and why.
    static bool allUsesReleasableInOwnBlock(mlir::Operation *op)
    {
        auto *block = op->getBlock();
        for (auto *user : op->getResult(0).getUsers())
        {
            if (user->getBlock() != block || user->hasTrait<mlir::OpTrait::IsTerminator>())
            {
                return false;
            }
        }

        // a resume point between the definition and the end of the block - see above
        for (auto it = std::next(mlir::Block::iterator(op)); it != block->end(); ++it)
        {
            if (mlir::isa<mlir_ts::StateLabelOp>(*it))
            {
                return false;
            }
        }

        return true;
    }

    static bool producesOwnedResult(mlir::Value value)
    {
        auto *definingOp = value.getDefiningOp();
        return definingOp && definingOp->hasAttr(OWNED_RESULT_ATTR_NAME);
    }

    // The symbol a call names, when it names one. An indirect call through a value - a callback,
    // a method off an interface, a function-typed field - answers empty and is left alone: there
    // is no one callee to inspect, so the caller keeps its retain and leaks rather than guessing.
    //
    // A method call is not that. `obj.m(x)` builds a bound function and then splits it apart
    // again - `GetMethod` for the code, `GetThis` for the receiver - so the callee is a step
    // further back than a plain function's. The shapes below are the ones the dialect's own
    // canonicalizer (SimplifyIndirectCallWithKnownCallee) already rewrites into direct calls,
    // which is the argument that they name one callee: it is the same judgement, made here
    // before canonicalization has run.
    static mlir::StringRef calleeNameOf(mlir_ts::CallIndirectOp callOp)
    {
        if (callOp.getNumOperands() == 0)
        {
            return {};
        }

        auto callee = callOp.getOperand(0);

        if (auto symbolRefOp = callee.getDefiningOp<mlir_ts::SymbolRefOp>())
        {
            return symbolRefOp.getIdentifier();
        }

        // a non-virtual method, called without the bound-function detour
        if (auto thisSymbolRefOp = callee.getDefiningOp<mlir_ts::ThisSymbolRefOp>())
        {
            return thisSymbolRefOp.getIdentifier();
        }

        auto getMethodOp = callee.getDefiningOp<mlir_ts::GetMethodOp>();
        if (!getMethodOp)
        {
            return {};
        }

        auto boundFunc = getMethodOp.getBoundFunc();

        if (auto thisSymbolRefOp = boundFunc.getDefiningOp<mlir_ts::ThisSymbolRefOp>())
        {
            return thisSymbolRefOp.getIdentifier();
        }

        // A bound function built here from a known function - a trampoline - names it outright.
        if (auto createBoundFunctionOp = boundFunc.getDefiningOp<mlir_ts::CreateBoundFunctionOp>())
        {
            if (auto symbolRefOp = createBoundFunctionOp.getFunc().getDefiningOp<mlir_ts::SymbolRefOp>())
            {
                return symbolRefOp.getIdentifier();
            }

            return {};
        }

        // `ts.ThisVirtualSymbolRef` is deliberately absent. It carries an identifier, but that
        // names the declaration the call was written against, not what the runtime class put in
        // the slot - so reading it as the callee would consume a reference an override may never
        // have taken. `private` looks like it would settle this and does not: this compiler
        // accepts a subclass redeclaring a private method and dispatches to the override, where
        // TypeScript rejects the program outright. See §9.32.
        return {};
    }

    // Does every return of a heap-owning value in this function retain it first?
    //
    // Looked up rather than assumed, and answered "no" for anything unclear: a function with no
    // body, a return whose retain is not in the same block, a return with no retain at all. A
    // false "no" costs a leak; a false "yes" frees a value the callee never retained.
    static bool functionReturnsOwned(MLIRTypeHelper &mth, mlir_ts::FuncOp funcOp)
    {
        if (funcOp.isExternal() || funcOp.getBody().empty())
        {
            return false;
        }

        // A generator's returns are not the value its caller receives - the caller gets the
        // generator object, built by the transformation rather than by these returns. Rather
        // than reason about what that transformation leaves behind, leave any function with a
        // yield in it alone; its callers keep retaining, and leak.
        auto isGenerator = false;
        funcOp.walk([&](mlir_ts::YieldReturnValOp) { isGenerator = true; });
        if (isGenerator)
        {
            return false;
        }

        auto sawOwningReturn = false;
        auto everyReturnRetains = true;
        funcOp.walk([&](mlir_ts::ReturnValOp returnOp) {
            auto value = returnOp.getOperand();
            if (!mth.ownsHeapMemory(returnOp.getLoc(), value.getType()))
            {
                return;
            }

            sawOwningReturn = true;

            // Two ways a return can hand back a reference. It retains one of its own, which is
            // the ordinary case - or it forwards a reference it was already given, and then
            // there is no retain to find: `return new C()` consumes the instance's own +1
            // rather than adding a second (§9.25). Reading only the first shape as "returns
            // owned" left every `static times(...) { return new Vector(...) }` unclassified,
            // which is most of what expression-shaped code is built from.
            if (!retainPrecedes(returnOp, value) && !producesOwnedResult(value))
            {
                everyReturnRetains = false;
            }
        });

        return sawOwningReturn && everyReturnRetains;
    }

    // Is there a `ts.Retain` of `value` ahead of `op` in its own block? The scope-exit releases
    // sit between the two in the ordinary case, so this is not an adjacency test - but it stays
    // inside one block deliberately. A retain somewhere else may not run on the path that
    // reaches this return, and reading "retained" off a path that never retained is the one
    // mistake here that frees live memory.
    static bool retainPrecedes(mlir::Operation *op, mlir::Value value)
    {
        for (auto it = mlir::Block::iterator(op); it != op->getBlock()->begin();)
        {
            --it;
            if (auto retainOp = mlir::dyn_cast<mlir_ts::RetainOp>(*it))
            {
                if (retainOp.getReference() == value)
                {
                    return true;
                }
            }
        }

        return false;
    }

    // The retain a receiver put on this call's result, if it took one. Two shapes, matching the
    // two ways §9.25's receivers acquire: a `ts.Retain` on the value itself (a field or element
    // store, a literal capturing it, a return passing it on), and a `ts.RetainSlot` on the
    // storage of a local declared from it.
    //
    // Returning null is the ordinary answer for a result nobody took - `f();` on its own, or
    // `f().n` - and it is left exactly as it is. Consuming a reference no receiver balances
    // would free the value while the expression is still using it.
    static mlir::Operation *findReceiverRetain(mlir::Value result)
    {
        for (auto *user : result.getUsers())
        {
            if (auto retainOp = mlir::dyn_cast<mlir_ts::RetainOp>(user))
            {
                if (retainOp.getReference() == result)
                {
                    return retainOp.getOperation();
                }
            }
        }

        for (auto *user : result.getUsers())
        {
            auto varOp = mlir::dyn_cast<mlir_ts::VariableOp>(user);
            if (!varOp || !varOp->hasAttr(OWNED_LOCAL_ATTR_NAME) ||
                varOp->hasAttr(OWNED_LOCAL_CONSUMED_ATTR_NAME))
            {
                continue;
            }

            for (auto *slotUser : varOp.getResult().getUsers())
            {
                if (auto retainSlotOp = mlir::dyn_cast<mlir_ts::RetainSlotOp>(slotUser))
                {
                    // the declaration becomes the acquisition, which is what keeps the
                    // ownership verifier able to pair the release still to come
                    varOp->setAttr(OWNED_LOCAL_CONSUMED_ATTR_NAME, mlir::UnitAttr::get(varOp.getContext()));
                    return retainSlotOp.getOperation();
                }
            }
        }

        return nullptr;
    }
};

} // end anonymous namespace

#undef DEBUG_TYPE

/// Create pass.
std::unique_ptr<mlir::Pass> mlir_ts::createOwnedReturnConsumptionPass(CompileOptions &compileOptions)
{
    return std::make_unique<OwnedReturnConsumptionPass>(compileOptions);
}
