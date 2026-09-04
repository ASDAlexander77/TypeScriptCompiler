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

        if (returnsOwned.empty())
        {
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
                toErase.push_back(retain);
            }
        });

        for (auto *op : toErase)
        {
            op->erase();
        }
    }

  private:
    // The symbol a call names, when it names one directly. An indirect call through a value -
    // a callback, a method off an interface - answers empty and is left alone: there is no one
    // callee to inspect, so the caller keeps its retain and leaks rather than guessing.
    static mlir::StringRef calleeNameOf(mlir_ts::CallIndirectOp callOp)
    {
        if (callOp.getNumOperands() == 0)
        {
            return {};
        }

        auto symbolRefOp = callOp.getOperand(0).getDefiningOp<mlir_ts::SymbolRefOp>();
        if (!symbolRefOp)
        {
            return {};
        }

        return symbolRefOp.getIdentifier();
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
            if (!retainPrecedes(returnOp, value))
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
