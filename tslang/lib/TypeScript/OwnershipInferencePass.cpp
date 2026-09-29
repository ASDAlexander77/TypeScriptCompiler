#include "mlir/Pass/Pass.h"
#include "mlir/IR/Dominance.h"
#include "mlir/Dialect/LLVMIR/LLVMAttrs.h"

#include "TypeScript/TypeScriptDialect.h"
#include "TypeScript/TypeScriptOps.h"
#include "TypeScript/TypeScriptFunctionPass.h"
#include "TypeScript/Passes.h"
#include "TypeScript/Defines.h"

#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"

#define DEBUG_TYPE "own"

namespace mlir_ts = mlir::typescript;

namespace
{

// Ownership inference for -mm=own, phases 0 and 1. See
// docs/superpowers/specs/2026-09-24-own-memory-model-design.md, sections 4.2, 10.6, 11 and 12.
//
// A fresh heap value may have at most one owner: one use that takes it - a local's
// declaration, a store into a field, element, global or return slot, an insertion into an
// array, a return - or, with none of those, the temporary itself, which rc gives back with a
// `ts.Release`. That owner's release is then its destroy, and every retain rc planted on the
// way is erased. Owners are counted from the uses themselves, not from rc's retains: rc also
// transfers ownership by *consuming* a value (`__owned_consumed`, a push, a field store),
// which leaves no retain behind to count.
//
// A move is decided by reachability on the affine CFG (spec 4.2 step 3): nothing may use the value
// after its taker, a taker that control comes back to without the value being made again is a
// move on every loop iteration (2.6), and each release the move reaches must be dominated by it
// and is erased. A release the move cannot reach stays - the owner on the paths that did not
// move. Anything else is an error at its location.
class OwnershipInferencePass : public mlir::PassWrapper<OwnershipInferencePass, TypeScriptFunctionPass>
{
  public:
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(OwnershipInferencePass)

    void runOnFunction() override
    {
        auto f = getFunction();
        mlir::DominanceInfo dominanceInfo(f);
        dominance = &dominanceInfo;

        // every value some ownership operation names, in the order the walk meets them
        llvm::SetVector<mlir::Value> candidates;
        llvm::SmallVector<mlir::Operation *> retains;

        f.walk([&](mlir::Operation *op) {
            if (auto retainOp = mlir::dyn_cast<mlir_ts::RetainOp>(op))
            {
                retains.push_back(op);
                candidates.insert(retainOp.getReference());
            }
            else if (auto retainSlotOp = mlir::dyn_cast<mlir_ts::RetainSlotOp>(op))
            {
                retains.push_back(op);
            }
            else if (auto releaseOp = mlir::dyn_cast<mlir_ts::ReleaseOp>(op))
            {
                candidates.insert(releaseOp.getReference());
            }
            else if (auto varOp = mlir::dyn_cast<mlir_ts::VariableOp>(op))
            {
                if (varOp.getInitializer() && isOwningVariable(varOp))
                {
                    candidates.insert(varOp.getInitializer());
                }
            }
            else if (mlir::isa<mlir_ts::RetainCellOp>(op))
            {
                op->emitError("closures that capture a variable are not supported by -mm=own yet");
                signalPassFailure();
            }
            else if (mlir::isa<mlir_ts::DeleteOp>(op))
            {
                op->emitError("'delete' is not supported by -mm=own yet");
                signalPassFailure();
            }

            for (auto result : op->getResults())
            {
                if (isFresh(result))
                {
                    candidates.insert(result);
                }
            }
        });

        // Every value gets the owner check, fresh or not: rc can hand a value it did not make to
        // two consuming declarations without a single retain (the result of an indirect call
        // taken by `let b = a; let c = a;`). Only a fresh value's retains can be erased.
        llvm::DenseSet<mlir::Value> proven;
        llvm::SetVector<mlir::Operation *> toErase;
        for (auto value : candidates)
        {
            if (checkValueMoves(value, toErase) && isFresh(value))
            {
                proven.insert(value);
            }
        }

        // What is proven loses its retains; any other retain is a second reference nobody
        // proved safe - of a loaded value, a parameter, an unknown producer, or of a fresh value
        // whose own check was reported above.
        for (auto *op : retains)
        {
            auto value = retainedValue(op);
            if (value && proven.contains(value))
            {
                toErase.insert(op);
            }
            else if (!value || isFresh(value) == false)
            {
                reportSecondReference(op);
            }
        }

        for (auto *op : toErase)
        {
            op->erase();
        }
    }

  private:
    mlir::DominanceInfo *dominance = nullptr;

    // The value a retain acquires: a Retain's operand, or a RetainSlot's variable's initializer.
    // None for a variable with no initializer - its storage was hoisted in front of a try and its
    // value arrives by a store this phase does not follow.
    static mlir::Value retainedValue(mlir::Operation *op)
    {
        if (auto retainOp = mlir::dyn_cast<mlir_ts::RetainOp>(op))
        {
            return retainOp.getReference();
        }

        auto varOp = mlir::cast<mlir_ts::RetainSlotOp>(op).getSlot().getDefiningOp<mlir_ts::VariableOp>();
        return varOp ? varOp.getInitializer() : mlir::Value();
    }

    static bool isOwningVariable(mlir_ts::VariableOp varOp)
    {
        return varOp->hasAttr(OWNED_LOCAL_ATTR_NAME) || varOp->hasAttr(OWNED_LOCAL_CONSUMED_ATTR_NAME);
    }

    // Made here and held by nobody else: an allocation, a literal cast to its value type, an
    // operation rc marks as arriving with a reference, or a direct call of a function this module
    // defines (every function returns its result retained, rc 9.24; the call's own mark does not
    // survive the affine lowering). A declared callee, an indirect call, a parameter, a load or a
    // value merged from branches is not fresh.
    static bool isFresh(mlir::Value value)
    {
        auto *def = value.getDefiningOp();
        if (!def)
        {
            return false;
        }

        if (mlir::isa<mlir_ts::NewOp, mlir_ts::CreateArrayOp, mlir_ts::NewArrayOp, mlir_ts::StringConcatOp,
                      mlir_ts::CharToStringOp>(def))
        {
            return true;
        }

        if (auto callOp = mlir::dyn_cast<mlir_ts::SymbolCallInternalOp>(def))
        {
            auto callee = mlir::SymbolTable::lookupNearestSymbolFrom<mlir_ts::FuncOp>(def, callOp.getCalleeAttr());
            return callee && !callee.isDeclaration();
        }

        if (mlir::isa<mlir_ts::CallOp, mlir_ts::CallIndirectOp, mlir_ts::CallInternalOp, mlir_ts::CallHybridInternalOp>(def))
        {
            return false;
        }

        return def->hasAttr(OWNED_RESULT_ATTR_NAME) || isLiteral(def);
    }

    // A string literal or a constant array cast to its value type. The result is either the
    // immortal global itself, which a release skips, or a copy nobody else holds, which a
    // release destroys - so, owned once, it has one owner either way.
    static bool isLiteral(mlir::Operation *def)
    {
        auto castOp = mlir::dyn_cast<mlir_ts::CastOp>(def);
        return castOp && castOp.getIn().getDefiningOp<mlir_ts::ConstantOp>();
    }

    // A use that reads the value without keeping it. Kept deliberately short: anything not listed
    // is treated as taking the value, which can only turn a program into an error.
    static bool isBorrow(mlir::Operation *user, mlir::Value value)
    {
        if (mlir::isa<mlir_ts::PrintOp, mlir_ts::PropertyRefOp, mlir_ts::ElementRefOp, mlir_ts::LengthOfOp,
                      mlir_ts::ThisSymbolRefOp, mlir_ts::VirtualSymbolRefOp, mlir_ts::ThisVirtualSymbolRefOp,
                      mlir_ts::InterfaceSymbolRefOp, mlir_ts::GetThisOp, mlir_ts::GetMethodOp,
                      mlir_ts::ArithmeticBinaryOp, mlir_ts::LogicalBinaryOp, mlir_ts::StringConcatOp>(user))
        {
            return true;
        }

        // `--di`'s record of a folded const's value for the debugger: names it, keeps nothing
        if (mlir::isa<mlir_ts::DebugVariableOp>(user))
        {
            return true;
        }

        // arguments are borrowed; a callee that keeps one retains it, and that retain is its own
        // error
        if (mlir::isa<mlir_ts::CallOp, mlir_ts::CallIndirectOp, mlir_ts::SymbolCallInternalOp, mlir_ts::CallInternalOp,
                      mlir_ts::CallHybridInternalOp>(user))
        {
            return true;
        }

        // the array being pushed onto is written, not taken; what is inserted is taken
        if (auto pushOp = mlir::dyn_cast<mlir_ts::ArrayPushOp>(user))
        {
            return pushOp.getOp() == value && !llvm::is_contained(pushOp.getItems(), value);
        }

        if (auto unshiftOp = mlir::dyn_cast<mlir_ts::ArrayUnshiftOp>(user))
        {
            return unshiftOp.getOp() == value && !llvm::is_contained(unshiftOp.getItems(), value);
        }

        if (auto spliceOp = mlir::dyn_cast<mlir_ts::ArraySpliceOp>(user))
        {
            return spliceOp.getOp() == value && !llvm::is_contained(spliceOp.getItems(), value);
        }

        // a declaration that does not own what it holds only reads it
        if (auto varOp = mlir::dyn_cast<mlir_ts::VariableOp>(user))
        {
            return !isOwningVariable(varOp);
        }

        return false;
    }

    // The moves out of a fresh or unknown SSA value. Each taking use (spec 11.1) is a move:
    //   - nothing else may use the value after it - another taker included - and a taker that
    //     control comes back to without the value being made again is a move on every iteration
    //     of a loop (spec 2.6);
    //   - where the temporary is also released (`__owned_result` rc gives back at block end), the
    //     taker must be one rc gave its own reference (takerAcquires), and each release the move
    //     reaches must be dominated by it - that release then has nothing to give back and is
    //     erased. A release the move cannot reach stays: it is the owner on the other paths.
    // Reports and returns false otherwise.
    bool checkValueMoves(mlir::Value value, llvm::SetVector<mlir::Operation *> &toErase)
    {
        llvm::SmallVector<mlir::Operation *> takers;
        llvm::SmallVector<mlir::Operation *> releases;
        llvm::SmallVector<mlir::Operation *> reads;
        for (auto *user : value.getUsers())
        {
            if (mlir::isa<mlir_ts::RetainOp>(user))
            {
                continue;
            }

            if (mlir::isa<mlir_ts::ReleaseOp>(user))
            {
                releases.push_back(user);
            }
            else if (isBorrow(user, value))
            {
                reads.push_back(user);
            }
            else
            {
                takers.push_back(user);
            }
        }

        auto *definition = value.getDefiningOp();
        auto name = describe(value);
        llvm::SmallVector<mlir::Operation *> moved;
        for (auto *taker : takers)
        {
            if (takerLoops(taker, definition))
            {
                reportMovedInLoop(taker, name);
                return false;
            }

            for (auto *use : llvm::concat<mlir::Operation *const>(takers, reads))
            {
                if (use != taker && reachableAfter(taker, use, definition))
                {
                    reportUseAfterMove(use, taker, name);
                    return false;
                }
            }

            if (releases.empty())
            {
                continue; // rc handed the taker the value's own reference: nothing to cancel
            }

            if (!takerAcquires(taker, value))
            {
                reportSecondReference(taker);
                return false;
            }

            auto *acquire = acquirePoint(taker, value);
            for (auto *release : releases)
            {
                if (!reachableAfter(acquire, release, definition))
                {
                    continue;
                }

                if (!dominance->properlyDominates(acquire, release))
                {
                    reportMovedOnSomePaths(taker, release, name);
                    return false;
                }

                moved.push_back(release);
            }
        }

        toErase.insert(moved.begin(), moved.end());
        return true;
    }

    // Does control come back round to `taker` without making the value again? Only then is it a
    // move on every iteration of a loop.
    bool takerLoops(mlir::Operation *taker, mlir::Operation *definition)
    {
        auto &body = getFunction().getBody();
        auto *top = body.findAncestorOpInRegion(*taker);
        if (!top)
        {
            return true;
        }

        auto *kill = definition ? body.findAncestorOpInRegion(*definition) : nullptr;
        llvm::SmallPtrSet<mlir::Block *, 16> seen;
        llvm::SmallVector<mlir::Block *> work(top->getBlock()->getSuccessors().begin(),
                                               top->getBlock()->getSuccessors().end());
        while (!work.empty())
        {
            auto *block = work.pop_back_val();
            if (!seen.insert(block).second)
            {
                continue;
            }

            if (kill && kill->getBlock() == block &&
                (block != top->getBlock() || kill->isBeforeInBlock(top)))
            {
                continue; // the value is made again before the taker runs again
            }

            if (block == top->getBlock())
            {
                return true;
            }

            work.append(block->getSuccessors().begin(), block->getSuccessors().end());
        }

        return false;
    }

    // Can control reach `to` after `from` has run, without passing `kill` on the way? `kill` is
    // the source's definition - the producer of an SSA value, or a local's `ts.Variable` - and a
    // path through it is a new value, not this one. Straight-line order within a block, then the
    // CFG from the block's successors; a path back into `from`'s own block reaches every op in it.
    //
    // Ops nested in a region that is not the function body (an `async.execute` body, a lambda)
    // are compared by their ancestors in the body. That answers "reachable" whenever the two
    // share an ancestor, which can only turn a program into an error.
    bool reachableAfter(mlir::Operation *from, mlir::Operation *to, mlir::Operation *kill)
    {
        auto &body = getFunction().getBody();
        from = body.findAncestorOpInRegion(*from);
        to = body.findAncestorOpInRegion(*to);
        if (kill)
        {
            kill = body.findAncestorOpInRegion(*kill);
        }

        if (!from || !to)
        {
            return true;
        }

        if (from == to)
        {
            // Two distinct ops that share one ancestor statement: the order inside it is not
            // known here, and "reachable" can only turn a program into an error.
            return true;
        }

        auto *fromBlock = from->getBlock();
        auto *toBlock = to->getBlock();
        auto killIn = [&](mlir::Block *block) { return kill && kill->getBlock() == block; };

        if (fromBlock == toBlock && from->isBeforeInBlock(to))
        {
            return !(killIn(fromBlock) && from->isBeforeInBlock(kill) && kill->isBeforeInBlock(to));
        }

        // every way out of this block runs `kill` first
        if (killIn(fromBlock) && from->isBeforeInBlock(kill))
        {
            return false;
        }

        llvm::SmallPtrSet<mlir::Block *, 16> seen;
        llvm::SmallVector<mlir::Block *> work(fromBlock->getSuccessors().begin(), fromBlock->getSuccessors().end());
        while (!work.empty())
        {
            auto *block = work.pop_back_val();
            if (!seen.insert(block).second)
            {
                continue;
            }

            if (block == toBlock && (!killIn(block) || to->isBeforeInBlock(kill)))
            {
                return true;
            }

            if (killIn(block))
            {
                continue;
            }

            work.append(block->getSuccessors().begin(), block->getSuccessors().end());
        }

        return false;
    }

    // Where rc takes the receiver's reference: its `ts.Retain` of the value, when that comes
    // first in the taker's block, else the taker itself (a declaration, whose `ts.RetainSlot`
    // follows it). A release between the two - a `return`'s scope exit runs after its retain
    // and before the store into the return slot - is after the move.
    static mlir::Operation *acquirePoint(mlir::Operation *taker, mlir::Value value)
    {
        mlir::Operation *first = taker;
        for (auto *user : value.getUsers())
        {
            if (mlir::isa<mlir_ts::RetainOp>(user) && user->getBlock() == taker->getBlock() &&
                user->isBeforeInBlock(first))
            {
                first = user;
            }
        }

        return first;
    }

    // Did rc give this taker a reference of its own - a `ts.Retain` of the value in front of it,
    // or the `ts.RetainSlot` of an owning local declared from it? Only then is there a release on
    // the taker's side to give the value back once it has moved. `ts.NewInterface` is a taker to
    // this pass but a view to rc, which retains nothing for it and relies on the instance's own
    // release; moving the instance into it would leave nothing to release it at all.
    static bool takerAcquires(mlir::Operation *taker, mlir::Value value)
    {
        if (auto varOp = mlir::dyn_cast<mlir_ts::VariableOp>(taker))
        {
            return llvm::any_of(varOp.getResult().getUsers(),
                                [](mlir::Operation *user) { return mlir::isa<mlir_ts::RetainSlotOp>(user); });
        }

        return llvm::any_of(value.getUsers(), [&](mlir::Operation *user) {
            return mlir::isa<mlir_ts::RetainOp>(user) && user->getBlock() == taker->getBlock() &&
                   user->isBeforeInBlock(taker);
        });
    }

    // The declaration's name survives only as debug metadata (`--di`), the way LowerToLLVM's
    // VariableOp lowering reads it. Without it the error still points at the right place.
    static llvm::StringRef varName(mlir_ts::VariableOp varOp)
    {
        if (auto fused = mlir::dyn_cast<mlir::FusedLocWith<mlir::LLVM::DILocalVariableAttr>>(varOp.getLoc()))
        {
            if (auto name = fused.getMetadata().getName())
            {
                return name.getValue();
            }
        }

        return "this value";
    }

    static llvm::StringRef describe(mlir::Value value)
    {
        for (auto *user : value.getUsers())
        {
            if (auto varOp = mlir::dyn_cast<mlir_ts::VariableOp>(user))
            {
                return varName(varOp);
            }
        }

        return "this value";
    }

    void reportSecondReference(mlir::Operation *op)
    {
        llvm::StringRef name = "this value";
        if (auto retainSlotOp = mlir::dyn_cast<mlir_ts::RetainSlotOp>(op))
        {
            if (auto varOp = retainSlotOp.getSlot().getDefiningOp<mlir_ts::VariableOp>())
            {
                name = varName(varOp);
            }
        }
        else if (auto varOp = mlir::dyn_cast<mlir_ts::VariableOp>(op))
        {
            name = varName(varOp);
        }

        op->emitError("'") << name << "' takes a second reference; -mm=own cannot prove a move or a borrow here yet";
        signalPassFailure();
    }

    void reportUseAfterMove(mlir::Operation *use, mlir::Operation *move, llvm::StringRef name)
    {
        auto diag = use->emitError("'") << name << "' is used here after its value was moved";
        diag.attachNote(move->getLoc()) << "value moved here";
        signalPassFailure();
    }

    void reportMovedInLoop(mlir::Operation *move, llvm::StringRef name)
    {
        move->emitError("'") << name
                             << "' is moved inside a loop but was made outside it; only a borrow could do "
                                "that, and -mm=own cannot prove one yet";
        signalPassFailure();
    }

    void reportMovedOnSomePaths(mlir::Operation *move, mlir::Operation *release, llvm::StringRef name)
    {
        auto diag = move->emitError("'") << name
                                         << "' is moved here on some paths only; -mm=own cannot release it "
                                            "on the others yet";
        diag.attachNote(release->getLoc()) << "released here on every path";
        signalPassFailure();
    }
};

} // end anonymous namespace

#undef DEBUG_TYPE

std::unique_ptr<mlir::Pass> mlir_ts::createOwnershipInferencePass()
{
    return std::make_unique<OwnershipInferencePass>();
}
