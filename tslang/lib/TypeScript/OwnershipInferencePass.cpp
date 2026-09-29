#include "mlir/Pass/Pass.h"
#include "mlir/IR/Dominance.h"
#include "mlir/Dialect/LLVMIR/LLVMAttrs.h"

#include "TypeScript/TypeScriptDialect.h"
#include "TypeScript/TypeScriptOps.h"
#include "TypeScript/TypeScriptFunctionPass.h"
#include "TypeScript/Passes.h"
#include "TypeScript/Defines.h"
#include "TypeScript/MLIRLogic/MLIRTypeHelper.h"

#include "llvm/ADT/MapVector.h"
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
                if (isFresh(result) && ownsHeap(result))
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
        // Receivers of each owning local are decided together: one may move it, but where there
        // are several, each borrows it, since a borrow pins its owner.
        llvm::MapVector<mlir::Value, llvm::SmallVector<SlotReceiver>> slotReceivers;
        for (auto *op : retains)
        {
            auto value = retainedValue(op);
            if (value && proven.contains(value))
            {
                toErase.insert(op);
                continue;
            }

            if (auto load = value ? slotLoadOf(value) : mlir_ts::LoadOp())
            {
                slotReceivers[load.getReference()].push_back({op, value, load});
                continue;
            }

            if (!value || isFresh(value) == false)
            {
                reportSecondReference(op);
            }
        }

        for (auto &entry : slotReceivers)
        {
            decideSlot(entry.first, entry.second, toErase);
        }

        for (auto *op : toErase)
        {
            op->erase();
        }
    }

  private:
    mlir::DominanceInfo *dominance = nullptr;

    // While non-zero, verdicts are tried without reporting: the next verdict gets its turn first.
    unsigned quiet = 0;

    struct SlotReceiver
    {
        mlir::Operation *retain;
        mlir::Value value;
        mlir_ts::LoadOp load;
    };

    template <typename F> bool quietly(F check)
    {
        ++quiet;
        auto result = check();
        --quiet;
        return result;
    }

    // Move, else borrow, else error (spec 2.2) for the receivers of one owning local.
    void decideSlot(mlir::Value slot, llvm::ArrayRef<SlotReceiver> receivers, llvm::SetVector<mlir::Operation *> &toErase)
    {
        auto owner = varName(slot.getDefiningOp<mlir_ts::VariableOp>());
        if (receivers.size() == 1)
        {
            auto &receiver = receivers.front();
            if (quietly([&] { return checkSlotMove(receiver.retain, receiver.value, receiver.load, toErase); }))
            {
                toErase.insert(receiver.retain);
                return;
            }

            if (auto borrower = borrowerOf(receiver.retain))
            {
                tryBorrow(borrower, slotEnds(slot), owner, toErase);
                return;
            }

            checkSlotMove(receiver.retain, receiver.value, receiver.load, toErase); // reports
            return;
        }

        decideSlotReceivers(slot, receivers, owner, toErase);
    }

    // More than one receiver of one owning local: phase 1's verdict for each (Task 2 replaces it).
    void decideSlotReceivers(mlir::Value slot, llvm::ArrayRef<SlotReceiver> receivers, llvm::StringRef owner,
                             llvm::SetVector<mlir::Operation *> &toErase)
    {
        for (auto &receiver : receivers)
        {
            if (checkSlotMove(receiver.retain, receiver.value, receiver.load, toErase))
            {
                toErase.insert(receiver.retain);
            }
        }
    }

    // A `let` that could borrow: owning, declared from a value, holding a reference it took
    // itself (a `ts.RetainSlot`, not the value's own consumed one), and not captured.
    static mlir_ts::VariableOp borrowerOf(mlir::Operation *acquisition)
    {
        mlir_ts::VariableOp varOp;
        if (auto retainSlotOp = mlir::dyn_cast<mlir_ts::RetainSlotOp>(acquisition))
        {
            varOp = retainSlotOp.getSlot().getDefiningOp<mlir_ts::VariableOp>();
        }
        else
        {
            varOp = mlir::dyn_cast<mlir_ts::VariableOp>(acquisition);
        }

        if (!varOp || !varOp.getInitializer() || !isOwningVariable(varOp) ||
            varOp->hasAttr(OWNED_LOCAL_CONSUMED_ATTR_NAME) || varOp.getCaptured().value_or(false))
        {
            return {};
        }

        auto retains = llvm::any_of(varOp.getResult().getUsers(),
                                    [](mlir::Operation *user) { return mlir::isa<mlir_ts::RetainSlotOp>(user); });
        return retains ? varOp : mlir_ts::VariableOp();
    }

    // Where an owning local stops holding its value: each release of its slot and each store
    // into it.
    static llvm::SmallVector<mlir::Operation *> slotEnds(mlir::Value slot)
    {
        llvm::SmallVector<mlir::Operation *> ends;
        for (auto *user : slot.getUsers())
        {
            if (mlir::isa<mlir_ts::ReleaseSlotOp, mlir_ts::StoreOp>(user))
            {
                ends.push_back(user);
            }
        }

        return ends;
    }

    // A borrow (spec 2.2, verdict 2): `borrower` takes no reference and gives none back, so its
    // retain and every release of its slot go, and the owner keeps its own. Sound while no use of
    // the borrower can run after any of `ends` - the owner's releases and assignments - and while
    // nothing it holds is kept: not stored, returned, retained or assigned. A path back to a use
    // through the borrower's own declaration is a new borrow, not this one. Reports and returns
    // false otherwise.
    bool tryBorrow(mlir_ts::VariableOp borrower, llvm::ArrayRef<mlir::Operation *> ends, llvm::StringRef owner,
                   llvm::SetVector<mlir::Operation *> &toErase)
    {
        auto slot = borrower.getResult();
        auto name = varName(borrower);
        llvm::SmallVector<mlir::Operation *> uses;
        llvm::SmallVector<mlir::Operation *> bookkeeping;
        for (auto *user : slot.getUsers())
        {
            if (mlir::isa<mlir_ts::RetainSlotOp, mlir_ts::ReleaseSlotOp>(user))
            {
                bookkeeping.push_back(user);
                continue;
            }

            if (mlir::isa<mlir_ts::StoreOp>(user))
            {
                reportBorrowerAssigned(user, name, owner);
                return false;
            }

            auto readOp = mlir::dyn_cast<mlir_ts::LoadOp>(user);
            if (!readOp)
            {
                reportBorrowEscapes(user, name, owner);
                return false;
            }

            uses.push_back(user);
            llvm::SmallVector<mlir::Value> values{readOp.getResult()};
            while (!values.empty())
            {
                auto current = values.pop_back_val();
                for (auto *valueUser : current.getUsers())
                {
                    uses.push_back(valueUser);
                    if (auto castOp = mlir::dyn_cast<mlir_ts::CastOp>(valueUser))
                    {
                        values.push_back(castOp.getResult());
                        continue;
                    }

                    if (mlir::isa<mlir_ts::RetainOp>(valueUser) || !isBorrow(valueUser, current))
                    {
                        reportBorrowEscapes(valueUser, name, owner);
                        return false;
                    }
                }
            }
        }

        for (auto *end : ends)
        {
            for (auto *use : uses)
            {
                if (reachableAfter(end, use, borrower))
                {
                    reportBorrowOutlives(use, end, name, owner);
                    return false;
                }
            }
        }

        toErase.insert(bookkeeping.begin(), bookkeeping.end());
        return true;
    }

    // The name of a temporary that owns: a folded `const` has no ts.Variable, but under --di it
    // has a ts.DebugVariable, whose location carries the name.
    static llvm::StringRef ownerName(mlir::Value value)
    {
        for (auto *user : value.getUsers())
        {
            if (mlir::isa<mlir_ts::DebugVariableOp>(user))
            {
                return nameAt(user->getLoc());
            }
        }

        return "this value";
    }

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

    // Does a value of this type own a block that a release would destroy?
    bool ownsHeap(mlir::Value value)
    {
        MLIRTypeHelper mth(&getContext(), CompileOptions{});
        return mth.ownsHeapMemory(value.getLoc(), value.getType());
    }

    // A string constant cast to `string` is the global itself, which a release skips.
    static bool isImmortalLiteral(mlir::Value value)
    {
        auto castOp = value.getDefiningOp<mlir_ts::CastOp>();
        return castOp && mlir::isa<mlir_ts::StringType>(castOp.getType()) &&
               castOp.getIn().getDefiningOp<mlir_ts::ConstantOp>();
    }

    // A string literal or a constant array cast to its value type. The result is either the
    // immortal global itself, which a release skips, or a copy nobody else holds, which a
    // release destroys - so, owned once, it has one owner either way. `null` and `undefined`
    // widened to a nullable type (`c: C | null = null`) hold no block at all.
    static bool isLiteral(mlir::Operation *def)
    {
        auto castOp = mlir::dyn_cast<mlir_ts::CastOp>(def);
        if (!castOp)
        {
            return false;
        }

        auto *in = castOp.getIn().getDefiningOp();
        return in && mlir::isa<mlir_ts::ConstantOp, mlir_ts::NullOp, mlir_ts::UndefOp>(in);
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
        // Nothing to move: a number owns no block, and a string literal is the immortal global,
        // which any number of places may hold. Under --opt, CSE merges identical literals before
        // this pass, so one such value is routinely stored into several places.
        if (!ownsHeap(value) || isImmortalLiteral(value))
        {
            return true;
        }

        if (quietly([&] { return movesOf(value, toErase); }))
        {
            return true;
        }

        // Not a move. Every taker a `let` that can borrow, and the temporary's own release the
        // owner that bounds them (spec 2.2, verdict 2) - else the move's error stands.
        llvm::SmallVector<mlir::Operation *> takers;
        llvm::SmallVector<mlir::Operation *> releases;
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
            else if (!isBorrow(user, value))
            {
                takers.push_back(user);
            }
        }

        if (!releases.empty() && !takers.empty() &&
            llvm::all_of(takers, [](mlir::Operation *taker) { return !!borrowerOf(taker); }))
        {
            auto owner = ownerName(value);
            auto ok = true;
            for (auto *taker : takers)
            {
                ok = tryBorrow(borrowerOf(taker), releases, owner, toErase) && ok;
            }

            return ok;
        }

        return movesOf(value, toErase);
    }

    // The move verdict for an SSA value (phase 1). Reports and returns false when it is not one.
    bool movesOf(mlir::Value value, llvm::SetVector<mlir::Operation *> &toErase)
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

    // The load of an owning local that `value` was read from, through casts that keep the same
    // block (a class widened to a union with null, an upcast). None when the value is anything
    // else - a parameter, a field, a boxing cast - which phase 1 does not move out of.
    static mlir_ts::LoadOp slotLoadOf(mlir::Value value)
    {
        while (value)
        {
            auto *def = value.getDefiningOp();
            if (auto castOp = mlir::dyn_cast_or_null<mlir_ts::CastOp>(def))
            {
                if (!mlir::isa<mlir_ts::ClassType, mlir_ts::UnionType, mlir_ts::OptionalType>(castOp.getType()))
                {
                    return {};
                }

                value = castOp.getIn();
                continue;
            }

            auto loadOp = mlir::dyn_cast_or_null<mlir_ts::LoadOp>(def);
            if (!loadOp)
            {
                return {};
            }

            auto varOp = loadOp.getReference().getDefiningOp<mlir_ts::VariableOp>();
            if (!varOp || !isOwningVariable(varOp) || varOp.getCaptured().value_or(false))
            {
                return {};
            }

            return loadOp;
        }

        return {};
    }

    // The use that takes `value` for this retain: the declaration a `ts.RetainSlot` belongs to,
    // or the one use of the value after a `ts.Retain` that is not a read.
    static mlir::Operation *takerOf(mlir::Operation *retain, mlir::Value value)
    {
        if (auto retainSlotOp = mlir::dyn_cast<mlir_ts::RetainSlotOp>(retain))
        {
            return retainSlotOp.getSlot().getDefiningOp();
        }

        mlir::Operation *taker = nullptr;
        for (auto *user : value.getUsers())
        {
            if (mlir::isa<mlir_ts::RetainOp>(user) || isBorrow(user, value))
            {
                continue;
            }

            if (taker)
            {
                return nullptr; // two takers of one read: not a move this phase proves
            }

            taker = user;
        }

        return taker;
    }

    // A move out of an owning local: `let b = a`, `h.c = a`, `return a`. rc read the slot and
    // retained what it read for the receiver; the move erases that retain and every release of
    // the slot the move reaches, which must be dominated by it. Any other use of the slot the
    // move reaches - a read, an assignment - is a use after the move. Reports and returns false
    // when it is not a move.
    bool checkSlotMove(mlir::Operation *retain, mlir::Value value, mlir_ts::LoadOp load,
                       llvm::SetVector<mlir::Operation *> &toErase)
    {
        auto slot = load.getReference();
        auto varOp = slot.getDefiningOp<mlir_ts::VariableOp>();
        auto name = varName(varOp);

        auto *taker = takerOf(retain, value);
        if (!taker)
        {
            reportSecondReference(retain);
            return false;
        }

        // the move's own read, met again around a loop the slot was declared outside of
        if (reachableAfter(taker, load, varOp))
        {
            reportMovedInLoop(taker, name);
            return false;
        }

        // Every read of the slot is a use of `a`, and so is every use of what a read returned: a
        // `const b = a` is folded into its load, so `b.x` after the move reads the moved value
        // through a load that came before it.
        for (auto *use : slot.getUsers())
        {
            if (mlir::isa<mlir_ts::ReleaseSlotOp>(use))
            {
                continue;
            }

            if (use != load.getOperation() && reachableAfter(taker, use, varOp))
            {
                reportUseAfterMove(use, taker, name);
                return false;
            }

            if (auto readOp = mlir::dyn_cast<mlir_ts::LoadOp>(use))
            {
                if (auto *late = readUsedAfter(readOp, retain, taker, varOp))
                {
                    reportUseAfterMove(late, taker, name);
                    return false;
                }
            }
        }

        auto *acquire = mlir::isa<mlir_ts::RetainSlotOp>(retain) ? taker : retain;
        llvm::SmallVector<mlir::Operation *> moved;
        for (auto *use : slot.getUsers())
        {
            if (!mlir::isa<mlir_ts::ReleaseSlotOp>(use) || !reachableAfter(acquire, use, varOp))
            {
                continue;
            }

            if (!dominance->properlyDominates(acquire, use))
            {
                reportMovedOnSomePaths(taker, use, name);
                return false;
            }

            moved.push_back(use);
        }

        toErase.insert(moved.begin(), moved.end());
        return true;
    }

    // A use of what `read` returned - directly or through casts - that the move at `taker` can
    // reach, other than the move's own retain and taker. None if there is none.
    mlir::Operation *readUsedAfter(mlir_ts::LoadOp read, mlir::Operation *retain, mlir::Operation *taker,
                                   mlir::Operation *kill)
    {
        llvm::SmallVector<mlir::Value> values{read.getResult()};
        while (!values.empty())
        {
            auto current = values.pop_back_val();
            for (auto *user : current.getUsers())
            {
                if (user == retain || user == taker || mlir::isa<mlir_ts::RetainOp>(user))
                {
                    continue;
                }

                if (auto castOp = mlir::dyn_cast<mlir_ts::CastOp>(user))
                {
                    values.push_back(castOp.getResult());
                    continue;
                }

                if (reachableAfter(taker, user, kill))
                {
                    return user;
                }
            }
        }

        return nullptr;
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

    // The declaration's name survives only as debug metadata (`--di`). At this level MLIRGen's
    // form of it is a NameLoc first inside the location fused with the scope - what LowerToLLVM's
    // preserveTypesForDebugInfo reads - and after that pass it is a DILocalVariable. Without
    // `--di` the error still points at the right place.
    static llvm::StringRef varName(mlir_ts::VariableOp varOp)
    {
        return nameAt(varOp.getLoc());
    }

    static llvm::StringRef nameAt(mlir::Location location)
    {
        if (auto fused = mlir::dyn_cast<mlir::FusedLocWith<mlir::LLVM::DILocalVariableAttr>>(location))
        {
            if (auto name = fused.getMetadata().getName())
            {
                return name.getValue();
            }
        }

        if (auto scoped = mlir::dyn_cast<mlir::FusedLocWith<mlir::LLVM::DIScopeAttr>>(location))
        {
            if (auto named = mlir::dyn_cast_or_null<mlir::NameLoc>(scoped.getLocations().front()))
            {
                return named.getName().getValue();
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
        if (quiet)
        {
            return;
        }

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
        if (quiet)
        {
            return;
        }

        auto diag = use->emitError("'") << name << "' is used here after its value was moved";
        diag.attachNote(move->getLoc()) << "value moved here";
        signalPassFailure();
    }

    void reportMovedInLoop(mlir::Operation *move, llvm::StringRef name)
    {
        if (quiet)
        {
            return;
        }

        move->emitError("'") << name
                             << "' is moved inside a loop but was made outside it; only a borrow could do "
                                "that, and -mm=own cannot prove one yet";
        signalPassFailure();
    }

    void reportBorrowOutlives(mlir::Operation *use, mlir::Operation *end, llvm::StringRef name, llvm::StringRef owner)
    {
        if (quiet)
        {
            return;
        }

        auto diag = use->emitError("'") << name << "' borrows '" << owner << "' but is used here after '" << owner
                                        << "' is released or overwritten";
        diag.attachNote(end->getLoc()) << "'" << owner << "' is released or overwritten here";
        signalPassFailure();
    }

    void reportBorrowEscapes(mlir::Operation *op, llvm::StringRef name, llvm::StringRef owner)
    {
        if (quiet)
        {
            return;
        }

        op->emitError("'") << name << "' borrows '" << owner << "' and cannot be stored, returned or captured";
        signalPassFailure();
    }

    void reportBorrowerAssigned(mlir::Operation *op, llvm::StringRef name, llvm::StringRef owner)
    {
        if (quiet)
        {
            return;
        }

        op->emitError("'") << name << "' borrows '" << owner << "' and cannot be assigned; -mm=own cannot prove that yet";
        signalPassFailure();
    }

    void reportMovedOnSomePaths(mlir::Operation *move, mlir::Operation *release, llvm::StringRef name)
    {
        if (quiet)
        {
            return;
        }

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
