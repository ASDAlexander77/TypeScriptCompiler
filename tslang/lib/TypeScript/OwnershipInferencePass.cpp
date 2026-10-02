#include "mlir/Pass/Pass.h"
#include "mlir/IR/Dominance.h"
#include "mlir/Dialect/LLVMIR/LLVMAttrs.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"

#include "TypeScript/TypeScriptDialect.h"
#include "TypeScript/TypeScriptOps.h"
#include "TypeScript/TypeScriptFunctionPass.h"
#include "TypeScript/Passes.h"
#include "TypeScript/Defines.h"
#include "TypeScript/MLIRLogic/MLIRTypeHelper.h"

#include "OwnershipFacts.h"

#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/STLFunctionalExtras.h"

#define DEBUG_TYPE "own"

namespace mlir_ts = mlir::typescript;

namespace
{

using namespace own_facts;

// This pass's own mark, for its own reading: a store of a value into the capture box of a closure
// that borrows what it captures (phase 5). The store is a read, not a taker.
#define OWN_CAPTURE_BORROW_ATTR_NAME "__own_capture_borrow"

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
        // reads out of fields and elements, and results that borrow an argument, and what may
        // destroy what they read (spec 2.3, 2.4)
        llvm::SmallVector<mlir::Operation *> placeReads;
        // calls whose callee keeps an argument: each such argument moves into the call
        llvm::SmallVector<mlir::Operation *> keepingCalls;
        drops.clear();
        derivedCache.clear();
        closures.clear();
        captureStores.clear();
        captureRetains.clear();
        returnsBorrowOf = resultBorrows(f);

        // first: whether a closure borrows what it captures changes what its box's stores are
        classifyClosures();

        // A view of a block is that block: the candidate is always the root.
        f.walk([&](mlir::Operation *op) {
            if (auto retainOp = mlir::dyn_cast<mlir_ts::RetainOp>(op))
            {
                retains.push_back(op);
                candidates.insert(rootOf(retainOp.getReference()));
            }
            else if (auto retainSlotOp = mlir::dyn_cast<mlir_ts::RetainSlotOp>(op))
            {
                retains.push_back(op);
            }
            else if (auto releaseOp = mlir::dyn_cast<mlir_ts::ReleaseOp>(op))
            {
                candidates.insert(rootOf(releaseOp.getReference()));
            }
            else if (auto varOp = mlir::dyn_cast<mlir_ts::VariableOp>(op))
            {
                if (varOp.getInitializer() && isOwningVariable(varOp))
                {
                    candidates.insert(rootOf(varOp.getInitializer()));
                }
            }
            else if (mlir::isa<mlir_ts::RetainCellOp>(op))
            {
                // a box taking a cell: a closure that borrows it (decideClosures) claimed it
                if (!captureRetains.contains(op))
                {
                    op->emitError("closures that capture a variable are not supported by -mm=own here yet");
                    signalPassFailure();
                }
            }
            else if (mlir::isa<mlir_ts::DeleteOp>(op))
            {
                op->emitError("'delete' is not supported by -mm=own yet");
                signalPassFailure();
            }
            else if (auto loadOp = mlir::dyn_cast<mlir_ts::LoadOp>(op))
            {
                if (placeReadOf(loadOp.getResult()))
                {
                    placeReads.push_back(loadOp);
                }
            }
            else if (auto releaseSlotOp = mlir::dyn_cast<mlir_ts::ReleaseSlotOp>(op))
            {
                // an overwrite of a field or an element, or an assignment to a captured variable
                auto slot = releaseSlotOp.getSlot();
                if (isPlace(slot) || isCellVariable(slot) || isLoadedCell(slot))
                {
                    drops.push_back(op);
                }
            }
            else if (mlir::isa<mlir_ts::ArrayPopOp, mlir_ts::ArrayShiftOp, mlir_ts::ArraySpliceOp,
                               mlir_ts::SetLengthOfOp>(op) ||
                     isCall(op))
            {
                drops.push_back(op);
                if (op->getNumResults() == 1 && placeReadOf(op->getResult(0)) == op)
                {
                    placeReads.push_back(op);
                }

                if (!ownedParams(op).empty())
                {
                    keepingCalls.push_back(op);
                }
            }

            for (auto result : op->getResults())
            {
                if (!isView(op) && isFresh(result) && ownsHeap(result))
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

        for (auto read : placeReads)
        {
            checkPlaceRead(read, toErase);
        }

        // What is proven loses its retains; any other retain is a second reference nobody
        // proved safe - of a loaded value, a parameter, an unknown producer, or of a fresh value
        // whose own check was reported above.
        // Receivers of each owning local are decided together: one may move it, but where there
        // are several, each borrows it, since a borrow pins its owner.
        llvm::MapVector<mlir::Value, llvm::SmallVector<SlotReceiver>> slotReceivers;
        llvm::MapVector<mlir::Value, llvm::SmallVector<SlotReceiver>> paramReceivers;
        for (auto *op : retains)
        {
            if (captureRetains.contains(op))
            {
                continue; // a borrowing box's copy of a value: decideClosures decides it
            }

            auto value = retainedValue(op);
            if (value && proven.contains(value))
            {
                toErase.insert(op);
                continue;
            }

            if (value && placeReadOf(value))
            {
                continue; // a read out of a container: checkPlaceRead decided it
            }

            // rc's reference for what keeps a parameter this function keeps
            if (auto load = value ? keptParamLoadOf(value) : mlir_ts::LoadOp())
            {
                paramReceivers[load.getReference()].push_back({op, value, load});
                continue;
            }

            // rc's reference for the caller, on a parameter this function returns a borrow of
            if (value && returnsBorrowOf >= 0 && borrowedParam(value) == returnsBorrowOf)
            {
                if (returnedParams.insert(value).second)
                {
                    checkReturnedParam(value, toErase);
                }

                continue;
            }

            if (auto load = value ? slotLoadOf(value) : mlir_ts::LoadOp())
            {
                slotReceivers[load.getReference()].push_back({op, value, load});
                continue;
            }

            if (value && isReadOutOfException(value))
            {
                toErase.insert(op);
                continue;
            }

            if (!value || isFresh(value) == false)
            {
                reportSecondReference(op);
            }
        }

        // An argument a callee keeps moves into the call (spec 2.4): out of an owning local or a
        // parameter this function keeps, like any other taker; a fresh value is decided with the
        // other takers of it, a read out of a container by its own check. Anything else is not
        // this function's to give.
        for (auto *call : keepingCalls)
        {
            auto args = callArgs(call);
            for (auto index : ownedParams(call))
            {
                if (static_cast<size_t>(index) >= args.size() || holdsNoBlock(args[index]) ||
                    isImmortalLiteral(args[index]))
                {
                    continue;
                }

                auto arg = args[index];
                auto root = rootOf(arg);
                if (auto load = slotLoadOf(arg))
                {
                    slotReceivers[load.getReference()].push_back({call, root, load});
                }
                else if (auto load = keptParamLoadOf(arg))
                {
                    paramReceivers[load.getReference()].push_back({call, root, load});
                }
                else if (!isFresh(root) && !placeReadOf(root))
                {
                    reportGivenNotOwned(call, arg);
                }
            }
        }

        for (auto &entry : slotReceivers)
        {
            decideSlot(entry.first, entry.second, toErase);
        }

        for (auto &entry : paramReceivers)
        {
            decideParam(entry.first, entry.second, toErase);
        }

        decideClosures(toErase);
        decideCells();

        for (auto *op : toErase)
        {
            op->erase();
        }

        // the facts were for this pass only
        returnedParams.clear();
        f.walk([](mlir::Operation *op) {
            for (auto *name : {OWN_PARAMS_ATTR_NAME, OWN_RESULT_BORROWS_ATTR_NAME, OWN_NO_DROPS_ATTR_NAME,
                               OWN_FACTS_LOST_ATTR_NAME, OWN_FRESH_RESULT_ATTR_NAME, OWN_ASSIGNS_CAPTURES_ATTR_NAME,
                               OWN_CAPTURE_BORROW_ATTR_NAME})
            {
                op->removeAttr(name);
            }
        });
    }

  private:
    mlir::DominanceInfo *dominance = nullptr;

    // What in this function may destroy a value held by a field or an element: an overwrite of a
    // field or an element, an array op that removes elements, a call.
    llvm::SmallVector<mlir::Operation *> drops;

    // derivedFrom's answer for each value it was asked about in this function; held by pointer, so
    // an answer stays put while the next one is added
    llvm::DenseMap<mlir::Value, std::shared_ptr<llvm::DenseSet<mlir::Value>>> derivedCache;

    // While non-zero, verdicts are tried without reporting: the next verdict gets its turn first.
    unsigned quiet = 0;

    // The parameter this function's result borrows (`__own_result_borrows`), or -1; and the
    // parameters (or views of them) whose rc reference for the caller was already decided.
    int returnsBorrowOf = -1;
    llvm::DenseSet<mlir::Value> returnedParams;

    // ---- Closures (spec 2.5, phase 5) ----
    //
    // A closure is `ts.CreateBoundFunction(box, @f) {__owns_capture}`; its box holds a cell for each
    // variable it captures by reference and a copy of each it captures by value. One that does not
    // escape borrows all of it: rc's references for the box go, the box frees itself only
    // (OWN_BORROWS_CAPTURES_ATTR_NAME), and nothing that may run the closure may come after anything
    // it borrows is given back. One that escapes would own all of it, which is not done yet.
    struct Closure
    {
        mlir_ts::CreateBoundFunctionOp op;
        mlir::Value box;
        // each store that fills the box, with the field it fills
        llvm::SmallVector<std::pair<int64_t, mlir_ts::StoreOp>> fills;
        bool hasCells = false;
        // the first use that takes the closure out of this function's hands
        mlir::Operation *escape = nullptr;
        // what may run the closure: calls given it, called through it, or through its box
        llvm::SmallVector<mlir::Operation *> uses;
        // the calls through the box itself, made without the closure value (a folded `const f`)
        llvm::SmallVector<mlir::Operation *> boxUses;
        // where the closure value is given back: its releases, and those of the `let`s holding it
        llvm::SmallVector<mlir::Operation *> ends;
        // rc's references for the box: `ts.RetainCell` of each cell, `ts.Retain` of each copy
        llvm::SmallVector<mlir::Operation *> retains;
        // held by a local that owns nothing (`let f: () => number; f = () => ...`): an alias, so
        // the closure's own releases stay its owner
        bool aliased = false;
    };

    llvm::SmallVector<std::shared_ptr<Closure>> closures;
    // a copy stored into a borrowing box -> its closure
    llvm::DenseMap<mlir::Operation *, Closure *> captureStores;
    // rc's references for boxes, which the closures decide
    llvm::DenseSet<mlir::Operation *> captureRetains;

    // Finds the closures over cells and what they borrow, before anything else: whether a box's
    // stores are reads or takers depends on it. A closure with copies only keeps what phase 4 does
    // with it - its box owns them, and each copy moves in.
    void classifyClosures()
    {
        getFunction().walk([&](mlir_ts::CreateBoundFunctionOp boundOp) {
            if (!boundOp->hasAttr(OWNS_CAPTURE_ATTR_NAME) || closureOfBox(boundOp.getThisVal()) != boundOp)
            {
                return;
            }

            auto closure = std::make_shared<Closure>();
            closure->op = boundOp;
            closure->box = boundOp.getThisVal();
            for (auto *user : closure->box.getUsers())
            {
                if (user == boundOp.getOperation())
                {
                    continue;
                }

                if (auto propertyRefOp = mlir::dyn_cast<mlir_ts::PropertyRefOp>(user))
                {
                    for (auto *fieldUser : propertyRefOp->getUsers())
                    {
                        auto storeOp = mlir::dyn_cast<mlir_ts::StoreOp>(fieldUser);
                        if (!storeOp || storeOp.getReference() != propertyRefOp.getResult())
                        {
                            closure->escape = closure->escape ? closure->escape : fieldUser;
                            continue;
                        }

                        closure->fills.push_back({propertyRefOp.getPosition(), storeOp});
                        closure->hasCells = closure->hasCells || mlir::isa<mlir_ts::RefType>(storeOp.getValue().getType());
                    }

                    continue;
                }

                if (isCall(user))
                {
                    closure->uses.push_back(user);
                    closure->boxUses.push_back(user);
                    continue;
                }

                closure->escape = closure->escape ? closure->escape : user;
            }

            if (auto *escape = followClosure(boundOp.getResult(), closure->uses, closure->ends, closure->aliased);
                escape && !closure->escape)
            {
                closure->escape = escape;
            }

            // copies only: the box owns them (phase 4); only an alias needs deciding
            if (!closure->hasCells)
            {
                if (!closure->escape && closure->aliased)
                {
                    closures.push_back(std::move(closure));
                }

                return;
            }

            for (auto &fill : closure->fills)
            {
                auto storeOp = fill.second;
                if (auto *retain = retainBefore(storeOp.getValue(), storeOp))
                {
                    closure->retains.push_back(retain);
                    captureRetains.insert(retain);
                }

                if (!closure->escape && !mlir::isa<mlir_ts::RefType>(storeOp.getValue().getType()))
                {
                    storeOp->setAttr(OWN_CAPTURE_BORROW_ATTR_NAME, mlir::UnitAttr::get(&getContext()));
                    captureStores[storeOp] = closure.get();
                }
            }

            if (!closure->escape)
            {
                boundOp->setAttr(OWN_BORROWS_CAPTURES_ATTR_NAME, mlir::UnitAttr::get(&getContext()));
            }

            closures.push_back(std::move(closure));
        });
    }

    // rc's reference for the box's copy of `value`: the retain of it nearest before the store that
    // fills the box, in the same block.
    mlir::Operation *retainBefore(mlir::Value value, mlir::Operation *store)
    {
        mlir::Operation *found = nullptr;
        for (auto *user : value.getUsers())
        {
            if (!mlir::isa<mlir_ts::RetainOp, mlir_ts::RetainCellOp>(user) || user->getBlock() != store->getBlock() ||
                !user->isBeforeInBlock(store) || captureRetains.contains(user))
            {
                continue;
            }

            if (!found || found->isBeforeInBlock(user))
            {
                found = user;
            }
        }

        return found;
    }

    // Where a closure value goes: through its views and the owning `let`s that hold it, each use
    // that may run it joins `uses`, and each release `ends`. Answers the first use that takes it
    // anywhere else - returned, stored, kept by a callee, captured - or null. Anything this does
    // not know is such a use.
    mlir::Operation *followClosure(mlir::Value closure, llvm::SmallVectorImpl<mlir::Operation *> &uses,
                                   llvm::SmallVectorImpl<mlir::Operation *> &ends, bool &aliased)
    {
        llvm::SmallVector<mlir::Value> values{closure};
        llvm::SmallPtrSet<mlir::Value, 8> seen;
        mlir::Operation *escape = nullptr;
        auto followSlot = [&](mlir::Value slot) {
            for (auto *slotUser : slot.getUsers())
            {
                if (mlir::isa<mlir_ts::RetainSlotOp, mlir_ts::DebugVariableOp>(slotUser))
                {
                    continue;
                }

                if (mlir::isa<mlir_ts::ReleaseSlotOp>(slotUser))
                {
                    ends.push_back(slotUser);
                }
                else if (auto loadOp = mlir::dyn_cast<mlir_ts::LoadOp>(slotUser))
                {
                    values.push_back(loadOp.getResult());
                }
                else if (auto storeOp = mlir::dyn_cast<mlir_ts::StoreOp>(slotUser); !storeOp || storeOp.getReference() != slot)
                {
                    escape = escape ? escape : slotUser;
                }
            }
        };

        while (!values.empty() && !escape)
        {
            auto value = values.pop_back_val();
            if (!seen.insert(value).second)
            {
                continue;
            }

            forEachUse(value, [&](mlir::Operation *user, mlir::Value used) {
                if (escape || mlir::isa<mlir_ts::RetainOp, mlir_ts::DebugVariableOp>(user))
                {
                    return;
                }

                if (mlir::isa<mlir_ts::ReleaseOp>(user))
                {
                    ends.push_back(user);
                    return;
                }

                if (isCall(user))
                {
                    if (isKeptArgument(user, used))
                    {
                        escape = user;
                    }
                    else
                    {
                        uses.push_back(user);
                    }

                    return;
                }

                // the function and the box a call through the closure is made with
                if (mlir::isa<mlir_ts::GetThisOp, mlir_ts::GetMethodOp>(user))
                {
                    for (auto *partUser : user->getResult(0).getUsers())
                    {
                        if (isCall(partUser) && !isKeptArgument(partUser, user->getResult(0)))
                        {
                            uses.push_back(partUser);
                        }
                        else
                        {
                            escape = escape ? escape : partUser;
                        }
                    }

                    return;
                }

                mlir::Value slot;
                if (auto varOp = mlir::dyn_cast<mlir_ts::VariableOp>(user))
                {
                    slot = varOp.getResult();
                }
                else if (auto storeOp = mlir::dyn_cast<mlir_ts::StoreOp>(user); storeOp && storeOp.getValue() == used)
                {
                    slot = storeOp.getReference();
                }

                // a local, owning or not (`let f: () => number;` owns nothing): its reads are the
                // closure; a function's result slot is a local too, and its read is returned
                auto slotVar = slot ? slot.getDefiningOp<mlir_ts::VariableOp>() : mlir_ts::VariableOp();
                if (slotVar && !slotVar.getCaptured().value_or(false))
                {
                    aliased = aliased || !isOwningVariable(slotVar);
                    followSlot(slot);
                    return;
                }

                escape = user;
            });
        }

        return escape;
    }

    // Does a call given this value give it a closure, which may run and touch what its box holds?
    static bool holdsCaptures(mlir::Value value)
    {
        if (mlir::isa<mlir_ts::BoundFunctionType, mlir_ts::HybridFunctionType>(value.getType()))
        {
            return true;
        }

        auto root = rootOf(value);
        if (auto varOp = root.getDefiningOp<mlir_ts::VariableOp>())
        {
            return varOp->hasAttr(CAPTURE_BOX_ATTR_NAME);
        }

        // a closure's `this` is its box, a reference; a bound method's is the object
        auto getThisOp = root.getDefiningOp<mlir_ts::GetThisOp>();
        return getThisOp && mlir::isa<mlir_ts::RefType>(getThisOp.getType());
    }

    // The verdicts on the closures over cells: an escape is an error; a borrow must not be run after
    // anything it borrows is given back - a cell this function owns at its releases, a copy at its
    // owner's ends. A cell an enclosing closure holds and a parameter end nowhere here; a copy of a
    // read out of a container is bounded by the read's own check, which sees the closure's uses
    // (walkBorrowed).
    void decideClosures(llvm::SetVector<mlir::Operation *> &toErase)
    {
        for (auto &closure : closures)
        {
            auto *boundOp = closure->op.getOperation();
            auto name = ownerName(closure->op.getResult());
            auto ok = true;
            // Held by a local that owns nothing: rc's birth reference reads to the move logic as the
            // store's, which would leave the box with no owner. The closure keeps its releases, and
            // nothing may run it after them.
            auto keepAliasedReleases = [&]() {
                for (auto *end : closure->ends)
                {
                    if (!mlir::isa<mlir_ts::ReleaseOp>(end))
                    {
                        continue;
                    }

                    toErase.remove(end);
                    for (auto *use : closure->uses)
                    {
                        if (ok && reachableAfter(end, use, boundOp))
                        {
                            reportClosureOutlives(use, end, name, name);
                            ok = false;
                        }
                    }
                }
            };

            if (!closure->hasCells)
            {
                keepAliasedReleases();
                continue;
            }

            if (closure->escape)
            {
                reportClosureEscapes(closure->escape, boundOp);
                continue;
            }

            for (auto &fill : closure->fills)
            {
                auto value = fill.second.getValue();
                llvm::SmallVector<mlir::Operation *> ends;
                llvm::StringRef captured = "this value";
                if (mlir::isa<mlir_ts::RefType>(value.getType()))
                {
                    if (isCellVariable(value))
                    {
                        captured = varName(value.getDefiningOp<mlir_ts::VariableOp>());
                        for (auto *user : value.getUsers())
                        {
                            if (mlir::isa<mlir_ts::ReleaseCellOp>(user))
                            {
                                ends.push_back(user);
                            }
                        }
                    }
                    else if (!isLoadedCell(value))
                    {
                        reportCaptureNotOwned(fill.second, captured);
                        ok = false;
                        continue;
                    }
                }
                else
                {
                    auto root = rootOf(value);
                    captured = ownerName(root);
                    Chain chain;
                    if (holdsNoBlock(value) || placeReadOf(root) || borrowedParam(root) >= 0)
                    {
                        // nothing to own, a read the read's check bounds, or a parameter
                    }
                    else if (auto slotLoad = slotLoadOf(root))
                    {
                        chain.roots.push_back({slotLoad.getReference(), Chain::Slot});
                    }
                    else if (isFresh(root))
                    {
                        chain.roots.push_back({root, Chain::Owned});
                    }
                    else
                    {
                        reportCaptureNotOwned(fill.second, captured);
                        ok = false;
                        continue;
                    }

                    ends = rootEnds(chain);
                }

                for (auto *end : ends)
                {
                    if (captureRetains.contains(end) || captureStores.count(end))
                    {
                        continue; // a box's own copy, this one's or another borrowing closure's
                    }

                    for (auto *use : closure->uses)
                    {
                        if (ok && reachableAfter(end, use, boundOp))
                        {
                            reportClosureOutlives(use, end, name, captured);
                            ok = false;
                        }
                    }
                }
            }

            if (closure->aliased)
            {
                keepAliasedReleases();
            }

            // a call through the box, after the closure value that owns the box is given back
            for (auto *end : closure->ends)
            {
                for (auto *use : closure->boxUses)
                {
                    if (ok && reachableAfter(end, use, boundOp))
                    {
                        reportClosureOutlives(use, end, name, name);
                        ok = false;
                    }
                }
            }

            toErase.insert(closure->retains.begin(), closure->retains.end());
        }
    }

    // A captured variable whose value this function does not own - a parameter, `this` - has a cell
    // that borrows the value: the cell's release frees the cell only (OWN_CELL_BORROWS_ATTR_NAME),
    // and nothing may assign the variable, here or in a closure over it, since the old value is the
    // caller's. A variable whose value nobody can be said to own is an error.
    void decideCells()
    {
        getFunction().walk([&](mlir_ts::VariableOp varOp) {
            auto cell = varOp.getResult();
            if (!isCellVariable(cell) || isOwningVariable(varOp))
            {
                return;
            }

            MLIRTypeHelper mth(&getContext(), CompileOptions{});
            if (!mth.ownsHeapMemory(varOp.getLoc(), mlir::cast<mlir_ts::RefType>(cell.getType()).getElementType()))
            {
                return;
            }

            auto name = varName(varOp);
            auto init = varOp.getInitializer();
            auto argument = init ? mlir::dyn_cast<mlir::BlockArgument>(init) : mlir::BlockArgument();
            if (!argument || !argument.getOwner()->isEntryBlock() ||
                !mlir::isa<mlir_ts::FuncOp>(argument.getOwner()->getParentOp()))
            {
                reportCaptureNotOwned(varOp, name);
                return;
            }

            for (auto *user : cell.getUsers())
            {
                if (mlir::isa<mlir_ts::ReleaseCellOp>(user))
                {
                    user->setAttr(OWN_CELL_BORROWS_ATTR_NAME, mlir::UnitAttr::get(&getContext()));
                    continue;
                }

                auto storeOp = mlir::dyn_cast<mlir_ts::StoreOp>(user);
                if (mlir::isa<mlir_ts::ReleaseSlotOp>(user) || (storeOp && storeOp.getReference() == cell))
                {
                    reportCapturedParamAssigned(user, name);
                    return;
                }

                // captured: the closure's body must not assign it either
                auto propertyRefOp = storeOp ? storeOp.getReference().getDefiningOp<mlir_ts::PropertyRefOp>()
                                             : mlir_ts::PropertyRefOp();
                auto boundOp = propertyRefOp ? closureOfBox(propertyRefOp.getObjectRef()) : mlir_ts::CreateBoundFunctionOp();
                if (!boundOp)
                {
                    continue;
                }

                auto assigned = boundOp->getAttrOfType<mlir::DenseI32ArrayAttr>(OWN_ASSIGNS_CAPTURES_ATTR_NAME);
                if (assigned && llvm::is_contained(assigned.asArrayRef(), static_cast<int32_t>(propertyRefOp.getPosition())))
                {
                    reportCapturedParamAssigned(boundOp, name);
                    return;
                }
            }
        });
    }

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
                // rc's retain goes with the move; a call that keeps the argument is the move itself
                if (!isCall(receiver.retain))
                {
                    toErase.insert(receiver.retain);
                }

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

    // A parameter this function keeps (`__own_params`): its caller moved the argument in, and this
    // function has no release for it, so exactly one taker must move it on, on every way out of
    // the function, and nothing may read it after that.
    void decideParam(mlir::Value slot, llvm::ArrayRef<SlotReceiver> receivers, llvm::SetVector<mlir::Operation *> &toErase)
    {
        auto name = varName(slot.getDefiningOp<mlir_ts::VariableOp>());
        auto &receiver = receivers.front();
        auto *taker = takerOf(receiver.retain, receiver.value);
        if (receivers.size() > 1)
        {
            auto &second = receivers[1];
            auto *secondTaker = takerOf(second.retain, second.value);
            reportUseAfterMove(secondTaker ? secondTaker : second.retain, taker ? taker : receiver.retain, name);
            return;
        }

        if (!checkSlotMove(receiver.retain, receiver.value, receiver.load, toErase))
        {
            return;
        }

        auto &body = getFunction().getBody();
        auto *moved = body.findAncestorOpInRegion(*taker);
        auto everyPath = true;
        getFunction().walk([&](mlir_ts::ReturnInternalOp returnOp) {
            auto *exit = body.findAncestorOpInRegion(*returnOp);
            if (everyPath && (!moved || !exit || !dominance->dominates(moved, exit)))
            {
                reportKeptOnSomePaths(taker, returnOp, name);
                everyPath = false;
            }
        });

        if (everyPath && mlir::isa<mlir_ts::RetainOp>(receiver.retain))
        {
            toErase.insert(receiver.retain);
        }
    }

    // The thrown value, read out of the exception object by a catch's copy thunk (`.eh.copy.*`,
    // which the C++ runtime calls to copy the exception into the catch). rc retains it for the
    // copy; own moves it: the exception object never gives its reference back (a throw hands the
    // thrower's reference to it, and nothing releases it - rc leaks it the same way), and each
    // exception is copied into one catch, since a TypeScript rethrow is a new throw.
    bool isReadOutOfException(mlir::Value value)
    {
        auto loadOp = rootOf(value).getDefiningOp<mlir_ts::LoadOp>();
        auto argument = loadOp ? mlir::dyn_cast<mlir::BlockArgument>(loadOp.getReference()) : mlir::BlockArgument();
        return argument && argument.getOwner()->isEntryBlock() && getFunction().getSymName().starts_with(".eh.copy.");
    }

    // The read of a parameter this function keeps, through its views. None for anything else.
    mlir_ts::LoadOp keptParamLoadOf(mlir::Value value)
    {
        auto loadOp = rootOf(value).getDefiningOp<mlir_ts::LoadOp>();
        auto varOp = loadOp ? loadOp.getReference().getDefiningOp<mlir_ts::VariableOp>() : mlir_ts::VariableOp();
        auto index = -1;
        if (!varOp || !isParameterSlot(varOp, index) || !llvm::is_contained(ownedParams(getFunction()), index))
        {
            return {};
        }

        return loadOp;
    }

    // More than one receiver of one owning local. Each has to borrow: a borrow pins the owner, so
    // none may move it. A receiver that is not a `let` reports its move error - another
    // receiver's reads come after it - or, if its move alone would pass, that a borrow pins it.
    void decideSlotReceivers(mlir::Value slot, llvm::ArrayRef<SlotReceiver> receivers, llvm::StringRef owner,
                             llvm::SetVector<mlir::Operation *> &toErase)
    {
        auto ends = slotEnds(slot);
        mlir_ts::VariableOp someBorrower;
        for (auto &receiver : receivers)
        {
            if (auto borrower = borrowerOf(receiver.retain))
            {
                someBorrower = borrower;
                tryBorrow(borrower, ends, owner, toErase);
            }
        }

        for (auto &receiver : receivers)
        {
            if (borrowerOf(receiver.retain))
            {
                continue;
            }

            llvm::SetVector<mlir::Operation *> unused;
            if (checkSlotMove(receiver.retain, receiver.value, receiver.load, unused))
            {
                reportMovedWhileBorrowed(receiver.retain, owner, someBorrower ? varName(someBorrower) : "another variable");
            }
        }
    }

    // A `let` that could borrow: owning, declared from a value, holding a reference it took
    // itself (a `ts.RetainSlot`, not the value's own consumed one), and not captured. With
    // `consumed`, also one that took over the reference rc thought the value carried: a call's
    // result that turned out to borrow an argument carries none.
    static mlir_ts::VariableOp borrowerOf(mlir::Operation *acquisition, bool consumed = false)
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

        if (!varOp || !varOp.getInitializer() || !isOwningVariable(varOp) || varOp.getCaptured().value_or(false))
        {
            return {};
        }

        if (varOp->hasAttr(OWNED_LOCAL_CONSUMED_ATTR_NAME))
        {
            return consumed ? varOp : mlir_ts::VariableOp();
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
        auto name = varName(borrower);
        llvm::SmallVector<BorrowUse> uses;
        llvm::SmallVector<mlir::Operation *> bookkeeping;
        mlir::Operation *declaration = borrower;
        if (auto *kept = collectBorrowerUses(borrower, declaration, uses, bookkeeping))
        {
            if (isAssignmentOf(kept, borrower))
            {
                reportBorrowerAssigned(kept, name, owner);
            }
            else
            {
                reportBorrowEscapes(kept, name, owner);
            }

            return false;
        }

        for (auto *end : ends)
        {
            for (auto &use : uses)
            {
                if (reachableAfter(end, use.op, use.kills))
                {
                    reportBorrowOutlives(use.op, end, name, owner);
                    return false;
                }
            }
        }

        toErase.insert(bookkeeping.begin(), bookkeeping.end());
        return true;
    }

    // A use of a borrowed value, and where the value it sees is made again: a path from a release
    // to the use through one of `kills` reaches a new value, not this one. For the borrow's own
    // uses that is the read or the borrowing `let`; for a use through a merge it is the merge's
    // block, and through a local that owns nothing, that local's declaration and assignments.
    struct BorrowUse
    {
        mlir::Operation *op;
        llvm::SmallVector<mlir::Operation *, 2> kills;
        // the function value an indirect call goes through: read when the call starts, not while
        // the callee runs
        bool callee = false;
    };

    // A borrowing `let`'s slot: each read, and everything it leads to (walkBorrowed), joins
    // `uses`, made again at `kills`; its `ts.RetainSlot` and `ts.ReleaseSlot`s are `bookkeeping`,
    // which a borrow erases. Answers the first use that would keep what the slot holds - an
    // assignment, or anything else that is not a read - or null.
    mlir::Operation *collectBorrowerUses(mlir_ts::VariableOp borrower, llvm::ArrayRef<mlir::Operation *> kills,
                                         llvm::SmallVectorImpl<BorrowUse> &uses,
                                         llvm::SmallVectorImpl<mlir::Operation *> &bookkeeping)
    {
        for (auto *user : borrower.getResult().getUsers())
        {
            if (mlir::isa<mlir_ts::RetainSlotOp, mlir_ts::ReleaseSlotOp>(user))
            {
                bookkeeping.push_back(user);
                continue;
            }

            auto readOp = mlir::dyn_cast<mlir_ts::LoadOp>(user);
            if (!readOp)
            {
                return user;
            }

            uses.push_back({user, {kills.begin(), kills.end()}});
            if (auto *kept = walkBorrowed(readOp.getResult(), kills, uses))
            {
                return kept;
            }
        }

        return nullptr;
    }

    static bool isAssignmentOf(mlir::Operation *op, mlir_ts::VariableOp varOp)
    {
        auto storeOp = mlir::dyn_cast<mlir_ts::StoreOp>(op);
        return storeOp && storeOp.getReference() == varOp.getResult();
    }

    // What `use` hands the very value it uses on to, as that value: a cast or a view of it, the
    // argument of the block a branch passes it to (a merge: `a ?? b`), or the reads of a local
    // that owns nothing and is assigned it (`for (x of arr)` into an outer `let x`). Answers
    // whether the use is one of those, even where nothing comes out of it. Where the value lands
    // somewhere that is replaced - a block argument, a local - `renewedAt` gets what replaces it:
    // the block's entry, the local's declaration and assignments.
    static bool passesOn(mlir::OpOperand &use, llvm::SmallVectorImpl<mlir::Value> &values,
                         llvm::SmallVectorImpl<mlir::Operation *> *renewedAt = nullptr)
    {
        auto *user = use.getOwner();
        if (mlir::isa<mlir_ts::CastOp>(user) || isView(user))
        {
            values.push_back(user->getResult(0));
            return true;
        }

        if (auto branchOp = mlir::dyn_cast<mlir::BranchOpInterface>(user))
        {
            if (auto argument = branchOp.getSuccessorBlockArgument(use.getOperandNumber()))
            {
                values.push_back(*argument);
                if (renewedAt)
                {
                    renewedAt->push_back(&argument->getOwner()->front());
                }

                return true;
            }

            return false;
        }

        if (auto storeOp = mlir::dyn_cast<mlir_ts::StoreOp>(user); storeOp && storeOp.getValue() == use.get())
        {
            if (auto varOp = storeOp.getReference().getDefiningOp<mlir_ts::VariableOp>();
                varOp && !isOwningVariable(varOp) && !varOp.getCaptured().value_or(false))
            {
                if (renewedAt)
                {
                    renewedAt->push_back(varOp);
                }

                for (auto *varUser : varOp.getResult().getUsers())
                {
                    if (auto loadOp = mlir::dyn_cast<mlir_ts::LoadOp>(varUser))
                    {
                        values.push_back(loadOp.getResult());
                    }
                    else if (renewedAt && isAssignmentOf(varUser, varOp))
                    {
                        renewedAt->push_back(varUser);
                    }
                }

                return true;
            }
        }

        return false;
    }

    // Where a borrowed value goes. Every use of it, and of what it produces that may still point
    // into the block, joins `uses`, made again at `kills`. The value itself - through what passesOn
    // hands it on to - must not be kept anywhere; what it produces - a field reference, a value
    // loaded through one, a bound method, a catch clause's non-owning local - still points into
    // the borrowed block, so each of its uses is a use of the borrow too. A value that cannot point
    // anywhere (a number, a boolean) ends the walk. Where passesOn hands the value to something
    // that is replaced, the uses beyond are made again where it is replaced instead; the op that
    // hands it on is itself a use, bounded by `kills`. `isBorrower` accepts a use of the value
    // itself that the caller follows on its own (a borrowing `let`), made again at the kills it is
    // given. Answers the first use that would keep the value.
    //
    // `keeping` names what rc planted around a borrow that it took for an owner and that goes
    // with it: the temporary release of a result that borrows an argument (the start's, or that of
    // a call it is passed to, which is a borrow of it too), and, in a function whose result
    // borrows this very value's parameter, the retain rc made for the caller. Those join
    // `bookkeeping`, not `uses`, and the return itself is a use, not a keep.
    struct Keeping
    {
        // no default member initializers: GCC rejects them in a default argument of the class
        // that encloses the struct
        llvm::SmallVectorImpl<mlir::Operation *> *bookkeeping;
        bool returns;
    };

    mlir::Operation *walkBorrowed(
        mlir::Value start, llvm::ArrayRef<mlir::Operation *> kills, llvm::SmallVectorImpl<BorrowUse> &uses,
        llvm::function_ref<bool(mlir::Operation *, llvm::ArrayRef<mlir::Operation *>)> isBorrower = nullptr,
        Keeping keeping = Keeping{nullptr, false})
    {
        struct Pending
        {
            mlir::Value value;
            bool borrowed;
            llvm::SmallVector<mlir::Operation *, 2> kills;
        };

        llvm::SmallVector<Pending> values;
        values.push_back({start, true, {kills.begin(), kills.end()}});
        llvm::SmallPtrSet<mlir::Value, 16> seen{start};
        while (!values.empty())
        {
            auto pending = values.pop_back_val();
            for (auto &use : pending.value.getUses())
            {
                auto *user = use.getOwner();
                if (pending.borrowed && keeping.bookkeeping &&
                    ((mlir::isa<mlir_ts::ReleaseOp>(user) && borrowingCall(rootOf(pending.value))) ||
                     (keeping.returns && mlir::isa<mlir_ts::RetainOp>(user))))
                {
                    keeping.bookkeeping->push_back(user);
                    continue;
                }

                // rc's reference for a borrowing box's copy goes with the closure (decideClosures)
                if (captureRetains.contains(user))
                {
                    continue;
                }

                uses.push_back({user, pending.kills, isCall(user) && isCalleeOperand(use)});
                // copied into a borrowing box: whatever runs the closure reads it
                if (auto found = captureStores.find(user); found != captureStores.end())
                {
                    for (auto *closureUse : found->second->uses)
                    {
                        uses.push_back({closureUse, pending.kills});
                    }
                }

                if (pending.borrowed && keeping.returns && mlir::isa<mlir_ts::ReturnInternalOp>(user))
                {
                    continue;
                }

                if (pending.borrowed)
                {
                    llvm::SmallVector<mlir::Value> same;
                    llvm::SmallVector<mlir::Operation *, 2> renewedAt;
                    if (passesOn(use, same, &renewedAt))
                    {
                        for (auto value : same)
                        {
                            if (mayPointInto(value) && seen.insert(value).second)
                            {
                                values.push_back({value, true, renewedAt.empty() ? pending.kills : renewedAt});
                            }
                        }

                        continue;
                    }

                    if (isBorrower && isBorrower(user, pending.kills))
                    {
                        continue;
                    }

                    if (mlir::isa<mlir_ts::RetainOp>(user) || !isBorrow(user, pending.value))
                    {
                        return user;
                    }
                }

                // A result that borrows what it was given borrows this too, and is walked as it.
                for (auto result : user->getResults())
                {
                    if (mayPointInto(result) && seen.insert(result).second)
                    {
                        values.push_back({result, !!borrowingCall(result), pending.kills});
                    }
                }
            }
        }

        return nullptr;
    }

    // Is this what an indirect call goes through: its callee value, or, for a method read out of
    // a field of an object (`o.m()`, `m` a field holding a function), the object it is bound to,
    // which is the container of what was read, not what was borrowed. A method of a class bound
    // to a borrowed instance (`h.c.m()`) is not one: its `this` is the borrow.
    static bool isCalleeOperand(mlir::OpOperand &use)
    {
        auto *user = use.getOwner();
        if (!mlir::isa<mlir_ts::CallInternalOp, mlir_ts::CallIndirectOp, mlir_ts::CallHybridInternalOp>(user))
        {
            return false;
        }

        if (use.getOperandNumber() == 0)
        {
            return true;
        }

        auto getThisOp = use.get().getDefiningOp<mlir_ts::GetThisOp>();
        auto getMethodOp = user->getOperand(0).getDefiningOp<mlir_ts::GetMethodOp>();
        auto methodField = getThisOp ? getThisOp.getOperand().getDefiningOp<mlir_ts::LoadOp>() : mlir_ts::LoadOp();
        return use.getOperandNumber() == 1 && getMethodOp && methodField &&
               getThisOp.getOperand() == getMethodOp.getBoundFunc() &&
               mlir::isa<mlir_ts::BoundRefType>(methodField.getReference().getType());
    }

    // Can a value of this type point into a heap block - a reference into one, a block of its own,
    // or a method bound to one? A number or a boolean read out of a borrowed block cannot.
    bool mayPointInto(mlir::Value value)
    {
        auto type = value.getType();
        return mlir::isa<mlir_ts::RefType, mlir_ts::BoundRefType, mlir_ts::ValueRefType, mlir_ts::BoundFunctionType,
                         mlir_ts::HybridFunctionType, mlir_ts::ExtensionFunctionType, mlir_ts::OpaqueType>(type) ||
               ownsHeap(value);
    }

    // ---- Reads out of containers (spec 2.3, phase 3) ----
    //
    // `h.c` and `arr[i]` read a value the container owns. Such a read is a borrow: it may not be
    // kept, and none of its uses - nor any use of what it produces - may run after something that
    // may destroy it (spec 2.2's dropping mutations): an overwrite of a place on the way from the
    // read up to its root, an array op that removes elements, an end of the root itself, or a call
    // that may reach the root. A path back to a use through the read itself is a new borrow.

    // A read of a heap value out of a field or an element, or a call whose result borrows an
    // argument (spec 2.4, phase 4), seen through its views. Null for anything else.
    mlir::Operation *placeReadOf(mlir::Value value)
    {
        auto root = rootOf(value);
        auto *def = root.getDefiningOp();
        if (!def || !ownsHeap(root))
        {
            return nullptr;
        }

        if (auto loadOp = mlir::dyn_cast<mlir_ts::LoadOp>(def); loadOp && isPlace(loadOp.getReference()))
        {
            return def;
        }

        return borrowingCall(root);
    }

    // The call a value is the result of, when that result borrows an argument.
    static mlir::Operation *borrowingCall(mlir::Value value)
    {
        auto *def = value.getDefiningOp();
        auto index = def && isCall(def) ? resultBorrows(def) : -1;
        return index >= 0 && static_cast<size_t>(index) < callArgs(def).size() ? def : nullptr;
    }

    // The places from a read up to its roots, and the roots: the values the function reached the
    // first container through. A container reached through a merge (`(c ? a : b).x`) has a root
    // on each way in.
    struct Chain
    {
        enum Kind
        {
            Slot,    // an owning local, or a value-type local reached by reference
            Owned,   // a value this function made
            NotOwned // a parameter, `this`, a global, anything else
        };

        // a field's position and reference type; an element is position -1
        llvm::SmallVector<std::pair<int64_t, mlir::Type>> places;
        llvm::SmallVector<std::pair<mlir::Value, Kind>> roots;
        // Somewhere under a root, but where is not known: a result that borrows an argument may
        // be any field or element under it. Every overwrite and every removal drops it.
        bool anyPlace = false;

        bool owned() const
        {
            return llvm::none_of(roots, [](auto &root) { return root.second == NotOwned; });
        }
    };

    Chain chainOf(mlir::Operation *read)
    {
        Chain chain;
        llvm::SmallVector<mlir::Value> refs;
        llvm::SmallVector<mlir::Value> first;
        if (auto loadOp = mlir::dyn_cast<mlir_ts::LoadOp>(read))
        {
            refs.push_back(loadOp.getReference());
        }
        else
        {
            chain.anyPlace = true;
            first.push_back(rootOf(callArgs(read)[resultBorrows(read)]));
        }

        llvm::SmallPtrSet<mlir::Value, 8> seen;
        while (!refs.empty() || !first.empty())
        {
            llvm::SmallVector<mlir::Value> bases;
            if (!first.empty())
            {
                bases.swap(first);
            }
            else
            {
                auto ref = refs.pop_back_val();
                mlir::Value base;
                if (auto propertyRefOp = ref.getDefiningOp<mlir_ts::PropertyRefOp>())
                {
                    chain.places.push_back({propertyRefOp.getPosition(), propertyRefOp.getType()});
                    base = propertyRefOp.getObjectRef();
                }
                else
                {
                    auto elementRefOp = ref.getDefiningOp<mlir_ts::ElementRefOp>();
                    chain.places.push_back({-1, elementRefOp.getType()});
                    base = elementRefOp.getArray();
                }

                bases.push_back(rootOf(base));
            }

            while (!bases.empty())
            {
                auto base = bases.pop_back_val();
                if (!seen.insert(base).second)
                {
                    continue;
                }

                if (isPlace(base))
                {
                    refs.push_back(base); // a field of a value-type field: the same object
                    continue;
                }

                if (auto loadOp = base.getDefiningOp<mlir_ts::LoadOp>(); loadOp && isPlace(loadOp.getReference()))
                {
                    refs.push_back(loadOp.getReference());
                    continue;
                }

                // the object a method is called on
                if (auto object = boundThis(base))
                {
                    bases.push_back(rootOf(object));
                    continue;
                }

                // a result that borrows an argument is somewhere under the argument
                if (auto *call = borrowingCall(base))
                {
                    chain.anyPlace = true;
                    bases.push_back(rootOf(callArgs(call)[resultBorrows(call)]));
                    continue;
                }

                if (auto argument = mlir::dyn_cast<mlir::BlockArgument>(base); argument && mergedInto(argument, bases))
                {
                    continue;
                }

                if (base.getDefiningOp<mlir_ts::VariableOp>())
                {
                    chain.roots.push_back({base, Chain::Slot});
                }
                else if (auto slotLoad = slotLoadOf(base))
                {
                    chain.roots.push_back({slotLoad.getReference(), Chain::Slot});
                }
                else
                {
                    chain.roots.push_back({base, isFresh(base) ? Chain::Owned : Chain::NotOwned});
                }
            }
        }

        return chain;
    }

    // Where the roots stop holding what the chain reads: a local's releases and assignments and
    // every declaration or retain that may move it away; a made value's releases and every use
    // that takes it; a global's assignments.
    llvm::SmallVector<mlir::Operation *> rootEnds(const Chain &chain)
    {
        llvm::SmallVector<mlir::Operation *> ends;
        for (auto [root, kind] : chain.roots)
        {
            if (kind == Chain::Slot)
            {
                for (auto *user : root.getUsers())
                {
                    if (mlir::isa<mlir_ts::ReleaseSlotOp, mlir_ts::StoreOp>(user))
                    {
                        ends.push_back(user);
                    }
                    else if (auto loadOp = mlir::dyn_cast<mlir_ts::LoadOp>(user))
                    {
                        forEachUse(loadOp.getResult(), [&](mlir::Operation *use, mlir::Value used) {
                            auto varOp = mlir::dyn_cast<mlir_ts::VariableOp>(use);
                            if (mlir::isa<mlir_ts::RetainOp>(use) || (varOp && isOwningVariable(varOp)) ||
                                (isCall(use) && isKeptArgument(use, used)))
                            {
                                ends.push_back(use);
                            }
                        });
                    }
                }
            }
            else if (kind == Chain::Owned)
            {
                forEachUse(root, [&](mlir::Operation *use, mlir::Value used) {
                    if (mlir::isa<mlir_ts::ReleaseOp, mlir_ts::RetainOp>(use) || !isBorrow(use, used))
                    {
                        ends.push_back(use);
                    }
                });
            }
            else if (auto loadOp = root.getDefiningOp<mlir_ts::LoadOp>())
            {
                if (auto global = loadOp.getReference().getDefiningOp<mlir_ts::AddressOfOp>())
                {
                    getFunction().walk([&](mlir::Operation *op) {
                        mlir::Value slot;
                        if (auto releaseSlotOp = mlir::dyn_cast<mlir_ts::ReleaseSlotOp>(op))
                        {
                            slot = releaseSlotOp.getSlot();
                        }
                        else if (auto storeOp = mlir::dyn_cast<mlir_ts::StoreOp>(op))
                        {
                            slot = storeOp.getReference();
                        }

                        auto other = slot ? slot.getDefiningOp<mlir_ts::AddressOfOp>() : mlir_ts::AddressOfOp();
                        if (other && other.getGlobalName() == global.getGlobalName())
                        {
                            ends.push_back(op);
                        }
                    });
                }
            }
        }

        return ends;
    }

    // What a call has to be given to reach a root: the root, and where the root is a local
    // declared from another (a borrow, or a move that left the other dead), that one too.
    static llvm::SmallVector<mlir::Value> rootSources(const Chain &chain)
    {
        llvm::SmallVector<mlir::Value> sources;
        for (auto &root : chain.roots)
        {
            auto value = root.first;
            sources.push_back(value);
            for (auto steps = 0; steps < 8; ++steps)
            {
                auto varOp = value.getDefiningOp<mlir_ts::VariableOp>();
                if (!varOp || !varOp.getInitializer())
                {
                    break;
                }

                auto initializer = rootOf(varOp.getInitializer());
                auto slotLoad = slotLoadOf(initializer);
                value = slotLoad ? slotLoad.getReference() : initializer;
                sources.push_back(value);
            }
        }

        return sources;
    }

    // Everything that may point into what `start` points to: the forward closure over users'
    // results that may point anywhere. A call's result is not followed: the call hands back a
    // value it owns.
    const llvm::DenseSet<mlir::Value> &derivedFrom(mlir::Value start)
    {
        if (auto found = derivedCache.find(start); found != derivedCache.end())
        {
            return *found->second;
        }

        auto &slot = derivedCache[start];
        slot = std::make_shared<llvm::DenseSet<mlir::Value>>();
        auto &derived = *slot;
        llvm::SmallVector<mlir::Value> work{start};
        while (!work.empty())
        {
            auto value = work.pop_back_val();
            if (!derived.insert(value).second)
            {
                continue;
            }

            for (auto &use : value.getUses())
            {
                auto *user = use.getOwner();
                if (isCall(user))
                {
                    continue;
                }

                llvm::SmallVector<mlir::Value> next(user->getResults().begin(), user->getResults().end());
                passesOn(use, next);
                for (auto result : next)
                {
                    if (mayPointInto(result))
                    {
                        work.push_back(result);
                    }
                }
            }
        }

        return derived;
    }

    // May this op of the function's drops destroy what the chain reads? An overwrite of a place
    // on it, matched by position and type (two references to one field of one object always
    // agree on both; an unrelated field that happens to agree only adds an error); an array op
    // that removes elements of the type of an element on it; a call whose callee may destroy
    // anything (`__own_no_drops`, phase 4) - any such call, for a root this function does not
    // own, since the owner may be a global the callee assigns; else one given something derived
    // from the root that the read did not produce.
    bool dropsChain(mlir::Operation *drop, const Chain &chain, mlir::Operation *read)
    {
        if (drop == read)
        {
            return false; // a call that returns a borrow has not destroyed it by returning it
        }

        if (chain.anyPlace && !isCall(drop))
        {
            return true;
        }

        if (auto releaseSlotOp = mlir::dyn_cast<mlir_ts::ReleaseSlotOp>(drop))
        {
            // An assignment to a captured variable destroys its old value, which is reached only
            // through a read of the variable's cell: a root this function does not own.
            auto slot = releaseSlotOp.getSlot();
            if (isCellVariable(slot) || isLoadedCell(slot))
            {
                return !chain.owned();
            }

            int64_t position = -1;
            if (auto propertyRefOp = slot.getDefiningOp<mlir_ts::PropertyRefOp>())
            {
                position = propertyRefOp.getPosition();
            }

            return llvm::any_of(chain.places, [&](auto &place) {
                return place.first == position && place.second == slot.getType();
            });
        }

        if (isCall(drop))
        {
            if (callNoDrops(drop))
            {
                return false;
            }

            // a call given a closure may run it, and its box may hold anything
            if (!chain.owned() || llvm::any_of(drop->getOperands(), holdsCaptures))
            {
                return true;
            }

            auto &fromRead = derivedFrom(read->getResult(0));
            auto sources = rootSources(chain);
            return llvm::any_of(drop->getOperands(), [&](mlir::Value operand) {
                return !fromRead.contains(operand) && llvm::any_of(sources, [&](mlir::Value source) {
                           return derivedFrom(source).contains(operand);
                       });
            });
        }

        // pop, shift, splice, `length =`
        auto refType = mlir::dyn_cast<mlir_ts::RefType>(drop->getOperand(0).getType());
        auto arrayType = refType ? mlir::dyn_cast<mlir_ts::ArrayType>(refType.getElementType()) : mlir_ts::ArrayType();
        if (!arrayType)
        {
            return true;
        }

        return llvm::any_of(chain.places, [&](auto &place) {
            auto placeType = mlir::dyn_cast<mlir_ts::RefType>(place.second);
            return place.first == -1 && placeType && placeType.getElementType() == arrayType.getElementType();
        });
    }

    // A read out of a container is a borrow of it (spec 2.3), and so is a `let` declared from one:
    // its retain and releases go when the check passes, as for any borrowing `let` (phase 2).
    // Reports and returns false when the read is kept, or used after something that may destroy
    // it.
    bool checkPlaceRead(mlir::Operation *read, llvm::SetVector<mlir::Operation *> &toErase)
    {
        auto loadOp = mlir::dyn_cast<mlir_ts::LoadOp>(read);
        auto place = loadOp ? describePlace(loadOp.getReference()) : describeCall(read);
        auto result = read->getResult(0);
        llvm::StringRef name = ownerName(result);
        llvm::SmallVector<BorrowUse> uses;
        llvm::SmallVector<mlir::Operation *> bookkeeping;
        mlir_ts::VariableOp letKeeps;
        mlir::Operation *letKept = nullptr;
        // A result that borrows arrives with rc's temporary release, or rc's consuming `let`, and
        // in a function returning a borrow of the same parameter it may go out with rc's retain.
        Keeping keeping{&bookkeeping, returnsBorrowOf >= 0 && borrowedParam(result) == returnsBorrowOf};
        auto *kept = walkBorrowed(result, read, uses, [&](mlir::Operation *user,
                                                          llvm::ArrayRef<mlir::Operation *> kills) {
            auto borrower = borrowerOf(user, /*consumed=*/true);
            if (!borrower || borrower.getOperation() != user)
            {
                return false;
            }

            name = varName(borrower);
            if (auto *keeps = collectBorrowerUses(borrower, kills, uses, bookkeeping); keeps && !letKept)
            {
                letKept = keeps;
                letKeeps = borrower;
            }

            return true;
        }, keeping);

        if (kept || letKept)
        {
            if (!kept && isAssignmentOf(letKept, letKeeps))
            {
                reportPlaceBorrowerAssigned(letKept, varName(letKeeps), place);
            }
            else
            {
                reportPlaceBorrowEscapes(kept ? kept : letKept, name, place);
            }

            return false;
        }

        auto chain = chainOf(read);
        // A call given the borrow is a use of it for as long as the callee runs, so a call that
        // may destroy it is a use after that, whatever the order of its arguments. The function
        // value it goes through is only read as it starts.
        auto outlives = [&](mlir::Operation *drop) {
            for (auto &use : uses)
            {
                if ((use.op != drop || !use.callee) && reachableAfter(drop, use.op, use.kills))
                {
                    reportPlaceBorrowOutlives(use.op, drop, name, place);
                    return true;
                }
            }

            return false;
        };

        for (auto *end : rootEnds(chain))
        {
            if (outlives(end))
            {
                return false;
            }
        }

        for (auto *drop : drops)
        {
            if (dropsChain(drop, chain, read) && outlives(drop))
            {
                return false;
            }
        }

        toErase.insert(bookkeeping.begin(), bookkeeping.end());
        return true;
    }

    // A parameter, or a view of one (`<C>u`, the payload `ts.Unbox` reads out of an `any`), that
    // this function returns as a borrow (`__own_result_borrows`). rc retained it for the caller,
    // who now takes no reference: the retain goes, and nothing may keep it. The caller's own
    // borrow of the argument bounds it; nothing here can end that but a call, and a call that may
    // is the caller's use of what it passed (spec 2.4, phase 4).
    void checkReturnedParam(mlir::Value value, llvm::SetVector<mlir::Operation *> &toErase)
    {
        llvm::SmallVector<BorrowUse> uses;
        llvm::SmallVector<mlir::Operation *> bookkeeping;
        if (auto *kept = walkBorrowed(value, {}, uses, nullptr, Keeping{&bookkeeping, true}))
        {
            reportBorrowEscapes(kept, ownerName(value), "a parameter");
            return;
        }

        toErase.insert(bookkeeping.begin(), bookkeeping.end());
    }

    // `the result of 'first'`, for a call whose result borrows an argument.
    static std::string describeCall(mlir::Operation *call)
    {
        auto callee = calleeName(call);
        return callee.empty() ? "a call's result" : "the result of '" + callee + "'";
    }

    // The function a call names - directly, or the method it calls through a vtable - or empty.
    static std::string calleeName(mlir::Operation *call)
    {
        mlir::StringAttr callee;
        if (auto callOp = mlir::dyn_cast<mlir_ts::SymbolCallInternalOp>(call))
        {
            callee = callOp.getCalleeAttr().getAttr();
        }
        else if (auto calleeOp = call->getOperand(0).getDefiningOp())
        {
            if (auto getMethodOp = mlir::dyn_cast<mlir_ts::GetMethodOp>(calleeOp))
            {
                calleeOp = getMethodOp.getBoundFunc().getDefiningOp();
            }

            if (auto refOp = mlir::dyn_cast_or_null<mlir_ts::ThisVirtualSymbolRefOp>(calleeOp))
            {
                callee = refOp.getIdentifierAttr().getAttr();
            }
            else if (auto refOp = mlir::dyn_cast_or_null<mlir_ts::VirtualSymbolRefOp>(calleeOp))
            {
                callee = refOp.getIdentifierAttr().getAttr();
            }
            else if (auto refOp = mlir::dyn_cast_or_null<mlir_ts::ThisSymbolRefOp>(calleeOp))
            {
                callee = refOp.getIdentifierAttr().getAttr();
            }
        }

        return callee ? callee.getValue().str() : std::string();
    }

    // `'h.c'`, `'arr[]'`, `'h.c.v'` - or `a field`, `an element` when the root has no name (a
    // temporary, or no --di).
    std::string describePlace(mlir::Value ref)
    {
        auto path = pathOf(ref);
        if (path.empty())
        {
            return ref.getDefiningOp<mlir_ts::ElementRefOp>() ? "an element" : "a field";
        }

        return "'" + path + "'";
    }

    std::string pathOf(mlir::Value ref)
    {
        mlir::Value base;
        std::string step;
        if (auto propertyRefOp = ref.getDefiningOp<mlir_ts::PropertyRefOp>())
        {
            auto field = fieldName(propertyRefOp);
            if (field.empty())
            {
                return {};
            }

            base = propertyRefOp.getObjectRef();
            step = "." + field;
        }
        else if (auto elementRefOp = ref.getDefiningOp<mlir_ts::ElementRefOp>())
        {
            base = elementRefOp.getArray();
            step = "[]";
        }
        else
        {
            return {};
        }

        base = rootOf(base);
        std::string prefix;
        if (isPlace(base))
        {
            prefix = pathOf(base);
        }
        else if (auto loadOp = base.getDefiningOp<mlir_ts::LoadOp>())
        {
            auto from = loadOp.getReference();
            if (isPlace(from))
            {
                prefix = pathOf(from);
            }
            else if (auto varOp = from.getDefiningOp<mlir_ts::VariableOp>())
            {
                prefix = varName(varOp).str();
            }
            else if (auto global = from.getDefiningOp<mlir_ts::AddressOfOp>())
            {
                prefix = global.getGlobalName().str();
            }
        }
        else if (auto varOp = base.getDefiningOp<mlir_ts::VariableOp>())
        {
            prefix = varName(varOp).str();
        }
        else
        {
            prefix = ownerName(base).str();
        }

        // a name MLIRGen made up (`.a`, the array a `for...of` walks) means nothing to the reader
        if (prefix.empty() || prefix == "this value" || prefix.front() == '.')
        {
            return {};
        }

        return prefix + step;
    }

    static std::string fieldName(mlir_ts::PropertyRefOp propertyRefOp)
    {
        auto type = propertyRefOp.getObjectRef().getType();
        if (auto refType = mlir::dyn_cast<mlir_ts::RefType>(type))
        {
            type = refType.getElementType();
        }

        if (auto classType = mlir::dyn_cast<mlir_ts::ClassType>(type))
        {
            type = classType.getStorageType();
        }

        auto position = propertyRefOp.getPosition();
        mlir::Attribute id;
        if (auto storageType = mlir::dyn_cast<mlir_ts::ClassStorageType>(type); storageType && position < storageType.size())
        {
            id = storageType.getId(position);
        }
        else if (auto tupleType = mlir::dyn_cast<mlir_ts::TupleType>(type); tupleType && position < tupleType.size())
        {
            id = tupleType.getId(position);
        }

        if (auto name = mlir::dyn_cast_or_null<mlir::StringAttr>(id))
        {
            return name.str();
        }

        if (auto name = mlir::dyn_cast_or_null<mlir::FlatSymbolRefAttr>(id))
        {
            return name.getValue().str();
        }

        return {};
    }

    // The name of a temporary that owns: a folded `const` has no ts.Variable, but under --di it
    // has a ts.DebugVariable, whose location carries the name.
    static llvm::StringRef ownerName(mlir::Value value)
    {
        llvm::StringRef name = "this value";
        forEachUse(value, [&](mlir::Operation *user, mlir::Value) {
            if (mlir::isa<mlir_ts::DebugVariableOp>(user))
            {
                name = nameAt(user->getLoc());
            }
        });

        return name;
    }

    // The value a retain acquires: a Retain's operand, or a RetainSlot's variable's initializer,
    // seen through its views. None for a variable with no initializer - its storage was hoisted
    // in front of a try and its value arrives by a store this phase does not follow.
    static mlir::Value retainedValue(mlir::Operation *op)
    {
        if (auto retainOp = mlir::dyn_cast<mlir_ts::RetainOp>(op))
        {
            return rootOf(retainOp.getReference());
        }

        auto varOp = mlir::cast<mlir_ts::RetainSlotOp>(op).getSlot().getDefiningOp<mlir_ts::VariableOp>();
        return varOp && varOp.getInitializer() ? rootOf(varOp.getInitializer()) : mlir::Value();
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
        auto castOp = rootOf(value).getDefiningOp<mlir_ts::CastOp>();
        return castOp && mlir::isa<mlir_ts::StringType>(castOp.getType()) &&
               castOp.getIn().getDefiningOp<mlir_ts::ConstantOp>();
    }

    // A use that reads the value without keeping it. Kept deliberately short: anything not listed
    // is treated as taking the value, which can only turn a program into an error.
    static bool isBorrow(mlir::Operation *user, mlir::Value value)
    {
        if (mlir::isa<mlir_ts::PrintOp, mlir_ts::PropertyRefOp, mlir_ts::ElementRefOp, mlir_ts::LengthOfOp,
                      mlir_ts::ThisSymbolRefOp, mlir_ts::VirtualSymbolRefOp, mlir_ts::ThisVirtualSymbolRefOp,
                      mlir_ts::InterfaceSymbolRefOp, mlir_ts::GetThisOp, mlir_ts::GetMethodOp,
                      mlir_ts::ArithmeticBinaryOp, mlir_ts::LogicalBinaryOp, mlir_ts::StringConcatOp,
                      mlir_ts::StringResizeOp, mlir_ts::StringLengthOp>(user))
        {
            return true;
        }

        // `--di`'s record of a folded const's value for the debugger: names it, keeps nothing
        if (mlir::isa<mlir_ts::DebugVariableOp>(user))
        {
            return true;
        }

        // a test - `if (d)`, `d?.x`, `typeof u` - asks about the value, and keeps nothing
        if (mlir::isa<mlir_ts::HasValueOp, mlir_ts::GetTypeInfoFromUnionOp, mlir_ts::TypeOfOp,
                      mlir_ts::TypeOfAnyOp>(user))
        {
            return true;
        }

        if (auto castOp = mlir::dyn_cast<mlir_ts::CastOp>(user))
        {
            return mlir::isa<mlir_ts::BooleanType>(castOp.getType()) || castOp.getType().isInteger(1) ||
                   isBoundForCall(castOp);
        }

        // arguments are borrowed, but one the callee keeps (`__own_params`) is taken; a callee
        // with no facts that keeps one retains it, and that retain is its own error
        if (isCall(user))
        {
            return !isKeptArgument(user, value);
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

        // the box of a closure that borrows what it captures holds a copy that owns nothing; the
        // closure's own uses are the borrow's (walkBorrowed, decideClosures)
        if (mlir::isa<mlir_ts::StoreOp>(user) && user->hasAttr(OWN_CAPTURE_BORROW_ATTR_NAME))
        {
            return true;
        }

        return false;
    }

    // `ts.CreateBoundFunction(ts.Cast(object), method)` split straight back into `ts.GetThis` and
    // `ts.GetMethod` for a call: a method of another module's class - its constructor, when it is
    // built with `new` - called on an object here. It reads the object as `ts.ThisSymbolRef` does
    // for a method of this module; nothing else may use the cast or the bound function, and the
    // object comes out again only as a call's argument, which is a borrow of its own.
    static bool isBoundForCall(mlir_ts::CastOp castOp)
    {
        if (!mlir::isa<mlir_ts::OpaqueType>(castOp.getType()) || castOp->use_empty())
        {
            return false;
        }

        return llvm::all_of(castOp->getUses(), [](mlir::OpOperand &use) {
            auto boundOp = mlir::dyn_cast<mlir_ts::CreateBoundFunctionOp>(use.getOwner());
            if (!boundOp || boundOp.getThisVal() != use.get())
            {
                return false;
            }

            return llvm::all_of(boundOp->getUsers(), [](mlir::Operation *boundUser) {
                if (mlir::isa<mlir_ts::GetMethodOp>(boundUser))
                {
                    return true;
                }

                return mlir::isa<mlir_ts::GetThisOp>(boundUser) &&
                       llvm::all_of(boundUser->getUsers(), [](mlir::Operation *thisUser) { return isCall(thisUser); });
            });
        });
    }

    // Is `value` an argument the call's callee keeps?
    static bool isKeptArgument(mlir::Operation *call, mlir::Value value)
    {
        auto args = callArgs(call);
        return llvm::any_of(ownedParams(call), [&](int32_t index) {
            return static_cast<size_t>(index) < args.size() && args[index] == value;
        });
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
        // this pass, so one such value is routinely stored into several places. A read out of a
        // container owns nothing either: checkPlaceRead decides it.
        if (!ownsHeap(value) || isImmortalLiteral(value) || placeReadOf(value))
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
        forEachUse(value, [&](mlir::Operation *user, mlir::Value used) {
            if (mlir::isa<mlir_ts::RetainOp>(user))
            {
                return;
            }

            if (mlir::isa<mlir_ts::ReleaseOp>(user))
            {
                releases.push_back(user);
            }
            else if (!isBorrow(user, used))
            {
                takers.push_back(user);
            }
        });

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
        forEachUse(value, [&](mlir::Operation *user, mlir::Value used) {
            if (mlir::isa<mlir_ts::RetainOp>(user))
            {
                return;
            }

            if (mlir::isa<mlir_ts::ReleaseOp>(user))
            {
                releases.push_back(user);
            }
            else if (isBorrow(user, used))
            {
                reads.push_back(user);
            }
            else
            {
                takers.push_back(user);
            }
        });

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
        return reachableAfter(from, to, kill ? llvm::ArrayRef<mlir::Operation *>(kill) : llvm::ArrayRef<mlir::Operation *>());
    }

    // The same, where a path through any of `kills` is a new value. A use that is itself a kill
    // runs after the value is made again.
    bool reachableAfter(mlir::Operation *from, mlir::Operation *to, llvm::ArrayRef<mlir::Operation *> kills)
    {
        if (llvm::is_contained(kills, to))
        {
            return false;
        }

        auto &body = getFunction().getBody();
        from = body.findAncestorOpInRegion(*from);
        to = body.findAncestorOpInRegion(*to);
        llvm::SmallVector<mlir::Operation *, 4> killAt;
        for (auto *kill : kills)
        {
            if (auto *top = body.findAncestorOpInRegion(*kill))
            {
                killAt.push_back(top);
            }
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

        // is there a kill in `block` after `after` (from its start when null) and before `before`
        // (to its end when null)?
        auto killsIn = [&](mlir::Block *block, mlir::Operation *after, mlir::Operation *before) {
            return llvm::any_of(killAt, [&](mlir::Operation *kill) {
                return kill->getBlock() == block && (!after || after->isBeforeInBlock(kill)) &&
                       (!before || kill->isBeforeInBlock(before));
            });
        };

        auto *fromBlock = from->getBlock();
        auto *toBlock = to->getBlock();
        if (fromBlock == toBlock && from->isBeforeInBlock(to))
        {
            return !killsIn(fromBlock, from, to);
        }

        // every way out of this block runs a kill first
        if (killsIn(fromBlock, from, nullptr))
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

            if (block == toBlock && !killsIn(block, nullptr, to))
            {
                return true;
            }

            if (killsIn(block, nullptr, nullptr))
            {
                continue;
            }

            work.append(block->getSuccessors().begin(), block->getSuccessors().end());
        }

        return false;
    }

    // The use that takes `value` for this retain: the declaration a `ts.RetainSlot` belongs to,
    // or the one use of the value, or of a view of it, after a `ts.Retain` that is not a read.
    static mlir::Operation *takerOf(mlir::Operation *retain, mlir::Value value)
    {
        if (auto retainSlotOp = mlir::dyn_cast<mlir_ts::RetainSlotOp>(retain))
        {
            return retainSlotOp.getSlot().getDefiningOp();
        }

        // a call that keeps the argument: rc's retain for it is in the callee
        if (isCall(retain))
        {
            return retain;
        }

        mlir::Operation *taker = nullptr;
        auto several = false;
        forEachUse(value, [&](mlir::Operation *user, mlir::Value used) {
            if (mlir::isa<mlir_ts::RetainOp>(user) || isBorrow(user, used))
            {
                return;
            }

            several = several || taker;
            taker = user;
        });

        return several ? nullptr : taker; // two takers of one read: not a move this phase proves
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

                if (mlir::isa<mlir_ts::CastOp>(user) || isView(user))
                {
                    values.push_back(user->getResult(0));
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
        forEachUse(value, [&](mlir::Operation *user, mlir::Value) {
            if (mlir::isa<mlir_ts::RetainOp>(user) && user->getBlock() == taker->getBlock() &&
                user->isBeforeInBlock(first))
            {
                first = user;
            }
        });

        return first;
    }

    // Did rc give this taker a reference of its own - a `ts.Retain` of the value in front of it,
    // or the `ts.RetainSlot` of an owning local declared from it? Only then is there a release on
    // the taker's side to give the value back once it has moved. `ts.NewInterface` is a taker to
    // this pass but a view to rc, which retains nothing for it and relies on the instance's own
    // release; moving the instance into it would leave nothing to release it at all.
    static bool takerAcquires(mlir::Operation *taker, mlir::Value value)
    {
        // a callee that keeps its argument retained it in its own body under rc
        if (isCall(taker))
        {
            return true;
        }

        if (auto varOp = mlir::dyn_cast<mlir_ts::VariableOp>(taker))
        {
            return llvm::any_of(varOp.getResult().getUsers(),
                                [](mlir::Operation *user) { return mlir::isa<mlir_ts::RetainSlotOp>(user); });
        }

        auto acquires = false;
        forEachUse(value, [&](mlir::Operation *user, mlir::Value) {
            acquires = acquires || (mlir::isa<mlir_ts::RetainOp>(user) && user->getBlock() == taker->getBlock() &&
                                    user->isBeforeInBlock(taker));
        });

        return acquires;
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
        llvm::StringRef name = "this value";
        forEachUse(value, [&](mlir::Operation *user, mlir::Value) {
            if (auto varOp = mlir::dyn_cast<mlir_ts::VariableOp>(user); varOp && name == "this value")
            {
                name = varName(varOp);
            }
        });

        return name;
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

        auto diag = op->emitError("'") << name << "' takes a second reference; -mm=own cannot prove a move or a borrow here yet";
        noteLostFacts(diag);
        signalPassFailure();
    }

    // Where the body would have been fine had its callers known a fact about it, say why they
    // cannot.
    void noteLostFacts(mlir::InFlightDiagnostic &diag)
    {
        auto f = getFunction();
        if (auto why = f->getAttrOfType<mlir::StringAttr>(OWN_FACTS_LOST_ATTR_NAME))
        {
            diag.attachNote(f.getLoc()) << "'" << f.getSymName() << "' could return a borrow or keep a parameter, but "
                                        << why.getValue();
        }
    }

    void reportKeptOnSomePaths(mlir::Operation *move, mlir::Operation *exit, llvm::StringRef name)
    {
        if (quiet)
        {
            return;
        }

        auto diag = move->emitError("'") << name
                                         << "' is moved here on some paths only; -mm=own cannot release it "
                                            "on the others yet";
        diag.attachNote(exit->getLoc()) << "returns here without moving it; the caller gave it up";
        signalPassFailure();
    }

    void reportGivenNotOwned(mlir::Operation *call, mlir::Value arg)
    {
        if (quiet)
        {
            return;
        }

        llvm::StringRef name = ownerName(arg);
        if (auto loadOp = rootOf(arg).getDefiningOp<mlir_ts::LoadOp>())
        {
            if (auto varOp = loadOp.getReference().getDefiningOp<mlir_ts::VariableOp>())
            {
                name = varName(varOp);
            }
        }

        auto callee = calleeName(call);
        auto diag = call->emitError("argument '") << name << "' is given to '" << (callee.empty() ? "a function" : callee)
                                                  << "', which keeps it, but -mm=own cannot move it here: this "
                                                     "function does not own it";
        noteLostFacts(diag);
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

    void reportPlaceBorrowOutlives(mlir::Operation *use, mlir::Operation *drop, llvm::StringRef name,
                                   llvm::StringRef place)
    {
        if (quiet)
        {
            return;
        }

        auto diag = use->emitError("'") << name << "' borrows " << place
                                        << " but is used here after it may be released or overwritten";
        diag.attachNote(drop->getLoc()) << "it may be released or overwritten here";
        signalPassFailure();
    }

    void reportPlaceBorrowerAssigned(mlir::Operation *op, llvm::StringRef name, llvm::StringRef place)
    {
        if (quiet)
        {
            return;
        }

        op->emitError("'") << name << "' borrows " << place << " and cannot be assigned; -mm=own cannot prove that yet";
        signalPassFailure();
    }

    void reportPlaceBorrowEscapes(mlir::Operation *op, llvm::StringRef name, llvm::StringRef place)
    {
        if (quiet)
        {
            return;
        }

        auto diag = op->emitError("'") << name << "' borrows " << place << " and cannot be stored, returned or captured";
        noteLostFacts(diag);
        signalPassFailure();
    }

    void reportMovedWhileBorrowed(mlir::Operation *op, llvm::StringRef owner, llvm::StringRef borrower)
    {
        if (quiet)
        {
            return;
        }

        op->emitError("'") << owner << "' is moved here while '" << borrower
                           << "' borrows it; a borrowed value cannot be moved";
        signalPassFailure();
    }

    void reportBorrowEscapes(mlir::Operation *op, llvm::StringRef name, llvm::StringRef owner)
    {
        if (quiet)
        {
            return;
        }

        auto diag = op->emitError("'") << name << "' borrows '" << owner << "' and cannot be stored, returned or captured";
        noteLostFacts(diag);
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

    void reportClosureEscapes(mlir::Operation *escape, mlir::Operation *closure)
    {
        auto diag = escape->emitError("a closure that captures a variable and escapes here is not supported by -mm=own yet");
        diag.attachNote(closure->getLoc()) << "the closure is created here";
        signalPassFailure();
    }

    void reportClosureOutlives(mlir::Operation *use, mlir::Operation *end, llvm::StringRef closure,
                               llvm::StringRef captured)
    {
        auto diag = use->emitError("'") << closure << "' borrows '" << captured << "', which it captures, but runs here after '"
                                        << captured << "' is released";
        diag.attachNote(end->getLoc()) << "'" << captured << "' is released here";
        signalPassFailure();
    }

    void reportCaptureNotOwned(mlir::Operation *op, llvm::StringRef captured)
    {
        op->emitError("'") << captured << "' is captured by a closure, but -mm=own cannot tell who owns its value here";
        signalPassFailure();
    }

    void reportCapturedParamAssigned(mlir::Operation *op, llvm::StringRef name)
    {
        op->emitError("'") << name
                           << "' is a captured parameter and cannot be assigned under -mm=own: its value is the "
                              "caller's";
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
