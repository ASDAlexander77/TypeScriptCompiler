#include "mlir/Pass/Pass.h"
#include "mlir/IR/Dominance.h"
#include "mlir/Dialect/LLVMIR/LLVMAttrs.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/FunctionInterfaces.h"

#include "TypeScript/TypeScriptDialect.h"
#include "TypeScript/TypeScriptOps.h"
#include "TypeScript/TypeScriptFunctionPass.h"
#include "TypeScript/Passes.h"
#include "TypeScript/Defines.h"
#include "TypeScript/MLIRLogic/MLIRTypeHelper.h"

#include "OwnershipFacts.h"

#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/ScopeExit.h"
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
//
// It runs on every function: a `ts.Func`, and the `func.func` the async lowering outlines an
// `await`'s continuation or a `for await` body into, whose arguments are the values it captures.
class OwnershipInferencePass
    : public mlir::PassWrapper<OwnershipInferencePass, mlir::InterfacePass<mlir::FunctionOpInterface>>
{
  public:
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(OwnershipInferencePass)

    void runOnOperation() override
    {
        if (!getFunction().isExternal())
        {
            runOnFunction();
        }
    }

    mlir::FunctionOpInterface getFunction()
    {
        return getOperation();
    }

    void runOnFunction()
    {
        // the dry run (spec 22.3): does today's analysis pass as it is?
        {
            llvm::SetVector<mlir::Operation *> toErase;
            dryRun = true;
            failures = 0;
            auto passed = analyze(toErase);
            dryRun = false;
            // The copies are judged with the dry run's facts still set: a store into a borrowing
            // closure's box, or a borrowing field's, is a borrow only by its
            // `__own_capture_borrow`. They are taken off after, before the real run sets its own.
            // makeStringCopies erases only retains, which the dry run sets no attribute on.
            if (!passed)
            {
                makeStringCopies(toErase);
            }

            for (auto &[op, name] : dryRunAttrs)
            {
                op->removeAttr(name);
            }

            dryRunAttrs.clear();
        }

        llvm::SetVector<mlir::Operation *> toErase;
        analyze(toErase);
    }

    // The analysis of one function; true when it passed. In a dry run it erases nothing and strips
    // no facts.
    bool analyze(llvm::SetVector<mlir::Operation *> &toErase)
    {
        auto f = getFunction();
        mlir::DominanceInfo dominanceInfo(f);
        dominance = &dominanceInfo;
        // it is this call's: nothing after it (the copies, the next run) may read it
        auto dominanceGone = llvm::make_scope_exit([&]() { dominance = nullptr; });

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
        closureOf.clear();
        stateBoxes.clear();
        fieldBorrowRetains.clear();
        fieldBorrowReleases.clear();
        captureStores.clear();
        captureRetains.clear();
        returnedParams.clear();
        returnsBorrowOf = resultBorrows(f);

        // first: whether a closure borrows what it captures changes what its box's stores are
        classifyClosures();
        classifyStateBoxes();
        classifyBorrowingFields();

        // A view of a block is that block: the candidate is always the root.
        f.walk([&](mlir::Operation *op) {
            // A handle put into a union with a member that owns a block of its own: counting the
            // union would count that member's single-owner block, and not counting it would lose
            // the handle's count (spec 23.2).
            reportHandleInOwningUnion(op);

            // A handle's retains and releases are rc's, counted, and nobody's to decide (spec 23.2).
            // An overwrite of a field or an element that holds one is still a drop: the old handle
            // may be the last, and its payload goes with it.
            if (touchesHandle(op) && !mlir::isa<mlir_ts::ReleaseSlotOp>(op))
            {
                return;
            }

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
                if (varOp.getInitializer() && isOwningVariable(varOp) && !isHandle(varOp.getInitializer()))
                {
                    candidates.insert(rootOf(varOp.getInitializer()));
                }
            }
            else if (mlir::isa<mlir_ts::RetainCellOp>(op))
            {
                // a box taking a cell: a closure that borrows it (decideClosures) claimed it
                if (!captureRetains.contains(op) && reporting())
                {
                    op->emitError("closures that capture a variable are not supported by -mm=own here yet");
                    signalPassFailure();
                }
            }
            else if (mlir::isa<mlir_ts::DeleteOp>(op))
            {
                if (reporting())
                {
                    op->emitError("'delete' is not supported by -mm=own yet");
                    signalPassFailure();
                }
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
            else if (mlir::isa<mlir_ts::ArrayPushOp, mlir_ts::ArrayUnshiftOp>(op))
            {
                // through a handle, it may move the elements another handle's borrow reads (spec 23.3)
                if (reachesSharedValue(op->getOperand(0)))
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
                if (!isView(op) && isFresh(result) && ownsHeap(result) && !isHandle(result))
                {
                    candidates.insert(result);
                }
            }
        });

        // Every value gets the owner check, fresh or not: rc can hand a value it did not make to
        // two consuming declarations without a single retain (the result of an indirect call
        // taken by `let b = a; let c = a;`). Only a fresh value's retains can be erased.
        llvm::DenseSet<mlir::Value> proven;
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
            if (touchesHandle(op))
            {
                continue;
            }

            if (captureRetains.contains(op))
            {
                continue; // a borrowing box's copy of a value: decideClosures decides it
            }

            if (fieldBorrowRetains.contains(op))
            {
                toErase.insert(op); // a borrowing field's: it takes nothing
                continue;
            }

            // a retain of what holds no block takes nothing: a union's number payload made into
            // another union is a view of the union it came from, but a number all the way
            if (auto retainOp = mlir::dyn_cast<mlir_ts::RetainOp>(op); retainOp && holdsNoBlock(retainOp.getReference()))
            {
                toErase.insert(op);
                continue;
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

            if (value && holdsConstantsOrMoves(value))
            {
                toErase.insert(op);
                continue;
            }

            // data nothing owns (isConstantData): a number or null in a union, a function, a
            // literal, a tuple of literals. A slot's retain is decided on its initializer, so only
            // where the slot still holds it.
            if (value && isConstantData(value) && (mlir::isa<mlir_ts::RetainOp>(op) || slotHoldsInitializerAt(op)))
            {
                toErase.insert(op);
                continue;
            }

            // an array a spread builds, read into the owning local it is made for
            if (value && mlir::isa<mlir_ts::RetainSlotOp>(op) && isBuiltArrayRead(op, value))
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
                // a handle given to a callee that keeps it is a counted copy (spec 23.2)
                if (static_cast<size_t>(index) >= args.size() || holdsNoBlock(args[index]) ||
                    isImmortalLiteral(args[index]) || isHandle(args[index]))
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
        checkBorrowedReads();
        checkBoundedResults();
        toErase.insert(fieldBorrowReleases.begin(), fieldBorrowReleases.end());

        if (dryRun)
        {
            return failures == 0;
        }

        // a handle's retains and releases are rc's, and stay (spec 23.2)
        toErase.remove_if([](mlir::Operation *op) { return touchesHandle(op); });
        for (auto *op : toErase)
        {
            op->erase();
        }

        // the facts were for this pass only
        returnedParams.clear();
        f.walk([](mlir::Operation *op) {
            for (auto *name : {OWN_PARAMS_ATTR_NAME, OWN_RESULT_BORROWS_ATTR_NAME, OWN_NO_DROPS_ATTR_NAME,
                               OWN_FACTS_LOST_ATTR_NAME, OWN_FRESH_RESULT_ATTR_NAME, OWN_ASSIGNS_CAPTURES_ATTR_NAME,
                               OWN_RESULT_BOUNDED_ATTR_NAME,
                               OWN_CAPTURE_BORROW_ATTR_NAME})
            {
                op->removeAttr(name);
            }
        });

        return true;
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

    // The dry run (spec 22.3): the whole analysis, but nothing reported and nothing changed. A
    // report it would make is counted in `failures`.
    bool dryRun = false;
    unsigned failures = 0;
    // attributes the dry run set on ops that did not have them, taken off again after it
    llvm::SmallVector<std::pair<mlir::Operation *, mlir::StringAttr>> dryRunAttrs;

    // Whether a report is to be emitted: not under `quietly`'s trial, and not in the dry run, which
    // only counts it.
    bool reporting()
    {
        if (quiet)
        {
            return false;
        }

        if (dryRun)
        {
            ++failures;
            return false;
        }

        return true;
    }

    // `op->setAttr(name, unit)`, undone after the dry run when the op did not have it
    void setAttrTracked(mlir::Operation *op, llvm::StringRef name)
    {
        auto attrName = mlir::StringAttr::get(&getContext(), name);
        if (dryRun && !op->hasAttr(attrName))
        {
            dryRunAttrs.push_back({op, attrName});
        }

        op->setAttr(attrName, mlir::UnitAttr::get(&getContext()));
    }

    // Strings as values (spec 22.3): each string retain the dry run did not erase is rewritten, where
    // its one use can be found, into a copy for that use.
    void makeStringCopies(const llvm::SetVector<mlir::Operation *> &erased)
    {
        llvm::SmallVector<mlir::Operation *> retains;
        getFunction()->walk([&](mlir::Operation *op) {
            if (mlir::isa<mlir_ts::RetainOp, mlir_ts::RetainSlotOp>(op) && !erased.contains(op) &&
                !captureRetains.contains(op) && !fieldBorrowRetains.contains(op))
            {
                retains.push_back(op);
            }
        });

        for (auto *retain : retains)
        {
            copyForRetain(retain);
        }
    }

    // The value a parameter of this function is, when its callers were told this function keeps it
    // or returns a borrow of it (spec 15): copying it would leak what the caller gave up.
    bool borrowsKnownParam(mlir::Value value)
    {
        auto param = borrowedParam(value);
        return param >= 0 && (llvm::is_contained(ownedParams(getFunction()), param) || param == returnsBorrowOf);
    }

    // Rewrites one string retain into a copy for the use it was made for (spec 22.3), and erases it;
    // false, leaving it for the real run to report, where there is no single such use in its block,
    // or the value is not a string or is a known parameter's (spec 22.4). Its use walk is takerOf's
    // without the twinReadAfter fallback: a retain whose value has no use of its own is not copied,
    // so its error stays.
    bool copyForRetain(mlir::Operation *retain)
    {
        // `let x = v`: the local starts from a copy
        if (auto retainSlotOp = mlir::dyn_cast<mlir_ts::RetainSlotOp>(retain))
        {
            auto varOp = retainSlotOp.getSlot().getDefiningOp<mlir_ts::VariableOp>();
            auto init = varOp ? varOp.getInitializer() : mlir::Value();
            if (!init || !isCopyableString(init.getType()) || borrowsKnownParam(init) || isCellVariable(varOp.getResult()))
            {
                return false;
            }

            mlir::OpBuilder builder(varOp);
            auto copy = builder.create<mlir_ts::StringCopyOp>(varOp.getLoc(), init.getType(), init);
            varOp.getInitializerMutable().assign(copy);
            retain->erase();
            return true;
        }

        auto retainOp = mlir::cast<mlir_ts::RetainOp>(retain);
        auto value = retainOp.getReference();
        if (!isCopyableString(value.getType()) || borrowsKnownParam(value))
        {
            return false;
        }

        // the one use rc retained it for
        mlir::OpOperand *taking = nullptr;
        auto several = false;
        forEachUse(value, [&](mlir::Operation *user, mlir::Value used) {
            if (mlir::isa<mlir_ts::RetainOp, mlir_ts::ReleaseOp>(user) || isBorrow(user, used))
            {
                return;
            }

            for (auto &operand : user->getOpOperands())
            {
                if (operand.get() == used)
                {
                    several = several || taking;
                    taking = &operand;
                }
            }
        });

        if (!taking || several || !isCopyableString(taking->get().getType()))
        {
            return false;
        }

        // only a retain in the use's block: a copy made elsewhere (rc's birth retain, after the
        // producer) has no owner on a path that does not reach the use; rc retains for the use
        // beside it too
        if (taking->getOwner()->getBlock() != retain->getBlock())
        {
            return false;
        }

        // made where rc took its reference: a store releases what it overwrites first, and that may
        // be the very string (`b.s = b.s`), or one the analysis cannot tell from it (`b.s = a.s`)
        auto *at = taking->getOwner();
        auto *def = taking->get().getDefiningOp();
        if (taking->get() == value || (def && def->getBlock() == retain->getBlock() && def->isBeforeInBlock(retain)))
        {
            // the use reads the copy, so the retain must come before it
            if (!retain->isBeforeInBlock(taking->getOwner()))
            {
                return false;
            }

            at = retain;
        }

        mlir::OpBuilder builder(at);
        auto copy = builder.create<mlir_ts::StringCopyOp>(retain->getLoc(), taking->get().getType(), taking->get());
        taking->set(copy);
        retain->erase();
        return true;
    }

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
        // a generator's box whose state object borrows from the generator's arguments
        // (OWN_BORROWING_FIELDS_ATTR_NAME): a parameter's cell moves in, and the box frees the cell only
        bool borrowsValues = false;
        // copies only, and given to a generator's maker that borrows from it (`.map(f)`): the box
        // borrows its copies, as a box with cells would
        bool borrowsCopies = false;
    };

    llvm::SmallVector<std::shared_ptr<Closure>> closures;
    // a copy stored into a borrowing box -> its closure
    llvm::DenseMap<mlir::Operation *, Closure *> captureStores;
    // rc's references for boxes, which the closures decide
    llvm::DenseSet<mlir::Operation *> captureRetains;
    // each closure over a cell, by its `ts.CreateBoundFunction`
    llvm::DenseMap<mlir::Operation *, Closure *> closureOf;
    // each generator's capture box, which escapes with its state object (classifyStateBoxes)
    llvm::DenseMap<mlir::Value, Closure *> stateBoxes;
    // rc's retains for stores into borrowing fields, and those fields' releases (classifyBorrowingFields)
    llvm::DenseSet<mlir::Operation *> fieldBorrowRetains;
    llvm::SmallVector<mlir::Operation *> fieldBorrowReleases;

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

            // copies only: the box owns them (phase 4); only an alias needs deciding - unless the box
            // is given to a generator's maker whose result borrows from it
            if (!closure->hasCells)
            {
                // ...and only when every copy has an owner of its own: the box frees itself alone, so a
                // value made for it (a callback that captures, `a.filter((x) => x > n)`) would be
                // nobody's
                closure->borrowsCopies =
                    !closure->escape &&
                    llvm::any_of(closure->boxUses,
                                 [](mlir::Operation *use) { return use->hasAttr(OWN_RESULT_BOUNDED_ATTR_NAME); }) &&
                    llvm::none_of(closure->fills, [&](auto &fill) {
                        auto value = fill.second.getValue();
                        return !holdsNoBlock(value) && isFresh(rootOf(value));
                    });
                if (!closure->borrowsCopies)
                {
                    if (!closure->escape && closure->aliased)
                    {
                        closures.push_back(std::move(closure));
                    }

                    return;
                }
            }

            for (auto &fill : closure->fills)
            {
                auto storeOp = fill.second;
                auto isCell = mlir::isa<mlir_ts::RefType>(storeOp.getValue().getType());
                // an escaping closure's copies move into its box as in phase 4, rc's retain with them
                if (closure->escape && !isCell)
                {
                    continue;
                }

                if (auto *retain = retainBefore(storeOp.getValue(), storeOp))
                {
                    closure->retains.push_back(retain);
                    captureRetains.insert(retain);
                }

                if (!closure->escape && !isCell)
                {
                    setAttrTracked(storeOp, OWN_CAPTURE_BORROW_ATTR_NAME);
                    captureStores[storeOp] = closure.get();
                }
            }

            if (!closure->escape)
            {
                setAttrTracked(boundOp, OWN_BORROWS_CAPTURES_ATTR_NAME);
            }

            closureOf[boundOp.getOperation()] = closure.get();
            closures.push_back(std::move(closure));
        });
    }

    // The fields of a generator's state object of this type that borrow from its arguments
    // (OWN_BORROWING_FIELDS_ATTR_NAME, the signature pass's), the `.captured` box's first.
    llvm::SmallVector<int32_t> borrowingFieldsOf(mlir::Type recordType)
    {
        llvm::SmallVector<int32_t> fields;
        if (auto objectType = mlir::dyn_cast<mlir_ts::ObjectType>(recordType))
        {
            recordType = objectType.getStorageType();
        }

        if (auto storageType = mlir::dyn_cast<mlir_ts::ObjectStorageType>(recordType))
        {
            recordType = mlir_ts::TupleType::get(&getContext(), storageType.getFields());
        }

        auto module = getFunction()->getParentOfType<mlir::ModuleOp>();
        auto entries = module ? module->getAttrOfType<mlir::ArrayAttr>(OWN_BORROWING_FIELDS_ATTR_NAME) : mlir::ArrayAttr();
        if (!entries)
        {
            return fields;
        }

        for (auto entry : entries.getAsRange<mlir::ArrayAttr>())
        {
            if (mlir::cast<mlir::TypeAttr>(entry[0]).getValue() == recordType)
            {
                auto indices = mlir::cast<mlir::DenseI32ArrayAttr>(entry[1]).asArrayRef();
                fields.append(indices.begin(), indices.end());
            }
        }

        return fields;
    }

    // Is `ref` a field of a generator's state object that borrows (other than the `.captured` box)?
    bool isBorrowingField(mlir::Value ref)
    {
        auto propertyRefOp = ref.getDefiningOp<mlir_ts::PropertyRefOp>();
        if (!propertyRefOp)
        {
            return false;
        }

        auto fields = borrowingFieldsOf(propertyRefOp->getOperand(0).getType());
        return fields.size() > 1 && llvm::is_contained(llvm::ArrayRef<int32_t>(fields).drop_front(),
                                                       static_cast<int32_t>(propertyRefOp.getPosition()));
    }

    // A store into a borrowing field holds what it stores without owning it, as a borrowing box's
    // copy does (OWN_CAPTURE_BORROW_ATTR_NAME), and rc's retain for it goes; its release, when the
    // field is overwritten, gives nothing back.
    void classifyBorrowingFields()
    {
        getFunction().walk([&](mlir::Operation *op) {
            if (auto storeOp = mlir::dyn_cast<mlir_ts::StoreOp>(op); storeOp && isBorrowingField(storeOp.getReference()))
            {
                setAttrTracked(storeOp, OWN_CAPTURE_BORROW_ATTR_NAME);
                if (auto *retain = retainBefore(storeOp.getValue(), storeOp))
                {
                    fieldBorrowRetains.insert(retain);
                }
            }
            else if (auto releaseSlotOp = mlir::dyn_cast<mlir_ts::ReleaseSlotOp>(op);
                     releaseSlotOp && isBorrowingField(releaseSlotOp.getSlot()))
            {
                fieldBorrowReleases.push_back(op);
            }
        });
    }

    // Is `ref` a cell read out of the `.captured` box of a generator's state object that borrows:
    // `ts.Load(ts.PropertyRef(ts.Load(ts.PropertyRef(state, captured)), i))`?
    bool isBorrowingCell(mlir::Value ref)
    {
        auto cellLoad = ref.getDefiningOp<mlir_ts::LoadOp>();
        return cellLoad && isBorrowingBoxField(cellLoad.getReference());
    }

    // Is `ref` a field of the `.captured` box of a generator's state object that borrows:
    // `ts.PropertyRef(ts.Load(ts.PropertyRef(state, captured)), i)`? A field copied in by value holds
    // the caller's value itself, a cell field the caller's cell.
    bool isBorrowingBoxField(mlir::Value ref)
    {
        auto cellRef = ref.getDefiningOp<mlir_ts::PropertyRefOp>();
        auto boxLoad = cellRef ? cellRef->getOperand(0).getDefiningOp<mlir_ts::LoadOp>() : mlir_ts::LoadOp();
        auto boxRef = boxLoad ? boxLoad.getReference().getDefiningOp<mlir_ts::PropertyRefOp>() : mlir_ts::PropertyRefOp();
        if (!boxRef)
        {
            return false;
        }

        auto fields = borrowingFieldsOf(boxRef->getOperand(0).getType());
        return !fields.empty() && fields.front() == static_cast<int32_t>(boxRef.getPosition());
    }

    // What a generator reads out of a borrowing cell or field is the caller's: every use of it must
    // be a borrow - a store into another borrowing field is one (classifyBorrowingFields). A yield of
    // it, a push, a return, a store anywhere else would give the caller's value a second owner, and
    // rc retains nothing for some of these, so nothing else would see it.
    void checkBorrowedReads()
    {
        getFunction().walk([&](mlir_ts::LoadOp loadOp) {
            auto ref = loadOp.getReference();
            if (!ownsHeap(loadOp.getResult()) || !(isBorrowingCell(ref) || isBorrowingBoxField(ref) || isBorrowingField(ref)))
            {
                return;
            }

            auto reported = false;
            forEachUse(loadOp.getResult(), [&](mlir::Operation *user, mlir::Value used) {
                if (reported || isBorrow(user, used) || fieldBorrowRetains.contains(user))
                {
                    return;
                }

                reportPlaceBorrowEscapes(user, ownerName(loadOp.getResult()), "the generator's argument");
                reported = true;
            });
        });
    }

    // A call whose result borrows from its arguments (OWN_RESULT_BOUNDED_ATTR_NAME): a generator over
    // a parameter's block. The result is owned as any fresh value is, but nothing may use it after
    // one of those arguments is given back - released, assigned, or dropped from the place it was
    // read from - and it may not be kept anywhere that could outlive them. Its releases are not
    // uses: the state object's release gives back nothing it borrows.
    void checkBoundedResults()
    {
        getFunction().walk([&](mlir::Operation *call) {
            auto bounded = call->getAttrOfType<mlir::DenseI32ArrayAttr>(OWN_RESULT_BOUNDED_ATTR_NAME);
            if (!bounded || !isCall(call) || call->getNumResults() != 1)
            {
                return;
            }

            auto name = ownerName(call->getResult(0));
            llvm::SmallVector<mlir::Operation *> uses;
            if (auto *kept = boundedUses(call->getResult(0), uses))
            {
                reportPlaceBorrowEscapes(kept, name, "its generator's arguments");
                return;
            }

            auto args = callArgs(call);
            for (auto index : bounded.asArrayRef())
            {
                if (static_cast<size_t>(index) >= args.size())
                {
                    continue;
                }

                // a closure's box (`.map(f)` calls its maker with it): it lives while the closures on
                // it do, and what it holds while their owners do. A reference to storage, so first.
                auto root = rootOf(args[index]);
                if (auto boxOp = root.getDefiningOp<mlir_ts::VariableOp>(); boxOp && boxOp->hasAttr(CAPTURE_BOX_ATTR_NAME))
                {
                    if (!checkBoundedByBox(call, root, uses, name))
                    {
                        return;
                    }

                    continue;
                }

                if (holdsNoBlock(args[index]) || isImmortalLiteral(args[index]))
                {
                    continue;
                }

                if (borrowedParam(root) >= 0)
                {
                    continue; // the caller's caller's: alive for all of this function
                }

                Chain chain;
                auto *read = placeReadOf(root);
                if (read)
                {
                    chain = chainOf(read);
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
                    chain.roots.push_back({root, Chain::NotOwned});
                }

                // the root's own ends; and, for a read out of a place, whatever may destroy what the
                // place holds (a local or a made value is not destroyed by a call it is given)
                auto ends = rootEnds(chain);
                for (auto *drop : read ? llvm::ArrayRef<mlir::Operation *>(drops) : llvm::ArrayRef<mlir::Operation *>())
                {
                    if (drop != call && dropsChain(drop, chain, read))
                    {
                        ends.push_back(drop);
                    }
                }

                auto owner = ownerName(root);
                for (auto *end : ends)
                {
                    if (end == call)
                    {
                        continue;
                    }

                    for (auto *use : uses)
                    {
                        if (use != end && reachableAfter(end, use, call))
                        {
                            reportBorrowOutlives(use, end, name, owner);
                            return;
                        }
                    }
                }
            }
        });
    }

    // The ends of a capture box given to a call with a bounded result: each release of a closure on
    // it (the box goes with its last), and each end of what a field of it was filled from. Reports
    // and returns false when a use of the result comes after one.
    bool checkBoundedByBox(mlir::Operation *call, mlir::Value box, llvm::ArrayRef<mlir::Operation *> uses,
                           llvm::StringRef name)
    {
        llvm::SmallVector<mlir::Operation *> ends;
        for (auto *user : box.getUsers())
        {
            if (auto boundOp = mlir::dyn_cast<mlir_ts::CreateBoundFunctionOp>(user); boundOp && boundOp.getThisVal() == box)
            {
                llvm::SmallVector<mlir::Operation *> closureUses;
                auto aliased = false;
                if (followClosure(boundOp.getResult(), closureUses, ends, aliased))
                {
                    reportPlaceBorrowEscapes(call, name, "a closure that escapes"); // the box may go anywhere
                    return false;
                }

                continue;
            }

            auto propertyRefOp = mlir::dyn_cast<mlir_ts::PropertyRefOp>(user);
            if (!propertyRefOp)
            {
                continue;
            }

            for (auto *fieldUser : propertyRefOp->getUsers())
            {
                auto storeOp = mlir::dyn_cast<mlir_ts::StoreOp>(fieldUser);
                if (!storeOp || storeOp.getReference() != propertyRefOp.getResult() || holdsNoBlock(storeOp.getValue()))
                {
                    continue;
                }

                auto root = rootOf(storeOp.getValue());
                Chain chain;
                if (auto slotLoad = slotLoadOf(root))
                {
                    chain.roots.push_back({slotLoad.getReference(), Chain::Slot});
                }
                else if (borrowedParam(root) >= 0 || isImmortalLiteral(root))
                {
                    continue;
                }
                else if (isFresh(root))
                {
                    chain.roots.push_back({root, Chain::Owned});
                }
                else
                {
                    reportPlaceBorrowEscapes(call, name, "a value this function cannot bound");
                    return false;
                }

                auto rootEndsOf = rootEnds(chain);
                ends.append(rootEndsOf.begin(), rootEndsOf.end());
            }
        }

        for (auto *end : ends)
        {
            for (auto *use : uses)
            {
                if (end != call && use != end && reachableAfter(end, use, call))
                {
                    reportBorrowOutlives(use, end, name, "the generator's source");
                    return false;
                }
            }
        }

        return true;
    }

    // Every use of a bounded result, through its views, the locals that hold it, and the function
    // and object a call through it is made with. Answers the first use that keeps it - anything
    // that is not a read, a call, a local, or rc's bookkeeping - or null.
    mlir::Operation *boundedUses(mlir::Value result, llvm::SmallVectorImpl<mlir::Operation *> &uses)
    {
        llvm::SmallVector<mlir::Value> values{result};
        llvm::SmallPtrSet<mlir::Value, 8> seen;
        mlir::Operation *kept = nullptr;
        while (!values.empty() && !kept)
        {
            auto value = values.pop_back_val();
            if (!seen.insert(value).second)
            {
                continue;
            }

            forEachUse(value, [&](mlir::Operation *user, mlir::Value used) {
                if (kept || mlir::isa<mlir_ts::RetainOp, mlir_ts::ReleaseOp, mlir_ts::DebugVariableOp>(user))
                {
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

                if (slot)
                {
                    auto slotVar = slot.getDefiningOp<mlir_ts::VariableOp>();
                    if (!slotVar || slotVar.getCaptured().value_or(false))
                    {
                        kept = user;
                        return;
                    }

                    for (auto *slotUser : slot.getUsers())
                    {
                        if (auto loadOp = mlir::dyn_cast<mlir_ts::LoadOp>(slotUser))
                        {
                            values.push_back(loadOp.getResult());
                        }
                    }

                    return;
                }

                uses.push_back(user);
                if (isCall(user))
                {
                    if (isKeptArgument(user, used))
                    {
                        kept = user;
                    }

                    return;
                }

                // `it.next()`: the field read, the bound function, its `this`
                if (mlir::isa<mlir_ts::PropertyRefOp, mlir_ts::LoadOp, mlir_ts::GetThisOp, mlir_ts::GetMethodOp>(user))
                {
                    for (auto partResult : user->getResults())
                    {
                        values.push_back(partResult);
                    }

                    return;
                }

                if (!isBorrow(user, used))
                {
                    kept = user;
                }
            });
        }

        return kept;
    }

    // A generator's maker puts what the generator captures - its parameters, an outer local - into
    // a box, and the box into its state object's `.captured` field: the box of a closure that
    // escapes, with the state object, at that store. There is no closure value, so no use runs it
    // here and nothing calls through it. rc's `ts.RetainCell` of the box is the state object's
    // reference, which moves in with it. A box used any other way is left to the RetainCell error.
    void classifyStateBoxes()
    {
        getFunction().walk([&](mlir_ts::VariableOp boxOp) {
            auto box = boxOp.getResult();
            if (!boxOp->hasAttr(CAPTURE_BOX_ATTR_NAME) || closureOfBox(box))
            {
                return;
            }

            auto closure = std::make_shared<Closure>();
            closure->box = box;
            llvm::SmallVector<mlir::Operation *> boxRetains;
            for (auto *user : box.getUsers())
            {
                if (auto propertyRefOp = mlir::dyn_cast<mlir_ts::PropertyRefOp>(user))
                {
                    for (auto *fieldUser : propertyRefOp->getUsers())
                    {
                        auto storeOp = mlir::dyn_cast<mlir_ts::StoreOp>(fieldUser);
                        if (!storeOp || storeOp.getReference() != propertyRefOp.getResult())
                        {
                            return;
                        }

                        closure->fills.push_back({propertyRefOp.getPosition(), storeOp});
                        closure->hasCells = closure->hasCells || mlir::isa<mlir_ts::RefType>(storeOp.getValue().getType());
                    }

                    continue;
                }

                if (mlir::isa<mlir_ts::RetainCellOp>(user))
                {
                    boxRetains.push_back(user);
                    continue;
                }

                if (!closure->escape && isStoreIntoState(user, box))
                {
                    closure->escape = user;
                    continue;
                }

                return;
            }

            if (!closure->escape)
            {
                return;
            }

            auto stateRef = mlir::cast<mlir_ts::StoreOp>(closure->escape).getReference().getDefiningOp<mlir_ts::PropertyRefOp>();
            auto stateType = mlir::cast<mlir_ts::RefType>(stateRef->getOperand(0).getType()).getElementType();
            auto borrowing = borrowingFieldsOf(stateType);
            closure->borrowsValues = !borrowing.empty() && borrowing.front() == stateRef.getPosition();
            // `.map`'s maker copies what its parameter box holds into the generator's box: a borrow
            if (closure->borrowsValues)
            {
                for (auto &fill : closure->fills)
                {
                    if (!mlir::isa<mlir_ts::RefType>(fill.second.getValue().getType()))
                    {
                        setAttrTracked(fill.second, OWN_CAPTURE_BORROW_ATTR_NAME);
                    }
                }
            }

            for (auto &fill : closure->fills)
            {
                if (auto *retain = retainBefore(fill.second.getValue(), fill.second))
                {
                    closure->retains.push_back(retain);
                    captureRetains.insert(retain);
                }
            }

            for (auto *retain : boxRetains)
            {
                closure->retains.push_back(retain);
                captureRetains.insert(retain);
            }

            stateBoxes[box] = closure.get();
            closures.push_back(std::move(closure));
        });
    }

    // `ts.Store(box, ts.PropertyRef(state, i))`, where `state` is a local a `ts.Constant`
    // initializes whose every read fills a block `ts.New` made: a generator's initial state.
    static bool isStoreIntoState(mlir::Operation *user, mlir::Value box)
    {
        auto storeOp = mlir::dyn_cast<mlir_ts::StoreOp>(user);
        auto propertyRefOp = storeOp && storeOp.getValue() == box
                                 ? storeOp.getReference().getDefiningOp<mlir_ts::PropertyRefOp>()
                                 : mlir_ts::PropertyRefOp();
        auto stateOp = propertyRefOp ? propertyRefOp->getOperand(0).getDefiningOp<mlir_ts::VariableOp>() : mlir_ts::VariableOp();
        if (!stateOp || !isConstantRecordLocal(stateOp))
        {
            return false;
        }

        auto reads = 0;
        for (auto *stateUser : stateOp->getUsers())
        {
            auto loadOp = mlir::dyn_cast<mlir_ts::LoadOp>(stateUser);
            if (!loadOp)
            {
                continue;
            }

            ++reads;
            for (auto *readUser : loadOp->getUsers())
            {
                // rc's retain of the initial state, for fields that own (holdsConstantsOrMoves)
                if (mlir::isa<mlir_ts::RetainOp>(readUser))
                {
                    continue;
                }

                auto fillOp = mlir::dyn_cast<mlir_ts::StoreOp>(readUser);
                if (!fillOp || fillOp.getValue() != loadOp.getResult() || !fillOp.getReference().getDefiningOp<mlir_ts::NewOp>())
                {
                    return false;
                }
            }
        }

        return reads > 0;
    }

    // rc's reference for the box's copy of `value`: the retain of it nearest before the store that
    // fills the box, in the same block.
    mlir::Operation *retainBefore(mlir::Value value, mlir::Operation *store)
    {
        // rc's retain of an earlier read of the same place (twinReadAfter)
        if (auto loadOp = value.getDefiningOp<mlir_ts::LoadOp>())
        {
            for (auto *user : loadOp.getReference().getUsers())
            {
                auto earlier = mlir::dyn_cast<mlir_ts::LoadOp>(user);
                auto *retain = earlier && earlier.getResult().hasOneUse() ? *earlier->getUsers().begin() : nullptr;
                if (retain && mlir::isa<mlir_ts::RetainOp>(retain) && !captureRetains.contains(retain) &&
                    twinReadAfter(retain, earlier.getResult()) == value && retain->getBlock() == store->getBlock() &&
                    retain->isBeforeInBlock(store))
                {
                    return retain;
                }
            }
        }

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
            if (!closure->op)
            {
                decideEscaping(*closure, toErase); // a generator's box
                continue;
            }

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

            if (!closure->hasCells && !closure->borrowsCopies)
            {
                keepAliasedReleases();
                continue;
            }

            if (closure->escape)
            {
                decideEscaping(*closure, toErase);
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
                            if (mlir::isa<mlir_ts::ReleaseCellOp>(user) || movesCellAway(user))
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
                    if (holdsNoBlock(value) || placeReadOf(root) || borrowedParam(root) >= 0 || isHandle(root))
                    {
                        // nothing to own, a read the read's check bounds, a parameter, or a handle,
                        // whose copy in the box is counted (spec 23.2)
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

    // Is this the store that moves a cell into the box of a closure that escapes?
    bool movesCellAway(mlir::Operation *op)
    {
        auto storeOp = mlir::dyn_cast<mlir_ts::StoreOp>(op);
        auto propertyRefOp = storeOp ? storeOp.getReference().getDefiningOp<mlir_ts::PropertyRefOp>() : mlir_ts::PropertyRefOp();
        if (!propertyRefOp || !mlir::isa<mlir_ts::RefType>(storeOp.getValue().getType()))
        {
            return false;
        }

        if (stateBoxes.count(propertyRefOp.getObjectRef()))
        {
            return true;
        }

        auto boundOp = closureOfBox(propertyRefOp.getObjectRef());
        auto found = boundOp ? closureOf.find(boundOp.getOperation()) : closureOf.end();
        return found != closureOf.end() && found->second->escape;
    }

    // A closure that escapes owns what its box holds (spec 2.5, phase 5b). Its copies moved in as
    // in phase 4. Each cell moves in where the box takes it: that store is the move, rc's
    // `ts.RetainCell` for it goes, and so does each of the frame's `ts.ReleaseCell`s the move reaches,
    // which it must dominate. Nothing in the frame may use the variable after the move - read it,
    // assign it, capture it again - and the move may not come round a loop to a cell made outside
    // it. A cell that borrows its value (a parameter's) or one an enclosing closure holds has
    // nothing to move. And nothing may call through the box once the closure has gone.
    void decideEscaping(Closure &closure, llvm::SetVector<mlir::Operation *> &toErase)
    {
        auto ok = true;
        for (auto &fill : closure.fills)
        {
            auto value = fill.second.getValue();
            mlir::Operation *move = fill.second;
            if (!mlir::isa<mlir_ts::RefType>(value.getType()))
            {
                continue;
            }

            if (isLoadedCell(value))
            {
                reportEscapingCellNotOwned(move, closure.escape, "it is captured from an enclosing function");
                ok = false;
                continue;
            }

            auto varOp = isCellVariable(value) ? value.getDefiningOp<mlir_ts::VariableOp>() : mlir_ts::VariableOp();
            if (!varOp)
            {
                reportCaptureNotOwned(move, "this value");
                ok = false;
                continue;
            }

            auto name = varName(varOp);
            MLIRTypeHelper mth(&getContext(), CompileOptions{});
            auto elementType = mlir::cast<mlir_ts::RefType>(value.getType()).getElementType();
            // a handle's cell holds a count of its own, whoever's the value was (spec 23.2)
            if (!isOwningVariable(varOp) && !closure.borrowsValues && mth.ownsHeapMemory(varOp.getLoc(), elementType) &&
                !MLIRTypeHelper::isSharedHandleType(elementType))
            {
                reportEscapingCellNotOwned(move, closure.escape, "its value is the caller's");
                ok = false;
                continue;
            }

            if (takerLoops(move, varOp))
            {
                reportMovedInLoop(move, name);
                ok = false;
                continue;
            }

            llvm::SmallVector<mlir::Operation *> moved;
            for (auto *user : value.getUsers())
            {
                if (user == move || llvm::is_contained(closure.retains, user) || !reachableAfter(move, user, varOp))
                {
                    continue;
                }

                if (!mlir::isa<mlir_ts::ReleaseCellOp>(user))
                {
                    reportUsedAfterEscapingCapture(user, move, name);
                    ok = false;
                    break;
                }

                if (!dominance->properlyDominates(move, user))
                {
                    reportMovedOnSomePaths(move, user, name);
                    ok = false;
                    break;
                }

                moved.push_back(user);
            }

            toErase.insert(moved.begin(), moved.end());
        }

        // A call through the box itself after the closure was given away or given back. followClosure
        // stops at the escape, so `ends` may miss releases past it; that is enough while the only
        // calls through a box are those of a folded `const f`, whose release ends its block. A
        // generator's box has no closure value and no such call.
        if (!closure.op)
        {
            toErase.insert(closure.retains.begin(), closure.retains.end());
            return;
        }

        auto name = ownerName(closure.op.getResult());
        llvm::SmallVector<mlir::Operation *> gone(closure.ends.begin(), closure.ends.end());
        gone.push_back(closure.escape);
        for (auto *end : gone)
        {
            for (auto *use : closure.boxUses)
            {
                if (ok && use != end && reachableAfter(end, use, closure.op.getOperation()))
                {
                    reportUseAfterMove(use, end, name);
                    ok = false;
                }
            }
        }

        toErase.insert(closure.retains.begin(), closure.retains.end());
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
            auto elementType = mlir::cast<mlir_ts::RefType>(cell.getType()).getElementType();
            if (!mth.ownsHeapMemory(varOp.getLoc(), elementType))
            {
                return;
            }

            // A handle's cell holds a count of its own, taken where it is made (spec 23.2): its
            // release gives that back, as under rc, and an assignment is a counted copy.
            if (MLIRTypeHelper::isSharedHandleType(elementType))
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
                    setAttrTracked(user, OWN_CELL_BORROWS_ATTR_NAME);
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
                borrowWith(borrower, receiver.retain, slotEnds(slot), owner, toErase);
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

        auto &body = getFunction().getFunctionBody();
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
        return argument && argument.getOwner()->isEntryBlock() && getFunction().getName().starts_with(".eh.copy.");
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
                borrowWith(borrower, receiver.retain, ends, owner, toErase);
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
        else if (auto retainOp = mlir::dyn_cast<mlir_ts::RetainOp>(acquisition))
        {
            // rc's reference for a `let` that takes the value over: `let ri = <I>raw` retains the
            // interface and declares `ri` consuming it
            varOp = consumingLetOf(retainOp.getReference());
            consumed = true;
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

    // tryBorrow for the `let` an acquisition gives a reference: a `ts.Retain` of the value a
    // consuming `let` takes over is that `let`'s, and goes with the borrow; a `ts.RetainSlot` is the
    // `let`'s own bookkeeping already.
    bool borrowWith(mlir_ts::VariableOp borrower, mlir::Operation *acquisition, llvm::ArrayRef<mlir::Operation *> ends,
                    llvm::StringRef owner, llvm::SetVector<mlir::Operation *> &toErase)
    {
        if (!tryBorrow(borrower, ends, owner, toErase))
        {
            return false;
        }

        if (mlir::isa<mlir_ts::RetainOp>(acquisition))
        {
            toErase.insert(acquisition);
        }

        return true;
    }

    // The owning `let` declared from `value` that takes over its reference, or none.
    static mlir_ts::VariableOp consumingLetOf(mlir::Value value)
    {
        for (auto *user : value.getUsers())
        {
            auto varOp = mlir::dyn_cast<mlir_ts::VariableOp>(user);
            if (varOp && varOp.getInitializer() == value && varOp->hasAttr(OWNED_LOCAL_CONSUMED_ATTR_NAME))
            {
                return varOp;
            }
        }

        return {};
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

                // rc's reference for a borrowing box's copy goes with the closure (decideClosures), and
                // one for a borrowing field's with the field (classifyBorrowingFields)
                if (captureRetains.contains(user) || fieldBorrowRetains.contains(user))
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

                // a copy (spec 22.2) reads the borrow into a block of its own, which points into nothing
                if (mlir::isa<mlir_ts::StringCopyOp>(user))
                {
                    continue;
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
        auto callee = isCall(user) ? calleeValue(user) : mlir::Value();
        if (!callee)
        {
            return false;
        }

        if (use.getOperandNumber() == 0)
        {
            return true;
        }

        auto getThisOp = use.get().getDefiningOp<mlir_ts::GetThisOp>();
        auto getMethodOp = callee.getDefiningOp<mlir_ts::GetMethodOp>();
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
        // a handle read out of a place that rc gives an owner of its own is a counted copy, not a
        // borrow (spec 23.3); any other is a borrow of the place, as a field read is
        if (!def || !ownsHeap(root) || (isHandle(root) && isCountedHandleRead(root)))
        {
            return nullptr;
        }

        if (auto loadOp = mlir::dyn_cast<mlir_ts::LoadOp>(def); loadOp && isPlace(loadOp.getReference()))
        {
            // a field of a generator's borrowing box: only its maker fills it, and nothing a call
            // can reach overwrites it; it is the caller's, as a parameter is (checkBorrowedReads)
            if (isBorrowingBoxField(loadOp.getReference()))
            {
                return nullptr;
            }

            return def;
        }

        return borrowingCall(root);
    }

    // A handle read out of a place that rc gives a count of its own (spec 23.3): retained, which
    // own leaves in place - a store into a field, an element or a global, a push, `new Shared`,
    // an `any` box, a return - or the initializer of an owning local, which takes its own count
    // (`ts.RetainSlot`). `cur = cur.value.next` is one. Seen through casts, merges and locals
    // that own nothing. A handle read with none of these - an argument given straight to a call
    // (`f(p, p.child)`), a `.value` read through it - is a borrow of the place.
    static bool isCountedHandleRead(mlir::Value value)
    {
        llvm::SmallVector<mlir::Value> values{value};
        llvm::SmallPtrSet<mlir::Value, 8> seen{value};
        while (!values.empty())
        {
            auto current = values.pop_back_val();
            for (auto &use : current.getUses())
            {
                auto *user = use.getOwner();
                if (mlir::isa<mlir_ts::RetainOp>(user))
                {
                    return true;
                }

                if (auto varOp = mlir::dyn_cast<mlir_ts::VariableOp>(user); varOp && isOwningVariable(varOp) &&
                    llvm::any_of(varOp->getUsers(), [](mlir::Operation *varUser) {
                        return mlir::isa<mlir_ts::RetainSlotOp>(varUser);
                    }))
                {
                    return true;
                }

                llvm::SmallVector<mlir::Value> same;
                if (passesOn(use, same))
                {
                    for (auto next : same)
                    {
                        if (isHandle(next) && seen.insert(next).second)
                        {
                            values.push_back(next);
                        }
                    }
                }
            }
        }

        return false;
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

        // a field's position and reference type; an element is position -1, a handle's payload
        // (`s.value`) is SharedValue
        llvm::SmallVector<std::pair<int64_t, mlir::Type>> places;
        static constexpr int64_t SharedValue = -2;
        llvm::SmallVector<std::pair<mlir::Value, Kind>> roots;
        // The roots that are handles - what a `s.value` was read through - each also NotOwned
        // among `roots` (spec 23.3). Known from the op, not the root's type: a view may hide it.
        llvm::SmallVector<mlir::Value> handles;
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
        // the bases a `s.value` was read through, and what merges into them
        llvm::SmallPtrSet<mlir::Value, 4> handleBases;
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
                else if (auto sharedValueRefOp = ref.getDefiningOp<mlir_ts::SharedValueRefOp>())
                {
                    chain.places.push_back({Chain::SharedValue, sharedValueRefOp.getType()});
                    base = sharedValueRefOp.getShared();
                    handleBases.insert(rootOf(base));
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

                // A handle (spec 23.3): other handles may reach its block, so this function owns
                // nothing under it. A handle read out of a place still has that place above it, and
                // one merged from several has each of them.
                if (handleBases.contains(base) || isHandle(base))
                {
                    llvm::SmallVector<mlir::Value> merged;
                    if (auto argument = mlir::dyn_cast<mlir::BlockArgument>(base); argument && mergedInto(argument, merged))
                    {
                        handleBases.insert(merged.begin(), merged.end());
                        bases.append(merged.begin(), merged.end());
                        continue;
                    }

                    chain.roots.push_back({base, Chain::NotOwned});
                    chain.handles.push_back(base);
                    if (auto loadOp = base.getDefiningOp<mlir_ts::LoadOp>(); loadOp && isPlace(loadOp.getReference()))
                    {
                        refs.push_back(loadOp.getReference());
                    }

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
            // A handle (spec 23.3), never owned: where this one is given back or overwritten - a
            // local's release and assignments, a temporary's release; a global's assignments are
            // below. A copy of it is counted, and moves nothing away.
            if (llvm::is_contained(chain.handles, root))
            {
                auto loadOp = root.getDefiningOp<mlir_ts::LoadOp>();
                if (auto varOp = loadOp ? loadOp.getReference().getDefiningOp<mlir_ts::VariableOp>() : mlir_ts::VariableOp())
                {
                    for (auto *user : varOp->getUsers())
                    {
                        if (mlir::isa<mlir_ts::ReleaseSlotOp, mlir_ts::StoreOp>(user))
                        {
                            ends.push_back(user);
                        }
                    }
                }
                else if (!loadOp)
                {
                    forEachUse(root, [&](mlir::Operation *use, mlir::Value) {
                        if (mlir::isa<mlir_ts::ReleaseOp>(use))
                        {
                            ends.push_back(use);
                        }
                    });
                }
            }

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

            // `x.value = ...` through any handle, of any type, may replace the payload another
            // handle reads (spec 23.3): a handle converts to one of a base class's type
            if (slot.getDefiningOp<mlir_ts::SharedValueRefOp>())
            {
                return llvm::any_of(chain.places, [&](auto &place) { return place.first == Chain::SharedValue; });
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

        // pop, shift, splice, `length =`; push and unshift on an array reached through a handle
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
        auto value = calleeValue(call);
        if (auto symbol = directCallee(call))
        {
            callee = symbol.getAttr();
        }
        else if (auto calleeOp = value ? value.getDefiningOp() : nullptr)
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
        else if (auto sharedValueRefOp = ref.getDefiningOp<mlir_ts::SharedValueRefOp>())
        {
            base = sharedValueRefOp.getShared();
            step = ".value";
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

    // A `ts.RetainSlot`'s slot holds its initializer where the retain is. MLIRGen puts the retain at
    // the declaration (takeOwnershipOfLocal), and nothing between the two may write the slot: in
    // their block, no op is given the slot, no call runs, and no op has regions (a store could hide
    // in them). What is stored into the slot later is decided at its own store; this retain takes
    // only the initializer. No closure captures it.
    static bool slotHoldsInitializerAt(mlir::Operation *op)
    {
        auto slot = mlir::cast<mlir_ts::RetainSlotOp>(op).getSlot();
        auto varOp = slot.getDefiningOp<mlir_ts::VariableOp>();
        if (!varOp || !varOp.getInitializer() || isCellVariable(slot) || varOp->getBlock() != op->getBlock())
        {
            return false;
        }

        for (auto *between = varOp->getNextNode(); between; between = between->getNextNode())
        {
            if (between == op)
            {
                return true;
            }

            if (llvm::is_contained(between->getOperands(), slot) || isCall(between) || between->getNumRegions() > 0)
            {
                return false;
            }
        }

        return false;
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
        // a `value_ref` points at storage it does not own, but one `ts.New` made and seen as its
        // object is the object's block: the root of a generator's state object or an object literal
        if (value.getDefiningOp<mlir_ts::NewOp>() && mlir::isa<mlir_ts::ValueRefType>(value.getType()) &&
            llvm::any_of(value.getUsers(), [](mlir::Operation *user) { return mlir::isa<mlir_ts::CastOp>(user) && isView(user); }))
        {
            return true;
        }

        return ownsHeap(value.getLoc(), value.getType());
    }

    bool ownsHeap(mlir::Location location, mlir::Type type)
    {
        MLIRTypeHelper mth(&getContext(), CompileOptions{});
        return mth.ownsHeapMemory(location, type);
    }

    // A tuple read out of a local that a `ts.Constant` initializes, whose owning fields hold what
    // that constant put there - null, a number, an immortal literal - or a fresh value moved in:
    // the initial state a generator's maker copies into its state object, or a record built to be
    // returned (`return {value: new C(), done: false}`, a `yield`'s result). rc's retain of it is
    // its reference for the values moved in, which own moves instead (recordRetainOf).
    bool holdsConstantsOrMoves(mlir::Value value)
    {
        auto loadOp = value.getDefiningOp<mlir_ts::LoadOp>();
        auto varOp = loadOp ? loadOp.getReference().getDefiningOp<mlir_ts::VariableOp>() : mlir_ts::VariableOp();
        if (!varOp || !isConstantRecordLocal(varOp))
        {
            return false;
        }

        for (auto *user : varOp->getUsers())
        {
            auto propertyRefOp = mlir::dyn_cast<mlir_ts::PropertyRefOp>(user);
            if (!propertyRefOp)
            {
                continue;
            }

            // a method's field is a `!ts.bound_ref`, which holds a function, not a block
            auto field = propertyRefOp.getResult();
            auto refType = mlir::dyn_cast<mlir_ts::RefType>(field.getType());
            for (auto *fieldUser : field.getUsers())
            {
                if (mlir::isa<mlir_ts::LoadOp>(fieldUser))
                {
                    continue;
                }

                if (refType && !ownsHeap(field.getLoc(), refType.getElementType()))
                {
                    continue;
                }

                // a fresh value moved in, its move decided by its own check against this retain
                auto storeOp = mlir::dyn_cast<mlir_ts::StoreOp>(fieldUser);
                if (!storeOp || storeOp.getReference() != field || !isFresh(rootOf(storeOp.getValue())) ||
                    recordRetainOf(varOp) == nullptr)
                {
                    return false;
                }
            }
        }

        return true;
    }

    // A non-owning local a `ts.Constant` initializes, used only through its fields and its reads.
    static bool isConstantRecordLocal(mlir_ts::VariableOp varOp)
    {
        if (isOwningVariable(varOp) || varOp.getCaptured().value_or(false) || !varOp.getInitializer() ||
            !varOp.getInitializer().getDefiningOp<mlir_ts::ConstantOp>())
        {
            return false;
        }

        return llvm::all_of(varOp->getUsers(), [](mlir::Operation *user) {
            return mlir::isa<mlir_ts::LoadOp, mlir_ts::PropertyRefOp>(user);
        });
    }

    // rc's one `ts.Retain` of a record local's one read, where the read is only returned: stored
    // into a local that owns nothing and is read only by returns (the result slot). It is the
    // reference rc gives each fresh value stored into the record, so it is where they move in.
    // Null for any other record.
    static mlir::Operation *recordRetainOf(mlir_ts::VariableOp varOp)
    {
        if (!isConstantRecordLocal(varOp))
        {
            return nullptr;
        }

        mlir_ts::LoadOp read;
        for (auto *user : varOp->getUsers())
        {
            if (auto loadOp = mlir::dyn_cast<mlir_ts::LoadOp>(user))
            {
                if (read)
                {
                    return nullptr;
                }

                read = loadOp;
            }
        }

        if (!read)
        {
            return nullptr;
        }

        mlir::Operation *retain = nullptr;
        for (auto *user : read->getUsers())
        {
            if (mlir::isa<mlir_ts::RetainOp>(user))
            {
                if (retain)
                {
                    return nullptr;
                }

                retain = user;
                continue;
            }

            if (mlir::isa<mlir_ts::ReturnInternalOp>(user))
            {
                continue;
            }

            // `let a = { item: new C() }`: the read is what an owning local starts with, and rc's
            // `ts.RetainSlot` of that local is the record's retain
            if (auto ownerOp = mlir::dyn_cast<mlir_ts::VariableOp>(user))
            {
                auto *slotRetain = ownerOp.getInitializer() == read.getResult() && isOwningVariable(ownerOp)
                                       ? onlyRetainSlotOf(ownerOp)
                                       : nullptr;
                if (!slotRetain || retain)
                {
                    return nullptr;
                }

                retain = slotRetain;
                continue;
            }

            // stored into the result slot, or into an owning local (`a = { item: new C() }`)
            auto storeOp = mlir::dyn_cast<mlir_ts::StoreOp>(user);
            auto slot = storeOp && storeOp.getValue() == read.getResult()
                            ? storeOp.getReference().getDefiningOp<mlir_ts::VariableOp>()
                            : mlir_ts::VariableOp();
            if (!slot || !(isResultSlot(slot) || isOwningVariable(slot)))
            {
                return nullptr;
            }
        }

        return retain;
    }

    // `const b = [...a, ...a]`: MLIRGen builds the array in a local of its own that owns nothing,
    // starting from a fresh one and pushing into it, and reads it once, when it is done, into the
    // owning local it is made for. That local's `ts.RetainSlot` (`retain`) is the reference: the
    // fresh array moves into the builder and then into the owner, and nothing else holds it.
    bool isBuiltArrayRead(mlir::Operation *retain, mlir::Value value)
    {
        auto read = value.getDefiningOp<mlir_ts::LoadOp>();
        auto builder = read ? read.getReference().getDefiningOp<mlir_ts::VariableOp>() : mlir_ts::VariableOp();
        auto owner = mlir::cast<mlir_ts::RetainSlotOp>(retain).getSlot().getDefiningOp<mlir_ts::VariableOp>();
        if (!builder || isOwningVariable(builder) || builder.getCaptured().value_or(false) || !builder.getInitializer() ||
            !mlir::isa_and_nonnull<mlir_ts::CreateArrayOp, mlir_ts::NewArrayOp>(builder.getInitializer().getDefiningOp()) ||
            !owner || owner.getInitializer() != read.getResult() || !read->hasOneUse() || onlyRetainSlotOf(owner) != retain)
        {
            return false;
        }

        // only filled, and only before it is read
        return llvm::all_of(builder->getUses(), [&](mlir::OpOperand &use) {
            auto *user = use.getOwner();
            if (user == read)
            {
                return true;
            }

            return mlir::isa<mlir_ts::ArrayPushOp, mlir_ts::ArrayUnshiftOp>(user) && use.getOperandNumber() == 0 &&
                   !reachableAfter(read, user, builder);
        });
    }

    // The one `ts.RetainSlot` of a local; null when there are none or several.
    static mlir::Operation *onlyRetainSlotOf(mlir_ts::VariableOp varOp)
    {
        mlir::Operation *found = nullptr;
        for (auto *user : varOp->getUsers())
        {
            if (mlir::isa<mlir_ts::RetainSlotOp>(user))
            {
                if (found)
                {
                    return nullptr;
                }

                found = user;
            }
        }

        return found;
    }

    // A local that owns nothing, is never captured, and is read only to be returned.
    static bool isResultSlot(mlir_ts::VariableOp varOp)
    {
        if (isOwningVariable(varOp) || varOp.getCaptured().value_or(false))
        {
            return false;
        }

        return llvm::all_of(varOp->getUsers(), [&](mlir::Operation *user) {
            if (auto storeOp = mlir::dyn_cast<mlir_ts::StoreOp>(user))
            {
                return storeOp.getReference() == varOp.getResult();
            }

            auto loadOp = mlir::dyn_cast<mlir_ts::LoadOp>(user);
            return loadOp && llvm::all_of(loadOp->getUsers(), [](mlir::Operation *loadUser) {
                       return mlir::isa<mlir_ts::ReturnInternalOp>(loadUser);
                   });
        });
    }

    // The record whose field `taker` stores into, and rc's retain that moves the stored value in;
    // null when `taker` is not such a store.
    static mlir::Operation *recordRetainOfStore(mlir::Operation *taker)
    {
        auto storeOp = mlir::dyn_cast<mlir_ts::StoreOp>(taker);
        auto propertyRefOp = storeOp ? storeOp.getReference().getDefiningOp<mlir_ts::PropertyRefOp>() : mlir_ts::PropertyRefOp();
        auto varOp = propertyRefOp ? propertyRefOp->getOperand(0).getDefiningOp<mlir_ts::VariableOp>() : mlir_ts::VariableOp();
        auto *retain = varOp ? recordRetainOf(varOp) : nullptr;
        if (!retain || retain->getBlock() != taker->getBlock() || !taker->isBeforeInBlock(retain))
        {
            return nullptr;
        }

        return retain;
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
                      mlir_ts::StringResizeOp, mlir_ts::StringLengthOp, mlir_ts::StringCopyOp>(user))
        {
            return true;
        }

        // `s.value`'s place and `Shared.count(s)` read the handle and keep nothing (spec 23.1)
        if (mlir::isa<mlir_ts::SharedValueRefOp, mlir_ts::SharedCountOp>(user))
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

        if (mlir::isa<mlir_ts::ExtractInterfaceThisOp>(user))
        {
            return isBoundForCall(user);
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

        // a store through a `ts.New`'d `value_ref` fills its block and keeps nothing of it
        if (auto storeOp = mlir::dyn_cast<mlir_ts::StoreOp>(user))
        {
            return mlir::isa<mlir_ts::ValueRefType>(value.getType()) && storeOp.getReference() == value &&
                   storeOp.getValue() != value;
        }

        return false;
    }

    // `ts.CreateBoundFunction(ts.Cast(object), method)` split straight back into `ts.GetThis` and
    // `ts.GetMethod` for a call: a method of another module's class - its constructor, when it is
    // built with `new` - called on an object here. It reads the object as `ts.ThisSymbolRef` does
    // for a method of this module; nothing else may use the cast or the bound function, and the
    // object comes out again only as a call's argument, which is a borrow of its own. An
    // interface's `this` (`ts.ExtractInterfaceThis`) is bound so for a call of a function-typed
    // field (`i.toString()` where `toString: () => string`).
    static bool isBoundForCall(mlir::Operation *thisOp)
    {
        if (thisOp->getNumResults() != 1 || !mlir::isa<mlir_ts::OpaqueType>(thisOp->getResult(0).getType()) ||
            thisOp->use_empty())
        {
            return false;
        }

        return llvm::all_of(thisOp->getUses(), [](mlir::OpOperand &use) {
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
        // container owns nothing either: checkPlaceRead decides it. Nor does other data nothing
        // owns: an empty optional, a function, a tuple of literals (isConstantData).
        if (!ownsHeap(value) || isImmortalLiteral(value) || isConstantData(value) || placeReadOf(value))
        {
            return true;
        }

        // A temporary's releases are one owner, one release on each way out. Two on one path are
        // two owners: each view rc gave a reference of its own (`use(o); use(o)`, with `o` a
        // folded object literal and `use` taking an interface: two `ts.NewInterface`s of one block).
        if (!checkReleasedOnce(value))
        {
            return false;
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

    // No release of `value` comes after another on any path. Reports and returns false otherwise.
    bool checkReleasedOnce(mlir::Value value)
    {
        llvm::SmallVector<mlir::Operation *> releases;
        forEachUse(value, [&](mlir::Operation *user, mlir::Value) {
            if (mlir::isa<mlir_ts::ReleaseOp>(user))
            {
                releases.push_back(user);
            }
        });

        auto *definition = value.getDefiningOp();
        for (auto *first : releases)
        {
            for (auto *second : releases)
            {
                if (first != second && reachableAfter(first, second, definition))
                {
                    // the view the second release gives back, where rc took its reference
                    auto *view = second->getOperand(0).getDefiningOp();
                    reportSecondReference(view ? view : second, value);
                    return false;
                }
            }
        }

        return true;
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
                reportSecondReference(taker, value);
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
        auto &body = getFunction().getFunctionBody();
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

        auto &body = getFunction().getFunctionBody();
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

        // rc retained this read and hands the receiver the next read of the same place
        if (!taker && !several)
        {
            if (auto twin = twinReadAfter(retain, value))
            {
                return takerOf(retain, twin);
            }
        }

        return several ? nullptr : taker; // two takers of one read: not a move this phase proves
    }

    // rc reads a place twice for one copy - `.map(f)` boxes its array so - and retains the first
    // read, whose only use is that retain: the read of the same place that follows it in its block,
    // with nothing stored into the place between, is the same value. Null when there is none.
    static mlir::Value twinReadAfter(mlir::Operation *retain, mlir::Value value)
    {
        auto first = value.getDefiningOp<mlir_ts::LoadOp>();
        if (!first || !mlir::isa<mlir_ts::RetainOp>(retain) || !value.hasOneUse() || retain->getBlock() != first->getBlock())
        {
            return {};
        }

        auto place = first.getReference();
        for (auto *op = retain->getNextNode(); op; op = op->getNextNode())
        {
            if (auto loadOp = mlir::dyn_cast<mlir_ts::LoadOp>(op); loadOp && loadOp.getReference() == place)
            {
                return loadOp.getResult();
            }

            // anything else given the place may write through it - a store, a push - and a call may
            // write anything
            if (llvm::is_contained(op->getOperands(), place) || isCall(op))
            {
                return {};
            }
        }

        return {};
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
        // a store into a record, which rc retains as a whole after it is filled
        if (auto *retain = recordRetainOfStore(taker))
        {
            return retain;
        }

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

        if (recordRetainOfStore(taker))
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

    // A second reference nobody proved safe. `value` is the value it is to, when the caller knows
    // it; a retain names its own. The shapes this pass recognises get the rule they break (spec
    // 21.2); anything else, the catch-all.
    void reportSecondReference(mlir::Operation *op, mlir::Value value = {})
    {
        if (!reporting())
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

        if (!value && mlir::isa<mlir_ts::RetainOp, mlir_ts::RetainSlotOp>(op))
        {
            value = retainedValue(op);
        }

        if (!value || !reportShapeOfSecondReference(op, rootOf(value), name))
        {
            auto diag = op->emitError("'") << name << "' takes a second reference; -mm=own cannot prove a move or a borrow here yet";
            noteLostFacts(diag);
        }

        signalPassFailure();
    }

    // The named shapes of a second reference to `root` (spec 21.2). False when it is none of them.
    bool reportShapeOfSecondReference(mlir::Operation *op, mlir::Value root, llvm::StringRef name)
    {
        auto f = getFunction();

        // returned, where another return gives a borrow of an argument and this one does not, or
        // the other way round: the callers cannot be told which they get
        if (isReturned(root))
        {
            auto param = borrowedParam(root);
            if (auto *other = otherReturn(param))
            {
                auto diag = op->emitError("'") << f.getName()
                                               << "' returns a borrow of its argument on some paths and another value on "
                                                  "others; -mm=own cannot tell its callers whether they own the result";
                diag.attachNote(other->getLoc())
                    << (param >= 0 ? "returns another value here" : "returns a borrow of its argument here");
                return true;
            }
        }

        // a parameter's value, or a part of it, returned or kept: a fact its callers must know
        // (spec 15), and noteLostFacts says why they cannot
        if (auto param = borrowedParam(root); param >= 0)
        {
            auto diag = op->emitError("'") << (name == "this value" ? paramName(param) : name);
            if (isReturned(root))
            {
                diag << "' is a parameter, returned here, but -mm=own cannot tell the callers of '" << f.getName()
                     << "' that the result borrows it";
            }
            else
            {
                diag << "' is a parameter, kept here, but -mm=own cannot make the callers of '" << f.getName()
                     << "' give it up";
            }

            noteLostFacts(diag);
            return true;
        }

        // a store into a local whose value is not this function's to replace: a parameter's slot, or
        // a local that owns nothing (MLIRGen's takeOwnershipOfLocal)
        auto storeOp = mlir::dyn_cast<mlir_ts::StoreOp>(op);
        if (auto slot = storeOp ? storeOp.getReference().getDefiningOp<mlir_ts::VariableOp>() : mlir_ts::VariableOp();
            slot && !isOwningVariable(slot) && !isResultSlot(slot))
        {
            // a parameter's slot, assigned (isParameterSlot is the unassigned one)
            auto argument = mlir::dyn_cast_or_null<mlir::BlockArgument>(slot.getInitializer());
            if (argument && argument.getOwner()->isEntryBlock() &&
                mlir::isa<mlir_ts::FuncOp>(argument.getOwner()->getParentOp()))
            {
                op->emitError("'") << varName(slot) << "' is a parameter and cannot be assigned under -mm=own yet";
                return true;
            }

            // MLIRGen makes no owner of a local declared without a value, nor of one declared in a
            // catch or finally clause, whose release would be a call inside an exception funclet
            if (!slot.getInitializer())
            {
                op->emitError("'") << varName(slot)
                                   << "' owns nothing under -mm=own (it is declared without a value, or in a catch or "
                                      "finally clause), so it cannot be given a new value here";
                return true;
            }
        }

        // merged from branches or loop iterations
        if (auto argument = mlir::dyn_cast<mlir::BlockArgument>(root); argument && !argument.getOwner()->isEntryBlock())
        {
            op->emitError("'") << name
                               << "' is merged from several branches (`?:`, `&&`, `||`, `??`, a loop); -mm=own cannot "
                                  "prove which one owns it yet";
            return true;
        }

        auto loadOp = root.getDefiningOp<mlir_ts::LoadOp>();

        // a read of a global
        if (auto addressOfOp = loadOp ? loadOp.getReference().getDefiningOp<mlir_ts::AddressOfOp>() : mlir_ts::AddressOfOp())
        {
            // a local that reads it: phase 3's borrow, which a call or an assignment of the global
            // may end
            if (mlir::isa<mlir_ts::RetainSlotOp>(op))
            {
                op->emitError("'") << name << "' borrows the global '" << addressOfOp.getGlobalName()
                                   << "', but -mm=own cannot prove the global keeps its value while '" << name
                                   << "' is used: a call or an assignment may overwrite it";
                return true;
            }

            op->emitError("'") << name << "' borrows the global '" << addressOfOp.getGlobalName()
                               << "' and cannot be stored, returned or captured";
            return true;
        }

        // a read of a captured variable, through its cell
        if (loadOp && isLoadedCell(loadOp.getReference()))
        {
            op->emitError("'") << name << "' borrows a captured variable and cannot be stored, returned or captured";
            return true;
        }

        // an object literal built in a local, holding a value it owns: moved only into an owning
        // local or the result (recordRetainOf)
        auto record = loadOp ? loadOp.getReference().getDefiningOp<mlir_ts::VariableOp>() : mlir_ts::VariableOp();
        if (!record && storeOp)
        {
            auto propertyRefOp = storeOp.getReference().getDefiningOp<mlir_ts::PropertyRefOp>();
            record = propertyRefOp ? propertyRefOp->getOperand(0).getDefiningOp<mlir_ts::VariableOp>() : mlir_ts::VariableOp();
        }

        if (record && !isOwningVariable(record) && record.getInitializer() &&
            record.getInitializer().getDefiningOp<mlir_ts::ConstantOp>())
        {
            op->emitError("an object literal that holds a value it owns is moved only into a local or "
                          "returned as it is; -mm=own cannot move it here yet");
            return true;
        }

        return false;
    }

    // The name of parameter `index`, as its slot carries it (under `--di`).
    llvm::StringRef paramName(int index)
    {
        llvm::StringRef name = "this value";
        getFunction()->walk([&](mlir_ts::VariableOp varOp) {
            auto argument = mlir::dyn_cast_or_null<mlir::BlockArgument>(varOp.getInitializer());
            if (argument && argument.getOwner()->isEntryBlock() && static_cast<int>(argument.getArgNumber()) == index)
            {
                name = varName(varOp);
            }
        });

        return name;
    }

    // Is `root`, or a view of it, stored into the result slot?
    static bool isReturned(mlir::Value root)
    {
        auto returned = false;
        forEachUse(root, [&](mlir::Operation *user, mlir::Value used) {
            auto storeOp = mlir::dyn_cast<mlir_ts::StoreOp>(user);
            auto slot = storeOp && storeOp.getValue() == used ? storeOp.getReference().getDefiningOp<mlir_ts::VariableOp>()
                                                              : mlir_ts::VariableOp();
            returned = returned || (slot && isResultSlot(slot));
        });

        return returned;
    }

    // A return that disagrees with one of a borrow of parameter `param` (with -1, with one that
    // borrows nothing): a store into the result slot of a value that holds a block and borrows
    // another parameter or none - or, for -1, borrows a parameter. Null when there is none.
    mlir::Operation *otherReturn(int param)
    {
        mlir::Operation *found = nullptr;
        getFunction()->walk([&](mlir_ts::StoreOp storeOp) {
            auto slot = storeOp.getReference().getDefiningOp<mlir_ts::VariableOp>();
            if (found || !slot || !isResultSlot(slot) || holdsNoBlock(storeOp.getValue()))
            {
                return;
            }

            auto borrows = borrowedParam(storeOp.getValue());
            if (param >= 0 ? borrows != param : borrows >= 0)
            {
                found = storeOp;
            }
        });

        return found;
    }

    // Where the body would have been fine had its callers known a fact about it, say why they
    // cannot.
    void noteLostFacts(mlir::InFlightDiagnostic &diag)
    {
        auto f = getFunction();
        if (auto why = f->getAttrOfType<mlir::StringAttr>(OWN_FACTS_LOST_ATTR_NAME))
        {
            diag.attachNote(f.getLoc()) << "'" << f.getName() << "' could return a borrow or keep a parameter, but "
                                        << why.getValue();
        }
    }

    void reportKeptOnSomePaths(mlir::Operation *move, mlir::Operation *exit, llvm::StringRef name)
    {
        if (!reporting())
        {
            return;
        }

        auto diag = move->emitError("'") << name
                                         << "' is moved here on some paths only; -mm=own cannot release it "
                                            "on the others yet";
        diag.attachNote(exit->getLoc()) << "returns here without moving it; the caller gave it up";
        signalPassFailure();
    }

    // A handle made into a union whose other members include one that owns a block (spec 23.2):
    // `Shared<T> | string` given a Shared<T>. gc, none and rc accept it.
    void reportHandleInOwningUnion(mlir::Operation *op)
    {
        auto handleIn = llvm::any_of(op->getOperandTypes(), [](mlir::Type type) {
            return MLIRTypeHelper::isSharedHandleType(type);
        });
        if (!handleIn)
        {
            return;
        }

        for (auto resultType : op->getResultTypes())
        {
            if (!MLIRTypeHelper::isSharedInOwningUnionType(resultType))
            {
                continue;
            }

            if (!reporting())
            {
                return;
            }

            auto unionType = resultType;
            if (auto optionalType = mlir::dyn_cast<mlir_ts::OptionalType>(unionType))
            {
                unionType = optionalType.getElementType();
            }

            mlir::Type shared;
            mlir::Type owning;
            for (auto member : mlir::cast<mlir_ts::UnionType>(unionType).getTypes())
            {
                if (mlir::isa<mlir_ts::SharedType>(member))
                {
                    shared = shared ? shared : member;
                }
                else if (!owning && MLIRTypeHelper::ownsHeapMemoryOfUnionMember(member))
                {
                    owning = member;
                }
            }

            auto diag = op->emitError("a ") << printed(shared) << " in a union with ";
            if (owning)
            {
                diag << "'" << printed(owning) << "', which -mm=own does not count,";
            }
            else
            {
                diag << "a member -mm=own does not count,";
            }

            diag << " is not supported";
            signalPassFailure();
            return;
        }
    }

    static std::string printed(mlir::Type type)
    {
        llvm::SmallString<128> text;
        llvm::raw_svector_ostream out(text);
        MLIRPrinter printer{};
        printer.printType<llvm::raw_svector_ostream>(out, type);
        return text.str().str();
    }

    void reportGivenNotOwned(mlir::Operation *call, mlir::Value arg)
    {
        if (!reporting())
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
        if (!reporting())
        {
            return;
        }

        auto diag = use->emitError("'") << name << "' is used here after its value was moved";
        diag.attachNote(move->getLoc()) << "value moved here";
        signalPassFailure();
    }

    void reportMovedInLoop(mlir::Operation *move, llvm::StringRef name)
    {
        if (!reporting())
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
        if (!reporting())
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
        if (!reporting())
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
        if (!reporting())
        {
            return;
        }

        op->emitError("'") << name << "' borrows " << place << " and cannot be assigned; -mm=own cannot prove that yet";
        signalPassFailure();
    }

    void reportPlaceBorrowEscapes(mlir::Operation *op, llvm::StringRef name, llvm::StringRef place)
    {
        if (!reporting())
        {
            return;
        }

        auto diag = op->emitError("'") << name << "' borrows " << place << " and cannot be stored, returned or captured";
        noteLostFacts(diag);
        signalPassFailure();
    }

    void reportMovedWhileBorrowed(mlir::Operation *op, llvm::StringRef owner, llvm::StringRef borrower)
    {
        if (!reporting())
        {
            return;
        }

        op->emitError("'") << owner << "' is moved here while '" << borrower
                           << "' borrows it; a borrowed value cannot be moved";
        signalPassFailure();
    }

    void reportBorrowEscapes(mlir::Operation *op, llvm::StringRef name, llvm::StringRef owner)
    {
        if (!reporting())
        {
            return;
        }

        auto diag = op->emitError("'") << name << "' borrows '" << owner << "' and cannot be stored, returned or captured";
        noteLostFacts(diag);
        signalPassFailure();
    }

    void reportBorrowerAssigned(mlir::Operation *op, llvm::StringRef name, llvm::StringRef owner)
    {
        if (!reporting())
        {
            return;
        }

        op->emitError("'") << name << "' borrows '" << owner << "' and cannot be assigned; -mm=own cannot prove that yet";
        signalPassFailure();
    }

    void reportEscapingCellNotOwned(mlir::Operation *capture, mlir::Operation *escape, llvm::StringRef why)
    {
        if (!reporting())
        {
            return;
        }

        auto diag = capture->emitError("a closure that escapes owns what it captures, but -mm=own cannot move this "
                                       "variable into it: ")
                    << why;
        diag.attachNote(escape->getLoc()) << "the closure escapes here";
        signalPassFailure();
    }

    void reportUsedAfterEscapingCapture(mlir::Operation *use, mlir::Operation *capture, llvm::StringRef name)
    {
        if (!reporting())
        {
            return;
        }

        auto diag = use->emitError("'") << name << "' is captured by a closure that escapes, and is still used here";
        diag.attachNote(capture->getLoc()) << "the closure takes it here";
        signalPassFailure();
    }

    void reportClosureOutlives(mlir::Operation *use, mlir::Operation *end, llvm::StringRef closure,
                               llvm::StringRef captured)
    {
        if (!reporting())
        {
            return;
        }

        auto diag = use->emitError("'") << closure << "' borrows '" << captured << "', which it captures, but runs here after '"
                                        << captured << "' is released";
        diag.attachNote(end->getLoc()) << "'" << captured << "' is released here";
        signalPassFailure();
    }

    void reportCaptureNotOwned(mlir::Operation *op, llvm::StringRef captured)
    {
        if (!reporting())
        {
            return;
        }

        op->emitError("'") << captured << "' is captured by a closure, but -mm=own cannot tell who owns its value here";
        signalPassFailure();
    }

    void reportCapturedParamAssigned(mlir::Operation *op, llvm::StringRef name)
    {
        if (!reporting())
        {
            return;
        }

        op->emitError("'") << name
                           << "' is a captured parameter and cannot be assigned under -mm=own: its value is the "
                              "caller's";
        signalPassFailure();
    }

    void reportMovedOnSomePaths(mlir::Operation *move, mlir::Operation *release, llvm::StringRef name)
    {
        if (!reporting())
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
