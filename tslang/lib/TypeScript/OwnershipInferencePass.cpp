#include "mlir/Pass/Pass.h"
#include "mlir/IR/Dominance.h"
#include "mlir/Dialect/LLVMIR/LLVMAttrs.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
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
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/raw_ostream.h"

#include <functional>
#include <optional>

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
// move. A local's release that paths which moved it and paths which did not both reach is placed
// instead on the edges where the latter join the former (spec 30). Anything else is an error at
// its location.
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

        resetFunctionState();

        // first: whether a closure borrows what it captures changes what its box's stores are
        classifyClosures();
        classifyStateBoxes();
        classifyBorrowingFields();
        // ...and whether an object literal borrows what its fields hold (spec 32)
        classifyRecords();
        // a kept parameter the body may not move on every path is released at its exits (spec 33)
        releaseKeptParamsAtExits();
        auto exitReleasesGone = llvm::make_scope_exit([&]() {
            // the dry run changes nothing: what it put in goes again
            if (dryRun)
            {
                for (auto *release : exitReleases)
                {
                    toErase.remove(release);
                    release->erase();
                }
            }

            exitReleases.clear();
        });

        FunctionOps ops;
        collectOwnershipOps(ops, toErase);
        auto proven = checkOwnershipOps(ops, toErase);

        Receivers slotReceivers;
        Receivers paramReceivers;
        decideRetains(ops, proven, slotReceivers, paramReceivers, toErase);
        giveKeptArguments(ops, slotReceivers, paramReceivers);

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
        checkBorrowingRecords();
        checkRecordResults();
        toErase.insert(fieldBorrowReleases.begin(), fieldBorrowReleases.end());

        if (dryRun)
        {
            return failures == 0;
        }

        // a handle's retains and releases are rc's, and stay (spec 23.2)
        toErase.remove_if([](mlir::Operation *op) { return touchesHandle(op); });
        placeEdgeReleases();
        for (auto *op : toErase)
        {
            op->erase();
        }

        stripFacts();
        return true;
    }

    // Each analysis starts from nothing this function has not set up again.
    void resetFunctionState()
    {
        auto f = getFunction();
        drops.clear();
        edgeReleases.clear();
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
        returnsRecordOf = isRecordResult(f) ? resultBorrowsAll(f) : llvm::SmallVector<int>();
        records.clear();
        recordStoreUses.clear();
    }

    // What the ownership operations of one function name, in the order the walk meets them
    // (collectOwnershipOps).
    struct FunctionOps
    {
        // every value some ownership operation names, in the order the walk meets them
        llvm::SetVector<mlir::Value> candidates;
        llvm::SmallVector<mlir::Operation *> retains;
        // reads out of fields and elements, and results that borrow an argument, and what may
        // destroy what they read (spec 2.3, 2.4)
        llvm::SmallVector<mlir::Operation *> placeReads;
        // reads of what a mortal global holds (isGlobalRead)
        llvm::SmallVector<mlir_ts::LoadOp> globalReads;
        // calls whose callee keeps an argument: each such argument moves into the call
        llvm::SmallVector<mlir::Operation *> keepingCalls;
    };

    struct SlotReceiver
    {
        mlir::Operation *retain;
        mlir::Value value;
        mlir_ts::LoadOp load;
    };

    // the receivers of each owning local, or of each parameter this function keeps
    using Receivers = llvm::MapVector<mlir::Value, llvm::SmallVector<SlotReceiver>>;

    // Every op of the function that names an ownership: the values to check, the retains to
    // decide, the reads out of places and globals, the calls that keep an argument, and the drops.
    void collectOwnershipOps(FunctionOps &ops, llvm::SetVector<mlir::Operation *> &toErase)
    {
        auto f = getFunction();
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
                ops.retains.push_back(op);
                ops.candidates.insert(rootOf(retainOp.getReference()));
            }
            else if (auto retainSlotOp = mlir::dyn_cast<mlir_ts::RetainSlotOp>(op))
            {
                ops.retains.push_back(op);
            }
            else if (auto releaseOp = mlir::dyn_cast<mlir_ts::ReleaseOp>(op))
            {
                ops.candidates.insert(rootOf(releaseOp.getReference()));
            }
            else if (auto varOp = mlir::dyn_cast<mlir_ts::VariableOp>(op))
            {
                if (varOp.getInitializer() && isOwningVariable(varOp) && !isHandle(varOp.getInitializer()))
                {
                    ops.candidates.insert(rootOf(varOp.getInitializer()));
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
                // a yield's read of a field that only yields (OWN_YIELDED_FIELDS_ATTR_NAME) takes
                // the value out: fresh, not a borrow of the state
                if (placeReadOf(loadOp.getResult()) && !loadOp->hasAttr(OWN_FRESH_RESULT_ATTR_NAME))
                {
                    ops.placeReads.push_back(loadOp);
                }
                else if (isGlobalRead(loadOp))
                {
                    ops.globalReads.push_back(loadOp);
                }
            }
            else if (auto releaseSlotOp = mlir::dyn_cast<mlir_ts::ReleaseSlotOp>(op); releaseSlotOp && isYieldedField(releaseSlotOp.getSlot()))
            {
                // what the field held was yielded, the caller's now
                toErase.insert(op);
                return;
            }
            else if (auto releaseSlotOp = mlir::dyn_cast<mlir_ts::ReleaseSlotOp>(op))
            {
                // an overwrite of a field or an element, or an assignment to a captured variable or
                // a global (spec 29: the global may hold what `this` or a parameter is)
                auto slot = releaseSlotOp.getSlot();
                if (isPlace(slot) || isCellVariable(slot) || isLoadedCell(slot) || slot.getDefiningOp<mlir_ts::AddressOfOp>())
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
                    ops.placeReads.push_back(op);
                }

                if (!ownedParams(op).empty())
                {
                    ops.keepingCalls.push_back(op);
                }
            }

            for (auto result : op->getResults())
            {
                if (!isView(op) && isFresh(result) && ownsHeap(result) && !isHandle(result))
                {
                    ops.candidates.insert(result);
                }
            }
        });
    }

    // The owner check of each value, then of each read out of a place and of a mortal global. The
    // fresh values proven, whose retains go.
    llvm::DenseSet<mlir::Value> checkOwnershipOps(const FunctionOps &ops, llvm::SetVector<mlir::Operation *> &toErase)
    {
        // Every value gets the owner check, fresh or not: rc can hand a value it did not make to
        // two consuming declarations without a single retain (the result of an indirect call
        // taken by `let b = a; let c = a;`). Only a fresh value's retains can be erased.
        llvm::DenseSet<mlir::Value> proven;
        for (auto value : ops.candidates)
        {
            if (checkValueMoves(value, toErase) && isFresh(value))
            {
                proven.insert(value);
            }
        }

        for (auto read : ops.placeReads)
        {
            checkPlaceRead(read, toErase);
        }

        for (auto read : ops.globalReads)
        {
            checkGlobalRead(read);
        }

        checkArgumentsAssigned();

        return proven;
    }

    // Each retain: erased, given to the receivers of an owning local or of a kept parameter, or
    // reported.
    void decideRetains(const FunctionOps &ops, const llvm::DenseSet<mlir::Value> &proven, Receivers &slotReceivers,
                       Receivers &paramReceivers, llvm::SetVector<mlir::Operation *> &toErase)
    {
        // What is proven loses its retains; any other retain is a second reference nobody
        // proved safe - of a loaded value, a parameter, an unknown producer, or of a fresh value
        // whose own check was reported above.
        // Receivers of each owning local are decided together: one may move it, but where there
        // are several, each borrows it, since a borrow pins its owner.
        for (auto *op : ops.retains)
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

            // an immortal global's value (spec 25): owned by no one, so returned, stored or kept
            // as often as it is read, and its releases skip it
            if (value && isImmortalGlobalRead(value) && (mlir::isa<mlir_ts::RetainOp>(op) || slotHoldsInitializerAt(op)))
            {
                toErase.insert(op);
                continue;
            }

            // an array a spread builds, read into the owning local it is made for, or returned
            if (value && mlir::isa<mlir_ts::RetainSlotOp>(op) && isBuiltArrayRead(op, value))
            {
                toErase.insert(op);
                continue;
            }

            if (value && mlir::isa<mlir_ts::RetainOp>(op) && isBuiltArrayReturned(op, value))
            {
                toErase.insert(op);
                continue;
            }

            if (!value || isFresh(value) == false)
            {
                reportSecondReference(op);
            }
        }
    }

    // The arguments a callee keeps, given to the receivers of the local or kept parameter they come
    // from.
    void giveKeptArguments(const FunctionOps &ops, Receivers &slotReceivers, Receivers &paramReceivers)
    {
        // An argument a callee keeps moves into the call (spec 2.4): out of an owning local or a
        // parameter this function keeps, like any other taker; a fresh value is decided with the
        // other takers of it, a read out of a container by its own check. Anything else is not
        // this function's to give.
        for (auto *call : ops.keepingCalls)
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
    }

    // The facts were for this pass only.
    void stripFacts()
    {
        returnedParams.clear();
        getFunction().walk([](mlir::Operation *op) {
            for (auto *name : {OWN_PARAMS_ATTR_NAME, OWN_PARAMS_RELEASED_ATTR_NAME, OWN_RESULT_BORROWS_ATTR_NAME,
                               OWN_RESULT_RECORD_ATTR_NAME,
                               OWN_NO_DROPS_ATTR_NAME, OWN_FACTS_LOST_ATTR_NAME, OWN_FRESH_RESULT_ATTR_NAME,
                               OWN_ASSIGNS_CAPTURES_ATTR_NAME,
                               OWN_RESULT_BOUNDED_ATTR_NAME,
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

    // The parameters this function's result is a borrowing record of (OWN_RESULT_RECORD_ATTR_NAME,
    // spec 32); empty for none.
    llvm::SmallVector<int> returnsRecordOf;

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

    // Is `ref` a field of a generator's state object that only yields (OWN_YIELDED_FIELDS_ATTR_NAME)?
    bool isYieldedField(mlir::Value ref)
    {
        auto propertyRefOp = ref.getDefiningOp<mlir_ts::PropertyRefOp>();
        if (!propertyRefOp)
        {
            return false;
        }

        auto recordType = propertyRefOp->getOperand(0).getType();
        if (auto objectType = mlir::dyn_cast<mlir_ts::ObjectType>(recordType))
        {
            recordType = objectType.getStorageType();
        }

        if (auto storageType = mlir::dyn_cast<mlir_ts::ObjectStorageType>(recordType))
        {
            recordType = mlir_ts::TupleType::get(&getContext(), storageType.getFields());
        }

        auto module = getFunction()->getParentOfType<mlir::ModuleOp>();
        auto entries = module ? module->getAttrOfType<mlir::ArrayAttr>(OWN_YIELDED_FIELDS_ATTR_NAME) : mlir::ArrayAttr();
        if (!entries)
        {
            return false;
        }

        for (auto entry : entries.getAsRange<mlir::ArrayAttr>())
        {
            if (mlir::cast<mlir::TypeAttr>(entry[0]).getValue() == recordType &&
                llvm::is_contained(mlir::cast<mlir::DenseI32ArrayAttr>(entry[1]).asArrayRef(),
                                   static_cast<int32_t>(propertyRefOp.getPosition())))
            {
                return true;
            }
        }

        return false;
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

                // returned by a function that says so (`@borrows`): its callers hold the result to
                // the same bound, and the signature pass checked the arguments are its own
                if (mlir::isa<mlir_ts::ReturnInternalOp>(user) && getFunction()->hasAttr(OWN_DECLARED_BORROWS_ATTR_NAME))
                {
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
                // a parameter this function does not own, in a box that does not borrow: the box
                // cannot take rc's reference for it, the caller's still holds it; left to the
                // second-reference error, which says the callers would have to give it up
                auto param = borrowedParam(fill.second.getValue());
                if (!closure->borrowsValues && param >= 0 && !llvm::is_contained(ownedParams(getFunction()), param))
                {
                    continue;
                }

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
                auto *retain = earlier ? soleUser(earlier.getResult()) : nullptr;
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
            auto elementType = mlir::cast<mlir_ts::RefType>(value.getType()).getElementType();
            // a handle's cell holds a count of its own, whoever's the value was (spec 23.2)
            if (!isOwningVariable(varOp) && !closure.borrowsValues && ownsHeap(varOp.getLoc(), elementType) &&
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

            auto elementType = mlir::cast<mlir_ts::RefType>(cell.getType()).getElementType();
            if (!ownsHeap(varOp.getLoc(), elementType))
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

    // An argument read out of a local is lent to the call, but the arguments are read in order: a
    // later one may give the local a new value before the call takes the first - `g(n, (n = mk(), 0))`,
    // in a conditional argument too - or call a closure that assigns it, when it is captured. The
    // assignment drops the block the argument is, and the callee reads freed memory. rc holds such an
    // argument across the call instead (OwnedReturnConsumptionPass, holdGlobalArguments). A read of a
    // global is checkGlobalRead's, and a store into a place in the local the place borrows' (a field
    // read `g(n.v, (n.v = mk(), 0))`); a store into the local's slot ends `n.v` too.
    void checkArgumentsAssigned()
    {
        getFunction().walk([&](mlir::Operation *call) {
            if (!isCall(call))
            {
                return mlir::WalkResult::advance();
            }

            for (auto argument : callArgs(call))
            {
                // a method's object: `this` of the method bound to it
                if (auto getThisOp = argument.getDefiningOp<mlir_ts::GetThisOp>())
                {
                    auto *boundOp = getThisOp->getOperand(0).getDefiningOp();
                    if (mlir::isa_and_nonnull<mlir_ts::ThisSymbolRefOp, mlir_ts::ThisVirtualSymbolRefOp>(boundOp))
                    {
                        argument = boundOp->getOperand(0);
                    }
                }

                auto *read = argument.getDefiningOp();
                auto local = mlir::dyn_cast_or_null<mlir_ts::VariableOp>(slotOf(argument));
                if (!read || !local || !ownsHeap(argument) || isHandle(argument))
                {
                    continue;
                }

                auto assignedByClosure = closureAssigns(local);
                auto *write = findBetween(read, call, [&](mlir::Operation *op) {
                    if (assignedByClosure && isCall(op))
                    {
                        return true;
                    }

                    auto storeOp = mlir::dyn_cast<mlir_ts::StoreOp>(op);
                    auto releaseSlotOp = mlir::dyn_cast<mlir_ts::ReleaseSlotOp>(op);
                    return (storeOp && storeOp.getReference() == local.getResult()) ||
                           (releaseSlotOp && releaseSlotOp.getSlot() == local.getResult());
                });
                if (!write)
                {
                    continue;
                }

                if (reporting())
                {
                    auto name = varName(local);
                    auto callee = calleeName(call);
                    auto diag = call->emitError("'")
                                << name << "' is lent to '" << (callee.empty() ? "a function" : callee)
                                << "', but is given a new value before the call takes it; -mm=own cannot prove "
                                   "it outlives the call";
                    diag.attachNote(write->getLoc()) << "'" << name << "' is given a new value here";
                    signalPassFailure();
                }

                return mlir::WalkResult::interrupt();
            }

            return mlir::WalkResult::advance();
        });
    }

    // Does a closure built over this captured local assign it (OWN_ASSIGNS_CAPTURES_ATTR_NAME)? Yes too
    // when its cell goes where this does not follow.
    bool closureAssigns(mlir_ts::VariableOp local)
    {
        if (!local.getCaptured().value_or(false))
        {
            return false;
        }

        for (auto *user : local->getUsers())
        {
            auto storeOp = mlir::dyn_cast<mlir_ts::StoreOp>(user);
            if (mlir::isa<mlir_ts::LoadOp, mlir_ts::ReleaseSlotOp, mlir_ts::RetainCellOp, mlir_ts::ReleaseCellOp,
                          mlir_ts::DebugVariableOp>(user) ||
                (storeOp && storeOp.getReference() == local.getResult()))
            {
                continue;
            }

            // the cell put into a closure's capture box
            auto propertyRefOp = storeOp ? storeOp.getReference().getDefiningOp<mlir_ts::PropertyRefOp>()
                                         : mlir_ts::PropertyRefOp();
            auto boundOp = propertyRefOp ? closureOfBox(propertyRefOp.getObjectRef()) : mlir_ts::CreateBoundFunctionOp();
            if (!boundOp)
            {
                return true;
            }

            auto assigned = boundOp->getAttrOfType<mlir::DenseI32ArrayAttr>(OWN_ASSIGNS_CAPTURES_ATTR_NAME);
            if (assigned && llvm::is_contained(assigned.asArrayRef(), static_cast<int32_t>(propertyRefOp.getPosition())))
            {
                return true;
            }
        }

        return false;
    }

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
        if (receivers.size() == 1 || takeValuesOfTheirOwn(slot, receivers))
        {
            // a move on some paths only is placed when it is the local's one receiver (spec 30)
            for (auto &receiver : receivers)
            {
                decideSlotReceiver(slot, receiver, owner, toErase, receivers.size() == 1);
            }

            return;
        }

        decideSlotReceivers(slot, receivers, owner, toErase);
    }

    void decideSlotReceiver(mlir::Value slot, const SlotReceiver &receiver, llvm::StringRef owner,
                            llvm::SetVector<mlir::Operation *> &toErase, bool sole)
    {
        if (quietly([&] { return checkSlotMove(receiver.retain, receiver.value, receiver.load, toErase, sole); }))
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

        checkSlotMove(receiver.retain, receiver.value, receiver.load, toErase, sole); // reports
    }

    // Receivers of one owning local that each read a value of its own (#528): `h.c = a; a = new C();
    // g.c = a`. An assignment comes between every two of their reads on every path from one to the
    // other, so each is decided alone, as the one receiver of the value it reads. So are receivers
    // no path joins at all, assigned or not: `if (c) return a; return a;` moves `a` once on each
    // way out.
    bool takeValuesOfTheirOwn(mlir::Value slot, llvm::ArrayRef<SlotReceiver> receivers)
    {
        auto assignments = slotAssignments(slot);
        llvm::SmallVector<mlir::Operation *> kills(assignments.begin(), assignments.end());
        if (auto varOp = slot.getDefiningOp<mlir_ts::VariableOp>())
        {
            kills.push_back(varOp);
        }

        for (auto &from : receivers)
        {
            for (auto &to : receivers)
            {
                if (&from != &to && reachableAfter(from.load, to.load, kills))
                {
                    return false;
                }
            }
        }

        return true;
    }

    // ---- A kept parameter released where it is not moved (spec 33) ----
    //
    // `return v.x > 2 ? acc : V.plus(acc, v)`: `acc` is kept (OwnershipSignaturePass), so the caller
    // moved it in; the path that returns it moves it out, and the others must release it. So must a
    // callback that only borrows the accumulator its family gives it to keep. MLIRGen made no owner
    // of a parameter, so its releases are put in here, before each way out of the body - a return,
    // a throw that leaves the function - as MLIRGen puts an owning local's. The move's own check then
    // erases those it reaches, and places the release on the edges into the join that did not move
    // it (spec 30), statically: no run-time flag. Unwinding out of the function (a call that throws
    // outside a `try`, a `finally` passed on the way out) gives nothing back: it leaks, as an owning
    // local does there.

    // the releases put in for this run
    llvm::SmallVector<mlir::Operation *> exitReleases;

    // The slot of parameter `index` (isParameterSlot), or null.
    mlir_ts::VariableOp parameterSlot(int32_t index)
    {
        auto &body = getFunction().getFunctionBody();
        if (body.empty() || static_cast<unsigned>(index) >= body.front().getNumArguments())
        {
            return {};
        }

        for (auto *user : body.front().getArgument(index).getUsers())
        {
            auto varOp = mlir::dyn_cast<mlir_ts::VariableOp>(user);
            auto position = -1;
            if (varOp && isParameterSlot(varOp, position) && position == index)
            {
                return varOp;
            }
        }

        return {};
    }

    // Is `slot` a parameter this function keeps and releases where it does not move it?
    bool isReleasedParam(mlir::Value slot)
    {
        auto f = getFunction();
        auto attr = f->getAttrOfType<mlir::DenseI32ArrayAttr>(OWN_PARAMS_RELEASED_ATTR_NAME);
        auto varOp = slot.getDefiningOp<mlir_ts::VariableOp>();
        auto index = -1;
        return attr && varOp && isParameterSlot(varOp, index) && llvm::is_contained(attr.asArrayRef(), index) &&
               llvm::is_contained(ownedParams(f), index);
    }

    // Does `value` come from a read of `slot`: computed from one, through any op?
    static bool readsSlot(mlir::Value value, mlir::Value slot, int depth = 0)
    {
        auto *def = value.getDefiningOp();
        if (!def)
        {
            return false;
        }

        // too deep to tell: as if it did
        if (depth > 16)
        {
            return true;
        }

        if (auto loadOp = mlir::dyn_cast<mlir_ts::LoadOp>(def); loadOp && loadOp.getReference() == slot)
        {
            return true;
        }

        return llvm::any_of(def->getOperands(), [&](mlir::Value operand) { return readsSlot(operand, slot, depth + 1); });
    }

    void releaseKeptParamsAtExits()
    {
        exitReleases.clear();
        auto f = getFunction();
        auto attr = f->getAttrOfType<mlir::DenseI32ArrayAttr>(OWN_PARAMS_RELEASED_ATTR_NAME);
        if (!attr)
        {
            return;
        }

        for (auto index : attr.asArrayRef())
        {
            auto varOp = parameterSlot(index);
            if (!varOp || !llvm::is_contained(ownedParams(f), index))
            {
                continue;
            }

            for (auto &block : f.getFunctionBody())
            {
                auto *exit = block.empty() ? nullptr : &block.back();
                // a thrown value made of the parameter takes it with it: no release there (a leak,
                // if it does not)
                auto leaves = mlir::isa_and_nonnull<mlir_ts::ReturnInternalOp>(exit) ||
                              (mlir::isa_and_nonnull<mlir_ts::ThrowCallOp>(exit) &&
                               !readsSlot(exit->getOperand(0), varOp.getResult()));
                if (!leaves)
                {
                    continue;
                }

                mlir::OpBuilder builder(exit);
                exitReleases.push_back(builder.create<mlir_ts::ReleaseSlotOp>(exit->getLoc(), varOp.getResult()));
            }
        }
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

        // one the body releases on the paths that do not move it (spec 33): the move's own check
        // erases the releases at the exits it reaches, and places one on each edge into where the
        // paths join that did not move it (spec 30)
        auto released = isReleasedParam(slot);
        if (!checkSlotMove(receiver.retain, receiver.value, receiver.load, toErase, released))
        {
            return;
        }

        if (released)
        {
            if (!isCall(receiver.retain))
            {
                toErase.insert(receiver.retain);
            }

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

        // rc's retain goes with the move - a `ts.RetainSlot` too, for an owning local the parameter
        // moves into (`let r = init`, spec 33); a call that keeps it is the move itself
        if (everyPath && !isCall(receiver.retain))
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
    // the block's entry, the local's declaration (when it has an initializer: `let l!: V` stores
    // nothing, and in a loop the slot keeps the last round's value) and assignments.
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
                if (renewedAt && varOp.getInitializer())
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
        Keeping keeping = Keeping{nullptr, false}, bool countedHandle = false, bool startBorrowed = true)
    {
        struct Pending
        {
            mlir::Value value;
            bool borrowed;
            llvm::SmallVector<mlir::Operation *, 2> kills;
        };

        llvm::SmallVector<Pending> values;
        // `startBorrowed` false: `start` is what a read produced (a field's reference), whose uses
        // are uses of the borrow but not the borrowed value itself
        values.push_back({start, startBorrowed, {kills.begin(), kills.end()}});
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

                // stored into a field a borrowing record borrows (spec 32): whatever uses the record
                // reads it
                if (auto found = recordStoreUses.find(user); found != recordStoreUses.end())
                {
                    for (auto &recordUse : found->second->walk.uses)
                    {
                        uses.push_back({recordUse.op, pending.kills, recordUse.callee});
                    }
                }

                if (pending.borrowed && keeping.returns && mlir::isa<mlir_ts::ReturnInternalOp>(user))
                {
                    continue;
                }

                if (pending.borrowed)
                {
                    // a handle read that rc counts (spec 23.3): a return or an `any` box rc
                    // retained for is a counted copy of its own; passesOn would hand them on
                    if (countedHandle && isCountedHandOff(user, pending.value))
                    {
                        continue;
                    }

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

                    // a handle read that rc counts (spec 23.3): its retain, and what takes the value
                    // it retains for, are a counted copy of their own, as a string copy is; the
                    // read's other uses are the borrow's (Ruling 7)
                    if (countedHandle && isCountedHandleReceiver(user, pending.value))
                    {
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

                // a copy (spec 22.2) reads the borrow into a block of its own, which points into
                // nothing, and so does a concatenation: `s + t` is a new string whatever s and t are
                // (a `+` of two strings lowers to ts.StringConcat, which always allocates)
                if (mlir::isa<mlir_ts::StringCopyOp, mlir_ts::StringConcatOp>(user) ||
                    (mlir::isa<mlir_ts::ArithmeticBinaryOp>(user) && mlir::isa<mlir_ts::StringType>(user->getResult(0).getType())))
                {
                    continue;
                }

                // a call whose result is the caller's own (callResultIsOwn) ends the walk at the call
                if (callResultIsOwn(user))
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

    // Whether a call hands back a block of the caller's own, which points into nothing it was given:
    // walkBorrowed then ends a borrow's walk at the call (the call itself is still a use for as long
    // as it runs). Never a generator's state that borrows its arguments (`__own_result_bounded`).
    // - A call known to make its result (`__own_fresh_result`: an iterator's `next()`, a field that
    //   only ever holds one function, another own module's `..new`) - the `{value, done}` record of
    //   `it.next()` points nowhere into `it`, and what it does after its next call cannot be a use of
    //   the method read out of `it`.
    // - A direct call of a function this module defines (isFresh): its result points into nothing it
    //   was given but what its facts name - a borrow of an argument (`__own_result_borrows`, never
    //   fresh), a borrowing record (isFresh, but bounded by what it borrows), a generator's state
    //   that borrows its arguments - and a body whose callers cannot all know such a fact is an
    //   error. `getPoint(x, scene.camera)` makes a new vector: a call after it that may destroy
    //   `scene.camera` does not end that vector (spec 34).
    static bool callResultIsOwn(mlir::Operation *call)
    {
        if (!isCall(call) || call->hasAttr(OWN_RESULT_BOUNDED_ATTR_NAME))
        {
            return false;
        }

        return call->hasAttr(OWN_FRESH_RESULT_ATTR_NAME) ||
               (call->getNumResults() == 1 && !isRecordResult(call) && isFresh(call->getResult(0)));
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

    // ---- Borrowing records (spec 32) ----
    //
    // An object literal boxed as an interface that holds values it does not own - `this`, a parameter,
    // a field read, an owning `let`'s value, a captured variable's - beside fresh values, which move
    // in. The signature pass marked its box (OWN_RECORD_BORROWED_FIELDS_ATTR_NAME): its release frees
    // the box and the fields it owns, and leaves the others to their owners. So the box has one owner
    // like any fresh block - the box's own check - and what it borrows bounds every use of it: a read
    // out of a place is checked with the record's uses as its own (walkBorrowed), a local's value
    // against the local's ends, a captured variable's against whatever may assign it. Returned, the
    // record is the caller's to own and to bound (checkRecordResults), which the function's facts
    // must say. Stored anywhere else that may outlive what it borrows, it is an error.
    struct RecordWalk
    {
        llvm::SmallVector<BorrowUse> uses;
        // the first use that would keep the record itself: a store into a place, a callee that keeps
        // it, a capture
        mlir::Operation *kept = nullptr;
        llvm::SmallVector<mlir::Operation *> returns;
    };

    struct BorrowedField
    {
        mlir_ts::StoreOp store;
        RecordFieldSource source;
    };

    struct BorrowingRecord
    {
        mlir_ts::NewOp box;
        mlir_ts::VariableOp local;
        llvm::SmallVector<mlir::Value> interfaces;
        llvm::SmallVector<BorrowedField> borrowed;
        RecordWalk walk;
    };

    llvm::SmallVector<std::shared_ptr<BorrowingRecord>> records;
    // a store of a place read into a borrowing record's field -> the record's uses, which are the
    // read's uses too (walkBorrowed)
    llvm::DenseMap<mlir::Operation *, BorrowingRecord *> recordStoreUses;

    // Can a value of this type point into a heap block - a reference into one, a block of its own,
    // or a method bound to one? A number or a boolean read out of a borrowed block cannot.
    bool mayPointInto(mlir::Value value)
    {
        auto type = value.getType();
        return mlir::isa<mlir_ts::RefType, mlir_ts::BoundRefType, mlir_ts::ValueRefType, mlir_ts::BoundFunctionType,
                         mlir_ts::HybridFunctionType, mlir_ts::ExtensionFunctionType, mlir_ts::OpaqueType>(type) ||
               ownsHeap(value);
    }

    // The borrowing record built in `varOp` (classifyRecords), or null.
    BorrowingRecord *borrowingRecordOf(mlir_ts::VariableOp varOp)
    {
        for (auto &record : records)
        {
            if (record->local == varOp)
            {
                return record.get();
            }
        }

        return nullptr;
    }

    // The boxes the signature pass marked as borrowing records, where every field still holds what it
    // was marked for: a borrowed field a value this pass bounds (recordFieldSource), any other one a
    // fresh value or nothing. A store into a borrowed field takes nothing, and whatever uses the record
    // reads what it stored (walkBorrowed, through recordStoreUses). A box that is not what it was
    // marked for is left to the record's retain, which is then an error.
    void classifyRecords()
    {
        getFunction().walk([&](mlir_ts::NewOp newOp) {
            auto fields = recordBorrowedFields(newOp);
            auto varOp = fields.empty() ? mlir_ts::VariableOp() : recordLocalOf(newOp.getResult());
            if (!varOp)
            {
                return;
            }

            auto record = std::make_shared<BorrowingRecord>();
            record->box = newOp;
            record->local = varOp;
            for (auto *user : varOp->getUsers())
            {
                auto propertyRefOp = mlir::dyn_cast<mlir_ts::PropertyRefOp>(user);
                if (!propertyRefOp)
                {
                    continue;
                }

                auto borrows = llvm::is_contained(fields, static_cast<int32_t>(propertyRefOp.getPosition()));
                for (auto *fieldUser : propertyRefOp->getUsers())
                {
                    auto storeOp = mlir::dyn_cast<mlir_ts::StoreOp>(fieldUser);
                    if (!storeOp || storeOp.getReference() != propertyRefOp.getResult())
                    {
                        continue;
                    }

                    auto source = recordFieldSource(storeOp.getValue());
                    if (source == RecordFieldSource::NoBlock)
                    {
                        continue;
                    }

                    if (source == RecordFieldSource::Unknown || borrows != (source != RecordFieldSource::Fresh))
                    {
                        return;
                    }

                    if (borrows)
                    {
                        record->borrowed.push_back({storeOp, source});
                    }
                }
            }

            for (auto *user : newOp->getUsers())
            {
                if (auto castOp = mlir::dyn_cast<mlir_ts::CastOp>(user))
                {
                    for (auto *castUser : castOp->getUsers())
                    {
                        if (auto newInterfaceOp = mlir::dyn_cast<mlir_ts::NewInterfaceOp>(castUser))
                        {
                            record->interfaces.push_back(newInterfaceOp.getResult());
                        }
                    }
                }
            }

            // a path back to a use through the record's making is another record
            llvm::SmallVector<mlir::Operation *, 2> kills{newOp.getOperation(), varOp.getOperation()};
            walkRecord(record->interfaces, kills, record->walk);
            for (auto &field : record->borrowed)
            {
                setAttrTracked(field.store, OWN_CAPTURE_BORROW_ATTR_NAME);
                recordStoreUses[field.store.getOperation()] = record.get();
            }

            records.push_back(std::move(record));
        });
    }

    // Every use of a borrowing record (spec 32), and of what is read out of it - each may read what
    // the record borrows - through its views, merges, and the locals that hold it, made again at
    // `kills`. Its own retains and releases are the box's, which its own check decides. The record
    // itself is `kept` when it is stored anywhere but a local, given to a callee that keeps it, or
    // captured; returned, it is in `returns`. A call given it hands back its own block, unless that
    // is a borrowing record that holds it, whose uses are then this one's as well.
    void walkRecord(llvm::ArrayRef<mlir::Value> starts, llvm::ArrayRef<mlir::Operation *> kills, RecordWalk &walk)
    {
        struct Pending
        {
            mlir::Value value;
            bool record;
        };

        llvm::SmallVector<Pending> work;
        llvm::SmallPtrSet<mlir::Value, 16> seen;
        auto push = [&](mlir::Value value, bool record) {
            if ((record || mayPointInto(value)) && seen.insert(value).second)
            {
                work.push_back({value, record});
            }
        };

        auto followSlot = [&](mlir_ts::VariableOp varOp, bool record) {
            for (auto *slotUser : varOp->getUsers())
            {
                // its releases and its other values are the slot's own
                if (auto loadOp = mlir::dyn_cast<mlir_ts::LoadOp>(slotUser))
                {
                    push(loadOp.getResult(), record);
                }
            }
        };

        auto keep = [&](mlir::Operation *user) { walk.kept = walk.kept ? walk.kept : user; };

        for (auto start : starts)
        {
            push(start, true);
        }

        while (!work.empty())
        {
            auto pending = work.pop_back_val();
            auto value = pending.value;
            for (auto &use : value.getUses())
            {
                auto *user = use.getOwner();
                if (mlir::isa<mlir_ts::DebugVariableOp>(user) ||
                    (pending.record && mlir::isa<mlir_ts::RetainOp, mlir_ts::ReleaseOp>(user)))
                {
                    continue;
                }

                // the same value: a view, a cast, a merge
                if ((isView(user) && use.getOperandNumber() == 0) || mlir::isa<mlir_ts::CastOp>(user))
                {
                    push(user->getResult(0), pending.record);
                    continue;
                }

                if (auto branchOp = mlir::dyn_cast<mlir::BranchOpInterface>(user))
                {
                    if (auto argument = branchOp.getSuccessorBlockArgument(use.getOperandNumber()))
                    {
                        push(*argument, pending.record);
                    }

                    continue;
                }

                // a local that holds it: its reads
                auto varOp = mlir::dyn_cast<mlir_ts::VariableOp>(user);
                auto storeOp = mlir::dyn_cast<mlir_ts::StoreOp>(user);
                if (varOp || (storeOp && storeOp.getValue() == value))
                {
                    auto slotOp = varOp ? varOp : storeOp.getReference().getDefiningOp<mlir_ts::VariableOp>();
                    if (slotOp && !slotOp.getCaptured().value_or(false))
                    {
                        followSlot(slotOp, pending.record);
                        continue;
                    }

                    // a place, a global, a captured variable: what is read out of the record is checked
                    // where it is read (checkPlaceRead)
                    walk.uses.push_back({user, {kills.begin(), kills.end()}});
                    if (pending.record)
                    {
                        keep(user);
                    }

                    continue;
                }

                walk.uses.push_back({user, {kills.begin(), kills.end()}, isCall(user) && isCalleeOperand(use)});
                if (mlir::isa<mlir_ts::ReturnInternalOp, mlir_ts::ReturnValOp>(user))
                {
                    if (pending.record)
                    {
                        walk.returns.push_back(user);
                    }

                    continue;
                }

                if (isCall(user))
                {
                    if (pending.record && isKeptArgument(user, value))
                    {
                        keep(user);
                        continue;
                    }

                    // a result that borrows it - a plain borrow (`id(r)`), or a borrowing record that
                    // holds it - points where it points
                    auto args = callArgs(user);
                    auto given = [&](llvm::ArrayRef<int> indices) {
                        return llvm::any_of(indices, [&](int index) {
                            return static_cast<size_t>(index) < args.size() && args[index] == value;
                        });
                    };

                    if (user->getNumResults() == 1 && given(resultBorrowsAll(user)))
                    {
                        push(user->getResult(0), pending.record || isRecordResult(user));
                    }

                    // ...and a generator's state that borrows it (OWN_RESULT_BOUNDED_ATTR_NAME)
                    if (auto bounded = user->getAttrOfType<mlir::DenseI32ArrayAttr>(OWN_RESULT_BOUNDED_ATTR_NAME);
                        bounded && user->getNumResults() == 1 &&
                        given(llvm::SmallVector<int>(bounded.asArrayRef().begin(), bounded.asArrayRef().end())))
                    {
                        push(user->getResult(0), false);
                    }

                    continue;
                }

                if (pending.record && !isBorrow(user, value))
                {
                    keep(user);
                    continue;
                }

                // a field written through the record: its old value may be one the record borrows,
                // which the overwrite would release, and the field may be one its release leaves
                if (pending.record && mlir::isa<mlir_ts::InterfaceSymbolRefOp, mlir_ts::PropertyRefOp>(user))
                {
                    for (auto *fieldUser : user->getResult(0).getUsers())
                    {
                        auto fieldStore = mlir::dyn_cast<mlir_ts::StoreOp>(fieldUser);
                        if (mlir::isa<mlir_ts::ReleaseSlotOp>(fieldUser) ||
                            (fieldStore && fieldStore.getReference() == user->getResult(0)))
                        {
                            keep(fieldUser);
                        }
                    }
                }

                // what it produces - a field's reference, a value read out of it, a bound method - may
                // still point into what the record borrows
                for (auto result : user->getResults())
                {
                    push(result, false);
                }
            }
        }
    }

    // The name the record goes by: the local that holds it, or a folded `const`'s.
    llvm::StringRef recordName(mlir::Value value)
    {
        llvm::StringRef name = ownerName(value);
        forEachUse(value, [&](mlir::Operation *user, mlir::Value used) {
            if (auto varOp = mlir::dyn_cast<mlir_ts::VariableOp>(user); varOp && varOp.getInitializer() == used)
            {
                name = varName(varOp);
            }
        });

        return name;
    }

    // What a borrowed field holds, for an error: `'ray'`, `'this.pos'`, `'livec'`, ...
    std::string describeRecordSource(BorrowedField field)
    {
        auto value = field.store.getValue();
        auto quoted = [](llvm::StringRef name) { return "'" + name.str() + "'"; };
        switch (field.source)
        {
        case RecordFieldSource::Param:
            return quoted(paramName(borrowedParam(value)));
        case RecordFieldSource::Place:
            if (auto loadOp = rootOf(value).getDefiningOp<mlir_ts::LoadOp>(); loadOp && isPlace(loadOp.getReference()))
            {
                return describePlace(loadOp.getReference());
            }

            return describeCall(rootOf(value).getDefiningOp());
        case RecordFieldSource::Slot:
            return quoted(varName(slotLoadOf(value).getReference().getDefiningOp<mlir_ts::VariableOp>()));
        case RecordFieldSource::Cell:
            return "a captured variable";
        default:
            return "a value it does not own";
        }
    }

    // A record this function makes (spec 32): kept, it is an error; returned, each field it borrows
    // must be a parameter's that this function's facts lend (`__own_result_record`), so that its
    // callers bound it; and no use of it may come after what a field borrows may be released or
    // overwritten - an owning local's ends, and whatever may assign a captured variable. A parameter
    // and `this` are the caller's for the whole call; a read out of a place has the record's uses as
    // its own (checkPlaceRead).
    void checkBorrowingRecords()
    {
        for (auto &record : records)
        {
            auto name = recordName(record->interfaces.empty() ? record->box.getResult() : record->interfaces.front());
            if (record->walk.kept)
            {
                reportPlaceBorrowEscapes(record->walk.kept, name,
                                         record->borrowed.empty() ? "a value it does not own"
                                                                  : describeRecordSource(record->borrowed.front()));
                continue;
            }

            auto ok = true;
            for (auto *returned : record->walk.returns)
            {
                for (auto &field : record->borrowed)
                {
                    if (!ok || field.source == RecordFieldSource::Immortal)
                    {
                        continue;
                    }

                    auto param = field.source == RecordFieldSource::Param || field.source == RecordFieldSource::Place
                                     ? borrowedParam(field.store.getValue())
                                     : -1;
                    if (param < 0 || !llvm::is_contained(returnsRecordOf, param))
                    {
                        reportPlaceBorrowEscapes(returned, name, describeRecordSource(field));
                        ok = false;
                    }
                }
            }

            for (auto &field : record->borrowed)
            {
                if (!ok)
                {
                    break;
                }

                auto value = field.store.getValue();
                Chain chain;
                if (field.source == RecordFieldSource::Slot)
                {
                    chain.roots.push_back({slotLoadOf(value).getReference(), Chain::Slot});
                }
                else if (auto load = field.source == RecordFieldSource::Param ? keptParamLoadOf(value) : mlir_ts::LoadOp())
                {
                    // a parameter this function keeps is its own to move away, as a local's value is
                    chain.roots.push_back({load.getReference(), Chain::Slot});
                }
                else if (field.source == RecordFieldSource::Cell)
                {
                    chain.roots.push_back({rootOf(value), Chain::NotOwned});
                }
                else
                {
                    continue;
                }

                auto ends = rootEnds(chain);
                if (field.source == RecordFieldSource::Cell)
                {
                    for (auto *drop : drops)
                    {
                        if (dropsChain(drop, chain, nullptr, value))
                        {
                            ends.push_back(drop);
                        }
                    }
                }

                for (auto *end : ends)
                {
                    for (auto &use : record->walk.uses)
                    {
                        if (ok && (use.op != end || !use.callee) && reachableAfter(end, use.op, use.kills))
                        {
                            if (field.source == RecordFieldSource::Slot)
                            {
                                reportBorrowOutlives(use.op, end, name, varName(chain.roots.front().first.getDefiningOp<mlir_ts::VariableOp>()));
                            }
                            else
                            {
                                reportPlaceBorrowOutlives(use.op, end, name, describeRecordSource(field));
                            }

                            ok = false;
                        }
                    }
                }
            }
        }
    }

    // A call whose result is a borrowing record (OWN_RESULT_RECORD_ATTR_NAME, spec 32): the box is
    // this function's to own - it is fresh, and its own check decides its moves and releases - but
    // it may hold a borrow of each argument the facts name, as a call's result that borrows them
    // does (checkPlaceRead): somewhere under each. So no use of it, through the locals it moves
    // into, may come after one of those arguments may be released or overwritten, and it may not be
    // kept. Returned, the arguments must be parameters this function's own facts lend. Its releases
    // are not uses: they free only what the record owns.
    void checkRecordResults()
    {
        getFunction().walk([&](mlir::Operation *call) {
            if (!isCall(call) || !isRecordResult(call) || call->getNumResults() != 1)
            {
                return;
            }

            auto result = call->getResult(0);
            auto place = describeCall(call);
            auto name = recordName(result);
            RecordWalk walk;
            llvm::SmallVector<mlir::Operation *, 1> kills{call};
            walkRecord({result}, kills, walk);
            if (walk.kept)
            {
                reportPlaceBorrowEscapes(walk.kept, name, place);
                return;
            }

            auto args = callArgs(call);
            for (auto *returned : walk.returns)
            {
                for (auto index : resultBorrowsAll(call))
                {
                    if (static_cast<size_t>(index) >= args.size() || argumentHoldsNoBlock(args[index]) ||
                        isImmortalLiteral(args[index]))
                    {
                        continue;
                    }

                    auto param = borrowedParam(args[index]);
                    if (param < 0 || !llvm::is_contained(returnsRecordOf, param))
                    {
                        reportPlaceBorrowEscapes(returned, name, place);
                        return;
                    }
                }
            }

            auto chain = chainOf(call);
            // rc's retain of a fresh argument right before the call is no end: the box's own check
            // decides it, and the use that takes it is an end of its own
            llvm::SmallVector<mlir::Operation *> ends;
            for (auto *end : rootEnds(chain))
            {
                auto ownedRetain = mlir::isa<mlir_ts::RetainOp>(end) && llvm::any_of(chain.roots, [&](auto &root) {
                                       return root.second == Chain::Owned && rootOf(end->getOperand(0)) == root.first;
                                   });
                if (!ownedRetain)
                {
                    ends.push_back(end);
                }
            }

            for (auto *drop : drops)
            {
                if (drop != call && dropsChain(drop, chain, call))
                {
                    ends.push_back(drop);
                }
            }

            for (auto *end : ends)
            {
                for (auto &use : walk.uses)
                {
                    if ((use.op != end || !use.callee) && reachableAfter(end, use.op, use.kills))
                    {
                        reportPlaceBorrowOutlives(use.op, end, name, place);
                        return;
                    }
                }
            }
        });
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

    // How a handle read out of a place is used (spec 23.3), seen through casts, merges and locals
    // that own nothing. Every use is classified, and nothing falls through (Ruling 9):
    // - `counted`: a receiver that takes a count rc made for that very use
    //   (isCountedHandleReceiver) - a store, a push or an unshift, `new Shared`, an array
    //   literal's element, a return, an `any` box, or an owning local's initializer;
    // - passed on: a cast or a view that is still a handle, a merge, a local that owns nothing -
    //   whose own uses are classified in turn;
    // - `borrowed`: everything else. A use that reads through the place (`f(p, p.child)`, a
    //   `.value` read, `Shared.count`), and every use that is neither of the above: a tuple's
    //   element, an object literal's field, a capture, a `yield`, a receiver rc made no retain
    //   for, any op not named here. The read is then a borrow, and tenure-checked
    //   (checkPlaceRead).
    // rc's retains count for nothing on their own: each is the count of the receiver right after
    // it. `cur = cur.value.next` is counted only.
    struct HandleReadUses
    {
        bool counted;
        bool borrowed;
    };

    static HandleReadUses handleReadUses(mlir::Value value)
    {
        HandleReadUses result{false, false};
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
                    continue;
                }

                // a return and an `any` box take the value, though passesOn would hand them on
                if (isHandOff(user, current))
                {
                    (isCountedHandleReceiver(user, current) ? result.counted : result.borrowed) = true;
                    continue;
                }

                llvm::SmallVector<mlir::Value> same;
                if (passesOn(use, same))
                {
                    for (auto next : same)
                    {
                        // what is no longer a handle is no receiver: a test (`if (p.child)`) only
                        // reads it
                        if (!isHandle(next))
                        {
                            result.borrowed = true;
                        }
                        else if (seen.insert(next).second)
                        {
                            values.push_back(next);
                        }
                    }

                    continue;
                }

                (isCountedHandleReceiver(user, current) ? result.counted : result.borrowed) = true;
            }
        }

        return result;
    }

    // A handle read out of a place that rc gives a count of its own is a counted copy, not a borrow
    // (spec 23.3) - but only when every use of it is one rc counts (Ruling 7), and a use that is
    // not known to be one is not (Ruling 9). A read with one use that reads through the place as
    // well (`f(p, q, q.child = p.child)`), or that keeps it with no count (`[(q.child = p.child),
    // 1]`), is a borrow, and that use is tenure-checked; the counted uses are copies
    // (checkPlaceRead).
    static bool isCountedHandleRead(mlir::Value value)
    {
        auto uses = handleReadUses(value);
        return uses.counted && !uses.borrowed;
    }

    // A use that gives a handle an owner rc counts, with rc's retain for that very use: rc's
    // retain itself, a store into a field, an element, a global, an owning local or the result
    // slot (a return) that rc retained for (countedStoreRetain); a push or an unshift, `new
    // Shared`, an array literal's element, an `any` box, a `ts.ReturnVal`, each right after rc's
    // retain of what it takes (retainedRightBefore); or the initializer of an owning local, which
    // takes a count of its own (`ts.RetainSlot`). Any of them without its retain is not one: it
    // keeps the read without a count (`{ s: (q.child = p.child) }`'s field).
    static bool isCountedHandleReceiver(mlir::Operation *user, mlir::Value value)
    {
        if (mlir::isa<mlir_ts::RetainOp>(user))
        {
            return true;
        }

        if (auto varOp = mlir::dyn_cast<mlir_ts::VariableOp>(user))
        {
            return isOwningVariable(varOp) && llvm::any_of(varOp->getUsers(), [](mlir::Operation *varUser) {
                       return mlir::isa<mlir_ts::RetainSlotOp>(varUser);
                   });
        }

        if (auto storeOp = mlir::dyn_cast<mlir_ts::StoreOp>(user))
        {
            return storeOp.getValue() == value && countedStoreRetain(storeOp);
        }

        if (auto pushOp = mlir::dyn_cast<mlir_ts::ArrayPushOp>(user))
        {
            return retainedRightBefore(user, pushOp.getItems(), value);
        }

        if (auto unshiftOp = mlir::dyn_cast<mlir_ts::ArrayUnshiftOp>(user))
        {
            return retainedRightBefore(user, unshiftOp.getItems(), value);
        }

        if (auto castOp = mlir::dyn_cast<mlir_ts::CastOp>(user))
        {
            return mlir::isa<mlir_ts::AnyType>(castOp.getType()) && retainedRightBefore(user, user->getOperands(), value);
        }

        // `ts.ReturnVal` is the return before the affine lowering, which makes it a store into
        // the result slot
        if (mlir::isa<mlir_ts::SharedNewOp, mlir_ts::CreateArrayOp, mlir_ts::ReturnValOp>(user))
        {
            return retainedRightBefore(user, user->getOperands(), value);
        }

        return false;
    }

    // A use that passesOn hands on but that takes the value: a return's store into the function's
    // result slot (isResultSlot), and a cast that boxes it into `any`.
    static bool isHandOff(mlir::Operation *user, mlir::Value value)
    {
        if (auto storeOp = mlir::dyn_cast<mlir_ts::StoreOp>(user))
        {
            auto slot = storeOp.getReference().getDefiningOp<mlir_ts::VariableOp>();
            return storeOp.getValue() == value && slot && isResultSlot(slot);
        }

        auto castOp = mlir::dyn_cast<mlir_ts::CastOp>(user);
        return castOp && castOp.getIn() == value && mlir::isa<mlir_ts::AnyType>(castOp.getType());
    }

    // A return or an `any` box of a handle that rc counts (walkBorrowed).
    static bool isCountedHandOff(mlir::Operation *user, mlir::Value value)
    {
        return isHandOff(user, value) && isCountedHandleReceiver(user, value);
    }

    // rc's retains for what `op` takes: MLIRGen puts them in a row right before it
    // (mlirGenRetainCaptured, retainInsertedElements), one for each operand it does not take
    // over. Answers whether that row retains `value` once for each time `operands` hold it. A
    // retain is right before one op only, so no other receiver can count it as well.
    static bool retainedRightBefore(mlir::Operation *op, mlir::ValueRange operands, mlir::Value value)
    {
        auto needed = llvm::count(operands, value);
        if (needed == 0)
        {
            return false;
        }

        for (auto *previous = op->getPrevNode(); needed > 0 && previous; previous = previous->getPrevNode())
        {
            auto retainOp = mlir::dyn_cast<mlir_ts::RetainOp>(previous);
            if (!retainOp)
            {
                break;
            }

            if (retainOp.getReference() == value)
            {
                --needed;
            }
        }

        return needed == 0;
    }

    // The releases of the old value that rc's counted stores of a handle read make: `ts.Retain` of
    // the value, `ts.ReleaseSlot` of the slot, `ts.Store` of the value into it, in a row. Seen
    // through the read's casts, merges and locals that own nothing, as handleReadUses sees it.
    static void collectCountedStoreReleases(mlir::Value value, llvm::SmallPtrSetImpl<mlir::Operation *> &releases)
    {
        llvm::SmallVector<mlir::Value> values{value};
        llvm::SmallPtrSet<mlir::Value, 8> seen{value};
        while (!values.empty())
        {
            auto current = values.pop_back_val();
            for (auto &use : current.getUses())
            {
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

                    continue;
                }

                auto storeOp = mlir::dyn_cast<mlir_ts::StoreOp>(use.getOwner());
                if (!storeOp || storeOp.getValue() != current || !countedStoreRetain(storeOp))
                {
                    continue;
                }

                if (auto releaseSlotOp = mlir::dyn_cast_or_null<mlir_ts::ReleaseSlotOp>(storeOp->getPrevNode()))
                {
                    releases.insert(releaseSlotOp);
                }
            }
        }
    }

    // rc's retain for a store of a value, when it made one: `ts.Retain` of the value right before
    // the store, or before the `ts.ReleaseSlot` of the old value right before it -
    // `Retain(v)`, [`ReleaseSlot(slot)`], `Store(v, slot)`. A return's store into the result
    // slot comes after the scope exit's releases of the frame's locals and their slots:
    // `Retain(v)`, [`Release(..)`, `ReleaseSlot(..)`...], `Store(v, result)`. Null for a store rc
    // counts nothing for.
    static mlir::Operation *countedStoreRetain(mlir_ts::StoreOp storeOp)
    {
        auto *previous = storeOp->getPrevNode();
        if (auto slot = storeOp.getReference().getDefiningOp<mlir_ts::VariableOp>(); slot && isResultSlot(slot))
        {
            // the scope exit's releases, of locals and of their slots, in any order
            while (mlir::isa_and_nonnull<mlir_ts::ReleaseOp, mlir_ts::ReleaseSlotOp>(previous) &&
                   previous->getOperand(0) != storeOp.getValue())
            {
                previous = previous->getPrevNode();
            }
        }
        else if (auto releaseSlotOp = mlir::dyn_cast_or_null<mlir_ts::ReleaseSlotOp>(previous))
        {
            previous = releaseSlotOp.getSlot() == storeOp.getReference() ? releaseSlotOp->getPrevNode() : nullptr;
        }

        auto retainOp = mlir::dyn_cast_or_null<mlir_ts::RetainOp>(previous);
        return retainOp && retainOp.getReference() == storeOp.getValue() ? retainOp.getOperation() : nullptr;
    }

    // The call a value is the result of, when that result is a plain borrow of an argument. A borrowing
    // record (spec 32) is not one: the caller owns the box (checkRecordResults).
    static mlir::Operation *borrowingCall(mlir::Value value)
    {
        auto *def = value.getDefiningOp();
        if (!def || !isCall(def) || resultBorrows(def) < 0)
        {
            return nullptr;
        }

        auto args = callArgs(def);
        auto indices = resultBorrowsAll(def);
        return llvm::all_of(indices, [&](int index) { return static_cast<size_t>(index) < args.size(); }) ? def : nullptr;
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
        // (`s.value`) is SharedValue, a field reached through an interface is InterfaceField: its
        // position in the object is not known here (spec 28)
        llvm::SmallVector<std::pair<int64_t, mlir::Type>> places;
        static constexpr int64_t SharedValue = -2;
        static constexpr int64_t InterfaceField = -3;
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

    // With `place`, the chain of that place itself: where a store into it moves a value (spec 29).
    Chain chainOf(mlir::Operation *read, mlir::Value place = {})
    {
        Chain chain;
        llvm::SmallVector<mlir::Value> refs;
        llvm::SmallVector<mlir::Value> first;
        if (place)
        {
            refs.push_back(place);
        }
        else if (auto loadOp = mlir::dyn_cast<mlir_ts::LoadOp>(read))
        {
            refs.push_back(loadOp.getReference());
        }
        else
        {
            // a call's result that borrows its arguments - a plain borrow, or a borrowing record's box
            // (spec 32), which may hold a borrow of each - is somewhere under each of them
            chain.anyPlace = true;
            auto args = callArgs(read);
            for (auto index : resultBorrowsAll(read))
            {
                if (static_cast<size_t>(index) < args.size() && !argumentHoldsNoBlock(args[index]) &&
                    !isImmortalLiteral(args[index]))
                {
                    first.push_back(rootOf(args[index]));
                }
            }
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
                auto place = placeBase(ref);
                assert(place && "chainOf walks places only");
                switch (place.kind)
                {
                case PlaceKind::Field:
                    chain.places.push_back({ref.getDefiningOp<mlir_ts::PropertyRefOp>().getPosition(), ref.getType()});
                    break;
                case PlaceKind::SharedValue:
                    chain.places.push_back({Chain::SharedValue, ref.getType()});
                    handleBases.insert(rootOf(place.base));
                    break;
                case PlaceKind::InterfaceField:
                    chain.places.push_back({Chain::InterfaceField, ref.getType()});
                    break;
                default:
                    chain.places.push_back({-1, ref.getType()});
                    break;
                }

                bases.push_back(rootOf(place.base));
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
                    auto args = callArgs(call);
                    for (auto index : resultBorrowsAll(call))
                    {
                        if (static_cast<size_t>(index) < args.size())
                        {
                            bases.push_back(rootOf(args[index]));
                        }
                    }

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
                    // ...but a borrowing record given this value holds it (spec 32)
                    auto args = callArgs(user);
                    if (isRecordResult(user) && llvm::any_of(resultBorrowsAll(user), [&](int index) {
                            return static_cast<size_t>(index) < args.size() && args[index] == value;
                        }))
                    {
                        work.push_back(user->getResult(0));
                    }

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
    // `borrowed`, when given, is the value the borrow holds in place of the read's result (a value
    // moved into the place, spec 29), and `read` may then be null.
    bool dropsChain(mlir::Operation *drop, const Chain &chain, mlir::Operation *read, mlir::Value borrowed = {})
    {
        if (read && drop == read)
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

            // An assignment to a global destroys its old value and all that value owns. A root this
            // function does not own - `this`, a parameter, a global's or a field's value - may be
            // that very block, or under it (spec 29). A root this function made cannot. An immortal
            // global's one store is its initializer, whose old value is the empty one (spec 25).
            if (auto addressOfOp = slot.getDefiningOp<mlir_ts::AddressOfOp>())
            {
                return !addressOfOp->hasAttr(OWN_IMMORTAL_GLOBAL_ATTR_NAME) && !chain.owned();
            }

            // `x.value = ...` through any handle, of any type, may replace the payload another
            // handle reads (spec 23.3): a handle converts to one of a base class's type
            if (slot.getDefiningOp<mlir_ts::SharedValueRefOp>())
            {
                return llvm::any_of(chain.places, [&](auto &place) { return place.first == Chain::SharedValue; });
            }

            auto isField = [](int64_t position) { return position >= 0 || position == Chain::InterfaceField; };
            int64_t position = -1;
            if (auto propertyRefOp = slot.getDefiningOp<mlir_ts::PropertyRefOp>())
            {
                position = propertyRefOp.getPosition();
            }
            else if (isInterfaceField(slot))
            {
                position = Chain::InterfaceField;
            }

            // A field reached through an interface may be any field of the object under it, of any
            // type (a class may declare the field narrower than the interface): an overwrite through
            // an interface drops every field on the chain, and any field's overwrite drops a field
            // read through an interface (spec 28).
            return llvm::any_of(chain.places, [&](auto &place) {
                if (isField(position) && isField(place.first) &&
                    (position == Chain::InterfaceField || place.first == Chain::InterfaceField))
                {
                    return true;
                }

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

            auto &fromRead = derivedFrom(borrowed ? borrowed : read->getResult(0));
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
        // A handle read rc counts (spec 23.3): its counted uses are copies (walkBorrowed), and the
        // release of the old value a counted store makes, right before it stores the retained
        // read, cannot free the block read: it holds the retain's count by then
        // (`Shared.count(q.child = p.child)`).
        auto countedHandle = isHandle(result) && handleReadUses(result).counted;
        llvm::SmallPtrSet<mlir::Operation *, 4> countedStoreReleases;
        if (countedHandle)
        {
            collectCountedStoreReleases(result, countedStoreReleases);
        }

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
        }, keeping, countedHandle);

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

        if (keptParamOutlives(chain, outlives))
        {
            return false;
        }

        for (auto *drop : drops)
        {
            if (countedStoreReleases.contains(drop))
            {
                continue;
            }

            if (dropsChain(drop, chain, read) && outlives(drop))
            {
                return false;
            }
        }

        toErase.insert(bookkeeping.begin(), bookkeeping.end());
        return true;
    }

    // A local that holds a borrow through `user`, and the field path under it that holds it (empty:
    // the local's value itself): a store of it into the local, or into a field of the local's value
    // tuple at any depth (an object literal built in place), or the local declared from it.
    struct BorrowHolder
    {
        mlir_ts::VariableOp local;
        llvm::SmallVector<int64_t, 2> path;
    };

    static std::optional<BorrowHolder> borrowHolderOf(mlir::Operation *user)
    {
        BorrowHolder holder;
        mlir::Value ref;
        if (auto storeOp = mlir::dyn_cast<mlir_ts::StoreOp>(user))
        {
            ref = storeOp.getReference();
            while (auto propertyRefOp = ref.getDefiningOp<mlir_ts::PropertyRefOp>())
            {
                holder.path.insert(holder.path.begin(), propertyRefOp.getPosition());
                ref = propertyRefOp.getObjectRef();
            }
        }
        else if (auto varOp = mlir::dyn_cast<mlir_ts::VariableOp>(user))
        {
            ref = varOp.getResult();
        }

        auto varOp = ref ? ref.getDefiningOp<mlir_ts::VariableOp>() : mlir_ts::VariableOp();
        if (!varOp || varOp.getCaptured().value_or(false) || isCellVariable(varOp.getResult()))
        {
            return std::nullopt;
        }

        holder.local = varOp;
        return holder;
    }

    // Where the holder is given a new value: the local's declaration when it has an initializer
    // (`let l!: V` stores nothing, and in a loop the slot keeps the last round's value: fix-up
    // re-review N4), and every store into the field path that holds the borrow or into a prefix of
    // it (the whole local, `o.p` for `o.p.v`). A store into a sibling field, or below the path,
    // gives the path no new value.
    static llvm::SmallVector<mlir::Operation *, 4> holderRenewals(BorrowHolder holder)
    {
        llvm::SmallVector<mlir::Operation *, 4> renewals;
        if (holder.local.getInitializer())
        {
            renewals.push_back(holder.local);
        }

        llvm::SmallVector<std::pair<mlir::Value, size_t>> refs{{holder.local.getResult(), 0}};
        while (!refs.empty())
        {
            auto [ref, depth] = refs.pop_back_val();
            for (auto *refUser : ref.getUsers())
            {
                if (auto propertyRefOp = mlir::dyn_cast<mlir_ts::PropertyRefOp>(refUser))
                {
                    if (depth < holder.path.size() && propertyRefOp.getPosition() == holder.path[depth])
                    {
                        refs.push_back({propertyRefOp.getResult(), depth + 1});
                    }
                }
                else if (auto storeOp = mlir::dyn_cast<mlir_ts::StoreOp>(refUser); storeOp && storeOp.getReference() == ref)
                {
                    renewals.push_back(storeOp);
                }
            }
        }

        return renewals;
    }

    // Whether one field path is on the way to the other: a read of either may reach the other's value.
    static bool pathsMeet(llvm::ArrayRef<int64_t> a, llvm::ArrayRef<int64_t> b)
    {
        auto common = std::min(a.size(), b.size());
        return a.take_front(common) == b.take_front(common);
    }

    // A read of what a mortal global holds (spec 25: an immortal one's one store is its
    // initializer): a heap value it owns, not a handle (spec 23.3 counts those).
    bool isGlobalRead(mlir_ts::LoadOp loadOp)
    {
        auto addressOfOp = loadOp.getReference().getDefiningOp<mlir_ts::AddressOfOp>();
        auto result = loadOp.getResult();
        return addressOfOp && !addressOfOp->hasAttr(OWN_IMMORTAL_GLOBAL_ATTR_NAME) && ownsHeap(result) && !isHandle(result);
    }

    // Such a read is a borrow of the value the global owns - `const c = K.w` names the read itself,
    // with no retain - and an assignment of the global releases it. So no use of the read, nor of
    // what it produces (walkBorrowed), may run after one in this function. A path back to a use
    // through the read itself is a new read. A call that may assign the global, itself or through
    // its callees, is such an assignment too (follow-up 11: `const c = G; reset(); print(c.x)`); the
    // read given to that call as an argument is #639's. (C1 of the fix-up review: `const c = K.w; K.w = new V(i + 1); print(c.x)` in a loop
    // read the block the assignment freed.)
    //
    // Where the walk meets a keep, the read is either followed into what holds it or rejected -
    // never let go unchecked (fix-up re-review N2: a kept read is not always another check's). A
    // local holds it (borrowHolderOf): an object literal built in place (`const o = { v: c }`,
    // `{ p: { v: K.w } }`), a `let` or a destructured local declared or assigned from it
    // (`let l = o.v`, `let { v } = o`). Every use of that local under the path that holds the borrow
    // - its reads, a copy, a spread (`ts.DeconstructTuple`), what those lead to - is a use of the
    // read, made again only where the local is given a new value (holderRenewals): never at the
    // read, which renews `c` and not `o.v` - a holder declared before a loop keeps the old borrow
    // across the back edge until it is stored again (fix-up re-review N3). The local's own retain
    // and releases are its bookkeeping, which the owner checks decide. A sibling field never given
    // the borrow is not followed. A value made from the borrow and stored into another local makes
    // that one a holder too. Any other keep - a field of an object, an array, a global, a callee
    // that keeps it, a capture, a return - cannot be bounded here and is an error. A string read is
    // copied where it is kept (spec 22), so its keeps are not this check's.
    void checkGlobalRead(mlir_ts::LoadOp read)
    {
        // each access of a global is a `ts.AddressOf` of its own
        auto global = read.getReference().getDefiningOp<mlir_ts::AddressOfOp>().getGlobalName();
        llvm::SmallVector<BorrowUse> uses;
        llvm::StringSet<> followed;
        mlir::Operation *keptAt = nullptr;
        // rc's temporary release of a call's result that borrows an argument goes with that result
        // (walkBorrowed's `keeping`); the owner checks decide it
        llvm::SmallVector<mlir::Operation *> bookkeeping;
        Keeping keeping{&bookkeeping, false};
        std::function<bool(mlir::Operation *, llvm::ArrayRef<mlir::Operation *>)> intoLocal;

        // the uses of what `holder` holds, made again at its renewals: those depend on the holder
        // alone, not on the way the walk reached it, so a holder is followed once. A value loaded
        // out of it is made again only by a path back through that load (the load itself is a use,
        // bounded by the renewals): a store into the path after the load gives the loaded value
        // nothing new.
        auto followHolder = [&](BorrowHolder holder) {
            std::string key;
            llvm::raw_string_ostream keyOut(key);
            keyOut << holder.local.getOperation();
            for (auto position : holder.path)
            {
                keyOut << "." << position;
            }

            if (!followed.insert(keyOut.str()).second)
            {
                return true;
            }

            auto localKills = holderRenewals(holder);
            llvm::SmallVector<std::pair<mlir::Value, llvm::SmallVector<int64_t, 2>>> refs{{holder.local.getResult(), {}}};
            while (!refs.empty())
            {
                auto [ref, path] = refs.pop_back_val();
                for (auto *refUser : ref.getUsers())
                {
                    if (auto propertyRefOp = mlir::dyn_cast<mlir_ts::PropertyRefOp>(refUser))
                    {
                        auto fieldPath = path;
                        fieldPath.push_back(propertyRefOp.getPosition());
                        if (pathsMeet(fieldPath, holder.path))
                        {
                            refs.push_back({propertyRefOp.getResult(), fieldPath});
                        }

                        continue;
                    }

                    if (ref == holder.local.getResult() &&
                        mlir::isa<mlir_ts::RetainSlotOp, mlir_ts::ReleaseSlotOp, mlir_ts::DebugVariableOp>(refUser))
                    {
                        continue;
                    }

                    uses.push_back({refUser, {localKills.begin(), localKills.end()}});
                    if (auto loadOp = mlir::dyn_cast<mlir_ts::LoadOp>(refUser))
                    {
                        if (auto *kept = walkBorrowed(loadOp.getResult(), loadOp.getOperation(), uses, intoLocal, keeping))
                        {
                            keptAt = keptAt ? keptAt : kept;
                            return false;
                        }
                    }
                }
            }

            return true;
        };

        intoLocal = [&](mlir::Operation *user, llvm::ArrayRef<mlir::Operation *> kills) {
            // a tuple taken apart (`{ ...b }`, a destructuring): each field is the borrow again
            if (mlir::isa<mlir_ts::DeconstructTupleOp>(user))
            {
                for (auto result : user->getResults())
                {
                    if (!mayPointInto(result))
                    {
                        continue;
                    }

                    if (auto *kept = walkBorrowed(result, kills, uses, intoLocal, keeping))
                    {
                        keptAt = keptAt ? keptAt : kept;
                        return false;
                    }
                }

                return true;
            }

            auto holder = borrowHolderOf(user);
            return holder && followHolder(*holder);
        };

        auto *kept = walkBorrowed(read.getResult(), read.getOperation(), uses, intoLocal, keeping);

        // a value made from the borrow (a field of a tuple read out of a holder, `{ ...b }`) stored
        // into another local, or declaring one: that local holds the borrow too
        for (size_t index = 0; !kept && !keptAt && index < uses.size(); index++)
        {
            auto *op = uses[index].op;
            auto storeOp = mlir::dyn_cast<mlir_ts::StoreOp>(op);
            auto varOp = mlir::dyn_cast<mlir_ts::VariableOp>(op);
            auto value = storeOp ? storeOp.getValue() : varOp ? varOp.getInitializer() : mlir::Value();
            if (!value || !mayPointInto(value))
            {
                continue;
            }

            if (auto holder = borrowHolderOf(op))
            {
                followHolder(*holder);
            }
        }

        if (kept || keptAt)
        {
            if (mlir::isa<mlir_ts::StringType>(read.getType()))
            {
                return;
            }

            if (reporting())
            {
                auto *at = keptAt ? keptAt : kept;
                at->emitError("the value read from '")
                    << global << "' is kept where its end at an assignment of '" << global
                    << "' cannot be checked; -mm=own cannot prove it outlives the global's value";
                signalPassFailure();
            }

            return;
        }

        llvm::SmallVector<mlir::Operation *> ends;
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
            if (other && other.getGlobalName() == global)
            {
                ends.push_back(op);
            }

            // a call that may give the global a new value (OwnershipSignaturePass, pinAssignedGlobals):
            // `const c = G; reset(); print(c.x)`
            if (isCall(op))
            {
                auto assigned = op->getAttrOfType<mlir::ArrayAttr>(OWN_ASSIGNS_GLOBALS_ATTR_NAME);
                if (op->hasAttr(OWN_ASSIGNS_ANY_GLOBAL_ATTR_NAME) ||
                    (assigned && llvm::any_of(assigned.getAsRange<mlir::StringAttr>(),
                                              [&](mlir::StringAttr name) { return name.getValue() == global; })))
                {
                    ends.push_back(op);
                }
            }
        });

        for (auto *end : ends)
        {
            for (auto &use : uses)
            {
                if (use.op != end && reachableAfter(end, use.op, use.kills))
                {
                    reportBorrowOutlives(use.op, end, ownerName(read.getResult()), global);
                    return;
                }
            }
        }
    }

    // A borrow read out of a parameter this function keeps, taken before the parameter moves into a
    // new owner (spec 33, review fix): from the move on it is bounded by that owner, as a read through
    // the owner would be (spec 29) - by the owner's ends and the drops of its place after the move.
    // `const inner = p.v; let r = p; r = new N(...); inner.x` reads what the assignment freed. A move
    // into anything that is not such a place (a call that keeps it, a return, a push) ends the borrow
    // there. So do the releases of the parameter's slot (spec 33's exit releases). True when some use
    // of the borrow comes after one of these (`outlives` reported it).
    bool keptParamOutlives(const Chain &chain, llvm::function_ref<bool(mlir::Operation *)> outlives)
    {
        for (auto &root : chain.roots)
        {
            auto load = keptParamLoadOf(root.first);
            if (!load)
            {
                continue;
            }

            // the moves of the parameter: what takes a read of its slot
            llvm::SmallVector<std::pair<mlir::Operation *, mlir::Value>> takers;
            for (auto *user : load.getReference().getUsers())
            {
                if (mlir::isa<mlir_ts::ReleaseSlotOp>(user))
                {
                    if (outlives(user))
                    {
                        return true;
                    }

                    continue;
                }

                auto readOp = mlir::dyn_cast<mlir_ts::LoadOp>(user);
                if (!readOp)
                {
                    continue;
                }

                forEachUse(readOp.getResult(), [&](mlir::Operation *use, mlir::Value used) {
                    auto varOp = mlir::dyn_cast<mlir_ts::VariableOp>(use);
                    if (mlir::isa<mlir_ts::RetainOp>(use))
                    {
                        auto *taker = takerOf(use, used);
                        takers.push_back({taker ? taker : use, used});
                    }
                    else if ((varOp && isOwningVariable(varOp)) || (isCall(use) && isKeptArgument(use, used)))
                    {
                        takers.push_back({use, used});
                    }
                });
            }

            for (auto [taker, moved] : takers)
            {
                Chain owner;
                if (!destinationChain(taker, owner))
                {
                    if (outlives(taker))
                    {
                        return true;
                    }

                    continue;
                }

                auto afterMove = [&](mlir::Operation *end) {
                    return end != taker && reachableAfter(taker, end, llvm::ArrayRef<mlir::Operation *>{}) && outlives(end);
                };

                for (auto *end : rootEnds(owner))
                {
                    if (afterMove(end))
                    {
                        return true;
                    }
                }

                for (auto *drop : drops)
                {
                    if (dropsChain(drop, owner, nullptr, moved) && afterMove(drop))
                    {
                        return true;
                    }
                }
            }
        }

        return false;
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
            else if (auto refOp = mlir::dyn_cast_or_null<mlir_ts::InterfaceSymbolRefOp>(calleeOp))
            {
                // a method reached through an interface (spec 31): `<interface>.<method>`
                auto method = mlir::dyn_cast<mlir::StringAttr>(refOp.getIdentifier());
                auto interfaceType = mlir::dyn_cast<mlir_ts::InterfaceType>(refOp.getInterfaceVal().getType());
                if (method && interfaceType)
                {
                    return (interfaceType.getName().getValue() + "." + method.getValue()).str();
                }
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
        auto place = placeBase(ref);
        std::string step;
        switch (place.kind)
        {
        case PlaceKind::Field: {
            auto field = fieldName(ref.getDefiningOp<mlir_ts::PropertyRefOp>());
            if (field.empty())
            {
                return {};
            }

            step = "." + field;
            break;
        }
        case PlaceKind::Element:
            step = "[]";
            break;
        case PlaceKind::SharedValue:
            step = ".value";
            break;
        case PlaceKind::InterfaceField: {
            auto field = mlir::dyn_cast<mlir::StringAttr>(ref.getDefiningOp<mlir_ts::InterfaceSymbolRefOp>().getIdentifier());
            if (!field)
            {
                return {};
            }

            step = "." + field.getValue().str();
            break;
        }
        case PlaceKind::None:
            return {};
        }

        auto base = rootOf(place.base);
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
        return own_facts::ownsHeap(location, type);
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
        // a borrowing record (spec 32) takes nothing for what it borrows, and its fresh values move in
        // as an owning record's do
        if (!varOp || !isConstantRecordLocal(varOp) || !(isOwningRecord(value) || borrowingRecordOf(varOp)))
        {
            return false;
        }

        // a fresh value moved in, its move decided by its own check against this retain
        auto movesIn = llvm::any_of(varOp->getUsers(), [&](mlir::Operation *user) {
            auto propertyRefOp = mlir::dyn_cast<mlir_ts::PropertyRefOp>(user);
            if (!propertyRefOp)
            {
                return false;
            }

            // a method's field is a `!ts.bound_ref`, which holds a function, not a block
            auto refType = mlir::dyn_cast<mlir_ts::RefType>(propertyRefOp.getType());
            if (refType && !ownsHeap(propertyRefOp.getLoc(), refType.getElementType()))
            {
                return false;
            }

            return llvm::any_of(propertyRefOp->getUsers(),
                                [](mlir::Operation *fieldUser) { return mlir::isa<mlir_ts::StoreOp>(fieldUser); });
        });

        return !movesIn || recordRetainOf(varOp) != nullptr;
    }

    // rc's one `ts.Retain` of a record local's one read, where the read is only returned: stored
    // into a local that owns nothing and is read only by returns (the result slot), or copied into
    // the box it is given in as an object or an interface (spec 27). It is the reference rc gives
    // each fresh value stored into the record, so it is where they move in. Null for any other
    // record.
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
        auto boxed = false;
        auto other = false;
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

            // copied into the block it is given in as an object or an interface (spec 27): the box
            // owns what the record held, and its own check decides where the box goes
            auto storeOp = mlir::dyn_cast<mlir_ts::StoreOp>(user);
            auto box = storeOp && storeOp.getValue() == read.getResult() ? recordBoxOf(storeOp.getReference())
                                                                         : mlir_ts::NewOp();
            if (box && box.getResult() == storeOp.getReference())
            {
                if (boxed)
                {
                    return nullptr;
                }

                boxed = true;
                continue;
            }

            other = true;
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
            auto slot = storeOp && storeOp.getValue() == read.getResult()
                            ? storeOp.getReference().getDefiningOp<mlir_ts::VariableOp>()
                            : mlir_ts::VariableOp();
            if (!slot || !(isResultSlot(slot) || isOwningVariable(slot)))
            {
                return nullptr;
            }
        }

        // a boxed record goes nowhere else: the box is its one owner
        if (boxed && other)
        {
            return nullptr;
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

    // `return [...a]`: the builder of isBuiltArrayRead, read once, when it is done, and returned.
    // The return's `ts.Retain` (`retain`) is the reference the caller gets: the fresh array moves
    // into the builder and then out of the function, and nothing else holds it. The builder's
    // other reads only bind its `push` (`ts.ThisSymbolRef @__array_push`) or are not used. A store
    // into a field is the same move, into a container: `h.c = [...a]`, and a generator's local
    // (`const groups = [...result]`), which is a field of its state object.
    bool isBuiltArrayReturned(mlir::Operation *retain, mlir::Value value)
    {
        auto read = value.getDefiningOp<mlir_ts::LoadOp>();
        auto builder = read ? read.getReference().getDefiningOp<mlir_ts::VariableOp>() : mlir_ts::VariableOp();
        if (!builder || isOwningVariable(builder) || builder.getCaptured().value_or(false) || !builder.getInitializer() ||
            !mlir::isa_and_nonnull<mlir_ts::CreateArrayOp, mlir_ts::NewArrayOp>(builder.getInitializer().getDefiningOp()))
        {
            return false;
        }

        // the read is retained and returned - stored into the function's result slot - and nothing else
        auto returned = false;
        for (auto *user : read->getUsers())
        {
            if (user == retain)
            {
                continue;
            }

            auto storeOp = mlir::dyn_cast<mlir_ts::StoreOp>(user);
            auto resultSlot = storeOp && storeOp.getValue() == read.getResult()
                                  ? storeOp.getReference().getDefiningOp<mlir_ts::VariableOp>()
                                  : mlir_ts::VariableOp();
            auto intoField = storeOp && storeOp.getValue() == read.getResult() &&
                             storeOp.getReference().getDefiningOp<mlir_ts::PropertyRefOp>();
            if (!((resultSlot && isResultSlot(resultSlot)) || intoField) || returned)
            {
                return false;
            }

            returned = true;
        }

        if (!returned)
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

            if (reachableAfter(read, user, builder))
            {
                return false;
            }

            if (auto loadOp = mlir::dyn_cast<mlir_ts::LoadOp>(user))
            {
                return llvm::all_of(loadOp->getUsers(), [](mlir::Operation *loadUser) {
                    return mlir::isa<mlir_ts::ThisSymbolRefOp>(loadUser) && loadUser->use_empty();
                });
            }

            return mlir::isa<mlir_ts::ArrayPushOp, mlir_ts::ArrayUnshiftOp>(user) && use.getOperandNumber() == 0;
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

    // A string constant cast to `string` is the global itself, which a release skips; so is a read
    // of an immortal global (spec 25).
    static bool isImmortalLiteral(mlir::Value value)
    {
        if (isImmortalGlobalRead(value))
        {
            return true;
        }

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

        // `assert(c, message)` prints the message when it fails, and keeps it either way
        if (mlir::isa<mlir_ts::AssertOp>(user))
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

        // ...and so does a cast to `null` or `undefined`, which holds no block: `--di` records the
        // `r` that `if (r == null)` narrows as one
        if (auto castOp = mlir::dyn_cast<mlir_ts::CastOp>(user))
        {
            return mlir::isa<mlir_ts::BooleanType, mlir_ts::NullType, mlir_ts::UndefinedType>(castOp.getType()) ||
                   castOp.getType().isInteger(1) || isBoundForCall(castOp);
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
        llvm::DenseMap<mlir::Operation *, mlir::Value> readOf;
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
                readOf.try_emplace(user, used);
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

            llvm::SmallVector<mlir::Operation *, 1> made;
            if (definition)
            {
                made.push_back(definition);
            }

            // the call that keeps it, given it again through another operand
            if (auto *again = callGivenAgain(taker, value, {value}))
            {
                reportUseAfterMove(again, taker, name);
                return false;
            }

            for (auto *use : llvm::concat<mlir::Operation *const>(takers, reads))
            {
                if (use != taker && reachableAfter(taker, use, definition))
                {
                    // a read through the new owner, while it holds the value (spec 29)
                    if (auto found = readOf.find(use);
                        found != readOf.end() && readsThroughOwner(taker, found->second, use, made))
                    {
                        continue;
                    }

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

    // A call that keeps an argument and is given the same value again through another operand - the
    // value itself, a view of it, something read out of it, another read of its slot (`f(n, n)`,
    // `f(h, n, n.v)` with only the first `n` kept): the callee takes the value while it borrows it
    // through the other operand, and may free it under that borrow, so the other operand is a use
    // after the move (spec 33, review). `sources` are what the moved value is: the value, or every
    // read of the owning slot it came from. The call itself, or null.
    mlir::Operation *callGivenAgain(mlir::Operation *taker, mlir::Value moved, llvm::ArrayRef<mlir::Value> sources)
    {
        if (!isCall(taker) || ownedParams(taker).empty())
        {
            return nullptr;
        }

        auto args = callArgs(taker);
        auto kept = ownedParams(taker);
        auto movedRoot = rootOf(moved);
        // one kept position is the move itself; the value at a second one (`f(n, n)` with both
        // kept) is a second owner, a use after the first move like any other operand
        auto skipped = false;
        for (auto &operand : taker->getOpOperands())
        {
            // the moved argument itself, where the callee keeps it
            auto number = operand.getOperandNumber();
            if (!skipped && !args.empty() && number >= args.getBeginOperandIndex() &&
                number < args.getBeginOperandIndex() + args.size() &&
                llvm::is_contained(kept, static_cast<int32_t>(number - args.getBeginOperandIndex())) &&
                rootOf(operand.get()) == movedRoot)
            {
                skipped = true;
                continue;
            }

            for (auto source : sources)
            {
                if (derivedFrom(source).contains(operand.get()))
                {
                    return taker;
                }
            }
        }

        return nullptr;
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
    // Ops nested in a region that is not the function body are compared by their ancestors in the
    // body. That answers "reachable" whenever the two share an ancestor, which can only turn a
    // program into an error. (By this pass async bodies and lambdas are functions of their own;
    // after the affine lowering no op of a body holds a region, so the case is kept for safety.)
    bool reachableAfter(mlir::Operation *from, mlir::Operation *to, mlir::Operation *kill)
    {
        return reachableAfter(from, to, kill ? llvm::ArrayRef<mlir::Operation *>(kill) : llvm::ArrayRef<mlir::Operation *>());
    }

    // Whether reachableAfter can order `from` and `to` at all: both inside the function body and not
    // one op of it (nested in the same statement). A check whose safe answer is "not reachable" must
    // refuse what it cannot order, where reachableAfter would answer "reachable".
    bool orderedInBody(mlir::Operation *from, mlir::Operation *to)
    {
        auto &body = getFunction().getFunctionBody();
        from = body.findAncestorOpInRegion(*from);
        to = body.findAncestorOpInRegion(*to);
        return from && to && from != to;
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
    // The one op that uses `value`, but `--di`'s ts.DebugVariable, which only names it; or null.
    static mlir::Operation *soleUser(mlir::Value value)
    {
        mlir::Operation *found = nullptr;
        for (auto *user : value.getUsers())
        {
            if (mlir::isa<mlir_ts::DebugVariableOp>(user))
            {
                continue;
            }

            if (found)
            {
                return nullptr;
            }

            found = user;
        }

        return found;
    }

    static mlir::Value twinReadAfter(mlir::Operation *retain, mlir::Value value)
    {
        auto first = value.getDefiningOp<mlir_ts::LoadOp>();
        if (!first || !mlir::isa<mlir_ts::RetainOp>(retain) || soleUser(value) != retain || retain->getBlock() != first->getBlock())
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
    // the slot the move reaches, which must be dominated by it - or, with `placeReleases`, is
    // placed on the paths that did not move it (spec 30). Any other use of the slot the
    // move reaches - a read, an assignment's own read of it - is a use after the move. Reports and
    // returns false when it is not a move.
    //
    // An assignment the move reaches gives the slot a new value to own (#528): a path through it
    // is a new value, as a path through the declaration is. Its release before the store goes
    // with the move - what the slot held is the receiver's - and the releases only a path through
    // an assignment reaches stay, to destroy the new value.
    bool checkSlotMove(mlir::Operation *retain, mlir::Value value, mlir_ts::LoadOp load,
                       llvm::SetVector<mlir::Operation *> &toErase, bool placeReleases = false)
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

        auto assignments = slotAssignments(slot);
        llvm::SmallVector<mlir::Operation *> kills(assignments.begin(), assignments.end());
        if (varOp)
        {
            kills.push_back(varOp);
        }

        // the move's own read, met again around a loop the slot was declared outside of
        if (reachableAfter(taker, load, kills))
        {
            reportMovedInLoop(taker, name);
            return false;
        }

        // the slot given another value on some path between the move's read and its taker - inside
        // the call's own argument list, `f(n, (n = mk(7), 0))`, on one arm of it, `f(n, c ? (n =
        // mk(7), 0) : 0)`: the assignment's release frees what the read holds before the callee takes
        // it, so the assignment is a use of the slot after the move. The taker may be an assignment
        // of the slot itself (`n = n`): it is the destination, not between. A read and a taker
        // reachableAfter cannot order (one statement kept as a region, or outside the body) refuse.
        {
            llvm::SmallVector<mlir::Operation *, 2> readAgain{load.getOperation()};
            if (varOp)
            {
                readAgain.push_back(varOp);
            }

            for (auto *assignment : assignments)
            {
                if (assignment != taker && reachableAfter(load, assignment, varOp) &&
                    reachableAfter(assignment, taker, readAgain))
                {
                    reportUseAfterMove(assignment, taker, name);
                    return false;
                }
            }

            llvm::SmallVector<mlir::Operation *> killsBetween(kills);
            llvm::erase(killsBetween, taker);
            if (!orderedInBody(load, taker) || !reachableAfter(load, taker, killsBetween))
            {
                reportUseAfterMove(taker, taker, name);
                return false;
            }
        }

        // the call that keeps it, given the slot's value again through another operand: a read of
        // the slot that reaches the call without crossing an assignment of it (not one made before
        // an assignment gave the slot its current value, `const arr = [n]; n = mk(5); f(n, arr)`);
        // since the move's own read reaches the call so too, such a read holds the moved value
        llvm::SmallVector<mlir::Value> slotReads;
        for (auto *user : slot.getUsers())
        {
            if (auto readOp = mlir::dyn_cast<mlir_ts::LoadOp>(user); readOp && reachableAfter(readOp, taker, kills))
            {
                slotReads.push_back(readOp.getResult());
            }
        }

        if (auto *again = callGivenAgain(taker, value, slotReads))
        {
            reportUseAfterMove(again, taker, name);
            return false;
        }

        // Every read of the slot is a use of `a`, and so is every use of what a read returned: a
        // `const b = a` is folded into its load, so `b.x` after the move reads the moved value
        // through a load that came before it. A read through the new owner, while it holds the
        // value, is not such a use (spec 29).
        auto throughOwner = [&](mlir::Operation *user, mlir::Value used) {
            return readsThroughOwner(taker, used, user, kills);
        };

        for (auto *use : slot.getUsers())
        {
            if (mlir::isa<mlir_ts::ReleaseSlotOp>(use))
            {
                continue;
            }

            if (use != load.getOperation() && reachableAfter(taker, use, kills))
            {
                auto readOp = mlir::dyn_cast<mlir_ts::LoadOp>(use);
                if (!readOp || !onlyReadThroughOwner(readOp.getResult(), throughOwner))
                {
                    reportUseAfterMove(use, taker, name);
                    return false;
                }

                continue;
            }

            if (auto readOp = mlir::dyn_cast<mlir_ts::LoadOp>(use))
            {
                if (auto *late = readUsedAfter(readOp, retain, taker, varOp, throughOwner))
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
            if (!mlir::isa<mlir_ts::ReleaseSlotOp>(use) || !reachableAfter(acquire, use, kills))
            {
                continue;
            }

            // a release that one path reaches holding the moved value and another holding a new
            // one an assignment after the move stored: there is no single answer for it
            auto someMoved = !dominance->properlyDominates(acquire, use);
            for (auto *assignment : assignments)
            {
                llvm::SmallVector<mlir::Operation *, 2> movedAgain{load.getOperation()};
                if (varOp)
                {
                    movedAgain.push_back(varOp);
                }

                someMoved = someMoved ||
                            (reachableAfter(acquire, assignment, varOp) && reachableAfter(assignment, use, movedAgain));
            }

            if (someMoved)
            {
                // ...but the paths that did not move it can release it before they join the
                // others (spec 30)
                if (placeReleases && placeOnUnmovedEdges(varOp, acquire, retain, taker, assignments, toErase))
                {
                    return true;
                }

                reportMovedOnSomePaths(taker, use, name, placeReleases ? joinedUse : nullptr);
                return false;
            }

            moved.push_back(use);
        }

        toErase.insert(moved.begin(), moved.end());
        return true;
    }

    // ---- A move on some paths only (spec 30) ----
    //
    // `if (c) { best = v; }`: after the paths join, `v` holds its value on one and a moved one on
    // the other, and its one release there has no single answer. No run-time record of the move is
    // kept (spec 2.2); instead, where nothing uses the value after the join, the release is placed
    // on each edge into the join that did not move it - freeing the value earlier, where it is
    // already dead - and the releases after the join go. Each path frees the value once: through
    // its new owner where it moved, through the edge where it did not.

    // What the slot of an owning local holds at a point: nothing yet or any more (declared later,
    // released), its value, or a value moved out of it. A point some paths reach holding the value
    // and others having moved it is where the paths join.
    enum SlotState : unsigned
    {
        SlotDead = 1,
        SlotLive = 2,
        SlotMoved = 4
    };

    static bool isJoined(unsigned state)
    {
        return (state & SlotLive) && (state & SlotMoved);
    }

    // A value of this type can point into a block: anything but a number, a boolean, a literal or
    // an empty value.
    static bool carriesPointer(mlir::Type type)
    {
        return !mlir::isa<mlir_ts::NumberType, mlir_ts::BooleanType, mlir_ts::LiteralType, mlir_ts::UndefinedType,
                          mlir_ts::NullType, mlir_ts::VoidType, mlir::IntegerType, mlir::IndexType, mlir::FloatType>(type);
    }

    // A release on an edge into the join that did not move the local's value (spec 30), put in after
    // the analysis.
    struct EdgeRelease
    {
        mlir::Value slot;
        mlir::Block *from;
        unsigned successor;
        mlir::Location loc;
    };

    llvm::SmallVector<EdgeRelease> edgeReleases;

    // The use that kept placeOnUnmovedEdges from placing the releases, for the error's note.
    mlir::Operation *joinedUse = nullptr;

    // Can control reach `to` from the start of `start`, without passing `kills` on the way?
    bool reachableFrom(mlir::Block *start, mlir::Operation *to, llvm::ArrayRef<mlir::Operation *> kills)
    {
        auto &body = getFunction().getFunctionBody();
        to = body.findAncestorOpInRegion(*to);
        if (!to)
        {
            return true;
        }

        llvm::SmallVector<mlir::Operation *, 2> killAt;
        for (auto *kill : kills)
        {
            if (auto *top = body.findAncestorOpInRegion(*kill))
            {
                killAt.push_back(top);
            }
        }

        llvm::SmallPtrSet<mlir::Block *, 16> seen;
        llvm::SmallVector<mlir::Block *> work{start};
        while (!work.empty())
        {
            auto *block = work.pop_back_val();
            if (!seen.insert(block).second)
            {
                continue;
            }

            auto killed = [&](mlir::Operation *before) {
                return llvm::any_of(killAt, [&](mlir::Operation *kill) {
                    return kill->getBlock() == block && (!before || kill->isBeforeInBlock(before) || kill == before);
                });
            };

            if (to->getBlock() == block && !killed(to))
            {
                return true;
            }

            if (!killed(nullptr))
            {
                work.append(block->getSuccessors().begin(), block->getSuccessors().end());
            }
        }

        return false;
    }

    // The blocks of an exception handler: from a landing pad to the end of its catch. A release there
    // would be a call inside an exception funclet.
    llvm::SmallPtrSet<mlir::Block *, 8> handlerBlocks()
    {
        llvm::SmallPtrSet<mlir::Block *, 8> handler;
        llvm::SmallVector<mlir::Block *> work;
        for (auto &block : getFunction().getFunctionBody())
        {
            if (!block.empty() && mlir::isa<mlir_ts::LandingPadOp>(block.front()))
            {
                work.push_back(&block);
            }
        }

        while (!work.empty())
        {
            auto *block = work.pop_back_val();
            if (!handler.insert(block).second)
            {
                continue;
            }

            auto ends = llvm::any_of(*block, [](mlir::Operation &op) {
                return mlir::isa<mlir_ts::EndCatchOp, mlir_ts::EndCleanupOp>(op);
            });
            if (!ends)
            {
                work.append(block->getSuccessors().begin(), block->getSuccessors().end());
            }
        }

        return handler;
    }

    // The move at `acquire` out of the owning local `varOp` reaches a release that other paths reach
    // holding the value. Places a release of the slot on every edge into the join that holds the value
    // and has not moved it, and erases every release a moved value reaches - when nothing uses the
    // value, through the slot or through anything read out of it, where the paths have joined. False,
    // with nothing placed or erased, when it cannot.
    // The slot of an owning local through the function (spec 30): its state at the entry and the exit
    // of each block, and before each op of the body.
    struct SlotStates
    {
        llvm::DenseMap<mlir::Block *, unsigned> entry;
        llvm::DenseMap<mlir::Block *, unsigned> exit;
        llvm::DenseMap<mlir::Operation *, unsigned> before;
    };

    bool placeOnUnmovedEdges(mlir_ts::VariableOp varOp, mlir::Operation *acquire, mlir::Operation *retain,
                             mlir::Operation *taker, llvm::ArrayRef<mlir::Operation *> assignments,
                             llvm::SetVector<mlir::Operation *> &toErase)
    {
        joinedUse = nullptr;
        llvm::SmallVector<mlir::Operation *> releases;
        if (!slotReleasesInBody(varOp, acquire, assignments, releases))
        {
            return false;
        }

        auto states = slotStates(varOp, acquire, assignments, releases);

        // every release a moved value reaches goes; one some path reaches with nothing in the slot
        // is not this rule's
        llvm::SmallVector<mlir::Operation *> erased;
        for (auto *release : releases)
        {
            auto state = states.before.lookup(release);
            if (state & SlotMoved)
            {
                if (state & SlotDead)
                {
                    return false;
                }

                erased.push_back(release);
            }
        }

        llvm::SmallVector<EdgeRelease> placed;
        llvm::SetVector<mlir::Block *> releasedInto;
        if (!unmovedEdges(varOp, states, placed, releasedInto))
        {
            return false;
        }

        if (!noUseWhereJoined(varOp, retain, taker, assignments, states, releasedInto))
        {
            return false;
        }

        toErase.insert(erased.begin(), erased.end());
        edgeReleases.append(placed.begin(), placed.end());
        return true;
    }

    // The releases of the slot, when the local, the move, its assignments and its releases are all in
    // the function's body. False when one is not.
    bool slotReleasesInBody(mlir_ts::VariableOp varOp, mlir::Operation *acquire,
                            llvm::ArrayRef<mlir::Operation *> assignments, llvm::SmallVectorImpl<mlir::Operation *> &releases)
    {
        auto &body = getFunction().getFunctionBody();
        if (!varOp || varOp.getCaptured().value_or(false) || varOp->getParentRegion() != &body ||
            acquire->getParentRegion() != &body ||
            llvm::any_of(assignments, [&](mlir::Operation *op) { return op->getParentRegion() != &body; }))
        {
            return false;
        }

        auto slot = varOp.getResult();
        for (auto *user : slot.getUsers())
        {
            if (mlir::isa<mlir_ts::ReleaseSlotOp>(user))
            {
                if (user->getParentRegion() != &body)
                {
                    return false;
                }

                releases.push_back(user);
            }
        }

        return true;
    }

    // The slot's state through the function, forward from its entry: assigned, moved at `acquire`,
    // released.
    SlotStates slotStates(mlir_ts::VariableOp varOp, mlir::Operation *acquire, llvm::ArrayRef<mlir::Operation *> assignments,
                          llvm::ArrayRef<mlir::Operation *> releases)
    {
        auto &body = getFunction().getFunctionBody();
        auto step = [&](mlir::Operation *op, unsigned state) -> unsigned {
            if (op == varOp.getOperation() || llvm::is_contained(assignments, op))
            {
                return SlotLive;
            }

            if (op == acquire)
            {
                return SlotMoved;
            }

            if (llvm::is_contained(releases, op))
            {
                return SlotDead;
            }

            return state;
        };

        SlotStates states;
        auto &entry = states.entry;
        auto exitOf = [&](mlir::Block *block) {
            auto state = entry.lookup(block);
            for (auto &op : *block)
            {
                state = step(&op, state);
            }

            return state;
        };

        entry[&body.front()] = SlotDead;
        llvm::SmallVector<mlir::Block *> work{&body.front()};
        while (!work.empty())
        {
            auto *block = work.pop_back_val();
            auto state = exitOf(block);
            for (auto *successor : block->getSuccessors())
            {
                auto &into = entry[successor];
                if ((into | state) != into)
                {
                    into |= state;
                    work.push_back(successor);
                }
            }
        }

        // ...and before each op of the body
        for (auto &block : body)
        {
            auto state = entry.lookup(&block);
            for (auto &op : block)
            {
                states.before[&op] = state;
                state = step(&op, state);
            }

            states.exit[&block] = state;
        }

        return states;
    }

    // Places a release of the slot on every edge into the join that holds the value and has not moved
    // it. False, with nothing placed, when an edge cannot take one, or there is none.
    bool unmovedEdges(mlir_ts::VariableOp varOp, const SlotStates &states, llvm::SmallVectorImpl<EdgeRelease> &placed,
                      llvm::SetVector<mlir::Block *> &releasedInto)
    {
        auto &body = getFunction().getFunctionBody();
        auto slot = varOp.getResult();
        auto &entry = states.entry;

        // the edges into the join that hold the value and did not move it
        auto handler = handlerBlocks();
        for (auto &block : body)
        {
            if (!entry.count(&block))
            {
                continue; // unreachable
            }

            auto state = states.exit.lookup(&block);
            for (auto index : llvm::seq<unsigned>(0, block.getNumSuccessors()))
            {
                auto into = entry.lookup(block.getSuccessor(index));
                if (!isJoined(into))
                {
                    continue;
                }

                // a join some path reaches with nothing in the slot is not this rule's
                if (into & SlotDead)
                {
                    return false;
                }

                // an edge that moved it, or from where the paths have joined already
                if (state != SlotLive)
                {
                    continue;
                }

                auto *terminator = block.getTerminator();
                if (!mlir::isa<mlir::cf::BranchOp, mlir::cf::CondBranchOp>(terminator) || handler.contains(&block) ||
                    !dominance->dominates(varOp.getOperation(), terminator))
                {
                    return false;
                }

                placed.push_back({slot, &block, static_cast<unsigned>(index), terminator->getLoc()});
                releasedInto.insert(block.getSuccessor(index));
            }
        }

        return !placed.empty();
    }

    // Whether nothing uses the value where the paths have joined, or after an edge released it.
    // False, with the use in `joinedUse` where there is one, when the release cannot be placed.
    bool noUseWhereJoined(mlir_ts::VariableOp varOp, mlir::Operation *retain, mlir::Operation *taker,
                          llvm::ArrayRef<mlir::Operation *> assignments, const SlotStates &states,
                          const llvm::SetVector<mlir::Block *> &releasedInto)
    {
        auto &body = getFunction().getFunctionBody();
        auto slot = varOp.getResult();

        auto joinedAt = [&](mlir::Operation *user) {
            auto *top = body.findAncestorOpInRegion(*user);
            return !top || isJoined(states.before.lookup(top));
        };

        // Nothing uses the value where the paths have joined - a read there, on the path that did
        // not move it, needs what the edge released - neither through the slot nor through anything
        // read out of it: a field, a view, a borrowing local, a closure. The move itself hands the
        // value to its new owner, whose reads are its own. A value the slot is given - its
        // initializer, an assignment's - is the slot's only once it is stored: a use before that (a
        // constructor's, or one around a loop before it is made again) is not a use of it.
        //
        // Past the joined points too: what was read out of the value before an edge released it
        // may not be used after that edge - after a new value is stored into the slot, or after its
        // scope ends - unless it was read out again (spec 30). A read of the slot after a new value
        // is stored reads that one.
        struct Origin
        {
            mlir::Operation *stored;
            mlir::Operation *made;
        };

        llvm::SmallVector<Origin> origins{{nullptr, nullptr}};
        llvm::SmallVector<std::pair<mlir::Value, unsigned>> values;
        llvm::DenseSet<std::pair<mlir::Value, unsigned>> seen;
        auto taint = [&](mlir::Value value, unsigned origin) {
            if (carriesPointer(value.getType()) && seen.insert({value, origin}).second)
            {
                values.push_back({value, origin});
            }
        };

        auto given = [&](mlir::Value value, mlir::Operation *stored) {
            origins.push_back({stored, value.getDefiningOp()});
            taint(value, origins.size() - 1);
        };

        taint(slot, 0);
        if (varOp.getInitializer())
        {
            given(varOp.getInitializer(), varOp);
        }

        for (auto *assignment : assignments)
        {
            given(mlir::cast<mlir_ts::StoreOp>(assignment).getValue(), assignment);
        }

        // reached from an edge that released the value, before the read it came from (or the op
        // that made it) runs again
        auto usedAfterRelease = [&](mlir::Operation *made, mlir::Operation *user) {
            llvm::SmallVector<mlir::Operation *, 1> madeAgain;
            if (made)
            {
                madeAgain.push_back(made);
            }

            return llvm::any_of(releasedInto, [&](mlir::Block *join) { return reachableFrom(join, user, madeAgain); });
        };

        auto heldAt = [&](unsigned origin, mlir::Operation *user) {
            auto &from = origins[origin];
            if (!from.stored)
            {
                return true;
            }

            llvm::SmallVector<mlir::Operation *, 1> madeAgain;
            if (from.made)
            {
                madeAgain.push_back(from.made);
            }

            return reachableAfter(from.stored, user, madeAgain);
        };

        while (!values.empty())
        {
            auto [value, origin] = values.pop_back_val();
            for (auto &use : value.getUses())
            {
                auto *user = use.getOwner();
                if (mlir::isa<mlir_ts::DebugVariableOp>(user) || user == retain)
                {
                    continue;
                }

                // the slot's own releases and assignments are decided above
                if (value == slot && (mlir::isa<mlir_ts::ReleaseSlotOp>(user) || llvm::is_contained(assignments, user)))
                {
                    continue;
                }

                if (joinedAt(user) && heldAt(origin, user))
                {
                    joinedUse = user;
                    return false;
                }

                if (value != slot && usedAfterRelease(origins[origin].made, user))
                {
                    joinedUse = user;
                    return false;
                }

                if (user == taker)
                {
                    continue;
                }

                // a store of it into a local makes every read of that local one; one into a place
                // is not followed, and the release is not placed
                if (auto storeOp = mlir::dyn_cast<mlir_ts::StoreOp>(user); storeOp && storeOp.getValue() == value)
                {
                    auto into = storeOp.getReference().getDefiningOp<mlir_ts::VariableOp>();
                    if (!into || into.getCaptured().value_or(false))
                    {
                        return false;
                    }

                    taint(into.getResult(), origin);
                    continue;
                }

                if (auto branch = mlir::dyn_cast<mlir::BranchOpInterface>(user))
                {
                    for (auto index : llvm::seq<unsigned>(0, user->getNumSuccessors()))
                    {
                        auto operands = branch.getSuccessorOperands(index);
                        for (auto [position, operand] : llvm::enumerate(operands.getForwardedOperands()))
                        {
                            if (operand == value)
                            {
                                taint(user->getSuccessor(index)->getArgument(operands.getProducedOperandCount() + position),
                                      origin);
                            }
                        }
                    }
                }

                // a call's result is its own block, unless its facts say it borrows this argument - a
                // plain borrow, or a borrowing record that holds it (spec 32)
                auto borrows = true;
                if (isCall(user))
                {
                    auto args = callArgs(user);
                    borrows = llvm::any_of(resultBorrowsAll(user), [&](int index) {
                        return static_cast<size_t>(index) < args.size() && args[index] == value;
                    });
                }

                // a read of the slot is what its uses came from
                auto from = origin;
                if (value == slot && user->getNumResults() > 0)
                {
                    origins.push_back({nullptr, user});
                    from = origins.size() - 1;
                }

                for (auto result : user->getResults())
                {
                    if (borrows)
                    {
                        taint(result, from);
                    }
                }

                // inside a region - a closure, an async body - it is the op that holds the region
                // that hands it on
                if (user->getParentRegion() != &body)
                {
                    for (auto result : body.findAncestorOpInRegion(*user)->getResults())
                    {
                        taint(result, origin);
                    }
                }
            }
        }

        return true;
    }

    // Puts in the releases placeOnUnmovedEdges decided: at the end of a block that goes only to the
    // join, else in a block of its own on the edge.
    void placeEdgeReleases()
    {
        for (auto &edge : edgeReleases)
        {
            auto *terminator = edge.from->getTerminator();
            mlir::OpBuilder builder(terminator);
            if (terminator->getNumSuccessors() == 1)
            {
                builder.create<mlir_ts::ReleaseSlotOp>(edge.loc, edge.slot);
                continue;
            }

            auto *join = terminator->getSuccessor(edge.successor);
            auto operands = mlir::cast<mlir::BranchOpInterface>(terminator).getSuccessorOperands(edge.successor);
            llvm::SmallVector<mlir::Value> forwarded(operands.getForwardedOperands());
            auto *split = new mlir::Block();
            split->insertAfter(edge.from);
            builder.setInsertionPointToEnd(split);
            builder.create<mlir_ts::ReleaseSlotOp>(edge.loc, edge.slot);
            builder.create<mlir::cf::BranchOp>(edge.loc, join, forwarded);
            if (!forwarded.empty())
            {
                operands.erase(0, forwarded.size());
            }

            terminator->setSuccessor(split, edge.successor);
        }

        edgeReleases.clear();
    }

    // The stores of the assignments to an owning local: rc releases what the slot holds right
    // before it stores the new value (an assignment, and a `var` declaration met again).
    static llvm::SmallVector<mlir::Operation *> slotAssignments(mlir::Value slot)
    {
        llvm::SmallVector<mlir::Operation *> assignments;
        for (auto *user : slot.getUsers())
        {
            auto storeOp = mlir::dyn_cast<mlir_ts::StoreOp>(user);
            auto releaseOp = storeOp ? mlir::dyn_cast_or_null<mlir_ts::ReleaseSlotOp>(storeOp->getPrevNode())
                                     : mlir_ts::ReleaseSlotOp();
            if (releaseOp && storeOp.getReference() == slot && releaseOp.getSlot() == slot)
            {
                assignments.push_back(storeOp);
            }
        }

        return assignments;
    }

    // A use of what `read` returned - directly or through casts - that the move at `taker` can
    // reach, other than the move's own retain and taker. None if there is none. A path back
    // through `read` itself reaches a use of a value read again: the read, met after the move,
    // is the slot's own use to check (around a loop, of a value an assignment stored).
    // `isRead` accepts a use that reads the value through its new owner (spec 29).
    mlir::Operation *readUsedAfter(mlir_ts::LoadOp read, mlir::Operation *retain, mlir::Operation *taker,
                                   mlir::Operation *kill,
                                   llvm::function_ref<bool(mlir::Operation *, mlir::Value)> isRead = nullptr)
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

                llvm::SmallVector<mlir::Operation *, 2> kills{read.getOperation()};
                if (kill)
                {
                    kills.push_back(kill);
                }

                if (reachableAfter(taker, user, kills) && !(isRead && isRead(user, current)))
                {
                    return user;
                }
            }
        }

        return nullptr;
    }

    // ---- Reads after a move, through the new owner (spec 29) ----
    //
    // `h.c = c; print(c.x)`, `best = v; s += v.x`: after the move the value is the new owner's, and
    // a read of it through the old name is a borrow of the place it moved into, as a read of
    // `h.c` would be. It is bounded the way checkPlaceRead bounds such a read - by the ends of the
    // place's roots and by the drops that may destroy what the place holds (an overwrite, through an
    // interface view too; a call whose facts may drop; a call given a closure) - on a path from the
    // move to the read on which the value is not made again or moved again.

    // Every use of `value`, through its views, is a read `isRead` accepts. A retain - rc's reference
    // for an owner - is not.
    static bool onlyReadThroughOwner(mlir::Value value, llvm::function_ref<bool(mlir::Operation *, mlir::Value)> isRead)
    {
        auto all = true;
        forEachUse(value, [&](mlir::Operation *user, mlir::Value used) {
            all = all && (mlir::isa<mlir_ts::DebugVariableOp>(user) ||
                          (!mlir::isa<mlir_ts::RetainOp>(user) && isRead(user, used)));
        });

        return all;
    }

    // A use that only reads the value: a field, element or length read through it, a method reached
    // through it, a comparison or arithmetic, a print, an assert, a test, or an argument the callee
    // borrows. Anything else - a store, an owning `let`, a keeping call, a return - would be a second
    // owner.
    static bool isReadAfterMove(mlir::Operation *user, mlir::Value value)
    {
        if (mlir::isa<mlir_ts::PropertyRefOp, mlir_ts::ElementRefOp, mlir_ts::LengthOfOp, mlir_ts::ThisSymbolRefOp,
                      mlir_ts::VirtualSymbolRefOp, mlir_ts::ThisVirtualSymbolRefOp, mlir_ts::InterfaceSymbolRefOp,
                      mlir_ts::GetThisOp, mlir_ts::GetMethodOp, mlir_ts::ArithmeticBinaryOp, mlir_ts::LogicalBinaryOp,
                      mlir_ts::PrintOp, mlir_ts::AssertOp, mlir_ts::HasValueOp, mlir_ts::StringLengthOp>(user))
        {
            return isBorrow(user, value);
        }

        return isCall(user) && !isKeptArgument(user, value);
    }

    // The place a taker moves its value into, as a chain checkPlaceRead would read it: an owning
    // local (a declaration, or an assignment), or a field or element - of a local or a parameter
    // (`this`). False for anything else: a call that keeps it, a push, a return, a global, a
    // captured variable, a handle's payload.
    bool destinationChain(mlir::Operation *taker, Chain &chain)
    {
        auto ownSlot = [](mlir_ts::VariableOp varOp) {
            return varOp && isOwningVariable(varOp) && !varOp.getCaptured().value_or(false);
        };

        if (auto varOp = mlir::dyn_cast<mlir_ts::VariableOp>(taker))
        {
            if (!ownSlot(varOp))
            {
                return false;
            }

            chain.roots.push_back({varOp.getResult(), Chain::Slot});
            return true;
        }

        auto storeOp = mlir::dyn_cast<mlir_ts::StoreOp>(taker);
        if (!storeOp)
        {
            return false;
        }

        auto ref = storeOp.getReference();
        if (auto varOp = ref.getDefiningOp<mlir_ts::VariableOp>())
        {
            if (!ownSlot(varOp))
            {
                return false;
            }

            chain.roots.push_back({ref, Chain::Slot});
            return true;
        }

        if (!isPlace(ref))
        {
            return false;
        }

        chain = chainOf(nullptr, ref);
        if (!chain.handles.empty() || chain.roots.empty())
        {
            return false;
        }

        // a global's field, or a captured variable's: not a place of this function
        return llvm::none_of(chain.roots, [](auto &root) {
            auto loadOp = root.first.template getDefiningOp<mlir_ts::LoadOp>();
            return loadOp && (loadOp.getReference().template getDefiningOp<mlir_ts::AddressOfOp>() ||
                              isLoadedCell(loadOp.getReference()));
        });
    }

    // Is `use`, a use of `value` after `taker` moved it, a read through the place it moved into
    // (spec 29)? `made` is where the value is made again (its definition, the local's declaration
    // and assignments). Sound when no path from the move through an end or a drop of the place to
    // the read - or to anything the read produced - makes the value again or moves it again: such a
    // path is the only way the place can stop holding the value before the read.
    bool readsThroughOwner(mlir::Operation *taker, mlir::Value value, mlir::Operation *use,
                           llvm::ArrayRef<mlir::Operation *> made)
    {
        if (!isReadAfterMove(use, value))
        {
            return false;
        }

        Chain chain;
        if (!destinationChain(taker, chain))
        {
            return false;
        }

        // A read again of the old name after a drop still finds the moved value - the slot is not
        // emptied by a move - so neither the read nor the slot's read is where it is made again.
        llvm::SmallVector<mlir::Operation *> kills(made.begin(), made.end());
        kills.push_back(taker);
        llvm::SmallVector<BorrowUse> uses;
        uses.push_back({use, {kills.begin(), kills.end()}});
        for (auto result : use->getResults())
        {
            if (!mayPointInto(result))
            {
                continue;
            }

            // what the read produced - a field's reference, a bound method - still points into the
            // block; a call's result that borrows its argument is the value again and may not be kept
            if (walkBorrowed(result, kills, uses, nullptr, Keeping{nullptr, false}, false, !!borrowingCall(result)))
            {
                return false;
            }
        }

        // a method reached through it whose callee keeps its `this`: a second owner
        for (auto &borrowUse : uses)
        {
            if (!isCall(borrowUse.op))
            {
                continue;
            }

            auto args = callArgs(borrowUse.op);
            for (auto index : ownedParams(borrowUse.op))
            {
                if (static_cast<size_t>(index) < args.size() &&
                    mlir::isa_and_nonnull<mlir_ts::GetThisOp, mlir_ts::ExtractInterfaceThisOp>(args[index].getDefiningOp()))
                {
                    return false;
                }
            }
        }

        auto outlives = [&](mlir::Operation *end) {
            if (end == taker || !reachableAfter(taker, end, made))
            {
                return false;
            }

            return llvm::any_of(uses, [&](BorrowUse &borrowUse) {
                return (borrowUse.op != end || !borrowUse.callee) && reachableAfter(end, borrowUse.op, borrowUse.kills);
            });
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
            if (dropsChain(drop, chain, nullptr, value) && outlives(drop))
            {
                return false;
            }
        }

        return true;
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

    void reportMovedOnSomePaths(mlir::Operation *move, mlir::Operation *release, llvm::StringRef name,
                                mlir::Operation *usedAt = nullptr)
    {
        if (!reporting())
        {
            return;
        }

        auto diag = move->emitError("'") << name
                                         << "' is moved here on some paths only; -mm=own cannot release it "
                                            "on the others yet";
        diag.attachNote(release->getLoc()) << "released here on every path";
        if (usedAt)
        {
            diag.attachNote(usedAt->getLoc())
                << "used here, after the paths join, so it cannot be released before they do";
        }

        signalPassFailure();
    }
};

} // end anonymous namespace

#undef DEBUG_TYPE

std::unique_ptr<mlir::Pass> mlir_ts::createOwnershipInferencePass()
{
    return std::make_unique<OwnershipInferencePass>();
}
