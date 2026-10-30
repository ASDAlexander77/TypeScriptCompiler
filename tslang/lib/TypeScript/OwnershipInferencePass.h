// OwnershipInferencePass, the per-function inference of -mm=own. Its methods are defined across
// the lib/TypeScript/OwnershipInference*.cpp files, one analysis each; templates stay here.

#ifndef MLIR_TYPESCRIPT_OWNERSHIPINFERENCEPASS_H
#define MLIR_TYPESCRIPT_OWNERSHIPINFERENCEPASS_H

#include "mlir/Pass/Pass.h"
#include "mlir/IR/Dominance.h"
#include "mlir/Dialect/LLVMIR/LLVMAttrs.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
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

namespace mlir_ts = mlir::typescript;

namespace own_inference
{

using namespace own_facts;

// This pass's own mark, for its own reading: a store of a value into the capture box of a closure
// that borrows what it captures (phase 5). The store is a read, not a taker.
#define OWN_CAPTURE_BORROW_ATTR_NAME "__own_capture_borrow"
// -mm=own only, on a `ts.CreateArray` given to one call that borrows it: it borrows its elements
// for the call (classifyArgumentArrays)
#define OWN_BORROWING_ARRAY_ATTR_NAME "__own_borrowing_array"

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

    void runOnOperation() override;

    mlir::FunctionOpInterface getFunction();

    void runOnFunction();

    // A read through an interface's field that a getter may have made (OWN_CELL_READ_ATTR_NAME), whose value and
    // what is borrowed from it stay in its block: a release after their last use, for the getter's path only. One
    // nothing uses - what an assignment `i.f = v` leaves of the place it wrote - goes.
    void releaseCellReads();

    // The analysis of one function; true when it passed. In a dry run it erases nothing and strips
    // no facts.
    bool analyze(llvm::SetVector<mlir::Operation *> &toErase);

    // Each analysis starts from nothing this function has not set up again.
    void resetFunctionState();

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
    void collectOwnershipOps(FunctionOps &ops, llvm::SetVector<mlir::Operation *> &toErase);

    // The owner check of each value, then of each read out of a place and of a mortal global. The
    // fresh values proven, whose retains go.
    llvm::DenseSet<mlir::Value> checkOwnershipOps(const FunctionOps &ops, llvm::SetVector<mlir::Operation *> &toErase);

    // Each retain: erased, given to the receivers of an owning local or of a kept parameter, or
    // reported.
    void decideRetains(const FunctionOps &ops, const llvm::DenseSet<mlir::Value> &proven, Receivers &slotReceivers,
                       Receivers &paramReceivers, llvm::SetVector<mlir::Operation *> &toErase);

    // The arguments a callee keeps, given to the receivers of the local or kept parameter they come
    // from.
    void giveKeptArguments(const FunctionOps &ops, Receivers &slotReceivers, Receivers &paramReceivers);

    // The facts were for this pass only.
    void stripFacts();

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
    bool reporting();

    // `op->setAttr(name, unit)`, undone after the dry run when the op did not have it
    void setAttrTracked(mlir::Operation *op, llvm::StringRef name);

    // Strings as values (spec 22.3): each string retain the dry run did not erase is rewritten, where
    // its one use can be found, into a copy for that use.
    void makeStringCopies(const llvm::SetVector<mlir::Operation *> &erased);

    // The value a parameter of this function is, when its callers were told this function keeps it
    // or returns a borrow of it (spec 15): copying it would leak what the caller gave up.
    bool borrowsKnownParam(mlir::Value value);

    // Rewrites one string retain into a copy for the use it was made for (spec 22.3), and erases it;
    // false, leaving it for the real run to report, where there is no single such use in its block,
    // or the value is not a string or is a known parameter's (spec 22.4). Its use walk is takerOf's
    // without the twinReadAfter fallback: a retain whose value has no use of its own is not copied,
    // so its error stays.
    bool copyForRetain(mlir::Operation *retain);
    // May what this local holds be what the function returns - read and returned, or handed to
    // another local that may be?
    bool mayBeReturned(mlir_ts::VariableOp varOp);
    // A retain of a tuple `ts.CreateTuple` made from borrowed strings and handles (`[s, shared]`): the
    // strings are copied into it instead, each handle is retained alone, and the retain goes.
    bool copyIntoTuple(mlir_ts::RetainOp retainOp);

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
    // the capture boxes of object literals that borrow what they capture (classifyLiteralCaptures):
    // `uses` only, what reaches the literal
    llvm::SmallVector<std::shared_ptr<Closure>> literalBoxes;
    // a copy stored into a borrowing box -> its closure
    llvm::DenseMap<mlir::Operation *, Closure *> captureStores;
    // rc's references for boxes, which the closures decide
    llvm::DenseSet<mlir::Operation *> captureRetains;
    // A string read out of a place, given to a call that may release it (the dry run's): the call
    // gets a copy instead, released after it (spec 22), read -> call; or stored into a place, which
    // owns the copy (spec 50), read -> store
    llvm::SmallVector<std::pair<mlir::Operation *, mlir::Operation *>> callCopies;
    // A move out of a field of an owning record local, being checked (spec 48): its retain and its
    // taker, which walkBorrowed takes for uses of the borrow and not for keeps
    llvm::SmallPtrSet<mlir::Operation *, 2> takenOps;
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
    void classifyClosures();
    bool isBackReference(Closure &closure);
    // An object literal's `.captured` box (its arrows' captures, spec 67): when the literal stays in
    // this function, the box borrows the copies it holds, as a closure that does not escape does
    void classifyLiteralCaptures();

    // The fields of a generator's state object of this type that borrow from its arguments
    // (OWN_BORROWING_FIELDS_ATTR_NAME, the signature pass's), the `.captured` box's first.
    llvm::SmallVector<int32_t> borrowingFieldsOf(mlir::Type recordType);

    // Is this the state object of a generator that says `@keeps` - one that owns its arguments
    // (spec 38; -1 in place of the box in borrowingFieldsOf)?
    bool ownsKeptState(mlir::Type recordType);

    // Is `ref` a field of a generator's state object that only yields (OWN_YIELDED_FIELDS_ATTR_NAME)?
    bool isYieldedField(mlir::Value ref);

    // Is `ref` a field of a generator's state object that borrows (other than the `.captured` box)?
    bool isBorrowingField(mlir::Value ref);

    // A store into a borrowing field holds what it stores without owning it, as a borrowing box's
    // copy does (OWN_CAPTURE_BORROW_ATTR_NAME), and rc's retain for it goes; its release, when the
    // field is overwritten, gives nothing back.
    void classifyBorrowingFields();
    // The arrays a rest parameter is given that only borrow their elements for the call (spec 49):
    // rc's references for the elements go, and the array is released after the call without them.
    void classifyArgumentArrays();

    // Is `ref` a cell read out of the `.captured` box of a generator's state object that borrows:
    // `ts.Load(ts.PropertyRef(ts.Load(ts.PropertyRef(state, captured)), i))`?
    bool isBorrowingCell(mlir::Value ref);

    // Is `ref` a field of the `.captured` box of a generator's state object that borrows:
    // `ts.PropertyRef(ts.Load(ts.PropertyRef(state, captured)), i)`? A field copied in by value holds
    // the caller's value itself, a cell field the caller's cell.
    bool isBorrowingBoxField(mlir::Value ref);

    // What a generator reads out of a borrowing cell or field is the caller's: every use of it must
    // be a borrow - a store into another borrowing field is one (classifyBorrowingFields). A yield of
    // it, a push, a return, a store anywhere else would give the caller's value a second owner, and
    // rc retains nothing for some of these, so nothing else would see it.
    void checkBorrowedReads();

    // A call whose result borrows from its arguments (OWN_RESULT_BOUNDED_ATTR_NAME): a generator over
    // a parameter's block. The result is owned as any fresh value is, but nothing may use it after
    // one of those arguments is given back - released, assigned, or dropped from the place it was
    // read from - and it may not be kept anywhere that could outlive them. Its releases are not
    // uses: the state object's release gives back nothing it borrows.
    void checkBoundedResults();

    // An element taken out (`pop`, `shift`) of a rest parameter the function does not keep (spec 72):
    // a caller's array may only borrow the elements it gives, so what comes out is the caller's. A
    // function that keeps what it takes keeps the parameter, and its callers give the elements up.
    void checkRestTakes();

    // A call to, or a reference to, a function made for an element type it cannot give a second owner
    // (`@sharesElements`, OWN_CANNOT_SHARE_ATTR_NAME, spec 73): `slice` of an Array<Item>.
    void checkCannotShare();

    // The ends of a capture box given to a call with a bounded result: each release of a closure on
    // it (the box goes with its last), and each end of what a field of it was filled from. Reports
    // and returns false when a use of the result comes after one.
    bool checkBoundedByBox(mlir::Operation *call, mlir::Value box, llvm::ArrayRef<mlir::Operation *> uses,
                           llvm::StringRef name);

    // Every use of a bounded result, through its views, the locals that hold it, and the function
    // and object a call through it is made with. Answers the first use that keeps it - anything
    // that is not a read, a call, a local, or rc's bookkeeping - or null.
    mlir::Operation *boundedUses(mlir::Value result, llvm::SmallVectorImpl<mlir::Operation *> &uses);

    // A generator's maker puts what the generator captures - its parameters, an outer local - into
    // a box, and the box into its state object's `.captured` field: the box of a closure that
    // escapes, with the state object, at that store. There is no closure value, so no use runs it
    // here and nothing calls through it. rc's `ts.RetainCell` of the box is the state object's
    // reference, which moves in with it. A box used any other way is left to the RetainCell error.
    void classifyStateBoxes();

    // `ts.Store(box, ts.PropertyRef(state, i))`, where `state` is a local a `ts.Constant`
    // initializes whose every read fills a block `ts.New` made: a generator's initial state.
    static bool isStoreIntoState(mlir::Operation *user, mlir::Value box);

    // rc's reference for the box's copy of `value`: the retain of it nearest before the store that
    // fills the box, in the same block.
    mlir::Operation *retainBefore(mlir::Value value, mlir::Operation *store);

    // Where a closure value goes: through its views and the owning `let`s that hold it, each use
    // that may run it joins `uses`, and each release `ends`. Answers the first use that takes it
    // anywhere else - returned, stored, kept by a callee, captured - or null. Anything this does
    // not know is such a use.
    mlir::Operation *followClosure(mlir::Value closure, llvm::SmallVectorImpl<mlir::Operation *> &uses,
                                   llvm::SmallVectorImpl<mlir::Operation *> &ends, bool &aliased);

    // Does a call given this value give it a closure, which may run and touch what its box holds?
    static bool holdsCaptures(mlir::Value value);

    // The verdicts on the closures over cells: an escape is an error; a borrow must not be run after
    // anything it borrows is given back - a cell this function owns at its releases, a copy at its
    // owner's ends. A cell an enclosing closure holds and a parameter end nowhere here; a copy of a
    // read out of a container is bounded by the read's own check, which sees the closure's uses
    // (walkBorrowed).
    void decideClosures(llvm::SetVector<mlir::Operation *> &toErase);

    // Is this the store that moves a cell into the box of a closure that escapes?
    bool movesCellAway(mlir::Operation *op);

    // A closure that escapes owns what its box holds (spec 2.5, phase 5b). Its copies moved in as
    // in phase 4. Each cell moves in where the box takes it: that store is the move, rc's
    // `ts.RetainCell` for it goes, and so does each of the frame's `ts.ReleaseCell`s the move reaches,
    // which it must dominate. Nothing in the frame may use the variable after the move - read it,
    // assign it, capture it again - and the move may not come round a loop to a cell made outside
    // it. A cell that borrows its value (a parameter's) or one an enclosing closure holds has
    // nothing to move. And nothing may call through the box once the closure has gone.
    void decideEscaping(Closure &closure, llvm::SetVector<mlir::Operation *> &toErase);

    // A captured variable whose value this function does not own - a parameter, `this` - has a cell
    // that borrows the value: the cell's release frees the cell only (OWN_CELL_BORROWS_ATTR_NAME),
    // and nothing may assign the variable, here or in a closure over it, since the old value is the
    // caller's. A variable whose value nobody can be said to own is an error.
    void decideCells();

    // An argument read out of a local is lent to the call, but the arguments are read in order: a
    // later one may give the local a new value before the call takes the first - `g(n, (n = mk(), 0))`,
    // in a conditional argument too - or call a closure that assigns it, when it is captured. The
    // assignment drops the block the argument is, and the callee reads freed memory. rc holds such an
    // argument across the call instead (OwnedReturnConsumptionPass, holdGlobalArguments). A read of a
    // global is checkGlobalRead's, and a store into a place in the local the place borrows' (a field
    // read `g(n.v, (n.v = mk(), 0))`); a store into the local's slot ends `n.v` too.
    void checkArgumentsAssigned();

    // Does a closure built over this captured local assign it (OWN_ASSIGNS_CAPTURES_ATTR_NAME)? Yes too
    // when its cell goes where this does not follow.
    bool closureAssigns(mlir_ts::VariableOp local);

    template <typename F> bool quietly(F check)
    {
        ++quiet;
        auto result = check();
        --quiet;
        return result;
    }

    // Move, else borrow, else error (spec 2.2) for the receivers of one owning local.
    void decideSlot(mlir::Value slot, llvm::ArrayRef<SlotReceiver> receivers, llvm::SetVector<mlir::Operation *> &toErase);

    void decideSlotReceiver(mlir::Value slot, const SlotReceiver &receiver, llvm::StringRef owner,
                            llvm::SetVector<mlir::Operation *> &toErase, bool sole);

    // Receivers of one owning local that each read a value of its own (#528): `h.c = a; a = new C();
    // g.c = a`. An assignment comes between every two of their reads on every path from one to the
    // other, so each is decided alone, as the one receiver of the value it reads. So are receivers
    // no path joins at all, assigned or not: `if (c) return a; return a;` moves `a` once on each
    // way out.
    bool takeValuesOfTheirOwn(mlir::Value slot, llvm::ArrayRef<SlotReceiver> receivers);

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
    mlir_ts::VariableOp parameterSlot(int32_t index);

    // Is `slot` a parameter this function keeps and releases where it does not move it?
    bool isReleasedParam(mlir::Value slot);

    // Does `value` come from a read of `slot`: computed from one, through any op?
    static bool readsSlot(mlir::Value value, mlir::Value slot, int depth = 0);

    void releaseKeptParamsAtExits();

    // A parameter this function keeps (`__own_params`): its caller moved the argument in, and this
    // function has no release for it, so exactly one taker must move it on, on every way out of
    // the function, and nothing may read it after that.
    void decideParam(mlir::Value slot, llvm::ArrayRef<SlotReceiver> receivers, llvm::SetVector<mlir::Operation *> &toErase);

    // The thrown value, read out of the exception object by a catch's copy thunk (`.eh.copy.*`,
    // which the C++ runtime calls to copy the exception into the catch). rc retains it for the
    // copy; own moves it: the exception object never gives its reference back (a throw hands the
    // thrower's reference to it, and nothing releases it - rc leaks it the same way), and each
    // exception is copied into one catch, since a TypeScript rethrow is a new throw.
    bool isReadOutOfException(mlir::Value value);

    // The read of a parameter this function keeps, through its views. None for anything else.
    mlir_ts::LoadOp keptParamLoadOf(mlir::Value value);

    // More than one receiver of one owning local. Each has to borrow: a borrow pins the owner, so
    // none may move it. A receiver that is not a `let` reports its move error - another
    // receiver's reads come after it - or, if its move alone would pass, that a borrow pins it.
    void decideSlotReceivers(mlir::Value slot, llvm::ArrayRef<SlotReceiver> receivers, llvm::StringRef owner,
                             llvm::SetVector<mlir::Operation *> &toErase);

    // A `let` that could borrow: owning, declared from a value, holding a reference it took
    // itself (a `ts.RetainSlot`, not the value's own consumed one), and not captured. With
    // `consumed`, also one that took over the reference rc thought the value carried: a call's
    // result that turned out to borrow an argument carries none.
    static mlir_ts::VariableOp borrowerOf(mlir::Operation *acquisition, bool consumed = false);

    // tryBorrow for the `let` an acquisition gives a reference: a `ts.Retain` of the value a
    // consuming `let` takes over is that `let`'s, and goes with the borrow; a `ts.RetainSlot` is the
    // `let`'s own bookkeeping already.
    bool borrowWith(mlir_ts::VariableOp borrower, mlir::Operation *acquisition, llvm::ArrayRef<mlir::Operation *> ends,
                    llvm::StringRef owner, llvm::SetVector<mlir::Operation *> &toErase);

    // The owning `let` declared from `value` that takes over its reference, or none.
    static mlir_ts::VariableOp consumingLetOf(mlir::Value value);

    // Where an owning local stops holding its value: each release of its slot and each store
    // into it.
    static llvm::SmallVector<mlir::Operation *> slotEnds(mlir::Value slot);

    // A borrow (spec 2.2, verdict 2): `borrower` takes no reference and gives none back, so its
    // retain and every release of its slot go, and the owner keeps its own. Sound while no use of
    // the borrower can run after any of `ends` - the owner's releases and assignments - and while
    // nothing it holds is kept: not stored, returned, retained or assigned. A path back to a use
    // through the borrower's own declaration is a new borrow, not this one. Reports and returns
    // false otherwise.
    bool tryBorrow(mlir_ts::VariableOp borrower, llvm::ArrayRef<mlir::Operation *> ends, llvm::StringRef owner,
                   llvm::SetVector<mlir::Operation *> &toErase);

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
                                         llvm::SmallVectorImpl<mlir::Operation *> &bookkeeping);

    static bool isAssignmentOf(mlir::Operation *op, mlir_ts::VariableOp varOp);

    // What `use` hands the very value it uses on to, as that value: a cast or a view of it, the
    // argument of the block a branch passes it to (a merge: `a ?? b`) or the `arith.select` that
    // chooses it (the merge folded: `c ? h : h2`), or the reads of a local
    // that owns nothing and is assigned it (`for (x of arr)` into an outer `let x`). Answers
    // whether the use is one of those, even where nothing comes out of it. Where the value lands
    // somewhere that is replaced - a block argument, a local - `renewedAt` gets what replaces it:
    // the block's entry, the local's declaration (when it has an initializer: `let l!: V` stores
    // nothing, and in a loop the slot keeps the last round's value) and assignments.
    static bool passesOn(mlir::OpOperand &use, llvm::SmallVectorImpl<mlir::Value> &values,
                         llvm::SmallVectorImpl<mlir::Operation *> *renewedAt = nullptr);

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
        Keeping keeping = Keeping{nullptr, false}, bool countedHandle = false, bool startBorrowed = true);

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
    static bool callResultIsOwn(mlir::Operation *call);

    // Is this what an indirect call goes through: its callee value, or, for a method read out of
    // a field of an object (`o.m()`, `m` a field holding a function), the object it is bound to,
    // which is the container of what was read, not what was borrowed. A method of a class bound
    // to a borrowed instance (`h.c.m()`) is not one: its `this` is the borrow.
    static bool isCalleeOperand(mlir::OpOperand &use);

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
        // null for an unboxed record (spec 36), whose `reads` of `local` are the record
        mlir_ts::NewOp box;
        mlir_ts::VariableOp local;
        llvm::SmallVector<mlir::Value> interfaces;
        llvm::SmallVector<mlir::Value> reads;
        llvm::SmallVector<BorrowedField> borrowed;
        RecordWalk walk;
    };

    llvm::SmallVector<std::shared_ptr<BorrowingRecord>> records;
    // a store of a place read into a borrowing record's field -> the record's uses, which are the
    // read's uses too (walkBorrowed)
    llvm::DenseMap<mlir::Operation *, BorrowingRecord *> recordStoreUses;

    // Can a value of this type point into a heap block - a reference into one, a block of its own,
    // or a method bound to one? A number or a boolean read out of a borrowed block cannot.
    bool mayPointInto(mlir::Value value);

    // The borrowing record built in `varOp` (classifyRecords), or null.
    BorrowingRecord *borrowingRecordOf(mlir_ts::VariableOp varOp);

    // The boxes the signature pass marked as borrowing records, where every field still holds what it
    // was marked for: a borrowed field a value this pass bounds (recordFieldSource), any other one a
    // fresh value or nothing. A store into a borrowed field takes nothing, and whatever uses the record
    // reads what it stored (walkBorrowed, through recordStoreUses). A box that is not what it was
    // marked for is left to the record's retain, which is then an error.
    void classifyRecords();

    // Every use of a borrowing record (spec 32), and of what is read out of it - each may read what
    // the record borrows - through its views, merges, and the locals that hold it, made again at
    // `kills`. Its own retains and releases are the box's, which its own check decides. The record
    // itself is `kept` when it is stored anywhere but a local, given to a callee that keeps it, or
    // captured; returned, it is in `returns`. A call given it hands back its own block, unless that
    // is a borrowing record that holds it, whose uses are then this one's as well.
    void walkRecord(llvm::ArrayRef<mlir::Value> starts, llvm::ArrayRef<mlir::Operation *> kills, RecordWalk &walk);

    // The name the record goes by: the local that holds it, or a folded `const`'s.
    llvm::StringRef recordName(mlir::Value value);

    // What a borrowed field holds, for an error: `'ray'`, `'this.pos'`, `'livec'`, ...
    std::string describeRecordSource(BorrowedField field);

    // A record this function makes (spec 32): kept, it is an error; returned, each field it borrows
    // must be a parameter's that this function's facts lend (`__own_result_record`), so that its
    // callers bound it; and no use of it may come after what a field borrows may be released or
    // overwritten - an owning local's ends, and whatever may assign a captured variable. A parameter
    // and `this` are the caller's for the whole call; a read out of a place has the record's uses as
    // its own (checkPlaceRead).
    void checkBorrowingRecords();

    // ---- Unboxed records (spec 36) ----
    //
    // A tuple that holds borrows and owns nothing (OWN_RECORD_UNBOXED_FIELDS_ATTR_NAME where it is made,
    // OWN_RESULT_RECORD_UNBOXED_ATTR_NAME on a call that returns one) is a record the way a box is - its
    // uses bounded by what it borrows, moved into the locals that hold it - but there is nothing to give
    // back: rc's retains and releases of it, and of the locals that hold only such records (or nothing
    // at all), are erased, whatever the moves decided.
    enum class UnboxedFlow
    {
        None,    // no unboxed record
        Unboxed, // unboxed records only, or values that hold no block
        Mixed    // an unboxed record beside a value that owns a block
    };

    // What `value` is, through its views, merges and the owning locals that hold it.
    UnboxedFlow unboxedRecordFlow(mlir::Value value);

    // ...and what an owning local holds: its initializer and every value stored into it.
    UnboxedFlow unboxedRecordSlotFlow(mlir_ts::VariableOp varOp);

    // Erases the counting of every unboxed record; a local or a merge that holds one beside a value it
    // owns is an error (its release could not tell them apart).
    void dropUnboxedRecordCounts(llvm::SetVector<mlir::Operation *> &toErase);

    // A call whose result is a borrowing record (OWN_RESULT_RECORD_ATTR_NAME, spec 32): the box is
    // this function's to own - it is fresh, and its own check decides its moves and releases - but
    // it may hold a borrow of each argument the facts name, as a call's result that borrows them
    // does (checkPlaceRead): somewhere under each. So no use of it, through the locals it moves
    // into, may come after one of those arguments may be released or overwritten, and it may not be
    // kept. Returned, the arguments must be parameters this function's own facts lend. Its releases
    // are not uses: they free only what the record owns.
    void checkRecordResults();

    // ---- Reads out of containers (spec 2.3, phase 3) ----
    //
    // `h.c` and `arr[i]` read a value the container owns. Such a read is a borrow: it may not be
    // kept, and none of its uses - nor any use of what it produces - may run after something that
    // may destroy it (spec 2.2's dropping mutations): an overwrite of a place on the way from the
    // read up to its root, an array op that removes elements, an end of the root itself, or a call
    // that may reach the root. A path back to a use through the read itself is a new borrow.

    // A read of a heap value out of a field or an element, or a call whose result borrows an
    // argument (spec 2.4, phase 4), seen through its views. Null for anything else.
    mlir::Operation *placeReadOf(mlir::Value value);

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

    static HandleReadUses handleReadUses(mlir::Value value);

    // A handle read out of a place that rc gives a count of its own is a counted copy, not a borrow
    // (spec 23.3) - but only when every use of it is one rc counts (Ruling 7), and a use that is
    // not known to be one is not (Ruling 9). A read with one use that reads through the place as
    // well (`f(p, q, q.child = p.child)`), or that keeps it with no count (`[(q.child = p.child),
    // 1]`), is a borrow, and that use is tenure-checked; the counted uses are copies
    // (checkPlaceRead).
    static bool isCountedHandleRead(mlir::Value value);

    // A use that gives a handle an owner rc counts, with rc's retain for that very use: rc's
    // retain itself, a store into a field, an element, a global, an owning local or the result
    // slot (a return) that rc retained for (countedStoreRetain); a push or an unshift, `new
    // Shared`, an array literal's element, an `any` box, a `ts.ReturnVal`, each right after rc's
    // retain of what it takes (retainedRightBefore); or the initializer of an owning local, which
    // takes a count of its own (`ts.RetainSlot`). Any of them without its retain is not one: it
    // keeps the read without a count (`{ s: (q.child = p.child) }`'s field).
    static bool isCountedHandleReceiver(mlir::Operation *user, mlir::Value value);

    // A use that passesOn hands on but that takes the value: a return's store into the function's
    // result slot (isResultSlot), and a cast that boxes it into `any`.
    static bool isHandOff(mlir::Operation *user, mlir::Value value);

    // A return or an `any` box of a handle that rc counts (walkBorrowed).
    static bool isCountedHandOff(mlir::Operation *user, mlir::Value value);

    // rc's retains for what `op` takes: MLIRGen puts them in a row right before it
    // (mlirGenRetainCaptured, retainInsertedElements), one for each operand it does not take
    // over. Answers whether that row retains `value` once for each time `operands` hold it. A
    // retain is right before one op only, so no other receiver can count it as well.
    static bool retainedRightBefore(mlir::Operation *op, mlir::ValueRange operands, mlir::Value value);

    // The releases of the old value that rc's counted stores of a handle read make: `ts.Retain` of
    // the value, `ts.ReleaseSlot` of the slot, `ts.Store` of the value into it, in a row. Seen
    // through the read's casts, merges and locals that own nothing, as handleReadUses sees it.
    static void collectCountedStoreReleases(mlir::Value value, llvm::SmallPtrSetImpl<mlir::Operation *> &releases);

    // rc's retain for a store of a value, when it made one: `ts.Retain` of the value right before
    // the store, or before the `ts.ReleaseSlot` of the old value right before it -
    // `Retain(v)`, [`ReleaseSlot(slot)`], `Store(v, slot)`. A return's store into the result
    // slot comes after the scope exit's releases of the frame's locals and their slots:
    // `Retain(v)`, [`Release(..)`, `ReleaseSlot(..)`...], `Store(v, result)`. Null for a store rc
    // counts nothing for.
    static mlir::Operation *countedStoreRetain(mlir_ts::StoreOp storeOp);

    // The call a value is the result of, when that result is a plain borrow of an argument. A borrowing
    // record (spec 32) is not one: the caller owns the box (checkRecordResults).
    static mlir::Operation *borrowingCall(mlir::Value value);

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
        static constexpr int64_t InterfaceField = OWN_INTERFACE_FIELD_POSITION;
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
    Chain chainOf(mlir::Operation *read, mlir::Value place = {});

    // Where the roots stop holding what the chain reads: a local's releases and assignments and
    // every declaration or retain that may move it away; a made value's releases and every use
    // that takes it; a global's assignments.
    llvm::SmallVector<mlir::Operation *> rootEnds(const Chain &chain);

    // What a call has to be given to reach a root: the root, and where the root is a local
    // declared from another (a borrow, or a move that left the other dead), that one too.
    static llvm::SmallVector<mlir::Value> rootSources(const Chain &chain);

    // Everything that may point into what `start` points to: the forward closure over users'
    // results that may point anywhere. A call's result is not followed: the call hands back a
    // value it owns.
    const llvm::DenseSet<mlir::Value> &derivedFrom(mlir::Value start);

    // May this op of the function's drops destroy what the chain reads? An overwrite of a place
    // on it, matched by position and type (two references to one field of one object always
    // agree on both; an unrelated field that happens to agree only adds an error); an array op
    // that removes elements of the type of an element on it; a call whose callee may destroy
    // anything (`__own_no_drops`, phase 4) - any such call, for a root this function does not
    // own, since the owner may be a global the callee assigns; else one given something derived
    // from the root that the read did not produce.
    // `borrowed`, when given, is the value the borrow holds in place of the read's result (a value
    // moved into the place, spec 29), and `read` may then be null.
    // Does an overwrite of a place at this position, with this slot type, drop the chain?
    bool overwritesChain(int64_t position, mlir::Type slotType, const Chain &chain);
    bool dropsChain(mlir::Operation *drop, const Chain &chain, mlir::Operation *read, mlir::Value borrowed = {});

    // A read out of a container is a borrow of it (spec 2.3), and so is a `let` declared from one:
    // its retain and releases go when the check passes, as for any borrowing `let` (phase 2).
    // Reports and returns false when the read is kept, or used after something that may destroy
    // it.
    bool checkPlaceRead(mlir::Operation *read, llvm::SetVector<mlir::Operation *> &toErase);

    // A move out of a field of an owning local that holds a record (spec 48): rc's retain of what
    // is read, the one taker that keeps it, and the field, which the move leaves empty.
    struct FieldTake
    {
        mlir::Operation *retain = nullptr;
        mlir::Operation *taker = nullptr;
        mlir::Value field;
        mlir_ts::VariableOp slot;
        // a field of a temporary (spec 63): the temporary's release, before which the field is emptied
        mlir::Operation *release = nullptr;
        // a field of an object a `const` names (spec 64): emptied right at the read
        bool emptyAtRead = false;
    };

    // Is `kept`, what keeps the borrow `read` (a read of a field of an owning record local), one
    // move of it into a container: rc's retain and a push, a store into a place or a kept argument?
    bool fieldTakeOf(mlir::Operation *read, mlir::Operation *kept, FieldTake &take);

    // ...made once for each value read, with nothing using the borrow after it, and nothing reading
    // the field or the record before the local ends?
    bool fieldTakeHolds(mlir::Operation *read, const FieldTake &take, llvm::ArrayRef<BorrowUse> uses);
    // Is `kept` rc's retain of `read`, a read of a field of a call's own result released as a
    // temporary right after, with nothing else reading that field (spec 63)?
    bool temporaryFieldTakeOf(mlir::Operation *read, mlir::Operation *kept, FieldTake &take);
    // ...or of an object a `const` names, `new`'s or a call's own, or an owning `let` given only such
    // objects holds, that nothing sees after the move but its ends, and that is not taken from twice
    // (specs 64, 65)?
    bool namedFieldTakeOf(mlir::Operation *read, mlir::Operation *retain, mlir_ts::PropertyRefOp field, FieldTake &take);
    // Can `call` be given a copy of `value`, a string read out of a place, in its place (callCopies)?
    static bool copiesInto(mlir::Operation *call, mlir::Value value);
    // Does `store` put `value`, a string read out of a place, into another place with no retain of
    // rc's for it, so that the place can take a copy instead (callCopies)?
    bool storesCopyOf(mlir::Operation *store, mlir::Value value);

    // A local that holds a borrow through `user`, and the field path under it that holds it (empty:
    // the local's value itself): a store of it into the local, or into a field of the local's value
    // tuple at any depth (an object literal built in place), or the local declared from it.
    struct BorrowHolder
    {
        mlir_ts::VariableOp local;
        llvm::SmallVector<int64_t, 2> path;
    };

    static std::optional<BorrowHolder> borrowHolderOf(mlir::Operation *user);

    // Where the holder is given a new value: the local's declaration when it has an initializer
    // (`let l!: V` stores nothing, and in a loop the slot keeps the last round's value: fix-up
    // re-review N4), and every store into the field path that holds the borrow or into a prefix of
    // it (the whole local, `o.p` for `o.p.v`). A store into a sibling field, or below the path,
    // gives the path no new value.
    static llvm::SmallVector<mlir::Operation *, 4> holderRenewals(BorrowHolder holder);

    // Whether one field path is on the way to the other: a read of either may reach the other's value.
    static bool pathsMeet(llvm::ArrayRef<int64_t> a, llvm::ArrayRef<int64_t> b);

    // A read of what a mortal global holds (spec 25: an immortal one's one store is its
    // initializer): a heap value it owns, not a handle (spec 23.3 counts those).
    bool isGlobalRead(mlir_ts::LoadOp loadOp);

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
    void checkGlobalRead(mlir_ts::LoadOp read);

    // A borrow read out of a parameter this function keeps, taken before the parameter moves into a
    // new owner (spec 33, review fix): from the move on it is bounded by that owner, as a read through
    // the owner would be (spec 29) - by the owner's ends and the drops of its place after the move.
    // `const inner = p.v; let r = p; r = new N(...); inner.x` reads what the assignment freed. A move
    // into anything that is not such a place (a call that keeps it, a return, a push) ends the borrow
    // there. So do the releases of the parameter's slot (spec 33's exit releases). True when some use
    // of the borrow comes after one of these (`outlives` reported it).
    bool keptParamOutlives(const Chain &chain, llvm::function_ref<bool(mlir::Operation *)> outlives);

    // A parameter, or a view of one (`<C>u`, the payload `ts.Unbox` reads out of an `any`), that
    // this function returns as a borrow (`__own_result_borrows`). rc retained it for the caller,
    // who now takes no reference: the retain goes, and nothing may keep it. The caller's own
    // borrow of the argument bounds it; nothing here can end that but a call, and a call that may
    // is the caller's use of what it passed (spec 2.4, phase 4).
    void checkReturnedParam(mlir::Value value, llvm::SetVector<mlir::Operation *> &toErase);

    // `the result of 'first'`, for a call whose result borrows an argument.
    static std::string describeCall(mlir::Operation *call);

    // The function a call names - directly, or the method it calls through a vtable - or empty.
    static std::string calleeName(mlir::Operation *call);

    // `'h.c'`, `'arr[]'`, `'h.c.v'` - or `a field`, `an element` when the root has no name (a
    // temporary, or no --di).
    std::string describePlace(mlir::Value ref);

    std::string pathOf(mlir::Value ref);

    static std::string fieldName(mlir_ts::PropertyRefOp propertyRefOp);

    // The name of a temporary that owns: a folded `const` has no ts.Variable, but under --di it
    // has a ts.DebugVariable, whose location carries the name.
    static llvm::StringRef ownerName(mlir::Value value);

    // A `ts.RetainSlot`'s slot holds its initializer where the retain is. MLIRGen puts the retain at
    // the declaration (takeOwnershipOfLocal), and nothing between the two may write the slot: in
    // their block, no op is given the slot, no call runs, and no op has regions (a store could hide
    // in them). What is stored into the slot later is decided at its own store; this retain takes
    // only the initializer. No closure captures it.
    static bool slotHoldsInitializerAt(mlir::Operation *op);

    // The value a retain acquires: a Retain's operand, or a RetainSlot's variable's initializer,
    // seen through its views. None for a variable with no initializer - its storage was hoisted
    // in front of a try and its value arrives by a store this phase does not follow.
    static mlir::Value retainedValue(mlir::Operation *op);
    // The store that gives a local declared without a value (one in a `try`) its value, right
    // before rc's `retainSlot` of it, when that is the local's one store; null otherwise.
    static mlir_ts::StoreOp slotDeclarationStore(mlir::Operation *retainSlot);
    // ...and the other way round: rc's `ts.RetainSlot` right after such a store, or null.
    static mlir::Operation *declarationRetainOf(mlir::Operation *store);

    // Does a value of this type own a block that a release would destroy?
    bool ownsHeap(mlir::Value value);

    bool ownsHeap(mlir::Location location, mlir::Type type);

    // A tuple read out of a local that a `ts.Constant` initializes, whose owning fields hold what
    // that constant put there - null, a number, an immortal literal - or a fresh value moved in:
    // the initial state a generator's maker copies into its state object, or a record built to be
    // returned (`return {value: new C(), done: false}`, a `yield`'s result). rc's retain of it is
    // its reference for the values moved in, which own moves instead (recordRetainOf).
    bool holdsConstantsOrMoves(mlir::Value value);

    // rc's one `ts.Retain` of a record local's one read, where the read is only returned: stored
    // into a local that owns nothing and is read only by returns (the result slot), or copied into
    // the box it is given in as an object or an interface (spec 27). It is the reference rc gives
    // each fresh value stored into the record, so it is where they move in. Null for any other
    // record.
    static mlir::Operation *recordRetainOf(mlir_ts::VariableOp varOp);

    // `const b = [...a, ...a]`: MLIRGen builds the array in a local of its own that owns nothing,
    // starting from a fresh one and pushing into it, and reads it once, when it is done, into the
    // owning local it is made for. That local's `ts.RetainSlot` (`retain`) is the reference: the
    // fresh array moves into the builder and then into the owner, and nothing else holds it.
    bool isBuiltArrayRead(mlir::Operation *retain, mlir::Value value);

    // `return [...a]`: the builder of isBuiltArrayRead, read once, when it is done, and returned.
    // The return's `ts.Retain` (`retain`) is the reference the caller gets: the fresh array moves
    // into the builder and then out of the function, and nothing else holds it. The builder's
    // other reads only bind its `push` (`ts.ThisSymbolRef @__array_push`) or are not used. A store
    // into a field is the same move, into a container: `h.c = [...a]`, and a generator's local
    // (`const groups = [...result]`), which is a field of its state object.
    bool isBuiltArrayReturned(mlir::Operation *retain, mlir::Value value);

    // The one `ts.RetainSlot` of a local; null when there are none or several.
    static mlir::Operation *onlyRetainSlotOf(mlir_ts::VariableOp varOp);

    // A local that owns nothing, is never captured, and is read only to be returned.
    static bool isResultSlot(mlir_ts::VariableOp varOp);

    // The record whose field `taker` stores into, and rc's retain that moves the stored value in;
    // null when `taker` is not such a store.
    static mlir::Operation *recordRetainOfStore(mlir::Operation *taker);

    // A string constant cast to `string` is the global itself, which a release skips; so is a read
    // of an immortal global (spec 25).
    static bool isImmortalLiteral(mlir::Value value);

    // A use that reads the value without keeping it. Kept deliberately short: anything not listed
    // is treated as taking the value, which can only turn a program into an error.
    static bool isBorrow(mlir::Operation *user, mlir::Value value);

    // `ts.CreateBoundFunction(ts.Cast(object), method)` split straight back into `ts.GetThis` and
    // `ts.GetMethod` for a call: a method of another module's class - its constructor, when it is
    // built with `new` - called on an object here. It reads the object as `ts.ThisSymbolRef` does
    // for a method of this module; nothing else may use the cast or the bound function, and the
    // object comes out again only as a call's argument, which is a borrow of its own. An
    // interface's `this` (`ts.ExtractInterfaceThis`) is bound so for a call of a function-typed
    // field (`i.toString()` where `toString: () => string`).
    static bool isBoundForCall(mlir::Operation *thisOp);

    // Is `value` an argument the call's callee keeps?
    static bool isKeptArgument(mlir::Operation *call, mlir::Value value);

    // The moves out of a fresh or unknown SSA value. Each taking use (spec 11.1) is a move:
    //   - nothing else may use the value after it - another taker included - and a taker that
    //     control comes back to without the value being made again is a move on every iteration
    //     of a loop (spec 2.6);
    //   - where the temporary is also released (`__owned_result` rc gives back at block end), the
    //     taker must be one rc gave its own reference (takerAcquires), and each release the move
    //     reaches must be dominated by it - that release then has nothing to give back and is
    //     erased. A release the move cannot reach stays: it is the owner on the other paths.
    // Reports and returns false otherwise.
    bool checkValueMoves(mlir::Value value, llvm::SetVector<mlir::Operation *> &toErase);

    // No release of `value` comes after another on any path. Reports and returns false otherwise.
    bool checkReleasedOnce(mlir::Value value);

    // The move verdict for an SSA value (phase 1). Reports and returns false when it is not one.
    bool movesOf(mlir::Value value, llvm::SetVector<mlir::Operation *> &toErase);

    // A field taken out of a record value (`ts.ExtractProperty`, `const r = rec(); r.a.x`) only to be
    // read, through any depth of fields: a read of the record, as a field read through a slot is.
    // Collects what reads it, each a read of the record that can come after a move of it.
    bool fieldOnlyRead(mlir::Operation *extract, llvm::SmallVectorImpl<mlir::Operation *> &reads);

    // What takes a field taken out of a record value, through any depth of fields: each use that is not a
    // read. checkValueMoves lets them borrow it when each is a `let` its record's release bounds.
    void fieldTakersOf(mlir::Operation *extract, llvm::SmallVectorImpl<mlir::Operation *> &takers);

    // What a merge of branches makes of `used` (`c ? h : h2`): the result of an `arith.select` it
    // is a choice of, or each block argument a branch passes it as. Empty for any other use.
    static llvm::SmallVector<mlir::Value, 2> mergedValuesOf(mlir::Operation *user, mlir::Value used);

    // Is the merge `user` makes of `used` only a borrow of it - everything the merged value leads
    // to reads it (walkBorrowed), a `let` declared from it included (`let each = c ? h : h2`: its
    // `ts.RetainSlot` and `ts.ReleaseSlot`s go, as a borrowing `let`'s do), and none of that runs
    // after one of `releases`, the value's own ends? Each such read joins `uses`. False when
    // nothing merges it: the merge then takes the value, as any other use that is not a read does.
    bool borrowsThroughMerge(mlir::Operation *user, mlir::Value used, mlir::Operation *definition,
                             llvm::ArrayRef<mlir::Operation *> releases, llvm::SmallVectorImpl<BorrowUse> &uses,
                             llvm::SetVector<mlir::Operation *> &toErase);

    // the `ts.RetainSlot`s of the `let`s that borrow a merge (borrowsThroughMerge): erased, and
    // decided there
    llvm::DenseSet<mlir::Operation *> mergeBorrowRetains;

    // A call that keeps an argument and is given the same value again through another operand - the
    // value itself, a view of it, something read out of it, another read of its slot (`f(n, n)`,
    // `f(h, n, n.v)` with only the first `n` kept): the callee takes the value while it borrows it
    // through the other operand, and may free it under that borrow, so the other operand is a use
    // after the move (spec 33, review). `sources` are what the moved value is: the value, or every
    // read of the owning slot it came from. The call itself, or null.
    mlir::Operation *callGivenAgain(mlir::Operation *taker, mlir::Value moved, llvm::ArrayRef<mlir::Value> sources);

    // Does control come back round to `taker` without making the value again? Only then is it a
    // move on every iteration of a loop.
    bool takerLoops(mlir::Operation *taker, mlir::Operation *definition);

    // Can control reach `to` after `from` has run, without passing `kill` on the way? `kill` is
    // the source's definition - the producer of an SSA value, or a local's `ts.Variable` - and a
    // path through it is a new value, not this one. Straight-line order within a block, then the
    // CFG from the block's successors; a path back into `from`'s own block reaches every op in it.
    //
    // Ops nested in a region that is not the function body are compared by their ancestors in the
    // body. That answers "reachable" whenever the two share an ancestor, which can only turn a
    // program into an error. (By this pass async bodies and lambdas are functions of their own;
    // after the affine lowering no op of a body holds a region, so the case is kept for safety.)
    bool reachableAfter(mlir::Operation *from, mlir::Operation *to, mlir::Operation *kill);

    // Whether reachableAfter can order `from` and `to` at all: both inside the function body and not
    // one op of it (nested in the same statement). A check whose safe answer is "not reachable" must
    // refuse what it cannot order, where reachableAfter would answer "reachable".
    bool orderedInBody(mlir::Operation *from, mlir::Operation *to);

    // The same, where a path through any of `kills` is a new value. A use that is itself a kill
    // runs after the value is made again.
    bool reachableAfter(mlir::Operation *from, mlir::Operation *to, llvm::ArrayRef<mlir::Operation *> kills);

    // The use that takes `value` for this retain: the declaration a `ts.RetainSlot` belongs to,
    // or the one use of the value, or of a view of it, after a `ts.Retain` that is not a read.
    static mlir::Operation *takerOf(mlir::Operation *retain, mlir::Value value);

    // rc reads a place twice for one copy - `.map(f)` boxes its array so - and retains the first
    // read, whose only use is that retain: the read of the same place that follows it in its block,
    // with nothing stored into the place between, is the same value. Null when there is none.
    // The one op that uses `value`, but `--di`'s ts.DebugVariable, which only names it; or null.
    static mlir::Operation *soleUser(mlir::Value value);

    static mlir::Value twinReadAfter(mlir::Operation *retain, mlir::Value value);

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
                       llvm::SetVector<mlir::Operation *> &toErase, bool placeReleases = false);

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

    static bool isJoined(unsigned state);

    // A value of this type can point into a block: anything but a number, a boolean, a literal or
    // an empty value.
    static bool carriesPointer(mlir::Type type);

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
    bool reachableFrom(mlir::Block *start, mlir::Operation *to, llvm::ArrayRef<mlir::Operation *> kills);

    // The blocks of an exception handler: from a landing pad to the end of its catch. A release there
    // would be a call inside an exception funclet.
    llvm::SmallPtrSet<mlir::Block *, 8> handlerBlocks();

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
                             llvm::SetVector<mlir::Operation *> &toErase);

    // The releases of the slot, when the local, the move, its assignments and its releases are all in
    // the function's body. False when one is not.
    bool slotReleasesInBody(mlir_ts::VariableOp varOp, mlir::Operation *acquire,
                            llvm::ArrayRef<mlir::Operation *> assignments, llvm::SmallVectorImpl<mlir::Operation *> &releases);

    // The slot's state through the function, forward from its entry: assigned, moved at `acquire`,
    // released.
    SlotStates slotStates(mlir_ts::VariableOp varOp, mlir::Operation *acquire, llvm::ArrayRef<mlir::Operation *> assignments,
                          llvm::ArrayRef<mlir::Operation *> releases);

    // Places a release of the slot on every edge into the join that holds the value and has not moved
    // it. False, with nothing placed, when an edge cannot take one, or there is none.
    bool unmovedEdges(mlir_ts::VariableOp varOp, const SlotStates &states, llvm::SmallVectorImpl<EdgeRelease> &placed,
                      llvm::SetVector<mlir::Block *> &releasedInto);

    // Whether nothing uses the value where the paths have joined, or after an edge released it.
    // False, with the use in `joinedUse` where there is one, when the release cannot be placed.
    bool noUseWhereJoined(mlir_ts::VariableOp varOp, mlir::Operation *retain, mlir::Operation *taker,
                          llvm::ArrayRef<mlir::Operation *> assignments, const SlotStates &states,
                          const llvm::SetVector<mlir::Block *> &releasedInto);

    // Puts in the releases placeOnUnmovedEdges decided: at the end of a block that goes only to the
    // join, else in a block of its own on the edge.
    void placeEdgeReleases();

    // The stores of the assignments to an owning local: rc releases what the slot holds right
    // before it stores the new value (an assignment, and a `var` declaration met again).
    static llvm::SmallVector<mlir::Operation *> slotAssignments(mlir::Value slot);

    // A use of what `read` returned - directly or through casts - that the move at `taker` can
    // reach, other than the move's own retain and taker. None if there is none. A path back
    // through `read` itself reaches a use of a value read again: the read, met after the move,
    // is the slot's own use to check (around a loop, of a value an assignment stored).
    // `isRead` accepts a use that reads the value through its new owner (spec 29).
    mlir::Operation *readUsedAfter(mlir_ts::LoadOp read, mlir::Operation *retain, mlir::Operation *taker,
                                   mlir::Operation *kill,
                                   llvm::function_ref<bool(mlir::Operation *, mlir::Value)> isRead = nullptr);

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
    static bool onlyReadThroughOwner(mlir::Value value, llvm::function_ref<bool(mlir::Operation *, mlir::Value)> isRead);

    // A use that only reads the value: a field, element or length read through it, a method reached
    // through it, a comparison or arithmetic, a print, an assert, a test, or an argument the callee
    // borrows. Anything else - a store, an owning `let`, a keeping call, a return - would be a second
    // owner.
    static bool isReadAfterMove(mlir::Operation *user, mlir::Value value);

    // The place a taker moves its value into, as a chain checkPlaceRead would read it: an owning
    // local (a declaration, or an assignment), or a field or element - of a local or a parameter
    // (`this`). False for anything else: a call that keeps it, a push, a return, a global, a
    // captured variable, a handle's payload.
    bool destinationChain(mlir::Operation *taker, Chain &chain);

    // Is `use`, a use of `value` after `taker` moved it, a read through the place it moved into
    // (spec 29)? `made` is where the value is made again (its definition, the local's declaration
    // and assignments). Sound when no path from the move through an end or a drop of the place to
    // the read - or to anything the read produced - makes the value again or moves it again: such a
    // path is the only way the place can stop holding the value before the read.
    bool readsThroughOwner(mlir::Operation *taker, mlir::Value value, mlir::Operation *use,
                           llvm::ArrayRef<mlir::Operation *> made);

    // Where rc takes the receiver's reference: its `ts.Retain` of the value, when that comes
    // first in the taker's block, else the taker itself (a declaration, whose `ts.RetainSlot`
    // follows it). A release between the two - a `return`'s scope exit runs after its retain
    // and before the store into the return slot - is after the move.
    static mlir::Operation *acquirePoint(mlir::Operation *taker, mlir::Value value);

    // Did rc give this taker a reference of its own - a `ts.Retain` of the value in front of it,
    // or the `ts.RetainSlot` of an owning local declared from it? Only then is there a release on
    // the taker's side to give the value back once it has moved. `ts.NewInterface` is a taker to
    // this pass but a view to rc, which retains nothing for it and relies on the instance's own
    // release; moving the instance into it would leave nothing to release it at all.
    static bool takerAcquires(mlir::Operation *taker, mlir::Value value);

    // The declaration's name survives only as debug metadata (`--di`). At this level MLIRGen's
    // form of it is a NameLoc first inside the location fused with the scope - what LowerToLLVM's
    // preserveTypesForDebugInfo reads - and after that pass it is a DILocalVariable. Without
    // `--di` the error still points at the right place.
    static llvm::StringRef varName(mlir_ts::VariableOp varOp);

    static llvm::StringRef nameAt(mlir::Location location);

    static llvm::StringRef describe(mlir::Value value);

    // A second reference nobody proved safe. `value` is the value it is to, when the caller knows
    // it; a retain names its own. The shapes this pass recognises get the rule they break (spec
    // 21.2); anything else, the catch-all.
    void reportSecondReference(mlir::Operation *op, mlir::Value value = {});

    // The named shapes of a second reference to `root` (spec 21.2). False when it is none of them.
    bool reportShapeOfSecondReference(mlir::Operation *op, mlir::Value root, llvm::StringRef name);

    // The name of parameter `index`, as its slot carries it (under `--di`).
    llvm::StringRef paramName(int index);

    // Is `root`, or a view of it, stored into the result slot?
    static bool isReturned(mlir::Value root);

    // A return that disagrees with one of a borrow of parameter `param` (with -1, with one that
    // borrows nothing): a store into the result slot of a value that holds a block and borrows
    // another parameter or none - or, for -1, borrows a parameter. Null when there is none.
    mlir::Operation *otherReturn(int param);

    // Where the body would have been fine had its callers known a fact about it, say why they
    // cannot.
    void noteLostFacts(mlir::InFlightDiagnostic &diag);

    void reportKeptOnSomePaths(mlir::Operation *move, mlir::Operation *exit, llvm::StringRef name);

    // A handle made into a union whose other members include one that owns a block (spec 23.2):
    // `Shared<T> | string` given a Shared<T>. gc, none and rc accept it.
    void reportHandleInOwningUnion(mlir::Operation *op);

    static std::string printed(mlir::Type type);

    void reportGivenNotOwned(mlir::Operation *call, mlir::Value arg);

    void reportUseAfterMove(mlir::Operation *use, mlir::Operation *move, llvm::StringRef name);

    void reportMovedInLoop(mlir::Operation *move, llvm::StringRef name);

    void reportBorrowOutlives(mlir::Operation *use, mlir::Operation *end, llvm::StringRef name, llvm::StringRef owner);

    void reportPlaceBorrowOutlives(mlir::Operation *use, mlir::Operation *drop, llvm::StringRef name,
                                   llvm::StringRef place);

    void reportPlaceBorrowerAssigned(mlir::Operation *op, llvm::StringRef name, llvm::StringRef place);

    void reportPlaceBorrowEscapes(mlir::Operation *op, llvm::StringRef name, llvm::StringRef place);

    void reportMovedWhileBorrowed(mlir::Operation *op, llvm::StringRef owner, llvm::StringRef borrower);

    void reportBorrowEscapes(mlir::Operation *op, llvm::StringRef name, llvm::StringRef owner);

    void reportBorrowerAssigned(mlir::Operation *op, llvm::StringRef name, llvm::StringRef owner);

    void reportEscapingCellNotOwned(mlir::Operation *capture, mlir::Operation *escape, llvm::StringRef why);

    void reportUsedAfterEscapingCapture(mlir::Operation *use, mlir::Operation *capture, llvm::StringRef name);

    void reportClosureOutlives(mlir::Operation *use, mlir::Operation *end, llvm::StringRef closure,
                               llvm::StringRef captured);

    void reportCaptureNotOwned(mlir::Operation *op, llvm::StringRef captured);

    void reportCapturedParamAssigned(mlir::Operation *op, llvm::StringRef name);

    void reportMovedOnSomePaths(mlir::Operation *move, mlir::Operation *release, llvm::StringRef name,
                                mlir::Operation *usedAt = nullptr);
};

} // namespace own_inference

#endif // MLIR_TYPESCRIPT_OWNERSHIPINFERENCEPASS_H
