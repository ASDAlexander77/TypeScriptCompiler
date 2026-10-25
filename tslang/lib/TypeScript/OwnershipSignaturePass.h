// OwnershipSignaturePass, the callee facts of -mm=own. Its methods are defined across the
// lib/TypeScript/OwnershipSignature*.cpp files, one analysis each; templates stay here.

#ifndef MLIR_TYPESCRIPT_OWNERSHIPSIGNATUREPASS_H
#define MLIR_TYPESCRIPT_OWNERSHIPSIGNATUREPASS_H

#include "mlir/Pass/Pass.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/FunctionInterfaces.h"

#include "TypeScript/TypeScriptDialect.h"
#include "TypeScript/TypeScriptOps.h"
#include "TypeScript/Passes.h"
#include "TypeScript/Defines.h"
#include "TypeScript/MLIRLogic/MLIRTypeHelper.h"

#include "OwnershipFacts.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringSet.h"

#include <map>

namespace mlir_ts = mlir::typescript;

namespace own_signature
{

using namespace own_facts;

// The callee facts of -mm=own (spec 2.4, 3.2), computed over the whole module before the
// per-function inference runs. Each call the pass can resolve - a direct call, a virtual call
// whose candidates are all defined here, the `.instanceOf` slot - gets the facts its candidates
// agree on, pinned on the call op, so the inference reads them without looking anything up.
//
// `__own_no_drops` goes one way: a call with no facts may destroy anything, which can only turn a
// program into an error. The facts a callee's own body relies on - a parameter it keeps, a result
// that borrows an argument - go the other way: a caller that does not know them borrows the
// argument and releases it, or releases a result it does not own, a double free either way. So a
// function has those only when every call that can reach it is one this pass resolved (the closed
// world, see `open`).
//
// `__own_no_drops` alone crosses a module boundary: a library built under own exports the names of
// its exported functions that have it (SHARED_LIB_OWN_FACTS), and a module importing it calls them
// knowing it. The other two cannot: a module linking the library statically re-parses its source
// and sees no facts at all, and a caller that does not know them frees twice.
class OwnershipSignaturePass : public mlir::PassWrapper<OwnershipSignaturePass, mlir::OperationPass<mlir::ModuleOp>>
{
  public:
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(OwnershipSignaturePass)

    // `executable`: the module is built into a program (`--emit=exe`, or the JIT's entry module),
    // whose exports no other module sees (USER decision, spec 31). Unknown is a library.
    explicit OwnershipSignaturePass(bool executable = false) : executable(executable)
    {
    }

    bool executable = false;

    // A global is a root (rc 5ai): what its initializer makes moves into it, and nothing gives it
    // back. rc retains a fresh value for its `ts.GlobalResult`, and that retain is the move: it
    // goes. The per-function inference never sees a global's region, so any other retain there is
    // reported here, by the global's name, rather than left to the lowering's "left a retain
    // behind".
    // Every use of `value` ends in the global's `ts.GlobalResult`: given to it, or an element of a
    // fresh array literal that is (`const nested = [[1, 2], [3]]`), or the payload of a
    // `new Shared(x)` (`let g = new Shared<S>({...})`). Its retains are judged on their own.
    static bool movesIntoResult(mlir::Value value);

    // A record copied out of a local a `ts.Constant` initializes, whose owning fields are given
    // nothing that holds a block: it holds constants only - numbers, immortal string literals,
    // functions - and nobody else's reference.
    static bool holdsOnlyConstants(mlir::Value value);

    // Whether `op` may run more than once in one run of its function: it is inside a loop op, or its
    // block - or the block of a region op holding it - is on a cycle of its region's CFG, a path from
    // the block's successors back to it. (By this pass the affine lowering has made every loop a CFG
    // cycle of the function body; the loop op is kept for safety.)
    static bool mayRunAgain(mlir::Operation *op);

    // The immortal globals (OWN_IMMORTAL_GLOBAL_ATTR_NAME, spec 25). A global of a class type,
    // not exported, is immortal when only its initializer writes it, once, and gives it a fresh
    // object, nothing, or another immortal global's value. Its initializer is either its own region
    // (`const c = new C()`) or, for a class static, the one store into it in the class's static
    // constructor, which a `ts.GlobalConstructor` runs (the region then yields only the default).
    // Top-level script code is such a constructor too, so a module-level `let c: C;` it stores
    // into once is one as well. "Once" is how often the store runs, not how many stores there are:
    // a store in a loop (`static { for (...) K.w = new V(i); }`) would mark every block it is given
    // immortal, and each old one would leak, so a store that may run again leaves the global mortal,
    // and each store releases the old value, as for any other global.
    //
    // The store - the region's, which the lowering makes - marks the block HEAP_BLOCK_IMMORTAL, so
    // every block the global ever holds is one no release frees. The value test makes sure the block
    // is the global's to keep: an object made right there and given back by nobody. A call's result
    // is not one, whatever its callee: these facts are found before the callees' (computeFacts), and
    // a global's region is no function the inference checks, so a result that borrows its argument -
    // a temporary the region frees, an element of another global's array - would be marked, and
    // stored, as the global's own. Such a global stays mortal, and a read of it keeps today's errors.
    void findImmortalGlobals(mlir::ModuleOp module);

    void moveIntoGlobals(mlir::ModuleOp module);

    void runOnOperation() override;

  private:
    // The function a closure runs, when it names one defined here.
    mlir_ts::FuncOp closureBody(mlir_ts::CreateBoundFunctionOp boundOp);

    // The fields of a closure body's capture box whose cell it assigns (OWN_ASSIGNS_CAPTURES_ATTR_NAME),
    // directly or through a closure it builds over the cell. Every field, when the body is unknown or
    // a cell goes anywhere this does not follow.
    llvm::SetVector<int32_t> assignedCaptures(mlir_ts::FuncOp funcOp, int depth = 0);

    // Is the variable behind this cell assigned - here, or by a closure built over it?
    bool cellAssigned(mlir::Value cell, int depth);

    llvm::DenseMap<mlir::Operation *, llvm::SetVector<int32_t>> assignedCache;

    // What a call may reach: known, when every candidate is a function defined in this module (or
    // the call goes through the `.instanceOf` slot, or reaches a function another module says
    // destroys nothing); else anything.
    struct Callees
    {
        bool known = false;
        bool instanceOf = false;
        bool imported = false;
        // a function read out of a field that only ever holds it (fieldFunctions)
        bool fieldFunction = false;
        // the slot of a closed interface (resolveInterfaceSlot)
        bool viaInterface = false;
        // a parameter of a closed function that callers pass only functions of this module
        // (callbackFamilies), with each member's shift: a closure's body takes its capture box
        // first, so its parameter k is the call's argument k - 1
        bool viaCallback = false;
        llvm::SmallVector<int32_t, 2> shifts;
        // the name the imported function is exported under
        llvm::StringRef importedName;
        llvm::SmallVector<mlir_ts::FuncOp, 2> funcs;
        // the symbols the call names, whether or not it is known: a direct callee, or every
        // member of a virtual slot's family (for the facts a declaration says, declaredFacts)
        llvm::SmallVector<mlir::StringAttr, 2> named;
    };

    struct Call
    {
        mlir::Operation *op;
        Callees callees;
    };

    llvm::StringMap<mlir_ts::FuncOp> functions;
    llvm::StringMap<mlir_ts::GlobalOp> globals;
    // Globals whose address is used other than to load from it: what they hold may change.
    llvm::StringSet<> writtenGlobals;
    llvm::SmallVector<Call> calls;

    // Functions of imported libraries that destroy nothing a caller can reach, by the name they
    // are exported under (SHARED_LIB_OWN_NO_DROPS_ATTR_NAME).
    llvm::StringSet<> importedNoDrops;

    // (tuple type, field) -> the one function an object of that type holds in that field. A
    // generator's state object holds its `next` so.
    own_facts::FieldFunctions fieldFunctions;

    // Class names, from their vtables, to split a method's symbol into class and method name.
    llvm::StringSet<> classNames;
    // (vtable position, method name) -> the functions at that position under that name, in every
    // class vtable of the module: what a virtual call through that slot may reach.
    std::map<std::pair<int64_t, std::string>, llvm::SmallVector<mlir::StringAttr, 2>> families;

    // Functions whose callers are not all known - a caller outside the module, or a reference to
    // the symbol that is not a call this pass resolves - and why, for the error in their body.
    llvm::DenseMap<mlir::Operation *, const char *> open;

    // Methods an interface's vtable holds (`<class>.<interface>..vtbl`, or `<interface>.<n>..vtbl`
    // for an object literal) whose calls through the interface are not resolved: the slot's family
    // is open (interfaceFamilies), or the vtable is one this pass cannot tie to an interface. Such
    // a call (`ts.InterfaceSymbolRef`, then an indirect call) is given none of the method's facts: a
    // caller that hands it an argument it keeps still releases it, and a caller of one whose result
    // borrows an argument or `this` releases the borrow as its own (a double free either way). So a
    // method reached this way that keeps a parameter or returns a borrow - by its body or by `@keeps`
    // / `@lends` - loses its facts, and the body's error says why (`is a parameter, kept here, but
    // -mm=own cannot make the callers of ... give it up`, `is a parameter, returned here, but -mm=own
    // cannot tell the callers of ... that the result borrows it`). A method reached only through
    // closed interfaces keeps its facts: their calls get them (spec 31).
    llvm::DenseSet<mlir::Operation *> interfaceReached;

    // ---- Interfaces closed in this module (G9, spec 31) ----
    //
    // A call through an interface reaches whatever function the slot of the interface value's vtable
    // holds. When every such vtable is one this module defines and every interface value is made
    // from one of them, the slot holds one of a known set of functions - the slot's family - and the
    // call is resolved to it like a virtual call of a private class family: it gets the facts its
    // members agree on (meetFacts demotes a family that disagrees), and `__own_no_drops` when every
    // member destroys nothing. Anything this cannot see opens the interface, and then the call is
    // what it was: no facts, and its members reached that way lose theirs (interfaceReached).
    //
    // Open, for an interface I:
    // - another module can see it: its name in an exported declaration (`__decls`), or `!ts.iface<@I>`
    //   in the type of a function or global another module can reach - an exported or declared
    //   function, a global that is not private, or is exported or imported;
    // - an I value made other than by `ts.NewInterface` over a vtable this pass reads (the global, a
    //   heap copy of it, or the one an object's class vtable holds for I): a cast from another
    //   interface (`J` to `I`, which opens `J` too: its members are reached through `I`), from
    //   `any` or an opaque pointer (which opens every interface put into one), from a tuple or array
    //   that does not hold I already (castMakesInterface), a `ts.InterfaceRef` used as a value;
    // - a vtable of I written after its initializer, or read other than to make an interface value;
    // - a class vtable that holds a vtable of I and is not private (an exported class: another
    //   module may extend it);
    // - and, per slot, a member that is not a function this module defines, a slot that holds no
    //   function (a field's offset, a thunk's cell), or a member that says `@keeps` or `@lends` (its
    //   facts are its signature's, which a family cannot change). A member that is not private (a
    //   `public` method) is open (findOpen): it has no facts of its own to give the family.
    struct InterfaceFamily
    {
        // the slot's name, which every vtable of the interface agrees on
        mlir::StringAttr field;
        llvm::SmallVector<mlir::StringAttr, 2> members;
        bool closed = true;
    };

    // (interface, vtable position) -> the functions every vtable of the interface holds there
    std::map<std::pair<std::string, int64_t>, InterfaceFamily> interfaceFamilies;

    // The words of the module's exported declarations (none in an executable), the classes a class
    // vtable is the own vtable of, and the classes another module can name: exported, or in the
    // type of an exported or declared function or of an exported or imported global (findOpen).
    llvm::StringSet<> exportWords;
    llvm::StringSet<> ownClasses;
    llvm::StringSet<> nameableClasses;

    // A field of function type (`f: (x: number) => number`), called through the interface: the slot
    // holds the offset of a field of each implementing object literal, and the call goes to whatever
    // function that field holds. When every vtable's slot is such an offset and every function ever
    // stored into those fields is one this module defines - the literal's own (`{ f: function ... }`,
    // in its `ts.Constant`), or one stored into the field later, through the object or through the
    // interface - the call is resolved to them (spec 31). A store of anything else (a parameter, a
    // call's result, a closure), a field reached other than to be read or stored, or an object of
    // the literal's type made other than by the literal opens the slot. A class implementing the
    // field holds a thunk's cell, which this does not follow: open.
    struct InterfaceFieldFamily
    {
        mlir::StringAttr field;
        // (object literal's storage tuple, field index) of each vtable
        llvm::SmallVector<std::pair<mlir::Type, int64_t>, 2> targets;
        llvm::SmallVector<mlir::StringAttr, 2> members;
        bool closed = true;
    };

    std::map<std::pair<std::string, int64_t>, InterfaceFieldFamily> interfaceFieldFamilies;

    // interface name -> the tuple types of the vtables its values are made with, when every one of
    // them is read here (holdsFunction: what a value of the interface carries)
    llvm::StringMap<llvm::SmallVector<mlir_ts::TupleType, 2>> interfaceVTableTypes;

    // `ts.PropertyRef(ts.Cast(null), i)`, seen through casts: the offset of field `i` of an object of
    // that type, as an object literal's interface vtable holds it. Null for anything else.
    std::pair<mlir::Type, int64_t> slotField(mlir::Value value);

    // What every object-literal field an interface's field slot may be holds (InterfaceFieldFamily).
    void collectFieldFunctions(mlir::ModuleOp module);

    // The interfaces mentioned by a type as values: the interface itself, a union or optional, a
    // tuple's fields, an array's elements, what a reference points to. Not a class's or an object's
    // fields (a reference to a block made elsewhere), nor a function's parameters and result.
    static void valueInterfaces(mlir::Type type, llvm::StringSet<> &found, int depth = 0);

    // Every interface a type mentions anywhere, a class's fields and a function's parameters
    // included: what another module handed a value of this type could reach. With `classes`, the
    // classes it mentions too (a base class is its derived class's first field).
    static void anyInterfaces(mlir::Type type, llvm::StringSet<> &found, llvm::DenseSet<mlir::Type> &seen,
                              llvm::StringSet<> *classes = nullptr);

    // The function values a value of this type holds: the type itself, a union's or an optional's
    // member, a tuple's field, an array's element. Not what a reference points to (a vtable's
    // address seen as an opaque pointer is the compiler's own, not a value).
    static void functionValues(mlir::Type type, llvm::SmallVectorImpl<mlir::Type> &found, int depth = 0);

    // A function type's parameters and results, whatever kind of function it is.
    static std::pair<llvm::SmallVector<mlir::Type>, llvm::SmallVector<mlir::Type>> signatureOf(mlir::Type type);

    // Does a cast reinterpret a function value - from `any` or `unknown`, or as a function of another
    // signature? Then an interface value its parameters or
    // results hold crosses unseen (a `(o: Op2) => ...` called as `(o: Op) => ...`): `interfaces`
    // gets every interface either side's signatures mention, and `boxed` the input's when it is put
    // into `any`.
    static bool castReinterpretsFunction(mlir::Type in, mlir::Type out, llvm::StringSet<> &interfaces, bool &boxing);

    // The interfaces an interface member's type mentions: a method's or a function field's
    // parameters and result, a field's type, anything they hold (a class's fields included).
    static void memberTypeInterfaces(mlir::Type type, llvm::StringSet<> &found);

    static bool isNullish(mlir::Type type);

    // May a cast from `in` to `out` make an interface value out of something that was not that
    // interface's value already? A union or optional narrowed to its interface member (whose other
    // members are null or undefined) or widened from it, null given a nullable interface type, a
    // class or object seen as another (the same block, its fields as they are) and a function seen as
    // another are not; a cast from another interface, `any`, an opaque pointer, or a tuple or an
    // array whose element is not the same interface is.
    static bool castMakesInterface(mlir::Type in, mlir::Type out, int depth = 0);

    // The function a vtable slot holds: `ts.SymbolRef`, seen through casts. Null for anything else.
    static mlir::StringAttr slotFunction(mlir::Value value);

    static mlir_ts::AddressOfOp addressOfThroughCasts(mlir::Value value);

    // The words of the module's exported declarations (`__decls*` globals): what another module
    // can name.
    static llvm::StringSet<> exportedWords(mlir::ModuleOp module);

    // the vtables: an interface's (`ts.InsertProperty` of each slot, slot 0 `.instanceOf` of a
    // class that is not the vtable's owner, or `.instanceOf.none` of an object literal), and the
    // slots of a class's own vtable that hold the address of one of its interface vtables
    struct InterfaceVTable
    {
        mlir_ts::TupleType type;
        llvm::DenseMap<int64_t, mlir::StringAttr> slots;
        // a field's slot: the object literal's field it is the offset of
        llvm::DenseMap<int64_t, std::pair<mlir::Type, int64_t>> fields;
        bool written = false;
    };

    // What collectInterfaceFamilies learns of the module, one step after another.
    struct InterfaceScan
    {
        // the interfaces open whatever their vtables hold
        llvm::StringSet<> open;
        llvm::StringMap<InterfaceVTable> vtables;
        // `..vtbl` globals that are neither (no `.instanceOf` this reads in slot 0)
        llvm::SmallVector<llvm::StringRef> unread;
        // interface name -> the interface vtables class vtables hold for it, read through an
        // object's class at run time (`ts.VTableOffsetRef` of the object's vtable)
        llvm::StringMap<llvm::SetVector<llvm::StringRef>> classHeld;
        // interface name -> the vtables its values are made with
        llvm::StringMap<llvm::SetVector<llvm::StringRef>> madeWith;
        // An interface value made out of another one reaches the other's members: both are open. One
        // made out of `any` or an opaque pointer may be any interface value put into one: every
        // interface so boxed is open with it, and every interface at all when it is untraceable.
        bool allOpen = false;
        llvm::StringSet<> boxed;
        llvm::StringSet<> unboxedInto;
        bool unboxedFunction = false;
        // the interfaces a cast makes a value of out of another interface's or an opaque one, and
        // the ones it came from: closed over their members' types below
        llvm::StringSet<> castOpened;
        // interface -> the interfaces its members' types mention: its vtables' slots and what is
        // read out of its values (`ts.InterfaceSymbolRef`), function signatures and class fields
        // included
        llvm::StringMap<llvm::StringSet<>> memberInterfaces;
        llvm::DenseSet<llvm::StringRef> tied;

        void openAll(const llvm::StringSet<> &names)
        {
            for (auto &entry : names)
            {
                open.insert(entry.getKey());
            }
        }
    };

    void collectInterfaceFamilies(mlir::ModuleOp module);

    // The vtables: each interface vtable's slots, and the interface vtables a class's own vtable holds.
    void readInterfaceVTables(mlir::ModuleOp module, InterfaceScan &scan);

    // What makes an interface value, out of which vtable; the casts that box, unbox or reinterpret one;
    // the interfaces its members' types mention; the vtables written.
    void traceInterfaceValues(mlir::ModuleOp module, InterfaceScan &scan);

    // The interfaces another module sees, and the ones a value boxed into `any` may be unboxed into.
    void openExposedInterfaces(mlir::ModuleOp module, InterfaceScan &scan);

    // Every interface the members of one made out of another mention, recursively.
    void openThroughMembers(InterfaceScan &scan);

    // An interface with a vtable this pass could not tie to its values.
    void openLooseVTables(InterfaceScan &scan);

    // The families of each interface's slots - its methods and its function-typed fields - closed
    // where nothing opened the interface.
    void buildInterfaceFamilies(InterfaceScan &scan);

    // What an open family holds, and what a vtable no interface value is made with holds.
    void markInterfaceReached(InterfaceScan &scan);
    static constexpr const char *INTERFACE_REACHED_WHY =
        "it is reached through an interface, whose calls are not given its facts";

    // The parameters a function's result borrows and the parameters it keeps (spec 2.4); absent is
    // rc's convention. A plain borrow names one parameter; a borrowing record (`record`, spec 32)
    // any number, and its caller owns the box.
    struct Facts
    {
        llvm::SmallVector<int32_t> borrows;
        llvm::SmallVector<int32_t> owned;
        bool record = false;
        // the record is a tuple that owns nothing (OWN_RESULT_RECORD_UNBOXED_ATTR_NAME)
        bool unboxed = false;

        bool empty() const
        {
            return borrows.empty() && owned.empty();
        }

        bool operator==(const Facts &other) const
        {
            return borrows == other.borrows && owned == other.owned && record == other.record && unboxed == other.unboxed;
        }

        bool operator!=(const Facts &other) const
        {
            return !(*this == other);
        }
    };

    // Each function's facts, from its body (local) and as its callers see them (effective: closed,
    // and agreed by every family it is in).
    llvm::DenseMap<mlir::Operation *, Facts> factsLocal;
    llvm::DenseMap<mlir::Operation *, Facts> factsEffective;
    // The parameters each body takes into something - a field, an element, a global, a keeping call,
    // an owning local it assigns (localOwnedParams). A kept parameter that is not one of them is
    // released on the paths that do not move it (OWN_PARAMS_RELEASED_ATTR_NAME, spec 33).
    llvm::DenseMap<mlir::Operation *, llvm::SmallVector<int32_t>> takenParams;

    // ---- Facts a signature says (`@keeps`, `@borrows`, `@lends`) ----
    //
    // A function that says them has them whoever calls it: a caller in another module reads them
    // from the declaration it re-parses. Its body is checked against them instead (checkDeclared).

    static llvm::ArrayRef<int32_t> declared(mlir::Operation *funcOp, llvm::StringRef name);

    static bool declaresFacts(mlir::Operation *funcOp);

    // The facts `@keeps` and `@lends` say (Facts): what callers move in, and what the result is a
    // borrow of.
    static bool declaresCallerFacts(mlir::Operation *funcOp);

    // The parameters `@lends` names: one for a plain borrow, any number for a borrowing record
    // (spec 32).
    static llvm::SmallVector<int32_t> declaredLends(mlir::Operation *funcOp);

    // What every function a call names declares under `name`, when they all declare the same;
    // nothing when one of them does not.
    llvm::SmallVector<int32_t> declaredByCallees(const Callees &callees, llvm::StringRef name);

    // Does every function a call names declare the unit marker `name` (`@record`)?
    bool calleesDeclare(const Callees &callees, llvm::StringRef name);

    // The parameters, for an error, by the names MLIRGen recorded beside the marker.
    static std::string describeParams(mlir_ts::FuncOp funcOp, llvm::ArrayRef<int32_t> indices);

    // Functions this module defines that `@borrows`, and those whose state was found to borrow.
    llvm::DenseMap<mlir::Operation *, llvm::SmallVector<int32_t>> borrowsFound;

    // The argument of `funcOp` that `value` is, or a field of: its own value, its slot's, or a
    // read out of either. -1 for anything else.
    static int32_t parameterOf(mlir::Value value, mlir_ts::FuncOp funcOp);

    // ---- The iterator protocol ----
    //
    // `it.next()` returns a `{value, done}` record, and under own the caller owns what it holds,
    // whoever implements it: a generator's `next` moves what it yields into the record (a yield of
    // a borrow is an error in its body), and any other `next` is a function whose callers this pass
    // cannot all see, which may not return a borrow. So the result is fresh even when the call goes
    // through the `Iterator` interface, or into another module - the default library's `matchAll`.

    // A call of a function named `next` - read through an interface, out of a field, or by its
    // symbol - whose one result is a `{value, done}` record.
    static bool isIteratorNext(mlir::Operation *op);

    // A generator another module made: its state object's type is only its shape here, and what each
    // field owns - a local, a yielded value now the caller's, what it borrows from its arguments - is
    // known in that module alone. A release routine made here from the shape would give back what is
    // not the state's. So this module's gives back none of it: only the `.captured` box's cells, and
    // their values too unless the maker says it borrows them (`@borrows`). What the generator's own
    // locals still hold when it is given up early leaks rather than being freed twice.
    void findImportedStates(mlir::ModuleOp module);

    // A call whose result is a generator's state object (an object whose fields are `.step` and
    // `next`): a generator function made it, a `ts.New` block it hands over (spec 18.1), which the
    // caller owns - in another module too, as the iteration that follows needs.
    static bool makesGeneratorState(mlir::Operation *op);

    // storageTupleOf, without the pass's context: an object's storage, a reference's or a tuple
    static mlir_ts::TupleType storageTupleOfType(mlir::Type type);

    // ---- Fields a generator yields ----
    //
    // A generator's locals are fields of its state object, and `yield groups` puts what one holds into
    // the `{value, done}` record its `next` returns: a borrow of the state, which the caller cannot
    // keep. When the field is read for nothing but yields, and each value stored into it is yielded
    // before the generator can stop (later in the same block), the yield moves the value instead:
    // the read is fresh (OWN_FRESH_RESULT_ATTR_NAME) and the field owns nothing
    // (OWN_YIELDED_FIELDS_ATTR_NAME).

    // `ts.Store(value, ts.PropertyRef(record, i))`, where `record` is a local a `ts.Constant`
    // initializes: a field of a record built to be returned, a yield's.
    static bool isRecordFieldStore(mlir::Operation *user, mlir::Value value);

    void findYieldedFields(mlir::ModuleOp module);

    // A function that says `@borrows` and makes no state itself, but returns what a call that has
    // a bounded result gives it (`return this.value.matchAll(expr)`): it borrows the parameters
    // the call's bounded arguments are, or are read out of.
    void findBoundedForwarders();

    // A body that does other than its signature says, and an override that says other than the
    // method it overrides: either way a caller that trusts the signature frees twice or leaks.
    void checkDeclared();

    // Functions that destroy nothing their caller can reach.
    llvm::DenseSet<mlir::Operation *> noDrops;

    // What else a function destroys that its caller can reach, when it is only the old values of
    // places it overwrites: each place's position and slot type (OWN_DROP_FIELDS_ATTR_NAME). A
    // caller judges a call of it as it would those stores, made in its own body. `any` when it may
    // destroy something else (a global, a handle's payload, a captured variable, an element popped,
    // a call that may).
    struct FieldDrops
    {
        bool any = false;
        llvm::SmallVector<std::pair<int64_t, mlir::Type>, 4> places;

        bool add(std::pair<int64_t, mlir::Type> place)
        {
            if (llvm::is_contained(places, place))
            {
                return false;
            }

            places.push_back(place);
            return true;
        }
    };

    llvm::DenseMap<mlir::Operation *, FieldDrops> fieldDrops;

    // The body's own share of a function's FieldDrops: its stores, not its calls.
    FieldDrops fieldDropsInBody(mlir_ts::FuncOp funcOp);

    // The FieldDrops of a call's callees together, when every one has them.
    std::optional<FieldDrops> calleesFieldDrops(const Callees &callees);

    // A class's vtable is a global named `<class>..vtbl` that holds its `..instanceOf`. An
    // interface's vtable for a class (`B.I..vtbl`) holds only the methods and is not one: a call
    // through an interface does not name what it reaches.
    void collectClassVTables(mlir::ModuleOp module);

    // Does a method reached through an open interface keep a parameter or return a borrow
    // (interfaceReached)? Then its callers through the interface cannot be told, and it has no facts
    // at all.
    bool factsThroughInterface(mlir::Operation *funcOp, const Facts &facts);

    mlir_ts::TupleType storageTupleOf(mlir::Type type);

    // The functions a class vtable holds, by position; empty for any other global.
    static llvm::SmallVector<std::pair<int64_t, mlir::StringAttr>> classVTableEntries(mlir_ts::GlobalOp globalOp);

    static bool isClassVTableEntry(mlir::Operation *op);

    // `B.get` -> `get`: the symbol with the longest class name that prefixes it taken off. An
    // override keeps its base's method name, which is what puts both in one family.
    std::string methodName(llvm::StringRef symbol);

    llvm::SmallVector<mlir::StringAttr, 2> familyOf(int64_t index, mlir::StringAttr identifier);

    // Adds `symbol`'s function to `callees`; false when it is not defined in this module.
    bool addDefined(mlir::StringAttr symbol, Callees &callees);

    // A function declared here and defined in a library that exported its `__own_no_drops`: one
    // linked with this module, through its import library. Or a C function declared
    // `@nodrops` (OWN_DECLARED_NO_DROPS_ATTR_NAME), or one whose parameters and result hold no
    // heap block (scalarDeclaration).
    bool addImported(mlir::StringAttr symbol, Callees &callees);

    // A number, a boolean, a character, or an enum or a literal of one: a value that holds no block
    // and no function of this module, so given to foreign code it hands it neither. Not an opaque
    // pointer, a reference, a string, a class, an interface, an object, a tuple, an array, a union
    // with such a member, a function, `any` or `unknown`.
    static bool scalarType(mlir::Type type);

    // Every parameter and result scalar (a void result too), and no rest parameter.
    static bool scalarSignature(mlir_ts::FunctionType functionType);

    // A body-less `declare function` of C (no `export` or import marker: not another TypeScript
    // module's, whose declarations carry `export`) whose parameters and result are scalar
    // (scalarType): given no block of this module, it can destroy none (controller ruling, spec 31:
    // `sqrt`, `floor`, `pow`, `fabs`). The risk is a C function that calls back into the module
    // (`signal(15, f as Opaque)` then `raise(15)`, review6r1 c3), which it can only do with a
    // function it was given - so the rule is off in a module whose foreign surface could hand it
    // one (functionsEscape) - or an exported function reached by name, which a `@nodrops`
    // declaration takes on as well.
    static bool scalarDeclaration(mlir_ts::FuncOp funcOp);

    // `ts.Load(ts.AddressOf @f)`, where the global `f` holds the address of a function exported
    // by a library loaded at run time, looked up by name: that function, if the library exported
    // its `__own_no_drops`.
    bool addImportedByAddress(mlir::Value callee, Callees &callees);

    Callees resolve(mlir::Operation *op);

    // The function value a call goes through: a method of a class, virtual or not, a symbol, or
    // the `.instanceOf` slot. Anything else - a closure, a function read from a field, an
    // interface's method - is unknown.
    void resolveValue(mlir::Value callee, Callees &callees);

    // A method of a closed interface (interfaceFamilies), `ts.InterfaceSymbolRef(value, index)`, or a
    // function-typed field of one (interfaceFieldFamilies), read and called: the family of the slot
    // after the vtable's header - every function the vtables hold there, or every function the
    // implementing literals' fields are ever given. Open, the call is unknown, as before.
    template <typename Family>
    void resolveInterfaceSlot(mlir_ts::InterfaceSymbolRefOp refOp, const std::map<std::pair<std::string, int64_t>, Family> &families,
                              Callees &callees)
    {
        auto interfaceType = mlir::dyn_cast<mlir_ts::InterfaceType>(refOp.getInterfaceVal().getType());
        auto identifier = mlir::dyn_cast<mlir::StringAttr>(refOp.getIdentifier());
        if (!interfaceType || !identifier)
        {
            return;
        }

        auto position = static_cast<int64_t>(refOp.getIndex()) + INTERFACE_VTABLE_HEADER_SLOTS;
        auto found = families.find({interfaceType.getName().getValue().str(), position});
        if (found == families.end() || !found->second.closed || found->second.field != identifier ||
            found->second.members.empty())
        {
            return;
        }

        callees.named.assign(found->second.members.begin(), found->second.members.end());
        callees.viaInterface = true;
        callees.known = true;
        for (auto member : found->second.members)
        {
            callees.known = addDefined(member, callees) && callees.known;
        }
    }

    // Can a module other than this one call the function? A `public` method (not private, not
    // exported itself) of a class no other module can name - not exported, not a word of the
    // exported declarations, not in the type of an exported or declared function or of an exported
    // or imported global (nameableClasses) - no: another module calls a function of this one through
    // the declaration `__decls` gives it (controller ruling, spec 31). But a method of a class that
    // extends a class another module can name, or one another module defines, yes, private or not:
    // that module calls it through the base's vtable.
    bool callableFromOutside(mlir_ts::FuncOp funcOp);

    // Can foreign code learn of a function of this module, to call it back during a call of a
    // scalar declaration (scalarDeclaration)? It can only through the module's foreign surface, or
    // through a function turned into data (controller ruling, spec 31, fix round 5: a closed-world
    // test, in place of listing the routes a function value takes out). So the rule is on only when:
    // - every foreign symbol is scalar (scalarType): every body-less function (C's, another
    //   module's - `import`, a DLL's - a `declare class`'s, a `@cpp` class's slot; a rest parameter
    //   is not scalar), every global declared with no initializer or imported (a `declare var`, a
    //   `declare class`'s vtable and type name), every `@cpp` class's fields and its bases', with
    //   no method of its own (C++ calls one through the C++ vtable), every function found in a
    //   library at run time (`SearchForAddressOfSymbol`, seen through its cast); an opaque pointer
    //   or a reference anywhere in such a type is a value of foreign memory, not scalar;
    // - no cast makes an opaque pointer or a reference out of a number (`<Opaque>0x1234`, then a
    //   store through it reaches foreign memory);
    // - no cast makes a value of a type that holds no function (`any`, `unknown`, an opaque
    //   pointer, an integer, a reference; a boolean cannot carry one and does not count) out of one
    //   that holds one (`onSig as Opaque`, `k.m as u64`, `<Reference<u64>>k`; a class or object
    //   takes its fields along unless seen as a class or an interface);
    // - no `inline_asm` or `call_intrinsic` is given a value whose type holds a function.
    bool functionsEscape = false;

    // The types a holdsFunction walk has been through, with or without their fields.
    using SeenTypes = llvm::DenseSet<std::pair<mlir::Type, int>>;

    // Does a value of this type hold a function: the type itself, a union's or an optional's
    // member, a tuple's field, an array's element, what a reference points to, an interface whose
    // vtable holds one past its header - a method, a function-typed field, or a field whose type
    // holds one, recursively (an interface with methods, a class with a function field, an array of
    // them) - or whose vtable this module does not make values with (a foreign one: `declare
    // function makeW(): W`), and with `fields` a class's or an object's fields. The header slot
    // (`.instanceOf`, the compiler's own, which destroys nothing) does not count.
    bool holdsFunction(mlir::Type type, bool fields, SeenTypes &seen);

    // The `@cpp` class storage a method's `this` is of, if any: the storage of its first
    // parameter's class, when the method's symbol is `Class.name`.
    mlir_ts::ClassStorageType cppClassOfMethod(mlir_ts::FuncOp funcOp);

    // Does a value of this type carry an address of memory this module made - so a cast of it to
    // an opaque pointer or a reference points into this module, not into foreign memory?
    static bool addressType(mlir::Type type);

    void findEscapingFunctions(mlir::ModuleOp module);

    // class name -> its storage, from every class type a function's signature or a global mentions
    llvm::StringMap<mlir_ts::ClassStorageType> classStorages;

    void collectClassStorages(mlir::ModuleOp module);

    // A class another module can see may be extended there, and its override is a candidate this
    // module never sees: every member has to be private and defined here.
    void resolveFamily(int64_t index, mlir::StringAttr identifier, Callees &callees);

    // `ts.Cast(ts.VTableOffsetRef(vtable, 0))`, where `vtable` is the one an object's first word
    // points to or an interface value's own: slot 0 of either is `.instanceOf` - a class's
    // generated one, a string compare, or `.instanceOf.none` for an object literal behind an
    // interface. Only mlirGenInstanceOfThroughVTable builds a call through it. Neither destroys
    // anything.
    static bool isInstanceOfSlot(mlir::Operation *def);

    // Is every use of this function value the callee of a call - through `ts.GetMethod`, with
    // `ts.GetThis` reading the object beside it?
    static bool onlyCalled(mlir::Value value);

    // The closed world. A function is open when a caller this pass cannot see may reach it: it is
    // not private (another module may call it), or its symbol is used other than by a direct call,
    // a class vtable's entry, or a method reference used only as a callee. A virtual reference
    // that escapes opens its whole family.
    void findOpen(mlir::ModuleOp module);

    void openFamily(int64_t index, mlir::StringAttr symbol, const char *why);

    // ---- Callbacks (spec 33) ----
    //
    // `___array_reduce<T, R>(arr, f, init)` calls its parameter `f`: an indirect call, which reaches
    // whatever function its callers pass. When the function is closed - private, and every use of its
    // symbol a direct call - the parameter is only ever called, and every caller passes there a
    // function this module defines, by its symbol or as a closure over a capture box made right
    // there, a call through the parameter reaches exactly those functions: the parameter's family.
    // Its members are closed too, when their symbols are used only to be called directly or passed
    // so; then the call gets the facts the members agree on (callFacts), and `__own_no_drops` when
    // none of them destroys anything. A closure's body takes its capture box first: its parameter k
    // is the call's argument k - 1 (CallbackMember::shift).
    //
    // A kept parameter is agreed on the other way round from a class family's (meetCallbacks): a
    // member that only borrows a parameter another member keeps is given it to keep too, and releases
    // it on every path (OWN_PARAMS_RELEASED_ATTR_NAME) - `reduce` with a callback that returns its
    // accumulator on some paths and one that never does, in one `___array_reduce<V, V>`. Anything
    // else the members disagree on demotes them all, as a class family's members are.

    struct CallbackMember
    {
        mlir::StringAttr symbol;
        int32_t shift = 0;
    };

    struct CallbackFamily
    {
        llvm::SmallVector<CallbackMember, 2> members;
        // the ops that name a member's symbol to pass it here
        llvm::SmallVector<mlir::Operation *, 2> passing;
        bool closed = true;
    };

    // (function, parameter index) -> the functions its callers pass there
    std::map<std::pair<mlir::Operation *, int32_t>, CallbackFamily> callbackFamilies;
    // the members whose families disagreed (meetFacts), for the error in their bodies
    llvm::DenseSet<mlir::Operation *> demotedCallbacks;
    // uses of a member's symbol that only pass it to a closed family: no escape (findOpen)
    llvm::DenseSet<mlir::Operation *> callbackPassing;

    static bool isFunctionValueType(mlir::Type type);

    // Is every use of this parameter - through its slot, never assigned or captured - the function
    // a call goes through?
    static bool onlyCalledParam(mlir::BlockArgument argument);

    // The function a caller passes as `value`: its symbol, seen as a function value through casts
    // between function types, or a closure over a capture box built of it. Null for anything else.
    static mlir_ts::SymbolRefOp passedFunction(mlir::Value value, int32_t &shift);

    // Does the function value `value` go only where `passes` allows - as an argument of a call - and
    // through casts between function types, a closure built of it, and that closure's own retains
    // and releases?
    static bool passedOnly(mlir::Value value, llvm::function_ref<bool(mlir::OpOperand &)> passes, int depth = 0);

    void collectCallbackFamilies(mlir::ModuleOp module);

    // A call through a parameter whose family is closed: its members.
    void resolveCallback(mlir_ts::LoadOp loadOp, Callees &callees);

    // A member's facts as a call through its family sees them: each parameter k as argument
    // k - shift. False when a fact is about the capture box itself, which the call does not pass.
    static bool shiftFacts(const Facts &facts, int32_t shift, Facts &shifted);

    // Can the body of `funcOp` be given parameter `index` to keep and release it where it does not
    // move it: a plain slot (isParameterSlot), a block, not a handle, not a string?
    bool canReleaseParam(mlir_ts::FuncOp funcOp, int32_t index);

    // The members of each closed callback family agree (the section above), by promotion of a kept
    // parameter or by demotion. True when any member's facts changed.
    bool meetCallbacks(llvm::DenseSet<mlir::Operation *> &demoted);

    // ---- Results that borrow an argument ----

    // The values a function returns: what is stored into the local its returns read (MLIRGen's
    // result slot), or what a return hands back directly.
    static llvm::SmallVector<mlir::Value> returnedValues(mlir_ts::FuncOp funcOp);

    struct ResultBorrows
    {
        llvm::SmallVector<int32_t> borrows;
        bool record = false;
        bool unboxed = false;
    };

    // What the function's result borrows (spec 2.4, 32): the one parameter every heap value it returns
    // is a borrow of, or, when every such value is a borrowing record, the parameters the records
    // borrow. A result that holds no block (`null`, a number) agrees with any; a record on one path
    // and anything else on another is nothing a caller can be told.
    ResultBorrows localResultBorrows(mlir_ts::FuncOp funcOp);

    // The parameter a function returns as it is on some paths, where every other value it returns
    // that holds a block is one it owns (spec 33): `return v.x > 2 ? acc : V.plus(acc, v)`. Such a
    // parameter is kept: the callers move the argument in, the path that returns it moves it out,
    // and every other path releases it. -1 for none - a parameter returned on every path is a
    // borrowed result; two parameters returned, or a part of one (`acc.next`), or a borrowing
    // record, have no such answer; a string is copied instead (spec 22).
    int32_t mixedReturnParam(mlir_ts::FuncOp funcOp);

    // ---- Borrowing records (spec 32) ----
    //
    // `{ thing: this, ray: ray, dist: d }` given as an interface: an object literal's box holding values
    // it does not own beside values it does. Rust's `struct Hit<'a> { thing: &'a Th, .. }`: the box is
    // its owner's - the caller's, when it is returned - and it may not outlive what it borrows. Each box
    // is classified where it is made: the fields it borrows (OWN_RECORD_BORROWED_FIELDS_ATTR_NAME, on the
    // box and on every interface made of it, whose tag then names a release routine that leaves them),
    // and, when every one of them is a parameter, `this`, or a read out of one, those parameters - what
    // a function returning it lends (OWN_RESULT_RECORD_ATTR_NAME). The inference bounds every use.

    struct RecordBox
    {
        llvm::SmallVector<int32_t> params;
        // every borrowed field is a parameter's, `this`'s, a read out of one, or immortal
        bool returnable = true;
        // a tuple built in a local, not boxed, that owns nothing (classifyUnboxedRecord)
        bool unboxed = false;
    };

    llvm::DenseMap<mlir::Operation *, RecordBox> recordBoxes;

    void classifyRecordBoxes();
    void giveBorrowingRecordsTheirVTables(mlir::ModuleOp module);

    void classifyRecordBox(mlir_ts::NewOp newOp);

    // An object literal of tuple type (a `@value` value, a type literal) that is never boxed and holds
    // a borrow beside nothing it owns - every field that holds a block borrows it - is an unboxed
    // borrowing record (spec 36): `{ thing: this, ray: ray, dist: d }` returned as a `@value` Hit. The
    // local it is built in is marked with the fields it borrows (OWN_RECORD_UNBOXED_FIELDS_ATTR_NAME),
    // and each read of it is a record the way a box is (recordBoxes).
    void classifyUnboxedRecord(mlir_ts::VariableOp varOp);

    enum class RecordFlow
    {
        None,   // no borrowing record
        Record, // borrowing records only (or values that hold no block)
        Bad     // a record that borrows what is not a parameter, or one beside anything else
    };

    // Whether `value` - through views, merges and the owning locals that hold it - is a borrowing
    // record, and the parameters it borrows: a box made here, or a call's result that is one, whose
    // arguments are this function's parameters or reads out of them. `unboxed` says whether the
    // records are tuples that own nothing (spec 36); boxes and such tuples together are Bad.
    RecordFlow recordParamsOf(mlir::Value value, llvm::SmallVectorImpl<int32_t> &params, bool *unboxed = nullptr);

    // ---- Parameters the callee keeps ----

    // Does `user` take `used` - a read of a parameter - into something that outlives the call: a
    // store into a field, an element or a global, an insertion into an array, or a call that keeps
    // it in turn? A local that holds it, and the result it is returned as, do not count: the first
    // is a borrow, the second a borrowed result.
    static bool keeps(mlir::Operation *user, mlir::Value used);

    // The parameters the body keeps (spec 2.4, owned-by-callee): a read of the parameter's slot -
    // never assigned - that something keeps.
    static llvm::SmallVector<int32_t> localOwnedParams(mlir_ts::FuncOp funcOp);

    // ---- The facts callers must know ----

    // Owned parameters and a result that borrows only hold where every caller knows them, so the
    // facts survive only on a closed function whose every family agrees, and families are
    // demoted until they do: a member that loses its facts can make another family disagree. A
    // caller that returns what such a call returns, or passes a parameter on to a kept one, gets
    // the fact too, so the facts are recomputed with the calls pinned until nothing changes; if
    // that does not settle, nobody gets any.
    void computeFacts();

    void meetFacts();

    Facts callFacts(const Callees &callees);

    static void setFacts(mlir::Operation *op, const Facts &facts);

    // The calls get the facts their callees agree on; each function gets its own, or the reason it
    // lost those its body has, for the error there.
    void pinFacts(const llvm::DenseMap<mlir::Operation *, Facts> &pinned);

    // ---- Generators that borrow (phase 7b) ----
    //
    // A generator over a parameter that holds a block - `function* each(a: number[])` - keeps the
    // parameter's cell in a box in its state object's `.captured` field. Under 5b's rule that is an
    // escaping closure over a value that is the caller's, an error. Instead the state object
    // borrows: the caller owns it, it may not outlive the argument (OWN_RESULT_BOUNDED_ATTR_NAME),
    // and its release gives back neither the parameter's value nor any field that only ever holds
    // what was read out of it (OWN_BORROWING_FIELDS_ATTR_NAME). A state type is unique to its
    // generator - its `next` field's type names a per-site `object_storage` - so a fact per type is
    // a fact per generator, which is what the per-type release routine needs.

    struct BorrowingState
    {
        mlir_ts::FuncOp maker;
        int32_t captured = -1;
        llvm::SmallVector<int32_t> bounded;
        llvm::SetVector<int32_t> fields;
    };

    static int32_t fieldIndex(mlir_ts::TupleType tupleType, llvm::StringRef name);

    // The box the maker stores into the `.captured` field of the initial state it copies into
    // `newOp`'s block: `ts.Store(box, ts.PropertyRef(state, captured))`, `state` a local whose only
    // read fills the block.
    static mlir::Value boxStoredIntoState(mlir_ts::NewOp newOp, int32_t captured);

    // Is every caller of `funcOp` a direct call this pass sees? It is private, and each use of its
    // symbol is a call, or a closure over it that is only retained and released - `.map(f)` builds
    // one and calls its body straight through the box.
    bool calledOnlyDirectly(mlir::ModuleOp module, mlir_ts::FuncOp funcOp, bool publicToo = false);

    // A parameter's cell: a captured local declared from an argument of `maker`'s entry block, that
    // owns nothing (its value is the caller's).
    static bool isParameterCell(mlir_ts::VariableOp cellOp, mlir_ts::FuncOp maker, int32_t &index);

    // `ts.Load` of one of `maker`'s parameter slots (isParameterSlot): `index` is its position.
    static bool readsParameterSlot(mlir::Value value, mlir_ts::FuncOp maker, int32_t &index);

    // `ts.Load(ts.PropertyRef(argument, i))` of one of `maker`'s parameters: `index` is its position.
    static bool readsParameterField(mlir::Value value, mlir_ts::FuncOp maker, int32_t &index);

    bool ownsHeap(mlir::Location location, mlir::Type type);

    void findBorrowingStates(mlir::ModuleOp module);

    // Is `value` a read of what a state of `tupleType` borrows: the value in one of its box's cells
    // (`ts.Load` of `ts.Load(ts.PropertyRef(ts.Load(ts.PropertyRef(state, captured)), i))`), or a
    // borrowing field's (`ts.Load(ts.PropertyRef(state, p))`)?
    bool readsBorrowed(mlir::Value value, mlir_ts::TupleType tupleType, const BorrowingState &state);

    // ---- Globals given a borrow ----

    // A global initialized with a read of another global, a field or an element (`let k = a`,
    // `let k = h.c`) would be a second owner of a block another owner keeps: the first overwrite
    // of either frees what the other still reads, and both give it back at exit. rc counts the
    // global's own reference (OwnedReturnConsumptionPass, claimGlobalInitializers); own cannot
    // move the value out of its owner, and a global cannot hold a borrow. A handle is counted
    // under own too.
    void rejectBorrowedGlobalInitializers(mlir::ModuleOp module);

    // ---- Drops ----

    // Least fixpoint: a function drops nothing until its body or a call it makes shows it may.
    void computeDrops();

    bool callMayDrop(const Callees &callees);

    // ---- Globals a call may assign ----

    // The globals each call may give a new value, pinned on the call for the inference
    // (OWN_ASSIGNS_GLOBALS_ATTR_NAME, OWN_ASSIGNS_ANY_GLOBAL_ATTR_NAME): a store to a global, or its
    // release, in a callee's body, or in a call it makes. A call that may drop and whose callees are
    // not known may assign any; one that drops nothing assigns none. A read of a global ends where
    // the global is given a new value (checkGlobalRead), and a call that does so ends it as surely as
    // a store in the same function: `const c = G; reset(); print(c.x)`.
    void pinAssignedGlobals();

    // An argument read out of a global (`run(G)`, `G.m()`) is lent to the callee, and the callee may
    // free it under its own parameter: `function run(h: C) { G = mk(); print(h.x); }`. The caller's
    // borrow ends at the call that takes it, so nothing in the caller sees that, and in the callee
    // the parameter is not known to be the global's. A call that may drop is an error for such an
    // argument, when something assigns that global. (A field of a global given that way is a
    // borrow of a place, which the inference already bounds by the call.)
    void rejectGlobalArgumentsDropped();

    // ---- Calls in a global's initializer ----

    // A global's region is no function: its calls get no facts pinned, and the inference never sees
    // it, so its calls were read with the defaults - an owned result, borrowed arguments. A result
    // that borrows an argument (`const c = child(n)`, `id(new V(1))`) then made the global a second
    // owner of a block its argument's owner keeps, or kept one the region freed; and a fresh
    // argument a callee keeps (`new N(new V(4))`, the constructor keeping `c`) was released at the
    // end of the region, under the object that kept it. A kept argument made in the region and used
    // by nothing after the call moves into the callee: its release goes. Anything else is an error.
    void ownGlobalInitializerCalls(mlir::ModuleOp module);

    // `value` was made in the call's block and nothing but its release, if it has one, comes after
    // the call: the release goes, and the callee owns the block. Constant data (a string literal)
    // is nobody's, and is given as it is.
    static bool moveIntoCallee(mlir::Operation *call, mlir::Value value);

    // ---- Export ----

    // A library says, beside its memory-model marker, which of its exported functions destroy
    // nothing (SHARED_LIB_OWN_FACTS). The global is a copy of the marker `__tsmm_own_<module>`
    // under another name and with another value, so it is exported exactly as the marker is, and
    // MLIRGen emits the same under own as under rc. No marker, nothing is exported.
    void exportNoDrops();

    // Does the body itself destroy something a caller may reach: overwrite a field or an element
    // of a block it did not make, assign a global, remove elements from an array it did not make,
    // or `delete`? A local's own releases are not drops, nor is the constructor's filling of the
    // object it is building.
    bool dropsInBody(mlir_ts::FuncOp funcOp);

    // Can the block this reference points into be one the function's caller can reach? Walks up
    // from a place to what holds it. Not when every way up ends in a block the function made (an
    // allocation, a call's result it owns) or in an owning local that only ever held such blocks,
    // or, in a constructor, in the object being built.
    bool reachesOutside(mlir::Value ref, mlir_ts::FuncOp funcOp);

    static bool isAssigned(mlir_ts::VariableOp varOp);

    // An owning, uncaptured local whose every value - its initializer and each assignment - is one
    // the function made.
    static bool holdsOnlyFresh(mlir_ts::VariableOp varOp);
};

} // namespace own_signature

#endif // MLIR_TYPESCRIPT_OWNERSHIPSIGNATUREPASS_H
