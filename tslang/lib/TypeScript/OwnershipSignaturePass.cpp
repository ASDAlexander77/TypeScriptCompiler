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
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringSet.h"

#include <map>

#define DEBUG_TYPE "own"

namespace mlir_ts = mlir::typescript;

namespace
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

    // A global is a root (rc 5ai): what its initializer makes moves into it, and nothing gives it
    // back. rc retains a fresh value for its `ts.GlobalResult`, and that retain is the move: it
    // goes. The per-function inference never sees a global's region, so any other retain there is
    // reported here, by the global's name, rather than left to the lowering's "left a retain
    // behind".
    // Every use of `value` ends in the global's `ts.GlobalResult`: given to it, or an element of a
    // fresh array literal that is (`const nested = [[1, 2], [3]]`). Its retains are judged on their
    // own.
    static bool movesIntoResult(mlir::Value value)
    {
        return llvm::all_of(value.getUses(), [](mlir::OpOperand &use) {
            auto *user = use.getOwner();
            if (mlir::isa<mlir_ts::RetainOp, mlir_ts::GlobalResultOp>(user))
            {
                return true;
            }

            if ((isView(user) && use.getOperandNumber() == 0) || mlir::isa<mlir_ts::CreateArrayOp>(user))
            {
                return movesIntoResult(user->getResult(0));
            }

            // copied into a block `ts.New` made, which itself moves in: an object literal's box
            if (auto storeOp = mlir::dyn_cast<mlir_ts::StoreOp>(user); storeOp && use.getOperandNumber() == 0)
            {
                auto block = storeOp.getReference();
                return block.getDefiningOp<mlir_ts::NewOp>() &&
                       llvm::all_of(block.getUses(), [&](mlir::OpOperand &blockUse) {
                           return blockUse.getOwner() == storeOp.getOperation() ||
                                  (isView(blockUse.getOwner()) && movesIntoResult(blockUse.getOwner()->getResult(0)));
                       });
            }

            return false;
        });
    }

    // A record copied out of a local a `ts.Constant` initializes, whose owning fields are given
    // nothing that holds a block: it holds constants only - numbers, immortal string literals,
    // functions - and nobody else's reference.
    static bool holdsOnlyConstants(mlir::Value value)
    {
        auto loadOp = value.getDefiningOp<mlir_ts::LoadOp>();
        auto varOp = loadOp ? loadOp.getReference().getDefiningOp<mlir_ts::VariableOp>() : mlir_ts::VariableOp();
        if (!varOp || !varOp.getInitializer() || !varOp.getInitializer().getDefiningOp<mlir_ts::ConstantOp>())
        {
            return false;
        }

        return llvm::all_of(varOp->getUsers(), [](mlir::Operation *user) {
            if (mlir::isa<mlir_ts::LoadOp>(user))
            {
                return true;
            }

            auto propertyRefOp = mlir::dyn_cast<mlir_ts::PropertyRefOp>(user);
            return propertyRefOp && llvm::all_of(propertyRefOp->getUsers(), [&](mlir::Operation *fieldUser) {
                       auto storeOp = mlir::dyn_cast<mlir_ts::StoreOp>(fieldUser);
                       return mlir::isa<mlir_ts::LoadOp>(fieldUser) ||
                              (storeOp && storeOp.getReference() == propertyRefOp.getResult() &&
                               holdsNoBlock(storeOp.getValue()));
                   });
        });
    }

    void moveIntoGlobals(mlir::ModuleOp module)
    {
        module.walk([&](mlir_ts::GlobalOp globalOp) {
            llvm::SmallVector<mlir_ts::RetainOp> retains;
            globalOp.walk([&](mlir_ts::RetainOp retainOp) { retains.push_back(retainOp); });
            for (auto retainOp : retains)
            {
                // a handle's retain is rc's: the global's count, which own leaves in place (spec 23.2)
                if (touchesHandle(retainOp))
                {
                    continue;
                }

                auto value = retainOp.getReference();

                // nothing in the region gives the block back, or takes a second reference to it:
                // the global is its one owner
                auto released = false;
                auto retained = 0;
                forEachUse(rootOf(value), [&](mlir::Operation *user, mlir::Value) {
                    released = released || mlir::isa<mlir_ts::ReleaseOp, mlir_ts::ReleaseSlotOp>(user);
                    retained += mlir::isa<mlir_ts::RetainOp>(user);
                });

                if (retained == 1 && !released && (isFresh(value) || holdsOnlyConstants(value) || isConstantData(value)) && movesIntoResult(value))
                {
                    retainOp.erase();
                    continue;
                }

                retainOp.emitError("the initializer of global '")
                    << globalOp.getSymName() << "' takes a second reference; -mm=own cannot prove a move or a borrow here yet";
                signalPassFailure();
            }
        });
    }

    void runOnOperation() override
    {
        auto module = getOperation();
        module.walk([&](mlir_ts::FuncOp funcOp) { functions[funcOp.getSymName()] = funcOp; });
        module.walk([&](mlir_ts::GlobalOp globalOp) { globals[globalOp.getSymName()] = globalOp; });
        module.walk([&](mlir_ts::AddressOfOp addressOfOp) {
            auto onlyLoaded = llvm::all_of(addressOfOp->getUsers(), [](mlir::Operation *user) {
                return mlir::isa<mlir_ts::LoadOp>(user);
            });
            if (!onlyLoaded)
            {
                writtenGlobals.insert(addressOfOp.getGlobalName());
            }
        });
        if (auto names = module->getAttrOfType<mlir::ArrayAttr>(SHARED_LIB_OWN_NO_DROPS_ATTR_NAME))
        {
            for (auto name : names.getAsRange<mlir::StringAttr>())
            {
                importedNoDrops.insert(name.getValue());
            }
        }

        moveIntoGlobals(module);
        rejectBorrowedGlobalInitializers(module);
        collectClassVTables(module);
        fieldFunctions.collect(module);

        module.walk([&](mlir::Operation *op) {
            // in any function: a `ts.Func`, or the `func.func` the async lowering outlines a body into
            if (isCall(op) && op->getParentOfType<mlir::FunctionOpInterface>())
            {
                calls.push_back({op, resolve(op)});
            }
        });

        findOpen(module);
        computeFacts();
        computeDrops();
        findBorrowingStates(module);
        findBoundedForwarders();
        findYieldedFields(module);
        findImportedStates(module);
        checkDeclared();

        for (auto &call : calls)
        {
            // a generator another module defines, whose signature says what its state borrows
            if (auto bounded = declaredByCallees(call.callees, OWN_DECLARED_BORROWS_ATTR_NAME);
                !bounded.empty() && call.op->getNumResults() == 1)
            {
                call.op->setAttr(OWN_RESULT_BOUNDED_ATTR_NAME, mlir::DenseI32ArrayAttr::get(&getContext(), bounded));
                // and the caller owns it (the generator's state), as `@borrows` says: fresh, though
                // the function may be another module's
                call.op->setAttr(OWN_FRESH_RESULT_ATTR_NAME, mlir::UnitAttr::get(&getContext()));
            }

            // an iterator's `next()`, whoever implements it, and a generator's state, whoever makes
            // it: see isIteratorNext
            if (isIteratorNext(call.op) || makesGeneratorState(call.op))
            {
                call.op->setAttr(OWN_FRESH_RESULT_ATTR_NAME, mlir::UnitAttr::get(&getContext()));
            }

            if (!call.callees.known)
            {
                continue;
            }

            if (call.callees.instanceOf || call.callees.imported ||
                llvm::all_of(call.callees.funcs, [&](mlir_ts::FuncOp callee) { return noDrops.contains(callee); }))
            {
                call.op->setAttr(OWN_NO_DROPS_ATTR_NAME, mlir::UnitAttr::get(&getContext()));
            }

            // `new C()` of a class from a library built under own: see OWN_FRESH_RESULT_ATTR_NAME
            if (call.callees.imported && call.callees.importedName.ends_with("." NEW_METHOD_NAME) &&
                call.op->getNumResults() == 1)
            {
                call.op->setAttr(OWN_FRESH_RESULT_ATTR_NAME, mlir::UnitAttr::get(&getContext()));
            }

            // a generator's `next`, read out of its state object: a function defined here, which
            // returns its result retained like any other (its symbol sits in a constant, so it is
            // open, and its body may not return a borrow)
            if (call.callees.fieldFunction && call.op->getNumResults() == 1)
            {
                call.op->setAttr(OWN_FRESH_RESULT_ATTR_NAME, mlir::UnitAttr::get(&getContext()));
            }

            // a generator whose state object borrows from its arguments
            if (call.callees.funcs.size() == 1 && !call.callees.imported)
            {
                if (auto bounded = call.callees.funcs.front()->getAttrOfType<mlir::DenseI32ArrayAttr>(OWN_RESULT_BOUNDED_ATTR_NAME))
                {
                    call.op->setAttr(OWN_RESULT_BOUNDED_ATTR_NAME, bounded);
                }
            }
        }

        exportNoDrops();

        module.walk([&](mlir_ts::CreateBoundFunctionOp boundOp) {
            if (!boundOp->hasAttr(OWNS_CAPTURE_ATTR_NAME))
            {
                return;
            }

            auto assigned = assignedCaptures(closureBody(boundOp));
            if (!assigned.empty())
            {
                boundOp->setAttr(OWN_ASSIGNS_CAPTURES_ATTR_NAME,
                                 mlir::DenseI32ArrayAttr::get(&getContext(), assigned.getArrayRef()));
            }
        });
    }

  private:
    // The function a closure runs, when it names one defined here.
    mlir_ts::FuncOp closureBody(mlir_ts::CreateBoundFunctionOp boundOp)
    {
        auto symbolRefOp = boundOp.getFunc().getDefiningOp<mlir_ts::SymbolRefOp>();
        if (!symbolRefOp)
        {
            return {};
        }

        auto found = functions.find(symbolRefOp.getIdentifier());
        return found != functions.end() && !found->second.isDeclaration() ? found->second : mlir_ts::FuncOp();
    }

    // The fields of a closure body's capture box whose cell it assigns (OWN_ASSIGNS_CAPTURES_ATTR_NAME),
    // directly or through a closure it builds over the cell. Every field, when the body is unknown or
    // a cell goes anywhere this does not follow.
    llvm::SetVector<int32_t> assignedCaptures(mlir_ts::FuncOp funcOp, int depth = 0)
    {
        llvm::SetVector<int32_t> assigned;
        auto boxType = funcOp && funcOp.getNumArguments() > 0 && !funcOp.getBody().empty()
                           ? mlir::dyn_cast<mlir_ts::RefType>(funcOp.getArgument(0).getType())
                           : mlir_ts::RefType();
        auto tupleType = boxType ? mlir::dyn_cast<mlir_ts::TupleType>(boxType.getElementType()) : mlir_ts::TupleType();
        if (!tupleType || depth > 16)
        {
            // unknown: a box of any shape may have had any of its cells assigned
            for (int32_t index = 0; index < 64; ++index)
            {
                assigned.insert(index);
            }

            return assigned;
        }

        if (auto found = assignedCache.find(funcOp); found != assignedCache.end())
        {
            return found->second;
        }

        for (auto *user : funcOp.getArgument(0).getUsers())
        {
            auto propertyRefOp = mlir::dyn_cast<mlir_ts::PropertyRefOp>(user);
            if (!propertyRefOp)
            {
                continue;
            }

            auto field = propertyRefOp.getPosition();
            for (auto *fieldUser : propertyRefOp->getUsers())
            {
                auto loadOp = mlir::dyn_cast<mlir_ts::LoadOp>(fieldUser);
                if (!loadOp || !isLoadedCell(loadOp.getResult()))
                {
                    continue; // a field captured by value: a copy, not a variable
                }

                if (cellAssigned(loadOp.getResult(), depth))
                {
                    assigned.insert(field);
                }
            }
        }

        assignedCache[funcOp] = assigned;
        return assigned;
    }

    // Is the variable behind this cell assigned - here, or by a closure built over it?
    bool cellAssigned(mlir::Value cell, int depth)
    {
        for (auto &use : cell.getUses())
        {
            auto *user = use.getOwner();
            if (mlir::isa<mlir_ts::LoadOp, mlir_ts::RetainCellOp>(user))
            {
                continue;
            }

            if (mlir::isa<mlir_ts::ReleaseSlotOp>(user))
            {
                return true;
            }

            auto storeOp = mlir::dyn_cast<mlir_ts::StoreOp>(user);
            if (!storeOp)
            {
                return true; // the cell goes somewhere this does not follow
            }

            if (storeOp.getReference() == cell)
            {
                return true;
            }

            // captured again, by a closure built here: the field it fills in that closure's box
            auto propertyRefOp = storeOp.getReference().getDefiningOp<mlir_ts::PropertyRefOp>();
            auto boundOp = propertyRefOp ? closureOfBox(propertyRefOp.getObjectRef()) : mlir_ts::CreateBoundFunctionOp();
            if (!boundOp || assignedCaptures(closureBody(boundOp), depth + 1).contains(propertyRefOp.getPosition()))
            {
                return true;
            }
        }

        return false;
    }

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

    // The parameter a function's result borrows and the parameters it keeps (spec 2.4); absent is
    // rc's convention.
    struct Facts
    {
        int borrows = -1;
        llvm::SmallVector<int32_t> owned;

        bool empty() const
        {
            return borrows < 0 && owned.empty();
        }

        bool operator==(const Facts &other) const
        {
            return borrows == other.borrows && owned == other.owned;
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

    // ---- Facts a signature says (`@keeps`, `@borrows`, `@lends`) ----
    //
    // A function that says them has them whoever calls it: a caller in another module reads them
    // from the declaration it re-parses. Its body is checked against them instead (checkDeclared).

    static llvm::ArrayRef<int32_t> declared(mlir::Operation *funcOp, llvm::StringRef name)
    {
        if (auto attr = funcOp->getAttrOfType<mlir::DenseI32ArrayAttr>(name))
        {
            return attr.asArrayRef();
        }

        return {};
    }

    static bool declaresFacts(mlir::Operation *funcOp)
    {
        return funcOp->hasAttr(OWN_DECLARED_KEEPS_ATTR_NAME) || funcOp->hasAttr(OWN_DECLARED_BORROWS_ATTR_NAME) ||
               funcOp->hasAttr(OWN_DECLARED_LENDS_ATTR_NAME);
    }

    // The facts `@keeps` and `@lends` say (Facts): what callers move in, and what the result is a
    // borrow of.
    static bool declaresCallerFacts(mlir::Operation *funcOp)
    {
        return funcOp->hasAttr(OWN_DECLARED_KEEPS_ATTR_NAME) || funcOp->hasAttr(OWN_DECLARED_LENDS_ATTR_NAME);
    }

    static int declaredLends(mlir::Operation *funcOp)
    {
        auto lends = declared(funcOp, OWN_DECLARED_LENDS_ATTR_NAME);
        return lends.empty() ? -1 : lends.front();
    }

    // What every function a call names declares under `name`, when they all declare the same;
    // nothing when one of them does not.
    llvm::SmallVector<int32_t> declaredByCallees(const Callees &callees, llvm::StringRef name)
    {
        llvm::SmallVector<int32_t> agreed;
        for (auto [index, symbol] : llvm::enumerate(callees.named))
        {
            // a function, or the global holding the address of one a library exports (a dynamic
            // import), which MLIRGen gave the declaration's markers
            mlir::Operation *declaring = functions.lookup(symbol.getValue());
            if (!declaring)
            {
                declaring = globals.lookup(symbol.getValue());
            }

            if (!declaring || !declaring->hasAttr(name))
            {
                return {};
            }

            auto indices = declared(declaring, name);
            if (index == 0)
            {
                agreed.assign(indices.begin(), indices.end());
            }
            else if (llvm::ArrayRef<int32_t>(agreed) != indices)
            {
                return {};
            }
        }

        return agreed;
    }

    // The parameters, for an error, by the names MLIRGen recorded beside the marker.
    static std::string describeParams(mlir_ts::FuncOp funcOp, llvm::ArrayRef<int32_t> indices)
    {
        if (indices.empty())
        {
            return "nothing";
        }

        auto names = funcOp->getAttrOfType<mlir::ArrayAttr>(OWN_DECLARED_PARAM_NAMES_ATTR_NAME);
        std::string text;
        llvm::raw_string_ostream os(text);
        for (auto [position, index] : llvm::enumerate(indices))
        {
            os << (position ? ", " : "") << "'";
            if (names && static_cast<size_t>(index) < names.size())
            {
                os << mlir::cast<mlir::StringAttr>(names[index]).getValue();
            }
            else
            {
                os << "#" << index;
            }

            os << "'";
        }

        return os.str();
    }

    // Functions this module defines that `@borrows`, and those whose state was found to borrow.
    llvm::DenseMap<mlir::Operation *, llvm::SmallVector<int32_t>> borrowsFound;

    // The argument of `funcOp` that `value` is, or a field of: its own value, its slot's, or a
    // read out of either. -1 for anything else.
    static int32_t parameterOf(mlir::Value value, mlir_ts::FuncOp funcOp)
    {
        auto root = rootOf(value);
        if (auto loadOp = root.getDefiningOp<mlir_ts::LoadOp>())
        {
            if (auto propertyRefOp = loadOp.getReference().getDefiningOp<mlir_ts::PropertyRefOp>())
            {
                return parameterOf(propertyRefOp->getOperand(0), funcOp);
            }

            auto varOp = loadOp.getReference().getDefiningOp<mlir_ts::VariableOp>();
            int index = -1;
            if (varOp && isParameterSlot(varOp, index) && varOp->getParentOfType<mlir_ts::FuncOp>() == funcOp)
            {
                return index;
            }

            return -1;
        }

        auto argument = mlir::dyn_cast<mlir::BlockArgument>(root);
        if (argument && argument.getOwner()->isEntryBlock() && argument.getOwner()->getParentOp() == funcOp.getOperation())
        {
            return static_cast<int32_t>(argument.getArgNumber());
        }

        return -1;
    }

    // ---- The iterator protocol ----
    //
    // `it.next()` returns a `{value, done}` record, and under own the caller owns what it holds,
    // whoever implements it: a generator's `next` moves what it yields into the record (a yield of
    // a borrow is an error in its body), and any other `next` is a function whose callers this pass
    // cannot all see, which may not return a borrow. So the result is fresh even when the call goes
    // through the `Iterator` interface, or into another module - the default library's `matchAll`.

    // A call of a function named `next` - read through an interface, out of a field, or by its
    // symbol - whose one result is a `{value, done}` record.
    static bool isIteratorNext(mlir::Operation *op)
    {
        if (op->getNumResults() != 1)
        {
            return false;
        }

        auto tupleType = mlir::dyn_cast<mlir_ts::TupleType>(op->getResult(0).getType());
        if (!tupleType || tupleType.getFields().size() != 2)
        {
            return false;
        }

        auto fieldName = [&](int index) {
            auto id = mlir::dyn_cast_or_null<mlir::StringAttr>(tupleType.getFields()[index].id);
            return id ? id.getValue() : llvm::StringRef();
        };
        if (fieldName(0) != "value" || fieldName(1) != "done")
        {
            return false;
        }

        auto isNext = [](llvm::StringRef name) { return name == "next" || name.ends_with(".next"); };
        if (auto callee = directCallee(op))
        {
            return isNext(callee.getValue());
        }

        auto function = calleeValue(op);
        if (!function)
        {
            return false;
        }

        if (auto getMethodOp = function.getDefiningOp<mlir_ts::GetMethodOp>())
        {
            function = getMethodOp.getBoundFunc();
        }

        if (auto boundOp = function.getDefiningOp<mlir_ts::CreateBoundFunctionOp>())
        {
            function = boundOp.getFunc();
        }

        if (auto loadOp = function.getDefiningOp<mlir_ts::LoadOp>())
        {
            if (auto refOp = loadOp.getReference().getDefiningOp<mlir_ts::InterfaceSymbolRefOp>())
            {
                auto identifier = mlir::dyn_cast<mlir::StringAttr>(refOp.getIdentifier());
                return identifier && isNext(identifier.getValue());
            }

            if (auto propertyRefOp = loadOp.getReference().getDefiningOp<mlir_ts::PropertyRefOp>())
            {
                auto storageType = storageTupleOfType(propertyRefOp->getOperand(0).getType());
                if (!storageType || propertyRefOp.getPosition() >= storageType.getFields().size())
                {
                    return false;
                }

                auto id = mlir::dyn_cast_or_null<mlir::StringAttr>(storageType.getFields()[propertyRefOp.getPosition()].id);
                return id && isNext(id.getValue());
            }

            return false;
        }

        if (auto refOp = function.getDefiningOp<mlir_ts::ThisVirtualSymbolRefOp>())
        {
            return isNext(refOp.getIdentifier());
        }

        if (auto refOp = function.getDefiningOp<mlir_ts::VirtualSymbolRefOp>())
        {
            return isNext(refOp.getIdentifier());
        }

        if (auto refOp = function.getDefiningOp<mlir_ts::ThisSymbolRefOp>())
        {
            return isNext(refOp.getIdentifier());
        }

        if (auto refOp = function.getDefiningOp<mlir_ts::SymbolRefOp>())
        {
            return isNext(refOp.getIdentifier());
        }

        return false;
    }

    // A generator another module made: its state object's type is only its shape here, and what each
    // field owns - a local, a yielded value now the caller's, what it borrows from its arguments - is
    // known in that module alone. A release routine made here from the shape would give back what is
    // not the state's. So this module's gives back none of it: only the `.captured` box's cells, and
    // their values too unless the maker says it borrows them (`@borrows`). What the generator's own
    // locals still hold when it is given up early leaks rather than being freed twice.
    void findImportedStates(mlir::ModuleOp module)
    {
        llvm::MapVector<mlir::Type, bool> states;
        for (auto &call : calls)
        {
            if (!makesGeneratorState(call.op) || (call.callees.known && !call.callees.funcs.empty()))
            {
                continue;
            }

            auto tupleType = storageTupleOfType(call.op->getResult(0).getType());
            auto borrows = !declaredByCallees(call.callees, OWN_DECLARED_BORROWS_ATTR_NAME).empty();
            states[tupleType] = states.lookup(tupleType) || borrows;
        }

        if (states.empty())
        {
            return;
        }

        auto *context = &getContext();
        llvm::SmallVector<mlir::Attribute> entries;
        if (auto existing = module->getAttrOfType<mlir::ArrayAttr>(OWN_BORROWING_FIELDS_ATTR_NAME))
        {
            entries.append(existing.begin(), existing.end());
        }

        for (auto &[type, borrows] : states)
        {
            auto tupleType = mlir::cast<mlir_ts::TupleType>(type);
            auto captured = fieldIndex(tupleType, CAPTURED_NAME);
            // the box first, when it borrows; -1 names no field, so a box that owns is released
            llvm::SmallVector<int32_t> fields{borrows && captured >= 0 ? captured : -1};
            for (auto [index, field] : llvm::enumerate(tupleType.getFields()))
            {
                if (static_cast<int32_t>(index) != captured && ownsHeap(module.getLoc(), field.type))
                {
                    fields.push_back(static_cast<int32_t>(index));
                }
            }

            entries.push_back(mlir::ArrayAttr::get(context, {mlir::TypeAttr::get(type), mlir::DenseI32ArrayAttr::get(context, fields)}));
        }

        module->setAttr(OWN_BORROWING_FIELDS_ATTR_NAME, mlir::ArrayAttr::get(context, entries));
    }

    // A call whose result is a generator's state object (an object whose fields are `.step` and
    // `next`): a generator function made it, a `ts.New` block it hands over (spec 18.1), which the
    // caller owns - in another module too, as the iteration that follows needs.
    static bool makesGeneratorState(mlir::Operation *op)
    {
        if (op->getNumResults() != 1 || !mlir::isa<mlir_ts::ObjectType>(op->getResult(0).getType()))
        {
            return false;
        }

        auto tupleType = storageTupleOfType(op->getResult(0).getType());
        if (!tupleType)
        {
            return false;
        }

        auto hasStep = false;
        auto hasNext = false;
        for (auto field : tupleType.getFields())
        {
            if (auto id = mlir::dyn_cast_or_null<mlir::StringAttr>(field.id))
            {
                hasStep = hasStep || id.getValue() == ".step";
                hasNext = hasNext || id.getValue() == "next";
            }
        }

        return hasStep && hasNext;
    }

    // storageTupleOf, without the pass's context: an object's storage, a reference's or a tuple
    static mlir_ts::TupleType storageTupleOfType(mlir::Type type)
    {
        if (auto objectType = mlir::dyn_cast<mlir_ts::ObjectType>(type))
        {
            if (auto storageType = mlir::dyn_cast<mlir_ts::ObjectStorageType>(objectType.getStorageType()))
            {
                return mlir_ts::TupleType::get(type.getContext(), storageType.getFields());
            }

            // an object a declaration names by its shape (another module's generator)
            return mlir::dyn_cast<mlir_ts::TupleType>(objectType.getStorageType());
        }

        if (auto refType = mlir::dyn_cast<mlir_ts::RefType>(type))
        {
            type = refType.getElementType();
        }
        else if (auto valueRefType = mlir::dyn_cast<mlir_ts::ValueRefType>(type))
        {
            type = valueRefType.getElementType();
        }

        return mlir::dyn_cast<mlir_ts::TupleType>(type);
    }

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
    static bool isRecordFieldStore(mlir::Operation *user, mlir::Value value)
    {
        auto storeOp = mlir::dyn_cast<mlir_ts::StoreOp>(user);
        auto propertyRefOp = storeOp && storeOp.getValue() == value
                                 ? storeOp.getReference().getDefiningOp<mlir_ts::PropertyRefOp>()
                                 : mlir_ts::PropertyRefOp();
        auto recordOp = propertyRefOp ? propertyRefOp->getOperand(0).getDefiningOp<mlir_ts::VariableOp>() : mlir_ts::VariableOp();
        return recordOp && recordOp.getInitializer() && recordOp.getInitializer().getDefiningOp<mlir_ts::ConstantOp>();
    }

    void findYieldedFields(mlir::ModuleOp module)
    {
        // (state type, field) -> its reads and its stores
        struct FieldUses
        {
            llvm::SmallVector<mlir_ts::LoadOp> reads;
            llvm::SmallVector<mlir_ts::StoreOp> stores;
            bool other = false;
        };

        llvm::MapVector<std::pair<mlir::Type, int32_t>, FieldUses> fields;
        module.walk([&](mlir_ts::PropertyRefOp propertyRefOp) {
            auto tupleType = storageTupleOf(propertyRefOp->getOperand(0).getType());
            auto position = static_cast<int32_t>(propertyRefOp.getPosition());
            if (!tupleType || fieldIndex(tupleType, ".step") < 0 || position == fieldIndex(tupleType, CAPTURED_NAME) ||
                !mlir::isa<mlir_ts::RefType>(propertyRefOp.getType()) ||
                !ownsHeap(propertyRefOp.getLoc(), mlir::cast<mlir_ts::RefType>(propertyRefOp.getType()).getElementType()))
            {
                return;
            }

            auto &uses = fields[{tupleType, position}];
            for (auto *user : propertyRefOp->getUsers())
            {
                if (auto loadOp = mlir::dyn_cast<mlir_ts::LoadOp>(user))
                {
                    uses.reads.push_back(loadOp);
                }
                else if (auto storeOp = mlir::dyn_cast<mlir_ts::StoreOp>(user); storeOp && storeOp.getReference() == propertyRefOp.getResult())
                {
                    uses.stores.push_back(storeOp);
                }
                else if (!mlir::isa<mlir_ts::ReleaseSlotOp, mlir_ts::RetainSlotOp>(user))
                {
                    uses.other = true;
                }
            }
        });

        // a field the signature pass already found borrowing is the caller's, not the generator's
        llvm::DenseSet<std::pair<mlir::Type, int32_t>> borrowing;
        if (auto entries = module->getAttrOfType<mlir::ArrayAttr>(OWN_BORROWING_FIELDS_ATTR_NAME))
        {
            for (auto entry : entries.getAsRange<mlir::ArrayAttr>())
            {
                for (auto index : mlir::cast<mlir::DenseI32ArrayAttr>(entry[1]).asArrayRef())
                {
                    borrowing.insert({mlir::cast<mlir::TypeAttr>(entry[0]).getValue(), index});
                }
            }
        }

        llvm::MapVector<mlir::Type, llvm::SmallVector<int32_t>> yielded;
        for (auto &[key, uses] : fields)
        {
            if (uses.other || uses.reads.empty() || uses.stores.empty() || borrowing.contains(key))
            {
                continue;
            }

            // read only to be put into a yield's record
            auto onlyYielded = llvm::all_of(uses.reads, [](mlir_ts::LoadOp loadOp) {
                auto users = llvm::make_filter_range(loadOp->getUsers(), [](mlir::Operation *user) {
                    return !mlir::isa<mlir_ts::RetainOp, mlir_ts::DebugVariableOp>(user);
                });
                return llvm::hasSingleElement(users) && isRecordFieldStore(*users.begin(), loadOp.getResult());
            });

            // each value stored is yielded later in its block: nothing between stops the generator
            auto eachYielded = llvm::all_of(uses.stores, [&](mlir_ts::StoreOp storeOp) {
                return llvm::any_of(uses.reads, [&](mlir_ts::LoadOp loadOp) {
                    return loadOp->getBlock() == storeOp->getBlock() && storeOp->isBeforeInBlock(loadOp);
                });
            });

            if (!onlyYielded || !eachYielded)
            {
                continue;
            }

            yielded[key.first].push_back(key.second);
            for (auto loadOp : uses.reads)
            {
                loadOp->setAttr(OWN_FRESH_RESULT_ATTR_NAME, mlir::UnitAttr::get(&getContext()));
            }
        }

        if (yielded.empty())
        {
            return;
        }

        auto *context = &getContext();
        llvm::SmallVector<mlir::Attribute> entries;
        for (auto &[type, indices] : yielded)
        {
            entries.push_back(mlir::ArrayAttr::get(context, {mlir::TypeAttr::get(type), mlir::DenseI32ArrayAttr::get(context, indices)}));
        }

        module->setAttr(OWN_YIELDED_FIELDS_ATTR_NAME, mlir::ArrayAttr::get(context, entries));
    }

    // A function that says `@borrows` and makes no state itself, but returns what a call that has
    // a bounded result gives it (`return this.value.matchAll(expr)`): it borrows the parameters
    // the call's bounded arguments are, or are read out of.
    void findBoundedForwarders()
    {
        llvm::DenseMap<mlir::Operation *, const Callees *> calleesOf;
        for (auto &call : calls)
        {
            calleesOf[call.op] = &call.callees;
        }

        for (auto &entry : functions)
        {
            auto funcOp = entry.second;
            if (funcOp.isDeclaration() || !funcOp->hasAttr(OWN_DECLARED_BORROWS_ATTR_NAME) || borrowsFound.count(funcOp))
            {
                continue;
            }

            llvm::SmallVector<int32_t> found;
            auto forwards = true;
            for (auto value : returnedValues(funcOp))
            {
                if (holdsNoBlock(value))
                {
                    continue;
                }

                auto *call = rootOf(value).getDefiningOp();
                auto known = call ? calleesOf.find(call) : calleesOf.end();
                if (known == calleesOf.end())
                {
                    forwards = false;
                    break;
                }

                llvm::SmallVector<int32_t> bounded = declaredByCallees(*known->second, OWN_DECLARED_BORROWS_ATTR_NAME);
                if (bounded.empty() && known->second->funcs.size() == 1)
                {
                    auto made = declared(known->second->funcs.front(), OWN_RESULT_BOUNDED_ATTR_NAME);
                    bounded.assign(made.begin(), made.end());
                }

                if (bounded.empty())
                {
                    forwards = false;
                    break;
                }

                auto args = callArgs(call);
                for (auto index : bounded)
                {
                    auto param = static_cast<size_t>(index) < args.size() ? parameterOf(args[index], funcOp) : -1;
                    if (param < 0)
                    {
                        forwards = false;
                        break;
                    }

                    if (!llvm::is_contained(found, param))
                    {
                        found.push_back(param);
                    }
                }
            }

            if (forwards && !found.empty())
            {
                llvm::sort(found);
                borrowsFound[funcOp] = found;
            }
        }
    }

    // A body that does other than its signature says, and an override that says other than the
    // method it overrides: either way a caller that trusts the signature frees twice or leaks.
    void checkDeclared()
    {
        for (auto &entry : functions)
        {
            auto funcOp = entry.second;
            if (funcOp.isDeclaration())
            {
                continue;
            }

            if (funcOp->hasAttr(OWN_DECLARED_LENDS_ATTR_NAME) && funcOp->hasAttr(OWN_DECLARED_BORROWS_ATTR_NAME))
            {
                funcOp.emitError("'") << funcOp.getSymName()
                                      << "' cannot say both '@" LENDS_DECORATOR "' and '@" BORROWS_DECORATOR
                                         "': its caller owns a result that borrows, and nothing of one it is lent";
                signalPassFailure();
                continue;
            }

            if (declaresCallerFacts(funcOp))
            {
                auto local = factsLocal.lookup(funcOp);
                auto keeps = declared(funcOp, OWN_DECLARED_KEEPS_ATTR_NAME);
                auto lends = declaredLends(funcOp);
                if (local.borrows != lends)
                {
                    if (lends < 0)
                    {
                        funcOp.emitError("'") << funcOp.getSymName() << "' returns a borrow of "
                                              << describeParams(funcOp, local.borrows)
                                              << ", which it has to say with '@" LENDS_DECORATOR "'";
                    }
                    else if (local.borrows < 0)
                    {
                        funcOp.emitError("'@" LENDS_DECORATOR "' of '")
                            << funcOp.getSymName() << "' names " << describeParams(funcOp, lends)
                            << ", but its body does not return a borrow of it";
                    }
                    else
                    {
                        funcOp.emitError("'@" LENDS_DECORATOR "' of '")
                            << funcOp.getSymName() << "' names " << describeParams(funcOp, lends)
                            << ", but its body returns a borrow of " << describeParams(funcOp, local.borrows);
                    }

                    signalPassFailure();
                }
                else if (llvm::ArrayRef<int32_t>(local.owned) != keeps)
                {
                    if (funcOp->hasAttr(OWN_DECLARED_KEEPS_ATTR_NAME))
                    {
                        funcOp.emitError("'@" KEEPS_DECORATOR "' of '")
                            << funcOp.getSymName() << "' names " << describeParams(funcOp, keeps) << ", but its body keeps "
                            << describeParams(funcOp, local.owned);
                    }
                    else
                    {
                        funcOp.emitError("'") << funcOp.getSymName() << "' keeps " << describeParams(funcOp, local.owned)
                                              << ", which it has to say with '@" KEEPS_DECORATOR "'";
                    }

                    signalPassFailure();
                }
            }

            if (funcOp->hasAttr(OWN_DECLARED_BORROWS_ATTR_NAME))
            {
                auto says = declared(funcOp, OWN_DECLARED_BORROWS_ATTR_NAME);
                auto found = borrowsFound.find(funcOp);
                if (found == borrowsFound.end())
                {
                    funcOp.emitError("'@" BORROWS_DECORATOR "' of '")
                        << funcOp.getSymName() << "' names " << describeParams(funcOp, says)
                        << ", but it neither makes a generator whose state borrows its arguments nor returns one";
                    signalPassFailure();
                }
                else if (llvm::ArrayRef<int32_t>(found->second) != says)
                {
                    funcOp.emitError("'@" BORROWS_DECORATOR "' of '")
                        << funcOp.getSymName() << "' names " << describeParams(funcOp, says)
                        << ", but the state it makes borrows " << describeParams(funcOp, found->second);
                    signalPassFailure();
                }
            }
        }

        // the members of a family a declaring method is in: each says the same
        for (auto &[key, members] : families)
        {
            mlir_ts::FuncOp declaring;
            for (auto member : members)
            {
                if (auto funcOp = functions.lookup(member.getValue()); funcOp && declaresFacts(funcOp))
                {
                    declaring = funcOp;
                    break;
                }
            }

            if (!declaring)
            {
                continue;
            }

            for (auto member : members)
            {
                auto funcOp = functions.lookup(member.getValue());
                if (!funcOp || funcOp == declaring)
                {
                    continue;
                }

                for (auto *name : {OWN_DECLARED_KEEPS_ATTR_NAME, OWN_DECLARED_BORROWS_ATTR_NAME, OWN_DECLARED_LENDS_ATTR_NAME})
                {
                    if (funcOp->hasAttr(name) != declaring->hasAttr(name) || declared(funcOp, name) != declared(declaring, name))
                    {
                        funcOp.emitError("'") << funcOp.getSymName() << "' overrides '" << declaring.getSymName()
                                              << "', and has to say the same '@" KEEPS_DECORATOR "', '@" BORROWS_DECORATOR
                                                 "' and '@" LENDS_DECORATOR "'";
                        signalPassFailure();
                        break;
                    }
                }
            }
        }
    }

    // Functions that destroy nothing their caller can reach.
    llvm::DenseSet<mlir::Operation *> noDrops;

    // A class's vtable is a global named `<class>..vtbl` that holds its `..instanceOf`. An
    // interface's vtable for a class (`B.I..vtbl`) holds only the methods and is not one: a call
    // through an interface does not name what it reaches.
    void collectClassVTables(mlir::ModuleOp module)
    {
        llvm::SmallVector<std::pair<llvm::StringRef, llvm::SmallVector<std::pair<int64_t, mlir::StringAttr>>>> vtables;
        module.walk([&](mlir_ts::GlobalOp globalOp) {
            auto entries = classVTableEntries(globalOp);
            if (!entries.empty())
            {
                auto name = globalOp.getSymName();
                classNames.insert(name.drop_back(llvm::StringRef("..vtbl").size()));
                vtables.push_back({name, std::move(entries)});
            }
        });

        for (auto &[name, entries] : vtables)
        {
            for (auto [position, symbol] : entries)
            {
                families[{position, methodName(symbol.getValue())}].push_back(symbol);
            }
        }
    }

    mlir_ts::TupleType storageTupleOf(mlir::Type type)
    {
        return own_facts::storageTupleOf(&getContext(), type);
    }

    // The functions a class vtable holds, by position; empty for any other global.
    static llvm::SmallVector<std::pair<int64_t, mlir::StringAttr>> classVTableEntries(mlir_ts::GlobalOp globalOp)
    {
        llvm::SmallVector<std::pair<int64_t, mlir::StringAttr>> entries;
        if (!globalOp.getSymName().ends_with("..vtbl"))
        {
            return entries;
        }

        auto isClass = false;
        globalOp.getInitializerRegion().walk([&](mlir_ts::InsertPropertyOp insertOp) {
            auto symbolRefOp = insertOp.getValue().getDefiningOp<mlir_ts::SymbolRefOp>();
            if (!symbolRefOp || insertOp.getPosition().size() != 1)
            {
                return;
            }

            auto symbol = symbolRefOp.getIdentifierAttr().getAttr();
            isClass = isClass || symbol.getValue().ends_with("..instanceOf");
            entries.push_back({insertOp.getPosition()[0], symbol});
        });

        if (!isClass)
        {
            entries.clear();
        }

        return entries;
    }

    static bool isClassVTableEntry(mlir::Operation *op)
    {
        auto globalOp = op->getParentOfType<mlir_ts::GlobalOp>();
        return globalOp && !classVTableEntries(globalOp).empty();
    }

    // `B.get` -> `get`: the symbol with the longest class name that prefixes it taken off. An
    // override keeps its base's method name, which is what puts both in one family.
    std::string methodName(llvm::StringRef symbol)
    {
        size_t best = 0;
        for (auto &entry : classNames)
        {
            auto name = entry.getKey();
            if (name.size() > best && symbol.size() > name.size() && symbol.starts_with(name) &&
                symbol[name.size()] == '.')
            {
                best = name.size() + 1;
            }
        }

        return symbol.drop_front(best).str();
    }

    llvm::SmallVector<mlir::StringAttr, 2> familyOf(int64_t index, mlir::StringAttr identifier)
    {
        llvm::SmallVector<mlir::StringAttr, 2> members;
        if (auto found = families.find({index, methodName(identifier.getValue())}); found != families.end())
        {
            members = found->second;
        }

        if (!llvm::is_contained(members, identifier))
        {
            members.push_back(identifier);
        }

        return members;
    }

    // Adds `symbol`'s function to `callees`; false when it is not defined in this module.
    bool addDefined(mlir::StringAttr symbol, Callees &callees)
    {
        auto funcOp = functions.lookup(symbol.getValue());
        if (!funcOp || funcOp.isDeclaration())
        {
            return false;
        }

        if (!llvm::is_contained(callees.funcs, funcOp))
        {
            callees.funcs.push_back(funcOp);
        }

        return true;
    }

    // A function declared here and defined in a library that exported its `__own_no_drops`: one
    // linked with this module, through its import library. Or a C function declared
    // `@nodrops` (OWN_DECLARED_NO_DROPS_ATTR_NAME).
    bool addImported(mlir::StringAttr symbol, Callees &callees)
    {
        auto funcOp = functions.lookup(symbol.getValue());
        if (!funcOp || !funcOp.isDeclaration() ||
            !(importedNoDrops.contains(symbol.getValue()) || funcOp->hasAttr(OWN_DECLARED_NO_DROPS_ATTR_NAME)))
        {
            return false;
        }

        callees.imported = true;
        callees.importedName = symbol.getValue();
        return true;
    }

    // `ts.Load(ts.AddressOf @f)`, where the global `f` holds the address of a function exported
    // by a library loaded at run time, looked up by name: that function, if the library exported
    // its `__own_no_drops`.
    bool addImportedByAddress(mlir::Value callee, Callees &callees)
    {
        auto loadOp = callee.getDefiningOp<mlir_ts::LoadOp>();
        auto addressOfOp = loadOp ? loadOp.getReference().getDefiningOp<mlir_ts::AddressOfOp>() : mlir_ts::AddressOfOp();
        if (!addressOfOp)
        {
            return false;
        }

        auto globalOp = globals.lookup(addressOfOp.getGlobalName());
        if (!globalOp || writtenGlobals.contains(addressOfOp.getGlobalName()))
        {
            return false;
        }

        // the global is never anything but that address: `Cast(SearchForAddressOfSymbol(name))`
        auto &region = globalOp.getInitializerRegion();
        if (!region.hasOneBlock())
        {
            return false;
        }

        auto resultOp = mlir::dyn_cast<mlir_ts::GlobalResultOp>(region.front().getTerminator());
        auto castOp = resultOp && resultOp->getNumOperands() == 1
                          ? resultOp->getOperand(0).getDefiningOp<mlir_ts::CastOp>()
                          : mlir_ts::CastOp();
        auto searchOp = castOp ? castOp.getIn().getDefiningOp<mlir_ts::SearchForAddressOfSymbolOp>()
                               : mlir_ts::SearchForAddressOfSymbolOp();
        auto nameOp = searchOp ? searchOp->getOperand(0).getDefiningOp<mlir_ts::ConstantOp>() : mlir_ts::ConstantOp();
        auto name = nameOp ? mlir::dyn_cast<mlir::StringAttr>(nameOp.getValue()) : mlir::StringAttr();
        if (!name || !importedNoDrops.contains(name.getValue()))
        {
            return false;
        }

        callees.imported = true;
        callees.importedName = name.getValue();
        return true;
    }

    Callees resolve(mlir::Operation *op)
    {
        Callees callees;
        if (auto callee = directCallee(op))
        {
            callees.named.push_back(callee.getAttr());
            callees.known = addDefined(callee.getAttr(), callees) || addImported(callee.getAttr(), callees);
        }
        else if (mlir::isa<mlir_ts::CallInternalOp, mlir_ts::CallIndirectOp, mlir_ts::InvokeOp>(op))
        {
            resolveValue(calleeValue(op), callees);
        }

        if (!callees.known)
        {
            callees.funcs.clear();
        }

        return callees;
    }

    // The function value a call goes through: a method of a class, virtual or not, a symbol, or
    // the `.instanceOf` slot. Anything else - a closure, a function read from a field, an
    // interface's method - is unknown.
    void resolveValue(mlir::Value callee, Callees &callees)
    {
        // `ts.Load(ts.AddressOf @g)`: a dynamic import's global, which may say what the function
        // keeps; a method of a class imported so is bound to its object first
        auto function = callee;
        if (auto getMethodOp = function.getDefiningOp<mlir_ts::GetMethodOp>())
        {
            function = getMethodOp.getBoundFunc();
        }

        if (auto boundOp = function.getDefiningOp<mlir_ts::CreateBoundFunctionOp>())
        {
            function = boundOp.getFunc();
        }

        auto loadOp = function.getDefiningOp<mlir_ts::LoadOp>();
        if (auto addressOfOp = loadOp ? loadOp.getReference().getDefiningOp<mlir_ts::AddressOfOp>() : mlir_ts::AddressOfOp();
            addressOfOp && !writtenGlobals.contains(addressOfOp.getGlobalName()))
        {
            callees.named.push_back(addressOfOp.getGlobalNameAttr().getAttr());
        }

        if (addImportedByAddress(callee, callees))
        {
            callees.known = true;
            return;
        }

        if (auto getMethodOp = callee.getDefiningOp<mlir_ts::GetMethodOp>())
        {
            callee = getMethodOp.getBoundFunc();
        }

        auto *def = callee.getDefiningOp();
        mlir::StringAttr identifier;
        int64_t index = -1;
        if (auto refOp = mlir::dyn_cast_or_null<mlir_ts::ThisVirtualSymbolRefOp>(def))
        {
            identifier = refOp.getIdentifierAttr().getAttr();
            index = refOp.getIndex();
        }
        else if (auto refOp = mlir::dyn_cast_or_null<mlir_ts::VirtualSymbolRefOp>(def))
        {
            identifier = refOp.getIdentifierAttr().getAttr();
            index = refOp.getIndex();
        }
        else if (auto refOp = mlir::dyn_cast_or_null<mlir_ts::ThisSymbolRefOp>(def))
        {
            identifier = refOp.getIdentifierAttr().getAttr();
        }
        else if (auto refOp = mlir::dyn_cast_or_null<mlir_ts::SymbolRefOp>(def))
        {
            identifier = refOp.getIdentifierAttr().getAttr();
        }
        else if (isInstanceOfSlot(def))
        {
            callees.known = true;
            callees.instanceOf = true;
            return;
        }
        else if (auto symbol = fieldFunctions.of(def))
        {
            callees.known = addDefined(symbol, callees);
            callees.fieldFunction = callees.known;
            return;
        }

        if (!identifier)
        {
            return;
        }

        if (index < 0)
        {
            callees.named.push_back(identifier);
            callees.known = addDefined(identifier, callees) || addImported(identifier, callees);
            return;
        }

        callees.named = familyOf(index, identifier);
        resolveFamily(index, identifier, callees);
    }

    // A class another module can see may be extended there, and its override is a candidate this
    // module never sees: every member has to be private and defined here.
    void resolveFamily(int64_t index, mlir::StringAttr identifier, Callees &callees)
    {
        callees.known = true;
        for (auto member : familyOf(index, identifier))
        {
            callees.known = addDefined(member, callees) && callees.known;
        }

        callees.known = callees.known && llvm::all_of(callees.funcs, [](mlir_ts::FuncOp funcOp) { return funcOp.isPrivate(); });
    }

    // `ts.Cast(ts.VTableOffsetRef(vtable, 0))`, where `vtable` is the one an object's first word
    // points to or an interface value's own: slot 0 of either is `.instanceOf` - a class's
    // generated one, a string compare, or `.instanceOf.none` for an object literal behind an
    // interface. Only mlirGenInstanceOfThroughVTable builds a call through it. Neither destroys
    // anything.
    static bool isInstanceOfSlot(mlir::Operation *def)
    {
        auto castOp = mlir::dyn_cast_or_null<mlir_ts::CastOp>(def);
        auto slot = castOp ? castOp.getIn().getDefiningOp<mlir_ts::VTableOffsetRefOp>() : mlir_ts::VTableOffsetRefOp();
        if (!slot || slot.getIndex() != INTERFACE_VTABLE_INSTANCEOF_SLOT)
        {
            return false;
        }

        // a class vtable, read out of an object's first word, or an interface's own vtable,
        // whose slot 0 is its implementer's `.instanceOf` (INTERFACE_VTABLE_HEADER_SLOTS)
        auto *vtable = slot.getVtable().getDefiningOp();
        if (auto classVTable = mlir::dyn_cast_or_null<mlir_ts::VTableOffsetRefOp>(vtable))
        {
            return classVTable.getIndex() == 0;
        }

        return mlir::isa_and_nonnull<mlir_ts::ExtractInterfaceVTableOp>(vtable);
    }

    // Is every use of this function value the callee of a call - through `ts.GetMethod`, with
    // `ts.GetThis` reading the object beside it?
    static bool onlyCalled(mlir::Value value)
    {
        for (auto &use : value.getUses())
        {
            auto *user = use.getOwner();
            if (mlir::isa<mlir_ts::CallInternalOp, mlir_ts::CallIndirectOp, mlir_ts::InvokeOp>(user) &&
                calleeValue(user) == use.get() && use.getOperandNumber() == 0)
            {
                continue;
            }

            if (mlir::isa<mlir_ts::GetThisOp>(user))
            {
                continue;
            }

            if (mlir::isa<mlir_ts::GetMethodOp>(user) && onlyCalled(user->getResult(0)))
            {
                continue;
            }

            return false;
        }

        return true;
    }

    // The closed world. A function is open when a caller this pass cannot see may reach it: it is
    // not private (another module may call it), or its symbol is used other than by a direct call,
    // a class vtable's entry, or a method reference used only as a callee. A virtual reference
    // that escapes opens its whole family.
    void findOpen(mlir::ModuleOp module)
    {
        for (auto &entry : functions)
        {
            if (!entry.second.isPrivate())
            {
                open.try_emplace(entry.second, "it can be called from another module; '@" LENDS_DECORATOR "' or '@" KEEPS_DECORATOR
                                               "' would say it in its signature");
            }
        }

        // the module is a symbol table, which getSymbolUses does not look inside: walk its body
        auto uses = mlir::SymbolTable::getSymbolUses(&module.getBodyRegion());
        if (!uses)
        {
            // an op this pass cannot read symbols through: nothing is closed
            for (auto &entry : functions)
            {
                open.try_emplace(entry.second, "its callers cannot all be found");
            }

            return;
        }

        for (auto &use : *uses)
        {
            auto symbol = use.getSymbolRef().getRootReference();
            auto funcOp = functions.lookup(symbol.getValue());
            if (!funcOp)
            {
                continue;
            }

            auto *user = use.getUser();
            if (mlir::isa<mlir_ts::SymbolCallInternalOp, mlir_ts::CallOp, mlir_ts::InvokeOp>(user))
            {
                continue;
            }

            if (mlir::isa<mlir_ts::SymbolRefOp>(user) && isClassVTableEntry(user))
            {
                continue;
            }

            int64_t index = -1;
            if (auto refOp = mlir::dyn_cast<mlir_ts::ThisVirtualSymbolRefOp>(user))
            {
                index = refOp.getIndex();
            }
            else if (auto refOp = mlir::dyn_cast<mlir_ts::VirtualSymbolRefOp>(user))
            {
                index = refOp.getIndex();
            }

            auto inFunction = !!user->getParentOfType<mlir::FunctionOpInterface>();
            if (inFunction && mlir::isa<mlir_ts::ThisVirtualSymbolRefOp, mlir_ts::VirtualSymbolRefOp,
                                        mlir_ts::ThisSymbolRefOp, mlir_ts::SymbolRefOp>(user) &&
                onlyCalled(user->getResult(0)))
            {
                // a virtual call this pass cannot resolve reaches its family without the facts
                if (index >= 0)
                {
                    Callees callees;
                    resolveFamily(index, symbol, callees);
                    if (!callees.known)
                    {
                        openFamily(index, symbol, "an override may be defined in another module");
                    }
                }

                continue;
            }

            open.try_emplace(funcOp, "it is used other than by a call");
            if (index >= 0)
            {
                openFamily(index, symbol, "it is used other than by a call");
            }
        }
    }

    void openFamily(int64_t index, mlir::StringAttr symbol, const char *why)
    {
        for (auto member : familyOf(index, symbol))
        {
            if (auto memberOp = functions.lookup(member.getValue()))
            {
                open.try_emplace(memberOp, why);
            }
        }
    }

    // ---- Results that borrow an argument ----

    // The values a function returns: what is stored into the local its returns read (MLIRGen's
    // result slot), or what a return hands back directly.
    static llvm::SmallVector<mlir::Value> returnedValues(mlir_ts::FuncOp funcOp)
    {
        llvm::SmallVector<mlir::Value> values;
        funcOp.walk([&](mlir_ts::ReturnInternalOp returnOp) {
            for (auto operand : returnOp.getRetOperands())
            {
                auto loadOp = operand.getDefiningOp<mlir_ts::LoadOp>();
                auto slot = loadOp ? loadOp.getReference().getDefiningOp<mlir_ts::VariableOp>() : mlir_ts::VariableOp();
                if (!slot || slot.getInitializer() || isOwningVariable(slot))
                {
                    values.push_back(operand);
                    continue;
                }

                for (auto *user : slot.getResult().getUsers())
                {
                    auto storeOp = mlir::dyn_cast<mlir_ts::StoreOp>(user);
                    if (storeOp && storeOp.getReference() == slot.getResult())
                    {
                        values.push_back(storeOp.getValue());
                    }
                }
            }
        });

        return values;
    }

    // The parameter every heap value the function returns borrows, or -1 (spec 2.4). A result
    // that holds no block (`null`, a number) agrees with any.
    static int localResultBorrows(mlir_ts::FuncOp funcOp)
    {
        auto found = -1;
        for (auto value : returnedValues(funcOp))
        {
            if (holdsNoBlock(value))
            {
                continue;
            }

            // a handle returned is a counted copy, never a borrow (spec 23.2)
            if (isHandle(value))
            {
                return -1;
            }

            auto index = borrowedParam(value);
            if (index < 0 || (found >= 0 && index != found))
            {
                return -1;
            }

            found = index;
        }

        return found;
    }

    // ---- Parameters the callee keeps ----

    // Does `user` take `used` - a read of a parameter - into something that outlives the call: a
    // store into a field, an element or a global, an insertion into an array, or a call that keeps
    // it in turn? A local that holds it, and the result it is returned as, do not count: the first
    // is a borrow, the second a borrowed result.
    static bool keeps(mlir::Operation *user, mlir::Value used)
    {
        // a number, a boolean, a string literal: nothing to keep
        if (holdsNoBlock(used))
        {
            return false;
        }

        if (auto storeOp = mlir::dyn_cast<mlir_ts::StoreOp>(user))
        {
            auto ref = storeOp.getReference();
            return storeOp.getValue() == used && (isPlace(ref) || ref.getDefiningOp<mlir_ts::AddressOfOp>());
        }

        if (auto pushOp = mlir::dyn_cast<mlir_ts::ArrayPushOp>(user))
        {
            return llvm::is_contained(pushOp.getItems(), used);
        }

        // `new Shared(x)` moves `x` into the block, as a store into a place does (spec 23.1)
        if (auto sharedNewOp = mlir::dyn_cast<mlir_ts::SharedNewOp>(user))
        {
            return sharedNewOp.getValue() == used;
        }

        if (auto unshiftOp = mlir::dyn_cast<mlir_ts::ArrayUnshiftOp>(user))
        {
            return llvm::is_contained(unshiftOp.getItems(), used);
        }

        if (auto spliceOp = mlir::dyn_cast<mlir_ts::ArraySpliceOp>(user))
        {
            return llvm::is_contained(spliceOp.getItems(), used);
        }

        if (isCall(user))
        {
            auto args = callArgs(user);
            return llvm::any_of(ownedParams(user), [&](int32_t index) {
                return static_cast<size_t>(index) < args.size() && args[index] == used;
            });
        }

        return false;
    }

    // The parameters the body keeps (spec 2.4, owned-by-callee): a read of the parameter's slot -
    // never assigned - that something keeps.
    static llvm::SmallVector<int32_t> localOwnedParams(mlir_ts::FuncOp funcOp)
    {
        llvm::SmallVector<int32_t> owned;
        if (funcOp.getBody().empty())
        {
            return owned;
        }

        for (auto argument : funcOp.getBody().front().getArguments())
        {
            // a handle kept is a counted copy: its caller gives nothing up (spec 23.2)
            if (isHandle(argument))
            {
                continue;
            }

            auto kept = false;
            for (auto *user : argument.getUsers())
            {
                auto varOp = mlir::dyn_cast<mlir_ts::VariableOp>(user);
                auto index = -1;
                if (!varOp || !isParameterSlot(varOp, index))
                {
                    continue;
                }

                for (auto *slotUser : varOp.getResult().getUsers())
                {
                    if (auto loadOp = mlir::dyn_cast<mlir_ts::LoadOp>(slotUser))
                    {
                        forEachUse(loadOp.getResult(), [&](mlir::Operation *use, mlir::Value used) {
                            kept = kept || keeps(use, used);
                        });
                    }
                }
            }

            if (kept)
            {
                owned.push_back(argument.getArgNumber());
            }
        }

        return owned;
    }

    // ---- The facts callers must know ----

    // Owned parameters and a result that borrows only hold where every caller knows them, so the
    // facts survive only on a closed function whose every family agrees, and families are
    // demoted until they do: a member that loses its facts can make another family disagree. A
    // caller that returns what such a call returns, or passes a parameter on to a kept one, gets
    // the fact too, so the facts are recomputed with the calls pinned until nothing changes; if
    // that does not settle, nobody gets any.
    void computeFacts()
    {
        llvm::DenseMap<mlir::Operation *, Facts> pinned;
        for (auto round = 0; round < 16; ++round)
        {
            for (auto &entry : functions)
            {
                if (!entry.second.isDeclaration())
                {
                    factsLocal[entry.second] = {localResultBorrows(entry.second), localOwnedParams(entry.second)};
                }
            }

            meetFacts();

            llvm::DenseMap<mlir::Operation *, Facts> next;
            for (auto &call : calls)
            {
                auto facts = callFacts(call.callees);
                if (facts.empty())
                {
                    // a call this pass cannot resolve still has what its callees' signatures say
                    auto keeps = declaredByCallees(call.callees, OWN_DECLARED_KEEPS_ATTR_NAME);
                    facts.owned.assign(keeps.begin(), keeps.end());
                    if (auto lends = declaredByCallees(call.callees, OWN_DECLARED_LENDS_ATTR_NAME); !lends.empty())
                    {
                        facts.borrows = lends.front();
                    }
                }

                if (!facts.empty())
                {
                    next[call.op] = facts;
                }
            }

            auto settled = next == pinned;
            pinned = std::move(next);
            pinFacts(pinned);
            if (settled)
            {
                return;
            }
        }

        factsEffective.clear();
        pinFacts(llvm::DenseMap<mlir::Operation *, Facts>());
    }

    void meetFacts()
    {
        factsEffective.clear();
        for (auto &[funcOp, facts] : factsLocal)
        {
            if (!facts.empty() && !open.contains(funcOp))
            {
                factsEffective[funcOp] = facts;
            }
        }

        // what a signature says stands, open or not; checkDeclared holds the body to it
        for (auto &entry : functions)
        {
            if (declaresCallerFacts(entry.second))
            {
                auto keeps = declared(entry.second, OWN_DECLARED_KEEPS_ATTR_NAME);
                factsEffective[entry.second] =
                    Facts{declaredLends(entry.second), llvm::SmallVector<int32_t>(keeps.begin(), keeps.end())};
            }
        }

        for (auto changed = true; changed;)
        {
            changed = false;
            for (auto &[key, members] : families)
            {
                llvm::SmallVector<Facts> seen;
                for (auto member : members)
                {
                    auto funcOp = functions.lookup(member.getValue());
                    seen.push_back(funcOp ? factsEffective.lookup(funcOp) : Facts());
                }

                if (llvm::all_equal(seen))
                {
                    continue;
                }

                for (auto member : members)
                {
                    if (auto funcOp = functions.lookup(member.getValue());
                        funcOp && !declaresCallerFacts(funcOp) && factsEffective.erase(funcOp))
                    {
                        changed = true;
                    }
                }
            }
        }
    }

    Facts callFacts(const Callees &callees)
    {
        if (!callees.known || callees.instanceOf || callees.funcs.empty())
        {
            return {};
        }

        auto facts = factsEffective.lookup(callees.funcs.front());
        auto agree = llvm::all_of(callees.funcs, [&](mlir_ts::FuncOp callee) {
            return factsEffective.lookup(callee) == facts;
        });
        return agree ? facts : Facts();
    }

    static void setFacts(mlir::Operation *op, const Facts &facts)
    {
        auto *context = op->getContext();
        op->removeAttr(OWN_RESULT_BORROWS_ATTR_NAME);
        op->removeAttr(OWN_PARAMS_ATTR_NAME);
        if (facts.borrows >= 0)
        {
            op->setAttr(OWN_RESULT_BORROWS_ATTR_NAME,
                        mlir::IntegerAttr::get(mlir::IntegerType::get(context, 32), facts.borrows));
        }

        if (!facts.owned.empty())
        {
            op->setAttr(OWN_PARAMS_ATTR_NAME, mlir::DenseI32ArrayAttr::get(context, facts.owned));
        }
    }

    // The calls get the facts their callees agree on; each function gets its own, or the reason it
    // lost those its body has, for the error there.
    void pinFacts(const llvm::DenseMap<mlir::Operation *, Facts> &pinned)
    {
        for (auto &call : calls)
        {
            setFacts(call.op, pinned.lookup(call.op));
        }

        for (auto &[funcOp, local] : factsLocal)
        {
            auto effective = factsEffective.lookup(funcOp);
            setFacts(funcOp, effective);
            funcOp->removeAttr(OWN_FACTS_LOST_ATTR_NAME);
            if (effective.empty() && !local.empty())
            {
                auto *why = open.lookup(funcOp);
                funcOp->setAttr(OWN_FACTS_LOST_ATTR_NAME,
                                mlir::StringAttr::get(&getContext(), why ? why : "an override in its class family disagrees"));
            }
        }
    }

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

    static int32_t fieldIndex(mlir_ts::TupleType tupleType, llvm::StringRef name)
    {
        for (auto [index, field] : llvm::enumerate(tupleType.getFields()))
        {
            if (auto id = mlir::dyn_cast_or_null<mlir::StringAttr>(field.id); id && id.getValue() == name)
            {
                return static_cast<int32_t>(index);
            }
        }

        return -1;
    }

    // The box the maker stores into the `.captured` field of the initial state it copies into
    // `newOp`'s block: `ts.Store(box, ts.PropertyRef(state, captured))`, `state` a local whose only
    // read fills the block.
    static mlir::Value boxStoredIntoState(mlir_ts::NewOp newOp, int32_t captured)
    {
        mlir_ts::StoreOp fill;
        for (auto *user : newOp->getUsers())
        {
            if (auto storeOp = mlir::dyn_cast<mlir_ts::StoreOp>(user); storeOp && storeOp.getReference() == newOp.getResult())
            {
                fill = storeOp;
            }
        }

        auto loadOp = fill ? fill.getValue().getDefiningOp<mlir_ts::LoadOp>() : mlir_ts::LoadOp();
        auto stateOp = loadOp ? loadOp.getReference().getDefiningOp<mlir_ts::VariableOp>() : mlir_ts::VariableOp();
        if (!stateOp)
        {
            return {};
        }

        mlir::Value box;
        for (auto *user : stateOp->getUsers())
        {
            auto propertyRefOp = mlir::dyn_cast<mlir_ts::PropertyRefOp>(user);
            if (!propertyRefOp || propertyRefOp.getPosition() != captured)
            {
                continue;
            }

            for (auto *fieldUser : propertyRefOp->getUsers())
            {
                auto storeOp = mlir::dyn_cast<mlir_ts::StoreOp>(fieldUser);
                if (!storeOp || box)
                {
                    return {};
                }

                box = storeOp.getValue();
            }
        }

        auto boxOp = box ? box.getDefiningOp<mlir_ts::VariableOp>() : mlir_ts::VariableOp();
        return boxOp && boxOp->hasAttr(CAPTURE_BOX_ATTR_NAME) ? box : mlir::Value();
    }

    // Is every caller of `funcOp` a direct call this pass sees? It is private, and each use of its
    // symbol is a call, or a closure over it that is only retained and released - `.map(f)` builds
    // one and calls its body straight through the box.
    bool calledOnlyDirectly(mlir::ModuleOp module, mlir_ts::FuncOp funcOp, bool publicToo = false)
    {
        if (!funcOp.isPrivate() && !publicToo)
        {
            return false;
        }

        auto uses = mlir::SymbolTable::getSymbolUses(funcOp.getSymNameAttr(), &module.getBodyRegion());
        if (!uses)
        {
            return false;
        }

        for (auto &use : *uses)
        {
            auto *user = use.getUser();
            if (mlir::isa<mlir_ts::SymbolCallInternalOp, mlir_ts::CallOp, mlir_ts::InvokeOp>(user))
            {
                continue;
            }

            // a method whose signature gives its callers the bound: called through its class's
            // vtable or a reference to it, which the facts reach too (declaredByCallees)
            if (publicToo &&
                (isClassVTableEntry(user) ||
                 (mlir::isa<mlir_ts::ThisVirtualSymbolRefOp, mlir_ts::VirtualSymbolRefOp, mlir_ts::ThisSymbolRefOp,
                            mlir_ts::SymbolRefOp>(user) &&
                  onlyCalled(user->getResult(0)))))
            {
                continue;
            }

            auto symbolRefOp = mlir::dyn_cast<mlir_ts::SymbolRefOp>(user);
            if (!symbolRefOp)
            {
                return false;
            }

            for (auto *refUser : symbolRefOp->getUsers())
            {
                auto boundOp = mlir::dyn_cast<mlir_ts::CreateBoundFunctionOp>(refUser);
                if (!boundOp || !llvm::all_of(boundOp->getUsers(), [](mlir::Operation *boundUser) {
                        return mlir::isa<mlir_ts::RetainOp, mlir_ts::ReleaseOp>(boundUser);
                    }))
                {
                    return false;
                }
            }
        }

        return true;
    }

    // A parameter's cell: a captured local declared from an argument of `maker`'s entry block, that
    // owns nothing (its value is the caller's).
    static bool isParameterCell(mlir_ts::VariableOp cellOp, mlir_ts::FuncOp maker, int32_t &index)
    {
        auto argument = cellOp.getInitializer() ? mlir::dyn_cast<mlir::BlockArgument>(cellOp.getInitializer())
                                                : mlir::BlockArgument();
        if (!cellOp.getCaptured().value_or(false) || isOwningVariable(cellOp) || !argument ||
            !argument.getOwner()->isEntryBlock() || argument.getOwner()->getParentOp() != maker.getOperation())
        {
            return false;
        }

        index = static_cast<int32_t>(argument.getArgNumber());
        return true;
    }

    // `ts.Load` of one of `maker`'s parameter slots (isParameterSlot): `index` is its position.
    static bool readsParameterSlot(mlir::Value value, mlir_ts::FuncOp maker, int32_t &index)
    {
        auto loadOp = rootOf(value).getDefiningOp<mlir_ts::LoadOp>();
        auto varOp = loadOp ? loadOp.getReference().getDefiningOp<mlir_ts::VariableOp>() : mlir_ts::VariableOp();
        int position = -1;
        if (!varOp || !isParameterSlot(varOp, position) || varOp->getParentOfType<mlir_ts::FuncOp>() != maker)
        {
            return false;
        }

        index = static_cast<int32_t>(position);
        return true;
    }

    // `ts.Load(ts.PropertyRef(argument, i))` of one of `maker`'s parameters: `index` is its position.
    static bool readsParameterField(mlir::Value value, mlir_ts::FuncOp maker, int32_t &index)
    {
        auto loadOp = rootOf(value).getDefiningOp<mlir_ts::LoadOp>();
        auto propertyRefOp = loadOp ? loadOp.getReference().getDefiningOp<mlir_ts::PropertyRefOp>() : mlir_ts::PropertyRefOp();
        auto argument = propertyRefOp ? mlir::dyn_cast<mlir::BlockArgument>(propertyRefOp->getOperand(0)) : mlir::BlockArgument();
        if (!argument || !argument.getOwner()->isEntryBlock() || argument.getOwner()->getParentOp() != maker.getOperation())
        {
            return false;
        }

        index = static_cast<int32_t>(argument.getArgNumber());
        return true;
    }

    bool ownsHeap(mlir::Location location, mlir::Type type)
    {
        MLIRTypeHelper mth(&getContext(), CompileOptions{});
        return mth.ownsHeapMemory(location, type);
    }

    void findBorrowingStates(mlir::ModuleOp module)
    {
        llvm::MapVector<mlir::Type, BorrowingState> states;
        llvm::DenseMap<mlir::Type, int> made;
        module.walk([&](mlir_ts::NewOp newOp) {
            auto valueRefType = mlir::dyn_cast<mlir_ts::ValueRefType>(newOp.getType());
            auto tupleType = valueRefType ? mlir::dyn_cast<mlir_ts::TupleType>(valueRefType.getElementType())
                                          : mlir_ts::TupleType();
            if (!tupleType)
            {
                return;
            }

            ++made[tupleType];
            auto captured = fieldIndex(tupleType, CAPTURED_NAME);
            auto maker = newOp->getParentOfType<mlir_ts::FuncOp>();
            // an open maker has callers this pass cannot give the bound to, unless its signature
            // gives it to them (`@borrows`)
            if (captured < 0 || !maker ||
                !calledOnlyDirectly(module, maker, /*publicToo=*/maker->hasAttr(OWN_DECLARED_BORROWS_ATTR_NAME)))
            {
                return;
            }

            auto box = boxStoredIntoState(newOp, captured);
            if (!box)
            {
                return;
            }

            BorrowingState state{maker, captured};
            for (auto *user : box.getUsers())
            {
                auto propertyRefOp = mlir::dyn_cast<mlir_ts::PropertyRefOp>(user);
                if (!propertyRefOp)
                {
                    continue;
                }

                for (auto *fieldUser : propertyRefOp->getUsers())
                {
                    auto storeOp = mlir::dyn_cast<mlir_ts::StoreOp>(fieldUser);
                    if (!storeOp)
                    {
                        return;
                    }

                    auto value = storeOp.getValue();
                    int32_t index = -1;
                    if (auto cellOp = value.getDefiningOp<mlir_ts::VariableOp>(); cellOp && isParameterCell(cellOp, maker, index))
                    {
                        auto cellType = mlir::cast<mlir_ts::RefType>(cellOp.getType());
                        // a handle parameter's cell holds a count of its own (spec 23.2), which only
                        // the box's release gives back: a box holding one owns what it holds
                        if (MLIRTypeHelper::isSharedHandleType(cellType.getElementType()))
                        {
                            return;
                        }

                        if (ownsHeap(cellOp.getLoc(), cellType.getElementType()))
                        {
                            state.bounded.push_back(index);
                        }

                        continue;
                    }

                    // a copy of what a parameter holds in a field: `.map`'s maker is a closure body
                    // that copies its own box's fields into the generator's
                    if (!ownsHeap(value.getLoc(), value.getType()))
                    {
                        continue;
                    }

                    // `this`, which the box holds by value (`.this`), not in a cell: read out of
                    // the parameter's slot
                    if (readsParameterSlot(value, maker, index))
                    {
                        if (MLIRTypeHelper::isSharedHandleType(value.getType()))
                        {
                            return;
                        }

                        if (!llvm::is_contained(state.bounded, index))
                        {
                            state.bounded.push_back(index);
                        }

                        continue;
                    }

                    if (!readsParameterField(value, maker, index))
                    {
                        return; // a box with anything but what the parameters hold owns what it holds
                    }

                    if (!llvm::is_contained(state.bounded, index))
                    {
                        state.bounded.push_back(index);
                    }
                }
            }

            if (!state.bounded.empty())
            {
                states[tupleType] = state;
            }
        });

        // made in one place only: the maker is the one function a bound can come from
        states.remove_if([&](auto &entry) { return made[entry.first] != 1; });
        for (auto &[type, state] : states)
        {
            auto bounded = state.bounded;
            llvm::sort(bounded);
            borrowsFound[state.maker] = bounded;
        }
        if (states.empty())
        {
            return;
        }

        // Each store into a field of a state type, by field: what the field may be given.
        llvm::DenseMap<std::pair<mlir::Type, int64_t>, llvm::SmallVector<mlir::Value>> stored;
        module.walk([&](mlir_ts::StoreOp storeOp) {
            auto propertyRefOp = storeOp.getReference().getDefiningOp<mlir_ts::PropertyRefOp>();
            auto tupleType = propertyRefOp ? storageTupleOf(propertyRefOp->getOperand(0).getType()) : mlir_ts::TupleType();
            if (tupleType && states.count(tupleType))
            {
                stored[{tupleType, propertyRefOp.getPosition()}].push_back(storeOp.getValue());
            }
        });

        for (auto &[type, state] : states)
        {
            auto tupleType = mlir::cast<mlir_ts::TupleType>(type);
            // least fixpoint: a field borrows once everything stored into it is a read of a borrowing
            // cell or field
            for (auto changed = true; changed;)
            {
                changed = false;
                for (auto [index, field] : llvm::enumerate(tupleType.getFields()))
                {
                    auto position = static_cast<int32_t>(index);
                    auto found = stored.find({type, position});
                    if (position == state.captured || state.fields.contains(position) || found == stored.end() ||
                        !ownsHeap(state.maker.getLoc(), field.type))
                    {
                        continue;
                    }

                    if (llvm::all_of(found->second, [&](mlir::Value value) { return readsBorrowed(value, tupleType, state); }))
                    {
                        state.fields.insert(position);
                        changed = true;
                    }
                }
            }
        }

        auto *context = &getContext();
        llvm::SmallVector<mlir::Attribute> entries;
        for (auto &[type, state] : states)
        {
            llvm::SmallVector<int32_t> fields{state.captured};
            fields.append(state.fields.begin(), state.fields.end());
            entries.push_back(mlir::ArrayAttr::get(
                context, {mlir::TypeAttr::get(type), mlir::DenseI32ArrayAttr::get(context, fields)}));
            state.maker->setAttr(OWN_RESULT_BOUNDED_ATTR_NAME, mlir::DenseI32ArrayAttr::get(context, state.bounded));
        }

        module->setAttr(OWN_BORROWING_FIELDS_ATTR_NAME, mlir::ArrayAttr::get(context, entries));
    }

    // Is `value` a read of what a state of `tupleType` borrows: the value in one of its box's cells
    // (`ts.Load` of `ts.Load(ts.PropertyRef(ts.Load(ts.PropertyRef(state, captured)), i))`), or a
    // borrowing field's (`ts.Load(ts.PropertyRef(state, p))`)?
    bool readsBorrowed(mlir::Value value, mlir_ts::TupleType tupleType, const BorrowingState &state)
    {
        auto loadOp = rootOf(value).getDefiningOp<mlir_ts::LoadOp>();
        if (!loadOp)
        {
            return false;
        }

        auto isStateField = [&](mlir::Value ref, int32_t &position) {
            auto propertyRefOp = ref.getDefiningOp<mlir_ts::PropertyRefOp>();
            if (!propertyRefOp || storageTupleOf(propertyRefOp->getOperand(0).getType()) != tupleType)
            {
                return false;
            }

            position = propertyRefOp.getPosition();
            return true;
        };

        int32_t position = -1;
        if (isStateField(loadOp.getReference(), position))
        {
            return state.fields.contains(position);
        }

        // a field of the box the state holds, copied in by value, or a cell read out of it
        auto isBoxField = [&](mlir::Value ref) {
            auto fieldRef = ref.getDefiningOp<mlir_ts::PropertyRefOp>();
            auto boxLoad = fieldRef ? fieldRef->getOperand(0).getDefiningOp<mlir_ts::LoadOp>() : mlir_ts::LoadOp();
            return boxLoad && isStateField(boxLoad.getReference(), position) && position == state.captured;
        };

        if (isBoxField(loadOp.getReference()))
        {
            return true;
        }

        auto cellLoad = loadOp.getReference().getDefiningOp<mlir_ts::LoadOp>();
        return cellLoad && isBoxField(cellLoad.getReference());
    }

    // ---- Globals given a borrow ----

    // A global initialized with a read of another global, a field or an element (`let k = a`,
    // `let k = h.c`) would be a second owner of a block another owner keeps: the first overwrite
    // of either frees what the other still reads, and both give it back at exit. rc counts the
    // global's own reference (OwnedReturnConsumptionPass, claimGlobalInitializers); own cannot
    // move the value out of its owner, and a global cannot hold a borrow. A handle is counted
    // under own too.
    void rejectBorrowedGlobalInitializers(mlir::ModuleOp module)
    {
        module.walk([&](mlir_ts::GlobalOp globalOp) {
            auto *block = globalOp.getInitializerBlock();
            auto resultOp = block && !block->empty() ? mlir::dyn_cast<mlir_ts::GlobalResultOp>(block->getTerminator())
                                                     : mlir_ts::GlobalResultOp();
            if (!resultOp || resultOp->getNumOperands() != 1)
            {
                return;
            }

            auto value = resultOp->getOperand(0);
            while (value.getDefiningOp() && isView(value.getDefiningOp()))
            {
                value = value.getDefiningOp()->getOperand(0);
            }

            auto loadOp = value.getDefiningOp<mlir_ts::LoadOp>();
            auto *place = loadOp ? loadOp.getReference().getDefiningOp() : nullptr;
            if (!place || !mlir::isa<mlir_ts::AddressOfOp, mlir_ts::PropertyRefOp, mlir_ts::ElementRefOp>(place) ||
                isHandle(value) || isConstantData(value) || !ownsHeap(value.getLoc(), value.getType()))
            {
                return;
            }

            resultOp.emitError("the initializer of global '")
                << globalOp.getSymName()
                << "' reads a value another owner keeps; -mm=own cannot make a global a second owner";
            signalPassFailure();
        });
    }

    // ---- Drops ----

    // Least fixpoint: a function drops nothing until its body or a call it makes shows it may.
    void computeDrops()
    {
        llvm::DenseMap<mlir::Operation *, llvm::SmallVector<Call *>> callsIn;
        for (auto &call : calls)
        {
            callsIn[call.op->getParentOfType<mlir_ts::FuncOp>()].push_back(&call);
        }

        for (auto &entry : functions)
        {
            auto funcOp = entry.second;
            if (!funcOp.isDeclaration() && !dropsInBody(funcOp))
            {
                noDrops.insert(funcOp);
            }
        }

        for (auto changed = true; changed;)
        {
            changed = false;
            for (auto &entry : functions)
            {
                auto funcOp = entry.second;
                if (!noDrops.contains(funcOp))
                {
                    continue;
                }

                for (auto *call : callsIn[funcOp])
                {
                    if (!callMayDrop(call->callees))
                    {
                        continue;
                    }

                    noDrops.erase(funcOp);
                    changed = true;
                    break;
                }
            }
        }
    }

    bool callMayDrop(const Callees &callees)
    {
        if (!callees.known)
        {
            return true;
        }

        return !callees.instanceOf && !callees.imported &&
               llvm::any_of(callees.funcs, [&](mlir_ts::FuncOp callee) { return !noDrops.contains(callee); });
    }

    // ---- Export ----

    // A library says, beside its memory-model marker, which of its exported functions destroy
    // nothing (SHARED_LIB_OWN_FACTS). The global is a copy of the marker `__tsmm_own_<module>`
    // under another name and with another value, so it is exported exactly as the marker is, and
    // MLIRGen emits the same under own as under rc. No marker, nothing is exported.
    void exportNoDrops()
    {
        auto markerPrefix = std::string(SHARED_LIB_MEMORY_MODEL) + "own_";
        mlir_ts::GlobalOp marker;
        for (auto &entry : globals)
        {
            if (entry.getKey().starts_with(markerPrefix))
            {
                marker = entry.second;
                break;
            }
        }

        if (!marker)
        {
            return;
        }

        llvm::SmallVector<llvm::StringRef> names;
        for (auto &entry : functions)
        {
            auto funcOp = entry.second;
            if (!funcOp.isDeclaration() && !funcOp.isPrivate() && noDrops.contains(funcOp))
            {
                names.push_back(funcOp.getSymName());
            }
        }

        llvm::sort(names);
        std::string text;
        for (auto name : names)
        {
            text += name.str();
            text += '\n';
        }

        auto suffix = marker.getSymName().drop_front(markerPrefix.size());
        auto name = (llvm::Twine(SHARED_LIB_OWN_FACTS) + suffix).str();

        mlir::OpBuilder builder(marker);
        builder.setInsertionPointAfter(marker);
        auto factsOp = mlir::cast<mlir_ts::GlobalOp>(builder.clone(*marker.getOperation()));
        factsOp.setSymName(name);
        factsOp.getInitializerRegion().walk([&](mlir_ts::ConstantOp constantOp) {
            if (mlir::isa<mlir::StringAttr>(constantOp.getValue()))
            {
                constantOp.setValueAttr(builder.getStringAttr(text));
            }
        });
    }

    // Does the body itself destroy something a caller may reach: overwrite a field or an element
    // of a block it did not make, assign a global, remove elements from an array it did not make,
    // or `delete`? A local's own releases are not drops, nor is the constructor's filling of the
    // object it is building.
    bool dropsInBody(mlir_ts::FuncOp funcOp)
    {
        auto drops = false;
        funcOp.walk([&](mlir::Operation *op) {
            if (drops)
            {
                return;
            }

            // A handle given back may be the last one, and its payload goes with it, and with that
            // anything the payload owns (spec 23.3). One held by a field or an element is a place's
            // overwrite, judged below like any other: a constructor filling its own `this` is not
            // a drop.
            if (mlir::isa<mlir_ts::ReleaseOp>(op) && touchesHandle(op))
            {
                drops = true;
            }
            else if (auto releaseSlotOp = mlir::dyn_cast<mlir_ts::ReleaseSlotOp>(op))
            {
                // a captured variable assigned through its cell, by a closure's body: the variable
                // is its declaring function's
                auto slot = releaseSlotOp.getSlot();
                drops = slot.getDefiningOp<mlir_ts::AddressOfOp>() || (isPlace(slot) && reachesOutside(slot, funcOp)) ||
                        isLoadedCell(slot) || (!isPlace(slot) && touchesHandle(op));
            }
            else if (mlir::isa<mlir_ts::ArrayPopOp, mlir_ts::ArrayShiftOp, mlir_ts::ArraySpliceOp, mlir_ts::SetLengthOfOp>(op))
            {
                drops = reachesOutside(op->getOperand(0), funcOp);
            }
            else if (mlir::isa<mlir_ts::ArrayPushOp, mlir_ts::ArrayUnshiftOp>(op))
            {
                // through a handle, it may move the elements another handle's borrow reads (spec 23.3)
                drops = reachesSharedValue(op->getOperand(0));
            }
            else if (mlir::isa<mlir_ts::DeleteOp>(op))
            {
                drops = true;
            }
        });

        return drops;
    }

    // Can the block this reference points into be one the function's caller can reach? Walks up
    // from a place to what holds it. Not when every way up ends in a block the function made (an
    // allocation, a call's result it owns) or in an owning local that only ever held such blocks,
    // or, in a constructor, in the object being built.
    bool reachesOutside(mlir::Value ref, mlir_ts::FuncOp funcOp)
    {
        auto isConstructor = funcOp.getSymName().ends_with(".constructor");
        llvm::SmallVector<mlir::Value> work{ref};
        llvm::DenseSet<mlir::Value> seen;
        while (!work.empty())
        {
            auto value = rootOf(work.pop_back_val());
            if (!seen.insert(value).second)
            {
                continue;
            }

            if (auto object = boundThis(value))
            {
                work.push_back(object);
                continue;
            }

            // a handle's payload: any other handle may reach it (spec 23.3)
            if (value.getDefiningOp<mlir_ts::SharedValueRefOp>())
            {
                return true;
            }

            if (auto propertyRefOp = value.getDefiningOp<mlir_ts::PropertyRefOp>())
            {
                work.push_back(propertyRefOp.getObjectRef());
                continue;
            }

            if (auto elementRefOp = value.getDefiningOp<mlir_ts::ElementRefOp>())
            {
                work.push_back(elementRefOp.getArray());
                continue;
            }

            if (auto argument = mlir::dyn_cast<mlir::BlockArgument>(value))
            {
                llvm::SmallVector<mlir::Value> merged;
                if (mergedInto(argument, merged))
                {
                    work.append(merged.begin(), merged.end());
                    continue;
                }

                if (isConstructor && argument.getOwner()->isEntryBlock() && argument.getArgNumber() == 0)
                {
                    continue; // the object being built
                }

                return true;
            }

            if (auto varOp = value.getDefiningOp<mlir_ts::VariableOp>())
            {
                if (holdsOnlyFresh(varOp))
                {
                    continue;
                }

                // a parameter's slot: what the caller passed
                if (auto init = varOp.getInitializer())
                {
                    if (auto argument = mlir::dyn_cast<mlir::BlockArgument>(init);
                        argument && isConstructor && argument.getOwner()->isEntryBlock() && argument.getArgNumber() == 0 &&
                        !isAssigned(varOp))
                    {
                        continue;
                    }
                }

                return true;
            }

            if (auto loadOp = value.getDefiningOp<mlir_ts::LoadOp>())
            {
                work.push_back(loadOp.getReference());
                continue;
            }

            if (isFresh(value))
            {
                continue;
            }

            return true;
        }

        return false;
    }

    static bool isAssigned(mlir_ts::VariableOp varOp)
    {
        return llvm::any_of(varOp.getResult().getUsers(), [&](mlir::Operation *user) {
            auto storeOp = mlir::dyn_cast<mlir_ts::StoreOp>(user);
            return storeOp && storeOp.getReference() == varOp.getResult();
        });
    }

    // An owning, uncaptured local whose every value - its initializer and each assignment - is one
    // the function made.
    static bool holdsOnlyFresh(mlir_ts::VariableOp varOp)
    {
        if (!isOwningVariable(varOp) || varOp.getCaptured().value_or(false))
        {
            return false;
        }

        if (auto init = varOp.getInitializer(); init && !isFresh(rootOf(init)))
        {
            return false;
        }

        return llvm::all_of(varOp.getResult().getUsers(), [&](mlir::Operation *user) {
            auto storeOp = mlir::dyn_cast<mlir_ts::StoreOp>(user);
            return !storeOp || storeOp.getReference() != varOp.getResult() || isFresh(rootOf(storeOp.getValue()));
        });
    }
};

} // end anonymous namespace

#undef DEBUG_TYPE

std::unique_ptr<mlir::Pass> mlir_ts::createOwnershipSignaturePass()
{
    return std::make_unique<OwnershipSignaturePass>();
}
