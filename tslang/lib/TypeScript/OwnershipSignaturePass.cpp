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
        collectClassVTables(module);
        collectFieldFunctions(module);

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

        for (auto &call : calls)
        {
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

    // (tuple type, field) -> the one function an object of that type holds in that field: every
    // `ts.Constant` of the type's shape puts it there, nothing stores into the field, and nothing
    // but a constant or a read makes a value of the type. A generator's state object holds its
    // `next` so.
    llvm::DenseMap<std::pair<mlir::Type, int64_t>, mlir::StringAttr> fieldFunctions;

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

    // The tuple an object, a reference or a tuple value of this type stores its fields in; null for
    // any other type. A method sees its object as `!ts.object<!ts.object_storage<@name>>`, whose
    // fields are the tuple's.
    mlir_ts::TupleType storageTupleOf(mlir::Type type)
    {
        if (auto objectType = mlir::dyn_cast<mlir_ts::ObjectType>(type))
        {
            type = objectType.getStorageType();
            if (auto storageType = mlir::dyn_cast<mlir_ts::ObjectStorageType>(type))
            {
                return mlir_ts::TupleType::get(&getContext(), storageType.getFields());
            }
        }
        else if (auto refType = mlir::dyn_cast<mlir_ts::RefType>(type))
        {
            type = refType.getElementType();
        }
        else if (auto valueRefType = mlir::dyn_cast<mlir_ts::ValueRefType>(type))
        {
            type = valueRefType.getElementType();
        }

        return mlir::dyn_cast<mlir_ts::TupleType>(type);
    }

    void collectFieldFunctions(mlir::ModuleOp module)
    {
        auto *context = &getContext();
        llvm::DenseSet<std::pair<mlir::Type, int64_t>> unknownFields;
        llvm::DenseSet<mlir::Type> unknownTuples;
        module.walk([&](mlir::Operation *op) {
            if (auto constantOp = mlir::dyn_cast<mlir_ts::ConstantOp>(op))
            {
                auto constTupleType = mlir::dyn_cast<mlir_ts::ConstTupleType>(constantOp.getType());
                auto values = mlir::dyn_cast<mlir::ArrayAttr>(constantOp.getValue());
                if (!constTupleType || !values)
                {
                    return;
                }

                mlir::Type tupleType = mlir_ts::TupleType::get(context, constTupleType.getFields());
                for (auto [index, value] : llvm::enumerate(values))
                {
                    auto symbol = mlir::dyn_cast<mlir::FlatSymbolRefAttr>(value);
                    if (!symbol)
                    {
                        continue;
                    }

                    std::pair<mlir::Type, int64_t> key{tupleType, static_cast<int64_t>(index)};
                    auto [found, inserted] = fieldFunctions.try_emplace(key, symbol.getAttr());
                    if (!inserted && found->second != symbol.getAttr())
                    {
                        unknownFields.insert(key);
                    }
                }

                return;
            }

            // a field reference that is not only read: the field may be given anything
            if (auto propertyRefOp = mlir::dyn_cast<mlir_ts::PropertyRefOp>(op))
            {
                auto tupleType = storageTupleOf(propertyRefOp->getOperand(0).getType());
                auto onlyRead = llvm::all_of(propertyRefOp->getUsers(),
                                             [](mlir::Operation *user) { return mlir::isa<mlir_ts::LoadOp>(user); });
                if (tupleType && !onlyRead)
                {
                    unknownFields.insert({tupleType, propertyRefOp.getPosition()});
                }

                return;
            }

            // a value of the type made any other way may hold anything in any field
            if (mlir::isa<mlir_ts::LoadOp>(op))
            {
                return;
            }

            for (auto result : op->getResults())
            {
                if (auto tupleType = mlir::dyn_cast<mlir_ts::TupleType>(result.getType()))
                {
                    unknownTuples.insert(tupleType);
                }
            }
        });

        for (auto &key : unknownFields)
        {
            fieldFunctions.erase(key);
        }

        llvm::SmallVector<std::pair<mlir::Type, int64_t>> dropped;
        for (auto &entry : fieldFunctions)
        {
            if (unknownTuples.contains(entry.first.first))
            {
                dropped.push_back(entry.first);
            }
        }

        for (auto &key : dropped)
        {
            fieldFunctions.erase(key);
        }
    }

    // `ts.Load(ts.PropertyRef(object, p))` of a field that only ever holds one function.
    mlir::StringAttr fieldFunctionOf(mlir::Operation *def)
    {
        auto loadOp = mlir::dyn_cast_or_null<mlir_ts::LoadOp>(def);
        auto propertyRefOp = loadOp ? loadOp.getReference().getDefiningOp<mlir_ts::PropertyRefOp>() : mlir_ts::PropertyRefOp();
        auto tupleType = propertyRefOp ? storageTupleOf(propertyRefOp->getOperand(0).getType()) : mlir_ts::TupleType();
        if (!tupleType)
        {
            return {};
        }

        return fieldFunctions.lookup({tupleType, propertyRefOp.getPosition()});
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
    // linked with this module, through its import library.
    bool addImported(mlir::StringAttr symbol, Callees &callees)
    {
        auto funcOp = functions.lookup(symbol.getValue());
        if (!funcOp || !funcOp.isDeclaration() || !importedNoDrops.contains(symbol.getValue()))
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
        else if (auto symbol = fieldFunctionOf(def))
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
            callees.known = addDefined(identifier, callees) || addImported(identifier, callees);
            return;
        }

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
                open.try_emplace(entry.second, "it can be called from another module");
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
                if (auto facts = callFacts(call.callees); !facts.empty())
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
                    if (auto funcOp = functions.lookup(member.getValue()); funcOp && factsEffective.erase(funcOp))
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
    bool calledOnlyDirectly(mlir::ModuleOp module, mlir_ts::FuncOp funcOp)
    {
        if (!funcOp.isPrivate())
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
            // an open maker has callers this pass cannot give the bound to
            if (captured < 0 || !maker || !calledOnlyDirectly(module, maker))
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
