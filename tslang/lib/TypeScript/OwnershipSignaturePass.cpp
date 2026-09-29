#include "mlir/Pass/Pass.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/SymbolTable.h"

#include "TypeScript/TypeScriptDialect.h"
#include "TypeScript/TypeScriptOps.h"
#include "TypeScript/Passes.h"
#include "TypeScript/Defines.h"

#include "OwnershipFacts.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
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
class OwnershipSignaturePass : public mlir::PassWrapper<OwnershipSignaturePass, mlir::OperationPass<mlir::ModuleOp>>
{
  public:
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(OwnershipSignaturePass)

    void runOnOperation() override
    {
        auto module = getOperation();
        module.walk([&](mlir_ts::FuncOp funcOp) { functions[funcOp.getSymName()] = funcOp; });
        collectClassVTables(module);

        module.walk([&](mlir::Operation *op) {
            if (isCall(op) && op->getParentOfType<mlir_ts::FuncOp>())
            {
                calls.push_back({op, resolve(op)});
            }
        });

        findOpen(module);
        computeFacts();
        computeDrops();

        for (auto &call : calls)
        {
            if (!call.callees.known)
            {
                continue;
            }

            if (call.callees.instanceOf ||
                llvm::all_of(call.callees.funcs, [&](mlir_ts::FuncOp callee) { return noDrops.contains(callee); }))
            {
                call.op->setAttr(OWN_NO_DROPS_ATTR_NAME, mlir::UnitAttr::get(&getContext()));
            }
        }
    }

  private:
    // What a call may reach: known, when every candidate is a function defined in this module (or
    // the call goes through the `.instanceOf` slot); else anything.
    struct Callees
    {
        bool known = false;
        bool instanceOf = false;
        llvm::SmallVector<mlir_ts::FuncOp, 2> funcs;
    };

    struct Call
    {
        mlir::Operation *op;
        Callees callees;
    };

    llvm::StringMap<mlir_ts::FuncOp> functions;
    llvm::SmallVector<Call> calls;

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

    Callees resolve(mlir::Operation *op)
    {
        Callees callees;
        if (auto callOp = mlir::dyn_cast<mlir_ts::SymbolCallInternalOp>(op))
        {
            callees.known = addDefined(callOp.getCalleeAttr().getAttr(), callees);
        }
        else if (auto callOp = mlir::dyn_cast<mlir_ts::CallOp>(op))
        {
            callees.known = addDefined(callOp.getCalleeAttr().getAttr(), callees);
        }
        else if (mlir::isa<mlir_ts::CallInternalOp, mlir_ts::CallIndirectOp>(op))
        {
            resolveValue(op->getOperand(0), callees);
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

        if (!identifier)
        {
            return;
        }

        if (index < 0)
        {
            callees.known = addDefined(identifier, callees);
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

    // `ts.Cast(ts.VTableOffsetRef(ts.VTableOffsetRef(object, 0), 0))`: the first slot of the
    // vtable an object's first word points to. Every class vtable keeps its generated
    // `..instanceOf` there, a string compare that destroys nothing. `___unbox` asks it.
    static bool isInstanceOfSlot(mlir::Operation *def)
    {
        auto castOp = mlir::dyn_cast_or_null<mlir_ts::CastOp>(def);
        auto slot = castOp ? castOp.getIn().getDefiningOp<mlir_ts::VTableOffsetRefOp>() : mlir_ts::VTableOffsetRefOp();
        auto vtable = slot ? slot.getVtable().getDefiningOp<mlir_ts::VTableOffsetRefOp>() : mlir_ts::VTableOffsetRefOp();
        return vtable && slot.getIndex() == 0 && vtable.getIndex() == 0;
    }

    // Is every use of this function value the callee of a call - through `ts.GetMethod`, with
    // `ts.GetThis` reading the object beside it?
    static bool onlyCalled(mlir::Value value)
    {
        for (auto &use : value.getUses())
        {
            auto *user = use.getOwner();
            if (mlir::isa<mlir_ts::CallInternalOp, mlir_ts::CallIndirectOp>(user) && use.getOperandNumber() == 0)
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
            if (mlir::isa<mlir_ts::SymbolCallInternalOp, mlir_ts::CallOp>(user))
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

            auto inFunction = !!user->getParentOfType<mlir_ts::FuncOp>();
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

        return !callees.instanceOf &&
               llvm::any_of(callees.funcs, [&](mlir_ts::FuncOp callee) { return !noDrops.contains(callee); });
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

            if (auto releaseSlotOp = mlir::dyn_cast<mlir_ts::ReleaseSlotOp>(op))
            {
                auto slot = releaseSlotOp.getSlot();
                drops = slot.getDefiningOp<mlir_ts::AddressOfOp>() || (isPlace(slot) && reachesOutside(slot, funcOp));
            }
            else if (mlir::isa<mlir_ts::ArrayPopOp, mlir_ts::ArrayShiftOp, mlir_ts::ArraySpliceOp, mlir_ts::SetLengthOfOp>(op))
            {
                drops = reachesOutside(op->getOperand(0), funcOp);
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
