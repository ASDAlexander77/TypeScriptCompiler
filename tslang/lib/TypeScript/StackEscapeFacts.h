#ifndef MLIR_TYPESCRIPT_STACKESCAPEFACTS_H
#define MLIR_TYPESCRIPT_STACKESCAPEFACTS_H

#include "TypeScript/TypeScriptOps.h"
#include "TypeScript/Defines.h"

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Interfaces/CallInterfaces.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/TypeSwitch.h"

#include <optional>

// Whether an object's pointer may outlive the function holding it - the question stack promotion
// asks of every allocation, and of every parameter of every function an object is passed to (spec
// docs/superpowers/specs/2026-10-18-stack-promotion-design.md section 5.2).
//
// A parameter's answer is a property of the callee's body, valid whoever calls it: unlike own's
// facts it needs no closed world. It is computed for the whole module up front, to a fixpoint,
// because a parameter passed on to another function escapes when that one's does. A callee this
// module has no body for - declared, imported, reached through a value - may keep anything.
//
// Kept apart from the pass so a later version can read the same facts from a declaration
// (section 8.2) as easily as compute them from a body.
namespace stack_escape
{

namespace mlir_ts = mlir::typescript;

// Facts another module published (its __tsborrow_ text): function symbol -> the indexes of the
// parameters its body only borrows.
using BorrowFactsMap = llvm::StringMap<llvm::SmallVector<unsigned>>;

// `<symbol> <index>...` per line, as BorrowFactsExportPass writes it. A line that does not parse
// says nothing: every parameter of a function no line names may escape.
inline BorrowFactsMap parseBorrowFacts(llvm::StringRef text)
{
    BorrowFactsMap facts;
    llvm::SmallVector<llvm::StringRef> lines;
    text.split(lines, '\n', -1, /*KeepEmpty=*/false);
    for (auto line : lines)
    {
        llvm::SmallVector<llvm::StringRef> words;
        line.trim().split(words, ' ', -1, /*KeepEmpty=*/false);
        if (words.size() < 2)
        {
            continue;
        }

        llvm::SmallVector<unsigned> indexes;
        for (auto word : llvm::drop_begin(words))
        {
            unsigned index;
            if (word.getAsInteger(10, index))
            {
                indexes.clear();
                break;
            }

            indexes.push_back(index);
        }

        auto &listed = facts[words[0]];
        listed.append(indexes.begin(), indexes.end());
    }

    return facts;
}

class StackEscapeFacts
{
  public:
    // trustLocalVTables: whether a vtable this module keeps to itself lists every method a virtual
    // call may reach. Not when the facts are for importers, who may subclass any class.
    explicit StackEscapeFacts(mlir::ModuleOp module, bool trustLocalVTables = true,
                              const BorrowFactsMap *imported = nullptr)
        : module(module), trustLocalVTables(trustLocalVTables), imported(imported)
    {
        collectVTables();

        llvm::SmallVector<std::pair<mlir_ts::FuncOp, unsigned>> params;
        module.walk([&](mlir_ts::FuncOp funcOp) {
            if (funcOp.getBody().empty())
            {
                return;
            }

            for (auto index : llvm::seq<unsigned>(0, funcOp.getBody().getNumArguments()))
            {
                params.push_back({funcOp, index});
                paramEscapes[{funcOp, index}] = false;
            }
        });

        // facts only go from "kept inside" to "may escape", so this ends
        for (auto changed = true; changed;)
        {
            changed = false;
            for (auto [funcOp, index] : params)
            {
                if (paramEscapes[{funcOp, index}])
                {
                    continue;
                }

                Walk walk;
                if (!escapes(funcOp.getBody().getArgument(index), walk).empty())
                {
                    paramEscapes[{funcOp, index}] = true;
                    changed = true;
                }
            }
        }
    }

    // What one question visited: the values that are the object (aliases through casts, slot loads
    // and `this`), and the local slots it was stored into. The pass's loop rule reads both.
    struct Walk
    {
        llvm::SmallPtrSet<mlir::Value, 16> aliases;
        llvm::SmallPtrSet<mlir::Operation *, 4> slots;
    };

    // Empty when nothing done with `value` lets it outlive the function; otherwise why it may.
    llvm::StringRef escapes(mlir::Value value, Walk &walk)
    {
        if (!walk.aliases.insert(value).second)
        {
            return "";
        }

        for (auto &use : value.getUses())
        {
            auto reason = useEscapes(value, use, walk);
            if (!reason.empty())
            {
                return reason;
            }
        }

        return "";
    }

    // On a cycle of the module's call graph - calls by name, and virtual calls through every
    // method they may reach.
    bool isRecursive(mlir_ts::FuncOp funcOp)
    {
        if (!recursive)
        {
            collectRecursive();
        }

        return recursive->contains(funcOp);
    }

    // A parameter of a function with a body that nothing done with it lets outlive the call.
    bool paramBorrowed(mlir_ts::FuncOp funcOp, unsigned index) const
    {
        auto it = paramEscapes.find({funcOp, index});
        return it != paramEscapes.end() && !it->second;
    }

    // Whether argument `argIndex` of a call through `candidates` may outlive the call: empty when
    // every candidate's parameter stays inside it.
    llvm::StringRef argumentEscapes(llvm::ArrayRef<mlir::FlatSymbolRefAttr> candidates, unsigned argIndex)
    {
        for (auto candidate : candidates)
        {
            auto funcOp = module.lookupSymbol<mlir_ts::FuncOp>(candidate);
            // declared here, defined in a library linked statically that published its facts - by
            // the name it binds: one renamed by @dllname / @linkname binds another symbol
            if (funcOp && funcOp.getBody().empty() && !funcOp->hasAttr(DLL_NAME) &&
                importedBorrows(candidate.getValue(), argIndex))
            {
                continue;
            }

            if (!funcOp || funcOp.getBody().empty() || argIndex >= funcOp.getBody().getNumArguments())
            {
                return "unknown-callee";
            }

            if (paramEscapes.lookup({funcOp, argIndex}))
            {
                return "argument";
            }
        }

        return "";
    }

  private:
    bool importedBorrows(llvm::StringRef symbol, unsigned argIndex) const
    {
        if (!imported)
        {
            return false;
        }

        auto it = imported->find(symbol);
        return it != imported->end() && llvm::is_contained(it->second, argIndex);
    }

    mlir::ModuleOp module;
    bool trustLocalVTables;
    const BorrowFactsMap *imported;
    std::optional<llvm::SmallPtrSet<mlir::Operation *, 16>> recursive;

    // Every function on a cycle of the call graph, in one pass (Tarjan's strongly connected
    // components, iterative: a module's call chains may be deeper than the native stack): a
    // function is recursive when its component holds another function, or it calls itself.
    void collectRecursive()
    {
        recursive.emplace();

        llvm::SmallVector<mlir_ts::FuncOp> funcs;
        llvm::DenseMap<mlir::Operation *, unsigned> numberOf;
        module.walk([&](mlir_ts::FuncOp funcOp) {
            numberOf[funcOp] = funcs.size();
            funcs.push_back(funcOp);
        });

        // the edges, each callee once; a callee with no definition here is on no cycle of ours
        llvm::SmallVector<llvm::SmallVector<unsigned>> edges(funcs.size());
        llvm::SmallVector<bool> callsItself(funcs.size(), false);
        for (auto [index, funcOp] : llvm::enumerate(funcs))
        {
            llvm::SmallPtrSet<mlir::Operation *, 16> seen;
            for (auto callee : calleesOfFunction(funcOp))
            {
                auto calleeOp = module.lookupSymbol<mlir_ts::FuncOp>(callee);
                if (!calleeOp || !seen.insert(calleeOp).second)
                {
                    continue;
                }

                auto calleeIndex = numberOf.lookup(calleeOp);
                callsItself[index] = callsItself[index] || calleeIndex == index;
                edges[index].push_back(calleeIndex);
            }
        }

        constexpr unsigned unvisited = ~0u;
        llvm::SmallVector<unsigned> order(funcs.size(), unvisited);
        llvm::SmallVector<unsigned> lowLink(funcs.size(), 0);
        llvm::SmallVector<bool> onStack(funcs.size(), false);
        llvm::SmallVector<unsigned> stack;
        // the depth-first path: a function and the index of the next of its edges to follow
        llvm::SmallVector<std::pair<unsigned, unsigned>> path;
        unsigned nextOrder = 0;

        for (auto root : llvm::seq<unsigned>(0, funcs.size()))
        {
            if (order[root] != unvisited)
            {
                continue;
            }

            auto visit = [&](unsigned node) {
                order[node] = lowLink[node] = nextOrder++;
                stack.push_back(node);
                onStack[node] = true;
                path.push_back({node, 0});
            };

            visit(root);
            while (!path.empty())
            {
                auto &[node, edgeIndex] = path.back();
                if (edgeIndex < edges[node].size())
                {
                    auto callee = edges[node][edgeIndex++];
                    if (order[callee] == unvisited)
                    {
                        visit(callee);
                    }
                    else if (onStack[callee])
                    {
                        lowLink[node] = std::min(lowLink[node], order[callee]);
                    }

                    continue;
                }

                // every edge followed: `node` closes a component when nothing below reached higher
                auto done = node;
                path.pop_back();
                if (!path.empty())
                {
                    auto parent = path.back().first;
                    lowLink[parent] = std::min(lowLink[parent], lowLink[done]);
                }

                if (lowLink[done] != order[done])
                {
                    continue;
                }

                llvm::SmallVector<unsigned> component;
                unsigned member;
                do
                {
                    member = stack.pop_back_val();
                    onStack[member] = false;
                    component.push_back(member);
                } while (member != done);

                if (component.size() > 1 || callsItself[done])
                {
                    for (auto index : component)
                    {
                        recursive->insert(funcs[index]);
                    }
                }
            }
        }
    }

    // What a function may call. A call through a value the module cannot resolve may reach any
    // function whose address is taken somewhere in it - a closure, a function stored in a variable -
    // which is how a recursion through `const g = f; g()` or a closure is still seen.
    llvm::SmallVector<mlir::FlatSymbolRefAttr> calleesOfFunction(mlir_ts::FuncOp funcOp)
    {
        llvm::SmallVector<mlir::FlatSymbolRefAttr> callees;
        auto throughValue = false;
        funcOp.walk([&](mlir::Operation *op) {
            if (auto callOp = dyn_cast<mlir_ts::SymbolCallInternalOp>(op))
            {
                callees.push_back(callOp.getCalleeAttr());
            }
            else if (auto callOp = dyn_cast<mlir_ts::CallInternalOp>(op))
            {
                throughValue |= !calleesOf(callOp->getOperand(0), callees);
            }
            else if (isa<mlir::CallOpInterface>(op))
            {
                throughValue = true;
            }
        });

        if (throughValue)
        {
            llvm::append_range(callees, addressTaken());
        }

        return callees;
    }

    // The functions whose address a function of the module takes: closures, function values.
    llvm::ArrayRef<mlir::FlatSymbolRefAttr> addressTaken()
    {
        if (!addressTakenFunctions)
        {
            addressTakenFunctions.emplace();
            llvm::SmallPtrSet<mlir::Operation *, 16> seen;
            module.walk([&](mlir_ts::FuncOp funcOp) {
                funcOp.walk([&](mlir_ts::SymbolRefOp symbolRefOp) {
                    auto target = module.lookupSymbol<mlir_ts::FuncOp>(symbolRefOp.getIdentifierAttr());
                    if (target && seen.insert(target).second)
                    {
                        addressTakenFunctions->push_back(symbolRefOp.getIdentifierAttr());
                    }
                });
            });
        }

        return *addressTakenFunctions;
    }

    std::optional<llvm::SmallVector<mlir::FlatSymbolRefAttr>> addressTakenFunctions;
    llvm::DenseMap<std::pair<mlir::Operation *, unsigned>, bool> paramEscapes;

    struct VTable
    {
        bool local = false;
        llvm::DenseMap<int64_t, mlir::FlatSymbolRefAttr> slots;
    };
    llvm::StringMap<VTable> vtables;

    // `<class>..vtbl`: which method each slot holds, and whether the class is this module's alone.
    // A vtable visible outside the module may have subclasses elsewhere, overriding anything.
    void collectVTables()
    {
        for (auto globalOp : module.getOps<mlir_ts::GlobalOp>())
        {
            auto name = globalOp.getSymName();
            // an interface's null vtable is no class's (mlirGenNullInterfaceVTable)
            if (!name.ends_with("..vtbl") || globalOp->hasAttr(NULL_INTERFACE_VTABLE_ATTR_NAME))
            {
                continue;
            }

            auto &vtable = vtables[name.drop_back(strlen("..vtbl"))];
            auto linkage = globalOp->getAttrOfType<mlir::LLVM::LinkageAttr>("linkage");
            vtable.local = trustLocalVTables && linkage && (linkage.getLinkage() == mlir::LLVM::Linkage::Private ||
                                       linkage.getLinkage() == mlir::LLVM::Linkage::Internal);
            globalOp.walk([&](mlir_ts::InsertPropertyOp insertOp) {
                auto symbolRef = insertOp.getValue().getDefiningOp<mlir_ts::SymbolRefOp>();
                if (symbolRef && insertOp.getPosition().size() == 1)
                {
                    vtable.slots[insertOp.getPosition()[0]] = symbolRef.getIdentifierAttr();
                }
            });
        }
    }

    static llvm::StringRef methodName(llvm::StringRef symbol)
    {
        return symbol.substr(symbol.rfind('.') + 1);
    }

    // The functions a virtual call through slot `index` of `className`'s vtable may run: that
    // slot's method in every vtable of the module holding a method of the same name there - the
    // class's own and its subclasses', and possibly unrelated classes', which only costs precision.
    bool virtualCandidates(llvm::StringRef className, int64_t index, llvm::StringRef method,
                           llvm::SmallVectorImpl<mlir::FlatSymbolRefAttr> &candidates)
    {
        auto it = vtables.find(className);
        if (it == vtables.end() || !it->second.local)
        {
            return false;
        }

        for (auto &entry : vtables)
        {
            auto slot = entry.second.slots.find(index);
            if (slot != entry.second.slots.end() && methodName(slot->second.getValue()) == method)
            {
                // a class visible outside the module - a subclass of the receiver's, exported - may
                // be extended there, and the override there is not one this module can see
                if (!entry.second.local)
                {
                    return false;
                }

                candidates.push_back(slot->second);
            }
        }

        return !candidates.empty();
    }

    // The functions a call's callee value may be, when the module can tell.
    bool calleesOf(mlir::Value callee, llvm::SmallVectorImpl<mlir::FlatSymbolRefAttr> &candidates)
    {
        if (auto symbolRefOp = callee.getDefiningOp<mlir_ts::SymbolRefOp>())
        {
            candidates.push_back(symbolRefOp.getIdentifierAttr());
            return true;
        }

        auto getMethodOp = callee.getDefiningOp<mlir_ts::GetMethodOp>();
        if (!getMethodOp)
        {
            return false;
        }

        auto boundMethod = getMethodOp->getOperand(0);
        if (auto thisSymbolRefOp = boundMethod.getDefiningOp<mlir_ts::ThisSymbolRefOp>())
        {
            candidates.push_back(thisSymbolRefOp.getIdentifierAttr());
            return true;
        }

        if (auto virtualRefOp = boundMethod.getDefiningOp<mlir_ts::ThisVirtualSymbolRefOp>())
        {
            auto classType = dyn_cast<mlir_ts::ClassType>(virtualRefOp.getThisVal().getType());
            return classType && virtualCandidates(classType.getName().getValue(), virtualRefOp.getIndex(),
                                                  methodName(virtualRefOp.getIdentifier()), candidates);
        }

        return false;
    }

    // A bound method made from the object (`p.m` before `p.m()`): its `this` is the object, its
    // method is called. Kept to those two uses, it lets nothing out by itself.
    llvm::StringRef boundMethodEscapes(mlir::Value boundMethod, Walk &walk)
    {
        for (auto *user : boundMethod.getUsers())
        {
            if (auto getThisOp = dyn_cast<mlir_ts::GetThisOp>(user))
            {
                auto reason = escapes(getThisOp.getResult(), walk);
                if (!reason.empty())
                {
                    return reason;
                }

                continue;
            }

            if (isa<mlir_ts::GetMethodOp>(user))
            {
                continue;
            }

            return "unclassified";
        }

        return "";
    }

    // A local slot the object was stored into: it is the object wherever it is loaded. A slot a
    // closure captures, or one whose address is handed on, is beyond what is followed here.
    llvm::StringRef slotEscapes(mlir_ts::VariableOp slot, Walk &walk)
    {
        if (slot.getCaptured().has_value() && slot.getCaptured().value())
        {
            return "captured";
        }

        if (!walk.slots.insert(slot).second)
        {
            return "";
        }

        for (auto &use : slot.getResult().getUses())
        {
            auto *user = use.getOwner();
            if (auto loadOp = dyn_cast<mlir_ts::LoadOp>(user))
            {
                auto reason = escapes(loadOp.getResult(), walk);
                if (!reason.empty())
                {
                    return reason;
                }

                continue;
            }

            if (auto storeOp = dyn_cast<mlir_ts::StoreOp>(user); storeOp && storeOp.getValue() != slot.getResult())
            {
                continue;
            }

            if (isa<mlir_ts::RetainSlotOp, mlir_ts::ReleaseSlotOp>(user))
            {
                continue;
            }

            // an array changed in place through the slot holding it
            if (isGrowth(user) && use.getOperandNumber() == 0)
            {
                return "growable";
            }

            return "unclassified";
        }

        return "";
    }

    // A reference into the object: only loaded through or stored through. Anything else - a call
    // taking it, a store of the reference itself - lets the object's address out.
    static llvm::StringRef fieldRefEscapes(mlir::Value fieldRef)
    {
        for (auto *user : fieldRef.getUsers())
        {
            // a release of the old value before a store, a retain of what it holds
            if (isa<mlir_ts::LoadOp, mlir_ts::RetainSlotOp, mlir_ts::ReleaseSlotOp>(user))
            {
                continue;
            }

            if (auto storeOp = dyn_cast<mlir_ts::StoreOp>(user);
                storeOp && storeOp.getReference() == fieldRef && storeOp.getValue() != fieldRef)
            {
                continue;
            }

            return "unclassified";
        }

        return "";
    }

    // What may change an array's length - and so move its elements to a block of another size.
    static bool isGrowth(mlir::Operation *op)
    {
        return isa<mlir_ts::ArrayPushOp, mlir_ts::ArrayPopOp, mlir_ts::ArrayUnshiftOp, mlir_ts::ArrayShiftOp,
                   mlir_ts::ArraySpliceOp, mlir_ts::SetLengthOfOp>(op);
    }

    llvm::StringRef useEscapes(mlir::Value value, mlir::OpOperand &use, Walk &walk)
    {
        if (isGrowth(use.getOwner()))
        {
            // the array itself, or a value going into it
            return use.getOperandNumber() == 0 ? "growable" : "stored";
        }

        auto *user = use.getOwner();
        return llvm::TypeSwitch<mlir::Operation *, llvm::StringRef>(user)
            .Case<mlir_ts::PropertyRefOp>([&](auto propertyRefOp) { return fieldRefEscapes(propertyRefOp.getResult()); })
            .Case<mlir_ts::StoreOp>([&](auto storeOp) -> llvm::StringRef {
                // into the object's own storage
                if (storeOp.getReference() == value && storeOp.getValue() != value)
                {
                    return "";
                }

                if (auto slot = storeOp.getReference().template getDefiningOp<mlir_ts::VariableOp>())
                {
                    return slotEscapes(slot, walk);
                }

                return "stored";
            })
            .Case<mlir_ts::VariableOp>([&](auto slot) { return slotEscapes(slot, walk); })
            .Case<mlir_ts::LoadOp, mlir_ts::LengthOfOp>([&](auto) { return ""; })
            .Case<mlir_ts::ElementRefOp>([&](auto elementRefOp) -> llvm::StringRef {
                return use.getOperandNumber() == 0 ? fieldRefEscapes(elementRefOp.getResult()) : "unclassified";
            })
            .Case<mlir_ts::RetainOp, mlir_ts::ReleaseOp>([&](auto) { return ""; })
            .Case<mlir_ts::LogicalBinaryOp>([&](auto) { return ""; })
            .Case<mlir_ts::CastOp>([&](auto castOp) -> llvm::StringRef {
                // a view of the same object - a boxed literal seen as an object, a class as its
                // own type - is the object; anything else converts it
                if (isa<mlir_ts::ObjectType, mlir_ts::ClassType, mlir_ts::ValueRefType, mlir_ts::BoundFunctionType,
                        mlir_ts::HybridFunctionType, mlir_ts::FunctionType>(castOp.getType()))
                {
                    return escapes(castOp.getResult(), walk);
                }

                return "converted";
            })
            .Case<mlir_ts::NewInterfaceOp>([&](auto) { return "converted"; })
            .Case<mlir_ts::ThisVirtualSymbolRefOp, mlir_ts::ThisSymbolRefOp>([&](auto refOp) -> llvm::StringRef {
                if (use.getOperandNumber() != 0)
                {
                    return "unclassified";
                }

                return boundMethodEscapes(refOp.getResult(), walk);
            })
            .Case<mlir_ts::SymbolCallInternalOp>([&](auto callOp) {
                return argumentEscapes({callOp.getCalleeAttr()}, use.getOperandNumber());
            })
            .Case<mlir_ts::CallInternalOp>([&](auto callOp) -> llvm::StringRef {
                // a closure being called: running it hands its capture box to nothing
                if (use.getOperandNumber() == 0)
                {
                    return "";
                }

                llvm::SmallVector<mlir::FlatSymbolRefAttr> candidates;
                if (!calleesOf(callOp->getOperand(0), candidates))
                {
                    return "unknown-callee";
                }

                return argumentEscapes(candidates, use.getOperandNumber() - 1);
            })
            .Case<mlir_ts::ReturnInternalOp, mlir_ts::ReturnValOp>([&](auto) { return "returned"; })
            .Case<mlir_ts::ThrowOp, mlir_ts::ThrowCallOp, mlir_ts::ThrowUnwindOp>([&](auto) { return "thrown"; })
            .Default([&](auto) { return "unclassified"; });
    }
};

} // namespace stack_escape

#endif // MLIR_TYPESCRIPT_STACKESCAPEFACTS_H
