#ifndef MLIR_TYPESCRIPT_STACKESCAPEFACTS_H
#define MLIR_TYPESCRIPT_STACKESCAPEFACTS_H

#include "TypeScript/TypeScriptOps.h"

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/TypeSwitch.h"

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

class StackEscapeFacts
{
  public:
    explicit StackEscapeFacts(mlir::ModuleOp module) : module(module)
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
        auto it = recursive.find(funcOp);
        if (it != recursive.end())
        {
            return it->second;
        }

        llvm::SmallPtrSet<mlir::Operation *, 16> seen;
        llvm::SmallVector<mlir_ts::FuncOp> work{funcOp};
        auto found = false;
        while (!work.empty() && !found)
        {
            auto current = work.pop_back_val();
            for (auto callee : calleesOfFunction(current))
            {
                auto calleeOp = module.lookupSymbol<mlir_ts::FuncOp>(callee);
                if (!calleeOp)
                {
                    continue;
                }

                if (calleeOp == funcOp)
                {
                    found = true;
                    break;
                }

                if (seen.insert(calleeOp).second)
                {
                    work.push_back(calleeOp);
                }
            }
        }

        recursive[funcOp] = found;
        return found;
    }

  private:
    mlir::ModuleOp module;
    llvm::DenseMap<mlir::Operation *, bool> recursive;

    llvm::SmallVector<mlir::FlatSymbolRefAttr> calleesOfFunction(mlir_ts::FuncOp funcOp)
    {
        llvm::SmallVector<mlir::FlatSymbolRefAttr> callees;
        funcOp.walk([&](mlir::Operation *op) {
            if (auto callOp = dyn_cast<mlir_ts::SymbolCallInternalOp>(op))
            {
                callees.push_back(callOp.getCalleeAttr());
            }
            else if (auto callOp = dyn_cast<mlir_ts::CallInternalOp>(op))
            {
                calleesOf(callOp->getOperand(0), callees);
            }
        });

        return callees;
    }
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
            if (!name.ends_with("..vtbl"))
            {
                continue;
            }

            auto &vtable = vtables[name.drop_back(strlen("..vtbl"))];
            auto linkage = globalOp->getAttrOfType<mlir::LLVM::LinkageAttr>("linkage");
            vtable.local = linkage && (linkage.getLinkage() == mlir::LLVM::Linkage::Private ||
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
                candidates.push_back(slot->second);
            }
        }

        return !candidates.empty();
    }

    // Whether argument `argIndex` of a call through `candidates` may outlive the call.
    llvm::StringRef argumentEscapes(llvm::ArrayRef<mlir::FlatSymbolRefAttr> candidates, unsigned argIndex)
    {
        for (auto candidate : candidates)
        {
            auto funcOp = module.lookupSymbol<mlir_ts::FuncOp>(candidate);
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

    llvm::StringRef useEscapes(mlir::Value value, mlir::OpOperand &use, Walk &walk)
    {
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
            .Case<mlir_ts::LoadOp>([&](auto) { return ""; })
            .Case<mlir_ts::RetainOp, mlir_ts::ReleaseOp>([&](auto) { return ""; })
            .Case<mlir_ts::LogicalBinaryOp>([&](auto) { return ""; })
            .Case<mlir_ts::CastOp>([&](auto castOp) -> llvm::StringRef {
                // a view of the same object - a boxed literal seen as an object, a class as its
                // own type - is the object; anything else converts it
                if (isa<mlir_ts::ObjectType, mlir_ts::ClassType, mlir_ts::ValueRefType>(castOp.getType()))
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
                if (use.getOperandNumber() == 0)
                {
                    return "unclassified";
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
