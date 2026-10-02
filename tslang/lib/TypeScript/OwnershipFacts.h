#ifndef MLIR_TYPESCRIPT_OWNERSHIPFACTS_H
#define MLIR_TYPESCRIPT_OWNERSHIPFACTS_H

#include "TypeScript/TypeScriptOps.h"
#include "TypeScript/Defines.h"
#include "TypeScript/MLIRLogic/MLIRTypeHelper.h"

#include "mlir/Interfaces/ControlFlowInterfaces.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/STLFunctionalExtras.h"

// What -mm=own's two passes both need to read the ownership ops: which op is a view of the block
// its operand holds, where a value was made, which ops are calls and places. See
// docs/superpowers/specs/2026-09-24-own-memory-model-design.md, sections 3.2, 3.3 and 14.

namespace own_facts
{

namespace mlir_ts = mlir::typescript;


// The facts OwnershipSignaturePass pins on a call it resolved, and on a function whose callers all
// know them (spec 2.4, 3.2). Absent means rc's convention: every parameter borrowed, the result
// owned, and the callee may destroy anything it can reach.
//
// The parameters the callee keeps: the caller moves each such argument into the call.
#define OWN_PARAMS_ATTR_NAME "__own_params"
// The argument the result is a borrow of: the caller owns nothing it gets back.
#define OWN_RESULT_BORROWS_ATTR_NAME "__own_result_borrows"
// The callee destroys nothing its caller can reach: no overwrite of a place it did not make, no
// assignment of a global, no call that may.
#define OWN_NO_DROPS_ATTR_NAME "__own_no_drops"
// On a function whose body relies on a fact its callers cannot all know (a parameter it keeps, a
// result that borrows): why, for the error the body gets instead.
#define OWN_FACTS_LOST_ATTR_NAME "__own_facts_lost"
// A call of another module's `<class>..new`, from a library built under own: it allocates the
// object and hands it over, so the result is fresh (isFresh). Only such a library says so - it lists
// the function among its `__own_no_drops` - and only its blocks are ones this module may destroy:
// a block made under rc still carries a count its own module holds.
#define OWN_FRESH_RESULT_ATTR_NAME "__own_fresh_result"
// On a closure (`ts.CreateBoundFunction`, OWNS_CAPTURE_ATTR_NAME): the fields of its capture box
// whose captured variable its body assigns, itself or through a closure it builds over the same
// cell. A cell that borrows its value (a captured parameter's) cannot be assigned: the old value is
// the caller's.
#define OWN_ASSIGNS_CAPTURES_ATTR_NAME "__own_assigns_captures"
// The arguments a generator's state object borrows from (OWN_BORROWING_FIELDS_ATTR_NAME): the
// result is the caller's to own, but no use of it may come after any of these arguments is given
// back, and it may not be kept anywhere that outlives them.
#define OWN_RESULT_BOUNDED_ATTR_NAME "__own_result_bounded"

// A call inside a try body is a `ts.Invoke` (`ts.InvokeHybrid` through a hybrid function): a
// terminator whose normal and unwind edges are its successors. It names its callee as a symbol, or
// is given the function value as its first call operand.

// The function a call names, or null when it goes through a value.
inline mlir::FlatSymbolRefAttr directCallee(mlir::Operation *op)
{
    if (auto callOp = mlir::dyn_cast<mlir_ts::SymbolCallInternalOp>(op))
    {
        return callOp.getCalleeAttr();
    }

    if (auto callOp = mlir::dyn_cast<mlir_ts::CallOp>(op))
    {
        return callOp.getCalleeAttr();
    }

    if (auto invokeOp = mlir::dyn_cast<mlir_ts::InvokeOp>(op))
    {
        return invokeOp.getCalleeAttr();
    }

    return {};
}

// The function value a call goes through, or null when it names its callee.
inline mlir::Value calleeValue(mlir::Operation *op)
{
    if (mlir::isa<mlir_ts::CallInternalOp, mlir_ts::CallIndirectOp, mlir_ts::CallHybridInternalOp,
                  mlir_ts::InvokeHybridOp>(op))
    {
        return op->getOperand(0);
    }

    if (auto invokeOp = mlir::dyn_cast<mlir_ts::InvokeOp>(op); invokeOp && !invokeOp.getCalleeAttr())
    {
        return invokeOp.getCallOperands().front();
    }

    return {};
}

// The arguments of a call, without the callee value of an indirect one.
inline mlir::OperandRange callArgs(mlir::Operation *op)
{
    if (mlir::isa<mlir_ts::CallInternalOp, mlir_ts::CallIndirectOp, mlir_ts::CallHybridInternalOp>(op))
    {
        return op->getOperands().drop_front();
    }

    if (auto invokeOp = mlir::dyn_cast<mlir_ts::InvokeOp>(op))
    {
        return invokeOp.getCalleeAttr() ? invokeOp.getCallOperands() : invokeOp.getCallOperands().drop_front();
    }

    if (auto invokeOp = mlir::dyn_cast<mlir_ts::InvokeHybridOp>(op))
    {
        return invokeOp.getCallOperands();
    }

    return op->getOperands();
}

inline bool callNoDrops(mlir::Operation *op)
{
    return op->hasAttr(OWN_NO_DROPS_ATTR_NAME);
}

// The argument indices a call's callee keeps.
inline llvm::ArrayRef<int32_t> ownedParams(mlir::Operation *op)
{
    if (auto attr = op->getAttrOfType<mlir::DenseI32ArrayAttr>(OWN_PARAMS_ATTR_NAME))
    {
        return attr.asArrayRef();
    }

    return {};
}

// The argument a call's result borrows, or -1.
inline int resultBorrows(mlir::Operation *op)
{
    if (auto attr = op->getAttrOfType<mlir::IntegerAttr>(OWN_RESULT_BORROWS_ATTR_NAME))
    {
        return static_cast<int>(attr.getInt());
    }

    return -1;
}

// An op whose result is the very block its operand holds: a class widened to a union or an
// optional or narrowed back, a union made from its payload or read back from it, an object seen
// through an interface (`{vtable, this}`, released through `this`). rc's retain or release of
// either is one of the block, so the pass looks through them. A cast that converts (a number
// printed into a string) or boxes (into `any`) makes a new block and is not one.
inline bool isView(mlir::Operation *op)
{
    if (mlir::isa_and_nonnull<mlir_ts::CreateUnionInstanceOp, mlir_ts::GetValueFromUnionOp,
                              mlir_ts::OptionalValueOp, mlir_ts::ValueOp, mlir_ts::NewInterfaceOp>(op))
    {
        return true;
    }

    auto castOp = mlir::dyn_cast_or_null<mlir_ts::CastOp>(op);
    if (!castOp)
    {
        return false;
    }

    auto keeps = [](mlir::Type type) {
        return mlir::isa<mlir_ts::ClassType, mlir_ts::UnionType, mlir_ts::OptionalType>(type);
    };
    // a bound function and a hybrid one are the same `{func, this, tag}`: a closure passed on as
    // a function value is still its box
    auto isClosure = [](mlir::Type type) {
        return mlir::isa<mlir_ts::BoundFunctionType, mlir_ts::HybridFunctionType>(type);
    };
    // a block `ts.New` made, seen as the object it is: a generator's state object
    auto isMadeObject = [&]() {
        return mlir::isa<mlir_ts::ValueRefType>(castOp.getIn().getType()) &&
               mlir::isa<mlir_ts::ObjectType>(castOp.getType());
    };
    return (keeps(castOp.getIn().getType()) && keeps(castOp.getType())) ||
           (isClosure(castOp.getIn().getType()) && isClosure(castOp.getType())) || isMadeObject();
}

// The block a value is a view of.
inline mlir::Value rootOf(mlir::Value value)
{
    while (auto *def = value.getDefiningOp())
    {
        if (!isView(def))
        {
            break;
        }

        value = def->getOperand(0);
    }

    return value;
}

// Every user of `root` and of the views of it, with the value it uses. A view itself is not a
// use.
inline void forEachUse(mlir::Value root, llvm::function_ref<void(mlir::Operation *, mlir::Value)> each)
{
    llvm::SmallVector<mlir::Value> values{root};
    while (!values.empty())
    {
        auto value = values.pop_back_val();
        for (auto &use : value.getUses())
        {
            auto *user = use.getOwner();
            if (isView(user) && use.getOperandNumber() == 0)
            {
                values.push_back(user->getResult(0));
                continue;
            }

            each(user, value);
        }
    }
}

inline bool isOwningVariable(mlir_ts::VariableOp varOp)
{
    return varOp->hasAttr(OWNED_LOCAL_ATTR_NAME) || varOp->hasAttr(OWNED_LOCAL_CONSUMED_ATTR_NAME);
}

// A string literal or a constant array cast to its value type. The result is either the
// immortal global itself, which a release skips, or a copy nobody else holds, which a
// release destroys - so, owned once, it has one owner either way. `null` and `undefined`
// widened to a nullable type (`c: C | null = null`), and an empty optional
// (`u: C | undefined = undefined`), hold no block at all.
inline bool isLiteral(mlir::Operation *def)
{
    if (mlir::isa<mlir_ts::OptionalUndefOp>(def))
    {
        return true;
    }

    auto castOp = mlir::dyn_cast<mlir_ts::CastOp>(def);
    if (!castOp)
    {
        return false;
    }

    auto *in = castOp.getIn().getDefiningOp();
    return in && mlir::isa<mlir_ts::ConstantOp, mlir_ts::NullOp, mlir_ts::UndefOp>(in);
}

// Made here and held by nobody else: an allocation, a literal cast to its value type, an
// operation rc marks as arriving with a reference, or a direct call of a function this module
// defines (every function returns its result retained, rc 9.24; the call's own mark does not
// survive the affine lowering), or another own module's `..new` (OWN_FRESH_RESULT_ATTR_NAME). A
// declared callee, an indirect call, a parameter, a load, a value merged from branches, or a result
// that borrows an argument is not fresh.
inline bool isFresh(mlir::Value value)
{
    auto *def = value.getDefiningOp();
    if (!def || resultBorrows(def) >= 0)
    {
        return false;
    }

    // a view is the block it shows: the object literal's interface cast is marked as arriving with
    // a reference, but `let raw = {...}; <I>raw` is `raw`'s block
    if (isView(def))
    {
        return isFresh(rootOf(value));
    }

    if (mlir::isa<mlir_ts::NewOp, mlir_ts::CreateArrayOp, mlir_ts::NewArrayOp, mlir_ts::StringConcatOp,
                  mlir_ts::StringResizeOp, mlir_ts::CharToStringOp>(def))
    {
        return true;
    }

    if (mlir::isa<mlir_ts::SymbolCallInternalOp, mlir_ts::InvokeOp>(def) && directCallee(def))
    {
        auto callee = mlir::SymbolTable::lookupNearestSymbolFrom<mlir_ts::FuncOp>(def, directCallee(def));
        return callee && !callee.isDeclaration();
    }

    if (def->hasAttr(OWN_FRESH_RESULT_ATTR_NAME))
    {
        return true;
    }

    if (mlir::isa<mlir_ts::CallOp, mlir_ts::CallIndirectOp, mlir_ts::CallInternalOp, mlir_ts::CallHybridInternalOp,
                  mlir_ts::InvokeOp, mlir_ts::InvokeHybridOp>(def))
    {
        return false;
    }

    return def->hasAttr(OWNED_RESULT_ATTR_NAME) || isLiteral(def);
}

inline bool isCall(mlir::Operation *op)
{
    return mlir::isa<mlir_ts::SymbolCallInternalOp, mlir_ts::CallOp, mlir_ts::CallIndirectOp,
                     mlir_ts::CallInternalOp, mlir_ts::CallHybridInternalOp, mlir_ts::InvokeOp,
                     mlir_ts::InvokeHybridOp>(op);
}

inline bool isPlace(mlir::Value ref)
{
    return mlir::isa_and_nonnull<mlir_ts::PropertyRefOp, mlir_ts::ElementRefOp>(ref.getDefiningOp());
}

// The values the branches into its block pass for `argument`, each seen through its views.
// False, adding nothing, when a way in is not a branch that passes one - the entry block's
// parameters, a region's arguments.
inline bool mergedInto(mlir::BlockArgument argument, llvm::SmallVectorImpl<mlir::Value> &values)
{
    auto *block = argument.getOwner();
    if (block->isEntryBlock())
    {
        return false;
    }

    llvm::SmallVector<mlir::Value> found;
    for (auto *predecessor : block->getPredecessors())
    {
        auto branchOp = mlir::dyn_cast<mlir::BranchOpInterface>(predecessor->getTerminator());
        if (!branchOp)
        {
            return false;
        }

        for (unsigned index = 0; index < branchOp->getNumSuccessors(); ++index)
        {
            if (branchOp->getSuccessor(index) != block)
            {
                continue;
            }

            auto passed = branchOp.getSuccessorOperands(index)[argument.getArgNumber()];
            if (!passed)
            {
                return false;
            }

            found.push_back(rootOf(passed));
        }
    }

    values.append(found.begin(), found.end());
    return true;
}

// The load of an owning local that `value` was read from, through its views (a class widened
// to a union with null, an upcast, a union made from it). None when the value is anything
// else - a parameter, a field, a boxing cast - which phase 1 does not move out of.
inline mlir_ts::LoadOp slotLoadOf(mlir::Value value)
{
    auto loadOp = rootOf(value).getDefiningOp<mlir_ts::LoadOp>();
    if (!loadOp)
    {
        return {};
    }

    auto varOp = loadOp.getReference().getDefiningOp<mlir_ts::VariableOp>();
    if (!varOp || !isOwningVariable(varOp) || varOp.getCaptured().value_or(false))
    {
        return {};
    }

    return loadOp;
}

// Does a value of this type own a block that a release would destroy?
inline bool ownsHeap(mlir::Value value)
{
    MLIRTypeHelper mth(value.getContext(), CompileOptions{});
    return mth.ownsHeapMemory(value.getLoc(), value.getType());
}

// Holds no block: a number, `null` or `undefined` widened, an empty optional, a string literal.
inline bool holdsNoBlock(mlir::Value value)
{
    auto *def = rootOf(value).getDefiningOp();
    return !ownsHeap(value) || (def && isLiteral(def));
}

// A captured variable's cell, as a closure body reaches it: read out of a field of its capture box
// (or of any other block - a RefType loaded from somewhere is a cell). Its variable lives in
// whatever function declared it, so an assignment through it may destroy a value that function, and
// the closures over the same cell, hold borrows of.
inline bool isLoadedCell(mlir::Value value)
{
    auto loadOp = value.getDefiningOp<mlir_ts::LoadOp>();
    return loadOp && mlir::isa<mlir_ts::RefType>(loadOp.getType());
}

// A captured variable's cell, in the function that declared it.
inline bool isCellVariable(mlir::Value value)
{
    auto varOp = value.getDefiningOp<mlir_ts::VariableOp>();
    return varOp && varOp.getCaptured().value_or(false) && !varOp->hasAttr(CAPTURE_BOX_ATTR_NAME);
}

// The capture box a closure was built over, if `value` is a capture box's own variable.
inline mlir_ts::CreateBoundFunctionOp closureOfBox(mlir::Value box)
{
    auto varOp = box.getDefiningOp<mlir_ts::VariableOp>();
    if (!varOp || !varOp->hasAttr(CAPTURE_BOX_ATTR_NAME))
    {
        return {};
    }

    for (auto *user : box.getUsers())
    {
        if (auto boundOp = mlir::dyn_cast<mlir_ts::CreateBoundFunctionOp>(user);
            boundOp && boundOp.getThisVal() == box && boundOp->hasAttr(OWNS_CAPTURE_ATTR_NAME))
        {
            return boundOp;
        }
    }

    return {};
}

// Is this the slot of parameter `argument` - a local declared from it that owns nothing, is not
// captured, and is never assigned, so every read of it is the argument?
inline bool isParameterSlot(mlir_ts::VariableOp varOp, int &index)
{
    auto init = varOp.getInitializer();
    auto argument = init ? mlir::dyn_cast<mlir::BlockArgument>(init) : mlir::BlockArgument();
    if (!argument || !argument.getOwner()->isEntryBlock() ||
        !mlir::isa<mlir_ts::FuncOp>(argument.getOwner()->getParentOp()) || isOwningVariable(varOp) ||
        varOp.getCaptured().value_or(false))
    {
        return false;
    }

    auto assigned = llvm::any_of(varOp.getResult().getUsers(), [&](mlir::Operation *user) {
        auto storeOp = mlir::dyn_cast<mlir_ts::StoreOp>(user);
        return storeOp && storeOp.getReference() == varOp.getResult();
    });
    if (assigned)
    {
        return false;
    }

    index = argument.getArgNumber();
    return true;
}

// The object a method call is made on: `ts.GetThis` of a method bound to it. Null otherwise.
inline mlir::Value boundThis(mlir::Value value)
{
    auto getThisOp = value.getDefiningOp<mlir_ts::GetThisOp>();
    auto *bound = getThisOp ? getThisOp.getOperand().getDefiningOp() : nullptr;
    if (auto refOp = mlir::dyn_cast_or_null<mlir_ts::ThisVirtualSymbolRefOp>(bound))
    {
        return refOp.getThisVal();
    }

    if (auto refOp = mlir::dyn_cast_or_null<mlir_ts::ThisSymbolRefOp>(bound))
    {
        return refOp.getThisVal();
    }

    return {};
}

// The parameter whose argument `value` is a borrow of, or -1 (spec 2.4, borrowed-from-argument).
// Seen through views, casts out of `!ts.opaque`, `ts.Unbox` (the payload an `any` box holds), the
// object a method is called on,
// reads out of fields and elements (the argument's block holds them), and results that borrow an
// argument, down to a read of a parameter's slot or the parameter itself. Merged values must all
// agree; one that holds no block (`null`) agrees with any. A value the function made, a local, a
// global: -1.
inline int borrowedParam(mlir::Value value, int depth = 0)
{
    if (depth > 32)
    {
        return -1;
    }

    value = rootOf(value);
    if (auto argument = mlir::dyn_cast<mlir::BlockArgument>(value))
    {
        if (argument.getOwner()->isEntryBlock())
        {
            return mlir::isa<mlir_ts::FuncOp>(argument.getOwner()->getParentOp()) ? argument.getArgNumber() : -1;
        }

        llvm::SmallVector<mlir::Value> merged;
        if (!mergedInto(argument, merged))
        {
            return -1;
        }

        auto found = -1;
        for (auto passed : merged)
        {
            if (holdsNoBlock(passed))
            {
                continue;
            }

            auto index = borrowedParam(passed, depth + 1);
            if (index < 0 || (found >= 0 && index != found))
            {
                return -1;
            }

            found = index;
        }

        return found;
    }

    if (auto object = boundThis(value))
    {
        return borrowedParam(object, depth + 1);
    }

    auto *def = value.getDefiningOp();
    // out of an opaque pointer, out of an interface (its `this`), or reading an interface value
    // back out of an `any` box: the same block
    if (auto castOp = mlir::dyn_cast_or_null<mlir_ts::CastOp>(def);
        castOp && (mlir::isa<mlir_ts::OpaqueType, mlir_ts::InterfaceType>(castOp.getIn().getType()) ||
                   (mlir::isa<mlir_ts::AnyType>(castOp.getIn().getType()) &&
                    mlir::isa<mlir_ts::InterfaceType>(castOp.getType()))))
    {
        return borrowedParam(castOp.getIn(), depth + 1);
    }

    if (auto extractOp = mlir::dyn_cast_or_null<mlir_ts::ExtractInterfaceThisOp>(def))
    {
        return borrowedParam(extractOp.getOperand(), depth + 1);
    }

    if (auto unboxOp = mlir::dyn_cast_or_null<mlir_ts::UnboxOp>(def))
    {
        return borrowedParam(unboxOp.getIn(), depth + 1);
    }

    if (auto propertyRefOp = mlir::dyn_cast_or_null<mlir_ts::PropertyRefOp>(def))
    {
        return borrowedParam(propertyRefOp.getObjectRef(), depth + 1);
    }

    if (auto elementRefOp = mlir::dyn_cast_or_null<mlir_ts::ElementRefOp>(def))
    {
        return borrowedParam(elementRefOp.getArray(), depth + 1);
    }

    if (auto loadOp = mlir::dyn_cast_or_null<mlir_ts::LoadOp>(def))
    {
        auto ref = loadOp.getReference();
        if (isPlace(ref))
        {
            return borrowedParam(ref, depth + 1);
        }

        auto index = -1;
        auto varOp = ref.getDefiningOp<mlir_ts::VariableOp>();
        return varOp && isParameterSlot(varOp, index) ? index : -1;
    }

    if (auto varOp = mlir::dyn_cast_or_null<mlir_ts::VariableOp>(def))
    {
        auto index = -1;
        return isParameterSlot(varOp, index) ? index : -1;
    }

    if (def && isCall(def) && resultBorrows(def) >= 0 &&
        static_cast<size_t>(resultBorrows(def)) < callArgs(def).size())
    {
        return borrowedParam(callArgs(def)[resultBorrows(def)], depth + 1);
    }

    return -1;
}

} // namespace own_facts

#endif // MLIR_TYPESCRIPT_OWNERSHIPFACTS_H
