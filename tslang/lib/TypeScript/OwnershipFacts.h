#ifndef MLIR_TYPESCRIPT_OWNERSHIPFACTS_H
#define MLIR_TYPESCRIPT_OWNERSHIPFACTS_H

#include "TypeScript/TypeScriptOps.h"
#include "TypeScript/Defines.h"
#include "TypeScript/MLIRLogic/MLIRTypeHelper.h"

#include "mlir/Interfaces/ControlFlowInterfaces.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallPtrSet.h"
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
// On a function: the parameters it keeps (OWN_PARAMS_ATTR_NAME) that its body never takes into a
// field, an element, a global, a keeping call or an assigned owning local - one it returns on some
// paths only, one a callback family gives it to keep (spec 33). The inference releases each on the
// paths that do not move it, statically.
#define OWN_PARAMS_RELEASED_ATTR_NAME "__own_params_released"
// The arguments the result is a borrow of, as argument indices (the receiver is 0 for a method): the
// result may not outlive any of them. Alone, the caller owns nothing it gets back, and it names one
// argument. With OWN_RESULT_RECORD_ATTR_NAME it may name several (spec 32).
#define OWN_RESULT_BORROWS_ATTR_NAME "__own_result_borrows"
// Beside OWN_RESULT_BORROWS_ATTR_NAME: the result is a borrowing record (spec 32) - an object literal
// box the callee made, holding borrows of those arguments. The caller owns the box and releases it;
// the box's release (its interface's tag, OWN_RECORD_BORROWED_FIELDS_ATTR_NAME) frees only the
// fields the record owns.
#define OWN_RESULT_RECORD_ATTR_NAME "__own_result_record"
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
// The globals a call may give a new value, by name: a store to one, or its release, in a callee or
// in a call it makes. A call that may drop and whose callees are not known may assign any of them
// (OWN_ASSIGNS_ANY_GLOBAL_ATTR_NAME). A read of a global ends at such a call (checkGlobalRead).
#define OWN_ASSIGNS_GLOBALS_ATTR_NAME "__own_assigns_globals"
#define OWN_ASSIGNS_ANY_GLOBAL_ATTR_NAME "__own_assigns_any_global"
// An immortal global (spec 25): a class-typed global of this module, not exported, whose only
// write is its initializer, which gives it an object made there (not a call's result) or another
// immortal global's. Like a
// string literal, its block is marked HEAP_BLOCK_IMMORTAL, so every release skips it, and a read of
// it owns nothing and may go anywhere. Pinned by OwnershipSignaturePass on the `ts.Global`, on
// every `ts.AddressOf` of it (isImmortalGlobalRead), and on the `ts.Store` that initializes it,
// whose lowering writes the header word.
#define OWN_IMMORTAL_GLOBAL_ATTR_NAME "__own_immortal"

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

// Every argument a call's (or a function's) result borrows: one for a plain borrow, any number for a
// borrowing record (spec 32). Empty for none.
inline llvm::SmallVector<int> resultBorrowsAll(mlir::Operation *op)
{
    llvm::SmallVector<int> indices;
    if (auto attr = op->getAttrOfType<mlir::DenseI32ArrayAttr>(OWN_RESULT_BORROWS_ATTR_NAME))
    {
        indices.append(attr.asArrayRef().begin(), attr.asArrayRef().end());
    }

    return indices;
}

// A result that is a borrowing record (OWN_RESULT_RECORD_ATTR_NAME): a box the caller owns, bounded by
// the arguments it borrows.
inline bool isRecordResult(mlir::Operation *op)
{
    return op && op->hasAttr(OWN_RESULT_RECORD_ATTR_NAME) && op->hasAttr(OWN_RESULT_BORROWS_ATTR_NAME);
}

// The argument a call's result is a plain borrow of, or -1: none, or a borrowing record, which the
// caller owns (isRecordResult). Kept beside resultBorrowsAll for the plain borrow, which names one.
inline int resultBorrows(mlir::Operation *op)
{
    if (isRecordResult(op))
    {
        return -1;
    }

    auto indices = resultBorrowsAll(op);
    return indices.empty() ? -1 : indices.front();
}

// Is `narrow` a member of the union or optional `wide` as it is, so that a cast between them keeps
// its value? With `onlyNullable`, every other member is null or undefined.
inline bool holdsAsIs(mlir::Type wide, mlir::Type narrow, bool onlyNullable = false)
{
    llvm::SmallVector<mlir::Type> members;
    if (auto optionalType = mlir::dyn_cast<mlir_ts::OptionalType>(wide))
    {
        members.push_back(optionalType.getElementType());
    }
    else if (auto unionType = mlir::dyn_cast<mlir_ts::UnionType>(wide))
    {
        members.append(unionType.getTypes().begin(), unionType.getTypes().end());
    }
    else
    {
        return false;
    }

    return llvm::is_contained(members, narrow) &&
           (!onlyNullable || llvm::all_of(members, [&](mlir::Type member) {
                return member == narrow || mlir::isa<mlir_ts::NullType, mlir_ts::UndefinedType>(member);
            }));
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
    // a string or an array widened to a union that holds it as it is (`string | null`), or narrowed
    // back from one whose other members are only null or undefined. A union with other members can
    // convert on the way out (a number printed into a string), and an array of another element type
    // is a new array.
    auto isStringOrArray = [](mlir::Type type) { return mlir::isa<mlir_ts::StringType, mlir_ts::ArrayType>(type); };
    auto in = castOp.getIn().getType();
    auto out = castOp.getType();
    auto isNullableView = (isStringOrArray(in) && holdsAsIs(out, in, false)) ||
                          (isStringOrArray(out) && holdsAsIs(in, out, true));

    return (keeps(in) && keeps(out)) || (isClosure(in) && isClosure(out)) || isMadeObject() || isNullableView;
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

// The `ts.AddressOf` of the global a value is read out of, seen through its views, or null.
inline mlir_ts::AddressOfOp globalReadOf(mlir::Value value)
{
    auto loadOp = rootOf(value).getDefiningOp<mlir_ts::LoadOp>();
    return loadOp ? loadOp.getReference().getDefiningOp<mlir_ts::AddressOfOp>() : mlir_ts::AddressOfOp();
}

// A read of an immortal global (OWN_IMMORTAL_GLOBAL_ATTR_NAME), seen through its views: the
// global's own block, which no release frees. Its fields are not immortal: a read out of one is a
// place read like any other.
inline bool isImmortalGlobalRead(mlir::Value value)
{
    auto addressOfOp = globalReadOf(value);
    return addressOfOp && addressOfOp->hasAttr(OWN_IMMORTAL_GLOBAL_ATTR_NAME);
}

// Does a global's initializer region yield only the default (`ts.Default`, `ts.Null`,
// `ts.Undef`)? A class static's does: the one store in its static constructor gives it its value
// (spec 25). OwnershipSignaturePass marks such a global immortal only with that store, and the
// lowering then makes no `__cctor` of the region, which would put the default back after it.
inline bool regionYieldsDefault(mlir_ts::GlobalOp globalOp)
{
    auto *block = globalOp.getInitializerBlock();
    auto *terminator = block ? block->getTerminator() : nullptr;
    auto *valueOp = terminator && terminator->getNumOperands() == 1 ? terminator->getOperand(0).getDefiningOp() : nullptr;
    return mlir::isa_and_nonnull<mlir_ts::DefaultOp, mlir_ts::NullOp, mlir_ts::UndefOp>(valueOp);
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

    // a borrowing record (spec 32): the box is the caller's, whatever the call goes through; what it
    // borrows bounds it (OwnershipInferencePass's checkRecordResult)
    if (isRecordResult(def))
    {
        return true;
    }

    // a view is the block it shows: the object literal's interface cast is marked as arriving with
    // a reference, but `let raw = {...}; <I>raw` is `raw`'s block
    if (isView(def))
    {
        return isFresh(rootOf(value));
    }

    if (mlir::isa<mlir_ts::NewOp, mlir_ts::CreateArrayOp, mlir_ts::NewArrayOp, mlir_ts::StringConcatOp,
                  mlir_ts::StringResizeOp, mlir_ts::StringCopyOp, mlir_ts::CharToStringOp>(def))
    {
        return true;
    }

    // what the signature pass knows of a call: another own module's `..new`, a `@borrows` callee's
    // state, an iterator's `next()` - a declared callee included
    if (def->hasAttr(OWN_FRESH_RESULT_ATTR_NAME))
    {
        return true;
    }

    if (mlir::isa<mlir_ts::SymbolCallInternalOp, mlir_ts::InvokeOp>(def) && directCallee(def))
    {
        auto callee = mlir::SymbolTable::lookupNearestSymbolFrom<mlir_ts::FuncOp>(def, directCallee(def));
        return callee && !callee.isDeclaration();
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

// A field of an object seen through an interface (`i.f`): the object's own field, reached by the
// offset the interface's vtable holds. A method of the interface is a bound function, not a place.
inline bool isInterfaceField(mlir::Value ref)
{
    return ref.getDefiningOp<mlir_ts::InterfaceSymbolRefOp>() && mlir::isa<mlir_ts::RefType>(ref.getType());
}

// A field, an element, or `s.value` - the payload of a handle (spec 23.1) - and a field reached
// through an interface (spec 28).
inline bool isPlace(mlir::Value ref)
{
    return mlir::isa_and_nonnull<mlir_ts::PropertyRefOp, mlir_ts::ElementRefOp, mlir_ts::SharedValueRefOp>(
               ref.getDefiningOp()) ||
           isInterfaceField(ref);
}

// A handle (spec 23.2): counted in every model that tracks ownership, and left alone by own.
inline bool isHandle(mlir::Value value)
{
    return value && MLIRTypeHelper::isSharedHandleType(value.getType());
}

// A retain or release of a handle, or of a slot holding one: rc's, which own leaves in place.
inline bool touchesHandle(mlir::Operation *op)
{
    if (mlir::isa<mlir_ts::RetainOp, mlir_ts::ReleaseOp>(op))
    {
        return isHandle(op->getOperand(0));
    }

    if (mlir::isa<mlir_ts::RetainSlotOp, mlir_ts::ReleaseSlotOp>(op))
    {
        auto slotType = mlir::dyn_cast<mlir_ts::RefType>(op->getOperand(0).getType());
        return slotType && MLIRTypeHelper::isSharedHandleType(slotType.getElementType());
    }

    return false;
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

// Is this reference a place reached through a handle's payload (`s.value.items`), at any depth?
inline bool reachesSharedValue(mlir::Value ref)
{
    llvm::SmallVector<mlir::Value> work{ref};
    llvm::SmallPtrSet<mlir::Value, 8> seen;
    while (!work.empty())
    {
        auto value = rootOf(work.pop_back_val());
        if (!seen.insert(value).second)
        {
            continue;
        }

        if (value.getDefiningOp<mlir_ts::SharedValueRefOp>())
        {
            return true;
        }

        if (auto propertyRefOp = value.getDefiningOp<mlir_ts::PropertyRefOp>())
        {
            work.push_back(propertyRefOp.getObjectRef());
        }
        else if (auto elementRefOp = value.getDefiningOp<mlir_ts::ElementRefOp>())
        {
            work.push_back(elementRefOp.getArray());
        }
        else if (auto interfaceRefOp = value.getDefiningOp<mlir_ts::InterfaceSymbolRefOp>(); interfaceRefOp && isInterfaceField(value))
        {
            work.push_back(interfaceRefOp.getInterfaceVal());
        }
        else if (auto loadOp = value.getDefiningOp<mlir_ts::LoadOp>(); loadOp && isPlace(loadOp.getReference()))
        {
            work.push_back(loadOp.getReference());
        }
        else if (auto argument = mlir::dyn_cast<mlir::BlockArgument>(value))
        {
            llvm::SmallVector<mlir::Value> merged;
            if (mergedInto(argument, merged))
            {
                work.append(merged.begin(), merged.end());
            }
        }
    }

    return false;
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
// Also a view of a value whose type owns none: a union's number payload made into another union
// (`GetValueFromUnion`, then `CreateUnionInstance`) is a number all the way. A `value_ref` is not
// such a type: it owns nothing by type, but one `ts.New` made is the object its view shows.
inline bool holdsNoBlock(mlir::Value value)
{
    if (!ownsHeap(value))
    {
        return true;
    }

    for (auto *def = value.getDefiningOp(); def; def = def->getOperand(0).getDefiningOp())
    {
        if (isLiteral(def))
        {
            return true;
        }

        if (!isView(def))
        {
            return false;
        }

        auto in = def->getOperand(0);
        if (!ownsHeap(in) && !mlir::isa<mlir_ts::ValueRefType>(in.getType()))
        {
            return true;
        }
    }

    return false;
}

// An argument that holds no block (holdsNoBlock) - but not an opaque pointer, by which a call through
// an interface or a bound method is given its receiver (`ts.GetThis`), whose type owns nothing.
inline bool argumentHoldsNoBlock(mlir::Value arg)
{
    return !mlir::isa<mlir_ts::OpaqueType>(arg.getType()) && holdsNoBlock(arg);
}

// A value -mm=own copies rather than give a second owner (spec 22.1): a string, a string literal,
// a union of strings and null that needs no tag (one pointer), or an optional of one of those
// (`{ptr, i1}`).
inline bool isCopyableString(mlir::Type type)
{
    auto isText = [](mlir::Type member) {
        if (auto literalType = mlir::dyn_cast<mlir_ts::LiteralType>(member))
        {
            return mlir::isa<mlir_ts::StringType>(literalType.getElementType());
        }

        return mlir::isa<mlir_ts::StringType>(member);
    };

    if (auto optionalType = mlir::dyn_cast<mlir_ts::OptionalType>(type))
    {
        return !mlir::isa<mlir_ts::OptionalType>(optionalType.getElementType()) &&
               isCopyableString(optionalType.getElementType());
    }

    if (auto unionType = mlir::dyn_cast<mlir_ts::UnionType>(type))
    {
        if (!llvm::any_of(unionType.getTypes(), isText) ||
            !llvm::all_of(unionType.getTypes(), [&](mlir::Type member) {
                return isText(member) || mlir::isa<mlir_ts::NullType>(member);
            }))
        {
            return false;
        }

        // only a union that lowers to one pointer: a tagged union is a struct
        MLIRTypeHelper mth(type.getContext(), CompileOptions{});
        return !mth.isUnionTypeNeedsTag(mlir::UnknownLoc::get(type.getContext()), unionType);
    }

    return isText(type);
}

// Some view of `value` is a `ts.NewInterface`, which boxes what it shows into a block.
inline bool boxedByView(mlir::Value value)
{
    llvm::SmallVector<mlir::Value> values{value};
    while (!values.empty())
    {
        for (auto &use : values.pop_back_val().getUses())
        {
            auto *user = use.getOwner();
            if (isView(user) && use.getOperandNumber() == 0)
            {
                if (mlir::isa<mlir_ts::NewInterfaceOp>(user))
                {
                    return true;
                }

                values.push_back(user->getResult(0));
            }
        }
    }

    return false;
}

// Data nothing owns: what holds no block, a `ts.Constant` (its strings are immortal) or a field of
// one, and a tuple built of such (`[1, "a"]`, an object literal's fields read out of its constant).
// Retaining it takes nothing anyone gives back.
inline bool isConstantData(mlir::Value value)
{
    // a view that boxes it makes a block of its own: a folded object literal given as an interface
    // is a new block each time, which its own release destroys. And a `ts.New` is a block, though
    // its `value_ref` type owns nothing: it is the object its view shows.
    if (value.getDefiningOp<mlir_ts::NewOp>() || boxedByView(value))
    {
        return false;
    }

    if (holdsNoBlock(value))
    {
        return true;
    }

    auto *def = value.getDefiningOp();
    if (!def)
    {
        return false;
    }

    if (mlir::isa<mlir_ts::ConstantOp, mlir_ts::SymbolRefOp>(def))
    {
        return true;
    }

    // a function, which a `ts.SymbolRef` names, seen as a function value: it has no capture box
    if (auto castOp = mlir::dyn_cast<mlir_ts::CastOp>(def); castOp && castOp.getIn().getDefiningOp<mlir_ts::SymbolRefOp>())
    {
        return true;
    }

    // a field of such data: of a `ts.Constant`, or of a tuple of literals re-made as a named one
    if (auto extractOp = mlir::dyn_cast<mlir_ts::ExtractPropertyOp>(def))
    {
        return isConstantData(extractOp.getObject());
    }

    if (mlir::isa<mlir_ts::CreateTupleOp>(def))
    {
        return llvm::all_of(def->getOperands(), [](mlir::Value operand) { return isConstantData(operand); });
    }

    return false;
}

// A non-owning local a `ts.Constant` initializes, used only through its fields and its reads: the
// local MLIRGen builds an object literal in.
inline bool isConstantRecordLocal(mlir_ts::VariableOp varOp)
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

// The block an object literal is boxed into when it is given as an object or an interface: a
// `ts.New` of a `value_ref`, filled by one store of the record's read (spec 27). Its views (the
// object, the interface) are it. Null for anything else.
inline mlir_ts::NewOp recordBoxOf(mlir::Value value)
{
    auto newOp = rootOf(value).getDefiningOp<mlir_ts::NewOp>();
    if (!newOp || !mlir::isa<mlir_ts::ValueRefType>(newOp.getType()))
    {
        return {};
    }

    mlir_ts::StoreOp fill;
    for (auto &use : newOp->getUses())
    {
        auto storeOp = mlir::dyn_cast<mlir_ts::StoreOp>(use.getOwner());
        if (!storeOp || storeOp.getReference() != use.get())
        {
            continue;
        }

        if (fill)
        {
            return {};
        }

        fill = storeOp;
    }

    auto read = fill ? fill.getValue().getDefiningOp<mlir_ts::LoadOp>() : mlir_ts::LoadOp();
    auto varOp = read ? read.getReference().getDefiningOp<mlir_ts::VariableOp>() : mlir_ts::VariableOp();
    return varOp && isConstantRecordLocal(varOp) ? newOp : mlir_ts::NewOp();
}

// The local an object literal is built in, from its read, or from the box it is copied into (or a
// view of that box). Null for anything else.
inline mlir_ts::VariableOp recordLocalOf(mlir::Value value)
{
    if (auto boxOp = recordBoxOf(value))
    {
        for (auto *user : boxOp->getUsers())
        {
            auto storeOp = mlir::dyn_cast<mlir_ts::StoreOp>(user);
            if (storeOp && storeOp.getReference() == boxOp.getResult())
            {
                value = storeOp.getValue();
            }
        }
    }

    auto read = value.getDefiningOp<mlir_ts::LoadOp>();
    auto varOp = read ? read.getReference().getDefiningOp<mlir_ts::VariableOp>() : mlir_ts::VariableOp();
    return varOp && isConstantRecordLocal(varOp) ? varOp : mlir_ts::VariableOp();
}

// An object literal that owns what it holds (spec 27): its read, the box it is copied into, or a view
// of that box (the interface it is given as), where every field that owns a block holds what the
// constant put there (null, a number, an immortal literal), nothing at all, or a fresh value stored
// into it - a new object, a call's result, a literal, a `const` local folded into the value it holds.
// Each such value moves into the record: its own check, against the record's retain, decides that
// nothing uses it after (OwnershipInferencePass's recordRetainOf). A read out of a place (`this.pos`)
// or a parameter is a borrow, and such a record is not this one. Neither is a `let` read into a field:
// its slot keeps its own release, which nothing here cancels.
inline bool isOwningRecord(mlir::Value value)
{
    auto varOp = recordLocalOf(value);
    if (!varOp)
    {
        return false;
    }

    MLIRTypeHelper mth(value.getContext(), CompileOptions{});
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
        if (refType && !mth.ownsHeapMemory(field.getLoc(), refType.getElementType()))
        {
            continue;
        }

        for (auto *fieldUser : field.getUsers())
        {
            if (mlir::isa<mlir_ts::LoadOp>(fieldUser))
            {
                continue;
            }

            auto storeOp = mlir::dyn_cast<mlir_ts::StoreOp>(fieldUser);
            if (!storeOp || storeOp.getReference() != field ||
                !(isFresh(rootOf(storeOp.getValue())) || holdsNoBlock(storeOp.getValue())))
            {
                return false;
            }
        }
    }

    return true;
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

// The object a method call is made on: `ts.GetThis` of a method bound to it - by its class, or by an
// interface value whose method it is (`i.m`: the object the interface value is a view of, spec 31).
// Null otherwise.
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

    if (auto refOp = mlir::dyn_cast_or_null<mlir_ts::InterfaceSymbolRefOp>(bound); refOp && !isInterfaceField(refOp.getResult()))
    {
        return refOp.getInterfaceVal();
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

    if (auto interfaceRefOp = mlir::dyn_cast_or_null<mlir_ts::InterfaceSymbolRefOp>(def); interfaceRefOp && isInterfaceField(value))
    {
        return borrowedParam(interfaceRefOp.getInterfaceVal(), depth + 1);
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

    // a plain borrow of its arguments (a declared `@lends` may name several): all of one parameter
    if (def && isCall(def) && resultBorrows(def) >= 0)
    {
        auto args = callArgs(def);
        auto found = -1;
        for (auto index : resultBorrowsAll(def))
        {
            auto param = static_cast<size_t>(index) < args.size() ? borrowedParam(args[index], depth + 1) : -1;
            if (param < 0 || (found >= 0 && param != found))
            {
                return -1;
            }

            found = param;
        }

        return found;
    }

    return -1;
}

// ---- Borrowing records (spec 32) ----
//
// An object literal boxed as an interface whose fields hold values it does not own - `this`, a
// parameter, a field read, an owning `let`'s value, a captured variable's - beside fresh values it
// owns. OwnershipSignaturePass classifies each box where it is made and marks it (and every interface
// made of it) with the fields it borrows (OWN_RECORD_BORROWED_FIELDS_ATTR_NAME): the box's release,
// through the interface's tag, frees the fields it owns and leaves the others to their owners.
// OwnershipInferencePass then bounds every use of the record by what each borrowed field is read from.

// Where a field of a record gets a value that holds a block, for the record's release and its bound.
enum class RecordFieldSource
{
    NoBlock,  // a number, null: nothing to own
    Fresh,    // made here (moves into the record, which owns it)
    Immortal, // an immortal global or string literal: nobody frees it
    Param,    // a parameter or `this`, seen through views: the caller's for the whole call
    Place,    // a read out of a field or an element, or a call's result that borrows an argument
    Slot,     // the value of an owning local, which keeps it
    Cell,     // a captured variable's value, read through its cell
    Unknown
};

inline RecordFieldSource recordFieldSource(mlir::Value value)
{
    if (holdsNoBlock(value))
    {
        return RecordFieldSource::NoBlock;
    }

    if (isImmortalGlobalRead(value))
    {
        return RecordFieldSource::Immortal;
    }

    auto root = rootOf(value);
    if (auto castOp = root.getDefiningOp<mlir_ts::CastOp>();
        castOp && mlir::isa<mlir_ts::StringType>(castOp.getType()) && castOp.getIn().getDefiningOp<mlir_ts::ConstantOp>())
    {
        return RecordFieldSource::Immortal;
    }

    if (isFresh(root))
    {
        return RecordFieldSource::Fresh;
    }

    if (auto loadOp = root.getDefiningOp<mlir_ts::LoadOp>())
    {
        if (isPlace(loadOp.getReference()))
        {
            return RecordFieldSource::Place;
        }

        if (isLoadedCell(loadOp.getReference()))
        {
            return RecordFieldSource::Cell;
        }
    }

    if (auto *def = root.getDefiningOp(); def && isCall(def) && resultBorrows(def) >= 0)
    {
        return RecordFieldSource::Place;
    }

    if (borrowedParam(root) >= 0)
    {
        return RecordFieldSource::Param;
    }

    if (slotLoadOf(value))
    {
        return RecordFieldSource::Slot;
    }

    return RecordFieldSource::Unknown;
}

// The box of the object literal built in `varOp`: its one read, copied into a block `ts.New` made
// (recordBoxOf). Null for anything else.
inline mlir_ts::NewOp recordBoxOfLocal(mlir_ts::VariableOp varOp)
{
    if (!varOp || !isConstantRecordLocal(varOp))
    {
        return {};
    }

    for (auto *user : varOp->getUsers())
    {
        auto loadOp = mlir::dyn_cast<mlir_ts::LoadOp>(user);
        if (!loadOp)
        {
            continue;
        }

        for (auto *readUser : loadOp->getUsers())
        {
            auto storeOp = mlir::dyn_cast<mlir_ts::StoreOp>(readUser);
            if (storeOp && storeOp.getValue() == loadOp.getResult())
            {
                if (auto boxOp = recordBoxOf(storeOp.getReference()); boxOp && boxOp.getResult() == storeOp.getReference())
                {
                    return boxOp;
                }
            }
        }
    }

    return {};
}

// The fields a borrowing record's box (or an interface made of it) leaves to their owners; empty for
// any other value.
inline llvm::ArrayRef<int32_t> recordBorrowedFields(mlir::Operation *op)
{
    if (auto attr = op ? op->getAttrOfType<mlir::DenseI32ArrayAttr>(OWN_RECORD_BORROWED_FIELDS_ATTR_NAME) : nullptr)
    {
        return attr.asArrayRef();
    }

    return {};
}

// A store into a field of a borrowing record that the record borrows: it takes nothing.
inline bool isBorrowedRecordFieldStore(mlir::Operation *op)
{
    auto storeOp = mlir::dyn_cast<mlir_ts::StoreOp>(op);
    auto propertyRefOp = storeOp ? storeOp.getReference().getDefiningOp<mlir_ts::PropertyRefOp>() : mlir_ts::PropertyRefOp();
    auto varOp = propertyRefOp ? propertyRefOp.getObjectRef().getDefiningOp<mlir_ts::VariableOp>() : mlir_ts::VariableOp();
    auto boxOp = varOp ? recordBoxOfLocal(varOp) : mlir_ts::NewOp();
    return boxOp && llvm::is_contained(recordBorrowedFields(boxOp), static_cast<int32_t>(propertyRefOp.getPosition()));
}

// The tuple an object, a reference or a tuple value of this type stores its fields in; null for
// any other type. A method sees its object as `!ts.object<!ts.object_storage<@name>>`, whose
// fields are the tuple's.
inline mlir_ts::TupleType storageTupleOf(mlir::MLIRContext *context, mlir::Type type)
{
    if (auto objectType = mlir::dyn_cast<mlir_ts::ObjectType>(type))
    {
        type = objectType.getStorageType();
        if (auto storageType = mlir::dyn_cast<mlir_ts::ObjectStorageType>(type))
        {
            return mlir_ts::TupleType::get(context, storageType.getFields());
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

// (tuple type, field) -> the one function an object of that type holds in that field: every
// `ts.Constant` of the type's shape puts it there, nothing stores into the field, and nothing but
// a constant or a read makes a value of the type. A generator's state object holds its `next` so.
// Used by OwnershipSignaturePass (own) and OwnedReturnConsumptionPass (rc).
class FieldFunctions
{
    llvm::DenseMap<std::pair<mlir::Type, int64_t>, mlir::StringAttr> fieldFunctions;
    mlir::MLIRContext *context = nullptr;

  public:
    void collect(mlir::ModuleOp module)
    {
        context = module->getContext();
        fieldFunctions.clear();

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
                auto tupleType = storageTupleOf(context, propertyRefOp->getOperand(0).getType());
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
    mlir::StringAttr of(mlir::Operation *def) const
    {
        auto loadOp = mlir::dyn_cast_or_null<mlir_ts::LoadOp>(def);
        auto propertyRefOp = loadOp ? loadOp.getReference().getDefiningOp<mlir_ts::PropertyRefOp>() : mlir_ts::PropertyRefOp();
        auto tupleType = propertyRefOp ? storageTupleOf(context, propertyRefOp->getOperand(0).getType()) : mlir_ts::TupleType();
        if (!tupleType)
        {
            return {};
        }

        return fieldFunctions.lookup({tupleType, propertyRefOp.getPosition()});
    }
};

} // namespace own_facts

#endif // MLIR_TYPESCRIPT_OWNERSHIPFACTS_H
