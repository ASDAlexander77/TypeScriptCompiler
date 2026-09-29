#ifndef MLIR_TYPESCRIPT_OWNERSHIPFACTS_H
#define MLIR_TYPESCRIPT_OWNERSHIPFACTS_H

#include "TypeScript/TypeScriptOps.h"
#include "TypeScript/Defines.h"

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

// The arguments of a call, without the callee value of an indirect one.
inline mlir::OperandRange callArgs(mlir::Operation *op)
{
    if (mlir::isa<mlir_ts::CallInternalOp, mlir_ts::CallIndirectOp, mlir_ts::CallHybridInternalOp>(op))
    {
        return op->getOperands().drop_front();
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
// optional or narrowed back, a union made from its payload or read back from it. rc's retain
// or release of either is one of the block, so the pass looks through them. A cast that
// converts (a number printed into a string) or boxes (into `any`) makes a new block and is
// not one.
inline bool isView(mlir::Operation *op)
{
    if (mlir::isa_and_nonnull<mlir_ts::CreateUnionInstanceOp, mlir_ts::GetValueFromUnionOp,
                              mlir_ts::OptionalValueOp, mlir_ts::ValueOp>(op))
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
    return keeps(castOp.getIn().getType()) && keeps(castOp.getType());
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
// survive the affine lowering). A declared callee, an indirect call, a parameter, a load or a
// value merged from branches is not fresh.
inline bool isFresh(mlir::Value value)
{
    auto *def = value.getDefiningOp();
    if (!def)
    {
        return false;
    }

    if (mlir::isa<mlir_ts::NewOp, mlir_ts::CreateArrayOp, mlir_ts::NewArrayOp, mlir_ts::StringConcatOp,
                  mlir_ts::CharToStringOp>(def))
    {
        return true;
    }

    if (auto callOp = mlir::dyn_cast<mlir_ts::SymbolCallInternalOp>(def))
    {
        auto callee = mlir::SymbolTable::lookupNearestSymbolFrom<mlir_ts::FuncOp>(def, callOp.getCalleeAttr());
        return callee && !callee.isDeclaration();
    }

    if (mlir::isa<mlir_ts::CallOp, mlir_ts::CallIndirectOp, mlir_ts::CallInternalOp, mlir_ts::CallHybridInternalOp>(def))
    {
        return false;
    }

    return def->hasAttr(OWNED_RESULT_ATTR_NAME) || isLiteral(def);
}

inline bool isCall(mlir::Operation *op)
{
    return mlir::isa<mlir_ts::SymbolCallInternalOp, mlir_ts::CallOp, mlir_ts::CallIndirectOp,
                     mlir_ts::CallInternalOp, mlir_ts::CallHybridInternalOp>(op);
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

} // namespace own_facts

#endif // MLIR_TYPESCRIPT_OWNERSHIPFACTS_H
