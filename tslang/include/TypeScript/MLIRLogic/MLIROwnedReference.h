#ifndef MLIR_TYPESCRIPT_COMMONGENLOGIC_MLIROWNEDREFERENCE_H_
#define MLIR_TYPESCRIPT_COMMONGENLOGIC_MLIROWNEDREFERENCE_H_

#include "TypeScript/Defines.h"

#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"

namespace typescript
{

// Does this value still carry a reference nobody has taken over?
//
// The producer has to carry one at all (OWNED_RESULT_ATTR_NAME), and no receiver can have taken
// it already (OWNED_RESULT_CONSUMED_ATTR_NAME). There is only ever one to take, and one value can
// reach two receivers: `let b = h.c = new C()` offers the same `new` to the field and then to the
// local.
//
// A receiver that stores the value asks mayTakeOverReference, and one after which nothing reads
// it - a `return`, a `delete` - asks mayTakeOverReferenceAtLastUse.
inline bool carriesUnclaimedReference(mlir::Value value)
{
    auto *definingOp = value ? value.getDefiningOp() : nullptr;
    return definingOp && definingOp->hasAttr(OWNED_RESULT_ATTR_NAME) &&
           !definingOp->hasAttr(OWNED_RESULT_CONSUMED_ATTR_NAME);
}

// May a receiver that stores this value - a local, a field or element, an array, a literal - take
// its reference over instead of retaining?
//
// Not when the value is a folded `const` (OWNED_RESULT_NAMED_ATTR_NAME). Every mention of the
// name is this very value, so the receiver cannot tell the last mention from the first: one in an
// inner scope releases the reference while the name still reads it, and one in a loop takes the
// same reference on every iteration.
//
// Answering no costs a retain and a release. Answering yes wrongly frees live memory.
inline bool mayTakeOverReference(mlir::Value value)
{
    return carriesUnclaimedReference(value) && !value.getDefiningOp()->hasAttr(OWNED_RESULT_NAMED_ATTR_NAME);
}

// May a `return` or a `delete` take the reference over? Nothing reads the value after either.
//
// A folded `const` may be taken over even when already marked consumed: no storing receiver
// takes a named value, so what consumed it can only be another `return` or `delete`, on a path
// that excludes this one - `if (x) return c; return c;`. Retaining here instead would hand the
// caller two references on the second path.
inline bool mayTakeOverReferenceAtLastUse(mlir::Value value)
{
    auto *definingOp = value ? value.getDefiningOp() : nullptr;
    return definingOp && definingOp->hasAttr(OWNED_RESULT_ATTR_NAME) &&
           (definingOp->hasAttr(OWNED_RESULT_NAMED_ATTR_NAME) ||
            !definingOp->hasAttr(OWNED_RESULT_CONSUMED_ATTR_NAME));
}

} // namespace typescript

#endif // MLIR_TYPESCRIPT_COMMONGENLOGIC_MLIROWNEDREFERENCE_H_
