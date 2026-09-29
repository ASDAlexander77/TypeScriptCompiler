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
// This is the whole question for a receiver after which nothing reads the value - a `return`, a
// `delete`. Every other receiver asks mayTakeOverReference.
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

} // namespace typescript

#endif // MLIR_TYPESCRIPT_COMMONGENLOGIC_MLIROWNEDREFERENCE_H_
