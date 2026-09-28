#ifndef MLIR_TYPESCRIPT_OWNERSHIPOPS_H
#define MLIR_TYPESCRIPT_OWNERSHIPOPS_H

#include "TypeScript/TypeScriptOps.h"

namespace mlir
{
namespace typescript
{

// Is this release the giving-up half of an overwrite, rather than an unmatched release?
//
// An assignment into owning storage hands the count over: `ts.Retain` on the value coming
// in, `ts.ReleaseSlot` on what the slot still holds, then the store. The two halves are not
// expressed on the same thing - the retain names a value, the release names a slot - so the
// slot never appears in a `ts.RetainSlot` and the plain "released but never retained" test
// reports every field store in the suite. Recognising the store that follows is what tells
// the two apart: a release about to be overwritten is a hand-over, and the reference it
// gives up was taken by whoever stored the old value.
//
// Shared by the ownership verifier and, under -mm=own, ownership inference.
inline bool isHandOver(ReleaseSlotOp releaseOp)
{
    auto slot = releaseOp.getSlot();
    for (auto it = std::next(releaseOp->getIterator()); it != releaseOp->getBlock()->end(); ++it)
    {
        if (auto storeOp = mlir::dyn_cast<StoreOp>(*it))
        {
            if (storeOp.getReference() == slot)
            {
                return true;
            }
        }

        // another release of the same slot first means this one was not the overwrite's
        if (auto otherRelease = mlir::dyn_cast<ReleaseSlotOp>(*it))
        {
            if (otherRelease.getSlot() == slot)
            {
                return false;
            }
        }
    }

    return false;
}

} // namespace typescript
} // namespace mlir

#endif // MLIR_TYPESCRIPT_OWNERSHIPOPS_H
