#include "mlir/Pass/Pass.h"

#include "TypeScript/TypeScriptDialect.h"
#include "TypeScript/TypeScriptOps.h"
#include "TypeScript/TypeScriptFunctionPass.h"
#include "TypeScript/Passes.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "pass"

namespace mlir_ts = mlir::typescript;

namespace
{

// Checks the invariant ownership insertion is built on: a slot that takes a reference gives it
// back on every path out of the function, unwind paths included.
//
// This runs at the affine level, after TryOpLowering has turned scopes into blocks, because
// that is the first point where the unwind paths are ordinary CFG edges and can be walked like
// any other. It runs in every memory model, not just `-mm=rc`: ts.RetainSlot and ts.ReleaseSlot
// survive to here regardless and are only erased on the way to LLVM, so a collected build
// checks the same invariant a counted one does. That matters - most of the suite, and most of
// CI, is collected.
//
// It deliberately checks the direction that leaks rather than the direction that frees live
// memory. Step 5a's insertion is balanced by construction (retain at the declaration, release
// at every scope exit), so an unmatched release cannot currently be generated; what an
// extension to fields, elements, arguments or returns will get wrong first is a path out that
// nobody released on. The cheap structural half of the other direction is here too: a release
// naming a slot that is never retained.
class OwnershipVerifierPass : public mlir::PassWrapper<OwnershipVerifierPass, TypeScriptFunctionPass>
{
  public:
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(OwnershipVerifierPass)

    void runOnFunction() override
    {
        auto f = getFunction();

        llvm::SmallVector<mlir_ts::RetainSlotOp> retains;
        llvm::DenseSet<mlir::Value> retainedSlots;
        llvm::SmallVector<mlir_ts::ReleaseSlotOp> releases;
        f.walk([&](mlir::Operation *op) {
            if (auto retainOp = mlir::dyn_cast<mlir_ts::RetainSlotOp>(op))
            {
                retains.push_back(retainOp);
                retainedSlots.insert(retainOp.getSlot());
            }
            else if (auto releaseOp = mlir::dyn_cast<mlir_ts::ReleaseSlotOp>(op))
            {
                releases.push_back(releaseOp);
            }
        });

        if (retains.empty() && releases.empty())
        {
            return;
        }

        for (auto releaseOp : releases)
        {
            if (!retainedSlots.contains(releaseOp.getSlot()))
            {
                releaseOp.emitError("ownership: this slot is released but never retained");
                signalPassFailure();
            }
        }

        for (auto retainOp : retains)
        {
            verifyReleasedOnEveryPath(f, retainOp);
        }
    }

  private:
    // Whether this block gives the slot back - directly, or inside a region of one of its own
    // operations. Nested regions count as releasing rather than as opaque: reporting a leak
    // that the IR does pay, somewhere this walk does not follow, would be the one kind of
    // failure that makes a verifier get switched off.
    static bool blockReleases(mlir::Block *block, mlir::Value slot)
    {
        auto found = false;
        for (auto &op : *block)
        {
            op.walk([&](mlir_ts::ReleaseSlotOp releaseOp) {
                if (releaseOp.getSlot() == slot)
                {
                    found = true;
                }
            });

            if (found)
            {
                return true;
            }
        }

        return false;
    }

    // A terminator that leaves the function: ts.ReturnInternal, and the abrupt exits that carry
    // no successor of their own, such as a throw with nothing in this function to catch it.
    static bool blockExitsFunction(mlir::Block *block)
    {
        auto *terminator = block->getTerminator();
        return terminator != nullptr && terminator->getNumSuccessors() == 0;
    }

    void verifyReleasedOnEveryPath(mlir_ts::FuncOp f, mlir_ts::RetainSlotOp retainOp)
    {
        auto slot = retainOp.getSlot();
        auto *retainBlock = retainOp->getBlock();
        auto *region = retainBlock->getParent();
        if (region == nullptr)
        {
            return;
        }

        // releasedFromStart[B]: every path from the start of B to a function exit passes a
        // release of this slot. A backward must-analysis, so it starts optimistic and is driven
        // down to a fixed point - which leaves a loop with no exit at all reading as satisfied,
        // correctly: it has no path to an exit to leak on.
        llvm::DenseMap<mlir::Block *, bool> releasedFromStart;
        llvm::DenseMap<mlir::Block *, bool> hasRelease;
        for (auto &block : *region)
        {
            hasRelease[&block] = blockReleases(&block, slot);
            releasedFromStart[&block] = true;
        }

        auto changed = true;
        while (changed)
        {
            changed = false;
            for (auto &block : *region)
            {
                auto released = true;
                if (hasRelease[&block])
                {
                    released = true;
                }
                else if (blockExitsFunction(&block))
                {
                    released = false;
                }
                else
                {
                    for (auto *successor : block.getSuccessors())
                    {
                        auto it = releasedFromStart.find(successor);
                        if (it != releasedFromStart.end() && !it->second)
                        {
                            released = false;
                            break;
                        }
                    }
                }

                if (releasedFromStart[&block] != released)
                {
                    releasedFromStart[&block] = released;
                    changed = true;
                }
            }
        }

        // The retain's own block is special: only what follows the retain in it counts, since a
        // release ahead of the retain belongs to some earlier trip round a loop.
        auto releasedAfterRetain = false;
        for (auto it = std::next(retainOp->getIterator()); it != retainBlock->end(); ++it)
        {
            it->walk([&](mlir_ts::ReleaseSlotOp releaseOp) {
                if (releaseOp.getSlot() == slot)
                {
                    releasedAfterRetain = true;
                }
            });
        }

        if (releasedAfterRetain)
        {
            return;
        }

        if (blockExitsFunction(retainBlock))
        {
            reportLeak(retainOp);
            return;
        }

        for (auto *successor : retainBlock->getSuccessors())
        {
            auto it = releasedFromStart.find(successor);
            if (it != releasedFromStart.end() && !it->second)
            {
                reportLeak(retainOp);
                return;
            }
        }
    }

    void reportLeak(mlir_ts::RetainSlotOp retainOp)
    {
        retainOp.emitError("ownership: this slot takes a reference that some path out of the "
                           "function never gives back");
        signalPassFailure();
    }
};

} // end anonymous namespace

#undef DEBUG_TYPE

/// Create pass.
std::unique_ptr<mlir::Pass> mlir_ts::createOwnershipVerifierPass()
{
    return std::make_unique<OwnershipVerifierPass>();
}
