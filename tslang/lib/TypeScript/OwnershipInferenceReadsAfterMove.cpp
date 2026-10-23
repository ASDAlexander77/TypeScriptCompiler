// OwnershipInferencePass (see OwnershipInferencePass.h): reads after a move, through the new
// owner (spec 29).

#include "OwnershipInferencePass.h"

#define DEBUG_TYPE "own"

namespace own_inference
{

    bool OwnershipInferencePass::onlyReadThroughOwner(mlir::Value value, llvm::function_ref<bool(mlir::Operation *, mlir::Value)> isRead)
    {
        auto all = true;
        forEachUse(value, [&](mlir::Operation *user, mlir::Value used) {
            all = all && (mlir::isa<mlir_ts::DebugVariableOp>(user) ||
                          (!mlir::isa<mlir_ts::RetainOp>(user) && isRead(user, used)));
        });

        return all;
    }

    bool OwnershipInferencePass::isReadAfterMove(mlir::Operation *user, mlir::Value value)
    {
        if (mlir::isa<mlir_ts::PropertyRefOp, mlir_ts::ElementRefOp, mlir_ts::LengthOfOp, mlir_ts::ThisSymbolRefOp,
                      mlir_ts::VirtualSymbolRefOp, mlir_ts::ThisVirtualSymbolRefOp, mlir_ts::InterfaceSymbolRefOp,
                      mlir_ts::GetThisOp, mlir_ts::GetMethodOp, mlir_ts::ArithmeticBinaryOp, mlir_ts::LogicalBinaryOp,
                      mlir_ts::PrintOp, mlir_ts::AssertOp, mlir_ts::HasValueOp, mlir_ts::StringLengthOp>(user))
        {
            return isBorrow(user, value);
        }

        return isCall(user) && !isKeptArgument(user, value);
    }

    bool OwnershipInferencePass::destinationChain(mlir::Operation *taker, Chain &chain)
    {
        auto ownSlot = [](mlir_ts::VariableOp varOp) {
            return varOp && isOwningVariable(varOp) && !varOp.getCaptured().value_or(false);
        };

        if (auto varOp = mlir::dyn_cast<mlir_ts::VariableOp>(taker))
        {
            if (!ownSlot(varOp))
            {
                return false;
            }

            chain.roots.push_back({varOp.getResult(), Chain::Slot});
            return true;
        }

        auto storeOp = mlir::dyn_cast<mlir_ts::StoreOp>(taker);
        if (!storeOp)
        {
            return false;
        }

        auto ref = storeOp.getReference();
        if (auto varOp = ref.getDefiningOp<mlir_ts::VariableOp>())
        {
            if (!ownSlot(varOp))
            {
                return false;
            }

            chain.roots.push_back({ref, Chain::Slot});
            return true;
        }

        if (!isPlace(ref))
        {
            return false;
        }

        chain = chainOf(nullptr, ref);
        if (!chain.handles.empty() || chain.roots.empty())
        {
            return false;
        }

        // a global's field, or a captured variable's: not a place of this function
        return llvm::none_of(chain.roots, [](auto &root) {
            auto loadOp = root.first.template getDefiningOp<mlir_ts::LoadOp>();
            return loadOp && (loadOp.getReference().template getDefiningOp<mlir_ts::AddressOfOp>() ||
                              isLoadedCell(loadOp.getReference()));
        });
    }

    bool OwnershipInferencePass::readsThroughOwner(mlir::Operation *taker, mlir::Value value, mlir::Operation *use,
                           llvm::ArrayRef<mlir::Operation *> made)
    {
        if (!isReadAfterMove(use, value))
        {
            return false;
        }

        Chain chain;
        if (!destinationChain(taker, chain))
        {
            return false;
        }

        // A read again of the old name after a drop still finds the moved value - the slot is not
        // emptied by a move - so neither the read nor the slot's read is where it is made again.
        llvm::SmallVector<mlir::Operation *> kills(made.begin(), made.end());
        kills.push_back(taker);
        llvm::SmallVector<BorrowUse> uses;
        uses.push_back({use, {kills.begin(), kills.end()}});
        for (auto result : use->getResults())
        {
            if (!mayPointInto(result))
            {
                continue;
            }

            // what the read produced - a field's reference, a bound method - still points into the
            // block; a call's result that borrows its argument is the value again and may not be kept
            if (walkBorrowed(result, kills, uses, nullptr, Keeping{nullptr, false}, false, !!borrowingCall(result)))
            {
                return false;
            }
        }

        // a method reached through it whose callee keeps its `this`: a second owner
        for (auto &borrowUse : uses)
        {
            if (!isCall(borrowUse.op))
            {
                continue;
            }

            auto args = callArgs(borrowUse.op);
            for (auto index : ownedParams(borrowUse.op))
            {
                if (static_cast<size_t>(index) < args.size() &&
                    mlir::isa_and_nonnull<mlir_ts::GetThisOp, mlir_ts::ExtractInterfaceThisOp>(args[index].getDefiningOp()))
                {
                    return false;
                }
            }
        }

        auto outlives = [&](mlir::Operation *end) {
            if (end == taker || !reachableAfter(taker, end, made))
            {
                return false;
            }

            return llvm::any_of(uses, [&](BorrowUse &borrowUse) {
                return (borrowUse.op != end || !borrowUse.callee) && reachableAfter(end, borrowUse.op, borrowUse.kills);
            });
        };

        for (auto *end : rootEnds(chain))
        {
            if (outlives(end))
            {
                return false;
            }
        }

        for (auto *drop : drops)
        {
            if (dropsChain(drop, chain, nullptr, value) && outlives(drop))
            {
                return false;
            }
        }

        return true;
    }

    mlir::Operation *OwnershipInferencePass::acquirePoint(mlir::Operation *taker, mlir::Value value)
    {
        // a store into a record, which rc retains as a whole after it is filled
        if (auto *retain = recordRetainOfStore(taker))
        {
            return retain;
        }

        mlir::Operation *first = taker;
        forEachUse(value, [&](mlir::Operation *user, mlir::Value) {
            if (mlir::isa<mlir_ts::RetainOp>(user) && user->getBlock() == taker->getBlock() &&
                user->isBeforeInBlock(first))
            {
                first = user;
            }
        });

        return first;
    }

    bool OwnershipInferencePass::takerAcquires(mlir::Operation *taker, mlir::Value value)
    {
        // a callee that keeps its argument retained it in its own body under rc
        if (isCall(taker))
        {
            return true;
        }

        if (recordRetainOfStore(taker))
        {
            return true;
        }

        if (auto varOp = mlir::dyn_cast<mlir_ts::VariableOp>(taker))
        {
            return llvm::any_of(varOp.getResult().getUsers(),
                                [](mlir::Operation *user) { return mlir::isa<mlir_ts::RetainSlotOp>(user); });
        }

        auto acquires = false;
        forEachUse(value, [&](mlir::Operation *user, mlir::Value) {
            acquires = acquires || (mlir::isa<mlir_ts::RetainOp>(user) && user->getBlock() == taker->getBlock() &&
                                    user->isBeforeInBlock(taker));
        });

        return acquires;
    }

} // namespace own_inference
