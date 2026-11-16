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

        // a closure's box: it lives as long as the closure value that owns it, and goes where that is
        // given back (`{ let c = () => n.v.get(); } n.v` - the `let`'s release at the block's end frees
        // the box, and `n` with it). A closure that may go anywhere else is no owner to read through.
        for (auto &root : chain.roots)
        {
            auto boundOp = closureOfBox(root.first);
            if (!boundOp)
            {
                continue;
            }

            llvm::SmallVector<mlir::Operation *> closureUses;
            llvm::SmallVector<mlir::Operation *> closureEnds;
            auto aliased = false;
            if (followClosure(boundOp.getResult(), closureUses, closureEnds, aliased) || aliased)
            {
                return false;
            }

            for (auto *end : closureEnds)
            {
                if (outlives(end))
                {
                    return false;
                }
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

        // a store that declares a local in a `try`: the local acquires it at rc's `ts.RetainSlot`
        if (auto *retainSlot = declarationRetainOf(taker))
        {
            return retainSlot;
        }

        // a select of two fresh values: rc's retain of the selected one
        if (auto *retain = selectRetainOf(taker, value))
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

        // the store that declares a local in a `try`, rc's `ts.RetainSlot` right after it
        if (declarationRetainOf(taker))
        {
            return true;
        }

        if (mergeRetainOf(taker, value) || selectRetainOf(taker, value))
        {
            return true;
        }

        auto acquires = false;
        forEachUse(value, [&](mlir::Operation *user, mlir::Value) {
            acquires = acquires || (mlir::isa<mlir_ts::RetainOp>(user) && user->getBlock() == taker->getBlock() &&
                                    user->isBeforeInBlock(taker));
        });

        return acquires;
    }

    mlir::Operation *OwnershipInferencePass::mergeRetainOf(mlir::Operation *taker, mlir::Value value)
    {
        auto branchOp = mlir::dyn_cast<mlir::BranchOpInterface>(taker);
        if (!branchOp)
        {
            return nullptr;
        }

        // the block argument this branch passes the value (or a view of it) as
        auto root = rootOf(value);
        mlir::BlockArgument merged;
        for (auto &operand : taker->getOpOperands())
        {
            if (rootOf(operand.get()) != root)
            {
                continue;
            }

            auto arguments = mergedValuesOf(taker, operand.get());
            if (arguments.size() != 1 || merged)
            {
                return nullptr;
            }

            merged = mlir::dyn_cast<mlir::BlockArgument>(arguments.front());
        }

        if (!merged)
        {
            return nullptr;
        }

        // rc's one retain of the merged value, in its block
        mlir::Operation *retain = nullptr;
        auto other = false;
        forEachUse(merged, [&](mlir::Operation *user, mlir::Value) {
            if (mlir::isa<mlir_ts::RetainOp>(user))
            {
                other = other || retain || user->getBlock() != merged.getOwner();
                retain = user;
            }
        });

        if (!retain || other)
        {
            return nullptr;
        }

        // every way in: null or undefined, or a fresh value - not a loop's back edge
        auto *block = merged.getOwner();
        mlir::DominanceInfo dominance(block->getParentOp());
        for (auto predecessor = block->pred_begin(); predecessor != block->pred_end(); ++predecessor)
        {
            auto predecessorBranch = mlir::dyn_cast<mlir::BranchOpInterface>((*predecessor)->getTerminator());
            if (!predecessorBranch || dominance.dominates(block, *predecessor))
            {
                return nullptr;
            }

            auto operands = predecessorBranch.getSuccessorOperands(predecessor.getSuccessorIndex());
            if (merged.getArgNumber() < operands.getProducedOperandCount())
            {
                return nullptr;
            }

            auto incoming = operands[merged.getArgNumber()];
            if (!incoming)
            {
                return nullptr;
            }

            auto constant = incoming;
            while (auto castOp = constant.getDefiningOp<mlir_ts::CastOp>())
            {
                constant = castOp.getIn();
            }

            if (constant.getDefiningOp<mlir_ts::NullOp>() || constant.getDefiningOp<mlir_ts::UndefOp>() ||
                constant.getDefiningOp<mlir_ts::OptionalUndefOp>())
            {
                continue;
            }

            if (mlir::isa<mlir::BlockArgument>(rootOf(incoming)) || !isFresh(rootOf(incoming)))
            {
                return nullptr;
            }
        }

        return retain;
    }

    mlir::Operation *OwnershipInferencePass::selectRetainOf(mlir::Operation *taker, mlir::Value value)
    {
        auto selectOp = mlir::dyn_cast_or_null<mlir::arith::SelectOp>(taker);
        if (!selectOp)
        {
            return nullptr;
        }

        // two fresh values of this function, each made apart (not the same value chosen either way)
        auto first = rootOf(selectOp.getTrueValue());
        auto second = rootOf(selectOp.getFalseValue());
        auto root = rootOf(value);
        if (first == second || (root != first && root != second))
        {
            return nullptr;
        }

        for (auto chosen : {first, second})
        {
            if (mlir::isa<mlir::BlockArgument>(chosen) || !isFresh(chosen) ||
                !own_facts::ownsHeap(chosen.getLoc(), chosen.getType()) || isHandle(chosen))
            {
                return nullptr;
            }
        }

        // rc's one retain of the selected value, in its block, after it
        mlir::Operation *retain = nullptr;
        auto other = false;
        forEachUse(selectOp.getResult(), [&](mlir::Operation *user, mlir::Value) {
            if (mlir::isa<mlir_ts::RetainOp>(user))
            {
                other = other || retain || user->getBlock() != selectOp->getBlock() || !selectOp->isBeforeInBlock(user);
                retain = user;
            }
        });

        return retain && !other ? retain : nullptr;
    }

} // namespace own_inference
