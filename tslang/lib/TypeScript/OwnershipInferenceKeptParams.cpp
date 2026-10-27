// OwnershipInferencePass (see OwnershipInferencePass.h): a kept parameter released where it is
// not moved (spec 33).

#include "OwnershipInferencePass.h"

#define DEBUG_TYPE "own"

namespace own_inference
{

    mlir_ts::VariableOp OwnershipInferencePass::parameterSlot(int32_t index)
    {
        auto &body = getFunction().getFunctionBody();
        if (body.empty() || static_cast<unsigned>(index) >= body.front().getNumArguments())
        {
            return {};
        }

        auto slots = parameterSlotsOf(body.front().getArgument(index));
        return slots.empty() ? mlir_ts::VariableOp() : slots.front();
    }

    bool OwnershipInferencePass::isReleasedParam(mlir::Value slot)
    {
        auto f = getFunction();
        auto attr = f->getAttrOfType<mlir::DenseI32ArrayAttr>(OWN_PARAMS_RELEASED_ATTR_NAME);
        auto varOp = slot.getDefiningOp<mlir_ts::VariableOp>();
        auto index = -1;
        return attr && varOp && isParameterSlot(varOp, index) && llvm::is_contained(attr.asArrayRef(), index) &&
               llvm::is_contained(ownedParams(f), index);
    }

    bool OwnershipInferencePass::readsSlot(mlir::Value value, mlir::Value slot, int depth)
    {
        auto *def = value.getDefiningOp();
        if (!def)
        {
            return false;
        }

        // too deep to tell: as if it did
        if (depth > 16)
        {
            return true;
        }

        if (auto loadOp = mlir::dyn_cast<mlir_ts::LoadOp>(def); loadOp && loadOp.getReference() == slot)
        {
            return true;
        }

        return llvm::any_of(def->getOperands(), [&](mlir::Value operand) { return readsSlot(operand, slot, depth + 1); });
    }

    void OwnershipInferencePass::releaseKeptParamsAtExits()
    {
        exitReleases.clear();
        auto f = getFunction();
        auto attr = f->getAttrOfType<mlir::DenseI32ArrayAttr>(OWN_PARAMS_RELEASED_ATTR_NAME);
        if (!attr)
        {
            return;
        }

        for (auto index : attr.asArrayRef())
        {
            auto varOp = parameterSlot(index);
            if (!varOp || !llvm::is_contained(ownedParams(f), index))
            {
                continue;
            }

            for (auto &block : f.getFunctionBody())
            {
                auto *exit = block.empty() ? nullptr : &block.back();
                // a thrown value made of the parameter takes it with it: no release there (a leak,
                // if it does not)
                auto leaves = mlir::isa_and_nonnull<mlir_ts::ReturnInternalOp>(exit) ||
                              (mlir::isa_and_nonnull<mlir_ts::ThrowCallOp>(exit) &&
                               !readsSlot(exit->getOperand(0), varOp.getResult()));
                if (!leaves)
                {
                    continue;
                }

                mlir::OpBuilder builder(exit);
                exitReleases.push_back(builder.create<mlir_ts::ReleaseSlotOp>(exit->getLoc(), varOp.getResult()));
            }
        }
    }

    void OwnershipInferencePass::decideParam(mlir::Value slot, llvm::ArrayRef<SlotReceiver> receivers, llvm::SetVector<mlir::Operation *> &toErase)
    {
        auto name = varName(slot.getDefiningOp<mlir_ts::VariableOp>());
        auto &receiver = receivers.front();
        auto *taker = takerOf(receiver.retain, receiver.value);
        if (receivers.size() > 1)
        {
            auto &second = receivers[1];
            auto *secondTaker = takerOf(second.retain, second.value);
            reportUseAfterMove(secondTaker ? secondTaker : second.retain, taker ? taker : receiver.retain, name);
            return;
        }

        // one the body releases on the paths that do not move it (spec 33): the move's own check
        // erases the releases at the exits it reaches, and places one on each edge into where the
        // paths join that did not move it (spec 30)
        auto released = isReleasedParam(slot);
        if (!checkSlotMove(receiver.retain, receiver.value, receiver.load, toErase, released))
        {
            return;
        }

        if (released)
        {
            if (!isCall(receiver.retain))
            {
                toErase.insert(receiver.retain);
            }

            return;
        }

        auto &body = getFunction().getFunctionBody();
        auto *moved = body.findAncestorOpInRegion(*taker);
        auto everyPath = true;
        getFunction().walk([&](mlir_ts::ReturnInternalOp returnOp) {
            auto *exit = body.findAncestorOpInRegion(*returnOp);
            if (everyPath && (!moved || !exit || !dominance->dominates(moved, exit)))
            {
                reportKeptOnSomePaths(taker, returnOp, name);
                everyPath = false;
            }
        });

        // rc's retain goes with the move - a `ts.RetainSlot` too, for an owning local the parameter
        // moves into (`let r = init`, spec 33); a call that keeps it is the move itself
        if (everyPath && !isCall(receiver.retain))
        {
            toErase.insert(receiver.retain);
        }
    }

    bool OwnershipInferencePass::isReadOutOfException(mlir::Value value)
    {
        auto loadOp = rootOf(value).getDefiningOp<mlir_ts::LoadOp>();
        auto argument = loadOp ? mlir::dyn_cast<mlir::BlockArgument>(loadOp.getReference()) : mlir::BlockArgument();
        return argument && argument.getOwner()->isEntryBlock() && getFunction().getName().starts_with(".eh.copy.");
    }

    mlir_ts::LoadOp OwnershipInferencePass::keptParamLoadOf(mlir::Value value)
    {
        auto loadOp = rootOf(value).getDefiningOp<mlir_ts::LoadOp>();
        auto varOp = loadOp ? loadOp.getReference().getDefiningOp<mlir_ts::VariableOp>() : mlir_ts::VariableOp();
        auto index = -1;
        if (!varOp || !isParameterSlot(varOp, index) || !llvm::is_contained(ownedParams(getFunction()), index))
        {
            return {};
        }

        return loadOp;
    }

    void OwnershipInferencePass::decideSlotReceivers(mlir::Value slot, llvm::ArrayRef<SlotReceiver> receivers, llvm::StringRef owner,
                             llvm::SetVector<mlir::Operation *> &toErase)
    {
        auto ends = slotEnds(slot);
        mlir_ts::VariableOp someBorrower;
        for (auto &receiver : receivers)
        {
            if (auto borrower = borrowerOf(receiver.retain))
            {
                someBorrower = borrower;
                borrowWith(borrower, receiver.retain, ends, owner, toErase);
            }
        }

        for (auto &receiver : receivers)
        {
            if (borrowerOf(receiver.retain))
            {
                continue;
            }

            llvm::SetVector<mlir::Operation *> unused;
            if (checkSlotMove(receiver.retain, receiver.value, receiver.load, unused))
            {
                reportMovedWhileBorrowed(receiver.retain, owner, someBorrower ? varName(someBorrower) : "another variable");
            }
        }
    }

} // namespace own_inference
