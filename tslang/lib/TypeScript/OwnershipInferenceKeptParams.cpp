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
        llvm::SmallVector<int32_t> released;
        if (attr)
        {
            released.append(attr.asArrayRef().begin(), attr.asArrayRef().end());
        }

        // A counted parameter this function keeps (--own-fallback=count): rc's caller held a share for
        // the call and gave it back after; own's caller moved that share in, so the body gives it back
        // at its exits, and keeps rc's retains for the places it stored it in. Not in the dry run, which
        // counts nothing yet.
        if (!dryRun)
        {
            for (auto unit : countedValues)
            {
                auto varOp = unit.getDefiningOp<mlir_ts::VariableOp>();
                auto index = -1;
                if (varOp && isParameterSlot(varOp, index) && !llvm::is_contained(released, index))
                {
                    released.push_back(index);
                }
            }

            llvm::sort(released.begin() + (attr ? attr.size() : 0), released.end());
        }

        for (auto index : released)
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
                // ...and the end of a cleanup the exception leaves the function from (its own `finally`, no
                // landing pad of its own after it): the kept parameter leaked on that way out (final-review M-1)
                auto leaves = mlir::isa_and_nonnull<mlir_ts::ReturnInternalOp>(exit) ||
                              (mlir::isa_and_nonnull<mlir_ts::ThrowCallOp>(exit) &&
                               !readsSlot(exit->getOperand(0), varOp.getResult())) ||
                              (mlir::isa_and_nonnull<mlir_ts::EndCleanupOp>(exit) && exit->getNumSuccessors() == 0);
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
            // moved on paths that exclude each other (`if (f) a.push(v); else a[0] = v;`)
            if (movesOnExclusivePaths(slot, receivers, isReleasedParam(slot), toErase))
            {
                return;
            }

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

        // ...and a `throw` the function is left by before the move (`if (c.n < 0) throw "no"; this.items.push(c);`):
        // the exception takes the function out holding what its caller gave it up for, which nothing gave back
        // (TSLANG 209). Released there, as releaseKeptParamsAtExits releases one at its throws - unless the move may
        // have run on the way (it took the value), or what is thrown is made of it (it goes with the exception)
        if (everyPath && moved)
        {
            // ...and for a call that may throw on the way to the move, UnwindReleasePass gives it back on its unwind
            if (auto slotOp = slot.getDefiningOp<mlir_ts::VariableOp>())
            {
                setAttrTracked(slotOp, OWN_KEPT_PARAM_ATTR_NAME);
                setAttrTracked(taker, OWN_KEPT_PARAM_TAKER_ATTR_NAME);
            }

            getFunction().walk([&](mlir_ts::ThrowCallOp throwOp) {
                if (reachableAfter(taker, throwOp, nullptr) || readsSlot(throwOp->getOperand(0), slot))
                {
                    return;
                }

                mlir::OpBuilder builder(throwOp);
                exitReleases.push_back(builder.create<mlir_ts::ReleaseSlotOp>(throwOp->getLoc(), slot));
            });
        }

        // rc's retain goes with the move - a `ts.RetainSlot` too, for an owning local the parameter
        // moves into (`let r = init`, spec 33); a call that keeps it is the move itself
        if (everyPath && !isCall(receiver.retain))
        {
            toErase.insert(receiver.retain);
        }
    }

    // A parameter the body releases on the paths that do not move it (released): each move empties
    // its slot after its taker, so the releases at the exits free nothing on the paths that moved
    // it. One the function keeps on every path: each exit is reached only through one of the moves.
    // No read of the slot may follow a move, and no move may meet itself again around a loop.
    bool OwnershipInferencePass::movesOnExclusivePaths(mlir::Value slot, llvm::ArrayRef<SlotReceiver> receivers, bool released,
                                                       llvm::SetVector<mlir::Operation *> &toErase)
    {
        auto assignments = slotAssignments(slot);
        llvm::SmallVector<mlir::Operation *> kills(assignments.begin(), assignments.end());
        if (auto varOp = slot.getDefiningOp<mlir_ts::VariableOp>())
        {
            kills.push_back(varOp);
        }

        // the taker of a retain: what takerOf says, or - for a read unwrapped once (`value` of
        // `T | undefined`, narrowed) and moved on each path - the next op that uses what was retained
        auto takerAfter = [&](const SlotReceiver &receiver) -> mlir::Operation * {
            if (auto *taker = takerOf(receiver.retain, receiver.value))
            {
                return taker;
            }

            if (!mlir::isa<mlir_ts::RetainOp>(receiver.retain))
            {
                return nullptr;
            }

            auto retained = receiver.retain->getOperand(0);
            for (auto *op = receiver.retain->getNextNode(); op; op = op->getNextNode())
            {
                if (!mlir::isa<mlir_ts::RetainOp>(op) && llvm::is_contained(op->getOperands(), retained))
                {
                    return op;
                }
            }

            return nullptr;
        };

        // what the read returned reaches only the retains and the takers, unwrapped on the way
        std::function<bool(mlir::Value)> onlyMoved = [&](mlir::Value value) {
            return llvm::all_of(value.getUsers(), [&](mlir::Operation *user) {
                if (mlir::isa<mlir_ts::DebugVariableOp>(user))
                {
                    return true;
                }

                if (mlir::isa<mlir_ts::ValueOp, mlir_ts::CastOp>(user))
                {
                    return onlyMoved(user->getResult(0));
                }

                return llvm::any_of(receivers, [&](const SlotReceiver &other) {
                    return user == other.retain || user == takerAfter(other);
                });
            });
        };

        llvm::SmallVector<mlir::Operation *> takers;
        for (auto &receiver : receivers)
        {
            auto *taker = takerAfter(receiver);
            auto load = receiver.load;
            if (!taker || !orderedInBody(load, taker) || reachableAfter(taker, load, kills))
            {
                return false;
            }

            if (!onlyMoved(receiver.value))
            {
                return false;
            }

            for (auto *use : slot.getUsers())
            {
                if (!mlir::isa<mlir_ts::ReleaseSlotOp>(use) && use != load.getOperation() &&
                    reachableAfter(taker, use, kills))
                {
                    return false;
                }
            }

            takers.push_back(taker);
        }

        // the moves exclude each other: none is reached from another
        for (auto *from : takers)
        {
            for (auto *to : takers)
            {
                if (from != to && reachableAfter(from, to, kills))
                {
                    return false;
                }
            }
        }

        if (!released)
        {
            auto *definition = slot.getDefiningOp();
            auto everyPath = definition != nullptr;
            getFunction().walk([&](mlir_ts::ReturnInternalOp returnOp) {
                everyPath = everyPath && !reachableAfter(definition, returnOp, takers);
            });
            if (!everyPath)
            {
                return false;
            }
        }

        for (auto [receiver, taker] : llvm::zip(receivers, takers))
        {
            if (released)
            {
                slotClears.push_back({slot, taker});
            }

            markBoundMethodsTaking(taker, receiver.value);
            if (!isCall(receiver.retain))
            {
                toErase.insert(receiver.retain);
            }
        }

        return true;
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
        // the uses of borrows by `let`s that only borrow (`let b = q; ... b = r`), each up to the
        // `let`'s next assignment: when every borrower is one, a move none of them reaches is a move
        llvm::SmallVector<BorrowUse> localUses;
        auto onlyLocalBorrowers = true;
        for (auto &receiver : receivers)
        {
            mlir::Operation *from = nullptr;
            if (auto borrower = localBorrowerOf(receiver.retain, from))
            {
                someBorrower = borrower;
                if (borrowWith(borrower, receiver.retain, ends, owner, toErase, from))
                {
                    llvm::SmallVector<mlir::Operation *> bookkeeping;
                    mlir::Operation *declaration = borrower;
                    collectBorrowerUses(borrower, declaration, localUses, bookkeeping, from);
                }
            }
            else if (auto borrower = borrowerOf(receiver.retain))
            {
                someBorrower = borrower;
                onlyLocalBorrowers = false;
                borrowWith(borrower, receiver.retain, ends, owner, toErase);
            }
        }

        auto moves = 0;
        for (auto &receiver : receivers)
        {
            mlir::Operation *from = nullptr;
            if (borrowerOf(receiver.retain) || localBorrowerOf(receiver.retain, from))
            {
                continue;
            }

            ++moves;
        }

        for (auto &receiver : receivers)
        {
            mlir::Operation *from = nullptr;
            if (borrowerOf(receiver.retain) || localBorrowerOf(receiver.retain, from))
            {
                continue;
            }

            llvm::SetVector<mlir::Operation *> unused;
            if (!checkSlotMove(receiver.retain, receiver.value, receiver.load, unused))
            {
                continue;
            }

            auto reached = llvm::any_of(localUses, [&](BorrowUse &use) { return reachableAfter(receiver.retain, use.op, use.kills); });
            if (moves == 1 && onlyLocalBorrowers && !reached)
            {
                checkSlotMove(receiver.retain, receiver.value, receiver.load, toErase, true);
                if (!isCall(receiver.retain))
                {
                    toErase.insert(receiver.retain);
                }

                continue;
            }

            reportMovedWhileBorrowed(receiver.retain, owner, someBorrower ? varName(someBorrower) : llvm::StringRef());
        }
    }

} // namespace own_inference
