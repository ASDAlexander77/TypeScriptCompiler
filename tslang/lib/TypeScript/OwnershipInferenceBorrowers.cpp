// OwnershipInferencePass (see OwnershipInferencePass.h): locals that borrow - the borrower of a
// value, and every use of the borrow up to where the owner ends; a local lent to a call and
// given a new value before the call takes it.

#include "OwnershipInferencePass.h"

#define DEBUG_TYPE "own"

namespace own_inference
{

    void OwnershipInferencePass::checkArgumentsAssigned()
    {
        getFunction().walk([&](mlir::Operation *call) {
            if (!isCall(call))
            {
                return mlir::WalkResult::advance();
            }

            for (auto argument : callArgs(call))
            {
                // a method's object: `this` of the method bound to it
                if (auto getThisOp = argument.getDefiningOp<mlir_ts::GetThisOp>())
                {
                    auto *boundOp = getThisOp->getOperand(0).getDefiningOp();
                    if (mlir::isa_and_nonnull<mlir_ts::ThisSymbolRefOp, mlir_ts::ThisVirtualSymbolRefOp>(boundOp))
                    {
                        argument = boundOp->getOperand(0);
                    }
                }

                auto *read = argument.getDefiningOp();
                auto local = mlir::dyn_cast_or_null<mlir_ts::VariableOp>(slotOf(argument));
                if (!read || !local || !ownsHeap(argument) || isHandle(argument))
                {
                    continue;
                }

                auto assignedByClosure = closureAssigns(local);
                auto *write = findBetween(read, call, [&](mlir::Operation *op) {
                    if (assignedByClosure && isCall(op))
                    {
                        return true;
                    }

                    auto storeOp = mlir::dyn_cast<mlir_ts::StoreOp>(op);
                    auto releaseSlotOp = mlir::dyn_cast<mlir_ts::ReleaseSlotOp>(op);
                    return (storeOp && storeOp.getReference() == local.getResult()) ||
                           (releaseSlotOp && releaseSlotOp.getSlot() == local.getResult());
                });
                if (!write)
                {
                    continue;
                }

                if (reporting())
                {
                    auto name = varName(local);
                    auto callee = calleeName(call);
                    auto diag = call->emitError("'")
                                << name << "' is lent to '" << (callee.empty() ? "a function" : callee)
                                << "', but is given a new value before the call takes it; -mm=own cannot prove "
                                   "it outlives the call";
                    diag.attachNote(write->getLoc()) << "'" << name << "' is given a new value here";
                    signalPassFailure();
                }

                return mlir::WalkResult::interrupt();
            }

            return mlir::WalkResult::advance();
        });
    }

    bool OwnershipInferencePass::closureAssigns(mlir_ts::VariableOp local)
    {
        if (!local.getCaptured().value_or(false))
        {
            return false;
        }

        for (auto *user : local->getUsers())
        {
            auto storeOp = mlir::dyn_cast<mlir_ts::StoreOp>(user);
            if (mlir::isa<mlir_ts::LoadOp, mlir_ts::ReleaseSlotOp, mlir_ts::RetainCellOp, mlir_ts::ReleaseCellOp,
                          mlir_ts::DebugVariableOp>(user) ||
                (storeOp && storeOp.getReference() == local.getResult()))
            {
                continue;
            }

            // the cell put into a closure's capture box
            auto propertyRefOp = storeOp ? storeOp.getReference().getDefiningOp<mlir_ts::PropertyRefOp>()
                                         : mlir_ts::PropertyRefOp();
            auto boundOp = propertyRefOp ? closureOfBox(propertyRefOp.getObjectRef()) : mlir_ts::CreateBoundFunctionOp();
            if (!boundOp)
            {
                return true;
            }

            auto assigned = boundOp->getAttrOfType<mlir::DenseI32ArrayAttr>(OWN_ASSIGNS_CAPTURES_ATTR_NAME);
            if (assigned && llvm::is_contained(assigned.asArrayRef(), static_cast<int32_t>(propertyRefOp.getPosition())))
            {
                return true;
            }
        }

        return false;
    }

    mlir_ts::VariableOp OwnershipInferencePass::borrowerOf(mlir::Operation *acquisition, bool consumed)
    {
        mlir_ts::VariableOp varOp;
        if (auto retainSlotOp = mlir::dyn_cast<mlir_ts::RetainSlotOp>(acquisition))
        {
            varOp = retainSlotOp.getSlot().getDefiningOp<mlir_ts::VariableOp>();
        }
        else if (auto retainOp = mlir::dyn_cast<mlir_ts::RetainOp>(acquisition))
        {
            // rc's reference for a `let` that takes the value over: `let ri = <I>raw` retains the
            // interface and declares `ri` consuming it
            varOp = consumingLetOf(retainOp.getReference());
            consumed = true;
        }
        else
        {
            varOp = mlir::dyn_cast<mlir_ts::VariableOp>(acquisition);
        }

        if (!varOp || !varOp.getInitializer() || !isOwningVariable(varOp) || varOp.getCaptured().value_or(false))
        {
            return {};
        }

        if (varOp->hasAttr(OWNED_LOCAL_CONSUMED_ATTR_NAME))
        {
            return consumed ? varOp : mlir_ts::VariableOp();
        }

        auto retains = llvm::any_of(varOp.getResult().getUsers(),
                                    [](mlir::Operation *user) { return mlir::isa<mlir_ts::RetainSlotOp>(user); });
        return retains ? varOp : mlir_ts::VariableOp();
    }

    bool OwnershipInferencePass::borrowWith(mlir_ts::VariableOp borrower, mlir::Operation *acquisition, llvm::ArrayRef<mlir::Operation *> ends,
                    llvm::StringRef owner, llvm::SetVector<mlir::Operation *> &toErase)
    {
        if (!tryBorrow(borrower, ends, owner, toErase))
        {
            return false;
        }

        if (mlir::isa<mlir_ts::RetainOp>(acquisition))
        {
            toErase.insert(acquisition);
        }

        return true;
    }

    mlir_ts::VariableOp OwnershipInferencePass::consumingLetOf(mlir::Value value)
    {
        for (auto *user : value.getUsers())
        {
            auto varOp = mlir::dyn_cast<mlir_ts::VariableOp>(user);
            if (varOp && varOp.getInitializer() == value && varOp->hasAttr(OWNED_LOCAL_CONSUMED_ATTR_NAME))
            {
                return varOp;
            }
        }

        return {};
    }

    llvm::SmallVector<mlir::Operation *> OwnershipInferencePass::slotEnds(mlir::Value slot)
    {
        llvm::SmallVector<mlir::Operation *> ends;
        for (auto *user : slot.getUsers())
        {
            if (mlir::isa<mlir_ts::ReleaseSlotOp, mlir_ts::StoreOp>(user))
            {
                ends.push_back(user);
            }
        }

        return ends;
    }

    bool OwnershipInferencePass::tryBorrow(mlir_ts::VariableOp borrower, llvm::ArrayRef<mlir::Operation *> ends, llvm::StringRef owner,
                   llvm::SetVector<mlir::Operation *> &toErase)
    {
        auto name = varName(borrower);
        llvm::SmallVector<BorrowUse> uses;
        llvm::SmallVector<mlir::Operation *> bookkeeping;
        mlir::Operation *declaration = borrower;
        if (auto *kept = collectBorrowerUses(borrower, declaration, uses, bookkeeping))
        {
            if (isAssignmentOf(kept, borrower))
            {
                reportBorrowerAssigned(kept, name, owner);
            }
            else
            {
                reportBorrowEscapes(kept, name, owner);
            }

            return false;
        }

        for (auto *end : ends)
        {
            for (auto &use : uses)
            {
                if (reachableAfter(end, use.op, use.kills))
                {
                    reportBorrowOutlives(use.op, end, name, owner);
                    return false;
                }
            }
        }

        toErase.insert(bookkeeping.begin(), bookkeeping.end());
        return true;
    }

    mlir::Operation *OwnershipInferencePass::collectBorrowerUses(mlir_ts::VariableOp borrower, llvm::ArrayRef<mlir::Operation *> kills,
                                         llvm::SmallVectorImpl<BorrowUse> &uses,
                                         llvm::SmallVectorImpl<mlir::Operation *> &bookkeeping)
    {
        for (auto *user : borrower.getResult().getUsers())
        {
            if (mlir::isa<mlir_ts::RetainSlotOp, mlir_ts::ReleaseSlotOp>(user))
            {
                bookkeeping.push_back(user);
                continue;
            }

            // `--di`'s record of the local for the debugger names it and keeps nothing
            if (mlir::isa<mlir_ts::DebugVariableOp>(user))
            {
                continue;
            }

            auto readOp = mlir::dyn_cast<mlir_ts::LoadOp>(user);
            if (!readOp)
            {
                return user;
            }

            uses.push_back({user, {kills.begin(), kills.end()}});
            if (auto *kept = walkBorrowed(readOp.getResult(), kills, uses))
            {
                return kept;
            }
        }

        return nullptr;
    }

    bool OwnershipInferencePass::isAssignmentOf(mlir::Operation *op, mlir_ts::VariableOp varOp)
    {
        auto storeOp = mlir::dyn_cast<mlir_ts::StoreOp>(op);
        return storeOp && storeOp.getReference() == varOp.getResult();
    }

    bool OwnershipInferencePass::passesOn(mlir::OpOperand &use, llvm::SmallVectorImpl<mlir::Value> &values,
                         llvm::SmallVectorImpl<mlir::Operation *> *renewedAt)
    {
        auto *user = use.getOwner();
        if (mlir::isa<mlir_ts::CastOp>(user) || isView(user))
        {
            values.push_back(user->getResult(0));
            return true;
        }

        // one of the values a select chooses between, not its condition
        if (auto selectOp = mlir::dyn_cast<mlir::arith::SelectOp>(user); selectOp && use.get() != selectOp.getCondition())
        {
            values.push_back(selectOp.getResult());
            return true;
        }

        if (auto branchOp = mlir::dyn_cast<mlir::BranchOpInterface>(user))
        {
            if (auto argument = branchOp.getSuccessorBlockArgument(use.getOperandNumber()))
            {
                values.push_back(*argument);
                if (renewedAt)
                {
                    renewedAt->push_back(&argument->getOwner()->front());
                }

                return true;
            }

            return false;
        }

        if (auto storeOp = mlir::dyn_cast<mlir_ts::StoreOp>(user); storeOp && storeOp.getValue() == use.get())
        {
            if (auto varOp = storeOp.getReference().getDefiningOp<mlir_ts::VariableOp>();
                varOp && !isOwningVariable(varOp) && !varOp.getCaptured().value_or(false))
            {
                if (renewedAt && varOp.getInitializer())
                {
                    renewedAt->push_back(varOp);
                }

                for (auto *varUser : varOp.getResult().getUsers())
                {
                    if (auto loadOp = mlir::dyn_cast<mlir_ts::LoadOp>(varUser))
                    {
                        values.push_back(loadOp.getResult());
                    }
                    else if (renewedAt && isAssignmentOf(varUser, varOp))
                    {
                        renewedAt->push_back(varUser);
                    }
                }

                return true;
            }
        }

        return false;
    }

    mlir::Operation *OwnershipInferencePass::walkBorrowed(
        mlir::Value start, llvm::ArrayRef<mlir::Operation *> kills, llvm::SmallVectorImpl<BorrowUse> &uses,
        llvm::function_ref<bool(mlir::Operation *, llvm::ArrayRef<mlir::Operation *>)> isBorrower,
        Keeping keeping, bool countedHandle, bool startBorrowed)
    {
        struct Pending
        {
            mlir::Value value;
            bool borrowed;
            llvm::SmallVector<mlir::Operation *, 2> kills;
        };

        llvm::SmallVector<Pending> values;
        // `startBorrowed` false: `start` is what a read produced (a field's reference), whose uses
        // are uses of the borrow but not the borrowed value itself
        values.push_back({start, startBorrowed, {kills.begin(), kills.end()}});
        llvm::SmallPtrSet<mlir::Value, 16> seen{start};
        while (!values.empty())
        {
            auto pending = values.pop_back_val();
            for (auto &use : pending.value.getUses())
            {
                auto *user = use.getOwner();
                if (pending.borrowed && keeping.bookkeeping &&
                    ((mlir::isa<mlir_ts::ReleaseOp>(user) && borrowingCall(rootOf(pending.value))) ||
                     (keeping.returns && mlir::isa<mlir_ts::RetainOp>(user))))
                {
                    keeping.bookkeeping->push_back(user);
                    continue;
                }

                // rc's reference for a borrowing box's copy goes with the closure (decideClosures), and
                // one for a borrowing field's with the field (classifyBorrowingFields)
                if (captureRetains.contains(user) || fieldBorrowRetains.contains(user))
                {
                    continue;
                }

                uses.push_back({user, pending.kills, isCall(user) && isCalleeOperand(use)});
                // copied into a borrowing box: whatever runs the closure reads it
                if (auto found = captureStores.find(user); found != captureStores.end())
                {
                    for (auto *closureUse : found->second->uses)
                    {
                        uses.push_back({closureUse, pending.kills});
                    }
                }

                // stored into a field a borrowing record borrows (spec 32): whatever uses the record
                // reads it
                if (auto found = recordStoreUses.find(user); found != recordStoreUses.end())
                {
                    for (auto &recordUse : found->second->walk.uses)
                    {
                        uses.push_back({recordUse.op, pending.kills, recordUse.callee});
                    }
                }

                if (pending.borrowed && keeping.returns && mlir::isa<mlir_ts::ReturnInternalOp>(user))
                {
                    continue;
                }

                if (pending.borrowed)
                {
                    // a handle read that rc counts (spec 23.3): a return or an `any` box rc
                    // retained for is a counted copy of its own; passesOn would hand them on
                    if (countedHandle && isCountedHandOff(user, pending.value))
                    {
                        continue;
                    }

                    // a field read out of a tuple value (`console.log(found)` of an optional record):
                    // a number keeps nothing, and a field that holds a block borrows it as the tuple does
                    if (mlir::isa<mlir_ts::ExtractPropertyOp>(user))
                    {
                        auto field = user->getResult(0);
                        if (mayPointInto(field) && seen.insert(field).second)
                        {
                            values.push_back({field, true, pending.kills});
                        }

                        continue;
                    }

                    llvm::SmallVector<mlir::Value> same;
                    llvm::SmallVector<mlir::Operation *, 2> renewedAt;
                    if (passesOn(use, same, &renewedAt))
                    {
                        for (auto value : same)
                        {
                            if (mayPointInto(value) && seen.insert(value).second)
                            {
                                values.push_back({value, true, renewedAt.empty() ? pending.kills : renewedAt});
                            }
                        }

                        continue;
                    }

                    // a handle read that rc counts (spec 23.3): its retain, and what takes the value
                    // it retains for, are a counted copy of their own, as a string copy is; the
                    // read's other uses are the borrow's (Ruling 7)
                    if (countedHandle && isCountedHandleReceiver(user, pending.value))
                    {
                        continue;
                    }

                    if (isBorrower && isBorrower(user, pending.kills))
                    {
                        continue;
                    }

                    if (mlir::isa<mlir_ts::RetainOp>(user) || !isBorrow(user, pending.value))
                    {
                        return user;
                    }
                }

                // a copy (spec 22.2) reads the borrow into a block of its own, which points into
                // nothing, and so does a concatenation: `s + t` is a new string whatever s and t are
                // (a `+` of two strings lowers to ts.StringConcat, which always allocates)
                if (mlir::isa<mlir_ts::StringCopyOp, mlir_ts::StringConcatOp>(user) ||
                    (mlir::isa<mlir_ts::ArithmeticBinaryOp>(user) && mlir::isa<mlir_ts::StringType>(user->getResult(0).getType())))
                {
                    continue;
                }

                // a call whose result is the caller's own (callResultIsOwn) ends the walk at the call
                if (callResultIsOwn(user))
                {
                    continue;
                }

                // A result that borrows what it was given borrows this too, and is walked as it.
                for (auto result : user->getResults())
                {
                    if (mayPointInto(result) && seen.insert(result).second)
                    {
                        values.push_back({result, !!borrowingCall(result), pending.kills});
                    }
                }
            }
        }

        return nullptr;
    }

    bool OwnershipInferencePass::callResultIsOwn(mlir::Operation *call)
    {
        if (!isCall(call) || call->hasAttr(OWN_RESULT_BOUNDED_ATTR_NAME))
        {
            return false;
        }

        return call->hasAttr(OWN_FRESH_RESULT_ATTR_NAME) ||
               (call->getNumResults() == 1 && !isRecordResult(call) && isFresh(call->getResult(0)));
    }

    bool OwnershipInferencePass::isCalleeOperand(mlir::OpOperand &use)
    {
        auto *user = use.getOwner();
        auto callee = isCall(user) ? calleeValue(user) : mlir::Value();
        if (!callee)
        {
            return false;
        }

        if (use.getOperandNumber() == 0)
        {
            return true;
        }

        auto getThisOp = use.get().getDefiningOp<mlir_ts::GetThisOp>();
        auto getMethodOp = callee.getDefiningOp<mlir_ts::GetMethodOp>();
        auto methodField = getThisOp ? getThisOp.getOperand().getDefiningOp<mlir_ts::LoadOp>() : mlir_ts::LoadOp();
        return use.getOperandNumber() == 1 && getMethodOp && methodField &&
               getThisOp.getOperand() == getMethodOp.getBoundFunc() &&
               mlir::isa<mlir_ts::BoundRefType>(methodField.getReference().getType());
    }

} // namespace own_inference
