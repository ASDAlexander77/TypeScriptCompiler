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

                // what a view of it shows (`f(n, ...)` with `n` given as an interface): the value read
                for (auto depth = 0; depth < 8 && isView(argument.getDefiningOp()) && !argument.getDefiningOp<mlir_ts::CastOp>();
                     depth++)
                {
                    argument = argument.getDefiningOp()->getOperand(0);
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
                // ...or the call itself, when it may run a closure that gives a captured object a new value
                // (OWN_CAPTURED_CELL_POSITION): its callee's borrow of the argument would not see that
                auto places = callDropFields(call);
                if (!write && assignedByClosure && places && llvm::any_of(*places, [](auto &place) {
                        return place.first == OWN_CAPTURED_CELL_POSITION;
                    }))
                {
                    write = call;
                }

                if (!write)
                {
                    continue;
                }

                if (reporting() && write == call)
                {
                    auto name = varName(local);
                    auto callee = calleeName(call);
                    call->emitError("'") << name << "' is lent to '" << (callee.empty() ? "a function" : callee)
                                         << "', which may run a closure that gives '" << name
                                         << "' a new value; -mm=own cannot prove it outlives the call";
                    signalPassFailure();
                }
                else if (reporting())
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
        // a local declared in a `try` is declared without a value (the cleanup releases it) and given
        // its value by a store right before rc's `ts.RetainSlot` (slotDeclarationStore): that store,
        // or that retain, declares it
        auto declaredByStore = false;
        if (auto retainSlotOp = mlir::dyn_cast<mlir_ts::RetainSlotOp>(acquisition))
        {
            varOp = retainSlotOp.getSlot().getDefiningOp<mlir_ts::VariableOp>();
            declaredByStore = static_cast<bool>(slotDeclarationStore(acquisition));
        }
        else if (auto retainOp = mlir::dyn_cast<mlir_ts::RetainOp>(acquisition))
        {
            // rc's reference for a `let` that takes the value over: `let ri = <I>raw` retains the
            // interface and declares `ri` consuming it
            varOp = consumingLetOf(retainOp.getReference());
            consumed = true;
        }
        else if (auto storeOp = mlir::dyn_cast<mlir_ts::StoreOp>(acquisition); storeOp && declarationRetainOf(storeOp))
        {
            varOp = storeOp.getReference().getDefiningOp<mlir_ts::VariableOp>();
            declaredByStore = true;
        }
        else
        {
            varOp = mlir::dyn_cast<mlir_ts::VariableOp>(acquisition);
        }

        if (!varOp || !(varOp.getInitializer() || declaredByStore) || !isOwningVariable(varOp) ||
            varOp.getCaptured().value_or(false))
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
                    llvm::StringRef owner, llvm::SetVector<mlir::Operation *> &toErase, mlir::Operation *from)
    {
        if (!tryBorrow(borrower, ends, owner, toErase, from))
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
                   llvm::SetVector<mlir::Operation *> &toErase, mlir::Operation *from)
    {
        auto name = varName(borrower);
        llvm::SmallVector<BorrowUse> uses;
        llvm::SmallVector<mlir::Operation *> bookkeeping;
        mlir::Operation *declaration = borrower;
        if (auto *kept = collectBorrowerUses(borrower, declaration, uses, bookkeeping, from))
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

    bool OwnershipInferencePass::letBorrows(mlir::Value value)
    {
        // a read of a place or of a parameter - or of a split `let`'s holder (TSLANG 66)
        auto root = rootOf(value);
        auto rootLoad = root.getDefiningOp<mlir_ts::LoadOp>();
        auto slot = rootLoad ? rootLoad.getReference().getDefiningOp<mlir_ts::VariableOp>() : mlir_ts::VariableOp();
        int index = -1;
        return placeReadOf(root) || (slot && (isParameterSlot(slot, index) || slot->hasAttr(OWN_LET_HOLDER_ATTR_NAME)));
    }

    bool OwnershipInferencePass::borrowsOnly(mlir_ts::VariableOp varOp)
    {
        // `let layout = this.computeLayout; if (fp16) layout = this.computeLayoutFP16;` (TSLANG 56):
        // an owning `let` given only reads of places and parameters, each a borrow of its own
        // ...or reads of other owning locals (`let b = q; ... b = r`): classifyLocalBorrowingLets
        auto localReads = localBorrowingLets.contains(varOp);
        auto borrowed = [&](mlir::Value value) { return letBorrows(value) || (localReads && readsOwningLocals(value, varOp)); };

        if (!varOp.getInitializer() || !isOwningVariable(varOp) || varOp.getCaptured().value_or(false) ||
            !borrowed(varOp.getInitializer()))
        {
            return false;
        }

        auto assigned = false;
        for (auto *user : varOp.getResult().getUsers())
        {
            if (isAssignmentOf(user, varOp))
            {
                if (!borrowed(mlir::cast<mlir_ts::StoreOp>(user).getValue()))
                {
                    return false;
                }

                assigned = true;
            }
        }

        return assigned;
    }

    mlir_ts::VariableOp OwnershipInferencePass::borrowingAssignmentOf(mlir::Operation *user, mlir::Value value)
    {
        // the assignment itself
        if (auto storeOp = mlir::dyn_cast<mlir_ts::StoreOp>(user); storeOp && storeOp.getValue() == value)
        {
            auto varOp = storeOp.getReference().getDefiningOp<mlir_ts::VariableOp>();
            return varOp && borrowsOnly(varOp) ? varOp : mlir_ts::VariableOp();
        }

        // rc's reference for it, right before it
        if (mlir::isa<mlir_ts::RetainOp>(user) && user->getOperand(0) == value)
        {
            for (auto *next = user->getNextNode(); next; next = next->getNextNode())
            {
                if (auto storeOp = mlir::dyn_cast<mlir_ts::StoreOp>(next); storeOp && storeOp.getValue() == value)
                {
                    auto varOp = storeOp.getReference().getDefiningOp<mlir_ts::VariableOp>();
                    return varOp && borrowsOnly(varOp) ? varOp : mlir_ts::VariableOp();
                }

                if (!mlir::isa<mlir_ts::ReleaseSlotOp>(next))
                {
                    break;
                }
            }
        }

        return {};
    }

    mlir_ts::VariableOp OwnershipInferencePass::owningLocalReadOf(mlir::Value value)
    {
        auto root = rootOf(value);
        auto loadOp = root.getDefiningOp<mlir_ts::LoadOp>();
        auto slot = loadOp ? loadOp.getReference().getDefiningOp<mlir_ts::VariableOp>() : mlir_ts::VariableOp();
        auto index = -1;
        if (!slot || !isOwningVariable(slot) || slot.getCaptured().value_or(false) || isParameterSlot(slot, index) ||
            slot->hasAttr(OWN_LET_HOLDER_ATTR_NAME))
        {
            return {};
        }

        return slot;
    }

    bool OwnershipInferencePass::readsOwningLocals(mlir::Value value, mlir_ts::VariableOp self)
    {
        if (auto owner = owningLocalReadOf(value))
        {
            return owner != self;
        }

        // `qq = r && q`: a merge of reads of owning locals (and of null)
        auto argument = mlir::dyn_cast<mlir::BlockArgument>(rootOf(value));
        llvm::SmallVector<mlir_ts::LoadOp> loads;
        if (!argument || !slotLoadsOfMerge(argument, loads) || loads.empty())
        {
            return false;
        }

        return llvm::all_of(loads, [&](mlir_ts::LoadOp load) {
            auto owner = owningLocalReadOf(load.getResult());
            return owner && owner != self;
        });
    }

    void OwnershipInferencePass::classifyLocalBorrowingLets()
    {
        localBorrowingLets.clear();
        llvm::SmallVector<mlir_ts::VariableOp> candidates;
        getFunction()->walk([&](mlir_ts::VariableOp varOp) {
            if (!varOp.getInitializer() || !isOwningVariable(varOp) || varOp.getCaptured().value_or(false))
            {
                return;
            }

            auto local = false;
            auto borrowed = [&](mlir::Value value) {
                if (letBorrows(value))
                {
                    return true;
                }

                if (!readsOwningLocals(value, varOp))
                {
                    return false;
                }

                local = true;
                return true;
            };

            if (!borrowed(varOp.getInitializer()))
            {
                return;
            }

            auto assigned = false;
            for (auto *user : varOp.getResult().getUsers())
            {
                if (isAssignmentOf(user, varOp))
                {
                    if (!borrowed(mlir::cast<mlir_ts::StoreOp>(user).getValue()))
                    {
                        return;
                    }

                    assigned = true;
                }
            }

            if (local && assigned)
            {
                candidates.push_back(varOp);
            }
        });

        // ...and nothing read out of it is kept: stored, returned, captured (else it stays an owner,
        // and its values move into it as before)
        for (auto varOp : candidates)
        {
            localBorrowingLets.insert(varOp);
            auto keeps = quietly([&] {
                llvm::SmallVector<BorrowUse> uses;
                llvm::SmallVector<mlir::Operation *> bookkeeping;
                mlir::Operation *declaration = varOp;
                return collectBorrowerUses(varOp, declaration, uses, bookkeeping) != nullptr;
            });
            if (keeps)
            {
                localBorrowingLets.erase(varOp);
            }        }
    }

    mlir_ts::VariableOp OwnershipInferencePass::localBorrowerOf(mlir::Operation *acquisition, mlir::Operation *&from)
    {
        if (auto retainSlotOp = mlir::dyn_cast<mlir_ts::RetainSlotOp>(acquisition))
        {
            auto varOp = retainSlotOp.getSlot().getDefiningOp<mlir_ts::VariableOp>();
            if (varOp && localBorrowingLets.contains(varOp))
            {
                from = varOp;
                return varOp;
            }

            return {};
        }

        auto retainOp = mlir::dyn_cast<mlir_ts::RetainOp>(acquisition);
        if (!retainOp)
        {
            return {};
        }

        // rc's reference for an assignment: right before the store, past the release of the old value
        for (auto *next = retainOp->getNextNode(); next; next = next->getNextNode())
        {
            if (auto storeOp = mlir::dyn_cast<mlir_ts::StoreOp>(next); storeOp && storeOp.getValue() == retainOp->getOperand(0))
            {
                auto varOp = storeOp.getReference().getDefiningOp<mlir_ts::VariableOp>();
                if (varOp && localBorrowingLets.contains(varOp))
                {
                    from = storeOp;
                    return varOp;
                }

                return {};
            }

            if (!mlir::isa<mlir_ts::ReleaseSlotOp>(next))
            {
                break;
            }
        }

        return {};
    }

    mlir::Operation *OwnershipInferencePass::collectBorrowerUses(mlir_ts::VariableOp borrower, llvm::ArrayRef<mlir::Operation *> givenKills,
                                         llvm::SmallVectorImpl<BorrowUse> &uses,
                                         llvm::SmallVectorImpl<mlir::Operation *> &bookkeeping, mlir::Operation *from)
    {
        // a `let` given only borrows (TSLANG 56): each borrow lasts until the next assignment, and its
        // reads are those its own store (`from`: the declaration or an assignment) reaches before one
        llvm::SmallVector<mlir::Operation *, 4> kills(givenKills.begin(), givenKills.end());
        llvm::SmallVector<mlir::Operation *, 4> assignments;
        auto onlyBorrows = borrowsOnly(borrower);
        if (onlyBorrows)
        {
            // its declaration gives it its initializer again, as an assignment gives it a value: what
            // it held before, in an earlier round of a loop around the declaration, it holds no more
            kills.push_back(borrower);
            for (auto *user : borrower.getResult().getUsers())
            {
                if (isAssignmentOf(user, borrower))
                {
                    kills.push_back(user);
                    if (user != from)
                    {
                        assignments.push_back(user);
                    }
                }
            }
        }

        for (auto *user : borrower.getResult().getUsers())
        {
            if (onlyBorrows && isAssignmentOf(user, borrower))
            {
                continue;
            }

            if (onlyBorrows && from && mlir::isa<mlir_ts::LoadOp>(user) && !reachableAfter(from, user, assignments))
            {
                continue;
            }

            if (mlir::isa<mlir_ts::RetainSlotOp, mlir_ts::ReleaseSlotOp>(user))
            {
                bookkeeping.push_back(user);
                continue;
            }

            // the store that declares a local declared in a `try` (borrowerOf): its value, not an assignment
            if (!borrower.getInitializer() && declarationRetainOf(user))
            {
                continue;
            }

            // `--di`'s record of the local for the debugger names it and keeps nothing
            if (mlir::isa<mlir_ts::DebugVariableOp>(user))
            {
                continue;
            }

            // `const model = this.model; model.push(0.0)` (TSLANG 62): the array changed in place,
            // through the slot that holds its header - the borrowed array, which the field sees. A
            // null header (an unset field) gets one in the local's slot only, which nothing frees:
            // that is `undefined` in TypeScript terms, where `.push` throws (array spec, ruling T4-D)
            if (mlir::isa<mlir_ts::ArrayPushOp, mlir_ts::ArrayPopOp, mlir_ts::ArrayUnshiftOp, mlir_ts::ArrayShiftOp,
                          mlir_ts::ArraySpliceOp>(user) &&
                user->getOperand(0) == borrower.getResult())
            {
                uses.push_back({user, {kills.begin(), kills.end()}});
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

    void OwnershipInferencePass::splitBorrowingLets()
    {
        // `let dir = this.dir; if (turn) dir = rotate(dir); this.dir = normalize(dir);` (TSLANG 66): each
        // fresh value the `let` is given goes into a holder, an owning slot of its own released where the
        // `let` is, and the `let` borrows it from the holder - as it borrows the field
        for (auto *candidate : splitLetCandidates)
        {
            auto varOp = mlir::cast<mlir_ts::VariableOp>(candidate);
            auto slot = varOp.getResult();
            if (!varOp.getInitializer() || !letBorrows(varOp.getInitializer()) || varOp.getCaptured().value_or(false))
            {
                continue;
            }

            llvm::SmallVector<mlir_ts::StoreOp> fresh;
            llvm::SmallVector<mlir::Operation *> exits;
            auto splits = true;
            for (auto *user : slot.getUsers())
            {
                if (mlir::isa<mlir_ts::LoadOp, mlir_ts::RetainSlotOp, mlir_ts::DebugVariableOp>(user))
                {
                    continue;
                }

                // the release of its old value before an assignment, or of its last at a way out
                if (mlir::isa<mlir_ts::ReleaseSlotOp>(user))
                {
                    auto next = mlir::dyn_cast_or_null<mlir_ts::StoreOp>(user->getNextNode());
                    if (!next || next.getReference() != slot)
                    {
                        exits.push_back(user);
                    }

                    continue;
                }

                auto storeOp = mlir::dyn_cast<mlir_ts::StoreOp>(user);
                if (!storeOp || storeOp.getReference() != slot)
                {
                    splits = false;
                    break;
                }

                if (letBorrows(storeOp.getValue()))
                {
                    continue;
                }

                auto release = mlir::dyn_cast_or_null<mlir_ts::ReleaseSlotOp>(storeOp->getPrevNode());
                if (!release || release->getOperand(0) != slot)
                {
                    splits = false;
                    break;
                }

                fresh.push_back(storeOp);
            }

            if (!splits || fresh.empty())
            {
                continue;
            }

            auto type = mlir::cast<mlir_ts::RefType>(slot.getType()).getElementType();
            mlir::OpBuilder builder(varOp);
            builder.setInsertionPointAfter(varOp);
            auto empty = builder.create<mlir_ts::DefaultOp>(varOp.getLoc(), type);
            auto holder = builder.create<mlir_ts::VariableOp>(varOp.getLoc(), slot.getType(), empty, varOp.getCapturedAttr(),
                                                              mlir::IntegerAttr());
            holder->setAttr(OWNED_LOCAL_ATTR_NAME, builder.getUnitAttr());
            holder->setAttr(OWN_LET_HOLDER_ATTR_NAME, builder.getUnitAttr());
            for (auto storeOp : fresh)
            {
                storeOp->getPrevNode()->setOperand(0, holder.getResult());
                storeOp.getReferenceMutable().assign(holder.getResult());
                builder.setInsertionPointAfter(storeOp);
                auto read = builder.create<mlir_ts::LoadOp>(storeOp.getLoc(), type, holder.getResult());
                builder.create<mlir_ts::StoreOp>(storeOp.getLoc(), read, slot);
            }

            for (auto *exit : exits)
            {
                builder.setInsertionPointAfter(exit);
                builder.create<mlir_ts::ReleaseSlotOp>(exit->getLoc(), holder.getResult());
            }
        }
    }

    void OwnershipInferencePass::checkLetHolders(llvm::SetVector<mlir::Operation *> &toErase)
    {
        getFunction()->walk([&](mlir_ts::VariableOp holder) {
            if (!holder->hasAttr(OWN_LET_HOLDER_ATTR_NAME))
            {
                return;
            }

            auto ends = slotEnds(holder.getResult());
            for (auto *user : holder.getResult().getUsers())
            {
                auto readOp = mlir::dyn_cast<mlir_ts::LoadOp>(user);
                if (!readOp || !readOp->hasOneUse())
                {
                    continue;
                }

                // the `let` given the holder's value: its reads up to its next assignment borrow it
                auto storeOp = mlir::dyn_cast<mlir_ts::StoreOp>(*readOp->getUsers().begin());
                auto borrower = storeOp ? storeOp.getReference().getDefiningOp<mlir_ts::VariableOp>() : mlir_ts::VariableOp();
                if (!borrower || storeOp.getValue() != readOp.getResult())
                {
                    continue;
                }

                auto name = varName(borrower);
                llvm::SmallVector<BorrowUse> uses;
                llvm::SmallVector<mlir::Operation *> bookkeeping;
                // (a `let` declared in a loop: its declaration starts it afresh)
                mlir::Operation *declaration = borrower;
                if (auto *kept = collectBorrowerUses(borrower, declaration, uses, bookkeeping, storeOp))
                {
                    if (isAssignmentOf(kept, borrower))
                    {
                        reportBorrowerAssigned(kept, name, name);
                    }
                    else
                    {
                        reportBorrowEscapes(kept, name, name);
                    }

                    return;
                }

                for (auto *end : ends)
                {
                    for (auto &use : uses)
                    {
                        if (reachableAfter(end, use.op, use.kills))
                        {
                            reportBorrowOutlives(use.op, end, name, name);
                            return;
                        }
                    }
                }

                toErase.insert(bookkeeping.begin(), bookkeeping.end());
            }
        });
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

        // a tuple made of it only to be given to calls that keep none of it (isBorrow): the calls have
        // it (an object literal's method, bound to the literal, read into the object type a parameter
        // is - `applyT(p)` of a literal whose arrows capture, spec 66). Any other tuple keeps it.
        if (mlir::isa<mlir_ts::CreateTupleOp>(user) && isBorrow(user, use.get()))
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

                // the store of a value taken out of the object its place holds (`this.head =
                // this.head.next`): the value is the place's from then on, which checkPlaceRead decides
                // for the read of it. No use of the object, which the store's release frees.
                if (splicedTakers.contains(user))
                {
                    continue;
                }

                uses.push_back({user, pending.kills, isCall(user) && isCalleeOperand(use)});
                // a move out of a record's field, its retain and its taker, which checkPlaceRead
                // decides (spec 48): a use of the borrow until it is taken
                if (takenOps.contains(user))
                {
                    continue;
                }

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

                // an element of an array literal that only borrows (TSLANG 55): whatever uses the
                // array reads it - and nothing else: its release leaves the element to its owner
                if (auto found = arrayElementUses.find(user); found != arrayElementUses.end())
                {
                    for (auto &arrayUse : found->second)
                    {
                        uses.push_back({arrayUse.op, pending.kills, arrayUse.callee});
                    }

                    continue;
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

                    // `u instanceof A` of a borrowed union (spec 61): the object as an opaque pointer,
                    // whose vtable is read for its class's `.instanceOf` - a test that keeps nothing
                    // (isBorrow), not a view whose vtable read would keep it (TSLANG 46)
                    if (auto castOp = mlir::dyn_cast<mlir_ts::CastOp>(user);
                        castOp && mlir::isa<mlir_ts::ClassType>(castOp.getIn().getType()) &&
                        mlir::isa<mlir_ts::OpaqueType>(castOp.getType()) && isBorrow(user, pending.value))
                    {
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

                // the function value a call goes through: what it returns is the caller's, not a part
                // of the function (`a.fn()` of a field `fn: () => string`, a string kept past `a`)
                if (isCall(user) && use.getOperandNumber() == 0 && calleeValue(user) == pending.value)
                {
                    continue;
                }

                // A WeakRef made of the borrow, or the object a borrowed WeakRef refers to, is no part
                // of it: the WeakRef finds the object gone, and the object is checked as its own
                // borrow (spec 84)
                if (mlir::isa<mlir_ts::WeakRefNewOp, mlir_ts::WeakRefDerefOp>(user))
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
