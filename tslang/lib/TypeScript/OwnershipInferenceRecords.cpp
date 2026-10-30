// OwnershipInferencePass (see OwnershipInferencePass.h): borrowing records (spec 32).

#include "OwnershipInferencePass.h"

#define DEBUG_TYPE "own"

namespace own_inference
{

    bool OwnershipInferencePass::mayPointInto(mlir::Value value)
    {
        auto type = value.getType();
        return mlir::isa<mlir_ts::RefType, mlir_ts::BoundRefType, mlir_ts::ValueRefType, mlir_ts::BoundFunctionType,
                         mlir_ts::HybridFunctionType, mlir_ts::ExtensionFunctionType, mlir_ts::OpaqueType>(type) ||
               ownsHeap(value);
    }

    OwnershipInferencePass::BorrowingRecord *OwnershipInferencePass::borrowingRecordOf(mlir_ts::VariableOp varOp)
    {
        for (auto &record : records)
        {
            if (record->local == varOp)
            {
                return record.get();
            }
        }

        return nullptr;
    }

    void OwnershipInferencePass::classifyRecords()
    {
        getFunction().walk([&](mlir_ts::NewOp newOp) {
            auto fields = recordBorrowedFields(newOp);
            auto varOp = fields.empty() ? mlir_ts::VariableOp() : recordLocalOf(newOp.getResult());
            if (!varOp)
            {
                return;
            }

            auto record = std::make_shared<BorrowingRecord>();
            record->box = newOp;
            record->local = varOp;
            for (auto *user : varOp->getUsers())
            {
                auto propertyRefOp = mlir::dyn_cast<mlir_ts::PropertyRefOp>(user);
                if (!propertyRefOp)
                {
                    continue;
                }

                auto borrows = llvm::is_contained(fields, static_cast<int32_t>(propertyRefOp.getPosition()));
                for (auto *fieldUser : propertyRefOp->getUsers())
                {
                    auto storeOp = mlir::dyn_cast<mlir_ts::StoreOp>(fieldUser);
                    if (!storeOp || storeOp.getReference() != propertyRefOp.getResult())
                    {
                        continue;
                    }

                    // the `.captured` box: what it borrows is the literal's box's (classifyLiteralCaptures)
                    if (auto boxOp = storeOp.getValue().getDefiningOp<mlir_ts::VariableOp>();
                        borrows && boxOp && boxOp->hasAttr(CAPTURE_BOX_ATTR_NAME))
                    {
                        continue;
                    }

                    auto source = recordFieldSource(storeOp.getValue());
                    if (source == RecordFieldSource::NoBlock)
                    {
                        continue;
                    }

                    if (source == RecordFieldSource::Unknown || borrows != (source != RecordFieldSource::Fresh))
                    {
                        return;
                    }

                    if (borrows)
                    {
                        record->borrowed.push_back({storeOp, source});
                    }
                }
            }

            for (auto *user : newOp->getUsers())
            {
                if (auto castOp = mlir::dyn_cast<mlir_ts::CastOp>(user))
                {
                    for (auto *castUser : castOp->getUsers())
                    {
                        if (auto newInterfaceOp = mlir::dyn_cast<mlir_ts::NewInterfaceOp>(castUser))
                        {
                            record->interfaces.push_back(newInterfaceOp.getResult());
                        }
                    }
                }
            }

            // a path back to a use through the record's making is another record
            llvm::SmallVector<mlir::Operation *, 2> kills{newOp.getOperation(), varOp.getOperation()};
            walkRecord(record->interfaces, kills, record->walk);
            for (auto &field : record->borrowed)
            {
                setAttrTracked(field.store, OWN_CAPTURE_BORROW_ATTR_NAME);
                recordStoreUses[field.store.getOperation()] = record.get();
            }

            records.push_back(std::move(record));
        });

        // an unboxed record (spec 36): the tuple read out of the local it is built in is the record, and
        // every field of it that holds a block borrows
        getFunction().walk([&](mlir_ts::VariableOp varOp) {
            auto fields = unboxedRecordFields(varOp);
            if (fields.empty() || !isConstantRecordLocal(varOp))
            {
                return;
            }

            auto record = std::make_shared<BorrowingRecord>();
            record->local = varOp;
            llvm::SmallVector<mlir::Value> reads;
            for (auto *user : varOp->getUsers())
            {
                if (auto loadOp = mlir::dyn_cast<mlir_ts::LoadOp>(user))
                {
                    reads.push_back(loadOp.getResult());
                    continue;
                }

                auto propertyRefOp = mlir::dyn_cast<mlir_ts::PropertyRefOp>(user);
                if (!propertyRefOp)
                {
                    continue;
                }

                auto borrows = llvm::is_contained(fields, static_cast<int32_t>(propertyRefOp.getPosition()));
                for (auto *fieldUser : propertyRefOp->getUsers())
                {
                    auto storeOp = mlir::dyn_cast<mlir_ts::StoreOp>(fieldUser);
                    if (!storeOp || storeOp.getReference() != propertyRefOp.getResult())
                    {
                        continue;
                    }

                    auto source = recordFieldSource(storeOp.getValue());
                    if (source == RecordFieldSource::NoBlock)
                    {
                        continue;
                    }

                    // it owns nothing: a fresh value is a temporary's it borrows (the signature pass
                    // marked only those), and a field it would own is not what it was marked for
                    if (source == RecordFieldSource::Unknown || !borrows ||
                        (source == RecordFieldSource::Fresh && !varOp->hasAttr(OWN_RECORD_TEMPORARY_ATTR_NAME)))
                    {
                        return;
                    }

                    record->borrowed.push_back({storeOp, source});
                }
            }

            record->reads = reads;
            llvm::SmallVector<mlir::Operation *, 1> kills{varOp.getOperation()};
            walkRecord(reads, kills, record->walk);
            for (auto &field : record->borrowed)
            {
                setAttrTracked(field.store, OWN_CAPTURE_BORROW_ATTR_NAME);
                recordStoreUses[field.store.getOperation()] = record.get();
            }

            records.push_back(std::move(record));
        });
    }

    OwnershipInferencePass::UnboxedFlow OwnershipInferencePass::unboxedRecordFlow(mlir::Value value)
    {
        llvm::SmallVector<mlir::Value> work{value};
        llvm::SmallPtrSet<mlir::Value, 8> seen;
        llvm::SmallPtrSet<mlir::Operation *, 4> slots;
        auto unboxed = false;
        auto other = false;
        while (!work.empty())
        {
            auto next = work.pop_back_val();
            auto current = rootOf(next);
            if (!seen.insert(current).second || holdsNoBlock(next) || holdsNoBlock(current))
            {
                continue;
            }

            auto *def = current.getDefiningOp();
            if (def && isCall(def))
            {
                (isUnboxedRecordResult(def) ? unboxed : other) = true;
                continue;
            }

            if (auto argument = mlir::dyn_cast<mlir::BlockArgument>(current))
            {
                llvm::SmallVector<mlir::Value> merged;
                if (mergedInto(argument, merged))
                {
                    work.append(merged.begin(), merged.end());
                    continue;
                }

                other = true;
                continue;
            }

            auto loadOp = mlir::dyn_cast_or_null<mlir_ts::LoadOp>(def);
            auto varOp = loadOp ? slotVariableOf(loadOp.getReference()) : mlir_ts::VariableOp();
            if (varOp && !unboxedRecordFields(varOp).empty())
            {
                unboxed = true;
                continue;
            }

            // what an owning local holds: its initializer and every value stored into it
            if (varOp && isOwningVariable(varOp) && !varOp.getCaptured().value_or(false))
            {
                if (!slots.insert(varOp).second)
                {
                    continue;
                }

                if (varOp.getInitializer())
                {
                    work.push_back(varOp.getInitializer());
                }

                for (auto stored : valuesStoredInto(varOp))
                {
                    work.push_back(stored);
                }

                continue;
            }

            other = true;
        }

        if (!unboxed)
        {
            return UnboxedFlow::None;
        }

        return other ? UnboxedFlow::Mixed : UnboxedFlow::Unboxed;
    }

    OwnershipInferencePass::UnboxedFlow OwnershipInferencePass::unboxedRecordSlotFlow(mlir_ts::VariableOp varOp)
    {
        if (!isOwningVariable(varOp) || varOp.getCaptured().value_or(false))
        {
            return UnboxedFlow::None;
        }

        // a read of the local, as unboxedRecordFlow follows one; a local never read is asked through a
        // load made for the question
        for (auto *user : varOp->getUsers())
        {
            if (auto loadOp = mlir::dyn_cast<mlir_ts::LoadOp>(user))
            {
                return unboxedRecordFlow(loadOp.getResult());
            }
        }

        llvm::SmallVector<mlir::Value> values;
        if (varOp.getInitializer())
        {
            values.push_back(varOp.getInitializer());
        }

        for (auto stored : valuesStoredInto(varOp))
        {
            values.push_back(stored);
        }

        auto flow = UnboxedFlow::None;
        auto other = false;
        for (auto stored : values)
        {
            switch (unboxedRecordFlow(stored))
            {
            case UnboxedFlow::Mixed:
                return UnboxedFlow::Mixed;
            case UnboxedFlow::Unboxed:
                flow = UnboxedFlow::Unboxed;
                break;
            case UnboxedFlow::None:
                other = other || !holdsNoBlock(stored);
                break;
            }
        }

        return flow == UnboxedFlow::Unboxed && other ? UnboxedFlow::Mixed : flow;
    }

    void OwnershipInferencePass::dropUnboxedRecordCounts(llvm::SetVector<mlir::Operation *> &toErase)
    {
        llvm::SmallPtrSet<mlir::Operation *, 4> reported;
        auto mixed = [&](mlir::Operation *op, mlir::Operation *at) {
            if (reported.insert(at).second && reporting())
            {
                op->emitError("a value that borrows what it holds and owns nothing (a tuple that holds a borrow) meets "
                              "a value that owns a block here; -mm=own cannot release them alike");
                signalPassFailure();
            }
        };

        getFunction().walk([&](mlir::Operation *op) {
            // a handle's counts are rc's (spec 23.2)
            if (touchesHandle(op))
            {
                return;
            }

            if (mlir::isa<mlir_ts::RetainOp, mlir_ts::ReleaseOp>(op))
            {
                switch (unboxedRecordFlow(op->getOperand(0)))
                {
                case UnboxedFlow::Unboxed:
                    toErase.insert(op);
                    break;
                case UnboxedFlow::Mixed:
                    mixed(op, rootOf(op->getOperand(0)).getDefiningOp() ? rootOf(op->getOperand(0)).getDefiningOp() : op);
                    break;
                case UnboxedFlow::None:
                    break;
                }

                return;
            }

            mlir::Value slot;
            if (auto retainSlotOp = mlir::dyn_cast<mlir_ts::RetainSlotOp>(op))
            {
                slot = retainSlotOp.getSlot();
            }
            else if (auto releaseSlotOp = mlir::dyn_cast<mlir_ts::ReleaseSlotOp>(op))
            {
                slot = releaseSlotOp.getSlot();
            }

            auto varOp = slot ? slotVariableOf(slot) : mlir_ts::VariableOp();
            if (!varOp)
            {
                return;
            }

            switch (unboxedRecordSlotFlow(varOp))
            {
            case UnboxedFlow::Unboxed:
                toErase.insert(op);
                break;
            case UnboxedFlow::Mixed:
                mixed(op, varOp);
                break;
            case UnboxedFlow::None:
                break;
            }
        });
    }

    void OwnershipInferencePass::walkRecord(llvm::ArrayRef<mlir::Value> starts, llvm::ArrayRef<mlir::Operation *> kills, RecordWalk &walk)
    {
        struct Pending
        {
            mlir::Value value;
            bool record;
        };

        llvm::SmallVector<Pending> work;
        llvm::SmallPtrSet<mlir::Value, 16> seen;
        auto push = [&](mlir::Value value, bool record) {
            if ((record || mayPointInto(value)) && seen.insert(value).second)
            {
                work.push_back({value, record});
            }
        };

        auto keep = [&](mlir::Operation *user) { walk.kept = walk.kept ? walk.kept : user; };

        auto followSlot = [&](mlir_ts::VariableOp varOp, bool record) {
            // the local, and the local seen as its storage tuple (a `@value` local)
            llvm::SmallVector<mlir::Value, 2> refs{varOp.getResult()};
            for (auto *slotUser : varOp->getUsers())
            {
                if (isValueStorageRefCast(slotUser))
                {
                    refs.push_back(slotUser->getResult(0));
                }
            }

            llvm::SmallVector<mlir::Operation *> slotUsers;
            for (auto ref : refs)
            {
                slotUsers.append(ref.getUsers().begin(), ref.getUsers().end());
            }

            for (auto *slotUser : slotUsers)
            {
                // its releases and its other values are the slot's own
                if (auto loadOp = mlir::dyn_cast<mlir_ts::LoadOp>(slotUser))
                {
                    push(loadOp.getResult(), record);
                    continue;
                }

                // a field of a record that is a tuple, read in place out of the local that holds it
                // (spec 36): a read of the record, and what it reads is what the record borrows. A
                // write of the field would release what the record borrows.
                auto propertyRefOp = mlir::dyn_cast<mlir_ts::PropertyRefOp>(slotUser);
                if (!record || !propertyRefOp || !llvm::is_contained(refs, propertyRefOp.getObjectRef()))
                {
                    continue;
                }

                walk.uses.push_back({propertyRefOp, {kills.begin(), kills.end()}});
                for (auto *fieldUser : propertyRefOp->getUsers())
                {
                    walk.uses.push_back({fieldUser, {kills.begin(), kills.end()}});
                    if (auto fieldLoad = mlir::dyn_cast<mlir_ts::LoadOp>(fieldUser))
                    {
                        push(fieldLoad.getResult(), mayPointInto(fieldLoad.getResult()));
                    }
                    else if (mlir::isa<mlir_ts::StoreOp, mlir_ts::ReleaseSlotOp>(fieldUser))
                    {
                        keep(fieldUser);
                    }
                    else
                    {
                        for (auto result : fieldUser->getResults())
                        {
                            push(result, false);
                        }
                    }
                }
            }
        };

        for (auto start : starts)
        {
            push(start, true);
        }

        while (!work.empty())
        {
            auto pending = work.pop_back_val();
            auto value = pending.value;
            for (auto &use : value.getUses())
            {
                auto *user = use.getOwner();
                if (mlir::isa<mlir_ts::DebugVariableOp>(user) ||
                    (pending.record && mlir::isa<mlir_ts::RetainOp, mlir_ts::ReleaseOp>(user)))
                {
                    continue;
                }

                // the same value: a view, a cast, a merge
                if ((isView(user) && use.getOperandNumber() == 0) || mlir::isa<mlir_ts::CastOp>(user))
                {
                    push(user->getResult(0), pending.record);
                    continue;
                }

                // (not a call in a `try`, a `ts.Invoke`: a branch too, but its operands are the call's)
                if (auto branchOp = mlir::dyn_cast<mlir::BranchOpInterface>(user); branchOp && !isCall(user))
                {
                    if (auto argument = branchOp.getSuccessorBlockArgument(use.getOperandNumber()))
                    {
                        push(*argument, pending.record);
                    }

                    continue;
                }

                // a local that holds it: its reads
                auto varOp = mlir::dyn_cast<mlir_ts::VariableOp>(user);
                auto storeOp = mlir::dyn_cast<mlir_ts::StoreOp>(user);
                if (varOp || (storeOp && storeOp.getValue() == value))
                {
                    auto slotOp = varOp ? varOp : storeOp.getReference().getDefiningOp<mlir_ts::VariableOp>();
                    if (slotOp && !slotOp.getCaptured().value_or(false))
                    {
                        followSlot(slotOp, pending.record);
                        continue;
                    }

                    // a place, a global, a captured variable: what is read out of the record is checked
                    // where it is read (checkPlaceRead)
                    walk.uses.push_back({user, {kills.begin(), kills.end()}});
                    if (pending.record)
                    {
                        keep(user);
                    }

                    continue;
                }

                walk.uses.push_back({user, {kills.begin(), kills.end()}, isCall(user) && isCalleeOperand(use)});
                if (mlir::isa<mlir_ts::ReturnInternalOp, mlir_ts::ReturnValOp>(user))
                {
                    if (pending.record)
                    {
                        walk.returns.push_back(user);
                    }

                    continue;
                }

                if (isCall(user))
                {
                    if (pending.record && isKeptArgument(user, value))
                    {
                        keep(user);
                        continue;
                    }

                    // a callee that may overwrite a field of what it is given (`p.v = ...`) gives back
                    // the old value, which may be one the record borrows - as a write of the field here
                    // would
                    if (pending.record && !isCalleeOperand(use) && !callNoDrops(user))
                    {
                        auto places = callDropFields(user);
                        if (places && llvm::any_of(*places, [](auto &place) {
                                return place.first >= 0 || place.first == OWN_INTERFACE_FIELD_POSITION;
                            }))
                        {
                            keep(user);
                            continue;
                        }
                    }

                    // a result that borrows it - a plain borrow (`id(r)`), or a borrowing record that
                    // holds it - points where it points
                    auto args = callArgs(user);
                    auto given = [&](llvm::ArrayRef<int> indices) {
                        return llvm::any_of(indices, [&](int index) {
                            return static_cast<size_t>(index) < args.size() && args[index] == value;
                        });
                    };

                    if (user->getNumResults() == 1 && given(resultBorrowsAll(user)))
                    {
                        push(user->getResult(0), pending.record || isRecordResult(user));
                    }

                    // ...and a generator's state that borrows it (OWN_RESULT_BOUNDED_ATTR_NAME)
                    if (auto bounded = user->getAttrOfType<mlir::DenseI32ArrayAttr>(OWN_RESULT_BOUNDED_ATTR_NAME);
                        bounded && user->getNumResults() == 1 &&
                        given(llvm::SmallVector<int>(bounded.asArrayRef().begin(), bounded.asArrayRef().end())))
                    {
                        push(user->getResult(0), false);
                    }

                    continue;
                }

                // a field read out of a record that is a tuple (spec 36): what it borrows, or nothing,
                // and its keeps are the record's
                if (pending.record && mlir::isa<mlir_ts::ExtractPropertyOp>(user))
                {
                    push(user->getResult(0), mayPointInto(user->getResult(0)));
                    continue;
                }

                if (pending.record && !isBorrow(user, value))
                {
                    keep(user);
                    continue;
                }

                // a field written through the record: its old value may be one the record borrows,
                // which the overwrite would release, and the field may be one its release leaves
                if (pending.record && mlir::isa<mlir_ts::InterfaceSymbolRefOp, mlir_ts::PropertyRefOp>(user))
                {
                    for (auto *fieldUser : user->getResult(0).getUsers())
                    {
                        auto fieldStore = mlir::dyn_cast<mlir_ts::StoreOp>(fieldUser);
                        if (mlir::isa<mlir_ts::ReleaseSlotOp>(fieldUser) ||
                            (fieldStore && fieldStore.getReference() == user->getResult(0)))
                        {
                            keep(fieldUser);
                        }
                    }
                }

                // what it produces - a field's reference, a value read out of it, a bound method - may
                // still point into what the record borrows
                for (auto result : user->getResults())
                {
                    push(result, false);
                }
            }
        }
    }

    llvm::StringRef OwnershipInferencePass::recordName(mlir::Value value)
    {
        llvm::StringRef name = ownerName(value);
        forEachUse(value, [&](mlir::Operation *user, mlir::Value used) {
            if (auto varOp = mlir::dyn_cast<mlir_ts::VariableOp>(user); varOp && varOp.getInitializer() == used)
            {
                name = varName(varOp);
            }
        });

        return name;
    }

    std::string OwnershipInferencePass::describeRecordSource(BorrowedField field)
    {
        auto value = field.store.getValue();
        auto quoted = [](llvm::StringRef name) { return "'" + name.str() + "'"; };
        switch (field.source)
        {
        case RecordFieldSource::Param:
            return quoted(paramName(borrowedParam(value)));
        case RecordFieldSource::Place:
            if (auto loadOp = rootOf(value).getDefiningOp<mlir_ts::LoadOp>(); loadOp && isPlace(loadOp.getReference()))
            {
                return describePlace(loadOp.getReference());
            }

            return describeCall(rootOf(value).getDefiningOp());
        case RecordFieldSource::Slot:
            return quoted(varName(slotLoadOf(value).getReference().getDefiningOp<mlir_ts::VariableOp>()));
        case RecordFieldSource::Cell:
            return "a captured variable";
        case RecordFieldSource::Fresh:
            return "a temporary";
        default:
            return "a value it does not own";
        }
    }

    void OwnershipInferencePass::checkBorrowingRecords()
    {
        for (auto &record : records)
        {
            auto name = recordName(!record->interfaces.empty() ? record->interfaces.front()
                                   : record->box                ? record->box.getResult()
                                   : !record->reads.empty()     ? record->reads.front()
                                                                : record->local.getResult());
            if (record->walk.kept)
            {
                reportPlaceBorrowEscapes(record->walk.kept, name,
                                         record->borrowed.empty() ? "a value it does not own"
                                                                  : describeRecordSource(record->borrowed.front()));
                continue;
            }

            auto ok = true;
            for (auto *returned : record->walk.returns)
            {
                for (auto &field : record->borrowed)
                {
                    if (!ok || field.source == RecordFieldSource::Immortal)
                    {
                        continue;
                    }

                    auto param = field.source == RecordFieldSource::Param || field.source == RecordFieldSource::Place
                                     ? borrowedParam(field.store.getValue())
                                     : -1;
                    if (param < 0 || !llvm::is_contained(returnsRecordOf, param))
                    {
                        reportPlaceBorrowEscapes(returned, name, describeRecordSource(field));
                        ok = false;
                    }
                }
            }

            for (auto &field : record->borrowed)
            {
                if (!ok)
                {
                    break;
                }

                auto value = field.store.getValue();
                Chain chain;
                if (field.source == RecordFieldSource::Slot)
                {
                    chain.roots.push_back({slotLoadOf(value).getReference(), Chain::Slot});
                }
                else if (auto load = field.source == RecordFieldSource::Param ? keptParamLoadOf(value) : mlir_ts::LoadOp())
                {
                    // a parameter this function keeps is its own to move away, as a local's value is
                    chain.roots.push_back({load.getReference(), Chain::Slot});
                }
                else if (field.source == RecordFieldSource::Cell)
                {
                    chain.roots.push_back({rootOf(value), Chain::NotOwned});
                }
                else if (field.source == RecordFieldSource::Fresh)
                {
                    // a temporary's fresh value an unboxed record borrows (spec 36): its own release
                    chain.roots.push_back({rootOf(value), Chain::Owned});
                }
                else
                {
                    continue;
                }

                auto ends = rootEnds(chain);
                if (field.source == RecordFieldSource::Cell)
                {
                    for (auto *drop : drops)
                    {
                        if (dropsChain(drop, chain, nullptr, value))
                        {
                            ends.push_back(drop);
                        }
                    }
                }

                for (auto *end : ends)
                {
                    for (auto &use : record->walk.uses)
                    {
                        if (ok && (use.op != end || !use.callee) && reachableAfter(end, use.op, use.kills))
                        {
                            if (field.source == RecordFieldSource::Slot)
                            {
                                reportBorrowOutlives(use.op, end, name, varName(chain.roots.front().first.getDefiningOp<mlir_ts::VariableOp>()));
                            }
                            else
                            {
                                reportPlaceBorrowOutlives(use.op, end, name, describeRecordSource(field));
                            }

                            ok = false;
                        }
                    }
                }
            }
        }
    }

    void OwnershipInferencePass::checkRecordResults()
    {
        getFunction().walk([&](mlir::Operation *call) {
            if (!isCall(call) || !isRecordResult(call) || call->getNumResults() != 1)
            {
                return;
            }

            auto result = call->getResult(0);
            auto place = describeCall(call);
            auto name = recordName(result);
            RecordWalk walk;
            llvm::SmallVector<mlir::Operation *, 1> kills{call};
            walkRecord({result}, kills, walk);
            if (walk.kept)
            {
                reportPlaceBorrowEscapes(walk.kept, name, place);
                return;
            }

            auto args = callArgs(call);
            for (auto *returned : walk.returns)
            {
                for (auto index : resultBorrowsAll(call))
                {
                    if (static_cast<size_t>(index) >= args.size() || argumentHoldsNoBlock(args[index]) ||
                        isImmortalLiteral(args[index]))
                    {
                        continue;
                    }

                    auto param = borrowedParam(args[index]);
                    if (param < 0 || !llvm::is_contained(returnsRecordOf, param))
                    {
                        reportPlaceBorrowEscapes(returned, name, place);
                        return;
                    }
                }
            }

            auto chain = chainOf(call);
            // rc's retain of a fresh argument right before the call is no end: the box's own check
            // decides it, and the use that takes it is an end of its own
            llvm::SmallVector<mlir::Operation *> ends;
            for (auto *end : rootEnds(chain))
            {
                auto ownedRetain = mlir::isa<mlir_ts::RetainOp>(end) && llvm::any_of(chain.roots, [&](auto &root) {
                                       return root.second == Chain::Owned && rootOf(end->getOperand(0)) == root.first;
                                   });
                if (!ownedRetain)
                {
                    ends.push_back(end);
                }
            }

            for (auto *drop : drops)
            {
                if (drop != call && dropsChain(drop, chain, call))
                {
                    ends.push_back(drop);
                }
            }

            for (auto *end : ends)
            {
                for (auto &use : walk.uses)
                {
                    if ((use.op != end || !use.callee) && reachableAfter(end, use.op, use.kills))
                    {
                        reportPlaceBorrowOutlives(use.op, end, name, place);
                        return;
                    }
                }
            }
        });
    }

} // namespace own_inference
