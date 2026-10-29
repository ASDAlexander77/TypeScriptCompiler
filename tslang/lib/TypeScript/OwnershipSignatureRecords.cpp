// OwnershipSignaturePass (see OwnershipSignaturePass.h): borrowing records (spec 32) - the boxes an
// object literal is made into, and the parameters a record result borrows.

#include "OwnershipSignaturePass.h"

#define DEBUG_TYPE "own"

namespace own_signature
{

    void OwnershipSignaturePass::classifyRecordBoxes()
    {
        recordBoxes.clear();
        getOperation().walk([&](mlir_ts::NewOp newOp) {
            newOp->removeAttr(OWN_RECORD_BORROWED_FIELDS_ATTR_NAME);
            for (auto *user : newOp->getUsers())
            {
                for (auto *viewUser : user->getUsers())
                {
                    if (mlir::isa<mlir_ts::NewInterfaceOp>(viewUser))
                    {
                        viewUser->removeAttr(OWN_RECORD_BORROWED_FIELDS_ATTR_NAME);
                    }
                }
            }

            classifyRecordBox(newOp);
        });

        getOperation().walk([&](mlir_ts::VariableOp varOp) {
            varOp->removeAttr(OWN_RECORD_UNBOXED_FIELDS_ATTR_NAME);
            varOp->removeAttr(OWN_RECORD_TEMPORARY_ATTR_NAME);
            classifyUnboxedRecord(varOp);
        });
    }

    void OwnershipSignaturePass::classifyUnboxedRecord(mlir_ts::VariableOp varOp)
    {
        // an object literal's local whose read is never boxed: the tuple is the value
        if (!isConstantRecordLocal(varOp) || recordBoxOfLocal(varOp))
        {
            return;
        }

        // the record is its read, the whole tuple; a literal whose fields are read out one by one (to be
        // re-made as another tuple) is not one
        auto read = false;
        for (auto *user : varOp->getUsers())
        {
            if (mlir::isa<mlir_ts::LoadOp>(user))
            {
                read = true;
            }
            else if (llvm::any_of(user->getUsers(), [](mlir::Operation *fieldUser) {
                         return mlir::isa<mlir_ts::LoadOp>(fieldUser) && !fieldUser->use_empty();
                     }))
            {
                return;
            }
        }

        if (!read)
        {
            return;
        }

        // a literal no one takes - its read is only given to calls that do not keep it, and read - is
        // a temporary (a call's argument): what it holds keeps its own owner, a fresh value too, whose
        // release (the temporary's) bounds the record
        auto temporary = true;
        for (auto *user : varOp->getUsers())
        {
            auto loadOp = mlir::dyn_cast<mlir_ts::LoadOp>(user);
            if (!loadOp)
            {
                continue;
            }

            forEachUse(loadOp.getResult(), [&](mlir::Operation *readUser, mlir::Value used) {
                if ((!isCall(readUser) && !mlir::isa<mlir_ts::ExtractPropertyOp, mlir_ts::DebugVariableOp>(readUser)) ||
                    keeps(readUser, used))
                {
                    temporary = false;
                }
            });
        }

        RecordBox info;
        info.unboxed = true;
        llvm::SmallVector<int32_t> borrowed;
        auto mth = typeHelper(&getContext());
        for (auto *user : varOp->getUsers())
        {
            auto propertyRefOp = mlir::dyn_cast<mlir_ts::PropertyRefOp>(user);
            if (!propertyRefOp)
            {
                continue;
            }

            auto field = propertyRefOp.getResult();
            auto refType = mlir::dyn_cast<mlir_ts::RefType>(field.getType());
            if (!refType || !mth.ownsHeapMemory(field.getLoc(), refType.getElementType()))
            {
                continue;
            }

            auto borrows = false;
            for (auto *fieldUser : field.getUsers())
            {
                if (mlir::isa<mlir_ts::LoadOp>(fieldUser))
                {
                    continue;
                }

                auto storeOp = mlir::dyn_cast<mlir_ts::StoreOp>(fieldUser);
                if (!storeOp || storeOp.getReference() != field)
                {
                    return;
                }

                // a handle is counted, never borrowed (spec 23.2)
                auto value = storeOp.getValue();
                if (isHandle(value))
                {
                    return;
                }

                // a field of a generator's state, read into the record a yield or `next()` makes: the
                // generator's rules decide whether that read moves the value out (findYieldedFields,
                // after the facts) or borrows it, not this one
                auto stateRead = rootOf(value).getDefiningOp<mlir_ts::LoadOp>();
                auto stateField = stateRead ? stateRead.getReference().getDefiningOp<mlir_ts::PropertyRefOp>() : mlir_ts::PropertyRefOp();
                if (auto stateType = stateField ? storageTupleOf(stateField.getObjectRef().getType()) : mlir_ts::TupleType();
                    stateType && fieldIndex(stateType, ".step") >= 0)
                {
                    return;
                }

                switch (recordFieldSource(value))
                {
                case RecordFieldSource::NoBlock:
                    continue;
                // a fresh value the temporary borrows: its own release bounds the record, which
                // cannot be returned
                case RecordFieldSource::Fresh:
                    if (!temporary)
                    {
                        // a field it would own: the tuple is no borrow (an owning tuple, #679), and
                        // its release would free the field
                        return;
                    }

                    info.returnable = false;
                    borrows = true;
                    continue;
                case RecordFieldSource::Unknown:
                    return;
                case RecordFieldSource::Immortal:
                    borrows = true;
                    continue;
                case RecordFieldSource::Param:
                case RecordFieldSource::Place:
                    if (auto param = borrowedParam(value); param >= 0)
                    {
                        info.params.push_back(param);
                    }
                    else
                    {
                        info.returnable = false;
                    }

                    borrows = true;
                    continue;
                case RecordFieldSource::Slot:
                case RecordFieldSource::Cell:
                    info.returnable = false;
                    borrows = true;
                    continue;
                }
            }

            // a string is copied, not borrowed (spec 22)
            if (borrows && isCopyableString(refType.getElementType()))
            {
                return;
            }

            if (borrows)
            {
                borrowed.push_back(propertyRefOp.getPosition());
            }
        }

        if (borrowed.empty())
        {
            return;
        }

        llvm::sort(borrowed);
        borrowed.erase(std::unique(borrowed.begin(), borrowed.end()), borrowed.end());
        llvm::sort(info.params);
        info.params.erase(std::unique(info.params.begin(), info.params.end()), info.params.end());

        varOp->setAttr(OWN_RECORD_UNBOXED_FIELDS_ATTR_NAME, mlir::DenseI32ArrayAttr::get(&getContext(), borrowed));
        if (temporary)
        {
            varOp->setAttr(OWN_RECORD_TEMPORARY_ATTR_NAME, mlir::UnitAttr::get(&getContext()));
        }
        for (auto *user : varOp->getUsers())
        {
            if (mlir::isa<mlir_ts::LoadOp>(user))
            {
                recordBoxes[user] = info;
            }
        }
    }

    bool OwnershipSignaturePass::capturedBoxBorrows(mlir::Value field, llvm::ArrayRef<mlir::Operation *> interfaces)
    {
        // the box, stored once
        mlir_ts::VariableOp boxOp;
        for (auto *user : field.getUsers())
        {
            if (mlir::isa<mlir_ts::LoadOp>(user))
            {
                continue;
            }

            auto storeOp = mlir::dyn_cast<mlir_ts::StoreOp>(user);
            auto varOp = storeOp && storeOp.getReference() == field ? storeOp.getValue().getDefiningOp<mlir_ts::VariableOp>()
                                                                    : mlir_ts::VariableOp();
            if (!varOp || !varOp->hasAttr(CAPTURE_BOX_ATTR_NAME) || boxOp)
            {
                return false;
            }

            boxOp = varOp;
        }

        if (!boxOp)
        {
            return false;
        }

        // ...filled with copies of values that have owners of their own
        for (auto *user : boxOp->getUsers())
        {
            if (mlir::isa<mlir_ts::RetainCellOp, mlir_ts::ReleaseCellOp>(user))
            {
                continue;
            }

            if (auto storeOp = mlir::dyn_cast<mlir_ts::StoreOp>(user); storeOp && storeOp.getReference() == field)
            {
                continue;
            }

            auto propertyRefOp = mlir::dyn_cast<mlir_ts::PropertyRefOp>(user);
            if (!propertyRefOp)
            {
                return false;
            }

            for (auto *fieldUser : propertyRefOp->getUsers())
            {
                auto fillOp = mlir::dyn_cast<mlir_ts::StoreOp>(fieldUser);
                if (!fillOp || fillOp.getReference() != propertyRefOp.getResult())
                {
                    return false;
                }

                auto value = fillOp.getValue();
                // a cell of a `let` of this frame, which frees it (spec 69), or a copy
                if (mlir::isa<mlir_ts::RefType>(value.getType()) ? !isCellVariable(value)
                                                                  : !holdsNoBlock(value) && !hasOwnerOfItsOwn(value))
                {
                    return false;
                }
            }
        }

        // ...and nothing keeps the record: its retains and releases, and calls that only borrow it
        for (auto *newInterfaceOp : interfaces)
        {
            auto stays = true;
            forEachUse(newInterfaceOp->getResult(0), [&](mlir::Operation *use, mlir::Value used) {
                stays = stays && (mlir::isa<mlir_ts::RetainOp, mlir_ts::ReleaseOp>(use) ||
                                  (isCall(use) && !keeps(use, used) && resultBorrows(use) < 0));
            });
            if (!stays)
            {
                return false;
            }
        }

        return true;
    }

    void OwnershipSignaturePass::classifyRecordBox(mlir_ts::NewOp newOp)
    {
        auto boxOp = recordBoxOf(newOp.getResult());
        auto varOp = boxOp == newOp ? recordLocalOf(newOp.getResult()) : mlir_ts::VariableOp();
        if (!varOp)
        {
            return;
        }

        // the record read once, and only into its box (rc's retain of it beside)
        mlir_ts::LoadOp read;
        for (auto *user : varOp->getUsers())
        {
            if (auto loadOp = mlir::dyn_cast<mlir_ts::LoadOp>(user))
            {
                if (read)
                {
                    return;
                }

                read = loadOp;
            }
        }

        for (auto *user : read->getUsers())
        {
            auto storeOp = mlir::dyn_cast<mlir_ts::StoreOp>(user);
            if (!mlir::isa<mlir_ts::RetainOp>(user) && !(storeOp && storeOp.getReference() == newOp.getResult()))
            {
                return;
            }
        }

        // the box seen only as interfaces: every release of it reads their tag
        llvm::SmallVector<mlir::Operation *> interfaces;
        for (auto &use : newOp->getUses())
        {
            auto *user = use.getOwner();
            if (auto storeOp = mlir::dyn_cast<mlir_ts::StoreOp>(user); storeOp && storeOp.getReference() == newOp.getResult())
            {
                continue;
            }

            auto castOp = mlir::dyn_cast<mlir_ts::CastOp>(user);
            if (!castOp || !isView(castOp))
            {
                return;
            }

            for (auto *castUser : castOp->getUsers())
            {
                auto newInterfaceOp = mlir::dyn_cast<mlir_ts::NewInterfaceOp>(castUser);
                if (!newInterfaceOp || newInterfaceOp.getThisVal() != castOp.getResult())
                {
                    return;
                }

                interfaces.push_back(castUser);
            }
        }

        if (interfaces.empty())
        {
            return;
        }

        // each field that holds a block: owned (made here, or nothing) or borrowed, never both
        RecordBox info;
        llvm::SmallVector<int32_t> borrowed;
        auto mth = typeHelper(&getContext());
        for (auto *user : varOp->getUsers())
        {
            auto propertyRefOp = mlir::dyn_cast<mlir_ts::PropertyRefOp>(user);
            if (!propertyRefOp)
            {
                continue;
            }

            auto field = propertyRefOp.getResult();
            if (capturedBoxBorrows(field, interfaces))
            {
                // never returned: what the box borrows is this function's
                info.returnable = false;
                borrowed.push_back(propertyRefOp.getPosition());
                continue;
            }

            auto refType = mlir::dyn_cast<mlir_ts::RefType>(field.getType());
            if (!refType || !mth.ownsHeapMemory(field.getLoc(), refType.getElementType()))
            {
                continue;
            }

            auto owns = false;
            auto borrows = false;
            for (auto *fieldUser : field.getUsers())
            {
                if (mlir::isa<mlir_ts::LoadOp>(fieldUser))
                {
                    continue;
                }

                auto storeOp = mlir::dyn_cast<mlir_ts::StoreOp>(fieldUser);
                if (!storeOp || storeOp.getReference() != field)
                {
                    return;
                }

                auto value = storeOp.getValue();
                switch (recordFieldSource(value))
                {
                case RecordFieldSource::NoBlock:
                    continue;
                case RecordFieldSource::Fresh:
                    owns = true;
                    continue;
                case RecordFieldSource::Unknown:
                    return;
                case RecordFieldSource::Immortal:
                    borrows = true;
                    continue;
                case RecordFieldSource::Param:
                case RecordFieldSource::Place:
                    if (auto param = borrowedParam(value); param >= 0)
                    {
                        info.params.push_back(param);
                    }
                    else
                    {
                        info.returnable = false;
                    }

                    borrows = true;
                    continue;
                case RecordFieldSource::Slot:
                case RecordFieldSource::Cell:
                    info.returnable = false;
                    borrows = true;
                    continue;
                }
            }

            // a string is copied, not borrowed (spec 22)
            if ((owns && borrows) || (borrows && isCopyableString(refType.getElementType())))
            {
                return;
            }

            if (borrows)
            {
                borrowed.push_back(propertyRefOp.getPosition());
            }
        }

        if (borrowed.empty())
        {
            return;
        }

        llvm::sort(borrowed);
        borrowed.erase(std::unique(borrowed.begin(), borrowed.end()), borrowed.end());
        llvm::sort(info.params);
        info.params.erase(std::unique(info.params.begin(), info.params.end()), info.params.end());

        auto attr = mlir::DenseI32ArrayAttr::get(&getContext(), borrowed);
        newOp->setAttr(OWN_RECORD_BORROWED_FIELDS_ATTR_NAME, attr);
        for (auto *newInterfaceOp : interfaces)
        {
            newInterfaceOp->setAttr(OWN_RECORD_BORROWED_FIELDS_ATTR_NAME, attr);
        }

        recordBoxes[newOp.getOperation()] = std::move(info);
    }

    // A borrowing record's box is released by a routine of its own, which leaves the fields it
    // borrows to their owners (spec 32) - through the type tag of the vtable it is seen through as an
    // interface (INTERFACE_VTABLE_TYPE_SLOT). That vtable is the object literal's, which every record
    // of its shape shares, so a box that borrows is seen through a copy of it whose tag names the
    // borrowing routine (TypeDescriptorOpLowering), one copy per set of fields borrowed.
    void OwnershipSignaturePass::giveBorrowingRecordsTheirVTables(mlir::ModuleOp module)
    {
        mlir::OpBuilder builder(&getContext());
        module.walk([&](mlir_ts::NewInterfaceOp newInterfaceOp) {
            auto attr = newInterfaceOp->getAttrOfType<mlir::DenseI32ArrayAttr>(OWN_RECORD_BORROWED_FIELDS_ATTR_NAME);
            if (!attr)
            {
                return;
            }

            // where the vtable's address is taken: straight (through casts), or for the copy an object
            // literal's absent members are patched in, the load it is copied from
            mlir::OpOperand *sourceUse = &newInterfaceOp.getInterfaceVTableMutable();
            while (auto castOp = sourceUse->get().getDefiningOp<mlir_ts::CastOp>())
            {
                sourceUse = &castOp.getInMutable();
            }

            if (auto newOp = sourceUse->get().getDefiningOp<mlir_ts::NewOp>())
            {
                sourceUse = nullptr;
                for (auto *user : newOp->getUsers())
                {
                    auto storeOp = mlir::dyn_cast<mlir_ts::StoreOp>(user);
                    auto loadOp = storeOp && storeOp.getReference() == newOp.getResult()
                                      ? storeOp.getValue().getDefiningOp<mlir_ts::LoadOp>()
                                      : mlir_ts::LoadOp();
                    if (loadOp)
                    {
                        sourceUse = &loadOp.getReferenceMutable();
                    }
                }
            }

            auto addressOfOp = sourceUse ? sourceUse->get().getDefiningOp<mlir_ts::AddressOfOp>() : mlir_ts::AddressOfOp();
            auto globalOp = addressOfOp ? module.lookupSymbol<mlir_ts::GlobalOp>(addressOfOp.getGlobalName()) : mlir_ts::GlobalOp();
            if (!globalOp || !globalOp.getSymName().ends_with("..vtbl"))
            {
                newInterfaceOp.emitError("-mm=own: the vtable of a record that borrows its fields can't be given the box's own type tag");
                signalPassFailure();
                return;
            }

            std::string variantName = globalOp.getSymName().drop_back(llvm::StringRef("..vtbl").size()).str();
            for (auto index : attr.asArrayRef())
            {
                variantName += ".r" + std::to_string(index);
            }

            variantName += "..vtbl";

            auto variantOp = module.lookupSymbol<mlir_ts::GlobalOp>(variantName);
            if (!variantOp)
            {
                builder.setInsertionPointAfter(globalOp);
                variantOp = mlir::cast<mlir_ts::GlobalOp>(builder.clone(*globalOp));
                variantOp.setSymName(variantName);
                variantOp.getInitializerRegion().walk([&](mlir_ts::TypeDescriptorOp descriptorOp) {
                    descriptorOp->setAttr(OWN_RECORD_BORROWED_FIELDS_ATTR_NAME, attr);
                });
            }

            builder.setInsertionPoint(addressOfOp);
            auto variantAddressOfOp = builder.create<mlir_ts::AddressOfOp>(addressOfOp.getLoc(), variantOp);
            sourceUse->set(variantAddressOfOp);
        });
    }

    OwnershipSignaturePass::RecordFlow OwnershipSignaturePass::recordParamsOf(mlir::Value value, llvm::SmallVectorImpl<int32_t> &params,
                                                                              bool *unboxed)
    {
        llvm::SmallVector<mlir::Value> work{value};
        llvm::SmallPtrSet<mlir::Value, 8> seen;
        auto record = false;
        auto other = false;
        // boxes and unboxed records met in one value: their releases differ
        auto boxes = false;
        auto tuples = false;
        while (!work.empty())
        {
            // what holds no block, by its own type (a box's `value_ref` owns nothing by type, but
            // the object its view shows does)
            auto next = work.pop_back_val();
            auto current = rootOf(next);
            if (!seen.insert(current).second || holdsNoBlock(next))
            {
                continue;
            }

            auto *def = current.getDefiningOp();
            if (auto found = def ? recordBoxes.find(def) : recordBoxes.end(); found != recordBoxes.end())
            {
                if (!found->second.returnable)
                {
                    return RecordFlow::Bad;
                }

                params.append(found->second.params.begin(), found->second.params.end());
                record = true;
                (found->second.unboxed ? tuples : boxes) = true;
                continue;
            }

            if (def && isCall(def) && isRecordResult(def))
            {
                (isUnboxedRecordResult(def) ? tuples : boxes) = true;
                auto args = callArgs(def);
                for (auto index : resultBorrowsAll(def))
                {
                    if (static_cast<size_t>(index) >= args.size())
                    {
                        return RecordFlow::Bad;
                    }

                    auto arg = args[index];
                    if (argumentHoldsNoBlock(arg) || isImmortalGlobalRead(arg))
                    {
                        continue;
                    }

                    auto param = borrowedParam(arg);
                    if (param < 0)
                    {
                        return RecordFlow::Bad;
                    }

                    params.push_back(param);
                }

                record = true;
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

            // what an owning local holds: its initializer and every value stored into it
            auto loadOp = mlir::dyn_cast_or_null<mlir_ts::LoadOp>(def);
            auto varOp = loadOp ? loadOp.getReference().getDefiningOp<mlir_ts::VariableOp>() : mlir_ts::VariableOp();
            if (varOp && isOwningVariable(varOp) && !varOp.getCaptured().value_or(false))
            {
                if (varOp.getInitializer())
                {
                    work.push_back(varOp.getInitializer());
                }

                for (auto *user : varOp->getUsers())
                {
                    if (auto storeOp = mlir::dyn_cast<mlir_ts::StoreOp>(user); storeOp && storeOp.getReference() == varOp.getResult())
                    {
                        work.push_back(storeOp.getValue());
                    }
                }

                continue;
            }

            other = true;
        }

        if (!record)
        {
            return RecordFlow::None;
        }

        if (unboxed)
        {
            *unboxed = tuples;
        }

        return other || (boxes && tuples) ? RecordFlow::Bad : RecordFlow::Record;
    }

} // namespace own_signature
