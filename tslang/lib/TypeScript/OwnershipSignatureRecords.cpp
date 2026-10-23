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

    OwnershipSignaturePass::RecordFlow OwnershipSignaturePass::recordParamsOf(mlir::Value value, llvm::SmallVectorImpl<int32_t> &params)
    {
        llvm::SmallVector<mlir::Value> work{value};
        llvm::SmallPtrSet<mlir::Value, 8> seen;
        auto record = false;
        auto other = false;
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
                continue;
            }

            if (def && isCall(def) && isRecordResult(def))
            {
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

        return other ? RecordFlow::Bad : RecordFlow::Record;
    }

} // namespace own_signature
