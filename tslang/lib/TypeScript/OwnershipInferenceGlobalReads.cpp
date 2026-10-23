// OwnershipInferencePass (see OwnershipInferencePass.h): a read of a mortal global, ended at the
// global's assignment, and the locals that hold it (spec 25).

#include "OwnershipInferencePass.h"

#define DEBUG_TYPE "own"

namespace own_inference
{

    std::optional<OwnershipInferencePass::BorrowHolder> OwnershipInferencePass::borrowHolderOf(mlir::Operation *user)
    {
        BorrowHolder holder;
        mlir::Value ref;
        if (auto storeOp = mlir::dyn_cast<mlir_ts::StoreOp>(user))
        {
            ref = storeOp.getReference();
            while (auto propertyRefOp = ref.getDefiningOp<mlir_ts::PropertyRefOp>())
            {
                holder.path.insert(holder.path.begin(), propertyRefOp.getPosition());
                ref = propertyRefOp.getObjectRef();
            }
        }
        else if (auto varOp = mlir::dyn_cast<mlir_ts::VariableOp>(user))
        {
            ref = varOp.getResult();
        }

        auto varOp = ref ? ref.getDefiningOp<mlir_ts::VariableOp>() : mlir_ts::VariableOp();
        if (!varOp || varOp.getCaptured().value_or(false) || isCellVariable(varOp.getResult()))
        {
            return std::nullopt;
        }

        holder.local = varOp;
        return holder;
    }

    llvm::SmallVector<mlir::Operation *, 4> OwnershipInferencePass::holderRenewals(BorrowHolder holder)
    {
        llvm::SmallVector<mlir::Operation *, 4> renewals;
        if (holder.local.getInitializer())
        {
            renewals.push_back(holder.local);
        }

        llvm::SmallVector<std::pair<mlir::Value, size_t>> refs{{holder.local.getResult(), 0}};
        while (!refs.empty())
        {
            auto [ref, depth] = refs.pop_back_val();
            for (auto *refUser : ref.getUsers())
            {
                if (auto propertyRefOp = mlir::dyn_cast<mlir_ts::PropertyRefOp>(refUser))
                {
                    if (depth < holder.path.size() && propertyRefOp.getPosition() == holder.path[depth])
                    {
                        refs.push_back({propertyRefOp.getResult(), depth + 1});
                    }
                }
                else if (auto storeOp = mlir::dyn_cast<mlir_ts::StoreOp>(refUser); storeOp && storeOp.getReference() == ref)
                {
                    renewals.push_back(storeOp);
                }
            }
        }

        return renewals;
    }

    bool OwnershipInferencePass::pathsMeet(llvm::ArrayRef<int64_t> a, llvm::ArrayRef<int64_t> b)
    {
        auto common = std::min(a.size(), b.size());
        return a.take_front(common) == b.take_front(common);
    }

    bool OwnershipInferencePass::isGlobalRead(mlir_ts::LoadOp loadOp)
    {
        auto addressOfOp = loadOp.getReference().getDefiningOp<mlir_ts::AddressOfOp>();
        auto result = loadOp.getResult();
        return addressOfOp && !addressOfOp->hasAttr(OWN_IMMORTAL_GLOBAL_ATTR_NAME) && ownsHeap(result) && !isHandle(result);
    }

    void OwnershipInferencePass::checkGlobalRead(mlir_ts::LoadOp read)
    {
        // each access of a global is a `ts.AddressOf` of its own
        auto global = read.getReference().getDefiningOp<mlir_ts::AddressOfOp>().getGlobalName();
        llvm::SmallVector<BorrowUse> uses;
        llvm::StringSet<> followed;
        mlir::Operation *keptAt = nullptr;
        // rc's temporary release of a call's result that borrows an argument goes with that result
        // (walkBorrowed's `keeping`); the owner checks decide it
        llvm::SmallVector<mlir::Operation *> bookkeeping;
        Keeping keeping{&bookkeeping, false};
        std::function<bool(mlir::Operation *, llvm::ArrayRef<mlir::Operation *>)> intoLocal;

        // the uses of what `holder` holds, made again at its renewals: those depend on the holder
        // alone, not on the way the walk reached it, so a holder is followed once. A value loaded
        // out of it is made again only by a path back through that load (the load itself is a use,
        // bounded by the renewals): a store into the path after the load gives the loaded value
        // nothing new.
        auto followHolder = [&](BorrowHolder holder) {
            std::string key;
            llvm::raw_string_ostream keyOut(key);
            keyOut << holder.local.getOperation();
            for (auto position : holder.path)
            {
                keyOut << "." << position;
            }

            if (!followed.insert(keyOut.str()).second)
            {
                return true;
            }

            auto localKills = holderRenewals(holder);
            llvm::SmallVector<std::pair<mlir::Value, llvm::SmallVector<int64_t, 2>>> refs{{holder.local.getResult(), {}}};
            while (!refs.empty())
            {
                auto [ref, path] = refs.pop_back_val();
                for (auto *refUser : ref.getUsers())
                {
                    if (auto propertyRefOp = mlir::dyn_cast<mlir_ts::PropertyRefOp>(refUser))
                    {
                        auto fieldPath = path;
                        fieldPath.push_back(propertyRefOp.getPosition());
                        if (pathsMeet(fieldPath, holder.path))
                        {
                            refs.push_back({propertyRefOp.getResult(), fieldPath});
                        }

                        continue;
                    }

                    if (ref == holder.local.getResult() &&
                        mlir::isa<mlir_ts::RetainSlotOp, mlir_ts::ReleaseSlotOp, mlir_ts::DebugVariableOp>(refUser))
                    {
                        continue;
                    }

                    uses.push_back({refUser, {localKills.begin(), localKills.end()}});
                    if (auto loadOp = mlir::dyn_cast<mlir_ts::LoadOp>(refUser))
                    {
                        if (auto *kept = walkBorrowed(loadOp.getResult(), loadOp.getOperation(), uses, intoLocal, keeping))
                        {
                            keptAt = keptAt ? keptAt : kept;
                            return false;
                        }
                    }
                }
            }

            return true;
        };

        intoLocal = [&](mlir::Operation *user, llvm::ArrayRef<mlir::Operation *> kills) {
            // a tuple taken apart (`{ ...b }`, a destructuring): each field is the borrow again
            if (mlir::isa<mlir_ts::DeconstructTupleOp>(user))
            {
                for (auto result : user->getResults())
                {
                    if (!mayPointInto(result))
                    {
                        continue;
                    }

                    if (auto *kept = walkBorrowed(result, kills, uses, intoLocal, keeping))
                    {
                        keptAt = keptAt ? keptAt : kept;
                        return false;
                    }
                }

                return true;
            }

            auto holder = borrowHolderOf(user);
            return holder && followHolder(*holder);
        };

        auto *kept = walkBorrowed(read.getResult(), read.getOperation(), uses, intoLocal, keeping);

        // a value made from the borrow (a field of a tuple read out of a holder, `{ ...b }`) stored
        // into another local, or declaring one: that local holds the borrow too
        for (size_t index = 0; !kept && !keptAt && index < uses.size(); index++)
        {
            auto *op = uses[index].op;
            auto storeOp = mlir::dyn_cast<mlir_ts::StoreOp>(op);
            auto varOp = mlir::dyn_cast<mlir_ts::VariableOp>(op);
            auto value = storeOp ? storeOp.getValue() : varOp ? varOp.getInitializer() : mlir::Value();
            if (!value || !mayPointInto(value))
            {
                continue;
            }

            if (auto holder = borrowHolderOf(op))
            {
                followHolder(*holder);
            }
        }

        if (kept || keptAt)
        {
            if (mlir::isa<mlir_ts::StringType>(read.getType()))
            {
                return;
            }

            if (reporting())
            {
                auto *at = keptAt ? keptAt : kept;
                at->emitError("the value read from '")
                    << global << "' is kept where its end at an assignment of '" << global
                    << "' cannot be checked; -mm=own cannot prove it outlives the global's value";
                signalPassFailure();
            }

            return;
        }

        llvm::SmallVector<mlir::Operation *> ends;
        getFunction().walk([&](mlir::Operation *op) {
            mlir::Value slot;
            if (auto releaseSlotOp = mlir::dyn_cast<mlir_ts::ReleaseSlotOp>(op))
            {
                slot = releaseSlotOp.getSlot();
            }
            else if (auto storeOp = mlir::dyn_cast<mlir_ts::StoreOp>(op))
            {
                slot = storeOp.getReference();
            }

            auto other = slot ? slot.getDefiningOp<mlir_ts::AddressOfOp>() : mlir_ts::AddressOfOp();
            if (other && other.getGlobalName() == global)
            {
                ends.push_back(op);
            }

            // a call that may give the global a new value (OwnershipSignaturePass, pinAssignedGlobals):
            // `const c = G; reset(); print(c.x)`
            if (isCall(op))
            {
                auto assigned = op->getAttrOfType<mlir::ArrayAttr>(OWN_ASSIGNS_GLOBALS_ATTR_NAME);
                if (op->hasAttr(OWN_ASSIGNS_ANY_GLOBAL_ATTR_NAME) ||
                    (assigned && llvm::any_of(assigned.getAsRange<mlir::StringAttr>(),
                                              [&](mlir::StringAttr name) { return name.getValue() == global; })))
                {
                    ends.push_back(op);
                }
            }
        });

        for (auto *end : ends)
        {
            for (auto &use : uses)
            {
                if (use.op != end && reachableAfter(end, use.op, use.kills))
                {
                    reportBorrowOutlives(use.op, end, ownerName(read.getResult()), global);
                    return;
                }
            }
        }
    }

} // namespace own_inference
