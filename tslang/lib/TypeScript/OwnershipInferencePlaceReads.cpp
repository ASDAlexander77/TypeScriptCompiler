// OwnershipInferencePass (see OwnershipInferencePass.h): reads out of containers (spec 2.3,
// phase 3) - handle reads, the chain from a read to its roots, and what may drop it.

#include "OwnershipInferencePass.h"

#define DEBUG_TYPE "own"

namespace own_inference
{

    mlir::Operation *OwnershipInferencePass::placeReadOf(mlir::Value value)
    {
        auto root = rootOf(value);
        // a read behind a test whose other side is the default (`let [x] = a`) is that read; the
        // walk from it (checkPlaceRead) goes on through the branch to what takes the merge
        if (auto sole = soleBlockMergedInto(root))
        {
            root = sole;
        }

        auto *def = root.getDefiningOp();
        // a handle read out of a place that rc gives an owner of its own is a counted copy, not a
        // borrow (spec 23.3); any other is a borrow of the place, as a field read is
        if (!def || !ownsHeap(root) || (isHandle(root) && isCountedHandleRead(root)))
        {
            return nullptr;
        }

        if (auto loadOp = mlir::dyn_cast<mlir_ts::LoadOp>(def); loadOp && isPlace(loadOp.getReference()))
        {
            // a field of a generator's borrowing box: only its maker fills it, and nothing a call
            // can reach overwrites it; it is the caller's, as a parameter is (checkBorrowedReads)
            if (isBorrowingBoxField(loadOp.getReference()))
            {
                return nullptr;
            }

            return def;
        }

        return borrowingCall(root);
    }

    OwnershipInferencePass::HandleReadUses OwnershipInferencePass::handleReadUses(mlir::Value value)
    {
        HandleReadUses result{false, false};
        llvm::SmallVector<mlir::Value> values{value};
        llvm::SmallPtrSet<mlir::Value, 8> seen{value};
        while (!values.empty())
        {
            auto current = values.pop_back_val();
            for (auto &use : current.getUses())
            {
                auto *user = use.getOwner();
                if (mlir::isa<mlir_ts::RetainOp>(user))
                {
                    continue;
                }

                // the count a call was given, given back right after it (spec 50)
                if (auto *call = mlir::isa<mlir_ts::ReleaseOp>(user) ? user->getPrevNode() : nullptr;
                    call && isCall(call) && isCountedHandleReceiver(call, current))
                {
                    continue;
                }

                // a return and an `any` box take the value, though passesOn would hand them on
                if (isHandOff(user, current))
                {
                    (isCountedHandleReceiver(user, current) ? result.counted : result.borrowed) = true;
                    continue;
                }

                llvm::SmallVector<mlir::Value> same;
                if (passesOn(use, same))
                {
                    for (auto next : same)
                    {
                        // what is no longer a handle is no receiver: a test (`if (p.child)`) only
                        // reads it
                        if (!isHandle(next))
                        {
                            result.borrowed = true;
                        }
                        else if (seen.insert(next).second)
                        {
                            values.push_back(next);
                        }
                    }

                    continue;
                }

                (isCountedHandleReceiver(user, current) ? result.counted : result.borrowed) = true;
            }
        }

        return result;
    }

    bool OwnershipInferencePass::aliasReadAfter(mlir::Operation *read, mlir::Value fieldRef)
    {
        auto propertyRefOp = fieldRef.getDefiningOp<mlir_ts::PropertyRefOp>();
        auto objectRead = propertyRefOp ? rootOf(propertyRefOp.getObjectRef()).getDefiningOp<mlir_ts::LoadOp>() : mlir_ts::LoadOp();
        auto holder = objectRead ? objectRead.getReference().getDefiningOp<mlir_ts::VariableOp>() : mlir_ts::VariableOp();
        // (through a chain of such locals: `let a = n; let b = a; o.v = b.v`)
        for (auto depth = 0; holder && holder.getInitializer() && depth < 4; depth++)
        {
            auto initial = rootOf(holder.getInitializer());
            auto sourceRead = initial.getDefiningOp<mlir_ts::LoadOp>();
            auto source = sourceRead ? sourceRead.getReference().getDefiningOp<mlir_ts::VariableOp>() : mlir_ts::VariableOp();
            if (!source)
            {
                // a `const` that is the value itself (`const n = mk(); let n2 = n;`): read again where
                // it is used besides the declaration, rc's counts and the end of its scope
                // (round a loop, through where it is made again, it is another value)
                llvm::SmallVector<mlir::Operation *, 1> made;
                if (auto *definition = initial.getDefiningOp())
                {
                    made.push_back(definition);
                }

                return llvm::any_of(initial.getUsers(), [&](mlir::Operation *user) {
                    return user != holder.getOperation() &&
                           !mlir::isa<mlir_ts::RetainOp, mlir_ts::ReleaseOp, mlir_ts::DebugVariableOp>(user) &&
                           reachableAfter(read, user, made);
                });
            }

            // (...or the local declared or given another value again)
            llvm::SmallVector<mlir::Operation *> given{source.getOperation()};
            for (auto *assignment : slotAssignments(source.getResult()))
            {
                given.push_back(assignment);
            }

            for (auto *user : source.getResult().getUsers())
            {
                if (mlir::isa<mlir_ts::LoadOp>(user) && user != sourceRead.getOperation() && reachableAfter(read, user, given))
                {
                    return true;
                }
            }

            holder = source;
        }

        return false;
    }

    bool OwnershipInferencePass::isCountedHandleRead(mlir::Value value)
    {
        auto uses = handleReadUses(value);
        return uses.counted && !uses.borrowed;
    }

    bool OwnershipInferencePass::isCountedHandleReceiver(mlir::Operation *user, mlir::Value value)
    {
        if (mlir::isa<mlir_ts::RetainOp>(user))
        {
            return true;
        }

        if (auto varOp = mlir::dyn_cast<mlir_ts::VariableOp>(user))
        {
            return isOwningVariable(varOp) && llvm::any_of(varOp->getUsers(), [](mlir::Operation *varUser) {
                       return mlir::isa<mlir_ts::RetainSlotOp>(varUser);
                   });
        }

        if (auto storeOp = mlir::dyn_cast<mlir_ts::StoreOp>(user))
        {
            return storeOp.getValue() == value && countedStoreRetain(storeOp);
        }

        if (auto pushOp = mlir::dyn_cast<mlir_ts::ArrayPushOp>(user))
        {
            return retainedRightBefore(user, pushOp.getItems(), value);
        }

        if (auto unshiftOp = mlir::dyn_cast<mlir_ts::ArrayUnshiftOp>(user))
        {
            return retainedRightBefore(user, unshiftOp.getItems(), value);
        }

        if (auto spliceOp = mlir::dyn_cast<mlir_ts::ArraySpliceOp>(user))
        {
            return retainedRightBefore(user, spliceOp.getItems(), value);
        }

        // a call given a count of its own (callCopies, spec 50): retained right before it, and
        // given back right after it unless the callee keeps it
        if (isCall(user) && !mlir::isa<mlir_ts::InvokeOp>(user) && retainedRightBefore(user, user->getOperands(), value))
        {
            auto *next = user->getNextNode();
            auto releaseOp = mlir::dyn_cast_or_null<mlir_ts::ReleaseOp>(next);
            return isKeptArgument(user, value) || (releaseOp && releaseOp.getReference() == value);
        }

        if (auto castOp = mlir::dyn_cast<mlir_ts::CastOp>(user))
        {
            return mlir::isa<mlir_ts::AnyType>(castOp.getType()) && retainedRightBefore(user, user->getOperands(), value);
        }

        // `ts.ReturnVal` is the return before the affine lowering, which makes it a store into
        // the result slot
        if (mlir::isa<mlir_ts::SharedNewOp, mlir_ts::CreateArrayOp, mlir_ts::ReturnValOp>(user))
        {
            return retainedRightBefore(user, user->getOperands(), value);
        }

        return false;
    }

    bool OwnershipInferencePass::isHandOff(mlir::Operation *user, mlir::Value value)
    {
        if (auto storeOp = mlir::dyn_cast<mlir_ts::StoreOp>(user))
        {
            auto slot = storeOp.getReference().getDefiningOp<mlir_ts::VariableOp>();
            return storeOp.getValue() == value && slot && isResultSlot(slot);
        }

        auto castOp = mlir::dyn_cast<mlir_ts::CastOp>(user);
        return castOp && castOp.getIn() == value && mlir::isa<mlir_ts::AnyType>(castOp.getType());
    }

    bool OwnershipInferencePass::isCountedHandOff(mlir::Operation *user, mlir::Value value)
    {
        return isHandOff(user, value) && isCountedHandleReceiver(user, value);
    }

    bool OwnershipInferencePass::retainedRightBefore(mlir::Operation *op, mlir::ValueRange operands, mlir::Value value)
    {
        auto needed = llvm::count(operands, value);
        if (needed == 0)
        {
            return false;
        }

        for (auto *previous = op->getPrevNode(); needed > 0 && previous; previous = previous->getPrevNode())
        {
            auto retainOp = mlir::dyn_cast<mlir_ts::RetainOp>(previous);
            if (!retainOp)
            {
                break;
            }

            if (retainOp.getReference() == value)
            {
                --needed;
            }
        }

        return needed == 0;
    }

    void OwnershipInferencePass::collectCountedStoreReleases(mlir::Value value, llvm::SmallPtrSetImpl<mlir::Operation *> &releases)
    {
        llvm::SmallVector<mlir::Value> values{value};
        llvm::SmallPtrSet<mlir::Value, 8> seen{value};
        while (!values.empty())
        {
            auto current = values.pop_back_val();
            for (auto &use : current.getUses())
            {
                llvm::SmallVector<mlir::Value> same;
                if (passesOn(use, same))
                {
                    for (auto next : same)
                    {
                        if (isHandle(next) && seen.insert(next).second)
                        {
                            values.push_back(next);
                        }
                    }

                    continue;
                }

                auto storeOp = mlir::dyn_cast<mlir_ts::StoreOp>(use.getOwner());
                if (!storeOp || storeOp.getValue() != current || !countedStoreRetain(storeOp))
                {
                    continue;
                }

                if (auto releaseSlotOp = mlir::dyn_cast_or_null<mlir_ts::ReleaseSlotOp>(storeOp->getPrevNode()))
                {
                    releases.insert(releaseSlotOp);
                }
            }
        }
    }

    mlir::Operation *OwnershipInferencePass::countedStoreRetain(mlir_ts::StoreOp storeOp)
    {
        auto *previous = storeOp->getPrevNode();
        if (auto slot = storeOp.getReference().getDefiningOp<mlir_ts::VariableOp>(); slot && isResultSlot(slot))
        {
            // the scope exit's releases, of locals and of their slots, in any order
            while (mlir::isa_and_nonnull<mlir_ts::ReleaseOp, mlir_ts::ReleaseSlotOp>(previous) &&
                   previous->getOperand(0) != storeOp.getValue())
            {
                previous = previous->getPrevNode();
            }
        }
        else if (auto releaseSlotOp = mlir::dyn_cast_or_null<mlir_ts::ReleaseSlotOp>(previous))
        {
            previous = releaseSlotOp.getSlot() == storeOp.getReference() ? releaseSlotOp->getPrevNode() : nullptr;
        }

        auto retainOp = mlir::dyn_cast_or_null<mlir_ts::RetainOp>(previous);
        return retainOp && retainOp.getReference() == storeOp.getValue() ? retainOp.getOperation() : nullptr;
    }

    mlir::Operation *OwnershipInferencePass::borrowingCall(mlir::Value value)
    {
        auto *def = value.getDefiningOp();
        if (!def || !isCall(def) || resultBorrows(def) < 0)
        {
            return nullptr;
        }

        auto args = callArgs(def);
        auto indices = resultBorrowsAll(def);
        return llvm::all_of(indices, [&](int index) { return static_cast<size_t>(index) < args.size(); }) ? def : nullptr;
    }

    OwnershipInferencePass::Chain OwnershipInferencePass::chainOf(mlir::Operation *read, mlir::Value place)
    {
        Chain chain;
        llvm::SmallVector<mlir::Value> refs;
        llvm::SmallVector<mlir::Value> first;
        if (place)
        {
            refs.push_back(place);
        }
        else if (auto loadOp = mlir::dyn_cast<mlir_ts::LoadOp>(read))
        {
            refs.push_back(loadOp.getReference());
        }
        else if (mlir::isa<mlir_ts::WeakRefDerefOp>(read))
        {
            // the object a WeakRef refers to (spec 84): no root here owns it, and wherever it is
            // held, a release of a holder of its type or a call that may drop ends it
            chain.anyPlace = true;
        }
        else
        {
            // a call's result that borrows its arguments - a plain borrow, or a borrowing record's box
            // (spec 32), which may hold a borrow of each - is somewhere under each of them
            chain.anyPlace = true;
            auto args = callArgs(read);
            for (auto index : resultBorrowsAll(read))
            {
                if (static_cast<size_t>(index) < args.size() && !argumentHoldsNoBlock(args[index]) &&
                    !isImmortalLiteral(args[index]))
                {
                    first.push_back(rootOf(args[index]));
                }
            }
        }

        llvm::SmallPtrSet<mlir::Value, 8> seen;
        // the bases a `s.value` was read through, and what merges into them
        llvm::SmallPtrSet<mlir::Value, 4> handleBases;
        while (!refs.empty() || !first.empty())
        {
            llvm::SmallVector<mlir::Value> bases;
            if (!first.empty())
            {
                bases.swap(first);
            }
            else
            {
                auto ref = refs.pop_back_val();
                auto place = placeBase(ref);
                assert(place && "chainOf walks places only");
                switch (place.kind)
                {
                case PlaceKind::Field:
                    chain.places.push_back({ref.getDefiningOp<mlir_ts::PropertyRefOp>().getPosition(), ref.getType()});
                    break;
                case PlaceKind::SharedValue:
                    chain.places.push_back({Chain::SharedValue, ref.getType()});
                    handleBases.insert(rootOf(place.base));
                    break;
                case PlaceKind::InterfaceField:
                    chain.places.push_back({Chain::InterfaceField, ref.getType()});
                    break;
                default:
                    chain.places.push_back({-1, ref.getType()});
                    break;
                }

                bases.push_back(rootOf(place.base));
            }

            while (!bases.empty())
            {
                auto base = bases.pop_back_val();
                if (!seen.insert(base).second)
                {
                    continue;
                }

                // A handle (spec 23.3): other handles may reach its block, so this function owns
                // nothing under it. A handle read out of a place still has that place above it, and
                // one merged from several has each of them.
                if (handleBases.contains(base) || isHandle(base))
                {
                    llvm::SmallVector<mlir::Value> merged;
                    if (auto argument = mlir::dyn_cast<mlir::BlockArgument>(base); argument && mergedInto(argument, merged))
                    {
                        handleBases.insert(merged.begin(), merged.end());
                        bases.append(merged.begin(), merged.end());
                        continue;
                    }

                    chain.roots.push_back({base, Chain::NotOwned});
                    chain.handles.push_back(base);
                    if (auto loadOp = base.getDefiningOp<mlir_ts::LoadOp>(); loadOp && isPlace(loadOp.getReference()))
                    {
                        refs.push_back(loadOp.getReference());
                    }

                    continue;
                }

                if (isPlace(base))
                {
                    refs.push_back(base); // a field of a value-type field: the same object
                    continue;
                }

                if (auto loadOp = base.getDefiningOp<mlir_ts::LoadOp>(); loadOp && isPlace(loadOp.getReference()))
                {
                    refs.push_back(loadOp.getReference());
                    continue;
                }

                // the object a method is called on
                if (auto object = boundThis(base))
                {
                    bases.push_back(rootOf(object));
                    continue;
                }

                // an object seen as the storage it points at, its fields read through it: the object
                // (an object literal this function made is its own, `applyT(p)` of one whose arrows
                // capture reads a method bound to it into the parameter's object type)
                if (auto castOp = base.getDefiningOp<mlir_ts::CastOp>();
                    castOp && mlir::isa<mlir_ts::ObjectType>(castOp.getIn().getType()) &&
                    mlir::isa<mlir_ts::RefType>(castOp.getType()))
                {
                    bases.push_back(rootOf(castOp.getIn()));
                    continue;
                }

                // a result that borrows an argument is somewhere under the argument
                if (auto *call = borrowingCall(base))
                {
                    chain.anyPlace = true;
                    auto args = callArgs(call);
                    for (auto index : resultBorrowsAll(call))
                    {
                        if (static_cast<size_t>(index) < args.size())
                        {
                            bases.push_back(rootOf(args[index]));
                        }
                    }

                    continue;
                }

                if (auto argument = mlir::dyn_cast<mlir::BlockArgument>(base); argument && mergedInto(argument, bases))
                {
                    continue;
                }

                // A `let` given only borrows (TSLANG 56) owns nothing: its assignments and its release
                // free nothing, so what is read through it lasts as long as what the `let` borrowed -
                // each value it is given, somewhere under those values' own roots. `c = c.next` in a
                // loop over a list is the case: the nodes are the list's, not `c`'s.
                auto slotValue = base.getDefiningOp<mlir_ts::VariableOp>() ? base : mlir::Value();
                if (auto slotLoad = slotValue ? mlir_ts::LoadOp() : slotLoadOf(base))
                {
                    slotValue = slotLoad.getReference();
                }

                auto borrowingLet = slotValue ? slotValue.getDefiningOp<mlir_ts::VariableOp>() : mlir_ts::VariableOp();
                if (borrowingLet && borrowsOnly(borrowingLet))
                {
                    chain.anyPlace = true;
                    bases.push_back(rootOf(borrowingLet.getInitializer()));
                    for (auto *user : slotValue.getUsers())
                    {
                        if (isAssignmentOf(user, borrowingLet))
                        {
                            bases.push_back(rootOf(mlir::cast<mlir_ts::StoreOp>(user).getValue()));
                        }
                    }
                }
                else if (slotValue)
                {
                    chain.roots.push_back({slotValue, Chain::Slot});
                }
                else
                {
                    chain.roots.push_back({base, isFresh(base) ? Chain::Owned : Chain::NotOwned});
                }
            }
        }

        return chain;
    }

    llvm::SmallVector<mlir::Operation *> OwnershipInferencePass::rootEnds(const Chain &chain)
    {
        llvm::SmallVector<mlir::Operation *> ends;
        for (auto [root, kind] : chain.roots)
        {
            // A handle (spec 23.3), never owned: where this one is given back or overwritten - a
            // local's release and assignments, a temporary's release; a global's assignments are
            // below. A copy of it is counted, and moves nothing away.
            if (llvm::is_contained(chain.handles, root))
            {
                auto loadOp = root.getDefiningOp<mlir_ts::LoadOp>();
                if (auto varOp = loadOp ? loadOp.getReference().getDefiningOp<mlir_ts::VariableOp>() : mlir_ts::VariableOp())
                {
                    for (auto *user : varOp->getUsers())
                    {
                        if (mlir::isa<mlir_ts::ReleaseSlotOp, mlir_ts::StoreOp>(user))
                        {
                            ends.push_back(user);
                        }
                    }
                }
                else if (!loadOp)
                {
                    forEachUse(root, [&](mlir::Operation *use, mlir::Value) {
                        if (mlir::isa<mlir_ts::ReleaseOp>(use))
                        {
                            ends.push_back(use);
                        }
                    });
                }
            }

            if (kind == Chain::Slot)
            {
                for (auto *user : root.getUsers())
                {
                    if (mlir::isa<mlir_ts::ReleaseSlotOp, mlir_ts::StoreOp>(user))
                    {
                        ends.push_back(user);
                    }
                    else if (auto loadOp = mlir::dyn_cast<mlir_ts::LoadOp>(user))
                    {
                        forEachUse(loadOp.getResult(), [&](mlir::Operation *use, mlir::Value used) {
                            // (a `let` that only borrows it takes nothing: `let node = head; ... node = node.next`)
                            mlir::Operation *from = nullptr;
                            if (localBorrowerOf(use, from) ||
                                (mlir::isa<mlir_ts::VariableOp>(use) && localBorrowingLets.contains(use)))
                            {
                                return;
                            }

                            auto varOp = mlir::dyn_cast<mlir_ts::VariableOp>(use);
                            if (mlir::isa<mlir_ts::RetainOp>(use) || (varOp && isOwningVariable(varOp)) ||
                                (isCall(use) && isKeptArgument(use, used)))
                            {
                                ends.push_back(use);
                            }
                        });
                    }
                }
            }
            else if (kind == Chain::Owned)
            {
                forEachUse(root, [&](mlir::Operation *use, mlir::Value used) {
                    if (mlir::isa<mlir_ts::ReleaseOp, mlir_ts::RetainOp>(use) || !isBorrow(use, used))
                    {
                        ends.push_back(use);
                    }
                });
            }
            else if (auto loadOp = root.getDefiningOp<mlir_ts::LoadOp>())
            {
                if (auto global = loadOp.getReference().getDefiningOp<mlir_ts::AddressOfOp>())
                {
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
                        if (other && other.getGlobalName() == global.getGlobalName())
                        {
                            ends.push_back(op);
                        }
                    });
                }
            }
        }

        return ends;
    }

    llvm::SmallVector<mlir::Value> OwnershipInferencePass::rootSources(const Chain &chain)
    {
        llvm::SmallVector<mlir::Value> sources;
        for (auto &root : chain.roots)
        {
            auto value = root.first;
            sources.push_back(value);
            for (auto steps = 0; steps < 8; ++steps)
            {
                auto varOp = value.getDefiningOp<mlir_ts::VariableOp>();
                if (!varOp || !varOp.getInitializer())
                {
                    break;
                }

                auto initializer = rootOf(varOp.getInitializer());
                auto slotLoad = slotLoadOf(initializer);
                value = slotLoad ? slotLoad.getReference() : initializer;
                sources.push_back(value);
            }
        }

        return sources;
    }

    const llvm::DenseSet<mlir::Value> &OwnershipInferencePass::derivedFrom(mlir::Value start)
    {
        if (auto found = derivedCache.find(start); found != derivedCache.end())
        {
            return *found->second;
        }

        auto &slot = derivedCache[start];
        slot = std::make_shared<llvm::DenseSet<mlir::Value>>();
        auto &derived = *slot;
        llvm::SmallVector<mlir::Value> work{start};
        while (!work.empty())
        {
            auto value = work.pop_back_val();
            if (!derived.insert(value).second)
            {
                continue;
            }

            for (auto &use : value.getUses())
            {
                auto *user = use.getOwner();
                if (isCall(user))
                {
                    // ...but a borrowing record given this value holds it (spec 32)
                    auto args = callArgs(user);
                    if (isRecordResult(user) && llvm::any_of(resultBorrowsAll(user), [&](int index) {
                            return static_cast<size_t>(index) < args.size() && args[index] == value;
                        }))
                    {
                        work.push_back(user->getResult(0));
                    }

                    continue;
                }

                llvm::SmallVector<mlir::Value> next(user->getResults().begin(), user->getResults().end());
                passesOn(use, next);
                for (auto result : next)
                {
                    if (mayPointInto(result))
                    {
                        work.push_back(result);
                    }
                }
            }
        }

        return derived;
    }

    bool OwnershipInferencePass::overwritesChain(int64_t position, mlir::Type slotType, const Chain &chain)
    {
        auto isField = [](int64_t at) { return at >= 0 || at == Chain::InterfaceField; };

        // A field reached through an interface may be any field of the object under it, of any
        // type (a class may declare the field narrower than the interface): an overwrite through
        // an interface drops every field on the chain, and any field's overwrite drops a field
        // read through an interface (spec 28).
        return llvm::any_of(chain.places, [&](auto &place) {
            if (isField(position) && isField(place.first) &&
                (position == Chain::InterfaceField || place.first == Chain::InterfaceField))
            {
                return true;
            }

            return place.first == position && place.second == slotType;
        });
    }

    bool OwnershipInferencePass::readsStableFieldsOfHeldRoot(mlir::Operation *read, const Chain &chain)
    {
        auto loadOp = mlir::dyn_cast<mlir_ts::LoadOp>(read);
        auto ref = loadOp ? loadOp.getReference() : mlir::Value();
        auto stableFields = 0;
        while (ref)
        {
            auto propertyRefOp = ref.getDefiningOp<mlir_ts::PropertyRefOp>();
            if (!propertyRefOp)
            {
                return false;
            }

            // the part of a subclass's object that is its base class's is the object itself, not a field
            auto refType = mlir::dyn_cast<mlir_ts::RefType>(propertyRefOp.getType());
            auto basePart = refType && mlir::isa<mlir_ts::ClassStorageType>(refType.getElementType());
            if (!basePart)
            {
                if (!isStableField(propertyRefOp.getResult()))
                {
                    return false;
                }

                stableFields++;
            }

            auto object = propertyRefOp.getObjectRef();
            if (object.getDefiningOp<mlir_ts::PropertyRefOp>())
            {
                ref = object;
                continue;
            }

            // a value this function made, held by no slot (a `const` kept as its value): it ends where it
            // is released or taken (rootEnds)
            auto objectLoad = object.getDefiningOp<mlir_ts::LoadOp>();
            if (!objectLoad)
            {
                return stableFields > 0 && llvm::any_of(chain.roots, [&](auto &root) {
                           return root.first == object && root.second == Chain::Owned;
                       });
            }

            if (objectLoad.getReference().getDefiningOp<mlir_ts::PropertyRefOp>())
            {
                ref = objectLoad.getReference();
                continue;
            }

            auto slot = objectLoad.getReference().getDefiningOp<mlir_ts::VariableOp>();
            if (!slot || stableFields == 0 || isCellVariable(slot.getResult()))
            {
                return false;
            }

            int index = -1;
            if (isParameterSlot(slot, index))
            {
                return isUnassignedSlot(slot.getResult());
            }

            return llvm::any_of(chain.roots, [&](auto &root) { return root.first == slot.getResult() && root.second == Chain::Slot; });
        }

        return false;
    }

    bool OwnershipInferencePass::dropsChain(mlir::Operation *drop, const Chain &chain, mlir::Operation *read, mlir::Value borrowed)
    {
        if (read && drop == read)
        {
            return false; // a call that returns a borrow has not destroyed it by returning it
        }

        // a chain whose places are not known (a borrowing call's result): what a release frees is what
        // the released value's type can hold, so one that cannot hold the borrowed value's type does not
        // free it (TSLANG 84)
        auto borrowedValue = borrowed ? borrowed : (read && read->getNumResults() == 1 ? read->getResult(0) : mlir::Value());
        auto module = drop->getParentOfType<mlir::ModuleOp>();
        if (chain.anyPlace && !isCall(drop))
        {
            if (auto releaseSlotOp = mlir::dyn_cast<mlir_ts::ReleaseSlotOp>(drop); releaseSlotOp && borrowedValue)
            {
                auto slotType = mlir::cast<mlir_ts::RefType>(releaseSlotOp.getSlot().getType()).getElementType();
                return typeMayHold(module, slotType, borrowedValue.getType());
            }

            return true;
        }

        // An exchange frees nothing, but what a borrow of either place reads is the other place's
        // now, under another root perhaps, which may end first (spec 42)
        if (auto swapOp = mlir::dyn_cast<mlir_ts::SwapOp>(drop))
        {
            return llvm::any_of(swapOp->getOperands(), [&](mlir::Value place) {
                return overwritesChain(swappedPosition(place), place.getType(), chain);
            });
        }

        if (auto releaseSlotOp = mlir::dyn_cast<mlir_ts::ReleaseSlotOp>(drop))
        {
            // An assignment to a captured variable destroys its old value, which is reached only
            // through a read of the variable's cell: a root this function does not own.
            auto slot = releaseSlotOp.getSlot();
            if (isCellVariable(slot) || isLoadedCell(slot))
            {
                return !chain.owned();
            }

            // An assignment to a global destroys its old value and all that value owns. A root this
            // function does not own - `this`, a parameter, a global's or a field's value - may be
            // that very block, or under it (spec 29). A root this function made cannot. An immortal
            // global's one store is its initializer, whose old value is the empty one (spec 25).
            if (auto addressOfOp = slot.getDefiningOp<mlir_ts::AddressOfOp>())
            {
                return !addressOfOp->hasAttr(OWN_IMMORTAL_GLOBAL_ATTR_NAME) && !chain.owned();
            }

            // `x.value = ...` through any handle, of any type, may replace the payload another
            // handle reads (spec 23.3): a handle converts to one of a base class's type
            if (slot.getDefiningOp<mlir_ts::SharedValueRefOp>())
            {
                return llvm::any_of(chain.places, [&](auto &place) { return place.first == Chain::SharedValue; });
            }

            int64_t position = -1;
            if (auto propertyRefOp = slot.getDefiningOp<mlir_ts::PropertyRefOp>())
            {
                position = propertyRefOp.getPosition();
            }
            else if (isInterfaceField(slot))
            {
                position = Chain::InterfaceField;
            }

            return overwritesChain(position, slot.getType(), chain);
        }

        if (isCall(drop))
        {
            if (callNoDrops(drop))
            {
                return false;
            }

            // the object a WeakRef refers to (spec 84): owned anywhere, so freed by any call that may
            // drop, unless all it overwrites are fields that cannot hold it
            if (read && mlir::isa<mlir_ts::WeakRefDerefOp>(read))
            {
                auto targetType = mlir::cast<mlir_ts::WeakRefType>(read->getOperand(0).getType()).getElementType();
                if (auto places = callDropFields(drop))
                {
                    return llvm::any_of(*places, [&](auto &place) {
                        auto refType = mlir::dyn_cast<mlir_ts::RefType>(place.second);
                        return !refType || typeMayHold(module, refType.getElementType(), targetType);
                    });
                }

                return true;
            }

            // stable fields - private ones nothing writes after construction (TSLANG 73), and `readonly` ones,
            // which only their class's constructor writes - read off a parameter, `this` or a local this
            // function owns, one after another (`o.mid.inner`): no call overwrites any of them, and each is
            // released only with its object. The caller holds a parameter's object while this function runs;
            // a local's ends where it is released, assigned or moved away (rootEnds), not inside a call that
            // borrows it. Not a local a closure may assign through its cell.
            if (read && !chain.anyPlace && readsStableFieldsOfHeldRoot(read, chain))
            {
                return false;
            }

            // a call given a closure may run it, and its box may hold anything
            auto givenClosure = llvm::any_of(drop->getOperands(), holdsCaptures);
            auto mayDrop = true;
            // A caught value is the exception's until the catch ends: a throw takes what it throws
            // (a borrow cannot be thrown), so nothing a call reaches holds it but what it is given
            auto caught = llvm::all_of(chain.roots, [&](auto &root) {
                return root.second != Chain::NotOwned || isCaughtValue(root.first);
            });
            if (caught && !givenClosure)
            {
                auto &fromRead = derivedFrom(borrowed ? borrowed : read->getResult(0));
                auto sources = rootSources(chain);
                mayDrop = llvm::any_of(drop->getOperands(), [&](mlir::Value operand) {
                    return !fromRead.contains(operand) && llvm::any_of(sources, [&](mlir::Value source) {
                               return derivedFrom(source).contains(operand);
                           });
                });
            }

            // Callees that destroy only the old values of places they overwrite drop the chain as
            // those stores would here. Not a chain whose places are not all known (a borrowing
            // call's result is somewhere under its arguments). A closure the call runs and does not
            // keep is one of its callback parameter's family, whose bodies its places count (spec 52)
            auto keepsClosure = givenClosure && llvm::any_of(drop->getOperands(), [&](mlir::Value operand) {
                                    return holdsCaptures(operand) && isKeptArgument(drop, operand);
                                });
            // ...and a chain whose places are not known, by the types those places hold (TSLANG 84)
            if (mayDrop && !keepsClosure && chain.anyPlace && borrowedValue)
            {
                if (auto places = callDropFields(drop))
                {
                    mayDrop = llvm::any_of(*places, [&](auto &place) {
                        if (place.first == OWN_CAPTURED_STRING_POSITION || place.first == OWN_CAPTURED_CELL_POSITION)
                        {
                            return true;
                        }

                        auto placeType = place.second;
                        if (auto refType = mlir::dyn_cast<mlir_ts::RefType>(placeType))
                        {
                            placeType = refType.getElementType();
                        }

                        return typeMayHold(module, placeType, borrowedValue.getType());
                    });
                }
            }

            if (mayDrop && !keepsClosure && !chain.anyPlace)
            {
                if (auto places = callDropFields(drop))
                {
                    // a captured string assigned frees that string only: a borrow of anything else
                    // is not it, nor under it (spec 52)
                    auto value = borrowed ? borrowed : (read ? read->getResult(0) : mlir::Value());
                    auto notString = value && mlir::isa<mlir_ts::ArrayType, mlir_ts::ClassType, mlir_ts::TupleType,
                                                        mlir_ts::ObjectType, mlir_ts::InterfaceType>(value.getType());
                    mayDrop = llvm::any_of(*places, [&](auto &place) {
                        if (place.first == OWN_CAPTURED_STRING_POSITION)
                        {
                            return !notString;
                        }

                        // a captured object assigned frees what only its cell owns: a chain read through
                        // a captured variable's cell may be under it, nothing else (OWN_CAPTURED_CELL_POSITION)
                        if (place.first == OWN_CAPTURED_CELL_POSITION)
                        {
                            return llvm::any_of(chain.roots, [](auto &root) {
                                auto loadOp = root.first.template getDefiningOp<mlir_ts::LoadOp>();
                                auto through = loadOp ? loadOp.getReference() : mlir::Value();
                                return isCellVariable(root.first) || isLoadedCell(root.first) ||
                                       (through && (isCellVariable(through) || isLoadedCell(through)));
                            });
                        }

                        return overwritesChain(place.first, place.second, chain);
                    });
                }
            }

            return mayDrop;
        }

        // pop, shift, splice, `length =`; push and unshift on an array reached through a handle
        auto refType = mlir::dyn_cast<mlir_ts::RefType>(drop->getOperand(0).getType());
        auto arrayType = refType ? mlir::dyn_cast<mlir_ts::ArrayType>(refType.getElementType()) : mlir_ts::ArrayType();
        if (!arrayType)
        {
            return true;
        }

        return llvm::any_of(chain.places, [&](auto &place) {
            auto placeType = mlir::dyn_cast<mlir_ts::RefType>(place.second);
            return place.first == -1 && placeType && placeType.getElementType() == arrayType.getElementType();
        });
    }

    bool OwnershipInferencePass::copiesInto(mlir::Operation *call, mlir::Value value)
    {
        // given once: a copy given twice would have two owners. Not an invoke, whose release after
        // it would go on the normal path only.
        return (isCopyableValueAt(call, value.getType()) || isHandle(value)) && !mlir::isa<mlir_ts::InvokeOp>(call) &&
               llvm::count(call->getOperands(), value) == 1;
    }

    bool OwnershipInferencePass::isAwaitResultStore(mlir_ts::StoreOp storeOp)
    {
        // `await f(x)` runs the call in an `async.execute` region, which the async lowering outlines
        // into a `func.func` given a reference to the caller's local, and stores the value through
        // it for the code after the await to read. The outlined function's temporaries (`x`) are
        // released when it ends, so what borrows them cannot be given back as it is.
        auto argument = mlir::dyn_cast<mlir::BlockArgument>(storeOp.getReference());
        auto *owner = argument ? argument.getOwner() : nullptr;
        return owner && owner->isEntryBlock() && owner->getParentOp() &&
               owner->getParentOp()->getName().getStringRef() == "func.func";
    }

    bool OwnershipInferencePass::storesCopyOf(mlir::Operation *store, mlir::Value value)
    {
        auto storeOp = mlir::dyn_cast<mlir_ts::StoreOp>(store);
        if (!storeOp || storeOp.getValue() != value || !(isCopyableValueAt(store, value.getType()) || isHandle(value)) ||
            !(isPlace(storeOp.getReference()) || isAwaitResultStore(storeOp)))
        {
            return false;
        }

        // a retained one is copyForRetain's: the retain becomes the copy
        return llvm::none_of(value.getUsers(), [](mlir::Operation *user) { return mlir::isa<mlir_ts::RetainOp>(user); });
    }

    bool OwnershipInferencePass::checkPlaceRead(mlir::Operation *read, llvm::SetVector<mlir::Operation *> &toErase)
    {
        auto loadOp = mlir::dyn_cast<mlir_ts::LoadOp>(read);
        auto place = loadOp ? describePlace(loadOp.getReference()) : describeCall(read);
        auto result = read->getResult(0);
        // A handle read rc counts (spec 23.3): its counted uses are copies (walkBorrowed), and the
        // release of the old value a counted store makes, right before it stores the retained
        // read, cannot free the block read: it holds the retain's count by then
        // (`Shared.count(q.child = p.child)`).
        auto countedHandle = isHandle(result) && handleReadUses(result).counted;
        llvm::SmallPtrSet<mlir::Operation *, 4> countedStoreReleases;
        if (countedHandle)
        {
            collectCountedStoreReleases(result, countedStoreReleases);
        }

        llvm::StringRef name;
        llvm::SmallVector<BorrowUse> uses;
        llvm::SmallVector<mlir::Operation *> bookkeeping;
        mlir_ts::VariableOp letKeeps;
        mlir::Operation *letKept = nullptr;
        // the local that borrows the read, and its declaration or the assignment that gives it the read
        mlir_ts::VariableOp assignedLet;
        mlir::Operation *assignment = nullptr;
        // A result that borrows arrives with rc's temporary release, or rc's consuming `let`, and
        // in a function returning a borrow of the same parameter it may go out with rc's retain.
        auto walk = [&]() {
            name = ownerName(result);
            uses.clear();
            bookkeeping.clear();
            letKeeps = {};
            letKept = nullptr;
            assignedLet = {};
            assignment = nullptr;
            Keeping keeping{&bookkeeping, returnsBorrowOf >= 0 && borrowedParam(result) == returnsBorrowOf};
            return walkBorrowed(result, read, uses, [&](mlir::Operation *user, llvm::ArrayRef<mlir::Operation *> kills) {
                auto borrower = borrowerOf(user, /*consumed=*/true);
                // (the store that declares a local declared in a `try` is its declaration)
                if (!borrower || (borrower.getOperation() != user && !declarationRetainOf(user)))
                {
                    // assigned to a `let` given only borrows (TSLANG 56): the `let` borrows it from the
                    // assignment on, and rc's retain for the assignment goes
                    auto storeOp = mlir::dyn_cast<mlir_ts::StoreOp>(user);
                    auto operand = storeOp ? storeOp.getValue()
                                           : (mlir::isa<mlir_ts::RetainOp>(user) ? user->getOperand(0) : mlir::Value());
                    auto assigned = operand && rootOf(operand) == rootOf(result) ? borrowingAssignmentOf(user, operand)
                                                                                 : mlir_ts::VariableOp();
                    if (!assigned)
                    {
                        return false;
                    }

                    if (!storeOp)
                    {
                        bookkeeping.push_back(user);
                        return true;
                    }

                    borrower = assigned;
                }

                assignedLet = borrower;
                assignment = user;

                name = varName(borrower);
                if (auto *keeps = collectBorrowerUses(borrower, kills, uses, bookkeeping, user); keeps && !letKept)
                {
                    letKept = keeps;
                    letKeeps = borrower;
                }

                return true;
            }, keeping, countedHandle);
        };

        auto *kept = walk();

        // A value taken out of a field of the object a place holds, by the store that gives the place
        // that value (`this.head = this.head.next`): the store's release frees the object, so the
        // field is emptied right before it and the value is the place's. Nothing else may keep the
        // value, or use it after the store through what was read here.
        if (FieldTake splice; !countedHandle && splicedFieldTakeOf(read, splice))
        {
            takenOps.insert(splice.retain);
            kept = walk();
            takenOps.clear();
            auto *after = kept ? kept : letKept;
            for (auto &use : uses)
            {
                if (!after && use.op != splice.retain && reachableAfter(splice.taker, use.op, use.kills))
                {
                    after = use.op;
                }
            }

            if (after)
            {
                reportPlaceBorrowEscapes(after, name, place);
                return false;
            }

            if (!dryRun)
            {
                // (an object an append-only global keeps too is not freed: it keeps its field, and the
                // value is made immortal - ts.TakeOutOfField)
                mlir::OpBuilder builder(splice.release);
                auto object = splice.field.getDefiningOp<mlir_ts::PropertyRefOp>().getObjectRef();
                builder.create<mlir_ts::TakeOutOfFieldOp>(splice.release->getLoc(), object, splice.field);
            }

            bookkeeping.push_back(splice.retain);
            toErase.insert(bookkeeping.begin(), bookkeeping.end());
            return true;
        }

        // A value taken out of a field of an owning record local by one move, the field read again
        // by nothing before the local ends (spec 48): the `value` of the record `next()` returns, in
        // a `for...of` or a spread that keeps what a generator yields
        // nothing is taken out of a field a back-reference is stored into (spec 70): what it holds may
        // borrow the object it is in. Such a read stays a borrow, bounded by the object
        auto readRef = mlir::isa<mlir_ts::LoadOp>(read) ? mlir::cast<mlir_ts::LoadOp>(read).getReference() : mlir::Value();
        // ...nor out of a field of an object another local still names after the read (`let n2 = n;
        // o.v = n2.v; ... n.v`, fuzz 956382): `n` is read through its new owner, and would read the
        // emptied field
        auto backReference = readRef && (isBackReferenceField(readRef) || aliasReadAfter(read, readRef));
        FieldTake take;
        if ((kept || letKept) && !countedHandle && !backReference && fieldTakeOf(read, kept ? kept : letKept, take))
        {
            takenOps.insert(take.retain);
            takenOps.insert(take.taker);
            kept = walk();
            takenOps.clear();
            if (kept || letKept || !fieldTakeHolds(read, take, uses))
            {
                take = {};
                kept = walk();
            }
        }

        // ...or out of a field of a call's own result, released as a temporary right after the read
        // (spec 63): `init ?? next(xs, 0).value`. The temporary frees nothing of it then, so nothing
        // after the move can see the release.
        if (!take.taker && kept && !countedHandle && !backReference && temporaryFieldTakeOf(read, kept, take))
        {
            takenOps.insert(take.retain);
            takenOps.insert(take.taker);
            kept = walk();
            takenOps.clear();
            auto usedAfter = llvm::any_of(uses, [&](auto &use) {
                return use.op != take.taker && use.op != take.retain && reachableAfter(take.taker, use.op, use.kills);
            });
            if (kept || letKept || usedAfter)
            {
                take = {};
                kept = walk();
            }
            else
            {
                if (!dryRun)
                {
                    // (right after the read: rc's retain of it is erased with the bookkeeping)
                    auto *at = take.emptyAtRead ? read->getNextNode() : take.release;
                    mlir::OpBuilder builder(at);
                    auto fieldType = mlir::cast<mlir_ts::RefType>(take.field.getType()).getElementType();
                    auto empty = builder.create<mlir_ts::DefaultOp>(at->getLoc(), fieldType);
                    builder.create<mlir_ts::StoreOp>(at->getLoc(), empty, take.field);
                }

                bookkeeping.push_back(take.retain);
                toErase.insert(bookkeeping.begin(), bookkeeping.end());
                return true;
            }
        }

        // ...or out of a place refilled before the taker has what it held (spec 77): `[data, this.data]
        // = [this.data, data]`. The refill's release would free it: the value is handed over instead,
        // and the refill leaves nothing of it in the place.
        mlir::Operation *refill = nullptr;
        if (!take.taker && kept && !countedHandle && !backReference && refilledFieldTakeOf(read, kept, take, refill))
        {
            takenOps.insert(take.retain);
            takenOps.insert(take.taker);
            kept = walk();
            takenOps.clear();
            auto usedAfter = llvm::any_of(uses, [&](auto &use) {
                return use.op != take.taker && use.op != take.retain && reachableAfter(take.taker, use.op, use.kills);
            });
            if (kept || letKept || usedAfter)
            {
                take = {};
                refill = nullptr;
                kept = walk();
            }
            else
            {
                bookkeeping.push_back(take.retain);
                bookkeeping.push_back(refill);
                toErase.insert(bookkeeping.begin(), bookkeeping.end());
                return true;
            }
        }

        // ...or given to a call that keeps it, whose result refills the field right after (`this.head =
        // new Node(v, this.head)`, `t.left = insert(t.left, v)`): the field is given up before the call,
        // and the refill's release frees what it holds then - nothing, unless the callee put something
        // there through another way to the object.
        if (FieldTake called; !take.taker && kept && !letKept && !countedHandle && !backReference &&
                              calledFieldTakeOf(read, kept, called))
        {
            takenOps.insert(called.taker);
            auto *stillKept = walk();
            takenOps.clear();
            auto usedAfter = llvm::any_of(uses, [&](auto &use) {
                return use.op != called.taker && reachableAfter(called.taker, use.op, use.kills);
            });
            if (!stillKept && !letKept && !usedAfter)
            {
                if (!dryRun)
                {
                    mlir::OpBuilder builder(called.taker);
                    auto object = called.field.getDefiningOp<mlir_ts::PropertyRefOp>().getObjectRef();
                    builder.create<mlir_ts::TakeOutOfFieldOp>(called.taker->getLoc(), object, called.field);
                }

                toErase.insert(bookkeeping.begin(), bookkeeping.end());
                return true;
            }

            kept = walk();
        }

        if (take.taker)
        {
            bookkeeping.push_back(take.retain);
        }
        else if (kept || letKept)
        {
            // a string given to a call that keeps it: the call takes a copy (spec 22)
            if (dryRun && kept && isCall(kept) && copiesInto(kept, result))
            {
                callCopies.push_back({read, kept});
            }

            // ...and one stored into a place rc does not retain it for, as a yield's record field
            // (`if (c) yield v` of a string `v` the generator's state holds): the place takes a copy,
            // or a handle a count (spec 50)
            if (dryRun && kept && storesCopyOf(kept, result))
            {
                callCopies.push_back({read, kept});
            }

            if (!kept && isAssignmentOf(letKept, letKeeps))
            {
                // a candidate for a holder of its fresh values (TSLANG 66)
                if (dryRun)
                {
                    splitLetCandidates.insert(letKeeps);
                }

                reportPlaceBorrowerAssigned(letKept, varName(letKeeps), place);
            }
            else
            {
                reportPlaceBorrowEscapes(kept ? kept : letKept, name, place);
            }

            return false;
        }

        auto chain = chainOf(read);
        // A call given the borrow is a use of it for as long as the callee runs, so a call that
        // may destroy it is a use after that, whatever the order of its arguments. The function
        // value it goes through is only read as it starts.
        auto outlives = [&](mlir::Operation *drop) {
            for (auto &use : uses)
            {
                if ((use.op != drop || !use.callee) && reachableAfter(drop, use.op, use.kills))
                {
                    // a string given to the very call that may release it, when the call keeps it, or
                    // is a callback - a function value, which nothing can say `@nodrops` of: the call
                    // gets a copy
                    auto callee = calleeValue(drop);
                    auto args = callArgs(drop);
                    auto keepsIt = llvm::any_of(ownedParams(drop), [&](int32_t index) {
                        return static_cast<size_t>(index) < args.size() && args[index] == result;
                    });
                    if (dryRun && use.op == drop && (keepsIt || (callee && callee.getDefiningOp<mlir_ts::LoadOp>())) &&
                        copiesInto(drop, result))
                    {
                        callCopies.push_back({read, drop});
                    }
                    // ...and, with --own-fallback=count, a read that cannot be copied: given a count
                    // of its own for the call, as a handle is, when the repair counts it
                    else if (dryRun && countsFallback && use.op == drop && !copiesInto(drop, result) &&
                             !mlir::isa<mlir_ts::InvokeOp>(drop) &&
                             llvm::count_if(drop->getOperands(), [&](mlir::Value operand) { return rootOf(operand) == result; }) == 1)
                    {
                        callCopies.push_back({read, drop});
                    }

                    // a @weak field's read given to the call that may free it (spec 93)
                    if (auto derefOp = mlir::dyn_cast<mlir_ts::WeakRefDerefOp>(read);
                        derefOp && derefOp.getChecked() && use.op == drop)
                    {
                        reportWeakFieldCallDrops(drop, derefOp);
                        return true;
                    }

                    reportPlaceBorrowOutlives(use.op, drop, name, place);
                    return true;
                }
            }

            return false;
        };

        for (auto *end : rootEnds(chain))
        {            if (outlives(end))
            {
                return false;
            }
        }

        if (keptParamOutlives(chain, outlives))
        {
            return false;
        }

        // `c = c.next` walks down: the `let` holds the value read from here on, and every value it is
        // given later is read out of the one it holds, so it is that value or one under it. A field
        // or an element of what the `let` holds, overwritten (`c.next = n` at the end of a list), is
        // then under the value read too, and frees nothing above it: an owner is never under what it
        // owns.
        // a place whose value is its object's own: an element of an array, or a field of a class
        // object that is no back-reference and borrows nothing. What is read out of one is under
        // the object, never above it
        auto owningPlace = [&](mlir::Operation *placeOp) {
            if (mlir::isa_and_nonnull<mlir_ts::ElementRefOp>(placeOp))
            {
                return true;
            }

            auto propertyRefOp = mlir::dyn_cast_or_null<mlir_ts::PropertyRefOp>(placeOp);
            return propertyRefOp && mlir::isa<mlir_ts::ClassType>(propertyRefOp->getOperand(0).getType()) &&
                   !isBackReferenceField(propertyRefOp.getResult()) && !isBorrowingField(propertyRefOp.getResult());
        };
        // is `value` the value read (or, with `underLet`, what the local holds), or read out of it
        // through such places; `letRead` is then that read of the local
        auto readUnder = [&](mlir::Value value, bool underLet, mlir::Operation **letRead = nullptr) {
            for (auto root = rootOf(value); root;)
            {
                if (!underLet && root == result)
                {
                    return true;
                }

                auto loadOp = root.getDefiningOp<mlir_ts::LoadOp>();
                if (!loadOp || isHandle(root))
                {
                    return false;
                }

                if (underLet && loadOp.getReference() == assignedLet.getResult())
                {
                    if (letRead)
                    {
                        *letRead = loadOp;
                    }

                    return true;
                }

                auto *placeOp = loadOp.getReference().getDefiningOp();
                if (!owningPlace(placeOp))
                {
                    return false;
                }

                root = rootOf(placeOp->getOperand(0));
            }

            return false;
        };
        auto onlyWalksDown = [&]() {
            // declared anew after it is given the read (a declaration in a loop): given its
            // initializer again
            auto *after = assignedLet && assignment == assignedLet.getOperation() ? assignment->getNextNode() : assignment;
            if (!after || reachableAfter(after, assignedLet, llvm::ArrayRef<mlir::Operation *>()))
            {
                return false;
            }

            return llvm::all_of(assignedLet.getResult().getUsers(), [&](mlir::Operation *user) {
                return !isAssignmentOf(user, assignedLet) ||
                       !(user == assignment || reachableAfter(assignment, user, llvm::ArrayRef<mlir::Operation *>())) ||
                       readUnder(mlir::cast<mlir_ts::StoreOp>(user).getValue(), true);
            });
        };
        auto walksDown = -1;
        auto frees = [&](mlir::Operation *drop) {
            auto releaseSlotOp = mlir::dyn_cast<mlir_ts::ReleaseSlotOp>(drop);
            auto *placeOp = releaseSlotOp ? releaseSlotOp.getSlot().getDefiningOp() : nullptr;
            if (!mlir::isa_and_nonnull<mlir_ts::PropertyRefOp, mlir_ts::ElementRefOp>(placeOp))
            {
                return true;
            }

            // a field or an element of the value read itself (`if (c) c.next = null`), or of a value
            // under it (`const d = c.next; if (d) d.next = null`)
            auto object = placeOp->getOperand(0);
            if (readUnder(object, false))
            {
                return false;
            }

            mlir::Operation *letRead = nullptr;
            if (!assignedLet || !readUnder(object, true, &letRead))
            {
                return true;
            }

            // ...when the local is read for it after it is given the value read: every way from the
            // read to that read of the local runs the assignment, and none from there reaches the
            // drop through the read again. `const nx = n.next; n.next = null; n = nx` reads `n`
            // while it still holds the object above, and frees what `nx` borrows.
            if (!letRead || reachableAfter(read, letRead, llvm::ArrayRef<mlir::Operation *>{assignment}) ||
                (reachableAfter(letRead, read, llvm::ArrayRef<mlir::Operation *>()) &&
                 reachableAfter(read, drop, llvm::ArrayRef<mlir::Operation *>{letRead})))
            {
                return true;
            }

            if (walksDown < 0)
            {
                walksDown = onlyWalksDown() ? 1 : 0;
            }

            return walksDown == 0;
        };

        for (auto *drop : drops)
        {
            if (countedStoreReleases.contains(drop))
            {
                continue;
            }

            if (frees(drop) && dropsChain(drop, chain, read) && outlives(drop))
            {
                return false;
            }
        }

        // The object a WeakRef refers to may be any owner's, a local's too: a release of a local or
        // of a temporary that may hold its type ends the borrow as well (spec 84). Not a WeakRef's.
        if (mlir::isa<mlir_ts::WeakRefDerefOp>(read))
        {
            auto targetType = mlir::cast<mlir_ts::WeakRefType>(read->getOperand(0).getType()).getElementType();
            auto module = read->getParentOfType<mlir::ModuleOp>();
            auto ended = false;
            getFunction()->walk([&](mlir::Operation *op) {
                if (ended)
                {
                    return;
                }

                mlir::Type held;
                if (auto releaseSlotOp = mlir::dyn_cast<mlir_ts::ReleaseSlotOp>(op))
                {
                    held = mlir::cast<mlir_ts::RefType>(releaseSlotOp.getSlot().getType()).getElementType();
                }
                else if (auto releaseOp = mlir::dyn_cast<mlir_ts::ReleaseOp>(op))
                {
                    held = releaseOp.getReference().getType();
                }

                if (!held || MLIRTypeHelper::isWeakRefType(held) || !typeMayHold(module, held, targetType))
                {
                    return;
                }

                ended = outlives(op);
            });

            if (ended)
            {
                return false;
            }
        }

        // the taker has the value now: the field is left empty, so the local's release frees nothing
        // of it
        if (take.taker && !dryRun)
        {
            mlir::OpBuilder builder(take.taker->getContext());
            builder.setInsertionPointAfter(take.taker);
            auto fieldType = mlir::cast<mlir_ts::RefType>(take.field.getType()).getElementType();
            auto empty = builder.create<mlir_ts::DefaultOp>(take.taker->getLoc(), fieldType);
            builder.create<mlir_ts::StoreOp>(take.taker->getLoc(), empty, take.field);
        }

        toErase.insert(bookkeeping.begin(), bookkeeping.end());
        return true;
    }

    bool OwnershipInferencePass::fieldTakeOf(mlir::Operation *read, mlir::Operation *kept, FieldTake &take)
    {
        // a field of an owning local that holds a record (a tuple), read in place: the local's release
        // frees what the field holds, so a move out of it leaves the field empty. Not a string, which
        // is copied where it is kept (spec 22).
        auto loadOp = mlir::dyn_cast<mlir_ts::LoadOp>(read);
        auto field = loadOp ? loadOp.getReference().getDefiningOp<mlir_ts::PropertyRefOp>() : mlir_ts::PropertyRefOp();
        auto slot = field ? field.getObjectRef().getDefiningOp<mlir_ts::VariableOp>() : mlir_ts::VariableOp();
        auto slotType = slot ? mlir::dyn_cast<mlir_ts::RefType>(slot.getType()) : mlir_ts::RefType();
        // MLIRGen makes no owner of a local at a script's top level or in a `try`: nothing releases
        // what it holds. A move out of it frees nothing twice when every value it is given is a call's
        // own result, not a borrow of a record someone else holds.
        auto ownsOrHoldsResults = [&]() {
            if (isOwningVariable(slot))
            {
                return true;
            }

            llvm::SmallVector<mlir::Value> given;
            if (slot.getInitializer())
            {
                given.push_back(slot.getInitializer());
            }

            for (auto *user : slot->getUsers())
            {
                if (auto storeOp = mlir::dyn_cast<mlir_ts::StoreOp>(user); storeOp && storeOp.getReference() == slot.getResult())
                {
                    given.push_back(storeOp.getValue());
                }
            }

            return !given.empty() && llvm::all_of(given, [&](mlir::Value value) {
                auto *def = value.getDefiningOp();
                return def && isCall(def) && !borrowingCall(value);
            });
        };

        if (!slot || slot.getCaptured().value_or(false) || !ownsOrHoldsResults() || !slotType ||
            !mlir::isa<mlir_ts::TupleType>(slotType.getElementType()) || isCopyableValueAt(read, read->getResult(0).getType()))
        {
            return false;
        }

        // rc's retain for the taker, and the taker: the retain's value is the read, or a read of a
        // local that borrows it (`for (const v of g()) a.push(v)`)
        // ...or a store into an owning local declared without a value (in a `try`), and rc's
        // `ts.RetainSlot` of it: the local owns what it is given, and its own check decides the rest
        if (auto storeOp = mlir::dyn_cast<mlir_ts::StoreOp>(kept); storeOp && rootOf(storeOp.getValue()) == read->getResult(0))
        {
            auto owner = storeOp.getReference().getDefiningOp<mlir_ts::VariableOp>();
            auto *retainSlot = owner && isOwningVariable(owner) && !owner.getCaptured().value_or(false) && !owner.getInitializer()
                                   ? onlyRetainSlotOf(owner)
                                   : nullptr;
            if (!retainSlot || retainSlot->getBlock() != storeOp->getBlock() || !storeOp->isBeforeInBlock(retainSlot))
            {
                return false;
            }

            take.retain = retainSlot;
            take.taker = storeOp;
            take.field = field.getResult();
            take.slot = slot;
            return true;
        }

        auto retainOp = mlir::dyn_cast<mlir_ts::RetainOp>(kept);
        mlir::Value value;
        if (retainOp)
        {
            value = retainOp.getReference();
        }
        else
        {
            for (auto operand : kept->getOperands())
            {
                for (auto *user : rootOf(operand).getUsers())
                {
                    if (auto found = mlir::dyn_cast<mlir_ts::RetainOp>(user); found && !retainOp)
                    {
                        retainOp = found;
                        value = found.getReference();
                    }
                }
            }
        }

        auto *taker = retainOp ? takerOf(retainOp, value) : nullptr;
        if (!taker || (!retainOp && taker != kept))
        {
            return false;
        }

        // a move into a container: an item pushed, a store into a place or a global, a call that keeps it
        auto movesIn = false;
        if (auto pushOp = mlir::dyn_cast<mlir_ts::ArrayPushOp>(taker))
        {
            movesIn = llvm::is_contained(pushOp.getItems(), value);
        }
        else if (auto storeOp = mlir::dyn_cast<mlir_ts::StoreOp>(taker))
        {
            movesIn = storeOp.getValue() == value &&
                      (isPlace(storeOp.getReference()) || storeOp.getReference().getDefiningOp<mlir_ts::AddressOfOp>());
        }
        else if (isCall(taker))
        {
            movesIn = isKeptArgument(taker, value);
        }

        if (!movesIn || retainOp->getBlock() != taker->getBlock() || !retainOp->isBeforeInBlock(taker))
        {
            return false;
        }

        take.retain = retainOp;
        take.taker = taker;
        take.field = field.getResult();
        take.slot = slot;
        return true;
    }

    bool OwnershipInferencePass::fieldTakeHolds(mlir::Operation *read, const FieldTake &take, llvm::ArrayRef<BorrowUse> uses)
    {
        auto slot = take.slot;
        auto ends = slotEnds(slot.getResult());
        // the taker runs once for each value read: never again before the field is read anew
        llvm::SmallVector<mlir::Operation *> again(ends.begin(), ends.end());
        again.push_back(read);
        // (from the op after it, around a loop: reachableAfter takes an op and itself for reachable)
        auto *next = take.taker->getNextNode();
        if (!next || reachableAfter(next, take.taker, again))
        {
            return false;
        }

        // nothing uses the borrow after the taker has it
        for (auto &use : uses)
        {
            if (use.op != take.taker && use.op != take.retain && reachableAfter(take.taker, use.op, use.kills))
            {
                return false;
            }
        }

        // and nothing reads the field, or the whole record, before the local ends
        auto position = mlir::cast<mlir_ts::PropertyRefOp>(take.field.getDefiningOp()).getPosition();
        for (auto *user : slot->getUsers())
        {
            llvm::SmallVector<mlir::Operation *> reads;
            if (mlir::isa<mlir_ts::LoadOp>(user))
            {
                reads.push_back(user);
            }
            else if (auto propertyRefOp = mlir::dyn_cast<mlir_ts::PropertyRefOp>(user);
                     propertyRefOp && propertyRefOp.getPosition() == position)
            {
                reads.append(propertyRefOp->getUsers().begin(), propertyRefOp->getUsers().end());
            }

            for (auto *fieldRead : reads)
            {
                if (reachableAfter(take.taker, fieldRead, ends))
                {
                    return false;
                }
            }
        }

        return true;
    }

    bool OwnershipInferencePass::temporaryFieldTakeOf(mlir::Operation *read, mlir::Operation *kept, FieldTake &take)
    {
        // rc's retain of the read itself, right after it: nothing runs between the read and the move. The
        // walk may meet the taker first (`r = next(xs, 0).value`). A string too: it moves, as nothing
        // else holds it.
        auto loadOp = mlir::dyn_cast<mlir_ts::LoadOp>(read);
        auto field = loadOp ? loadOp.getReference().getDefiningOp<mlir_ts::PropertyRefOp>() : mlir_ts::PropertyRefOp();
        auto temporary = field ? field.getObjectRef() : mlir::Value();
        auto *call = temporary ? temporary.getDefiningOp() : nullptr;
        if (!call || !field->hasOneUse())
        {
            return false;
        }

        // ...or out of an object a `const` names, `new`'s or a call's own (spec 64): `return r.value`.
        // The read may be used before rc's retain of it, later in its block (`const v = r.value;
        // print(v.n); return v`): the field is empty from the read on, and the read holds what it held.
        // An owning `let` the same (spec 65): `let r = new Holder(); return r.value`.
        auto slotRead = mlir::dyn_cast<mlir_ts::LoadOp>(call);
        auto slot = slotRead ? slotRead.getReference().getDefiningOp<mlir_ts::VariableOp>() : mlir_ts::VariableOp();
        if (call->hasAttr(OWNED_RESULT_NAMED_ATTR_NAME) || (slot && isOwningVariable(slot)))
        {
            for (auto *user : loadOp->getUsers())
            {
                auto namedRetain = mlir::dyn_cast<mlir_ts::RetainOp>(user);
                if (namedRetain && namedRetain->getBlock() == read->getBlock() &&
                    (kept == namedRetain || takerOf(namedRetain, loadOp.getResult()) == kept))
                {
                    return namedFieldTakeOf(read, namedRetain, field, take);
                }
            }

            return false;
        }

        auto retainOp = mlir::dyn_cast_or_null<mlir_ts::RetainOp>(read->getNextNode());
        if (!retainOp || retainOp.getReference() != loadOp.getResult() ||
            (kept != retainOp && takerOf(retainOp, loadOp.getResult()) != kept))
        {
            return false;
        }

        kept = retainOp;

        // a field read in place out of a call's own result, not a borrow of what the call was given

        if (!callResultIsOwn(call))
        {
            return false;
        }

        // released once, as a temporary, after the move: the release is the only end of what the field
        // holds, and no other read of the result reaches that field
        mlir::Operation *release = nullptr;
        for (auto *user : temporary.getUsers())
        {
            if (mlir::isa<mlir_ts::ReleaseOp>(user) && !release)
            {
                release = user;
                continue;
            }

            auto propertyRefOp = mlir::dyn_cast<mlir_ts::PropertyRefOp>(user);
            if (!propertyRefOp || (propertyRefOp != field && propertyRefOp.getPosition() == field.getPosition()))
            {
                return false;
            }
        }

        if (!release || release->getBlock() != kept->getBlock() || !kept->isBeforeInBlock(release))
        {
            return false;
        }

        auto *taker = takerOf(kept, loadOp.getResult());
        if (!taker)
        {
            return false;
        }

        take.retain = kept;
        take.taker = taker;
        take.field = field.getResult();
        take.release = release;
        return true;
    }

    bool OwnershipInferencePass::refilledFieldTakeOf(mlir::Operation *read, mlir::Operation *kept, FieldTake &take,
                                                     mlir::Operation *&refill)
    {
        // a read of a field or an element, of any object: what the place held goes to the taker, the
        // place is given another value before that, and its release then frees nothing anyone keeps.
        // Not a string, which is copied where it is kept (spec 22), nor a handle, which is counted.
        auto loadOp = mlir::dyn_cast<mlir_ts::LoadOp>(read);
        auto place = loadOp ? loadOp.getReference() : mlir::Value();
        if (!place || !mlir::isa_and_nonnull<mlir_ts::PropertyRefOp, mlir_ts::ElementRefOp>(place.getDefiningOp()) ||
            isCopyableValueAt(read, loadOp.getType()) || isHandle(loadOp.getResult()))
        {
            return false;
        }

        // rc's retain of the read, in its block
        mlir_ts::RetainOp retainOp;
        for (auto *user : loadOp->getUsers())
        {
            if (auto found = mlir::dyn_cast<mlir_ts::RetainOp>(user); found && found->getBlock() == read->getBlock())
            {
                if (retainOp)
                {
                    return false;
                }

                retainOp = found;
            }
        }

        auto *taker = retainOp ? takerOf(retainOp, loadOp.getResult()) : nullptr;
        if (!taker || (kept != retainOp && kept != taker) || taker->getBlock() != read->getBlock())
        {
            return false;
        }

        // the taker: a store into a local that owns what it holds, or into a place
        auto storeOp = mlir::dyn_cast<mlir_ts::StoreOp>(taker);
        auto slot = storeOp ? storeOp.getReference().getDefiningOp<mlir_ts::VariableOp>() : mlir_ts::VariableOp();
        auto intoLocal = slot && isOwningVariable(slot) && !slot.getCaptured().value_or(false);
        if (!storeOp || storeOp.getValue() != loadOp.getResult() || (!intoLocal && !isPlace(storeOp.getReference())))
        {
            return false;
        }

        // From the read to the taker, in order: rc's retain, then the place's release right before a
        // store into it - the refill - and nothing but reads, counts and stores of locals besides
        // (no call, which might read the place or drop the object).
        auto *refillRelease = static_cast<mlir::Operation *>(nullptr);
        auto retained = false;
        for (auto *op = read->getNextNode(); op && op != taker; op = op->getNextNode())
        {
            if (op == retainOp.getOperation())
            {
                retained = true;
                continue;
            }

            if (auto releaseSlotOp = mlir::dyn_cast<mlir_ts::ReleaseSlotOp>(op); releaseSlotOp && releaseSlotOp.getSlot() == place)
            {
                auto refillStore = mlir::dyn_cast_or_null<mlir_ts::StoreOp>(op->getNextNode());
                if (refillRelease || !retained || !refillStore || refillStore.getReference() != place)
                {
                    return false;
                }

                refillRelease = op;
                continue;
            }

            // the release of what the taker's place held, right before the store into it (TSLANG 49)
            if (auto releaseSlotOp = mlir::dyn_cast<mlir_ts::ReleaseSlotOp>(op);
                releaseSlotOp && !intoLocal && releaseSlotOp.getSlot() == storeOp.getReference() && op->getNextNode() == taker)
            {
                continue;
            }

            if (auto releaseSlotOp = mlir::dyn_cast<mlir_ts::ReleaseSlotOp>(op))
            {
                // a local's own release (the taker's assignment), never of the place's object
                auto local = releaseSlotOp.getSlot().getDefiningOp<mlir_ts::VariableOp>();
                auto objectRead = rootOf(place.getDefiningOp()->getOperand(0)).getDefiningOp<mlir_ts::LoadOp>();
                if (!local || local.getCaptured().value_or(false) ||
                    (objectRead && objectRead.getReference() == local.getResult()))
                {
                    return false;
                }

                continue;
            }

            if (auto innerStore = mlir::dyn_cast<mlir_ts::StoreOp>(op))
            {
                auto local = innerStore.getReference().getDefiningOp<mlir_ts::VariableOp>();
                if (innerStore.getReference() != place && !local)
                {
                    return false;
                }

                continue;
            }

            if (mlir::isa<mlir_ts::LoadOp, mlir_ts::RetainOp, mlir_ts::PropertyRefOp, mlir_ts::ConstantOp>(op) || isView(op))
            {
                continue;
            }

            return false;
        }

        // ...or refilled right after the taker (TSLANG 49): `this.other = this.level; this.level = null;`
        // moves the field into another. The same field of the same object, its release right before
        // the store that refills it, with only reads, constants and views between - nothing that could
        // free or read what both places hold until then.
        if (!refillRelease && retained)
        {
            auto field = place.getDefiningOp<mlir_ts::PropertyRefOp>();
            auto sameFieldAs = [&](mlir::Value slot) {
                auto other = slot.getDefiningOp<mlir_ts::PropertyRefOp>();
                return other && sameField(other, field);
            };
            // a retain followed by the release of the field and the store of the retained value into it
            auto refillsWith = [](mlir::Operation *retain, llvm::function_ref<bool(mlir::Value)> isField) {
                auto releaseSlotOp = mlir::dyn_cast_or_null<mlir_ts::ReleaseSlotOp>(retain->getNextNode());
                auto storeOp = releaseSlotOp ? mlir::dyn_cast_or_null<mlir_ts::StoreOp>(releaseSlotOp->getNextNode()) : mlir_ts::StoreOp();
                return storeOp && isField(releaseSlotOp.getSlot()) && storeOp.getReference() == releaseSlotOp.getSlot() &&
                       storeOp.getValue() == retain->getOperand(0);
            };
            for (auto *op = field ? taker->getNextNode() : nullptr; op; op = op->getNextNode())
            {
                if (auto releaseSlotOp = mlir::dyn_cast<mlir_ts::ReleaseSlotOp>(op))
                {
                    auto refillField = releaseSlotOp.getSlot().getDefiningOp<mlir_ts::PropertyRefOp>();
                    auto refillStore = mlir::dyn_cast_or_null<mlir_ts::StoreOp>(op->getNextNode());
                    if (refillField && sameField(refillField, field) && refillStore &&
                        refillStore.getReference() == releaseSlotOp.getSlot())
                    {
                        refillRelease = op;
                    }

                    break;
                }

                // rc's retain of the refill's value: one that holds no block (`null`), or the value the
                // refill stores (`n.next = this.head; this.head = n;`), which is checked where it is stored
                auto retainsNothing = mlir::isa<mlir_ts::RetainOp>(op) &&
                                      (holdsNoBlock(op->getOperand(0)) || refillsWith(op, sameFieldAs));
                // ...and that value, `null` cast to the field's union
                auto castsNothing = mlir::isa<mlir_ts::CastOp>(op) && holdsNoBlock(op->getResult(0));
                if (!mlir::isa<mlir_ts::LoadOp, mlir_ts::PropertyRefOp, mlir_ts::ConstantOp, mlir_ts::NullOp,
                               mlir_ts::UndefOp, mlir_ts::OptionalUndefOp>(op) &&
                    !isView(op) && !retainsNothing && !castsNothing)
                {
                    break;
                }
            }
        }

        if (!refillRelease)
        {
            return false;
        }

        take.retain = retainOp;
        take.taker = taker;
        take.field = place;
        refill = refillRelease;
        return true;
    }

    bool OwnershipInferencePass::splicedFieldTakeOf(mlir::Operation *read, FieldTake &take)
    {
        // a field of a class object that owns what it holds: not a string, which is copied where it is
        // kept (spec 22), nor a handle, which is counted
        auto loadOp = mlir::dyn_cast<mlir_ts::LoadOp>(read);
        auto field = loadOp ? loadOp.getReference().getDefiningOp<mlir_ts::PropertyRefOp>() : mlir_ts::PropertyRefOp();
        if (!field || !mlir::isa<mlir_ts::ClassType>(field.getObjectRef().getType()) || isBackReferenceField(field.getResult()) ||
            isBorrowingField(field.getResult()) || isCopyableValueAt(read, loadOp.getType()) || isHandle(loadOp.getResult()))
        {
            return false;
        }

        // ...that holds a class object or null: one pointer, which ts.TakeOutOfField empties
        auto holdsObjectOrNull = mlir::isa<mlir_ts::ClassType>(loadOp.getType());
        if (auto unionType = mlir::dyn_cast<mlir_ts::UnionType>(loadOp.getType()))
        {
            auto types = unionType.getTypes();
            holdsObjectOrNull = llvm::all_of(types, [](mlir::Type type) { return mlir::isa<mlir_ts::ClassType, mlir_ts::NullType>(type); }) &&
                                llvm::count_if(types, [](mlir::Type type) { return mlir::isa<mlir_ts::ClassType>(type); }) == 1;
        }

        if (!holdsObjectOrNull)
        {
            return false;
        }

        // rc's retain of the read, in its block, and the store it is for
        mlir_ts::RetainOp retainOp;
        for (auto *user : loadOp->getUsers())
        {
            if (auto found = mlir::dyn_cast<mlir_ts::RetainOp>(user); found && found->getBlock() == read->getBlock())
            {
                if (retainOp)
                {
                    return false;
                }

                retainOp = found;
            }
        }

        auto storeOp = mlir::dyn_cast_or_null<mlir_ts::StoreOp>(retainOp ? takerOf(retainOp, loadOp.getResult()) : nullptr);
        if (!storeOp || storeOp.getValue() != loadOp.getResult() || storeOp->getBlock() != read->getBlock())
        {
            return false;
        }

        // the store's place, released right before it: a field of a class object that owns what it
        // holds, so that the release frees the object read from
        auto place = storeOp.getReference().getDefiningOp<mlir_ts::PropertyRefOp>();
        auto release = mlir::dyn_cast_or_null<mlir_ts::ReleaseSlotOp>(storeOp->getPrevNode());
        if (!place || !release || release.getSlot() != storeOp.getReference() ||
            !mlir::isa<mlir_ts::ClassType>(place.getObjectRef().getType()) || isBackReferenceField(place.getResult()) ||
            isBorrowingField(place.getResult()))
        {
            return false;
        }

        // the object read from is what that place holds: a read of the same field of the same object
        // (the borrow check of that read sees to it that the place still holds it here)
        auto held = rootOf(field.getObjectRef()).getDefiningOp<mlir_ts::LoadOp>();
        auto heldPlace = held ? held.getReference().getDefiningOp<mlir_ts::PropertyRefOp>() : mlir_ts::PropertyRefOp();
        if (!heldPlace || isHandle(held.getResult()) || heldPlace.getPosition() != place.getPosition() ||
            heldPlace.getType() != place.getType() ||
            !sameObjectInBlock(rootOf(heldPlace.getObjectRef()), rootOf(place.getObjectRef())))
        {
            return false;
        }

        // from the read to the store: reads, rc's retain and the release, nothing that runs other code
        for (auto *op = read->getNextNode(); op && op != storeOp.getOperation(); op = op->getNextNode())
        {
            if (op != release.getOperation() &&
                !mlir::isa<mlir_ts::LoadOp, mlir_ts::RetainOp, mlir_ts::PropertyRefOp, mlir_ts::ConstantOp>(op) && !isView(op))
            {
                return false;
            }
        }

        take.retain = retainOp;
        take.taker = storeOp;
        take.field = field.getResult();
        take.release = release;
        return true;
    }

    bool OwnershipInferencePass::sameObjectInBlock(mlir::Value object, mlir::Value other)
    {
        if (sameObject(object, other))
        {
            return true;
        }

        // two reads of one local in a block, with no assignment of it between
        auto first = object.getDefiningOp<mlir_ts::LoadOp>();
        auto second = other.getDefiningOp<mlir_ts::LoadOp>();
        if (!first || !second || first.getReference() != second.getReference() ||
            !first.getReference().getDefiningOp<mlir_ts::VariableOp>() || first->getBlock() != second->getBlock())
        {
            return false;
        }

        if (!first->isBeforeInBlock(second))
        {
            std::swap(first, second);
        }

        for (auto *op = first->getNextNode(); op && op != second.getOperation(); op = op->getNextNode())
        {
            if (auto assigned = mlir::dyn_cast<mlir_ts::StoreOp>(op); assigned && assigned.getReference() == first.getReference())
            {
                return false;
            }

            if (op->getNumRegions() > 0 || isCall(op))
            {
                return false;
            }
        }

        return true;
    }

    bool OwnershipInferencePass::calledFieldTakeOf(mlir::Operation *read, mlir::Operation *kept, FieldTake &take)
    {
        // a field of a class object that owns what it holds: not a string, which is copied where it is
        // kept (spec 22), nor a handle, which is counted, nor a field a back-reference or a borrow is
        // stored into
        auto loadOp = mlir::dyn_cast<mlir_ts::LoadOp>(read);
        auto field = loadOp ? loadOp.getReference().getDefiningOp<mlir_ts::PropertyRefOp>() : mlir_ts::PropertyRefOp();
        if (!field || !mlir::isa<mlir_ts::ClassType>(field.getObjectRef().getType()) || isBackReferenceField(field.getResult()) ||
            isBorrowingField(field.getResult()) || isCopyableValueAt(read, loadOp.getType()) || isHandle(loadOp.getResult()))
        {
            return false;
        }

        // ...that holds a class object or null: one pointer, which ts.TakeOutOfField empties
        auto holdsObjectOrNull = mlir::isa<mlir_ts::ClassType>(loadOp.getType());
        if (auto unionType = mlir::dyn_cast<mlir_ts::UnionType>(loadOp.getType()))
        {
            auto types = unionType.getTypes();
            holdsObjectOrNull = llvm::all_of(types, [](mlir::Type type) { return mlir::isa<mlir_ts::ClassType, mlir_ts::NullType>(type); }) &&
                                llvm::count_if(types, [](mlir::Type type) { return mlir::isa<mlir_ts::ClassType>(type); }) == 1;
        }

        if (!holdsObjectOrNull)
        {
            return false;
        }

        // The call keeps the read, given it once and nothing else uses it. One that never throws: a
        // throw caught anywhere would find the field given up, where the other models keep it.
        if (!kept || !isCall(kept) || !callNoThrow(kept) || mlir::isa<mlir_ts::InvokeOp, mlir_ts::InvokeHybridOp>(kept) ||
            kept->getBlock() != read->getBlock() || !read->isBeforeInBlock(kept) || !loadOp->hasOneUse() ||
            *loadOp->user_begin() != kept || !isKeptArgument(kept, loadOp.getResult()))
        {
            return false;
        }

        // Its other arguments hold nothing the callee could reach the value by: values with no block,
        // and the object a `new` made for a constructor. Not the object the field is in (`this.wrap(
        // this.head)`), nor another read of the field (`f(t.l, t.l.r)`).
        for (auto arg : callArgs(kept))
        {
            if (arg == loadOp.getResult() || holdsNoBlock(arg) || isCopyableValueAt(kept, arg.getType()))
            {
                continue;
            }

            auto *def = rootOf(arg).getDefiningOp();
            if (!mlir::isa_and_nonnull<mlir_ts::NewOp>(def))
            {
                return false;
            }
        }

        // From the read to the call: reads, constants and views of the arguments - nothing that writes
        // or runs other code.
        for (auto *op = read->getNextNode(); op && op != kept; op = op->getNextNode())
        {
            if (!mlir::isa<mlir_ts::LoadOp, mlir_ts::PropertyRefOp, mlir_ts::ConstantOp>(op) && !isView(op))
            {
                return false;
            }
        }

        // From the call to the refill: views of its result and rc's retain of them, then the release
        // of the same field of the same object right before the store of the result into it. A
        // constructor's result is the object `new` made, its first argument.
        auto args = callArgs(kept);
        auto callResult = kept->getNumResults() == 1                                   ? kept->getResult(0)
                          : kept->getNumResults() == 0 && !args.empty() &&
                                    mlir::isa_and_nonnull<mlir_ts::NewOp>(args.front().getDefiningOp()) ? args.front()
                                                                                                         : mlir::Value();
        if (!callResult)
        {
            return false;
        }

        for (auto *op = kept->getNextNode(); op; op = op->getNextNode())
        {
            if (auto releaseSlotOp = mlir::dyn_cast<mlir_ts::ReleaseSlotOp>(op))
            {
                auto place = releaseSlotOp.getSlot().getDefiningOp<mlir_ts::PropertyRefOp>();
                auto storeOp = mlir::dyn_cast_or_null<mlir_ts::StoreOp>(op->getNextNode());
                if (!place || !storeOp || storeOp.getReference() != releaseSlotOp.getSlot() ||
                    rootOf(storeOp.getValue()) != callResult || place.getPosition() != field.getPosition() ||
                    place.getType() != field.getType() || !sameObjectInBlock(rootOf(place.getObjectRef()), rootOf(field.getObjectRef())))
                {
                    return false;
                }

                take.taker = kept;
                take.field = field.getResult();
                take.release = op;
                return true;
            }

            auto viewOfResult = isView(op) && rootOf(op->getResult(0)) == callResult;
            auto retainOfResult = mlir::isa<mlir_ts::RetainOp>(op) && rootOf(op->getOperand(0)) == callResult;
            if (!viewOfResult && !retainOfResult)
            {
                return false;
            }
        }

        return false;
    }

    bool OwnershipInferencePass::namedFieldTakeOf(mlir::Operation *read, mlir::Operation *retain,
                                                  mlir_ts::PropertyRefOp field, FieldTake &take)
    {
        // The object a `const` names is `new`'s own, or a call's own result: nothing else holds it,
        // and its releases end it. The field is left empty right at the read, so each release frees
        // nothing of what the taker has, on a path that took it or not (`c ? init : r.value`).
        // ...or the object an owning `let` holds (spec 65): every read of the local is the object, until
        // the local ends or is given another - its releases and its stores are the object's ends, the
        // store or the declaration in a loop makes it anew. Nothing else may use the local itself.
        auto object = field.getObjectRef();
        auto *producer = object.getDefiningOp();
        // made here: `new`'s own, or a call's own result - not a global (immortal under own, spec 25,
        // so a local given one is "owning" all the same) nor a parameter
        auto madeHere = [&](mlir::Value value) {
            auto *definingOp = value ? value.getDefiningOp() : nullptr;
            return definingOp && ((mlir::isa<mlir_ts::NewOp>(definingOp) && definingOp->hasAttr(OWNED_RESULT_ATTR_NAME)) ||
                                  callResultIsOwn(definingOp));
        };
        llvm::SmallVector<mlir::Value> objects;
        llvm::SmallVector<mlir::Operation *> ends;
        llvm::SmallVector<mlir::Operation *> renewals;
        if (auto slotRead = mlir::dyn_cast<mlir_ts::LoadOp>(producer))
        {
            auto slot = slotRead.getReference().getDefiningOp<mlir_ts::VariableOp>();
            if (!slot || slot.getCaptured().value_or(false) || !isOwningVariable(slot) ||
                (slot.getInitializer() && !madeHere(slot.getInitializer())))
            {
                return false;
            }

            renewals.push_back(slot);
            for (auto *user : slot->getUsers())
            {
                if (auto loadOp = mlir::dyn_cast<mlir_ts::LoadOp>(user))
                {
                    objects.push_back(loadOp.getResult());
                }
                else if (auto storeOp = mlir::dyn_cast<mlir_ts::StoreOp>(user); storeOp && storeOp.getReference() == slot.getResult())
                {
                    if (!madeHere(storeOp.getValue()))
                    {
                        return false;
                    }

                    ends.push_back(user);
                    renewals.push_back(user);
                }
                else if (mlir::isa<mlir_ts::ReleaseSlotOp>(user))
                {
                    ends.push_back(user);
                }
                else if (!mlir::isa<mlir_ts::RetainSlotOp>(user))
                {
                    return false;
                }
            }
        }
        else
        {
            if (!madeHere(object))
            {
                return false;
            }

            objects.push_back(object);
            renewals.push_back(producer);
            for (auto *user : object.getUsers())
            {
                if (mlir::isa<mlir_ts::ReleaseOp>(user))
                {
                    ends.push_back(user);
                }
            }
        }

        if (ends.empty())
        {
            return false;
        }

        // taken once for each object: not again around a loop that does not make it anew
        auto *next = read->getNextNode();
        if (!next || reachableAfter(next, read, renewals))
        {
            return false;
        }

        // Nothing sees the object after the move but its ends: no method, no move of it, no read of that
        // field, nor anything made of the object before it that is used after (a borrow of it, a local
        // given it). Its other fields are its own still.
        auto seenAfter = [&](mlir::Operation *op) { return op != read && op != retain && reachableAfter(read, op, ends); };
        for (auto value : objects)
        {
            for (auto *user : value.getUsers())
            {
                if (mlir::isa<mlir_ts::ReleaseOp>(user) || user == field.getOperation())
                {
                    continue;
                }

                if (auto propertyRefOp = mlir::dyn_cast<mlir_ts::PropertyRefOp>(user);
                    propertyRefOp && propertyRefOp.getPosition() != field.getPosition())
                {
                    continue;
                }

                // the user, and what is made of its results in turn, through a local it is stored into
                llvm::SmallVector<mlir::Operation *> derived{user};
                llvm::SmallPtrSet<mlir::Operation *, 8> visited;
                while (!derived.empty())
                {
                    auto *op = derived.pop_back_val();
                    if (!visited.insert(op).second)
                    {
                        continue;
                    }

                    if (seenAfter(op))
                    {
                        return false;
                    }

                    for (auto result : op->getResults())
                    {
                        derived.append(result.getUsers().begin(), result.getUsers().end());
                    }

                    if (auto storeOp = mlir::dyn_cast<mlir_ts::StoreOp>(op))
                    {
                        auto reference = storeOp.getReference();
                        derived.append(reference.getUsers().begin(), reference.getUsers().end());
                    }
                }
            }
        }

        auto *taker = takerOf(retain, read->getResult(0));
        if (!taker)
        {
            return false;
        }

        take.retain = retain;
        take.taker = taker;
        take.field = field.getResult();
        take.emptyAtRead = true;
        return true;
    }

    bool OwnershipInferencePass::keptParamOutlives(const Chain &chain, llvm::function_ref<bool(mlir::Operation *)> outlives)
    {
        for (auto &root : chain.roots)
        {
            auto load = keptParamLoadOf(root.first);
            if (!load)
            {
                continue;
            }

            // the moves of the parameter: what takes a read of its slot
            llvm::SmallVector<std::pair<mlir::Operation *, mlir::Value>> takers;
            for (auto *user : load.getReference().getUsers())
            {
                if (mlir::isa<mlir_ts::ReleaseSlotOp>(user))
                {
                    if (outlives(user))
                    {
                        return true;
                    }

                    continue;
                }

                auto readOp = mlir::dyn_cast<mlir_ts::LoadOp>(user);
                if (!readOp)
                {
                    continue;
                }

                forEachUse(readOp.getResult(), [&](mlir::Operation *use, mlir::Value used) {
                    auto varOp = mlir::dyn_cast<mlir_ts::VariableOp>(use);
                    if (mlir::isa<mlir_ts::RetainOp>(use))
                    {
                        auto *taker = takerOf(use, used);
                        takers.push_back({taker ? taker : use, used});
                    }
                    else if ((varOp && isOwningVariable(varOp)) || (isCall(use) && isKeptArgument(use, used)))
                    {
                        takers.push_back({use, used});
                    }
                });
            }

            for (auto [taker, moved] : takers)
            {
                Chain owner;
                if (!destinationChain(taker, owner))
                {
                    if (outlives(taker))
                    {
                        return true;
                    }

                    continue;
                }

                auto afterMove = [&](mlir::Operation *end) {
                    return end != taker && reachableAfter(taker, end, llvm::ArrayRef<mlir::Operation *>{}) && outlives(end);
                };

                for (auto *end : rootEnds(owner))
                {
                    if (afterMove(end))
                    {
                        return true;
                    }
                }

                for (auto *drop : drops)
                {
                    if (dropsChain(drop, owner, nullptr, moved) && afterMove(drop))
                    {
                        return true;
                    }
                }
            }
        }

        return false;
    }

    void OwnershipInferencePass::checkReturnedParam(mlir::Value value, llvm::SetVector<mlir::Operation *> &toErase)
    {
        llvm::SmallVector<BorrowUse> uses;
        llvm::SmallVector<mlir::Operation *> bookkeeping;
        if (auto *kept = walkBorrowed(value, {}, uses, nullptr, Keeping{&bookkeeping, true}))
        {
            reportBorrowEscapes(kept, ownerName(value), "a parameter");
            return;
        }

        toErase.insert(bookkeeping.begin(), bookkeeping.end());
    }

} // namespace own_inference
