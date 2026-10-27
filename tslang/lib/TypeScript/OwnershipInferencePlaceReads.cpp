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

                if (base.getDefiningOp<mlir_ts::VariableOp>())
                {
                    chain.roots.push_back({base, Chain::Slot});
                }
                else if (auto slotLoad = slotLoadOf(base))
                {
                    chain.roots.push_back({slotLoad.getReference(), Chain::Slot});
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

    bool OwnershipInferencePass::dropsChain(mlir::Operation *drop, const Chain &chain, mlir::Operation *read, mlir::Value borrowed)
    {
        if (read && drop == read)
        {
            return false; // a call that returns a borrow has not destroyed it by returning it
        }

        if (chain.anyPlace && !isCall(drop))
        {
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
        return (isCopyableString(value.getType()) || isHandle(value)) && !mlir::isa<mlir_ts::InvokeOp>(call) &&
               llvm::count(call->getOperands(), value) == 1;
    }

    bool OwnershipInferencePass::storesCopyOf(mlir::Operation *store, mlir::Value value)
    {
        auto storeOp = mlir::dyn_cast<mlir_ts::StoreOp>(store);
        if (!storeOp || storeOp.getValue() != value || !(isCopyableString(value.getType()) || isHandle(value)) ||
            !isPlace(storeOp.getReference()))
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
        // A result that borrows arrives with rc's temporary release, or rc's consuming `let`, and
        // in a function returning a borrow of the same parameter it may go out with rc's retain.
        auto walk = [&]() {
            name = ownerName(result);
            uses.clear();
            bookkeeping.clear();
            letKeeps = {};
            letKept = nullptr;
            Keeping keeping{&bookkeeping, returnsBorrowOf >= 0 && borrowedParam(result) == returnsBorrowOf};
            return walkBorrowed(result, read, uses, [&](mlir::Operation *user, llvm::ArrayRef<mlir::Operation *> kills) {
                auto borrower = borrowerOf(user, /*consumed=*/true);
                if (!borrower || borrower.getOperation() != user)
                {
                    return false;
                }

                name = varName(borrower);
                if (auto *keeps = collectBorrowerUses(borrower, kills, uses, bookkeeping); keeps && !letKept)
                {
                    letKept = keeps;
                    letKeeps = borrower;
                }

                return true;
            }, keeping, countedHandle);
        };

        auto *kept = walk();

        // A value taken out of a field of an owning record local by one move, the field read again
        // by nothing before the local ends (spec 48): the `value` of the record `next()` returns, in
        // a `for...of` or a spread that keeps what a generator yields
        FieldTake take;
        if ((kept || letKept) && !countedHandle && fieldTakeOf(read, kept ? kept : letKept, take))
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

                    reportPlaceBorrowOutlives(use.op, drop, name, place);
                    return true;
                }
            }

            return false;
        };

        for (auto *end : rootEnds(chain))
        {
            if (outlives(end))
            {
                return false;
            }
        }

        if (keptParamOutlives(chain, outlives))
        {
            return false;
        }

        for (auto *drop : drops)
        {
            if (countedStoreReleases.contains(drop))
            {
                continue;
            }

            if (dropsChain(drop, chain, read) && outlives(drop))
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
            !mlir::isa<mlir_ts::TupleType>(slotType.getElementType()) || isCopyableString(read->getResult(0).getType()))
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
