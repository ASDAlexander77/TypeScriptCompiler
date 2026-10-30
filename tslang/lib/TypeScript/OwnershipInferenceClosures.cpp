// OwnershipInferencePass (see OwnershipInferencePass.h): closures (spec 2.5, phase 5) - captures,
// cells, generator state boxes and the fields they borrow, and the slots a closure takes.

#include "OwnershipInferencePass.h"

#define DEBUG_TYPE "own"

namespace own_inference
{

    // A closure stored into a field of the object it captures (spec 70; isBackReferenceBox)
    bool OwnershipInferencePass::isBackReference(Closure &closure)
    {
        return !closure.hasCells && isBackReferenceBox(closure.box);
    }

    void OwnershipInferencePass::classifyClosures()
    {
        // a bound method of `this` stored into a field of `this` (spec 70): it borrows the object, so
        // rc's reference for it goes, as a borrowing field's does
        getFunction().walk([&](mlir_ts::StoreOp storeOp) {
            if (isFieldOfMemberThis(storeOp.getReference()) && isBoundMethodOfMemberThis(storeOp.getValue()))
            {
                if (auto *retain = retainBefore(storeOp.getValue(), storeOp))
                {
                    fieldBorrowRetains.insert(retain);
                }
            }
        });

        getFunction().walk([&](mlir_ts::CreateBoundFunctionOp boundOp) {
            if (!boundOp->hasAttr(OWNS_CAPTURE_ATTR_NAME) || closureOfBox(boundOp.getThisVal()) != boundOp)
            {
                return;
            }

            auto closure = std::make_shared<Closure>();
            closure->op = boundOp;
            closure->box = boundOp.getThisVal();
            for (auto *user : closure->box.getUsers())
            {
                if (user == boundOp.getOperation())
                {
                    continue;
                }

                if (auto propertyRefOp = mlir::dyn_cast<mlir_ts::PropertyRefOp>(user))
                {
                    for (auto *fieldUser : propertyRefOp->getUsers())
                    {
                        auto storeOp = mlir::dyn_cast<mlir_ts::StoreOp>(fieldUser);
                        if (!storeOp || storeOp.getReference() != propertyRefOp.getResult())
                        {
                            closure->escape = closure->escape ? closure->escape : fieldUser;
                            continue;
                        }

                        closure->fills.push_back({propertyRefOp.getPosition(), storeOp});
                        closure->hasCells = closure->hasCells || mlir::isa<mlir_ts::RefType>(storeOp.getValue().getType());
                    }

                    continue;
                }

                if (isCall(user))
                {
                    closure->uses.push_back(user);
                    closure->boxUses.push_back(user);
                    continue;
                }

                closure->escape = closure->escape ? closure->escape : user;
            }

            if (auto *escape = followClosure(boundOp.getResult(), closure->uses, closure->ends, closure->aliased);
                escape && !closure->escape)
            {
                closure->escape = escape;
            }

            // a back-reference (spec 70): the field owns the closure, and its box borrows the object -
            // its routine frees the box alone (`__own_borrows_captures`), and rc's reference for the
            // copy goes, as a borrowing field's does
            if (isBackReference(*closure))
            {
                for (auto &fill : closure->fills)
                {
                    if (auto *retain = retainBefore(fill.second.getValue(), fill.second))
                    {
                        fieldBorrowRetains.insert(retain);
                    }

                    setAttrTracked(fill.second, OWN_CAPTURE_BORROW_ATTR_NAME);
                }

                setAttrTracked(boundOp, OWN_BORROWS_CAPTURES_ATTR_NAME);
                return;
            }

            // copies only: the box owns them (phase 4); only an alias needs deciding - unless the box
            // is given to a generator's maker whose result borrows from it
            if (!closure->hasCells)
            {
                // ...and only when every copy has an owner of its own: the box frees itself alone, so a
                // value made for it (a callback that captures, `a.filter((x) => x > n)`) would be
                // nobody's
                // ...or when it holds the member's `this` (a copy, spec 70), the caller's, as a closure
                // that does not escape borrowed the cell it held before
                auto noneFresh = llvm::none_of(closure->fills, [&](auto &fill) {
                    auto value = fill.second.getValue();
                    return !holdsNoBlock(value) && isFresh(rootOf(value));
                });
                auto holdsThis = llvm::any_of(closure->fills, [&](auto &fill) { return isMemberThis(fill.second.getValue()); });
                closure->borrowsCopies =
                    !closure->escape && noneFresh &&
                    (holdsThis || llvm::any_of(closure->boxUses, [](mlir::Operation *use) {
                         return use->hasAttr(OWN_RESULT_BOUNDED_ATTR_NAME);
                     }));
                if (!closure->borrowsCopies)
                {
                    if (!closure->escape && closure->aliased)
                    {
                        closures.push_back(std::move(closure));
                    }

                    return;
                }
            }

            for (auto &fill : closure->fills)
            {
                auto storeOp = fill.second;
                auto isCell = mlir::isa<mlir_ts::RefType>(storeOp.getValue().getType());
                // an escaping closure's copies move into its box as in phase 4, rc's retain with them
                if (closure->escape && !isCell)
                {
                    continue;
                }

                if (auto *retain = retainBefore(storeOp.getValue(), storeOp))
                {
                    closure->retains.push_back(retain);
                    captureRetains.insert(retain);
                }

                if (!closure->escape && !isCell)
                {
                    setAttrTracked(storeOp, OWN_CAPTURE_BORROW_ATTR_NAME);
                    captureStores[storeOp] = closure.get();
                }
            }

            if (!closure->escape)
            {
                setAttrTracked(boundOp, OWN_BORROWS_CAPTURES_ATTR_NAME);
            }

            closureOf[boundOp.getOperation()] = closure.get();
            closures.push_back(std::move(closure));
        });
    }

    void OwnershipInferencePass::classifyLiteralCaptures()
    {
        getFunction().walk([&](mlir_ts::VariableOp boxOp) {
            if (!boxOp->hasAttr(CAPTURE_BOX_ATTR_NAME))
            {
                return;
            }

            // filled field by field, and stored once into a literal's `.captured`: `{ w: (x) => x * b.n }`
            // builds the literal in a local, then copies it into the object `ts.New` made
            llvm::SmallVector<mlir_ts::StoreOp> fills;
            llvm::SmallVector<mlir::Operation *> boxRetains;
            mlir_ts::VariableOp literalOp;
            int32_t capturedPosition = -1;
            for (auto &use : boxOp->getUses())
            {
                auto *user = use.getOwner();
                if (auto propertyRefOp = mlir::dyn_cast<mlir_ts::PropertyRefOp>(user))
                {
                    for (auto *fieldUser : propertyRefOp->getUsers())
                    {
                        auto storeOp = mlir::dyn_cast<mlir_ts::StoreOp>(fieldUser);
                        if (!storeOp || storeOp.getReference() != propertyRefOp.getResult())
                        {
                            return;
                        }

                        fills.push_back(storeOp);
                    }

                    continue;
                }

                // rc's count of the box, for the literal's reference
                if (mlir::isa<mlir_ts::RetainCellOp, mlir_ts::ReleaseCellOp>(user))
                {
                    if (mlir::isa<mlir_ts::RetainCellOp>(user))
                    {
                        boxRetains.push_back(user);
                    }

                    continue;
                }

                auto storeOp = mlir::dyn_cast<mlir_ts::StoreOp>(user);
                auto fieldOp = storeOp && storeOp.getValue() == boxOp.getResult()
                                   ? storeOp.getReference().getDefiningOp<mlir_ts::PropertyRefOp>()
                                   : mlir_ts::PropertyRefOp();
                auto ownerOp = fieldOp ? fieldOp.getObjectRef().getDefiningOp<mlir_ts::VariableOp>() : mlir_ts::VariableOp();
                auto tupleType = ownerOp ? mlir::dyn_cast<mlir_ts::TupleType>(
                                               mlir::cast<mlir_ts::RefType>(ownerOp.getType()).getElementType())
                                         : mlir_ts::TupleType();
                auto fieldName = tupleType && fieldOp.getPosition() < tupleType.size()
                                     ? mlir::dyn_cast_or_null<mlir::StringAttr>(tupleType.getFieldInfo(fieldOp.getPosition()).id)
                                     : mlir::StringAttr();
                if (literalOp || !fieldName || fieldName.getValue() != CAPTURED_NAME)
                {
                    return;
                }

                literalOp = ownerOp;
                capturedPosition = static_cast<int32_t>(fieldOp.getPosition());
            }

            if (!literalOp)
            {
                return;
            }

            // ...read once, into the object it becomes
            llvm::SmallVector<mlir::Value> objects;
            for (auto *user : literalOp->getUsers())
            {
                if (mlir::isa<mlir_ts::PropertyRefOp>(user))
                {
                    continue;
                }

                auto loadOp = mlir::dyn_cast<mlir_ts::LoadOp>(user);
                if (!loadOp || !loadOp.getResult().hasOneUse())
                {
                    return;
                }

                auto storeOp = mlir::dyn_cast<mlir_ts::StoreOp>(*loadOp.getResult().getUsers().begin());
                auto newOp = storeOp ? storeOp.getReference().getDefiningOp<mlir_ts::NewOp>() : mlir_ts::NewOp();
                if (!newOp)
                {
                    return;
                }

                objects.push_back(newOp.getResult());
            }

            if (objects.size() != 1)
            {
                return;
            }

            // the literal stays here: nothing stores, returns, captures or keeps it
            auto box = std::make_shared<Closure>();
            box->box = boxOp.getResult();
            llvm::SmallVector<BorrowUse> uses;
            auto newOp = objects.front().getDefiningOp<mlir_ts::NewOp>();
            if (llvm::is_contained(recordBorrowedFields(newOp), capturedPosition))
            {
                // ...or a borrowing record's box, seen only as interfaces given to calls that keep them
                // not (spec 68): their uses are the box's, and its release frees the box alone
                llvm::SmallVector<mlir::Value> interfaces;
                for (auto *user : newOp->getUsers())
                {
                    if (mlir::isa<mlir_ts::CastOp>(user))
                    {
                        for (auto *castUser : user->getUsers())
                        {
                            if (mlir::isa<mlir_ts::NewInterfaceOp>(castUser))
                            {
                                interfaces.push_back(castUser->getResult(0));
                            }
                        }
                    }
                }

                RecordWalk walk;
                llvm::SmallVector<mlir::Operation *, 2> kills{newOp.getOperation(), literalOp.getOperation()};
                walkRecord(interfaces, kills, walk);
                if (walk.kept || !walk.returns.empty())
                {
                    auto *at = walk.kept ? walk.kept : walk.returns.front();
                    at->emitError("-mm=own: an object literal that borrows what its arrows capture can't outlive this function");
                    signalPassFailure();
                    return;
                }

                uses.append(walk.uses.begin(), walk.uses.end());
            }
            else if (walkBorrowed(objects.front(), {}, uses))
            {
                return;
            }

            for (auto &use : uses)
            {
                box->uses.push_back(use.op);
            }

            // copies of values with an owner of their own, not one rc retained for the box
            llvm::SmallVector<mlir_ts::StoreOp> borrowed;
            for (auto storeOp : fills)
            {
                auto value = storeOp.getValue();
                // a cell of a `let` of this frame (spec 69): borrowed as a closure that does not escape
                // borrows it, the cell's own releases bounding the literal's uses (decideClosures)
                if (mlir::isa<mlir_ts::RefType>(value.getType()))
                {
                    if (!isCellVariable(value))
                    {
                        return;
                    }

                    if (auto *retain = retainBefore(value, storeOp))
                    {
                        box->retains.push_back(retain);
                    }

                    box->fills.push_back({0, storeOp});
                    box->hasCells = true;
                    continue;
                }

                if (holdsNoBlock(value))
                {
                    continue;
                }

                if (!hasOwnerOfItsOwn(value))
                {
                    return;
                }

                // rc's reference for the box goes with the borrow (decideClosures)
                if (auto *retain = retainBefore(value, storeOp))
                {
                    box->retains.push_back(retain);
                }

                borrowed.push_back(storeOp);
            }

            // ...and the box's own: no state's (classifyStateBoxes)
            box->retains.append(boxRetains.begin(), boxRetains.end());

            captureRetains.insert(box->retains.begin(), box->retains.end());
            for (auto storeOp : borrowed)
            {
                setAttrTracked(storeOp, OWN_CAPTURE_BORROW_ATTR_NAME);
                captureStores[storeOp] = box.get();
            }

            literalBoxes.push_back(std::move(box));
        });
    }

    llvm::SmallVector<int32_t> OwnershipInferencePass::borrowingFieldsOf(mlir::Type recordType)
    {
        llvm::SmallVector<int32_t> fields;
        if (auto objectType = mlir::dyn_cast<mlir_ts::ObjectType>(recordType))
        {
            recordType = objectType.getStorageType();
        }

        if (auto storageType = mlir::dyn_cast<mlir_ts::ObjectStorageType>(recordType))
        {
            recordType = mlir_ts::TupleType::get(&getContext(), storageType.getFields());
        }

        auto module = getFunction()->getParentOfType<mlir::ModuleOp>();
        auto entries = module ? module->getAttrOfType<mlir::ArrayAttr>(OWN_BORROWING_FIELDS_ATTR_NAME) : mlir::ArrayAttr();
        if (!entries)
        {
            return fields;
        }

        for (auto entry : entries.getAsRange<mlir::ArrayAttr>())
        {
            if (mlir::cast<mlir::TypeAttr>(entry[0]).getValue() == recordType)
            {
                auto indices = mlir::cast<mlir::DenseI32ArrayAttr>(entry[1]).asArrayRef();
                fields.append(indices.begin(), indices.end());
            }
        }

        return fields;
    }

    bool OwnershipInferencePass::ownsKeptState(mlir::Type recordType)
    {
        auto fields = borrowingFieldsOf(recordType);
        return !fields.empty() && fields.front() < 0;
    }

    bool OwnershipInferencePass::isYieldedField(mlir::Value ref)
    {
        auto propertyRefOp = ref.getDefiningOp<mlir_ts::PropertyRefOp>();
        if (!propertyRefOp)
        {
            return false;
        }

        auto recordType = propertyRefOp->getOperand(0).getType();
        if (auto objectType = mlir::dyn_cast<mlir_ts::ObjectType>(recordType))
        {
            recordType = objectType.getStorageType();
        }

        if (auto storageType = mlir::dyn_cast<mlir_ts::ObjectStorageType>(recordType))
        {
            recordType = mlir_ts::TupleType::get(&getContext(), storageType.getFields());
        }

        auto module = getFunction()->getParentOfType<mlir::ModuleOp>();
        auto entries = module ? module->getAttrOfType<mlir::ArrayAttr>(OWN_YIELDED_FIELDS_ATTR_NAME) : mlir::ArrayAttr();
        if (!entries)
        {
            return false;
        }

        for (auto entry : entries.getAsRange<mlir::ArrayAttr>())
        {
            if (mlir::cast<mlir::TypeAttr>(entry[0]).getValue() == recordType &&
                llvm::is_contained(mlir::cast<mlir::DenseI32ArrayAttr>(entry[1]).asArrayRef(),
                                   static_cast<int32_t>(propertyRefOp.getPosition())))
            {
                return true;
            }
        }

        return false;
    }

    bool OwnershipInferencePass::isBorrowingField(mlir::Value ref)
    {
        auto propertyRefOp = ref.getDefiningOp<mlir_ts::PropertyRefOp>();
        if (!propertyRefOp)
        {
            return false;
        }

        auto fields = borrowingFieldsOf(propertyRefOp->getOperand(0).getType());
        return fields.size() > 1 && llvm::is_contained(llvm::ArrayRef<int32_t>(fields).drop_front(),
                                                       static_cast<int32_t>(propertyRefOp.getPosition()));
    }

    void OwnershipInferencePass::classifyBorrowingFields()
    {
        getFunction().walk([&](mlir::Operation *op) {
            if (auto storeOp = mlir::dyn_cast<mlir_ts::StoreOp>(op); storeOp && isBorrowingField(storeOp.getReference()))
            {
                setAttrTracked(storeOp, OWN_CAPTURE_BORROW_ATTR_NAME);
                if (auto *retain = retainBefore(storeOp.getValue(), storeOp))
                {
                    fieldBorrowRetains.insert(retain);
                }
            }
            else if (auto releaseSlotOp = mlir::dyn_cast<mlir_ts::ReleaseSlotOp>(op);
                     releaseSlotOp && isBorrowingField(releaseSlotOp.getSlot()))
            {
                fieldBorrowReleases.push_back(op);
            }
        });
    }

    void OwnershipInferencePass::classifyArgumentArrays()
    {
        getFunction().walk([&](mlir_ts::CreateArrayOp arrayOp) {
            // the array a rest parameter is given (`a.concat(b)`), made of values that are not its
            // own to take, and given to one call that borrows it (spec 49)
            auto result = arrayOp.getResult();
            if (arrayOp->getNumOperands() == 0 || !result.hasOneUse())
            {
                return;
            }

            auto *call = *result.getUsers().begin();
            if (!isCall(call) || call->hasTrait<mlir::OpTrait::IsTerminator>() || isKeptArgument(call, result) ||
                !llvm::is_contained(callArgs(call), result))
            {
                return;
            }

            llvm::SmallVector<mlir::Operation *> retains;
            for (auto element : arrayOp->getOperands())
            {
                if (holdsNoBlock(element))
                {
                    continue;
                }

                // a fresh value nothing else releases would be the array's alone, which a release of
                // the array alone leaks; a named one (`const a = new C()`) has its own release
                auto root = rootOf(element);
                auto releasedElsewhere = false;
                forEachUse(root, [&](mlir::Operation *user, mlir::Value) {
                    releasedElsewhere = releasedElsewhere || mlir::isa<mlir_ts::ReleaseOp>(user);
                });
                auto *retain = isFresh(root) && !releasedElsewhere ? nullptr : retainBefore(element, arrayOp);
                if (!retain)
                {
                    return;
                }

                retains.push_back(retain);
            }

            // nothing borrowed (an array of numbers), or a result that borrows the array and outlives
            // the call: the array stays the plain value it is
            auto args = callArgs(call);
            auto resultBorrowsArray = llvm::any_of(resultBorrowsAll(call), [&](int index) {
                return static_cast<size_t>(index) < args.size() && args[index] == result;
            });
            if (retains.empty() || resultBorrowsArray || (call->getNumResults() == 1 && borrowingCall(call->getResult(0))))
            {
                return;
            }

            // rc's references for the elements go: the array borrows them, for as long as the call
            setAttrTracked(arrayOp, OWN_BORROWING_ARRAY_ATTR_NAME);
            fieldBorrowRetains.insert(retains.begin(), retains.end());

            // and the array itself is given back after the call, without what it holds
            if (!dryRun)
            {
                mlir::OpBuilder builder(call->getContext());
                builder.setInsertionPointAfter(call);
                auto releaseOp = builder.create<mlir_ts::ReleaseOp>(call->getLoc(), result);
                releaseOp->setAttr(OWN_SHALLOW_RELEASE_ATTR_NAME, mlir::UnitAttr::get(&getContext()));
            }
        });
    }

    bool OwnershipInferencePass::isBorrowingCell(mlir::Value ref)
    {
        auto cellLoad = ref.getDefiningOp<mlir_ts::LoadOp>();
        return cellLoad && isBorrowingBoxField(cellLoad.getReference());
    }

    bool OwnershipInferencePass::isBorrowingBoxField(mlir::Value ref)
    {
        auto cellRef = ref.getDefiningOp<mlir_ts::PropertyRefOp>();
        auto boxLoad = cellRef ? cellRef->getOperand(0).getDefiningOp<mlir_ts::LoadOp>() : mlir_ts::LoadOp();
        auto boxRef = boxLoad ? boxLoad.getReference().getDefiningOp<mlir_ts::PropertyRefOp>() : mlir_ts::PropertyRefOp();
        if (!boxRef)
        {
            return false;
        }

        auto fields = borrowingFieldsOf(boxRef->getOperand(0).getType());
        return !fields.empty() && fields.front() == static_cast<int32_t>(boxRef.getPosition());
    }

    void OwnershipInferencePass::checkBorrowedReads()
    {
        getFunction().walk([&](mlir_ts::LoadOp loadOp) {
            auto ref = loadOp.getReference();
            if (!ownsHeap(loadOp.getResult()) || !(isBorrowingCell(ref) || isBorrowingBoxField(ref) || isBorrowingField(ref)))
            {
                return;
            }

            auto reported = false;
            forEachUse(loadOp.getResult(), [&](mlir::Operation *user, mlir::Value used) {
                if (reported || isBorrow(user, used) || fieldBorrowRetains.contains(user))
                {
                    return;
                }

                reportPlaceBorrowEscapes(user, ownerName(loadOp.getResult()), "the generator's argument");
                reported = true;
            });
        });
    }

    void OwnershipInferencePass::checkBoundedResults()
    {
        getFunction().walk([&](mlir::Operation *call) {
            auto bounded = call->getAttrOfType<mlir::DenseI32ArrayAttr>(OWN_RESULT_BOUNDED_ATTR_NAME);
            // a number holds nothing a bound could outlive: `return apply({ w: (x) => x + item.n })`
            // makes a state-shaped literal, not a result that borrows `item`
            if (!bounded || !isCall(call) || call->getNumResults() != 1 || !ownsHeap(call->getResult(0)))
            {
                return;
            }

            auto name = ownerName(call->getResult(0));
            auto args = callArgs(call);

            // Borrowing only immortal globals (spec 25: never given another value, never freed) it
            // outlives nothing, wherever it is kept: `const it = map.values()` at the top of a script.
            // One of them at least: a callback that holds no block says nothing of the array `.map` reads.
            auto immortals = 0;
            auto boundedByImmortals = llvm::all_of(bounded.asArrayRef(), [&](int32_t index) {
                if (static_cast<size_t>(index) >= args.size())
                {
                    return true;
                }

                auto root = rootOf(args[index]);
                if (auto object = boundThis(root))
                {
                    root = rootOf(object);
                }

                if (isImmortalGlobalRead(root))
                {
                    immortals++;
                    return true;
                }

                return holdsNoBlock(args[index]);
            });
            if (boundedByImmortals && immortals > 0)
            {
                return;
            }

            llvm::SmallVector<mlir::Operation *> uses;
            if (auto *kept = boundedUses(call->getResult(0), uses))
            {
                reportPlaceBorrowEscapes(kept, name, "its generator's arguments");
                return;
            }
            for (auto index : bounded.asArrayRef())
            {
                if (static_cast<size_t>(index) >= args.size())
                {
                    continue;
                }

                // a closure's box (`.map(f)` calls its maker with it): it lives while the closures on
                // it do, and what it holds while their owners do. A reference to storage, so first.
                auto root = rootOf(args[index]);
                // the object a method is called on (`w.iter()` of a class that is not `@final`)
                if (auto object = boundThis(root))
                {
                    root = rootOf(object);
                }

                if (auto boxOp = root.getDefiningOp<mlir_ts::VariableOp>(); boxOp && boxOp->hasAttr(CAPTURE_BOX_ATTR_NAME))
                {
                    if (!checkBoundedByBox(call, root, uses, name))
                    {
                        return;
                    }

                    continue;
                }

                if (holdsNoBlock(args[index]) || isImmortalLiteral(args[index]))
                {
                    continue;
                }

                if (borrowedParam(root) >= 0)
                {
                    continue; // the caller's caller's: alive for all of this function
                }

                Chain chain;
                auto *read = placeReadOf(root);
                if (read)
                {
                    chain = chainOf(read);
                }
                else if (auto slotLoad = slotLoadOf(root))
                {
                    chain.roots.push_back({slotLoad.getReference(), Chain::Slot});
                }
                else if (isFresh(root))
                {
                    chain.roots.push_back({root, Chain::Owned});
                }
                else
                {
                    chain.roots.push_back({root, Chain::NotOwned});
                }

                // the root's own ends; and, for a read out of a place, whatever may destroy what the
                // place holds (a local or a made value is not destroyed by a call it is given)
                auto ends = rootEnds(chain);
                for (auto *drop : read ? llvm::ArrayRef<mlir::Operation *>(drops) : llvm::ArrayRef<mlir::Operation *>())
                {
                    if (drop != call && dropsChain(drop, chain, read))
                    {
                        ends.push_back(drop);
                    }
                }

                auto owner = ownerName(root);
                for (auto *end : ends)
                {
                    if (end == call)
                    {
                        continue;
                    }

                    for (auto *use : uses)
                    {
                        if (use != end && reachableAfter(end, use, call))
                        {
                            reportBorrowOutlives(use, end, name, owner);
                            return;
                        }
                    }
                }
            }
        });
    }

    bool OwnershipInferencePass::checkBoundedByBox(mlir::Operation *call, mlir::Value box, llvm::ArrayRef<mlir::Operation *> uses,
                           llvm::StringRef name)
    {
        llvm::SmallVector<mlir::Operation *> ends;
        for (auto *user : box.getUsers())
        {
            if (auto boundOp = mlir::dyn_cast<mlir_ts::CreateBoundFunctionOp>(user); boundOp && boundOp.getThisVal() == box)
            {
                llvm::SmallVector<mlir::Operation *> closureUses;
                auto aliased = false;
                if (followClosure(boundOp.getResult(), closureUses, ends, aliased))
                {
                    reportPlaceBorrowEscapes(call, name, "a closure that escapes"); // the box may go anywhere
                    return false;
                }

                continue;
            }

            auto propertyRefOp = mlir::dyn_cast<mlir_ts::PropertyRefOp>(user);
            if (!propertyRefOp)
            {
                continue;
            }

            for (auto *fieldUser : propertyRefOp->getUsers())
            {
                auto storeOp = mlir::dyn_cast<mlir_ts::StoreOp>(fieldUser);
                if (!storeOp || storeOp.getReference() != propertyRefOp.getResult() || holdsNoBlock(storeOp.getValue()))
                {
                    continue;
                }

                auto root = rootOf(storeOp.getValue());
                Chain chain;
                if (auto slotLoad = slotLoadOf(root))
                {
                    chain.roots.push_back({slotLoad.getReference(), Chain::Slot});
                }
                else if (borrowedParam(root) >= 0 || isImmortalLiteral(root))
                {
                    continue;
                }
                else if (isFresh(root))
                {
                    chain.roots.push_back({root, Chain::Owned});
                }
                else
                {
                    reportPlaceBorrowEscapes(call, name, "a value this function cannot bound");
                    return false;
                }

                auto rootEndsOf = rootEnds(chain);
                ends.append(rootEndsOf.begin(), rootEndsOf.end());
            }
        }

        for (auto *end : ends)
        {
            for (auto *use : uses)
            {
                if (end != call && use != end && reachableAfter(end, use, call))
                {
                    reportBorrowOutlives(use, end, name, "the generator's source");
                    return false;
                }
            }
        }

        return true;
    }

    mlir::Operation *OwnershipInferencePass::boundedUses(mlir::Value result, llvm::SmallVectorImpl<mlir::Operation *> &uses)
    {
        llvm::SmallVector<mlir::Value> values{result};
        llvm::SmallPtrSet<mlir::Value, 8> seen;
        mlir::Operation *kept = nullptr;
        while (!values.empty() && !kept)
        {
            auto value = values.pop_back_val();
            if (!seen.insert(value).second)
            {
                continue;
            }

            forEachUse(value, [&](mlir::Operation *user, mlir::Value used) {
                if (kept || mlir::isa<mlir_ts::RetainOp, mlir_ts::ReleaseOp, mlir_ts::DebugVariableOp>(user))
                {
                    return;
                }

                mlir::Value slot;
                if (auto varOp = mlir::dyn_cast<mlir_ts::VariableOp>(user))
                {
                    slot = varOp.getResult();
                }
                else if (auto storeOp = mlir::dyn_cast<mlir_ts::StoreOp>(user); storeOp && storeOp.getValue() == used)
                {
                    slot = storeOp.getReference();
                }

                if (slot)
                {
                    auto slotVar = slot.getDefiningOp<mlir_ts::VariableOp>();
                    if (!slotVar || slotVar.getCaptured().value_or(false))
                    {
                        kept = user;
                        return;
                    }

                    for (auto *slotUser : slot.getUsers())
                    {
                        if (auto loadOp = mlir::dyn_cast<mlir_ts::LoadOp>(slotUser))
                        {
                            values.push_back(loadOp.getResult());
                        }
                    }

                    return;
                }

                // returned by a function that says so (`@borrows`): its callers hold the result to
                // the same bound, and the signature pass checked the arguments are its own
                if (mlir::isa<mlir_ts::ReturnInternalOp>(user) && getFunction()->hasAttr(OWN_DECLARED_BORROWS_ATTR_NAME))
                {
                    return;
                }

                uses.push_back(user);
                if (isCall(user))
                {
                    if (isKeptArgument(user, used))
                    {
                        // given to a generator that says `@keeps` (`s.values().take(2)`): its state
                        // owns the result, so is bounded by the same arguments - its uses are this
                        // result's (spec 38)
                        if (user->getNumResults() == 1 && ownsKeptState(user->getResult(0).getType()))
                        {
                            values.push_back(user->getResult(0));
                            return;
                        }

                        kept = user;
                    }

                    return;
                }

                // `it.next()`: the field read, the bound function, its `this`
                if (mlir::isa<mlir_ts::PropertyRefOp, mlir_ts::LoadOp, mlir_ts::GetThisOp, mlir_ts::GetMethodOp>(user))
                {
                    for (auto partResult : user->getResults())
                    {
                        values.push_back(partResult);
                    }

                    return;
                }

                if (!isBorrow(user, used))
                {
                    kept = user;
                }
            });
        }

        return kept;
    }

    void OwnershipInferencePass::classifyStateBoxes()
    {
        getFunction().walk([&](mlir_ts::VariableOp boxOp) {
            auto box = boxOp.getResult();
            if (!boxOp->hasAttr(CAPTURE_BOX_ATTR_NAME) || closureOfBox(box))
            {
                return;
            }

            // an object literal's box that borrows (classifyLiteralCaptures): no state's
            if (llvm::any_of(literalBoxes, [&](auto &literalBox) { return literalBox->box == box; }))
            {
                return;
            }

            auto closure = std::make_shared<Closure>();
            closure->box = box;
            llvm::SmallVector<mlir::Operation *> boxRetains;
            for (auto *user : box.getUsers())
            {
                if (auto propertyRefOp = mlir::dyn_cast<mlir_ts::PropertyRefOp>(user))
                {
                    for (auto *fieldUser : propertyRefOp->getUsers())
                    {
                        auto storeOp = mlir::dyn_cast<mlir_ts::StoreOp>(fieldUser);
                        if (!storeOp || storeOp.getReference() != propertyRefOp.getResult())
                        {
                            return;
                        }

                        closure->fills.push_back({propertyRefOp.getPosition(), storeOp});
                        closure->hasCells = closure->hasCells || mlir::isa<mlir_ts::RefType>(storeOp.getValue().getType());
                    }

                    continue;
                }

                if (mlir::isa<mlir_ts::RetainCellOp>(user))
                {
                    boxRetains.push_back(user);
                    continue;
                }

                if (!closure->escape && isStoreIntoState(user, box))
                {
                    closure->escape = user;
                    continue;
                }

                return;
            }

            if (!closure->escape)
            {
                return;
            }

            auto stateRef = mlir::cast<mlir_ts::StoreOp>(closure->escape).getReference().getDefiningOp<mlir_ts::PropertyRefOp>();
            auto stateType = mlir::cast<mlir_ts::RefType>(stateRef->getOperand(0).getType()).getElementType();
            auto borrowing = borrowingFieldsOf(stateType);
            closure->borrowsValues = !borrowing.empty() && borrowing.front() == stateRef.getPosition();
            // `.map`'s maker copies what its parameter box holds into the generator's box: a borrow
            if (closure->borrowsValues)
            {
                for (auto &fill : closure->fills)
                {
                    if (!mlir::isa<mlir_ts::RefType>(fill.second.getValue().getType()))
                    {
                        setAttrTracked(fill.second, OWN_CAPTURE_BORROW_ATTR_NAME);
                    }
                }
            }

            for (auto &fill : closure->fills)
            {
                // a parameter this function does not own, in a box that does not borrow: the box
                // cannot take rc's reference for it, the caller's still holds it; left to the
                // second-reference error, which says the callers would have to give it up
                auto param = borrowedParam(fill.second.getValue());
                if (!closure->borrowsValues && param >= 0 && !llvm::is_contained(ownedParams(getFunction()), param))
                {
                    continue;
                }

                if (auto *retain = retainBefore(fill.second.getValue(), fill.second))
                {
                    closure->retains.push_back(retain);
                    captureRetains.insert(retain);
                }
            }

            for (auto *retain : boxRetains)
            {
                closure->retains.push_back(retain);
                captureRetains.insert(retain);
            }

            stateBoxes[box] = closure.get();
            closures.push_back(std::move(closure));
        });
    }

    bool OwnershipInferencePass::isStoreIntoState(mlir::Operation *user, mlir::Value box)
    {
        auto storeOp = mlir::dyn_cast<mlir_ts::StoreOp>(user);
        auto propertyRefOp = storeOp && storeOp.getValue() == box
                                 ? storeOp.getReference().getDefiningOp<mlir_ts::PropertyRefOp>()
                                 : mlir_ts::PropertyRefOp();
        auto stateOp = propertyRefOp ? propertyRefOp->getOperand(0).getDefiningOp<mlir_ts::VariableOp>() : mlir_ts::VariableOp();
        if (!stateOp || !isConstantRecordLocal(stateOp))
        {
            return false;
        }

        auto reads = 0;
        for (auto *stateUser : stateOp->getUsers())
        {
            auto loadOp = mlir::dyn_cast<mlir_ts::LoadOp>(stateUser);
            if (!loadOp)
            {
                continue;
            }

            ++reads;
            for (auto *readUser : loadOp->getUsers())
            {
                // rc's retain of the initial state, for fields that own (holdsConstantsOrMoves)
                if (mlir::isa<mlir_ts::RetainOp>(readUser))
                {
                    continue;
                }

                auto fillOp = mlir::dyn_cast<mlir_ts::StoreOp>(readUser);
                if (!fillOp || fillOp.getValue() != loadOp.getResult() || !fillOp.getReference().getDefiningOp<mlir_ts::NewOp>())
                {
                    return false;
                }
            }
        }

        return reads > 0;
    }

    mlir::Operation *OwnershipInferencePass::retainBefore(mlir::Value value, mlir::Operation *store)
    {
        // rc's retain of an earlier read of the same place (twinReadAfter)
        if (auto loadOp = value.getDefiningOp<mlir_ts::LoadOp>())
        {
            for (auto *user : loadOp.getReference().getUsers())
            {
                auto earlier = mlir::dyn_cast<mlir_ts::LoadOp>(user);
                auto *retain = earlier ? soleUser(earlier.getResult()) : nullptr;
                if (retain && mlir::isa<mlir_ts::RetainOp>(retain) && !captureRetains.contains(retain) &&
                    twinReadAfter(retain, earlier.getResult()) == value && retain->getBlock() == store->getBlock() &&
                    retain->isBeforeInBlock(store))
                {
                    return retain;
                }
            }
        }

        mlir::Operation *found = nullptr;
        for (auto *user : value.getUsers())
        {
            if (!mlir::isa<mlir_ts::RetainOp, mlir_ts::RetainCellOp>(user) || user->getBlock() != store->getBlock() ||
                !user->isBeforeInBlock(store) || captureRetains.contains(user))
            {
                continue;
            }

            if (!found || found->isBeforeInBlock(user))
            {
                found = user;
            }
        }

        return found;
    }

    mlir::Operation *OwnershipInferencePass::followClosure(mlir::Value closure, llvm::SmallVectorImpl<mlir::Operation *> &uses,
                                   llvm::SmallVectorImpl<mlir::Operation *> &ends, bool &aliased)
    {
        llvm::SmallVector<mlir::Value> values{closure};
        llvm::SmallPtrSet<mlir::Value, 8> seen;
        mlir::Operation *escape = nullptr;
        auto followSlot = [&](mlir::Value slot) {
            for (auto *slotUser : slot.getUsers())
            {
                if (mlir::isa<mlir_ts::RetainSlotOp, mlir_ts::DebugVariableOp>(slotUser))
                {
                    continue;
                }

                if (mlir::isa<mlir_ts::ReleaseSlotOp>(slotUser))
                {
                    ends.push_back(slotUser);
                }
                else if (auto loadOp = mlir::dyn_cast<mlir_ts::LoadOp>(slotUser))
                {
                    values.push_back(loadOp.getResult());
                }
                else if (auto storeOp = mlir::dyn_cast<mlir_ts::StoreOp>(slotUser); !storeOp || storeOp.getReference() != slot)
                {
                    escape = escape ? escape : slotUser;
                }
            }
        };
        // the closure's field of an unboxed record: the record's reads, and that field's
        llvm::SmallVector<std::pair<mlir::Value, int64_t>> records;
        auto followUnboxedRecord = [&](mlir_ts::VariableOp recordOp, int64_t position) {
            for (auto *recordUser : recordOp->getUsers())
            {
                if (auto loadOp = mlir::dyn_cast<mlir_ts::LoadOp>(recordUser))
                {
                    records.push_back({loadOp.getResult(), position});
                    values.push_back(loadOp.getResult());
                }
                else if (auto propertyRefOp = mlir::dyn_cast<mlir_ts::PropertyRefOp>(recordUser))
                {
                    // its fields, filled and read in place
                    for (auto *fieldUser : propertyRefOp->getUsers())
                    {
                        if (auto loadOp = mlir::dyn_cast<mlir_ts::LoadOp>(fieldUser);
                            loadOp && propertyRefOp.getPosition() != position)
                        {
                            continue;
                        }

                        if (auto loadOp = mlir::dyn_cast<mlir_ts::LoadOp>(fieldUser))
                        {
                            values.push_back(loadOp.getResult());
                        }
                        else if (auto storeOp = mlir::dyn_cast<mlir_ts::StoreOp>(fieldUser);
                                 !storeOp || storeOp.getReference() != propertyRefOp.getResult())
                        {
                            escape = escape ? escape : fieldUser;
                        }
                    }
                }
                else if (!mlir::isa<mlir_ts::DebugVariableOp>(recordUser))
                {
                    escape = escape ? escape : recordUser;
                }
            }
        };

        while (!values.empty() && !escape)
        {
            auto value = values.pop_back_val();
            if (!seen.insert(value).second)
            {
                continue;
            }

            forEachUse(value, [&](mlir::Operation *user, mlir::Value used) {
                if (escape || mlir::isa<mlir_ts::RetainOp, mlir_ts::DebugVariableOp>(user))
                {
                    return;
                }

                if (mlir::isa<mlir_ts::ReleaseOp>(user))
                {
                    ends.push_back(user);
                    return;
                }

                // kept for good (a keeper's): the box, its cells and its copies are immortal from
                // here, so the frame's releases of them free nothing, and whatever is done with the
                // immortal value - pushed, stored, run later - outlives nothing it relies on
                if (mlir::isa<mlir_ts::MakeImmortalOp>(user))
                {
                    uses.push_back(user);
                    return;
                }

                if (isCall(user))
                {
                    if (isKeptArgument(user, used))
                    {
                        escape = user;
                    }
                    else
                    {
                        uses.push_back(user);
                    }

                    return;
                }

                // the function and the box a call through the closure is made with
                if (mlir::isa<mlir_ts::GetThisOp, mlir_ts::GetMethodOp>(user))
                {
                    for (auto *partUser : user->getResult(0).getUsers())
                    {
                        if (isCall(partUser) && !isKeptArgument(partUser, user->getResult(0)))
                        {
                            uses.push_back(partUser);
                        }
                        else
                        {
                            escape = escape ? escape : partUser;
                        }
                    }

                    return;
                }

                mlir::Value slot;
                if (auto varOp = mlir::dyn_cast<mlir_ts::VariableOp>(user))
                {
                    slot = varOp.getResult();
                }
                else if (auto storeOp = mlir::dyn_cast<mlir_ts::StoreOp>(user); storeOp && storeOp.getValue() == used)
                {
                    slot = storeOp.getReference();
                }

                // a local, owning or not (`let f: () => number;` owns nothing): its reads are the
                // closure; a function's result slot is a local too, and its read is returned
                auto slotVar = slot ? slot.getDefiningOp<mlir_ts::VariableOp>() : mlir_ts::VariableOp();
                if (slotVar && !slotVar.getCaptured().value_or(false))
                {
                    aliased = aliased || !isOwningVariable(slotVar);
                    followSlot(slot);
                    return;
                }

                // a field of an unboxed record (spec 36), which owns nothing: `apply({ w: (x) => x + d.n })`
                // - whatever the record is given to, and whatever is read out of it, may run the closure
                if (auto fieldOp = slot ? slot.getDefiningOp<mlir_ts::PropertyRefOp>() : mlir_ts::PropertyRefOp())
                {
                    auto recordOp = fieldOp.getObjectRef().getDefiningOp<mlir_ts::VariableOp>();
                    if (recordOp && !recordOp.getCaptured().value_or(false) && !unboxedRecordFields(recordOp).empty())
                    {
                        followUnboxedRecord(recordOp, fieldOp.getPosition());
                        return;
                    }
                }

                // ...and a record read out of one: the closure's field read, not the others
                if (auto extractOp = mlir::dyn_cast<mlir_ts::ExtractPropertyOp>(user))
                {
                    auto record = llvm::find_if(records, [&](auto &entry) { return entry.first == used; });
                    if (record != records.end())
                    {
                        auto position = extractOp.getPosition();
                        if (position.size() == 1 && position.front() == record->second)
                        {
                            values.push_back(extractOp.getResult());
                        }

                        return;
                    }
                }

                escape = user;
            });
        }

        return escape;
    }

    bool OwnershipInferencePass::holdsCaptures(mlir::Value value)
    {
        if (mlir::isa<mlir_ts::BoundFunctionType, mlir_ts::HybridFunctionType>(value.getType()))
        {
            return true;
        }

        auto root = rootOf(value);
        if (auto varOp = root.getDefiningOp<mlir_ts::VariableOp>())
        {
            return varOp->hasAttr(CAPTURE_BOX_ATTR_NAME);
        }

        // a closure's `this` is its box, a reference; a bound method's is the object
        auto getThisOp = root.getDefiningOp<mlir_ts::GetThisOp>();
        return getThisOp && mlir::isa<mlir_ts::RefType>(getThisOp.getType());
    }

    void OwnershipInferencePass::decideClosures(llvm::SetVector<mlir::Operation *> &toErase)
    {
        for (auto &box : literalBoxes)
        {
            toErase.insert(box->retains.begin(), box->retains.end());

            // the cells a literal borrows (spec 69): nothing that uses it after one is given back
            auto ok = true;
            for (auto &fill : box->fills)
            {
                auto cell = fill.second.getValue();
                auto name = varName(cell.getDefiningOp<mlir_ts::VariableOp>());
                for (auto *end : cell.getUsers())
                {
                    if (!mlir::isa<mlir_ts::ReleaseCellOp>(end) && !movesCellAway(end))
                    {
                        continue;
                    }

                    for (auto *use : box->uses)
                    {
                        if (ok && reachableAfter(end, use, box->box.getDefiningOp()))
                        {
                            reportClosureOutlives(use, end, "the object literal", name);
                            ok = false;
                        }
                    }
                }
            }
        }

        for (auto &closure : closures)
        {
            if (!closure->op)
            {
                decideEscaping(*closure, toErase); // a generator's box
                continue;
            }

            auto *boundOp = closure->op.getOperation();
            auto name = ownerName(closure->op.getResult());
            auto ok = true;
            // Held by a local that owns nothing: rc's birth reference reads to the move logic as the
            // store's, which would leave the box with no owner. The closure keeps its releases, and
            // nothing may run it after them.
            auto keepAliasedReleases = [&]() {
                for (auto *end : closure->ends)
                {
                    if (!mlir::isa<mlir_ts::ReleaseOp>(end))
                    {
                        continue;
                    }

                    toErase.remove(end);
                    for (auto *use : closure->uses)
                    {
                        if (ok && reachableAfter(end, use, boundOp))
                        {
                            reportClosureOutlives(use, end, name, name);
                            ok = false;
                        }
                    }
                }
            };

            if (!closure->hasCells && !closure->borrowsCopies)
            {
                keepAliasedReleases();
                continue;
            }

            if (closure->escape)
            {
                decideEscaping(*closure, toErase);
                continue;
            }

            for (auto &fill : closure->fills)
            {
                auto value = fill.second.getValue();
                llvm::SmallVector<mlir::Operation *> ends;
                llvm::StringRef captured = "this value";
                if (mlir::isa<mlir_ts::RefType>(value.getType()))
                {
                    if (isCellVariable(value))
                    {
                        captured = varName(value.getDefiningOp<mlir_ts::VariableOp>());
                        for (auto *user : value.getUsers())
                        {
                            if (mlir::isa<mlir_ts::ReleaseCellOp>(user) || movesCellAway(user))
                            {
                                ends.push_back(user);
                            }
                        }
                    }
                    else if (!isLoadedCell(value))
                    {
                        // a local of this frame, or storage a parameter refers to (`this` of a @value
                        // constructor, the slot its `new` made), that holds no block - its fields are
                        // numbers: the closure, which does not escape, reads storage that lives as
                        // long as this call, and nothing in it can be freed
                        auto elementType = mlir::cast<mlir_ts::RefType>(value.getType()).getElementType();
                        auto argument = mlir::dyn_cast<mlir::BlockArgument>(value);
                        auto frameStorage = value.getDefiningOp<mlir_ts::VariableOp>() ||
                                            (argument && argument.getOwner()->isEntryBlock() &&
                                             mlir::isa<mlir_ts::FuncOp>(argument.getOwner()->getParentOp()));
                        if (frameStorage && !ownsHeap(value.getLoc(), elementType))
                        {
                            continue;
                        }

                        reportCaptureNotOwned(fill.second, captured);
                        ok = false;
                        continue;
                    }
                }
                else
                {
                    auto root = rootOf(value);
                    captured = ownerName(root);
                    Chain chain;
                    if (holdsNoBlock(value) || placeReadOf(root) || borrowedParam(root) >= 0 || isHandle(root))
                    {
                        // nothing to own, a read the read's check bounds, a parameter, or a handle,
                        // whose copy in the box is counted (spec 23.2)
                    }
                    else if (auto slotLoad = slotLoadOf(root))
                    {
                        chain.roots.push_back({slotLoad.getReference(), Chain::Slot});
                    }
                    else if (isFresh(root))
                    {
                        chain.roots.push_back({root, Chain::Owned});
                    }
                    else
                    {
                        reportCaptureNotOwned(fill.second, captured);
                        ok = false;
                        continue;
                    }

                    ends = rootEnds(chain);
                }

                for (auto *end : ends)
                {
                    if (captureRetains.contains(end) || captureStores.count(end))
                    {
                        continue; // a box's own copy, this one's or another borrowing closure's
                    }

                    for (auto *use : closure->uses)
                    {
                        if (ok && reachableAfter(end, use, boundOp))
                        {
                            reportClosureOutlives(use, end, name, captured);
                            ok = false;
                        }
                    }
                }
            }

            if (closure->aliased)
            {
                keepAliasedReleases();
            }

            // a call through the box, after the closure value that owns the box is given back
            for (auto *end : closure->ends)
            {
                for (auto *use : closure->boxUses)
                {
                    if (ok && reachableAfter(end, use, boundOp))
                    {
                        reportClosureOutlives(use, end, name, name);
                        ok = false;
                    }
                }
            }

            toErase.insert(closure->retains.begin(), closure->retains.end());
        }
    }

    bool OwnershipInferencePass::movesCellAway(mlir::Operation *op)
    {
        auto storeOp = mlir::dyn_cast<mlir_ts::StoreOp>(op);
        auto propertyRefOp = storeOp ? storeOp.getReference().getDefiningOp<mlir_ts::PropertyRefOp>() : mlir_ts::PropertyRefOp();
        if (!propertyRefOp || !mlir::isa<mlir_ts::RefType>(storeOp.getValue().getType()))
        {
            return false;
        }

        if (stateBoxes.count(propertyRefOp.getObjectRef()))
        {
            return true;
        }

        auto boundOp = closureOfBox(propertyRefOp.getObjectRef());
        auto found = boundOp ? closureOf.find(boundOp.getOperation()) : closureOf.end();
        return found != closureOf.end() && found->second->escape;
    }

    void OwnershipInferencePass::decideEscaping(Closure &closure, llvm::SetVector<mlir::Operation *> &toErase)
    {
        auto ok = true;
        for (auto &fill : closure.fills)
        {
            auto value = fill.second.getValue();
            mlir::Operation *move = fill.second;
            if (!mlir::isa<mlir_ts::RefType>(value.getType()))
            {
                continue;
            }

            if (isLoadedCell(value))
            {
                reportEscapingCellNotOwned(move, closure.escape, "it is captured from an enclosing function");
                ok = false;
                continue;
            }

            auto varOp = isCellVariable(value) ? value.getDefiningOp<mlir_ts::VariableOp>() : mlir_ts::VariableOp();
            if (!varOp)
            {
                reportCaptureNotOwned(move, "this value");
                ok = false;
                continue;
            }

            auto name = varName(varOp);
            auto elementType = mlir::cast<mlir_ts::RefType>(value.getType()).getElementType();
            // the cell of a parameter this function keeps: its callers gave the value up, so it moves
            // in like an owning local's (a `@keeps` generator's state, spec 38)
            auto init = varOp.getInitializer();
            auto argument = init ? mlir::dyn_cast<mlir::BlockArgument>(init) : mlir::BlockArgument();
            auto keptParam = argument && argument.getOwner()->isEntryBlock() &&
                             argument.getOwner()->getParentOp() == getFunction().getOperation() &&
                             llvm::is_contained(ownedParams(getFunction()), static_cast<int32_t>(argument.getArgNumber()));
            // a handle's cell holds a count of its own, whoever's the value was (spec 23.2)
            if (!isOwningVariable(varOp) && !keptParam && !closure.borrowsValues && ownsHeap(varOp.getLoc(), elementType) &&
                !MLIRTypeHelper::isSharedHandleType(elementType))
            {
                reportEscapingCellNotOwned(move, closure.escape, "its value is the caller's");
                ok = false;
                continue;
            }

            if (takerLoops(move, varOp))
            {
                reportMovedInLoop(move, name);
                ok = false;
                continue;
            }

            llvm::SmallVector<mlir::Operation *> moved;
            for (auto *user : value.getUsers())
            {
                if (user == move || llvm::is_contained(closure.retains, user) || !reachableAfter(move, user, varOp))
                {
                    continue;
                }

                if (!mlir::isa<mlir_ts::ReleaseCellOp>(user))
                {
                    reportUsedAfterEscapingCapture(user, move, name);
                    ok = false;
                    break;
                }

                if (!dominance->properlyDominates(move, user))
                {
                    reportMovedOnSomePaths(move, user, name);
                    ok = false;
                    break;
                }

                moved.push_back(user);
            }

            toErase.insert(moved.begin(), moved.end());
        }

        // A call through the box itself after the closure was given away or given back. followClosure
        // stops at the escape, so `ends` may miss releases past it; that is enough while the only
        // calls through a box are those of a folded `const f`, whose release ends its block. A
        // generator's box has no closure value and no such call.
        if (!closure.op)
        {
            toErase.insert(closure.retains.begin(), closure.retains.end());
            return;
        }

        auto name = ownerName(closure.op.getResult());
        llvm::SmallVector<mlir::Operation *> gone(closure.ends.begin(), closure.ends.end());
        gone.push_back(closure.escape);
        for (auto *end : gone)
        {
            for (auto *use : closure.boxUses)
            {
                if (ok && use != end && reachableAfter(end, use, closure.op.getOperation()))
                {
                    reportUseAfterMove(use, end, name);
                    ok = false;
                }
            }
        }

        toErase.insert(closure.retains.begin(), closure.retains.end());
    }

    void OwnershipInferencePass::decideCells()
    {
        getFunction().walk([&](mlir_ts::VariableOp varOp) {
            auto cell = varOp.getResult();
            if (!isCellVariable(cell) || isOwningVariable(varOp))
            {
                return;
            }

            auto elementType = mlir::cast<mlir_ts::RefType>(cell.getType()).getElementType();
            if (!ownsHeap(varOp.getLoc(), elementType))
            {
                return;
            }

            // A handle's cell holds a count of its own, taken where it is made (spec 23.2): its
            // release gives that back, as under rc, and an assignment is a counted copy.
            if (MLIRTypeHelper::isSharedHandleType(elementType))
            {
                return;
            }

            auto name = varName(varOp);
            auto init = varOp.getInitializer();
            auto argument = init ? mlir::dyn_cast<mlir::BlockArgument>(init) : mlir::BlockArgument();
            if (!argument || !argument.getOwner()->isEntryBlock() ||
                !mlir::isa<mlir_ts::FuncOp>(argument.getOwner()->getParentOp()))
            {
                reportCaptureNotOwned(varOp, name);
                return;
            }

            for (auto *user : cell.getUsers())
            {
                if (mlir::isa<mlir_ts::ReleaseCellOp>(user))
                {
                    setAttrTracked(user, OWN_CELL_BORROWS_ATTR_NAME);
                    continue;
                }

                auto storeOp = mlir::dyn_cast<mlir_ts::StoreOp>(user);
                if (mlir::isa<mlir_ts::ReleaseSlotOp>(user) || (storeOp && storeOp.getReference() == cell))
                {
                    reportCapturedParamAssigned(user, name);
                    return;
                }

                // captured: the closure's body must not assign it either
                auto propertyRefOp = storeOp ? storeOp.getReference().getDefiningOp<mlir_ts::PropertyRefOp>()
                                             : mlir_ts::PropertyRefOp();
                auto boundOp = propertyRefOp ? closureOfBox(propertyRefOp.getObjectRef()) : mlir_ts::CreateBoundFunctionOp();
                if (!boundOp)
                {
                    continue;
                }

                auto assigned = boundOp->getAttrOfType<mlir::DenseI32ArrayAttr>(OWN_ASSIGNS_CAPTURES_ATTR_NAME);
                if (assigned && llvm::is_contained(assigned.asArrayRef(), static_cast<int32_t>(propertyRefOp.getPosition())))
                {
                    reportCapturedParamAssigned(boundOp, name);
                    return;
                }
            }
        });
    }

    void OwnershipInferencePass::decideSlot(mlir::Value slot, llvm::ArrayRef<SlotReceiver> receivers, llvm::SetVector<mlir::Operation *> &toErase)
    {
        auto owner = varName(slot.getDefiningOp<mlir_ts::VariableOp>());
        if (receivers.size() == 1 || takeValuesOfTheirOwn(slot, receivers))
        {
            // a move on some paths only is placed when it is the local's one receiver (spec 30)
            for (auto &receiver : receivers)
            {
                decideSlotReceiver(slot, receiver, owner, toErase, receivers.size() == 1);
            }

            return;
        }

        decideSlotReceivers(slot, receivers, owner, toErase);
    }

    void OwnershipInferencePass::decideSlotReceiver(mlir::Value slot, const SlotReceiver &receiver, llvm::StringRef owner,
                            llvm::SetVector<mlir::Operation *> &toErase, bool sole)
    {
        if (quietly([&] { return checkSlotMove(receiver.retain, receiver.value, receiver.load, toErase, sole); }))
        {
            // rc's retain goes with the move; a call that keeps the argument is the move itself
            if (!isCall(receiver.retain))
            {
                toErase.insert(receiver.retain);
            }

            return;
        }

        if (auto borrower = borrowerOf(receiver.retain))
        {
            borrowWith(borrower, receiver.retain, slotEnds(slot), owner, toErase);
            return;
        }

        checkSlotMove(receiver.retain, receiver.value, receiver.load, toErase, sole); // reports
    }

    bool OwnershipInferencePass::takeValuesOfTheirOwn(mlir::Value slot, llvm::ArrayRef<SlotReceiver> receivers)
    {
        auto assignments = slotAssignments(slot);
        llvm::SmallVector<mlir::Operation *> kills(assignments.begin(), assignments.end());
        if (auto varOp = slot.getDefiningOp<mlir_ts::VariableOp>())
        {
            kills.push_back(varOp);
        }

        for (auto &from : receivers)
        {
            for (auto &to : receivers)
            {
                if (&from != &to && reachableAfter(from.load, to.load, kills))
                {
                    return false;
                }
            }
        }

        return true;
    }

} // namespace own_inference
