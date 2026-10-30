// OwnershipInferencePass (see OwnershipInferencePass.h): the pass itself - each function analyzed,
// its retains decided and erased, string copies made.

#include "OwnershipInferencePass.h"
#include "TypeScript/TypeScriptPassContext.h"

#define DEBUG_TYPE "own"

namespace own_inference
{

    void OwnershipInferencePass::runOnOperation()
    {
        if (!getFunction().isExternal())
        {
            runOnFunction();
        }
    }

    mlir::FunctionOpInterface OwnershipInferencePass::getFunction()
    {
        return getOperation();
    }

    void OwnershipInferencePass::runOnFunction()
    {
        // the dry run (spec 22.3): does today's analysis pass as it is?
        {
            llvm::SetVector<mlir::Operation *> toErase;
            dryRun = true;
            failures = 0;
            auto passed = analyze(toErase);
            dryRun = false;
            // The copies are judged with the dry run's facts still set: a store into a borrowing
            // closure's box, or a borrowing field's, is a borrow only by its
            // `__own_capture_borrow`. They are taken off after, before the real run sets its own.
            // makeStringCopies erases only retains, which the dry run sets no attribute on.
            if (!passed)
            {
                makeStringCopies(toErase);
            }

            for (auto &[op, name] : dryRunAttrs)
            {
                op->removeAttr(name);
            }

            dryRunAttrs.clear();
        }

        llvm::SetVector<mlir::Operation *> toErase;
        if (analyze(toErase))
        {
            releaseCellReads();
        }
    }

    // Does nothing `value` gives - its own value, a field read out of it, a call's result that may be it - leave
    // `block`? Then the last of them that uses it is collected into `last`. A scalar is not followed: it is a copy.
    static bool staysInBlock(mlir::Value value, mlir::Block *block, mlir::Operation *&last, int depth = 0)
    {
        auto isScalar = [](mlir::Type type) {
            return mlir::isa<mlir_ts::NumberType, mlir_ts::BooleanType, mlir::IntegerType, mlir::IndexType,
                             mlir::FloatType>(type);
        };

        auto follow = [&](mlir::Value result) {
            return isScalar(result.getType()) || staysInBlock(result, block, last, depth + 1);
        };

        if (depth > 8)
        {
            return false;
        }

        for (auto *user : value.getUsers())
        {
            if (user->getBlock() != block)
            {
                return false;
            }

            if (!last || last->isBeforeInBlock(user))
            {
                last = user;
            }

            if (mlir::isa<mlir_ts::LengthOfOp, mlir_ts::PrintOp>(user))
            {
                continue;
            }

            // a field of it: what is read from the field is borrowed from it too
            if (mlir::isa<mlir_ts::PropertyRefOp, mlir_ts::InterfaceSymbolRefOp>(user))
            {
                for (auto *refUser : user->getResult(0).getUsers())
                {
                    if (refUser->getBlock() != block)
                    {
                        return false;
                    }

                    if (last->isBeforeInBlock(refUser))
                    {
                        last = refUser;
                    }

                    auto loadOp = mlir::dyn_cast<mlir_ts::LoadOp>(refUser);
                    if (!loadOp || !follow(loadOp.getResult()))
                    {
                        return false;
                    }
                }

                continue;
            }

            // an argument: the result may be it, or borrowed from it
            auto callIndirectOp = mlir::dyn_cast<mlir_ts::CallIndirectOp>(user);
            if ((mlir::isa<mlir_ts::SymbolCallInternalOp>(user) || callIndirectOp) &&
                (!callIndirectOp || callIndirectOp.getCallee() != value))
            {
                for (auto result : user->getResults())
                {
                    if (!follow(result))
                    {
                        return false;
                    }
                }

                continue;
            }

            if (mlir::isa<mlir_ts::LogicalBinaryOp, mlir_ts::ArithmeticBinaryOp, mlir_ts::CastOp>(user) &&
                user->getNumResults() == 1 && follow(user->getResult(0)))
            {
                continue;
            }

            return false;
        }

        return true;
    }

    void OwnershipInferencePass::releaseCellReads()
    {
        llvm::SmallVector<mlir_ts::LoadOp> reads;
        getFunction()->walk([&](mlir_ts::LoadOp loadOp) {
            if (loadOp->hasAttr(OWN_CELL_READ_ATTR_NAME) && isInterfaceAccessorCellRef(loadOp.getReference()))
            {
                reads.push_back(loadOp);
            }
        });

        for (auto loadOp : reads)
        {
            if (loadOp->use_empty())
            {
                loadOp->erase();
                continue;
            }

            mlir::Operation *last = nullptr;
            if (!staysInBlock(loadOp.getResult(), loadOp->getBlock(), last) || !last ||
                last->hasTrait<mlir::OpTrait::IsTerminator>())
            {
                continue;
            }

            mlir::OpBuilder builder(last->getContext());
            builder.setInsertionPointAfter(last);
            auto releaseOp = builder.create<mlir_ts::ReleaseOp>(loadOp.getLoc(), loadOp.getResult());
            releaseOp->setAttr(OWN_CELL_READ_ATTR_NAME, builder.getUnitAttr());
        }
    }

    bool OwnershipInferencePass::analyze(llvm::SetVector<mlir::Operation *> &toErase)
    {
        auto f = getFunction();
        mlir::DominanceInfo dominanceInfo(f);
        dominance = &dominanceInfo;
        // it is this call's: nothing after it (the copies, the next run) may read it
        auto dominanceGone = llvm::make_scope_exit([&]() { dominance = nullptr; });

        resetFunctionState();

        // first: whether a closure borrows what it captures changes what its box's stores are
        classifyClosures();
        classifyLiteralCaptures();
        classifyStateBoxes();
        classifyBorrowingFields();
        // ...and whether an object literal borrows what its fields hold (spec 32)
        classifyRecords();
        // ...and whether an array given to a rest parameter takes its elements (spec 49)
        classifyArgumentArrays();
        // a kept parameter the body may not move on every path is released at its exits (spec 33)
        releaseKeptParamsAtExits();
        auto exitReleasesGone = llvm::make_scope_exit([&]() {
            // the dry run changes nothing: what it put in goes again
            if (dryRun)
            {
                for (auto *release : exitReleases)
                {
                    toErase.remove(release);
                    release->erase();
                }
            }

            exitReleases.clear();
        });

        FunctionOps ops;
        collectOwnershipOps(ops, toErase);
        auto proven = checkOwnershipOps(ops, toErase);

        Receivers slotReceivers;
        Receivers paramReceivers;
        decideRetains(ops, proven, slotReceivers, paramReceivers, toErase);
        giveKeptArguments(ops, slotReceivers, paramReceivers);

        for (auto &entry : slotReceivers)
        {
            decideSlot(entry.first, entry.second, toErase);
        }

        for (auto &entry : paramReceivers)
        {
            decideParam(entry.first, entry.second, toErase);
        }

        decideClosures(toErase);
        decideCells();
        checkBorrowedReads();
        checkBoundedResults();
        checkBorrowingRecords();
        checkRecordResults();
        toErase.insert(fieldBorrowReleases.begin(), fieldBorrowReleases.end());

        if (dryRun)
        {
            return failures == 0;
        }

        // a handle's retains and releases are rc's, and stay (spec 23.2); but not the release of a
        // yielded field's old value: the yield gave the state's count to the caller, who owns the
        // record `next` returns and releases it
        toErase.remove_if([&](mlir::Operation *op) {
            auto releaseSlotOp = mlir::dyn_cast<mlir_ts::ReleaseSlotOp>(op);
            return touchesHandle(op) && !(releaseSlotOp && isYieldedField(releaseSlotOp.getSlot()));
        });
        placeEdgeReleases();
        // what owns nothing gives nothing back, wherever the moves put its releases (spec 36)
        dropUnboxedRecordCounts(toErase);
        for (auto *op : toErase)
        {
            op->erase();
        }

        stripFacts();
        return true;
    }

    void OwnershipInferencePass::resetFunctionState()
    {
        auto f = getFunction();
        drops.clear();
        edgeReleases.clear();
        derivedCache.clear();
        closures.clear();
        literalBoxes.clear();
        closureOf.clear();
        stateBoxes.clear();
        fieldBorrowRetains.clear();
        fieldBorrowReleases.clear();
        mergeBorrowRetains.clear();
        captureStores.clear();
        captureRetains.clear();
        callCopies.clear();
        returnedParams.clear();
        returnsBorrowOf = resultBorrows(f);
        returnsRecordOf = isRecordResult(f) ? resultBorrowsAll(f) : llvm::SmallVector<int>();
        records.clear();
        recordStoreUses.clear();
    }

    // A store of a value that owns a block into a parameter's own slot: the function's argument, which no local
    // that owns what it holds took over (MLIRGen's mlirGenFunctionOwnAssignedParams makes one only of a string or a
    // Shared<T> parameter). Another parameter's value may go there: the caller holds both for the whole call.
    static bool assignsParameter(mlir_ts::StoreOp storeOp)
    {
        auto isParameterSlot = [](mlir_ts::VariableOp slot) {
            auto argument = slot ? mlir::dyn_cast_or_null<mlir::BlockArgument>(slot.getInitializer()) : nullptr;
            return slot && !isOwningVariable(slot) && argument && argument.getOwner()->isEntryBlock() &&
                   mlir::isa<mlir_ts::FuncOp>(argument.getOwner()->getParentOp());
        };

        if (!isParameterSlot(storeOp.getReference().getDefiningOp<mlir_ts::VariableOp>()) || !ownsHeap(storeOp.getValue()))
        {
            return false;
        }

        auto value = storeOp.getValue();
        while (auto castOp = value.getDefiningOp<mlir_ts::CastOp>())
        {
            value = castOp.getIn();
        }

        auto loadOp = value.getDefiningOp<mlir_ts::LoadOp>();
        return !loadOp || !isParameterSlot(loadOp.getReference().getDefiningOp<mlir_ts::VariableOp>());
    }

    void OwnershipInferencePass::collectOwnershipOps(FunctionOps &ops, llvm::SetVector<mlir::Operation *> &toErase)
    {
        auto f = getFunction();
        // A view of a block is that block: the candidate is always the root.
        f.walk([&](mlir::Operation *op) {
            // A handle put into a union with a member that owns a block of its own: counting the
            // union would count that member's single-owner block, and not counting it would lose
            // the handle's count (spec 23.2).
            reportHandleInOwningUnion(op);

            // A handle's retains and releases are rc's, counted, and nobody's to decide (spec 23.2).
            // An overwrite of a field or an element that holds one is still a drop: the old handle
            // may be the last, and its payload goes with it.
            if (touchesHandle(op) && !mlir::isa<mlir_ts::ReleaseSlotOp>(op))
            {
                return;
            }

            if (auto retainOp = mlir::dyn_cast<mlir_ts::RetainOp>(op))
            {
                ops.retains.push_back(op);
                ops.candidates.insert(rootOf(retainOp.getReference()));
            }
            else if (auto retainSlotOp = mlir::dyn_cast<mlir_ts::RetainSlotOp>(op))
            {
                ops.retains.push_back(op);
            }
            else if (auto releaseOp = mlir::dyn_cast<mlir_ts::ReleaseOp>(op))
            {
                ops.candidates.insert(rootOf(releaseOp.getReference()));
            }
            else if (auto varOp = mlir::dyn_cast<mlir_ts::VariableOp>(op))
            {
                if (varOp.getInitializer() && isOwningVariable(varOp) && !isHandle(varOp.getInitializer()))
                {
                    ops.candidates.insert(rootOf(varOp.getInitializer()));
                }
            }
            else if (mlir::isa<mlir_ts::RetainCellOp>(op))
            {
                // a box taking a cell: a closure that borrows it (decideClosures) claimed it
                if (!captureRetains.contains(op) && reporting())
                {
                    op->emitError("closures that capture a variable are not supported by -mm=own here yet");
                    signalPassFailure();
                }
            }
            else if (auto storeOp = mlir::dyn_cast<mlir_ts::StoreOp>(op); storeOp && assignsParameter(storeOp))
            {
                // A parameter's slot given another value: the slot owns nothing, so the value is neither
                // taken from its owner nor retained, and the owner frees it - `p = n; n = mk()` left `p`
                // reading freed memory. Reported where a retain for it would have been, but there is none.
                if (reporting())
                {
                    // a parameter's name is known only from its debug location (--di)
                    auto name = varName(storeOp.getReference().getDefiningOp<mlir_ts::VariableOp>());
                    if (name == "this value")
                    {
                        op->emitError("a parameter cannot be assigned under -mm=own yet");
                    }
                    else
                    {
                        op->emitError("'") << name << "' is a parameter and cannot be assigned under -mm=own yet";
                    }

                    signalPassFailure();
                }
            }
            else if (mlir::isa<mlir_ts::DeleteOp>(op))
            {
                if (reporting())
                {
                    op->emitError("'delete' is not supported by -mm=own yet");
                    signalPassFailure();
                }
            }
            else if (auto loadOp = mlir::dyn_cast<mlir_ts::LoadOp>(op))
            {
                // a yield's read of a field that only yields (OWN_YIELDED_FIELDS_ATTR_NAME) takes
                // the value out: fresh, not a borrow of the state
                // nor is a keeper's element, which no release frees (OWN_KEEPER_READ_ATTR_NAME)
                if (placeReadOf(loadOp.getResult()) && !loadOp->hasAttr(OWN_FRESH_RESULT_ATTR_NAME) &&
                    !loadOp->hasAttr(OWN_KEEPER_READ_ATTR_NAME))
                {
                    ops.placeReads.push_back(loadOp);
                }
                else if (isGlobalRead(loadOp))
                {
                    ops.globalReads.push_back(loadOp);
                }
            }
            else if (auto releaseSlotOp = mlir::dyn_cast<mlir_ts::ReleaseSlotOp>(op); releaseSlotOp && isYieldedField(releaseSlotOp.getSlot()))
            {
                // what the field held was yielded, the caller's now
                toErase.insert(op);
                return;
            }
            else if (auto releaseSlotOp = mlir::dyn_cast<mlir_ts::ReleaseSlotOp>(op))
            {
                // an overwrite of a field or an element, or an assignment to a captured variable or
                // a global (spec 29: the global may hold what `this` or a parameter is)
                auto slot = releaseSlotOp.getSlot();
                if (!releasesNothing(releaseSlotOp) &&
                    (isPlace(slot) || isCellVariable(slot) || isLoadedCell(slot) || slot.getDefiningOp<mlir_ts::AddressOfOp>()))
                {
                    drops.push_back(op);
                }
            }
            else if (mlir::isa<mlir_ts::ArrayPushOp, mlir_ts::ArrayUnshiftOp>(op))
            {
                // through a handle, it may move the elements another handle's borrow reads (spec 23.3)
                if (reachesSharedValue(op->getOperand(0)))
                {
                    drops.push_back(op);
                }
            }
            else if (mlir::isa<mlir_ts::SwapOp>(op))
            {
                // an exchange of two places, which moves their values (spec 42)
                drops.push_back(op);
            }
            else if (mlir::isa<mlir_ts::ArrayPopOp, mlir_ts::ArrayShiftOp, mlir_ts::ArraySpliceOp,
                               mlir_ts::SetLengthOfOp>(op) ||
                     isCall(op))
            {
                drops.push_back(op);
                if (op->getNumResults() == 1 && placeReadOf(op->getResult(0)) == op)
                {
                    ops.placeReads.push_back(op);
                }

                if (!ownedParams(op).empty())
                {
                    ops.keepingCalls.push_back(op);
                }
            }

            for (auto result : op->getResults())
            {
                if (!isView(op) && isFresh(result) && ownsHeap(result) && !isHandle(result))
                {
                    ops.candidates.insert(result);
                }
            }
        });
    }

    llvm::DenseSet<mlir::Value> OwnershipInferencePass::checkOwnershipOps(const FunctionOps &ops, llvm::SetVector<mlir::Operation *> &toErase)
    {
        // Every value gets the owner check, fresh or not: rc can hand a value it did not make to
        // two consuming declarations without a single retain (the result of an indirect call
        // taken by `let b = a; let c = a;`). Only a fresh value's retains can be erased.
        llvm::DenseSet<mlir::Value> proven;
        for (auto value : ops.candidates)
        {
            // a tuple that owns nothing has no owner to find: copies of it are free, and what it
            // borrows bounds every use of it (spec 36)
            if (unboxedRecordFlow(value) == UnboxedFlow::Unboxed)
            {
                continue;
            }

            if (checkValueMoves(value, toErase) && isFresh(value))
            {
                proven.insert(value);
            }
        }

        for (auto read : ops.placeReads)
        {
            checkPlaceRead(read, toErase);
        }

        for (auto read : ops.globalReads)
        {
            checkGlobalRead(read);
        }

        checkArgumentsAssigned();

        return proven;
    }

    void OwnershipInferencePass::decideRetains(const FunctionOps &ops, const llvm::DenseSet<mlir::Value> &proven, Receivers &slotReceivers,
                       Receivers &paramReceivers, llvm::SetVector<mlir::Operation *> &toErase)
    {
        // What is proven loses its retains; any other retain is a second reference nobody
        // proved safe - of a loaded value, a parameter, an unknown producer, or of a fresh value
        // whose own check was reported above.
        // Receivers of each owning local are decided together: one may move it, but where there
        // are several, each borrows it, since a borrow pins its owner.
        for (auto *op : ops.retains)
        {
            if (touchesHandle(op))
            {
                continue;
            }

            if (captureRetains.contains(op))
            {
                continue; // a borrowing box's copy of a value: decideClosures decides it
            }

            if (fieldBorrowRetains.contains(op))
            {
                toErase.insert(op); // a borrowing field's: it takes nothing
                continue;
            }

            if (mergeBorrowRetains.contains(op))
            {
                continue; // a `let` that borrows a merge: borrowsThroughMerge erased it
            }

            if (toErase.contains(op))
            {
                continue; // decided already: a `let` that borrows a field of a record (checkValueMoves)
            }

            // a tuple that owns nothing takes nothing (spec 36)
            if (auto value = retainedValue(op); value && unboxedRecordFlow(value) == UnboxedFlow::Unboxed)
            {
                toErase.insert(op);
                continue;
            }

            // a retain of what holds no block takes nothing: a union's number payload made into
            // another union is a view of the union it came from, but a number all the way
            if (auto retainOp = mlir::dyn_cast<mlir_ts::RetainOp>(op); retainOp && holdsNoBlock(retainOp.getReference()))
            {
                toErase.insert(op);
                continue;
            }

            auto value = retainedValue(op);
            if (value && proven.contains(value))
            {
                toErase.insert(op);
                continue;
            }

            // (not a keeper's element, which is no place read: immortal, decided below)
            if (value && placeReadOf(value) && !isKeptForGood(value))
            {
                continue; // a read out of a container: checkPlaceRead decided it
            }

            // rc's reference for what keeps a parameter this function keeps
            if (auto load = value ? keptParamLoadOf(value) : mlir_ts::LoadOp())
            {
                paramReceivers[load.getReference()].push_back({op, value, load});
                continue;
            }

            // rc's reference for the caller, on a parameter this function returns a borrow of
            if (value && returnsBorrowOf >= 0 && borrowedParam(value) == returnsBorrowOf)
            {
                if (returnedParams.insert(value).second)
                {
                    checkReturnedParam(value, toErase);
                }

                continue;
            }

            if (auto load = value ? slotLoadOf(value) : mlir_ts::LoadOp())
            {
                slotReceivers[load.getReference()].push_back({op, value, load});
                continue;
            }

            if (value && isReadOutOfException(value))
            {
                toErase.insert(op);
                continue;
            }

            if (value && holdsConstantsOrMoves(value))
            {
                toErase.insert(op);
                continue;
            }

            // data nothing owns (isConstantData): a number or null in a union, a function, a
            // literal, a tuple of literals. A slot's retain is decided on its initializer, so only
            // where the slot still holds it.
            if (value && isConstantData(value) && (mlir::isa<mlir_ts::RetainOp>(op) || slotHoldsInitializerAt(op)))
            {
                toErase.insert(op);
                continue;
            }

            // an immortal global's value (spec 25): owned by no one, so returned, stored or kept
            // as often as it is read, and its releases skip it
            if (value && (isImmortalGlobalRead(value) || isKeptForGood(value)) &&
                (mlir::isa<mlir_ts::RetainOp>(op) || slotHoldsInitializerAt(op)))
            {
                toErase.insert(op);
                continue;
            }

            // an array a spread builds, read into the owning local it is made for, or returned
            if (value && mlir::isa<mlir_ts::RetainSlotOp>(op) && isBuiltArrayRead(op, value))
            {
                toErase.insert(op);
                continue;
            }

            if (value && mlir::isa<mlir_ts::RetainOp>(op) && isBuiltArrayReturned(op, value))
            {
                toErase.insert(op);
                continue;
            }

            if (!value || isFresh(value) == false)
            {
                reportSecondReference(op);
            }
        }
    }

    void OwnershipInferencePass::giveKeptArguments(const FunctionOps &ops, Receivers &slotReceivers, Receivers &paramReceivers)
    {
        // An argument a callee keeps moves into the call (spec 2.4): out of an owning local or a
        // parameter this function keeps, like any other taker; a fresh value is decided with the
        // other takers of it, a read out of a container by its own check. Anything else is not
        // this function's to give.
        for (auto *call : ops.keepingCalls)
        {
            auto args = callArgs(call);
            for (auto index : ownedParams(call))
            {
                // a handle given to a callee that keeps it is a counted copy (spec 23.2)
                if (static_cast<size_t>(index) >= args.size() || holdsNoBlock(args[index]) ||
                    isImmortalLiteral(args[index]) || isHandle(args[index]))
                {
                    continue;
                }

                auto arg = args[index];
                auto root = rootOf(arg);
                if (auto load = slotLoadOf(arg))
                {
                    slotReceivers[load.getReference()].push_back({call, root, load});
                }
                else if (auto load = keptParamLoadOf(arg))
                {
                    paramReceivers[load.getReference()].push_back({call, root, load});
                }
                else if (!isFresh(root) && !placeReadOf(root))
                {
                    reportGivenNotOwned(call, arg);
                }
            }
        }
    }

    void OwnershipInferencePass::stripFacts()
    {
        returnedParams.clear();
        getFunction().walk([](mlir::Operation *op) {
            for (auto *name : {OWN_PARAMS_ATTR_NAME, OWN_PARAMS_RELEASED_ATTR_NAME, OWN_RESULT_BORROWS_ATTR_NAME,
                               OWN_RESULT_RECORD_ATTR_NAME, OWN_RESULT_RECORD_UNBOXED_ATTR_NAME,
                               OWN_RECORD_UNBOXED_FIELDS_ATTR_NAME, OWN_RECORD_TEMPORARY_ATTR_NAME,
                               OWN_NO_DROPS_ATTR_NAME, OWN_DROP_FIELDS_ATTR_NAME, OWN_FACTS_LOST_ATTR_NAME,
                               OWN_FRESH_RESULT_ATTR_NAME,
                               OWN_ASSIGNS_CAPTURES_ATTR_NAME,
                               OWN_RESULT_BOUNDED_ATTR_NAME, OWN_STATE_BORROW_ATTR_NAME,
                               OWN_CAPTURE_BORROW_ATTR_NAME, OWN_BORROWING_ARRAY_ATTR_NAME})
            {
                op->removeAttr(name);
            }
        });
    }

    bool OwnershipInferencePass::reporting()
    {
        if (quiet)
        {
            return false;
        }

        if (dryRun)
        {
            ++failures;
            return false;
        }

        return true;
    }

    void OwnershipInferencePass::setAttrTracked(mlir::Operation *op, llvm::StringRef name)
    {
        auto attrName = mlir::StringAttr::get(&getContext(), name);
        if (dryRun && !op->hasAttr(attrName))
        {
            dryRunAttrs.push_back({op, attrName});
        }

        op->setAttr(attrName, mlir::UnitAttr::get(&getContext()));
    }

    void OwnershipInferencePass::makeStringCopies(const llvm::SetVector<mlir::Operation *> &erased)
    {
        llvm::SmallVector<mlir::Operation *> retains;
        getFunction()->walk([&](mlir::Operation *op) {
            if (mlir::isa<mlir_ts::RetainOp, mlir_ts::RetainSlotOp>(op) && !erased.contains(op) &&
                !captureRetains.contains(op) && !fieldBorrowRetains.contains(op))
            {
                retains.push_back(op);
            }
        });

        for (auto *retain : retains)
        {
            copyForRetain(retain);
        }

        for (auto [read, call] : callCopies)
        {
            auto value = read->getResult(0);
            if (!llvm::is_contained(call->getOperands(), value))
            {
                continue;
            }

            // the place owns the copy, or a count of a handle's of its own (spec 23.2)
            if (auto storeOp = mlir::dyn_cast<mlir_ts::StoreOp>(call))
            {
                mlir::OpBuilder builder(storeOp);
                if (isHandle(value))
                {
                    builder.create<mlir_ts::RetainOp>(storeOp->getLoc(), value);
                    continue;
                }

                auto copy = builder.create<mlir_ts::StringCopyOp>(storeOp->getLoc(), value.getType(), value);
                storeOp.getValueMutable().assign(copy);
                continue;
            }

            // a callee that keeps it takes the copy; else it is released after the call
            auto args = callArgs(call);
            auto kept = llvm::any_of(ownedParams(call), [&](int32_t index) {
                return static_cast<size_t>(index) < args.size() && args[index] == value;
            });

            mlir::OpBuilder builder(call);
            // a handle: a count of its own for the call, given back after it (spec 50)
            if (isHandle(value))
            {
                builder.create<mlir_ts::RetainOp>(call->getLoc(), value);
                if (!kept)
                {
                    builder.setInsertionPointAfter(call);
                    builder.create<mlir_ts::ReleaseOp>(call->getLoc(), value);
                }

                continue;
            }

            auto copy = builder.create<mlir_ts::StringCopyOp>(call->getLoc(), value.getType(), value);
            for (auto &operand : call->getOpOperands())
            {
                if (operand.get() == value)
                {
                    operand.set(copy);
                }
            }

            if (!kept)
            {
                builder.setInsertionPointAfter(call);
                builder.create<mlir_ts::ReleaseOp>(call->getLoc(), copy);
            }
        }
    }

    bool OwnershipInferencePass::borrowsKnownParam(mlir::Value value)
    {
        auto param = borrowedParam(value);
        return param >= 0 && (llvm::is_contained(ownedParams(getFunction()), param) || param == returnsBorrowOf);
    }

    bool OwnershipInferencePass::mayBeReturned(mlir_ts::VariableOp varOp)
    {
        llvm::SmallVector<mlir::Value> slots{varOp.getResult()};
        llvm::SmallPtrSet<mlir::Value, 4> seen{varOp.getResult()};
        while (!slots.empty())
        {
            auto slot = slots.pop_back_val();
            for (auto *user : slot.getUsers())
            {
                auto loadOp = mlir::dyn_cast<mlir_ts::LoadOp>(user);
                if (!loadOp)
                {
                    continue;
                }

                auto returned = false;
                forEachUse(loadOp.getResult(), [&](mlir::Operation *loadUser, mlir::Value used) {
                    if (mlir::isa<mlir_ts::ReturnInternalOp>(loadUser))
                    {
                        returned = true;
                        return;
                    }

                    mlir::Value other;
                    if (auto otherVar = mlir::dyn_cast<mlir_ts::VariableOp>(loadUser))
                    {
                        other = otherVar.getResult();
                    }
                    else if (auto storeOp = mlir::dyn_cast<mlir_ts::StoreOp>(loadUser);
                             storeOp && storeOp.getValue() == used && storeOp.getReference().getDefiningOp<mlir_ts::VariableOp>())
                    {
                        other = storeOp.getReference();
                    }

                    if (other && seen.insert(other).second)
                    {
                        slots.push_back(other);
                    }
                });

                if (returned)
                {
                    return true;
                }
            }
        }

        return false;
    }

    bool OwnershipInferencePass::copyForRetain(mlir::Operation *retain)
    {
        // `let x = v`: the local starts from a copy
        if (auto retainSlotOp = mlir::dyn_cast<mlir_ts::RetainSlotOp>(retain))
        {
            auto varOp = retainSlotOp.getSlot().getDefiningOp<mlir_ts::VariableOp>();
            auto init = varOp ? varOp.getInitializer() : mlir::Value();
            if (!init || !isCopyableString(init.getType()) || isCellVariable(varOp.getResult()))
            {
                return false;
            }

            // a read of the parameter the result borrows, into a local that is not returned (a swap's
            // `let v = this[i]` in a function that returns `this`): a copy, as any read is
            if (borrowsKnownParam(init) &&
                !(borrowedParam(init) == returnsBorrowOf &&
                  !llvm::is_contained(ownedParams(getFunction()), returnsBorrowOf) && !mayBeReturned(varOp)))
            {
                return false;
            }

            mlir::OpBuilder builder(varOp);
            auto copy = builder.create<mlir_ts::StringCopyOp>(varOp.getLoc(), init.getType(), init);
            varOp.getInitializerMutable().assign(copy);
            retain->erase();
            return true;
        }

        auto retainOp = mlir::cast<mlir_ts::RetainOp>(retain);
        auto value = retainOp.getReference();
        if (copyIntoTuple(retainOp))
        {
            return true;
        }

        if (!isCopyableString(value.getType()))
        {
            return false;
        }

        // the one use rc retained it for
        mlir::OpOperand *taking = nullptr;
        auto several = false;
        forEachUse(value, [&](mlir::Operation *user, mlir::Value used) {
            if (mlir::isa<mlir_ts::RetainOp, mlir_ts::ReleaseOp>(user) || isBorrow(user, used))
            {
                return;
            }

            for (auto &operand : user->getOpOperands())
            {
                if (operand.get() == used)
                {
                    several = several || taking;
                    taking = &operand;
                }
            }
        });

        if (!taking || several || !isCopyableString(taking->get().getType()))
        {
            return false;
        }

        // A read of a parameter this function owns, or of the one its result borrows, stays what it
        // is: the borrow it returns, or the parameter's own. Stored into an element or a field
        // (`this[t] = this[s]` in a function that returns `this`) it is a copy, as any read is; not
        // into a local, which may be what is returned.
        auto storeOp = mlir::dyn_cast<mlir_ts::StoreOp>(taking->getOwner());
        auto intoPlace = storeOp && storeOp.getValue() == taking->get() &&
                         mlir::isa_and_nonnull<mlir_ts::ElementRefOp, mlir_ts::PropertyRefOp>(storeOp.getReference().getDefiningOp());
        if (borrowsKnownParam(value) &&
            !(intoPlace && borrowedParam(value) == returnsBorrowOf &&
              !llvm::is_contained(ownedParams(getFunction()), returnsBorrowOf)))
        {
            return false;
        }

        // only a retain in the use's block: a copy made elsewhere (rc's birth retain, after the
        // producer) has no owner on a path that does not reach the use; rc retains for the use
        // beside it too
        if (taking->getOwner()->getBlock() != retain->getBlock())
        {
            return false;
        }

        // made where rc took its reference: a store releases what it overwrites first, and that may
        // be the very string (`b.s = b.s`), or one the analysis cannot tell from it (`b.s = a.s`)
        auto *at = taking->getOwner();
        auto *def = taking->get().getDefiningOp();
        if (taking->get() == value || (def && def->getBlock() == retain->getBlock() && def->isBeforeInBlock(retain)))
        {
            // the use reads the copy, so the retain must come before it
            if (!retain->isBeforeInBlock(taking->getOwner()))
            {
                return false;
            }

            at = retain;
        }

        mlir::OpBuilder builder(at);
        auto copy = builder.create<mlir_ts::StringCopyOp>(retain->getLoc(), taking->get().getType(), taking->get());
        taking->set(copy);
        retain->erase();
        return true;
    }

    bool OwnershipInferencePass::copyIntoTuple(mlir_ts::RetainOp retainOp)
    {
        // `[s, n]` of a borrowed string: rc retains the tuple, which retains the string in it. The
        // tuple starts from a copy instead, which is its own (spec 22). A handle in it takes a count
        // of its own, which own leaves to rc (spec 23.2): `[key, shared]`. Only when strings and
        // handles are all it holds blocks of: a borrowed object in it cannot be copied.
        auto createTupleOp = retainOp.getReference().getDefiningOp<mlir_ts::CreateTupleOp>();
        if (!createTupleOp || createTupleOp->getBlock() != retainOp->getBlock() ||
            !createTupleOp->isBeforeInBlock(retainOp))
        {
            return false;
        }

        llvm::SmallVector<mlir::OpOperand *> borrowed;
        llvm::SmallVector<mlir::Value> handles;
        for (auto &operand : createTupleOp->getOpOperands())
        {
            auto field = operand.get();
            if (holdsNoBlock(field) || isConstantData(field) || isFresh(rootOf(field)))
            {
                continue;
            }

            if (isHandle(field))
            {
                handles.push_back(field);
                continue;
            }

            if (!isCopyableString(field.getType()) || borrowsKnownParam(field))
            {
                return false;
            }

            borrowed.push_back(&operand);
        }

        if (borrowed.empty() && handles.empty())
        {
            return false;
        }

        mlir::OpBuilder builder(createTupleOp);
        for (auto *operand : borrowed)
        {
            operand->set(builder.create<mlir_ts::StringCopyOp>(retainOp->getLoc(), operand->get().getType(), operand->get()));
        }

        builder.setInsertionPoint(retainOp);
        for (auto handle : handles)
        {
            builder.create<mlir_ts::RetainOp>(retainOp->getLoc(), handle);
        }

        retainOp->erase();
        return true;
    }

} // namespace own_inference

#undef DEBUG_TYPE

std::unique_ptr<mlir::Pass> mlir_ts::createOwnershipInferencePass()
{
    return std::make_unique<own_inference::OwnershipInferencePass>();
}
