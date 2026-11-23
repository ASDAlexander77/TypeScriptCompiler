// OwnershipInferencePass (see OwnershipInferencePass.h): the pass itself - each function analyzed,
// its retains decided and erased, string copies made.

#include "OwnershipInferencePass.h"
#include "TypeScript/TypeScriptPassContext.h"

#include "mlir/IR/Diagnostics.h"

#include <optional>
#include <thread>

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

    // `testIFace(o); testIFace(o)` of an object literal `o`: each argument is a view of the literal's
    // block made for its call (`ts.NewInterface`), retained before the call and released at the end
    // of the scope. Under rc each pair gives back what it took. Under own the first retain takes the
    // block and the last release frees it: the pairs in between take nothing and give nothing back,
    // as long as every view is used before that release, and only by calls that do not keep it.
    static void joinViewRetainPairs(mlir::FunctionOpInterface f)
    {
        struct View
        {
            mlir::Operation *view;
            mlir::Operation *retain;
            mlir::Operation *release;
            llvm::SmallVector<mlir::Operation *> uses;
        };

        llvm::MapVector<mlir::Value, llvm::SmallVector<View>> groups;
        f->walk([&](mlir_ts::NewInterfaceOp viewOp) {
            auto root = rootOf(viewOp.getResult());
            auto newOp = root.getDefiningOp<mlir_ts::NewOp>();
            if (!newOp || !mlir::isa<mlir_ts::ValueRefType>(newOp.getType()))
            {
                return;
            }

            View view{viewOp, nullptr, nullptr, {}};
            for (auto &use : viewOp->getUses())
            {
                auto *user = use.getOwner();
                if (user->getBlock() != viewOp->getBlock())
                {
                    return;
                }

                // (`--di`'s record of a `const` that holds the view)
                if (mlir::isa<mlir_ts::DebugVariableOp>(user))
                {
                    view.uses.push_back(user);
                    continue;
                }

                if (mlir::isa<mlir_ts::RetainOp>(user) && !view.retain)
                {
                    view.retain = user;
                    continue;
                }

                if (mlir::isa<mlir_ts::ReleaseOp>(user) && !view.release)
                {
                    view.release = user;
                    continue;
                }

                // an argument of a call that does not keep it
                if (isCall(user) && !llvm::is_contained(ownedParams(user), static_cast<int32_t>(
                                                                                 use.getOperandNumber() - callArgs(user).getBeginOperandIndex())))
                {
                    view.uses.push_back(user);
                    continue;
                }

                // ...or its object and its methods, read for a call (`o.toString()` through the view): the
                // object, the method, the method bound to the object, each only on its way to a call
                std::function<bool(mlir::Operation *, mlir::Value, int)> forCall = [&](mlir::Operation *op, mlir::Value from, int depth) {
                    if (depth > 6 || op->getBlock() != viewOp->getBlock())
                    {
                        return false;
                    }

                    view.uses.push_back(op);
                    if (isCall(op))
                    {
                        // (not kept by it)
                        auto args = callArgs(op);
                        return llvm::none_of(ownedParams(op), [&](int32_t index) {
                            return static_cast<size_t>(index) < args.size() && args[index] == from;
                        });
                    }

                    // ...or a field read out of it that owns nothing (`o.r` of a number)
                    if (mlir::isa<mlir_ts::LoadOp>(op) && op->getNumResults() == 1 && !ownsHeap(op->getResult(0)))
                    {
                        return true;
                    }

                    if (!mlir::isa<mlir_ts::ExtractInterfaceThisOp, mlir_ts::InterfaceSymbolRefOp, mlir_ts::LoadOp,
                                   mlir_ts::CreateBoundFunctionOp, mlir_ts::GetThisOp, mlir_ts::GetMethodOp>(op) ||
                        op->getNumResults() != 1)
                    {
                        return false;
                    }

                    return llvm::all_of(op->getUsers(), [&](mlir::Operation *next) { return forCall(next, op->getResult(0), depth + 1); });
                };

                if (mlir::isa<mlir_ts::ExtractInterfaceThisOp, mlir_ts::InterfaceSymbolRefOp>(user) &&
                    forCall(user, viewOp.getResult(), 0))
                {
                    continue;
                }

                return;
            }

            if (view.retain && view.release && view.retain->isBeforeInBlock(view.release))
            {
                groups[root].push_back(view);
            }
        });

        for (auto &[root, views] : groups)
        {
            if (views.size() < 2)
            {
                continue;
            }

            // every view in one block, and nothing else made of the block but these views
            auto *block = views.front().view->getBlock();
            auto viewsOnly = llvm::all_of(views, [&](View &view) { return view.view->getBlock() == block; });
            forEachUse(root, [&](mlir::Operation *user, mlir::Value) {
                auto storeOp = mlir::dyn_cast<mlir_ts::StoreOp>(user);
                // (the store that fills the block, and `--di`'s record of the local for the debugger,
                // which names it and keeps nothing)
                if ((storeOp && storeOp.getReference() == root) || mlir::isa<mlir_ts::DebugVariableOp>(user))
                {
                    return;
                }

                viewsOnly = viewsOnly && llvm::any_of(views, [&](View &view) {
                                return view.retain == user || view.release == user || llvm::is_contained(view.uses, user);
                            });
            });

            if (!viewsOnly)
            {
                continue;
            }

            auto *first = views.front().retain;
            auto *last = views.front().release;
            for (auto &view : views)
            {
                first = view.retain->isBeforeInBlock(first) ? view.retain : first;
                last = last->isBeforeInBlock(view.release) ? view.release : last;
            }

            // every use before the one release left
            auto before = llvm::all_of(views, [&](View &view) {
                return llvm::all_of(view.uses, [&](mlir::Operation *use) { return use->isBeforeInBlock(last); });
            });
            if (!before)
            {
                continue;
            }

            for (auto &view : views)
            {
                if (view.retain != first)
                {
                    view.retain->erase();
                }

                if (view.release != last)
                {
                    view.release->erase();
                }
            }
        }
    }

    void OwnershipInferencePass::runOnFunction()
    {
        // what the real run may do that the dry run did not: only where the dry run failed, so a
        // function today's analysis accepts is decided as it was
        localLetCandidates.clear();
        countedValues.clear();
        countedOrder.clear();
        moduleHoldsDescriptorObjects.reset();
        dryRunDiagnostics.clear();
        dryRunCapture.clear();
        countedErrors.clear();
        reportSubject = {};
        armedSubject = {};
        failureSignalled = false;
        uncountedErrors = 0;

        // --own-fallback=count: what the report sites emit is seen here first. In the dry run it is
        // recorded and swallowed; in the real run an error about a counted value becomes a note of
        // its warning, and any other stays an error. Only this thread's: the pass runs on the
        // module's functions at once, and a handler is the context's.
        std::optional<mlir::ScopedDiagnosticHandler> fallbackHandler;
        if (countsFallback)
        {
            auto thread = std::this_thread::get_id();
            fallbackHandler.emplace(&getContext(), [this, thread](mlir::Diagnostic &diag) -> mlir::LogicalResult {
                if (std::this_thread::get_id() != thread || diag.getSeverity() != mlir::DiagnosticSeverity::Error)
                {
                    return mlir::failure();
                }

                // the value the report is about: the one its `reporting()` armed, used once
                auto subject = armedSubject;
                armedSubject = {};

                // a refusal by design is no reason to count, and is never a counted value's note
                if (refusingByDesign > 0)
                {
                    if (dryRun)
                    {
                        return mlir::success();
                    }

                    ++uncountedErrors;
                    return mlir::failure();
                }

                RecordedDiagnostic recorded{diag.getLocation(), diag.str(), {}};
                for (auto &note : diag.getNotes())
                {
                    recorded.notes.push_back({note.getLocation(), note.str()});
                }

                if (dryRun)
                {
                    dryRunCapture.push_back({subject, std::move(recorded)});
                    return mlir::success();
                }

                // only an error about a counted value is that value's note
                if (auto unit = countedUnitOf(subject); unit && countedValues.contains(unit))
                {
                    countedErrors.push_back({unit, std::move(recorded)});
                    return mlir::success();
                }

                ++uncountedErrors;
                return mlir::failure();
            });
        }

        // the dry run (spec 22.3): does today's analysis pass as it is?
        {
            llvm::SetVector<mlir::Operation *> toErase;
            dryRun = true;
            failures = 0;
            auto passed = analyze(toErase);
            dryRun = false;
            // (what a report signalled under the flag is not this function's failure: the real run decides)
            failureSignalled = false;
            // The copies are judged with the dry run's facts still set: a store into a borrowing
            // closure's box, or a borrowing field's, is a borrow only by its
            // `__own_capture_borrow`. They are taken off after, before the real run sets its own.
            // makeStringCopies erases only retains, which the dry run sets no attribute on.
            if (!passed)
            {
                // the values the reports name, counted where they can be (the copies leave their
                // retains alone)
                if (countsFallback)
                {
                    countRefusedValues(toErase);
                }

                makeStringCopies(toErase);
                if (!countedValues.empty())
                {
                    retainKeptCountedArguments();
                }

                splitBorrowingLets();
                // (after the copies, which read the dry run's erased retains)
                joinViewRetainPairs(getFunction());
            }
            else
            {
                localLetCandidates.clear();
            }

            splitLetCandidates.clear();

            for (auto &[op, name] : dryRunAttrs)
            {
                op->removeAttr(name);
            }

            dryRunAttrs.clear();
        }

        llvm::SetVector<mlir::Operation *> toErase;
        if (analyze(toErase))
        {
            // (its refusals are about a getter's value, never a count: they stay errors)
            ++refusingByDesign;
            releaseCellReads();
            --refusingByDesign;
        }

        if (!countsFallback)
        {
            return;
        }

        markCountedRetains();
        warnCounted();
        // an error nobody counted, or a failure no counted value's error explains
        if (uncountedErrors > 0 || (failureSignalled && countedErrors.empty()))
        {
            PassWrapper::signalPassFailure();
        }
    }

    void OwnershipInferencePass::signalPassFailure()
    {
        if (countsFallback)
        {
            failureSignalled = true;
            return;
        }

        PassWrapper::signalPassFailure();
    }

    // Does nothing `value` gives - its own value, a field read out of it, a call's result that may be it - leave
    // `block` (the function, when null)? Then the last of them that uses it is collected into `last` (in `block`
    // only). A scalar is not followed: it is a copy.
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
            // --di's record of a const's value (`const x = i.item`) only names it to the debugger
            if (mlir::isa<mlir_ts::DebugVariableOp>(user))
            {
                continue;
            }

            if (block && user->getBlock() != block)
            {
                return false;
            }

            if (block && (!last || last->isBeforeInBlock(user)))
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
                    if (block && refUser->getBlock() != block)
                    {
                        return false;
                    }

                    if (block && last->isBeforeInBlock(refUser))
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

            // a method of it (`i.item.get()`): the method bound to it, its object and its function, which only
            // the call below uses
            if (mlir::isa<mlir_ts::ThisSymbolRefOp, mlir_ts::ThisVirtualSymbolRefOp, mlir_ts::GetThisOp,
                          mlir_ts::GetMethodOp>(user) &&
                user->getNumResults() == 1 && follow(user->getResult(0)))
            {
                continue;
            }

            // an argument, or the method called: the result may be it, or borrowed from it
            if (mlir::isa<mlir_ts::SymbolCallInternalOp, mlir_ts::CallIndirectOp>(user))
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

            mlir::OpBuilder builder(loadOp->getContext());
            auto release = [&](mlir::Operation *at) {
                auto releaseOp = builder.create<mlir_ts::ReleaseOp>(loadOp.getLoc(), loadOp.getResult());
                releaseOp->setAttr(OWN_CELL_READ_ATTR_NAME, builder.getUnitAttr());
            };

            mlir::Operation *last = nullptr;
            if (staysInBlock(loadOp.getResult(), loadOp->getBlock(), last) && last &&
                !last->hasTrait<mlir::OpTrait::IsTerminator>())
            {
                builder.setInsertionPointAfter(last);
                release(last);
                continue;
            }

            // ...or put in a `let` (`let x = i.item`) that is only read, across blocks too, and nothing read
            // out of it leaves the function: freed at the function's returns, when the read reaches them all
            // and the object it reads through is a parameter's, which outlives them
            auto exits = localHeldToExits(loadOp);
            for (auto *exit : exits)
            {
                builder.setInsertionPoint(exit);
                release(exit);
            }

            // ...else a getter's new value would be lost: returned, stored, or kept where nothing frees it. Only
            // a field some class of this module fills with accessors has one; its value cannot leave its read
            auto refOp = loadOp.getReference().getDefiningOp<mlir_ts::InterfaceSymbolRefOp>();
            if (!exits.empty() || !refOp)
            {
                continue;
            }

            auto fieldAttr = mlir::dyn_cast<mlir::StringAttr>(refOp.getIdentifier());
            auto filler = fieldAttr ? accessorFiller(fieldAttr.getValue()) : llvm::StringRef();
            if (!filler.empty())
            {
                loadOp.emitError("'")
                    << fieldAttr.getValue() << "' is read through an interface, and '" << filler
                    << "' fills it with a getter, whose new value -mm=own frees only where it is read: it cannot "
                       "be returned, kept or outlive its block; read the field where it is used";
                signalPassFailure();
            }
        }
    }

    llvm::StringRef OwnershipInferencePass::accessorFiller(llvm::StringRef field)
    {
        // the cell of a class's accessors for an interface's field: `<class>.<interface>..vtbl.<field>.accessor`
        auto suffix = (llvm::Twine(".") + field + ".accessor").str();
        auto module = getFunction()->getParentOfType<mlir::ModuleOp>();
        llvm::StringRef found;
        // Only the module's own operations, which no function's pass changes: this pass runs on the
        // module's functions at once, on several threads, and module.walk went into the bodies of
        // the functions the other threads were putting releases in and erasing reads from. It read a
        // list as it was relinked, and the compiler ended with an access violation about once in
        // 500 runs (test/tester/own/own_field_chain_let.ts).
        for (auto &moduleOp : *module.getBody())
        {
            auto globalOp = mlir::dyn_cast<mlir_ts::GlobalOp>(&moduleOp);
            if (!globalOp)
            {
                continue;
            }

            auto name = globalOp.getSymName();
            auto vtbl = name.find("..vtbl.");
            if (vtbl != llvm::StringRef::npos && name.ends_with(suffix) && vtbl + 6 + suffix.size() == name.size())
            {
                // the class: up to the interface's name
                auto head = name.take_front(vtbl);
                found = head.take_front(head.rfind('.') == llvm::StringRef::npos ? head.size() : head.rfind('.'));
                break;
            }
        }

        return found;
    }

    llvm::SmallVector<mlir::Operation *> OwnershipInferencePass::localHeldToExits(mlir_ts::LoadOp read)
    {
        auto result = read.getResult();
        auto varOp = result.hasOneUse() ? mlir::dyn_cast<mlir_ts::VariableOp>(*result.getUsers().begin()) : mlir_ts::VariableOp();
        if (!varOp || varOp.getInitializer() != result || varOp.getCaptured().value_or(false))
        {
            return {};
        }

        // only read, and what is read stays in the function
        for (auto *user : varOp->getUsers())
        {
            mlir::Operation *ignored = nullptr;
            auto slotRead = mlir::dyn_cast<mlir_ts::LoadOp>(user);
            if (!mlir::isa<mlir_ts::DebugVariableOp>(user) &&
                (!slotRead || !staysInBlock(slotRead.getResult(), /*block=*/nullptr, ignored)))
            {
                return {};
            }
        }

        // read through a parameter: `i.item` of an interface parameter `i`
        auto refOp = read.getReference().getDefiningOp<mlir_ts::InterfaceSymbolRefOp>();
        auto objectRead = refOp ? refOp->getOperand(0).getDefiningOp<mlir_ts::LoadOp>() : mlir_ts::LoadOp();
        auto objectSlot = objectRead ? objectRead.getReference().getDefiningOp<mlir_ts::VariableOp>() : mlir_ts::VariableOp();
        auto index = -1;
        if (!objectSlot || !isParameterSlot(objectSlot, index))
        {
            return {};
        }

        // every return, after the read
        auto f = getFunction();
        mlir::DominanceInfo dominanceInfo(f);
        llvm::SmallVector<mlir::Operation *> exits;
        auto all = true;
        f.walk([&](mlir_ts::ReturnInternalOp returnOp) {
            exits.push_back(returnOp);
            all = all && dominanceInfo.properlyDominates(read.getOperation(), returnOp.getOperation());
        });

        return all ? exits : llvm::SmallVector<mlir::Operation *>();
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
        // ...and whether an array literal in a local only borrows its elements (TSLANG 55)
        classifyLocalBorrowingArrays();
        classifyArrayViews();
        classifyBorrowingBoxes();
        // ...and whether a `let` given reads of owning locals borrows each of them
        classifyLocalBorrowingLets();
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
        // what the walk refuses is a shape (a handle in an owning union, a cell, `delete`, a parameter
        // assigned, an Opaque or @weak store), never a value with a retain to keep: no count explains it
        ++refusingByDesign;
        collectOwnershipOps(ops, toErase);
        --refusingByDesign;
        auto proven = checkOwnershipOps(ops, toErase);

        Receivers slotReceivers;
        Receivers paramReceivers;
        decideRetains(ops, proven, slotReceivers, paramReceivers, toErase);
        giveKeptArguments(ops, slotReceivers, paramReceivers);

        for (auto &entry : slotReceivers)
        {
            SubjectScope subject(*this, entry.first);
            decideSlot(entry.first, entry.second, toErase);
        }

        for (auto &entry : paramReceivers)
        {
            SubjectScope subject(*this, entry.first);
            decideParam(entry.first, entry.second, toErase);
        }

        decideClosures(toErase);
        decideCells();
        checkBorrowedReads();
        checkBoundedResults();
        ++refusingByDesign;
        checkRestTakes();
        checkCannotShare();
        --refusingByDesign;
        checkBorrowingRecords();
        checkRecordResults();
        checkLetHolders(toErase);
        toErase.insert(fieldBorrowReleases.begin(), fieldBorrowReleases.end());

        if (dryRun)
        {
            return failures == 0;
        }

        // a handle's retains and releases are rc's, and stay (spec 23.2); but not the release of a
        // yielded field's old value: the yield gave the state's count to the caller, who owns the
        // record `next` returns and releases it; nor the retain for a box that borrows what it holds:
        // that box gives nothing back
        // ...nor the release of a fresh handle stored into a record literal that moves (recordRetainOf:
        // `const o = { w: new WeakRef(n) }`, `return { s: new Shared(k) }`): rc's retain of the record
        // is erased as the move, so the handle's count moved in with it, and the record's release
        // gives it back; the temporary's release would give it back twice
        llvm::SmallPtrSet<mlir::Operation *, 4> movedHandleReleases;
        getFunction().walk([&](mlir_ts::ReleaseOp releaseOp) {
            auto value = releaseOp.getReference();
            if (!isHandle(value))
            {
                return;
            }

            for (auto *user : value.getUsers())
            {
                auto *retain = recordRetainOfStore(user);
                if (retain && toErase.contains(retain))
                {
                    movedHandleReleases.insert(releaseOp);
                    return;
                }
            }
        });

        toErase.remove_if([&](mlir::Operation *op) {
            auto releaseSlotOp = mlir::dyn_cast<mlir_ts::ReleaseSlotOp>(op);
            return touchesHandle(op) && !(releaseSlotOp && isYieldedField(releaseSlotOp.getSlot())) && !captureRetains.contains(op) &&
                   !movedHandleReleases.contains(op);
        });
        toErase.insert(movedHandleReleases.begin(), movedHandleReleases.end());
        keepCountedOps(toErase);
        placeEdgeReleases();
        // what owns nothing gives nothing back, wherever the moves put its releases (spec 36)
        dropUnboxedRecordCounts(toErase);
        // a select of two fresh values moved both into what it selected: the other is released where
        // rc's retain of the selected one was - the point the moves were checked from, after which
        // nothing uses either - chosen by the same condition the other way round
        for (auto *op : selectMoves)
        {
            auto selectOp = mlir::cast<mlir::arith::SelectOp>(op);
            auto *retain = selectRetainOf(selectOp, selectOp.getTrueValue());
            mlir::OpBuilder builder(selectOp->getContext());
            if (retain)
            {
                builder.setInsertionPoint(retain);
            }
            else
            {
                builder.setInsertionPointAfter(selectOp);
            }

            auto notSelected = builder.create<mlir::arith::SelectOp>(selectOp.getLoc(), selectOp.getCondition(),
                                                                      selectOp.getFalseValue(), selectOp.getTrueValue());
            builder.create<mlir_ts::ReleaseOp>(selectOp.getLoc(), notSelected.getResult());
        }

        // (what the steps after keepCountedOps put in, too)
        if (!countedValues.empty())
        {
            toErase.remove_if([&](mlir::Operation *op) { return touchesCounted(op); });
        }

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
        slotClears.clear();
        derivedCache.clear();
        closures.clear();
        literalBoxes.clear();
        closureOf.clear();
        stateBoxes.clear();
        fieldBorrowRetains.clear();
        fieldBorrowReleases.clear();
        mergeBorrowRetains.clear();
        mergedSlotRetains.clear();
        localBorrowingLets.clear();
        captureStores.clear();
        captureRetains.clear();
        callCopies.clear();
        selectMoves.clear();
        returnedParams.clear();
        // the stores that take a value out of the object their place holds, for every walk of this
        // function to see (never another function's: its operations are gone, their addresses reused)
        splicedTakers.clear();
        f->walk([&](mlir_ts::LoadOp loadOp) {
            FieldTake take;
            if (splicedFieldTakeOf(loadOp, take))
            {
                splicedTakers.insert(take.taker);
            }
        });
        returnsBorrowOf = resultBorrows(f);
        returnsRecordOf = isRecordResult(f) ? resultBorrowsAll(f) : llvm::SmallVector<int>();
        records.clear();
        recordStoreUses.clear();
        arrayElementUses.clear();
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

            // ...nor a counted value's (--own-fallback=count): rc's, and they stay
            if (!dryRun && touchesCounted(op))
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
            else if (auto castOp = mlir::dyn_cast<mlir_ts::CastOp>(op);
                     castOp && mlir::isa<mlir_ts::OpaqueType>(castOp.getType()) && ownsHeap(castOp.getIn()))
            {
                // `<Opaque>new Item()`: an Opaque only borrows, so a temporary nothing else holds would be
                // freed under it (spec 92). A `const` or a local holding the object owns it. Only an
                // Opaque that is kept - stored in a field or a variable - outlives the statement: one
                // given straight to a call (a bound method's `this`, a generic's argument) is read while
                // the temporary is still alive. A scalar boxed into `any` (`<Opaque><any>0`) holds no object.
                std::function<bool(mlir::Value)> storedValue = [&](mlir::Value value) {
                    return llvm::any_of(value.getUsers(), [&](mlir::Operation *user) {
                        if (auto storeOp = mlir::dyn_cast<mlir_ts::StoreOp>(user))
                            return storeOp.getValue() == value;
                        if (mlir::isa<mlir_ts::CastOp, mlir_ts::OptionalValueOp>(user))
                            return storedValue(user->getResult(0));
                        return mlir::isa<mlir_ts::VariableOp>(user);
                    });
                };
                auto stored = storedValue(castOp.getResult());
                auto *def = rootOf(castOp.getIn()).getDefiningOp();
                if (stored && !mlir::isa<mlir_ts::AnyType>(castOp.getIn().getType()) && def &&
                    isFresh(rootOf(castOp.getIn())) && !def->hasAttr(OWNED_RESULT_NAMED_ATTR_NAME) &&
                    llvm::none_of(def->getUsers(), [](mlir::Operation *user) { return mlir::isa<mlir_ts::VariableOp>(user); }) &&
                    reporting())
                {
                    op->emitError("an Opaque only borrows what it points at, and nothing else keeps this object under "
                                  "-mm=own: keep the object in a variable first");
                    signalPassFailure();
                }
            }
            else if (auto weakRefNewOp = mlir::dyn_cast<mlir_ts::WeakRefNewOp>(op);
                     weakRefNewOp && weakRefNewOp.getWeakField())
            {
                // a @weak field owns nothing (spec 93): a new object written to it, which nothing else
                // holds, would be freed at once and leave the field pointing at a tombstone
                std::function<bool(mlir::Value, unsigned)> newUnheld = [&](mlir::Value value, unsigned depth) -> bool {
                    // `= null`: casts of a null or undefined constant
                    auto *source = value.getDefiningOp();
                    while (source && mlir::isa<mlir_ts::CastOp, mlir_ts::ValueOrDefaultOp, mlir_ts::ValueOp>(source) &&
                           source->getNumOperands() > 0)
                    {
                        source = source->getOperand(0).getDefiningOp();
                    }

                    auto target = rootOf(value);
                    // `cond ? new Node() : p`: what either arm gives, merged into a block argument
                    if (auto argument = mlir::dyn_cast<mlir::BlockArgument>(target))
                    {
                        auto *block = argument.getOwner();
                        if (block->isEntryBlock() || depth > 8)
                        {
                            return false;
                        }

                        for (auto predecessor = block->pred_begin(); predecessor != block->pred_end(); ++predecessor)
                        {
                            auto branchOp = mlir::dyn_cast<mlir::BranchOpInterface>((*predecessor)->getTerminator());
                            if (!branchOp)
                            {
                                continue;
                            }

                            auto operands = branchOp.getSuccessorOperands(predecessor.getSuccessorIndex());
                            if (argument.getArgNumber() < operands.getProducedOperandCount())
                            {
                                continue;
                            }

                            if (auto incoming = operands[argument.getArgNumber()]; incoming && newUnheld(incoming, depth + 1))
                            {
                                return true;
                            }
                        }

                        return false;
                    }

                    auto *def = target.getDefiningOp();
                    if (!def)
                    {
                        return false;
                    }

                    // the same, before the conditional is flattened into blocks
                    if (auto ifOp = mlir::dyn_cast<mlir_ts::IfOp>(def))
                    {
                        auto index = mlir::cast<mlir::OpResult>(target).getResultNumber();
                        for (auto &region : ifOp->getRegions())
                        {
                            for (auto &block : region)
                            {
                                auto resultOp = mlir::dyn_cast_or_null<mlir_ts::ResultOp>(block.getTerminator());
                                if (resultOp && index < resultOp->getNumOperands() &&
                                    newUnheld(resultOp->getOperand(index), depth + 1))
                                {
                                    return true;
                                }
                            }
                        }

                        return false;
                    }

                    auto isNull = mlir::isa_and_nonnull<mlir_ts::NullOp, mlir_ts::UndefOp, mlir_ts::OptionalUndefOp>(source);
                    // a read of a variable, a parameter or a field has an owner: that place; so does a value a
                    // `const` holds (declare() names its defining op)
                    auto isRead = mlir::isa_and_nonnull<mlir_ts::LoadOp>(source) || nameAt(def->getLoc()) != "this value";
                    return !isNull && !isRead && ownsHeap(target) && isFresh(target) &&
                           !def->hasAttr(OWNED_RESULT_NAMED_ATTR_NAME) &&
                           llvm::none_of(def->getUsers(), [](mlir::Operation *user) { return mlir::isa<mlir_ts::VariableOp>(user); });
                };

                if (newUnheld(weakRefNewOp.getTarget(), 0) && reporting())
                {
                    op->emitError("the @" WEAK_DECORATOR " field '")
                        << weakFieldShortName(weakRefNewOp.getWeakField().value_or(""))
                        << "' cannot own a new object; it would be freed at once";
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
                // a fill of an empty field frees nothing (TSLANG 47, the signature pass's verdict)
                if (!releasesNothing(releaseSlotOp) && !releaseSlotOp->hasAttr(OWN_FILL_ATTR_NAME) &&
                    (isPlace(slot) || isCellVariable(slot) || isLoadedCell(slot) || slot.getDefiningOp<mlir_ts::AddressOfOp>()))
                {
                    drops.push_back(op);
                }
            }
            else if (mlir::isa<mlir_ts::ArrayPushOp, mlir_ts::ArrayUnshiftOp>(op) || growsOnly(op))
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
            else if (mlir::isa<mlir_ts::WeakRefDerefOp>(op))
            {
                // the object a WeakRef refers to: a borrow of a place nobody here knows, which anything
                // that may free an object of its type ends (spec 84)
                ops.placeReads.push_back(op);
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

            SubjectScope subject(*this, value);
            if (checkValueMoves(value, toErase) && isFresh(value))
            {
                proven.insert(value);
            }
        }

        for (auto read : ops.placeReads)
        {
            // a counted read keeps rc's ops: nothing to decide
            if (!dryRun && read->getNumResults() == 1 && countedValues.contains(countedUnitOf(read->getResult(0))))
            {
                continue;
            }

            SubjectScope subject(*this, read->getNumResults() == 1 ? read->getResult(0) : mlir::Value());
            checkPlaceRead(read, toErase);
        }

        for (auto read : ops.globalReads)
        {
            SubjectScope subject(*this, read.getResult());
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
            SubjectScope subject(*this, countedUnitOfOp(op));
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

            // a `let` declared from a merge of owning locals (TSLANG 58): it borrows each of them;
            // and one that only borrows, assigned a merge (`qq = r && q`)
            mlir::Operation *from = nullptr;
            if (auto argument = value ? mlir::dyn_cast<mlir::BlockArgument>(rootOf(value)) : mlir::BlockArgument();
                argument && ((mlir::isa<mlir_ts::RetainSlotOp>(op) && borrowerOf(op)) || localBorrowerOf(op, from)))
            {
                llvm::SmallVector<mlir_ts::LoadOp> loads;
                if (slotLoadsOfMerge(argument, loads) && !loads.empty())
                {
                    mergedSlotRetains.insert(op);
                    llvm::SmallPtrSet<mlir::Value, 4> slots;
                    for (auto load : loads)
                    {
                        if (slots.insert(load.getReference()).second)
                        {
                            slotReceivers[load.getReference()].push_back({op, value, load});
                        }
                    }

                    continue;
                }
            }

            if (value && isReadOutOfException(value))
            {
                toErase.insert(op);
                continue;
            }

            if (value && holdsConstantsOrMoves(value))
            {
                // the reads of owning `let`s stored into its fields move out of them, at this retain
                for (auto load : recordSlotLoads(value))
                {
                    slotReceivers[load.getReference()].push_back({op, load.getResult(), load});
                }

                // ...and the reads of parameters this function keeps, as rc's reference for the record
                // (spec 95)
                for (auto load : recordSlotLoads(value, /*keptParams=*/true))
                {
                    paramReceivers[load.getReference()].push_back({op, load.getResult(), load});
                }

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

            // a merge of what nothing owns (`first ? shared : other` of immortal globals,
            // `c ? 1 : "one"`): it owns nothing either
            if (value && (mlir::isa<mlir_ts::RetainOp>(op) || slotHoldsInitializerAt(op)) && mergeOwnsNothing(value))
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
                    isImmortalLiteral(args[index]) || isHandle(args[index]) ||
                    (!dryRun && countedValues.contains(countedUnitOf(args[index]))))
                {
                    continue;
                }

                auto arg = args[index];
                SubjectScope subject(*this, arg);
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
                    // a string read out of a parameter this function only borrows: the callee takes a
                    // copy (spec 22), as it does of one read out of a place (callCopies). The adapter
                    // that calls a callback declaring fewer parameters (`(v) => names.push(v)`) passes
                    // its parameters on so, once the callbacks passed in its place disagree on them.
                    // The dry run still fails here: its failure is what makes the copies, and the real
                    // run then finds a fresh copy given
                    auto loadOp = arg.getDefiningOp<mlir_ts::LoadOp>();
                    auto varOp = loadOp ? loadOp.getReference().getDefiningOp<mlir_ts::VariableOp>() : mlir_ts::VariableOp();
                    auto position = -1;
                    if (dryRun && varOp && isParameterSlot(varOp, position) && copiesInto(call, arg))
                    {
                        callCopies.push_back({loadOp, call});
                    }

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
                               OWN_OVERRIDABLE_CALL_ATTR_NAME, OWN_OVERRIDABLE_VIA_ATTR_NAME, OWN_FILL_ATTR_NAME,
                               OWN_FRESH_RESULT_ATTR_NAME,
                               OWN_ASSIGNS_CAPTURES_ATTR_NAME,
                               OWN_RESULT_BOUNDED_ATTR_NAME, OWN_STATE_BORROW_ATTR_NAME,
                               OWN_CAPTURE_BORROW_ATTR_NAME, OWN_BORROWING_ARRAY_ATTR_NAME,
                               OWN_BORROWING_BOX_ATTR_NAME})
            {
                op->removeAttr(name);
            }
        });
    }

    bool OwnershipInferencePass::mergeOwnsNothing(mlir::Value value, unsigned depth)
    {
        auto argument = mlir::dyn_cast<mlir::BlockArgument>(rootOf(value));
        if (!argument || argument.getOwner()->isEntryBlock() || depth > 8)
        {
            return false;
        }

        auto *block = argument.getOwner();
        for (auto predecessor = block->pred_begin(); predecessor != block->pred_end(); ++predecessor)
        {
            auto branchOp = mlir::dyn_cast<mlir::BranchOpInterface>((*predecessor)->getTerminator());
            if (!branchOp)
            {
                return false;
            }

            auto operands = branchOp.getSuccessorOperands(predecessor.getSuccessorIndex());
            if (argument.getArgNumber() < operands.getProducedOperandCount())
            {
                return false;
            }

            auto incoming = operands[argument.getArgNumber()];
            if (!incoming)
            {
                return false;
            }

            auto root = rootOf(incoming);
            if (mlir::isa<mlir::BlockArgument>(root))
            {
                if (!mergeOwnsNothing(root, depth + 1))
                {
                    return false;
                }

                continue;
            }

            if (!(isConstantData(incoming) || isConstantData(root) || isImmortalGlobalRead(root) || isImmortalLiteral(root)))
            {
                return false;
            }
        }

        return true;
    }

    bool OwnershipInferencePass::reporting()
    {
        if (quiet)
        {
            return false;
        }

        // the report is about the value the running check is about (fallbackHandler takes it)
        armedSubject = reportSubject;
        if (dryRun)
        {
            ++failures;
            // with --own-fallback=count the dry run's report is emitted and recorded (dryRunCapture),
            // never shown: what it names may be counted
            return countsFallback;
        }

        return true;
    }

    bool OwnershipInferencePass::reportingUncounted()
    {
        auto result = reporting();
        armedSubject = {};
        return result;
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
            // a counted value keeps its retains (countRefusedValues)
            if (touchesCounted(retain))
            {
                continue;
            }

            copyForRetain(retain);
        }

        for (auto [read, call] : callCopies)
        {
            auto value = read->getResult(0);
            // a counted read is given a count of its own, as a handle is (--own-fallback=count), through
            // the view the call is given too; any other read that cannot be copied stays the dry run's
            // error
            auto counted = countedValues.contains(countedUnitOf(value)) &&
                           llvm::any_of(call->getOperands(), [&](mlir::Value operand) { return rootOf(operand) == value; });
            if (!counted && !llvm::is_contained(call->getOperands(), value))
            {
                continue;
            }

            if (countsFallback && !counted && !isHandle(value) && !isCopyableValueAt(call, value.getType()))
            {
                continue;
            }

            // the place owns the copy, or a count of a handle's of its own (spec 23.2)
            if (auto storeOp = mlir::dyn_cast<mlir_ts::StoreOp>(call))
            {
                mlir::OpBuilder builder(storeOp);
                if (isHandle(value) || counted)
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
                return static_cast<size_t>(index) < args.size() && (args[index] == value || (counted && rootOf(args[index]) == value));
            });

            mlir::OpBuilder builder(call);
            // a handle: a count of its own for the call, given back after it (spec 50); a counted value
            // the same
            if (isHandle(value) || counted)
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
            if (!init || !isCopyableValueAt(varOp, init.getType()) || isCellVariable(varOp.getResult()))
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

        if (!isCopyableValueAt(retain, value.getType()))
        {
            return false;
        }

        // the one use rc retained it for. Of several (`f(s); f(s)` boxing `s` into `any` twice), the
        // first after the retain in its block: rc retains each beside the use it takes it for.
        mlir::OpOperand *taking = nullptr;
        mlir::OpOperand *nearest = nullptr;
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
                    if (user->getBlock() == retain->getBlock() && retain->isBeforeInBlock(user) &&
                        (!nearest || user->isBeforeInBlock(nearest->getOwner())))
                    {
                        nearest = &operand;
                    }
                }
            }
        });

        if (several)
        {
            taking = nearest;
            several = false;
        }

        if (!taking || several || !isCopyableValueAt(retain, taking->get().getType()))
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

            if (!isCopyableValueAt(retainOp, field.getType()) || borrowsKnownParam(field))
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

    mlir::Value OwnershipInferencePass::countedUnitOf(mlir::Value value)
    {
        if (!value)
        {
            return {};
        }

        auto root = rootOf(value);
        if (auto loadOp = root.getDefiningOp<mlir_ts::LoadOp>())
        {
            if (auto varOp = loadOp.getReference().getDefiningOp<mlir_ts::VariableOp>())
            {
                return varOp.getResult();
            }
        }

        return root;
    }

    mlir::Value OwnershipInferencePass::countedUnitOfOp(mlir::Operation *op)
    {
        if (mlir::isa<mlir_ts::RetainOp, mlir_ts::ReleaseOp>(op))
        {
            return countedUnitOf(op->getOperand(0));
        }

        if (auto retainSlotOp = mlir::dyn_cast<mlir_ts::RetainSlotOp>(op))
        {
            return retainSlotOp.getSlot().getDefiningOp<mlir_ts::VariableOp>() ? retainSlotOp.getSlot() : mlir::Value();
        }

        if (auto releaseSlotOp = mlir::dyn_cast<mlir_ts::ReleaseSlotOp>(op))
        {
            return releaseSlotOp.getSlot().getDefiningOp<mlir_ts::VariableOp>() ? releaseSlotOp.getSlot() : mlir::Value();
        }

        return {};
    }

    bool OwnershipInferencePass::touchesCounted(mlir::Operation *op)
    {
        if (countedValues.empty())
        {
            return false;
        }

        auto unit = countedUnitOfOp(op);
        return unit && countedValues.contains(unit);
    }

    bool OwnershipInferencePass::countable(mlir::Value unit)
    {
        auto *def = unit ? unit.getDefiningOp() : nullptr;
        // a result that borrows a parameter, or a record of borrows: the caller gives back nothing rc's
        // return retained for it
        if (!def || returnsBorrowOf >= 0 || !returnsRecordOf.empty())
        {
            return false;
        }

        // A root born owning one share under rc and under own: a `new` of a class (rc writes its count
        // as one, LowerToLLVM's NewOpLowering) and a fresh call's result (rc's callee retains it for the
        // return). A read of a place: no share under either. Anything else - a block rc makes at zero
        // whose first owner's retain own erases (`[...]`, `"a" + i`) - stays an error.
        auto bornRoot = [&](mlir::Value root) {
            auto *rootDef = root.getDefiningOp();
            if (!rootDef || isHandle(root) || !ownsHeap(root))
            {
                return false;
            }

            if (auto newOp = mlir::dyn_cast<mlir_ts::NewOp>(rootDef))
            {
                return newOp->hasAttr(OWNED_RESULT_ATTR_NAME) && !newOp.getStackAlloc().value_or(false);
            }

            if (isCall(rootDef))
            {
                return isFresh(root);
            }

            return mlir::isa<mlir_ts::LoadOp>(rootDef) && placeReadOf(root) == rootDef;
        };

        // A read of a place kept as a borrow past its retains. rc reads a place where it takes it, but
        // the signature pass (TSLANG 47) reads a field in place of the local just moved into it, and
        // that read stands for the local's value further on: it is counted with the value stored there
        // (fieldReadPartner), whose own release holds the block to the end of its scope, as rc's does.
        // Any other read only where nothing may free what it reads before its last use. A store's
        // release of the old value right after a retain of the read is that store's, as under rc.
        auto readCountable = [&](mlir::Value root) {
            // (the partner is in the read's group: countedGroup)
            if (fieldReadPartner(root))
            {
                return true;
            }

            auto *read = root.getDefiningOp();
            mlir::Operation *last = read;
            auto sameBlock = true;
            forEachUse(root, [&](mlir::Operation *user, mlir::Value) {
                if (user->getBlock() != read->getBlock())
                {
                    sameBlock = false;
                    return;
                }

                last = last->isBeforeInBlock(user) ? user : last;
            });

            if (!sameBlock)
            {
                return false;
            }

            llvm::SmallPtrSet<mlir::Operation *, 16> dropping(drops.begin(), drops.end());
            // (a call it is lent to is given a count of its own for the call: callCopies)
            llvm::SmallPtrSet<mlir::Operation *, 4> lentTo;
            for (auto [lent, call] : callCopies)
            {
                if (lent == read)
                {
                    lentTo.insert(call);
                }
            }

            for (auto *op = read->getNextNode(); op && op != last->getNextNode(); op = op->getNextNode())
            {
                auto *previous = op->getPrevNode();
                auto retainedBefore = (mlir::isa<mlir_ts::ReleaseSlotOp>(op) && previous &&
                                       mlir::isa<mlir_ts::RetainOp>(previous) && rootOf(previous->getOperand(0)) == root) ||
                                      (lentTo.contains(op) && op == last);
                if ((dropping.contains(op) || isCall(op)) && !retainedBefore)
                {
                    return false;
                }
            }

            return true;
        };

        if (auto varOp = mlir::dyn_cast<mlir_ts::VariableOp>(def))
        {
            auto type = mlir::cast<mlir_ts::RefType>(varOp.getType()).getElementType();
            if (varOp.getCaptured().value_or(false) || isCellVariable(varOp.getResult()) || isHandle(varOp.getResult()) ||
                !ownsHeap(def->getLoc(), type) || (isCopyableValueAt(def, type) && !isCopyableString(type)))
            {
                return false;
            }

            // a parameter: borrowed, or kept and released at the exits (releaseKeptParamsAtExits)
            auto index = -1;
            if (isParameterSlot(varOp, index))
            {
                return true;
            }

            // an owning local declared with a value rc counts as own does: rc's retain of the slot, or
            // the value it consumes, is the local's share (what it is assigned is in its group:
            // countedGroup). Not one declared from a literal, which rc makes at zero (a constant
            // array's copy) and own's first owner holds: its retain would be one too many.
            auto init = varOp.getInitializer();
            if (!isOwningVariable(varOp) || !init)
            {
                return false;
            }

            auto initUnit = countedUnitOf(init);
            if (auto initVar = initUnit.getDefiningOp<mlir_ts::VariableOp>())
            {
                auto initIndex = -1;
                return isParameterSlot(initVar, initIndex) || isOwningVariable(initVar);
            }

            return bornRoot(initUnit);
        }

        // (a string copyForRetain refused counts as an object does; `any` and `@copies` arrays stay copies)
        if ((isCopyableValueAt(def, unit.getType()) && !isCopyableString(unit.getType())) || !bornRoot(unit))
        {
            return false;
        }

        return !mlir::isa<mlir_ts::LoadOp>(def) || readCountable(unit);
    }

    mlir::Value OwnershipInferencePass::fieldReadPartner(mlir::Value read)
    {
        // `h.node = n; ...n...` as the signature pass leaves it: the store of `n`, then right after it
        // the read of the same field that later uses of `n` read
        auto loadOp = read.getDefiningOp<mlir_ts::LoadOp>();
        auto refOp = loadOp ? loadOp.getReference().getDefiningOp<mlir_ts::PropertyRefOp>() : mlir_ts::PropertyRefOp();
        if (!refOp)
        {
            return {};
        }

        // the same object: one value, or reads of one local (`h.node` twice reads `h` twice)
        auto sameObject = [](mlir::Value first, mlir::Value second) {
            auto firstLoad = first.getDefiningOp<mlir_ts::LoadOp>();
            auto secondLoad = second.getDefiningOp<mlir_ts::LoadOp>();
            return first == second || (firstLoad && secondLoad && firstLoad.getReference() == secondLoad.getReference() &&
                                       firstLoad.getReference().getDefiningOp<mlir_ts::VariableOp>());
        };

        for (auto *op = refOp->getPrevNode(); op; op = op->getPrevNode())
        {
            auto variableRead = mlir::isa<mlir_ts::LoadOp>(op) && op->getOperand(0).getDefiningOp<mlir_ts::VariableOp>();
            if (mlir::isa<mlir_ts::PropertyRefOp, mlir_ts::CastOp, mlir_ts::ConstantOp>(op) || variableRead)
            {
                continue;
            }

            auto storeOp = mlir::dyn_cast<mlir_ts::StoreOp>(op);
            auto storedRef = storeOp ? storeOp.getReference().getDefiningOp<mlir_ts::PropertyRefOp>() : mlir_ts::PropertyRefOp();
            if (!storedRef || !sameObject(storedRef.getObjectRef(), refOp.getObjectRef()) ||
                storedRef.getPosition() != refOp.getPosition())
            {
                return {};
            }

            // only a value own takes its place for: one the analysis can count (a slot's read is the slot)
            auto stored = countedUnitOf(storeOp.getValue());
            return stored.getDefiningOp<mlir_ts::LoadOp>() ? mlir::Value() : stored;
        }

        return {};
    }

    void OwnershipInferencePass::countRefusedValues(const llvm::SetVector<mlir::Operation *> &erased)
    {
        // the value each report is about (its subject, SubjectScope), and its first report
        llvm::DenseMap<mlir::Value, const RecordedDiagnostic *> reportOf;
        for (auto &[subject, recorded] : dryRunCapture)
        {
            if (auto unit = countedUnitOf(subject))
            {
                reportOf.try_emplace(unit, &recorded);
            }
        }

        if (reportOf.empty())
        {
            return;
        }

        // The `let`s the repair gives another shape, which plain own accepts in that shape: its fresh
        // values into a holder (splitBorrowingLets), or each value it is given a borrow of its owner
        // (classifyLocalBorrowingLets, as the real run will, asked now with the dry run's facts). A report
        // the repair resolves is never counted, so the flag decides such a `let` as plain own does.
        // Counted and reshaped too, its rc ops would no longer be whole (the holder takes the releases
        // of its assignments, a borrower's retains go) and its block would be freed while the place it
        // was read from still holds it. Where the guess misses, the two repairs leave a counted `let`
        // alone, and one that is reshaped but still refused is reported again, about a value nobody
        // counted.
        llvm::SmallPtrSet<mlir::Operation *, 4> repairedLets;
        for (auto *candidate : splitLetCandidates)
        {
            llvm::SmallVector<mlir_ts::StoreOp> fresh;
            llvm::SmallVector<mlir::Operation *> exits;
            if (splitsLet(mlir::cast<mlir_ts::VariableOp>(candidate), fresh, exits))
            {
                repairedLets.insert(candidate);
            }
        }

        {
            // (analyze's dominance is gone with its call: this one is asked for here)
            mlir::DominanceInfo dominanceInfo(getFunction());
            dominance = &dominanceInfo;
            auto dominanceGone = llvm::make_scope_exit([&]() { dominance = nullptr; });
            classifyLocalBorrowingLets();
            repairedLets.insert(localBorrowingLets.begin(), localBorrowingLets.end());
            localBorrowingLets.clear();
        }

        // The retains the dry run could not erase (the set makeStringCopies walks), the reads it lent
        // to a call that may free them (callCopies), and a value reported that a call keeps (rc gives
        // a keeping call no retain: retainKeptCountedArguments makes one for each). A string gets its
        // copy first, as today; one copyForRetain refuses is refused.
        llvm::SetVector<mlir::Value> refused;
        llvm::SmallVector<mlir::Operation *> retains;
        getFunction()->walk([&](mlir::Operation *op) {
            if (mlir::isa<mlir_ts::RetainOp, mlir_ts::RetainSlotOp>(op) && !erased.contains(op) &&
                !captureRetains.contains(op) && !fieldBorrowRetains.contains(op) && !touchesHandle(op))
            {
                retains.push_back(op);
            }
        });

        for (auto *retain : retains)
        {
            auto unit = countedUnitOfOp(retain);
            if (!unit)
            {
                continue;
            }

            auto type = unit.getType();
            if (auto refType = mlir::dyn_cast<mlir_ts::RefType>(type); refType && unit.getDefiningOp<mlir_ts::VariableOp>())
            {
                type = refType.getElementType();
            }

            if (isCopyableValueAt(retain, type) && copyForRetain(retain))
            {
                continue;
            }

            refused.insert(unit);
        }

        for (auto [read, call] : callCopies)
        {
            if (!isCopyableValueAt(call, read->getResult(0).getType()) && !isHandle(read->getResult(0)))
            {
                refused.insert(countedUnitOf(read->getResult(0)));
            }
        }

        getFunction()->walk([&](mlir::Operation *op) {
            if (!isCall(op))
            {
                return;
            }

            auto args = callArgs(op);
            for (auto index : ownedParams(op))
            {
                if (static_cast<size_t>(index) >= args.size() || !reportOf.count(countedUnitOf(args[index])))
                {
                    continue;
                }

                // (a value the repair copies for that call - a string, `any`: callCopies - is given its
                // copy as without the flag, never a count)
                auto copied = llvm::any_of(callCopies, [&](auto &copy) {
                    return copy.second == op && rootOf(copy.first->getResult(0)) == rootOf(args[index]) &&
                           isCopyableValueAt(op, copy.first->getResult(0).getType());
                });
                if (!copied)
                {
                    refused.insert(countedUnitOf(args[index]));
                }
            }
        });

        for (auto unit : refused)
        {
            if (!unit || countedValues.contains(unit))
            {
                continue;
            }

            // counted with its group, or not at all: when a report is about one of the group;
            // warned once, with that report
            llvm::SetVector<mlir::Value> group;
            if (!countedGroup(unit, group))
            {
                continue;
            }

            // a `let` the repair reshapes is what plain own repairs: never counted (repairedLets)
            auto repaired = llvm::any_of(group, [&](mlir::Value member) {
                auto *def = member.getDefiningOp();
                return def && repairedLets.contains(def);
            });
            if (repaired)
            {
                continue;
            }

            const RecordedDiagnostic *report = nullptr;
            for (auto member : group)
            {
                if (auto found = reportOf.find(member); found != reportOf.end())
                {
                    report = found->second;
                    break;
                }
            }

            if (!report)
            {
                continue;
            }

            // A store of a member whose release of the place's old value (rc's, right after the
            // member's retain) the dry run erased as part of another value's move or borrow (`p.v =
            // p.v; p.v = p.v`: the first read's retain is erased with the second store's release).
            // Counted, the member keeps its retain but that release stays erased, and the release of
            // the first store - which rc's erased retain held up - frees the block the field still
            // holds. Not counted: the error stays.
            auto claimed = false;
            getFunction()->walk([&](mlir::Operation *op) {
                if (claimed || !mlir::isa<mlir_ts::RetainOp, mlir_ts::RetainSlotOp>(op) || erased.contains(op))
                {
                    return;
                }

                // rc's store into a place: the retain of the value, the release of the place's old
                // value, the store
                auto unit = countedUnitOfOp(op);
                auto release = mlir::dyn_cast_or_null<mlir_ts::ReleaseSlotOp>(op->getNextNode());
                auto storeOp = release ? mlir::dyn_cast_or_null<mlir_ts::StoreOp>(release->getNextNode()) : mlir_ts::StoreOp();
                claimed = unit && group.contains(unit) && storeOp && storeOp.getReference() == release.getSlot() &&
                          !release.getSlot().getDefiningOp<mlir_ts::VariableOp>() && countedUnitOf(storeOp.getValue()) == unit &&
                          erased.contains(release);
            });
            if (claimed)
            {
                continue;
            }

            countedValues.insert(group.begin(), group.end());
            countedOrder.push_back(unit);
            dryRunDiagnostics.try_emplace(unit, *report);
        }
    }

    bool OwnershipInferencePass::countedGroup(mlir::Value unit, llvm::SetVector<mlir::Value> &group)
    {
        // The values one block's rc ops pass between in this function: a local and each value it is
        // declared from or assigned, a value and each local declared from it, a field read the
        // signature pass made in place of a local's value and that value. They keep rc's ops together
        // or not at all: a local counted alone keeps a release of a value own let it borrow, a value
        // counted alone may be moved into a local own then erases the release of.
        auto add = [&](mlir::Value value) {
            // (a literal is not skipped: it is no root rc and own count alike, so its group is refused)
            if (value && !holdsNoBlock(value))
            {
                group.insert(countedUnitOf(value));
            }
        };

        group.insert(unit);
        for (size_t i = 0; i < group.size(); ++i)
        {
            auto member = group[i];
            if (!countable(member))
            {
                return false;
            }

            // a tagged union's payload, a closure: the counted retain under own has nothing to go
            // through (OwnershipRoutineLogic::canCountRetainOf), so it stays an error here, at its place
            auto memberType = member.getType();
            if (auto refType = mlir::dyn_cast<mlir_ts::RefType>(memberType); refType && member.getDefiningOp<mlir_ts::VariableOp>())
            {
                memberType = refType.getElementType();
            }

            // ...nor an object that may have a field of one: a counted object can reach any take of a
            // field out of it (a call's result, through any function), and a take cannot give such a
            // field's value an owner. So no counted object has one, and every take may empty it.
            if (reachedThroughDescriptor(memberType) || mayHoldDescriptorField(memberType))
            {
                return false;
            }

            if (auto varOp = member.getDefiningOp<mlir_ts::VariableOp>())
            {
                auto index = -1;
                if (!isParameterSlot(varOp, index))
                {
                    add(varOp.getInitializer());
                }

                for (auto *user : member.getUsers())
                {
                    if (auto storeOp = mlir::dyn_cast<mlir_ts::StoreOp>(user); storeOp && storeOp.getReference() == member)
                    {
                        add(storeOp.getValue());
                    }

                    // ...and each local declared from or given a read of it (`const n16 = n15`): rc
                    // retains for that local and releases it, and own may erase the one and keep the
                    // other as a move out of the counted local, which it no longer is
                    auto loadOp = mlir::dyn_cast<mlir_ts::LoadOp>(user);
                    if (!loadOp || loadOp.getReference() != member)
                    {
                        continue;
                    }

                    forEachUse(loadOp.getResult(), [&](mlir::Operation *readUser, mlir::Value used) {
                        auto varOp = mlir::dyn_cast<mlir_ts::VariableOp>(readUser);
                        if (varOp && varOp.getInitializer() == used && isOwningVariable(varOp))
                        {
                            group.insert(varOp.getResult());
                        }

                        auto storeOp = mlir::dyn_cast<mlir_ts::StoreOp>(readUser);
                        auto slot = storeOp && storeOp.getValue() == used ? storeOp.getReference().getDefiningOp<mlir_ts::VariableOp>()
                                                                         : mlir_ts::VariableOp();
                        if (slot && slot.getResult() != member && isOwningVariable(slot))
                        {
                            group.insert(slot.getResult());
                        }
                    });
                }

                continue;
            }

            if (auto partner = fieldReadPartner(member))
            {
                group.insert(partner);
            }

            forEachUse(member, [&](mlir::Operation *user, mlir::Value used) {
                auto varOp = mlir::dyn_cast<mlir_ts::VariableOp>(user);
                if (varOp && varOp.getInitializer() == used && isOwningVariable(varOp))
                {
                    group.insert(varOp.getResult());
                }

                auto storeOp = mlir::dyn_cast<mlir_ts::StoreOp>(user);
                auto slot = storeOp && storeOp.getValue() == used ? storeOp.getReference().getDefiningOp<mlir_ts::VariableOp>()
                                                                 : mlir_ts::VariableOp();
                if (slot && isOwningVariable(slot))
                {
                    group.insert(slot.getResult());
                }
            });
        }

        // A value something borrows - an array literal's or a rest argument's element, a capture of a
        // closure or a literal's box, a value boxed in an `any` and cast back, a store into a borrowing
        // field, a `let` borrowing a merge (classifyClosures, classifyLiteralCaptures,
        // classifyStateBoxes, classifyBorrowingFields, classifyArgumentArrays,
        // classifyLocalBorrowingArrays, classifyBorrowingBoxes, borrowsThroughMerge): rc's retain for
        // the borrower goes, and the borrower's release is shallow or none. Counted, the retain would be
        // kept (keepCountedOps) and nothing would give it back - a leak. Its ops are not whole, so it
        // is not counted: it stays an error. (The view's report, classifyArrayViews, has no subject.)
        auto borrowed = [&](const llvm::DenseSet<mlir::Operation *> &retains) {
            return llvm::any_of(retains, [&](mlir::Operation *retain) {
                auto unit = countedUnitOfOp(retain);
                return unit && group.contains(unit);
            });
        };

        if (borrowed(fieldBorrowRetains) || borrowed(captureRetains) || borrowed(mergeBorrowRetains))
        {
            return false;
        }

        // A value put into a field of a record held by value (`{ v: n, items: [n] }` built in a local
        // or a box, a tuple): rc takes that field's count with a retain of the whole record when it is
        // boxed or copied, not with a retain of the value, and own erases that retain while the
        // record's release still goes through its fields. The member's ops are not whole, so it is
        // not counted: it stays an error.
        auto intoRecord = [&](mlir::Value value) {
            auto found = false;
            forEachUse(value, [&](mlir::Operation *user, mlir::Value used) {
                if (mlir::isa<mlir_ts::CreateTupleOp, mlir_ts::InsertPropertyOp>(user))
                {
                    found = true;
                    return;
                }

                auto storeOp = mlir::dyn_cast<mlir_ts::StoreOp>(user);
                auto refOp = storeOp && storeOp.getValue() == used ? storeOp.getReference().getDefiningOp<mlir_ts::PropertyRefOp>()
                                                                   : mlir_ts::PropertyRefOp();
                if (!refOp || !mlir::isa<mlir_ts::RefType, mlir_ts::ValueRefType>(refOp.getObjectRef().getType()))
                {
                    return;
                }

                // (a closure's capture box is not a record: what it captures is counted through the
                // capture's own ops, captureRetains)
                if (auto *boxDef = refOp.getObjectRef().getDefiningOp(); boxDef && boxDef->hasAttr(CAPTURE_BOX_ATTR_NAME))
                {
                    return;
                }

                // (a store with a retain of its own, as an assignment to the field after the record
                // is made has - `a.item = x` - takes its count as any field store does)
                for (auto *prev = storeOp->getPrevNode(); prev; prev = prev->getPrevNode())
                {
                    if (mlir::isa<mlir_ts::ReleaseSlotOp, mlir_ts::ReleaseOp>(prev))
                    {
                        continue;
                    }

                    if (mlir::isa<mlir_ts::RetainOp>(prev) && rootOf(prev->getOperand(0)) == rootOf(used))
                    {
                        return;
                    }

                    break;
                }

                found = true;
            });
            return found;
        };

        for (auto member : group)
        {
            if (!member.getDefiningOp<mlir_ts::VariableOp>())
            {
                if (intoRecord(member))
                {
                    return false;
                }

                continue;
            }

            for (auto *user : member.getUsers())
            {
                if (auto loadOp = mlir::dyn_cast<mlir_ts::LoadOp>(user); loadOp && intoRecord(loadOp.getResult()))
                {
                    return false;
                }
            }
        }

        // ...and each retain that would be kept (markCountedRetains), or made for a parameter that
        // keeps the value (retainKeptCountedArguments), is of the type the retain has, not the member's:
        // a tuple made a tagged union's payload on its way into a `T | null` field is retained as the
        // union, which under own has nothing to go through either. Refused here, it stays own's error at
        // its place, not a lowering failure.
        auto uncountable = [&](mlir::Type type) {
            if (auto refType = mlir::dyn_cast<mlir_ts::RefType>(type))
            {
                type = refType.getElementType();
            }

            return reachedThroughDescriptor(type) || mayHoldDescriptorField(type);
        };

        auto refusedRetain = false;
        getFunction()->walk([&](mlir::Operation *op) {
            if (refusedRetain)
            {
                return;
            }

            if (mlir::isa<mlir_ts::RetainOp, mlir_ts::RetainSlotOp>(op))
            {
                auto unit = countedUnitOfOp(op);
                refusedRetain = unit && group.contains(unit) && uncountable(op->getOperand(0).getType());
                return;
            }

            if (!isCall(op))
            {
                return;
            }

            auto args = callArgs(op);
            for (auto index : ownedParams(op))
            {
                if (static_cast<size_t>(index) < args.size() && group.contains(countedUnitOf(args[index])) &&
                    uncountable(args[index].getType()))
                {
                    refusedRetain = true;
                    return;
                }
            }
        });

        return !refusedRetain;
    }

    void OwnershipInferencePass::retainKeptCountedArguments()
    {
        getFunction()->walk([&](mlir::Operation *op) {
            if (!isCall(op))
            {
                return;
            }

            // one count for each kept parameter a counted value is given to, less the retains made
            // right before the call already (callCopies: of the read the argument is a view of)
            auto args = callArgs(op);
            llvm::MapVector<mlir::Value, llvm::SmallVector<mlir::Value>> kept;
            for (auto index : ownedParams(op))
            {
                if (static_cast<size_t>(index) < args.size() && countedValues.contains(countedUnitOf(args[index])))
                {
                    kept[rootOf(args[index])].push_back(args[index]);
                }
            }

            for (auto &[root, given] : kept)
            {
                size_t retained = 0;
                for (auto *prev = op->getPrevNode(); prev && mlir::isa<mlir_ts::RetainOp>(prev); prev = prev->getPrevNode())
                {
                    retained += rootOf(prev->getOperand(0)) == root ? 1 : 0;
                }

                mlir::OpBuilder builder(op);
                for (auto i = retained; i < given.size(); ++i)
                {
                    builder.create<mlir_ts::RetainOp>(op->getLoc(), given[i]);
                }
            }
        });
    }

    void OwnershipInferencePass::keepCountedOps(llvm::SetVector<mlir::Operation *> &toErase)
    {
        if (countedValues.empty())
        {
            return;
        }

        auto counted = [&](mlir::Value slot) {
            return countedValues.contains(slot.getDefiningOp<mlir_ts::VariableOp>() ? slot : countedUnitOf(slot));
        };

        // rc's releases stay where it placed them: none on the edges a move did not take, and no slot
        // emptied after a move
        llvm::erase_if(edgeReleases, [&](EdgeRelease &edge) { return counted(edge.slot); });
        llvm::erase_if(slotClears, [&](SlotClear &clear) { return counted(clear.slot); });
        // ...nor a release of the value a select of a counted one did not take: rc's releases stay
        selectMoves.remove_if([&](mlir::Operation *op) {
            auto selectOp = mlir::cast<mlir::arith::SelectOp>(op);
            return counted(selectOp.getTrueValue()) || counted(selectOp.getFalseValue());
        });
        toErase.remove_if([&](mlir::Operation *op) { return touchesCounted(op); });
    }

    void OwnershipInferencePass::markCountedRetains()
    {
        if (countedValues.empty())
        {
            return;
        }

        auto unit = mlir::UnitAttr::get(&getContext());
        getFunction()->walk([&](mlir::Operation *op) {
            if (mlir::isa<mlir_ts::RetainOp, mlir_ts::RetainSlotOp>(op) && touchesCounted(op))
            {
                op->setAttr(OWN_COUNTED_ATTR_NAME, unit);
            }
        });
    }

    void OwnershipInferencePass::warnCounted()
    {
        for (auto unit : countedOrder)
        {
            // the dry run's error, then each the real run raised there in its place (not one already said)
            llvm::SmallVector<const RecordedDiagnostic *> recorded;
            if (auto found = dryRunDiagnostics.find(unit); found != dryRunDiagnostics.end())
            {
                recorded.push_back(&found->second);
            }

            for (auto &[countedUnit, error] : countedErrors)
            {
                auto said = llvm::any_of(recorded, [&](const RecordedDiagnostic *other) {
                    return other->loc == error.loc && other->message == error.message;
                });
                if (countedUnit == unit && !said)
                {
                    recorded.push_back(&error);
                }
            }

            auto loc = recorded.empty() ? unit.getLoc() : recorded.front()->loc;
            auto diag = mlir::emitWarning(loc, "-mm=own counts this object here (--own-fallback=count); a cycle through "
                                               "it is never freed");
            for (auto *error : recorded)
            {
                diag.attachNote(error->loc) << error->message;
                for (auto &[noteLoc, message] : error->notes)
                {
                    diag.attachNote(noteLoc) << message;
                }
            }
        }
    }

} // namespace own_inference

#undef DEBUG_TYPE

std::unique_ptr<mlir::Pass> mlir_ts::createOwnershipInferencePass(bool countsFallback)
{
    return std::make_unique<own_inference::OwnershipInferencePass>(countsFallback);
}
