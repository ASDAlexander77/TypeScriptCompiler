// OwnershipSignaturePass (see OwnershipSignaturePass.h): the pass itself, the facts a signature says,
// the facts callers must know and their fixpoint, drops, and the facts a library exports.

#include "OwnershipSignaturePass.h"

#include "llvm/ADT/ScopeExit.h"

#define DEBUG_TYPE "own"

namespace own_signature
{

    // Do a call's arguments stand where its callees' parameters do (calleesFieldDrops' owners)? Not through a bound
    // function or a closure value outside a callback family, whose `this` or capture box the call does not pass.
    static bool argumentsAlignWithParams(mlir::Operation *callOp, bool viaCallback)
    {
        return viaCallback || !mlir::isa<mlir_ts::CallHybridInternalOp, mlir_ts::InvokeHybridOp>(callOp);
    }

    // A function made for an element type -mm=own cannot give a second owner (`@sharesElements`, spec
    // 73): its body becomes a stop, so nothing analyses what it would do. It keeps its place in a
    // vtable; a call to it is inference's error.
    static void stubCannotShare(mlir::ModuleOp module)
    {
        llvm::SmallVector<mlir_ts::FuncOp> work;
        module.walk([&](mlir_ts::FuncOp funcOp) {
            if (funcOp->hasAttr(OWN_CANNOT_SHARE_ATTR_NAME) && !funcOp.getBody().empty())
            {
                work.push_back(funcOp);
            }
        });

        llvm::SmallPtrSet<mlir::Operation *, 8> stubbed;
        while (!work.empty())
        {
            auto funcOp = work.pop_back_val();
            if (!stubbed.insert(funcOp).second)
            {
                continue;
            }

            auto elementType = funcOp->getAttrOfType<mlir::StringAttr>(OWN_CANNOT_SHARE_ATTR_NAME);
            auto &body = funcOp.getBody();

            // what the body made for itself - a generator's `next`, a closure - goes with it once
            // nothing else refers to it
            llvm::SetVector<mlir_ts::FuncOp> made;
            // (a generator's state is a constant record that names its `next`)
            funcOp.walk([&](mlir::Operation *op) {
                op->getAttrDictionary().walk([&](mlir::FlatSymbolRefAttr symbol) {
                    if (auto callee = module.lookupSymbol<mlir_ts::FuncOp>(symbol); callee && callee != funcOp)
                    {
                        made.insert(callee);
                    }
                });
            });

            // where the body began: a debug location of this function's own scope (the function's location
            // may be the instantiation's, another subprogram's)
            auto location = body.front().empty() ? funcOp.getLoc() : body.front().front().getLoc();

            auto *stub = new mlir::Block();
            for (auto argument : body.front().getArguments())
            {
                stub->addArgument(argument.getType(), argument.getLoc());
            }

            body.dropAllReferences();
            while (!body.empty())
            {
                body.front().erase();
            }

            body.push_back(stub);

            // what the declaration says of its parameters described the body that is gone: nothing
            // calls the stub, so nothing needs it
            for (auto *name : {OWN_DECLARED_KEEPS_ATTR_NAME, OWN_DECLARED_BORROWS_ATTR_NAME, OWN_DECLARED_LENDS_ATTR_NAME,
                               OWN_DECLARED_RECORD_ATTR_NAME, OWN_DECLARED_RETURNS_THIS_ATTR_NAME,
                               OWN_IMPLIED_LENDS_ATTR_NAME, OWN_DECLARED_IMMORTALIZES_ATTR_NAME})
            {
                funcOp->removeAttr(name);
            }

            mlir::OpBuilder builder(stub, stub->begin());
            auto never = builder.create<mlir_ts::ConstantOp>(location, mlir_ts::BooleanType::get(builder.getContext()),
                                                             builder.getBoolAttr(false));
            std::string message = "-mm=own cannot give an element of type '" + elementType.getValue().str() +
                                  "' a second owner: '" + funcOp.getSymName().str() + "'";
            builder.create<mlir_ts::AssertOp>(location, never, mlir::Value(), builder.getStringAttr(message));
            llvm::SmallVector<mlir::Value> results;
            for (auto type : funcOp.getFunctionType().getResults())
            {
                results.push_back(builder.create<mlir_ts::DefaultOp>(location, type));
            }

            builder.create<mlir_ts::ReturnInternalOp>(location, results);

            for (auto made : made)
            {
                if (!made.getBody().empty() && mlir::SymbolTable::symbolKnownUseEmpty(made, module))
                {
                    made->setAttr(OWN_CANNOT_SHARE_ATTR_NAME, elementType);
                    work.push_back(made);
                }
            }
        }
    }

    void OwnershipSignaturePass::runOnOperation()
    {
        auto module = getOperation();
        stubCannotShare(module);
        borrowParametersInLocals(module);
        readFieldsForLocals(module);
        module.walk([&](mlir_ts::FuncOp funcOp) { functions[funcOp.getSymName()] = funcOp; });
        module.walk([&](mlir_ts::GlobalOp globalOp) { globals[globalOp.getSymName()] = globalOp; });
        // ...but a module `const` beside top-level code, which the module's code stores once where it is declared
        // (WRITTEN_ONCE_GLOBAL_ATTR_NAME), is written by no call, as a const its initializer sets: a function value it
        // holds is that function, and what it holds is not released while a function borrows it (TSLANG 191)
        llvm::StringMap<llvm::SmallVector<mlir_ts::StoreOp, 1>> storesOfWrittenOnce;
        module.walk([&](mlir_ts::AddressOfOp addressOfOp) {
            auto globalOp = globals.lookup(addressOfOp.getGlobalName());
            auto writtenOnce = globalOp && globalOp->hasAttr(WRITTEN_ONCE_GLOBAL_ATTR_NAME);
            auto onlyLoaded = true;
            for (auto *user : addressOfOp->getUsers())
            {
                if (mlir::isa<mlir_ts::LoadOp>(user))
                {
                    continue;
                }

                // the store, and the release of the default it replaces, in the root's code
                auto storeOp = mlir::dyn_cast<mlir_ts::StoreOp>(user);
                if (writtenOnce && storeOp && storeOp.getReference() == addressOfOp.getResult() &&
                    storeOp.getValue() != addressOfOp.getResult() && !mayRunAgain(storeOp))
                {
                    storesOfWrittenOnce[addressOfOp.getGlobalName()].push_back(storeOp);
                    continue;
                }

                if (writtenOnce && mlir::isa<mlir_ts::ReleaseSlotOp>(user))
                {
                    continue;
                }

                onlyLoaded = false;
            }

            if (!onlyLoaded)
            {
                writtenGlobals.insert(addressOfOp.getGlobalName());
            }
        });
        for (auto &stores : storesOfWrittenOnce)
        {
            if (stores.getValue().size() > 1)
            {
                writtenGlobals.insert(stores.getKey());
            }
            else if (!writtenGlobals.contains(stores.getKey()))
            {
                writtenOnceStores[stores.getKey()] = stores.getValue().front();
            }
        }
        if (auto names = module->getAttrOfType<mlir::ArrayAttr>(SHARED_LIB_OWN_NO_DROPS_ATTR_NAME))
        {
            for (auto name : names.getAsRange<mlir::StringAttr>())
            {
                importedNoDrops.insert(name.getValue());
            }
        }

        // the fields back-references are stored into (spec 70): what they hold borrows the object they
        // are in, so inference takes nothing out of them
        llvm::SetVector<mlir::Attribute> backReferenceFields;
        module.walk([&](mlir_ts::StoreOp storeOp) {
            auto propertyRefOp = storeOp.getReference().getDefiningOp<mlir_ts::PropertyRefOp>();
            auto storage = propertyRefOp ? fieldOwnerStorage(propertyRefOp) : mlir_ts::ClassStorageType();
            if (!storage || !isBackReferenceStore(storeOp))
            {
                return;
            }

            backReferenceFields.insert(mlir::ArrayAttr::get(
                &getContext(), {mlir::TypeAttr::get(storage),
                                mlir::IntegerAttr::get(mlir::IntegerType::get(&getContext(), 32), propertyRefOp.getPosition())}));
            backReferenceFieldSet.insert({storage, propertyRefOp.getPosition()});
        });

        // the fields of classes given a value not made where it is stored - a parameter's (`set(a) { this.names = a
        // }`), a read, a global's (`this.names = G` of an immortal `const`, which own allows, given straight or
        // through a parameter): what such a field holds may be another's too, so what is under it is not apart from
        // the rest because the object is (ownerParam, TSLANG 216). A field only ever given new values (`names:
        // string[] = []`) holds what its object alone owns.
        module.walk([&](mlir_ts::StoreOp storeOp) {
            auto propertyRefOp = storeOp.getReference().getDefiningOp<mlir_ts::PropertyRefOp>();
            auto storage = propertyRefOp ? fieldOwnerStorage(propertyRefOp) : mlir_ts::ClassStorageType();
            if (storage && !isFresh(storeOp.getValue()))
            {
                sharedFieldSet.insert({storage, propertyRefOp.getPosition()});
            }
        });
        if (!backReferenceFields.empty())
        {
            module->setAttr(OWN_BACK_REFERENCE_FIELDS_ATTR_NAME,
                            mlir::ArrayAttr::get(&getContext(), backReferenceFields.getArrayRef()));
        }

        // the private fields nothing writes but their class's constructor (TSLANG 73): `constructor(private
        // sample: NriSample)`. Only the class's own methods may write a private field, and they are all here
        // (its constructor is), so once the object is made no call - a callback included - overwrites it:
        // the field is released only with its object.
        llvm::DenseMap<std::pair<mlir::Type, int64_t>, bool> writtenAfterConstruction;
        module.walk([&](mlir_ts::PropertyRefOp propertyRefOp) {
            auto storage = fieldOwnerStorage(propertyRefOp);
            if (!storage)
            {
                return;
            }

            auto &written = writtenAfterConstruction[{storage, propertyRefOp.getPosition()}];
            auto funcOp = propertyRefOp->getParentOfType<mlir_ts::FuncOp>();
            auto inConstructor = funcOp && funcOp.getSymName() == (storage.getName().getAttr().getValue() + ".constructor").str();
            auto onlyRead = llvm::all_of(propertyRefOp->getUsers(), [](mlir::Operation *user) { return mlir::isa<mlir_ts::LoadOp>(user); });
            written = written || (!onlyRead && !inConstructor);
        });
        llvm::SetVector<mlir::Attribute> stableFields;
        for (auto &[key, written] : writtenAfterConstruction)
        {
            auto storage = mlir::cast<mlir_ts::ClassStorageType>(key.first);
            auto constructor = functions.lookup((storage.getName().getAttr().getValue() + ".constructor").str());
            auto fields = storage.getFields();
            if (written || !constructor || constructor.isExternal() || key.second < 0 ||
                static_cast<size_t>(key.second) >= fields.size() || fields[key.second].accessLevel != mlir_ts::AccessLevel::Private)
            {
                continue;
            }

            stableFields.insert(mlir::ArrayAttr::get(
                &getContext(), {mlir::TypeAttr::get(storage), mlir::IntegerAttr::get(mlir::IntegerType::get(&getContext(), 32), key.second)}));
        }

        // ...and the fields declared `readonly` (MLIRGen's markReadonlyFields): only their class's constructor
        // writes them, whatever their access level, so they are stable too. One of a class code outside this
        // module may extend counts only in a program, whose classes nothing else extends (spec 31).
        if (auto readonlyFields = module->getAttrOfType<mlir::ArrayAttr>(OWN_READONLY_FIELDS_ATTR_NAME))
        {
            for (auto entry : readonlyFields.getAsRange<mlir::ArrayAttr>())
            {
                if (mlir::cast<mlir::BoolAttr>(entry[2]).getValue() && !executable)
                {
                    continue;
                }

                stableFields.insert(mlir::ArrayAttr::get(
                    &getContext(), {entry[0], mlir::IntegerAttr::get(mlir::IntegerType::get(&getContext(), 32),
                                                                     mlir::cast<mlir::IntegerAttr>(entry[1]).getInt())}));
            }
        }

        if (!stableFields.empty())
        {
            // in the order of the class names and positions: writtenAfterConstruction is keyed by type pointer, and
            // its order changed from one compile to the next (nondeterministic IR)
            llvm::SmallVector<mlir::Attribute> sorted(stableFields.begin(), stableFields.end());
            auto keyOf = [](mlir::Attribute entry) {
                auto pair = mlir::cast<mlir::ArrayAttr>(entry);
                auto storage = mlir::cast<mlir_ts::ClassStorageType>(mlir::cast<mlir::TypeAttr>(pair[0]).getValue());
                return std::make_pair(storage.getName().getAttr().getValue(), mlir::cast<mlir::IntegerAttr>(pair[1]).getInt());
            };
            llvm::stable_sort(sorted, [&](mlir::Attribute a, mlir::Attribute b) { return keyOf(a) < keyOf(b); });
            module->setAttr(OWN_STABLE_FIELDS_ATTR_NAME, mlir::ArrayAttr::get(&getContext(), sorted));
        }

        if (auto functionsDropping = module->getAttrOfType<mlir::DictionaryAttr>(SHARED_LIB_OWN_DROP_FIELDS_ATTR_NAME))
        {
            for (auto entry : functionsDropping)
            {
                auto &drops = importedFieldDrops[entry.getName().getValue()];
                for (auto place : mlir::cast<mlir::ArrayAttr>(entry.getValue()).getAsRange<mlir::ArrayAttr>())
                {
                    drops.add({mlir::cast<mlir::IntegerAttr>(place[0]).getInt(), mlir::cast<mlir::TypeAttr>(place[1]).getValue()});
                }
            }
        }

        if (auto names = module->getAttrOfType<mlir::ArrayAttr>(SHARED_LIB_OWN_FRESH_RESULTS_ATTR_NAME))
        {
            for (auto name : names.getAsRange<mlir::StringAttr>())
            {
                importedFreshResults.insert(name.getValue());
            }
        }

        findImmortalGlobals(module);
        moveIntoGlobals(module);
        rejectBorrowedGlobalInitializers(module);
        // the storages first: a vtable's family is that of its class's root (familyRoot)
        collectClassStorages(module);
        collectClassVTables(module);
        collectInterfaceFamilies(module);
        fieldFunctions.collect(module);
        // after the class storages (`@cpp`) and the families (holdsFunction reads the interface vtables)
        findEscapingFunctions(module);
        // before the calls are resolved: a call through a callback parameter reaches its family
        collectCallbackFamilies(module);

        module.walk([&](mlir::Operation *op) {
            // in any function: a `ts.Func`, or the `func.func` the async lowering outlines a body into
            if (isCall(op) && op->getParentOfType<mlir::FunctionOpInterface>())
            {
                calls.push_back({op, resolve(op)});
            }
        });

        findOpen(module);
        // the `@lends("this")` a result type `this` implies, on a function only this module calls: its
        // callers go by its body, as for any other function here (spec 78). Not when the function says
        // what it keeps (`@keeps`): it then says what its callers give and get, and `this` is part of it.
        for (auto &entry : functions)
        {
            if (entry.second->hasAttr(OWN_IMPLIED_LENDS_ATTR_NAME) && !entry.second.isDeclaration() &&
                !open.contains(entry.second) && !entry.second->hasAttr(OWN_DECLARED_KEEPS_ATTR_NAME))
            {
                entry.second->removeAttr(OWN_DECLARED_LENDS_ATTR_NAME);
            }
        }
        // before the facts: an object literal holding `new Date(0)` borrows it as it does `new C()`,
        // which classifyRecordBoxes decides from whether the value is fresh (the loop below marks it
        // again, with the rest)
        for (auto &call : calls)
        {
            if (isImportedFreshResult(call))
            {
                call.op->setAttr(OWN_FRESH_RESULT_ATTR_NAME, mlir::UnitAttr::get(&getContext()));
            }
        }

        // before the facts: a value kept for good is immortal, which keeps nothing of its owner
        findKeepers(module);
        decideKeptForGood(module);
        // before the facts: a generator method whose state borrows `this` does not keep it
        findBorrowingStates(module);
        computeFacts();
        computeDrops();
        computeNoThrow();
        pinAssignedGlobals();
        rejectGlobalArgumentsDropped();
        rejectWeakTargetsShared(module);
        findBoundedForwarders();
        findYieldedFields(module);
        findImportedStates(module);
        checkDeclared();
        transferTemporaryReceivers();

        for (auto &call : calls)
        {
            // a generator another module defines, whose signature says what its state borrows
            if (auto bounded = declaredByCallees(call.callees, OWN_DECLARED_BORROWS_ATTR_NAME);
                !bounded.empty() && call.op->getNumResults() == 1)
            {
                call.op->setAttr(OWN_RESULT_BOUNDED_ATTR_NAME, mlir::DenseI32ArrayAttr::get(&getContext(), bounded));
                // and the caller owns it (the generator's state), as `@borrows` says: fresh, though
                // the function may be another module's
                call.op->setAttr(OWN_FRESH_RESULT_ATTR_NAME, mlir::UnitAttr::get(&getContext()));
            }

            // an iterator's `next()`, whoever implements it, and a generator's state, whoever makes
            // it: see isIteratorNext. Not a `next()` through a closed interface: its members keep
            // their facts (spec 31), and the call gets theirs like any other - unless every member is
            // a generator's own `next` (isGeneratorNext), whose facts say it reads its state though
            // it moves what it yields out. Nor one whose result is a borrowing record (spec 32): the
            // caller owns the box, but what it borrows bounds it.
            // ...or the thunk a generator is given a `Generator<T>` with (`next(value?)` over its own
            // `next()`, a method cell's function): it makes that one call and returns what it gives
            auto forwardsGeneratorNext = [&](mlir_ts::FuncOp thunk) {
                if (!thunk.getSymName().ends_with(".thunk") || thunk.getFunctionType().getNumResults() != 1)
                {
                    return false;
                }

                auto forwarded = 0;
                for (auto &inner : calls)
                {
                    if (inner.op->getParentOfType<mlir_ts::FuncOp>() != thunk)
                    {
                        continue;
                    }

                    if (!isIteratorNext(inner.op) || inner.callees.viaInterface || !inner.callees.known || inner.callees.funcs.empty() ||
                        !llvm::all_of(inner.callees.funcs, [](mlir_ts::FuncOp callee) { return isGeneratorNext(callee); }) ||
                        inner.op->getResult(0).getType() != thunk.getFunctionType().getResult(0))
                    {
                        return false;
                    }

                    forwarded++;
                }

                return forwarded == 1;
            };
            auto generatorsOnly = call.callees.viaInterface && call.callees.known && !call.callees.funcs.empty() &&
                                  llvm::all_of(call.callees.funcs, [&](mlir_ts::FuncOp callee) {
                                      return isGeneratorNext(callee) || forwardsGeneratorNext(callee);
                                  });
            // ...and an iterator's `throw()`, as its `next()` (TSLANG 187): a generator's own returns what its `next`
            // gives or throws (isGeneratorThrow), and any other is reached through calls this pass can't all see, so
            // it may not return a borrow either. Through a closed interface only when every member is a generator's
            auto generatorThrowsOnly = call.callees.viaInterface && call.callees.known && !call.callees.funcs.empty() &&
                                       llvm::all_of(call.callees.funcs, [](mlir_ts::FuncOp callee) { return isGeneratorThrow(callee); });
            if (((isIteratorNext(call.op) && (!call.callees.viaInterface || generatorsOnly)) ||
                 (isIteratorNext(call.op, "throw") && (!call.callees.viaInterface || generatorThrowsOnly)) ||
                 makesGeneratorState(call.op)) &&
                !isRecordResult(call.op))
            {
                call.op->setAttr(OWN_FRESH_RESULT_ATTR_NAME, mlir::UnitAttr::get(&getContext()));
            }

            call.op->removeAttr(OWN_DROP_FIELDS_ATTR_NAME);
            call.op->removeAttr(OWN_DROP_FIELD_OWNERS_ATTR_NAME);
            if (!call.callees.known)
            {
                // a method of a class another module may extend, called through its vtable: say so
                // where a borrow ends at the call (TSLANG 48)
                auto callee = mlir::isa<mlir_ts::CallInternalOp, mlir_ts::CallIndirectOp, mlir_ts::InvokeOp>(call.op)
                                  ? calleeValue(call.op)
                                  : mlir::Value();
                if (auto getMethodOp = callee ? callee.getDefiningOp<mlir_ts::GetMethodOp>() : mlir_ts::GetMethodOp())
                {
                    callee = getMethodOp.getBoundFunc();
                }

                // a method's slot, or an accessor's
                auto *refDef = callee ? callee.getDefiningOp() : nullptr;
                auto methodRef = mlir::dyn_cast_or_null<mlir_ts::ThisVirtualSymbolRefOp>(refDef);
                auto accessorRef = mlir::dyn_cast_or_null<mlir_ts::VirtualSymbolRefOp>(refDef);
                auto *refOp = methodRef ? methodRef.getOperation() : accessorRef ? accessorRef.getOperation() : nullptr;
                auto identifier = methodRef ? methodRef.getIdentifierAttr() : accessorRef ? accessorRef.getIdentifierAttr()
                                                                                         : mlir::FlatSymbolRefAttr();
                int64_t slotIndex = methodRef ? static_cast<int64_t>(methodRef.getIndex())
                                  : accessorRef ? static_cast<int64_t>(accessorRef.getIndex()) : -1;
                if (refOp && !refOp->hasAttr(FINAL_RECEIVER_ATTR_NAME))
                {
                    call.op->setAttr(OWN_OVERRIDABLE_CALL_ATTR_NAME, identifier);

                    // ...whose result is the caller's when no method of the family says it borrows: an
                    // override in another module may not say so either (checkOverrideResultFacts), and
                    // one that returns a borrow without saying it is an error there (TSLANG 49)
                    if (call.op->getNumResults() == 1 && !isRecordResult(call.op) &&
                        familyResultsFresh(slotIndex, identifier.getAttr()))
                    {
                        call.op->setAttr(OWN_FRESH_RESULT_ATTR_NAME, mlir::UnitAttr::get(&getContext()));
                    }
                }

                continue;
            }

            if (call.callees.instanceOf || (call.callees.imported && !call.callees.importedOverwrites) ||
                (!call.callees.imported &&
                 llvm::all_of(call.callees.funcs, [&](mlir_ts::FuncOp callee) { return noDrops.contains(callee); })))
            {
                call.op->setAttr(OWN_NO_DROPS_ATTR_NAME, mlir::UnitAttr::get(&getContext()));
            }
            else if (auto drops = calleesFieldDrops(call.callees))
            {
                llvm::SmallVector<mlir::Attribute> places;
                for (auto [position, type] : drops->places)
                {
                    places.push_back(mlir::ArrayAttr::get(
                        &getContext(), {mlir::IntegerAttr::get(mlir::IntegerType::get(&getContext(), 64), position),
                                        mlir::TypeAttr::get(type)}));
                }

                call.op->setAttr(OWN_DROP_FIELDS_ATTR_NAME, mlir::ArrayAttr::get(&getContext(), places));

                // ...and the argument whose object owns each (TSLANG 216)
                if (argumentsAlignWithParams(call.op, call.callees.viaCallback) &&
                    llvm::any_of(drops->owners, [](int32_t owner) { return owner >= 0; }))
                {
                    call.op->setAttr(OWN_DROP_FIELD_OWNERS_ATTR_NAME, mlir::DenseI32ArrayAttr::get(&getContext(), drops->owners));
                }
            }

            // `new C()` of a class from a library built under own: see OWN_FRESH_RESULT_ATTR_NAME
            // ...and of a function another own module's object says returns one its caller owns
            // (--own-facts)
            if (isImportedFreshResult(call))
            {
                call.op->setAttr(OWN_FRESH_RESULT_ATTR_NAME, mlir::UnitAttr::get(&getContext()));
            }

            // a generator's `next`, read out of its state object: a function defined here, which
            // returns its result retained like any other (its symbol sits in a constant, so it is
            // open, and its body may not return a borrow)
            if (call.callees.fieldFunction && call.op->getNumResults() == 1 && !isRecordResult(call.op))
            {
                call.op->setAttr(OWN_FRESH_RESULT_ATTR_NAME, mlir::UnitAttr::get(&getContext()));
            }

            // a callback's call, where every function that can reach it is defined here (spec 74): each
            // returns its result retained, as a direct call's does, unless its facts say it borrows
            // an argument. A body that returns anything else it does not own is refused on its own.
            if (call.callees.viaCallback && !call.callees.imported && !call.callees.funcs.empty() &&
                call.op->getNumResults() == 1 && !isRecordResult(call.op) &&
                llvm::all_of(call.callees.funcs, [&](mlir_ts::FuncOp callee) {
                    auto facts = factsEffective.lookup(callee);
                    return facts.borrows.empty() && !facts.record;
                }))
            {
                call.op->setAttr(OWN_FRESH_RESULT_ATTR_NAME, mlir::UnitAttr::get(&getContext()));
            }

            // a generator whose state object borrows from its arguments (not through a callback
            // family, whose member's parameters may be shifted)
            if (call.callees.funcs.size() == 1 && !call.callees.imported && !call.callees.viaCallback)
            {
                if (auto bounded = call.callees.funcs.front()->getAttrOfType<mlir::DenseI32ArrayAttr>(OWN_RESULT_BOUNDED_ATTR_NAME))
                {
                    call.op->setAttr(OWN_RESULT_BOUNDED_ATTR_NAME, bounded);
                }
            }
        }

        markOverridableReached();
        exportNoDrops();
        ownGlobalInitializerCalls(module);

        module.walk([&](mlir_ts::CreateBoundFunctionOp boundOp) {
            if (!boundOp->hasAttr(OWNS_CAPTURE_ATTR_NAME))
            {
                return;
            }

            auto assigned = assignedCaptures(closureBody(boundOp));
            if (!assigned.empty())
            {
                boundOp->setAttr(OWN_ASSIGNS_CAPTURES_ATTR_NAME,
                                 mlir::DenseI32ArrayAttr::get(&getContext(), assigned.getArrayRef()));
            }
        });

        // last: the boxes that borrow are settled
        giveBorrowingRecordsTheirVTables(module);
    }

    mlir_ts::FuncOp OwnershipSignaturePass::closureBody(mlir_ts::CreateBoundFunctionOp boundOp)
    {
        auto symbolRefOp = boundOp.getFunc().getDefiningOp<mlir_ts::SymbolRefOp>();
        if (!symbolRefOp)
        {
            return {};
        }

        auto found = functions.find(symbolRefOp.getIdentifier());
        return found != functions.end() && !found->second.isDeclaration() ? found->second : mlir_ts::FuncOp();
    }

    llvm::SetVector<int32_t> OwnershipSignaturePass::assignedCaptures(mlir_ts::FuncOp funcOp, int depth)
    {
        llvm::SetVector<int32_t> assigned;
        auto boxType = funcOp && funcOp.getNumArguments() > 0 && !funcOp.getBody().empty()
                           ? mlir::dyn_cast<mlir_ts::RefType>(funcOp.getArgument(0).getType())
                           : mlir_ts::RefType();
        auto tupleType = boxType ? mlir::dyn_cast<mlir_ts::TupleType>(boxType.getElementType()) : mlir_ts::TupleType();
        if (!tupleType || depth > 16)
        {
            // unknown: a box of any shape may have had any of its cells assigned
            for (int32_t index = 0; index < 64; ++index)
            {
                assigned.insert(index);
            }

            return assigned;
        }

        if (auto found = assignedCache.find(funcOp); found != assignedCache.end())
        {
            return found->second;
        }

        for (auto *user : funcOp.getArgument(0).getUsers())
        {
            auto propertyRefOp = mlir::dyn_cast<mlir_ts::PropertyRefOp>(user);
            if (!propertyRefOp)
            {
                continue;
            }

            auto field = propertyRefOp.getPosition();
            for (auto *fieldUser : propertyRefOp->getUsers())
            {
                auto loadOp = mlir::dyn_cast<mlir_ts::LoadOp>(fieldUser);
                if (!loadOp || !isLoadedCell(loadOp.getResult()))
                {
                    continue; // a field captured by value: a copy, not a variable
                }

                if (cellAssigned(loadOp.getResult(), depth))
                {
                    assigned.insert(field);
                }
            }
        }

        assignedCache[funcOp] = assigned;
        return assigned;
    }

    bool OwnershipSignaturePass::cellAssigned(mlir::Value cell, int depth)
    {
        for (auto &use : cell.getUses())
        {
            auto *user = use.getOwner();
            if (mlir::isa<mlir_ts::LoadOp, mlir_ts::RetainCellOp>(user))
            {
                continue;
            }

            if (mlir::isa<mlir_ts::ReleaseSlotOp>(user))
            {
                return true;
            }

            auto storeOp = mlir::dyn_cast<mlir_ts::StoreOp>(user);
            if (!storeOp)
            {
                return true; // the cell goes somewhere this does not follow
            }

            if (storeOp.getReference() == cell)
            {
                return true;
            }

            // captured again, by a closure built here: the field it fills in that closure's box
            auto propertyRefOp = storeOp.getReference().getDefiningOp<mlir_ts::PropertyRefOp>();
            auto boundOp = propertyRefOp ? closureOfBox(propertyRefOp.getObjectRef()) : mlir_ts::CreateBoundFunctionOp();
            if (!boundOp || assignedCaptures(closureBody(boundOp), depth + 1).contains(propertyRefOp.getPosition()))
            {
                return true;
            }
        }

        return false;
    }

    llvm::ArrayRef<int32_t> OwnershipSignaturePass::declared(mlir::Operation *funcOp, llvm::StringRef name)
    {
        if (auto attr = funcOp->getAttrOfType<mlir::DenseI32ArrayAttr>(name))
        {
            return attr.asArrayRef();
        }

        return {};
    }

    bool OwnershipSignaturePass::declaresFacts(mlir::Operation *funcOp)
    {
        return funcOp->hasAttr(OWN_DECLARED_KEEPS_ATTR_NAME) || funcOp->hasAttr(OWN_DECLARED_BORROWS_ATTR_NAME) ||
               funcOp->hasAttr(OWN_DECLARED_LENDS_ATTR_NAME);
    }

    bool OwnershipSignaturePass::declaresCallerFacts(mlir::Operation *funcOp)
    {
        return funcOp->hasAttr(OWN_DECLARED_KEEPS_ATTR_NAME) || funcOp->hasAttr(OWN_DECLARED_LENDS_ATTR_NAME);
    }

    llvm::SmallVector<int32_t> OwnershipSignaturePass::declaredLends(mlir::Operation *funcOp)
    {
        auto lends = declared(funcOp, OWN_DECLARED_LENDS_ATTR_NAME);
        llvm::SmallVector<int32_t> sorted(lends.begin(), lends.end());
        llvm::sort(sorted);
        sorted.erase(std::unique(sorted.begin(), sorted.end()), sorted.end());
        return sorted;
    }

    bool OwnershipSignaturePass::calleesDeclare(const Callees &callees, llvm::StringRef name)
    {
        return !callees.named.empty() && llvm::all_of(callees.named, [&](mlir::StringAttr symbol) {
            mlir::Operation *declaring = functions.lookup(symbol.getValue());
            if (!declaring)
            {
                declaring = globals.lookup(symbol.getValue());
            }

            return declaring && declaring->hasAttr(name);
        });
    }

    llvm::SmallVector<int32_t> OwnershipSignaturePass::declaredByCallees(const Callees &callees, llvm::StringRef name)
    {
        llvm::SmallVector<int32_t> agreed;
        for (auto [index, symbol] : llvm::enumerate(callees.named))
        {
            // a function, or the global holding the address of one a library exports (a dynamic
            // import), which MLIRGen gave the declaration's markers
            mlir::Operation *declaring = functions.lookup(symbol.getValue());
            if (!declaring)
            {
                declaring = globals.lookup(symbol.getValue());
            }

            if (!declaring || !declaring->hasAttr(name))
            {
                return {};
            }

            auto indices = declared(declaring, name);
            if (index == 0)
            {
                agreed.assign(indices.begin(), indices.end());
            }
            else if (llvm::ArrayRef<int32_t>(agreed) != indices)
            {
                return {};
            }
        }

        return agreed;
    }

    std::string OwnershipSignaturePass::describeParams(mlir_ts::FuncOp funcOp, llvm::ArrayRef<int32_t> indices)
    {
        if (indices.empty())
        {
            return "nothing";
        }

        auto names = funcOp->getAttrOfType<mlir::ArrayAttr>(OWN_DECLARED_PARAM_NAMES_ATTR_NAME);
        std::string text;
        llvm::raw_string_ostream os(text);
        for (auto [position, index] : llvm::enumerate(indices))
        {
            os << (position ? ", " : "") << "'";
            if (names && static_cast<size_t>(index) < names.size())
            {
                os << mlir::cast<mlir::StringAttr>(names[index]).getValue();
            }
            else
            {
                os << "#" << index;
            }

            os << "'";
        }

        return os.str();
    }

    int32_t OwnershipSignaturePass::parameterOf(mlir::Value value, mlir_ts::FuncOp funcOp)
    {
        auto root = rootOf(value);
        // the object a method is called on (`this.iter()` of a class that is not `@final`)
        if (auto object = boundThis(root))
        {
            return parameterOf(object, funcOp);
        }

        if (auto loadOp = root.getDefiningOp<mlir_ts::LoadOp>())
        {
            if (auto propertyRefOp = loadOp.getReference().getDefiningOp<mlir_ts::PropertyRefOp>())
            {
                return parameterOf(propertyRefOp->getOperand(0), funcOp);
            }

            auto varOp = loadOp.getReference().getDefiningOp<mlir_ts::VariableOp>();
            int index = -1;
            if (varOp && isParameterSlot(varOp, index) && varOp->getParentOfType<mlir_ts::FuncOp>() == funcOp)
            {
                return index;
            }

            return -1;
        }

        auto argument = mlir::dyn_cast<mlir::BlockArgument>(root);
        if (argument && argument.getOwner()->isEntryBlock() && argument.getOwner()->getParentOp() == funcOp.getOperation())
        {
            return static_cast<int32_t>(argument.getArgNumber());
        }

        return -1;
    }

    void OwnershipSignaturePass::checkDeclared()
    {
        for (auto &entry : functions)
        {
            auto funcOp = entry.second;
            if (funcOp.isDeclaration())
            {
                continue;
            }

            if (funcOp->hasAttr(OWN_DECLARED_LENDS_ATTR_NAME) && funcOp->hasAttr(OWN_DECLARED_BORROWS_ATTR_NAME))
            {
                funcOp.emitError("'") << funcOp.getSymName()
                                      << "' cannot say both '@" LENDS_DECORATOR "' and '@" BORROWS_DECORATOR
                                         "': its caller owns a result that borrows, and nothing of one it is lent";
                signalPassFailure();
                continue;
            }

            if (funcOp->hasAttr(OWN_DECLARED_RECORD_ATTR_NAME) && !funcOp->hasAttr(OWN_DECLARED_LENDS_ATTR_NAME))
            {
                funcOp.emitError("'@" RECORD_DECORATOR "' of '")
                    << funcOp.getSymName()
                    << "' says its result is a borrowing record, but no '@" LENDS_DECORATOR "' names what it borrows";
                signalPassFailure();
                continue;
            }

            // a method whose result type is `this` (spec 78): it lends its `this` alone, and a body here
            // returns that object, as it is, on every path
            if (funcOp->hasAttr(OWN_DECLARED_RETURNS_THIS_ATTR_NAME))
            {
                auto inputs = funcOp.getFunctionType().getInputs();
                auto lends = declaredLends(funcOp);
                // (no `@lends` left: the one the result type implied, dropped for a function only this module calls)
                if ((!lends.empty() && (lends.size() != 1 || lends.front() != 0)) || inputs.empty() ||
                    !mlir::isa<mlir_ts::ClassType>(inputs.front()) ||
                    funcOp->hasAttr(OWN_DECLARED_RECORD_ATTR_NAME))
                {
                    funcOp.emitError("'") << funcOp.getSymName()
                                          << "' returns 'this', so it lends 'this' alone: '@" LENDS_DECORATOR
                                             "' cannot name another parameter, nor '@" RECORD_DECORATOR "' be said";
                    signalPassFailure();
                    continue;
                }

                if (!funcOp.isDeclaration() && !factsLocal.lookup(funcOp).itself)
                {
                    funcOp.emitError("'") << funcOp.getSymName()
                                          << "' has the result type 'this', but its body returns something else than the "
                                             "object it is called on, on some path; -mm=own takes 'this' for that object";
                    signalPassFailure();
                    continue;
                }
            }

            if (declaresCallerFacts(funcOp))
            {
                auto local = factsLocal.lookup(funcOp);
                auto keeps = declared(funcOp, OWN_DECLARED_KEEPS_ATTR_NAME);
                auto lends = declaredLends(funcOp);
                if (local.borrows != lends)
                {
                    if (lends.empty())
                    {
                        funcOp.emitError("'") << funcOp.getSymName() << "' returns a borrow of "
                                              << describeParams(funcOp, local.borrows)
                                              << ", which it has to say with '@" LENDS_DECORATOR "'";
                    }
                    else if (local.borrows.empty())
                    {
                        funcOp.emitError("'@" LENDS_DECORATOR "' of '")
                            << funcOp.getSymName() << "' names " << describeParams(funcOp, lends)
                            << ", but its body does not return a borrow of it";
                    }
                    else
                    {
                        funcOp.emitError("'@" LENDS_DECORATOR "' of '")
                            << funcOp.getSymName() << "' names " << describeParams(funcOp, lends)
                            << ", but its body returns a borrow of " << describeParams(funcOp, local.borrows);
                    }

                    signalPassFailure();
                }
                else if (llvm::ArrayRef<int32_t>(local.owned) != keeps)
                {
                    if (funcOp->hasAttr(OWN_DECLARED_KEEPS_ATTR_NAME))
                    {
                        funcOp.emitError("'@" KEEPS_DECORATOR "' of '")
                            << funcOp.getSymName() << "' names " << describeParams(funcOp, keeps) << ", but its body keeps "
                            << describeParams(funcOp, local.owned);
                    }
                    else
                    {
                        funcOp.emitError("'") << funcOp.getSymName() << "' keeps " << describeParams(funcOp, local.owned)
                                              << ", which it has to say with '@" KEEPS_DECORATOR "'";
                    }

                    signalPassFailure();
                }
                else if (!lends.empty() && local.record && local.unboxed && !executable &&
                         (funcOp->hasAttr("export") || funcOp->hasAttr(DLL_EXPORT)))
                {
                    // `@record` tells an importer it owns a box; an unboxed record (spec 36) owns nothing,
                    // and the importer's release of it would free what it borrows
                    funcOp.emitError("'")
                        << funcOp.getSymName() << "' returns a tuple that borrows " << describeParams(funcOp, lends)
                        << " and owns nothing, which -mm=own cannot tell another module yet";
                    signalPassFailure();
                }
                else if (!lends.empty() && funcOp->hasAttr(OWN_DECLARED_RECORD_ATTR_NAME) != local.record)
                {
                    // An importer knows the result is a record only from `@record` (spec 32): told a plain
                    // borrow, it never frees the box; told a record, it frees an argument it does not own.
                    // Callers in this module go by the body, so a function no other module calls need
                    // not say it.
                    if (funcOp->hasAttr(OWN_DECLARED_RECORD_ATTR_NAME))
                    {
                        funcOp.emitError("'@" RECORD_DECORATOR "' of '")
                            << funcOp.getSymName() << "' says its result is a record that borrows "
                            << describeParams(funcOp, lends) << ", but its body returns a borrow of it as it is";
                        signalPassFailure();
                    }
                    else if (!executable && (funcOp->hasAttr("export") || funcOp->hasAttr(DLL_EXPORT)))
                    {
                        funcOp.emitError("'")
                            << funcOp.getSymName() << "' returns a record that borrows " << describeParams(funcOp, lends)
                            << ", which another module can learn only from '@" RECORD_DECORATOR "' beside its '@" LENDS_DECORATOR
                               "'";
                        signalPassFailure();
                    }
                }
            }

            if (funcOp->hasAttr(OWN_DECLARED_BORROWS_ATTR_NAME))
            {
                auto says = declared(funcOp, OWN_DECLARED_BORROWS_ATTR_NAME);
                auto found = borrowsFound.find(funcOp);
                if (found == borrowsFound.end())
                {
                    funcOp.emitError("'@" BORROWS_DECORATOR "' of '")
                        << funcOp.getSymName() << "' names " << describeParams(funcOp, says)
                        << ", but it neither makes a generator whose state borrows its arguments nor returns one";
                    signalPassFailure();
                }
                else if (llvm::ArrayRef<int32_t>(found->second) != says)
                {
                    funcOp.emitError("'@" BORROWS_DECORATOR "' of '")
                        << funcOp.getSymName() << "' names " << describeParams(funcOp, says)
                        << ", but the state it makes borrows " << describeParams(funcOp, found->second);
                    signalPassFailure();
                }
            }
        }

        // the members of a family a declaring method is in: each says the same
        for (auto &[key, members] : families)
        {
            mlir_ts::FuncOp declaring;
            for (auto member : members)
            {
                if (auto funcOp = functions.lookup(member.getValue()); funcOp && declaresFacts(funcOp))
                {
                    declaring = funcOp;
                    break;
                }
            }

            if (!declaring)
            {
                continue;
            }

            for (auto member : members)
            {
                auto funcOp = functions.lookup(member.getValue());
                if (!funcOp || funcOp == declaring)
                {
                    continue;
                }

                for (auto *name : {OWN_DECLARED_KEEPS_ATTR_NAME, OWN_DECLARED_BORROWS_ATTR_NAME, OWN_DECLARED_LENDS_ATTR_NAME,
                                   OWN_DECLARED_RECORD_ATTR_NAME, OWN_DECLARED_RETURNS_THIS_ATTR_NAME})
                {
                    if (funcOp->hasAttr(name) != declaring->hasAttr(name) || declared(funcOp, name) != declared(declaring, name))
                    {
                        funcOp.emitError("'") << funcOp.getSymName() << "' overrides '" << declaring.getSymName()
                                              << "', and has to say the same '@" KEEPS_DECORATOR "', '@" BORROWS_DECORATOR
                                                 "', '@" LENDS_DECORATOR "' and '@" RECORD_DECORATOR "'";
                        signalPassFailure();
                        break;
                    }
                }
            }
        }
    }

    bool OwnershipSignaturePass::factsThroughInterface(mlir::Operation *funcOp, const Facts &facts)
    {
        return !facts.empty() && interfaceReached.contains(funcOp);
    }

    llvm::SmallVector<mlir::Value> OwnershipSignaturePass::returnedValues(mlir_ts::FuncOp funcOp)
    {
        llvm::SmallVector<mlir::Value> values;
        funcOp.walk([&](mlir_ts::ReturnInternalOp returnOp) {
            for (auto operand : returnOp.getRetOperands())
            {
                auto loadOp = operand.getDefiningOp<mlir_ts::LoadOp>();
                auto slot = loadOp ? loadOp.getReference().getDefiningOp<mlir_ts::VariableOp>() : mlir_ts::VariableOp();
                if (!slot || slot.getInitializer() || isOwningVariable(slot))
                {
                    values.push_back(operand);
                    continue;
                }

                for (auto *user : slot.getResult().getUsers())
                {
                    auto storeOp = mlir::dyn_cast<mlir_ts::StoreOp>(user);
                    if (storeOp && storeOp.getReference() == slot.getResult())
                    {
                        values.push_back(storeOp.getValue());
                    }
                }
            }
        });

        return values;
    }

    OwnershipSignaturePass::ResultBorrows OwnershipSignaturePass::localResultBorrows(mlir_ts::FuncOp funcOp)
    {
        auto found = -1;
        auto record = false;
        std::optional<bool> unboxed;
        llvm::SmallVector<int32_t> params;
        for (auto value : returnedValues(funcOp))
        {
            if (holdsNoBlock(value))
            {
                continue;
            }

            // a handle returned is a counted copy, never a borrow (spec 23.2)
            if (isHandle(value))
            {
                return {};
            }

            auto tuples = false;
            switch (recordParamsOf(value, params, &tuples))
            {
            case RecordFlow::Bad:
                return {};
            case RecordFlow::Record:
                // a box on one path and a tuple that owns nothing on another: the caller's release of
                // the one would be wrong for the other
                if (unboxed && *unboxed != tuples)
                {
                    return {};
                }

                unboxed = tuples;
                record = true;
                continue;
            case RecordFlow::None:
                break;
            }

            auto index = borrowedParam(value);
            if (index < 0 || (found >= 0 && index != found))
            {
                return {};
            }

            found = index;
        }

        if (record && found >= 0)
        {
            return {};
        }

        if (record)
        {
            llvm::sort(params);
            params.erase(std::unique(params.begin(), params.end()), params.end());
            return {params, true, unboxed.value_or(false)};
        }

        if (found >= 0)
        {
            ResultBorrows result{{found}, false};
            auto returned = returnedValues(funcOp);
            result.itself = !returned.empty() && llvm::all_of(returned, [&](mlir::Value value) { return isParameterItself(value, found); });
            return result;
        }

        return {};
    }

    void OwnershipSignaturePass::transferTemporaryReceivers()
    {
        // in the order of the program, so a chain's next call sees the one before it fresh; the
        // releases go once every call is decided (a walk must not erase what it has yet to visit)
        llvm::SmallVector<mlir::Operation *> candidates;
        getOperation().walk([&](mlir::Operation *op) {
            if (op->hasAttr(OWN_RESULT_ITSELF_ATTR_NAME) && op->getNumResults() == 1 && resultBorrowsAll(op).size() == 1)
            {
                candidates.push_back(op);
            }
        });

        llvm::SetVector<mlir::Operation *> erased;
        for (auto *op : candidates)
        {
            // a temporary: made fresh, not kept by the callee, given to nothing after the call but
            // its releases
            auto args = callArgs(op);
            auto index = resultBorrowsAll(op).front();
            if (index < 0 || static_cast<size_t>(index) >= args.size() || llvm::is_contained(ownedParams(op), index))
            {
                continue;
            }

            // a method called through the object's vtable or by its symbol: the object it is bound to
            auto receiver = args[index];
            if (auto object = boundThis(receiver); object && mlir::isa<mlir_ts::ClassType>(object.getType()))
            {
                receiver = object;
            }

            // ...seen through its views: `same(new A())` to an `A | null` parameter is given
            // `<A | null>new A()`, and the releases to take over are the object's. Judged on the
            // view, whose only user is the call, the result was made fresh while the object's own
            // release stayed: freed twice (TSLANG 46).
            receiver = rootOf(receiver);

            // the caller's own: fresh, or what a call that borrows nothing returns (another module's
            // `JsonValue.newObject()`), which the caller releases
            auto *producer = receiver.getDefiningOp();
            auto owned = isFresh(receiver) ||
                         (producer && isCall(producer) && resultBorrowsAll(producer).empty() && !isRecordResult(producer));
            if (!owned)
            {
                continue;
            }

            // a release after the call: in its block, or in one only the call leads to (a condition
            // after it, `x instanceof A`, splits the function's blocks) where the result is released
            // too, so the result's release is there to take the temporary's place
            mlir::DominanceInfo dominance(op->getParentOfType<mlir_ts::FuncOp>());
            auto releasedAfter = [&](mlir::Operation *user) {
                if (user->getBlock() == op->getBlock())
                {
                    return op->isBeforeInBlock(user);
                }

                return dominance.properlyDominates(op, user) && llvm::any_of(op->getResult(0).getUsers(), [&](mlir::Operation *use) {
                           return mlir::isa<mlir_ts::ReleaseOp>(use) && use->getBlock() == user->getBlock();
                       });
            };

            llvm::SmallVector<mlir::Operation *> releases;
            auto temporary = llvm::all_of(receiver.getUsers(), [&](mlir::Operation *user) {
                if (user == op)
                {
                    return true;
                }

                if (mlir::isa<mlir_ts::ReleaseOp>(user) && releasedAfter(user))
                {
                    releases.push_back(user);
                    return true;
                }

                // a use before the call, whose own uses are before it too or the call itself (rc's
                // header of a new object, the method bound to it)
                std::function<bool(mlir::Operation *, int)> before = [&](mlir::Operation *other, int depth) {
                    return depth < 4 && other->getBlock() == op->getBlock() && other->isBeforeInBlock(op) &&
                           llvm::all_of(other->getUsers(), [&](mlir::Operation *use) { return use == op || before(use, depth + 1); });
                };
                return before(user, 0);
            });
            if (!temporary)
            {
                continue;
            }

            // the result is the temporary itself: its releases are the result's to make
            erased.insert(releases.begin(), releases.end());
            op->removeAttr(OWN_RESULT_BORROWS_ATTR_NAME);
            op->removeAttr(OWN_RESULT_ITSELF_ATTR_NAME);
            op->setAttr(OWN_FRESH_RESULT_ATTR_NAME, mlir::UnitAttr::get(op->getContext()));
        }

        for (auto *release : erased)
        {
            release->erase();
        }
    }

    bool OwnershipSignaturePass::isParameterItself(mlir::Value value, int32_t index)
    {
        auto root = rootOf(value);
        if (auto argument = mlir::dyn_cast<mlir::BlockArgument>(root))
        {
            return argument.getOwner()->isEntryBlock() && mlir::isa<mlir_ts::FuncOp>(argument.getOwner()->getParentOp()) &&
                   static_cast<int32_t>(argument.getArgNumber()) == index;
        }

        auto loadOp = root.getDefiningOp<mlir_ts::LoadOp>();
        auto varOp = loadOp ? loadOp.getReference().getDefiningOp<mlir_ts::VariableOp>() : mlir_ts::VariableOp();
        auto position = -1;
        return varOp && isParameterSlot(varOp, position) && position == index;
    }

    int32_t OwnershipSignaturePass::mixedReturnParam(mlir_ts::FuncOp funcOp)
    {
        auto param = -1;
        auto other = false;
        auto returned = returnedValues(funcOp);
        llvm::SmallVector<mlir::Value> work(returned.begin(), returned.end());
        llvm::DenseSet<mlir::Value> seen;
        while (!work.empty())
        {
            auto value = work.pop_back_val();
            if (!seen.insert(value).second || holdsNoBlock(value))
            {
                continue;
            }

            if (isHandle(value) || isCopyableString(value.getType()))
            {
                return -1;
            }

            auto root = rootOf(value);
            if (auto argument = mlir::dyn_cast<mlir::BlockArgument>(root); argument && !argument.getOwner()->isEntryBlock())
            {
                llvm::SmallVector<mlir::Value> merged;
                if (!mergedInto(argument, merged))
                {
                    return -1;
                }

                work.append(merged.begin(), merged.end());
                continue;
            }

            int32_t index = -1;
            if (readsParameterSlot(root, funcOp, index))
            {
                if ((param >= 0 && param != index) || isHandle(root) || isCopyableString(root.getType()))
                {
                    return -1;
                }

                param = index;
                continue;
            }

            llvm::SmallVector<int32_t> recordParams;
            if (borrowedParam(root) >= 0 || recordParamsOf(root, recordParams) != RecordFlow::None)
            {
                return -1;
            }

            other = true;
        }

        return other ? param : -1;
    }

    bool OwnershipSignaturePass::keeps(mlir::Operation *user, mlir::Value used)
    {
        // a number, a boolean, a string literal: nothing to keep
        if (holdsNoBlock(used))
        {
            return false;
        }

        if (auto storeOp = mlir::dyn_cast<mlir_ts::StoreOp>(user))
        {
            // a field a borrowing record borrows (spec 32), or `this` in the box of a generator whose
            // state borrows it: the record keeps nothing
            if (isBorrowedRecordFieldStore(user) || user->hasAttr(OWN_STATE_BORROW_ATTR_NAME))
            {
                return false;
            }

            // `this` in a closure's box (captured by copy, spec 70), any other parameter nothing
            // assigns, which a closure captures by copy too, or a copy the enclosing closure's box
            // holds: the object is the caller's, which no box can own. A closure that does not escape
            // borrows it, as it did the cell; one stored into a field of `this` is a back-reference;
            // any other one is inference's error - unless the closure is kept itself (stored into a
            // field, a global, an array, or given to a call that keeps it): then it owns the parameter,
            // as a field it is stored into would, and the callers give it up (spec 76). Not the
            // member's `this`, which its callers cannot give to an object the object owns.
            if (auto propertyRefOp = storeOp.getReference().getDefiningOp<mlir_ts::PropertyRefOp>();
                propertyRefOp && isBorrowableCapture(used) && closureOfBox(propertyRefOp->getOperand(0)))
            {
                return storeOp.getValue() == used && parameterRead(used) && !isMemberThis(used) &&
                       keepsClosure(propertyRefOp->getOperand(0));
            }

            // ...or a bound method of `this` stored into a field of `this`: it borrows the object
            if (isBoundMethodOfMemberThis(storeOp.getValue()) && isFieldOfMemberThis(storeOp.getReference()))
            {
                return false;
            }

            // a string stored by a statement that may run again (`a[i] = value` in `fill`'s loop) is
            // no one move: each store takes a copy, and the parameter stays a borrow (spec 50)
            if (isCopyableString(used.getType()) && mayRunAgain(user))
            {
                return false;
            }

            auto ref = storeOp.getReference();
            return storeOp.getValue() == used && (isPlace(ref) || ref.getDefiningOp<mlir_ts::AddressOfOp>());
        }

        if (auto pushOp = mlir::dyn_cast<mlir_ts::ArrayPushOp>(user))
        {
            return llvm::is_contained(pushOp.getItems(), used);
        }

        // `throw reason` takes the value with the exception: the catch owns it, not the caller
        // (releaseKeptParamsAtExits places no release at such a throw)
        if (mlir::isa<mlir_ts::ThrowOp, mlir_ts::ThrowCallOp, mlir_ts::ThrowUnwindOp>(user))
        {
            return user->getOperand(0) == used;
        }

        // `new Shared(x)` moves `x` into the block, as a store into a place does (spec 23.1)
        if (auto sharedNewOp = mlir::dyn_cast<mlir_ts::SharedNewOp>(user))
        {
            return sharedNewOp.getValue() == used;
        }

        if (auto unshiftOp = mlir::dyn_cast<mlir_ts::ArrayUnshiftOp>(user))
        {
            return llvm::is_contained(unshiftOp.getItems(), used);
        }

        if (auto spliceOp = mlir::dyn_cast<mlir_ts::ArraySpliceOp>(user))
        {
            return llvm::is_contained(spliceOp.getItems(), used);
        }

        if (isCall(user))
        {
            auto args = callArgs(user);
            return llvm::any_of(ownedParams(user), [&](int32_t index) {
                return static_cast<size_t>(index) < args.size() && args[index] == used;
            });
        }

        // `let r = init; ... r = f(r, v)`: a local that owns what it is declared from and is given
        // another value is no borrow of the parameter - the assignment releases what it holds - so
        // the parameter moves into it (spec 33: `___array_reduce`'s accumulator). Not a string: the
        // local copies it (spec 22). Nor a local that only walks down from it (`n = n.next`, spec 83):
        // what it is given is read out of what it holds, and it borrows all of it.
        if (auto varOp = mlir::dyn_cast<mlir_ts::VariableOp>(user))
        {
            return varOp.getInitializer() == used && isOwningVariable(varOp) && !varOp.getCaptured().value_or(false) &&
                   isAssigned(varOp) && !isCopyableString(used.getType()) && !walksDownOnly(varOp) &&
                   !givenOnlyBorrows(varOp);
        }

        return false;
    }

    bool OwnershipSignaturePass::keepsClosure(mlir::Value box)
    {
        auto boundOp = closureOfBox(box);
        if (!boundOp || isBackReferenceBox(box))
        {
            return false;
        }

        auto kept = false;
        forEachUse(boundOp.getResult(), [&](mlir::Operation *use, mlir::Value used) { kept = kept || keeps(use, used); });
        return kept;
    }

    int32_t OwnershipSignaturePass::parameterOfCell(mlir_ts::VariableOp cellOp)
    {
        auto argument = cellOp.getInitializer() ? mlir::dyn_cast<mlir::BlockArgument>(cellOp.getInitializer())
                                                : mlir::BlockArgument();
        if (!cellOp.getCaptured().value_or(false) || isOwningVariable(cellOp) || !argument ||
            !argument.getOwner()->isEntryBlock() || !mlir::isa<mlir_ts::FuncOp>(argument.getOwner()->getParentOp()))
        {
            return -1;
        }

        return static_cast<int32_t>(argument.getArgNumber());
    }

    llvm::SmallVector<int32_t> OwnershipSignaturePass::localOwnedParams(mlir_ts::FuncOp funcOp)
    {
        llvm::SmallVector<int32_t> owned;
        if (funcOp.getBody().empty())
        {
            return owned;
        }

        for (auto argument : funcOp.getBody().front().getArguments())
        {
            // a handle kept is a counted copy: its caller gives nothing up (spec 23.2)
            if (isHandle(argument))
            {
                continue;
            }

            auto kept = false;
            for (auto varOp : parameterSlotsOf(argument))
            {
                for (auto *slotUser : varOp.getResult().getUsers())
                {
                    if (auto loadOp = mlir::dyn_cast<mlir_ts::LoadOp>(slotUser))
                    {
                        forEachUse(loadOp.getResult(), [&](mlir::Operation *use, mlir::Value used) {
                            kept = kept || keeps(use, used);
                        });
                    }

                    // an element taken out of a rest parameter and kept (`this.data.push(args.shift())`):
                    // the function takes what its callers move into the array, so it keeps the array and
                    // they give their elements up (spec 72)
                    if (mlir::isa<mlir_ts::ArrayPopOp, mlir_ts::ArrayShiftOp>(slotUser) && restParameterOf(varOp.getResult()))
                    {
                        forEachUse(slotUser->getResult(0), [&](mlir::Operation *use, mlir::Value used) {
                            kept = kept || keeps(use, used);
                        });
                    }
                }
            }

            // the parameter's cell, given to a closure that is kept: the closure owns the cell, and
            // the value in it, so the function keeps the parameter (spec 76)
            for (auto *argumentUser : argument.getUsers())
            {
                auto cellOp = mlir::dyn_cast<mlir_ts::VariableOp>(argumentUser);
                if (kept || !cellOp || parameterOfCell(cellOp) != static_cast<int32_t>(argument.getArgNumber()) ||
                    !own_facts::ownsHeap(cellOp.getLoc(), argument.getType()))
                {
                    continue;
                }

                for (auto *cellUser : cellOp->getUsers())
                {
                    auto storeOp = mlir::dyn_cast<mlir_ts::StoreOp>(cellUser);
                    auto propertyRefOp = storeOp && storeOp.getValue() == cellOp.getResult()
                                             ? storeOp.getReference().getDefiningOp<mlir_ts::PropertyRefOp>()
                                             : mlir_ts::PropertyRefOp();
                    kept = kept || (propertyRefOp && keepsClosure(propertyRefOp->getOperand(0)));
                }
            }

            if (kept)
            {
                owned.push_back(argument.getArgNumber());
            }
        }

        // a generator whose state owns what it holds of its arguments, as its `@keeps` says: the
        // callers move them in, and the state's release frees them (spec 38)
        if (funcOp->hasAttr(OWN_DECLARED_KEEPS_ATTR_NAME))
        {
            for (auto index : paramsCapturedIntoState(funcOp))
            {
                if (!llvm::is_contained(owned, index))
                {
                    owned.insert(llvm::upper_bound(owned, index), index);
                }
            }
        }

        return owned;
    }

    llvm::SmallVector<int32_t> OwnershipSignaturePass::takenOnEveryPath(mlir_ts::FuncOp funcOp, llvm::ArrayRef<int32_t> kept)
    {
        if (kept.empty())
        {
            return {};
        }

        if (auto found = everyPathCache.find(funcOp); found != everyPathCache.end())
        {
            return found->second;
        }

        auto &body = funcOp.getBody();
        mlir::DominanceInfo dominance(funcOp);
        llvm::SmallVector<mlir::Operation *> exits;
        funcOp.walk([&](mlir_ts::ReturnInternalOp returnOp) { exits.push_back(body.findAncestorOpInRegion(*returnOp)); });

        llvm::SmallVector<int32_t> taken;
        for (auto index : kept)
        {
            // the statements that keep the parameter, as decideParam sees a move
            llvm::SmallVector<mlir::Operation *> takers;
            auto rest = false;
            for (auto varOp : parameterSlotsOf(body.front().getArgument(index)))
            {
                rest = rest || restParameterOf(varOp.getResult());
                for (auto *slotUser : varOp.getResult().getUsers())
                {
                    if (auto loadOp = mlir::dyn_cast<mlir_ts::LoadOp>(slotUser))
                    {
                        forEachUse(loadOp.getResult(), [&](mlir::Operation *use, mlir::Value used) {
                            if (keeps(use, used))
                            {
                                takers.push_back(body.findAncestorOpInRegion(*use));
                            }
                        });
                    }
                }
            }

            // a rest parameter kept only for the elements taken out of it (spec 72): the array itself is
            // never taken, so the function releases it
            if (takers.empty() && rest)
            {
                continue;
            }

            // a parameter kept only through what this pass cannot see (a mixed return): as before
            auto everyPath = takers.empty() || llvm::all_of(exits, [&](mlir::Operation *exit) {
                return exit && llvm::any_of(takers, [&](mlir::Operation *taker) { return taker && dominance.dominates(taker, exit); });
            });
            if (everyPath)
            {
                taken.push_back(index);
            }
        }

        everyPathCache[funcOp] = taken;
        return taken;
    }

    void OwnershipSignaturePass::computeFacts()
    {
        llvm::DenseMap<mlir::Operation *, Facts> pinned;
        for (auto round = 0; round < 16; ++round)
        {
            // which object literals borrow depends on what the calls stored into them return, and a
            // function's facts on which literals borrow (spec 32)
            classifyRecordBoxes();

            for (auto &entry : functions)
            {
                if (!entry.second.isDeclaration())
                {
                    auto result = localResultBorrows(entry.second);
                    Facts facts;
                    facts.borrows = std::move(result.borrows);
                    facts.record = result.record;
                    facts.unboxed = result.unboxed;
                    facts.itself = result.itself;
                    facts.owned = localOwnedParams(entry.second);
                    // what the body takes into something on every path; a parameter it returns on
                    // some paths only is kept too, and released on the others (spec 33)
                    takenParams[entry.second] = takenOnEveryPath(entry.second, facts.owned);
                    if (auto mixed = mixedReturnParam(entry.second); mixed >= 0 && !llvm::is_contained(facts.owned, mixed))
                    {
                        facts.owned.insert(llvm::upper_bound(facts.owned, mixed), mixed);
                    }

                    factsLocal[entry.second] = std::move(facts);
                }
            }

            meetFacts();

            llvm::DenseMap<mlir::Operation *, Facts> next;
            for (auto &call : calls)
            {
                auto facts = callFacts(call.callees);
                if (facts.empty())
                {
                    // a call this pass cannot resolve still has what its callees' signatures say.
                    // A declaration's `@lends` is a plain borrow to its callers, and a record the
                    // caller owns (spec 32) with `@record`: its body is not here to say which.
                    auto keeps = declaredByCallees(call.callees, OWN_DECLARED_KEEPS_ATTR_NAME);
                    facts.owned.assign(keeps.begin(), keeps.end());
                    if (auto lends = declaredByCallees(call.callees, OWN_DECLARED_LENDS_ATTR_NAME); !lends.empty())
                    {
                        llvm::sort(lends);
                        lends.erase(std::unique(lends.begin(), lends.end()), lends.end());
                        facts.borrows = lends;
                        facts.record = calleesDeclare(call.callees, OWN_DECLARED_RECORD_ATTR_NAME);
                        facts.itself = lends.size() == 1 && !facts.record &&
                                       calleesDeclare(call.callees, OWN_DECLARED_RETURNS_THIS_ATTR_NAME);
                    }
                }

                if (!facts.empty())
                {
                    next[call.op] = facts;
                }
            }

            auto settled = next == pinned;
            pinned = std::move(next);
            pinFacts(pinned);
            if (settled)
            {
                return;
            }
        }

        factsEffective.clear();
        pinFacts(llvm::DenseMap<mlir::Operation *, Facts>());
    }

    void OwnershipSignaturePass::meetFacts()
    {
        factsEffective.clear();
        for (auto &[funcOp, facts] : factsLocal)
        {
            if (!facts.empty() && !open.contains(funcOp) && !factsThroughInterface(funcOp, facts))
            {
                factsEffective[funcOp] = facts;
            }
        }

        // what a signature says stands, open or not; checkDeclared holds the body to it. But not a
        // `@keeps` or `@lends` through an interface: the interface's callers are not told
        // (interfaceReached).
        for (auto &entry : functions)
        {
            if (declaresCallerFacts(entry.second))
            {
                auto keeps = declared(entry.second, OWN_DECLARED_KEEPS_ATTR_NAME);
                Facts said;
                said.borrows = declaredLends(entry.second);
                said.owned.assign(keeps.begin(), keeps.end());
                // a body here that returns a borrowing record of what `@lends` names: callers in this
                // module own the box (spec 32); checkDeclared holds the body to the parameters. A
                // declaration has no body: its `@record` says so.
                auto local = factsLocal.lookup(entry.second);
                said.record = !said.borrows.empty() &&
                              (entry.second.isDeclaration() ? entry.second->hasAttr(OWN_DECLARED_RECORD_ATTR_NAME)
                                                            : local.record && local.borrows == said.borrows);
                said.unboxed = said.record && !entry.second.isDeclaration() && local.unboxed;
                // the result is the object itself: a declaration's result type `this` says so, a body
                // here shows it (checkDeclared holds a body to its `: this`)
                said.itself = said.borrows.size() == 1 && !said.record &&
                              (entry.second.isDeclaration() ? entry.second->hasAttr(OWN_DECLARED_RETURNS_THIS_ATTR_NAME)
                                                            : local.itself && local.borrows == said.borrows);
                if (factsThroughInterface(entry.second, said))
                {
                    continue;
                }

                factsEffective[entry.second] = said;
            }
        }

        // the members of a family that disagree lose their facts: a call through the family is
        // given only what all of them say. A closed interface's slot is such a family too.
        auto demote = [&](llvm::ArrayRef<mlir::StringAttr> members) {
            llvm::SmallVector<Facts> seen;
            for (auto member : members)
            {
                auto funcOp = functions.lookup(member.getValue());
                seen.push_back(funcOp ? factsEffective.lookup(funcOp) : Facts());
            }

            auto changed = false;
            if (llvm::all_equal(seen))
            {
                return changed;
            }

            for (auto member : members)
            {
                if (auto funcOp = functions.lookup(member.getValue());
                    funcOp && !declaresCallerFacts(funcOp) && factsEffective.erase(funcOp))
                {
                    changed = true;
                }
            }

            return changed;
        };

        // a member a callback family demoted is never promoted again: each step only adds or only
        // takes away, so this settles
        demotedCallbacks.clear();
        for (auto changed = true; changed;)
        {
            changed = false;
            for (auto &[key, members] : families)
            {
                changed = demote(members) || changed;
            }

            changed = meetCallbacks(demotedCallbacks) || changed;

            for (auto &[key, family] : interfaceFamilies)
            {
                if (family.closed)
                {
                    changed = demote(family.members) || changed;
                }
            }

            for (auto &[key, family] : interfaceFieldFamilies)
            {
                if (family.closed)
                {
                    changed = demote(family.members) || changed;
                }
            }
        }
    }

    OwnershipSignaturePass::Facts OwnershipSignaturePass::callFacts(const Callees &callees)
    {
        if (!callees.known || callees.instanceOf || callees.funcs.empty())
        {
            return {};
        }

        // each member's facts as the call's arguments see them (a closure's body takes its box first)
        if (callees.viaCallback)
        {
            std::optional<Facts> agreed;
            for (auto [funcOp, shift] : llvm::zip(callees.funcs, callees.shifts))
            {
                Facts shifted;
                if (!shiftFacts(factsEffective.lookup(funcOp), shift, shifted) || (agreed && *agreed != shifted))
                {
                    return {};
                }

                agreed = shifted;
            }

            return agreed.value_or(Facts());
        }

        auto facts = factsEffective.lookup(callees.funcs.front());
        auto agree = llvm::all_of(callees.funcs, [&](mlir_ts::FuncOp callee) {
            return factsEffective.lookup(callee) == facts;
        });
        return agree ? facts : Facts();
    }

    void OwnershipSignaturePass::setFacts(mlir::Operation *op, const Facts &facts)
    {
        auto *context = op->getContext();
        op->removeAttr(OWN_RESULT_BORROWS_ATTR_NAME);
        op->removeAttr(OWN_RESULT_RECORD_ATTR_NAME);
        op->removeAttr(OWN_RESULT_RECORD_UNBOXED_ATTR_NAME);
        op->removeAttr(OWN_RESULT_ITSELF_ATTR_NAME);
        op->removeAttr(OWN_PARAMS_ATTR_NAME);
        if (!facts.borrows.empty())
        {
            op->setAttr(OWN_RESULT_BORROWS_ATTR_NAME, mlir::DenseI32ArrayAttr::get(context, facts.borrows));
            if (facts.itself && facts.borrows.size() == 1 && !facts.record)
            {
                op->setAttr(OWN_RESULT_ITSELF_ATTR_NAME, mlir::UnitAttr::get(context));
            }

            if (facts.record)
            {
                op->setAttr(OWN_RESULT_RECORD_ATTR_NAME, mlir::UnitAttr::get(context));
                if (facts.unboxed)
                {
                    op->setAttr(OWN_RESULT_RECORD_UNBOXED_ATTR_NAME, mlir::UnitAttr::get(context));
                }
            }
        }

        if (!facts.owned.empty())
        {
            op->setAttr(OWN_PARAMS_ATTR_NAME, mlir::DenseI32ArrayAttr::get(context, facts.owned));
        }
    }

    void OwnershipSignaturePass::pinFacts(const llvm::DenseMap<mlir::Operation *, Facts> &pinned)
    {
        for (auto &call : calls)
        {
            setFacts(call.op, pinned.lookup(call.op));
        }

        for (auto &[funcOp, local] : factsLocal)
        {
            auto effective = factsEffective.lookup(funcOp);
            setFacts(funcOp, effective);
            funcOp->removeAttr(OWN_FACTS_LOST_ATTR_NAME);
            funcOp->removeAttr(OWN_PARAMS_RELEASED_ATTR_NAME);
            // a signature's `@keeps` is checked against the body as it is (checkDeclared). But a rest
            // parameter it names is the array the callers move their elements into (spec 72): the body
            // takes the elements out, and the emptied array is released as an inferred keep's is
            auto ownFunc = mlir::dyn_cast<mlir_ts::FuncOp>(funcOp);
            auto funcType = ownFunc ? mlir::dyn_cast<mlir_ts::FunctionType>(ownFunc.getFunctionType()) : mlir_ts::FunctionType();
            auto restIndex = funcType && funcType.isVarArg() ? static_cast<int32_t>(funcType.getNumInputs()) - 1 : -1;
            llvm::SmallVector<int32_t> released;
            for (auto index : effective.owned)
            {
                if ((!declaresCallerFacts(funcOp) || index == restIndex) && !llvm::is_contained(takenParams.lookup(funcOp), index))
                {
                    released.push_back(index);
                }
            }

            if (!released.empty())
            {
                funcOp->setAttr(OWN_PARAMS_RELEASED_ATTR_NAME, mlir::DenseI32ArrayAttr::get(&getContext(), released));
            }

            if (effective.empty() && !local.empty())
            {
                // through an interface first: a marker would not help there (interfaceReached)
                auto *why = factsThroughInterface(funcOp, local) ? INTERFACE_REACHED_WHY : open.lookup(funcOp);
                if (!why && demotedCallbacks.contains(funcOp))
                {
                    why = "a function passed in its place disagrees";
                }

                funcOp->setAttr(OWN_FACTS_LOST_ATTR_NAME,
                                mlir::StringAttr::get(&getContext(), why ? why : "an override in its class family disagrees"));
            }
        }
    }

    bool OwnershipSignaturePass::ownsHeap(mlir::Location location, mlir::Type type)
    {
        return own_facts::ownsHeap(location, type);
    }

    void OwnershipSignaturePass::computeDrops()
    {
        callOf.clear();
        for (auto &call : calls)
        {
            callOf[call.op] = &call;
        }

        computeDropsRound(false);

        // A fill of an empty field (TSLANG 47) frees nothing, unless a call between the test and the
        // store may write the field. Each is decided once, by the facts that count every fill as a
        // drop: a callee that only fills may still write the field (it is no-drops once fills are
        // not drops). The verdict is pinned on the release, for the second round and the inference.
        getOperation().walk([&](mlir_ts::ReleaseSlotOp releaseSlotOp) {
            if (releaseSlotOp->getParentOfType<mlir::FunctionOpInterface>() && decideFill(releaseSlotOp))
            {
                releaseSlotOp->setAttr(OWN_FILL_ATTR_NAME, mlir::UnitAttr::get(&getContext()));
            }
            else
            {
                releaseSlotOp->removeAttr(OWN_FILL_ATTR_NAME);
            }
        });

        computeDropsRound(true);
    }

    void OwnershipSignaturePass::rejectWeakTargetsShared(mlir::ModuleOp module)
    {
        llvm::SmallVector<mlir_ts::WeakRefNewOp> weakRefs;
        llvm::SmallVector<mlir_ts::MarkSharedOp> shares;
        module.walk([&](mlir::Operation *op) {
            if (auto weakRefNewOp = mlir::dyn_cast<mlir_ts::WeakRefNewOp>(op))
            {
                weakRefs.push_back(weakRefNewOp);
            }
            else if (auto markSharedOp = mlir::dyn_cast<mlir_ts::MarkSharedOp>(op))
            {
                shares.push_back(markSharedOp);
            }
        });

        for (auto weakRefNewOp : weakRefs)
        {
            auto targetType = mlir::cast<mlir_ts::WeakRefType>(weakRefNewOp.getType()).getElementType();
            for (auto markSharedOp : shares)
            {
                if (!typeMayHold(module, markSharedOp->getOperand(0).getType(), targetType))
                {
                    continue;
                }

                llvm::SmallString<64> name;
                llvm::raw_svector_ostream nameStream(name);
                MLIRPrinter printer{};
                printer.printType<llvm::raw_svector_ostream>(nameStream, targetType);

                auto diag = weakRefNewOp->emitError("-mm=own cannot make a WeakRef to a '")
                            << name
                            << "' in a program that shares what may hold one with another thread: a weak count is not atomic";
                diag.attachNote(markSharedOp->getLoc()) << "shared with another thread here";
                signalPassFailure();
                break;
            }
        }
    }

    void OwnershipSignaturePass::computeNoThrow()
    {
        llvm::DenseMap<mlir::Operation *, llvm::SmallVector<Call *>> callsIn;
        for (auto &call : calls)
        {
            callsIn[call.op->getParentOfType<mlir_ts::FuncOp>()].push_back(&call);
        }

        // the bodies first: no `throw` anywhere in them (a failed cast is one, made in the body), and
        // no async code, whose await rethrows what the awaited body threw
        noThrow.clear();
        for (auto &entry : functions)
        {
            auto funcOp = entry.second;
            if (funcOp.isDeclaration())
            {
                continue;
            }

            auto throws = funcOp->walk([](mlir::Operation *op) {
                                    auto *dialect = op->getDialect();
                                    return mlir::isa<mlir_ts::ThrowOp, mlir_ts::ThrowCallOp, mlir_ts::ThrowUnwindOp>(op) ||
                                                   (dialect && dialect->getNamespace() == "async")
                                               ? mlir::WalkResult::interrupt()
                                               : mlir::WalkResult::advance();
                                }).wasInterrupted();
            if (!throws)
            {
                noThrow.insert(funcOp);
            }
        }

        // ...then the calls they make: to functions of this module, every one of which never throws
        auto callNeverThrows = [&](const Callees &callees) {
            return callees.known && !callees.imported && !callees.instanceOf && !callees.funcs.empty() &&
                   llvm::all_of(callees.funcs, [&](mlir_ts::FuncOp callee) { return noThrow.contains(callee); });
        };
        for (auto changed = true; changed;)
        {
            changed = false;
            for (auto &entry : functions)
            {
                auto funcOp = entry.second;
                if (noThrow.contains(funcOp) &&
                    llvm::any_of(callsIn[funcOp], [&](Call *call) { return !callNeverThrows(call->callees); }))
                {
                    noThrow.erase(funcOp);
                    changed = true;
                }
            }
        }

        for (auto &call : calls)
        {
            if (callNeverThrows(call.callees))
            {
                call.op->setAttr(OWN_NO_THROW_ATTR_NAME, mlir::UnitAttr::get(&getContext()));
            }
            else
            {
                call.op->removeAttr(OWN_NO_THROW_ATTR_NAME);
            }
        }
    }

    bool OwnershipSignaturePass::familyResultsFresh(int64_t index, mlir::StringAttr identifier)
    {
        for (auto member : familyOf(index, identifier))
        {
            auto funcOp = functions.lookup(member.getValue());
            if (!funcOp || funcOp->hasAttr(OWN_DECLARED_LENDS_ATTR_NAME) || funcOp->hasAttr(OWN_DECLARED_BORROWS_ATTR_NAME) ||
                funcOp->hasAttr(OWN_DECLARED_RECORD_ATTR_NAME) || funcOp->hasAttr(OWN_IMPLIED_LENDS_ATTR_NAME) ||
                (!funcOp.isDeclaration() && !returnsFresh(funcOp)))
            {
                return false;
            }
        }

        return true;
    }

    bool OwnershipSignaturePass::isFill(mlir_ts::ReleaseSlotOp releaseSlotOp)
    {
        return releaseSlotOp->hasAttr(OWN_FILL_ATTR_NAME);
    }

    bool OwnershipSignaturePass::decideFill(mlir_ts::ReleaseSlotOp releaseSlotOp)
    {
        auto field = releaseSlotOp.getSlot().getDefiningOp<mlir_ts::PropertyRefOp>();
        return field && fillsEmptyField(releaseSlotOp, [&](mlir::Operation *op) {
                   auto *call = callOf.lookup(op);
                   if (!call)
                   {
                       return true;
                   }

                   // another module's facts count its fills as no drops: they cannot clear a call.
                   // A C function declared `@nodrops` runs no function of this module, and given only
                   // numbers, strings and `@cpp` handles it reaches no field of an object here at all:
                   // `if (!this.cached) this.cached = new B(C_Get(this.handle))`, a lazy getter in
                   // native_ts_graphics' core/nri.ts (TSLANG 63)
                   if (call->callees.imported)
                   {
                       auto funcOp = functions.lookup(call->callees.importedName);
                       return !funcOp || !foreignOnlyDeclaration(funcOp);
                   }

                   if (!callMayDrop(call->callees))
                   {
                       return false;
                   }

                   auto drops = calleesFieldDrops(call->callees);
                   return !drops || drops->any || llvm::any_of(drops->places, [&](auto &place) {
                              return place.first == OWN_INTERFACE_FIELD_POSITION ||
                                     (place.first == field.getPosition() && place.second == field.getType());
                          });
               });
    }

    void OwnershipSignaturePass::computeDropsRound(bool fills)
    {
        llvm::DenseMap<mlir::Operation *, llvm::SmallVector<Call *>> callsIn;
        for (auto &call : calls)
        {
            callsIn[call.op->getParentOfType<mlir_ts::FuncOp>()].push_back(&call);
        }

        // the bodies first, all against the previous round's facts
        llvm::DenseSet<mlir::Operation *> bodyNoDrops;
        llvm::DenseMap<mlir::Operation *, FieldDrops> bodyFieldDrops;
        for (auto &entry : functions)
        {
            auto funcOp = entry.second;
            if (funcOp.isDeclaration())
            {
                continue;
            }

            if (!dropsInBody(funcOp, fills))
            {
                bodyNoDrops.insert(funcOp);
            }
            else
            {
                bodyFieldDrops[funcOp] = fieldDropsInBody(funcOp, fills);
            }
        }

        noDrops = std::move(bodyNoDrops);
        fieldDrops.clear();

        for (auto changed = true; changed;)
        {
            changed = false;
            for (auto &entry : functions)
            {
                auto funcOp = entry.second;
                if (!noDrops.contains(funcOp))
                {
                    continue;
                }

                for (auto *call : callsIn[funcOp])
                {
                    if (!callMayDrop(call->callees))
                    {
                        continue;
                    }

                    noDrops.erase(funcOp);
                    changed = true;
                    break;
                }
            }
        }

        // What the others destroy, when it is only places they overwrite: their bodies' stores and
        // their callees' together, to a fixed point (a recursive call adds nothing new)
        for (auto &entry : functions)
        {
            auto funcOp = entry.second;
            if (!funcOp.isDeclaration() && !noDrops.contains(funcOp))
            {
                auto found = bodyFieldDrops.find(funcOp);
                fieldDrops[funcOp] = found != bodyFieldDrops.end() ? found->second : FieldDrops();
            }
        }

        for (auto changed = true; changed;)
        {
            changed = false;
            for (auto &entry : functions)
            {
                auto funcOp = entry.second;
                auto found = fieldDrops.find(funcOp);
                if (found == fieldDrops.end() || found->second.any)
                {
                    continue;
                }

                for (auto *call : callsIn[funcOp])
                {
                    if (!callMayDrop(call->callees))
                    {
                        continue;
                    }

                    auto callee = calleesFieldDrops(call->callees);
                    auto &mine = fieldDrops[funcOp];
                    if (!callee || callee->any)
                    {
                        mine.any = true;
                        changed = true;
                        break;
                    }

                    // a place the callee's argument owns is this function's parameter's when the argument is one
                    // (or is reached from one as ownerParam reaches it); else it is owned some other way
                    auto aligned = argumentsAlignWithParams(call->op, call->callees.viaCallback);
                    auto args = callArgs(call->op);
                    for (auto [place, owner] : llvm::zip(callee->places, callee->owners))
                    {
                        // a place of an object made here holds nothing this function's callers have
                        if (aligned && owner >= 0 && static_cast<size_t>(owner) < args.size() && madeHere(args[owner], funcOp))
                        {
                            continue;
                        }

                        auto mineOwner = aligned && owner >= 0 && static_cast<size_t>(owner) < args.size()
                                             ? ownerParam(args[owner], funcOp, /*insideChanged=*/true)
                                             : -1;
                        changed |= mine.add(place, mineOwner);
                    }
                }
            }
        }
    }

    void OwnershipSignaturePass::markOverridableReached()
    {
        // `this.props[i].tick(t)`, where `Prop.tick` calls `setNodeTransform`, which calls
        // `world.setGameNodeTransform(...)` of an exported class: the outer call may destroy anything
        // only because of that virtual call, which the error at a borrow it ends names (TSLANG 60). The
        // way to it is found breadth first, through calls that may destroy anything, so it is a short one.
        llvm::DenseMap<mlir::Operation *, llvm::SmallVector<Call *>> callsIn;
        for (auto &call : calls)
        {
            callsIn[call.op->getParentOfType<mlir_ts::FuncOp>()].push_back(&call);
        }

        auto mayDropAnything = [&](const Call &call) {
            if (call.op->hasAttr(OWN_NO_DROPS_ATTR_NAME) || call.op->hasAttr(OWN_DROP_FIELDS_ATTR_NAME))
            {
                return false;
            }

            auto drops = calleesFieldDrops(call.callees);
            return !drops || drops->any;
        };

        // a function's own body destroying anything (not through its calls), as the last round decided it
        llvm::DenseMap<mlir::Operation *, bool> bodyAny;
        auto bodyDropsAnything = [&](mlir_ts::FuncOp funcOp) {
            auto [found, inserted] = bodyAny.try_emplace(funcOp, false);
            if (inserted)
            {
                found->second = dropsInBody(funcOp, true) && fieldDropsInBody(funcOp, true).any;
            }

            return found->second;
        };

        // The one overridable method the functions reach, and the way to it - none when they reach
        // another, or something else that may destroy anything (an unknown or imported callee, a body
        // that does): `@final` on that method would not be enough then
        auto explain = [&](llvm::ArrayRef<mlir_ts::FuncOp> funcs) -> mlir::ArrayAttr {
            // each function reached, and the one that called it on the way
            llvm::DenseMap<mlir::Operation *, mlir::Operation *> reachedFrom;
            llvm::SmallVector<mlir_ts::FuncOp> work;
            for (auto callee : funcs)
            {
                if (reachedFrom.try_emplace(callee, nullptr).second)
                {
                    work.push_back(callee);
                }
            }

            mlir::FlatSymbolRefAttr method;
            mlir::Operation *holder = nullptr;
            for (auto next = 0u; next < work.size(); ++next)
            {
                auto funcOp = work[next];
                if (work.size() > 256 || bodyDropsAnything(funcOp))
                {
                    return {};
                }

                for (auto *inner : callsIn[funcOp])
                {
                    if (auto overridable = inner->op->getAttrOfType<mlir::FlatSymbolRefAttr>(OWN_OVERRIDABLE_CALL_ATTR_NAME))
                    {
                        if (method && method != overridable)
                        {
                            return {};
                        }

                        if (!method)
                        {
                            method = overridable;
                            holder = funcOp;
                        }

                        continue;
                    }

                    if (!mayDropAnything(*inner))
                    {
                        continue;
                    }

                    if (!inner->callees.known || inner->callees.imported || inner->callees.funcs.empty())
                    {
                        return {};
                    }

                    for (auto callee : inner->callees.funcs)
                    {
                        if (reachedFrom.try_emplace(callee, funcOp).second)
                        {
                            work.push_back(callee);
                        }
                    }
                }
            }

            if (!method)
            {
                return {};
            }

            llvm::SmallVector<mlir::Attribute> path;
            for (auto *at = holder; at; at = reachedFrom.lookup(at))
            {
                path.push_back(mlir::cast<mlir_ts::FuncOp>(at).getSymNameAttr());
            }

            std::reverse(path.begin(), path.end());
            path.insert(path.begin(), method);
            return mlir::ArrayAttr::get(&getContext(), path);
        };

        // a call of one function, the common case, explained once per function
        llvm::DenseMap<mlir::Operation *, mlir::ArrayAttr> explained;
        for (auto &call : calls)
        {
            if (!call.callees.known || call.callees.imported || call.callees.funcs.empty() ||
                call.op->hasAttr(OWN_OVERRIDABLE_CALL_ATTR_NAME) || !mayDropAnything(call))
            {
                continue;
            }

            mlir::ArrayAttr via;
            if (call.callees.funcs.size() == 1)
            {
                auto [found, inserted] = explained.try_emplace(call.callees.funcs.front(), mlir::ArrayAttr());
                if (inserted)
                {
                    found->second = explain(call.callees.funcs);
                }

                via = found->second;
            }
            else
            {
                via = explain(call.callees.funcs);
            }

            if (via)
            {
                call.op->setAttr(OWN_OVERRIDABLE_VIA_ATTR_NAME, via);
            }
        }
    }

    std::optional<OwnershipSignaturePass::FieldDrops> OwnershipSignaturePass::calleesFieldDrops(const Callees &callees)
    {
        if (!callees.known || callees.instanceOf)
        {
            return std::nullopt;
        }

        FieldDrops all;
        if (callees.imported)
        {
            // what the imported module's object says (--own-facts), or nothing at all
            for (auto place : callees.importedPlaces)
            {
                all.add(place);
            }

            return all;
        }

        for (auto [index, callee] : llvm::enumerate(callees.funcs))
        {
            if (noDrops.contains(callee))
            {
                continue;
            }

            auto found = fieldDrops.find(callee);
            if (found == fieldDrops.end() || found->second.any)
            {
                return std::nullopt;
            }

            // each owner as the call's argument: a callback family's member's parameter k is the call's argument
            // k - its shift (a closure's body takes its capture box first)
            auto shift = callees.viaCallback && index < callees.shifts.size() ? callees.shifts[index] : 0;
            for (auto [place, owner] : llvm::zip(found->second.places, found->second.owners))
            {
                all.add(place, owner >= 0 && owner - shift >= 0 ? owner - shift : -1);
            }
        }

        return all;
    }

    OwnershipSignaturePass::FieldDrops OwnershipSignaturePass::fieldDropsInBody(mlir_ts::FuncOp funcOp, bool fills)
    {
        // what dropsInBody counts, with the overwrites of places it can name kept apart
        FieldDrops drops;
        funcOp.walk([&](mlir::Operation *op) {
            if (drops.any)
            {
                return;
            }

            if (mlir::isa<mlir_ts::ReleaseOp>(op) && touchesHandle(op))
            {
                drops.any = true;
            }
            else if (auto releaseSlotOp = mlir::dyn_cast<mlir_ts::ReleaseSlotOp>(op))
            {
                auto slot = releaseSlotOp.getSlot();
                auto slotType = mlir::dyn_cast<mlir_ts::RefType>(slot.getType());
                if (releasesNothing(releaseSlotOp) || (fills && isFill(releaseSlotOp)))
                {
                    return;
                }

                if (isLoadedCell(slot) && slotType && isCopyableString(slotType.getElementType()))
                {
                    // a captured string assigned (`each += v` in a callback): it frees that string only
                    drops.add({OWN_CAPTURED_STRING_POSITION, slot.getType()});
                }
                else if (isLoadedCell(slot) && !touchesHandle(op))
                {
                    // a captured object assigned (`last = new Box()` in a callback): it frees what the cell
                    // owns, which only a read through a captured variable may borrow
                    drops.add({OWN_CAPTURED_CELL_POSITION, slot.getType()});
                }
                else if (slot.getDefiningOp<mlir_ts::AddressOfOp>() || isLoadedCell(slot) ||
                    (!isPlace(slot) && touchesHandle(op) && !isCountedReadSlot(slot)))
                {
                    drops.any = true;
                }
                else if (isPlace(slot) && reachesOutside(slot, funcOp))
                {
                    // `x.value = ...` of a handle may replace the payload any other handle reads
                    if (slot.getDefiningOp<mlir_ts::SharedValueRefOp>())
                    {
                        drops.any = true;
                        return;
                    }

                    int64_t position = -1;
                    if (auto propertyRefOp = slot.getDefiningOp<mlir_ts::PropertyRefOp>())
                    {
                        position = propertyRefOp.getPosition();
                    }
                    else if (isInterfaceField(slot))
                    {
                        position = OWN_INTERFACE_FIELD_POSITION;
                    }

                    // (`this.x = v`: under `this`)
                    drops.add({position, slot.getType()},
                              position == OWN_INTERFACE_FIELD_POSITION ? -1 : ownerParam(slot, funcOp, /*insideChanged=*/false));
                }
            }
            else if (auto swapOp = mlir::dyn_cast<mlir_ts::SwapOp>(op))
            {
                // an exchange drops a borrow of either place as an overwrite of it would (spec 42)
                for (auto place : swapOp->getOperands())
                {
                    if (reachesOutside(place, funcOp))
                    {
                        drops.add({swappedPosition(place), place.getType()});
                    }
                }
            }
            else if (mlir::isa<mlir_ts::ArrayPopOp, mlir_ts::ArrayShiftOp, mlir_ts::ArraySpliceOp, mlir_ts::SetLengthOfOp>(op) &&
                     !growsOnly(op))
            {
                if (reachesOutside(op->getOperand(0), funcOp))
                {
                    // the elements of an array of its element type, as a store into one drops them
                    auto refType = mlir::dyn_cast<mlir_ts::RefType>(op->getOperand(0).getType());
                    auto arrayType = refType ? mlir::dyn_cast<mlir_ts::ArrayType>(refType.getElementType()) : mlir_ts::ArrayType();
                    if (arrayType)
                    {
                        // (`this._keys.splice(i, 1)`: under `this`)
                        drops.add({-1, mlir_ts::RefType::get(arrayType.getElementType())},
                                  ownerParam(op->getOperand(0), funcOp, /*insideChanged=*/true));
                    }
                    else
                    {
                        drops.any = true;
                    }
                }
            }
            else if (mlir::isa<mlir_ts::ArrayPushOp, mlir_ts::ArrayUnshiftOp>(op) || growsOnly(op))
            {
                drops.any |= reachesSharedValue(op->getOperand(0));
            }
            else if (mlir::isa<mlir_ts::DeleteOp>(op))
            {
                drops.any = true;
            }
        });

        return drops;
    }

    bool OwnershipSignaturePass::callMayDrop(const Callees &callees)
    {
        if (!callees.known)
        {
            return true;
        }

        if (callees.imported)
        {
            return callees.importedOverwrites;
        }

        return !callees.instanceOf &&
               llvm::any_of(callees.funcs, [&](mlir_ts::FuncOp callee) { return !noDrops.contains(callee); });
    }

    // Its one result is the caller's to own, as a call of it here is (isFresh): its body borrows no
    // argument for it, nor says one does, and it does not hand back a record that borrows
    bool OwnershipSignaturePass::returnsFresh(mlir_ts::FuncOp funcOp)
    {
        if (funcOp.getFunctionType().getNumResults() != 1 || funcOp->hasAttr(OWN_DECLARED_LENDS_ATTR_NAME) ||
            funcOp->hasAttr(OWN_DECLARED_BORROWS_ATTR_NAME) || funcOp->hasAttr(OWN_DECLARED_RECORD_ATTR_NAME))
        {
            return false;
        }

        // the effective facts are kept only when not empty
        auto local = factsLocal.find(funcOp);
        auto effective = factsEffective.lookup(funcOp);
        return local != factsLocal.end() && local->second.borrows.empty() && !local->second.record &&
               effective.borrows.empty() && !effective.record;
    }

    void OwnershipSignaturePass::exportNoDrops()
    {
        llvm::SmallVector<llvm::StringRef> names;
        for (auto &entry : functions)
        {
            auto funcOp = entry.second;
            if (!funcOp.isDeclaration() && !funcOp.isPrivate() && noDrops.contains(funcOp))
            {
                names.push_back(funcOp.getSymName());
            }
        }

        llvm::sort(names);
        std::string text;
        for (auto name : names)
        {
            text += name.str();
            text += '\n';
        }

        // The same names, and the places each other exported function overwrites when that is all
        // it destroys, as the char array an importer reads from this module's object
        // (SHARED_LIB_OWN_OBJECT_FACTS): `<name>` or `<name>\t<position> <slot type>...`, and a last
        // field OWN_OBJECT_FACTS_FRESH_RESULT when its result is its caller's to own. Written
        // whatever the module exports, as a static library is compiled with --export=none.
        llvm::SmallVector<std::pair<llvm::StringRef, std::string>> lines;
        for (auto &entry : functions)
        {
            auto funcOp = entry.second;
            if (funcOp.isDeclaration() || funcOp.isPrivate())
            {
                continue;
            }

            auto found = fieldDrops.find(funcOp);
            auto overwrites = !noDrops.contains(funcOp) && found != fieldDrops.end() && !found->second.any &&
                              !found->second.places.empty();
            if (!noDrops.contains(funcOp) && !overwrites)
            {
                continue;
            }

            std::string line;
            llvm::raw_string_ostream os(line);
            if (overwrites)
            {
                for (auto [position, type] : found->second.places)
                {
                    os << '\t' << position << ' ' << type;
                }
            }

            if (returnsFresh(funcOp))
            {
                os << '\t' << OWN_OBJECT_FACTS_FRESH_RESULT;
            }

            lines.push_back({funcOp.getSymName(), line});
        }

        llvm::sort(lines);
        std::string objectText;
        for (auto &[functionName, fields] : lines)
        {
            objectText += functionName.str() + fields + "\n";
        }

        mlir::Builder attrBuilder(&getContext());
        if (!objectFactsSymbol.empty() && !objectText.empty())
        {
            getOperation()->setAttr(SHARED_LIB_OWN_OBJECT_FACTS_ATTR_NAME,
                                    attrBuilder.getArrayAttr({attrBuilder.getStringAttr(objectFactsSymbol),
                                                              attrBuilder.getStringAttr(objectText)}));
        }

        // The names, as the shared library's exported SHARED_LIB_OWN_FACTS, beside its memory model
        // marker: only a module that has one, exporting its declarations, is imported as -shared
        auto markerPrefix = std::string(SHARED_LIB_MEMORY_MODEL) + "own_";
        mlir_ts::GlobalOp marker;
        for (auto &entry : globals)
        {
            if (entry.getKey().starts_with(markerPrefix))
            {
                marker = entry.second;
                break;
            }
        }

        if (!marker)
        {
            return;
        }

        auto suffix = marker.getSymName().drop_front(markerPrefix.size());
        auto name = (llvm::Twine(SHARED_LIB_OWN_FACTS) + suffix).str();

        mlir::OpBuilder builder(marker);
        builder.setInsertionPointAfter(marker);
        auto factsOp = mlir::cast<mlir_ts::GlobalOp>(builder.clone(*marker.getOperation()));
        factsOp.setSymName(name);
        factsOp.getInitializerRegion().walk([&](mlir_ts::ConstantOp constantOp) {
            if (mlir::isa<mlir::StringAttr>(constantOp.getValue()))
            {
                constantOp.setValueAttr(builder.getStringAttr(text));
            }
        });
    }

    bool OwnershipSignaturePass::isCountedReadSlot(mlir::Value slot)
    {
        // `for (const v of this)` over handles: the local is declared from a read, which rc counts
        // (`ts.RetainSlot`), and is never assigned. Where the read came from still holds a count of
        // its own while the local lives - a store into it, a pop, a call that does either is a
        // drop of its own - so the local's release is never the last (spec 51).
        auto varOp = slot.getDefiningOp<mlir_ts::VariableOp>();
        if (!varOp || varOp.getCaptured().value_or(false) || !varOp.getInitializer() ||
            !varOp.getInitializer().getDefiningOp<mlir_ts::LoadOp>())
        {
            return false;
        }

        auto counted = false;
        for (auto *user : slot.getUsers())
        {
            if (mlir::isa<mlir_ts::RetainSlotOp>(user))
            {
                counted = true;
            }
            else if (!mlir::isa<mlir_ts::LoadOp, mlir_ts::ReleaseSlotOp, mlir_ts::DebugVariableOp>(user))
            {
                return false;
            }
        }

        return counted;
    }

    bool OwnershipSignaturePass::dropsInBody(mlir_ts::FuncOp funcOp, bool fills)
    {
        auto drops = false;
        funcOp.walk([&](mlir::Operation *op) {
            if (drops)
            {
                return;
            }

            // A handle given back may be the last one, and its payload goes with it, and with that
            // anything the payload owns (spec 23.3). One held by a field or an element is a place's
            // overwrite, judged below like any other: a constructor filling its own `this` is not
            // a drop.
            if (mlir::isa<mlir_ts::ReleaseOp>(op) && touchesHandle(op))
            {
                // a WeakRef's frees at most a tombstone (spec 84)
                drops = !MLIRTypeHelper::isWeakRefType(op->getOperand(0).getType());
            }
            else if (auto releaseSlotOp = mlir::dyn_cast<mlir_ts::ReleaseSlotOp>(op))
            {
                // a captured variable assigned through its cell, by a closure's body: the variable
                // is its declaring function's
                auto slot = releaseSlotOp.getSlot();
                drops = !releasesNothing(releaseSlotOp) && !(fills && isFill(releaseSlotOp)) &&
                        (slot.getDefiningOp<mlir_ts::AddressOfOp>() || (isPlace(slot) && reachesOutside(slot, funcOp)) ||
                         isLoadedCell(slot) || (!isPlace(slot) && touchesHandle(op) && !isCountedReadSlot(slot)));
            }
            else if (mlir::isa<mlir_ts::ArrayPopOp, mlir_ts::ArrayShiftOp, mlir_ts::ArraySpliceOp, mlir_ts::SetLengthOfOp>(op) &&
                     !growsOnly(op))
            {
                drops = reachesOutside(op->getOperand(0), funcOp);
            }
            else if (auto swapOp = mlir::dyn_cast<mlir_ts::SwapOp>(op))
            {
                // the caller's borrow of a place it exchanges reads another place's value (spec 42)
                drops = reachesOutside(swapOp.getFirst(), funcOp) || reachesOutside(swapOp.getSecond(), funcOp);
            }
            else if (mlir::isa<mlir_ts::ArrayPushOp, mlir_ts::ArrayUnshiftOp>(op) || growsOnly(op))
            {
                // through a handle, it may move the elements another handle's borrow reads (spec 23.3)
                drops = reachesSharedValue(op->getOperand(0));
            }
            else if (mlir::isa<mlir_ts::DeleteOp>(op))
            {
                drops = true;
            }
        });

        return drops;
    }

    bool OwnershipSignaturePass::reachesOutside(mlir::Value ref, mlir_ts::FuncOp funcOp)
    {
        auto isConstructor = funcOp.getSymName().ends_with(".constructor");
        llvm::SmallVector<mlir::Value> work{ref};
        llvm::DenseSet<mlir::Value> seen;
        while (!work.empty())
        {
            auto value = rootOf(work.pop_back_val());
            if (!seen.insert(value).second)
            {
                continue;
            }

            if (auto object = boundThis(value))
            {
                work.push_back(object);
                continue;
            }

            if (auto place = placeBase(value))
            {
                // a handle's payload: any other handle may reach it (spec 23.3)
                if (place.kind == PlaceKind::SharedValue)
                {
                    return true;
                }

                work.push_back(place.base);
                continue;
            }

            if (auto argument = mlir::dyn_cast<mlir::BlockArgument>(value))
            {
                llvm::SmallVector<mlir::Value> merged;
                if (mergedInto(argument, merged))
                {
                    work.append(merged.begin(), merged.end());
                    continue;
                }

                if (isConstructor && argument.getOwner()->isEntryBlock() && argument.getArgNumber() == 0)
                {
                    continue; // the object being built
                }

                return true;
            }

            if (auto varOp = value.getDefiningOp<mlir_ts::VariableOp>())
            {
                if (holdsOnlyFresh(varOp))
                {
                    continue;
                }

                // a parameter's slot: what the caller passed
                if (auto init = varOp.getInitializer())
                {
                    if (auto argument = mlir::dyn_cast<mlir::BlockArgument>(init);
                        argument && isConstructor && argument.getOwner()->isEntryBlock() && argument.getArgNumber() == 0 &&
                        !isAssigned(varOp))
                    {
                        continue;
                    }
                }

                return true;
            }

            if (auto loadOp = value.getDefiningOp<mlir_ts::LoadOp>())
            {
                work.push_back(loadOp.getReference());
                continue;
            }

            if (isFresh(value))
            {
                continue;
            }

            return true;
        }

        return false;
    }

    bool OwnershipSignaturePass::madeHere(mlir::Value value, mlir_ts::FuncOp funcOp)
    {
        auto root = rootOf(value);
        if (own_facts::isFresh(root))
        {
            return true;
        }

        auto loadOp = root.getDefiningOp<mlir_ts::LoadOp>();
        auto varOp = loadOp ? loadOp.getReference().getDefiningOp<mlir_ts::VariableOp>() : mlir_ts::VariableOp();
        if (!varOp || varOp->getParentOfType<mlir_ts::FuncOp>() != funcOp || varOp.getCaptured().value_or(false) ||
            !varOp.getInitializer() || !own_facts::isFresh(rootOf(varOp.getInitializer())))
        {
            return false;
        }

        // nothing assigns it, or gives its slot away
        return llvm::all_of(varOp.getResult().getUsers(), [&](mlir::Operation *user) {
            return mlir::isa<mlir_ts::LoadOp, mlir_ts::RetainSlotOp, mlir_ts::ReleaseSlotOp, mlir_ts::DebugVariableOp>(user);
        });
    }

    int32_t OwnershipSignaturePass::ownerParam(mlir::Value value, mlir_ts::FuncOp funcOp, bool insideChanged)
    {
        auto isParameter = [&](mlir::BlockArgument argument) {
            return argument.getOwner()->isEntryBlock() && argument.getOwner()->getParent() == &funcOp.getBody();
        };

        for (auto depth = 0; depth < 32 && value; depth++)
        {
            value = rootOf(value);
            if (auto propertyRefOp = value.getDefiningOp<mlir_ts::PropertyRefOp>())
            {
                // the part of a subclass's object that is its base class's is the object itself, not a field
                auto object = propertyRefOp.getObjectRef();
                auto resultRef = mlir::dyn_cast<mlir_ts::RefType>(propertyRefOp.getType());
                if (resultRef && mlir::isa<mlir_ts::ClassStorageType>(resultRef.getElementType()))
                {
                    value = object;
                    continue;
                }

                // a field of an object of a class (not of a value class's storage, which is copied as a tuple is:
                // a field read off the base part is of the object itself)
                auto objectRef = mlir::dyn_cast<mlir_ts::RefType>(object.getType());
                auto basePart = objectRef && mlir::isa<mlir_ts::ClassStorageType>(objectRef.getElementType()) &&
                                object.getDefiningOp<mlir_ts::PropertyRefOp>();
                auto storage = fieldOwnerStorage(propertyRefOp);
                if (!storage || (!mlir::isa<mlir_ts::ClassType>(object.getType()) && !basePart) ||
                    backReferenceFieldSet.contains({storage, propertyRefOp.getPosition()}) ||
                    sharedFieldSet.contains({storage, propertyRefOp.getPosition()}))
                {
                    return -1;
                }

                auto fieldRef = mlir::dyn_cast<mlir_ts::RefType>(propertyRefOp.getType());
                if (!fieldRef || mlir::isa<mlir_ts::WeakRefType>(fieldRef.getElementType()) ||
                    MLIRTypeHelper::isSharedHandleType(fieldRef.getElementType()))
                {
                    return -1;
                }

                value = object;
                continue;
            }

            // the element a store replaces (`this._values[i] = v`), of an array: only the place itself - what is under
            // an element is not apart, as an array may be given any value as an element (`this.lists.push(G)`)
            if (auto elementRefOp = value.getDefiningOp<mlir_ts::ElementRefOp>())
            {
                if (depth > 0 || insideChanged)
                {
                    return -1;
                }

                value = elementRefOp.getArray();
                continue;
            }

            if (auto loadOp = value.getDefiningOp<mlir_ts::LoadOp>())
            {
                value = loadOp.getReference();
                continue;
            }

            // a parameter's slot the body does not assign or capture: what the caller passed
            if (auto varOp = value.getDefiningOp<mlir_ts::VariableOp>())
            {
                auto argument = varOp.getInitializer() ? mlir::dyn_cast<mlir::BlockArgument>(varOp.getInitializer())
                                                       : mlir::BlockArgument();
                if (argument && isParameter(argument) && !isAssigned(varOp) && !varOp.getCaptured().value_or(false))
                {
                    return static_cast<int32_t>(argument.getArgNumber());
                }

                return -1;
            }

            if (auto paramOp = value.getDefiningOp<mlir_ts::ParamOp>())
            {
                auto argument = mlir::dyn_cast<mlir::BlockArgument>(paramOp.getArgValue());
                auto assigned = llvm::any_of(paramOp.getResult().getUsers(), [&](mlir::Operation *user) {
                    auto storeOp = mlir::dyn_cast<mlir_ts::StoreOp>(user);
                    return storeOp && storeOp.getReference() == paramOp.getResult();
                });
                if (argument && isParameter(argument) && !assigned && !paramOp.getCaptured().value_or(false))
                {
                    return static_cast<int32_t>(argument.getArgNumber());
                }

                return -1;
            }

            if (auto argument = mlir::dyn_cast<mlir::BlockArgument>(value); argument && isParameter(argument))
            {
                return static_cast<int32_t>(argument.getArgNumber());
            }

            return -1;
        }

        return -1;
    }

    bool OwnershipSignaturePass::isAssigned(mlir_ts::VariableOp varOp)
    {
        return llvm::any_of(varOp.getResult().getUsers(), [&](mlir::Operation *user) {
            auto storeOp = mlir::dyn_cast<mlir_ts::StoreOp>(user);
            return storeOp && storeOp.getReference() == varOp.getResult();
        });
    }

    // `let m = p; if (c) m = this.b;` (TSLANG 56), a parameter the body gives a field's value (`p = q.items[0]`):
    // every value the local is given besides the parameter is a borrow - a read of a field or an element,
    // or of a parameter - and nothing keeps what is read out of it. Inference makes such a local a borrower
    // of each (borrowsOnly), so it does not take the parameter either: the function borrows it, and its
    // callers give it back. Kept, nothing did - the argument leaked on every call - and a move out of the
    // local on another path freed the field's value (fuzz 949146). A swap (`[data, this.data] = [this.data,
    // data]`, spec 77) gives the field the local's value, which is a keep.
    bool OwnershipSignaturePass::givenOnlyBorrows(mlir_ts::VariableOp varOp)
    {
        // a local whose reads go into one being asked about already: as if kept
        static thread_local llvm::SmallPtrSet<mlir::Operation *, 8> borrowCheckVisiting;
        if (!borrowCheckVisiting.insert(varOp).second)
        {
            return false;
        }

        auto visiting = llvm::make_scope_exit([&]() { borrowCheckVisiting.erase(varOp); });

        auto assigned = false;
        for (auto *user : varOp.getResult().getUsers())
        {
            auto storeOp = mlir::dyn_cast<mlir_ts::StoreOp>(user);
            if (!storeOp || storeOp.getReference() != varOp.getResult())
            {
                // reads, rc's counts and the debugger's record of it; anything else (its address given
                // away) may give it a value no store here shows
                if (!mlir::isa<mlir_ts::LoadOp, mlir_ts::RetainSlotOp, mlir_ts::ReleaseSlotOp, mlir_ts::DebugVariableOp>(user))
                {
                    return false;
                }

                continue;
            }

            auto loadOp = rootOf(storeOp.getValue()).getDefiningOp<mlir_ts::LoadOp>();
            if (!loadOp || isHandle(loadOp.getResult()))
            {
                return false;
            }

            auto ref = loadOp.getReference();
            auto slot = ref.getDefiningOp<mlir_ts::VariableOp>();
            auto index = -1;
            if (!isPlace(ref) && !(slot && isParameterSlot(slot, index)))
            {
                return false;
            }

            assigned = true;
        }

        return assigned && !readKept(varOp.getResult(), varOp.getResult(), 0);
    }

    // May `later` run after `first`? By the blocks of the region holding both: later in `first`'s
    // block, or in a block a branch from it reaches (its own block again, round a loop). Ops in
    // different regions, or nested deeper than one another: assumed so.
    static bool mayRunAfter(mlir::Operation *first, mlir::Operation *later)
    {
        auto *region = first->getParentRegion();
        auto *laterOp = region ? region->findAncestorOpInRegion(*later) : nullptr;
        if (!laterOp)
        {
            return true;
        }

        auto *start = first->getBlock();
        if (laterOp->getBlock() == start && first->isBeforeInBlock(laterOp))
        {
            return true;
        }

        llvm::SmallPtrSet<mlir::Block *, 16> seen;
        llvm::SmallVector<mlir::Block *> work(start->getSuccessors().begin(), start->getSuccessors().end());
        while (!work.empty())
        {
            auto *block = work.pop_back_val();
            if (block == laterOp->getBlock())
            {
                return true;
            }

            if (seen.insert(block).second)
            {
                work.append(block->getSuccessors().begin(), block->getSuccessors().end());
            }
        }

        return false;
    }

    // Is a value read out of the slot kept - by a store into a place, a push, a call that keeps it, a
    // return? A local declared from it takes it where the owner (the parameter's local) may be given
    // another value after the read (`const n = p; p = G.n;`, fuzz 952414: inference moves the value
    // into the local and frees it with it); elsewhere (`if (c) p = q.items[0]; else { const v = p; }`,
    // fuzz 949146) the local holds what the owner does, and what its reads do decides.
    bool OwnershipSignaturePass::readKept(mlir::Value slot, mlir::Value owner, int depth)
    {
        if (depth > 4)
        {
            return true;
        }

        auto kept = false;
        for (auto *user : slot.getUsers())
        {
            auto loadOp = mlir::dyn_cast<mlir_ts::LoadOp>(user);
            if (!loadOp || kept)
            {
                continue;
            }

            forEachUse(loadOp.getResult(), [&](mlir::Operation *use, mlir::Value used) {
                if (kept)
                {
                    return;
                }

                if (auto local = mlir::dyn_cast<mlir_ts::VariableOp>(use))
                {
                    auto ownerGivenAfter = llvm::any_of(owner.getUsers(), [&](mlir::Operation *ownerUser) {
                        auto storeOp = mlir::dyn_cast<mlir_ts::StoreOp>(ownerUser);
                        return storeOp && storeOp.getReference() == owner && mayRunAfter(loadOp, storeOp);
                    });
                    kept = local.getInitializer() != used || local.getCaptured().value_or(false) || ownerGivenAfter ||
                           readKept(local.getResult(), owner, depth + 1);
                    return;
                }

                if (keeps(use, used))
                {
                    kept = true;
                    return;
                }

                // what only reads it is no keep: a call that does not keep it, its fields' or elements'
                // addresses, its method looked up for a call, a store into it, or a number, a boolean
                // made of it. Anything else - an array literal it is an element of (fuzz 958275: the
                // array freed the caller's object), a tuple, a box - may own it, as the parameter did
                // before.
                auto storeOp = mlir::dyn_cast<mlir_ts::StoreOp>(use);
                auto reads = isCall(use) ||
                             mlir::isa<mlir_ts::LoadOp, mlir_ts::PropertyRefOp, mlir_ts::ElementRefOp, mlir_ts::LengthOfOp,
                                       mlir_ts::ThisSymbolRefOp, mlir_ts::ThisVirtualSymbolRefOp, mlir_ts::GetThisOp,
                                       mlir_ts::GetMethodOp, mlir_ts::DebugVariableOp>(use) ||
                             (storeOp && storeOp.getValue() != used) ||
                             (use->getNumResults() > 0 &&
                              llvm::all_of(use->getResults(), [&](mlir::Value result) { return holdsNoBlock(result); }));
                kept = !reads;
            });
        }

        return kept;
    }

    bool OwnershipSignaturePass::walksDownOnly(mlir_ts::VariableOp varOp)
    {
        auto assigned = false;
        for (auto *user : varOp.getResult().getUsers())
        {
            // reads, rc's counts and the debugger's record of it; anything else (a swap, its address
            // given away) may give it a value no store here shows
            auto storeOp = mlir::dyn_cast<mlir_ts::StoreOp>(user);
            if (!storeOp || storeOp.getReference() != varOp.getResult())
            {
                if (!mlir::isa<mlir_ts::LoadOp, mlir_ts::RetainSlotOp, mlir_ts::ReleaseSlotOp, mlir_ts::DebugVariableOp>(user))
                {
                    return false;
                }

                continue;
            }

            // the value, read out of a field of a class object, that read out of another... down to a
            // read of the local itself. Not through a handle, whose payload others hold too.
            auto value = storeOp.getValue();
            auto throughField = false;
            for (;;)
            {
                auto loadOp = rootOf(value).getDefiningOp<mlir_ts::LoadOp>();
                if (!loadOp || isHandle(loadOp.getResult()))
                {
                    return false;
                }

                if (loadOp.getReference() == varOp.getResult())
                {
                    break;
                }

                auto field = loadOp.getReference().getDefiningOp<mlir_ts::PropertyRefOp>();
                if (!field || !mlir::isa<mlir_ts::ClassType>(field.getObjectRef().getType()))
                {
                    return false;
                }

                throughField = true;
                value = field.getObjectRef();
            }

            if (!throughField)
            {
                return false;
            }

            assigned = true;
        }

        return assigned;
    }

    bool OwnershipSignaturePass::holdsOnlyFresh(mlir_ts::VariableOp varOp)
    {
        if (!isOwningVariable(varOp) || varOp.getCaptured().value_or(false))
        {
            return false;
        }

        if (auto init = varOp.getInitializer(); init && !isFresh(rootOf(init)))
        {
            return false;
        }

        return llvm::all_of(varOp.getResult().getUsers(), [&](mlir::Operation *user) {
            auto storeOp = mlir::dyn_cast<mlir_ts::StoreOp>(user);
            return !storeOp || storeOp.getReference() != varOp.getResult() || isFresh(rootOf(storeOp.getValue()));
        });
    }

} // namespace own_signature

#undef DEBUG_TYPE

std::unique_ptr<mlir::Pass> mlir_ts::createOwnershipSignaturePass(bool executable, std::string objectFactsSymbol)
{
    return std::make_unique<own_signature::OwnershipSignaturePass>(executable, std::move(objectFactsSymbol));
}
