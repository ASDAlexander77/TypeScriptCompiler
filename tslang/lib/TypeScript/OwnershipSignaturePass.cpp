// OwnershipSignaturePass (see OwnershipSignaturePass.h): the pass itself, the facts a signature says,
// the facts callers must know and their fixpoint, drops, and the facts a library exports.

#include "OwnershipSignaturePass.h"

#define DEBUG_TYPE "own"

namespace own_signature
{

    void OwnershipSignaturePass::runOnOperation()
    {
        auto module = getOperation();
        module.walk([&](mlir_ts::FuncOp funcOp) { functions[funcOp.getSymName()] = funcOp; });
        module.walk([&](mlir_ts::GlobalOp globalOp) { globals[globalOp.getSymName()] = globalOp; });
        module.walk([&](mlir_ts::AddressOfOp addressOfOp) {
            auto onlyLoaded = llvm::all_of(addressOfOp->getUsers(), [](mlir::Operation *user) {
                return mlir::isa<mlir_ts::LoadOp>(user);
            });
            if (!onlyLoaded)
            {
                writtenGlobals.insert(addressOfOp.getGlobalName());
            }
        });
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
        });
        if (!backReferenceFields.empty())
        {
            module->setAttr(OWN_BACK_REFERENCE_FIELDS_ATTR_NAME,
                            mlir::ArrayAttr::get(&getContext(), backReferenceFields.getArrayRef()));
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
        // before the facts: a value kept for good is immortal, which keeps nothing of its owner
        findKeepers(module);
        decideKeptForGood(module);
        // before the facts: a generator method whose state borrows `this` does not keep it
        findBorrowingStates(module);
        computeFacts();
        computeDrops();
        pinAssignedGlobals();
        rejectGlobalArgumentsDropped();
        findBoundedForwarders();
        findYieldedFields(module);
        findImportedStates(module);
        checkDeclared();

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
            // their facts (spec 31), and the call gets theirs like any other. Nor one whose result is
            // a borrowing record (spec 32): the caller owns the box, but what it borrows bounds it.
            if (((isIteratorNext(call.op) && !call.callees.viaInterface) || makesGeneratorState(call.op)) &&
                !isRecordResult(call.op))
            {
                call.op->setAttr(OWN_FRESH_RESULT_ATTR_NAME, mlir::UnitAttr::get(&getContext()));
            }

            call.op->removeAttr(OWN_DROP_FIELDS_ATTR_NAME);
            if (!call.callees.known)
            {
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
            }

            // `new C()` of a class from a library built under own: see OWN_FRESH_RESULT_ATTR_NAME
            // ...and of a function another own module's object says returns one its caller owns
            // (--own-facts)
            if (call.callees.imported && call.op->getNumResults() == 1 &&
                ((!call.callees.importedOverwrites && call.callees.importedName.ends_with("." NEW_METHOD_NAME)) ||
                 importedFreshResults.contains(call.callees.importedName)))
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
                                   OWN_DECLARED_RECORD_ATTR_NAME})
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
            return {{found}, false};
        }

        return {};
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

            // `this` in a closure's box (captured by copy, spec 70): the object is the caller's, which
            // no box can own. A closure that does not escape borrows it, as it did the cell; one stored
            // into a field of `this` is a back-reference; any other one is inference's error
            if (auto propertyRefOp = storeOp.getReference().getDefiningOp<mlir_ts::PropertyRefOp>();
                propertyRefOp && isMemberThis(used) && closureOfBox(propertyRefOp->getOperand(0)))
            {
                return false;
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
        // local copies it (spec 22)
        if (auto varOp = mlir::dyn_cast<mlir_ts::VariableOp>(user))
        {
            return varOp.getInitializer() == used && isOwningVariable(varOp) && !varOp.getCaptured().value_or(false) &&
                   isAssigned(varOp) && !isCopyableString(used.getType());
        }

        return false;
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
        op->removeAttr(OWN_PARAMS_ATTR_NAME);
        if (!facts.borrows.empty())
        {
            op->setAttr(OWN_RESULT_BORROWS_ATTR_NAME, mlir::DenseI32ArrayAttr::get(context, facts.borrows));
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
            // a signature's `@keeps` is checked against the body as it is (checkDeclared)
            llvm::SmallVector<int32_t> released;
            for (auto index : effective.owned)
            {
                if (!declaresCallerFacts(funcOp) && !llvm::is_contained(takenParams.lookup(funcOp), index))
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
        llvm::DenseMap<mlir::Operation *, llvm::SmallVector<Call *>> callsIn;
        for (auto &call : calls)
        {
            callsIn[call.op->getParentOfType<mlir_ts::FuncOp>()].push_back(&call);
        }

        for (auto &entry : functions)
        {
            auto funcOp = entry.second;
            if (!funcOp.isDeclaration() && !dropsInBody(funcOp))
            {
                noDrops.insert(funcOp);
            }
        }

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
                fieldDrops[funcOp] = fieldDropsInBody(funcOp);
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

                    for (auto place : callee->places)
                    {
                        changed |= mine.add(place);
                    }
                }
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

        for (auto callee : callees.funcs)
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

            for (auto place : found->second.places)
            {
                all.add(place);
            }
        }

        return all;
    }

    OwnershipSignaturePass::FieldDrops OwnershipSignaturePass::fieldDropsInBody(mlir_ts::FuncOp funcOp)
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
                if (releasesNothing(releaseSlotOp))
                {
                    return;
                }

                if (isLoadedCell(slot) && slotType && isCopyableString(slotType.getElementType()))
                {
                    // a captured string assigned (`each += v` in a callback): it frees that string only
                    drops.add({OWN_CAPTURED_STRING_POSITION, slot.getType()});
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

                    drops.add({position, slot.getType()});
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
            else if (mlir::isa<mlir_ts::ArrayPopOp, mlir_ts::ArrayShiftOp, mlir_ts::ArraySpliceOp, mlir_ts::SetLengthOfOp>(op))
            {
                if (reachesOutside(op->getOperand(0), funcOp))
                {
                    // the elements of an array of its element type, as a store into one drops them
                    auto refType = mlir::dyn_cast<mlir_ts::RefType>(op->getOperand(0).getType());
                    auto arrayType = refType ? mlir::dyn_cast<mlir_ts::ArrayType>(refType.getElementType()) : mlir_ts::ArrayType();
                    if (arrayType)
                    {
                        drops.add({-1, mlir_ts::RefType::get(arrayType.getElementType())});
                    }
                    else
                    {
                        drops.any = true;
                    }
                }
            }
            else if (mlir::isa<mlir_ts::ArrayPushOp, mlir_ts::ArrayUnshiftOp>(op))
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

    bool OwnershipSignaturePass::dropsInBody(mlir_ts::FuncOp funcOp)
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
                drops = true;
            }
            else if (auto releaseSlotOp = mlir::dyn_cast<mlir_ts::ReleaseSlotOp>(op))
            {
                // a captured variable assigned through its cell, by a closure's body: the variable
                // is its declaring function's
                auto slot = releaseSlotOp.getSlot();
                drops = !releasesNothing(releaseSlotOp) &&slot.getDefiningOp<mlir_ts::AddressOfOp>() || (isPlace(slot) && reachesOutside(slot, funcOp)) ||
                        isLoadedCell(slot) || (!isPlace(slot) && touchesHandle(op) && !isCountedReadSlot(slot));
            }
            else if (mlir::isa<mlir_ts::ArrayPopOp, mlir_ts::ArrayShiftOp, mlir_ts::ArraySpliceOp, mlir_ts::SetLengthOfOp>(op))
            {
                drops = reachesOutside(op->getOperand(0), funcOp);
            }
            else if (auto swapOp = mlir::dyn_cast<mlir_ts::SwapOp>(op))
            {
                // the caller's borrow of a place it exchanges reads another place's value (spec 42)
                drops = reachesOutside(swapOp.getFirst(), funcOp) || reachesOutside(swapOp.getSecond(), funcOp);
            }
            else if (mlir::isa<mlir_ts::ArrayPushOp, mlir_ts::ArrayUnshiftOp>(op))
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

    bool OwnershipSignaturePass::isAssigned(mlir_ts::VariableOp varOp)
    {
        return llvm::any_of(varOp.getResult().getUsers(), [&](mlir::Operation *user) {
            auto storeOp = mlir::dyn_cast<mlir_ts::StoreOp>(user);
            return storeOp && storeOp.getReference() == varOp.getResult();
        });
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
