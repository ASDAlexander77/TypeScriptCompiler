// OwnershipSignaturePass (see OwnershipSignaturePass.h): generators - the iterator protocol, the
// fields a generator yields, and states that borrow their arguments (phase 7b).

#include "OwnershipSignaturePass.h"

#define DEBUG_TYPE "own"

namespace own_signature
{

    bool OwnershipSignaturePass::isIteratorNext(mlir::Operation *op)
    {
        if (op->getNumResults() != 1)
        {
            return false;
        }

        auto tupleType = mlir::dyn_cast<mlir_ts::TupleType>(op->getResult(0).getType());
        if (!tupleType || tupleType.getFields().size() != 2)
        {
            return false;
        }

        auto fieldName = [&](int index) {
            auto id = mlir::dyn_cast_or_null<mlir::StringAttr>(tupleType.getFields()[index].id);
            return id ? id.getValue() : llvm::StringRef();
        };
        if (fieldName(0) != "value" || fieldName(1) != "done")
        {
            return false;
        }

        auto isNext = [](llvm::StringRef name) { return name == "next" || name.ends_with(".next"); };
        if (auto callee = directCallee(op))
        {
            return isNext(callee.getValue());
        }

        auto function = calleeValue(op);
        if (!function)
        {
            return false;
        }

        if (auto getMethodOp = function.getDefiningOp<mlir_ts::GetMethodOp>())
        {
            function = getMethodOp.getBoundFunc();
        }

        if (auto boundOp = function.getDefiningOp<mlir_ts::CreateBoundFunctionOp>())
        {
            function = boundOp.getFunc();
        }

        if (auto loadOp = function.getDefiningOp<mlir_ts::LoadOp>())
        {
            if (auto refOp = loadOp.getReference().getDefiningOp<mlir_ts::InterfaceSymbolRefOp>())
            {
                auto identifier = mlir::dyn_cast<mlir::StringAttr>(refOp.getIdentifier());
                return identifier && isNext(identifier.getValue());
            }

            if (auto propertyRefOp = loadOp.getReference().getDefiningOp<mlir_ts::PropertyRefOp>())
            {
                auto storageType = storageTupleOfType(propertyRefOp->getOperand(0).getType());
                if (!storageType || propertyRefOp.getPosition() >= storageType.getFields().size())
                {
                    return false;
                }

                auto id = mlir::dyn_cast_or_null<mlir::StringAttr>(storageType.getFields()[propertyRefOp.getPosition()].id);
                return id && isNext(id.getValue());
            }

            return false;
        }

        if (auto refOp = function.getDefiningOp<mlir_ts::ThisVirtualSymbolRefOp>())
        {
            return isNext(refOp.getIdentifier());
        }

        if (auto refOp = function.getDefiningOp<mlir_ts::VirtualSymbolRefOp>())
        {
            return isNext(refOp.getIdentifier());
        }

        if (auto refOp = function.getDefiningOp<mlir_ts::ThisSymbolRefOp>())
        {
            return isNext(refOp.getIdentifier());
        }

        if (auto refOp = function.getDefiningOp<mlir_ts::SymbolRefOp>())
        {
            return isNext(refOp.getIdentifier());
        }

        return false;
    }

    void OwnershipSignaturePass::findImportedStates(mlir::ModuleOp module)
    {
        llvm::MapVector<mlir::Type, bool> states;
        for (auto &call : calls)
        {
            if (!makesGeneratorState(call.op) || (call.callees.known && !call.callees.funcs.empty()))
            {
                continue;
            }

            auto tupleType = storageTupleOfType(call.op->getResult(0).getType());
            auto borrows = !declaredByCallees(call.callees, OWN_DECLARED_BORROWS_ATTR_NAME).empty();
            states[tupleType] = states.lookup(tupleType) || borrows;
        }

        if (states.empty())
        {
            return;
        }

        auto *context = &getContext();
        llvm::SmallVector<mlir::Attribute> entries;
        if (auto existing = module->getAttrOfType<mlir::ArrayAttr>(OWN_BORROWING_FIELDS_ATTR_NAME))
        {
            entries.append(existing.begin(), existing.end());
        }

        for (auto &[type, borrows] : states)
        {
            auto tupleType = mlir::cast<mlir_ts::TupleType>(type);
            auto captured = fieldIndex(tupleType, CAPTURED_NAME);
            // the box first, when it borrows; -1 names no field, so a box that owns is released
            llvm::SmallVector<int32_t> fields{borrows && captured >= 0 ? captured : -1};
            for (auto [index, field] : llvm::enumerate(tupleType.getFields()))
            {
                if (static_cast<int32_t>(index) != captured && ownsHeap(module.getLoc(), field.type))
                {
                    fields.push_back(static_cast<int32_t>(index));
                }
            }

            entries.push_back(mlir::ArrayAttr::get(context, {mlir::TypeAttr::get(type), mlir::DenseI32ArrayAttr::get(context, fields)}));
        }

        module->setAttr(OWN_BORROWING_FIELDS_ATTR_NAME, mlir::ArrayAttr::get(context, entries));
    }

    bool OwnershipSignaturePass::makesGeneratorState(mlir::Operation *op)
    {
        if (op->getNumResults() != 1 || !mlir::isa<mlir_ts::ObjectType>(op->getResult(0).getType()))
        {
            return false;
        }

        auto tupleType = storageTupleOfType(op->getResult(0).getType());
        if (!tupleType)
        {
            return false;
        }

        auto hasStep = false;
        auto hasNext = false;
        for (auto field : tupleType.getFields())
        {
            if (auto id = mlir::dyn_cast_or_null<mlir::StringAttr>(field.id))
            {
                hasStep = hasStep || id.getValue() == ".step";
                hasNext = hasNext || id.getValue() == "next";
            }
        }

        return hasStep && hasNext;
    }

    mlir_ts::TupleType OwnershipSignaturePass::storageTupleOfType(mlir::Type type)
    {
        if (auto objectType = mlir::dyn_cast<mlir_ts::ObjectType>(type))
        {
            if (auto storageType = mlir::dyn_cast<mlir_ts::ObjectStorageType>(objectType.getStorageType()))
            {
                return mlir_ts::TupleType::get(type.getContext(), storageType.getFields());
            }

            // an object a declaration names by its shape (another module's generator)
            return mlir::dyn_cast<mlir_ts::TupleType>(objectType.getStorageType());
        }

        if (auto refType = mlir::dyn_cast<mlir_ts::RefType>(type))
        {
            type = refType.getElementType();
        }
        else if (auto valueRefType = mlir::dyn_cast<mlir_ts::ValueRefType>(type))
        {
            type = valueRefType.getElementType();
        }

        return mlir::dyn_cast<mlir_ts::TupleType>(type);
    }

    bool OwnershipSignaturePass::isRecordFieldStore(mlir::Operation *user, mlir::Value value)
    {
        auto storeOp = mlir::dyn_cast<mlir_ts::StoreOp>(user);
        auto propertyRefOp = storeOp && storeOp.getValue() == value
                                 ? storeOp.getReference().getDefiningOp<mlir_ts::PropertyRefOp>()
                                 : mlir_ts::PropertyRefOp();
        auto recordOp = propertyRefOp ? propertyRefOp->getOperand(0).getDefiningOp<mlir_ts::VariableOp>() : mlir_ts::VariableOp();
        return recordOp && recordOp.getInitializer() && recordOp.getInitializer().getDefiningOp<mlir_ts::ConstantOp>();
    }

    void OwnershipSignaturePass::findYieldedFields(mlir::ModuleOp module)
    {
        // (state type, field) -> its reads and its stores
        struct FieldUses
        {
            llvm::SmallVector<mlir_ts::LoadOp> reads;
            llvm::SmallVector<mlir_ts::StoreOp> stores;
            bool other = false;
        };

        llvm::MapVector<std::pair<mlir::Type, int32_t>, FieldUses> fields;
        module.walk([&](mlir_ts::PropertyRefOp propertyRefOp) {
            auto tupleType = storageTupleOf(propertyRefOp->getOperand(0).getType());
            auto position = static_cast<int32_t>(propertyRefOp.getPosition());
            if (!tupleType || fieldIndex(tupleType, ".step") < 0 || position == fieldIndex(tupleType, CAPTURED_NAME) ||
                !mlir::isa<mlir_ts::RefType>(propertyRefOp.getType()) ||
                !ownsHeap(propertyRefOp.getLoc(), mlir::cast<mlir_ts::RefType>(propertyRefOp.getType()).getElementType()))
            {
                return;
            }

            auto &uses = fields[{tupleType, position}];
            for (auto *user : propertyRefOp->getUsers())
            {
                if (auto loadOp = mlir::dyn_cast<mlir_ts::LoadOp>(user))
                {
                    uses.reads.push_back(loadOp);
                }
                else if (auto storeOp = mlir::dyn_cast<mlir_ts::StoreOp>(user); storeOp && storeOp.getReference() == propertyRefOp.getResult())
                {
                    uses.stores.push_back(storeOp);
                }
                else if (!mlir::isa<mlir_ts::ReleaseSlotOp, mlir_ts::RetainSlotOp>(user))
                {
                    uses.other = true;
                }
            }
        });

        // a field the signature pass already found borrowing is the caller's, not the generator's
        llvm::DenseSet<std::pair<mlir::Type, int32_t>> borrowing;
        if (auto entries = module->getAttrOfType<mlir::ArrayAttr>(OWN_BORROWING_FIELDS_ATTR_NAME))
        {
            for (auto entry : entries.getAsRange<mlir::ArrayAttr>())
            {
                for (auto index : mlir::cast<mlir::DenseI32ArrayAttr>(entry[1]).asArrayRef())
                {
                    borrowing.insert({mlir::cast<mlir::TypeAttr>(entry[0]).getValue(), index});
                }
            }
        }

        llvm::MapVector<mlir::Type, llvm::SmallVector<int32_t>> yielded;
        for (auto &[key, uses] : fields)
        {
            if (uses.other || uses.reads.empty() || uses.stores.empty() || borrowing.contains(key))
            {
                continue;
            }

            // read only to be put into a yield's record
            auto onlyYielded = llvm::all_of(uses.reads, [](mlir_ts::LoadOp loadOp) {
                auto users = llvm::make_filter_range(loadOp->getUsers(), [](mlir::Operation *user) {
                    return !mlir::isa<mlir_ts::RetainOp, mlir_ts::DebugVariableOp>(user);
                });
                return llvm::hasSingleElement(users) && isRecordFieldStore(*users.begin(), loadOp.getResult());
            });

            // each value stored is yielded later in its block: nothing between stops the generator
            auto eachYielded = llvm::all_of(uses.stores, [&](mlir_ts::StoreOp storeOp) {
                return llvm::any_of(uses.reads, [&](mlir_ts::LoadOp loadOp) {
                    return loadOp->getBlock() == storeOp->getBlock() && storeOp->isBeforeInBlock(loadOp);
                });
            });

            if (!onlyYielded || !eachYielded)
            {
                continue;
            }

            yielded[key.first].push_back(key.second);
            for (auto loadOp : uses.reads)
            {
                loadOp->setAttr(OWN_FRESH_RESULT_ATTR_NAME, mlir::UnitAttr::get(&getContext()));
            }
        }

        if (yielded.empty())
        {
            return;
        }

        auto *context = &getContext();
        llvm::SmallVector<mlir::Attribute> entries;
        for (auto &[type, indices] : yielded)
        {
            entries.push_back(mlir::ArrayAttr::get(context, {mlir::TypeAttr::get(type), mlir::DenseI32ArrayAttr::get(context, indices)}));
        }

        module->setAttr(OWN_YIELDED_FIELDS_ATTR_NAME, mlir::ArrayAttr::get(context, entries));
    }

    void OwnershipSignaturePass::findBoundedForwarders()
    {
        llvm::DenseMap<mlir::Operation *, const Callees *> calleesOf;
        for (auto &call : calls)
        {
            calleesOf[call.op] = &call.callees;
        }

        for (auto &entry : functions)
        {
            auto funcOp = entry.second;
            if (funcOp.isDeclaration() || !funcOp->hasAttr(OWN_DECLARED_BORROWS_ATTR_NAME) || borrowsFound.count(funcOp))
            {
                continue;
            }

            llvm::SmallVector<int32_t> found;
            auto forwards = true;
            for (auto value : returnedValues(funcOp))
            {
                if (holdsNoBlock(value))
                {
                    continue;
                }

                auto *call = rootOf(value).getDefiningOp();
                auto known = call ? calleesOf.find(call) : calleesOf.end();
                if (known == calleesOf.end())
                {
                    forwards = false;
                    break;
                }

                llvm::SmallVector<int32_t> bounded = declaredByCallees(*known->second, OWN_DECLARED_BORROWS_ATTR_NAME);
                if (bounded.empty() && known->second->funcs.size() == 1 && !known->second->viaCallback)
                {
                    auto made = declared(known->second->funcs.front(), OWN_RESULT_BOUNDED_ATTR_NAME);
                    bounded.assign(made.begin(), made.end());
                }

                if (bounded.empty())
                {
                    forwards = false;
                    break;
                }

                auto args = callArgs(call);
                for (auto index : bounded)
                {
                    auto param = static_cast<size_t>(index) < args.size() ? parameterOf(args[index], funcOp) : -1;
                    if (param < 0)
                    {
                        forwards = false;
                        break;
                    }

                    if (!llvm::is_contained(found, param))
                    {
                        found.push_back(param);
                    }
                }
            }

            if (forwards && !found.empty())
            {
                llvm::sort(found);
                borrowsFound[funcOp] = found;
            }
        }
    }

    int32_t OwnershipSignaturePass::fieldIndex(mlir_ts::TupleType tupleType, llvm::StringRef name)
    {
        for (auto [index, field] : llvm::enumerate(tupleType.getFields()))
        {
            if (auto id = mlir::dyn_cast_or_null<mlir::StringAttr>(field.id); id && id.getValue() == name)
            {
                return static_cast<int32_t>(index);
            }
        }

        return -1;
    }

    mlir::Value OwnershipSignaturePass::boxStoredIntoState(mlir_ts::NewOp newOp, int32_t captured)
    {
        mlir_ts::StoreOp fill;
        for (auto *user : newOp->getUsers())
        {
            if (auto storeOp = mlir::dyn_cast<mlir_ts::StoreOp>(user); storeOp && storeOp.getReference() == newOp.getResult())
            {
                fill = storeOp;
            }
        }

        auto loadOp = fill ? fill.getValue().getDefiningOp<mlir_ts::LoadOp>() : mlir_ts::LoadOp();
        auto stateOp = loadOp ? loadOp.getReference().getDefiningOp<mlir_ts::VariableOp>() : mlir_ts::VariableOp();
        if (!stateOp)
        {
            return {};
        }

        mlir::Value box;
        for (auto *user : stateOp->getUsers())
        {
            auto propertyRefOp = mlir::dyn_cast<mlir_ts::PropertyRefOp>(user);
            if (!propertyRefOp || propertyRefOp.getPosition() != captured)
            {
                continue;
            }

            for (auto *fieldUser : propertyRefOp->getUsers())
            {
                auto storeOp = mlir::dyn_cast<mlir_ts::StoreOp>(fieldUser);
                if (!storeOp || box)
                {
                    return {};
                }

                box = storeOp.getValue();
            }
        }

        auto boxOp = box ? box.getDefiningOp<mlir_ts::VariableOp>() : mlir_ts::VariableOp();
        return boxOp && boxOp->hasAttr(CAPTURE_BOX_ATTR_NAME) ? box : mlir::Value();
    }

    bool OwnershipSignaturePass::calledOnlyDirectly(mlir::ModuleOp module, mlir_ts::FuncOp funcOp, bool publicToo)
    {
        if (!funcOp.isPrivate() && !publicToo)
        {
            return false;
        }

        auto uses = mlir::SymbolTable::getSymbolUses(funcOp.getSymNameAttr(), &module.getBodyRegion());
        if (!uses)
        {
            return false;
        }

        for (auto &use : *uses)
        {
            auto *user = use.getUser();
            if (mlir::isa<mlir_ts::SymbolCallInternalOp, mlir_ts::CallOp, mlir_ts::InvokeOp>(user))
            {
                continue;
            }

            // a method whose signature gives its callers the bound: called through its class's
            // vtable or a reference to it, which the facts reach too (declaredByCallees)
            if (publicToo &&
                (isClassVTableEntry(user) ||
                 (mlir::isa<mlir_ts::ThisVirtualSymbolRefOp, mlir_ts::VirtualSymbolRefOp, mlir_ts::ThisSymbolRefOp,
                            mlir_ts::SymbolRefOp>(user) &&
                  onlyCalled(user->getResult(0)))))
            {
                continue;
            }

            auto symbolRefOp = mlir::dyn_cast<mlir_ts::SymbolRefOp>(user);
            if (!symbolRefOp)
            {
                return false;
            }

            for (auto *refUser : symbolRefOp->getUsers())
            {
                auto boundOp = mlir::dyn_cast<mlir_ts::CreateBoundFunctionOp>(refUser);
                if (!boundOp || !llvm::all_of(boundOp->getUsers(), [](mlir::Operation *boundUser) {
                        return mlir::isa<mlir_ts::RetainOp, mlir_ts::ReleaseOp>(boundUser);
                    }))
                {
                    return false;
                }
            }
        }

        return true;
    }

    bool OwnershipSignaturePass::isParameterCell(mlir_ts::VariableOp cellOp, mlir_ts::FuncOp maker, int32_t &index)
    {
        auto argument = cellOp.getInitializer() ? mlir::dyn_cast<mlir::BlockArgument>(cellOp.getInitializer())
                                                : mlir::BlockArgument();
        if (!cellOp.getCaptured().value_or(false) || isOwningVariable(cellOp) || !argument ||
            !argument.getOwner()->isEntryBlock() || argument.getOwner()->getParentOp() != maker.getOperation())
        {
            return false;
        }

        index = static_cast<int32_t>(argument.getArgNumber());
        return true;
    }

    bool OwnershipSignaturePass::readsParameterSlot(mlir::Value value, mlir_ts::FuncOp maker, int32_t &index)
    {
        auto loadOp = rootOf(value).getDefiningOp<mlir_ts::LoadOp>();
        auto varOp = loadOp ? loadOp.getReference().getDefiningOp<mlir_ts::VariableOp>() : mlir_ts::VariableOp();
        int position = -1;
        if (!varOp || !isParameterSlot(varOp, position) || varOp->getParentOfType<mlir_ts::FuncOp>() != maker)
        {
            return false;
        }

        index = static_cast<int32_t>(position);
        return true;
    }

    bool OwnershipSignaturePass::readsParameterField(mlir::Value value, mlir_ts::FuncOp maker, int32_t &index)
    {
        auto loadOp = rootOf(value).getDefiningOp<mlir_ts::LoadOp>();
        auto propertyRefOp = loadOp ? loadOp.getReference().getDefiningOp<mlir_ts::PropertyRefOp>() : mlir_ts::PropertyRefOp();
        auto argument = propertyRefOp ? mlir::dyn_cast<mlir::BlockArgument>(propertyRefOp->getOperand(0)) : mlir::BlockArgument();
        if (!argument || !argument.getOwner()->isEntryBlock() || argument.getOwner()->getParentOp() != maker.getOperation())
        {
            return false;
        }

        index = static_cast<int32_t>(argument.getArgNumber());
        return true;
    }

    void OwnershipSignaturePass::findBorrowingStates(mlir::ModuleOp module)
    {
        llvm::MapVector<mlir::Type, BorrowingState> states;
        llvm::DenseMap<mlir::Type, int> made;
        module.walk([&](mlir_ts::NewOp newOp) {
            auto valueRefType = mlir::dyn_cast<mlir_ts::ValueRefType>(newOp.getType());
            auto tupleType = valueRefType ? mlir::dyn_cast<mlir_ts::TupleType>(valueRefType.getElementType())
                                          : mlir_ts::TupleType();
            if (!tupleType)
            {
                return;
            }

            ++made[tupleType];
            auto captured = fieldIndex(tupleType, CAPTURED_NAME);
            auto maker = newOp->getParentOfType<mlir_ts::FuncOp>();
            // an open maker has callers this pass cannot give the bound to, unless its signature
            // gives it to them (`@borrows`)
            if (captured < 0 || !maker ||
                !calledOnlyDirectly(module, maker, /*publicToo=*/maker->hasAttr(OWN_DECLARED_BORROWS_ATTR_NAME)))
            {
                return;
            }

            auto box = boxStoredIntoState(newOp, captured);
            if (!box)
            {
                return;
            }

            BorrowingState state{maker, captured};
            for (auto *user : box.getUsers())
            {
                auto propertyRefOp = mlir::dyn_cast<mlir_ts::PropertyRefOp>(user);
                if (!propertyRefOp)
                {
                    continue;
                }

                for (auto *fieldUser : propertyRefOp->getUsers())
                {
                    auto storeOp = mlir::dyn_cast<mlir_ts::StoreOp>(fieldUser);
                    if (!storeOp)
                    {
                        return;
                    }

                    auto value = storeOp.getValue();
                    int32_t index = -1;
                    if (auto cellOp = value.getDefiningOp<mlir_ts::VariableOp>(); cellOp && isParameterCell(cellOp, maker, index))
                    {
                        auto cellType = mlir::cast<mlir_ts::RefType>(cellOp.getType());
                        // a handle parameter's cell holds a count of its own (spec 23.2), which only
                        // the box's release gives back: a box holding one owns what it holds
                        if (MLIRTypeHelper::isSharedHandleType(cellType.getElementType()))
                        {
                            return;
                        }

                        if (ownsHeap(cellOp.getLoc(), cellType.getElementType()))
                        {
                            state.bounded.push_back(index);
                        }

                        continue;
                    }

                    // a copy of what a parameter holds in a field: `.map`'s maker is a closure body
                    // that copies its own box's fields into the generator's
                    if (!ownsHeap(value.getLoc(), value.getType()))
                    {
                        continue;
                    }

                    // `this`, which the box holds by value (`.this`), not in a cell: read out of
                    // the parameter's slot
                    if (readsParameterSlot(value, maker, index))
                    {
                        if (MLIRTypeHelper::isSharedHandleType(value.getType()))
                        {
                            return;
                        }

                        if (!llvm::is_contained(state.bounded, index))
                        {
                            state.bounded.push_back(index);
                        }

                        continue;
                    }

                    if (!readsParameterField(value, maker, index))
                    {
                        return; // a box with anything but what the parameters hold owns what it holds
                    }

                    if (!llvm::is_contained(state.bounded, index))
                    {
                        state.bounded.push_back(index);
                    }
                }
            }

            if (!state.bounded.empty())
            {
                states[tupleType] = state;
            }
        });

        // made in one place only: the maker is the one function a bound can come from
        states.remove_if([&](auto &entry) { return made[entry.first] != 1; });
        for (auto &[type, state] : states)
        {
            auto bounded = state.bounded;
            llvm::sort(bounded);
            borrowsFound[state.maker] = bounded;
        }
        if (states.empty())
        {
            return;
        }

        // Each store into a field of a state type, by field: what the field may be given.
        llvm::DenseMap<std::pair<mlir::Type, int64_t>, llvm::SmallVector<mlir::Value>> stored;
        module.walk([&](mlir_ts::StoreOp storeOp) {
            auto propertyRefOp = storeOp.getReference().getDefiningOp<mlir_ts::PropertyRefOp>();
            auto tupleType = propertyRefOp ? storageTupleOf(propertyRefOp->getOperand(0).getType()) : mlir_ts::TupleType();
            if (tupleType && states.count(tupleType))
            {
                stored[{tupleType, propertyRefOp.getPosition()}].push_back(storeOp.getValue());
            }
        });

        for (auto &[type, state] : states)
        {
            auto tupleType = mlir::cast<mlir_ts::TupleType>(type);
            // least fixpoint: a field borrows once everything stored into it is a read of a borrowing
            // cell or field
            for (auto changed = true; changed;)
            {
                changed = false;
                for (auto [index, field] : llvm::enumerate(tupleType.getFields()))
                {
                    auto position = static_cast<int32_t>(index);
                    auto found = stored.find({type, position});
                    if (position == state.captured || state.fields.contains(position) || found == stored.end() ||
                        !ownsHeap(state.maker.getLoc(), field.type))
                    {
                        continue;
                    }

                    if (llvm::all_of(found->second, [&](mlir::Value value) { return readsBorrowed(value, tupleType, state); }))
                    {
                        state.fields.insert(position);
                        changed = true;
                    }
                }
            }
        }

        auto *context = &getContext();
        llvm::SmallVector<mlir::Attribute> entries;
        for (auto &[type, state] : states)
        {
            llvm::SmallVector<int32_t> fields{state.captured};
            fields.append(state.fields.begin(), state.fields.end());
            entries.push_back(mlir::ArrayAttr::get(
                context, {mlir::TypeAttr::get(type), mlir::DenseI32ArrayAttr::get(context, fields)}));
            state.maker->setAttr(OWN_RESULT_BOUNDED_ATTR_NAME, mlir::DenseI32ArrayAttr::get(context, state.bounded));
        }

        module->setAttr(OWN_BORROWING_FIELDS_ATTR_NAME, mlir::ArrayAttr::get(context, entries));
    }

    bool OwnershipSignaturePass::readsBorrowed(mlir::Value value, mlir_ts::TupleType tupleType, const BorrowingState &state)
    {
        auto loadOp = rootOf(value).getDefiningOp<mlir_ts::LoadOp>();
        if (!loadOp)
        {
            return false;
        }

        auto isStateField = [&](mlir::Value ref, int32_t &position) {
            auto propertyRefOp = ref.getDefiningOp<mlir_ts::PropertyRefOp>();
            if (!propertyRefOp || storageTupleOf(propertyRefOp->getOperand(0).getType()) != tupleType)
            {
                return false;
            }

            position = propertyRefOp.getPosition();
            return true;
        };

        int32_t position = -1;
        if (isStateField(loadOp.getReference(), position))
        {
            return state.fields.contains(position);
        }

        // a field of the box the state holds, copied in by value, or a cell read out of it
        auto isBoxField = [&](mlir::Value ref) {
            auto fieldRef = ref.getDefiningOp<mlir_ts::PropertyRefOp>();
            auto boxLoad = fieldRef ? fieldRef->getOperand(0).getDefiningOp<mlir_ts::LoadOp>() : mlir_ts::LoadOp();
            return boxLoad && isStateField(boxLoad.getReference(), position) && position == state.captured;
        };

        if (isBoxField(loadOp.getReference()))
        {
            return true;
        }

        auto cellLoad = loadOp.getReference().getDefiningOp<mlir_ts::LoadOp>();
        return cellLoad && isBoxField(cellLoad.getReference());
    }

} // namespace own_signature
