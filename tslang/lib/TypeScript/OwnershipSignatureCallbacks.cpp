// OwnershipSignaturePass (see OwnershipSignaturePass.h): callbacks (spec 33) - the functions passed
// for a parameter, and the facts their families agree on.

#include "OwnershipSignaturePass.h"

#define DEBUG_TYPE "own"

namespace own_signature
{

    static bool isFunctionValue(mlir::Type type)
    {
        return mlir::isa<mlir_ts::FunctionType, mlir_ts::HybridFunctionType, mlir_ts::BoundFunctionType>(type);
    }

    bool OwnershipSignaturePass::isFunctionValueType(mlir::Type type)
    {
        return isFunctionValue(type);
    }

    // A callback parameter's type: a function value, or an optional one
    static bool isCallbackType(mlir::Type type)
    {
        if (auto optionalType = mlir::dyn_cast<mlir_ts::OptionalType>(type))
        {
            type = optionalType.getElementType();
        }

        return isFunctionValue(type);
    }

    // A read of a callback parameter: the function a call goes through, tested for being given
    // (`if (f)`), unwrapped, or an argument of a call or stored into a class's field, either of
    // which gives it on (`forwards`)
    static bool onlyCalledValue(mlir::Value value, llvm::SmallVectorImpl<mlir::OpOperand *> &forwards, int depth = 0)
    {
        if (depth > 8)
        {
            return false;
        }

        for (auto &use : value.getUses())
        {
            auto *user = use.getOwner();
            if (mlir::isa<mlir_ts::HasValueOp>(user))
            {
                continue;
            }

            if (auto castOp = mlir::dyn_cast<mlir_ts::CastOp>(user))
            {
                if (mlir::isa<mlir_ts::BooleanType>(castOp.getType()))
                {
                    continue;
                }

                if (!isFunctionValue(castOp.getType()) ||
                    !onlyCalledValue(castOp.getResult(), forwards, depth + 1))
                {
                    return false;
                }

                continue;
            }

            if (auto valueOp = mlir::dyn_cast<mlir_ts::ValueOp>(user))
            {
                if (!onlyCalledValue(valueOp.getResult(), forwards, depth + 1))
                {
                    return false;
                }

                continue;
            }

            // rc's count for a store of it
            if (mlir::isa<mlir_ts::RetainOp>(user))
            {
                continue;
            }

            // stored into a class's field: given on to the field's family (spec 55)
            if (auto storeOp = mlir::dyn_cast<mlir_ts::StoreOp>(user))
            {
                auto fieldOp = storeOp.getReference().getDefiningOp<mlir_ts::PropertyRefOp>();
                if (storeOp.getValue() != use.get() || !fieldOp ||
                    !mlir::isa<mlir_ts::ClassType>(fieldOp->getOperand(0).getType()))
                {
                    return false;
                }

                forwards.push_back(&use);
                continue;
            }

            // the callee, or one argument: never both, never twice
            auto called = calleeValue(user) == use.get();
            if (!isCall(user) || llvm::count(callArgs(user), use.get()) + (called ? 1 : 0) != 1)
            {
                return false;
            }

            if (!called)
            {
                forwards.push_back(&use);
            }
        }

        return true;
    }

    // The index a function's result family is kept under: no parameter's, nor a field's (spec 56)
    static constexpr int32_t RESULT_FAMILY = std::numeric_limits<int32_t>::min();

    // Through what unwraps or makes optional a function value, or casts it between function types
    static mlir::Value unwrapFunction(mlir::Value value)
    {
        for (auto depth = 0; value && depth < 8; ++depth)
        {
            auto *def = value.getDefiningOp();
            if (auto valueOp = mlir::dyn_cast_or_null<mlir_ts::ValueOp>(def))
            {
                value = valueOp.getIn();
            }
            else if (auto optionalOp = mlir::dyn_cast_or_null<mlir_ts::OptionalValueOp>(def))
            {
                value = optionalOp.getIn();
            }
            else if (auto castOp = mlir::dyn_cast_or_null<mlir_ts::CastOp>(def); castOp && isFunctionValue(castOp.getType()))
            {
                value = castOp.getIn();
            }
            else
            {
                break;
            }
        }

        return value;
    }

    // A local declared from `value` and never assigned nor captured: each read of it is `value`
    static mlir::Value initializerOf(mlir_ts::VariableOp varOp)
    {
        if (!varOp || !varOp.getInitializer() || varOp.getCaptured().value_or(false) ||
            llvm::any_of(varOp->getUsers(), [](mlir::Operation *user) { return mlir::isa<mlir_ts::StoreOp>(user); }))
        {
            return {};
        }

        return varOp.getInitializer();
    }

    // The function whose call made `value` - directly, or read out of a local declared from it - when
    // it returns a function value: its result family is the value's (spec 56)
    static mlir::FlatSymbolRefAttr givenResult(mlir::Value value)
    {
        value = unwrapFunction(value);
        if (auto loadOp = value ? value.getDefiningOp<mlir_ts::LoadOp>() : mlir_ts::LoadOp())
        {
            value = unwrapFunction(initializerOf(loadOp.getReference().getDefiningOp<mlir_ts::VariableOp>()));
        }

        auto *call = value ? value.getDefiningOp() : nullptr;
        return call && isCall(call) && call->getNumResults() == 1 ? directCallee(call) : mlir::FlatSymbolRefAttr();
    }

    // The caller's parameter `value` gives on: a read of its slot, unwrapped, made optional or cast
    // between function types. Its function, or null.
    static mlir_ts::FuncOp givenParam(mlir::Value value, int &index)
    {
        for (auto depth = 0; value && depth < 8; ++depth)
        {
            auto *def = value.getDefiningOp();
            if (auto valueOp = mlir::dyn_cast_or_null<mlir_ts::ValueOp>(def))
            {
                value = valueOp.getIn();
            }
            else if (auto optionalOp = mlir::dyn_cast_or_null<mlir_ts::OptionalValueOp>(def))
            {
                value = optionalOp.getIn();
            }
            else if (auto castOp = mlir::dyn_cast_or_null<mlir_ts::CastOp>(def);
                     castOp && isFunctionValue(castOp.getType()))
            {
                value = castOp.getIn();
            }
            else
            {
                auto loadOp = mlir::dyn_cast_or_null<mlir_ts::LoadOp>(def);
                auto varOp = loadOp ? loadOp.getReference().getDefiningOp<mlir_ts::VariableOp>() : mlir_ts::VariableOp();
                return varOp && isParameterSlot(varOp, index) ? varOp->getParentOfType<mlir_ts::FuncOp>() : mlir_ts::FuncOp();
            }
        }

        return {};
    }

    mlir_ts::FuncOp OwnershipSignaturePass::closureFunctionOf(mlir::Value box, mlir::Value cell, int32_t position)
    {
        auto boundOp = closureOfBox(box);
        auto refOp = boundOp ? boundOp.getFunc().getDefiningOp<mlir_ts::SymbolRefOp>() : mlir_ts::SymbolRefOp();
        auto funcOp = refOp ? functions.lookup(refOp.getIdentifier()) : mlir_ts::FuncOp();
        if (!funcOp || funcOp.isDeclaration() || funcOp.getBody().front().getNumArguments() == 0)
        {
            return {};
        }

        // every box that comes in as its first argument is one of these
        auto uses =mlir::SymbolTable::getSymbolUses(funcOp, funcOp->getParentOfType<mlir::ModuleOp>());
        if (!uses)
        {
            return {};
        }

        // each box it is bound to holds the same cell in the same field
        auto holdsCell = [&](mlir::Value other) {
            if (!closureOfBox(other))
            {
                return false;
            }

            auto stored = false;
            for (auto *boxUser : other.getUsers())
            {
                auto fieldOp = mlir::dyn_cast<mlir_ts::PropertyRefOp>(boxUser);
                if (!fieldOp || static_cast<int32_t>(fieldOp.getPosition()) != position)
                {
                    continue;
                }

                for (auto *fieldUser : fieldOp->getUsers())
                {
                    auto storeOp = mlir::dyn_cast<mlir_ts::StoreOp>(fieldUser);
                    if (!storeOp || storeOp.getValue() != cell)
                    {
                        return false;
                    }

                    stored = true;
                }
            }

            return stored;
        };

        for (auto &use : *uses)
        {
            // called by name with the box (the built-in `forEach`, inlined, calls the closure so)
            if (directCallee(use.getUser()))
            {
                auto args = callArgs(use.getUser());
                if (args.empty() || !holdsCell(args.front()))
                {
                    return {};
                }

                continue;
            }

            auto useRef = mlir::dyn_cast<mlir_ts::SymbolRefOp>(use.getUser());
            if (!useRef || llvm::any_of(useRef->getUsers(), [&](mlir::Operation *refUser) {
                    auto bound = mlir::dyn_cast<mlir_ts::CreateBoundFunctionOp>(refUser);
                    return !bound || !holdsCell(bound.getThisVal());
                }))
            {
                return {};
            }
        }

        return funcOp;
    }

    bool OwnershipSignaturePass::onlyCalledParam(mlir::BlockArgument argument, llvm::SmallVectorImpl<mlir::OpOperand *> &forwards,
                                                 llvm::SmallVectorImpl<std::pair<mlir::Operation *, int32_t>> &captured)
    {
        for (auto *user : argument.getUsers())
        {
            auto varOp = mlir::dyn_cast<mlir_ts::VariableOp>(user);
            auto index = -1;
            // a captured one: its cell, never assigned, read here or by the closures it is given to
            auto cell = varOp && varOp.getCaptured().value_or(false) && !isOwningVariable(varOp) &&
                        varOp.getInitializer() == argument;
            if (!varOp || (!cell && !isParameterSlot(varOp, index)))
            {
                return false;
            }

            for (auto &slotUse : varOp->getUses())
            {
                auto *slotUser = slotUse.getOwner();
                if (mlir::isa<mlir_ts::DebugVariableOp>(slotUser) ||
                    (cell && mlir::isa<mlir_ts::RetainCellOp, mlir_ts::ReleaseCellOp>(slotUser)))
                {
                    continue;
                }

                if (auto loadOp = mlir::dyn_cast<mlir_ts::LoadOp>(slotUser))
                {
                    if (!onlyCalledValue(loadOp.getResult(), forwards))
                    {
                        return false;
                    }

                    continue;
                }

                // the cell into a field of a closure's capture box: the closure's body reads it out of
                // its box (`ts.Load(ts.Load(ts.PropertyRef(box, i)))`) only to call it
                auto storeOp = mlir::dyn_cast<mlir_ts::StoreOp>(slotUser);
                auto fieldOp = storeOp && cell && storeOp.getValue() == varOp.getResult()
                                   ? storeOp.getReference().getDefiningOp<mlir_ts::PropertyRefOp>()
                                   : mlir_ts::PropertyRefOp();
                auto position = fieldOp ? static_cast<int32_t>(fieldOp.getPosition()) : -1;
                auto closure = fieldOp ? closureFunctionOf(fieldOp->getOperand(0), varOp.getResult(), position) : mlir_ts::FuncOp();
                if (!closure)
                {
                    return false;
                }

                // the same closure met again through its other box
                if (llvm::is_contained(captured, std::make_pair(closure.getOperation(), position)))
                {
                    continue;
                }

                auto box = closure.getBody().front().getArgument(0);
                for (auto *boxUser : box.getUsers())
                {
                    auto readField = mlir::dyn_cast<mlir_ts::PropertyRefOp>(boxUser);
                    if (!readField)
                    {
                        return false;
                    }

                    if (readField.getPosition() != static_cast<uint32_t>(position))
                    {
                        continue;
                    }

                    for (auto *fieldUser : readField->getUsers())
                    {
                        auto cellRead = mlir::dyn_cast<mlir_ts::LoadOp>(fieldUser);
                        if (!cellRead)
                        {
                            return false;
                        }

                        for (auto *cellUser : cellRead->getUsers())
                        {
                            auto valueRead = mlir::dyn_cast<mlir_ts::LoadOp>(cellUser);
                            if (!valueRead || !onlyCalledValue(valueRead.getResult(), forwards))
                            {
                                return false;
                            }
                        }
                    }
                }

                captured.push_back({closure.getOperation(), position});
            }
        }

        return true;
    }

    mlir_ts::SymbolRefOp OwnershipSignaturePass::passedFunction(mlir::Value value, int32_t &shift)
    {
        if (auto optionalOp = value.getDefiningOp<mlir_ts::OptionalValueOp>())
        {
            value = optionalOp.getIn();
        }

        while (auto castOp = value.getDefiningOp<mlir_ts::CastOp>())
        {
            if (!isFunctionValueType(castOp.getIn().getType()) || !isFunctionValueType(castOp.getType()))
            {
                return {};
            }

            value = castOp.getIn();
        }

        if (auto refOp = value.getDefiningOp<mlir_ts::SymbolRefOp>())
        {
            shift = 0;
            return refOp;
        }

        auto boundOp = value.getDefiningOp<mlir_ts::CreateBoundFunctionOp>();
        if (!boundOp || closureOfBox(boundOp.getThisVal()) != boundOp)
        {
            return {};
        }

        shift = 1;
        return boundOp.getFunc().getDefiningOp<mlir_ts::SymbolRefOp>();
    }

    bool OwnershipSignaturePass::passedOnly(mlir::Value value, llvm::function_ref<bool(mlir::OpOperand &)> passes, int depth)
    {
        if (depth > 8)
        {
            return false;
        }

        for (auto &use : value.getUses())
        {
            auto *user = use.getOwner();
            if (mlir::isa<mlir_ts::RetainOp, mlir_ts::ReleaseOp, mlir_ts::DebugVariableOp>(user) || passes(use))
            {
                continue;
            }

            auto castOp = mlir::dyn_cast<mlir_ts::CastOp>(user);
            auto optionalOp = mlir::dyn_cast<mlir_ts::OptionalValueOp>(user);
            auto boundOp = mlir::dyn_cast<mlir_ts::CreateBoundFunctionOp>(user);
            if (castOp && isFunctionValueType(castOp.getType()) && passedOnly(castOp.getResult(), passes, depth + 1))
            {
                continue;
            }

            if (optionalOp && passedOnly(optionalOp.getResult(), passes, depth + 1))
            {
                continue;
            }

            if (boundOp && boundOp.getFunc() == use.get() && closureOfBox(boundOp.getThisVal()) == boundOp &&
                passedOnly(boundOp.getResult(), passes, depth + 1))
            {
                continue;
            }

            return false;
        }

        return true;
    }

    // Is `use` a way of a value to its function's return only: the return, a store into the slot
    // only returned, a cast between function types or a branch to a block argument that is?
    static bool flowsToReturn(mlir::OpOperand &use, int depth = 0)
    {
        if (depth > 8)
        {
            return false;
        }

        auto *user = use.getOwner();
        if (mlir::isa<mlir_ts::ReturnInternalOp>(user))
        {
            return true;
        }

        if (auto storeOp = mlir::dyn_cast<mlir_ts::StoreOp>(user))
        {
            auto slot = storeOp.getReference().getDefiningOp<mlir_ts::VariableOp>();
            return storeOp.getValue() == use.get() && slot && !slot.getInitializer() && !slot.getCaptured().value_or(false) &&
                   llvm::all_of(slot->getUsers(), [&](mlir::Operation *slotUser) {
                       if (auto otherStore = mlir::dyn_cast<mlir_ts::StoreOp>(slotUser))
                       {
                           return otherStore.getReference() == slot.getResult();
                       }

                       auto loadOp = mlir::dyn_cast<mlir_ts::LoadOp>(slotUser);
                       return loadOp && llvm::all_of(loadOp->getUsers(), [](mlir::Operation *loadUser) {
                                  return mlir::isa<mlir_ts::ReturnInternalOp>(loadUser);
                              });
                   });
        }

        auto flowsOn = [&](mlir::Value next) {
            return llvm::all_of(next.getUses(), [&](mlir::OpOperand &nextUse) { return flowsToReturn(nextUse, depth + 1); });
        };

        if (auto castOp = mlir::dyn_cast<mlir_ts::CastOp>(user); castOp && isFunctionValue(castOp.getType()))
        {
            return flowsOn(castOp.getResult());
        }

        if (auto branchOp = mlir::dyn_cast<mlir::BranchOpInterface>(user))
        {
            auto passed = false;
            for (auto [index, successor] : llvm::enumerate(user->getSuccessors()))
            {
                auto operands = branchOp.getSuccessorOperands(index);
                for (auto [position, operand] : llvm::enumerate(operands.getForwardedOperands()))
                {
                    if (operand != use.get())
                    {
                        continue;
                    }

                    passed = true;
                    if (!flowsOn(successor->getArgument(operands.getProducedOperandCount() + position)))
                    {
                        return false;
                    }
                }
            }

            return passed;
        }

        return false;
    }

    bool OwnershipSignaturePass::takeFamilyValue(CallbackFamily &family, mlir::Value value, mlir::Operation *at)
    {
        auto shift = 0;
        if (auto refOp = passedFunction(value, shift))
        {
            auto member = functions.lookup(refOp.getIdentifier());
            if (!member || member.isDeclaration() || callableFromOutside(member) || declaresFacts(member))
            {
                return false;
            }

            auto symbol = member.getSymNameAttr();
            auto same = llvm::find_if(family.members, [&](const CallbackMember &known) { return known.symbol == symbol; });
            if (same != family.members.end() && same->shift != shift)
            {
                return false;
            }

            if (same == family.members.end())
            {
                family.members.push_back({symbol, shift});
            }

            family.passing.push_back(refOp);
            return true;
        }

        // an optional callback not given: no function
        if (mlir::isa_and_nonnull<mlir_ts::UndefOp, mlir_ts::OptionalUndefOp>(value.getDefiningOp()) &&
            mlir::isa<mlir_ts::OptionalType>(value.getType()))
        {
            return true;
        }

        // a callback parameter of the function `at` is in, given on: the functions passed for it
        auto position = -1;
        if (auto source = givenParam(value, position); source && source->isAncestor(at))
        {
            family.sources.push_back({source, position});
            return true;
        }

        // what a function that returns a function value returned (spec 56)
        if (auto callee = givenResult(value))
        {
            if (auto producer = functions.lookup(callee.getValue()))
            {
                family.sources.push_back({producer.getOperation(), RESULT_FAMILY});
                return true;
            }
        }

        return false;
    }

    void OwnershipSignaturePass::collectResultFamilies()
    {
        for (auto &entry : functions)
        {
            auto funcOp = entry.second;
            auto results = funcOp.getFunctionType().getResults();
            if (funcOp.isDeclaration() || results.size() != 1 || !isCallbackType(results.front()))
            {
                continue;
            }

            CallbackFamily family;
            llvm::SmallVector<mlir::Value> work(returnedValues(funcOp));
            llvm::DenseSet<mlir::Value> seen;
            while (family.closed && !work.empty())
            {
                auto value = work.pop_back_val();
                if (!seen.insert(value).second)
                {
                    continue;
                }

                // merged from branches: each value passed for the block's argument
                if (auto argument = mlir::dyn_cast<mlir::BlockArgument>(value); argument && !argument.getOwner()->isEntryBlock())
                {
                    llvm::SmallVector<mlir::Value> merged;
                    family.closed = mergedInto(argument, merged);
                    work.append(merged.begin(), merged.end());
                    continue;
                }

                family.closed = takeFamilyValue(family, value, funcOp);
            }

            callbackFamilies[{funcOp.getOperation(), RESULT_FAMILY}] = std::move(family);
        }
    }

    std::pair<mlir::Operation *, int32_t> OwnershipSignaturePass::fieldFamilyKey(mlir::Value ref)
    {
        auto fieldOp = ref.getDefiningOp<mlir_ts::PropertyRefOp>();
        auto classType = fieldOp ? mlir::dyn_cast<mlir_ts::ClassType>(fieldOp->getOperand(0).getType()) : mlir_ts::ClassType();
        auto constructor = classType ? functions.lookup((classType.getName().getValue() + ".constructor").str()) : mlir_ts::FuncOp();
        if (!constructor)
        {
            return {nullptr, -1};
        }

        return {constructor.getOperation(), -1 - static_cast<int32_t>(fieldOp.getPosition())};
    }

    void OwnershipSignaturePass::collectFieldFamilies(
        mlir::ModuleOp module, llvm::function_ref<std::pair<mlir::Operation *, int32_t>(mlir::OpOperand &)> parameterOf)
    {
        // the fields of a callback type, by class and position
        std::map<std::pair<mlir::Operation *, int32_t>, llvm::SmallVector<mlir_ts::PropertyRefOp>> fields;
        // the classes whose objects go where their fields may be reached otherwise: cast to an
        // interface, `any` or a union
        llvm::DenseSet<mlir::Type> escaping;
        module.walk([&](mlir::Operation *op) {
            if (auto fieldOp = mlir::dyn_cast<mlir_ts::PropertyRefOp>(op))
            {
                auto refType = mlir::dyn_cast<mlir_ts::RefType>(fieldOp.getType());
                auto key = fieldFamilyKey(fieldOp.getResult());
                if (key.first && refType && isCallbackType(refType.getElementType()))
                {
                    fields[key].push_back(fieldOp);
                }

                return;
            }

            auto reachedOtherwise = llvm::any_of(op->getResultTypes(), [](mlir::Type type) {
                return mlir::isa<mlir_ts::InterfaceType, mlir_ts::AnyType, mlir_ts::UnionType, mlir_ts::UnknownType,
                                 mlir_ts::ObjectType>(type);
            });
            for (auto operand : op->getOperands())
            {
                if (reachedOtherwise && mlir::isa<mlir_ts::ClassType>(operand.getType()))
                {
                    escaping.insert(operand.getType());
                }
            }
        });

        for (auto &[key, refs] : fields)
        {
            auto constructor = mlir::cast<mlir_ts::FuncOp>(key.first);
            auto classType = mlir::cast<mlir_ts::ClassType>(refs.front()->getOperand(0).getType());
            CallbackFamily family;

            // a class another module may build, or another class extends (its storage's first field
            // is this one's: the field is reached at another position there)
            family.closed = !callableFromOutside(constructor) && !escaping.contains(classType) &&
                            llvm::none_of(classStorages, [&](auto &entry) {
                                auto storageFields = entry.second.getFields();
                                return !storageFields.empty() && storageFields.front().type == classType.getStorageType();
                            });

            for (auto fieldOp : refs)
            {
                for (auto &use : fieldOp->getUses())
                {
                    auto *user = use.getOwner();
                    if (!family.closed)
                    {
                        break;
                    }

                    if (mlir::isa<mlir_ts::ReleaseSlotOp, mlir_ts::RetainSlotOp>(user))
                    {
                        continue;
                    }

                    // read only to be called, or given on
                    if (auto loadOp = mlir::dyn_cast<mlir_ts::LoadOp>(user))
                    {
                        llvm::SmallVector<mlir::OpOperand *> forwards;
                        family.closed = onlyCalledValue(loadOp.getResult(), forwards);
                        for (auto *forward : forwards)
                        {
                            auto target = parameterOf(*forward);
                            family.closed = family.closed && target.first;
                            family.forwards.push_back(target);
                        }

                        continue;
                    }

                    // given a function by name, a callback parameter of the function that stores it, or
                    // what a function returning one returned
                    auto storeOp = mlir::dyn_cast<mlir_ts::StoreOp>(user);
                    if (!storeOp || storeOp.getReference() != fieldOp.getResult())
                    {
                        family.closed = false;
                        break;
                    }

                    family.closed = takeFamilyValue(family, storeOp.getValue(), storeOp);
                }
            }

            callbackFamilies[key] = std::move(family);
        }
    }

    void OwnershipSignaturePass::collectCallbackFamilies(mlir::ModuleOp module)
    {
        capturedCallbacks.clear();
        auto uses = mlir::SymbolTable::getSymbolUses(&module.getBodyRegion());
        if (!uses)
        {
            return;
        }

        llvm::DenseMap<mlir::Operation *, llvm::SmallVector<mlir::Operation *>> usersOf;
        for (auto &use : *uses)
        {
            if (auto funcOp = functions.lookup(use.getSymbolRef().getRootReference().getValue()))
            {
                usersOf[funcOp].push_back(use.getUser());
            }
        }

        // A method no other method overrides: a call through its class's vtable, or of it bound to
        // its object (`b.forEach(f)`), reaches it alone. A family holds every method of a slot's
        // name, of any class: another member overrides it when its class extends this one (a base is
        // the first field of a class's storage), or may, when its storage is not known here.
        auto classOf = [&](llvm::StringRef symbol) { return symbol.drop_back(methodName(symbol).size() + 1); };
        auto extends = [&](llvm::StringRef derived, llvm::StringRef base) {
            auto storage = classStorages.lookup(derived);
            for (auto depth = 0; storage && depth < 64; ++depth)
            {
                auto fields = storage.getFields();
                auto next = fields.empty() ? mlir_ts::ClassStorageType() : mlir::dyn_cast<mlir_ts::ClassStorageType>(fields.front().type);
                if (!next)
                {
                    return false;
                }

                if (next.getName().getValue() == base)
                {
                    return true;
                }

                storage = next;
            }

            return true;
        };

        auto notOverridden = [&](mlir_ts::FuncOp funcOp) {
            auto symbol = funcOp.getSymNameAttr();
            auto inVTable = false;
            for (auto &[key, members] : families)
            {
                if (!llvm::is_contained(members, symbol))
                {
                    continue;
                }

                inVTable = true;
                for (auto member : members)
                {
                    if (member != symbol && extends(classOf(member.getValue()), classOf(symbol.getValue())))
                    {
                        return false;
                    }
                }
            }

            return inVTable;
        };

        // The calls a use of a function's symbol makes: the use itself, a direct call, or those made
        // through the method it binds. Null when the symbol goes anywhere else.
        auto callsOf = [&](mlir_ts::FuncOp funcOp, mlir::Operation *user, llvm::SmallVectorImpl<mlir::Operation *> &found) {
            if (auto callee = directCallee(user))
            {
                found.push_back(user);
                return callee.getValue() == funcOp.getSymName();
            }

            if (!mlir::isa<mlir_ts::ThisVirtualSymbolRefOp, mlir_ts::ThisSymbolRefOp>(user) || !notOverridden(funcOp) ||
                !onlyCalled(user->getResult(0)))
            {
                return isClassVTableEntry(user) && notOverridden(funcOp);
            }

            // `ts.CallInternal(ts.GetMethod(m), ts.GetThis(m), ...)`: its arguments are the method's
            // parameters, `this` first
            for (auto *boundUser : user->getResult(0).getUsers())
            {
                if (mlir::isa<mlir_ts::GetThisOp>(boundUser))
                {
                    continue;
                }

                if (!mlir::isa<mlir_ts::GetMethodOp>(boundUser))
                {
                    return false;
                }

                for (auto *call : boundUser->getResult(0).getUsers())
                {
                    auto args = callArgs(call);
                    if (!mlir::isa<mlir_ts::CallInternalOp>(call) || args.empty() ||
                        !args.front().getDefiningOp<mlir_ts::GetThisOp>())
                    {
                        return false;
                    }

                    found.push_back(call);
                }
            }

            return true;
        };

        // The function a call made through a method's symbol calls, when it is the only one
        auto methodCalled = [&](mlir::Operation *call) {
            auto getMethodOp = calleeValue(call) ? calleeValue(call).getDefiningOp<mlir_ts::GetMethodOp>() : mlir_ts::GetMethodOp();
            auto *ref = getMethodOp ? getMethodOp.getBoundFunc().getDefiningOp() : nullptr;
            mlir::FlatSymbolRefAttr identifier;
            if (auto refOp = mlir::dyn_cast_or_null<mlir_ts::ThisVirtualSymbolRefOp>(ref))
            {
                identifier = refOp.getIdentifierAttr();
            }
            else if (auto refOp = mlir::dyn_cast_or_null<mlir_ts::ThisSymbolRefOp>(ref))
            {
                identifier = refOp.getIdentifierAttr();
            }

            auto funcOp = identifier ? functions.lookup(identifier.getValue()) : mlir_ts::FuncOp();
            return funcOp && notOverridden(funcOp) ? funcOp : mlir_ts::FuncOp();
        };

        auto callsOnly = [&](mlir_ts::FuncOp funcOp, llvm::SmallVectorImpl<mlir::Operation *> &calls) {
            auto &users = usersOf[funcOp];
            return !users.empty() &&
                   llvm::all_of(users, [&](mlir::Operation *user) { return callsOf(funcOp, user, calls); });
        };

        // The parameter of the one function a call reaches that `use` is the argument for, by its
        // operand's position; null when the call reaches no one function or `use` is not an argument.
        auto parameterOf = [&](mlir::OpOperand &use) -> std::pair<mlir::Operation *, int32_t> {
            auto *call = use.getOwner();
            // ...or the result of the function it is returned from (spec 56)
            if (flowsToReturn(use))
            {
                auto funcOp = call->getParentOfType<mlir_ts::FuncOp>();
                return {funcOp ? funcOp.getOperation() : nullptr, RESULT_FAMILY};
            }

            // ...or the field of a class it is stored into (spec 55)
            if (auto storeOp = mlir::dyn_cast<mlir_ts::StoreOp>(call))
            {
                return fieldFamilyKey(storeOp.getReference());
            }

            if (!isCall(call))
            {
                return {nullptr, -1};
            }

            auto callee = directCallee(call);
            auto calleeOp = callee ? functions.lookup(callee.getValue()) : methodCalled(call);
            auto args = callArgs(call);
            if (!calleeOp || args.empty() || use.getOperandNumber() < args.getBeginOperandIndex() ||
                use.getOperandNumber() >= args.getBeginOperandIndex() + args.size())
            {
                return {nullptr, -1};
            }

            return {calleeOp, static_cast<int32_t>(use.getOperandNumber() - args.getBeginOperandIndex())};
        };

        for (auto &entry : functions)
        {
            auto funcOp = entry.second;
            llvm::SmallVector<mlir::Operation *> calls;
            if (funcOp.isDeclaration() || callableFromOutside(funcOp) || declaresFacts(funcOp) || !callsOnly(funcOp, calls))
            {
                continue;
            }

            for (auto argument : funcOp.getBody().front().getArguments())
            {
                llvm::SmallVector<mlir::OpOperand *> forwards;
                llvm::SmallVector<std::pair<mlir::Operation *, int32_t>> captured;
                if (!isCallbackType(argument.getType()) || !onlyCalledParam(argument, forwards, captured))
                {
                    continue;
                }

                auto index = static_cast<int32_t>(argument.getArgNumber());
                for (auto field : captured)
                {
                    capturedCallbacks[field] = {funcOp.getOperation(), index};
                }

                CallbackFamily family;
                for (auto *use : forwards)
                {
                    auto target = parameterOf(*use);
                    family.closed = family.closed && target.first;
                    family.forwards.push_back(target);
                }

                for (auto *user : calls)
                {
                    if (!family.closed)
                    {
                        break;
                    }

                    // a function by name, an optional one not given, the caller's own callback
                    // parameter given on, or what a function returning one returned
                    auto args = callArgs(user);
                    auto arg = static_cast<size_t>(index) < args.size() ? args[index] : mlir::Value();
                    family.closed = arg && takeFamilyValue(family, arg, user);
                }

                callbackFamilies[{funcOp, index}] = std::move(family);
            }
        }

        collectFieldFamilies(module, parameterOf);
        collectResultFamilies();

        // A member's symbol used anywhere else - a call through another value, a field, a vtable, a
        // global - reaches callers this cannot see; so does one passed to a family that is not closed.
        // A parameter given on is closed only while the parameters it is given from and on to are:
        // the functions passed for those are its own, and the calls it gives them to must resolve.
        // Each family that has such a member is open, until nothing changes.
        auto closedFamily = [&](std::pair<mlir::Operation *, int32_t> key) -> CallbackFamily * {
            auto found = callbackFamilies.find(key);
            return found != callbackFamilies.end() && found->second.closed ? &found->second : nullptr;
        };

        auto passesToClosed = [&](mlir::OpOperand &use) {
            auto key = parameterOf(use);
            return key.first && closedFamily(key);
        };

        // Takes in the members of the parameters given here: false when one is not closed, or gives
        // a member with another shift.
        auto mergeSources = [&](CallbackFamily &family, bool &changed) {
            for (auto source : family.sources)
            {
                auto *from = closedFamily(source);
                if (!from)
                {
                    return false;
                }

                if (from == &family)
                {
                    continue;
                }

                for (auto &member : from->members)
                {
                    auto same = llvm::find_if(family.members, [&](const CallbackMember &known) { return known.symbol == member.symbol; });
                    if (same != family.members.end() && same->shift != member.shift)
                    {
                        return false;
                    }

                    if (same == family.members.end())
                    {
                        family.members.push_back(member);
                        changed = true;
                    }
                }

                for (auto *refOp : from->passing)
                {
                    if (!llvm::is_contained(family.passing, refOp))
                    {
                        family.passing.push_back(refOp);
                        changed = true;
                    }
                }
            }

            return true;
        };

        for (auto changed = true; changed;)
        {
            changed = false;
            for (auto &[key, family] : callbackFamilies)
            {
                if (!family.closed)
                {
                    continue;
                }

                if (!mergeSources(family, changed) ||
                    llvm::any_of(family.forwards, [&](auto target) { return !closedFamily(target); }))
                {
                    family.closed = false;
                    changed = true;
                    continue;
                }

                for (auto &member : family.members)
                {
                    auto memberOp = functions.lookup(member.symbol.getValue());
                    auto escapes = llvm::any_of(usersOf[memberOp], [&](mlir::Operation *user) {
                        if (auto callee = directCallee(user); callee && callee.getValue() == member.symbol.getValue())
                        {
                            return false;
                        }

                        auto refOp = mlir::dyn_cast<mlir_ts::SymbolRefOp>(user);
                        return !refOp || !user->getParentOfType<mlir::FunctionOpInterface>() ||
                               !passedOnly(refOp.getResult(), passesToClosed);
                    });

                    if (escapes)
                    {
                        family.closed = false;
                        changed = true;
                        break;
                    }
                }
            }
        }

        for (auto &[key, family] : callbackFamilies)
        {
            if (family.closed)
            {
                callbackPassing.insert(family.passing.begin(), family.passing.end());
            }
        }
    }

    void OwnershipSignaturePass::resolveCallback(mlir::Value callee, Callees &callees)
    {
        auto loadOp = callee.getDefiningOp<mlir_ts::LoadOp>();
        auto producer = givenResult(callee);
        if (!loadOp && !producer)
        {
            return;
        }

        auto varOp = loadOp ? loadOp.getReference().getDefiningOp<mlir_ts::VariableOp>() : mlir_ts::VariableOp();
        auto funcOp = callee.getParentRegion()->getParentOfType<mlir_ts::FuncOp>();
        auto index = -1;
        std::pair<mlir::Operation *, int32_t> key{funcOp.getOperation(), -1};
        if (producer && functions.lookup(producer.getValue()))
        {
            // what a function that returns a function value returned, called as it is or through a
            // local declared from it (spec 56)
            key = {functions.lookup(producer.getValue()).getOperation(), RESULT_FAMILY};
        }
        else if (!loadOp)
        {
            return;
        }
        else if (varOp && funcOp && isParameterSlot(varOp, index))
        {
            key.second = index;
        }
        else if (auto argument = varOp && varOp.getCaptured().value_or(false) && varOp.getInitializer()
                                     ? mlir::dyn_cast<mlir::BlockArgument>(varOp.getInitializer())
                                     : mlir::BlockArgument();
                 argument && argument.getOwner()->isEntryBlock() && argument.getOwner()->getParentOp() == funcOp.getOperation())
        {
            // the captured parameter's cell, read where it is declared (spec 54)
            key.second = static_cast<int32_t>(argument.getArgNumber());
        }
        else if (auto fieldKey = fieldFamilyKey(loadOp.getReference()); fieldKey.first)
        {
            // a class's field of a callback type (spec 55)
            key = fieldKey;
        }
        else if (auto cellRead = loadOp.getReference().getDefiningOp<mlir_ts::LoadOp>())
        {
            // read out of a closure's capture box: the parameter that field holds the cell of
            auto fieldOp = cellRead.getReference().getDefiningOp<mlir_ts::PropertyRefOp>();
            auto box = fieldOp ? mlir::dyn_cast<mlir::BlockArgument>(fieldOp->getOperand(0)) : mlir::BlockArgument();
            auto found = box && box.getArgNumber() == 0 && box.getOwner()->isEntryBlock()
                             ? capturedCallbacks.find({funcOp.getOperation(), static_cast<int32_t>(fieldOp.getPosition())})
                             : capturedCallbacks.end();
            if (found == capturedCallbacks.end())
            {
                return;
            }

            key = found->second;
        }
        else
        {
            return;
        }

        // a closed family with no members: a method no one calls but through its vtable, never - the
        // call frees nothing
        auto found = callbackFamilies.find(key);
        if (found == callbackFamilies.end() || !found->second.closed)
        {
            return;
        }

        callees.viaCallback = true;
        callees.known = true;
        for (auto &member : found->second.members)
        {
            callees.named.push_back(member.symbol);
            callees.known = addDefined(member.symbol, callees) && callees.known;
            callees.shifts.push_back(member.shift);
        }

        callees.known = callees.known && callees.shifts.size() == callees.funcs.size();
    }

    bool OwnershipSignaturePass::shiftFacts(const Facts &facts, int32_t shift, Facts &shifted)
    {
        shifted = Facts();
        shifted.record = facts.record;
        shifted.unboxed = facts.unboxed;
        for (auto [from, to] : {std::make_pair(&facts.owned, &shifted.owned), std::make_pair(&facts.borrows, &shifted.borrows)})
        {
            for (auto index : *from)
            {
                if (index < shift)
                {
                    return false;
                }

                to->push_back(index - shift);
            }
        }

        return true;
    }

    bool OwnershipSignaturePass::canReleaseParam(mlir_ts::FuncOp funcOp, int32_t index)
    {
        if (funcOp.getBody().empty() || static_cast<unsigned>(index) >= funcOp.getNumArguments())
        {
            return false;
        }

        auto argument = funcOp.getBody().front().getArgument(index);
        if (isHandle(argument) || isCopyableString(argument.getType()) || holdsNoBlock(argument))
        {
            return false;
        }

        return llvm::any_of(argument.getUsers(), [&](mlir::Operation *user) {
            auto varOp = mlir::dyn_cast<mlir_ts::VariableOp>(user);
            auto position = -1;
            return varOp && isParameterSlot(varOp, position) && position == index;
        });
    }

    bool OwnershipSignaturePass::meetCallbacks(llvm::DenseSet<mlir::Operation *> &demoted)
    {
        auto changed = false;
        for (auto &[key, family] : callbackFamilies)
        {
            if (!family.closed)
            {
                continue;
            }

            llvm::SmallVector<mlir_ts::FuncOp, 2> members;
            llvm::SmallVector<Facts, 2> seen;
            auto shiftable = true;
            for (auto &member : family.members)
            {
                auto funcOp = functions.lookup(member.symbol.getValue());
                Facts shifted;
                shiftable = shiftable && funcOp && shiftFacts(factsEffective.lookup(funcOp), member.shift, shifted);
                members.push_back(funcOp);
                seen.push_back(shifted);
            }

            if (shiftable && llvm::all_equal(seen))
            {
                continue;
            }

            // a parameter some member keeps, and each other one only borrows: kept by all
            llvm::SmallVector<int32_t> kept;
            for (auto &facts : seen)
            {
                kept.append(facts.owned.begin(), facts.owned.end());
            }

            llvm::sort(kept);
            kept.erase(std::unique(kept.begin(), kept.end()), kept.end());
            auto promotable = shiftable && llvm::all_of(seen, [&](const Facts &facts) {
                return facts.borrows == seen.front().borrows && facts.record == seen.front().record &&
                       facts.unboxed == seen.front().unboxed &&
                       llvm::none_of(kept, [&](int32_t index) { return llvm::is_contained(facts.borrows, index); });
            });

            for (auto [position, member] : llvm::enumerate(family.members))
            {
                auto funcOp = members[position];
                // a member another caller reaches without the facts is given none (findOpen)
                for (auto index : kept)
                {
                    promotable = promotable && !demoted.contains(funcOp) && !open.contains(funcOp) &&
                                 !interfaceReached.contains(funcOp) &&
                                 (llvm::is_contained(seen[position].owned, index) ||
                                  canReleaseParam(funcOp, index + member.shift));
                }
            }

            if (promotable)
            {
                for (auto [position, member] : llvm::enumerate(family.members))
                {
                    auto funcOp = members[position];
                    auto facts = factsEffective.lookup(funcOp);
                    for (auto index : kept)
                    {
                        auto param = index + member.shift;
                        if (!llvm::is_contained(facts.owned, param))
                        {
                            facts.owned.insert(llvm::upper_bound(facts.owned, param), param);
                        }
                    }

                    if (facts != factsEffective.lookup(funcOp))
                    {
                        factsEffective[funcOp] = facts;
                        changed = true;
                    }
                }

                continue;
            }

            for (auto funcOp : members)
            {
                if (funcOp && !declaresCallerFacts(funcOp))
                {
                    demoted.insert(funcOp);
                    if (factsEffective.erase(funcOp))
                    {
                        changed = true;
                    }
                }
            }
        }

        return changed;
    }

} // namespace own_signature
