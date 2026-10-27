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
    // (`if (f)`), unwrapped, or an argument of a call that gives it on (`forwards`)
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

    bool OwnershipSignaturePass::onlyCalledParam(mlir::BlockArgument argument, llvm::SmallVectorImpl<mlir::OpOperand *> &forwards)
    {
        for (auto *user : argument.getUsers())
        {
            auto varOp = mlir::dyn_cast<mlir_ts::VariableOp>(user);
            auto index = -1;
            if (!varOp || !isParameterSlot(varOp, index))
            {
                return false;
            }

            for (auto *slotUser : varOp->getUsers())
            {
                if (mlir::isa<mlir_ts::DebugVariableOp>(slotUser))
                {
                    continue;
                }

                auto loadOp = mlir::dyn_cast<mlir_ts::LoadOp>(slotUser);
                if (!loadOp || !onlyCalledValue(loadOp.getResult(), forwards))
                {
                    return false;
                }
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

    void OwnershipSignaturePass::collectCallbackFamilies(mlir::ModuleOp module)
    {
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
                if (!isCallbackType(argument.getType()) || !onlyCalledParam(argument, forwards))
                {
                    continue;
                }

                auto index = static_cast<int32_t>(argument.getArgNumber());
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

                    auto args = callArgs(user);
                    auto arg = static_cast<size_t>(index) < args.size() ? args[index] : mlir::Value();
                    auto shift = 0;
                    auto refOp = arg ? passedFunction(arg, shift) : mlir_ts::SymbolRefOp();
                    if (!refOp && arg)
                    {
                        // an optional callback not given: no function
                        if (mlir::isa_and_nonnull<mlir_ts::UndefOp, mlir_ts::OptionalUndefOp>(arg.getDefiningOp()) &&
                            mlir::isa<mlir_ts::OptionalType>(arg.getType()))
                        {
                            continue;
                        }

                        // the caller's own callback parameter, given on: the functions passed for it
                        auto position = -1;
                        if (auto source = givenParam(arg, position); source && source->isAncestor(user))
                        {
                            family.sources.push_back({source, position});
                            continue;
                        }
                    }

                    auto member = refOp ? functions.lookup(refOp.getIdentifier()) : mlir_ts::FuncOp();
                    if (!member || member.isDeclaration() || callableFromOutside(member) || declaresFacts(member))
                    {
                        family.closed = false;
                        break;
                    }

                    auto symbol = member.getSymNameAttr();
                    auto same = llvm::find_if(family.members, [&](const CallbackMember &known) { return known.symbol == symbol; });
                    if (same != family.members.end() && same->shift != shift)
                    {
                        family.closed = false;
                        break;
                    }

                    if (same == family.members.end())
                    {
                        family.members.push_back({symbol, shift});
                    }

                    family.passing.push_back(refOp);
                }

                callbackFamilies[{funcOp, index}] = std::move(family);
            }
        }

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

    void OwnershipSignaturePass::resolveCallback(mlir_ts::LoadOp loadOp, Callees &callees)
    {
        auto varOp = loadOp.getReference().getDefiningOp<mlir_ts::VariableOp>();
        auto funcOp = loadOp->getParentOfType<mlir_ts::FuncOp>();
        auto index = -1;
        if (!varOp || !funcOp || !isParameterSlot(varOp, index))
        {
            return;
        }

        // a closed family with no members: a method no one calls but through its vtable, never - the
        // call frees nothing
        auto found = callbackFamilies.find({funcOp.getOperation(), index});
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
