// OwnershipSignaturePass (see OwnershipSignaturePass.h): callbacks (spec 33) - the functions passed
// for a parameter, and the facts their families agree on.

#include "OwnershipSignaturePass.h"

#define DEBUG_TYPE "own"

namespace own_signature
{

    bool OwnershipSignaturePass::isFunctionValueType(mlir::Type type)
    {
        return mlir::isa<mlir_ts::FunctionType, mlir_ts::HybridFunctionType, mlir_ts::BoundFunctionType>(type);
    }

    bool OwnershipSignaturePass::onlyCalledParam(mlir::BlockArgument argument)
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
                if (!loadOp)
                {
                    return false;
                }

                for (auto &use : loadOp->getUses())
                {
                    auto *call = use.getOwner();
                    if (!isCall(call) || calleeValue(call) != use.get() ||
                        llvm::is_contained(callArgs(call), use.get()))
                    {
                        return false;
                    }
                }
            }
        }

        return true;
    }

    mlir_ts::SymbolRefOp OwnershipSignaturePass::passedFunction(mlir::Value value, int32_t &shift)
    {
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
            auto boundOp = mlir::dyn_cast<mlir_ts::CreateBoundFunctionOp>(user);
            if (castOp && isFunctionValueType(castOp.getType()) && passedOnly(castOp.getResult(), passes, depth + 1))
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

        auto callsOnly = [&](mlir_ts::FuncOp funcOp) {
            auto &users = usersOf[funcOp];
            return !users.empty() && llvm::all_of(users, [&](mlir::Operation *user) {
                auto callee = directCallee(user);
                return callee && callee.getValue() == funcOp.getSymName();
            });
        };

        for (auto &entry : functions)
        {
            auto funcOp = entry.second;
            if (funcOp.isDeclaration() || callableFromOutside(funcOp) || declaresFacts(funcOp) || !callsOnly(funcOp))
            {
                continue;
            }

            for (auto argument : funcOp.getBody().front().getArguments())
            {
                if (!isFunctionValueType(argument.getType()) || !onlyCalledParam(argument))
                {
                    continue;
                }

                auto index = static_cast<int32_t>(argument.getArgNumber());
                CallbackFamily family;
                for (auto *user : usersOf[funcOp])
                {
                    auto args = callArgs(user);
                    auto shift = 0;
                    auto refOp = static_cast<size_t>(index) < args.size() ? passedFunction(args[index], shift)
                                                                          : mlir_ts::SymbolRefOp();
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
        // Each family that has such a member is open, until nothing changes.
        auto passesToClosed = [&](mlir::OpOperand &use) {
            auto *call = use.getOwner();
            auto callee = directCallee(call);
            auto calleeOp = callee ? functions.lookup(callee.getValue()) : mlir_ts::FuncOp();
            if (!calleeOp)
            {
                return false;
            }

            // the argument this use is, by its operand's position
            auto args = callArgs(call);
            if (args.empty() || use.getOperandNumber() < args.getBeginOperandIndex() ||
                use.getOperandNumber() >= args.getBeginOperandIndex() + args.size())
            {
                return false;
            }

            auto index = static_cast<int32_t>(use.getOperandNumber() - args.getBeginOperandIndex());
            auto found = callbackFamilies.find({calleeOp, index});
            return found != callbackFamilies.end() && found->second.closed;
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

        auto found = callbackFamilies.find({funcOp.getOperation(), index});
        if (found == callbackFamilies.end() || !found->second.closed || found->second.members.empty())
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
