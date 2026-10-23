// OwnershipInferencePass (see OwnershipInferencePass.h): a move on some paths only (spec 30) -
// the release placed on the edges where the paths that did not move join the ones that did.

#include "OwnershipInferencePass.h"

#define DEBUG_TYPE "own"

namespace own_inference
{

    bool OwnershipInferencePass::isJoined(unsigned state)
    {
        return (state & SlotLive) && (state & SlotMoved);
    }

    bool OwnershipInferencePass::carriesPointer(mlir::Type type)
    {
        return !mlir::isa<mlir_ts::NumberType, mlir_ts::BooleanType, mlir_ts::LiteralType, mlir_ts::UndefinedType,
                          mlir_ts::NullType, mlir_ts::VoidType, mlir::IntegerType, mlir::IndexType, mlir::FloatType>(type);
    }

    bool OwnershipInferencePass::reachableFrom(mlir::Block *start, mlir::Operation *to, llvm::ArrayRef<mlir::Operation *> kills)
    {
        auto &body = getFunction().getFunctionBody();
        to = body.findAncestorOpInRegion(*to);
        if (!to)
        {
            return true;
        }

        llvm::SmallVector<mlir::Operation *, 2> killAt;
        for (auto *kill : kills)
        {
            if (auto *top = body.findAncestorOpInRegion(*kill))
            {
                killAt.push_back(top);
            }
        }

        llvm::SmallPtrSet<mlir::Block *, 16> seen;
        llvm::SmallVector<mlir::Block *> work{start};
        while (!work.empty())
        {
            auto *block = work.pop_back_val();
            if (!seen.insert(block).second)
            {
                continue;
            }

            auto killed = [&](mlir::Operation *before) {
                return llvm::any_of(killAt, [&](mlir::Operation *kill) {
                    return kill->getBlock() == block && (!before || kill->isBeforeInBlock(before) || kill == before);
                });
            };

            if (to->getBlock() == block && !killed(to))
            {
                return true;
            }

            if (!killed(nullptr))
            {
                work.append(block->getSuccessors().begin(), block->getSuccessors().end());
            }
        }

        return false;
    }

    llvm::SmallPtrSet<mlir::Block *, 8> OwnershipInferencePass::handlerBlocks()
    {
        llvm::SmallPtrSet<mlir::Block *, 8> handler;
        llvm::SmallVector<mlir::Block *> work;
        for (auto &block : getFunction().getFunctionBody())
        {
            if (!block.empty() && mlir::isa<mlir_ts::LandingPadOp>(block.front()))
            {
                work.push_back(&block);
            }
        }

        while (!work.empty())
        {
            auto *block = work.pop_back_val();
            if (!handler.insert(block).second)
            {
                continue;
            }

            auto ends = llvm::any_of(*block, [](mlir::Operation &op) {
                return mlir::isa<mlir_ts::EndCatchOp, mlir_ts::EndCleanupOp>(op);
            });
            if (!ends)
            {
                work.append(block->getSuccessors().begin(), block->getSuccessors().end());
            }
        }

        return handler;
    }

    bool OwnershipInferencePass::placeOnUnmovedEdges(mlir_ts::VariableOp varOp, mlir::Operation *acquire, mlir::Operation *retain,
                             mlir::Operation *taker, llvm::ArrayRef<mlir::Operation *> assignments,
                             llvm::SetVector<mlir::Operation *> &toErase)
    {
        joinedUse = nullptr;
        llvm::SmallVector<mlir::Operation *> releases;
        if (!slotReleasesInBody(varOp, acquire, assignments, releases))
        {
            return false;
        }

        auto states = slotStates(varOp, acquire, assignments, releases);

        // every release a moved value reaches goes; one some path reaches with nothing in the slot
        // is not this rule's
        llvm::SmallVector<mlir::Operation *> erased;
        for (auto *release : releases)
        {
            auto state = states.before.lookup(release);
            if (state & SlotMoved)
            {
                if (state & SlotDead)
                {
                    return false;
                }

                erased.push_back(release);
            }
        }

        llvm::SmallVector<EdgeRelease> placed;
        llvm::SetVector<mlir::Block *> releasedInto;
        if (!unmovedEdges(varOp, states, placed, releasedInto))
        {
            return false;
        }

        if (!noUseWhereJoined(varOp, retain, taker, assignments, states, releasedInto))
        {
            return false;
        }

        toErase.insert(erased.begin(), erased.end());
        edgeReleases.append(placed.begin(), placed.end());
        return true;
    }

    bool OwnershipInferencePass::slotReleasesInBody(mlir_ts::VariableOp varOp, mlir::Operation *acquire,
                            llvm::ArrayRef<mlir::Operation *> assignments, llvm::SmallVectorImpl<mlir::Operation *> &releases)
    {
        auto &body = getFunction().getFunctionBody();
        if (!varOp || varOp.getCaptured().value_or(false) || varOp->getParentRegion() != &body ||
            acquire->getParentRegion() != &body ||
            llvm::any_of(assignments, [&](mlir::Operation *op) { return op->getParentRegion() != &body; }))
        {
            return false;
        }

        auto slot = varOp.getResult();
        for (auto *user : slot.getUsers())
        {
            if (mlir::isa<mlir_ts::ReleaseSlotOp>(user))
            {
                if (user->getParentRegion() != &body)
                {
                    return false;
                }

                releases.push_back(user);
            }
        }

        return true;
    }

    OwnershipInferencePass::SlotStates OwnershipInferencePass::slotStates(mlir_ts::VariableOp varOp, mlir::Operation *acquire, llvm::ArrayRef<mlir::Operation *> assignments,
                          llvm::ArrayRef<mlir::Operation *> releases)
    {
        auto &body = getFunction().getFunctionBody();
        auto step = [&](mlir::Operation *op, unsigned state) -> unsigned {
            if (op == varOp.getOperation() || llvm::is_contained(assignments, op))
            {
                return SlotLive;
            }

            if (op == acquire)
            {
                return SlotMoved;
            }

            if (llvm::is_contained(releases, op))
            {
                return SlotDead;
            }

            return state;
        };

        SlotStates states;
        auto &entry = states.entry;
        auto exitOf = [&](mlir::Block *block) {
            auto state = entry.lookup(block);
            for (auto &op : *block)
            {
                state = step(&op, state);
            }

            return state;
        };

        entry[&body.front()] = SlotDead;
        llvm::SmallVector<mlir::Block *> work{&body.front()};
        while (!work.empty())
        {
            auto *block = work.pop_back_val();
            auto state = exitOf(block);
            for (auto *successor : block->getSuccessors())
            {
                auto &into = entry[successor];
                if ((into | state) != into)
                {
                    into |= state;
                    work.push_back(successor);
                }
            }
        }

        // ...and before each op of the body
        for (auto &block : body)
        {
            auto state = entry.lookup(&block);
            for (auto &op : block)
            {
                states.before[&op] = state;
                state = step(&op, state);
            }

            states.exit[&block] = state;
        }

        return states;
    }

    bool OwnershipInferencePass::unmovedEdges(mlir_ts::VariableOp varOp, const SlotStates &states, llvm::SmallVectorImpl<EdgeRelease> &placed,
                      llvm::SetVector<mlir::Block *> &releasedInto)
    {
        auto &body = getFunction().getFunctionBody();
        auto slot = varOp.getResult();
        auto &entry = states.entry;

        // the edges into the join that hold the value and did not move it
        auto handler = handlerBlocks();
        for (auto &block : body)
        {
            if (!entry.count(&block))
            {
                continue; // unreachable
            }

            auto state = states.exit.lookup(&block);
            for (auto index : llvm::seq<unsigned>(0, block.getNumSuccessors()))
            {
                auto into = entry.lookup(block.getSuccessor(index));
                if (!isJoined(into))
                {
                    continue;
                }

                // a join some path reaches with nothing in the slot is not this rule's
                if (into & SlotDead)
                {
                    return false;
                }

                // an edge that moved it, or from where the paths have joined already
                if (state != SlotLive)
                {
                    continue;
                }

                auto *terminator = block.getTerminator();
                if (!mlir::isa<mlir::cf::BranchOp, mlir::cf::CondBranchOp>(terminator) || handler.contains(&block) ||
                    !dominance->dominates(varOp.getOperation(), terminator))
                {
                    return false;
                }

                placed.push_back({slot, &block, static_cast<unsigned>(index), terminator->getLoc()});
                releasedInto.insert(block.getSuccessor(index));
            }
        }

        return !placed.empty();
    }

    bool OwnershipInferencePass::noUseWhereJoined(mlir_ts::VariableOp varOp, mlir::Operation *retain, mlir::Operation *taker,
                          llvm::ArrayRef<mlir::Operation *> assignments, const SlotStates &states,
                          const llvm::SetVector<mlir::Block *> &releasedInto)
    {
        auto &body = getFunction().getFunctionBody();
        auto slot = varOp.getResult();

        auto joinedAt = [&](mlir::Operation *user) {
            auto *top = body.findAncestorOpInRegion(*user);
            return !top || isJoined(states.before.lookup(top));
        };

        // Nothing uses the value where the paths have joined - a read there, on the path that did
        // not move it, needs what the edge released - neither through the slot nor through anything
        // read out of it: a field, a view, a borrowing local, a closure. The move itself hands the
        // value to its new owner, whose reads are its own. A value the slot is given - its
        // initializer, an assignment's - is the slot's only once it is stored: a use before that (a
        // constructor's, or one around a loop before it is made again) is not a use of it.
        //
        // Past the joined points too: what was read out of the value before an edge released it
        // may not be used after that edge - after a new value is stored into the slot, or after its
        // scope ends - unless it was read out again (spec 30). A read of the slot after a new value
        // is stored reads that one.
        struct Origin
        {
            mlir::Operation *stored;
            mlir::Operation *made;
        };

        llvm::SmallVector<Origin> origins{{nullptr, nullptr}};
        llvm::SmallVector<std::pair<mlir::Value, unsigned>> values;
        llvm::DenseSet<std::pair<mlir::Value, unsigned>> seen;
        auto taint = [&](mlir::Value value, unsigned origin) {
            if (carriesPointer(value.getType()) && seen.insert({value, origin}).second)
            {
                values.push_back({value, origin});
            }
        };

        auto given = [&](mlir::Value value, mlir::Operation *stored) {
            origins.push_back({stored, value.getDefiningOp()});
            taint(value, origins.size() - 1);
        };

        taint(slot, 0);
        if (varOp.getInitializer())
        {
            given(varOp.getInitializer(), varOp);
        }

        for (auto *assignment : assignments)
        {
            given(mlir::cast<mlir_ts::StoreOp>(assignment).getValue(), assignment);
        }

        // reached from an edge that released the value, before the read it came from (or the op
        // that made it) runs again
        auto usedAfterRelease = [&](mlir::Operation *made, mlir::Operation *user) {
            llvm::SmallVector<mlir::Operation *, 1> madeAgain;
            if (made)
            {
                madeAgain.push_back(made);
            }

            return llvm::any_of(releasedInto, [&](mlir::Block *join) { return reachableFrom(join, user, madeAgain); });
        };

        auto heldAt = [&](unsigned origin, mlir::Operation *user) {
            auto &from = origins[origin];
            if (!from.stored)
            {
                return true;
            }

            llvm::SmallVector<mlir::Operation *, 1> madeAgain;
            if (from.made)
            {
                madeAgain.push_back(from.made);
            }

            return reachableAfter(from.stored, user, madeAgain);
        };

        while (!values.empty())
        {
            auto [value, origin] = values.pop_back_val();
            for (auto &use : value.getUses())
            {
                auto *user = use.getOwner();
                if (mlir::isa<mlir_ts::DebugVariableOp>(user) || user == retain)
                {
                    continue;
                }

                // the slot's own releases and assignments are decided above
                if (value == slot && (mlir::isa<mlir_ts::ReleaseSlotOp>(user) || llvm::is_contained(assignments, user)))
                {
                    continue;
                }

                if (joinedAt(user) && heldAt(origin, user))
                {
                    joinedUse = user;
                    return false;
                }

                if (value != slot && usedAfterRelease(origins[origin].made, user))
                {
                    joinedUse = user;
                    return false;
                }

                if (user == taker)
                {
                    continue;
                }

                // a store of it into a local makes every read of that local one; one into a place
                // is not followed, and the release is not placed
                if (auto storeOp = mlir::dyn_cast<mlir_ts::StoreOp>(user); storeOp && storeOp.getValue() == value)
                {
                    auto into = storeOp.getReference().getDefiningOp<mlir_ts::VariableOp>();
                    if (!into || into.getCaptured().value_or(false))
                    {
                        return false;
                    }

                    taint(into.getResult(), origin);
                    continue;
                }

                if (auto branch = mlir::dyn_cast<mlir::BranchOpInterface>(user))
                {
                    for (auto index : llvm::seq<unsigned>(0, user->getNumSuccessors()))
                    {
                        auto operands = branch.getSuccessorOperands(index);
                        for (auto [position, operand] : llvm::enumerate(operands.getForwardedOperands()))
                        {
                            if (operand == value)
                            {
                                taint(user->getSuccessor(index)->getArgument(operands.getProducedOperandCount() + position),
                                      origin);
                            }
                        }
                    }
                }

                // a call's result is its own block, unless its facts say it borrows this argument - a
                // plain borrow, or a borrowing record that holds it (spec 32)
                auto borrows = true;
                if (isCall(user))
                {
                    auto args = callArgs(user);
                    borrows = llvm::any_of(resultBorrowsAll(user), [&](int index) {
                        return static_cast<size_t>(index) < args.size() && args[index] == value;
                    });
                }

                // a read of the slot is what its uses came from
                auto from = origin;
                if (value == slot && user->getNumResults() > 0)
                {
                    origins.push_back({nullptr, user});
                    from = origins.size() - 1;
                }

                for (auto result : user->getResults())
                {
                    if (borrows)
                    {
                        taint(result, from);
                    }
                }

                // inside a region - a closure, an async body - it is the op that holds the region
                // that hands it on
                if (user->getParentRegion() != &body)
                {
                    for (auto result : body.findAncestorOpInRegion(*user)->getResults())
                    {
                        taint(result, origin);
                    }
                }
            }
        }

        return true;
    }

    void OwnershipInferencePass::placeEdgeReleases()
    {
        for (auto &edge : edgeReleases)
        {
            auto *terminator = edge.from->getTerminator();
            mlir::OpBuilder builder(terminator);
            if (terminator->getNumSuccessors() == 1)
            {
                builder.create<mlir_ts::ReleaseSlotOp>(edge.loc, edge.slot);
                continue;
            }

            auto *join = terminator->getSuccessor(edge.successor);
            auto operands = mlir::cast<mlir::BranchOpInterface>(terminator).getSuccessorOperands(edge.successor);
            llvm::SmallVector<mlir::Value> forwarded(operands.getForwardedOperands());
            auto *split = new mlir::Block();
            split->insertAfter(edge.from);
            builder.setInsertionPointToEnd(split);
            builder.create<mlir_ts::ReleaseSlotOp>(edge.loc, edge.slot);
            builder.create<mlir::cf::BranchOp>(edge.loc, join, forwarded);
            if (!forwarded.empty())
            {
                operands.erase(0, forwarded.size());
            }

            terminator->setSuccessor(split, edge.successor);
        }

        edgeReleases.clear();
    }

    llvm::SmallVector<mlir::Operation *> OwnershipInferencePass::slotAssignments(mlir::Value slot)
    {
        llvm::SmallVector<mlir::Operation *> assignments;
        for (auto *user : slot.getUsers())
        {
            auto storeOp = mlir::dyn_cast<mlir_ts::StoreOp>(user);
            auto releaseOp = storeOp ? mlir::dyn_cast_or_null<mlir_ts::ReleaseSlotOp>(storeOp->getPrevNode())
                                     : mlir_ts::ReleaseSlotOp();
            if (releaseOp && storeOp.getReference() == slot && releaseOp.getSlot() == slot)
            {
                assignments.push_back(storeOp);
            }
        }

        return assignments;
    }

    mlir::Operation *OwnershipInferencePass::readUsedAfter(mlir_ts::LoadOp read, mlir::Operation *retain, mlir::Operation *taker,
                                   mlir::Operation *kill,
                                   llvm::function_ref<bool(mlir::Operation *, mlir::Value)> isRead)
    {
        llvm::SmallVector<mlir::Value> values{read.getResult()};
        while (!values.empty())
        {
            auto current = values.pop_back_val();
            for (auto *user : current.getUsers())
            {
                if (user == retain || user == taker || mlir::isa<mlir_ts::RetainOp>(user))
                {
                    continue;
                }

                if (mlir::isa<mlir_ts::CastOp>(user) || isView(user))
                {
                    values.push_back(user->getResult(0));
                    continue;
                }

                llvm::SmallVector<mlir::Operation *, 2> kills{read.getOperation()};
                if (kill)
                {
                    kills.push_back(kill);
                }

                if (reachableAfter(taker, user, kills) && !(isRead && isRead(user, current)))
                {
                    return user;
                }
            }
        }

        return nullptr;
    }

} // namespace own_inference
