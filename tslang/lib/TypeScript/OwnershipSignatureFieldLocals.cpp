// OwnershipSignaturePass (see OwnershipSignaturePass.h): a local that holds what a field holds is
// read as the field (TSLANG 47).
//
// A lazy cache keeps its object in a field and hands out a borrow of it:
//
//     let result = this.cached;
//     if (!result) { result = new Q(1); this.cached = result; }
//     return result;
//
// On both ways to the return `result` is the object `this.cached` holds: read from it, or moved into
// it. Own's checks follow values, not equalities, so they saw a borrow of a field returned on one
// way and a local used after its value moved on the other. Each read of the local where it holds
// what the field holds is made a read of the field, which the checks take for the borrow it is:
// `return this.cached`. The local's own value, if no read is left to see it, is then never used,
// and the local is declared empty: `let result = undefined`, given a new object and moving it into
// the field, as `if (!this.cached) this.cached = new Q(1)` does.

#include "OwnershipSignaturePass.h"

#define DEBUG_TYPE "own"

namespace own_signature
{

    namespace
    {
        // The value a store puts into a place, seen through the views that widen it to the place's
        // type (`OptionalValue(v)`, a class widened to a union with null): the read of `local` it
        // is, when nothing writes `local` between that read and the store, in its block.
        bool storesLocal(mlir_ts::StoreOp storeOp, mlir::Value local)
        {
            auto value = storeOp.getValue();
            while (auto *def = value.getDefiningOp())
            {
                if (mlir::isa<mlir_ts::OptionalValueOp>(def) || (mlir::isa<mlir_ts::CastOp>(def) && isView(def)))
                {
                    value = def->getOperand(0);
                    continue;
                }

                break;
            }

            auto loadOp = value.getDefiningOp<mlir_ts::LoadOp>();
            if (!loadOp || loadOp.getReference() != local || loadOp->getBlock() != storeOp->getBlock())
            {
                return false;
            }

            for (auto *op = loadOp->getNextNode(); op && op != storeOp.getOperation(); op = op->getNextNode())
            {
                auto writes = false;
                op->walk([&](mlir::Operation *inner) {
                    auto innerStore = mlir::dyn_cast<mlir_ts::StoreOp>(inner);
                    auto innerRelease = mlir::dyn_cast<mlir_ts::ReleaseSlotOp>(inner);
                    writes |= (innerStore && innerStore.getReference() == local) ||
                              (innerRelease && innerRelease.getSlot() == local) || isCall(inner);
                });

                if (writes)
                {
                    return false;
                }
            }

            return true;
        }

        // The read of the field that replaces a read of the local, made right before it: the object
        // read again out of the local or parameter that holds it (sameObject), and the value seen as
        // the local's type - itself, or the payload of the optional the field holds.
        mlir::Value readFieldAt(mlir::OpBuilder &builder, mlir::Location location, mlir::Type type, mlir_ts::PropertyRefOp field)
        {
            auto objectLoad = field.getObjectRef().getDefiningOp<mlir_ts::LoadOp>();
            mlir::Value object = field.getObjectRef();
            if (objectLoad)
            {
                object = builder.create<mlir_ts::LoadOp>(location, objectLoad.getType(), objectLoad.getReference());
            }

            auto ref = builder.create<mlir_ts::PropertyRefOp>(location, field.getType(), object, field.getPositionAttr());
            auto fieldType = mlir::cast<mlir_ts::RefType>(field.getType()).getElementType();
            mlir::Value value = builder.create<mlir_ts::LoadOp>(location, fieldType, ref);
            if (fieldType != type)
            {
                // the local held the field's value, so it is there: no warning about an empty optional
                auto valueOp = builder.create<mlir_ts::ValueOrDefaultOp>(location, type, value);
                valueOp->setAttr(EXPLICIT_CAST_ATTR_NAME, builder.getUnitAttr());
                value = valueOp;
            }

            return value;
        }

        // Can a read of `field` stand in for a read of a local of this type?
        bool fieldReadsAs(mlir_ts::PropertyRefOp field, mlir::Type localType)
        {
            auto fieldType = mlir::cast<mlir_ts::RefType>(field.getType()).getElementType();
            if (fieldType == localType)
            {
                return true;
            }

            auto optionalType = mlir::dyn_cast<mlir_ts::OptionalType>(fieldType);
            return optionalType && optionalType.getElementType() == localType;
        }

        // The object a field is reached through can be read again anywhere in the function: a value
        // defined in the entry block, or read out of a local or parameter nothing assigns.
        bool objectReadable(mlir_ts::PropertyRefOp field)
        {
            // read out of a local or parameter that nothing assigns: two reads of it are one object
            auto loadOp = field.getObjectRef().getDefiningOp<mlir_ts::LoadOp>();
            if (!loadOp)
            {
                return false;
            }

            return isUnassignedSlot(loadOp.getReference());
        }

        // The object can be read again right after a store into its field or a push into it - nothing
        // runs between, so a read of an element or of a field is the same object (TSLANG 59: `const
        // frame = this.frames[i]; frame.view = view;`, with `frame` the element read itself).
        bool objectReadableRightAfter(mlir_ts::PropertyRefOp field)
        {
            if (objectReadable(field))
            {
                return true;
            }

            // ...or the object a value, not a read (`const frame = new Frame(); frame.allocators.push(a)`,
            // TSLANG 78): the read right after uses that very value
            auto loadOp = field.getObjectRef().getDefiningOp<mlir_ts::LoadOp>();
            return !loadOp || loadOp.getReference().getDefiningOp<mlir_ts::ElementRefOp>() ||
                   loadOp.getReference().getDefiningOp<mlir_ts::PropertyRefOp>();
        }

        // The truthiness test of `value` a branch makes: `cond_br(DialectCast(Cast(value -> boolean)))`.
        static bool testsTruthOf(mlir::cf::CondBranchOp branch, mlir::Value value)
        {
            auto dialectCast = branch.getCondition().getDefiningOp<mlir_ts::DialectCastOp>();
            auto cast = dialectCast ? dialectCast->getOperand(0).getDefiningOp<mlir_ts::CastOp>() : mlir_ts::CastOp();
            return cast && mlir::isa<mlir_ts::BooleanType>(cast.getType()) && cast->getOperand(0) == value;
        }

        // `if (value) array.push(value);` and uses of `value` after it (TSLANG 67): the block the test's
        // false edge goes to, reached otherwise only through the push, gets an argument - the element
        // read back after the push (`read`) on those paths, null on the false edge - and the uses of
        // `value` that block dominates read it.
        void mergePushedOrNull(mlir::Operation *pushOp, mlir::Value value, mlir::Value read, mlir::DominanceInfo &dominance)
        {
            auto *pushBlock = pushOp->getBlock();
            mlir::cf::CondBranchOp test;
            for (auto *user : value.getUsers())
            {
                auto cast = mlir::dyn_cast<mlir_ts::CastOp>(user);
                if (!cast)
                {
                    continue;
                }

                for (auto *castUser : cast->getUsers())
                {
                    for (auto *dialectUser : castUser->getUsers())
                    {
                        auto branch = mlir::dyn_cast<mlir::cf::CondBranchOp>(dialectUser);
                        if (branch && testsTruthOf(branch, value) && dominance.dominates(branch.getTrueDest(), pushBlock))
                        {
                            test = branch;
                        }
                    }
                }
            }

            if (!test || test.getFalseDest() == test.getTrueDest())
            {
                return;
            }

            auto *merge = test.getFalseDest();
            for (auto *pred : merge->getPredecessors())
            {
                if (pred == test->getBlock())
                {
                    continue;
                }

                // every other way in has pushed it, and goes in by a plain branch
                if (!mlir::isa<mlir::cf::BranchOp>(pred->getTerminator()) || !dominance.dominates(pushOp, pred->getTerminator()))
                {
                    return;
                }
            }

            llvm::SmallVector<mlir::OpOperand *> uses;
            for (auto &use : value.getUses())
            {
                auto *user = use.getOwner();
                if (!mlir::isa<mlir_ts::ReleaseOp>(user) && user->getBlock() != test->getBlock() &&
                    dominance.dominates(merge, user->getBlock()))
                {
                    uses.push_back(&use);
                }
            }

            if (uses.empty())
            {
                return;
            }

            auto location = pushOp->getLoc();
            auto argument = merge->addArgument(value.getType(), location);
            mlir::OpBuilder builder(test);
            mlir::Value null = builder.create<mlir_ts::NullOp>(location, mlir_ts::NullType::get(builder.getContext()));
            null = builder.create<mlir_ts::CastOp>(location, value.getType(), null);
            test.getFalseDestOperandsMutable().append(null);
            for (auto *pred : merge->getPredecessors())
            {
                if (auto branch = mlir::dyn_cast<mlir::cf::BranchOp>(pred->getTerminator()))
                {
                    branch.getDestOperandsMutable().append(read);
                }
            }

            for (auto *use : uses)
            {
                use->set(argument);
            }
        }

        // `value` seen through the views that keep its block: an optional made of it or opened, a
        // widening or narrowing cast that is a view.
        mlir::Value stripViews(mlir::Value value)
        {
            while (auto *def = value.getDefiningOp())
            {
                if (mlir::isa<mlir_ts::OptionalValueOp, mlir_ts::ValueOp, mlir_ts::ValueOrDefaultOp>(def) ||
                    (mlir::isa<mlir_ts::CastOp>(def) && isView(def)))
                {
                    value = def->getOperand(0);
                    continue;
                }

                break;
            }

            return value;
        }

        // Does nothing that may write `field` run on any way from `from` to `at`, both ops of `body`'s
        // blocks? A must-analysis over the blocks: it holds right after `from`, and stops at anything
        // that may write the field, at any call.
        bool unwrittenSince(mlir::Region *body, mlir::Operation *from, mlir::Operation *at, mlir_ts::PropertyRefOp field)
        {
            auto conservative = [](mlir::Operation *) { return true; };
            auto run = [&](mlir::Block *block, bool holds, mlir::Operation *stopAt) {
                for (auto &op : *block)
                {
                    if (&op == stopAt)
                    {
                        break;
                    }

                    holds = &op == from ? true : holds && !mayWriteField(&op, field, mlir::Value(), conservative);
                }

                return holds;
            };

            llvm::DenseMap<mlir::Block *, bool> in;
            for (auto &block : *body)
            {
                in[&block] = !block.isEntryBlock();
            }

            for (auto changed = true; changed;)
            {
                changed = false;
                for (auto &block : *body)
                {
                    if (block.isEntryBlock())
                    {
                        continue;
                    }

                    auto holds = !block.hasNoPredecessors() && llvm::all_of(block.getPredecessors(), [&](mlir::Block *predecessor) {
                                     return run(predecessor, in[predecessor], nullptr);
                                 });
                    if (holds != in[&block])
                    {
                        in[&block] = holds;
                        changed = true;
                    }
                }
            }

            return run(at->getBlock(), in[at->getBlock()], at);
        }

        // A value the steps below may follow to where it moves: made here (not a read of a local or a
        // place), or merged from branches that each bring a fresh value or null (TSLANG 54) - a call's
        // result among them, as unmerged: `cond ? null : device.loadTexture(...)` of another module, whose
        // facts are not read yet (TSLANG 74).
        bool ownValue(mlir::Value root, mlir::Operation *def)
        {
            if (auto argument = mlir::dyn_cast<mlir::BlockArgument>(root))
            {
                return own_facts::isFreshMerge(argument, 0, true);
            }

            return def && !mlir::isa<mlir_ts::LoadOp>(def);
        }

        // The fields a constructor keeps its parameters in (TSLANG 53): a parameter stored as it is into a
        // field of `this` that nothing else in the constructor writes, where `this` never escapes - it
        // is only read for its fields, so no call the constructor makes can reach the field. By the
        // parameter's number (`this` is 0): the field's position and its reference type.
        llvm::DenseMap<unsigned, std::pair<int32_t, mlir::Type>> keptParameters(mlir_ts::FuncOp ctor)
        {
            llvm::DenseMap<unsigned, std::pair<int32_t, mlir::Type>> kept;
            if (ctor.isExternal())
            {
                return kept;
            }

            auto &entry = ctor.getFunctionBody().front();
            llvm::DenseMap<mlir::Value, unsigned> parameters;
            mlir::Value thisSlot;
            // a parameter's slot: `ts.Param(%arg)`, or `ts.Variable(%arg)` once the slots are made
            for (auto &op : entry)
            {
                if (!mlir::isa<mlir_ts::ParamOp, mlir_ts::VariableOp>(&op) || op.getNumOperands() != 1)
                {
                    continue;
                }

                if (auto argument = mlir::dyn_cast<mlir::BlockArgument>(op.getOperand(0)); argument && argument.getOwner() == &entry)
                {
                    parameters[op.getResult(0)] = argument.getArgNumber();
                    if (argument.getArgNumber() == 0)
                    {
                        thisSlot = op.getResult(0);
                    }
                }
            }

            if (!thisSlot)
            {
                return kept;
            }

            // `this` only read, and only for its fields, which are only read or written. Or `this` given
            // to a call (`this.reset()`), which may reach any field it can write: then only a private
            // field nothing but this constructor writes (TSLANG 76), which no call can
            auto escapes = false;
            llvm::DenseMap<int32_t, unsigned> stores;
            llvm::SmallVector<mlir_ts::StoreOp> candidates;
            for (auto *user : thisSlot.getUsers())
            {
                auto loadOp = mlir::dyn_cast<mlir_ts::LoadOp>(user);
                if (!loadOp)
                {
                    return kept;
                }

                for (auto *loadUser : loadOp->getUsers())
                {
                    auto propertyRefOp = mlir::dyn_cast<mlir_ts::PropertyRefOp>(loadUser);
                    if (!propertyRefOp)
                    {
                        escapes = true;
                        continue;
                    }

                    for (auto *refUser : propertyRefOp->getUsers())
                    {
                        auto storeOp = mlir::dyn_cast<mlir_ts::StoreOp>(refUser);
                        auto releaseSlotOp = mlir::dyn_cast<mlir_ts::ReleaseSlotOp>(refUser);
                        if (storeOp && storeOp.getReference() == propertyRefOp.getResult())
                        {
                            stores[propertyRefOp.getPosition()]++;
                            candidates.push_back(storeOp);
                            continue;
                        }

                        // `this.values.push(0.0)`: a write of that array field, and of nothing else
                        if (mlir::isa<mlir_ts::ArrayPushOp, mlir_ts::ArrayPopOp>(refUser) && refUser->getOperand(0) == propertyRefOp.getResult())
                        {
                            stores[propertyRefOp.getPosition()]++;
                            continue;
                        }

                        if (!mlir::isa<mlir_ts::LoadOp>(refUser) && !(releaseSlotOp && releaseSlotOp.getSlot() == propertyRefOp.getResult()))
                        {
                            return kept;
                        }
                    }
                }
            }

            // a private field of the constructor's class that no other function writes
            auto writtenOnlyHere = [&](mlir_ts::PropertyRefOp field) {
                auto storage = fieldOwnerStorage(field);
                auto fields = storage ? storage.getFields() : decltype(storage.getFields())();
                auto position = field.getPosition();
                if (!storage || position < 0 || static_cast<size_t>(position) >= fields.size() ||
                    fields[position].accessLevel != mlir_ts::AccessLevel::Private)
                {
                    return false;
                }

                auto module = ctor->getParentOfType<mlir::ModuleOp>();
                auto written = false;
                module.walk([&](mlir_ts::PropertyRefOp propertyRefOp) {
                    if (written || propertyRefOp.getPosition() != position || fieldOwnerStorage(propertyRefOp) != storage ||
                        propertyRefOp->getParentOfType<mlir_ts::FuncOp>() == ctor)
                    {
                        return;
                    }

                    written = !llvm::all_of(propertyRefOp->getUsers(), [](mlir::Operation *user) { return mlir::isa<mlir_ts::LoadOp>(user); });
                });
                return !written;
            };

            for (auto storeOp : candidates)
            {
                auto propertyRefOp = storeOp.getReference().getDefiningOp<mlir_ts::PropertyRefOp>();
                auto loadOp = stripViews(storeOp.getValue()).getDefiningOp<mlir_ts::LoadOp>();
                if (storeOp->getBlock() != &entry || stores[propertyRefOp.getPosition()] != 1 || !loadOp ||
                    (escapes && !writtenOnlyHere(propertyRefOp)))
                {
                    continue;
                }

                auto parameter = parameters.find(loadOp.getReference());
                if (parameter == parameters.end() || parameter->second == 0 ||
                    llvm::any_of(loadOp.getReference().getUsers(), [](mlir::Operation *user) { return mlir::isa<mlir_ts::StoreOp>(user); }))
                {
                    continue;
                }

                kept[parameter->second] = {static_cast<int32_t>(propertyRefOp.getPosition()), propertyRefOp.getType()};
            }

            return kept;
        }

        // One local and one field: where the local holds what the field holds, a must-analysis over
        // the function's blocks. It starts to hold where the local is declared from a read of the
        // field, or where the field is given a read of the local; it stops at a store into the local,
        // at anything that may write the field, and at any call.
        struct LocalField
        {
            mlir_ts::VariableOp local;
            mlir_ts::PropertyRefOp field;

            // the state after `op`, given the state before it
            bool step(mlir::Operation *op, bool holds)
            {
                auto slot = local.getResult();
                if (op == local.getOperation())
                {
                    auto init = local.getInitializer();
                    auto initLoad = init ? init.getDefiningOp<mlir_ts::LoadOp>() : mlir_ts::LoadOp();
                    auto initField =
                        initLoad ? initLoad.getReference().getDefiningOp<mlir_ts::PropertyRefOp>() : mlir_ts::PropertyRefOp();
                    if (!initField || !sameField(initField, field) || initLoad->getBlock() != op->getBlock() ||
                        !initLoad->isBeforeInBlock(op))
                    {
                        return false;
                    }

                    for (auto *between = initLoad->getNextNode(); between != op; between = between->getNextNode())
                    {
                        if (mayWriteField(between, field, slot, [](mlir::Operation *) { return true; }))
                        {
                            return false;
                        }
                    }

                    return true;
                }

                if (auto storeOp = mlir::dyn_cast<mlir_ts::StoreOp>(op))
                {
                    if (storeOp.getReference() == slot)
                    {
                        // given a read of the field made right before, in its block (as the stores
                        // rewritten above are), it holds what the field holds again
                        auto loadOp = stripViews(storeOp.getValue()).getDefiningOp<mlir_ts::LoadOp>();
                        auto readField = loadOp ? loadOp.getReference().getDefiningOp<mlir_ts::PropertyRefOp>() : mlir_ts::PropertyRefOp();
                        if (!readField || !sameField(readField, field) || loadOp->getBlock() != op->getBlock() ||
                            !loadOp->isBeforeInBlock(op))
                        {
                            return false;
                        }

                        for (auto *between = loadOp->getNextNode(); between != op; between = between->getNextNode())
                        {
                            if (mayWriteField(between, field, mlir::Value(), [](mlir::Operation *) { return true; }))
                            {
                                return false;
                            }
                        }

                        return true;
                    }

                    if (auto propertyRefOp = storeOp.getReference().getDefiningOp<mlir_ts::PropertyRefOp>();
                        propertyRefOp && sameField(propertyRefOp, field))
                    {
                        return storesLocal(storeOp, slot);
                    }
                }

                if (auto releaseSlotOp = mlir::dyn_cast<mlir_ts::ReleaseSlotOp>(op); releaseSlotOp && releaseSlotOp.getSlot() == slot)
                {
                    return false;
                }

                return holds && !mayWriteField(op, field, slot, [](mlir::Operation *) { return true; });
            }
        };
    } // namespace

    void OwnershipSignaturePass::borrowParametersInLocals(mlir::ModuleOp module)
    {
        // `let src = order; let dst = scratch; ... const swap = src; src = dst; dst = swap;` (TSLANG
        // 57, native_ts_graphics' sortDescending): a `let` that only ever holds the function's
        // parameters - read out of their slots, or out of another such `let` - holds the caller's
        // values for the whole call. MLIRGen made it an owning local, so storing a parameter into it
        // kept the parameter. It owns nothing: it is made a plain local, its rc counting goes, and what
        // is done with its reads is what is done with the parameter.
        module.walk([&](mlir::FunctionOpInterface funcOp) {
            if (funcOp.isExternal())
            {
                return;
            }

            auto &body = funcOp.getFunctionBody();
            llvm::SetVector<mlir_ts::VariableOp> aliases;
            body.walk([&](mlir_ts::VariableOp varOp) {
                // not a Shared<T> handle, which own counts (spec 23.2), nor a string and the like,
                // which a local takes a copy of
                auto type = varOp.getInitializer() ? varOp.getInitializer().getType() : mlir::Type();
                if (varOp->hasAttr(OWNED_LOCAL_ATTR_NAME) && !varOp->hasAttr(OWNED_LOCAL_CONSUMED_ATTR_NAME) &&
                    type && !varOp.getCaptured().value_or(false) && !MLIRTypeHelper::isSharedHandleType(type) &&
                    !own_facts::isCopyableValueAt(varOp, type))
                {
                    aliases.insert(varOp);
                }
            });

            // a parameter's slot nothing assigns, or a `let` still in the set
            auto aliased = [&](mlir::Value value) {
                auto loadOp = own_facts::rootOf(value).getDefiningOp<mlir_ts::LoadOp>();
                auto slot = loadOp ? loadOp.getReference().getDefiningOp<mlir_ts::VariableOp>() : mlir_ts::VariableOp();
                int index = -1;
                return slot && (aliases.contains(slot) || (own_facts::isParameterSlot(slot, index) && isUnassignedSlot(slot.getResult())));
            };

            auto holdsOnlyParameters = [&](mlir_ts::VariableOp varOp) {
                if (!aliased(varOp.getInitializer()))
                {
                    return false;
                }

                for (auto *user : varOp->getUsers())
                {
                    if (mlir::isa<mlir_ts::LoadOp, mlir_ts::RetainSlotOp, mlir_ts::ReleaseSlotOp, mlir_ts::DebugVariableOp>(user))
                    {
                        continue;
                    }

                    auto storeOp = mlir::dyn_cast<mlir_ts::StoreOp>(user);
                    if (!storeOp || storeOp.getReference() != varOp.getResult() || !aliased(storeOp.getValue()))
                    {
                        return false;
                    }
                }

                return true;
            };

            // to a fixed point: a `let` that takes another's value stays only while that one does
            for (auto changed = true; changed;)
            {
                changed = false;
                for (auto varOp : llvm::to_vector(aliases))
                {
                    if (!holdsOnlyParameters(varOp))
                    {
                        aliases.remove(varOp);
                        changed = true;
                    }
                }
            }

            for (auto varOp : aliases)
            {
                varOp->removeAttr(OWNED_LOCAL_ATTR_NAME);
                llvm::SmallVector<mlir::Operation *> counting;
                for (auto *user : varOp->getUsers())
                {
                    if (mlir::isa<mlir_ts::RetainSlotOp, mlir_ts::ReleaseSlotOp>(user))
                    {
                        counting.push_back(user);
                    }

                    // rc's retain for an assignment, right before it (`Retain(v); ReleaseSlot(x); Store(v, x)`)
                    if (auto storeOp = mlir::dyn_cast<mlir_ts::StoreOp>(user))
                    {
                        for (auto *before = user->getPrevNode(); before; before = before->getPrevNode())
                        {
                            if (auto retainOp = mlir::dyn_cast<mlir_ts::RetainOp>(before); retainOp && retainOp->getOperand(0) == storeOp.getValue())
                            {
                                counting.push_back(before);
                                break;
                            }

                            if (!mlir::isa<mlir_ts::ReleaseSlotOp>(before))
                            {
                                break;
                            }
                        }
                    }
                }

                for (auto *op : counting)
                {
                    op->erase();
                }
            }
        });
    }

    void OwnershipSignaturePass::readFieldsForLocals(mlir::ModuleOp module)
    {
        llvm::SmallVector<mlir::Region *> bodies;
        module.walk([&](mlir::FunctionOpInterface funcOp) {
            if (!funcOp.isExternal())
            {
                bodies.push_back(&funcOp.getFunctionBody());
            }
        });

        for (auto *body : bodies)
        {
            // A value stored into a local while it is what a field holds - a read of the field nothing
            // has written since (`const hit = this.cached; if (hit) return hit;`), or what was just
            // moved into the field (`this.cached = result; return result;`, `result` a `const`) - is
            // stored as a read of the field made right there, and so is rc's retain of it for that
            // store.
            llvm::SmallVector<mlir_ts::StoreOp> fieldStores;
            llvm::SmallVector<mlir_ts::StoreOp> localStores;
            for (auto &block : *body)
            {
                for (auto &op : block)
                {
                    auto storeOp = mlir::dyn_cast<mlir_ts::StoreOp>(&op);
                    if (!storeOp)
                    {
                        continue;
                    }

                    auto field = storeOp.getReference().getDefiningOp<mlir_ts::PropertyRefOp>();
                    auto varOp = storeOp.getReference().getDefiningOp<mlir_ts::VariableOp>();
                    if (field && objectReadable(field))
                    {
                        fieldStores.push_back(storeOp);
                    }
                    else if (varOp && !varOp.getCaptured().value_or(false) && varOp->getParentRegion() == body)
                    {
                        localStores.push_back(storeOp);
                    }
                }
            }

            for (auto storeOp : localStores)
            {
                auto value = storeOp.getValue();
                auto root = stripViews(value);
                mlir_ts::PropertyRefOp field;
                if (auto loadOp = root.getDefiningOp<mlir_ts::LoadOp>())
                {
                    field = loadOp.getReference().getDefiningOp<mlir_ts::PropertyRefOp>();
                    if (!field || field->getParentRegion() != body || loadOp->getParentRegion() != body || !objectReadable(field) ||
                        !unwrittenSince(body, loadOp, storeOp, field))
                    {
                        field = {};
                    }
                }
                else if (own_facts::ownsHeap(root))
                {
                    for (auto fieldStore : fieldStores)
                    {
                        auto stored = fieldStore.getReference().getDefiningOp<mlir_ts::PropertyRefOp>();
                        if (stripViews(fieldStore.getValue()) == root && unwrittenSince(body, fieldStore, storeOp, stored))
                        {
                            field = stored;
                            break;
                        }
                    }
                }

                if (!field || !fieldReadsAs(field, value.getType()))
                {
                    continue;
                }

                // rc's retain for the store, right before it, or before its release of the old value
                auto *before = storeOp->getPrevNode();
                if (auto releaseSlotOp = mlir::dyn_cast_or_null<mlir_ts::ReleaseSlotOp>(before);
                    releaseSlotOp && releaseSlotOp.getSlot() == storeOp.getReference())
                {
                    before = before->getPrevNode();
                }

                auto retainOp = mlir::dyn_cast_or_null<mlir_ts::RetainOp>(before);
                if (retainOp && retainOp->getOperand(0) != value)
                {
                    retainOp = {};
                }

                mlir::OpBuilder builder(retainOp ? retainOp.getOperation() : storeOp.getOperation());
                auto read = readFieldAt(builder, storeOp.getLoc(), value.getType(), field);
                storeOp->setOperand(0, read);
                if (retainOp)
                {
                    retainOp->setOperand(0, read);
                }
            }

            // A value of the function's own moved into a field and used after (TSLANG 50): `const
            // layout = device.create(); this.layout = layout; if (!layout) return false;
            // use(layout);`. The `const` is the call's result itself, with no local to pair with a
            // field. After the store it is what the field holds, so each use the store dominates
            // reads the field instead, read right after the store: a borrow of the field, which the
            // checks then follow through the calls between with their facts. Its release stays: the
            // move takes it over.
            // The stores in nested regions count too (`if (ok) { this.f = v; use(v); }`).
            mlir::DominanceInfo dominance(body->getParentOp());
            // where each value moved by the steps below lives after the move: the move, and the read
            // of its new place made right after it
            llvm::DenseMap<mlir::Value, llvm::SmallVector<std::pair<mlir::Operation *, mlir::Value>>> movedTo;
            // Run again after the pushes below (TSLANG 64): `const q = new Frame(); this.frames.push(q);
            // q.view = view; if (!view) ...` - once the push makes `q` a read of the array's last
            // element, `q.view` is a field of an object that can be read again. Each store once.
            llvm::DenseSet<mlir::Operation *> followedStores;
            // The uses after `store` of the value it moves: those of `root` itself, the store
            // dominates; or, when `root` is a read of an owning local every other assignment of which
            // comes before the store (TSLANG 65: `let pipeline = null; if (ok) pipeline = make(); this.
            // pipelines[i] = pipeline; return pipeline;`), every later read of the local the store
            // dominates. False when the value is neither.
            // Can control go from `from` to `to`, `to` after `from` in its block or in a block after -
            // not through `avoided` (a local's declaration, which starts it afresh)?
            auto reaches = [&](mlir::Operation *from, mlir::Operation *to, mlir::Block *avoided = nullptr) {
                if (from->getBlock() == to->getBlock() && from->isBeforeInBlock(to))
                {
                    return true;
                }

                llvm::SmallPtrSet<mlir::Block *, 16> seen;
                llvm::SmallVector<mlir::Block *> work(from->getBlock()->getSuccessors().begin(), from->getBlock()->getSuccessors().end());
                while (!work.empty())
                {
                    auto *next = work.pop_back_val();
                    if (next == to->getBlock())
                    {
                        return true;
                    }

                    if (next == avoided)
                    {
                        continue;
                    }

                    if (seen.insert(next).second)
                    {
                        work.append(next->getSuccessors().begin(), next->getSuccessors().end());
                    }
                }

                return false;
            };

            // Does control come back to `op` on a way that does not declare `slot` again first?
            auto runsAgainUndeclared = [&](mlir::Operation *op, mlir_ts::VariableOp slot) {
                if (op->getParentRegion() != slot->getParentRegion() || !dominance.properlyDominates(slot.getOperation(), op))
                {
                    return true;
                }

                // the declaration's block, between it and `op` (in the same block: before it)
                auto *declared = slot->getBlock();
                llvm::SmallPtrSet<mlir::Block *, 16> seen;
                llvm::SmallVector<mlir::Block *> work(op->getBlock()->getSuccessors().begin(), op->getBlock()->getSuccessors().end());
                while (!work.empty())
                {
                    auto *next = work.pop_back_val();
                    if (next == declared)
                    {
                        continue;
                    }

                    if (next == op->getBlock())
                    {
                        return true;
                    }

                    if (seen.insert(next).second)
                    {
                        work.append(next->getSuccessors().begin(), next->getSuccessors().end());
                    }
                }

                return false;
            };

            // `inPlaceAfter`: an array changed in place through the local after the move is a use too, of
            // the local's slot (TSLANG 77) - for a step that gives it the place the array moved to
            auto usesAfterMove = [&](mlir::Operation *store, mlir::Value root, llvm::SmallVectorImpl<mlir::OpOperand *> &uses,
                                     bool inPlaceAfter = false) {
                auto loadOp = root.getDefiningOp<mlir_ts::LoadOp>();
                auto slot = loadOp ? loadOp.getReference().getDefiningOp<mlir_ts::VariableOp>() : mlir_ts::VariableOp();
                if (!slot)
                {
                    if (!ownValue(root, root.getDefiningOp()))
                    {
                        return false;
                    }

                    for (auto &use : root.getUses())
                    {
                        auto *user = use.getOwner();
                        if (user != store && !mlir::isa<mlir_ts::ReleaseOp>(user) && dominance.properlyDominates(store, user))
                        {
                            uses.push_back(&use);
                        }
                    }

                    return true;
                }

                // run once (not in a loop), so an assignment the store does not dominate runs before it
                // or on a way that does not reach the reads after it - or once for each time the local
                // is declared: `for (...) { const t = []; ...; new Node(t); t[0] }` (TSLANG 72)
                if (!own_facts::isOwningVariable(slot) || slot.getCaptured().value_or(false) || slot->getParentRegion() != body ||
                    (mayRunAgain(store) && runsAgainUndeclared(store, slot)))
                {
                    return false;
                }

                for (auto *user : slot.getResult().getUsers())
                {
                    if (auto assignment = mlir::dyn_cast<mlir_ts::StoreOp>(user);
                        assignment && assignment.getReference() == slot.getResult() && dominance.dominates(store, assignment))
                    {
                        return false;
                    }

                    // an array filled in place through the local before the move: `t.push(x)` (TSLANG 72)
                    if (mlir::isa<mlir_ts::ArrayPushOp, mlir_ts::ArrayPopOp, mlir_ts::ArrayUnshiftOp, mlir_ts::ArrayShiftOp,
                                  mlir_ts::ArraySpliceOp>(user) &&
                        !reaches(store, user, slot->getBlock()))
                    {
                        continue;
                    }

                    // ...and after it, through the place it moved to: `const sets = []; this.frames.push(new
                    // Frame(sets)); for (...) sets.push(set)` (TSLANG 77)
                    if (inPlaceAfter &&
                        mlir::isa<mlir_ts::ArrayPushOp, mlir_ts::ArrayPopOp, mlir_ts::ArrayUnshiftOp, mlir_ts::ArrayShiftOp,
                                  mlir_ts::ArraySpliceOp>(user) &&
                        dominance.properlyDominates(store, user))
                    {
                        for (auto &operand : user->getOpOperands())
                        {
                            if (operand.get() == slot.getResult())
                            {
                                uses.push_back(&operand);
                            }
                        }

                        continue;
                    }

                    if (!mlir::isa<mlir_ts::LoadOp, mlir_ts::StoreOp, mlir_ts::RetainSlotOp, mlir_ts::ReleaseSlotOp,
                                   mlir_ts::DebugVariableOp>(user))
                    {
                        return false; // its address taken: anything may change it
                    }
                }

                for (auto *user : slot.getResult().getUsers())
                {
                    if (user != loadOp.getOperation() && mlir::isa<mlir_ts::LoadOp>(user) && dominance.properlyDominates(store, user))
                    {
                        for (auto &use : user->getResult(0).getUses())
                        {
                            uses.push_back(&use);
                        }
                    }
                }

                return true;
            };
            auto followFieldStores = [&]() {
            llvm::SmallVector<mlir_ts::StoreOp> movedStores;
            body->walk([&](mlir_ts::StoreOp storeOp) {
                auto field = storeOp.getReference().getDefiningOp<mlir_ts::PropertyRefOp>();
                if (field && objectReadableRightAfter(field) && !followedStores.contains(storeOp))
                {
                    movedStores.push_back(storeOp);
                }
            });

            for (auto fieldStore : movedStores)
            {
                followedStores.insert(fieldStore);
                auto root = stripViews(fieldStore.getValue());
                auto *def = root.getDefiningOp();
                auto field = fieldStore.getReference().getDefiningOp<mlir_ts::PropertyRefOp>();
                // an object stored into a field that may also be null or hold another type
                // (`new Layout()` into `Layout | null`): read back, the field is narrowed to it
                auto fieldType = mlir::cast<mlir_ts::RefType>(field.getType()).getElementType();
                auto narrowed = !fieldReadsAs(field, root.getType()) &&
                                mlir::isa<mlir_ts::ClassType, mlir_ts::InterfaceType>(root.getType()) &&
                                mlir::isa<mlir_ts::UnionType>(fieldType);
                // a string and the like are copied into the field, and a Shared<T> handle counted:
                // the value stays the function's own, not moved
                // (a read of a local: TSLANG 65, usesAfterMove)
                llvm::SmallVector<mlir::OpOperand *> uses;
                (void)def;
                if (!own_facts::ownsHeap(root) || own_facts::isCopyableValueAt(fieldStore, root.getType()) ||
                    MLIRTypeHelper::isSharedHandleType(root.getType()) || (!narrowed && !fieldReadsAs(field, root.getType())) ||
                    !usesAfterMove(fieldStore, root, uses))
                {
                    continue;
                }

                // made even with no use left: an argument kept by this object may read through it
                // (below); unused, it goes at the end
                mlir::OpBuilder builder(fieldStore->getContext());
                builder.setInsertionPointAfter(fieldStore);
                auto read = readFieldAt(builder, fieldStore.getLoc(), narrowed ? fieldType : root.getType(), field);
                if (narrowed)
                {
                    read = builder.create<mlir_ts::CastOp>(fieldStore.getLoc(), root.getType(), read);
                }

                for (auto *use : uses)
                {
                    use->set(read);
                }

                movedTo[root].push_back({fieldStore.getOperation(), read});
            }

            // The same for a value stored into an element of an array that is a field or a local nothing
            // assigns (TSLANG 65): `this.baselinePipelines[logicOp] = pipeline; return pipeline;`. The
            // element's reference is there, and nothing runs between: a load of it right after the
            // store reads what was stored - a borrow of the element.
            llvm::SmallVector<mlir_ts::StoreOp> elementStores;
            body->walk([&](mlir_ts::StoreOp storeOp) {
                auto elementRefOp = storeOp.getReference().getDefiningOp<mlir_ts::ElementRefOp>();
                if (!elementRefOp || followedStores.contains(storeOp))
                {
                    return;
                }

                auto arrayLoad = elementRefOp->getOperand(0).getDefiningOp<mlir_ts::LoadOp>();
                auto arrayField = arrayLoad ? arrayLoad.getReference().getDefiningOp<mlir_ts::PropertyRefOp>() : mlir_ts::PropertyRefOp();
                if (arrayLoad && ((arrayField && objectReadableRightAfter(arrayField)) || isUnassignedSlot(arrayLoad.getReference())))
                {
                    elementStores.push_back(storeOp);
                }
            });

            for (auto elementStore : elementStores)
            {
                followedStores.insert(elementStore);
                auto root = stripViews(elementStore.getValue());
                auto elementType = mlir::cast<mlir_ts::RefType>(elementStore.getReference().getType()).getElementType();
                llvm::SmallVector<mlir::OpOperand *> uses;
                if (elementType != root.getType() || !own_facts::ownsHeap(root) ||
                    own_facts::isCopyableValueAt(elementStore, root.getType()) || MLIRTypeHelper::isSharedHandleType(root.getType()) ||
                    !usesAfterMove(elementStore, root, uses) || uses.empty())
                {
                    continue;
                }

                mlir::OpBuilder builder(elementStore->getContext());
                builder.setInsertionPointAfter(elementStore);
                mlir::Value read = builder.create<mlir_ts::LoadOp>(elementStore.getLoc(), elementType, elementStore.getReference());
                for (auto *use : uses)
                {
                    use->set(read);
                }

                movedTo[root].push_back({elementStore.getOperation(), read});
            }
            };

            followFieldStores();

            // The same for a value pushed into an array that is a field (TSLANG 52): `this.textures.push(
            // texture); this.views.push(texture.createView(...))`. After the push the value is the
            // array's last element, so each use the push dominates reads `array[newLength - 1]`,
            // read right after the push: a borrow of the element, followed through the calls between.
            llvm::SmallVector<mlir_ts::ArrayPushOp> pushes;
            body->walk([&](mlir_ts::ArrayPushOp pushOp) { pushes.push_back(pushOp); });
            for (auto pushOp : pushes)
            {
                auto field = pushOp.getOp().getDefiningOp<mlir_ts::PropertyRefOp>();
                if (pushOp.getItems().size() != 1 || !field || !objectReadableRightAfter(field))
                {
                    continue;
                }

                auto arrayType = mlir::dyn_cast<mlir_ts::ArrayType>(mlir::cast<mlir_ts::RefType>(field.getType()).getElementType());
                auto root = stripViews(pushOp.getItems().front());
                auto *def = root.getDefiningOp();
                // ...or a read of an owning local (TSLANG 72), its later reads the uses (usesAfterMove)
                auto rootLoad = def ? mlir::dyn_cast<mlir_ts::LoadOp>(def) : mlir_ts::LoadOp();
                auto slotRead = rootLoad && rootLoad.getReference().getDefiningOp<mlir_ts::VariableOp>();
                if (!arrayType || (!ownValue(root, def) && !slotRead) || !own_facts::ownsHeap(root) ||
                    own_facts::isCopyableValueAt(pushOp, root.getType()) || MLIRTypeHelper::isSharedHandleType(root.getType()))
                {
                    continue;
                }

                // the element read back as the value's type: itself, or widened back to it
                auto elementType = arrayType.getElementType();
                if (elementType != root.getType() && !(mlir::isa<mlir_ts::ClassType, mlir_ts::InterfaceType>(elementType) &&
                                                       mlir::isa<mlir_ts::UnionType, mlir_ts::OptionalType>(root.getType())))
                {
                    continue;
                }

                llvm::SmallVector<mlir::OpOperand *> uses;
                if (slotRead)
                {
                    if (!usesAfterMove(pushOp, root, uses) || uses.empty())
                    {
                        continue;
                    }
                }
                else
                {
                    for (auto &use : root.getUses())
                    {
                        auto *user = use.getOwner();
                        if (!mlir::isa<mlir_ts::ReleaseOp>(user) && dominance.properlyDominates(pushOp.getOperation(), user))
                        {
                            uses.push_back(&use);
                        }
                    }
                }

                auto location = pushOp.getLoc();
                mlir::OpBuilder builder(pushOp->getContext());
                builder.setInsertionPointAfter(pushOp);
                auto array = readFieldAt(builder, location, arrayType, field);
                mlir::Value one = builder.create<mlir_ts::ConstantOp>(location, builder.getI32IntegerAttr(1));
                one = builder.create<mlir_ts::CastOp>(location, builder.getIndexType(), one);
                mlir::Value last = builder.create<mlir_ts::ArithmeticBinaryOp>(
                    location, builder.getIndexType(), builder.getI32IntegerAttr(static_cast<int32_t>(SyntaxKind::MinusToken)),
                    pushOp.getNewSize(), one);
                last = builder.create<mlir_ts::CastOp>(location, builder.getI32Type(), last);
                auto elementRef = builder.create<mlir_ts::ElementRefOp>(location, mlir_ts::RefType::get(elementType), array, last);
                mlir::Value read = builder.create<mlir_ts::LoadOp>(location, elementType, elementRef);
                if (elementType != root.getType())
                {
                    read = builder.create<mlir_ts::CastOp>(location, root.getType(), read);
                }

                for (auto *use : uses)
                {
                    use->set(read);
                }

                movedTo[root].push_back({pushOp.getOperation(), read});

                // ...and pushed only where it is not null (TSLANG 67): `const view = make(); if (view)
                // this.views.push(view); return view;`. Where the paths meet again it is the element
                // read back, or the null it was: a merge of the two, which each use there reads.
                mergePushedOrNull(pushOp, root, read, dominance);
            }

            followFieldStores();

            // A value of the function's own given to a constructor that keeps it in a field
            // (TSLANG 53): `const app = new App(sample); sample.use();`, or the object pushed on
            // (`this.frames.push(new Frame(viewSet)); viewSet.update(...)`). After the call the
            // value is the object's field, so each use the call dominates reads that field - through
            // the object itself, or through the place it was moved to last, as the steps above made
            // it: a borrow of the field, followed through the calls between.
            // the constructor calls: `ts.SymbolCallInternal(object, args...)` naming it, or a call of
            // the method bound to the object; their operands are the object, then the parameters
            struct Construction
            {
                mlir::Operation *call;
                llvm::StringRef callee;
                llvm::SmallVector<mlir::Value> operands;
            };
            llvm::SmallVector<Construction> constructions;
            body->walk([&](mlir::Operation *op) {
                if (auto callOp = mlir::dyn_cast<mlir_ts::SymbolCallInternalOp>(op))
                {
                    constructions.push_back({op, callOp.getCallee(), llvm::to_vector(callOp.getCallOperands())});
                }
                else if (auto callOp = mlir::dyn_cast<mlir_ts::CallIndirectOp>(op))
                {
                    auto getMethodOp = callOp.getCallee().getDefiningOp<mlir_ts::GetMethodOp>();
                    auto symbolRefOp = getMethodOp ? getMethodOp.getBoundFunc().getDefiningOp<mlir_ts::ThisSymbolRefOp>() : mlir_ts::ThisSymbolRefOp();
                    if (symbolRefOp && callOp.getCallOperands().size() >= 1)
                    {
                        llvm::SmallVector<mlir::Value> operands{symbolRefOp.getThisVal()};
                        operands.append(callOp.getCallOperands().begin() + 1, callOp.getCallOperands().end());
                        constructions.push_back({op, symbolRefOp.getIdentifier(), operands});
                    }
                }
            });

            for (auto &construction : constructions)
            {
                auto *callOp = construction.call;
                auto object = construction.operands.empty() ? mlir::Value() : construction.operands.front();
                if (!object || !object.getDefiningOp<mlir_ts::NewOp>() || !construction.callee.ends_with(".constructor"))
                {
                    continue;
                }

                auto ctor = module.lookupSymbol<mlir_ts::FuncOp>(construction.callee);
                if (!ctor)
                {
                    continue;
                }

                auto &operands = construction.operands;
                for (auto &[number, kept] : keptParameters(ctor))
                {
                    if (number >= operands.size())
                    {
                        continue;
                    }

                    auto root = stripViews(operands[number]);
                    auto *def = root.getDefiningOp();
                    auto fieldRefType = mlir::cast<mlir_ts::RefType>(kept.second);
                    auto fieldType = fieldRefType.getElementType();
                    // ...or a value narrowed to the field's class from that class or null: `const texture =
                    // load(); if (!texture) return; new Target(texture)` (TSLANG 71). Read back, the field
                    // is widened to the value's type again.
                    auto narrowed = fieldType != root.getType() &&
                                    ((mlir::isa<mlir_ts::ClassType, mlir_ts::InterfaceType>(root.getType()) &&
                                      mlir::isa<mlir_ts::UnionType>(fieldType)) ||
                                     (mlir::isa<mlir_ts::ClassType, mlir_ts::InterfaceType>(fieldType) &&
                                      own_facts::holdsAsIs(root.getType(), fieldType, true)));
                    // ...or a read of an owning local (TSLANG 72), its later reads the uses (usesAfterMove)
                    auto rootLoad = def ? mlir::dyn_cast<mlir_ts::LoadOp>(def) : mlir_ts::LoadOp();
                    auto slotRead = rootLoad && rootLoad.getReference().getDefiningOp<mlir_ts::VariableOp>();
                    if ((!ownValue(root, def) && !slotRead) || !own_facts::ownsHeap(root) ||
                        own_facts::isCopyableValueAt(callOp, root.getType()) || MLIRTypeHelper::isSharedHandleType(root.getType()) ||
                        (fieldType != root.getType() && !narrowed))
                    {
                        continue;
                    }

                    // each use read through the object's latest place before it: the last move
                    // of the object that the use comes after, else the object right after the call
                    llvm::DenseMap<mlir::Value, mlir::Value> reads;
                    llvm::SmallVector<mlir::OpOperand *> uses;
                    // the object's own releases: a use after one, in its block, would read a freed field
                    llvm::SmallVector<mlir::Operation *> releases;
                    for (auto *user : object.getUsers())
                    {
                        if (mlir::isa<mlir_ts::ReleaseOp>(user))
                        {
                            releases.push_back(user);
                        }
                    }

                    auto afterRelease = [&](mlir::Operation *user) {
                        return llvm::any_of(releases, [&](mlir::Operation *release) {
                            auto *ancestor = release->getBlock()->findAncestorOpInBlock(*user);
                            return ancestor && release->isBeforeInBlock(ancestor);
                        });
                    };

                    if (slotRead)
                    {
                        llvm::SmallVector<mlir::OpOperand *> later;
                        if (!usesAfterMove(callOp, root, later, true))
                        {
                            continue;
                        }

                        for (auto *use : later)
                        {
                            if (!afterRelease(use->getOwner()))
                            {
                                uses.push_back(use);
                            }
                        }
                    }
                    else
                    {
                        for (auto &use : root.getUses())
                        {
                            if (!mlir::isa<mlir_ts::ReleaseOp>(use.getOwner()) && dominance.properlyDominates(callOp, use.getOwner()) &&
                                !afterRelease(use.getOwner()))
                            {
                                uses.push_back(&use);
                            }
                        }
                    }

                    // an array changed in place after the move changes the field itself: its reference, of the
                    // local's type (TSLANG 77)
                    if (llvm::any_of(uses, [&](mlir::OpOperand *use) {
                            return mlir::isa<mlir_ts::RefType>(use->get().getType()) && (narrowed || use->get().getType() != fieldRefType);
                        }))
                    {
                        continue;
                    }

                    llvm::DenseMap<mlir::Value, mlir::Value> refs;
                    for (auto *use : uses)
                    {
                        auto *user = use->getOwner();
                        mlir::Operation *after = callOp;
                        mlir::Value place = object;
                        for (auto &[move, read] : movedTo.lookup(object))
                        {
                            if (dominance.properlyDominates(move, user) && dominance.properlyDominates(after, move))
                            {
                                after = read.getDefiningOp();
                                place = read;
                            }
                        }

                        auto &fieldRead = reads[place];
                        if (!fieldRead)
                        {
                            mlir::OpBuilder builder(callOp->getContext());
                            builder.setInsertionPointAfter(after);
                            auto ref = builder.create<mlir_ts::PropertyRefOp>(callOp->getLoc(), fieldRefType, place,
                                                                              builder.getI32IntegerAttr(kept.first));
                            refs[place] = ref;
                            fieldRead = builder.create<mlir_ts::LoadOp>(callOp->getLoc(), fieldType, ref);
                            if (narrowed)
                            {
                                fieldRead = builder.create<mlir_ts::CastOp>(callOp->getLoc(), root.getType(), fieldRead);
                            }
                        }

                        use->set(mlir::isa<mlir_ts::RefType>(use->get().getType()) ? refs[place] : fieldRead);
                    }
                }
            }

            // the reads of a moved value's new place nothing uses
            for (auto &[value, moves] : movedTo)
            {
                for (auto &[move, read] : moves)
                {
                    llvm::SmallVector<mlir::Operation *> dead{read.getDefiningOp()};
                    while (!dead.empty())
                    {
                        auto *op = dead.pop_back_val();
                        if (!op || !op->use_empty() || !mlir::isa<mlir_ts::LoadOp, mlir_ts::CastOp, mlir_ts::PropertyRefOp, mlir_ts::ElementRefOp,
                                                                  mlir_ts::ArithmeticBinaryOp, mlir_ts::ConstantOp, mlir_ts::ValueOrDefaultOp>(op))
                        {
                            continue;
                        }

                        for (auto operand : op->getOperands())
                        {
                            dead.push_back(operand.getDefiningOp());
                        }

                        op->erase();
                    }
                }
            }

            // the pairs: a local not captured, and a field it is declared from or that is given it
            llvm::SmallVector<LocalField> pairs;
            body->walk([&](mlir_ts::VariableOp varOp) {
                if (varOp.getCaptured().value_or(false) || varOp->getParentRegion() != body)
                {
                    return;
                }

                auto slot = varOp.getResult();
                auto localType = mlir::cast<mlir_ts::RefType>(slot.getType()).getElementType();
                auto addField = [&](mlir_ts::PropertyRefOp field) {
                    if (!field || field->getParentRegion() != body || !fieldReadsAs(field, localType) || !objectReadable(field) ||
                        llvm::any_of(pairs, [&](LocalField &pair) { return pair.local == varOp && sameField(pair.field, field); }))
                    {
                        return;
                    }

                    pairs.push_back({varOp, field});
                };

                if (auto init = varOp.getInitializer())
                {
                    if (auto initLoad = init.getDefiningOp<mlir_ts::LoadOp>())
                    {
                        addField(initLoad.getReference().getDefiningOp<mlir_ts::PropertyRefOp>());
                    }
                }

                for (auto *user : slot.getUsers())
                {
                    auto loadOp = mlir::dyn_cast<mlir_ts::LoadOp>(user);
                    if (!loadOp)
                    {
                        continue;
                    }

                    for (auto *loadUser : loadOp->getUsers())
                    {
                        auto *storeUser = loadUser;
                        while (storeUser && storeUser->getNumResults() == 1 && !mlir::isa<mlir_ts::StoreOp>(storeUser) &&
                               (mlir::isa<mlir_ts::OptionalValueOp>(storeUser) ||
                                (mlir::isa<mlir_ts::CastOp>(storeUser) && isView(storeUser))) &&
                               storeUser->getResult(0).hasOneUse())
                        {
                            storeUser = *storeUser->getResult(0).getUsers().begin();
                        }

                        if (auto storeOp = mlir::dyn_cast_or_null<mlir_ts::StoreOp>(storeUser))
                        {
                            addField(storeOp.getReference().getDefiningOp<mlir_ts::PropertyRefOp>());
                        }
                    }
                }
            });

            for (auto &pair : pairs)
            {
                // where the local holds what the field holds, at each block's entry: optimistic, to a
                // fixed point; nothing holds at the entry
                llvm::DenseMap<mlir::Block *, bool> in;
                for (auto &block : *body)
                {
                    in[&block] = !block.isEntryBlock();
                }

                auto outOf = [&](mlir::Block *block) {
                    auto holds = in[block];
                    for (auto &op : *block)
                    {
                        holds = pair.step(&op, holds);
                    }

                    return holds;
                };

                for (auto changed = true; changed;)
                {
                    changed = false;
                    for (auto &block : *body)
                    {
                        if (block.isEntryBlock())
                        {
                            continue;
                        }

                        auto holds = !block.hasNoPredecessors() &&
                                     llvm::all_of(block.getPredecessors(), [&](mlir::Block *predecessor) { return outOf(predecessor); });
                        if (holds != in[&block])
                        {
                            in[&block] = holds;
                            changed = true;
                        }
                    }
                }

                // each read of the local where it holds the field's value, but the one given to the field
                auto slot = pair.local.getResult();
                llvm::SmallVector<mlir_ts::LoadOp> reads;
                for (auto &block : *body)
                {
                    auto holds = in[&block];
                    for (auto &op : block)
                    {
                        if (auto loadOp = mlir::dyn_cast<mlir_ts::LoadOp>(&op); loadOp && holds && loadOp.getReference() == slot)
                        {
                            auto feedsField = llvm::any_of(loadOp->getUsers(), [&](mlir::Operation *user) {
                                auto storeOp = mlir::dyn_cast<mlir_ts::StoreOp>(user);
                                return storeOp && storeOp.getValue() == loadOp.getResult();
                            });
                            if (!feedsField)
                            {
                                reads.push_back(loadOp);
                            }
                        }

                        holds = pair.step(&op, holds);
                    }
                }

                for (auto read : reads)
                {
                    mlir::OpBuilder builder(read);
                    auto value = readFieldAt(builder, read.getLoc(), read.getType(), pair.field);
                    read.getResult().replaceAllUsesWith(value);
                    read->erase();
                }

                // the local's own value, when no read is left to see it: the local is declared empty
                auto init = pair.local.getInitializer();
                auto localType = mlir::cast<mlir_ts::RefType>(slot.getType()).getElementType();
                if (reads.empty() || !init || !mlir::isa<mlir_ts::OptionalType>(localType) || !init.getDefiningOp<mlir_ts::LoadOp>())
                {
                    continue;
                }

                // a may-analysis: where the declared value may still be in the local
                llvm::DenseMap<mlir::Block *, bool> mayIn;
                auto declared = [&](mlir::Operation *op, bool holds) {
                    if (op == pair.local.getOperation())
                    {
                        return true;
                    }

                    auto storeOp = mlir::dyn_cast<mlir_ts::StoreOp>(op);
                    return holds && !(storeOp && storeOp.getReference() == slot);
                };

                for (auto changed = true; changed;)
                {
                    changed = false;
                    for (auto &block : *body)
                    {
                        auto holds = llvm::any_of(block.getPredecessors(), [&](mlir::Block *predecessor) {
                            auto out = mayIn[predecessor];
                            for (auto &op : *predecessor)
                            {
                                out = declared(&op, out);
                            }

                            return out;
                        });
                        if (holds != mayIn[&block])
                        {
                            mayIn[&block] = holds;
                            changed = true;
                        }
                    }
                }

                auto seen = false;
                for (auto &block : *body)
                {
                    auto holds = mayIn[&block];
                    for (auto &op : block)
                    {
                        auto loadOp = mlir::dyn_cast<mlir_ts::LoadOp>(&op);
                        seen |= holds && loadOp && loadOp.getReference() == slot;
                        // a read in a nested region sees whatever the local holds
                        op.walk([&](mlir_ts::LoadOp nested) {
                            seen |= nested != loadOp && nested.getReference() == slot;
                        });
                        holds = declared(&op, holds);
                    }
                }

                if (seen)
                {
                    continue;
                }

                mlir::OpBuilder builder(pair.local);
                auto empty = builder.create<mlir_ts::OptionalUndefOp>(pair.local.getLoc(), localType);
                pair.local->setOperand(0, empty);
            }

            // A local whose reads all became reads of the field holds what nobody reads: an assignment
            // of a field read to it (`r = n` once `this.cached = n`) is dropped, with rc's retain of
            // the value and release of the old one, `Retain(v); ReleaseSlot(r); Store(v, r)`. The
            // local keeps its old value until its own release.
            for (auto &pair : pairs)
            {
                auto slot = pair.local.getResult();
                auto read = llvm::any_of(slot.getUsers(), [](mlir::Operation *user) { return mlir::isa<mlir_ts::LoadOp>(user); });
                if (read)
                {
                    continue;
                }

                llvm::SmallVector<mlir::Operation *> dead;
                for (auto *user : slot.getUsers())
                {
                    auto storeOp = mlir::dyn_cast<mlir_ts::StoreOp>(user);
                    if (!storeOp || storeOp.getReference() != slot)
                    {
                        continue;
                    }

                    auto loadOp = stripViews(storeOp.getValue()).getDefiningOp<mlir_ts::LoadOp>();
                    auto releaseSlotOp = mlir::dyn_cast_or_null<mlir_ts::ReleaseSlotOp>(storeOp->getPrevNode());
                    auto retainOp = releaseSlotOp ? mlir::dyn_cast_or_null<mlir_ts::RetainOp>(releaseSlotOp->getPrevNode())
                                                  : mlir_ts::RetainOp();
                    if (!loadOp || !loadOp.getReference().getDefiningOp<mlir_ts::PropertyRefOp>() ||
                        !retainOp || releaseSlotOp.getSlot() != slot || retainOp->getOperand(0) != storeOp.getValue())
                    {
                        continue;
                    }

                    dead.append({storeOp.getOperation(), releaseSlotOp.getOperation(), retainOp.getOperation()});
                }

                for (auto *op : dead)
                {
                    op->erase();
                }
            }
        }
    }

} // namespace own_signature

#undef DEBUG_TYPE
