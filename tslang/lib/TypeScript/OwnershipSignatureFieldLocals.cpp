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
            llvm::SmallVector<mlir_ts::StoreOp> movedStores;
            body->walk([&](mlir_ts::StoreOp storeOp) {
                auto field = storeOp.getReference().getDefiningOp<mlir_ts::PropertyRefOp>();
                if (field && objectReadable(field))
                {
                    movedStores.push_back(storeOp);
                }
            });

            for (auto fieldStore : movedStores)
            {
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
                if (!def || mlir::isa<mlir_ts::LoadOp>(def) || !own_facts::ownsHeap(root) ||
                    own_facts::isCopyableValueAt(fieldStore, root.getType()) ||
                    MLIRTypeHelper::isSharedHandleType(root.getType()) || (!narrowed && !fieldReadsAs(field, root.getType())))
                {
                    continue;
                }

                llvm::SmallVector<mlir::OpOperand *> uses;
                for (auto &use : root.getUses())
                {
                    auto *user = use.getOwner();
                    if (user != fieldStore.getOperation() && !mlir::isa<mlir_ts::ReleaseOp>(user) &&
                        dominance.properlyDominates(fieldStore.getOperation(), user))
                    {
                        uses.push_back(&use);
                    }
                }

                if (uses.empty())
                {
                    continue;
                }

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
            }

            // The same for a value pushed into an array that is a field (TSLANG 52): `this.textures.push(
            // texture); this.views.push(texture.createView(...))`. After the push the value is the
            // array's last element, so each use the push dominates reads `array[newLength - 1]`,
            // read right after the push: a borrow of the element, followed through the calls between.
            llvm::SmallVector<mlir_ts::ArrayPushOp> pushes;
            body->walk([&](mlir_ts::ArrayPushOp pushOp) { pushes.push_back(pushOp); });
            for (auto pushOp : pushes)
            {
                auto field = pushOp.getOp().getDefiningOp<mlir_ts::PropertyRefOp>();
                if (pushOp.getItems().size() != 1 || !field || !objectReadable(field))
                {
                    continue;
                }

                auto arrayType = mlir::dyn_cast<mlir_ts::ArrayType>(mlir::cast<mlir_ts::RefType>(field.getType()).getElementType());
                auto root = stripViews(pushOp.getItems().front());
                auto *def = root.getDefiningOp();
                if (!arrayType || !def || mlir::isa<mlir_ts::LoadOp>(def) || !own_facts::ownsHeap(root) ||
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
                for (auto &use : root.getUses())
                {
                    auto *user = use.getOwner();
                    if (!mlir::isa<mlir_ts::ReleaseOp>(user) && dominance.properlyDominates(pushOp.getOperation(), user))
                    {
                        uses.push_back(&use);
                    }
                }

                if (uses.empty())
                {
                    continue;
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
