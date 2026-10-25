// OwnershipInferencePass (see OwnershipInferencePass.h): moves - the one taker of a fresh value,
// the retains that are its bookkeeping, and the paths that use it after the move.

#include "OwnershipInferencePass.h"

#define DEBUG_TYPE "own"

namespace own_inference
{

    bool OwnershipInferencePass::slotHoldsInitializerAt(mlir::Operation *op)
    {
        auto slot = mlir::cast<mlir_ts::RetainSlotOp>(op).getSlot();
        auto varOp = slot.getDefiningOp<mlir_ts::VariableOp>();
        if (!varOp || !varOp.getInitializer() || isCellVariable(slot) || varOp->getBlock() != op->getBlock())
        {
            return false;
        }

        for (auto *between = varOp->getNextNode(); between; between = between->getNextNode())
        {
            if (between == op)
            {
                return true;
            }

            if (llvm::is_contained(between->getOperands(), slot) || isCall(between) || between->getNumRegions() > 0)
            {
                return false;
            }
        }

        return false;
    }

    mlir::Value OwnershipInferencePass::retainedValue(mlir::Operation *op)
    {
        if (auto retainOp = mlir::dyn_cast<mlir_ts::RetainOp>(op))
        {
            return rootOf(retainOp.getReference());
        }

        auto varOp = mlir::cast<mlir_ts::RetainSlotOp>(op).getSlot().getDefiningOp<mlir_ts::VariableOp>();
        return varOp && varOp.getInitializer() ? rootOf(varOp.getInitializer()) : mlir::Value();
    }

    bool OwnershipInferencePass::ownsHeap(mlir::Value value)
    {
        // a `value_ref` points at storage it does not own, but one `ts.New` made and seen as its
        // object is the object's block: the root of a generator's state object or an object literal
        if (value.getDefiningOp<mlir_ts::NewOp>() && mlir::isa<mlir_ts::ValueRefType>(value.getType()) &&
            llvm::any_of(value.getUsers(), [](mlir::Operation *user) { return mlir::isa<mlir_ts::CastOp>(user) && isView(user); }))
        {
            return true;
        }

        return ownsHeap(value.getLoc(), value.getType());
    }

    bool OwnershipInferencePass::ownsHeap(mlir::Location location, mlir::Type type)
    {
        return own_facts::ownsHeap(location, type);
    }

    bool OwnershipInferencePass::holdsConstantsOrMoves(mlir::Value value)
    {
        auto loadOp = value.getDefiningOp<mlir_ts::LoadOp>();
        auto varOp = loadOp ? loadOp.getReference().getDefiningOp<mlir_ts::VariableOp>() : mlir_ts::VariableOp();
        // a borrowing record (spec 32) takes nothing for what it borrows, and its fresh values move in
        // as an owning record's do
        if (!varOp || !isConstantRecordLocal(varOp) || !(isOwningRecord(value) || borrowingRecordOf(varOp)))
        {
            return false;
        }

        // a fresh value moved in, its move decided by its own check against this retain
        auto movesIn = llvm::any_of(varOp->getUsers(), [&](mlir::Operation *user) {
            auto propertyRefOp = mlir::dyn_cast<mlir_ts::PropertyRefOp>(user);
            if (!propertyRefOp)
            {
                return false;
            }

            // a method's field is a `!ts.bound_ref`, which holds a function, not a block
            auto refType = mlir::dyn_cast<mlir_ts::RefType>(propertyRefOp.getType());
            if (refType && !ownsHeap(propertyRefOp.getLoc(), refType.getElementType()))
            {
                return false;
            }

            return llvm::any_of(propertyRefOp->getUsers(),
                                [](mlir::Operation *fieldUser) { return mlir::isa<mlir_ts::StoreOp>(fieldUser); });
        });

        return !movesIn || recordRetainOf(varOp) != nullptr;
    }

    mlir::Operation *OwnershipInferencePass::recordRetainOf(mlir_ts::VariableOp varOp)
    {
        if (!isConstantRecordLocal(varOp))
        {
            return nullptr;
        }

        mlir_ts::LoadOp read;
        for (auto *user : varOp->getUsers())
        {
            if (auto loadOp = mlir::dyn_cast<mlir_ts::LoadOp>(user))
            {
                if (read)
                {
                    return nullptr;
                }

                read = loadOp;
            }
        }

        if (!read)
        {
            return nullptr;
        }

        mlir::Operation *retain = nullptr;
        auto boxed = false;
        auto other = false;
        auto bad = false;
        // the read and its views (a `@value` value seen as its tuple, or the other way round)
        forEachUse(read.getResult(), [&](mlir::Operation *user, mlir::Value used) {
            if (bad)
            {
                return;
            }

            if (mlir::isa<mlir_ts::RetainOp>(user))
            {
                bad = retain != nullptr;
                retain = user;
                return;
            }

            // copied into the block it is given in as an object or an interface (spec 27): the box
            // owns what the record held, and its own check decides where the box goes
            auto storeOp = mlir::dyn_cast<mlir_ts::StoreOp>(user);
            auto box = storeOp && storeOp.getValue() == used ? recordBoxOf(storeOp.getReference()) : mlir_ts::NewOp();
            if (box && box.getResult() == storeOp.getReference())
            {
                bad = boxed;
                boxed = true;
                return;
            }

            other = true;
            if (mlir::isa<mlir_ts::ReturnInternalOp>(user))
            {
                return;
            }

            // `let a = { item: new C() }`: the read is what an owning local starts with, and rc's
            // `ts.RetainSlot` of that local is the record's retain
            if (auto ownerOp = mlir::dyn_cast<mlir_ts::VariableOp>(user))
            {
                auto *slotRetain = ownerOp.getInitializer() == used && isOwningVariable(ownerOp) ? onlyRetainSlotOf(ownerOp)
                                                                                                 : nullptr;
                bad = !slotRetain || retain;
                retain = slotRetain;
                return;
            }

            // stored into the result slot, or into an owning local (`a = { item: new C() }`)
            auto slot = storeOp && storeOp.getValue() == used ? storeOp.getReference().getDefiningOp<mlir_ts::VariableOp>()
                                                              : mlir_ts::VariableOp();
            bad = !slot || !(isResultSlot(slot) || isOwningVariable(slot));
        });

        if (bad)
        {
            return nullptr;
        }

        // a boxed record goes nowhere else: the box is its one owner
        if (boxed && other)
        {
            return nullptr;
        }

        return retain;
    }

    bool OwnershipInferencePass::isBuiltArrayRead(mlir::Operation *retain, mlir::Value value)
    {
        auto read = value.getDefiningOp<mlir_ts::LoadOp>();
        auto builder = read ? read.getReference().getDefiningOp<mlir_ts::VariableOp>() : mlir_ts::VariableOp();
        auto owner = mlir::cast<mlir_ts::RetainSlotOp>(retain).getSlot().getDefiningOp<mlir_ts::VariableOp>();
        if (!builder || isOwningVariable(builder) || builder.getCaptured().value_or(false) || !builder.getInitializer() ||
            !mlir::isa_and_nonnull<mlir_ts::CreateArrayOp, mlir_ts::NewArrayOp>(builder.getInitializer().getDefiningOp()) ||
            !owner || owner.getInitializer() != read.getResult() || !read->hasOneUse() || onlyRetainSlotOf(owner) != retain)
        {
            return false;
        }

        // only filled, and only before it is read
        return llvm::all_of(builder->getUses(), [&](mlir::OpOperand &use) {
            auto *user = use.getOwner();
            if (user == read)
            {
                return true;
            }

            return mlir::isa<mlir_ts::ArrayPushOp, mlir_ts::ArrayUnshiftOp>(user) && use.getOperandNumber() == 0 &&
                   !reachableAfter(read, user, builder);
        });
    }

    bool OwnershipInferencePass::isBuiltArrayReturned(mlir::Operation *retain, mlir::Value value)
    {
        auto read = value.getDefiningOp<mlir_ts::LoadOp>();
        auto builder = read ? read.getReference().getDefiningOp<mlir_ts::VariableOp>() : mlir_ts::VariableOp();
        if (!builder || isOwningVariable(builder) || builder.getCaptured().value_or(false) || !builder.getInitializer() ||
            !mlir::isa_and_nonnull<mlir_ts::CreateArrayOp, mlir_ts::NewArrayOp>(builder.getInitializer().getDefiningOp()))
        {
            return false;
        }

        // the read is retained and returned - stored into the function's result slot - and nothing else
        auto returned = false;
        for (auto *user : read->getUsers())
        {
            if (user == retain)
            {
                continue;
            }

            auto storeOp = mlir::dyn_cast<mlir_ts::StoreOp>(user);
            auto resultSlot = storeOp && storeOp.getValue() == read.getResult()
                                  ? storeOp.getReference().getDefiningOp<mlir_ts::VariableOp>()
                                  : mlir_ts::VariableOp();
            auto intoField = storeOp && storeOp.getValue() == read.getResult() &&
                             storeOp.getReference().getDefiningOp<mlir_ts::PropertyRefOp>();
            if (!((resultSlot && isResultSlot(resultSlot)) || intoField) || returned)
            {
                return false;
            }

            returned = true;
        }

        if (!returned)
        {
            return false;
        }

        // only filled, and only before it is read
        return llvm::all_of(builder->getUses(), [&](mlir::OpOperand &use) {
            auto *user = use.getOwner();
            if (user == read)
            {
                return true;
            }

            if (reachableAfter(read, user, builder))
            {
                return false;
            }

            if (auto loadOp = mlir::dyn_cast<mlir_ts::LoadOp>(user))
            {
                return llvm::all_of(loadOp->getUsers(), [](mlir::Operation *loadUser) {
                    return mlir::isa<mlir_ts::ThisSymbolRefOp>(loadUser) && loadUser->use_empty();
                });
            }

            return mlir::isa<mlir_ts::ArrayPushOp, mlir_ts::ArrayUnshiftOp>(user) && use.getOperandNumber() == 0;
        });
    }

    mlir::Operation *OwnershipInferencePass::onlyRetainSlotOf(mlir_ts::VariableOp varOp)
    {
        mlir::Operation *found = nullptr;
        for (auto *user : varOp->getUsers())
        {
            if (mlir::isa<mlir_ts::RetainSlotOp>(user))
            {
                if (found)
                {
                    return nullptr;
                }

                found = user;
            }
        }

        return found;
    }

    bool OwnershipInferencePass::isResultSlot(mlir_ts::VariableOp varOp)
    {
        if (isOwningVariable(varOp) || varOp.getCaptured().value_or(false))
        {
            return false;
        }

        return llvm::all_of(varOp->getUsers(), [&](mlir::Operation *user) {
            if (auto storeOp = mlir::dyn_cast<mlir_ts::StoreOp>(user))
            {
                return storeOp.getReference() == varOp.getResult();
            }

            auto loadOp = mlir::dyn_cast<mlir_ts::LoadOp>(user);
            return loadOp && llvm::all_of(loadOp->getUsers(), [](mlir::Operation *loadUser) {
                       return mlir::isa<mlir_ts::ReturnInternalOp>(loadUser);
                   });
        });
    }

    mlir::Operation *OwnershipInferencePass::recordRetainOfStore(mlir::Operation *taker)
    {
        auto storeOp = mlir::dyn_cast<mlir_ts::StoreOp>(taker);
        auto propertyRefOp = storeOp ? storeOp.getReference().getDefiningOp<mlir_ts::PropertyRefOp>() : mlir_ts::PropertyRefOp();
        auto varOp = propertyRefOp ? propertyRefOp->getOperand(0).getDefiningOp<mlir_ts::VariableOp>() : mlir_ts::VariableOp();
        auto *retain = varOp ? recordRetainOf(varOp) : nullptr;
        if (!retain || retain->getBlock() != taker->getBlock() || !taker->isBeforeInBlock(retain))
        {
            return nullptr;
        }

        return retain;
    }

    bool OwnershipInferencePass::isImmortalLiteral(mlir::Value value)
    {
        // ...or a value a keeper keeps for good, which is made immortal (ts.MakeImmortal), and a
        // keeper's element
        if (isImmortalGlobalRead(value) || isKeptForGood(value))
        {
            return true;
        }

        auto castOp = rootOf(value).getDefiningOp<mlir_ts::CastOp>();
        return castOp && mlir::isa<mlir_ts::StringType>(castOp.getType()) &&
               castOp.getIn().getDefiningOp<mlir_ts::ConstantOp>();
    }

    bool OwnershipInferencePass::isBorrow(mlir::Operation *user, mlir::Value value)
    {
        // marking a value immortal (a keeper's) reads it; the result is another value, owned by none
        if (mlir::isa<mlir_ts::MakeImmortalOp, mlir_ts::MarkImmortalCellOp>(user))
        {
            return true;
        }

        if (mlir::isa<mlir_ts::PrintOp, mlir_ts::PropertyRefOp, mlir_ts::ElementRefOp, mlir_ts::LengthOfOp,
                      mlir_ts::ThisSymbolRefOp, mlir_ts::VirtualSymbolRefOp, mlir_ts::ThisVirtualSymbolRefOp,
                      mlir_ts::InterfaceSymbolRefOp, mlir_ts::GetThisOp, mlir_ts::GetMethodOp,
                      mlir_ts::ArithmeticBinaryOp, mlir_ts::LogicalBinaryOp, mlir_ts::StringConcatOp,
                      mlir_ts::StringResizeOp, mlir_ts::StringLengthOp, mlir_ts::StringCopyOp>(user))
        {
            return true;
        }

        // `assert(c, message)` prints the message when it fails, and keeps it either way
        if (mlir::isa<mlir_ts::AssertOp>(user))
        {
            return true;
        }

        // `s.value`'s place and `Shared.count(s)` read the handle and keep nothing (spec 23.1)
        if (mlir::isa<mlir_ts::SharedValueRefOp, mlir_ts::SharedCountOp>(user))
        {
            return true;
        }

        // `--di`'s record of a folded const's value for the debugger: names it, keeps nothing
        if (mlir::isa<mlir_ts::DebugVariableOp>(user))
        {
            return true;
        }

        // a test - `if (d)`, `d?.x`, `typeof u` - asks about the value, and keeps nothing
        if (mlir::isa<mlir_ts::HasValueOp, mlir_ts::GetTypeInfoFromUnionOp, mlir_ts::TypeOfOp,
                      mlir_ts::TypeOfAnyOp>(user))
        {
            return true;
        }

        // ...and so does a cast to `null` or `undefined`, which holds no block: `--di` records the
        // `r` that `if (r == null)` narrows as one
        if (auto castOp = mlir::dyn_cast<mlir_ts::CastOp>(user))
        {
            return mlir::isa<mlir_ts::BooleanType, mlir_ts::NullType, mlir_ts::UndefinedType>(castOp.getType()) ||
                   castOp.getType().isInteger(1) || isBoundForCall(castOp);
        }

        if (mlir::isa<mlir_ts::ExtractInterfaceThisOp>(user))
        {
            return isBoundForCall(user);
        }

        // arguments are borrowed, but one the callee keeps (`__own_params`) is taken; a callee
        // with no facts that keeps one retains it, and that retain is its own error
        if (isCall(user))
        {
            return !isKeptArgument(user, value);
        }

        // the array being pushed onto is written, not taken; what is inserted is taken
        if (auto pushOp = mlir::dyn_cast<mlir_ts::ArrayPushOp>(user))
        {
            return pushOp.getOp() == value && !llvm::is_contained(pushOp.getItems(), value);
        }

        if (auto unshiftOp = mlir::dyn_cast<mlir_ts::ArrayUnshiftOp>(user))
        {
            return unshiftOp.getOp() == value && !llvm::is_contained(unshiftOp.getItems(), value);
        }

        if (auto spliceOp = mlir::dyn_cast<mlir_ts::ArraySpliceOp>(user))
        {
            return spliceOp.getOp() == value && !llvm::is_contained(spliceOp.getItems(), value);
        }

        // a declaration that does not own what it holds only reads it
        if (auto varOp = mlir::dyn_cast<mlir_ts::VariableOp>(user))
        {
            return !isOwningVariable(varOp);
        }

        // the box of a closure that borrows what it captures holds a copy that owns nothing; the
        // closure's own uses are the borrow's (walkBorrowed, decideClosures)
        if (mlir::isa<mlir_ts::StoreOp>(user) && user->hasAttr(OWN_CAPTURE_BORROW_ATTR_NAME))
        {
            return true;
        }

        // a store through a `ts.New`'d `value_ref` fills its block and keeps nothing of it
        if (auto storeOp = mlir::dyn_cast<mlir_ts::StoreOp>(user))
        {
            return mlir::isa<mlir_ts::ValueRefType>(value.getType()) && storeOp.getReference() == value &&
                   storeOp.getValue() != value;
        }

        return false;
    }

    bool OwnershipInferencePass::isBoundForCall(mlir::Operation *thisOp)
    {
        if (thisOp->getNumResults() != 1 || !mlir::isa<mlir_ts::OpaqueType>(thisOp->getResult(0).getType()) ||
            thisOp->use_empty())
        {
            return false;
        }

        return llvm::all_of(thisOp->getUses(), [](mlir::OpOperand &use) {
            auto boundOp = mlir::dyn_cast<mlir_ts::CreateBoundFunctionOp>(use.getOwner());
            if (!boundOp || boundOp.getThisVal() != use.get())
            {
                return false;
            }

            return llvm::all_of(boundOp->getUsers(), [](mlir::Operation *boundUser) {
                if (mlir::isa<mlir_ts::GetMethodOp>(boundUser))
                {
                    return true;
                }

                return mlir::isa<mlir_ts::GetThisOp>(boundUser) &&
                       llvm::all_of(boundUser->getUsers(), [](mlir::Operation *thisUser) { return isCall(thisUser); });
            });
        });
    }

    bool OwnershipInferencePass::isKeptArgument(mlir::Operation *call, mlir::Value value)
    {
        auto args = callArgs(call);
        return llvm::any_of(ownedParams(call), [&](int32_t index) {
            return static_cast<size_t>(index) < args.size() && args[index] == value;
        });
    }

    bool OwnershipInferencePass::checkValueMoves(mlir::Value value, llvm::SetVector<mlir::Operation *> &toErase)
    {
        // Nothing to move: a number owns no block, and a string literal is the immortal global,
        // which any number of places may hold. Under --opt, CSE merges identical literals before
        // this pass, so one such value is routinely stored into several places. A read out of a
        // container owns nothing either: checkPlaceRead decides it. Nor does other data nothing
        // owns: an empty optional, a function, a tuple of literals (isConstantData).
        if (!ownsHeap(value) || isImmortalLiteral(value) || isConstantData(value) || placeReadOf(value))
        {
            return true;
        }

        // A temporary's releases are one owner, one release on each way out. Two on one path are
        // two owners: each view rc gave a reference of its own (`use(o); use(o)`, with `o` a
        // folded object literal and `use` taking an interface: two `ts.NewInterface`s of one block).
        if (!checkReleasedOnce(value))
        {
            return false;
        }

        if (quietly([&] { return movesOf(value, toErase); }))
        {
            return true;
        }

        // Not a move. Every taker a `let` that can borrow, and the temporary's own release the
        // owner that bounds them (spec 2.2, verdict 2) - else the move's error stands.
        llvm::SmallVector<std::pair<mlir::Operation *, mlir::Value>> notReads;
        llvm::SmallVector<mlir::Operation *> releases;
        forEachUse(value, [&](mlir::Operation *user, mlir::Value used) {
            if (mlir::isa<mlir_ts::RetainOp>(user))
            {
                return;
            }

            if (mlir::isa<mlir_ts::ReleaseOp>(user))
            {
                releases.push_back(user);
            }
            else if (!isBorrow(user, used))
            {
                notReads.push_back({user, used});
            }
        });

        // a merge that only borrows it (movesOf) takes nothing; a field taken out of a record (`let [a, b] =
        // pair()`) is taken by what takes the field
        llvm::SmallVector<mlir::Operation *> takers;
        for (auto [user, used] : notReads)
        {
            if (mlir::isa<mlir_ts::ExtractPropertyOp>(user))
            {
                fieldTakersOf(user, takers);
                continue;
            }

            llvm::SmallVector<BorrowUse> uses;
            if (!borrowsThroughMerge(user, used, value.getDefiningOp(), releases, uses, toErase))
            {
                takers.push_back(user);
            }
        }

        if (!releases.empty() && !takers.empty() &&
            llvm::all_of(takers, [](mlir::Operation *taker) { return !!borrowerOf(taker); }))
        {
            auto owner = ownerName(value);
            auto ok = true;
            for (auto *taker : takers)
            {
                ok = tryBorrow(borrowerOf(taker), releases, owner, toErase) && ok;
            }

            return ok;
        }

        return movesOf(value, toErase);
    }

    bool OwnershipInferencePass::checkReleasedOnce(mlir::Value value)
    {
        llvm::SmallVector<mlir::Operation *> releases;
        forEachUse(value, [&](mlir::Operation *user, mlir::Value) {
            if (mlir::isa<mlir_ts::ReleaseOp>(user))
            {
                releases.push_back(user);
            }
        });

        auto *definition = value.getDefiningOp();
        for (auto *first : releases)
        {
            for (auto *second : releases)
            {
                if (first != second && reachableAfter(first, second, definition))
                {
                    // the view the second release gives back, where rc took its reference
                    auto *view = second->getOperand(0).getDefiningOp();
                    reportSecondReference(view ? view : second, value);
                    return false;
                }
            }
        }

        return true;
    }

    llvm::SmallVector<mlir::Value, 2> OwnershipInferencePass::mergedValuesOf(mlir::Operation *user, mlir::Value used)
    {
        llvm::SmallVector<mlir::Value, 2> merged;
        if (auto selectOp = mlir::dyn_cast<mlir::arith::SelectOp>(user))
        {
            if (used != selectOp.getCondition())
            {
                merged.push_back(selectOp.getResult());
            }

            return merged;
        }

        auto branchOp = mlir::dyn_cast<mlir::BranchOpInterface>(user);
        if (!branchOp)
        {
            return merged;
        }

        for (auto &operand : user->getOpOperands())
        {
            if (operand.get() != used)
            {
                continue;
            }

            auto argument = branchOp.getSuccessorBlockArgument(operand.getOperandNumber());
            if (!argument)
            {
                return {}; // the branch's own operand - its condition - is not a merge
            }

            merged.push_back(*argument);
        }

        return merged;
    }

    bool OwnershipInferencePass::borrowsThroughMerge(mlir::Operation *user, mlir::Value used, mlir::Operation *definition,
                                                     llvm::ArrayRef<mlir::Operation *> releases,
                                                     llvm::SmallVectorImpl<BorrowUse> &uses,
                                                     llvm::SetVector<mlir::Operation *> &toErase)
    {
        auto merged = mergedValuesOf(user, used);
        if (merged.empty())
        {
            return false;
        }

        llvm::SmallVector<mlir::Operation *, 1> made;
        if (definition)
        {
            made.push_back(definition);
        }

        llvm::SmallVector<mlir_ts::VariableOp> borrowers;
        auto isLet = [&](mlir::Operation *varUser, llvm::ArrayRef<mlir::Operation *>) {
            auto borrower = mlir::isa<mlir_ts::VariableOp>(varUser) ? borrowerOf(varUser) : mlir_ts::VariableOp();
            if (borrower)
            {
                borrowers.push_back(borrower);
            }

            return !!borrower;
        };

        llvm::SmallVector<BorrowUse> found;
        for (auto value : merged)
        {
            // a store, a return, a capture or a retain of the merged value keeps it: the merge is a
            // move, decided as any other taker
            if (walkBorrowed(value, made, found, isLet))
            {
                return false;
            }
        }

        llvm::SmallVector<mlir::Operation *> bookkeeping;
        for (auto borrower : borrowers)
        {
            if (collectBorrowerUses(borrower, borrower.getOperation(), found, bookkeeping))
            {
                return false;
            }
        }

        for (auto *release : releases)
        {
            for (auto &use : found)
            {
                if (reachableAfter(release, use.op, use.kills))
                {
                    return false;
                }
            }
        }

        uses.append(found.begin(), found.end());
        for (auto *op : bookkeeping)
        {
            toErase.insert(op);
            if (mlir::isa<mlir_ts::RetainSlotOp>(op))
            {
                mergeBorrowRetains.insert(op);
            }
        }

        return true;
    }

    bool OwnershipInferencePass::fieldOnlyRead(mlir::Operation *extract, llvm::SmallVectorImpl<mlir::Operation *> &reads)
    {
        auto onlyRead = true;
        if (holdsNoBlock(extract->getResult(0)))
        {
            reads.push_back(extract);
            return true;
        }

        forEachUse(extract->getResult(0), [&](mlir::Operation *user, mlir::Value used) {
            if (mlir::isa<mlir_ts::ExtractPropertyOp>(user))
            {
                onlyRead = fieldOnlyRead(user, reads) && onlyRead;
                return;
            }

            if (!isBorrow(user, used))
            {
                onlyRead = false;
                return;
            }

            reads.push_back(user);
        });

        return onlyRead;
    }

    void OwnershipInferencePass::fieldTakersOf(mlir::Operation *extract, llvm::SmallVectorImpl<mlir::Operation *> &takers)
    {
        // a number taken out takes nothing of the record: `print(h.dist)` converts it to a string
        if (holdsNoBlock(extract->getResult(0)))
        {
            return;
        }

        forEachUse(extract->getResult(0), [&](mlir::Operation *user, mlir::Value used) {
            if (mlir::isa<mlir_ts::ExtractPropertyOp>(user))
            {
                fieldTakersOf(user, takers);
            }
            else if (!mlir::isa<mlir_ts::RetainOp>(user) && !isBorrow(user, used))
            {
                takers.push_back(user);
            }
        });
    }

    bool OwnershipInferencePass::movesOf(mlir::Value value, llvm::SetVector<mlir::Operation *> &toErase)
    {
        llvm::SmallVector<mlir::Operation *> takers;
        llvm::SmallVector<mlir::Operation *> releases;
        llvm::SmallVector<mlir::Operation *> reads;
        llvm::DenseMap<mlir::Operation *, mlir::Value> readOf;
        llvm::SmallVector<std::pair<mlir::Operation *, mlir::Value>> merges;
        forEachUse(value, [&](mlir::Operation *user, mlir::Value used) {
            if (mlir::isa<mlir_ts::RetainOp>(user))
            {
                return;
            }

            llvm::SmallVector<mlir::Operation *> fieldReads;
            if (mlir::isa<mlir_ts::ReleaseOp>(user))
            {
                releases.push_back(user);
            }
            else if (isBorrow(user, used))
            {
                reads.push_back(user);
                readOf.try_emplace(user, used);
            }
            else if (mlir::isa<mlir_ts::ExtractPropertyOp>(user) && fieldOnlyRead(user, fieldReads))
            {
                reads.push_back(user);
                reads.append(fieldReads.begin(), fieldReads.end());
            }
            else if (!mergedValuesOf(user, used).empty())
            {
                merges.push_back({user, used});
            }
            else
            {
                takers.push_back(user);
            }
        });

        auto *definition = value.getDefiningOp();

        // `c ? h : h2` that is only read borrows `h` and `h2`, and in a loop moves neither. What
        // reads the merged value reads this one, so each such read can be a use after a move of
        // it like any other.
        for (auto [merge, used] : merges)
        {
            llvm::SmallVector<BorrowUse> uses;
            if (!borrowsThroughMerge(merge, used, definition, releases, uses, toErase))
            {
                takers.push_back(merge);
                continue;
            }

            reads.push_back(merge);
            for (auto &use : uses)
            {
                reads.push_back(use.op);
            }
        }

        auto name = describe(value);
        llvm::SmallVector<mlir::Operation *> moved;
        for (auto *taker : takers)
        {
            if (takerLoops(taker, definition))
            {
                reportMovedInLoop(taker, name);
                return false;
            }

            llvm::SmallVector<mlir::Operation *, 1> made;
            if (definition)
            {
                made.push_back(definition);
            }

            // the call that keeps it, given it again through another operand
            if (auto *again = callGivenAgain(taker, value, {value}))
            {
                reportUseAfterMove(again, taker, name);
                return false;
            }

            for (auto *use : llvm::concat<mlir::Operation *const>(takers, reads))
            {
                if (use != taker && reachableAfter(taker, use, definition))
                {
                    // a read through the new owner, while it holds the value (spec 29)
                    if (auto found = readOf.find(use);
                        found != readOf.end() && readsThroughOwner(taker, found->second, use, made))
                    {
                        continue;
                    }

                    reportUseAfterMove(use, taker, name);
                    return false;
                }
            }

            if (releases.empty())
            {
                continue; // rc handed the taker the value's own reference: nothing to cancel
            }

            if (!takerAcquires(taker, value))
            {
                reportSecondReference(taker, value);
                return false;
            }

            auto *acquire = acquirePoint(taker, value);
            for (auto *release : releases)
            {
                if (!reachableAfter(acquire, release, definition))
                {
                    continue;
                }

                if (!dominance->properlyDominates(acquire, release))
                {
                    reportMovedOnSomePaths(taker, release, name);
                    return false;
                }

                moved.push_back(release);
            }
        }

        toErase.insert(moved.begin(), moved.end());
        return true;
    }

    mlir::Operation *OwnershipInferencePass::callGivenAgain(mlir::Operation *taker, mlir::Value moved, llvm::ArrayRef<mlir::Value> sources)
    {
        if (!isCall(taker) || ownedParams(taker).empty())
        {
            return nullptr;
        }

        auto args = callArgs(taker);
        auto kept = ownedParams(taker);
        auto movedRoot = rootOf(moved);
        // one kept position is the move itself; the value at a second one (`f(n, n)` with both
        // kept) is a second owner, a use after the first move like any other operand
        auto skipped = false;
        for (auto &operand : taker->getOpOperands())
        {
            // the moved argument itself, where the callee keeps it
            auto number = operand.getOperandNumber();
            if (!skipped && !args.empty() && number >= args.getBeginOperandIndex() &&
                number < args.getBeginOperandIndex() + args.size() &&
                llvm::is_contained(kept, static_cast<int32_t>(number - args.getBeginOperandIndex())) &&
                rootOf(operand.get()) == movedRoot)
            {
                skipped = true;
                continue;
            }

            for (auto source : sources)
            {
                if (derivedFrom(source).contains(operand.get()))
                {
                    return taker;
                }
            }
        }

        return nullptr;
    }

    bool OwnershipInferencePass::takerLoops(mlir::Operation *taker, mlir::Operation *definition)
    {
        auto &body = getFunction().getFunctionBody();
        auto *top = body.findAncestorOpInRegion(*taker);
        if (!top)
        {
            return true;
        }

        auto *kill = definition ? body.findAncestorOpInRegion(*definition) : nullptr;
        llvm::SmallPtrSet<mlir::Block *, 16> seen;
        llvm::SmallVector<mlir::Block *> work(top->getBlock()->getSuccessors().begin(),
                                               top->getBlock()->getSuccessors().end());
        while (!work.empty())
        {
            auto *block = work.pop_back_val();
            if (!seen.insert(block).second)
            {
                continue;
            }

            if (kill && kill->getBlock() == block &&
                (block != top->getBlock() || kill->isBeforeInBlock(top)))
            {
                continue; // the value is made again before the taker runs again
            }

            if (block == top->getBlock())
            {
                return true;
            }

            work.append(block->getSuccessors().begin(), block->getSuccessors().end());
        }

        return false;
    }

    bool OwnershipInferencePass::reachableAfter(mlir::Operation *from, mlir::Operation *to, mlir::Operation *kill)
    {
        return reachableAfter(from, to, kill ? llvm::ArrayRef<mlir::Operation *>(kill) : llvm::ArrayRef<mlir::Operation *>());
    }

    bool OwnershipInferencePass::orderedInBody(mlir::Operation *from, mlir::Operation *to)
    {
        auto &body = getFunction().getFunctionBody();
        from = body.findAncestorOpInRegion(*from);
        to = body.findAncestorOpInRegion(*to);
        return from && to && from != to;
    }

    bool OwnershipInferencePass::reachableAfter(mlir::Operation *from, mlir::Operation *to, llvm::ArrayRef<mlir::Operation *> kills)
    {
        if (llvm::is_contained(kills, to))
        {
            return false;
        }

        auto &body = getFunction().getFunctionBody();
        from = body.findAncestorOpInRegion(*from);
        to = body.findAncestorOpInRegion(*to);
        llvm::SmallVector<mlir::Operation *, 4> killAt;
        for (auto *kill : kills)
        {
            if (auto *top = body.findAncestorOpInRegion(*kill))
            {
                killAt.push_back(top);
            }
        }

        if (!from || !to)
        {
            return true;
        }

        if (from == to)
        {
            // Two distinct ops that share one ancestor statement: the order inside it is not
            // known here, and "reachable" can only turn a program into an error.
            return true;
        }

        // is there a kill in `block` after `after` (from its start when null) and before `before`
        // (to its end when null)?
        auto killsIn = [&](mlir::Block *block, mlir::Operation *after, mlir::Operation *before) {
            return llvm::any_of(killAt, [&](mlir::Operation *kill) {
                return kill->getBlock() == block && (!after || after->isBeforeInBlock(kill)) &&
                       (!before || kill->isBeforeInBlock(before));
            });
        };

        auto *fromBlock = from->getBlock();
        auto *toBlock = to->getBlock();
        if (fromBlock == toBlock && from->isBeforeInBlock(to))
        {
            return !killsIn(fromBlock, from, to);
        }

        // every way out of this block runs a kill first
        if (killsIn(fromBlock, from, nullptr))
        {
            return false;
        }

        llvm::SmallPtrSet<mlir::Block *, 16> seen;
        llvm::SmallVector<mlir::Block *> work(fromBlock->getSuccessors().begin(), fromBlock->getSuccessors().end());
        while (!work.empty())
        {
            auto *block = work.pop_back_val();
            if (!seen.insert(block).second)
            {
                continue;
            }

            if (block == toBlock && !killsIn(block, nullptr, to))
            {
                return true;
            }

            if (killsIn(block, nullptr, nullptr))
            {
                continue;
            }

            work.append(block->getSuccessors().begin(), block->getSuccessors().end());
        }

        return false;
    }

    mlir::Operation *OwnershipInferencePass::takerOf(mlir::Operation *retain, mlir::Value value)
    {
        if (auto retainSlotOp = mlir::dyn_cast<mlir_ts::RetainSlotOp>(retain))
        {
            return retainSlotOp.getSlot().getDefiningOp();
        }

        // a call that keeps the argument: rc's retain for it is in the callee
        if (isCall(retain))
        {
            return retain;
        }

        mlir::Operation *taker = nullptr;
        auto several = false;
        forEachUse(value, [&](mlir::Operation *user, mlir::Value used) {
            if (mlir::isa<mlir_ts::RetainOp>(user) || isBorrow(user, used))
            {
                return;
            }

            several = several || taker;
            taker = user;
        });

        // rc retained this read and hands the receiver the next read of the same place
        if (!taker && !several)
        {
            if (auto twin = twinReadAfter(retain, value))
            {
                return takerOf(retain, twin);
            }
        }

        return several ? nullptr : taker; // two takers of one read: not a move this phase proves
    }

    mlir::Operation *OwnershipInferencePass::soleUser(mlir::Value value)
    {
        mlir::Operation *found = nullptr;
        for (auto *user : value.getUsers())
        {
            if (mlir::isa<mlir_ts::DebugVariableOp>(user))
            {
                continue;
            }

            if (found)
            {
                return nullptr;
            }

            found = user;
        }

        return found;
    }

    mlir::Value OwnershipInferencePass::twinReadAfter(mlir::Operation *retain, mlir::Value value)
    {
        auto first = value.getDefiningOp<mlir_ts::LoadOp>();
        if (!first || !mlir::isa<mlir_ts::RetainOp>(retain) || soleUser(value) != retain || retain->getBlock() != first->getBlock())
        {
            return {};
        }

        auto place = first.getReference();
        for (auto *op = retain->getNextNode(); op; op = op->getNextNode())
        {
            if (auto loadOp = mlir::dyn_cast<mlir_ts::LoadOp>(op); loadOp && loadOp.getReference() == place)
            {
                return loadOp.getResult();
            }

            // anything else given the place may write through it - a store, a push - and a call may
            // write anything
            if (llvm::is_contained(op->getOperands(), place) || isCall(op))
            {
                return {};
            }
        }

        return {};
    }

    bool OwnershipInferencePass::checkSlotMove(mlir::Operation *retain, mlir::Value value, mlir_ts::LoadOp load,
                       llvm::SetVector<mlir::Operation *> &toErase, bool placeReleases)
    {
        auto slot = load.getReference();
        auto varOp = slot.getDefiningOp<mlir_ts::VariableOp>();
        auto name = varName(varOp);

        auto *taker = takerOf(retain, value);
        if (!taker)
        {
            reportSecondReference(retain);
            return false;
        }

        auto assignments = slotAssignments(slot);
        llvm::SmallVector<mlir::Operation *> kills(assignments.begin(), assignments.end());
        if (varOp)
        {
            kills.push_back(varOp);
        }

        // the move's own read, met again around a loop the slot was declared outside of
        if (reachableAfter(taker, load, kills))
        {
            reportMovedInLoop(taker, name);
            return false;
        }

        // the slot given another value on some path between the move's read and its taker - inside
        // the call's own argument list, `f(n, (n = mk(7), 0))`, on one arm of it, `f(n, c ? (n =
        // mk(7), 0) : 0)`: the assignment's release frees what the read holds before the callee takes
        // it, so the assignment is a use of the slot after the move. The taker may be an assignment
        // of the slot itself (`n = n`): it is the destination, not between. A read and a taker
        // reachableAfter cannot order (one statement kept as a region, or outside the body) refuse.
        {
            llvm::SmallVector<mlir::Operation *, 2> readAgain{load.getOperation()};
            if (varOp)
            {
                readAgain.push_back(varOp);
            }

            for (auto *assignment : assignments)
            {
                if (assignment != taker && reachableAfter(load, assignment, varOp) &&
                    reachableAfter(assignment, taker, readAgain))
                {
                    reportUseAfterMove(assignment, taker, name);
                    return false;
                }
            }

            llvm::SmallVector<mlir::Operation *> killsBetween(kills);
            llvm::erase(killsBetween, taker);
            if (!orderedInBody(load, taker) || !reachableAfter(load, taker, killsBetween))
            {
                reportUseAfterMove(taker, taker, name);
                return false;
            }
        }

        // the call that keeps it, given the slot's value again through another operand: a read of
        // the slot that reaches the call without crossing an assignment of it (not one made before
        // an assignment gave the slot its current value, `const arr = [n]; n = mk(5); f(n, arr)`);
        // since the move's own read reaches the call so too, such a read holds the moved value
        llvm::SmallVector<mlir::Value> slotReads;
        for (auto *user : slot.getUsers())
        {
            if (auto readOp = mlir::dyn_cast<mlir_ts::LoadOp>(user); readOp && reachableAfter(readOp, taker, kills))
            {
                slotReads.push_back(readOp.getResult());
            }
        }

        if (auto *again = callGivenAgain(taker, value, slotReads))
        {
            reportUseAfterMove(again, taker, name);
            return false;
        }

        // Every read of the slot is a use of `a`, and so is every use of what a read returned: a
        // `const b = a` is folded into its load, so `b.x` after the move reads the moved value
        // through a load that came before it. A read through the new owner, while it holds the
        // value, is not such a use (spec 29).
        auto throughOwner = [&](mlir::Operation *user, mlir::Value used) {
            return readsThroughOwner(taker, used, user, kills);
        };

        for (auto *use : slot.getUsers())
        {
            if (mlir::isa<mlir_ts::ReleaseSlotOp>(use))
            {
                continue;
            }

            if (use != load.getOperation() && reachableAfter(taker, use, kills))
            {
                auto readOp = mlir::dyn_cast<mlir_ts::LoadOp>(use);
                if (!readOp || !onlyReadThroughOwner(readOp.getResult(), throughOwner))
                {
                    reportUseAfterMove(use, taker, name);
                    return false;
                }

                continue;
            }

            if (auto readOp = mlir::dyn_cast<mlir_ts::LoadOp>(use))
            {
                if (auto *late = readUsedAfter(readOp, retain, taker, varOp, throughOwner))
                {
                    reportUseAfterMove(late, taker, name);
                    return false;
                }
            }
        }

        auto *acquire = mlir::isa<mlir_ts::RetainSlotOp>(retain) ? taker : retain;
        llvm::SmallVector<mlir::Operation *> moved;
        for (auto *use : slot.getUsers())
        {
            if (!mlir::isa<mlir_ts::ReleaseSlotOp>(use) || !reachableAfter(acquire, use, kills))
            {
                continue;
            }

            // a release that one path reaches holding the moved value and another holding a new
            // one an assignment after the move stored: there is no single answer for it
            auto someMoved = !dominance->properlyDominates(acquire, use);
            for (auto *assignment : assignments)
            {
                llvm::SmallVector<mlir::Operation *, 2> movedAgain{load.getOperation()};
                if (varOp)
                {
                    movedAgain.push_back(varOp);
                }

                someMoved = someMoved ||
                            (reachableAfter(acquire, assignment, varOp) && reachableAfter(assignment, use, movedAgain));
            }

            if (someMoved)
            {
                // ...but the paths that did not move it can release it before they join the
                // others (spec 30)
                if (placeReleases && placeOnUnmovedEdges(varOp, acquire, retain, taker, assignments, toErase))
                {
                    return true;
                }

                reportMovedOnSomePaths(taker, use, name, placeReleases ? joinedUse : nullptr);
                return false;
            }

            moved.push_back(use);
        }

        toErase.insert(moved.begin(), moved.end());
        return true;
    }

} // namespace own_inference
