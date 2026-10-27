// OwnershipSignaturePass (see OwnershipSignaturePass.h): globals - what an initializer moves in,
// immortal globals (spec 25), globals given a borrow, calls in a global's initializer, the globals a
// call may assign.

#include "OwnershipSignaturePass.h"

#define DEBUG_TYPE "own"

namespace own_signature
{

    bool OwnershipSignaturePass::movesIntoResult(mlir::Value value)
    {
        return llvm::all_of(value.getUses(), [](mlir::OpOperand &use) {
            auto *user = use.getOwner();
            if (mlir::isa<mlir_ts::RetainOp, mlir_ts::GlobalResultOp>(user))
            {
                return true;
            }

            // given to `new Shared(x)`: it moves into the handle's block, which owns it from then on
            // whatever becomes of the handle - rc counts the handle (spec 23.2, 26)
            if (auto sharedNewOp = mlir::dyn_cast<mlir_ts::SharedNewOp>(user))
            {
                return sharedNewOp.getValue() == use.get();
            }

            if ((isView(user) && use.getOperandNumber() == 0) || mlir::isa<mlir_ts::CreateArrayOp>(user))
            {
                return movesIntoResult(user->getResult(0));
            }

            // copied into a block `ts.New` made, which itself moves in: an object literal's box
            if (auto storeOp = mlir::dyn_cast<mlir_ts::StoreOp>(user); storeOp && use.getOperandNumber() == 0)
            {
                auto block = storeOp.getReference();
                return block.getDefiningOp<mlir_ts::NewOp>() &&
                       llvm::all_of(block.getUses(), [&](mlir::OpOperand &blockUse) {
                           return blockUse.getOwner() == storeOp.getOperation() ||
                                  (isView(blockUse.getOwner()) && movesIntoResult(blockUse.getOwner()->getResult(0)));
                       });
            }

            return false;
        });
    }

    bool OwnershipSignaturePass::holdsOnlyConstants(mlir::Value value)
    {
        auto loadOp = value.getDefiningOp<mlir_ts::LoadOp>();
        auto varOp = loadOp ? loadOp.getReference().getDefiningOp<mlir_ts::VariableOp>() : mlir_ts::VariableOp();
        if (!varOp || !varOp.getInitializer() || !varOp.getInitializer().getDefiningOp<mlir_ts::ConstantOp>())
        {
            return false;
        }

        return llvm::all_of(varOp->getUsers(), [](mlir::Operation *user) {
            if (mlir::isa<mlir_ts::LoadOp>(user))
            {
                return true;
            }

            auto propertyRefOp = mlir::dyn_cast<mlir_ts::PropertyRefOp>(user);
            return propertyRefOp && llvm::all_of(propertyRefOp->getUsers(), [&](mlir::Operation *fieldUser) {
                       auto storeOp = mlir::dyn_cast<mlir_ts::StoreOp>(fieldUser);
                       return mlir::isa<mlir_ts::LoadOp>(fieldUser) ||
                              (storeOp && storeOp.getReference() == propertyRefOp.getResult() &&
                               holdsNoBlock(storeOp.getValue()));
                   });
        });
    }

    bool OwnershipSignaturePass::mayRunAgain(mlir::Operation *op)
    {
        for (auto *at = op; at && !mlir::isa<mlir_ts::FuncOp>(at); at = at->getParentOp())
        {
            if (mlir::isa<mlir::LoopLikeOpInterface>(at))
            {
                return true;
            }

            auto *block = at->getBlock();
            if (!block)
            {
                return true; // not in a function: nothing is known of how often it runs
            }

            llvm::SmallPtrSet<mlir::Block *, 16> seen;
            llvm::SmallVector<mlir::Block *> work(block->getSuccessors().begin(), block->getSuccessors().end());
            while (!work.empty())
            {
                auto *next = work.pop_back_val();
                if (next == block)
                {
                    return true;
                }

                if (seen.insert(next).second)
                {
                    work.append(next->getSuccessors().begin(), next->getSuccessors().end());
                }
            }
        }

        return false;
    }

    void OwnershipSignaturePass::findImmortalGlobals(mlir::ModuleOp module)
    {
        // the static constructors, which run once before the program
        llvm::StringSet<> constructors;
        module.walk([&](mlir_ts::GlobalConstructorOp constructorOp) { constructors.insert(constructorOp.getGlobalName()); });

        // a static's address taken out of a class's vtable (`k.w = x`, `k: typeof K`) and written
        // through: it may be any class's static the vtable slot holds, so no static is immortal.
        // MLIRGen makes a `ts.ThisVirtualSymbolRef` only for a method (a bound function, never a
        // `!ts.ref`), so no program reaches this through one today; it is listed for an IR that would.
        auto vtableWritten = false;
        module.walk([&](mlir::Operation *op) {
            if (mlir::isa<mlir_ts::VirtualSymbolRefOp, mlir_ts::ThisVirtualSymbolRefOp>(op) &&
                mlir::isa<mlir_ts::RefType>(op->getResult(0).getType()) &&
                !llvm::all_of(op->getUsers(), [](mlir::Operation *user) { return mlir::isa<mlir_ts::LoadOp>(user); }))
            {
                vtableWritten = true;
            }
        });

        auto uses = mlir::SymbolTable::getSymbolUses(&module.getBodyRegion());
        if (!uses)
        {
            return; // an op this pass cannot read symbols through: nothing is known never written
        }

        llvm::StringMap<llvm::SmallVector<mlir::Operation *>> usesOf;
        for (auto &use : *uses)
        {
            usesOf[use.getSymbolRef().getRootReference().getValue()].push_back(use.getUser());
        }

        // each candidate with the value its initializer gives it, and the store that does, if any
        llvm::MapVector<mlir_ts::GlobalOp, std::pair<mlir::Value, mlir_ts::StoreOp>> candidates;
        auto mth = typeHelper(&getContext());
        module.walk([&](mlir_ts::GlobalOp globalOp) {
            if (!mlir::isa<mlir_ts::ClassType>(globalOp.getType()) || !mth.ownsHeapMemory(globalOp.getLoc(), globalOp.getType()) ||
                globalOp->hasAttr("export") || globalOp->hasAttr("import") || globalOp->hasAttr(DLL_EXPORT) ||
                globalOp->hasAttr(DLL_IMPORT) || !globalOp.getInitializerRegion().hasOneBlock())
            {
                return;
            }

            auto resultOp = mlir::dyn_cast<mlir_ts::GlobalResultOp>(globalOp.getInitializerBlock()->getTerminator());
            if (!resultOp || resultOp->getNumOperands() != 1)
            {
                return;
            }

            mlir_ts::StoreOp initStore;
            for (auto *user : usesOf.lookup(globalOp.getSymName()))
            {
                // its vtable's entry, which only a `ts.VirtualSymbolRef` reads (vtableWritten)
                if (mlir::isa<mlir_ts::SymbolRefOp>(user) && user->getParentOfType<mlir_ts::GlobalOp>())
                {
                    if (vtableWritten)
                    {
                        return;
                    }

                    continue;
                }

                if (mlir::isa<mlir_ts::VirtualSymbolRefOp, mlir_ts::ThisVirtualSymbolRefOp>(user))
                {
                    continue; // only loaded, or vtableWritten
                }

                auto addressOfOp = mlir::dyn_cast<mlir_ts::AddressOfOp>(user);
                if (!addressOfOp)
                {
                    return;
                }

                auto released = false;
                auto stored = false;
                for (auto *addressUser : addressOfOp->getUsers())
                {
                    if (mlir::isa<mlir_ts::LoadOp>(addressUser))
                    {
                        continue;
                    }

                    // the old value, the default, given up before the store
                    if (mlir::isa<mlir_ts::ReleaseSlotOp>(addressUser))
                    {
                        released = true;
                        continue;
                    }

                    // the one store, in a static constructor or the program's `main` that nothing
                    // calls (a script's top level), run once
                    auto storeOp = mlir::dyn_cast<mlir_ts::StoreOp>(addressUser);
                    auto funcOp = storeOp ? storeOp->getParentOfType<mlir_ts::FuncOp>() : mlir_ts::FuncOp();
                    auto runsOnce = funcOp && (constructors.contains(funcOp.getSymName()) ||
                                               (funcOp.getSymName() == MAIN_ENTRY_NAME && usesOf.lookup(MAIN_ENTRY_NAME).empty()));
                    if (initStore || !storeOp || storeOp.getReference() != addressOfOp.getResult() ||
                        storeOp.getValue() == addressOfOp.getResult() || !runsOnce || mayRunAgain(storeOp))
                    {
                        return;
                    }

                    initStore = storeOp;
                    stored = true;
                }

                if (released && !stored)
                {
                    return;
                }
            }

            // a static's region yields only its default, which the store replaces
            if (initStore && !regionYieldsDefault(globalOp))
            {
                return;
            }

            candidates[globalOp] = {initStore ? initStore.getValue() : resultOp->getOperand(0), initStore};
        });

        // given a fresh object, nothing, or another immortal global's value: in module order, until
        // no more are found
        llvm::StringSet<> immortal;
        auto madeHere = [&](mlir::Value value) {
            auto root = rootOf(value);
            // a class's `..new` (a regular expression literal is `RegExp..new`) makes the object as
            // `ts.New` does: a library's is a declaration, which runOnOperation marks fresh only after
            // this runs
            auto *def = root.getDefiningOp();
            auto callee = def ? directCallee(def) : mlir::FlatSymbolRefAttr();
            auto constructs = callee && callee.getValue().ends_with("." NEW_METHOD_NAME);
            if (!def || (isCall(def) ? !constructs : !isFresh(root)))
            {
                return false;
            }

            // and not given back before the global takes it
            auto released = false;
            forEachUse(root, [&](mlir::Operation *user, mlir::Value) { released = released || mlir::isa<mlir_ts::ReleaseOp>(user); });
            return !released;
        };
        // isImmortalGlobalRead, of the globals found so far: none is marked yet
        auto readsImmortal = [&](mlir::Value value) {
            auto addressOfOp = globalReadOf(value);
            return addressOfOp && immortal.contains(addressOfOp.getGlobalName());
        };

        for (auto found = true; found;)
        {
            found = false;
            for (auto &[globalOp, init] : candidates)
            {
                auto value = init.first;
                if (!immortal.contains(globalOp.getSymName()) && (madeHere(value) || holdsNoBlock(value) || readsImmortal(value)))
                {
                    immortal.insert(globalOp.getSymName());
                    found = true;
                }
            }
        }

        auto unit = mlir::UnitAttr::get(&getContext());
        for (auto &[globalOp, init] : candidates)
        {
            if (!immortal.contains(globalOp.getSymName()))
            {
                continue;
            }

            globalOp->setAttr(OWN_IMMORTAL_GLOBAL_ATTR_NAME, unit);
            if (init.second)
            {
                init.second->setAttr(OWN_IMMORTAL_GLOBAL_ATTR_NAME, unit);
            }

            for (auto *user : usesOf.lookup(globalOp.getSymName()))
            {
                if (mlir::isa<mlir_ts::AddressOfOp>(user))
                {
                    user->setAttr(OWN_IMMORTAL_GLOBAL_ATTR_NAME, unit);
                }
            }
        }
    }

    void OwnershipSignaturePass::moveIntoGlobals(mlir::ModuleOp module)
    {
        module.walk([&](mlir_ts::GlobalOp globalOp) {
            llvm::SmallVector<mlir_ts::RetainOp> retains;
            globalOp.walk([&](mlir_ts::RetainOp retainOp) { retains.push_back(retainOp); });
            for (auto retainOp : retains)
            {
                // a handle's retain is rc's: the global's count, which own leaves in place (spec 23.2)
                if (touchesHandle(retainOp))
                {
                    continue;
                }

                auto value = retainOp.getReference();

                // an immortal global's value (spec 25): owned by no one, held by any number
                if (isImmortalGlobalRead(value))
                {
                    retainOp.erase();
                    continue;
                }

                // nothing in the region gives the block back, or takes a second reference to it:
                // the global is its one owner
                auto released = false;
                auto retained = 0;
                forEachUse(rootOf(value), [&](mlir::Operation *user, mlir::Value) {
                    released = released || mlir::isa<mlir_ts::ReleaseOp, mlir_ts::ReleaseSlotOp>(user);
                    retained += mlir::isa<mlir_ts::RetainOp>(user);
                });

                if (retained == 1 && !released && (isFresh(value) || holdsOnlyConstants(value) || isConstantData(value)) && movesIntoResult(value))
                {
                    retainOp.erase();
                    continue;
                }

                retainOp.emitError("the initializer of global '")
                    << globalOp.getSymName() << "' takes a second reference; -mm=own cannot prove a move or a borrow here yet";
                signalPassFailure();
            }
        });
    }

    void OwnershipSignaturePass::rejectBorrowedGlobalInitializers(mlir::ModuleOp module)
    {
        module.walk([&](mlir_ts::GlobalOp globalOp) {
            auto *block = globalOp.getInitializerBlock();
            auto resultOp = block && !block->empty() ? mlir::dyn_cast<mlir_ts::GlobalResultOp>(block->getTerminator())
                                                     : mlir_ts::GlobalResultOp();
            if (!resultOp || resultOp->getNumOperands() != 1)
            {
                return;
            }

            auto value = resultOp->getOperand(0);
            while (value.getDefiningOp() && isView(value.getDefiningOp()))
            {
                value = value.getDefiningOp()->getOperand(0);
            }

            auto loadOp = value.getDefiningOp<mlir_ts::LoadOp>();
            auto *place = loadOp ? loadOp.getReference().getDefiningOp() : nullptr;
            // an immortal global's value (spec 25) is owned by no one: any number of globals hold it
            if (!place || !mlir::isa<mlir_ts::AddressOfOp, mlir_ts::PropertyRefOp, mlir_ts::ElementRefOp>(place) ||
                isHandle(value) || isConstantData(value) || isImmortalGlobalRead(value) ||
                !ownsHeap(value.getLoc(), value.getType()))
            {
                return;
            }

            resultOp.emitError("the initializer of global '")
                << globalOp.getSymName()
                << "' reads a value another owner keeps; -mm=own cannot make a global a second owner";
            signalPassFailure();
        });
    }

    void OwnershipSignaturePass::pinAssignedGlobals()
    {
        llvm::DenseMap<mlir::Operation *, llvm::SmallVector<Call *>> callsIn;
        for (auto &call : calls)
        {
            callsIn[call.op->getParentOfType<mlir_ts::FuncOp>()].push_back(&call);
        }

        llvm::DenseMap<mlir::Operation *, llvm::SetVector<mlir::StringRef>> assigns;
        llvm::DenseSet<mlir::Operation *> assignsAny;
        for (auto &entry : functions)
        {
            auto funcOp = entry.second;
            funcOp.walk([&](mlir::Operation *op) {
                mlir::Value slot;
                if (auto storeOp = mlir::dyn_cast<mlir_ts::StoreOp>(op))
                {
                    slot = storeOp.getReference();
                }
                else if (auto releaseSlotOp = mlir::dyn_cast<mlir_ts::ReleaseSlotOp>(op))
                {
                    slot = releaseSlotOp.getSlot();
                }

                if (auto addressOfOp = slot ? slot.getDefiningOp<mlir_ts::AddressOfOp>() : mlir_ts::AddressOfOp())
                {
                    assigns[funcOp].insert(addressOfOp.getGlobalName());
                }
            });
        }

        // what a call may assign, from what its callees do so far
        auto assignedBy = [&](Call &call, llvm::SetVector<mlir::StringRef> &globals) {
            if (!callMayDrop(call.callees))
            {
                return false;
            }

            if (!call.callees.known)
            {
                return true;
            }

            auto any = false;
            for (auto callee : call.callees.funcs)
            {
                any |= assignsAny.contains(callee);
                if (auto found = assigns.find(callee); found != assigns.end())
                {
                    globals.insert(found->second.begin(), found->second.end());
                }
            }

            return any;
        };

        for (auto changed = true; changed;)
        {
            changed = false;
            for (auto &entry : functions)
            {
                auto funcOp = entry.second;
                for (auto *call : callsIn[funcOp])
                {
                    llvm::SetVector<mlir::StringRef> globals;
                    if (assignedBy(*call, globals))
                    {
                        changed |= assignsAny.insert(funcOp).second;
                    }

                    for (auto global : globals)
                    {
                        changed |= assigns[funcOp].insert(global);
                    }
                }
            }
        }

        for (auto &call : calls)
        {
            llvm::SetVector<mlir::StringRef> globals;
            if (assignedBy(call, globals))
            {
                call.op->setAttr(OWN_ASSIGNS_ANY_GLOBAL_ATTR_NAME, mlir::UnitAttr::get(&getContext()));
                continue;
            }

            if (globals.empty())
            {
                continue;
            }

            llvm::SmallVector<mlir::Attribute> names;
            for (auto global : globals)
            {
                names.push_back(mlir::StringAttr::get(&getContext(), global));
            }

            call.op->setAttr(OWN_ASSIGNS_GLOBALS_ATTR_NAME, mlir::ArrayAttr::get(&getContext(), names));
        }
    }

    void OwnershipSignaturePass::rejectGlobalArgumentsDropped()
    {
        for (auto &call : calls)
        {
            if (!callMayDrop(call.callees))
            {
                continue;
            }

            for (auto argument : callArgs(call.op))
            {
                // a method's object: `this` of the method bound to it
                if (auto getThisOp = argument.getDefiningOp<mlir_ts::GetThisOp>())
                {
                    auto *boundOp = getThisOp->getOperand(0).getDefiningOp();
                    if (mlir::isa_and_nonnull<mlir_ts::ThisSymbolRefOp, mlir_ts::ThisVirtualSymbolRefOp>(boundOp))
                    {
                        argument = boundOp->getOperand(0);
                    }
                }

                // an immortal global (spec 25) is written by its initializer only, and that runs once
                // (a static's is a store in its static constructor, outside any loop, which
                // writtenGlobals counts), and no release frees its block
                if (isHandle(argument) || isImmortalGlobalRead(argument) || !ownsHeap(call.op->getLoc(), argument.getType()))
                {
                    continue;
                }

                auto loadOp = rootOf(argument).getDefiningOp<mlir_ts::LoadOp>();
                auto global = loadOp ? loadOp.getReference().getDefiningOp<mlir_ts::AddressOfOp>() : mlir_ts::AddressOfOp();
                if (!global || !writtenGlobals.contains(global.getGlobalName()))
                {
                    continue;
                }

                // only a store to the global releases its block: a callee that assigns other globals, and
                // none it cannot name (pinAssignedGlobals, run before), leaves this one's value as it is
                if (!call.op->hasAttr(OWN_ASSIGNS_ANY_GLOBAL_ATTR_NAME))
                {
                    auto assigned = call.op->getAttrOfType<mlir::ArrayAttr>(OWN_ASSIGNS_GLOBALS_ATTR_NAME);
                    if (!assigned || !llvm::is_contained(assigned.getValue(), mlir::StringAttr::get(&getContext(), global.getGlobalName())))
                    {
                        continue;
                    }
                }

                call.op->emitError("a value read out of the global '")
                    << global.getGlobalName()
                    << "' is lent to a call that may release or overwrite it; -mm=own cannot prove it outlives the call";
                signalPassFailure();
            }
        }
    }

    void OwnershipSignaturePass::ownGlobalInitializerCalls(mlir::ModuleOp module)
    {
        module.walk([&](mlir_ts::GlobalOp globalOp) {
            // collected first: a move erases a release the walk has not reached yet
            llvm::SmallVector<mlir::Operation *> calls;
            globalOp.walk([&](mlir::Operation *op) {
                if (isCall(op))
                {
                    calls.push_back(op);
                }
            });

            for (auto *op : calls)
            {
                auto callees = resolve(op);
                auto facts = callFacts(callees);
                auto calleeName = callees.funcs.empty() ? llvm::StringRef("the callee") : callees.funcs.front().getSymName();
                if (!facts.borrows.empty())
                {
                    op->emitError("the initializer of global '")
                        << globalOp.getSymName() << "' holds the result of '" << calleeName
                        << "', which borrows its argument; a global cannot hold a borrow under -mm=own";
                    signalPassFailure();
                    continue;
                }

                auto args = callArgs(op);
                llvm::SmallPtrSet<mlir::Value, 4> given;
                for (auto index : facts.owned)
                {
                    if (index < 0 || index >= static_cast<int32_t>(args.size()) ||
                        !given.insert(rootOf(args[index])).second || !moveIntoCallee(op, args[index]))
                    {
                        op->emitError("the initializer of global '")
                            << globalOp.getSymName() << "' gives '" << calleeName
                            << "' a value it keeps, but -mm=own cannot move it there";
                        signalPassFailure();
                        break;
                    }
                }
            }
        });
    }

    bool OwnershipSignaturePass::moveIntoCallee(mlir::Operation *call, mlir::Value value)
    {
        auto root = rootOf(value);
        if (isConstantData(root))
        {
            return true;
        }

        if (!isFresh(root) || root.getDefiningOp()->getBlock() != call->getBlock())
        {
            return false;
        }

        mlir::Operation *release = nullptr;
        auto usedAfter = false;
        forEachUse(root, [&](mlir::Operation *user, mlir::Value) {
            if (user == call)
            {
                return;
            }

            if (user->getBlock() != call->getBlock())
            {
                usedAfter = true;
            }
            else if (call->isBeforeInBlock(user))
            {
                usedAfter = usedAfter || release || !mlir::isa<mlir_ts::ReleaseOp>(user);
                release = mlir::isa<mlir_ts::ReleaseOp>(user) ? user : release;
            }
        });

        if (usedAfter)
        {
            return false;
        }

        if (release)
        {
            release->erase();
        }

        return true;
    }

} // namespace own_signature
