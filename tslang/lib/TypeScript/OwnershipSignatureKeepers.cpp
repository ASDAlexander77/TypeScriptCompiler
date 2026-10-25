// OwnershipSignaturePass (see OwnershipSignaturePass.h): keepers - append-only array globals whose
// elements live to the end of the program - and the values they keep, made immortal where it is
// known what they rely on (docs/superpowers/specs/2026-10-25-own-immortal-keepers-design.md).

#include "OwnershipSignaturePass.h"

#include <optional>

#define DEBUG_TYPE "own"

namespace own_signature
{

    // The value a chain of casts starts from: `this.onTick` cast to the array's function type
    static mlir::Value castRoot(mlir::Value value)
    {
        while (auto castOp = value.getDefiningOp<mlir_ts::CastOp>())
        {
            value = castOp.getIn();
        }

        return value;
    }

    // Does a global's initializer region give it a new array - `[]`, an array literal - and nothing
    // else?
    static bool regionYieldsNewArray(mlir_ts::GlobalOp globalOp)
    {
        if (!globalOp.getInitializerRegion().hasOneBlock())
        {
            return false;
        }

        auto resultOp = mlir::dyn_cast<mlir_ts::GlobalResultOp>(globalOp.getInitializerBlock()->getTerminator());
        if (!resultOp || resultOp->getNumOperands() != 1)
        {
            return false;
        }

        auto *def = castRoot(resultOp->getOperand(0)).getDefiningOp();
        return mlir::isa_and_nonnull<mlir_ts::ConstantOp, mlir_ts::CreateArrayOp, mlir_ts::NewArrayOp>(def);
    }

    void OwnershipSignaturePass::findKeepers(mlir::ModuleOp module)
    {
        auto uses = mlir::SymbolTable::getSymbolUses(&module.getBodyRegion());
        if (!uses)
        {
            return; // an op this pass cannot read symbols through: nothing is known append-only
        }

        llvm::StringMap<llvm::SmallVector<mlir::Operation *>> usesOf;
        for (auto &use : *uses)
        {
            usesOf[use.getSymbolRef().getRootReference().getValue()].push_back(use.getUser());
        }

        auto mth = typeHelper(&getContext());
        module.walk([&](mlir_ts::GlobalOp globalOp) {
            auto arrayType = mlir::dyn_cast<mlir_ts::ArrayType>(globalOp.getType());
            if (!arrayType || globalOp->hasAttr("export") || globalOp->hasAttr("import") || globalOp->hasAttr(DLL_EXPORT) ||
                globalOp->hasAttr(DLL_IMPORT) || !regionYieldsNewArray(globalOp))
            {
                return;
            }

            // an element ts.MakeImmortal can mark, that owns a block (not a @cpp object)
            auto elementType = arrayType.getElementType();
            if (!mlir::isa<mlir_ts::ClassType, mlir_ts::ObjectType, mlir_ts::StringType, mlir_ts::ArrayType,
                           mlir_ts::BoundFunctionType, mlir_ts::HybridFunctionType>(elementType) ||
                !mth.ownsHeapMemory(globalOp.getLoc(), elementType))
            {
                return;
            }

            // pushed onto, its length read, its elements loaded: nothing else
            llvm::SmallVector<mlir_ts::AddressOfOp> addresses;
            llvm::SmallVector<mlir_ts::LoadOp> elementReads;
            for (auto *user : usesOf.lookup(globalOp.getSymName()))
            {
                auto addressOfOp = mlir::dyn_cast<mlir_ts::AddressOfOp>(user);
                if (!addressOfOp)
                {
                    return;
                }

                for (auto *addressUser : addressOfOp->getUsers())
                {
                    // in a function, which decideKeptForGood looks at: not a global's region
                    if (auto pushOp = mlir::dyn_cast<mlir_ts::ArrayPushOp>(addressUser);
                        pushOp && pushOp.getOp() == addressOfOp.getResult() &&
                        !llvm::is_contained(pushOp.getItems(), addressOfOp.getResult()) &&
                        pushOp->getParentOfType<mlir_ts::FuncOp>())
                    {
                        continue;
                    }

                    auto loadOp = mlir::dyn_cast<mlir_ts::LoadOp>(addressUser);
                    if (!loadOp)
                    {
                        return;
                    }

                    for (auto *arrayUser : loadOp->getUsers())
                    {
                        if (mlir::isa<mlir_ts::LengthOfOp>(arrayUser))
                        {
                            continue;
                        }

                        auto elementRefOp = mlir::dyn_cast<mlir_ts::ElementRefOp>(arrayUser);
                        if (!elementRefOp || elementRefOp->getOperand(0) != loadOp.getResult())
                        {
                            return;
                        }

                        for (auto *elementUser : elementRefOp->getUsers())
                        {
                            auto elementLoad = mlir::dyn_cast<mlir_ts::LoadOp>(elementUser);
                            if (!elementLoad)
                            {
                                return;
                            }

                            elementReads.push_back(elementLoad);
                        }
                    }
                }

                addresses.push_back(addressOfOp);
            }

            keepers.insert(globalOp.getSymName());
            for (auto addressOfOp : addresses)
            {
                addressOfOp->setAttr(OWN_KEEPER_GLOBAL_ATTR_NAME, mlir::UnitAttr::get(&getContext()));
            }

            for (auto elementLoad : elementReads)
            {
                elementLoad->setAttr(OWN_KEEPER_READ_ATTR_NAME, mlir::UnitAttr::get(&getContext()));
            }
        });
    }

    OwnershipSignaturePass::Kept OwnershipSignaturePass::classifyKept(mlir::Value value, mlir_ts::FuncOp funcOp)
    {
        auto root = castRoot(value);
        if (holdsNoBlock(root) || isConstantData(root) || isKeptForGood(root) || isImmortalGlobalRead(root))
        {
            return {KeptKind::Immortal};
        }

        int32_t index = -1;
        if (readsParameterSlot(root, funcOp, index))
        {
            return {KeptKind::Param, index, root};
        }

        auto *def = root.getDefiningOp();
        if (!def)
        {
            return {KeptKind::Unknown};
        }

        // read out of an owning local declared from it and never assigned: what it was declared from,
        // the block the read is
        if (auto loadOp = mlir::dyn_cast<mlir_ts::LoadOp>(def))
        {
            auto varOp = loadOp.getReference().getDefiningOp<mlir_ts::VariableOp>();
            if (varOp && isOwningVariable(varOp) && !varOp.getCaptured().value_or(false) && varOp.getInitializer() &&
                !isAssigned(varOp))
            {
                auto declared = classifyKept(varOp.getInitializer(), funcOp);
                if (declared.kind == KeptKind::Mark)
                {
                    declared.root = root;
                }

                if (declared.kind == KeptKind::Mark || declared.kind == KeptKind::Immortal)
                {
                    return declared;
                }
            }

            return {KeptKind::Unknown};
        }

        // a closure: its box, and the cells and values the box borrows from this frame
        if (auto boundOp = mlir::dyn_cast<mlir_ts::CreateBoundFunctionOp>(def))
        {
            auto boxOp = boundOp.getThisVal().getDefiningOp<mlir_ts::VariableOp>();
            if (boxOp && boxOp->hasAttr(CAPTURE_BOX_ATTR_NAME))
            {
                return {KeptKind::Mark, -1, root, boxOp.getResult()};
            }

            return {KeptKind::Unknown};
        }

        // a bound method: the object, which it holds without owning
        if (mlir::isa<mlir_ts::ThisSymbolRefOp, mlir_ts::ThisVirtualSymbolRefOp>(def) &&
            mlir::isa<mlir_ts::BoundFunctionType>(root.getType()))
        {
            auto object = def->getOperand(0);
            if (mlir::isa<mlir_ts::ClassType>(object.getType()) && own_facts::ownsHeap(object))
            {
                return {KeptKind::Mark, -1, root};
            }

            return {KeptKind::Unknown};
        }

        // made here: an object, an array, a string
        if (mlir::isa<mlir_ts::ClassType, mlir_ts::ObjectType, mlir_ts::ArrayType, mlir_ts::StringType>(root.getType()) &&
            own_facts::ownsHeap(root) && isFresh(root))
        {
            return {KeptKind::Mark, -1, root};
        }

        return {KeptKind::Unknown};
    }

    llvm::SmallVector<int32_t> OwnershipSignaturePass::calleesImmortalize(const Callees &callees, bool &disagree)
    {
        disagree = false;
        if (callees.viaCallback)
        {
            return {};
        }

        // the functions it may reach; a call this pass knows nothing else of - another module's
        // function, which says what it keeps for good in its declaration - by the symbols it names
        llvm::SmallVector<mlir_ts::FuncOp, 2> funcs(callees.funcs.begin(), callees.funcs.end());
        if (funcs.empty())
        {
            for (auto name : callees.named)
            {
                if (auto funcOp = functions.lookup(name.getValue()))
                {
                    funcs.push_back(funcOp);
                }
                else
                {
                    return {};
                }
            }
        }

        std::optional<llvm::SmallVector<int32_t>> common;
        for (auto callee : funcs)
        {
            auto found = immortalizes.find(callee);
            llvm::SmallVector<int32_t> those;
            if (found != immortalizes.end())
            {
                those = found->second;
            }

            if (!common)
            {
                common = those;
            }
            else if (*common != those)
            {
                disagree = true;
                return {};
            }
        }

        return common ? *common : llvm::SmallVector<int32_t>();
    }

    bool OwnershipSignaturePass::takesAddress(mlir_ts::FuncOp funcOp, mlir::ModuleOp module)
    {
        auto uses = mlir::SymbolTable::getSymbolUses(funcOp.getSymNameAttr(), &module.getBodyRegion());
        if (!uses)
        {
            return true;
        }

        for (auto &use : *uses)
        {
            auto *user = use.getUser();
            // a direct call, or a virtual one, whose family is resolved; the class vtable's entry
            if (mlir::isa<mlir_ts::SymbolCallInternalOp, mlir_ts::CallOp, mlir_ts::InvokeOp, mlir_ts::ThisVirtualSymbolRefOp,
                          mlir_ts::VirtualSymbolRefOp>(user) ||
                (mlir::isa<mlir_ts::SymbolRefOp>(user) && isClassVTableEntry(user)))
            {
                continue;
            }

            return true;
        }

        return false;
    }

    void OwnershipSignaturePass::makeImmortal(const Kept &kept, bool assumed)
    {
        auto root = kept.root;
        mlir::OpBuilder builder(&getContext());
        builder.setInsertionPointAfterValue(root);
        auto makeOp = builder.create<mlir_ts::MakeImmortalOp>(root.getLoc(), root.getType(), root,
                                                              assumed ? builder.getUnitAttr() : mlir::UnitAttr());
        root.replaceAllUsesExcept(makeOp.getResult(), makeOp);
        if (!kept.box)
        {
            return;
        }

        // what the box holds and borrows from this frame, which releases it: each captured cell, and
        // each value captured by value that owns a block
        for (auto *boxUser : kept.box.getUsers())
        {
            auto propertyRefOp = mlir::dyn_cast<mlir_ts::PropertyRefOp>(boxUser);
            if (!propertyRefOp)
            {
                continue;
            }

            for (auto *fieldUser : propertyRefOp->getUsers())
            {
                auto storeOp = mlir::dyn_cast<mlir_ts::StoreOp>(fieldUser);
                if (!storeOp || storeOp.getReference() != propertyRefOp.getResult())
                {
                    continue;
                }

                auto captured = storeOp.getValue();
                if (mlir::isa<mlir_ts::RefType>(captured.getType()))
                {
                    builder.create<mlir_ts::MarkImmortalCellOp>(captured.getLoc(), captured);
                }
                else if (own_facts::ownsHeap(captured) && !holdsNoBlock(captured))
                {
                    builder.create<mlir_ts::MakeImmortalOp>(captured.getLoc(), captured.getType(), captured, mlir::UnitAttr());
                }
            }
        }
    }

    void OwnershipSignaturePass::decideKeptForGood(mlir::ModuleOp module)
    {
        // a value kept for good: a keeper's push of it, or a call's argument its callee keeps so
        struct Site
        {
            mlir::Operation *op;
            unsigned operand;
        };

        llvm::MapVector<mlir::Operation *, llvm::SmallVector<Site>> pushes;
        module.walk([&](mlir_ts::ArrayPushOp pushOp) {
            auto addressOfOp = pushOp.getOp().getDefiningOp<mlir_ts::AddressOfOp>();
            auto funcOp = pushOp->getParentOfType<mlir_ts::FuncOp>();
            if (!addressOfOp || !addressOfOp->hasAttr(OWN_KEEPER_GLOBAL_ATTR_NAME) || !funcOp)
            {
                return;
            }

            for (auto index : llvm::seq<unsigned>(1, pushOp->getNumOperands()))
            {
                pushes[funcOp].push_back({pushOp, index});
            }
        });

        // what declarations say: a C function that keeps a callback for good
        immortalizes.clear();
        for (auto &entry : functions)
        {
            auto funcOp = entry.second;
            if (auto declared = funcOp->getAttrOfType<mlir::DenseI32ArrayAttr>(OWN_DECLARED_IMMORTALIZES_ATTR_NAME);
                declared && funcOp.isDeclaration())
            {
                immortalizes[funcOp].assign(declared.asArrayRef().begin(), declared.asArrayRef().end());
            }
        }

        auto addParam = [&](mlir_ts::FuncOp funcOp, int32_t param) {
            auto &params = immortalizes[funcOp];
            if (llvm::is_contained(params, param))
            {
                return false;
            }

            params.insert(llvm::upper_bound(params, param), param);
            return true;
        };

        // a parameter pushed, or given to a callee's parameter kept for good, is kept for good
        for (auto changed = true; changed;)
        {
            changed = false;
            for (auto &[op, sites] : pushes)
            {
                auto funcOp = mlir::cast<mlir_ts::FuncOp>(op);
                for (auto site : sites)
                {
                    auto kept = classifyKept(site.op->getOperand(site.operand), funcOp);
                    if (kept.kind == KeptKind::Param)
                    {
                        changed |= addParam(funcOp, kept.param);
                    }
                }
            }

            for (auto &call : calls)
            {
                auto funcOp = call.op->getParentOfType<mlir_ts::FuncOp>();
                bool disagree = false;
                auto args = callArgs(call.op);
                for (auto index : calleesImmortalize(call.callees, disagree))
                {
                    if (funcOp && static_cast<size_t>(index) < args.size())
                    {
                        auto kept = classifyKept(args[index], funcOp);
                        if (kept.kind == KeptKind::Param)
                        {
                            changed |= addParam(funcOp, kept.param);
                        }
                    }
                }
            }
        }

        auto failed = false;
        auto describe = [](mlir_ts::FuncOp funcOp, llvm::ArrayRef<int32_t> params) { return describeParams(funcOp, params); };

        // a function's callers must all know to make what they give it immortal
        for (auto &entry : functions)
        {
            auto funcOp = entry.second;
            if (funcOp.isDeclaration())
            {
                continue;
            }

            auto found = immortalizes.find(funcOp);
            llvm::SmallVector<int32_t> computed;
            if (found != immortalizes.end())
            {
                computed = found->second;
            }

            auto declared = funcOp->getAttrOfType<mlir::DenseI32ArrayAttr>(OWN_DECLARED_IMMORTALIZES_ATTR_NAME);
            if (declared && llvm::ArrayRef<int32_t>(computed) != declared.asArrayRef())
            {
                funcOp.emitError("'@" IMMORTALIZES_DECORATOR "' of '")
                    << funcOp.getSymName() << "' names " << describe(funcOp, declared.asArrayRef())
                    << ", but its body keeps for good " << describe(funcOp, computed);
                failed = true;
                continue;
            }

            if (computed.empty())
            {
                continue;
            }

            if (takesAddress(funcOp, module))
            {
                funcOp.emitError("'") << funcOp.getSymName() << "' keeps " << describe(funcOp, computed)
                                      << " for good, but -mm=own cannot make every caller make it immortal: the function is "
                                         "called through a value";
                failed = true;
            }
            else if (!declared && open.contains(funcOp))
            {
                funcOp.emitError("'") << funcOp.getSymName() << "' keeps " << describe(funcOp, computed)
                                      << " for good, but -mm=own cannot tell its callers in other modules; '@" IMMORTALIZES_DECORATOR
                                         "' would say it in its signature";
                failed = true;
            }
        }

        // a family whose members disagree: some calls of the one that keeps would not mark
        for (auto &call : calls)
        {
            bool disagree = false;
            calleesImmortalize(call.callees, disagree);
            if (disagree)
            {
                call.op->emitError("the functions this call may reach do not all keep the same arguments for good; -mm=own "
                                   "cannot make what it passes immortal");
                failed = true;
            }
        }

        if (failed)
        {
            signalPassFailure();
            return;
        }

        // every value kept for good: immortal from where it is made, or where it is a parameter
        auto decide = [&](mlir::Operation *site, mlir::Value value, mlir_ts::FuncOp funcOp, llvm::StringRef where) {
            auto kept = classifyKept(value, funcOp);
            switch (kept.kind)
            {
            case KeptKind::Immortal:
                return;
            case KeptKind::Param:
                makeImmortal(kept, /*assumed=*/true);
                return;
            case KeptKind::Mark:
                makeImmortal(kept, /*assumed=*/false);
                return;
            case KeptKind::Unknown:
                site->emitError("-mm=own cannot make the value ") << where << " immortal here: it is not made here, nor a parameter";
                failed = true;
                return;
            }
        };

        for (auto &[op, sites] : pushes)
        {
            for (auto site : sites)
            {
                auto pushOp = mlir::cast<mlir_ts::ArrayPushOp>(site.op);
                auto addressOfOp = pushOp.getOp().getDefiningOp<mlir_ts::AddressOfOp>();
                decide(site.op, site.op->getOperand(site.operand), mlir::cast<mlir_ts::FuncOp>(op),
                       ("kept in '" + addressOfOp.getGlobalName() + "'").str());
            }
        }

        for (auto &call : calls)
        {
            auto funcOp = call.op->getParentOfType<mlir_ts::FuncOp>();
            bool disagree = false;
            auto params = calleesImmortalize(call.callees, disagree);
            for (auto index : params)
            {
                auto args = callArgs(call.op);
                if (funcOp && static_cast<size_t>(index) < args.size())
                {
                    decide(call.op, args[index], funcOp, "given to a parameter its callee keeps for good");
                }
            }
        }

        if (failed)
        {
            signalPassFailure();
        }
    }

} // namespace own_signature

#undef DEBUG_TYPE
