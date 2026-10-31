// OwnershipInferencePass (see OwnershipInferencePass.h): names and diagnostics - what a value,
// a place or a call is called in an error, and the errors themselves.

#include "OwnershipInferencePass.h"

#define DEBUG_TYPE "own"

namespace own_inference
{

    std::string OwnershipInferencePass::describeCall(mlir::Operation *call)
    {
        auto callee = calleeName(call);
        return callee.empty() ? "a call's result" : "the result of '" + callee + "'";
    }

    std::string OwnershipInferencePass::calleeName(mlir::Operation *call)
    {
        mlir::StringAttr callee;
        auto value = calleeValue(call);
        if (auto symbol = directCallee(call))
        {
            callee = symbol.getAttr();
        }
        else if (auto calleeOp = value ? value.getDefiningOp() : nullptr)
        {
            if (auto getMethodOp = mlir::dyn_cast<mlir_ts::GetMethodOp>(calleeOp))
            {
                calleeOp = getMethodOp.getBoundFunc().getDefiningOp();
            }

            if (auto refOp = mlir::dyn_cast_or_null<mlir_ts::ThisVirtualSymbolRefOp>(calleeOp))
            {
                callee = refOp.getIdentifierAttr().getAttr();
            }
            else if (auto refOp = mlir::dyn_cast_or_null<mlir_ts::VirtualSymbolRefOp>(calleeOp))
            {
                callee = refOp.getIdentifierAttr().getAttr();
            }
            else if (auto refOp = mlir::dyn_cast_or_null<mlir_ts::ThisSymbolRefOp>(calleeOp))
            {
                callee = refOp.getIdentifierAttr().getAttr();
            }
            else if (auto refOp = mlir::dyn_cast_or_null<mlir_ts::InterfaceSymbolRefOp>(calleeOp))
            {
                // a method reached through an interface (spec 31): `<interface>.<method>`
                auto method = mlir::dyn_cast<mlir::StringAttr>(refOp.getIdentifier());
                auto interfaceType = mlir::dyn_cast<mlir_ts::InterfaceType>(refOp.getInterfaceVal().getType());
                if (method && interfaceType)
                {
                    return (interfaceType.getName().getValue() + "." + method.getValue()).str();
                }
            }
        }

        return callee ? callee.getValue().str() : std::string();
    }

    std::string OwnershipInferencePass::describePlace(mlir::Value ref)
    {
        auto path = pathOf(ref);
        if (path.empty())
        {
            return ref.getDefiningOp<mlir_ts::ElementRefOp>() ? "an element" : "a field";
        }

        return "'" + path + "'";
    }

    std::string OwnershipInferencePass::pathOf(mlir::Value ref)
    {
        auto place = placeBase(ref);
        std::string step;
        switch (place.kind)
        {
        case PlaceKind::Field: {
            auto field = fieldName(ref.getDefiningOp<mlir_ts::PropertyRefOp>());
            if (field.empty())
            {
                return {};
            }

            step = "." + field;
            break;
        }
        case PlaceKind::Element:
            step = "[]";
            break;
        case PlaceKind::SharedValue:
            step = ".value";
            break;
        case PlaceKind::InterfaceField: {
            auto field = mlir::dyn_cast<mlir::StringAttr>(ref.getDefiningOp<mlir_ts::InterfaceSymbolRefOp>().getIdentifier());
            if (!field)
            {
                return {};
            }

            step = "." + field.getValue().str();
            break;
        }
        case PlaceKind::None:
            return {};
        }

        auto base = rootOf(place.base);
        std::string prefix;
        if (isPlace(base))
        {
            prefix = pathOf(base);
        }
        else if (auto loadOp = base.getDefiningOp<mlir_ts::LoadOp>())
        {
            auto from = loadOp.getReference();
            if (isPlace(from))
            {
                prefix = pathOf(from);
            }
            else if (auto varOp = from.getDefiningOp<mlir_ts::VariableOp>())
            {
                prefix = varName(varOp).str();
            }
            else if (auto global = from.getDefiningOp<mlir_ts::AddressOfOp>())
            {
                prefix = global.getGlobalName().str();
            }
        }
        else if (auto varOp = base.getDefiningOp<mlir_ts::VariableOp>())
        {
            prefix = varName(varOp).str();
        }
        else
        {
            prefix = ownerName(base).str();
        }

        // a name MLIRGen made up (`.a`, the array a `for...of` walks) means nothing to the reader
        if (prefix.empty() || prefix == "this value" || prefix.front() == '.')
        {
            return {};
        }

        return prefix + step;
    }

    std::string OwnershipInferencePass::fieldName(mlir_ts::PropertyRefOp propertyRefOp)
    {
        auto type = propertyRefOp.getObjectRef().getType();
        if (auto refType = mlir::dyn_cast<mlir_ts::RefType>(type))
        {
            type = refType.getElementType();
        }

        if (auto classType = mlir::dyn_cast<mlir_ts::ClassType>(type))
        {
            type = classType.getStorageType();
        }

        auto position = propertyRefOp.getPosition();
        mlir::Attribute id;
        if (auto storageType = mlir::dyn_cast<mlir_ts::ClassStorageType>(type); storageType && position < storageType.size())
        {
            id = storageType.getId(position);
        }
        else if (auto tupleType = mlir::dyn_cast<mlir_ts::TupleType>(asStorageTuple(type)); tupleType && position < tupleType.size())
        {
            id = tupleType.getId(position);
        }

        if (auto name = mlir::dyn_cast_or_null<mlir::StringAttr>(id))
        {
            return name.str();
        }

        if (auto name = mlir::dyn_cast_or_null<mlir::FlatSymbolRefAttr>(id))
        {
            return name.getValue().str();
        }

        return {};
    }

    llvm::StringRef OwnershipInferencePass::ownerName(mlir::Value value)
    {
        llvm::StringRef name = "this value";
        forEachUse(value, [&](mlir::Operation *user, mlir::Value) {
            if (mlir::isa<mlir_ts::DebugVariableOp>(user))
            {
                name = nameAt(user->getLoc());
            }
        });

        return name;
    }

    llvm::StringRef OwnershipInferencePass::varName(mlir_ts::VariableOp varOp)
    {
        return nameAt(varOp.getLoc());
    }

    llvm::StringRef OwnershipInferencePass::nameAt(mlir::Location location)
    {
        if (auto fused = mlir::dyn_cast<mlir::FusedLocWith<mlir::LLVM::DILocalVariableAttr>>(location))
        {
            if (auto name = fused.getMetadata().getName())
            {
                return name.getValue();
            }
        }

        if (auto scoped = mlir::dyn_cast<mlir::FusedLocWith<mlir::LLVM::DIScopeAttr>>(location))
        {
            if (auto named = mlir::dyn_cast_or_null<mlir::NameLoc>(scoped.getLocations().front()))
            {
                return named.getName().getValue();
            }
        }

        return "this value";
    }

    llvm::StringRef OwnershipInferencePass::describe(mlir::Value value)
    {
        llvm::StringRef name = "this value";
        forEachUse(value, [&](mlir::Operation *user, mlir::Value) {
            if (auto varOp = mlir::dyn_cast<mlir_ts::VariableOp>(user); varOp && name == "this value")
            {
                name = varName(varOp);
            }
        });

        return name;
    }

    void OwnershipInferencePass::reportSecondReference(mlir::Operation *op, mlir::Value value)
    {
        if (!reporting())
        {
            return;
        }

        llvm::StringRef name = "this value";
        if (auto retainSlotOp = mlir::dyn_cast<mlir_ts::RetainSlotOp>(op))
        {
            if (auto varOp = retainSlotOp.getSlot().getDefiningOp<mlir_ts::VariableOp>())
            {
                name = varName(varOp);
            }
        }
        else if (auto varOp = mlir::dyn_cast<mlir_ts::VariableOp>(op))
        {
            name = varName(varOp);
        }

        if (!value && mlir::isa<mlir_ts::RetainOp, mlir_ts::RetainSlotOp>(op))
        {
            value = retainedValue(op);
        }

        if (!value || !reportShapeOfSecondReference(op, rootOf(value), name))
        {
            auto diag = op->emitError("'") << name << "' takes a second reference; -mm=own cannot prove a move or a borrow here yet";
            noteLostFacts(diag);
        }

        signalPassFailure();
    }

    bool OwnershipInferencePass::reportShapeOfSecondReference(mlir::Operation *op, mlir::Value root, llvm::StringRef name)
    {
        auto f = getFunction();

        // returned, where another return gives a borrow of an argument and this one does not, or
        // the other way round: the callers cannot be told which they get
        if (isReturned(root))
        {
            auto param = borrowedParam(root);
            if (auto *other = otherReturn(param))
            {
                auto diag = op->emitError("'") << f.getName()
                                               << "' returns a borrow of its argument on some paths and another value on "
                                                  "others; -mm=own cannot tell its callers whether they own the result";
                diag.attachNote(other->getLoc())
                    << (param >= 0 ? "returns another value here" : "returns a borrow of its argument here");
                return true;
            }
        }

        // a parameter's value, or a part of it, returned or kept: a fact its callers must know
        // (spec 15), and noteLostFacts says why they cannot
        if (auto param = borrowedParam(root); param >= 0)
        {
            // the retain of a parameter's copy for a closure's box (MLIRGen's mlirGenRetainCaptured
            // reads the slot again for the box's field), the box of a closure that escapes - a
            // parameter nothing assigns is captured by copy, and its cell used to be refused the same
            // way (decideEscaping)
            auto isBoxFill = [&](mlir::Operation *user) {
                auto storeOp = mlir::dyn_cast<mlir_ts::StoreOp>(user);
                auto propertyRefOp = storeOp ? storeOp.getReference().getDefiningOp<mlir_ts::PropertyRefOp>()
                                             : mlir_ts::PropertyRefOp();
                return propertyRefOp && closureOfBox(propertyRefOp.getObjectRef()) && user->getLoc() == op->getLoc();
            };
            auto retainOp = mlir::dyn_cast<mlir_ts::RetainOp>(op);
            auto retainedRead = retainOp ? retainOp->getOperand(0).getDefiningOp<mlir_ts::LoadOp>() : mlir_ts::LoadOp();
            auto boxFill = retainedRead && llvm::any_of(retainedRead.getReference().getUsers(), [&](mlir::Operation *user) {
                auto loadOp = mlir::dyn_cast<mlir_ts::LoadOp>(user);
                return loadOp && llvm::any_of(loadOp->getUsers(), isBoxFill);
            });
            if (boxFill)
            {
                op->emitError("a closure that escapes owns what it captures, but -mm=own cannot move this variable "
                              "into it: its value is the caller's");
                return true;
            }

            auto diag = op->emitError("'") << (name == "this value" ? paramName(param) : name);
            if (isReturned(root))
            {
                diag << "' is a parameter, returned here, but -mm=own cannot tell the callers of '" << f.getName()
                     << "' that the result borrows it";
            }
            else
            {
                diag << "' is a parameter, kept here, but -mm=own cannot make the callers of '" << f.getName()
                     << "' give it up";
            }

            noteLostFacts(diag);
            return true;
        }

        // a store into a local whose value is not this function's to replace: a parameter's slot, or
        // a local that owns nothing (MLIRGen's takeOwnershipOfLocal)
        auto storeOp = mlir::dyn_cast<mlir_ts::StoreOp>(op);
        if (auto slot = storeOp ? storeOp.getReference().getDefiningOp<mlir_ts::VariableOp>() : mlir_ts::VariableOp();
            slot && !isOwningVariable(slot) && !isResultSlot(slot))
        {
            // a parameter's slot, assigned (isParameterSlot is the unassigned one)
            auto argument = mlir::dyn_cast_or_null<mlir::BlockArgument>(slot.getInitializer());
            if (argument && argument.getOwner()->isEntryBlock() &&
                mlir::isa<mlir_ts::FuncOp>(argument.getOwner()->getParentOp()))
            {
                op->emitError("'") << varName(slot) << "' is a parameter and cannot be assigned under -mm=own yet";
                return true;
            }

            // MLIRGen makes no owner of a local declared without a value, nor of one declared in a
            // catch or finally clause, whose release would be a call inside an exception funclet
            if (!slot.getInitializer())
            {
                op->emitError("'") << varName(slot)
                                   << "' owns nothing under -mm=own (it is declared without a value, or in a catch or "
                                      "finally clause), so it cannot be given a new value here";
                return true;
            }
        }

        // merged from branches or loop iterations
        if (auto argument = mlir::dyn_cast<mlir::BlockArgument>(root); argument && !argument.getOwner()->isEntryBlock())
        {
            op->emitError("'") << name
                               << "' is merged from several branches (`?:`, `&&`, `||`, `??`, a loop); -mm=own cannot "
                                  "prove which one owns it yet";
            return true;
        }

        auto loadOp = root.getDefiningOp<mlir_ts::LoadOp>();

        // a read of a global
        if (auto addressOfOp = loadOp ? loadOp.getReference().getDefiningOp<mlir_ts::AddressOfOp>() : mlir_ts::AddressOfOp())
        {
            // a local that reads it: phase 3's borrow, which a call or an assignment of the global
            // may end
            if (mlir::isa<mlir_ts::RetainSlotOp>(op))
            {
                op->emitError("'") << name << "' borrows the global '" << addressOfOp.getGlobalName()
                                   << "', but -mm=own cannot prove the global keeps its value while '" << name
                                   << "' is used: a call or an assignment may overwrite it";
                return true;
            }

            op->emitError("'") << name << "' borrows the global '" << addressOfOp.getGlobalName()
                               << "' and cannot be stored, returned or captured";
            return true;
        }

        // a read of a captured variable, through its cell
        if (loadOp && isLoadedCell(loadOp.getReference()))
        {
            op->emitError("'") << name << "' borrows a captured variable and cannot be stored, returned or captured";
            return true;
        }

        // an object literal built in a local, holding a value it owns: moved only into an owning
        // local or the result (recordRetainOf)
        auto record = loadOp ? loadOp.getReference().getDefiningOp<mlir_ts::VariableOp>() : mlir_ts::VariableOp();
        if (!record && storeOp)
        {
            auto propertyRefOp = storeOp.getReference().getDefiningOp<mlir_ts::PropertyRefOp>();
            record = propertyRefOp ? propertyRefOp->getOperand(0).getDefiningOp<mlir_ts::VariableOp>() : mlir_ts::VariableOp();
        }

        if (record && !isOwningVariable(record) && record.getInitializer() &&
            record.getInitializer().getDefiningOp<mlir_ts::ConstantOp>())
        {
            op->emitError("an object literal that holds a value it owns is moved only into a local or "
                          "returned as it is; -mm=own cannot move it here yet");
            return true;
        }

        return false;
    }

    llvm::StringRef OwnershipInferencePass::paramName(int index)
    {
        llvm::StringRef name = "this value";
        getFunction()->walk([&](mlir_ts::VariableOp varOp) {
            auto argument = mlir::dyn_cast_or_null<mlir::BlockArgument>(varOp.getInitializer());
            if (argument && argument.getOwner()->isEntryBlock() && static_cast<int>(argument.getArgNumber()) == index)
            {
                name = varName(varOp);
            }
        });

        return name;
    }

    bool OwnershipInferencePass::isReturned(mlir::Value root)
    {
        auto returned = false;
        forEachUse(root, [&](mlir::Operation *user, mlir::Value used) {
            auto storeOp = mlir::dyn_cast<mlir_ts::StoreOp>(user);
            auto slot = storeOp && storeOp.getValue() == used ? storeOp.getReference().getDefiningOp<mlir_ts::VariableOp>()
                                                              : mlir_ts::VariableOp();
            returned = returned || (slot && isResultSlot(slot));
        });

        return returned;
    }

    mlir::Operation *OwnershipInferencePass::otherReturn(int param)
    {
        mlir::Operation *found = nullptr;
        getFunction()->walk([&](mlir_ts::StoreOp storeOp) {
            auto slot = storeOp.getReference().getDefiningOp<mlir_ts::VariableOp>();
            if (found || !slot || !isResultSlot(slot) || holdsNoBlock(storeOp.getValue()))
            {
                return;
            }

            auto borrows = borrowedParam(storeOp.getValue());
            if (param >= 0 ? borrows != param : borrows >= 0)
            {
                found = storeOp;
            }
        });

        return found;
    }

    void OwnershipInferencePass::noteLostFacts(mlir::InFlightDiagnostic &diag)
    {
        auto f = getFunction();
        if (auto why = f->getAttrOfType<mlir::StringAttr>(OWN_FACTS_LOST_ATTR_NAME))
        {
            diag.attachNote(f.getLoc()) << "'" << f.getName() << "' could return a borrow or keep a parameter, but "
                                        << why.getValue();
        }
    }

    void OwnershipInferencePass::reportKeptOnSomePaths(mlir::Operation *move, mlir::Operation *exit, llvm::StringRef name)
    {
        if (!reporting())
        {
            return;
        }

        auto diag = move->emitError("'") << name
                                         << "' is moved here on some paths only; -mm=own cannot release it "
                                            "on the others yet";
        diag.attachNote(exit->getLoc()) << "returns here without moving it; the caller gave it up";
        signalPassFailure();
    }

    void OwnershipInferencePass::reportHandleInOwningUnion(mlir::Operation *op)
    {
        auto handleIn = llvm::any_of(op->getOperandTypes(), [](mlir::Type type) {
            return MLIRTypeHelper::isSharedHandleType(type);
        });
        if (!handleIn)
        {
            return;
        }

        for (auto resultType : op->getResultTypes())
        {
            if (!MLIRTypeHelper::isSharedInOwningUnionType(resultType))
            {
                continue;
            }

            if (!reporting())
            {
                return;
            }

            auto unionType = resultType;
            if (auto optionalType = mlir::dyn_cast<mlir_ts::OptionalType>(unionType))
            {
                unionType = optionalType.getElementType();
            }

            mlir::Type shared;
            mlir::Type owning;
            for (auto member : mlir::cast<mlir_ts::UnionType>(unionType).getTypes())
            {
                if (mlir::isa<mlir_ts::SharedType>(member))
                {
                    shared = shared ? shared : member;
                }
                else if (!owning && MLIRTypeHelper::ownsHeapMemoryOfUnionMember(member))
                {
                    owning = member;
                }
            }

            auto diag = op->emitError("a ") << printed(shared) << " in a union with ";
            if (owning)
            {
                diag << "'" << printed(owning) << "', which -mm=own does not count,";
            }
            else
            {
                diag << "a member -mm=own does not count,";
            }

            diag << " is not supported";
            signalPassFailure();
            return;
        }
    }

    std::string OwnershipInferencePass::printed(mlir::Type type)
    {
        llvm::SmallString<128> text;
        llvm::raw_svector_ostream out(text);
        MLIRPrinter printer{};
        printer.printType<llvm::raw_svector_ostream>(out, type);
        return text.str().str();
    }

    void OwnershipInferencePass::reportGivenNotOwned(mlir::Operation *call, mlir::Value arg)
    {
        if (!reporting())
        {
            return;
        }

        llvm::StringRef name = ownerName(arg);
        if (auto loadOp = rootOf(arg).getDefiningOp<mlir_ts::LoadOp>())
        {
            if (auto varOp = loadOp.getReference().getDefiningOp<mlir_ts::VariableOp>())
            {
                name = varName(varOp);
            }
        }

        auto callee = calleeName(call);
        auto diag = call->emitError("argument '") << name << "' is given to '" << (callee.empty() ? "a function" : callee)
                                                  << "', which keeps it, but -mm=own cannot move it here: this "
                                                     "function does not own it";
        noteLostFacts(diag);
        signalPassFailure();
    }

    void OwnershipInferencePass::reportUseAfterMove(mlir::Operation *use, mlir::Operation *move, llvm::StringRef name)
    {
        if (!reporting())
        {
            return;
        }

        auto diag = use->emitError("'") << name << "' is used here after its value was moved";
        diag.attachNote(move->getLoc()) << "value moved here";
        signalPassFailure();
    }

    void OwnershipInferencePass::reportMovedInLoop(mlir::Operation *move, llvm::StringRef name)
    {
        if (!reporting())
        {
            return;
        }

        move->emitError("'") << name
                             << "' is moved inside a loop but was made outside it; only a borrow could do "
                                "that, and -mm=own cannot prove one yet";
        signalPassFailure();
    }

    void OwnershipInferencePass::reportBorrowOutlives(mlir::Operation *use, mlir::Operation *end, llvm::StringRef name, llvm::StringRef owner)
    {
        if (!reporting())
        {
            return;
        }

        auto diag = use->emitError("'") << name << "' borrows '" << owner << "' but is used here after '" << owner
                                        << "' is released or overwritten";
        diag.attachNote(end->getLoc()) << "'" << owner << "' is released or overwritten here";
        signalPassFailure();
    }

    void OwnershipInferencePass::reportPlaceBorrowOutlives(mlir::Operation *use, mlir::Operation *drop, llvm::StringRef name,
                                   llvm::StringRef place)
    {
        if (!reporting())
        {
            return;
        }

        auto diag = use->emitError("'") << name << "' borrows " << place
                                        << " but is used here after it may be released or overwritten";
        diag.attachNote(drop->getLoc()) << "it may be released or overwritten here";
        signalPassFailure();
    }

    void OwnershipInferencePass::reportPlaceBorrowerAssigned(mlir::Operation *op, llvm::StringRef name, llvm::StringRef place)
    {
        if (!reporting())
        {
            return;
        }

        op->emitError("'") << name << "' borrows " << place << " and cannot be assigned; -mm=own cannot prove that yet";
        signalPassFailure();
    }

    void OwnershipInferencePass::reportPlaceBorrowEscapes(mlir::Operation *op, llvm::StringRef name, llvm::StringRef place)
    {
        if (!reporting())
        {
            return;
        }

        auto diag = op->emitError("'") << name << "' borrows " << place << " and cannot be stored, returned or captured";
        noteLostFacts(diag);
        signalPassFailure();
    }

    void OwnershipInferencePass::reportMovedWhileBorrowed(mlir::Operation *op, llvm::StringRef owner, llvm::StringRef borrower)
    {
        if (!reporting())
        {
            return;
        }

        op->emitError("'") << owner << "' is moved here while '" << borrower
                           << "' borrows it; a borrowed value cannot be moved";
        signalPassFailure();
    }

    void OwnershipInferencePass::reportBorrowEscapes(mlir::Operation *op, llvm::StringRef name, llvm::StringRef owner)
    {
        if (!reporting())
        {
            return;
        }

        auto diag = op->emitError("'") << name << "' borrows '" << owner << "' and cannot be stored, returned or captured";
        noteLostFacts(diag);
        signalPassFailure();
    }

    void OwnershipInferencePass::reportBorrowerAssigned(mlir::Operation *op, llvm::StringRef name, llvm::StringRef owner)
    {
        if (!reporting())
        {
            return;
        }

        op->emitError("'") << name << "' borrows '" << owner << "' and cannot be assigned; -mm=own cannot prove that yet";
        signalPassFailure();
    }

    void OwnershipInferencePass::reportEscapingCellNotOwned(mlir::Operation *capture, mlir::Operation *escape, llvm::StringRef why)
    {
        if (!reporting())
        {
            return;
        }

        auto diag = capture->emitError("a closure that escapes owns what it captures, but -mm=own cannot move this "
                                       "variable into it: ")
                    << why;
        diag.attachNote(escape->getLoc()) << "the closure escapes here";
        signalPassFailure();
    }

    void OwnershipInferencePass::reportUsedAfterEscapingCapture(mlir::Operation *use, mlir::Operation *capture, llvm::StringRef name)
    {
        if (!reporting())
        {
            return;
        }

        auto diag = use->emitError("'") << name << "' is captured by a closure that escapes, and is still used here";
        diag.attachNote(capture->getLoc()) << "the closure takes it here";
        signalPassFailure();
    }

    void OwnershipInferencePass::reportClosureOutlives(mlir::Operation *use, mlir::Operation *end, llvm::StringRef closure,
                               llvm::StringRef captured)
    {
        if (!reporting())
        {
            return;
        }

        auto diag = use->emitError("'") << closure << "' borrows '" << captured << "', which it captures, but runs here after '"
                                        << captured << "' is released";
        diag.attachNote(end->getLoc()) << "'" << captured << "' is released here";
        signalPassFailure();
    }

    void OwnershipInferencePass::reportCaptureNotOwned(mlir::Operation *op, llvm::StringRef captured)
    {
        if (!reporting())
        {
            return;
        }

        op->emitError("'") << captured << "' is captured by a closure, but -mm=own cannot tell who owns its value here";
        signalPassFailure();
    }

    void OwnershipInferencePass::reportCapturedParamAssigned(mlir::Operation *op, llvm::StringRef name)
    {
        if (!reporting())
        {
            return;
        }

        op->emitError("'") << name
                           << "' is a captured parameter and cannot be assigned under -mm=own: its value is the "
                              "caller's";
        signalPassFailure();
    }

    void OwnershipInferencePass::reportMovedOnSomePaths(mlir::Operation *move, mlir::Operation *release, llvm::StringRef name,
                                mlir::Operation *usedAt)
    {
        if (!reporting())
        {
            return;
        }

        auto diag = move->emitError("'") << name
                                         << "' is moved here on some paths only; -mm=own cannot release it "
                                            "on the others yet";
        diag.attachNote(release->getLoc()) << "released here on every path";
        if (usedAt)
        {
            diag.attachNote(usedAt->getLoc())
                << "used here, after the paths join, so it cannot be released before they do";
        }

        signalPassFailure();
    }

} // namespace own_inference
