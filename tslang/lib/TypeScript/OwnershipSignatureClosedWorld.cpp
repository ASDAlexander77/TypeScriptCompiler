// OwnershipSignaturePass (see OwnershipSignaturePass.h): the closed world - which functions another
// module can call, which escape as values, and the families that are open.

#include "OwnershipSignaturePass.h"

#define DEBUG_TYPE "own"

namespace own_signature
{

    bool OwnershipSignaturePass::callableFromOutside(mlir_ts::FuncOp funcOp)
    {
        if (!funcOp.isPrivate() && (funcOp.isDeclaration() || funcOp->hasAttr("export") || funcOp->hasAttr(DLL_EXPORT)))
        {
            return true;
        }

        // its class: the longest own class name that prefixes its symbol
        auto symbol = funcOp.getSymName();
        llvm::StringRef owner;
        for (auto &entry : ownClasses)
        {
            auto name = entry.getKey();
            if (name.size() > owner.size() && symbol.size() > name.size() && symbol.starts_with(name) && symbol[name.size()] == '.')
            {
                owner = name;
            }
        }

        if (owner.empty())
        {
            return !funcOp.isPrivate();
        }

        auto nameable = [&](llvm::StringRef name) {
            auto last = name.rsplit('.').second;
            return nameableClasses.contains(name) || exportWords.contains(name) || (!last.empty() && exportWords.contains(last));
        };

        if (!funcOp.isPrivate() && nameable(owner))
        {
            return true;
        }

        // An override of a method of a nameable class, or of one another module defines (no own
        // vtable here), fills a slot of that class's vtable: another module calls it through the
        // base (review6r1 p5, p6, p7; controller ruling: any base of that kind on the chain). The
        // base is the first field of a derived class's storage.
        auto storage = classStorages.lookup(owner);
        if (!storage)
        {
            // its storage is not known here: no telling what it extends
            return !funcOp.isPrivate();
        }

        for (auto depth = 0; depth < 64; ++depth)
        {
            auto fields = storage.getFields();
            auto base = fields.empty() ? mlir_ts::ClassStorageType() : mlir::dyn_cast<mlir_ts::ClassStorageType>(fields.front().type);
            if (!base)
            {
                return false;
            }

            auto name = base.getName().getValue();
            if (nameable(name) || !ownClasses.contains(name))
            {
                return true;
            }

            storage = base;
        }

        return true;
    }

    bool OwnershipSignaturePass::holdsFunction(mlir::Type type, bool fields, SeenTypes &seen)
    {
        if (!type || !seen.insert({type, fields}).second)
        {
            return false;
        }

        if (mlir::isa<mlir_ts::FunctionType, mlir_ts::BoundFunctionType, mlir_ts::HybridFunctionType,
                      mlir_ts::ExtensionFunctionType, mlir_ts::ConstructFunctionType>(type))
        {
            return true;
        }

        if (auto interfaceType = mlir::dyn_cast<mlir_ts::InterfaceType>(type))
        {
            auto found = interfaceVTableTypes.find(interfaceType.getName().getValue());
            if (found == interfaceVTableTypes.end())
            {
                return true;
            }

            return llvm::any_of(found->second, [&](mlir_ts::TupleType vtableType) {
                return llvm::any_of(llvm::drop_begin(vtableType.getFields(), 1),
                                    [&](const mlir_ts::FieldInfo &info) { return holdsFunction(info.type, /*fields=*/true, seen); });
            });
        }

        auto any = [&](llvm::ArrayRef<mlir_ts::FieldInfo> infos) {
            return llvm::any_of(infos, [&](const mlir_ts::FieldInfo &info) { return holdsFunction(info.type, fields, seen); });
        };

        if (auto tupleType = mlir::dyn_cast<mlir_ts::TupleType>(type))
        {
            return any(tupleType.getFields());
        }

        if (auto constTupleType = mlir::dyn_cast<mlir_ts::ConstTupleType>(type))
        {
            return any(constTupleType.getFields());
        }

        if (auto classStorageType = mlir::dyn_cast<mlir_ts::ClassStorageType>(type))
        {
            return fields && any(classStorageType.getFields());
        }

        if (auto objectStorageType = mlir::dyn_cast<mlir_ts::ObjectStorageType>(type))
        {
            return fields && any(objectStorageType.getFields());
        }

        if (mlir::isa<mlir_ts::ClassType, mlir_ts::ObjectType>(type) && !fields)
        {
            return false;
        }

        auto found = false;
        type.walkImmediateSubElements([](mlir::Attribute) {}, [&](mlir::Type sub) { found = found || holdsFunction(sub, fields, seen); });
        return found;
    }

    mlir_ts::ClassStorageType OwnershipSignaturePass::cppClassOfMethod(mlir_ts::FuncOp funcOp)
    {
        auto inputs = funcOp.getFunctionType().getInputs();
        auto classType = inputs.empty() ? mlir_ts::ClassType() : mlir::dyn_cast<mlir_ts::ClassType>(inputs.front());
        auto storage = classType ? mlir::dyn_cast<mlir_ts::ClassStorageType>(classType.getStorageType()) : mlir_ts::ClassStorageType();
        if (!storage || !storage.isCpp())
        {
            return {};
        }

        auto name = storage.getName().getValue();
        auto symbol = funcOp.getSymName();
        return symbol.size() > name.size() && symbol.starts_with(name) && symbol[name.size()] == '.' ? storage : mlir_ts::ClassStorageType();
    }

    bool OwnershipSignaturePass::addressType(mlir::Type type)
    {
        return mlir::isa<mlir_ts::OpaqueType, mlir_ts::RefType, mlir_ts::ValueRefType, mlir_ts::BoundRefType, mlir_ts::ClassType,
                         mlir_ts::ClassStorageType, mlir_ts::ObjectType, mlir_ts::ObjectStorageType, mlir_ts::InterfaceType,
                         mlir_ts::ArrayType, mlir_ts::ConstArrayType, mlir_ts::StringType, mlir_ts::TupleType,
                         mlir_ts::ConstTupleType, mlir_ts::FunctionType, mlir_ts::BoundFunctionType, mlir_ts::HybridFunctionType,
                         mlir_ts::ExtensionFunctionType, mlir_ts::ConstructFunctionType, mlir_ts::NullType,
                         mlir_ts::UndefinedType>(type);
    }

    void OwnershipSignaturePass::findEscapingFunctions(mlir::ModuleOp module)
    {
        // every body-less function: C's, another module's, a `declare class`'s (its methods, its
        // `..instanceOf`, its `..new`), a `@cpp` class's slot
        for (auto &entry : functions)
        {
            auto funcOp = entry.second;
            if (funcOp.isDeclaration() && !scalarSignature(funcOp.getFunctionType()))
            {
                functionsEscape = true;
                return;
            }

            // a method of a `@cpp` class: C++ calls it through the class's C++ vtable
            if (cppClassOfMethod(funcOp))
            {
                functionsEscape = true;
                return;
            }
        }

        // every global declared with no initializer or imported: a `declare var`, a `declare
        // class`'s vtable (functions) and type name (a string)
        for (auto &entry : globals)
        {
            auto globalOp = entry.second;
            auto foreign = globalOp->hasAttr("import") || globalOp->hasAttr(DLL_IMPORT) || globalOp.getInitializerRegion().empty() ||
                           globalOp.getInitializerRegion().front().empty();
            if (foreign && !scalarType(globalOp.getType()))
            {
                functionsEscape = true;
                return;
            }
        }

        // every `@cpp` class's fields; a base (the first field, a storage) is a `@cpp` class of its
        // own, judged on its own
        for (auto &entry : classStorages)
        {
            auto storage = entry.second;
            if (!storage.isCpp())
            {
                continue;
            }

            for (auto [index, field] : llvm::enumerate(storage.getFields()))
            {
                auto base = index == 0 ? mlir::dyn_cast<mlir_ts::ClassStorageType>(field.type) : mlir_ts::ClassStorageType();
                if (base ? !base.isCpp() : !scalarType(field.type))
                {
                    functionsEscape = true;
                    return;
                }
            }
        }

        module.walk([&](mlir::Operation *op) {
            if (functionsEscape)
            {
                return;
            }

            // a function found in a library at run time: scalar through every cast of its address
            if (mlir::isa<mlir_ts::SearchForAddressOfSymbolOp>(op))
            {
                functionsEscape = !llvm::all_of(op->getResult(0).getUsers(), [](mlir::Operation *user) {
                    auto castOp = mlir::dyn_cast<mlir_ts::CastOp>(user);
                    auto functionType = castOp ? mlir::dyn_cast<mlir_ts::FunctionType>(castOp.getType()) : mlir_ts::FunctionType();
                    return functionType && scalarSignature(functionType);
                });
                return;
            }

            // `inline_asm<T>(...)` and `call_intrinsic<T>(...)`: a function given to either is data
            // of no type (review6r5 h1, h2: `inline_asm<u64>("mov $1, $0", "=r,r", onSig)`)
            if (mlir::isa<mlir_ts::InlineAsmOp, mlir_ts::CallIntrinsicOp>(op))
            {
                functionsEscape = llvm::any_of(op->getOperands(), [&](mlir::Value operand) {
                    SeenTypes seen;
                    return holdsFunction(operand.getType(), /*fields=*/true, seen);
                });
                return;
            }

            auto castOp = mlir::dyn_cast<mlir_ts::CastOp>(op);
            if (!castOp)
            {
                return;
            }

            // a vtable's address (a class's, an interface's, a thunk's cell) seen as an opaque
            // pointer or an integer: the compiler's own, inside a vtable or storing an object's
            // vtable, never a value a program gives C
            auto vtableOp = op->getParentOfType<mlir_ts::GlobalOp>();
            auto addressOfOp = addressOfThroughCasts(castOp.getIn());
            auto isVTable = [](llvm::StringRef name) { return name.ends_with("..vtbl") || name.contains("..vtbl."); };
            if ((vtableOp && isVTable(vtableOp.getSymName())) || (addressOfOp && isVTable(addressOfOp.getGlobalName())))
            {
                return;
            }

            // an opaque pointer or a reference made out of a number: an address of foreign memory
            if (mlir::isa<mlir_ts::OpaqueType, mlir_ts::RefType>(castOp.getType()) && !addressType(castOp.getIn().getType()))
            {
                functionsEscape = true;
                return;
            }

            // a function seen as something that holds none; a boolean cannot carry one. A class or
            // object takes its fields along, seen as anything (`any`, an opaque pointer, an integer,
            // a reference: `<Reference<u64>>k`, then a load of the field, review6r5 g2) but a class
            // (its base) or an interface (a view of it, read through its vtable only)
            if (mlir::isa<mlir_ts::BooleanType>(castOp.getType()))
            {
                return;
            }

            auto fields = !mlir::isa<mlir_ts::ClassType, mlir_ts::InterfaceType>(castOp.getType());
            SeenTypes seenIn;
            SeenTypes seenOut;
            functionsEscape = holdsFunction(castOp.getIn().getType(), fields, seenIn) &&
                              !holdsFunction(castOp.getType(), /*fields=*/true, seenOut);
        });
    }

    void OwnershipSignaturePass::collectClassStorages(mlir::ModuleOp module)
    {
        llvm::DenseSet<mlir::Type> seen;
        std::function<void(mlir::Type)> visit = [&](mlir::Type type) {
            if (!type || !seen.insert(type).second)
            {
                return;
            }

            if (auto storage = mlir::dyn_cast<mlir_ts::ClassStorageType>(type))
            {
                classStorages[storage.getName().getValue()] = storage;
                for (auto &field : storage.getFields())
                {
                    visit(field.type);
                }

                return;
            }

            type.walkImmediateSubElements([](mlir::Attribute) {}, visit);
        };

        for (auto &entry : functions)
        {
            visit(entry.second.getFunctionType());
        }

        module.walk([&](mlir::Operation *op) {
            for (auto result : op->getResults())
            {
                if (mlir::isa<mlir_ts::ClassType>(result.getType()))
                {
                    visit(result.getType());
                }
            }
        });
    }

    bool OwnershipSignaturePass::onlyCalled(mlir::Value value)
    {
        for (auto &use : value.getUses())
        {
            auto *user = use.getOwner();
            if (mlir::isa<mlir_ts::CallInternalOp, mlir_ts::CallIndirectOp, mlir_ts::InvokeOp>(user) &&
                calleeValue(user) == use.get() && use.getOperandNumber() == 0)
            {
                continue;
            }

            if (mlir::isa<mlir_ts::GetThisOp>(user))
            {
                continue;
            }

            if (mlir::isa<mlir_ts::GetMethodOp>(user) && onlyCalled(user->getResult(0)))
            {
                continue;
            }

            return false;
        }

        return true;
    }

    void OwnershipSignaturePass::findOpen(mlir::ModuleOp module)
    {
        for (auto &entry : functions)
        {
            if (callableFromOutside(entry.second))
            {
                open.try_emplace(entry.second, "it can be called from another module; '@" LENDS_DECORATOR "' or '@" KEEPS_DECORATOR
                                               "' would say it in its signature");
            }
        }

        // the module is a symbol table, which getSymbolUses does not look inside: walk its body
        auto uses = mlir::SymbolTable::getSymbolUses(&module.getBodyRegion());
        if (!uses)
        {
            // an op this pass cannot read symbols through: nothing is closed
            for (auto &entry : functions)
            {
                open.try_emplace(entry.second, "its callers cannot all be found");
            }

            return;
        }

        for (auto &use : *uses)
        {
            auto symbol = use.getSymbolRef().getRootReference();
            auto funcOp = functions.lookup(symbol.getValue());
            if (!funcOp)
            {
                continue;
            }

            auto *user = use.getUser();
            if (mlir::isa<mlir_ts::SymbolCallInternalOp, mlir_ts::CallOp, mlir_ts::InvokeOp>(user))
            {
                continue;
            }

            if (mlir::isa<mlir_ts::SymbolRefOp>(user) && isClassVTableEntry(user))
            {
                continue;
            }

            // passed only to a parameter whose family is closed: its calls are resolved (spec 33)
            if (callbackPassing.contains(user))
            {
                continue;
            }

            int64_t index = -1;
            if (auto refOp = mlir::dyn_cast<mlir_ts::ThisVirtualSymbolRefOp>(user))
            {
                index = refOp.getIndex();
            }
            else if (auto refOp = mlir::dyn_cast<mlir_ts::VirtualSymbolRefOp>(user))
            {
                index = refOp.getIndex();
            }

            auto inFunction = !!user->getParentOfType<mlir::FunctionOpInterface>();
            if (inFunction && mlir::isa<mlir_ts::ThisVirtualSymbolRefOp, mlir_ts::VirtualSymbolRefOp,
                                        mlir_ts::ThisSymbolRefOp, mlir_ts::SymbolRefOp>(user) &&
                onlyCalled(user->getResult(0)))
            {
                // a virtual call this pass cannot resolve reaches its family without the facts
                if (index >= 0)
                {
                    Callees callees;
                    resolveFamily(index, symbol, callees);
                    if (!callees.known)
                    {
                        openFamily(index, symbol, "an override may be defined in another module");
                    }
                }

                continue;
            }

            open.try_emplace(funcOp, "it is used other than by a call");
            if (index >= 0)
            {
                openFamily(index, symbol, "it is used other than by a call");
            }
        }
    }

    void OwnershipSignaturePass::openFamily(int64_t index, mlir::StringAttr symbol, const char *why)
    {
        for (auto member : familyOf(index, symbol))
        {
            if (auto memberOp = functions.lookup(member.getValue()))
            {
                open.try_emplace(memberOp, why);
            }
        }
    }

} // namespace own_signature
