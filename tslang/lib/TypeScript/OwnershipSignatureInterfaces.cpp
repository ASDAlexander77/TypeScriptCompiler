// OwnershipSignaturePass (see OwnershipSignaturePass.h): interfaces closed in this module (G9,
// spec 31) - their method and field families, and the casts that open them.

#include "OwnershipSignaturePass.h"

#define DEBUG_TYPE "own"

namespace own_signature
{

    std::pair<mlir::Type, int64_t> OwnershipSignaturePass::slotField(mlir::Value value)
    {
        while (auto castOp = value ? value.getDefiningOp<mlir_ts::CastOp>() : mlir_ts::CastOp())
        {
            value = castOp.getIn();
        }

        auto propertyRefOp = value ? value.getDefiningOp<mlir_ts::PropertyRefOp>() : mlir_ts::PropertyRefOp();
        auto nullObject = propertyRefOp ? propertyRefOp.getObjectRef().getDefiningOp<mlir_ts::CastOp>() : mlir_ts::CastOp();
        if (!nullObject || !nullObject.getIn().getDefiningOp<mlir_ts::NullOp>() ||
            !mlir::isa<mlir_ts::ObjectType>(nullObject.getType()))
        {
            return {};
        }

        auto tupleType = storageTupleOf(nullObject.getType());
        if (!tupleType)
        {
            return {};
        }

        return {tupleType, static_cast<int64_t>(propertyRefOp.getPosition())};
    }

    void OwnershipSignaturePass::collectFieldFunctions(mlir::ModuleOp module)
    {
        llvm::DenseMap<std::pair<mlir::Type, int64_t>, llvm::SetVector<mlir::StringAttr>> stored;
        llvm::DenseSet<std::pair<mlir::Type, int64_t>> unknownFields;
        llvm::DenseSet<mlir::Type> unknownTypes;
        // a function of unknown origin stored through an interface this pass cannot follow
        auto unknownAnywhere = false;

        auto store = [&](std::pair<mlir::Type, int64_t> key, mlir::Value value) {
            auto symbol = slotFunction(value);
            auto funcOp = symbol ? functions.lookup(symbol.getValue()) : mlir_ts::FuncOp();
            if (funcOp && !funcOp.isDeclaration())
            {
                stored[key].insert(symbol);
            }
            else
            {
                unknownFields.insert(key);
            }
        };

        auto isFunctionType = [](mlir::Type type) {
            return mlir::isa<mlir_ts::FunctionType, mlir_ts::BoundFunctionType, mlir_ts::HybridFunctionType>(type);
        };

        module.walk([&](mlir::Operation *op) {
            // the literal's own values
            if (auto constantOp = mlir::dyn_cast<mlir_ts::ConstantOp>(op))
            {
                auto constTupleType = mlir::dyn_cast<mlir_ts::ConstTupleType>(constantOp.getType());
                auto values = mlir::dyn_cast<mlir::ArrayAttr>(constantOp.getValue());
                if (!constTupleType || !values)
                {
                    return;
                }

                mlir::Type tupleType = mlir_ts::TupleType::get(&getContext(), constTupleType.getFields());
                for (auto [index, value] : llvm::enumerate(values))
                {
                    std::pair<mlir::Type, int64_t> key{tupleType, static_cast<int64_t>(index)};
                    if (auto symbol = mlir::dyn_cast<mlir::FlatSymbolRefAttr>(value))
                    {
                        auto funcOp = functions.lookup(symbol.getValue());
                        if (funcOp && !funcOp.isDeclaration())
                        {
                            stored[key].insert(symbol.getAttr());
                        }
                        else
                        {
                            unknownFields.insert(key);
                        }
                    }
                    else if (!mlir::isa<mlir::UnitAttr>(value) && index < constTupleType.getFields().size() &&
                             isFunctionType(constTupleType.getFields()[index].type))
                    {
                        unknownFields.insert(key);
                    }
                }

                return;
            }

            // a store into the field through the object
            if (auto propertyRefOp = mlir::dyn_cast<mlir_ts::PropertyRefOp>(op))
            {
                auto tupleType = storageTupleOf(propertyRefOp.getObjectRef().getType());
                if (!tupleType)
                {
                    return;
                }

                std::pair<mlir::Type, int64_t> key{tupleType, static_cast<int64_t>(propertyRefOp.getPosition())};
                for (auto &use : propertyRefOp->getUses())
                {
                    auto *user = use.getOwner();
                    if (mlir::isa<mlir_ts::LoadOp, mlir_ts::ReleaseSlotOp, mlir_ts::RetainSlotOp>(user))
                    {
                        continue;
                    }

                    auto storeOp = mlir::dyn_cast<mlir_ts::StoreOp>(user);
                    if (storeOp && storeOp.getReference() == propertyRefOp.getResult() && storeOp.getValue() != propertyRefOp.getResult())
                    {
                        store(key, storeOp.getValue());
                        continue;
                    }

                    // an interface vtable's offset of the field (slotField) reads nothing
                    if (user->getParentOfType<mlir_ts::GlobalOp>() && mlir::isa<mlir_ts::CastOp, mlir_ts::InsertPropertyOp>(user))
                    {
                        continue;
                    }

                    unknownFields.insert(key);
                }

                return;
            }

            // a store into the field through an interface: into each literal its vtables reach
            if (auto interfaceRefOp = mlir::dyn_cast<mlir_ts::InterfaceSymbolRefOp>(op))
            {
                auto refType = mlir::dyn_cast<mlir_ts::RefType>(interfaceRefOp.getType());
                if (!refType || !isFunctionType(refType.getElementType()))
                {
                    return;
                }

                auto interfaceType = mlir::dyn_cast<mlir_ts::InterfaceType>(interfaceRefOp.getInterfaceVal().getType());
                auto found = interfaceType ? interfaceFieldFamilies.find({interfaceType.getName().getValue().str(),
                                                                         static_cast<int64_t>(interfaceRefOp.getIndex()) +
                                                                             INTERFACE_VTABLE_HEADER_SLOTS})
                                           : interfaceFieldFamilies.end();
                for (auto &use : interfaceRefOp->getUses())
                {
                    auto *user = use.getOwner();
                    if (mlir::isa<mlir_ts::LoadOp, mlir_ts::ReleaseSlotOp, mlir_ts::RetainSlotOp>(user))
                    {
                        continue;
                    }

                    auto storeOp = mlir::dyn_cast<mlir_ts::StoreOp>(user);
                    if (storeOp && storeOp.getReference() == interfaceRefOp.getResult() && found != interfaceFieldFamilies.end() &&
                        found->second.closed)
                    {
                        for (auto target : found->second.targets)
                        {
                            store(target, storeOp.getValue());
                        }

                        continue;
                    }

                    unknownAnywhere = true;
                }

                return;
            }

            // an object of the literal's type made other than by the literal: its fields may hold
            // anything. A cast keeps the type's own value (the literal's box seen as the object, an
            // object, a reference to its storage) or makes the null object a vtable's offsets are
            // taken from; from anything else it reinterprets.
            if (auto castOp = mlir::dyn_cast<mlir_ts::CastOp>(op))
            {
                auto tupleType = storageTupleOf(castOp.getType());
                if (tupleType && !mlir::isa<mlir_ts::NullType>(castOp.getIn().getType()) &&
                    storageTupleOf(castOp.getIn().getType()) != tupleType)
                {
                    unknownTypes.insert(tupleType);
                }

                return;
            }

            if (mlir::isa<mlir_ts::LoadOp>(op) || isCall(op) || isView(op) || mlir::isa<mlir_ts::ExtractPropertyOp>(op))
            {
                return;
            }

            for (auto result : op->getResults())
            {
                if (mlir::isa<mlir_ts::TupleType, mlir_ts::ObjectType>(result.getType()))
                {
                    if (auto tupleType = storageTupleOf(result.getType()))
                    {
                        unknownTypes.insert(tupleType);
                    }
                }
            }
        });

        for (auto &[key, family] : interfaceFieldFamilies)
        {
            if (!family.closed)
            {
                continue;
            }

            for (auto target : family.targets)
            {
                if (unknownAnywhere || unknownTypes.contains(target.first) || unknownFields.contains(target))
                {
                    family.closed = false;
                    break;
                }

                for (auto member : stored.lookup(target))
                {
                    if (!llvm::is_contained(family.members, member))
                    {
                        family.members.push_back(member);
                    }
                }
            }

            family.closed = family.closed && !family.members.empty();
        }
    }

    void OwnershipSignaturePass::valueInterfaces(mlir::Type type, llvm::StringSet<> &found, int depth)
    {
        if (!type || depth > 16)
        {
            return;
        }

        if (auto interfaceType = mlir::dyn_cast<mlir_ts::InterfaceType>(type))
        {
            found.insert(interfaceType.getName().getValue());
            return;
        }

        auto fields = [&](llvm::ArrayRef<mlir_ts::FieldInfo> infos) {
            for (auto &info : infos)
            {
                valueInterfaces(info.type, found, depth + 1);
            }
        };

        if (auto tupleType = mlir::dyn_cast<mlir_ts::TupleType>(type))
        {
            fields(tupleType.getFields());
        }
        else if (auto constTupleType = mlir::dyn_cast<mlir_ts::ConstTupleType>(type))
        {
            fields(constTupleType.getFields());
        }
        else if (auto unionType = mlir::dyn_cast<mlir_ts::UnionType>(type))
        {
            for (auto member : unionType.getTypes())
            {
                valueInterfaces(member, found, depth + 1);
            }
        }
        else if (auto optionalType = mlir::dyn_cast<mlir_ts::OptionalType>(type))
        {
            valueInterfaces(optionalType.getElementType(), found, depth + 1);
        }
        else if (auto arrayType = mlir::dyn_cast<mlir_ts::ArrayType>(type))
        {
            valueInterfaces(arrayType.getElementType(), found, depth + 1);
        }
        else if (auto constArrayType = mlir::dyn_cast<mlir_ts::ConstArrayType>(type))
        {
            valueInterfaces(constArrayType.getElementType(), found, depth + 1);
        }
        else if (auto refType = mlir::dyn_cast<mlir_ts::RefType>(type))
        {
            valueInterfaces(refType.getElementType(), found, depth + 1);
        }
        else if (auto valueRefType = mlir::dyn_cast<mlir_ts::ValueRefType>(type))
        {
            valueInterfaces(valueRefType.getElementType(), found, depth + 1);
        }
        else if (auto boundRefType = mlir::dyn_cast<mlir_ts::BoundRefType>(type))
        {
            valueInterfaces(boundRefType.getElementType(), found, depth + 1);
        }
        else if (auto sharedType = mlir::dyn_cast<mlir_ts::SharedType>(type))
        {
            valueInterfaces(sharedType.getElementType(), found, depth + 1);
        }
    }

    void OwnershipSignaturePass::anyInterfaces(mlir::Type type, llvm::StringSet<> &found, llvm::DenseSet<mlir::Type> &seen,
                              llvm::StringSet<> *classes)
    {
        if (!type || !seen.insert(type).second)
        {
            return;
        }

        if (auto interfaceType = mlir::dyn_cast<mlir_ts::InterfaceType>(type))
        {
            found.insert(interfaceType.getName().getValue());
            return;
        }

        auto fields = [&](llvm::ArrayRef<mlir_ts::FieldInfo> infos) {
            for (auto &info : infos)
            {
                anyInterfaces(info.type, found, seen, classes);
            }
        };

        // the types whose fields are not parameters of the type
        if (auto classStorageType = mlir::dyn_cast<mlir_ts::ClassStorageType>(type))
        {
            if (classes)
            {
                classes->insert(classStorageType.getName().getValue());
            }

            fields(classStorageType.getFields());
            return;
        }

        if (auto objectStorageType = mlir::dyn_cast<mlir_ts::ObjectStorageType>(type))
        {
            fields(objectStorageType.getFields());
            return;
        }

        if (auto tupleType = mlir::dyn_cast<mlir_ts::TupleType>(type))
        {
            fields(tupleType.getFields());
            return;
        }

        if (auto constTupleType = mlir::dyn_cast<mlir_ts::ConstTupleType>(type))
        {
            fields(constTupleType.getFields());
            return;
        }

        type.walkImmediateSubElements([](mlir::Attribute) {}, [&](mlir::Type sub) { anyInterfaces(sub, found, seen, classes); });
    }

    void OwnershipSignaturePass::functionValues(mlir::Type type, llvm::SmallVectorImpl<mlir::Type> &found, int depth)
    {
        if (!type || depth > 16)
        {
            return;
        }

        if (mlir::isa<mlir_ts::FunctionType, mlir_ts::BoundFunctionType, mlir_ts::HybridFunctionType,
                      mlir_ts::ExtensionFunctionType, mlir_ts::ConstructFunctionType>(type))
        {
            found.push_back(type);
            return;
        }

        auto fields = [&](llvm::ArrayRef<mlir_ts::FieldInfo> infos) {
            for (auto &info : infos)
            {
                functionValues(info.type, found, depth + 1);
            }
        };

        if (auto tupleType = mlir::dyn_cast<mlir_ts::TupleType>(type))
        {
            fields(tupleType.getFields());
        }
        else if (auto constTupleType = mlir::dyn_cast<mlir_ts::ConstTupleType>(type))
        {
            fields(constTupleType.getFields());
        }
        else if (auto unionType = mlir::dyn_cast<mlir_ts::UnionType>(type))
        {
            for (auto member : unionType.getTypes())
            {
                functionValues(member, found, depth + 1);
            }
        }
        else if (auto optionalType = mlir::dyn_cast<mlir_ts::OptionalType>(type))
        {
            functionValues(optionalType.getElementType(), found, depth + 1);
        }
        else if (auto arrayType = mlir::dyn_cast<mlir_ts::ArrayType>(type))
        {
            functionValues(arrayType.getElementType(), found, depth + 1);
        }
        else if (auto constArrayType = mlir::dyn_cast<mlir_ts::ConstArrayType>(type))
        {
            functionValues(constArrayType.getElementType(), found, depth + 1);
        }
        else if (auto sharedType = mlir::dyn_cast<mlir_ts::SharedType>(type))
        {
            functionValues(sharedType.getElementType(), found, depth + 1);
        }
    }

    std::pair<llvm::SmallVector<mlir::Type>, llvm::SmallVector<mlir::Type>> OwnershipSignaturePass::signatureOf(mlir::Type type)
    {
        llvm::SmallVector<mlir::Type> inputs;
        llvm::SmallVector<mlir::Type> results;
        if (auto functionType = mlir::dyn_cast<mlir_ts::FunctionType>(type))
        {
            inputs.append(functionType.getInputs().begin(), functionType.getInputs().end());
            results.append(functionType.getResults().begin(), functionType.getResults().end());
        }
        else if (auto boundType = mlir::dyn_cast<mlir_ts::BoundFunctionType>(type))
        {
            inputs.append(boundType.getInputs().begin(), boundType.getInputs().end());
            results.append(boundType.getResults().begin(), boundType.getResults().end());
        }
        else if (auto hybridType = mlir::dyn_cast<mlir_ts::HybridFunctionType>(type))
        {
            inputs.append(hybridType.getInputs().begin(), hybridType.getInputs().end());
            results.append(hybridType.getResults().begin(), hybridType.getResults().end());
        }
        else
        {
            // a kind this does not read: never the same signature as another
            inputs.push_back(type);
        }

        return {inputs, results};
    }

    bool OwnershipSignaturePass::castReinterpretsFunction(mlir::Type in, mlir::Type out, llvm::StringSet<> &interfaces, bool &boxing)
    {
        // `!ts.opaque` is the compiler's own (a vtable slot read as the `.instanceOf` it holds), not
        // a value a program can put a function into
        auto isOpaque = [](mlir::Type type) { return mlir::isa<mlir_ts::AnyType, mlir_ts::UnknownType>(type); };
        llvm::SmallVector<mlir::Type> inFunctions;
        llvm::SmallVector<mlir::Type> outFunctions;
        functionValues(in, inFunctions);
        functionValues(out, outFunctions);
        boxing = false;
        if ((inFunctions.empty() && outFunctions.empty()) || in == out)
        {
            return false;
        }

        auto mention = [&](llvm::ArrayRef<mlir::Type> types) {
            for (auto type : types)
            {
                llvm::DenseSet<mlir::Type> seen;
                anyInterfaces(type, interfaces, seen);
            }
        };

        if (isOpaque(out))
        {
            boxing = true;
            mention(inFunctions);
            return false;
        }

        // the same signatures, but for `this` seen as an opaque pointer on one side, or bound on one
        // side only (a closure's box, a bound function's object: the caller supplies neither)
        auto sameSignature = [](mlir::Type a, mlir::Type b) {
            auto [aInputs, aResults] = signatureOf(a);
            auto [bInputs, bResults] = signatureOf(b);
            auto dropBoundThis = [](mlir::Type type, llvm::SmallVector<mlir::Type> &inputs) {
                if (!inputs.empty() && (mlir::isa<mlir_ts::BoundFunctionType>(type) || mlir::isa<mlir_ts::OpaqueType>(inputs.front())))
                {
                    inputs.erase(inputs.begin());
                }
            };

            if (aInputs.size() == bInputs.size() + 1)
            {
                dropBoundThis(a, aInputs);
            }
            else if (bInputs.size() == aInputs.size() + 1)
            {
                dropBoundThis(b, bInputs);
            }

            if (aResults != bResults || aInputs.size() != bInputs.size())
            {
                return false;
            }

            for (auto [index, pair] : llvm::enumerate(llvm::zip(aInputs, bInputs)))
            {
                auto [aInput, bInput] = pair;
                if (aInput != bInput &&
                    !(index == 0 && (mlir::isa<mlir_ts::OpaqueType>(aInput) || mlir::isa<mlir_ts::OpaqueType>(bInput))))
                {
                    return false;
                }
            }

            return true;
        };

        // a function out of `any`, or one function seen as another; a function seen as something
        // that is no function (a vtable slot's reference to it) is the compiler's own
        auto reinterprets = isOpaque(in) && !outFunctions.empty();
        if (!inFunctions.empty() && !outFunctions.empty())
        {
            reinterprets = reinterprets || inFunctions.size() != outFunctions.size();
            for (auto [inFunction, outFunction] : llvm::zip(inFunctions, outFunctions))
            {
                reinterprets = reinterprets || !sameSignature(inFunction, outFunction);
            }
        }

        if (reinterprets)
        {
            mention(inFunctions);
            mention(outFunctions);
        }

        return reinterprets;
    }

    void OwnershipSignaturePass::memberTypeInterfaces(mlir::Type type, llvm::StringSet<> &found)
    {
        llvm::DenseSet<mlir::Type> seen;
        anyInterfaces(type, found, seen);
    }

    bool OwnershipSignaturePass::isNullish(mlir::Type type)
    {
        return mlir::isa<mlir_ts::NullType, mlir_ts::UndefinedType>(type);
    }

    bool OwnershipSignaturePass::castMakesInterface(mlir::Type in, mlir::Type out, int depth)
    {
        // null seen as an interface value has no vtable: a call through it reaches nothing
        if (in == out || depth > 16 || isNullish(in))
        {
            return false;
        }

        llvm::StringSet<> made;
        valueInterfaces(out, made);
        if (made.empty())
        {
            return false;
        }

        if (mlir::isa<mlir_ts::InterfaceType>(out))
        {
            return !holdsAsIs(in, out, /*onlyNullable=*/true);
        }

        if (mlir::isa<mlir_ts::UnionType, mlir_ts::OptionalType>(out))
        {
            if (isNullish(in) || holdsAsIs(out, in))
            {
                return false;
            }

            llvm::SmallVector<mlir::Type> members;
            if (auto unionType = mlir::dyn_cast<mlir_ts::UnionType>(in))
            {
                members.append(unionType.getTypes().begin(), unionType.getTypes().end());
            }
            else if (auto optionalType = mlir::dyn_cast<mlir_ts::OptionalType>(in))
            {
                members.push_back(optionalType.getElementType());
            }
            else
            {
                return true;
            }

            return !llvm::all_of(members, [&](mlir::Type member) { return isNullish(member) || holdsAsIs(out, member); });
        }

        auto tupleFields = [](mlir::Type type) -> std::optional<llvm::ArrayRef<mlir_ts::FieldInfo>> {
            if (auto tupleType = mlir::dyn_cast<mlir_ts::TupleType>(type))
            {
                return tupleType.getFields();
            }

            if (auto constTupleType = mlir::dyn_cast<mlir_ts::ConstTupleType>(type))
            {
                return constTupleType.getFields();
            }

            return std::nullopt;
        };

        if (auto outFields = tupleFields(out))
        {
            auto inFields = tupleFields(in);
            if (!inFields || inFields->size() != outFields->size())
            {
                return true;
            }

            for (auto [inField, outField] : llvm::zip(*inFields, *outFields))
            {
                if (castMakesInterface(inField.type, outField.type, depth + 1))
                {
                    return true;
                }
            }

            return false;
        }

        auto elementOf = [](mlir::Type type) -> mlir::Type {
            if (auto arrayType = mlir::dyn_cast<mlir_ts::ArrayType>(type))
            {
                return arrayType.getElementType();
            }

            if (auto constArrayType = mlir::dyn_cast<mlir_ts::ConstArrayType>(type))
            {
                return constArrayType.getElementType();
            }

            return {};
        };

        if (auto outElement = elementOf(out))
        {
            auto inElement = elementOf(in);
            return !inElement || castMakesInterface(inElement, outElement, depth + 1);
        }

        // a reference seen as another (`!ts.bound_ref<T>` as `!ts.ref<T>`): what it points to
        auto pointee = [](mlir::Type type) -> mlir::Type {
            if (auto refType = mlir::dyn_cast<mlir_ts::RefType>(type))
            {
                return refType.getElementType();
            }

            if (auto valueRefType = mlir::dyn_cast<mlir_ts::ValueRefType>(type))
            {
                return valueRefType.getElementType();
            }

            if (auto boundRefType = mlir::dyn_cast<mlir_ts::BoundRefType>(type))
            {
                return boundRefType.getElementType();
            }

            return {};
        };

        if (auto outPointee = pointee(out))
        {
            auto inPointee = pointee(in);
            return !inPointee || castMakesInterface(inPointee, outPointee, depth + 1);
        }

        // anything else reinterpreted: what is read out of it was never checked
        return true;
    }

    mlir::StringAttr OwnershipSignaturePass::slotFunction(mlir::Value value)
    {
        while (auto castOp = value ? value.getDefiningOp<mlir_ts::CastOp>() : mlir_ts::CastOp())
        {
            value = castOp.getIn();
        }

        auto symbolRefOp = value ? value.getDefiningOp<mlir_ts::SymbolRefOp>() : mlir_ts::SymbolRefOp();
        return symbolRefOp ? symbolRefOp.getIdentifierAttr().getAttr() : mlir::StringAttr();
    }

    mlir_ts::AddressOfOp OwnershipSignaturePass::addressOfThroughCasts(mlir::Value value)
    {
        while (auto castOp = value ? value.getDefiningOp<mlir_ts::CastOp>() : mlir_ts::CastOp())
        {
            value = castOp.getIn();
        }

        return value ? value.getDefiningOp<mlir_ts::AddressOfOp>() : mlir_ts::AddressOfOp();
    }

    llvm::StringSet<> OwnershipSignaturePass::exportedWords(mlir::ModuleOp module)
    {
        llvm::StringSet<> words;
        module.walk([&](mlir_ts::GlobalOp globalOp) {
            if (!globalOp.getSymName().starts_with("__decls"))
            {
                return;
            }

            globalOp.getInitializerRegion().walk([&](mlir_ts::ConstantOp constantOp) {
                auto text = mlir::dyn_cast<mlir::StringAttr>(constantOp.getValue());
                if (!text)
                {
                    return;
                }

                llvm::StringRef rest = text.getValue();
                while (!rest.empty())
                {
                    auto isWordChar = [](char c) { return llvm::isAlnum(c) || c == '_' || c == '$'; };
                    rest = rest.drop_while([&](char c) { return !isWordChar(c); });
                    auto word = rest.take_while(isWordChar);
                    if (!word.empty())
                    {
                        words.insert(word);
                    }

                    rest = rest.drop_front(word.size());
                }
            });
        });

        return words;
    }

    void OwnershipSignaturePass::collectInterfaceFamilies(mlir::ModuleOp module)
    {
        InterfaceScan scan;
        readInterfaceVTables(module, scan);
        traceInterfaceValues(module, scan);
        openExposedInterfaces(module, scan);
        openThroughMembers(scan);
        openLooseVTables(scan);
        buildInterfaceFamilies(scan);
        collectFieldFunctions(module);
        markInterfaceReached(scan);
    }

    void OwnershipSignaturePass::readInterfaceVTables(mlir::ModuleOp module, InterfaceScan &scan)
    {
        module.walk([&](mlir_ts::GlobalOp globalOp) {
            auto name = globalOp.getSymName();
            // an interface's null vtable implements nothing (mlirGenNullInterfaceVTable): a null
            // interface is seen through it, and its methods throw
            if (!name.ends_with("..vtbl") || globalOp->hasAttr(NULL_INTERFACE_VTABLE_ATTR_NAME))
            {
                return;
            }

            auto owner = name.drop_back(llvm::StringRef("..vtbl").size());
            llvm::DenseMap<int64_t, mlir::Value> values;
            globalOp.getInitializerRegion().walk([&](mlir_ts::InsertPropertyOp insertOp) {
                if (insertOp.getPosition().size() == 1)
                {
                    values[insertOp.getPosition()[0]] = insertOp.getValue();
                }
            });

            // a class's own vtable starts with its `..instanceOf`; an interface vtable has the type tag
            // ahead of it (INTERFACE_VTABLE_TYPE_SLOT)
            auto classHeader = slotFunction(values.lookup(0));
            auto isClassVTable = classHeader && classHeader.getValue() == (owner + "..instanceOf").str();
            auto header = isClassVTable ? classHeader : slotFunction(values.lookup(INTERFACE_VTABLE_INSTANCEOF_SLOT));
            auto tupleType = mlir::dyn_cast<mlir_ts::TupleType>(globalOp.getType());
            if (!header || !tupleType)
            {
                scan.unread.push_back(name);
                return;
            }

            if (isClassVTable)
            {
                // an exported class (not in an executable, whose exports no module sees)
                auto exposed = !executable && (!globalOp.isPrivate() || globalOp->hasAttr("export") || globalOp->hasAttr(DLL_EXPORT));
                ownClasses.insert(owner);
                if (exposed)
                {
                    nameableClasses.insert(owner);
                }

                for (auto &[position, value] : values)
                {
                    auto addressOfOp = addressOfThroughCasts(value);
                    auto id = position < static_cast<int64_t>(tupleType.size())
                                  ? mlir::dyn_cast_or_null<mlir::StringAttr>(tupleType.getFieldInfo(position).id)
                                  : mlir::StringAttr();
                    if (!addressOfOp || !id)
                    {
                        continue;
                    }

                    scan.classHeld[id.getValue()].insert(addressOfOp.getGlobalName());
                    if (exposed)
                    {
                        scan.open.insert(id.getValue()); // another module may extend the class
                    }
                }

                return;
            }

            if (!header.getValue().ends_with("..instanceOf") && header.getValue() != ".instanceOf.none")
            {
                scan.unread.push_back(name);
                return;
            }

            auto &vtable = scan.vtables[name];
            vtable.type = tupleType;
            for (auto &[position, value] : values)
            {
                if (position >= INTERFACE_VTABLE_HEADER_SLOTS)
                {
                    vtable.slots[position] = slotFunction(value);
                    if (auto field = slotField(value); field.first)
                    {
                        vtable.fields[position] = field;
                    }
                }
            }
        });
    }

    void OwnershipSignaturePass::traceInterfaceValues(mlir::ModuleOp module, InterfaceScan &scan)
    {
        auto vtableOf = [&](mlir::Value value) -> llvm::StringRef {
            // the global itself
            if (auto addressOfOp = addressOfThroughCasts(value))
            {
                return scan.vtables.count(addressOfOp.getGlobalName()) ? addressOfOp.getGlobalName() : llvm::StringRef();
            }

            // a copy of it on the heap, its absent members' slots patched
            while (auto castOp = value.getDefiningOp<mlir_ts::CastOp>())
            {
                value = castOp.getIn();
            }

            if (auto newOp = value.getDefiningOp<mlir_ts::NewOp>())
            {
                llvm::StringRef copied;
                for (auto *user : newOp->getUsers())
                {
                    auto storeOp = mlir::dyn_cast<mlir_ts::StoreOp>(user);
                    auto loadOp = storeOp && storeOp.getReference() == newOp.getResult()
                                      ? storeOp.getValue().getDefiningOp<mlir_ts::LoadOp>()
                                      : mlir_ts::LoadOp();
                    auto addressOfOp = loadOp ? addressOfThroughCasts(loadOp.getReference()) : mlir_ts::AddressOfOp();
                    if (addressOfOp && scan.vtables.count(addressOfOp.getGlobalName()))
                    {
                        if (!copied.empty())
                        {
                            return {};
                        }

                        copied = addressOfOp.getGlobalName();
                    }
                }

                return copied;
            }

            return {};
        };

        // an object's class vtable read at run time: `ts.VTableOffsetRef(ts.Load(ts.PropertyRef(object, 0)), k)`
        auto readFromClass = [](mlir::Value value) {
            auto offsetRefOp = value.getDefiningOp<mlir_ts::VTableOffsetRefOp>();
            auto loadOp = offsetRefOp ? offsetRefOp.getVtable().getDefiningOp<mlir_ts::LoadOp>() : mlir_ts::LoadOp();
            auto propertyRefOp = loadOp ? loadOp.getReference().getDefiningOp<mlir_ts::PropertyRefOp>() : mlir_ts::PropertyRefOp();
            return propertyRefOp && propertyRefOp.getPosition() == 0 &&
                   mlir::isa<mlir_ts::ClassType>(propertyRefOp.getObjectRef().getType());
        };

        // what reads a vtable's address: making an interface value, a class vtable's slot, the copy
        std::function<bool(mlir::Value)> onlyMakesValues = [&](mlir::Value address) {
            return llvm::all_of(address.getUses(), [&](mlir::OpOperand &use) {
                auto *user = use.getOwner();
                if (auto newInterfaceOp = mlir::dyn_cast<mlir_ts::NewInterfaceOp>(user))
                {
                    return newInterfaceOp.getInterfaceVTable() == use.get();
                }

                if (mlir::isa<mlir_ts::CastOp>(user))
                {
                    return onlyMakesValues(user->getResult(0));
                }

                if (mlir::isa<mlir_ts::InsertPropertyOp>(user))
                {
                    return use.getOperandNumber() == 0 && user->getParentOfType<mlir_ts::GlobalOp>() != nullptr;
                }

                // the vtable a class object is seen through, chosen by a test for null
                // (castToInterfaceSpecialCases): what the choice makes, a `ts.If`'s result or, once
                // lowered, the block argument the branches pass it to
                if (mlir::isa<mlir_ts::ResultOp>(user))
                {
                    auto ifOp = mlir::dyn_cast<mlir_ts::IfOp>(user->getParentOp());
                    return ifOp && onlyMakesValues(ifOp.getResult(use.getOperandNumber()));
                }

                if (auto branchOp = mlir::dyn_cast<mlir::BranchOpInterface>(user))
                {
                    auto blockArgument = branchOp.getSuccessorBlockArgument(use.getOperandNumber());
                    return blockArgument && onlyMakesValues(*blockArgument);
                }

                // the copy: read whole, and stored into a block `ts.New` made
                if (auto loadOp = mlir::dyn_cast<mlir_ts::LoadOp>(user))
                {
                    return llvm::all_of(loadOp->getUsers(), [&](mlir::Operation *loadUser) {
                        auto storeOp = mlir::dyn_cast<mlir_ts::StoreOp>(loadUser);
                        return storeOp && storeOp.getValue() == loadOp.getResult() &&
                               storeOp.getReference().getDefiningOp<mlir_ts::NewOp>();
                    });
                }

                return false;
            });
        };

        auto isOpaque = [](mlir::Type type) { return mlir::isa<mlir_ts::AnyType, mlir_ts::UnknownType, mlir_ts::OpaqueType>(type); };
        module.walk([&](mlir::Operation *op) {
            if (auto newInterfaceOp = mlir::dyn_cast<mlir_ts::NewInterfaceOp>(op))
            {
                auto name = mlir::cast<mlir_ts::InterfaceType>(newInterfaceOp.getType()).getName().getValue();

                // a class object's vtable is chosen by a test for null (castToInterfaceSpecialCases):
                // a `ts.If` yields one from each branch or, once lowered, the branches pass one each
                // to a block argument. A null object's is a cast of null (the lowering puts in the
                // interface's null vtable, whose methods throw), made by nothing that implements the
                // interface.
                llvm::SmallVector<mlir::Value> vtableChoices;
                auto addChoice = [&](mlir::Value yielded) {
                    auto stripped = yielded;
                    while (auto castOp = stripped.getDefiningOp<mlir_ts::CastOp>())
                    {
                        stripped = castOp.getIn();
                    }

                    if (!stripped.getDefiningOp<mlir_ts::NullOp>())
                    {
                        vtableChoices.push_back(yielded);
                    }
                };

                auto vtableValue = newInterfaceOp.getInterfaceVTable();
                if (auto ifOp = vtableValue.getDefiningOp<mlir_ts::IfOp>())
                {
                    auto resultIndex = mlir::cast<mlir::OpResult>(vtableValue).getResultNumber();
                    for (auto *region : {&ifOp.getThenRegion(), &ifOp.getElseRegion()})
                    {
                        auto resultOp = region->empty() ? mlir_ts::ResultOp()
                                                        : mlir::dyn_cast<mlir_ts::ResultOp>(region->front().getTerminator());
                        if (!resultOp)
                        {
                            scan.allOpen = true;
                            return;
                        }

                        addChoice(resultOp.getOperand(resultIndex));
                    }
                }
                else if (auto blockArgument = mlir::dyn_cast<mlir::BlockArgument>(vtableValue))
                {
                    auto *block = blockArgument.getOwner();
                    if (block->hasNoPredecessors())
                    {
                        scan.allOpen = true;
                        return;
                    }

                    for (auto *predecessor : block->getPredecessors())
                    {
                        auto branchOp = mlir::dyn_cast<mlir::BranchOpInterface>(predecessor->getTerminator());
                        if (!branchOp)
                        {
                            scan.allOpen = true;
                            return;
                        }

                        for (auto successorIndex : llvm::seq<unsigned>(0, predecessor->getTerminator()->getNumSuccessors()))
                        {
                            if (predecessor->getTerminator()->getSuccessor(successorIndex) != block)
                            {
                                continue;
                            }

                            auto passed = branchOp.getSuccessorOperands(successorIndex)[blockArgument.getArgNumber()];
                            if (!passed)
                            {
                                scan.allOpen = true;
                                return;
                            }

                            addChoice(passed);
                        }
                    }
                }
                else
                {
                    addChoice(vtableValue);
                }

                for (auto vtable : vtableChoices)
                {
                    if (auto found = vtableOf(vtable); !found.empty())
                    {
                        scan.madeWith[name].insert(found);
                    }
                    else if (readFromClass(vtable))
                    {
                        // any class's vtable for the interface (classHeld, all in its families below)
                        if (!scan.classHeld.count(name))
                        {
                            scan.open.insert(name);
                        }
                    }
                    else if (auto extractOp = vtable.getDefiningOp<mlir_ts::ExtractInterfaceVTableOp>())
                    {
                        scan.castOpened.insert(name);
                        valueInterfaces(extractOp.getInterfaceVal().getType(), scan.castOpened);
                    }
                    else
                    {
                        scan.allOpen = true;
                    }
                }

                return;
            }

            if (auto castOp = mlir::dyn_cast<mlir_ts::CastOp>(op))
            {
                auto in = castOp.getIn().getType();
                auto out = castOp.getType();
                if (isOpaque(out))
                {
                    valueInterfaces(in, scan.boxed);
                }

                if (castMakesInterface(in, out))
                {
                    valueInterfaces(out, scan.castOpened);
                    valueInterfaces(in, scan.castOpened);
                    if (isOpaque(in))
                    {
                        valueInterfaces(out, scan.unboxedInto);
                    }
                }

                // a function value reinterpreted carries the interface values of its parameters and
                // results across unseen (review6 x1c, x1r)
                llvm::StringSet<> signatures;
                auto boxing = false;
                if (castReinterpretsFunction(in, out, signatures, boxing))
                {
                    scan.openAll(signatures);
                    if (isOpaque(in))
                    {
                        scan.unboxedFunction = true;
                    }
                }
                else if (boxing)
                {
                    for (auto &entry : signatures)
                    {
                        scan.boxed.insert(entry.getKey());
                    }
                }

                return;
            }

            if (auto interfaceRefOp = mlir::dyn_cast<mlir_ts::InterfaceRefOp>(op))
            {
                if (!interfaceRefOp->use_empty())
                {
                    scan.open.insert(interfaceRefOp.getIdentifier());
                }

                return;
            }

            if (auto symbolRefOp = mlir::dyn_cast<mlir_ts::InterfaceSymbolRefOp>(op))
            {
                if (auto interfaceType = mlir::dyn_cast<mlir_ts::InterfaceType>(symbolRefOp.getInterfaceVal().getType()))
                {
                    memberTypeInterfaces(symbolRefOp.getType(), scan.memberInterfaces[interfaceType.getName().getValue()]);
                }

                return;
            }

            if (auto addressOfOp = mlir::dyn_cast<mlir_ts::AddressOfOp>(op))
            {
                if (auto found = scan.vtables.find(addressOfOp.getGlobalName()); found != scan.vtables.end() && !onlyMakesValues(addressOfOp.getResult()))
                {
                    found->second.written = true;
                }
            }
        });
    }

    void OwnershipSignaturePass::openExposedInterfaces(mlir::ModuleOp module, InterfaceScan &scan)
    {
        // seen from another module: a word of its exported declarations, or in a reachable type. An
        // executable's exports are seen by no other module (USER decision, spec 31): only what it
        // imports - a declared function, an imported global - comes from outside.
        if (!executable)
        {
            exportWords = exportedWords(module);
        }

        llvm::StringSet<> exposed;
        llvm::DenseSet<mlir::Type> seen;
        for (auto &entry : functions)
        {
            // a `public` method of a class no other module can name is not one: only a declaration
            // re-parsed from `__decls` names a function of another module
            auto exported = !executable && (entry.second->hasAttr("export") || entry.second->hasAttr(DLL_EXPORT));
            if (entry.second.isDeclaration() || exported)
            {
                anyInterfaces(entry.second.getFunctionType(), exposed, seen, &nameableClasses);
            }
        }

        for (auto &entry : globals)
        {
            auto globalOp = entry.second;
            // another module's global, declared here with no initializer (an import of a .ts module
            // carries `export`)
            auto imported = globalOp->hasAttr("import") || globalOp->hasAttr(DLL_IMPORT) ||
                            globalOp.getInitializerRegion().empty() || globalOp.getInitializerRegion().front().empty();
            auto exported = !globalOp.isPrivate() || globalOp->hasAttr("export") || globalOp->hasAttr(DLL_EXPORT);
            if (imported || (exported && !executable))
            {
                anyInterfaces(globalOp.getType(), exposed, seen, &nameableClasses);
            }
        }

        scan.openAll(exposed);
        if (!scan.unboxedInto.empty() || scan.unboxedFunction)
        {
            scan.openAll(scan.boxed);
            for (auto &entry : scan.boxed)
            {
                scan.castOpened.insert(entry.getKey());
            }
        }
    }

    void OwnershipSignaturePass::openThroughMembers(InterfaceScan &scan)
    {
        // An interface value made out of another one (or out of `any`) is called with the other's
        // members: `HasF2 { f: (o: Op2, ...) }` seen as `HasF { f: (o: Op, ...) }` hands an `Op` to a
        // function that calls through `Op2` (review6r1 v5, v11f). So every interface either side's
        // members mention is open too, and theirs, recursively (controller ruling, spec 31).
        //
        // A class's vtables for its interfaces are its members' families whether or not this module
        // makes an interface value of the class (a method called directly then keeps its facts), and
        // their slots' types are the interface's members' types: an interface whose values are all
        // made from class objects is read through them (review6r2 d1-d3).
        for (auto &[name, held] : scan.classHeld)
        {
            scan.madeWith[name].insert(held.begin(), held.end());
        }

        for (auto &[name, made] : scan.madeWith)
        {
            for (auto vtableName : made)
            {
                if (auto found = scan.vtables.find(vtableName); found != scan.vtables.end() && found->second.type)
                {
                    for (auto [position, field] : llvm::enumerate(found->second.type.getFields()))
                    {
                        if (position >= INTERFACE_VTABLE_HEADER_SLOTS)
                        {
                            memberTypeInterfaces(field.type, scan.memberInterfaces[name]);
                        }
                    }
                }
            }
        }

        llvm::SmallVector<std::string> work;
        for (auto &entry : scan.castOpened)
        {
            work.push_back(entry.getKey().str());
        }

        llvm::StringSet<> expanded;
        while (!work.empty())
        {
            auto name = work.pop_back_val();
            if (!expanded.insert(name).second)
            {
                continue;
            }

            scan.open.insert(name);
            for (auto &member : scan.memberInterfaces[name])
            {
                work.push_back(member.getKey().str());
            }
        }
    }

    void OwnershipSignaturePass::openLooseVTables(InterfaceScan &scan)
    {
        // A vtable this pass could not read, or one no value of any interface is made with, that is
        // named for an interface (`<class>.<I>..vtbl`, `<I>.<n>..vtbl`): the interface may have an
        // implementation outside its families. Belt and braces - every way to make an interface
        // value is traced above - but a missed member would get its family's facts.
        llvm::DenseSet<llvm::StringRef> made;
        for (auto &entry : scan.madeWith)
        {
            made.insert(entry.second.begin(), entry.second.end());
        }

        llvm::SmallVector<llvm::StringRef> loose(scan.unread.begin(), scan.unread.end());
        for (auto &entry : scan.vtables)
        {
            if (!made.contains(entry.getKey()))
            {
                loose.push_back(entry.getKey());
            }
        }

        for (auto &entry : scan.madeWith)
        {
            auto name = entry.getKey().str();
            if (llvm::any_of(loose, [&](llvm::StringRef vtableName) {
                    return vtableName.ends_with("." + name + "..vtbl") || vtableName.starts_with(name + ".");
                }))
            {
                scan.open.insert(entry.getKey());
            }
        }
    }

    void OwnershipSignaturePass::buildInterfaceFamilies(InterfaceScan &scan)
    {
        auto &words = exportWords;
        auto isOpen = [&](llvm::StringRef name) {
            // a namespaced interface is declared by its last name
            auto last = name.rsplit('.').second;
            return scan.allOpen || scan.open.contains(name) || words.contains(name) || (!last.empty() && words.contains(last));
        };

        // the families
        for (auto &[name, made] : scan.madeWith)
        {
            // a class vtable's slot holding a vtable this pass did not read (another module's)
            auto allRead = llvm::all_of(made, [&](llvm::StringRef vtableName) { return scan.vtables.count(vtableName); });
            auto closed = !isOpen(name) && allRead;
            for (auto vtableName : made)
            {
                if (!scan.vtables.count(vtableName))
                {
                    continue;
                }

                if (allRead)
                {
                    interfaceVTableTypes[name].push_back(scan.vtables[vtableName].type);
                }

                scan.tied.insert(vtableName);
                auto &vtable = scan.vtables[vtableName];
                for (auto [position, field] : llvm::enumerate(vtable.type.getFields()))
                {
                    if (position < INTERFACE_VTABLE_HEADER_SLOTS)
                    {
                        continue;
                    }

                    auto &family = interfaceFamilies[{name.str(), static_cast<int64_t>(position)}];
                    auto id = mlir::dyn_cast_or_null<mlir::StringAttr>(field.id);
                    auto member = vtable.slots.lookup(position);
                    auto funcOp = member ? functions.lookup(member.getValue()) : mlir_ts::FuncOp();
                    if (!family.field)
                    {
                        family.field = id;
                    }

                    // a member that is not private (a `public` method) is open (findOpen): it has no
                    // facts to give, and gives the family none
                    family.closed = family.closed && closed && !vtable.written && id && family.field == id && funcOp &&
                                    !funcOp.isDeclaration() && !declaresCallerFacts(funcOp);
                    if (member && !llvm::is_contained(family.members, member))
                    {
                        family.members.push_back(member);
                    }

                    // the same slot as a field: an object literal's field of function type
                    auto &fieldFamily = interfaceFieldFamilies[{name.str(), static_cast<int64_t>(position)}];
                    auto target = vtable.fields.lookup(position);
                    auto refType = mlir::dyn_cast<mlir_ts::RefType>(field.type);
                    if (!fieldFamily.field)
                    {
                        fieldFamily.field = id;
                    }

                    fieldFamily.closed = fieldFamily.closed && closed && !vtable.written && id && fieldFamily.field == id &&
                                         target.first && refType && mlir::isa<mlir_ts::FunctionType>(refType.getElementType());
                    if (target.first && !llvm::is_contained(fieldFamily.targets, target))
                    {
                        fieldFamily.targets.push_back(target);
                    }
                }
            }
        }
    }

    void OwnershipSignaturePass::markInterfaceReached(InterfaceScan &scan)
    {
        // what an open family holds, and a vtable no interface value is made with, is reached
        // through calls that are not resolved
        auto reached = [&](mlir::StringAttr member) {
            if (auto funcOp = member ? functions.lookup(member.getValue()) : mlir_ts::FuncOp())
            {
                interfaceReached.insert(funcOp);
            }
        };

        for (auto &[key, family] : interfaceFamilies)
        {
            if (!family.closed)
            {
                llvm::for_each(family.members, reached);
            }
        }

        for (auto &entry : scan.vtables)
        {
            if (!scan.tied.contains(entry.getKey()))
            {
                for (auto &slot : entry.second.slots)
                {
                    reached(slot.second);
                }
            }
        }
    }

} // namespace own_signature
