// Interface and enum declaration code generation methods of MLIRGenImpl (see MLIRGenImpl.h).

#include "MLIRGenImpl.h"

namespace typescript
{
namespace mlirgen
{

    mlir::LogicalResult MLIRGenImpl::mlirGen(EnumDeclaration enumDeclarationAST, const GenContext &genContext)
    {
        auto namePtr = MLIRHelper::getName(enumDeclarationAST->name, stringAllocator);
        if (namePtr.empty())
        {
            return mlir::failure();
        }

        SymbolTableScopeT varScope(symbolTable);

        SmallVector<mlir::Type> enumLiteralTypes;
        StringMap<mlir::Attribute> enumValues;

        auto appending = false;
        if (getEnumsMap().contains(namePtr))
        {
            auto dict = getEnumsMap().lookup(namePtr).second;
            for (auto key : dict)
            {
                enumValues[key.getName()] = key.getValue();
            }

            appending = true;
        }
        else
        {
            getEnumsMap().insert(
                { namePtr, { getEnumType().getElementType(), mlir::DictionaryAttr::get(builder.getContext(), {}) } });
        }

        auto &enumInfo = getEnumsMap()[namePtr];

        auto activeBits = 32;
        mlir::IntegerType::SignednessSemantics currentEnumValueSigedness = mlir::IntegerType::SignednessSemantics::Signless;
        llvm::APInt currentEnumValue(32, 0);
        for (auto enumMember : enumDeclarationAST->members)
        {
            auto location = loc(enumMember);

            auto memberNamePtr = MLIRHelper::getName(enumMember->name, stringAllocator);
            if (memberNamePtr.empty())
            {
                return mlir::failure();
            }

            mlir::Attribute enumValueAttr;
            if (enumMember->initializer)
            {
                GenContext enumValueGenContext(genContext);
                enumValueGenContext.allowConstEval = true;
                auto result = mlirGen(enumMember->initializer, enumValueGenContext);
                EXIT_IF_FAILED_OR_NO_VALUE(result)
                auto enumValue = V(result);

                LLVM_DEBUG(llvm::dbgs() << "\n!! enum member: [ " << memberNamePtr << " ] = [ " << enumValue << " ]\n");

                if (auto constOp = dyn_cast<mlir_ts::ConstantOp>(enumValue.getDefiningOp()))
                {
                    enumValueAttr = constOp.getValueAttr();
                    if (auto intAttr = dyn_cast<mlir::IntegerAttr>(enumValueAttr))
                    {
                        if (intAttr.getType().isSignlessInteger())
                        {
                            currentEnumValueSigedness = mlir::IntegerType::SignednessSemantics::Signless;
                        }
                        else if (intAttr.getType().isSignedInteger())
                        {
                            currentEnumValueSigedness = mlir::IntegerType::SignednessSemantics::Signed;
                        }
                        else if (intAttr.getType().isUnsignedInteger())
                        {
                            currentEnumValueSigedness = mlir::IntegerType::SignednessSemantics::Unsigned;
                        }

                        currentEnumValue = intAttr.getValue();
                        auto currentActiveBits = (int)intAttr.getValue().getActiveBits();
                        if (currentActiveBits > activeBits)
                        {
                            activeBits = currentActiveBits;
                        }
                    }
                }
                else
                {
                    emitError(loc(enumMember->initializer))
                        << "enum member '" << memberNamePtr << "' must be constant";
                    return mlir::failure();
                }

                enumLiteralTypes.push_back(enumValue.getType());
                
                auto varDecl = std::make_shared<VariableDeclarationDOM>(memberNamePtr, enumValue.getType(), location);
                DECLARE(varDecl, enumValue);

            }
            else
            {
                if (appending && currentEnumValue == 0 && stage == Stages::Discovering && !enumValues.contains(memberNamePtr))
                {
                    emitError(loc(enumMember))
                        << "In an enum with multiple declarations, only one declaration can omit an initializer for its first enum element";                    
                    return mlir::failure();
                }

                auto typeInt = mlir::IntegerType::get(builder.getContext(), activeBits, currentEnumValueSigedness);
                enumValueAttr = builder.getIntegerAttr(typeInt, currentEnumValue);
                auto indexType = mlir_ts::LiteralType::get(enumValueAttr, typeInt);
                enumLiteralTypes.push_back(indexType);

                LLVM_DEBUG(llvm::dbgs() << "\n!! enum member: " << memberNamePtr << " <- " << indexType << "\n");

                auto varDecl = std::make_shared<VariableDeclarationDOM>(memberNamePtr, indexType, location);
                auto enumVal = builder.create<mlir_ts::ConstantOp>(location, indexType, enumValueAttr);
                DECLARE(varDecl, enumVal);
            }

            LLVM_DEBUG(llvm::dbgs() << "\n!! enum: " << namePtr << " value attr: " << enumValueAttr << "\n");

            enumValues[memberNamePtr] = enumValueAttr;

            // update enum to support req. access
            SmallVector<mlir::NamedAttribute> namedEnumValues;
            for (auto &key : enumValues)
            {
                namedEnumValues.push_back({builder.getStringAttr(key.first()), key.second});
            }

            enumInfo.second = mlir::DictionaryAttr::get(builder.getContext(), namedEnumValues /*adjustedEnumValues*/);

            currentEnumValue++;
        }

        auto location = loc(enumDeclarationAST);
        auto storeType = mth.getUnionTypeWithMerge(location, enumLiteralTypes);

        LLVM_DEBUG(llvm::dbgs() << "\n!! enum: " << namePtr << " storage type: " << storeType << "\n");

        // update enum to support req. access
        enumInfo.first = storeType;

        // register fullName for enum
        auto fullNamePtr = getFullNamespaceName(namePtr); 

        auto enumType = getEnumType(
                    mlir::FlatSymbolRefAttr::get(builder.getContext(), fullNamePtr), 
                    enumInfo.first, 
                    enumInfo.second);

        EnumInfo::TypePtr newEnumPtr;
        if (fullNameEnumsMap.count(fullNamePtr))
        {
            newEnumPtr = fullNameEnumsMap.lookup(fullNamePtr);
            newEnumPtr->enumType = enumType;      
        }
        else
        {
            // register class
            newEnumPtr = std::make_shared<EnumInfo>();
            newEnumPtr->name = namePtr;
            newEnumPtr->fullName = fullNamePtr;
            newEnumPtr->elementNamespace = currentNamespace;      
            newEnumPtr->enumType = enumType;      
            fullNameEnumsMap.insert(fullNamePtr, newEnumPtr);        
        }

        if (getExportModifier(enumDeclarationAST))
        {
            addEnumDeclarationToExport(namePtr, currentNamespace, enumType);
        }

        return mlir::success();
    }

    ValueOrLogicalResult MLIRGenImpl::mlirGenCreateInterfaceVTableForObject(mlir::Location location, mlir::Value in, 
            mlir_ts::ObjectType objectType, InterfaceInfo::TypePtr newInterfacePtr, const GenContext &genContext)
    {
        auto fullObjectInterfaceVTableFieldName = interfaceVTableNameForObject(objectType, newInterfacePtr);
        auto existValue = resolveFullNameIdentifier(location, fullObjectInterfaceVTableFieldName, true, genContext);
        if (existValue)
        {
            return existValue;
        }

        if (mlir::succeeded(
                mlirGenObjectVirtualTableDefinitionForInterface(location, objectType, newInterfacePtr, genContext)))
        {
            auto globalVTableRefValue = resolveFullNameIdentifier(location, fullObjectInterfaceVTableFieldName, true, genContext);

            // we need to update methods references in VTable with functions from object;
            // includes methods inherited via `extends`, not just this interface's own -
            // an inherited method's vtable slot needs patching (or at least visiting) the
            // same as an own one; only checking newInterfacePtr->methods here left every
            // inherited method's slot holding its initial offset-placeholder value
            // (never a real function pointer), crashing on the first call through it.
            llvm::SmallVector<InterfaceMethodInfo *> allMethods;
            newInterfacePtr->getAllMethods(allMethods);
            if (allMethods.size() > 0) {

                mlir_ts::TupleType storeType;
                if (auto objectStoreType = dyn_cast<mlir_ts::ObjectStorageType>(objectType.getStorageType()))
                {
                    storeType = mlir_ts::TupleType::get(builder.getContext(), objectStoreType.getFields());
                }
                else if (auto tupleType = dyn_cast<mlir_ts::TupleType>(objectType.getStorageType()))
                {
                    storeType = tupleType;
                }
                else
                {
                    return mlir::failure();
                }

                // methods this literal's own funcName is known for were already baked into
                // globalVTableRefValue as constant SymbolRefOps by
                // mlirGenObjectVirtualTableDefinitionForInterface (see there, and
                // docs/interface-vtable-simplification-design.md §3) - only the rest (an
                // imported object type reconstructed from a @dllimport declaration, with no
                // local funcOp to name) still need their function pointer read out of the
                // actual object `in` at cast time.
                llvm::SmallVector<InterfaceMethodInfo *> methodsNeedingPatch;
                for (auto* methodPtr : allMethods)
                {
                    auto& method = *methodPtr;
                    auto fieldId = builder.getStringAttr(method.name);
                    auto index = mth.getFieldIndexByFieldName(storeType, fieldId);
                    if (index == -1)
                    {
                        return mlir::failure();
                    }

                    auto fieldInfo = mth.getFieldInfoByIndex(storeType, index);
                    if (lookupObjectLiteralMethodSymbol(fieldInfo.type, fieldId).empty())
                    {
                        methodsNeedingPatch.push_back(methodPtr);
                    }
                }

                if (methodsNeedingPatch.empty())
                {
                    // every method slot is already a compile-time constant - no per-cast
                    // heap allocation needed, same footing as a method-less interface.
                    return globalVTableRefValue;
                }

                // match VTable
                // 1) clone vtable onto the GC heap (NOT a stack VariableOp): this per-object
                // patched vtable is pointed at by the resulting interface value, and that
                // interface can be stored into a global whose initializer lowers to a
                // __cctor function -- a stack alloca would dangle once the __cctor returns,
                // crashing on the first field/method access through the interface. Heap
                // allocation puts the vtable on the same footing as the object itself
                // (already `NewOp`-allocated). See docs/object-literal-boxing-design.md.
                auto vtableType = mlir::cast<mlir_ts::TupleType>(mlir::cast<mlir_ts::RefType>(globalVTableRefValue.getType()).getElementType());
                auto valueVTable = builder.create<mlir_ts::LoadOp>(location, vtableType, globalVTableRefValue);
                auto heapVTable = builder.create<mlir_ts::NewOp>(location, mlir_ts::ValueRefType::get(vtableType), builder.getBoolAttr(false));
                builder.create<mlir_ts::StoreOp>(location, valueVTable, heapVTable);
                auto varVTable = builder.create<mlir_ts::CastOp>(location, globalVTableRefValue.getType(), heapVTable);

                for (auto* methodPtr : methodsNeedingPatch)
                {
                    auto& method = *methodPtr;
                    auto index = mth.getFieldIndexByFieldName(storeType, builder.getStringAttr(method.name));
                    if (index == -1)
                    {
                        return mlir::failure();
                    }

                    auto fieldInfo = mth.getFieldInfoByIndex(storeType, index);

                    auto methodRef = builder.create<mlir_ts::PropertyRefOp>(location, mlir_ts::RefType::get(fieldInfo.type), in, index);

                    LLVM_DEBUG(llvm::dbgs() << "\n!!\n\t vtable method: " << method.name
                                            << "\n\t object method ref: " << V(methodRef) << "\n\n";);

                    // where to save
                    auto fieldInfoVT = mth.getFieldInfoByIndex(vtableType, method.virtualIndex);
                    auto methodRefVT = builder.create<mlir_ts::PropertyRefOp>(location, fieldInfoVT.type, varVTable, method.virtualIndex);

                    LLVM_DEBUG(llvm::dbgs() << "\n!!\n\t vtable method: " << method.name
                                            << "\n\t vtable method ref: " << V(methodRefVT) << "\n\n";);

                    builder.create<mlir_ts::LoadSaveOp>(location, methodRefVT, methodRef);
                }

                // patched VTable
                return V(varVTable);
            }

            return globalVTableRefValue;
        }

        return mlir::failure();
    }

    mlir::LogicalResult MLIRGenImpl::mlirGenObjectVirtualTableDefinitionForInterface(mlir::Location location,
                                                                        mlir_ts::ObjectType objectType,
                                                                        InterfaceInfo::TypePtr newInterfacePtr,
                                                                        const GenContext &genContext)
    {

        MLIRCodeLogic mcl(builder, compileOptions);

        auto storeType = objectType.getStorageType();

        // TODO: should object accept only ObjectStorageType?
        if (auto objectStoreType = dyn_cast<mlir_ts::ObjectStorageType>(storeType))
        {
            storeType = mlir_ts::TupleType::get(builder.getContext(), objectStoreType.getFields());
        }

        auto tupleStorageType = mlir::cast<mlir_ts::TupleType>(mth.convertConstTupleTypeToTupleType(storeType));

        SmallVector<VirtualMethodOrFieldInfo> virtualTable;
        auto result = getInterfaceVirtualTableForObject(location, tupleStorageType, newInterfacePtr, virtualTable);
        if (mlir::failed(result))
        {
            return result;
        }

        // register global
        auto fullClassInterfaceVTableFieldName = interfaceVTableNameForObject(objectType, newInterfacePtr);
        registerVariable(
            location, fullClassInterfaceVTableFieldName, true, VariableType::Var,
            [&](mlir::Location location, const GenContext &genContext) {
                // build vtable from names of methods

                auto virtTuple = getVirtualTableType(virtualTable);

                mlir::Value vtableValue = builder.create<mlir_ts::UndefOp>(location, virtTuple);
                auto fieldIndex = 0;
                for (auto methodOrField : virtualTable)
                {
                    if (methodOrField.isField)
                    {
                        // an object-literal method is resolved via the FIELD path too
                        // (methodsAsFields=true, getInterfaceVirtualTableForObject) since it's
                        // literally stored as a func-typed field on the object - but if it's
                        // capture-free-or-not-relevant (its funcName is compile-time constant
                        // regardless of captures, see addObjectFuncFieldInfo), we already know
                        // exactly which function it is and don't need to derive an object-relative
                        // offset for it at all: emit the function symbol directly, same as the
                        // class vtable path (mlirGenClassVirtualTableDefinitionForInterface) does
                        // for its own (isField=false) methods. This makes the slot a genuine
                        // compile-time constant, skipping mlirGenCreateInterfaceVTableForObject's
                        // per-cast heap-clone-and-patch entirely for such methods (see there).
                        // See docs/interface-vtable-simplification-design.md §3.
                        //
                        // Gate on the INTERFACE's own categorization, not merely "is this
                        // vtable entry func-typed": `methodsAsFields=true` makes every entry
                        // here isField=true regardless of whether the interface declared it as
                        // a MethodSignature (`inc(): void`) or a PropertySignature with a
                        // function type (`toString: () => string`) - only the former is read
                        // through InterfaceMethodAccess (BoundFunctionType access, raw funcptr
                        // slot semantics) at the access site; the latter goes through
                        // InterfaceFieldAccess, which computes thisVal + slotValue expecting an
                        // OFFSET. Substituting a raw function pointer into a
                        // PropertySignature-with-function-type's slot corrupts that
                        // computation (crashes on the resulting garbage address).
                        std::string methodFuncName;
                        if (!methodOrField.isMissing)
                        {
                            auto fieldNameAttr = dyn_cast<mlir::StringAttr>(methodOrField.fieldInfo.id);
                            if (fieldNameAttr && newInterfacePtr->findMethod(fieldNameAttr.getValue()))
                            {
                                methodFuncName = lookupObjectLiteralMethodSymbol(methodOrField.fieldInfo.type, methodOrField.fieldInfo.id);
                            }
                        }

                        if (!methodFuncName.empty())
                        {
                            auto methodConstName = builder.create<mlir_ts::SymbolRefOp>(
                                location, mlir::cast<mlir_ts::FunctionType>(methodOrField.fieldInfo.type),
                                mlir::FlatSymbolRefAttr::get(builder.getContext(), methodFuncName));
                            auto methodConstNameRef = cast(location, mlir_ts::RefType::get(methodOrField.fieldInfo.type),
                                                           methodConstName, genContext);

                            vtableValue = builder.create<mlir_ts::InsertPropertyOp>(
                                location, virtTuple, methodConstNameRef, vtableValue,
                                MLIRHelper::getStructIndex(builder, fieldIndex));
                        }
                        else if (!methodOrField.isMissing)
                        {
                            auto nullObj = builder.create<mlir_ts::NullOp>(location, getNullType());

                            // TODO: test cast result
                            auto objectNull = cast(location, objectType, nullObj, genContext, true);
                            if (!objectNull)
                            {
                                return TypeValueInitType{mlir::Type(), mlir::Value(), TypeProvided::Yes};
                            }

                            auto fieldValue = mlirGenPropertyAccessExpression(location, objectNull,
                                                                              methodOrField.fieldInfo.id, genContext);
                            assert(fieldValue);
                            if (!fieldValue)
                            {
                                return TypeValueInitType{mlir::Type(), mlir::Value(), TypeProvided::Yes};
                            }

                            auto fieldRef = mcl.GetReferenceFromValue(location, fieldValue);

                            LLVM_DEBUG(llvm::dbgs() << "\n!!\n\t vtable field: " << methodOrField.fieldInfo.id
                                                    << "\n\t type: " << methodOrField.fieldInfo.type
                                                    << "\n\t provided data: " << fieldRef << "\n\n";);

                            if (isa<mlir_ts::BoundRefType>(fieldRef.getType()))
                            {
                                fieldRef = cast(location, mlir_ts::RefType::get(methodOrField.fieldInfo.type), fieldRef,
                                                genContext);
                            }
                            else
                            {
                                assert(mlir::cast<mlir_ts::RefType>(fieldRef.getType()).getElementType() ==
                                       methodOrField.fieldInfo.type);
                            }

                            // insert &(null)->field
                            vtableValue = builder.create<mlir_ts::InsertPropertyOp>(
                                location, virtTuple, fieldRef, vtableValue,
                                MLIRHelper::getStructIndex(builder, fieldIndex));
                        }
                        else
                        {
                            // null value, as missing field/method
                            // auto nullObj = builder.create<mlir_ts::NullOp>(location, getNullType());
                            auto negative1 = builder.create<mlir_ts::ConstantOp>(location, builder.getI64Type(),
                                                                                 mth.getI64AttrValue(-1));
                            auto castedPtr = cast(location, mlir_ts::RefType::get(methodOrField.fieldInfo.type),
                                                   negative1, genContext);
                            vtableValue = builder.create<mlir_ts::InsertPropertyOp>(
                                location, virtTuple, castedPtr, vtableValue,
                                MLIRHelper::getStructIndex(builder, fieldIndex));
                        }
                    }
                    else
                    {
                        llvm_unreachable("not implemented yet");
                        /*
                        auto methodConstName = builder.create<mlir_ts::SymbolRefOp>(
                            location, methodOrField.methodInfo.funcOp.getType(),
                            mlir::FlatSymbolRefAttr::get(builder.getContext(),
                        methodOrField.methodInfo.funcOp.getSymName()));

                        vtableValue =
                            builder.create<mlir_ts::InsertPropertyOp>(location, virtTuple, methodConstName, vtableValue,
                                                                      MLIRHelper::getStructIndex(rewriter, fieldIndex));
                        */
                    }

                    fieldIndex++;
                }

                return TypeValueInitType{virtTuple, vtableValue, TypeProvided::Yes};
            },
            genContext);

        return mlir::success();
    }

    mlir::LogicalResult MLIRGenImpl::registerGenericInterface(InterfaceDeclaration interfaceDeclarationAST,
                                                 const GenContext &genContext)
    {
        auto name = MLIRHelper::getName(interfaceDeclarationAST->name);
        if (!name.empty())
        {
            auto namePtr = StringRef(name).copy(stringAllocator);
            auto fullNamePtr = getFullNamespaceName(namePtr);
            if (fullNameGenericInterfacesMap.count(fullNamePtr))
            {
                return mlir::success();
            }

            llvm::SmallVector<TypeParameterDOM::TypePtr> typeParameters;
            if (mlir::failed(
                    processTypeParameters(interfaceDeclarationAST->typeParameters, typeParameters, genContext)))
            {
                return mlir::failure();
            }

            GenericInterfaceInfo::TypePtr newGenericInterfacePtr = std::make_shared<GenericInterfaceInfo>();
            newGenericInterfacePtr->name = namePtr;
            newGenericInterfacePtr->fullName = fullNamePtr;
            newGenericInterfacePtr->elementNamespace = currentNamespace;
            newGenericInterfacePtr->typeParams = typeParameters;
            newGenericInterfacePtr->interfaceDeclaration = interfaceDeclarationAST;
            newGenericInterfacePtr->sourceFile = sourceFile;
            newGenericInterfacePtr->fileName = mainSourceFileName;

            mlirGenInterfaceType(newGenericInterfacePtr, genContext);

            getGenericInterfacesMap().insert({namePtr, newGenericInterfacePtr});
            fullNameGenericInterfacesMap.insert(fullNamePtr, newGenericInterfacePtr);

            return mlir::success();
        }

        return mlir::failure();
    }

    InterfaceInfo::TypePtr MLIRGenImpl::mlirGenInterfaceInfo(InterfaceDeclaration interfaceDeclarationAST, bool &declareInterface,
                                                const GenContext &genContext)
    {
        auto name = getNameWithArguments(interfaceDeclarationAST, genContext);
        return mlirGenInterfaceInfo(name, declareInterface, genContext);
    }

    InterfaceInfo::TypePtr MLIRGenImpl::mlirGenInterfaceInfo(const std::string &name, bool &declareInterface,
                                                const GenContext &genContext)
    {
        declareInterface = false;

        auto namePtr = StringRef(name).copy(stringAllocator);
        auto fullNamePtr = getFullNamespaceName(namePtr);

        InterfaceInfo::TypePtr newInterfacePtr;
        if (fullNameInterfacesMap.count(fullNamePtr))
        {
            newInterfacePtr = fullNameInterfacesMap.lookup(fullNamePtr);
            getInterfacesMap().insert({namePtr, newInterfacePtr});
            declareInterface = !newInterfacePtr->interfaceType;
        }
        else
        {
            // register class
            newInterfacePtr = std::make_shared<InterfaceInfo>();
            newInterfacePtr->name = namePtr;
            newInterfacePtr->fullName = fullNamePtr;
            newInterfacePtr->elementNamespace = currentNamespace;

            getInterfacesMap().insert({namePtr, newInterfacePtr});
            fullNameInterfacesMap.insert(fullNamePtr, newInterfacePtr);
            declareInterface = true;
        }

        if (declareInterface && mlir::succeeded(mlirGenInterfaceType(newInterfacePtr, genContext)))
        {
            newInterfacePtr->typeParamsWithArgs = genContext.typeParamsWithArgs;
        }

        return newInterfacePtr;
    }

    mlir::LogicalResult MLIRGenImpl::mlirGenInterfaceHeritageClauseExtends(InterfaceDeclaration interfaceDeclarationAST,
                                                              InterfaceInfo::TypePtr newInterfacePtr,
                                                              HeritageClause heritageClause, int &orderWeight, bool declareClass,
                                                              const GenContext &genContext)
    {
        if (heritageClause->token != SyntaxKind::ExtendsKeyword)
        {
            return mlir::success();
        }

        for (auto &extendsType : heritageClause->types)
        {
            auto result = mlirGen(extendsType, genContext);
            EXIT_IF_FAILED(result);
            auto ifaceType = V(result);
            auto success = false;
            mlir::TypeSwitch<mlir::Type>(ifaceType.getType())
                .template Case<mlir_ts::InterfaceType>([&](auto interfaceType) {
                    auto interfaceInfo = getInterfaceInfoByFullName(interfaceType.getName().getValue());
                    if (interfaceInfo)
                    {
                        newInterfacePtr->extends.push_back({-1, interfaceInfo});
                        success = true;
                    }
                })
                .template Case<mlir_ts::TupleType>([&](auto tupleType) {
                    llvm::SmallVector<mlir_ts::FieldInfo> destTupleFields;
                    if (mlir::succeeded(mth.getFields(tupleType, destTupleFields)))
                    {
                        orderWeight++;
                        success = true;
                        for (auto field : destTupleFields)
                            success &= mlir::succeeded(
                                mlirGenInterfaceAddFieldMember(newInterfacePtr, field.id, field.type, field.isConditional, orderWeight));
                    }
                })
                .Default([&](auto type) { llvm_unreachable("not implemented"); });

            if (!success)
            {
                return mlir::failure();
            }
        }

        return mlir::success();
    }

    mlir::LogicalResult MLIRGenImpl::mlirGen(InterfaceDeclaration interfaceDeclarationAST, const GenContext &genContext)
    {
        // do not proceed for Generic Interfaces for declaration
        if (interfaceDeclarationAST->typeParameters.size() > 0 && genContext.typeParamsWithArgs.size() == 0)
        {
            return registerGenericInterface(interfaceDeclarationAST, genContext);
        }

        auto declareInterface = false;
        auto newInterfacePtr = mlirGenInterfaceInfo(interfaceDeclarationAST, declareInterface, genContext);
        if (!newInterfacePtr)
        {
            return mlir::failure();
        }

        // do not process specialized interface second time;
        if (!declareInterface && interfaceDeclarationAST->typeParameters.size() > 0 &&
            genContext.typeParamsWithArgs.size() > 0)
        {
            return mlir::success();
        }

        auto location = loc(interfaceDeclarationAST);

        auto ifaceGenContext = GenContext(genContext);
        ifaceGenContext.thisType = newInterfacePtr->interfaceType;

        auto orderWeight = 0;
        for (auto &heritageClause : interfaceDeclarationAST->heritageClauses)
        {
            if (mlir::failed(mlirGenInterfaceHeritageClauseExtends(interfaceDeclarationAST, newInterfacePtr,
                                                                   heritageClause, orderWeight, declareInterface, genContext)))
            {
                return mlir::failure();
            }
        }

        newInterfacePtr->recalcOffsets();

        // clear all flags
        for (auto &interfaceMember : interfaceDeclarationAST->members)
        {
            interfaceMember->processed = false;
        }

        // add methods when we have classType
        auto notResolved = 0;
        do
        {
            auto lastTimeNotResolved = notResolved;
            notResolved = 0;

            for (auto &interfaceMember : interfaceDeclarationAST->members)
            {
                orderWeight++;
                if (mlir::failed(mlirGenInterfaceMethodMember(
                        interfaceDeclarationAST, newInterfacePtr, interfaceMember, orderWeight, declareInterface, ifaceGenContext)))
                {
                    notResolved++;
                }
            }

            // repeat if not all resolved
            if (lastTimeNotResolved > 0 && lastTimeNotResolved == notResolved)
            {
                // interface can depend on other interface declarations
                // theModule.emitError("can't resolve dependencies in intrerface: ") << newInterfacePtr->name;
                return mlir::failure();
            }

        } while (notResolved > 0);

        // fix up vtable slot numbers to the canonical methods-then-fields order now that
        // all members are known - see assignCanonicalVirtualIndexes() for why this can't
        // be left to getVirtualTable()'s per-cast assignment alone.
        newInterfacePtr->assignCanonicalVirtualIndexes();

        // add to export if any
        if (auto hasExport = getExportModifier(interfaceDeclarationAST))
        {
            addInterfaceDeclarationToExport(newInterfacePtr);
        }

        return mlir::success();
    }

    mlir::LogicalResult MLIRGenImpl::mlirGenInterfaceAddFieldMember(InterfaceInfo::TypePtr newInterfacePtr, mlir::Attribute fieldId, mlir::Type typeIn, bool isConditional, int orderWeight, bool declareInterface)
    {
        auto &fieldInfos = newInterfacePtr->fields;
        auto type = typeIn;

        // fix type for fields with FuncType
        if (auto hybridFuncType = dyn_cast<mlir_ts::HybridFunctionType>(type))
        {

            auto funcType = getFunctionType(hybridFuncType.getInputs(), hybridFuncType.getResults(), hybridFuncType.isVarArg());
            type = mth.getFunctionTypeAddingFirstArgType(funcType, getOpaqueType());
        }
        else if (auto funcType = dyn_cast<mlir_ts::FunctionType>(type))
        {

            type = mth.getFunctionTypeAddingFirstArgType(funcType, getOpaqueType());
        }

        if (mth.isNoneType(type))
        {
            LLVM_DEBUG(dbgs() << "\n!! interface field: " << fieldId << " FAILED\n");
            return mlir::failure();
        }

        auto fieldIndex = newInterfacePtr->getFieldIndex(fieldId);
        if (fieldIndex == -1)
        {
            fieldInfos.push_back({fieldId, type, isConditional, orderWeight, newInterfacePtr->getNextVTableMemberIndex()});
        }
        else
        {
            // update
            fieldInfos[fieldIndex].type = type;
            fieldInfos[fieldIndex].isConditional = isConditional;
        }

        return mlir::success();
    }

    mlir::LogicalResult MLIRGenImpl::mlirGenInterfaceMethodMember(InterfaceDeclaration interfaceDeclarationAST,
                                                     InterfaceInfo::TypePtr newInterfacePtr,
                                                     TypeElement interfaceMember, int orderWeight, bool declareInterface,
                                                     const GenContext &genContext)
    {
        if (interfaceMember->processed)
        {
            return mlir::success();
        }

        auto location = loc(interfaceMember);

        auto &methodInfos = newInterfacePtr->methods;

        mlir::Value initValue;
        mlir::Attribute fieldId;
        mlir::Type type;
        StringRef memberNamePtr;

        MLIRCodeLogic mcl(builder, compileOptions);

        SyntaxKind kind = interfaceMember;
        if (kind == SyntaxKind::PropertySignature)
        {
            // property declaration
            auto propertySignature = interfaceMember.as<PropertySignature>();
            auto isConditional = !!propertySignature->questionToken;

            fieldId = TupleFieldName(propertySignature->name, genContext);

            auto [type, init, typeProvided] = getTypeAndInit(propertySignature, genContext);
            if (!type)
            {
                return mlir::failure();
            }

            if (mlir::failed(mlirGenInterfaceAddFieldMember(newInterfacePtr, fieldId, type, isConditional, orderWeight, declareInterface)))
            {
                return mlir::failure();
            }
        }
        else if (kind == SyntaxKind::MethodSignature 
                || kind == SyntaxKind::ConstructSignature || kind == SyntaxKind::CallSignature 
                || kind == SyntaxKind::GetAccessor || kind == SyntaxKind::SetAccessor)
        {
            auto methodSignature = interfaceMember.as<MethodSignature>();
            auto isConditional = !!methodSignature->questionToken;

            newInterfacePtr->hasNew |= kind == SyntaxKind::ConstructSignature;
            // we need this code to add "THIS" param to declaration
            interfaceMember->parent = interfaceDeclarationAST;

            std::string methodName;
            std::string propertyName;
            mlir_ts::FunctionType funcType;
            if (mlir::failed(getInterfaceMethodNameAndType(location, newInterfacePtr->interfaceType, methodSignature, 
                    methodName, propertyName, funcType, genContext)))
            {
                return mlir::failure();
            }

            if (mlir::failed(addInterfaceMethod(location, newInterfacePtr, methodInfos, 
                methodName, funcType, isConditional, orderWeight, declarationMode, genContext))) 
            {
                return mlir::failure();
            }

            // add info about property
            if (kind == SyntaxKind::GetAccessor || kind == SyntaxKind::SetAccessor)
            {
                auto accessor = newInterfacePtr->findAccessor(propertyName);
                
                auto &accessors = newInterfacePtr->accessors;
                if (accessor == nullptr)
                {
                    if (kind == SyntaxKind::GetAccessor)
                    {
                        accessors.push_back({funcType.getResult(0), propertyName, methodName, ""});
                    }
                    else
                    {
                        accessors.push_back({funcType.getInputs().back(), propertyName, "", methodName});
                    }
                }
                else
                {
                    if (kind == SyntaxKind::GetAccessor)
                    {
                        accessor->getMethod = methodName;
                    }
                    else
                    {
                        accessor->setMethod = methodName;
                    }                    
                }
            }

            methodSignature->processed = true;
        }
        else if (kind == SyntaxKind::IndexSignature)
        {
            auto methodSignature = interfaceMember.as<MethodSignature>();
            // we need this code to add "THIS" param to declaration
            interfaceMember->parent = interfaceDeclarationAST;

            std::string methodName;
            std::string propertyName;
            mlir_ts::FunctionType funcType;
            if (mlir::failed(getInterfaceMethodNameAndType(
                location, newInterfacePtr->interfaceType, methodSignature, methodName, propertyName, funcType, genContext)))
            {
                return mlir::failure();
            }

            // add get method
            if (mlir::failed(addInterfaceMethod(location, newInterfacePtr, methodInfos, 
                INDEX_ACCESS_GET_FIELD_NAME, mth.getIndexGetFunctionType(funcType), true, orderWeight, declarationMode, genContext))) 
            {
                return mlir::failure();
            }

            if (mlir::failed(addInterfaceMethod(location, newInterfacePtr, methodInfos, 
                INDEX_ACCESS_SET_FIELD_NAME, mth.getIndexSetFunctionType(funcType), true, orderWeight, declarationMode, genContext))) 
            {
                return mlir::failure();
            }

            auto found = llvm::find_if(newInterfacePtr->indexes, [&] (auto indexInfo) {
                return indexInfo.indexSignature == funcType;
            });        

            if (found == newInterfacePtr->indexes.end())
            {
                newInterfacePtr->indexes.push_back({funcType, INDEX_ACCESS_GET_FIELD_NAME, INDEX_ACCESS_SET_FIELD_NAME});
            }   

            methodSignature->processed = true;            
        }
        else
        {
            llvm_unreachable("not implemented");
        }

        return mlir::success();
    }

    std::tuple<std::string, bool> MLIRGenImpl::getNameForMethod(SignatureDeclarationBase methodSignature, const GenContext &genContext)
    {
        auto [attr, result] = getNameFromComputedPropertyName(methodSignature->name, genContext);
        if (mlir::failed(result))
        {
            return {"", false};
        }

        if (attr)
        {
            if (auto strAttr = dyn_cast<mlir::StringAttr>(attr))
            {
                return {strAttr.getValue().str(), true};
            }
            else
            {
                llvm_unreachable("not implemented");
            }
        }

        return {MLIRHelper::getName(methodSignature->name), true};
    }

    mlir::LogicalResult MLIRGenImpl::getMethodNameOrPropertyName(bool isStaticClass, SignatureDeclarationBase methodSignature, std::string &methodName,
                                                    std::string &propertyName, const GenContext &genContext)
    {
        SyntaxKind kind = methodSignature;
        if (kind == SyntaxKind::Constructor)
        {
            auto isStatic = isStaticClass || hasModifier(methodSignature, SyntaxKind::StaticKeyword);
            if (isStatic)
            {
                methodName = std::string(STATIC_CONSTRUCTOR_NAME);
            }
            else
            {
                methodName = std::string(CONSTRUCTOR_NAME);
            }
        }
        else if (kind == SyntaxKind::ConstructSignature)
        {
            methodName = std::string(NEW_CTOR_METHOD_NAME);
        }
        else if (kind == SyntaxKind::IndexSignature)
        {
            methodName = std::string(INDEX_ACCESS_FIELD_NAME);
        }
        else if (kind == SyntaxKind::CallSignature)
        {
            methodName = std::string(CALL_FIELD_NAME);
        }
        else if (kind == SyntaxKind::GetAccessor)
        {
            auto [name, result] = getNameForMethod(methodSignature, genContext);
            if (!result)
            {
                return mlir::failure();
            }

            propertyName = name;
            methodName = std::string("get_") + propertyName;
        }
        else if (kind == SyntaxKind::SetAccessor)
        {
            auto [name, result] = getNameForMethod(methodSignature, genContext);
            if (!result)
            {
                return mlir::failure();
            }

            propertyName = name;
            methodName = std::string("set_") + propertyName;
        }
        else
        {
            auto [name, result] = getNameForMethod(methodSignature, genContext);
            if (!result)
            {
                return mlir::failure();
            }            

            methodName = name;
        }

        return mlir::success();
    }

} // namespace mlirgen
} // namespace typescript
