// Property/element access and call/new code generation methods of MLIRGenImpl (see MLIRGenImpl.h).

#include "MLIRGenImpl.h"

namespace typescript
{
namespace mlirgen
{

    ValueOrLogicalResult MLIRGenImpl::mlirGenCallThisMethod(mlir::Location location, mlir::Value thisValue, StringRef methodName,
                                               NodeArray<TypeNode> typeArguments, NodeArray<Expression> arguments,
                                               const GenContext &genContext)
    {
        // to remove temp var after call
        SymbolTableScopeT varScope(symbolTable);

        auto varDecl = std::make_shared<VariableDeclarationDOM>(THIS_TEMPVAR_NAME, thisValue.getType(), location);
        DECLARE(varDecl, thisValue);

        NodeFactory nf(NodeFactoryFlags::None);

        auto thisToken = nf.createIdentifier(S(THIS_TEMPVAR_NAME));
        auto callLogic = nf.createCallExpression(
            nf.createPropertyAccessExpression(thisToken, nf.createIdentifier(stows(methodName.str()))), typeArguments,
            arguments);

        return mlirGen(callLogic, genContext);
    }

    ValueOrLogicalResult MLIRGenImpl::mlirGenPropertyAccessExpression(mlir::Location location, mlir::Value objectValue,
                                                         mlir::StringRef name, const GenContext &genContext)
    {
        assert(objectValue);
        MLIRPropertyAccessCodeLogic cl(compileOptions, builder, location, objectValue, name);
        if (!genContext.dummyRun && !genContext.allowPartialResolve)
        {
            cl.setBoundRefMaterializedCache(&boundRefMaterializedCache);
        }
        return mlirGenPropertyAccessExpressionLogic(location, objectValue, false, cl, genContext);
    }

    ValueOrLogicalResult MLIRGenImpl::mlirGenPropertyAccessExpression(mlir::Location location, mlir::Value objectValue,
                                                         mlir::StringRef name, bool isConditional,
                                                         const GenContext &genContext)
    {
        assert(objectValue);
        MLIRPropertyAccessCodeLogic cl(compileOptions, builder, location, objectValue, name);
        if (!genContext.dummyRun && !genContext.allowPartialResolve)
        {
            cl.setBoundRefMaterializedCache(&boundRefMaterializedCache);
        }
        return mlirGenPropertyAccessExpressionLogic(location, objectValue, isConditional, cl, genContext);
    }

    ValueOrLogicalResult MLIRGenImpl::mlirGenPropertyAccessExpression(mlir::Location location, mlir::Value objectValue,
                                                         mlir::Attribute id, const GenContext &genContext)
    {
        MLIRPropertyAccessCodeLogic cl(compileOptions, builder, location, objectValue, id);
        if (!genContext.dummyRun && !genContext.allowPartialResolve)
        {
            cl.setBoundRefMaterializedCache(&boundRefMaterializedCache);
        }
        return mlirGenPropertyAccessExpressionLogic(location, objectValue, false, cl, genContext);
    }

    ValueOrLogicalResult MLIRGenImpl::mlirGenPropertyAccessExpression(mlir::Location location, mlir::Value objectValue,
                                                         mlir::Attribute id, bool isConditional,
                                                         const GenContext &genContext)
    {
        MLIRPropertyAccessCodeLogic cl(compileOptions, builder, location, objectValue, id);
        if (!genContext.dummyRun && !genContext.allowPartialResolve)
        {
            cl.setBoundRefMaterializedCache(&boundRefMaterializedCache);
        }
        return mlirGenPropertyAccessExpressionLogic(location, objectValue, isConditional, cl, genContext);
    }

    ValueOrLogicalResult MLIRGenImpl::mlirGenPropertyAccessExpression(mlir::Location location, mlir::Value objectValue,
                                                         mlir::Attribute id, bool isConditional,
                                                         mlir::Value argument/*for index access*/,
                                                         const GenContext &genContext)
    {
        MLIRPropertyAccessCodeLogic cl(compileOptions, builder, location, objectValue, id, argument);
        if (!genContext.dummyRun && !genContext.allowPartialResolve)
        {
            cl.setBoundRefMaterializedCache(&boundRefMaterializedCache);
        }
        return mlirGenPropertyAccessExpressionLogic(location, objectValue, isConditional, cl, genContext);
    }

    ValueOrLogicalResult MLIRGenImpl::mlirGenPropertyAccessExpressionLogic(mlir::Location location, mlir::Value objectValue,
                                                              bool isConditional, MLIRPropertyAccessCodeLogic &cl,
                                                              const GenContext &genContext)
    {
        if (isConditional && MLIRTypeCore::isNullableOrOptionalType(objectValue.getType()))
        {
            // TODO: replace with one op "Optional <has_value>, <value>"
            CAST_A(condValue, location, getBooleanType(), objectValue, genContext);

            auto propType = evaluateProperty(location, objectValue, cl.getName().str(), genContext);
            if (!propType)
            {
                emitError(location, "Can't resolve property '") << cl.getName() << "' of type " << to_print(objectValue.getType());
                return mlir::failure();
            }

            auto ifOp = builder.create<mlir_ts::IfOp>(location, getOptionalType(propType), condValue, true);

            builder.setInsertionPointToStart(&ifOp.getThenRegion().front());

            // value if true
            auto result = mlirGenPropertyAccessExpressionBaseLogic(location, objectValue, cl, genContext);
            auto value = V(result);

            // special case: conditional extension function <xxx>?.<ext>();
            if (auto createExtentionFunction = value.getDefiningOp<mlir_ts::CreateExtensionFunctionOp>())
            {
                // we need to convert into CreateBoundFunction, so it should be reference type for this, do I need to case value type into reference type?
                value = createBoundMethodFromExtensionMethod(location, createExtentionFunction);
                ifOp.getResults().front().setType(getOptionalType(value.getType()));
            }

            auto optValue = isa<mlir_ts::OptionalType>(value.getType())
                    ? value : builder.create<mlir_ts::OptionalValueOp>(location, getOptionalType(value.getType()), value);
            builder.create<mlir_ts::ResultOp>(location, mlir::ValueRange{optValue});

            // else
            builder.setInsertionPointToStart(&ifOp.getElseRegion().front());

            auto optUndefValue = builder.create<mlir_ts::OptionalUndefOp>(location, getOptionalType(value.getType()));
            builder.create<mlir_ts::ResultOp>(location, mlir::ValueRange{optUndefValue});

            builder.setInsertionPointAfter(ifOp);

            return ifOp.getResults().front();
        }
        else
        {
            return mlirGenPropertyAccessExpressionBaseLogic(location, objectValue, cl, genContext);
        }
    }

    ValueOrLogicalResult MLIRGenImpl::mlirGenPropertyAccessExpressionBaseLogic(mlir::Location location, mlir::Value objectValue,
                                                                  MLIRPropertyAccessCodeLogic &cl,
                                                                  const GenContext &genContext)
    {
        auto name = cl.getName();
        auto argument = cl.getArgument();
        auto actualType = objectValue.getType();

        LLVM_DEBUG(llvm::dbgs() << "\n\tResolving property '" << name << "' of type " << objectValue.getType(););

        // load reference if needed, except TupleTuple, ConstTupleType
        if (auto refType = dyn_cast<mlir_ts::RefType>(actualType))
        {
            auto elementType = refType.getElementType();
            if (!isa<mlir_ts::TupleType>(elementType) && !isa<mlir_ts::ConstTupleType>(elementType))
            {
                objectValue = builder.create<mlir_ts::LoadOp>(location, elementType, objectValue);
                actualType = objectValue.getType();
            }
        }

        // collapse union type
        if (auto unionType = dyn_cast<mlir_ts::UnionType>(actualType)) 
        {
            mlir::Type baseType;
            if (!mth.isUnionTypeNeedsTag(location, unionType, baseType))
            {
                LLVM_DEBUG(llvm::dbgs() << "\n!! mlirGenPropertyAccessExpressionBaseLogic: union type " << baseType << "\n";);
                actualType = baseType;
            }
        }        

        // class member access
        auto classAccessWithObject = [&](mlir_ts::ClassType classType, mlir::Value objectValue) {

            LLVM_DEBUG(llvm::dbgs() << "\n\t...field: \t" << cl.getName(););
            auto accessingFromLevel = detectAccessLevel(classType, genContext);
            LLVM_DEBUG(llvm::dbgs() << "\n\t = Accessing from level '" << accessingFromLevel << "'\n\n";);

            if (auto value = cl.Class(classType, accessingFromLevel))
            {
                return value;
            }

            return ClassMembersAccess(location, objectValue, classType.getName().getValue(), name, 
                false, argument, accessingFromLevel, genContext);
        };

        auto classAccess = [&](mlir_ts::ClassType classType) {
            return classAccessWithObject(classType, objectValue);
        };

        auto castFn = [this](mlir::Location location, mlir::Type type, mlir::Value value, const GenContext &genContext, bool disableStrictNullCheck) { return cast(location, type, value, genContext, disableStrictNullCheck); };

        mlir::Value value = 
            mlir::TypeSwitch<mlir::Type, mlir::Value>(actualType)
                .Case<mlir_ts::EnumType>([&](auto enumType) { return cl.Enum(enumType); })
                .Case<mlir_ts::ConstTupleType>([&](auto constTupleType) { return cl.Tuple(constTupleType); })
                .Case<mlir_ts::TupleType>([&](auto tupleType) { return cl.Tuple(tupleType); })
                .Case<mlir_ts::StringType>([&](auto stringType) { 
                    if (auto value = cl.String(stringType))
                    {
                        return value;
                    }

                    return mlir::Value();
                })
                .Case<mlir_ts::ConstArrayType>([&](auto arrayType) { 
#ifdef ARRAY_TYPE_AS_ARRAY_CLASS                    
                    if (auto genericClassTypeInfo = getGenericClassInfoByFullName("Array"))
                    {
                        auto classType = genericClassTypeInfo->classType;
                        SmallVector<mlir::Type> typeArg{arrayType.getElementType()};
                        auto [result, specType] = instantiateSpecializedClassType(location, classType,
                                typeArg, genContext, true);
                        auto accessFailed = false;
                        if (mlir::succeeded(result))
                        {
                            auto arrayNonConst = cast(location, mlir_ts::ArrayType::get(arrayType.getElementType()), objectValue, genContext);
                            if (arrayNonConst.failed())
                            {
                                return mlir::Value();
                            }

                            if (auto value = classAccessWithObject(mlir::cast<mlir_ts::ClassType>(specType), arrayNonConst))
                            {
                                return value;
                            }

                            accessFailed = true;
                        }

                        if (mlir::failed(result) && !accessFailed)
                        {
                            genContext.stop();
                            return mlir::Value();
                        }

                        genContext.postponedMessages->clear();
                    }
#endif

                    // find Array type
                    // TODO: should I mix use of Array and Array<T>?
                    // if (auto classInfo = getClassInfoByFullName("Array"))
                    // {
                    //     return classAccess(classInfo->classType);
                    // }

                    if (auto value = cl.Array(arrayType, compileOptions, castFn, genContext))
                    {
                        return value;
                    }

                    return mlir::Value();
                })
                .Case<mlir_ts::ArrayType>([&](auto arrayType) { 
#ifdef ARRAY_TYPE_AS_ARRAY_CLASS                    
                    if (auto genericClassTypeInfo = getGenericClassInfoByFullName("Array"))
                    {
                        auto classType = genericClassTypeInfo->classType;
                        SmallVector<mlir::Type> typeArg{arrayType.getElementType()};
                        auto [result, specType] = instantiateSpecializedClassType(location, classType,
                                typeArg, genContext, true);
                        auto accessFailed = false;
                        if (mlir::succeeded(result))
                        {
                            if (auto value = classAccess(mlir::cast<mlir_ts::ClassType>(specType)))
                            {
                                return value;
                            }

                            accessFailed = true;
                        }

                        if (mlir::failed(result) && !accessFailed)
                        {
                            genContext.stop();
                            return mlir::Value();
                        }

                        genContext.postponedMessages->clear();
                    }
#endif
                    // find Array type
                    // TODO: should I mix use of Array and Array<T>?
                    // if (auto classInfo = getClassInfoByFullName("Array"))
                    // {
                    //     return classAccess(classInfo->classType);
                    // }

                    if (auto value = cl.Array(arrayType, compileOptions, castFn, genContext))
                    {
                        return value;
                    }

                    return mlir::Value();
                })
                .Case<mlir_ts::RefType>([&](auto refType) { return cl.Ref(refType); })
                .Case<mlir_ts::ObjectType>([&](auto objectType) { 
                    if (auto value = cl.Object(objectType))
                    {
                        return value;
                    }

                    return mlir::Value();                    
                })
                .Case<mlir_ts::ObjectStorageType>([&](auto objectStorageType) { 
                    if (auto value = cl.RefLogic(objectStorageType))
                    {
                        return value;
                    }

                    return mlir::Value();                    
                })
                .Case<mlir_ts::SymbolType>([&](auto symbolType) { return cl.Symbol(symbolType); })
                .Case<mlir_ts::NamespaceType>([&](auto namespaceType) {
                    auto namespaceInfo = getNamespaceByFullName(namespaceType.getName().getValue());
                    assert(namespaceInfo);

                    MLIRNamespaceGuard ng(currentNamespace);
                    currentNamespace = namespaceInfo;

                    return mlirGen(location, name, genContext);
                })
                .Case<mlir_ts::ClassStorageType>([&](auto classStorageType) {
                    LLVM_DEBUG(llvm::dbgs() << "\n\t...field: \t" << cl.getName(););
                    auto accessingFromLevel = detectAccessLevel(classStorageType, genContext);
                    LLVM_DEBUG(llvm::dbgs() << "\n\t = Accessing from level '" << accessingFromLevel << "'\n\n";);

                    if (auto value = cl.TupleNoError(classStorageType, accessingFromLevel))
                    {
                        return value;
                    }

                    return ClassMembersAccess(location, objectValue, 
                        classStorageType.getName().getValue(), name, true, argument, accessingFromLevel, genContext);
                })
                .Case<mlir_ts::ClassType>(classAccess)
                .Case<mlir_ts::InterfaceType>([&](auto interfaceType) {
                    return InterfaceMembers(
                        location, objectValue, interfaceType.getName().getValue(), cl.getAttribute(), 
                        argument, genContext);
                })
                .Case<mlir_ts::OptionalType>([&](auto optionalType) {
                    // this is needed for conditional access to properties
                    auto elementType = optionalType.getElementType();
                    auto loadedValue = builder.create<mlir_ts::ValueOp>(location, elementType, objectValue);
                    return mlirGenPropertyAccessExpression(location, loadedValue, name, false, genContext);                
                })
                .Case<mlir_ts::UnionType>([&](auto unionType) {
                    // TODO: when access of property in union is finished use it instead of using first type
                    // all union types must have the same property
                    // 1) cast to first type
                    auto frontType = mth.getFirstNonNullUnionType(unionType);
                    //auto casted = cast(location, frontType, objectValue, genContext);
                    auto casted = builder.create<mlir_ts::GetValueFromUnionOp>(location, frontType, objectValue);

                    return mlirGenPropertyAccessExpression(location, casted, name, false, genContext);
                })
                .Case<mlir_ts::LiteralType>([&](auto literalType) {
                    auto elementType = literalType.getElementType();
                    auto castedValue = builder.create<mlir_ts::CastOp>(location, elementType, objectValue);
                    return mlirGenPropertyAccessExpression(location, castedValue, name, false, genContext);
                })
                .Default([&](auto type) {
                    LLVM_DEBUG(llvm::dbgs() << "\n\tCan't resolve property '" << name << "' of type " << objectValue.getType(););
                    return mlir::Value();
                });

        // extention logic: <obj>.<functionName>(this)
        if (!value)
        {
            if (auto funcRef = extensionFunction(location, objectValue, name, genContext))
            {
                return funcRef;
            }
        }

        if (!value)
        {
            // During a speculative discovery/dummy run (e.g. inferring an enclosing
            // function's return type, which recurses into an object literal's method
            // bodies to guess ITS return type too), a sibling method's prototype may
            // not be registered into the literal's (mutable) object-storage type yet -
            // `this.siblingMethod(...)` used in an expression (not a bare statement)
            // then fails to resolve here even though it will resolve fine once real
            // compilation runs with all prototypes registered. Don't hard-fail the
            // whole discovery run over that; let the caller treat this as "unknown for
            // now" (same idiom as mlirGenCallExpression's `!result.value &&
            // genContext.allowPartialResolve` case above).
            if (genContext.dummyRun || genContext.allowPartialResolve)
            {
                return mlir::success();
            }

            emitError(location, "Can't resolve property '") << name << "' of type " << to_print(objectValue.getType());
            return mlir::failure();
        }

        return value;
    }

    mlir::Value MLIRGenImpl::extensionFunctionLogic(mlir::Location location, mlir::Value funcRef, mlir::Value thisValue, StringRef name,
                                  const GenContext &genContext)
    {
        if (!mth.isAnyFunctionType(funcRef.getType()))
        {
            return mlir::Value();
        }

        LLVM_DEBUG(llvm::dbgs() << "!! found extension by name for type: " << thisValue.getType()
                                << " function: " << name << ", value: " << funcRef << "\n";);

        auto thisTypeFromFunc = mth.getFirstParamFromFuncRef(funcRef.getType());

        LLVM_DEBUG(llvm::dbgs() << "!! this type of function is : " << thisTypeFromFunc << "\n";);

        if (auto symbolOp = funcRef.getDefiningOp<mlir_ts::SymbolRefOp>())
        {
            // if (!isa<mlir_ts::GenericType>(symbolOp.getType()))
            if (!symbolOp->hasAttrOfType<mlir::BoolAttr>(GENERIC_ATTR_NAME))
            {
                auto funcType = mlir::cast<mlir_ts::FunctionType>(funcRef.getType());
                if (thisTypeFromFunc == thisValue.getType())
                {
                    // return funcRef;
                    auto thisRef = thisValue;
                    auto extensFuncVal = builder.create<mlir_ts::CreateExtensionFunctionOp>(
                        location, getExtensionFunctionType(funcType), thisRef, funcRef);
                    return extensFuncVal;
                }
            }
            else
            {
                // TODO: add checking constraint
                auto funcName = symbolOp.getIdentifierAttr().getValue();
                auto functionGenericTypeInfo = getGenericFunctionInfoByFullName(funcName);
                auto first = functionGenericTypeInfo->typeParams.front();
                if (first->hasConstraint())
                {
                    if (auto constraintType = getType(first->getConstraint(), genContext))
                    {
                        llvm::StringMap<std::pair<ts::TypeParameterDOM::TypePtr,mlir::Type>> pairs{};
                        auto extendsResult = mth.extendsType(location, thisValue.getType(), constraintType, pairs);
                        if (extendsResult == ExtendsResult::False || extendsResult == ExtendsResult::Never)
                        {
                            // failed due to generic type constraints
                            return mlir::Value();
                        }
                    }                    
                }

                // TODO: finish it
                // it is generic function
                StringMap<mlir::Type> inferredTypes;
                inferType(location, thisTypeFromFunc, thisValue.getType(), inferredTypes, genContext);
                if (inferredTypes.size() > 0)
                {
                    // we found needed function
                    // return funcRef;
                    auto thisRef = thisValue;

                    LLVM_DEBUG(llvm::dbgs() << "\n!! recreate ExtensionFunctionOp (generic interface): '" << name << "'\n this ref: '" << thisRef << "'\n func ref: '" << funcRef
                    << "'\n";);

                    auto funcType = mlir::cast<mlir_ts::FunctionType>(funcRef.getType());
                    auto extensFuncVal = builder.create<mlir_ts::CreateExtensionFunctionOp>(
                        location, getExtensionFunctionType(funcType), thisRef, funcRef);
                    return extensFuncVal;                        
                }
            }
        }

        return mlir::Value();
    }

    mlir::Value MLIRGenImpl::ClassMembersAccess(mlir::Location location, mlir::Value thisValue, mlir::StringRef classFullName,
                             mlir::StringRef name, bool baseClass, mlir::Value argument, mlir_ts::AccessLevel accessingFromLevel, const GenContext &genContext)
    {
        auto classInfo = getClassInfoByFullName(classFullName);
        if (!classInfo)
        {
            auto genericClassInfo = getGenericClassInfoByFullName(classFullName);
            if (genericClassInfo)
            {
                // we can't discover anything in generic class
                return mlir::Value();
            }

            emitError(location, "Class can't be found ") << classFullName;
            return mlir::Value();
        }

        // static field access
        auto value = ClassMembersAccess(location, thisValue, classInfo, name, baseClass, argument, accessingFromLevel, genContext);
        if (!value)
        {
            emitError(location, "Class member '") << name << "' can't be found";
        }

        return value;
    }

    mlir::Value MLIRGenImpl::ClassGenericMethodAccess(ClassInfo::TypePtr classInfo, 
            mlir::Location location, mlir::Value thisValue, int genericMethodIndex, 
            bool isSuperClass, mlir_ts::AccessLevel accessingFromLevel, const GenContext &genContext)
    {
        auto genericMethodInfo = classInfo->staticGenericMethods[genericMethodIndex];
        if (accessingFromLevel < genericMethodInfo.accessLevel) {
            emitError(location, "Class member '") << genericMethodInfo.name << "' is not accessable";
            return mlir::Value();
        }

        auto paramsArray = genericMethodInfo.funcProto->getParams();
        auto explicitThis = paramsArray.size() > 0 && paramsArray.front()->getName() == THIS_NAME;
        if (genericMethodInfo.isStatic && !explicitThis)
        {
            auto funcSymbolOp = builder.create<mlir_ts::SymbolRefOp>(
                location, genericMethodInfo.funcType,
                mlir::FlatSymbolRefAttr::get(builder.getContext(), genericMethodInfo.funcProto->getName()));
            funcSymbolOp->setAttr(GENERIC_ATTR_NAME, mlir::BoolAttr::get(builder.getContext(), true));
            return funcSymbolOp;
        }
        else
        {
            auto effectiveThisValue = getThisRefOfClass(location, classInfo->classType, thisValue, isSuperClass, genContext);
            auto effectiveFuncType = genericMethodInfo.funcProto->getFuncType();

            auto thisSymbOp = builder.create<mlir_ts::ThisSymbolRefOp>(
                location, getBoundFunctionType(effectiveFuncType), effectiveThisValue,
                mlir::FlatSymbolRefAttr::get(builder.getContext(), genericMethodInfo.funcProto->getName()));
            thisSymbOp->setAttr(GENERIC_ATTR_NAME, mlir::BoolAttr::get(builder.getContext(), true));
            return thisSymbOp;                
        }
    }

    mlir::Value MLIRGenImpl::ClassAccessorAccess(ClassInfo::TypePtr classInfo,
            mlir::Location location, mlir::Value thisValue, int accessorIndex,
            bool isSuperClass, mlir_ts::AccessLevel accessingFromLevel, const GenContext &genContext)
    {

        auto accessorInfo = classInfo->accessors[accessorIndex];

        // TODO: finish access check for get/set methods

        auto getFunc = accessorInfo.get;
        auto setFunc = accessorInfo.set;
        mlir::Type accessorResultType;
        if (getFunc)
        {
            auto funcType = getFunc.funcType;
            if (funcType.getNumResults() > 0)
            {
                accessorResultType = funcType.getResult(0);
            }
        }

        if (!accessorResultType && setFunc)
        {
            accessorResultType = setFunc.funcType.getInput(accessorInfo.isStatic ? 0 : 1);
        }

        if (!accessorResultType)
        {
            emitError(location) << "can't resolve type of property";
            return mlir::Value();
        }

        // remove funcs if access level is not high
        if (getFunc && accessingFromLevel < accessorInfo.getAccessLevel) {
            getFunc = {};
        }
        if (setFunc && accessingFromLevel < accessorInfo.setAccessLevel) {
            setFunc = {};
        }

        if (accessorInfo.isStatic)
        {
            auto accessorOp = builder.create<mlir_ts::AccessorOp>(
                location, accessorResultType,
                getFunc ? mlir::FlatSymbolRefAttr::get(builder.getContext(), getFunc.name)
                            : mlir::FlatSymbolRefAttr{},
                setFunc ? mlir::FlatSymbolRefAttr::get(builder.getContext(), setFunc.name)
                            : mlir::FlatSymbolRefAttr{},
                mlir::Value());
            return accessorOp.getResult(0);
        }
        else
        {
            auto effectiveThisValue = getThisRefOfClass(location, classInfo->classType, thisValue, isSuperClass, genContext);

            if (classInfo->isDynamicImport)
            {
                // A plain FlatSymbolRefAttr (the path below) lowers to a call the static linker
                // must resolve - but a -shared importer never links against the exporter's .lib
                // (see cross-module-instanceof-investigation.md), so it needs the same runtime
                // resolution ClassMethodAccess already does for dynamic-import members: resolve
                // each accessor function to a VALUE (direct ref / dlsym-style global / inline
                // SearchForAddressOfSymbolOp) and drive the call through the "indirect"
                // (function-pointer) accessor op instead of the symbol-attr one.
                auto resolveAccessorFunc = [&](FunctionEntry funcEntry) -> mlir::Value {
                    if (!funcEntry)
                    {
                        return mlir::Value();
                    }

                    StringRef funcName = funcEntry.name;
                    auto effectiveFuncType = funcEntry.funcType;

                    if (auto funcOp = theModule.lookupSymbol<mlir_ts::FuncOp>(funcName))
                    {
                        if (!funcOp.getBody().empty())
                        {
                            return builder.create<mlir_ts::SymbolRefOp>(
                                location, effectiveFuncType, mlir::FlatSymbolRefAttr::get(builder.getContext(), funcName));
                        }
                    }

                    auto globalFuncVar = resolveFullNameIdentifier(location, funcName, false, genContext);
                    if (!globalFuncVar)
                    {
                        auto symbolNameValue = V(mlirGenStringValue(location, funcName.str(), true));
                        auto referenceToFuncOpaque = builder.create<mlir_ts::SearchForAddressOfSymbolOp>(
                            location, getOpaqueType(), symbolNameValue);
                        auto castResult = cast(location, effectiveFuncType, referenceToFuncOpaque, genContext);
                        if (castResult.failed_or_no_value())
                        {
                            emitError(location, "Class member '") << funcName << "' can't be resolved (dynamic import)";
                            return mlir::Value();
                        }

                        globalFuncVar = V(castResult);
                    }

                    return globalFuncVar;
                };

                auto getterValue = resolveAccessorFunc(getFunc);
                auto setterValue = resolveAccessorFunc(setFunc);
                if ((getFunc && !getterValue) || (setFunc && !setterValue))
                {
                    return mlir::Value();
                }

                auto opaqueThisType = mlir_ts::OpaqueType::get(builder.getContext());
                if (!getterValue)
                {
                    getterValue = builder.create<mlir_ts::UndefOp>(
                        location, mlir_ts::FunctionType::get(builder.getContext(), {opaqueThisType}, {accessorResultType}, false));
                }

                if (!setterValue)
                {
                    setterValue = builder.create<mlir_ts::UndefOp>(
                        location, mlir_ts::FunctionType::get(builder.getContext(), {opaqueThisType, accessorResultType}, {}, false));
                }

                auto thisIndirectAccessorOp = builder.create<mlir_ts::ThisIndirectAccessorOp>(
                    location, accessorResultType, effectiveThisValue, getterValue, setterValue, mlir::Value());
                return thisIndirectAccessorOp.getResult(0);
            }

            auto thisAccessorOp = builder.create<mlir_ts::ThisAccessorOp>(
                location, accessorResultType, effectiveThisValue,
                getFunc ? mlir::FlatSymbolRefAttr::get(builder.getContext(), getFunc.name)
                            : mlir::FlatSymbolRefAttr{},
                setFunc ? mlir::FlatSymbolRefAttr::get(builder.getContext(), setFunc.name)
                            : mlir::FlatSymbolRefAttr{},
                mlir::Value());
            return thisAccessorOp.getResult(0);
        }

    }

    mlir::Value MLIRGenImpl::ClassIndexAccess(ClassInfo::TypePtr classInfo, 
            mlir::Location location, mlir::Value thisValue, mlir::Value argument, mlir_ts::AccessLevel accessingFromLevel, const GenContext &genContext)
    {

        if (classInfo->indexes.size() == 0)
        {
            emitError(location) << "indexer is not declared";
            return mlir::Value();            
        }

        auto indexInfo = classInfo->indexes.front();
        auto getFunc = indexInfo.get;
        auto setFunc = indexInfo.set;

        if (!indexInfo.indexSignature || indexInfo.indexSignature.getNumResults() == 0)
        {
            emitError(location) << "can't resolve type of indexer";
            return mlir::Value();
        }

        // remove funcs if access level is not high
        if (getFunc && accessingFromLevel < indexInfo.getAccessLevel) {
            getFunc = {};
        }
        if (setFunc && accessingFromLevel < indexInfo.setAccessLevel) {
            setFunc = {};
        }

        auto indexResultType = indexInfo.indexSignature.getResult(0);
        auto argumentType = indexInfo.indexSignature.getInput(0);

        // sync index
        CAST_A(result, location, argumentType, argument, genContext);

        auto thisIndexAccessorOp = builder.create<mlir_ts::ThisIndexAccessorOp>(
            location, indexResultType, thisValue, V(result),
            getFunc ? mlir::FlatSymbolRefAttr::get(builder.getContext(), getFunc.name)
                        : mlir::FlatSymbolRefAttr{},
            setFunc ? mlir::FlatSymbolRefAttr::get(builder.getContext(), setFunc.name)
                        : mlir::FlatSymbolRefAttr{},
            mlir::Value());
        return thisIndexAccessorOp.getResult(0);
    }

    mlir::Value MLIRGenImpl::ClassBaseClassAccess(ClassInfo::TypePtr classInfo, ClassInfo::TypePtr baseClass, int index,
            mlir::Location location, mlir::Value thisValue, StringRef name, mlir::Value argument, mlir_ts::AccessLevel accessingFromLevel, const GenContext &genContext)
    {

        // first base is "super."
        if (index == 0 && name == SUPER_NAME)
        {
            auto result = mlirGenPropertyAccessExpression(location, thisValue, baseClass->fullName, genContext);
            auto value = V(result);
            return value;
        }

        auto value = ClassMembersAccess(location, thisValue, baseClass, name, true, argument, accessingFromLevel, genContext);
        if (value)
        {
            return value;
        }

        SmallVector<ClassInfo::TypePtr> fieldPath;
        if (classHasField(baseClass, name, fieldPath))
        {
            // load value from path
            auto currentObject = thisValue;
            for (auto &chain : fieldPath)
            {
                auto fieldValue =
                    mlirGenPropertyAccessExpression(location, currentObject, chain->fullName, genContext);
                if (!fieldValue)
                {
                    emitError(location) << "Can't resolve field/property/base '" << chain->fullName
                                        << "' of class '" << classInfo->fullName << "'\n";
                    return fieldValue;
                }

                assert(fieldValue);
                currentObject = fieldValue;
            }

            // last value
            auto result = mlirGenPropertyAccessExpression(location, currentObject, name, genContext);
            auto value = V(result);
            if (value)
            {
                return value;
            }
        }

        return mlir::Value();
    }    

    mlir::Value MLIRGenImpl::ClassMembersAccess(mlir::Location location, mlir::Value thisValue, ClassInfo::TypePtr classInfo,
                             mlir::StringRef name, bool isSuperClass, mlir::Value argument, mlir_ts::AccessLevel accessingFromLevel, const GenContext &genContext)
    {
        assert(classInfo);

        LLVM_DEBUG(llvm::dbgs() << "\n\t looking for member: " << name << " in class '" << classInfo->fullName << "'\n";);

        // indexer access
        if (name == INDEX_ACCESS_FIELD_NAME)
        {
            if (!classInfo->indexes.empty())
            {
                return ClassIndexAccess(classInfo, location, thisValue, argument, accessingFromLevel, genContext);
            }
        }

        auto staticFieldIndex = classInfo->getStaticFieldIndex(
            MLIRHelper::TupleFieldName(name, builder.getContext()));
        if (staticFieldIndex >= 0)
        {
            return ClassStaticFieldAccess(classInfo, location, thisValue, staticFieldIndex, accessingFromLevel, genContext);
        }

        // check method access
        auto methodIndex = classInfo->getMethodIndex(name);
        if (methodIndex >= 0)
        {
            return ClassMethodAccess(classInfo, location, thisValue, methodIndex, isSuperClass, accessingFromLevel, genContext);
        }

        // static generic methods
        auto genericMethodIndex = classInfo->getGenericMethodIndex(name);
        if (genericMethodIndex >= 0)
        {        
            return ClassGenericMethodAccess(classInfo, location, thisValue, genericMethodIndex, isSuperClass, accessingFromLevel, genContext);
        }        

        // check accessor
        auto accessorIndex = classInfo->getAccessorIndex(name);
        if (accessorIndex >= 0)
        {
            return ClassAccessorAccess(classInfo, location, thisValue, accessorIndex, isSuperClass, accessingFromLevel, genContext);
        }

        for (auto [index, baseClass] : enumerate(classInfo->baseClasses))
        {
            auto effectiveAccessingFromLevel = accessingFromLevel == mlir_ts::AccessLevel::Private 
                ? mlir_ts::AccessLevel::Protected : accessingFromLevel;
            auto value = ClassBaseClassAccess(classInfo, baseClass, index, location, 
                thisValue, name, argument, effectiveAccessingFromLevel, genContext);
            if (value)
            {
                return value;
            }
        }

        if (isSuperClass || genContext.allowPartialResolve)
        {
            return mlir::Value();
        }

        emitError(location) << "can't resolve property/field/base '" << name << "' of class '" << classInfo->fullName
                            << "'\n";

        return mlir::Value();
    }

    mlir::Value MLIRGenImpl::InterfaceMembers(mlir::Location location, mlir::Value interfaceValue, mlir::StringRef interfaceFullName,
                                 mlir::Attribute id, mlir::Value argument, const GenContext &genContext)
    {
        auto interfaceInfo = getInterfaceInfoByFullName(interfaceFullName);
        if (!interfaceInfo)
        {
            auto genericInterfaceInfo = getGenericInterfaceInfoByFullName(interfaceFullName);
            if (genericInterfaceInfo)
            {
                // we can't detect value of generic interface (we can only if it is specialization)
                emitError(location, "Interface can't be found ") << interfaceFullName;
                return mlir::Value();
            }

            return mlir::Value();
        }

        assert(interfaceInfo);

        // static field access
        auto value = InterfaceMembers(location, interfaceValue, interfaceInfo, id, argument, genContext);
        if (!value)
        {
            emitError(location, "Interface member ") << id << " can't be found in interface '" << interfaceInfo->name << "'";
        }

        return value;
    }

    mlir::Value MLIRGenImpl::InterfaceMembers(mlir::Location location, mlir::Value interfaceValue, InterfaceInfo::TypePtr interfaceInfo, 
        mlir::Attribute id, mlir::Value argument, const GenContext &genContext)
    {
        assert(interfaceInfo);

        // indexer access
        auto nameAttr = mlir::dyn_cast<mlir::StringAttr>(id);
        if (nameAttr && nameAttr.getValue() == INDEX_ACCESS_FIELD_NAME)
        {
            return InterfaceIndexAccess(interfaceInfo, location, interfaceValue, argument, genContext);
        }

        // check field access
        int fieldVTableOffset;
        if (auto fieldInfo = interfaceInfo->findField(id, fieldVTableOffset))
        {
            return InterfaceFieldAccess(location, interfaceValue, fieldInfo, fieldVTableOffset);
        }

        // check method access
        if (nameAttr)
        {
            int methodVTableOffset;
            if (auto methodInfo = interfaceInfo->findMethod(nameAttr.getValue(), methodVTableOffset))
            {
                return InterfaceMethodAccess(location, interfaceValue, methodInfo, methodVTableOffset);
            }

            if (auto accessorInfo = interfaceInfo->findAccessor(nameAttr.getValue()))
            {
                return InterfaceAccessorAccess(location, interfaceInfo, interfaceValue, accessorInfo, genContext);
            }

        }

        return mlir::Value();
    }

    ValueOrLogicalResult MLIRGenImpl::mlirGenElementAccess(mlir::Location location, mlir::Value expression, mlir::Value argumentExpression, bool isConditionalAccess, const GenContext &genContext)
    {
        auto arrayType = expression.getType();

        // collapse union type
        if (auto unionType = dyn_cast<mlir_ts::UnionType>(expression.getType())) 
        {
            mlir::Type baseType;
            if (!mth.isUnionTypeNeedsTag(location, unionType, baseType))
            {
                LLVM_DEBUG(llvm::dbgs() << "\n!! ElementAccessExpression: union type " << baseType << "\n";);
                arrayType = baseType;
            }
        }

        if (isa<mlir_ts::LiteralType>(arrayType))
        {
            arrayType = mth.stripLiteralType(arrayType);
            CAST(expression, location, arrayType, expression, genContext);
        }

        if (auto optType = dyn_cast<mlir_ts::OptionalType>(arrayType))
        {
            arrayType = optType.getElementType();
            // loading value from opt value
            expression = builder.create<mlir_ts::ValueOp>(location, arrayType, expression);
        }

        mlir::Type elementType;
        if (auto arrayTyped = dyn_cast<mlir_ts::ArrayType>(arrayType))
        {
            if (auto fieldName = argumentExpression.getDefiningOp<mlir_ts::ConstantOp>())
            {
                auto attr = fieldName.getValue();
                if (isa<mlir::StringAttr>(attr)) 
                {
                    return mlirGenPropertyAccessExpression(location, expression, attr, isConditionalAccess, genContext);
                }
            }

            elementType = arrayTyped.getElementType();
        }
        else if (auto vectorType = dyn_cast<mlir_ts::ConstArrayType>(arrayType))
        {
            if (auto fieldName = argumentExpression.getDefiningOp<mlir_ts::ConstantOp>())
            {
                auto attr = fieldName.getValue();
                if (isa<mlir::StringAttr>(attr)) 
                {
                    return mlirGenPropertyAccessExpression(location, expression, attr, isConditionalAccess, genContext);
                }
            }

            elementType = vectorType.getElementType();
        }
        else if (isa<mlir_ts::StringType>(arrayType))
        {
            elementType = getCharType();
        }
        else if (auto tupleType = dyn_cast<mlir_ts::TupleType>(arrayType))
        {
            return mlirGenElementAccessTuple(location, expression, argumentExpression, tupleType);
        }
        else if (auto constTupleType = dyn_cast<mlir_ts::ConstTupleType>(arrayType))
        {
            return mlirGenElementAccessTuple(location, expression, argumentExpression, constTupleType);
        }
        else if (auto objectType = dyn_cast<mlir_ts::ObjectType>(arrayType))
        {
            // boxed object literal (docs/object-literal-boxing-design.md): field access on
            // the pointer already works via mlirGenPropertyAccessExpression's ObjectType
            // case (cl.Object -> RefLogic), same recipe as the ClassType/InterfaceType
            // cases below -- only computed string-key access (`obj[Symbol.x]`,
            // `obj["field"]`) is supported, matching those cases.
            if (auto fieldName = argumentExpression.getDefiningOp<mlir_ts::ConstantOp>())
            {
                auto attr = fieldName.getValue();
                if (isa<mlir::StringAttr>(attr))
                {
                    return mlirGenPropertyAccessExpression(location, expression, attr, isConditionalAccess, genContext);
                }
            }

            llvm_unreachable("not implemented (ElementAccessExpression)");
        }
        else if (auto classType = dyn_cast<mlir_ts::ClassType>(arrayType))
        {
            if (auto fieldName = argumentExpression.getDefiningOp<mlir_ts::ConstantOp>())
            {
                auto attr = fieldName.getValue();
                if (isa<mlir::StringAttr>(attr))
                {
                    // TODO: implement '[string]' access here
                    return mlirGenPropertyAccessExpression(location, expression, attr, isConditionalAccess, genContext);
                }
            }

            // else access of index
            auto indexAccessor = builder.getStringAttr(INDEX_ACCESS_FIELD_NAME);
            return mlirGenPropertyAccessExpression(location, expression, indexAccessor, isConditionalAccess, argumentExpression, genContext);
        }
        else if (auto classStorageType = dyn_cast<mlir_ts::ClassStorageType>(arrayType))
        {
            // seems we are calling "super"
            if (auto fieldName = argumentExpression.getDefiningOp<mlir_ts::ConstantOp>())
            {
                auto attr = fieldName.getValue();
                return mlirGenPropertyAccessExpression(location, expression, attr, isConditionalAccess, genContext);
            }

            llvm_unreachable("not implemented (ElementAccessExpression)");
        }        
        else if (auto interfaceType = dyn_cast<mlir_ts::InterfaceType>(arrayType))
        {
            if (auto fieldName = argumentExpression.getDefiningOp<mlir_ts::ConstantOp>())
            {
                auto attr = fieldName.getValue();
                if (isa<mlir::StringAttr>(attr))
                {
                    return mlirGenPropertyAccessExpression(location, expression, attr, isConditionalAccess, genContext);
                }
            }

            // else access of index
            auto indexAccessor = builder.getStringAttr(INDEX_ACCESS_FIELD_NAME);
            return mlirGenPropertyAccessExpression(location, expression, indexAccessor, isConditionalAccess, argumentExpression, genContext);
        }        
        else if (auto enumType = dyn_cast<mlir_ts::EnumType>(arrayType))
        {
            if (auto fieldName = argumentExpression.getDefiningOp<mlir_ts::ConstantOp>())
            {
                auto attr = fieldName.getValue();
                return mlirGenPropertyAccessExpression(location, expression, attr, isConditionalAccess, genContext);
            }

            llvm_unreachable("not implemented (ElementAccessExpression)");
        }          
        else if (auto refType = dyn_cast<mlir_ts::RefType>(arrayType)) 
        {
            CAST_A(index, location, mth.getIndexType(), argumentExpression, genContext);

            LLVM_DEBUG(llvm::dbgs() << "\n!! ref type: " << refType << " index value: " << index << "\n";);

            auto elemRef = builder.create<mlir_ts::PointerOffsetRefOp>(
                location, refType, expression, index);            

            return V(elemRef);
        }
        else if (auto anyType = dyn_cast<mlir_ts::AnyType>(arrayType))
        {
            emitError(location, "not supported");
            return mlir::failure();
        }          
        else
        {
            LLVM_DEBUG(llvm::dbgs() << "\n!! ElementAccessExpression: " << arrayType
                                    << "\n";);

            emitError(location) << "access expression is not applicable to " << to_print(arrayType);
            return mlir::failure();
        }

        auto indexType = argumentExpression.getType();
        CAST(argumentExpression, location, mth.getStructIndexType(), argumentExpression, genContext);
  
        auto elemRef = builder.create<mlir_ts::ElementRefOp>(location, mlir_ts::RefType::get(elementType), expression,
                                                             argumentExpression);
        return V(builder.create<mlir_ts::LoadOp>(location, elementType, elemRef));
    }

    ValueOrLogicalResult MLIRGenImpl::mlirGenArrayReduce(mlir::Location location, SmallVector<mlir::Value, 4> &operands,
                                            const GenContext &genContext)
    {
        // info, we add "_" extra as scanner append "_" in front of "__";
        auto funcName = "___array_reduce";

        if (!existGenericFunctionMap(funcName))
        {
            auto src = S("function __array_reduce<T, R>(arr: T[], f: (s: R, v: T) => R, init: R) \
            {   \
                let r = init;   \
                for (const v of arr) r = f(r, v);   \
                return r;   \
            }");

            {
                MLIRLocationGuard vgLoc(overwriteLoc); 
                overwriteLoc = location;
                if (mlir::failed(parsePartialStatements(src)))
                {
                    assert(false);
                    return mlir::failure();
                }
            }
        }

        auto funcResult = resolveIdentifier(location, funcName, genContext);

        assert(funcResult);

        return mlirGenCallExpression(location, funcResult, {}, operands, genContext);
    }

    ValueOrLogicalResult MLIRGenImpl::mlirGenCallBuiltInFunction(
        mlir::Location location, mlir::Value actualFuncRefValue, NodeArray<TypeNode> typeArguments, 
        SmallVector<mlir::Value, 4> &operands, const GenContext &genContext)
    {
        // TODO: when you resolve names such as "print", "parseInt" should return names in mlirGen(Identifier)
        auto calleeName = actualFuncRefValue.getDefiningOp()->getAttrOfType<mlir::FlatSymbolRefAttr>(StringRef(IDENTIFIER_ATTR_NAME));
        auto functionName = calleeName.getValue();

        if (auto thisSymbolRefOp = actualFuncRefValue.getDefiningOp<mlir_ts::ThisSymbolRefOp>())
        {
            // do not remove it, it is needed for custom methods to be called correctly
            operands.insert(operands.begin(), thisSymbolRefOp.getThisVal());
        }

        // temp hack
        if (functionName == "__array_foreach")
        {
            mlirGenArrayForEach(location, operands, genContext);
            return mlir::success();
        }

        if (functionName == "__array_every")
        {
            return mlirGenArrayEvery(location, operands, genContext);
        }

        if (functionName == "__array_some")
        {
            return mlirGenArraySome(location, operands, genContext);
        }

        if (functionName == "__array_map")
        {
            return mlirGenArrayMap(location, operands, genContext);
        }

        if (functionName == "__array_filter")
        {
            return mlirGenArrayFilter(location, operands, genContext);
        }

        if (functionName == "__array_reduce")
        {
            return mlirGenArrayReduce(location, operands, genContext);
        }

        // resolve function           
        MLIRCustomMethods cm(builder, location, compileOptions);
        mlir::SmallVector<mlir::Type> typeArgs;
        for (auto typeArgNode : typeArguments)
        {
            auto typeArg = getType(typeArgNode, genContext);
            if (!typeArg)
            {
                return mlir::failure();
            }

            typeArgs.push_back(typeArg);
        }

        return cm.callMethod(
            functionName, 
            typeArgs,
            operands, 
            [this](mlir::Location location, mlir::Type type, mlir::Value value, const GenContext &genContext, bool disableStrictNullCheck) { return cast(location, type, value, genContext, disableStrictNullCheck); }, 
            genContext);
    }

    ValueOrLogicalResult MLIRGenImpl::mlirGenCallExpression(mlir::Location location, mlir::Value funcResult,
                                               NodeArray<TypeNode> typeArguments, SmallVector<mlir::Value, 4> &operands,
                                               const GenContext &genContext)
    {
        GenContext specGenContext(genContext);
        specGenContext.callOperands = operands;

        // get function ref.
        auto result = mlirGenSpecialized(location, funcResult, typeArguments, operands, specGenContext);
        EXIT_IF_FAILED(result)
        auto actualFuncRefValue = V(result);

        if (!result.value && genContext.allowPartialResolve)
        {
            return mlir::success();
        }

        // special case when TypePredicateType is used in generic function and failed constraints 
        if (auto symbolRefOp = actualFuncRefValue.getDefiningOp<mlir_ts::SymbolRefOp>())
        {
            if (symbolRefOp.getIdentifier() == "")
            {
                if (auto funcType = mlir::dyn_cast<mlir_ts::FunctionType>(symbolRefOp.getType()))
                {
                    if (funcType.getNumInputs() == 0 && funcType.getNumResults() == 1)
                    {
                        if (auto litType = dyn_cast<mlir_ts::LiteralType>(funcType.getResult(0)))
                        {
                            return V(builder.create<mlir_ts::ConstantOp>(location, litType, litType.getValue()));                            
                        }
                    }
                }
            }
        }

        if (mth.isBuiltinFunctionType(actualFuncRefValue))
        {
            return mlirGenCallBuiltInFunction(location, 
                actualFuncRefValue, typeArguments, operands, genContext);
        }

        if (auto optFuncRef = dyn_cast<mlir_ts::OptionalType>(actualFuncRefValue.getType()))
        {
            CAST_A(condValue, location, getBooleanType(), actualFuncRefValue, genContext);

            auto resultType = mth.getReturnTypeFromFuncRef(optFuncRef.getElementType());

            LLVM_DEBUG(llvm::dbgs() << "\n!! Conditional call, return type: " << resultType << "\n";);

            auto hasReturn = !mth.isNoneType(resultType) && resultType != getVoidType();
            auto ifOp = hasReturn
                            ? builder.create<mlir_ts::IfOp>(location, getOptionalType(resultType), condValue, true)
                            : builder.create<mlir_ts::IfOp>(location, condValue, false);

            builder.setInsertionPointToStart(&ifOp.getThenRegion().front());

            // value if true

            auto innerFuncRef =
                builder.create<mlir_ts::ValueOp>(location, optFuncRef.getElementType(), actualFuncRefValue);

            auto result = mlirGenCallExpression(location, innerFuncRef, typeArguments, operands, genContext);
            auto value = V(result);
            if (value)
            {
                auto optValue =
                    builder.create<mlir_ts::OptionalValueOp>(location, getOptionalType(value.getType()), value);
                builder.create<mlir_ts::ResultOp>(location, mlir::ValueRange{optValue});

                // else
                builder.setInsertionPointToStart(&ifOp.getElseRegion().front());

                auto optUndefValue = builder.create<mlir_ts::OptionalUndefOp>(location, getOptionalType(resultType));
                builder.create<mlir_ts::ResultOp>(location, mlir::ValueRange{optUndefValue});
            }

            builder.setInsertionPointAfter(ifOp);

            if (hasReturn)
            {
                return ifOp.getResults().front();
            }

            return mlir::success();
        }

        return mlirGenCall(location, actualFuncRefValue, operands, genContext);
    }

    ValueOrLogicalResult MLIRGenImpl::NewClassInstanceOnStack(mlir::Location location, mlir_ts::ClassType classType,
                                     SmallVector<mlir::Value, 4> &operands, const GenContext &genContext)
    {
        // seems we are calling type constructor
        // TODO: review it, really u should forbid to use "a = Class1();" to allocate in stack, or finish it
        // using Class..new(true) method

        return NewClassInstance(location, classType, operands, genContext, true /*on stack*/);
    }

    ValueOrLogicalResult MLIRGenImpl::NewClassInstance(mlir::Location location, mlir_ts::ClassType classType,
                                     SmallVector<mlir::Value, 4> &operands, const GenContext &genContext, bool onStack)
    {
        auto classInfo = getClassInfoByFullName(classType.getName().getValue());
        if (onStack && classInfo->hasVirtualTable)
        {
            emitError(location, "") << "can't instantiate new instance of " << to_print(classType) << " which has 'virtual table' on stack";
            return mlir::failure();
        }

        auto newOp = onStack 
            ? NewClassInstanceLogicAsOp(location, classType, onStack, genContext)
            : ValueOrLogicalResult(NewClassInstanceAsMethodCallOp(location, classInfo, true, genContext));
        EXIT_IF_FAILED_OR_NO_VALUE(newOp)
        if (mlir::failed(mlirGenCallConstructor(location, classInfo, V(newOp), operands, false, genContext)))
        {
            return mlir::failure();
        }

        return V(newOp);
    }

    ValueOrLogicalResult MLIRGenImpl::mlirGenCall(mlir::Location location, mlir::Value funcRefValue,
                                     SmallVector<mlir::Value, 4> &operands, const GenContext &genContext)
    {
        ValueOrLogicalResult value(mlir::failure());
        mlir::TypeSwitch<mlir::Type>(funcRefValue.getType())
            .Case<mlir_ts::FunctionType>([&](auto calledFuncType) {
                value = mlirGenCallFunction(location, calledFuncType, funcRefValue, operands, genContext);
            })
            .Case<mlir_ts::HybridFunctionType>([&](auto calledFuncType) {
                value = mlirGenCallFunction(location, calledFuncType, funcRefValue, operands, genContext);
            })
            .Case<mlir_ts::BoundFunctionType>([&](auto calledBoundFuncType) {
                auto calledFuncType =
                    getFunctionType(calledBoundFuncType.getInputs(), calledBoundFuncType.getResults(), calledBoundFuncType.isVarArg());
                auto thisValue = builder.create<mlir_ts::GetThisOp>(location, calledFuncType.getInput(0), funcRefValue);
                auto unboundFuncRefValue = builder.create<mlir_ts::GetMethodOp>(location, calledFuncType, funcRefValue);
                value = mlirGenCallFunction(location, calledFuncType, unboundFuncRefValue, thisValue, operands, genContext);
            })
            .Case<mlir_ts::ExtensionFunctionType>([&](auto calledExtentFuncType) {
                auto calledFuncType =
                    getFunctionType(calledExtentFuncType.getInputs(), calledExtentFuncType.getResults(), calledExtentFuncType.isVarArg());
                if (auto createExtensionFunctionOp = funcRefValue.getDefiningOp<mlir_ts::CreateExtensionFunctionOp>())
                {
                    auto thisValue = createExtensionFunctionOp.getThisVal();
                    auto funcRefValue = createExtensionFunctionOp.getFunc();
                    value = mlirGenCallFunction(location, calledFuncType, funcRefValue, thisValue, operands, genContext);
                }
                else
                {
                    emitError(location, "not supported");
                    value = mlir::Value();
                }
            })
            .Case<mlir_ts::ClassType>([&](auto classType) {
                value = NewClassInstanceOnStack(location, classType, operands, genContext);
            })
            .Case<mlir_ts::ClassStorageType>([&](auto classStorageType) {
                MLIRCodeLogic mcl(builder, compileOptions);
                auto refValue = mcl.GetReferenceFromValue(location, funcRefValue);
                if (refValue)
                {
                    // seems we are calling type constructor for super()
                    auto classInfo = getClassInfoByFullName(classStorageType.getName().getValue());
                    // to track result call
                    value = mlirGenCallConstructor(location, classInfo, refValue, operands, true, genContext);
                }
                else
                {
                    llvm_unreachable("not implemented");
                }
            })
            .Default([&](auto type) {
                emitError(location, "not supported function type");
                value = mlir::Value();
            });

        return value;
    }

    ValueOrLogicalResult MLIRGenImpl::callIteratorNext(mlir::Location location, mlir::Value nextProperty, 
        OperandsProcessingInfo* operandsProcessingInfo, const GenContext &genContext) 
    {
        // call nextProperty
        SmallVector<mlir::Value, 4> callOperands;
        auto callResult = mlirGenCall(location, nextProperty, callOperands, genContext);
        EXIT_IF_FAILED_OR_NO_VALUE(callResult)

        // load property "value"
        auto doneProperty = mlirGenPropertyAccessExpression(location, callResult, "done", false, genContext);
        EXIT_IF_FAILED_OR_NO_VALUE(doneProperty)

        auto valueProperty = mlirGenPropertyAccessExpression(location, callResult, "value", false, genContext);
        EXIT_IF_FAILED_OR_NO_VALUE(valueProperty)

        auto valueProp = V(valueProperty);

        if (operandsProcessingInfo != nullptr)
        {
            if (auto receiverType = operandsProcessingInfo->isCastNeededWithOptionalUnwrap(valueProp.getType()))
            {
                CAST(valueProp, location, receiverType, valueProp, genContext);
            }
        }                        

        // conditional expr:  done ? undefined : value
        auto doneInvValue =  V(builder.create<mlir_ts::ArithmeticUnaryOp>(location, getBooleanType(),
            builder.getI32IntegerAttr((int)SyntaxKind::ExclamationToken), doneProperty));

        mlir::Value condValue = builder.create<mlir_ts::OptionalOp>(
            location, getOptionalType(valueProp.getType()), valueProp, doneInvValue);

        return condValue;
    }

    mlir::LogicalResult MLIRGenImpl::processOperandSpreadElement(mlir::Location location, mlir::Value source, OperandsProcessingInfo &operandsProcessingInfo, const GenContext &genContext)
    {
        auto count = operandsProcessingInfo.restCount();

        if (hasIterator(location, source, genContext))
        {
            // treat it as <???>.next().value structure
            // property
            auto nextProperty = mlirGenPropertyAccessExpression(
                location, source, ITERATOR_NEXT, false, genContext);

            for (auto spreadIndex = 0; spreadIndex < count; spreadIndex++)
            {
                auto result = callIteratorNext(location, nextProperty, &operandsProcessingInfo, genContext);
                EXIT_IF_FAILED_OR_NO_VALUE(result)
                operandsProcessingInfo.addOperandAndMoveToNextParameter(V(result));
            }

            return mlir::success();    
        }                                        

        if (isArrayLike(location, source, genContext))
        {
            // treat it as <???>[index] structure
            auto lengthValue = mlirGenPropertyAccessExpression(location, source, LENGTH_FIELD_NAME, false, genContext);
            EXIT_IF_FAILED_OR_NO_VALUE(lengthValue)
            CAST(lengthValue, location, builder.getIndexType(), lengthValue, genContext);

            auto elementType = evaluateElementAccess(location, source, false, genContext);
            if (genContext.receiverType && genContext.receiverType != elementType)
            {
                elementType = genContext.receiverType;
            }

            auto valueFactory =
            (isa<mlir_ts::AnyType>(elementType))
                ? &MLIRGenImpl::anyOrUndefined
                : &MLIRGenImpl::optionalValueOrUndefined;

            for (auto spreadIndex = 0;  spreadIndex < count; spreadIndex++)
            {
                auto indexVal = builder.create<mlir_ts::ConstantOp>(location, mth.getIndexType(),
                                                    mth.getIndexAttrValue(spreadIndex));

                // conditional expr: length > "spreadIndex" ? value[index] : undefined
                auto inBoundsValue = V(builder.create<mlir_ts::LogicalBinaryOp>(location, getBooleanType(),
                    builder.getI32IntegerAttr((int)SyntaxKind::GreaterThanToken), 
                    lengthValue,
                    indexVal));

                auto spreadValue = (this->*valueFactory)(location, inBoundsValue,
                    [&](auto genContext) {
                        auto result = mlirGenElementAccess(location, source, indexVal, false, genContext);
                        EXIT_IF_FAILED_OR_NO_VALUE(result)
                        auto value = V(result);

                        if (auto receiverType = operandsProcessingInfo.isCastNeeded(value.getType()))
                        {
                            CAST(value, location, receiverType, value, genContext);
                        }

                        return ValueOrLogicalResult(value);
                    }, genContext);
                EXIT_IF_FAILED_OR_NO_VALUE(spreadValue)

                operandsProcessingInfo.addOperandAndMoveToNextParameter(spreadValue);
            }

            return mlir::success();
        }

        // this is defualt behavior for tuple
        // treat it as <???>[index] structure
        for (auto spreadIndex = 0;  spreadIndex < count; spreadIndex++)
        {
            auto indexVal = builder.create<mlir_ts::ConstantOp>(location, mth.getStructIndexType(),
                                                mth.getStructIndexAttrValue(spreadIndex));

            auto result = mlirGenElementAccess(location, source, indexVal, false, genContext); 
            EXIT_IF_FAILED_OR_NO_VALUE(result)
            auto value = V(result);

            operandsProcessingInfo.addOperandAndMoveToNextParameter(value);
        }

        return mlir::success();        
    }

    mlir::LogicalResult MLIRGenImpl::mlirGenCallConstructor(mlir::Location location, ClassInfo::TypePtr classInfo,
                                               mlir::Value thisValue, SmallVector<mlir::Value, 4> &operands,
                                               bool castThisValueToClass, const GenContext &genContext)
    {
        assert(classInfo);

        auto virtualTable = classInfo->getHasVirtualTable();
        auto hasConstructor = classInfo->getHasConstructor();
        if (!hasConstructor && !virtualTable)
        {
            return mlir::success();
        }

        auto effectiveThisValue = thisValue;
        if (castThisValueToClass)
        {
            CAST(effectiveThisValue, location, classInfo->classType, thisValue, genContext);
        }

        if (classInfo->getHasConstructor())
        {
            auto accessingFromLevel = detectAccessLevel(mlir::cast<mlir_ts::ClassType>(effectiveThisValue.getType()), genContext);
            if (accessingFromLevel < classInfo->constructorAccessLevel) {
                emitError(location, "Class constructor is not accessable");
                return mlir::failure();
            }

            auto propAccess =
                mlirGenPropertyAccessExpression(location, effectiveThisValue, CONSTRUCTOR_NAME, false, genContext);

            if (!propAccess && !genContext.allowPartialResolve)
            {
                emitError(location) << "Call Constructor: can't find constructor";
            }

            EXIT_IF_FAILED_OR_NO_VALUE(propAccess)
            return mlirGenCall(location, propAccess, operands, genContext);
        }

        return mlir::success();
    }

    ValueOrLogicalResult MLIRGenImpl::NewClassInstance(mlir::Location location, mlir::Value value, NodeArray<Expression> arguments,
                                          NodeArray<TypeNode> typeArguments, bool suppressConstructorCall, 
                                          const GenContext &genContext)
    {

        auto type = value.getType();
        type = mth.convertConstTupleTypeToTupleType(type);

        assert(type);

        auto resultType = type;
        if (mth.isValueType(type))
        {
            resultType = getValueRefType(type);
        }

        // if true, will call Class..new method, otheriwise ts::NewOp which we need to implement Class..new method 
        auto methodCallWay = !suppressConstructorCall;

        mlir::Value newOp;
        if (auto classType = dyn_cast<mlir_ts::ClassType>(resultType))
        {
            auto classInfo = getClassInfoByFullName(classType.getName().getValue());
            if (!classInfo)
            {
                auto genericClassInfo = getGenericClassInfoByFullName(classType.getName().getValue());
                if (genericClassInfo)
                {
                    emitError(location) << "Generic class '"<< to_print(classType) << "' is missing type arguments ";
                    return mlir::failure(); 
                }

                emitError(location) << "Can't find class " << to_print(classType);
                return mlir::failure(); 
            }

            if (genContext.dummyRun)
            {
                // just to cut a lot of calls
                newOp = builder.create<mlir_ts::NewOp>(location, classInfo->classType, builder.getBoolAttr(false));
                return newOp;
            }

            auto newOp = NewClassInstanceAsMethodCallOp(location, classInfo, methodCallWay, genContext);
            if (!newOp)
            {
                return mlir::failure();
            }

            if (methodCallWay)
            {
                // evaluate constructor
                mlir::Type tupleParamsType;

                // we need context with correct thisType to get access to contructor
                GenContext thisTypeGenContext(genContext);
                thisTypeGenContext.thisType = mlir::cast<mlir_ts::ClassType>(newOp.getType());

                auto funcValueRef = evaluateProperty(location, newOp, CONSTRUCTOR_NAME, thisTypeGenContext);
                if (funcValueRef)
                {
                    SmallVector<mlir::Value, 4> operands;
                    if (mlir::failed(mlirGenOperands(arguments, operands, funcValueRef, genContext, 1/*this params shift*/)))
                    {
                        emitError(location) << "Call constructor: can't resolve values of all parameters";
                        return mlir::failure();
                    }

                    assert(newOp);
                    auto result  = mlirGenCallConstructor(location, classInfo, newOp, operands, false, genContext);
                    EXIT_IF_FAILED(result)
                }
            }

            return newOp;
        }

        return NewClassInstanceLogicAsOp(location, resultType, false, genContext);
    }

    ValueOrLogicalResult MLIRGenImpl::NewClassInstanceLogicAsOp(mlir::Location location, mlir::Type typeOfInstance, bool stackAlloc,
                                                   const GenContext &genContext)
    {
        if (auto classType = dyn_cast<mlir_ts::ClassType>(typeOfInstance))
        {
            // set virtual table
            auto classInfo = getClassInfoByFullName(classType.getName().getValue());
            if (!classInfo)
            {
                auto genericClassInfo = getGenericClassInfoByFullName(classType.getName().getValue());
                if (genericClassInfo)
                {
                    emitError(location) << "Generic class '"<< to_print(classType) << "' is missing type arguments ";
                    return mlir::failure(); 
                }

                emitError(location) << "Can't find class " << to_print(classType);
                return mlir::Value(); 
            }

            return NewClassInstanceLogicAsOp(location, classInfo, stackAlloc, genContext);
        }

        LLVM_DEBUG(llvm::dbgs() << "\n!! new op (no method): " << typeOfInstance << "\n";);

        auto newOp = builder.create<mlir_ts::NewOp>(location, typeOfInstance, builder.getBoolAttr(stackAlloc));
        return V(newOp);
    }

    mlir::Value MLIRGenImpl::NewClassInstanceLogicAsOp(mlir::Location location, ClassInfo::TypePtr classInfo, bool stackAlloc,
                                          const GenContext &genContext)
    {
        mlir::Value newOp;
#if ENABLE_TYPED_GC
        auto enabledGC = !compileOptions.disableGC;
        if (enabledGC && !stackAlloc)
        {
            auto typeDescrType = builder.getI64Type();
            auto typeDescGlobalName = getTypeDescriptorFieldName(classInfo);
            auto typeDescRef = resolveFullNameIdentifier(location, typeDescGlobalName, true, genContext);
            auto typeDescCurrentValue = builder.create<mlir_ts::LoadOp>(location, typeDescrType, typeDescRef);

            CAST_A(condVal, location, getBooleanType(), typeDescCurrentValue, genContext);

            auto ifOp = builder.create<mlir_ts::IfOp>(
                location, mlir::TypeRange{typeDescrType}, condVal,
                [&](mlir::OpBuilder &opBuilder, mlir::Location loc) {
                    builder.create<mlir_ts::ResultOp>(loc, mlir::ValueRange{typeDescCurrentValue});
                },
                [&](mlir::OpBuilder &opBuilder, mlir::Location loc) {
                    // call typr bitmap
                    auto fullClassStaticFieldName = getTypeBitmapMethodName(classInfo);

                    auto funcType = getFunctionType({}, {typeDescrType}, false);

                    auto funcSymbolOp = builder.create<mlir_ts::SymbolRefOp>(
                        location, funcType,
                        mlir::FlatSymbolRefAttr::get(builder.getContext(), fullClassStaticFieldName));

                    auto callIndirectOp =
                        builder.create<mlir_ts::CallIndirectOp>(
                            MLIRHelper::getCallSiteLocation(funcSymbolOp->getLoc(), location),
                            funcSymbolOp, mlir::ValueRange{});
                    auto typeDescr = callIndirectOp.getResult(0);

                    // save value
                    builder.create<mlir_ts::StoreOp>(location, typeDescr, typeDescRef);

                    builder.create<mlir_ts::ResultOp>(loc, mlir::ValueRange{typeDescr});
                });

            auto typeDescrValue = ifOp.getResult(0);

            assert(!stackAlloc);
            newOp = builder.create<mlir_ts::GCNewExplicitlyTypedOp>(location, classInfo->classType, typeDescrValue);
        }
        else
        {
            newOp = builder.create<mlir_ts::NewOp>(location, classInfo->classType, builder.getBoolAttr(stackAlloc));
        }
#else
        newOp = builder.create<mlir_ts::NewOp>(location, classInfo->classType, builder.getBoolAttr(stackAlloc));
#endif
        mlirGenSetVTableToInstance(location, classInfo, newOp, genContext);
        return newOp;
    }

    mlir::Value MLIRGenImpl::NewClassInstanceAsMethodCallOp(mlir::Location location, ClassInfo::TypePtr classInfo, bool asMethodCall,
                                             const GenContext &genContext)
    {
#ifdef USE_NEW_AS_METHOD
        if (asMethodCall)
        {
            auto classRefVal = builder.create<mlir_ts::ClassRefOp>(
                location, classInfo->classType,
                mlir::FlatSymbolRefAttr::get(builder.getContext(), classInfo->classType.getName().getValue()));

            // call <Class>..new to create new instance
            auto result = mlirGenPropertyAccessExpression(location, classRefVal, NEW_METHOD_NAME, false, genContext);
            EXIT_IF_FAILED_OR_NO_VALUE(result)
            auto newFuncRef = V(result);

            assert(newFuncRef);

            SmallVector<mlir::Value, 4> emptyOperands;
            auto resultCall = mlirGenCallExpression(location, newFuncRef, {}, emptyOperands, genContext);
            EXIT_IF_FAILED_OR_NO_VALUE(resultCall)
            auto newOp = V(resultCall);
            return newOp;
        }
#endif

        return NewClassInstanceLogicAsOp(location, classInfo, false, genContext);
    }

    ValueOrLogicalResult MLIRGenImpl::NewClassInstanceByCallingNewCtor(mlir::Location location, mlir::Value value, NodeArray<Expression> arguments,
            NodeArray<TypeNode> typeArguments, const GenContext &genContext)
    {
        auto result = mlirGenPropertyAccessExpression(location, value, NEW_CTOR_METHOD_NAME, genContext);
        EXIT_IF_FAILED_OR_NO_VALUE(result)
        auto newCtorMethod = V(result);        

        SmallVector<mlir::Value, 4> operands;
        if (mlir::failed(mlirGenOperands(arguments, operands, newCtorMethod.getType(), genContext)))
        {
            emitError(location) << "Call new instance: can't resolve values of all parameters";
            return mlir::failure();
        }

        return mlirGenCallExpression(location, newCtorMethod, typeArguments, operands, genContext);        
    }

} // namespace mlirgen
} // namespace typescript
