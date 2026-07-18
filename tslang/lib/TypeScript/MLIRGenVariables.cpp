// Variable declaration/binding and identifier resolution methods of MLIRGenImpl (see MLIRGenImpl.h).

#include "MLIRGenImpl.h"

namespace typescript
{
namespace mlirgen
{

    ValueOrLogicalResult MLIRGenImpl::registerVariableInThisContext(mlir::Location location, StringRef name, mlir::Type type,
                                                       const GenContext &genContext)
    {
        if (genContext.passResult)
        {

            // create new type with added field
            genContext.passResult->extraFieldsInThisContext.push_back(
                {MLIRHelper::TupleFieldName(name, builder.getContext()), type, false, mlir_ts::AccessLevel::Public});
            return mlir::Value();
        }

        // resolve object property

        NodeFactory nf(NodeFactoryFlags::None);
        // load this.<var name>
        auto _this = nf.createToken(SyntaxKind::ThisKeyword);
        auto _name = nf.createIdentifier(stows(std::string(name)));
        auto _this_name = nf.createPropertyAccessExpression(_this, _name);

        auto result = mlirGen(_this_name, genContext);
        EXIT_IF_FAILED_OR_NO_VALUE(result)
        auto thisVarValue = V(result);

        assert(thisVarValue);

        MLIRCodeLogic mcl(builder, compileOptions);
        auto thisVarValueRef = mcl.GetReferenceFromValue(location, thisVarValue);

        assert(thisVarValueRef);

        return V(thisVarValueRef);
    }

    mlir::LogicalResult MLIRGenImpl::registerVariableDeclaration(mlir::Location location, VariableDeclarationDOM::TypePtr variableDeclaration, struct VariableDeclarationInfo &variableDeclarationInfo, bool showWarnings, const GenContext &genContext)
    {
        if (variableDeclarationInfo.deleted)
        {
            return mlir::success();
        }
        else if (!variableDeclarationInfo.isGlobal)
        {
            if (mlir::failed(declare(
                location, 
                variableDeclaration, 
                variableDeclarationInfo.storage 
                    ? variableDeclarationInfo.storage 
                    : variableDeclarationInfo.initial, 
                genContext, 
                showWarnings)))
            {
                return mlir::failure();
            }

            if (this->compileOptions.generateDebugInfo 
                && variableDeclarationInfo.initial 
                && !variableDeclarationInfo.storage 
                && !mth.isGenericType(variableDeclarationInfo.initial.getType())
                && !mth.isAnyFunctionType(variableDeclarationInfo.initial.getType()))
            {
                // to show const values
                MLIRDebugInfoHelper mti(builder, debugScope);
                auto namedLoc = mti.combineWithCurrentScopeAndName(location, variableDeclarationInfo.variableName);
                builder.create<mlir_ts::DebugVariableOp>(namedLoc, variableDeclarationInfo.initial);
            }
        }
        else if (variableDeclarationInfo.isFullName)
        {
            fullNameGlobalsMap.insert(variableDeclarationInfo.fullName, variableDeclaration);
        }
        else
        {
            getGlobalsMap().insert({variableDeclarationInfo.variableName, variableDeclaration});
        }

        return mlir::success();
    }

    mlir::Type MLIRGenImpl::registerVariable(mlir::Location location, StringRef name, bool isFullName, VariableClass varClass,
                                TypeValueInitFuncType func, const GenContext &genContext, bool showWarnings, bool forceLocalVar)
    {
        struct VariableDeclarationInfo variableDeclarationInfo(
            compileOptions, func, [this](StringRef name) { return getGlobalsFullNamespaceName(name); });

        variableDeclarationInfo.detectFlags(isFullName, varClass, forceLocalVar, genContext);
        variableDeclarationInfo.setName(name);

        if (declarationMode)
            variableDeclarationInfo.setExternal(true);

        if (!variableDeclarationInfo.isGlobal)
        {
            if (variableDeclarationInfo.isConst) {
                if (mlir::failed(variableDeclarationInfo.processConstRef(location, builder, genContext)))
                    return mlir::Type();

                // a const binding that turns out to need identity storage (see
                // processConstRef / hasBoundMethodField) falls through to the same
                // real-storage path as `let` instead of staying a bare SSA value.
                if (variableDeclarationInfo.needsIdentityStorage
                    && mlir::failed(createLocalVariable(location, variableDeclarationInfo, genContext)))
                    return mlir::Type();
            } else if (mlir::failed(createLocalVariable(location, variableDeclarationInfo, genContext)))
                return mlir::Type();
        }
        else
        {
            variableDeclarationInfo.isSpecialization = genContext.specialization;
            if (mlir::failed(createGlobalVariable(location, variableDeclarationInfo, genContext))) {
                return mlir::Type();
            }

            if (mlir::succeeded(isGlobalConstLambda(location, variableDeclarationInfo, genContext)))
            {
                variableDeclarationInfo.globalOp->erase();
                variableDeclarationInfo.deleted = true;
            }
        }

        if (!variableDeclarationInfo.type)
        {
            emitError(location) << "type of variable '" << variableDeclarationInfo.variableName << "' is not valid";
            return variableDeclarationInfo.type;
        }

        //LLVM_DEBUG(variableDeclarationInfo.printDebugInfo(););

        auto varDecl = variableDeclarationInfo.createVariableDeclaration(location, genContext);
        if (genContext.usingVars != nullptr && varDecl->getUsing())
        {
            genContext.usingVars->push_back(varDecl);
        }

        registerVariableDeclaration(location, varDecl, variableDeclarationInfo, showWarnings, genContext);
        return varDecl->getType();
    }

    ValueOrLogicalResult MLIRGenImpl::processDeclarationArrayBindingPatternSubPath(
        mlir::Location location, int index, mlir::Type type, mlir::Value init, 
        bool isDotDotDot, bool isIterator, bool isArrayLike, mlir::Value arrayLikeLengthValue, mlir::Type arrayLikeElementType, const GenContext &genContext)
    {
        MLIRPropertyAccessCodeLogic cl(compileOptions, builder, location, init, builder.getI32IntegerAttr(index));
        mlir::Value subInit =
            mlir::TypeSwitch<mlir::Type, mlir::Value>(type)
                .template Case<mlir_ts::ConstTupleType>([&](auto constTupleType) { 
                    if (isDotDotDot)
                    {
                        SmallVector<mlir::Value> arrayValues;
                        SmallVector<mlir_ts::FieldInfo> fieldInfos;

                        SmallVector<mlir_ts::FieldInfo> srcFieldInfos;
                        if (mlir::failed(mth.getFields(constTupleType, srcFieldInfos)))
                        {
                            return mlir::Value();
                        }

                        for (auto indexSpread = index; indexSpread < srcFieldInfos.size(); indexSpread++)                        
                        {
                            MLIRPropertyAccessCodeLogic cl(
                                compileOptions, builder, location, init, builder.getI32IntegerAttr(indexSpread));
                            auto value = cl.Tuple(constTupleType, true);

                            //fieldInfos.push_back({mlir::Attribute(), value.getType(), false, mlir_ts::AccessLevel::Public});
                            fieldInfos.push_back(srcFieldInfos[indexSpread]);
                            arrayValues.push_back(value);
                        }
                
                        return V(builder.create<mlir_ts::CreateTupleOp>(location, getTupleType(fieldInfos), arrayValues));
                    }

                    return cl.Tuple(constTupleType, true); 
                })
                .template Case<mlir_ts::TupleType>([&](auto tupleType) { 
                    if (isDotDotDot)
                    {
                        SmallVector<mlir::Value> arrayValues;
                        SmallVector<mlir_ts::FieldInfo> fieldInfos;

                        SmallVector<mlir_ts::FieldInfo> srcFieldInfos;
                        if (mlir::failed(mth.getFields(tupleType, srcFieldInfos)))
                        {
                            return mlir::Value();
                        }

                        for (auto indexSpread = index; indexSpread < srcFieldInfos.size(); indexSpread++)                        
                        {
                            MLIRPropertyAccessCodeLogic cl(
                                compileOptions, builder, location, init, builder.getI32IntegerAttr(indexSpread));
                            auto value = cl.Tuple(tupleType, true);

                            //fieldInfos.push_back({mlir::Attribute(), value.getType(), false, mlir_ts::AccessLevel::Public});
                            fieldInfos.push_back(srcFieldInfos[indexSpread]);
                            arrayValues.push_back(value);
                        }
                
                        return V(builder.create<mlir_ts::CreateTupleOp>(location, getTupleType(fieldInfos), arrayValues));
                    }

                    return cl.Tuple(tupleType, true); 
                })
                .template Case<mlir_ts::ConstArrayType>([&](auto constArrayType) {
                    if (isDotDotDot)
                    {   
                        auto indexType = builder.getIndexType();

                        auto arrayType = mth.removeConstType(constArrayType);

                        auto arrayValue = cast(location, arrayType, init, genContext);
                        if (!arrayValue)
                        {
                            return mlir::Value();
                        }

                        auto constIndex = builder.create<mlir_ts::ConstantOp>(
                            location, indexType, builder.getIndexAttr(index));

                        auto length = builder.create<mlir_ts::LengthOfOp>(location, indexType, arrayValue);

                        auto count = builder.create<mlir_ts::ArithmeticBinaryOp>(
                            location, indexType, builder.getI32IntegerAttr(static_cast<int32_t>(SyntaxKind::MinusToken)), length, constIndex);

                        mlir::Value arrayViewValue =
                            builder.create<mlir_ts::ArrayViewOp>(
                                location, 
                                arrayType, 
                                arrayValue, 
                                constIndex, 
                                count);                        

                        return arrayViewValue;
                    }

                    // TODO: unify it with ElementAccess
                    auto constIndex = builder.create<mlir_ts::ConstantOp>(location, builder.getI32Type(),
                                                                        builder.getI32IntegerAttr(index));
                    auto elemRef = builder.create<mlir_ts::ElementRefOp>(
                        location, mlir_ts::RefType::get(constArrayType.getElementType()), init, constIndex);
                    return V(builder.create<mlir_ts::LoadOp>(location, constArrayType.getElementType(), elemRef));
                })
                .template Case<mlir_ts::ArrayType>([&](auto arrayType) {

                    if (isDotDotDot)
                    {   
                        auto indexType = builder.getIndexType();

                        auto constIndex = builder.create<mlir_ts::ConstantOp>(
                            location, indexType, builder.getIndexAttr(index));

                        auto length = builder.create<mlir_ts::LengthOfOp>(location, indexType, init);

                        auto count = builder.create<mlir_ts::ArithmeticBinaryOp>(
                            location, indexType, builder.getI32IntegerAttr(static_cast<int32_t>(SyntaxKind::MinusToken)), length, constIndex);

                        mlir::Value arrayViewValue =
                            builder.create<mlir_ts::ArrayViewOp>(
                                location, 
                                arrayType, 
                                init, 
                                constIndex, 
                                count);                        

                        return arrayViewValue;
                    }

                    // TODO: unify it with ElementAccess
                    auto constIndex = builder.create<mlir_ts::ConstantOp>(location, builder.getI32Type(),
                                                                        builder.getI32IntegerAttr(index));
                    auto elemRef = builder.create<mlir_ts::ElementRefOp>(
                        location, mlir_ts::RefType::get(arrayType.getElementType()), init, constIndex);
                    return V(builder.create<mlir_ts::LoadOp>(location, arrayType.getElementType(), elemRef));
                })
                .Default([&](auto type) { 

                    if (isDotDotDot)
                    {
                        emitError(location) << "Array Binding Pattern: spread is not implemented for type: " << to_print(type);
                        return mlir::Value();  
                    }

                    if (isIterator)
                    {
                        // seems it is "iterator"
                        auto nextProperty = init;
                        auto result = callIteratorNext(location, nextProperty, nullptr, genContext);
                        return V(result);
                    }

                    // add array like access
                    if (isArrayLike)
                    {
                        auto valueFactory =
                        (isa<mlir_ts::AnyType>(arrayLikeElementType))
                            ? &MLIRGenImpl::anyOrUndefined
                            : &MLIRGenImpl::optionalValueOrUndefined;
            
                        auto indexVal = builder.create<mlir_ts::ConstantOp>(location, mth.getIndexType(),
                                                            mth.getIndexAttrValue(index));
        
                        // conditional expr: length > "spreadIndex" ? value[index] : undefined
                        auto inBoundsValue = V(builder.create<mlir_ts::LogicalBinaryOp>(location, getBooleanType(),
                            builder.getI32IntegerAttr((int)SyntaxKind::GreaterThanToken), 
                            arrayLikeLengthValue,
                            indexVal));
        
                        auto spreadValue = (this->*valueFactory)(location, inBoundsValue,
                            [&](auto genContext) {
                                auto result = mlirGenElementAccess(location, init, indexVal, false, genContext);
                                EXIT_IF_FAILED_OR_NO_VALUE(result)
                                return result;
                            }, genContext);
                        //EXIT_IF_FAILED_OR_NO_VALUE(spreadValue)
                        return V(spreadValue);
                    }

                    emitError(location) << "Array Binding Pattern: not implemented for type: " << to_print(type);
                    return mlir::Value(); 
                });

        if (!subInit)
        {
            return mlir::failure();
        }

        return subInit; 
    }

    mlir::LogicalResult MLIRGenImpl::processDeclarationArrayBindingPattern(mlir::Location location, ArrayBindingPattern arrayBindingPattern,
                                               VariableClass varClass,
                                               TypeValueInitFuncType func,
                                               const GenContext &genContext)
    {
        auto [typeRef, initRef, typeProvidedRef] = func(location, genContext);
        mlir::Type type = typeRef;
        mlir::Value init = initRef;
        //TypeProvided typeProvided = typeProvidedRef;

        if (!init)
        {
            return mlir::failure();
        }

        mlir::Value arrayLikeLengthValue;
        mlir::Type arrayLikeElementType;
        auto isIterator = false;
        auto isSourceArrayLike = false;
        auto isArrayOrTuple = isa<mlir_ts::ArrayType>(typeRef)
            || isa<mlir_ts::TupleType>(typeRef)
            || isa<mlir_ts::ConstArrayType>(typeRef)
            || isa<mlir_ts::ConstTupleType>(typeRef);
        if (!isArrayOrTuple)
        {
            if (auto iteratorType = evaluateProperty(location, init, SYMBOL_ITERATOR, genContext))
            {
                if (auto iteratorResult = mlirGenCallThisMethod(location, init, SYMBOL_ITERATOR, undefined, undefined, genContext))
                {
                    auto iteratorValue = V(iteratorResult);

                    // request iterator
                    auto nextProperty = mlirGenPropertyAccessExpression(
                        location, iteratorValue, ITERATOR_NEXT, false, genContext);
                    if (nextProperty)
                    {
                        init = V(nextProperty);
                        isIterator = true;
                    }                    
                }
            }
            else if (hasIterator(location, init, genContext))
            {
                // request iterator
                auto nextProperty = mlirGenPropertyAccessExpression(
                    location, init, ITERATOR_NEXT, false, genContext);
                if (nextProperty)
                {
                    init = V(nextProperty);
                    isIterator = true;
                }
            }
            else if (isArrayLike(location, init, genContext))
            {
                auto lengthValue = mlirGenPropertyAccessExpression(location, init, LENGTH_FIELD_NAME, false, genContext);
                EXIT_IF_FAILED_OR_NO_VALUE(lengthValue)
                arrayLikeLengthValue = V(lengthValue);
                CAST(arrayLikeLengthValue, location, builder.getIndexType(), arrayLikeLengthValue, genContext);
                auto elementType = evaluateElementAccess(location, init, false, genContext);
                if (elementType)
                {
                    isSourceArrayLike = true;
                    arrayLikeElementType = elementType;
                }
            }            
            else
            {
                emitError(location) << "Array Binding Pattern: unsupported source of array data";
                return mlir::failure();
            }
        }        

        for (auto [index, element] : enumerate(arrayBindingPattern->elements))
        {
            if (element == SyntaxKind::OmittedExpression)
            {
                continue;
            }

            if (element != SyntaxKind::BindingElement)
            {
                emitError(location) << "Array Binding Pattern: unsupported element";
                return mlir::failure();
            }

            auto arrayBindingElement = element.as<BindingElement>();

            auto subValueFunc = [&](mlir::Location location, const GenContext &genContext) { 
                auto result = processDeclarationArrayBindingPatternSubPath(
                    location, index, type, init, !!arrayBindingElement->dotDotDotToken, isIterator, isSourceArrayLike, arrayLikeLengthValue, arrayLikeElementType, genContext);
                if (result.failed_or_no_value()) 
                {
                    return std::make_tuple(mlir::Type(), mlir::Value(), TypeProvided::No); 
                }

                auto value = V(result);
                return std::make_tuple(value.getType(), value, TypeProvided::No); 
            };

            if (mlir::failed(processDeclaration(
                    arrayBindingElement, varClass, subValueFunc, genContext)))
            {
                return mlir::failure();
            }
        }

        return mlir::success();
    }

    ValueOrLogicalResult MLIRGenImpl::processDeclarationObjectBindingPatternSubPath(
        mlir::Location location, BindingElement objectBindingElement, mlir::Type type, mlir::Value init, const GenContext &genContext)
    {
        auto fieldName = getFieldNameFromBindingElement(objectBindingElement);
        auto isNumericAccess = isa<mlir::IntegerAttr>(fieldName);

        LLVM_DEBUG(llvm::dbgs() << "ObjectBindingPattern:\n\t" << init << "\n\tprop: " << fieldName << "\n");

        mlir::Value subInit;
        mlir::Type subInitType;

        mlir::Value value;
        if (isNumericAccess)
        {
            MLIRPropertyAccessCodeLogic cl(compileOptions, builder, location, init, fieldName);
            if (auto tupleType = dyn_cast<mlir_ts::TupleType>(type))
            {
                value = cl.Tuple(tupleType, true);
            }
            else if (auto constTupleType = dyn_cast<mlir_ts::ConstTupleType>(type))
            {
                value = cl.Tuple(constTupleType, true);
            }
        }
        else
        {
            auto result = mlirGenPropertyAccessExpression(location, init, fieldName, false, genContext);
            EXIT_IF_FAILED_OR_NO_VALUE(result)
            value = V(result);
        }

        if (!value)
        {
            return mlir::failure();
        }

        if (objectBindingElement->initializer)
        {
            auto tupleType = mlir::cast<mlir_ts::TupleType>(type);
            auto subType = mlir::cast<mlir_ts::OptionalType>(tupleType.getFieldInfo(tupleType.getIndex(fieldName)).type).getElementType();
            auto res = optionalValueOrDefault(location, subType, value, objectBindingElement->initializer, genContext);
            subInit = V(res);
            subInitType = subInit.getType();                    
        }
        else
        {
            subInit = value;
            subInitType = subInit.getType();
        }

        assert(subInit);

        return subInit; 
    }

    ValueOrLogicalResult MLIRGenImpl::processDeclarationObjectBindingPatternSubPathSpread(
        mlir::Location location, ObjectBindingPattern objectBindingPattern, mlir::Type type, mlir::Value init, const GenContext &genContext)
    {
        mlir::Value subInit;
        mlir::Type subInitType;

        SmallVector<mlir::Attribute> names;

        // take all used fields
        for (auto objectBindingElement : objectBindingPattern->elements)
        {
            auto isSpreadBinding = !!objectBindingElement->dotDotDotToken;
            if (isSpreadBinding)
            {
                continue;
            }

            auto fieldId = getFieldNameFromBindingElement(objectBindingElement);
            names.push_back(fieldId);
        }                

        // filter all fields
        llvm::SmallVector<mlir_ts::FieldInfo> tupleFields;
        llvm::SmallVector<mlir_ts::FieldInfo> destTupleFields;
        if (mlir::succeeded(mth.getFields(init.getType(), tupleFields)))
        {
            for (auto fieldInfo : tupleFields)
            {
                if (std::find_if(names.begin(), names.end(), [&] (auto& item) { return item == fieldInfo.id; }) == names.end())
                {
                    // filter;
                    destTupleFields.push_back(fieldInfo);
                }
            }
        }

        // create object
        subInitType = getTupleType(destTupleFields);
        CAST(subInit, location, subInitType, init, genContext);

        assert(subInit);

        return subInit; 
    }

    mlir::LogicalResult MLIRGenImpl::processDeclarationObjectBindingPattern(mlir::Location location, ObjectBindingPattern objectBindingPattern,
                                                VariableClass varClass,
                                                TypeValueInitFuncType func,
                                                const GenContext &genContext)
    {
        auto [typeRef, initRef, typeProvidedRef] = func(location, genContext);
        mlir::Type type = typeRef;
        mlir::Value init = initRef;
        //TypeProvided typeProvided = typeProvidedRef;

        for (auto objectBindingElement : objectBindingPattern->elements)
        {
            auto subValueFunc = [&] (mlir::Location location, const GenContext &genContext) {

                auto isSpreadBinding = !!objectBindingElement->dotDotDotToken;
                auto result = isSpreadBinding 
                    ? processDeclarationObjectBindingPatternSubPathSpread(location, objectBindingPattern, type, init, genContext)
                    : processDeclarationObjectBindingPatternSubPath(location, objectBindingElement, type, init, genContext);
                if (result.failed_or_no_value()) 
                {
                    return std::make_tuple(mlir::Type(), mlir::Value(), TypeProvided::No); 
                }                    

                auto value = V(result);
                return std::make_tuple(value.getType(), value, TypeProvided::No); 
            };

            // nested obj, objectBindingElement->propertyName -> name
            if (objectBindingElement->name == SyntaxKind::ObjectBindingPattern)
            {
                auto objectBindingPattern = objectBindingElement->name.as<ObjectBindingPattern>();

                return processDeclarationObjectBindingPattern(
                    location, objectBindingPattern, varClass, subValueFunc, genContext);
            }

            if (mlir::failed(processDeclaration(
                    objectBindingElement, varClass, subValueFunc, genContext)))
            { 
                return mlir::failure();
            }
        }

        return mlir::success();;
    }

    mlir::LogicalResult MLIRGenImpl::processDeclarationName(DeclarationName name, VariableClass varClass,
                            TypeValueInitFuncType func, const GenContext &genContext, bool showWarnings)
    {
        auto location = loc(name);

        if (name == SyntaxKind::ArrayBindingPattern)
        {
            auto arrayBindingPattern = name.as<ArrayBindingPattern>();
            return processDeclarationArrayBindingPattern(location, arrayBindingPattern, varClass, func, genContext);
        }
        else if (name == SyntaxKind::ObjectBindingPattern)
        {
            auto objectBindingPattern = name.as<ObjectBindingPattern>();
            return processDeclarationObjectBindingPattern(location, objectBindingPattern, varClass, func, genContext);
        }
        else
        {
            // name
            auto nameStr = MLIRHelper::getName(name);

            // register
            auto varType = registerVariable(location, nameStr, false, varClass, func, genContext, showWarnings);
            if (!varType)
            {
                return mlir::failure();
            }

            if (varClass.isExport)
            {
                auto isConst = varClass.type == VariableType::Const || varClass.type == VariableType::ConstRef;
                addVariableDeclarationToExport(nameStr, currentNamespace, varType, isConst);
            }

            return mlir::success();
        }

        return mlir::failure();       
    }

    mlir::LogicalResult MLIRGenImpl::processDeclaration(NamedDeclaration item, VariableClass varClass,
                            TypeValueInitFuncType func, const GenContext &genContext, bool showWarnings)
    {
        if (item == SyntaxKind::OmittedExpression)
        {
            return mlir::success();
        }

        item->name->parent = item;
        return processDeclarationName(item->name, varClass, func, genContext, showWarnings);
    }

    mlir::LogicalResult MLIRGenImpl::mlirGen(VariableDeclaration item, VariableClass varClass, const GenContext &genContext)
    {
        auto location = loc(item);

#ifndef ANY_AS_DEFAULT
        auto isExternal = varClass == VariableType::External;
        if (declarationMode)
        {
            isExternal = true;
        }

        if (mth.isNoneType(item->type) && !item->initializer && !isExternal)
        {
            auto name = MLIRHelper::getName(item->name);
            emitError(loc(item)) << "type of variable '" << name
                                 << "' is not provided, variable must have type or initializer";
            return mlir::failure();
        }
#endif

        auto initFunc = [&](mlir::Location location, const GenContext &genContext) {
            if (declarationMode)
            {
                auto [t, b, p] = evaluateTypeAndInit(item, genContext);
                return std::make_tuple(t, mlir::Value(), p ? TypeProvided::Yes : TypeProvided::No);
            }

            auto typeAndInit = getTypeAndInit(item, genContext);

            if (varClass.isDynamicImport)
            {
                auto nameStr = concatFullNamespaceName(MLIRHelper::getName(item->name));
                auto fieldType = std::get<0>(typeAndInit);
                if (fieldType)
                {
                    auto dllVarName = V(mlirGenStringValue(location, nameStr, true));
                    auto referenceToStaticFieldOpaque = builder.create<mlir_ts::SearchForAddressOfSymbolOp>(
                        location, getOpaqueType(), dllVarName);
                    auto refToTyped = cast(location, mlir_ts::RefType::get(fieldType), referenceToStaticFieldOpaque, genContext);
                    auto valueOfField = builder.create<mlir_ts::LoadOp>(location, fieldType, refToTyped);
                    return std::make_tuple(valueOfField.getType(), V(valueOfField), TypeProvided::Yes);                
                }
            }

            return typeAndInit;
        };

        auto valClassItem = varClass;
        if ((item->internalFlags & InternalFlags::ForceConst) == InternalFlags::ForceConst)
        {
            valClassItem = VariableType::Const;
        }

        if ((item->internalFlags & InternalFlags::ForceConstRef) == InternalFlags::ForceConstRef)
        {
            valClassItem = VariableType::ConstRef;
        }

        if (!genContext.funcOp && (item->name == SyntaxKind::ObjectBindingPattern || item->name == SyntaxKind::ArrayBindingPattern))
        {
            auto name = MLIRHelper::getAnonymousName(location, ".gc", "");
            auto fullInitGlobalFuncName = getFullNamespaceName(name);

            {
                mlir::OpBuilder::InsertionGuard insertGuard(builder);

                // create global construct
                valClassItem = VariableType::Var;

                auto funcType = getFunctionType({}, {}, false);

                if (mlir::failed(mlirGenFunctionBody(location, name, fullInitGlobalFuncName, funcType,
                    [&](mlir::Location location, const GenContext &genContext) {
                        return processDeclaration(item, valClassItem, initFunc, genContext, true);
                    }, genContext)))
                {
                    return mlir::failure();
                }

                addGlobalConstructor(location, fullInitGlobalFuncName);
            }
        }
        else if (mlir::failed(processDeclaration(item, valClassItem, initFunc, genContext, true)))
        {
            return mlir::failure();
        }

        return mlir::success();
    }

    mlir::LogicalResult MLIRGenImpl::mlirGen(VariableDeclarationList variableDeclarationListAST, const GenContext &genContext)
    {
        auto isLet = (variableDeclarationListAST->flags & NodeFlags::Let) == NodeFlags::Let;
        auto isConst = (variableDeclarationListAST->flags & NodeFlags::Const) == NodeFlags::Const;
        auto isUsing = (variableDeclarationListAST->flags & NodeFlags::Using) == NodeFlags::Using;
        auto isExternal = (variableDeclarationListAST->flags & NodeFlags::Ambient) == NodeFlags::Ambient;
        VariableClass varClass = isExternal ? VariableType::External
                        : isLet    ? VariableType::Let
                        : isConst || isUsing ? VariableType::Const
                                   : VariableType::Var;

        varClass.isUsing = isUsing;

        if (variableDeclarationListAST->parent)
        {
            varClass.isPublic = hasModifier(variableDeclarationListAST->parent, SyntaxKind::ExportKeyword);
            varClass.isExport = getExportModifier(variableDeclarationListAST->parent);
            iterateDecorators(variableDeclarationListAST->parent, genContext, [&](StringRef name, SmallVector<StringRef> args) {
                if (name == DLL_EXPORT)
                {
                    varClass.isExport = true;
                }

                if (name == DLL_IMPORT)
                {
                    varClass.type = isLet ? VariableType::Let : isConst || isUsing ? VariableType::Const : VariableType::Var;                    
                    varClass.isImport = true;
                    // it has parameter, means this is dynamic import, should point to dll path
                    // TODO: finish it, look at mlirGenCustomRTTIDynamicImport as example how to load it
                    if (args.size() > 0)
                    {
                        varClass.type = VariableType::Var; 
                        varClass.isDynamicImport = true;
                        varClass.isImport = false;
                    }
                }                

                if (name == "used") {
                    varClass.isUsed = true;
                }

                if (name == "atomic") {
                    varClass.atomic = true;
                    if (args.size() > 0) 
                    {
                        auto ordering = 0;
                        if (llvm::to_integer(args[0], ordering))
                        {
                            varClass.ordering = ordering;
                        }
                    }

                    if (args.size() > 1)
                        varClass.syncscope = args[1];
                }

                if (name == "volatile") {
                    varClass.isVolatile = true;
                }

                if (name == "nontemporal") {
                    varClass.nonTemporal = true;
                }

                if (name == "invariant") {
                    varClass.invariant = true;
                }
            });
        }

        for (auto &item : variableDeclarationListAST->declarations)
        {
            // we need it for support "undefined type" in 'let' without initialization
            item->parent = variableDeclarationListAST;
            if (mlir::failed(mlirGen(item, varClass, genContext)))
            {
                return mlir::failure();
            }
        }

        return mlir::success();
    }

    mlir::Type MLIRGenImpl::mlirGenParameterObjectOrArrayBinding(Node name, const GenContext &genContext)
    {
        // TODO: put it into function to support recursive call
        if (name == SyntaxKind::ObjectBindingPattern)
        {
            SmallVector<mlir_ts::FieldInfo> fieldInfos;

            // we need to construct object type
            auto objectBindingPattern = name.as<ObjectBindingPattern>();
            for (auto objectBindingElement : objectBindingPattern->elements)
            {
                mlirGenParameterBindingElement(objectBindingElement, fieldInfos, genContext);
            }

            return getTupleType(fieldInfos);
        } 
        else if (name == SyntaxKind::ArrayBindingPattern)
        {
            SmallVector<mlir_ts::FieldInfo> fieldInfos;

            // we need to construct object type
            auto arrayBindingPattern = name.as<ArrayBindingPattern>();
            for (auto arrayBindingElement : arrayBindingPattern->elements)
            {
                if (arrayBindingElement == SyntaxKind::OmittedExpression)
                {
                    continue;
                }

                if (arrayBindingElement == SyntaxKind::BindingElement)
                {
                    auto objectBindingElement = arrayBindingElement.as<BindingElement>();
                    mlirGenParameterBindingElement(objectBindingElement, fieldInfos, genContext);
                }
            }

            return getTupleType(fieldInfos);
        }        

        return mlir::Type();
    }

    mlir::Value MLIRGenImpl::resolveIdentifierAsVariable(mlir::Location location, StringRef name, const GenContext &genContext)
    {
        if (name.empty())
        {
            return mlir::Value();
        }

        auto value = symbolTable.lookup(name);
        if (value.second && value.first)
        {
            //LLVM_DEBUG(dbgs() << "\n!! resolveIdentifierAsVariable: " << name << " type: " << value.second->getType() <<  " value: " << value.first;);

            // begin of logic: outer vars
            auto valueRegion = value.first.getParentRegion();
            auto isOuterVar = false;
            // TODO: review code "valueRegion && valueRegion->getParentOp()" is to support async.execute
            if (genContext.funcOp && genContext.funcOp != tempFuncOp && valueRegion &&
                valueRegion->getParentOp() /* && valueRegion->getParentOp()->getParentOp()*/)
            {
                mlir_ts::FuncOp contextFuncOp = genContext.funcOp;
                auto funcRegion = contextFuncOp.getCallableRegion();

                isOuterVar = !funcRegion->isAncestor(valueRegion);
                // TODO: HACK
                if (isOuterVar && value.second->getIgnoreCapturing())
                {
                    // special case when "ForceConstRef" pointering to outer variable but it is not outer var
                    isOuterVar = false;
                }

                LLVM_DEBUG(if (isOuterVar) dbgs() << "\n!! outer var: [" << value.second->getName()
                                  << "] \n\n\tvalue region: " << *valueRegion->getParentOp()
                                  << " \n\n\tFuncOp: " << contextFuncOp << "";);
            }

            if (isOuterVar && genContext.passResult && !isGenericFunctionReference(value.first))
            {
                LLVM_DEBUG(dbgs() << "\n!! capturing var: [" << value.second->getName()
                                  << "] \n\tvalue pair: " << value.first << " \n\ttype: " << value.second->getType()
                                  << " \n\treadwrite: " << value.second->getReadWriteAccess() << "";);

                // debug ref of ref
                assert(!isa<mlir_ts::RefType>(value.second->getType()));

                // valueRegion->viewGraph();

                // special case, to prevent capturing ".a" because of reference to outer VaribleOp, which is hack (review
                // solution for it)
                genContext.passResult->outerVariables.insert({value.second->getName(), value.second});
            }

            // end of logic: outer vars

            if (!value.second->getReadWriteAccess())
            {
                return value.first;
            }

            //LLVM_DEBUG(dbgs() << "\n!! variable: " << name << " type: " << value.first.getType() << "\n");

            // load value if memref
            auto valueType = mlir::cast<mlir_ts::RefType>(value.first.getType()).getElementType();
            auto loadOp = builder.create<mlir_ts::LoadOp>(location, valueType, value.first);
            if (value.second->getAtomic())
            {
                loadOp->setAttr(ATOMIC_ATTR_NAME, builder.getBoolAttr(true));
                loadOp->setAttr(ORDERING_ATTR_NAME, builder.getI32IntegerAttr(value.second->getOrdering()));
                loadOp->setAttr(SYNCSCOPE_ATTR_NAME, builder.getStringAttr(value.second->getSyncScope()));
            }

            if (value.second->getVolatile())
            {
                loadOp->setAttr(VOLATILE_ATTR_NAME, builder.getBoolAttr(true));
            }

            if (value.second->getNonTemporal())
            {
                loadOp->setAttr(NONTEMPORAL_ATTR_NAME, builder.getBoolAttr(true));
            }            

            if (value.second->getInvariant())
            {
                loadOp->setAttr(INVARIANT_ATTR_NAME, builder.getBoolAttr(true));
            }            

            return loadOp;
        }

        return mlir::Value();
    }

    mlir::Value MLIRGenImpl::resolveFunctionNameInNamespace(mlir::Location location, StringRef name, const GenContext &genContext)
    {
        // resolving function
        auto fn = getFunctionMap().find(name);
        if (fn != getFunctionMap().end())
        {
            auto &funcEntry = fn->getValue();
            return resolveFunctionWithCapture(location, funcEntry.name, funcEntry.funcType, mlir::Value(), false, genContext);
        }

        return mlir::Value();
    }

    mlir::Type MLIRGenImpl::resolveTypeByNameInNamespace(mlir::Location location, StringRef name, const GenContext &genContext)
    {
        // support generic types
        if (genContext.typeParamsWithArgs.size() > 0)
        {
            auto type = getResolveTypeParameter(name, false, genContext);
            if (type)
            {
                return type;
            }
        }

        if (genContext.typeAliasMap.count(name))
        {
            auto typeAliasInfo = genContext.typeAliasMap.lookup(name);
            assert(typeAliasInfo);
            return typeAliasInfo;
        }

        if (getTypeAliasMap().count(name))
        {
            auto typeAliasInfo = getTypeAliasMap().lookup(name);
            if (typeAliasInfo.first)
            {
                return typeAliasInfo.first;
            }

            assert(typeAliasInfo.second);
            GenContext typeAliasGenContext(genContext);
            auto type = getType(typeAliasInfo.second, typeAliasGenContext);
            if (!type)
            {
                typeAliasInfo.first = type;
            }

            return type;
        }

        if (getClassesMap().count(name))
        {
            auto classInfo = getClassesMap().lookup(name);
            if (!classInfo->classType)
            {
                emitError(location) << "can't find class: " << name << "\n";
                return mlir::Type();
            }

            return classInfo->classType;
        }

        if (getGenericClassesMap().count(name))
        {
            auto genericClassInfo = getGenericClassesMap().lookup(name);

            return genericClassInfo->classType;
        }

        if (getInterfacesMap().count(name))
        {
            auto interfaceInfo = getInterfacesMap().lookup(name);
            if (!interfaceInfo->interfaceType)
            {
                emitError(location) << "can't find interface: " << name << "\n";
                return mlir::Type();
            }

            return interfaceInfo->interfaceType;
        }

        if (getGenericInterfacesMap().count(name))
        {
            auto genericInterfaceInfo = getGenericInterfacesMap().lookup(name);
            return genericInterfaceInfo->interfaceType;
        }

        // check if we have enum
        if (getEnumsMap().count(name))
        {
            auto enumTypeInfo = getEnumsMap().lookup(name);
            return getEnumType(
                mlir::FlatSymbolRefAttr::get(builder.getContext(), concatFullNamespaceName(name)),
                enumTypeInfo.first, 
                enumTypeInfo.second);
        }

        if (getImportEqualsMap().count(name))
        {
            auto fullName = getImportEqualsMap().lookup(name);
            auto classInfo = getClassInfoByFullName(fullName);
            if (classInfo)
            {
                return classInfo->classType;
            }

            auto interfaceInfo = getInterfaceInfoByFullName(fullName);
            if (interfaceInfo)
            {
                return interfaceInfo->interfaceType;
            }
        }        

        return mlir::Type();
    }

    mlir::Type MLIRGenImpl::resolveTypeByName(mlir::Location location, StringRef name, const GenContext &genContext)
    {
        auto type = resolveTypeByNameInNamespace(location, name, genContext);
        if (type)
        {
            return type;
        }

        {
            MLIRNamespaceGuard ng(currentNamespace);

            // search in outer namespaces
            while (currentNamespace->isFunctionNamespace)
            {
                currentNamespace = currentNamespace->parentNamespace;
                type = resolveTypeByNameInNamespace(location, name, genContext);
                if (type)
                {
                    return type;
                }
            }

            // search in root namespace
            currentNamespace = rootNamespace;
            type = resolveTypeByNameInNamespace(location, name, genContext);
            if (type)
            {
                return type;
            }
        }    

        if (!isEmbededType(name))
            emitError(location, "can't find type by name: ") << name;

        return mlir::Type();    
    }

    mlir::Value MLIRGenImpl::resolveIdentifierInNamespace(mlir::Location location, StringRef name, const GenContext &genContext)
    {
        if (getGenericFunctionMap().count(name))
        {
            auto genericFunctionInfo = getGenericFunctionMap().lookup(name);

            auto funcSymbolOp = builder.create<mlir_ts::SymbolRefOp>(
                location, genericFunctionInfo->funcType,
                mlir::FlatSymbolRefAttr::get(builder.getContext(), genericFunctionInfo->name));
            funcSymbolOp->setAttr(GENERIC_ATTR_NAME, mlir::BoolAttr::get(builder.getContext(), true));
            return funcSymbolOp;
        }

        auto value = resolveFunctionNameInNamespace(location, name, genContext);
        if (value)
        {
            return value;
        }

        if (getGlobalsMap().count(name))
        {
            auto value = getGlobalsMap().lookup(name);
            return globalVariableAccess(location, value, false, genContext);
        }

        // check if we have enum
        if (getEnumsMap().count(name))
        {
            auto enumTypeInfo = getEnumsMap().lookup(name);
            return builder.create<mlir_ts::ConstantOp>(
                location, 
                getEnumType(
                    mlir::FlatSymbolRefAttr::get(builder.getContext(), concatFullNamespaceName(name)),
                    enumTypeInfo.first, 
                    enumTypeInfo.second), 
                enumTypeInfo.second);
        }

        if (getNamespaceMap().count(name))
        {
            auto namespaceInfo = getNamespaceMap().lookup(name);
            assert(namespaceInfo);
            auto nsName = mlir::FlatSymbolRefAttr::get(builder.getContext(), namespaceInfo->fullName);
            return builder.create<mlir_ts::NamespaceRefOp>(location, namespaceInfo->namespaceType, nsName);
        }

        if (getImportEqualsMap().count(name))
        {
            auto fullName = getImportEqualsMap().lookup(name);
            auto namespaceInfo = getNamespaceByFullName(fullName);
            if (namespaceInfo)
            {
                assert(namespaceInfo);
                auto nsName = mlir::FlatSymbolRefAttr::get(builder.getContext(), namespaceInfo->fullName);
                return builder.create<mlir_ts::NamespaceRefOp>(location, namespaceInfo->namespaceType, nsName);
            }
        }

        auto type = resolveTypeByNameInNamespace(location, name, genContext);
        if (type)
        {
            if (auto classType = dyn_cast<mlir_ts::ClassType>(type))
            {
                return builder.create<mlir_ts::ClassRefOp>(
                    location, classType, mlir::FlatSymbolRefAttr::get(builder.getContext(), classType.getName().getValue()));
            }

            if (auto interfaceType = dyn_cast<mlir_ts::InterfaceType>(type))
            {
                return builder.create<mlir_ts::InterfaceRefOp>(
                    location, interfaceType, mlir::FlatSymbolRefAttr::get(builder.getContext(), interfaceType.getName().getValue()));
            }

            return builder.create<mlir_ts::TypeRefOp>(location, type);
        }        

        return mlir::Value();
    }

    mlir::Value MLIRGenImpl::resolveFullNameIdentifier(mlir::Location location, StringRef name, bool asAddess,
                                          const GenContext &genContext)
    {
        if (fullNameGlobalsMap.count(name))
        {
            auto value = fullNameGlobalsMap.lookup(name);
            return globalVariableAccess(location, value, asAddess, genContext);
        }

        return mlir::Value();
    }

    mlir::Value MLIRGenImpl::resolveIdentifier(mlir::Location location, StringRef name, const GenContext &genContext)
    {
        auto value = resolveIdentifierAsVariable(location, name, genContext);
        if (value)
        {
            return value;
        }

        value = resolveIdentifierInNamespace(location, name, genContext);
        if (value)
        {
            return value;
        }

        {
            MLIRNamespaceGuard ng(currentNamespace);

            // search in outer namespaces
            while (currentNamespace->isFunctionNamespace)
            {
                currentNamespace = currentNamespace->parentNamespace;
                value = resolveIdentifierInNamespace(location, name, genContext);
                if (value)
                {
                    return value;
                }
            }

            // search in root namespace
            currentNamespace = rootNamespace;
            value = resolveIdentifierInNamespace(location, name, genContext);
            if (value)
            {
                return value;
            }
        }

        // try to resolve 'this' if not resolved yet
        if (genContext.thisType && name == THIS_NAME)
        {
            if (auto classType = dyn_cast<mlir_ts::ClassType>(genContext.thisType)) {
                return builder.create<mlir_ts::ClassRefOp>(
                    location, classType, mlir::FlatSymbolRefAttr::get(builder.getContext(), 
                    classType.getName().getValue()));
            }

            return builder.create<mlir_ts::TypeRefOp>(location, genContext.thisType);
        }

        if (genContext.thisType && name == SUPER_NAME)
        {
            mlir::Value thisValue;
            auto thisType = genContext.thisType;
            if (!isa<mlir_ts::ClassType>(genContext.thisType) && !isa<mlir_ts::ClassStorageType>(genContext.thisType))
            {
                auto result = mlirGen(location, THIS_ALIAS, genContext);
                if (result.failed_or_no_value()) {
                    return mlir::Value();
                }

                thisValue = V(result);
                thisType = thisValue.getType();
                if (!isa<mlir_ts::ClassType>(thisType) && !isa<mlir_ts::ClassStorageType>(thisType)) {
                    return mlir::Value();
                }
            }
            else
            {
                auto result = mlirGen(location, THIS_NAME, genContext);
                thisValue = V(result);
            }

            auto fullName = isa<mlir_ts::ClassStorageType>(thisType) 
                ? mlir::cast<mlir_ts::ClassStorageType>(thisType).getName().getValue() 
                : mlir::cast<mlir_ts::ClassType>(thisType).getName().getValue();
            auto classInfo = getClassInfoByFullName(fullName);
            auto baseClassInfo = classInfo->baseClasses.front();

            // this is access to static base class
            if (thisValue.getDefiningOp<mlir_ts::ClassRefOp>())
            {
                return builder.create<mlir_ts::ClassRefOp>(
                    location, baseClassInfo->classType,
                    mlir::FlatSymbolRefAttr::get(builder.getContext(),
                                                baseClassInfo->classType.getName().getValue()));                   
            }

            return mlirGenPropertyAccessExpression(location, thisValue, baseClassInfo->fullName, genContext);
        }

        // built-in types
        if (name == UNDEFINED_NAME)
        {
            return getUndefined(location);
        }

        if (name == INFINITY_NAME)
        {
            return getInfinity(location);
        }

        if (name == NAN_NAME)
        {
            return getNaN(location);
        }

        // end of built-in types

        value = resolveFullNameIdentifier(location, name, false, genContext);
        if (value)
        {
            return value;
        }

        return mlir::Value();
    }

} // namespace mlirgen
} // namespace typescript
