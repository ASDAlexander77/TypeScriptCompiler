// Generic type inference and specialization/instantiation methods of MLIRGenImpl (see MLIRGenImpl.h).

#include "MLIRGenImpl.h"

namespace typescript
{
namespace mlirgen
{

    bool MLIRGenImpl::tryInferNamedGeneric(mlir::Location location, mlir::Type templateType, mlir::Type concreteType,
                              StringMap<mlir::Type> &results)
    {
        auto namedGenType = dyn_cast<mlir_ts::NamedGenericType>(templateType);
        if (!namedGenType)
        {
            return false;
        }

        // merge if exists

        auto currentType = concreteType;
        auto name = namedGenType.getName().getValue();
        auto existType = results.lookup(name);
        if (existType)
        {
            auto merged = false;
            currentType = mth.mergeType(location, existType, currentType, merged);

            LLVM_DEBUG(llvm::dbgs() << "\n!! result type: " << currentType << "\n";);
            results[name] = currentType;
        }
        else
        {
            // TODO: when u use literal type to validate extends u need to use original type
            // currentType = mth.wideStorageType(currentType);
            LLVM_DEBUG(llvm::dbgs() << "\n!! type: " << name << " = " << currentType << "\n";);
            results.insert({name, currentType});
        }

        assert(results.lookup(name) == currentType);

        return true;
    }

    bool MLIRGenImpl::tryInferClass(mlir::Location location, mlir::Type templateType, mlir::Type concreteType,
                       StringMap<mlir::Type> &results, const GenContext &genContext)
    {
        auto tempClass = dyn_cast<mlir_ts::ClassType>(templateType);
        auto typeClass = dyn_cast<mlir_ts::ClassType>(concreteType);
        if (!tempClass || !typeClass)
        {
            return false;
        }

        auto typeClassInfo = getClassInfoByFullName(typeClass.getName().getValue());
        if (auto tempClassInfo = getClassInfoByFullName(tempClass.getName().getValue()))
        {
            for (auto &templateParam : tempClassInfo->typeParamsWithArgs)
            {
                auto name = templateParam.getValue().first->getName();
                auto found = typeClassInfo->typeParamsWithArgs.find(name);
                if (found != typeClassInfo->typeParamsWithArgs.end())
                {
                    // TODO: convert GenericType -> AnyGenericType,  and NamedGenericType -> GenericType, and
                    // add 2 type Parameters to it Constrain, Default
                    inferType(location, templateParam.getValue().second, found->getValue().second, results, genContext);
                }
            }

            return true;
        }
        else if (auto tempGenericClassInfo = getGenericClassInfoByFullName(tempClass.getName().getValue()))
        {
            for (auto &templateParam : tempGenericClassInfo->typeParams)
            {
                auto name = templateParam->getName();
                auto found = typeClassInfo->typeParamsWithArgs.find(name);
                if (found != typeClassInfo->typeParamsWithArgs.end())
                {
                    inferType(location, getNamedGenericType(found->getValue().first->getName()),
                              found->getValue().second, results, genContext);
                }
            }

            return true;
        }

        return false;
    }

    bool MLIRGenImpl::tryInferInterface(mlir::Location location, mlir::Type templateType, mlir::Type concreteType,
                           StringMap<mlir::Type> &results, const GenContext &genContext)
    {
        auto tempInterface = dyn_cast<mlir_ts::InterfaceType>(templateType);
        auto typeInterface = dyn_cast<mlir_ts::InterfaceType>(concreteType);
        if (!tempInterface || !typeInterface)
        {
            return false;
        }

        auto typeInterfaceInfo = getInterfaceInfoByFullName(typeInterface.getName().getValue());
        if (auto tempInterfaceInfo = getInterfaceInfoByFullName(tempInterface.getName().getValue()))
        {
            for (auto &templateParam : tempInterfaceInfo->typeParamsWithArgs)
            {
                auto name = templateParam.getValue().first->getName();
                auto found = typeInterfaceInfo->typeParamsWithArgs.find(name);
                if (found != typeInterfaceInfo->typeParamsWithArgs.end())
                {
                    // TODO: convert GenericType -> AnyGenericType,  and NamedGenericType -> GenericType, and
                    // add 2 type Parameters to it Constrain, Default
                    inferType(location, templateParam.getValue().second, found->getValue().second, results, genContext);
                }
            }

            return true;
        }
        else if (auto tempGenericInterfaceInfo = getGenericInterfaceInfoByFullName(tempInterface.getName().getValue()))
        {
            for (auto &templateParam : tempGenericInterfaceInfo->typeParams)
            {
                auto name = templateParam->getName();
                auto found = typeInterfaceInfo->typeParamsWithArgs.find(name);
                if (found != typeInterfaceInfo->typeParamsWithArgs.end())
                {
                    inferType(location, getNamedGenericType(found->getValue().first->getName()),
                              found->getValue().second, results, genContext);
                }
            }

            return true;
        }

        return false;
    }

    bool MLIRGenImpl::tryInferArray(mlir::Location location, mlir::Type templateType, mlir::Type concreteType,
                       StringMap<mlir::Type> &results, const GenContext &genContext)
    {
        auto tempArray = dyn_cast<mlir_ts::ArrayType>(templateType);
        if (!tempArray)
        {
            return false;
        }

        if (auto typeArray = dyn_cast<mlir_ts::ArrayType>(concreteType))
        {
            inferType(location, tempArray.getElementType(), typeArray.getElementType(), results, genContext);
            return true;
        }

        if (auto typeArray = dyn_cast<mlir_ts::ConstArrayType>(concreteType))
        {
            inferType(location, tempArray.getElementType(), typeArray.getElementType(), results, genContext);
            return true;
        }

        return false;
    }

    bool MLIRGenImpl::tryInferTuple(mlir::Location location, mlir::Type templateType, mlir::Type concreteType,
                       StringMap<mlir::Type> &results, const GenContext &genContext)
    {
        auto tempTuple = dyn_cast<mlir_ts::TupleType>(templateType);
        if (!tempTuple)
        {
            return false;
        }

        if (auto typeTuple = dyn_cast<mlir_ts::TupleType>(concreteType))
        {
            return tryInferTupleFields(location, tempTuple, typeTuple, results, genContext);
        }

        if (auto typeTuple = dyn_cast<mlir_ts::ConstTupleType>(concreteType))
        {
            return tryInferTupleFields(location, tempTuple, typeTuple, results, genContext);
        }

        return false;
    }

    bool MLIRGenImpl::tryInferOptional(mlir::Location location, mlir::Type templateType, mlir::Type concreteType,
                          StringMap<mlir::Type> &results, const GenContext &genContext)
    {
        auto tempOpt = dyn_cast<mlir_ts::OptionalType>(templateType);
        if (!tempOpt)
        {
            return false;
        }

        if (auto typeOpt = dyn_cast<mlir_ts::OptionalType>(concreteType))
        {
            inferType(location, tempOpt.getElementType(), typeOpt.getElementType(), results, genContext);
            return true;
        }

        // optional -> value
        inferType(location, tempOpt.getElementType(), concreteType, results, genContext);
        return true;
    }

    bool MLIRGenImpl::tryInferFunction(mlir::Location location, mlir::Type templateType, mlir::Type concreteType,
                          StringMap<mlir::Type> &results, const GenContext &genContext)
    {
        if (!mth.isAnyFunctionType(templateType) || !mth.isAnyFunctionType(concreteType))
        {
            return false;
        }

        auto tempfuncType = mth.getParamsFromFuncRef(templateType);
        if (tempfuncType.size() > 0)
        {
            auto funcType = mth.getParamsFromFuncRef(concreteType);
            if (funcType.size() > 0)
            {
                inferTypeFuncType(location, tempfuncType, funcType, results, genContext);

                // lambda(return) -> lambda(return)
                auto tempfuncRetType = mth.getReturnsFromFuncRef(templateType);
                if (tempfuncRetType.size() > 0)
                {
                    auto funcRetType = mth.getReturnsFromFuncRef(concreteType);
                    if (funcRetType.size() > 0)
                    {
                        inferTypeFuncType(location, tempfuncRetType, funcRetType, results, genContext);
                    }
                }

                return true;
            }
        }

        return false;
    }

    bool MLIRGenImpl::tryInferUnion(mlir::Location location, mlir::Type templateType, mlir::Type concreteType,
                       StringMap<mlir::Type> &results, const GenContext &genContext)
    {
        auto tempUnionType = dyn_cast<mlir_ts::UnionType>(templateType);
        if (!tempUnionType)
        {
            return false;
        }

        if (auto typeUnionType = dyn_cast<mlir_ts::UnionType>(concreteType))
        {
            auto types = typeUnionType.getTypes();
            if (types.size() != tempUnionType.getTypes().size())
            {
                return true;
            }

            for (auto [index, tempSubType] : enumerate(tempUnionType.getTypes()))
            {
                inferType(location, tempSubType, types[index], results, genContext);
            }

            return true;
        }

        // TODO: review how to call functions such as: "function* Map<T, R>(a: T[] | Iterable<T>, f: (i: T) => R) { ... }"
        // special case when UnionType is used in generic method
        for (auto tempSubType : tempUnionType.getTypes())
        {
            auto count = results.size();
            inferType(location, tempSubType, concreteType, results, genContext);
            if (count < results.size())
            {
                return true;
            }
        }

        return true;
    }

    void MLIRGenImpl::inferType(mlir::Location location, mlir::Type templateType, mlir::Type concreteType, StringMap<mlir::Type> &results, const GenContext &genContext)
    {
        LLVM_DEBUG(llvm::dbgs() << "\n!! inferring \n\ttemplate type: " << templateType << ", \n\ttype: " << concreteType
                                << "\n";);

        if (!templateType || !concreteType)
        {
            // nothing todo here
            return;
        }

        if (templateType == concreteType)
        {
            // nothing todo here
            return;
        }

        if (tryInferNamedGeneric(location, templateType, concreteType, results))
        {
            return;
        }

        // class -> class
        if (tryInferClass(location, templateType, concreteType, results, genContext))
        {
            return;
        }

        // interface -> interface
        if (tryInferInterface(location, templateType, concreteType, results, genContext))
        {
            return;
        }

        // array -> array
        if (tryInferArray(location, templateType, concreteType, results, genContext))
        {
            return;
        }

        // tuple -> tuple
        if (tryInferTuple(location, templateType, concreteType, results, genContext))
        {
            return;
        }

        // optional -> optional / optional -> value
        if (tryInferOptional(location, templateType, concreteType, results, genContext))
        {
            return;
        }

        // lambda -> lambda
        if (tryInferFunction(location, templateType, concreteType, results, genContext))
        {
            return;
        }

        // union -> union / union -> value
        if (tryInferUnion(location, templateType, concreteType, results, genContext))
        {
            return;
        }

        // conditional type
        auto currentTemplateType = templateType;
        if (auto templateCondType = dyn_cast<mlir_ts::ConditionalType>(currentTemplateType))
        {
            inferType(location, templateCondType.getTrueType(), concreteType, results, genContext);
            currentTemplateType = templateCondType.getFalseType();
            inferType(location, currentTemplateType, concreteType, results, genContext);
        }

        // typeref -> type; note: intentionally also tests the false branch of a conditional type from above
        if (auto tempTypeRefType = dyn_cast<mlir_ts::TypeReferenceType>(currentTemplateType))
        {
            inferType(location, getTypeByTypeReference(location, tempTypeRefType, genContext), concreteType, results, genContext);
        }
    }

    void MLIRGenImpl::inferTypeFuncType(mlir::Location location, mlir::ArrayRef<mlir::Type> tempfuncType, mlir::ArrayRef<mlir::Type> funcType,
                           StringMap<mlir::Type> &results, const GenContext &genContext)
    {
        if (tempfuncType.size() != funcType.size())
        {
            return;
        }

        for (auto paramIndex = 0; paramIndex < tempfuncType.size(); paramIndex++)
        {
            auto currentTemplateType = tempfuncType[paramIndex];
            auto currentType = funcType[paramIndex];
            inferType(location, currentTemplateType, currentType, results, genContext);
        }
    }

    bool MLIRGenImpl::isGenericFunctionReference(mlir::Value functionRefValue)
    {
        auto currValue = functionRefValue;
        if (auto createBoundFunctionOp = currValue.getDefiningOp<mlir_ts::CreateBoundFunctionOp>())
        {
            currValue = createBoundFunctionOp.getFunc();
        }

        if (auto symbolOp = currValue.getDefiningOp<mlir_ts::SymbolRefOp>())
        {
            return symbolOp->hasAttrOfType<mlir::BoolAttr>(GENERIC_ATTR_NAME);
        }

        return false;
    }

    mlir::Type MLIRGenImpl::instantiateSpecializedFunctionTypeHelper(mlir::Location location, mlir::Value functionRefValue,
                                                        mlir::Type recieverType, bool discoverReturnType,
                                                        const GenContext &genContext)
    {
        auto currValue = functionRefValue;
        if (auto createBoundFunctionOp = currValue.getDefiningOp<mlir_ts::CreateBoundFunctionOp>())
        {
            currValue = createBoundFunctionOp.getFunc();
        }

        if (auto symbolOp = currValue.getDefiningOp<mlir_ts::SymbolRefOp>())
        {
            auto functionName = symbolOp.getIdentifier();

            // it is not generic arrow function
            auto functionGenericTypeInfo = getGenericFunctionInfoByFullName(functionName);

            MLIRNamespaceGuard nsGuard(currentNamespace);
            currentNamespace = functionGenericTypeInfo->elementNamespace;

            SourceFileScope sourceFileScope(*this, functionGenericTypeInfo->sourceFile, functionGenericTypeInfo->fileName);

            return instantiateSpecializedFunctionTypeHelper(location, functionGenericTypeInfo->functionDeclaration,
                                                            recieverType, discoverReturnType, genContext);
        }

        llvm_unreachable("not implemented");
    }

    mlir::Type MLIRGenImpl::instantiateSpecializedFunctionTypeHelper(mlir::Location location, FunctionLikeDeclarationBase funcDecl,
                                                        mlir::Type recieverType, bool discoverReturnType,
                                                        const GenContext &genContext)
    {
        GenContext funcGenContext(genContext);
        funcGenContext.receiverFuncType = recieverType;

        mlir::OpBuilder::InsertionGuard guard(builder);
        builder.setInsertionPointToStart(theModule.getBody());

        auto [result, funcOp] = getFuncArgTypesOfGenericMethod(funcDecl, {}, discoverReturnType, funcGenContext);
        if (mlir::failed(result))
        {
            if (!genContext.dummyRun)
            {
                emitError(location) << "can't instantiate specialized arrow function.";
            }

            return mlir::Type();
        }

        return funcOp->getFuncType();
    }

    void MLIRGenImpl::rollbackPostponedErrorMessages(mlir::SmallVector<std::unique_ptr<mlir::Diagnostic>> *postponedMessages, size_t size)
    {
        while (size < postponedMessages->size())
            postponedMessages->pop_back();
    }

    ValueOrLogicalResult MLIRGenImpl::instantiateSpecializedFunction(mlir::Location location,
        mlir::Value functionRefValue, mlir::Type recieverType, const GenContext &genContext)
    {
        auto currValue = functionRefValue;
        auto createBoundFunctionOp = currValue.getDefiningOp<mlir_ts::CreateBoundFunctionOp>();
        if (createBoundFunctionOp)
        {
            currValue = createBoundFunctionOp.getFunc();
        }

        LLVM_DEBUG(llvm::dbgs() << "\n!! spec. func ref: " << currValue << "\n";);

        auto symbolOp = currValue.getDefiningOp<mlir_ts::SymbolRefOp>();
        if (!symbolOp)
        {
            emitError(currValue.getLoc()) << "generic function should be used in 'const' variable declaration.";
            return mlir::failure();            
        }

        auto functionName = symbolOp.getIdentifier();

        // it is not generic arrow function
        auto functionGenericTypeInfo = getGenericFunctionInfoByFullName(functionName);
        if (!functionGenericTypeInfo)
        {
            emitError(location) << "can't find information about generic function. " << functionName;
            return mlir::failure();            
        }

        GenContext funcGenContext(genContext);
        funcGenContext.receiverFuncType = recieverType;
        funcGenContext.specialization = true;
        funcGenContext.instantiateSpecializedFunction = true;
        funcGenContext.typeParamsWithArgs = functionGenericTypeInfo->typeParamsWithArgs;

        auto savedErrorMessagesCount = funcGenContext.postponedMessages->size();

        if (mlir::failed(processTypeArgumentsFromFunctionParameters(
            functionGenericTypeInfo->functionDeclaration, funcGenContext)))
        {
            emitError(location) << "can't instantiate specialized function from function parameters.";
            return mlir::failure();
        }

        rollbackPostponedErrorMessages(funcGenContext.postponedMessages, savedErrorMessagesCount);

        {
            mlir::OpBuilder::InsertionGuard guard(builder);
            builder.setInsertionPointToStart(theModule.getBody());

            MLIRNamespaceGuard nsGuard(currentNamespace);
            currentNamespace = functionGenericTypeInfo->elementNamespace;
            
            SourceFileScope sourceFileScope(*this, functionGenericTypeInfo->sourceFile, functionGenericTypeInfo->fileName);

            auto [result, specFuncOp, specFuncName, isGeneric] =
                mlirGenFunctionLikeDeclaration(functionGenericTypeInfo->functionDeclaration, funcGenContext);
            if (mlir::failed(result))
            {
                emitError(location) << "can't instantiate specialized function.";
                return mlir::failure();
            }

            LLVM_DEBUG(llvm::dbgs() << "\n!! fixing spec. func: " << specFuncName << " type: ["
                                    << specFuncOp.getFunctionType() << "\n";);

            // update symbolref
            currValue.setType(specFuncOp.getFunctionType());
            if (functionName != specFuncName)
            {
                symbolOp.setIdentifier(specFuncName);
            }

            if (createBoundFunctionOp)
            {
                auto funcType = specFuncOp.getFunctionType();
                // fix create bound if any
                mlir::TypeSwitch<mlir::Type>(createBoundFunctionOp.getType())
                    .template Case<mlir_ts::BoundFunctionType>([&](auto boundFunc) {
                        functionRefValue.setType(getBoundFunctionType(funcType));
                    })
                    .template Case<mlir_ts::HybridFunctionType>([&](auto hybridFuncType) {
                        functionRefValue.setType(
                            mlir_ts::HybridFunctionType::get(builder.getContext(), funcType));
                    })
                    .Default([&](auto type) { llvm_unreachable("not implemented"); });
            }

            symbolOp->removeAttr(GENERIC_ATTR_NAME);

            builder.setInsertionPoint(symbolOp);

            // TODO: append captures vars to generic arrow function
            auto newOpWithCapture = resolveFunctionWithCapture(
                location, StringRef(specFuncName), specFuncOp.getFunctionType(), mlir::Value(), false, genContext);
            if (!newOpWithCapture.getDefiningOp<mlir_ts::SymbolRefOp>())
            {
                // symbolOp will be removed as unsed
                LLVM_DEBUG(llvm::dbgs() << "\n!! newOpWithCapture: " << newOpWithCapture << "\n";);
                return newOpWithCapture;
            }
            else
            {
                // newOpWithCapture will be removed as unsed
            }
        }

        return mlir::success();
    }

    mlir::LogicalResult MLIRGenImpl::appendInferredTypes(mlir::Location location,
                                            llvm::SmallVector<TypeParameterDOM::TypePtr> &typeParams,
                                            StringMap<mlir::Type> &inferredTypes, IsGeneric &anyNamedGenericType,
                                            GenContext &genericTypeGenContext,
                                            bool arrayMerge, bool noExtendsTest)
    {
        for (auto &pair : inferredTypes)
        {
            // find typeParam
            auto typeParamName = pair.getKey();
            auto inferredType = pair.getValue();
            auto found = std::find_if(typeParams.begin(), typeParams.end(),
                                      [&](auto &paramItem) { return paramItem->getName() == typeParamName; });
            if (found == typeParams.end())
            {
                LLVM_DEBUG(llvm::dbgs() << "\n!! can't find : " << typeParamName << " in type params: " << "\n";);
                LLVM_DEBUG(for (auto typeParam : typeParams) llvm::dbgs() << "\t!! type param: " << typeParam->getName() << "\n";);

                // experiment
                //auto typeParameterDOM = std::make_shared<TypeParameterDOM>(typeParamName.str());
                //genericTypeGenContext.typeParamsWithArgs[typeParamName] = {typeParameterDOM, inferredType};
                
                //return mlir::failure();
                // just ignore it
                continue;
            }

            auto typeParam = (*found);

            // we need to find out type and constrains is not allowing to do it
            auto [result, hasAnyNamedGenericType] =
                zipTypeParameterWithArgument(location, genericTypeGenContext.typeParamsWithArgs, typeParam,
                                             inferredType, noExtendsTest, genericTypeGenContext, true, arrayMerge);
            if (mlir::failed(result))
            {
                return mlir::failure();
            }

            if (hasAnyNamedGenericType == IsGeneric::True)
            {
                anyNamedGenericType = hasAnyNamedGenericType;
            }
        }

        return mlir::success();
    }

    std::pair<mlir::LogicalResult, bool> MLIRGenImpl::resolveGenericParamFromFunctionCall(mlir::Location location, mlir::Type paramType, mlir::Value argOp, int paramIndex,
        GenericFunctionInfo::TypePtr functionGenericTypeInfo, IsGeneric &anyNamedGenericType,  GenContext &genericTypeGenContext)
    {
        if (paramType == argOp.getType())
        {
            return {mlir::success(), true};
        }

        StringMap<mlir::Type> inferredTypes;
        inferType(location, paramType, argOp.getType(), inferredTypes, genericTypeGenContext);
        if (mlir::failed(appendInferredTypes(location, functionGenericTypeInfo->typeParams, inferredTypes, anyNamedGenericType,
                                                genericTypeGenContext, false, true)))
        {
            return {mlir::failure(), true};
        }

        if (isGenericFunctionReference(argOp))
        {
            GenContext typeGenContext(genericTypeGenContext);
            typeGenContext.dummyRun = true;
            auto recreatedFuncType = instantiateSpecializedFunctionTypeHelper(
                location, functionGenericTypeInfo->functionDeclaration, mlir::Type(), false,
                typeGenContext);
            if (!recreatedFuncType)
            {
                // next param
                return {mlir::failure(), true};
            }

            LLVM_DEBUG(llvm::dbgs()
                            << "\n!! instantiate specialized  type function: '"
                            << functionGenericTypeInfo->name << "' type: " << recreatedFuncType << "\n";);

            auto recreatedParamType = mth.getParamFromFuncRef(recreatedFuncType, paramIndex);

            LLVM_DEBUG(llvm::dbgs()
                            << "\n!! param type for arrow func[" << paramIndex << "]: " << recreatedParamType << "\n";);

            auto newArrowFuncType = instantiateSpecializedFunctionTypeHelper(location, argOp, recreatedParamType,
                                                                                true, genericTypeGenContext);

            LLVM_DEBUG(llvm::dbgs() << "\n!! instantiate specialized arrow type function: "
                                    << newArrowFuncType << "\n";);

            if (!newArrowFuncType)
            {
                return {mlir::failure(), true};
            }

            // infer second type when ArrowType is fully built
            StringMap<mlir::Type> inferredTypes;
            inferType(location, paramType, newArrowFuncType, inferredTypes, genericTypeGenContext);
            if (mlir::failed(appendInferredTypes(location, functionGenericTypeInfo->typeParams, inferredTypes, anyNamedGenericType,
                                                    genericTypeGenContext, false, true)))
            {
                return {mlir::failure(), false};
            }
        }

        return {mlir::success(), true};
    }

    mlir::LogicalResult MLIRGenImpl::resolveGenericParamsFromFunctionCall(mlir::Location location,
                                                             GenericFunctionInfo::TypePtr functionGenericTypeInfo,
                                                             NodeArray<TypeNode> typeArguments,
                                                             bool skipThisParam,
                                                             IsGeneric &anyNamedGenericType,
                                                             GenContext &genericTypeGenContext)
    {
        // add provided type arguments, ignoring defaults
        auto typeParams = functionGenericTypeInfo->typeParams;
        if (typeArguments)
        {
            auto [result, hasAnyNamedGenericType] = zipTypeParametersWithArgumentsNoDefaults(
                location, typeParams, typeArguments, genericTypeGenContext.typeParamsWithArgs, genericTypeGenContext);
            if (mlir::failed(result))
            {
                return mlir::failure();
            }

            if (hasAnyNamedGenericType == IsGeneric::True)
            {
                anyNamedGenericType = hasAnyNamedGenericType;
            }
        }

        // TODO: investigate, in [...].reduce, lambda function does not have funcOp, why?
        auto funcOp = functionGenericTypeInfo->funcOp;
        assert(funcOp);
        if (funcOp)
        {
            // TODO: we have func params.
            for (auto paramInfo : funcOp->getParams())
            {
                paramInfo->processed = false;
            }

            auto callOpsCount = genericTypeGenContext.callOperands.size();
            auto totalProcessed = 0;
            do
            {
                auto paramIndex = -1;
                auto processed = 0;
                auto startParamIndex = skipThisParam ? 1 : 0;
                auto skipCount = startParamIndex;
                for (auto paramInfo : funcOp->getParams())
                {
                    if (skipCount-- > 0)
                    {
                        continue;
                    }

                    paramIndex++;
                    if (paramInfo->processed)
                    {
                        continue;
                    }

                    auto paramType = paramInfo->getType();

                    if (callOpsCount <= paramIndex)
                    {
                        // there is no more ops; mark processed so the param is counted once -
                        // recounting it every round inflated totalProcessed past the termination
                        // equality below and spun the loop into the "loop detected" guard
                        if (paramInfo->getIsOptional() || isa<mlir_ts::OptionalType>(paramType))
                        {
                            paramInfo->processed = true;
                            processed++;
                            continue;
                        }

                        if (paramInfo->getIsMultiArgsParam())
                        {
                            paramInfo->processed = true;
                            processed++;
                            continue;
                        }

                        break;
                    }

                    auto argOp = genericTypeGenContext.callOperands[paramIndex];

                    LLVM_DEBUG(llvm::dbgs()
                        << "\n!! resolving param for generic function: '"
                        << functionGenericTypeInfo->name << "'\n\t parameter #" << paramIndex << " type: [ " << paramType << " ] \n\t argument type: [ " << argOp.getType() << " ]\n";);

                    if (!paramInfo->getIsMultiArgsParam())
                    {
                        auto [result, cont] = resolveGenericParamFromFunctionCall(
                            location, paramType, argOp, paramIndex + startParamIndex, functionGenericTypeInfo, anyNamedGenericType, genericTypeGenContext);
                        if (mlir::succeeded(result))
                        {
                            paramInfo->processed = true;
                            processed++;
                        }
                        else if (!cont)
                        {
                            return mlir::failure();
                        }
                    }
                    else
                    {
                        struct ArrayInfo arrayInfo{};
                        for (auto varArgIndex = paramIndex; varArgIndex < callOpsCount; varArgIndex++)
                        {
                            auto argOp = genericTypeGenContext.callOperands[varArgIndex];

                            accumulateArrayItemType(location, argOp.getType(), arrayInfo);                            
                        }

                        mlir::Type arrayType = getArrayType(arrayInfo.accumulatedArrayElementType);

                        StringMap<mlir::Type> inferredTypes;
                        inferType(location, paramType, arrayType, inferredTypes, genericTypeGenContext);
                        if (mlir::failed(appendInferredTypes(location, functionGenericTypeInfo->typeParams, inferredTypes, anyNamedGenericType,
                                                                genericTypeGenContext, true)))
                        {
                            return mlir::failure();
                        }                        

                        paramInfo->processed = true;
                        processed++;
                    }
                }

                if (processed == 0)
                {
                    // no progress in a full round: some params (e.g. a callback typed by a
                    // type param that only gets its value from a default) can't be inferred
                    // here; the default zipping and the completeness check below decide
                    // whether that is an error
                    break;
                }

                totalProcessed += processed;

                if (totalProcessed == funcOp->getParams().size() - startParamIndex)
                {
                    break;
                }

                if (totalProcessed > funcOp->getParams().size() + 100)
                {
                    // defensive only: with params counted exactly once this is unreachable
                    emitError(location) << "loop detected.";
                    return mlir::failure();
                }
            } while (true);
        }

        // add default params if not provided
        auto [resultDefArg, hasNamedGenericType] = zipTypeParametersWithDefaultArguments(
            location, typeParams, typeArguments, genericTypeGenContext.typeParamsWithArgs, genericTypeGenContext);
        if (mlir::failed(resultDefArg))
        {
            return mlir::failure();
        }

        if (hasNamedGenericType == IsGeneric::True)
        {
            anyNamedGenericType = hasNamedGenericType;
        }

        // TODO: check if all typeParams are there
        if (genericTypeGenContext.typeParamsWithArgs.size() < typeParams.size())
        {
            // no resolve needed, this type without param
            emitError(location) << "not all types could be inferred";
            return mlir::failure();
        }

        return mlir::success();
    }

    std::tuple<mlir::LogicalResult, mlir_ts::FunctionType, std::string> MLIRGenImpl::instantiateSpecializedFunction(
        mlir::Location location, StringRef name, NodeArray<TypeNode> typeArguments, bool skipThisParam, 
        SmallVector<mlir::Value, 4> &operands, const GenContext &genContext)
    {
        // local copy so the 'this'-type override below stays scoped to this instantiation
        GenContext instantiateGenContext(genContext);

        auto functionGenericTypeInfo = getGenericFunctionInfoByFullName(name);
        if (functionGenericTypeInfo)
        {
            if (functionGenericTypeInfo->functionDeclaration == SyntaxKind::ArrowFunction
                || functionGenericTypeInfo->functionDeclaration == SyntaxKind::FunctionExpression)
            {
                // we need to avoid wrong redeclaration of arrow functions (when thisType is provided it will add THIS parameter as first)
                instantiateGenContext.thisType = nullptr;
            }

            MLIRNamespaceGuard ng(currentNamespace);
            currentNamespace = functionGenericTypeInfo->elementNamespace;

            SourceFileScope sourceFileScope(*this, functionGenericTypeInfo->sourceFile, functionGenericTypeInfo->fileName);

            auto anyNamedGenericType = IsGeneric::False;

            // step 1, add type arguments first
            GenContext genericTypeGenContext(instantiateGenContext);
            genericTypeGenContext.specialization = true;
            genericTypeGenContext.instantiateSpecializedFunction = true;
            genericTypeGenContext.typeParamsWithArgs = functionGenericTypeInfo->typeParamsWithArgs;
            genericTypeGenContext.thisType = functionGenericTypeInfo->thisType; // to support methods
            genericTypeGenContext.thisClassType = functionGenericTypeInfo->thisClassType; // to support methods

            auto typeParams = functionGenericTypeInfo->typeParams;
            if (typeArguments && typeParams.size() == typeArguments.size())
            {
                // create typeParamsWithArgs from typeArguments
                auto [result, hasAnyNamedGenericType] = zipTypeParametersWithArguments(
                    location, typeParams, typeArguments, genericTypeGenContext.typeParamsWithArgs, instantiateGenContext);
                if (mlir::failed(result))
                {
                    return {mlir::failure(), mlir_ts::FunctionType(), ""};
                }

                if (hasAnyNamedGenericType == IsGeneric::True)
                {
                    anyNamedGenericType = hasAnyNamedGenericType;
                }
            }
            else if (genericTypeGenContext.callOperands.size() > 0 ||
                     functionGenericTypeInfo->functionDeclaration->parameters.size() > 0)
            {
                auto result =
                    resolveGenericParamsFromFunctionCall(location, functionGenericTypeInfo, typeArguments,
                                                         skipThisParam, anyNamedGenericType, genericTypeGenContext);
                if (mlir::failed(result))
                {
                    return {mlir::failure(), mlir_ts::FunctionType(), ""};
                }
            }
            else
            {
                llvm_unreachable("not implemented");
            }

            // we need to wide all types when initializing function
            // TODO: add checking constraints
            for (auto &typeParam : genericTypeGenContext.typeParamsWithArgs)
            {
                auto &typeParamValue = typeParam.getValue();
                auto typeInfo = std::get<0>(typeParamValue);
                auto name = typeInfo->getName();
                auto type = std::get<1>(typeParamValue);
                auto widenType = mth.wideStorageType(type);
                genericTypeGenContext.typeParamsWithArgs[name] = std::make_pair(typeInfo, widenType);

                if (typeParam.getValue().first->getConstraint())
                {
                    auto reason = testConstraint(location, genericTypeGenContext.typeParamsWithArgs, typeParamValue.first, widenType, instantiateGenContext);
                    if (reason == Reason::Failure)
                    {
                        LLVM_DEBUG(llvm::dbgs() << "\n!! skip. failed. should be resolved later\n";);
                        return {mlir::failure(), mlir_ts::FunctionType(), ""};
                    }

                    if (reason == Reason::FailedConstraint)
                    {
                        if (functionGenericTypeInfo->funcType.getNumResults() > 0
                            && mlir::isa<mlir_ts::TypePredicateType>(functionGenericTypeInfo->funcType.getResult(0)))
                        {
                            return {
                                mlir::success(), 
                                mlir_ts::FunctionType::get(builder.getContext(), {}, { getBooleanLiteral(false) }, false), 
                                ""
                            };
                        }

                        return {mlir::failure(), mlir_ts::FunctionType(), ""};
                    }
                }
            }

            LLVM_DEBUG(llvm::dbgs() << "\n!! instantiate specialized function: " << functionGenericTypeInfo->name
                                    << " ";
                       for (auto &typeParam
                            : genericTypeGenContext.typeParamsWithArgs) llvm::dbgs()
                       << " param: " << std::get<0>(typeParam.getValue())->getName()
                       << " type: " << std::get<1>(typeParam.getValue());
                       llvm::dbgs() << "\n";);

            LLVM_DEBUG(if (genericTypeGenContext.typeAliasMap.size()) llvm::dbgs() << "\n!! type alias: ";
                       for (auto &typeAlias
                            : genericTypeGenContext.typeAliasMap) llvm::dbgs()
                       << " name: " << typeAlias.getKey() << " type: " << typeAlias.getValue();
                       llvm::dbgs() << "\n";);

            // revalidate all types
            if (anyNamedGenericType == IsGeneric::True)
            {
                anyNamedGenericType = IsGeneric::False;
                for (auto &typeParamWithArg : genericTypeGenContext.typeParamsWithArgs)
                {
                    if (mth.isGenericType(std::get<1>(typeParamWithArg.second)))
                    {
                        anyNamedGenericType = IsGeneric::True;
                    }
                }
            }

            if (anyNamedGenericType == IsGeneric::False)
            {
                if (functionGenericTypeInfo->processing)
                {
                    auto [fullName, name] =
                        getNameOfFunction(functionGenericTypeInfo->functionDeclaration, genericTypeGenContext);

                    auto funcType = lookupFunctionTypeMap(fullName);
                    if (funcType)
                    {
                        return {mlir::success(), funcType, fullName};
                    }

                    if (instantiateGenContext.allowPartialResolve)
                    {
                        return {mlir::success(), mlir_ts::FunctionType(), fullName};
                    }

                    return {mlir::failure(), mlir_ts::FunctionType(), ""};
                }

                // create new instance of function with TypeArguments
                functionGenericTypeInfo->processing = true;
                auto [result, funcOp, funcName, isGeneric] =
                    mlirGenFunctionLikeDeclaration(functionGenericTypeInfo->functionDeclaration, genericTypeGenContext);
                functionGenericTypeInfo->processing = false;
                if (mlir::failed(result))
                {
                    return {mlir::failure(), mlir_ts::FunctionType(), ""};
                }

                functionGenericTypeInfo->processed = true;

                // instatiate all ArrowFunctions which are not yet instantiated
                auto opIndex = skipThisParam ? 0 : -1;
                // TODO: this is hack, somehow we have difference between operands and call Operands due to CreateExtentionsFunction call
                // review example raytrace.ts function addLight in getNaturalColor (due to captured params)
                long operandsShift = static_cast<long>(operands.size()) - static_cast<long>(instantiateGenContext.callOperands.size());
                for (auto [callOpIndex, op] : enumerate(instantiateGenContext.callOperands))
                {
                    opIndex++;
                    if (isGenericFunctionReference(op))
                    {
                        LLVM_DEBUG(llvm::dbgs() << "\n!! delayed arrow func instantiation for func type: "
                                                << funcOp.getFunctionType() << "\n";);
                        auto result = instantiateSpecializedFunction(
                            location, op, funcOp.getFunctionType().getInput(opIndex), instantiateGenContext);
                        if (mlir::failed(result))
                        {
                            return {mlir::failure(), mlir_ts::FunctionType(), ""};
                        }

                        auto resultValue = V(result);
                        if (resultValue)
                        {
                            operands[callOpIndex + operandsShift] = resultValue;
                        }
                    }
                }

                return {mlir::success(), funcOp.getFunctionType(), funcOp.getName().str()};
            }

            emitError(location) << "can't instantiate specialized function [" << name << "].";
            return {mlir::failure(), mlir_ts::FunctionType(), ""};
        }

        emitError(location) << "can't find generic [" << name << "] function.";
        return {mlir::failure(), mlir_ts::FunctionType(), ""};
    }

    std::pair<mlir::LogicalResult, FunctionPrototypeDOM::TypePtr> MLIRGenImpl::getFuncArgTypesOfGenericMethod(
        FunctionLikeDeclarationBase functionLikeDeclarationAST, ArrayRef<TypeParameterDOM::TypePtr> typeParams,
        bool discoverReturnType, const GenContext &genContext)
    {
        GenContext funcGenContext(genContext);
        funcGenContext.discoverParamsOnly = !discoverReturnType;

        // we need to map generic parameters to generic types to be able to resolve function parameters which
        // are not generic
        for (auto typeParam : typeParams)
        {
            funcGenContext.typeAliasMap.insert({typeParam->getName(), getNamedGenericType(typeParam->getName())});
        }

        auto [funcOp, funcProto, result, isGenericType] =
            mlirGenFunctionPrototype(functionLikeDeclarationAST, funcGenContext);
        if (mlir::failed(result) || !funcOp)
        {
            return {mlir::failure(), {}};
        }

        LLVM_DEBUG(llvm::dbgs() << "\n!! func name: " << funcProto->getName()
                                << ", Op type (resolving from operands): " << funcOp.getFunctionType() << "\n";);

        LLVM_DEBUG(llvm::dbgs() << "\n!! func args: "; for (auto [index, paramInfo]
                                                                            : enumerate(funcProto->getParams())) {
            llvm::dbgs() << "\n_ " << paramInfo->getName() << ": " << paramInfo->getType() << " = (" << index << ") ";
            if (genContext.callOperands.size() > index)
                llvm::dbgs() << genContext.callOperands[index];
            llvm::dbgs() << "\n";
        });

        return {mlir::success(), funcProto};
    }

    std::pair<mlir::LogicalResult, mlir::Type> MLIRGenImpl::instantiateSpecializedClassType(mlir::Location location,
                                                                               mlir_ts::ClassType genericClassType,
                                                                               NodeArray<TypeNode> typeArguments,
                                                                               const GenContext &genContext,
                                                                               bool allowNamedGenerics)
    {
        auto fullNameGenericClassTypeName = genericClassType.getName().getValue();
        auto genericClassInfo = getGenericClassInfoByFullName(fullNameGenericClassTypeName);
        if (genericClassInfo)
        {
            MLIRNamespaceGuard ng(currentNamespace);
            currentNamespace = genericClassInfo->elementNamespace;

            SourceFileScope sourceFileScope(*this, genericClassInfo->sourceFile, genericClassInfo->fileName);

            GenContext genericTypeGenContext(genContext);
            genericTypeGenContext.instantiateSpecializedFunction = false;
            auto typeParams = genericClassInfo->typeParams;
            auto [result, hasAnyNamedGenericType] = zipTypeParametersWithArguments(
                location, typeParams, typeArguments, genericTypeGenContext.typeParamsWithArgs, genContext);
            if (mlir::failed(result) && hasAnyNamedGenericType == IsGeneric::NoDefaults)
            {
                // can't instantiate generic type, so check if normal type without generic types exists
                return {mlir::success(), mlir::Type()};
            }

            if (mlir::failed(result) || (hasAnyNamedGenericType == IsGeneric::True && !allowNamedGenerics))
            {
                return {mlir::failure(), mlir::Type()};
            }

            LLVM_DEBUG(llvm::dbgs() << "\n!! instantiate specialized class: " << fullNameGenericClassTypeName << " ";
                       for (auto &typeParam
                            : genericTypeGenContext.typeParamsWithArgs) llvm::dbgs()
                       << " param: " << std::get<0>(typeParam.getValue())->getName()
                       << " type: " << std::get<1>(typeParam.getValue());
                       llvm::dbgs() << "\n";);

            LLVM_DEBUG(if (genericTypeGenContext.typeAliasMap.size()) llvm::dbgs() << "\n!! type alias: ";
                       for (auto &typeAlias
                            : genericTypeGenContext.typeAliasMap) llvm::dbgs()
                       << " name: " << typeAlias.getKey() << " type: " << typeAlias.getValue();
                       llvm::dbgs() << "\n";);

            // create new instance of interface with TypeArguments
            if (mlir::failed(std::get<0>(mlirGen(genericClassInfo->classDeclaration, genericTypeGenContext))))
            {
                return {mlir::failure(), mlir::Type()};
            }

            // get instance of generic interface type
            auto specType = getSpecializationClassType(genericClassInfo, genericTypeGenContext);
            return {mlir::success(), specType};
        }

        // special case: Array<T>
        // if (fullNameGenericClassTypeName == "Array" && typeArguments.size() == 1)
        // {
        //     auto arraySpecType = getEmbeddedTypeWithParam(fullNameGenericClassTypeName, typeArguments, genContext);
        //     return {mlir::success(), arraySpecType};
        // }

        // can't find generic instance
        return {mlir::success(), mlir::Type()};
    }

    std::pair<mlir::LogicalResult, mlir::Type> MLIRGenImpl::instantiateSpecializedClassType(mlir::Location location,
                                                                               mlir_ts::ClassType genericClassType,
                                                                               ArrayRef<mlir::Type> typeArguments,
                                                                               const GenContext &genContext,
                                                                               bool allowNamedGenerics)
    {
        auto fullNameGenericClassTypeName = genericClassType.getName().getValue();
        auto genericClassInfo = getGenericClassInfoByFullName(fullNameGenericClassTypeName);
        if (genericClassInfo)
        {
            MLIRNamespaceGuard ng(currentNamespace);
            currentNamespace = genericClassInfo->elementNamespace;

            SourceFileScope sourceFileScope(*this, genericClassInfo->sourceFile, genericClassInfo->fileName);

            GenContext genericTypeGenContext(genContext);
            genericTypeGenContext.instantiateSpecializedFunction = false;
            auto typeParams = genericClassInfo->typeParams;
            auto [result, hasAnyNamedGenericType] = zipTypeParametersWithArguments(
                location, typeParams, typeArguments, genericTypeGenContext.typeParamsWithArgs, genContext);
            if (mlir::failed(result) || (hasAnyNamedGenericType == IsGeneric::True && !allowNamedGenerics))
            {
                return {mlir::failure(), mlir::Type()};
            }

            LLVM_DEBUG(llvm::dbgs() << "\n!! instantiate specialized class: " << fullNameGenericClassTypeName << " ";
                       for (auto &typeParam
                            : genericTypeGenContext.typeParamsWithArgs) llvm::dbgs()
                       << " param: " << std::get<0>(typeParam.getValue())->getName()
                       << " type: " << std::get<1>(typeParam.getValue());
                       llvm::dbgs() << "\n";);

            LLVM_DEBUG(if (genericTypeGenContext.typeAliasMap.size()) llvm::dbgs() << "\n!! type alias: ";
                       for (auto &typeAlias
                            : genericTypeGenContext.typeAliasMap) llvm::dbgs()
                       << " name: " << typeAlias.getKey() << " type: " << typeAlias.getValue();
                       llvm::dbgs() << "\n";);

            static auto count = 0;
            count++;
            if (count > 99)
            {
                count--;
                emitError(location) << "can't instantiate type. '" << genericClassType
                                    << "'. Circular initialization is detected.";
                return {mlir::failure(), mlir::Type()};

                // std::string s;
                // s += "can't instantiate type. '";
                // s += fullNameGenericClassTypeName;
                // s += "'. Circular initialization is detected.";
                // llvm_unreachable(s.c_str());
            }

            auto res = std::get<0>(mlirGen(genericClassInfo->classDeclaration, genericTypeGenContext));
            count--;

            // create new instance of class with TypeArguments
            if (mlir::failed(res))
            {
                return {mlir::failure(), mlir::Type()};
            }

            // get instance of generic interface type
            auto specType = getSpecializationClassType(genericClassInfo, genericTypeGenContext);
            return {mlir::success(), specType};
        }

        // can't find generic instance
        return {mlir::success(), mlir::Type()};
    }

    std::pair<mlir::LogicalResult, mlir::Type> MLIRGenImpl::instantiateSpecializedInterfaceType(
        mlir::Location location, mlir_ts::InterfaceType genericInterfaceType, NodeArray<TypeNode> typeArguments,
        const GenContext &genContext, bool allowNamedGenerics)
    {
        auto fullNameGenericInterfaceTypeName = genericInterfaceType.getName().getValue();
        auto genericInterfaceInfo = getGenericInterfaceInfoByFullName(fullNameGenericInterfaceTypeName);
        if (genericInterfaceInfo)
        {
            MLIRNamespaceGuard ng(currentNamespace);
            currentNamespace = genericInterfaceInfo->elementNamespace;

            SourceFileScope sourceFileScope(*this, genericInterfaceInfo->sourceFile, genericInterfaceInfo->fileName);

            GenContext genericTypeGenContext(genContext);
            auto typeParams = genericInterfaceInfo->typeParams;
            auto [result, hasAnyNamedGenericType] = zipTypeParametersWithArguments(
                location, typeParams, typeArguments, genericTypeGenContext.typeParamsWithArgs, genContext);
            if (mlir::failed(result) || (hasAnyNamedGenericType == IsGeneric::True && !allowNamedGenerics))
            {
                return {mlir::failure(), mlir::Type()};
            }

            LLVM_DEBUG(llvm::dbgs() << "\n!! instantiate specialized interface: " << fullNameGenericInterfaceTypeName
                                    << " ";
                       for (auto &typeParam
                            : genericTypeGenContext.typeParamsWithArgs) llvm::dbgs()
                       << " param: " << std::get<0>(typeParam.getValue())->getName()
                       << " type: " << std::get<1>(typeParam.getValue());
                       llvm::dbgs() << "\n";);

            LLVM_DEBUG(if (genericTypeGenContext.typeAliasMap.size()) llvm::dbgs() << "\n!! type alias: ";
                       for (auto &typeAlias
                            : genericTypeGenContext.typeAliasMap) llvm::dbgs()
                       << " name: " << typeAlias.getKey() << " type: " << typeAlias.getValue();
                       llvm::dbgs() << "\n";);

            // create new instance of interface with TypeArguments
            if (mlir::failed(mlirGen(genericInterfaceInfo->interfaceDeclaration, genericTypeGenContext)))
            {
                // return mlir::Type();
                // type can't be resolved, so return generic base type
                //return {mlir::success(), genericInterfaceInfo->interfaceType};
                return {mlir::failure(), mlir::Type()};
            }

            // get instance of generic interface type
            auto specType = getSpecializationInterfaceType(genericInterfaceInfo, genericTypeGenContext);
            return {mlir::success(), specType};
        }

        // can't find generic instance
        return {mlir::success(), mlir::Type()};
    }

    ValueOrLogicalResult MLIRGenImpl::mlirGenSpecialized(mlir::Location location, mlir::Value genResult,
                                            NodeArray<TypeNode> typeArguments, SmallVector<mlir::Value, 4> &operands,
                                            const GenContext &genContext)
    {
        // in case it is generic arrow function
        auto currValue = genResult;

        // in case of this.generic_func<T>();
        if (auto extensFuncRef = currValue.getDefiningOp<mlir_ts::CreateExtensionFunctionOp>())
        {
            currValue = extensFuncRef.getFunc();

            SmallVector<mlir::Value, 4> operandsSpec;
            operandsSpec.push_back(extensFuncRef.getThisVal());
            operandsSpec.append(genContext.callOperands.begin(), genContext.callOperands.end());

            GenContext specGenContext(genContext);
            specGenContext.callOperands = operandsSpec;

            auto newFuncRefOrLogicResult = mlirGenSpecialized(location, currValue, typeArguments, operands, specGenContext);
            EXIT_IF_FAILED(newFuncRefOrLogicResult)
            if (newFuncRefOrLogicResult && currValue != newFuncRefOrLogicResult)
            {
                mlir::Value newFuncRefValue = newFuncRefOrLogicResult;

                // special case to work with interfaces
                // TODO: finish it, bug
                auto thisRef = extensFuncRef.getThisVal();
                auto funcType = mlir::cast<mlir_ts::FunctionType>(newFuncRefValue.getType());

                mlir::Value newExtensionFuncVal = builder.create<mlir_ts::CreateExtensionFunctionOp>(
                                location, getExtensionFunctionType(funcType), thisRef, newFuncRefValue);

                extensFuncRef.erase();

                return newExtensionFuncVal;
            }
            else
            {
                return genResult;
            }
        }

        if (currValue.getDefiningOp()->hasAttrOfType<mlir::BoolAttr>(GENERIC_ATTR_NAME))
        {
            // create new function instance
            GenContext initSpecGenContext(genContext);
            initSpecGenContext.forceDiscover = true;
            initSpecGenContext.thisType = mlir::Type();

            auto skipThisParam = false;
            mlir::Value thisValue;
            StringRef funcName;
            if (auto symbolOp = currValue.getDefiningOp<mlir_ts::SymbolRefOp>())
            {
                funcName = symbolOp.getIdentifierAttr().getValue();
            }
            else if (auto thisSymbolOp = currValue.getDefiningOp<mlir_ts::ThisSymbolRefOp>())
            {
                funcName = thisSymbolOp.getIdentifierAttr().getValue();
                skipThisParam = true;
                thisValue = thisSymbolOp.getThisVal();
                initSpecGenContext.thisType = thisValue.getType();
            }
            else
            {
                llvm_unreachable("not implemented");
            }

            auto [result, funcType, funcSymbolName] =
                instantiateSpecializedFunction(location, funcName, typeArguments, skipThisParam, operands, initSpecGenContext);
            if (mlir::failed(result))
            {
                emitError(location) << "can't instantiate function. '" << funcName
                                    << "' not all generic types can be identified";
                return mlir::failure();
            }

            if (!funcType && genContext.allowPartialResolve)
            {
                return mlir::success();
            }

            return resolveFunctionWithCapture(location, StringRef(funcSymbolName), funcType, thisValue, false, genContext);
        }

        if (auto classOp = genResult.getDefiningOp<mlir_ts::ClassRefOp>())
        {
            auto classType = classOp.getType();
            auto [result, specType] = instantiateSpecializedClassType(location, classType, typeArguments, genContext);
            if (mlir::failed(result))
            {
                return mlir::failure();
            }

            if (auto specClassType = dyn_cast_or_null<mlir_ts::ClassType>(specType))
            {
                return V(builder.create<mlir_ts::ClassRefOp>(
                    location, specClassType,
                    mlir::FlatSymbolRefAttr::get(builder.getContext(), specClassType.getName().getValue())));
            }

            if (specType)
            {
                return V(builder.create<mlir_ts::TypeRefOp>(location, specType));
            }

            return genResult;
        }

        if (auto ifaceOp = genResult.getDefiningOp<mlir_ts::InterfaceRefOp>())
        {
            auto interfaceType = ifaceOp.getType();
            auto [result, specType] =
                instantiateSpecializedInterfaceType(location, interfaceType, typeArguments, genContext);
            if (auto specInterfaceType = dyn_cast_or_null<mlir_ts::InterfaceType>(specType))
            {
                return V(builder.create<mlir_ts::InterfaceRefOp>(
                    location, specInterfaceType,
                    mlir::FlatSymbolRefAttr::get(builder.getContext(), specInterfaceType.getName().getValue())));
            }

            return genResult;
        }

        return genResult;
    }

    ValueOrLogicalResult MLIRGenImpl::mlirGen(Expression expression, NodeArray<TypeNode> typeArguments, const GenContext &genContext)
    {
        auto result = mlirGen(expression, genContext);
        EXIT_IF_FAILED_OR_NO_VALUE(result)
        auto genResult = V(result);
        // we can't leave here, template can have all parameters as default
        // if (typeArguments.size() == 0)
        // {
        //     return genResult;
        // }

        auto location = loc(expression);

        SmallVector<mlir::Value, 4> emptyOperands;
        return mlirGenSpecialized(location, genResult, typeArguments, emptyOperands, genContext);
    }

    ValueOrLogicalResult MLIRGenImpl::mlirGen(ExpressionWithTypeArguments expressionWithTypeArgumentsAST,
                                 const GenContext &genContext)
    {
        return mlirGen(expressionWithTypeArgumentsAST->expression, expressionWithTypeArgumentsAST->typeArguments,
                       genContext);
    }

} // namespace mlirgen
} // namespace typescript
