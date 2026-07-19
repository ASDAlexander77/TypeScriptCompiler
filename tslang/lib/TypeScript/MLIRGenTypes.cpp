// Type-resolution methods of MLIRGenImpl (see MLIRGenImpl.h).

#include "MLIRGenImpl.h"

namespace typescript
{
namespace mlirgen
{

    mlir::Type MLIRGenImpl::getType(Node typeReferenceAST, const GenContext &genContext)
    {
        auto kind = (SyntaxKind)typeReferenceAST;
        if (kind == SyntaxKind::BooleanKeyword)
        {
            return getBooleanType();
        }
        else if (kind == SyntaxKind::NumberKeyword)
        {
            return getNumberType();
        }
        else if (kind == SyntaxKind::BigIntKeyword)
        {
            return getBigIntType();
        }
        else if (kind == SyntaxKind::StringKeyword)
        {
            return getStringType();
        }
        else if (kind == SyntaxKind::VoidKeyword)
        {
            return getVoidType();
        }
        else if (kind == SyntaxKind::FunctionType)
        {
            return getFunctionType(typeReferenceAST.as<FunctionTypeNode>(), genContext);
        }
        else if (kind == SyntaxKind::ConstructorType)
        {
            // TODO: do I need to add flag to FunctionType to show that this is ConstructorType?
            return getConstructorType(typeReferenceAST.as<ConstructorTypeNode>(), genContext);
        }
        else if (kind == SyntaxKind::CallSignature)
        {
            return getCallSignature(typeReferenceAST.as<CallSignatureDeclaration>(), genContext);
        }
        else if (kind == SyntaxKind::MethodSignature)
        {
            return getMethodSignature(typeReferenceAST.as<MethodSignature>(), genContext);
        }
        else if (kind == SyntaxKind::ConstructSignature)
        {
            return getConstructSignature(typeReferenceAST.as<ConstructSignatureDeclaration>(), genContext);
        }
        else if (kind == SyntaxKind::IndexSignature)
        {
            return getIndexSignature(typeReferenceAST.as<IndexSignatureDeclaration>(), genContext);
        }
        else if (kind == SyntaxKind::TupleType)
        {
            return getTupleType(typeReferenceAST.as<TupleTypeNode>(), genContext);
        }
        else if (kind == SyntaxKind::TypeLiteral)
        {
            // TODO: review it, I think it should be ObjectType
            // return getObjectType(getTupleType(typeReferenceAST.as<TypeLiteralNode>(), genContext));
            return getTupleType(typeReferenceAST.as<TypeLiteralNode>(), genContext);
        }
        else if (kind == SyntaxKind::ArrayType)
        {
            return getArrayType(typeReferenceAST.as<ArrayTypeNode>(), genContext);
        }
        else if (kind == SyntaxKind::UnionType)
        {
            return getUnionType(typeReferenceAST.as<UnionTypeNode>(), genContext);
        }
        else if (kind == SyntaxKind::IntersectionType)
        {
            return getIntersectionType(typeReferenceAST.as<IntersectionTypeNode>(), genContext);
        }
        else if (kind == SyntaxKind::ParenthesizedType)
        {
            return getParenthesizedType(typeReferenceAST.as<ParenthesizedTypeNode>(), genContext);
        }
        else if (kind == SyntaxKind::LiteralType)
        {
            return getLiteralType(typeReferenceAST.as<LiteralTypeNode>());
        }
        else if (kind == SyntaxKind::TypeReference)
        {
            return getTypeByTypeReference(typeReferenceAST.as<TypeReferenceNode>(), genContext);
        }
        else if (kind == SyntaxKind::TypeQuery)
        {
            return getTypeByTypeQuery(typeReferenceAST.as<TypeQueryNode>(), genContext);
        }
        else if (kind == SyntaxKind::ObjectKeyword)
        {
            return getObjectType(getAnyType());
        }
        else if (kind == SyntaxKind::AnyKeyword)
        {
            return getAnyType();
        }
        else if (kind == SyntaxKind::UnknownKeyword)
        {
            // TODO: do I need to have special type?
            return getUnknownType();
        }
        else if (kind == SyntaxKind::SymbolKeyword)
        {
            return getSymbolType();
        }
        else if (kind == SyntaxKind::UndefinedKeyword)
        {
            return getUndefinedType();
        }
        else if (kind == SyntaxKind::NullKeyword)
        {
            return getNullType();
        }
        else if (kind == SyntaxKind::TypePredicate)
        {
            // in runtime it is boolean (it is needed to track types)
            return getTypePredicateType(typeReferenceAST.as<TypePredicateNode>(), genContext);
        }
        else if (kind == SyntaxKind::ThisType)
        {
            if (genContext.thisType)
            {
                return genContext.thisType;
            }
            
            NodeFactory nf(NodeFactoryFlags::None);
            auto thisType = evaluate(nf.createToken(SyntaxKind::ThisKeyword), genContext);
            LLVM_DEBUG(llvm::dbgs() << "\n!! this type from variable: [" << thisType << "]\n";);
            return thisType;
        }
        else if (kind == SyntaxKind::Unknown)
        {
            return getUnknownType();
        }
        else if (kind == SyntaxKind::ConditionalType)
        {
            return getConditionalType(typeReferenceAST.as<ConditionalTypeNode>(), genContext);
        }
        else if (kind == SyntaxKind::TypeOperator)
        {
            return getTypeOperator(typeReferenceAST.as<TypeOperatorNode>(), genContext);
        }
        else if (kind == SyntaxKind::IndexedAccessType)
        {
            return getIndexedAccessType(typeReferenceAST.as<IndexedAccessTypeNode>(), genContext);
        }
        else if (kind == SyntaxKind::MappedType)
        {
            return getMappedType(typeReferenceAST.as<MappedTypeNode>(), genContext);
        }
        else if (kind == SyntaxKind::TemplateLiteralType)
        {
            return getTemplateLiteralType(typeReferenceAST.as<TemplateLiteralTypeNode>(), genContext);
        }
        else if (kind == SyntaxKind::TypeParameter)
        {
            return getResolveTypeParameter(typeReferenceAST.as<TypeParameterDeclaration>(), genContext);
        }
        else if (kind == SyntaxKind::InferType)
        {
            return getInferType(loc(typeReferenceAST), typeReferenceAST.as<InferTypeNode>(), genContext);
        }
        else if (kind == SyntaxKind::OptionalType)
        {
            return getOptionalType(typeReferenceAST.as<OptionalTypeNode>(), genContext);
        }
        else if (kind == SyntaxKind::RestType)
        {
            return getRestType(typeReferenceAST.as<RestTypeNode>(), genContext);
        }
        else if (kind == SyntaxKind::NeverKeyword)
        {
            return getNeverType();
        }

        llvm_unreachable("not implemented type declaration");
        // return getAnyType();
    }

    mlir::Type MLIRGenImpl::getInferType(mlir::Location location, InferTypeNode inferTypeNodeAST, const GenContext &genContext)
    {
        auto type = getType(inferTypeNodeAST->typeParameter, genContext);
        if (!mlir::isa<mlir_ts::NamedGenericType>(type))
        {
            LLVM_DEBUG(llvm::dbgs() << "\n!! resolved infer type [" << type << "]\n";);
            // seems type has been resolved already in context
            return type;
        }

        auto inferType = getInferType(type);

        LLVM_DEBUG(llvm::dbgs() << "\n!! infer type [" << inferType << "]\n";);

        // TODO: review function 'extends' in MLIRTypeHelper with the same logic adding infer types to context

        if (genContext.inferTypes == nullptr)
        {
            emitError(location, "infer can be used in Conditional Type only");
            return mlir::Type();
        }

        auto &typeParamsWithArgs = *genContext.inferTypes;
        mth.appendInferTypeToContext(location, type, inferType, typeParamsWithArgs);

        return inferType;
    }

    mlir::Type MLIRGenImpl::getResolveTypeParameter(StringRef typeParamName, bool defaultType, const GenContext &genContext)
    {
        // to build generic type with generic names
        auto foundAlias = genContext.typeAliasMap.find(typeParamName);
        if (foundAlias != genContext.typeAliasMap.end())
        {
            auto type = (*foundAlias).getValue();
            // LLVM_DEBUG(llvm::dbgs() << "\n!! type gen. param as alias [" << typeParamName << "] -> [" << type
            //                         << "]\n";);
            return type;
        }

        auto found = genContext.typeParamsWithArgs.find(typeParamName);
        if (found != genContext.typeParamsWithArgs.end())
        {
            auto type = (*found).getValue().second;
            //LLVM_DEBUG(llvm::dbgs() << "\n!! type gen. param [" << typeParamName << "] -> [" << type << "]\n";);
            return type;
        }

        if (defaultType)
        {
            // unresolved generic
            return getNamedGenericType(typeParamName);
        }

        // name is not found
        return mlir::Type();
    }

    mlir::Type MLIRGenImpl::getResolveTypeParameter(TypeParameterDeclaration typeParameterDeclaration, const GenContext &genContext)
    {
        auto name = MLIRHelper::getName(typeParameterDeclaration->name);
        if (name.empty())
        {
            llvm_unreachable("not implemented");
            return mlir::Type();
        }

        return getResolveTypeParameter(name, true, genContext);
    }

    mlir::Type MLIRGenImpl::getTypeByTypeName(Node node, const GenContext &genContext)
    {
        if (node == SyntaxKind::Identifier)
        {
            auto name = MLIRHelper::getName(node);
            return resolveTypeByName(loc(node), name, genContext);
        }        
        else if (node == SyntaxKind::QualifiedName)
        {
            // TODO: it seems namespace access, can u optimize it somehow?
            auto result = mlirGen(node.as<QualifiedName>(), genContext);
            if (result.failed_or_no_value())
            {
                return mlir::Type();
            }

            auto val = V(result);
            return val.getType();
        }
        
        llvm_unreachable("not implemented");
    }

    mlir::Type MLIRGenImpl::getFirstTypeFromTypeArguments(NodeArray<TypeNode> &typeArguments, const GenContext &genContext)
    {
        return getType(typeArguments->front(), genContext);
    }

    mlir::Type MLIRGenImpl::getSecondTypeFromTypeArguments(NodeArray<TypeNode> &typeArguments, const GenContext &genContext)
    {
        return getType(typeArguments[1], genContext);
    }

    Reason MLIRGenImpl::testConstraint(mlir::Location location, llvm::StringMap<std::pair<TypeParameterDOM::TypePtr, mlir::Type>> &pairs,
        const ts::TypeParameterDOM::TypePtr &typeParam, mlir::Type type, const GenContext &genContext) {
        // we need to add current type into context to be able to use it in resolving "extends" constraints
        GenContext constraintGenContext(genContext);
        for (auto &typeParamWithArg : pairs)
        {
            constraintGenContext.typeParamsWithArgs.insert({typeParamWithArg.getKey(), typeParamWithArg.getValue()});
        }

        constraintGenContext.typeParamsWithArgs.insert({typeParam->getName(), std::make_pair(typeParam, type)});

        auto constraintType = getType(typeParam->getConstraint(), constraintGenContext);
        if (!constraintType)
        {
            LLVM_DEBUG(llvm::dbgs() << "\n!! skip. failed. should be resolved later\n";);
            return Reason::Failure;
        }

        auto extendsResult = mth.extendsType(location, type, constraintType, pairs);
        if (extendsResult != ExtendsResult::True)
        {
            // special case when we work with generic type(which are not specialized yet)
            if (mth.isGenericType(type))
            {
                pairs.insert({typeParam->getName(), std::make_pair(typeParam, type)});
                LLVM_DEBUG(llvm::dbgs() << "Extends result: " << type << " (because of generic).";);
                return Reason::None;                    
            }

            if (extendsResult == ExtendsResult::Any)
            {
                pairs.insert({typeParam->getName(), std::make_pair(typeParam, getAnyType())});
                LLVM_DEBUG(llvm::dbgs() << "Extends result: any.";);
                return Reason::None;                    
            }                

            if (extendsResult == ExtendsResult::Never)
            {
                pairs.insert({typeParam->getName(), std::make_pair(typeParam, getNeverType())});
                LLVM_DEBUG(llvm::dbgs() << "Extends result: never.";);
                return Reason::None;                    
            }

            LLVM_DEBUG(llvm::dbgs() << "Type " << type << " does extend "
                                    << constraintType << ".";);

            emitWarning(location, "") << "Type " << to_print(type) << " does not satisfy the constraint "
                                    << to_print(constraintType) << ".";

            return Reason::FailedConstraint;
        }

        return Reason::NoConstraint;
    }

    std::tuple<mlir::LogicalResult, IsGeneric> MLIRGenImpl::zipTypeParameterWithArgument(
        mlir::Location location, llvm::StringMap<std::pair<TypeParameterDOM::TypePtr, mlir::Type>> &pairs,
        const ts::TypeParameterDOM::TypePtr &typeParam, mlir::Type type, bool noExtendTest,
        const GenContext &genContext, bool mergeTypes, bool arrayMerge)
    {
        LLVM_DEBUG(llvm::dbgs() << "\n!! assigning generic type: " << typeParam->getName() << " type: " << type
                                << "\n";);

        if (mth.isNoneType(type))
        {
            LLVM_DEBUG(llvm::dbgs() << "\n!! skip. failed.\n";);
            return {mlir::failure(), IsGeneric::False};
        }

        if (isa<mlir_ts::NamedGenericType>(type))
        {
            pairs.insert({typeParam->getName(), std::make_pair(typeParam, type)});
            return {mlir::success(), IsGeneric::True};
        }

        auto name = typeParam->getName();
        auto existType = pairs.lookup(name);
        if (existType.second)
        {
            if (existType.second != type)
            {
                LLVM_DEBUG(llvm::dbgs() << "\n!! replacing existing type for: " << name
                                        << " exist type: " << existType.second << " new type: " << type << "\n";);

                if (!isa<mlir_ts::NamedGenericType>(existType.second) && mergeTypes)
                {
                    auto merged = false;
                    if (arrayMerge)
                    {
                        type = mth.arrayMergeType(location, existType.second, type, merged);
                    }
                    else
                    {
                        type = mth.mergeType(location, existType.second, type, merged);
                    }

                    LLVM_DEBUG(llvm::dbgs() << "\n!! result (after merge) type: " << type << "\n";);
                }

                // TODO: Do I need to join types?
                pairs[name] = std::make_pair(typeParam, type);
            }
        }
        else
        {
            pairs.insert({name, std::make_pair(typeParam, type)});
        }

        // we need to test constaint to infer some types
        auto constraint = typeParam->getConstraint();
        if (constraint)
        {
            // we ignore the test result but we need infered types, constraint will be checked later
            testConstraint(location, pairs, typeParam, type, genContext);
        }

        return {mlir::success(), IsGeneric::False};
    }

    std::pair<mlir::LogicalResult, IsGeneric> MLIRGenImpl::zipTypeParametersWithArguments(
        mlir::Location location, llvm::ArrayRef<TypeParameterDOM::TypePtr> typeParams, llvm::ArrayRef<mlir::Type> typeArgs,
        llvm::StringMap<std::pair<TypeParameterDOM::TypePtr, mlir::Type>> &pairs, const GenContext &genContext)
    {
        auto anyNamedGenericType = IsGeneric::False;
        auto argsCount = typeArgs.size();
        for (auto [index, typeParam] : enumerate(typeParams))
        {
            auto isDefault = false;
            auto type = index < argsCount
                            ? typeArgs[index]
                            : (isDefault = true, typeParam->hasDefault() 
                                ? getType(typeParam->getDefault(), genContext) 
                                : typeParam->hasConstraint() 
                                    ? getType(typeParam->getConstraint(), genContext) 
                                    : mlir::Type());
            if (!type)
            {
                return {mlir::failure(), anyNamedGenericType};
            }

            auto [result, hasNamedGenericType] =
                zipTypeParameterWithArgument(location, pairs, typeParam, type, isDefault, genContext);
            if (mlir::failed(result))
            {
                return {mlir::failure(), anyNamedGenericType};
            }

            if (hasNamedGenericType == IsGeneric::True)
            {
                anyNamedGenericType = hasNamedGenericType;
            }
        }

        return {mlir::success(), anyNamedGenericType};
    }

    std::tuple<mlir::LogicalResult, IsGeneric> MLIRGenImpl::zipTypeParametersWithArguments(
        mlir::Location location, llvm::ArrayRef<TypeParameterDOM::TypePtr> typeParams, NodeArray<TypeNode> typeArgs,
        llvm::StringMap<std::pair<TypeParameterDOM::TypePtr, mlir::Type>> &pairs, const GenContext &genContext)
    {
        auto anyNamedGenericType = IsGeneric::False;
        auto argsCount = typeArgs.size();
        for (auto [index, typeParam] : enumerate(typeParams))
        {
            auto isDefault = false;
            mlir::Type type;
            if (index < argsCount)
            {
                type = getType(typeArgs[index], genContext);
            }
            else
            {
                isDefault = true;
                if (typeParam->hasDefault())
                {
                    type = getType(typeParam->getDefault(), genContext);
                }
                else if (typeParam->hasConstraint())
                {
                    type = getType(typeParam->getConstraint(), genContext);
                }
            }

            if (!type)
            {
                if (isDefault && !typeParam->hasDefault() && argsCount == 0)
                {
                    // seems creating instance without TypeParams, can be used instance with the same name
                    // such as Point and Point<T>
                    return {mlir::failure(), IsGeneric::NoDefaults};    
                }

                return {mlir::failure(), anyNamedGenericType};
            }

            auto [result, hasNamedGenericType] =
                zipTypeParameterWithArgument(location, pairs, typeParam, type, isDefault, genContext);
            if (mlir::failed(result))
            {
                return {mlir::failure(), anyNamedGenericType};
            }

            if (hasNamedGenericType == IsGeneric::True)
            {
                anyNamedGenericType = hasNamedGenericType;
            }
        }

        return {mlir::success(), anyNamedGenericType};
    }

    std::pair<mlir::LogicalResult, IsGeneric> MLIRGenImpl::zipTypeParametersWithArgumentsNoDefaults(
        mlir::Location location, llvm::ArrayRef<TypeParameterDOM::TypePtr> typeParams, NodeArray<TypeNode> typeArgs,
        llvm::StringMap<std::pair<TypeParameterDOM::TypePtr, mlir::Type>> &pairs, const GenContext &genContext)
    {
        auto anyNamedGenericType = IsGeneric::False;
        auto argsCount = typeArgs.size();
        for (auto [index, typeParam] : enumerate(typeParams))
        {
            auto isDefault = false;
            auto type = index < argsCount
                            ? getType(typeArgs[index], genContext)
                            : (isDefault = true,
                               typeParam->hasDefault() 
                               ? getType(typeParam->getDefault(), genContext) 
                               : typeParam->hasConstraint() 
                                    ? getType(typeParam->getConstraint(), genContext) 
                                    : mlir::Type());
            if (!type)
            {
                return {mlir::success(), anyNamedGenericType};
            }

            if (isDefault)
            {
                return {mlir::success(), anyNamedGenericType};
            }

            auto [result, hasNamedGenericType] =
                zipTypeParameterWithArgument(location, pairs, typeParam, type, isDefault, genContext);
            if (mlir::failed(result))
            {
                return {mlir::failure(), anyNamedGenericType};
            }

            if (hasNamedGenericType == IsGeneric::True)
            {
                anyNamedGenericType = hasNamedGenericType;
            }
        }

        return {mlir::success(), anyNamedGenericType};
    }

    std::pair<mlir::LogicalResult, IsGeneric> MLIRGenImpl::zipTypeParametersWithDefaultArguments(
        mlir::Location location, llvm::ArrayRef<TypeParameterDOM::TypePtr> typeParams, NodeArray<TypeNode> typeArgs,
        llvm::StringMap<std::pair<TypeParameterDOM::TypePtr, mlir::Type>> &pairs, const GenContext &genContext)
    {
        auto anyNamedGenericType = IsGeneric::False;
        auto argsCount = typeArgs ? typeArgs.size() : 0;
        for (auto [index, typeParam] : enumerate(typeParams))
        {
            auto isDefault = false;
            if (index < argsCount)
            {
                // we need to process only default values
                continue;
            }
            auto type = typeParam->hasDefault() 
                            ? getType(typeParam->getDefault(), genContext) 
                            : typeParam->hasConstraint() 
                                ? getType(typeParam->getConstraint(), genContext) 
                                : mlir::Type();
            if (!type)
            {
                continue;
            }

            auto name = typeParam->getName();
            auto existType = pairs.lookup(name);
            if (existType.second)
            {
                // type is resolved
                continue;
            }

            LLVM_DEBUG(llvm::dbgs() << "\n!! adding default type: " << typeParam->getName() << " type: " << type
                                << "\n";);

            auto [result, hasNamedGenericType] =
                zipTypeParameterWithArgument(location, pairs, typeParam, type, isDefault, genContext);
            if (mlir::failed(result))
            {
                return {mlir::failure(), anyNamedGenericType};
            }

            if (hasNamedGenericType == IsGeneric::True)
            {
                anyNamedGenericType = hasNamedGenericType;
            }
        }

        return {mlir::success(), anyNamedGenericType};
    }

    mlir::Type MLIRGenImpl::createTypeReferenceType(TypeReferenceNode typeReferenceAST, const GenContext &genContext)
    {
        mlir::SmallVector<mlir::Type> typeArgs;
        for (auto typeArgNode : typeReferenceAST->typeArguments)
        {
            auto typeArg = getType(typeArgNode, genContext);
            if (!typeArg)
            {
                return mlir::Type();
            }

            typeArgs.push_back(typeArg);
        }

        auto nameRef = MLIRHelper::getName(typeReferenceAST->typeName, stringAllocator);
        auto typeRefType = getTypeReferenceType(nameRef, typeArgs);

        LLVM_DEBUG(llvm::dbgs() << "\n!! generic TypeReferenceType: " << typeRefType;);

        return typeRefType;
    };

    mlir::Type MLIRGenImpl::getTypeByTypeReference(mlir::Location location, mlir_ts::TypeReferenceType typeReferenceType, const GenContext &genContext)
    {
        // check utility types
        auto name = typeReferenceType.getName().getValue();

        // try to resolve from type alias first
        auto genericTypeAliasInfo = lookupGenericTypeAliasMap(name);
        if (!is_default(genericTypeAliasInfo))
        {
            GenContext genericTypeGenContext(genContext);

            auto typeParams = std::get<0>(genericTypeAliasInfo);
            auto typeNode = std::get<1>(genericTypeAliasInfo);

            auto [result, hasAnyNamedGenericType] =
                zipTypeParametersWithArguments(location, typeParams, typeReferenceType.getTypes(),
                                               genericTypeGenContext.typeParamsWithArgs, genericTypeGenContext);

            if (mlir::failed(result))
            {
                return mlir::Type();
            }

            return getType(typeNode, genericTypeGenContext);
        }  

        return mlir::Type();      
    }

    mlir::Type MLIRGenImpl::resolveGenericTypeInNamespace(mlir::Location location, StringRef name, TypeReferenceNode typeReferenceAST, const GenContext &genContext)
    {
        // try to resolve from type alias first
        auto genericTypeAliasInfo = lookupGenericTypeAliasMap(name);
        if (!is_default(genericTypeAliasInfo))
        {
            GenContext genericTypeGenContext(genContext);

            auto typeParams = std::get<0>(genericTypeAliasInfo);
            auto typeNode = std::get<1>(genericTypeAliasInfo);

            auto [result, hasAnyNamedGenericType] =
                zipTypeParametersWithArguments(location, typeParams, typeReferenceAST->typeArguments,
                                            genericTypeGenContext.typeParamsWithArgs, genericTypeGenContext);

            if (mlir::failed(result))
            {
                return mlir::Type();
            }

            if (hasAnyNamedGenericType == IsGeneric::True)
            {
                return createTypeReferenceType(typeReferenceAST, genericTypeGenContext);
            }

            return getType(typeNode, genericTypeGenContext);
        }

        if (auto genericClassTypeInfo = lookupGenericClassesMap(name))
        {
            auto classType = genericClassTypeInfo->classType;
            auto [result, specType] = instantiateSpecializedClassType(location, classType,
                                                                    typeReferenceAST->typeArguments, genContext, true);
            if (mlir::succeeded(result))
            {
                return specType;
            }

            return classType;
        }

        if (auto genericInterfaceTypeInfo = lookupGenericInterfacesMap(name))
        {
            auto interfaceType = genericInterfaceTypeInfo->interfaceType;
            auto [result, specType] = instantiateSpecializedInterfaceType(location, interfaceType,
                                                                        typeReferenceAST->typeArguments, genContext, true);
            if (mlir::succeeded(result))
            {
                return specType;
            }

            return interfaceType;
        }

        return mlir::Type();
    }

    mlir::Type MLIRGenImpl::resolveGenericType(mlir::Location location, StringRef name, TypeReferenceNode typeReferenceAST, const GenContext &genContext) 
    {
        MLIRNamespaceGuard ng(currentNamespace);

        // search in outer namespaces
        while (currentNamespace->isFunctionNamespace)
        {
            currentNamespace = currentNamespace->parentNamespace;
            if (auto type = resolveGenericTypeInNamespace(location, name, typeReferenceAST, genContext))
            {
                return type;
            }
        }

        // search in root namespace
        currentNamespace = rootNamespace;
        if (auto type = resolveGenericTypeInNamespace(location, name, typeReferenceAST, genContext))
        {
            return type;
        }

        return mlir::Type();
    }

    mlir::Type MLIRGenImpl::getTypeByTypeReference(TypeReferenceNode typeReferenceAST, const GenContext &genContext)
    {
        auto location = loc(typeReferenceAST);

        // check utility types
        auto name = MLIRHelper::getName(typeReferenceAST->typeName);

        {
            MLIRNamespaceGuard ng(currentNamespace);
            if (typeReferenceAST->typeName == SyntaxKind::QualifiedName)
            {
                auto qualifiedName = typeReferenceAST->typeName.as<QualifiedName>();
                auto location = loc(qualifiedName);

                auto expression = qualifiedName->left;
                auto result = mlirGenModuleReference(expression, genContext);
                if (result.failed_or_no_value())
                {
                    return mlir::Type();
                }

                auto expressionValue = V(result);

                if (auto namespaceOp = expressionValue.getDefiningOp<mlir_ts::NamespaceRefOp>())
                {
                    auto namespaceType = mlir::cast<mlir_ts::NamespaceType>(namespaceOp.getType());
                    
                    auto namespaceInfo = getNamespaceByFullName(namespaceType.getName().getValue());
                    assert(namespaceInfo);

                    currentNamespace = namespaceInfo;            
                }
                else
                {
                    emitError(location, "QualifiedName ") << print(qualifiedName) << " is not namespace";
                    return mlir::Type();
                }

                name = MLIRHelper::getName(qualifiedName->right);
            }

            if (typeReferenceAST->typeArguments.size())
            {
                if (auto type = resolveGenericTypeInNamespace(location, name, typeReferenceAST, genContext))
                {
                    return type;
                }

                if (auto type = resolveGenericType(location, name, typeReferenceAST, genContext))
                {
                    return type;
                }

                if (auto embedType = findEmbeddedType(location, name, typeReferenceAST->typeArguments, genContext))
                {
                    return embedType;
                }

                emitError(location, "generic type ") << name << " can't be found";
                return mlir::Type();
            }
        }

        if (auto type = getTypeByTypeName(typeReferenceAST->typeName, genContext))
        {
            return type;
        }

        if (auto embedType = findEmbeddedType(location, name, typeReferenceAST->typeArguments, genContext))
        {
            return embedType;
        }

        return mlir::Type();
    }

    mlir::Type MLIRGenImpl::findEmbeddedType(mlir::Location location, std::string name, NodeArray<TypeNode> &typeArguments, const GenContext &genContext)
    {
        auto typeArgumentsSize = typeArguments->size();
        if (typeArgumentsSize == 0)
        {
            if (auto type = getEmbeddedType(name))
            {
                return type;
            }
        }

        if (typeArgumentsSize == 1)
        {
            if (auto type = getEmbeddedTypeWithParam(name, typeArguments, genContext))
            {
                return type;
            }
        }

        if (typeArgumentsSize > 1)
        {
            if (auto type = getEmbeddedTypeWithManyParams(location, name, typeArguments, genContext))
            {
                return type;
            }
        }

        return mlir::Type();
    }

    bool MLIRGenImpl::isEmbededType(mlir::StringRef name)
    {
        return compileOptions.enableBuiltins ? isEmbededTypeWithBuiltins(name) : isEmbededTypeWithNoBuiltins(name);
    }

    bool MLIRGenImpl::isEmbededTypeWithBuiltins(mlir::StringRef name)
    {
        static llvm::StringMap<bool> embeddedTypes {
            {"TemplateStringsArray", true },
            {"const", true },
#ifdef ENABLE_JS_BUILTIN_TYPES
            {"Number", true },
            {"Object", true },
            {"String", true },
            {"Boolean", true },
            {"Function", true },
#endif
#ifdef ENABLE_NATIVE_TYPES
            {"byte", true },
            {"short", true },
            {"ushort", true },
            {"int", true },
            {"uint", true },
            {"index", true },
            {"long", true },
            {"ulong", true },
            {"char", true },
            {"i8", true },
            {"i16", true },
            {"i32", true },
            {"i64", true },
            {"u8", true},
            {"u16", true},
            {"u32", true},
            {"u64", true},
            {"s8", true},
            {"s16", true},
            {"s32", true},
            {"s64", true},
            {"f16", true},
            {"f32", true},
            {"f64", true},
            {"f128", true},
            {"half", true},
            {"float", true},
            {"double", true},
#endif
#ifdef ENABLE_JS_TYPEDARRAYS
            {"Int8Array", true },
            {"Uint8Array", true },
            {"Int16Array", true },
            {"Uint16Array", true },
            {"Int32Array", true },
            {"Uint32Array", true },
            {"BigInt64Array", true },
            {"BigUint64Array", true },
            {"Float16Array", true },
            {"Float32Array", true },
            {"Float64Array", true },
            {"Float128Array", true},
#endif

            {"TypeOf", true },
            {"Opaque", true }, // to support void*
            {"Reference", true }, // to support dll import
            {"Ref", true }, // alias of Reference
            {"Readonly", true },
            {"Partial", true },
            {"Required", true },
            {"ThisType", true },
            {"NonNullable", true },
            //{"Array", true },
            //{"ReadonlyArray", true },
            {"ReturnType", true },
            {"Parameters", true },
            {"ConstructorParameters", true },
            {"ThisParameterType", true },
            {"OmitThisParameter", true },
            {"Uppercase", true },
            {"Lowercase", true },
            {"Capitalize", true },
            {"Uncapitalize", true },
            {"Exclude",  true },
            {"Extract", true },
            {"Pick", true },
            {"Omit",  true },
            {"Record", true },
        };

        auto type = embeddedTypes[name];
        return type;
    }

    bool MLIRGenImpl::isEmbededTypeWithNoBuiltins(mlir::StringRef name)
    {
        static llvm::StringMap<bool> embeddedTypes {
            {"TemplateStringsArray", true },
            {"const", true },
#ifdef ENABLE_JS_BUILTIN_TYPES
            {"Number", true },
            {"Object", true },
            {"String", true },
            {"Boolean", true },
            {"Function", true },
#endif
#ifdef ENABLE_NATIVE_TYPES
            {"byte", true },
            {"short", true },
            {"ushort", true },
            {"int", true },
            {"uint", true },
            {"index", true },
            {"long", true },
            {"ulong", true },
            {"char", true },
            {"i8", true },
            {"i16", true },
            {"i32", true },
            {"i64", true },
            {"u8", true},
            {"u16", true},
            {"u32", true},
            {"u64", true},
            {"s8", true},
            {"s16", true},
            {"s32", true},
            {"s64", true},
            {"f16", true},
            {"f32", true},
            {"f64", true},
            {"f128", true},
            {"half", true},
            {"float", true},
            {"double", true},
#endif
#ifdef ENABLE_JS_TYPEDARRAYS_NOBUILTINS
            {"Int8Array", true },
            {"Uint8Array", true },
            {"Int16Array", true },
            {"Uint16Array", true },
            {"Int32Array", true },
            {"Uint32Array", true },
            {"BigInt64Array", true },
            {"BigUint64Array", true },
            {"Float16Array", true },
            {"Float32Array", true },
            {"Float64Array", true },
            {"Float128Array", true},
#endif

            {"TypeOf", true },
            {"Opaque", true }, // to support void*
            {"Reference", true }, // to support dll import
            {"Ref", true }, // alias of Reference
            {"ThisType", true },
            //{"Array", true }
        };

        auto type = embeddedTypes[name];
        return type;
    }

    mlir::Type MLIRGenImpl::getEmbeddedType(mlir::StringRef name)
    {
        return compileOptions.enableBuiltins ? getEmbeddedTypeBuiltins(name) : getEmbeddedTypeNoBuiltins(name);
    }

    mlir::Type MLIRGenImpl::getEmbeddedTypeBuiltins(mlir::StringRef name)
    {
        static llvm::StringMap<mlir::Type> embeddedTypes {
            {"TemplateStringsArray", getArrayType(getStringType()) },
            {"const",getConstType() },
#ifdef ENABLE_JS_BUILTIN_TYPES
            {"Number", getNumberType() },
            {"Object", getObjectType(getAnyType()) },
            {"String", getStringType()},
            {"Boolean", getBooleanType()},
            {"Function", getFunctionType({getArrayType(getAnyType())}, {getAnyType()}, true)},
#endif
#ifdef ENABLE_NATIVE_TYPES
            {"byte", builder.getIntegerType(8) },
            {"short", builder.getIntegerType(16, true) },
            {"ushort", builder.getIntegerType(16, false) },
            {"int", builder.getIntegerType(32, true) },
            {"uint", builder.getIntegerType(32, false) },
            {"index", builder.getIndexType() },
            {"long", builder.getIntegerType(64, true) },
            {"ulong", builder.getIntegerType(64, false) },
            {"char", getCharType() },
            {"i8", builder.getIntegerType(8) },
            {"i16", builder.getIntegerType(16) },
            {"i32", builder.getIntegerType(32) },
            {"i64", builder.getIntegerType(64) },
            {"u8", builder.getIntegerType(8, false)},
            {"u16", builder.getIntegerType(16, false)},
            {"u32", builder.getIntegerType(32, false)},
            {"u64", builder.getIntegerType(64, false)},
            {"s8", builder.getIntegerType(8, true) },
            {"s16", builder.getIntegerType(16, true) },
            {"s32", builder.getIntegerType(32, true) },
            {"s64", builder.getIntegerType(64, true) },
            {"f16", builder.getF16Type()},
            {"f32", builder.getF32Type()},
            {"f64", builder.getF64Type()},
            {"f128", builder.getF128Type()},
            {"half", builder.getF16Type()},
            {"float", builder.getF32Type()},
            {"double", builder.getF64Type()},
#endif
#ifdef ENABLE_JS_TYPEDARRAYS
            {"Int8Array", getArrayType(builder.getIntegerType(8, true)) },
            {"Uint8Array", getArrayType(builder.getIntegerType(8, false))},
            {"Int16Array", getArrayType(builder.getIntegerType(16, true)) },
            {"Uint16Array", getArrayType(builder.getIntegerType(16, false))},
            {"Int32Array", getArrayType(builder.getIntegerType(32, true)) },
            {"Uint32Array", getArrayType(builder.getIntegerType(32, false))},
            {"BigInt64Array", getArrayType(builder.getIntegerType(64, true)) },
            {"BigUint64Array", getArrayType(builder.getIntegerType(64, false))},
            {"Float16Array", getArrayType(builder.getF16Type())},
            {"Float32Array", getArrayType(builder.getF32Type())},
            {"Float64Array", getArrayType(builder.getF64Type())},
            {"Float128Array", getArrayType(builder.getF128Type())},
#endif
            {"Opaque", getOpaqueType()},
        };

        auto type = embeddedTypes[name];
        return type;
    }

    mlir::Type MLIRGenImpl::getEmbeddedTypeNoBuiltins(mlir::StringRef name)
    {
        static llvm::StringMap<mlir::Type> embeddedTypes {
            {"TemplateStringsArray", getArrayType(getStringType()) },
            {"const",getConstType() },
#ifdef ENABLE_JS_BUILTIN_TYPES
            {"Number", getNumberType() },
            {"Object", getObjectType(getAnyType()) },
            {"String", getStringType()},
            {"Boolean", getBooleanType()},
            {"Function", getFunctionType({getArrayType(getAnyType())}, {getAnyType()}, true)},
#endif
#ifdef ENABLE_NATIVE_TYPES
            {"byte", builder.getIntegerType(8) },
            {"short", builder.getIntegerType(16, true) },
            {"ushort", builder.getIntegerType(16, false) },
            {"int", builder.getIntegerType(32, true) },
            {"uint", builder.getIntegerType(32, false) },
            {"index", builder.getIndexType() },
            {"long", builder.getIntegerType(64, true) },
            {"ulong", builder.getIntegerType(64, false) },
            {"char", getCharType() },
            {"i8", builder.getIntegerType(8) },
            {"i16", builder.getIntegerType(16) },
            {"i32", builder.getIntegerType(32) },
            {"i64", builder.getIntegerType(64) },
            {"u8", builder.getIntegerType(8, false)},
            {"u16", builder.getIntegerType(16, false)},
            {"u32", builder.getIntegerType(32, false)},
            {"u64", builder.getIntegerType(64, false)},
            {"s8", builder.getIntegerType(8, true) },
            {"s16", builder.getIntegerType(16, true) },
            {"s32", builder.getIntegerType(32, true) },
            {"s64", builder.getIntegerType(64, true) },
            {"f16", builder.getF16Type()},
            {"f32", builder.getF32Type()},
            {"f64", builder.getF64Type()},
            {"f128", builder.getF128Type()},
            {"half", builder.getF16Type()},
            {"float", builder.getF32Type()},
            {"double", builder.getF64Type()},
#endif
#ifdef ENABLE_JS_TYPEDARRAYS_NOBUILTINS
            {"Int8Array", getArrayType(builder.getIntegerType(8, true)) },
            {"Uint8Array", getArrayType(builder.getIntegerType(8, false))},
            {"Int16Array", getArrayType(builder.getIntegerType(16, true)) },
            {"Uint16Array", getArrayType(builder.getIntegerType(16, false))},
            {"Int32Array", getArrayType(builder.getIntegerType(32, true)) },
            {"Uint32Array", getArrayType(builder.getIntegerType(32, false))},
            {"BigInt64Array", getArrayType(builder.getIntegerType(64, true)) },
            {"BigUint64Array", getArrayType(builder.getIntegerType(64, false))},
            {"Float16Array", getArrayType(builder.getF16Type())},
            {"Float32Array", getArrayType(builder.getF32Type())},
            {"Float64Array", getArrayType(builder.getF64Type())},
            {"Float128Array", getArrayType(builder.getF128Type())},
#endif

            {"Opaque", getOpaqueType()},
        };

        auto type = embeddedTypes[name];
        return type;
    }    

    mlir::Type MLIRGenImpl::getEmbeddedTypeWithParam(mlir::StringRef name, NodeArray<TypeNode> &typeArguments,
                                        const GenContext &genContext)
    {
        return compileOptions.enableBuiltins 
            ? getEmbeddedTypeWithParamBuiltins(name, typeArguments, genContext) 
            : getEmbeddedTypeWithParamNoBuiltins(name, typeArguments, genContext);
    }

    mlir::Type MLIRGenImpl::getEmbeddedTypeWithParamBuiltins(mlir::StringRef name, NodeArray<TypeNode> &typeArguments,
                                        const GenContext &genContext)
    {
        enum class EmbeddedType
        {
            None, TypeOf, Reference, FirstTypeArgument, NonNullable, Array, ReadonlyArray, ReturnType,
            Parameters, ThisParameterType, OmitThisParameter, Uppercase, Lowercase, Capitalize, Uncapitalize
        };

        auto kind = llvm::StringSwitch<EmbeddedType>(name)
            .Case("TypeOf", EmbeddedType::TypeOf)
            .Cases("Reference", "Ref", EmbeddedType::Reference)
            .Cases("Readonly", "Partial", "Required", "ThisType", EmbeddedType::FirstTypeArgument)
            .Case("NonNullable", EmbeddedType::NonNullable)
#ifdef ARRAY_TYPE_AS_ARRAY_CLASS
            .Case("Array", EmbeddedType::Array)
#endif
            .Case("ReadonlyArray", EmbeddedType::ReadonlyArray)
            .Case("ReturnType", EmbeddedType::ReturnType)
            .Cases("Parameters", "ConstructorParameters", EmbeddedType::Parameters)
            .Case("ThisParameterType", EmbeddedType::ThisParameterType)
            .Case("OmitThisParameter", EmbeddedType::OmitThisParameter)
            .Case("Uppercase", EmbeddedType::Uppercase)
            .Case("Lowercase", EmbeddedType::Lowercase)
            .Case("Capitalize", EmbeddedType::Capitalize)
            .Case("Uncapitalize", EmbeddedType::Uncapitalize)
            .Default(EmbeddedType::None);

        if (kind == EmbeddedType::None)
        {
            return mlir::Type();
        }

        auto type = getFirstTypeFromTypeArguments(typeArguments, genContext);
        if (!type)
        {
            return mlir::Type();
        }

        switch (kind)
        {
            case EmbeddedType::TypeOf:
                return mth.wideStorageType(type);
            case EmbeddedType::Reference:
                return mlir_ts::RefType::get(type);
            case EmbeddedType::FirstTypeArgument:
                return type;
            case EmbeddedType::NonNullable:
                return NonNullableTypes(type);
#ifdef ARRAY_TYPE_AS_ARRAY_CLASS
            case EmbeddedType::Array:
                return getArrayType(type);
#endif
            case EmbeddedType::ReadonlyArray:
                return getArrayType(type);
            case EmbeddedType::ReturnType:
            {
                LLVM_DEBUG(llvm::dbgs() << "\n!! ReturnType Of: " << type;);
                auto retType = mth.getReturnTypeFromFuncRef(type);
                LLVM_DEBUG(llvm::dbgs() << " is " << retType << "\n";);
                return retType;
            }
            case EmbeddedType::Parameters:
            {
                LLVM_DEBUG(llvm::dbgs() << "\n!! ElementType Of: " << type;);
                auto retType = mth.getParamsTupleTypeFromFuncRef(type);
                LLVM_DEBUG(llvm::dbgs() << " is " << retType << "\n";);
                return retType;
            }
            case EmbeddedType::ThisParameterType:
            {
                LLVM_DEBUG(llvm::dbgs() << "\n!! ElementType Of: " << type;);
                auto retType = mth.getFirstParamFromFuncRef(type);
                LLVM_DEBUG(llvm::dbgs() << " is " << retType << "\n";);
                return retType;
            }
            case EmbeddedType::OmitThisParameter:
            {
                LLVM_DEBUG(llvm::dbgs() << "\n!! ElementType Of: " << type;);
                auto retType = mth.getOmitThisFunctionTypeFromFuncRef(type);
                LLVM_DEBUG(llvm::dbgs() << " is " << retType << "\n";);
                return retType;
            }
            case EmbeddedType::Uppercase:
                return UppercaseType(type);
            case EmbeddedType::Lowercase:
                return LowercaseType(type);
            case EmbeddedType::Capitalize:
                return CapitalizeType(type);
            case EmbeddedType::Uncapitalize:
                return UncapitalizeType(type);
            default:
                return mlir::Type();
        }
    }

    mlir::Type MLIRGenImpl::getEmbeddedTypeWithParamNoBuiltins(mlir::StringRef name, NodeArray<TypeNode> &typeArguments,
                                        const GenContext &genContext)
    {
        enum class EmbeddedType
        {
            None, TypeOf, Reference, ThisType, Array
        };

        auto kind = llvm::StringSwitch<EmbeddedType>(name)
            .Case("TypeOf", EmbeddedType::TypeOf)
            .Cases("Reference", "Ref", EmbeddedType::Reference)
            .Case("ThisType", EmbeddedType::ThisType)
#ifdef ARRAY_TYPE_AS_ARRAY_CLASS
            .Case("Array", EmbeddedType::Array)
#endif
            .Default(EmbeddedType::None);

        if (kind == EmbeddedType::None)
        {
            return mlir::Type();
        }

        auto type = getFirstTypeFromTypeArguments(typeArguments, genContext);
        switch (kind)
        {
            case EmbeddedType::TypeOf:
                return mth.wideStorageType(type);
            case EmbeddedType::Reference:
                return mlir_ts::RefType::get(type);
            case EmbeddedType::ThisType:
                return type;
#ifdef ARRAY_TYPE_AS_ARRAY_CLASS
            case EmbeddedType::Array:
                return getArrayType(type);
#endif
            default:
                return mlir::Type();
        }
    }

    mlir::Type MLIRGenImpl::getEmbeddedTypeWithManyParams(mlir::Location location, mlir::StringRef name, NodeArray<TypeNode> &typeArguments,
                                             const GenContext &genContext)
    {
        return compileOptions.enableBuiltins 
            ? getEmbeddedTypeWithManyParamsBuiltins(location, name, typeArguments, genContext) 
            : mlir::Type();
    }

    mlir::Type MLIRGenImpl::getEmbeddedTypeWithManyParamsBuiltins(mlir::Location location, mlir::StringRef name, NodeArray<TypeNode> &typeArguments,
                                             const GenContext &genContext)
    {
        enum class EmbeddedType
        {
            None, Exclude, Extract, Pick, Omit, Record
        };

        auto kind = llvm::StringSwitch<EmbeddedType>(name)
            .Case("Exclude", EmbeddedType::Exclude)
            .Case("Extract", EmbeddedType::Extract)
            .Case("Pick", EmbeddedType::Pick)
            .Case("Omit", EmbeddedType::Omit)
            .Case("Record", EmbeddedType::Record)
            .Default(EmbeddedType::None);

        if (kind == EmbeddedType::None)
        {
            return mlir::Type();
        }

        auto firstType = getFirstTypeFromTypeArguments(typeArguments, genContext);
        auto secondType = getSecondTypeFromTypeArguments(typeArguments, genContext);

        switch (kind)
        {
            case EmbeddedType::Exclude:
                return ExcludeTypes(location, firstType, secondType);
            case EmbeddedType::Extract:
                return ExtractTypes(location, firstType, secondType);
            case EmbeddedType::Pick:
                return PickTypes(firstType, secondType);
            case EmbeddedType::Omit:
                return OmitTypes(firstType, secondType);
            case EmbeddedType::Record:
                return RecordType(firstType, secondType);
            default:
                return mlir::Type();
        }
    }

    mlir::Type MLIRGenImpl::StringLiteralTypeFunc(mlir::Type type, std::function<std::string(StringRef)> f)
    {
        if (auto literalType = dyn_cast<mlir_ts::LiteralType>(type))
        {
            if (isa<mlir_ts::StringType>(literalType.getElementType()))
            {
                auto newStr = f(mlir::cast<mlir::StringAttr>(literalType.getValue()).getValue());
                auto copyVal = StringRef(newStr).copy(stringAllocator);
                return mlir_ts::LiteralType::get(builder.getStringAttr(copyVal), getStringType());
            }
        }

        LLVM_DEBUG(llvm::dbgs() << "\n!! can't apply string literal type for:" << type << "\n";);

        return mlir::Type();
    }

    mlir::Type MLIRGenImpl::UppercaseType(mlir::Type type)
    {
        return StringLiteralTypeFunc(type, [](auto val) { return val.upper(); });
    }

    mlir::Type MLIRGenImpl::LowercaseType(mlir::Type type)
    {
        return StringLiteralTypeFunc(type, [](auto val) { return val.lower(); });
    }

    mlir::Type MLIRGenImpl::CapitalizeType(mlir::Type type)
    {
        return StringLiteralTypeFunc(type,
                                     [](auto val) { return val.slice(0, 1).upper().append(val.slice(1, val.size())); });
    }

    mlir::Type MLIRGenImpl::UncapitalizeType(mlir::Type type)
    {
        return StringLiteralTypeFunc(type,
                                     [](auto val) { return val.slice(0, 1).lower().append(val.slice(1, val.size())); });
    }

    mlir::Type MLIRGenImpl::NonNullableTypes(mlir::Type type)
    {
        if (mth.isGenericType(type))
        {
            return type;
        }

        SmallPtrSet<mlir::Type, 2> types;

        MLIRHelper::flatUnionTypes(types, type);

        SmallVector<mlir::Type> resTypes;
        for (auto item : types)
        {
            if (isa<mlir_ts::NullType>(item) || item == getUndefinedType())
            {
                continue;
            }

            resTypes.push_back(item);
        }

        return getUnionType(resTypes);
    }

    mlir::Type MLIRGenImpl::ExcludeTypes(mlir::Location location, mlir::Type type, mlir::Type exclude)
    {
        if (mth.isGenericType(type) || mth.isGenericType(exclude))
        {
            return getAnyType();
        }

        SmallPtrSet<mlir::Type, 2> types;
        SmallPtrSet<mlir::Type, 2> excludeTypes;

        MLIRHelper::flatUnionTypes(types, type);
        MLIRHelper::flatUnionTypes(excludeTypes, exclude);

        SmallVector<mlir::Type> resTypes;
        for (auto item : types)
        {
            // TODO: should I use TypeParamsWithArgs from genContext?
            llvm::StringMap<std::pair<ts::TypeParameterDOM::TypePtr,mlir::Type>> emptyTypeParamsWithArgs;
            if (llvm::any_of(excludeTypes, [&](mlir::Type type) { 
                return isTrue(mth.extendsType(location, item, type, emptyTypeParamsWithArgs)); 
            }))
            {
                continue;
            }

            resTypes.push_back(item);
        }

        return getUnionType(resTypes);
    }

    mlir::Type MLIRGenImpl::ExtractTypes(mlir::Location location, mlir::Type type, mlir::Type extract)
    {
        if (mth.isGenericType(type) || mth.isGenericType(extract))
        {
            return getAnyType();
        }

        SmallPtrSet<mlir::Type, 2> types;
        SmallPtrSet<mlir::Type, 2> extractTypes;

        MLIRHelper::flatUnionTypes(types, type);
        MLIRHelper::flatUnionTypes(extractTypes, extract);

        SmallVector<mlir::Type> resTypes;
        for (auto item : types)
        {
            // TODO: should I use TypeParamsWithArgs from genContext?
            llvm::StringMap<std::pair<ts::TypeParameterDOM::TypePtr,mlir::Type>> emptyTypeParamsWithArgs;
            if (llvm::any_of(extractTypes, [&](mlir::Type type) { 
                return isTrue(mth.extendsType(location, item, type, emptyTypeParamsWithArgs)); 
            }))
            {
                resTypes.push_back(item);
            }
        }

        auto resultType = getUnionType(resTypes);
        LLVM_DEBUG(llvm::dbgs() << "\n!! Extract: " << resultType << "\n";);
        return resultType;
    }

    mlir::Type MLIRGenImpl::RecordType(mlir::Type keys, mlir::Type valueType)
    {
        LLVM_DEBUG(llvm::dbgs() << "\n!! Record: " << valueType << ", keys: " << keys << "\n";);
        
        SmallVector<mlir_ts::FieldInfo> fields;

        auto addTypeProcessKey = [&](mlir::Type keyType)
        {
            // get string
            if (auto litType = dyn_cast<mlir_ts::LiteralType>(keyType))
            {
                fields.push_back({ litType.getValue(), valueType, false, mlir_ts::AccessLevel::Public });
            }
        };

        if (auto unionType = dyn_cast<mlir_ts::UnionType>(keys))
        {
            for (auto keyType : unionType.getTypes())
            {
                addTypeProcessKey(keyType);
            }
        }
        else if (auto litType = dyn_cast<mlir_ts::LiteralType>(keys))
        {
            addTypeProcessKey(litType);
        }
        else
        {
            llvm_unreachable("not implemented");
        }        

        return getTupleType(fields);
    }

    mlir::Type MLIRGenImpl::PickTypes(mlir::Type type, mlir::Type keys)
    {
        LLVM_DEBUG(llvm::dbgs() << "\n!! Pick: " << type << ", keys: " << keys << "\n";);

        if (!keys)
        {
            return mlir::Type();
        }

        if (mth.isGenericType(type))
        {
            return getAnyType();
        }        

        if (auto unionType = dyn_cast<mlir_ts::UnionType>(type))
        {
            SmallVector<mlir::Type> pickedTypes;
            for (auto subType : unionType)
            {
                pickedTypes.push_back(PickTypes(subType, keys));
            }

            return getUnionType(pickedTypes);
        }

        SmallVector<mlir_ts::FieldInfo> pickedFields;
        SmallVector<mlir_ts::FieldInfo> fields;
        if (mlir::succeeded(mth.getFields(type, fields)))
        {
            auto pickTypesProcessKey = [&](mlir::Type keyType)
            {
                // get string
                if (auto litType = dyn_cast<mlir_ts::LiteralType>(keyType))
                {
                    // find field
                    auto found = std::find_if(fields.begin(), fields.end(), [&] (auto& item) { return item.id == litType.getValue(); });
                    if (found != fields.end())
                    {
                        pickedFields.push_back(*found);
                    }
                }
            };

            if (auto unionType = dyn_cast<mlir_ts::UnionType>(keys))
            {
                for (auto keyType : unionType.getTypes())
                {
                    pickTypesProcessKey(keyType);
                }
            }
            else if (auto litType = dyn_cast<mlir_ts::LiteralType>(keys))
            {
                pickTypesProcessKey(litType);
            }
        }

        return getTupleType(pickedFields);
    }

    mlir::Type MLIRGenImpl::OmitTypes(mlir::Type type, mlir::Type keys)
    {
        LLVM_DEBUG(llvm::dbgs() << "\n!! Omit: " << type << ", keys: " << keys << "\n";);

        SmallVector<mlir_ts::FieldInfo> pickedFields;

        SmallVector<mlir_ts::FieldInfo> fields;

        std::function<boolean(mlir_ts::FieldInfo& fieldInfo, mlir::Type keys)> existKey;
        existKey = [&](mlir_ts::FieldInfo& fieldInfo, mlir::Type keys)
        {
            // get string
            if (auto unionType = dyn_cast<mlir_ts::UnionType>(keys))
            {
                for (auto keyType : unionType.getTypes())
                {
                    if (existKey(fieldInfo, keyType))
                    {
                        return true;
                    }
                }
            }
            else if (auto litType = dyn_cast<mlir_ts::LiteralType>(keys))
            {
                return fieldInfo.id == litType.getValue();
            }
            else
            {
                llvm_unreachable("not implemented");
            }

            return false;
        };

        if (mlir::succeeded(mth.getFields(type, fields)))
        {
            for (auto& field : fields)
            {
                if (!existKey(field, keys))
                {
                    pickedFields.push_back(field);
                }
            }
        }

        return getTupleType(pickedFields);
    }        

    mlir::Type MLIRGenImpl::getTypeByTypeQuery(TypeQueryNode typeQueryAST, const GenContext &genContext)
    {
        auto exprName = typeQueryAST->exprName;
        if (exprName == SyntaxKind::QualifiedName)
        {
            // TODO: it seems namespace access, can u optimize it somehow?
            auto result = mlirGen(exprName.as<QualifiedName>(), genContext);
            if (result.failed_or_no_value())
            {
                return mlir::Type();
            }

            auto val = V(result);
            return val.getType();
        }

        auto type = evaluate(exprName.as<Expression>(), genContext);
        return type;
    }

    mlir::Type MLIRGenImpl::getTypePredicateType(TypePredicateNode typePredicateNode, const GenContext &genContext)
    {
        auto type = getType(typePredicateNode->type, genContext);
        if (!type)
        {
            return mlir::Type();
        }

        auto namePtr = 
            typePredicateNode->parameterName == SyntaxKind::ThisType
            ? THIS_NAME
            : MLIRHelper::getName(typePredicateNode->parameterName, stringAllocator);

        // find index of parameter
        auto hasThis = false;
        auto foundParamIndex = -1;
        if (genContext.funcProto)
        {
            for (auto [index, param] : enumerate(genContext.funcProto->getParams()))
            {
                if (foundParamIndex == -1 && param->getName() == namePtr)
                {
                    foundParamIndex = index;
                }

                hasThis |= param->getName() == THIS_NAME;
            }
        }

        auto parametereNameSymbol = mlir::FlatSymbolRefAttr::get(builder.getContext(), namePtr);
        return mlir_ts::TypePredicateType::get(parametereNameSymbol, type, !!typePredicateNode->assertsModifier, foundParamIndex - (hasThis ? 1 : 0));
    }

    mlir::Type MLIRGenImpl::processConditionalForType(ConditionalTypeNode conditionalTypeNode, mlir::Type checkType, mlir::Type extendsType, mlir::Type inferType, GenContext &genContext)
    {
        auto &typeParamsWithArgs = genContext.typeParamsWithArgs;

        auto location = loc(conditionalTypeNode);

        mlir::Type resType;
        auto extendsResult = mth.extendsType(location, checkType, extendsType, typeParamsWithArgs);
        if (extendsResult == ExtendsResult::Never)
        {
            return getNeverType();
        }

        if (isTrue(extendsResult))
        {
            if (inferType)
            {
                if (auto namedGenType = mlir::dyn_cast<mlir_ts::NamedGenericType>(inferType))
                {
                    auto typeParam = std::make_shared<TypeParameterDOM>(namedGenType.getName().getValue().str());
                    zipTypeParameterWithArgument(location, typeParamsWithArgs, typeParam, checkType, false, genContext, false);
                }
            }

            resType = getType(conditionalTypeNode->trueType, genContext);

            LLVM_DEBUG(llvm::dbgs() << "\n!! condition type [TRUE] = " << resType << "\n";);

            if (extendsResult != ExtendsResult::Any)
            {
                // in case of any we need "union" of true & false
                return resType;
            }
        }

        // false case
        if (inferType)
        {
            auto namedGenType = mlir::cast<mlir_ts::NamedGenericType>(inferType);
            auto typeParam = std::make_shared<TypeParameterDOM>(namedGenType.getName().getValue().str());
            zipTypeParameterWithArgument(location, typeParamsWithArgs, typeParam, checkType, false, genContext, false);
        }

        auto falseType = getType(conditionalTypeNode->falseType, genContext);

        if (extendsResult != ExtendsResult::Any || !resType)
        {
            resType = falseType;
            LLVM_DEBUG(llvm::dbgs() << "\n!! condition type [FALSE] = " << resType << "\n";);
        }
        else
        {
            resType = getUnionType(location, resType, falseType);
            LLVM_DEBUG(llvm::dbgs() << "\n!! condition type [TRUE | FALSE] = " << resType << "\n";);
        }

        return resType;
    }

    mlir::Type MLIRGenImpl::getConditionalType(ConditionalTypeNode conditionalTypeNode, const GenContext &genContext)
    {
        GenContext condTypeGenContext(genContext);
        condTypeGenContext.inferTypes = &condTypeGenContext.typeParamsWithArgs;

        auto checkType = getType(conditionalTypeNode->checkType, condTypeGenContext);
        auto extendsType = getType(conditionalTypeNode->extendsType, condTypeGenContext);
        if (!checkType || !extendsType)
        {
            return mlir::Type();
        }

        LLVM_DEBUG(llvm::dbgs() << "\n!! condition type check: " << checkType << ", extends: " << extendsType << "\n";);

        if (isa<mlir_ts::NamedGenericType>(checkType) || isa<mlir_ts::NamedGenericType>(extendsType))
        {
            // we do not need to resolve it, it is generic
            auto trueType = getType(conditionalTypeNode->trueType, condTypeGenContext);
            auto falseType = getType(conditionalTypeNode->falseType, condTypeGenContext);

            LLVM_DEBUG(llvm::dbgs() << "\n!! condition type, check: " << checkType << " extends: " << extendsType << " true: " << trueType << " false: " << falseType << " \n";);

            return getConditionalType(checkType, extendsType, trueType, falseType);
        }

        if (auto unionType = dyn_cast<mlir_ts::UnionType>(checkType))
        {
            // we need to have original type to infer types from union
            GenContext noTypeArgsContext(condTypeGenContext);
            llvm::StringMap<std::pair<TypeParameterDOM::TypePtr, mlir::Type>> typeParamsOnly;
            for (auto &pair : noTypeArgsContext.typeParamsWithArgs)
            {
                typeParamsOnly[pair.getKey()] = std::make_pair(std::get<0>(pair.getValue()), getNamedGenericType(pair.getKey()));
            }

            noTypeArgsContext.typeParamsWithArgs = typeParamsOnly;

            auto originalCheckType = getType(conditionalTypeNode->checkType, noTypeArgsContext);

            LLVM_DEBUG(llvm::dbgs() << "\n!! check type: " << checkType << " original: " << originalCheckType << " \n";);

            SmallVector<mlir::Type> results;
            for (auto subType : unionType.getTypes())
            {
                auto resSubType = processConditionalForType(conditionalTypeNode, subType, extendsType, originalCheckType, condTypeGenContext);
                if (!resSubType)
                {
                    return mlir::Type();
                }

                if (resSubType != getNeverType())
                {
                    results.push_back(resSubType);
                }
            }            

            return getUnionType(results);
        }

        return processConditionalForType(conditionalTypeNode, checkType, extendsType, mlir::Type(), condTypeGenContext);
    }

    mlir::Type MLIRGenImpl::getKeyOf(TypeOperatorNode typeOperatorNode, const GenContext &genContext)
    {
        auto location = loc(typeOperatorNode);

        auto type = getType(typeOperatorNode->type, genContext);
        if (!type)
        {
            LLVM_DEBUG(llvm::dbgs() << "\n!! can't take 'keyof'\n";);
            emitError(location, "can't take keyof");
            return mlir::Type();
        }

        return getKeyOf(location, type, genContext);
    }

    mlir::Type MLIRGenImpl::getKeyOf(mlir::Location location, mlir::Type type, const GenContext &genContext)
    {
        LLVM_DEBUG(llvm::dbgs() << "\n!! 'keyof' from: " << type << "\n";);

        if (isa<mlir_ts::AnyType>(type))
        {
            // TODO: and all methods etc
            return getUnionType(location, getStringType(), getNumberType());
        }

        if (isa<mlir_ts::UnknownType>(type))
        {
            // TODO: should be the same as Any?
            return getNeverType();
        }

        if (isa<mlir_ts::ArrayType>(type))
        {
            return mth.getFieldNames(type);
        }

        if (isa<mlir_ts::StringType>(type))
        {
            return mth.getFieldNames(type);
        }

        if (auto objType = dyn_cast<mlir_ts::ObjectType>(type))
        {
            // TODO: I think this is mistake
            type = objType.getStorageType();
        }

        if (auto classType = dyn_cast<mlir_ts::ClassType>(type))
        {
            return mth.getFieldNames(type);
        }

        if (auto tupleType = dyn_cast<mlir_ts::TupleType>(type))
        {
            return mth.getFieldNames(type);
        }

        if (auto interfaceType = dyn_cast<mlir_ts::InterfaceType>(type))
        {
            return mth.getFieldNames(type);
        }

        if (auto unionType = dyn_cast<mlir_ts::UnionType>(type))
        {
            SmallVector<mlir::Type> literalTypes;
            for (auto subType : unionType.getTypes())
            {
                auto keyType = getKeyOf(location, subType, genContext);
                literalTypes.push_back(keyType);
            }

            return getUnionType(literalTypes);
        }

        if (auto enumType = dyn_cast<mlir_ts::EnumType>(type))
        {
            SmallVector<mlir::Type> literalTypes;
            for (auto dictValuePair : enumType.getValues())
            {
                auto litType = mlir_ts::LiteralType::get(builder.getStringAttr(dictValuePair.getName().str()), getStringType());
                literalTypes.push_back(litType);
            }

            return getUnionType(literalTypes);
        }

        if (auto namedGenericType = dyn_cast<mlir_ts::NamedGenericType>(type))
        {
            return getKeyOfType(namedGenericType);
        }

        LLVM_DEBUG(llvm::dbgs() << "\n!! can't take 'keyof' from: " << type << "\n";);

        emitError(location, "can't take keyof: ") << to_print(type);

        return mlir::Type();
    }

    mlir::Type MLIRGenImpl::getTypeOperator(TypeOperatorNode typeOperatorNode, const GenContext &genContext)
    {
        if (typeOperatorNode->_operator == SyntaxKind::UniqueKeyword)
        {
            // TODO: finish it
            return getType(typeOperatorNode->type, genContext);
        }
        else if (typeOperatorNode->_operator == SyntaxKind::KeyOfKeyword)
        {
            return getKeyOf(typeOperatorNode, genContext);
        }
        else if (typeOperatorNode->_operator == SyntaxKind::ReadonlyKeyword)
        {
            // TODO: finish it
            return getType(typeOperatorNode->type, genContext);
        }        

        llvm_unreachable("not implemented");
    }

    mlir::Type MLIRGenImpl::getIndexedAccessTypeForArrayElement(mlir_ts::ArrayType type)
    {
        return type.getElementType();
    }

    mlir::Type MLIRGenImpl::getIndexedAccessTypeForArrayElement(mlir_ts::ConstArrayType type)
    {
        return type.getElementType();
    }

    mlir::Type MLIRGenImpl::getIndexedAccessTypeForArrayElement(mlir_ts::StringType type)
    {
        return getCharType();
    }

    mlir::Type MLIRGenImpl::getIndexedAccessType(mlir::Type type, mlir::Type indexType, const GenContext &genContext)
    {
        // in case of Generic Methods but not specialized yet
        if (auto namedGenericType = dyn_cast<mlir_ts::NamedGenericType>(type))
        {
            return getIndexAccessType(type, indexType);
        }

        if (auto namedGenericType = dyn_cast<mlir_ts::NamedGenericType>(indexType))
        {
            return getIndexAccessType(type, indexType);
        }

        if (isa<mlir_ts::StringType>(indexType))
        {
            LLVM_DEBUG(llvm::dbgs() << "\n!! IndexedAccessType for : " << type << " index " << indexType << " is not implemeneted, index type should not be 'string' it should be literal type \n";);
            llvm_unreachable("not implemented");
        }

        if (auto literalType = dyn_cast<mlir_ts::LiteralType>(type))
        {
            return getIndexedAccessType(literalType.getElementType(), indexType, genContext);
        }

        if (auto unionType = dyn_cast<mlir_ts::UnionType>(type))
        {
            SmallVector<mlir::Type> types;
            for (auto subType : unionType)
            {
                auto typeByKey = getIndexedAccessType(subType, indexType, genContext);
                if (!typeByKey)
                {
                    return mlir::Type();
                }

                types.push_back(typeByKey);
            }

            return getUnionType(types);
        }        

        if (auto unionType = dyn_cast<mlir_ts::UnionType>(indexType))
        {
            SmallVector<mlir::Type> resolvedTypes;
            for (auto itemType : unionType.getTypes())
            {
                auto resType = getIndexedAccessType(type, itemType, genContext);
                if (!resType)
                {
                    return mlir::Type();
                }

                resolvedTypes.push_back(resType);
            }

            return getUnionType(resolvedTypes);
        }

        if (auto arrayType = dyn_cast<mlir_ts::ArrayType>(type))
        {
            // TODO: rewrite using mth.getFieldTypeByIndex(type, indexType);
            return getIndexedAccessTypeForArray(arrayType, indexType, genContext);
        }

        if (auto arrayType = dyn_cast<mlir_ts::ConstArrayType>(type))
        {
            return getIndexedAccessTypeForArray(arrayType, indexType, genContext);
        }

        if (auto stringType = dyn_cast<mlir_ts::StringType>(type))
        {
            return getIndexedAccessTypeForArray(stringType, indexType, genContext);
        }

        if (auto objType = dyn_cast<mlir_ts::ObjectType>(type))
        {
            return mth.getFieldTypeByIndexType(type, indexType);
        }

        if (auto classType = dyn_cast<mlir_ts::ClassType>(type))
        {
            return mth.getFieldTypeByIndexType(type, indexType);
        }

        // TODO: sync it with mth.getFields
        if (auto tupleType = dyn_cast<mlir_ts::TupleType>(type))
        {
            return mth.getFieldTypeByIndexType(type, indexType);
        }

        if (auto interfaceType = dyn_cast<mlir_ts::InterfaceType>(type))
        {
            return mth.getFieldTypeByIndexType(type, indexType);
        }

        if (auto anyType = dyn_cast<mlir_ts::AnyType>(type))
        {
            return anyType;
        }

        if (isa<mlir_ts::NeverType>(type))
        {
            return type;
        }

        LLVM_DEBUG(llvm::dbgs() << "\n!! IndexedAccessType for : \n\t" << type << " \n\tindex " << indexType << " is not implemeneted \n";);

        llvm_unreachable("not implemented");
        //return mlir::Type();
    }

    mlir::Type MLIRGenImpl::getIndexedAccessType(IndexedAccessTypeNode indexedAccessTypeNode, const GenContext &genContext)
    {
        auto type = getType(indexedAccessTypeNode->objectType, genContext);
        if (!type)
        {
            return type;
        }

        auto indexType = getType(indexedAccessTypeNode->indexType, genContext);
        if (!indexType)
        {
            return indexType;
        }

        return getIndexedAccessType(type, indexType, genContext);
    }

    mlir::Type MLIRGenImpl::getTemplateLiteralType(TemplateLiteralTypeNode templateLiteralTypeNode, const GenContext &genContext)
    {
        auto location = loc(templateLiteralTypeNode);

        // first string
        auto text = convertWideToUTF8(templateLiteralTypeNode->head->rawText);

        SmallVector<mlir::Type> types;
        getTemplateLiteralSpan(types, text, templateLiteralTypeNode->templateSpans, 0, genContext);

        if (types.size() == 1)
        {
            return types.front();
        }

        return getUnionType(types);
    }

    void MLIRGenImpl::getTemplateLiteralSpan(SmallVector<mlir::Type> &types, const std::string &head,
                                NodeArray<TemplateLiteralTypeSpan> &spans, int spanIndex, const GenContext &genContext)
    {
        if (spanIndex >= spans.size())
        {
            auto newLiteralType = mlir_ts::LiteralType::get(builder.getStringAttr(head), getStringType());
            types.push_back(newLiteralType);
            return;
        }

        auto span = spans[spanIndex];
        auto type = getType(span->type, genContext);

        if (auto unionType = dyn_cast<mlir_ts::UnionType>(type))
        {
            getTemplateLiteralUnionType(types, unionType, head, spans, spanIndex, genContext);
        }
        else if (auto litType = dyn_cast<mlir_ts::LiteralType>(type))
        {
            getTemplateLiteralTypeItem(types, litType, head, spans, spanIndex, genContext);
        }
        else
        {
            // it is just type as example: type HexColor<T extends Color> = `#${string}`;
            // as 'string' is not union literal type then we have just type in result
            types.push_back(type);
        }
    }

    void MLIRGenImpl::getTemplateLiteralTypeItem(SmallVector<mlir::Type> &types, mlir_ts::LiteralType literalType, const std::string &head,
                                    NodeArray<TemplateLiteralTypeSpan> &spans, int spanIndex,
                                    const GenContext &genContext)
    {
        LLVM_DEBUG(llvm::dbgs() << "\n!! TemplateLiteralType, processing type: " << literalType << ", span: " << spanIndex
                                << "\n";);

        auto span = spans[spanIndex];

        std::stringstream ss;
        ss << head;

        auto typeText = mlir::cast<mlir::StringAttr>(literalType.getValue()).getValue();
        ss << typeText.str();

        auto spanText = convertWideToUTF8(span->literal->rawText);
        ss << spanText;

        getTemplateLiteralSpan(types, ss.str(), spans, spanIndex + 1, genContext);
    }

    void MLIRGenImpl::getTemplateLiteralUnionType(SmallVector<mlir::Type> &types, mlir::Type unionType, const std::string &head,
                                     NodeArray<TemplateLiteralTypeSpan> &spans, int spanIndex,
                                     const GenContext &genContext)
    {
        for (auto unionTypeItem : mlir::cast<mlir_ts::UnionType>(unionType).getTypes())
        {
            if (auto unionType = dyn_cast<mlir_ts::UnionType>(unionTypeItem))
            {
                getTemplateLiteralUnionType(types, unionType, head, spans, spanIndex, genContext);
            }
            else if (auto litType = dyn_cast<mlir_ts::LiteralType>(unionTypeItem))
            {
                getTemplateLiteralTypeItem(types, litType, head, spans, spanIndex, genContext);
            }            
            else 
            {
                // it is just type as example: type HexColor<T extends Color> = `#${string}`;
                // as 'string' is not union literal type then we have just type in result
                types.push_back(unionTypeItem);
            }
        }
    }

    mlir::Type MLIRGenImpl::getMappedType(MappedTypeNode mappedTypeNode, const GenContext &genContext)
    {
        // PTR(Node) /**ReadonlyToken | PlusToken | MinusToken*/ readonlyToken;
        // PTR(TypeParameterDeclaration) typeParameter;
        // PTR(TypeNode) nameType;
        // PTR(Node) /**QuestionToken | PlusToken | MinusToken*/ questionToken;
        // PTR(TypeNode) type;

        auto typeParam = processTypeParameter(mappedTypeNode->typeParameter, genContext);
        auto hasNameType = !!mappedTypeNode->nameType;

        auto constrainType = getType(typeParam->getConstraint(), genContext);
        if (!constrainType)
        {
            return mlir::Type();
        }

        if (auto keyOfType = dyn_cast<mlir_ts::KeyOfType>(constrainType))
        {
            auto type = getType(mappedTypeNode->type, genContext);
            auto nameType = getType(mappedTypeNode->nameType, genContext);
            if (!type || hasNameType && !nameType)
            {
                return mlir::Type();
            }

            return getMappedType(type, nameType, constrainType);
        }

        // the key type param is visible only while resolving this mapped type; use a local
        // context copy so the caller's typeParamsWithArgs (incl. a pre-existing entry with
        // the same name) is never touched
        GenContext mappedTypeGenContext(genContext);
        auto processKeyItem = [&] (mlir::SmallVector<mlir_ts::FieldInfo> &fields, mlir::Type typeParamItem) {
            mappedTypeGenContext.typeParamsWithArgs.insert({typeParam->getName(), std::make_pair(typeParam, typeParamItem)});

            auto type = getType(mappedTypeNode->type, mappedTypeGenContext);
            if (!type)
            {
                // TODO: do we need to return error?
                // finish it
                return;
            }

            if (isa<mlir_ts::NeverType>(type))
            {
                return;
            }

            mlir::Type nameType = typeParamItem;
            if (hasNameType)
            {
                nameType = getType(mappedTypeNode->nameType, mappedTypeGenContext);
            }

            // remove type param
            mappedTypeGenContext.typeParamsWithArgs.erase(typeParam->getName());

            LLVM_DEBUG(llvm::dbgs() << "\n!! mapped type... \n\t type param: [" << typeParam->getName()
                                    << " \n\t\tconstraint item: " << typeParamItem << ", \n\t\tname: " << nameType
                                    << "] \n\ttype: " << type << "\n";);

            if (mth.isNoneType(nameType) || isa<mlir_ts::NeverType>(nameType) || mth.isEmptyTuple(nameType))
            {
                // filterting out
                LLVM_DEBUG(llvm::dbgs() << "\n!! mapped type... filtered.\n";);
                return;
            }

            if (auto literalType = dyn_cast<mlir_ts::LiteralType>(nameType))
            {
                LLVM_DEBUG(llvm::dbgs() << "\n!! mapped type... name: " << literalType << " type: " << type << "\n";);
                fields.push_back({literalType.getValue(), type, false, mlir_ts::AccessLevel::Public});
            }
            else
            {
                auto nameSubType = dyn_cast<mlir_ts::UnionType>(nameType);
                auto subType = dyn_cast<mlir_ts::UnionType>(type);
                if (nameSubType && subType)
                {
                    for (auto pair : llvm::zip(nameSubType, subType))
                    {
                        if (auto literalType = dyn_cast<mlir_ts::LiteralType>(std::get<0>(pair)))
                        {
                            auto mappedType = std::get<1>(pair);

                            LLVM_DEBUG(llvm::dbgs() << "\n!! mapped type... name: " << literalType << " type: " << mappedType << "\n";);
                            fields.push_back({literalType.getValue(), mappedType, false, mlir_ts::AccessLevel::Public});
                        }
                        else
                        {
                            llvm_unreachable("not implemented");
                        }
                    }
                }
                else
                {
                    llvm_unreachable("not implemented");
                }
            }
        };

        SmallVector<mlir_ts::FieldInfo> fields;
        if (auto unionType = dyn_cast<mlir_ts::UnionType>(constrainType))
        {
            for (auto typeParamItem : unionType.getTypes())
            {
                processKeyItem(fields, typeParamItem);
            }
        }
        else if (auto litType = dyn_cast<mlir_ts::LiteralType>(constrainType))
        {
            processKeyItem(fields, litType);
        }

        if (fields.size() == 0)
        {
            LLVM_DEBUG(llvm::dbgs() << "\n!! mapped type is empty for constrain: " << constrainType << ".\n";);
            emitWarning(loc(mappedTypeNode), "mapped type is empty for constrain: ")  << constrainType;
        }

        return getTupleType(fields);            
    }

    mlir_ts::VoidType MLIRGenImpl::getVoidType()
    {
        return mlir_ts::VoidType::get(builder.getContext());
    }

    mlir_ts::ByteType MLIRGenImpl::getByteType()
    {
        return mlir_ts::ByteType::get(builder.getContext());
    }

    mlir_ts::BooleanType MLIRGenImpl::getBooleanType()
    {
        return mlir_ts::BooleanType::get(builder.getContext());
    }

    mlir_ts::NumberType MLIRGenImpl::getNumberType()
    {
        return mlir_ts::NumberType::get(builder.getContext());
    }

    mlir_ts::BigIntType MLIRGenImpl::getBigIntType()
    {
        return mlir_ts::BigIntType::get(builder.getContext());
    }

    mlir::IndexType MLIRGenImpl::getIndexType()
    {
        return mlir::IndexType::get(builder.getContext());
    }

    mlir_ts::StringType MLIRGenImpl::getStringType()
    {
        return mlir_ts::StringType::get(builder.getContext());
    }

    mlir_ts::CharType MLIRGenImpl::getCharType()
    {
        return mlir_ts::CharType::get(builder.getContext());
    }

    mlir_ts::EnumType MLIRGenImpl::getEnumType()
    {
        return mlir_ts::EnumType::get(
            mlir::FlatSymbolRefAttr::get(builder.getContext(), StringRef{}), 
            builder.getI32Type(), 
            {});
    }

    mlir_ts::EnumType MLIRGenImpl::getEnumType(mlir::FlatSymbolRefAttr name, mlir::Type elementType, mlir::DictionaryAttr values)
    {
        return mlir_ts::EnumType::get(name, elementType ? elementType : builder.getI32Type(), values);
    }

    mlir_ts::ObjectStorageType MLIRGenImpl::getObjectStorageType(mlir::FlatSymbolRefAttr name)
    {
        return mlir_ts::ObjectStorageType::get(builder.getContext(), name);
    }

    mlir_ts::ClassStorageType MLIRGenImpl::getClassStorageType(mlir::FlatSymbolRefAttr name)
    {
        return mlir_ts::ClassStorageType::get(builder.getContext(), name);
    }

    mlir_ts::ClassType MLIRGenImpl::getClassType(mlir::FlatSymbolRefAttr name, mlir::Type storageType)
    {
        return mlir_ts::ClassType::get(name, storageType);
    }

    mlir_ts::NamespaceType MLIRGenImpl::getNamespaceType(mlir::StringRef name)
    {
        auto nsNameAttr = mlir::FlatSymbolRefAttr::get(builder.getContext(), name);
        return mlir_ts::NamespaceType::get(nsNameAttr);
    }

    mlir_ts::InterfaceType MLIRGenImpl::getInterfaceType(StringRef fullName)
    {
        auto interfaceFullNameSymbol = mlir::FlatSymbolRefAttr::get(builder.getContext(), fullName);
        return getInterfaceType(interfaceFullNameSymbol);
    }

    mlir_ts::InterfaceType MLIRGenImpl::getInterfaceType(mlir::FlatSymbolRefAttr name)
    {
        return mlir_ts::InterfaceType::get(name);
    }

    mlir::Type MLIRGenImpl::getConstArrayType(ArrayTypeNode arrayTypeAST, unsigned size, const GenContext &genContext)
    {
        auto type = getType(arrayTypeAST->elementType, genContext);
        return getConstArrayType(type, size);
    }

    mlir::Type MLIRGenImpl::getConstArrayType(mlir::Type elementType, unsigned size)
    {
        if (!elementType)
        {
            return mlir::Type();
        }

        return mlir_ts::ConstArrayType::get(elementType, size);
    }

    mlir::Type MLIRGenImpl::getArrayType(ArrayTypeNode arrayTypeAST, const GenContext &genContext)
    {
        auto type = getType(arrayTypeAST->elementType, genContext);
        return getArrayType(type);
    }

    mlir::Type MLIRGenImpl::getArrayType(mlir::Type elementType)
    {
        if (!elementType)
        {
            return mlir::Type();
        }

        return mlir_ts::ArrayType::get(elementType);
    }

    mlir::Type MLIRGenImpl::getValueRefType(mlir::Type elementType)
    {
        if (!elementType)
        {
            return mlir::Type();
        }

        return mlir_ts::ValueRefType::get(elementType);
    }

    mlir_ts::NamedGenericType MLIRGenImpl::getNamedGenericType(StringRef name)
    {
        return mlir_ts::NamedGenericType::get(builder.getContext(),
                                              mlir::FlatSymbolRefAttr::get(builder.getContext(), name));
    }

    mlir_ts::InferType MLIRGenImpl::getInferType(mlir::Type paramType)
    {
        assert(paramType);
        return mlir_ts::InferType::get(paramType);
    }

    mlir::Type MLIRGenImpl::getConditionalType(mlir::Type checkType, mlir::Type extendsType, mlir::Type trueType, mlir::Type falseType)
    {
        assert(checkType);
        assert(extendsType);
        assert(trueType);
        assert(falseType);

        if (!checkType || !extendsType || !trueType || !falseType)
        {
            return mlir::Type();
        }

        return mlir_ts::ConditionalType::get(checkType, extendsType, trueType, falseType);
    }

    mlir::Type MLIRGenImpl::getIndexAccessType(mlir::Type index, mlir::Type indexAccess)
    {
        assert(index);
        assert(indexAccess);

        if (!index || !indexAccess)
        {
            return mlir::Type();
        }

        return mlir_ts::IndexAccessType::get(index, indexAccess);
    }    

    mlir::Type MLIRGenImpl::getKeyOfType(mlir::Type type)
    {
        assert(type);

        if (!type)
        {
            return mlir::Type();
        }

        return mlir_ts::KeyOfType::get(type);
    }      

    mlir::Type MLIRGenImpl::getMappedType(mlir::Type elementType, mlir::Type nameType, mlir::Type constrainType)
    {
        assert(elementType);
        assert(nameType);
        assert(constrainType);

        if (!elementType || !nameType || !constrainType)
        {
            return mlir::Type();
        }

        return mlir_ts::MappedType::get(elementType, nameType, constrainType);
    }    

    mlir_ts::TypeReferenceType MLIRGenImpl::getTypeReferenceType(mlir::StringRef nameRef, mlir::SmallVector<mlir::Type> &types)
    {
        return mlir_ts::TypeReferenceType::get(builder.getContext(), mlir::FlatSymbolRefAttr::get(builder.getContext(), nameRef), types);
    }    

    mlir::Value MLIRGenImpl::getUndefined(mlir::Location location)
    {
        return builder.create<mlir_ts::UndefOp>(location, getUndefinedType());
    }

    mlir::Value MLIRGenImpl::getInfinity(mlir::Location location)
    {
#ifdef NUMBER_F64
        union { double dbl; int64_t int64; } val{};
        val.int64 = 0x7FF0000000000000;
        return builder.create<mlir_ts::ConstantOp>(location, getNumberType(), builder.getF64FloatAttr(val.dbl));
#else
        union { float flt; int32_t int32; } val;
        val.int32 = 0x7FF00000;
        return builder.create<mlir_ts::ConstantOp>(location, getNumberType(), builder.getF32FloatAttr(val.int32));
#endif
    }

    mlir::Value MLIRGenImpl::getNaN(mlir::Location location)
    {
#ifdef NUMBER_F64
        union { double dbl; int64_t int64; } val{};
        val.int64 = 0x7FF0000000000001;
        return builder.create<mlir_ts::ConstantOp>(location, getNumberType(), builder.getF64FloatAttr(val.dbl));
#else
        union { float flt; int32_t int32; } val;
        val.int32 = 0x7FF00001;
        return builder.create<mlir_ts::ConstantOp>(location, getNumberType(), builder.getF32FloatAttr(val.int32));
#endif
    }

    std::pair<mlir::Attribute, mlir::LogicalResult> MLIRGenImpl::getNameFromComputedPropertyName(Node name, const GenContext &genContext)
    {
        if (name == SyntaxKind::ComputedPropertyName)
        {
            MLIRCodeLogic mcl(builder, compileOptions);
            auto result = mlirGen(name.as<ComputedPropertyName>(), genContext);
            auto value = V(result);
            LLVM_DEBUG(llvm::dbgs() << "!! ComputedPropertyName: " << value << "\n";);
            auto attr = mcl.ExtractAttr(value);
            if (!attr)
            {
                emitError(loc(name), "not supported 'Computed Property Name' expression");
            }

            return {attr, attr ? mlir::success() : mlir::failure()};
        }

        return {mlir::Attribute(), mlir::success()};
    }

    mlir::Attribute MLIRGenImpl::TupleFieldName(Node name, const GenContext &genContext)
    {
        auto namePtr = MLIRHelper::getName(name, stringAllocator);
        if (namePtr.empty())
        {
            auto [attrComputed, attrResult] = getNameFromComputedPropertyName(name, genContext);
            if (attrComputed || mlir::failed(attrResult))
            {
                return attrComputed;
            }
                        
            MLIRCodeLogic mcl(builder, compileOptions);
            auto result = mlirGen(name.as<Expression>(), genContext);
            auto value = V(result);
            auto attr = mcl.ExtractAttr(value);
            if (!attr)
            {
                emitError(loc(name), "not supported name");
            }

            return attr;
        }

        return MLIRHelper::TupleFieldName(namePtr, builder.getContext());
    }

    std::pair<bool, mlir::LogicalResult> MLIRGenImpl::getTupleFieldInfo(TupleTypeNode tupleType, mlir::SmallVector<mlir_ts::FieldInfo> &types,
                           const GenContext &genContext)
    {
        MLIRCodeLogic mcl(builder, compileOptions);
        mlir::Attribute attrVal;
        auto arrayMode = true;
        auto index = 0;
        for (auto typeItem : tupleType->elements)
        {
            if (typeItem == SyntaxKind::NamedTupleMember)
            {
                auto namedTupleMember = typeItem.as<NamedTupleMember>();

                auto type = getType(namedTupleMember->type, genContext);
                if (!type)
                {
                    return {arrayMode, mlir::failure()};
                }

                types.push_back({TupleFieldName(namedTupleMember->name, genContext), type, false, mlir_ts::AccessLevel::Public});
                arrayMode = false;
            }
            else if (typeItem == SyntaxKind::LiteralType)
            {
                auto literalTypeNode = typeItem.as<LiteralTypeNode>();
                auto result = mlirGen(literalTypeNode->literal.as<Expression>(), genContext);
                if (result.failed_or_no_value())
                {
                    return {arrayMode, mlir::failure()};
                }

                auto literalValue = V(result);
                auto constantOp = literalValue.getDefiningOp<mlir_ts::ConstantOp>();

                assert(constantOp);
                attrVal = constantOp.getValueAttr();

                if (arrayMode)
                {
                    types.push_back({builder.getIntegerAttr(builder.getI32Type(), index), constantOp.getType(), false, mlir_ts::AccessLevel::Public});
                }

                index++;
                continue;
            }
            else
            {
                auto type = getType(typeItem, genContext);
                if (!type)
                {
                    return {arrayMode, mlir::failure()};
                }

                types.push_back({attrVal, type, false, mlir_ts::AccessLevel::Public});
            }

            attrVal = mlir::Attribute();
        }

        return {arrayMode, mlir::success()};
    }

    mlir::LogicalResult MLIRGenImpl::getTupleFieldInfo(TypeLiteralNode typeLiteral, mlir::SmallVector<mlir_ts::FieldInfo> &types,
                           const GenContext &genContext)
    {
        MLIRCodeLogic mcl(builder, compileOptions);
        for (auto typeItem : typeLiteral->members)
        {
            SyntaxKind kind = typeItem;
            if (kind == SyntaxKind::PropertySignature)
            {
                auto propertySignature = typeItem.as<PropertySignature>();

                auto originalType = getType(propertySignature->type, genContext);
                if (!originalType)
                {
                    return mlir::failure();
                }

                auto type = mcl.getEffectiveFunctionTypeForTupleField(originalType);

                assert(type);
                types.push_back({TupleFieldName(propertySignature->name, genContext), type, false, mlir_ts::AccessLevel::Public});
            }
            else if (kind == SyntaxKind::MethodSignature)
            {
                auto methodSignature = typeItem.as<MethodSignature>();

                auto type = getType(typeItem, genContext);
                if (!type)
                {
                    return mlir::failure();
                }

                types.push_back({TupleFieldName(methodSignature->name, genContext), type, false, mlir_ts::AccessLevel::Public});
            }
            else if (kind == SyntaxKind::ConstructSignature)
            {
                auto type = getType(typeItem, genContext);
                if (!type)
                {
                    return mlir::failure();
                }

                types.push_back({MLIRHelper::TupleFieldName(NEW_CTOR_METHOD_NAME, builder.getContext()), type, false, mlir_ts::AccessLevel::Public});
            }            
            else if (kind == SyntaxKind::IndexSignature)
            {
                auto type = getType(typeItem, genContext);
                if (!type)
                {
                    return mlir::failure();
                }

                types.push_back({MLIRHelper::TupleFieldName(INDEX_ACCESS_GET_FIELD_NAME, builder.getContext()), mth.getIndexGetFunctionType(type), false, mlir_ts::AccessLevel::Public});
                types.push_back({MLIRHelper::TupleFieldName(INDEX_ACCESS_SET_FIELD_NAME, builder.getContext()), mth.getIndexSetFunctionType(type), false, mlir_ts::AccessLevel::Public});
            }
            else if (kind == SyntaxKind::CallSignature)
            {
                auto type = getType(typeItem, genContext);
                if (!type)
                {
                    return mlir::failure();
                }

                types.push_back({MLIRHelper::TupleFieldName(CALL_FIELD_NAME, builder.getContext()), type, false, mlir_ts::AccessLevel::Public});
            }
            else
            {
                llvm_unreachable("not implemented");
            }
        }

        return mlir::success();
    }

    mlir::Type MLIRGenImpl::getConstTupleType(TupleTypeNode tupleType, const GenContext &genContext)
    {
        mlir::SmallVector<mlir_ts::FieldInfo> types;
        auto [arrayMode, result] = getTupleFieldInfo(tupleType, types, genContext);
        if (mlir::failed(result))
        {
            return mlir::Type();
        }

        return getConstTupleType(types);
    }

    mlir_ts::ConstTupleType MLIRGenImpl::getConstTupleType(mlir::SmallVector<mlir_ts::FieldInfo> &fieldInfos)
    {
        return mlir_ts::ConstTupleType::get(builder.getContext(), fieldInfos);
    }

    mlir::Type MLIRGenImpl::getTupleType(TupleTypeNode tupleType, const GenContext &genContext)
    {
        mlir::SmallVector<mlir_ts::FieldInfo> types;
        auto [arrayMode, result] = getTupleFieldInfo(tupleType, types, genContext);
        if (mlir::failed(result))
        {
            return mlir::Type();
        }

        if (arrayMode && types.size() == 1)
        {
            return getArrayType(types.front().type);
        }

        return getTupleType(types);
    }

    mlir::Type MLIRGenImpl::getTupleType(TypeLiteralNode typeLiteral, const GenContext &genContext)
    {
        mlir::SmallVector<mlir_ts::FieldInfo> types;
        auto result = getTupleFieldInfo(typeLiteral, types, genContext);
        if (mlir::failed(result))
        {
            return mlir::Type();
        }

        // TODO: remove the following hack
        // TODO: this is hack, add type IndexSignatureFunctionType to see if it is index declaration
        if (types.size() == 1)
        {
            auto indexAccessName = MLIRHelper::TupleFieldName(INDEX_ACCESS_FIELD_NAME, builder.getContext());
            if (types.front().id == indexAccessName)
            {
                auto [arg, res] = mth.getIndexSignatureArgumentAndResultTypes(types.front().type);
                if (auto elementTypeOfIndexSignature = arg)
                {
                    auto arrayType = getArrayType(elementTypeOfIndexSignature);
                    LLVM_DEBUG(llvm::dbgs() << "\n!! this is array type: " << arrayType << "\n";);
                    return arrayType;
                }
            }
        }

        // == TODO: remove the following hack
        // TODO: this is hack, add type IndexSignatureFunctionType to see if it is index declaration
        if (types.size() == 2)
        {
            mlir::Type indexSignatureType;
            auto lengthName = MLIRHelper::TupleFieldName(LENGTH_FIELD_NAME, builder.getContext());
            auto indexAccessName = MLIRHelper::TupleFieldName(INDEX_ACCESS_FIELD_NAME, builder.getContext());
            if (types.front().id == lengthName && types.back().id == indexAccessName)
            {
                indexSignatureType = types.back().type;
            }
            
            if (types.back().id == lengthName && types.front().id == indexAccessName)
            {
                indexSignatureType = types.front().type;
            }

            if (indexSignatureType)
            {
                // TODO: this is hack, add type IndexSignatureFunctionType to see if it is index declaration
                auto [arg, res] = mth.getIndexSignatureArgumentAndResultTypes(indexSignatureType);
                if (auto elementTypeOfIndexSignature = arg)
                {
                    auto arrayType = getArrayType(elementTypeOfIndexSignature);
                    LLVM_DEBUG(llvm::dbgs() << "\n!! this is array type: " << arrayType << "\n";);
                    return arrayType;
                }
            }
        }        

        return getTupleType(types);
    }

    mlir::Type MLIRGenImpl::getTupleType(mlir::SmallVector<mlir_ts::FieldInfo> &fieldInfos)
    {
        return mlir_ts::TupleType::get(builder.getContext(), fieldInfos);
    }

    mlir_ts::ObjectType MLIRGenImpl::getObjectType(mlir::Type type)
    {
        return mlir_ts::ObjectType::get(type);
    }

    mlir_ts::OpaqueType MLIRGenImpl::getOpaqueType()
    {
        return mlir_ts::OpaqueType::get(builder.getContext());
    }    

    mlir_ts::BoundFunctionType MLIRGenImpl::getBoundFunctionType(mlir_ts::FunctionType funcType)
    {
        return mlir_ts::BoundFunctionType::get(builder.getContext(), funcType);
    }

    mlir_ts::BoundFunctionType MLIRGenImpl::getBoundFunctionType(ArrayRef<mlir::Type> inputs, ArrayRef<mlir::Type> results,
                                                    bool isVarArg)
    {
        return mlir_ts::BoundFunctionType::get(builder.getContext(), inputs, results, isVarArg);
    }

    mlir_ts::FunctionType MLIRGenImpl::getFunctionType(ArrayRef<mlir::Type> inputs, ArrayRef<mlir::Type> results,
                                          bool isVarArg)
    {
        return mlir_ts::FunctionType::get(builder.getContext(), inputs, results, isVarArg);
    }

    mlir_ts::ExtensionFunctionType MLIRGenImpl::getExtensionFunctionType(mlir_ts::FunctionType funcType)
    {
        return mlir_ts::ExtensionFunctionType::get(builder.getContext(), funcType);
    }

    mlir::Type MLIRGenImpl::getSignature(SignatureDeclarationBase signature, const GenContext &genContext)
    {
        GenContext genericTypeGenContext(genContext);

        // preparing generic context to resolve types
        if (signature->typeParameters.size())
        {
            llvm::SmallVector<TypeParameterDOM::TypePtr> typeParameters;
            if (mlir::failed(
                    processTypeParameters(signature->typeParameters, typeParameters, genericTypeGenContext)))
            {
                return mlir::Type();
            }

            auto [result, hasAnyNamedGenericType] =
                zipTypeParametersWithArguments(loc(signature), typeParameters, signature->typeArguments,
                                               genericTypeGenContext.typeParamsWithArgs, genericTypeGenContext);

            if (mlir::failed(result))
            {
                return mlir::Type();
            }
        }

        auto resultType = getType(signature->type, genericTypeGenContext);
        if (!resultType && !genContext.allowPartialResolve)
        {
            return mlir::Type();
        }

        SmallVector<mlir::Type> argTypes;
        auto isVarArg = false;
        for (auto paramItem : signature->parameters)
        {
            auto type = getType(paramItem->type, genericTypeGenContext);
            if (!type)
            {
                return mlir::Type();
            }

            if (paramItem->questionToken)
            {
                type = getOptionalType(type);
            }

            argTypes.push_back(type);

            isVarArg |= !!paramItem->dotDotDotToken;
        }

        auto funcType = mlir_ts::FunctionType::get(builder.getContext(), argTypes, resultType, isVarArg);
        return funcType;
    }

    mlir::Type MLIRGenImpl::getFunctionType(SignatureDeclarationBase signature, const GenContext &genContext)
    {
        auto signatureType = getSignature(signature, genContext);
        if (!signatureType)
        {
            return mlir::Type();
        }

        auto funcType = mlir_ts::HybridFunctionType::get(builder.getContext(), mlir::cast<mlir_ts::FunctionType>(signatureType));
        return funcType;
    }

    mlir::Type MLIRGenImpl::getConstructorType(SignatureDeclarationBase signature, const GenContext &genContext)
    {
        auto signatureType = getSignature(signature, genContext);
        if (!signatureType)
        {
            return mlir::Type();
        }

        auto funcType = mlir_ts::ConstructFunctionType::get(
            builder.getContext(), 
            mlir::cast<mlir_ts::FunctionType>(signatureType), 
            hasModifier(signature, SyntaxKind::AbstractKeyword));
        return funcType;
    }

    mlir::Type MLIRGenImpl::getCallSignature(CallSignatureDeclaration signature, const GenContext &genContext)
    {
        auto signatureType = getSignature(signature, genContext);
        if (!signatureType)
        {
            return mlir::Type();
        }

        auto funcType = mlir_ts::HybridFunctionType::get(builder.getContext(), mlir::cast<mlir_ts::FunctionType>(signatureType));
        return funcType;
    }

    mlir::Type MLIRGenImpl::getConstructSignature(ConstructSignatureDeclaration constructSignature,
                                                const GenContext &genContext)
    {
        return getSignature(constructSignature, genContext);
    }

    mlir::Type MLIRGenImpl::getMethodSignature(MethodSignature methodSignature, const GenContext &genContext)
    {
        return getSignature(methodSignature, genContext);
    }

    mlir::Type MLIRGenImpl::getIndexSignature(IndexSignatureDeclaration indexSignature, const GenContext &genContext)
    {
        return getSignature(indexSignature, genContext);
    }

    mlir::Type MLIRGenImpl::getUnionType(UnionTypeNode unionTypeNode, const GenContext &genContext)
    {
        MLIRTypeHelper::UnionTypeProcessContext unionContext = {};
        for (auto typeItem : unionTypeNode->types)
        {
            auto type = getType(typeItem, genContext);
            if (!type)
            {
                LLVM_DEBUG(llvm::dbgs() << "\n!! wrong type: " << loc(typeItem) << "\n";);

                //llvm_unreachable("wrong type");
                return mlir::Type();
            }

            mth.processUnionTypeItem(type, unionContext);
        }

        // default wide types
        if (unionContext.isAny)
        {
            return getAnyType();
        }

        return mth.getUnionTypeMergeTypes(loc(unionTypeNode), unionContext, false, false);
    }

    mlir::Type MLIRGenImpl::getUnionType(mlir::Location location, mlir::Type type1, mlir::Type type2)
    {
        if (mth.isNoneType(type1) || mth.isNoneType(type2))
        {
            return mlir::Type();
        }

        LLVM_DEBUG(llvm::dbgs() << "\n!! join: " << type1 << " | " << type2;);

        auto resType = mth.getUnionType(location, type1, type2, false);

        LLVM_DEBUG(llvm::dbgs() << " = " << resType << "\n";);

        return resType;
    }

    mlir::Type MLIRGenImpl::getUnionType(mlir::SmallVector<mlir::Type> &types)
    {
        return mth.getUnionType(types);
    }

    mlir::LogicalResult MLIRGenImpl::processIntersectionType(InterfaceInfo::TypePtr newInterfaceInfo, mlir::Type type, bool conditional)
    {
        if (auto ifaceType = dyn_cast<mlir_ts::InterfaceType>(type))
        {
            auto srcInterfaceInfo = getInterfaceInfoByFullName(ifaceType.getName().getValue());
            assert(srcInterfaceInfo);
            newInterfaceInfo->extends.push_back({-1, srcInterfaceInfo});
        }
        else if (auto tupleType = dyn_cast<mlir_ts::TupleType>(type))
        {
            mergeInterfaces(newInterfaceInfo, tupleType, conditional);
        }
        else if (auto constTupleType = dyn_cast<mlir_ts::ConstTupleType>(type))
        {
            mergeInterfaces(newInterfaceInfo, mlir::cast<mlir_ts::TupleType>(mth.removeConstType(constTupleType)), conditional);
        }
        else if (auto objectType = dyn_cast<mlir_ts::ObjectType>(type))
        {
            // boxed object literal (docs/object-literal-boxing-design.md): a generic type
            // parameter bound to a method-bearing literal (e.g. `M` in `D & M`) now resolves
            // to ObjectType rather than a tuple directly -- look through its storage type,
            // same as MLIRTypeHelper::getFields does for property access.
            return processIntersectionType(newInterfaceInfo, objectType.getStorageType(), conditional);
        }
        else if (auto unionType = dyn_cast<mlir_ts::UnionType>(type))
        {
            for (auto type : unionType.getTypes())
            {
                if (mlir::failed(processIntersectionType(newInterfaceInfo, type, true)))
                {
                    return mlir::failure();
                }
            }            
        }              
        else
        {
            return mlir::failure();
        }      

        return mlir::success();
    }

    mlir::Type MLIRGenImpl::getIntersectionType(IntersectionTypeNode intersectionTypeNode, const GenContext &genContext)
    {
        mlir_ts::InterfaceType baseInterfaceType;
        mlir_ts::TupleType baseTupleType;
        mlir::SmallVector<mlir::Type> types;
        mlir::SmallVector<mlir::Type> typesForUnion;
        auto allTupleTypesConst = true;
        auto unionTypes = false;
        for (auto typeItem : intersectionTypeNode->types)
        {
            auto type = getType(typeItem, genContext);
            if (!type)
            {
                return mlir::Type();
            }

            if (auto tupleType = dyn_cast<mlir_ts::TupleType>(type))
            {
                allTupleTypesConst = false;
                if (!baseTupleType)
                {
                    baseTupleType = tupleType;
                }
            }

            if (auto constTupleType = dyn_cast<mlir_ts::ConstTupleType>(type))
            {
                if (!baseTupleType)
                {
                    baseTupleType = mlir_ts::TupleType::get(builder.getContext(), constTupleType.getFields());
                }
            }

            if (auto ifaceType = dyn_cast<mlir_ts::InterfaceType>(type))
            {
                if (!baseInterfaceType)
                {
                    baseInterfaceType = ifaceType;
                }
            }

            types.push_back(type);
        }

        if (types.size() == 0)
        {
            // this is never type
            return getNeverType();
        }

        if (types.size() == 1)
        {
            return types.front();
        }

        // find base type
        if (baseInterfaceType)
        {
            auto declareInterface = false;
            auto newInterfaceInfo = newInterfaceType(intersectionTypeNode, declareInterface, genContext);
            if (declareInterface)
            {
                // merge all interfaces;
                for (auto type : types)
                {
                    if (mlir::failed(processIntersectionType(newInterfaceInfo, type)))
                    {
                        emitWarning(loc(intersectionTypeNode), "Intersection can't be resolved.");
                        return getIntersectionType(types);
                    }
                }
            }

            newInterfaceInfo->recalcOffsets();

            // canonical (extends, then own methods, then own fields) slot numbering for the
            // synthesized interface's OWN members - see InterfaceInfo::assignCanonicalVirtualIndexes
            // (MLIRGenStore.h) and mlirGen(InterfaceDeclaration)'s equivalent call
            // (MLIRGenInterfaces.cpp), which this construction path (intersection types) bypasses
            // entirely, being a separate programmatic InterfaceInfo builder rather than an AST walk.
            newInterfaceInfo->assignCanonicalVirtualIndexes();

            return newInterfaceInfo->interfaceType;
        }

        if (baseTupleType)
        {
            auto anyTypesInBaseTupleType = baseTupleType.getFields().size() > 0;

            SmallVector<::mlir::typescript::FieldInfo> typesForNewTuple;
            for (auto type : types)
            {
                LLVM_DEBUG(llvm::dbgs() << "\n!! processing ... & {...} :" << type << "\n";);

                // umwrap optional
                if (!anyTypesInBaseTupleType)
                {
                    type = mth.stripOptionalType(type);
                }

                if (auto tupleType = dyn_cast<mlir_ts::TupleType>(type))
                {
                    allTupleTypesConst = false;
                    for (auto field : tupleType.getFields())
                    {
                        typesForNewTuple.push_back(field);
                    }
                }
                else if (auto constTupleType = dyn_cast<mlir_ts::ConstTupleType>(type))
                {
                    for (auto field : constTupleType.getFields())
                    {
                        typesForNewTuple.push_back(field);
                    }
                }
                else if (auto objectType = dyn_cast<mlir_ts::ObjectType>(type))
                {
                    // boxed object literal (docs/object-literal-boxing-design.md): a generic
                    // type parameter bound to a method-bearing literal (e.g. M in D & M) now
                    // resolves to ObjectType rather than a tuple directly -- look through its
                    // storage type, same as MLIRTypeHelper::getFields does for property access.
                    allTupleTypesConst = false;
                    SmallVector<mlir_ts::FieldInfo> objectFields;
                    if (mlir::succeeded(mth.getFields(objectType, objectFields)))
                    {
                        for (auto field : objectFields)
                        {
                            typesForNewTuple.push_back(field);
                        }
                    }
                }
                else if (auto unionType = dyn_cast<mlir_ts::UnionType>(type))
                {
                    if (!anyTypesInBaseTupleType)
                    {
                        unionTypes = true;
                        for (auto subType : unionType.getTypes())
                        {
                            if (subType == getNullType() || subType == getUndefinedType())
                            {
                                continue;
                            }

                            typesForUnion.push_back(subType);
                        }
                    }                    
                }
                else
                {
                    if (!anyTypesInBaseTupleType)
                    {
                        unionTypes = true; 
                        typesForUnion.push_back(type);
                    }
                    else
                    {
                        // no intersection
                        return getNeverType();
                    }
                }
            }

            if (unionTypes)
            {
                auto resUnion = getUnionType(typesForUnion);
                LLVM_DEBUG(llvm::dbgs() << "\n!! &=: " << resUnion << "\n";);
                return resUnion;                
            }

            auto resultType = allTupleTypesConst 
                ? (mlir::Type)getConstTupleType(typesForNewTuple)
                : (mlir::Type)getTupleType(typesForNewTuple);

            LLVM_DEBUG(llvm::dbgs() << "\n!! &=: " << resultType << "\n";);

            return resultType;
        }

        // calculate of intersection between types and literal types
        mlir::Type resType;
        for (auto typeItem : types)
        {
            if (!resType)
            {
                resType = typeItem;
                continue;
            }

            LLVM_DEBUG(llvm::dbgs() << "\n!! &: " << resType << " & " << typeItem;);

            resType = AndType(resType, typeItem);

            LLVM_DEBUG(llvm::dbgs() << " = " << resType << "\n";);

            if (isa<mlir_ts::NeverType>(resType))
            {
                return getNeverType();
            }
        }

        if (resType)
        {
            return resType;
        }

        return getNeverType();
    }

    mlir::Type MLIRGenImpl::getIntersectionType(mlir::Type type1, mlir::Type type2)
    {
        if (!type1 || !type2)
        {
            return mlir::Type();
        }

        LLVM_DEBUG(llvm::dbgs() << "\n!! intersection: " << type1 << " & " << type2;);

        auto resType = mth.getIntersectionType(type1, type2);

        LLVM_DEBUG(llvm::dbgs() << " = " << resType << "\n";);

        return resType;
    }

    mlir::Type MLIRGenImpl::getIntersectionType(mlir::SmallVector<mlir::Type> &types)
    {
        return mth.getIntersectionType(types);
    }

    mlir::Type MLIRGenImpl::AndType(mlir::Type left, mlir::Type right)
    {
        // TODO: 00types_unknown1.ts contains examples of results with & | for types,  T & {} == T & {}, T | {} == T |
        // {}, (they do not change)
        if (left == right)
        {
            return left;
        }

        if (auto literalType = dyn_cast<mlir_ts::LiteralType>(right))
        {
            if (literalType.getElementType() == left)
            {
                if (isa<mlir_ts::LiteralType>(left))
                {
                    return getNeverType();
                }

                return literalType;
            }
        }

        if (auto leftUnionType = dyn_cast<mlir_ts::UnionType>(left))
        {
            return AndUnionType(leftUnionType, right);
        }

        if (auto unionType = dyn_cast<mlir_ts::UnionType>(right))
        {
            mlir::SmallPtrSet<mlir::Type, 2> newUniqueTypes;
            for (auto unionTypeItem : unionType.getTypes())
            {
                auto resType = AndType(left, unionTypeItem);
                newUniqueTypes.insert(resType);
            }

            SmallVector<mlir::Type> newTypes;
            for (auto uniqType : newUniqueTypes)
            {
                newTypes.push_back(uniqType);
            }

            return getUnionType(newTypes);
        }

        if (isa<mlir_ts::NullType>(left))
        {

            if (mth.isValueType(right))
            {
                return getNeverType();
            }

            return left;
        }

        if (isa<mlir_ts::NullType>(right))
        {

            if (mth.isValueType(left))
            {
                return getNeverType();
            }

            return right;
        }

        if (isa<mlir_ts::NullType>(left))
        {

            if (mth.isValueType(right))
            {
                return getNeverType();
            }

            return left;
        }

        if (isa<mlir_ts::AnyType>(left) || isa<mlir_ts::UnknownType>(left))
        {
            return right;
        }

        if (isa<mlir_ts::AnyType>(right) || isa<mlir_ts::UnknownType>(right))
        {
            return left;
        }

        // TODO: should I add, interface, tuple types here?
        // PS: string & { __b: number } creating type "string & { __b: number }".

        return getIntersectionType(left, right);
    }

    mlir::Type MLIRGenImpl::AndUnionType(mlir_ts::UnionType leftUnion, mlir::Type right)
    {
        mlir::SmallPtrSet<mlir::Type, 2> newUniqueTypes;
        for (auto unionTypeItem : leftUnion.getTypes())
        {
            auto resType = AndType(unionTypeItem, right);
            newUniqueTypes.insert(resType);
        }

        SmallVector<mlir::Type> newTypes;
        for (auto uniqType : newUniqueTypes)
        {
            newTypes.push_back(uniqType);
        }

        return getUnionType(newTypes);
    }

    InterfaceInfo::TypePtr MLIRGenImpl::newInterfaceType(IntersectionTypeNode intersectionTypeNode, bool &declareInterface,
                                            const GenContext &genContext)
    {
        auto newName = MLIRHelper::getAnonymousName(loc_check(intersectionTypeNode), "ifce", "");

        // clone into new interface
        auto interfaceInfo = mlirGenInterfaceInfo(newName, declareInterface, genContext);

        return interfaceInfo;
    }

    mlir::LogicalResult MLIRGenImpl::mergeInterfaces(InterfaceInfo::TypePtr dest, mlir_ts::TupleType src, bool conditional)
    {
        // TODO: use it to merge with TupleType
        for (auto &item : src.getFields())
        {
            // InterfaceFieldInfo is {id, type, isConditional, interfacePosIndex, virtualIndex} - the
            // getNextVTableMemberIndex() value belongs in virtualIndex, not interfacePosIndex (a
            // separate, unrelated field only meaningful for methods - MLIRGenClasses.cpp). This
            // mismatch previously left virtualIndex zero-initialized for every field merged in from
            // an intersection type's own `{ ... }` member (e.g. `F1 & F2 & { c: number }`'s `c`),
            // masked only because getVirtualTable() used to overwrite virtualIndex as a side effect
            // of every cast - see docs/interface-vtable-simplification-design.md §4.
            dest->fields.push_back({item.id, item.type, item.isConditional || conditional, 0, dest->getNextVTableMemberIndex()});
        }

        return mlir::success();
    }

    mlir::Type MLIRGenImpl::getParenthesizedType(ParenthesizedTypeNode parenthesizedTypeNode, const GenContext &genContext)
    {
        return getType(parenthesizedTypeNode->type, genContext);
    }

    mlir::Type MLIRGenImpl::getLiteralType(LiteralTypeNode literalTypeNode)
    {
        GenContext genContext{};
        genContext.dummyRun = true;
        genContext.allowPartialResolve = true;
        auto result = mlirGen(literalTypeNode->literal.as<Expression>(), genContext);
        auto value = V(result);
        auto type = value.getType();

        if (auto literalType = dyn_cast<mlir_ts::LiteralType>(type))
        {
            return literalType;
        }

        auto constantOp = value.getDefiningOp<mlir_ts::ConstantOp>();
        if (constantOp)
        {
            auto valueAttr = value.getDefiningOp<mlir_ts::ConstantOp>().getValueAttr();
            auto literalType = mlir_ts::LiteralType::get(valueAttr, type);
            return literalType;
        }

        auto nullOp = value.getDefiningOp<mlir_ts::NullOp>();
        if (nullOp)
        {
            return getNullType();
        }

        LLVM_DEBUG(llvm::dbgs() << "\n!! value of literal: " << value << "\n";);

        llvm_unreachable("not implemented");
    }

    mlir::Type MLIRGenImpl::getOptionalType(OptionalTypeNode optionalTypeNode, const GenContext &genContext)
    {
        return getOptionalType(getType(optionalTypeNode->type, genContext));
    }

    mlir::Type MLIRGenImpl::getOptionalType(mlir::Type type)
    {
        if (!type)
        {
            return mlir::Type();
        }

        if (isa<mlir_ts::OptionalType>(type))
        {
            return type;
        }        

        return mlir_ts::OptionalType::get(type);
    }

    mlir::Type MLIRGenImpl::getRestType(RestTypeNode restTypeNode, const GenContext &genContext)
    {
        auto arrayType = getType(restTypeNode->type, genContext);
        if (!arrayType)
        {
            return mlir::Type();
        }

        return getConstArrayType(mlir::cast<mlir_ts::ArrayType>(arrayType).getElementType(), 0);
    }

    mlir_ts::AnyType MLIRGenImpl::getAnyType()
    {
        return mlir_ts::AnyType::get(builder.getContext());
    }

    mlir_ts::UnknownType MLIRGenImpl::getUnknownType()
    {
        return mlir_ts::UnknownType::get(builder.getContext());
    }

    mlir_ts::NeverType MLIRGenImpl::getNeverType()
    {
        return mlir_ts::NeverType::get(builder.getContext());
    }

    mlir_ts::ConstType MLIRGenImpl::getConstType()
    {
        return mlir_ts::ConstType::get(builder.getContext());
    }    

    mlir_ts::SymbolType MLIRGenImpl::getSymbolType()
    {
        return mlir_ts::SymbolType::get(builder.getContext());
    }

    mlir_ts::UndefinedType MLIRGenImpl::getUndefinedType()
    {
        return mlir_ts::UndefinedType::get(builder.getContext());
    }

    mlir_ts::NullType MLIRGenImpl::getNullType()
    {
        return mlir_ts::NullType::get(builder.getContext());
    }

} // namespace mlirgen
} // namespace typescript
