// Class declaration/member/vtable code generation methods of MLIRGenImpl (see MLIRGenImpl.h).

#include "MLIRGenImpl.h"

namespace typescript
{
namespace mlirgen
{

    mlir::LogicalResult MLIRGenImpl::registerGenericClass(ClassLikeDeclaration classDeclarationAST, const GenContext &genContext)
    {
        auto name = className(classDeclarationAST, genContext);
        if (!name.empty())
        {
            auto namePtr = StringRef(name).copy(stringAllocator);
            auto fullNamePtr = getFullNamespaceName(namePtr);
            if (fullNameGenericClassesMap.count(fullNamePtr))
            {
                return mlir::success();
            }

            llvm::SmallVector<TypeParameterDOM::TypePtr> typeParameters;
            if (mlir::failed(processTypeParameters(classDeclarationAST->typeParameters, typeParameters, genContext)))
            {
                return mlir::failure();
            }

            // register class
            GenericClassInfo::TypePtr newGenericClassPtr = std::make_shared<GenericClassInfo>();
            newGenericClassPtr->name = namePtr;
            newGenericClassPtr->fullName = fullNamePtr;
            newGenericClassPtr->typeParams = typeParameters;
            newGenericClassPtr->classDeclaration = classDeclarationAST;
            newGenericClassPtr->elementNamespace = currentNamespace;
            newGenericClassPtr->sourceFile = sourceFile;
            newGenericClassPtr->fileName = mainSourceFileName;

            mlirGenClassType(newGenericClassPtr, genContext);

            getGenericClassesMap().insert({namePtr, newGenericClassPtr});
            fullNameGenericClassesMap.insert(fullNamePtr, newGenericClassPtr);

            return mlir::success();
        }

        return mlir::failure();
    }

    mlir::LogicalResult MLIRGenImpl::mlirGen(ClassDeclaration classDeclarationAST, const GenContext &genContext)
    {
        mlir::OpBuilder::InsertionGuard guard(builder);
        builder.setInsertionPointToStart(theModule.getBody());

        auto value = mlirGen(classDeclarationAST.as<ClassLikeDeclaration>(), genContext);
        return std::get<0>(value);
    }

    std::pair<mlir::LogicalResult, mlir::StringRef> MLIRGenImpl::mlirGen(ClassLikeDeclaration classDeclarationAST,
                                                            const GenContext &genContext)
    {
        // do not proceed for Generic Interfaces for declaration
        auto isGenericClass = classDeclarationAST->typeParameters.size() > 0;
        if (isGenericClass && genContext.typeParamsWithArgs.size() == 0)
        {
            return {registerGenericClass(classDeclarationAST, genContext), ""};
        }

        auto newClassPtr = mlirGenClassInfo(classDeclarationAST, genContext);
        if (!newClassPtr)
        {
            return {mlir::failure(), ""};
        }

        // do not process specialized class second time;
        if (isGenericClass && genContext.typeParamsWithArgs.size() > 0)
        {
            // TODO: investigate why classType is provided already for class
            if (testProcessingState(newClassPtr, ProcessingStages::Processing, genContext))
            {
                return {mlir::success(), newClassPtr->classType.getName().getValue()};
            }
        }

        setProcessingState(newClassPtr, ProcessingStages::Processing, genContext);

        auto location = loc(classDeclarationAST);

        if (mlir::succeeded(mlirGenClassType(newClassPtr, genContext)))
        {
            newClassPtr->typeParamsWithArgs = genContext.typeParamsWithArgs;
        }

        // if this is generic specialized class then do not generate code for it
        if (mth.isGenericType(newClassPtr->classType))
        {
            return {mlir::success(), newClassPtr->classType.getName().getValue()};
        }

        // init this type (needed to use in property evaluations)
        GenContext classGenContext(genContext);
        classGenContext.thisType = newClassPtr->classType;
        classGenContext.thisClassType = newClassPtr->classType;
        classGenContext.specialization = isGenericClass;

        // we need THIS in params
        SymbolTableScopeT varScope(symbolTable);
        resetScope();   

        setProcessingState(newClassPtr, ProcessingStages::ProcessingStorageClass, genContext);
        if (mlir::failed(mlirGenClassStorageType(location, classDeclarationAST, newClassPtr, classGenContext)))
        {
            setProcessingState(newClassPtr, ProcessingStages::ErrorInStorageClass, genContext);
            return {mlir::failure(), ""};
        }

        setProcessingState(newClassPtr, ProcessingStages::ProcessedStorageClass, genContext);

        // if it is ClassExpression we need to know if it is declaration
        mlirGenClassCheckIfDeclaration(location, classDeclarationAST, newClassPtr, classGenContext);

        // go to root
        mlir::OpBuilder::InsertPoint savePoint;
        llvm::SmallVector<bool> membersProcessStates;
        if (isGenericClass)
        {
            savePoint = builder.saveInsertionPoint();
            builder.setInsertionPointToStart(theModule.getBody());

            saveMembersProcessStates(classDeclarationAST, newClassPtr, membersProcessStates);

            // before processing generic class for example array<int> array<string> we need to drop all states of processed members
            clearMembersProcessStates(classDeclarationAST, newClassPtr);
        }

        setProcessingState(newClassPtr, ProcessingStages::ProcessingBody, genContext);

        // prepare VTable
        llvm::SmallVector<VirtualMethodOrInterfaceVTableInfo> virtualTable;
        newClassPtr->getVirtualTable(virtualTable);

        if (!newClassPtr->isStatic)
        {
            mlirGenClassDefaultConstructor(classDeclarationAST, newClassPtr, classGenContext);
        }

#ifdef ENABLE_RTTI
        if (!newClassPtr->isStatic)
        {
            // INFO: .instanceOf must be first element in VTable for Cast Any
            mlirGenClassInstanceOfMethod(classDeclarationAST, newClassPtr, classGenContext);
        }
#endif

#if ENABLE_TYPED_GC
        auto enabledGC = !compileOptions.disableGC;
        if (enabledGC && !newClassPtr->isStatic)
        {
            mlirGenClassTypeBitmap(location, newClassPtr, classGenContext);
            mlirGenClassTypeDescriptorField(location, newClassPtr, classGenContext);
        }
#endif

        if (!newClassPtr->isStatic)
        {
            mlirGenClassNew(classDeclarationAST, newClassPtr, classGenContext);
        }

        mlirGenClassDefaultStaticConstructor(classDeclarationAST, newClassPtr, classGenContext);

        /*
        // to support call 'static v = new Class();'
        if (mlir::failed(mlirGenClassStaticFields(location, classDeclarationAST, newClassPtr, classGenContext)))
        {
            return {mlir::failure(), ""};
        }
        */

        if (mlir::failed(mlirGenClassMembers(location, classDeclarationAST, newClassPtr, classGenContext)))
        {
            setProcessingState(newClassPtr, ProcessingStages::ErrorInMembers, genContext);
            return {mlir::failure(), ""};
        }

        // generate vtable for interfaces in base class
        if (mlir::failed(mlirGenClassBaseInterfaces(location, newClassPtr, classGenContext)))
        {
            setProcessingState(newClassPtr, ProcessingStages::ErrorInBaseInterfaces, genContext);
            return {mlir::failure(), ""};
        }

        // generate vtable for interfaces
        for (auto &heritageClause : classDeclarationAST->heritageClauses)
        {
            if (mlir::failed(mlirGenClassHeritageClauseImplements(classDeclarationAST, newClassPtr, heritageClause,
                                                                  classGenContext)))
            {
                setProcessingState(newClassPtr, ProcessingStages::ErrorInHeritageClauseImplements, genContext);
                return {mlir::failure(), ""};
            }
        }

        if (!newClassPtr->isStatic)
        {
            if (mlir::failed(mlirGenClassVirtualTableDefinition(location, newClassPtr, classGenContext)))
            {
                setProcessingState(newClassPtr, ProcessingStages::ErrorInVTable, genContext);
                return {mlir::failure(), ""};
            }
        }

        // here we need to process New method;

        if (isGenericClass)
        {
            builder.restoreInsertionPoint(savePoint);

            restoreMembersProcessStates(classDeclarationAST, newClassPtr, membersProcessStates);
            //LLVM_DEBUG(llvm::dbgs() << "\n>>>>>>>>>>>>>>>>> module: \n" << theModule << "\n";);
        }

        setProcessingState(newClassPtr, ProcessingStages::ProcessedBody, genContext);

        // support dynamic loading
        if (getExportModifier(classDeclarationAST))
        {
            addClassDeclarationToExport(newClassPtr);
        }

        setProcessingState(newClassPtr, ProcessingStages::Processed, genContext);

        return {mlir::success(), newClassPtr->classType.getName().getValue()};
    }

    ClassInfo::TypePtr MLIRGenImpl::mlirGenClassInfo(ClassLikeDeclaration classDeclarationAST, const GenContext &genContext)
    {
        return mlirGenClassInfo(className(classDeclarationAST, genContext), classDeclarationAST, genContext);
    }

    ClassInfo::TypePtr MLIRGenImpl::mlirGenClassInfo(const std::string &name, ClassLikeDeclaration classDeclarationAST,
                                        const GenContext &genContext)
    {
        auto namePtr = StringRef(name).copy(stringAllocator);
        auto fullNamePtr = getFullNamespaceName(namePtr);

        ClassInfo::TypePtr newClassPtr;
        if (fullNameClassesMap.count(fullNamePtr))
        {
            newClassPtr = fullNameClassesMap.lookup(fullNamePtr);
            getClassesMap().insert({namePtr, newClassPtr});
        }
        else
        {
            // register class
            newClassPtr = std::make_shared<ClassInfo>();
            newClassPtr->name = namePtr;
            newClassPtr->fullName = fullNamePtr;
            newClassPtr->elementNamespace = currentNamespace;
            newClassPtr->isAbstract = hasModifier(classDeclarationAST, SyntaxKind::AbstractKeyword);
            newClassPtr->isDeclaration =
                declarationMode || hasModifier(classDeclarationAST, SyntaxKind::DeclareKeyword);
            newClassPtr->isStatic = hasModifier(classDeclarationAST, SyntaxKind::StaticKeyword);
            newClassPtr->isExport = getExportModifier(classDeclarationAST);
            newClassPtr->isPublic = hasModifier(classDeclarationAST, SyntaxKind::ExportKeyword);
            newClassPtr->hasVirtualTable = newClassPtr->isAbstract;

            // check decorator for class
            iterateDecorators(classDeclarationAST, genContext, [&](StringRef name, SmallVector<StringRef> args) {
                if (name == DLL_EXPORT)
                {
                    newClassPtr->isExport = true;
                }

                if (name == DLL_IMPORT)
                {
                    newClassPtr->isDeclaration = true;
                    newClassPtr->isImport = true;
                    // it has parameter, means this is dynamic import, should point to dll path
                    if (args.size() > 0)
                    {
                        newClassPtr->isDynamicImport = true;
                    }
                }
            });

            getClassesMap().insert({namePtr, newClassPtr});
            fullNameClassesMap.insert(fullNamePtr, newClassPtr);
        }

        return newClassPtr;
    }

    mlir::LogicalResult MLIRGenImpl::mlirGenClassCheckIfDeclaration(mlir::Location location,
                                                       ClassLikeDeclaration classDeclarationAST,
                                                       ClassInfo::TypePtr newClassPtr, const GenContext &genContext)
    {
        if (declarationMode)
        {
            newClassPtr->isDeclaration = true;
            return mlir::success();
        }

        if (classDeclarationAST != SyntaxKind::ClassExpression)
        {
            return mlir::success();
        }

        for (auto &classMember : classDeclarationAST->members)
        {
            // TODO:
            if (classMember == SyntaxKind::PropertyDeclaration)
            {
                // property declaration
                auto propertyDeclaration = classMember.as<PropertyDeclaration>();
                if (propertyDeclaration->initializer)
                {
                    // no definition
                    return mlir::success();
                }
            }

            if (classMember == SyntaxKind::MethodDeclaration || classMember == SyntaxKind::Constructor ||
                classMember == SyntaxKind::GetAccessor || classMember == SyntaxKind::SetAccessor)
            {
                auto funcLikeDeclaration = classMember.as<FunctionLikeDeclarationBase>();
                if (funcLikeDeclaration->body)
                {
                    // no definition
                    return mlir::success();
                }
            }
        }

        newClassPtr->isDeclaration = true;

        return mlir::success();
    }

    mlir::LogicalResult MLIRGenImpl::mlirGenClassTypeSetFields(ClassInfo::TypePtr newClassPtr,
                                                  SmallVector<mlir_ts::FieldInfo> &fieldInfos)
    {
        if (newClassPtr)
        {
            mlir::cast<mlir_ts::ClassStorageType>(newClassPtr->classType.getStorageType()).setFields(fieldInfos);
            return mlir::success();
        }

        return mlir::failure();
    }

    mlir::LogicalResult MLIRGenImpl::mlirGenClassStorageType(mlir::Location location, ClassLikeDeclaration classDeclarationAST,
                                                ClassInfo::TypePtr newClassPtr, const GenContext &genContext)
    {
        MLIRCodeLogic mcl(builder, compileOptions);
        SmallVector<mlir_ts::FieldInfo> fieldInfos;

        // add base classes
        for (auto &heritageClause : classDeclarationAST->heritageClauses)
        {
            if (mlir::failed(mlirGenClassHeritageClause(classDeclarationAST, newClassPtr, heritageClause, fieldInfos,
                                                        genContext)))
            {
                return mlir::failure();
            }
        }

#if ENABLE_RTTI
        if (newClassPtr->isDynamicImport)
        {
            mlirGenCustomRTTIDynamicImport(location, classDeclarationAST, newClassPtr, genContext);
        }
        else if (!newClassPtr->isStatic)
        {
            newClassPtr->hasVirtualTable = true;
            mlirGenCustomRTTI(location, classDeclarationAST, newClassPtr, genContext);
        }
#endif

        if (!newClassPtr->isStatic)
        {
            mlirGenClassSizeStaticField(location, classDeclarationAST, newClassPtr, genContext);
        }

        // non-static first
        for (auto &classMember : classDeclarationAST->members)
        {
            if (mlir::failed(mlirGenClassFieldMember(newClassPtr, classMember, fieldInfos, false, genContext)))
            {
                return mlir::failure();
            }
        }

        if (newClassPtr->getHasVirtualTableVariable())
        {
            auto fieldId = MLIRHelper::TupleFieldName(VTABLE_NAME, builder.getContext());
            if (fieldInfos.size() == 0 || fieldInfos.front().id != fieldId)
            {
                fieldInfos.insert(fieldInfos.begin(), {fieldId, getOpaqueType(), false, mlir_ts::AccessLevel::Public});
            }
        }

        mlirGenClassTypeSetFields(newClassPtr, fieldInfos);

        return mlir::success();
    }

    mlir::LogicalResult MLIRGenImpl::mlirGenClassStaticFields(mlir::Location location, ClassLikeDeclaration classDeclarationAST,
                                                 ClassInfo::TypePtr newClassPtr, const GenContext &genContext)
    {
        // dummy class, not used, needed to sync code
        // TODO: refactor it
        SmallVector<mlir_ts::FieldInfo> fieldInfos;

        // static second
        // TODO: if I use static method in static field initialization, test if I need process static fields after
        // static methods
        for (auto &classMember : classDeclarationAST->members)
        {
            if (mlir::failed(mlirGenClassFieldMember(newClassPtr, classMember, fieldInfos, true, genContext)))
            {
                return mlir::failure();
            }
        }

        return mlir::success();
    }

    mlir::LogicalResult MLIRGenImpl::mlirGenClassMembers(mlir::Location location, ClassLikeDeclaration classDeclarationAST,
                                            ClassInfo::TypePtr newClassPtr, const GenContext &genContext)
    {
        // clear all flags
        // extra fields - first, we need .instanceOf first for typr Any

        // dummy class, not used, needed to sync code
        // TODO: refactor it
        SmallVector<mlir_ts::FieldInfo> fieldInfos;

        // process indexes first
        for (auto &classMember : classDeclarationAST->members)
        {
            if (classMember == SyntaxKind::IndexSignature)
            {
                if (mlir::failed(mlirGenClassIndexMember(newClassPtr, classMember, genContext)))
                {
                    return mlir::failure();
                }
            }
        }

        // add methods when we have classType
        auto notResolved = 0;
        do
        {
            LLVM_DEBUG(llvm::dbgs() << "\n****** \tclass members: " << newClassPtr->fullName << " not resolved: " << notResolved;);

            auto lastTimeNotResolved = notResolved;
            notResolved = 0;

            auto orderWeight = 0;
            for (auto &classMember : newClassPtr->extraMembers)
            {
                orderWeight++;
                if (mlir::failed(mlirGenClassMethodMember(classDeclarationAST, newClassPtr, classMember, orderWeight, genContext)))
                {
                    notResolved++;
                }
            }

            for (auto &classMember : classDeclarationAST->members)
            {
                orderWeight++;

                // DEBUG ON
                LLVM_DEBUG(ClassMethodMemberInfo classMethodMemberInfo(newClassPtr, classMember);\
                    auto funcLikeDeclaration = classMember.as<FunctionLikeDeclarationBase>();\
                    getMethodNameOrPropertyName(\
                        newClassPtr->isStatic,\
                        funcLikeDeclaration,\
                        classMethodMemberInfo.methodName,\
                        classMethodMemberInfo.propertyName,\
                        genContext);\
                llvm::dbgs() << "\n****** \tprocessing: " << newClassPtr->fullName << "." << classMethodMemberInfo.methodName;);

                // static fields
                if (mlir::failed(mlirGenClassFieldMember(newClassPtr, classMember, fieldInfos, true, genContext)))
                {
                    LLVM_DEBUG(llvm::dbgs() << "\n\tNOT RESOLVED FIELD.");
                    notResolved++;
                }

                if (mlir::failed(mlirGenClassMethodMember(classDeclarationAST, newClassPtr, classMember, orderWeight, genContext)))
                {
                    LLVM_DEBUG(ClassMethodMemberInfo classMethodMemberInfo(newClassPtr, classMember);\
                        auto funcLikeDeclaration = classMember.as<FunctionLikeDeclarationBase>();\
                        getMethodNameOrPropertyName(\
                            newClassPtr->isStatic,\
                            funcLikeDeclaration,\
                            classMethodMemberInfo.methodName,\
                            classMethodMemberInfo.propertyName,\
                            genContext);\
                        llvm::dbgs() << "\n\tNOT RESOLVED MEMBER: " << classMethodMemberInfo.methodName;);
                    notResolved++;
                }

                if (mlir::failed(mlirGenClassStaticBlockMember(classDeclarationAST, newClassPtr, classMember, genContext)))
                {
                    return mlir::failure();
                }
            }

            for (auto &classMember : newClassPtr->extraMembersPost)
            {
                orderWeight++;

                if (mlir::failed(mlirGenClassMethodMember(classDeclarationAST, newClassPtr, classMember, orderWeight, genContext)))
                {
                    notResolved++;
                }
            }            

            // repeat if not all resolved
            if (lastTimeNotResolved > 0 && lastTimeNotResolved == notResolved)
            {
                // class can depend on other class declarations
                // theModule.emitError("can't resolve dependencies in class: ") << newClassPtr->name;
                return mlir::failure();
            }

        } while (notResolved > 0);

        clearMembersProcessStates(classDeclarationAST, newClassPtr);

        return mlir::success();
    }

    mlir::LogicalResult MLIRGenImpl::mlirGenClassHeritageClause(ClassLikeDeclaration classDeclarationAST,
                                                   ClassInfo::TypePtr newClassPtr, HeritageClause heritageClause,
                                                   SmallVector<mlir_ts::FieldInfo> &fieldInfos,
                                                   const GenContext &genContext)
    {
        MLIRCodeLogic mcl(builder, compileOptions);

        if (heritageClause->token == SyntaxKind::ExtendsKeyword)
        {
            auto &baseClassInfos = newClassPtr->baseClasses;

            for (auto &extendingType : heritageClause->types)
            {
                auto result = mlirGen(extendingType, genContext);
                EXIT_IF_FAILED_OR_NO_VALUE(result)
                auto baseType = V(result);
                mlir::TypeSwitch<mlir::Type>(baseType.getType())
                    .template Case<mlir_ts::ClassType>([&](auto baseClassType) {
                        auto baseName = baseClassType.getName().getValue();
                        auto fieldId = MLIRHelper::TupleFieldName(baseName, builder.getContext());
                        fieldInfos.push_back({fieldId, baseClassType.getStorageType(), false, mlir_ts::AccessLevel::Public});

                        auto classInfo = getClassInfoByFullName(baseName);
                        if (std::find(baseClassInfos.begin(), baseClassInfos.end(), classInfo) == baseClassInfos.end())
                        {
                            baseClassInfos.push_back(classInfo);
                        }
                    })
                    .Default([&](auto type) { llvm_unreachable("not implemented"); });
            }
            return mlir::success();
        }

        if (heritageClause->token == SyntaxKind::ImplementsKeyword)
        {
            newClassPtr->hasVirtualTable = true;

            auto &interfaceInfos = newClassPtr->implements;

            for (auto &implementingType : heritageClause->types)
            {
                auto result = mlirGen(implementingType, genContext);
                EXIT_IF_FAILED_OR_NO_VALUE(result)
                auto ifaceType = V(result);
                mlir::TypeSwitch<mlir::Type>(ifaceType.getType())
                    .template Case<mlir_ts::InterfaceType>([&](mlir_ts::InterfaceType interfaceType) {

                        auto ifaceName = interfaceType.getName().getValue();
                        auto found = llvm::find_if(interfaceInfos, [&](ImplementInfo &ifaceInfo) {
                            return ifaceInfo.interface->fullName == ifaceName;
                        });

                        auto interfaceInfo = getInterfaceInfoByFullName(interfaceType.getName().getValue());
                        assert(interfaceInfo);
                        if (found != interfaceInfos.end()) {
                            found->interface = interfaceInfo;
                        } else {
                            interfaceInfos.push_back({interfaceInfo, -1, false});
                        }
                    })
                    .Default([&](auto type) { llvm_unreachable("not implemented"); });
            }
        }

        return mlir::success();
    }

    mlir::LogicalResult MLIRGenImpl::mlirGenClassDataFieldAccessor(mlir::Location location, ClassInfo::TypePtr newClassPtr, 
            PropertyDeclaration propertyDeclaration, MemberName name, mlir::Type typeIfNotProvided, const GenContext &genContext)
    {
        NodeFactory nf(NodeFactoryFlags::None);

        NodeArray<ModifierLike> modifiers;
        for (auto modifier : propertyDeclaration->modifiers)
        {
            if (modifier == SyntaxKind::AccessorKeyword)
            {
                continue;
            }

            modifiers.push_back(modifier);
        }

        // add accessor methods
        if ((propertyDeclaration->internalFlags & InternalFlags::GenerationProcessed) != InternalFlags::GenerationProcessed)
        {            
            // set as generated
            propertyDeclaration->internalFlags |= InternalFlags::GenerationProcessed;

            {
                NodeArray<Statement> statements;

                auto thisToken = nf.createToken(SyntaxKind::ThisKeyword);

                auto propAccess = nf.createPropertyAccessExpression(thisToken, name);

                auto returnStat = nf.createReturnStatement(propAccess);
                statements.push_back(returnStat);

                auto body = nf.createBlock(statements, /*multiLine*/ false);

                auto getMethod = nf.createGetAccessorDeclaration(modifiers, propertyDeclaration->name, {}, undefined, body);

                newClassPtr->extraMembersPost->push_back(getMethod);
            }

            {
                NodeArray<Statement> statements;

                auto thisToken = nf.createToken(SyntaxKind::ThisKeyword);

                auto propAccess = nf.createPropertyAccessExpression(thisToken, name);

                auto setValue =
                    nf.createExpressionStatement(
                        nf.createBinaryExpression(propAccess, nf.createToken(SyntaxKind::EqualsToken), nf.createIdentifier(S("value"))));
                statements.push_back(setValue);

                auto body = nf.createBlock(statements, /*multiLine*/ false);

                auto type = propertyDeclaration->type;
                if (!type && typeIfNotProvided)
                {
                    std::string fieldTypeAlias;
                    fieldTypeAlias += ".";
                    fieldTypeAlias += newClassPtr->fullName.str();
                    fieldTypeAlias += ".";
                    fieldTypeAlias += MLIRHelper::getName(name);
                    type = nf.createTypeReferenceNode(nf.createIdentifier(stows(fieldTypeAlias)), undefined);    

                    getTypeAliasMap().insert({fieldTypeAlias, { typeIfNotProvided, undefined }});
                }

                if (!type)
                {
                    emitError(location) << "type for field accessor '" << MLIRHelper::getName(propertyDeclaration->name) << "' must be provided";
                    return mlir::failure();
                }

                auto setMethod = nf.createSetAccessorDeclaration(
                    modifiers, 
                    propertyDeclaration->name, 
                    { nf.createParameterDeclaration(undefined, undefined, nf.createIdentifier(S("value")), undefined, type) }, 
                    body);

                newClassPtr->extraMembersPost->push_back(setMethod);
            }
        }        

        return mlir::success();
    }    

    mlir::LogicalResult MLIRGenImpl::mlirGenClassDataFieldMember(mlir::Location location, ClassInfo::TypePtr newClassPtr, SmallVector<mlir_ts::FieldInfo> &fieldInfos, 
                                                    PropertyDeclaration propertyDeclaration, const GenContext &genContext)
    {
        auto accessLevel = getAccessLevel(propertyDeclaration);

        auto name = propertyDeclaration->name;
        auto isAccessor = hasModifier(propertyDeclaration, SyntaxKind::AccessorKeyword);
        if (isAccessor)
        {
            name = getFieldNameForAccessor(name);
        }
        
        auto fieldId = TupleFieldName(name, genContext);
        if (auto strAttr = dyn_cast<mlir::StringAttr>(fieldId)) 
        {
            if (strAttr.getValue().starts_with("#"))
            {
                accessLevel = mlir_ts::AccessLevel::Private;
            }
        }

        auto [type, init, typeProvided] = evaluateTypeAndInit(propertyDeclaration, genContext);
        if (init)
        {
            newClassPtr->hasInitializers = true;
            type = mth.wideStorageType(type);
        }

        LLVM_DEBUG(dbgs() << "\n!! class field: " << fieldId << " type: " << type << " access level: " << accessLevel);

        auto hasType = !!propertyDeclaration->type;
        if (mth.isNoneType(type))
        {
            if (hasType)
            {
                return mlir::failure();
            }

#ifndef ANY_AS_DEFAULT
            emitError(location)
                << "type for field '" << fieldId << "' is not provided, field must have type or initializer";
            return mlir::failure();
#else
            emitWarning(location) << "type for field '" << fieldId << "' is any";
            type = getAnyType();
#endif
        }

        fieldInfos.push_back({fieldId, type, false, accessLevel});

        // add accessor methods
        if (isAccessor)
        {            
            auto res = mlirGenClassDataFieldAccessor(location, newClassPtr, propertyDeclaration, name, type, genContext);
            EXIT_IF_FAILED(res)
        }        

        return mlir::success();
    }

    mlir::LogicalResult MLIRGenImpl::mlirGenClassStaticFieldMember(mlir::Location location, ClassInfo::TypePtr newClassPtr, PropertyDeclaration propertyDeclaration, const GenContext &genContext)
    {
        auto accessLevel = getAccessLevel(propertyDeclaration);
        auto isPublic = accessLevel == mlir_ts::AccessLevel::Public;
        auto name = propertyDeclaration->name;

        auto isAccessor = hasModifier(propertyDeclaration, SyntaxKind::AccessorKeyword);
        if (isAccessor)
        {
            isPublic = false;
            name = getFieldNameForAccessor(name);
        }

        auto fieldId = TupleFieldName(name, genContext);

        if (auto strAttr = dyn_cast<mlir::StringAttr>(fieldId)) 
        {
            if (strAttr.getValue().starts_with("#"))
            {
                isPublic = false;
                accessLevel = mlir_ts::AccessLevel::Private;
            }
        }

        // process static field - register global
        auto fullClassStaticFieldName =
            concat(newClassPtr->fullName, mlir::cast<mlir::StringAttr>(fieldId).getValue());
        VariableClass varClass = newClassPtr->isDeclaration ? VariableType::External : VariableType::Var;
        varClass.isExport = newClassPtr->isExport && isPublic;
        varClass.isImport = newClassPtr->isImport && isPublic;
        varClass.isPublic = isPublic;

        auto staticFieldType = registerVariable(
            location, fullClassStaticFieldName, true, varClass,
            [&](mlir::Location location, const GenContext &genContext) {
                auto isConst = false;
                mlir::Type typeInit;
                evaluate(
                    propertyDeclaration->initializer,
                    [&](mlir::Value val) {
                        typeInit = val.getType();
                        typeInit = mth.wideStorageType(typeInit);
                        isConst = isConstValue(val);
                    },
                    genContext);

                if (!newClassPtr->isDeclaration)
                {
                    if (isConst)
                    {
                        return getTypeAndInit(propertyDeclaration, genContext);
                    }

                    newClassPtr->hasStaticInitializers = true;
                }

                return getTypeOnly(propertyDeclaration, typeInit, genContext);
            },
            genContext);

        auto &staticFieldInfos = newClassPtr->staticFields;
        pushStaticField(staticFieldInfos, fieldId, staticFieldType, fullClassStaticFieldName, -1, accessLevel);

        // add accessor methods
        if (isAccessor)
        {            
            auto res = mlirGenClassDataFieldAccessor(location, newClassPtr, propertyDeclaration, name, staticFieldType, genContext);
            EXIT_IF_FAILED(res)
        }  

        return mlir::success();
    }

    mlir::LogicalResult MLIRGenImpl::mlirGenClassStaticFieldMemberDynamicImport(mlir::Location location, ClassInfo::TypePtr newClassPtr, PropertyDeclaration propertyDeclaration, const GenContext &genContext)
    {
        auto fieldId = TupleFieldName(propertyDeclaration->name, genContext);
        auto accessLevel = getAccessLevel(propertyDeclaration);

        // process static field - register global
        auto fullClassStaticFieldName =
            concat(newClassPtr->fullName, mlir::cast<mlir::StringAttr>(fieldId).getValue());
        
        auto staticFieldType = registerVariable(
            location, fullClassStaticFieldName, true, VariableType::Var,
            [&](mlir::Location location, const GenContext &genContext) -> TypeValueInitType {
                // detect field Type
                auto isConst = false;
                mlir::Type typeInit;
                if (propertyDeclaration->type)
                {
                    typeInit = getType(propertyDeclaration->type, genContext);
                }
                else if (propertyDeclaration->initializer)
                {
                    evaluate(
                        propertyDeclaration->initializer,
                        [&](mlir::Value val) {
                            typeInit = val.getType();
                            typeInit = mth.wideStorageType(typeInit);
                            isConst = isConstValue(val);
                        },
                        genContext);
                }
                else
                {
                    return {mlir::Type(), mlir::Value(), TypeProvided::No};
                }

                // add command to load reference from DLL
                auto fullName = V(mlirGenStringValue(location, fullClassStaticFieldName.str(), true));
                auto referenceToStaticFieldOpaque = builder.create<mlir_ts::SearchForAddressOfSymbolOp>(location, getOpaqueType(), fullName);
                auto result = cast(location, mlir_ts::RefType::get(typeInit), referenceToStaticFieldOpaque, genContext);
                auto referenceToStaticField = V(result);
                return {referenceToStaticField.getType(), referenceToStaticField, TypeProvided::No};
            },
            genContext);

        if (!staticFieldType)
        {
            return mlir::failure();
        }

        auto &staticFieldInfos = newClassPtr->staticFields;
        pushStaticField(staticFieldInfos, fieldId, staticFieldType, fullClassStaticFieldName, -1, accessLevel);

        return mlir::success();
    }    

    mlir::LogicalResult MLIRGenImpl::mlirGenClassConstructorPublicDataFieldMembers(mlir::Location location, SmallVector<mlir_ts::FieldInfo> &fieldInfos, 
                                                                      ConstructorDeclaration constructorDeclaration, const GenContext &genContext)
    {
        for (auto &parameter : constructorDeclaration->parameters)
        {
            auto isPublic = hasModifier(parameter, SyntaxKind::PublicKeyword);
            auto isProtected = hasModifier(parameter, SyntaxKind::ProtectedKeyword);
            auto isPrivate = hasModifier(parameter, SyntaxKind::PrivateKeyword);

            if (!(isPublic || isProtected || isPrivate))
            {
                continue;
            }

            auto fieldId = TupleFieldName(parameter->name, genContext);
            if (auto strAttr = dyn_cast<mlir::StringAttr>(fieldId)) {
                isPrivate |= strAttr.getValue().starts_with("#");
            }

            auto [type, init, typeProvided] = evaluateTypeAndInit(parameter, genContext);

            LLVM_DEBUG(dbgs() << "\n+++ class auto-gen field: " << fieldId << " type: " << type << "");
            if (mth.isNoneType(type))
            {
                return mlir::failure();
            }

            fieldInfos.push_back(
            {
                fieldId, 
                type, 
                false, 
                isPrivate 
                    ? mlir_ts::AccessLevel::Private 
                    : isProtected 
                        ? mlir_ts::AccessLevel::Protected 
                        : mlir_ts::AccessLevel::Public 
            });
        }

        return mlir::success();
    }

    mlir::LogicalResult MLIRGenImpl::mlirGenClassProcessClassPropertyByFieldMember(ClassInfo::TypePtr newClassPtr, ClassElement classMember)
    {
        auto isStatic = newClassPtr->isStatic || hasModifier(classMember, SyntaxKind::StaticKeyword);
        auto isConstructor = classMember == SyntaxKind::Constructor;
        if (isConstructor)
        {
            if (isStatic)
            {
                newClassPtr->hasStaticConstructor = true;
            }
            else
            {
                newClassPtr->hasConstructor = true;
                newClassPtr->constructorAccessLevel = getAccessLevel(classMember);
            }
        }

        if (newClassPtr->isStatic)
        {
            return mlir::success();
        }

        auto isMemberAbstract = hasModifier(classMember, SyntaxKind::AbstractKeyword);
        if (isMemberAbstract)
        {
            newClassPtr->hasVirtualTable = true;
        }

        auto isVirtual = (classMember->internalFlags & InternalFlags::ForceVirtual) == InternalFlags::ForceVirtual;
#ifdef ALL_METHODS_VIRTUAL
        isVirtual = !isConstructor;
#endif
        if (isVirtual)
        {
            newClassPtr->hasVirtualTable = true;
        }

        return mlir::success();
    }

    mlir::LogicalResult MLIRGenImpl::mlirGenClassFieldMember(ClassInfo::TypePtr newClassPtr, ClassElement classMember,
                                                SmallVector<mlir_ts::FieldInfo> &fieldInfos, bool staticOnly,
                                                const GenContext &genContext)
    {
        auto isStatic = newClassPtr->isStatic || hasModifier(classMember, SyntaxKind::StaticKeyword);
        if (staticOnly != isStatic)
        {
            return mlir::success();
        }

        auto location = loc(classMember);

        mlirGenClassProcessClassPropertyByFieldMember(newClassPtr, classMember);

        if (classMember == SyntaxKind::PropertyDeclaration)
        {
            // property declaration
            auto propertyDeclaration = classMember.as<PropertyDeclaration>();
            if (!isStatic)
            {
                if (mlir::failed(mlirGenClassDataFieldMember(location, newClassPtr, fieldInfos, propertyDeclaration, genContext)))
                {
                    return mlir::failure();
                }
            }
            else
            {
                if (newClassPtr->isDynamicImport)
                {
                    if (mlir::failed(mlirGenClassStaticFieldMemberDynamicImport(location, newClassPtr, propertyDeclaration, genContext)))
                    {
                        return mlir::failure();
                    }
                }
                else if (mlir::failed(mlirGenClassStaticFieldMember(location, newClassPtr, propertyDeclaration, genContext)))
                {
                    return mlir::failure();
                }
            }
        }

        if (classMember == SyntaxKind::Constructor && !isStatic)
        {
            auto constructorDeclaration = classMember.as<ConstructorDeclaration>();
            if (mlir::failed(mlirGenClassConstructorPublicDataFieldMembers(location, fieldInfos, constructorDeclaration, genContext)))
            {
                return mlir::failure();
            }
        }

        return mlir::success();
    }

    mlir::LogicalResult MLIRGenImpl::mlirGenClassNew(ClassLikeDeclaration classDeclarationAST, ClassInfo::TypePtr newClassPtr,
                                        const GenContext &genContext)
    {
        if (newClassPtr->isAbstract || newClassPtr->hasNew)
        {
            return mlir::success();
        }

        // create constructor
        newClassPtr->hasNew = true;

        // if we do not have constructor but have initializers we need to create empty dummy constructor
        NodeFactory nf(NodeFactoryFlags::None);

        ts::Block body;
        auto thisToken = nf.createToken(SyntaxKind::ThisKeyword);

        if (!newClassPtr->isDeclaration)
        {
            NodeArray<Statement> statements;

            auto newCall = nf.createNewExpression(thisToken, undefined, undefined);
            newCall->internalFlags |= InternalFlags::SuppressConstructorCall;

            auto returnStat = nf.createReturnStatement(newCall);
            statements.push_back(returnStat);

            body = nf.createBlock(statements, /*multiLine*/ false);
        }

        ModifiersArray modifiers;
        modifiers->push_back(nf.createToken(SyntaxKind::StaticKeyword));

        if (newClassPtr->isExport || newClassPtr->isImport)
        {
            modifiers.push_back(nf.createToken(SyntaxKind::PublicKeyword));
        }

        auto generatedNew = nf.createMethodDeclaration(modifiers, undefined, nf.createIdentifier(S(NEW_METHOD_NAME)),
                                                       undefined, undefined, undefined, nf.createThisTypeNode(), body);

        /*
        // advance declaration of "new"
        auto isStatic = false;
#ifdef ALL_METHODS_VIRTUAL
        auto isVirtual = true;
#else
        auto isVirtual = false;
#endif
        SmallVector<mlir::Type> inputs;
        SmallVector<mlir::Type> results{newClassPtr->classType};
        mlirGenForwardDeclaration(NEW_METHOD_NAME, getFunctionType(inputs, results), isStatic, isVirtual, newClassPtr,
genContext);

        newClassPtr->extraMembersPost.push_back(generatedNew);
        */

        LLVM_DEBUG(printDebug(generatedNew););

        newClassPtr->extraMembers.push_back(generatedNew);

        return mlir::success();
    }

    mlir::LogicalResult MLIRGenImpl::mlirGenClassDefaultConstructor(ClassLikeDeclaration classDeclarationAST,
                                                       ClassInfo::TypePtr newClassPtr, const GenContext &genContext)
    {
        // if we do not have constructor but have initializers we need to create empty dummy constructor
        if (newClassPtr->hasInitializers && !newClassPtr->hasConstructor)
        {
            // create constructor
            newClassPtr->hasConstructor = true;

            NodeFactory nf(NodeFactoryFlags::None);

            NodeArray<Statement> statements;

            if (!newClassPtr->baseClasses.empty())
            {
                auto superExpr = nf.createToken(SyntaxKind::SuperKeyword);
                auto callSuper = nf.createCallExpression(superExpr, undefined, undefined);
                statements.push_back(nf.createExpressionStatement(callSuper));
            }

            auto body = nf.createBlock(statements, /*multiLine*/ false);

            auto generatedConstructor = nf.createConstructorDeclaration(undefined, undefined, body);
            newClassPtr->extraMembers.push_back(generatedConstructor);
        }

        return mlir::success();
    }

    mlir::LogicalResult MLIRGenImpl::mlirGenClassDefaultStaticConstructor(ClassLikeDeclaration classDeclarationAST,
                                                             ClassInfo::TypePtr newClassPtr,
                                                             const GenContext &genContext)
    {
        // if we do not have constructor but have initializers we need to create empty dummy constructor
        if (newClassPtr->hasStaticInitializers && !newClassPtr->hasStaticConstructor)
        {
            // create constructor
            newClassPtr->hasStaticConstructor = true;

            NodeFactory nf(NodeFactoryFlags::None);

            NodeArray<Statement> statements;
            auto body = nf.createBlock(statements, /*multiLine*/ false);
            ModifiersArray modifiers;
            modifiers.push_back(nf.createToken(SyntaxKind::StaticKeyword));
            auto generatedConstructor = nf.createConstructorDeclaration(modifiers, undefined, body);
            newClassPtr->extraMembersPost.push_back(generatedConstructor);
        }

        return mlir::success();
    }

    mlir::LogicalResult MLIRGenImpl::mlirGenClassSizeStaticField(mlir::Location location, ClassLikeDeclaration classDeclarationAST,
                                          ClassInfo::TypePtr newClassPtr, const GenContext &genContext)
    {
        auto &staticFieldInfos = newClassPtr->staticFields;

        auto fieldId = MLIRHelper::TupleFieldName(SIZE_NAME, builder.getContext());

        // register global
        auto fullClassStaticFieldName = concat(newClassPtr->fullName, SIZE_NAME);

        auto staticFieldType = getIndexType();

        if (!fullNameGlobalsMap.count(fullClassStaticFieldName))
        {
            // saving state
            auto declarationModeStore = declarationMode;

            // prevent double generating
            //VariableClass varClass = newClassPtr->isDeclaration ? VariableType::External : VariableType::Var;
            VariableClass varClass = VariableType::Var;
            varClass.isExport = newClassPtr->isExport;
            varClass.isImport = newClassPtr->isImport;
            varClass.isPublic = true;
            if (!newClassPtr->isImport)
            {                           
                declarationMode = false;
#ifdef WIN32                
                varClass.comdat = Select::ExactMatch;
#else
                varClass.comdat = Select::Any;
#endif                
            }
            else if (newClassPtr->isDeclaration)
            {
                varClass.type = VariableType::External;
            }

            registerVariable(
                location, fullClassStaticFieldName, true, varClass,
                [&](mlir::Location location, const GenContext &genContext) {
                    // if (newClassPtr->isDeclaration)
                    // {
                    //     return std::make_tuple(staticFieldType, mlir::Value(), TypeProvided::Yes);
                    // }

                    // TODO: review usage of SizeOf in code, as size of class pointer is not size of data struct
                    auto sizeOfType =
                        builder.create<mlir_ts::SizeOfOp>(location, mth.getIndexType(), newClassPtr->classType.getStorageType());

                    mlir::Value init = sizeOfType;
                    return std::make_tuple(staticFieldType, init, TypeProvided::Yes);
                },
                genContext);

            // restore state
            declarationMode = declarationModeStore;
        }

        pushStaticField(staticFieldInfos, fieldId, staticFieldType, fullClassStaticFieldName, -1, mlir_ts::AccessLevel::Public);

        return mlir::success();    
    }

    mlir::LogicalResult MLIRGenImpl::mlirGenClassTypeDescriptorField(mlir::Location location, ClassInfo::TypePtr newClassPtr,
                                                        const GenContext &genContext)
    {
        // TODO: experiment if we need it at all even external declaration
        if (newClassPtr->isDeclaration)
        {
            return mlir::success();
        }

        // register global
        auto fullClassStaticFieldName = getTypeDescriptorFieldName(newClassPtr);

        if (!fullNameGlobalsMap.count(fullClassStaticFieldName))
        {
            registerVariable(
                location, fullClassStaticFieldName, true,
                newClassPtr->isDeclaration ? VariableType::External : VariableType::Var,
                [&](mlir::Location location, const GenContext &genContext) {
                    auto init =
                        builder.create<mlir_ts::ConstantOp>(location, builder.getI64Type(), mth.getI64AttrValue(0));
                    return std::make_tuple(init.getType(), init, TypeProvided::Yes);
                },
                genContext);
        }

        return mlir::success();
    }

    mlir::LogicalResult MLIRGenImpl::mlirGenClassTypeBitmap(mlir::Location location, ClassInfo::TypePtr newClassPtr,
                                               const GenContext &genContext)
    {
        // no need to generate
        if (newClassPtr->isDeclaration)
        {
            return mlir::success();
        }

        MLIRCodeLogic mcl(builder, compileOptions);

        // register global
        auto name = TYPE_BITMAP_NAME;
        auto fullClassStaticFieldName = getTypeBitmapMethodName(newClassPtr);

        auto funcType = getFunctionType({}, builder.getI64Type(), false);

        mlirGenFunctionBody(
            location, name, fullClassStaticFieldName, funcType,
            [&](mlir::Location location, const GenContext &genContext) {
                auto bitmapValueType = mth.getTypeBitmapValueType();

                auto nullOp = builder.create<mlir_ts::NullOp>(location, getNullType());
                CAST_A_NULLCHECK(classNull, location, newClassPtr->classType, nullOp, genContext, true);

                auto sizeOfStoreElement =
                    builder.create<mlir_ts::SizeOfOp>(location, mth.getIndexType(), mth.getTypeBitmapValueType());

                auto _8Value = builder.create<mlir_ts::ConstantOp>(location, mth.getIndexType(),
                                                                   builder.getIntegerAttr(mth.getIndexType(), 8));
                auto sizeOfStoreElementInBits = builder.create<mlir_ts::ArithmeticBinaryOp>(
                    location, mth.getIndexType(), builder.getI32IntegerAttr((int)SyntaxKind::AsteriskToken),
                    sizeOfStoreElement, _8Value);

                // calc bitmap size
                auto sizeOfType =
                    builder.create<mlir_ts::SizeOfOp>(location, mth.getIndexType(), newClassPtr->classType.getStorageType());

                // calc count of store elements of type size
                auto sizeOfTypeInBitmapTypes = builder.create<mlir_ts::ArithmeticBinaryOp>(
                    location, mth.getIndexType(), builder.getI32IntegerAttr((int)SyntaxKind::SlashToken), sizeOfType,
                    sizeOfStoreElement);

                // size alligned by size of bits
                auto sizeOfTypeAligned = builder.create<mlir_ts::ArithmeticBinaryOp>(
                    location, mth.getIndexType(), builder.getI32IntegerAttr((int)SyntaxKind::PlusToken),
                    sizeOfTypeInBitmapTypes, sizeOfStoreElementInBits);

                auto _1I64Value = builder.create<mlir_ts::ConstantOp>(location, mth.getIndexType(),
                                                                      builder.getIntegerAttr(mth.getIndexType(), 1));

                sizeOfTypeAligned = builder.create<mlir_ts::ArithmeticBinaryOp>(
                    location, mth.getIndexType(), builder.getI32IntegerAttr((int)SyntaxKind::MinusToken),
                    sizeOfTypeAligned, _1I64Value);

                sizeOfTypeAligned = builder.create<mlir_ts::ArithmeticBinaryOp>(
                    location, mth.getIndexType(), builder.getI32IntegerAttr((int)SyntaxKind::SlashToken),
                    sizeOfTypeAligned, sizeOfStoreElementInBits);

                // allocate in stack
                auto arrayValue = builder.create<mlir_ts::AllocaOp>(location, mlir_ts::RefType::get(bitmapValueType),
                                                                    sizeOfTypeAligned);

                // property ref
                auto count = newClassPtr->fieldsCount();
                for (auto index = 0; (unsigned)index < count; index++)
                {
                    auto fieldInfo = newClassPtr->fieldInfoByIndex(index);
                    // skip virrual table for speed adv.
                    if (index == 0 && isa<mlir_ts::OpaqueType>(fieldInfo.type))
                    {
                        continue;
                    }

                    if (mth.isValueType(fieldInfo.type))
                    {
                        continue;
                    }

                    auto fieldValue = mlirGenPropertyAccessExpression(location, classNull, fieldInfo.id, genContext);
                    assert(fieldValue);
                    auto fieldRef = mcl.GetReferenceFromValue(location, fieldValue);

                    // cast to int64
                    CAST_A(fieldAddrAsInt, location, mth.getIndexType(), fieldRef, genContext);

                    // calc index
                    auto calcIndex = builder.create<mlir_ts::ArithmeticBinaryOp>(
                        location, mth.getIndexType(), builder.getI32IntegerAttr((int)SyntaxKind::SlashToken),
                        fieldAddrAsInt, sizeOfStoreElement);

                    auto elemRef = builder.create<mlir_ts::PointerOffsetRefOp>(
                        location, mlir_ts::RefType::get(bitmapValueType), arrayValue, calcIndex);

                    // calc bit
                    auto indexModIndex = builder.create<mlir_ts::ArithmeticBinaryOp>(
                        location, mth.getIndexType(), builder.getI32IntegerAttr((int)SyntaxKind::PercentToken),
                        calcIndex, sizeOfStoreElementInBits);

                    auto indexMod = builder.create<mlir_ts::CastOp>(location, bitmapValueType, indexModIndex);

                    auto _1Value = builder.create<mlir_ts::ConstantOp>(location, bitmapValueType,
                                                                       builder.getIntegerAttr(bitmapValueType, 1));

                    // 1 << index_mod
                    auto bitValue = builder.create<mlir_ts::ArithmeticBinaryOp>(
                        location, bitmapValueType,
                        builder.getI32IntegerAttr((int)SyntaxKind::GreaterThanGreaterThanToken), _1Value, indexMod);

                    // load val
                    auto val = builder.create<mlir_ts::LoadOp>(location, bitmapValueType, elemRef);

                    // apply or
                    auto valWithBit = builder.create<mlir_ts::ArithmeticBinaryOp>(
                        location, bitmapValueType, builder.getI32IntegerAttr((int)SyntaxKind::BarToken), val, bitValue);

                    // save value
                    auto saveToElement = builder.create<mlir_ts::StoreOp>(location, valWithBit, elemRef);
                }

                auto typeDescr = builder.create<mlir_ts::GCMakeDescriptorOp>(location, builder.getI64Type(), arrayValue,
                                                                             sizeOfTypeInBitmapTypes);

                auto retVarInfo = symbolTable.lookup(RETURN_VARIABLE_NAME);
                builder.create<mlir_ts::ReturnValOp>(location, typeDescr, retVarInfo.first);
                return ValueOrLogicalResult(mlir::success());
            },
            genContext);

        return mlir::success();
    }

    mlir::LogicalResult MLIRGenImpl::mlirGenClassInstanceOfMethod(ClassLikeDeclaration classDeclarationAST,
                                                     ClassInfo::TypePtr newClassPtr, const GenContext &genContext)
    {
        // if we do not have constructor but have initializers we need to create empty dummy constructor
        // if (newClassPtr->getHasVirtualTable())
        {
            if (newClassPtr->hasRTTI)
            {
                return mlir::success();
            }

            newClassPtr->hasRTTI = true;

            NodeFactory nf(NodeFactoryFlags::None);

            ts::Block body = undefined;
            if (!newClassPtr->isDeclaration)
            {
                NodeArray<Statement> statements;

                /*
                if (!newClassPtr->baseClasses.empty())
                {
                    auto superExpr = nf.createToken(SyntaxKind::SuperKeyword);
                    auto callSuper = nf.createCallExpression(superExpr, undefined, undefined);
                    statements.push_back(nf.createExpressionStatement(callSuper));
                }
                */

                // access .rtti via this (as virtual method)
                // auto cmpRttiToParam = nf.createBinaryExpression(
                //     nf.createIdentifier(LINSTANCEOF_PARAM_NAME), nf.createToken(SyntaxKind::EqualsEqualsToken),
                //     nf.createPropertyAccessExpression(nf.createToken(SyntaxKind::ThisKeyword),
                //                                       nf.createIdentifier(S(RTTI_NAME))));

                // access .rtti via static field
                auto fullClassStaticFieldName = concat(newClassPtr->fullName, RTTI_NAME);

                auto cmpRttiToParam = nf.createBinaryExpression(
                     nf.createIdentifier(S(INSTANCEOF_PARAM_NAME)), nf.createToken(SyntaxKind::EqualsEqualsToken),
                     nf.createIdentifier(convertUTF8toWide(std::string(fullClassStaticFieldName))));

                auto cmpLogic = cmpRttiToParam;

                if (!newClassPtr->baseClasses.empty())
                {
                    NodeArray<Expression> argumentsArray;
                    argumentsArray.push_back(nf.createIdentifier(S(INSTANCEOF_PARAM_NAME)));
                    cmpLogic =
                        nf.createBinaryExpression(cmpRttiToParam, nf.createToken(SyntaxKind::BarBarToken),
                                                  nf.createCallExpression(nf.createPropertyAccessExpression(
                                                                              nf.createToken(SyntaxKind::SuperKeyword),
                                                                              nf.createIdentifier(S(INSTANCEOF_NAME))),
                                                                          undefined, argumentsArray));
                }

                auto returnStat = nf.createReturnStatement(cmpLogic);
                statements.push_back(returnStat);

                body = nf.createBlock(statements, false);
            }

            NodeArray<ParameterDeclaration> parameters;
            parameters.push_back(nf.createParameterDeclaration(undefined, undefined,
                                                               nf.createIdentifier(S(INSTANCEOF_PARAM_NAME)), undefined,
                                                               nf.createToken(SyntaxKind::StringKeyword), undefined));

            ModifiersArray modifiers;
            if (newClassPtr->isExport || newClassPtr->isImport)
            {
                modifiers.push_back(nf.createToken(SyntaxKind::PublicKeyword));
            }

            auto instanceOfMethod = nf.createMethodDeclaration(
                modifiers, undefined, nf.createIdentifier(S(INSTANCEOF_NAME)), undefined, undefined,
                parameters, nf.createToken(SyntaxKind::BooleanKeyword), body);

            instanceOfMethod->internalFlags |= InternalFlags::ForceVirtual;
            // TODO: you adding new member to the same DOM(parse) instance but it is used for 2 instances of generic
            // type ERROR: do not change members!!!!

            // INFO: .instanceOf must be first element in VTable for Cast Any
            for (auto member : newClassPtr->extraMembers)
            {
                assert(member == SyntaxKind::Constructor);
            }

            newClassPtr->extraMembers.push_back(instanceOfMethod);
        }

        return mlir::success();
    }

    ValueOrLogicalResult MLIRGenImpl::mlirGenCreateInterfaceVTableForClass(mlir::Location location, ClassInfo::TypePtr newClassPtr,
                                                              InterfaceInfo::TypePtr newInterfacePtr,
                                                              const GenContext &genContext)
    {
        auto fullClassInterfaceVTableFieldName = interfaceVTableNameForClass(newClassPtr, newInterfacePtr);
        auto existValue = resolveFullNameIdentifier(location, fullClassInterfaceVTableFieldName, true, genContext);
        if (existValue)
        {
            return existValue;
        }

        if (mlir::succeeded(
                mlirGenClassVirtualTableDefinitionForInterface(location, newClassPtr, newInterfacePtr, genContext)))
        {
            return resolveFullNameIdentifier(location, fullClassInterfaceVTableFieldName, true, genContext);
        }

        return mlir::failure();
    }

    mlir::LogicalResult MLIRGenImpl::mlirGenClassVirtualTableDefinitionForInterface(mlir::Location location,
                                                                       ClassInfo::TypePtr newClassPtr,
                                                                       InterfaceInfo::TypePtr newInterfacePtr,
                                                                       const GenContext &genContext)
    {

        MLIRCodeLogic mcl(builder, compileOptions);

        MethodInfo emptyMethod;
        mlir_ts::FieldInfo emptyFieldInfo;
        // TODO: ...
        auto classStorageType = mlir::cast<mlir_ts::ClassStorageType>(newClassPtr->classType.getStorageType());

        llvm::SmallVector<VirtualMethodOrFieldInfo> virtualTable;
        auto result = newInterfacePtr->getVirtualTable(
            virtualTable,
            [&](mlir::Attribute id, mlir::Type fieldType, bool isConditional) -> std::pair<mlir_ts::FieldInfo, mlir::LogicalResult> {
                auto found = false;
                auto foundField = newClassPtr->findField(id, found);
                if (!found || fieldType != foundField.type)
                {
                    if (!found && !isConditional || found)
                    {
                        emitError(location)
                            << "field type not matching for " << id << " for interface '" << newInterfacePtr->fullName
                            << "' in class '" << newClassPtr->fullName << "'";

                        return {emptyFieldInfo, mlir::failure()};
                    }

                    return {emptyFieldInfo, mlir::success()};
                }

                return {foundField, mlir::success()};
            },
            [&](std::string name, mlir_ts::FunctionType funcType, bool isConditional, int interfacePosIndex) -> std::pair<MethodInfo &, mlir::LogicalResult> {
                auto foundMethodPtr = newClassPtr->findMethod(name);
                if (!foundMethodPtr)
                {
                    // TODO: generate method wrapper for calling new/ctor method
                    if (name == NEW_CTOR_METHOD_NAME)
                    {
                        // TODO: generate method                        
                        foundMethodPtr = generateSynthMethodToCallNewCtor(
                            location, newClassPtr, newInterfacePtr, funcType, interfacePosIndex, genContext);
                    }

                    if (!foundMethodPtr)
                    {
                        if (!isConditional)
                        {
                            emitError(location)
                                << "can't find method '" << name << "' for interface '" << newInterfacePtr->fullName
                                << "' in class '" << newClassPtr->fullName << "'";

                            return {emptyMethod, mlir::failure()};
                        }

                        return {emptyMethod, mlir::success()};
                    }
                }

                auto foundMethodFunctionType = foundMethodPtr->funcType;

                auto result = mth.TestFunctionTypesMatch(funcType, foundMethodFunctionType, 1);
                if (result.result != MatchResultType::Match)
                {
                    emitError(location) << "method signature not matching '" << name << ":" << to_print(funcType)
                                        << "' for interface '" << newInterfacePtr->fullName << "' in class '"
                                        << newClassPtr->fullName << "'."
                                        << " Found method: " << name << ":" << to_print(foundMethodFunctionType);
                    return {emptyMethod, mlir::failure()};
                }

                return {*foundMethodPtr, mlir::success()};
            });

        if (mlir::failed(result))
        {
            return result;
        }

        // register global
        auto fullClassInterfaceVTableFieldName = interfaceVTableNameForClass(newClassPtr, newInterfacePtr);
        auto registeredType = registerVariable(
            location, fullClassInterfaceVTableFieldName, true, VariableType::Var,
            [&](mlir::Location location, const GenContext &genContext) {
                // build vtable from names of methods

                MLIRCodeLogic mcl(builder, compileOptions);

                auto virtTuple = getVirtualTableType(virtualTable);

                mlir::Value vtableValue = builder.create<mlir_ts::UndefOp>(location, virtTuple);
                auto fieldIndex = 0;
                for (auto methodOrField : virtualTable)
                {
                    if (methodOrField.isMissing)
                    {
                        // a genuinely-absent optional (`?`) interface field/method the class
                        // doesn't implement - mirrors the object-literal handling in
                        // mlirGenObjectVirtualTableDefinitionForInterface
                        // (MLIRGenInterfaces.cpp): querying the class for a field/method it
                        // doesn't have (via mlirGenPropertyAccessExpression, below) crashes
                        // internally instead of failing gracefully, so this case must be
                        // detected and handled up front rather than falling into the "present"
                        // branches - use the same -1 sentinel placeholder, cast to the slot's
                        // ref/func-pointer type.
                        auto negative1 = builder.create<mlir_ts::ConstantOp>(location, builder.getI64Type(),
                                                                             mth.getI64AttrValue(-1));
                        auto slotType = methodOrField.isField ? methodOrField.fieldInfo.type : methodOrField.methodInfo.funcType;
                        auto castedPtr = cast(location, mlir_ts::RefType::get(slotType), negative1, genContext);
                        vtableValue = builder.create<mlir_ts::InsertPropertyOp>(
                            location, virtTuple, castedPtr, vtableValue,
                            MLIRHelper::getStructIndex(builder, fieldIndex));
                    }
                    else if (methodOrField.isField)
                    {
                        auto nullObj = builder.create<mlir_ts::NullOp>(location, getNullType());
                        auto classNull = cast(location, newClassPtr->classType, nullObj, genContext, true);
                        auto fieldValue = mlirGenPropertyAccessExpression(location, classNull,
                                                                          methodOrField.fieldInfo.id, genContext);
                        if (!fieldValue)
                        {
                            emitError(location) << "can't find field (or it is inaccessible): " << methodOrField.fieldInfo.id
                                                << " in interface: " << newInterfacePtr->fullName
                                                << " for class: " << newClassPtr->fullName;
                            return TypeValueInitType{mlir::Type(), mlir::Value(), TypeProvided::No};
                        }

                        auto fieldRef = mcl.GetReferenceFromValue(location, fieldValue);
                        if (!fieldRef)
                        {
                            emitError(location) << "can't find reference for field: " << methodOrField.fieldInfo.id
                                                << " in interface: " << newInterfacePtr->fullName
                                                << " for class: " << newClassPtr->fullName;
                            return TypeValueInitType{mlir::Type(), mlir::Value(), TypeProvided::No};
                        }

                        // insert &(null)->field
                        vtableValue = builder.create<mlir_ts::InsertPropertyOp>(
                            location, virtTuple, fieldRef, vtableValue,
                            MLIRHelper::getStructIndex(builder, fieldIndex));
                    }
                    else
                    {
                        auto methodConstName = builder.create<mlir_ts::SymbolRefOp>(
                            location, methodOrField.methodInfo.funcType,
                            mlir::FlatSymbolRefAttr::get(builder.getContext(),
                                                         methodOrField.methodInfo.funcName));

                        vtableValue = builder.create<mlir_ts::InsertPropertyOp>(
                            location, virtTuple, methodConstName, vtableValue,
                            MLIRHelper::getStructIndex(builder, fieldIndex));
                    }

                    fieldIndex++;
                }

                return TypeValueInitType{virtTuple, vtableValue, TypeProvided::Yes};
            },
            genContext);

        return registeredType ? mlir::success() : mlir::failure();
    }

    mlir::LogicalResult MLIRGenImpl::mlirGenClassBaseInterfaces(mlir::Location location, ClassInfo::TypePtr newClassPtr,
                                                   const GenContext &genContext)
    {
        if (newClassPtr->isDeclaration)
        {
            return mlir::success();
        }

        for (auto &baseClass : newClassPtr->baseClasses)
        {
            for (auto &implement : baseClass->implements)
            {
                if (mlir::failed(mlirGenClassVirtualTableDefinitionForInterface(location, newClassPtr,
                                                                                implement.interface, genContext)))
                {
                    return mlir::failure();
                }
            }
        }

        return mlir::success();
    }

    mlir::LogicalResult MLIRGenImpl::mlirGenClassHeritageClauseImplements(ClassLikeDeclaration classDeclarationAST,
                                                             ClassInfo::TypePtr newClassPtr,
                                                             HeritageClause heritageClause,
                                                             const GenContext &genContext)
    {
        if (heritageClause->token != SyntaxKind::ImplementsKeyword)
        {
            return mlir::success();
        }

        for (auto &implementingType : heritageClause->types)
        {
            auto result = mlirGen(implementingType, genContext);
            EXIT_IF_FAILED_OR_NO_VALUE(result)
            auto ifaceType = V(result);
            auto success = false;
            mlir::TypeSwitch<mlir::Type>(ifaceType.getType())
                .template Case<mlir_ts::InterfaceType>([&](auto interfaceType) {
                    auto interfaceInfo = getInterfaceInfoByFullName(interfaceType.getName().getValue());
                    assert(interfaceInfo);
                    if (!newClassPtr->isDeclaration)
                    {
                        success = !failed(mlirGenClassVirtualTableDefinitionForInterface(
                            loc(implementingType), newClassPtr, interfaceInfo, genContext));
                    }
                    else
                    {
                        success = true;
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

    mlir::LogicalResult MLIRGenImpl::mlirGenClassVirtualTableDefinition(mlir::Location location, ClassInfo::TypePtr newClassPtr,
                                                           const GenContext &genContext)
    {
        if (!newClassPtr->getHasVirtualTable() || newClassPtr->isAbstract)
        {
            return mlir::success();
        }
       
        // TODO: ...
        llvm::SmallVector<VirtualMethodOrInterfaceVTableInfo> virtualTable;
        newClassPtr->getVirtualTable(virtualTable);

        // TODO: this is pure hack, add ability to clean up created globals while "dummyRun = true"
        // look into examnple with class declaraion in generic function
        auto fullClassVTableFieldName = concat(newClassPtr->fullName, VTABLE_NAME);
        if (fullNameGlobalsMap.count(fullClassVTableFieldName))
        {
            return mlir::success();
        }

        // register global
        VariableClass varClass = newClassPtr->isDeclaration ? VariableType::External : VariableType::Var;
        varClass.isExport = newClassPtr->isExport;
        varClass.isImport = newClassPtr->isImport;
        varClass.isPublic = newClassPtr->isPublic;            
        auto vtableRegisteredType = registerVariable(
            location, fullClassVTableFieldName, true,
            varClass,
            [&](mlir::Location location, const GenContext &genContext) {
                auto virtTuple = getVirtualTableType(virtualTable);
                if (newClassPtr->isDeclaration)
                {
                    return TypeValueInitType{virtTuple, mlir::Value(), TypeProvided::Yes};
                }

                // build vtable from names of methods
                MLIRCodeLogic mcl(builder, compileOptions);
                mlir::Value vtableValue = builder.create<mlir_ts::UndefOp>(location, virtTuple);
                auto fieldIndex = 0;
                for (auto vtRecord : virtualTable)
                {
                    if (vtRecord.isInterfaceVTable)
                    {
                        // TODO: write correct full name for vtable
                        auto fullClassInterfaceVTableFieldName =
                            concat(newClassPtr->fullName, vtRecord.methodInfo.name, VTABLE_NAME);
                        auto interfaceVTableValue =
                            resolveFullNameIdentifier(location, fullClassInterfaceVTableFieldName, true, genContext);

                        if (!interfaceVTableValue)
                        {
                            return TypeValueInitType{mlir::Type(), mlir::Value(), TypeProvided::No};
                        }

                        auto interfaceVTableValueAsAny =
                            cast(location, getOpaqueType(), interfaceVTableValue, genContext);

                        vtableValue = builder.create<mlir_ts::InsertPropertyOp>(
                            location, virtTuple, interfaceVTableValueAsAny, vtableValue,
                            MLIRHelper::getStructIndex(builder, fieldIndex++));
                    }
                    else
                    {
                        // The vtable is extends-recursive, so a derived class's vtable can
                        // contain slots whose symbols (inherited virtual methods, and - with
                        // ADD_STATIC_MEMBERS_TO_VTABLE - inherited static fields like the
                        // RTTI `.rtti`/`.size`) are owned by a base class that lives in a
                        // dynamically imported module. Those cannot be constant SymbolRefOp
                        // references: with no import library, the address is not a link-time
                        // constant. Resolve them at runtime instead (SearchForAddressOfSymbolOp
                        // + cast) - GlobalOpLowering already routes any initializer containing
                        // a SearchForAddressOfSymbolOp through the __cctor global-constructor
                        // path, and the module-load ctor is emitted before all per-symbol
                        // ctors, so the DLL is loaded by the time this resolves.
                        auto isOwnedByDynamicImport = [&](mlir::StringRef symbolName, bool isStaticField) {
                            std::function<ClassInfo::TypePtr(ClassInfo::TypePtr)> findOwner =
                                [&](ClassInfo::TypePtr cls) -> ClassInfo::TypePtr {
                                if (isStaticField
                                        ? llvm::any_of(cls->staticFields, [&](auto &f) { return f.globalVariableName == symbolName; })
                                        : llvm::any_of(cls->methods, [&](auto &m) { return m.funcName == symbolName; }))
                                {
                                    return cls;
                                }

                                for (auto &base : cls->baseClasses)
                                {
                                    if (auto owner = findOwner(base))
                                    {
                                        return owner;
                                    }
                                }

                                return ClassInfo::TypePtr();
                            };

                            auto owner = findOwner(newClassPtr);
                            return owner && owner->isDynamicImport;
                        };

                        mlir::Value methodOrFieldNameRef;
                        mlir::StringRef symbolName;
                        mlir::Type slotType;
                        if (!vtRecord.isStaticField)
                        {
                            if (vtRecord.methodInfo.isAbstract)
                            {
                                emitError(location) << "Abstract method '" << vtRecord.methodInfo.name <<  "' is not implemented in '" << newClassPtr->name << "'";
                                return TypeValueInitType{mlir::Type(), mlir::Value(), TypeProvided::No};
                            }

                            symbolName = vtRecord.methodInfo.funcName;
                            slotType = vtRecord.methodInfo.funcType;
                        }
                        else
                        {
                            symbolName = vtRecord.staticFieldInfo.globalVariableName;
                            slotType = mlir_ts::RefType::get(vtRecord.staticFieldInfo.type);
                        }

                        if (isOwnedByDynamicImport(symbolName, vtRecord.isStaticField))
                        {
                            auto symbolNameValue = V(mlirGenStringValue(location, symbolName.str(), true));
                            auto referenceToSymbolOpaque = builder.create<mlir_ts::SearchForAddressOfSymbolOp>(
                                location, getOpaqueType(), symbolNameValue);
                            auto castResult = cast(location, slotType, referenceToSymbolOpaque, genContext);
                            if (castResult.failed_or_no_value())
                            {
                                return TypeValueInitType{mlir::Type(), mlir::Value(), TypeProvided::No};
                            }

                            methodOrFieldNameRef = V(castResult);
                        }
                        else
                        {
                            methodOrFieldNameRef = builder.create<mlir_ts::SymbolRefOp>(
                                location, slotType,
                                mlir::FlatSymbolRefAttr::get(builder.getContext(), symbolName));
                        }

                        vtableValue = builder.create<mlir_ts::InsertPropertyOp>(
                            location, virtTuple, methodOrFieldNameRef, vtableValue,
                            MLIRHelper::getStructIndex(builder, fieldIndex++));
                    }
                }

                return TypeValueInitType{virtTuple, vtableValue, TypeProvided::Yes};
            },
            genContext);

        return (vtableRegisteredType) ? mlir::success() : mlir::failure();
    }

    mlir::LogicalResult MLIRGenImpl::mlirGenClassIndexMember(ClassInfo::TypePtr newClassPtr, ClassElement classMember,
                                                const GenContext &genContext)
    {
        if (classMember->processed)
        {
            LLVM_DEBUG(llvm::dbgs() << "\n\tALREADY PROCESSED.");
            return mlir::success();
        }

        // TODO:
        auto indexElement = classMember.as<IndexSignatureDeclaration>();

        auto &indexInfos = newClassPtr->indexes;

        auto res = mlirGenFunctionSignaturePrototype(indexElement.as<SignatureDeclaration>(), false, genContext);
        auto funcType = std::get<1>(res);

        LLVM_DEBUG(llvm::dbgs() << "\n\tindex signature: " << funcType << "\n");

        if (std::find_if(
            indexInfos.begin(), 
            indexInfos.end(), 
            [&] (auto& item) { 
                return item.indexSignature == funcType; 
            }) == indexInfos.end())
        {
            indexInfos.push_back({funcType, {}, {}});
        } 

        return mlir::success();
    }    

    mlir::LogicalResult MLIRGenImpl::mlirGenClassMethodMember(ClassLikeDeclaration classDeclarationAST,
                                                 ClassInfo::TypePtr newClassPtr, ClassElement classMember,
                                                 int orderWeight,
                                                 const GenContext &genContext)
    {
        if (classMember->processed)
        {
            LLVM_DEBUG(llvm::dbgs() << "\n\tALREADY PROCESSED.");
            return mlir::success();
        }

        ClassMethodMemberInfo classMethodMemberInfo(newClassPtr, classMember);
        if (!classMethodMemberInfo.isFunctionLike())
        {
            // process indexer here
            return mlir::success();
        }

        auto location = loc(classMember);

        auto accessLevel = getAccessLevel(classMember);        
        auto funcLikeDeclaration = classMember.as<FunctionLikeDeclarationBase>();
        if (mlir::failed(getMethodNameOrPropertyName(
            newClassPtr->isStatic,
            funcLikeDeclaration, 
            classMethodMemberInfo.methodName, 
            classMethodMemberInfo.propertyName, 
            genContext)))
        {
            return mlir::failure();
        }

        assert (!classMethodMemberInfo.methodName.empty());

        // update access based on name
        if (StringRef(classMethodMemberInfo.getName()).starts_with("#")) {
            accessLevel = mlir_ts::AccessLevel::Private;
        }

        if (classMethodMemberInfo.isAbstract && !newClassPtr->isAbstract)
        {
            emitError(location) << "Can't use abstract member '" 
                << classMethodMemberInfo.getName()
                << "' in non-abstract class '" << newClassPtr->fullName << "'";
            return mlir::failure();
        }

        classMember->parent = classDeclarationAST;

        auto funcGenContext = GenContext(genContext);
        funcGenContext.clearScopeVars();
        funcGenContext.thisType = newClassPtr->classType;
        funcGenContext.thisClassType = newClassPtr->classType;
        if (classMethodMemberInfo.isConstructor)
        {
            if (classMethodMemberInfo.isStatic && !genContext.allowPartialResolve)
            {
                createGlobalConstructor(classMember, genContext);
            }

            // adding missing statements
            generateConstructorStatements(classDeclarationAST, classMethodMemberInfo.isStatic, funcGenContext);
        }

        // process dynamic import
        // TODO: why ".new" is virtual method?
        if (newClassPtr->isDynamicImport 
            && (classMethodMemberInfo.isStatic || classMethodMemberInfo.isConstructor || classMethodMemberInfo.methodName == NEW_METHOD_NAME))
        {
            return mlirGenClassMethodMemberDynamicImport(classMethodMemberInfo, orderWeight, genContext);
        }

        if (classMethodMemberInfo.isExport)
        {
            funcLikeDeclaration->internalFlags |= InternalFlags::DllExport;
        }

        if (classMethodMemberInfo.isImport)
        {
            funcLikeDeclaration->internalFlags |= InternalFlags::DllImport;
            //MLIRHelper::addDecoratorIfNotPresent(funcLikeDeclaration, DLL_IMPORT);
        }

        if (newClassPtr->isPublic && accessLevel != mlir_ts::AccessLevel::Private)
        {
            funcLikeDeclaration->internalFlags |= InternalFlags::IsPublic;
        }

        auto [result, funcOp, funcName, isGeneric] =
            mlirGenFunctionLikeDeclaration(funcLikeDeclaration, funcGenContext);
        if (mlir::failed(result))
        {
            return mlir::failure();
        }

        if (funcOp)
        {
            classMethodMemberInfo.setFuncOp(funcOp);
            if (classMethodMemberInfo.registerClassMethodMember(loc(funcLikeDeclaration), orderWeight, accessLevel))
            {
                funcLikeDeclaration->processed = true;
                return mlir::success();
            }

            return mlir::failure();
        }

        return registerGenericClassMethod(classMethodMemberInfo, genContext);
    }

    mlir::LogicalResult MLIRGenImpl::mlirGenClassStaticBlockMember(ClassLikeDeclaration classDeclarationAST,
                                                 ClassInfo::TypePtr newClassPtr, ClassElement classMember,
                                                 const GenContext &genContext)
    {
        // we need to add all static blocks to it
        if (classMember == SyntaxKind::ClassStaticBlockDeclaration)
        {
            auto classStaticBlock = classMember.as<ClassStaticBlockDeclaration>();

            // create function
            auto location = loc(classStaticBlock);

            auto name = MLIRHelper::getAnonymousName(location, ".csb", "");
            auto fullInitGlobalFuncName = getFullNamespaceName(name);

            mlir::OpBuilder::InsertionGuard insertGuard(builder);

            // create global construct
            auto funcType = getFunctionType({}, {}, false);

            if (mlir::failed(mlirGenFunctionBody(location, name, fullInitGlobalFuncName, funcType,
                [&](mlir::Location location, const GenContext &genContext) {
                    return mlirGen(classStaticBlock->body, genContext);
                }, genContext)))
            {
                return mlir::failure();
            }

            addGlobalConstructor(location, fullInitGlobalFuncName);
        }

        return mlir::success();
    }

    mlir::LogicalResult MLIRGenImpl::registerGenericClassMethod(ClassMethodMemberInfo &classMethodMemberInfo, const GenContext &genContext)
    {
        // if funcOp is null, means it is generic
        if (classMethodMemberInfo.funcOp)
        {
            return mlir::success();
        }

        auto funcLikeDeclaration = classMethodMemberInfo.classMember.as<FunctionLikeDeclarationBase>();

        // if it is generic - remove virtual flag
        if (classMethodMemberInfo.isForceVirtual)
        {
            classMethodMemberInfo.isVirtual = false;
        }

        if (classMethodMemberInfo.isStatic || (!classMethodMemberInfo.isAbstract && !classMethodMemberInfo.isVirtual))
        {
            if (classMethodMemberInfo.newClassPtr->getGenericMethodIndex(classMethodMemberInfo.methodName) < 0)
            {
                llvm::SmallVector<TypeParameterDOM::TypePtr> typeParameters;
                if (mlir::failed(
                        processTypeParameters(funcLikeDeclaration->typeParameters, typeParameters, genContext)))
                {
                    return mlir::failure();
                }

                // TODO: review it, ignore in case of ArrowFunction,
                auto [result, funcProto] =
                    getFuncArgTypesOfGenericMethod(funcLikeDeclaration, typeParameters, false, genContext);
                if (mlir::failed(result))
                {
                    return mlir::failure();
                }

                LLVM_DEBUG(llvm::dbgs() << "\n!! registered generic method: " << classMethodMemberInfo.methodName
                                        << ", type: " << funcProto->getFuncType() << "\n";);

                auto &genericMethodInfos = classMethodMemberInfo.newClassPtr->staticGenericMethods;

                // this is generic method
                // the main logic will use Global Generic Functions
                genericMethodInfos.push_back({
                    classMethodMemberInfo.methodName, 
                    funcProto->getFuncType(), 
                    funcProto, 
                    classMethodMemberInfo.isStatic,
                    classMethodMemberInfo.accessLevel});
            }

            return mlir::success();
        }

        emitError(loc(classMethodMemberInfo.classMember)) << "virtual generic methods in class are not allowed";
        return mlir::failure();
    }

    mlir::LogicalResult MLIRGenImpl::mlirGenClassMethodMemberDynamicImport(ClassMethodMemberInfo &classMethodMemberInfo, int orderWeight, const GenContext &genContext)
    {
        auto funcLikeDeclaration = classMethodMemberInfo.classMember.as<FunctionLikeDeclarationBase>();

        auto [funcOp, funcProto, result, isGeneric] =
            mlirGenFunctionPrototype(funcLikeDeclaration, genContext);
        if (mlir::failed(result))
        {
            // in case of ArrowFunction without params and receiver is generic function as well
            return mlir::failure();
        }

        classMethodMemberInfo.setFuncOp(funcOp);

        auto location = loc(funcLikeDeclaration);
        if (mlir::succeeded(mlirGenFunctionLikeDeclarationDynamicImport(
            location, funcOp.getName(), funcOp.getFunctionType(), funcOp.getName(), genContext)))
        {
            // no need to generate method in code
            funcLikeDeclaration->processed = true;
            classMethodMemberInfo.registerClassMethodMember(location, orderWeight, classMethodMemberInfo.getAccessLevel());
            return mlir::success();
        }

        return mlir::failure();
    }

    mlir::LogicalResult MLIRGenImpl::createGlobalConstructor(ClassElement classMember, const GenContext &genContext)
    {
        auto location = loc(classMember);
        auto funcName = getNameOfFunction(classMember, genContext);

        addGlobalConstructor(location, std::get<0>(funcName));

        return mlir::success();
    }

    mlir::LogicalResult MLIRGenImpl::generateConstructorStatements(ClassLikeDeclaration classDeclarationAST, bool staticConstructor,
                                                      const GenContext &genContext)
    {
        NodeFactory nf(NodeFactoryFlags::None);

        auto isClassStatic = hasModifier(classDeclarationAST, SyntaxKind::StaticKeyword);
        for (auto &classMember : classDeclarationAST->members)
        {
            auto isStatic = isClassStatic || hasModifier(classMember, SyntaxKind::StaticKeyword);
            if (classMember == SyntaxKind::PropertyDeclaration)
            {
                if (isStatic != staticConstructor)
                {
                    continue;
                }

                auto propertyDeclaration = classMember.as<PropertyDeclaration>();
                if (!propertyDeclaration->initializer)
                {
                    continue;
                }

                if (staticConstructor)
                {
                    auto isConst = isConstValue(propertyDeclaration->initializer, genContext);
                    if (isConst)
                    {
                        continue;
                    }
                }

                auto memberNamePtr = MLIRHelper::getName(propertyDeclaration->name, stringAllocator);
                if (memberNamePtr.empty())
                {
                    llvm_unreachable("not implemented");
                    return mlir::failure();
                }

                auto _this = nf.createIdentifier(S(THIS_NAME));
                auto _name = nf.createIdentifier(stows(std::string(memberNamePtr)));
                auto _this_name = nf.createPropertyAccessExpression(_this, _name);
                auto _this_name_equal = nf.createBinaryExpression(_this_name, nf.createToken(SyntaxKind::EqualsToken),
                                                                  propertyDeclaration->initializer);
                auto expr_statement = nf.createExpressionStatement(_this_name_equal);

                // NOTE: upward mailbox: drained when the constructor body is generated - see A7
                const_cast<GenContext &>(genContext).generatedStatements.push_back(expr_statement.as<Statement>());
            }

            if (classMember == SyntaxKind::Constructor)
            {
                if (isStatic != staticConstructor)
                {
                    continue;
                }

                auto constructorDeclaration = classMember.as<ConstructorDeclaration>();
                for (auto &parameter : constructorDeclaration->parameters)
                {
                    auto isPublic = hasModifier(parameter, SyntaxKind::PublicKeyword);
                    auto isProtected = hasModifier(parameter, SyntaxKind::ProtectedKeyword);
                    auto isPrivate = hasModifier(parameter, SyntaxKind::PrivateKeyword);

                    if (!(isPublic || isProtected || isPrivate))
                    {
                        continue;
                    }

                    auto propertyNamePtr = MLIRHelper::getName(parameter->name, stringAllocator);
                    if (propertyNamePtr.empty())
                    {
                        llvm_unreachable("not implemented");
                        return mlir::failure();
                    }

                    auto _this = nf.createIdentifier(stows(THIS_NAME));
                    auto _name = nf.createIdentifier(stows(std::string(propertyNamePtr)));
                    auto _this_name = nf.createPropertyAccessExpression(_this, _name);
                    auto _this_name_equal =
                        nf.createBinaryExpression(_this_name, nf.createToken(SyntaxKind::EqualsToken), _name);
                    auto expr_statement = nf.createExpressionStatement(_this_name_equal);

                    // NOTE: upward mailbox: drained when the constructor body is generated - see A7
                    const_cast<GenContext &>(genContext).generatedStatements.push_back(expr_statement.as<Statement>());
                }
            }
        }

        return mlir::success();
    }

} // namespace mlirgen
} // namespace typescript
