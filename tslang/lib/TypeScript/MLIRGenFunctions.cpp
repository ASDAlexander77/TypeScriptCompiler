// Function declaration/prototype/body/capture code generation methods of MLIRGenImpl (see MLIRGenImpl.h).

#include "MLIRGenImpl.h"

namespace typescript
{
namespace mlirgen
{

    std::tuple<mlir::LogicalResult, bool, std::vector<std::shared_ptr<FunctionParamDOM>>> MLIRGenImpl::mlirGenParameters(
        SignatureDeclarationBase parametersContextAST, const GenContext &genContext)
    {
        // to remove variables such as "this" from scope after using it in params context
        SymbolTableScopeT varScope(symbolTable);

        auto isGenericTypes = false;
        std::vector<std::shared_ptr<FunctionParamDOM>> params;

        SyntaxKind kind = parametersContextAST;
        // add this param
        auto isStatic = 
            hasModifier(parametersContextAST->parent, SyntaxKind::StaticKeyword) 
            || hasModifier(parametersContextAST, SyntaxKind::StaticKeyword);

        if (parametersContextAST->parent == SyntaxKind::InterfaceDeclaration)
        {
            params.push_back(std::make_shared<FunctionParamDOM>(THIS_NAME, getOpaqueType(), loc(parametersContextAST)));
        }
        else if (!isStatic &&
            (kind == SyntaxKind::MethodDeclaration || kind == SyntaxKind::Constructor ||
             kind == SyntaxKind::GetAccessor || kind == SyntaxKind::SetAccessor))
        {
            params.push_back(
                std::make_shared<FunctionParamDOM>(THIS_NAME, genContext.thisType, loc(parametersContextAST)));
        }
        else if (!isStatic && genContext.thisType && !!parametersContextAST->parent &&
            (kind == SyntaxKind::FunctionExpression ||
             kind == SyntaxKind::ArrowFunction))
        {            
            // TODO: this is very tricky code, if we rediscover function again and if by any chance thisType is not null, it will append thisType to lambda which very wrong code
            params.push_back(
                std::make_shared<FunctionParamDOM>(THIS_NAME, genContext.thisType, loc(parametersContextAST)));
        }

        auto formalParams = parametersContextAST->parameters;
        for (auto [index, arg] : enumerate(formalParams))
        {
            auto namePtr = MLIRHelper::getName(arg->name, stringAllocator);
            if (namePtr.empty())
            {
                namePtr = getArgumentName(index);
            }

            auto isBindingPattern = arg->name == SyntaxKind::ObjectBindingPattern || arg->name == SyntaxKind::ArrayBindingPattern;

            mlir::Type type;
            auto isMultiArgs = !!arg->dotDotDotToken;
            auto isOptional = !!arg->questionToken;
            auto typeParameter = arg->type;

            auto location = loc(typeParameter);

            if (typeParameter)
            {
                type = getType(typeParameter, genContext);
            }

            // special case, setup 'this' and type provided 
            if (namePtr == THIS_NAME && type) 
            {
                // NOTE: upward mailbox: explicit this-parameter type must reach the prototype chain - see A7
                const_cast<GenContext &>(genContext).thisType = type;
                LLVM_DEBUG(dbgs() << "\n!! param " << THIS_NAME << " mapped to type " << type << "\n");

                auto varDecl = std::make_shared<VariableDeclarationDOM>(THIS_NAME, type, location);
                auto typeRefVal = builder.create<mlir_ts::TypeRefOp>(location, type);
                declare(location, varDecl, typeRefVal, genContext);
            }

            // process init value
            auto initializer = arg->initializer;
            if (initializer)
            {
                auto evalType = evaluate(initializer, genContext);
                if (evalType)
                {
                    evalType = mth.wideStorageType(evalType);

                    // TODO: set type if not provided
                    isOptional = true;
                    if (mth.isNoneType(type))
                    {
                        type = evalType;
                    }
                }
            }

            if (mth.isNoneType(type) && genContext.receiverFuncType && mth.isAnyFunctionType(genContext.receiverFuncType))
            {
                type = mth.getParamFromFuncRef(genContext.receiverFuncType, index);
                if (!type)
                {
                    emitError(location) << "can't resolve type for parameter '" << namePtr << "', the receiver function has less parameters.";
                    return {mlir::failure(), isGenericTypes, params};                    
                }

                LLVM_DEBUG(dbgs() << "\n!! param " << namePtr << " mapped to type " << type << "\n");

                isGenericTypes |= mth.isGenericType(type);
            }

            // in case of binding
            if (mth.isNoneType(type) && isBindingPattern)
            {
                type = mlirGenParameterObjectOrArrayBinding(arg->name, genContext);
                LLVM_DEBUG(dbgs() << "\n!! binding param " << namePtr << " is type " << type << "\n");
            }

            if (mth.isNoneType(type))
            {
                if (!typeParameter && !initializer)
                {
#ifndef ANY_AS_DEFAULT
                    if (!genContext.allowPartialResolve && !genContext.dummyRun)
                    {
                        auto funcName = MLIRHelper::getName(parametersContextAST->name);
                        emitError(loc(arg))
                            << "type of parameter '" << namePtr
                            << "' is not provided, parameter must have type or initializer, function: " << funcName;
                    }
                    return {mlir::failure(), isGenericTypes, params};
#else
                    emitWarning(loc(parametersContextAST)) << "type for parameter '" << namePtr << "' is any";
                    type = getAnyType();
#endif
                }
                else
                {
                    emitError(location) << "can't resolve type for parameter '" << namePtr << "'";
                    return {mlir::failure(), isGenericTypes, params};
                }
            }

            if (isa<mlir_ts::VoidType>(type))
            {
                emitError(location, "'Void' can't be used as parameter type");
                return {mlir::failure(), isGenericTypes, params};
            }

            if (isa<mlir_ts::NeverType>(type))
            {
                emitError(location, "'Never' can't be used as parameter type");
                return {mlir::failure(), isGenericTypes, params};
            }

            if (isBindingPattern)
            {
                params.push_back(
                    std::make_shared<FunctionParamDOM>(
                        namePtr, type, loc(arg), isOptional, isMultiArgs, initializer, arg->name));
            }
            else
            {
                params.push_back(
                    std::make_shared<FunctionParamDOM>(
                        namePtr, type, loc(arg), isOptional, isMultiArgs, initializer));
            }
        }

        return {mlir::success(), isGenericTypes, params};
    }

    std::tuple<std::string, std::string> MLIRGenImpl::getNameOfFunction(SignatureDeclarationBase signatureDeclarationBaseAST,
                                                           const GenContext &genContext)
    {
        auto name = getNameWithArguments(signatureDeclarationBaseAST, genContext);
        std::string objectOwnerName;
        if (signatureDeclarationBaseAST->parent == SyntaxKind::ClassDeclaration ||
            signatureDeclarationBaseAST->parent == SyntaxKind::ClassExpression)
        {
            objectOwnerName =
                getNameWithArguments(signatureDeclarationBaseAST->parent.as<ClassDeclaration>(), genContext);
        }
        else if (signatureDeclarationBaseAST->parent == SyntaxKind::InterfaceDeclaration)
        {
            objectOwnerName =
                getNameWithArguments(signatureDeclarationBaseAST->parent.as<InterfaceDeclaration>(), genContext);
        }
        else if (signatureDeclarationBaseAST->parent == SyntaxKind::ObjectLiteralExpression)
        {
            objectOwnerName = mlir::cast<mlir_ts::ObjectStorageType>(
                mlir::cast<mlir_ts::ObjectType>(genContext.thisType).getStorageType()).getName().getValue();
        }
        else if (genContext.funcOp)
        {
            mlir_ts::FuncOp funcOp = genContext.funcOp;
            objectOwnerName = funcOp.getSymName().str();
        }

        if (signatureDeclarationBaseAST == SyntaxKind::MethodDeclaration)
        {
            if (!objectOwnerName.empty())
            {
                // class method name
                name = objectOwnerName + "." + name;
            }
            else
            {
                name = MLIRHelper::getAnonymousName(loc_check(signatureDeclarationBaseAST), ".md", "");
            }
        }
        // TODO: for new () interfaces
        else if (signatureDeclarationBaseAST == SyntaxKind::MethodSignature 
                || signatureDeclarationBaseAST == SyntaxKind::ConstructSignature)
        {
            // class method name
            name = objectOwnerName + "." + name;
        }
        else if (signatureDeclarationBaseAST == SyntaxKind::GetAccessor)
        {
            // class method name
            name = objectOwnerName + ".get_" + name;
        }
        else if (signatureDeclarationBaseAST == SyntaxKind::SetAccessor)
        {
            // class method name
            name = objectOwnerName + ".set_" + name;
        }
        else if (signatureDeclarationBaseAST == SyntaxKind::Constructor)
        {
            // class method name
            auto isStatic = 
                hasModifier(signatureDeclarationBaseAST->parent, SyntaxKind::StaticKeyword)
                || hasModifier(signatureDeclarationBaseAST, SyntaxKind::StaticKeyword);
            if (isStatic)
            {
                name = objectOwnerName + "." + STATIC_NAME + "_" + name;
            }
            else
            {
                name = objectOwnerName + "." + name;
            }
        }

        auto fullName = concatFullNamespaceName(name);
        return std::make_tuple(fullName, name);
    }

    std::tuple<mlir_ts::FuncOp, FunctionPrototypeDOM::TypePtr, mlir::LogicalResult, bool> MLIRGenImpl::mlirGenFunctionPrototype(
        FunctionLikeDeclarationBase functionLikeDeclarationBaseAST, const GenContext &genContext)
    {
        auto location = loc(functionLikeDeclarationBaseAST);

        mlir_ts::FuncOp funcOp;

        auto [funcProto, funcType, argTypes] =
            mlirGenFunctionSignaturePrototype(
                functionLikeDeclarationBaseAST, 
                hasModifier(functionLikeDeclarationBaseAST, SyntaxKind::DeclareKeyword), 
                genContext);
        if (!funcProto)
        {
            return std::make_tuple(funcOp, funcProto, mlir::failure(), false);
        }

        GenContext funcProtoGenContext(genContext);
        funcProtoGenContext.funcProto = funcProto;

        auto fullName = funcProto->getName();

        mlir_ts::FunctionType functionDiscovered;
        auto funcTypeIt = getFunctionTypeMap().find(fullName);
        if (funcTypeIt != getFunctionTypeMap().end())
        {
            functionDiscovered = (*funcTypeIt).second;
        }        

        // discover type & args
        // seems we need to discover it all the time due to captured vars
        auto detectReturnType = (!funcType || funcProtoGenContext.forceDiscover || !functionDiscovered)
            && !funcProto->getIsGeneric();
        if (detectReturnType)
        {
            // register function to be able to call it if used in recursive call
            // auto funcTypeTemp = getFunctionType(argTypes, builder.getNoneType(), funcProto->isMultiArgs());
            // auto funcOpTemp = mlir_ts::FuncOp::create(location, fullName, funcTypeTemp, {});
            // registerFunctionOp(funcProto, funcOpTemp);        

            if (mlir::succeeded(discoverFunctionReturnTypeAndCapturedVars(functionLikeDeclarationBaseAST, fullName,
                                                                          argTypes, funcProto, funcProtoGenContext)))
            {
                if (!funcProtoGenContext.forceDiscover && funcType && funcType.getNumResults() > 0)
                {
                    funcProto->setReturnType(funcType.getResult(0));
                }
                else if (auto typeParameter = functionLikeDeclarationBaseAST->type)
                {
                    // rewrite ret type with actual value in case of specialized generic
                    auto returnType = getType(typeParameter, funcProtoGenContext);
                    funcProto->setReturnType(returnType);
                }
                else if (funcProtoGenContext.receiverFuncType)
                {
                    // rewrite ret type with actual value
                    auto &argTypeDestFuncType = funcProtoGenContext.receiverFuncType;
                    auto retTypeFromReceiver = mth.isAnyFunctionType(argTypeDestFuncType) 
                        ? mth.getReturnTypeFromFuncRef(argTypeDestFuncType)
                        : mlir::Type();
                    if (retTypeFromReceiver 
                        && !mth.isNoneType(retTypeFromReceiver) 
                        && !mth.isGenericType(retTypeFromReceiver))
                    {
                        funcProto->setReturnType(retTypeFromReceiver);
                        LLVM_DEBUG(llvm::dbgs()
                                       << "\n!! set return type from receiver: " << retTypeFromReceiver << "\n";);
                    }
                }

                // create funcType
                if (funcProto->getReturnType())
                {
                    funcType = getFunctionType(argTypes, funcProto->getReturnType(), funcProto->isMultiArgs());
                }
                else
                {
                    // no return type
                    funcType = getFunctionType(argTypes, {}, funcProto->isMultiArgs());
                }
            }
            else
            {
                // false result
                return std::make_tuple(funcOp, funcProto, mlir::failure(), false);
            }
        }
        else if (functionDiscovered)
        {
            funcType = functionDiscovered;
        }

        // we need it, when we run rediscovery second time
        if (!funcProto->getHasExtraFields())
        {
            funcProto->setHasExtraFields(existLocalVarsInThisContextMap(funcProto->getName()));
        }

        // a concrete instantiation of a generic function (e.g. identity<number>) reprocesses
        // the SAME AST node as the bare template (`export function identity<T>(...)`), so
        // processFunctionAttributes's own getExportModifier check below would otherwise mark
        // this LOCAL, per-instantiation specialization as exported too - wrong for the same
        // two reasons the class-generic fix (class-generic-declaration-export-fix) suppresses
        // isExport on a specialized ClassInfo: (1) a specialization materialized by whichever
        // module instantiates it is local to that module, not a re-export of it (the bare
        // template's own export is handled separately via registerGenericFunctionLike /
        // addGenericFunctionDeclarationToExport); (2) a multi-type-param instantiation's
        // mangled name contains a raw comma (e.g. M.pair<!ts.number,!ts.string>), a
        // metacharacter in the linker's `/EXPORT:name[,option]` directive syntax - lld rejects
        // it outright ("invalid /export:"). NOTE: can't key this off
        // genContext.instantiateSpecializedFunction - mlirGenFunctionLikeDeclaration
        // deliberately clears that flag on funcDeclGenContext before calling
        // mlirGenFunctionPrototype (to stop nested generics being instantiated by mistake),
        // so by the time we get here it always reads false. typeParamsWithArgs is the
        // signal that actually survives - same one the class fix uses. Must be passed INTO
        // processFunctionAttributes (not applied to its return value afterward) because it
        // pushes the "export" attribute into `attrs` itself before returning.
        auto suppressExportForGenericInstantiation =
            functionLikeDeclarationBaseAST->typeParameters.size() > 0 && !genContext.typeParamsWithArgs.empty();

        SmallVector<mlir::NamedAttribute> attrs;
        auto dllExport = processFunctionAttributes(location, fullName, functionLikeDeclarationBaseAST, attrs, funcProtoGenContext,
            suppressExportForGenericInstantiation);

        if (funcType)
        {
            auto it = getCaptureVarsMap().find(funcProto->getName());
            auto hasCapturedVars = funcProto->getHasCapturedVars() || (it != getCaptureVarsMap().end());
            if (hasCapturedVars)
            {
                // important set when it is discovered and in process second type
                funcProto->setHasCapturedVars(true);
                funcOp = mlir_ts::FuncOp::create(location, fullName, funcType, attrs);
            }
            else
            {
                funcOp = mlir_ts::FuncOp::create(location, fullName, funcType, attrs);
            }

            funcProto->setFuncType(funcType);

            if (dllExport)
            {
                if (functionLikeDeclarationBaseAST == SyntaxKind::FunctionDeclaration
                    || functionLikeDeclarationBaseAST == SyntaxKind::ArrowFunction)
                {
                    addFunctionDeclarationToExport(funcProto, currentNamespace);
                }
            }
        }

        if (!funcProto->getIsGeneric())
        {
            auto funcTypeIt = getFunctionTypeMap().find(fullName);
            if (funcTypeIt != getFunctionTypeMap().end())
            {
                getFunctionTypeMap().erase(funcTypeIt);
            }

            getFunctionTypeMap().insert({fullName, funcType});

            LLVM_DEBUG(llvm::dbgs() << "\n!! register func name: " << fullName << ", type: " << funcType << "\n";);
        }

        return std::make_tuple(funcOp, funcProto, mlir::success(), funcProto->getIsGeneric());
    }

    mlir::LogicalResult MLIRGenImpl::discoverFunctionReturnTypeAndCapturedVars(
        FunctionLikeDeclarationBase functionLikeDeclarationBaseAST, StringRef name, SmallVector<mlir::Type> &argTypes,
        const FunctionPrototypeDOM::TypePtr &funcProto, const GenContext &genContext)
    {
        if (funcProto->getDiscovered())
        {
            return mlir::failure();
        }

        LLVM_DEBUG(llvm::dbgs() << "\n\tdiscovering 'return type' & 'captured variables' for : " << name << "\n";);

        mlir::OpBuilder::InsertionGuard guard(builder);

        auto partialDeclFuncType = getFunctionType(argTypes, {}, false);
        auto dummyFuncOp = mlir_ts::FuncOp::create(loc(functionLikeDeclarationBaseAST), name, partialDeclFuncType);

        {
            // simulate scope
            SymbolTableScopeT varScope(symbolTable);

            llvm::ScopedHashTableScope<StringRef, VariableDeclarationDOM::TypePtr> 
                fullNameGlobalsMapScope(fullNameGlobalsMap);

            // owned here; GenContext borrows pointers to them (see GenContext::clean)
            SmallVector<mlir::Block *> cleanUpsList;
            SmallVector<mlir::Operation *> cleanUpOpsList;
            PassResult passResultData;
            int discoverState = 1;

            GenContext genContextWithPassResult{};
            genContextWithPassResult.funcOp = dummyFuncOp;
            genContextWithPassResult.thisType = genContext.thisType;
            genContextWithPassResult.thisClassType = genContext.thisClassType;
            genContextWithPassResult.allowPartialResolve = true;
            genContextWithPassResult.dummyRun = true;
            genContextWithPassResult.cleanUps = &cleanUpsList;
            genContextWithPassResult.cleanUpOps = &cleanUpOpsList;
            genContextWithPassResult.passResult = &passResultData;
            genContextWithPassResult.state = &discoverState;
            genContextWithPassResult.allocateVarsInContextThis =
                (functionLikeDeclarationBaseAST->internalFlags & InternalFlags::VarsInObjectContext) ==
                InternalFlags::VarsInObjectContext;
            genContextWithPassResult.discoverParamsOnly = genContext.discoverParamsOnly;
            genContextWithPassResult.typeAliasMap = genContext.typeAliasMap;
            genContextWithPassResult.typeParamsWithArgs = genContext.typeParamsWithArgs;
            genContextWithPassResult.postponedMessages = genContext.postponedMessages;

            registerNamespace(funcProto->getNameWithoutNamespace(), true);

            if (succeeded(mlirGenFunctionBody(functionLikeDeclarationBaseAST, name, dummyFuncOp, funcProto,
                                              genContextWithPassResult)))
            {
                exitNamespace();

                auto &passResult = genContextWithPassResult.passResult;
                if (passResult->functionReturnTypeShouldBeProvided
                    && mth.isNoneType(passResult->functionReturnType))
                {
                    // has return value but type is not provided yet
                    genContextWithPassResult.clean();

                    // if THIS discovery is itself nested inside an outer speculative
                    // discovery/dummy run (e.g. an object literal's method being
                    // return-type-discovered as a side effect of discovering the
                    // enclosing function - see the allowPartialResolve tolerance in
                    // mlirGenPropertyAccessExpressionBaseLogic), a sibling member's
                    // prototype may not be registered yet, so a return expression that
                    // depends on it can legitimately come back as "unknown" here. Don't
                    // hard-fail the whole discovery over that - the outer caller (and
                    // the real, non-dummy compile pass) will resolve it once every
                    // sibling's prototype is registered.
                    if (genContext.dummyRun || genContext.allowPartialResolve)
                    {
                        return mlir::failure();
                    }

                    emitError(loc(functionLikeDeclarationBaseAST)) << "'return' is not found in function or return type can't be resolved";
                    return mlir::failure();
                }

                funcProto->setDiscovered(true);
                auto discoveredType = passResult->functionReturnType;
                if (discoveredType && discoveredType != funcProto->getReturnType())
                {
                    // TODO: do we need to convert it here? maybe send it as const object?

                    funcProto->setReturnType(mth.convertConstArrayTypeToArrayType(discoveredType));
                    LLVM_DEBUG(llvm::dbgs()
                                   << "\n!! ret type: " << funcProto->getReturnType() << ", name: " << name << "\n";);
                }

                // if we have captured parameters, add first param to send lambda's type(class)
                if (passResult->outerVariables.size() > 0)
                {
                    MLIRCodeLogic mcl(builder, compileOptions);
                    auto isObjectType =
                        genContext.thisType != nullptr && isa<mlir_ts::ObjectType>(genContext.thisType);
                    if (!isObjectType)
                    {
                        argTypes.insert(argTypes.begin(), mcl.CaptureType(passResult->outerVariables));
                    }

                    getCaptureVarsMap().insert({name, passResult->outerVariables});
                    funcProto->setHasCapturedVars(true);

                    LLVM_DEBUG(llvm::dbgs() << "\n!! has captured vars, name: " << name << "\n";);

                    LLVM_DEBUG(for (auto& var : passResult->outerVariables)
                    {
                        llvm::dbgs() << "\n!! ...captured var - name: " << var.second->getName() << ", type: " << var.second->getType() << "\n";
                    });
                }

                if (passResult->extraFieldsInThisContext.size() > 0)
                {
                    getLocalVarsInThisContextMap().insert({name, passResult->extraFieldsInThisContext});

                    funcProto->setHasExtraFields(true);
                }

                genContextWithPassResult.clean();

                LLVM_DEBUG(llvm::dbgs() << "\n\tSUCCESS - discovering 'return type' & 'captured variables' for : " << name << "\n";);

                return mlir::success();
            }
            else
            {
                exitNamespace();

                genContextWithPassResult.clean();

                LLVM_DEBUG(llvm::dbgs() << "\n\tFAILED - discovering 'return type' & 'captured variables' for : " << name << "\n";);

                return mlir::failure();
            }
        }
    }

    mlir::LogicalResult MLIRGenImpl::mlirGen(FunctionDeclaration functionDeclarationAST, const GenContext &genContext)
    {
        auto funcGenContext = GenContext(genContext);
        funcGenContext.clearScopeVars();
        // declaring function which is nested and object should not have this context (unless it is part of object declaration)
        if (!functionDeclarationAST->parent && funcGenContext.thisType != nullptr)
        {
            funcGenContext.thisType = nullptr;
        }

        mlir::OpBuilder::InsertionGuard guard(builder);
        auto res = mlirGenFunctionLikeDeclaration(functionDeclarationAST, funcGenContext);
        return std::get<0>(res);
    }

    FunctionLikeDeclarationBase MLIRGenImpl::buildGeneratorWrapperDeclaration(
        FunctionLikeDeclarationBase functionLikeDeclarationBaseAST, mlir::Location location)
    {
        auto fixThisReference = functionLikeDeclarationBaseAST == SyntaxKind::MethodDeclaration;
        if (functionLikeDeclarationBaseAST->parameters.size() > 0)
        {
            auto nameNode = functionLikeDeclarationBaseAST->parameters.front()->name;
            if (nameNode == SyntaxKind::Identifier)
            {
                auto ident = nameNode.as<Identifier>();
                if (ident->escapedText == S(THIS_NAME))
                {
                    fixThisReference = true;
                }
            }
        }
        
        NodeFactory nf(NodeFactoryFlags::None);

        auto stepIdent = nf.createIdentifier(S(GENERATOR_STEP));

        // create return object
        NodeArray<ObjectLiteralElementLike> generatorObjectProperties;

        // add step field
        auto stepProp = nf.createPropertyAssignment(stepIdent, nf.createNumericLiteral(S("0"), TokenFlags::None));
        generatorObjectProperties.push_back(stepProp);

        // create body of next method
        NodeArray<Statement> nextStatements;

        // add main switcher
        auto stepAccess = nf.createPropertyAccessExpression(nf.createToken(SyntaxKind::ThisKeyword), stepIdent);

        // call stateswitch
        auto callStat = nf.createExpressionStatement(
            nf.createCallExpression(nf.createIdentifier(S(GENERATOR_SWITCHSTATE)), undefined, {stepAccess}));

        nextStatements.push_back(callStat);

        // add function body to statements to first step
        if (functionLikeDeclarationBaseAST->body == SyntaxKind::Block)
        {
            // process every statement
            auto block = functionLikeDeclarationBaseAST->body.as<ts::Block>();
            for (auto statement : block->statements)
            {
                nextStatements.push_back(statement);
            }
        }
        else if (functionLikeDeclarationBaseAST->body)
        {
            nextStatements.push_back(functionLikeDeclarationBaseAST->body);
        }

        // add next statements
        // add default return with empty
        nextStatements.push_back(
            nf.createReturnStatement(getYieldReturnObject(nf, location, nf.createIdentifier(S(UNDEFINED_NAME)), true)));

        // create next body
        auto nextBody = nf.createBlock(nextStatements, /*multiLine*/ false);

        // create method next in object
        auto nextMethodDecl =
            nf.createMethodDeclaration(undefined, undefined, nf.createIdentifier(S(ITERATOR_NEXT)), undefined,
                                       undefined, undefined, undefined, nextBody);
        nextMethodDecl->internalFlags |= InternalFlags::VarsInObjectContext;

        // copy location info, to fix issue with names of anonymous functions
        nextMethodDecl->pos = functionLikeDeclarationBaseAST->pos;
        nextMethodDecl->_end = functionLikeDeclarationBaseAST->_end;

        if (fixThisReference)
        {
            FilterVisitorSkipFuncsAST<Node> visitor(SyntaxKind::ThisKeyword, [&](auto thisNode) {
                thisNode->internalFlags |= InternalFlags::ThisArgAlias;
            });

            for (auto it = begin(nextStatements) + 1; it != end(nextStatements); ++it)
            {
                visitor.visit(*it);
            }
        }

        generatorObjectProperties.push_back(nextMethodDecl);

        auto generatorObject = nf.createObjectLiteralExpression(generatorObjectProperties, false);
        // the generator object has mutable identity (`step` advanced by next());
        // it must be a reference type so aliases (params, closures, const bindings)
        // share state -- box it on the GC heap instead of the default value tuple
        generatorObject->internalFlags |= InternalFlags::BoxAsObject;

        // copy location info, to fix issue with names of anonymous functions
        generatorObject->pos = functionLikeDeclarationBaseAST->pos;
        generatorObject->_end = functionLikeDeclarationBaseAST->_end;

        // generator body
        NodeArray<Statement> generatorStatements;

        // TODO: this is hack, adding this as thisArg alias
        if (fixThisReference)
        {
            // TODO: this is temp hack, add this alias as thisArg, 
            NodeArray<VariableDeclaration> _thisArgDeclarations;
            auto _thisArg = nf.createIdentifier(S(THIS_ALIAS));
            _thisArgDeclarations.push_back(nf.createVariableDeclaration(_thisArg, undefined, undefined, nf.createToken(SyntaxKind::ThisKeyword)));
            auto _thisArgList = nf.createVariableDeclarationList(_thisArgDeclarations, NodeFlags::Const);

            generatorStatements.push_back(nf.createVariableStatement(undefined, _thisArgList));
        }

        // step 1, add return object
        auto retStat = nf.createReturnStatement(generatorObject);
        generatorStatements.push_back(retStat);

        auto body = nf.createBlock(generatorStatements, /*multiLine*/ false);

        if (functionLikeDeclarationBaseAST == SyntaxKind::MethodDeclaration)
        {
            auto methodOp = nf.createMethodDeclaration(
                functionLikeDeclarationBaseAST->modifiers, undefined,
                functionLikeDeclarationBaseAST->name, undefined, functionLikeDeclarationBaseAST->typeParameters,
                functionLikeDeclarationBaseAST->parameters, functionLikeDeclarationBaseAST->type, body);

            // copy location info, to fix issue with names of anonymous functions
            methodOp->pos = functionLikeDeclarationBaseAST->pos;
            methodOp->_end = functionLikeDeclarationBaseAST->_end;

            // to ensure correct full name
            methodOp->parent = functionLikeDeclarationBaseAST->parent;

            LLVM_DEBUG(printDebug(methodOp););

            return methodOp;
        }
        else
        {
            auto funcOp = nf.createFunctionDeclaration(
                functionLikeDeclarationBaseAST->modifiers, undefined,
                functionLikeDeclarationBaseAST->name, functionLikeDeclarationBaseAST->typeParameters,
                functionLikeDeclarationBaseAST->parameters, functionLikeDeclarationBaseAST->type, body);

            // copy location info, to fix issue with names of anonymous functions
            funcOp->pos = functionLikeDeclarationBaseAST->pos;
            funcOp->_end = functionLikeDeclarationBaseAST->_end;

            LLVM_DEBUG(printDebug(funcOp););

            return funcOp;
        }
    }

    std::tuple<mlir::LogicalResult, mlir_ts::FuncOp, std::string, bool> MLIRGenImpl::mlirGenFunctionGenerator(
        FunctionLikeDeclarationBase functionLikeDeclarationBaseAST, const GenContext &genContext)
    {
        auto location = loc(functionLikeDeclarationBaseAST);
        auto wrapperDecl = buildGeneratorWrapperDeclaration(functionLikeDeclarationBaseAST, location);
        return mlirGenFunctionLikeDeclaration(wrapperDecl, genContext);
    }

    bool MLIRGenImpl::registerFunctionOp(FunctionPrototypeDOM::TypePtr funcProto, mlir_ts::FuncOp funcOp)
    {
        auto name = funcProto->getNameWithoutNamespace();
        if (!getFunctionMap().count(name))
        {
            getFunctionMap().insert({name, makeFunctionEntry(funcOp)});

            LLVM_DEBUG(llvm::dbgs() << "\n!! reg. func: " << name << " type:" << funcOp.getFunctionType() << " function name: " << funcProto->getName()
                                    << " num inputs:" << mlir::cast<mlir_ts::FunctionType>(funcOp.getFunctionType()).getNumInputs()
                                    << "\n";);

            return true;
        }

        LLVM_DEBUG(llvm::dbgs() << "\n!! re-reg. func: " << name << " type:" << funcOp.getFunctionType() << " function name: " << funcProto->getName()
                                << " num inputs:" << mlir::cast<mlir_ts::FunctionType>(funcOp.getFunctionType()).getNumInputs()
                                << "\n";);

        return false;
    }

    std::tuple<mlir::LogicalResult, mlir_ts::FuncOp, std::string, bool> MLIRGenImpl::mlirGenFunctionLikeDeclaration(
        FunctionLikeDeclarationBase functionLikeDeclarationBaseAST, const GenContext &genContext)
    {
        auto funcDeclGenContext = GenContext(genContext);

        auto instantiateSpecializedFunction = funcDeclGenContext.instantiateSpecializedFunction;
                
        auto isGenericFunction = 
            functionLikeDeclarationBaseAST->typeParameters.size() > 0 
            || !genContext.isGlobalVarReceiver && isGenericParameters(functionLikeDeclarationBaseAST, genContext);
        if (isGenericFunction && !instantiateSpecializedFunction)
        {
            auto [result, name] = registerGenericFunctionLike(functionLikeDeclarationBaseAST, false, funcDeclGenContext);
            return {result, mlir_ts::FuncOp(), name, false};
        }

        // check if it is generator
        if (functionLikeDeclarationBaseAST->asteriskToken)
        {
            // this is generator, let's generate other function out of it
            return mlirGenFunctionGenerator(functionLikeDeclarationBaseAST, funcDeclGenContext);
        }

        // we need to clear instantiateSpecializedFunction otherwise nested generics will be 
        // instantiated as well by mistake
        funcDeclGenContext.instantiateSpecializedFunction = false;

        // do not process generic functions more then 1 time
        auto checkIfCreated = isGenericFunction && instantiateSpecializedFunction;
        if (checkIfCreated)
        {
            auto [fullFunctionName, functionName] = getNameOfFunction(functionLikeDeclarationBaseAST, funcDeclGenContext);

            auto funcEntry = lookupFunctionMap(functionName);
            if (funcEntry && theModule.lookupSymbol(functionName)
                || theModule.lookupSymbol(fullFunctionName))
            {
                // resolve a live op from the module instead of returning a cached handle;
                // the registered symbol is usually the full name
                auto funcOp = theModule.lookupSymbol<mlir_ts::FuncOp>(functionName);
                if (!funcOp)
                {
                    funcOp = theModule.lookupSymbol<mlir_ts::FuncOp>(fullFunctionName);
                }

                return {mlir::success(), funcOp, functionName, false};
            }
        }

        // go to root
        mlir::OpBuilder::InsertPoint savePoint;
        if (isGenericFunction)
        {
            savePoint = builder.saveInsertionPoint();
            builder.setInsertionPointToStart(theModule.getBody());
        }

        auto location = loc(functionLikeDeclarationBaseAST);

        auto [funcOp, funcProto, result, isGeneric] =
            mlirGenFunctionPrototype(functionLikeDeclarationBaseAST, funcDeclGenContext);
        if (mlir::failed(result))
        {
            // in case of ArrowFunction without params and receiver is generic function as well
            return {result, funcOp, "", false};
        }

        if (mlir::succeeded(result) && isGeneric)
        {
            auto [result, name] = registerGenericFunctionLike(functionLikeDeclarationBaseAST, true, funcDeclGenContext);
            return {result, funcOp, name, isGeneric};
        }

        // check decorator for class
        auto dynamicImport = false;
        iterateDecorators(functionLikeDeclarationBaseAST, genContext, [&](StringRef name, SmallVector<StringRef> args) {
            if (name == DLL_IMPORT && args.size() > 0)
            {
                dynamicImport = true;
            }
        });

        if (dynamicImport)
        {
            // TODO: we do not need to register funcOp as we need to reference global variables
            auto result = mlirGenFunctionLikeDeclarationDynamicImport(
                location, funcProto->getNameWithoutNamespace(), funcOp.getFunctionType(), 
                funcProto->getName(), funcDeclGenContext, false);
            return {result, funcOp, funcProto->getName().str(), false};
        }

        auto funcGenContext = GenContext(funcDeclGenContext);
        funcGenContext.clearScopeVars();
        funcGenContext.funcOp = funcOp;
        int funcState = 1;
        funcGenContext.state = &funcState;
        // if funcGenContext.passResult is null and allocateVarsInContextThis is true, this type should contain fully
        // defined object with local variables as fields
        funcGenContext.allocateVarsInContextThis =
            (functionLikeDeclarationBaseAST->internalFlags & InternalFlags::VarsInObjectContext) ==
            InternalFlags::VarsInObjectContext;

        auto it = getCaptureVarsMap().find(funcProto->getName());
        if (it != getCaptureVarsMap().end())
        {
            funcGenContext.capturedVars = &it->getValue();

            LLVM_DEBUG(llvm::dbgs() << "\n!! func has captured vars: " << funcProto->getName() << "\n";);
        }
        else
        {
            assert(funcGenContext.capturedVars == nullptr);
        }

        // register function to be able to call it if used in recursive call
        registerFunctionOp(funcProto, funcOp);

        // generate body
        auto resultFromBody = mlir::failure();
        {
            MLIRNamespaceGuard nsGuard(currentNamespace);
            registerNamespace(funcProto->getNameWithoutNamespace(), true);

            SymbolTableScopeT varScope(symbolTable);
            resultFromBody = mlirGenFunctionBody(
                functionLikeDeclarationBaseAST, funcProto->getNameWithoutNamespace(), funcOp, funcProto, funcGenContext);
        }

        if (mlir::failed(resultFromBody))
        {
            return {mlir::failure(), funcOp, "", false};
        }

        // set visibility index
        auto isPublic = 
            getExportModifier(functionLikeDeclarationBaseAST)
            /* we need to forcebly set to Public to prevent SymbolDCEPass to remove unused name */
            || hasModifier(functionLikeDeclarationBaseAST, SyntaxKind::ExportKeyword);

        // force public
        isPublic |= 
            ((functionLikeDeclarationBaseAST->internalFlags & InternalFlags::DllExport) == InternalFlags::DllExport)
            || ((functionLikeDeclarationBaseAST->internalFlags & InternalFlags::IsPublic) == InternalFlags::IsPublic)
            || funcProto->getName() == MAIN_ENTRY_NAME;

        // if explicit public/protected - set public visibility
        if (hasModifier(functionLikeDeclarationBaseAST, SyntaxKind::PublicKeyword) 
            || hasModifier(functionLikeDeclarationBaseAST, SyntaxKind::ProtectedKeyword)) 
        {
            isPublic = true;
        }

        // if explicit private - do not set public visibility
        if (hasModifier(functionLikeDeclarationBaseAST, SyntaxKind::PrivateKeyword)) 
        {
            isPublic = false;
        }

        if (isPublic && !funcProto->getNoBody() && !declarationMode)
        {
            funcOp.setPublic();
        }
        else
        {
            funcOp.setPrivate();
        }

        if (!funcDeclGenContext.dummyRun)
        {
            theModule.push_back(funcOp);
        }

        if (isGenericFunction)
        {
            builder.restoreInsertionPoint(savePoint);
        }
        else
        {
            builder.setInsertionPointAfter(funcOp);
        }

        return {mlir::success(), funcOp, funcProto->getName().str(), false};
    }

    mlir::LogicalResult MLIRGenImpl::mlirGenFunctionLikeDeclarationDynamicImport(mlir::Location location, StringRef funcName, 
        mlir_ts::FunctionType functionType, StringRef dllFuncName, const GenContext &genContext, bool isFullNamespaceName)
    {
        registerVariable(location, funcName, isFullNamespaceName, VariableType::Var,
            [&](mlir::Location location, const GenContext &context) -> TypeValueInitType {
                // add command to load reference fron DLL
                auto fullName = V(mlirGenStringValue(location, dllFuncName.str(), true));
                auto referenceToFuncOpaque = builder.create<mlir_ts::SearchForAddressOfSymbolOp>(location, getOpaqueType(), fullName);
                auto result = cast(location, functionType, referenceToFuncOpaque, genContext);
                auto referenceToFunc = V(result);
                return {referenceToFunc.getType(), referenceToFunc, TypeProvided::No};
            },
            genContext);

        return mlir::success();
    }    

    mlir::LogicalResult MLIRGenImpl::mlirGenFunctionEntry(mlir::Location location, FunctionPrototypeDOM::TypePtr funcProto,
                                             const GenContext &genContext)
    {
        return mlirGenFunctionEntry(location, funcProto->getReturnType(), genContext);
    }

    mlir::LogicalResult MLIRGenImpl::mlirGenFunctionEntry(mlir::Location location, mlir::Type retType, const GenContext &genContext)
    {
        auto hasReturn = retType && !isa<mlir_ts::VoidType>(retType);
        if (hasReturn)
        {
            auto entryOp = builder.create<mlir_ts::EntryOp>(location, mlir_ts::RefType::get(retType));
            auto varDecl = std::make_shared<VariableDeclarationDOM>(RETURN_VARIABLE_NAME, retType, location);
            varDecl->setReadWriteAccess();
            DECLARE(varDecl, entryOp.getReference());
        }
        else
        {
            builder.create<mlir_ts::EntryOp>(location, mlir::Type());
        }

        return mlir::success();
    }

    mlir::LogicalResult MLIRGenImpl::mlirGenFunctionExit(mlir::Location location, const GenContext &genContext)
    {
        mlir_ts::FuncOp contextFuncOp = genContext.funcOp;
        auto callableResult = contextFuncOp.getCallableResults();
        auto retType = callableResult.size() > 0 ? callableResult.front() : mlir::Type();
        auto hasReturn = retType && !isa<mlir_ts::VoidType>(retType);
        if (hasReturn)
        {
            auto retVarInfo = symbolTable.lookup(RETURN_VARIABLE_NAME);
            if (!retVarInfo.second)
            {
                if (genContext.allowPartialResolve)
                {
                    return mlir::success();
                }

                emitError(location) << "can't find return variable";
                return mlir::failure();
            }

            builder.create<mlir_ts::ExitOp>(location, retVarInfo.first);
        }
        else
        {
            builder.create<mlir_ts::ExitOp>(location, mlir::Value());
        }

        return mlir::success();
    }

    mlir::LogicalResult MLIRGenImpl::mlirGenFunctionCapturedParam(mlir::Location location, int &firstIndex,
                                                     FunctionPrototypeDOM::TypePtr funcProto,
                                                     mlir::Block::BlockArgListType arguments,
                                                     const GenContext &genContext)
    {
        if (genContext.capturedVars == nullptr)
        {
            return mlir::success();
        }

        auto isObjectType = genContext.thisType != nullptr && isa<mlir_ts::ObjectType>(genContext.thisType);
        if (isObjectType)
        {
            return mlir::success();
        }

        auto capturedParam = arguments[firstIndex++];
        auto capturedRefType = capturedParam.getType();

        auto capturedParamVar = std::make_shared<VariableDeclarationDOM>(CAPTURED_NAME, capturedRefType, location);

        DECLARE(capturedParamVar, capturedParam);

        return mlir::success();
    }

    mlir::LogicalResult MLIRGenImpl::mlirGenFunctionCapturedParamIfObject(mlir::Location location, int &firstIndex,
                                                             FunctionPrototypeDOM::TypePtr funcProto,
                                                             mlir::Block::BlockArgListType arguments,
                                                             const GenContext &genContext)
    {
        if (genContext.capturedVars == nullptr)
        {
            return mlir::success();
        }

        auto isObjectType = genContext.thisType != nullptr && isa<mlir_ts::ObjectType>(genContext.thisType);
        if (isObjectType)
        {

            auto thisVal = resolveIdentifier(location, THIS_NAME, genContext);

            LLVM_DEBUG(llvm::dbgs() << "\n!! this value: " << thisVal << "\n";);

            auto capturedNameResult =
                mlirGenPropertyAccessExpression(location, thisVal, MLIRHelper::TupleFieldName(CAPTURED_NAME, builder.getContext()), genContext);
            EXIT_IF_FAILED_OR_NO_VALUE(capturedNameResult)

            mlir::Value propValue = V(capturedNameResult);

            LLVM_DEBUG(llvm::dbgs() << "\n!! this->.captured value: " << propValue << "\n";);

            assert(propValue);

            // captured is in this->".captured"
            auto capturedParamVar = std::make_shared<VariableDeclarationDOM>(CAPTURED_NAME, propValue.getType(), location);
            DECLARE(capturedParamVar, propValue);
        }

        return mlir::success();
    }

    mlir::LogicalResult MLIRGenImpl::mlirGenFunctionParams(int firstIndex, FunctionPrototypeDOM::TypePtr funcProto,
                                              mlir::Block::BlockArgListType arguments, const GenContext &genContext)
    {
        for (auto [paramIndex, param] : enumerate(funcProto->getParams()))
        {
            auto index = firstIndex + (int)paramIndex;
            mlir::Value paramValue;

            // process init expression
            // we need reset scope for location as location of funcProto was created before correct scope
            auto location = locFuseWithScope(stripMetadata(param->getLoc()));

            LLVM_DEBUG(llvm::dbgs() << "Location for Param: " << location << "\n");

            // alloc all args
            // process optional parameters
            if (param->hasInitValue())
            {
                auto result = processOptionalParam(location, index, param->getType(), arguments[index], param->getInitValue(), genContext);
                EXIT_IF_FAILED_OR_NO_VALUE(result)
                paramValue = V(result);
            }
            else if (param->getIsOptional() && !isa<mlir_ts::OptionalType>(param->getType()))
            {
                auto optType = getOptionalType(param->getType());
                param->setType(optType);
                paramValue = builder.create<mlir_ts::ParamOp>(location, mlir_ts::RefType::get(optType),
                        arguments[index], builder.getBoolAttr(false), builder.getIndexAttr(index + 1));
            }
            else
            {
                paramValue = builder.create<mlir_ts::ParamOp>(location, mlir_ts::RefType::get(param->getType()),
                        arguments[index], builder.getBoolAttr(false), builder.getIndexAttr(index + 1));
            }

            if (paramValue)
            {
                // redefine variable
                param->setReadWriteAccess();
                DECLARE(param, paramValue);
            }
        }

        return mlir::success();
    }

    mlir::LogicalResult MLIRGenImpl::mlirGenFunctionParams(mlir::Location location, int firstIndex, mlir::Block::BlockArgListType arguments, const GenContext &genContext)
    {
        for (auto index = firstIndex; index < arguments.size(); index++)
        {
            std::string paramName("p");
            paramName += std::to_string(index - firstIndex);
            
            auto paramDecl = std::make_shared<VariableDeclarationDOM>(paramName, arguments[index].getType(), location);        
            DECLARE(paramDecl, arguments[index]);
        }

        return mlir::success();
    }    

    mlir::LogicalResult MLIRGenImpl::mlirGenFunctionParamsBindings(int firstIndex, FunctionPrototypeDOM::TypePtr funcProto,
                                                      mlir::Block::BlockArgListType arguments,
                                                      const GenContext &genContext)
    {
        for (const auto &param : funcProto->getParams())
        {
            if (auto bindingPattern = param->getBindingPattern())
            {
                auto location = loc(bindingPattern);
                auto val = resolveIdentifier(location, param->getName(), genContext);
                assert(val);
                auto initFunc = [&](mlir::Location, const GenContext &) { return std::make_tuple(val.getType(), val, TypeProvided::No); };

                if (bindingPattern == SyntaxKind::ArrayBindingPattern)
                {
                    auto arrayBindingPattern = bindingPattern.as<ArrayBindingPattern>();
                    if (mlir::failed(processDeclarationArrayBindingPattern(location, arrayBindingPattern, VariableType::Let,
                                                               initFunc, genContext)))
                    {
                        return mlir::failure();
                    }
                }
                else if (bindingPattern == SyntaxKind::ObjectBindingPattern)
                {
                    auto objectBindingPattern = bindingPattern.as<ObjectBindingPattern>();
                    if (mlir::failed(processDeclarationObjectBindingPattern(location, objectBindingPattern, VariableType::Let,
                                                                initFunc, genContext)))
                    {
                        return mlir::failure();
                    }
                }
            }
        }

        return mlir::success();
    }

    mlir::LogicalResult MLIRGenImpl::mlirGenFunctionCaptures(mlir::Location location, FunctionPrototypeDOM::TypePtr funcProto, const GenContext &genContext)
    {
        if (genContext.capturedVars == nullptr)
        {
            return mlir::success();
        }

        auto capturedVars = *genContext.capturedVars;

        NodeFactory nf(NodeFactoryFlags::None);

        // create variables
        for (auto &capturedVar : capturedVars)
        {
            auto varItem = capturedVar.getValue();
            auto variableInfo = varItem;
            auto name = variableInfo->getName();

            // load this.<var name>
            auto _captured = nf.createIdentifier(stows(CAPTURED_NAME));
            auto _name = nf.createIdentifier(stows(std::string(name)));
            auto _captured_name = nf.createPropertyAccessExpression(_captured, _name);
            auto result = mlirGen(_captured_name, genContext);
            EXIT_IF_FAILED_OR_NO_VALUE(result)
            auto capturedVarValue = V(result);

            auto capturedParam =
                std::make_shared<VariableDeclarationDOM>(name, variableInfo->getType(), variableInfo->getLoc());
            if (isa<mlir_ts::RefType>(capturedVarValue.getType()))
            {
                capturedParam->setReadWriteAccess();
            }

            LLVM_DEBUG(dbgs() << "\n!! captured '\".captured\"->" << name << "' [ " << capturedVarValue
                              << " ] captured type: " << capturedVarValue.getType() << "\n";);

            DECLARE(capturedParam, capturedVarValue);
        }

        return mlir::success();
    }

    mlir::LogicalResult MLIRGenImpl::mlirGenFunctionBody(FunctionLikeDeclarationBase functionLikeDeclarationBaseAST,
                                            StringRef name, mlir_ts::FuncOp funcOp, FunctionPrototypeDOM::TypePtr funcProto,
                                            const GenContext &genContext)
    {
        LLVM_DEBUG(llvm::dbgs() << "\n!! >>>> FUNCTION: '" << funcProto->getName() << "' ~~~ " << (genContext.dummyRun ? "dummy run" : "") <<  (genContext.allowPartialResolve ? " allowed partial resolve" : "") << "\n";);

        if (!functionLikeDeclarationBaseAST->body || declarationMode && !genContext.dummyRun)
        {
            // it is just declaration
            funcProto->setNoBody(true);
            return mlir::success();
        }

        SymbolTableScopeT varScope(symbolTable);
        BoundRefCacheScopeT boundRefCacheScope(boundRefMaterializedCache);

        auto location = loc(functionLikeDeclarationBaseAST);

        // Debug Info
        DITableScopeT debugFuncScope(debugScope);
        if (compileOptions.generateDebugInfo)
        {
            MLIRDebugInfoHelper mdi(builder, debugScope);
            auto locWithDI = 
                mdi.getSubprogram(
                    location, 
                    name,
                    funcOp.getName(), 
                    functionLikeDeclarationBaseAST->body 
                        ? loc(functionLikeDeclarationBaseAST->body) 
                        : location);

            LLVM_DEBUG(llvm::dbgs() << "Location of func: " << locWithDI << "\n");

            funcOp->setLoc(locWithDI);
        }

        // new location withing FunctionScope
        location = loc(functionLikeDeclarationBaseAST->body);

        GenContext funcGenContext(genContext);
        funcGenContext.funcOp = funcOp;

        auto *blockPtr = funcOp.addEntryBlock();
        auto &entryBlock = *blockPtr;

        builder.setInsertionPointToStart(&entryBlock);

        auto arguments = entryBlock.getArguments();
        auto firstIndex = 0;

        // add exit code
        if (failed(mlirGenFunctionEntry(location, funcProto, funcGenContext)))
        {
            return mlir::failure();
        }

        // register this if lambda function
        if (failed(mlirGenFunctionCapturedParam(location, firstIndex, funcProto, arguments, funcGenContext)))
        {
            return mlir::failure();
        }

        // allocate function parameters as variable
        if (failed(mlirGenFunctionParams(firstIndex, funcProto, arguments, funcGenContext)))
        {
            return mlir::failure();
        }

        if (failed(mlirGenFunctionParamsBindings(firstIndex, funcProto, arguments, funcGenContext)))
        {
            return mlir::failure();
        }

        if (failed(mlirGenFunctionCapturedParamIfObject(location, firstIndex, funcProto, arguments, funcGenContext)))
        {
            return mlir::failure();
        }

        if (failed(mlirGenFunctionCaptures(location, funcProto, funcGenContext)))
        {
            return mlir::failure();
        }

        // if we need params only we do not need to process body
        auto discoverParamsOnly = funcGenContext.allowPartialResolve && funcGenContext.discoverParamsOnly;
        if (!discoverParamsOnly)
        {
            // we need it to skip lexical block
            functionLikeDeclarationBaseAST->body->parent = functionLikeDeclarationBaseAST->body;
            if (failed(mlirGenBody(functionLikeDeclarationBaseAST->body, funcGenContext)))
            {
                return mlir::failure();
            }
        }

        // add exit code
        if (failed(mlirGenFunctionExit(location, funcGenContext)))
        {
            return mlir::failure();
        }

        if (funcGenContext.dummyRun && funcGenContext.cleanUps)
        {
            funcGenContext.cleanUps->push_back(blockPtr);
        }

        LLVM_DEBUG(llvm::dbgs() << "\n!! >>>> FUNCTION (SUCCESS END): '" << funcProto->getName() << "' ~~~ " << (funcGenContext.dummyRun ? "dummy run" : "") <<  (funcGenContext.allowPartialResolve ? " allowed partial resolve" : "") << "\n";);

        return mlir::success();
    }

    mlir::LogicalResult MLIRGenImpl::mlirGenFunctionBody(mlir::Location location, StringRef funcName, StringRef fullFuncName,
                                            mlir_ts::FunctionType funcType, std::function<mlir::LogicalResult(mlir::Location, const GenContext &)> funcBody,                                            
                                            const GenContext &genContext,
                                            int firstParam, bool isPublic)
    {
        if (theModule.lookupSymbol(fullFuncName))
        {
            return mlir::success();
        }

        LLVM_DEBUG(llvm::dbgs() << "\n!! >>>> SYNTH. FUNCTION: '" << fullFuncName << "' ~~~ " << (genContext.dummyRun ? "dummy run" : "") <<  (genContext.allowPartialResolve ? " allowed partial resolve" : "") << "\n";);

        SymbolTableScopeT varScope(symbolTable);
        BoundRefCacheScopeT boundRefCacheScope(boundRefMaterializedCache);

        SmallVector<mlir::NamedAttribute> attrs;
        processFunctionAttributes(attrs, genContext);

        auto funcOp = mlir_ts::FuncOp::create(location, fullFuncName, funcType, attrs);

        // Debug Info
        DITableScopeT debugFuncScope(debugScope);
        if (compileOptions.generateDebugInfo)
        {
            MLIRDebugInfoHelper mdi(builder, debugScope);
            auto locWithDI = 
                mdi.getSubprogram(
                    location, 
                    funcName,
                    fullFuncName, 
                    location);
            funcOp->setLoc(locWithDI);

            // new location withing FunctionScope
            location = locFuseWithScope(stripMetadata(location));
        }

        GenContext funcGenContext(genContext);
        funcGenContext.funcOp = funcOp;

        auto *blockPtr = funcOp.addEntryBlock();
        auto &entryBlock = *blockPtr;

        builder.setInsertionPointToStart(&entryBlock);

        auto arguments = entryBlock.getArguments();

        // add exit code
        if (failed(mlirGenFunctionEntry(location, mth.getReturnTypeFromFuncRef(funcType), funcGenContext)))
        {
            return mlir::failure();
        }

        if (failed(mlirGenFunctionParams(location, firstParam, arguments, funcGenContext)))
        {
            return mlir::failure();
        }

        if (failed(funcBody(location, funcGenContext)))
        {
            return mlir::failure();
        }

        // add exit code
        auto retVarInfo = symbolTable.lookup(RETURN_VARIABLE_NAME);
        if (retVarInfo.first)
        {
            builder.create<mlir_ts::ExitOp>(location, retVarInfo.first);
        }
        else
        {
            builder.create<mlir_ts::ExitOp>(location, mlir::Value());
        }

        if (genContext.dummyRun)
        {
            if (genContext.cleanUps)
            {
                genContext.cleanUps->push_back(blockPtr);
            }
        }
        else
        {
            theModule.push_back(funcOp);
        }

        if (isPublic)
        {
            funcOp.setPublic();
        }
        else
        {
            funcOp.setPrivate();
        }

        LLVM_DEBUG(llvm::dbgs() << "\n!! >>>> SYNTH. FUNCTION (SUCCESS END): '" << fullFuncName << "' ~~~ " << (genContext.dummyRun ? "dummy run" : "") <<  (genContext.allowPartialResolve ? " allowed partial resolve" : "") << "\n";);

        return mlir::success();
    }

    mlir::LogicalResult MLIRGenImpl::mlirGenResolveCapturedVars(mlir::Location location,
                                                   llvm::StringMap<ts::VariableDeclarationDOM::TypePtr> captureVars,
                                                   SmallVector<mlir::Value> &capturedValues,
                                                   const GenContext &genContext)
    {
        MLIRCodeLogic mcl(builder, compileOptions);
        for (auto &item : captureVars)
        {
            auto result = mlirGen(location, item.first(), genContext);
            auto varValue = V(result);
            if (!varValue)
            {
                return mlir::failure();
            }

            // review capturing by ref.  it should match storage type
            auto refValue = mcl.GetReferenceFromValue(location, varValue);
            if (refValue)
            {
                capturedValues.push_back(refValue);
                // set var as captures
                if (auto varOp = refValue.getDefiningOp<mlir_ts::VariableOp>())
                {
                    varOp.setCapturedAttr(builder.getBoolAttr(true));
                }
                else if (auto paramOp = refValue.getDefiningOp<mlir_ts::ParamOp>())
                {
                    paramOp.setCapturedAttr(builder.getBoolAttr(true));
                }
                else if (auto paramOptOp = refValue.getDefiningOp<mlir_ts::ParamOptionalOp>())
                {
                    paramOptOp.setCapturedAttr(builder.getBoolAttr(true));
                }
                else
                {
                    // TODO: review it.
                    // find out if u need to ensure that data is captured and belong to VariableOp or ParamOp with
                    // captured = true
                    LLVM_DEBUG(llvm::dbgs()
                                   << "\n!! var must be captured when loaded from other Op: " << refValue << "\n";);
                    // llvm_unreachable("variable must be captured.");
                }
            }
            else
            {
                // this is not ref, this is const value
                capturedValues.push_back(varValue);
            }
        }

        return mlir::success();
    }

    ValueOrLogicalResult MLIRGenImpl::mlirGenCreateCapture(mlir::Location location, mlir::Type capturedType,
                                              SmallVector<mlir::Value> capturedValues, const GenContext &genContext)
    {
        LLVM_DEBUG(for (auto &val : capturedValues) llvm::dbgs() << "\n!! captured val: " << val << "\n";);
        LLVM_DEBUG(llvm::dbgs() << "\n!! captured type: " << capturedType << "\n";);

        // add attributes to track which one sent by ref.
        auto captured = builder.create<mlir_ts::CaptureOp>(location, capturedType, capturedValues);
        return V(captured);
    }

    mlir::Value MLIRGenImpl::resolveFunctionWithCapture(mlir::Location location, StringRef name, mlir_ts::FunctionType funcType,
                                           mlir::Value thisValue, bool addGenericAttrFlag,
                                           const GenContext &genContext)
    {
        // check if required capture of vars
        auto captureVars = getCaptureVarsMap().find(name);
        if (captureVars != getCaptureVarsMap().end())
        {
            auto funcSymbolOp = builder.create<mlir_ts::SymbolRefOp>(
                location, funcType, mlir::FlatSymbolRefAttr::get(builder.getContext(), name));
            if (addGenericAttrFlag)
            {
                funcSymbolOp->setAttr(GENERIC_ATTR_NAME, mlir::BoolAttr::get(builder.getContext(), true));
            }

            LLVM_DEBUG(llvm::dbgs() << "\n!! func with capture: first type: [ " << funcType.getInput(0)
                                    << " ], \n\tfunc name: " << name << " \n\tfunc type: " << funcType << "\n");

            SmallVector<mlir::Value> capturedValues;
            if (mlir::failed(mlirGenResolveCapturedVars(location, captureVars->getValue(), capturedValues, genContext)))
            {
                return mlir::Value();
            }

            MLIRCodeLogic mcl(builder, compileOptions);

            auto captureType = mcl.CaptureType(captureVars->getValue());
            auto result = mlirGenCreateCapture(location, captureType, capturedValues, genContext);
            auto captured = V(result);
            return builder.create<mlir_ts::CreateBoundFunctionOp>(location, getBoundFunctionType(funcType), captured, funcSymbolOp);
        }

        if (thisValue)
        {
            auto thisFuncSymbolOp = builder.create<mlir_ts::ThisSymbolRefOp>(
                location, getBoundFunctionType(funcType), thisValue, mlir::FlatSymbolRefAttr::get(builder.getContext(), name));
            if (addGenericAttrFlag)
            {
                thisFuncSymbolOp->setAttr(GENERIC_ATTR_NAME, mlir::BoolAttr::get(builder.getContext(), true));
            }

            return V(thisFuncSymbolOp);
        }

        auto funcSymbolOp = builder.create<mlir_ts::SymbolRefOp>(
            location, funcType, mlir::FlatSymbolRefAttr::get(builder.getContext(), name));
        if (addGenericAttrFlag)
        {
            funcSymbolOp->setAttr(GENERIC_ATTR_NAME, mlir::BoolAttr::get(builder.getContext(), true));
        }

        return V(funcSymbolOp);
    }

} // namespace mlirgen
} // namespace typescript
