// Function declaration/prototype/body/capture code generation methods of MLIRGenImpl (see MLIRGenImpl.h).

#include "MLIRGenImpl.h"

#include "llvm/ADT/ScopeExit.h"

namespace typescript
{
namespace mlirgen
{

    // A `break` that leaves `statement` (the body of a loop or the clauses of a switch): an unlabeled one not
    // inside a nested loop or switch, or any labeled one - it may target this statement or one around it, and
    // either way control leaves. Nested functions and classes have breaks of their own.
    class BreakOutVisitor : public ts::FilterVisitorSkipFuncsAST<ts::BreakStatement>
    {
      public:
        bool found = false;
        bool labeledOnly;

        BreakOutVisitor(bool labeledOnly)
            : ts::FilterVisitorSkipFuncsAST<ts::BreakStatement>(SyntaxKind::BreakStatement, [&](ts::BreakStatement breakStatement) {
                  if (!this->labeledOnly || breakStatement->label)
                  {
                      found = true;
                  }
              }),
              labeledOnly(labeledOnly)
        {
        }

      protected:
        virtual bool isFiltered(ts::Node node) override
        {
            if (ts::FilterVisitorSkipFuncsAST<ts::BreakStatement>::isFiltered(node))
            {
                return true;
            }

            // inside a nested loop or switch only a labeled break can leave this statement
            if (!labeledOnly)
            {
                SyntaxKind kind = node;
                switch (kind)
                {
                case SyntaxKind::WhileStatement:
                case SyntaxKind::DoStatement:
                case SyntaxKind::ForStatement:
                case SyntaxKind::ForInStatement:
                case SyntaxKind::ForOfStatement:
                case SyntaxKind::SwitchStatement:
                    return true;
                default:
                    break;
                }
            }

            return false;
        }
    };

    static bool hasBreakOut(ts::Node statement)
    {
        if (statement == SyntaxKind::BreakStatement)
        {
            return true;
        }

        // unlabeled breaks of this statement
        BreakOutVisitor ownBreaks(false);
        ownBreaks.visit(statement);
        if (ownBreaks.found)
        {
            return true;
        }

        // labeled breaks anywhere in it
        BreakOutVisitor labeledBreaks(true);
        labeledBreaks.visit(statement);
        return labeledBreaks.found;
    }

    static bool isTrueCondition(ts::Expression expression)
    {
        return !expression || expression == SyntaxKind::TrueKeyword;
    }

    static bool canCompleteNormally(ts::Statement statement);

    static bool canCompleteNormally(ts::NodeArray<ts::Statement> statements)
    {
        for (auto statement : statements)
        {
            if (!canCompleteNormally(statement))
            {
                return false;
            }
        }

        return true;
    }

    // Whether control can reach the point after `statement`: after a function's body, that is falling off its end,
    // which returns undefined. Exact for return, throw, blocks, if/else, try/catch/finally, a switch with a
    // default, and `while (true)` / `for (;;)` / `do .. while (true)`; any other statement is taken to complete.
    static bool canCompleteNormally(ts::Statement statement)
    {
        if (!statement)
        {
            return true;
        }

        SyntaxKind kind = statement;
        switch (kind)
        {
        case SyntaxKind::ReturnStatement:
        case SyntaxKind::ThrowStatement:
            return false;
        case SyntaxKind::Block:
            return canCompleteNormally(statement.as<ts::Block>()->statements);
        case SyntaxKind::IfStatement:
        {
            auto ifStatement = statement.as<ts::IfStatement>();
            return !ifStatement->elseStatement || canCompleteNormally(ifStatement->thenStatement)
                || canCompleteNormally(ifStatement->elseStatement);
        }
        case SyntaxKind::TryStatement:
        {
            auto tryStatement = statement.as<ts::TryStatement>();
            if (tryStatement->finallyBlock && !canCompleteNormally(tryStatement->finallyBlock.as<ts::Statement>()))
            {
                return false;
            }

            return canCompleteNormally(tryStatement->tryBlock.as<ts::Statement>())
                || (tryStatement->catchClause && canCompleteNormally(tryStatement->catchClause->block.as<ts::Statement>()));
        }
        case SyntaxKind::WhileStatement:
        {
            auto whileStatement = statement.as<ts::WhileStatement>();
            return !isTrueCondition(whileStatement->expression) || hasBreakOut(whileStatement->statement);
        }
        case SyntaxKind::DoStatement:
        {
            auto doStatement = statement.as<ts::DoStatement>();
            return !isTrueCondition(doStatement->expression) || hasBreakOut(doStatement->statement);
        }
        case SyntaxKind::ForStatement:
        {
            auto forStatement = statement.as<ts::ForStatement>();
            return !isTrueCondition(forStatement->condition) || hasBreakOut(forStatement->statement);
        }
        case SyntaxKind::SwitchStatement:
        {
            // a clause falls through to the next one, so only the end of the last clause - or a break - leaves it;
            // without a default no clause may run at all
            auto clauses = statement.as<ts::SwitchStatement>()->caseBlock->clauses;
            auto hasDefault = false;
            for (auto clause : clauses)
            {
                hasDefault |= clause == SyntaxKind::DefaultClause;
            }

            if (!hasDefault || clauses.size() == 0)
            {
                return true;
            }

            for (auto clause : clauses)
            {
                for (auto clauseStatement : clause->statements)
                {
                    if (hasBreakOut(clauseStatement))
                    {
                        return true;
                    }
                }
            }

            return canCompleteNormally(clauses[clauses.size() - 1]->statements);
        }
        default:
            return true;
        }
    }

    // a function body whose end is reachable: falling off it returns undefined
    static bool bodyFallsOffEnd(ts::Node body)
    {
        return body == SyntaxKind::Block && canCompleteNormally(body.as<ts::Statement>());
    }


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

            // a parameter's default value is evaluated here for its type, and may read `this` - in an arrow too
            // (`bar(f = () => this)`), which captures it: named, as an explicit `this:` is below ("can't resolve
            // name: this")
            if (genContext.thisType && isa<mlir_ts::ClassType>(genContext.thisType)
                && llvm::any_of(parametersContextAST->parameters, [](auto parameter) { return !!parameter->initializer; }))
            {
                auto location = loc(parametersContextAST);
                auto varDecl = std::make_shared<VariableDeclarationDOM>(THIS_NAME, genContext.thisType, location);
                auto typeRefVal = builder.create<mlir_ts::TypeRefOp>(location, genContext.thisType);
                declare(location, varDecl, typeRefVal, genContext);
            }
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
            if (MLIRHelper::hasDecorator(arg, WEAK_DECORATOR))
            {
                return {refuseWeakDecorator(arg), false, params};
            }

            auto namePtr = MLIRHelper::getName(arg->name, stringAllocator);
            if (namePtr.empty())
            {
                namePtr = getArgumentName(index);
            }

            // A written `this:` on a method only annotates the receiver the method already has; as a
            // parameter of its own it shifted every argument by one slot.
            if (namePtr == THIS_NAME && !params.empty() && params.front()->getName() == THIS_NAME)
            {
                continue;
            }

            // `this: void` says the function uses no `this`: no parameter at all ("'Void' can't be used as
            // parameter type")
            if (namePtr == THIS_NAME && arg->type && arg->type == SyntaxKind::VoidKeyword)
            {
                continue;
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

                // a C function's callback is a bare code pointer (CDECL_DECORATOR)
                if (auto hybridFuncType = dyn_cast_or_null<mlir_ts::HybridFunctionType>(type))
                {
                    if (MLIRHelper::hasDecorator(parametersContextAST, CDECL_DECORATOR))
                    {
                        type = mlir_ts::FunctionType::get(builder.getContext(), hybridFuncType.getInputs(),
                                                          hybridFuncType.getResults(), hybridFuncType.isVarArg());
                    }
                }
            }

            // special case, setup 'this' and type provided
            if (namePtr == THIS_NAME && type) 
            {
                // NOTE: upward mailbox: explicit this-parameter type must reach the prototype chain - see GenContext
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
                GenContext defaultGenContext(genContext);
                defaultGenContext.evaluatingParameterDefault = true;
                auto evalType = evaluate(initializer, defaultGenContext);
                if (evalType)
                {
                    evalType = mth.wideStorageType(evalType);

                    // a function given as the default (`f = () => this.n`) may be a closure, which a plain function
                    // type has no room for: held as a hybrid one, as a `let` given an arrow is (cast to the plain
                    // type, the closure lost its captures - "invalid cast" in lowering)
                    if (auto funcType = dyn_cast<mlir_ts::FunctionType>(evalType); funcType && !funcType.isCVarArg())
                    {
                        evalType = mlir_ts::HybridFunctionType::get(builder.getContext(), funcType);
                    }
                    // ...and so is the closure itself, as the default is evaluated where it captures (a bound function
                    // whose first input is its capture box): typed so, the parameter disagreed with the signature, and a
                    // plain function given for it was called with the box - "function signature mismatch" in wasm
                    else if (auto boundFuncType = dyn_cast<mlir_ts::BoundFunctionType>(evalType);
                             boundFuncType && boundFuncType.getInputs().size() > 0)
                    {
                        evalType = mlir_ts::HybridFunctionType::get(
                            builder.getContext(),
                            mlir_ts::FunctionType::get(builder.getContext(), boundFuncType.getInputs().drop_front(),
                                                       boundFuncType.getResults(), boundFuncType.isVarArg()));
                    }

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

            // an untyped rest parameter is an `any[]`, and a pattern over it (`...[a, b]`) reads that
            if (mth.isNoneType(type) && isBindingPattern && isMultiArgs && !initializer)
            {
                type = getArrayType(getAnyType());
            }

            // in case of binding
            if (mth.isNoneType(type) && isBindingPattern)
            {
                type = mlirGenParameterObjectOrArrayBinding(arg->name, genContext);
                LLVM_DEBUG(dbgs() << "\n!! binding param " << namePtr << " is type " << type << "\n");
            }

            // a set accessor's value without a type is of the type its get accessor returns
            if (mth.isNoneType(type) && kind == SyntaxKind::SetAccessor && !typeParameter && !initializer)
            {
                auto [result, getAccessorType] = getPairedGetAccessorType(parametersContextAST, genContext);
                if (mlir::failed(result))
                {
                    return {mlir::failure(), isGenericTypes, params};
                }

                if (getAccessorType)
                {
                    type = getAccessorType;
                    isGenericTypes |= mth.isGenericType(type);
                }
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

            // `function assertNever(x: never)`: only a value narrowed to nothing can be given for it, so a
            // call that compiles is never reached. A `never` tslang computed (an intersection it can't
            // express) stays an error.
            if (isa<mlir_ts::NeverType>(type) && !(typeParameter && typeParameter == SyntaxKind::NeverKeyword))
            {
                emitError(location, "'Never' can't be used as parameter type");
                return {mlir::failure(), isGenericTypes, params};
            }

            // the arguments a rest parameter takes are put in an array: one of a number type only
            // failed the module's verification
            auto restBaseType = mth.getBaseType(type);
            if (isMultiArgs
                && (mth.isNumericType(restBaseType) || isa<mlir_ts::StringType>(restBaseType)
                    || isa<mlir_ts::BooleanType>(restBaseType) || isa<mlir_ts::BigIntType>(restBaseType)
                    || isa<mlir_ts::CharType>(restBaseType)))
            {
                emitError(loc(arg)) << "A rest parameter must be of an array type, not '" << to_print(type) << "'";
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

                // a later parameter's default may read this one (`function f(x: number, y = x * 2)`): its type is
                // evaluated here, so the name has to be known by then
                // ...and so may its type (`function f(a: number, b: typeof a)`)
                auto laterHasInitializer = std::any_of(formalParams.begin() + index + 1, formalParams.end(),
                                                       [](auto later) {
                                                           return !!later->initializer
                                                               || (later->type && later->type == SyntaxKind::TypeQuery);
                                                       });
                if (laterHasInitializer)
                {
                    auto varDecl = std::make_shared<VariableDeclarationDOM>(namePtr, type, loc(arg));
                    auto typeRefVal = builder.create<mlir_ts::TypeRefOp>(loc(arg), type);
                    declare(loc(arg), varDecl, typeRefVal, genContext);
                }
            }
        }

        return {mlir::success(), isGenericTypes, params};
    }

    // the type a class's set accessor takes when its parameter has none: what the get accessor of the
    // same name returns. No type when the class has no such get accessor; a failure while the get
    // accessor, declared after the set accessor, is not resolved yet (mlirGenClassMembers retries it).
    std::tuple<mlir::LogicalResult, mlir::Type> MLIRGenImpl::getPairedGetAccessorType(
        SignatureDeclarationBase setAccessorAST, const GenContext &genContext)
    {
        auto parent = setAccessorAST->parent;
        if (parent != SyntaxKind::ClassDeclaration && parent != SyntaxKind::ClassExpression)
        {
            return {mlir::success(), mlir::Type()};
        }

        auto [name, nameResult] = getNameForMethod(setAccessorAST, genContext);
        if (!nameResult || name.empty())
        {
            return {mlir::success(), mlir::Type()};
        }

        auto isStatic = hasModifier(setAccessorAST, SyntaxKind::StaticKeyword);
        FunctionLikeDeclarationBase getAccessorAST;
        for (auto &classMember : parent.as<ClassLikeDeclaration>()->members)
        {
            if (classMember != SyntaxKind::GetAccessor || hasModifier(classMember, SyntaxKind::StaticKeyword) != isStatic)
            {
                continue;
            }

            auto getAccessor = classMember.as<FunctionLikeDeclarationBase>();
            auto [getName, getNameResult] = getNameForMethod(getAccessor, genContext);
            if (getNameResult && getName == name)
            {
                getAccessorAST = getAccessor;
                break;
            }
        }

        if (!getAccessorAST)
        {
            return {mlir::success(), mlir::Type()};
        }

        if (getAccessorAST->type)
        {
            auto type = getType(getAccessorAST->type, genContext);
            return {type ? mlir::success() : mlir::failure(), type};
        }

        // the type the get accessor's body gives it, once the get accessor is registered
        if (genContext.thisClassType)
        {
            if (auto classInfo = getClassInfoByFullName(genContext.thisClassType.getName().getValue()))
            {
                auto accessorIndex = classInfo->getAccessorIndex(name);
                if (accessorIndex >= 0)
                {
                    auto &getFunc = classInfo->accessors[accessorIndex].get;
                    if (getFunc)
                    {
                        return {mlir::success(), getFunc.funcType.getNumResults() > 0
                                                     ? getFunc.funcType.getResult(0)
                                                     : mlir::Type()};
                    }
                }
            }
        }

        return {mlir::failure(), mlir::Type()};
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
        else if (signatureDeclarationBaseAST->parent == SyntaxKind::ObjectLiteralExpression
                 && isa_and_nonnull<mlir_ts::ObjectType>(genContext.thisType)
                 && isa<mlir_ts::ObjectStorageType>(mlir::cast<mlir_ts::ObjectType>(genContext.thisType).getStorageType()))
        {
            // (an object literal whose type is not made yet has no `this` type: it is named by the function, below)
            objectOwnerName = mlir::cast<mlir_ts::ObjectStorageType>(
                mlir::cast<mlir_ts::ObjectType>(genContext.thisType).getStorageType()).getName().getValue();
        }
        else if (genContext.funcOp)
        {
            mlir_ts::FuncOp funcOp = genContext.funcOp;
            objectOwnerName = funcOp.getSymName().str();
        }

        // A static method or accessor whose class has an instance one of the same name (TypeScript allows
        // `compare(other)` beside `static compare(a, b)`): the two are two functions, and the static one's
        // symbol says so. A static member with no such twin keeps its plain name, which a library built
        // before has.
        auto staticTwin = [&]() {
            if (!hasModifier(signatureDeclarationBaseAST, SyntaxKind::StaticKeyword)
                || (signatureDeclarationBaseAST->parent != SyntaxKind::ClassDeclaration
                    && signatureDeclarationBaseAST->parent != SyntaxKind::ClassExpression))
            {
                return false;
            }

            SyntaxKind kind = signatureDeclarationBaseAST;
            auto memberName = MLIRHelper::getName(signatureDeclarationBaseAST->name);
            for (auto member : signatureDeclarationBaseAST->parent.as<ClassLikeDeclaration>()->members)
            {
                if (member != kind || hasModifier(member, SyntaxKind::StaticKeyword) || !member->name)
                {
                    continue;
                }

                if (MLIRHelper::getName(member->name) == memberName)
                {
                    return true;
                }
            }

            return false;
        };
        auto staticPrefix = staticTwin() ? std::string(STATIC_NAME) + "_" : std::string();

        if (signatureDeclarationBaseAST == SyntaxKind::MethodDeclaration)
        {
            if (!objectOwnerName.empty())
            {
                // class method name
                name = objectOwnerName + "." + staticPrefix + name;
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
            name = objectOwnerName + "." + staticPrefix + "get_" + name;
        }
        else if (signatureDeclarationBaseAST == SyntaxKind::SetAccessor)
        {
            // class method name
            name = objectOwnerName + "." + staticPrefix + "set_" + name;
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
            && !funcProto->getIsGeneric();        // An instance of a generic function (countDown<string>) whose return type is declared: its type is known
        // before its body is, and a call of itself in the body (`return countDown(value, n - 1) + 1`) takes it
        // from here (instantiateSpecializedFunction). A function at the top level was registered by the pass
        // over the module's declarations before its body was discovered; an instance is made only when called.
        auto isGenericInstance =
            functionLikeDeclarationBaseAST->typeParameters.size() > 0 && !genContext.typeParamsWithArgs.empty();
        if (detectReturnType && isGenericInstance && funcType && !functionDiscovered)
        {
            getFunctionTypeMap().insert({fullName, funcType});
        }

        // A generator that names itself (`function* walk(n) { ... yield* walk(n - 1); }`, a method's
        // `yield* this.left.walk()`): its body is next() of the object its wrapper returns, and the variables of
        // that body are fields of the object, so they must all have their types before the wrapper has its own -
        // while a function is known by its name only once its type is found ("can't resolve name: walk", "Class
        // member 'walk' can't be found"). With a declared type (`: Generator<number>`) the wrapper's type is known
        // before its body is: it is known by its name, or as its class's method, while the body is discovered.
        // (Without one the type of the call is what is being found.)
        // The entry stays when the body can't be generated yet - it names another generator that is declared after
        // it, and names this one - so that the other finds this one, and this one is done on the next try. A
        // function's entry is the one it is registered with; a method's is taken out by mlirGenClassMethodMember,
        // which registers the method as what it is (GENERATOR_SELF_METHOD_ORDER marks it).
        if (detectReturnType
            && (functionLikeDeclarationBaseAST->internalFlags & InternalFlags::GeneratorWrapper) == InternalFlags::GeneratorWrapper
            && functionLikeDeclarationBaseAST->type && funcType && funcType.getNumResults() > 0
            && !mth.isNoneType(funcType.getResult(0)))
        {
            SyntaxKind wrapperKind = functionLikeDeclarationBaseAST;
            auto classType = genContext.thisType ? dyn_cast<mlir_ts::ClassType>(genContext.thisType) : mlir_ts::ClassType();
            if (wrapperKind == SyntaxKind::MethodDeclaration && classType)
            {
                auto classInfo = getClassInfoByFullName(classType.getName().getValue());
                auto methodName = MLIRHelper::getName(functionLikeDeclarationBaseAST->name);
                auto isStatic = hasModifier(functionLikeDeclarationBaseAST, SyntaxKind::StaticKeyword);
                if (classInfo && !methodName.empty() && classInfo->getMethodIndexExact(methodName, isStatic) < 0)
                {
                    classInfo->methods.push_back({methodName, funcType, fullName.str(), isStatic, false, false, -1,
                                                  GENERATOR_SELF_METHOD_ORDER, mlir_ts::AccessLevel::Public, std::string()});
                }
            }
            else if (wrapperKind == SyntaxKind::FunctionDeclaration && !getFunctionMap().count(funcProto->getNameWithoutNamespace()))
            {
                getFunctionMap().insert({funcProto->getNameWithoutNamespace(), FunctionEntry{fullName.str(), funcType, std::string()}});
            }
        }

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
                else if (funcProtoGenContext.overriddenReturnType
                    && funcProtoGenContext.overriddenReturnFor == functionLikeDeclarationBaseAST.operator->()
                    && mth.isNumericType(funcProto->getReturnType()))
                {
                    funcProto->setReturnType(funcProtoGenContext.overriddenReturnType);
                }
                else if (funcProtoGenContext.receiverFuncType)
                {
                    // rewrite ret type with actual value
                    auto &argTypeDestFuncType = funcProtoGenContext.receiverFuncType;
                    auto retTypeFromReceiver = mth.isAnyFunctionType(argTypeDestFuncType) 
                        ? mth.getReturnTypeFromFuncRef(argTypeDestFuncType)
                        : mlir::Type();
                    auto ownReturnType = funcProto->getReturnType();
                    if (ownReturnType
                        && !isa<mlir_ts::VoidType>(ownReturnType)
                        && mth.isAnyFunctionType(argTypeDestFuncType)
                        // a declared `void` only: a receiver with no result, or a null one, is one whose
                        // result is not known yet (`(i: T) => R` while R is inferred from this arrow,
                        // a builtin array method's callback)
                        && isa_and_nonnull<mlir_ts::VoidType>(retTypeFromReceiver)
                        && (functionLikeDeclarationBaseAST == SyntaxKind::ArrowFunction
                            || functionLikeDeclarationBaseAST == SyntaxKind::FunctionExpression))
                    {
                        // `(x: string) => void` given `n => n`: TS lets a callback whose result is
                        // ignored return one. Generated as returning nothing, its returns drop the value.
                        funcProto->setReturnType(mlir::Type());
                        funcProto->setDiscardsReturnValue(true);
                    }
                    else if (retTypeFromReceiver 
                        && !mth.isNoneType(retTypeFromReceiver) 
                        && !mth.isGenericType(retTypeFromReceiver))
                    {
                        funcProto->setReturnType(retTypeFromReceiver);
                        LLVM_DEBUG(llvm::dbgs()
                                       << "\n!! set return type from receiver: " << retTypeFromReceiver << "\n";);
                    }
                }

                getFunctionsDiscardingReturnValue()[fullName] = funcProto->getDiscardsReturnValue();

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

        if (!detectReturnType && getFunctionsDiscardingReturnValue().lookup(fullName))
        {
            funcProto->setDiscardsReturnValue(true);
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

        if (mlir::failed(checkLinkNameDecorators(location, functionLikeDeclarationBaseAST, genContext)))
        {
            return std::make_tuple(funcOp, funcProto, mlir::failure(), false);
        }

        SmallVector<mlir::NamedAttribute> attrs;
        auto dllExport = processFunctionAttributes(location, fullName, functionLikeDeclarationBaseAST, attrs, funcProtoGenContext,
            suppressExportForGenericInstantiation);

        // A C variadic function (`@varargs`) says so in its type, so a call through a value of
        // it - `let p = printf`, a `typeof printf` parameter - is a variadic call too. The type
        // comes here from several places (discovered, cached, declared), so it is set here.
        if (funcType && !funcType.isCVarArg() && llvm::any_of(attrs, [](mlir::NamedAttribute &attr) {
                return attr.getName() == "func.varargs";
            }))
        {
            funcType = mlir_ts::FunctionType::get(builder.getContext(), funcType.getInputs(), funcType.getResults(),
                                                  funcType.isVarArg(), /*isCVarArg=*/true);
        }

        if (mlir::failed(processOwnParamMarkers(location, functionLikeDeclarationBaseAST, funcProto->getParams(), funcType,
                                                attrs, funcProtoGenContext)))
        {
            return std::make_tuple(funcOp, funcProto, mlir::failure(), false);
        }

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
                    auto dllNameAttr = funcOp->getAttrOfType<mlir::StringAttr>(DLL_NAME);
                    addFunctionDeclarationToExport(funcProto, currentNamespace, dllNameAttr ? dllNameAttr.getValue() : StringRef(), funcOp);
                }
            }
        }

        // An arrow or a function expression met by the module's discovery pass (a dummy run of the top-level
        // statements outside any function) is not registered: a local of the module's code it reads
        // (`{ let x = 1; let a = () => x; }`) is not one there, so what it captures is not known yet. Registered,
        // it was not discovered again where it is generated, captured nothing, and read `x` across its isolated
        // region ("'ts.Load' op using value defined outside the region", TSLANG 105).
        // So is a function declared in a block of the module's code (`{ let x = 1; function foo() { x } }`,
        // `for (let x; ;) { function foo() { x } }`): it reads the block's locals as well.
        // (marked by mlirGen(Block): the parser leaves no parent link to tell)
        auto isNestedInModuleBlock = functionLikeDeclarationBaseAST == SyntaxKind::FunctionDeclaration
            && (functionLikeDeclarationBaseAST->internalFlags & InternalFlags::InModuleBlock) == InternalFlags::InModuleBlock;
        // And a method or an accessor of an object literal (`{ let x = 1; const o = { foo() { return x; } }; }`).
        auto memberParentKind = functionLikeDeclarationBaseAST->parent ? (SyntaxKind)functionLikeDeclarationBaseAST->parent
                                                                       : SyntaxKind::Unknown;
        auto isObjectLiteralMember = (functionLikeDeclarationBaseAST == SyntaxKind::MethodDeclaration
                                      || functionLikeDeclarationBaseAST == SyntaxKind::GetAccessor
                                      || functionLikeDeclarationBaseAST == SyntaxKind::SetAccessor)
            && memberParentKind == SyntaxKind::ObjectLiteralExpression;
        // ...and an arrow or a function expression met where an expression is only evaluated for its type (a
        // parameter's default value, `bar(f = () => this)`): what it captures there is not what it captures where it
        // is generated
        auto isEvaluatedExpression = genContext.evaluatingParameterDefault && genContext.funcOp
            && genContext.funcOp == tempFuncOp
            && (functionLikeDeclarationBaseAST == SyntaxKind::ArrowFunction
                || functionLikeDeclarationBaseAST == SyntaxKind::FunctionExpression);
        auto isModuleDiscoveryExpression = (genContext.dummyRun && !genContext.funcOp
            && (functionLikeDeclarationBaseAST == SyntaxKind::ArrowFunction
                || functionLikeDeclarationBaseAST == SyntaxKind::FunctionExpression
                || isNestedInModuleBlock
                || isObjectLiteralMember)) || isEvaluatedExpression;
        if (!funcProto->getIsGeneric() && !isModuleDiscoveryExpression)
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
            passResultData.iteratorResult =
                (functionLikeDeclarationBaseAST->internalFlags & InternalFlags::GeneratorNext) == InternalFlags::GeneratorNext;
            int discoverState = 1;

            GenContext genContextWithPassResult{};
            genContextWithPassResult.funcOp = dummyFuncOp;
            genContextWithPassResult.thisType = genContext.thisType;
            genContextWithPassResult.thisClassType = genContext.thisClassType;
            genContextWithPassResult.staticThis = genContext.staticThis;
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
            // a constructor of a class declared in a function: its field initializers
            // (generateConstructorStatements) are discovered with its body, so what they read of the
            // function is found as the body's reads are. This pass drains its copy; the real one
            // still has them. (A class at the top level has nothing of a function to read.)
            if (genContext.funcOp)
            {
                genContextWithPassResult.generatedStatements = genContext.generatedStatements;
            }

            registerNamespace(funcProto->getNameWithoutNamespace(), true);

            if (succeeded(mlirGenFunctionBody(functionLikeDeclarationBaseAST, name, dummyFuncOp, funcProto,
                                              genContextWithPassResult)))
            {
                exitNamespace();

                auto &passResult = genContextWithPassResult.passResult;

                // Root cause of the chained-sibling-method-return-gap's third layer (see
                // that memory): discovery always runs even when the function/method
                // already has an EXPLICIT return-type annotation - "// seems we need to
                // discover it all the time due to captured vars" above - so
                // functionReturnTypeShouldBeProvided/functionReturnType still get set from
                // trying to INFER a type from the body's return expression(s), even though
                // that inferred value will never be used when an explicit one already
                // exists. If a sibling method's prototype isn't registered yet (e.g.
                // `return this.scale(f) + e;` inside an object literal, discovered as a
                // side effect of discovering an unrelated OUTER function), the call can't
                // produce a value during this dummy run, so functionReturnType comes back
                // "none" - previously treated as "discovery failed to converge" even
                // though the explicit annotation makes that inferred value moot.
                auto hasExplicitReturnType = !!functionLikeDeclarationBaseAST->type;

                if (passResult->functionReturnTypeShouldBeProvided
                    && mth.isNoneType(passResult->functionReturnType)
                    && !hasExplicitReturnType)
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

                // a bare `return;` beside value returns, or an end of the body that is reachable: T | undefined
                if ((passResult->hasBareReturn || bodyFallsOffEnd(functionLikeDeclarationBaseAST->body))
                    && !hasExplicitReturnType && discoveredType && !mth.isNoneType(discoveredType)
                    && !isa<mlir_ts::VoidType>(discoveredType))
                {
                    discoveredType = getUnionType(loc(functionLikeDeclarationBaseAST), discoveredType, getUndefinedType());
                }
                // a "none" discoveredType means inference didn't converge (see above) -
                // never let that silently overwrite an already-known (explicit or
                // previously-cached) return type.
                if (discoveredType && !mth.isNoneType(discoveredType) && discoveredType != funcProto->getReturnType())
                {
                    // TODO: do we need to convert it here? maybe send it as const object?

                    funcProto->setReturnType(mth.convertConstArrayTypeToArrayType(discoveredType));
                    LLVM_DEBUG(llvm::dbgs()
                                   << "\n!! ret type: " << funcProto->getReturnType() << ", name: " << name << "\n";);
                }

                // a member of a class declared in a function, reading that function's locals: nothing passes
                // them to it - a call through the object, or `new`, has no captures to give - so it is an
                // error, not a call with an argument missing (docs/classes-as-values-design.md §3)
                auto kind = (SyntaxKind)functionLikeDeclarationBaseAST;
                if (passResult->outerVariables.size() > 0 && genContext.thisType &&
                    isa<mlir_ts::ClassType>(genContext.thisType) &&
                    (kind == SyntaxKind::MethodDeclaration || kind == SyntaxKind::Constructor ||
                     kind == SyntaxKind::GetAccessor || kind == SyntaxKind::SetAccessor))
                {
                    std::string captured = passResult->outerVariables.begin()->getKey().str();
                    auto classInfo = getClassInfoByFullName(mlir::cast<mlir_ts::ClassType>(genContext.thisType).getName().getValue());
                    std::string className = classInfo ? classInfo->name.str() : to_print(genContext.thisType);
                    emitError(loc(functionLikeDeclarationBaseAST))
                        << "a member of class '" << className.c_str() << "' reads '" << captured.c_str()
                        << "', a local of the function the class is declared in: a class can't capture yet";
                    return mlir::failure();
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
                    getCaptureUsesMap().insert({name, passResult->outerVariableUses});
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
        // Outside an ambient context (`declare`, a .d.ts) a function without a body is an overload
        // signature, and the implementation that follows it is the function. Registering the
        // signature too made the implementation take its parameter list.
        auto isAmbient = (functionDeclarationAST->flags & NodeFlags::Ambient) == NodeFlags::Ambient
            || hasModifier(functionDeclarationAST, SyntaxKind::DeclareKeyword);
        if (!functionDeclarationAST->body && !isAmbient)
        {
            return mlir::success();
        }

        // A declared function binds one external symbol, and that has one signature: a second
        // declaration of it with another was "redefinition of symbol" from the verifier. The same
        // declaration again - one C function in two .d.ts files - is left to the first.
        if (!functionDeclarationAST->body)
        {
            auto [fullName, name] = getNameOfFunction(functionDeclarationAST, genContext);
            auto declared = declaredFunctions.find(fullName);
            if (declared == declaredFunctions.end() || declared->second.second != theModule.getOperation())
            {
                declaredFunctions[fullName] = {functionDeclarationAST, theModule.getOperation()};
                if (fullName == name)
                {
                    rootDeclaredFunctions.insert(fullName);
                }
            }
            else if (declared->second.first != functionDeclarationAST)
            {
                if (print(declared->second.first) == print(functionDeclarationAST))
                {
                    return mlir::success();
                }

                emitError(loc(functionDeclarationAST))
                    << "'" << name << "' is declared again with another signature: a declared function is one external "
                    << "symbol, so it can't have overloads. Declare it once, with optional or union parameters, or "
                    << "give each signature a name of its own and bind them with @linkname";
                return mlir::failure();
            }
        }

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

    // return(value: T) and throw(e) of a generator typed Generator<T, ...>, as JavaScript's: a generator paused at a
    // yield resumes there - through next() - finishing (`return`, so its finally blocks run) or throwing e (which
    // its own try/catch may catch, and then it goes on to its next yield); one not started yet, or finished, does
    // not run: return(value) gives `{ value, done: true }`, throw(e) throws e. Either finishes the generator
    // (step -1), unless its body caught the throw. GENERATOR_MODE_NAME tells the yield how to resume
    // (mlirGenGeneratorResumeChecks). A finally block that yields on the way out pauses the generator again:
    // return() gives what it yields, and the value is the generator's result when the finally is done.
    void MLIRGenImpl::addGeneratorResumeMethods(NodeFactory &nf, FunctionLikeDeclarationBase functionLikeDeclarationBaseAST,
                                                TypeNode yieldType, NodeArray<ObjectLiteralElementLike> &generatorObjectProperties)
    {
        // the yields of the body - not of a function inside it - resume checking the mode
        // (a yield* hands it on to the yields its loop makes, mlirGenYieldStar)
        FilterVisitorSkipFuncsAST<YieldExpression> yieldsVisitor(SyntaxKind::YieldExpression, [&](auto yieldExpression) {
            yieldExpression->internalFlags |= InternalFlags::GeneratorResumable;
        });

        if (functionLikeDeclarationBaseAST->body)
        {
            yieldsVisitor.visit(functionLikeDeclarationBaseAST->body);
        }

        auto thisProp = [&](const char *name) {
            return nf.createPropertyAccessExpression(nf.createToken(SyntaxKind::ThisKeyword), nf.createIdentifier(stows(std::string(name))));
        };
        auto number = [&](const char *value) { return nf.createNumericLiteral(stows(std::string(value)), TokenFlags::None); };
        auto assign = [&](Expression target, Expression value) {
            return nf.createExpressionStatement(
                nf.createBinaryExpression(target, nf.createToken(SyntaxKind::EqualsToken), value));
        };
        auto finished = [&]() {
            return assign(thisProp(GENERATOR_STEP),
                          nf.createPrefixUnaryExpression(SyntaxKind::MinusToken, number("1")));
        };
        auto paused = [&]() {
            return nf.createBinaryExpression(thisProp(GENERATOR_STEP), nf.createToken(SyntaxKind::GreaterThanToken),
                                             number("0"));
        };
        auto resultType = [&]() {
            NodeArray<TypeElement> members;
            members.push_back(nf.createPropertySignature(undefined, nf.createIdentifier(S("value")), undefined, yieldType));
            members.push_back(nf.createPropertySignature(undefined, nf.createIdentifier(S("done")), undefined,
                                                         nf.createToken(SyntaxKind::BooleanKeyword).as<TypeNode>()));
            return nf.createTypeLiteralNode(members);
        };
        auto anyType = [&]() { return nf.createToken(SyntaxKind::AnyKeyword).as<TypeNode>(); };
        auto callNext = [&]() {
            return nf.createCallExpression(thisProp(ITERATOR_NEXT), undefined, NodeArray<Expression>());
        };

        // the mode, and what throw() was given
        generatorObjectProperties.push_back(
            nf.createPropertyAssignment(nf.createIdentifier(S(GENERATOR_MODE_NAME)), number("0")));
        // (a number in it, not `<any>undefined`: under -mm=own an object literal's field of `<any>undefined` or
        // `<any>null` was released twice - its box stored into the object and given back at the end as well, a
        // double free wasm's heap caught; TSLANG 153)
        generatorObjectProperties.push_back(nf.createPropertyAssignment(
            nf.createIdentifier(S(GENERATOR_THROWN_NAME)),
            nf.createTypeAssertion(anyType(), number("0"))));

        auto addMethod = [&](const char *name, NodeArray<ParameterDeclaration> parameters, NodeArray<Statement> statements) {
            auto method = nf.createMethodDeclaration(undefined, undefined, nf.createIdentifier(stows(std::string(name))), undefined,
                                                     undefined, parameters, resultType(),
                                                     nf.createBlock(statements, /*multiLine*/ false));
            method->pos = functionLikeDeclarationBaseAST->pos;
            method->_end = functionLikeDeclarationBaseAST->_end;
            generatorObjectProperties.push_back(method);
        };

        // return(value: T) {
        //     if (this.step > 0) { this.mode = 2; this.thrown = <any>value; return this.next(); }
        //     this.step = -1;
        //     return { value: value, done: true };
        // }
        // Under -mm=own only for a T that owns nothing (a number, a boolean): the result would hold the value the
        // caller lends ("an object literal that holds a value it owns is moved only into a local ..."), as next(v)
        // keeps one; the generator then has no return() (the library declares it optional)
        SyntaxKind yieldKind = yieldType;
        auto ownsNothing = yieldKind == SyntaxKind::NumberKeyword || yieldKind == SyntaxKind::BooleanKeyword
                           || yieldKind == SyntaxKind::BigIntKeyword;
        if (compileOptions.memoryModel != MemoryModelOwn || ownsNothing)
        {
            NodeArray<ParameterDeclaration> parameters;
            parameters.push_back(
                nf.createParameterDeclaration(undefined, undefined, nf.createIdentifier(S("value")), undefined, yieldType));

            NodeArray<Statement> resume;
            resume.push_back(assign(thisProp(GENERATOR_MODE_NAME), number("2")));
            // the value the generator finishes with (the resume point's `return this.thrown`), a yield of a finally
            // on the way out given back as the result
            resume.push_back(assign(thisProp(GENERATOR_THROWN_NAME), nf.createTypeAssertion(anyType(), nf.createIdentifier(S("value")))));
            resume.push_back(nf.createReturnStatement(callNext()));

            NodeArray<ObjectLiteralElementLike> resultProperties;
            resultProperties.push_back(nf.createPropertyAssignment(nf.createIdentifier(S("value")), nf.createIdentifier(S("value"))));
            resultProperties.push_back(nf.createPropertyAssignment(nf.createIdentifier(S("done")), nf.createToken(SyntaxKind::TrueKeyword)));

            NodeArray<Statement> statements;
            statements.push_back(nf.createIfStatement(paused(), nf.createBlock(resume, false), undefined));
            statements.push_back(finished());
            statements.push_back(nf.createReturnStatement(nf.createObjectLiteralExpression(resultProperties, false)));
            addMethod("return", parameters, statements);
        }

        // throw(e: any) {
        //     if (this.step > 0) {
        //         this.mode = 1; this.thrown = e;
        //         try { return this.next(); } catch (x) { this.step = -1; throw x; }
        //     }
        //     this.step = -1;
        //     throw e;
        // }
        {
            NodeArray<ParameterDeclaration> parameters;
            parameters.push_back(
                nf.createParameterDeclaration(undefined, undefined, nf.createIdentifier(S("e")), undefined, anyType()));

            NodeArray<Statement> tryStatements;
            tryStatements.push_back(nf.createReturnStatement(callNext()));

            NodeArray<Statement> catchStatements;
            catchStatements.push_back(finished());
            catchStatements.push_back(nf.createThrowStatement(nf.createIdentifier(S("x"))));
            auto catchClause = nf.createCatchClause(
                nf.createVariableDeclaration(nf.createIdentifier(S("x")), undefined, undefined, undefined),
                nf.createBlock(catchStatements, false));

            NodeArray<Statement> resume;
            resume.push_back(assign(thisProp(GENERATOR_MODE_NAME), number("1")));
            resume.push_back(assign(thisProp(GENERATOR_THROWN_NAME), nf.createIdentifier(S("e"))));
            resume.push_back(nf.createTryStatement(nf.createBlock(tryStatements, false), catchClause, undefined));

            NodeArray<Statement> statements;
            statements.push_back(nf.createIfStatement(paused(), nf.createBlock(resume, false), undefined));
            statements.push_back(finished());
            statements.push_back(nf.createThrowStatement(nf.createIdentifier(S("e"))));
            addMethod("throw", parameters, statements);
        }
    }

    // Where a yield of a generator typed Generator<...> resumes (GeneratorResumable): throw() asked it to throw
    // there, return() to finish there (addGeneratorResumeMethods).
    //     if (this.mode == 1) { this.mode = 0; throw this.thrown; }
    //     if (this.mode == 2) { this.mode = 0; return; }
    mlir::LogicalResult MLIRGenImpl::mlirGenGeneratorResumeChecks(mlir::Location location, const GenContext &genContext)
    {
        NodeFactory nf(NodeFactoryFlags::None);
        auto thisProp = [&](const char *name) {
            return nf.createPropertyAccessExpression(nf.createToken(SyntaxKind::ThisKeyword), nf.createIdentifier(stows(std::string(name))));
        };
        auto modeIs = [&](const char *value) {
            return nf.createBinaryExpression(thisProp(GENERATOR_MODE_NAME), nf.createToken(SyntaxKind::EqualsEqualsToken),
                                             nf.createNumericLiteral(stows(std::string(value)), TokenFlags::None));
        };
        auto resetMode = [&]() {
            return nf.createExpressionStatement(nf.createBinaryExpression(
                thisProp(GENERATOR_MODE_NAME), nf.createToken(SyntaxKind::EqualsToken),
                nf.createNumericLiteral(S("0"), TokenFlags::None)));
        };

        NodeArray<Statement> throwStatements;
        throwStatements.push_back(resetMode());
        throwStatements.push_back(nf.createThrowStatement(thisProp(GENERATOR_THROWN_NAME)));

        // with the value return() was given (kept in `.thrown`)
        auto finish = nf.createReturnStatement(thisProp(GENERATOR_THROWN_NAME));
        finish->internalFlags |= InternalFlags::GeneratorReturn;
        NodeArray<Statement> returnStatements;
        returnStatements.push_back(resetMode());
        // at the yield of a yield*: the delegate's return() first, with the same value (mlirGenYieldStar)
        if (genContext.yieldStarDelegate && compileOptions.memoryModel == MemoryModelOwn && genContext.typeAliasMap.count(".YS_G"))
        {
            // -mm=own: the delegate is exchanged into a local of next() for the call and back, as for its
            // next() (mlirGenYieldStar): `{ let .yq: G | undefined = undefined; [.yq, .yd] = [.yd, .yq];
            // try { .yq!.return?.(this.thrown); } finally { [.yq, .yd] = [.yd, .yq]; } }`
            auto delegateIdent = nf.createIdentifier(stows(std::string(genContext.yieldStarDelegate)));
            auto localIdent = nf.createIdentifier(stows(".yq" + std::string(genContext.yieldStarDelegate)));
            auto exchange = [&]() {
                NodeArray<Expression> targets;
                targets.push_back(localIdent);
                targets.push_back(delegateIdent);
                NodeArray<Expression> values;
                values.push_back(delegateIdent);
                values.push_back(localIdent);
                return nf.createExpressionStatement(nf.createBinaryExpression(
                           nf.createArrayLiteralExpression(targets, false), nf.createToken(SyntaxKind::EqualsToken),
                           nf.createArrayLiteralExpression(values, false))).as<Statement>();
            };

            NodeArray<TypeNode> members;
            members.push_back(nf.createTypeReferenceNode(nf.createIdentifier(S(".YS_G")).as<Node>()).as<TypeNode>());
            members.push_back(nf.createToken(SyntaxKind::UndefinedKeyword).as<TypeNode>());
            auto localDeclaration = nf.createVariableDeclaration(localIdent, undefined, nf.createUnionTypeNode(members).as<TypeNode>(),
                                                                 nf.createIdentifier(S(UNDEFINED_NAME)));
            localDeclaration->internalFlags |= InternalFlags::StackLocal;
            NodeArray<VariableDeclaration> localDeclarations;
            localDeclarations.push_back(localDeclaration);

            NodeArray<Expression> arguments;
            arguments.push_back(thisProp(GENERATOR_THROWN_NAME));
            NodeArray<Statement> callStatements;
            callStatements.push_back(nf.createExpressionStatement(nf.createCallChain(
                nf.createPropertyAccessExpression(nf.createNonNullExpression(localIdent), nf.createIdentifier(S("return"))),
                nf.createToken(SyntaxKind::QuestionDotToken), undefined, arguments)));
            NodeArray<Statement> backStatements;
            backStatements.push_back(exchange());

            NodeArray<Statement> delegateStatements;
            delegateStatements.push_back(
                nf.createVariableStatement(undefined, nf.createVariableDeclarationList(localDeclarations, NodeFlags::Let)));
            delegateStatements.push_back(exchange());
            delegateStatements.push_back(nf.createTryStatement(nf.createBlock(callStatements, false), undefined,
                                                               nf.createBlock(backStatements, false)));
            returnStatements.push_back(nf.createBlock(delegateStatements, false));
        }
        else if (genContext.yieldStarDelegate)
        {
            NodeArray<Expression> arguments;
            arguments.push_back(thisProp(GENERATOR_THROWN_NAME));
            // `.yd.return?.(this.thrown)`
            auto delegateReturn = nf.createCallChain(
                nf.createPropertyAccessExpression(nf.createIdentifier(stows(std::string(genContext.yieldStarDelegate))),
                                                  nf.createIdentifier(S("return"))),
                nf.createToken(SyntaxKind::QuestionDotToken), undefined, arguments);
            returnStatements.push_back(nf.createExpressionStatement(delegateReturn));
        }

        returnStatements.push_back(finish);

        LocationHelper lh(builder.getContext());
        auto [pos, _end] = lh.getSpan(location);
        for (auto [mode, statements] : {std::make_pair("1", throwStatements), std::make_pair("2", returnStatements)})
        {
            auto ifStatement = nf.createIfStatement(modeIs(mode), nf.createBlock(statements, false), undefined);
            ifStatement->pos = pos;
            ifStatement->_end = _end;
            if (mlir::failed(mlirGen(ifStatement, genContext)))
            {
                return mlir::failure();
            }
        }

        return mlir::success();
    }

    FunctionLikeDeclarationBase MLIRGenImpl::buildGeneratorWrapperDeclaration(
        FunctionLikeDeclarationBase functionLikeDeclarationBaseAST, mlir::Location location)
    {
        auto fixThisReference = functionLikeDeclarationBaseAST == SyntaxKind::MethodDeclaration;
        auto thisIsParameter = false;
        if (functionLikeDeclarationBaseAST->parameters.size() > 0)
        {
            auto nameNode = functionLikeDeclarationBaseAST->parameters.front()->name;
            if (nameNode == SyntaxKind::Identifier)
            {
                auto ident = nameNode.as<Identifier>();
                if (ident->escapedText == S(THIS_NAME))
                {
                    fixThisReference = true;
                    thisIsParameter = true;
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

        // a finished generator - `return` sets step to -1 - gives {value: undefined, done: true} from then on
        auto finishedCheck = nf.createIfStatement(
            nf.createBinaryExpression(stepAccess, nf.createToken(SyntaxKind::LessThanToken), nf.createNumericLiteral(S("0"), TokenFlags::None)),
            nf.createReturnStatement(getYieldReturnObject(nf, location, nf.createIdentifier(S(UNDEFINED_NAME)), true)),
            undefined);
        nextStatements.push_back(finishedCheck);

        // call stateswitch
        auto callStat = nf.createExpressionStatement(
            nf.createCallExpression(nf.createIdentifier(S(GENERATOR_SWITCHSTATE)), undefined, {stepAccess}));

        nextStatements.push_back(callStat);

        // the statements of next() that use the generator object's own `this`, not the body's
        auto ownStatementsCount = nextStatements.size();

        // Generator<T, TReturn> with a TReturn of its own (not T, void, undefined or any): a returned value has no
        // place in next()'s `{ value: T }` (see nextResultType below)
        auto returnTypeOfItsOwn = false;
        if (auto declaredType = functionLikeDeclarationBaseAST->type; declaredType && declaredType == SyntaxKind::TypeReference)
        {
            auto typeReference = declaredType.as<TypeReferenceNode>();
            if (typeReference->typeName == SyntaxKind::Identifier &&
                MLIRHelper::getName(typeReference->typeName.as<Identifier>()) == "Generator" &&
                typeReference->typeArguments.size() >= 2)
            {
                auto yieldType = typeReference->typeArguments[0];
                auto returnType = typeReference->typeArguments[1];
                SyntaxKind returnKind = returnType;
                returnTypeOfItsOwn = returnKind != SyntaxKind::VoidKeyword && returnKind != SyntaxKind::UndefinedKeyword &&
                                     returnKind != SyntaxKind::AnyKeyword && print(returnType) != print(yieldType);
            }
        }

        // a yield in a catch clause (YieldInCatch) or a finally block (YieldInFinally) can't be resumed: said at the
        // yield
        FilterVisitorSkipFuncsAST<TryStatement> trysVisitor(SyntaxKind::TryStatement, [&](auto tryStatement) {
            if (tryStatement->catchClause && tryStatement->catchClause->block)
            {
                FilterVisitorSkipFuncsAST<YieldExpression> yieldsInCatch(SyntaxKind::YieldExpression, [&](auto yieldExpression) {
                    yieldExpression->internalFlags |= InternalFlags::YieldInCatch;
                });
                yieldsInCatch.visit(tryStatement->catchClause->block);
            }

            if (tryStatement->finallyBlock)
            {
                FilterVisitorSkipFuncsAST<YieldExpression> yieldsInFinally(SyntaxKind::YieldExpression, [&](auto yieldExpression) {
                    yieldExpression->internalFlags |= InternalFlags::YieldInFinally;
                    tryStatement->internalFlags |= InternalFlags::YieldInFinally;
                });
                yieldsInFinally.visit(tryStatement->finallyBlock);
            }
        });

        if (functionLikeDeclarationBaseAST->body)
        {
            trysVisitor.visit(functionLikeDeclarationBaseAST->body);
        }

        // a `return` of the body finishes the generator; one of a nested function is that function's own
        FilterVisitorSkipFuncsAST<ReturnStatement> returnsVisitor(SyntaxKind::ReturnStatement, [&](auto returnStatement) {
            returnStatement->internalFlags |= InternalFlags::GeneratorReturn;
            if (returnTypeOfItsOwn && returnStatement->expression)
            {
                returnStatement->internalFlags |= InternalFlags::GeneratorReturnOfOtherType;
            }
        });

        if (functionLikeDeclarationBaseAST->body)
        {
            returnsVisitor.visit(functionLikeDeclarationBaseAST->body);
        }

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

        // `function* g(): Generator<number>` (Iterator<T>, Iterable<T>, IterableIterator<T>): next() returns
        // `{ value: T; done: boolean }`, so what is yielded is given T as its type (`yield 1` a number, not the
        // s32 the literal is), as TypeScript types it; without it the yields' own type was the element's, and
        // `yield 1` could not be a Generator<number> ("can't assign '() => {value:s32, ...}'")
        TypeNode nextResultType = undefined;
        NodeArray<ParameterDeclaration> nextParameters;
        // Generator<T, ...>: return(value: T) and throw(e) besides next() (generatorResumeMethods)
        TypeNode resumableYieldType = undefined;
        if (auto declaredType = functionLikeDeclarationBaseAST->type;
            declaredType && declaredType == SyntaxKind::TypeReference && !hasModifier(functionLikeDeclarationBaseAST, SyntaxKind::AsyncKeyword))
        {
            auto typeReference = declaredType.as<TypeReferenceNode>();
            if (typeReference->typeName == SyntaxKind::Identifier && typeReference->typeArguments.size() >= 1)
            {
                auto typeName = MLIRHelper::getName(typeReference->typeName.as<Identifier>());
                if (typeName == "Generator" || typeName == "Iterator" || typeName == "Iterable" ||
                    typeName == "IterableIterator")
                {
                    NodeArray<TypeElement> members;
                    members.push_back(nf.createPropertySignature(undefined, nf.createIdentifier(S("value")), undefined,
                                                                 typeReference->typeArguments.front()));
                    members.push_back(nf.createPropertySignature(undefined, nf.createIdentifier(S("done")), undefined,
                                                                 nf.createToken(SyntaxKind::BooleanKeyword).as<TypeNode>()));
                    nextResultType = nf.createTypeLiteralNode(members);

                    if (typeName == "Generator")
                    {
                        resumableYieldType = typeReference->typeArguments.front();
                    }

                    // Generator<T, TReturn, TNext>: next(value?: TNext), the value what the resumed `yield` is
                    if (typeName == "Generator" && typeReference->typeArguments.size() >= 3)
                    {
                        nextParameters.push_back(nf.createParameterDeclaration(
                            undefined, undefined, nf.createIdentifier(S(GENERATOR_SENT_NAME)),
                            nf.createToken(SyntaxKind::QuestionToken), typeReference->typeArguments[2]));
                    }
                }
            }
        }

        // create method next in object
        auto nextMethodDecl =
            nf.createMethodDeclaration(undefined, undefined, nf.createIdentifier(S(ITERATOR_NEXT)), undefined,
                                       undefined, nextParameters, nextResultType, nextBody);
        nextMethodDecl->internalFlags |= InternalFlags::VarsInObjectContext | InternalFlags::GeneratorNext;

        // copy location info, to fix issue with names of anonymous functions
        nextMethodDecl->pos = functionLikeDeclarationBaseAST->pos;
        nextMethodDecl->_end = functionLikeDeclarationBaseAST->_end;

        if (fixThisReference)
        {
            FilterVisitorSkipFuncsAST<Node> visitor(SyntaxKind::ThisKeyword, [&](auto thisNode) {
                thisNode->internalFlags |= InternalFlags::ThisArgAlias;
            });

            for (auto it = begin(nextStatements) + ownStatementsCount; it != end(nextStatements); ++it)
            {
                visitor.visit(*it);
            }
        }

        generatorObjectProperties.push_back(nextMethodDecl);

        if (resumableYieldType)
        {
            addGeneratorResumeMethods(nf, functionLikeDeclarationBaseAST, resumableYieldType, generatorObjectProperties);
        }

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
            auto _thisArgDeclaration = nf.createVariableDeclaration(_thisArg, undefined, undefined, nf.createToken(SyntaxKind::ThisKeyword));
            // a `this` parameter (`function* keys<T>(this: T[])`): the alias is the parameter's own
            // storage, so the generator captures the parameter, as it does any other - under
            // -mm=own its state borrows it (spec 19) rather than owning a copy of it
            if (thisIsParameter)
            {
                _thisArgDeclaration->internalFlags |= InternalFlags::ForceConstRef;
            }

            _thisArgDeclarations.push_back(_thisArgDeclaration);
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

            // what the class gave the method (mlirGenClassMethodMember): exported, imported, public.
            // The wrapper is the method's function; without them a public generator method of an
            // exported class was private, and an importer calling it by name found no symbol.
            methodOp->internalFlags |= functionLikeDeclarationBaseAST->internalFlags &
                                       (InternalFlags::DllExport | InternalFlags::DllImport | InternalFlags::IsPublic);
            methodOp->internalFlags |= InternalFlags::GeneratorWrapper;

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
            funcOp->internalFlags |= InternalFlags::GeneratorWrapper;

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
                
        // A field of an object literal holds one function, not a generic one: `{ f: (a) => 0 }` and `{ f(a) {} }`
        // take an untyped parameter as `any`, as a global variable's function does. Made generic, they crashed
        // in getNameOfFunction or failed with "can't find type by name: TGenParam_a".
        auto parentKind = (SyntaxKind)functionLikeDeclarationBaseAST->parent;
        auto isObjectLiteralMember = parentKind == SyntaxKind::PropertyAssignment
            || parentKind == SyntaxKind::ObjectLiteralExpression;
        // ...nor is a constructor, which can't have type parameters: `constructor(public x, y)` takes `any`s
        // (made generic, its class failed with "can't find type by name: TGenParam_x")
        auto isConstructor = functionLikeDeclarationBaseAST == SyntaxKind::Constructor;
        // ...and its type parameters (`{ test: <T>(value: T): T => value }`, `{ m<T>(x: T) {} }`) are `any` - the
        // field holds one function ("can't find type by name: T")
        auto objectMemberTypeParamsAny = isObjectLiteralMember && !instantiateSpecializedFunction
            && bindTypeParamsToAny(functionLikeDeclarationBaseAST, funcDeclGenContext);

        auto isGenericFunction =
            (functionLikeDeclarationBaseAST->typeParameters.size() > 0 && !objectMemberTypeParamsAny)
            || !genContext.isGlobalVarReceiver && !isObjectLiteralMember && !isConstructor
                && isGenericParameters(functionLikeDeclarationBaseAST, genContext);
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
            if (funcEntry && lookupModuleSymbol(functionName)
                || lookupModuleSymbol(fullFunctionName))
            {
                // resolve a live op from the module instead of returning a cached handle;
                // the registered symbol is usually the full name
                auto funcOp = lookupModuleSymbol<mlir_ts::FuncOp>(functionName);
                if (!funcOp)
                {
                    funcOp = lookupModuleSymbol<mlir_ts::FuncOp>(fullFunctionName);
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
        StringRef dllFuncName = funcProto->getName();
        iterateDecorators(functionLikeDeclarationBaseAST, genContext, [&](StringRef name, SmallVector<StringRef> args) {
            if (name == DLL_IMPORT && args.size() > 0)
            {
                dynamicImport = true;
            }

            if (name == DLL_NAME && args.size() > 0)
            {
                dllFuncName = args.front();
            }
        });

        if (dynamicImport)
        {
            // TODO: we do not need to register funcOp as we need to reference global variables
            auto result = mlirGenFunctionLikeDeclarationDynamicImport(
                location, funcOp, funcProto->getNameWithoutNamespace(), funcOp.getFunctionType(),
                dllFuncName, funcDeclGenContext, false);
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

        // A function of a block of the module's code was entered by the pass over the module's declarations,
        // without the captures found here (isModuleDiscoveryExpression): its entry is this one. Called with the
        // old one's type, `foo()` had no captures to pass and crashed.
        if ((functionLikeDeclarationBaseAST->internalFlags & InternalFlags::InModuleBlock) == InternalFlags::InModuleBlock
            && !genContext.dummyRun)
        {
            getFunctionMap().erase(funcProto->getNameWithoutNamespace());
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
            || funcProto->getName() == MAIN_ENTRY_NAME
            // @dllname/@linkname: the function defines a symbol that something else binds by that
            // name - a `declare` in this module, or another object. Private, SymbolDCE may erase it
            // before TypeScriptToLLVMLoweringPass gives it the name (--opt inlines its callers
            // first), and the declaration then binds nothing.
            || funcOp->hasAttr(DLL_NAME);

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
            pushFunction(funcOp);
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

    mlir::LogicalResult MLIRGenImpl::mlirGenFunctionLikeDeclarationDynamicImport(mlir::Location location, mlir_ts::FuncOp declared,
        StringRef funcName, mlir_ts::FunctionType functionType, StringRef dllFuncName, const GenContext &genContext, bool isFullNamespaceName)
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

        // what the declaration says (`@keeps`, `@borrows`) goes with the global its calls go
        // through, which is all -mm=own sees of the function
        if (declared && !genContext.dummyRun)
        {
            auto globalName = isFullNamespaceName ? funcName.str() : getFullNamespaceName(funcName).str();
            if (auto globalOp = lookupModuleSymbol<mlir_ts::GlobalOp>(globalName))
            {
                for (auto *name : {OWN_DECLARED_KEEPS_ATTR_NAME, OWN_DECLARED_BORROWS_ATTR_NAME, OWN_DECLARED_LENDS_ATTR_NAME,
                                   OWN_DECLARED_RECORD_ATTR_NAME, OWN_DECLARED_RETURNS_THIS_ATTR_NAME, OWN_IMPLIED_LENDS_ATTR_NAME,
                                   OWN_DECLARED_IMMORTALIZES_ATTR_NAME, OWN_DECLARED_PARAM_NAMES_ATTR_NAME})
                {
                    if (auto attr = declared->getAttr(name))
                    {
                        globalOp->setAttr(name, attr);
                    }
                }

                if (declared->hasAttr(STDCALL_ATTR_NAME) && compileOptions.targetInfo.stdcallDecoratesCxxThrow)
                {
                    stdcallDynamicImports[globalName] = dllFuncName.str();
                }
            }
        }

        return mlir::success();
    }    

    mlir::LogicalResult MLIRGenImpl::mlirGenFunctionEntry(mlir::Location location, FunctionPrototypeDOM::TypePtr funcProto,
                                             const GenContext &genContext)
    {
        return mlirGenFunctionEntry(location, funcProto->getReturnType(), genContext);
    }

    mlir::LogicalResult MLIRGenImpl::mlirGenFunctionEntry(mlir::Location location, mlir::Type retType, const GenContext &genContext)
    {
        // a function that returns `never` has no value to return either: it does not come back
        auto hasReturn = retType && !isa<mlir_ts::VoidType, mlir_ts::NeverType>(retType);
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
        // a function that returns `never` has no value to return either: it does not come back
        auto hasReturn = retType && !isa<mlir_ts::VoidType, mlir_ts::NeverType>(retType);
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
        // the function was made from another signature of the same name (an overload) with fewer
        // parameters than this body has
        if (firstIndex + funcProto->getParams().size() > arguments.size())
        {
            auto &firstExtraParam = funcProto->getParams()[std::max(0, (int)arguments.size() - firstIndex)];
            emitError(firstExtraParam->getLoc()) << "function '" << funcProto->getName()
                                           << "' has more parameters than another declaration of the same name";
            return mlir::failure();
        }

        for (auto [paramIndex, param] : enumerate(funcProto->getParams()))
        {
            auto index = firstIndex + (int)paramIndex;
            mlir::Value paramValue;

            // process init expression
            // we need reset scope for location as location of funcProto was created before correct scope
            auto location = locFuseWithScope(stripMetadata(param->getLoc()));

            LLVM_DEBUG(llvm::dbgs() << "Location for Param: " << location << "\n");

            // `this` of a @value class's constructor is the slot `new` made for the value: the
            // variable `this` lives in it, so `this.a = a` writes the value's field and `this.m()` sees
            // the fields stored so far; read whole (`f(this)`, `const c = this`) it is an error
            // (checkValueConstructorThisUses)
            if (param->getName() == THIS_NAME && index == firstIndex)
            {
                auto refType = dyn_cast<mlir_ts::RefType>(param->getType());
                if (refType && getValueClassInfo(refType.getElementType()))
                {
                    auto thisDecl = std::make_shared<VariableDeclarationDOM>(THIS_NAME, refType.getElementType(), location);
                    thisDecl->setReadWriteAccess();
                    DECLARE(thisDecl, arguments[index]);
                    continue;
                }
            }

            // `x: never` holds no value to keep: the name is the argument itself, as a `const`'s is its value
            if (isa<mlir_ts::NeverType>(param->getType()) && !param->hasInitValue() && !param->getIsOptional())
            {
                DECLARE(param, arguments[index]);
                continue;
            }

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
                // `this` of a @value class's method is the value, which nothing assigns: a closure
                // takes a copy of it, not its cell (so a `ref` named `this` in a capture is the
                // constructor's slot alone). So does `this` of any class's member: the copy is the
                // same object, with no cell for it (own spec 70: a field's closure over `this`
                // borrows the object, which a cell moved into the box could not)
                if (param->getName() == THIS_NAME && index == firstIndex &&
                    (getValueClassInfo(param->getType()) || isa<mlir_ts::ClassType>(param->getType())))
                {
                    param->setCaptureByValue();
                }

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

    // The names a function body assigns: `x = ...` and every compound assignment, the targets of a destructuring
    // assignment, and the variable of `for (x of ...)` / `for (x in ...)`. Nested functions are walked as well, and
    // a destructuring target counts every name in it: naming one too many costs a retain and a release, missing
    // one leaves a parameter pointing at freed memory. `lengthNames`, when given, collects the names whose
    // `.length` is assigned (`this` for `this.length = n`): resizing a string makes a new one (spec 22.3), so for a
    // string that is an assignment too.
    static void collectAssignedNames(ts::Node body, llvm::StringSet<> &names, llvm::StringSet<> *lengthNames = nullptr)
    {
        auto addTargets = [&](ts::Node target) {
            while (target == SyntaxKind::ParenthesizedExpression)
            {
                target = target.as<ts::ParenthesizedExpression>()->expression;
            }

            if (target == SyntaxKind::Identifier)
            {
                names.insert(MLIRHelper::getName(target.as<ts::Identifier>()));
            }
            else if (target == SyntaxKind::ArrayLiteralExpression || target == SyntaxKind::ObjectLiteralExpression)
            {
                ts::VisitorAST targetNames([&](ts::Node node) {
                    if (node == SyntaxKind::Identifier)
                    {
                        names.insert(MLIRHelper::getName(node.as<ts::Identifier>()));
                    }
                });
                targetNames.visit(target);
            }
        };

        ts::VisitorAST visitor([&](ts::Node node) {
            if (node == SyntaxKind::BinaryExpression)
            {
                auto binaryExpression = node.as<ts::BinaryExpression>();
                if (ts::isAssignmentOperator(binaryExpression->operatorToken))
                {
                    addTargets(binaryExpression->left);

                    auto left = binaryExpression->left;
                    if (lengthNames && left == SyntaxKind::PropertyAccessExpression)
                    {
                        auto propertyAccess = left.as<ts::PropertyAccessExpression>();
                        auto object = propertyAccess->expression;
                        if (propertyAccess->name == SyntaxKind::Identifier &&
                            MLIRHelper::getName(propertyAccess->name.as<ts::Identifier>()) == LENGTH_FIELD_NAME)
                        {
                            if (object == SyntaxKind::Identifier)
                            {
                                lengthNames->insert(MLIRHelper::getName(object.as<ts::Identifier>()));
                            }
                            else if (object == SyntaxKind::ThisKeyword)
                            {
                                lengthNames->insert(THIS_NAME);
                            }
                        }
                    }
                }
            }
            else if (node == SyntaxKind::PrefixUnaryExpression || node == SyntaxKind::PostfixUnaryExpression)
            {
                auto opCode = node == SyntaxKind::PrefixUnaryExpression
                                  ? node.as<ts::PrefixUnaryExpression>()->_operator
                                  : node.as<ts::PostfixUnaryExpression>()->_operator;
                if (opCode == SyntaxKind::PlusPlusToken || opCode == SyntaxKind::MinusMinusToken)
                {
                    if (node == SyntaxKind::PrefixUnaryExpression)
                    {
                        addTargets(node.as<ts::PrefixUnaryExpression>()->operand);
                    }
                    else
                    {
                        addTargets(node.as<ts::PostfixUnaryExpression>()->operand);
                    }
                }
            }
            else if (node == SyntaxKind::ForOfStatement)
            {
                auto initializer = node.as<ts::ForOfStatement>()->initializer;
                if (initializer != SyntaxKind::VariableDeclarationList)
                {
                    addTargets(initializer);
                }
            }
            else if (node == SyntaxKind::ForInStatement)
            {
                auto initializer = node.as<ts::ForInStatement>()->initializer;
                if (initializer != SyntaxKind::VariableDeclarationList)
                {
                    addTargets(initializer);
                }
            }
        });
        visitor.visit(body);
    }

    // Every name a function nested in this body mentions: what a closure there may capture. A name too many only
    // keeps a parameter as it was.
    static void collectNestedFunctionNames(ts::Node body, llvm::StringSet<> &names)
    {
        ts::VisitorAST visitor([&](ts::Node node) {
            if (node == SyntaxKind::ArrowFunction || node == SyntaxKind::FunctionExpression ||
                node == SyntaxKind::FunctionDeclaration || node == SyntaxKind::MethodDeclaration ||
                node == SyntaxKind::GetAccessor || node == SyntaxKind::SetAccessor || node == SyntaxKind::Constructor)
            {
                ts::VisitorAST nested([&](ts::Node inner) {
                    if (inner == SyntaxKind::Identifier)
                    {
                        names.insert(MLIRHelper::getName(inner.as<ts::Identifier>()));
                    }
                });
                nested.visit(node);
            }
        });
        visitor.visit(body);
    }

    // -mm=rc: a parameter is borrowed - the caller owns its value (takeOwnershipOfLocal), so its slot neither
    // retains nor releases. One the body assigns holds values the frame made, though: `x = new B(1)` stored the
    // instance with no reference of its own, the block it was made in released it, and `x` pointed at freed
    // memory - in a loop, the condition read it on the next iteration (#512). A parameter the body assigns is
    // copied into a local of the same name, which owns what it holds as any local does: a reference taken on the
    // caller's value here, handed over by each assignment, given back at the function's exit.
    //
    // -mm=own: a string parameter the body assigns - `s = s + t`, or `this.length = n`, which resizes into a new
    // string - starts a local of its own the same way, which own makes from a copy of the caller's string (spec
    // 22.3) and which then owns each string assigned to it. Left as the parameter's slot, what was assigned to it
    // had no owner: it leaked, and `return s` gave back a copy of it - read up to its first NUL, so the default
    // library's `"".clone().resize(n)` returned a string of one byte that its caller wrote n into.
    //
    // -mm=own: a Shared<T> parameter the body assigns is copied too. Own steps aside for Shared handles - their
    // counts are rc's - so it accepts the assignment, and the slot held a handle that only borrowed the new
    // value: assigned in a loop, the handle was released at the end of the loop body and a later `p.value` read
    // freed memory.
    //
    // -mm=own: any other parameter the body assigns - an object, an array - starts a local as well. The local
    // owns what it holds, so the caller's value has to become its own: the function keeps the parameter, as one
    // that stores it in a field does, and its callers give the argument up (the ownership passes' kept
    // parameters; '@keeps' says it for a function another module may call). A caller that reads the argument
    // after the call is told it was moved. A parameter left without a local own still refuses to assign
    // (OwnershipInferencePass): one with a binding pattern, and one a nested function mentions - a closure's
    // cell borrows the caller's value, and the error for that names the capture. A collector needs none of this.
    mlir::LogicalResult MLIRGenImpl::mlirGenFunctionOwnAssignedParams(mlir::Location location,
                                                                      FunctionLikeDeclarationBase functionLikeDeclarationBaseAST,
                                                                      FunctionPrototypeDOM::TypePtr funcProto,
                                                                      const GenContext &genContext)
    {
        auto own = compileOptions.memoryModel == MemoryModelOwn;
        if (!compileOptions.isRefCounted() && !own)
        {
            return mlir::success();
        }

        llvm::StringSet<> assignedNames;
        llvm::StringSet<> lengthNames;
        llvm::StringSet<> nestedNames;
        auto scanned = false;
        for (const auto &param : funcProto->getParams())
        {
            if (param->getBindingPattern() || !mth.ownsHeapMemory(location, param->getType()))
            {
                continue;
            }

            if (!scanned)
            {
                collectAssignedNames(functionLikeDeclarationBaseAST->body, assignedNames, own ? &lengthNames : nullptr);
                if (own)
                {
                    collectNestedFunctionNames(functionLikeDeclarationBaseAST->body, nestedNames);
                }

                scanned = true;
            }

            // own: a string or a Shared<T> handle is a copy whoever captures it; any other parameter a closure may
            // capture stays the caller's
            if (own && !mlir::isa<mlir_ts::StringType, mlir_ts::SharedType>(param->getType()) &&
                nestedNames.contains(param->getName()))
            {
                continue;
            }

            // `p.length = n` makes a new value only of a string: an array is resized where it is
            if (!assignedNames.contains(param->getName()) &&
                !(lengthNames.contains(param->getName()) && mlir::isa<mlir_ts::StringType>(param->getType())))
            {
                continue;
            }

            auto paramValue = resolveIdentifier(location, param->getName(), genContext);
            if (!paramValue)
            {
                return mlir::failure();
            }

            auto paramType = param->getType();
            if (!registerVariable(
                    location, param->getName(), false, VariableType::Let,
                    [&](mlir::Location, const GenContext &) -> TypeValueInitType {
                        return {paramType, paramValue, TypeProvided::Yes};
                    },
                    genContext))
            {
                return mlir::failure();
            }
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

        // the labels around a function are not the body's to jump to
        auto outerLabels = std::move(activeLabels);
        activeLabels.clear();
        auto outerLoopLabelAliases = std::move(loopLabelAliases);
        loopLabelAliases.clear();
        auto restoreLabels = llvm::make_scope_exit([&]() {
            activeLabels = std::move(outerLabels);
            loopLabelAliases = std::move(outerLoopLabelAliases);
        });

        // nor are the locals of the function a generic is instantiated from this body's: it starts a new
        // lexical root, and a function generated within it joins it (isolateGeneratedBodies)
        auto outerLexicalFuncOps = lexicalFuncOps;
        auto outerIsolateGeneratedBodies = isolateGeneratedBodies;
        if (isolateGeneratedBodies)
        {
            lexicalFuncOps.clear();
            lexicalFuncOps.push_back(funcOp);
            isolateGeneratedBodies = false;
        }
        else if (!lexicalFuncOps.empty())
        {
            lexicalFuncOps.push_back(funcOp);
        }
        auto restoreLexicalRoot = llvm::make_scope_exit([&]() {
            lexicalFuncOps = std::move(outerLexicalFuncOps);
            isolateGeneratedBodies = outerIsolateGeneratedBodies;
        });

        // nor is a @value type a body needs one the field being typed around it waits for (an
        // initializer's return type inferred from this body)
        auto outerValueLayoutFrames = std::move(valueLayoutFrames);
        valueLayoutFrames.clear();
        auto restoreValueLayoutFrames = llvm::make_scope_exit([&]() { valueLayoutFrames = std::move(outerValueLayoutFrames); });

        // nor are the objects an optional chain around it read (mlirGenOptionalChain) this body's: made
        // again inside the chain - a recursive function's result inferred - the chain's nodes are made
        // anew
        auto outerOptionalChainObjects = std::move(optionalChainObjects);
        optionalChainObjects.clear();
        auto restoreOptionalChainObjects = llvm::make_scope_exit([&]() { optionalChainObjects = std::move(outerOptionalChainObjects); });

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
        // its parameters, for `arguments` (mlirGenArguments)
        funcGenContext.bodyFuncProto = funcProto;
        funcGenContext.bodyIsArrow = functionLikeDeclarationBaseAST == SyntaxKind::ArrowFunction;
        funcGenContext.discardReturnValue = funcProto->getDiscardsReturnValue();

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

        // what a closure captures is named before its parameters are: a parameter's default value reads it
        // (`(b = y) => b`, `(b = super.m()) => b`), and read across the closure's isolated region it was
        // "'ts.Load' op using value defined outside the region". Captures held by an object's `this` are
        // reached through that parameter, and come after it (below).
        auto capturesInThis = funcGenContext.thisType != nullptr && isa<mlir_ts::ObjectType>(funcGenContext.thisType);
        if (!capturesInThis && failed(mlirGenFunctionCaptures(location, funcProto, funcGenContext)))
        {
            return mlir::failure();
        }

        // allocate function parameters as variable
        if (failed(mlirGenFunctionParams(firstIndex, funcProto, arguments, funcGenContext)))
        {
            return mlir::failure();
        }

        // A parameter nothing assigns - not this body, not a closure in it - holds the caller's value
        // for the whole call, so a closure takes a copy of it, as it does of `this` and of a `const`,
        // instead of moving it into a heap cell the frame and the box share: a closure made per call
        // (`reduce((col, light) => ... pos ...)`) cost one cell per parameter on every call.
        // -mm=own keeps a generator's parameters in cells, whose state keeps or borrows them by its
        // own rules (spec 38) - the generator is the wrapper buildGeneratorWrapperDeclaration made,
        // whose body holds the `next` method - and two kinds anywhere: a callback, whose reach own
        // follows through the cell (spec 54), and a Shared<T> handle, whose cell holds a count of its
        // own (spec 23.2)
        auto own = compileOptions.memoryModel == MemoryModelOwn;
        auto generatorWrapper = false;
        if (own && functionLikeDeclarationBaseAST->body)
        {
            ts::VisitorAST nextVisitor([&](ts::Node node) {
                generatorWrapper = generatorWrapper ||
                                   (node->internalFlags & InternalFlags::GeneratorNext) == InternalFlags::GeneratorNext;
            });
            nextVisitor.visit(functionLikeDeclarationBaseAST->body);
        }

        if (functionLikeDeclarationBaseAST->body && !generatorWrapper)
        {
            llvm::StringSet<> assignedNames;
            llvm::StringSet<> lengthNames;
            collectAssignedNames(functionLikeDeclarationBaseAST->body, assignedNames, &lengthNames);
            for (const auto &param : funcProto->getParams())
            {
                auto keptInCell = own && (mth.isAnyFunctionType(param->getType()) ||
                                          MLIRTypeHelper::isSharedHandleType(param->getType()));
                if (!keptInCell && !param->getBindingPattern() && param->getName() != THIS_NAME &&
                    !assignedNames.contains(param->getName()) && !lengthNames.contains(param->getName()))
                {
                    param->setCaptureByValue();
                }
            }
        }

        if (failed(mlirGenFunctionParamsBindings(firstIndex, funcProto, arguments, funcGenContext)))
        {
            return mlir::failure();
        }

        if (failed(mlirGenFunctionCapturedParamIfObject(location, firstIndex, funcProto, arguments, funcGenContext)))
        {
            return mlir::failure();
        }

        if (capturesInThis && failed(mlirGenFunctionCaptures(location, funcProto, funcGenContext)))
        {
            return mlir::failure();
        }

        // A parameter a closure captures is stored in a cell of its own, like a captured local,
        // and the cell has to go back the same way - so the frame is listed as an owner of every
        // parameter's storage, and scope exit gives back the ones that turned out to be cells.
        //
        // This list belongs to the function rather than to the body block, which is the scope a
        // parameter actually has, and it is wired in only now that the prologue is generated:
        // while it was null the prologue's own declarations - a destructured parameter's
        // bindings - were not owned by anything, and this is not the slice that changes that.
        auto paramCells = std::make_unique<SmallVector<mlir::Value>>();
        for (auto &prologueOp : entryBlock)
        {
            if (isa<mlir_ts::ParamOp, mlir_ts::ParamOptionalOp>(prologueOp))
            {
                paramCells->push_back(prologueOp.getResult(0));
            }
        }

        funcGenContext.ownedVars = paramCells.get();

        // A scope exit walking outwards stops here. The chain came in from whatever context this
        // function was generated under - for a nested function, the enclosing function's own
        // blocks - and a `return` releasing those would be releasing another frame's locals from
        // inside this one. It has always been wrong; it only became reachable now that this
        // context has a list of its own, since the walk used to stop at the first context
        // without one.
        funcGenContext.parentBlockContext = nullptr;
        // nor is it the body of an enclosing function's TryOp
        funcGenContext.isUnwindCleanupScope = false;

        // if we need params only we do not need to process body
        auto discoverParamsOnly = funcGenContext.allowPartialResolve && funcGenContext.discoverParamsOnly;
        llvm::StringSet<> functionVars;
        llvm::DenseSet<int> functionTopLevelVars;
        if (!discoverParamsOnly)
        {
            // after the list is wired in: the copy is the function's to give back at exit
            if (failed(mlirGenFunctionOwnAssignedParams(location, functionLikeDeclarationBaseAST, funcProto, funcGenContext)))
            {
                return mlir::failure();
            }

            // a @value class's constructor reads `this` only through its members - in a discovery run
            // too, where a use of `this` whole may fail first with an error of its own (`{ ...this }`)
            if (functionLikeDeclarationBaseAST == SyntaxKind::Constructor
                && !funcProto->getParams().empty() && funcProto->getParams().front()->getName() == THIS_NAME)
            {
                auto thisRefType = dyn_cast<mlir_ts::RefType>(funcProto->getParams().front()->getType());
                if (auto classInfo = thisRefType ? getValueClassInfo(thisRefType.getElementType()) : ClassInfo::TypePtr())
                {
                    auto thisUses = checkValueConstructorThisUses(functionLikeDeclarationBaseAST->body, classInfo->name);

                    // ...and so do its instance fields' initializers, which run in it (in the one made
                    // for them when none is written)
                    if (classInfo->processingDeclaration)
                    {
                        for (auto &member : classInfo->processingDeclaration->members)
                        {
                            if (member == SyntaxKind::PropertyDeclaration && !hasModifier(member, SyntaxKind::StaticKeyword)
                                && member.as<PropertyDeclaration>()->initializer
                                && failed(checkValueConstructorThisUses(member.as<PropertyDeclaration>()->initializer, classInfo->name)))
                            {
                                thisUses = mlir::failure();
                            }
                        }
                    }

                    if (failed(thisUses))
                    {
                        return mlir::failure();
                    }
                }
            }

            // the function's `var`s (MDN `var`): one variable each for the whole function, whichever block declares
            // it, declared before the body; a parameter of the same name is that variable
            for (const auto &param : funcProto->getParams())
            {
                functionVars.insert(param->getName());
            }

            funcGenContext.functionVars = &functionVars;
            funcGenContext.functionTopLevelVars = &functionTopLevelVars;
            funcGenContext.functionVarScope = symbolTable.getCurScope();
            funcGenContext.functionVarContext = &funcGenContext;
            funcGenContext.functionVarBlock = &entryBlock;
            funcGenContext.functionVarAnchor = entryBlock.empty() ? nullptr : &entryBlock.back();
            if (failed(declareFunctionVars(functionLikeDeclarationBaseAST->body, funcGenContext)))
            {
                return mlir::failure();
            }

            // we need it to skip lexical block
            functionLikeDeclarationBaseAST->body->parent = functionLikeDeclarationBaseAST->body;
            if (failed(mlirGenBody(functionLikeDeclarationBaseAST->body, funcGenContext)))
            {
                return mlir::failure();
            }

            // falling off the end of a function returning T | undefined (inferred so, or declared) returns undefined;
            // the return slot would otherwise be read as it was left
            if (auto returnType = getExplicitReturnTypeOfCurrentFunction(funcGenContext))
            {
                if ((isa<mlir_ts::OptionalType>(returnType) || isa<mlir_ts::AnyType>(returnType)
                     || isa<mlir_ts::UnknownType>(returnType))
                    && bodyFallsOffEnd(functionLikeDeclarationBaseAST->body))
                {
                    NodeFactory nf(NodeFactoryFlags::None);
                    auto undefinedReturn = nf.createReturnStatement(nf.createIdentifier(S(UNDEFINED_NAME)));
                    undefinedReturn->pos = functionLikeDeclarationBaseAST->body->_end - 1;
                    undefinedReturn->_end = functionLikeDeclarationBaseAST->body->_end;
                    if (failed(mlirGen(undefinedReturn, funcGenContext)))
                    {
                        return mlir::failure();
                    }
                }
            }
        }

        // Falling off the end of the body reaches here rather than through a `return`, and the
        // body block's own exit stops at itself. A `return` inside walks the whole stack outwards
        // and so has already released these on its own path.
        if (failed(mlirGenReleaseOwned(location, DisposeDepth::CurrentScope, {}, &funcGenContext)))
        {
            return mlir::failure();
        }

        // add exit code
        if (failed(mlirGenFunctionExit(location, funcGenContext)))
        {
            return mlir::failure();
        }

        releaseCapturedSlotsAsCells(funcOp);

        if (!funcGenContext.dummyRun && failed(checkValueConstructorClosures(funcOp)))
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
        if (lookupModuleSymbol(fullFuncName))
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
        funcGenContext.discardReturnValue = false;
        // a synthesized function has no `var`s of its own, and is not the function it may be made in
        funcGenContext.functionVars = nullptr;
        funcGenContext.functionTopLevelVars = nullptr;
        funcGenContext.functionVarContext = nullptr;
        funcGenContext.functionVarScope = nullptr;

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

        // add exit code: a function returning nothing declares no return variable of its own, and one
        // made while another function is being generated (a null interface's method stub for a cast
        // in it) finds that function's instead
        auto retVarInfo = symbolTable.lookup(RETURN_VARIABLE_NAME);
        auto *retVarDef = retVarInfo.first ? retVarInfo.first.getDefiningOp() : nullptr;
        if (retVarDef && funcOp->isProperAncestor(retVarDef))
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
            pushFunction(funcOp);
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

            // How the box will store this variable decides which reference has to be taken, and
            // it is not the same question as whether a reference to the variable can be had.
            // MLIRCodeLogic::CaptureTypeStorage gives a read-write capture a `ref` field - the
            // address of the variable's cell - and everything else a field of the variable's own
            // type, which CaptureOpLowering fills by dereferencing. A by-value field is released
            // by `releaseCapturedFields` like any other owning field, so the copy needs a
            // reference of its own; retaining the cell instead leaves the value with one owner
            // fewer than the releases that will run for it, which is what freed the source array
            // of a generator out from under the generator (§9.50).
            auto capturedByRef = item.second && item.second->getCaptureByReference();

            // review capturing by ref.  it should match storage type
            auto refValue = mcl.GetReferenceFromValue(location, varValue);

            // A copy is taken through the reference only when that reference is the variable's own
            // storage. A `const` folded into a read (`const v = n.v`, `const v = G`) has none: the
            // reference is where it was read from, and dereferenced now it gave the closure what the
            // place holds by the time the closure is made - `n.v = d` in between, and `v` in the
            // closure was `d`. The value itself is what the const names.
            auto isOwnStorage = refValue && (refValue.getDefiningOp<mlir_ts::VariableOp>() ||
                                             refValue.getDefiningOp<mlir_ts::ParamOp>() ||
                                             refValue.getDefiningOp<mlir_ts::ParamOptionalOp>() ||
                                             isCapturedCellSlot(refValue));
            if (refValue && !capturedByRef && !isOwnStorage)
            {
                refValue = mlir::Value();
            }

            if (refValue && !capturedByRef)
            {
                // the box holds a copy, exactly as it does for a value with no reference at all
                capturedValues.push_back(refValue);
                mlirGenRetainCaptured(location, mlir::ValueRange{varValue});
            }
            else if (refValue)
            {
                capturedValues.push_back(refValue);
                // set var as captures
                if (auto varOp = refValue.getDefiningOp<mlir_ts::VariableOp>())
                {
                    varOp.setCapturedAttr(builder.getBoolAttr(true));
                    // the box about to be built is a further owner of this variable's cell
                    builder.create<mlir_ts::RetainCellOp>(location, refValue);
                }
                else if (auto paramOp = refValue.getDefiningOp<mlir_ts::ParamOp>())
                {
                    paramOp.setCapturedAttr(builder.getBoolAttr(true));
                    builder.create<mlir_ts::RetainCellOp>(location, refValue);
                }
                else if (auto paramOptOp = refValue.getDefiningOp<mlir_ts::ParamOptionalOp>())
                {
                    paramOptOp.setCapturedAttr(builder.getBoolAttr(true));
                    builder.create<mlir_ts::RetainCellOp>(location, refValue);
                }
                else if (MLIRTypeHelper::isValueConstructorThisCapture(builder.getStringAttr(item.first()),
                                                                       refValue.getType()))
                {
                    // `this` of a @value class's constructor, seen in a closure through its box: the
                    // slot `new` made, which this box borrows as the first one does - not a cell
                }
                else if (isCapturedCellSlot(refValue))
                {
                    // A closure inside a closure, capturing the same variable. There is nothing
                    // to mark here - the frame that declared the variable already made its
                    // storage a cell, and this function only knows the cell through its own
                    // capture box - but the box about to be built owns it exactly as that first
                    // box does, and releases it when the bound function goes. Without the
                    // matching retain the inner box gives back a count nobody added.
                    builder.create<mlir_ts::RetainCellOp>(location, refValue);
                }
                else
                {
                    // no retain here: what makes a variable's storage a cell is being marked
                    // captured, and nothing was marked. Retaining a stack slot would write a
                    // count into the frame word in front of it. `this` of a @value class's
                    // constructor lands here by design: its storage is the slot `new` made, which
                    // the box borrows (OwnershipRoutineLogic::releaseCapturedFields skips it).
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
                // this is not ref, this is const value - the box holds a copy, and a copy of a
                // reference is a further owner of what it points at
                capturedValues.push_back(varValue);
                mlirGenRetainCaptured(location, mlir::ValueRange{varValue});
            }
        }

        return mlir::success();
    }

    // A use of a closure holding the slot of a @value constructor's `this` other than calling it
    // (GetThis/GetMethod) or counting it (Retain/Release); null when there is none. Captured by another
    // closure is such a use: inside that one it is a field of a box, which could be stored or returned
    // from there unseen.
    static mlir::Operation *escapingUseOfValueConstructorClosure(mlir::Value closure)
    {
        for (auto *user : closure.getUsers())
        {
            if (!isa<mlir_ts::GetThisOp, mlir_ts::GetMethodOp, mlir_ts::RetainOp, mlir_ts::ReleaseOp>(user))
            {
                return user;
            }
        }

        return nullptr;
    }

    // A closure that captures `this` of a @value class's constructor holds the address of the slot
    // `new` made, which dies with `new`'s caller (MLIRTypeHelper::isValueConstructorThisCapture). So
    // it may only be called, in the function that made it: the constructor, or a closure inside it
    // that captures `this` itself. Stored, returned, passed on or captured, it could read a dead slot.
    mlir::LogicalResult MLIRGenImpl::checkValueConstructorClosures(mlir_ts::FuncOp funcOp)
    {
        auto result = mlir::success();
        funcOp.walk([&](mlir_ts::CreateBoundFunctionOp boundFuncOp) {
            auto captureOp = boundFuncOp.getThisVal().getDefiningOp<mlir_ts::CaptureOp>();
            auto captureRefType = captureOp ? dyn_cast<mlir_ts::RefType>(captureOp->getResult(0).getType()) : mlir_ts::RefType();
            auto captureTupleType = captureRefType ? dyn_cast<mlir_ts::TupleType>(captureRefType.getElementType()) : mlir_ts::TupleType();
            if (!captureTupleType)
            {
                return;
            }

            for (auto &field : captureTupleType.getFields())
            {
                if (!MLIRTypeHelper::isValueConstructorThisCapture(field.id, field.type))
                {
                    continue;
                }

                if (auto *use = escapingUseOfValueConstructorClosure(boundFuncOp.getResult()))
                {
                    auto valueType = mlir::cast<mlir_ts::RefType>(field.type).getElementType();
                    auto classInfo = getValueClassInfo(valueType);
                    auto className = classInfo ? classInfo->name.str() : to_print(valueType);
                    emitError(use->getLoc()) << "a closure that captures 'this' of the @" VALUE_DECORATOR " class '"
                                             << className << "' can only be called inside its constructor";
                    result = mlir::failure();
                }

                return;
            }
        });

        return result;
    }

    // `this` of a @value class's constructor is the slot `new` is filling (mlirGenFunctionParams): the
    // value is made when the constructor returns. Read whole before that - `const self = this`,
    // `f(this)`, `[this]`, `{ v: this }`, `...this`, `return this`, `() => this` - it would be a copy
    // of the fields stored so far, and a later `this.a = 1` would not reach it, where TypeScript's
    // object would have it. So `this` may only be the object of a member access (`this.a`,
    // `this.a = x`, `this.a++`, `this.m()`, `this["a"]`), read as the slot is at that point, in the
    // constructor or in an arrow function inside it, whose `this` is the constructor's. A function,
    // method, accessor or class inside it has a `this` of its own. `root` is the constructor's body, or
    // an instance field's initializer, which runs in the constructor too (mlirGenFunctionBody).
    mlir::LogicalResult MLIRGenImpl::checkValueConstructorThisUses(Node root, StringRef className)
    {
        auto result = mlir::success();

        auto isThis = [](Node expression) {
            while (expression == SyntaxKind::ParenthesizedExpression)
            {
                expression = expression.as<ParenthesizedExpression>()->expression;
            }

            return expression == SyntaxKind::ThisKeyword;
        };

        ts::FuncT<> visitNode;
        ts::ArrayFuncT<> visitArray;

        visitNode = [&](Node node) -> Node {
            switch ((SyntaxKind)node)
            {
            case SyntaxKind::FunctionExpression:
            case SyntaxKind::FunctionDeclaration:
            case SyntaxKind::MethodDeclaration:
            case SyntaxKind::GetAccessor:
            case SyntaxKind::SetAccessor:
            case SyntaxKind::Constructor:
            case SyntaxKind::ClassDeclaration:
            case SyntaxKind::ClassExpression:
                return undefined;
            case SyntaxKind::PropertyAccessExpression:
                if (isThis(node.as<PropertyAccessExpression>()->expression))
                {
                    return undefined;
                }

                break;
            case SyntaxKind::ElementAccessExpression:
                if (auto elementAccess = node.as<ElementAccessExpression>(); isThis(elementAccess->expression))
                {
                    return visitNode(elementAccess->argumentExpression);
                }

                break;
            case SyntaxKind::ThisKeyword:
                emitError(loc(node)) << "'this' of the @" VALUE_DECORATOR " class '" << className
                                     << "' is not a value until its constructor returns; read 'this.f'";
                result = mlir::failure();
                return undefined;
            default:
                break;
            }

            ts::forEachChild(node, visitNode, visitArray);
            return undefined;
        };

        visitArray = [&](NodeArray<Node> array) -> Node {
            for (auto node : array)
            {
                visitNode(node);
            }

            return undefined;
        };

        visitNode(root);
        return result;
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
            auto boundFuncVal = builder.create<mlir_ts::CreateBoundFunctionOp>(location, getBoundFunctionType(funcType), captured, funcSymbolOp);

            // The capture box was allocated for this closure and nothing else holds it, so the
            // closure is its owner - which is the one shape of function value that owns its
            // `this`, and the reason for OWNS_CAPTURE_ATTR_NAME. A closure built here and then
            // dropped - `apply(v => v.x + base.x, v)` - is released at the end of the block that
            // made it (§9.30), which is what gives the box back.
            boundFuncVal->setAttr(OWNS_CAPTURE_ATTR_NAME, builder.getUnitAttr());
            if (compileOptions.memoryModel == MemoryModelOwn)
            {
                if (auto captureUses = getCaptureUsesMap().find(name); captureUses != getCaptureUsesMap().end())
                {
                    SmallVector<mlir::NamedAttribute> uses;
                    for (auto &use : captureUses->getValue())
                    {
                        uses.push_back(builder.getNamedAttr(use.getKey(), use.getValue()));
                    }

                    boundFuncVal->setAttr(CAPTURE_USES_ATTR_NAME, builder.getDictionaryAttr(uses));
                }
            }

            builder.create<mlir_ts::RetainOp>(location, boundFuncVal);
            boundFuncVal->setAttr(OWNED_RESULT_ATTR_NAME, builder.getUnitAttr());

            return V(boundFuncVal);
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
