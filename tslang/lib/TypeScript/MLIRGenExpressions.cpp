// Expression-level code generation methods of MLIRGenImpl (see MLIRGenImpl.h).

#include "MLIRGenImpl.h"

#include "mlir/Dialect/Async/IR/Async.h"

#undef DEBUG_TYPE
#define DEBUG_TYPE "mlir"

namespace typescript
{
namespace mlirgen
{

    ValueOrLogicalResult MLIRGenImpl::mlirGen(Expression expressionAST, const GenContext &genContext)
    {
        auto kind = (SyntaxKind)expressionAST;
        if (kind == SyntaxKind::Identifier)
        {
            return mlirGen(expressionAST.as<Identifier>(), genContext);
        }
        else if (kind == SyntaxKind::PropertyAccessExpression)
        {
            return mlirGen(expressionAST.as<PropertyAccessExpression>(), genContext);
        }
        else if (kind == SyntaxKind::CallExpression)
        {
            return mlirGen(expressionAST.as<CallExpression>(), genContext);
        }
        else if (kind == SyntaxKind::NumericLiteral)
        {
            return mlirGen(expressionAST.as<NumericLiteral>(), genContext);
        }
        else if (kind == SyntaxKind::StringLiteral)
        {
            return mlirGen(expressionAST.as<ts::StringLiteral>(), genContext);
        }
        else if (kind == SyntaxKind::NoSubstitutionTemplateLiteral)
        {
            return mlirGen(expressionAST.as<NoSubstitutionTemplateLiteral>(), genContext);
        }
        else if (kind == SyntaxKind::BigIntLiteral)
        {
            return mlirGen(expressionAST.as<BigIntLiteral>(), genContext);
        }
        else if (kind == SyntaxKind::RegularExpressionLiteral)
        {
            return mlirGen(expressionAST.as<RegularExpressionLiteral>(), genContext);
        }
        else if (kind == SyntaxKind::NullKeyword)
        {
            return mlirGen(expressionAST.as<NullLiteral>(), genContext);
        }
        else if (kind == SyntaxKind::TrueKeyword)
        {
            return mlirGen(expressionAST.as<TrueLiteral>(), genContext);
        }
        else if (kind == SyntaxKind::FalseKeyword)
        {
            return mlirGen(expressionAST.as<FalseLiteral>(), genContext);
        }
        else if (kind == SyntaxKind::ArrayLiteralExpression)
        {
            return mlirGen(expressionAST.as<ArrayLiteralExpression>(), genContext);
        }
        else if (kind == SyntaxKind::ObjectLiteralExpression)
        {
            return mlirGen(expressionAST.as<ObjectLiteralExpression>(), genContext);
        }
        else if (kind == SyntaxKind::SpreadElement)
        {
            return mlirGen(expressionAST.as<SpreadElement>(), genContext);
        }
        else if (kind == SyntaxKind::BinaryExpression)
        {
            return mlirGen(expressionAST.as<BinaryExpression>(), genContext);
        }
        else if (kind == SyntaxKind::PrefixUnaryExpression)
        {
            return mlirGen(expressionAST.as<PrefixUnaryExpression>(), genContext);
        }
        else if (kind == SyntaxKind::PostfixUnaryExpression)
        {
            return mlirGen(expressionAST.as<PostfixUnaryExpression>(), genContext);
        }
        else if (kind == SyntaxKind::ParenthesizedExpression)
        {
            return mlirGen(expressionAST.as<ParenthesizedExpression>(), genContext);
        }
        else if (kind == SyntaxKind::TypeOfExpression)
        {
            return mlirGen(expressionAST.as<TypeOfExpression>(), genContext);
        }
        else if (kind == SyntaxKind::ConditionalExpression)
        {
            return mlirGen(expressionAST.as<ConditionalExpression>(), genContext);
        }
        else if (kind == SyntaxKind::ElementAccessExpression)
        {
            return mlirGen(expressionAST.as<ElementAccessExpression>(), genContext);
        }
        else if (kind == SyntaxKind::FunctionExpression)
        {
            return mlirGen(expressionAST.as<FunctionExpression>(), genContext);
        }
        else if (kind == SyntaxKind::ArrowFunction)
        {
            return mlirGen(expressionAST.as<ArrowFunction>(), genContext);
        }
        else if (kind == SyntaxKind::TypeAssertionExpression)
        {
            return mlirGen(expressionAST.as<TypeAssertion>(), genContext);
        }
        else if (kind == SyntaxKind::AsExpression)
        {
            return mlirGen(expressionAST.as<AsExpression>(), genContext);
        }
        else if (kind == SyntaxKind::TemplateExpression)
        {
            return mlirGen(expressionAST.as<TemplateLiteralLikeNode>(), genContext);
        }
        else if (kind == SyntaxKind::TaggedTemplateExpression)
        {
            return mlirGen(expressionAST.as<TaggedTemplateExpression>(), genContext);
        }
        else if (kind == SyntaxKind::NewExpression)
        {
            return mlirGen(expressionAST.as<NewExpression>(), genContext);
        }
        else if (kind == SyntaxKind::DeleteExpression)
        {
            mlirGen(expressionAST.as<DeleteExpression>(), genContext);
            return mlir::success();
        }
        else if (kind == SyntaxKind::ThisKeyword)
        {
            if ((expressionAST->internalFlags & InternalFlags::ThisArgAlias) == InternalFlags::ThisArgAlias)
            {
                return mlirGen(loc(expressionAST), THIS_ALIAS, genContext);
            }

            return mlirGen(loc(expressionAST), THIS_NAME, genContext);
        }
        else if (kind == SyntaxKind::SuperKeyword)
        {
            return mlirGen(loc(expressionAST), SUPER_NAME, genContext);
        }
        else if (kind == SyntaxKind::VoidExpression)
        {
            return mlirGen(expressionAST.as<VoidExpression>(), genContext);
        }
        else if (kind == SyntaxKind::YieldExpression)
        {
            return mlirGen(expressionAST.as<YieldExpression>(), genContext);
        }
        else if (kind == SyntaxKind::AwaitExpression)
        {
            return mlirGen(expressionAST.as<AwaitExpression>(), genContext);
        }
        else if (kind == SyntaxKind::NonNullExpression)
        {
            return mlirGen(expressionAST.as<NonNullExpression>(), genContext);
        }
        else if (kind == SyntaxKind::ClassExpression)
        {
            return mlirGen(expressionAST.as<ClassExpression>(), genContext);
        }
        else if (kind == SyntaxKind::OmittedExpression)
        {
            return mlirGen(expressionAST.as<OmittedExpression>(), genContext);
        }
        else if (kind == SyntaxKind::ExpressionWithTypeArguments)
        {
            return mlirGen(expressionAST.as<ExpressionWithTypeArguments>(), genContext);
        }
        else if (kind == SyntaxKind::Unknown /*TODO: temp solution to treat null expr as empty expr*/)
        {
            return mlir::success();
        }

        llvm_unreachable("unknown expression");
    }

    ValueOrLogicalResult MLIRGenImpl::mlirGen(FunctionExpression functionExpressionAST, const GenContext &genContext)
    {
        auto location = loc(functionExpressionAST);
        mlir_ts::FuncOp funcOp;
        std::string funcName;

        {
            mlir::OpBuilder::InsertionGuard guard(builder);
            builder.setInsertionPointToStart(theModule.getBody());

            // provide name for it
            auto funcGenContext = GenContext(genContext);
            funcGenContext.clearScopeVars();
            funcGenContext.thisType = nullptr;

            auto [result, funcOpRet, funcNameRet, isGenericRet] =
                mlirGenFunctionLikeDeclaration(functionExpressionAST, funcGenContext);
            if (mlir::failed(result))
            {
                return mlir::failure();
            }

            funcOp = funcOpRet;
            funcName = funcNameRet;
        }

        // if funcOp is null, means lambda is generic]
        if (!funcOp)
        {
            // return reference to generic method
            if (getGenericFunctionMap().count(funcName))
            {
                auto genericFunctionInfo = getGenericFunctionMap().lookup(funcName);
                // info: it will not take any capture now
                return resolveFunctionWithCapture(location, genericFunctionInfo->name, genericFunctionInfo->funcType,
                                                  mlir::Value(), true, genContext);
            }
            else
            {
                emitError(location) << "can't find generic function: " << funcName;
                return mlir::failure();
            }
        }

        return resolveFunctionWithCapture(location, funcOp.getName(), funcOp.getFunctionType(), mlir::Value(), false, genContext);
    }

    ValueOrLogicalResult MLIRGenImpl::mlirGen(ArrowFunction arrowFunctionAST, const GenContext &genContext)
    {
        auto location = loc(arrowFunctionAST);
        mlir_ts::FuncOp funcOp;
        std::string funcName;
        bool isGeneric;


        {
            mlir::OpBuilder::InsertionGuard guard(builder);
            builder.setInsertionPointToStart(theModule.getBody());

            // provide name for it
            auto allowFuncGenContext = GenContext(genContext);
            allowFuncGenContext.clearScopeVars();
            // if we set it to value we will not capture 'this' references
            allowFuncGenContext.thisType = nullptr;
            auto [result, funcOpRet, funcNameRet, isGenericRet] =
                mlirGenFunctionLikeDeclaration(arrowFunctionAST, allowFuncGenContext);
            if (mlir::failed(result))
            {
                return mlir::failure();
            }

            funcOp = funcOpRet;
            funcName = funcNameRet;
            isGeneric = isGenericRet;
        }

        // if funcOp is null, means lambda is generic
        if (!funcOp)
        {
            // return reference to generic method
            if (getGenericFunctionMap().count(funcName))
            {
                auto genericFunctionInfo = getGenericFunctionMap().lookup(funcName);

                auto funcType = genericFunctionInfo->funcType ? genericFunctionInfo->funcType : getFunctionType({}, {}, false);

                // info: it will not take any capture now
                return resolveFunctionWithCapture(location, genericFunctionInfo->name, funcType,
                                                  mlir::Value(), true, genContext);
            }
            else
            {
                emitError(location) << "can't find generic function: " << funcName;
                return mlir::failure();
            }
        }

        assert(funcOp);

        return resolveFunctionWithCapture(location, funcOp.getName(), funcOp.getFunctionType(), mlir::Value(), isGeneric, genContext);
    }

    ValueOrLogicalResult MLIRGenImpl::mlirGen(TypeAssertion typeAssertionAST, const GenContext &genContext)
    {
        auto location = loc(typeAssertionAST);

        auto typeInfo = getType(typeAssertionAST->type, genContext);
        if (!typeInfo)
        {
            return mlir::failure();
        }

        GenContext noReceiverGenContext(genContext);
        noReceiverGenContext.clearReceiverTypes();
        noReceiverGenContext.receiverType = typeInfo;

        auto result = mlirGen(typeAssertionAST->expression, noReceiverGenContext);
        EXIT_IF_FAILED_OR_NO_VALUE(result)
        auto exprValue = V(result);

        CAST_A(castedValue, location, typeInfo, exprValue, genContext);
        return castedValue;
    }

    ValueOrLogicalResult MLIRGenImpl::mlirGen(AsExpression asExpressionAST, const GenContext &genContext)
    {
        auto location = loc(asExpressionAST);

        auto typeInfo = getType(asExpressionAST->type, genContext);
        if (!typeInfo)
        {
            return mlir::failure();
        }

        GenContext noReceiverGenContext(genContext);
        noReceiverGenContext.clearReceiverTypes();
        noReceiverGenContext.receiverType = typeInfo;

        auto result = mlirGen(asExpressionAST->expression, noReceiverGenContext);
        EXIT_IF_FAILED_OR_NO_VALUE(result)
        auto exprValue = V(result);

        CAST_A(castedValue, location, typeInfo, exprValue, genContext);
        return castedValue;
    }

    ValueOrLogicalResult MLIRGenImpl::mlirGen(ComputedPropertyName computedPropertyNameAST, const GenContext &genContext)
    {
        auto result = mlirGen(computedPropertyNameAST->expression, genContext);
        EXIT_IF_FAILED_OR_NO_VALUE(result)
        auto exprValue = V(result);
        return exprValue;
    }

    ValueOrLogicalResult MLIRGenImpl::mlirGen(YieldExpression yieldExpressionAST, const GenContext &genContext)
    {
        if (yieldExpressionAST->asteriskToken)
        {
            return mlirGenYieldStar(yieldExpressionAST, genContext);
        }

        auto location = loc(yieldExpressionAST);

        if (genContext.passResult)
        {
            genContext.passResult->functionReturnTypeShouldBeProvided = true;
        }

        // get state
        auto state = 0;
        if (genContext.state)
        {
            state = (*genContext.state)++;
        }
        else
        {
            assert(false);
        }

        // set restore point (return point)
        stringstream num;
        num << state;

        NodeFactory nf(NodeFactoryFlags::None);

        if (evaluateProperty(nf.createToken(SyntaxKind::ThisKeyword), GENERATOR_STEP, genContext))
        {
            // save return point - state -> this.step = xxx
            auto setStateExpr = nf.createBinaryExpression(
                nf.createPropertyAccessExpression(nf.createToken(SyntaxKind::ThisKeyword), nf.createIdentifier(S(GENERATOR_STEP))),
                nf.createToken(SyntaxKind::EqualsToken), nf.createNumericLiteral(num.str(), TokenFlags::None));
            mlirGen(setStateExpr, genContext);
        }
        else
        {
            // save return point - state -> step = xxx
            auto setStateExpr = nf.createBinaryExpression(
                nf.createIdentifier(S(GENERATOR_STEP)),
                nf.createToken(SyntaxKind::EqualsToken), nf.createNumericLiteral(num.str(), TokenFlags::None));
            mlirGen(setStateExpr, genContext);
        }

        // return value
        auto yieldRetValue = getYieldReturnObject(nf, location, yieldExpressionAST->expression, false);
        auto result = mlirGen(yieldRetValue, genContext);
        EXIT_IF_FAILED_OR_NO_VALUE(result)
        auto yieldValue = V(result);

        mlirGenReturnValue(location, yieldValue, true, genContext);

        std::stringstream label;
        label << GENERATOR_STATELABELPREFIX << state;
        builder.create<mlir_ts::StateLabelOp>(location, label.str());

        // TODO: yield value to continue, should be loaded from "next(value)" parameter
        // return yieldValue;
        return mlir::success();
    }

    ValueOrLogicalResult MLIRGenImpl::mlirGen(AwaitExpression awaitExpressionAST, const GenContext &genContext)
    {
#ifdef ENABLE_ASYNC
        // TODO: due to cloning code into next function, it is not fixing scope properly
        auto location = stripMetadata(loc(awaitExpressionAST));

        auto resultType = evaluate(awaitExpressionAST->expression, genContext);

        ValueOrLogicalResult result(mlir::failure());
        auto asyncExecOp = builder.create<mlir::async::ExecuteOp>(
            location, resultType ? mlir::TypeRange{resultType} : mlir::TypeRange(), mlir::ValueRange{},
            mlir::ValueRange{}, [&](mlir::OpBuilder &builder, mlir::Location location, mlir::ValueRange values) {
                DITableScopeT debugAsyncCodeScope(debugScope);
                MLIRDebugInfoHelper mdi(builder, debugScope);

                // TODO: temp hack to break wrong chain on scopes because 'await' create extra function wrap
                mdi.clearDebugScope();
                mdi.setLexicalBlock(location);

                result = mlirGen(awaitExpressionAST->expression, genContext);
                if (result)
                {
                    auto value = V(result);
                    if (value)
                    {
                        builder.create<mlir::async::YieldOp>(location, mlir::ValueRange{value});
                    }
                    else
                    {
                        builder.create<mlir::async::YieldOp>(location, mlir::ValueRange{});
                    }
                }
            });
        EXIT_IF_FAILED_OR_NO_VALUE(result)

        if (resultType)
        {
            auto asyncAwaitOp = builder.create<mlir::async::AwaitOp>(location, asyncExecOp.getResults().back());
            return asyncAwaitOp.getResult();
        }
        else
        {
            auto asyncAwaitOp = builder.create<mlir::async::AwaitOp>(location, asyncExecOp.getToken());
        }

        return mlir::success();
#else
        return mlirGen(awaitExpressionAST->expression, genContext);
#endif
    }

    ValueOrLogicalResult MLIRGenImpl::mlirGen(UnaryExpression unaryExpressionAST, const GenContext &genContext)
    {
        return mlirGen(unaryExpressionAST.as<Expression>(), genContext);
    }

    ValueOrLogicalResult MLIRGenImpl::mlirGen(LeftHandSideExpression leftHandSideExpressionAST, const GenContext &genContext)
    {
        return mlirGen(leftHandSideExpressionAST.as<Expression>(), genContext);
    }

    ValueOrLogicalResult MLIRGenImpl::mlirGen(PrefixUnaryExpression prefixUnaryExpressionAST, const GenContext &genContext)
    {
        auto location = loc(prefixUnaryExpressionAST);

        auto opCode = prefixUnaryExpressionAST->_operator;

        auto expression = prefixUnaryExpressionAST->operand;
        auto result = mlirGen(expression, genContext);
        EXIT_IF_FAILED_OR_NO_VALUE(result)
        auto expressionValue = V(result);

        // special case "-" for literal value
        if (opCode == SyntaxKind::PlusToken || opCode == SyntaxKind::MinusToken || opCode == SyntaxKind::TildeToken || opCode == SyntaxKind::ExclamationToken)
        {
            if (auto constantOp = expressionValue.getDefiningOp<mlir_ts::ConstantOp>())
            {
                auto res = mlirGenPrefixUnaryExpression(location, opCode, constantOp, genContext);
                EXIT_IF_FAILED(res)
                if (res.value)
                {
                    return res.value;
                }
            }
        }

        switch (opCode)
        {
        case SyntaxKind::ExclamationToken:
            {
                auto boolValue = expressionValue;
                if (expressionValue.getType() != getBooleanType())
                {
                    CAST(boolValue, location, getBooleanType(), expressionValue, genContext);
                }

                return V(builder.create<mlir_ts::ArithmeticUnaryOp>(location, getBooleanType(),
                                                                    builder.getI32IntegerAttr((int)opCode), boolValue));
            }
        case SyntaxKind::TildeToken:
            {
                auto numberValue = expressionValue;
                if (!expressionValue.getType().isIntOrIndexOrFloat())
                {
                    CAST(numberValue, location, builder.getI32Type(), expressionValue, genContext);
                }

                return V(builder.create<mlir_ts::ArithmeticUnaryOp>(
                    location, numberValue.getType(), builder.getI32IntegerAttr((int)opCode), numberValue));
            }
        case SyntaxKind::PlusToken:
        case SyntaxKind::MinusToken:
            {
                auto numberValue = expressionValue;
                if (expressionValue.getType() != getNumberType() && !expressionValue.getType().isIntOrIndexOrFloat())
                {
                    CAST(numberValue, location, getNumberType(), expressionValue, genContext);
                }

                return V(builder.create<mlir_ts::ArithmeticUnaryOp>(
                    location, numberValue.getType(), builder.getI32IntegerAttr((int)opCode), numberValue));
            }
        case SyntaxKind::PlusPlusToken:
        case SyntaxKind::MinusMinusToken:
            return V(builder.create<mlir_ts::PrefixUnaryOp>(location, expressionValue.getType(),
                                                            builder.getI32IntegerAttr((int)opCode), expressionValue));
        default:
            llvm_unreachable("not implemented");
        }
    }

    ValueOrLogicalResult MLIRGenImpl::mlirGen(PostfixUnaryExpression postfixUnaryExpressionAST, const GenContext &genContext)
    {
        auto location = loc(postfixUnaryExpressionAST);

        auto opCode = postfixUnaryExpressionAST->_operator;

        auto expression = postfixUnaryExpressionAST->operand;
        auto result = mlirGen(expression, genContext);
        EXIT_IF_FAILED_OR_NO_VALUE(result)
        auto expressionValue = V(result);

        switch (opCode)
        {
        case SyntaxKind::PlusPlusToken:
        case SyntaxKind::MinusMinusToken:
            return V(builder.create<mlir_ts::PostfixUnaryOp>(location, expressionValue.getType(),
                                                             builder.getI32IntegerAttr((int)opCode), expressionValue));
        default:
            llvm_unreachable("not implemented");
        }
    }

    ValueOrLogicalResult MLIRGenImpl::mlirGen(ConditionalExpression conditionalExpressionAST, const GenContext &genContext)
    {
        auto location = loc(conditionalExpressionAST);

        // condition
        auto condExpression = conditionalExpressionAST->condition;
        auto result = mlirGen(condExpression, genContext);
        EXIT_IF_FAILED_OR_NO_VALUE(result)

        auto condValue = V(result);
        if (condValue.getType() != getBooleanType())
        {
            CAST(condValue, location, getBooleanType(), condValue, genContext);
        }

        // detect value type
        // TODO: sync types for 'when' and 'else'

        auto ifOp = builder.create<mlir_ts::IfOp>(location, mlir::TypeRange{getVoidType()}, condValue, true);

        builder.setInsertionPointToStart(&ifOp.getThenRegion().front());
        auto whenTrueExpression = conditionalExpressionAST->whenTrue;

        ElseSafeCase elseSafeCase;
        mlir::Value resultTrue;
        {
            // check if we do safe-cast here
            SymbolTableScopeT varScope(symbolTable);
            SafeTypesMapScopeT safeTypesMapScope(safeTypesMap);
            checkSafeCast(conditionalExpressionAST->condition, V(result), &elseSafeCase, genContext);
            auto result = mlirGen(whenTrueExpression, genContext);
            if (!genContext.allowPartialResolve)
            {
                EXIT_IF_FAILED_OR_NO_VALUE(result)
            }
            
            resultTrue = V(result);
        }

        builder.setInsertionPointToStart(&ifOp.getElseRegion().front());
        auto whenFalseExpression = conditionalExpressionAST->whenFalse;

        mlir::Value resultFalse;
        {
            SymbolTableScopeT varScope(symbolTable);
            if (elseSafeCase.safeType)
            {
                addSafeCastStatement(elseSafeCase.expr, elseSafeCase.safeType, false, nullptr, genContext);
            }        
            
            auto result2 = mlirGen(whenFalseExpression, genContext);
            if (!genContext.allowPartialResolve)
            {
                EXIT_IF_FAILED_OR_NO_VALUE(result2)
            }

            resultFalse = V(result2);
        }

        if (resultTrue && resultFalse)
        {
            auto defaultUnionType = getUnionType(location, resultTrue.getType(), resultFalse.getType());
            auto merged = false;
            auto resultType = mth.findBaseType(resultTrue.getType(), resultFalse.getType(), merged, defaultUnionType);

            ifOp.getResult(0).setType(resultType);

            CAST_A(falseRes, location, resultType, resultFalse, genContext)
            builder.create<mlir_ts::ResultOp>(location, mlir::ValueRange{falseRes});

            // finish type of IfOp and WhenTrue clause
            builder.setInsertionPointToEnd(&ifOp.getThenRegion().back());

            CAST_A(trueRes, location, resultType, resultTrue, genContext);
            builder.create<mlir_ts::ResultOp>(location, mlir::ValueRange{trueRes});
        }
        else
        {
            // to support partial result
            auto partialResult = resultTrue ? resultTrue : resultFalse;
            if (partialResult)
            {
                ifOp.getResult(0).setType(partialResult.getType());
            }
            else
            {
                return mlir::failure();
            }
        }

        builder.setInsertionPointAfter(ifOp);

        return ifOp.getResult(0);
    }

    ValueOrLogicalResult MLIRGenImpl::mlirGen(BinaryExpression binaryExpressionAST, const GenContext &genContext)
    {
        auto location = loc(binaryExpressionAST);

        auto opCode = (SyntaxKind)binaryExpressionAST->operatorToken;

        auto saveResult = MLIRLogicHelper::isNeededToSaveData(opCode);

        auto leftExpression = binaryExpressionAST->left;
        auto rightExpression = binaryExpressionAST->right;

        if (opCode == SyntaxKind::AmpersandAmpersandToken || opCode == SyntaxKind::BarBarToken)
        {
            return mlirGenAndOrLogic(binaryExpressionAST, genContext, opCode == SyntaxKind::AmpersandAmpersandToken,
                                     saveResult);
        }

        if (opCode == SyntaxKind::QuestionQuestionToken)
        {
            return mlirGenQuestionQuestionLogic(binaryExpressionAST, saveResult, genContext);
        }

        if (opCode == SyntaxKind::InKeyword)
        {
            return mlirGenInLogic(binaryExpressionAST, genContext);
        }

        if (opCode == SyntaxKind::InstanceOfKeyword)
        {
            return mlirGenInstanceOfLogic(binaryExpressionAST, genContext);
        }

        if (opCode == SyntaxKind::EqualsToken)
        {
            return mlirGenSaveLogic(binaryExpressionAST, genContext);
        }

        auto result = mlirGen(leftExpression, genContext);
        if (opCode == SyntaxKind::CommaToken)
        {
            //in case of "comma" op the result of left op can be "nothing"
            EXIT_IF_FAILED(result)
        }
        else
        {
            EXIT_IF_FAILED_OR_NO_VALUE(result)    
        }

        auto leftExpressionValue = V(result);
        auto result2 = mlirGen(rightExpression, genContext);
        EXIT_IF_FAILED_OR_NO_VALUE(result2)
        auto rightExpressionValue = V(result2);

        // check if const expr.
        if (genContext.allowConstEval)
        {
            LLVM_DEBUG(llvm::dbgs() << "Evaluate const: '" << leftExpressionValue << "' and '" << rightExpressionValue << "'\n";);

            auto leftConstOp = dyn_cast<mlir_ts::ConstantOp>(leftExpressionValue.getDefiningOp());
            auto rightConstOp = dyn_cast<mlir_ts::ConstantOp>(rightExpressionValue.getDefiningOp());
            if (leftConstOp && rightConstOp)
            {
                // try to evaluate
                return evaluateBinaryOp(location, opCode, leftConstOp, rightConstOp, genContext);
            }
        }

        auto resultReturnUnions = 
            binaryOpLogicForUnions(location, opCode, leftExpressionValue, rightExpressionValue, genContext);
        if (resultReturnUnions.value || resultReturnUnions.failed())
        {
            return resultReturnUnions;
        }

        auto leftExpressionValueBeforeCast = leftExpressionValue;
        auto rightExpressionValueBeforeCast = rightExpressionValue;

        if (mlir::failed(unwrapForBinaryOp(location, opCode, leftExpressionValue, rightExpressionValue, genContext)))
        {
            return mlir::failure();
        }

        if (mlir::failed(instantiateGenericsForBinaryOp(location, leftExpressionValue, rightExpressionValue, genContext))) 
        {
            return mlir::failure();
        }

        if (mlir::failed(adjustTypesForBinaryOp(location, opCode, leftExpressionValue, rightExpressionValue, genContext)))
        {
            return mlir::failure();
        }

        auto resultReturn = binaryOpLogic(location, opCode, leftExpressionValue, rightExpressionValue, genContext);

        if (saveResult)
        {
            return mlirGenSaveLogicOneItem(location, leftExpressionValueBeforeCast, resultReturn, genContext);
        }

        return resultReturn;
    }

    ValueOrLogicalResult MLIRGenImpl::mlirGen(SpreadElement spreadElement, const GenContext &genContext)
    {
        return mlirGen(spreadElement->expression, genContext);
    }

    ValueOrLogicalResult MLIRGenImpl::mlirGen(ParenthesizedExpression parenthesizedExpression, const GenContext &genContext)
    {
        return mlirGen(parenthesizedExpression->expression, genContext);
    }

    ValueOrLogicalResult MLIRGenImpl::mlirGen(QualifiedName qualifiedName, const GenContext &genContext)
    {
        auto location = loc(qualifiedName);

        auto expression = qualifiedName->left;
        auto result = mlirGenModuleReference(expression, genContext);
        EXIT_IF_FAILED_OR_NO_VALUE(result)
        auto expressionValue = V(result);

        auto name = MLIRHelper::getName(qualifiedName->right);

        return mlirGenPropertyAccessExpression(location, expressionValue, name, genContext);
    }

    ValueOrLogicalResult MLIRGenImpl::mlirGen(PropertyAccessExpression propertyAccessExpression, const GenContext &genContext)
    {
        auto location = loc(propertyAccessExpression);

        auto expression = propertyAccessExpression->expression.as<Expression>();
        auto result = mlirGen(expression, genContext);
        EXIT_IF_FAILED_OR_NO_VALUE(result)
        auto expressionValue = V(result);

        auto namePtr = MLIRHelper::getName(propertyAccessExpression->name, stringAllocator);
        auto propAccessStrRef = mlir::StringRef(print(propertyAccessExpression)).copy(stringAllocator);

        // check if we have safe type mapped value
        auto safeTypedValue = safeTypesMap.lookup({ expressionValue.getType(), propAccessStrRef });
        if (safeTypedValue)
        {
            LLVM_DEBUG(llvm::dbgs() << "\n\t...safe type fieldname: \t " 
                << propAccessStrRef << "." << namePtr << "type: " << expressionValue.getType() << " = " << safeTypedValue;);
            return safeTypedValue;
        }

        return mlirGenPropertyAccessExpression(location, expressionValue, namePtr,
                                               !!propertyAccessExpression->questionDotToken, genContext);
    }

    ValueOrLogicalResult MLIRGenImpl::mlirGen(ElementAccessExpression elementAccessExpression, const GenContext &genContext)
    {
        auto location = loc(elementAccessExpression);

        auto conditinalAccess = !!elementAccessExpression->questionDotToken;

        auto result = mlirGen(elementAccessExpression->expression.as<Expression>(), genContext);
        EXIT_IF_FAILED_OR_NO_VALUE(result)
        auto expression = V(result);

        // default access <array>[index]
        if (!conditinalAccess)
        {
            auto result2 = mlirGen(elementAccessExpression->argumentExpression.as<Expression>(), genContext);
            EXIT_IF_FAILED_OR_NO_VALUE(result2)
            auto argumentExpression = V(result2);

            return mlirGenElementAccess(location, expression, argumentExpression, conditinalAccess, genContext);
        }

        // <array>?.[index] access
        CAST_A(condValue, location, getBooleanType(), expression, genContext);
        return conditionalValue(location, condValue, 
            [&]() { 
                auto result2 = mlirGen(elementAccessExpression->argumentExpression.as<Expression>(), genContext);
                EXIT_IF_FAILED_OR_NO_VALUE(result2)
                auto argumentExpression = V(result2);

                // conditinalAccess should be false here
                auto result3 = mlirGenElementAccess(location, expression, argumentExpression, false, genContext);
                EXIT_IF_FAILED_OR_NO_VALUE(result3)
                auto value = V(result3);

                auto optValue = 
                    isa<mlir_ts::OptionalType>(value.getType())
                        ? value
                        : builder.create<mlir_ts::OptionalValueOp>(location, getOptionalType(value.getType()), value);
                return ValueOrLogicalResult(optValue); 
            }, 
            [&](mlir::Type trueValueType) { 
                auto optUndefValue = builder.create<mlir_ts::OptionalUndefOp>(location, trueValueType);
                return ValueOrLogicalResult(optUndefValue); 
            });
    }

    ValueOrLogicalResult MLIRGenImpl::mlirGen(CallExpression callExpression, const GenContext &genContext)
    {
        auto location = loc(callExpression);

        auto callExpr = callExpression->expression.as<Expression>();

        auto result = mlirGen(callExpr, genContext);
        // in case of detecting value for recursive calls we need to ignore failed calls
        // last condition we need to reduce posobilities to ignore legitimate failure
        // TODO: register dummy function declaration at the begginnning of detecting function output
        if (result.failed_or_no_value() && genContext.allowPartialResolve && 
            (callExpr == SyntaxKind::Identifier || callExpr == SyntaxKind::PropertyAccessExpression))
        {            
            // we need to return success to continue code traversing
            return V(builder.create<mlir_ts::UndefOp>(location, builder.getNoneType()));
        }

        EXIT_IF_FAILED_OR_NO_VALUE(result)
        auto funcResult = V(result);

        LLVM_DEBUG(llvm::dbgs() << "\n!! evaluate function: " << funcResult << "\n";);

        auto funcType = funcResult.getType();
        if (!mth.isAnyFunctionType(funcType) 
            && !mth.isBuiltinFunctionType(funcResult)
            // TODO: do I need to use ConstructFunction instead?
            // to support constructor calls
            && !isa<mlir_ts::ClassType>(funcType)
            // to support super.constructor calls
            && !isa<mlir_ts::ClassStorageType>(funcType))
        {      
            LLVM_DEBUG(llvm::dbgs() << "\n!! function type: " << funcType << "\n";);
            emitError(location, "not a function to call");
            return mlir::failure();
        }

        // so if method is generic and you need to infer types you can cast to generic types
        auto noReceiverTypesForGenericCall = 
            mth.isGenericType(funcResult.getType()) 
            && callExpression->typeArguments.size() == 0;

        SmallVector<mlir::Value, 4> operands;
        auto offsetArgs = isa<mlir_ts::BoundFunctionType>(funcType) || isa<mlir_ts::ExtensionFunctionType>(funcType) ? 1 : 0;
        if (mlir::failed(mlirGenOperands(callExpression->arguments, operands, funcResult.getType(), genContext, offsetArgs, noReceiverTypesForGenericCall)))
        {
            return mlir::failure();
        }

        LLVM_DEBUG(llvm::dbgs() << "\n!! function: [" << funcResult << "] ops: "; for (auto o
                                                                                       : operands) llvm::dbgs()
                                                                                  << "\n param type: " << o.getType();
                   llvm::dbgs() << "\n";);

        return mlirGenCallExpression(location, funcResult, callExpression->typeArguments, operands, genContext);
    }

    ValueOrLogicalResult MLIRGenImpl::mlirGen(NewExpression newExpression, const GenContext &genContext)
    {
        auto location = loc(newExpression);

        // 3 cases, name, index access, method call
        mlir::Type type;
        auto typeExpression = newExpression->expression;
        ////auto isNewArray = typeExpression == SyntaxKind::ElementAccessExpression && newExpression->arguments.isTextRangeEmpty();
        auto result = mlirGen(typeExpression, newExpression->typeArguments, genContext);
        if (result.failed())
        {
            if (typeExpression == SyntaxKind::Identifier)
            {
                // TODO: review it, seems it should be resolved earlier
                auto name = MLIRHelper::getName(typeExpression.as<Identifier>());
                type = findEmbeddedType(location, name, newExpression->typeArguments, genContext);
                if (type)
                {
                    result = V(builder.create<mlir_ts::TypeRefOp>(location, type));
                }
            }
        }

        EXIT_IF_FAILED_OR_NO_VALUE(result)
        auto value = V(result);

        if (auto arrayType = dyn_cast<mlir_ts::ArrayType>(value.getType()))
        {
            return NewArray(location, arrayType, newExpression->arguments, genContext);
        }

#ifdef ARRAY_TYPE_AS_ARRAY_CLASS
        // to support custom Array<T>
        if (auto classType = dyn_cast<mlir_ts::ClassType>(value.getType()))
        {
            if (newExpression->typeArguments > 0 && classType.getName().getValue().starts_with("Array<"))
            {
                auto arrayType = findEmbeddedType(location, "Array", newExpression->typeArguments, genContext);
                if (arrayType)
                {
                    return NewArray(location, arrayType, newExpression->arguments, genContext);
                }
            }
        }
#endif        

        if (auto interfaceType = dyn_cast<mlir_ts::InterfaceType>(value.getType()))
        {
            return NewClassInstanceByCallingNewCtor(location, value, newExpression->arguments, newExpression->typeArguments, genContext);
        }

        if (auto tupleType = dyn_cast<mlir_ts::TupleType>(value.getType()))
        {
            auto newCtorMethod = evaluateProperty(location, value, NEW_CTOR_METHOD_NAME, genContext);
            if (newCtorMethod)
            {
                return NewClassInstanceByCallingNewCtor(location, value, newExpression->arguments, newExpression->typeArguments, genContext);
            }
        }

        // default - class instance
        auto suppressConstructorCall = (newExpression->internalFlags & InternalFlags::SuppressConstructorCall) ==
                                        InternalFlags::SuppressConstructorCall;

        return NewClassInstance(location, value, newExpression->arguments, newExpression->typeArguments, suppressConstructorCall, genContext);
    }

    mlir::LogicalResult MLIRGenImpl::mlirGen(DeleteExpression deleteExpression, const GenContext &genContext)
    {

        auto location = loc(deleteExpression);

        auto result = mlirGen(deleteExpression->expression, genContext);
        EXIT_IF_FAILED_OR_NO_VALUE(result)
        auto expr = V(result);

        if (!isa<mlir_ts::RefType>(expr.getType()) && !isa<mlir_ts::ValueRefType>(expr.getType()) &&
            !isa<mlir_ts::ClassType>(expr.getType()))
        {
            if (auto arrayType = dyn_cast<mlir_ts::ArrayType>(expr.getType()))
            {
                CAST(expr, location, mlir_ts::RefType::get(arrayType.getElementType()), expr, genContext);
            }
            else
            {
                llvm_unreachable("not implemented");
            }
        }

        builder.create<mlir_ts::DeleteOp>(location, expr);

        return mlir::success();
    }

    ValueOrLogicalResult MLIRGenImpl::mlirGen(VoidExpression voidExpression, const GenContext &genContext)
    {

        auto location = loc(voidExpression);

        auto result = mlirGen(voidExpression->expression, genContext);
        EXIT_IF_FAILED_OR_NO_VALUE(result)
        auto expr = V(result);

        auto value = getUndefined(location);

        return value;
    }

    ValueOrLogicalResult MLIRGenImpl::mlirGen(TypeOfExpression typeOfExpression, const GenContext &genContext)
    {
        auto location = loc(typeOfExpression);

        auto result = mlirGen(typeOfExpression->expression, genContext);
        EXIT_IF_FAILED_OR_NO_VALUE(result)
        auto resultValue = V(result);
        // auto typeOfValue = builder.create<mlir_ts::TypeOfOp>(location, getStringType(), resultValue);
        // return V(typeOfValue);

        // needed to use optimizers
        TypeOfOpHelper toh(builder);
        auto typeOfValue = toh.typeOfLogic(location, resultValue, resultValue.getType(), compileOptions);
        return typeOfValue;
    }

    ValueOrLogicalResult MLIRGenImpl::mlirGen(NonNullExpression nonNullExpression, const GenContext &genContext)
    {
        return mlirGen(nonNullExpression->expression, genContext);
    }

    ValueOrLogicalResult MLIRGenImpl::mlirGen(OmittedExpression ommitedExpression, const GenContext &genContext)
    {
        auto location = loc(ommitedExpression);

        return V(builder.create<mlir_ts::UndefOp>(location, getUndefinedType()));
    }

    ValueOrLogicalResult MLIRGenImpl::mlirGen(TemplateLiteralLikeNode templateExpressionAST, const GenContext &genContext)
    {
        auto location = loc(templateExpressionAST);

        auto stringType = getStringType();
        SmallVector<mlir::Value, 4> strs;

        auto text = convertWideToUTF8(templateExpressionAST->head->rawText);
        auto head = builder.create<mlir_ts::ConstantOp>(location, stringType, getStringAttr(text));

        // first string
        strs.push_back(head);
        for (auto span : templateExpressionAST->templateSpans)
        {
            auto expression = span->expression;
            auto result = mlirGen(expression, genContext);
            EXIT_IF_FAILED_OR_NO_VALUE(result)
            auto exprValue = V(result);

            if (exprValue.getType() != stringType)
            {
                CAST(exprValue, location, stringType, exprValue, genContext);
            }

            // expr value
            strs.push_back(exprValue);

            auto spanText = convertWideToUTF8(span->literal->rawText);
            auto spanValue = builder.create<mlir_ts::ConstantOp>(location, stringType, getStringAttr(spanText));

            // text
            strs.push_back(spanValue);
        }

        if (strs.size() <= 1)
        {
            return V(head);
        }

        auto concatValues =
            builder.create<mlir_ts::StringConcatOp>(location, stringType, mlir::ArrayRef<mlir::Value>{strs});

        return V(concatValues);
    }

    ValueOrLogicalResult MLIRGenImpl::mlirGen(TaggedTemplateExpression taggedTemplateExpressionAST, const GenContext &genContext)
    {
        auto location = loc(taggedTemplateExpressionAST);

        auto templateExpressionAST = taggedTemplateExpressionAST->_template;

        SmallVector<mlir::Attribute, 4> strs;
        SmallVector<mlir::Value, 4> vals;

        std::string text = convertWideToUTF8(
            templateExpressionAST->head 
                ? templateExpressionAST->head->rawText 
                : templateExpressionAST->rawText);

        // first string
        strs.push_back(getStringAttr(text));
        for (auto span : templateExpressionAST->templateSpans)
        {
            // expr value
            auto expression = span->expression;
            auto result = mlirGen(expression, genContext);
            EXIT_IF_FAILED_OR_NO_VALUE(result)
            auto exprValue = V(result);

            vals.push_back(exprValue);

            auto spanText = convertWideToUTF8(span->literal->rawText);
            // text
            strs.push_back(getStringAttr(spanText));
        }

        // tag method
        auto arrayAttr = mlir::ArrayAttr::get(builder.getContext(), strs);
        auto constStringArray =
            builder.create<mlir_ts::ConstantOp>(location, getConstArrayType(getStringType(), strs.size()), arrayAttr);

        CAST_A(strArrayValue, location, getArrayType(getStringType()), constStringArray, genContext);

        vals.insert(vals.begin(), strArrayValue);

        auto result = mlirGen(taggedTemplateExpressionAST->tag, genContext);
        EXIT_IF_FAILED_OR_NO_VALUE(result)
        auto callee = V(result);

        if (!mth.isAnyFunctionType(callee.getType()))
        {
            emitError(location, "is not callable");
            return mlir::failure();
        }

        VALIDATE_FUNC(callee.getType(), location)

        auto inputs = mth.getParamsFromFuncRef(callee.getType());

        SmallVector<mlir::Value, 4> operands;

        auto i = 0;
        for (auto value : vals)
        {
            if (inputs.size() <= i)
            {
                emitError(location, "not matching to tag parameters count");
                return mlir::Value();
            }

            if (value.getType() != inputs[i])
            {
                CAST_A(castValue, location, inputs[i], value, genContext);
                operands.push_back(castValue);
            }
            else
            {
                operands.push_back(value);
            }

            i++;
        }

        // call
        auto callIndirectOp = builder.create<mlir_ts::CallIndirectOp>(
            MLIRHelper::getCallSiteLocation(callee, location),
            callee, operands);
        if (callIndirectOp.getNumResults() > 0)
        {
            return callIndirectOp.getResult(0);
        }

        return mlir::success();
    }

    ValueOrLogicalResult MLIRGenImpl::mlirGen(NullLiteral nullLiteral, const GenContext &genContext)
    {
        return V(builder.create<mlir_ts::NullOp>(loc(nullLiteral), getNullType()));
    }

    ValueOrLogicalResult MLIRGenImpl::mlirGen(TrueLiteral trueLiteral, const GenContext &genContext)
    {
        return mlirGenBooleanValue(loc(trueLiteral), true);
    }

    ValueOrLogicalResult MLIRGenImpl::mlirGen(FalseLiteral falseLiteral, const GenContext &genContext)
    {
        return mlirGenBooleanValue(loc(falseLiteral), false);
    }

    ValueOrLogicalResult MLIRGenImpl::mlirGen(NumericLiteral numericLiteral, const GenContext &genContext)
    {
        auto attrVal = getNumericLiteralAttribute(numericLiteral);
        auto attrType = mlir::cast<mlir::TypedAttr>(attrVal).getType();
        auto valueType = isa<mlir::FloatType>(attrType) ? getNumberType() : attrType;
        auto literalType = mlir_ts::LiteralType::get(attrVal, valueType);
        return V(builder.create<mlir_ts::ConstantOp>(loc(numericLiteral), literalType, attrVal));
    }

    ValueOrLogicalResult MLIRGenImpl::mlirGen(BigIntLiteral bigIntLiteral, const GenContext &genContext)
    {
        APSInt newVal(wstos(
            *(bigIntLiteral->text.end() - 1) == S('n') 
                ? bigIntLiteral->text.substr(0, bigIntLiteral->text.length() - 1) 
                : bigIntLiteral->text.c_str()));
        auto type = builder.getI64Type();
        auto attrVal = mlir::IntegerAttr::get(type, newVal.getExtValue());
        auto literalType = mlir_ts::LiteralType::get(attrVal, type);
        return V(builder.create<mlir_ts::ConstantOp>(loc(bigIntLiteral), literalType, attrVal));
    }

    ValueOrLogicalResult MLIRGenImpl::mlirGen(ts::StringLiteral stringLiteral, const GenContext &genContext)
    {
        auto text = convertWideToUTF8(stringLiteral->text);
        return mlirGenStringValue(loc(stringLiteral), text);
    }

    ValueOrLogicalResult MLIRGenImpl::mlirGen(ts::RegularExpressionLiteral regularExpressionLiteral, const GenContext &genContext)
    {
        NodeFactory nf(NodeFactoryFlags::None);

        auto regName = nf.createIdentifier(S("RegExp"));

        auto begin = regularExpressionLiteral->text.find_first_of('/');
        auto end = regularExpressionLiteral->text.find_last_of('/');
        auto text = regularExpressionLiteral->text.substr(begin + 1, end - 1);
        auto flags = regularExpressionLiteral->text.substr(end + 1);

        NodeArray<Expression> argumentsArray;
        argumentsArray.push_back(
            nf.createStringLiteral(
                text, 
                false, 
                regularExpressionLiteral->hasExtendedUnicodeEscape));
        argumentsArray.push_back(
            nf.createStringLiteral(
                flags, 
                false, 
                regularExpressionLiteral->hasExtendedUnicodeEscape));

        auto newRegExpr = nf.createNewExpression(regName, undefined, argumentsArray);

        LLVM_DEBUG(printDebug(newRegExpr););

        return mlirGen(newRegExpr, genContext);
    }

    ValueOrLogicalResult MLIRGenImpl::mlirGen(ts::NoSubstitutionTemplateLiteral noSubstitutionTemplateLiteral,
                                 const GenContext &genContext)
    {
        auto text = convertWideToUTF8(noSubstitutionTemplateLiteral->text);

        auto attrVal = getStringAttr(text);
        auto literalType = mlir_ts::LiteralType::get(attrVal, getStringType());
        return V(builder.create<mlir_ts::ConstantOp>(loc(noSubstitutionTemplateLiteral), literalType, attrVal));
    }

    ValueOrLogicalResult MLIRGenImpl::mlirGen(ts::ArrayLiteralExpression arrayLiteral, const GenContext &genContext)
    {
        auto location = loc(arrayLiteral);

        SmallVector<ArrayElement> values;
        struct ArrayInfo arrayInfo{};
        if (mlir::failed(processArrayValues(arrayLiteral->elements, values, arrayInfo, genContext)))
        {
            return mlir::failure();
        }

        return createArrayFromArrayInfo(location, values, arrayInfo, genContext);
    }

    ValueOrLogicalResult MLIRGenImpl::mlirGen(ts::ObjectLiteralExpression objectLiteral, const GenContext &genContext)
    {
        auto location = loc(objectLiteral);

        ObjectLiteralInfo oli{};
        oli.objectLiteral = objectLiteral;

        oli.receiverType = genContext.receiverType;
        if (oli.receiverType)
        {
            oli.receiverType = mth.stripOptionalType(oli.receiverType);

            LLVM_DEBUG(llvm::dbgs() << "\n!! Recevier type: " << oli.receiverType << "\n";);

            if ((isa<mlir_ts::TupleType>(oli.receiverType) || isa<mlir_ts::ConstTupleType>(oli.receiverType) || isa<mlir_ts::InterfaceType>(oli.receiverType))
                 && objectLiteral->properties.size() == 0)
            {
                // return undef tuple
                llvm::SmallVector<mlir_ts::FieldInfo> destTupleFields;
                if (mlir::succeeded(mth.getFields(oli.receiverType, destTupleFields)))
                {
                    auto tupleType = getTupleType(destTupleFields);
                    return V(builder.create<mlir_ts::UndefOp>(location, tupleType));
                }
            }
        }

        // Object This Type
        auto name = MLIRHelper::getAnonymousName(loc_check(objectLiteral), ".obj", getFullNamespaceName());
        auto objectNameSymbol = mlir::FlatSymbolRefAttr::get(builder.getContext(), name);
        auto objectStorageType = getObjectStorageType(objectNameSymbol);
        oli.objThis = getObjectType(objectStorageType);

        // add all fields
        if (auto earlyResult = mlirGenObjectLiteralFields(location, oli, genContext))
        {
            return *earlyResult;
        }

        // update after processing all fields
        objectStorageType.setFields(oli.fieldInfos);

        if (mlir::failed(mlirGenObjectLiteralMethodPrototypes(oli, genContext)))
        {
            return mlir::failure();
        }

        if (mlir::failed(mlirGenObjectLiteralCaptures(location, oli, genContext)))
        {
            return mlir::failure();
        }

        // final type, update
        objectStorageType.setFields(oli.fieldInfos);

        if (mlir::failed(mlirGenObjectLiteralMethodBodies(oli, genContext)))
        {
            return mlir::failure();
        }

        auto constTupleTypeWithReplacedThis = getConstTupleType(oli.fieldInfos);

        auto arrayAttr = mlir::ArrayAttr::get(builder.getContext(), oli.values);
        auto constantVal =
            builder.create<mlir_ts::ConstantOp>(location, constTupleTypeWithReplacedThis, arrayAttr);
        if (oli.fieldsToSet.empty())
        {
            return V(constantVal);
        }

        auto tupleType = mth.convertConstTupleTypeToTupleType(constantVal.getType());
        auto tupleValue = mlirGenCreateTuple(location, tupleType, constantVal, oli.fieldsToSet, genContext);
        return V(tupleValue);
    }

    ValueOrLogicalResult MLIRGenImpl::mlirGen(Identifier identifier, const GenContext &genContext)
    {
        auto location = loc(identifier);

        // resolve name
        auto name = MLIRHelper::getName(identifier);

        // info: can't validate it here, in case of "print" etc
        return mlirGen(location, name, genContext);
    }

    ValueOrLogicalResult MLIRGenImpl::mlirGen(mlir::Location location, StringRef name, const GenContext &genContext)
    {
        auto value = resolveIdentifier(location, name, genContext);
        if (value)
        {
            return value;
        }

        if (MLIRCustomMethods::isInternalFunctionName(compileOptions, name))
        {
            auto symbOp = builder.create<mlir_ts::SymbolRefOp>(
                location, builder.getNoneType(), mlir::FlatSymbolRefAttr::get(builder.getContext(), name));
            symbOp->setAttr(BUILTIN_FUNC_ATTR_NAME, mlir::BoolAttr::get(builder.getContext(), true));
            return V(symbOp);
        }

        if (MLIRCustomMethods::isInternalObjectName(name))
        {
            mlir::Type type;

            if (name == "Symbol")
            {
                type = getSymbolType();
            }
            else
            {
                type = builder.getNoneType();
            }

            // set correct type
            auto symbOp = builder.create<mlir_ts::SymbolRefOp>(
                location, type, mlir::FlatSymbolRefAttr::get(builder.getContext(), name));            
            return V(symbOp);
        }

        // TODO: error, when we use  function_name(index: index) and index value is not provided in call function_name(index), index will be mistakenly tearted
        // as embeded type "index"
        if (!isEmbededType(name))
            emitError(location, "can't resolve name: ") << name;

        return mlir::failure();
    }

    ValueOrLogicalResult MLIRGenImpl::mlirGen(ClassExpression classExpressionAST, const GenContext &genContext)
    {
        std::string fullName;

        // go to root
        {
            mlir::OpBuilder::InsertionGuard guard(builder);
            builder.setInsertionPointToStart(theModule.getBody());

            auto [result, fullNameRet] = mlirGen(classExpressionAST.as<ClassLikeDeclaration>(), genContext);
            if (mlir::failed(result))
            {
                return mlir::failure();
            }

            fullName = fullNameRet;
        }

        auto location = loc(classExpressionAST);

        auto classInfo = getClassInfoByFullName(fullName);
        if (classInfo)
        {
            if (classInfo->isDeclaration)
            {
                auto undefClass = builder.create<mlir_ts::UndefOp>(location, classInfo->classType);
                return V(undefClass);
            }
            else
            {
                auto classValue = builder.create<mlir_ts::ClassRefOp>(
                    location, classInfo->classType,
                    mlir::FlatSymbolRefAttr::get(builder.getContext(), classInfo->classType.getName().getValue()));

                // TODO: find out if you need to pass generics info, typeParams + typeArgs
                return NewClassInstance(location, classValue, undefined, undefined, false, genContext);
            }
        }

        return mlir::failure();
    }

} // namespace mlirgen
} // namespace typescript
