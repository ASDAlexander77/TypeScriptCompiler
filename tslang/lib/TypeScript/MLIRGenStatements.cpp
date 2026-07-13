// Statement-level code generation methods of MLIRGenImpl (see MLIRGenImpl.h).

#include "MLIRGenImpl.h"

#include "TypeScript/MLIRLogic/MLIRRTTIHelperVC.h"

#include "mlir/Dialect/Async/IR/Async.h"

#undef DEBUG_TYPE
#define DEBUG_TYPE "mlir"

namespace typescript
{
namespace mlirgen
{

    mlir::LogicalResult MLIRGenImpl::mlirGenBody(Node body, const GenContext &genContext)
    {
        auto kind = (SyntaxKind)body;
        if (kind == SyntaxKind::Block)
        {
            return mlirGen(body.as<ts::Block>(), genContext);
        }

        if (kind == SyntaxKind::ModuleBlock)
        {
            return mlirGen(body.as<ModuleBlock>(), genContext);
        }

        if (isStatement(body))
        {
            return mlirGen(body.as<Statement>(), genContext);
        }

        if (isExpression(body))
        {
            auto result = mlirGen(body.as<Expression>(), genContext);
            EXIT_IF_FAILED(result)
            auto resultValue = V(result);
            if (resultValue)
            {
                return mlirGenReturnValue(loc(body), resultValue, false, genContext);
            }

            builder.create<mlir_ts::ReturnOp>(loc(body));
            return mlir::success();
        }

        llvm_unreachable("unknown body type");
    }

    void MLIRGenImpl::clearState(NodeArray<Statement> statements)
    {
        for (auto &statement : statements)
        {
            statement->processed = false;
        }
    }

    mlir::LogicalResult MLIRGenImpl::mlirGen(NodeArray<Statement> statements, const GenContext &genContext)
    {
        SymbolTableScopeT varScope(symbolTable);

        clearState(statements);

        auto notResolved = 0;
        do
        {
            auto noErrorLocation = true;
            mlir::Location errorLocation = mlir::UnknownLoc::get(builder.getContext());
            auto lastTimeNotResolved = notResolved;
            notResolved = 0;
            for (auto &statement : statements)
            {
                if (statement->processed)
                {
                    continue;
                }

                if (failed(mlirGen(statement, genContext)))
                {
                    if (noErrorLocation)
                    {
                        errorLocation = loc(statement);
                        noErrorLocation = false;
                    }

                    notResolved++;
                }
                else
                {
                    statement->processed = true;
                }
            }

            // repeat if not all resolved
            if (lastTimeNotResolved > 0 && lastTimeNotResolved == notResolved)
            {
                // class can depends on other class declarations
                emitError(errorLocation, "can't resolve dependencies in namespace");
                return mlir::failure();
            }
        } while (notResolved > 0);

        // clear states to be able to run second time
        clearState(statements);

        return mlir::success();
    }

    mlir::LogicalResult MLIRGenImpl::mlirGen(ts::Block blockAST, const GenContext &genContext, int skipStatements)
    {
        auto location = loc(blockAST);

        SymbolTableScopeT varScope(symbolTable);
        GenContext genContextUsing(genContext);
        genContextUsing.parentBlockContext = &genContext;

        DITableScopeT debugBlockScope(debugScope);
        if (compileOptions.generateDebugInfo && !blockAST->parent)
        {
            MLIRDebugInfoHelper mdi(builder, debugScope);
            mdi.setLexicalBlock(location);
        }

        auto usingVars = std::make_unique<SmallVector<ts::VariableDeclarationDOM::TypePtr>>();
        genContextUsing.usingVars = usingVars.get();

        EXIT_IF_FAILED(mlirGenNoScopeVarsAndDisposable(blockAST, genContextUsing, skipStatements));

        // we need to call dispose for those which are in "using"
        // default value for genContext.cleanUpUsingVarsFlag = CurrentScope
        EXIT_IF_FAILED(mlirGenDisposable(location, DisposeDepth::CurrentScope, {}, &genContextUsing));

        return mlir::success();
    }

    mlir::LogicalResult MLIRGenImpl::mlirGen(Statement statementAST, const GenContext &genContext)
    {
        auto kind = (SyntaxKind)statementAST;
        if (kind == SyntaxKind::FunctionDeclaration)
        {
            return mlirGen(statementAST.as<FunctionDeclaration>(), genContext);
        }
        else if (kind == SyntaxKind::ExpressionStatement)
        {
            return mlirGen(statementAST.as<ExpressionStatement>(), genContext);
        }
        else if (kind == SyntaxKind::VariableStatement)
        {
            return mlirGen(statementAST.as<VariableStatement>(), genContext);
        }
        else if (kind == SyntaxKind::IfStatement)
        {
            return mlirGen(statementAST.as<IfStatement>(), genContext);
        }
        else if (kind == SyntaxKind::ReturnStatement)
        {
            return mlirGen(statementAST.as<ReturnStatement>(), genContext);
        }
        else if (kind == SyntaxKind::LabeledStatement)
        {
            return mlirGen(statementAST.as<LabeledStatement>(), genContext);
        }
        else if (kind == SyntaxKind::DoStatement)
        {
            return mlirGen(statementAST.as<DoStatement>(), genContext);
        }
        else if (kind == SyntaxKind::WhileStatement)
        {
            return mlirGen(statementAST.as<WhileStatement>(), genContext);
        }
        else if (kind == SyntaxKind::ForStatement)
        {
            return mlirGen(statementAST.as<ForStatement>(), genContext);
        }
        else if (kind == SyntaxKind::ForInStatement)
        {
            return mlirGen(statementAST.as<ForInStatement>(), genContext);
        }
        else if (kind == SyntaxKind::ForOfStatement)
        {
            return mlirGen(statementAST.as<ForOfStatement>(), genContext);
        }
        else if (kind == SyntaxKind::ContinueStatement)
        {
            return mlirGen(statementAST.as<ContinueStatement>(), genContext);
        }
        else if (kind == SyntaxKind::BreakStatement)
        {
            return mlirGen(statementAST.as<BreakStatement>(), genContext);
        }
        else if (kind == SyntaxKind::SwitchStatement)
        {
            return mlirGen(statementAST.as<SwitchStatement>(), genContext);
        }
        else if (kind == SyntaxKind::ThrowStatement)
        {
            return mlirGen(statementAST.as<ThrowStatement>(), genContext);
        }
        else if (kind == SyntaxKind::TryStatement)
        {
            return mlirGen(statementAST.as<TryStatement>(), genContext);
        }
        else if (kind == SyntaxKind::TypeAliasDeclaration)
        {
            // declaration
            return mlirGen(statementAST.as<TypeAliasDeclaration>(), genContext);
        }
        else if (kind == SyntaxKind::Block)
        {
            return mlirGen(statementAST.as<ts::Block>(), genContext);
        }
        else if (kind == SyntaxKind::EnumDeclaration)
        {
            // declaration
            return mlirGen(statementAST.as<EnumDeclaration>(), genContext);
        }
        else if (kind == SyntaxKind::ClassDeclaration)
        {
            // declaration
            return mlirGen(statementAST.as<ClassDeclaration>(), genContext);
        }
        else if (kind == SyntaxKind::InterfaceDeclaration) 
        {
            // declaration
            return mlirGen(statementAST.as<InterfaceDeclaration>(), genContext);
        }
        else if (kind == SyntaxKind::ImportEqualsDeclaration)
        {
            // declaration
            return mlirGen(statementAST.as<ImportEqualsDeclaration>(), genContext);
        }
        else if (kind == SyntaxKind::ImportDeclaration)
        {
            // declaration
            return mlirGen(statementAST.as<ImportDeclaration>(), genContext);
        }
        else if (kind == SyntaxKind::ModuleDeclaration)
        {
            return mlirGen(statementAST.as<ModuleDeclaration>(), genContext);
        }
        else if (kind == SyntaxKind::DebuggerStatement)
        {
            return mlirGen(statementAST.as<DebuggerStatement>(), genContext);
        }
        else if (kind == SyntaxKind::EmptyStatement ||
                 kind == SyntaxKind::Unknown /*TODO: temp solution to treat null statements as empty*/)
        {
            return mlir::success();
        }

        llvm_unreachable("unknown statement type");
    }

    mlir::LogicalResult MLIRGenImpl::mlirGen(ExpressionStatement expressionStatementAST, const GenContext &genContext)
    {
        auto result = mlirGen(expressionStatementAST->expression, genContext);
        EXIT_IF_FAILED(result)
        return mlir::success();
    }

    mlir::LogicalResult MLIRGenImpl::mlirGen(VariableStatement variableStatementAST, const GenContext &genContext)
    {
        // we need it for support "export" keyword
        variableStatementAST->declarationList->parent = variableStatementAST;
        return mlirGen(variableStatementAST->declarationList, genContext);
    }

    mlir::LogicalResult MLIRGenImpl::mlirGen(ReturnStatement returnStatementAST, const GenContext &genContext)
    {
        auto location = loc(returnStatementAST);
        if (auto expression = returnStatementAST->expression)
        {
            GenContext receiverTypeGenContext(genContext);
            receiverTypeGenContext.clearReceiverTypes();
            auto exactReturnType = getExplicitReturnTypeOfCurrentFunction(genContext);
            if (exactReturnType)
            {
                receiverTypeGenContext.receiverType = exactReturnType;
            }

            auto result = mlirGen(expression, receiverTypeGenContext);
            EXIT_IF_FAILED(result)
            
            auto expressionValue = V(result);
            if (!expressionValue)
            {
                emitError(location, "No return value");
            }

            if (!genContext.allowPartialResolve)
            {
                VALIDATE(expressionValue, location)
            }

            EXIT_IF_FAILED(mlirGenDisposable(location, DisposeDepth::FullStack, {}, &genContext));

            return mlirGenReturnValue(location, expressionValue, false, genContext);
        }

        EXIT_IF_FAILED(mlirGenDisposable(location, DisposeDepth::FullStack, {}, &genContext));

        builder.create<mlir_ts::ReturnOp>(location);
        return mlir::success();
    }

    mlir::LogicalResult MLIRGenImpl::mlirGen(IfStatement ifStatementAST, const GenContext &genContext)
    {
        auto location = loc(ifStatementAST);

        auto hasElse = !!ifStatementAST->elseStatement;

        // condition
        auto result = mlirGen(ifStatementAST->expression, genContext);
        EXIT_IF_FAILED_OR_NO_VALUE(result)
        auto condValue = V(result);

        // special case: in case of LiteralValue do not process If value is False
        std::optional<bool> literalValue;
        if (auto litType = mlir::dyn_cast<mlir_ts::LiteralType>(condValue.getType()))
        {
            if (auto boolVal = mlir::dyn_cast<mlir::BoolAttr>(litType.getValue()))
            {
                literalValue = boolVal.getValue();
            }
        }

        // default implementation of IfOp
        if (condValue.getType() != getBooleanType())
        {
            CAST(condValue, location, getBooleanType(), condValue, genContext);
        }

        auto ifOp = builder.create<mlir_ts::IfOp>(location, condValue, hasElse);

        builder.setInsertionPointToStart(&ifOp.getThenRegion().front());

        ElseSafeCase elseSafeCase{};
        {
            // check if we do safe-cast here
            SymbolTableScopeT varScope(symbolTable);
            SafeTypesMapScopeT safeTypesMapScope(safeTypesMap);
            checkSafeCast(ifStatementAST->expression, V(result), hasElse ? &elseSafeCase : nullptr, genContext);

            auto processIf = !literalValue.has_value() || literalValue.value();
            if (processIf)
            {
                auto result = mlirGen(ifStatementAST->thenStatement, genContext);
                EXIT_IF_FAILED(result)
            }
        }

        if (hasElse)
        {
            builder.setInsertionPointToStart(&ifOp.getElseRegion().front());
            SymbolTableScopeT varScope(symbolTable);
            if (elseSafeCase.safeType)
            {
                // add case statement
                addSafeCastStatement(elseSafeCase.expr, elseSafeCase.safeType, false, nullptr, genContext);
            }

            auto processIf = !literalValue.has_value() || !literalValue.value();
            if (processIf)
            {
                auto result = mlirGen(ifStatementAST->elseStatement, genContext);
                EXIT_IF_FAILED(result)
            }
        }

        builder.setInsertionPointAfter(ifOp);

        return mlir::success();
    }

    mlir::LogicalResult MLIRGenImpl::mlirGen(DoStatement doStatementAST, const GenContext &genContext)
    {
        SymbolTableScopeT varScope(symbolTable);

        auto location = loc(doStatementAST);

        SmallVector<mlir::Type, 0> types;
        SmallVector<mlir::Value, 0> operands;

        auto doWhileOp = builder.create<mlir_ts::DoWhileOp>(location, types, operands);
        if (!label.empty())
        {
            doWhileOp->setAttr(LABEL_ATTR_NAME, builder.getStringAttr(label));
            label = "";
        }

        GenContext loopGenContext(genContext);
        loopGenContext.isLoop = true;
        loopGenContext.loopLabel = label;

        /*auto *cond =*/builder.createBlock(&doWhileOp.getCond(), {}, types);
        /*auto *body =*/builder.createBlock(&doWhileOp.getBody(), {}, types);

        // body in condition
        builder.setInsertionPointToStart(&doWhileOp.getBody().front());
        auto result2 = mlirGen(doStatementAST->statement, loopGenContext);
        EXIT_IF_FAILED(result2)
        // just simple return, as body in cond
        builder.create<mlir_ts::ResultOp>(location);

        builder.setInsertionPointToStart(&doWhileOp.getCond().front());
        auto result = mlirGen(doStatementAST->expression, loopGenContext);
        EXIT_IF_FAILED(result)
        auto conditionValue = V(result);

        if (conditionValue.getType() != getBooleanType())
        {
            CAST(conditionValue, location, getBooleanType(), conditionValue, loopGenContext);
        }

        builder.create<mlir_ts::ConditionOp>(location, conditionValue, mlir::ValueRange{});

        builder.setInsertionPointAfter(doWhileOp);
        return mlir::success();
    }

    mlir::LogicalResult MLIRGenImpl::mlirGen(WhileStatement whileStatementAST, const GenContext &genContext)
    {
        SymbolTableScopeT varScope(symbolTable);

        auto location = loc(whileStatementAST);

        SmallVector<mlir::Type, 0> types;
        SmallVector<mlir::Value, 0> operands;

        auto whileOp = builder.create<mlir_ts::WhileOp>(location, types, operands);
        if (!label.empty())
        {
            whileOp->setAttr(LABEL_ATTR_NAME, builder.getStringAttr(label));
            label = "";
        }

        GenContext loopGenContext(genContext);
        loopGenContext.isLoop = true;
        loopGenContext.loopLabel = label;

        /*auto *cond =*/builder.createBlock(&whileOp.getCond(), {}, types);
        /*auto *body =*/builder.createBlock(&whileOp.getBody(), {}, types);

        // condition
        builder.setInsertionPointToStart(&whileOp.getCond().front());
        auto result = mlirGen(whileStatementAST->expression, loopGenContext);
        EXIT_IF_FAILED_OR_NO_VALUE(result)
        auto conditionValue = V(result);

        if (conditionValue.getType() != getBooleanType())
        {
            CAST(conditionValue, location, getBooleanType(), conditionValue, loopGenContext);
        }

        builder.create<mlir_ts::ConditionOp>(location, conditionValue, mlir::ValueRange{});

        // body
        builder.setInsertionPointToStart(&whileOp.getBody().front());

        // check if we do safe-cast here
        SymbolTableScopeT varScopeBody(symbolTable);
        SafeTypesMapScopeT safeTypesMapScope(safeTypesMap);
        checkSafeCast(whileStatementAST->expression, conditionValue, nullptr, loopGenContext);

        auto result2 = mlirGen(whileStatementAST->statement, loopGenContext);
        EXIT_IF_FAILED(result2)
        builder.create<mlir_ts::ResultOp>(location);

        builder.setInsertionPointAfter(whileOp);
        return mlir::success();
    }

    mlir::LogicalResult MLIRGenImpl::mlirGen(ForStatement forStatementAST, const GenContext &genContext)
    {
        SymbolTableScopeT varScope(symbolTable);

        auto location = loc(forStatementAST);

        auto hasAwait = InternalFlags::ForAwait == (forStatementAST->internalFlags & InternalFlags::ForAwait);

        // initializer
        // TODO: why do we have ForInitialier
        if (isExpression(forStatementAST->initializer))
        {
            auto result = mlirGen(forStatementAST->initializer.as<Expression>(), genContext);
            EXIT_IF_FAILED_OR_NO_VALUE(result)
            auto init = V(result);
            if (!init)
            {
                return mlir::failure();
            }
        }
        else if (forStatementAST->initializer == SyntaxKind::VariableDeclarationList)
        {
            auto result = mlirGen(forStatementAST->initializer.as<VariableDeclarationList>(), genContext);
            EXIT_IF_FAILED(result)
            if (failed(result))
            {
                return result;
            }
        }

        SmallVector<mlir::Type, 0> types;
        SmallVector<mlir::Value, 0> operands;

        mlir::Value asyncGroupResult;
        if (hasAwait)
        {
            auto groupType = mlir::async::GroupType::get(builder.getContext());
            auto blockSize = builder.create<mlir_ts::ConstantOp>(location, builder.getIndexAttr(0));
            auto asyncGroupOp = builder.create<mlir::async::CreateGroupOp>(location, groupType, blockSize);
            asyncGroupResult = asyncGroupOp.getResult();
            // operands.push_back(asyncGroupOp);
            // types.push_back(groupType);
        }

        auto forOp = builder.create<mlir_ts::ForOp>(location, types, operands);
        if (!label.empty())
        {
            forOp->setAttr(LABEL_ATTR_NAME, builder.getStringAttr(label));
            label = "";
        }

        GenContext loopGenContext(genContext);
        loopGenContext.isLoop = true;
        loopGenContext.loopLabel = label;

        /*auto *cond =*/builder.createBlock(&forOp.getCond(), {}, types);
        /*auto *body =*/builder.createBlock(&forOp.getBody(), {}, types);
        /*auto *incr =*/builder.createBlock(&forOp.getIncr(), {}, types);

        builder.setInsertionPointToStart(&forOp.getCond().front());
        auto result = mlirGen(forStatementAST->condition, loopGenContext);
        EXIT_IF_FAILED(result)
        auto conditionValue = V(result);
        if (conditionValue)
        {
            builder.create<mlir_ts::ConditionOp>(location, conditionValue, mlir::ValueRange{});
        }
        else
        {
            builder.create<mlir_ts::NoConditionOp>(location, mlir::ValueRange{});
        }

        // body
        builder.setInsertionPointToStart(&forOp.getBody().front());
        if (hasAwait)
        {
            if (forStatementAST->statement == SyntaxKind::Block)
            {
                auto firstStatement = forStatementAST->statement.as<ts::Block>()->statements.front();
                auto result = mlirGen(firstStatement, loopGenContext);
                EXIT_IF_FAILED(result)
            }

            // TODO: we need to strip metadata to fix issue with debug info
            // async body
            auto isFailed = false;
            auto asyncExecOp = builder.create<mlir::async::ExecuteOp>(
                stripMetadata(location), mlir::TypeRange{}, mlir::ValueRange{}, mlir::ValueRange{},
                [&](mlir::OpBuilder &builder, mlir::Location location, mlir::ValueRange values) {
                    GenContext execOpBodyGenContext(loopGenContext);
                    DITableScopeT debugAsyncCodeScope(debugScope);
                    MLIRDebugInfoHelper mdi(builder, debugScope);
                    
                    // TODO: temp hack to break wrong chain on scopes because 'await' create extra function wrap
                    mdi.clearDebugScope();
                    mdi.setLexicalBlock(location);

                    if (forStatementAST->statement == SyntaxKind::Block)
                    {
                        if (mlir::failed(mlirGen(forStatementAST->statement.as<ts::Block>(), execOpBodyGenContext, 1)))
                        {
                            isFailed = true;
                        }
                    }
                    else
                    {
                        if (mlir::failed(mlirGen(forStatementAST->statement, execOpBodyGenContext))) 
                        {
                            isFailed = true;
                        }
                    }

                    builder.create<mlir::async::YieldOp>(location, mlir::ValueRange{});
                });    

            if (isFailed)
            {
                return mlir::failure();
            }

            // add to group
            auto rankType = mlir::IndexType::get(builder.getContext());
            // TODO: should i replace with value from arg0?
            builder.create<mlir::async::AddToGroupOp>(location, rankType, asyncExecOp.getToken(), asyncGroupResult);
        }
        else
        {
            // default
            auto result = mlirGen(forStatementAST->statement, loopGenContext);
            EXIT_IF_FAILED(result)
        }

        builder.create<mlir_ts::ResultOp>(location);

        // increment
        builder.setInsertionPointToStart(&forOp.getIncr().front());
        mlirGen(forStatementAST->incrementor, loopGenContext);
        builder.create<mlir_ts::ResultOp>(location);

        builder.setInsertionPointAfter(forOp);

        if (hasAwait)
        {
            // Not helping
            /*
            // async await all, see convert-to-llvm.mlir
            auto asyncExecAwaitAllOp =
                builder.create<mlir::async::ExecuteOp>(location, mlir::TypeRange{}, mlir::ValueRange{},
            mlir::ValueRange{},
                                                       [&](mlir::OpBuilder &builder, mlir::Location location,
            mlir::ValueRange values) { builder.create<mlir::async::AwaitAllOp>(location, asyncGroupResult);
                                                           builder.create<mlir::async::YieldOp>(location,
            mlir::ValueRange{});
                                                       });
            */

            // Wait for the completion of all subtasks.
            builder.create<mlir::async::AwaitAllOp>(location, asyncGroupResult);
        }

        return mlir::success();
    }

    mlir::LogicalResult MLIRGenImpl::mlirGen(ForInStatement forInStatementAST, const GenContext &genContext)
    {
        SymbolTableScopeT varScope(symbolTable);

        auto location = loc(forInStatementAST);

        NodeFactory nf(NodeFactoryFlags::None);

        // init
        NodeArray<VariableDeclaration> declarations;
        auto _i = nf.createIdentifier(S(".i"));
        declarations.push_back(nf.createVariableDeclaration(_i, undefined, undefined, nf.createNumericLiteral(S("0"))));

        auto _a = nf.createIdentifier(S(".a"));
        auto arrayVar = nf.createVariableDeclaration(_a, undefined, undefined, forInStatementAST->expression);
        arrayVar->internalFlags |= InternalFlags::ForceConstRef;
        declarations.push_back(arrayVar);

        auto initVars = nf.createVariableDeclarationList(declarations, NodeFlags::Let);

        // condition
        // auto cond = nf.createBinaryExpression(_i, nf.createToken(SyntaxKind::LessThanToken),
        // nf.createCallExpression(nf.createIdentifier(S("#_last_field")), undefined, NodeArray<Expression>(_a)));
        auto cond = nf.createBinaryExpression(_i, nf.createToken(SyntaxKind::LessThanToken),
                                              nf.createPropertyAccessExpression(_a, nf.createIdentifier(S(LENGTH_FIELD_NAME))));

        // incr
        auto incr = nf.createPrefixUnaryExpression(nf.createToken(SyntaxKind::PlusPlusToken), _i);

        // block
        NodeArray<ts::Statement> statements;

        auto varDeclList = forInStatementAST->initializer.as<VariableDeclarationList>();
        varDeclList->declarations.front()->initializer = _i;

        statements.push_back(nf.createVariableStatement(undefined, varDeclList));
        statements.push_back(forInStatementAST->statement);
        auto block = nf.createBlock(statements);

        // final For statement
        auto forStatNode = nf.createForStatement(initVars, cond, incr, block);

        return mlirGen(forStatNode, genContext);
    }

    mlir::LogicalResult MLIRGenImpl::mlirGen(ForOfStatement forOfStatementAST, const GenContext &genContext)
    {
        auto location = loc(forOfStatementAST);

        auto result = mlirGen(forOfStatementAST->expression, genContext);
        EXIT_IF_FAILED_OR_NO_VALUE(result)
        auto exprValue = V(result);

        auto skip = isa<mlir_ts::ArrayType>(exprValue.getType()) 
                 || isa<mlir_ts::StringType>(exprValue.getType());
        // we need to ignore SYMBOL_ITERATOR for array to use simplier method and do not cause the stackoverflow
        if (!skip)
        {
            auto iteratorIdent = (forOfStatementAST->awaitModifier) ? SYMBOL_ASYNC_ITERATOR : SYMBOL_ITERATOR;
            if (auto iteratorType = evaluateProperty(location, exprValue, iteratorIdent, genContext))
            {
                if (auto iteratorValue = mlirGenCallThisMethod(location, exprValue, iteratorIdent, undefined, undefined, genContext))
                {
                    exprValue = V(iteratorValue);
                }
            }

            auto propertyType = evaluateProperty(location, exprValue, ITERATOR_NEXT, genContext);
            if (propertyType)
            {
                if (mlir::succeeded(mlirGenES2015(forOfStatementAST, exprValue, genContext)))
                {
                    return mlir::success();
                }
            }
        }

        return mlirGenES3(forOfStatementAST, exprValue, genContext);
    }

    mlir::LogicalResult MLIRGenImpl::mlirGen(LabeledStatement labeledStatementAST, const GenContext &genContext)
    {
        auto location = loc(labeledStatementAST);

        label = MLIRHelper::getName(labeledStatementAST->label);

        auto kind = (SyntaxKind)labeledStatementAST->statement;
        if (kind == SyntaxKind::EmptyStatement && StringRef(label).starts_with(GENERATOR_STATELABELPREFIX))
        {
            builder.create<mlir_ts::StateLabelOp>(location, builder.getStringAttr(label));
            return mlir::success();
        }

        auto noLabelOp = kind == SyntaxKind::WhileStatement || kind == SyntaxKind::DoStatement ||
                         kind == SyntaxKind::ForStatement || kind == SyntaxKind::ForInStatement ||
                         kind == SyntaxKind::ForOfStatement;

        if (noLabelOp)
        {
            return mlirGen(labeledStatementAST->statement, genContext);
        }

        auto labelOp = builder.create<mlir_ts::LabelOp>(location, builder.getStringAttr(label));

        // add merge block
        labelOp.addMergeBlock();
        auto *mergeBlock = labelOp.getMergeBlock();

        builder.setInsertionPointToStart(mergeBlock);

        auto res = mlirGen(labeledStatementAST->statement, genContext);

        builder.setInsertionPointAfter(labelOp);

        return res;
    }

    mlir::LogicalResult MLIRGenImpl::mlirGen(DebuggerStatement debuggerStatementAST, const GenContext &genContext)
    {
        auto location = loc(debuggerStatementAST);

        builder.create<mlir_ts::DebuggerOp>(location);
        return mlir::success();
    }

    mlir::LogicalResult MLIRGenImpl::mlirGen(ContinueStatement continueStatementAST, const GenContext &genContext)
    {
        auto location = loc(continueStatementAST);

        auto label = MLIRHelper::getName(continueStatementAST->label);

        EXIT_IF_FAILED(mlirGenDisposable(location, DisposeDepth::LoopScope, label, &genContext));

        builder.create<mlir_ts::ContinueOp>(location, builder.getStringAttr(label));
        return mlir::success();
    }

    mlir::LogicalResult MLIRGenImpl::mlirGen(BreakStatement breakStatementAST, const GenContext &genContext)
    {
        auto location = loc(breakStatementAST);

        auto label = MLIRHelper::getName(breakStatementAST->label);

        EXIT_IF_FAILED(mlirGenDisposable(location, DisposeDepth::LoopScope, label, &genContext));

        builder.create<mlir_ts::BreakOp>(location, builder.getStringAttr(label));
        return mlir::success();
    }

    mlir::LogicalResult MLIRGenImpl::mlirGen(SwitchStatement switchStatementAST, const GenContext &genContext)
    {
        SymbolTableScopeT varScope(symbolTable);

        auto location = loc(switchStatementAST);

        auto switchExpr = switchStatementAST->expression;
        auto result = mlirGen(switchExpr, genContext);
        EXIT_IF_FAILED_OR_NO_VALUE(result)
        auto switchValue = V(result);

        auto switchOp = builder.create<mlir_ts::SwitchOp>(location, switchValue);

        GenContext switchGenContext(genContext);
        switchGenContext.allocateVarsOutsideOfOperation = true;
        switchGenContext.currentOperation = switchOp;
        switchGenContext.insertIntoParentScope = true;

        // add merge block
        switchOp.addMergeBlock();
        auto *mergeBlock = switchOp.getMergeBlock();

        auto &clauses = switchStatementAST->caseBlock->clauses;

        SmallVector<mlir::cf::CondBranchOp> pendingConditions;
        SmallVector<mlir::cf::BranchOp> pendingBranches;
        mlir::Operation *previousConditionOrFirstBranchOp = nullptr;
        mlir::Block *defaultBlock = nullptr;

        // to support safe cast
        std::function<void(Expression, mlir::Value)> safeCastLogic;
        if (switchExpr == SyntaxKind::PropertyAccessExpression)
        {
            auto propertyAccessExpressionOp = switchExpr.as<PropertyAccessExpression>();
            auto objAccessExpression = propertyAccessExpressionOp->expression;
            auto typeOfObject = evaluate(objAccessExpression, switchGenContext);
            auto name = propertyAccessExpressionOp->name;

            safeCastLogic = [=, &switchGenContext](Expression caseExpr, mlir::Value constVal) {
                GenContext safeCastGenContext(switchGenContext);
                switchGenContext.insertIntoParentScope = false;

                // Safe Cast
                if (mlir::failed(checkSafeCastTypeOf(switchExpr, caseExpr, false, nullptr, switchGenContext)))
                {
                    checkSafeCastPropertyAccessLogic(caseExpr, objAccessExpression, typeOfObject, name, constVal,
                                                     false, nullptr, switchGenContext);
                }
            };
        }
        else
        {
            safeCastLogic = [&](Expression caseExpr, mlir::Value constVal) {};
        }

        // process without default
        for (int index = 0; index < clauses.size(); index++)
        {
            if (mlir::failed(mlirGenSwitchCase(location, switchExpr, switchValue, clauses, index, mergeBlock,
                                               defaultBlock, pendingConditions, pendingBranches,
                                               previousConditionOrFirstBranchOp, safeCastLogic, switchGenContext)))
            {
                return mlir::failure();
            }
        }

        LLVM_DEBUG(llvm::dbgs() << "\n!! SWITCH: " << switchOp << "\n");

        return mlir::success();
    }

    mlir::LogicalResult MLIRGenImpl::mlirGen(ThrowStatement throwStatementAST, const GenContext &genContext)
    {
        auto location = loc(throwStatementAST);

        auto result = mlirGen(throwStatementAST->expression, genContext);
        EXIT_IF_FAILED_OR_NO_VALUE(result)
        auto exception = V(result);

        auto throwOp = builder.create<mlir_ts::ThrowOp>(location, exception);

        if (!genContext.allowPartialResolve)
        {
            MLIRRTTIHelperVC rtti(builder, theModule, compileOptions);
            if (!rtti.setRTTIForType(
                location, exception.getType(), 
                [&](StringRef classFullName) { return getClassInfoByFullName(classFullName); }))
            {
                emitError(location, "Not supported type in throw");
                return mlir::failure();
            }
        }

        return mlir::success();
    }

    mlir::LogicalResult MLIRGenImpl::mlirGen(TryStatement tryStatementAST, const GenContext &genContext)
    {
        auto location = loc(tryStatementAST);

        std::string varName;
        auto catchClause = tryStatementAST->catchClause;
        if (catchClause)
        {
            auto varDecl = catchClause->variableDeclaration;
            if (varDecl)
            {
                varName = MLIRHelper::getName(varDecl->name);
                if (mlir::failed(mlirGen(varDecl, VariableType::Let, genContext)))
                {
                    return mlir::failure();
                }
            }
        }

        if (genContext.funcOp)
        {
            mlir_ts::FuncOp funcOp = genContext.funcOp;
            funcOp.setPersonalityAttr(builder.getBoolAttr(true));
        }

        auto tryOp = builder.create<mlir_ts::TryOp>(location);

        GenContext tryGenContext(genContext);
        // TODO: why do I need to allocate variables outside of "try" block?
        // well - short answer: to get access to vars in nested blocks for example 'cleanup'
        tryGenContext.allocateUsingVarsOutsideOfOperation = true;
        tryGenContext.currentOperation = tryOp;

        SmallVector<mlir::Type, 0> types;

        /*auto *body =*/builder.createBlock(&tryOp.getBody(), {}, types);
        /*auto cleanup =*/builder.createBlock(&tryOp.getCleanup(), {}, types);
        /*auto *catches =*/builder.createBlock(&tryOp.getCatches(), {}, types);
        /*auto *finallyBlock =*/builder.createBlock(&tryOp.getFinally(), {}, types);

        {
            // body
            builder.setInsertionPointToStart(&tryOp.getBody().front());

            // prepare custom scope
            SymbolTableScopeT varScope(symbolTable);
            GenContext tryBodyGenContext(tryGenContext);
            tryBodyGenContext.parentBlockContext = &tryGenContext;

            auto usingVars = std::make_unique<SmallVector<ts::VariableDeclarationDOM::TypePtr>>();
            tryBodyGenContext.usingVars = usingVars.get();

            auto result = mlirGenNoScopeVarsAndDisposable(tryStatementAST->tryBlock, tryBodyGenContext);
            EXIT_IF_FAILED(result)

            EXIT_IF_FAILED(mlirGenDisposable(location, DisposeDepth::CurrentScopeKeepAfterUse, {}, &tryBodyGenContext));

            // terminator
            builder.create<mlir_ts::ResultOp>(location);

            // cleanup
            builder.setInsertionPointToStart(&tryOp.getCleanup().front());
            // we need to call dispose for those which are in "using"
            // usingVars are empty here
            EXIT_IF_FAILED(mlirGenDisposable(location, DisposeDepth::CurrentScope, {}, &tryBodyGenContext));

            // terminator
            builder.create<mlir_ts::ResultOp>(location);
        }

        // catches
        builder.setInsertionPointToStart(&tryOp.getCatches().front());
        if (catchClause && catchClause->block)
        {
            auto location = loc(catchClause->block);
            if (!varName.empty())
            {
                MLIRCodeLogic mcl(builder, compileOptions);
                auto varInfo = resolveIdentifier(location, varName, tryGenContext);
                auto varRef = mcl.GetReferenceFromValue(location, varInfo);
                builder.create<mlir_ts::CatchOp>(location, varRef);

                if (!genContext.allowPartialResolve)
                {
                    MLIRRTTIHelperVC rtti(builder, theModule, compileOptions);
                    if (!rtti.setRTTIForType(
                        location, 
                        varInfo.getType(),
                        [&](StringRef classFullName) { return getClassInfoByFullName(classFullName); }))
                    {
                        emitError(location, "Not supported type in catch");
                        return mlir::failure();
                    }
                }
            }

            auto result = mlirGen(tryStatementAST->catchClause->block, tryGenContext);
            EXIT_IF_FAILED(result)
        }

        // terminator
        builder.create<mlir_ts::ResultOp>(location);

        // finally
        builder.setInsertionPointToStart(&tryOp.getFinally().front());
        if (tryStatementAST->finallyBlock)
        {
            auto result = mlirGen(tryStatementAST->finallyBlock, tryGenContext);
            EXIT_IF_FAILED(result)
        }

        // terminator
        builder.create<mlir_ts::ResultOp>(location);

        builder.setInsertionPointAfter(tryOp);
        return mlir::success();
    }

} // namespace mlirgen
} // namespace typescript
