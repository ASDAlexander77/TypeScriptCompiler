#include "TypeScript/MLIRGen.h"
#include "TypeScript/TypeScriptDialect.h"
#include "TypeScript/TypeScriptOps.h"

#include "mlir/IR/Types.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Verifier.h"

#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/ScopedHashTable.h"
#include "llvm/Support/raw_ostream.h"

#include "TypeScript/VisitorAST.h"

#include "TypeScript/DOM.h"
#include "TypeScript/Defines.h"

// parser includes
#include "parser.h"
#include "utilities.h"
#include "file_helper.h"

#include <numeric>

using namespace typescript;
using namespace ts;
namespace mlir_ts = mlir::typescript;

using llvm::ArrayRef;
using llvm::cast;
using llvm::dyn_cast;
using llvm::isa;
using llvm::makeArrayRef;
using llvm::ScopedHashTableScope;
using llvm::SmallVector;
using llvm::StringRef;
using llvm::Twine;

namespace
{
    struct PassResult
    {
        mlir::Type functionReturnType;
    };

    struct GenContext
    {
        bool allowPartialResolve;
        mlir::Type functionReturnType;
        PassResult *passResult;
    };

    /// Implementation of a simple MLIR emission from the TypeScript AST.
    ///
    /// This will emit operations that are specific to the TypeScript language, preserving
    /// the semantics of the language and (hopefully) allow to perform accurate
    /// analysis and transformation based on these high level semantics.
    class MLIRGenImpl
    {
        using VariablePairT = std::pair<mlir::Value, VariableDeclarationDOM::TypePtr>;
        using SymbolTableScopeT = llvm::ScopedHashTableScope<StringRef, VariablePairT>;

    public:
        MLIRGenImpl(const mlir::MLIRContext &context) : builder(&const_cast<mlir::MLIRContext &>(context))
        {
            fileName = "<unknown>";
        }

        MLIRGenImpl(const mlir::MLIRContext &context, const llvm::StringRef &fileNameParam) : builder(&const_cast<mlir::MLIRContext &>(context))
        {
            fileName = fileNameParam;
        }

        mlir::ModuleOp mlirGen(SourceFile module)
        {
            sourceFile = module;

            // We create an empty MLIR module and codegen functions one at a time and
            // add them to the module.
            theModule = mlir::ModuleOp::create(loc(module), fileName);
            builder.setInsertionPointToStart(theModule.getBody());

            declareAllFunctionDeclarations(module);

            SymbolTableScopeT varScope(symbolTable);

            // Process generating here
            GenContext genContext = {0};
            for (auto &statement : module->statements)
            {
                if (failed(mlirGen(statement, genContext)))
                {
                    return nullptr;
                }
            }

            // Verify the module after we have finished constructing it, this will check
            // the structural properties of the IR and invoke any specific verifiers we
            // have on the TypeScript operations.
            if (failed(mlir::verify(theModule)))
            {
                // TODO: uncomment it
                theModule.emitError("module verification error");
                //return nullptr;
            }

            return theModule;
        }

        mlir::LogicalResult declareAllFunctionDeclarations(SourceFile module)
        {
            auto unresolvedFunctions = -1;

            // VisitorAST
            // TODO: test recursive references
            do
            {
                auto unresolvedFunctionsCurrentRun = 0;
                FilterVisitorAST<FunctionDeclaration> visitorAST(
                    SyntaxKind::FunctionDeclaration,
                    [&](auto funcDecl) {
                        GenContext genContextDecl = {0};
                        genContextDecl.allowPartialResolve = true;

                        auto funcOpAndFuncProto = mlirGenFunctionPrototype(funcDecl, genContextDecl);
                        auto result = std::get<2>(funcOpAndFuncProto);
                        if (!result)
                        {
                            unresolvedFunctionsCurrentRun++;
                            return;
                        }

                        auto funcOp = std::get<0>(funcOpAndFuncProto);
                        auto &funcProto = std::get<1>(funcOpAndFuncProto);
                        if (auto funcOp = theModule.lookupSymbol<mlir::FuncOp>(funcProto->getName()))
                        {
                            return;
                        }

                        functionMap.insert({funcOp.getName(), funcOp});
                    });
                visitorAST.visit(module);

                if (unresolvedFunctionsCurrentRun == unresolvedFunctions)
                {
                    emitError(loc(module)) << "can't resolve recursive references of functions '" << fileName << "'";
                    return mlir::failure();
                }

                unresolvedFunctions = unresolvedFunctionsCurrentRun;
            } while (unresolvedFunctions > 0);

            return mlir::success();
        }

        mlir::LogicalResult mlirGenBody(Node body, const GenContext &genContext)
        {
            auto kind = (SyntaxKind)body;
            if (kind == SyntaxKind::Block)
            {
                return mlirGen(body.as<Block>(), genContext);
            }

            llvm_unreachable("unknown body type");
        }        

        mlir::LogicalResult mlirGen(Block blockAST, const GenContext &genContext)
        {
            for (auto &statement : blockAST->statements)
            {
                if (failed(mlirGen(statement, genContext)))
                {
                    return mlir::failure();
                }
            }

            return mlir::success();
        }

        mlir::LogicalResult mlirGen(Statement statementAST, const GenContext &genContext)
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
            else if (kind == SyntaxKind::Block)
            {
                return mlirGen(statementAST.as<Block>(), genContext);
            }
            else if (kind == SyntaxKind::EmptyStatement)
            {
                return mlir::success();
            }

            llvm_unreachable("unknown statement type");
        }

        mlir::LogicalResult mlirGen(ExpressionStatement expressionStatementAST, const GenContext &genContext)
        {
            mlirGen(expressionStatementAST->expression, genContext);
            return mlir::success();
        }

        mlir::Value mlirGen(Expression expressionAST, const GenContext &genContext)
        {
            auto kind = (SyntaxKind)expressionAST;
            if (kind == SyntaxKind::NumericLiteral)
            {
                return mlirGen(expressionAST.as<NumericLiteral>(), genContext);
            }
            else if (kind == SyntaxKind::StringLiteral)
            {
                return mlirGen(expressionAST.as<ts::StringLiteral>(), genContext);
            }
            else if (kind == SyntaxKind::NullKeyword)
            {
                return mlirGen(expressionAST.as<NullLiteral>(), genContext);
            }
            else if (kind == SyntaxKind::UndefinedKeyword)
            {
                // TODO: finish it
                //return mlirGen(expressionAST.as<UndefinedLiteral>(), genContext);
                llvm_unreachable("unknown expression");
            }
            else if (kind == SyntaxKind::TrueKeyword)
            {
                return mlirGen(expressionAST.as<TrueLiteral>(), genContext);
            }
            else if (kind == SyntaxKind::FalseKeyword)
            {
                return mlirGen(expressionAST.as<FalseLiteral>(), genContext);
            }
            else if (kind == SyntaxKind::Identifier)
            {
                return mlirGen(expressionAST.as<Identifier>(), genContext);
            }
            else if (kind == SyntaxKind::CallExpression)
            {
                return mlirGen(expressionAST.as<CallExpression>(), genContext);
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

            llvm_unreachable("unknown expression");
        }

        mlir::LogicalResult mlirGen(VariableStatement variableStatementAST, const GenContext &genContext)
        {
            auto location = loc(variableStatementAST);

            for (auto &item : variableStatementAST->declarationList->declarations)
            {
                mlir::Type type;

                auto name = wstos(item->name.as<Identifier>()->escapedText);

                if (item->type)
                {
                    type = getType(item->type);
                }

                mlir::Value init;
                if (auto initializer = item->initializer)
                {
                    init = mlirGen(initializer, genContext);
                    if (!type)
                    {
                        type = init.getType();
                    }
                    else if (type != init.getType())
                    {
                        auto castValue = builder.create<mlir_ts::CastOp>(loc(initializer), type, init);
                        init = castValue;
                    }
                }

                auto isGlobal = symbolTable.getCurScope()->getParentScope() == nullptr;

                auto varDecl = std::make_shared<VariableDeclarationDOM>(name, type, location);
                auto isConst = hasConstModifier(variableStatementAST);
                if (!isConst)
                {
                    varDecl->setReadWriteAccess();
                }

                varDecl->setIsGlobal(isGlobal);

                if (!isGlobal)
                {
                    if (isConst)
                    {
                        declare(varDecl, init);
                    }
                    else
                    {
                        auto variableOp = builder.create<mlir_ts::VariableOp>(
                            location,
                            mlir_ts::RefType::get(type),
                            init);

                        declare(varDecl, variableOp);
                    }
                }
                else
                {
                    // get constant
                    auto value = mlir::Attribute();
                    if (init)
                    {
                        if (auto constOp = dyn_cast_or_null<mlir::ConstantOp>(init.getDefiningOp()))
                        {
                            value = constOp.value();
                        }
                        else if (auto stringOp = dyn_cast_or_null<mlir_ts::StringOp>(init.getDefiningOp()))
                        {
                            value = stringOp.txtAttr();
                        }

                        // TODO global init value
                        init.getDefiningOp()->erase();
                    }

                    auto globalOp = 
                        builder.create<mlir_ts::GlobalOp>(
                            location,
                            type,
                            isConst,
                            name,
                            value);

                    declare(varDecl, mlir::Value());
                }
            }

            return mlir::success();
        }
        
        std::vector<std::shared_ptr<FunctionParamDOM>> mlirGenParameters(SignatureDeclarationBase parametersContextAST,
                                                               const GenContext &genContext)
        {
            std::vector<std::shared_ptr<FunctionParamDOM>> params;
            if (!parametersContextAST)
            {
                return params;
            }

            auto formalParams = parametersContextAST->parameters;

            // add extra parameter to send number of parameters
            auto anyOptionalParam =
                formalParams.end() != std::find_if(formalParams.begin(), formalParams.end(), [](auto param) {
                    return param->questionToken || !!param->initializer;
                });

            if (anyOptionalParam)
            {
                params.push_back(std::make_shared<FunctionParamDOM>(COUNT_PARAMS_PARAMETERNAME, builder.getI32Type(), loc(parametersContextAST), false));
            }

            for (auto arg : formalParams)
            {
                auto name = wstos(arg->name.as<Identifier>()->escapedText);
                mlir::Type type;
                auto isOptional = !!arg->questionToken;
                auto typeParameter = arg->type;
                if (typeParameter)
                {
                    type = getType(typeParameter);
                    if (!type)
                    {
                        if (!genContext.allowPartialResolve)
                        {
                            emitError(loc(typeParameter)) << "can't resolve type for parameter '" << name << "'";
                        }

                        return params;
                    }
                }

                // process init value
                auto initializer = arg->initializer;
                if (initializer)
                {
                    // we need to add temporary block
                    auto tempFuncType = builder.getFunctionType(llvm::None, llvm::None);
                    auto tempFuncOp = mlir::FuncOp::create(loc(initializer), name, tempFuncType);
                    auto &entryBlock = *tempFuncOp.addEntryBlock();

                    auto insertPoint = builder.saveInsertionPoint();
                    builder.setInsertionPointToStart(&entryBlock);

                    auto initValue = mlirGen(initializer, genContext);
                    if (initValue)
                    {
                        // TODO: set type if not provided
                        isOptional = true;
                        if (!type)
                        {
                            auto baseType = initValue.getType();
                            //type = OptionalType::get(baseType);
                            type = baseType;
                        }

                        // remove generated node as we need to detect type only
                        initValue.getDefiningOp()->erase();
                    }

                    // remove temp block
                    builder.restoreInsertionPoint(insertPoint);
                    entryBlock.erase();
                }

                params.push_back(std::make_shared<FunctionParamDOM>(name, type, loc(arg), isOptional, initializer));
            }

            return params;
        }

        std::tuple<mlir_ts::FuncOp, FunctionPrototypeDOM::TypePtr, bool> mlirGenFunctionPrototype(
            FunctionDeclaration functionDeclarationAST, const GenContext &genContext)
        {
            auto location = loc(functionDeclarationAST);

            std::vector<FunctionParamDOM::TypePtr> params = mlirGenParameters(functionDeclarationAST, genContext);
            SmallVector<mlir::Type> argTypes;
            auto argNumber = 0;
            auto argOptionalFrom = -1;

            for (const auto &param : params)
            {
                auto paramType = param->getType();
                if (!paramType)
                {
                    return std::make_tuple(mlir_ts::FuncOp(), FunctionPrototypeDOM::TypePtr(nullptr), false);
                }

                argTypes.push_back(paramType);
                if (param->getIsOptional() && argOptionalFrom < 0)
                {
                    argOptionalFrom = argNumber;
                }

                argNumber++;
            }

            std::string name;
            auto identifier = functionDeclarationAST->name.as<Identifier>();
            if (identifier)
            {
                name = wstos(identifier->escapedText);
            }
            else
            {
                // auto calculate name
                // __func+location
            }

            auto funcProto = std::make_shared<FunctionPrototypeDOM>(name, params);

            mlir::FunctionType funcType;
            if (auto typeParameter = functionDeclarationAST->type)
            {
                auto returnType = getType(typeParameter);
                funcProto->setReturnType(returnType);
                funcType = builder.getFunctionType(argTypes, returnType);
            }
            else if (auto returnType = getReturnType(functionDeclarationAST, name, argTypes, funcProto, genContext))
            {
                funcProto->setReturnType(returnType);
                funcType = builder.getFunctionType(argTypes, returnType);
            }
            else
            {
                // no return type
                funcType = builder.getFunctionType(argTypes, llvm::None);
            }

            SmallVector<mlir::NamedAttribute> attrs;
            // save info about optional parameters
            if (argOptionalFrom >= 0)
            {
                attrs.push_back(builder.getNamedAttr(FUNC_OPTIONAL_ATTR_NAME, builder.getI8IntegerAttr(argOptionalFrom)));
            }

            auto funcOp = mlir_ts::FuncOp::create(location, name, funcType, ArrayRef<mlir::NamedAttribute>(attrs));

            return std::make_tuple(funcOp, std::move(funcProto), true);
        }

        mlir::Type getReturnType(FunctionDeclaration functionDeclarationAST, StringRef name,
                                 const SmallVector<mlir::Type> &argTypes, const FunctionPrototypeDOM::TypePtr &funcProto, const GenContext &genContext)
        {
            mlir::Type returnType;

            // check if we have any return with expression
            auto hasReturnStatementWithExpr = false;
            FilterVisitorAST<ReturnStatement> visitorAST1(
                SyntaxKind::ReturnStatement,
                [&](auto retStatement) {
                    if (retStatement->expression)
                    {
                        hasReturnStatementWithExpr = true;
                    }
                });

            visitorAST1.visit(functionDeclarationAST);

            if (!hasReturnStatementWithExpr)
            {
                return returnType;
            }

            mlir::OpBuilder::InsertionGuard guard(builder);

            auto partialDeclFuncType = builder.getFunctionType(argTypes, llvm::None);
            auto dummyFuncOp = mlir_ts::FuncOp::create(loc(functionDeclarationAST), name, partialDeclFuncType);

            // simulate scope
            SymbolTableScopeT varScope(symbolTable);

            GenContext genContextWithPassResult(genContext);
            genContextWithPassResult.allowPartialResolve = true;
            genContextWithPassResult.passResult = new PassResult();
            if (failed(mlirGenFunctionBody(functionDeclarationAST, dummyFuncOp, funcProto, genContextWithPassResult, true)))
            {
                return mlir::Type();
            }

            return genContextWithPassResult.passResult->functionReturnType;
        }

        mlir::LogicalResult mlirGen(FunctionDeclaration functionDeclarationAST, const GenContext &genContext)
        {
            SymbolTableScopeT varScope(symbolTable);
            auto funcOpWithFuncProto = mlirGenFunctionPrototype(functionDeclarationAST, genContext);

            auto &funcOp = std::get<0>(funcOpWithFuncProto);
            auto &funcProto = std::get<1>(funcOpWithFuncProto);
            auto result = std::get<2>(funcOpWithFuncProto);
            if (!result || !funcOp)
            {
                return mlir::failure();
            }

            auto funcGenContext = GenContext(genContext);
            if (funcOp.getNumResults() > 0)
            {
                funcGenContext.functionReturnType = funcOp.getType().getResult(0);
            }

            auto returnType = mlirGenFunctionBody(functionDeclarationAST, funcOp, funcProto, funcGenContext);

            // set visibility index
            if (functionDeclarationAST->name.as<Identifier>()->escapedText != S("main"))
            {
                funcOp.setPrivate();
            }

            theModule.push_back(funcOp);
            functionMap.insert({funcOp.getName(), funcOp});

            return mlir::success();
        }

        mlir::LogicalResult mlirGenFunctionBody(FunctionDeclaration functionDeclarationAST, mlir_ts::FuncOp funcOp,
                                                FunctionPrototypeDOM::TypePtr funcProto, const GenContext &genContext, bool dummyRun = false)
        {
            auto &entryBlock = *funcOp.addEntryBlock();

            // process function params
            for (auto paramPairs : llvm::zip(funcProto->getArgs(), entryBlock.getArguments()))
            {
                if (failed(declare(std::get<0>(paramPairs), std::get<1>(paramPairs))))
                {
                    return mlir::failure();
                }
            }

            // allocate all params

            builder.setInsertionPointToStart(&entryBlock);

            // add exit code
            auto retType = funcProto->getReturnType();
            auto hasReturn = retType && !retType.isa<mlir_ts::VoidType>();
            if (hasReturn)
            {
                auto location = loc(functionDeclarationAST);
                auto entryOp = builder.create<mlir_ts::EntryOp>(location, mlir_ts::RefType::get(retType));
                auto varDecl = std::make_shared<VariableDeclarationDOM>(RETURN_VARIABLE_NAME, retType, location);
                varDecl->setReadWriteAccess();
                declare(varDecl, entryOp.reference());
            }
            else
            {
                builder.create<mlir_ts::EntryOp>(loc(functionDeclarationAST), mlir::Type());
            }

            auto arguments = entryBlock.getArguments();

            auto index = -1;
            for (const auto &param : funcProto->getArgs())
            {
                index++;

                // skip __const_params, it is not real param
                if (param->getName() == COUNT_PARAMS_PARAMETERNAME)
                {
                    continue;
                }

                mlir::Value paramValue;

                // alloc all args
                // process optional parameters
                if (param->getIsOptional() || param->hasInitValue())
                {
                    // process init expression
                    auto location = param->getLoc();

                    auto countArgsValue = arguments[0];

                    auto paramOptionalOp = builder.create<mlir_ts::ParamOptionalOp>(
                        location,
                        mlir_ts::RefType::get(param->getType()),
                        arguments[index],
                        countArgsValue,
                        builder.getI32IntegerAttr(index));

                    paramValue = paramOptionalOp;

                    if (param->hasInitValue())
                    {
                        auto *defValueBlock = new mlir::Block();
                        paramOptionalOp.defaultValueRegion().push_back(defValueBlock);

                        auto sp = builder.saveInsertionPoint();
                        builder.setInsertionPointToStart(defValueBlock);

                        mlir::Value defaultValue;
                        auto initExpression = param->getInitValue();
                        if (initExpression)
                        {
                            defaultValue = mlirGen(initExpression, genContext);
                        }
                        else
                        {
                            llvm_unreachable("unknown statement");
                        }

                        if (param->getType() != defaultValue.getType())
                        {
                            defaultValue = builder.create<mlir_ts::CastOp>(location, param->getType(), defaultValue);
                        }

                        builder.create<mlir_ts::ParamDefaultValueOp>(location, defaultValue);

                        builder.restoreInsertionPoint(sp);
                    }
                }
                else
                {
                    paramValue = builder.create<mlir_ts::ParamOp>(
                        param->getLoc(),
                        mlir_ts::RefType::get(param->getType()),
                        arguments[index]);
                }

                if (paramValue)
                {
                    // redefine variable
                    param->setReadWriteAccess();
                    declare(param, paramValue, true);
                }
            }

            if (failed(mlirGenBody(functionDeclarationAST->body, genContext)))
            {
                return mlir::failure();
            }

            // add exit code
            builder.create<mlir_ts::ExitOp>(loc(functionDeclarationAST));

            if (dummyRun)
            {
                entryBlock.erase();
            }

            return mlir::success();
        }

        mlir::LogicalResult mlirGen(ReturnStatement returnStatementAST, const GenContext &genContext)
        {
            auto location = loc(returnStatementAST);
            if (auto expression = returnStatementAST->expression)
            {
                auto expressionValue = mlirGen(expression, genContext);
                if (genContext.functionReturnType && genContext.functionReturnType != expressionValue.getType())
                {
                    auto castValue = builder.create<mlir_ts::CastOp>(loc(expression), genContext.functionReturnType, expressionValue);
                    expressionValue = castValue;
                }

                // record return type if not provided
                if (genContext.passResult)
                {
                    genContext.passResult->functionReturnType = expressionValue.getType();
                }

                auto retVarInfo = symbolTable.lookup(RETURN_VARIABLE_NAME);
                if (!retVarInfo.second)
                {
                    if (genContext.allowPartialResolve)
                    {
                        return mlir::success();
                    }

                    emitError(location) << "can't find return variable";
                }

                builder.create<mlir_ts::ReturnValOp>(location, expressionValue, retVarInfo.first);
            }
            else
            {
                builder.create<mlir_ts::ReturnOp>(location);
            }

            return mlir::success();
        }

        mlir::LogicalResult mlirGen(IfStatement ifStatementAST, const GenContext &genContext)
        {
            auto location = loc(ifStatementAST);

            auto hasElse = !!ifStatementAST->elseStatement;

            // condition
            auto condValue = mlirGen(ifStatementAST->expression, genContext);

            auto ifOp = builder.create<mlir_ts::IfOp>(location, condValue, hasElse);

            builder.setInsertionPointToStart(&ifOp.thenRegion().front());
            mlirGen(ifStatementAST->thenStatement, genContext);

            if (hasElse)
            {
                builder.setInsertionPointToStart(&ifOp.elseRegion().front());
                mlirGen(ifStatementAST->elseStatement, genContext);
            }

            builder.setInsertionPointAfter(ifOp);

            return mlir::success();
        }

        mlir::Value mlirGen(UnaryExpression unaryExpressionAST, const GenContext &genContext)
        {
            auto kind = (SyntaxKind)unaryExpressionAST;
            if (kind == SyntaxKind::PrefixUnaryExpression)
            {
                return mlirGen(unaryExpressionAST.as<PrefixUnaryExpression>(), genContext);
            }

            llvm_unreachable("unknown statement type");            
        }

        mlir::Value mlirGen(PrefixUnaryExpression prefixUnaryExpressionAST, const GenContext &genContext)
        {
            auto location = loc(prefixUnaryExpressionAST);

            auto opCode = prefixUnaryExpressionAST->_operator;

            auto expression = prefixUnaryExpressionAST->operand;
            auto expressionValue = mlirGen(expression, genContext);
            auto boolValue = expressionValue;

            switch (opCode)
            {
            case SyntaxKind::ExclamationToken:
                
                if (expressionValue.getType() != getBooleanType())
                {
                    boolValue = builder.create<mlir_ts::CastOp>(location, getBooleanType(), expressionValue);
                }

                return builder.create<mlir_ts::ArithmeticUnaryOp>(
                    location,
                    getBooleanType(),
                    builder.getI32IntegerAttr((int)opCode),
                    boolValue);            
            case SyntaxKind::TildeToken:
            case SyntaxKind::PlusToken:
            case SyntaxKind::MinusToken:
                return builder.create<mlir_ts::ArithmeticUnaryOp>(
                    location,
                    expressionValue.getType(),
                    builder.getI32IntegerAttr((int)opCode),
                    expressionValue);
            default:
                llvm_unreachable("not implemented");
            }
        }        

        mlir::Value mlirGen(ConditionalExpression conditionalExpressionAST, const GenContext &genContext)
        {
            auto location = loc(conditionalExpressionAST);

            // condition
            auto condValue = mlirGen(conditionalExpressionAST->condition, genContext);

            // detect value type
            mlir::Type resultType;
            {
                mlir::OpBuilder::InsertionGuard guard(builder);
                auto resultTrueTemp = mlirGen(conditionalExpressionAST->whenTrue, genContext);
                resultType = resultTrueTemp.getType();
                resultTrueTemp.getDefiningOp()->erase();
            }

            auto ifOp = builder.create<mlir_ts::IfOp>(location, mlir::TypeRange{resultType}, condValue, true);

            builder.setInsertionPointToStart(&ifOp.thenRegion().front());
            auto resultTrue = mlirGen(conditionalExpressionAST->whenTrue, genContext);
            builder.create<mlir_ts::YieldOp>(location, mlir::ValueRange{resultTrue});

            builder.setInsertionPointToStart(&ifOp.elseRegion().front());
            auto resultFalse = mlirGen(conditionalExpressionAST->whenFalse, genContext);
            builder.create<mlir_ts::YieldOp>(location, mlir::ValueRange{resultFalse});

            builder.setInsertionPointAfter(ifOp);

            return ifOp.getResult(0);
        }

        mlir::Value mlirGenAndOrLogic(BinaryExpression binaryExpressionAST, const GenContext &genContext, bool andOp)
        {
            auto location = loc(binaryExpressionAST);

            auto leftExpression = binaryExpressionAST->left;
            auto rightExpression = binaryExpressionAST->right;

            // condition
            auto leftExpressionValue = mlirGen(leftExpression, genContext);
            auto resultType = leftExpressionValue.getType();

            auto condValue = builder.create<mlir_ts::CastOp>(location, getBooleanType(), leftExpressionValue);

            auto ifOp = builder.create<mlir_ts::IfOp>(location, mlir::TypeRange{resultType}, condValue, true);

            builder.setInsertionPointToStart(&ifOp.thenRegion().front());
            auto resultTrue = andOp ? mlirGen(rightExpression, genContext) : leftExpressionValue;
            builder.create<mlir_ts::YieldOp>(location, mlir::ValueRange{resultTrue});

            builder.setInsertionPointToStart(&ifOp.elseRegion().front());
            auto resultFalse = andOp ? leftExpressionValue : mlirGen(rightExpression, genContext);
            builder.create<mlir_ts::YieldOp>(location, mlir::ValueRange{resultFalse});

            builder.setInsertionPointAfter(ifOp);

            return ifOp.getResult(0);            
        }

        mlir::Value mlirGen(BinaryExpression binaryExpressionAST, const GenContext &genContext)
        {
            auto location = loc(binaryExpressionAST);

            auto opCode = (SyntaxKind) binaryExpressionAST->operatorToken;

            auto leftExpression = binaryExpressionAST->left;
            auto rightExpression = binaryExpressionAST->right;

            if (opCode == SyntaxKind::AmpersandAmpersandToken || opCode == SyntaxKind::BarBarToken)
            {
                return mlirGenAndOrLogic(binaryExpressionAST, genContext, opCode == SyntaxKind::AmpersandAmpersandToken);
            }

            auto leftExpressionValue = mlirGen(leftExpression, genContext);
            auto rightExpressionValue = mlirGen(rightExpression, genContext);

            auto leftExpressionValueBeforeCast = leftExpressionValue;
            auto rightExpressionValueBeforeCast = rightExpressionValue;

            // cast step
            switch (opCode)
            {
                case SyntaxKind::CommaToken:
                    // no cast needed
                    break;
                case SyntaxKind::LessThanLessThanToken:
                case SyntaxKind::GreaterThanGreaterThanToken:
                case SyntaxKind::GreaterThanGreaterThanGreaterThanToken:
                    // cast to int
                    if (leftExpressionValue.getType() != builder.getI32Type())
                    {
                        leftExpressionValue = builder.create<mlir_ts::CastOp>(loc(leftExpression), builder.getI32Type(), leftExpressionValue);
                    }

                    if (rightExpressionValue.getType() != builder.getI32Type())
                    {
                        rightExpressionValue = builder.create<mlir_ts::CastOp>(loc(rightExpression), builder.getI32Type(), rightExpressionValue);
                    }                    

                    break;
                case SyntaxKind::SlashToken:
                case SyntaxKind::PercentToken:
                case SyntaxKind::AsteriskAsteriskToken:

                    if (leftExpressionValue.getType() != builder.getF32Type())
                    {
                        leftExpressionValue = builder.create<mlir_ts::CastOp>(loc(leftExpression), builder.getF32Type(), leftExpressionValue);
                    }                

                    if (rightExpressionValue.getType() != builder.getF32Type())
                    {
                        rightExpressionValue = builder.create<mlir_ts::CastOp>(loc(rightExpression), builder.getF32Type(), rightExpressionValue);
                    }    

                    break;
            case SyntaxKind::EqualsEqualsToken:
            case SyntaxKind::EqualsEqualsEqualsToken:
            case SyntaxKind::ExclamationEqualsToken:
            case SyntaxKind::ExclamationEqualsEqualsToken:
            case SyntaxKind::GreaterThanToken:
            case SyntaxKind::GreaterThanEqualsToken:
            case SyntaxKind::LessThanToken:
            case SyntaxKind::LessThanEqualsToken:

                    if (leftExpressionValue.getType() != rightExpressionValue.getType())
                    {
                        // cast to base type
                        auto hasF32 = leftExpressionValue.getType() == builder.getF32Type() || rightExpressionValue.getType() == builder.getF32Type();
                        if (hasF32)
                        {
                            if (leftExpressionValue.getType() != builder.getF32Type())
                            {
                                leftExpressionValue = builder.create<mlir_ts::CastOp>(loc(leftExpression), builder.getF32Type(), leftExpressionValue);
                            }                            
                            
                            if (rightExpressionValue.getType() != builder.getF32Type())
                            {
                                rightExpressionValue = builder.create<mlir_ts::CastOp>(loc(rightExpression), builder.getF32Type(), rightExpressionValue);
                            }                              
                        }
                        else
                        {
                            auto hasI32 = leftExpressionValue.getType() == builder.getI32Type() || rightExpressionValue.getType() == builder.getI32Type();
                            if (hasI32)
                            {
                                if (leftExpressionValue.getType() != builder.getI32Type())
                                {
                                    leftExpressionValue = builder.create<mlir_ts::CastOp>(loc(leftExpression), builder.getI32Type(), leftExpressionValue);
                                }                            
                                
                                if (rightExpressionValue.getType() != builder.getI32Type())
                                {
                                    rightExpressionValue = builder.create<mlir_ts::CastOp>(loc(rightExpression), builder.getI32Type(), rightExpressionValue);
                                }                                   
                            }
                        }
                    }

                    break;                    
                default:
                    if (leftExpressionValue.getType() != rightExpressionValue.getType())
                    {
                        rightExpressionValue = builder.create<mlir_ts::CastOp>(loc(rightExpression), leftExpressionValue.getType(), rightExpressionValue);
                    }

                    break;
            }

            switch (opCode)
            {
            case SyntaxKind::EqualsToken:
            {
                auto loadOp = dyn_cast<mlir_ts::LoadOp>(leftExpressionValue.getDefiningOp());
                if (loadOp)
                {
                    builder.create<mlir_ts::StoreOp>(
                        location,
                        rightExpressionValue,
                        loadOp.reference());
                }
                else
                {
                    builder.create<mlir_ts::StoreOp>(
                        location,
                        rightExpressionValue,
                        leftExpressionValue);
                }

                return rightExpressionValue;
            }
            case SyntaxKind::EqualsEqualsToken:
            case SyntaxKind::EqualsEqualsEqualsToken:
            case SyntaxKind::ExclamationEqualsToken:
            case SyntaxKind::ExclamationEqualsEqualsToken:
            case SyntaxKind::GreaterThanToken:
            case SyntaxKind::GreaterThanEqualsToken:
            case SyntaxKind::LessThanToken:
            case SyntaxKind::LessThanEqualsToken:
                return builder.create<mlir_ts::LogicalBinaryOp>(
                    location,
                    getBooleanType(),
                    builder.getI32IntegerAttr((int)opCode),
                    leftExpressionValue,
                    rightExpressionValue);
            case SyntaxKind::CommaToken:
                return rightExpressionValue;
            default:
                return builder.create<mlir_ts::ArithmeticBinaryOp>(
                    location,
                    leftExpressionValue.getType(),
                    builder.getI32IntegerAttr((int)opCode),
                    leftExpressionValue,
                    rightExpressionValue);
            }
        }

        mlir::Value mlirGen(SpreadElement spreadElement, const GenContext &genContext)
        {
            return mlirGen(spreadElement->expression, genContext);
        }

        mlir::Value mlirGen(ParenthesizedExpression parenthesizedExpression, const GenContext &genContext)
        {
            return mlirGen(parenthesizedExpression->expression, genContext);
        }

        mlir::Value mlirGen(CallExpression callExpression, const GenContext &genContext)
        {
            auto location = loc(callExpression);

            // get function ref.
            auto result = mlirGen(callExpression->expression.as<Expression>(), genContext);

            auto definingOp = result.getDefiningOp();
            if (definingOp)
            {
                auto opName = definingOp->getName().getStringRef();
                auto attrName = StringRef(IDENTIFIER_ATTR_NAME);
                if (definingOp->hasAttrOfType<mlir::FlatSymbolRefAttr>(attrName))
                {
                    auto calleeName = definingOp->getAttrOfType<mlir::FlatSymbolRefAttr>(attrName);
                    auto functionName = calleeName.getValue();
                    auto argumentsContext = callExpression->arguments;
                    auto opArgsCount = std::distance(argumentsContext.begin(), argumentsContext.end());

                    // resolve function
                    auto calledFuncIt = functionMap.find(functionName);
                    if (calledFuncIt == functionMap.end())
                    {
                        // print - internal command;
                        if (functionName.compare(StringRef("print")) == 0)
                        {
                            SmallVector<mlir::Value, 4> operands;
                            mlirGen(argumentsContext, operands, genContext);
                            mlir::succeeded(mlirGenPrint(location, operands));
                            return nullptr;
                        }

                        // assert - internal command;
                        if (functionName.compare(StringRef("assert")) == 0 && opArgsCount > 0)
                        {
                            SmallVector<mlir::Value, 4> operands;
                            mlirGen(argumentsContext, operands, genContext);
                            mlir::succeeded(mlirGenAssert(location, operands));
                            return nullptr;
                        }

                        // assert - internal command;
                        if (functionName.compare(StringRef("parseInt")) == 0 && opArgsCount > 0)
                        {
                            SmallVector<mlir::Value, 4> operands;
                            mlirGen(argumentsContext, operands, genContext);
                            return mlirGenParseInt(location, operands);
                        }

                        if (functionName.compare(StringRef("parseFloat")) == 0 && opArgsCount > 0)
                        {
                            SmallVector<mlir::Value, 4> operands;
                            mlirGen(argumentsContext, operands, genContext);
                            return mlirGenParseFloat(location, operands);
                        }

                        if (!genContext.allowPartialResolve)
                        {
                            emitError(location) << "no defined function found for '" << functionName << "'";
                        }

                        return nullptr;
                    }

                    auto calledFunc = calledFuncIt->second;
                    auto calledFuncType = calledFunc.getType();
                    auto funcArgsCount = calledFunc.getNumArguments();

                    // process arguments
                    SmallVector<mlir::Value, 4> operands;

                    auto hasOptionalFrom = calledFunc.getOperation()->hasAttrOfType<mlir::IntegerAttr>(FUNC_OPTIONAL_ATTR_NAME);
                    if (hasOptionalFrom)
                    {
                        auto constNumOfParams = builder.create<mlir_ts::ConstantOp>(location, builder.getI32Type(), builder.getI32IntegerAttr(opArgsCount));
                        operands.push_back(constNumOfParams);
                    }

                    mlirGen(argumentsContext, operands, calledFuncType, hasOptionalFrom, genContext);

                    if (hasOptionalFrom)
                    {
                        auto optionalFrom = funcArgsCount - opArgsCount;
                        if (hasOptionalFrom && optionalFrom > 0)
                        {
                            // -1 to exclude count params
                            for (auto i = (size_t)opArgsCount; i < funcArgsCount - 1; i++)
                            {
                                operands.push_back(builder.create<mlir_ts::UndefOp>(location, calledFuncType.getInput(i + 1)));
                            }
                        }
                    }

                    // default call by name
                    auto callOp =
                        builder.create<mlir_ts::CallOp>(
                            location,
                            calledFunc,
                            operands);

                    if (calledFunc.getType().getNumResults() > 0)
                    {
                        return callOp.getResult(0);
                    }

                    return nullptr;
                }
            }

            return nullptr;
        }

        mlir::LogicalResult mlirGenPrint(const mlir::Location &location, const SmallVector<mlir::Value, 4> &operands)
        {
            auto printOp =
                builder.create<mlir_ts::PrintOp>(
                    location,
                    operands);

            return mlir::success();
        }

        mlir::LogicalResult mlirGenAssert(const mlir::Location &location, const SmallVector<mlir::Value, 4> &operands)
        {
            auto msg = StringRef("assert");
            if (operands.size() > 1)
            {
                auto param2 = operands[1];
                auto stringOp = dyn_cast_or_null<mlir_ts::StringOp>(param2.getDefiningOp());
                if (stringOp)
                {
                    msg = stringOp.txtAttr().getValue();
                }

                param2.getDefiningOp()->erase();
            }

            auto assertOp =
                builder.create<mlir_ts::AssertOp>(
                    location,
                    operands.front(),
                    mlir::StringAttr::get(msg, theModule.getContext()));

            return mlir::success();
        }

        mlir::Value mlirGenParseInt(const mlir::Location &location, const SmallVector<mlir::Value, 4> &operands)
        {
            auto parseIntOp =
                builder.create<mlir_ts::ParseIntOp>(
                    location,
                    builder.getI32Type(),
                    operands.front());

            return parseIntOp;
        }

        mlir::Value mlirGenParseFloat(const mlir::Location &location, const SmallVector<mlir::Value, 4> &operands)
        {
            auto parseFloatOp =
                builder.create<mlir_ts::ParseFloatOp>(
                    location,
                    builder.getF32Type(),
                    operands.front());

            return parseFloatOp;
        }

        mlir::LogicalResult mlirGen(NodeArray<Expression> arguments, SmallVector<mlir::Value, 4> &operands, const GenContext &genContext)
        {
            for (auto expression : arguments)
            {
                auto value = mlirGen(expression, genContext);
                operands.push_back(value);
            }

            return mlir::success();
        }

        mlir::LogicalResult mlirGen(NodeArray<Expression> arguments, SmallVector<mlir::Value, 4> &operands, mlir::FunctionType funcType, bool hasOptionalFrom, const GenContext &genContext)
        {
            auto i = hasOptionalFrom ? 0 : -1;
            for (auto expression : arguments)
            {
                i++;
                auto value = mlirGen(expression, genContext);

                if (value.getType() != funcType.getInput(i))
                {
                    auto castValue = builder.create<mlir_ts::CastOp>(loc(expression), funcType.getInput(i), value);
                    operands.push_back(castValue);
                }
                else
                {
                    operands.push_back(value);
                }
            }

            return mlir::success();
        }

        mlir::Value mlirGen(TypeOfExpression typeOfExpression, const GenContext &genContext)
        {
            auto result = mlirGen(typeOfExpression->expression, genContext);
            auto type = result.getType();
            if (type.isIntOrIndexOrFloat() && !type.isIntOrIndex())
            {
                // return "number"
                auto typeOfValue = builder.create<mlir_ts::StringOp>(loc(typeOfExpression), getStringType(), StringRef("number"));
                return typeOfValue;
            }

            llvm_unreachable("not implemented");
        }

        mlir::Value mlirGen(NullLiteral nullLiteral, const GenContext &genContext)
        {
            return builder.create<mlir_ts::NullOp>(
                loc(nullLiteral),
                getAnyType());
        }

        /*
        mlir::Value mlirGen(UndefinedLiteral undefinedLiteral, const GenContext &genContext)
        {
            return builder.create<mlir_ts::UndefOp>(
                loc(undefinedLiteral),
                getAnyType());
        }
        */

        mlir::Value mlirGen(TrueLiteral trueLiteral, const GenContext &genContext)
        {
            return builder.create<mlir_ts::ConstantOp>(
                loc(trueLiteral),
                getBooleanType(),
                mlir::BoolAttr::get(true, theModule.getContext()));
        }

        mlir::Value mlirGen(FalseLiteral falseLiteral, const GenContext &genContext)
        {
            return builder.create<mlir_ts::ConstantOp>(
                loc(falseLiteral),
                getBooleanType(),
                mlir::BoolAttr::get(false, theModule.getContext()));
        }

        mlir::Value mlirGen(NumericLiteral numericLiteral, const GenContext &genContext)
        {
            if (numericLiteral->text.find(S(".")) == string::npos)
            {
                return builder.create<mlir::ConstantOp>(
                    loc(numericLiteral),
                    builder.getI32Type(),
                    builder.getI32IntegerAttr(to_unsigned_integer(numericLiteral->text)));
            }

            if (!(numericLiteral->numericLiteralFlags & TokenFlags::NumericLiteralFlags))
            {
                return builder.create<mlir::ConstantOp>(
                    loc(numericLiteral),
                    builder.getF32Type(),
                    builder.getF32FloatAttr(to_float(numericLiteral->text)));
            }

            llvm_unreachable("unknown numeric literal");
        }

        mlir::Value mlirGen(ts::StringLiteral stringLiteral, const GenContext &genContext)
        {
            auto text = wstos(stringLiteral->text);

            return builder.create<mlir_ts::StringOp>(
                loc(stringLiteral),
                getStringType(),
                builder.getStringAttr(StringRef(text.data(), text.length() + 1)));
        }

        mlir::Value mlirGen(Identifier identifier, const GenContext &genContext)
        {
            // resolve name
            auto name = wstos(identifier->escapedText);

            auto value = symbolTable.lookup(name);
            if (value.second)
            {
                if (value.first)
                {
                    if (!value.second->getReadWriteAccess())
                    {
                        return value.first;
                    }

                    // load value if memref
                    return builder.create<mlir_ts::LoadOp>(value.first.getLoc(), value.first.getType().cast<mlir_ts::RefType>().getElementType(), value.first);
                }
                else if (value.second->getIsGlobal())
                {
                    // global var
                    auto location = loc(identifier);
                    if (!value.second->getReadWriteAccess() && value.second->getType().isa<mlir_ts::StringType>())
                    {
                        // load address of const object in global
                        return builder.create<mlir_ts::AddressOfConstStringOp>(location, value.second->getType(), value.second->getName());
                    }
                    else
                    {
                        auto address = builder.create<mlir_ts::AddressOfOp>(location, mlir_ts::RefType::get(value.second->getType()), value.second->getName());
                        return builder.create<mlir_ts::LoadOp>(location, value.second->getType(), address);
                    }
                }
            }

            // unresolved reference (for call for example)
            return mlir_ts::IdentifierReference::create(loc(identifier), name);
        }

        mlir::Type getType(Node typeReferenceAST)
        {
            auto kind = (SyntaxKind) typeReferenceAST;
            if (kind == SyntaxKind::BooleanKeyword)
            {
                return getBooleanType();
            }
            else if (kind == SyntaxKind::NumberKeyword)
            {
                return builder.getF32Type();
            }
            else if (kind == SyntaxKind::BigIntKeyword)
            {
                return builder.getI64Type();
            }
            else if (kind == SyntaxKind::StringKeyword)
            {
                return getStringType();
            }
            else if (kind == SyntaxKind::VoidKeyword)
            {
                return getVoidType();
            }            

            return getAnyType();
        }

        mlir_ts::VoidType getVoidType()
        {
            return mlir_ts::VoidType::get(builder.getContext());
        }

        mlir::Type getBooleanType()
        {
            return mlir_ts::BooleanType::get(builder.getContext());
        }
        
        mlir_ts::StringType getStringType()
        {
            return mlir_ts::StringType::get(builder.getContext());
        }

        mlir::Type getAnyType()
        {
            return mlir_ts::AnyType::get(builder.getContext());
        }

        mlir::LogicalResult declare(VariableDeclarationDOM::TypePtr var, mlir::Value value, bool redeclare = false)
        {
            const auto &name = var->getName();
            if (!redeclare && symbolTable.count(name))
            {
                return mlir::failure();
            }

            symbolTable.insert(name, {value, var});
            return mlir::success();
        }

    private:
        /// Helper conversion for a TypeScript AST location to an MLIR location.
        mlir::Location loc(TextRange loc)
        {
            //return builder.getFileLineColLoc(builder.getIdentifier(fileName), loc->pos, loc->_end);
            auto posLineChar = parser.getLineAndCharacterOfPosition(sourceFile, loc->pos);
            return builder.getFileLineColLoc(builder.getIdentifier(fileName), posLineChar.line + 1, posLineChar.character + 1);
        }

        /// A "module" matches a TypeScript source file: containing a list of functions.
        mlir::ModuleOp theModule;

        /// The builder is a helper class to create IR inside a function. The builder
        /// is stateful, in particular it keeps an "insertion point": this is where
        /// the next operations will be introduced.
        mlir::OpBuilder builder;

        mlir::StringRef fileName;

        llvm::ScopedHashTable<StringRef, VariablePairT> symbolTable;

        llvm::StringMap<mlir_ts::FuncOp> functionMap;

        // helper to get line number
        Parser parser;
        ts::SourceFile sourceFile;
    };
} // namespace

namespace typescript
{
    ::std::string dumpFromSource(const llvm::StringRef &fileName, const llvm::StringRef &source)
    {
        auto showLineCharPos = false;

        Parser parser;
        auto sourceFile = parser.parseSourceFile(stows(static_cast<std::string>(fileName)), stows(static_cast<std::string>(source)), ScriptTarget::Latest);

        stringstream s;

        FuncT<> visitNode;
        ArrayFuncT<> visitArray;

        auto intent = 0;

        visitNode = [&](Node child) -> Node {

            for (auto i = 0; i < intent; i++)
            {
                s << "\t";
            }

            if (showLineCharPos)
            {
                auto posLineChar = parser.getLineAndCharacterOfPosition(sourceFile, child->pos);
                auto endLineChar = parser.getLineAndCharacterOfPosition(sourceFile, child->_end);

                s 
                    << S("Node: ")
                    << parser.syntaxKindString(child).c_str()
                    << S(" @ [ ") << child->pos << S("(") << posLineChar.line + 1 << S(":") << posLineChar.character + 1 << S(") - ") 
                    << child->_end << S("(") << endLineChar.line + 1  << S(":") << endLineChar.character  << S(") ]") << std::endl;
            }
            else
            {
                s << S("Node: ") << parser.syntaxKindString(child).c_str() << S(" @ [ ") << child->pos << S(" - ") << child->_end << S(" ]") << std::endl;
            }

            intent++;
            ts::forEachChild(child, visitNode, visitArray);    
            intent--;

            return undefined;
        };

        visitArray = [&](NodeArray<Node> array) -> Node {
            for (auto node : array)
            {
                visitNode(node);
            }

            return undefined;
        };

        auto result = forEachChild(sourceFile.as<Node>(), visitNode, visitArray);
        return wstos(s.str());
    }

    mlir::OwningModuleRef mlirGenFromSource(const mlir::MLIRContext &context, const llvm::StringRef &fileName, const llvm::StringRef &source)
    {
        Parser parser;
        auto sourceFile = parser.parseSourceFile(stows(static_cast<std::string>(fileName)), stows(static_cast<std::string>(source)), ScriptTarget::Latest);
        return MLIRGenImpl(context, fileName).mlirGen(sourceFile);
    }

} // namespace typescript
