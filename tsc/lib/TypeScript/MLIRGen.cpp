#include "TypeScript/MLIRGen.h"
#include "TypeScript/TypeScriptDialect.h"
#include "TypeScript/TypeScriptOps.h"

#include "TypeScript/DOM.h"
#include "TypeScript/Defines.h"
#include "TypeScript/AST.h"

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

#include "TypeScriptLexerANTLR.h"
#include "TypeScriptParserANTLR.h"
#include "TypeScript/VisitorAST.h"

#include <numeric>

using namespace mlir::typescript;
using namespace typescript;
namespace ts = mlir::typescript;

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

        mlir::ModuleOp mlirGen(TypeScriptParserANTLR::MainContext *module)
        {
            auto moduleAST = parse(module);
            if (moduleAST)
            {
                return mlirGen(moduleAST);
            }

            return mlir::ModuleOp();
        }

        /// Public API: convert the AST for a TypeScript module (source file) to an MLIR
        /// Module operation.
        mlir::ModuleOp mlirGen(ModuleAST::TypePtr module)
        {
            // We create an empty MLIR module and codegen functions one at a time and
            // add them to the module.
            theModule = mlir::ModuleOp::create(loc(module->getLoc()), fileName);
            builder.setInsertionPointToStart(theModule.getBody());

            declareAllFunctionDeclarations(module);

            // Process generating here
            GenContext genContext = {0};
            for (auto &statement : *module.get())
            {
                if (failed(mlirGenStatement(statement, genContext)))
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

        mlir::LogicalResult declareAllFunctionDeclarations(ModuleAST::TypePtr module)
        {
            auto unresolvedFunctions = -1;

            // VisitorAST
            // TODO: test recursive references
            do
            {
                auto unresolvedFunctionsCurrentRun = 0;
                FilterVisitorAST<FunctionDeclarationAST> visitorAST(
                    SyntaxKind::FunctionDeclaration,
                    [&](auto *funcDecl) {
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
                module->accept(&visitorAST);

                if (unresolvedFunctionsCurrentRun == unresolvedFunctions)
                {
                    emitError(loc(module->getLoc())) << "can't resolve recursive references of functions'" << fileName << "'";
                    return mlir::failure();
                }

                unresolvedFunctions = unresolvedFunctionsCurrentRun;
            } while (unresolvedFunctions > 0);

            return mlir::success();
        }

        mlir::LogicalResult mlirGen(BlockAST::TypePtr blockAST, const GenContext &genContext)
        {
            for (auto &statement : *blockAST)
            {
                if (failed(mlirGenStatement(statement, genContext)))
                {
                    return mlir::failure();
                }
            }

            return mlir::success();
        }

        mlir::LogicalResult mlirGenStatement(NodeAST::TypePtr statementAST, const GenContext &genContext)
        {
            // TODO:
            if (statementAST->getKind() == SyntaxKind::FunctionDeclaration)
            {
                return mlirGen(std::dynamic_pointer_cast<FunctionDeclarationAST>(statementAST), genContext);
            }
            else if (statementAST->getKind() == SyntaxKind::ExpressionStatement)
            {
                return mlirGen(std::dynamic_pointer_cast<ExpressionStatementAST>(statementAST), genContext);
            }
            else if (statementAST->getKind() == SyntaxKind::VariableStatement)
            {
                return mlirGen(std::dynamic_pointer_cast<VariableStatementAST>(statementAST), genContext);
            }
            else if (statementAST->getKind() == SyntaxKind::IfStatement)
            {
                return mlirGen(std::dynamic_pointer_cast<IfStatementAST>(statementAST), genContext);
            }
            else if (statementAST->getKind() == SyntaxKind::ReturnStatement)
            {
                return mlirGen(std::dynamic_pointer_cast<ReturnStatementAST>(statementAST), genContext);
            }
            else if (statementAST->getKind() == SyntaxKind::Block)
            {
                return mlirGen(std::dynamic_pointer_cast<BlockAST>(statementAST), genContext);
            }
            else if (statementAST->getKind() == SyntaxKind::EmptyStatement)
            {
                return mlir::success();
            }

            llvm_unreachable("unknown statement type");
        }

        mlir::Value mlirGenExpression(NodeAST::TypePtr expressionAST, const GenContext &genContext)
        {
            if (expressionAST->getKind() == SyntaxKind::NumericLiteral)
            {
                return mlirGen(std::dynamic_pointer_cast<NumericLiteralAST>(expressionAST), genContext);
            }
            else if (expressionAST->getKind() == SyntaxKind::StringLiteral)
            {
                return mlirGen(std::dynamic_pointer_cast<StringLiteralAST>(expressionAST), genContext);
            }
            else if (expressionAST->getKind() == SyntaxKind::NullKeyword)
            {
                return mlirGen(std::dynamic_pointer_cast<NullLiteralAST>(expressionAST), genContext);
            }
            else if (expressionAST->getKind() == SyntaxKind::UndefinedKeyword)
            {
                return mlirGen(std::dynamic_pointer_cast<UndefinedLiteralAST>(expressionAST), genContext);
            }
            else if (expressionAST->getKind() == SyntaxKind::TrueKeyword)
            {
                return mlirGen(std::dynamic_pointer_cast<TrueLiteralAST>(expressionAST), genContext);
            }
            else if (expressionAST->getKind() == SyntaxKind::FalseKeyword)
            {
                return mlirGen(std::dynamic_pointer_cast<FalseLiteralAST>(expressionAST), genContext);
            }
            else if (expressionAST->getKind() == SyntaxKind::Identifier)
            {
                return mlirGen(std::dynamic_pointer_cast<IdentifierAST>(expressionAST), genContext);
            }
            else if (expressionAST->getKind() == SyntaxKind::CallExpression)
            {
                return mlirGen(std::dynamic_pointer_cast<CallExpressionAST>(expressionAST), genContext);
            }
            else if (expressionAST->getKind() == SyntaxKind::BinaryExpression)
            {
                return mlirGen(std::dynamic_pointer_cast<BinaryExpressionAST>(expressionAST), genContext);
            }

            llvm_unreachable("unknown expression");
        }

        mlir::LogicalResult mlirGen(ExpressionStatementAST::TypePtr expressionStatementAST, const GenContext &genContext)
        {
            mlirGenExpression(expressionStatementAST->getExpression(), genContext);
            return mlir::success();
        }

        mlir::LogicalResult mlirGen(VariableStatementAST::TypePtr variableStatementAST, const GenContext &genContext)
        {
            auto location = loc(variableStatementAST->getLoc());

            for (auto &item : *variableStatementAST->getDeclarationList())
            {
                mlir::Value init;
                mlir::Type type;

                auto name = item->getIdentifier()->getName();

                if (item->getType())
                {
                    type = getType(item->getType());
                }

                if (auto initializer = item->getInitializer())
                {
                    init = mlirGenExpression(initializer, genContext);
                    if (!type)
                    {
                        type = init.getType();
                    }
                    else if (type != init.getType())
                    {
                        auto castValue = builder.create<CastOp>(loc(initializer->getLoc()), type, init);
                        init = castValue;
                    }
                }

                auto varDecl = std::make_shared<VariableDeclarationDOM>(name, type, location);
                if (variableStatementAST->getIsConst())
                {
                    declare(varDecl, init);
                }
                else
                {
                    varDecl->setReadWriteAccess();

                    auto variableOp = builder.create<VariableOp>(
                        location,
                        ts::RefType::get(type),
                        init);

                    declare(varDecl, variableOp);
                }
            }

            return mlir::success();
        }

        std::vector<std::shared_ptr<FunctionParamDOM>> mlirGen(ParametersDeclarationAST::TypePtr parametersContextAST,
                                                               const GenContext &genContext)
        {
            std::vector<std::shared_ptr<FunctionParamDOM>> params;
            if (!parametersContextAST)
            {
                return params;
            }

            auto formalParams = parametersContextAST->getParameters();

            // add extra parameter to send number of parameters
            auto anyOptionalParam =
                formalParams.end() != std::find_if(formalParams.begin(), formalParams.end(), [](auto &param) {
                    return param->getIsOptional() || !!param->getInitializer();
                });

            if (anyOptionalParam)
            {
                params.push_back(std::make_shared<FunctionParamDOM>(COUNT_PARAMS_PARAMETERNAME, builder.getI32Type(), loc(parametersContextAST->getLoc()), false));
            }

            for (auto &arg : formalParams)
            {
                auto name = arg->getIdentifier()->getName();
                mlir::Type type;
                auto isOptional = arg->getIsOptional();
                auto typeParameter = arg->getType();
                if (typeParameter)
                {
                    type = getType(typeParameter);
                    if (!type)
                    {
                        if (!genContext.allowPartialResolve)
                        {
                            emitError(loc(typeParameter->getLoc())) << "can't resolve type for parameter '" << name << "'";
                        }

                        return params;
                    }
                }

                // process init value
                auto initializer = arg->getInitializer();
                if (initializer)
                {
                    // we need to add temporary block
                    auto tempFuncType = builder.getFunctionType(llvm::None, llvm::None);
                    auto tempFuncOp = mlir::FuncOp::create(loc(initializer->getLoc()), StringRef(name), tempFuncType);
                    auto &entryBlock = *tempFuncOp.addEntryBlock();

                    auto insertPoint = builder.saveInsertionPoint();
                    builder.setInsertionPointToStart(&entryBlock);

                    auto initValue = mlirGenExpression(initializer, genContext);
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

                params.push_back(std::make_shared<FunctionParamDOM>(name, type, loc(arg->getLoc()), isOptional, initializer));
            }

            return params;
        }

        std::tuple<FuncOp, FunctionPrototypeDOM::TypePtr, bool> mlirGenFunctionPrototype(
            FunctionDeclarationAST *functionDeclarationAST, const GenContext &genContext)
        {
            auto location = loc(functionDeclarationAST->getLoc());

            std::vector<FunctionParamDOM::TypePtr> params = mlirGen(functionDeclarationAST->getParameters(), genContext);
            SmallVector<mlir::Type> argTypes;
            auto argNumber = 0;
            auto argOptionalFrom = -1;

            for (const auto &param : params)
            {
                auto paramType = param->getType();
                if (!paramType)
                {
                    return std::make_tuple(FuncOp(), FunctionPrototypeDOM::TypePtr(nullptr), false);
                }

                argTypes.push_back(paramType);
                if (param->getIsOptional() && argOptionalFrom < 0)
                {
                    argOptionalFrom = argNumber;
                }

                argNumber++;
            }

            std::string name;
            auto identifier = functionDeclarationAST->getIdentifier();
            if (identifier)
            {
                name = identifier->getName();
            }
            else
            {
                // auto calculate name
                // __func+location
            }

            auto funcProto = std::make_shared<FunctionPrototypeDOM>(name, params);

            mlir::FunctionType funcType;
            if (auto typeParameter = functionDeclarationAST->getTypeParameter())
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

            auto funcOp = FuncOp::create(location, StringRef(name), funcType, ArrayRef<mlir::NamedAttribute>(attrs));

            return std::make_tuple(funcOp, std::move(funcProto), true);
        }

        mlir::Type getReturnType(FunctionDeclarationAST *functionDeclarationAST, std::string name,
                                 const SmallVector<mlir::Type> &argTypes, const FunctionPrototypeDOM::TypePtr &funcProto, const GenContext &genContext)
        {
            mlir::Type returnType;

            // check if we have any return with expration
            auto hasReturnStatementWithExpr = false;
            FilterVisitorAST<ReturnStatementAST> visitorAST1(
                SyntaxKind::ReturnStatement,
                [&](auto *retStatement) {
                    if (retStatement->getExpression())
                    {
                        hasReturnStatementWithExpr = true;
                    }
                });

            functionDeclarationAST->accept(&visitorAST1);

            if (!hasReturnStatementWithExpr)
            {
                return returnType;
            }

            auto partialDeclFuncType = builder.getFunctionType(argTypes, llvm::None);
            auto dummyFuncOp = FuncOp::create(loc(functionDeclarationAST->getLoc()), StringRef(name), partialDeclFuncType);

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

        mlir::LogicalResult mlirGen(FunctionDeclarationAST::TypePtr functionDeclarationAST, const GenContext &genContext)
        {
            SymbolTableScopeT varScope(symbolTable);
            auto funcOpWithFuncProto = mlirGenFunctionPrototype(functionDeclarationAST.get(), genContext);

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

            auto returnType = mlirGenFunctionBody(functionDeclarationAST.get(), funcOp, funcProto, funcGenContext);

            // set visibility index
            if (functionDeclarationAST->getIdentifier()->getName() != "main")
            {
                funcOp.setPrivate();
            }

            theModule.push_back(funcOp);
            functionMap.insert({funcOp.getName(), funcOp});

            return mlir::success();
        }

        mlir::LogicalResult mlirGenFunctionBody(FunctionDeclarationAST *functionDeclarationAST, FuncOp funcOp,
                                                FunctionPrototypeDOM::TypePtr funcProto, const GenContext &genContext, bool dummyRun = false)
        {
            auto &entryBlock = *funcOp.addEntryBlock();

            // process function params
            for (const auto paramPairs : llvm::zip(funcProto->getArgs(), entryBlock.getArguments()))
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
            if (retType)
            {
                auto location = loc(functionDeclarationAST->getLoc());
                auto entryOp = builder.create<EntryOp>(location, ts::RefType::get(retType));
                auto varDecl = std::make_shared<VariableDeclarationDOM>(RETURN_VARIABLE_NAME, retType, location);
                varDecl->setReadWriteAccess();
                declare(varDecl, entryOp.reference());
            }
            else
            {
                builder.create<EntryOp>(loc(functionDeclarationAST->getLoc()), mlir::Type());
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

                    auto paramOptionalOp = builder.create<ParamOptionalOp>(
                        location,
                        ts::RefType::get(param->getType()),
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
                            defaultValue = mlirGenExpression(initExpression, genContext);
                        }
                        else
                        {
                            llvm_unreachable("unknown statement");
                        }

                        if (param->getType() != defaultValue.getType())
                        {
                            defaultValue = builder.create<CastOp>(location, param->getType(), defaultValue);
                        }

                        builder.create<ParamDefaultValueOp>(location, defaultValue);

                        builder.restoreInsertionPoint(sp);
                    }
                }
                else
                {
                    paramValue = builder.create<ParamOp>(
                        param->getLoc(),
                        ts::RefType::get(param->getType()),
                        arguments[index]);
                }

                if (paramValue)
                {
                    // redefine variable
                    param->setReadWriteAccess();
                    declare(param, paramValue, true);
                }
            }

            if (failed(mlirGen(functionDeclarationAST->getFunctionBody(), genContext)))
            {
                return mlir::failure();
            }

            // add exit code
            builder.create<ExitOp>(loc(functionDeclarationAST->getLoc()));

            if (dummyRun)
            {
                entryBlock.erase();
            }

            return mlir::success();
        }

        mlir::LogicalResult mlirGen(ReturnStatementAST::TypePtr returnStatementAST, const GenContext &genContext)
        {
            auto location = loc(returnStatementAST->getLoc());
            if (auto expression = returnStatementAST->getExpression())
            {
                auto expressionValue = mlirGenExpression(expression, genContext);
                if (genContext.functionReturnType && genContext.functionReturnType != expressionValue.getType())
                {
                    auto castValue = builder.create<CastOp>(loc(expression->getLoc()), genContext.functionReturnType, expressionValue);
                    expressionValue = castValue;
                }

                // record return type if not provided
                if (genContext.passResult)
                {
                    genContext.passResult->functionReturnType = expressionValue.getType();
                }

                auto retVarInfo = resolve(RETURN_VARIABLE_NAME);
                if (!retVarInfo.first)
                {
                    if (genContext.allowPartialResolve)
                    {
                        return mlir::success();
                    }

                    emitError(location) << "can't find return variable";
                }

                builder.create<ReturnValOp>(location, expressionValue, retVarInfo.first);
            }
            else
            {
                builder.create<ReturnOp>(location);
            }

            return mlir::success();
        }

        mlir::LogicalResult mlirGen(IfStatementAST::TypePtr ifStatementAST, const GenContext &genContext)
        {
            auto location = loc(ifStatementAST->getLoc());

            auto hasElse = !!ifStatementAST->getWhenFalse();

            // condition
            auto condValue = mlirGenExpression(ifStatementAST->getCondition(), genContext);

            auto ifOp = builder.create<ts::IfOp>(location, condValue, hasElse);

            builder.setInsertionPointToStart(&ifOp.thenRegion().front());
            mlirGenStatement(ifStatementAST->getWhenTrue(), genContext);

            if (hasElse)
            {
                builder.setInsertionPointToStart(&ifOp.elseRegion().front());
                mlirGenStatement(ifStatementAST->getWhenFalse(), genContext);
            }

            builder.setInsertionPointAfter(ifOp);

            return mlir::success();
        }

        mlir::Value mlirGen(BinaryExpressionAST::TypePtr binaryExpressionAST, const GenContext &genContext)
        {
            auto location = loc(binaryExpressionAST->getLoc());

            auto opCode = binaryExpressionAST->getOpCode();

            auto leftExpression = binaryExpressionAST->getLeftExpression();
            auto rightExpression = binaryExpressionAST->getRightExpression();

            auto leftExpressionValue = mlirGenExpression(leftExpression, genContext);
            auto rightExpressionValue = mlirGenExpression(rightExpression, genContext);

            auto rightExpressionValueBeforeCast = rightExpressionValue;

            if (leftExpressionValue.getType() != rightExpressionValue.getType())
            {
                rightExpressionValue = builder.create<CastOp>(loc(rightExpression->getLoc()), leftExpressionValue.getType(), rightExpressionValue);
            }

            switch (opCode)
            {
            case SyntaxKind::EqualsToken:
                {
                    auto loadOp = dyn_cast<ts::LoadOp>(leftExpressionValue.getDefiningOp());
                    if (loadOp)
                    {
                        builder.create<ts::StoreOp>(
                            location,
                            rightExpressionValue,
                            loadOp.reference());
                    }
                    else
                    {
                        builder.create<ts::StoreOp>(
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
                return builder.create<LogicalBinaryOp>(
                    location,
                    builder.getI1Type(),
                    builder.getI32IntegerAttr((int)opCode),
                    leftExpressionValue,
                    rightExpressionValue);
            default:
                return builder.create<ArithmeticBinaryOp>(
                    location,
                    leftExpressionValue.getType(),
                    builder.getI32IntegerAttr((int)opCode),
                    leftExpressionValue,
                    rightExpressionValue);
            }
        }

        mlir::Value mlirGen(CallExpressionAST::TypePtr callExpression, const GenContext &genContext)
        {
            auto location = loc(callExpression->getLoc());

            // get function ref.
            auto result = mlirGenExpression(callExpression->getExpression(), genContext);

            auto definingOp = result.getDefiningOp();
            if (definingOp)
            {
                auto opName = definingOp->getName().getStringRef();
                auto attrName = StringRef(IDENTIFIER_ATTR_NAME);
                if (definingOp->hasAttrOfType<mlir::FlatSymbolRefAttr>(attrName))
                {
                    auto calleeName = definingOp->getAttrOfType<mlir::FlatSymbolRefAttr>(attrName);
                    auto functionName = calleeName.getValue();
                    auto argumentsContext = callExpression->getArguments();
                    auto opArgsCount = argumentsContext.size();

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
                        auto constNumOfParams = builder.create<mlir::ConstantOp>(location, builder.getI32Type(), builder.getI32IntegerAttr(opArgsCount));
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
                                operands.push_back(builder.create<UndefOp>(location, calledFuncType.getInput(i + 1)));
                            }
                        }
                    }

                    // default call by name
                    auto callOp =
                        builder.create<CallOp>(
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
                builder.create<PrintOp>(
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
                auto definingOpParam2 = param2.getDefiningOp();
                auto valueAttrName = StringRef("value");
                if (definingOpParam2)
                {
                    auto valueAttr = definingOpParam2->getAttrOfType<mlir::StringAttr>(valueAttrName);
                    if (valueAttr)
                    {
                        msg = valueAttr.getValue();
                        definingOpParam2->erase();
                    }
                }
            }

            auto assertOp =
                builder.create<AssertOp>(
                    location,
                    operands.front(),
                    mlir::StringAttr::get(msg, theModule.getContext()));

            return mlir::success();
        }

        mlir::Value mlirGenParseInt(const mlir::Location &location, const SmallVector<mlir::Value, 4> &operands)
        {
            auto parseIntOp =
                builder.create<ParseIntOp>(
                    location,
                    builder.getI32Type(),
                    operands.front());

            return parseIntOp;
        }

        mlir::Value mlirGenParseFloat(const mlir::Location &location, const SmallVector<mlir::Value, 4> &operands)
        {
            auto parseFloatOp =
                builder.create<ParseFloatOp>(
                    location,
                    builder.getF32Type(),
                    operands.front());

            return parseFloatOp;
        }

        mlir::LogicalResult mlirGen(std::vector<NodeAST::TypePtr> arguments, SmallVector<mlir::Value, 4> &operands, const GenContext &genContext)
        {
            for (auto expression : arguments)
            {
                auto value = mlirGenExpression(expression, genContext);
                operands.push_back(value);
            }

            return mlir::success();
        }

        mlir::LogicalResult mlirGen(std::vector<NodeAST::TypePtr> arguments, SmallVector<mlir::Value, 4> &operands, mlir::FunctionType funcType, bool hasOptionalFrom, const GenContext &genContext)
        {
            auto i = hasOptionalFrom ? 0 : -1;
            for (auto expression : arguments)
            {
                i++;
                auto value = mlirGenExpression(expression, genContext);

                if (value.getType() != funcType.getInput(i))
                {
                    auto castValue = builder.create<CastOp>(loc(expression->getLoc()), funcType.getInput(i), value);
                    operands.push_back(castValue);
                }
                else
                {
                    operands.push_back(value);
                }
            }

            return mlir::success();
        }

        mlir::Value mlirGen(NullLiteralAST::TypePtr nullLiteral, const GenContext &genContext)
        {
            return builder.create<NullOp>(
                loc(nullLiteral->getLoc()),
                getAnyType());
        }

        mlir::Value mlirGen(UndefinedLiteralAST::TypePtr undefinedLiteral, const GenContext &genContext)
        {
            return builder.create<UndefOp>(
                loc(undefinedLiteral->getLoc()),
                getAnyType());
        }

        mlir::Value mlirGen(TrueLiteralAST::TypePtr trueLiteral, const GenContext &genContext)
        {
            return builder.create<mlir::ConstantOp>(
                loc(trueLiteral->getLoc()),
                builder.getI1Type(),
                mlir::BoolAttr::get(true, theModule.getContext()));
        }

        mlir::Value mlirGen(FalseLiteralAST::TypePtr falseLiteral, const GenContext &genContext)
        {
            return builder.create<mlir::ConstantOp>(
                loc(falseLiteral->getLoc()),
                builder.getI1Type(),
                mlir::BoolAttr::get(false, theModule.getContext()));
        }

        mlir::Value mlirGen(NumericLiteralAST::TypePtr numericLiteral, const GenContext &genContext)
        {
            if (numericLiteral->getIsInt())
            {
                return builder.create<mlir::ConstantOp>(
                    loc(numericLiteral->getLoc()),
                    builder.getI32Type(),
                    builder.getI32IntegerAttr(numericLiteral->getIntValue()));
            }

            if (numericLiteral->getIsFloat())
            {
                return builder.create<mlir::ConstantOp>(
                    loc(numericLiteral->getLoc()),
                    builder.getF32Type(),
                    builder.getF32FloatAttr(numericLiteral->getFloatValue()));
            }

            llvm_unreachable("unknown numeric literal");
        }

        mlir::Value mlirGen(StringLiteralAST::TypePtr stringLiteral, const GenContext &genContext)
        {
            auto text = stringLiteral->getString();
            auto innerText = text.substr(1, text.length() - 2);

            return builder.create<ts::StringOp>(
                loc(stringLiteral->getLoc()),
                getStringType(),
                builder.getStringAttr(StringRef(innerText)));
        }

        mlir::Value mlirGen(IdentifierAST::TypePtr identifier, const GenContext &genContext)
        {
            // resolve name
            auto name = identifier->getName();

            auto value = resolve(name);
            if (value.first)
            {
                // load value if memref
                if (value.second)
                {
                    return builder.create<ts::LoadOp>(value.first.getLoc(), value.first.getType().cast<ts::RefType>().getElementType(), value.first);
                }

                return value.first;
            }

            // unresolved reference (for call for example)
            return IdentifierReference::create(loc(identifier->getLoc()), name);
        }

        mlir::Type getType(TypeReferenceAST::TypePtr typeReferenceAST)
        {
            if (typeReferenceAST->getTypeKind() == SyntaxKind::BooleanKeyword)
            {
                return builder.getI1Type();
            }
            else if (typeReferenceAST->getTypeKind() == SyntaxKind::NumberKeyword)
            {
                return builder.getF32Type();
            }
            else if (typeReferenceAST->getTypeKind() == SyntaxKind::BigIntKeyword)
            {
                return builder.getI64Type();
            }
            else if (typeReferenceAST->getTypeKind() == SyntaxKind::StringKeyword)
            {
                return getStringType();
            }

            return getAnyType();
        }

        ts::StringType getStringType()
        {
            return ts::StringType::get(builder.getContext());
        }

        mlir::Type getAnyType()
        {
            return ts::AnyType::get(builder.getContext());
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

        std::pair<mlir::Value, bool> resolve(StringRef name)
        {
            auto varIt = symbolTable.lookup(name);
            if (varIt.first)
            {
                return std::make_pair(varIt.first, varIt.second->getReadWriteAccess());
            }

            return std::make_pair(mlir::Value(), false);
        }

    private:
        /// Helper conversion for a TypeScript AST location to an MLIR location.
        mlir::Location loc(const typescript::TextRange &loc)
        {
            return builder.getFileLineColLoc(builder.getIdentifier(fileName), loc.pos, loc.end);
        }

        /// A "module" matches a TypeScript source file: containing a list of functions.
        mlir::ModuleOp theModule;

        /// The builder is a helper class to create IR inside a function. The builder
        /// is stateful, in particular it keeps an "insertion point": this is where
        /// the next operations will be introduced.
        mlir::OpBuilder builder;

        mlir::StringRef fileName;

        llvm::ScopedHashTable<StringRef, VariablePairT> symbolTable;

        llvm::StringMap<FuncOp> functionMap;
    };

} // namespace

namespace typescript
{
    llvm::StringRef dumpFromSource(const llvm::StringRef &source)
    {
        antlr4::ANTLRInputStream input((std::string)source);
        typescript::TypeScriptLexerANTLR lexer(&input);
        antlr4::CommonTokenStream tokens(&lexer);
        typescript::TypeScriptParserANTLR parser(&tokens);
        auto *moduleAST = parser.main();
        return llvm::StringRef(moduleAST->toStringTree());
    }

    mlir::OwningModuleRef mlirGenFromSource(const mlir::MLIRContext &context, const llvm::StringRef &source, const llvm::StringRef &fileName)
    {
        antlr4::ANTLRInputStream input((std::string)source);
        typescript::TypeScriptLexerANTLR lexer(&input);
        antlr4::CommonTokenStream tokens(&lexer);
        typescript::TypeScriptParserANTLR parser(&tokens);
        return MLIRGenImpl(context, fileName).mlirGen(parser.main());
    }

} // namespace typescript
