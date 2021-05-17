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
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/raw_ostream.h"

#include "TypeScript/VisitorAST.h"
#include "TypeScript/MLIRGenLogic.h"

#include "TypeScript/DOM.h"
#include "TypeScript/Defines.h"

// parser includes
#include "parser.h"
#include "utilities.h"
#include "file_helper.h"

#include <numeric>

using namespace ::typescript;
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

// TODO: optimize of amount of calls to detect return types and if it is was calculated before then do not run it all the time

namespace
{
    struct PassResult
    {
        mlir::Type functionReturnType;
    };

    struct GenContext
    {
        GenContext() = default;

        void clean() 
        {
            if (cleanUps)
            {
                for (auto op : *cleanUps)
                {
                    op->erase();
                }

                delete cleanUps;
            }

            if (passResult)
            {
                delete passResult;
            }
        }

        bool allowPartialResolve;
        bool dummyRun;
        bool allowConstEval;
        mlir::Type functionReturnType;
        PassResult *passResult;
        mlir::SmallVector<mlir::Block *>* cleanUps;
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

        mlir::ModuleOp mlirGenSourceFile(SourceFile module)
        {
            sourceFile = module;

            // We create an empty MLIR module and codegen functions one at a time and
            // add them to the module.
            theModule = mlir::ModuleOp::create(loc(module), fileName);
            builder.setInsertionPointToStart(theModule.getBody());

            declareAllTypesAndEnumsDeclarations(module);
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

        mlir::LogicalResult declareAllTypesAndEnumsDeclarations(SourceFile module)
        {
            FilterVisitorAST<EnumDeclaration> visitorASTEnum(
                SyntaxKind::EnumDeclaration,
                [&](auto enumDecl) {
                    GenContext genContext;
                    mlirGen(enumDecl, genContext);
                });
            visitorASTEnum.visit(module);

            FilterVisitorAST<TypeAliasDeclaration> visitorASTType(
                SyntaxKind::TypeAliasDeclaration,
                [&](auto typeAliasDecl) {
                    GenContext genContext;
                    mlirGen(typeAliasDecl, genContext);
                });
            visitorASTType.visit(module);            

            return mlir::success();
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
                        if (auto funcOp = theModule.lookupSymbol<mlir_ts::FuncOp>(funcProto->getName()))
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

            if (body.is<Statement>())
            {
                return mlirGen(body.as<Statement>(), genContext);
            }

            if (body.is<Expression>())
            {
                auto result = mlirGen(body.as<Expression>(), genContext);
                return mlirGenReturnValue(loc(body), result, genContext);
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
            else if (kind == SyntaxKind::TypeAliasDeclaration)
            {
                // must be processed already
                //return mlirGen(statementAST.as<TypeAliasDeclaration>(), genContext);
                return mlir::success();
            }
            else if (kind == SyntaxKind::EnumDeclaration)
            {
                // must be processed already
                //return mlirGen(statementAST.as<EnumDeclaration>(), genContext);
                return mlir::success();
            }
            else if (kind == SyntaxKind::Block)
            {
                return mlirGen(statementAST.as<Block>(), genContext);
            }
            else if (kind == SyntaxKind::EmptyStatement || kind == SyntaxKind::Unknown/*TODO: temp solution to treat null statements as empty*/)
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
            else if (kind == SyntaxKind::ArrayLiteralExpression)
            {
                return mlirGen(expressionAST.as<ArrayLiteralExpression>(), genContext);
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
            else if (kind == SyntaxKind::PropertyAccessExpression)
            {
                return mlirGen(expressionAST.as<PropertyAccessExpression>(), genContext);
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
                return mlirGen(expressionAST.as<TemplateExpression>(), genContext);
            }                   
            else if (kind == SyntaxKind::Unknown/*TODO: temp solution to treat null expr as empty expr*/)
            {
                return mlir::Value();
            }

            llvm_unreachable("unknown expression");
        }

        mlir::LogicalResult mlirGen(VariableDeclarationList variableDeclarationListAST, bool isConst, const GenContext &genContext)
        {
            auto location = loc(variableDeclarationListAST);

            for (auto &item : variableDeclarationListAST->declarations)
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
                        assert (type);

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
                        if (auto constOp = dyn_cast_or_null<mlir_ts::ConstantOp>(init.getDefiningOp()))
                        {
                            value = constOp.value();
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

        mlir::LogicalResult mlirGen(VariableStatement variableStatementAST, const GenContext &genContext)
        {
            return mlirGen(variableStatementAST->declarationList, hasConstModifier(variableStatementAST), genContext);
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
            FunctionLikeDeclarationBase functionLikeDeclarationBaseAST, const GenContext &genContext)
        {
            auto location = loc(functionLikeDeclarationBaseAST);

            std::vector<FunctionParamDOM::TypePtr> params = mlirGenParameters(functionLikeDeclarationBaseAST, genContext);
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
            auto identifier = functionLikeDeclarationBaseAST->name.as<Identifier>();
            if (identifier)
            {
                name = wstos(identifier->escapedText);
            }
            else
            {
                // auto calculate name
                std::stringstream ssName;
                ssName << "__uf" << hash_value(location);
                name = ssName.str();
            }

            auto funcProto = std::make_shared<FunctionPrototypeDOM>(name, params);

            mlir::FunctionType funcType;
            if (auto typeParameter = functionLikeDeclarationBaseAST->type)
            {
                auto returnType = getType(typeParameter);
                funcProto->setReturnType(returnType);
                funcType = builder.getFunctionType(argTypes, returnType);
            }
            else if (auto returnType = getReturnType(functionLikeDeclarationBaseAST, name, argTypes, funcProto, genContext))
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

        mlir::Type getReturnType(FunctionLikeDeclarationBase functionLikeDeclarationBaseAST, StringRef name,
                                 const SmallVector<mlir::Type> &argTypes, const FunctionPrototypeDOM::TypePtr &funcProto, const GenContext &genContext)
        {
            mlir::Type returnType;

            // check if we have any return with expression
            auto hasReturnStatementWithExpr = false;
            FilterVisitorSkipFuncsAST<ReturnStatement> visitorAST1(
                SyntaxKind::ReturnStatement,
                [&](auto retStatement) {
                    if (retStatement->expression)
                    {
                        hasReturnStatementWithExpr = true;
                    }
                });

            visitorAST1.visit(functionLikeDeclarationBaseAST);

            if (!hasReturnStatementWithExpr)
            {
                auto allowFuncExprWithOneLine = 
                    (SyntaxKind)functionLikeDeclarationBaseAST == SyntaxKind::ArrowFunction
                    && (SyntaxKind)functionLikeDeclarationBaseAST->body != SyntaxKind::Block
                    && functionLikeDeclarationBaseAST->body.is<Expression>();

                if (!allowFuncExprWithOneLine)
                {
                    return returnType;
                }
            }

            mlir::OpBuilder::InsertionGuard guard(builder);

            auto partialDeclFuncType = builder.getFunctionType(argTypes, llvm::None);
            auto dummyFuncOp = mlir_ts::FuncOp::create(loc(functionLikeDeclarationBaseAST), name, partialDeclFuncType);

            // simulate scope
            SymbolTableScopeT varScope(symbolTable);

            GenContext genContextWithPassResult(genContext);
            genContextWithPassResult.allowPartialResolve = true;
            genContextWithPassResult.dummyRun = true;
            genContextWithPassResult.cleanUps = new SmallVector<mlir::Block*>();
            genContextWithPassResult.passResult = new PassResult();
            mlir::Type functionReturnType;
            if (succeeded(mlirGenFunctionBody(functionLikeDeclarationBaseAST, dummyFuncOp, funcProto, genContextWithPassResult)))
            {
                functionReturnType = genContextWithPassResult.passResult->functionReturnType;
            }

            genContextWithPassResult.clean();
            return functionReturnType;
        }

        mlir::LogicalResult mlirGen(FunctionDeclaration functionDeclarationAST, const GenContext &genContext)
        {
            mlir::OpBuilder::InsertionGuard guard(builder);
            if (mlirGenFunctionLikeDeclaration(functionDeclarationAST, genContext))            
            {
                return mlir::success();
            }

            return mlir::failure();
        }

        mlir::Value mlirGen(FunctionExpression functionExpressionAST, const GenContext &genContext)
        {
            mlir_ts::FuncOp funcOp;

            {
                mlir::OpBuilder::InsertionGuard guard(builder);
                builder.restoreInsertionPoint(functionBeginPoint);

                // provide name for it
                funcOp = mlirGenFunctionLikeDeclaration(functionExpressionAST, genContext);
                if (!funcOp)
                {
                    return mlir::Value();
                }
            }

            auto funcSymbolRef = 
                builder.create<mlir_ts::SymbolRefOp>(
                    loc(functionExpressionAST), 
                    funcOp.getType(), 
                    mlir::FlatSymbolRefAttr::get(funcOp.getName(), builder.getContext()));
            return funcSymbolRef;
        }        

        mlir::Value mlirGen(ArrowFunction arrowFunctionAST, const GenContext &genContext)
        {
            mlir_ts::FuncOp funcOp;

            {
                mlir::OpBuilder::InsertionGuard guard(builder);
                builder.restoreInsertionPoint(functionBeginPoint);

                // provide name for it
                funcOp = mlirGenFunctionLikeDeclaration(arrowFunctionAST, genContext);
                if (!funcOp)
                {
                    return mlir::Value();
                }
            }

            auto funcSymbolRef = 
                builder.create<mlir_ts::SymbolRefOp>(
                    loc(arrowFunctionAST), 
                    funcOp.getType(), 
                    mlir::FlatSymbolRefAttr::get(funcOp.getName(), builder.getContext()));
            return funcSymbolRef;
        }          

        mlir_ts::FuncOp mlirGenFunctionLikeDeclaration(FunctionLikeDeclarationBase functionLikeDeclarationBaseAST, const GenContext &genContext)
        {
            auto location = loc(functionLikeDeclarationBaseAST);

            SymbolTableScopeT varScope(symbolTable);
            auto funcOpWithFuncProto = mlirGenFunctionPrototype(functionLikeDeclarationBaseAST, genContext);

            auto &funcOp = std::get<0>(funcOpWithFuncProto);
            auto &funcProto = std::get<1>(funcOpWithFuncProto);
            auto result = std::get<2>(funcOpWithFuncProto);
            if (!result || !funcOp)
            {
                return funcOp;
            }

            auto funcGenContext = GenContext(genContext);
            if (funcOp.getNumResults() > 0)
            {
                funcGenContext.functionReturnType = funcOp.getType().getResult(0);
            }

            auto returnType = mlirGenFunctionBody(functionLikeDeclarationBaseAST, funcOp, funcProto, funcGenContext);

            // set visibility index
            if (funcOp.getName() != StringRef("main"))
            {
                funcOp.setPrivate();
            }

            if (!genContext.dummyRun)
            {
                theModule.push_back(funcOp);
            }

            functionMap.insert({funcOp.getName(), funcOp});

            return funcOp;
        }  

        mlir::LogicalResult mlirGenFunctionBody(FunctionLikeDeclarationBase functionLikeDeclarationBaseAST, mlir_ts::FuncOp funcOp,
                                                FunctionPrototypeDOM::TypePtr funcProto, const GenContext &genContext)
        {
            auto *blockPtr = funcOp.addEntryBlock();
            auto &entryBlock = *blockPtr;

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
                auto location = loc(functionLikeDeclarationBaseAST);
                auto entryOp = builder.create<mlir_ts::EntryOp>(location, mlir_ts::RefType::get(retType));
                auto varDecl = std::make_shared<VariableDeclarationDOM>(RETURN_VARIABLE_NAME, retType, location);
                varDecl->setReadWriteAccess();
                declare(varDecl, entryOp.reference());
            }
            else
            {
                builder.create<mlir_ts::EntryOp>(loc(functionLikeDeclarationBaseAST), mlir::Type());
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

            if (failed(mlirGenBody(functionLikeDeclarationBaseAST->body, genContext)))
            {
                return mlir::failure();
            }

            // add exit code
            builder.create<mlir_ts::ExitOp>(loc(functionLikeDeclarationBaseAST));

            if (genContext.dummyRun)
            {
                genContext.cleanUps->push_back(blockPtr);
            }

            return mlir::success();
        }

        mlir::Value mlirGen(TypeAssertion typeAssertionAST, const GenContext &genContext)
        {
            auto location = loc(typeAssertionAST);

            auto typeInfo = getType(typeAssertionAST->type);
            auto exprValue = mlirGen(typeAssertionAST->expression, genContext);

            auto castedValue = builder.create<mlir_ts::CastOp>(location, typeInfo, exprValue);
            return castedValue;
        }

        mlir::Value mlirGen(AsExpression asExpressionAST, const GenContext &genContext)
        {
            auto location = loc(asExpressionAST);

            auto typeInfo = getType(asExpressionAST->type);
            auto exprValue = mlirGen(asExpressionAST->expression, genContext);

            auto castedValue = builder.create<mlir_ts::CastOp>(location, typeInfo, exprValue);
            return castedValue;
        }        

        mlir::LogicalResult mlirGen(ReturnStatement returnStatementAST, const GenContext &genContext)
        {
            auto location = loc(returnStatementAST);
            if (auto expression = returnStatementAST->expression)
            {
                auto expressionValue = mlirGen(expression, genContext);
                return mlirGenReturnValue(location, expressionValue, genContext);
            }

            builder.create<mlir_ts::ReturnOp>(location);
            return mlir::success();
        }

        mlir::LogicalResult mlirGenReturnValue(mlir::Location location, mlir::Value expressionValue, const GenContext &genContext)
        {
            // empty return
            if (!expressionValue)
            {
                builder.create<mlir_ts::ReturnOp>(location);
                return mlir::success();                
            }

            if (genContext.functionReturnType && genContext.functionReturnType != expressionValue.getType())
            {
                auto castValue = builder.create<mlir_ts::CastOp>(location, genContext.functionReturnType, expressionValue);
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
                return mlir::failure();
            }

            builder.create<mlir_ts::ReturnValOp>(location, expressionValue, retVarInfo.first);

            return mlir::success();
        }

        mlir::LogicalResult mlirGen(IfStatement ifStatementAST, const GenContext &genContext)
        {
            auto location = loc(ifStatementAST);

            auto hasElse = !!ifStatementAST->elseStatement;

            // condition
            auto condValue = mlirGen(ifStatementAST->expression, genContext);
            if (condValue.getType() != getBooleanType())
            {
                condValue = builder.create<mlir_ts::CastOp>(location, getBooleanType(), condValue);
            }

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

        mlir::LogicalResult mlirGen(DoStatement doStatementAST, const GenContext &genContext)
        {
            auto location = loc(doStatementAST);

            SmallVector<mlir::Type, 0> types;
            SmallVector<mlir::Value, 0> operands;

            auto doWhileOp = builder.create<mlir_ts::DoWhileOp>(location, types, operands);
            /*auto *cond =*/ builder.createBlock(&doWhileOp.cond(), {}, types);
            /*auto *body =*/ builder.createBlock(&doWhileOp.body(), {}, types);

            // body in condition
            builder.setInsertionPointToStart(&doWhileOp.body().front());
            mlirGen(doStatementAST->statement, genContext);
            // just simple return, as body in cond
            builder.create<mlir_ts::YieldOp>(location);

            builder.setInsertionPointToStart(&doWhileOp.cond().front());
            auto conditionValue = mlirGen(doStatementAST->expression, genContext);
            builder.create<mlir_ts::ConditionOp>(location, conditionValue, mlir::ValueRange{});

            builder.setInsertionPointAfter(doWhileOp);
            return mlir::success();
        }        

        mlir::LogicalResult mlirGen(WhileStatement whileStatementAST, const GenContext &genContext)
        {
            auto location = loc(whileStatementAST);

            SmallVector<mlir::Type, 0> types;
            SmallVector<mlir::Value, 0> operands;

            auto whileOp = builder.create<mlir_ts::WhileOp>(location, types, operands);
            /*auto *cond =*/ builder.createBlock(&whileOp.cond(), {}, types);
            /*auto *body =*/ builder.createBlock(&whileOp.body(), {}, types);

            // condition
            builder.setInsertionPointToStart(&whileOp.cond().front());
            auto conditionValue = mlirGen(whileStatementAST->expression, genContext);
            builder.create<mlir_ts::ConditionOp>(location, conditionValue, mlir::ValueRange{});

            // body
            builder.setInsertionPointToStart(&whileOp.body().front());
            mlirGen(whileStatementAST->statement, genContext);
            builder.create<mlir_ts::YieldOp>(location);

            builder.setInsertionPointAfter(whileOp);
            return mlir::success();
        }           

        mlir::LogicalResult mlirGen(ForStatement forStatementAST, const GenContext &genContext)
        {
            auto location = loc(forStatementAST);

            // initializer
            // TODO: why do we have ForInitialier
            if (forStatementAST->initializer.is<Expression>())
            {
                auto init = mlirGen(forStatementAST->initializer.as<Expression>(), genContext);
                if (!init)
                {
                    return mlir::failure();
                }
            }
            else if (forStatementAST->initializer.is<VariableDeclarationList>())
            {
                auto result = mlirGen(forStatementAST->initializer.as<VariableDeclarationList>(), false, genContext);
                if (failed(result))
                {
                    return result;
                }
            }

            SmallVector<mlir::Type, 0> types;
            SmallVector<mlir::Value, 0> operands;

            auto forOp = builder.create<mlir_ts::ForOp>(location, types, operands);
            /*auto *cond =*/ builder.createBlock(&forOp.cond(), {}, types);
            /*auto *body =*/ builder.createBlock(&forOp.body(), {}, types);
            /*auto *incr =*/ builder.createBlock(&forOp.incr(), {}, types);

            builder.setInsertionPointToStart(&forOp.cond().front());
            auto conditionValue = mlirGen(forStatementAST->condition, genContext);
            if (conditionValue)
            {
                builder.create<mlir_ts::ConditionOp>(location, conditionValue, mlir::ValueRange{});
            }
            else
            {
                builder.create<mlir_ts::NoConditionOp>(location, mlir::ValueRange{});
            }

            // body
            builder.setInsertionPointToStart(&forOp.body().front());
            mlirGen(forStatementAST->statement, genContext);
            builder.create<mlir_ts::YieldOp>(location);

            // increment
            builder.setInsertionPointToStart(&forOp.incr().front());
            mlirGen(forStatementAST->incrementor, genContext);
            builder.create<mlir_ts::YieldOp>(location);

            builder.setInsertionPointAfter(forOp);

            return mlir::success();
        }         

        mlir::LogicalResult mlirGen(ContinueStatement continueStatementAST, const GenContext &genContext)
        {
            auto location = loc(continueStatementAST);

            builder.create<mlir_ts::ContinueOp>(location);
            return mlir::success();
        }

        mlir::LogicalResult mlirGen(BreakStatement breakStatementAST, const GenContext &genContext)
        {
            auto location = loc(breakStatementAST);

            builder.create<mlir_ts::BreakOp>(location);
            return mlir::success();
        }

        mlir::LogicalResult mlirGen(SwitchStatement switchStatementAST, const GenContext &genContext)
        {
            auto location = loc(switchStatementAST);

            auto switchValue = mlirGen(switchStatementAST->expression, genContext);

            auto switchOp = builder.create<mlir_ts::SwitchOp>(location, switchValue);

            // add merge block
            switchOp.addMergeBlock();
            auto *mergeBlock = switchOp.getMergeBlock();

            auto *lastBlock = mergeBlock;
            auto *lastConditionBlock = mergeBlock;

            auto &clauses = switchStatementAST->caseBlock->clauses;
            for (int index = clauses.size() - 1; index >=0; index--)
            {
                auto caseBlock = clauses[index];
                auto statements = caseBlock->statements;
                // inline block
                if (statements.size() == 1)
                {
                    auto firstStatement = statements.front();
                    if ((SyntaxKind)firstStatement == SyntaxKind::Block)
                    {
                        statements = statements.front().as<Block>()->statements;
                    }
                }

                mlir::Block *caseBodyBlock = nullptr;
                mlir::Block *caseConditionBlock = nullptr;

                {
                    mlir::OpBuilder::InsertionGuard guard(builder);
                    caseBodyBlock = builder.createBlock(lastConditionBlock);

                    auto hasBreak = false;
                    for (auto statement : statements)
                    {
                        if ((SyntaxKind)statement == SyntaxKind::BreakStatement)
                        {
                            hasBreak = true;
                            break;
                        }

                        mlirGen(statement, genContext);
                    }

                    // exit;
                    builder.create<mlir::BranchOp>(location, hasBreak ? mergeBlock : lastBlock);

                    lastBlock = caseBodyBlock;
                }

                switch ((SyntaxKind)caseBlock)
                {
                    case SyntaxKind::CaseClause:
                        {
                            {

                                mlir::OpBuilder::InsertionGuard guard(builder);
                                caseConditionBlock = builder.createBlock(lastBlock);

                                auto caseValue = mlirGen(caseBlock.as<CaseClause>()->expression, genContext);
                                                
                                auto condition = 
                                    builder.create<mlir_ts::LogicalBinaryOp>(
                                        location,
                                        getBooleanType(),
                                        builder.getI32IntegerAttr((int)SyntaxKind::EqualsEqualsToken),
                                        switchValue,
                                        caseValue);                                

                                auto conditionI1 = builder.create<mlir_ts::CastOp>(location, builder.getI1Type(), condition);

                                builder.create<mlir::CondBranchOp>(
                                    location, 
                                    conditionI1, 
                                    caseBodyBlock, /*trueArguments=*/mlir::ValueRange{}, 
                                    lastConditionBlock, /*falseArguments=*/mlir::ValueRange{});

                                lastConditionBlock = caseConditionBlock;
                            }

                            // create condition block
                        }
                        break;
                    case SyntaxKind::DefaultClause:
                        {
                            lastConditionBlock = lastBlock;
                        }
                        break;                        
                }
            }

            return mlir::success();
        }  

        mlir::LogicalResult mlirGen(ThrowStatement throwStatementAST, const GenContext &genContext)
        {
            auto location = loc(throwStatementAST);

            // TODO: read about LLVM_ResumeOp,  maybe this is what you need (+LLVM_InvokeOp, LLVM_LandingpadOp)
            llvm_unreachable("not implemented");

            // TODO: PS, you can add param to each method to process return "exception info", and check every call for methods if they return exception info

            /*
llvm.mlir.global external constant @_ZTIi() : !llvm.ptr<i8>
llvm.func @foo(!llvm.ptr<i8>)
llvm.func @bar(!llvm.ptr<i8>) -> !llvm.ptr<i8>
llvm.func @__gxx_personality_v0(...) -> i32

// CHECK-LABEL: @invokeLandingpad
llvm.func @invokeLandingpad() -> i32 attributes { personality = @__gxx_personality_v0 } {
// CHECK: %[[a1:[0-9]+]] = alloca i8
  %0 = llvm.mlir.constant(0 : i32) : i32
  %1 = llvm.mlir.constant("\01") : !llvm.array<1 x i8>
  %2 = llvm.mlir.addressof @_ZTIi : !llvm.ptr<ptr<i8>>
  %3 = llvm.bitcast %2 : !llvm.ptr<ptr<i8>> to !llvm.ptr<i8>
  %4 = llvm.mlir.null : !llvm.ptr<ptr<i8>>
  %5 = llvm.mlir.constant(1 : i32) : i32
  %6 = llvm.alloca %5 x i8 : (i32) -> !llvm.ptr<i8>
// CHECK: invoke void @foo(i8* %[[a1]])
// CHECK-NEXT: to label %[[normal:[0-9]+]] unwind label %[[unwind:[0-9]+]]
  llvm.invoke @foo(%6) to ^bb2 unwind ^bb1 : (!llvm.ptr<i8>) -> ()

// CHECK: [[unwind]]:
^bb1:
// CHECK: %{{[0-9]+}} = landingpad { i8*, i32 }
// CHECK-NEXT:             catch i8** null
// CHECK-NEXT:             catch i8* bitcast (i8** @_ZTIi to i8*)
// CHECK-NEXT:             filter [1 x i8] c"\01"
  %7 = llvm.landingpad (catch %4 : !llvm.ptr<ptr<i8>>) (catch %3 : !llvm.ptr<i8>) (filter %1 : !llvm.array<1 x i8>) : !llvm.struct<(ptr<i8>, i32)>
// CHECK: br label %[[final:[0-9]+]]
  llvm.br ^bb3

// CHECK: [[normal]]:
// CHECK-NEXT: ret i32 1
^bb2:	// 2 preds: ^bb0, ^bb3
  llvm.return %5 : i32

// CHECK: [[final]]:
// CHECK-NEXT: %{{[0-9]+}} = invoke i8* @bar(i8* %[[a1]])
// CHECK-NEXT:          to label %[[normal]] unwind label %[[unwind]]
^bb3:	// pred: ^bb1
  %8 = llvm.invoke @bar(%6) to ^bb2 unwind ^bb1 : (!llvm.ptr<i8>) -> !llvm.ptr<i8>
}

            */

            return mlir::success();
        }

        mlir::Value mlirGen(UnaryExpression unaryExpressionAST, const GenContext &genContext)
        {
            return mlirGen(unaryExpressionAST.as<Expression>(), genContext);               
        }

        mlir::Value mlirGen(LeftHandSideExpression leftHandSideExpressionAST, const GenContext &genContext)
        {
            return mlirGen(leftHandSideExpressionAST.as<Expression>(), genContext);
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
            case SyntaxKind::PlusPlusToken:
            case SyntaxKind::MinusMinusToken:
                return builder.create<mlir_ts::PrefixUnaryOp>(
                    location,
                    expressionValue.getType(),
                    builder.getI32IntegerAttr((int)opCode),
                    expressionValue);
            default:
                llvm_unreachable("not implemented");
            }
        }        

        mlir::Value mlirGen(PostfixUnaryExpression postfixUnaryExpressionAST, const GenContext &genContext)
        {
            auto location = loc(postfixUnaryExpressionAST);

            auto opCode = postfixUnaryExpressionAST->_operator;

            auto expression = postfixUnaryExpressionAST->operand;
            auto expressionValue = mlirGen(expression, genContext);

            switch (opCode)
            {
            case SyntaxKind::PlusPlusToken:
            case SyntaxKind::MinusMinusToken:
                return builder.create<mlir_ts::PostfixUnaryOp>(
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
            if (condValue.getType() != getBooleanType())
            {
                condValue = builder.create<mlir_ts::CastOp>(location, getBooleanType(), condValue);
            }

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

            // sync right part
            if (resultTrue.getType() != resultFalse.getType())
            {
                resultFalse = builder.create<mlir_ts::CastOp>(location, resultTrue.getType(), resultFalse);
            }                    

            builder.create<mlir_ts::YieldOp>(location, mlir::ValueRange{resultFalse});

            builder.setInsertionPointAfter(ifOp);

            return ifOp.getResult(0);                                
        }

        mlir::Value evaluateBinaryOp(mlir::Location location, SyntaxKind opCode, mlir_ts::ConstantOp leftConstOp, mlir_ts::ConstantOp rightConstOp, const GenContext &genContext)
        {
            auto leftInt = leftConstOp.valueAttr().dyn_cast<mlir::IntegerAttr>().getInt();
            auto rightInt = rightConstOp.valueAttr().dyn_cast<mlir::IntegerAttr>().getInt();
            auto resultType = leftConstOp.getType();

            int64_t result = 0;
            switch (opCode)
            {
                case SyntaxKind::PlusEqualsToken: 
                    result = leftInt + rightInt;
                    break;
                case SyntaxKind::LessThanLessThanToken: 
                    result = leftInt << rightInt;
                    break;
                case SyntaxKind::GreaterThanGreaterThanToken: 
                    result = leftInt >> rightInt;
                    break;
                case SyntaxKind::AmpersandToken: 
                    result = leftInt & rightInt;
                    break;
                case SyntaxKind::BarToken:
                    result = leftInt | rightInt;
                    break;
                default: llvm_unreachable("not implemented"); break;
            }

            leftConstOp.erase();
            rightConstOp.erase();

            return builder.create<mlir_ts::ConstantOp>(location, resultType, builder.getI64IntegerAttr(result));
        }

        mlir::Value mlirGen(BinaryExpression binaryExpressionAST, const GenContext &genContext)
        {
            auto location = loc(binaryExpressionAST);

            auto saveResult = true;
            auto opCode = (SyntaxKind) binaryExpressionAST->operatorToken;
            // check if we need to save result
            switch (opCode)
            {
                case SyntaxKind::PlusEqualsToken: opCode = SyntaxKind::PlusToken; break;
                case SyntaxKind::MinusEqualsToken: opCode = SyntaxKind::MinusToken; break;
                case SyntaxKind::AsteriskEqualsToken: opCode = SyntaxKind::AsteriskToken; break;
                case SyntaxKind::AsteriskAsteriskEqualsToken: opCode = SyntaxKind::AsteriskAsteriskToken; break;
                case SyntaxKind::SlashEqualsToken: opCode = SyntaxKind::SlashToken; break;
                case SyntaxKind::PercentEqualsToken: opCode = SyntaxKind::PercentToken; break;
                case SyntaxKind::LessThanLessThanEqualsToken: opCode = SyntaxKind::LessThanLessThanToken; break;
                case SyntaxKind::GreaterThanGreaterThanEqualsToken: opCode = SyntaxKind::GreaterThanGreaterThanToken; break;
                case SyntaxKind::GreaterThanGreaterThanGreaterThanEqualsToken: opCode = SyntaxKind::GreaterThanGreaterThanGreaterThanToken; break;
                case SyntaxKind::AmpersandEqualsToken: opCode = SyntaxKind::AmpersandToken; break;
                case SyntaxKind::BarEqualsToken: opCode = SyntaxKind::BarToken; break;
                case SyntaxKind::BarBarEqualsToken: opCode = SyntaxKind::BarBarToken; break;
                case SyntaxKind::AmpersandAmpersandEqualsToken: opCode = SyntaxKind::AmpersandAmpersandToken; break;
                case SyntaxKind::QuestionQuestionEqualsToken: opCode = SyntaxKind::QuestionQuestionToken; break;
                case SyntaxKind::CaretEqualsToken: opCode = SyntaxKind::CaretToken; break;
                case SyntaxKind::EqualsToken: /*nothing to do*/ break;
                default: saveResult = false; break;
            }

            auto leftExpression = binaryExpressionAST->left;
            auto rightExpression = binaryExpressionAST->right;

            if (opCode == SyntaxKind::AmpersandAmpersandToken || opCode == SyntaxKind::BarBarToken)
            {
                return mlirGenAndOrLogic(binaryExpressionAST, genContext, opCode == SyntaxKind::AmpersandAmpersandToken);
            }

            auto leftExpressionValue = mlirGen(leftExpression, genContext);
            auto rightExpressionValue = mlirGen(rightExpression, genContext);

            if (auto unresolvedLeft = dyn_cast_or_null<mlir_ts::SymbolRefOp>(leftExpressionValue.getDefiningOp()))
            {
                emitError(location, "can't find variable: ") << unresolvedLeft.identifier();
                return mlir::Value();
            }

            if (auto unresolvedRight = dyn_cast_or_null<mlir_ts::SymbolRefOp>(rightExpressionValue.getDefiningOp()))
            {
                emitError(location, "can't find variable: ") << unresolvedRight.identifier();
                return mlir::Value();
            }

            // check if const expr.
            if (genContext.allowConstEval)
            {
                auto leftConstOp = dyn_cast_or_null<mlir_ts::ConstantOp>(leftExpressionValue.getDefiningOp());
                auto rightConstOp = dyn_cast_or_null<mlir_ts::ConstantOp>(rightExpressionValue.getDefiningOp());
                if (leftConstOp && rightConstOp)
                {
                    // try to evaluate
                    return evaluateBinaryOp(location, opCode, leftConstOp, rightConstOp, genContext);
                }
            }

            auto leftExpressionValueBeforeCast = leftExpressionValue;
            auto rightExpressionValueBeforeCast = rightExpressionValue;

            // TODO: temporary hack
            if (leftExpressionValue.getType() != rightExpressionValue.getType())
            {
                if (leftExpressionValue.getType().dyn_cast_or_null<mlir_ts::CharType>())
                {
                    leftExpressionValue = builder.create<mlir_ts::CastOp>(loc(leftExpression), getStringType(), leftExpressionValue);
                }

                if (rightExpressionValue.getType().dyn_cast_or_null<mlir_ts::CharType>())
                {
                    rightExpressionValue = builder.create<mlir_ts::CastOp>(loc(rightExpression), getStringType(), rightExpressionValue);
                }                
            }

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

            auto result = rightExpressionValue;
            switch (opCode)
            {
            case SyntaxKind::EqualsToken:
                // nothing to do;
                break;
            case SyntaxKind::EqualsEqualsToken:
            case SyntaxKind::EqualsEqualsEqualsToken:
            case SyntaxKind::ExclamationEqualsToken:
            case SyntaxKind::ExclamationEqualsEqualsToken:
            case SyntaxKind::GreaterThanToken:
            case SyntaxKind::GreaterThanEqualsToken:
            case SyntaxKind::LessThanToken:
            case SyntaxKind::LessThanEqualsToken:
                result = 
                    builder.create<mlir_ts::LogicalBinaryOp>(
                        location,
                        getBooleanType(),
                        builder.getI32IntegerAttr((int)opCode),
                        leftExpressionValue,
                        rightExpressionValue);
                break;
            case SyntaxKind::CommaToken:
                return rightExpressionValue;
            default:
                result = 
                    builder.create<mlir_ts::ArithmeticBinaryOp>(
                        location,
                        leftExpressionValue.getType(),
                        builder.getI32IntegerAttr((int)opCode),
                        leftExpressionValue,
                        rightExpressionValue);
                break;
            }

            if (saveResult)
            {
                if (leftExpressionValueBeforeCast.getType() != result.getType())
                {
                    result = builder.create<mlir_ts::CastOp>(loc(leftExpression), leftExpressionValueBeforeCast.getType(), result);
                }

                // TODO: finish it for field access, review CodeLogicHelper.saveResult
                if (auto loadOp = dyn_cast<mlir_ts::LoadOp>(leftExpressionValueBeforeCast.getDefiningOp()))
                {
                    // TODO: when saving const array into variable we need to allocate space and copy array as we need to have writable array
                    builder.create<mlir_ts::StoreOp>(
                        location,
                        result,
                        loadOp.reference());
                }
                else
                {
                    llvm_unreachable("not implemented");
                }

                if (opCode == SyntaxKind::EqualsToken)
                {
                    // special case when loadop not needed for "=" op
                    leftExpressionValueBeforeCast.getDefiningOp()->erase();
                }
            }

            return result;
        }

        mlir::Value mlirGen(SpreadElement spreadElement, const GenContext &genContext)
        {
            return mlirGen(spreadElement->expression, genContext);
        }

        mlir::Value mlirGen(ParenthesizedExpression parenthesizedExpression, const GenContext &genContext)
        {
            return mlirGen(parenthesizedExpression->expression, genContext);
        }

        mlir::Value mlirGen(PropertyAccessExpression propertyAccessExpression, const GenContext &genContext)
        {
            auto location = loc(propertyAccessExpression);

            auto expression = mlirGen(propertyAccessExpression->expression.as<Expression>(), genContext);
            auto name = mlirGen(propertyAccessExpression->name.as<Expression>(), genContext);

            mlir::Value value;

            if (!expression.getType() || expression.getType() == mlir::NoneType::get(builder.getContext()))
            {
                if (auto symRef = dyn_cast_or_null<mlir_ts::SymbolRefOp>(expression.getDefiningOp()))
                {
                    emitError(location, "can't resolve '") << symRef.identifier() << "' ...";
                }
                else
                {
                    emitError(location, "can't resolve property left expression");
                }

                return value;
            }

            MLIRPropertyAccessCodeLogic cl(builder);

            TypeSwitch<mlir::Type>(expression.getType())
                .Case<mlir_ts::EnumType>([&](auto node) 
                { 
                    value = cl.Enum(location, expression, name);
                })
                .Case<mlir_ts::TupleType>([&](auto tupleType)
                {
                    // get index
                    auto symRef = dyn_cast_or_null<mlir_ts::SymbolRefOp>(name.getDefiningOp());
                    if (symRef)
                    {
                        auto fieldIndex = tupleType.getIndex(symRef.identifier());
                        if (fieldIndex < 0)
                        {
                            emitError(location, "Tuple member '") << symRef.identifier() << "' can't be found";
                            return;
                        }

                        auto elementType = tupleType.getType(fieldIndex);

                        symRef->erase();

                        if (auto loadOp = dyn_cast_or_null<mlir_ts::LoadOp>(expression.getDefiningOp()))
                        {
                            auto propRef = builder.create<mlir_ts::PropertyRefOp>(
                                location, 
                                mlir_ts::RefType::get(elementType), 
                                loadOp.reference(), 
                                builder.getI32IntegerAttr(fieldIndex));
                            loadOp->erase();

                            value = builder.create<mlir_ts::LoadOp>(location, elementType, propRef);
                        }
                        else
                        {
                            llvm_unreachable("not implemented");            
                        }
                    }
                    else
                    {
                        llvm_unreachable("not implemented");                        
                    }                              
                })
                .Case<mlir_ts::StringType>([&](auto stringType)
                {
                    auto symRef = dyn_cast_or_null<mlir_ts::SymbolRefOp>(name.getDefiningOp());
                    if (symRef)
                    {
                        if (symRef.identifier() == "length")
                        {
                            symRef->erase();

                            // call strlen
                            value = builder.create<mlir_ts::StringLengthOp>(location, builder.getI32Type(), expression);                            
                        }
                        else
                        {
                            llvm_unreachable("not implemented");                        
                        }                         
                    }
                    else
                    {
                        llvm_unreachable("not implemented");                        
                    }                    
                })
                .Case<mlir_ts::ArrayType>([&](auto arrayType)
                {
                    auto symRef = dyn_cast_or_null<mlir_ts::SymbolRefOp>(name.getDefiningOp());
                    if (symRef)
                    {
                        if (symRef.identifier() == "length")
                        {
                            symRef->erase();

                            if (auto constOp = dyn_cast_or_null<mlir_ts::ConstantOp>(expression.getDefiningOp()))
                            {
                                // call strlen
                                auto size = constOp.getValue().dyn_cast_or_null<mlir::ArrayAttr>().size();
                                value = builder.create<mlir_ts::ConstantOp>(location, builder.getI32Type(), builder.getI32IntegerAttr(size));                            
                            }
                            else
                            {
                                llvm_unreachable("not implemented");                        
                            }                         
                        }
                        else
                        {
                            llvm_unreachable("not implemented");                        
                        }                         
                    }
                    else
                    {
                        llvm_unreachable("not implemented");                        
                    }                    
                });                

            if (value)
            {
                return value;
            }

            emitError(location, "Can't resolve property name");

            llvm_unreachable("not implemented");
        }

        mlir::Value mlirGen(ElementAccessExpression elementAccessExpression, const GenContext &genContext)
        {
            auto location = loc(elementAccessExpression);

            auto expression = mlirGen(elementAccessExpression->expression.as<Expression>(), genContext);
            auto argumentExpression = mlirGen(elementAccessExpression->argumentExpression.as<Expression>(), genContext);

            auto arrayType = expression.getType();

            mlir::Type elementType;
            if (auto arrayTyped = arrayType.dyn_cast_or_null<mlir_ts::ArrayType>())
            {
                elementType = arrayTyped.getElementType();
            }
            else if (auto vectorType = arrayType.dyn_cast_or_null<mlir::VectorType>())
            {
                elementType = vectorType.getElementType();
            }
            else if (arrayType.isa<mlir_ts::StringType>())
            {
                elementType = getCharType();
            }
            else if (auto tupleType = arrayType.dyn_cast_or_null<mlir_ts::TupleType>())
            {
                // get index
                if (auto indexConstOp = dyn_cast_or_null<mlir_ts::ConstantOp>(argumentExpression.getDefiningOp()))
                {
                    if (auto loadOp = dyn_cast_or_null<mlir_ts::LoadOp>(expression.getDefiningOp()))
                    {
                        auto constIndex = indexConstOp.value().dyn_cast_or_null<mlir::IntegerAttr>().getInt();
                        elementType = tupleType.getType(constIndex);
                        
                        auto propRef = builder.create<mlir_ts::PropertyRefOp>(location, mlir_ts::RefType::get(elementType), loadOp.reference(), builder.getI32IntegerAttr(constIndex));
                        loadOp->erase();

                        return builder.create<mlir_ts::LoadOp>(location, elementType, propRef);
                    }
                    else
                    {
                        llvm_unreachable("not implemented (load ref)");
                    }
                }
                else
                {
                    llvm_unreachable("not implemented (index)");
                }
            }
            else 
            {
                llvm_unreachable("not implemented");
            }

            auto elemRef = builder.create<mlir_ts::ElementRefOp>(location, mlir_ts::RefType::get(elementType), expression, argumentExpression);
            return builder.create<mlir_ts::LoadOp>(location, elementType, elemRef);
        }

        mlir::Value mlirGen(CallExpression callExpression, const GenContext &genContext)
        {
            auto location = loc(callExpression);

            // get function ref.
            auto funcRefValue = mlirGen(callExpression->expression.as<Expression>(), genContext);

            auto definingOp = funcRefValue.getDefiningOp();
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

                    definingOp->erase();

                    // resolve function
                    auto calledFuncIt = functionMap.find(functionName);
                    if (calledFuncIt == functionMap.end())
                    {
                        mlir::Value result;
                        // print - internal command;
                        if (functionName.compare(StringRef("print")) == 0)
                        {
                            SmallVector<mlir::Value, 4> operands;
                            mlirGen(argumentsContext, operands, genContext);
                            mlir::succeeded(mlirGenPrint(location, operands));
                        }
                        else 
                        // assert - internal command;
                        if (functionName.compare(StringRef("assert")) == 0 && opArgsCount > 0)
                        {
                            SmallVector<mlir::Value, 4> operands;
                            mlirGen(argumentsContext, operands, genContext);
                            mlir::succeeded(mlirGenAssert(location, operands));
                        }
                        else 
                        // assert - internal command;
                        if (functionName.compare(StringRef("parseInt")) == 0 && opArgsCount > 0)
                        {
                            SmallVector<mlir::Value, 4> operands;
                            mlirGen(argumentsContext, operands, genContext);
                            result = mlirGenParseInt(location, operands);
                        }
                        else 
                        if (functionName.compare(StringRef("parseFloat")) == 0 && opArgsCount > 0)
                        {
                            SmallVector<mlir::Value, 4> operands;
                            mlirGen(argumentsContext, operands, genContext);
                            result = mlirGenParseFloat(location, operands);
                        }
                        else 
                        if (!genContext.allowPartialResolve)
                        {
                            emitError(location) << "no defined function found for '" << functionName << "'";
                        }

                        return result;
                    }

                    SmallVector<mlir::Value, 4> operands;

                    auto calledFunc = calledFuncIt->second;
                    auto calledFuncType = calledFunc.getType();
                    mlirGenCallOperands(location, calledFuncType, callExpression->arguments, calledFunc.getOperation(), operands, genContext);

                    // default call by name
                    auto callOp =
                        builder.create<mlir_ts::CallOp>(
                            location,
                            calledFunc,
                            operands);

                    if (calledFuncType.getNumResults() > 0)
                    {
                        return callOp.getResult(0);
                    }

                    return nullptr;

                }
                else
                {
                    // indirect call
                    SmallVector<mlir::Value, 4> operands;

                    auto calledFuncType = funcRefValue.getType().cast<mlir::FunctionType>();
                    mlirGenCallOperands(location, calledFuncType, callExpression->arguments, nullptr/*TODO: should I finish it before refactoring?*/, operands, genContext);

                    // default call by name
                    auto callIndirectOp =
                        builder.create<mlir_ts::CallIndirectOp>(
                            location,
                            funcRefValue,
                            operands);

                    if (calledFuncType.getNumResults() > 0)
                    {
                        return callIndirectOp.getResult(0);
                    }

                    return nullptr;                    
                }
            }

            return nullptr;
        }

        mlir::LogicalResult mlirGenCallOperands(mlir::Location location, mlir::FunctionType calledFuncType, NodeArray<Expression> argumentsContext, mlir::Operation* funcOp, SmallVector<mlir::Value, 4> &operands, const GenContext &genContext) 
        {
            auto opArgsCount = std::distance(argumentsContext.begin(), argumentsContext.end());

            //auto funcArgsCount = calledFunc.getNumArguments();
            auto funcArgsCount = calledFuncType.getNumInputs();

            auto hasOptionalFrom = funcOp && funcOp->hasAttrOfType<mlir::IntegerAttr>(FUNC_OPTIONAL_ATTR_NAME);
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

            return mlir::success();
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
                auto constantOp = dyn_cast_or_null<mlir_ts::ConstantOp>(param2.getDefiningOp());
                if (constantOp && constantOp.getType().isa<mlir_ts::StringType>())
                {
                    msg = constantOp.value().cast<mlir::StringAttr>().getValue();
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
                auto typeOfValue = builder.create<mlir_ts::ConstantOp>(loc(typeOfExpression), getStringType(), getStringAttr(std::string("number")));
                return typeOfValue;
            }

            if (type == getStringType())
            {
                auto typeOfValue = builder.create<mlir_ts::ConstantOp>(loc(typeOfExpression), getStringType(), getStringAttr(std::string("string")));
                return typeOfValue;
            }

            if (type.dyn_cast_or_null<mlir_ts::ArrayType>())
            {
                auto typeOfValue = builder.create<mlir_ts::ConstantOp>(loc(typeOfExpression), getStringType(), getStringAttr(std::string("array")));
                return typeOfValue;
            }

            if (type == getBooleanType())
            {
                auto typeOfValue = builder.create<mlir_ts::ConstantOp>(loc(typeOfExpression), getStringType(), getStringAttr(std::string("boolean")));
                return typeOfValue;
            }

            if (type == getAnyType())
            {
                auto typeOfValue = builder.create<mlir_ts::ConstantOp>(loc(typeOfExpression), getStringType(), getStringAttr(std::string("object")));
                return typeOfValue;
            }

            llvm_unreachable("not implemented");
        }

        mlir::Value mlirGen(TemplateExpression templateExpressionAST, const GenContext &genContext)
        {
            auto location = loc(templateExpressionAST);

            auto stringType = getStringType();
            SmallVector<mlir::Value, 4> strs;

            auto text = wstos(templateExpressionAST->head->rawText);
            auto head = builder.create<mlir_ts::ConstantOp>(
                location,
                stringType,
                getStringAttr(text));

            // first string
            strs.push_back(head);
            for (auto span : templateExpressionAST->templateSpans)
            {
                auto exprValue = mlirGen(span->expression, genContext);
                if (exprValue.getType() != stringType)
                {
                    exprValue = builder.create<mlir_ts::CastOp>(location, stringType, exprValue);
                }

                // expr value
                strs.push_back(exprValue);

                auto spanText = wstos(span->literal->rawText);
                auto spanValue = builder.create<mlir_ts::ConstantOp>(
                    location,
                    stringType,
                    getStringAttr(spanText));

                // text
                strs.push_back(spanValue);
            }

            if (strs.size() <= 1)
            {
                return head;
            } 

            auto concatValues = builder.create<mlir_ts::StringConcatOp>(
                location,
                stringType,
                mlir::ArrayRef<mlir::Value>{strs});

            return concatValues;
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
                return builder.create<mlir_ts::ConstantOp>(
                    loc(numericLiteral),
                    builder.getI32Type(),
                    builder.getI32IntegerAttr(to_unsigned_integer(numericLiteral->text)));
            }

            if (!(numericLiteral->numericLiteralFlags & TokenFlags::NumericLiteralFlags))
            {
                return builder.create<mlir_ts::ConstantOp>(
                    loc(numericLiteral),
                    builder.getF32Type(),
                    builder.getF32FloatAttr(to_float(numericLiteral->text)));
            }

            llvm_unreachable("unknown numeric literal");
        }

        mlir::Value mlirGen(ts::StringLiteral stringLiteral, const GenContext &genContext)
        {
            auto text = wstos(stringLiteral->text);

            return builder.create<mlir_ts::ConstantOp>(
                loc(stringLiteral),
                getStringType(),
                getStringAttr(text));
        }

        mlir::Value mlirGen(ts::ArrayLiteralExpression arrayLiteral, const GenContext &genContext)
        {
            // first value
            auto isTuple = false;
            mlir::Type elementType;
            SmallVector<mlir::Type> types;
            SmallVector<mlir::Attribute> values;
            for (auto &item : arrayLiteral->elements)
            {
                auto itemValue = mlirGen(item, genContext);
                auto constOp = cast<mlir_ts::ConstantOp>(itemValue.getDefiningOp());
                if (!constOp)
                {
                    llvm_unreachable("array literal is not implemented(1)");
                    continue;
                }

                auto type = constOp.getType();

                values.push_back(constOp.valueAttr());
                types.push_back(type);
                if (!elementType)
                {
                    elementType = type;
                }
                else if (elementType != type)
                {
                    // this is tuple.
                    isTuple = true;
                }

                itemValue.getDefiningOp()->erase();            
            }

            auto arrayAttr = mlir::ArrayAttr::get(values, builder.getContext());            
            if (isTuple)
            {
                SmallVector<mlir_ts::FieldInfo> fieldInfos;
                for (auto type : types)
                {
                    fieldInfos.push_back({mlir::StringRef(), type});
                }

                return builder.create<mlir_ts::ConstantOp>(
                    loc(arrayLiteral),
                    getTupleType(fieldInfos),
                    arrayAttr);
            }

            return builder.create<mlir_ts::ConstantOp>(
                loc(arrayLiteral),
                getArrayType(elementType),
                arrayAttr);
        }

        mlir::Value mlirGen(Identifier identifier, const GenContext &genContext)
        {
            auto location = loc(identifier);

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
                    auto valueType = value.first.getType().cast<mlir_ts::RefType>().getElementType();
                    return builder.create<mlir_ts::LoadOp>(value.first.getLoc(), valueType, value.first);
                }
                else if (value.second->getIsGlobal())
                {
                    // global var
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

            // resolving function
            auto fn = theModule.lookupSymbol<mlir_ts::FuncOp>(name);
            if (fn)
            {
                return builder.create<mlir_ts::SymbolRefOp>(location, fn.getType(), mlir::FlatSymbolRefAttr::get(name, builder.getContext()));
            }            

            // check if we have enum
            if (enumsMap.count(name))
            {
                auto enumTypeInfo = enumsMap.lookup(name);
                return builder.create<mlir_ts::ConstantOp>(location, getEnumType(enumTypeInfo.first), enumTypeInfo.second);
            }

            // unresolved reference (for call for example)
            // TODO: put assert here to see which ref names are not resolved
            return builder.create<mlir_ts::SymbolRefOp>(location, mlir::FlatSymbolRefAttr::get(name, builder.getContext()));
        }

        mlir::LogicalResult mlirGen(TypeAliasDeclaration typeAliasDeclarationAST, const GenContext &genContext)
        {
            auto identOp = mlirGen(typeAliasDeclarationAST->name, genContext);
            if (auto ident = dyn_cast_or_null<mlir_ts::SymbolRefOp>(identOp.getDefiningOp()))
            {
                auto type = getType(typeAliasDeclarationAST->type);
                auto name = ident.identifier();
                typeAliasMap.insert({name, type});

                identOp.getDefiningOp()->erase();

                return mlir::success();
            }
            else
            {
                llvm_unreachable("not implemented");
            }

            return mlir::failure();
        }

        mlir::LogicalResult mlirGen(EnumDeclaration enumDeclarationAST, const GenContext &genContext)
        {
            auto identOp = mlirGen(enumDeclarationAST->name, genContext);
            if (auto ident = dyn_cast_or_null<mlir_ts::SymbolRefOp>(identOp.getDefiningOp()))
            {
                auto name = ident.identifier();

                SmallVector<mlir::NamedAttribute> enumValues;
                int64_t index = 0;
                auto activeBits = 0;
                for (auto enumMember : enumDeclarationAST->members)
                {                    
                    StringRef memberName;
                    auto memberIdentOp = mlirGen(enumMember->name.as<Expression>(), genContext);
                    if (auto memberIdent = dyn_cast_or_null<mlir_ts::SymbolRefOp>(memberIdentOp.getDefiningOp()))
                    {
                        memberName = memberIdent.identifier();
                        memberIdent->erase();
                    }
                    else
                    {
                        llvm_unreachable("not implemented");
                    }

                    mlir::Attribute enumValueAttr;
                    if (enumMember->initializer)
                    {
                        GenContext enumValueGenContext(genContext);
                        enumValueGenContext.allowConstEval = true;
                        auto enumValue = mlirGen(enumMember->initializer, enumValueGenContext);
                        if (auto constOp = dyn_cast_or_null<mlir_ts::ConstantOp>(enumValue.getDefiningOp()))
                        {
                            enumValueAttr = constOp.valueAttr();
                            if (auto intAttr = enumValueAttr.dyn_cast_or_null<mlir::IntegerAttr>())
                            {
                                index = intAttr.getInt();
                                auto currentActiveBits = (int) intAttr.getValue().getActiveBits();
                                if (currentActiveBits > activeBits)
                                {
                                    activeBits = currentActiveBits;
                                }
                            }

                            constOp->erase();
                        }
                        else
                        {
                            llvm_unreachable("not implemented");
                        }
                    }
                    else
                    {
                        enumValueAttr = builder.getI32IntegerAttr(index);
                    }

                    enumValues.push_back({ mlir::Identifier::get(memberName, builder.getContext()), enumValueAttr });
                    index++;
                }

                // count used bits
                auto indexUsingBits = std::floor(std::log2(index)) + 1;
                if (indexUsingBits > activeBits)
                {
                    activeBits = indexUsingBits;
                }                    

                // get type by size
                auto bits = 32;
                if (bits < activeBits)
                {
                    bits = 64;
                    if (bits < activeBits)
                    {
                        bits = 128;
                    }
                }

                auto enumIntType = builder.getIntegerType(bits);
                SmallVector<mlir::NamedAttribute> adjustedEnumValues;
                for (auto enumItem : enumValues)
                {
                    if (auto intAttr = enumItem.second.dyn_cast_or_null<mlir::IntegerAttr>())
                    {
                        adjustedEnumValues.push_back({ enumItem.first, mlir::IntegerAttr::get(enumIntType, intAttr.getInt() ) });
                    }
                    else
                    {
                        adjustedEnumValues.push_back(enumItem);
                    }
                }

                enumsMap.insert({ name, std::make_pair(enumIntType, mlir::DictionaryAttr::get(adjustedEnumValues, builder.getContext())) });

                identOp.getDefiningOp()->erase();

                return mlir::success();
            }
            else
            {
                llvm_unreachable("not implemented");
            }

            return mlir::failure();
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
                return getFunctionType(typeReferenceAST.as<FunctionTypeNode>());
            }
            else if (kind == SyntaxKind::TupleType)
            {
                return getTupleType(typeReferenceAST.as<TupleTypeNode>());
            }
            else if (kind == SyntaxKind::ArrayType)
            {
                return getArrayType(typeReferenceAST.as<ArrayTypeNode>());
            }
            else if (kind == SyntaxKind::UnionType)
            {
                return getUnionType(typeReferenceAST.as<UnionTypeNode>());
            }
            else if (kind == SyntaxKind::IntersectionType)
            {
                return getIntersectionType(typeReferenceAST.as<IntersectionTypeNode>());
            }
            else if (kind == SyntaxKind::ParenthesizedType)
            {
                return getParenthesizedType(typeReferenceAST.as<ParenthesizedTypeNode>());
            }            
            else if (kind == SyntaxKind::LiteralType)
            {
                return getLiteralType(typeReferenceAST.as<LiteralTypeNode>());
            } 
            else if (kind == SyntaxKind::TypeReference)
            {
                return getTypeByTypeReference(typeReferenceAST.as<TypeReferenceNode>());
            }             

            llvm_unreachable("not implemented type declaration");
            //return getAnyType();
        }

        mlir::Type getTypeByTypeReference(TypeReferenceNode typeReferenceAST)
        {
            GenContext genContext;
            auto value = mlirGen(typeReferenceAST->typeName.as<Expression>(), genContext);
            if (auto symRefOp = dyn_cast_or_null<mlir_ts::SymbolRefOp>(value.getDefiningOp()))
            {
                auto name = symRefOp.identifier();
                auto type = typeAliasMap.lookup(name);
                
                value.getDefiningOp()->erase();

                if (type)
                {
                    return type;
                }

                theModule.emitError("Type alias '") << name << "' can't be found";
            }
            else if (auto constOp = dyn_cast_or_null<mlir_ts::ConstantOp>(value.getDefiningOp()))
            {
                auto type = constOp.getType();
                if (auto enumType = type.dyn_cast_or_null<mlir_ts::EnumType>())
                {
                    // we do not exact type enum as we want to avoid casting it all the time
                    type = enumType.getElementType();
                }

                value.getDefiningOp()->erase();
                return type;
            }

            value.getDefiningOp()->erase();

            llvm_unreachable("not implemented");
        }        

        mlir_ts::VoidType getVoidType()
        {
            return mlir_ts::VoidType::get(builder.getContext());
        }

        mlir_ts::ByteType getByteType()
        {
            return mlir_ts::ByteType::get(builder.getContext());
        }

        mlir_ts::BooleanType getBooleanType()
        {
            return mlir_ts::BooleanType::get(builder.getContext());
        }

        mlir_ts::NumberType getNumberType()
        {
            return mlir_ts::NumberType::get(builder.getContext());
        }

        mlir_ts::BigIntType getBigIntType()
        {
            return mlir_ts::BigIntType::get(builder.getContext());
        }

        mlir_ts::StringType getStringType()
        {
            return mlir_ts::StringType::get(builder.getContext());
        }

        mlir_ts::CharType getCharType()
        {
            return mlir_ts::CharType::get(builder.getContext());
        }

        mlir_ts::EnumType getEnumType()
        {
            return getEnumType(builder.getI32Type());
        }  
        
        mlir_ts::EnumType getEnumType(mlir::Type elementType)
        {
            return mlir_ts::EnumType::get(elementType);
        }

        mlir_ts::ArrayType getArrayType(ArrayTypeNode arrayTypeAST)
        {
            auto type = getType(arrayTypeAST->elementType);
            return getArrayType(type);
        }   

        mlir_ts::ArrayType getArrayType(mlir::Type elementType)
        {
            return mlir_ts::ArrayType::get(elementType);
        }

        mlir_ts::TupleType getTupleType(TupleTypeNode tupleType)
        {
            mlir::SmallVector<mlir_ts::FieldInfo> types;
            for (auto typeItem : tupleType->elements)
            {
                if ((SyntaxKind)typeItem == SyntaxKind::NamedTupleMember)
                {
                    auto namedTupleMember = typeItem.as<NamedTupleMember>();
                    auto name = wstos(namedTupleMember->name.as<Identifier>()->escapedText);
                    auto namePtr = StringRef(name).copy(stringAllocator);

                    auto type = getType(namedTupleMember->type);

                    assert(type);         
                    types.push_back({namePtr, type});
                }
                else
                {
                    auto type = getType(typeItem);

                    assert(type);
                    types.push_back({mlir::StringRef(), type});
                }
            }

            return getTupleType(types);
        }        

        mlir_ts::TupleType getTupleType(mlir::SmallVector<mlir_ts::FieldInfo> &fieldInfos)
        {
            return mlir_ts::TupleType::get(builder.getContext(), fieldInfos);
        }         

        mlir::FunctionType getFunctionType(FunctionTypeNode functionType)
        {
            auto resultType = getType(functionType->type);
            SmallVector<mlir::Type> argTypes;
            for (auto paramItem : functionType->parameters)
            {
                argTypes.push_back(getType(paramItem->type));
            }

            return mlir::FunctionType::get(builder.getContext(), argTypes, resultType);
        }

        mlir::Type getUnionType(UnionTypeNode unionTypeNode)
        {
            mlir::SmallVector<mlir::Type> types;
            auto oneType = true;
            mlir::Type currentType;
            for (auto typeItem : unionTypeNode->types)
            {
                auto type = getType(typeItem);
                if (!type)
                {
                    llvm_unreachable("wrong type");
                }

                if (currentType && currentType != type)
                {
                    oneType = false;
                }

                currentType = type;

                types.push_back(type);
            }

            if (oneType)
            {
                return currentType;
            }

            return getUnionType(types);
        }        

        mlir_ts::UnionType getUnionType(mlir::SmallVector<mlir::Type> &types)
        {
            return mlir_ts::UnionType::get(builder.getContext(), types);
        }         

        mlir_ts::IntersectionType getIntersectionType(IntersectionTypeNode intersectionTypeNode)
        {
            mlir::SmallVector<mlir::Type> types;
            for (auto typeItem : intersectionTypeNode->types)
            {
                auto type = getType(typeItem);
                if (!type)
                {
                    llvm_unreachable("wrong type");
                }

                types.push_back(type);
            }

            return getIntersectionType(types);
        }        

        mlir_ts::IntersectionType getIntersectionType(mlir::SmallVector<mlir::Type> &types)
        {
            return mlir_ts::IntersectionType::get(builder.getContext(), types);
        }            

        mlir::Type getParenthesizedType(ParenthesizedTypeNode parenthesizedTypeNode)
        {
            return getType(parenthesizedTypeNode->type);
        }          

        mlir::Type getLiteralType(LiteralTypeNode literalTypeNode)
        {
            GenContext genContext;
            genContext.dummyRun = true;
            genContext.allowPartialResolve = true;
            auto value = mlirGen(literalTypeNode->literal.as<Expression>(), genContext);
            auto type = value.getType();
            value.getDefiningOp()->erase();
            return type;
        }         

        mlir_ts::AnyType getAnyType()
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

    protected:

        mlir::StringAttr getStringAttr(std::string text)
        {
            return builder.getStringAttr(StringRef(text.data(), text.length() + 1));
        }

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

        /// An allocator used for alias names.
        llvm::BumpPtrAllocator stringAllocator;

        llvm::ScopedHashTable<StringRef, VariablePairT> symbolTable;

        llvm::StringMap<mlir_ts::FuncOp> functionMap;

        llvm::StringMap<mlir::Type> typeAliasMap;

        llvm::StringMap< std::pair<mlir::Type, mlir::DictionaryAttr> > enumsMap;

        // helper to get line number
        Parser parser;
        ts::SourceFile sourceFile;

        mlir::OpBuilder::InsertPoint functionBeginPoint;
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
        return MLIRGenImpl(context, fileName).mlirGenSourceFile(sourceFile);
    }

} // namespace typescript
