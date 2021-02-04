#include "TypeScript/MLIRGen.h"
#include "TypeScript/TypeScriptDialect.h"
#include "TypeScript/TypeScriptOps.h"

#include "TypeScript/DOM.h"

#include "mlir/IR/Types.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Verifier.h"

#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/ScopedHashTable.h"
#include "llvm/Support/raw_ostream.h"

#include "TypeScriptLexerANTLR.h"
#include "TypeScriptParserANTLR.h"

#include <numeric>

using namespace mlir::typescript;
using namespace typescript;

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
    /// Implementation of a simple MLIR emission from the TypeScript AST.
    ///
    /// This will emit operations that are specific to the TypeScript language, preserving
    /// the semantics of the language and (hopefully) allow to perform accurate
    /// analysis and transformation based on these high level semantics.
    class MLIRGenImpl
    {
        using VariablePairT = std::pair<mlir::Value, VariableDeclarationDOM*>;
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

        /// Public API: convert the AST for a TypeScript module (source file) to an MLIR
        /// Module operation.
        mlir::ModuleOp mlirGen(TypeScriptParserANTLR::MainContext *module)
        {
            // We create an empty MLIR module and codegen functions one at a time and
            // add them to the module.
            theModule = mlir::ModuleOp::create(loc(module), fileName);
            builder.setInsertionPointToStart(theModule.getBody());

            theModuleDOM.parseTree = module;

            // Process generating here
            for (auto *declaration : module->declaration())
            {
                if (failed(mlirGen(declaration)))
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
                //theModule.emitError("module verification error");
                //return nullptr;
            }

            return theModule;
        }

        mlir::LogicalResult mlirGen(TypeScriptParserANTLR::DeclarationContext *declarationAST)
        {
            if (auto *functionDeclaration = declarationAST->functionDeclaration())
            {
                mlirGen(functionDeclaration);
            }
            else
            {
                llvm_unreachable("unknown statement");
            }

            return mlir::success();
        }

        std::vector<std::unique_ptr<FunctionParamDOM>> mlirGen(TypeScriptParserANTLR::FormalParametersContext *formalParametersContextAST)
        {
            std::vector<std::unique_ptr<FunctionParamDOM>> params;
            if (!formalParametersContextAST)
            {
                return params;
            }

            for (auto &arg : formalParametersContextAST->formalParameter())
            {
                auto name = arg->IdentifierName()->getText();
                auto type = getType(arg);
                if (!type)
                {
                    return params;
                }

                params.push_back(std::make_unique<FunctionParamDOM>(arg, name, type));
            }

            return params;
        }

        std::pair<mlir::FuncOp, FunctionPrototypeDOM::TypePtr> mlirGenFunctionPrototype(TypeScriptParserANTLR::FunctionDeclarationContext *functionDeclarationAST)
        {
            auto location = loc(functionDeclarationAST);

            // This is a generic function, the return type will be inferred later.
            std::vector<FunctionParamDOM::TypePtr> params = mlirGen(functionDeclarationAST->formalParameters());

            std::string name;
            auto *identifier = functionDeclarationAST->IdentifierName();
            if (identifier)
            {
                name = identifier->getText();
            }
            else
            {
                // auto calculate name
                // __func+location
            }

            SmallVector<mlir::Type> argTypes;
            for (const auto &param : params)
            {
                argTypes.push_back(param->getType());
            }

            mlir::FunctionType funcType;
            if (auto *typeParameter = functionDeclarationAST->typeParameter())
            {
                auto returnType = getType(typeParameter);
                funcType = builder.getFunctionType(argTypes, returnType);
            }
            else
            {
                funcType = builder.getFunctionType(argTypes, llvm::None);
            }

            auto funcOp = mlir::FuncOp::create(location, StringRef(name), funcType);

            return std::make_pair(funcOp, std::make_unique<FunctionPrototypeDOM>(functionDeclarationAST, name, std::move(params)));
        }

        mlir::LogicalResult mlirGen(TypeScriptParserANTLR::FunctionDeclarationContext *functionDeclarationAST)
        {
            SymbolTableScopeT varScope(symbolTable);
            auto funcOpWithFuncProto = mlirGenFunctionPrototype(functionDeclarationAST);
            auto &funcOp = funcOpWithFuncProto.first;
            auto &funcProto = funcOpWithFuncProto.second;
            if (!funcOp)
            {
                return mlir::failure();
            }

            auto &entryBlock = *funcOp.addEntryBlock();

            // process function params
            for (const auto paramPairs : llvm::zip(funcProto->getArgs(), entryBlock.getArguments()))
            {
                if (failed(declare(*std::get<0>(paramPairs), std::get<1>(paramPairs))))
                {
                    return mlir::failure();
                }
            }

            builder.setInsertionPointToStart(&entryBlock);

            for (auto *statementListItem : functionDeclarationAST->functionBody()->statementListItem())
            {
                if (auto *statement = statementListItem->statement())
                {
                    mlirGen(statement);
                }
                else if (auto *declaration = statementListItem->declaration())
                {
                    mlirGen(declaration);
                }
                else
                {
                    llvm_unreachable("unknown statement");
                }
            }

            // add return
            mlir::ReturnOp returnOp;
            if (!entryBlock.empty())
            {
                returnOp = dyn_cast<mlir::ReturnOp>(entryBlock.back());
            }

            if (!returnOp)
            {
                builder.create<mlir::ReturnOp>(loc(functionDeclarationAST->functionBody()));
            }
            else if (!returnOp.operands().empty())
            {
                // Otherwise, if this return operation has an operand then add a result to
                // the function.
                funcOp.setType(
                    builder.getFunctionType(
                        funcOp.getType().getInputs(),
                        *returnOp.operand_type_begin()));
            }

            // set visibility index
            if (functionDeclarationAST->IdentifierName()->getText() != "main")
            {
                funcOp.setPrivate();
            }

            theModule.push_back(funcOp);
            functionMap.insert({funcOp.getName(), funcOp});
            theModuleDOM.getFunctionProtos().push_back(std::move(funcProto));

            return mlir::success();
        }

        mlir::LogicalResult mlirGen(TypeScriptParserANTLR::StatementContext *statementAST)
        {
            if (auto *expression = statementAST->expression())
            {
                mlirGen(expression);
                // ignore result in statement
                return mlir::success();
            }
            else if (auto *returnStatement = statementAST->returnStatement())
            {
                return mlirGen(returnStatement);
            }
            else
            {
                llvm_unreachable("unknown statement");
            }
        }

        mlir::LogicalResult mlirGen(TypeScriptParserANTLR::ReturnStatementContext *returnStatementAST)
        {
            if (auto *expression = returnStatementAST->expression())
            {
                auto expressionValue = mlirGen(expression);
                builder.create<mlir::ReturnOp>(loc(returnStatementAST), expressionValue);
            }
            else
            {
                builder.create<mlir::ReturnOp>(loc(returnStatementAST));
            }

            return mlir::success();
        }

        mlir::Value mlirGen(TypeScriptParserANTLR::ExpressionContext *expressionAST)
        {
            if (auto *primaryExpression = expressionAST->primaryExpression())
            {
                return mlirGen(primaryExpression);
            }
            else if (auto *leftHandSideExpression = expressionAST->leftHandSideExpression())
            {
                return mlirGen(leftHandSideExpression);
            }
            else
            {
                llvm_unreachable("unknown statement");
            }
        }

        mlir::Value mlirGen(TypeScriptParserANTLR::PrimaryExpressionContext *primaryExpression)
        {
            if (auto *literal = primaryExpression->literal())
            {
                return mlirGen(literal);
            }
            else if (auto *identifierReference = primaryExpression->identifierReference())
            {
                return mlirGen(identifierReference);
            }
            else
            {
                llvm_unreachable("unknown statement");
            }
        }

        mlir::Value mlirGen(TypeScriptParserANTLR::LeftHandSideExpressionContext *leftHandSideExpression)
        {
            if (auto *callExpression = leftHandSideExpression->callExpression())
            {
                return mlirGen(callExpression);
            }
            else
            {
                llvm_unreachable("unknown statement");
            }
        }

        mlir::Value mlirGen(TypeScriptParserANTLR::CallExpressionContext *callExpression)
        {
            auto location = loc(callExpression);

            mlir::Value result;

            // get function ref.
            if (auto *memberExpression = callExpression->memberExpression())
            {
                result = mlirGen(memberExpression);
            }
            else if (auto *callExpressionRecursive = callExpression->callExpression())
            {
                result = mlirGen(callExpressionRecursive);
            }

            auto definingOp = result.getDefiningOp();
            if (definingOp)
            {
                auto opName = definingOp->getName().getStringRef();
                auto attrName = StringRef("identifier");
                if (definingOp->hasAttrOfType<mlir::FlatSymbolRefAttr>(attrName))
                {
                    auto calleeName = definingOp->getAttrOfType<mlir::FlatSymbolRefAttr>(attrName);
                    auto functionName = calleeName.getValue();

                    // process arguments
                    SmallVector<mlir::Value, 0> operands;
                    mlirGen(callExpression->arguments(), operands);

                    // result;
                    auto hasResult = false;
                    auto resultType = builder.getNoneType();

                    // print - internal command;
                    if (functionName.compare(StringRef("print")) == 0)
                    {
                        if (mlir::succeeded(mlirGenPrint(location, operands)))
                        {
                            return nullptr;
                        }
                    }

                    // assert - internal command;
                    if (functionName.compare(StringRef("assert")) == 0 && operands.size() > 0)
                    {
                        if (mlir::succeeded(mlirGenAssert(location, operands)))
                        {
                            return nullptr;
                        }
                    }

                    // resolve function
                    auto calledFuncIt = functionMap.find(functionName);
                    if (calledFuncIt == functionMap.end()) {
                        emitError(location) << "no defined function found for '" << functionName << "'";
                        return nullptr;
                    }

                    auto calledFunc = calledFuncIt->second;

                    // default call by name
                    auto callOp =
                        builder.create<mlir::CallOp>(
                            location,
                            calledFunc,
                            operands);

                    if (hasResult)
                    {
                        return callOp.getResult(0);
                    }

                    return nullptr;
                }
            }

            return nullptr;
        }

        mlir::LogicalResult mlirGenPrint(const mlir::Location &location, const SmallVector<mlir::Value, 0> &operands)
        {
            auto printOp =
                builder.create<PrintOp>(
                    location,
                    operands.front());

            return mlir::success();
        }

        mlir::LogicalResult mlirGenAssert(const mlir::Location &location, const SmallVector<mlir::Value, 0> &operands)
        {
            auto msg = StringRef("assert");
            if (operands.size() > 1)
            {
                auto param2 = operands[1];
                auto definingOpParam2 = param2.getDefiningOp();
                auto valueAttrName = StringRef("value");
                if (definingOpParam2 && definingOpParam2->hasAttrOfType<mlir::StringAttr>(valueAttrName))
                {
                    auto valueAttr = definingOpParam2->getAttrOfType<mlir::StringAttr>(valueAttrName);
                    msg = valueAttr.getValue();
                    definingOpParam2->erase();
                }
            }

            auto assertOp =
                builder.create<AssertOp>(
                    location,
                    operands.front(),
                    mlir::StringAttr::get(msg, theModule.getContext()));

            return mlir::success();
        }

        mlir::Value mlirGen(TypeScriptParserANTLR::MemberExpressionContext *memberExpression)
        {
            if (auto *primaryExpression = memberExpression->primaryExpression())
            {
                return mlirGen(primaryExpression);
            }
            else if (auto *memberExpressionRecursive = memberExpression->memberExpression())
            {
                return mlirGen(memberExpressionRecursive);
            }
            else
            {
                return mlirGenIdentifierName(memberExpression->IdentifierName());
            }
        }

        mlir::LogicalResult mlirGen(TypeScriptParserANTLR::ArgumentsContext *arguments, SmallVector<mlir::Value, 0> &operands)
        {
            for (auto &next : arguments->expression())
            {
                operands.push_back(mlirGen(next));
            }

            return mlir::success();
        }

        mlir::Value mlirGen(TypeScriptParserANTLR::LiteralContext *literal)
        {
            if (auto *nullLiteral = literal->nullLiteral())
            {
                return mlirGen(nullLiteral);
            }
            else if (auto *booleanLiteral = literal->booleanLiteral())
            {
                return mlirGen(booleanLiteral);
            }
            else if (auto *numericLiteral = literal->numericLiteral())
            {
                return mlirGen(numericLiteral);
            }
            else
            {
                return mlirGenStringLiteral(literal->StringLiteral());
            }
        }

        mlir::Value mlirGen(TypeScriptParserANTLR::NullLiteralContext *nullLiteral)
        {
            llvm_unreachable("not implemented");
        }

        mlir::Value mlirGen(TypeScriptParserANTLR::BooleanLiteralContext *booleanLiteral)
        {
            bool result;
            if (booleanLiteral->TRUE_KEYWORD())
            {
                result = true;
            }
            else if (booleanLiteral->FALSE_KEYWORD())
            {
                result = false;
            }
            else
            {
                llvm_unreachable("not implemented");
            }

            return 
                builder.create<mlir::ConstantOp>(
                    loc(booleanLiteral),
                    builder.getI1Type(),
                    mlir::BoolAttr::get(result, theModule.getContext()));
        }

        mlir::Value mlirGen(TypeScriptParserANTLR::NumericLiteralContext *numericLiteral)
        {
            if (auto *decimalLiteral = numericLiteral->DecimalLiteral())
            {
                return mlirGenDecimalLiteral(decimalLiteral);
            }
            else if (auto *decimalIntegerLiteral = numericLiteral->DecimalIntegerLiteral())
            {
                return mlirGenDecimalIntegerLiteral(decimalIntegerLiteral);
            }
            else if (auto *decimalBigIntegerLiteral = numericLiteral->DecimalBigIntegerLiteral())
            {
                return mlirGenDecimalBigIntegerLiteral(decimalBigIntegerLiteral);
            }
            else if (auto *binaryBigIntegerLiteral = numericLiteral->BinaryBigIntegerLiteral())
            {
                return mlirGenBinaryBigIntegerLiteral(binaryBigIntegerLiteral);
            }
            else if (auto *octalBigIntegerLiteral = numericLiteral->OctalBigIntegerLiteral())
            {
                return mlirGenOctalBigIntegerLiteral(octalBigIntegerLiteral);
            }
            else if (auto *hexBigIntegerLiteral = numericLiteral->HexBigIntegerLiteral())
            {
                return mlirGenHexBigIntegerLiteral(hexBigIntegerLiteral);
            }
            else
            {
                llvm_unreachable("unknown statement");
            }
        }

        mlir::Value mlirGen(TypeScriptParserANTLR::IdentifierReferenceContext *identifierReference)
        {
            return mlirGenIdentifierName(identifierReference->IdentifierName());
        }

        mlir::Value mlirGenIdentifierName(antlr4::tree::TerminalNode *identifierName)
        {
            // resolve name
            auto name = identifierName->getText();

            auto value = resolve(name);
            if (value)
            {
                return value;
            }

            // unresolved reference (for call for example)
            return IdentifierReference::create(loc(identifierName), name);
        }

        mlir::Value mlirGenStringLiteral(antlr4::tree::TerminalNode *stringLiteral)
        {
            auto text = stringLiteral->getText();
            auto innerText = text.substr(1, text.length() - 2);

            return builder.create<mlir::ConstantOp>(
                loc(stringLiteral),
                mlir::UnrankedTensorType::get(mlir::IntegerType::get(theModule.getContext(), 8)),
                builder.getStringAttr(StringRef(innerText)));
        }

        mlir::Value mlirGenDecimalLiteral(antlr4::tree::TerminalNode *decimalLiteral)
        {
            // TODO:
            llvm_unreachable("not implemented");
        }

        mlir::Value mlirGenDecimalIntegerLiteral(antlr4::tree::TerminalNode *decimalIntegerLiteral)
        {
            return builder.create<mlir::ConstantOp>(
                loc(decimalIntegerLiteral),
                builder.getI32Type(),
                builder.getI32IntegerAttr(std::stoi(decimalIntegerLiteral->getText())));
        }

        mlir::Value mlirGenDecimalBigIntegerLiteral(antlr4::tree::TerminalNode *decimalBigIntegerLiteraligIntegerLiteral)
        {
            // TODO:
            llvm_unreachable("not implemented");
        }

        mlir::Value mlirGenBinaryBigIntegerLiteral(antlr4::tree::TerminalNode *binaryBigIntegerLiteral)
        {
            // TODO:
            llvm_unreachable("not implemented");
        }

        mlir::Value mlirGenOctalBigIntegerLiteral(antlr4::tree::TerminalNode *octalBigIntegerLiteral)
        {
            // TODO:
            llvm_unreachable("not implemented");
        }

        mlir::Value mlirGenHexBigIntegerLiteral(antlr4::tree::TerminalNode *hexBigIntegerLiteral)
        {
            // TODO:
            llvm_unreachable("not implemented");
        }

        mlir::Type getType(TypeScriptParserANTLR::FormalParameterContext *formalParameterAST)
        {
            if (auto *typeParameter = formalParameterAST->typeParameter())
            {
                return getType(typeParameter);
            }

            return getAnyType();
        }

        mlir::Type getType(TypeScriptParserANTLR::TypeParameterContext *typeParameterAST)
        {
            if (auto *typeDeclaration = typeParameterAST->typeDeclaration())
            {
                return getType(typeDeclaration);
            }

            return getAnyType();
        }        

        mlir::Type getType(TypeScriptParserANTLR::TypeDeclarationContext *typeDeclarationAST)
        {
            if (auto boolean = typeDeclarationAST->BOOLEAN_KEYWORD())
            {
                return builder.getI1Type();
            }
            else if (auto boolean = typeDeclarationAST->NUMBER_KEYWORD())
            {
                return builder.getF32Type();
            }
            else if (auto boolean = typeDeclarationAST->BIGINT_KEYWORD())
            {
                return builder.getI64Type();
            }
            else if (auto boolean = typeDeclarationAST->STRING_KEYWORD())
            {
                return getStringType();
            }

            return getAnyType();
        }        

        mlir::Type getStringType()
        {
            return mlir::UnrankedMemRefType::get(builder.getI1Type(), 0);
        }

        mlir::Type getAnyType()
        {
            return mlir::UnrankedMemRefType::get(builder.getI1Type(), 0);
        }

        mlir::LogicalResult declare(VariableDeclarationDOM& var, mlir::Value value)
        {
            const auto &name = var.getName();
            if (symbolTable.count(name))
            {
                return mlir::failure();
            }

            symbolTable.insert(name, {value, &var});
            return mlir::success();
        }

        mlir::Value resolve(StringRef name)
        {
            auto varIt = symbolTable.lookup(name);
            if (varIt.first)
            {
                return varIt.first;
            }

            return nullptr;
        }        

    private:
        /// Helper conversion for a TypeScript AST location to an MLIR location.
        mlir::Location loc(antlr4::tree::ParseTree *tree)
        {
            const antlr4::misc::Interval &loc = tree->getSourceInterval();
            return builder.getFileLineColLoc(builder.getIdentifier(fileName), loc.a, loc.b);
        }

        /// A "module" matches a TypeScript source file: containing a list of functions.
        mlir::ModuleOp theModule;
        ModuleDOM theModuleDOM;

        /// The builder is a helper class to create IR inside a function. The builder
        /// is stateful, in particular it keeps an "insertion point": this is where
        /// the next operations will be introduced.
        mlir::OpBuilder builder;

        mlir::StringRef fileName;

        llvm::ScopedHashTable<StringRef, VariablePairT> symbolTable;

        llvm::StringMap<mlir::FuncOp> functionMap;
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
