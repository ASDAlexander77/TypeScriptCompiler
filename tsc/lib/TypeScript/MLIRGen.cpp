#include "TypeScript/MLIRGen.h"
#include "TypeScript/TypeScriptDialect.h"
#include "TypeScript/TypeScriptOps.h"

#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Verifier.h"

#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"

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
    struct ValueOrString
    {
        enum VariantEnum
        {
            Empty,
            Value,
            StringRef
        };

    public:
        ValueOrString() : variant(VariantEnum::Empty), values{} {}

        ValueOrString(mlir::Value value) : variant(VariantEnum::Value), values{value} {}

        ValueOrString(llvm::StringRef value) : variant(VariantEnum::StringRef), values{value} {}

        template <typename T>
        bool constexpr has()
        {
            return false;
        }

        template <>
        bool constexpr has<mlir::Value>()
        {
            return variant == VariantEnum::Value;
        }

        template <>
        bool constexpr has<llvm::StringRef>()
        {
            return variant == VariantEnum::StringRef;
        }

        explicit constexpr operator mlir::Value()
        {
            return values.value;
        }

        explicit constexpr operator llvm::StringRef()
        {
            return values.strRef;
        }

        ValueOrString &operator=(mlir::Value value)
        {
            variant = VariantEnum::Value;
            values = value;
            return *this;
        }

        ValueOrString &operator=(llvm::StringRef value)
        {
            variant = VariantEnum::StringRef;
            values = value;
            return *this;
        }

        VariantEnum variant;
        union Union
        {
            Union() : value(nullptr) {}

            Union(mlir::Value value) : value(value) {}

            Union(llvm::StringRef value) : strRef(value) {}

            void *empty;
            mlir::Value value;
            llvm::StringRef strRef;
        } values;
    };

    /// Implementation of a simple MLIR emission from the TypeScript AST.
    ///
    /// This will emit operations that are specific to the TypeScript language, preserving
    /// the semantics of the language and (hopefully) allow to perform accurate
    /// analysis and transformation based on these high level semantics.
    class MLIRGenImpl
    {
    public:
        MLIRGenImpl(const mlir::MLIRContext &context) : builder(&const_cast<mlir::MLIRContext &>(context)) {}

        /// Public API: convert the AST for a TypeScript module (source file) to an MLIR
        /// Module operation.
        mlir::ModuleOp mlirGen(TypeScriptParserANTLR::MainContext *moduleAST)
        {
            // We create an empty MLIR module and codegen functions one at a time and
            // add them to the module.
            theModule = mlir::ModuleOp::create(builder.getUnknownLoc());
            builder.setInsertionPointToStart(theModule.getBody());

            // Process generating here
            for (auto *declaration : moduleAST->declaration())
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

        mlir::LogicalResult mlirGen(TypeScriptParserANTLR::FormalParametersContext *formalParametersContextAST, const llvm::SmallVector<mlir::Type, 0> &argTypesParam)
        {
            auto argTypes = const_cast<llvm::SmallVector<mlir::Type, 0> &>(argTypesParam);
            if (formalParametersContextAST)
            {
                argTypes.reserve(formalParametersContextAST->formalParameter().size());
                for (auto &arg : formalParametersContextAST->formalParameter())
                {
                    mlir::Type type = getType(arg, theModule.getLoc());
                    if (!type)
                    {
                        return mlir::failure();
                    }

                    argTypes.push_back(type);
                }
            }

            return mlir::success();
        }

        mlir::FuncOp mlirGenFunctionPrototype(TypeScriptParserANTLR::FunctionDeclarationContext *functionDeclarationAST)
        {
            auto location = theModule.getLoc();

            // This is a generic function, the return type will be inferred later.
            llvm::SmallVector<mlir::Type, 0> argTypes;
            if (mlir::failed(mlirGen(functionDeclarationAST->formalParameters(), argTypes)))
            {
                return nullptr;
            }

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

            auto func_type = builder.getFunctionType(argTypes, llvm::None);
            return mlir::FuncOp::create(location, StringRef(name), func_type);
        }

        mlir::LogicalResult mlirGen(TypeScriptParserANTLR::FunctionDeclarationContext *functionDeclarationAST)
        {
            auto funcOp = mlirGenFunctionPrototype(functionDeclarationAST);
            if (!funcOp)
            {
                return mlir::failure();
            }

            auto &entryBlock = *funcOp.addEntryBlock();
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
                builder.create<mlir::ReturnOp>(theModule.getLoc());
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

            return mlir::success();
        }

        mlir::Value mlirGen(TypeScriptParserANTLR::StatementContext *statementAST)
        {
            if (auto *expressionStatement = statementAST->expressionStatement())
            {
                return mlirGen(expressionStatement);
            }
            else
            {
                llvm_unreachable("unknown statement");
            }
        }

        mlir::Value mlirGen(TypeScriptParserANTLR::ExpressionStatementContext *expressionStatementAST)
        {
            return mlirGen(expressionStatementAST->expression());
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
            auto location = loc(callExpression->getSourceInterval());

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

                    // default call by name
                    auto callOp =
                        builder.create<mlir::CallOp>(
                            location,
                            resultType, // result
                            functionName,
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

        mlir::LogicalResult mlirGenPrint(const mlir::Location& location, const SmallVector<mlir::Value, 0>& operands)
        {
            auto printOp =
                builder.create<PrintOp>(
                    location,
                    operands.front());

            return mlir::success();
        }

        mlir::LogicalResult mlirGenAssert(const mlir::Location& location, const SmallVector<mlir::Value, 0>& operands)
        {
            auto msg = StringRef("assert");
            if (operands.size() > 1)
            {
                auto param2 = operands[1];
                auto definingOpParam2 = param2.getDefiningOp();
                auto valueAttrName = StringRef("value");
                if (definingOpParam2
                    && definingOpParam2->hasAttrOfType<mlir::StringAttr>(valueAttrName))
                {
                    auto valueAttr = definingOpParam2->getAttrOfType<mlir::StringAttr>(valueAttrName);
                    msg = valueAttr.getValue();
                    definingOpParam2->erase();
                }
            }

            auto assertOp =
                builder.create<mlir::AssertOp>(
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
            if (booleanLiteral->TRUE_KEYWORD())
            {
                return 
                    builder.create<mlir::ConstantOp>(
                        theModule.getLoc(),
                        builder.getI1Type(),
                        mlir::BoolAttr::get(true, theModule.getContext()));
            }
            else if (booleanLiteral->FALSE_KEYWORD())
            {
                return 
                    builder.create<mlir::ConstantOp>(
                        theModule.getLoc(),
                        builder.getI1Type(),
                        mlir::BoolAttr::get(false, theModule.getContext()));
            }
            else
            {
                llvm_unreachable("not implemented");
            }
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
            //return builder.create<IdentifierReference>(theModule.getLoc(), builder.getNoneType(), identifierName->getText());
            return IdentifierReference::create(theModule.getLoc(), identifierName->getText());
        }

        mlir::Value mlirGenStringLiteral(antlr4::tree::TerminalNode *stringLiteral)
        {
            auto text = stringLiteral->getText();
            auto innerText = text.substr(1, text.length() - 2);

            return builder.create<mlir::ConstantOp>(
                theModule.getLoc(),
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
                theModule.getLoc(),
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

        mlir::Type getType(TypeScriptParserANTLR::FormalParameterContext *formalParameterAST, mlir::Location loc)
        {
            // TODO: finish it.
            // return default type, pointer to any type
            return mlir::UnrankedMemRefType::get(builder.getI1Type(), 0);
        }

    private:
        /// A "module" matches a TypeScript source file: containing a list of functions.
        mlir::ModuleOp theModule;

        /// The builder is a helper class to create IR inside a function. The builder
        /// is stateful, in particular it keeps an "insertion point": this is where
        /// the next operations will be introduced.
        mlir::OpBuilder builder;

        /// Helper conversion for a TypeScript AST location to an MLIR location.
        mlir::Location loc(const antlr4::misc::Interval &loc)
        {
            //return builder.getFileLineColLoc(builder.getIdentifier(""), loc.a, loc.b);
            return builder.getUnknownLoc();
        }
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

    mlir::OwningModuleRef mlirGenFromSource(const mlir::MLIRContext &context, const llvm::StringRef &source)
    {
        antlr4::ANTLRInputStream input((std::string)source);
        typescript::TypeScriptLexerANTLR lexer(&input);
        antlr4::CommonTokenStream tokens(&lexer);
        typescript::TypeScriptParserANTLR parser(&tokens);
        return MLIRGenImpl(context).mlirGen(parser.main());
    }

} // namespace typescript
