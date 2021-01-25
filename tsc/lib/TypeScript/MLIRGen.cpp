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
            for (auto *statement : moduleAST->statement())
            {
                if (auto *expressionStatement = statement->expressionStatement())
                {
                    auto op = mlirGen(expressionStatement);
                    if (op)
                    {
                        //theModule.push_back(op);
                    }
                }
                else
                {
                    llvm_unreachable("unknown statement");
                }
            }

            // Verify the module after we have finished constructing it, this will check
            // the structural properties of the IR and invoke any specific verifiers we
            // have on the TypeScript operations.
            if (failed(mlir::verify(theModule)))
            {
                theModule.emitError("module verification error");
                return nullptr;
            }

            return theModule;
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

            // get function ref.
            if (auto *memberExpression = callExpression->memberExpression())
            {
                mlirGen(memberExpression);
            }
            else if (auto *callExpressionRecursive = callExpression->callExpression())
            {
                mlirGen(callExpressionRecursive);
            }

            // process arguments
            mlirGen(callExpression->arguments());

            SmallVector<mlir::Value, 0> operands;
            /*
            SmallVector<mlir::Value, 4> operands;
            for (auto &expr : call.getArgs())
            {
                auto arg = mlirGen(*expr);
                if (!arg)
                {
                    return nullptr;
                }
                
                operands.push_back(arg);
            }
            */

            // declare
            SmallVector<mlir::Type, 0> argTypes;
            auto fnType = builder.getFunctionType(argTypes, llvm::None);
            auto function = builder.create<mlir::FuncOp>(theModule.getLoc(), "test", fnType);
            function.setPrivate();

            // empty block
            /*
            auto &entryBlock = *function.addEntryBlock();
            auto point = builder.saveInsertionPoint();
            builder.setInsertionPointToStart(&entryBlock);
            builder.create<mlir::ReturnOp>(theModule.getLoc());   
            builder.restoreInsertionPoint(point);
            */

            auto funcOp = mlir::FuncOp::create(theModule.getLoc(), "test", fnType);
            auto &entryBlock = *function.addEntryBlock();
            builder.setInsertionPointToStart(&entryBlock);

            // // declare printf
            // // Create a function declaration for printf, the signature is:
            // //   * `i32 (i8*, ...)`
            // auto context = theModule.getContext();
            // auto llvmI32Ty = mlir::LLVM::LLVMIntegerType::get(context, 32);
            // auto llvmI8PtrTy = mlir::LLVM::LLVMPointerType::get(mlir::LLVM::LLVMIntegerType::get(context, 8));
            // auto llvmFnType = mlir::LLVM::LLVMFunctionType::get(llvmI32Ty, llvmI8PtrTy, /*isVarArg=*/true);
            // builder.create<mlir::LLVM::LLVMFuncOp>(theModule.getLoc(), "printf", llvmFnType);

            auto callOp = builder.create<mlir::CallOp>(
                    location,
                    builder.getSymbolRefAttr("test"),
                    llvm::None,
                    operands);

            // no result
            return nullptr;
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
                //return mlirGenIdentifierName(memberExpression->IdentifierName());
                llvm_unreachable("not implemented");
            }
        }

        void mlirGen(TypeScriptParserANTLR::ArgumentsContext *arguments)
        {
            auto firstExpression = arguments->expression();
            // first argument
            if (firstExpression.size() == 0)
            {
                return;
            }

            auto *first = firstExpression.front();

            mlirGen(first);

            auto index = 0;
            while (true)
            {
                auto *next = arguments->expression(index++);
                if (!next)
                {
                    break;
                }

                mlirGen(next);
            }
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
            llvm_unreachable("not implemented");
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
            auto name = mlirGenIdentifierName(identifierReference->IdentifierName());
            return IdentifierReferenceOp::create(theModule.getLoc(), name);
        }

        StringRef mlirGenIdentifierName(antlr4::tree::TerminalNode *identifierName)
        {
            return identifierName->getText();
        }

        mlir::Value mlirGenStringLiteral(antlr4::tree::TerminalNode *stringLiteral)
        {
            // TODO:
            llvm_unreachable("not implemented");
        }

        mlir::Value mlirGenDecimalLiteral(antlr4::tree::TerminalNode *decimalLiteral)
        {
            // TODO:
            llvm_unreachable("not implemented");
        }

        mlir::Value mlirGenDecimalIntegerLiteral(antlr4::tree::TerminalNode *decimalIntegerLiteral)
        {
            // TODO:
            llvm_unreachable("not implemented");
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
