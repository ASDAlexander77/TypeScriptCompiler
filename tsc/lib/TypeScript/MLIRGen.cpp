#include "TypeScript/MLIRGen.h"
#include "TypeScript/TypeScriptDialect.h"

#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Verifier.h"

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

            // Process generating here
            for (auto *statement : moduleAST->statement())
            {
                if (auto *expressionStatement = statement->expressionStatement())
                {
                    mlirGen(expressionStatement);
                }
                else
                {
                    llvm_unreachable("unknown record type");
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

        void mlirGen(TypeScriptParserANTLR::ExpressionStatementContext *expressionStatementAST)
        {
            mlirGen(expressionStatementAST->expression());
        }

        void mlirGen(TypeScriptParserANTLR::ExpressionContext *expressionAST)
        {
            if (auto *primaryExpression = expressionAST->primaryExpression())
            {
                mlirGen(primaryExpression);
            }
            else if (auto *leftHandSideExpression = expressionAST->leftHandSideExpression())
            {
                mlirGen(leftHandSideExpression);
            }
        }

        void mlirGen(TypeScriptParserANTLR::PrimaryExpressionContext *primaryExpression)
        {
            if (auto *literal = primaryExpression->literal())
            {
                mlirGen(literal);
            }
            else if (auto *identifierReference = primaryExpression->identifierReference())
            {
                mlirGen(identifierReference);
            }
        }

        void mlirGen(TypeScriptParserANTLR::LeftHandSideExpressionContext *leftHandSideExpression)
        {
            if (auto *callExpression = leftHandSideExpression->callExpression())
            {
                mlirGen(callExpression);
            }
        }        

        void mlirGen(TypeScriptParserANTLR::CallExpressionContext *callExpression)
        {
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
        }         

        void mlirGen(TypeScriptParserANTLR::MemberExpressionContext *memberExpression)
        {
            // TODO: finish it
        }          

        void mlirGen(TypeScriptParserANTLR::ArgumentsContext *arguments)
        {
            // TODO: finish it
        }               

        void mlirGen(TypeScriptParserANTLR::LiteralContext *literal)
        {
            if (auto *nullLiteral = literal->nullLiteral())
            {
                mlirGen(nullLiteral);
            }
            else if (auto *booleanLiteral = literal->booleanLiteral())
            {
                mlirGen(booleanLiteral);
            }     
            else if (auto *numericLiteral = literal->numericLiteral())
            {
                mlirGen(numericLiteral);
            }       
            else 
            {
                mlirGenStringLiteral(literal->StringLiteral());
            }
        }

        void mlirGen(TypeScriptParserANTLR::NullLiteralContext *nullLiteral)
        {
        }

        void mlirGen(TypeScriptParserANTLR::BooleanLiteralContext *booleanLiteral)
        {
        }

        void mlirGen(TypeScriptParserANTLR::NumericLiteralContext *numericLiteral)
        {
            if (auto *decimalLiteral = numericLiteral->DecimalLiteral())
            {
                mlirGenDecimalLiteral(decimalLiteral);
            }   
            else if (auto *decimalIntegerLiteral = numericLiteral->DecimalIntegerLiteral())
            {
                mlirGenDecimalIntegerLiteral(decimalIntegerLiteral);
            }
            else if (auto *decimalBigIntegerLiteral = numericLiteral->DecimalBigIntegerLiteral())
            {
                mlirGenDecimalBigIntegerLiteral(decimalBigIntegerLiteral);
            }
            else if (auto *binaryBigIntegerLiteral = numericLiteral->BinaryBigIntegerLiteral())
            {
                mlirGenBinaryBigIntegerLiteral(binaryBigIntegerLiteral);
            }
            else if (auto *octalBigIntegerLiteral = numericLiteral->OctalBigIntegerLiteral())
            {
                mlirGenOctalBigIntegerLiteral(octalBigIntegerLiteral);
            }
            else if (auto *hexBigIntegerLiteral = numericLiteral->HexBigIntegerLiteral())
            {
                mlirGenHexBigIntegerLiteral(hexBigIntegerLiteral);
            }
        }

        void mlirGen(TypeScriptParserANTLR::IdentifierReferenceContext *identifierReference)
        {
            mlirGenIdentifierName(identifierReference->IdentifierName());
        }

        void mlirGenIdentifierName(antlr4::tree::TerminalNode *identifierName)
        {
            // TODO:
        }        

        void mlirGenStringLiteral(antlr4::tree::TerminalNode *stringLiteral)
        {
            // TODO:
        }   

        void mlirGenDecimalLiteral(antlr4::tree::TerminalNode *decimalLiteral)
        {
            // TODO:
        }  

        void mlirGenDecimalIntegerLiteral(antlr4::tree::TerminalNode *decimalIntegerLiteral)
        {
            // TODO:
        }  

        void mlirGenDecimalBigIntegerLiteral(antlr4::tree::TerminalNode *decimalBigIntegerLiteraligIntegerLiteral)
        {
            // TODO:
        }  

        void mlirGenBinaryBigIntegerLiteral(antlr4::tree::TerminalNode *binaryBigIntegerLiteral)
        {
            // TODO:
        }  

        void mlirGenOctalBigIntegerLiteral(antlr4::tree::TerminalNode *octalBigIntegerLiteral)
        {
            // TODO:
        }         

        void mlirGenHexBigIntegerLiteral(antlr4::tree::TerminalNode *hexBigIntegerLiteral)
        {
            // TODO:
        }                                  

    private:
        /// A "module" matches a TypeScript source file: containing a list of functions.
        mlir::ModuleOp theModule;

        /// The builder is a helper class to create IR inside a function. The builder
        /// is stateful, in particular it keeps an "insertion point": this is where
        /// the next operations will be introduced.
        mlir::OpBuilder builder;

        /// Helper conversion for a TypeScript AST location to an MLIR location.
        /*
        mlir::Location loc(Location loc)
        {
            return builder.getFileLineColLoc(builder.getIdentifier(*loc.file), loc.line, loc.col);
        }
        */
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
