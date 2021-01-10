#include "TypeScript/MLIRGen.h"
#include "TypeScript/AST.h"
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
    MLIRGenImpl(mlir::MLIRContext &context) : builder(&context) {}

    /// Public API: convert the AST for a TypeScript module (source file) to an MLIR
    /// Module operation.
    mlir::ModuleOp mlirGen(ModuleAST &moduleAST)
    {
      // We create an empty MLIR module and codegen functions one at a time and
      // add them to the module.
      theModule = mlir::ModuleOp::create(builder.getUnknownLoc());

      for (auto &record : moduleAST)
      {
        if (NumberExprAST *num = llvm::dyn_cast<NumberExprAST>(record.get()))
        {
            mlirGen(*num);
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

  private:
    /// A "module" matches a TypeScript source file: containing a list of functions.
    mlir::ModuleOp theModule;

    /// The builder is a helper class to create IR inside a function. The builder
    /// is stateful, in particular it keeps an "insertion point": this is where
    /// the next operations will be introduced.
    mlir::OpBuilder builder;

    /// Helper conversion for a TypeScript AST location to an MLIR location.
    mlir::Location loc(Location loc)
    {
      return builder.getFileLineColLoc(builder.getIdentifier(*loc.file), loc.line, loc.col);
    }

    mlir::Value mlirGen(NumberExprAST &num)
    {
      // todo:
      return mlir::Value(nullptr);
    }

    /// Dispatch codegen for the right expression subclass using RTTI.
    mlir::Value mlirGen(ExprAST &expr)
    {
      switch (expr.getKind())
      {
      case typescript::ExprAST::Expr_Num:
        return mlirGen(cast<NumberExprAST>(expr));
      default:
        emitError(loc(expr.loc()))
            << "MLIR codegen encountered an unhandled expr kind '"
            << Twine(expr.getKind()) << "'";
        return nullptr;
      }
    }
  };

} // namespace

namespace typescript
{

  // The public API for codegen.
  mlir::OwningModuleRef mlirGen(mlir::MLIRContext &context,
                                ModuleAST &moduleAST)
  {
    return MLIRGenImpl(context).mlirGen(moduleAST);
  }

} // namespace typescript
