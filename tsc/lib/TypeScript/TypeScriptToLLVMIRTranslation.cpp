#define DEBUG_TYPE "llvm"

#include "TypeScript/TypeScriptToLLVMIRTranslation.h"
#include "TypeScript/Defines.h"
#include "TypeScript/TypeScriptDialect.h"
#include "TypeScript/TypeScriptOps.h"

#include "mlir/IR/Operation.h"
#include "mlir/Target/LLVMIR/ModuleTranslation.h"

#include "llvm/IR/IRBuilder.h"

using namespace mlir;
using namespace ::typescript;
namespace mlir_ts = mlir::typescript;

//===----------------------------------------------------------------------===//
// TypeScriptDialectLLVMIRTranslationInterface
//===----------------------------------------------------------------------===//

namespace
{

/// Implementation of the dialect interface that converts operations belonging
/// to the TypeScript dialect to LLVM IR.
class TypeScriptDialectLLVMIRTranslationInterface : public LLVMTranslationDialectInterface
{
  public:
    using LLVMTranslationDialectInterface::LLVMTranslationDialectInterface;

    /// Translates the given operation to LLVM IR using the provided IR builder
    /// and saving the state in `moduleTranslation`.
    LogicalResult convertOperation(Operation *op, llvm::IRBuilderBase &builder, LLVM::ModuleTranslation &moduleTranslation) const final
    {
        Operation &opInst = *op;
        //#include "....Conversions.inc"

        return failure();
    }

    /// Attaches module-level metadata for functions marked as kernels.
    LogicalResult amendOperation(Operation *op, NamedAttribute attribute, LLVM::ModuleTranslation &moduleTranslation) const final
    {
        // TODO:
        return success();
    }
};

} // end anonymous namespace

void mlir::typescript::registerTypeScriptDialectTranslation(DialectRegistry &registry)
{
    registry.insert<mlir::typescript::TypeScriptDialect>();
    registry.addDialectInterface<mlir::typescript::TypeScriptDialect, TypeScriptDialectLLVMIRTranslationInterface>();
}

void mlir::typescript::registerTypeScriptDialectTranslation(MLIRContext &context)
{
    DialectRegistry registry;
    registerTypeScriptDialectTranslation(registry);
    context.appendDialectRegistry(registry);
}
