#define DEBUG_TYPE "llvm"

#include "TypeScript/TypeScriptToLLVMIRTranslation.h"
#include "TypeScript/Defines.h"
#include "TypeScript/TypeScriptDialect.h"
#include "TypeScript/TypeScriptOps.h"

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/Operation.h"
#include "mlir/Target/LLVMIR/ModuleTranslation.h"

#include "llvm/IR/IRBuilder.h"

using namespace mlir;
using namespace mlir::LLVM;
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

    /// Attaches module-level metadata for functions marked as kernels.
    LogicalResult amendOperation(Operation *op, NamedAttribute attribute, LLVM::ModuleTranslation &moduleTranslation) const final
    {
        LLVM_DEBUG(llvm::dbgs() << "\n === amendOperation === \n");
        LLVM_DEBUG(llvm::dbgs() << "attribute: " << attribute.first << " val: " << attribute.second << "\n");
        // LLVM_DEBUG(op->dump());
        // TODO:
        if (attribute.first != "ts.nest")
        {
            return success();
        }

        auto func = dyn_cast_or_null<LLVMFuncOp>(op);
        llvm::Function *llvmFunc = moduleTranslation.lookupFunction(func.getName());

        unsigned int argIdx = 0;
        for (auto kvp : llvm::zip(func.getArguments(), llvmFunc->args()))
        {
            llvm::Argument &llvmArg = std::get<1>(kvp);
            BlockArgument mlirArg = std::get<0>(kvp);

            if (auto attr = func.getArgAttrOfType<UnitAttr>(argIdx, "ts.nest"))
            {
                auto argTy = mlirArg.getType();
                llvmArg.addAttr(llvm::Attribute::AttrKind::Nest);
            }

            argIdx++;
        }

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
