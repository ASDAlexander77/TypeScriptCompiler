#include "TypeScript/TypeScriptDialectTranslation.h"
#include "TypeScript/Defines.h"
#include "TypeScript/TypeScriptDialect.h"
#include "TypeScript/TypeScriptOps.h"

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/Operation.h"
#include "mlir/Target/LLVMIR/ModuleTranslation.h"

#include "llvm/IR/IRBuilder.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "llvm"

using namespace mlir;
using namespace mlir::LLVM;
using namespace ::typescript;
namespace mlir_ts = mlir::typescript;

//===----------------------------------------------------------------------===//
// TypeScriptDialectLLVMIRTranslationInterface
//===----------------------------------------------------------------------===//

namespace
{

class ProcessNestAttribute
{
    Operation *op;
    LLVM::ModuleTranslation &moduleTranslation;
    LLVMFuncOp funcOp;
    llvm::Function *llvmFunc;

  public:
    ProcessNestAttribute(Operation *op, LLVM::ModuleTranslation &moduleTranslation) : op(op), moduleTranslation(moduleTranslation)
    {
        funcOp = dyn_cast_or_null<LLVMFuncOp>(op);
        llvmFunc = moduleTranslation.lookupFunction(funcOp.getName());
    }

    LogicalResult processNested()
    {
        unsigned int argIdx = 0;
        for (auto kvp : llvm::zip(funcOp.getArguments(), llvmFunc->args()))
        {
            llvm::Argument &llvmArg = std::get<1>(kvp);
            BlockArgument mlirArg = std::get<0>(kvp);

            if (auto attr = funcOp.getArgAttrOfType<UnitAttr>(argIdx, TS_NEST_ATTRIBUTE))
            {
                auto argTy = mlirArg.getType();
                llvmArg.addAttr(llvm::Attribute::AttrKind::Nest);
            }

            argIdx++;
        }

        return success();
    }

    LogicalResult processGc()
    {
        // llvmFunc->setGC(TYPESCRIPT_GC_NAME);
        llvmFunc->setGC("coreclr");
        // llvmFunc->setGC("erlang");
        // llvmFunc->setGC("ocaml");
        return success();
    }
};

/// Implementation of the dialect interface that converts operations belonging
/// to the TypeScript dialect to LLVM IR.
class TypeScriptDialectLLVMIRTranslationInterface : public LLVMTranslationDialectInterface
{
  public:
    using LLVMTranslationDialectInterface::LLVMTranslationDialectInterface;

    /// Attaches module-level metadata for functions marked as kernels.
    virtual LogicalResult 
    amendOperation(Operation *op, NamedAttribute attribute, LLVM::ModuleTranslation &moduleTranslation) const final
    {
        LLVM_DEBUG(llvm::dbgs() << "\n === amendOperation === \n");
        LLVM_DEBUG(llvm::dbgs() << "attribute: " << attribute.getName() << " val: " << attribute.getValue() << "\n");

        auto isNestAttr = attribute.getName() == TS_NEST_ATTRIBUTE;
        auto isGcAttr = dyn_cast<StringAttr>(attribute.getValue()).str() == TS_GC_ATTRIBUTE;

        // TODO:
        if (isNestAttr || isGcAttr)
        {
            ProcessNestAttribute pna(op, moduleTranslation);
            if (isNestAttr && mlir::failed(pna.processNested()))
            {
                return mlir::failure();
            }

            if (isGcAttr && mlir::failed(pna.processGc()))
            {
                return mlir::failure();
            }
        }

        return success();
    }
};

} // end anonymous namespace

void mlir::typescript::registerTypeScriptDialectTranslation(DialectRegistry &registry)
{
    registry.insert<mlir::typescript::TypeScriptDialect>();
    registry.addExtension(+[](MLIRContext *ctx, mlir::typescript::TypeScriptDialect *dialect) {
        dialect->addInterfaces<TypeScriptDialectLLVMIRTranslationInterface>();
    });
}

void mlir::typescript::registerTypeScriptDialectTranslation(MLIRContext &context)
{
    DialectRegistry registry;
    registerTypeScriptDialectTranslation(registry);
    context.appendDialectRegistry(registry);
}
