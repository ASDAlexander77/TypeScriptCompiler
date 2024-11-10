#include "TypeScript/Passes.h"
#include "TypeScript/TypeScriptDialect.h"
#include "TypeScript/TypeScriptOps.h"

#include "mlir/Pass/Pass.h"
#include "llvm/Support/Debug.h"

using namespace mlir;
using namespace ::typescript;
namespace mlir_ts = mlir::typescript;

class ModulePass : public OperationPass<mlir::ModuleOp>
{
  public:
    using OperationPass<mlir::ModuleOp>::OperationPass;

    /// The polymorphic API that runs the pass over the currently held function.
    virtual void runOnModule() = 0;

    /// The polymorphic API that runs the pass over the currently held operation.
    void runOnOperation() final
    {
        runOnModule();
    }

    /// Return the current function being transformed.
    mlir::ModuleOp getModule()
    {
        return this->getOperation();
    }
};
