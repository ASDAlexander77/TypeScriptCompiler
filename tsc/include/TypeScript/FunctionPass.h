#include "TypeScript/Passes.h"
#include "TypeScript/TypeScriptDialect.h"
#include "TypeScript/TypeScriptOps.h"

#include "mlir/Pass/Pass.h"
#include "llvm/Support/Debug.h"

using namespace mlir;
using namespace ::typescript;
namespace mlir_ts = mlir::typescript;

class FunctionPass : public OperationPass<mlir::func::FuncOp>
{
  public:
    using OperationPass<mlir::func::FuncOp>::OperationPass;

    /// The polymorphic API that runs the pass over the currently held function.
    virtual void runOnFunction() = 0;

    /// The polymorphic API that runs the pass over the currently held operation.
    void runOnOperation() final
    {
        if (!getFunction().isExternal())
            runOnFunction();
    }

    /// Return the current function being transformed.
    mlir::func::FuncOp getFunction()
    {
        return this->getOperation();
    }
};