#include "mlir/Pass/Pass.h"

#include "TypeScript/TypeScriptDialect.h"
#include "TypeScript/TypeScriptOps.h"
#include "TypeScript/Passes.h"
#include "TypeScript/Defines.h"

#define DEBUG_TYPE "stack-promotion"

namespace mlir_ts = mlir::typescript;

namespace
{

// Stack promotion (docs/superpowers/specs/2026-10-18-stack-promotion-design.md): an allocation
// whose object provably never outlives its function, and is small enough, goes on the stack.
// Every other allocation stays on the heap, silently. Runs at the affine level, where control
// flow is an ordinary CFG and every allocation and ownership op is still present.
class StackPromotionPass : public mlir::PassWrapper<StackPromotionPass, mlir::OperationPass<mlir::ModuleOp>>
{
    CompileOptions compileOptions;
    mlir_ts::StackPromotionLimits limits;

  public:
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(StackPromotionPass)

    StackPromotionPass(CompileOptions &compileOptions, mlir_ts::StackPromotionLimits limits)
        : compileOptions(compileOptions), limits(limits)
    {
    }

    void runOnOperation() override
    {
        if (!limits.report)
        {
            return;
        }

        getOperation().walk([&](mlir_ts::FuncOp funcOp) {
            mlir::emitRemark(funcOp.getLoc()) << "stack-promotion: pass-ran";
        });
    }
};

} // namespace

std::unique_ptr<mlir::Pass> mlir_ts::createStackPromotionPass(CompileOptions &compileOptions,
                                                              mlir_ts::StackPromotionLimits limits)
{
    return std::make_unique<StackPromotionPass>(compileOptions, limits);
}
