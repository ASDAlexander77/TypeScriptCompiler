#define DEBUG_TYPE "pass"

#include "mlir/Pass/Pass.h"

#include "TypeScript/TypeScriptDialect.h"
#include "TypeScript/TypeScriptOps.h"
#include "TypeScript/TypeScriptFunctionPass.h"
#include "TypeScript/Passes.h"

#include "TypeScript/LowerToLLVMLogic.h"

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"

#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

namespace mlir_ts = mlir::typescript;

namespace
{

class RelocateConstantPass : public mlir::PassWrapper<RelocateConstantPass, TypeScriptFunctionPass>
{
  public:
    void runOnFunction() override
    {
        auto f = getFunction();

        SmallPtrSet<Operation *, 16> workSet;

        f.walk([&](mlir::Operation *op) {
            if (auto constantOp = dyn_cast_or_null<mlir_ts::ConstantOp>(op))
            {
                workSet.insert(constantOp);
            }
        });

        // find fist non-constant op
        auto firstNonConstOp = seekFirstNonConstantOp(f);
        if (firstNonConstOp)
        {
            ConversionPatternRewriter rewriter(f.getContext());
            rewriter.setInsertionPoint(firstNonConstOp);

            for (auto op : workSet)
            {
                auto constantOp = cast<mlir_ts::ConstantOp>(op);
                auto newOp = rewriter.create<mlir_ts::ConstantOp>(constantOp->getLoc(), constantOp.getType(), constantOp.value());
                constantOp->replaceAllUsesWith(newOp);
                rewriter.eraseOp(constantOp);
            }
        }
    }

    Operation *seekFirstNonConstantOp(mlir_ts::FuncOp funcOp)
    {
        auto found = false;
        Operation *foundOp;
        // find last string
        auto lastUse = [&](Operation *op) {
            if (found)
            {
                return;
            }

            auto constantOp = dyn_cast_or_null<mlir_ts::ConstantOp>(op);
            if (!constantOp)
            {
                auto constOp = dyn_cast_or_null<mlir::ConstantOp>(op);
                if (!constOp)
                {
                    found = true;
                    foundOp = op;
                }
            }
        };

        funcOp.walk(lastUse);

        return foundOp;
    }
};
} // end anonymous namespace

/// Create pass.
std::unique_ptr<mlir::Pass> mlir_ts::createRelocateConstantPass()
{
    return std::make_unique<RelocateConstantPass>();
}
