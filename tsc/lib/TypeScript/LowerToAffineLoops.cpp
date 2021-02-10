#include "TypeScript/TypeScriptDialect.h"
#include "TypeScript/TypeScriptOps.h"
#include "TypeScript/Passes.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/Sequence.h"

using namespace mlir;

//===----------------------------------------------------------------------===//
// TypeScriptToAffine RewritePatterns
//===----------------------------------------------------------------------===//

struct CallOpLowering : public OpRewritePattern<typescript::CallOp>
{
    using OpRewritePattern<typescript::CallOp>::OpRewritePattern;

    LogicalResult matchAndRewrite(typescript::CallOp op, PatternRewriter &rewriter) const final
    {
        auto fn = cast<FuncOp>(SymbolTable::lookupNearestSymbolFrom(op, op.getCallee()));
        if (!fn)
        {
            return failure();
        }

        // getNumFuncArguments;
        auto opArgs = op.getArgOperands();
        auto funcArgsCount = fn.getType().getInputs().size();
        auto optionalFrom = funcArgsCount - opArgs.size();
        if (optionalFrom > 0)
        {
            SmallVector<Value, 0> newOpArgs(opArgs);
            // -1 to exclude count params
            for (auto i = (size_t)opArgs.size(); i < funcArgsCount - 1; i++)
            {
                newOpArgs.push_back(rewriter.create<typescript::UndefOp>(op.getLoc(), fn.getType().getInput(i)));
            }

            auto constNumOfParams = rewriter.create<mlir::ConstantOp>(op.getLoc(), rewriter.getI32Type(), rewriter.getI32IntegerAttr(opArgs.size()));
            // TODO: uncomment it when finish
            newOpArgs.push_back(constNumOfParams);

            rewriter.replaceOpWithNewOp<mlir::CallOp>(op, op.getCallee(), op.getResultTypes(), newOpArgs);
        }
        else
        {
            // just replace
            rewriter.replaceOpWithNewOp<mlir::CallOp>(op, op.getCallee(), op.getResultTypes(), op.getArgOperands());
        }

        return success();
    }
};

//===----------------------------------------------------------------------===//
// TypeScriptToAffineLoweringPass
//===----------------------------------------------------------------------===//

/// This is a partial lowering to affine loops of the typescript operations that are
/// computationally intensive (like add+mul for example...) while keeping the
/// rest of the code in the TypeScript dialect.
namespace
{
    struct TypeScriptToAffineLoweringPass : public PassWrapper<TypeScriptToAffineLoweringPass, FunctionPass>
    {
        void getDependentDialects(DialectRegistry &registry) const override
        {
            registry.insert<AffineDialect, StandardOpsDialect>();
        }

        void runOnFunction() final;
    };
} // end anonymous namespace.

void TypeScriptToAffineLoweringPass::runOnFunction()
{
    auto function = getFunction();

    // We only lower the main function as we expect that all other functions have been inlined.
    if (function.getName() != "main")
    {
        return;
    }

    // Verify that the given main has no inputs and results.
    if (function.getNumArguments() || function.getType().getNumResults())
    {
        function.emitError("expected 'main' to have 0 inputs and 0 results");
        return signalPassFailure();
    }

    // The first thing to define is the conversion target. This will define the
    // final target for this lowering.
    ConversionTarget target(getContext());

    // We define the specific operations, or dialects, that are legal targets for
    // this lowering. In our case, we are lowering to a combination of the
    // `Affine` and `Standard` dialects.
    target.addLegalDialect<AffineDialect, StandardOpsDialect>();

    // We also define the TypeScript dialect as Illegal so that the conversion will fail
    // if any of these operations are *not* converted. Given that we actually want
    // a partial lowering, we explicitly mark the TypeScript operations that don't want
    // to lower, `typescript.print`, as `legal`.
    target.addIllegalDialect<typescript::TypeScriptDialect>();
    target.addLegalOp<
        typescript::PrintOp,
        typescript::AssertOp,
        typescript::UndefOp>();

    // Now that the conversion target has been defined, we just need to provide
    // the set of patterns that will lower the TypeScript operations.
    OwningRewritePatternList patterns;
    patterns.insert<
        CallOpLowering>(&getContext());

    // With the target and rewrite patterns defined, we can now attempt the
    // conversion. The conversion will signal failure if any of our `illegal`
    // operations were not converted successfully.
    if (failed(applyPartialConversion(function, target, std::move(patterns))))
    {
        signalPassFailure();
    }
}

/// Create a pass for lowering operations in the `Affine` and `Std` dialects,
/// for a subset of the TypeScript IR.
std::unique_ptr<Pass> mlir::typescript::createLowerToAffinePass()
{
    return std::make_unique<TypeScriptToAffineLoweringPass>();
}
