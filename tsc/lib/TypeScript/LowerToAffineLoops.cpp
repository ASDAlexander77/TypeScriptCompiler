#include "TypeScript/TypeScriptDialect.h"
#include "TypeScript/TypeScriptOps.h"
#include "TypeScript/Passes.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/SCF/SCF.h"
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
        // just replace
        rewriter.replaceOpWithNewOp<mlir::CallOp>(op, op.getCallee(), op.getResultTypes(), op.getArgOperands());
        return success();
    }
};

struct ParamOpLowering : public OpRewritePattern<typescript::ParamOp>
{
    using OpRewritePattern<typescript::ParamOp>::OpRewritePattern;

    LogicalResult matchAndRewrite(typescript::ParamOp varOp, PatternRewriter &rewriter) const final
    {
        mlir::Value allocated = rewriter.create<mlir::AllocaOp>(varOp.getLoc(), varOp.getType().cast<MemRefType>());
        rewriter.create<mlir::StoreOp>(varOp.getLoc(), varOp.argValue(), allocated);
        rewriter.replaceOp(varOp, allocated);
        return success();
    }
};

struct ParamOptionalOpLowering : public OpRewritePattern<typescript::ParamOptionalOp>
{
    using OpRewritePattern<typescript::ParamOptionalOp>::OpRewritePattern;

    LogicalResult matchAndRewrite(typescript::ParamOptionalOp varOp, PatternRewriter &rewriter) const final
    {
        auto location = varOp.getLoc();

        mlir::Value allocated = rewriter.create<mlir::AllocaOp>(location, varOp.getType().cast<MemRefType>());

        // scf.if
        auto index = varOp.paramIndex();
        auto indexConstant = rewriter.create<mlir::ConstantOp>(location, rewriter.getI32IntegerAttr(index.getValue()));
        auto condValue = rewriter.create<mlir::CmpIOp>(location, mlir::CmpIPredicate::ult, varOp.params_count(), indexConstant);
        auto ifOp = rewriter.create<mlir::scf::IfOp>(location, varOp.argValue().getType(), condValue, true);

        auto sp = rewriter.saveInsertionPoint();

        // then block
        auto &thenRegion = ifOp.thenRegion();

        rewriter.setInsertionPointToEnd(&thenRegion.back());

        rewriter.create<mlir::scf::YieldOp>(location, varOp.argValue());

        // else block
        auto &elseRegion = ifOp.elseRegion();

        rewriter.setInsertionPointToEnd(&elseRegion.back());

        rewriter.create<mlir::scf::YieldOp>(location, varOp.argValue());

        rewriter.restoreInsertionPoint(sp);

        // save op
        rewriter.create<mlir::StoreOp>(location, ifOp.results().front(), allocated);
        rewriter.replaceOp(varOp, allocated);
        return success();
    }
};

struct VariableOpLowering : public OpRewritePattern<typescript::VariableOp>
{
    using OpRewritePattern<typescript::VariableOp>::OpRewritePattern;

    LogicalResult matchAndRewrite(typescript::VariableOp varOp, PatternRewriter &rewriter) const final
    {
        auto init = varOp.initializer();
        if (!init)
        {
            rewriter.replaceOpWithNewOp<mlir::AllocaOp>(varOp, varOp.getType().cast<MemRefType>());
            return success();
        }

        mlir::Value allocated = rewriter.create<mlir::AllocaOp>(varOp.getLoc(), varOp.getType().cast<MemRefType>());
        rewriter.create<LLVM::StoreOp>(varOp.getLoc(), init, allocated);
        rewriter.replaceOp(varOp, allocated);
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
            registry.insert<AffineDialect, StandardOpsDialect, scf::SCFDialect>();
        }

        void runOnFunction() final;
    };
} // end anonymous namespace.

void TypeScriptToAffineLoweringPass::runOnFunction()
{
    auto function = getFunction();

    // We only lower the main function as we expect that all other functions have been inlined.
    if (function.getName() == "main")
    {
        // Verify that the given main has no inputs and results.
        if (function.getNumArguments() || function.getType().getNumResults())
        {
            function.emitError("expected 'main' to have 0 inputs and 0 results");
            return signalPassFailure();
        }
    }

    // The first thing to define is the conversion target. This will define the
    // final target for this lowering.
    ConversionTarget target(getContext());

    // We define the specific operations, or dialects, that are legal targets for
    // this lowering. In our case, we are lowering to a combination of the
    // `Affine` and `Standard` dialects.
    target.addLegalDialect<AffineDialect, StandardOpsDialect, scf::SCFDialect>();

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
        CallOpLowering,
        ParamOpLowering,
        ParamOptionalOpLowering,
        VariableOpLowering
    >(&getContext());

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
