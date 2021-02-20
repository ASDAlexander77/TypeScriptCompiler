#include "TypeScript/TypeScriptDialect.h"
#include "TypeScript/TypeScriptOps.h"
#include "TypeScript/Passes.h"
#include "TypeScript/EnumsAST.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/Sequence.h"

using namespace mlir::typescript;
using namespace typescript;

//===----------------------------------------------------------------------===//
// TypeScriptToAffine RewritePatterns
//===----------------------------------------------------------------------===//

struct CallOpLowering : public mlir::OpRewritePattern<CallOp>
{
    using mlir::OpRewritePattern<CallOp>::OpRewritePattern;

    mlir::LogicalResult matchAndRewrite(CallOp op, mlir::PatternRewriter &rewriter) const final
    {
        // just replace
        rewriter.replaceOpWithNewOp<CallOp>(op, op.getCallee(), op.getResultTypes(), op.getArgOperands());
        return mlir::success();
    }
};

struct ParamOpLowering : public mlir::OpRewritePattern<ParamOp>
{
    using mlir::OpRewritePattern<ParamOp>::OpRewritePattern;

    mlir::LogicalResult matchAndRewrite(ParamOp paramOp, mlir::PatternRewriter &rewriter) const final
    {
        mlir::Value allocated = rewriter.create<mlir::AllocaOp>(paramOp.getLoc(), paramOp.getType().cast<mlir::MemRefType>());
        rewriter.create<mlir::StoreOp>(paramOp.getLoc(), paramOp.argValue(), allocated);
        rewriter.replaceOp(paramOp, allocated);
        return mlir::success();
    }
};

struct ParamOptionalOpLowering : public mlir::OpRewritePattern<ParamOptionalOp>
{
    using mlir::OpRewritePattern<ParamOptionalOp>::OpRewritePattern;

    mlir::LogicalResult matchAndRewrite(ParamOptionalOp paramOp, mlir::PatternRewriter &rewriter) const final
    {
        auto location = paramOp.getLoc();

        mlir::Value allocated = rewriter.create<mlir::AllocaOp>(location, paramOp.getType().cast<mlir::MemRefType>());

        // scf.if
        auto index = paramOp.paramIndex();
        auto indexConstant = rewriter.create<mlir::ConstantOp>(location, rewriter.getI32IntegerAttr(index.getValue()));
        auto condValue = rewriter.create<mlir::CmpIOp>(location, mlir::CmpIPredicate::ult, paramOp.params_count(), indexConstant);
        auto ifOp = rewriter.create<mlir::scf::IfOp>(location, paramOp.argValue().getType(), condValue, true);

        auto sp = rewriter.saveInsertionPoint();

        // then block
        auto &thenRegion = ifOp.thenRegion();

        rewriter.setInsertionPointToEnd(&thenRegion.back());

        rewriter.inlineRegionBefore(paramOp.defaultValueRegion(), &ifOp.thenRegion().back());
        rewriter.eraseBlock(&ifOp.thenRegion().back());

        // else block
        auto &elseRegion = ifOp.elseRegion();

        rewriter.setInsertionPointToEnd(&elseRegion.back());

        rewriter.create<mlir::scf::YieldOp>(location, paramOp.argValue());

        rewriter.restoreInsertionPoint(sp);

        // save op
        rewriter.create<mlir::StoreOp>(location, ifOp.results().front(), allocated);
        rewriter.replaceOp(paramOp, allocated);
        return mlir::success();
    }
};

struct ParamDefaultValueOpLowering : public mlir::OpRewritePattern<ParamDefaultValueOp>
{
    using mlir::OpRewritePattern<ParamDefaultValueOp>::OpRewritePattern;

    mlir::LogicalResult matchAndRewrite(ParamDefaultValueOp op, mlir::PatternRewriter &rewriter) const final
    {
        rewriter.replaceOpWithNewOp<mlir::scf::YieldOp>(op, op.results());
        return mlir::success();
    }
};


struct VariableOpLowering : public mlir::OpRewritePattern<VariableOp>
{
    using mlir::OpRewritePattern<VariableOp>::OpRewritePattern;

    mlir::LogicalResult matchAndRewrite(VariableOp varOp, mlir::PatternRewriter &rewriter) const final
    {
        auto init = varOp.initializer();
        if (!init)
        {
            rewriter.replaceOpWithNewOp<mlir::AllocaOp>(varOp, varOp.getType().cast<mlir::MemRefType>());
            return mlir::success();
        }

        mlir::Value allocated = rewriter.create<mlir::AllocaOp>(varOp.getLoc(), varOp.getType().cast<mlir::MemRefType>());
        rewriter.create<mlir::LLVM::StoreOp>(varOp.getLoc(), init, allocated);
        rewriter.replaceOp(varOp, allocated);
        return mlir::success();
    }
};

struct ArithmeticBinaryOpLowering : public mlir::OpRewritePattern<ArithmeticBinaryOp>
{
    using mlir::OpRewritePattern<ArithmeticBinaryOp>::OpRewritePattern;

    mlir::LogicalResult matchAndRewrite(ArithmeticBinaryOp arithmeticBinaryOp, mlir::PatternRewriter &rewriter) const final
    {
        llvm_unreachable("not implemented");
        return mlir::success();
    }
};

struct LogicalBinaryOpLowering : public mlir::OpRewritePattern<LogicalBinaryOp>
{
    using mlir::OpRewritePattern<LogicalBinaryOp>::OpRewritePattern;

    mlir::LogicalResult matchAndRewrite(LogicalBinaryOp logicalBinaryOp, mlir::PatternRewriter &rewriter) const final
    {
        switch ((SyntaxKind)logicalBinaryOp.opCode())
        {
            case SyntaxKind::EqualsEqualsToken:
                rewriter.replaceOpWithNewOp<mlir::CmpIOp>(logicalBinaryOp, mlir::CmpIPredicate::eq, logicalBinaryOp.getOperand(0), logicalBinaryOp.getOperand(1));
                return mlir::success(); 
            default:
                llvm_unreachable("not implemented");
        }
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
    struct TypeScriptToAffineLoweringPass : public mlir::PassWrapper<TypeScriptToAffineLoweringPass, mlir::FunctionPass>
    {
        void getDependentDialects(mlir::DialectRegistry &registry) const override
        {
            registry.insert<mlir::AffineDialect, mlir::StandardOpsDialect, mlir::scf::SCFDialect>();
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
    mlir::ConversionTarget target(getContext());

    // We define the specific operations, or dialects, that are legal targets for
    // this lowering. In our case, we are lowering to a combination of the
    // `Affine` and `Standard` dialects.
    target.addLegalDialect<mlir::AffineDialect, mlir::StandardOpsDialect, mlir::scf::SCFDialect>();

    // We also define the TypeScript dialect as Illegal so that the conversion will fail
    // if any of these operations are *not* converted. Given that we actually want
    // a partial lowering, we explicitly mark the TypeScript operations that don't want
    // to lower, `typescript.print`, as `legal`.
    target.addIllegalDialect<TypeScriptDialect>();
    target.addLegalOp<
        PrintOp,
        AssertOp,
        UndefOp>();

    // Now that the conversion target has been defined, we just need to provide
    // the set of patterns that will lower the TypeScript operations.
    mlir::OwningRewritePatternList patterns;
    patterns.insert<
        CallOpLowering,
        ParamOpLowering,
        ParamOptionalOpLowering,
        ParamDefaultValueOpLowering,
        VariableOpLowering,
        ArithmeticBinaryOpLowering,
        LogicalBinaryOpLowering
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
std::unique_ptr<mlir::Pass> mlir::typescript::createLowerToAffinePass()
{
    return std::make_unique<TypeScriptToAffineLoweringPass>();
}
