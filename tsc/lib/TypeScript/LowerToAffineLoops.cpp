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

#include "TypeScript/LowerToLLVMLogic.h"

#include "scanner_enums.h"

using namespace mlir;
using namespace ::typescript;
namespace mlir_ts = mlir::typescript;

//===----------------------------------------------------------------------===//
// TypeScriptToAffine RewritePatterns
//===----------------------------------------------------------------------===//

struct ParamOpLowering : public OpRewritePattern<mlir_ts::ParamOp>
{
    using OpRewritePattern<mlir_ts::ParamOp>::OpRewritePattern;

    LogicalResult matchAndRewrite(mlir_ts::ParamOp paramOp, PatternRewriter &rewriter) const final
    {
        rewriter.replaceOpWithNewOp<mlir_ts::VariableOp>(paramOp, paramOp.getType(), paramOp.argValue());
        return success();
    }
};

struct ParamOptionalOpLowering : public OpRewritePattern<mlir_ts::ParamOptionalOp>
{
    using OpRewritePattern<mlir_ts::ParamOptionalOp>::OpRewritePattern;

    LogicalResult matchAndRewrite(mlir_ts::ParamOptionalOp paramOp, PatternRewriter &rewriter) const final
    {
        TypeHelper th(rewriter); 

        auto location = paramOp.getLoc();

        if (paramOp.defaultValueRegion().empty())
        {
            rewriter.replaceOpWithNewOp<mlir_ts::VariableOp>(paramOp, paramOp.getType(), paramOp.argValue());
            return success();            
        }

        Value variable = rewriter.create<mlir_ts::VariableOp>(location, paramOp.getType(), mlir::Value());

        // scf.if
        auto index = paramOp.paramIndex();
        auto indexConstant = rewriter.create<mlir_ts::ConstantOp>(location, rewriter.getI32IntegerAttr(index.getValue()));
        // replace with ts op to avoid cast
        auto condValue = rewriter.create<CmpIOp>(location, CmpIPredicate::ult, paramOp.params_count(), indexConstant);
        auto compare = rewriter.create<mlir_ts::LogicalBinaryOp>(
            location,
            th.getBooleanType(),
            rewriter.getI32IntegerAttr((int)SyntaxKind::LessThanToken),
            paramOp.params_count(),
            indexConstant);
        auto ifOp = rewriter.create<mlir_ts::IfOp>(location, paramOp.argValue().getType(), compare, true);

        auto sp = rewriter.saveInsertionPoint();

        // then block
        auto &thenRegion = ifOp.thenRegion();

        rewriter.setInsertionPointToStart(&thenRegion.back());

        rewriter.inlineRegionBefore(paramOp.defaultValueRegion(), &ifOp.thenRegion().back());
        rewriter.eraseBlock(&ifOp.thenRegion().back());

        // else block
        auto &elseRegion = ifOp.elseRegion();

        rewriter.setInsertionPointToStart(&elseRegion.back());

        rewriter.create<mlir_ts::YieldOp>(location, paramOp.argValue());

        rewriter.restoreInsertionPoint(sp);

        // save op
        rewriter.create<mlir_ts::StoreOp>(location, ifOp.results().front(), variable);
        rewriter.replaceOp(paramOp, variable);
        return success();
    }
};

struct ParamDefaultValueOpLowering : public OpRewritePattern<mlir_ts::ParamDefaultValueOp>
{
    using OpRewritePattern<mlir_ts::ParamDefaultValueOp>::OpRewritePattern;

    LogicalResult matchAndRewrite(mlir_ts::ParamDefaultValueOp op, PatternRewriter &rewriter) const final
    {
        rewriter.replaceOpWithNewOp<mlir_ts::YieldOp>(op, op.results());
        return success();
    }
};

//===----------------------------------------------------------------------===//
// TypeScriptToAffineLoweringPass
//===----------------------------------------------------------------------===//

/// This is a partial lowering to affine loops of the typescript operations that are
/// computationally intensive (like add+mul for example...) while keeping the
/// rest of the code in the TypeScript dialect.

class TypeScriptFunctionPass : public OperationPass<mlir_ts::FuncOp>
{
public:
    using OperationPass<mlir_ts::FuncOp>::OperationPass;

    /// The polymorphic API that runs the pass over the currently held function.
    virtual void runOnFunction() = 0;

    /// The polymorphic API that runs the pass over the currently held operation.
    void runOnOperation() final
    {
        if (!getFunction().isExternal())
            runOnFunction();
    }

    /// Return the current function being transformed.
    mlir_ts::FuncOp getFunction() { return this->getOperation(); }
};

namespace
{
    struct TypeScriptToAffineLoweringPass : public PassWrapper<TypeScriptToAffineLoweringPass, TypeScriptFunctionPass>
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
    target.addLegalDialect<AffineDialect, StandardOpsDialect>();

    // We also define the TypeScript dialect as Illegal so that the conversion will fail
    // if any of these operations are *not* converted. Given that we actually want
    // a partial lowering, we explicitly mark the TypeScript operations that don't want
    // to lower, `typescript.print`, as `legal`.
    target.addIllegalDialect<mlir_ts::TypeScriptDialect>();
    target.addLegalOp<
        mlir_ts::AddressOfOp,
        mlir_ts::AddressOfConstStringOp,
        mlir_ts::AddressOfElementOp,
        mlir_ts::ArithmeticBinaryOp,
        mlir_ts::ArithmeticUnaryOp,
        mlir_ts::AssertOp,
        mlir_ts::CallOp,
        mlir_ts::CastOp,
        mlir_ts::ConstantOp,
        mlir_ts::EntryOp,
        mlir_ts::ExitOp,
        mlir_ts::FuncOp,
        mlir_ts::IfOp,
        mlir_ts::NullOp,
        mlir_ts::ParseFloatOp,
        mlir_ts::ParseIntOp,
        mlir_ts::PrintOp,
        mlir_ts::ReturnOp,
        mlir_ts::ReturnValOp,
        mlir_ts::StoreOp,
        mlir_ts::LoadOp,
        mlir_ts::LogicalBinaryOp,
        mlir_ts::UndefOp,
        mlir_ts::VariableOp,
        mlir_ts::YieldOp
    >();

    // Now that the conversion target has been defined, we just need to provide
    // the set of patterns that will lower the TypeScript operations.
    OwningRewritePatternList patterns;
    patterns.insert<
        ParamOpLowering,
        ParamOptionalOpLowering,
        ParamDefaultValueOpLowering
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
std::unique_ptr<Pass> mlir_ts::createLowerToAffinePass()
{
    return std::make_unique<TypeScriptToAffineLoweringPass>();
}
