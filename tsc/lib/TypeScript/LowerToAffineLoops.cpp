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

using namespace mlir;
using namespace ::typescript;
namespace ts = mlir::typescript;

//===----------------------------------------------------------------------===//
// TypeScriptToAffine RewritePatterns
//===----------------------------------------------------------------------===//

struct CallOpLowering : public OpRewritePattern<ts::CallOp>
{
    using OpRewritePattern<ts::CallOp>::OpRewritePattern;

    LogicalResult matchAndRewrite(ts::CallOp op, PatternRewriter &rewriter) const final
    {
        // just replace
        rewriter.replaceOpWithNewOp<CallOp>(
            op,
            op.getCallee(),
            op.getResultTypes(),
            op.getArgOperands());
        return success();
    }
};

struct ParamOpLowering : public OpRewritePattern<ts::ParamOp>
{
    using OpRewritePattern<ts::ParamOp>::OpRewritePattern;

    LogicalResult matchAndRewrite(ts::ParamOp paramOp, PatternRewriter &rewriter) const final
    {
        Value allocated = rewriter.create<AllocaOp>(paramOp.getLoc(), paramOp.getType().cast<MemRefType>());
        rewriter.create<StoreOp>(paramOp.getLoc(), paramOp.argValue(), allocated);
        rewriter.replaceOp(paramOp, allocated);
        return success();
    }
};

struct ParamOptionalOpLowering : public OpRewritePattern<ts::ParamOptionalOp>
{
    using OpRewritePattern<ts::ParamOptionalOp>::OpRewritePattern;

    LogicalResult matchAndRewrite(ts::ParamOptionalOp paramOp, PatternRewriter &rewriter) const final
    {
        auto location = paramOp.getLoc();

        Value allocated = rewriter.create<AllocaOp>(location, paramOp.getType().cast<MemRefType>());

        // scf.if
        auto index = paramOp.paramIndex();
        auto indexConstant = rewriter.create<ConstantOp>(location, rewriter.getI32IntegerAttr(index.getValue()));
        auto condValue = rewriter.create<CmpIOp>(location, CmpIPredicate::ult, paramOp.params_count(), indexConstant);
        auto ifOp = rewriter.create<scf::IfOp>(location, paramOp.argValue().getType(), condValue, true);

        auto sp = rewriter.saveInsertionPoint();

        // then block
        auto &thenRegion = ifOp.thenRegion();

        rewriter.setInsertionPointToStart(&thenRegion.back());

        rewriter.inlineRegionBefore(paramOp.defaultValueRegion(), &ifOp.thenRegion().back());
        rewriter.eraseBlock(&ifOp.thenRegion().back());

        // else block
        auto &elseRegion = ifOp.elseRegion();

        rewriter.setInsertionPointToStart(&elseRegion.back());

        rewriter.create<scf::YieldOp>(location, paramOp.argValue());

        rewriter.restoreInsertionPoint(sp);

        // save op
        rewriter.create<StoreOp>(location, ifOp.results().front(), allocated);
        rewriter.replaceOp(paramOp, allocated);
        return success();
    }
};

struct ParamDefaultValueOpLowering : public OpRewritePattern<ts::ParamDefaultValueOp>
{
    using OpRewritePattern<ts::ParamDefaultValueOp>::OpRewritePattern;

    LogicalResult matchAndRewrite(ts::ParamDefaultValueOp op, PatternRewriter &rewriter) const final
    {
        rewriter.replaceOpWithNewOp<scf::YieldOp>(op, op.results());
        return success();
    }
};

struct VariableOpLowering : public OpRewritePattern<ts::VariableOp>
{
    using OpRewritePattern<ts::VariableOp>::OpRewritePattern;

    LogicalResult matchAndRewrite(ts::VariableOp varOp, PatternRewriter &rewriter) const final
    {
        auto init = varOp.initializer();
        if (!init)
        {
            rewriter.replaceOpWithNewOp<AllocaOp>(varOp, varOp.getType().cast<MemRefType>());
            return success();
        }

        Value allocated = rewriter.create<AllocaOp>(varOp.getLoc(), varOp.getType().cast<MemRefType>());
        rewriter.create<LLVM::StoreOp>(varOp.getLoc(), init, allocated);
        rewriter.replaceOp(varOp, allocated);
        return success();
    }
};

struct CastOpLowering : public OpRewritePattern<ts::CastOp>
{
    using OpRewritePattern<ts::CastOp>::OpRewritePattern;

    LogicalResult matchAndRewrite(ts::CastOp op, PatternRewriter &rewriter) const final
    {
        auto in = op.in();
        auto res = op.res();
        auto op1 = in.getType();
        auto op2 = res.getType();

        if (op1.isInteger(32) && op2.isF32())
        {
            rewriter.replaceOpWithNewOp<SIToFPOp>(op, op2, in);
            return success();
        }

        if (op2.isF32() && op1.isInteger(32))
        {
            rewriter.replaceOpWithNewOp<FPToSIOp>(op, op2, in);
            return success();
        }

        llvm_unreachable("not implemented");
    }
};

template <typename BinOpTy, typename StdIOpTy, typename StdFOpTy>
void BinOp(BinOpTy &binOp, mlir::PatternRewriter &builder)
{
    auto leftType = binOp.getOperand(0).getType();
    if (leftType.isIntOrIndex())
    {
        builder.replaceOpWithNewOp<StdIOpTy>(binOp, binOp.getOperand(0), binOp.getOperand(1));
    }
    else if (!leftType.isIntOrIndex() && leftType.isIntOrIndexOrFloat())
    {
        builder.replaceOpWithNewOp<StdFOpTy>(binOp, binOp.getOperand(0), binOp.getOperand(1));
    }
    else
    {
        llvm_unreachable("not implemented");
    }
}

template <typename BinOpTy, typename StdIOpTy, typename V1, V1 v1, typename StdFOpTy, typename V2, V2 v2>
void LogicOp(BinOpTy &binOp, mlir::PatternRewriter &builder)
{
    auto leftType = binOp.getOperand(0).getType();
    if (leftType.isIntOrIndex())
    {
        builder.replaceOpWithNewOp<StdIOpTy>(binOp, v1, binOp.getOperand(0), binOp.getOperand(1));
    }
    else if (!leftType.isIntOrIndex() && leftType.isIntOrIndexOrFloat())
    {
        builder.replaceOpWithNewOp<StdFOpTy>(binOp, v2, binOp.getOperand(0), binOp.getOperand(1));
    }
    else
    {
        llvm_unreachable("not implemented");
    }
}

struct ArithmeticBinaryOpLowering : public OpRewritePattern<ts::ArithmeticBinaryOp>
{
    using OpRewritePattern<ts::ArithmeticBinaryOp>::OpRewritePattern;

    LogicalResult matchAndRewrite(ts::ArithmeticBinaryOp arithmeticBinaryOp, PatternRewriter &rewriter) const final
    {
        switch ((SyntaxKind)arithmeticBinaryOp.opCode())
        {
        case SyntaxKind::PlusToken:
            BinOp<ts::ArithmeticBinaryOp, AddIOp, AddFOp>(arithmeticBinaryOp, rewriter);
            return success();

        case SyntaxKind::MinusToken:
            BinOp<ts::ArithmeticBinaryOp, SubIOp, SubFOp>(arithmeticBinaryOp, rewriter);
            return success();
        default:
            llvm_unreachable("not implemented");
        }
    }
};

struct LogicalBinaryOpLowering : public OpRewritePattern<ts::LogicalBinaryOp>
{
    using OpRewritePattern<ts::LogicalBinaryOp>::OpRewritePattern;

    LogicalResult matchAndRewrite(ts::LogicalBinaryOp logicalBinaryOp, PatternRewriter &rewriter) const final
    {
        switch ((SyntaxKind)logicalBinaryOp.opCode())
        {
        case SyntaxKind::EqualsEqualsToken:
            LogicOp<ts::LogicalBinaryOp,
                    CmpIOp, CmpIPredicate, CmpIPredicate::eq,
                    CmpFOp, CmpFPredicate, CmpFPredicate::OEQ>(logicalBinaryOp, rewriter);
            return success();
        default:
            llvm_unreachable("not implemented");
        }
    }
};

struct EntryOpLowering : public OpRewritePattern<ts::EntryOp>
{
    using OpRewritePattern<ts::EntryOp>::OpRewritePattern;

    LogicalResult matchAndRewrite(ts::EntryOp op, PatternRewriter &rewriter) const final
    {
        auto *opBlock = rewriter.getInsertionBlock();

        auto *block = rewriter.createBlock(opBlock->getParent());

        rewriter.setInsertionPointToEnd(block);

        // body of new block
        rewriter.create<mlir::ReturnOp>(op.getLoc());

        rewriter.setInsertionPointToEnd(opBlock);

        rewriter.eraseOp(op);
        return success();
    }
};

struct ReturnOpLowering : public OpRewritePattern<ts::ReturnOp>
{
    using OpRewritePattern<ts::ReturnOp>::OpRewritePattern;

    LogicalResult matchAndRewrite(ts::ReturnOp op, PatternRewriter &rewriter) const final
    {
        /*
        auto funcOp = op.getOperation()->getParentOfType<FuncOp>();
        auto *region = funcOp.getCallableRegion();
        if (!region)
        {
            return failure();
        }

        auto result = std::find_if(region->begin(), region->end(), [&](auto &item) {
            if (item.empty())
            {
                return false;
            }

            auto *op = &item.back();
            //auto name = op->getName().getStringRef();
            auto isReturn = dyn_cast<ReturnOp>(op) != nullptr;
            return isReturn;
        });

        if (result == region->end())
        {
            // found block with return;
            return failure();
        }

        rewriter.create<mlir::BranchOp>(op.getLoc(), &*result);
        */

        auto *opBlock = rewriter.getInsertionBlock();
        auto opPosition = rewriter.getInsertionPoint();
        auto *continuationBlock = rewriter.splitBlock(opBlock, opPosition);

        rewriter.setInsertionPointToEnd(opBlock);
        rewriter.create<mlir::BranchOp>(op.getLoc(), continuationBlock);

        rewriter.eraseOp(op);
        return success();
    }
};

struct ExitOpLowering : public OpConversionPattern<ts::ExitOp>
{
    using OpConversionPattern<ts::ExitOp>::OpConversionPattern;

    LogicalResult matchAndRewrite(ts::ExitOp op, ArrayRef<Value> operands, ConversionPatternRewriter &rewriter) const final
    {
        auto funcOp = op.getOperation()->getParentOfType<FuncOp>();
        auto *region = funcOp.getCallableRegion();
        if (!region)
        {
            return failure();
        }

        auto result = std::find_if(region->begin(), region->end(), [&](auto &item) {
            if (item.empty())
            {
                return false;
            }

            auto *op = &item.back();
            //auto name = op->getName().getStringRef();
            auto isReturn = dyn_cast<ReturnOp>(op) != nullptr;
            return isReturn;
        });

        if (result == region->end())
        {
            // found block with return;
            return failure();
        }

        rewriter.create<mlir::BranchOp>(op.getLoc(), &*result);

        rewriter.eraseOp(op);
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
    target.addIllegalDialect<ts::TypeScriptDialect>();
    target.addLegalOp<
        ts::PrintOp,
        ts::AssertOp,
        ts::UndefOp,
        ts::ReturnOp>();

    // Now that the conversion target has been defined, we just need to provide
    // the set of patterns that will lower the TypeScript operations.
    OwningRewritePatternList patterns;
    patterns.insert<
        CallOpLowering,
        ParamOpLowering,
        ParamOptionalOpLowering,
        ParamDefaultValueOpLowering,
        VariableOpLowering,
        CastOpLowering,
        ArithmeticBinaryOpLowering,
        LogicalBinaryOpLowering,
        EntryOpLowering,
        ExitOpLowering>(&getContext());

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
std::unique_ptr<Pass> ts::createLowerToAffinePass()
{
    return std::make_unique<TypeScriptToAffineLoweringPass>();
}
