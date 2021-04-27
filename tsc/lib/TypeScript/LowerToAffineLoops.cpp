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

namespace
{

    struct TSContext
    {
        // name, break, continue
        DenseMap<Operation *, std::tuple<StringRef, mlir::Block *>> jumps;
    };

    class TSTypeConverter : public TypeConverter {
    public:
        explicit TSTypeConverter() {}
    };

    template <typename OpTy>
    class TsPattern : public OpConversionPattern<OpTy> 
    {
    public:
        TsPattern<OpTy>(MLIRContext *context, TSTypeConverter &converter, TSContext *tsContext, PatternBenefit benefit = 1) 
            : OpConversionPattern<OpTy>::OpConversionPattern(context, benefit), tsContext(tsContext), typeConverter(converter) {}

    protected:
        TSContext *tsContext;
        TSTypeConverter &typeConverter;
    };

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

    struct PrefixUnaryOpLowering : public OpRewritePattern<mlir_ts::PrefixUnaryOp>
    {
        using OpRewritePattern<mlir_ts::PrefixUnaryOp>::OpRewritePattern;

        LogicalResult matchAndRewrite(mlir_ts::PrefixUnaryOp op, PatternRewriter &rewriter) const final
        {
            CodeLogicHelper clh(op, rewriter);
            auto cst1 = rewriter.create<mlir_ts::ConstantOp>(op->getLoc(), rewriter.getI32IntegerAttr(1));

            SyntaxKind opCode;
            switch (op.opCode())
            {
                case SyntaxKind::PlusPlusToken:
                    opCode = SyntaxKind::PlusToken;
                    break;
                case SyntaxKind::MinusMinusToken:
                    opCode = SyntaxKind::MinusToken;
                    break;
            }

            rewriter.replaceOpWithNewOp<mlir_ts::ArithmeticBinaryOp>(op, op.getType(), rewriter.getI32IntegerAttr(static_cast<int32_t>(opCode)), op.operand1(), cst1);
            clh.saveResult(op, op->getResult(0));

            return success();        
        }
    };  

    struct PostfixUnaryOpLowering : public OpRewritePattern<mlir_ts::PostfixUnaryOp>
    {
        using OpRewritePattern<mlir_ts::PostfixUnaryOp>::OpRewritePattern;

        LogicalResult matchAndRewrite(mlir_ts::PostfixUnaryOp op, PatternRewriter &rewriter) const final
        {
            CodeLogicHelper clh(op, rewriter);
            auto cst1 = rewriter.create<mlir_ts::ConstantOp>(op->getLoc(), rewriter.getI32IntegerAttr(1));

            SyntaxKind opCode;
            switch (op.opCode())
            {
                case SyntaxKind::PlusPlusToken:
                    opCode = SyntaxKind::PlusToken;
                    break;
                case SyntaxKind::MinusMinusToken:
                    opCode = SyntaxKind::MinusToken;
                    break;
            }

            auto result = rewriter.create<mlir_ts::ArithmeticBinaryOp>(op->getLoc(), op.getType(), rewriter.getI32IntegerAttr(static_cast<int32_t>(opCode)), op.operand1(), cst1);
            clh.saveResult(op, result);

            rewriter.replaceOp(op, op.operand1());
            return success();  
        }
    };  

    struct ArithmeticBinaryOpLowering : public OpRewritePattern<mlir_ts::ArithmeticBinaryOp>
    {
        using OpRewritePattern<mlir_ts::ArithmeticBinaryOp>::OpRewritePattern;

        LogicalResult matchAndRewrite(mlir_ts::ArithmeticBinaryOp arithmeticBinaryOp, PatternRewriter &rewriter) const final
        {
            auto opCode = (SyntaxKind)arithmeticBinaryOp.opCode();
            switch (opCode)
            {
            case SyntaxKind::PlusToken:
                BinOp<mlir_ts::ArithmeticBinaryOp, AddIOp, AddFOp>(arithmeticBinaryOp, rewriter);
                return success();

            case SyntaxKind::MinusToken:
                BinOp<mlir_ts::ArithmeticBinaryOp, SubIOp, SubFOp>(arithmeticBinaryOp, rewriter);
                return success();

            case SyntaxKind::AsteriskToken:
                BinOp<mlir_ts::ArithmeticBinaryOp, MulIOp, MulFOp>(arithmeticBinaryOp, rewriter);
                return success();

            case SyntaxKind::SlashToken:
                BinOp<mlir_ts::ArithmeticBinaryOp, DivFOp, DivFOp>(arithmeticBinaryOp, rewriter);
                return success();

            case SyntaxKind::GreaterThanGreaterThanToken:
                BinOp<mlir_ts::ArithmeticBinaryOp, SignedShiftRightOp, SignedShiftRightOp>(arithmeticBinaryOp, rewriter);
                return success();

            case SyntaxKind::GreaterThanGreaterThanGreaterThanToken:
                BinOp<mlir_ts::ArithmeticBinaryOp, UnsignedShiftRightOp, UnsignedShiftRightOp>(arithmeticBinaryOp, rewriter);
                return success();

            case SyntaxKind::LessThanLessThanToken:
                BinOp<mlir_ts::ArithmeticBinaryOp, ShiftLeftOp, ShiftLeftOp>(arithmeticBinaryOp, rewriter);
                return success();                

            case SyntaxKind::AmpersandToken:
                BinOp<mlir_ts::ArithmeticBinaryOp, AndOp, AndOp>(arithmeticBinaryOp, rewriter);
                return success();                    

            case SyntaxKind::BarToken:
                BinOp<mlir_ts::ArithmeticBinaryOp, OrOp, OrOp>(arithmeticBinaryOp, rewriter);
                return success();                    

            case SyntaxKind::CaretToken:
                BinOp<mlir_ts::ArithmeticBinaryOp, XOrOp, XOrOp>(arithmeticBinaryOp, rewriter);
                return success();                    

            case SyntaxKind::PercentToken:
                BinOp<mlir_ts::ArithmeticBinaryOp, RemFOp, RemFOp>(arithmeticBinaryOp, rewriter);
                return success();                    

            case SyntaxKind::AsteriskAsteriskToken:
                BinOp<mlir_ts::ArithmeticBinaryOp, PowFOp, PowFOp>(arithmeticBinaryOp, rewriter);
                return success();                    

            default:
                llvm_unreachable("not implemented");
            }
        }
    };

    struct IfOpLowering : public OpRewritePattern<mlir_ts::IfOp>
    {
        using OpRewritePattern<mlir_ts::IfOp>::OpRewritePattern;

        LogicalResult matchAndRewrite(mlir_ts::IfOp ifOp, PatternRewriter &rewriter) const final
        {
            auto loc = ifOp.getLoc();

            // Start by splitting the block containing the 'scf.if' into two parts.
            // The part before will contain the condition, the part after will be the
            // continuation point.
            auto *condBlock = rewriter.getInsertionBlock();
            auto opPosition = rewriter.getInsertionPoint();
            auto *remainingOpsBlock = rewriter.splitBlock(condBlock, opPosition);
            Block *continueBlock;
            if (ifOp.getNumResults() == 0) {
                continueBlock = remainingOpsBlock;
            } else {
                continueBlock =
                    rewriter.createBlock(remainingOpsBlock, ifOp.getResultTypes());
                rewriter.create<BranchOp>(loc, remainingOpsBlock);
            }

            // Move blocks from the "then" region to the region containing 'scf.if',
            // place it before the continuation block, and branch to it.
            auto &thenRegion = ifOp.thenRegion();
            auto *thenBlock = &thenRegion.front();
            Operation *thenTerminator = thenRegion.back().getTerminator();
            ValueRange thenTerminatorOperands = thenTerminator->getOperands();
            rewriter.setInsertionPointToEnd(&thenRegion.back());
            rewriter.create<BranchOp>(loc, continueBlock, thenTerminatorOperands);
            rewriter.eraseOp(thenTerminator);
            rewriter.inlineRegionBefore(thenRegion, continueBlock);

            // Move blocks from the "else" region (if present) to the region containing
            // 'scf.if', place it before the continuation block and branch to it.  It
            // will be placed after the "then" regions.
            auto *elseBlock = continueBlock;
            auto &elseRegion = ifOp.elseRegion();
            if (!elseRegion.empty()) {
                elseBlock = &elseRegion.front();
                Operation *elseTerminator = elseRegion.back().getTerminator();
                ValueRange elseTerminatorOperands = elseTerminator->getOperands();
                rewriter.setInsertionPointToEnd(&elseRegion.back());
                rewriter.create<BranchOp>(loc, continueBlock, elseTerminatorOperands);
                rewriter.eraseOp(elseTerminator);
                rewriter.inlineRegionBefore(elseRegion, continueBlock);
            }

            rewriter.setInsertionPointToEnd(condBlock);
            auto castToI1 = rewriter.create<mlir_ts::CastOp>(loc, rewriter.getI1Type(), ifOp.condition());
            rewriter.create<CondBranchOp>(loc, castToI1, thenBlock,
                                            /*trueArgs=*/ArrayRef<Value>(), elseBlock,
                                            /*falseArgs=*/ArrayRef<Value>());

            // Ok, we're done!
            rewriter.replaceOp(ifOp, continueBlock->getArguments());
            return success();
        }
    };

    struct WhileOpLowering : public TsPattern<mlir_ts::WhileOp>
    {
        using TsPattern<mlir_ts::WhileOp>::TsPattern;

        LogicalResult matchAndRewrite(mlir_ts::WhileOp whileOp, ArrayRef<Value> operands, ConversionPatternRewriter &rewriter) const final
        {
            OpBuilder::InsertionGuard guard(rewriter);
            Location loc = whileOp.getLoc();

            // Split the current block before the WhileOp to create the inlining point.
            auto *currentBlock = rewriter.getInsertionBlock();
            auto *continuation = rewriter.splitBlock(currentBlock, rewriter.getInsertionPoint());

            // Inline both regions.
            auto *after = &whileOp.after().front();
            auto *afterLast = &whileOp.after().back();
            auto *before = &whileOp.before().front();
            auto *beforeLast = &whileOp.before().back();

            whileOp.after().walk([&](Operation* op) {
                if (op->getName().getStringRef() == "ts.break") {
                    tsContext->jumps[op] = std::make_tuple(StringRef(""), continuation);
                }
            });

            rewriter.inlineRegionBefore(whileOp.after(), continuation);
            rewriter.inlineRegionBefore(whileOp.before(), after);

            // Branch to the "before" region.
            rewriter.setInsertionPointToEnd(currentBlock);
            rewriter.create<BranchOp>(loc, before, whileOp.inits());

            // Replace terminators with branches. Assuming bodies are SESE, which holds
            // given only the patterns from this file, we only need to look at the last
            // block. This should be reconsidered if we allow break/continue.
            rewriter.setInsertionPointToEnd(beforeLast);
            auto condOp = cast<mlir_ts::ConditionOp>(beforeLast->getTerminator());
            auto castToI1 = rewriter.create<mlir_ts::CastOp>(loc, rewriter.getI1Type(), condOp.condition());
            rewriter.replaceOpWithNewOp<CondBranchOp>(condOp, castToI1, after, condOp.args(), continuation, ValueRange());

            rewriter.setInsertionPointToEnd(afterLast);

            auto yieldOp = cast<mlir_ts::YieldOp>(afterLast->getTerminator());
            rewriter.replaceOpWithNewOp<BranchOp>(yieldOp, before, yieldOp.results());

            // Replace the op with values "yielded" from the "before" region, which are
            // visible by dominance.
            rewriter.replaceOp(whileOp, condOp.args());

            return success();  
        }
    };  

    /// Optimized version of the above for the case of the "after" region merely
    /// forwarding its arguments back to the "before" region (i.e., a "do-while"
    /// loop). This avoid inlining the "after" region completely and branches back
    /// to the "before" entry instead.
    struct DoWhileOpLowering : public TsPattern<mlir_ts::WhileOp>
    {
        using TsPattern<mlir_ts::WhileOp>::TsPattern;

        LogicalResult matchAndRewrite(mlir_ts::WhileOp whileOp, ArrayRef<Value> operands, ConversionPatternRewriter &rewriter) const final
        {
            Location loc = whileOp.getLoc();
            
            if (!llvm::hasSingleElement(whileOp.after()))
                return rewriter.notifyMatchFailure(whileOp,
                                                "do-while simplification applicable to "
                                                "single-block 'after' region only");

            Block &afterBlock = whileOp.after().front();
            if (!llvm::hasSingleElement(afterBlock))
                return rewriter.notifyMatchFailure(whileOp,
                                                "do-while simplification applicable "
                                                "only if 'after' region has no payload");

            auto yield = dyn_cast<mlir_ts::YieldOp>(&afterBlock.front());
            if (!yield || yield.results() != afterBlock.getArguments())
                return rewriter.notifyMatchFailure(whileOp,
                                                "do-while simplification applicable "
                                                "only to forwarding 'after' regions");

            // Split the current block before the WhileOp to create the inlining point.
            OpBuilder::InsertionGuard guard(rewriter);
            Block *currentBlock = rewriter.getInsertionBlock();
            Block *continuation =
                rewriter.splitBlock(currentBlock, rewriter.getInsertionPoint());

            // Only the "before" region should be inlined.
            Block *before = &whileOp.before().front();
            Block *beforeLast = &whileOp.before().back();
            rewriter.inlineRegionBefore(whileOp.before(), continuation);

            // Branch to the "before" region.
            rewriter.setInsertionPointToEnd(currentBlock);
            rewriter.create<BranchOp>(whileOp.getLoc(), before, whileOp.inits());

            // Loop around the "before" region based on condition.
            rewriter.setInsertionPointToEnd(beforeLast);
            auto condOp = cast<mlir_ts::ConditionOp>(beforeLast->getTerminator());
            auto castToI1 = rewriter.create<mlir_ts::CastOp>(loc, rewriter.getI1Type(), condOp.condition());
            rewriter.replaceOpWithNewOp<CondBranchOp>(condOp, castToI1, before,
                                                    condOp.args(), continuation,
                                                    ValueRange());

            // Replace the op with values "yielded" from the "before" region, which are
            // visible by dominance.
            rewriter.replaceOp(whileOp, condOp.args());

            return success();
        }
    };

    struct BreakOpLowering : public TsPattern<mlir_ts::BreakOp>
    {
        using TsPattern<mlir_ts::BreakOp>::TsPattern;

        LogicalResult matchAndRewrite(mlir_ts::BreakOp breakOp, ArrayRef<Value> operands, ConversionPatternRewriter &rewriter) const final
        {
            auto jump = tsContext->jumps[breakOp];
            rewriter.replaceOpWithNewOp<BranchOp>(breakOp, std::get<1>(jump)/*break=continuation*/);
            return success();
        }
    };

} // end anonymous namespace

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
    
        TSTypeConverter typeConverter;
        TSContext tsContext;
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
        mlir_ts::ArithmeticUnaryOp,
        mlir_ts::AssertOp,
        mlir_ts::CallOp,
        mlir_ts::CastOp,
        mlir_ts::ConstantOp,
        mlir_ts::EntryOp,
        mlir_ts::ExitOp,
        mlir_ts::FuncOp,
        mlir_ts::NullOp,
        mlir_ts::ParseFloatOp,
        mlir_ts::ParseIntOp,
        mlir_ts::PrintOp,
        mlir_ts::ReturnOp,
        mlir_ts::ReturnValOp,
        mlir_ts::StoreOp,
        mlir_ts::StoreElementOp,
        mlir_ts::LoadOp,
        mlir_ts::LoadElementOp,
        mlir_ts::LogicalBinaryOp,
        mlir_ts::UndefOp,
        mlir_ts::VariableOp,
        mlir_ts::YieldOp
    >();

    // Now that the conversion target has been defined, we just need to provide
    // the set of patterns that will lower the TypeScript operations.
    OwningRewritePatternList patterns;
    patterns.insert<
        ArithmeticBinaryOpLowering,
        ParamOpLowering,
        ParamOptionalOpLowering,
        ParamDefaultValueOpLowering,
        PrefixUnaryOpLowering,
        PostfixUnaryOpLowering,
        IfOpLowering
    >(&getContext());

    patterns.insert<
        WhileOpLowering,
        BreakOpLowering
    >(&getContext(), typeConverter, &tsContext);    

    patterns.insert<
        DoWhileOpLowering
    >(&getContext(), typeConverter, &tsContext, /*benefit=*/2);

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
