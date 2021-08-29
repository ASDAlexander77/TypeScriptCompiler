#define DEBUG_TYPE "affine"

#include "TypeScript/DataStructs.h"
#include "TypeScript/Passes.h"
#include "TypeScript/TypeScriptDialect.h"
#include "TypeScript/TypeScriptOps.h"
#include "TypeScript/TypeScriptFunctionPass.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Sequence.h"
#include "llvm/Support/Debug.h"

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
    DenseMap<Operation *, mlir::Block *> jumps;
};

template <typename OpTy> class TsPattern : public OpRewritePattern<OpTy>
{
  public:
    TsPattern<OpTy>(MLIRContext *context, TSContext *tsContext, PatternBenefit benefit = 1)
        : OpRewritePattern<OpTy>::OpRewritePattern(context, benefit), tsContext(tsContext)
    {
    }

  protected:
    TSContext *tsContext;
};

//===----------------------------------------------------------------------===//
// TypeScriptToAffine RewritePatterns
//===----------------------------------------------------------------------===//

struct ParamOpLowering : public TsPattern<mlir_ts::ParamOp>
{
    using TsPattern<mlir_ts::ParamOp>::TsPattern;

    LogicalResult matchAndRewrite(mlir_ts::ParamOp paramOp, PatternRewriter &rewriter) const final
    {
        rewriter.replaceOpWithNewOp<mlir_ts::VariableOp>(paramOp, paramOp.getType(), paramOp.argValue(), rewriter.getBoolAttr(false));
        return success();
    }
};

struct ParamOptionalOpLowering : public TsPattern<mlir_ts::ParamOptionalOp>
{
    using TsPattern<mlir_ts::ParamOptionalOp>::TsPattern;

    LogicalResult matchAndRewrite(mlir_ts::ParamOptionalOp paramOp, PatternRewriter &rewriter) const final
    {
        TypeHelper th(rewriter);

        auto location = paramOp.getLoc();

        auto dataTypeIn = paramOp.argValue().getType().cast<mlir_ts::OptionalType>().getElementType();
        auto storeType = paramOp.getType().cast<mlir_ts::RefType>().getElementType();

        // ts.if
        auto hasValue = rewriter.create<mlir_ts::HasValueOp>(location, th.getBooleanType(), paramOp.argValue());
        auto ifOp = rewriter.create<mlir_ts::IfOp>(location, storeType, hasValue, true);

        // then block
        auto &thenRegion = ifOp.thenRegion();

        rewriter.setInsertionPointToStart(&thenRegion.back());

        mlir::Value value = rewriter.create<mlir_ts::ValueOp>(location, storeType, paramOp.argValue());
        rewriter.create<mlir_ts::ResultOp>(location, value);

        // else block
        auto &elseRegion = ifOp.elseRegion();

        rewriter.setInsertionPointToStart(&elseRegion.back());

        rewriter.inlineRegionBefore(paramOp.defaultValueRegion(), &ifOp.elseRegion().back());
        rewriter.eraseBlock(&ifOp.elseRegion().back());

        rewriter.setInsertionPointAfter(ifOp);

        Value variable =
            rewriter.create<mlir_ts::VariableOp>(location, paramOp.getType(), ifOp.results().front(), rewriter.getBoolAttr(false));

        rewriter.replaceOp(paramOp, variable);

        return success();
    }
};

struct ParamDefaultValueOpLowering : public TsPattern<mlir_ts::ParamDefaultValueOp>
{
    using TsPattern<mlir_ts::ParamDefaultValueOp>::TsPattern;

    LogicalResult matchAndRewrite(mlir_ts::ParamDefaultValueOp op, PatternRewriter &rewriter) const final
    {
        rewriter.replaceOpWithNewOp<mlir_ts::ResultOp>(op, op.results());
        return success();
    }
};

struct PrefixUnaryOpLowering : public TsPattern<mlir_ts::PrefixUnaryOp>
{
    using TsPattern<mlir_ts::PrefixUnaryOp>::TsPattern;

    LogicalResult matchAndRewrite(mlir_ts::PrefixUnaryOp op, PatternRewriter &rewriter) const final
    {
        CodeLogicHelper clh(op, rewriter);
        auto cst1 = rewriter.create<mlir_ts::ConstantOp>(op->getLoc(), rewriter.getI32IntegerAttr(1));

        SyntaxKind opCode = SyntaxKind::Unknown;
        switch ((SyntaxKind)op.opCode())
        {
        case SyntaxKind::PlusPlusToken:
            opCode = SyntaxKind::PlusToken;
            break;
        case SyntaxKind::MinusMinusToken:
            opCode = SyntaxKind::MinusToken;
            break;
        }

        auto value = op.operand1();
        auto effectiveType = op.getType();
        bool castBack = false;
        if (auto optType = effectiveType.dyn_cast_or_null<mlir_ts::OptionalType>())
        {
            castBack = true;
            effectiveType = optType.getElementType();
            value = rewriter.create<mlir_ts::CastOp>(value.getLoc(), effectiveType, value);
        }

        mlir::Value result = rewriter.create<mlir_ts::ArithmeticBinaryOp>(
            op->getLoc(), effectiveType, rewriter.getI32IntegerAttr(static_cast<int32_t>(opCode)), value, cst1);

        if (castBack)
        {
            result = rewriter.create<mlir_ts::CastOp>(value.getLoc(), op.getType(), result);
        }

        rewriter.replaceOp(op, result);

        clh.saveResult(op, op->getResult(0));

        return success();
    }
};

struct PostfixUnaryOpLowering : public TsPattern<mlir_ts::PostfixUnaryOp>
{
    using TsPattern<mlir_ts::PostfixUnaryOp>::TsPattern;

    LogicalResult matchAndRewrite(mlir_ts::PostfixUnaryOp op, PatternRewriter &rewriter) const final
    {
        CodeLogicHelper clh(op, rewriter);
        auto cst1 = rewriter.create<mlir_ts::ConstantOp>(op->getLoc(), rewriter.getI32IntegerAttr(1));

        SyntaxKind opCode = SyntaxKind::Unknown;
        switch ((SyntaxKind)op.opCode())
        {
        case SyntaxKind::PlusPlusToken:
            opCode = SyntaxKind::PlusToken;
            break;
        case SyntaxKind::MinusMinusToken:
            opCode = SyntaxKind::MinusToken;
            break;
        }

        auto value = op.operand1();
        auto effectiveType = op.getType();
        bool castBack = false;
        if (auto optType = effectiveType.dyn_cast_or_null<mlir_ts::OptionalType>())
        {
            castBack = true;
            effectiveType = optType.getElementType();
            value = rewriter.create<mlir_ts::CastOp>(value.getLoc(), effectiveType, value);
        }

        mlir::Value result = rewriter.create<mlir_ts::ArithmeticBinaryOp>(
            op->getLoc(), effectiveType, rewriter.getI32IntegerAttr(static_cast<int32_t>(opCode)), value, cst1);
        if (castBack)
        {
            result = rewriter.create<mlir_ts::CastOp>(value.getLoc(), op.getType(), result);
        }

        clh.saveResult(op, result);

        rewriter.replaceOp(op, op.operand1());

        return success();
    }
};

struct IfOpLowering : public TsPattern<mlir_ts::IfOp>
{
    using TsPattern<mlir_ts::IfOp>::TsPattern;

    LogicalResult matchAndRewrite(mlir_ts::IfOp ifOp, PatternRewriter &rewriter) const final
    {
        auto loc = ifOp.getLoc();

        // Start by splitting the block containing the 'ts.if' into two parts.
        // The part before will contain the condition, the part after will be the
        // continuation point.
        auto *condBlock = rewriter.getInsertionBlock();
        auto opPosition = rewriter.getInsertionPoint();
        auto *remainingOpsBlock = rewriter.splitBlock(condBlock, opPosition);
        Block *continueBlock;
        if (ifOp.getNumResults() == 0)
        {
            continueBlock = remainingOpsBlock;
        }
        else
        {
            continueBlock = rewriter.createBlock(remainingOpsBlock, ifOp.getResultTypes());
            rewriter.create<BranchOp>(loc, remainingOpsBlock);
        }

        // Move blocks from the "then" region to the region containing 'ts.if',
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
        // 'ts.if', place it before the continuation block and branch to it.  It
        // will be placed after the "then" regions.
        auto *elseBlock = continueBlock;
        auto &elseRegion = ifOp.elseRegion();
        if (!elseRegion.empty())
        {
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

    LogicalResult matchAndRewrite(mlir_ts::WhileOp whileOp, PatternRewriter &rewriter) const final
    {
        OpBuilder::InsertionGuard guard(rewriter);
        Location loc = whileOp.getLoc();

        auto labelAttr = whileOp->getAttrOfType<StringAttr>(LABEL_ATTR_NAME);

        // Split the current block before the WhileOp to create the inlining point.
        auto *currentBlock = rewriter.getInsertionBlock();
        auto *continuation = rewriter.splitBlock(currentBlock, rewriter.getInsertionPoint());

        auto *body = &whileOp.body().front();
        auto *bodyLast = &whileOp.body().back();
        auto *cond = &whileOp.cond().front();
        auto *condLast = &whileOp.cond().back();

        // logic to support continue/break

        auto visitorBreakContinue = [&](Operation *op) {
            if (auto breakOp = dyn_cast_or_null<mlir_ts::BreakOp>(op))
            {
                auto set = MLIRHelper::matchLabelOrNotSet(labelAttr, breakOp.labelAttr());
                if (set)
                    tsContext->jumps[op] = continuation;
            }
            else if (auto continueOp = dyn_cast_or_null<mlir_ts::ContinueOp>(op))
            {
                auto set = MLIRHelper::matchLabelOrNotSet(labelAttr, continueOp.labelAttr());
                if (set)
                    tsContext->jumps[op] = cond;
            }
        };

        whileOp.body().walk(visitorBreakContinue);

        // end of logic for break/continue

        rewriter.inlineRegionBefore(whileOp.body(), continuation);
        rewriter.inlineRegionBefore(whileOp.cond(), body);

        // Branch to the "before" region.
        rewriter.setInsertionPointToEnd(currentBlock);
        rewriter.create<BranchOp>(loc, cond, whileOp.inits());

        // Replace terminators with branches. Assuming bodies are SESE, which holds
        // given only the patterns from this file, we only need to look at the last
        // block. This should be reconsidered if we allow break/continue.
        rewriter.setInsertionPointToEnd(condLast);
        auto condOp = cast<mlir_ts::ConditionOp>(condLast->getTerminator());
        auto castToI1 = rewriter.create<mlir_ts::CastOp>(loc, rewriter.getI1Type(), condOp.condition());
        rewriter.replaceOpWithNewOp<CondBranchOp>(condOp, castToI1, body, condOp.args(), continuation, ValueRange());

        rewriter.setInsertionPointToEnd(bodyLast);
        auto yieldOp = cast<mlir_ts::ResultOp>(bodyLast->getTerminator());
        rewriter.replaceOpWithNewOp<BranchOp>(yieldOp, cond, yieldOp.results());

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
struct DoWhileOpLowering : public TsPattern<mlir_ts::DoWhileOp>
{
    using TsPattern<mlir_ts::DoWhileOp>::TsPattern;

    LogicalResult matchAndRewrite(mlir_ts::DoWhileOp doWhileOp, PatternRewriter &rewriter) const final
    {
        Location loc = doWhileOp.getLoc();

        auto labelAttr = doWhileOp->getAttrOfType<StringAttr>(LABEL_ATTR_NAME);

        // Split the current block before the WhileOp to create the inlining point.
        OpBuilder::InsertionGuard guard(rewriter);
        Block *currentBlock = rewriter.getInsertionBlock();
        Block *continuation = rewriter.splitBlock(currentBlock, rewriter.getInsertionPoint());

        // Only the "before" region should be inlined.
        auto *body = &doWhileOp.body().front();
        auto *bodyLast = &doWhileOp.body().back();
        auto *cond = &doWhileOp.cond().front();
        auto *condLast = &doWhileOp.cond().back();

        // logic to support continue/break

        auto visitorBreakContinue = [&](Operation *op) {
            if (auto breakOp = dyn_cast_or_null<mlir_ts::BreakOp>(op))
            {
                auto set = MLIRHelper::matchLabelOrNotSet(labelAttr, breakOp.labelAttr());
                if (set)
                    tsContext->jumps[op] = continuation;
            }
            else if (auto continueOp = dyn_cast_or_null<mlir_ts::ContinueOp>(op))
            {
                auto set = MLIRHelper::matchLabelOrNotSet(labelAttr, continueOp.labelAttr());
                if (set)
                    tsContext->jumps[op] = cond;
            }
        };

        doWhileOp.body().walk(visitorBreakContinue);

        // end of logic for break/continue

        rewriter.inlineRegionBefore(doWhileOp.cond(), continuation);
        rewriter.inlineRegionBefore(doWhileOp.body(), cond);

        // Branch to the "before" region.
        rewriter.setInsertionPointToEnd(currentBlock);
        rewriter.create<BranchOp>(doWhileOp.getLoc(), body, doWhileOp.inits());

        rewriter.setInsertionPointToEnd(bodyLast);
        auto yieldOp = cast<mlir_ts::ResultOp>(bodyLast->getTerminator());
        rewriter.replaceOpWithNewOp<BranchOp>(yieldOp, cond, yieldOp.results());

        // Loop around the "before" region based on condition.
        rewriter.setInsertionPointToEnd(condLast);
        auto condOp = cast<mlir_ts::ConditionOp>(condLast->getTerminator());
        auto castToI1 = rewriter.create<mlir_ts::CastOp>(loc, rewriter.getI1Type(), condOp.condition());
        rewriter.replaceOpWithNewOp<CondBranchOp>(condOp, castToI1, body, condOp.args(), continuation, ValueRange());

        // Replace the op with values "yielded" from the "before" region, which are
        // visible by dominance.
        rewriter.replaceOp(doWhileOp, condOp.args());

        return success();
    }
};

struct ForOpLowering : public TsPattern<mlir_ts::ForOp>
{
    using TsPattern<mlir_ts::ForOp>::TsPattern;

    LogicalResult matchAndRewrite(mlir_ts::ForOp forOp, PatternRewriter &rewriter) const final
    {
        OpBuilder::InsertionGuard guard(rewriter);
        Location loc = forOp.getLoc();

        auto labelAttr = forOp->getAttrOfType<StringAttr>(LABEL_ATTR_NAME);

        // Split the current block before the WhileOp to create the inlining point.
        auto *currentBlock = rewriter.getInsertionBlock();
        auto *continuation = rewriter.splitBlock(currentBlock, rewriter.getInsertionPoint());

        auto *incr = &forOp.incr().front();
        auto *incrLast = &forOp.incr().back();
        auto *body = &forOp.body().front();
        auto *bodyLast = &forOp.body().back();
        auto *cond = &forOp.cond().front();
        auto *condLast = &forOp.cond().back();

        // logic to support continue/break

        auto visitorBreakContinue = [&](Operation *op) {
            if (auto breakOp = dyn_cast_or_null<mlir_ts::BreakOp>(op))
            {
                auto set = MLIRHelper::matchLabelOrNotSet(labelAttr, breakOp.labelAttr());
                if (set)
                    tsContext->jumps[op] = continuation;
            }
            else if (auto continueOp = dyn_cast_or_null<mlir_ts::ContinueOp>(op))
            {
                auto set = MLIRHelper::matchLabelOrNotSet(labelAttr, continueOp.labelAttr());
                if (set)
                    tsContext->jumps[op] = incr;
            }
        };

        forOp.body().walk(visitorBreakContinue);

        // end of logic for break/continue

        rewriter.inlineRegionBefore(forOp.incr(), continuation);
        rewriter.inlineRegionBefore(forOp.body(), incr);
        rewriter.inlineRegionBefore(forOp.cond(), body);

        // Branch to the "before" region.
        rewriter.setInsertionPointToEnd(currentBlock);
        rewriter.create<BranchOp>(loc, cond, forOp.inits());

        // Replace terminators with branches. Assuming bodies are SESE, which holds
        // given only the patterns from this file, we only need to look at the last
        // block. This should be reconsidered if we allow break/continue.
        rewriter.setInsertionPointToEnd(condLast);
        ValueRange args;
        if (auto condOp = dyn_cast_or_null<mlir_ts::ConditionOp>(condLast->getTerminator()))
        {
            args = condOp.args();
            auto castToI1 = rewriter.create<mlir_ts::CastOp>(loc, rewriter.getI1Type(), condOp.condition());
            rewriter.replaceOpWithNewOp<CondBranchOp>(condOp, castToI1, body, condOp.args(), continuation, ValueRange());
        }
        else
        {
            auto noCondOp = cast<mlir_ts::NoConditionOp>(condLast->getTerminator());
            rewriter.replaceOpWithNewOp<BranchOp>(noCondOp, body, noCondOp.args());
        }

        rewriter.setInsertionPointToEnd(bodyLast);

        auto yieldOpBody = cast<mlir_ts::ResultOp>(bodyLast->getTerminator());
        rewriter.replaceOpWithNewOp<BranchOp>(yieldOpBody, incr, yieldOpBody.results());

        rewriter.setInsertionPointToEnd(incrLast);

        auto yieldOpIncr = cast<mlir_ts::ResultOp>(incrLast->getTerminator());
        rewriter.replaceOpWithNewOp<BranchOp>(yieldOpIncr, cond, yieldOpIncr.results());

        // Replace the op with values "yielded" from the "before" region, which are
        // visible by dominance.
        rewriter.replaceOp(forOp, args);

        return success();
    }
};

struct LabelOpLowering : public TsPattern<mlir_ts::LabelOp>
{
    using TsPattern<mlir_ts::LabelOp>::TsPattern;

    LogicalResult matchAndRewrite(mlir_ts::LabelOp labelOp, PatternRewriter &rewriter) const final
    {
        // Split the current block before the WhileOp to create the inlining point.
        OpBuilder::InsertionGuard guard(rewriter);
        Location loc = labelOp.getLoc();

        Block *currentBlock = rewriter.getInsertionBlock();
        Block *continuation = rewriter.splitBlock(currentBlock, rewriter.getInsertionPoint());

        auto *begin = &labelOp.labelRegion().front();

        auto labelAttr = labelOp.labelAttr();

        // logic to support continue/break

        auto visitorBreakContinue = [&](Operation *op) {
            if (auto breakOp = dyn_cast_or_null<mlir_ts::BreakOp>(op))
            {
                auto set = MLIRHelper::matchLabelOrNotSet(labelAttr, breakOp.labelAttr());
                if (set)
                    tsContext->jumps[op] = continuation;
            }
            else if (auto continueOp = dyn_cast_or_null<mlir_ts::ContinueOp>(op))
            {
                auto set = MLIRHelper::matchLabelOrNotSet(labelAttr, continueOp.labelAttr());
                if (set)
                    tsContext->jumps[op] = begin;
            }
        };

        labelOp.labelRegion().walk(visitorBreakContinue);

        // end of logic for break/continue

        auto *labelRegion = &labelOp.labelRegion().front();

        auto *labelRegionWithMerge = &labelOp.labelRegion().back();
        for (auto &block : labelOp.labelRegion())
        {
            if (isa<mlir_ts::MergeOp>(block.getTerminator()))
            {
                labelRegionWithMerge = &block;
            }
        }

        // Branch to the "labelRegion" region.
        rewriter.setInsertionPointToEnd(currentBlock);
        rewriter.create<BranchOp>(loc, labelRegion, ValueRange{});

        rewriter.inlineRegionBefore(labelOp.labelRegion(), continuation);

        // replace merge with br
        assert(labelRegionWithMerge);
        rewriter.setInsertionPointToEnd(labelRegionWithMerge);

        if (auto mergeOp = dyn_cast_or_null<mlir_ts::MergeOp>(labelRegionWithMerge->getTerminator()))
        {
            rewriter.replaceOpWithNewOp<BranchOp>(mergeOp, continuation, ValueRange{});
        }
        else
        {
            assert(false);
        }

        rewriter.replaceOp(labelOp, continuation->getArguments());

        return success();
    }
};

struct BreakOpLowering : public TsPattern<mlir_ts::BreakOp>
{
    using TsPattern<mlir_ts::BreakOp>::TsPattern;

    LogicalResult matchAndRewrite(mlir_ts::BreakOp breakOp, PatternRewriter &rewriter) const final
    {
        OpBuilder::InsertionGuard guard(rewriter);

        auto jump = tsContext->jumps[breakOp];
        assert(jump);
        rewriter.replaceOpWithNewOp<BranchOp>(breakOp, jump /*break=continuation*/);

        auto *opBlock = rewriter.getInsertionBlock();
        auto opPosition = rewriter.getInsertionPoint();
        /*auto *continuationBlock = */ rewriter.splitBlock(opBlock, opPosition);

        return success();
    }
};

struct ContinueOpLowering : public TsPattern<mlir_ts::ContinueOp>
{
    using TsPattern<mlir_ts::ContinueOp>::TsPattern;

    LogicalResult matchAndRewrite(mlir_ts::ContinueOp continueOp, PatternRewriter &rewriter) const final
    {
        OpBuilder::InsertionGuard guard(rewriter);

        auto jump = tsContext->jumps[continueOp];
        assert(jump);
        rewriter.replaceOpWithNewOp<BranchOp>(continueOp, jump /*break=incremental-or-condition block*/);

        auto *opBlock = rewriter.getInsertionBlock();
        auto opPosition = rewriter.getInsertionPoint();
        /*auto *continuationBlock = */ rewriter.splitBlock(opBlock, opPosition);

        return success();
    }
};

struct SwitchOpLowering : public TsPattern<mlir_ts::SwitchOp>
{
    using TsPattern<mlir_ts::SwitchOp>::TsPattern;

    LogicalResult matchAndRewrite(mlir_ts::SwitchOp switchOp, PatternRewriter &rewriter) const final
    {
        Location loc = switchOp.getLoc();

        // Split the current block before the WhileOp to create the inlining point.
        OpBuilder::InsertionGuard guard(rewriter);
        Block *currentBlock = rewriter.getInsertionBlock();
        Block *continuation = rewriter.splitBlock(currentBlock, rewriter.getInsertionPoint());

        auto *casesRegion = &switchOp.casesRegion().front();

        auto *casesRegionWithMerge = &switchOp.casesRegion().back();
        for (auto &block : switchOp.casesRegion())
        {
            if (isa<mlir_ts::MergeOp>(block.getTerminator()))
            {
                casesRegionWithMerge = &block;
            }
        }

        // Branch to the "casesRegion" region.
        rewriter.setInsertionPointToEnd(currentBlock);
        rewriter.create<BranchOp>(loc, casesRegion, ValueRange{});

        rewriter.inlineRegionBefore(switchOp.casesRegion(), continuation);

        // replace merge with br
        assert(casesRegionWithMerge);
        rewriter.setInsertionPointToEnd(casesRegionWithMerge);

        if (auto mergeOp = dyn_cast_or_null<mlir_ts::MergeOp>(casesRegionWithMerge->getTerminator()))
        {
            rewriter.replaceOpWithNewOp<BranchOp>(mergeOp, continuation, ValueRange{});
        }
        else
        {
            assert(false);
        }

        rewriter.replaceOp(switchOp, continuation->getArguments());

        return success();
    }
};

struct AccessorRefOpLowering : public TsPattern<mlir_ts::AccessorRefOp>
{
    using TsPattern<mlir_ts::AccessorRefOp>::TsPattern;

    LogicalResult matchAndRewrite(mlir_ts::AccessorRefOp accessorRefOp, PatternRewriter &rewriter) const final
    {
        Location loc = accessorRefOp.getLoc();

        auto callRes =
            rewriter.create<mlir_ts::CallOp>(loc, accessorRefOp.getAccessor().getValue(), TypeRange{accessorRefOp.getType()}, ValueRange{});

        rewriter.replaceOp(accessorRefOp, callRes.getResult(0));
        return success();
    }
};

struct ThisAccessorRefOpLowering : public TsPattern<mlir_ts::ThisAccessorRefOp>
{
    using TsPattern<mlir_ts::ThisAccessorRefOp>::TsPattern;

    LogicalResult matchAndRewrite(mlir_ts::ThisAccessorRefOp thisAccessorRefOp, PatternRewriter &rewriter) const final
    {
        Location loc = thisAccessorRefOp.getLoc();

        auto callRes = rewriter.create<mlir_ts::CallOp>(loc, thisAccessorRefOp.getAccessor().getValue(),
                                                        TypeRange{thisAccessorRefOp.getType()}, ValueRange{thisAccessorRefOp.thisVal()});

        rewriter.replaceOp(thisAccessorRefOp, callRes.getResult(0));

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

namespace
{
struct TypeScriptToAffineLoweringPass : public PassWrapper<TypeScriptToAffineLoweringPass, TypeScriptFunctionPass>
{
    void getDependentDialects(DialectRegistry &registry) const override
    {
        registry.insert<StandardOpsDialect>();
    }

    void runOnFunction() final;

    TSContext tsContext;
};
} // end anonymous namespace.

void TypeScriptToAffineLoweringPass::runOnFunction()
{
    auto function = getFunction();

    // We only lower the main function as we expect that all other functions have been inlined.
    if (function.getName() == "main")
    {
        auto voidType = mlir_ts::VoidType::get(function.getContext());
        // Verify that the given main has no inputs and results.
        if (function.getNumArguments() || llvm::any_of(function.getType().getResults(), [&](mlir::Type type) { return type != voidType; }))
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
    target.addLegalDialect<StandardOpsDialect>();

    // We also define the TypeScript dialect as Illegal so that the conversion will fail
    // if any of these operations are *not* converted. Given that we actually want
    // a partial lowering, we explicitly mark the TypeScript operations that don't want
    // to lower, `typescript.print`, as `legal`.
    target.addIllegalDialect<mlir_ts::TypeScriptDialect>();
    target.addLegalOp<mlir_ts::AddressOfOp, mlir_ts::AddressOfConstStringOp, mlir_ts::AddressOfElementOp, mlir_ts::ArithmeticBinaryOp,
                      mlir_ts::ArithmeticUnaryOp, mlir_ts::AssertOp, mlir_ts::CallOp, mlir_ts::CallIndirectOp, mlir_ts::CaptureOp,
                      mlir_ts::CastOp, mlir_ts::ConstantOp, mlir_ts::EntryOp, mlir_ts::ExitOp, mlir_ts::ElementRefOp, mlir_ts::FuncOp,
                      mlir_ts::GlobalOp, mlir_ts::GlobalResultOp, mlir_ts::HasValueOp, mlir_ts::ValueOp, mlir_ts::NullOp,
                      mlir_ts::ParseFloatOp, mlir_ts::ParseIntOp, mlir_ts::PrintOp, mlir_ts::SizeOfOp, mlir_ts::ReturnOp,
                      mlir_ts::ReturnValOp, mlir_ts::StoreOp, mlir_ts::SymbolRefOp, mlir_ts::LengthOfOp, mlir_ts::StringLengthOp,
                      mlir_ts::StringConcatOp, mlir_ts::StringCompareOp, mlir_ts::LoadOp, mlir_ts::NewOp, mlir_ts::CreateTupleOp,
                      mlir_ts::DeconstructTupleOp, mlir_ts::CreateArrayOp, mlir_ts::NewEmptyArrayOp, mlir_ts::NewArrayOp, mlir_ts::DeleteOp,
                      mlir_ts::PropertyRefOp, mlir_ts::InsertPropertyOp, mlir_ts::ExtractPropertyOp, mlir_ts::LogicalBinaryOp,
                      mlir_ts::UndefOp, mlir_ts::VariableOp, mlir_ts::ThrowOp, mlir_ts::TryOp, mlir_ts::TrampolineOp, mlir_ts::InvokeOp,
                      mlir_ts::ResultOp, mlir_ts::ThisVirtualSymbolRefOp, mlir_ts::InterfaceSymbolRefOp, mlir_ts::PushOp, mlir_ts::PopOp,
                      mlir_ts::NewInterfaceOp, mlir_ts::VTableOffsetRefOp, mlir_ts::ThisPropertyRefOp, mlir_ts::GetThisOp,
                      mlir_ts::GetMethodOp, mlir_ts::TypeOfOp, mlir_ts::DebuggerOp, mlir_ts::SwitchStateOp, mlir_ts::StateLabelOp>();

    // Now that the conversion target has been defined, we just need to provide
    // the set of patterns that will lower the TypeScript operations.
    OwningRewritePatternList patterns(&getContext());
    patterns.insert<ParamOpLowering, ParamOptionalOpLowering, ParamDefaultValueOpLowering, PrefixUnaryOpLowering, PostfixUnaryOpLowering,
                    IfOpLowering, DoWhileOpLowering, WhileOpLowering, ForOpLowering, BreakOpLowering, ContinueOpLowering, SwitchOpLowering,
                    AccessorRefOpLowering, ThisAccessorRefOpLowering, LabelOpLowering>(&getContext(), &tsContext);

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
