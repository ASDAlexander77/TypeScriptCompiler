#include "TypeScript/Config.h"
#include "TypeScript/DataStructs.h"
#include "TypeScript/Passes.h"
#include "TypeScript/TypeScriptDialect.h"
#include "TypeScript/TypeScriptFunctionPass.h"
#include "TypeScript/TypeScriptOps.h"
#include "TypeScript/TypeScriptPassContext.h"

#include "TypeScript/MLIRLogic/MLIRRTTIHelperVC.h"

#include "TypeScript/LowerToLLVMLogic.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Async/IR/Async.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Sequence.h"
#include "mlir/IR/Dialect.h"
#include "llvm/Support/Debug.h"

#include "scanner_enums.h"

using namespace mlir;
using namespace ::typescript;
namespace mlir_ts = mlir::typescript;

#define ENABLE_SWITCH_STATE_PASS 1

#define DEBUG_TYPE "affine"

namespace
{

//===----------------------------------------------------------------------===//
// TypeScriptToAffine RewritePatterns
//===----------------------------------------------------------------------===//

struct EntryOpLowering : public TsPattern<mlir_ts::EntryOp>
{
    using TsPattern<mlir_ts::EntryOp>::TsPattern;

    LogicalResult matchAndRewrite(mlir_ts::EntryOp op, PatternRewriter &rewriter) const final
    {
        auto location = op.getLoc();

        mlir::Value allocValue;
        mlir::Type returnType;
        auto anyResult = op.getNumResults() > 0;
        if (anyResult)
        {
            auto result = op.getResult(0);
            returnType = result.getType();
            allocValue = rewriter.create<mlir_ts::VariableOp>(
                location, returnType, mlir::Value(), rewriter.getBoolAttr(false), rewriter.getIndexAttr(0));
        }

        // create return block
        auto *opBlock = rewriter.getInsertionBlock();
        auto *region = opBlock->getParent();

        tsContext->returnBlock = rewriter.createBlock(region);

        if (anyResult)
        {
            auto loadedValue = rewriter.create<mlir_ts::LoadOp>(
                location, returnType.cast<mlir_ts::RefType>().getElementType(), allocValue);
            rewriter.create<mlir_ts::ReturnInternalOp>(location, mlir::ValueRange{loadedValue});
            rewriter.replaceOp(op, allocValue);
        }
        else
        {
            rewriter.create<mlir_ts::ReturnInternalOp>(location, mlir::ValueRange{});
            rewriter.eraseOp(op);
        }

        return success();
    }
};

struct ExitOpLowering : public TsPattern<mlir_ts::ExitOp>
{
    using TsPattern<mlir_ts::ExitOp>::TsPattern;

    LogicalResult matchAndRewrite(mlir_ts::ExitOp op, PatternRewriter &rewriter) const final
    {
        assert(tsContext->returnBlock);

        auto retBlock = tsContext->returnBlock;

        rewriter.create<mlir::cf::BranchOp>(op.getLoc(), retBlock);

        rewriter.eraseOp(op);
        return success();
    }
};

struct ReturnOpLowering : public TsPattern<mlir_ts::ReturnOp>
{
    using TsPattern<mlir_ts::ReturnOp>::TsPattern;

    LogicalResult matchAndRewrite(mlir_ts::ReturnOp op, PatternRewriter &rewriter) const final
    {
        auto loc = op.getLoc();

        assert(tsContext->returnBlock);

        auto retBlock = tsContext->returnBlock;
        if (auto unwind = tsContext->unwind[op])
        {
            rewriter.create<mlir_ts::EndCatchOp>(loc);
        }

        auto *opBlock = rewriter.getInsertionBlock();
        auto opPosition = rewriter.getInsertionPoint();
        auto *continuationBlock = rewriter.splitBlock(opBlock, opPosition);

        rewriter.setInsertionPointToEnd(opBlock);

        if (auto cleanup = tsContext->cleanup[op])
        {
            rewriter.create<mlir::cf::BranchOp>(loc, cleanup);
        }
        else
        {
            rewriter.create<mlir::cf::BranchOp>(loc, retBlock);
        }

        rewriter.setInsertionPointToStart(continuationBlock);

        rewriter.eraseOp(op);
        return success();
    }
};

struct ReturnValOpLowering : public TsPattern<mlir_ts::ReturnValOp>
{
    using TsPattern<mlir_ts::ReturnValOp>::TsPattern;

    LogicalResult matchAndRewrite(mlir_ts::ReturnValOp op, PatternRewriter &rewriter) const final
    {
        auto loc = op.getLoc();

        assert(tsContext->returnBlock);

        auto retBlock = tsContext->returnBlock;

        // save value into return
        rewriter.create<mlir_ts::StoreOp>(op.getLoc(), op.getOperand(), op.getReference());
        if (auto unwind = tsContext->unwind[op])
        {
            rewriter.create<mlir_ts::EndCatchOp>(loc);
        }
        
        // Split block at `assert` operation.
        auto *opBlock = rewriter.getInsertionBlock();
        auto opPosition = rewriter.getInsertionPoint();
        auto *continuationBlock = rewriter.splitBlock(opBlock, opPosition);

        rewriter.setInsertionPointToEnd(opBlock);

        if (auto cleanup = tsContext->cleanup[op])
        {
            rewriter.create<mlir::cf::BranchOp>(loc, cleanup);
        }
        else
        {
            rewriter.create<mlir::cf::BranchOp>(loc, retBlock);
        }

        rewriter.setInsertionPointToStart(continuationBlock);

        rewriter.eraseOp(op);
        return success();
    }
};

struct ParamOpLowering : public TsPattern<mlir_ts::ParamOp>
{
    using TsPattern<mlir_ts::ParamOp>::TsPattern;

    LogicalResult matchAndRewrite(mlir_ts::ParamOp paramOp, PatternRewriter &rewriter) const final
    {
        rewriter.replaceOpWithNewOp<mlir_ts::VariableOp>(paramOp, paramOp.getType(), paramOp.getArgValue(),
                                                         paramOp.getCapturedAttr(), paramOp.getDiArgNumberAttr());
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

        auto dataTypeIn = paramOp.getArgValue().getType().cast<mlir_ts::OptionalType>().getElementType();
        auto storeType = paramOp.getType().cast<mlir_ts::RefType>().getElementType();

        // ts.if
        auto hasValue = rewriter.create<mlir_ts::HasValueOp>(location, th.getBooleanType(), paramOp.getArgValue());
        auto ifOp = rewriter.create<mlir_ts::IfOp>(location, storeType, hasValue, true);

        // then block
        auto &thenRegion = ifOp.getThenRegion();

        rewriter.setInsertionPointToStart(&thenRegion.back());

        mlir::Value value = rewriter.create<mlir_ts::ValueOp>(location, storeType, paramOp.getArgValue());
        rewriter.create<mlir_ts::ResultOp>(location, value);

        // else block
        auto &elseRegion = ifOp.getElseRegion();

        rewriter.setInsertionPointToStart(&elseRegion.back());

        rewriter.inlineRegionBefore(paramOp.getDefaultValueRegion(), &ifOp.getElseRegion().back());
        // TODO: do I need next line?
        rewriter.eraseBlock(&ifOp.getElseRegion().back());

        rewriter.setInsertionPointAfter(ifOp);

        auto variable = rewriter.create<mlir_ts::VariableOp>(location, paramOp.getType(), ifOp.getResults().front(),
                                                             paramOp.getCapturedAttr(), paramOp.getDiArgNumberAttr());
        rewriter.replaceOp(paramOp, mlir::Value(variable));

        return success();
    }
};

struct ParamDefaultValueOpLowering : public TsPattern<mlir_ts::ParamDefaultValueOp>
{
    using TsPattern<mlir_ts::ParamDefaultValueOp>::TsPattern;

    LogicalResult matchAndRewrite(mlir_ts::ParamDefaultValueOp op, PatternRewriter &rewriter) const final
    {
        rewriter.replaceOpWithNewOp<mlir_ts::ResultOp>(op, op.getResults());
        return success();
    }
};

struct OptionalValueOrDefaultOpLowering : public TsPattern<mlir_ts::OptionalValueOrDefaultOp>
{
    using TsPattern<mlir_ts::OptionalValueOrDefaultOp>::TsPattern;

    LogicalResult matchAndRewrite(mlir_ts::OptionalValueOrDefaultOp optionalValueOrDefaultOp, PatternRewriter &rewriter) const final
    {
        TypeHelper th(rewriter);

        auto location = optionalValueOrDefaultOp.getLoc();

        auto dataTypeIn = optionalValueOrDefaultOp.getArgValue().getType().cast<mlir_ts::OptionalType>().getElementType();
        auto resultType = optionalValueOrDefaultOp.getType();

        // ts.if
        auto hasValue = rewriter.create<mlir_ts::HasValueOp>(location, th.getBooleanType(), optionalValueOrDefaultOp.getArgValue());
        auto ifOp = rewriter.create<mlir_ts::IfOp>(location, resultType, hasValue, true);

        // then block
        auto &thenRegion = ifOp.getThenRegion();

        rewriter.setInsertionPointToStart(&thenRegion.back());

        mlir::Value value = rewriter.create<mlir_ts::ValueOp>(location, resultType, optionalValueOrDefaultOp.getArgValue());
        rewriter.create<mlir_ts::ResultOp>(location, value);

        // else block
        auto &elseRegion = ifOp.getElseRegion();

        rewriter.setInsertionPointToStart(&elseRegion.back());

        rewriter.inlineRegionBefore(optionalValueOrDefaultOp.getDefaultValueRegion(), &ifOp.getElseRegion().back());
        // TODO: do I need next line?
        rewriter.eraseBlock(&ifOp.getElseRegion().back());

        rewriter.setInsertionPointAfter(ifOp);

        rewriter.replaceOp(optionalValueOrDefaultOp, ifOp.getResults().front());

        return success();
    }
};

struct PrefixUnaryOpLowering : public TsPattern<mlir_ts::PrefixUnaryOp>
{
    using TsPattern<mlir_ts::PrefixUnaryOp>::TsPattern;

    LogicalResult matchAndRewrite(mlir_ts::PrefixUnaryOp op, PatternRewriter &rewriter) const final
    {
        CodeLogicHelper clh(op, rewriter);
        mlir::Value cst1 = rewriter.create<mlir_ts::ConstantOp>(op->getLoc(), rewriter.getI32IntegerAttr(1));

        SyntaxKind opCode = SyntaxKind::Unknown;
        switch ((SyntaxKind)op.getOpCode())
        {
        case SyntaxKind::PlusPlusToken:
            opCode = SyntaxKind::PlusToken;
            break;
        case SyntaxKind::MinusMinusToken:
            opCode = SyntaxKind::MinusToken;
            break;
        }

        auto value = op.getOperand1();
        auto effectiveType = op.getType();
        bool castBack = false;
        if (auto optType = effectiveType.dyn_cast<mlir_ts::OptionalType>())
        {
            castBack = true;
            effectiveType = optType.getElementType();
            value = rewriter.create<mlir_ts::CastOp>(value.getLoc(), effectiveType, value);
        }

        if (value.getType() != cst1.getType())
        {
            cst1 = rewriter.create<mlir_ts::CastOp>(value.getLoc(), value.getType(), cst1);
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
        mlir::Value cst1 = rewriter.create<mlir_ts::ConstantOp>(op->getLoc(), rewriter.getI32IntegerAttr(1));

        SyntaxKind opCode = SyntaxKind::Unknown;
        switch ((SyntaxKind)op.getOpCode())
        {
        case SyntaxKind::PlusPlusToken:
            opCode = SyntaxKind::PlusToken;
            break;
        case SyntaxKind::MinusMinusToken:
            opCode = SyntaxKind::MinusToken;
            break;
        }

        auto value = op.getOperand1();
        auto effectiveType = op.getType();
        bool castBack = false;
        if (auto optType = effectiveType.dyn_cast<mlir_ts::OptionalType>())
        {
            castBack = true;
            effectiveType = optType.getElementType();
            value = rewriter.create<mlir_ts::CastOp>(value.getLoc(), effectiveType, value);
        }

        if (value.getType() != cst1.getType())
        {
            cst1 = rewriter.create<mlir_ts::CastOp>(value.getLoc(), value.getType(), cst1);
        }

        mlir::Value result = rewriter.create<mlir_ts::ArithmeticBinaryOp>(
            op->getLoc(), effectiveType, rewriter.getI32IntegerAttr(static_cast<int32_t>(opCode)), value, cst1);
        if (castBack)
        {
            result = rewriter.create<mlir_ts::CastOp>(value.getLoc(), op.getType(), result);
        }

        clh.saveResult(op, result);

        rewriter.replaceOp(op, op.getOperand1());

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
        mlir::Block *continueBlock;
        if (ifOp.getNumResults() == 0)
        {
            continueBlock = remainingOpsBlock;
        }
        else
        {
            continueBlock =
                rewriter.createBlock(remainingOpsBlock, ifOp.getResultTypes(),
                                     SmallVector<Location>(ifOp->getNumResults(), ifOp.getLoc()));
            rewriter.create<mlir::cf::BranchOp>(loc, remainingOpsBlock);
        }

        // Move blocks from the "then" region to the region containing 'ts.if',
        // place it before the continuation block, and branch to it.
        auto &thenRegion = ifOp.getThenRegion();
        auto *thenBlock = &thenRegion.front();
        Operation *thenTerminator = thenRegion.back().getTerminator();
        ValueRange thenTerminatorOperands = thenTerminator->getOperands();
        rewriter.setInsertionPointToEnd(&thenRegion.back());
        rewriter.create<mlir::cf::BranchOp>(loc, continueBlock, thenTerminatorOperands);
        rewriter.eraseOp(thenTerminator);
        rewriter.inlineRegionBefore(thenRegion, continueBlock);

        // Move blocks from the "else" region (if present) to the region containing
        // 'ts.if', place it before the continuation block and branch to it.  It
        // will be placed after the "then" regions.
        auto *elseBlock = continueBlock;
        auto &elseRegion = ifOp.getElseRegion();
        if (!elseRegion.empty())
        {
            elseBlock = &elseRegion.front();
            Operation *elseTerminator = elseRegion.back().getTerminator();
            ValueRange elseTerminatorOperands = elseTerminator->getOperands();
            rewriter.setInsertionPointToEnd(&elseRegion.back());
            rewriter.create<mlir::cf::BranchOp>(loc, continueBlock, elseTerminatorOperands);
            rewriter.eraseOp(elseTerminator);
            rewriter.inlineRegionBefore(elseRegion, continueBlock);
        }

        rewriter.setInsertionPointToEnd(condBlock);
        auto castToI1 = rewriter.create<mlir_ts::CastOp>(loc, rewriter.getI1Type(), ifOp.getCondition());
        rewriter.create<mlir::cf::CondBranchOp>(loc, castToI1, thenBlock,
                                      /*trueArgs=*/ArrayRef<mlir::Value>(), elseBlock,
                                      /*falseArgs=*/ArrayRef<mlir::Value>());

        // Ok, we're done!
        rewriter.replaceOp(ifOp, continueBlock->getArguments());

        LLVM_DEBUG(llvm::dbgs() << "\n!! IfOpLowering AFTER DUMP: \n" << *ifOp->getParentOp() << "\n";);

        return success();
    }
};

struct ResultOpLowering : public TsPattern<mlir_ts::ResultOp>
{
    using TsPattern<mlir_ts::ResultOp>::TsPattern;

    LogicalResult matchAndRewrite(mlir_ts::ResultOp resultOp, PatternRewriter &rewriter) const final
    {
        rewriter.eraseOp(resultOp);
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

        auto *body = &whileOp.getBody().front();
        auto *bodyLast = &whileOp.getBody().back();
        auto *cond = &whileOp.getCond().front();
        auto *condLast = &whileOp.getCond().back();

        // logic to support continue/break

        auto visitorBreakContinue = [&](Operation *op) {
            if (auto breakOp = dyn_cast_or_null<mlir_ts::BreakOp>(op))
            {
                auto set = MLIRHelper::matchLabelOrNotSet(labelAttr, breakOp.getLabelAttr());
                if (set)
                    tsContext->jumps[op] = continuation;
            }
            else if (auto continueOp = dyn_cast_or_null<mlir_ts::ContinueOp>(op))
            {
                auto set = MLIRHelper::matchLabelOrNotSet(labelAttr, continueOp.getLabelAttr());
                if (set)
                    tsContext->jumps[op] = cond;
            }
        };

        whileOp.getBody().walk(visitorBreakContinue);

        // end of logic for break/continue

        rewriter.inlineRegionBefore(whileOp.getBody(), continuation);
        rewriter.inlineRegionBefore(whileOp.getCond(), body);

        // Branch to the "before" region.
        rewriter.setInsertionPointToEnd(currentBlock);
        rewriter.create<mlir::cf::BranchOp>(loc, cond, whileOp.getInits());

        // Replace terminators with branches. Assuming bodies are SESE, which holds
        // given only the patterns from this file, we only need to look at the last
        // block. This should be reconsidered if we allow break/continue.
        rewriter.setInsertionPointToEnd(condLast);
        auto condOp = cast<mlir_ts::ConditionOp>(condLast->getTerminator());
        auto castToI1 = rewriter.create<mlir_ts::CastOp>(loc, rewriter.getI1Type(), condOp.getCondition());
        rewriter.replaceOpWithNewOp<mlir::cf::CondBranchOp>(condOp, castToI1, body, condOp.getArgs(), continuation, ValueRange());

        rewriter.setInsertionPointToEnd(bodyLast);
        auto yieldOp = cast<mlir_ts::ResultOp>(bodyLast->getTerminator());
        rewriter.replaceOpWithNewOp<mlir::cf::BranchOp>(yieldOp, cond, yieldOp.getResults());

        // Replace the op with values "yielded" from the "before" region, which are
        // visible by dominance.
        rewriter.replaceOp(whileOp, condOp.getArgs());

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
        mlir::Block *currentBlock = rewriter.getInsertionBlock();
        mlir::Block *continuation = rewriter.splitBlock(currentBlock, rewriter.getInsertionPoint());

        // Only the "before" region should be inlined.
        auto *body = &doWhileOp.getBody().front();
        auto *bodyLast = &doWhileOp.getBody().back();
        auto *cond = &doWhileOp.getCond().front();
        auto *condLast = &doWhileOp.getCond().back();

        // logic to support continue/break

        auto visitorBreakContinue = [&](Operation *op) {
            if (auto breakOp = dyn_cast_or_null<mlir_ts::BreakOp>(op))
            {
                auto set = MLIRHelper::matchLabelOrNotSet(labelAttr, breakOp.getLabelAttr());
                if (set)
                    tsContext->jumps[op] = continuation;
            }
            else if (auto continueOp = dyn_cast_or_null<mlir_ts::ContinueOp>(op))
            {
                auto set = MLIRHelper::matchLabelOrNotSet(labelAttr, continueOp.getLabelAttr());
                if (set)
                    tsContext->jumps[op] = cond;
            }
        };

        doWhileOp.getBody().walk(visitorBreakContinue);

        // end of logic for break/continue

        rewriter.inlineRegionBefore(doWhileOp.getCond(), continuation);
        rewriter.inlineRegionBefore(doWhileOp.getBody(), cond);

        // Branch to the "before" region.
        rewriter.setInsertionPointToEnd(currentBlock);
        rewriter.create<mlir::cf::BranchOp>(doWhileOp.getLoc(), body, doWhileOp.getInits());

        rewriter.setInsertionPointToEnd(bodyLast);
        auto yieldOp = cast<mlir_ts::ResultOp>(bodyLast->getTerminator());
        rewriter.replaceOpWithNewOp<mlir::cf::BranchOp>(yieldOp, cond, yieldOp.getResults());

        // Loop around the "before" region based on condition.
        rewriter.setInsertionPointToEnd(condLast);
        auto condOp = cast<mlir_ts::ConditionOp>(condLast->getTerminator());
        auto castToI1 = rewriter.create<mlir_ts::CastOp>(loc, rewriter.getI1Type(), condOp.getCondition());
        rewriter.replaceOpWithNewOp<mlir::cf::CondBranchOp>(condOp, castToI1, body, condOp.getArgs(), continuation, ValueRange());

        // Replace the op with values "yielded" from the "before" region, which are
        // visible by dominance.
        rewriter.replaceOp(doWhileOp, condOp.getArgs());

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

        auto *incr = &forOp.getIncr().front();
        auto *incrLast = &forOp.getIncr().back();
        auto *body = &forOp.getBody().front();
        auto *bodyLast = &forOp.getBody().back();
        auto *cond = &forOp.getCond().front();
        auto *condLast = &forOp.getCond().back();

        // logic to support continue/break

        auto visitorBreakContinue = [&](Operation *op) {
            if (auto breakOp = dyn_cast_or_null<mlir_ts::BreakOp>(op))
            {
                auto set = MLIRHelper::matchLabelOrNotSet(labelAttr, breakOp.getLabelAttr());
                if (set)
                    tsContext->jumps[op] = continuation;
            }
            else if (auto continueOp = dyn_cast_or_null<mlir_ts::ContinueOp>(op))
            {
                auto set = MLIRHelper::matchLabelOrNotSet(labelAttr, continueOp.getLabelAttr());
                if (set)
                    tsContext->jumps[op] = incr;
            }
        };

        forOp.getBody().walk(visitorBreakContinue);

        // end of logic for break/continue

        rewriter.inlineRegionBefore(forOp.getIncr(), continuation);
        rewriter.inlineRegionBefore(forOp.getBody(), incr);
        rewriter.inlineRegionBefore(forOp.getCond(), body);

        // Branch to the "before" region.
        rewriter.setInsertionPointToEnd(currentBlock);
        rewriter.create<mlir::cf::BranchOp>(loc, cond, forOp.getInits());

        // Replace terminators with branches. Assuming bodies are SESE, which holds
        // given only the patterns from this file, we only need to look at the last
        // block. This should be reconsidered if we allow break/continue.
        rewriter.setInsertionPointToEnd(condLast);
        ValueRange args;
        if (auto condOp = dyn_cast_or_null<mlir_ts::ConditionOp>(condLast->getTerminator()))
        {
            args = condOp.getArgs();
            auto castToI1 = rewriter.create<mlir_ts::CastOp>(loc, rewriter.getI1Type(), condOp.getCondition());
            rewriter.replaceOpWithNewOp<mlir::cf::CondBranchOp>(condOp, castToI1, body, condOp.getArgs(), continuation,
                                                      ValueRange());
        }
        else
        {
            auto noCondOp = cast<mlir_ts::NoConditionOp>(condLast->getTerminator());
            rewriter.replaceOpWithNewOp<mlir::cf::BranchOp>(noCondOp, body, noCondOp.getArgs());
        }

        rewriter.setInsertionPointToEnd(bodyLast);

        auto yieldOpBody = cast<mlir_ts::ResultOp>(bodyLast->getTerminator());
        rewriter.replaceOpWithNewOp<mlir::cf::BranchOp>(yieldOpBody, incr, yieldOpBody.getResults());

        rewriter.setInsertionPointToEnd(incrLast);

        auto yieldOpIncr = cast<mlir_ts::ResultOp>(incrLast->getTerminator());
        rewriter.replaceOpWithNewOp<mlir::cf::BranchOp>(yieldOpIncr, cond, yieldOpIncr.getResults());

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

        mlir::Block *currentBlock = rewriter.getInsertionBlock();
        mlir::Block *continuation = rewriter.splitBlock(currentBlock, rewriter.getInsertionPoint());

        auto *begin = &labelOp.getLabelRegion().front();

        auto labelAttr = labelOp.getLabelAttr();

        // logic to support continue/break

        auto visitorBreakContinue = [&](Operation *op) {
            if (auto breakOp = dyn_cast_or_null<mlir_ts::BreakOp>(op))
            {
                auto set = MLIRHelper::matchLabelOrNotSet(labelAttr, breakOp.getLabelAttr());
                if (set)
                    tsContext->jumps[op] = continuation;
            }
            else if (auto continueOp = dyn_cast_or_null<mlir_ts::ContinueOp>(op))
            {
                auto set = MLIRHelper::matchLabelOrNotSet(labelAttr, continueOp.getLabelAttr());
                if (set)
                    tsContext->jumps[op] = begin;
            }
        };

        labelOp.getLabelRegion().walk(visitorBreakContinue);

        // end of logic for break/continue

        auto *labelRegion = &labelOp.getLabelRegion().front();

        auto *labelRegionWithMerge = &labelOp.getLabelRegion().back();
        for (auto &block : labelOp.getLabelRegion())
        {
            if (isa<mlir_ts::MergeOp>(block.getTerminator()))
            {
                labelRegionWithMerge = &block;
            }
        }

        // Branch to the "labelRegion" region.
        rewriter.setInsertionPointToEnd(currentBlock);
        rewriter.create<mlir::cf::BranchOp>(loc, labelRegion, ValueRange{});

        rewriter.inlineRegionBefore(labelOp.getLabelRegion(), continuation);

        // replace merge with br
        assert(labelRegionWithMerge);
        rewriter.setInsertionPointToEnd(labelRegionWithMerge);

        if (auto mergeOp = dyn_cast_or_null<mlir_ts::MergeOp>(labelRegionWithMerge->getTerminator()))
        {
            rewriter.replaceOpWithNewOp<mlir::cf::BranchOp>(mergeOp, continuation, ValueRange{});
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
        CodeLogicHelper clh(breakOp, rewriter);

        OpBuilder::InsertionGuard guard(rewriter);
        Location loc = breakOp.getLoc();

        auto jump = tsContext->jumps[breakOp];
        assert(jump);

        rewriter.replaceOpWithNewOp<mlir::cf::BranchOp>(breakOp, jump);
        clh.CutBlock();

        return success();
    }
};

struct ContinueOpLowering : public TsPattern<mlir_ts::ContinueOp>
{
    using TsPattern<mlir_ts::ContinueOp>::TsPattern;

    LogicalResult matchAndRewrite(mlir_ts::ContinueOp continueOp, PatternRewriter &rewriter) const final
    {
        CodeLogicHelper clh(continueOp, rewriter);

        OpBuilder::InsertionGuard guard(rewriter);
        Location loc = continueOp.getLoc();

        auto jump = tsContext->jumps[continueOp];
        assert(jump);

        rewriter.replaceOpWithNewOp<mlir::cf::BranchOp>(continueOp, jump);
        clh.CutBlock();

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
        mlir::Block *currentBlock = rewriter.getInsertionBlock();
        mlir::Block *continuation = rewriter.splitBlock(currentBlock, rewriter.getInsertionPoint());

        auto *casesRegion = &switchOp.getCasesRegion().front();

        auto *casesRegionWithMerge = &switchOp.getCasesRegion().back();
        for (auto &block : switchOp.getCasesRegion())
        {
            if (isa<mlir_ts::MergeOp>(block.getTerminator()))
            {
                casesRegionWithMerge = &block;
            }
        }

        // Branch to the "casesRegion" region.
        rewriter.setInsertionPointToEnd(currentBlock);
        rewriter.create<mlir::cf::BranchOp>(loc, casesRegion, ValueRange{});

        rewriter.inlineRegionBefore(switchOp.getCasesRegion(), continuation);

        // replace merge with br
        assert(casesRegionWithMerge);
        rewriter.setInsertionPointToEnd(casesRegionWithMerge);

        if (auto mergeOp = dyn_cast_or_null<mlir_ts::MergeOp>(casesRegionWithMerge->getTerminator()))
        {
            rewriter.replaceOpWithNewOp<mlir::cf::BranchOp>(mergeOp, continuation, ValueRange{});
        }
        else
        {
            assert(false);
        }

        rewriter.replaceOp(switchOp, continuation->getArguments());

        return success();
    }
};

struct AccessorOpLowering : public TsPattern<mlir_ts::AccessorOp>
{
    using TsPattern<mlir_ts::AccessorOp>::TsPattern;

    LogicalResult matchAndRewrite(mlir_ts::AccessorOp accessorOp, PatternRewriter &rewriter) const final
    {
        Location loc = accessorOp.getLoc();

        if (!accessorOp.getGetAccessor().has_value())
        {
            emitError(loc) << "property does not have get accessor";
            return failure();
        }

        auto callRes = rewriter.create<mlir_ts::CallOp>(loc, accessorOp.getGetAccessor().value(),
                                                        TypeRange{accessorOp.getType()}, ValueRange{});

        rewriter.replaceOp(accessorOp, callRes.getResult(0));
        return success();
    }
};

struct ThisAccessorOpLowering : public TsPattern<mlir_ts::ThisAccessorOp>
{
    using TsPattern<mlir_ts::ThisAccessorOp>::TsPattern;

    LogicalResult matchAndRewrite(mlir_ts::ThisAccessorOp thisAccessorOp, PatternRewriter &rewriter) const final
    {
        Location loc = thisAccessorOp.getLoc();

        if (!thisAccessorOp.getGetAccessor().has_value())
        {
            emitError(loc) << "property does not have get accessor";
            return failure();
        }

        auto callRes =
            rewriter.create<mlir_ts::CallOp>(loc, thisAccessorOp.getGetAccessor().value(),
                                             TypeRange{thisAccessorOp.getType()}, ValueRange{thisAccessorOp.getThisVal()});

        rewriter.replaceOp(thisAccessorOp, callRes.getResult(0));

        return success();
    }
};

struct TryOpLowering : public TsPattern<mlir_ts::TryOp>
{
    using TsPattern<mlir_ts::TryOp>::TsPattern;

    // TODO: set 'loc' correctly to newly created ops
    LogicalResult matchAndRewrite(mlir_ts::TryOp tryOp, PatternRewriter &rewriter) const final
    {
        Location loc = tryOp.getLoc();

        LLVM_DEBUG(llvm::dbgs() << "\n!! BEFORE TRY OP DUMP: \n" << *tryOp->getParentOp() << "\n";);

        MLIRTypeHelper mth(rewriter.getContext());
        CodeLogicHelper clh(tryOp, rewriter);

        auto module = tryOp->getParentOfType<mlir::ModuleOp>();
        auto parentTryOp = tsContext->parentTryOp[tryOp.getOperation()];
        mlir::Block *parentTryOpLandingPad = parentTryOp ? tsContext->landingBlockOf[parentTryOp] : nullptr;

        MLIRRTTIHelperVC rttih(rewriter, module, tsContext->compileOptions);
        auto i8PtrTy = mth.getOpaqueType();

        // find catch var
        Operation *catchOpPtr = nullptr;
        auto visitorCatchContinue = [&](Operation *op) {
            if (auto catchOp = dyn_cast_or_null<mlir_ts::CatchOp>(op))
            {
                rttih.setType(catchOp.getCatchArg().getType().cast<mlir_ts::RefType>().getElementType());
                assert(!catchOpPtr);
                catchOpPtr = op;
            }
        };
        tryOp.getCatches().walk(visitorCatchContinue);

        // set TryOp -> child TryOp
        auto visitorTryOps = [&](Operation *op) {
            if (auto childTryOp = dyn_cast_or_null<mlir_ts::TryOp>(op))
            {
                tsContext->parentTryOp[op] = tryOp.getOperation();
            }
        };
        tryOp.getBody().walk(visitorTryOps);
        tryOp.getCatches().walk(visitorTryOps);
        tryOp.getFinally().walk(visitorTryOps);

        mlir::SmallVector<Operation *> returns;
        auto visitorReturnOps = [&](Operation *op) {
            if (auto returnOp = dyn_cast_or_null<mlir_ts::ReturnOp>(op))
            {
                returns.push_back(op);
            }
            else if (auto returnValOp = dyn_cast_or_null<mlir_ts::ReturnValOp>(op))
            {
                returns.push_back(op);
            }
        };
        tryOp.getBody().walk(visitorReturnOps);
        tryOp.getCatches().walk(visitorReturnOps);

        // inline structure
        OpBuilder::InsertionGuard guard(rewriter);
        mlir::Block *currentBlock = rewriter.getInsertionBlock();
        mlir::Block *continuation = rewriter.splitBlock(currentBlock, rewriter.getInsertionPoint());

        auto cleanupHasOps =
            llvm::any_of(tryOp.getCleanup(), [](auto &block) { return &block.front() != block.getTerminator(); });
        auto catchHasOps =
            llvm::any_of(tryOp.getCatches(), [](auto &block) { return &block.front() != block.getTerminator(); });
        auto finallyHasOps =
            llvm::any_of(tryOp.getFinally(), [](auto &block) { return &block.front() != block.getTerminator(); });

        // body
        rewriter.inlineRegionBefore(tryOp.getBody(), continuation);
        auto bodyBlock = currentBlock->getNextNode();
        auto bodyBlockLast = continuation->getPrevNode();

        // Branch to the "body" region.
        rewriter.setInsertionPointToEnd(currentBlock);
        rewriter.create<mlir::cf::BranchOp>(loc, bodyBlock, ValueRange{});

        // cleanup
        auto linuxHasCleanups = cleanupHasOps && !tsContext->compileOptions.isWindows;
        mlir::Block * cleanupBlock = nullptr;
        mlir::Block * cleanupBlockLast = nullptr;
        if (cleanupHasOps)
        {
            rewriter.inlineRegionBefore(tryOp.getCleanup(), continuation);
            cleanupBlock = bodyBlockLast->getNextNode();
            cleanupBlockLast = continuation->getPrevNode();
        }
        else
        {
            while (!tryOp.getCleanup().empty())
            {
                rewriter.eraseBlock(&tryOp.getCleanup().front());
            }
        }

        // catch
        // in case of WIN32 we do not need catch logic if we have only finally
        mlir::Block * catchesBlock = nullptr;
        mlir::Block * catchesBlockLast = nullptr;
        if (catchHasOps)
        {
            rewriter.inlineRegionBefore(tryOp.getCatches(), continuation);
            catchesBlock = (cleanupBlockLast ? cleanupBlockLast : bodyBlockLast)->getNextNode();
            catchesBlockLast = continuation->getPrevNode();        

            // logic to set Invoke attribute CallOp
            auto visitorReturnOpContinue = [&](Operation *op) {
                if (auto returnOp = dyn_cast_or_null<mlir_ts::ReturnOp>(op))
                {
                    tsContext->unwind[op] = catchesBlock;
                }
                else if (auto returnValOp = dyn_cast_or_null<mlir_ts::ReturnValOp>(op))
                {
                    tsContext->unwind[op] = catchesBlock;
                }
            };
            auto it = catchesBlock;
            do
            {
                it->walk(visitorReturnOpContinue);
                if (it != catchesBlockLast)
                {
                    it = it->getNextNode();
                    continue;
                }
            } while (false);
        }
        else
        {
            while (!tryOp.getCatches().empty())
            {
                rewriter.eraseBlock(&tryOp.getCatches().front());
            }
        }

        mlir::Block *finallyBlock = nullptr;
        mlir::Block *finallyBlockLast = nullptr;
        mlir::Block *exitBlockLast = nullptr;
        if (finallyHasOps)
        {
            auto beforeFinallyBlock = continuation->getPrevNode();
            rewriter.cloneRegionBefore(tryOp.getFinally(), continuation);
            finallyBlock = beforeFinallyBlock->getNextNode();
            finallyBlockLast = continuation->getPrevNode();

            // add clone for 'return'
            if (returns.size() > 0)
            {
                rewriter.cloneRegionBefore(tryOp.getFinally(), continuation);
                auto returnFinallyBlockLast = continuation->getPrevNode();
                rewriter.setInsertionPoint(returnFinallyBlockLast->getTerminator());
                auto resultOpOfReturnFinallyBlock = cast<mlir_ts::ResultOp>(returnFinallyBlockLast->getTerminator());
                rewriter.replaceOpWithNewOp<mlir_ts::ReturnOp>(resultOpOfReturnFinallyBlock);
                // if has returns we need to create return cleanup block
                for (auto retOp : returns)
                {
                    tsContext->cleanup[retOp] = returnFinallyBlockLast;                
                }
            }

            rewriter.inlineRegionBefore(tryOp.getFinally(), continuation);
            exitBlockLast = continuation->getPrevNode();            
        }
        else
        {
            while (!tryOp.getFinally().empty())
            {
                rewriter.eraseBlock(&tryOp.getFinally().front());
            }
        }

        auto exitBlock = finallyHasOps ? exitBlockLast : continuation;
        auto landingBlock = cleanupHasOps ? cleanupBlock : catchHasOps ? catchesBlock : finallyBlock;
        tsContext->landingBlockOf[tryOp.getOperation()] = landingBlock;
        if (landingBlock)
        {
            // TODO: check for nested ops for example in if block
            auto visitorCallOpContinue = [&](Operation *op) {
                if (auto callOp = dyn_cast_or_null<mlir_ts::CallOp>(op))
                {
                    tsContext->unwind[op] = landingBlock;
                }
                else if (auto callIndirectOp = dyn_cast_or_null<mlir_ts::CallIndirectOp>(op))
                {
                    tsContext->unwind[op] = landingBlock;
                }
                else if (auto throwOp = dyn_cast_or_null<mlir_ts::ThrowOp>(op))
                {
                    tsContext->unwind[op] = landingBlock;
                }
            };
            auto it = bodyBlock;
            do
            {
                it->walk(visitorCallOpContinue);
                if (it != bodyBlockLast)
                {
                    it = it->getNextNode();
                    continue;
                }
            } while (false);
        }

        // Body:catch vars
        rewriter.setInsertionPointToStart(bodyBlock);
        auto catch1 = rttih.hasType()
                          ? (mlir::Value)rttih.typeInfoPtrValue(loc)
                          : /*catch all*/ (mlir::Value)rewriter.create<mlir_ts::NullOp>(loc, mth.getNullType());

        mlir::Value catchAll;
        if (parentTryOpLandingPad && finallyHasOps || linuxHasCleanups && rttih.hasType())
        {
            catchAll = (mlir::Value)rewriter.create<mlir_ts::NullOp>(loc, mth.getNullType());
        }

        mlir::Value undefArrayValue;
        if (finallyHasOps || cleanupHasOps)
        {
            // BUG: HACK, i need to add marker type to treat it as cleanup landing pad later
            auto arrTy = mth.getConstArrayValueType(mth.getOpaqueType(), 1);
            undefArrayValue = rewriter.create<mlir_ts::UndefOp>(loc, arrTy);
            auto nullVal = rewriter.create<mlir_ts::NullOp>(loc, mth.getNullType());
            // TODO: InsertPropertyOp review using ArrayAttr (should it be I64Array?)

            undefArrayValue = rewriter.create<mlir_ts::InsertPropertyOp>(loc, undefArrayValue.getType(), nullVal,
                                                                         undefArrayValue, MLIRHelper::getStructIndex(rewriter, 0));
        }

        rewriter.setInsertionPoint(bodyBlockLast->getTerminator());
        auto resultOp = cast<mlir_ts::ResultOp>(bodyBlockLast->getTerminator());
        rewriter.replaceOpWithNewOp<mlir::cf::BranchOp>(resultOp, exitBlock, ValueRange{});

        if (cleanupHasOps && tsContext->compileOptions.isWindows)
        {
            rewriter.setInsertionPointToStart(cleanupBlock);

            auto landingPadCleanupOp = rewriter.create<mlir_ts::LandingPadOp>(
                loc, rttih.getLandingPadType(), rewriter.getBoolAttr(true), ValueRange{undefArrayValue});
            auto beginCleanupCallInfo = rewriter.create<mlir_ts::BeginCleanupOp>(loc);

            rewriter.setInsertionPoint(cleanupBlockLast->getTerminator());
            mlir::SmallVector<mlir::Block *> unwindDests;
            unwindDests.push_back(catchesBlock ? catchesBlock : finallyBlock);

            auto resultOpCleanup = cast<mlir_ts::ResultOp>(cleanupBlockLast->getTerminator());
            rewriter.replaceOpWithNewOp<mlir_ts::EndCleanupOp>(resultOpCleanup, landingPadCleanupOp, unwindDests);                
        }

        mlir::Value cmpValue;
        if (catchHasOps)
        {
            // catches:landingpad
            rewriter.setInsertionPointToStart(linuxHasCleanups ? cleanupBlock : catchesBlock);

            SmallVector<mlir::Value> catchTypes{catch1};
            if (linuxHasCleanups && rttih.hasType())
            {
                // we need to catch all exceptions for cleanup code
                catchTypes.push_back(catchAll);
            }

            auto landingPadOp = rewriter.create<mlir_ts::LandingPadOp>(loc, rttih.getLandingPadType(),
                                                                       rewriter.getBoolAttr(false), catchTypes);

            if (!tsContext->compileOptions.isWindows && rttih.hasType())
            {
                if (linuxHasCleanups)
                {
                    // go into catchesBlock
                    rewriter.setInsertionPointToStart(catchesBlock);
                }

                cmpValue = rewriter.create<mlir_ts::CompareCatchTypeOp>(loc, mth.getBooleanType(), landingPadOp,
                                                                        rttih.throwInfoPtrValue(loc));
            }

            // catch: begin catch
            auto beginCatchCallInfo = rewriter.create<mlir_ts::BeginCatchOp>(loc, mth.getOpaqueType(), landingPadOp);
            if (catchOpPtr)
            {
                tsContext->catchOpData[catchOpPtr] = beginCatchCallInfo->getResult(0);
            }

            // catches: end catch
            rewriter.setInsertionPoint(catchesBlockLast->getTerminator());

            rewriter.create<mlir_ts::EndCatchOp>(loc);
        }

        if (finallyHasOps)
        {
            auto linuxHasCleanupsForFinally = !catchHasOps && linuxHasCleanups;

            // point all throw in catch block to clean up block
            auto visitorCallOpContinueCleanup = [&](Operation *op) {
                if (auto callOp = dyn_cast_or_null<mlir_ts::CallOp>(op))
                {
                    tsContext->unwind[op] = finallyBlock;
                }
                else if (auto callIndirectOp = dyn_cast_or_null<mlir_ts::CallIndirectOp>(op))
                {
                    tsContext->unwind[op] = finallyBlock;
                }
                else if (auto throwOp = dyn_cast_or_null<mlir_ts::ThrowOp>(op))
                {
                    tsContext->unwind[op] = finallyBlock;
                }
            };

            if (catchesBlock)
            {
                auto it = catchesBlock;
                do
                {
                    it->walk(visitorCallOpContinueCleanup);
                    if (it != catchesBlockLast)
                    {
                        it = it->getNextNode();
                        continue;
                    }
                } while (false);
            }

            if (tsContext->compileOptions.isWindows)
            {
                rewriter.setInsertionPointToStart(finallyBlock);
    
                auto landingPadCleanupOp = rewriter.create<mlir_ts::LandingPadOp>(
                    loc, rttih.getLandingPadType(), rewriter.getBoolAttr(true), ValueRange{undefArrayValue});
                auto beginCleanupCallInfo = rewriter.create<mlir_ts::BeginCleanupOp>(loc);

                rewriter.setInsertionPoint(finallyBlockLast->getTerminator());
                mlir::SmallVector<mlir::Block *> unwindDests;
                if (parentTryOpLandingPad)
                {
                    unwindDests.push_back(parentTryOpLandingPad);
                }

                auto resultOpFinally = cast<mlir_ts::ResultOp>(finallyBlockLast->getTerminator());
                rewriter.replaceOpWithNewOp<mlir_ts::EndCleanupOp>(resultOpFinally, landingPadCleanupOp, unwindDests);                
            }
            else
            {
                if (!parentTryOpLandingPad)
                {
                    rewriter.setInsertionPointToStart(linuxHasCleanupsForFinally ? cleanupBlock : finallyBlock);

                    auto landingPadCleanupOp = rewriter.create<mlir_ts::LandingPadOp>(
                        loc, rttih.getLandingPadType(), rewriter.getBoolAttr(true), ValueRange{undefArrayValue});

                    if (linuxHasCleanupsForFinally)
                    {
                        rewriter.setInsertionPointToStart(finallyBlock);
                    }                        

                    auto beginCleanupCallInfo = rewriter.create<mlir_ts::BeginCleanupOp>(loc);

                    rewriter.setInsertionPoint(finallyBlockLast->getTerminator());
                    mlir::SmallVector<mlir::Block *> unwindDests;
                    if (parentTryOpLandingPad)
                    {
                        unwindDests.push_back(parentTryOpLandingPad);
                    }

                    if (catchHasOps)
                    {
                        rewriter.setInsertionPoint(finallyBlockLast->getTerminator());
                        rewriter.create<mlir_ts::EndCatchOp>(loc);
                    }

                    auto resultOpFinally = cast<mlir_ts::ResultOp>(finallyBlockLast->getTerminator());
                    rewriter.replaceOpWithNewOp<mlir_ts::EndCleanupOp>(resultOpFinally, landingPadCleanupOp, unwindDests);
                }
                else
                {
                    rewriter.setInsertionPointToStart(linuxHasCleanupsForFinally ? cleanupBlock : finallyBlock);

                    auto landingPadCleanupOp = rewriter.create<mlir_ts::LandingPadOp>(
                        loc, rttih.getLandingPadType(), rewriter.getBoolAttr(false), ValueRange{catchAll});

                    if (linuxHasCleanupsForFinally)
                    {
                        rewriter.setInsertionPointToStart(finallyBlock);
                    }                        

                    auto beginCleanupCallInfo =
                        rewriter.create<mlir_ts::BeginCatchOp>(loc, mth.getOpaqueType(), landingPadCleanupOp);

                    // We do not need EndCatch as throw will redirect execution anyway
                    // rethrow
                    rewriter.setInsertionPoint(finallyBlockLast->getTerminator());
                    auto nullVal = rewriter.create<mlir_ts::NullOp>(loc, mth.getNullType());

                    auto resultOpOfFinally = cast<mlir_ts::ResultOp>(finallyBlockLast->getTerminator());
                    auto throwOp = rewriter.replaceOpWithNewOp<mlir_ts::ThrowOp>(resultOpOfFinally, nullVal);
                    tsContext->unwind[throwOp] = parentTryOpLandingPad;
                }
                // cleanup end
            }
        }

        if (catchHasOps)
        {
            // exit br
            rewriter.setInsertionPoint(catchesBlockLast->getTerminator());

            auto resultOpCatches = cast<mlir_ts::ResultOp>(catchesBlockLast->getTerminator());
            rewriter.replaceOpWithNewOp<mlir::cf::BranchOp>(resultOpCatches, exitBlock, ValueRange{});
        }

        if (cmpValue)
        {
            // condbr
            rewriter.setInsertionPointAfterValue(cmpValue);

            mlir::Block *currentBlockBrCmp = rewriter.getInsertionBlock();
            mlir::Block *continuationBrCmp = rewriter.splitBlock(currentBlockBrCmp, rewriter.getInsertionPoint());

            rewriter.setInsertionPointAfterValue(cmpValue);
            // TODO: when catch not matching - should go into result (rethrow)
            auto castToI1 = rewriter.create<mlir_ts::CastOp>(loc, rewriter.getI1Type(), cmpValue);
            rewriter.create<mlir::cf::CondBranchOp>(loc, castToI1, continuationBrCmp, continuation);
            // end of condbr
        }

        // end of jumps

        if (finallyHasOps)
        {
            // finally:exit
            rewriter.setInsertionPoint(exitBlockLast->getTerminator());

            auto resultOpOfFinallyBlock = cast<mlir_ts::ResultOp>(exitBlockLast->getTerminator());
            rewriter.replaceOpWithNewOp<mlir::cf::BranchOp>(resultOpOfFinallyBlock, continuation, resultOpOfFinallyBlock.getResults());
        }

        if (linuxHasCleanups)
        {
            auto resultOp = cast<mlir_ts::ResultOp>(cleanupBlockLast->getTerminator());
            rewriter.eraseOp(resultOp);

            if (catchHasOps)
            {
                rewriter.mergeBlocks(catchesBlock, cleanupBlockLast);
            }        
            else if (finallyHasOps)
            {
                rewriter.mergeBlocks(finallyBlock, cleanupBlockLast);
            }
        }        

        rewriter.replaceOp(tryOp, continuation->getArguments());

        LLVM_DEBUG(llvm::dbgs() << "\n!! AFTER - TRY_OP DUMP: \n" << *tryOp->getParentOp() << "\n";);

        return success();
    }
};

struct CatchOpLowering : public TsPattern<mlir_ts::CatchOp>
{
    using TsPattern<mlir_ts::CatchOp>::TsPattern;

    LogicalResult matchAndRewrite(mlir_ts::CatchOp catchOp, PatternRewriter &rewriter) const final
    {
        TypeHelper th(rewriter);

        Location loc = catchOp.getLoc();

        auto catchDataValue = tsContext->catchOpData[catchOp];
        if (catchDataValue)
        {
            rewriter.create<mlir_ts::SaveCatchVarOp>(loc, catchDataValue, catchOp.getCatchArg());
        }
        else
        {
            llvm_unreachable("missing catch data.");
        }

        rewriter.eraseOp(catchOp);

        return success();
    }
};

struct CallOpLowering : public TsPattern<mlir_ts::CallOp>
{
    using TsPattern<mlir_ts::CallOp>::TsPattern;

    LogicalResult matchAndRewrite(mlir_ts::CallOp op, PatternRewriter &rewriter) const final
    {
        if (auto unwind = tsContext->unwind[op])
        {
            {
                OpBuilder::InsertionGuard guard(rewriter);
                CodeLogicHelper clh(op, rewriter);
                auto *continuationBlock = clh.CutBlockAndSetInsertPointToEndOfBlock();

                LLVM_DEBUG(llvm::dbgs() << "!! ...call -> invoke: " << op.getCalleeAttr() << "\n";);
                LLVM_DEBUG(for (auto opit
                                : op.getOperands()) llvm::dbgs()
                               << "!! ...call -> invoke operands: " << opit << "\n";);

                rewriter.replaceOpWithNewOp<mlir_ts::InvokeOp>(op, op.getResultTypes(), op.getCalleeAttr(),
                                                               op.getArgOperands(), continuationBlock, ValueRange{},
                                                               unwind, ValueRange{});
                return success();
            }

            rewriter.eraseOp(op);
            return success();
        }

        // just replace
        rewriter.replaceOpWithNewOp<mlir_ts::SymbolCallInternalOp>(op, op.getResultTypes(), op.getCalleeAttr(),
                                                                   op.getArgOperands());
        return success();
    }
};

struct CallIndirectOpLowering : public TsPattern<mlir_ts::CallIndirectOp>
{
    using TsPattern<mlir_ts::CallIndirectOp>::TsPattern;

    LogicalResult matchAndRewrite(mlir_ts::CallIndirectOp op, PatternRewriter &rewriter) const final
    {
        if (auto unwind = tsContext->unwind[op])
        {
            {
                OpBuilder::InsertionGuard guard(rewriter);
                CodeLogicHelper clh(op, rewriter);
                auto *continuationBlock = clh.CutBlockAndSetInsertPointToEndOfBlock();

                LLVM_DEBUG(llvm::dbgs() << "!! ...! call indirect:\n";);
                LLVM_DEBUG(for (auto opit
                                : op.getOperands()) llvm::dbgs()
                               << "!! ...call -> invoke operands: " << opit << "\n";);

                auto res = rewriter.replaceOpWithNewOp<mlir_ts::InvokeOp>(
                    op, op.getResultTypes(), op.getOperands(), continuationBlock, ValueRange{}, unwind, ValueRange{});
                return success();
            }

            rewriter.eraseOp(op);
            return success();
        }

        // just replace
        rewriter.replaceOpWithNewOp<mlir_ts::CallInternalOp>(op, op.getResultTypes(), op.getOperands());
        return success();
    }
};

struct ThrowOpLowering : public TsPattern<mlir_ts::ThrowOp>
{
    using TsPattern<mlir_ts::ThrowOp>::TsPattern;

    LogicalResult matchAndRewrite(mlir_ts::ThrowOp throwOp, PatternRewriter &rewriter) const final
    {
        // TODO: add it to CallOp, CallIndirectOp
        CodeLogicHelper clh(throwOp, rewriter);

        Location loc = throwOp.getLoc();

        if (auto unwind = tsContext->unwind[throwOp])
        {
            rewriter.replaceOpWithNewOp<mlir_ts::ThrowUnwindOp>(throwOp, throwOp.getException(), unwind);
        }
        else
        {
            rewriter.replaceOpWithNewOp<mlir_ts::ThrowCallOp>(throwOp, throwOp.getException());
        }

        clh.CutBlock();

        return success();
    }
};

struct StateLabelOpLowering : public TsPattern<mlir_ts::StateLabelOp>
{
    using TsPattern<mlir_ts::StateLabelOp>::TsPattern;

    LogicalResult matchAndRewrite(mlir_ts::StateLabelOp op, PatternRewriter &rewriter) const final
    {
        mlir::Location loc = op.getLoc();

        CodeLogicHelper clh(op, rewriter);

        auto *continueBlock = clh.BeginBlock(loc);

        tsFuncContext->stateLabels.push_back(continueBlock);

        rewriter.eraseOp(op);

        return success();
    }
};

class SwitchStateOpLowering : public TsPattern<mlir_ts::SwitchStateOp>
{
  public:
    using TsPattern<mlir_ts::SwitchStateOp>::TsPattern;

    LogicalResult matchAndRewrite(mlir_ts::SwitchStateOp switchStateOp, PatternRewriter &rewriter) const final
    {
        CodeLogicHelper clh(switchStateOp, rewriter);

        auto loc = switchStateOp->getLoc();

        assert(tsContext->returnBlock);

        auto retBlock = tsContext->returnBlock;

        auto returnBlock = retBlock;

        assert(returnBlock);

        LLVM_DEBUG(llvm::dbgs() << "\n!! return block: "; returnBlock->dump(); llvm::dbgs() << "\n";);

        auto defaultBlock = returnBlock;

        assert(defaultBlock != nullptr);

        SmallVector<mlir::Block *> caseDestinations;

        SmallPtrSet<Operation *, 16> stateLabels;

        // select all states
        auto visitorAllStateLabels = [&](Operation *op) {
            if (auto stateLabelOp = dyn_cast_or_null<mlir_ts::StateLabelOp>(op))
            {
                stateLabels.insert(op);
            }
        };

        switchStateOp->getParentOp()->walk(visitorAllStateLabels);

        {
            mlir::OpBuilder::InsertionGuard insertGuard(rewriter);
            for (auto op : stateLabels)
            {
                auto stateLabelOp = dyn_cast_or_null<mlir_ts::StateLabelOp>(op);
                rewriter.setInsertionPoint(stateLabelOp);

                auto *continuationBlock = clh.BeginBlock(loc);

                rewriter.eraseOp(stateLabelOp);

                // add switch
                caseDestinations.push_back(continuationBlock);
            }
        }

        // insert 0 state label
        caseDestinations.insert(caseDestinations.begin(), switchStateOp.getDefaultDest());

        rewriter.replaceOpWithNewOp<mlir_ts::SwitchStateInternalOp>(
            switchStateOp, switchStateOp.getState(), defaultBlock ? defaultBlock : switchStateOp.getDefaultDest(),
            caseDestinations);

        return success();
    }
};

struct YieldReturnValOpLowering : public TsPattern<mlir_ts::YieldReturnValOp>
{
    using TsPattern<mlir_ts::YieldReturnValOp>::TsPattern;

    LogicalResult matchAndRewrite(mlir_ts::YieldReturnValOp yieldReturnValOp, PatternRewriter &rewriter) const final
    {
        CodeLogicHelper clh(yieldReturnValOp, rewriter);

        auto loc = yieldReturnValOp->getLoc();

        assert(tsContext->returnBlock);

        auto retBlock = tsContext->returnBlock;

        rewriter.replaceOpWithNewOp<mlir_ts::StoreOp>(yieldReturnValOp, yieldReturnValOp.getOperand(),
                                                      yieldReturnValOp.getReference());

        rewriter.setInsertionPointAfter(yieldReturnValOp);
        clh.JumpTo(yieldReturnValOp.getLoc(), retBlock);

        return success();
    }
};

struct TypeOfOpLowering : public TsPattern<mlir_ts::TypeOfOp>
{
    using TsPattern<mlir_ts::TypeOfOp>::TsPattern;

    LogicalResult matchAndRewrite(mlir_ts::TypeOfOp typeOfOp, PatternRewriter &rewriter) const final
    {
        TypeOfOpHelper toh(rewriter);
        auto typeOfValue = toh.typeOfLogic(typeOfOp->getLoc(), typeOfOp.getValue(), typeOfOp.getValue().getType());

        rewriter.replaceOp(typeOfOp, ValueRange{typeOfValue});
        return success();
    }
};

struct CaptureOpLowering : public TsPattern<mlir_ts::CaptureOp>
{
    using TsPattern<mlir_ts::CaptureOp>::TsPattern;

    LogicalResult matchAndRewrite(mlir_ts::CaptureOp captureOp, PatternRewriter &rewriter) const final
    {
        auto location = captureOp->getLoc();

        TypeHelper th(rewriter);

        auto captureRefType = captureOp.getType();

        LLVM_DEBUG(llvm::dbgs() << "\n!! ...capture result type: " << captureRefType << "\n\n";);

        assert(captureRefType.isa<mlir_ts::RefType>());
        auto captureStoreType = captureRefType.cast<mlir_ts::RefType>().getElementType().cast<mlir_ts::TupleType>();

        LLVM_DEBUG(llvm::dbgs() << "\n!! ...capture store type: " << captureStoreType << "\n\n";);

        // true => we need to allocate capture in heap memory
#ifdef ALLOC_CAPTURE_IN_HEAP
        auto inHeapMemory = true;
#else
        auto inHeapMemory = false;
#endif
        mlir::Value allocTempStorage = rewriter.create<mlir_ts::VariableOp>(
            location, captureRefType, mlir::Value(), rewriter.getBoolAttr(inHeapMemory), rewriter.getIndexAttr(0));

        for (auto [index, val] : enumerate(captureOp.getCaptured()))
        {
            auto thisStoreFieldType = captureStoreType.getType(index);
            auto thisStoreFieldTypeRef = mlir_ts::RefType::get(thisStoreFieldType);
            auto fieldRef = rewriter.create<mlir_ts::PropertyRefOp>(location, thisStoreFieldTypeRef, allocTempStorage,
                                                                    th.getStructIndexAttrValue(index));

            LLVM_DEBUG(llvm::dbgs() << "\n!! ...storing val: [" << val << "] in (" << index << ") ref: " << fieldRef
                                    << "\n\n";);

            // dereference value in case of sending value by ref but stored as value
            // TODO: review capture logic
            if (auto valRefType = val.getType().dyn_cast<mlir_ts::RefType>())
            {
                if (!thisStoreFieldType.isa<mlir_ts::RefType>() && thisStoreFieldType == valRefType.getElementType())
                {
                    // load value to dereference
                    val = rewriter.create<mlir_ts::LoadOp>(location, valRefType.getElementType(), val);
                }
            }

            assert(val.getType() == fieldRef.getType().cast<mlir_ts::RefType>().getElementType());

            rewriter.create<mlir_ts::StoreOp>(location, val, fieldRef);
        }

        rewriter.replaceOp(captureOp, allocTempStorage);

        return success();
    }
};

} // end anonymous namespace

//===----------------------------------------------------------------------===//
// TypeScriptToAffineLoweringTSFuncPass
//===----------------------------------------------------------------------===//

/// This is a partial lowering to affine loops of the typescript operations that are
/// computationally intensive (like add+mul for example...) while keeping the
/// rest of the code in the TypeScript dialect.

namespace
{
struct TypeScriptToAffineLoweringTSFuncPass : public PassWrapper<TypeScriptToAffineLoweringTSFuncPass, TypeScriptFunctionPass>
{
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TypeScriptToAffineLoweringTSFuncPass)

    TSContext tsContext;

    TypeScriptToAffineLoweringTSFuncPass(CompileOptions &compileOptions) : tsContext(compileOptions)
    {
    }

    void getDependentDialects(DialectRegistry &registry) const override
    {
        registry.insert<arith::ArithDialect>();
        registry.insert<cf::ControlFlowDialect>();
        registry.insert<func::FuncDialect>();
    }

    void runOnFunction() final;
};
} // end anonymous namespace.

namespace
{
struct TypeScriptToAffineLoweringFuncPass : public PassWrapper<TypeScriptToAffineLoweringFuncPass, OperationPass<func::FuncOp>>
{
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TypeScriptToAffineLoweringFuncPass)

    TSContext tsContext;

    TypeScriptToAffineLoweringFuncPass(CompileOptions &compileOptions) : tsContext(compileOptions)
    {
    }

    void getDependentDialects(DialectRegistry &registry) const override
    {
        registry.insert<arith::ArithDialect>();
        registry.insert<cf::ControlFlowDialect>();
        registry.insert<func::FuncDialect>();
    }

    void runOnOperation() final;

};
} // end anonymous namespace.

namespace
{
struct TypeScriptToAffineLoweringModulePass : public PassWrapper<TypeScriptToAffineLoweringModulePass, OperationPass<ModuleOp>>
{
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TypeScriptToAffineLoweringModulePass)

    TSContext tsContext;

    TypeScriptToAffineLoweringModulePass(CompileOptions &compileOptions) : tsContext(compileOptions)
    {
    }

    void getDependentDialects(DialectRegistry &registry) const override
    {
        registry.insert<arith::ArithDialect>();
        registry.insert<cf::ControlFlowDialect>();
        registry.insert<func::FuncDialect>();
    }

    void runOnOperation() final;
};
} // end anonymous namespace.

static LogicalResult verifySuccessors(Operation *op)
{
    auto *parent = op->getParentRegion();

    // Verify that the operands lines up with the BB arguments in the successor.
    for (mlir::Block *succ : op->getSuccessors())
        if (succ->getParent() != parent)
        {
            LLVM_DEBUG(llvm::dbgs() << "\n!! reference to block defined in another region: "; op->dump();
                       llvm::dbgs() << "\n";);
            assert(false);
            return op->emitError("!! DEBUG TEST: reference to block defined in another region");
        }

    return success();
}

static LogicalResult verifyFunction(mlir::typescript::FuncOp &funcOp)
{
    for (auto &region : funcOp->getRegions())
    {
        for (auto &regionBlock : region)
        {
            for (auto &op : regionBlock)
            {
                if (failed(verifySuccessors(&op)))
                {
                    return failure();
                }
            }
        }
    }

    return success();
}

void finishSwitchState(mlir_ts::FuncOp f, TSFunctionContext &tsFuncContext)
{
    // change SwitchStateOp
    if (tsFuncContext.stateLabels.size() == 0)
    {
        return;
    }

    ConversionPatternRewriter rewriter(f.getContext());
    CodeLogicHelper clh(f, rewriter);
    auto switchStateOp = clh.FindOp<mlir_ts::SwitchStateOp>(f);
    assert(switchStateOp);

    mlir::SmallVector<mlir::Block *> stateLabels(tsFuncContext.stateLabels);

    rewriter.replaceOpWithNewOp<mlir_ts::SwitchStateInternalOp>(switchStateOp, switchStateOp.getState(),
                                                                switchStateOp.getDefaultDest(), stateLabels);
}

void cleanupEmptyBlocksWithoutPredecessors(mlir_ts::FuncOp f)
{
    auto any = false;
    do
    {
        any = false;
        SmallPtrSet<mlir::Block *, 16> workSet;
        for (auto &regionBlock : f.getRegion())
        {
            if (regionBlock.isEntryBlock())
            {
                continue;
            }

            if (regionBlock.getPredecessors().empty())
            {
                auto count = std::distance(regionBlock.begin(), regionBlock.end());
                if (count == 0 || count == 1 && (isa<mlir::cf::BranchOp>(regionBlock.begin()) ||
                                                 isa<mlir_ts::UnreachableOp>(regionBlock.begin())))
                {
                    workSet.insert(&regionBlock);
                }
            }
        }

        ConversionPatternRewriter rewriter(f.getContext());
        for (auto blockPtr : workSet)
        {
            blockPtr->dropAllDefinedValueUses();
            blockPtr->dropAllUses();
            blockPtr->dropAllReferences();
            rewriter.eraseBlock(blockPtr);
            any = true;
        }
    } while (any);
}

void AddTsAffineLegalOps(ConversionTarget &target)
{
    // We define the specific operations, or dialects, that are legal targets for
    // this lowering. In our case, we are lowering to a combination of the
    // `Affine` and `Standard` dialects.
    target.addLegalDialect<arith::ArithDialect>();
    target.addLegalDialect<cf::ControlFlowDialect>();
    target.addLegalDialect<func::FuncDialect>();

    // We also define the TypeScript dialect as Illegal so that the conversion will fail
    // if any of these operations are *not* converted. Given that we actually want
    // a partial lowering, we explicitly mark the TypeScript operations that don't want
    // to lower, `typescript.print`, as `legal`.
    target.addIllegalDialect<mlir_ts::TypeScriptDialect>();
    target.addLegalOp<
        mlir_ts::AddressOfOp, mlir_ts::AddressOfConstStringOp, mlir_ts::AddressOfElementOp, mlir_ts::ArithmeticBinaryOp,
        mlir_ts::ArithmeticUnaryOp, mlir_ts::AssertOp, mlir_ts::CastOp, mlir_ts::ConstantOp, mlir_ts::ElementRefOp,
        mlir_ts::PointerOffsetRefOp, mlir_ts::FuncOp, mlir_ts::GlobalOp, mlir_ts::GlobalResultOp, mlir_ts::HasValueOp,
        mlir_ts::ValueOp, mlir_ts::ValueOrDefaultOp, mlir_ts::NullOp, mlir_ts::ParseFloatOp, mlir_ts::ParseIntOp, mlir_ts::IsNaNOp,
        mlir_ts::PrintOp, mlir_ts::ConvertFOp, mlir_ts::SizeOfOp, mlir_ts::StoreOp, mlir_ts::SymbolRefOp, mlir_ts::LengthOfOp, mlir_ts::SetLengthOfOp,
        mlir_ts::StringLengthOp, mlir_ts::SetStringLengthOp, mlir_ts::StringConcatOp, mlir_ts::StringCompareOp, mlir_ts::LoadOp, mlir_ts::NewOp,
        mlir_ts::CreateTupleOp, mlir_ts::DeconstructTupleOp, mlir_ts::CreateArrayOp, mlir_ts::NewEmptyArrayOp,
        mlir_ts::NewArrayOp, mlir_ts::DeleteOp, mlir_ts::PropertyRefOp, mlir_ts::InsertPropertyOp, mlir_ts::ExtractPropertyOp, 
        mlir_ts::LogicalBinaryOp, mlir_ts::UndefOp, mlir_ts::VariableOp, mlir_ts::DebugVariableOp, mlir_ts::AllocaOp, mlir_ts::InvokeOp, 
        /*mlir_ts::ResultOp,*/ mlir_ts::VirtualSymbolRefOp, mlir_ts::ThisVirtualSymbolRefOp, mlir_ts::InterfaceSymbolRefOp, 
        mlir_ts::ExtractInterfaceThisOp, mlir_ts::ExtractInterfaceVTableOp, mlir_ts::ArrayPushOp, mlir_ts::ArrayPopOp, 
        mlir_ts::ArrayUnshiftOp, mlir_ts::ArrayShiftOp, mlir_ts::ArraySpliceOp, mlir_ts::ArrayViewOp, 
        mlir_ts::NewInterfaceOp, mlir_ts::VTableOffsetRefOp, mlir_ts::GetThisOp, mlir_ts::GetMethodOp, mlir_ts::DebuggerOp,
        mlir_ts::LandingPadOp, mlir_ts::CompareCatchTypeOp, mlir_ts::BeginCatchOp, mlir_ts::SaveCatchVarOp,
        mlir_ts::EndCatchOp, mlir_ts::BeginCleanupOp, mlir_ts::EndCleanupOp, mlir_ts::ThrowUnwindOp,
        mlir_ts::ThrowCallOp, mlir_ts::SymbolCallInternalOp, mlir_ts::CallInternalOp, mlir_ts::ReturnInternalOp,
        mlir_ts::NoOp, mlir_ts::SwitchStateInternalOp, mlir_ts::UnreachableOp, mlir_ts::GlobalConstructorOp,
        mlir_ts::CreateBoundFunctionOp, mlir_ts::TypeOfAnyOp, mlir_ts::BoxOp, mlir_ts::UnboxOp,
        mlir_ts::CreateUnionInstanceOp, mlir_ts::GetValueFromUnionOp, mlir_ts::GetTypeInfoFromUnionOp,
        mlir_ts::OptionalOp, mlir_ts::OptionalValueOp, mlir_ts::OptionalUndefOp,
        mlir_ts::LoadLibraryPermanentlyOp, mlir_ts::SearchForAddressOfSymbolOp>();
#ifdef ENABLE_TYPED_GC
    target.addLegalOp<
        mlir_ts::GCMakeDescriptorOp, GCNewExplicitlyTypedOp>();
#endif

}

void AddTsAffinePatterns(MLIRContext &context, ConversionTarget &target, RewritePatternSet &patterns,
                         TSContext &tsContext, TSFunctionContext &tsFuncContext)
{
    // Now that the conversion target has been defined, we just need to provide
    // the set of patterns that will lower the TypeScript operations.

    patterns.insert<EntryOpLowering, ExitOpLowering, ReturnOpLowering, ReturnValOpLowering, ParamOpLowering,
                    ParamOptionalOpLowering, ParamDefaultValueOpLowering, OptionalValueOrDefaultOpLowering, PrefixUnaryOpLowering, 
                    PostfixUnaryOpLowering, IfOpLowering, /*ResultOpLowering,*/
                    DoWhileOpLowering, WhileOpLowering, ForOpLowering, BreakOpLowering, ContinueOpLowering,
                    SwitchOpLowering, AccessorOpLowering, ThisAccessorOpLowering, LabelOpLowering, CallOpLowering,
                    CallIndirectOpLowering, TryOpLowering, ThrowOpLowering, CatchOpLowering, StateLabelOpLowering,
                    SwitchStateOpLowering, YieldReturnValOpLowering, TypeOfOpLowering, CaptureOpLowering>(
        &context, &tsContext, &tsFuncContext);
}

void TypeScriptToAffineLoweringTSFuncPass::runOnFunction()
{
    auto function = getFunction();

    LLVM_DEBUG(llvm::dbgs() << "\n!! BEFORE FUNC DUMP: \n" << function << "\n";);

    // We only lower the main function as we expect that all other functions have been inlined.
    if (tsContext.compileOptions.isJit && function.getName() == "main")
    {
        auto voidType = mlir_ts::VoidType::get(function.getContext());
        // Verify that the given main has no inputs and results.
        if (function.getNumArguments() ||
            llvm::any_of(function.getFunctionType().getResults(), [&](mlir::Type type) { return type != voidType; }))
        {
            function.emitError("expected 'main' to have 0 inputs and 0 results");
            return signalPassFailure();
        }
    }

    // The first thing to define is the conversion target. This will define the
    // final target for this lowering.
    ConversionTarget target(getContext());
    RewritePatternSet patterns(&getContext());

    TSFunctionContext tsFuncContext{};
    AddTsAffineLegalOps(target);
    AddTsAffinePatterns(getContext(), target, patterns, tsContext, tsFuncContext);

    // With the target and rewrite patterns defined, we can now attempt the
    // conversion. The conversion will signal failure if any of our `illegal`
    // operations were not converted successfully.
    if (failed(applyPartialConversion(function, target, std::move(patterns))))
    {
        signalPassFailure();
    }

    LLVM_DEBUG(llvm::dbgs() << "\n!! Processing function: \n" << function.getName() << "\n";);

    cleanupEmptyBlocksWithoutPredecessors(function);

    LLVM_DEBUG(llvm::dbgs() << "\n!! AFTER FUNC DUMP: \n" << function << "\n";);

    LLVM_DEBUG(verifyFunction(function););
}

void TypeScriptToAffineLoweringFuncPass::runOnOperation()
{
    auto function = getOperation();

    // The first thing to define is the conversion target. This will define the
    // final target for this lowering.
    ConversionTarget target(getContext());
    RewritePatternSet patterns(&getContext());

    TSFunctionContext tsFuncContext{};
    AddTsAffineLegalOps(target);
    AddTsAffinePatterns(getContext(), target, patterns, tsContext, tsFuncContext);

    // TODO: Hack to fix issue with Async
    target.addLegalOp<mlir::func::FuncOp>();

    // With the target and rewrite patterns defined, we can now attempt the
    // conversion. The conversion will signal failure if any of our `illegal`
    // operations were not converted successfully.
    if (failed(applyPartialConversion(function, target, std::move(patterns))))
    {
        signalPassFailure();
    }

    LLVM_DEBUG(llvm::dbgs() << "\n!! (FUNC) AFTER FUNC DUMP: \n" << function << "\n";);
}

void TypeScriptToAffineLoweringModulePass::runOnOperation()
{
    auto module = getOperation();

    // The first thing to define is the conversion target. This will define the
    // final target for this lowering.
    ConversionTarget target(getContext());
    RewritePatternSet patterns(&getContext());

    TSFunctionContext tsFuncContext{};
    AddTsAffineLegalOps(target);
    AddTsAffinePatterns(getContext(), target, patterns, tsContext, tsFuncContext);

    // + Global ops
    target.addLegalOp<ModuleOp>();    

    // TODO: Hack to fix issue with Async
    target.addLegalOp<mlir::func::FuncOp>();    
    target.addLegalDialect<mlir::async::AsyncDialect>();

    // With the target and rewrite patterns defined, we can now attempt the
    // conversion. The conversion will signal failure if any of our `illegal`
    // operations were not converted successfully.
    if (failed(applyFullConversion(module, target, std::move(patterns))))
    {
        signalPassFailure();
    }    
}

/// Create a pass for lowering operations in the `Affine` and `Std` dialects,
/// for a subset of the TypeScript IR.
std::unique_ptr<mlir::Pass> mlir_ts::createLowerToAffineTSFuncPass(CompileOptions &compileOptions)
{
    return std::make_unique<TypeScriptToAffineLoweringTSFuncPass>(compileOptions);
}

std::unique_ptr<mlir::Pass> mlir_ts::createLowerToAffineFuncPass(CompileOptions &compileOptions)
{
    return std::make_unique<TypeScriptToAffineLoweringFuncPass>(compileOptions);
}

std::unique_ptr<mlir::Pass> mlir_ts::createLowerToAffineModulePass(CompileOptions &compileOptions)
{
    return std::make_unique<TypeScriptToAffineLoweringModulePass>(compileOptions);
}
