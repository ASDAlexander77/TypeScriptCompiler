#ifndef MLIR_TYPESCRIPT_LOWERTOLLVMLOGIC_CODELOGICHELPER_H_
#define MLIR_TYPESCRIPT_LOWERTOLLVMLOGIC_CODELOGICHELPER_H_

#include "TypeScript/Config.h"
#include "TypeScript/Defines.h"
#include "TypeScript/Passes.h"
#include "TypeScript/TypeScriptDialect.h"
#include "TypeScript/TypeScriptOps.h"

#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "TypeScript/LowerToLLVM/TypeHelper.h"

#include "mlir/IR/PatternMatch.h"

using namespace mlir;
namespace mlir_ts = mlir::typescript;

namespace typescript
{

class CodeLogicHelper
{
    Location loc;
    PatternRewriter &rewriter;

  public:
    CodeLogicHelper(Operation *op, PatternRewriter &rewriter) : loc(op->getLoc()), rewriter(rewriter)
    {
    }

    CodeLogicHelper(Location loc, PatternRewriter &rewriter) : loc(loc), rewriter(rewriter)
    {
    }

    mlir::Value createIConstantOf(unsigned width, unsigned value)
    {
        return rewriter.create<LLVM::ConstantOp>(loc, rewriter.getIntegerType(width),
                                                 rewriter.getIntegerAttr(rewriter.getIntegerType(width), value));
    }

    mlir::Value createFConstantOf(unsigned width, double value)
    {
#ifdef NUMBER_F64
        auto ftype = rewriter.getF64Type();
#else
        auto ftype = rewriter.getF32Type();
#endif
        if (width == 16)
        {
            ftype = rewriter.getF16Type();
        }
        else if (width == 64)
        {
            ftype = rewriter.getF64Type();
        }
        else if (width == 128)
        {
            ftype = rewriter.getF128Type();
        }

        return rewriter.create<LLVM::ConstantOp>(loc, ftype, rewriter.getFloatAttr(ftype, value));
    }

    mlir::Value createI8ConstantOf(int8_t value)
    {
        return rewriter.create<LLVM::ConstantOp>(loc, rewriter.getIntegerType(8),
                                                 rewriter.getIntegerAttr(rewriter.getIntegerType(8), value));
    }

    mlir::Value createI32ConstantOf(int32_t value)
    {
        return rewriter.create<LLVM::ConstantOp>(loc, rewriter.getIntegerType(32), rewriter.getIntegerAttr(rewriter.getI32Type(), value));
    }

    mlir::Value createI64ConstantOf(int64_t value)
    {
        return rewriter.create<LLVM::ConstantOp>(loc, rewriter.getIntegerType(64), rewriter.getIntegerAttr(rewriter.getI64Type(), value));
    }

    mlir::Value createI1ConstantOf(bool value)
    {
        return rewriter.create<LLVM::ConstantOp>(loc, rewriter.getIntegerType(1), rewriter.getIntegerAttr(rewriter.getI1Type(), value));
    }

    mlir::Value createF32ConstantOf(float value)
    {
        return rewriter.create<LLVM::ConstantOp>(loc, rewriter.getF32Type(), rewriter.getF32FloatAttr(value));
    }

    mlir::Value createF64ConstantOf(double value)
    {
        return rewriter.create<LLVM::ConstantOp>(loc, rewriter.getF64Type(), rewriter.getF64FloatAttr(value));
    }

    mlir::Value createIndexConstantOf(mlir::Type llvmIndexType, int64_t value)
    {
        return rewriter.create<LLVM::ConstantOp>(loc, llvmIndexType, rewriter.getIndexAttr(value));
    }

    mlir::Value createStructIndexConstantOf(int32_t value)
    {
        return rewriter.create<LLVM::ConstantOp>(loc, rewriter.getIntegerType(32), rewriter.getIntegerAttr(rewriter.getI32Type(), value));
    }

    mlir::Value castToI8Ptr(mlir::Value value)
    {
        TypeHelper th(rewriter);
        return rewriter.create<LLVM::BitcastOp>(loc, th.getI8PtrType(), value);
    }

    mlir::Value castToI8PtrPtr(mlir::Value value)
    {
        TypeHelper th(rewriter);
        return rewriter.create<LLVM::BitcastOp>(loc, th.getI8PtrPtrType(), value);
    }

    mlir::Value conditionalExpressionLowering(mlir::Location loc, mlir::Type type, mlir::Value condition,
                                              mlir::function_ref<mlir::Value(OpBuilder &, Location)> thenBuilder,
                                              mlir::function_ref<mlir::Value(OpBuilder &, Location)> elseBuilder)
    {
        // Split block
        auto *opBlock = rewriter.getInsertionBlock();
        auto opPosition = rewriter.getInsertionPoint();
        auto *continuationBlock = rewriter.splitBlock(opBlock, opPosition);

        // then block
        auto *thenBlock = rewriter.createBlock(continuationBlock);
        auto thenValue = thenBuilder(rewriter, loc);

        // else block
        auto *elseBlock = rewriter.createBlock(continuationBlock);
        auto elseValue = elseBuilder(rewriter, loc);

        // result block
        auto *resultBlock = rewriter.createBlock(continuationBlock, TypeRange{type}, {loc});
        rewriter.create<LLVM::BrOp>(loc, ValueRange{}, continuationBlock);

        rewriter.setInsertionPointToEnd(thenBlock);
        rewriter.create<LLVM::BrOp>(loc, ValueRange{thenValue}, resultBlock);

        rewriter.setInsertionPointToEnd(elseBlock);
        rewriter.create<LLVM::BrOp>(loc, ValueRange{elseValue}, resultBlock);

        // Generate assertion test.
        rewriter.setInsertionPointToEnd(opBlock);
        rewriter.create<LLVM::CondBrOp>(loc, condition, thenBlock, elseBlock);

        rewriter.setInsertionPointToStart(continuationBlock);

        return resultBlock->getArguments().front();
    }

    mlir::LogicalResult conditionalBlocksLowering(mlir::Value condition, mlir::function_ref<ValueRange(OpBuilder &, Location)> thenBuilder,
                                                  mlir::function_ref<ValueRange(OpBuilder &, Location)> elseBuilder)
    {
        // Split block
        auto *opBlock = rewriter.getInsertionBlock();
        auto opPosition = rewriter.getInsertionPoint();
        auto *continuationBlock = rewriter.splitBlock(opBlock, opPosition);

        // then block
        auto *thenBlock = rewriter.createBlock(continuationBlock);

        // else block
        auto *elseBlock = rewriter.createBlock(continuationBlock);

        // result block
        auto *resultBlock = rewriter.createBlock(continuationBlock, TypeRange{});
        rewriter.create<LLVM::BrOp>(loc, ValueRange{}, continuationBlock);

        rewriter.setInsertionPointToStart(thenBlock);
        /*auto thenValues = */ thenBuilder(rewriter, loc);
        rewriter.create<LLVM::BrOp>(loc, ValueRange{}, resultBlock);

        rewriter.setInsertionPointToStart(elseBlock);
        /*auto elseValues = */ elseBuilder(rewriter, loc);
        rewriter.create<LLVM::BrOp>(loc, ValueRange{}, resultBlock);

        // Generate condition test.
        rewriter.setInsertionPointToEnd(opBlock);
        rewriter.create<LLVM::CondBrOp>(loc, condition, thenBlock, elseBlock);

        rewriter.setInsertionPointToStart(continuationBlock);

        return mlir::success();
    }

    ValueRange conditionalBlocksLowering(TypeRange types, mlir::Value condition,
                                         mlir::function_ref<ValueRange(OpBuilder &, Location)> thenBuilder,
                                         mlir::function_ref<ValueRange(OpBuilder &, Location)> elseBuilder)
    {
        auto conditionAsI1 = condition;
        if (condition.getType() != rewriter.getI1Type())
        {
            conditionAsI1 = 
                rewriter.create<mlir_ts::DialectCastOp>(loc, rewriter.getI1Type(), condition);
        }

        if (types.size() == 0)
        {
            conditionalBlocksLowering(conditionAsI1, thenBuilder, elseBuilder);
            return ValueRange{};
        }

        // Split block
        auto *opBlock = rewriter.getInsertionBlock();
        auto opPosition = rewriter.getInsertionPoint();
        auto *continuationBlock = rewriter.splitBlock(opBlock, opPosition);

        // then block
        auto *thenBlock = rewriter.createBlock(continuationBlock);

        // else block
        auto *elseBlock = rewriter.createBlock(continuationBlock);

        // result block
        auto *resultBlock = rewriter.createBlock(continuationBlock, types, SmallVector<Location>(types.size(), loc));
        rewriter.create<LLVM::BrOp>(loc, ValueRange{}, continuationBlock);

        rewriter.setInsertionPointToStart(thenBlock);
        auto thenValues = thenBuilder(rewriter, loc);
        rewriter.create<LLVM::BrOp>(loc, thenValues, resultBlock);

        rewriter.setInsertionPointToStart(elseBlock);
        auto elseValues = elseBuilder(rewriter, loc);
        rewriter.create<LLVM::BrOp>(loc, elseValues, resultBlock);

        // Generate condition test.
        rewriter.setInsertionPointToEnd(opBlock);
        rewriter.create<LLVM::CondBrOp>(loc, conditionAsI1, thenBlock, elseBlock);

        rewriter.setInsertionPointToStart(continuationBlock);

        return resultBlock->getArguments();
    }

    template <typename OpTy> void saveResult(OpTy &op, mlir::Value result)
    {
        auto defOp = op.getOperand1().getDefiningOp();
        // TODO: finish it for field access
        if (auto loadOp = dyn_cast<mlir_ts::LoadOp>(defOp))
        {
            rewriter.create<mlir_ts::StoreOp>(loc, result, loadOp.getReference());
        }
        else
        {
            llvm_unreachable("not implemented");
        }
    }

    mlir::Block *FindReturnBlock(bool createReturnBlock = false)
    {
        auto *region = rewriter.getInsertionBlock()->getParent();
        if (!region)
        {
            return nullptr;
        }

        mlir::Block *newReturnBlock = nullptr;

        auto result = std::find_if(region->begin(), region->end(), [&](auto &block) {
            if (block.empty())
            {
                return false;
            }

            auto *op = block.getTerminator();
            // auto name = op->getName().getStringRef();
            auto returnInternalOp = dyn_cast<mlir_ts::ReturnInternalOp>(op);
            auto isReturn = !!returnInternalOp;
            if (isReturn && op != &block.front())
            {
                if (createReturnBlock)
                {
                    if (returnInternalOp.getOperands().size() > 0)
                    {
                        auto argOp = returnInternalOp.getOperands().front().getDefiningOp();
                        if (argOp == &block.front())
                        {
                            // no need to create what already created
                            return true;
                        }

                        rewriter.setInsertionPoint(argOp);
                    }
                    else
                    {
                        rewriter.setInsertionPoint(op);
                    }

                    CodeLogicHelper clh(op, rewriter);
                    auto *contBlock = clh.BeginBlock(op->getLoc());

                    newReturnBlock = contBlock;
                    return true;
                }

                // llvm_unreachable("return must be only operator in block");
            }

            return isReturn;
        });

        if (newReturnBlock)
        {
            return newReturnBlock;
        }

        if (result == region->end())
        {
            llvm_unreachable("return op. can't be found");
            return nullptr;
        }

        return &*result;
    }

    mlir::Block *FindUnreachableBlockOrCreate()
    {
        auto *region = rewriter.getInsertionBlock()->getParent();
        if (!region)
        {
            return nullptr;
        }

        auto result = std::find_if(region->begin(), region->end(), [&](auto &item) {
            if (item.empty())
            {
                return false;
            }

            auto *op = &item.back();
            // auto name = op->getName().getStringRef();
            auto isUnreachable = dyn_cast<LLVM::UnreachableOp>(op) != nullptr;
            return isUnreachable;
        });

        if (result == region->end())
        {
            OpBuilder::InsertionGuard guard(rewriter);

            auto *opBlock = &region->back();
            rewriter.setInsertionPointToEnd(opBlock);
            auto *continuationBlock = rewriter.splitBlock(opBlock, rewriter.getInsertionPoint());
            rewriter.setInsertionPointToStart(continuationBlock);
            rewriter.create<LLVM::UnreachableOp>(loc);
            return continuationBlock;
        }

        return &*result;
    }

    template <typename TyOp> TyOp FindOp(mlir_ts::FuncOp function)
    {
        for (auto &item : function.getBody().getBlocks())
        {
            if (item.empty())
            {
                continue;
            }

            for (auto &op : item)
            {
                if (auto tyOp = dyn_cast<TyOp>(&op))
                {
                    return tyOp;
                }
            }
        }

        return nullptr;
    }

    mlir::Block *BeginBlock(mlir::Location loc)
    {
        auto *opBlock = rewriter.getInsertionBlock();
        auto opPosition = rewriter.getInsertionPoint();
        auto *continuationBlock = rewriter.splitBlock(opBlock, opPosition);

        rewriter.setInsertionPointToEnd(opBlock);

        rewriter.create<mlir::cf::BranchOp>(loc, continuationBlock);

        rewriter.setInsertionPointToStart(continuationBlock);
        return continuationBlock;
    }

    mlir::Block *JumpTo(mlir::Location loc, mlir::Block *toBlock)
    {
        auto *opBlock = rewriter.getInsertionBlock();
        auto opPosition = rewriter.getInsertionPoint();
        auto *continuationBlock = rewriter.splitBlock(opBlock, opPosition);

        rewriter.setInsertionPointToEnd(opBlock);

        rewriter.create<mlir::cf::BranchOp>(loc, toBlock);

        rewriter.setInsertionPointToStart(continuationBlock);
        return continuationBlock;
    }

    mlir::Block *Invoke(mlir::Location loc, std::function<void(mlir::Block *)> bld)
    {
        auto *opBlock = rewriter.getInsertionBlock();
        auto opPosition = rewriter.getInsertionPoint();
        auto *continuationBlock = rewriter.splitBlock(opBlock, opPosition);

        rewriter.setInsertionPointToEnd(opBlock);

        bld(continuationBlock);

        rewriter.setInsertionPointToStart(continuationBlock);
        return continuationBlock;
    }

    mlir::Block *CutBlock()
    {
        auto *opBlock = rewriter.getInsertionBlock();
        auto opPosition = rewriter.getInsertionPoint();
        auto *continuationBlock = rewriter.splitBlock(opBlock, opPosition);
        return continuationBlock;
    }

    mlir::Block *CutBlockAndSetInsertPointToEndOfBlock()
    {
        auto *opBlock = rewriter.getInsertionBlock();
        auto opPosition = rewriter.getInsertionPoint();
        auto *continuationBlock = rewriter.splitBlock(opBlock, opPosition);

        rewriter.setInsertionPointToEnd(opBlock);

        return continuationBlock;
    }
};
} // namespace typescript

#endif // MLIR_TYPESCRIPT_LOWERTOLLVMLOGIC_CODELOGICHELPER_H_
