#include "TypeScript/TypeScriptDialect.h"
#include "TypeScript/TypeScriptOps.h"
#include "TypeScript/Passes.h"
#include "TypeScript/Defines.h"
#include "TypeScript/EnumsAST.h"

#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Conversion/SCFToStandard/SCFToStandard.h"
#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVM.h"
#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVMPass.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/Sequence.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace ::typescript;
namespace ts = mlir::typescript;

//===----------------------------------------------------------------------===//
// TypeScriptToLLVM RewritePatterns
//===----------------------------------------------------------------------===//

namespace typescript
{
    mlir::Type typeToConvertedType(mlir::Type type, mlir::TypeConverter &typeConverter)
    {
        auto convertedType = typeConverter.convertType(type);
        assert(convertedType);
        return convertedType;
    }

    Value createI32ConstantOf(Location loc, PatternRewriter &rewriter, unsigned value)
    {
        return rewriter.create<LLVM::ConstantOp>(loc, rewriter.getIntegerType(32), rewriter.getIntegerAttr(rewriter.getI32Type(), value));
    }

    Value createI1ConstantOf(Location loc, PatternRewriter &rewriter, bool value)
    {
        return rewriter.create<LLVM::ConstantOp>(loc, rewriter.getIntegerType(1), rewriter.getIntegerAttr(rewriter.getI1Type(), value));
    }    

    Type getIntPtrType(unsigned addressSpace, LLVMTypeConverter &typeConverter)
    {
        return IntegerType::get(&typeConverter.getContext(), typeConverter.getPointerBitwidth(addressSpace));
    }

    Value conditionalExpressionLowering(
        Location loc, Type type, Value condition,
        function_ref<Value(OpBuilder &, Location)> thenBuilder,
        function_ref<Value(OpBuilder &, Location)> elseBuilder,
        PatternRewriter &rewriter)
    {
        // TODO: or maybe only result should have types arguments as BR to Result has values from branch?

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
        auto *resultBlock = rewriter.createBlock(continuationBlock, TypeRange{type});
        rewriter.create<LLVM::BrOp>(
            loc,
            ValueRange{},
            continuationBlock);

        rewriter.setInsertionPointToEnd(thenBlock);
        rewriter.create<LLVM::BrOp>(
            loc,
            ValueRange{thenValue},
            resultBlock);

        rewriter.setInsertionPointToEnd(elseBlock);
        rewriter.create<LLVM::BrOp>(
            loc,
            ValueRange{elseValue},
            resultBlock);

        // Generate assertion test.
        rewriter.setInsertionPointToEnd(opBlock);
        rewriter.create<LLVM::CondBrOp>(
            loc,
            condition,
            thenBlock,
            elseBlock);

        rewriter.setInsertionPointToStart(continuationBlock);

        return resultBlock->getArguments().front();
    }

    void NegativeOp(ts::ArithmeticUnaryOp &unaryOp, mlir::PatternRewriter &builder)
    {
        auto oper = unaryOp.operand1();
        auto type = oper.getType();
        if (type.isIntOrIndex())
        {
            mlir::Value lhs;
            if (type.isInteger(1))
            {
                lhs = createI1ConstantOf(unaryOp->getLoc(), builder, true);
            }
            else
            {
                lhs = createI32ConstantOf(unaryOp->getLoc(), builder, 0xffff);
            }

            builder.replaceOpWithNewOp<LLVM::XOrOp>(unaryOp, type, oper, lhs);
        }
        else if (!type.isIntOrIndex() && type.isIntOrIndexOrFloat())
        {
            builder.replaceOpWithNewOp<LLVM::XOrOp>(unaryOp, oper);
        }
        else
        {
            llvm_unreachable("not implemented");
        }
    }    
}