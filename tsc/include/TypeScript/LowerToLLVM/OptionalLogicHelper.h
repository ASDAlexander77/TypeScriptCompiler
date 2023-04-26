#ifndef MLIR_TYPESCRIPT_LOWERTOLLVMLOGIC_OPTIONALLOGICHELPER_H_
#define MLIR_TYPESCRIPT_LOWERTOLLVMLOGIC_OPTIONALLOGICHELPER_H_

#include "TypeScript/Config.h"
#include "TypeScript/Defines.h"
#include "TypeScript/Passes.h"
#include "TypeScript/TypeScriptDialect.h"
#include "TypeScript/TypeScriptOps.h"

#include "TypeScript/LowerToLLVM/UnaryBinLogicalOrHelper.h"
#include "TypeScript/LowerToLLVM/LLVMTypeConverterHelper.h"

#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Builders.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"

#include "scanner_enums.h"

using namespace mlir;
namespace mlir_ts = mlir::typescript;

namespace typescript
{

class OptionalLogicHelper
{
    Operation *op;
    PatternRewriter &rewriter;
    LLVMTypeConverter &typeConverter;

  public:
    OptionalLogicHelper(Operation *op, PatternRewriter &rewriter, LLVMTypeConverter &typeConverter)
        : op(op), rewriter(rewriter), typeConverter(typeConverter)
    {
    }

    template <typename StdIOpTy, typename V1, V1 v1, typename StdFOpTy, typename V2, V2 v2>
    mlir::Value logicalOp(Operation *binOp, SyntaxKind opCmpCode)
    {
        auto loc = binOp->getLoc();

        TypeHelper th(rewriter);
        CodeLogicHelper clh(op, rewriter);

        auto left = binOp->getOperand(0);
        auto right = binOp->getOperand(1);
        auto leftType = left.getType();
        auto rightType = right.getType();
        auto leftOptType = leftType.dyn_cast<mlir_ts::OptionalType>();
        auto rightOptType = rightType.dyn_cast<mlir_ts::OptionalType>();

        assert(leftOptType || rightOptType);

        // case 1, when both are optional
        if (leftOptType && rightOptType)
        {
            // both are optional types
            // compare hasvalue first
            auto leftUndefFlagValueBool = rewriter.create<mlir_ts::HasValueOp>(loc, th.getBooleanType(), left);
            auto rightUndefFlagValueBool = rewriter.create<mlir_ts::HasValueOp>(loc, th.getBooleanType(), right);

            auto leftUndefFlagValue = rewriter.create<mlir_ts::CastOp>(loc, th.getI32Type(), leftUndefFlagValueBool);
            auto rightUndefFlagValue = rewriter.create<mlir_ts::CastOp>(loc, th.getI32Type(), rightUndefFlagValueBool);

            auto whenBothHasNoValues = [&](OpBuilder &builder, Location loc) {
                mlir::Value undefFlagCmpResult;
                switch (opCmpCode)
                {
                case SyntaxKind::EqualsEqualsToken:
                case SyntaxKind::EqualsEqualsEqualsToken:
                    undefFlagCmpResult =
                        rewriter.create<LLVM::ICmpOp>(loc, LLVM::ICmpPredicate::eq, leftUndefFlagValue, rightUndefFlagValue);
                    break;
                case SyntaxKind::ExclamationEqualsToken:
                case SyntaxKind::ExclamationEqualsEqualsToken:
                    undefFlagCmpResult =
                        rewriter.create<LLVM::ICmpOp>(loc, LLVM::ICmpPredicate::ne, leftUndefFlagValue, rightUndefFlagValue);
                    break;
                case SyntaxKind::GreaterThanToken:
                    undefFlagCmpResult =
                        rewriter.create<LLVM::ICmpOp>(loc, LLVM::ICmpPredicate::sgt, leftUndefFlagValue, rightUndefFlagValue);
                    break;
                case SyntaxKind::GreaterThanEqualsToken:
                    undefFlagCmpResult =
                        rewriter.create<LLVM::ICmpOp>(loc, LLVM::ICmpPredicate::sge, leftUndefFlagValue, rightUndefFlagValue);
                    break;
                case SyntaxKind::LessThanToken:
                    undefFlagCmpResult =
                        rewriter.create<LLVM::ICmpOp>(loc, LLVM::ICmpPredicate::slt, leftUndefFlagValue, rightUndefFlagValue);
                    break;
                case SyntaxKind::LessThanEqualsToken:
                    undefFlagCmpResult =
                        rewriter.create<LLVM::ICmpOp>(loc, LLVM::ICmpPredicate::sle, leftUndefFlagValue, rightUndefFlagValue);
                    break;
                default:
                    llvm_unreachable("not implemented");
                }

                return undefFlagCmpResult;
            };

            if (leftOptType.getElementType().isa<mlir_ts::UndefPlaceHolderType>() ||
                rightOptType.getElementType().isa<mlir_ts::UndefPlaceHolderType>())
            {
                // when we have undef in 1 of values we do not condition to test actual values
                return whenBothHasNoValues(rewriter, loc);
            }

            auto andOpResult = rewriter.create<LLVM::AndOp>(loc, th.getI32Type(), leftUndefFlagValue, rightUndefFlagValue);
            auto const0 = clh.createI32ConstantOf(0);
            auto bothHasResult = rewriter.create<LLVM::ICmpOp>(loc, LLVM::ICmpPredicate::ne, andOpResult, const0);

            auto result = clh.conditionalExpressionLowering(
                loc, th.getBooleanType(), bothHasResult,
                [&](OpBuilder &builder, Location loc) {
                    auto leftSubType = leftOptType.getElementType();
                    auto rightSubType = rightOptType.getElementType();
                    left = rewriter.create<mlir_ts::ValueOp>(loc, leftSubType, left);
                    right = rewriter.create<mlir_ts::ValueOp>(loc, rightSubType, right);
                    return LogicOp<StdIOpTy, V1, v1, StdFOpTy, V2, v2>(binOp, opCmpCode, left, leftSubType, right, rightSubType, rewriter,
                                                                       typeConverter);
                },
                whenBothHasNoValues);

            return result;
        }
        else
        {
            // case when 1 value is optional
            auto whenOneValueIsUndef = [&](OpBuilder &builder, Location loc) {
                mlir::Value undefFlagCmpResult;
                switch (opCmpCode)
                {
                case SyntaxKind::EqualsEqualsToken:
                case SyntaxKind::EqualsEqualsEqualsToken:
                    undefFlagCmpResult = clh.createI1ConstantOf(false);
                    break;
                case SyntaxKind::ExclamationEqualsToken:
                case SyntaxKind::ExclamationEqualsEqualsToken:
                    undefFlagCmpResult = clh.createI1ConstantOf(true);
                    break;
                case SyntaxKind::GreaterThanToken:
                    undefFlagCmpResult = clh.createI1ConstantOf(leftOptType ? false : true);
                    break;
                case SyntaxKind::GreaterThanEqualsToken:
                    undefFlagCmpResult = clh.createI1ConstantOf(leftOptType ? false : true);
                    break;
                case SyntaxKind::LessThanToken:
                    undefFlagCmpResult = clh.createI1ConstantOf(leftOptType ? true : false);
                    break;
                case SyntaxKind::LessThanEqualsToken:
                    undefFlagCmpResult = clh.createI1ConstantOf(leftOptType ? true : false);
                    break;
                default:
                    llvm_unreachable("not implemented");
                }

                return undefFlagCmpResult;
            };

            auto processUndefVale = [&](OpBuilder &builder, Location loc, mlir::Type t1, mlir::Value val1, mlir::Type t2,
                                        mlir::Value val2) {
                if (t2.isa<mlir_ts::InterfaceType>() || t2.isa<mlir_ts::ClassType>())
                {
                    auto casted = rewriter.create<mlir_ts::CastOp>(loc, t2, val1);
                    return LogicOp<StdIOpTy, V1, v1, StdFOpTy, V2, v2>(binOp, opCmpCode, val2, val2.getType(), casted, casted.getType(),
                                                                       rewriter, typeConverter);
                }
                else
                {
                    // result is false already
                    return whenOneValueIsUndef(rewriter, loc);
                }
            };

            // when 1 of them is optional
            if (leftOptType && leftOptType.getElementType().isa<mlir_ts::UndefPlaceHolderType>())
            {
                return processUndefVale(rewriter, loc, leftType, left, rightType, right);
            }

            if (rightOptType && rightOptType.getElementType().isa<mlir_ts::UndefPlaceHolderType>())
            {
                return processUndefVale(rewriter, loc, rightType, right, leftType, left);
            }

            if (leftOptType)
            {
                auto leftSubType = leftOptType.getElementType();
                left = rewriter.create<mlir_ts::ValueOp>(loc, leftSubType, left);
            }

            if (rightOptType)
            {
                auto rightSubType = rightOptType.getElementType();
                right = rewriter.create<mlir_ts::ValueOp>(loc, rightSubType, right);
            }

            return LogicOp<StdIOpTy, V1, v1, StdFOpTy, V2, v2>(binOp, opCmpCode, left, left.getType(), right, right.getType(), rewriter,
                                                               typeConverter);
        }
    }
};

template <typename StdIOpTy, typename V1, V1 v1, typename StdFOpTy, typename V2, V2 v2>
mlir::Value OptionalTypeLogicalOp(Operation *binOp, SyntaxKind opCmpCode, PatternRewriter &builder, LLVMTypeConverter &typeConverter)
{
    OptionalLogicHelper olh(binOp, builder, typeConverter);
    auto value = olh.logicalOp<StdIOpTy, V1, v1, StdFOpTy, V2, v2>(binOp, opCmpCode);
    return value;
}

} // namespace typescript

#endif // MLIR_TYPESCRIPT_LOWERTOLLVMLOGIC_OPTIONALLOGICHELPER_H_