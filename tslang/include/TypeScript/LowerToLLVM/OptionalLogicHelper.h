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
    Operation *binOp;
    PatternRewriter &rewriter;
    const LLVMTypeConverter &typeConverter;
    CompileOptions &compileOptions;

  public:
    OptionalLogicHelper(Operation *binOp, PatternRewriter &rewriter, const LLVMTypeConverter &typeConverter, CompileOptions &compileOptions)
        : binOp(binOp), rewriter(rewriter), typeConverter(typeConverter), compileOptions(compileOptions)
    {
    }

    template <typename StdIOpTy, typename V1, V1 v1, typename StdFOpTy, typename V2, V2 v2>
    mlir::Value logicalOp(SyntaxKind opCmpCode)
    {
        auto loc = binOp->getLoc();

        auto left = binOp->getOperand(0);
        auto right = binOp->getOperand(1);
        auto leftType = left.getType();
        auto rightType = right.getType();
        auto leftOptType = dyn_cast<mlir_ts::OptionalType>(leftType);
        auto rightOptType = dyn_cast<mlir_ts::OptionalType>(rightType);

        assert(leftOptType || rightOptType);

        // case 1, when both are optional
        if (leftOptType && rightOptType)
        {
            return WhenBothOptValues<StdIOpTy, V1, v1, StdFOpTy, V2, v2>(opCmpCode);
        }

        if (isa<mlir_ts::UndefinedType>(rightType))
        {
            // when we have undef in 1 of values we do not condition to test actual values
            return whenOneValueIsUndef(opCmpCode, left, right);
        }

        if (isa<mlir_ts::UndefinedType>(leftType))
        {
            // when we have undef in 1 of values we do not condition to test actual values
            return whenOneValueIsUndef(opCmpCode, right, left);
        }

        // TODO: rewrite code to take in account that Opt value can be undefined
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
                                                            typeConverter, compileOptions);
    }

    template <typename StdIOpTy, typename V1, V1 v1, typename StdFOpTy, typename V2, V2 v2>
    mlir::Value WhenBothOptValues(SyntaxKind opCmpCode)
    {
        auto loc = binOp->getLoc();

        TypeHelper th(rewriter);
        CodeLogicHelper clh(binOp, rewriter);

        auto llvmBoolType = typeConverter.convertType(th.getBooleanType());

        auto left = binOp->getOperand(0);
        auto right = binOp->getOperand(1);
        auto leftType = left.getType();
        auto rightType = right.getType();
        auto leftOptType = dyn_cast<mlir_ts::OptionalType>(leftType);
        auto rightOptType = dyn_cast<mlir_ts::OptionalType>(rightType);        

        // both are optional types
        // compare hasvalue first
        auto leftUndefFlagValueBool = rewriter.create<mlir_ts::HasValueOp>(loc, th.getBooleanType(), left);
        auto rightUndefFlagValueBool = rewriter.create<mlir_ts::HasValueOp>(loc, th.getBooleanType(), right);

        auto leftUndefFlagValue = rewriter.create<mlir_ts::CastOp>(loc, th.getI32Type(), leftUndefFlagValueBool);
        auto rightUndefFlagValue = rewriter.create<mlir_ts::CastOp>(loc, th.getI32Type(), rightUndefFlagValueBool);

        auto whenOneOrBothHaveNoValues = [&](OpBuilder &builder, Location loc) {
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

        auto andOpResult = rewriter.create<LLVM::AndOp>(loc, th.getI32Type(), leftUndefFlagValue, rightUndefFlagValue);
        auto const0 = clh.createI32ConstantOf(0);
        auto bothHasResult = rewriter.create<LLVM::ICmpOp>(loc, LLVM::ICmpPredicate::ne, andOpResult, const0);

        auto result = clh.conditionalExpressionLowering(
            loc, llvmBoolType, bothHasResult,
            [&](OpBuilder &builder, Location loc) {
                auto leftSubType = leftOptType.getElementType();
                auto rightSubType = rightOptType.getElementType();
                left = rewriter.create<mlir_ts::ValueOp>(loc, leftSubType, left);
                right = rewriter.create<mlir_ts::ValueOp>(loc, rightSubType, right);
                return LogicOp<StdIOpTy, V1, v1, StdFOpTy, V2, v2>(
                    binOp, opCmpCode, left, leftSubType, right, rightSubType, rewriter, typeConverter, compileOptions);
            },
            whenOneOrBothHaveNoValues);

        return result;
    }

    mlir::Value whenOneValueIsUndef(SyntaxKind opCmpCode, mlir::Value left, mlir::Value right)
    {
        auto loc = binOp->getLoc();

        TypeHelper th(rewriter);
        CodeLogicHelper clh(binOp, rewriter);

        assert(isa<mlir_ts::UndefinedType>(right.getType()));

        auto leftUndefFlagValueBool = rewriter.create<mlir_ts::HasValueOp>(loc, th.getBooleanType(), left);

        auto leftUndefFlagValue = rewriter.create<mlir_ts::CastOp>(loc, th.getI32Type(), leftUndefFlagValueBool);

        auto rightUndefFlagValue = clh.createI32ConstantOf(0);

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
    }
};

template <typename StdIOpTy, typename V1, V1 v1, typename StdFOpTy, typename V2, V2 v2>
mlir::Value OptionalTypeLogicalOp(Operation *binOp, SyntaxKind opCmpCode, PatternRewriter &builder, const LLVMTypeConverter &typeConverter, CompileOptions &compileOptions)
{
    OptionalLogicHelper olh(binOp, builder, typeConverter, compileOptions);
    auto value = olh.logicalOp<StdIOpTy, V1, v1, StdFOpTy, V2, v2>(opCmpCode);
    return value;
}

} // namespace typescript

#endif // MLIR_TYPESCRIPT_LOWERTOLLVMLOGIC_OPTIONALLOGICHELPER_H_