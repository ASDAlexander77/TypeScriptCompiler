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

        if (isa<mlir_ts::UndefinedType>(rightType) || isa<mlir_ts::UndefinedType>(leftType))
        {
            auto optValue = isa<mlir_ts::UndefinedType>(rightType) ? left : right;
            auto undefValue = isa<mlir_ts::UndefinedType>(rightType) ? right : left;
            if (isLooseEquality(opCmpCode) && canHoldNull(mlir::cast<mlir_ts::OptionalType>(optValue.getType()).getElementType()))
            {
                return whenLooseUndefAgainstNullable<StdIOpTy, V1, v1, StdFOpTy, V2, v2>(opCmpCode, optValue);
            }

            // when we have undef in 1 of values we do not condition to test actual values
            return whenOneValueIsUndef(opCmpCode, optValue, undefValue);
        }

        return whenOneOptValue<StdIOpTy, V1, v1, StdFOpTy, V2, v2>(opCmpCode);
    }

    static bool isLooseEquality(SyntaxKind opCmpCode)
    {
        return opCmpCode == SyntaxKind::EqualsEqualsToken || opCmpCode == SyntaxKind::ExclamationEqualsToken;
    }

    // a `T | null` (strict null checks), or a type whose own pointer holds null (without them)
    static bool canHoldNull(mlir::Type type)
    {
        if (auto unionType = dyn_cast<mlir_ts::UnionType>(type))
        {
            return llvm::any_of(unionType.getTypes(), [](mlir::Type member) { return isa<mlir_ts::NullType>(member); });
        }

        return MLIRTypeCore::isNullableTypeNoUnion(type);
    }

    // Loosely, null equals undefined: an optional holding null is `== undefined` as well, so it
    // is not enough to ask whether it holds a value - a value it holds may be null. The value is
    // read only when there is one.
    template <typename StdIOpTy, typename V1, V1 v1, typename StdFOpTy, typename V2, V2 v2>
    mlir::Value whenLooseUndefAgainstNullable(SyntaxKind opCmpCode, mlir::Value optValue)
    {
        auto loc = binOp->getLoc();

        TypeHelper th(rewriter);
        CodeLogicHelper clh(binOp, rewriter);

        auto llvmBoolType = typeConverter.convertType(th.getBooleanType());
        auto optType = mlir::cast<mlir_ts::OptionalType>(optValue.getType());

        auto hasValueBool = rewriter.create<mlir_ts::HasValueOp>(loc, th.getBooleanType(), optValue);
        auto hasValue = rewriter.create<mlir_ts::DialectCastOp>(loc, llvmBoolType, hasValueBool);

        return clh.conditionalExpressionLowering(
            loc, llvmBoolType, hasValue,
            [&](OpBuilder &builder, Location loc) {
                mlir::Value value = rewriter.create<mlir_ts::ValueOp>(loc, optType.getElementType(), optValue);
                mlir::Value nullValue = rewriter.create<mlir_ts::NullOp>(loc, mlir_ts::NullType::get(rewriter.getContext()));
                mlir::Value result = LogicOp<StdIOpTy, V1, v1, StdFOpTy, V2, v2>(
                    binOp, opCmpCode, value, value.getType(), nullValue, nullValue.getType(), rewriter, typeConverter, compileOptions);
                if (result && result.getType() != llvmBoolType)
                {
                    result = rewriter.create<mlir_ts::DialectCastOp>(loc, llvmBoolType, result);
                }

                return result;
            },
            [&](OpBuilder &builder, Location loc) {
                return clh.createI1ConstantOf(opCmpCode == SyntaxKind::EqualsEqualsToken);
            });
    }

    // One side is optional, the other a value. An optional holding undefined equals no value - not
    // 0, not false, not null - so the values are compared only when it holds one: unwrapped without
    // that test it read whatever the empty optional stored, and `undefined === 0` was true.
    template <typename StdIOpTy, typename V1, V1 v1, typename StdFOpTy, typename V2, V2 v2>
    mlir::Value whenOneOptValue(SyntaxKind opCmpCode)
    {
        auto loc = binOp->getLoc();

        TypeHelper th(rewriter);
        CodeLogicHelper clh(binOp, rewriter);

        auto llvmBoolType = typeConverter.convertType(th.getBooleanType());

        auto left = binOp->getOperand(0);
        auto right = binOp->getOperand(1);
        auto leftOptType = dyn_cast<mlir_ts::OptionalType>(left.getType());
        auto rightOptType = dyn_cast<mlir_ts::OptionalType>(right.getType());
        auto otherIsNull = isa<mlir_ts::NullType>((leftOptType ? right : left).getType());

        auto hasValueBool = rewriter.create<mlir_ts::HasValueOp>(loc, th.getBooleanType(), leftOptType ? left : right);
        auto hasValue = rewriter.create<mlir_ts::DialectCastOp>(loc, llvmBoolType, hasValueBool);

        return clh.conditionalExpressionLowering(
            loc, llvmBoolType, hasValue,
            [&](OpBuilder &builder, Location loc) {
                if (leftOptType)
                {
                    left = rewriter.create<mlir_ts::ValueOp>(loc, leftOptType.getElementType(), left);
                }

                if (rightOptType)
                {
                    right = rewriter.create<mlir_ts::ValueOp>(loc, rightOptType.getElementType(), right);
                }

                mlir::Value result = LogicOp<StdIOpTy, V1, v1, StdFOpTy, V2, v2>(
                    binOp, opCmpCode, left, left.getType(), right, right.getType(), rewriter, typeConverter, compileOptions);
                if (result && result.getType() != llvmBoolType)
                {
                    result = rewriter.create<mlir_ts::DialectCastOp>(loc, llvmBoolType, result);
                }

                return result;
            },
            [&](OpBuilder &builder, Location loc) {
                // undefined against a value: only "not equal" holds, and no ordering does - except
                // that loosely undefined equals null
                switch (opCmpCode)
                {
                case SyntaxKind::EqualsEqualsToken:
                    return clh.createI1ConstantOf(otherIsNull);
                case SyntaxKind::ExclamationEqualsToken:
                    return clh.createI1ConstantOf(!otherIsNull);
                case SyntaxKind::ExclamationEqualsEqualsToken:
                    return clh.createI1ConstantOf(true);
                default:
                    return clh.createI1ConstantOf(false);
                }
            });
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
                // opCmpCode's only caller (LowerToLLVM.cpp's LogicalBinaryOpLowering) switches
                // on the exact same 8 comparison SyntaxKinds this switch handles above and
                // never passes anything else; fail cleanly instead of crashing if that ever
                // stops being true.
                mlir::emitError(loc, "unsupported comparison operator for optional type");
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
            // same call-site-guaranteed shape as the sibling switch above.
            mlir::emitError(loc, "unsupported comparison operator for optional type");
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