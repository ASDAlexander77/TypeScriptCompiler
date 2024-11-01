#ifndef MLIR_TYPESCRIPT_LOWERTOLLVMLOGIC_UNDEFLOGICHELPER_H_
#define MLIR_TYPESCRIPT_LOWERTOLLVMLOGIC_UNDEFLOGICHELPER_H_

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

class UndefLogicHelper
{
    Operation *op;
    PatternRewriter &rewriter;
    const LLVMTypeConverter &typeConverter;
    CompileOptions &compileOptions;

  public:
    UndefLogicHelper(Operation *op, PatternRewriter &rewriter, const LLVMTypeConverter &typeConverter, CompileOptions &compileOptions)
        : op(op), rewriter(rewriter), typeConverter(typeConverter), compileOptions(compileOptions)
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
        auto leftUndefType = dyn_cast<mlir_ts::UndefinedType>(leftType);
        auto rightUndefType = dyn_cast<mlir_ts::UndefinedType>(rightType);

        assert(leftUndefType || rightUndefType);

        // case 1, when both are optional
        if (leftUndefType && rightUndefType)
        {
            mlir::Value undefFlagCmpResult;
            switch (opCmpCode)
            {
            case SyntaxKind::EqualsEqualsToken:
            case SyntaxKind::EqualsEqualsEqualsToken:
            case SyntaxKind::GreaterThanEqualsToken:
            case SyntaxKind::LessThanEqualsToken:
                undefFlagCmpResult = clh.createI1ConstantOf(true);
                break;
            case SyntaxKind::ExclamationEqualsToken:
            case SyntaxKind::ExclamationEqualsEqualsToken:
            case SyntaxKind::GreaterThanToken:
            case SyntaxKind::LessThanToken:
                undefFlagCmpResult = clh.createI1ConstantOf(false);
                break;
            default:
                llvm_unreachable("not implemented");
            }

            return undefFlagCmpResult;
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
                    undefFlagCmpResult = clh.createI1ConstantOf(leftUndefType ? false : true);
                    break;
                case SyntaxKind::GreaterThanEqualsToken:
                    undefFlagCmpResult = clh.createI1ConstantOf(leftUndefType ? false : true);
                    break;
                case SyntaxKind::LessThanToken:
                    undefFlagCmpResult = clh.createI1ConstantOf(leftUndefType ? true : false);
                    break;
                case SyntaxKind::LessThanEqualsToken:
                    undefFlagCmpResult = clh.createI1ConstantOf(leftUndefType ? true : false);
                    break;
                default:
                    llvm_unreachable("not implemented");
                }

                return undefFlagCmpResult;
            };

            auto processUndefVale = [&](OpBuilder &builder, Location loc, mlir::Type t1, mlir::Value val1, mlir::Type t2,
                                        mlir::Value val2) {
                if (isa<mlir_ts::InterfaceType>(t2) || isa<mlir_ts::ClassType>(t2))
                {
                    auto casted = rewriter.create<mlir_ts::CastOp>(loc, t2, val1);
                    return LogicOp<StdIOpTy, V1, v1, StdFOpTy, V2, v2>(binOp, opCmpCode, val2, val2.getType(), casted, casted.getType(),
                                                                       rewriter, typeConverter, compileOptions);
                }
                else
                {
                    // result is false already
                    return whenOneValueIsUndef(rewriter, loc);
                }
            };

            // when 1 of them is optional
            if (isa<mlir_ts::UndefinedType>(leftType))
            {
                return processUndefVale(rewriter, loc, leftType, left, rightType, right);
            }

            if (isa<mlir_ts::UndefinedType>(rightType))
            {
                return processUndefVale(rewriter, loc, rightType, right, leftType, left);
            }

            assert(false);
            llvm_unreachable("bug");
        }
    }
};

template <typename StdIOpTy, typename V1, V1 v1, typename StdFOpTy, typename V2, V2 v2>
mlir::Value UndefTypeLogicalOp(Operation *binOp, SyntaxKind opCmpCode, PatternRewriter &builder, const LLVMTypeConverter &typeConverter, CompileOptions &compileOptions)
{
    UndefLogicHelper olh(binOp, builder, typeConverter, compileOptions);
    auto value = olh.logicalOp<StdIOpTy, V1, v1, StdFOpTy, V2, v2>(binOp, opCmpCode);
    return value;
}

} // namespace typescript

#endif // MLIR_TYPESCRIPT_LOWERTOLLVMLOGIC_UNDEFLOGICHELPER_H_