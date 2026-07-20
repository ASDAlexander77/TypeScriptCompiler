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
                else if (isa<mlir_ts::BoundFunctionType>(t2))
                {
                    // an optional (`?`) interface/class METHOD compared against
                    // `undefined` (e.g. `missing.opt == undefined` -
                    // 00interface_optional_method_extends.ts). Unlike an optional
                    // FIELD (OptionalType + HasValueOp runtime check), a bound_func has
                    // no such wrapper, so this used to fall to whenOneValueIsUndef below
                    // and return a compile-time-constant false/true regardless of the
                    // actual value - meaning the comparison never reflected whether the
                    // method was really present. InterfaceSymbolRefOpLowering now selects
                    // a null `this` pointer when the vtable slot holds the "missing
                    // optional member" -1 sentinel (a real bound method's `this` is
                    // never null), so check that instead.
                    //
                    // NOTE: deliberately NOT using LogicOp<StdIOpTy, V1, v1, ...> here to
                    // compute "is this null" - v1 is a template parameter baked in from
                    // the ORIGINAL comparison operator that triggered this whole call
                    // (e.g. arith::CmpIPredicate::ne for a source-level `!=`), not
                    // something the `op`/SyntaxKind argument can override at the call
                    // site (LogicOp's isIntOrIndex branch ignores `op` entirely and uses
                    // `v1` directly) - passing SyntaxKind::EqualsEqualsToken here while v1
                    // is still `ne` silently computed "this != null" instead of "this ==
                    // null", inverting the result. Emit the LLVM::ICmpOp directly instead,
                    // pointer-converting both sides the same way LogicOp's
                    // isNullableTypeNoUnion branch would.
                    auto thisVal = rewriter.create<mlir_ts::GetThisOp>(loc, mlir_ts::OpaqueType::get(rewriter.getContext()), val2);
                    auto nullVal = rewriter.create<mlir_ts::NullOp>(loc, mlir_ts::NullType::get(rewriter.getContext()));
                    LLVMTypeConverterHelper llvmtch(&typeConverter);
                    auto intPtrType = llvmtch.getIntPtrType(0);
                    auto thisValAsLLVMType = rewriter.create<mlir_ts::DialectCastOp>(loc, typeConverter.convertType(thisVal.getType()), thisVal);
                    auto nullValAsLLVMType = rewriter.create<mlir_ts::DialectCastOp>(loc, typeConverter.convertType(nullVal.getType()), nullVal);
                    auto thisPtrValue = rewriter.create<LLVM::PtrToIntOp>(loc, intPtrType, thisValAsLLVMType);
                    auto nullPtrValue = rewriter.create<LLVM::PtrToIntOp>(loc, intPtrType, nullValAsLLVMType);
                    mlir::Value isNull = rewriter.create<LLVM::ICmpOp>(loc, LLVM::ICmpPredicate::eq, thisPtrValue, nullPtrValue);

                    switch (opCmpCode)
                    {
                    case SyntaxKind::EqualsEqualsToken:
                    case SyntaxKind::EqualsEqualsEqualsToken:
                        return isNull;
                    case SyntaxKind::ExclamationEqualsToken:
                    case SyntaxKind::ExclamationEqualsEqualsToken:
                    {
                        auto trueVal = clh.createI1ConstantOf(true);
                        return (mlir::Value)rewriter.create<LLVM::XOrOp>(loc, isNull, trueVal);
                    }
                    default:
                        // ordering comparisons against undefined aren't meaningful for a
                        // callable - same "result is false already" fallback as any
                        // other unhandled type below.
                        return whenOneValueIsUndef(rewriter, loc);
                    }
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