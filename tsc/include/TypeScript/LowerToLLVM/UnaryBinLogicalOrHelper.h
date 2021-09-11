#ifndef MLIR_TYPESCRIPT_LOWERTOLLVMLOGIC_LOGICALORHELPER_H_
#define MLIR_TYPESCRIPT_LOWERTOLLVMLOGIC_LOGICALORHELPER_H_

#include "TypeScript/Config.h"
#include "TypeScript/Defines.h"
#include "TypeScript/Passes.h"
#include "TypeScript/TypeScriptDialect.h"
#include "TypeScript/TypeScriptOps.h"

#include "TypeScript/LowerToLLVM/LLVMTypeConverterHelper.h"

#include "scanner_enums.h"

using namespace mlir;
namespace mlir_ts = mlir::typescript;

namespace typescript
{

template <typename StdIOpTy, typename V1, V1 v1, typename StdFOpTy, typename V2, V2 v2>
Value OptinalTypeLogicalOp(Operation *binOp, SyntaxKind opCmpCode);

template <typename UnaryOpTy, typename StdIOpTy, typename StdFOpTy> void UnaryOp(UnaryOpTy &unaryOp, PatternRewriter &builder)
{
    auto oper = unaryOp.operand1();
    auto type = oper.getType();
    if (type.isIntOrIndex())
    {
        builder.replaceOpWithNewOp<StdIOpTy>(unaryOp, type, oper);
    }
    else if (!type.isIntOrIndex() && type.isIntOrIndexOrFloat())
    {
        builder.replaceOpWithNewOp<StdFOpTy>(unaryOp, type, oper);
    }
    else
    {
        emitError(unaryOp.getLoc(), "Not implemented operator for type 1: '") << type << "'";
        llvm_unreachable("not implemented");
    }
}

template <typename BinOpTy, typename StdIOpTy, typename StdFOpTy> void BinOp(BinOpTy &binOp, PatternRewriter &builder)
{
    auto loc = binOp->getLoc();

    auto left = binOp->getOperand(0);
    auto right = binOp->getOperand(1);
    auto leftType = left.getType();
    if (leftType.isIntOrIndex())
    {
        builder.replaceOpWithNewOp<StdIOpTy>(binOp, left, right);
    }
    else if (!leftType.isIntOrIndex() && leftType.isIntOrIndexOrFloat())
    {
        builder.replaceOpWithNewOp<StdFOpTy>(binOp, left, right);
    }
    else if (leftType.template dyn_cast_or_null<mlir_ts::NumberType>())
    {
        auto castLeft = builder.create<mlir_ts::CastOp>(loc, builder.getF32Type(), left);
        auto castRight = builder.create<mlir_ts::CastOp>(loc, builder.getF32Type(), right);
        builder.replaceOpWithNewOp<StdFOpTy>(binOp, castLeft, castRight);
    }
    else
    {
        emitError(binOp.getLoc(), "Not implemented operator for type 1: '") << leftType << "'";
        llvm_unreachable("not implemented");
    }
}

template <typename StdIOpTy, typename V1, V1 v1, typename StdFOpTy, typename V2, V2 v2>
mlir::Value LogicOp(Operation *binOp, SyntaxKind op, mlir::Value left, mlir::Value right, PatternRewriter &builder,
                    LLVMTypeConverter &typeConverter)
{
    auto loc = binOp->getLoc();

    LLVMTypeConverterHelper llvmtch(typeConverter);

    auto leftType = left.getType();
    auto rightType = right.getType();

    if (leftType.isa<mlir_ts::OptionalType>() || rightType.isa<mlir_ts::OptionalType>())
    {
        return return OptinalTypeLogicalOp<StdIOpTy, V1, v1, StdFOpTy, V2, v2>(binOp, op);
    }
    else if (leftType.isIntOrIndex() || leftType.dyn_cast_or_null<mlir_ts::BooleanType>())
    {
        auto value = builder.create<StdIOpTy>(loc, v1, left, right);
        return value;
    }
    else if (!leftType.isIntOrIndex() && leftType.isIntOrIndexOrFloat())
    {
        auto value = builder.create<StdFOpTy>(loc, v2, left, right);
        return value;
    }
    else if (leftType.dyn_cast_or_null<mlir_ts::NumberType>())
    {
        auto castLeft = builder.create<mlir_ts::CastOp>(loc, builder.getF32Type(), left);
        auto castRight = builder.create<mlir_ts::CastOp>(loc, builder.getF32Type(), right);
        auto value = builder.create<StdFOpTy>(loc, v2, castLeft, castRight);
        return value;
    }
    /*
    else if (auto leftEnumType = leftType.dyn_cast_or_null<mlir_ts::EnumType>())
    {
        auto castLeft = builder.create<mlir_ts::CastOp>(loc, leftEnumType.getElementType(), left);
        auto castRight = builder.create<mlir_ts::CastOp>(loc, leftEnumType.getElementType(), right);
        auto res = builder.create<StdFOpTy>(loc, v2, castLeft, castRight);
        builder.create<mlir_ts::CastOp>(binOp, leftEnumType, res);
        return value;
    }
    */
    else if (leftType.dyn_cast_or_null<mlir_ts::StringType>())
    {
        if (left.getType() != right.getType())
        {
            right = builder.create<mlir_ts::CastOp>(loc, left.getType(), right);
        }

        auto value = builder.create<mlir_ts::StringCompareOp>(loc, mlir_ts::BooleanType::get(builder.getContext()), left, right,
                                                              builder.getI32IntegerAttr((int)op));

        return value;
    }
    else if (leftType.dyn_cast_or_null<mlir_ts::AnyType>() || leftType.dyn_cast_or_null<mlir_ts::ClassType>())
    {
        // excluded string
        auto intPtrType = llvmtch.getIntPtrType(0);

        Value leftPtrValue = builder.create<LLVM::PtrToIntOp>(loc, intPtrType, left);
        Value rightPtrValue = builder.create<LLVM::PtrToIntOp>(loc, intPtrType, right);

        auto value = builder.create<StdIOpTy>(loc, v1, leftPtrValue, rightPtrValue);
        return value;
    }
    else
    {
        emitError(loc, "Not implemented operator for type 1: '") << leftType << "'";
        llvm_unreachable("not implemented");
    }
}
} // namespace typescript

#endif // MLIR_TYPESCRIPT_LOWERTOLLVMLOGIC_LOGICALORHELPER_H_
