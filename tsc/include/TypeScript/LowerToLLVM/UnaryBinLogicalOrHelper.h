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
mlir::Value OptionalTypeLogicalOp(Operation *, SyntaxKind, PatternRewriter &, LLVMTypeConverter &, CompileOptions&);

template <typename StdIOpTy, typename V1, V1 v1, typename StdFOpTy, typename V2, V2 v2>
mlir::Value UndefTypeLogicalOp(Operation *, SyntaxKind, PatternRewriter &, LLVMTypeConverter &, CompileOptions&);

template <typename UnaryOpTy, typename StdIOpTy, typename StdFOpTy>
void UnaryOp(UnaryOpTy &unaryOp, mlir::Value oper, PatternRewriter &builder)
{
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

template <typename BinOpTy, typename StdIOpTy, typename StdFOpTy, typename UnsignedStdIOpTy = StdIOpTy>
LogicalResult BinOp(BinOpTy &binOp, mlir::Value left, mlir::Value right, PatternRewriter &builder)
{
    auto loc = binOp->getLoc();

    auto leftType = left.getType();
    if (leftType.isIntOrIndex())
    {
        if (leftType.isUnsignedInteger())
        {
            builder.replaceOpWithNewOp<UnsignedStdIOpTy>(binOp, left, right);
        }
        else
        {
            builder.replaceOpWithNewOp<StdIOpTy>(binOp, left, right);
        }
    }
    else if (!leftType.isIntOrIndex() && leftType.isIntOrIndexOrFloat())
    {
        builder.replaceOpWithNewOp<StdFOpTy>(binOp, left, right);
    }
    else if (leftType.template dyn_cast_or_null<mlir_ts::NumberType>())
    {
        auto castLeft = builder.create<mlir_ts::CastOp>(loc, leftType, left);
        auto castRight = builder.create<mlir_ts::CastOp>(loc, leftType, right);
        builder.replaceOpWithNewOp<StdFOpTy>(binOp, castLeft, castRight);
    }
    else
    {
        emitError(binOp.getLoc(), "Binary operation is not supported for type: ") << leftType;
        return failure();
    }

    return success();
}

template <typename StdIOpTy, typename V1, V1 v1, typename StdFOpTy, typename V2, V2 v2>
mlir::Value LogicOp(Operation *binOp, SyntaxKind op, mlir::Value left, mlir::Type leftType, mlir::Value right, mlir::Type rightType,
                    PatternRewriter &builder, LLVMTypeConverter &typeConverter, CompileOptions &compileOptions)
{
    auto loc = binOp->getLoc();

    LLVMTypeConverterHelper llvmtch(typeConverter);

    if (leftType.isa<mlir_ts::OptionalType>() || rightType.isa<mlir_ts::OptionalType>())
    {
        return OptionalTypeLogicalOp<StdIOpTy, V1, v1, StdFOpTy, V2, v2>(binOp, op, builder, typeConverter, compileOptions);
    }
    else if (leftType.isa<mlir_ts::UndefinedType>() || rightType.isa<mlir_ts::UndefinedType>())
    {
        return UndefTypeLogicalOp<StdIOpTy, V1, v1, StdFOpTy, V2, v2>(binOp, op, builder, typeConverter, compileOptions);
    }
    else if (leftType.isIntOrIndex() || leftType.dyn_cast<mlir_ts::BooleanType>() || leftType.dyn_cast<mlir_ts::CharType>())
    {
        auto value = builder.create<StdIOpTy>(loc, v1, left, right);
        return value;
    }
    else if (!leftType.isIntOrIndex() && leftType.isIntOrIndexOrFloat())
    {
        auto value = builder.create<StdFOpTy>(loc, v2, left, right);
        return value;
    }
    else if (leftType.dyn_cast<mlir_ts::NumberType>())
    {
        auto castLeft = builder.create<mlir_ts::CastOp>(loc, leftType, left);
        auto castRight = builder.create<mlir_ts::CastOp>(loc, leftType, right);
        auto value = builder.create<StdFOpTy>(loc, v2, castLeft, castRight);
        return value;
    }
    /*
    else if (auto leftEnumType = leftType.dyn_cast<mlir_ts::EnumType>())
    {
        auto castLeft = builder.create<mlir_ts::CastOp>(loc, leftEnumType.getElementType(), left);
        auto castRight = builder.create<mlir_ts::CastOp>(loc, leftEnumType.getElementType(), right);
        auto res = builder.create<StdFOpTy>(loc, v2, castLeft, castRight);
        builder.create<mlir_ts::CastOp>(binOp, leftEnumType, res);
        return value;
    }
    */
    else if (leftType.dyn_cast<mlir_ts::StringType>())
    {
        if (left.getType() != right.getType())
        {
            right = builder.create<mlir_ts::CastOp>(loc, left.getType(), right);
        }

        auto value = builder.create<mlir_ts::StringCompareOp>(loc, mlir_ts::BooleanType::get(builder.getContext()), left, right,
                                                              builder.getI32IntegerAttr((int)op));

        return value;
    }
    else if (leftType.dyn_cast<mlir_ts::AnyType>() || leftType.dyn_cast<mlir_ts::ClassType>() ||
             leftType.dyn_cast<mlir_ts::OpaqueType>() || leftType.dyn_cast<mlir_ts::NullType>())
    {
        // excluded string
        auto intPtrType = llvmtch.getIntPtrType(0);

        mlir::Value leftAsLLVMType = builder.create<mlir_ts::DialectCastOp>(loc, typeConverter.convertType(left.getType()), left);
        mlir::Value rightAsLLVMType = builder.create<mlir_ts::DialectCastOp>(loc, typeConverter.convertType(right.getType()), right);

        mlir::Value leftPtrValue = builder.create<LLVM::PtrToIntOp>(loc, intPtrType, leftAsLLVMType);
        mlir::Value rightPtrValue = builder.create<LLVM::PtrToIntOp>(loc, intPtrType, rightAsLLVMType);

        auto value = builder.create<StdIOpTy>(loc, v1, leftPtrValue, rightPtrValue);
        return value;
    }
    else if (leftType.dyn_cast<mlir_ts::InterfaceType>())
    {
        // TODO, extract interface VTable to compare
        auto leftVtableValue =
            left.getDefiningOp<mlir_ts::NullOp>() || left.getDefiningOp<LLVM::NullOp>()
                ? left
                : builder.create<mlir_ts::ExtractInterfaceVTableOp>(loc, mlir_ts::OpaqueType::get(leftType.getContext()), left);
        auto rightVtableValue =
            right.getDefiningOp<mlir_ts::NullOp>() || right.getDefiningOp<LLVM::NullOp>()
                ? right
                : builder.create<mlir_ts::ExtractInterfaceVTableOp>(loc, mlir_ts::OpaqueType::get(rightType.getContext()), right);

        // excluded string
        auto intPtrType = llvmtch.getIntPtrType(0);

        mlir::Value leftVtableValueAsLLVMType = builder.create<mlir_ts::DialectCastOp>(loc, typeConverter.convertType(leftVtableValue.getType()), leftVtableValue);
        mlir::Value rightVtableValueAsLLVMType = builder.create<mlir_ts::DialectCastOp>(loc, typeConverter.convertType(rightVtableValue.getType()), rightVtableValue);

        mlir::Value leftPtrValue = builder.create<LLVM::PtrToIntOp>(loc, intPtrType, leftVtableValueAsLLVMType);
        mlir::Value rightPtrValue = builder.create<LLVM::PtrToIntOp>(loc, intPtrType, rightVtableValueAsLLVMType);

        auto value = builder.create<StdIOpTy>(loc, v1, leftPtrValue, rightPtrValue);
        return value;
    }
    else if (auto leftArrayType = leftType.dyn_cast<mlir_ts::ArrayType>())
    {
        // TODO, extract array pointer to compare
        TypeHelper th(builder);
        LLVMCodeHelper ch(binOp, builder, &typeConverter, compileOptions);
        TypeConverterHelper tch(&typeConverter);

        CastLogicHelper castLogic(binOp, builder, tch, compileOptions);

        auto leftArrayPtrValue =
            left.getDefiningOp<mlir_ts::NullOp>() || left.getDefiningOp<LLVM::NullOp>()
                ? left
                : castLogic.extractArrayPtr(left, leftArrayType);
        auto rightArrayPtrValue =
            right.getDefiningOp<mlir_ts::NullOp>() || right.getDefiningOp<LLVM::NullOp>()
                ? right
                : castLogic.extractArrayPtr(right, rightType.dyn_cast<mlir_ts::ArrayType>());

        // excluded string
        auto intPtrType = llvmtch.getIntPtrType(0);

        mlir::Value leftArrayPtrValueAsLLVMType = builder.create<mlir_ts::DialectCastOp>(loc, typeConverter.convertType(leftArrayPtrValue.getType()), leftArrayPtrValue);
        mlir::Value rightArrayPtrValueAsLLVMType = builder.create<mlir_ts::DialectCastOp>(loc, typeConverter.convertType(rightArrayPtrValue.getType()), rightArrayPtrValue);

        mlir::Value leftPtrValue = builder.create<LLVM::PtrToIntOp>(loc, intPtrType, leftArrayPtrValueAsLLVMType);
        mlir::Value rightPtrValue = builder.create<LLVM::PtrToIntOp>(loc, intPtrType, rightArrayPtrValueAsLLVMType);

        auto value = builder.create<StdIOpTy>(loc, v1, leftPtrValue, rightPtrValue);
        return value;
    }    
    else if (auto leftTupleType = leftType.dyn_cast<mlir_ts::TupleType>())
    {        
        // TODO: finish comparing 2 the same tuples
    }

    emitWarning(loc, "Not applicable logical operator for type: '") << leftType << "'";
    // false by default
    CodeLogicHelper clh(binOp, builder);
    return clh.createI1ConstantOf(false);            
}
} // namespace typescript

#endif // MLIR_TYPESCRIPT_LOWERTOLLVMLOGIC_LOGICALORHELPER_H_
