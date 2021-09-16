#ifndef MLIR_TYPESCRIPT_LOWERTOLLVMLOGIC_LLVMRTTIHELPERVCWIN32_H_
#define MLIR_TYPESCRIPT_LOWERTOLLVMLOGIC_LLVMRTTIHELPERVCWIN32_H_

#include "TypeScript/Config.h"
#include "TypeScript/Defines.h"
#include "TypeScript/Passes.h"
#include "TypeScript/TypeScriptDialect.h"
#include "TypeScript/TypeScriptOps.h"

#include "TypeScript/LowerToLLVM/TypeHelper.h"
#include "TypeScript/LowerToLLVM/LLVMCodeHelper.h"
#include "TypeScript/LowerToLLVM/LLVMRTTIHelperVCWin32Const.h"

#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
namespace mlir_ts = mlir::typescript;

namespace typescript
{

constexpr auto typeInfoExtRef = "??_7type_info@@6B@";
constexpr auto imageBaseRef = "__ImageBase";
class LLVMRTTIHelperVCWin32
{
    Operation *op;
    PatternRewriter &rewriter;
    ModuleOp parentModule;
    TypeHelper th;
    LLVMCodeHelper ch;

  public:
    const char *typeName;
    const char *typeName2;
    const char *typeInfoRef;
    const char *typeInfoRef2;
    const char *catchableTypeInfoRef;
    const char *catchableTypeInfoRef2;
    const char *catchableTypeInfoArrayRef;
    const char *throwInfoRef;
    bool type2 = false;

    LLVMRTTIHelperVCWin32(Operation *op, PatternRewriter &rewriter, TypeConverter &typeConverter)
        : op(op), rewriter(rewriter), parentModule(op->getParentOfType<ModuleOp>()), th(rewriter), ch(op, rewriter, &typeConverter)
    {
        // setI32AsCatchType();
    }

    void setF32AsCatchType()
    {
        typeName = F32Type::typeName;
        typeInfoRef = F32Type::typeInfoRef;
        catchableTypeInfoRef = F32Type::catchableTypeInfoRef;
        catchableTypeInfoArrayRef = F32Type::catchableTypeInfoArrayRef;
        throwInfoRef = F32Type::throwInfoRef;
        type2 = false;
    }

    void setI32AsCatchType()
    {
        typeName = I32Type::typeName;
        typeInfoRef = I32Type::typeInfoRef;
        catchableTypeInfoRef = I32Type::catchableTypeInfoRef;
        catchableTypeInfoArrayRef = I32Type::catchableTypeInfoArrayRef;
        throwInfoRef = I32Type::throwInfoRef;
        type2 = false;
    }

    void setStringTypeAsCatchType()
    {
        typeName = StringType::typeName;
        typeName2 = StringType::typeName2;
        typeInfoRef = StringType::typeInfoRef;
        typeInfoRef2 = StringType::typeInfoRef2;
        catchableTypeInfoRef = StringType::catchableTypeInfoRef;
        catchableTypeInfoRef2 = StringType::catchableTypeInfoRef2;
        catchableTypeInfoArrayRef = StringType::catchableTypeInfoArrayRef;
        throwInfoRef = StringType::throwInfoRef;
        type2 = true;
    }

    void setI8PtrAsCatchType()
    {
        typeName = I8PtrType::typeName;
        typeInfoRef = I8PtrType::typeInfoRef;
        catchableTypeInfoRef = I8PtrType::catchableTypeInfoRef;
        catchableTypeInfoArrayRef = I8PtrType::catchableTypeInfoArrayRef;
        throwInfoRef = I8PtrType::throwInfoRef;
        type2 = false;
    }

    LogicalResult setPersonality(mlir::FuncOp newFuncOp)
    {
        auto cxxFrameHandler3 = ch.getOrInsertFunction("__CxxFrameHandler3", th.getFunctionType(th.getI32Type(), {}, true));

        newFuncOp->setAttr(rewriter.getIdentifier("personality"), FlatSymbolRefAttr::get(rewriter.getContext(), "__CxxFrameHandler3"));
        return success();
    }

    void setType(mlir::Type type)
    {
        TypeSwitch<Type>(type)
            .Case<mlir::IntegerType>([&](auto intType) {
                if (intType.getIntOrFloatBitWidth() == 32)
                {
                    setI32AsCatchType();
                }
                else
                {
                    llvm_unreachable("not implemented");
                }
            })
            .Case<mlir::FloatType>([&](auto floatType) {
                if (floatType.getIntOrFloatBitWidth() == 32)
                {
                    setF32AsCatchType();
                }
                else
                {
                    llvm_unreachable("not implemented");
                }
            })
            .Case<mlir_ts::NumberType>([&](auto numberType) { setF32AsCatchType(); })
            .Case<mlir_ts::StringType>([&](auto stringType) { setStringTypeAsCatchType(); })
            .Case<mlir_ts::ClassType>([&](auto classType) { setI8PtrAsCatchType(); })
            .Case<mlir_ts::AnyType>([&](auto stringType) { setI8PtrAsCatchType(); })
            .Default([&](auto type) { llvm_unreachable("not implemented"); });
    }

    void setRTTIForType(mlir::Location loc, mlir::Type type)
    {
        setType(type);

        auto parentModule = op->getParentOfType<ModuleOp>();

        OpBuilder::InsertionGuard guard(rewriter);

        rewriter.setInsertionPointToStart(parentModule.getBody());
        ch.seekLast(parentModule.getBody());

        // ??_7type_info@@6B@
        typeInfo(loc);

        // ??_R0N@8
        typeDescriptors(loc);

        // __ImageBase
        imageBase(loc);

        // _CT??_R0N@88
        catchableTypes(loc);

        // _CTA1N
        catchableArrayType(loc);

        // _TI1N
        throwInfo(loc);
    }

    LogicalResult typeInfo(mlir::Location loc)
    {
        auto name = typeInfoExtRef;
        if (parentModule.lookupSymbol<LLVM::GlobalOp>(name))
        {
            return failure();
        }

        rewriter.create<LLVM::GlobalOp>(loc, th.getI8PtrType(), true, LLVM::Linkage::External, name, Attribute{});
        return success();
    }

    LogicalResult typeDescriptors(mlir::Location loc)
    {
        if (mlir::failed(typeDescriptor(loc, typeInfoRef, typeName)))
        {
            return mlir::failure();
        }

        if (type2)
        {
            return typeDescriptor(loc, typeInfoRef2, typeName2);
        }

        return mlir::success();
    }

    LogicalResult typeDescriptor(mlir::Location loc, StringRef typeInfoRefName, StringRef typeName)
    {
        auto name = typeInfoRefName;
        if (parentModule.lookupSymbol<LLVM::GlobalOp>(name))
        {
            return failure();
        }

        auto rttiTypeDescriptor2Ty = getRttiTypeDescriptor2Ty(StringRef(typeName).size());
        auto _r0n_Value = rewriter.create<LLVM::GlobalOp>(loc, rttiTypeDescriptor2Ty, false, LLVM::Linkage::LinkonceODR, name, Attribute{});

        {
            ch.setStructWritingPoint(_r0n_Value);

            // begin
            Value structVal = rewriter.create<LLVM::UndefOp>(loc, rttiTypeDescriptor2Ty);

            auto itemValue1 =
                rewriter.create<mlir::ConstantOp>(loc, th.getI8PtrPtrType(), FlatSymbolRefAttr::get(rewriter.getContext(), typeInfoExtRef));
            ch.setStructValue(loc, structVal, itemValue1, 0);

            auto itemValue2 = rewriter.create<LLVM::NullOp>(loc, th.getI8PtrType());
            ch.setStructValue(loc, structVal, itemValue2, 1);

            auto itemValue3 = rewriter.create<mlir::ConstantOp>(loc, th.getI8Array(StringRef(typeName).size() + 1),
                                                                ch.getStringAttrWith0(typeName.str()));
            ch.setStructValue(loc, structVal, itemValue3, 2);

            // end
            rewriter.create<LLVM::ReturnOp>(loc, ValueRange{structVal});

            rewriter.setInsertionPointAfter(_r0n_Value);
        }

        return success();
    }

    LogicalResult imageBase(mlir::Location loc)
    {
        auto name = imageBaseRef;
        if (parentModule.lookupSymbol<LLVM::GlobalOp>(name))
        {
            return failure();
        }

        rewriter.create<LLVM::GlobalOp>(loc, th.getI8Type(), true, LLVM::Linkage::External, name, Attribute{});
        return success();
    }

    LogicalResult catchableTypes(mlir::Location loc)
    {
        if (mlir::failed(catchableType(loc, catchableTypeInfoRef, typeInfoRef, typeName)))
        {
            return mlir::failure();
        }

        if (type2)
        {
            return catchableType(loc, catchableTypeInfoRef2, typeInfoRef2, typeName2);
        }

        return mlir::success();
    }

    LogicalResult catchableType(mlir::Location loc, StringRef catchableTypeInfoRefName, StringRef typeInfoRefName, StringRef typeName)
    {
        auto name = catchableTypeInfoRefName;
        if (parentModule.lookupSymbol<LLVM::GlobalOp>(name))
        {
            return failure();
        }

        // _CT??_R0N@88
        auto ehCatchableTypeTy = getCatchableTypeTy();
        auto _ct_r0n_Value = rewriter.create<LLVM::GlobalOp>(loc, ehCatchableTypeTy, true, LLVM::Linkage::LinkonceODR, name, Attribute{});

        {
            ch.setStructWritingPoint(_ct_r0n_Value);

            // begin
            Value structVal = rewriter.create<LLVM::UndefOp>(loc, ehCatchableTypeTy);

            auto itemValue1 = rewriter.create<mlir::ConstantOp>(loc, th.getI32Type(), rewriter.getI32IntegerAttr(1));
            ch.setStructValue(loc, structVal, itemValue1, 0);

            // value 2
            auto rttiTypeDescriptor2PtrValue =
                rewriter.create<mlir::ConstantOp>(loc, getRttiTypeDescriptor2PtrTy(StringRef(typeName).size()),
                                                  FlatSymbolRefAttr::get(rewriter.getContext(), typeInfoRefName));
            auto rttiTypeDescriptor2IntValue = rewriter.create<LLVM::PtrToIntOp>(loc, th.getI64Type(), rttiTypeDescriptor2PtrValue);

            auto imageBasePtrValue =
                rewriter.create<mlir::ConstantOp>(loc, th.getI8PtrType(), FlatSymbolRefAttr::get(rewriter.getContext(), imageBaseRef));
            auto imageBaseIntValue = rewriter.create<LLVM::PtrToIntOp>(loc, th.getI64Type(), imageBasePtrValue);

            // sub
            auto subResValue = rewriter.create<LLVM::SubOp>(loc, th.getI64Type(), rttiTypeDescriptor2IntValue, imageBaseIntValue);

            // trunc
            auto subRes32Value = rewriter.create<LLVM::TruncOp>(loc, th.getI32Type(), subResValue);

            auto itemValue2 = subRes32Value;
            ch.setStructValue(loc, structVal, itemValue2, 1);

            auto itemValue3 = rewriter.create<mlir::ConstantOp>(loc, th.getI32Type(), rewriter.getI32IntegerAttr(0));
            ch.setStructValue(loc, structVal, itemValue3, 2);

            auto itemValue4 = rewriter.create<mlir::ConstantOp>(loc, th.getI32Type(), rewriter.getI32IntegerAttr(-1));
            ch.setStructValue(loc, structVal, itemValue4, 3);

            auto itemValue5 = rewriter.create<mlir::ConstantOp>(loc, th.getI32Type(), rewriter.getI32IntegerAttr(0));
            ch.setStructValue(loc, structVal, itemValue5, 4);

            auto itemValue6 = rewriter.create<mlir::ConstantOp>(loc, th.getI32Type(), rewriter.getI32IntegerAttr(8));
            ch.setStructValue(loc, structVal, itemValue6, 5);

            auto itemValue7 = rewriter.create<mlir::ConstantOp>(loc, th.getI32Type(), rewriter.getI32IntegerAttr(0));
            ch.setStructValue(loc, structVal, itemValue7, 6);

            // end
            rewriter.create<LLVM::ReturnOp>(loc, ValueRange{structVal});

            rewriter.setInsertionPointAfter(_ct_r0n_Value);
        }

        return success();
    }

    mlir::Value catchableArrayTypeItem(mlir::Location loc, StringRef catchableTypeRefName)
    {
        auto rttiCatchableTypePtrValue = rewriter.create<mlir::ConstantOp>(
            loc, getCatchableTypePtrTy(), FlatSymbolRefAttr::get(rewriter.getContext(), catchableTypeRefName));
        auto rttiCatchableTypeIntValue = rewriter.create<LLVM::PtrToIntOp>(loc, th.getI64Type(), rttiCatchableTypePtrValue);

        auto imageBasePtrValue =
            rewriter.create<mlir::ConstantOp>(loc, th.getI8PtrType(), FlatSymbolRefAttr::get(rewriter.getContext(), imageBaseRef));
        auto imageBaseIntValue = rewriter.create<LLVM::PtrToIntOp>(loc, th.getI64Type(), imageBasePtrValue);

        // sub
        auto subResValue = rewriter.create<LLVM::SubOp>(loc, th.getI64Type(), rttiCatchableTypeIntValue, imageBaseIntValue);

        // trunc
        auto subRes32Value = rewriter.create<LLVM::TruncOp>(loc, th.getI32Type(), subResValue);

        return subRes32Value;
    }

    LogicalResult catchableArrayType(mlir::Location loc)
    {
        auto name = catchableTypeInfoArrayRef;
        if (parentModule.lookupSymbol<LLVM::GlobalOp>(name))
        {
            return failure();
        }

        auto arraySize = type2 ? 2 : 1;

        // _CT??_R0N@88
        auto ehCatchableArrayTypeTy = getCatchableArrayTypeTy(arraySize);
        auto _cta1nValue =
            rewriter.create<LLVM::GlobalOp>(loc, ehCatchableArrayTypeTy, true, LLVM::Linkage::LinkonceODR, name, Attribute{});

        {
            ch.setStructWritingPoint(_cta1nValue);

            // begin
            Value structVal = rewriter.create<LLVM::UndefOp>(loc, ehCatchableArrayTypeTy);

            auto sizeValue = rewriter.create<mlir::ConstantOp>(loc, th.getI32Type(), rewriter.getI32IntegerAttr(arraySize));
            ch.setStructValue(loc, structVal, sizeValue, 0);

            // value 2
            auto value1 = catchableArrayTypeItem(loc, catchableTypeInfoRef);
            mlir::Value value2;
            if (type2)
            {
                value2 = catchableArrayTypeItem(loc, catchableTypeInfoRef2);
            }

            // make array
            Value arrayVal = rewriter.create<LLVM::UndefOp>(loc, th.getArrayType(th.getI32Type(), arraySize));
            ch.setStructValue(loc, arrayVal, value1, 0);
            if (type2)
            {
                ch.setStructValue(loc, arrayVal, value2, 1);
            }

            // [size, {values...}]
            ch.setStructValue(loc, structVal, arrayVal, 1);

            // end
            rewriter.create<LLVM::ReturnOp>(loc, ValueRange{structVal});

            rewriter.setInsertionPointAfter(_cta1nValue);
        }

        return success();
    }

    LogicalResult throwInfo(mlir::Location loc)
    {
        auto name = throwInfoRef;
        if (parentModule.lookupSymbol<LLVM::GlobalOp>(name))
        {
            return failure();
        }

        auto arraySize = type2 ? 2 : 1;

        auto throwInfoTy = getThrowInfoTy();
        auto _TI1NValue = rewriter.create<LLVM::GlobalOp>(loc, throwInfoTy, true, LLVM::Linkage::LinkonceODR, name, Attribute{});

        // Throw Info
        ch.setStructWritingPoint(_TI1NValue);

        Value structValue =
            ch.getStructFromArrayAttr(loc, throwInfoTy,
                                      rewriter.getArrayAttr({rewriter.getI32IntegerAttr(type2 ? 1 : 0), rewriter.getI32IntegerAttr(0),
                                                             rewriter.getI32IntegerAttr(0)}));

        // value 3
        auto rttiCatchableArrayTypePtrValue = rewriter.create<mlir::ConstantOp>(
            loc, getCatchableArrayTypePtrTy(arraySize), FlatSymbolRefAttr::get(rewriter.getContext(), catchableTypeInfoArrayRef));
        auto rttiCatchableArrayTypeIntValue = rewriter.create<LLVM::PtrToIntOp>(loc, th.getI64Type(), rttiCatchableArrayTypePtrValue);

        auto imageBasePtrValue =
            rewriter.create<mlir::ConstantOp>(loc, th.getI8PtrType(), FlatSymbolRefAttr::get(rewriter.getContext(), imageBaseRef));
        auto imageBaseIntValue = rewriter.create<LLVM::PtrToIntOp>(loc, th.getI64Type(), imageBasePtrValue);

        // sub
        auto subResValue = rewriter.create<LLVM::SubOp>(loc, th.getI64Type(), rttiCatchableArrayTypeIntValue, imageBaseIntValue);

        // trunc
        auto subRes32Value = rewriter.create<LLVM::TruncOp>(loc, th.getI32Type(), subResValue);
        ch.setStructValue(loc, structValue, subRes32Value, 3);

        rewriter.create<LLVM::ReturnOp>(loc, ValueRange{structValue});

        rewriter.setInsertionPointAfter(_TI1NValue);

        return success();
    }

    mlir::Value typeInfoPtrValue(mlir::Location loc)
    {
        auto typeInfoPtr = rewriter.create<mlir::ConstantOp>(loc, getRttiTypeDescriptor2PtrTy(StringRef(typeName).size()),
                                                             FlatSymbolRefAttr::get(rewriter.getContext(), typeInfoRef));
        return typeInfoPtr;
    }

    mlir::Value throwInfoPtrValue(mlir::Location loc)
    {
        auto throwInfoPtr =
            rewriter.create<mlir::ConstantOp>(loc, getThrowInfoPtrTy(), FlatSymbolRefAttr::get(rewriter.getContext(), throwInfoRef));
        return throwInfoPtr;
    }

    LLVM::LLVMStructType getThrowInfoTy()
    {
        return LLVM::LLVMStructType::getLiteral(rewriter.getContext(), {th.getI32Type(), th.getI32Type(), th.getI32Type(), th.getI32Type()},
                                                false);
    }

    LLVM::LLVMPointerType getThrowInfoPtrTy()
    {
        return LLVM::LLVMPointerType::get(getThrowInfoTy());
    }

    LLVM::LLVMStructType getRttiTypeDescriptor2Ty(int nameSize)
    {
        return LLVM::LLVMStructType::getLiteral(rewriter.getContext(),
                                                {th.getI8PtrPtrType(), th.getI8PtrType(), th.getI8Array(nameSize + 1)}, false);
    }

    LLVM::LLVMPointerType getRttiTypeDescriptor2PtrTy(int nameSize)
    {
        return LLVM::LLVMPointerType::get(getRttiTypeDescriptor2Ty(nameSize));
    }

    LLVM::LLVMStructType getCatchableTypeTy()
    {
        return LLVM::LLVMStructType::getLiteral(
            rewriter.getContext(),
            {th.getI32Type(), th.getI32Type(), th.getI32Type(), th.getI32Type(), th.getI32Type(), th.getI32Type(), th.getI32Type()}, false);
    }

    LLVM::LLVMPointerType getCatchableTypePtrTy()
    {
        return LLVM::LLVMPointerType::get(getCatchableTypeTy());
    }

    LLVM::LLVMStructType getCatchableArrayTypeTy(int arraySize)
    {
        return LLVM::LLVMStructType::getLiteral(rewriter.getContext(), {th.getI32Type(), th.getI32Array(arraySize)}, false);
    }

    LLVM::LLVMPointerType getCatchableArrayTypePtrTy(int arraySize)
    {
        return LLVM::LLVMPointerType::get(getCatchableArrayTypeTy(arraySize));
    }
};
} // namespace typescript

#endif // MLIR_TYPESCRIPT_LOWERTOLLVMLOGIC_LLVMRTTIHELPERVCWIN32_H_
