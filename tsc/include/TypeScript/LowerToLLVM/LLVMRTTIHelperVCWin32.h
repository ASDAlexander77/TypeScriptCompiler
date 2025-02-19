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

#include <sstream>

using namespace mlir;
namespace mlir_ts = mlir::typescript;

namespace typescript
{

using namespace windows;

class LLVMRTTIHelperVCWin32
{
    struct TypeNames
    {
        std::string typeName;
        std::string typeInfoRef;
        std::string catchableTypeInfoRef;
    };

    Operation *op;
    PatternRewriter &rewriter;
    ModuleOp parentModule;
    TypeHelper th;
    LLVMCodeHelper ch;

    SmallVector<TypeNames> types;

  public:
    std::string catchableTypeInfoArrayRef;
    std::string throwInfoRef;

    LLVMRTTIHelperVCWin32(Operation *op, PatternRewriter &rewriter, TypeConverter &typeConverter, CompileOptions &compileOptions)
        : op(op), rewriter(rewriter), parentModule(op->getParentOfType<ModuleOp>()), th(rewriter), ch(op, rewriter, &typeConverter, compileOptions)
    {
        // setI32AsCatchType();
    }

    void setF32AsCatchType()
    {
        types.push_back({F32Type::typeName, F32Type::typeInfoRef, F32Type::catchableTypeInfoRef});

        catchableTypeInfoArrayRef = F32Type::catchableTypeInfoArrayRef;
        throwInfoRef = F32Type::throwInfoRef;
    }

    void setF64AsCatchType()
    {
        types.push_back({F64Type::typeName, F64Type::typeInfoRef, F64Type::catchableTypeInfoRef});

        catchableTypeInfoArrayRef = F64Type::catchableTypeInfoArrayRef;
        throwInfoRef = F64Type::throwInfoRef;
    }

    void setI32AsCatchType()
    {
        types.push_back({I32Type::typeName, I32Type::typeInfoRef, I32Type::catchableTypeInfoRef});

        catchableTypeInfoArrayRef = I32Type::catchableTypeInfoArrayRef;
        throwInfoRef = I32Type::throwInfoRef;
    }

    void setStringTypeAsCatchType()
    {
        types.push_back({StringType::typeName, StringType::typeInfoRef, StringType::catchableTypeInfoRef});
        types.push_back({StringType::typeName2, StringType::typeInfoRef2, StringType::catchableTypeInfoRef2});

        catchableTypeInfoArrayRef = StringType::catchableTypeInfoArrayRef;
        throwInfoRef = StringType::throwInfoRef;
    }

    void setI8PtrAsCatchType()
    {
        types.push_back({I8PtrType::typeName, I8PtrType::typeInfoRef, I8PtrType::catchableTypeInfoRef});

        catchableTypeInfoArrayRef = I8PtrType::catchableTypeInfoArrayRef;
        throwInfoRef = I8PtrType::throwInfoRef;
    }

    void setClassTypeAsCatchType(StringRef name)
    {
        types.push_back({join(name, ClassType::typeName, ClassType::typeNameSuffix),
                         join(name, ClassType::typeInfoRef, ClassType::typeInfoRefSuffix),
                         join(name, ClassType::catchableTypeInfoRef, ClassType::catchableTypeInfoRefSuffix)});

        types.push_back({ClassType::typeName2, ClassType::typeInfoRef2, ClassType::catchableTypeInfoRef2});

        catchableTypeInfoArrayRef = ClassType::catchableTypeInfoArrayRef;
        throwInfoRef = ClassType::throwInfoRef;
    }

    std::string join(StringRef name, const char *prefix, const char *suffix)
    {
        std::stringstream ss;
        ss << prefix;
        ss << name.str();
        ss << suffix;
        return ss.str();
    }

    LogicalResult setPersonality(mlir::func::FuncOp newFuncOp)
    {
        auto name = "__CxxFrameHandler3";
        auto cxxFrameHandler3 = ch.getOrInsertFunction(name, th.getFunctionType(th.getI32Type(), {}, true));

        newFuncOp->setAttr(StringAttr::get(rewriter.getContext(), "personality"), FlatSymbolRefAttr::get(rewriter.getContext(), name));
        return success();
    }

    void setType(mlir::Type type)
    {
        auto normalizedType = MLIRHelper::stripLiteralType(type);
        if (auto enumType = normalizedType.dyn_cast<mlir_ts::EnumType>())
        {
            normalizedType = enumType.getElementType();
        }

        mlir::TypeSwitch<mlir::Type>(normalizedType)
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
                auto width = floatType.getIntOrFloatBitWidth();
                if (width == 32)
                {
                    setF32AsCatchType();
                }
                else if (width == 64)
                {
                    setF64AsCatchType();
                }
                else
                {
                    llvm_unreachable("not implemented");
                }
            })
            .Case<mlir_ts::NumberType>([&](auto numberType) {
#ifdef NUMBER_F64
                setF32AsCatchType();
#else
                setF64AsCatchType();
#endif
            })
            .Case<mlir_ts::StringType>([&](auto stringType) { setStringTypeAsCatchType(); })
            .Case<mlir_ts::ClassType>([&](auto classType) { setClassTypeAsCatchType(classType.getName().getValue()); })
            .Case<mlir_ts::AnyType>([&](auto anyType) { setI8PtrAsCatchType(); })
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

        rewriter.create<LLVM::GlobalOp>(loc, th.getI8PtrType(), true, LLVM::Linkage::External, name, mlir::Attribute{});
        return success();
    }

    LogicalResult typeDescriptors(mlir::Location loc)
    {
        for (auto type : types)
        {
            if (mlir::failed(typeDescriptor(loc, type.typeInfoRef, type.typeName)))
            {
                return mlir::failure();
            }
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
        auto _r0n_Value =
            rewriter.create<LLVM::GlobalOp>(loc, rttiTypeDescriptor2Ty, false, LLVM::Linkage::LinkonceODR, name, mlir::Attribute{});

        {
            ch.setStructWritingPoint(_r0n_Value);

            // begin
            mlir::Value structVal = rewriter.create<LLVM::UndefOp>(loc, rttiTypeDescriptor2Ty);

            auto itemValue1 =
                rewriter.create<LLVM::AddressOfOp>(loc, th.getI8PtrPtrType(), FlatSymbolRefAttr::get(rewriter.getContext(), typeInfoExtRef));
            ch.setStructValue(loc, structVal, itemValue1, 0);

            auto itemValue2 = rewriter.create<LLVM::NullOp>(loc, th.getI8PtrType());
            ch.setStructValue(loc, structVal, itemValue2, 1);

            auto itemValue3 = rewriter.create<LLVM::ConstantOp>(loc, th.getI8Array(StringRef(typeName).size() + 1),
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

        rewriter.create<LLVM::GlobalOp>(loc, th.getI8Type(), true, LLVM::Linkage::External, name, mlir::Attribute{});
        return success();
    }

    LogicalResult catchableTypes(mlir::Location loc)
    {
        for (auto type : types)
        {
            if (mlir::failed(catchableType(loc, type.catchableTypeInfoRef, type.typeInfoRef, type.typeName)))
            {
                return mlir::failure();
            }
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
        auto _ct_r0n_Value =
            rewriter.create<LLVM::GlobalOp>(loc, ehCatchableTypeTy, true, LLVM::Linkage::LinkonceODR, name, mlir::Attribute{});

        {
            ch.setStructWritingPoint(_ct_r0n_Value);

            // begin
            mlir::Value structVal = rewriter.create<LLVM::UndefOp>(loc, ehCatchableTypeTy);

            auto itemValue1 = rewriter.create<LLVM::ConstantOp>(loc, th.getI32Type(), rewriter.getI32IntegerAttr(1));
            ch.setStructValue(loc, structVal, itemValue1, 0);

            // value 2
            auto rttiTypeDescriptor2PtrValue =
                rewriter.create<LLVM::AddressOfOp>(loc, getRttiTypeDescriptor2PtrTy(StringRef(typeName).size()),
                                                  FlatSymbolRefAttr::get(rewriter.getContext(), typeInfoRefName));
            auto rttiTypeDescriptor2IntValue = rewriter.create<LLVM::PtrToIntOp>(loc, th.getI64Type(), rttiTypeDescriptor2PtrValue);

            auto imageBasePtrValue =
                rewriter.create<LLVM::AddressOfOp>(loc, th.getI8PtrType(), FlatSymbolRefAttr::get(rewriter.getContext(), imageBaseRef));
            auto imageBaseIntValue = rewriter.create<LLVM::PtrToIntOp>(loc, th.getI64Type(), imageBasePtrValue);

            // sub
            auto subResValue = rewriter.create<LLVM::SubOp>(loc, th.getI64Type(), rttiTypeDescriptor2IntValue, imageBaseIntValue);

            // trunc
            auto subRes32Value = rewriter.create<LLVM::TruncOp>(loc, th.getI32Type(), subResValue);

            auto itemValue2 = subRes32Value;
            ch.setStructValue(loc, structVal, itemValue2, 1);

            auto itemValue3 = rewriter.create<LLVM::ConstantOp>(loc, th.getI32Type(), rewriter.getI32IntegerAttr(0));
            ch.setStructValue(loc, structVal, itemValue3, 2);

            auto itemValue4 = rewriter.create<LLVM::ConstantOp>(loc, th.getI32Type(), rewriter.getI32IntegerAttr(-1));
            ch.setStructValue(loc, structVal, itemValue4, 3);

            auto itemValue5 = rewriter.create<LLVM::ConstantOp>(loc, th.getI32Type(), rewriter.getI32IntegerAttr(0));
            ch.setStructValue(loc, structVal, itemValue5, 4);

            auto itemValue6 = rewriter.create<LLVM::ConstantOp>(loc, th.getI32Type(), rewriter.getI32IntegerAttr(8));
            ch.setStructValue(loc, structVal, itemValue6, 5);

            auto itemValue7 = rewriter.create<LLVM::ConstantOp>(loc, th.getI32Type(), rewriter.getI32IntegerAttr(0));
            ch.setStructValue(loc, structVal, itemValue7, 6);

            // end
            rewriter.create<LLVM::ReturnOp>(loc, ValueRange{structVal});

            rewriter.setInsertionPointAfter(_ct_r0n_Value);
        }

        return success();
    }

    mlir::Value catchableArrayTypeItem(mlir::Location loc, StringRef catchableTypeRefName)
    {
        auto rttiCatchableTypePtrValue = rewriter.create<LLVM::AddressOfOp>(
            loc, getCatchableTypePtrTy(), FlatSymbolRefAttr::get(rewriter.getContext(), catchableTypeRefName));
        auto rttiCatchableTypeIntValue = rewriter.create<LLVM::PtrToIntOp>(loc, th.getI64Type(), rttiCatchableTypePtrValue);

        auto imageBasePtrValue =
            rewriter.create<LLVM::AddressOfOp>(loc, th.getI8PtrType(), FlatSymbolRefAttr::get(rewriter.getContext(), imageBaseRef));
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

        auto arraySize = types.size();

        // _CT??_R0N@88
        auto ehCatchableArrayTypeTy = getCatchableArrayTypeTy(arraySize);
        auto _cta1nValue =
            rewriter.create<LLVM::GlobalOp>(loc, ehCatchableArrayTypeTy, true, LLVM::Linkage::LinkonceODR, name, mlir::Attribute{});

        {
            ch.setStructWritingPoint(_cta1nValue);

            // begin
            mlir::Value structVal = rewriter.create<LLVM::UndefOp>(loc, ehCatchableArrayTypeTy);

            auto sizeValue = rewriter.create<LLVM::ConstantOp>(loc, th.getI32Type(), rewriter.getI32IntegerAttr(arraySize));
            ch.setStructValue(loc, structVal, sizeValue, 0);

            // value 2
            SmallVector<mlir::Value> values;
            for (auto type : types)
            {
                auto value = catchableArrayTypeItem(loc, type.catchableTypeInfoRef);
                values.push_back(value);
            }

            // make array
            mlir::Value arrayVal = rewriter.create<LLVM::UndefOp>(loc, th.getArrayType(th.getI32Type(), arraySize));

            for (auto [index, value] : enumerate(values))
            {
                ch.setStructValue(loc, arrayVal, value, index);
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

        auto arraySize = types.size();

        auto throwInfoTy = getThrowInfoTy();
        auto _TI1NValue = rewriter.create<LLVM::GlobalOp>(loc, throwInfoTy, true, LLVM::Linkage::LinkonceODR, name, mlir::Attribute{});

        // Throw Info
        ch.setStructWritingPoint(_TI1NValue);

        mlir::Value structValue = ch.getStructFromArrayAttr(
            loc, throwInfoTy,
            rewriter.getArrayAttr({rewriter.getI32IntegerAttr(0), rewriter.getI32IntegerAttr(0), rewriter.getI32IntegerAttr(0)}));

        // value 3
        auto rttiCatchableArrayTypePtrValue = rewriter.create<LLVM::AddressOfOp>(
            loc, getCatchableArrayTypePtrTy(arraySize), FlatSymbolRefAttr::get(rewriter.getContext(), catchableTypeInfoArrayRef));
        auto rttiCatchableArrayTypeIntValue = rewriter.create<LLVM::PtrToIntOp>(loc, th.getI64Type(), rttiCatchableArrayTypePtrValue);

        auto imageBasePtrValue =
            rewriter.create<LLVM::AddressOfOp>(loc, th.getI8PtrType(), FlatSymbolRefAttr::get(rewriter.getContext(), imageBaseRef));
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

    bool hasType()
    {
        return types.size() > 0;
    }

    mlir::Value typeInfoPtrValue(mlir::Location loc)
    {
        auto firstType = types.front();
        auto typeInfoPtr = rewriter.create<LLVM::AddressOfOp>(loc, getRttiTypeDescriptor2PtrTy(StringRef(firstType.typeName).size()),
                                                             FlatSymbolRefAttr::get(rewriter.getContext(), firstType.typeInfoRef));
        return typeInfoPtr;
    }

    mlir::Value throwInfoPtrValue(mlir::Location loc)
    {
        auto throwInfoPtr =
            rewriter.create<LLVM::AddressOfOp>(loc, getThrowInfoPtrTy(), FlatSymbolRefAttr::get(rewriter.getContext(), throwInfoRef));
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
