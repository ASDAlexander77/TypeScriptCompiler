#ifndef MLIR_TYPESCRIPT_LOWERTOLLVMLOGIC_MLIRRTTIHELPERVCWIN32_H_
#define MLIR_TYPESCRIPT_LOWERTOLLVMLOGIC_MLIRRTTIHELPERVCWIN32_H_

#include "TypeScript/Config.h"
#include "TypeScript/Defines.h"
#include "TypeScript/Passes.h"
#include "TypeScript/TypeScriptDialect.h"
#include "TypeScript/TypeScriptOps.h"

#include "TypeScript/MLIRLogic/MLIRTypeHelper.h"
#include "TypeScript/MLIRLogic/MLIRCodeLogic.h"
#include "TypeScript/LowerToLLVM/LLVMRTTIHelperVCWin32Const.h"

#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/TypeSwitch.h"

#include <sstream>

using namespace ::typescript;
using namespace ts;
namespace mlir_ts = mlir::typescript;

namespace typescript
{

constexpr auto typeInfoExtRef = "??_7type_info@@6B@";
constexpr auto imageBaseRef = "__ImageBase";

struct TypeNames
{
    std::string typeName;
    std::string typeInfoRef;
    std::string catchableTypeInfoRef;
};

class MLIRRTTIHelperVCWin32
{
    mlir::Operation *op;
    mlir::OpBuilder &rewriter;
    MLIRTypeHelper mth;
    MLIRLogicHelper mlh;
    mlir::ModuleOp parentModule;

    SmallVector<TypeNames> types;

  public:
    std::string catchableTypeInfoArrayRef;
    std::string throwInfoRef;

    MLIRRTTIHelperVCWin32(mlir::Operation *op, mlir::OpBuilder &rewriter, mlir::TypeConverter &typeConverter)
        : op(op), rewriter(rewriter), mth(rewriter.getContext()), parentModule(op->getParentOfType<mlir::ModuleOp>())
    {
        // setI32AsCatchType();
    }

    void setF32AsCatchType()
    {
        types.push_back({F32Type::typeName, F32Type::typeInfoRef, F32Type::catchableTypeInfoRef});

        catchableTypeInfoArrayRef = F32Type::catchableTypeInfoArrayRef;
        throwInfoRef = F32Type::throwInfoRef;
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

    void setType(mlir::Type type)
    {
        llvm::TypeSwitch<mlir::Type>(type)
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
            .Case<mlir_ts::ClassType>([&](auto classType) { setClassTypeAsCatchType(classType.getName().getValue()); })
            .Case<mlir_ts::AnyType>([&](auto anyType) { setI8PtrAsCatchType(); })
            .Default([&](auto type) { llvm_unreachable("not implemented"); });
    }

    void seekLast(mlir::Block *block)
    {
        // find last string
        auto lastUse = [&](mlir::Operation *op) {
            if (auto globalOp = dyn_cast_or_null<mlir_ts::GlobalOp>(op))
            {
                rewriter.setInsertionPointAfter(globalOp);
            }
        };

        block->walk(lastUse);
    }

    void setRTTIForType(mlir::Location loc, mlir::Type type)
    {
        setType(type);

        auto parentModule = op->getParentOfType<mlir::ModuleOp>();

        mlir::OpBuilder::InsertionGuard guard(rewriter);

        rewriter.setInsertionPointToStart(parentModule.getBody());
        seekLast(parentModule.getBody());

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

    mlir::LogicalResult typeInfo(mlir::Location loc)
    {
        auto name = typeInfoExtRef;
        if (parentModule.lookupSymbol<mlir_ts::GlobalOp>(name))
        {
            return mlir::failure();
        }

        rewriter.create<mlir_ts::GlobalOp>(loc, mth.getOpaqueType(), true, /*LLVM::Linkage::External,*/ name, mlir::Attribute{});
        return mlir::success();
    }

    mlir::LogicalResult typeDescriptors(mlir::Location loc)
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

    void setGlobalOpWritingPoint(mlir_ts::GlobalOp globalOp)
    {
        auto &region = globalOp.getInitializerRegion();
        auto *block = rewriter.createBlock(&region);

        rewriter.setInsertionPoint(block, block->begin());
    }

    mlir::LogicalResult setStructValue(mlir::Location loc, mlir::Value &tupleValue, mlir::Value value, int index)
    {
        auto tpl = tupleValue.getType().cast<mlir::TupleType>();
        tupleValue = rewriter.create<mlir_ts::InsertPropertyOp>(loc, tpl, value, tupleValue, rewriter.getI64ArrayAttr(index));
        return mlir::success();
    }

    mlir::LogicalResult typeDescriptor(mlir::Location loc, StringRef typeInfoRefName, StringRef typeName)
    {
        auto name = typeInfoRefName;
        if (parentModule.lookupSymbol<mlir_ts::GlobalOp>(name))
        {
            return mlir::failure();
        }

        auto rttiTypeDescriptor2Ty = getRttiTypeDescriptor2Ty(StringRef(typeName).size());
        auto _r0n_Value =
            rewriter.create<mlir_ts::GlobalOp>(loc, rttiTypeDescriptor2Ty, false, /*LLVM::Linkage::LinkonceODR,*/ name, mlir::Attribute{});

        {
            setGlobalOpWritingPoint(_r0n_Value);

            // begin
            mlir::Value structVal = rewriter.create<mlir_ts::UndefOp>(loc, rttiTypeDescriptor2Ty);

            auto itemValue1 = rewriter.create<mlir_ts::ConstantOp>(loc, mth.getRefType(mth.getOpaqueType()),
                                                                   mlir::FlatSymbolRefAttr::get(rewriter.getContext(), typeInfoExtRef));
            setStructValue(loc, structVal, itemValue1, 0);

            auto itemValue2 = rewriter.create<mlir_ts::NullOp>(loc, mth.getOpaqueType());
            setStructValue(loc, structVal, itemValue2, 1);

            auto itemValue3 = rewriter.create<mlir_ts::ConstantOp>(loc, mth.getStringType(), rewriter.getStringAttr(typeName));
            setStructValue(loc, structVal, itemValue3, 2);

            // end
            rewriter.create<mlir_ts::GlobalResultOp>(loc, mlir::ValueRange{structVal});

            rewriter.setInsertionPointAfter(_r0n_Value);
        }

        return mlir::success();
    }

    mlir::LogicalResult imageBase(mlir::Location loc)
    {
        auto name = imageBaseRef;
        if (parentModule.lookupSymbol<mlir_ts::GlobalOp>(name))
        {
            return mlir::failure();
        }

        rewriter.create<mlir_ts::GlobalOp>(loc, mth.getOpaqueType(), true, /*LLVM::Linkage::External,*/ name, mlir::Attribute{});
        return mlir::success();
    }

    mlir::LogicalResult catchableTypes(mlir::Location loc)
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

    mlir::LogicalResult catchableType(mlir::Location loc, StringRef catchableTypeInfoRefName, StringRef typeInfoRefName, StringRef typeName)
    {
        auto name = catchableTypeInfoRefName;
        if (parentModule.lookupSymbol<mlir_ts::GlobalOp>(name))
        {
            return mlir::failure();
        }

        // _CT??_R0N@88
        auto ehCatchableTypeTy = getCatchableTypeTy();
        auto _ct_r0n_Value =
            rewriter.create<mlir_ts::GlobalOp>(loc, ehCatchableTypeTy, true, /*LLVM::Linkage::LinkonceODR,*/ name, mlir::Attribute{});

        {
            setGlobalOpWritingPoint(_ct_r0n_Value);

            // begin
            mlir::Value structVal = rewriter.create<mlir_ts::UndefOp>(loc, ehCatchableTypeTy);

            auto itemValue1 = rewriter.create<mlir_ts::ConstantOp>(loc, mth.getI32Type(), rewriter.getI32IntegerAttr(1));
            setStructValue(loc, structVal, itemValue1, 0);

            // value 2
            auto rttiTypeDescriptor2PtrValue =
                rewriter.create<mlir_ts::ConstantOp>(loc, getRttiTypeDescriptor2PtrTy(StringRef(typeName).size()),
                                                     mlir::FlatSymbolRefAttr::get(rewriter.getContext(), typeInfoRefName));
            auto rttiTypeDescriptor2IntValue = rewriter.create<mlir_ts::CastOp>(loc, mth.getI64Type(), rttiTypeDescriptor2PtrValue);

            auto imageBasePtrValue = rewriter.create<mlir_ts::ConstantOp>(
                loc, mth.getOpaqueType(), mlir::FlatSymbolRefAttr::get(rewriter.getContext(), imageBaseRef));
            auto imageBaseIntValue = rewriter.create<mlir_ts::CastOp>(loc, mth.getI64Type(), imageBasePtrValue);

            // sub
            auto subResValue = rewriter.create<mlir_ts::ArithmeticBinaryOp>(
                loc, mth.getI64Type(), rewriter.getI32IntegerAttr(static_cast<int32_t>(SyntaxKind::MinusToken)),
                rttiTypeDescriptor2IntValue, imageBaseIntValue);

            // trunc
            auto subRes32Value = rewriter.create<mlir_ts::CastOp>(loc, mth.getI32Type(), subResValue);

            auto itemValue2 = subRes32Value;
            setStructValue(loc, structVal, itemValue2, 1);

            auto itemValue3 = rewriter.create<mlir_ts::ConstantOp>(loc, mth.getI32Type(), rewriter.getI32IntegerAttr(0));
            setStructValue(loc, structVal, itemValue3, 2);

            auto itemValue4 = rewriter.create<mlir_ts::ConstantOp>(loc, mth.getI32Type(), rewriter.getI32IntegerAttr(-1));
            setStructValue(loc, structVal, itemValue4, 3);

            auto itemValue5 = rewriter.create<mlir_ts::ConstantOp>(loc, mth.getI32Type(), rewriter.getI32IntegerAttr(0));
            setStructValue(loc, structVal, itemValue5, 4);

            auto itemValue6 = rewriter.create<mlir_ts::ConstantOp>(loc, mth.getI32Type(), rewriter.getI32IntegerAttr(8));
            setStructValue(loc, structVal, itemValue6, 5);

            auto itemValue7 = rewriter.create<mlir_ts::ConstantOp>(loc, mth.getI32Type(), rewriter.getI32IntegerAttr(0));
            setStructValue(loc, structVal, itemValue7, 6);

            // end
            rewriter.create<mlir_ts::GlobalResultOp>(loc, mlir::ValueRange{structVal});

            rewriter.setInsertionPointAfter(_ct_r0n_Value);
        }

        return mlir::success();
    }

    mlir::Value catchableArrayTypeItem(mlir::Location loc, StringRef catchableTypeRefName)
    {
        auto rttiCatchableTypePtrValue = rewriter.create<mlir_ts::ConstantOp>(
            loc, getCatchableTypePtrTy(), mlir::FlatSymbolRefAttr::get(rewriter.getContext(), catchableTypeRefName));
        auto rttiCatchableTypeIntValue = rewriter.create<mlir_ts::CastOp>(loc, mth.getI64Type(), rttiCatchableTypePtrValue);

        auto imageBasePtrValue = rewriter.create<mlir_ts::ConstantOp>(loc, mth.getOpaqueType(),
                                                                      mlir::FlatSymbolRefAttr::get(rewriter.getContext(), imageBaseRef));
        auto imageBaseIntValue = rewriter.create<mlir_ts::CastOp>(loc, mth.getI64Type(), imageBasePtrValue);

        // sub
        auto subResValue = rewriter.create<mlir_ts::ArithmeticBinaryOp>(
            loc, mth.getI64Type(), rewriter.getI32IntegerAttr(static_cast<int32_t>(SyntaxKind::MinusToken)), rttiCatchableTypeIntValue,
            imageBaseIntValue);

        // trunc
        auto subRes32Value = rewriter.create<mlir_ts::CastOp>(loc, mth.getI32Type(), subResValue);

        return subRes32Value;
    }

    mlir::LogicalResult catchableArrayType(mlir::Location loc)
    {
        auto name = catchableTypeInfoArrayRef;
        if (parentModule.lookupSymbol<mlir_ts::GlobalOp>(name))
        {
            return mlir::failure();
        }

        auto arraySize = types.size();

        // _CT??_R0N@88
        auto ehCatchableArrayTypeTy = getCatchableArrayTypeTy(arraySize);
        auto _cta1nValue =
            rewriter.create<mlir_ts::GlobalOp>(loc, ehCatchableArrayTypeTy, true, /*LLVM::Linkage::LinkonceODR,*/ name, mlir::Attribute{});

        {
            setGlobalOpWritingPoint(_cta1nValue);

            // begin
            mlir::Value structVal = rewriter.create<mlir_ts::UndefOp>(loc, ehCatchableArrayTypeTy);

            auto sizeValue = rewriter.create<mlir_ts::ConstantOp>(loc, mth.getI32Type(), rewriter.getI32IntegerAttr(arraySize));
            setStructValue(loc, structVal, sizeValue, 0);

            // value 2
            SmallVector<mlir::Value> values;
            for (auto type : types)
            {
                auto value = catchableArrayTypeItem(loc, type.catchableTypeInfoRef);
                values.push_back(value);
            }

            // make array
            mlir::Value arrayVal = rewriter.create<mlir_ts::UndefOp>(loc, mth.getConstArrayType(mth.getI32Type(), arraySize));

            auto index = 0;
            for (auto value : values)
            {
                setStructValue(loc, arrayVal, value, index++);
            }

            // [size, {values...}]
            setStructValue(loc, structVal, arrayVal, 1);

            // end
            rewriter.create<mlir_ts::GlobalResultOp>(loc, mlir::ValueRange{structVal});

            rewriter.setInsertionPointAfter(_cta1nValue);
        }

        return mlir::success();
    }

    mlir::Value getTupleFromArrayAttr(mlir::Location loc, mlir_ts::TupleType tupleType, mlir::ArrayAttr arrayAttr)
    {
        mlir::Value structVal = rewriter.create<mlir_ts::UndefOp>(loc, tupleType);

        auto position = 0;
        for (auto item : arrayAttr.getValue())
        {
            auto fieldType = tupleType.getType(position);

            // DO NOT Replace with LLVM::ConstantOp - to use AddressOf for global symbol names
            auto itemValue = rewriter.create<mlir_ts::ConstantOp>(loc, fieldType, item);
            structVal =
                rewriter.create<mlir_ts::InsertPropertyOp>(loc, tupleType, itemValue, structVal, rewriter.getI64ArrayAttr(position++));
        }

        return structVal;
    }

    mlir::LogicalResult throwInfo(mlir::Location loc)
    {
        auto name = throwInfoRef;
        if (parentModule.lookupSymbol<mlir_ts::GlobalOp>(name))
        {
            return mlir::failure();
        }

        auto arraySize = types.size();

        auto throwInfoTy = getThrowInfoTy();
        auto _TI1NValue =
            rewriter.create<mlir_ts::GlobalOp>(loc, throwInfoTy, true, /*LLVM::Linkage::LinkonceODR,*/ name, mlir::Attribute{});

        // Throw Info
        setGlobalOpWritingPoint(_TI1NValue);

        mlir::Value structValue = getTupleFromArrayAttr(
            loc, throwInfoTy,
            rewriter.getArrayAttr({rewriter.getI32IntegerAttr(0), rewriter.getI32IntegerAttr(0), rewriter.getI32IntegerAttr(0)}));

        // value 3
        auto rttiCatchableArrayTypePtrValue = rewriter.create<mlir_ts::ConstantOp>(
            loc, getCatchableArrayTypePtrTy(arraySize), mlir::FlatSymbolRefAttr::get(rewriter.getContext(), catchableTypeInfoArrayRef));
        auto rttiCatchableArrayTypeIntValue = rewriter.create<mlir_ts::CastOp>(loc, mth.getI64Type(), rttiCatchableArrayTypePtrValue);

        auto imageBasePtrValue = rewriter.create<mlir_ts::ConstantOp>(loc, mth.getOpaqueType(),
                                                                      mlir::FlatSymbolRefAttr::get(rewriter.getContext(), imageBaseRef));
        auto imageBaseIntValue = rewriter.create<mlir_ts::CastOp>(loc, mth.getI64Type(), imageBasePtrValue);

        // sub
        auto subResValue = rewriter.create<mlir_ts::ArithmeticBinaryOp>(
            loc, mth.getI64Type(), rewriter.getI32IntegerAttr(static_cast<int32_t>(SyntaxKind::MinusToken)), rttiCatchableArrayTypeIntValue,
            imageBaseIntValue);

        // trunc
        auto subRes32Value = rewriter.create<mlir_ts::CastOp>(loc, mth.getI32Type(), subResValue);
        setStructValue(loc, structValue, subRes32Value, 3);

        rewriter.create<mlir_ts::GlobalResultOp>(loc, mlir::ValueRange{structValue});

        rewriter.setInsertionPointAfter(_TI1NValue);

        return mlir::success();
    }

    mlir::Value typeInfoPtrValue(mlir::Location loc)
    {
        auto firstType = types.front();
        auto typeInfoPtr = rewriter.create<mlir_ts::ConstantOp>(loc, getRttiTypeDescriptor2PtrTy(StringRef(firstType.typeName).size()),
                                                                mlir::FlatSymbolRefAttr::get(rewriter.getContext(), firstType.typeInfoRef));
        return typeInfoPtr;
    }

    mlir::Value throwInfoPtrValue(mlir::Location loc)
    {
        auto throwInfoPtr = rewriter.create<mlir_ts::ConstantOp>(loc, getThrowInfoPtrTy(),
                                                                 mlir::FlatSymbolRefAttr::get(rewriter.getContext(), throwInfoRef));
        return throwInfoPtr;
    }

    mlir_ts::TupleType getThrowInfoTy()
    {
        return mth.getTupleType({mth.getI32Type(), mth.getI32Type(), mth.getI32Type(), mth.getI32Type()});
    }

    mlir_ts::RefType getThrowInfoPtrTy()
    {
        return mlir_ts::RefType::get(getThrowInfoTy());
    }

    mlir_ts::TupleType getRttiTypeDescriptor2Ty(int nameSize)
    {
        return mth.getTupleType({mth.getRefType(mth.getRefType(mth.getI8Type())), mth.getOpaqueType(), mth.getI8Array(nameSize + 1)});
    }

    mlir_ts::RefType getRttiTypeDescriptor2PtrTy(int nameSize)
    {
        return mlir_ts::RefType::get(getRttiTypeDescriptor2Ty(nameSize));
    }

    mlir_ts::TupleType getCatchableTypeTy()
    {
        return mth.getTupleType(
            {mth.getI32Type(), mth.getI32Type(), mth.getI32Type(), mth.getI32Type(), mth.getI32Type(), mth.getI32Type(), mth.getI32Type()});
    }

    mlir_ts::RefType getCatchableTypePtrTy()
    {
        return mlir_ts::RefType::get(getCatchableTypeTy());
    }

    mlir_ts::TupleType getCatchableArrayTypeTy(int arraySize)
    {
        return mth.getTupleType({mth.getI32Type(), mth.getI32Array(arraySize)});
    }

    mlir_ts::RefType getCatchableArrayTypePtrTy(int arraySize)
    {
        return mlir_ts::RefType::get(getCatchableArrayTypeTy(arraySize));
    }
};
} // namespace typescript

#endif // MLIR_TYPESCRIPT_LOWERTOLLVMLOGIC_LLVMRTTIHELPERVCWIN32_H_
