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
#include <functional>

#define DEBUG_TYPE "mlir"

using namespace ::typescript;
using namespace ts;
namespace mlir_ts = mlir::typescript;

namespace typescript
{

class MLIRRTTIHelperVCWin32
{

    struct TypeNames
    {
        std::string typeName;
        std::string typeInfoRef;
        std::string catchableTypeInfoRef;
    };

    mlir::OpBuilder &rewriter;
    mlir::ModuleOp &parentModule;
    MLIRTypeHelper mth;
    MLIRLogicHelper mlh;

    SmallVector<TypeNames> types;

  public:
    std::string catchableTypeInfoArrayRef;
    std::string throwInfoRef;

    MLIRRTTIHelperVCWin32(mlir::OpBuilder &rewriter, mlir::ModuleOp &parentModule)
        : rewriter(rewriter), parentModule(parentModule), mth(rewriter.getContext()), mlh()
    {
        // setI32AsCatchType();
    }

    void setF32AsCatchType()
    {
        types.push_back({windows::F32Type::typeName, windows::F32Type::typeInfoRef, windows::F32Type::catchableTypeInfoRef});

        catchableTypeInfoArrayRef = windows::F32Type::catchableTypeInfoArrayRef;
        throwInfoRef = windows::F32Type::throwInfoRef;
    }

    void setF64AsCatchType()
    {
        types.push_back({windows::F64Type::typeName, windows::F64Type::typeInfoRef, windows::F64Type::catchableTypeInfoRef});

        catchableTypeInfoArrayRef = windows::F64Type::catchableTypeInfoArrayRef;
        throwInfoRef = windows::F64Type::throwInfoRef;
    }

    void setI32AsCatchType()
    {
        types.push_back({windows::I32Type::typeName, windows::I32Type::typeInfoRef, windows::I32Type::catchableTypeInfoRef});

        catchableTypeInfoArrayRef = windows::I32Type::catchableTypeInfoArrayRef;
        throwInfoRef = windows::I32Type::throwInfoRef;
    }

    void setStringTypeAsCatchType()
    {
        types.push_back({windows::StringType::typeName, windows::StringType::typeInfoRef, windows::StringType::catchableTypeInfoRef});
        types.push_back({windows::StringType::typeName2, windows::StringType::typeInfoRef2, windows::StringType::catchableTypeInfoRef2});

        catchableTypeInfoArrayRef = windows::StringType::catchableTypeInfoArrayRef;
        throwInfoRef = windows::StringType::throwInfoRef;
    }

    void setI8PtrAsCatchType()
    {
        types.push_back({windows::I8PtrType::typeName, windows::I8PtrType::typeInfoRef, windows::I8PtrType::catchableTypeInfoRef});

        catchableTypeInfoArrayRef = windows::I8PtrType::catchableTypeInfoArrayRef;
        throwInfoRef = windows::I8PtrType::throwInfoRef;
    }

    void setClassTypeAsCatchType(ArrayRef<StringRef> names)
    {
        for (auto name : names)
        {
            types.push_back({join(name, windows::ClassType::typeName, windows::ClassType::typeNameSuffix),
                             join(name, windows::ClassType::typeInfoRef, windows::ClassType::typeInfoRefSuffix),
                             join(name, windows::ClassType::catchableTypeInfoRef, windows::ClassType::catchableTypeInfoRefSuffix)});
        }

        types.push_back({windows::ClassType::typeName2, windows::ClassType::typeInfoRef2, windows::ClassType::catchableTypeInfoRef2});

        catchableTypeInfoArrayRef = windows::ClassType::catchableTypeInfoArrayRef;
        throwInfoRef = windows::ClassType::throwInfoRef;
    }

    void setClassTypeAsCatchType(StringRef name)
    {
        types.push_back({join(name, windows::ClassType::typeName, windows::ClassType::typeNameSuffix),
                         join(name, windows::ClassType::typeInfoRef, windows::ClassType::typeInfoRefSuffix),
                         join(name, windows::ClassType::catchableTypeInfoRef, windows::ClassType::catchableTypeInfoRefSuffix)});

        types.push_back({windows::ClassType::typeName2, windows::ClassType::typeInfoRef2, windows::ClassType::catchableTypeInfoRef2});

        catchableTypeInfoArrayRef = windows::ClassType::catchableTypeInfoArrayRef;
        throwInfoRef = windows::ClassType::throwInfoRef;
    }

    std::string join(StringRef name, const char *prefix, const char *suffix)
    {
        std::stringstream ss;
        ss << prefix;
        ss << name.str();
        ss << suffix;
        return ss.str();
    }

    bool setType(mlir::Type type, std::function<ClassInfo::TypePtr(StringRef fullClassName)> resolveClassInfo)
    {
        if (!type || type == rewriter.getNoneType())
        {
            return false;
        }

        auto normalizedType = mth.stripLiteralType(type);
        if (auto enumType = normalizedType.dyn_cast<mlir_ts::EnumType>())
        {
            normalizedType = enumType.getElementType();
        }

        auto result = true;
        llvm::TypeSwitch<mlir::Type>(normalizedType)
            .Case<mlir::IntegerType>([&](auto intType) {
                if (intType.getIntOrFloatBitWidth() == 32)
                {
                    setI32AsCatchType();
                }
                else
                {
                    result = false;
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
                    result = false;
                }
            })
            .Case<mlir_ts::NumberType>([&](auto numberType) {
#ifdef NUMBER_F64
                setF64AsCatchType();
#else
                setF32AsCatchType();
#endif
            })
            .Case<mlir_ts::StringType>([&](auto stringType) { setStringTypeAsCatchType(); })
            .Case<mlir_ts::ClassType>([&](auto classType) {
                // we need all bases as well
                auto classInfo = resolveClassInfo(classType.getName().getValue());

                SmallVector<StringRef> classAndBases;
                classInfo->getBasesWithRoot(classAndBases);

                setClassTypeAsCatchType(classAndBases);
            })
            .Case<mlir_ts::AnyType>([&](auto anyType) { setI8PtrAsCatchType(); })
            .Default([&](auto type) {
                LLVM_DEBUG(llvm::dbgs() << "...throw type: " << type << "\n";);
                result = false;
            });

        return result;
    }

    bool setType(mlir::Type type)
    {
        if (!type || type == rewriter.getNoneType())
        {
            return false;
        }

        llvm::TypeSwitch<mlir::Type>(mth.stripLiteralType(type))
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
            .Case<mlir_ts::NumberType>([&](auto numberType) {
#ifdef NUMBER_F64
                setF64AsCatchType();
#else
                setF32AsCatchType();
#endif
            })
            .Case<mlir_ts::StringType>([&](auto stringType) { setStringTypeAsCatchType(); })
            .Case<mlir_ts::ClassType>([&](auto classType) { setClassTypeAsCatchType(classType.getName().getValue()); })
            .Case<mlir_ts::AnyType>([&](auto anyType) { setI8PtrAsCatchType(); })
            .Default([&](auto type) {
                LLVM_DEBUG(llvm::dbgs() << "...throw type: " << type << "\n";);
                llvm_unreachable("not implemented");
            });

        return true;
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

    bool setRTTIForType(mlir::Location loc, mlir::Type type, std::function<ClassInfo::TypePtr(StringRef fullClassName)> resolveClassInfo)
    {
        if (!setType(type, resolveClassInfo))
        {
            // no type provided
            return false;
        }

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

        return true;
    }

    mlir::LogicalResult typeInfo(mlir::Location loc)
    {
        auto name = windows::typeInfoExtRef;
        if (parentModule.lookupSymbol<mlir_ts::GlobalOp>(name))
        {
            return mlir::failure();
        }

        SmallVector<mlir::NamedAttribute> attrs;
        attrs.push_back({ATTR("Linkage"), ATTR("External")});

        rewriter.create<mlir_ts::GlobalOp>(loc, mth.getOpaqueType(), true, /*LLVM::Linkage::External,*/ name, mlir::Attribute{}, attrs);
        return mlir::success();
    }

    mlir::LogicalResult typeDescriptors(mlir::Location loc)
    {
        for (auto type : types)
        {
            typeDescriptor(loc, type.typeInfoRef, type.typeName);
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
        auto tpl = tupleValue.getType();
        assert(tpl.isa<mlir_ts::TupleType>() || tpl.isa<mlir_ts::ConstTupleType>() || tpl.isa<mlir_ts::ConstArrayValueType>());
        tupleValue = rewriter.create<mlir_ts::InsertPropertyOp>(loc, tpl, value, tupleValue, MLIRHelper::getStructIndex(rewriter, index));
        return mlir::success();
    }

    mlir::LogicalResult typeDescriptor(mlir::Location loc, StringRef typeInfoRefName, StringRef typeName)
    {
        auto name = typeInfoRefName;
        if (parentModule.lookupSymbol<mlir_ts::GlobalOp>(name))
        {
            return mlir::failure();
        }

        MLIRCodeLogic mcl(rewriter);

        auto rttiTypeDescriptor2Ty = getRttiTypeDescriptor2Ty(StringRef(typeName).size());

        SmallVector<mlir::NamedAttribute> attrs;
        attrs.push_back({ATTR("Linkage"), ATTR("LinkonceODR")});

        auto _r0n_Value = rewriter.create<mlir_ts::GlobalOp>(loc, rttiTypeDescriptor2Ty, false, /*LLVM::Linkage::LinkonceODR,*/ name,
                                                             mlir::Attribute{}, attrs);

        {
            setGlobalOpWritingPoint(_r0n_Value);

            // begin
            mlir::Value structVal = rewriter.create<mlir_ts::UndefOp>(loc, rttiTypeDescriptor2Ty);

            auto itemValue1 = rewriter.create<mlir_ts::ConstantOp>(loc, mth.getRefType(mth.getOpaqueType()),
                                                                   mlir::FlatSymbolRefAttr::get(rewriter.getContext(), windows::typeInfoExtRef));
            setStructValue(loc, structVal, itemValue1, 0);

            auto itemValue2 = rewriter.create<mlir_ts::NullOp>(loc, mth.getNullType());
            setStructValue(loc, structVal, itemValue2, 1);

            auto itemValue3 = rewriter.create<mlir_ts::ConstantOp>(loc, mth.getI8Array(StringRef(typeName).size() + 1),
                                                                   mcl.getStringAttrWith0(typeName.str()));
            setStructValue(loc, structVal, itemValue3, 2);

            // end
            rewriter.create<mlir_ts::GlobalResultOp>(loc, mlir::ValueRange{structVal});

            rewriter.setInsertionPointAfter(_r0n_Value);
        }

        return mlir::success();
    }

    mlir::LogicalResult imageBase(mlir::Location loc)
    {
        auto name = windows::imageBaseRef;
        if (parentModule.lookupSymbol<mlir_ts::GlobalOp>(name))
        {
            return mlir::failure();
        }

        SmallVector<mlir::NamedAttribute> attrs;
        attrs.push_back({ATTR("Linkage"), ATTR("External")});

        rewriter.create<mlir_ts::GlobalOp>(loc, mth.getI8Type(), true, /*LLVM::Linkage::External,*/ name, mlir::Attribute{}, attrs);
        return mlir::success();
    }

    mlir::LogicalResult catchableTypes(mlir::Location loc)
    {
        for (auto type : types)
        {
            catchableType(loc, type.catchableTypeInfoRef, type.typeInfoRef, type.typeName);
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

        SmallVector<mlir::NamedAttribute> attrs;
        attrs.push_back({ATTR("Linkage"), ATTR("LinkonceODR")});

        auto _ct_r0n_Value = rewriter.create<mlir_ts::GlobalOp>(loc, ehCatchableTypeTy, true, /*LLVM::Linkage::LinkonceODR,*/ name,
                                                                mlir::Attribute{}, attrs);

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
                loc, mth.getOpaqueType(), mlir::FlatSymbolRefAttr::get(rewriter.getContext(), windows::imageBaseRef));
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
                                                                      mlir::FlatSymbolRefAttr::get(rewriter.getContext(), windows::imageBaseRef));
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

        SmallVector<mlir::NamedAttribute> attrs;
        attrs.push_back({ATTR("Linkage"), ATTR("LinkonceODR")});

        auto _cta1nValue = rewriter.create<mlir_ts::GlobalOp>(loc, ehCatchableArrayTypeTy, true, /*LLVM::Linkage::LinkonceODR,*/ name,
                                                              mlir::Attribute{}, attrs);

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
            mlir::Value arrayVal = rewriter.create<mlir_ts::UndefOp>(loc, mth.getConstArrayValueType(mth.getI32Type(), arraySize));

            for (auto [index, value] : enumerate(values))
            {
                setStructValue(loc, arrayVal, value, index);
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
                rewriter.create<mlir_ts::InsertPropertyOp>(loc, tupleType, itemValue, structVal, MLIRHelper::getStructIndex(rewriter, position++));
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

        SmallVector<mlir::NamedAttribute> attrs;
        attrs.push_back({ATTR("Linkage"), ATTR("LinkonceODR")});

        auto _TI1NValue =
            rewriter.create<mlir_ts::GlobalOp>(loc, throwInfoTy, true, /*LLVM::Linkage::LinkonceODR,*/ name, mlir::Attribute{}, attrs);

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
                                                                      mlir::FlatSymbolRefAttr::get(rewriter.getContext(), windows::imageBaseRef));
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

    bool hasType()
    {
        return types.size() > 0;
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

    mlir_ts::TupleType getLandingPadType()
    {
        return mth.getTupleType({mth.getOpaqueType(), mth.getI32Type(), mth.getOpaqueType()});
    }
};
} // namespace typescript

#undef DEBUG_TYPE

#endif // MLIR_TYPESCRIPT_LOWERTOLLVMLOGIC_LLVMRTTIHELPERVCWIN32_H_
