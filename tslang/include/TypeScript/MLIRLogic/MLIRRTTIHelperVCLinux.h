#ifndef MLIR_TYPESCRIPT_LOWERTOLLVMLOGIC_MLIRRTTIHELPERVCLINUX_H_
#define MLIR_TYPESCRIPT_LOWERTOLLVMLOGIC_MLIRRTTIHELPERVCLINUX_H_

#include "TypeScript/Config.h"
#include "TypeScript/Defines.h"
#include "TypeScript/Passes.h"
#include "TypeScript/TypeScriptDialect.h"
#include "TypeScript/TypeScriptOps.h"

#include "TypeScript/LowerToLLVM/LLVMRTTIHelperVCLinuxConst.h"
#include "TypeScript/MLIRLogic/MLIRCodeLogic.h"
#include "TypeScript/MLIRLogic/MLIRTypeHelper.h"

#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/TypeSwitch.h"

#include <functional>
#include <sstream>

#define DEBUG_TYPE "mlir"

using namespace ::typescript;
using namespace ts;
namespace mlir_ts = mlir::typescript;

namespace typescript
{

class MLIRRTTIHelperVCLinux
{

    enum class TypeInfo
    {
        Value,
        ClassTypeInfo,
        SingleInheritance_ClassTypeInfo,
        Pointer_TypeInfo
    };

    struct TypeNames
    {
        std::string typeName;
        TypeInfo infoType;
        int baseIndex;
    };

    mlir::OpBuilder &rewriter;
    mlir::ModuleOp &parentModule;
    MLIRTypeHelper mth;
    MLIRLogicHelper mlh;
    MLIRCodeLogic mcl;

    SmallVector<TypeNames> types;

  public:
    // Builds the thunk `name`: void name(ref<target> dest, ref<source> src) - the same builder as
    // MLIRRTTIHelperVCWin32's. Here it makes a class's box thunk, see linux::ClassType::boxThunkPrefix.
    using CopyThunkBuilder = std::function<mlir::LogicalResult(mlir::Location loc, StringRef name, mlir::Type source,
                                                               mlir::Type target)>;

  private:
    // the class a class type_info is emitted for: the box thunk's source
    mlir::Type classType;
    CopyThunkBuilder copyThunkBuilder;

  public:
    MLIRRTTIHelperVCLinux(mlir::OpBuilder &rewriter, mlir::ModuleOp &parentModule, CompileOptions& compileOptions)
        : rewriter(rewriter), parentModule(parentModule), mth(rewriter.getContext(), compileOptions), mlh(), mcl(rewriter, compileOptions)
    {
        // setI32AsCatchType();
    }

    void setCopyThunkBuilder(CopyThunkBuilder builder)
    {
        copyThunkBuilder = builder;
    }

    void setF32AsCatchType()
    {
        types.push_back({linux::F32Type::typeName, TypeInfo::Value, -1});
    }

    void setF64AsCatchType()
    {
        types.push_back({linux::F64Type::typeName, TypeInfo::Value, -1});
    }

    void setI32AsCatchType()
    {
        types.push_back({linux::I32Type::typeName, TypeInfo::Value, -1});
    }

    void setU32AsCatchType()
    {
        types.push_back({linux::U32Type::typeName, TypeInfo::Value, -1});
    }

    void setBoolAsCatchType()
    {
        types.push_back({linux::BoolType::typeName, TypeInfo::Value, -1});
    }

    void setI64AsCatchType()
    {
        types.push_back({linux::I64Type::typeName, TypeInfo::Value, -1});
    }

    void setStringTypeAsCatchType()
    {
        types.push_back({linux::StringType::typeName, TypeInfo::Value, -1});
    }

    void setI8PtrAsCatchType()
    {
        types.push_back({linux::I8PtrType::typeName, TypeInfo::Value, -1});
    }

    void setClassTypeAsCatchType(ArrayRef<StringRef> names)
    {
        auto first = true;
        auto countM1 = names.size() - 1;
        for (auto [index, name] : enumerate(names))
        {
            if (first)
            {
                types.push_back({name.str(), TypeInfo::Pointer_TypeInfo, 1});
            }

            if (index < countM1)
            {
                types.push_back({ name.str(), TypeInfo::SingleInheritance_ClassTypeInfo, (int)index + 2 });
            }
            else
            {
                types.push_back({ name.str(), TypeInfo::ClassTypeInfo, -1 });
            }

            first = false;
        }
    }

    void setClassTypeAsCatchType(StringRef name)
    {
        std::stringstream ss;
        ss << "_ZTIP";
        ss << name.str().size();
        ss << name.str();

        types.push_back({ss.str(), TypeInfo::ClassTypeInfo, -1});
    }

    bool setType(mlir::Type type, std::function<ClassInfo::TypePtr(StringRef fullClassName)> resolveClassInfo)
    {
        if (!type || type == rewriter.getNoneType())
        {
            return false;
        }

        auto result = true;
        llvm::TypeSwitch<mlir::Type>(mth.stripLiteralType(type))
            .Case<mlir::IntegerType>([&](auto intType) {
                auto width = intType.getIntOrFloatBitWidth();
                if (width == 32)
                {
                    if (intType.isUnsigned())
                    {
                        setU32AsCatchType();
                    }
                    else
                    {
                        setI32AsCatchType();
                    }
                }
                else if (width == 64)
                {
                    setI64AsCatchType();
                }
                else
                {
                    LLVM_DEBUG(llvm::dbgs() << "...unsupported throw/catch integer width: " << intType << "\n";);
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
                    LLVM_DEBUG(llvm::dbgs() << "...unsupported throw/catch float width: " << floatType << "\n";);
                    result = false;
                }
            })
            .Case<mlir_ts::NumberType>([&](auto numberType) {
#ifdef NUMBER_F64
                setF64AsCatchType();
#else
                setF32AsCatchType();
#endif
                // a `catch (e: number)` also takes a thrown int (`throw 1`), u32 or f32, and binding
                // it compares against their type_infos - see TryOpLowering, SaveCatchVarOpLowering
                setI32AsCatchType();
                setU32AsCatchType();
                setF32AsCatchType();
            })
            .Case<mlir_ts::BooleanType>([&](auto boolType) { setBoolAsCatchType(); })
            .Case<mlir_ts::BigIntType>([&](auto bigIntType) { setI64AsCatchType(); })
            .Case<mlir_ts::StringType>([&](auto stringType) { setStringTypeAsCatchType(); })
            .Case<mlir_ts::ClassType>([&](auto classType) {
                // we need all bases as well
                auto classInfo = resolveClassInfo(classType.getName().getValue());

                SmallVector<StringRef> classAndBases;
                classInfo->getBasesWithRoot(classAndBases);

                setClassTypeAsCatchType(classAndBases);
                this->classType = classInfo->classType;
            })
            .Case<mlir_ts::AnyType>([&](auto anyType) {
                // This overload only declares the RTTI globals a catch may reference. An
                // untyped/`any` catch is a catch-all (see the other overload), and binding its
                // variable compares the caught exception's type_info against each of these -
                // see linux::SaveCatchVarOpLowering. `_ZTIPv` is also what a `throw e` of the
                // caught value throws again.
                setI32AsCatchType();
                setU32AsCatchType();
                setF64AsCatchType();
                setF32AsCatchType();
                setBoolAsCatchType();
                setI64AsCatchType();
                setStringTypeAsCatchType();
                setI8PtrAsCatchType();
                // what tells a thrown class apart from the rest: its type_info is a
                // __pointer_type_info, which carries the box thunk
                types.push_back({linux::ClassType::pointerTypeInfoName, TypeInfo::Value, -1});
            })
            .Default([&](auto type) {
                LLVM_DEBUG(llvm::dbgs() << "...unsupported throw/catch type: " << type << "\n";);
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
                
        auto result = true;
        llvm::TypeSwitch<mlir::Type>(mth.stripLiteralType(type))
            .Case<mlir::IntegerType>([&](auto intType) {
                auto width = intType.getIntOrFloatBitWidth();
                if (width == 32)
                {
                    if (intType.isUnsigned())
                    {
                        setU32AsCatchType();
                    }
                    else
                    {
                        setI32AsCatchType();
                    }
                }
                else if (width == 64)
                {
                    // keep in sync with the resolveClassInfo overload above, which
                    // already supports this width.
                    setI64AsCatchType();
                }
                else
                {
                    LLVM_DEBUG(llvm::dbgs() << "...unsupported throw/catch integer width: " << intType << "\n";);
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
                    LLVM_DEBUG(llvm::dbgs() << "...unsupported throw/catch float width: " << floatType << "\n";);
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
            .Case<mlir_ts::BooleanType>([&](auto boolType) { setBoolAsCatchType(); })
            .Case<mlir_ts::BigIntType>([&](auto bigIntType) { setI64AsCatchType(); })
            .Case<mlir_ts::StringType>([&](auto stringType) { setStringTypeAsCatchType(); })
            .Case<mlir_ts::ClassType>([&](auto classType) { setClassTypeAsCatchType(classType.getName().getValue()); })
            // An untyped `catch (e)` / `catch (e: any)` has to catch everything, and on the
            // Itanium path that is a clause-less `catch ptr null`, which is what TryOpLowering
            // emits when no type is set. A `_ZTIPv` (void*) clause is not a catch-all: it
            // matches pointer-typed exceptions only, so a thrown number left the program.
            .Case<mlir_ts::AnyType>([&](auto anyType) {})
            .Default([&](auto type) {
                LLVM_DEBUG(llvm::dbgs() << "...unsupported throw/catch type: " << type << "\n";);
                result = false;
            });

        return result;
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

        // _ZTId
        for (auto type : types)
        {
            switch (type.infoType)
            {
            case TypeInfo::ClassTypeInfo:
            case TypeInfo::Pointer_TypeInfo:
            case TypeInfo::SingleInheritance_ClassTypeInfo:
                if (type.baseIndex >= 0)
                {
                    typeInfoClass(loc, type.typeName, type.infoType, types[type.baseIndex].typeName, types[type.baseIndex].infoType);
                }
                else
                {
                    typeInfoClass(loc, type.typeName, type.infoType, "", TypeInfo::Value);
                }

                break;
            default:
                typeInfoValue(loc, type.typeName);
                break;
            }
        }

        return true;
    }

    mlir::LogicalResult typeInfoValue(mlir::Location loc, StringRef name)
    {
        if (parentModule.lookupSymbol<mlir_ts::GlobalOp>(name))
        {
            return mlir::failure();
        }

        rewriter.create<mlir_ts::GlobalOp>(loc, mth.getOpaqueType(), true, name, LLVM::Linkage::External);
        return mlir::success();
    }

    std::string join(StringRef name, const char *prefix, int nameLength, const char *suffix)
    {
        std::stringstream ss;
        ss << prefix;
        ss << nameLength;
        ss << name.str();
        ss << suffix;
        return ss.str();
    }

    std::string join(StringRef name, const char *prefix, const char *suffix)
    {
        std::stringstream ss;
        ss << prefix;
        ss << name.str();
        ss << suffix;
        return ss.str();
    }

    mlir::Type getStringConstType(int size)
    {
        return mth.getConstArrayValueType(mth.getI8Type(), size);
    }

    const char *prefixLabel(TypeInfo ti)
    {
        switch (ti)
        {
        case TypeInfo::Pointer_TypeInfo:
            return "P";
        default:
            return "";
        }
    }

    std::string labelValue(StringRef className, TypeInfo ti)
    {
        assert(className.size());
        auto label = join(className, prefixLabel(ti), className.size(), "");
        return label;
    }

    std::string stringConstRefName(StringRef className, TypeInfo ti)
    {
        assert(className.size());
        auto name = join(labelValue(className, ti), "_ZTS", "");
        return name;
    }

    std::string typeInfoRefName(StringRef className, TypeInfo ti)
    {
        assert(className.size());
        auto name = join(labelValue(className, ti), "_ZTI", "");
        return name;
    }

    mlir::Type stringConstType(StringRef className, TypeInfo ti)
    {
        auto label = labelValue(className, ti);
        return getStringConstType(label.size() + 1);
    }

    mlir::LogicalResult stringConst(mlir::Location loc, StringRef className, TypeInfo ti)
    {
        auto label = labelValue(className, ti);
        auto name = stringConstRefName(className, ti);
        if (parentModule.lookupSymbol<mlir_ts::GlobalOp>(name))
        {
            return mlir::failure();
        }

        rewriter.create<mlir_ts::GlobalOp>(loc, stringConstType(className, ti), true, name, LLVM::Linkage::LinkonceODR, mcl.getStringAttrWith0(label));

        return mlir::success();
    }

    mlir::Type getTIType(TypeInfo ti)
    {
        switch (ti)
        {
        case TypeInfo::SingleInheritance_ClassTypeInfo:
            return mth.getTupleType({mth.getOpaqueType(), mth.getOpaqueType(), mth.getOpaqueType()});
        case TypeInfo::Pointer_TypeInfo:
            // the last field is ours, not libstdc++'s - see linux::ClassType::boxThunkPrefix
            return mth.getTupleType({mth.getOpaqueType(), mth.getOpaqueType(), mth.getI32Type(), mth.getOpaqueType(), mth.getOpaqueType()});
        default:
            return mth.getTupleType({mth.getOpaqueType(), mth.getOpaqueType()});
        }
    }

    const char *getClassInfoName(TypeInfo ti)
    {
        switch (ti)
        {
        case TypeInfo::SingleInheritance_ClassTypeInfo:
            return linux::ClassType::singleInheritanceClassTypeInfoName;
        case TypeInfo::Pointer_TypeInfo:
            return linux::ClassType::pointerTypeInfoName;
        case TypeInfo::ClassTypeInfo:
            return linux::ClassType::classTypeInfoName;
        default:
            // only reached for TypeInfo::Value, which every caller's own switch
            // already routes to typeInfoValue() instead of typeInfoClass()/here.
            return linux::ClassType::classTypeInfoName;
        }
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
        assert(isa<mlir_ts::TupleType>(tpl) || isa<mlir_ts::ConstTupleType>(tpl) || isa<mlir_ts::ConstArrayValueType>(tpl));
        tupleValue = rewriter.create<mlir_ts::InsertPropertyOp>(loc, tpl, value, tupleValue, MLIRHelper::getStructIndex(rewriter, index));
        return mlir::success();
    }

    bool hasType()
    {
        return types.size() > 0;
    }

    mlir::Value typeInfoPtrValue(mlir::Location loc)
    {
        // TODO:
        return throwInfoPtrValue(loc);
    }

    mlir::Value throwInfoPtrValue(mlir::Location loc)
    {
        auto typeName = types.front().typeName;
        auto classType = types.front().infoType == TypeInfo::ClassTypeInfo;

        assert(typeName.size() > 0);

        LLVM_DEBUG(llvm::dbgs() << "\n Throw info name: " << typeName << "\n");

        mlir::Type tiType;
        if (classType)
        {
            tiType = getTIType(TypeInfo::Pointer_TypeInfo);
        }
        else
        {
            tiType = mth.getOpaqueType();
        }

        mlir::Value throwInfoPtr = rewriter.create<mlir_ts::ConstantOp>(loc, mth.getRefType(tiType),
                                                                        mlir::FlatSymbolRefAttr::get(rewriter.getContext(), typeName));
        return throwInfoPtr;
    }

    // The box thunk field of a class's pointer type_info: the thunk's address, or null when
    // nothing here can build it - an untyped catch then boxes the instance as a plain object.
    mlir::Value boxThunkValue(mlir::Location loc, StringRef label)
    {
        auto thunkName = std::string(linux::ClassType::boxThunkPrefix) + label.str();
        if (!parentModule.lookupSymbol(thunkName))
        {
            auto built = false;
            if (copyThunkBuilder && classType)
            {
                mlir::OpBuilder::InsertionGuard guard(rewriter);
                built = mlir::succeeded(copyThunkBuilder(loc, thunkName, classType, mlir_ts::AnyType::get(rewriter.getContext())));
            }

            if (!built)
            {
                auto nullValue = rewriter.create<mlir_ts::NullOp>(loc, mth.getNullType());
                return rewriter.create<mlir_ts::CastOp>(loc, mth.getOpaqueType(), nullValue);
            }
        }

        return rewriter.create<mlir_ts::SymbolRefOp>(loc, mth.getOpaqueType(),
                                                     mlir::FlatSymbolRefAttr::get(rewriter.getContext(), thunkName));
    }

    mlir::LogicalResult typeInfoRef(mlir::Location loc, StringRef className, TypeInfo ti, StringRef baseName = "",
                                    TypeInfo baseti = TypeInfo::Value)
    {
        auto name = typeInfoRefName(className, ti);
        if (parentModule.lookupSymbol<mlir_ts::GlobalOp>(name))
        {
            return mlir::failure();
        }

        auto typeInfoType = getTIType(ti);
        auto globalOp = rewriter.create<mlir_ts::GlobalOp>(loc, typeInfoType, true, name, LLVM::Linkage::LinkonceODR);

        {
            setGlobalOpWritingPoint(globalOp);

            // begin
            mlir::Value structVal = rewriter.create<mlir_ts::UndefOp>(loc, typeInfoType);

            auto itemValue1 = rewriter.create<mlir_ts::AddressOfOp>(
                loc, mth.getRefType(mth.getOpaqueType()), mlir::FlatSymbolRefAttr::get(rewriter.getContext(), getClassInfoName(ti)),
                mlir::IntegerAttr::get(mth.getI32Type(), 2));
            auto castValue1 = rewriter.create<mlir_ts::CastOp>(loc, mth.getOpaqueType(), itemValue1);
            setStructValue(loc, structVal, castValue1, 0);

            auto itemValue2 = rewriter.create<mlir_ts::AddressOfOp>(
                loc, mth.getRefType(stringConstType(className, ti)),
                mlir::FlatSymbolRefAttr::get(rewriter.getContext(), stringConstRefName(className, ti)), mlir::IntegerAttr());

            auto castValue2 = rewriter.create<mlir_ts::CastOp>(loc, mth.getOpaqueType(), itemValue2);
            setStructValue(loc, structVal, castValue2, 1);

            if (ti == TypeInfo::Pointer_TypeInfo)
            {
                // no qualifiers; the flag says the box thunk field below is there
                auto itemValueI32 = rewriter.create<mlir_ts::ConstantOp>(loc, mth.getI32Type(),
                                                                         mth.getI32AttrValue(linux::ClassType::boxThunkFlag));
                setStructValue(loc, structVal, itemValueI32, linux::ClassType::flagsField);

                // add base class name
                auto itemValue4 = rewriter.create<mlir_ts::AddressOfOp>(
                    loc, mth.getRefType(getTIType(baseti)),
                    mlir::FlatSymbolRefAttr::get(rewriter.getContext(), typeInfoRefName(baseName, baseti)), mlir::IntegerAttr());

                auto castValue4 = rewriter.create<mlir_ts::CastOp>(loc, mth.getOpaqueType(), itemValue4);
                setStructValue(loc, structVal, castValue4, 3);

                setStructValue(loc, structVal, boxThunkValue(loc, labelValue(className, ti)), linux::ClassType::boxThunkField);
            }
            else if (ti == TypeInfo::SingleInheritance_ClassTypeInfo)
            {
                // add base class name
                auto itemValue3 = rewriter.create<mlir_ts::AddressOfOp>(
                    loc, mth.getRefType(getTIType(baseti)),
                    mlir::FlatSymbolRefAttr::get(rewriter.getContext(), typeInfoRefName(baseName, baseti)), mlir::IntegerAttr());

                auto castValue3 = rewriter.create<mlir_ts::CastOp>(loc, mth.getOpaqueType(), itemValue3);
                setStructValue(loc, structVal, castValue3, 2);
            }

            // end
            rewriter.create<mlir_ts::GlobalResultOp>(loc, mlir::ValueRange{structVal});

            rewriter.setInsertionPointAfter(globalOp);
        }

        return mlir::success();
    }

    mlir::LogicalResult typeInfoClass(mlir::Location loc, StringRef name, TypeInfo ti, StringRef baseName, TypeInfo baseti)
    {
        typeInfoValue(loc, getClassInfoName(ti));
        stringConst(loc, name, ti);
        typeInfoRef(loc, name, ti, baseName, baseti);
        return mlir::success();
    }

    mlir_ts::TupleType getLandingPadType()
    {
        return mth.getTupleType({mth.getOpaqueType(), mth.getI32Type()});
    }
};
} // namespace typescript

#undef DEBUG_TYPE

#endif // MLIR_TYPESCRIPT_LOWERTOLLVMLOGIC_MLIRRTTIHELPERVCLINUX_H_
