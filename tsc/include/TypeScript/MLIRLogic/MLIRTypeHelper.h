#ifndef MLIR_TYPESCRIPT_COMMONGENLOGIC_MLIRTYPEHELPER_H_
#define MLIR_TYPESCRIPT_COMMONGENLOGIC_MLIRTYPEHELPER_H_

#include "TypeScript/TypeScriptOps.h"
#include "TypeScript/MLIRLogic/MLIRHelper.h"

namespace mlir_ts = mlir::typescript;

namespace typescript
{

enum class MatchResultType
{
    Match,
    NotMatchArgCount,
    NotMatchArg,
    NotMatchResultCount,
    NotMatchResult
};

struct MatchResult
{
    MatchResultType result;
    unsigned index;
};

class MLIRTypeHelper
{
    mlir::MLIRContext *context;

  public:
    MLIRTypeHelper(mlir::MLIRContext *context) : context(context)
    {
    }

    // types
    mlir::IntegerType getI8Type()
    {
        return mlir::IntegerType::get(context, 8);
    }

    mlir::IntegerType getI32Type()
    {
        return mlir::IntegerType::get(context, 32);
    }

    mlir::IntegerType getI64Type()
    {
        return mlir::IntegerType::get(context, 64);
    }

    mlir_ts::StringType getStringType()
    {
        return mlir_ts::StringType::get(context);
    }

    mlir_ts::OpaqueType getOpaqueType()
    {
        return mlir_ts::OpaqueType::get(context);
    }

    mlir_ts::OpaqueType getInterfaceVTableType(mlir_ts::InterfaceType ifaceType)
    {
        return mlir_ts::OpaqueType::get(context);
    }

    mlir_ts::BooleanType getBooleanType()
    {
        return mlir_ts::BooleanType::get(context);
    }

    mlir_ts::NullType getNullType()
    {
        return mlir_ts::NullType::get(context);
    }

    mlir_ts::RefType getRefType(mlir::Type type)
    {
        return mlir_ts::RefType::get(type);
    }

    mlir_ts::ConstArrayValueType getConstArrayValueType(mlir::Type elementType, unsigned size)
    {
        assert(elementType);
        return mlir_ts::ConstArrayValueType::get(elementType, size);
    }

    mlir_ts::ConstArrayType getConstArrayType(mlir::Type elementType, unsigned size)
    {
        assert(elementType);
        return mlir_ts::ConstArrayType::get(elementType, size);
    }

    mlir_ts::ConstArrayValueType getI8Array(unsigned size)
    {
        return getConstArrayValueType(getI8Type(), size);
    }

    mlir_ts::ConstArrayValueType getI32Array(unsigned size)
    {
        return getConstArrayValueType(getI32Type(), size);
    }

    mlir_ts::TupleType getTupleType(mlir::ArrayRef<mlir::Type> types)
    {
        llvm::SmallVector<mlir_ts::FieldInfo> fields;
        for (auto type : types)
        {
            fields.push_back(mlir_ts::FieldInfo{nullptr, type});
        }

        return mlir_ts::TupleType::get(context, fields);
    }

    mlir::IntegerAttr getI32AttrValue(int32_t value)
    {
        return mlir::IntegerAttr::get(getI32Type(), mlir::APInt(32, value));
    }

    mlir::IntegerAttr getI64AttrValue(int64_t value)
    {
        return mlir::IntegerAttr::get(getI64Type(), mlir::APInt(64, value));
    }

    mlir::Type getStructIndexType()
    {
        return getI32Type();
    }

    mlir::IntegerAttr getStructIndexAttrValue(int32_t value)
    {
        return mlir::IntegerAttr::get(getI32Type(), mlir::APInt(32, value));
    }

    bool isValueType(mlir::Type type)
    {
        return type && (type.isIntOrIndexOrFloat() || type.isa<mlir_ts::TupleType>() || type.isa<mlir_ts::ConstTupleType>() ||
                        type.isa<mlir_ts::ConstArrayType>());
    }

    mlir::Attribute TupleFieldName(mlir::StringRef name)
    {
        assert(!name.empty());
        return mlir::StringAttr::get(context, name);
    }

    bool isUndefinedType(mlir::Type type)
    {
        if (auto optType = type.dyn_cast_or_null<mlir_ts::OptionalType>())
        {
            return optType == mlir_ts::UndefPlaceHolderType::get(context);
        }

        return false;
    }

    mlir::Type isBoundReference(mlir::Type elementType, bool &isBound)
    {
#ifdef USE_BOUND_FUNCTION_FOR_OBJECTS
        if (auto funcType = elementType.dyn_cast_or_null<mlir_ts::FunctionType>())
        {
            isBound = true;
            return mlir_ts::BoundFunctionType::get(context, funcType);
        }
#endif

        isBound = false;
        return elementType;
    }

    mlir::Type convertConstArrayTypeToArrayType(mlir::Type type)
    {
        if (auto constArrayType = type.dyn_cast_or_null<mlir_ts::ConstArrayType>())
        {
            return mlir_ts::ArrayType::get(constArrayType.getElementType());
        }

        return type;
    }

    mlir::Type convertConstTupleTypeToTupleType(mlir::Type type)
    {
        // tuple is value and copied already
        if (auto constTupleType = type.dyn_cast_or_null<mlir_ts::ConstTupleType>())
        {
            return mlir_ts::TupleType::get(context, constTupleType.getFields());
        }

        return type;
    }

    mlir::Type convertTupleTypeToConstTupleType(mlir::Type type)
    {
        // tuple is value and copied already
        if (auto tupleType = type.dyn_cast_or_null<mlir_ts::TupleType>())
        {
            return mlir_ts::ConstTupleType::get(context, tupleType.getFields());
        }

        return type;
    }

    mlir_ts::FunctionType getFunctionTypeWithThisType(mlir_ts::FunctionType funcType, mlir::Type thisType, bool replace = false)
    {
        mlir::SmallVector<mlir::Type> args;
        args.push_back(thisType);
        auto offset = replace || funcType.getNumInputs() > 0 && funcType.getInput(0) == mlir_ts::OpaqueType::get(context) ? 1 : 0;
        auto sliced = funcType.getInputs().slice(offset);
        args.append(sliced.begin(), sliced.end());
        auto newFuncType = mlir_ts::FunctionType::get(context, args, funcType.getResults());
        return newFuncType;
    }

    mlir_ts::FunctionType getFunctionTypeWithOpaqueThis(mlir_ts::FunctionType funcType, bool replace = false)
    {
        return getFunctionTypeWithThisType(funcType, mlir_ts::OpaqueType::get(context), replace);
    }

    mlir_ts::FunctionType getFunctionTypeAddingFirstArgType(mlir_ts::FunctionType funcType, mlir::Type firstArgType)
    {
        mlir::SmallVector<mlir::Type> funcArgTypes(funcType.getInputs().begin(), funcType.getInputs().end());
        funcArgTypes.insert(funcArgTypes.begin(), firstArgType);
        return mlir_ts::FunctionType::get(context, funcArgTypes, funcType.getResults());
    }

    mlir_ts::FunctionType getFunctionTypeReplaceOpaqueWithThisType(mlir_ts::FunctionType funcType, mlir::Type opaqueReplacementType)
    {
        mlir::SmallVector<mlir::Type> funcArgTypes(funcType.getInputs().begin(), funcType.getInputs().end());
        for (auto &type : funcArgTypes)
        {
            if (type.isa<mlir_ts::OpaqueType>())
            {
                type = opaqueReplacementType;
                break;
            }
        }

        return mlir_ts::FunctionType::get(context, funcArgTypes, funcType.getResults());
    }

    mlir_ts::BoundFunctionType getBoundFunctionTypeReplaceOpaqueWithThisType(mlir_ts::BoundFunctionType funcType,
                                                                             mlir::Type opaqueReplacementType)
    {
        mlir::SmallVector<mlir::Type> funcArgTypes(funcType.getInputs().begin(), funcType.getInputs().end());
        for (auto &type : funcArgTypes)
        {
            if (type.isa<mlir_ts::OpaqueType>())
            {
                type = opaqueReplacementType;
                break;
            }
        }

        return mlir_ts::BoundFunctionType::get(context, funcArgTypes, funcType.getResults(), funcType.isVarArg());
    }

    MatchResult TestFunctionTypesMatch(mlir_ts::FunctionType inFuncType, mlir_ts::FunctionType resFuncType, unsigned startParam = 0)
    {
        // TODO: make 1 common function
        if (inFuncType.getInputs().size() != resFuncType.getInputs().size())
        {
            return {MatchResultType::NotMatchArgCount, 0};
        }

        for (unsigned i = startParam, e = inFuncType.getInputs().size(); i != e; ++i)
        {
            if (inFuncType.getInput(i) != resFuncType.getInput(i))
            {
                /*
                if (i == 0 && (inFuncType.getInput(i).isa<mlir_ts::OpaqueType>() || resFuncType.getInput(i).isa<mlir_ts::OpaqueType>()))
                {
                    // allow not to match opaque time at first position
                    continue;
                }
                */

                return {MatchResultType::NotMatchArg, i};
            }
        }

        auto inRetCount = inFuncType.getResults().size();
        auto resRetCount = resFuncType.getResults().size();

        auto noneType = mlir::NoneType::get(context);
        auto voidType = mlir_ts::VoidType::get(context);

        for (auto retType : inFuncType.getResults())
        {
            auto isVoid = !retType || retType == noneType || retType == voidType;
            if (isVoid)
            {
                inRetCount--;
            }
        }

        for (auto retType : resFuncType.getResults())
        {
            auto isVoid = !retType || retType == noneType || retType == voidType;
            if (isVoid)
            {
                resRetCount--;
            }
        }

        if (inRetCount != resRetCount)
        {
            return {MatchResultType::NotMatchResultCount, 0};
        }

        for (unsigned i = 0, e = inRetCount; i != e; ++i)
        {
            auto inRetType = inFuncType.getResult(i);
            auto resRetType = resFuncType.getResult(i);

            auto isInVoid = !inRetType || inRetType == noneType || inRetType == voidType;
            auto isResVoid = !resRetType || resRetType == noneType || resRetType == voidType;
            if (!isInVoid && !isResVoid && inRetType != resRetType)
            {
                return {MatchResultType::NotMatchResult, i};
            }
        }

        return {MatchResultType::Match, 0};
    }

    MatchResult TestFunctionTypesMatchWithObjectMethods(mlir::Type inFuncType, mlir::Type resFuncType, unsigned startParamIn = 0,
                                                        unsigned startParamRes = 0)
    {
        return TestFunctionTypesMatchWithObjectMethods(inFuncType.cast<mlir_ts::FunctionType>(), resFuncType.cast<mlir_ts::FunctionType>(),
                                                       startParamIn, startParamRes);
    }

    MatchResult TestFunctionTypesMatchWithObjectMethods(mlir_ts::FunctionType inFuncType, mlir_ts::FunctionType resFuncType,
                                                        unsigned startParamIn = 0, unsigned startParamRes = 0)
    {
        // 1 we need to skip opaque and objects
        if (startParamIn <= 0 && inFuncType.getNumInputs() > 0 &&
            (inFuncType.getInput(0).isa<mlir_ts::OpaqueType>() || inFuncType.getInput(0).isa<mlir_ts::ObjectType>()))
        {
            startParamIn = 1;
        }

        if (startParamIn <= 1 && inFuncType.getNumInputs() > 1 &&
            (inFuncType.getInput(1).isa<mlir_ts::OpaqueType>() || inFuncType.getInput(1).isa<mlir_ts::ObjectType>()))
        {
            startParamIn = 2;
        }

        if (startParamRes <= 0 && resFuncType.getNumInputs() > 0 &&
            (resFuncType.getInput(0).isa<mlir_ts::OpaqueType>() || resFuncType.getInput(0).isa<mlir_ts::ObjectType>()))
        {
            startParamRes = 1;
        }

        if (startParamRes <= 1 && resFuncType.getNumInputs() > 1 &&
            (resFuncType.getInput(1).isa<mlir_ts::OpaqueType>() || resFuncType.getInput(1).isa<mlir_ts::ObjectType>()))
        {
            startParamRes = 2;
        }

        if (inFuncType.getInputs().size() - startParamIn != resFuncType.getInputs().size() - startParamRes)
        {
            return {MatchResultType::NotMatchArgCount, 0};
        }

        for (unsigned i = 0, e = inFuncType.getInputs().size() - startParamIn; i != e; ++i)
        {
            if (inFuncType.getInput(i + startParamIn) != resFuncType.getInput(i + startParamRes))
            {
                /*
                if (i == 0 && (inFuncType.getInput(i).isa<mlir_ts::OpaqueType>() || resFuncType.getInput(i).isa<mlir_ts::OpaqueType>()))
                {
                    // allow not to match opaque time at first position
                    continue;
                }
                */

                return {MatchResultType::NotMatchArg, i};
            }
        }

        auto inRetCount = inFuncType.getResults().size();
        auto resRetCount = resFuncType.getResults().size();

        auto noneType = mlir::NoneType::get(context);
        auto voidType = mlir_ts::VoidType::get(context);

        for (auto retType : inFuncType.getResults())
        {
            auto isVoid = !retType || retType == noneType || retType == voidType;
            if (isVoid)
            {
                inRetCount--;
            }
        }

        for (auto retType : resFuncType.getResults())
        {
            auto isVoid = !retType || retType == noneType || retType == voidType;
            if (isVoid)
            {
                resRetCount--;
            }
        }

        if (inRetCount != resRetCount)
        {
            return {MatchResultType::NotMatchResultCount, 0};
        }

        for (unsigned i = 0, e = inRetCount; i != e; ++i)
        {
            auto inRetType = inFuncType.getResult(i);
            auto resRetType = resFuncType.getResult(i);

            auto isInVoid = !inRetType || inRetType == noneType || inRetType == voidType;
            auto isResVoid = !resRetType || resRetType == noneType || resRetType == voidType;
            if (!isInVoid && !isResVoid && inRetType != resRetType)
            {
                return {MatchResultType::NotMatchResult, i};
            }
        }

        return {MatchResultType::Match, 0};
    }

    mlir::Value GetReferenceOfLoadOp(mlir::Value value)
    {
        if (auto loadOp = mlir::dyn_cast_or_null<mlir_ts::LoadOp>(value.getDefiningOp()))
        {
            // this LoadOp will be removed later as unused
            auto refValue = loadOp.reference();
            return refValue;
        }

        return mlir::Value();
    }

    template <typename T1, typename T2> bool isCastableTypesLogic(T1 type, T2 matchType)
    {
        if (type.getFields().size() != matchType.getFields().size())
        {
            return false;
        }

        auto undefType = mlir_ts::UndefinedType::get(context);
        auto nullType = mlir_ts::NullType::get(context);
        auto undefPlaceHolderType = mlir_ts::UndefPlaceHolderType::get(context);

        std::function<bool(mlir::Type)> testType;
        testType = [&](mlir::Type type) {
            if (type == undefType || type == nullType || type == undefPlaceHolderType)
            {
                return true;
            }

            if (auto optType = type.dyn_cast_or_null<mlir_ts::OptionalType>())
            {
                return testType(optType.getElementType());
            }

            return false;
        };

        if (!llvm::all_of(llvm::zip(type.getFields(), matchType.getFields()),
                          [&](std::tuple<const ::mlir::typescript::FieldInfo &, const ::mlir::typescript::FieldInfo &> pair) {
                              return isSizeEqual(std::get<0>(pair).type, std::get<1>(pair).type) || testType(std::get<0>(pair).type);
                          }))
        {
            return false;
        }

        return true;
    }

    template <typename T1, typename T2> bool isSizeEqualLogic(T1 type, T2 matchType)
    {
        if (type.getFields().size() != matchType.getFields().size())
        {
            return false;
        }

        if (!llvm::all_of(llvm::zip(type.getFields(), matchType.getFields()),
                          [&](std::tuple<const ::mlir::typescript::FieldInfo &, const ::mlir::typescript::FieldInfo &> pair) {
                              return isSizeEqual(std::get<0>(pair).type, std::get<1>(pair).type);
                          }))
        {
            return false;
        }

        return true;
    }

    bool isCastableTypes(mlir::Type srcType, mlir::Type destType)
    {
        if (canCast(srcType, destType))
        {
            return true;
        }

        if (auto constTuple = srcType.dyn_cast_or_null<mlir_ts::ConstTupleType>())
        {
            if (auto matchConstTuple = destType.dyn_cast_or_null<mlir_ts::ConstTupleType>())
            {
                return isCastableTypesLogic(constTuple, matchConstTuple);
            }

            if (auto matchTuple = destType.dyn_cast_or_null<mlir_ts::TupleType>())
            {
                return isCastableTypesLogic(constTuple, matchTuple);
            }

            /*
            // TODO: finish it
            if (auto ifaceType = destType.dyn_cast_or_null<mlir_ts::InterfaceType>())
            {
            }
            */
        }

        if (auto tuple = srcType.dyn_cast_or_null<mlir_ts::TupleType>())
        {
            if (auto matchTuple = destType.dyn_cast_or_null<mlir_ts::TupleType>())
            {
                return isCastableTypesLogic(tuple, matchTuple);
            }
        }

        if (auto unionType = destType.dyn_cast_or_null<mlir_ts::UnionType>())
        {
            // calculate store size
            auto pred = [&](auto &item) { return isCastableTypes(item, srcType); };
            auto types = unionType.getTypes();
            if (std::find_if(types.begin(), types.end(), pred) == types.end())
            {
                return false;
            }
        }

        return false;
    }

    template <typename T> bool hasUndefinesLogic(T type)
    {
        auto undefType = mlir_ts::UndefinedType::get(context);
        auto nullType = mlir_ts::NullType::get(context);
        auto undefPlaceHolderType = mlir_ts::UndefPlaceHolderType::get(context);

        std::function<bool(mlir::Type)> testType;
        testType = [&](mlir::Type type) {
            if (type == undefType || type == nullType || type == undefPlaceHolderType)
            {
                return true;
            }

            if (auto optType = type.dyn_cast_or_null<mlir_ts::OptionalType>())
            {
                return testType(optType.getElementType());
            }

            return false;
        };

        if (!llvm::any_of(type.getFields(), [&](::mlir::typescript::FieldInfo fi) { return testType(fi.type); }))
        {
            return true;
        }

        return false;
    }

    bool hasUndefines(mlir::Type type)
    {
        if (auto constTuple = type.dyn_cast_or_null<mlir_ts::ConstTupleType>())
        {
            return hasUndefinesLogic(constTuple);
        }

        if (auto tuple = type.dyn_cast_or_null<mlir_ts::TupleType>())
        {
            return hasUndefinesLogic(tuple);
        }

        return false;
    }

    mlir::Type findBaseType(mlir::Type typeLeft, mlir::Type typeRight, mlir::Type defaultType = mlir::Type())
    {
        if (!typeLeft && !typeRight)
        {
            return mlir::Type();
        }

        if (typeLeft && !typeRight)
        {
            return typeLeft;
        }

        if (typeRight && !typeLeft)
        {
            return typeRight;
        }

        if (canCast(typeLeft, typeRight))
        {
            return typeRight;
        }

        if (defaultType)
        {
            return defaultType;
        }

        return typeLeft;
    }

    bool canCast(mlir::Type srcType, mlir::Type dstType)
    {
        if (!srcType || !dstType)
        {
            return false;
        }

        if (srcType == dstType)
        {
            return true;
        }

        if (srcType.isa<mlir::IntegerType>())
        {
            if (dstType.isa<mlir_ts::NumberType>())
            {
                return true;
            }
        }

        if (auto literalType = srcType.dyn_cast<mlir_ts::LiteralType>())
        {
            if (literalType.getElementType() == dstType)
            {
                return true;
            }
        }

        // wide range type can't be stored into literal
        if (auto literalType = dstType.dyn_cast<mlir_ts::LiteralType>())
        {
            if (literalType.getElementType() == srcType)
            {
                return true;
            }
        }

        return false;
    }

    bool isSizeEqual(mlir::Type srcType, mlir::Type dstType)
    {
        if (!srcType || !dstType)
        {
            return false;
        }

        if (srcType == dstType)
        {
            return true;
        }

        if (srcType == getBaseType(dstType))
        {
            return true;
        }

        if (auto constTuple = srcType.dyn_cast_or_null<mlir_ts::ConstTupleType>())
        {
            if (auto matchConstTuple = dstType.dyn_cast_or_null<mlir_ts::ConstTupleType>())
            {
                return isSizeEqualLogic(constTuple, matchConstTuple);
            }

            if (auto matchTuple = dstType.dyn_cast_or_null<mlir_ts::TupleType>())
            {
                return isSizeEqualLogic(constTuple, matchTuple);
            }
        }

        if (auto tuple = srcType.dyn_cast_or_null<mlir_ts::TupleType>())
        {
            if (auto matchTuple = dstType.dyn_cast_or_null<mlir_ts::TupleType>())
            {
                return isSizeEqualLogic(tuple, matchTuple);
            }
        }

        // TODO: finish it

        return false;
    }

    mlir::Type getBaseType(mlir::Type type)
    {
        if (auto literalType = type.dyn_cast_or_null<mlir_ts::LiteralType>())
        {
            return literalType.getElementType();
        }

        if (auto enumType = type.dyn_cast_or_null<mlir_ts::EnumType>())
        {
            return enumType.getElementType();
        }

        return type;
    }

    bool isUnionTypeNeedsTag(mlir_ts::UnionType unionType)
    {
        mlir::Type baseType;
        return isUnionTypeNeedsTag(unionType, baseType);
    }

    bool isUnionTypeNeedsTag(mlir_ts::UnionType unionType, mlir::Type &baseType)
    {
        bool anyNonTuple = false;
        bool allBaseTypes = true;
        for (auto type : unionType.getTypes())
        {
            anyNonTuple |= !type.isa<mlir_ts::TupleType>() || !type.isa<mlir_ts::ConstTupleType>();
            auto baseTypeOfCurrent = getBaseType(type);
            if (!baseType)
            {
                baseType = baseTypeOfCurrent;
                allBaseTypes = true;
            }
            else if (baseType != baseTypeOfCurrent)
            {
                allBaseTypes = false;
            }
        }

        if (allBaseTypes)
        {
            return false;
        }

        return anyNonTuple;
    }

    bool extendsType(mlir::Type srcType, mlir::Type extendType)
    {
        if (srcType == extendType)
        {
            return true;
        }

        if (auto unionType = extendType.dyn_cast_or_null<mlir_ts::UnionType>())
        {
            // calculate store size
            auto pred = [&](auto &item) { return extendsType(srcType, item); };
            auto types = unionType.getTypes();
            return std::find_if(types.begin(), types.end(), pred) != types.end();
        }

        return false;
    }

    mlir::Type getFirstNonNullUnionType(mlir_ts::UnionType unionType)
    {
        for (auto itemType : unionType.getTypes())
        {
            if (itemType.isa<mlir_ts::NullType>())
            {
                continue;
            }

            return itemType;
        }

        return mlir::Type();
    }
};

} // namespace typescript

#endif // MLIR_TYPESCRIPT_COMMONGENLOGIC_MLIRTYPEHELPER_H_
