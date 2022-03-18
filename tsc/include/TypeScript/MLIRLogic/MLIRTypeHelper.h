#ifndef MLIR_TYPESCRIPT_COMMONGENLOGIC_MLIRTYPEHELPER_H_
#define MLIR_TYPESCRIPT_COMMONGENLOGIC_MLIRTYPEHELPER_H_

#include "TypeScript/TypeScriptOps.h"
#include "TypeScript/DOM.h"
#include "TypeScript/MLIRLogic/MLIRTypeIterator.h"
#include "TypeScript/MLIRLogic/MLIRHelper.h"

#include "llvm/Support/Debug.h"

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

    mlir::IntegerType getU64Type()
    {
        return mlir::IntegerType::get(context, 64, mlir::IntegerType::Unsigned);
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
        return mlir::IntegerAttr::get(getI32Type(), mlir::APInt(32, value, true));
    }

    mlir::IntegerAttr getI64AttrValue(int64_t value)
    {
        return mlir::IntegerAttr::get(getI64Type(), mlir::APInt(64, value, true));
    }

    mlir::IntegerAttr getU64AttrValue(int64_t value)
    {
        return mlir::IntegerAttr::get(getU64Type(), mlir::APInt(64, value));
    }

    mlir::Type getIndexType()
    {
        return getI64Type();
    }

    mlir::Type getStructIndexType()
    {
        return getI32Type();
    }

    mlir::IntegerAttr getStructIndexAttrValue(int32_t value)
    {
        return mlir::IntegerAttr::get(getI32Type(), mlir::APInt(32, value));
    }

#ifdef ENABLE_TYPED_GC
    mlir::IntegerType getTypeBitmapValueType()
    {
        return getU64Type();
    }
#endif    

    bool isValueType(mlir::Type typeIn)
    {
        auto type = getBaseType(typeIn);
        return type && (type.isIntOrIndexOrFloat() || type.isa<mlir_ts::NumberType>() || type.isa<mlir_ts::BooleanType>() ||
                        type.isa<mlir_ts::TupleType>() || type.isa<mlir_ts::ConstTupleType>() || type.isa<mlir_ts::ConstArrayType>());
    }

    mlir::Attribute TupleFieldName(mlir::StringRef name)
    {
        assert(!name.empty());
        return mlir::StringAttr::get(context, name);
    }

    bool isUndefinedType(mlir::Type type)
    {
        if (auto optType = type.dyn_cast<mlir_ts::OptionalType>())
        {
            return optType == mlir_ts::UndefPlaceHolderType::get(context);
        }

        return false;
    }

    mlir::Type isBoundReference(mlir::Type elementType, bool &isBound)
    {
#ifdef USE_BOUND_FUNCTION_FOR_OBJECTS
        if (auto funcType = elementType.dyn_cast<mlir_ts::FunctionType>())
        {
            if (funcType.getNumInputs() > 0 &&
                (funcType.getInput(0).isa<mlir_ts::OpaqueType>() || funcType.getInput(0).isa<mlir_ts::ObjectType>()))
            {
                isBound = true;
                return mlir_ts::BoundFunctionType::get(context, funcType);
            }
        }
#endif

        isBound = false;
        return elementType;
    }

    mlir::Type wideStorageType(mlir::Type type)
    {
        auto actualType = type;
        if (actualType)
        {
            actualType = stripLiteralType(actualType);
            actualType = convertConstArrayTypeToArrayType(actualType);
            actualType = convertConstTupleTypeToTupleType(actualType);
        }

        return actualType;
    }    

    mlir::Type stripLiteralType(mlir::Type type)
    {
        if (auto literalType = type.dyn_cast<mlir_ts::LiteralType>())
        {
            return literalType.getElementType();
        }

        return type;
    }

    mlir::Type convertConstArrayTypeToArrayType(mlir::Type type)
    {
        if (auto constArrayType = type.dyn_cast<mlir_ts::ConstArrayType>())
        {
            return mlir_ts::ArrayType::get(constArrayType.getElementType());
        }

        return type;
    }

    mlir::Type convertConstTupleTypeToTupleType(mlir::Type type)
    {
        // tuple is value and copied already
        if (auto constTupleType = type.dyn_cast<mlir_ts::ConstTupleType>())
        {
            return mlir_ts::TupleType::get(context, constTupleType.getFields());
        }

        return type;
    }

    mlir::Type convertTupleTypeToConstTupleType(mlir::Type type)
    {
        // tuple is value and copied already
        if (auto tupleType = type.dyn_cast<mlir_ts::TupleType>())
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
        if (auto loadOp = mlir::dyn_cast<mlir_ts::LoadOp>(value.getDefiningOp()))
        {
            // this LoadOp will be removed later as unused
            auto refValue = loadOp.reference();
            return refValue;
        }

        return mlir::Value();
    }

    template <typename T1, typename T2> bool canCastFromToLogic(T1 type, T2 matchType)
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

            if (auto optType = type.dyn_cast<mlir_ts::OptionalType>())
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

    bool canCastFromTo(mlir::Type srcType, mlir::Type destType)
    {
        if (canWideTypeWithoutDataLoss(srcType, destType))
        {
            return true;
        }

        if (auto constTuple = srcType.dyn_cast<mlir_ts::ConstTupleType>())
        {
            if (auto matchConstTuple = destType.dyn_cast<mlir_ts::ConstTupleType>())
            {
                return canCastFromToLogic(constTuple, matchConstTuple);
            }

            if (auto matchTuple = destType.dyn_cast<mlir_ts::TupleType>())
            {
                return canCastFromToLogic(constTuple, matchTuple);
            }

            /*
            // TODO: finish it
            if (auto ifaceType = destType.dyn_cast<mlir_ts::InterfaceType>())
            {
            }
            */
        }

        if (auto tuple = srcType.dyn_cast<mlir_ts::TupleType>())
        {
            if (auto matchTuple = destType.dyn_cast<mlir_ts::TupleType>())
            {
                return canCastFromToLogic(tuple, matchTuple);
            }
        }

        if (auto unionType = destType.dyn_cast<mlir_ts::UnionType>())
        {
            // calculate store size
            auto pred = [&](auto &item) { return canCastFromTo(item, srcType); };
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

            if (auto optType = type.dyn_cast<mlir_ts::OptionalType>())
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
        if (auto constTuple = type.dyn_cast<mlir_ts::ConstTupleType>())
        {
            return hasUndefinesLogic(constTuple);
        }

        if (auto tuple = type.dyn_cast<mlir_ts::TupleType>())
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

        if (canWideTypeWithoutDataLoss(typeLeft, typeRight))
        {
            return typeRight;
        }

        if (canWideTypeWithoutDataLoss(typeRight, typeLeft))
        {
            return typeLeft;
        }

        if (defaultType)
        {
            return defaultType;
        }

        return typeLeft;
    }

    // TODO: review using canCast in detecting base Type, in case of "i32" & "number" + "number" & "i32"
    bool canWideTypeWithoutDataLoss(mlir::Type srcType, mlir::Type dstType)
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
            return canWideTypeWithoutDataLoss(literalType.getElementType(), dstType);
        }

        // wide range type can't be stored into literal
        if (auto optionalType = dstType.dyn_cast<mlir_ts::OptionalType>())
        {
            if (auto srcOptionalType = srcType.dyn_cast<mlir_ts::OptionalType>())
            {
                return canWideTypeWithoutDataLoss(srcOptionalType.getElementType(), optionalType.getElementType());
            }

            return canWideTypeWithoutDataLoss(srcType, optionalType.getElementType());
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

        if (auto constTuple = srcType.dyn_cast<mlir_ts::ConstTupleType>())
        {
            if (auto matchConstTuple = dstType.dyn_cast<mlir_ts::ConstTupleType>())
            {
                return isSizeEqualLogic(constTuple, matchConstTuple);
            }

            if (auto matchTuple = dstType.dyn_cast<mlir_ts::TupleType>())
            {
                return isSizeEqualLogic(constTuple, matchTuple);
            }
        }

        if (auto tuple = srcType.dyn_cast<mlir_ts::TupleType>())
        {
            if (auto matchTuple = dstType.dyn_cast<mlir_ts::TupleType>())
            {
                return isSizeEqualLogic(tuple, matchTuple);
            }
        }

        // TODO: finish it

        return false;
    }

    // TODO: obsolete, review usage (use stripLiteralType etc)
    mlir::Type getBaseType(mlir::Type type)
    {
        if (auto literalType = type.dyn_cast<mlir_ts::LiteralType>())
        {
            return literalType.getElementType();
        }

        if (auto enumType = type.dyn_cast<mlir_ts::EnumType>())
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
        auto storeType = getUnionTypeWithMerge(unionType.getTypes(), true);
        baseType = storeType;
        return storeType.isa<mlir_ts::UnionType>();
    }

    bool extendsType(mlir::Type srcType, mlir::Type extendType, llvm::StringMap<std::pair<ts::TypeParameterDOM::TypePtr,mlir::Type>> &typeParamsWithArgs)
    {
        if (srcType == extendType)
        {
            return true;
        }

        // to support infer types
        if (auto inferType = extendType.dyn_cast<mlir_ts::InferType>())
        {
            auto name = inferType.getElementType().cast<mlir_ts::NamedGenericType>().getName().getValue();
            auto currentType = srcType;

            auto existType = typeParamsWithArgs.lookup(name);
            if (existType.second)
            {
                auto defaultUnionType = getUnionType(existType.second, currentType);

                LLVM_DEBUG(llvm::dbgs() << "\n!! existing type: " << existType.second << " default type: " << defaultUnionType
                                        << "\n";);

                currentType = findBaseType(existType.second, currentType, defaultUnionType);

                LLVM_DEBUG(llvm::dbgs() << "\n!! result type: " << currentType << "\n";);
                typeParamsWithArgs[name].second = currentType;
            }
            else
            {
                // TODO: uncomment this line and find out what is the bug (+one more line)
                auto typeParam = std::make_shared<ts::TypeParameterDOM>(name.str());
                typeParamsWithArgs.insert({name, std::make_pair(typeParam, srcType)});
            }

            LLVM_DEBUG(llvm::dbgs() << "\n!! infered type for '" << name << "' = [" << typeParamsWithArgs[name].second << "]\n";);

            return true;
        }

        if (auto literalType = srcType.dyn_cast<mlir_ts::LiteralType>())
        {
            return extendsType(literalType.getElementType(), extendType, typeParamsWithArgs);
        }

        if (auto unionType = extendType.dyn_cast<mlir_ts::UnionType>())
        {
            auto pred = [&](auto &item) { return extendsType(srcType, item, typeParamsWithArgs); };
            auto types = unionType.getTypes();
            return std::find_if(types.begin(), types.end(), pred) != types.end();
        }

        if (auto srcArray = srcType.dyn_cast<mlir_ts::ArrayType>())
        {
            if (auto extArray = extendType.dyn_cast<mlir_ts::ArrayType>())
            {
                return extendsType(srcArray.getElementType(), extArray.getElementType(), typeParamsWithArgs);
            }
        }

        // TODO: finish Function Types, etc
        LLVM_DEBUG(llvm::dbgs() << "\n!! extendsType [false] for item type: " << srcType << " ext. type: " << extendType
                                << "\n";);
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

    // Union Type logic to merge types
    struct UnionTypeProcessContext
    {
        UnionTypeProcessContext() = default;

        bool isUndefined;
        bool isNullable;
        bool isNever;
        bool isAny;
        mlir::SmallPtrSet<mlir::Type, 2> types;
        mlir::SmallPtrSet<mlir::Type, 2> literalTypes;
    };

    mlir::LogicalResult processUnionTypeItem(mlir::Type type, UnionTypeProcessContext &unionContext)
    {
        if (type.isa<mlir_ts::UndefinedType>())
        {
            unionContext.isUndefined = true;
            return mlir::success();
        }

        if (type.isa<mlir_ts::NullType>())
        {
            unionContext.isNullable = true;
            return mlir::success();
        }

        if (type.isa<mlir_ts::AnyType>())
        {
            unionContext.isAny = true;
            return mlir::success();
        }

        if (type.isa<mlir_ts::NeverType>())
        {
            unionContext.isNever = true;
            return mlir::success();
        }

        if (auto literalType = type.dyn_cast<mlir_ts::LiteralType>())
        {
            unionContext.literalTypes.insert(literalType);
            return mlir::success();
        }

        if (auto optionalType = type.dyn_cast<mlir_ts::OptionalType>())
        {
            unionContext.isUndefined = true;
            unionContext.types.insert(optionalType.getElementType());
            return mlir::success();
        }

        if (auto unionType = type.dyn_cast<mlir_ts::UnionType>())
        {
            if (mlir::succeeded(processUnionType(unionType, unionContext)))
            {
                return mlir::success();
            }
        }

        unionContext.types.insert(type);
        return mlir::success();
    }

    mlir::LogicalResult processUnionType(mlir_ts::UnionType unionType, UnionTypeProcessContext &unionContext)
    {
        for (auto type : unionType.getTypes())
        {
            processUnionTypeItem(type, unionContext);
        }

        return mlir::success();
    }

    mlir::Type getUnionType(mlir::Type type1, mlir::Type type2, bool mergeLiterals = true)
    {
        if (canCastFromTo(type1, type2))
        {
            return type2;
        }

        if (canCastFromTo(type2, type1))
        {
            return type1;
        }

        mlir::SmallVector<mlir::Type> types;
        types.push_back(type1);
        types.push_back(type2);
        return getUnionTypeWithMerge(types, mergeLiterals);
    }

    mlir::Type getUnionTypeMergeTypes(UnionTypeProcessContext &unionContext, bool mergeLiterals = true)
    {
        // merge types with literal types
        for (auto literalType : unionContext.literalTypes)
        {
            if (mergeLiterals)
            {
                auto baseType = literalType.cast<mlir_ts::LiteralType>().getElementType();
                if (unionContext.types.count(baseType))
                {
                    continue;
                }

                unionContext.types.insert(baseType);
            }
            else
            {
                if (unionContext.types.count(literalType))
                {
                    continue;
                }

                unionContext.types.insert(literalType);
            }
        }

        auto isAllValueTypes = true;
        auto isAllLiteralTypes = true;

        mlir::SmallVector<mlir::Type> typesAll;
        for (auto type : unionContext.types)
        {
            typesAll.push_back(type);
            isAllValueTypes &= isValueType(type);
            isAllLiteralTypes &= type.isa<mlir_ts::LiteralType>();
        }

        if ((isAllValueTypes || isAllLiteralTypes) && unionContext.isNullable)
        {
            // return null type back
            typesAll.push_back(getNullType());
        }

        mlir::Type retType = typesAll.size() == 1 ? typesAll.front() : getUnionType(typesAll);
        if (unionContext.isUndefined)
        {
            return mlir_ts::OptionalType::get(retType);
        }

        return retType;
    }

    mlir::Type getUnionTypeWithMerge(mlir::ArrayRef<mlir::Type> types, bool mergeLiterals = true)
    {
        UnionTypeProcessContext unionContext = {};
        for (auto type : types)
        {
            if (!type)
            {
                llvm_unreachable("wrong type");
            }

            processUnionTypeItem(type, unionContext);

            // default wide types
            if (unionContext.isAny)
            {
                return mlir_ts::AnyType::get(context);
            }

            if (unionContext.isNever)
            {
                return mlir_ts::NeverType::get(context);
            }
        }

        return getUnionTypeMergeTypes(unionContext, mergeLiterals);
    }

    mlir::Type getUnionType(mlir::SmallVector<mlir::Type> &types)
    {
        if (types.size() == 0)
        {
            return mlir_ts::NeverType::get(context);
        }

        if (types.size() == 1)
        {
            return types.front();
        }

        return mlir_ts::UnionType::get(context, types);
    }

    mlir::Type getIntersectionType(mlir::Type type1, mlir::Type type2)
    {
        mlir::SmallVector<mlir::Type> types;
        types.push_back(type1);
        types.push_back(type2);
        return getIntersectionType(types);
    }

    mlir::Type getIntersectionType(mlir::SmallVector<mlir::Type> &types)
    {
        if (types.size() == 0)
        {
            return mlir_ts::NeverType::get(context);
        }

        if (types.size() == 1)
        {
            return types.front();
        }

        return mlir_ts::IntersectionType::get(context, types);
    }

    bool isGenericType(mlir::Type type)
    {
        MLIRTypeIteratorLogic iter{};
        return iter.some(type, [](mlir::Type type) { return type.isa<mlir_ts::NamedGenericType>(); });
    }

    mlir::Type mergeType(mlir::Type existType, mlir::Type currentType)
    {
        auto defaultUnionType = getUnionType(existType, currentType);

        LLVM_DEBUG(llvm::dbgs() << "\n!! existing type: " << existType << " default type: " << defaultUnionType
                                << "\n";);

        currentType = wideStorageType(currentType);
        currentType = findBaseType(existType, currentType, defaultUnionType);

        return currentType;
    }
};

} // namespace typescript

#endif // MLIR_TYPESCRIPT_COMMONGENLOGIC_MLIRTYPEHELPER_H_
