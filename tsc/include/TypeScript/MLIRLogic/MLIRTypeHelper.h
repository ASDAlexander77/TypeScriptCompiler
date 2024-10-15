#ifndef MLIR_TYPESCRIPT_COMMONGENLOGIC_MLIRTYPEHELPER_H_
#define MLIR_TYPESCRIPT_COMMONGENLOGIC_MLIRTYPEHELPER_H_

#include "TypeScript/TypeScriptOps.h"
#include "TypeScript/DOM.h"
#include "TypeScript/MLIRLogic/MLIRGenStore.h"
#include "TypeScript/MLIRLogic/MLIRTypeIterator.h"
#include "TypeScript/MLIRLogic/MLIRHelper.h"

#include "llvm/Support/Debug.h"
#include "llvm/ADT/APSInt.h"

#include <functional>

#define DEBUG_TYPE "mlir"

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

enum class ExtendsResult {
    False,
    True,
    Never,
    Any
};

inline bool isTrue(ExtendsResult val)
{
    return val == ExtendsResult::True || val == ExtendsResult::Any;
}

class MLIRTypeHelper
{
    mlir::MLIRContext *context;

  public:

    MLIRTypeHelper(
        mlir::MLIRContext *context)
        : context(context)
    {
    }

    MLIRTypeHelper(
        mlir::MLIRContext *context, 
        std::function<ClassInfo::TypePtr(StringRef)> getClassInfoByFullName,
        std::function<GenericClassInfo::TypePtr(StringRef)> getGenericClassInfoByFullName,
        std::function<InterfaceInfo::TypePtr(StringRef)> getInterfaceInfoByFullName,
        std::function<GenericInterfaceInfo::TypePtr(StringRef)> getGenericInterfaceInfoByFullName) 
        : context(context), 
          getClassInfoByFullName(getClassInfoByFullName),
          getGenericClassInfoByFullName(getGenericClassInfoByFullName),
          getInterfaceInfoByFullName(getInterfaceInfoByFullName),
          getGenericInterfaceInfoByFullName(getGenericInterfaceInfoByFullName)
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

    mlir_ts::TupleType getTupleType(mlir::SmallVector<mlir_ts::FieldInfo> &fieldInfos)
    {
        return mlir_ts::TupleType::get(context, fieldInfos);
    }    

    mlir_ts::TupleType getTupleType(mlir::ArrayRef<mlir::Type> types)
    {
        llvm::SmallVector<mlir_ts::FieldInfo> fields;
        for (auto type : types)
        {
            fields.push_back(mlir_ts::FieldInfo{nullptr, type, false});
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
        return mlir::IndexType::get(context);
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

    bool isNumericType(mlir::Type type)
    {
        return type && (type.isIntOrIndexOrFloat() || type.isa<mlir_ts::NumberType>());
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

    bool isNullableOrOptionalType(mlir::Type typeIn)
    {
        if (typeIn.isa<mlir_ts::NullType>() 
            || typeIn.isa<mlir_ts::UndefinedType>() 
            || typeIn.isa<mlir_ts::StringType>() 
            || typeIn.isa<mlir_ts::ObjectType>() 
            || typeIn.isa<mlir_ts::ClassType>() 
            || typeIn.isa<mlir_ts::InterfaceType>()
            || typeIn.isa<mlir_ts::OptionalType>()
            || typeIn.isa<mlir_ts::AnyType>()
            || typeIn.isa<mlir_ts::UnknownType>()
            || typeIn.isa<mlir_ts::RefType>()
            || typeIn.isa<mlir_ts::ValueRefType>())
        {
            return true;            
        }

        if (auto unionType = dyn_cast<mlir_ts::UnionType>(typeIn))
        {
            return llvm::any_of(unionType.getTypes(), [&](mlir::Type t) { return isNullableOrOptionalType(t); });
        }

        return false;
    }

    mlir::Type getElementType(mlir::Type type)
    {
        // TODO: get type of pointer element
        mlir::Type elementType;
        if (type)
        {
            if (type.isa<mlir_ts::StringType>())
            {
                elementType = mlir_ts::CharType::get(context);
            }
            else if (auto classType = type.dyn_cast<mlir_ts::ClassType>())
            {
                elementType = classType.getStorageType();
            }
            else if (auto refType = type.dyn_cast<mlir_ts::RefType>())
            {
                elementType = refType.getElementType();
            }
            else if (auto valueRefType = type.dyn_cast<mlir_ts::ValueRefType>())
            {
                elementType = valueRefType.getElementType();
            }
        }

        return elementType;
    }

    mlir::StringAttr getLabelName(mlir::Type typeIn)
    {
        if (typeIn.isIndex())
        {
            return mlir::StringAttr::get(context, std::string("index"));
        }
        else if (typeIn.isIntOrIndex())
        {
            return mlir::StringAttr::get(context, std::string("i") + std::to_string(typeIn.getIntOrFloatBitWidth()));
        }
        else if (typeIn.isIntOrFloat())
        {
            return mlir::StringAttr::get(context, std::string("f") + std::to_string(typeIn.getIntOrFloatBitWidth()));
        }
        else if (typeIn.isa<mlir_ts::NullType>())
        {
            return mlir::StringAttr::get(context, "null");
        }
        else if (typeIn.isa<mlir_ts::UndefinedType>()) 
        {
            return mlir::StringAttr::get(context, "undefined");
        }
        else if (typeIn.isa<mlir_ts::NumberType>())
        {
            return mlir::StringAttr::get(context, "number");
        }
        else if (typeIn.isa<mlir_ts::TupleType>())
        {
            return mlir::StringAttr::get(context, "tuple");
        }
        else if (typeIn.isa<mlir_ts::ObjectType>())
        {
            return mlir::StringAttr::get(context, "object");
        }
        else if (typeIn.isa<mlir_ts::StringType>()) 
        {
            return mlir::StringAttr::get(context, "string");
        }
        else if (typeIn.isa<mlir_ts::ObjectType>())
        {
            return mlir::StringAttr::get(context, "object");
        }
        else if (typeIn.isa<mlir_ts::ClassType>()) 
        {
            return mlir::StringAttr::get(context, "class");
        }
        else if (typeIn.isa<mlir_ts::InterfaceType>())
        {
            return mlir::StringAttr::get(context, "interface");
        }
        else if (typeIn.isa<mlir_ts::OptionalType>())
        {
            return mlir::StringAttr::get(context, "optional");
        }
        else if (typeIn.isa<mlir_ts::AnyType>())
        {
            return mlir::StringAttr::get(context, "any");
        }
        else if (typeIn.isa<mlir_ts::UnknownType>())
        {
            return mlir::StringAttr::get(context, "unknown");
        }
        else if (typeIn.isa<mlir_ts::RefType>())
        {
            return mlir::StringAttr::get(context, "ref");
        }
        else if (typeIn.isa<mlir_ts::ValueRefType>())
        {
            return mlir::StringAttr::get(context, "valueRef");
        }
        else if (typeIn.isa<mlir_ts::UnionType>())
        {
            return mlir::StringAttr::get(context, "union");
        }                

        return mlir::StringAttr::get(context, "<uknown>");
    }

    mlir::Attribute convertFromFloatAttrIntoType(mlir::Attribute attr, mlir::Type destType, mlir::OpBuilder &builder)
    {
        mlir::Type srcType;
        if (auto typedAttr = attr.dyn_cast<mlir::TypedAttr>())
        {
            srcType = typedAttr.getType();
        }

        auto isFloat = !destType.isIntOrIndex() && destType.isIntOrIndexOrFloat();
        if (!isFloat)
        {
            return mlir::Attribute();
        }

        mlir::Type type;
        auto width = destType.getIntOrFloatBitWidth();
        switch (width)
        {
            case 16: type = builder.getF16Type();
                break;
            case 32: type = builder.getF32Type();
                break;
            case 64: type = builder.getF64Type();
                break;
            case 128: type = builder.getF128Type();
                break;
        }
        
        // this is Int
        if (srcType.isIntOrIndex())
        {
            return builder.getFloatAttr(type, attr.cast<mlir::IntegerAttr>().getValue().signedRoundToDouble());
        }

        // this is Float
        if (srcType.isIntOrIndexOrFloat())
        {
            return builder.getFloatAttr(type, attr.cast<mlir::FloatAttr>().getValue().convertToDouble());
        }        

        return mlir::Attribute();
    }    

    mlir::Attribute convertAttrIntoType(mlir::Attribute attr, mlir::Type destType, mlir::OpBuilder &builder)
    {
        mlir::Type srcType;
        if (auto typedAttr = attr.dyn_cast<mlir::TypedAttr>())
        {
            srcType = typedAttr.getType();
        }

        LLVM_DEBUG(llvm::dbgs() << "\n!! attr from type: " << srcType << " to: " << destType << "\n";);      

        if (srcType == destType)
        {
            return attr;
        }

        if (destType.isa<mlir_ts::NumberType>())
        {
#ifdef NUMBER_F64
            return convertFromFloatAttrIntoType(attr, builder.getF64Type(), builder);
#else
            return convertFromFloatAttrIntoType(attr, builder.getF32Type(), builder);
#endif                
        }

        // this is Int
        if (destType.isIntOrIndex())
        {
            if (srcType.isIndex())
            {
                // index
                auto val = attr.cast<mlir::IntegerAttr>().getValue();
                APInt newVal(
                    destType.getIntOrFloatBitWidth(), 
                    val.getZExtValue());
                auto attrVal = builder.getIntegerAttr(destType, newVal);
                return attrVal;                
            }            
            else if (srcType.isIntOrIndex())
            {
                // integer
                auto val = attr.cast<mlir::IntegerAttr>().getValue();
                APInt newVal(
                    destType.getIntOrFloatBitWidth(), 
                    destType.isSignedInteger() ? val.getSExtValue() : val.getZExtValue(), 
                    destType.isSignedInteger());
                return builder.getIntegerAttr(destType, newVal);                
            }
            else if (srcType.isIntOrIndexOrFloat())
            {
                // float/double
                bool lossy;
                APSInt newVal(destType.getIntOrFloatBitWidth(), !destType.isIndex());
                attr.cast<mlir::FloatAttr>().getValue().convertToInteger(newVal, llvm::APFloatBase::rmNearestTiesToEven, &lossy);
                return builder.getIntegerAttr(destType, newVal);
            }
            else
            {
                llvm_unreachable("not implemented");
            }
        }        

        // this is Float
        if (destType.isIntOrIndexOrFloat())
        {
            return convertFromFloatAttrIntoType(attr, destType, builder);
        }        

        llvm_unreachable("not implemented");
    }

    mlir::Type wideStorageType(mlir::Type type)
    {
        auto actualType = type;
        if (actualType)
        {
            // we do not need to do it for UnionTypes to be able to validate which values it can have
            //actualType = mergeUnionType(actualType);
            actualType = stripLiteralType(actualType);
            actualType = removeConstType(actualType);
        }

        return actualType;
    }    

    mlir::Type removeConstType(mlir::Type type)
    {
        auto actualType = type;
        if (actualType)
        {
            // we do not need to do it for UnionTypes to be able to validate which values it can have
            //actualType = mergeUnionType(actualType);
            actualType = convertConstArrayTypeToArrayType(actualType);
            actualType = convertConstTupleTypeToTupleType(actualType);
        }

        return actualType;
    }    

    mlir::Type mergeUnionType(mlir::Type type)
    {
        if (auto unionType = type.dyn_cast<mlir_ts::UnionType>())
        {
            return getUnionTypeWithMerge(unionType);
        }

        return type;
    }

    mlir::Type stripLiteralType(mlir::Type type)
    {
        if (auto literalType = type.dyn_cast<mlir_ts::LiteralType>())
        {
            return literalType.getElementType();
        }

        return type;
    }

    mlir::Type stripOptionalType(mlir::Type type)
    {
        if (auto optType = type.dyn_cast<mlir_ts::OptionalType>())
        {
            return optType.getElementType();
        }

        return type;
    }    

    mlir::Type stripRefType(mlir::Type type)
    {
        if (auto refType = type.dyn_cast<mlir_ts::RefType>())
        {
            return refType.getElementType();
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

    mlir_ts::TupleType convertConstTupleTypeToTupleType(mlir_ts::ConstTupleType constTupleType)
    {
        return mlir_ts::TupleType::get(context, constTupleType.getFields());
    }

    mlir_ts::ConstTupleType convertTupleTypeToConstTupleType(mlir_ts::TupleType tupleType)
    {
        return mlir_ts::ConstTupleType::get(context, tupleType.getFields());
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
        auto newFuncType = mlir_ts::FunctionType::get(context, args, funcType.getResults(), funcType.isVarArg());
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
        return mlir_ts::FunctionType::get(context, funcArgTypes, funcType.getResults(), funcType.isVarArg());
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

        return mlir_ts::FunctionType::get(context, funcArgTypes, funcType.getResults(), funcType.isVarArg());
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

    // TODO: review virtual calls
    bool isNoneType(mlir::Type type)
    {
        return !type || type.isa<mlir::NoneType>();
    }

    bool isEmptyTuple(mlir::Type type)
    {
        if (auto tupleType = dyn_cast<mlir_ts::TupleType>(type))
        {
            return tupleType.getFields().size() == 0;
        }

        if (auto constTupleType = dyn_cast<mlir_ts::ConstTupleType>(type))
        {
            return constTupleType.getFields().size() == 0;
        }

        return false;
    }

    bool isVirtualFunctionType(mlir::Value actualFuncRefValue) 
    {
        auto attrName = StringRef(IDENTIFIER_ATTR_NAME);
        auto virtAttrName = StringRef(VIRTUALFUNC_ATTR_NAME);
        auto definingOp = actualFuncRefValue.getDefiningOp();
        return (isNoneType(actualFuncRefValue.getType()) || definingOp->hasAttrOfType<mlir::BoolAttr>(virtAttrName)) 
            && definingOp->hasAttrOfType<mlir::FlatSymbolRefAttr>(attrName);
    }

    // TODO: how about multi-index?
    mlir::Type getIndexSignatureElementType(mlir::Type indexSignatureType)
    {
        if (auto funcType = indexSignatureType.dyn_cast<mlir_ts::FunctionType>())
        {
            if (funcType.getNumInputs() == 1 && funcType.getNumResults() == 1 && isNumericType(funcType.getInput(0)))
            {
                return funcType.getResult(0);
            }
        }

        return mlir::Type();
    }

    mlir::Type getIndexSignatureType(mlir::Type elementType)
    {
        if (!elementType)
        {
            return mlir::Type();
        }

        return mlir_ts::FunctionType::get(context, {mlir_ts::NumberType::get(context)}, {elementType}, false);
    }

    bool isAnyFunctionType(mlir::Type funcType, bool stripRefTypeOpt = false)
    {
        funcType = stripOptionalType(funcType);    
        if (stripRefTypeOpt)
        {
            funcType = stripRefType(funcType);    
        }

        bool isFuncType = true;
        mlir::TypeSwitch<mlir::Type>(funcType)
            .Case<mlir_ts::FunctionType>([&](auto) {})
            .Case<mlir_ts::HybridFunctionType>([&](auto) {})
            .Case<mlir_ts::BoundFunctionType>([&](auto) {})
            .Case<mlir_ts::ConstructFunctionType>([&](auto) {})
            .Case<mlir_ts::ExtensionFunctionType>([&](auto) {})
            .Default([&](auto type) {
                isFuncType = false;
            });

        //LLVM_DEBUG(llvm::dbgs() << "\n!! isAnyFunctionType for " << funcType << " = " << isFuncType << "\n";);

        return isFuncType;
    }

    mlir::Type getReturnTypeFromFuncRef(mlir::Type funcType)
    {
        auto types = getReturnsFromFuncRef(funcType);
        if (types.size() > 0)
        {
            return types.front();
        }

        return mlir::Type();
    }

    bool hasReturnTypeFromFuncRef(mlir::Type funcType)
    {
        return getReturnsFromFuncRef(funcType, true).size() > 0;
    }

    mlir::ArrayRef<mlir::Type> getReturnsFromFuncRef(mlir::Type funcType, bool noError = false)
    {
        mlir::ArrayRef<mlir::Type> returnTypes;

        funcType = stripOptionalType(funcType);    

        auto f = [&](auto calledFuncType) { returnTypes = calledFuncType.getResults(); };

        mlir::TypeSwitch<mlir::Type>(funcType)
            .Case<mlir_ts::FunctionType>([&](auto calledFuncType) { f(calledFuncType); })
            .Case<mlir_ts::HybridFunctionType>([&](auto calledFuncType) { f(calledFuncType); })
            .Case<mlir_ts::BoundFunctionType>([&](auto calledFuncType) { f(calledFuncType); })
            .Case<mlir_ts::ConstructFunctionType>([&](auto calledFuncType) { f(calledFuncType); })
            .Case<mlir_ts::ExtensionFunctionType>([&](auto calledFuncType) { f(calledFuncType); })
            .Default([&](auto type) {
                if (noError)
                {
                    return;
                }

                LLVM_DEBUG(llvm::dbgs() << "\n!! getReturnTypeFromFuncRef is not implemented for " << type << "\n";);
                llvm_unreachable("not implemented");
            });

        return returnTypes;
    }

    mlir::Type getParamFromFuncRef(mlir::Type funcType, int index)
    {
        mlir::Type paramType;

        funcType = stripOptionalType(funcType);          

        auto f = [&](auto calledFuncType) { return (index >= (int) calledFuncType.getNumInputs()) ? mlir::Type() : calledFuncType.getInput(index); };

        mlir::TypeSwitch<mlir::Type>(funcType)
            .Case<mlir_ts::FunctionType>([&](auto calledFuncType) { paramType = f(calledFuncType); })
            .Case<mlir_ts::HybridFunctionType>([&](auto calledFuncType) { paramType = f(calledFuncType); })
            .Case<mlir_ts::BoundFunctionType>([&](auto calledFuncType) { paramType = f(calledFuncType); })
            .Case<mlir_ts::ConstructFunctionType>([&](auto calledFuncType) { paramType = f(calledFuncType); })
            .Case<mlir_ts::ExtensionFunctionType>([&](auto calledFuncType) { paramType = f(calledFuncType); })
            .Case<mlir::NoneType>([&](auto calledFuncType) { paramType = mlir::NoneType::get(context); })
            .Default([&](auto type) {
                LLVM_DEBUG(llvm::dbgs() << "\n!! getParamFromFuncRef is not implemented for " << type << "\n";);
                llvm_unreachable("not implemented");
            });

        return paramType;
    }

    // TODO: rename and put in helper class
    mlir::Type getFirstParamFromFuncRef(mlir::Type funcType)
    {
        mlir::Type paramType;

        funcType = stripOptionalType(funcType);    

        auto f = [&](auto calledFuncType) { return calledFuncType.getInputs().front(); };

        mlir::TypeSwitch<mlir::Type>(funcType)
            .Case<mlir_ts::FunctionType>([&](auto calledFuncType) { paramType = f(calledFuncType); })
            .Case<mlir_ts::HybridFunctionType>([&](auto calledFuncType) { paramType = f(calledFuncType); })
            .Case<mlir_ts::BoundFunctionType>([&](auto calledFuncType) { paramType = f(calledFuncType); })
            .Case<mlir_ts::ConstructFunctionType>([&](auto calledFuncType) { paramType = f(calledFuncType); })
            .Case<mlir_ts::ExtensionFunctionType>([&](auto calledFuncType) { paramType = f(calledFuncType); })
            .Case<mlir::NoneType>([&](auto calledFuncType) { paramType = mlir::NoneType::get(context); })
            .Default([&](auto type) {
                LLVM_DEBUG(llvm::dbgs() << "\n!! getFirstParamFromFuncRef is not implemented for " << type << "\n";);
                llvm_unreachable("not implemented");
            });

        return paramType;
    }

    // TODO: rename and put in helper class
    mlir::ArrayRef<mlir::Type> getParamsFromFuncRef(mlir::Type funcType)
    {
        mlir::ArrayRef<mlir::Type> paramsType;
        if (!funcType)
        {
            return paramsType;
        }

        funcType = stripOptionalType(funcType);       

        mlir::TypeSwitch<mlir::Type>(funcType)
            .Case<mlir_ts::FunctionType>([&](auto calledFuncType) { paramsType = calledFuncType.getInputs(); })
            .Case<mlir_ts::HybridFunctionType>([&](auto calledFuncType) { paramsType = calledFuncType.getInputs(); })
            .Case<mlir_ts::BoundFunctionType>([&](auto calledFuncType) { paramsType = calledFuncType.getInputs(); })
            .Case<mlir_ts::ConstructFunctionType>([&](auto calledFuncType) { paramsType = calledFuncType.getInputs(); })
            .Case<mlir_ts::ExtensionFunctionType>([&](auto calledFuncType) { paramsType = calledFuncType.getInputs(); })
            .Case<mlir::NoneType>([&](auto calledFuncType) { paramsType = mlir::NoneType::get(context); })
            .Default([&](auto type) {
                LLVM_DEBUG(llvm::dbgs() << "\n!! getParamsFromFuncRef is not implemented for " << type << "\n";);
                llvm_unreachable("not implemented");
            });

        return paramsType;
    }

    mlir::Type getParamsTupleTypeFromFuncRef(mlir::Type funcType)
    {
        mlir::Type paramsType;
        if (!funcType)
        {
            return paramsType;
        }

        funcType = stripOptionalType(funcType);     

        auto f = [&](auto calledFuncType) {
            SmallVector<mlir_ts::FieldInfo> fieldInfos;
            for (auto param : calledFuncType.getInputs())
            {
                fieldInfos.push_back({mlir::Attribute(), param, false});
            }

            return getTupleType(fieldInfos);
        };

        mlir::TypeSwitch<mlir::Type>(funcType)
            .Case<mlir_ts::FunctionType>([&](auto calledFuncType) { paramsType = f(calledFuncType); })
            .Case<mlir_ts::HybridFunctionType>([&](auto calledFuncType) { paramsType = f(calledFuncType); })
            .Case<mlir_ts::BoundFunctionType>([&](auto calledFuncType) { paramsType = f(calledFuncType); })
            .Case<mlir_ts::ConstructFunctionType>([&](auto calledFuncType) { paramsType = f(calledFuncType); })
            .Case<mlir_ts::ExtensionFunctionType>([&](auto calledFuncType) { paramsType = f(calledFuncType); })
            .Case<mlir::NoneType>([&](auto calledFuncType) { paramsType = mlir::NoneType::get(context); })
            .Default([&](auto type) {
                LLVM_DEBUG(llvm::dbgs() << "\n!! getParamsTupleTypeFromFuncRef is not implemented for " << type
                                        << "\n";);
                // TODO: uncomment it when rewrite code "5.ToString()"
                //llvm_unreachable("not implemented");
            });

        return paramsType;
    }

    bool getVarArgFromFuncRef(mlir::Type funcType)
    {
        bool isVarArg = false;

        funcType = stripOptionalType(funcType);    

        mlir::TypeSwitch<mlir::Type>(funcType)
            .Case<mlir_ts::FunctionType>([&](auto calledFuncType) { isVarArg = calledFuncType.isVarArg(); })
            .Case<mlir_ts::HybridFunctionType>([&](auto calledFuncType) { isVarArg = calledFuncType.isVarArg(); })
            .Case<mlir_ts::BoundFunctionType>([&](auto calledFuncType) { isVarArg = calledFuncType.isVarArg(); })
            .Case<mlir_ts::ConstructFunctionType>([&](auto calledFuncType) { isVarArg = calledFuncType.isVarArg(); })
            .Case<mlir_ts::ExtensionFunctionType>([&](auto calledFuncType) { isVarArg = calledFuncType.isVarArg(); })
            .Case<mlir::NoneType>([&](auto calledFuncType) {})
            .Default([&](auto type) {
                LLVM_DEBUG(llvm::dbgs() << "\n!! getVarArgFromFuncRef is not implemented for " << type << "\n";);
                llvm_unreachable("not implemented");
            });

        LLVM_DEBUG(llvm::dbgs() << "\n!! getVarArgFromFuncRef for " << funcType << " = " << isVarArg << "\n";);

        return isVarArg;
    }

    mlir::Type getOmitThisFunctionTypeFromFuncRef(mlir::Type funcType)
    {
        mlir::Type paramsType;

        auto isOptType = funcType.isa<mlir_ts::OptionalType>();

        funcType = stripOptionalType(funcType);    

        auto f = [&](auto calledFuncType) {
            using t = decltype(calledFuncType);
            SmallVector<mlir::Type> newInputTypes;
            if (calledFuncType.getInputs().size() > 0)
            {
                newInputTypes.append(calledFuncType.getInputs().begin() + 1, calledFuncType.getInputs().end());
            }

            auto newType = t::get(context, newInputTypes, calledFuncType.getResults(), calledFuncType.isVarArg());
            return newType;
        };

        mlir::TypeSwitch<mlir::Type>(funcType)
            .Case<mlir_ts::FunctionType>([&](auto calledFuncType) { paramsType = f(calledFuncType); })
            .Case<mlir_ts::HybridFunctionType>([&](auto calledFuncType) { paramsType = f(calledFuncType); })
            .Case<mlir_ts::BoundFunctionType>([&](auto calledFuncType) { paramsType = f(calledFuncType); })
            .Case<mlir_ts::ExtensionFunctionType>([&](auto calledFuncType) { paramsType = f(calledFuncType); })
            .Case<mlir::NoneType>([&](auto calledFuncType) { paramsType = mlir::NoneType::get(context); })
            .Default([&](auto type) { 
                llvm_unreachable("not implemented"); 
            });

        return isOptType ? mlir_ts::OptionalType::get(paramsType) : paramsType;
    }

    MatchResult TestFunctionTypesMatch(
        mlir::ArrayRef<mlir::Type> inInputs, 
        mlir::ArrayRef<mlir::Type> resInputs, 
        mlir::ArrayRef<mlir::Type> inResults, 
        mlir::ArrayRef<mlir::Type> resResults, 
        bool inVarArgs, 
        bool resVarArgs, 
        unsigned startParam = 0)
    {
        // TODO: make 1 common function
        if (inInputs.size() != resInputs.size())
        {
            return {MatchResultType::NotMatchArgCount, 0};
        }

        for (unsigned i = startParam, e = inInputs.size(); i != e; ++i)
        {
            if (inInputs[i] != resInputs[i])
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

        auto inRetCount = inResults.size();
        auto resRetCount = resResults.size();

        auto noneType = mlir::NoneType::get(context);
        auto voidType = mlir_ts::VoidType::get(context);

        for (auto retType : inResults)
        {
            auto isVoid = !retType || retType == noneType || retType == voidType;
            if (isVoid)
            {
                inRetCount--;
            }
        }

        for (auto retType : resResults)
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
            auto inRetType = inResults[i];
            auto resRetType = resResults[i];

            auto isInVoid = !inRetType || inRetType == noneType || inRetType == voidType;
            auto isResVoid = !resRetType || resRetType == noneType || resRetType == voidType;
            if (!isInVoid && !isResVoid && inRetType != resRetType)
            {
                return {MatchResultType::NotMatchResult, i};
            }
        }

        return {MatchResultType::Match, 0};        
    }    

    MatchResult TestFunctionTypesMatch(mlir_ts::FunctionType inFuncType, mlir_ts::FunctionType resFuncType, unsigned startParam = 0)
    {
        return TestFunctionTypesMatch(
            inFuncType.getInputs(), 
            resFuncType.getInputs(), 
            inFuncType.getResults(), 
            resFuncType.getResults(), 
            inFuncType.isVarArg(),
            resFuncType.isVarArg(),
            startParam);
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

    // it has different code to MLIRCodeLogic - GetReferenceOfLoadOp
    mlir::Value GetReferenceOfLoadOp(mlir::Value value)
    {
        if (auto loadOp = mlir::dyn_cast<mlir_ts::LoadOp>(value.getDefiningOp()))
        {
            // this LoadOp will be removed later as unused
            auto refValue = loadOp.getReference();
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

        // TODO: review code in case of using "undefined" & "null"
        auto undefType = mlir_ts::UndefinedType::get(context);
        auto nullType = mlir_ts::NullType::get(context);

        std::function<bool(mlir::Type)> testType;
        testType = [&](mlir::Type type) {
            if (type == undefType || type == nullType)
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

    mlir::LogicalResult canCastTupleToInterface(mlir_ts::TupleType tupleStorageType,
                                                InterfaceInfo::TypePtr newInterfacePtr)
    {
        SmallVector<VirtualMethodOrFieldInfo> virtualTable;
        auto location = mlir::UnknownLoc::get(context);
        return getInterfaceVirtualTableForObject(location, tupleStorageType, newInterfacePtr, virtualTable, true);
    }

    mlir::LogicalResult getInterfaceVirtualTableForObject(mlir::Location location, mlir_ts::TupleType tupleStorageType,
                                                          InterfaceInfo::TypePtr newInterfacePtr,
                                                          SmallVector<VirtualMethodOrFieldInfo> &virtualTable,
                                                          bool suppressErrors = false)
    {

        MethodInfo emptyMethod;
        mlir_ts::FieldInfo emptyFieldInfo;
        mlir_ts::FieldInfo missingFieldInfo;

        auto result = newInterfacePtr->getVirtualTable(
            virtualTable,
            [&](mlir::Attribute id, mlir::Type fieldType, bool isConditional) -> mlir_ts::FieldInfo {
                auto foundIndex = tupleStorageType.getIndex(id);
                if (foundIndex >= 0)
                {
                    auto foundField = tupleStorageType.getFieldInfo(foundIndex);
                    auto test = foundField.type.isa<mlir_ts::FunctionType>() && fieldType.isa<mlir_ts::FunctionType>()
                                    ? TestFunctionTypesMatchWithObjectMethods(foundField.type, fieldType).result ==
                                          MatchResultType::Match
                                    : fieldType == foundField.type;
                    if (!test)
                    {
                        LLVM_DEBUG(llvm::dbgs() << "field " << id << " not matching type: " << fieldType << " and "
                                            << foundField.type << " in interface '" << newInterfacePtr->fullName
                                            << "' for object '" << tupleStorageType << "'";);                                    

                        if (!suppressErrors)
                        {
                            emitError(location) << "field " << id << " not matching type: " << fieldType << " and "
                                                << foundField.type << " in interface '" << newInterfacePtr->fullName
                                                << "' for object '" << tupleStorageType << "'";
                        }

                        return emptyFieldInfo;
                    }

                    return foundField;
                }

                LLVM_DEBUG(llvm::dbgs() << "field can't be found " << id << " for interface '"
                                    << newInterfacePtr->fullName << "' in object '" << tupleStorageType << "'";);

                if (!isConditional)
                {
                    emitError(location) << "field can't be found " << id << " for interface '"
                                        << newInterfacePtr->fullName << "' in object '" << tupleStorageType << "'";
                }

                return emptyFieldInfo;
            },
            [&](std::string methodName, mlir_ts::FunctionType funcType, bool isConditional, int interfacePosIndex) -> MethodInfo & {
                llvm_unreachable("not implemented yet");
                /*
                auto id = MLIRHelper::TupleFieldName(methodName, funcType.getContext());
                auto foundIndex = tupleStorageType.getIndex(id);
                if (foundIndex >= 0)
                {
                    auto foundField = tupleStorageType.getFieldInfo(foundIndex);
                    auto test = foundField.type.isa<mlir_ts::FunctionType>()
                                    ? TestFunctionTypesMatchWithObjectMethods(foundField.type, funcType).result ==
                                          MatchResultType::Match
                                    : funcType == foundField.type;
                    if (!test)
                    {
                        LLVM_DEBUG(llvm::dbgs() << "method " << id << " not matching type: " << funcType << " and "
                                            << foundField.type << " in interface '" << newInterfacePtr->fullName
                                            << "' for object '" << tupleStorageType << "'";);                                    

                        if (!suppressErrors)
                        {
                            emitError(location) << "method " << id << " not matching type: " << funcType << " and "
                                                << foundField.type << " in interface '" << newInterfacePtr->fullName
                                                << "' for object '" << tupleStorageType << "'";
                        }

                        return emptyMethod;
                    }           

                    MethodInfo foundMethod{};
                    foundMethod.name = methodName;
                    foundMethod.funcType = foundField.type.cast<mlir_ts::FunctionType>();
                    // TODO: you need to load function from object
                    //foundMethod.funcOp
                    return foundMethod;
                }
                */
            });

        return result;
    }

    bool canCastFromTo(mlir::Type srcType, mlir::Type destType)
    {
        if (srcType == destType)
        {
            return true;
        }
        
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

            if (auto ifaceType = destType.dyn_cast<mlir_ts::InterfaceType>())
            {
                if (getInterfaceInfoByFullName)
                {
                    if (auto ifaceTypeInfo = getInterfaceInfoByFullName(ifaceType.getName().getValue()))
                    {
                        return mlir::succeeded(canCastTupleToInterface(convertConstTupleTypeToTupleType(constTuple), ifaceTypeInfo));
                    }
                    else
                    {
                        assert(false);                    
                    }
                }                
                else
                {
                    // TODO: we have Cast verification which does not have connected getInterfaceInfoByFullName
                    return false;
                }
            }
        }

        if (auto tuple = srcType.dyn_cast<mlir_ts::TupleType>())
        {
            if (auto matchTuple = destType.dyn_cast<mlir_ts::TupleType>())
            {
                return canCastFromToLogic(tuple, matchTuple);
            }

            if (auto ifaceType = destType.dyn_cast<mlir_ts::InterfaceType>())
            {
                if (getInterfaceInfoByFullName)
                {
                    if (auto ifaceTypeInfo = getInterfaceInfoByFullName(ifaceType.getName().getValue()))
                    {
                        return mlir::succeeded(canCastTupleToInterface(tuple, ifaceTypeInfo));
                    }
                    else
                    {
                        assert(false);                    
                    }
                }
                else
                {
                    // TODO: we have Cast verification which does not have connected getInterfaceInfoByFullName
                    return false;
                }
            }
        }

        if (isAnyFunctionType(srcType) && isAnyFunctionType(destType))
        {
            auto srcTypeUnwrapped = stripOptionalType(srcType);
            auto destTypeUnwrapped = stripOptionalType(destType);
            if (!srcTypeUnwrapped.isa<mlir_ts::FunctionType>() && destTypeUnwrapped.isa<mlir_ts::FunctionType>())
            {
                // because of data loss we need to return false;
                return false;
            }

            auto srcInputs = getParamsFromFuncRef(srcType);
            auto destInputs = getParamsFromFuncRef(destType);
            auto srcResults = getReturnsFromFuncRef(srcType);
            auto destResults = getReturnsFromFuncRef(destType);
            auto srcIsVarArg = getVarArgFromFuncRef(srcType);
            auto destIsVarArg = getVarArgFromFuncRef(destType);
            return TestFunctionTypesMatch(srcInputs, destInputs, srcResults, destResults, srcIsVarArg, destIsVarArg).result == MatchResultType::Match;
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

        std::function<bool(mlir::Type)> testType;
        testType = [&](mlir::Type type) {
            if (type == undefType || type == nullType)
            {
                return true;
            }

            if (auto optType = type.dyn_cast<mlir_ts::OptionalType>())
            {
                return testType(optType.getElementType());
            }

            return false;
        };

        return llvm::any_of(type.getFields(), [&](::mlir::typescript::FieldInfo fi) { return testType(fi.type); });
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

    mlir::Type mergeIntTypes(mlir::Type typeLeft, mlir::Type typeRight, bool& found)
    {
        found = false;
        auto intTypeLeft = typeLeft.dyn_cast<mlir::IntegerType>();
        auto intTypeRight = typeRight.dyn_cast<mlir::IntegerType>();
        if (intTypeLeft && intTypeRight)
        {
            auto width = std::max(intTypeLeft.getIntOrFloatBitWidth(), intTypeRight.getIntOrFloatBitWidth());
            auto maxSignedWidth = std::max(
                intTypeLeft.isSigned() ? intTypeLeft.getIntOrFloatBitWidth() : 0, 
                intTypeRight.isSigned() ? intTypeRight.getIntOrFloatBitWidth() : 0);
            auto maxUnsignedWidth = std::max(
                intTypeLeft.isUnsigned() ? intTypeLeft.getIntOrFloatBitWidth() : 0, 
                intTypeRight.isUnsigned() ? intTypeRight.getIntOrFloatBitWidth() : 0);

            auto anySigned = intTypeLeft.isSigned() || intTypeRight.isSigned();
            auto anyUnsigned = intTypeLeft.isUnsigned() || intTypeRight.isUnsigned();

            if (anySigned && anyUnsigned && maxSignedWidth <= maxUnsignedWidth)
            {
                // if we have 64 - we can't extend it
                if (width > 32)
                    return mlir::Type();

                width *= 2;
            }

            found = true;
            return mlir::IntegerType::get(context, 
                width, 
                    anySigned 
                        ? mlir::IntegerType::Signed 
                        : anyUnsigned 
                            ? mlir::IntegerType::Unsigned 
                            : mlir::IntegerType::Signless);
        }

        return mlir::Type();
    }

    mlir::Type findBaseType(mlir::Type typeLeft, mlir::Type typeRight, bool& found, mlir::Type defaultType = mlir::Type())
    {
        if (!typeLeft && !typeRight)
        {
            return mlir::Type();
        }

        found = true;
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

        found = false;
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

        // we should not treat boolean as integer
        // if (srcType.isa<mlir_ts::BooleanType>())
        // {
        //     if (dstType.isa<mlir::IntegerType>() && dstType.getIntOrFloatBitWidth() > 0)
        //     {
        //         return true;
        //     }

        //     if (dstType.isa<mlir_ts::NumberType>())
        //     {
        //         return true;
        //     }
        // }

        // but we can't cast TypePredicate to boolean as we will lose the information about type
        if (srcType.isa<mlir_ts::BooleanType>())
        {
            if (dstType.isa<mlir_ts::TypePredicateType>())
            {
                return true;
            }
        }        

        if (auto dstEnumType = dstType.dyn_cast<mlir_ts::EnumType>())
        {
            if (dstEnumType.getElementType() == srcType)
            {
                return true;
            }
        }          

        if (auto literalType = srcType.dyn_cast<mlir_ts::LiteralType>())
        {
            return canWideTypeWithoutDataLoss(literalType.getElementType(), dstType);
        }

        if (auto enumType = srcType.dyn_cast<mlir_ts::EnumType>())
        {
            return canWideTypeWithoutDataLoss(enumType.getElementType(), dstType);
        }        

        // wide range type can't be stored into literal
        if (auto optionalType = dstType.dyn_cast<mlir_ts::OptionalType>())
        {
            if (auto srcOptionalType = srcType.dyn_cast<mlir_ts::OptionalType>())
            {
                return canWideTypeWithoutDataLoss(srcOptionalType.getElementType(), optionalType.getElementType());
            }

            if (auto undefType = srcType.dyn_cast<mlir_ts::UndefinedType>())
            {
                return true;
            }

            return canWideTypeWithoutDataLoss(srcType, optionalType.getElementType());
        }

        if (auto arrayType = dstType.dyn_cast<mlir_ts::ArrayType>())
        {
            if (auto constArrayType = srcType.dyn_cast<mlir_ts::ConstArrayType>())
            {
                if (constArrayType.getSize() == 0)
                {
                    return true;
                }

                return canWideTypeWithoutDataLoss(constArrayType.getElementType(), arrayType.getElementType());
            }
        }        

        // native types
        if (auto destIntType = dstType.dyn_cast<mlir::IntegerType>())
        {
            if (auto srcIntType = srcType.dyn_cast<mlir::IntegerType>())
            {
                if ((srcIntType.getSignedness() == destIntType.getSignedness() || srcIntType.isSignless() || destIntType.isSignless())
                    && srcIntType.getIntOrFloatBitWidth() <= destIntType.getIntOrFloatBitWidth())
                {
                    return true;
                }

                if (destIntType.getIntOrFloatBitWidth() - srcIntType.getIntOrFloatBitWidth() > 1)
                {
                    return true;
                }

                if (srcIntType.getIntOrFloatBitWidth() == destIntType.getIntOrFloatBitWidth()
                    && srcIntType.isSigned() == destIntType.isSigned())
                {
                    return true;
                }

                return false;
            }    
        }          

        if (auto destFloatType = dstType.dyn_cast<mlir::FloatType>())
        {
            if (auto srcFloatType = srcType.dyn_cast<mlir::FloatType>())
            {
                if (srcFloatType.getIntOrFloatBitWidth() <= destFloatType.getIntOrFloatBitWidth())
                {
                    return true;
                }
            }

            if (auto srcIntType = srcType.dyn_cast<mlir::IntegerType>())
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

    ExtendsResult appendInferTypeToContext(mlir::Type srcType, mlir_ts::InferType inferType, llvm::StringMap<std::pair<ts::TypeParameterDOM::TypePtr,mlir::Type>> &typeParamsWithArgs, bool useTupleType = false)
    {
        auto name = inferType.getElementType().cast<mlir_ts::NamedGenericType>().getName().getValue();
        auto currentType = srcType;

        auto existType = typeParamsWithArgs.lookup(name);
        if (existType.second)
        {
            auto namedType = existType.second.dyn_cast<mlir_ts::NamedGenericType>();
            if (namedType && namedType.getName().getValue().equals(name))
            {
                // replace generic type
                typeParamsWithArgs[name].second = currentType;
            }
            else
            {
                if (useTupleType)
                {
                    SmallVector<mlir_ts::FieldInfo> fieldInfos;

                    if (auto tupleType = existType.second.dyn_cast<mlir_ts::TupleType>())
                    {
                        for (auto param : tupleType.getFields())
                        {
                            fieldInfos.push_back(param);
                        }
                    }
                    else
                    {
                        fieldInfos.push_back({mlir::Attribute(), existType.second, false});    
                    }

                    fieldInfos.push_back({mlir::Attribute(), currentType, false});

                    currentType = getTupleType(fieldInfos);                    
                }
                else
                {
                    auto defaultUnionType = getUnionType(existType.second, currentType);

                    LLVM_DEBUG(llvm::dbgs() << "\n!! existing type: " << existType.second << " default type: " << defaultUnionType
                                            << "\n";);

                    auto merged = false;
                    currentType = findBaseType(existType.second, currentType, merged, defaultUnionType);
                }

                LLVM_DEBUG(llvm::dbgs() << "\n!! result type: " << currentType << "\n";);
                typeParamsWithArgs[name].second = currentType;
            }
        }
        else
        {
            // TODO: uncomment this line and find out what is the bug (+one more line)
            auto typeParam = std::make_shared<ts::TypeParameterDOM>(name.str());
            typeParamsWithArgs.insert({name, std::make_pair(typeParam, srcType)});
        }

        LLVM_DEBUG(llvm::dbgs() << "\n!! infered type for '" << name << "' = [" << typeParamsWithArgs[name].second << "]\n";);

        return ExtendsResult::True;        
    }

    mlir::Type getAttributeType(mlir::Attribute attr)
    {
        if (!attr)
        {
            return mlir_ts::UnknownType::get(context);
        }

        if (attr.isa<mlir::StringAttr>())
        {
            return mlir_ts::StringType::get(context);
        }

        if (attr.isa<mlir::FloatAttr>())
        {
            return mlir_ts::NumberType::get(context);
        }

        if (attr.isa<mlir::IntegerAttr>())
        {
            return mlir_ts::NumberType::get(context);
        }

        if (auto typedAttr = attr.dyn_cast<mlir::TypedAttr>())
        {
            return typedAttr.getType();
        }

        llvm_unreachable("not implemented");
    }

    mlir::Type getFieldNames(mlir::Type srcType)
    { 
        llvm::SmallVector<mlir_ts::FieldInfo> destTupleFields;
        if (mlir::failed(getFields(srcType, destTupleFields)))
        {
            return mlir::Type();
        }

        SmallVector<mlir::Type> literalTypes;
        for (auto field : destTupleFields)
        {
            auto litType = field.id && field.id.isa<mlir::StringAttr>() 
                ? mlir_ts::LiteralType::get(field.id, mlir_ts::StringType::get(context))
                : getAttributeType(field.id);
            literalTypes.push_back(litType);
        }

        if (literalTypes.size() == 1)
        {
            return literalTypes.front();
        }

        if (literalTypes.size() == 0)
        {
            return mlir_ts::NeverType::get(context);
        }

        return getUnionType(literalTypes);    
    }

    mlir::Type getFieldTypeByIndexType(mlir::Type srcType, mlir::Type index)
    { 
        llvm::SmallVector<mlir_ts::FieldInfo> destTupleFields;
        if (mlir::failed(getFields(srcType, destTupleFields)))
        {
            return mlir::Type();
        }

        for (auto field : destTupleFields)
        {
            auto litType = field.id && field.id.isa<mlir::StringAttr>() 
                ? mlir_ts::LiteralType::get(field.id, mlir_ts::StringType::get(context))
                : getAttributeType(field.id);
            if (litType == index)
            {
                return field.type;
            }
        }

        return mlir_ts::NeverType::get(context);    
    }    

    mlir::LogicalResult getFields(mlir::Type srcType, llvm::SmallVector<mlir_ts::FieldInfo> &destTupleFields, bool noError = false)
    {       
        if (!srcType)
        {
            return mlir::failure();
        }

        if (auto constTupleType = srcType.dyn_cast<mlir_ts::ConstTupleType>())
        {
            for (auto &fieldInfo : constTupleType.getFields())
            {
                destTupleFields.push_back(fieldInfo);
            }

            return mlir::success();
        }          
        else if (auto tupleType = srcType.dyn_cast<mlir_ts::TupleType>())
        {
            for (auto &fieldInfo : tupleType.getFields())
            {
                destTupleFields.push_back(fieldInfo);
            }

            return mlir::success();
        }         
        else if (auto srcInterfaceType = srcType.dyn_cast<mlir_ts::InterfaceType>())
        {
            if (getInterfaceInfoByFullName)
            {
                if (auto srcInterfaceInfo = getInterfaceInfoByFullName(srcInterfaceType.getName().getValue()))
                {
                    if (mlir::succeeded(srcInterfaceInfo->getTupleTypeFields(destTupleFields, context)))
                    {
                        return mlir::success();
                    }
                }
            }

            return mlir::failure();
        } 
        else if (auto srcClassType = srcType.dyn_cast<mlir_ts::ClassType>())
        {
            for (auto &fieldInfo : srcClassType.getStorageType().cast<mlir_ts::ClassStorageType>().getFields())
            {
                destTupleFields.push_back(fieldInfo);
            }       
            
            return mlir::success();         
        }         
        else if (auto srcClassStorageType = srcType.dyn_cast<mlir_ts::ClassStorageType>())
        {
            for (auto &fieldInfo : srcClassStorageType.getFields())
            {
                destTupleFields.push_back(fieldInfo);
            }       
            
            return mlir::success();            
        }
        else if (srcType.dyn_cast<mlir_ts::ArrayType>() || srcType.dyn_cast<mlir_ts::ConstArrayType>() || srcType.dyn_cast<mlir_ts::StringType>())
        {
            // TODO: do not break the order as it is used in Debug info
            destTupleFields.push_back({ mlir::Attribute(), mlir_ts::NumberType::get(context), false });
            destTupleFields.push_back({ MLIRHelper::TupleFieldName(LENGTH_FIELD_NAME, context), mlir_ts::StringType::get(context), false });
            //destTupleFields.push_back({ MLIRHelper::TupleFieldName(INDEX_ACCESS_FIELD_NAME, context), mlir_ts::NumberType::get(context), false });
            return mlir::success();
        }
        else if (auto optType = srcType.dyn_cast<mlir_ts::OptionalType>())
        {
            // TODO: do not break the order as it is used in Debug info
            destTupleFields.push_back({ MLIRHelper::TupleFieldName("value", context), optType.getElementType(), false });
            destTupleFields.push_back({ MLIRHelper::TupleFieldName("hasValue", context), mlir_ts::BooleanType::get(context), false });
            return mlir::success();
        }

        if (noError)
        {
            return mlir::failure();
        }

        LLVM_DEBUG(llvm::dbgs() << "!! getFields is not implemented for type [" << srcType << "]\n";);
        llvm_unreachable("not implemented");
    }

    mlir::LogicalResult getFieldTypes(mlir::Type srcType, llvm::SmallVector<mlir::Type> &destTupleTypes)
    {  
        llvm::SmallVector<mlir_ts::FieldInfo> destTupleFields;
        if (mlir::failed(getFields(srcType, destTupleFields)))
        {
            return mlir::failure();
        }

        for (auto fieldInfo : destTupleFields)
        {
            destTupleTypes.push_back(fieldInfo.type);
        }

        return mlir::success();
    }

    mlir::Type getFieldTypeByFieldName(mlir::Type srcType, mlir::Attribute fieldName)
    {
        LLVM_DEBUG(llvm::dbgs() << "!! get type of field '" << fieldName << "' of '" << srcType << "'\n";);

        if (auto constTupleType = srcType.dyn_cast<mlir_ts::ConstTupleType>())
        {
            auto index = constTupleType.getIndex(fieldName);
            if (index < 0)
            {
                return mlir::Type();
            }

            return constTupleType.getFieldInfo(index).type;
        }          
        
        if (auto tupleType = srcType.dyn_cast<mlir_ts::TupleType>())
        {
            auto index = tupleType.getIndex(fieldName);
            if (index < 0)
            {
                return mlir::Type();
            }

            return tupleType.getFieldInfo(index).type;
        }  

        if (auto srcInterfaceType = srcType.dyn_cast<mlir_ts::InterfaceType>())
        {
            if (auto srcInterfaceInfo = getInterfaceInfoByFullName(srcInterfaceType.getName().getValue()))
            {
                int totalOffset = 0;
                auto fieldInfo = srcInterfaceInfo->findField(fieldName, totalOffset);
                if (fieldInfo)
                {
                    return fieldInfo->type;
                }

                if (auto strName = fieldName.dyn_cast<mlir::StringAttr>())
                {
                    auto methodInfo = srcInterfaceInfo->findMethod(strName, totalOffset);
                    if (methodInfo)
                    {
                        return methodInfo->funcType;
                    }
                    else
                    {
                        llvm_unreachable("not implemented");
                    }
                }
            }

            return mlir::Type();
        }

        if (auto srcClassType = srcType.dyn_cast<mlir_ts::ClassType>())
        {
            if (auto srcClassInfo = getInterfaceInfoByFullName(srcClassType.getName().getValue()))
            {
                int totalOffset = 0;
                auto fieldInfo = srcClassInfo->findField(fieldName, totalOffset);
                if (fieldInfo)
                {
                    return fieldInfo->type;
                }

                if (auto strName = fieldName.dyn_cast<mlir::StringAttr>())
                {
                    auto methodInfo = srcClassInfo->findMethod(strName, totalOffset);
                    if (methodInfo)
                    {
                        return methodInfo->funcType;
                    }
                    else
                    {
                        llvm_unreachable("not implemented");
                    }
                }
            }

            return mlir::Type();
        }

        // TODO: read fields info from class Array
        // TODO: sync it with getFields
        if (auto arrayType = srcType.dyn_cast<mlir_ts::ArrayType>())
        {
            if (fieldName == MLIRHelper::TupleFieldName(LENGTH_FIELD_NAME, context))
            {
                return mlir_ts::NumberType::get(context);
            }

            if (fieldName == MLIRHelper::TupleFieldName(INDEX_ACCESS_FIELD_NAME, context))
            {
                return  getIndexSignatureType(arrayType.getElementType());
            }

            llvm_unreachable("not implemented");
        }        

        // TODO: read fields info from class Array
        if (auto constArrayType = srcType.dyn_cast<mlir_ts::ConstArrayType>())
        {
            if (fieldName == MLIRHelper::TupleFieldName(LENGTH_FIELD_NAME, context))
            {
                return mlir_ts::NumberType::get(context);
            }

            if (fieldName == MLIRHelper::TupleFieldName(INDEX_ACCESS_FIELD_NAME, context))
            {
                return  getIndexSignatureType(constArrayType.getElementType());
            }

            llvm_unreachable("not implemented");
        }        

        // TODO: read data from String class
        if (auto stringType = srcType.dyn_cast<mlir_ts::StringType>())
        {
            if (fieldName == MLIRHelper::TupleFieldName(LENGTH_FIELD_NAME, context))
            {
                return mlir_ts::NumberType::get(context);
            }

            if (fieldName == MLIRHelper::TupleFieldName(INDEX_ACCESS_FIELD_NAME, context))
            {
                return  getIndexSignatureType(mlir_ts::CharType::get(context));
            }

            llvm_unreachable("not implemented");
        }        

        if (auto unionType = srcType.dyn_cast<mlir_ts::UnionType>())
        {
            llvm::SmallVector<mlir::Type> types;
            for (auto &item : unionType.getTypes())
            {
                auto fieldType = getFieldTypeByFieldName(item, fieldName);
                types.push_back(fieldType);
            }

            return mlir_ts::UnionType::get(context, types);
        }        

        if (auto namedGenericType = srcType.dyn_cast<mlir_ts::NamedGenericType>())
        {
            auto typedAttr = fieldName.dyn_cast<mlir::TypedAttr>();
            // TODO: make common function
            auto fieldNameLiteralType = mlir_ts::LiteralType::get(fieldName, typedAttr && !isNoneType(typedAttr.getType()) ? typedAttr.getType() : mlir_ts::StringType::get(context));
            return mlir_ts::IndexAccessType::get(srcType, fieldNameLiteralType);
        }

        if (auto anyType = srcType.dyn_cast<mlir_ts::AnyType>())
        {
            return anyType;
        }        

        if (auto unknownType = srcType.dyn_cast<mlir_ts::UnknownType>())
        {
            // TODO: but in index type it should return "any"
            return mlir::Type();
        }          

        llvm_unreachable("not implemented");
    }

    ExtendsResult extendsTypeFuncTypes(mlir::Type srcType, mlir::Type extendType,
        llvm::StringMap<std::pair<ts::TypeParameterDOM::TypePtr,mlir::Type>> &typeParamsWithArgs, int skipSrcParams = 0)
    {
            auto srcParams = getParamsFromFuncRef(srcType);
            auto extParams = getParamsFromFuncRef(extendType);

            //auto srcIsVarArgs = getVarArgFromFuncRef(srcType);
            auto extIsVarArgs = getVarArgFromFuncRef(extendType);

            auto srcReturnType = getReturnTypeFromFuncRef(srcType);
            auto extReturnType = getReturnTypeFromFuncRef(extendType);       

            return extendsTypeFuncTypes(srcParams, extParams, extIsVarArgs, srcReturnType, extReturnType, typeParamsWithArgs, skipSrcParams);    
    }

    ExtendsResult extendsTypeFuncTypes(ArrayRef<mlir::Type> srcParams, ArrayRef<mlir::Type> extParams, bool extIsVarArgs, 
        mlir::Type srcReturnType, mlir::Type extReturnType,
        llvm::StringMap<std::pair<ts::TypeParameterDOM::TypePtr,mlir::Type>> &typeParamsWithArgs, int skipSrcParams = 0)
    {
        auto maxParams = std::max(srcParams.size() - skipSrcParams, extParams.size());
        for (auto index = 0; index < maxParams; index++)
        {
            auto srcParamType = (index < srcParams.size() - skipSrcParams) ? srcParams[index + skipSrcParams] : mlir::Type();
            auto extParamType = (index < extParams.size()) ? extParams[index] : extIsVarArgs ? extParams[extParams.size() - 1] : mlir::Type();
            if (!extParamType)
            {
                return ExtendsResult::False;
            }

            auto isIndexAtExtVarArgs = extIsVarArgs && index >= extParams.size() - 1;

            auto useTupleWhenMergeTypes = isIndexAtExtVarArgs;
            if (auto inferType = extParamType.dyn_cast<mlir_ts::InferType>())
            {
                useTupleWhenMergeTypes = true;
                if (isIndexAtExtVarArgs && !srcParamType)
                {
                    // default empty tuple
                    SmallVector<mlir_ts::FieldInfo> fieldInfos;
                    srcParamType = getTupleType(fieldInfos);    
                }
            }

            if (isIndexAtExtVarArgs)
            {
                if (auto arrayType = extParamType.dyn_cast<mlir_ts::ArrayType>())
                {
                    extParamType = arrayType.getElementType();
                }
            }

            auto extendsResult = extendsType(srcParamType, extParamType, typeParamsWithArgs, useTupleWhenMergeTypes);
            if (extendsResult != ExtendsResult::True)
            {
                return extendsResult;
            }
        }      

        // compare return types
        return extendsType(srcReturnType, extReturnType, typeParamsWithArgs);

    }

    ExtendsResult extendsType(mlir::Type srcType, mlir::Type extendType, llvm::StringMap<std::pair<ts::TypeParameterDOM::TypePtr,mlir::Type>> &typeParamsWithArgs, bool useTupleWhenMergeTypes = false)
    {
        LLVM_DEBUG(llvm::dbgs() << "\n!! is extending type: [ " << srcType << " ] extend type: [ " << extendType
                                << " ]\n";);        

        if (!extendType)
        {
            return ExtendsResult::False;
        }

        if (srcType == extendType)
        {
            return ExtendsResult::True;
        }

        if (auto neverType = extendType.dyn_cast<mlir_ts::NeverType>())
        {
            return ExtendsResult::False;
        }        

        if (auto unknownType = extendType.dyn_cast<mlir_ts::UnknownType>())
        {
            return ExtendsResult::True;
        }

        if (auto anyType = extendType.dyn_cast<mlir_ts::AnyType>())
        {
            return ExtendsResult::True;
        }

        if (auto anyType = srcType.dyn_cast_or_null<mlir_ts::AnyType>())
        {
            SmallVector<mlir_ts::InferType> inferTypes;
            getAllInferTypes(extendType, inferTypes);
            for (auto inferType : inferTypes)
            {
                appendInferTypeToContext(mlir_ts::UnknownType::get(context), inferType, typeParamsWithArgs, useTupleWhenMergeTypes);
            }

            // TODO: add all infer types in extends to "unknown"
            return ExtendsResult::Any;
        }        

        if (auto neverType = srcType.dyn_cast_or_null<mlir_ts::NeverType>())
        {
            SmallVector<mlir_ts::InferType> inferTypes;
            getAllInferTypes(extendType, inferTypes);
            for (auto inferType : inferTypes)
            {
                appendInferTypeToContext(mlir_ts::NeverType::get(context), inferType, typeParamsWithArgs, useTupleWhenMergeTypes);
            }

            // TODO: add all infer types in extends to "never"
            return ExtendsResult::Never;
        }        

        // to support infer types
        if (auto inferType = extendType.dyn_cast<mlir_ts::InferType>())
        {
            return appendInferTypeToContext(srcType, inferType, typeParamsWithArgs, useTupleWhenMergeTypes);
        }

        if (!srcType)
        {
            return ExtendsResult::False;
        }        

        if (auto unionType = extendType.dyn_cast<mlir_ts::UnionType>())
        {
            auto falseResult = ExtendsResult::False;
            auto pred = [&](auto &item) { 
                auto unionExtResult = extendsType(srcType, item, typeParamsWithArgs);
                if (unionExtResult == ExtendsResult::Never)
                {
                    falseResult = unionExtResult;
                }                

                return isTrue(unionExtResult); 
            };
            auto types = unionType.getTypes();
            return std::find_if(types.begin(), types.end(), pred) != types.end() ? ExtendsResult::True : falseResult;
        }

        if (auto tupleType = extendType.dyn_cast<mlir_ts::TupleType>())
        {
            auto falseResult = ExtendsResult::False;
            auto pred = [&](auto &item) { 
                if (item.id)
                {
                    auto fieldType = getFieldTypeByFieldName(srcType, item.id);
                    auto fieldExtResult = extendsType(fieldType, item.type, typeParamsWithArgs);
                    if (fieldExtResult == ExtendsResult::Never)
                    {
                        falseResult = fieldExtResult;
                    }

                    return isTrue(fieldExtResult); 
                }
                else
                {
                    // TODO: get it by function
                    llvm_unreachable("not implemented");
                }
            };
            return std::all_of(tupleType.getFields().begin(), tupleType.getFields().end(), pred) ? ExtendsResult::True : falseResult;
        }

        if (auto constTupleType = extendType.dyn_cast<mlir_ts::ConstTupleType>())
        {
            auto falseResult = ExtendsResult::False;
            auto pred = [&](auto &item) { 
                if (item.id)
                {
                    auto fieldType = getFieldTypeByFieldName(srcType, item.id);
                    auto fieldExtResult = extendsType(fieldType, item.type, typeParamsWithArgs);
                    if (fieldExtResult == ExtendsResult::Never)
                    {
                        falseResult = fieldExtResult;
                    }

                    return isTrue(fieldExtResult); 
                }
                else
                {
                    // TODO: get it by function
                    llvm_unreachable("not implemented");
                }
            };
            return std::all_of(constTupleType.getFields().begin(), constTupleType.getFields().end(), pred) ? ExtendsResult::True : falseResult;
        }

        if (auto literalType = srcType.dyn_cast<mlir_ts::LiteralType>())
        {
            if (auto litExt = extendType.dyn_cast<mlir_ts::LiteralType>())
            {
                return ExtendsResult::False;
            }

            return extendsType(literalType.getElementType(), extendType, typeParamsWithArgs);
        }

        if (auto srcArray = srcType.dyn_cast<mlir_ts::ArrayType>())
        {
            if (auto extArray = extendType.dyn_cast<mlir_ts::ArrayType>())
            {
                return extendsType(srcArray.getElementType(), extArray.getElementType(), typeParamsWithArgs);
            }
        }

        if (auto srcArray = srcType.dyn_cast<mlir_ts::ConstArrayType>())
        {
            if (auto extArray = extendType.dyn_cast<mlir_ts::ArrayType>())
            {
                return extendsType(srcArray.getElementType(), extArray.getElementType(), typeParamsWithArgs);
            }
        }

        // Special case when we have string type (widen from Literal Type)
        if (auto literalType = extendType.dyn_cast<mlir_ts::LiteralType>())
        {
            return extendsType(srcType, literalType.getElementType(), typeParamsWithArgs);
        }        

        if (auto unionType = srcType.dyn_cast<mlir_ts::UnionType>())
        {
            auto falseResult = ExtendsResult::False;
            auto pred = [&](auto &item) { 
                auto unionExtResult = extendsType(item, extendType, typeParamsWithArgs);
                if (unionExtResult == ExtendsResult::Never)
                {
                    falseResult = unionExtResult;
                }                

                return isTrue(unionExtResult); 
            };
            auto types = unionType.getTypes();
            return std::find_if(types.begin(), types.end(), pred) != types.end() ? ExtendsResult::True : falseResult;
        }

        // seems it is generic interface
        if (auto srcInterfaceType = srcType.dyn_cast<mlir_ts::InterfaceType>())
        {
            if (auto extInterfaceType = extendType.dyn_cast<mlir_ts::InterfaceType>())
            {
                if (auto srcInterfaceInfo = getInterfaceInfoByFullName(srcInterfaceType.getName().getValue()))
                {
                    if (auto extInterfaceInfo = getInterfaceInfoByFullName(extInterfaceType.getName().getValue()))
                    {
                        if (srcInterfaceInfo->originInterfaceType 
                            && extInterfaceInfo->originInterfaceType 
                            && srcInterfaceInfo->originInterfaceType == extInterfaceInfo->originInterfaceType)
                        {
                            LLVM_DEBUG(llvm::dbgs() << "\n!! origin type for interfaces '" << srcInterfaceInfo->originInterfaceType << "' & '" << extInterfaceInfo->originInterfaceType << "'\n";);

                            if (auto genericInterface = getGenericInterfaceInfoByFullName(srcInterfaceInfo->originInterfaceType.getName().getValue()))
                            {
                                for (auto &typeParam : genericInterface->typeParams)
                                {
                                    auto name = typeParam->getName();
                                    auto srcFound = srcInterfaceInfo->typeParamsWithArgs.find(name);
                                    auto extFound = extInterfaceInfo->typeParamsWithArgs.find(name);
                                    if (srcFound != srcInterfaceInfo->typeParamsWithArgs.end() && 
                                        extFound != extInterfaceInfo->typeParamsWithArgs.end())
                                    {
                                        auto srcType = srcFound->getValue().second;
                                        auto extType = extFound->getValue().second;

                                        return extendsType(srcType, extType, typeParamsWithArgs);
                                    }
                                    else
                                    {
                                        return ExtendsResult::False;
                                    }
                                }
                            }

                            // default behavior - false, because something is different
                            return ExtendsResult::False;
                        }
                    }
                }
            }
        }

        if (auto srcClassType = srcType.dyn_cast<mlir_ts::ClassType>())
        {
            if (auto extClassType = extendType.dyn_cast<mlir_ts::ClassType>())
            {
                if (auto srcClassInfo = getClassInfoByFullName(srcClassType.getName().getValue()))
                {
                    if (auto extClassInfo = getClassInfoByFullName(extClassType.getName().getValue()))
                    {
                        if (srcClassInfo->originClassType 
                            && extClassInfo->originClassType
                            && srcClassInfo->originClassType == extClassInfo->originClassType)
                        {
                            LLVM_DEBUG(llvm::dbgs() << "\n!! origin type for class '" << srcClassInfo->originClassType << "' & '" << extClassInfo->originClassType << "'\n";);

                            if (auto genericClass = getGenericClassInfoByFullName(srcClassInfo->originClassType.getName().getValue()))
                            {
                                for (auto &typeParam : genericClass->typeParams)
                                {
                                    auto name = typeParam->getName();
                                    auto srcFound = srcClassInfo->typeParamsWithArgs.find(name);
                                    auto extFound = extClassInfo->typeParamsWithArgs.find(name);
                                    if (srcFound != srcClassInfo->typeParamsWithArgs.end() && 
                                        extFound != extClassInfo->typeParamsWithArgs.end())
                                    {
                                        auto srcType = srcFound->getValue().second;
                                        auto extType = extFound->getValue().second;

                                        return extendsType(srcType, extType, typeParamsWithArgs);
                                    }
                                    else
                                    {
                                        return ExtendsResult::False;
                                    }
                                }
                            }

                            // default behavior - false, because something is different
                            return ExtendsResult::False;
                        }
                    }
                }
            }
        }

        if (isAnyFunctionType(srcType) && isAnyFunctionType(extendType))
        {
            return extendsTypeFuncTypes(srcType, extendType, typeParamsWithArgs);
        }

        if (auto constructType = extendType.dyn_cast<mlir_ts::ConstructFunctionType>())
        {
            if (auto srcClassType = srcType.dyn_cast<mlir_ts::ClassType>())
            {
                if (auto srcClassInfo = getClassInfoByFullName(srcClassType.getName().getValue()))
                {
                    // we have class
                    if (!srcClassInfo->getHasConstructor())
                    {
                        return ExtendsResult::False;
                    }

                    // find constructor type
                    auto constrMethod = srcClassInfo->findMethod(CONSTRUCTOR_NAME);
                    auto constrWithRetType = mlir_ts::FunctionType::get(
                        constrMethod->funcType.getContext(), 
                        constrMethod->funcType.getInputs(), 
                        {srcClassInfo->classType}, 
                        constrMethod->funcType.isVarArg());
                    return extendsTypeFuncTypes(constrWithRetType, extendType, typeParamsWithArgs, 1/*because of this param*/);
                }
            }

            return ExtendsResult::False;
        }

        if (auto ifaceType = srcType.dyn_cast<mlir_ts::InterfaceType>())
        {
            auto interfaceInfo = getInterfaceInfoByFullName(ifaceType.getName().getValue());
            assert(interfaceInfo);
            auto falseResult = ExtendsResult::False;
            for (auto extend : interfaceInfo->extends)
            {
                auto extResult = extendsType(extend.second->interfaceType, extendType, typeParamsWithArgs);
                if (isTrue(extResult))
                {
                    return extResult;
                }

                if (extResult == ExtendsResult::Never)
                {
                    falseResult = ExtendsResult::Never;
                }                
            }

            return falseResult;
        }

        if (auto classType = srcType.dyn_cast<mlir_ts::ClassType>())
        {
            auto classInfo = getClassInfoByFullName(classType.getName().getValue());
            assert(classInfo);
            auto falseResult = ExtendsResult::False;
            for (auto extend : classInfo->baseClasses)
            {
                auto extResult = extendsType(extend->classType, extendType, typeParamsWithArgs);
                if (extResult == ExtendsResult::True)
                {
                    return ExtendsResult::True;
                }

                if (extResult == ExtendsResult::Never)
                {
                    falseResult = ExtendsResult::Never;
                }                  
            }

            if (auto ifaceType = extendType.dyn_cast<mlir_ts::InterfaceType>())
            {
                for (auto extend : classInfo->implements)
                {
                    auto extResult = extendsType(extend.interface->interfaceType, extendType, typeParamsWithArgs);
                    if (isTrue(extResult))
                    {
                        return extResult;
                    }

                    if (extResult == ExtendsResult::Never)
                    {
                        falseResult = ExtendsResult::Never;
                    }                
                }                
            }

            return falseResult;
        }

        if (auto objType = extendType.dyn_cast<mlir_ts::ObjectType>())
        {
            if (objType.getStorageType().isa<mlir_ts::AnyType>())
            {
                return (srcType.isa<mlir_ts::TupleType>() || srcType.isa<mlir_ts::ConstTupleType>() || srcType.isa<mlir_ts::ObjectType>()) 
                    ? ExtendsResult::True : ExtendsResult::False;
            }
        }        

        // TODO: finish Function Types, etc
        LLVM_DEBUG(llvm::dbgs() << "\n!! extendsType [FLASE]\n";);
        return ExtendsResult::False;
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
            return mlir::success();
        }

        if (auto literalType = type.dyn_cast<mlir_ts::LiteralType>())
        {
            unionContext.literalTypes.insert(literalType);
            return mlir::success();
        }

        if (auto optionalType = type.dyn_cast<mlir_ts::OptionalType>())
        {
            if (unionContext.isUndefined)
            {
                unionContext.types.insert(optionalType.getElementType());
            }
            else
            {
                unionContext.types.insert(optionalType);
            }

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

    mlir::Type getUnionType(mlir::Type type1, mlir::Type type2, bool mergeLiterals = true, bool mergeTypes = true)
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
        return getUnionTypeWithMerge(types, mergeLiterals, mergeTypes);
    }

    // TODO: review all union merge logic
    mlir::Type getUnionTypeMergeTypes(UnionTypeProcessContext &unionContext, bool mergeLiterals = true, bool mergeTypes = true)
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

        if (typesAll.size() == 1)
        {
            auto resType = typesAll.front();
            if (unionContext.isUndefined)
            {
                return mlir_ts::OptionalType::get(resType);
            }     

            return resType;       
        }

        // merge types
        auto doNotMergeLiterals = !mergeLiterals && isAllLiteralTypes;
        if (mergeTypes && !doNotMergeLiterals)
        {
            mlir::SmallVector<mlir::Type> mergedTypesAll;
            this->mergeTypes(typesAll, mergedTypesAll);

            mlir::Type retType = mergedTypesAll.size() == 1 ? mergedTypesAll.front() : getUnionType(mergedTypesAll);
            if (unionContext.isUndefined)
            {
                return mlir_ts::OptionalType::get(retType);
            }

            return retType;
        }

        mlir::Type retType = typesAll.size() == 1 ? typesAll.front() : getUnionType(typesAll);
        if (unionContext.isUndefined)
        {
            return mlir_ts::OptionalType::get(retType);
        }

        return retType;
    }

    void detectTypeForGroupOfTypes(mlir::ArrayRef<mlir::Type> types, UnionTypeProcessContext &unionContext)
    {
        // check if type is nullable or undefinable
        for (auto type : types)
        {
            if (type.isa<mlir_ts::UndefinedType>() || type.isa<mlir_ts::OptionalType>())
            {
                unionContext.isUndefined = true;
                continue;
            }

            if (type.isa<mlir_ts::NullType>())
            {
                unionContext.isNullable = true;
                continue;
            }            

            if (type.isa<mlir_ts::AnyType>())
            {
                unionContext.isAny = true;
                continue;
            }

            if (type.isa<mlir_ts::NeverType>())
            {
                continue;
            }

            if (auto unionType = type.dyn_cast<mlir_ts::UnionType>())
            {
                detectTypeForGroupOfTypes(unionType.getTypes(), unionContext);
            }
        }
    }

    mlir::Type getUnionTypeWithMerge(mlir::ArrayRef<mlir::Type> types, bool mergeLiterals = true, bool mergeTypes = true)
    {
        UnionTypeProcessContext unionContext = {};

        detectTypeForGroupOfTypes(types, unionContext);

        // default wide types
        if (unionContext.isAny)
        {
            return mlir_ts::AnyType::get(context);
        }

        for (auto type : types)
        {
            if (!type)
            {
                llvm_unreachable("wrong type");
            }

            processUnionTypeItem(type, unionContext);
        }

        return getUnionTypeMergeTypes(unionContext, mergeLiterals, mergeTypes);
    }

    mlir::Type getUnionType(mlir::SmallVector<mlir::Type> &types)
    {
        if (types.size() == 0)
        {
            // TODO:? should it be empty tuple or never type?
            //return mlir_ts::NeverType::get(context);
            return mlir_ts::TupleType::get(context, {});
        }

        if (types.size() == 1)
        {
            return types.front();
        }

        return normalizeUnionType(types);
    }

    mlir::Type normalizeUnionType(mlir::SmallVector<mlir::Type> &types)
    {
        mlir::SmallPtrSet<mlir::Type, 2> normalizedTypes;
        auto isUndefined = false;
        for (auto type : types)
        {
            if (type.isa<mlir_ts::UndefinedType>())
            {
                isUndefined = true; 
                continue;
            }

            if (auto optType = dyn_cast<mlir_ts::OptionalType>(type))
            {
                isUndefined = true; 
                normalizedTypes.insert(optType.getElementType());
                continue;
            }

            normalizedTypes.insert(type);
        }

        if (normalizedTypes.size() == 0)
        {
            return isUndefined ? mlir::Type(mlir_ts::UndefinedType::get(context)) : mlir::Type(mlir_ts::NeverType::get(context));
        }

        if (normalizedTypes.size() == 1)
        {
            return isUndefined ? mlir_ts::OptionalType::get(*normalizedTypes.begin()) : *normalizedTypes.begin();
        }

        mlir::SmallVector<mlir::Type> newTypes;
        for (auto type : normalizedTypes)
        {
            newTypes.push_back(type);
        }

        std::sort (newTypes.begin(), newTypes.end(), [](auto i, auto j) { return (i.getAsOpaquePointer() < j.getAsOpaquePointer()); });

        return isUndefined ? mlir::Type(mlir_ts::OptionalType::get(mlir_ts::UnionType::get(context, newTypes))) : mlir::Type(mlir_ts::UnionType::get(context, newTypes));
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
        MLIRTypeIteratorLogic iter(
            getClassInfoByFullName, getGenericClassInfoByFullName, 
            getInterfaceInfoByFullName, getGenericInterfaceInfoByFullName
        );
        return iter.some(type, [](mlir::Type type) { return type && type.isa<mlir_ts::NamedGenericType>(); });
    }

    bool hasInferType(mlir::Type type)
    {
        MLIRTypeIteratorLogic iter(
            getClassInfoByFullName, getGenericClassInfoByFullName, 
            getInterfaceInfoByFullName, getGenericInterfaceInfoByFullName
        );
        return iter.some(type, [](mlir::Type type) { return type && type.isa<mlir_ts::InferType>(); });
    }    

    bool getAllInferTypes(mlir::Type type, SmallVector<mlir_ts::InferType> &inferTypes)
    {
        MLIRTypeIteratorLogic iter(
            getClassInfoByFullName, getGenericClassInfoByFullName, 
            getInterfaceInfoByFullName, getGenericInterfaceInfoByFullName
        );
        return iter.every(type, [&](mlir::Type type) { 
            if (auto inferType = type.dyn_cast<mlir_ts::InferType>())
            {
                inferTypes.push_back(inferType);
            }

            return !!type; 
        });
    }        
    
    void mergeTypes(mlir::ArrayRef<mlir::Type> types, mlir::SmallVector<mlir::Type> &mergedTypes)
    {
        for (auto typeItem : types)
        {
            if (mergedTypes.size() == 0)
            {
                mergedTypes.push_back(typeItem);
                continue;
            }

            auto found = false;
            for (auto [index, mergedType] : enumerate(mergedTypes))
            {
                auto merged = false;
                auto resultType = mergeType(mergedType, typeItem, merged);
                if (merged)
                {
                    mergedTypes[index] = resultType;
                    found = true;
                    break;
                }
            }

            if (!found)
            {
                mergedTypes.push_back(typeItem);
            }
        }
    }

    mlir::Type arrayMergeType(mlir::Type existType, mlir::Type currentType, bool& merged)
    {
        LLVM_DEBUG(llvm::dbgs() << "\n!! merging existing type: " << existType << " with " << currentType << "\n";);

        if (existType == currentType)
        {
            merged = true;
            return existType;
        }

        // in case of array
        auto currentTypeArray = currentType.dyn_cast_or_null<mlir_ts::ArrayType>();
        auto existTypeArray = existType.dyn_cast_or_null<mlir_ts::ArrayType>();
        if (currentTypeArray && existTypeArray)
        {
            auto arrayElementMerged = mergeType(existTypeArray.getElementType(), currentTypeArray.getElementType(), merged);   
            return mlir_ts::ArrayType::get(arrayElementMerged);
        }

        return mergeType(existType, currentType, merged);
    }

    mlir::Type tupleMergeType(mlir_ts::TupleType existType, mlir_ts::TupleType currentType, bool& merged)
    {
        merged = false;
        LLVM_DEBUG(llvm::dbgs() << "\n!! merging existing type: " << existType << " with " << currentType << "\n";);

        if (existType == currentType)
        {
            merged = true;
            return existType;
        }

        auto existingFields = existType.getFields();
        auto currentFields = currentType.getFields();

        if (existingFields.size() != currentFields.size())
        {
            return mlir::Type();
        }

        llvm::SmallVector<mlir_ts::FieldInfo> resultFields;
        auto existingIt = existingFields.begin();
        auto currentIt = currentFields.begin();
        for (; existingIt != existingFields.end() && currentIt != currentFields.end(); ++existingIt, ++currentIt)
        {
            if (existingIt->id != currentIt->id)
            {
                return mlir::Type();
            }

            // try to merge types of tuple
            auto merged = false;
            auto mergedType = mergeType(existingIt->type, currentIt->type, merged);
            if (mergedType)
            {
                resultFields.push_back({ existingIt->id, mergedType, false });
            }
        }

        merged = true;
        return mlir_ts::TupleType::get(context, resultFields);
    }

    mlir::Type mergeType(mlir::Type existType, mlir::Type currentType, bool& merged)
    {
        merged = false;
        LLVM_DEBUG(llvm::dbgs() << "\n!! merging existing \n\ttype: \t" << existType << "\n\twith \t" << currentType << "\n";);

        if (existType == currentType)
        {
            merged = true;
            return existType;
        }

        if (canCastFromTo(currentType, existType))
        {
            merged = true;
            return existType;
        }

        if (canCastFromTo(existType, currentType))
        {
            merged = true;
            return currentType;
        }

        // in case of merging integer types with sign/no sign
        auto mergedInts = false;
        auto resNewIntType = mergeIntTypes(existType, currentType, mergedInts);
        if (mergedInts)
        {
            return resNewIntType;
        }
        
        // wide type - remove const & literal
        auto resType = wideStorageType(currentType);

        // check if can merge tuple types
        if (auto existingTupleType = existType.dyn_cast<mlir_ts::TupleType>())
        {
            if (auto currentTupleType = currentType.dyn_cast<mlir_ts::TupleType>())
            {
                auto tupleMerged = false;
                auto mergedTupleType = tupleMergeType(existingTupleType, currentTupleType, tupleMerged);
                if (tupleMerged)
                {
                    merged = true;
                    return mergedTupleType;
                }
            }
        }
        
        auto found = false;
        resType = findBaseType(existType, resType, found);
        if (!found)
        {
            mlir::Type defaultUnionType;
            mlir::SmallVector<mlir::Type> types;
            types.push_back(existType);
            types.push_back(currentType);

            if (existType.isa<mlir_ts::UnionType>())
            {
                defaultUnionType = getUnionTypeWithMerge(types);
            }
            else
            {
                defaultUnionType = getUnionType(types);
                LLVM_DEBUG(llvm::dbgs() << "\n!! default type: " << defaultUnionType
                                    << "\n";);
            }

            return defaultUnionType;
        }

        LLVM_DEBUG(llvm::dbgs() << "\n!! result type: " << resType << "\n";);

        merged = true;
        return resType;
    }

    template <typename T, typename F>
    void printFuncType(T &out, F t)
    {
        out << "(";
        auto first = true;
        auto index = 0;
        auto size = t.getInputs().size();
        auto isVar = t.getIsVarArg();
        for (auto subType : t.getInputs())
        {
            if (!first)
            {
                out << ", ";
            }

            if (isVar && size == 1)
            {
                out << "...";
            }

            out << "p" << index << ": ";

            printType(out, subType);
            first = false;
            index ++;
            size --;
        }
        out << ") => ";

        if (t.getNumResults() == 0)
        {
            out << "void";
        }
        else if (t.getNumResults() == 1)
        {
            printType(out, t.getResults().front());
        }
        else
        {
            out << "[";
            auto first = true;
            for (auto subType : t.getResults())
            {
                if (!first)
                {
                    out << ", ";
                }

                printType(out, subType);
                first = false;
            }

            out << "]";
        }
    }

    template <typename T, typename TPL>
    void printFields(T &out, TPL t)
    {
        auto first = true;
        for (auto field : t.getFields())
        {
            if (!first)
            {
                out << ", ";
            }

            if (field.id)
            {
                printAttribute(out, field.id);
                out << ":";
            }

            printType(out, field.type);
            first = false;
        }
    }    

    template <typename T, typename TPL>
    void printTupleType(T &out, TPL t)
    {
        out << "[";
        printFields(out, t);
        out << "]";        
    }

    template <typename T, typename TPL>
    void printObjectType(T &out, TPL t)
    {
        out << "{";
        printFields(out, t);
        out << "}";        
    }

    template <typename T, typename U>
    void printUnionType(T &out, U t, const char *S)
    {
        auto first = true;
        for (auto subType : t.getTypes())
        {
            if (!first)
            {
                out << S;
            }

            printType(out, subType);
            first = false;
        }        
    }
    
    template <typename T>
    void printAttribute(T &out, mlir::Attribute attr)
    {
        llvm::TypeSwitch<mlir::Attribute>(attr)
            .template Case<mlir::StringAttr>([&](auto a) {
                out << a.str().c_str();
            })
            .template Case<mlir::FlatSymbolRefAttr>([&](auto a) {
                out << a.getValue().str().c_str();
            })            
            .template Case<mlir::IntegerAttr>([&](auto a) {
                SmallVector<char> Str;
                a.getValue().toStringUnsigned(Str);
                StringRef strRef(Str.data(), Str.size());
                out << strRef.str().c_str();
            })
            .Default([](mlir::Attribute a) { 
                LLVM_DEBUG(llvm::dbgs() << "\n!! Type print is not implemented for : " << a << "\n";);
                llvm_unreachable("not implemented");
            });
    }

    template <typename T>
    void printType(T &out, mlir::Type type)
    {
        llvm::TypeSwitch<mlir::Type>(type)
            .template Case<mlir_ts::NullType>([&](auto t) {
                out << "null";
            })
            .template Case<mlir_ts::UndefinedType>([&](auto t) {
                out << "undefined";
            })
            .template Case<mlir_ts::ArrayType>([&](auto t) {
                printType(out, t.getElementType());
                out << "[]";
            })
            .template Case<mlir_ts::BoundFunctionType>([&](auto t) {
                printFuncType(out, t);
            })
            .template Case<mlir_ts::BoundRefType>([&](auto t) {
                printType(out, t.getElementType());
            })
            .template Case<mlir_ts::ClassType>([&](auto t) {
                out << t.getName().getValue().str().c_str();
            })
            .template Case<mlir_ts::ClassStorageType>([&](auto t) {
                printTupleType(out, t);                                
            })
            .template Case<mlir_ts::InterfaceType>([&](auto t) {
                out << t.getName().getValue().str().c_str();
            })
            .template Case<mlir_ts::ConstArrayType>([&](auto t) {
                printType(out, t.getElementType());
                out << "[]";
            })
            .template Case<mlir_ts::ConstArrayValueType>([&](auto t) {
                printType(out, t.getElementType());
                out << "[]";
            })
            .template Case<mlir_ts::ConstTupleType>([&](auto t) {
                printTupleType(out, t);
            })
            .template Case<mlir_ts::EnumType>([&](auto t) {
                printType(out, t.getElementType());
            })
            .template Case<mlir_ts::FunctionType>([&](auto t) {
                printFuncType(out, t);
            })
            .template Case<mlir_ts::HybridFunctionType>([&](auto t) {
                printFuncType(out, t);
            })
            .template Case<mlir_ts::ConstructFunctionType>([&](auto t) {
                out << "new ";
                printFuncType(out, t);
            })
            .template Case<mlir_ts::InferType>([&](auto t) {
                out << "infer ";
                printType(out, t.getElementType());
            })
            .template Case<mlir_ts::LiteralType>([&](auto t) {
                printType(out, t.getElementType());
            })
            .template Case<mlir_ts::OptionalType>([&](auto t) {
                printType(out, t.getElementType());
                out << " | undefined";
            })
            .template Case<mlir_ts::RefType>([&](auto t) {
                out << "Reference<";
                printType(out, t.getElementType());
                out << ">";
            })
            .template Case<mlir_ts::TupleType>([&](auto t) {
                printTupleType(out, t);
            })
            .template Case<mlir_ts::UnionType>([&](auto t) {
                printUnionType(out, t, " | ");
            })
            .template Case<mlir_ts::IntersectionType>([&](auto t) {
                printUnionType(out, t, " & ");
            })
            .template Case<mlir_ts::ValueRefType>([&](auto t) {
                printType(out, t.getElementType());
            })
            .template Case<mlir_ts::ConditionalType>([&](auto t) {
                printType(out, t.getCheckType());
                out << "extends";
                printType(out, t.getCheckType());
                out << " ? ";
                printType(out, t.getTrueType());
                out << " : ";
                printType(out, t.getFalseType());
            })
            .template Case<mlir_ts::IndexAccessType>([&](auto t) {
                printType(out, t.getType());
                out << "[";
                printType(out, t.getIndexType());
                out << "]";
            })
            .template Case<mlir_ts::KeyOfType>([&](auto t) {
                out << "keyof ";
                printType(out, t.getElementType());
            })
            .template Case<mlir_ts::MappedType>([&](auto t) {
                out << "[";
                printType(out, t.getElementType());
                out << " of ";
                printType(out, t.getNameType());
                out << " extends ";
                printType(out, t.getConstrainType());
                out << "]";
            })
            .template Case<mlir_ts::TypeReferenceType>([&](auto t) {
                printAttribute(out, t.getName());
                if (t.getTypes().size() > 0)
                {
                    out << "<";
                    auto first = true;
                    for (auto subType : t.getTypes())
                    {
                        if (!first)
                        {
                            out << ", ";
                        }

                        printType(out, subType);
                        first = false;
                    }
                    out << ">";
                }
            })
            .template Case<mlir_ts::TypePredicateType>([&](auto t) {
                printType(out, t.getElementType());
            })
            .template Case<mlir_ts::NamedGenericType>([&](auto t) {
                out << t.getName().getValue().str().c_str();
            })
            .template Case<mlir_ts::ObjectType>([&](auto t) {
                out << "object";
            })
            .template Case<mlir_ts::NeverType>([&](auto) { 
                out << "never";
            })
            .template Case<mlir_ts::UnknownType>([&](auto) {
                out << "unknown";
            })
            .template Case<mlir_ts::AnyType>([&](auto) {
                out << "any";
            })
            .template Case<mlir_ts::NumberType>([&](auto) {
                out << "number";
            })
            .template Case<mlir_ts::StringType>([&](auto) {
                out << "string";
            })
            .template Case<mlir_ts::BooleanType>([&](auto) {
                out << "boolean";
            })
            .template Case<mlir_ts::OpaqueType>([&](auto) {
                out << "Opaque";
            })
            .template Case<mlir_ts::VoidType>([&](auto) {
                out << "void";
            })
            .template Case<mlir_ts::ConstType>([&](auto) {
                out << "const";
            })
            .template Case<mlir::NoneType>([&](auto) {
                out << "void";
            })
            .template Case<mlir::IntegerType>([&](auto) {
                out << "TypeOf<1>";
            })
            .template Case<mlir::FloatType>([&](auto) {
                out << "TypeOf<1.0>";
            })
            .template Case<mlir::IndexType>([&](auto) {
                out << "TypeOf<1>";
            })
            .Default([](mlir::Type t) { 
                LLVM_DEBUG(llvm::dbgs() << "\n!! Type print is not implemented for : " << t << "\n";);
                llvm_unreachable("not implemented");
            });
    }

protected:
    std::function<ClassInfo::TypePtr(StringRef)> getClassInfoByFullName;

    std::function<GenericClassInfo::TypePtr(StringRef)> getGenericClassInfoByFullName;

    std::function<InterfaceInfo::TypePtr(StringRef)> getInterfaceInfoByFullName;

    std::function<GenericInterfaceInfo::TypePtr(StringRef)> getGenericInterfaceInfoByFullName;
};

} // namespace typescript

#undef DEBUG_TYPE

#endif // MLIR_TYPESCRIPT_COMMONGENLOGIC_MLIRTYPEHELPER_H_
