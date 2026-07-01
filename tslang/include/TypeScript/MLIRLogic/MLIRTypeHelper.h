#ifndef MLIR_TYPESCRIPT_COMMONGENLOGIC_MLIRTYPEHELPER_H_
#define MLIR_TYPESCRIPT_COMMONGENLOGIC_MLIRTYPEHELPER_H_

#include "TypeScript/TypeScriptOps.h"
#include "TypeScript/DOM.h"
#include "TypeScript/DataStructs.h"
#include "TypeScript/MLIRLogic/MLIRGenStore.h"
#include "TypeScript/MLIRLogic/MLIRTypeIterator.h"
#include "TypeScript/MLIRLogic/MLIRHelper.h"
#include "TypeScript/MLIRLogic/MLIRPrinter.h"
#include "TypeScript/MLIRLogic/MLIRTypeCore.h"

#include "llvm/Support/Debug.h"
#include "llvm/ADT/APSInt.h"

#include <functional>

#define DEBUG_TYPE "mlir"

namespace mlir_ts = mlir::typescript;

namespace typescript
{

class MLIRTypeHelper
{
    mlir::MLIRContext *context;
    CompileOptions compileOptions;

  public:

    MLIRTypeHelper(
        mlir::MLIRContext *context,
        CompileOptions compileOptions)
        : context(context), 
          compileOptions(compileOptions),           
          getClassInfoByFullName{},
          getGenericClassInfoByFullName{},
          getInterfaceInfoByFullName{},
          getGenericInterfaceInfoByFullName{}
    {
    }

    MLIRTypeHelper(
        mlir::MLIRContext *context, 
        CompileOptions compileOptions,
        std::function<ClassInfo::TypePtr(StringRef)> getClassInfoByFullName,
        std::function<GenericClassInfo::TypePtr(StringRef)> getGenericClassInfoByFullName,
        std::function<InterfaceInfo::TypePtr(StringRef)> getInterfaceInfoByFullName,
        std::function<GenericInterfaceInfo::TypePtr(StringRef)> getGenericInterfaceInfoByFullName) 
        : context(context), 
          compileOptions(compileOptions),
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
            fields.push_back(mlir_ts::FieldInfo{nullptr, type, false, mlir_ts::AccessLevel::Public});
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

    mlir::IntegerAttr getIndexAttrValue(int64_t value)
    {
        return mlir::IntegerAttr::get(getIndexType(), mlir::APInt(64, value));
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
        return type && (type.isIntOrIndexOrFloat() || isa<mlir_ts::NumberType>(type) || isa<mlir_ts::BooleanType>(type) || isa<mlir_ts::CharType>(type) ||
            isa<mlir_ts::TupleType>(type) || isa<mlir_ts::ConstTupleType>(type) || isa<mlir_ts::ConstArrayType>(type));
    }

    bool isNumericType(mlir::Type type)
    {
        return type && (type.isIntOrIndexOrFloat() || isa<mlir_ts::NumberType>(type));
    }

    mlir::Type isBoundReference(mlir::Type elementType, bool &isBound)
    {
#ifdef USE_BOUND_FUNCTION_FOR_OBJECTS
        if (auto funcType = dyn_cast<mlir_ts::FunctionType>(elementType))
        {
            if (funcType.getNumInputs() > 0 &&
                (isa<mlir_ts::OpaqueType>(funcType.getInput(0)) || isa<mlir_ts::ObjectType>(funcType.getInput(0))))
            {
                isBound = true;
                return mlir_ts::BoundFunctionType::get(context, funcType);
            }
        }
#endif

        isBound = false;
        return elementType;
    }

    mlir::Type getElementTypeOrSelf(mlir::Type type)
    {
        return MLIRHelper::getElementTypeOrSelf(type);
    }

    mlir::StringAttr getLabelName(mlir::Type typeIn)
    {
        if (typeIn.isIndex())
        {
            return mlir::StringAttr::get(context, std::string("index"));
        }
        else if (typeIn.isIntOrIndex())
        {
            return mlir::StringAttr::get(
                context, 
                std::string((typeIn.isSignlessInteger() 
                    ? "i" 
                    : typeIn.isSignedInteger() 
                        ? "s" 
                        : "u")) + std::to_string(typeIn.getIntOrFloatBitWidth()));
        }
        else if (typeIn.isIntOrFloat())
        {
            return mlir::StringAttr::get(context, std::string("f") + std::to_string(typeIn.getIntOrFloatBitWidth()));
        }
        else if (isa<mlir_ts::NullType>(typeIn))
        {
            return mlir::StringAttr::get(context, "null");
        }
        else if (isa<mlir_ts::UndefinedType>(typeIn)) 
        {
            return mlir::StringAttr::get(context, "undefined");
        }
        else if (isa<mlir_ts::NumberType>(typeIn))
        {
            return mlir::StringAttr::get(context, "number");
        }
        else if (isa<mlir_ts::TupleType>(typeIn))
        {
            return mlir::StringAttr::get(context, "tuple");
        }
        else if (isa<mlir_ts::ObjectType>(typeIn))
        {
            return mlir::StringAttr::get(context, "object");
        }
        else if (isa<mlir_ts::StringType>(typeIn)) 
        {
            return mlir::StringAttr::get(context, "string");
        }
        else if (isa<mlir_ts::ObjectType>(typeIn))
        {
            return mlir::StringAttr::get(context, "object");
        }
        else if (isa<mlir_ts::ClassType>(typeIn)) 
        {
            return mlir::StringAttr::get(context, "class");
        }
        else if (isa<mlir_ts::InterfaceType>(typeIn))
        {
            return mlir::StringAttr::get(context, "interface");
        }
        else if (isa<mlir_ts::OptionalType>(typeIn))
        {
            return mlir::StringAttr::get(context, "optional");
        }
        else if (isa<mlir_ts::AnyType>(typeIn))
        {
            return mlir::StringAttr::get(context, "any");
        }
        else if (isa<mlir_ts::UnknownType>(typeIn))
        {
            return mlir::StringAttr::get(context, "unknown");
        }
        else if (isa<mlir_ts::RefType>(typeIn))
        {
            return mlir::StringAttr::get(context, "ref");
        }
        else if (isa<mlir_ts::ValueRefType>(typeIn))
        {
            return mlir::StringAttr::get(context, "valueRef");
        }
        else if (isa<mlir_ts::UnionType>(typeIn))
        {
            return mlir::StringAttr::get(context, "union");
        }                

        return mlir::StringAttr::get(context, "<uknown>");
    }

    mlir::Attribute convertFromFloatAttrIntoType(mlir::Attribute attr, mlir::Type destType, mlir::OpBuilder &builder)
    {
        mlir::Type srcType;
        if (auto typedAttr = dyn_cast<mlir::TypedAttr>(attr))
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
            return builder.getFloatAttr(type, mlir::cast<mlir::IntegerAttr>(attr).getValue().signedRoundToDouble());
        }

        // this is Float
        if (srcType.isIntOrIndexOrFloat())
        {
            return builder.getFloatAttr(type, mlir::cast<mlir::FloatAttr>(attr).getValue().convertToDouble());
        }        

        return mlir::Attribute();
    }    

    mlir::Attribute convertAttrIntoType(mlir::Attribute attr, mlir::Type destType, mlir::OpBuilder &builder)
    {
        mlir::Type srcType;
        if (auto typedAttr = dyn_cast<mlir::TypedAttr>(attr))
        {
            srcType = typedAttr.getType();
        }

        LLVM_DEBUG(llvm::dbgs() << "\n!! attr from type: " << srcType << " to: " << destType << "\n";);      

        if (srcType == destType)
        {
            return attr;
        }

        if (isa<mlir_ts::NumberType>(destType))
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
                auto val = mlir::cast<mlir::IntegerAttr>(attr).getValue();
                APInt newVal(
                    destType.getIntOrFloatBitWidth(), 
                    val.getZExtValue());
                auto attrVal = builder.getIntegerAttr(destType, newVal);
                return attrVal;                
            }            
            else if (srcType.isIntOrIndex())
            {
                // integer
                auto val = mlir::cast<mlir::IntegerAttr>(attr).getValue();
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
                mlir::cast<mlir::FloatAttr>(attr).getValue().convertToInteger(newVal, llvm::APFloatBase::rmNearestTiesToEven, &lossy);
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

    mlir::Type mergeUnionType(mlir::Location location, mlir::Type type)
    {
        if (auto unionType = dyn_cast<mlir_ts::UnionType>(type))
        {
            return getUnionTypeWithMerge(location, unionType);
        }

        return type;
    }

    mlir::Type stripLiteralType(mlir::Type type)
    {
        if (auto literalType = dyn_cast<mlir_ts::LiteralType>(type))
        {
            return literalType.getElementType();
        }

        return type;
    }

    // TODO: can be static;
    mlir::Type stripOptionalType(mlir::Type type)
    {
        if (auto optType = dyn_cast<mlir_ts::OptionalType>(type))
        {
            return optType.getElementType();
        }

        return type;
    }    

    mlir::Type stripRefType(mlir::Type type)
    {
        if (auto refType = dyn_cast<mlir_ts::RefType>(type))
        {
            return refType.getElementType();
        }

        return type;
    }   

    mlir::Type convertConstArrayTypeToArrayType(mlir::Type type)
    {
        if (auto constArrayType = dyn_cast<mlir_ts::ConstArrayType>(type))
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
        if (auto constTupleType = dyn_cast<mlir_ts::ConstTupleType>(type))
        {
            return mlir_ts::TupleType::get(context, constTupleType.getFields());
        }

        return type;
    }

    mlir::Type convertTupleTypeToConstTupleType(mlir::Type type)
    {
        // tuple is value and copied already
        if (auto tupleType = dyn_cast<mlir_ts::TupleType>(type))
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
            if (isa<mlir_ts::OpaqueType>(type))
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
            if (isa<mlir_ts::OpaqueType>(type))
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
        return !type || isa<mlir::NoneType>(type);
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

    bool isBuiltinFunctionType(mlir::Value actualFuncRefValue) 
    {
        auto attrName = StringRef(IDENTIFIER_ATTR_NAME);
        auto virtAttrName = StringRef(BUILTIN_FUNC_ATTR_NAME);
        auto definingOp = actualFuncRefValue.getDefiningOp();
        return (isNoneType(actualFuncRefValue.getType()) || definingOp->hasAttrOfType<mlir::BoolAttr>(virtAttrName)) 
            && definingOp->hasAttrOfType<mlir::FlatSymbolRefAttr>(attrName);
    }

    // TODO: how about multi-index?
    std::pair<mlir::Type, mlir::Type> getIndexSignatureArgumentAndResultTypes(mlir::Type indexSignatureType)
    {
        if (auto funcType = dyn_cast<mlir_ts::FunctionType>(indexSignatureType))
        {
            if (funcType.getNumInputs() == 1 && funcType.getNumResults() == 1 
                && (isNumericType(funcType.getInput(0)) || isa<mlir_ts::StringType>(funcType.getInput(0))))
            {
                return {funcType.getInput(0), funcType.getResult(0)};
            }

            // in case if first parameter is Opaque
            if (funcType.getNumInputs() == 2 && funcType.getNumResults() == 1 
                && (isNumericType(funcType.getInput(1)) || isa<mlir_ts::StringType>(funcType.getInput(1))))
            {
                return {funcType.getInput(1), funcType.getResult(0)};
            }            
        }

        assert(false);

        return {mlir::Type(), mlir::Type()};
    }

    mlir::Type getIndexSignatureType(mlir::Type elementType)
    {
        if (!elementType)
        {
            return mlir::Type();
        }

        return mlir_ts::FunctionType::get(context, {mlir_ts::NumberType::get(context)}, {elementType}, false);
    }

    mlir_ts::FunctionType getIndexGetFunctionType(mlir::Type indexSignature)
    {
        auto [arg, res] = getIndexSignatureArgumentAndResultTypes(indexSignature);
        return mlir_ts::FunctionType::get(
            indexSignature.getContext(), 
            {getOpaqueType(), arg}, 
            {res}, 
            false);    
    }

    mlir_ts::FunctionType getIndexSetFunctionType(mlir::Type indexSignature)
    {
        auto [arg, res] = getIndexSignatureArgumentAndResultTypes(indexSignature);        
        return mlir_ts::FunctionType::get(
            indexSignature.getContext(), 
            {getOpaqueType(), arg, res}, 
            {}, 
            false);    
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

        auto f = [&](auto calledFuncType) { return calledFuncType.getInputs().size() > 0 ? calledFuncType.getInputs().front() : mlir::Type(); };

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
                fieldInfos.push_back({mlir::Attribute(), param, false, mlir_ts::AccessLevel::Public});
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

        auto isOptType = isa<mlir_ts::OptionalType>(funcType);

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
                if (i == 0 && (isa<mlir_ts::OpaqueType>(inFuncType.getInput(i)) || isa<mlir_ts::OpaqueType>(resFuncType.getInput(i))))
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

    mlir_ts::FunctionType GetFunctionType(mlir_ts::HybridFunctionType hybridFuncType) {
        return mlir_ts::FunctionType::get(hybridFuncType.getContext(), hybridFuncType.getInputs(), hybridFuncType.getResults(), hybridFuncType.isVarArg());
    }

    mlir_ts::FunctionType GetFunctionType(mlir_ts::BoundFunctionType boundFuncType) {
        return mlir_ts::FunctionType::get(boundFuncType.getContext(), boundFuncType.getInputs(), boundFuncType.getResults(), boundFuncType.isVarArg());
    }

    mlir_ts::FunctionType GetFunctionType(mlir_ts::ExtensionFunctionType extFuncType) {
        return mlir_ts::FunctionType::get(extFuncType.getContext(), extFuncType.getInputs(), extFuncType.getResults(), extFuncType.isVarArg());
    }

    mlir_ts::FunctionType GetFunctionType(mlir::Type inFuncType) {
        inFuncType = stripOptionalType(inFuncType);    
        
        mlir_ts::FunctionType funcType;
        mlir::TypeSwitch<mlir::Type>(inFuncType)
            .Case<mlir_ts::FunctionType>([&](auto functionType) { funcType = functionType; })
            .Case<mlir_ts::HybridFunctionType>([&](auto hybridFunctionType) { funcType = GetFunctionType(hybridFunctionType); })
            .Case<mlir_ts::BoundFunctionType>([&](auto boundFunctionType) { funcType = GetFunctionType(boundFunctionType); })
            .Case<mlir_ts::ExtensionFunctionType>([&](auto extensionFunctionType) { funcType = GetFunctionType(extensionFunctionType); })
            .Default([&](auto type) { funcType = mlir::cast<mlir_ts::FunctionType>(inFuncType); });
            
        return funcType;
    }

    bool CanCastFunctionTypeToFunctionType(mlir::Type inFuncType, mlir::Type resFuncType) {
        inFuncType = stripOptionalType(inFuncType);    
        resFuncType = stripOptionalType(resFuncType);    

        bool result = false;
        mlir::TypeSwitch<mlir::Type>(inFuncType)
            .Case<mlir_ts::FunctionType>([&](auto functionType) { result = CanCastFunctionTypeToFunctionType(functionType, resFuncType); })
            .Case<mlir_ts::HybridFunctionType>([&](auto hybridFunctionType) { result = CanCastFunctionTypeToFunctionType(hybridFunctionType, resFuncType); })
            .Case<mlir_ts::BoundFunctionType>([&](auto boundFunctionType) { result = CanCastFunctionTypeToFunctionType(boundFunctionType, resFuncType); })
            .Case<mlir_ts::ExtensionFunctionType>([&](auto extensionFunctionType) { result = CanCastFunctionTypeToFunctionType(extensionFunctionType, resFuncType); });
            
        return result;
    }    

    bool CanCastFunctionTypeToFunctionType(mlir_ts::FunctionType inFuncType, mlir::Type resFuncType) {
        bool result = false;
        mlir::TypeSwitch<mlir::Type>(resFuncType)
            .Case<mlir_ts::FunctionType>([&](auto functionType) { result = true; })
            .Case<mlir_ts::HybridFunctionType>([&](auto hybridFunctionType) { result = true; })
            .Case<mlir_ts::BoundFunctionType>([&](auto boundFunctionType) { result = false; })
            .Case<mlir_ts::ExtensionFunctionType>([&](auto extensionFunctionType) { result = false; });
            
        return result;
    }    

    bool CanCastFunctionTypeToFunctionType(mlir_ts::HybridFunctionType inFuncType, mlir::Type resFuncType) {
        bool result = false;
        mlir::TypeSwitch<mlir::Type>(resFuncType)
            .Case<mlir_ts::FunctionType>([&](auto functionType) { result = true; /*TODO: extra checking at runtime should be performed */ })
            .Case<mlir_ts::HybridFunctionType>([&](auto hybridFunctionType) { result = true; })
            .Case<mlir_ts::BoundFunctionType>([&](auto boundFunctionType) { result = true; /*TODO: extra checking at runtime should be performed */ }) 
            .Case<mlir_ts::ExtensionFunctionType>([&](auto extensionFunctionType) { result = true; /*TODO: extra checking at runtime should be performed */ });
            
        return result;
    }      

    bool CanCastFunctionTypeToFunctionType(mlir_ts::BoundFunctionType inFuncType, mlir::Type resFuncType) {
        bool result = false;
        mlir::TypeSwitch<mlir::Type>(resFuncType)
            .Case<mlir_ts::FunctionType>([&](auto functionType) { 
                result = true; // yes, it is allowable. to be able to create objects with functionsi
            })
            .Case<mlir_ts::HybridFunctionType>([&](auto hybridFunctionType) { result = true; })
            .Case<mlir_ts::BoundFunctionType>([&](auto boundFunctionType) { result = true; }) 
            .Case<mlir_ts::ExtensionFunctionType>([&](auto extensionFunctionType) { result = true; });
            
        return result;
    }     

    bool CanCastFunctionTypeToFunctionType(mlir_ts::ExtensionFunctionType inFuncType, mlir::Type resFuncType) {
        bool result = false;
        mlir::TypeSwitch<mlir::Type>(resFuncType)
            .Case<mlir_ts::FunctionType>([&](auto functionType) { result = false; })
            .Case<mlir_ts::HybridFunctionType>([&](auto hybridFunctionType) { result = true; })
            .Case<mlir_ts::BoundFunctionType>([&](auto boundFunctionType) { result = true; }) 
            .Case<mlir_ts::ExtensionFunctionType>([&](auto extensionFunctionType) { result = true; });
            
        return result;
    }          

    bool ShouldThisParamBeIgnored(mlir::Type inFuncType, mlir::Type resFuncType) {
        if (inFuncType == resFuncType) return false;
        return isa<mlir_ts::BoundFunctionType>(inFuncType) || isa<mlir_ts::ExtensionFunctionType>(inFuncType);
    }

    MatchResult TestFunctionTypesMatchWithObjectMethods(mlir::Location location, mlir::Type inFuncType, mlir::Type resFuncType, unsigned startParamIn = 0,
                                                        unsigned startParamRes = 0)
    {
        return TestFunctionTypesMatchWithObjectMethods(
            location,
            GetFunctionType(inFuncType), 
            GetFunctionType(resFuncType), 
            startParamIn + ShouldThisParamBeIgnored(inFuncType, resFuncType) ? 1 : 0, 
            startParamRes + ShouldThisParamBeIgnored(resFuncType, inFuncType) ? 1 : 0);
    }

    bool isBoolType(mlir::Type type) {
        if (isa<mlir_ts::BooleanType>(type) || isa<mlir_ts::TypePredicateType>(type)) return true;
        return isa<mlir::IntegerType>(type) && type.getIntOrFloatBitWidth() == 1;
    }

    bool isAnyOrUnknownOrObjectType(mlir::Type type) {
        return isa<mlir_ts::AnyType>(type) || isa<mlir_ts::UnknownType>(type) || isa<mlir_ts::ObjectType>(type);
    }

    // TODO: add types such as opt, reference, array as they may have nested types Is which is not equal
    // TODO: add travel logic and match only simple types
    bool canMatch(mlir::Location location, mlir::Type left, mlir::Type right) {
        if (left == right) return true;

        left = stripLiteralType(left);
        right = stripLiteralType(right);

        if (left == right) return true;

        if (isBoolType(left) && isBoolType(right)) return true;

        if (isAnyOrUnknownOrObjectType(left) && isAnyOrUnknownOrObjectType(right)) return true;

        // opts
        auto leftOpt = dyn_cast<mlir_ts::OptionalType>(left);
        auto rightOpt = dyn_cast<mlir_ts::OptionalType>(right);
        if (leftOpt && rightOpt)
        {
            return canMatch(location, leftOpt.getElementType(), rightOpt.getElementType());
        }

        // array 
        auto leftArray = dyn_cast<mlir_ts::ArrayType>(left);
        auto rightArray = dyn_cast<mlir_ts::ArrayType>(right);
        if (leftArray && rightArray)
        {
            return canMatch(location, leftArray.getElementType(), rightArray.getElementType());
        }

        // funcs
        if (isAnyFunctionType(left) && isAnyFunctionType(right)) {        
            return TestFunctionTypesMatchWithObjectMethods(location, left, right).result == MatchResultType::Match;
        }

        auto leftClass = dyn_cast<mlir_ts::ClassType>(left);
        auto rightClass = dyn_cast<mlir_ts::ClassType>(right);
        if (leftClass && rightClass)
        {
            llvm::StringMap<std::pair<ts::TypeParameterDOM::TypePtr,mlir::Type>> typeParamsWithArgs;
            return extendsType(location, rightClass, leftClass, typeParamsWithArgs) == ExtendsResult::True;
        }

        auto leftInteface = dyn_cast<mlir_ts::InterfaceType>(left);
        auto rightInteface = dyn_cast<mlir_ts::InterfaceType>(right);
        if (leftInteface && rightInteface)
        {
            llvm::StringMap<std::pair<ts::TypeParameterDOM::TypePtr,mlir::Type>> typeParamsWithArgs;
            return extendsType(location, leftInteface, rightInteface, typeParamsWithArgs) == ExtendsResult::True;
        }        

        return false;
    }

    bool isRefTuple(mlir::Type type)
    {
        if (auto refType = dyn_cast<mlir_ts::RefType>(type))
        {
            return isa<mlir_ts::TupleType>(refType.getElementType()) || isa<mlir_ts::ConstTupleType>(refType.getElementType());
        }

        return false;
    }

    MatchResult TestFunctionTypesMatchWithObjectMethods(mlir::Location location, mlir_ts::FunctionType inFuncType, mlir_ts::FunctionType resFuncType,
                                                        unsigned startParamIn = 0, unsigned startParamRes = 0)
    {
        LLVM_DEBUG(llvm::dbgs() << "\n!!"
                                << "@ TestFunctionTypesMatchWithObjectMethods:" << inFuncType << " -> " << resFuncType << "\n";);

        // TODO: seems capture function with wrong function type
        assert(startParamIn > 0 || inFuncType.getNumInputs() == 0 
            || startParamIn == 0 && inFuncType.getNumInputs() > 0 && !isRefTuple(inFuncType.getInput(0)));

        // 1 we need to skip opaque and objects
        // TODO: we need to refactor func types the way that we can understand if this is func with "captured var", bound to object etc
        if (startParamIn <= 0 && inFuncType.getNumInputs() > 0 &&
            (isa<mlir_ts::OpaqueType>(inFuncType.getInput(0)) || isa<mlir_ts::ObjectType>(inFuncType.getInput(0))))
        {
            startParamIn = 1;
        }

        if (startParamIn <= 1 && inFuncType.getNumInputs() > 1 &&
            (isa<mlir_ts::OpaqueType>(inFuncType.getInput(1)) || isa<mlir_ts::ObjectType>(inFuncType.getInput(1))))
        {
            startParamIn = 2;
        }

        if (startParamRes <= 0 && resFuncType.getNumInputs() > 0 &&
            (isa<mlir_ts::OpaqueType>(resFuncType.getInput(0)) || isa<mlir_ts::ObjectType>(resFuncType.getInput(0))))
        {
            startParamRes = 1;
        }

        if (startParamRes <= 1 && resFuncType.getNumInputs() > 1 &&
            (isa<mlir_ts::OpaqueType>(resFuncType.getInput(1)) || isa<mlir_ts::ObjectType>(resFuncType.getInput(1))))
        {
            startParamRes = 2;
        }

        if (inFuncType.getInputs().size() - startParamIn != resFuncType.getInputs().size() - startParamRes)
        {
            return {MatchResultType::NotMatchArgCount, 0};
        }

        for (unsigned i = 0, e = inFuncType.getInputs().size() - startParamIn; i != e; ++i)
        {
            if (!canMatch(location, inFuncType.getInput(i + startParamIn), resFuncType.getInput(i + startParamRes)))
            {
                // allow certan unmatches such as object & unknown
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
            if (!isInVoid && !isResVoid && !canMatch(location, inRetType, resRetType))
            {
                LLVM_DEBUG(llvm::dbgs() << "\n!! return types do not match [" << inRetType << "] and [" << resRetType << "]\n";);

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

            if (auto optType = dyn_cast<mlir_ts::OptionalType>(type))
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

    mlir::LogicalResult canCastTupleToInterface(mlir::Location location, mlir_ts::TupleType tupleStorageType,
                                                InterfaceInfo::TypePtr newInterfacePtr, bool suppressErrors = false)
    {
        SmallVector<VirtualMethodOrFieldInfo> virtualTable;
        return getInterfaceVirtualTableForObject(location, tupleStorageType, newInterfacePtr, virtualTable, suppressErrors);
    }

    std::string to_print(mlir::Type type)
    {
        SmallString<128> exportType;
        raw_svector_ostream rso(exportType);        

        MLIRPrinter mp{};
        mp.printType<raw_svector_ostream>(rso, type);
        return exportType.str().str();      
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
            [&](mlir::Attribute id, mlir::Type fieldType, bool isConditional) -> std::pair<mlir_ts::FieldInfo, mlir::LogicalResult> {
                auto foundIndex = tupleStorageType.getIndex(id);
                if (foundIndex >= 0)
                {
                    auto foundField = tupleStorageType.getFieldInfo(foundIndex);
                    auto test = isa<mlir_ts::FunctionType>(foundField.type) && isa<mlir_ts::FunctionType>(fieldType)
                                    ? TestFunctionTypesMatchWithObjectMethods(location, foundField.type, fieldType).result == MatchResultType::Match
                                    : stripLiteralType(fieldType) == stripLiteralType(foundField.type);
                    if (!test)
                    {
                        LLVM_DEBUG(llvm::dbgs() << "field " << id << " not matching type: " << fieldType << " and "
                                            << foundField.type << " in interface '" << newInterfacePtr->fullName
                                            << "' for object '" << to_print(tupleStorageType) << "'";);                                    

                        if (!suppressErrors)
                        {
                            emitError(location) << "field " << id << " not matching type: " << fieldType << " and "
                                                << foundField.type << " in interface '" << newInterfacePtr->fullName
                                                << "' for object '" << to_print(tupleStorageType) << "'";

                            return {emptyFieldInfo, mlir::failure()};
                        }

                        return {emptyFieldInfo, mlir::success()};
                    }

                    return {foundField, mlir::success()};
                }

                LLVM_DEBUG(llvm::dbgs() << id << " field can't be found for interface '"
                                    << newInterfacePtr->fullName << "' in object '" << to_print(tupleStorageType) << "'";);

                if (!isConditional && !suppressErrors)
                {
                    emitError(location) << id << " field can't be found for interface '"
                                        << newInterfacePtr->fullName << "' in object '" << to_print(tupleStorageType) << "'";
                    return {emptyFieldInfo, mlir::failure()};
                }

                return {emptyFieldInfo, mlir::success()};
            },
            [&](std::string methodName, mlir_ts::FunctionType funcType, bool isConditional, int interfacePosIndex) -> std::pair<MethodInfo &, mlir::LogicalResult> {
                auto id = MLIRHelper::TupleFieldName(methodName, funcType.getContext());
                auto foundIndex = tupleStorageType.getIndex(id);
                if (foundIndex >= 0)
                {
                    auto foundField = tupleStorageType.getFieldInfo(foundIndex);
                    auto test = isa<mlir_ts::FunctionType>(foundField.type)
                                    ? TestFunctionTypesMatchWithObjectMethods(location, foundField.type, funcType).result ==
                                          MatchResultType::Match
                                    : funcType == foundField.type;
                    if (!test)
                    {
                        LLVM_DEBUG(llvm::dbgs() << "method " << id << " not matching type: " << to_print(funcType) << " and "
                                            << to_print(foundField.type) << " in interface '" << newInterfacePtr->fullName
                                            << "' for object '" << to_print(tupleStorageType) << "'";);                                    

                        if (!suppressErrors)
                        {
                            emitError(location) << "method " << id << " not matching type: " << to_print(funcType) << " and "
                                                << to_print(foundField.type) << " in interface '" << newInterfacePtr->fullName
                                                << "' for object '" << to_print(tupleStorageType) << "'";

                            return {emptyMethod, mlir::failure()};
                        }

                        return {emptyMethod, mlir::success()};
                    }         

                    // TODO: we do not return method, as it should be resolved in fields request
                }

                return {emptyMethod, mlir::success()};
            },
            true);

        return result;
    }

    bool equalFunctionTypes(mlir::Type srcType, mlir::Type destType, bool ignoreThisType = false)
    {
        auto srcTypeUnwrapped = stripOptionalType(srcType);
        auto destTypeUnwrapped = stripOptionalType(destType);

        auto isSrcTypeFunc = isa<mlir_ts::FunctionType>(srcTypeUnwrapped);
        auto isDstTypeFunc = isa<mlir_ts::FunctionType>(destTypeUnwrapped);
        if (!isSrcTypeFunc && isDstTypeFunc)
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

        if (ignoreThisType)
        {
            if (!isSrcTypeFunc)
            {
                srcInputs = srcInputs.drop_back();
            }

            if (!isDstTypeFunc)
            {
                destInputs = destInputs.drop_back();
            }
        }

        return TestFunctionTypesMatch(srcInputs, destInputs, srcResults, destResults, srcIsVarArg, destIsVarArg).result == MatchResultType::Match;
    }

    bool canCastFromTo(mlir::Location location, mlir::Type srcType, mlir::Type destType)
    {
        if (srcType == destType)
        {
            return true;
        }
        
        if (canWideTypeWithoutDataLoss(srcType, destType))
        {
            return true;
        }

        if (auto constTuple = dyn_cast<mlir_ts::ConstTupleType>(srcType))
        {
            if (auto matchConstTuple = dyn_cast<mlir_ts::ConstTupleType>(destType))
            {
                return canCastFromToLogic(constTuple, matchConstTuple);
            }

            if (auto matchTuple = dyn_cast<mlir_ts::TupleType>(destType))
            {
                return canCastFromToLogic(constTuple, matchTuple);
            }

            if (auto ifaceType = dyn_cast<mlir_ts::InterfaceType>(destType))
            {
                if (getInterfaceInfoByFullName)
                {
                    if (auto ifaceTypeInfo = getInterfaceInfoByFullName(ifaceType.getName().getValue()))
                    {
                        return mlir::succeeded(canCastTupleToInterface(location, convertConstTupleTypeToTupleType(constTuple), ifaceTypeInfo));
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

        if (auto tuple = dyn_cast<mlir_ts::TupleType>(srcType))
        {
            if (auto matchTuple = dyn_cast<mlir_ts::TupleType>(destType))
            {
                return canCastFromToLogic(tuple, matchTuple);
            }

            if (auto ifaceType = dyn_cast<mlir_ts::InterfaceType>(destType))
            {
                if (getInterfaceInfoByFullName)
                {
                    if (auto ifaceTypeInfo = getInterfaceInfoByFullName(ifaceType.getName().getValue()))
                    {
                        return mlir::succeeded(canCastTupleToInterface(location, tuple, ifaceTypeInfo));
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
            return equalFunctionTypes(srcType, destType);
        }

        if (auto unionType = dyn_cast<mlir_ts::UnionType>(destType))
        {
            // calculate store size
            auto pred = [&](auto &item) { return canCastFromTo(location, item, srcType); };
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

            if (auto optType = dyn_cast<mlir_ts::OptionalType>(type))
            {
                return testType(optType.getElementType());
            }

            return false;
        };

        return llvm::any_of(type.getFields(), [&](::mlir::typescript::FieldInfo fi) { return testType(fi.type); });
    }

    bool hasUndefines(mlir::Type type)
    {
        if (auto constTuple = dyn_cast<mlir_ts::ConstTupleType>(type))
        {
            return hasUndefinesLogic(constTuple);
        }

        if (auto tuple = dyn_cast<mlir_ts::TupleType>(type))
        {
            return hasUndefinesLogic(tuple);
        }

        return false;
    }

    mlir::Type mergeIntTypes(mlir::Type typeLeft, mlir::Type typeRight, bool& found)
    {
        found = false;
        auto intTypeLeft = dyn_cast<mlir::IntegerType>(typeLeft);
        auto intTypeRight = dyn_cast<mlir::IntegerType>(typeRight);

        if (typeLeft.isIndex())
        {
            intTypeLeft = mlir::IntegerType::get(typeLeft.getContext(), compileOptions.sizeBits);
        }

        if (typeRight.isIndex())
        {
            intTypeRight = mlir::IntegerType::get(typeRight.getContext(), compileOptions.sizeBits);
        }

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

    mlir::Type mergeFuncTypes(mlir::Type typeLeft, mlir::Type typeRight, bool& found)
    {
        found = false;

        if (!isAnyFunctionType(typeLeft) || !isAnyFunctionType(typeRight))
        {
            return mlir::Type();
        }

        auto leftTypeUnwrapped = stripOptionalType(typeLeft);
        auto rightTypeUnwrapped = stripOptionalType(typeRight);

        auto isLeftTypeFunc = isa<mlir_ts::FunctionType>(leftTypeUnwrapped);
        auto isRightTypeFunc = isa<mlir_ts::FunctionType>(rightTypeUnwrapped);

        auto leftInputs = getParamsFromFuncRef(typeLeft);
        auto rightInputs = getParamsFromFuncRef(typeRight);
        auto leftResults = getReturnsFromFuncRef(typeLeft);
        auto rightResults = getReturnsFromFuncRef(typeRight);
        auto leftIsVarArg = getVarArgFromFuncRef(typeLeft);
        auto rightIsVarArg = getVarArgFromFuncRef(typeRight);

        auto hybridFuncIsNeeded = false;
        if (!isLeftTypeFunc)
        {
            leftInputs = leftInputs.drop_back();
            hybridFuncIsNeeded = true;
        }

        if (!isRightTypeFunc)
        {
            rightInputs = rightInputs.drop_back();
            hybridFuncIsNeeded = true;
        }

        auto equalFuncs = TestFunctionTypesMatch(leftInputs, rightInputs, leftResults, rightResults, leftIsVarArg, rightIsVarArg).result == MatchResultType::Match;
        if (equalFuncs)
        {
            found = true;
            
            if (isa<mlir_ts::BoundFunctionType>(leftTypeUnwrapped) && isa<mlir_ts::BoundFunctionType>(rightTypeUnwrapped))
            {
                return typeLeft;
            }

            if (isa<mlir_ts::FunctionType>(leftTypeUnwrapped) && isa<mlir_ts::FunctionType>(rightTypeUnwrapped))
            {
                return typeLeft;
            }

            if (isa<mlir_ts::HybridFunctionType>(leftTypeUnwrapped) && isa<mlir_ts::HybridFunctionType>(rightTypeUnwrapped))
            {
                return typeLeft;
            }

            if (isa<mlir_ts::ExtensionFunctionType>(leftTypeUnwrapped) && isa<mlir_ts::ExtensionFunctionType>(rightTypeUnwrapped))
            {
                return typeLeft;
            }

            return mlir_ts::HybridFunctionType::get(context, leftInputs, leftResults, leftIsVarArg);
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

    // TODO: review using canCast in detecting base Type. index, int, number
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

        if (isa<mlir::IntegerType>(srcType) || isa<mlir::IndexType>(srcType))
        {
            if (isa<mlir_ts::NumberType>(dstType))
            {
                return true;
            }
        }

        // we should not treat boolean as integer
        // if (isa<mlir_ts::BooleanType>(srcType))
        // {
        //     if (isa<mlir::IntegerType>(dstType) && dstType.getIntOrFloatBitWidth() > 0)
        //     {
        //         return true;
        //     }

        //     if (isa<mlir_ts::NumberType>(dstType))
        //     {
        //         return true;
        //     }
        // }

        // but we can't cast TypePredicate to boolean as we will lose the information about type
        if (isa<mlir_ts::BooleanType>(srcType))
        {
            if (isa<mlir_ts::TypePredicateType>(dstType))
            {
                return true;
            }
        }        

        if (auto dstEnumType = dyn_cast<mlir_ts::EnumType>(dstType))
        {
            if (dstEnumType.getElementType() == srcType)
            {
                return true;
            }
        }          

        if (auto literalType = dyn_cast<mlir_ts::LiteralType>(srcType))
        {
            auto litType = literalType.getElementType();
            if (auto litIntType = dyn_cast<mlir::IntegerType>(litType))
            {
                if (isa<mlir::IndexType>(dstType) && litIntType.getIntOrFloatBitWidth() <= (unsigned int)compileOptions.sizeBits)
                {
                    return true;
                }
            }

            return canWideTypeWithoutDataLoss(litType, dstType);
        }

        if (auto enumType = dyn_cast<mlir_ts::EnumType>(srcType))
        {
            return canWideTypeWithoutDataLoss(enumType.getElementType(), dstType);
        }        

        // wide range type can't be stored into literal
        if (auto optionalType = dyn_cast<mlir_ts::OptionalType>(dstType))
        {
            if (auto srcOptionalType = dyn_cast<mlir_ts::OptionalType>(srcType))
            {
                return canWideTypeWithoutDataLoss(srcOptionalType.getElementType(), optionalType.getElementType());
            }

            if (auto undefType = dyn_cast<mlir_ts::UndefinedType>(srcType))
            {
                return true;
            }

            return canWideTypeWithoutDataLoss(srcType, optionalType.getElementType());
        }

        if (auto arrayType = dyn_cast<mlir_ts::ArrayType>(dstType))
        {
            if (auto constArrayType = dyn_cast<mlir_ts::ConstArrayType>(srcType))
            {
                if (constArrayType.getSize() == 0)
                {
                    return true;
                }

                return canWideTypeWithoutDataLoss(constArrayType.getElementType(), arrayType.getElementType());
            }
        }        

        // native types
        auto destIntType = dyn_cast<mlir::IntegerType>(dstType);
        auto srcIntType = dyn_cast<mlir::IntegerType>(srcType);

        if (dstType.isIndex())
        {
            destIntType = mlir::IntegerType::get(dstType.getContext(), compileOptions.sizeBits);
        }

        if (srcType.isIndex())
        {
            srcIntType = mlir::IntegerType::get(srcType.getContext(), compileOptions.sizeBits);
        }

        if (destIntType && srcIntType)
        {
            if ((srcIntType.getSignedness() == destIntType.getSignedness() || srcIntType.isSignless() || destIntType.isSignless())
                && srcIntType.getIntOrFloatBitWidth() <= destIntType.getIntOrFloatBitWidth())
            {
                return true;
            }

            if (destIntType.getIntOrFloatBitWidth() > srcIntType.getIntOrFloatBitWidth() && (destIntType.getIntOrFloatBitWidth() - srcIntType.getIntOrFloatBitWidth()) > 1)
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

        if (auto destFloatType = dyn_cast<mlir::FloatType>(dstType))
        {
            if (auto srcFloatType = dyn_cast<mlir::FloatType>(srcType))
            {
                if (srcFloatType.getIntOrFloatBitWidth() <= destFloatType.getIntOrFloatBitWidth())
                {
                    return true;
                }
            }

            if (auto srcIntType = dyn_cast<mlir::IntegerType>(srcType))
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

        if (auto constTuple = dyn_cast<mlir_ts::ConstTupleType>(srcType))
        {
            if (auto matchConstTuple = dyn_cast<mlir_ts::ConstTupleType>(dstType))
            {
                return isSizeEqualLogic(constTuple, matchConstTuple);
            }

            if (auto matchTuple = dyn_cast<mlir_ts::TupleType>(dstType))
            {
                return isSizeEqualLogic(constTuple, matchTuple);
            }
        }

        if (auto tuple = dyn_cast<mlir_ts::TupleType>(srcType))
        {
            if (auto matchTuple = dyn_cast<mlir_ts::TupleType>(dstType))
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
        if (auto literalType = dyn_cast<mlir_ts::LiteralType>(type))
        {
            return literalType.getElementType();
        }

        if (auto enumType = dyn_cast<mlir_ts::EnumType>(type))
        {
            return enumType.getElementType();
        }

        return type;
    }

    bool isUnionTypeNeedsTag(mlir::Location location, mlir_ts::UnionType unionType)
    {
        mlir::Type baseType;
        return isUnionTypeNeedsTag(location, unionType, baseType);
    }

    bool isUnionTypeNeedsTag(mlir::Location location, mlir_ts::UnionType unionType, mlir::Type &baseType)
    {
        auto storeType = getUnionTypeWithMerge(location, unionType.getTypes(), true, true, true);
        baseType = storeType;
        return isa<mlir_ts::UnionType>(storeType);
    }

    ExtendsResult appendInferTypeToContext(mlir::Location location, mlir::Type srcType, mlir_ts::InferType inferType, llvm::StringMap<std::pair<ts::TypeParameterDOM::TypePtr,mlir::Type>> &typeParamsWithArgs, bool useTupleType = false)
    {
        auto name = cast<mlir_ts::NamedGenericType>(inferType.getElementType()).getName().getValue();
        auto currentType = srcType;

        auto existType = typeParamsWithArgs.lookup(name);
        if (existType.second)
        {
            auto namedType = dyn_cast<mlir_ts::NamedGenericType>(existType.second);
            if (namedType && namedType.getName().getValue().compare(name) == 0)
            {
                // replace generic type
                typeParamsWithArgs[name].second = currentType;
            }
            else
            {
                if (useTupleType)
                {
                    SmallVector<mlir_ts::FieldInfo> fieldInfos;

                    if (auto tupleType = dyn_cast<mlir_ts::TupleType>(existType.second))
                    {
                        for (auto param : tupleType.getFields())
                        {
                            fieldInfos.push_back(param);
                        }
                    }
                    else
                    {
                        fieldInfos.push_back({mlir::Attribute(), existType.second, false, mlir_ts::AccessLevel::Public});    
                    }

                    fieldInfos.push_back({mlir::Attribute(), currentType, false, mlir_ts::AccessLevel::Public});

                    currentType = getTupleType(fieldInfos);                    
                }
                else
                {
                    auto defaultUnionType = getUnionType(location, existType.second, currentType);

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

        if (isa<mlir::StringAttr>(attr))
        {
            return mlir_ts::StringType::get(context);
        }

        if (isa<mlir::FloatAttr>(attr))
        {
            return mlir_ts::NumberType::get(context);
        }

        if (isa<mlir::IntegerAttr>(attr))
        {
            return mlir_ts::NumberType::get(context);
        }

        if (auto typedAttr = dyn_cast<mlir::TypedAttr>(attr))
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
            auto litType = field.id && isa<mlir::StringAttr>(field.id) 
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
            auto litType = field.id && isa<mlir::StringAttr>(field.id) 
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

        if (auto constTupleType = dyn_cast<mlir_ts::ConstTupleType>(srcType))
        {
            for (auto &fieldInfo : constTupleType.getFields())
            {
                destTupleFields.push_back(fieldInfo);
            }

            return mlir::success();
        }          
        else if (auto tupleType = dyn_cast<mlir_ts::TupleType>(srcType))
        {
            for (auto &fieldInfo : tupleType.getFields())
            {
                destTupleFields.push_back(fieldInfo);
            }

            return mlir::success();
        }         
        else if (auto srcInterfaceType = dyn_cast<mlir_ts::InterfaceType>(srcType))
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
        else if (auto srcClassType = dyn_cast<mlir_ts::ClassType>(srcType))
        {
            for (auto &fieldInfo : cast<mlir_ts::ClassStorageType>(srcClassType.getStorageType()).getFields())
            {
                destTupleFields.push_back(fieldInfo);
            }       
            
            return mlir::success();         
        }         
        else if (auto srcClassStorageType = dyn_cast<mlir_ts::ClassStorageType>(srcType))
        {
            for (auto &fieldInfo : srcClassStorageType.getFields())
            {
                destTupleFields.push_back(fieldInfo);
            }       
            
            return mlir::success();            
        }
        else if (auto arrayType = dyn_cast<mlir_ts::ArrayType>(srcType))
        {
            // TODO: do not break the order as it is used in Debug info
            destTupleFields.push_back({ MLIRHelper::TupleFieldName("data", context), mlir_ts::RefType::get(arrayType.getElementType()), false, mlir_ts::AccessLevel::Public });
            destTupleFields.push_back({ MLIRHelper::TupleFieldName(LENGTH_FIELD_NAME, context), mlir::IndexType::get(context), false, mlir_ts::AccessLevel::Public });
            return mlir::success();

        }
        else if (auto stringType = dyn_cast<mlir_ts::StringType>(srcType))
        {
            // TODO: do not break the order as it is used in Debug info
            destTupleFields.push_back({ MLIRHelper::TupleFieldName("data", context), mlir_ts::RefType::get(mlir_ts::CharType::get(context)), false, mlir_ts::AccessLevel::Public });
            destTupleFields.push_back({ MLIRHelper::TupleFieldName(LENGTH_FIELD_NAME, context), mlir::IndexType::get(context), false, mlir_ts::AccessLevel::Public });
            return mlir::success();

        }
        else if (auto optType = dyn_cast<mlir_ts::OptionalType>(srcType))
        {
            // TODO: do not break the order as it is used in Debug info
            destTupleFields.push_back({ MLIRHelper::TupleFieldName("value", context), optType.getElementType(), false, mlir_ts::AccessLevel::Public });
            destTupleFields.push_back({ MLIRHelper::TupleFieldName("hasValue", context), mlir_ts::BooleanType::get(context), false, mlir_ts::AccessLevel::Public });
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

    int getFieldIndexByFieldName(mlir::Type srcType, mlir::Attribute fieldName)
    {
        LLVM_DEBUG(llvm::dbgs() << "!! get index of field '" << fieldName << "' of '" << srcType << "'\n";);

        if (auto constTupleType = dyn_cast<mlir_ts::ConstTupleType>(srcType))
        {
            return constTupleType.getIndex(fieldName);
        }          
        
        if (auto tupleType = dyn_cast<mlir_ts::TupleType>(srcType))
        {
            return tupleType.getIndex(fieldName);
        }  

        llvm_unreachable("not implemented");
    }

    mlir::typescript::FieldInfo getFieldInfoByIndex(mlir::Type srcType, int index)
    {
        LLVM_DEBUG(llvm::dbgs() << "!! get #" << index << " of '" << srcType << "'\n";);

        if (auto constTupleType = dyn_cast<mlir_ts::ConstTupleType>(srcType))
        {
            return constTupleType.getFieldInfo(index);
        }          
        
        if (auto tupleType = dyn_cast<mlir_ts::TupleType>(srcType))
        {
            return tupleType.getFieldInfo(index);
        }  

        llvm_unreachable("not implemented");
    }

    mlir::Type getFieldTypeByFieldName(mlir::Type srcType, mlir::Attribute fieldName)
    {
        LLVM_DEBUG(llvm::dbgs() << "!! get type of field '" << fieldName << "' of '" << srcType << "'\n";);

        if (auto constTupleType = dyn_cast<mlir_ts::ConstTupleType>(srcType))
        {
            auto index = constTupleType.getIndex(fieldName);
            if (index < 0)
            {
                return mlir::Type();
            }

            return constTupleType.getFieldInfo(index).type;
        }          
        
        if (auto tupleType = dyn_cast<mlir_ts::TupleType>(srcType))
        {
            auto index = tupleType.getIndex(fieldName);
            if (index < 0)
            {
                return mlir::Type();
            }

            return tupleType.getFieldInfo(index).type;
        }  

        if (auto srcInterfaceType = dyn_cast<mlir_ts::InterfaceType>(srcType))
        {
            if (auto srcInterfaceInfo = getInterfaceInfoByFullName(srcInterfaceType.getName().getValue()))
            {
                auto fieldInfo = srcInterfaceInfo->findField(fieldName);
                if (fieldInfo)
                {
                    return fieldInfo->type;
                }

                if (auto strName = dyn_cast<mlir::StringAttr>(fieldName))
                {
                    auto methodInfo = srcInterfaceInfo->findMethod(strName);
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

        if (auto srcClassType = dyn_cast<mlir_ts::ClassType>(srcType))
        {
            if (auto srcClassInfo = getInterfaceInfoByFullName(srcClassType.getName().getValue()))
            {
                auto fieldInfo = srcClassInfo->findField(fieldName);
                if (fieldInfo)
                {
                    return fieldInfo->type;
                }

                if (auto strName = dyn_cast<mlir::StringAttr>(fieldName))
                {
                    auto methodInfo = srcClassInfo->findMethod(strName);
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
        if (auto arrayType = dyn_cast<mlir_ts::ArrayType>(srcType))
        {
            if (fieldName == MLIRHelper::TupleFieldName(LENGTH_FIELD_NAME, context))
            {
                return mlir_ts::NumberType::get(context);
            }

            // TODO: temp hack to support extends { readonly length: number; readonly [n: number]: ElementOfArray<A>; }
            if (fieldName == MLIRHelper::TupleFieldName(INDEX_ACCESS_GET_FIELD_NAME, context))
            {
                return mlir_ts::AnyType::get(context);
            }

            // TODO: temp hack to support extends { readonly length: number; readonly [n: number]: ElementOfArray<A>; }
            if (fieldName == MLIRHelper::TupleFieldName(INDEX_ACCESS_SET_FIELD_NAME, context))
            {
                return mlir_ts::AnyType::get(context);
            }

            llvm_unreachable("not implemented");
        }        

        // TODO: read fields info from class Array
        if (auto constArrayType = dyn_cast<mlir_ts::ConstArrayType>(srcType))
        {
            if (fieldName == MLIRHelper::TupleFieldName(LENGTH_FIELD_NAME, context))
            {
                return mlir_ts::NumberType::get(context);
            }

            // TODO: temp hack to support extends { readonly length: number; readonly [n: number]: ElementOfArray<A>; }
            if (fieldName == MLIRHelper::TupleFieldName(INDEX_ACCESS_GET_FIELD_NAME, context))
            {
                return mlir_ts::AnyType::get(context);
            }

            llvm_unreachable("not implemented");
        }        

        // TODO: read data from String class
        if (auto stringType = dyn_cast<mlir_ts::StringType>(srcType))
        {
            if (fieldName == MLIRHelper::TupleFieldName(LENGTH_FIELD_NAME, context))
            {
                return mlir_ts::NumberType::get(context);
            }

            llvm_unreachable("not implemented");
        }        

        if (auto unionType = dyn_cast<mlir_ts::UnionType>(srcType))
        {
            llvm::SmallVector<mlir::Type> types;
            for (auto &item : unionType.getTypes())
            {
                auto fieldType = getFieldTypeByFieldName(item, fieldName);
                if (fieldType)
                {
                    types.push_back(fieldType);
                }
            }

            if (types.size() == 0)
            {
                return mlir::Type();
            }

            return mlir_ts::UnionType::get(context, types);
        }        

        if (auto namedGenericType = dyn_cast<mlir_ts::NamedGenericType>(srcType))
        {
            auto typedAttr = dyn_cast<mlir::TypedAttr>(fieldName);
            // TODO: make common function
            auto fieldNameLiteralType = mlir_ts::LiteralType::get(fieldName, typedAttr && !isNoneType(typedAttr.getType()) ? typedAttr.getType() : mlir_ts::StringType::get(context));
            return mlir_ts::IndexAccessType::get(srcType, fieldNameLiteralType);
        }

        if (auto anyType = dyn_cast<mlir_ts::AnyType>(srcType))
        {
            return anyType;
        }        

        if (auto unknownType = dyn_cast<mlir_ts::UnknownType>(srcType))
        {
            // TODO: but in index type it should return "any"
            return mlir::Type();
        }          

        return mlir::Type();
    }

    ExtendsResult extendsTypeFuncTypes(mlir::Location location, mlir::Type srcType, mlir::Type extendType,
        llvm::StringMap<std::pair<ts::TypeParameterDOM::TypePtr,mlir::Type>> &typeParamsWithArgs, int skipSrcParams = 0)
    {
            auto srcParams = getParamsFromFuncRef(srcType);
            auto extParams = getParamsFromFuncRef(extendType);

            //auto srcIsVarArgs = getVarArgFromFuncRef(srcType);
            auto extIsVarArgs = getVarArgFromFuncRef(extendType);

            auto srcReturnType = getReturnTypeFromFuncRef(srcType);
            auto extReturnType = getReturnTypeFromFuncRef(extendType);       

            return extendsTypeFuncTypes(location, srcParams, extParams, extIsVarArgs, srcReturnType, extReturnType, typeParamsWithArgs, skipSrcParams);    
    }

    ExtendsResult extendsTypeFuncTypes(mlir::Location location, ArrayRef<mlir::Type> srcParams, ArrayRef<mlir::Type> extParams, bool extIsVarArgs, 
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
            if (auto inferType = dyn_cast<mlir_ts::InferType>(extParamType))
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
                if (auto arrayType = dyn_cast<mlir_ts::ArrayType>(extParamType))
                {
                    extParamType = arrayType.getElementType();
                }
            }

            auto extendsResult = extendsType(location, srcParamType, extParamType, typeParamsWithArgs, useTupleWhenMergeTypes);
            if (extendsResult != ExtendsResult::True)
            {
                return extendsResult;
            }
        }      

        // compare return types
        return extendsType(location, srcReturnType, extReturnType, typeParamsWithArgs);

    }

    ExtendsResult extendsType(mlir::Location location, mlir::Type srcType, mlir::Type extendType, llvm::StringMap<std::pair<ts::TypeParameterDOM::TypePtr,mlir::Type>> &typeParamsWithArgs, bool useTupleWhenMergeTypes = false)
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

        // to support infer types
        if (auto inferType = dyn_cast<mlir_ts::InferType>(extendType))
        {
            return appendInferTypeToContext(location, srcType, inferType, typeParamsWithArgs, useTupleWhenMergeTypes);
        }

        if (auto anyType = dyn_cast_or_null<mlir_ts::AnyType>(srcType))
        {
            SmallVector<mlir_ts::InferType> inferTypes;
            getAllInferTypes(extendType, inferTypes);
            for (auto inferType : inferTypes)
            {
                appendInferTypeToContext(location, mlir_ts::UnknownType::get(context), inferType, typeParamsWithArgs, useTupleWhenMergeTypes);
            }

            // TODO: add all infer types in extends to "unknown"
            return ExtendsResult::Any;
        }        

        if (auto neverType = dyn_cast_or_null<mlir_ts::NeverType>(srcType))
        {
            SmallVector<mlir_ts::InferType> inferTypes;
            getAllInferTypes(extendType, inferTypes);
            for (auto inferType : inferTypes)
            {
                appendInferTypeToContext(location, mlir_ts::NeverType::get(context), inferType, typeParamsWithArgs, useTupleWhenMergeTypes);
            }

            // TODO: add all infer types in extends to "never"
            return ExtendsResult::Never;
        }        

        if (auto neverType = dyn_cast<mlir_ts::NeverType>(extendType))
        {
            return ExtendsResult::False;
        }        

        if (auto unknownType = dyn_cast<mlir_ts::UnknownType>(extendType))
        {
            return ExtendsResult::True;
        }

        if (auto anyType = dyn_cast<mlir_ts::AnyType>(extendType))
        {
            return ExtendsResult::True;
        }

        auto isOptional = false;
        if (auto optType = dyn_cast_or_null<mlir_ts::OptionalType>(srcType)) {
            isOptional = true;
            srcType = optType.getElementType();
        }

        if (!srcType)
        {
            return ExtendsResult::False;
        }        

        if (auto unionType = dyn_cast<mlir_ts::UnionType>(extendType))
        {
            auto falseResult = ExtendsResult::False;
            auto pred = [&](auto &item) { 
                auto unionExtResult = extendsType(location, srcType, item, typeParamsWithArgs);
                if (unionExtResult == ExtendsResult::Never)
                {
                    falseResult = unionExtResult;
                }                

                return isTrue(unionExtResult); 
            };
            auto types = unionType.getTypes();
            return std::find_if(types.begin(), types.end(), pred) != types.end() ? ExtendsResult::True : falseResult;
        }

        if (auto tupleType = dyn_cast<mlir_ts::TupleType>(extendType))
        {
            auto falseResult = ExtendsResult::False;
            auto pred = [&](auto &item) { 
                if (item.id)
                {
                    auto fieldType = getFieldTypeByFieldName(srcType, item.id);
                    auto fieldExtResult = extendsType(location, fieldType, item.type, typeParamsWithArgs);
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

        if (auto constTupleType = dyn_cast<mlir_ts::ConstTupleType>(extendType))
        {
            auto falseResult = ExtendsResult::False;
            auto pred = [&](auto &item) { 
                if (item.id)
                {
                    auto fieldType = getFieldTypeByFieldName(srcType, item.id);
                    auto fieldExtResult = extendsType(location, fieldType, item.type, typeParamsWithArgs);
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

        if (auto literalType = dyn_cast<mlir_ts::LiteralType>(srcType))
        {
            if (auto litExt = dyn_cast<mlir_ts::LiteralType>(extendType))
            {
                return ExtendsResult::False;
            }

            return extendsType(location, literalType.getElementType(), extendType, typeParamsWithArgs);
        }

        if (auto srcArray = dyn_cast<mlir_ts::ArrayType>(srcType))
        {
            if (auto extArray = dyn_cast<mlir_ts::ArrayType>(extendType))
            {
                return extendsType(location, srcArray.getElementType(), extArray.getElementType(), typeParamsWithArgs);
            }
        }

        if (auto srcArray = dyn_cast<mlir_ts::ConstArrayType>(srcType))
        {
            if (auto extArray = dyn_cast<mlir_ts::ArrayType>(extendType))
            {
                return extendsType(location, srcArray.getElementType(), extArray.getElementType(), typeParamsWithArgs);
            }
        }

        // Special case when we have string type (widen from Literal Type)
        if (auto literalType = dyn_cast<mlir_ts::LiteralType>(extendType))
        {
            return extendsType(location, srcType, literalType.getElementType(), typeParamsWithArgs);
        }        

        if (auto unionType = dyn_cast<mlir_ts::UnionType>(srcType))
        {
            auto falseResult = ExtendsResult::False;
            auto pred = [&](auto &item) { 
                auto unionExtResult = extendsType(location, item, extendType, typeParamsWithArgs);
                if (unionExtResult == ExtendsResult::Never)
                {
                    falseResult = unionExtResult;
                }                

                return isTrue(unionExtResult); 
            };
            auto types = unionType.getTypes();
            auto foundResult = std::find_if(types.begin(), types.end(), pred) != types.end() ? ExtendsResult::True : falseResult;
            if (isOptional && foundResult == falseResult)
            {
                auto undefType = mlir_ts::UndefinedType::get(srcType.getContext());
                foundResult = pred(undefType) ? ExtendsResult::True : falseResult;
            }

            return foundResult;
        }

        // seems it is generic interface
        if (auto srcInterfaceType = dyn_cast<mlir_ts::InterfaceType>(srcType))
        {
            if (auto extInterfaceType = dyn_cast<mlir_ts::InterfaceType>(extendType))
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

                                        return extendsType(location, srcType, extType, typeParamsWithArgs);
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

        if (auto srcClassType = dyn_cast<mlir_ts::ClassType>(srcType))
        {
            if (auto extClassType = dyn_cast<mlir_ts::ClassType>(extendType))
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

                                        return extendsType(location, srcType, extType, typeParamsWithArgs);
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
            auto thisType = getFirstParamFromFuncRef(srcType);
            auto skipFirst = thisType && (mlir::isa<mlir_ts::OpaqueType>(thisType) 
                || mlir::isa<mlir_ts::ClassType>(thisType) 
                || mlir::isa<mlir_ts::ClassStorageType>(thisType) 
                || mlir::isa<mlir_ts::ObjectType>(thisType) 
                || mlir::isa<mlir_ts::ObjectStorageType>(thisType));

            return extendsTypeFuncTypes(location, srcType, extendType, typeParamsWithArgs, skipFirst ? 1 : 0);
        }

        if (auto constructType = dyn_cast<mlir_ts::ConstructFunctionType>(extendType))
        {
            if (auto srcClassType = dyn_cast<mlir_ts::ClassType>(srcType))
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
                    return extendsTypeFuncTypes(location, constrWithRetType, extendType, typeParamsWithArgs, 1/*because of this param*/);
                }
            }

            return ExtendsResult::False;
        }

        if (auto ifaceType = dyn_cast<mlir_ts::InterfaceType>(srcType))
        {
            auto interfaceInfo = getInterfaceInfoByFullName(ifaceType.getName().getValue());
            assert(interfaceInfo);
            auto falseResult = ExtendsResult::False;
            for (auto extend : interfaceInfo->extends)
            {
                auto extResult = extendsType(location, extend.second->interfaceType, extendType, typeParamsWithArgs);
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

        if (auto classType = dyn_cast<mlir_ts::ClassType>(srcType))
        {
            auto classInfo = getClassInfoByFullName(classType.getName().getValue());
            assert(classInfo);
            auto falseResult = ExtendsResult::False;
            for (auto extend : classInfo->baseClasses)
            {
                auto extResult = extendsType(location, extend->classType, extendType, typeParamsWithArgs);
                if (extResult == ExtendsResult::True)
                {
                    return ExtendsResult::True;
                }

                if (extResult == ExtendsResult::Never)
                {
                    falseResult = ExtendsResult::Never;
                }                  
            }

            if (auto ifaceType = dyn_cast<mlir_ts::InterfaceType>(extendType))
            {
                for (auto extend : classInfo->implements)
                {
                    auto extResult = extendsType(location, extend.interface->interfaceType, extendType, typeParamsWithArgs);
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

        if (auto objType = dyn_cast<mlir_ts::ObjectType>(extendType))
        {
            if (isa<mlir_ts::AnyType>(objType.getStorageType()))
            {
                return (isa<mlir_ts::TupleType>(srcType) || isa<mlir_ts::ConstTupleType>(srcType) || isa<mlir_ts::ObjectType>(srcType)) 
                    ? ExtendsResult::True : ExtendsResult::False;
            }
        }        

        // TODO: do we need to check types inside?
        if ((isa<mlir_ts::TypePredicateType>(srcType) || isa<mlir_ts::BooleanType>(srcType))
            && (isa<mlir_ts::TypePredicateType>(extendType) || isa<mlir_ts::BooleanType>(extendType)))
        {
            return ExtendsResult::True;
        }        

        // TODO: finish Function Types, etc
        LLVM_DEBUG(llvm::dbgs() << "\n!! extendsType [FLASE]\n";);
        return ExtendsResult::False;
    }

    mlir::Type getFirstNonNullUnionType(mlir_ts::UnionType unionType)
    {
        for (auto itemType : unionType.getTypes())
        {
            if (isa<mlir_ts::NullType>(itemType))
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
        if (isa<mlir_ts::UndefinedType>(type))
        {
            unionContext.isUndefined = true;
            return mlir::success();
        }

        if (isa<mlir_ts::NullType>(type))
        {
            unionContext.isNullable = true;
            return mlir::success();
        }            

        if (isa<mlir_ts::AnyType>(type))
        {
            unionContext.isAny = true;
            return mlir::success();
        }

        if (isa<mlir_ts::NeverType>(type))
        {
            return mlir::success();
        }

        if (auto literalType = dyn_cast<mlir_ts::LiteralType>(type))
        {
            unionContext.literalTypes.insert(literalType);
            return mlir::success();
        }

        if (auto optionalType = dyn_cast<mlir_ts::OptionalType>(type))
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

        if (auto unionType = dyn_cast<mlir_ts::UnionType>(type))
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

    mlir::Type getUnionType(mlir::Location location, mlir::Type type1, mlir::Type type2, bool mergeLiterals = true, bool mergeTypes = true)
    {
        if (canCastFromTo(location, type1, type2))
        {
            return type2;
        }

        if (canCastFromTo(location, type2, type1))
        {
            return type1;
        }

        mlir::SmallVector<mlir::Type> types;
        types.push_back(type1);
        types.push_back(type2);
        return getUnionTypeWithMerge(location, types, mergeLiterals, mergeTypes);
    }

    // TODO: review all union merge logic
    mlir::Type getUnionTypeMergeTypes(mlir::Location location, UnionTypeProcessContext &unionContext, bool mergeLiterals = true, bool mergeTypes = true, bool disableStrickNullCheck = false)
    {
        // merge types with literal types
        for (auto literalType : unionContext.literalTypes)
        {
            if (mergeLiterals)
            {
                auto baseType = mlir::cast<mlir_ts::LiteralType>(literalType).getElementType();
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
            isAllLiteralTypes &= isa<mlir_ts::LiteralType>(type);
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

            if (compileOptions.strictNullChecks && !disableStrickNullCheck && unionContext.isNullable)
            {
                return mlir_ts::UnionType::get(context, {resType, getNullType()});             
            }

            return resType;       
        }

        // merge types
        auto doNotMergeLiterals = !mergeLiterals && isAllLiteralTypes;
        if (mergeTypes && !doNotMergeLiterals)
        {
            mlir::SmallVector<mlir::Type> mergedTypesAll;
            this->mergeTypes(location, typesAll, mergedTypesAll);

            if (compileOptions.strictNullChecks && unionContext.isNullable)
            {
                mergedTypesAll.push_back(getNullType());
            }

            mlir::Type retType = mergedTypesAll.size() == 1 ? mergedTypesAll.front() : getUnionType(mergedTypesAll);
            if (unionContext.isUndefined)
            {
                return mlir_ts::OptionalType::get(retType);
            }

            return retType;
        }

        if (compileOptions.strictNullChecks && !disableStrickNullCheck && unionContext.isNullable)
        {
            typesAll.push_back(getNullType());
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
            if (isa<mlir_ts::UndefinedType>(type) || isa<mlir_ts::OptionalType>(type))
            {
                unionContext.isUndefined = true;
                continue;
            }

            if (isa<mlir_ts::NullType>(type))
            {
                unionContext.isNullable = true;
                continue;
            }            

            if (isa<mlir_ts::AnyType>(type))
            {
                unionContext.isAny = true;
                continue;
            }

            if (isa<mlir_ts::NeverType>(type))
            {
                continue;
            }

            if (auto unionType = dyn_cast<mlir_ts::UnionType>(type))
            {
                detectTypeForGroupOfTypes(unionType.getTypes(), unionContext);
            }
        }
    }

    mlir::Type getUnionTypeWithMerge(mlir::Location location, mlir::ArrayRef<mlir::Type> types, bool mergeLiterals = true, bool mergeTypes = true, bool disableStrickNullCheck = false)
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

        return getUnionTypeMergeTypes(location, unionContext, mergeLiterals, mergeTypes, disableStrickNullCheck);
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
            if (isa<mlir_ts::UndefinedType>(type))
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

        return isUndefined 
            ? mlir::Type(mlir_ts::OptionalType::get(mlir_ts::UnionType::get(context, newTypes))) 
            : mlir::Type(mlir_ts::UnionType::get(context, newTypes));
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
        return iter.some(type, [](mlir::Type type) { return type && isa<mlir_ts::NamedGenericType>(type); });
    }

    bool hasInferType(mlir::Type type)
    {
        MLIRTypeIteratorLogic iter(
            getClassInfoByFullName, getGenericClassInfoByFullName, 
            getInterfaceInfoByFullName, getGenericInterfaceInfoByFullName
        );
        return iter.some(type, [](mlir::Type type) { return type && isa<mlir_ts::InferType>(type); });
    }    

    void forEachTypes(mlir::Type type, std::function<bool(mlir::Type)> f)
    {
        MLIRTypeIteratorLogic iter(
            getClassInfoByFullName, getGenericClassInfoByFullName, 
            getInterfaceInfoByFullName, getGenericInterfaceInfoByFullName
        );
        iter.forEach(type, f);
    }  

    bool getAllInferTypes(mlir::Type type, SmallVector<mlir_ts::InferType> &inferTypes)
    {
        MLIRTypeIteratorLogic iter(
            getClassInfoByFullName, getGenericClassInfoByFullName, 
            getInterfaceInfoByFullName, getGenericInterfaceInfoByFullName
        );
        return iter.every(type, [&](mlir::Type type) { 
            if (auto inferType = dyn_cast<mlir_ts::InferType>(type))
            {
                inferTypes.push_back(inferType);
            }

            return !!type; 
        });
    }        
    
    void mergeTypes(mlir::Location location, mlir::ArrayRef<mlir::Type> types, mlir::SmallVector<mlir::Type> &mergedTypes)
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
                auto resultType = mergeType(location, mergedType, typeItem, merged);
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

    mlir::Type arrayMergeType(mlir::Location location, mlir::Type existType, mlir::Type currentType, bool& merged)
    {
        LLVM_DEBUG(llvm::dbgs() << "\n!! merging existing type: " << existType << " with " << currentType << "\n";);

        if (existType == currentType)
        {
            merged = true;
            return existType;
        }

        // in case of array
        auto currentTypeArray = dyn_cast_or_null<mlir_ts::ArrayType>(currentType);
        auto existTypeArray = dyn_cast_or_null<mlir_ts::ArrayType>(existType);
        if (currentTypeArray && existTypeArray)
        {
            auto arrayElementMerged = mergeType(location, existTypeArray.getElementType(), currentTypeArray.getElementType(), merged);   
            return mlir_ts::ArrayType::get(arrayElementMerged);
        }

        return mergeType(location, existType, currentType, merged);
    }

    mlir::Type tupleMergeType(mlir::Location location, mlir_ts::TupleType existType, mlir_ts::TupleType currentType, bool& merged)
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
            auto mergedType = mergeType(location, existingIt->type, currentIt->type, merged);
            if (mergedType)
            {
                resultFields.push_back({ existingIt->id, mergedType, false, mlir_ts::AccessLevel::Public });
            }
        }

        merged = true;
        return mlir_ts::TupleType::get(context, resultFields);
    }

    mlir::Type mergeType(mlir::Location location, mlir::Type existType, mlir::Type currentType, bool& merged)
    {
        merged = false;
        LLVM_DEBUG(llvm::dbgs() << "\n!! merging existing \n\ttype: \t" << existType << "\n\twith \t" << currentType << "\n";);

        if (existType == currentType)
        {
            merged = true;
            return existType;
        }

        if (canCastFromTo(location, currentType, existType))
        {
            merged = true;
            return existType;
        }

        if (canCastFromTo(location, existType, currentType))
        {
            merged = true;
            return currentType;
        }

        // in case of merging integer types with sign/no sign
        auto mergedInts = false;
        auto resNewIntType = mergeIntTypes(existType, currentType, mergedInts);
        if (mergedInts)
        {
            merged = true;
            return resNewIntType;
        }

        // in case of merging function types
        auto mergedFuncs = false;
        auto resNewFuncType = mergeFuncTypes(existType, currentType, mergedFuncs);
        if (mergedFuncs)
        {
            merged = true;
            return resNewFuncType;
        }        
        
        // wide type - remove const & literal
        auto resType = wideStorageType(currentType);

        // check if can merge tuple types
        if (auto existingTupleType = dyn_cast<mlir_ts::TupleType>(existType))
        {
            if (auto currentTupleType = dyn_cast<mlir_ts::TupleType>(currentType))
            {
                auto tupleMerged = false;
                auto mergedTupleType = tupleMergeType(location, existingTupleType, currentTupleType, tupleMerged);
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

            if (isa<mlir_ts::UnionType>(existType))
            {
                defaultUnionType = getUnionTypeWithMerge(location, types);
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

protected:
    std::function<ClassInfo::TypePtr(StringRef)> getClassInfoByFullName;

    std::function<GenericClassInfo::TypePtr(StringRef)> getGenericClassInfoByFullName;

    std::function<InterfaceInfo::TypePtr(StringRef)> getInterfaceInfoByFullName;

    std::function<GenericInterfaceInfo::TypePtr(StringRef)> getGenericInterfaceInfoByFullName;
};

} // namespace typescript

#undef DEBUG_TYPE

#endif // MLIR_TYPESCRIPT_COMMONGENLOGIC_MLIRTYPEHELPER_H_
