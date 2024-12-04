#ifndef MLIR_TYPESCRIPT_MLIRGENLOGIC_MLIRCODELOGIC_H_
#define MLIR_TYPESCRIPT_MLIRGENLOGIC_MLIRCODELOGIC_H_

#include "TypeScript/TypeScriptDialect.h"
#include "TypeScript/TypeScriptOps.h"

#include "TypeScript/Defines.h"
#include "TypeScript/MLIRLogic/MLIRDefines.h"
#include "TypeScript/MLIRLogic/MLIRGenContext.h"
#include "TypeScript/MLIRLogic/MLIRTypeHelper.h"

#include "llvm/ADT/APSInt.h"
#include "llvm/Support/Debug.h"

#include "parser_types.h"

#include <numeric>

#define DEBUG_TYPE "mlir"

using namespace ::typescript;
using namespace ts;
namespace mlir_ts = mlir::typescript;

namespace typescript
{

class MLIRCodeLogic
{
    mlir::MLIRContext *context;
    mlir::OpBuilder builder;

  public:
    MLIRCodeLogic(mlir::MLIRContext *context) : context(context), builder(context)
    {
    }

    MLIRCodeLogic(mlir::OpBuilder builder) : context(builder.getContext()), builder(builder)
    {
    }

    mlir::StringAttr getStringAttrWith0(std::string value)
    {
        return mlir::StringAttr::get(context, StringRef(value.data(), value.length() + 1));
    }

    mlir::Attribute ExtractAttr(mlir::Value value)
    {
        LLVM_DEBUG(dbgs() << "\n!! ExtractAttr fron : " << value << "\n");

        if (auto constOp = value.getDefiningOp<mlir_ts::ConstantOp>())
        {
            auto val = constOp.getValueAttr();
            return val;
        }

        return mlir::Attribute();
    }

    mlir::Value GetReferenceFromValue(mlir::Location location, mlir::Value object)
    {
        MLIRTypeHelper mth(builder.getContext());
        if (auto refVal = mth.GetReferenceOfLoadOp(object))
        {
            return refVal;
        }        

        if (auto valueOp = object.getDefiningOp<mlir_ts::ValueOp>())
        {
            if (auto nestedRef = GetReferenceFromValue(location, valueOp.getIn()))
            {
                return builder.create<mlir_ts::PropertyRefOp>(
                    location, 
                    mlir_ts::RefType::get(valueOp.getType()), 
                    nestedRef, 
                    OPTIONAL_VALUE_INDEX);
            }
        }

        if (auto extractPropertyOp = object.getDefiningOp<mlir_ts::ExtractPropertyOp>())
        {
            if (auto nestedRef = GetReferenceFromValue(location, extractPropertyOp.getObject()))
            {
                return builder.create<mlir_ts::PropertyRefOp>(
                    location, 
                    mlir_ts::RefType::get(extractPropertyOp.getType()), 
                    nestedRef, 
                    extractPropertyOp.getPosition().front());
            }
        }

        return mlir::Value();
    }

    mlir::Type getEffectiveFunctionTypeForTupleField(mlir::Type elementType)
    {
#ifdef USE_BOUND_FUNCTION_FOR_OBJECTS
        if (auto boundFuncType = dyn_cast<mlir_ts::BoundFunctionType>(elementType))
        {
            return mlir_ts::FunctionType::get(context, boundFuncType.getInputs(), boundFuncType.getResults(), boundFuncType.isVarArg());
        }
#endif

        return elementType;
    }

    template <typename T>
    std::pair<int, mlir::Type> TupleFieldType(mlir::Location location, T tupleType, mlir::Attribute fieldId,
                                              bool indexAccess = false)
    {
        auto result = TupleFieldTypeNoError(tupleType, fieldId, indexAccess);
        if (result.first == -1)
        {
            errorNotFound(location, fieldId);
        }

        return result;
    }

    void errorNotFound(mlir::Location location, mlir::Attribute fieldId)
    {
        emitError(location, "Tuple member '") << fieldId << "' can't be found";
    }

    template <typename T>
    std::pair<int, int> TupleFieldGetterAndSetter(T tupleType, mlir::Attribute fieldId)
    {
        // try to find getter & setter
        if (auto strFieldName = dyn_cast<mlir::StringAttr>(fieldId))
        {
            auto getterIndex = -1;
            auto setterIndex = -1;
            for (auto [index, fldInfo] : enumerate(tupleType))
            {
                if (auto strAttr = dyn_cast_or_null<mlir::StringAttr>(fldInfo.id))
                {
                    auto str = strAttr.getValue();
                    auto isGetter = str.starts_with("get_");
                    auto isSetter = str.starts_with("set_");
                    if ((isGetter || isSetter) && str.ends_with(strFieldName) && str.size() == strFieldName.size() + 4/*'get_'.length*/)
                    {
                        // we found setter or getter;
                        if (isGetter)
                        {
                            getterIndex = index;
                        }
                        else if (isSetter)
                        {
                            setterIndex = index;
                        }
                    }
                }
            }

            return {getterIndex, setterIndex};
        }

        return {-1, -1};
    }

    template <typename T>
    std::pair<int, mlir::Type> TupleFieldTypeNoError(T tupleType, mlir::Attribute fieldId,
                                                     bool indexAccess = false)
    {
        auto fieldIndex = tupleType.getIndex(fieldId);
        auto notFound = fieldIndex < 0 || fieldIndex >= tupleType.size();
        if (indexAccess && notFound)
        {
            // try to resolve index
            auto intAttr = dyn_cast<mlir::IntegerAttr>(fieldId);
            if (intAttr)
            {
                fieldIndex = intAttr.getValue().getSExtValue();;
                notFound = fieldIndex < 0 || fieldIndex >= tupleType.size();
            }
        }

        if (notFound)
        {
            return std::make_pair<>(-1, mlir::Type());
        }

        // type
        auto elementType = tupleType.getType(fieldIndex);

        return std::make_pair(fieldIndex, elementType);
    }

    mlir::Type CaptureTypeStorage(llvm::StringMap<ts::VariableDeclarationDOM::TypePtr> &capturedVars)
    {
        SmallVector<mlir_ts::FieldInfo> fields;
        for (auto &varInfo : capturedVars)
        {
            auto &value = varInfo.getValue();
            fields.push_back(mlir_ts::FieldInfo{MLIRHelper::TupleFieldName(value->getName(), context),
                                                value->getReadWriteAccess() ? mlir_ts::RefType::get(value->getType())
                                                                            : value->getType()});
        }

        auto lambdaType = mlir_ts::TupleType::get(context, fields);
        return lambdaType;
    }

    mlir::Type CaptureType(llvm::StringMap<ts::VariableDeclarationDOM::TypePtr> &capturedVars)
    {
        return mlir_ts::RefType::get(CaptureTypeStorage(capturedVars));
    }
};

class MLIRCodeLogicHelper
{
    mlir::OpBuilder &builder;
    mlir::Location location;

  public:
    MLIRCodeLogicHelper(mlir::OpBuilder &builder, mlir::Location location) : builder(builder), location(location)
    {
    }

    mlir::Value conditionalExpression(mlir::Type type, mlir::Value condition,
                                      mlir::function_ref<mlir::Value(mlir::OpBuilder &, mlir::Location)> thenBuilder,
                                      mlir::function_ref<mlir::Value(mlir::OpBuilder &, mlir::Location)> elseBuilder)
    {
        // ts.if
        auto ifOp = builder.create<mlir_ts::IfOp>(location, type, condition, true);

        // then block
        auto &thenRegion = ifOp.getThenRegion();

        builder.setInsertionPointToStart(&thenRegion.back());

        mlir::Value value = thenBuilder(builder, location);
        builder.create<mlir_ts::ResultOp>(location, value);

        // else block
        auto &elseRegion = ifOp.getElseRegion();

        builder.setInsertionPointToStart(&elseRegion.back());

        mlir::Value elseValue = elseBuilder(builder, location);
        builder.create<mlir_ts::ResultOp>(location, elseValue);

        builder.setInsertionPointAfter(ifOp);

        return ifOp.getResults().front();
    }

    template <typename T>
    void seekLastOp(mlir::Block *block)
    {
        // find last string
        auto lastUse = [&](mlir::Operation *op) {
            if (auto globalOp = dyn_cast<T>(op))
            {
                builder.setInsertionPointAfter(globalOp);
            }
        };

        block->walk(lastUse);
    }    

    void seekLast(mlir::Block *block)
    {
        seekLastOp<mlir_ts::GlobalOp>(block);
    }
};

class MLIRCustomMethods
{
    mlir::OpBuilder &builder;
    mlir::Location &location;
    CompileOptions &compileOptions;

  public:
    MLIRCustomMethods(mlir::OpBuilder &builder, mlir::Location &location, CompileOptions &compileOptions) 
        : builder(builder), location(location), compileOptions(compileOptions)
    {
    }

    static bool isInternalObjectName (StringRef objectName)
    {
        static std::map<std::string, bool> o { {"Symbol", true}};
        return o[objectName.str()];    
    }

    static bool isInternalFunctionName (CompileOptions &compileOptions, StringRef functionName)
    {
        return compileOptions.enableBuiltins 
            ? isInternalFunctionNameBuiltin(functionName) 
            : isInternalFunctionNameNoBuiltin(functionName);
    }

    static bool isInternalFunctionNameBuiltin (StringRef functionName)
    {
        static std::map<std::string, bool> m { 
            {"print", true}, {"convertf", true}, {"assert", true}, {"parseInt", true}, {"parseFloat", true}, {"isNaN", true}, {"sizeof", true}, {GENERATOR_SWITCHSTATE, true}, 
            {"LoadLibraryPermanently", true}, { "SearchForAddressOfSymbol", true }, { "LoadReference", true }, { "ReferenceOf", true }};
        return m[functionName.str()];    
    }    

    static bool isInternalFunctionNameNoBuiltin (StringRef functionName)
    {
        static std::map<std::string, bool> m { 
            {"print", true}, {"convertf", true}, {"assert", true}, {"sizeof", true}, {GENERATOR_SWITCHSTATE, true}, 
            {"LoadLibraryPermanently", true}, { "SearchForAddressOfSymbol", true }, { "LoadReference", true }, { "ReferenceOf", true }};
        return m[functionName.str()];    
    }   

    ValueOrLogicalResult callMethod(StringRef functionName, mlir::SmallVector<mlir::Type> typeArgs, ArrayRef<mlir::Value> operands, std::function<ValueOrLogicalResult(mlir::Location, mlir::Type, mlir::Value, const GenContext &)> castFn, const GenContext &genContext)
    {
        if (functionName == "print")
        {
            // print - internal command;
            return mlirGenPrint(location, operands, castFn, genContext);
        }
        else if (functionName == "convertf")
        {
            // print - internal command;
            return mlirGenConvertF(location, operands, castFn, genContext);
        }
        else if (functionName == "assert")
        {
            // assert - internal command;
            return mlirGenAssert(location, operands);
        }
        else if (compileOptions.enableBuiltins && functionName == "parseInt")
        {
            // assert - internal command;
            return mlirGenParseInt(location, operands);
        }
        else if (compileOptions.enableBuiltins && functionName == "parseFloat")
        {
            return mlirGenParseFloat(location, operands);
        }
        else if (compileOptions.enableBuiltins && functionName == "isNaN")
        {
            return mlirGenIsNaN(location, operands);
        }
        else if (functionName == "sizeof")
        {
            return mlirGenSizeOf(location, typeArgs, operands);
        }
        else if (functionName == "__array_push")
        {
            return mlirGenArrayPush(location, operands);
        }
        else if (functionName == "__array_pop")
        {
            return mlirGenArrayPop(location, operands);
        }
        else if (functionName == "__array_unshift")
        {
            return mlirGenArrayUnshift(location, operands);
        }        
        else if (functionName == "__array_shift")
        {
            return mlirGenArrayShift(location, operands);
        }
        else if (functionName == "__array_splice")
        {
            return mlirGenArraySplice(location, operands);
        }        
        else if (functionName == "__array_view")
        {
            return mlirGenArrayView(location, operands);
        }
        else if (functionName == GENERATOR_SWITCHSTATE)
        {
            // switchstate - internal command;
            return mlirGenSwitchState(location, operands, genContext);
        }
        else if (functionName == "LoadLibraryPermanently")
        {
            return mlirGenLoadLibraryPermanently(location, operands);
        }
        else if (functionName == "SearchForAddressOfSymbol")
        {
            return mlirGenSearchForAddressOfSymbol(location, operands);
        }
        else if (functionName == "LoadReference")
        {
            return mlirGenLoadReference(location, operands);
        }
        else if (functionName == "ReferenceOf")
        {
            return mlirGenReferenceOf(location, operands);
        }
        else if (!genContext.allowPartialResolve)
        {
            emitError(location) << "no defined function found for '" << functionName << "'";
        }

        return mlir::failure();
    }

    mlir::LogicalResult mlirGenPrint(const mlir::Location &location, ArrayRef<mlir::Value> operands, std::function<ValueOrLogicalResult(mlir::Location, mlir::Type, mlir::Value, const GenContext &)> castFn, const GenContext &genContext)
    {
        SmallVector<mlir::Value> vals;
        for (auto &oper : operands)
        {
            if (!isa<mlir_ts::StringType>(oper.getType()))
            {
                auto strCast = castFn(location, mlir_ts::StringType::get(builder.getContext()), oper, genContext);
                if (!strCast)
                {
                    return mlir::failure();
                }

                vals.push_back(strCast);
            }
            else
            {
                vals.push_back(oper);
            }
        }

        auto printOp = builder.create<mlir_ts::PrintOp>(location, vals);

        return mlir::success();
    }

    ValueOrLogicalResult mlirGenConvertF(const mlir::Location &location, ArrayRef<mlir::Value> operands, std::function<ValueOrLogicalResult(mlir::Location, mlir::Type, mlir::Value, const GenContext &)> castFn, const GenContext &genContext)
    {
        mlir::Value bufferSize;
        mlir::Value format;

        if (operands.size() < 3) {
            return mlir::failure();
        }

        bufferSize = operands[0];
        if (!isa<mlir::IndexType>(bufferSize.getType()))
        {
            bufferSize = castFn(location, mlir::IndexType::get(builder.getContext()), bufferSize, genContext);
        }

        auto stringType = mlir_ts::StringType::get(builder.getContext());
        format = operands[1];
        if (!isa<mlir_ts::StringType>(format.getType()))
        {
            format = castFn(location, stringType, format, genContext);
        }

        SmallVector<mlir::Value> vals;
        for (auto &oper : operands.take_back(operands.size() - 2))
        {
            vals.push_back(oper);
        }

        auto convertFOp = builder.create<mlir_ts::ConvertFOp>(location, stringType, bufferSize, format, vals);

        return V(convertFOp);
    }

    mlir::LogicalResult mlirGenAssert(const mlir::Location &location, ArrayRef<mlir::Value> operands)
    {
        if (operands.size() == 0)
        {
            return mlir::failure();
        }

        auto msg = StringRef("assert");
        if (operands.size() > 1)
        {
            for (auto opIndex = 1; opIndex < operands.size(); opIndex++)
            {
                auto param2 = operands[opIndex];
                auto constantOp = dyn_cast<mlir_ts::ConstantOp>(param2.getDefiningOp());
                if (constantOp)
                {
                    auto type = constantOp.getType();
                    if (auto literalType = dyn_cast<mlir_ts::LiteralType>(type))
                    {
                        type = literalType.getElementType();
                    }

                    if (isa<mlir_ts::StringType>(type))
                    {
                        msg = cast<mlir::StringAttr>(constantOp.getValue()).getValue();
                    }
                }

                param2.getDefiningOp()->erase();
            }
        }

        auto op = operands.front();
        if (!isa<mlir_ts::BooleanType>(op.getType()))
        {
            op = builder.create<mlir_ts::CastOp>(location, mlir_ts::BooleanType::get(builder.getContext()), op);
        }

        auto assertOp =
            builder.create<mlir_ts::AssertOp>(location, op, mlir::StringAttr::get(builder.getContext(), msg));

        return mlir::success();
    }

    mlir::Value mlirGenParseInt(const mlir::Location &location, ArrayRef<mlir::Value> operands)
    {
        auto hasTwoOps = operands.size() == 2;
        auto op = operands.front();
        mlir::Value op2;

        if (!isa<mlir_ts::StringType>(op.getType()))
        {
            op = builder.create<mlir_ts::CastOp>(location, mlir_ts::StringType::get(builder.getContext()), op);
        }

        if (hasTwoOps)
        {
            op2 = operands[1];
            if (!isa<mlir::IntegerType>(op2.getType()))
            {
                op2 = builder.create<mlir_ts::CastOp>(location, mlir::IntegerType::get(builder.getContext(), 32), op2);
            }
        }

        auto parseIntOp = hasTwoOps ? builder.create<mlir_ts::ParseIntOp>(location, builder.getI32Type(), op, op2)
                                    : builder.create<mlir_ts::ParseIntOp>(location, builder.getI32Type(), op);

        return parseIntOp;
    }

    mlir::Value mlirGenParseFloat(const mlir::Location &location, ArrayRef<mlir::Value> operands)
    {
        auto op = operands.front();
        if (!isa<mlir_ts::StringType>(op.getType()))
        {
            op = builder.create<mlir_ts::CastOp>(location, mlir_ts::StringType::get(builder.getContext()), op);
        }

        auto parseFloatOp =
            builder.create<mlir_ts::ParseFloatOp>(location, mlir_ts::NumberType::get(builder.getContext()), op);

        return parseFloatOp;
    }

    mlir::Value mlirGenIsNaN(const mlir::Location &location, ArrayRef<mlir::Value> operands)
    {
        auto op = operands.front();
        if (!isa<mlir_ts::NumberType>(op.getType()))
        {
            op = builder.create<mlir_ts::CastOp>(location, mlir_ts::NumberType::get(builder.getContext()), op);
        }

        auto isNaNOp = builder.create<mlir_ts::IsNaNOp>(location, mlir_ts::BooleanType::get(builder.getContext()), op);

        return isNaNOp;
    }

    mlir::Value mlirGenSizeOf(const mlir::Location &location, mlir::SmallVector<mlir::Type> typeArgs, ArrayRef<mlir::Value> operands)
    {
        if (typeArgs.size() > 0)
        {
            return builder.create<mlir_ts::SizeOfOp>(location, builder.getIndexType(), mlir::TypeAttr::get(typeArgs.front()));
        }

        if (operands.size() > 0)
        {
            return builder.create<mlir_ts::SizeOfOp>(location, builder.getIndexType(), mlir::TypeAttr::get(operands.front().getType()));
        }

        return mlir::Value();
    }

    ValueOrLogicalResult mlirGenArrayPush(const mlir::Location &location, mlir::Value thisValue, ArrayRef<mlir::Value> values)
    {
        MLIRCodeLogic mcl(builder);

        auto arrayElement = cast<mlir_ts::ArrayType>(thisValue.getType()).getElementType();

        SmallVector<mlir::Value> castedValues;
        for (auto value : values)
        {
            if (value.getType() != arrayElement)
            {
                castedValues.push_back(builder.create<mlir_ts::CastOp>(location, arrayElement, value));
            }
            else
            {
                castedValues.push_back(value);
            }
        }

        auto thisValueLoaded = mcl.GetReferenceFromValue(location, thisValue);
        if (!thisValueLoaded)
        {
            emitError(location) << "Can't get reference of the array, ensure const array is not used";
            return mlir::failure();
        }

        mlir::Value sizeOfValue =
            builder.create<mlir_ts::ArrayPushOp>(location, builder.getI32Type(), thisValueLoaded, mlir::ValueRange{castedValues});

        return sizeOfValue;
    }    

    ValueOrLogicalResult mlirGenArrayPush(const mlir::Location &location, ArrayRef<mlir::Value> operands)
    {
        return mlirGenArrayPush(location, operands.front(), operands.slice(1));
    }

    ValueOrLogicalResult mlirGenArrayPop(const mlir::Location &location, ArrayRef<mlir::Value> operands)
    {
        MLIRCodeLogic mcl(builder);
        auto thisValue = mcl.GetReferenceFromValue(location, operands.front());
        if (!thisValue)
        {
            emitError(location) << "Can't get reference of the array, ensure const array is not used";
            return mlir::failure();
        }

        mlir::Value value = builder.create<mlir_ts::ArrayPopOp>(
            location, cast<mlir_ts::ArrayType>(operands.front().getType()).getElementType(), thisValue);

        return value;
    }

    ValueOrLogicalResult mlirGenArrayUnshift(const mlir::Location &location, mlir::Value thisValue, ArrayRef<mlir::Value> values)
    {
        MLIRCodeLogic mcl(builder);

        auto arrayElement = cast<mlir_ts::ArrayType>(thisValue.getType()).getElementType();

        SmallVector<mlir::Value> castedValues;
        for (auto value : values)
        {
            if (value.getType() != arrayElement)
            {
                castedValues.push_back(builder.create<mlir_ts::CastOp>(location, arrayElement, value));
            }
            else
            {
                castedValues.push_back(value);
            }
        }

        auto thisValueLoaded = mcl.GetReferenceFromValue(location, thisValue);
        if (!thisValueLoaded)
        {
            emitError(location) << "Can't get reference of the array, ensure const array is not used";
            return mlir::failure();
        }

        mlir::Value sizeOfValue =
            builder.create<mlir_ts::ArrayUnshiftOp>(location, builder.getI32Type(), thisValueLoaded, mlir::ValueRange{castedValues});

        return sizeOfValue;
    }    

    ValueOrLogicalResult mlirGenArrayUnshift(const mlir::Location &location, ArrayRef<mlir::Value> operands)
    {
        return mlirGenArrayUnshift(location, operands.front(), operands.slice(1));
    }

    ValueOrLogicalResult mlirGenArrayShift(const mlir::Location &location, ArrayRef<mlir::Value> operands)
    {
        MLIRCodeLogic mcl(builder);
        auto thisValue = mcl.GetReferenceFromValue(location, operands.front());
        if (!thisValue)
        {
            emitError(location) << "Can't get reference of the array, ensure const array is not used";
            return mlir::failure();
        }

        mlir::Value value = builder.create<mlir_ts::ArrayShiftOp>(
            location, cast<mlir_ts::ArrayType>(operands.front().getType()).getElementType(), thisValue);

        return value;
    }    

    ValueOrLogicalResult mlirGenArraySplice(const mlir::Location &location, mlir::Value thisValue, mlir::Value startValue, mlir::Value deleteCountValue, ArrayRef<mlir::Value> values)
    {
        MLIRCodeLogic mcl(builder);

        if (!isa<mlir::IndexType>(startValue.getType()))
        {
            startValue = builder.create<mlir_ts::CastOp>(location, mlir::IndexType::get(builder.getContext()), startValue);
        }

        if (!isa<mlir::IndexType>(deleteCountValue.getType()))
        {
            deleteCountValue = builder.create<mlir_ts::CastOp>(location, mlir::IndexType::get(builder.getContext()), deleteCountValue);
        }

        auto arrayElement = cast<mlir_ts::ArrayType>(thisValue.getType()).getElementType();

        SmallVector<mlir::Value> castedValues;
        for (auto value : values)
        {
            if (value.getType() != arrayElement)
            {
                castedValues.push_back(builder.create<mlir_ts::CastOp>(location, arrayElement, value));
            }
            else
            {
                castedValues.push_back(value);
            }
        }

        auto thisValueLoaded = mcl.GetReferenceFromValue(location, thisValue);
        if (!thisValueLoaded)
        {
            emitError(location) << "Can't get reference of the array, ensure const array is not used";
            return mlir::failure();
        }

        mlir::Value sizeOfValue =
            builder.create<mlir_ts::ArraySpliceOp>(location, builder.getI32Type(), thisValueLoaded, startValue, deleteCountValue, mlir::ValueRange{castedValues});

        return sizeOfValue;
    }    

    ValueOrLogicalResult mlirGenArraySplice(const mlir::Location &location, ArrayRef<mlir::Value> operands)
    {
        return mlirGenArraySplice(location, operands.front(), operands[1], operands[2], operands.slice(3));
    }    

    ValueOrLogicalResult mlirGenArrayView(const mlir::Location &location, mlir::Value thisValue, ArrayRef<mlir::Value> values)
    {
        MLIRCodeLogic mcl(builder);

        auto indexType = builder.getI32Type();

        SmallVector<mlir::Value> castedValues;
        for (auto value : values)
        {
            if (value.getType() != indexType)
            {
                castedValues.push_back(builder.create<mlir_ts::CastOp>(location, indexType, value));
            }
            else
            {
                castedValues.push_back(value);
            }
        }

        mlir::Value arrayViewValue =
            builder.create<mlir_ts::ArrayViewOp>(
                location, 
                cast<mlir_ts::ArrayType>(thisValue.getType()), 
                thisValue, 
                castedValues[0], 
                castedValues[1]);

        return arrayViewValue;
    }    

    ValueOrLogicalResult mlirGenArrayView(const mlir::Location &location, ArrayRef<mlir::Value> operands)
    {
        return mlirGenArrayView(location, operands.front(), operands.slice(1));
    }    

    mlir::LogicalResult mlirGenSwitchState(const mlir::Location &location, ArrayRef<mlir::Value> operands,
                                           const GenContext &genContext)
    {
        auto op = operands.front();

        auto int32Type = mlir::IntegerType::get(op.getType().getContext(), 32);
        if (op.getType() != int32Type)
        {
            op = builder.create<mlir_ts::CastOp>(location, int32Type, op);
        }

        auto switchStateOp =
            builder.create<mlir_ts::SwitchStateOp>(location, op, builder.getBlock(), mlir::BlockRange{});

        auto *block = builder.createBlock(builder.getBlock()->getParent());
        switchStateOp.setSuccessor(block, 0);

        const_cast<GenContext &>(genContext).allocateVarsOutsideOfOperation = true;
        const_cast<GenContext &>(genContext).currentOperation = switchStateOp;

        return mlir::success();
    }

    mlir::LogicalResult mlirGenLoadLibraryPermanently(const mlir::Location &location, ArrayRef<mlir::Value> operands)
    {
        mlir::Value fileNameValue;
        for (auto &oper : operands)
        {
            if (!isa<mlir_ts::StringType>(oper.getType()))
            {
                auto strCast =
                    builder.create<mlir_ts::CastOp>(location, mlir_ts::StringType::get(builder.getContext()), oper);
                fileNameValue = strCast;
            }
            else
            {
                fileNameValue = oper;
            }
        }

        if (!fileNameValue)
        {
            return mlir::failure();
        }

        auto loadLibraryPermanentlyOp = builder.create<mlir_ts::LoadLibraryPermanentlyOp>(location, mlir::IntegerType::get(builder.getContext(), 32), fileNameValue);

        return mlir::success();
    }   

    ValueOrLogicalResult mlirGenSearchForAddressOfSymbol(const mlir::Location &location, ArrayRef<mlir::Value> operands)
    {
        mlir::Value symbolNameValue;
        for (auto &oper : operands)
        {
            if (!isa<mlir_ts::StringType>(oper.getType()))
            {
                auto strCast =
                    builder.create<mlir_ts::CastOp>(location, mlir_ts::StringType::get(builder.getContext()), oper);
                symbolNameValue = strCast;
            }
            else
            {
                symbolNameValue = oper;
            }

            break;
        }

        if (!symbolNameValue)
        {
            return mlir::failure();
        }

        auto loadLibraryPermanentlyOp = builder.create<mlir_ts::SearchForAddressOfSymbolOp>(location, mlir_ts::OpaqueType::get(builder.getContext()), symbolNameValue);

        return V(loadLibraryPermanentlyOp);
    }     

    ValueOrLogicalResult mlirGenLoadReference(const mlir::Location &location, ArrayRef<mlir::Value> operands)
    {
        mlir::Value refValue;
        for (auto &oper : operands)
        {
            if (isa<mlir_ts::OpaqueType>(oper.getType()))
            {
                auto opaqueCast =
                    builder.create<mlir_ts::CastOp>(location, mlir_ts::RefType::get(builder.getContext(), oper.getType()), oper);
                refValue = opaqueCast;
            }
            else
            {
                refValue = oper;
            }

            break;
        }

        if (!refValue)
        {
            return mlir::failure();
        }

        auto loadedValue = builder.create<mlir_ts::LoadOp>(location, cast<mlir_ts::RefType>(refValue.getType()).getElementType(), refValue);
        return V(loadedValue);
    }

    ValueOrLogicalResult mlirGenReferenceOf(const mlir::Location &location, ArrayRef<mlir::Value> operands)
    {
        MLIRCodeLogic mcl(builder);
        auto refValue = mcl.GetReferenceFromValue(location, operands.front());        
        return V(refValue);
    }    
};

class MLIRPropertyAccessCodeLogic
{
    mlir::OpBuilder &builder;
    mlir::Location location;
    mlir::Value expression;
    mlir::StringRef name;
    mlir::Attribute fieldId;
    mlir::Value argument;

  public:
    MLIRPropertyAccessCodeLogic(mlir::OpBuilder &builder, mlir::Location location, mlir::Value expression,
                                StringRef name)
        : builder(builder), location(location), expression(expression), name(name)
    {
        fieldId = MLIRHelper::TupleFieldName(name, builder.getContext());
    }

    MLIRPropertyAccessCodeLogic(mlir::OpBuilder &builder, mlir::Location location, mlir::Value expression,
                                mlir::Attribute fieldId)
        : builder(builder), location(location), expression(expression), fieldId(fieldId)
    {
        if (auto strAttr = dyn_cast<mlir::StringAttr>(fieldId))
        {
            name = strAttr.getValue();
        }
    }

    MLIRPropertyAccessCodeLogic(mlir::OpBuilder &builder, mlir::Location location, mlir::Value expression,
                                mlir::Attribute fieldId, mlir::Value argument)
        : builder(builder), location(location), expression(expression), fieldId(fieldId), argument(argument)
    {
        if (auto strAttr = dyn_cast<mlir::StringAttr>(fieldId))
        {
            name = strAttr.getValue();
        }
    }    

    mlir::Value Enum(mlir_ts::EnumType enumType)
    {
        auto propName = getName();
        auto attrVal = getExprConstAttr();
        if (!attrVal)
        {
            return mlir::Value();
        }

        auto dictionaryAttr = mlir::cast<mlir::DictionaryAttr>(attrVal);
        auto valueAttr = dictionaryAttr.get(propName);
        if (!valueAttr)
        {
            emitError(location, "Enum member '") << propName << "' can't be found";
            return mlir::Value();
        }

        mlir::Type typeFromAttr;
        mlir::TypeSwitch<mlir::Attribute>(valueAttr)
            .Case<mlir::StringAttr>(
                [&](auto strAttr) { typeFromAttr = mlir_ts::StringType::get(builder.getContext()); })
            .Case<mlir::IntegerAttr>([&](auto intAttr) { typeFromAttr = intAttr.getType(); })
            .Case<mlir::FloatAttr>(
                [&](auto floatAttr) { typeFromAttr = mlir_ts::NumberType::get(builder.getContext()); })
            .Case<mlir::BoolAttr>(
                [&](auto boolAttr) { typeFromAttr = mlir_ts::BooleanType::get(builder.getContext()); })
            .Default([&](auto type) { llvm_unreachable("not implemented"); });

        LLVM_DEBUG(llvm::dbgs() << "\n!! enum: " << propName << " value attr: " << valueAttr << "\n");

        // return builder.create<mlir_ts::ConstantOp>(location, enumType.getElementType(), valueAttr);
        auto literalType = mlir_ts::LiteralType::get(valueAttr, typeFromAttr);
        return builder.create<mlir_ts::ConstantOp>(location, literalType, valueAttr);
    }

    template <typename T> mlir::Value Tuple(T tupleType, bool indexAccess = false)
    {
        mlir::Value value;

        MLIRTypeHelper mth(builder.getContext());
        MLIRCodeLogic mcl(builder);

        // resolve index
        auto pair = mcl.TupleFieldTypeNoError(tupleType, fieldId, indexAccess);
        auto fieldIndex = pair.first;
        if (fieldIndex < 0)
        {
            auto accessorValue = TupleGetSetAccessor(tupleType, fieldId);
            if (accessorValue)
            {
                return accessorValue;
            }

            mcl.errorNotFound(location, fieldId);
            return value;
        }

        bool isBoundRef = false;
        auto elementTypeForRef = pair.second;
        auto elementType = mth.isBoundReference(pair.second, isBoundRef);

        auto refValue = getExprLoadRefValue(location);
        if (isBoundRef && !refValue)
        {
            // allocate in stack
            refValue =
                builder.create<mlir_ts::VariableOp>(location, mlir_ts::RefType::get(expression.getType()), expression);
        }

        if (refValue)
        {
            auto refType = isBoundRef ? static_cast<mlir::Type>(mlir_ts::BoundRefType::get(elementTypeForRef))
                                      : static_cast<mlir::Type>(mlir_ts::RefType::get(elementTypeForRef));
            auto propRef = builder.create<mlir_ts::PropertyRefOp>(location, refType, refValue,
                                                                  builder.getI32IntegerAttr(fieldIndex));

            return builder.create<mlir_ts::LoadOp>(location, elementType, propRef);
        }

        return builder.create<mlir_ts::ExtractPropertyOp>(
            location, elementTypeForRef, expression, MLIRHelper::getStructIndex(builder, fieldIndex));
    }

    template <typename T> ValueOrLogicalResult TupleGetSetAccessor(T tupleType, mlir::Attribute fieldId) 
    {
        MLIRCodeLogic mcl(builder);

        // check if we have getter & setter
        auto [getterIndex, setterIndex] = mcl.TupleFieldGetterAndSetter(tupleType, fieldId);
        if (getterIndex >= 0 || setterIndex >= 0)
        {
            mlir::Type accessorResultType;
            mlir::Type getterFuncType;
            mlir::Type setterFuncType;
            if (getterIndex >= 0)
            {
                getterFuncType = tupleType.getType(getterIndex);
                if (auto funcType = dyn_cast<mlir_ts::FunctionType>(getterFuncType))
                {
                    if (funcType.getNumResults() > 0)
                    {
                        accessorResultType = funcType.getResult(0);
                    }
                }
            }

            if (setterIndex >= 0)
            {
                setterFuncType = tupleType.getType(setterIndex);
                if (!accessorResultType)
                {
                    if (auto funcType = dyn_cast<mlir_ts::FunctionType>(setterFuncType))
                    {
                        if (funcType.getNumInputs() > 1)
                        {
                            accessorResultType = funcType.getInput(1);
                        }
                    }
                }
            }

            if (!accessorResultType)
            {
                emitError(location) << "can't resolve type of property";
                return mlir::failure();
            }

            mlir::Value getterValue;
            mlir::Value setterValue;

            if (getterIndex >= 0)
            {
                getterValue = builder.create<mlir_ts::ExtractPropertyOp>(location, getterFuncType, expression, 
                    MLIRHelper::getStructIndex(builder, getterIndex));
            }
            else
            {
                getterValue = builder.create<mlir_ts::UndefOp>(location, 
                    mlir_ts::FunctionType::get(
                        builder.getContext(), 
                        {mlir_ts::OpaqueType::get(builder.getContext())}, 
                        {accessorResultType}, 
                        false));
            }

            if (setterIndex >= 0)
            {
                setterValue = builder.create<mlir_ts::ExtractPropertyOp>(location, setterFuncType, expression, 
                    MLIRHelper::getStructIndex(builder, setterIndex));
            }
            else
            {
                setterValue = builder.create<mlir_ts::UndefOp>(location, 
                    mlir_ts::FunctionType::get(
                        builder.getContext(), 
                        {mlir_ts::OpaqueType::get(builder.getContext()), 
                        accessorResultType}, 
                        {}, 
                        false));
            }

            auto refValue = getExprLoadRefValue(location);
            if (!refValue)
            {
                // allocate in stack
                refValue = builder.create<mlir_ts::VariableOp>(
                    location, mlir_ts::RefType::get(expression.getType()), expression);
            }                    

            auto thisValue = refValue;

            auto thisAccessorIndirectOp = builder.create<mlir_ts::ThisIndirectAccessorOp>(
                location, accessorResultType, thisValue, getterValue, setterValue, mlir::Value());  

            return thisAccessorIndirectOp.getResult(0);              
        }

        return mlir::Value();
    }

    template <typename T> mlir::Value TupleNoError(T tupleType, bool indexAccess = false)
    {
        mlir::Value value;

        MLIRTypeHelper mth(builder.getContext());
        MLIRCodeLogic mcl(builder);

        // resolve index
        auto pair = mcl.TupleFieldTypeNoError(tupleType, fieldId, indexAccess);
        auto fieldIndex = pair.first;
        if (fieldIndex < 0)
        {
            return value;
        }

        bool isBoundRef = false;
        auto elementTypeForRef = pair.second;
        auto elementType = mth.isBoundReference(pair.second, isBoundRef);

        auto refValue = getExprLoadRefValue(location);
        if (isBoundRef && !refValue)
        {
            // allocate in stack
            refValue =
                builder.create<mlir_ts::VariableOp>(location, mlir_ts::RefType::get(expression.getType()), expression);
        }

        if (refValue)
        {
            auto refType = isBoundRef ? static_cast<mlir::Type>(mlir_ts::BoundRefType::get(elementTypeForRef))
                                      : static_cast<mlir::Type>(mlir_ts::RefType::get(elementTypeForRef));
            auto propRef = builder.create<mlir_ts::PropertyRefOp>(location, refType, refValue,
                                                                  builder.getI32IntegerAttr(fieldIndex));

            return builder.create<mlir_ts::LoadOp>(location, elementType, propRef);
        }

        return builder.create<mlir_ts::ExtractPropertyOp>(
            location, elementTypeForRef, expression, MLIRHelper::getStructIndex(builder, fieldIndex));
    }

    mlir::Value Bool(mlir_ts::BooleanType intType)
    {
        auto propName = getName();
        if (propName == TO_STRING)
        {
            return builder.create<mlir_ts::CastOp>(location, mlir_ts::StringType::get(builder.getContext()),
                                                   expression);
        }

        return mlir::Value();
    }

    mlir::Value Int(mlir::IntegerType intType)
    {
        auto propName = getName();
        if (propName == TO_STRING)
        {
            return builder.create<mlir_ts::CastOp>(location, mlir_ts::StringType::get(builder.getContext()),
                                                   expression);
        }

        return mlir::Value();
    }

    mlir::Value Float(mlir::FloatType floatType)
    {
        auto propName = getName();
        if (propName == TO_STRING)
        {
            return builder.create<mlir_ts::CastOp>(location, mlir_ts::StringType::get(builder.getContext()),
                                                   expression);
        }

        return mlir::Value();
    }

    mlir::Value Number(mlir_ts::NumberType numberType)
    {
        auto propName = getName();
        if (propName == TO_STRING)
        {
            return builder.create<mlir_ts::CastOp>(location, mlir_ts::StringType::get(builder.getContext()),
                                                   expression);
        }

        return mlir::Value();
    }

    mlir::Value String(mlir_ts::StringType stringType)
    {
        LLVM_DEBUG(dbgs() << "\n!! string prop access for : " << expression << "\n");

        auto propName = getName();
        if (propName == LENGTH_FIELD_NAME)
        {
            auto effectiveVal = expression;
            if (auto castOp = expression.getDefiningOp<mlir_ts::CastOp>())
            {
                effectiveVal = castOp.getIn();
            }

            if (auto constOp = effectiveVal.getDefiningOp<mlir_ts::ConstantOp>())
            {
                auto length = cast<mlir::StringAttr>(constOp.getValueAttr()).getValue().size();
                return V(builder.create<mlir_ts::ConstantOp>(location, builder.getI32Type(), builder.getI32IntegerAttr(length)));
            }

            return builder.create<mlir_ts::StringLengthOp>(location, builder.getI32Type(), expression);
        }

        return mlir::Value();
    }

    bool isArrayCustomMethod(StringRef propName)
    {
        if (propName == "forEach") return true;
        if (propName == "every") return true;
        if (propName == "some") return true;
        if (propName == "map") return true;
        if (propName == "filter") return true;
        if (propName == "reduce") return true;
        return false;
    }

    bool isArrayCustomMethodReturnsBool(StringRef propName)
    {
        if (propName == "every") return true;
        if (propName == "some") return true;
        return false;
    }

    const char* getArrayCustomMethodName(StringRef propName)
    {
        if (propName == "forEach") return "__array_foreach";
        if (propName == "every") return "__array_every";
        if (propName == "some") return "__array_some";
        if (propName == "map") return "__array_map";
        if (propName == "filter") return "__array_filter";
        if (propName == "reduce") return "__array_reduce";
        return nullptr;
    }

    template <typename T> mlir::Value Array(T arrayType)
    {
        SmallVector<mlir::NamedAttribute> customAttrs;
        // customAttrs.push_back({MLIR_IDENT("__virtual"), MLIR_ATTR("true")});

        auto propName = getName();
        if (propName == LENGTH_FIELD_NAME)
        {
            if (auto constArrayType = dyn_cast<mlir_ts::ConstArrayType>(expression.getType()))
            {
                auto size = constArrayType.getSize();
                return builder.create<mlir_ts::ConstantOp>(location, builder.getI32Type(),
                            builder.getI32IntegerAttr(size));
            }
            else if (isa<mlir_ts::ArrayType>(expression.getType()))
            {
                return builder.create<mlir_ts::LengthOfOp>(location, builder.getI32Type(), expression);
            }

            return mlir::Value();
        }
        
        if (propName == "push" || propName == "pop" || propName == "unshift" || propName == "shift" || propName == "splice" || propName == "view")
        {
            if (isa<mlir_ts::ArrayType>(expression.getType()))
            {
                std::string name = "__array_";
                name += propName;

                auto symbOp = builder.create<mlir_ts::ThisSymbolRefOp>(
                    location, builder.getNoneType(), expression,
                    mlir::FlatSymbolRefAttr::get(builder.getContext(), name.c_str()));
                symbOp->setAttr(BUILTIN_FUNC_ATTR_NAME, mlir::BoolAttr::get(builder.getContext(), true));
                return symbOp;
            }

            return mlir::Value();
        }

        if (isArrayCustomMethod(propName))
        {
            auto arrayType = dyn_cast<mlir_ts::ArrayType>(expression.getType());
            auto constArrayType = dyn_cast<mlir_ts::ConstArrayType>(expression.getType());
            if (arrayType || constArrayType)
            {
                mlir::Type elementType;
                if (constArrayType)
                {
                    elementType = constArrayType.getElementType();

                    MLIRTypeHelper mth(builder.getContext());
                    auto nonConstArray = mth.convertConstArrayTypeToArrayType(expression.getType());
                    expression = builder.create<mlir_ts::CastOp>(location, nonConstArray, expression);
                }
                else
                {
                    elementType = arrayType.getElementType();
                }

                auto isReduce = propName == "reduce";
                SmallVector<mlir::Type> resultArgs;
                if (isArrayCustomMethodReturnsBool(propName))
                {
                    resultArgs.push_back(mlir_ts::BooleanType::get(builder.getContext()));
                }

                mlir::Type genericTypeT;
                SmallVector<mlir::Type> lambdaArgs{elementType};
                if (isReduce)
                {
                    // add sum param
                    genericTypeT = mlir_ts::NamedGenericType::get(builder.getContext(), mlir::FlatSymbolRefAttr::get(builder.getContext(), "T"));
                    lambdaArgs.insert(&lambdaArgs.front(), genericTypeT);
                }

                auto lambdaFuncType = mlir_ts::FunctionType::get(builder.getContext(), lambdaArgs, resultArgs);
                
                SmallVector<mlir::Type> funcArgs{lambdaFuncType};
                if (isReduce)
                {
                    funcArgs.push_back(genericTypeT);
                }

                auto funcType = mlir_ts::FunctionType::get(builder.getContext(), funcArgs, resultArgs);
                auto symbOp = builder.create<mlir_ts::ThisSymbolRefOp>(
                    location, funcType, expression,
                    mlir::FlatSymbolRefAttr::get(builder.getContext(), 
                    getArrayCustomMethodName(propName)));
                symbOp->setAttr(BUILTIN_FUNC_ATTR_NAME, mlir::BoolAttr::get(builder.getContext(), true));
                return symbOp;
            }

            return mlir::Value();
        }

        return mlir::Value();
    }

    template <typename T> mlir::Value Ref(T refType)
    {
        if (auto constTupleType = dyn_cast<mlir_ts::ConstTupleType>(refType.getElementType()))
        {
            return RefLogic(constTupleType);
        }
        else if (auto tupleType = dyn_cast<mlir_ts::TupleType>(refType.getElementType()))
        {
            return RefLogic(tupleType);
        }
        else
        {
            llvm_unreachable("not implemented");
        }
    }

    mlir::Value Object(mlir_ts::ObjectType objectType)
    {
        if (auto constTupleType = dyn_cast<mlir_ts::ConstTupleType>(objectType.getStorageType()))
        {
            return RefLogic(constTupleType);
        }
        else if (auto tupleType = dyn_cast<mlir_ts::TupleType>(objectType.getStorageType()))
        {
            return RefLogic(tupleType);
        }
        else if (auto objectStorageType = dyn_cast<mlir_ts::ObjectStorageType>(objectType.getStorageType()))
        {
            return RefLogic(objectStorageType);
        }        
        else
        {
            llvm_unreachable("not implemented");
        }
    }

    template <typename T> mlir::Value RefLogic(T tupleType)
    {
        MLIRTypeHelper mth(builder.getContext());
        MLIRCodeLogic mcl(builder);

        // resolve index
        auto pair = mcl.TupleFieldTypeNoError(tupleType, fieldId);
        auto fieldIndex = pair.first;
        if (fieldIndex < 0)
        {
            auto accessorValue = TupleGetSetAccessor(tupleType, fieldId);
            if (accessorValue)
            {
                return accessorValue;
            }

            mcl.errorNotFound(location, fieldId);
            return mlir::Value();
        }

        bool isBoundRef = false;
        auto elementTypeForRef = pair.second;
        auto elementType = mth.isBoundReference(pair.second, isBoundRef);

        auto refType = isBoundRef ? static_cast<mlir::Type>(mlir_ts::BoundRefType::get(elementTypeForRef))
                                  : static_cast<mlir::Type>(mlir_ts::RefType::get(elementTypeForRef));
        auto propRef = builder.create<mlir_ts::PropertyRefOp>(location, refType, expression,
                                                              builder.getI32IntegerAttr(fieldIndex));

        return builder.create<mlir_ts::LoadOp>(location, elementType, propRef);
    }

    mlir::Value Class(mlir_ts::ClassType classType)
    {
        if (auto classStorageType = dyn_cast<mlir_ts::ClassStorageType>(classType.getStorageType()))
        {
            return Class(classStorageType);
        }
        else
        {
            llvm_unreachable("not implemented");
        }
    }

    mlir::Value Class(mlir_ts::ClassStorageType classStorageType)
    {
        MLIRCodeLogic mcl(builder);

        // resolve index
        auto pair = mcl.TupleFieldTypeNoError(classStorageType, fieldId);
        auto fieldIndex = pair.first;
        if (fieldIndex < 0)
        {
            return mlir::Value();
        }

        bool isBoundRef = false;
        auto elementTypeForRef = pair.second;
        auto elementType = elementTypeForRef;
        /* as this is class, we do not take reference to class as for object */
        // auto elementType = mcl.isBoundReference(pair.second,
        // isBoundRef);

        auto refType = isBoundRef ? static_cast<mlir::Type>(mlir_ts::BoundRefType::get(elementTypeForRef))
                                  : static_cast<mlir::Type>(mlir_ts::RefType::get(elementTypeForRef));
        auto propRef = builder.create<mlir_ts::PropertyRefOp>(location, refType, expression,
                                                              builder.getI32IntegerAttr(fieldIndex));

        return builder.create<mlir_ts::LoadOp>(location, elementType, propRef);
    }

    mlir::Value Symbol(mlir_ts::SymbolType symbol)
    {
        auto symbOp = builder.create<mlir_ts::ConstantOp>(
            location, builder.getNoneType(), mlir::StringAttr::get(builder.getContext(), name));

        return symbOp;
    }

    StringRef getName()
    {
        return name;
    }

    mlir::Attribute getAttribute()
    {
        return fieldId;
    }

    mlir::Value getArgument()
    {
        return argument;
    }

  private:
    mlir::Attribute getExprConstAttr()
    {
        MLIRCodeLogic mcl(builder);
        return mcl.ExtractAttr(expression);
    }

    mlir::Value getExprLoadRefValue(mlir::Location location)
    {
        MLIRCodeLogic mcl(builder);
        auto value = mcl.GetReferenceFromValue(location, expression);
        return value;
    }
};

class MLIRLogicHelper
{
  public:
    static bool isNeededToSaveData(SyntaxKind &opCode)
    {
        switch (opCode)
        {
        case SyntaxKind::PlusEqualsToken:
            opCode = SyntaxKind::PlusToken;
            break;
        case SyntaxKind::MinusEqualsToken:
            opCode = SyntaxKind::MinusToken;
            break;
        case SyntaxKind::AsteriskEqualsToken:
            opCode = SyntaxKind::AsteriskToken;
            break;
        case SyntaxKind::AsteriskAsteriskEqualsToken:
            opCode = SyntaxKind::AsteriskAsteriskToken;
            break;
        case SyntaxKind::SlashEqualsToken:
            opCode = SyntaxKind::SlashToken;
            break;
        case SyntaxKind::PercentEqualsToken:
            opCode = SyntaxKind::PercentToken;
            break;
        case SyntaxKind::LessThanLessThanEqualsToken:
            opCode = SyntaxKind::LessThanLessThanToken;
            break;
        case SyntaxKind::GreaterThanGreaterThanEqualsToken:
            opCode = SyntaxKind::GreaterThanGreaterThanToken;
            break;
        case SyntaxKind::GreaterThanGreaterThanGreaterThanEqualsToken:
            opCode = SyntaxKind::GreaterThanGreaterThanGreaterThanToken;
            break;
        case SyntaxKind::AmpersandEqualsToken:
            opCode = SyntaxKind::AmpersandToken;
            break;
        case SyntaxKind::BarEqualsToken:
            opCode = SyntaxKind::BarToken;
            break;
        case SyntaxKind::BarBarEqualsToken:
            opCode = SyntaxKind::BarBarToken;
            break;
        case SyntaxKind::AmpersandAmpersandEqualsToken:
            opCode = SyntaxKind::AmpersandAmpersandToken;
            break;
        case SyntaxKind::QuestionQuestionEqualsToken:
            opCode = SyntaxKind::QuestionQuestionToken;
            break;
        case SyntaxKind::CaretEqualsToken:
            opCode = SyntaxKind::CaretToken;
            break;
        default:
            return false;
            break;
        }

        return true;
    }

    static bool isLogicOp(SyntaxKind opCode)
    {
        switch (opCode)
        {
        case SyntaxKind::EqualsEqualsToken:
        case SyntaxKind::EqualsEqualsEqualsToken:
        case SyntaxKind::ExclamationEqualsToken:
        case SyntaxKind::ExclamationEqualsEqualsToken:
        case SyntaxKind::GreaterThanToken:
        case SyntaxKind::GreaterThanEqualsToken:
        case SyntaxKind::LessThanToken:
        case SyntaxKind::LessThanEqualsToken:
            return true;
        }

        return false;
    }
};
} // namespace typescript

#undef DEBUG_TYPE

#endif // MLIR_TYPESCRIPT_MLIRGENLOGIC_MLIRCODELOGIC_H_
