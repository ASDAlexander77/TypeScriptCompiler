#ifndef MLIR_TYPESCRIPT_MLIRGENLOGIC_MLIRCODELOGIC_H_
#define MLIR_TYPESCRIPT_MLIRGENLOGIC_MLIRCODELOGIC_H_

#include "TypeScript/TypeScriptDialect.h"
#include "TypeScript/TypeScriptOps.h"

#include "TypeScript/Defines.h"
#include "TypeScript/MLIRLogic/MLIRDefines.h"
#include "TypeScript/MLIRLogic/MLIRGenContext.h"
#include "TypeScript/MLIRLogic/MLIROwnedReference.h"
#include "TypeScript/MLIRLogic/MLIRTypeHelper.h"

#include "llvm/ADT/APSInt.h"
#include "llvm/Support/Debug.h"

#include "TypeScript/MLIRLogic/MLIRGenContextDefines.h"

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
    CompileOptions& compileOptions;

  public:
    MLIRCodeLogic(mlir::MLIRContext *context, CompileOptions& compileOptions) : context(context), builder(context), compileOptions(compileOptions)
    {
    }

    MLIRCodeLogic(mlir::OpBuilder builder, CompileOptions& compileOptions) : context(builder.getContext()), builder(builder), compileOptions(compileOptions)
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


    mlir::IntegerAttr getIntegerAttr(mlir::Location location, mlir::Value oper, int width = 32, bool isSigned = false)
    {
        LLVM_DEBUG(llvm::dbgs() << "!! getIntAttr oper: " << oper << "'\n";);

        mlir::Attribute value;
        if (auto constantOp = oper.getDefiningOp<mlir_ts::ConstantOp>()) 
        {
            value = constantOp.getValue();
        }        
        else if (auto literalType = dyn_cast<mlir_ts::LiteralType>(oper.getType()))
        {
            value = literalType.getValue();
        }

        if (auto intAttr = dyn_cast<mlir::IntegerAttr>(value)) 
        {
            if (intAttr.getType().isInteger(width))
            {
                if (isSigned)
                {
                    return mlir::IntegerAttr::get(
                        mlir::IntegerType::get(intAttr.getContext(), width, mlir::IntegerType::SignednessSemantics::Signed), 
                        intAttr.getValue().getSExtValue());
                }
                else
                {
                    return mlir::IntegerAttr::get(
                        mlir::IntegerType::get(intAttr.getContext(), width, mlir::IntegerType::SignednessSemantics::Signless), 
                        intAttr.getValue().getZExtValue());
                }
            }            
        }        

        emitError(location) << "Must be constant integer(" << width << ")";
        return mlir::IntegerAttr();
    }

    mlir::StringAttr getStringAttr(mlir::Location location, mlir::Value oper)
    {
        mlir::Attribute value;
        if (auto constantOp = oper.getDefiningOp<mlir_ts::ConstantOp>()) 
        {
            value = constantOp.getValue();
        }        
        else if (auto literalType = dyn_cast<mlir_ts::LiteralType>(oper.getType()))
        {
            value = literalType.getValue();       
        }

        if (auto strAttr = dyn_cast<mlir::StringAttr>(value)) 
        {
            return strAttr;
        }      

        emitError(location) << "Must be constant string";
        return mlir::StringAttr();
    }

    mlir::Value GetReferenceFromValue(mlir::Location location, mlir::Value object)
    {
        MLIRTypeHelper mth(builder.getContext(), compileOptions);
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

        // A @value value seen as its storage tuple (MLIRGenImpl::valueStorage) read out of a place
        // has that place, seen as the tuple's: its fields are read in place, as a tuple's are, which
        // is what tells a read of a field from a value of its own (readsField, readsLiteralSlot).
        if (auto castOp = object.getDefiningOp<mlir_ts::CastOp>())
        {
            auto valueType = dyn_cast<mlir_ts::ValueType>(castOp.getIn().getType());
            if (valueType && valueType.getStorageType() == castOp.getType())
            {
                auto nestedRef = GetReferenceFromValue(location, castOp.getIn());
                if (nestedRef && isa<mlir_ts::RefType>(nestedRef.getType()))
                {
                    return builder.create<mlir_ts::CastOp>(location, mlir_ts::RefType::get(castOp.getType()), nestedRef);
                }
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
    std::tuple<int, mlir::Type, mlir_ts::AccessLevel> TupleFieldTypeNoError(T tupleType, mlir::Attribute fieldId,
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
            // `t["1"]`, as `for (const k in t)` gives it: the position of a field that has no name
            else if (auto strAttr = dyn_cast<mlir::StringAttr>(fieldId))
            {
                unsigned position;
                if (!strAttr.getValue().getAsInteger(10, position) && position < tupleType.size()
                    && !isa_and_nonnull<mlir::StringAttr>(tupleType.getFieldInfo(position).id))
                {
                    fieldIndex = position;
                    notFound = false;
                }
            }
        }

        if (notFound)
        {
            return {-1, mlir::Type(), mlir_ts::AccessLevel::Public};
        }

        // type
        auto elementType = tupleType.getType(fieldIndex);

        // access level
        auto accessLevel = tupleType.getAccessLevel(fieldIndex);

        return {fieldIndex, elementType, accessLevel};
    }

    mlir::Type CaptureTypeStorage(llvm::StringMap<ts::VariableDeclarationDOM::TypePtr> &capturedVars)
    {
        SmallVector<mlir_ts::FieldInfo> fields; 
        for (auto &varInfo : capturedVars)
        {
            auto &value = varInfo.getValue();
            fields.push_back(
                mlir_ts::FieldInfo{
                    MLIRHelper::TupleFieldName(value->getName(), context),
                    value->getCaptureByReference()
                        ? mlir_ts::RefType::get(value->getType())
                        : value->getType(),
                    false,
                    mlir_ts::AccessLevel::Public
                });
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
    CompileOptions& compileOptions;

  public:
    MLIRCodeLogicHelper(mlir::OpBuilder &builder, mlir::Location location, CompileOptions& compileOptions) 
        : builder(builder), location(location), compileOptions(compileOptions)
    {
    }

    ValueOrLogicalResult conditionalValue(mlir::Value condValue, 
        std::function<ValueOrLogicalResult()> trueValue, 
        std::function<ValueOrLogicalResult(mlir::Type trueValueType)> falseValue)
    {
        // type will be set later
        auto ifOp = builder.create<mlir_ts::IfOp>(location, builder.getNoneType(), condValue, true);

        // A branch that fails leaves nothing behind: not the half-built if (an empty region, or
        // a none-typed result), nor an insertion point inside it. Returning with both in place
        // turned the failure into bad IR further on - "'ts.If' op expects a non-empty block", or
        // code that silently took the other branch - whenever the caller carried on.
        auto abandon = [&]() {
            builder.setInsertionPoint(ifOp);
            ifOp.erase();
            return mlir::failure();
        };

        builder.setInsertionPointToStart(&ifOp.getThenRegion().front());

        // value if true
        auto trueResult = trueValue();
        if (trueResult.failed_or_no_value())
        {
            return abandon();
        }

        ifOp.getResults().front().setType(trueResult.value.getType());
        builder.create<mlir_ts::ResultOp>(location, mlir::ValueRange{trueResult});

        // else
        builder.setInsertionPointToStart(&ifOp.getElseRegion().front());

        // value if false
        auto falseResult = falseValue(trueResult.value.getType());
        if (falseResult.failed_or_no_value())
        {
            return abandon();
        }

        builder.create<mlir_ts::ResultOp>(location, mlir::ValueRange{falseResult});

        builder.setInsertionPointAfter(ifOp);

        return ValueOrLogicalResult(ifOp.getResults().front());        
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
            {"LoadLibraryPermanently", true}, { "SearchForAddressOfSymbol", true }, { "LoadReference", true }, { "ReferenceOf", true }, { "Deref", true }, { "Ref", true },
            {"atomicrmw", true}, {"cmpxchg", true}, {"atomicload", true}, {"atomicstore", true}, {"fence", true}, {"inline_asm", true}, {"call_intrinsic", true}, {"linker_options", true}
        };
        return m[functionName.str()];
    }

    static bool isInternalFunctionNameNoBuiltin (StringRef functionName)
    {
        static std::map<std::string, bool> m {
            {"print", true}, {"convertf", true}, {"assert", true}, {"sizeof", true}, {GENERATOR_SWITCHSTATE, true},
            {"LoadLibraryPermanently", true}, { "SearchForAddressOfSymbol", true }, { "LoadReference", true }, { "ReferenceOf", true }, { "Deref", true }, { "Ref", true },
            {"atomicrmw", true}, {"cmpxchg", true}, {"atomicload", true}, {"atomicstore", true}, {"fence", true}, {"inline_asm", true}, {"call_intrinsic", true}, {"linker_options", true}
        };
        return m[functionName.str()];    
    }   

    ValueOrLogicalResult callMethod(StringRef functionName, mlir::SmallVector<mlir::Type> typeArgs, ArrayRef<mlir::Value> operands, std::function<ValueOrLogicalResult(mlir::Location, mlir::Type, mlir::Value, const GenContext &, bool)> castFn, const GenContext &genContext)
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
            return mlirGenAssert(location, operands, castFn, genContext);
        }
        else if (compileOptions.enableBuiltins && functionName == "parseInt")
        {
            // assert - internal command;
            return mlirGenParseInt(location, operands, castFn, genContext);
        }
        else if (compileOptions.enableBuiltins && functionName == "parseFloat")
        {
            return mlirGenParseFloat(location, operands, castFn, genContext);
        }
        else if (compileOptions.enableBuiltins && functionName == "isNaN")
        {
            return mlirGenIsNaN(location, operands, castFn, genContext);
        }
        else if (functionName == "sizeof")
        {
            return mlirGenSizeOf(location, typeArgs, operands);
        }
        else if (functionName == "__array_push")
        {
            return mlirGenArrayPush(location, operands, castFn, genContext);
        }
        else if (functionName == "__array_pop")
        {
            return mlirGenArrayPop(location, operands);
        }
        else if (functionName == "__array_unshift")
        {
            return mlirGenArrayUnshift(location, operands, castFn, genContext);
        }        
        else if (functionName == "__array_shift")
        {
            return mlirGenArrayShift(location, operands);
        }
        else if (functionName == "__array_splice")
        {
            return mlirGenArraySplice(location, operands, castFn, genContext);
        }        
        else if (functionName == "__array_view")
        {
            return mlirGenArrayView(location, operands, castFn, genContext);
        }
        else if (functionName == GENERATOR_SWITCHSTATE)
        {
            // switchstate - internal command;
            return mlirGenSwitchState(location, operands, castFn, genContext);
        }
        else if (functionName == "LoadLibraryPermanently")
        {
            return mlirGenLoadLibraryPermanently(location, operands, castFn, genContext);
        }
        else if (functionName == "SearchForAddressOfSymbol")
        {
            return mlirGenSearchForAddressOfSymbol(location, operands, castFn, genContext);
        }
        else if (functionName == "LoadReference" || functionName == "Deref")
        {
            return mlirGenLoadReference(location, operands, castFn, genContext);
        }
        else if (functionName == "ReferenceOf" || functionName == "Ref")
        {
            return mlirGenReferenceOf(location, operands);
        }
        else if (functionName == "atomicrmw")
        {
            return mlirGenAtomicRMW(location, operands, castFn, genContext);
        }
        else if (functionName == "cmpxchg")
        {
            return mlirGenCmpXchg(location, operands, castFn, genContext);
        }
        else if (functionName == "atomicload")
        {
            return mlirGenAtomicLoad(location, operands);
        }
        else if (functionName == "atomicstore")
        {
            return mlirGenAtomicStore(location, operands, castFn, genContext);
        }
        else if (functionName == "fence")
        {
            return mlirGenFence(location, operands);
        }        
        else if (functionName == "inline_asm")
        {
            return mlirGenInlineAsm(location, typeArgs, operands);
        }        
        else if (functionName == "call_intrinsic")
        {
            return mlirGenCallIntrinsic(location, typeArgs, operands);
        }        
        else if (functionName == "linker_options")
        {
            return mlirGenLinkerOptions(location, operands);
        }        
        else if (!genContext.allowPartialResolve)
        {
            emitError(location) << "no defined function found for '" << functionName << "'";
        }

        return mlir::failure();
    }

    mlir::LogicalResult mlirGenPrint(const mlir::Location &location, ArrayRef<mlir::Value> operands, std::function<ValueOrLogicalResult(mlir::Location, mlir::Type, mlir::Value, const GenContext &, bool)> castFn, const GenContext &genContext)
    {
        SmallVector<mlir::Value> vals;
        for (auto &oper : operands)
        {
            if (!isa<mlir_ts::StringType>(oper.getType()))
            {
                auto strCast = castFn(location, mlir_ts::StringType::get(builder.getContext()), oper, genContext, false);
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

    ValueOrLogicalResult mlirGenConvertF(const mlir::Location &location, ArrayRef<mlir::Value> operands, 
        std::function<ValueOrLogicalResult(mlir::Location, mlir::Type, mlir::Value, const GenContext &, bool)> castFn, const GenContext &genContext)
    {
        mlir::Value bufferSize;
        mlir::Value format;

        if (operands.size() < 3) {
            return mlir::failure();
        }

        bufferSize = operands[0];
        if (!isa<mlir::IndexType>(bufferSize.getType()))
        {
            bufferSize = castFn(location, mlir::IndexType::get(builder.getContext()), bufferSize, genContext, false);
            if (!bufferSize)
            {
                return mlir::failure();
            }
        }

        auto stringType = mlir_ts::StringType::get(builder.getContext());
        format = operands[1];
        if (!isa<mlir_ts::StringType>(format.getType()))
        {
            format = castFn(location, stringType, format, genContext, false);
            if (!format)
            {
                return mlir::failure();
            }
        }

        SmallVector<mlir::Value> vals;
        for (auto &oper : operands.take_back(operands.size() - 2))
        {
            vals.push_back(oper);
        }

        auto convertFOp = builder.create<mlir_ts::ConvertFOp>(location, stringType, bufferSize, format, vals);

        return V(convertFOp);
    }

    mlir::LogicalResult mlirGenAssert(const mlir::Location &location, ArrayRef<mlir::Value> operands,
        std::function<ValueOrLogicalResult(mlir::Location, mlir::Type, mlir::Value, const GenContext &, bool)> castFn, const GenContext &genContext)
    {
        if (operands.size() == 0)
        {
            return mlir::failure();
        }

        // The message: a constant string is kept as text, anything else - `what + " failed"` - is
        // shown as it is at run time. Such a message used to be erased as if it were a constant,
        // with its other uses still there ("operation's operand is unlinked").
        auto msg = StringRef("assert");
        mlir::Value message;
        if (operands.size() > 1)
        {
            auto param2 = operands[1];
            auto constantOp = param2.getDefiningOp<mlir_ts::ConstantOp>();
            auto type = constantOp ? constantOp.getType() : mlir::Type();
            if (auto literalType = dyn_cast_or_null<mlir_ts::LiteralType>(type))
            {
                type = literalType.getElementType();
            }

            if (constantOp && isa<mlir_ts::StringType>(type))
            {
                msg = cast<mlir::StringAttr>(constantOp.getValue()).getValue();
            }
            else
            {
                message = param2;
                if (!isa<mlir_ts::StringType>(message.getType()))
                {
                    message = castFn(location, mlir_ts::StringType::get(builder.getContext()), message, genContext, false);
                    if (!message)
                    {
                        return mlir::failure();
                    }
                }
            }
        }

        auto op = operands.front();
        if (!isa<mlir_ts::BooleanType>(op.getType()))
        {
            op = castFn(location, mlir_ts::BooleanType::get(builder.getContext()), op, genContext, false);
            if (!op)
            {
                return mlir::failure();
            }
        }

        auto assertOp =
            builder.create<mlir_ts::AssertOp>(location, op, message, mlir::StringAttr::get(builder.getContext(), msg));

        return mlir::success();
    }

    ValueOrLogicalResult mlirGenParseInt(const mlir::Location &location, ArrayRef<mlir::Value> operands,
        std::function<ValueOrLogicalResult(mlir::Location, mlir::Type, mlir::Value, const GenContext &, bool)> castFn, const GenContext &genContext)
    {
        auto hasTwoOps = operands.size() == 2;
        auto op = operands.front();
        mlir::Value op2;

        if (!isa<mlir_ts::StringType>(op.getType()))
        {
            op = castFn(location, mlir_ts::StringType::get(builder.getContext()), op, genContext, false);
            if (!op)
            {
                return mlir::failure();
            }
        }

        if (hasTwoOps)
        {
            op2 = operands[1];
            if (!isa<mlir::IntegerType>(op2.getType()))
            {
                op2 = castFn(location, mlir::IntegerType::get(builder.getContext(), 32), op2, genContext, false);
                if (!op2)
                {
                    return mlir::failure();
                }
            }
        }

        auto parseIntOp = hasTwoOps ? builder.create<mlir_ts::ParseIntOp>(location, builder.getI32Type(), op, op2)
                                    : builder.create<mlir_ts::ParseIntOp>(location, builder.getI32Type(), op);

        return V(parseIntOp);
    }

    ValueOrLogicalResult mlirGenParseFloat(const mlir::Location &location, ArrayRef<mlir::Value> operands,
        std::function<ValueOrLogicalResult(mlir::Location, mlir::Type, mlir::Value, const GenContext &, bool)> castFn, const GenContext &genContext)
    {
        auto op = operands.front();
        if (!isa<mlir_ts::StringType>(op.getType()))
        {
            op = castFn(location, mlir_ts::StringType::get(builder.getContext()), op, genContext, false);
            if (!op)
            {
                return mlir::failure();
            }
        }

        auto parseFloatOp =
            builder.create<mlir_ts::ParseFloatOp>(location, mlir_ts::NumberType::get(builder.getContext()), op);

        return V(parseFloatOp);
    }

    ValueOrLogicalResult mlirGenIsNaN(const mlir::Location &location, ArrayRef<mlir::Value> operands,
        std::function<ValueOrLogicalResult(mlir::Location, mlir::Type, mlir::Value, const GenContext &, bool)> castFn, const GenContext &genContext)
    {
        auto op = operands.front();
        if (!isa<mlir_ts::NumberType>(op.getType()))
        {
            op = castFn(location, mlir_ts::NumberType::get(builder.getContext()), op, genContext, false);
            if (!op)
            {
                return mlir::failure();
            }
        }

        auto isNaNOp = builder.create<mlir_ts::IsNaNOp>(location, mlir_ts::BooleanType::get(builder.getContext()), op);

        return V(isNaNOp);
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

    // Takes a reference to each value about to be handed to an array's data block. That block
    // releases every element it holds when it dies, so it has to have taken one - the same debt
    // an array literal carries (§9.21 in docs/reference-counting-evaluation.md). These ops fill
    // the block through their own lowering rather than through an assignment, so nothing else on
    // the path takes it.
    //
    // The ops that take an element back out - pop and shift - need no counterpart here, and that
    // is not an omission. The block simply stops holding it: the size shrinks past the slot, so
    // the release routine never reaches it, and the reference the block held transfers to the
    // returned value. That leaves the result carrying the same "+1 nobody has consumed" every
    // freshly produced value already carries, which is removed with the rest of the slack rather
    // than one op at a time.
    //
    // What splice deletes is a different matter: those elements are memmoved over and their
    // references dropped without a release. That leaks rather than over-releases, so it waits -
    // and it cannot be fixed here anyway, because the count to release is only known inside the
    // lowering.
    // Records that this value already carries a reference for whoever receives it. Only used
    // where the transfer is a property of the operation itself - see OWNED_RESULT_ATTR_NAME.
    void markResultOwned(mlir::Value value)
    {
        MLIRTypeHelper mth(builder.getContext(), compileOptions);
        if (!value || !mth.ownsHeapMemory(location, value.getType()))
        {
            return;
        }

        if (auto *definingOp = value.getDefiningOp())
        {
            definingOp->setAttr(OWNED_RESULT_ATTR_NAME, builder.getUnitAttr());
        }
    }

    void retainInsertedElements(ArrayRef<mlir::Value> values)
    {
        MLIRTypeHelper mth(builder.getContext(), compileOptions);
        for (auto value : values)
        {
            if (!value || !mth.ownsHeapMemory(location, value.getType()))
            {
                continue;
            }

            // `arr.push(new C())` arrives already owned (§9.25) - the data block takes that
            // reference over rather than adding one of its own.
            //
            // Saying so is not optional. This was the one receiving site that skipped its retain
            // without recording the consumption, and once §9.30 began releasing what nothing
            // consumed, the pushed value was released at the end of the pushing block - freeing
            // an element the array still held. Invisible for as long as the block that pushes is
            // also the block that reads, which is why §9.30's own tests missed it;
            // `function add() { store.push(new C()) }` reads the freed block.
            //
            // There is only ever ONE reference to take over, so the second array to receive the
            // same value has to retain like any other holder. Reading `OWNED_RESULT` without also
            // asking whether it had already been consumed made both of these consume it:
            //
            //     const keep = new Box("kept");   // a const with no storage of its own, so both
            //     victim.push(keep);              // pushes see the `new` itself
            //     survivors.push(keep);
            //
            // and the value ended up with two holders and one reference, so whichever array died
            // first freed an element the other still held. Nothing to do with `splice` - it
            // reproduces with no splice anywhere - but §9.74's release is what made it visible,
            // because until then an array that outlived its elements never gave them back.
            //
            // And the same one array receiving it again in a loop, from a `const` made once
            // before the loop, is the same mistake repeated: mayTakeOverReference turns away a
            // folded `const` for that reason (see OWNED_RESULT_NAMED_ATTR_NAME).
            if (mayTakeOverReference(value))
            {
                value.getDefiningOp()->setAttr(OWNED_RESULT_CONSUMED_ATTR_NAME, builder.getUnitAttr());
                continue;
            }

            builder.create<mlir_ts::RetainOp>(location, value);
        }
    }

    // Each value cast to `type`, into `castedValues`. A value the cast rejects fails the builtin:
    // its null result went on to the op's builder, which crashed the compiler (#438).
    mlir::LogicalResult castEach(const mlir::Location &location, mlir::Type type, ArrayRef<mlir::Value> values,
        SmallVector<mlir::Value> &castedValues,
        std::function<ValueOrLogicalResult(mlir::Location, mlir::Type, mlir::Value, const GenContext &, bool)> castFn, const GenContext &genContext)
    {
        for (auto value : values)
        {
            if (value.getType() == type)
            {
                castedValues.push_back(value);
                continue;
            }

            auto casted = castFn(location, type, value, genContext, false);
            if (casted.failed_or_no_value())
            {
                return mlir::failure();
            }

            castedValues.push_back(V(casted));
        }

        return mlir::success();
    }

    ValueOrLogicalResult mlirGenArrayPush(const mlir::Location &location, mlir::Value thisValue, ArrayRef<mlir::Value> values,
        std::function<ValueOrLogicalResult(mlir::Location, mlir::Type, mlir::Value, const GenContext &, bool)> castFn, const GenContext &genContext)
    {
        MLIRCodeLogic mcl(builder, compileOptions);

        auto arrayElement = cast<mlir_ts::ArrayType>(thisValue.getType()).getElementType();

        SmallVector<mlir::Value> castedValues;
        if (mlir::failed(castEach(location, arrayElement, values, castedValues, castFn, genContext)))
        {
            return mlir::failure();
        }

        auto thisValueLoaded = mcl.GetReferenceFromValue(location, thisValue);
        if (!thisValueLoaded)
        {
            emitError(location) << "Can't get reference of the array, ensure const array is not used";
            return mlir::failure();
        }

        retainInsertedElements(castedValues);

        mlir::Value sizeOfValue =
            builder.create<mlir_ts::ArrayPushOp>(location, builder.getIndexType(), thisValueLoaded, mlir::ValueRange{castedValues});

        return sizeOfValue;
    }    

    ValueOrLogicalResult mlirGenArrayPush(const mlir::Location &location, ArrayRef<mlir::Value> operands,
        std::function<ValueOrLogicalResult(mlir::Location, mlir::Type, mlir::Value, const GenContext &, bool)> castFn, const GenContext &genContext)
    {
        return mlirGenArrayPush(location, operands.front(), operands.slice(1), castFn, genContext);
    }

    ValueOrLogicalResult mlirGenArrayPop(const mlir::Location &location, ArrayRef<mlir::Value> operands)
    {
        MLIRCodeLogic mcl(builder, compileOptions);
        auto thisValue = mcl.GetReferenceFromValue(location, operands.front());
        if (!thisValue)
        {
            emitError(location) << "Can't get reference of the array, ensure const array is not used";
            return mlir::failure();
        }

        mlir::Value value = builder.create<mlir_ts::ArrayPopOp>(
            location, cast<mlir_ts::ArrayType>(operands.front().getType()).getElementType(), thisValue);

        // The data block gives up the element without releasing it - the size shrinks past the
        // slot, so its release routine never reaches it again - which hands the block's own
        // reference to whoever receives the result. Saying so lets that receiver take it over
        // rather than add one of its own (§9.26).
        markResultOwned(value);

        return value;
    }

    ValueOrLogicalResult mlirGenArrayUnshift(const mlir::Location &location, mlir::Value thisValue, ArrayRef<mlir::Value> values,
        std::function<ValueOrLogicalResult(mlir::Location, mlir::Type, mlir::Value, const GenContext &, bool)> castFn, const GenContext &genContext)
    {
        MLIRCodeLogic mcl(builder, compileOptions);

        auto arrayElement = cast<mlir_ts::ArrayType>(thisValue.getType()).getElementType();

        SmallVector<mlir::Value> castedValues;
        if (mlir::failed(castEach(location, arrayElement, values, castedValues, castFn, genContext)))
        {
            return mlir::failure();
        }

        auto thisValueLoaded = mcl.GetReferenceFromValue(location, thisValue);
        if (!thisValueLoaded)
        {
            emitError(location) << "Can't get reference of the array, ensure const array is not used";
            return mlir::failure();
        }

        retainInsertedElements(castedValues);

        mlir::Value sizeOfValue =
            builder.create<mlir_ts::ArrayUnshiftOp>(location, builder.getIndexType(), thisValueLoaded, mlir::ValueRange{castedValues});

        return sizeOfValue;
    }    

    ValueOrLogicalResult mlirGenArrayUnshift(const mlir::Location &location, ArrayRef<mlir::Value> operands,
        std::function<ValueOrLogicalResult(mlir::Location, mlir::Type, mlir::Value, const GenContext &, bool)> castFn, const GenContext &genContext)
    {
        return mlirGenArrayUnshift(location, operands.front(), operands.slice(1), castFn, genContext);
    }

    ValueOrLogicalResult mlirGenArrayShift(const mlir::Location &location, ArrayRef<mlir::Value> operands)
    {
        MLIRCodeLogic mcl(builder, compileOptions);
        auto thisValue = mcl.GetReferenceFromValue(location, operands.front());
        if (!thisValue)
        {
            emitError(location) << "Can't get reference of the array, ensure const array is not used";
            return mlir::failure();
        }

        mlir::Value value = builder.create<mlir_ts::ArrayShiftOp>(
            location, cast<mlir_ts::ArrayType>(operands.front().getType()).getElementType(), thisValue);

        // same transfer as pop, from the front
        markResultOwned(value);

        return value;
    }    

    ValueOrLogicalResult mlirGenArraySplice(const mlir::Location &location, mlir::Value thisValue, mlir::Value startValue, mlir::Value deleteCountValue, ArrayRef<mlir::Value> values,
        std::function<ValueOrLogicalResult(mlir::Location, mlir::Type, mlir::Value, const GenContext &, bool)> castFn, const GenContext &genContext)
    {
        MLIRCodeLogic mcl(builder, compileOptions);

        // The lowering reads start and delete count as signed (a negative start counts from the
        // end), so they go to index through a signed 64-bit integer: straight from s32 to index
        // is a zero extension, and -1 arrived as 4294967295.
        auto toSignedIndex = [&](mlir::Value value) -> mlir::Value {
            if (isa<mlir::IndexType>(value.getType()))
            {
                return value;
            }

            auto si64Type = mlir::IntegerType::get(builder.getContext(), 64, mlir::IntegerType::Signed);
            if (value.getType() != si64Type)
            {
                value = castFn(location, si64Type, value, genContext, false);
                if (!value)
                {
                    return value;
                }
            }

            return castFn(location, mlir::IndexType::get(builder.getContext()), value, genContext, false);
        };

        startValue = toSignedIndex(startValue);
        deleteCountValue = toSignedIndex(deleteCountValue);
        if (!startValue || !deleteCountValue)
        {
            return mlir::failure();
        }

        auto arrayElement = cast<mlir_ts::ArrayType>(thisValue.getType()).getElementType();

        SmallVector<mlir::Value> castedValues;
        if (mlir::failed(castEach(location, arrayElement, values, castedValues, castFn, genContext)))
        {
            return mlir::failure();
        }

        auto thisValueLoaded = mcl.GetReferenceFromValue(location, thisValue);
        if (!thisValueLoaded)
        {
            emitError(location) << "Can't get reference of the array, ensure const array is not used";
            return mlir::failure();
        }

        retainInsertedElements(castedValues);

        mlir::Value sizeOfValue =
            builder.create<mlir_ts::ArraySpliceOp>(location, builder.getIndexType(), thisValueLoaded, startValue, deleteCountValue, mlir::ValueRange{castedValues});

        return sizeOfValue;
    }    

    ValueOrLogicalResult mlirGenArraySplice(const mlir::Location &location, ArrayRef<mlir::Value> operands,
        std::function<ValueOrLogicalResult(mlir::Location, mlir::Type, mlir::Value, const GenContext &, bool)> castFn, const GenContext &genContext)
    {
        if (operands.size() < 2)
        {
            emitError(location) << "splice needs the index to start at";
            return mlir::failure();
        }

        // a left-out delete count removes everything from start on; the lowering clamps it to
        // what is there (INT32_MAX stays positive in a 32-bit index too)
        auto deleteCountValue = operands.size() > 2
            ? operands[2]
            : builder.create<mlir_ts::ConstantOp>(location, builder.getIndexType(), builder.getIndexAttr(INT32_MAX)).getResult();

        return mlirGenArraySplice(location, operands.front(), operands[1], deleteCountValue,
                                  operands.size() > 3 ? operands.slice(3) : ArrayRef<mlir::Value>(), castFn, genContext);
    }    

    ValueOrLogicalResult mlirGenArrayView(const mlir::Location &location, mlir::Value thisValue, ArrayRef<mlir::Value> values,
        std::function<ValueOrLogicalResult(mlir::Location, mlir::Type, mlir::Value, const GenContext &, bool)> castFn, const GenContext &genContext)
    {
        MLIRCodeLogic mcl(builder, compileOptions);

        auto indexType = builder.getIndexType();

        SmallVector<mlir::Value> castedValues;
        if (mlir::failed(castEach(location, indexType, values, castedValues, castFn, genContext)))
        {
            return mlir::failure();
        }

        auto count = builder.create<mlir_ts::ArithmeticBinaryOp>(
            location, indexType, builder.getI32IntegerAttr(static_cast<int32_t>(SyntaxKind::MinusToken)), castedValues[1], castedValues[0]);
        
        auto oneIndex = builder.create<mlir_ts::ConstantOp>(
            location, indexType, builder.getIndexAttr(1));

        auto countPlusOne = builder.create<mlir_ts::ArithmeticBinaryOp>(
            location, indexType, builder.getI32IntegerAttr(static_cast<int32_t>(SyntaxKind::PlusToken)), count, oneIndex);
    
        mlir::Value arrayViewValue =
            builder.create<mlir_ts::ArrayViewOp>(
                location, 
                cast<mlir_ts::ArrayType>(thisValue.getType()), 
                thisValue, 
                castedValues[0], 
                countPlusOne);

        return arrayViewValue;
    }    

    ValueOrLogicalResult mlirGenArrayView(const mlir::Location &location, ArrayRef<mlir::Value> operands, 
        std::function<ValueOrLogicalResult(mlir::Location, mlir::Type, mlir::Value, const GenContext &, bool)> castFn, const GenContext &genContext)
    {
        return mlirGenArrayView(location, operands.front(), operands.slice(1), castFn, genContext);
    }    

    mlir::LogicalResult mlirGenSwitchState(const mlir::Location &location, ArrayRef<mlir::Value> operands,
        std::function<ValueOrLogicalResult(mlir::Location, mlir::Type, mlir::Value, const GenContext &, bool)> castFn, const GenContext &genContext)
    {
        auto op = operands.front();

        auto int32Type = mlir::IntegerType::get(op.getType().getContext(), 32);
        if (op.getType() != int32Type)
        {
            op = castFn(location, int32Type, op, genContext, false);
            if (!op)
            {
                return mlir::failure();
            }
        }

        auto switchStateOp =
            builder.create<mlir_ts::SwitchStateOp>(location, op, builder.getBlock(), mlir::BlockRange{});

        auto *block = builder.createBlock(builder.getBlock()->getParent());
        switchStateOp.setSuccessor(block, 0);

        // NOTE: upward mailbox: arms variable allocation placement for the statements following the
        // __switchstate call in the caller's scope - see GenContext
        const_cast<GenContext &>(genContext).allocateVarsOutsideOfOperation = true;
        const_cast<GenContext &>(genContext).currentOperation = switchStateOp;

        return mlir::success();
    }

    mlir::LogicalResult mlirGenLoadLibraryPermanently(const mlir::Location &location, ArrayRef<mlir::Value> operands,
        std::function<ValueOrLogicalResult(mlir::Location, mlir::Type, mlir::Value, const GenContext &, bool)> castFn, const GenContext &genContext)
    {
        mlir::Value fileNameValue;
        for (auto &oper : operands)
        {
            if (!isa<mlir_ts::StringType>(oper.getType()))
            {
                auto strCast =
                    castFn(location, mlir_ts::StringType::get(builder.getContext()), oper, genContext, false);
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

    ValueOrLogicalResult mlirGenSearchForAddressOfSymbol(const mlir::Location &location, ArrayRef<mlir::Value> operands,
        std::function<ValueOrLogicalResult(mlir::Location, mlir::Type, mlir::Value, const GenContext &, bool)> castFn, const GenContext &genContext)
    {
        mlir::Value symbolNameValue;
        for (auto &oper : operands)
        {
            if (!isa<mlir_ts::StringType>(oper.getType()))
            {
                auto strCast =
                    castFn(location, mlir_ts::StringType::get(builder.getContext()), oper, genContext, false);
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

    ValueOrLogicalResult mlirGenLoadReference(const mlir::Location &location, ArrayRef<mlir::Value> operands,
        std::function<ValueOrLogicalResult(mlir::Location, mlir::Type, mlir::Value, const GenContext &, bool)> castFn, const GenContext &genContext)
    {
        mlir::Value refValue;
        for (auto &oper : operands)
        {
            if (isa<mlir_ts::OpaqueType>(oper.getType()))
            {
                auto opaqueCast =
                    castFn(location, mlir_ts::RefType::get(builder.getContext(), oper.getType()), oper, genContext, false);
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
        MLIRCodeLogic mcl(builder, compileOptions);
        auto refValue = mcl.GetReferenceFromValue(location, operands.front()); 
        if (!refValue)       
        {
            emitError(location, "Are you trying to get reference of constant object or variable?");
            return mlir::failure();
        }

        return V(refValue);
    }    

    // The atomic builtins. Their op codes, orderings, alignments and flags become attributes of the
    // op, so each has to be a constant: a variable used to give the op a null attribute, and
    // tslang crashed building it. A constant LLVM does not accept was only caught by LLVM's
    // verifier, reported at `<unknown>:0`. Both are reported here, at the call.
    //
    //   atomicrmw(op, ref, value, ordering[, syncscope[, alignment[, volatile]]])
    //   cmpxchg(ref, expected, value, successOrdering, failureOrdering[, syncscope[, alignment[, weak[, volatile]]]])
    //   atomicload(ref, ordering[, syncscope[, alignment[, volatile]]])
    //   atomicstore(ref, value, ordering[, syncscope[, alignment[, volatile]]])
    //   fence(ordering[, syncscope])
    //
    // The orderings are LLVM's: 0 not_atomic, 1 unordered, 2 monotonic, 4 acquire, 5 release,
    // 6 acq_rel, 7 seq_cst (3 is not one). A syncscope of "" and an alignment of 0 mean the default
    // (the natural alignment of the value), so a call that wants a trailing flag passes them. The
    // value (and cmpxchg's expected value) is cast to the type the reference points at, which is
    // the type the atomic reads and writes.
    using AtomicCastFn = std::function<ValueOrLogicalResult(mlir::Location, mlir::Type, mlir::Value, const GenContext &, bool)>;

    static constexpr const char *atomicRMWSignature = "atomicrmw(op, ref, value, ordering[, syncscope[, alignment[, volatile]]])";
    static constexpr const char *cmpXchgSignature =
        "cmpxchg(ref, expected, value, successOrdering, failureOrdering[, syncscope[, alignment[, weak[, volatile]]]])";
    static constexpr const char *atomicLoadSignature = "atomicload(ref, ordering[, syncscope[, alignment[, volatile]]])";
    static constexpr const char *atomicStoreSignature = "atomicstore(ref, value, ordering[, syncscope[, alignment[, volatile]]])";
    static constexpr const char *fenceSignature = "fence(ordering[, syncscope])";

    mlir::Attribute getConstantAttr(mlir::Value oper)
    {
        if (auto constantOp = oper.getDefiningOp<mlir_ts::ConstantOp>())
        {
            return constantOp.getValue();
        }

        if (auto literalType = dyn_cast<mlir_ts::LiteralType>(oper.getType()))
        {
            return literalType.getValue();
        }

        return mlir::Attribute();
    }

    bool checkAtomicArgCount(const mlir::Location &location, StringRef signature, size_t size, size_t min, size_t max)
    {
        if (size >= min && size <= max)
        {
            return true;
        }

        emitError(location) << "expected " << signature << ", but got " << size << " argument" << (size == 1 ? "" : "s");
        return false;
    }

    std::optional<int64_t> getAtomicConstInt(const mlir::Location &location, StringRef builtin, StringRef argName, mlir::Value oper)
    {
        auto value = getConstantAttr(oper);
        if (auto intAttr = dyn_cast_or_null<mlir::IntegerAttr>(value); intAttr && !isa<mlir::BoolAttr>(value))
        {
            return intAttr.getValue().getSExtValue();
        }

        if (auto floatAttr = dyn_cast_or_null<mlir::FloatAttr>(value))
        {
            auto floatValue = floatAttr.getValueAsDouble();
            if (floatValue >= -1e9 && floatValue <= 1e9 && floatValue == static_cast<double>(static_cast<int64_t>(floatValue)))
            {
                return static_cast<int64_t>(floatValue);
            }
        }

        emitError(location) << builtin << ": the " << argName << " must be a constant integer";
        return std::nullopt;
    }

    std::optional<bool> getAtomicConstFlag(const mlir::Location &location, StringRef builtin, StringRef argName, mlir::Value oper)
    {
        if (auto boolAttr = dyn_cast_or_null<mlir::BoolAttr>(getConstantAttr(oper)))
        {
            return boolAttr.getValue();
        }

        emitError(location) << builtin << ": the " << argName << " flag must be a constant boolean";
        return std::nullopt;
    }

    static StringRef atomicOrderingName(int64_t ordering)
    {
        switch (ordering)
        {
        case 0: return "not_atomic";
        case 1: return "unordered";
        case 2: return "monotonic";
        case 4: return "acquire";
        case 5: return "release";
        case 6: return "acq_rel";
        case 7: return "seq_cst";
        default: return "";
        }
    }

    // `allowed` lists the orderings this argument accepts, in LLVM's numbering
    std::optional<int64_t> getAtomicOrdering(const mlir::Location &location, StringRef builtin, StringRef argName, mlir::Value oper,
        ArrayRef<int64_t> allowed)
    {
        auto ordering = getAtomicConstInt(location, builtin, argName, oper);
        if (!ordering)
        {
            return std::nullopt;
        }

        if (llvm::is_contained(allowed, *ordering))
        {
            return ordering;
        }

        auto diag = emitError(location) << builtin << ": the " << argName << " " << *ordering;
        auto name = atomicOrderingName(*ordering);
        if (!name.empty())
        {
            diag << " (" << name << ")";
        }

        diag << " is not allowed here; expected ";
        for (auto [index, value] : llvm::enumerate(allowed))
        {
            diag << (index == 0 ? "" : index + 1 == allowed.size() ? " or " : ", ") << value << " (" << atomicOrderingName(value) << ")";
        }

        return std::nullopt;
    }

    // the optional syncscope ("" is the default, system scope) and alignment (0 is the natural one)
    mlir::LogicalResult getAtomicScopeAndAlignment(const mlir::Location &location, StringRef builtin, ArrayRef<mlir::Value> operands,
        size_t scopeIndex, bool hasAlignment, mlir::StringAttr &syncscope, mlir::IntegerAttr &alignment)
    {
        if (operands.size() > scopeIndex)
        {
            auto scope = dyn_cast_or_null<mlir::StringAttr>(getConstantAttr(operands[scopeIndex]));
            if (!scope)
            {
                emitError(location) << builtin << ": the syncscope must be a constant string";
                return mlir::failure();
            }

            if (!scope.getValue().empty())
            {
                syncscope = scope;
            }
        }

        if (hasAlignment && operands.size() > scopeIndex + 1)
        {
            auto align = getAtomicConstInt(location, builtin, "alignment", operands[scopeIndex + 1]);
            if (!align)
            {
                return mlir::failure();
            }

            if (*align < 0 || (*align & (*align - 1)) != 0)
            {
                emitError(location) << builtin << ": the alignment " << *align << " is not a power of two (0 is the natural one)";
                return mlir::failure();
            }

            if (*align > 0)
            {
                alignment = builder.getI64IntegerAttr(*align);
            }
        }

        return mlir::success();
    }

    mlir::LogicalResult getAtomicFlag(const mlir::Location &location, StringRef builtin, StringRef argName, ArrayRef<mlir::Value> operands,
        size_t index, mlir::UnitAttr &flag)
    {
        if (operands.size() > index)
        {
            auto value = getAtomicConstFlag(location, builtin, argName, operands[index]);
            if (!value)
            {
                return mlir::failure();
            }

            if (*value)
            {
                flag = builder.getUnitAttr();
            }
        }

        return mlir::success();
    }

    enum class AtomicValueKind
    {
        // xchg, atomicload, atomicstore: an integer, a float or a pointer
        Any,
        // add, sub, and, nand, or, xor, max, min, umax, umin, uinc_wrap, udec_wrap, usub_cond, usub_sat
        Integer,
        // fadd, fsub, fmax, fmin, fmaximum, fminimum
        Float,
        // cmpxchg: an integer or a pointer
        IntegerOrPointer
    };

    static bool isAtomicIntegerType(mlir::Type type)
    {
        if (auto intType = dyn_cast<mlir::IntegerType>(type))
        {
            auto width = intType.getWidth();
            return width == 8 || width == 16 || width == 32 || width == 64;
        }

        return false;
    }

    static bool isAtomicFloatType(mlir::Type type)
    {
        return isa<mlir_ts::NumberType>(type) || type.isF16() || type.isBF16() || type.isF32() || type.isF64();
    }

    // the type the reference points at, which is the type the atomic reads and writes
    mlir::Type getAtomicElementType(const mlir::Location &location, StringRef builtin, mlir::Value ref, AtomicValueKind kind)
    {
        mlir::Type elementType;
        if (auto refType = dyn_cast<mlir_ts::RefType>(ref.getType()))
        {
            elementType = refType.getElementType();
        }
        else if (auto valueRefType = dyn_cast<mlir_ts::ValueRefType>(ref.getType()))
        {
            elementType = valueRefType.getElementType();
        }
        else
        {
            emitError(location) << builtin << ": expected a reference (ReferenceOf(x)), but got " << ref.getType();
            return mlir::Type();
        }

        auto isInteger = isAtomicIntegerType(elementType);
        auto isFloat = isAtomicFloatType(elementType);
        // a boolean is an i1, which no atomic takes; any other type is a pointer once lowered
        auto isOther = !isInteger && !isFloat && !isa<mlir::IntegerType, mlir::FloatType, mlir_ts::BooleanType>(elementType);
        auto ok = false;
        StringRef expected;
        switch (kind)
        {
        case AtomicValueKind::Any:
            ok = isInteger || isFloat || isOther;
            expected = "an 8, 16, 32 or 64-bit integer, a float or a reference";
            break;
        case AtomicValueKind::Integer:
            ok = isInteger;
            expected = "an 8, 16, 32 or 64-bit integer";
            break;
        case AtomicValueKind::Float:
            ok = isFloat;
            expected = "a float";
            break;
        case AtomicValueKind::IntegerOrPointer:
            ok = isInteger || isOther;
            expected = "an 8, 16, 32 or 64-bit integer or a reference";
            break;
        }

        if (!ok)
        {
            emitError(location) << builtin << ": the reference points at " << elementType << ", expected " << expected;
            return mlir::Type();
        }

        return elementType;
    }

    mlir::Value castAtomicValue(const mlir::Location &location, mlir::Type elementType, mlir::Value value, AtomicCastFn castFn,
        const GenContext &genContext)
    {
        if (value.getType() == elementType)
        {
            return value;
        }

        auto result = castFn(location, elementType, value, genContext, false);
        if (mlir::failed(result) || !result.value)
        {
            return mlir::Value();
        }

        return result;
    }

    ValueOrLogicalResult mlirGenAtomicRMW(const mlir::Location &location, ArrayRef<mlir::Value> operands, AtomicCastFn castFn,
        const GenContext &genContext)
    {
        const StringRef builtin = "atomicrmw";
        if (!checkAtomicArgCount(location, atomicRMWSignature, operands.size(), 4, 7))
        {
            return mlir::failure();
        }

        auto binOp = getAtomicConstInt(location, builtin, "op code", operands[0]);
        if (!binOp)
        {
            return mlir::failure();
        }

        // LLVM's AtomicRMWInst::BinOp
        if (*binOp < 0 || *binOp > 20)
        {
            emitError(location) << builtin << ": the op code " << *binOp
                                << " is not one of 0 (xchg), 1 (add), 2 (sub), 3 (and), 4 (nand), 5 (or), 6 (xor), 7 (max), "
                                   "8 (min), 9 (umax), 10 (umin), 11 (fadd), 12 (fsub), 13 (fmax), 14 (fmin), 15 (uinc_wrap), "
                                   "16 (udec_wrap), 17 (usub_cond), 18 (usub_sat), 19 (fmaximum) or 20 (fminimum)";
            return mlir::failure();
        }

        auto isFloatOp = (*binOp >= 11 && *binOp <= 14) || *binOp >= 19;
        auto kind = *binOp == 0 ? AtomicValueKind::Any : isFloatOp ? AtomicValueKind::Float : AtomicValueKind::Integer;
        auto elementType = getAtomicElementType(location, builtin, operands[1], kind);
        if (!elementType)
        {
            return mlir::failure();
        }

        auto ordering = getAtomicOrdering(location, builtin, "ordering", operands[3], {2, 4, 5, 6, 7});
        if (!ordering)
        {
            return mlir::failure();
        }

        mlir::StringAttr syncscope;
        mlir::IntegerAttr alignment;
        mlir::UnitAttr isVolatile;
        if (mlir::failed(getAtomicScopeAndAlignment(location, builtin, operands, 4, true, syncscope, alignment))
            || mlir::failed(getAtomicFlag(location, builtin, "volatile", operands, 6, isVolatile)))
        {
            return mlir::failure();
        }

        auto value = castAtomicValue(location, elementType, operands[2], castFn, genContext);
        if (!value)
        {
            return mlir::failure();
        }

        return V(builder.create<mlir_ts::AtomicRMWOp>(location, elementType, builder.getI32IntegerAttr(*binOp), operands[1], value,
            builder.getI32IntegerAttr(*ordering), syncscope, alignment, isVolatile));
    }

    mlir::Type getTupleType(mlir::SmallVector<mlir_ts::FieldInfo> &fieldInfos)
    {
        return mlir_ts::TupleType::get(builder.getContext(), fieldInfos);
    }

    ValueOrLogicalResult mlirGenCmpXchg(const mlir::Location &location, ArrayRef<mlir::Value> operands, AtomicCastFn castFn,
        const GenContext &genContext)
    {
        const StringRef builtin = "cmpxchg";
        if (!checkAtomicArgCount(location, cmpXchgSignature, operands.size(), 5, 9))
        {
            return mlir::failure();
        }

        auto elementType = getAtomicElementType(location, builtin, operands[0], AtomicValueKind::IntegerOrPointer);
        if (!elementType)
        {
            return mlir::failure();
        }

        auto successOrdering = getAtomicOrdering(location, builtin, "success ordering", operands[3], {2, 4, 5, 6, 7});
        if (!successOrdering)
        {
            return mlir::failure();
        }

        // the failure ordering is that of a load: it can't release
        auto failureOrdering = getAtomicOrdering(location, builtin, "failure ordering", operands[4], {2, 4, 7});
        if (!failureOrdering)
        {
            return mlir::failure();
        }

        mlir::StringAttr syncscope;
        mlir::IntegerAttr alignment;
        mlir::UnitAttr isWeak;
        mlir::UnitAttr isVolatile;
        if (mlir::failed(getAtomicScopeAndAlignment(location, builtin, operands, 5, true, syncscope, alignment))
            || mlir::failed(getAtomicFlag(location, builtin, "weak", operands, 7, isWeak))
            || mlir::failed(getAtomicFlag(location, builtin, "volatile", operands, 8, isVolatile)))
        {
            return mlir::failure();
        }

        auto expected = castAtomicValue(location, elementType, operands[1], castFn, genContext);
        if (!expected)
        {
            return mlir::failure();
        }

        auto value = castAtomicValue(location, elementType, operands[2], castFn, genContext);
        if (!value)
        {
            return mlir::failure();
        }

        mlir::SmallVector<mlir_ts::FieldInfo> fields;
        fields.push_back(
            mlir_ts::FieldInfo{
                mlir::Attribute(),
                elementType,
                false,
                mlir_ts::AccessLevel::Public
            });

        fields.push_back(
            mlir_ts::FieldInfo{
                mlir::Attribute(),
                mlir_ts::BooleanType::get(builder.getContext()),
                false,
                mlir_ts::AccessLevel::Public
            });

        auto resultTuple = getTupleType(fields);

        return V(builder.create<mlir_ts::AtomicCmpXchgOp>(location, resultTuple, operands[0], expected, value,
            builder.getI32IntegerAttr(*successOrdering), builder.getI32IntegerAttr(*failureOrdering), syncscope, alignment,
            isWeak, isVolatile));
    }

    ValueOrLogicalResult mlirGenAtomicLoad(const mlir::Location &location, ArrayRef<mlir::Value> operands)
    {
        const StringRef builtin = "atomicload";
        if (!checkAtomicArgCount(location, atomicLoadSignature, operands.size(), 2, 5))
        {
            return mlir::failure();
        }

        auto elementType = getAtomicElementType(location, builtin, operands[0], AtomicValueKind::Any);
        if (!elementType)
        {
            return mlir::failure();
        }

        // a load can't release
        auto ordering = getAtomicOrdering(location, builtin, "ordering", operands[1], {1, 2, 4, 7});
        if (!ordering)
        {
            return mlir::failure();
        }

        mlir::StringAttr syncscope;
        mlir::IntegerAttr alignment;
        mlir::UnitAttr isVolatile;
        if (mlir::failed(getAtomicScopeAndAlignment(location, builtin, operands, 2, true, syncscope, alignment))
            || mlir::failed(getAtomicFlag(location, builtin, "volatile", operands, 4, isVolatile)))
        {
            return mlir::failure();
        }

        return V(builder.create<mlir_ts::AtomicLoadOp>(location, elementType, operands[0], builder.getI32IntegerAttr(*ordering),
            syncscope, alignment, isVolatile));
    }

    ValueOrLogicalResult mlirGenAtomicStore(const mlir::Location &location, ArrayRef<mlir::Value> operands, AtomicCastFn castFn,
        const GenContext &genContext)
    {
        const StringRef builtin = "atomicstore";
        if (!checkAtomicArgCount(location, atomicStoreSignature, operands.size(), 3, 6))
        {
            return mlir::failure();
        }

        auto elementType = getAtomicElementType(location, builtin, operands[0], AtomicValueKind::Any);
        if (!elementType)
        {
            return mlir::failure();
        }

        // a store can't acquire
        auto ordering = getAtomicOrdering(location, builtin, "ordering", operands[2], {1, 2, 5, 7});
        if (!ordering)
        {
            return mlir::failure();
        }

        mlir::StringAttr syncscope;
        mlir::IntegerAttr alignment;
        mlir::UnitAttr isVolatile;
        if (mlir::failed(getAtomicScopeAndAlignment(location, builtin, operands, 3, true, syncscope, alignment))
            || mlir::failed(getAtomicFlag(location, builtin, "volatile", operands, 5, isVolatile)))
        {
            return mlir::failure();
        }

        auto value = castAtomicValue(location, elementType, operands[1], castFn, genContext);
        if (!value)
        {
            return mlir::failure();
        }

        builder.create<mlir_ts::AtomicStoreOp>(location, operands[0], value, builder.getI32IntegerAttr(*ordering), syncscope,
            alignment, isVolatile);
        return mlir::success();
    }

    ValueOrLogicalResult mlirGenFence(const mlir::Location &location, ArrayRef<mlir::Value> operands)
    {
        const StringRef builtin = "fence";
        if (!checkAtomicArgCount(location, fenceSignature, operands.size(), 1, 2))
        {
            return mlir::failure();
        }

        // a fence orders: acquire, release, acq_rel or seq_cst
        auto ordering = getAtomicOrdering(location, builtin, "ordering", operands[0], {4, 5, 6, 7});
        if (!ordering)
        {
            return mlir::failure();
        }

        mlir::StringAttr syncscope;
        mlir::IntegerAttr alignment;
        if (mlir::failed(getAtomicScopeAndAlignment(location, builtin, operands, 1, false, syncscope, alignment)))
        {
            return mlir::failure();
        }

        builder.create<mlir_ts::FenceOp>(location, builder.getI32IntegerAttr(*ordering), syncscope);
        return mlir::success();
    }

    ValueOrLogicalResult mlirGenInlineAsm(const mlir::Location &location, mlir::SmallVector<mlir::Type> typeArgs, ArrayRef<mlir::Value> operands)
    {
        MLIRCodeLogic mcl(builder, compileOptions);
        auto asm_string = mcl.getStringAttr(location, operands[0]);
        auto constraints = mcl.getStringAttr(location, operands[1]);
        auto args = operands.drop_front(2);
        if (typeArgs.size() > 0)
        {
            auto result = builder.create<mlir_ts::InlineAsmOp>(location, mlir::TypeRange(typeArgs), mlir::ValueRange(args), asm_string, constraints, mlir::UnitAttr(), mlir::UnitAttr(), mlir::IntegerAttr(), mlir::ArrayAttr());
            return result.getResults();
        }
        else
        {
            builder.create<mlir_ts::InlineAsmOp>(location, mlir::TypeRange(), mlir::ValueRange(args), asm_string, constraints, mlir::UnitAttr(), mlir::UnitAttr(), mlir::IntegerAttr(), mlir::ArrayAttr());
            // TODO: finish version with return value
            return mlir::success();
        }
    }     

    ValueOrLogicalResult mlirGenCallIntrinsic(const mlir::Location &location, mlir::SmallVector<mlir::Type> typeArgs, ArrayRef<mlir::Value> operands)
    {
        MLIRCodeLogic mcl(builder, compileOptions);
        auto intrin_string = mcl.getStringAttr(location, operands[0]);
        auto args = operands.drop_front(1);

        if (typeArgs.size() > 0)
        {
            auto result = builder.create<mlir_ts::CallIntrinsicOp>(location, mlir::TypeRange(typeArgs), intrin_string, 
                mlir::ValueRange(args), builder.getIntegerAttr(builder.getI32Type(), 0));
            return result.getResults();
        }
        else
        {
            builder.create<mlir_ts::CallIntrinsicOp>(location, mlir::TypeRange(), intrin_string, 
                mlir::ValueRange(args), builder.getIntegerAttr(builder.getI32Type(), 0));
            return mlir::success();
        }
    }     

    ValueOrLogicalResult mlirGenLinkerOptions(const mlir::Location &location, ArrayRef<mlir::Value> operands)
    {
        SmallVector<mlir::Attribute> strAttrs;
        MLIRCodeLogic mcl(builder, compileOptions);

        for (auto oper : operands) 
        {
            if (auto val = mcl.getStringAttr(location, oper))
            {
                strAttrs.push_back(val);
            }
            else
            {
                return mlir::failure();
            }
        }

        builder.create<mlir_ts::LinkerOptionsOp>(location, builder.getArrayAttr(strAttrs));
        return mlir::success();
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
    CompileOptions& compileOptions;
    llvm::ScopedHashTable<mlir::Value, mlir::Value> *boundRefMaterializedCache = nullptr;

  public:
    // optional: lets a bound-method property access (e.g. `g.next` on a storage-less
    // `const` binding) reuse the ref it materialized on a previous access instead of
    // minting a fresh one seeded from the pristine, never-mutated value each time. See
    // TupleNoError() and MLIRGenImpl::boundRefMaterializedCache.
    void setBoundRefMaterializedCache(llvm::ScopedHashTable<mlir::Value, mlir::Value> *cache)
    {
        boundRefMaterializedCache = cache;
    }

    MLIRPropertyAccessCodeLogic(CompileOptions& compileOptions, mlir::OpBuilder &builder, mlir::Location location, mlir::Value expression,
                                StringRef name)
        : builder(builder), location(location), expression(expression), name(name), compileOptions(compileOptions)
    {
        fieldId = MLIRHelper::TupleFieldName(name, builder.getContext());
    }

    MLIRPropertyAccessCodeLogic(CompileOptions& compileOptions, mlir::OpBuilder &builder, mlir::Location location, mlir::Value expression,
                                mlir::Attribute fieldId)
        : builder(builder), location(location), expression(expression), fieldId(fieldId), compileOptions(compileOptions)
    {
        if (auto strAttr = dyn_cast<mlir::StringAttr>(fieldId))
        {
            name = strAttr.getValue();
        }
    }

    MLIRPropertyAccessCodeLogic(CompileOptions& compileOptions, mlir::OpBuilder &builder, mlir::Location location, mlir::Value expression,
                                mlir::Attribute fieldId, mlir::Value argument)
        : builder(builder), location(location), expression(expression), fieldId(fieldId), argument(argument), compileOptions(compileOptions)
    {
        if (auto strAttr = dyn_cast<mlir::StringAttr>(fieldId))
        {
            name = strAttr.getValue();
        }
    }

    // the object is accessed as another value of it - a union as its base type
    void setExpression(mlir::Value newExpression)
    {
        expression = newExpression;
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
            .Default([&](auto type) {
                // TS enum members can only be string or numeric literals (string/int/float
                // attrs cover both, bool is an extra defensive case); fail cleanly instead of
                // crashing if some other constant attribute kind is ever produced here.
                emitError(location, "unsupported enum member value type");
            });

        if (!typeFromAttr)
        {
            return mlir::Value();
        }

        LLVM_DEBUG(llvm::dbgs() << "\n!! enum: " << propName << " value attr: " << valueAttr << "\n");

        // return builder.create<mlir_ts::ConstantOp>(location, enumType.getElementType(), valueAttr);
        auto literalType = mlir_ts::LiteralType::get(valueAttr, typeFromAttr);
        return builder.create<mlir_ts::ConstantOp>(location, literalType, valueAttr);
    }

    template <typename T> mlir::Value Tuple(T tupleType, bool indexAccess = false)
    {
        mlir::Value value;

        MLIRTypeHelper mth(builder.getContext(), compileOptions);
        MLIRCodeLogic mcl(builder, compileOptions);

        // resolve index
        auto [fieldIndex, elementTypeForRef, accessLevel] = mcl.TupleFieldTypeNoError(tupleType, fieldId, indexAccess);
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
        auto elementType = mth.isBoundReference(elementTypeForRef, isBoundRef);

        auto refValue = getExprLoadRefValue(location);
        if (isBoundRef && !refValue && boundRefMaterializedCache)
        {
            // only reuse a ref materialized in the SAME block as this access: a ref
            // minted inside a nested block (e.g. a `{ }` scope) does not dominate uses
            // outside that block, and checking real dominance would need MLIR's
            // DominanceInfo, which isn't otherwise used in this codebase. Same-block is
            // a conservative, cheap approximation -- it misses some reuse opportunities
            // across block boundaries but never returns a ref that fails to dominate.
            auto cached = boundRefMaterializedCache->lookup(expression);
            if (cached && cached.getParentBlock() == builder.getInsertionBlock())
            {
                refValue = cached;
            }
        }

        if (isBoundRef && !refValue)
        {
            // allocate in stack
            refValue =
                builder.create<mlir_ts::VariableOp>(location, mlir_ts::RefType::get(expression.getType()), expression);

            if (boundRefMaterializedCache)
            {
                boundRefMaterializedCache->insert(expression, refValue);
            }
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
        MLIRCodeLogic mcl(builder, compileOptions);

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

            // `expression` is pointer-like (ObjectType, or a Ref/BoundRef reached via
            // RefLogic) when this accessor lookup came from a boxed object literal
            // (docs/object-literal-boxing-design.md) rather than a value-typed tuple:
            // ExtractPropertyOp's operand is restricted to AnyStructLike (value tuples/
            // storage types), which ObjectType/RefType are not part of, so such operands
            // need PropertyRefOp + LoadOp instead (same recipe as RefLogic's own direct
            // field access just above/below this helper).
            auto expressionIsRefLike = isa<mlir_ts::ObjectType>(expression.getType())
                || isa<mlir_ts::RefType>(expression.getType())
                || isa<mlir_ts::BoundRefType>(expression.getType());

            mlir::Value getterValue;
            mlir::Value setterValue;

            if (getterIndex >= 0)
            {
                if (expressionIsRefLike)
                {
                    auto propRef = builder.create<mlir_ts::PropertyRefOp>(location, mlir_ts::RefType::get(getterFuncType),
                        expression, builder.getI32IntegerAttr(getterIndex));
                    getterValue = builder.create<mlir_ts::LoadOp>(location, getterFuncType, propRef);
                }
                else
                {
                    getterValue = builder.create<mlir_ts::ExtractPropertyOp>(location, getterFuncType, expression,
                        MLIRHelper::getStructIndex(builder, getterIndex));
                }
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
                if (expressionIsRefLike)
                {
                    auto propRef = builder.create<mlir_ts::PropertyRefOp>(location, mlir_ts::RefType::get(setterFuncType),
                        expression, builder.getI32IntegerAttr(setterIndex));
                    setterValue = builder.create<mlir_ts::LoadOp>(location, setterFuncType, propRef);
                }
                else
                {
                    setterValue = builder.create<mlir_ts::ExtractPropertyOp>(location, setterFuncType, expression,
                        MLIRHelper::getStructIndex(builder, setterIndex));
                }
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

            mlir::Value refValue;
            if (expressionIsRefLike)
            {
                // already a pointer (e.g. a boxed object literal's ObjectType) -- use it
                // directly as `this`, same as RefLogic's own direct field access does;
                // wrapping it in another VariableOp/RefType would add a spurious
                // pointer-to-pointer indirection the accessor call doesn't expect.
                refValue = expression;
            }
            else
            {
                refValue = getExprLoadRefValue(location);
                if (!refValue)
                {
                    // allocate in stack
                    refValue = builder.create<mlir_ts::VariableOp>(
                        location, mlir_ts::RefType::get(expression.getType()), expression);
                }
            }

            auto thisValue = refValue;

            auto thisAccessorIndirectOp = builder.create<mlir_ts::ThisIndirectAccessorOp>(
                location, accessorResultType, thisValue, getterValue, setterValue, mlir::Value());  

            return thisAccessorIndirectOp.getResult(0);              
        }

        return mlir::Value();
    }

    template <typename T> mlir::Value TupleNoError(T tupleType, mlir_ts::AccessLevel accessingFromLevel, bool indexAccess = false)
    {
        mlir::Value value;

        MLIRTypeHelper mth(builder.getContext(), compileOptions);
        MLIRCodeLogic mcl(builder, compileOptions);

        // resolve index
        auto [fieldIndex, elementTypeForRef, accessLevel] = mcl.TupleFieldTypeNoError(tupleType, fieldId, indexAccess);
        if (fieldIndex < 0)
        {
            return value;
        }

        if (accessingFromLevel < accessLevel) {
            emitError(location, "Class member ") << fieldId << " is not accessable";
            return mlir::Value();
        }        

        bool isBoundRef = false;
        auto elementType = mth.isBoundReference(elementTypeForRef, isBoundRef);

        auto refValue = getExprLoadRefValue(location);
        if (isBoundRef && !refValue && boundRefMaterializedCache)
        {
            // only reuse a ref materialized in the SAME block as this access: a ref
            // minted inside a nested block (e.g. a `{ }` scope) does not dominate uses
            // outside that block, and checking real dominance would need MLIR's
            // DominanceInfo, which isn't otherwise used in this codebase. Same-block is
            // a conservative, cheap approximation -- it misses some reuse opportunities
            // across block boundaries but never returns a ref that fails to dominate.
            auto cached = boundRefMaterializedCache->lookup(expression);
            if (cached && cached.getParentBlock() == builder.getInsertionBlock())
            {
                refValue = cached;
            }
        }

        if (isBoundRef && !refValue)
        {
            // allocate in stack
            refValue =
                builder.create<mlir_ts::VariableOp>(location, mlir_ts::RefType::get(expression.getType()), expression);

            if (boundRefMaterializedCache)
            {
                boundRefMaterializedCache->insert(expression, refValue);
            }
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

            return builder.create<mlir_ts::StringLengthOp>(location, builder.getIndexType(), expression);
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

    template <typename T> mlir::Value Array(T arrayType, CompileOptions& compileOptions,
        std::function<ValueOrLogicalResult(mlir::Location, mlir::Type, mlir::Value, const GenContext &, bool)> castFn, const GenContext &genContext)
    {
        SmallVector<mlir::NamedAttribute> customAttrs;
        // customAttrs.push_back({MLIR_IDENT("__virtual"), MLIR_ATTR("true")});

        auto propName = getName();
        if (propName == LENGTH_FIELD_NAME)
        {
            if (auto constArrayType = dyn_cast<mlir_ts::ConstArrayType>(expression.getType()))
            {
                auto size = constArrayType.getSize();
                return builder.create<mlir_ts::ConstantOp>(location, builder.getIndexType(),
                            builder.getIndexAttr(size));
            }
            else if (isa<mlir_ts::ArrayType>(expression.getType()))
            {
                return builder.create<mlir_ts::LengthOfOp>(location, builder.getIndexType(), expression);
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

        if (compileOptions.enableBuiltins && compileOptions.noDefaultLib && isArrayCustomMethod(propName))
        {
            auto arrayType = dyn_cast<mlir_ts::ArrayType>(expression.getType());
            auto constArrayType = dyn_cast<mlir_ts::ConstArrayType>(expression.getType());
            if (arrayType || constArrayType)
            {
                mlir::Type elementType;
                if (constArrayType)
                {
                    elementType = constArrayType.getElementType();

                    MLIRTypeHelper mth(builder.getContext(), compileOptions);
                    auto nonConstArray = mth.convertConstArrayTypeToArrayType(expression.getType());
                    expression = castFn(location, nonConstArray, expression, genContext, false);
                    if (!expression)
                    {
                        return mlir::Value();
                    }
                }
                else
                {
                    elementType = arrayType.getElementType();
                }

                return CustomMethod(elementType);
            }

            return mlir::Value();
        }

        return mlir::Value();
    }

    // The custom methods an iterator has too: `reduce`'s builtin takes an array (`T[]`).
    bool isIteratorCustomMethod()
    {
        auto propName = getName();
        return isArrayCustomMethod(propName) && propName != "reduce";
    }

    // The builtin custom method `getName()` of `expression`, whose elements are `elementType`:
    // an array, or an iterator, which the builtin goes through with for...of.
    mlir::Value CustomMethod(mlir::Type elementType)
    {
        auto propName = getName();
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
            // property access through a RefType whose element isn't a Tuple/ConstTuple; this
            // function's only caller (MLIRGenAccessCall.cpp's property-access TypeSwitch) only
            // reaches the RefType Case for values shaped like a mutable tuple local. Fail
            // cleanly instead of crashing if some other RefType element shape reaches here.
            emitError(location) << "property access on reference to type '" << refType.getElementType() << "' is not supported";
            return mlir::Value();
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
            // a boxed ObjectType's storage type is always synthesized as one of the three cases
            // above (see docs/object-literal-boxing-design.md) - fail cleanly instead of
            // crashing if that ever stops being true.
            emitError(location) << "property access on object storage type '" << objectType.getStorageType() << "' is not supported";
            return mlir::Value();
        }
    }

    template <typename T> mlir::Value RefLogic(T tupleType)
    {
        MLIRTypeHelper mth(builder.getContext(), compileOptions);
        MLIRCodeLogic mcl(builder, compileOptions);

        // resolve index
        auto [fieldIndex, elementTypeForRef, accessLevel] = mcl.TupleFieldTypeNoError(tupleType, fieldId);
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
        auto elementType = mth.isBoundReference(elementTypeForRef, isBoundRef);

        auto refType = isBoundRef ? static_cast<mlir::Type>(mlir_ts::BoundRefType::get(elementTypeForRef))
                                  : static_cast<mlir::Type>(mlir_ts::RefType::get(elementTypeForRef));
        auto propRef = builder.create<mlir_ts::PropertyRefOp>(location, refType, expression,
                                                              builder.getI32IntegerAttr(fieldIndex));

        return builder.create<mlir_ts::LoadOp>(location, elementType, propRef);
    }

    mlir::Value Class(mlir_ts::ClassType classType, mlir_ts::AccessLevel accessingFromLevel)
    {
        if (auto classStorageType = dyn_cast<mlir_ts::ClassStorageType>(classType.getStorageType()))
        {
            return Class(classStorageType, accessingFromLevel);
        }
        else
        {
            // a ClassType's storage type is always a ClassStorageType by construction; fail
            // cleanly instead of crashing if that ever stops being true.
            emitError(location) << "unsupported class storage type: " << classType.getStorageType();
            return mlir::Value();
        }
    }

    mlir::Value Class(mlir_ts::ClassStorageType classStorageType, mlir_ts::AccessLevel accessingFromLevel)
    {
        MLIRCodeLogic mcl(builder, compileOptions);

        // resolve index
        auto [fieldIndex, elementTypeForRef, accessLevel] = mcl.TupleFieldTypeNoError(classStorageType, fieldId);
        if (fieldIndex < 0)
        {
            return mlir::Value();
        }

        if (accessingFromLevel < accessLevel) {
            emitError(location, "Class member ") << fieldId << " is not accessable";
            return mlir::Value();
        }

        bool isBoundRef = false;
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
        auto symbOp = builder.create<mlir_ts::ConstantOp>(location, symbol, mlir::StringAttr::get(builder.getContext(), name));

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
        MLIRCodeLogic mcl(builder, compileOptions);
        return mcl.ExtractAttr(expression);
    }

    mlir::Value getExprLoadRefValue(mlir::Location location)
    {
        MLIRCodeLogic mcl(builder, compileOptions);
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
#undef V

#endif // MLIR_TYPESCRIPT_MLIRGENLOGIC_MLIRCODELOGIC_H_
