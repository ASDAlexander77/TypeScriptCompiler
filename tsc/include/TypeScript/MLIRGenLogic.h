#ifndef MLIR_TYPESCRIPT_MLIRGENLOGIC_H_
#define MLIR_TYPESCRIPT_MLIRGENLOGIC_H_

#include "TypeScript/TypeScriptDialect.h"
#include "TypeScript/TypeScriptOps.h"

#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Types.h"

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"

#include "TypeScript/CommonGenLogic.h"
#include "TypeScript/DOM.h"

#include <numeric>

#define DEBUG_TYPE "mlir"

namespace mlir_ts = mlir::typescript;

using llvm::ArrayRef;
using llvm::cast;
using llvm::dyn_cast;
using llvm::dyn_cast_or_null;
using llvm::isa;
using llvm::makeArrayRef;
using llvm::SmallVector;
using llvm::StringRef;
using llvm::Twine;

using VariablePairT = std::pair<mlir::Value, ts::VariableDeclarationDOM::TypePtr>;
using SymbolTableScopeT = llvm::ScopedHashTableScope<StringRef, VariablePairT>;

namespace typescript
{

class MLIRCustomMethods
{
    mlir::OpBuilder &builder;
    mlir::Location &location;

  public:
    MLIRCustomMethods(mlir::OpBuilder &builder, mlir::Location &location) : builder(builder), location(location)
    {
    }

    mlir::Value callMethod(StringRef functionName, ArrayRef<mlir::Value> operands, bool allowPartialResolve)
    {
        // validate params
        for (auto &oper : operands)
        {
            if (allowPartialResolve && !oper)
            {
                return mlir::Value();
            }

            if (auto unresolvedLeft = dyn_cast_or_null<mlir_ts::SymbolRefOp>(oper.getDefiningOp()))
            {
                if (!allowPartialResolve)
                {
                    emitError(oper.getLoc(), "can't find variable: ") << unresolvedLeft.identifier();
                }

                return mlir::Value();
            }
        }

        mlir::Value result;
        // print - internal command;
        if (functionName.compare(StringRef("print")) == 0)
        {
            mlir::succeeded(mlirGenPrint(location, operands));
        }
        else
            // assert - internal command;
            if (functionName.compare(StringRef("assert")) == 0)
        {
            mlir::succeeded(mlirGenAssert(location, operands));
        }
        else
            // assert - internal command;
            if (functionName.compare(StringRef("parseInt")) == 0)
        {
            result = mlirGenParseInt(location, operands);
        }
        else if (functionName.compare(StringRef("parseFloat")) == 0)
        {
            result = mlirGenParseFloat(location, operands);
        }
        else if (functionName.compare(StringRef("sizeof")) == 0)
        {
            result = mlirGenSizeOf(location, operands);
        }
        /*
        else
        if (functionName.compare(StringRef("#_last_field")) == 0)
        {
            mlir::TypeSwitch<mlir::Type>(operands.front().getType())
                .Case<mlir_ts::ConstTupleType>([&](auto tupleType)
                {
                    result = builder.create<mlir_ts::ConstantOp>(location, builder.getI32Type(),
        builder.getI32IntegerAttr(tupleType.size()));
                })
                .Case<mlir_ts::TupleType>([&](auto tupleType)
                {
                    result = builder.create<mlir_ts::ConstantOp>(location, builder.getI32Type(),
        builder.getI32IntegerAttr(tupleType.size()));
                })
                .Default([&](auto type)
                {
                    llvm_unreachable("not implemented");
                    //result = builder.create<mlir_ts::ConstantOp>(location, builder.getI32Type(), builder.getI32IntegerAttr(-1));
                });

        }
        */
        else if (!allowPartialResolve)
        {
            emitError(location) << "no defined function found for '" << functionName << "'";
        }

        return result;
    }

  private:
    mlir::LogicalResult mlirGenPrint(const mlir::Location &location, ArrayRef<mlir::Value> operands)
    {
        auto printOp = builder.create<mlir_ts::PrintOp>(location, operands);

        return mlir::success();
    }

    mlir::LogicalResult mlirGenAssert(const mlir::Location &location, ArrayRef<mlir::Value> operands)
    {
        auto msg = StringRef("assert");
        if (operands.size() > 1)
        {
            auto param2 = operands[1];
            auto constantOp = dyn_cast_or_null<mlir_ts::ConstantOp>(param2.getDefiningOp());
            if (constantOp && constantOp.getType().isa<mlir_ts::StringType>())
            {
                msg = constantOp.value().cast<mlir::StringAttr>().getValue();
            }

            param2.getDefiningOp()->erase();
        }

        auto assertOp = builder.create<mlir_ts::AssertOp>(location, operands.front(), mlir::StringAttr::get(builder.getContext(), msg));

        return mlir::success();
    }

    mlir::Value mlirGenParseInt(const mlir::Location &location, ArrayRef<mlir::Value> operands)
    {
        auto op = operands.front();
        if (!op.getType().isa<mlir_ts::StringType>())
        {
            op = builder.create<mlir_ts::CastOp>(location, mlir_ts::StringType::get(builder.getContext()), op);
        }

        auto parseIntOp = builder.create<mlir_ts::ParseIntOp>(location, builder.getI32Type(), op);

        return parseIntOp;
    }

    mlir::Value mlirGenParseFloat(const mlir::Location &location, ArrayRef<mlir::Value> operands)
    {
        auto op = operands.front();
        if (!op.getType().isa<mlir_ts::StringType>())
        {
            op = builder.create<mlir_ts::CastOp>(location, mlir_ts::StringType::get(builder.getContext()), op);
        }

        auto parseFloatOp = builder.create<mlir_ts::ParseFloatOp>(location, builder.getF32Type(), op);

        return parseFloatOp;
    }

    mlir::Value mlirGenSizeOf(const mlir::Location &location, ArrayRef<mlir::Value> operands)
    {
        auto sizeOfValue =
            builder.create<mlir_ts::SizeOfOp>(location, builder.getI64Type(), mlir::TypeAttr::get(operands.front().getType()));

        return sizeOfValue;
    }
};

class MLIRCodeLogic
{
    mlir::OpBuilder &builder;

  public:
    MLIRCodeLogic(mlir::OpBuilder &builder) : builder(builder)
    {
    }

    mlir::Attribute ExtractAttr(mlir::Value value, bool removeOpIfSuccess = false)
    {
        auto constOp = dyn_cast_or_null<mlir_ts::ConstantOp>(value.getDefiningOp());
        if (constOp)
        {
            auto val = constOp.valueAttr();
            return val;
        }

        return mlir::Attribute();
    }

    mlir::Value GetReferenceOfLoadOp(mlir::Value value)
    {
        if (auto loadOp = dyn_cast_or_null<mlir_ts::LoadOp>(value.getDefiningOp()))
        {
            // this LoadOp will be removed later as unused
            auto refValue = loadOp.reference();
            return refValue;
        }

        return mlir::Value();
    }

    mlir::Attribute TupleFieldName(StringRef name)
    {
        assert(!name.empty());
        return mlir::StringAttr::get(builder.getContext(), name);
    }

    template <typename T>
    std::pair<int, mlir::Type> TupleFieldType(mlir::Location location, T tupleType, mlir::Attribute fieldId, bool indexAccess = false)
    {
        auto result = TupleFieldTypeNoError(location, tupleType, fieldId, indexAccess);
        if (result.first == -1)
        {
            emitError(location, "Tuple member '") << fieldId << "' can't be found";
        }

        return result;
    }

    template <typename T>
    std::pair<int, mlir::Type> TupleFieldTypeNoError(mlir::Location location, T tupleType, mlir::Attribute fieldId,
                                                     bool indexAccess = false)
    {
        auto fieldIndex = tupleType.getIndex(fieldId);
        if (indexAccess && (fieldIndex < 0 || fieldIndex >= tupleType.size()))
        {
            // try to resolve index
            auto intAttr = fieldId.dyn_cast_or_null<mlir::IntegerAttr>();
            if (intAttr)
            {
                fieldIndex = intAttr.getInt();
            }
        }

        if (fieldIndex < 0 || fieldIndex >= tupleType.size())
        {
            return std::make_pair<>(-1, mlir::Type());
        }

        // type
        auto elementType = tupleType.getType(fieldIndex);

        return std::make_pair(fieldIndex, elementType);
    }

    mlir::Type CaptureTypeStorage(llvm::StringMap<VariablePairT> &capturedVars)
    {
        SmallVector<mlir_ts::FieldInfo> fields;
        for (auto &varInfo : capturedVars)
        {
            auto &actualValue = varInfo.getValue().first;
            auto &val = varInfo.getValue().second;
            fields.push_back(mlir_ts::FieldInfo{TupleFieldName(val->getName()), actualValue.getType()});
        }

        auto lambdaType = mlir_ts::TupleType::get(builder.getContext(), fields);
        return lambdaType;
    }

    mlir::Type CaptureType(llvm::StringMap<VariablePairT> &capturedVars)
    {
        return mlir_ts::RefType::get(CaptureTypeStorage(capturedVars));
    }
};

class MLIRPropertyAccessCodeLogic
{
    mlir::OpBuilder &builder;
    mlir::Location &location;
    mlir::Value &expression;
    mlir::StringRef name;
    mlir::Attribute fieldId;

  public:
    MLIRPropertyAccessCodeLogic(mlir::OpBuilder &builder, mlir::Location &location, mlir::Value &expression, StringRef name)
        : builder(builder), location(location), expression(expression), name(name)
    {
        MLIRCodeLogic mcl(builder);
        fieldId = mcl.TupleFieldName(name);
    }

    MLIRPropertyAccessCodeLogic(mlir::OpBuilder &builder, mlir::Location &location, mlir::Value &expression, mlir::Attribute fieldId)
        : builder(builder), location(location), expression(expression), fieldId(fieldId)
    {
    }

    mlir::Value Enum(mlir_ts::EnumType enumType)
    {
        auto propName = getName();
        auto dictionaryAttr = getExprConstAttr().cast<mlir::DictionaryAttr>();
        auto valueAttr = dictionaryAttr.get(propName);
        if (!valueAttr)
        {
            emitError(location, "Enum member '") << propName << "' can't be found";
            return mlir::Value();
        }

        return builder.create<mlir_ts::ConstantOp>(location, enumType.getElementType(), valueAttr);
    }

    template <typename T> mlir::Value Tuple(T tupleType, bool indexAccess = false)
    {
        mlir::Value value;

        MLIRCodeLogic mcl(builder);

        // resolve index
        auto pair = mcl.TupleFieldType(location, tupleType, fieldId, indexAccess);
        auto fieldIndex = pair.first;
        auto elementType = pair.second;

        if (fieldIndex < 0)
        {
            return value;
        }

        auto refValue = getExprLoadRefValue();
        if (refValue)
        {
            auto propRef = builder.create<mlir_ts::PropertyRefOp>(location, mlir_ts::RefType::get(elementType), refValue,
                                                                  builder.getI32IntegerAttr(fieldIndex));

            return builder.create<mlir_ts::LoadOp>(location, elementType, propRef);
        }

        MLIRTypeHelper mth(builder.getContext());
        return builder.create<mlir_ts::ExtractPropertyOp>(location, elementType, expression,
                                                          builder.getArrayAttr(mth.getStructIndexAttrValue(fieldIndex)));
    }

    template <typename T> mlir::Value TupleNoError(T tupleType, bool indexAccess = false)
    {
        mlir::Value value;

        MLIRCodeLogic mcl(builder);

        // resolve index
        auto pair = mcl.TupleFieldTypeNoError(location, tupleType, fieldId, indexAccess);
        auto fieldIndex = pair.first;
        auto elementType = pair.second;

        if (fieldIndex < 0)
        {
            return value;
        }

        auto refValue = getExprLoadRefValue();
        if (refValue)
        {
            auto propRef = builder.create<mlir_ts::PropertyRefOp>(location, mlir_ts::RefType::get(elementType), refValue,
                                                                  builder.getI32IntegerAttr(fieldIndex));

            return builder.create<mlir_ts::LoadOp>(location, elementType, propRef);
        }

        MLIRTypeHelper mth(builder.getContext());
        return builder.create<mlir_ts::ExtractPropertyOp>(location, elementType, expression,
                                                          builder.getArrayAttr(mth.getStructIndexAttrValue(fieldIndex)));
    }

    mlir::Value Bool(mlir_ts::BooleanType intType)
    {
        auto propName = getName();
        if (propName == "toString")
        {
            return builder.create<mlir_ts::CastOp>(location, mlir_ts::StringType::get(builder.getContext()), expression);
        }
        else
        {
            llvm_unreachable("not implemented");
        }
    }

    mlir::Value Int(mlir::IntegerType intType)
    {
        auto propName = getName();
        if (propName == "toString")
        {
            return builder.create<mlir_ts::CastOp>(location, mlir_ts::StringType::get(builder.getContext()), expression);
        }
        else
        {
            llvm_unreachable("not implemented");
        }
    }

    mlir::Value Float(mlir::FloatType intType)
    {
        auto propName = getName();
        if (propName == "toString")
        {
            return builder.create<mlir_ts::CastOp>(location, mlir_ts::StringType::get(builder.getContext()), expression);
        }
        else
        {
            llvm_unreachable("not implemented");
        }
    }

    mlir::Value String(mlir_ts::StringType stringType)
    {
        auto propName = getName();
        if (propName == "length")
        {
            return builder.create<mlir_ts::StringLengthOp>(location, builder.getI32Type(), expression);
        }
        else
        {
            llvm_unreachable("not implemented");
        }
    }

    template <typename T> mlir::Value Array(T arrayType)
    {
        auto propName = getName();
        if (propName == "length")
        {
            if (expression.getType().isa<mlir_ts::ConstArrayType>())
            {
                auto size = getExprConstAttr().cast<mlir::ArrayAttr>().size();
                return builder.create<mlir_ts::ConstantOp>(location, builder.getI32Type(), builder.getI32IntegerAttr(size));
            }
            else if (expression.getType().isa<mlir_ts::ArrayType>())
            {
                auto sizeValue = builder.create<mlir_ts::LengthOfOp>(location, builder.getI32Type(), expression);
                return sizeValue;
            }
            else
            {
                llvm_unreachable("not implemented");
            }
        }
        else
        {
            llvm_unreachable("not implemented");
        }
    }

    template <typename T> mlir::Value Ref(T refType)
    {
        if (auto tupleType = refType.getElementType().template dyn_cast_or_null<mlir_ts::TupleType>())
        {
            return Ref(tupleType);
        }
        else
        {
            llvm_unreachable("not implemented");
        }
    }

    mlir::Value Ref(mlir_ts::TupleType tupleType)
    {
        MLIRCodeLogic mcl(builder);

        // resolve index
        auto pair = mcl.TupleFieldType(location, tupleType, fieldId);
        auto fieldIndex = pair.first;
        auto elementType = pair.second;

        if (fieldIndex < 0)
        {
            return mlir::Value();
        }

        // LLVM_DEBUG(llvm::dbgs() << "property ref access: " << expression << " index:" << fieldIndex << " field type: " << elementType
        // << "\n");

        auto propRef = builder.create<mlir_ts::PropertyRefOp>(location, mlir_ts::RefType::get(elementType), expression,
                                                              builder.getI32IntegerAttr(fieldIndex));

        return builder.create<mlir_ts::LoadOp>(location, elementType, propRef);
    }

    mlir::Value Class(mlir_ts::ClassType classType)
    {
        if (auto classStorageType = classType.getStorageType().template dyn_cast_or_null<mlir_ts::ClassStorageType>())
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
        auto pair = mcl.TupleFieldTypeNoError(location, classStorageType, fieldId);
        auto fieldIndex = pair.first;
        auto elementType = pair.second;

        if (fieldIndex < 0)
        {
            return mlir::Value();
        }

        // LLVM_DEBUG(llvm::dbgs() << "property ref access: " << expression << " index:" << fieldIndex << " field type: " << elementType
        // << "\n");

        auto propRef = builder.create<mlir_ts::PropertyRefOp>(location, mlir_ts::RefType::get(elementType), expression,
                                                              builder.getI32IntegerAttr(fieldIndex));

        return builder.create<mlir_ts::LoadOp>(location, elementType, propRef);
    }

  private:
    StringRef getName()
    {
        return name;
    }

    mlir::Attribute getExprConstAttr()
    {
        MLIRCodeLogic mcl(builder);

        auto value = mcl.ExtractAttr(expression);
        if (!value)
        {
            llvm_unreachable("not implemented");
        }

        return value;
    }

    mlir::Value getExprLoadRefValue()
    {
        MLIRCodeLogic mcl(builder);
        auto value = mcl.GetReferenceOfLoadOp(expression);
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

#endif // MLIR_TYPESCRIPT_MLIRGENLOGIC_H_
