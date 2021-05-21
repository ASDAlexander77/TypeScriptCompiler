#ifndef MLIR_TYPESCRIPT_MLIRGENLOGIC_H_
#define MLIR_TYPESCRIPT_MLIRGENLOGIC_H_

#include "TypeScript/TypeScriptDialect.h"
#include "TypeScript/TypeScriptOps.h"

#include "mlir/IR/Types.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"

#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/TypeSwitch.h"

#include <numeric>

namespace mlir_ts = mlir::typescript;

using llvm::ArrayRef;
using llvm::cast;
using llvm::dyn_cast;
using llvm::dyn_cast_or_null;
using llvm::isa;
using llvm::makeArrayRef;
using llvm::ScopedHashTableScope;
using llvm::SmallVector;
using llvm::StringRef;
using llvm::Twine;

namespace typescript
{
    class MLIRCustomMethods
    {
        mlir::OpBuilder &builder;
        mlir::Location &location;

    public:        
        MLIRCustomMethods(mlir::OpBuilder &builder, mlir::Location &location) 
            : builder(builder), location(location) {}   

        mlir::Value callMethod(StringRef functionName, ArrayRef<mlir::Value> operands, bool allowPartialResolve)
        {
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
            else 
            if (functionName.compare(StringRef("parseFloat")) == 0)
            {
                result = mlirGenParseFloat(location, operands);
            }
            else 
            if (!allowPartialResolve)
            {
                emitError(location) << "no defined function found for '" << functionName << "'";
            }

            return result;
        }     

    private:
        mlir::LogicalResult mlirGenPrint(const mlir::Location &location, ArrayRef<mlir::Value> operands)
        {
            auto printOp =
                builder.create<mlir_ts::PrintOp>(
                    location,
                    operands);

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

            auto assertOp =
                builder.create<mlir_ts::AssertOp>(
                    location,
                    operands.front(),
                    mlir::StringAttr::get(msg, builder.getContext()));

            return mlir::success();
        }

        mlir::Value mlirGenParseInt(const mlir::Location &location, ArrayRef<mlir::Value> operands)
        {
            auto op = operands.front();
            if (!op.getType().isa<mlir_ts::StringType>())
            {
                op = builder.create<mlir_ts::CastOp>(location, mlir_ts::StringType::get(builder.getContext()), op);
            }

            auto parseIntOp =
                builder.create<mlir_ts::ParseIntOp>(
                    location,
                    builder.getI32Type(),
                    op);

            return parseIntOp;
        }

        mlir::Value mlirGenParseFloat(const mlir::Location &location, ArrayRef<mlir::Value> operands)
        {
            auto op = operands.front();
            if (!op.getType().isa<mlir_ts::StringType>())
            {
                op = builder.create<mlir_ts::CastOp>(location, mlir_ts::StringType::get(builder.getContext()), op);
            }

            auto parseFloatOp =
                builder.create<mlir_ts::ParseFloatOp>(
                    location,
                    builder.getF32Type(),
                    op);

            return parseFloatOp;
        }
    };

    class MLIRPropertyAccessCodeLogic
    {
        mlir::OpBuilder &builder;
        mlir::Location &location;
        mlir::Value &expression;
        mlir::Value &name;
    public:        
        MLIRPropertyAccessCodeLogic(mlir::OpBuilder &builder, mlir::Location &location, mlir::Value &expression, mlir::Value &name) 
            : builder(builder), location(location), expression(expression), name(name) {}

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

        mlir::Value Tuple(mlir_ts::TupleType tupleType)
        {
            mlir::Value value;

            auto propName = getName();

            auto fieldIndex = tupleType.getIndex(propName);
            if (fieldIndex < 0)
            {
                emitError(location, "Tuple member '") << propName << "' can't be found";
                return value;
            }

            auto elementType = tupleType.getType(fieldIndex);

            auto refValue = getExprLoadRefValue();

            auto propRef = builder.create<mlir_ts::PropertyRefOp>(
                location, 
                mlir_ts::RefType::get(elementType), 
                refValue, 
                builder.getI32IntegerAttr(fieldIndex));

            return builder.create<mlir_ts::LoadOp>(location, elementType, propRef);
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

        mlir::Value Array(mlir_ts::ArrayType arrayType)
        {
            auto propName = getName();
            if (propName == "length")
            {
                auto size = getExprConstAttr().cast<mlir::ArrayAttr>().size();
                return builder.create<mlir_ts::ConstantOp>(location, builder.getI32Type(), builder.getI32IntegerAttr(size));                            
            }
            else
            {
                llvm_unreachable("not implemented");                        
            }                         
        }           

    private:
        StringRef getName() 
        {
            auto symRef = dyn_cast_or_null<mlir_ts::SymbolRefOp>(name.getDefiningOp());
            if (symRef)
            {
                auto value = symRef.identifier();
                symRef->erase();
                return value;
            }

            llvm_unreachable("not implemented");                        
        }        

        mlir::Attribute getExprConstAttr()
        {
            if (auto constOp = dyn_cast_or_null<mlir_ts::ConstantOp>(expression.getDefiningOp()))
            {
                auto value = constOp.getValue();
                constOp->erase();
                return value;
            }
            else
            {
                llvm_unreachable("not implemented");                        
            }  
        }

        mlir::Value getExprLoadRefValue()
        {
            if (auto loadOp = dyn_cast_or_null<mlir_ts::LoadOp>(expression.getDefiningOp()))
            {
                auto refValue = loadOp.reference();
                loadOp->erase();
                return refValue;
            }
            else
            {
                llvm_unreachable("not implemented");            
            }
        }        
    };

    class MLIRLogicHelper
    {
    public:
        static bool isNeededToSaveData(SyntaxKind &opCode)
        {
            switch (opCode)
            {
                case SyntaxKind::PlusEqualsToken: opCode = SyntaxKind::PlusToken; break;
                case SyntaxKind::MinusEqualsToken: opCode = SyntaxKind::MinusToken; break;
                case SyntaxKind::AsteriskEqualsToken: opCode = SyntaxKind::AsteriskToken; break;
                case SyntaxKind::AsteriskAsteriskEqualsToken: opCode = SyntaxKind::AsteriskAsteriskToken; break;
                case SyntaxKind::SlashEqualsToken: opCode = SyntaxKind::SlashToken; break;
                case SyntaxKind::PercentEqualsToken: opCode = SyntaxKind::PercentToken; break;
                case SyntaxKind::LessThanLessThanEqualsToken: opCode = SyntaxKind::LessThanLessThanToken; break;
                case SyntaxKind::GreaterThanGreaterThanEqualsToken: opCode = SyntaxKind::GreaterThanGreaterThanToken; break;
                case SyntaxKind::GreaterThanGreaterThanGreaterThanEqualsToken: opCode = SyntaxKind::GreaterThanGreaterThanGreaterThanToken; break;
                case SyntaxKind::AmpersandEqualsToken: opCode = SyntaxKind::AmpersandToken; break;
                case SyntaxKind::BarEqualsToken: opCode = SyntaxKind::BarToken; break;
                case SyntaxKind::BarBarEqualsToken: opCode = SyntaxKind::BarBarToken; break;
                case SyntaxKind::AmpersandAmpersandEqualsToken: opCode = SyntaxKind::AmpersandAmpersandToken; break;
                case SyntaxKind::QuestionQuestionEqualsToken: opCode = SyntaxKind::QuestionQuestionToken; break;
                case SyntaxKind::CaretEqualsToken: opCode = SyntaxKind::CaretToken; break;
                default: return false; break;
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
}

#endif // MLIR_TYPESCRIPT_MLIRGENLOGIC_H_
