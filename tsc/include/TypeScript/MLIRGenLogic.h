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
            mlir::Value value;

            auto propName = getName();
            if (propName == "length")
            {
                value = builder.create<mlir_ts::StringLengthOp>(location, builder.getI32Type(), expression);                            
            }
            else
            {
                llvm_unreachable("not implemented");                        
            }                         

            return value;
        }    

        mlir::Value Array(mlir_ts::ArrayType arrayType)
        {
            mlir::Value value;

            auto propName = getName();
            if (propName == "length")
            {
                auto size = getExprConstAttr().cast<mlir::ArrayAttr>().size();
                value = builder.create<mlir_ts::ConstantOp>(location, builder.getI32Type(), builder.getI32IntegerAttr(size));                            
            }
            else
            {
                llvm_unreachable("not implemented");                        
            }                         

            return value;
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
}

#endif // MLIR_TYPESCRIPT_MLIRGENLOGIC_H_
