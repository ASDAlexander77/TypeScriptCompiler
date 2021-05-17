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
    public:        
        MLIRPropertyAccessCodeLogic(mlir::OpBuilder &builder) : builder(builder) {}

        mlir::Value Enum(mlir::Location location, mlir::Value expression, mlir::Value name)
        {
            mlir::Value value;

            auto constOp = dyn_cast_or_null<mlir_ts::ConstantOp>(expression.getDefiningOp());
            auto dictionaryAttr = constOp.getValue().cast<mlir::DictionaryAttr>();

            auto symRef = dyn_cast_or_null<mlir_ts::SymbolRefOp>(name.getDefiningOp());
            if (symRef)
            {
                auto valueAttr = dictionaryAttr.get(symRef.identifier());
                if (!valueAttr)
                {
                    emitError(location, "Enum member '") << symRef.identifier() << "' can't be found";
                    return value;
                }

                value = builder.create<mlir_ts::ConstantOp>(location, expression.getType().cast<mlir_ts::EnumType>().getElementType(), valueAttr);

                symRef->erase();
                constOp->erase();
            }
            else
            {
                llvm_unreachable("not implemented");                        
            }

            return value;
        }
    };
}

#endif // MLIR_TYPESCRIPT_MLIRGENLOGIC_H_
