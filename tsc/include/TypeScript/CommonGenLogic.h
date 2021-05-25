#ifndef MLIR_TYPESCRIPT_COMMONGENLOGIC_H_
#define MLIR_TYPESCRIPT_COMMONGENLOGIC_H_

namespace mlir_ts = mlir::typescript;

namespace typescript
{
    class MLIRTypeHelper
    {
        mlir::MLIRContext *context;
    public:        
        MLIRTypeHelper(mlir::MLIRContext *context) : context(context) {}

        mlir::Type convertConstTypeToType(mlir::Type type, bool &copyRequired)
        {
            if (auto constArrayType = type.dyn_cast_or_null<mlir_ts::ConstArrayType>())
            {
                copyRequired = true;
                return mlir_ts::ArrayType::get(constArrayType.getElementType());
            }

            /*
            // tuple is value and copied already
            if (auto constTupleType = type.dyn_cast_or_null<mlir_ts::ConstTupleType>())
            {
                copyRequired = true;
                return mlir_ts::TupleType::get(context, constTupleType.getFields());
            }
            */

            copyRequired = false;
            return type;
        }
    };
}

#endif // MLIR_TYPESCRIPT_COMMONGENLOGIC_H_
