#ifndef MLIR_TYPESCRIPT_COMMONGENLOGIC_H_
#define MLIR_TYPESCRIPT_COMMONGENLOGIC_H_

#include "parser.h"
#include "file_helper.h"

namespace mlir_ts = mlir::typescript;

namespace typescript
{
    class MLIRHelper
    {
    public:
        static std::string getName(ts::Node name)
        {
            std::string nameValue;
            if (name == SyntaxKind::Identifier)
            {
                nameValue = wstos(name.as<ts::Identifier>()->escapedText);
            }

            if (name == SyntaxKind::StringLiteral)
            {
                nameValue = wstos(name.as<ts::StringLiteral>()->text);
            }

            return nameValue;
        }

        static mlir::StringRef getName(ts::Node name, llvm::BumpPtrAllocator &stringAllocator)
        {
            auto nameValue = getName(name);
            return mlir::StringRef(nameValue).copy(stringAllocator);
        }        
    };

    class MLIRTypeHelper
    {
        mlir::MLIRContext *context;
    public:        
        MLIRTypeHelper(mlir::MLIRContext *context) : context(context) {}

        mlir::Type getI32Type()
        {
            return mlir::IntegerType::get(context, 32);
        }
        
        mlir::IntegerAttr getStructIndexAttrValue(int32_t value)
        {
            return mlir::IntegerAttr::get(getI32Type(), mlir::APInt(32, value));
        }        

        bool isValueType(mlir::Type type)
        {
            return type && (type.isIntOrIndexOrFloat() || type.isa<mlir_ts::TupleType>());
        }

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
