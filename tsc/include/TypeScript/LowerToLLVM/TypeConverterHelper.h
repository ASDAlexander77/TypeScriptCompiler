#ifndef MLIR_TYPESCRIPT_LOWERTOLLVMLOGIC_TYPECONVERTERHELPER_H_
#define MLIR_TYPESCRIPT_LOWERTOLLVMLOGIC_TYPECONVERTERHELPER_H_

#include "TypeScript/Config.h"
#include "TypeScript/Defines.h"
#include "TypeScript/Passes.h"
#include "TypeScript/TypeScriptDialect.h"
#include "TypeScript/TypeScriptOps.h"

#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"

using namespace mlir;
namespace mlir_ts = mlir::typescript;

namespace typescript
{
class TypeConverterHelper
{
  public:
    TypeConverter &typeConverter;

    TypeConverterHelper(TypeConverter *typeConverter) : typeConverter(*typeConverter)
    {
        assert(typeConverter);
    }

    mlir::Type convertType(mlir::Type type)
    {
        if (type)
        {
            if (auto convertedType = typeConverter.convertType(type))
            {
                return convertedType;
            }
        }

        return type;
    }

    mlir::Type makePtrToValue(mlir::Type type)
    {
        if (auto constArray = type.dyn_cast<mlir_ts::ConstArrayType>())
        {
            return LLVM::LLVMPointerType::get(LLVM::LLVMArrayType::get(convertType(constArray.getElementType()), constArray.getSize()));
        }

        llvm_unreachable("not implemented");
    }

    int getIndexTypeBitwidth()
    {
        return (*(mlir::LLVMTypeConverter *)&typeConverter).getIndexTypeBitwidth();
    }
};
} // namespace typescript

#endif // MLIR_TYPESCRIPT_LOWERTOLLVMLOGIC_TYPECONVERTERHELPER_H_
