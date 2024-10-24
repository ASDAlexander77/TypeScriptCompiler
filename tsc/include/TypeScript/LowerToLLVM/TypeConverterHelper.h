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
    const TypeConverter *typeConverter;

    TypeConverterHelper(const TypeConverter *typeConverter) : typeConverter(typeConverter)
    {
        assert(typeConverter);
    }

    mlir::Type convertType(mlir::Type type)
    {
        if (type)
        {
            if (auto convertedType = typeConverter->convertType(type))
            {
                return convertedType;
            }
        }

        return type;
    }

    int getIndexTypeBitwidth()
    {
        return ((mlir::LLVMTypeConverter *)typeConverter)->getIndexTypeBitwidth();
    }
};
} // namespace typescript

#endif // MLIR_TYPESCRIPT_LOWERTOLLVMLOGIC_TYPECONVERTERHELPER_H_
