#ifndef MLIR_TYPESCRIPT_LOWERTOLLVMLOGIC_TYPECONVERTERHELPER_H_
#define MLIR_TYPESCRIPT_LOWERTOLLVMLOGIC_TYPECONVERTERHELPER_H_

#include "TypeScript/Config.h"
#include "TypeScript/Defines.h"
#include "TypeScript/Passes.h"
#include "TypeScript/TypeScriptDialect.h"
#include "TypeScript/TypeScriptOps.h"

#include "mlir/Target/LLVMIR/TypeTranslation.h"
#include "mlir/Transforms/DialectConversion.h"

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

    Type convertType(Type type)
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

    Type makePtrToValue(Type type)
    {
        if (auto constArray = type.dyn_cast_or_null<mlir_ts::ConstArrayType>())
        {
            return LLVM::LLVMPointerType::get(LLVM::LLVMArrayType::get(convertType(constArray.getElementType()), constArray.getSize()));
        }

        llvm_unreachable("not implemented");
    }
};
} // namespace typescript

#endif // MLIR_TYPESCRIPT_LOWERTOLLVMLOGIC_TYPECONVERTERHELPER_H_
