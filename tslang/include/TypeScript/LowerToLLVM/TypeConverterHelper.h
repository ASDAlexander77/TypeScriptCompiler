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

    LLVM::LLVMFunctionType convertFunctionSignature(mlir::MLIRContext *context, TypeRange results, TypeRange args, bool isVariadic)
    {
        mlir::TypeConverter::SignatureConversion convResult(results.size());
        if (mlir::failed(typeConverter->convertSignatureArgs(results, convResult, 0)))
        {
            return {};
        }

        SmallVector<mlir::Type> convArgs;
        for (auto arg : args)
        {
            // a `never` parameter: a byte holds its place, as in FuncOpLowering
            auto convertedType = isa<mlir_ts::NeverType>(arg) ? mlir::IntegerType::get(context, 8) : typeConverter->convertType(arg);
            if (!convertedType)
            {
                return {};
            }

            convArgs.push_back(convertedType);
        }

        auto ret = convResult.getConvertedTypes().size() > 0 ?  convResult.getConvertedTypes().front() : LLVM::LLVMVoidType::get(context);

        return LLVM::LLVMFunctionType::get(context, ret, convArgs, isVariadic);
    }

    int getIndexTypeBitwidth()
    {
        return ((const mlir::LLVMTypeConverter *)typeConverter)->getIndexTypeBitwidth();
    }
};
} // namespace typescript

#endif // MLIR_TYPESCRIPT_LOWERTOLLVMLOGIC_TYPECONVERTERHELPER_H_
