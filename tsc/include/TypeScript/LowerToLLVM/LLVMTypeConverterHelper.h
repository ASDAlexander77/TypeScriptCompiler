#ifndef MLIR_TYPESCRIPT_LOWERTOLLVMLOGIC_LLVMTYPECONVERTERHELPER_H_
#define MLIR_TYPESCRIPT_LOWERTOLLVMLOGIC_LLVMTYPECONVERTERHELPER_H_

#include "TypeScript/Config.h"
#include "TypeScript/Defines.h"
#include "TypeScript/Passes.h"
#include "TypeScript/TypeScriptDialect.h"
#include "TypeScript/TypeScriptOps.h"

#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVM.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Target/LLVMIR/TypeToLLVM.h"

using namespace mlir;
namespace mlir_ts = mlir::typescript;

namespace typescript
{

class LLVMTypeConverterHelper
{
    LLVMTypeConverter &typeConverter;

  public:
    LLVMTypeConverterHelper(LLVMTypeConverter &typeConverter) : typeConverter(typeConverter)
    {
    }

    mlir::Type getIntPtrType(unsigned addressSpace)
    {
        return mlir::IntegerType::get(&typeConverter.getContext(), typeConverter.getPointerBitwidth(addressSpace));
    }

    int32_t getPointerBitwidth(unsigned addressSpace)
    {
        return typeConverter.getPointerBitwidth(addressSpace);
    }

    int32_t getTypeSize(mlir::Type llvmType)
    {
        llvm::LLVMContext llvmContext;
        LLVM::TypeToLLVMIRTranslator typeToLLVMIRTranslator(llvmContext);

        if (llvmType == LLVM::LLVMVoidType::get(llvmType.getContext()))
        {
            return 0;
        }

        auto type = typeToLLVMIRTranslator.translateType(llvmType);
        return typeConverter.getDataLayout().getTypeAllocSize(type);
    }

    mlir::Type findMaxSizeType(mlir_ts::UnionType unionType)
    {
        auto currentSize = 0;
        mlir::Type selectedType;
        for (auto subType : unionType.getTypes())
        {
            auto converted = typeConverter.convertType(subType);
            auto typeSize = getTypeSize(converted);
            if (typeSize > currentSize)
            {
                selectedType = converted;
                currentSize = typeSize;
            }
        }

        if (selectedType.isa<LLVM::LLVMPointerType>())
        {
            auto *context = &typeConverter.getContext();
            return LLVM::LLVMPointerType::get(mlir::IntegerType::get(context, 8));
        }

        return selectedType;
    }
};
} // namespace typescript

#endif // MLIR_TYPESCRIPT_LOWERTOLLVMLOGIC_LLVMTYPECONVERTERHELPER_H_
