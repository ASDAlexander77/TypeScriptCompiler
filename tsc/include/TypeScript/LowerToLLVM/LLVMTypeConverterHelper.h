#ifndef MLIR_TYPESCRIPT_LOWERTOLLVMLOGIC_LLVMTYPECONVERTERHELPER_H_
#define MLIR_TYPESCRIPT_LOWERTOLLVMLOGIC_LLVMTYPECONVERTERHELPER_H_

#ifndef DEBUG_TYPE
#define DEBUG_TYPE "llvm"
#endif

#include "TypeScript/Config.h"
#include "TypeScript/Defines.h"
#include "TypeScript/Passes.h"
#include "TypeScript/TypeScriptDialect.h"
#include "TypeScript/TypeScriptOps.h"

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

    unsigned getPointerBitwidth(unsigned addressSpace)
    {
        return typeConverter.getPointerBitwidth(addressSpace);
    }

    uint64_t getStructTypeSizeNonAligned(LLVM::LLVMStructType structType)
    {
        uint64_t size = 0;

        for (auto subType : structType.getBody())
        {
            size += getTypeSizeEstimateInBytes(subType);
        }

        LLVM_DEBUG(llvm::dbgs() << "\n!! struct type: " << structType 
                        << "\n estimated size: " << size << "\n";);

        return size;
    }

    uint64_t getTypeSizeEstimateInBytes(mlir::Type llvmType)
    {
        if (llvmType == LLVM::LLVMVoidType::get(llvmType.getContext()))
        {
            return 0;
        }

        if (auto structData = llvmType.dyn_cast<LLVM::LLVMStructType>())
        {
            return getStructTypeSizeNonAligned(structData);
        }        

        llvm::LLVMContext llvmContext;
        LLVM::TypeToLLVMIRTranslator typeToLLVMIRTranslator(llvmContext);

        auto type = typeToLLVMIRTranslator.translateType(llvmType);
        uint64_t typeSize = typeConverter.getDataLayout().getTypeAllocSize(type);
        
        LLVM_DEBUG(llvm::dbgs() << "\n!! src type: " << llvmType
                        << "\n size: " << typeSize << "\n";);

        return typeSize;
    }

    mlir::Type findMaxSizeType(mlir_ts::UnionType unionType)
    {
        auto currentSize = 0;
        mlir::Type selectedType;
        for (auto subType : unionType.getTypes())
        {
            auto converted = typeConverter.convertType(subType);
            auto typeSize = getTypeSizeEstimateInBytes(converted);
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
