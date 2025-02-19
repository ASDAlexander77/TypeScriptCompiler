#ifndef MLIR_TYPESCRIPT_LOWERTOLLVMLOGIC_LLVMTYPECONVERTERHELPER_H_
#define MLIR_TYPESCRIPT_LOWERTOLLVMLOGIC_LLVMTYPECONVERTERHELPER_H_

#include "TypeScript/Config.h"
#include "TypeScript/Defines.h"
#include "TypeScript/Passes.h"
#include "TypeScript/TypeScriptDialect.h"
#include "TypeScript/TypeScriptOps.h"

#include "mlir/Analysis/DataLayoutAnalysis.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Target/LLVMIR/TypeToLLVM.h"

#define DEBUG_TYPE "llvm"

using namespace mlir;
namespace mlir_ts = mlir::typescript;

namespace typescript
{

static llvm::LLVMContext &getGlobalContext() 
{
    static llvm::LLVMContext GlobalContext;
    return GlobalContext;
}

class LLVMTypeConverterHelper
{
  public:
    LLVMTypeConverterHelper(const LLVMTypeConverter *typeConverter) : typeConverter(typeConverter)
    {
    }

    mlir::Type getIntPtrType(unsigned addressSpace)
    {
        return mlir::IntegerType::get(&typeConverter->getContext(), typeConverter->getPointerBitwidth(addressSpace));
    }

    unsigned getPointerBitwidth(unsigned addressSpace)
    {
        return typeConverter->getPointerBitwidth(addressSpace);
    }

    mlir::Type getConvertedIndexType()
    {
        return typeConverter->getIndexType();
    }

    uint64_t getTypeAllocSizeInBits(mlir::Type type)
    {
        LLVM::TypeToLLVMIRTranslator typeToLLVMIRTranslator(getGlobalContext());
        auto llvmType = typeToLLVMIRTranslator.translateType(type);
        return  typeConverter->getDataLayout().getTypeAllocSize(llvmType) << 3;
    }

    uint64_t getTypeAlignSizeInBits(mlir::Type type)
    {
        LLVM::TypeToLLVMIRTranslator typeToLLVMIRTranslator(getGlobalContext());
        auto llvmType = typeToLLVMIRTranslator.translateType(type);
        return typeConverter->getDataLayout().getABITypeAlign(llvmType).value() << 3;
    }    

    uint64_t getStructTypeSize(LLVM::LLVMStructType structType)
    {
        uint64_t size = 0;

        auto structLayout = getStructLayout(structType);
        size = structLayout->getSizeInBytes();

        LLVM_DEBUG(llvm::dbgs() << "\n!! struct type: " << structType 
                        << "\n estimated size: " << size << "\n";);

        return size;
    }
        
    
    uint64_t TypeAllocSizeInBits(mlir::Type llvmType)
    {
        return getTypeAllocSizeInBytes(llvmType) << 3;
    }

    uint64_t getTypeAllocSizeInBytes(mlir::Type llvmType)
    {
        if (llvmType == LLVM::LLVMVoidType::get(llvmType.getContext()))
        {
            return 0;
        }

        LLVM::TypeToLLVMIRTranslator typeToLLVMIRTranslator(getGlobalContext());
        auto type = typeToLLVMIRTranslator.translateType(llvmType);

        LLVM_DEBUG(llvm::dbgs() << "\n!! checking type size - LLVM: " << llvmType << " and IR: " << *type << "\n";);

        // if (auto structData = dyn_cast<LLVM::LLVMStructType>(llvmType))
        // {
        //     auto layout = typeConverter->getDataLayout().getStructLayout(cast<llvm::StructType>(type));
            
        //     LLVM_DEBUG(llvm::dbgs() << "\n!! src type: " << llvmType
        //                     << "\n size: " << layout->getSizeInBytes() << " alignment: " << layout->getAlignment().value() << "\n";);

        //     assert(typeConverter->getDataLayout().getTypeAllocSize(type) == layout->getSizeInBytes());
        //     return layout->getSizeInBytes();
        // }        

        auto typeSize = typeConverter->getDataLayout().getTypeAllocSize(type);
        
        LLVM_DEBUG(llvm::dbgs() << "\n!! src type: " << llvmType
                        << "\n size: " << typeSize << "\n";);

        return typeSize;
    }

    const llvm::StructLayout* getStructLayout(LLVM::LLVMStructType structData)
    {
        LLVM::TypeToLLVMIRTranslator typeToLLVMIRTranslator(getGlobalContext());
        auto type = typeToLLVMIRTranslator.translateType(structData);      
        auto layout = typeConverter->getDataLayout().getStructLayout(cast<llvm::StructType>(type));  
        return layout;
    }    

    mlir::Type findMaxSizeType(mlir_ts::UnionType unionType)
    {
        auto currentSize = 0;
        mlir::Type selectedType;
        for (auto subType : unionType.getTypes())
        {
            auto converted = typeConverter->convertType(subType);
            auto typeSize = getTypeAllocSizeInBytes(converted);
            if (typeSize > currentSize)
            {
                selectedType = converted;
                currentSize = typeSize;
            }
        }

        return selectedType;
    }

    const LLVMTypeConverter *typeConverter;
};
} // namespace typescript

#undef DEBUG_TYPE

#endif // MLIR_TYPESCRIPT_LOWERTOLLVMLOGIC_LLVMTYPECONVERTERHELPER_H_
