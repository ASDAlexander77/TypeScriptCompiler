#ifndef MLIR_TYPESCRIPT_LOWERTOLLVMLOGIC_LLVMDEBUGINFO_H_
#define MLIR_TYPESCRIPT_LOWERTOLLVMLOGIC_LLVMDEBUGINFO_H_

#include "TypeScript/LowerToLLVM/LocationHelper.h"

#include "llvm/DebugInfo/DWARF/DWARFUnit.h"

using namespace mlir;
namespace mlir_ts = mlir::typescript;

namespace typescript
{

class LLVMDebugInfoHelper
{
  public:
    LLVMDebugInfoHelper(MLIRContext *context, LLVMTypeConverterHelper llvmtch) : context(context), llvmtch(llvmtch)
    {
    }

    LLVM::DITypeAttr getDIType(mlir::Type type, LLVM::DIFileAttr file, uint32_t line, LLVM::DIScopeAttr scope)
    {
        LLVM::DITypeAttr diTypeAttr;

        mlir::TypeSwitch<mlir::Type>(type)
            .Case<mlir::IntegerType>([&](auto intType) {  
                auto typeCode = dwarf::DW_ATE_signed;
                StringRef typeName = "int";
                auto size = intType.getIntOrFloatBitWidth(); 
                if (size == 8)
                {
                    typeName = "char";
                    typeCode = dwarf::DW_ATE_signed_char;
                }
                else if (size == 1)
                {
                    typeName = "bool";
                    typeCode = dwarf::DW_ATE_boolean;
                    size = 8;
                }

                diTypeAttr = LLVM::DIBasicTypeAttr::get(context, dwarf::DW_TAG_base_type, StringAttr::get(context, typeName), size, typeCode);
            })
            .Case<mlir::FloatType>([&](auto floatType) {  
                StringRef typeName =  "double";                
                if (floatType.getIntOrFloatBitWidth() <= 32)
                {
                    typeName =  "float";                
                }

                diTypeAttr = LLVM::DIBasicTypeAttr::get(context, dwarf::DW_TAG_base_type, StringAttr::get(context, typeName), floatType.getIntOrFloatBitWidth(), dwarf::DW_ATE_float);
            })
            .Case<LLVM::LLVMStructType>([&](auto structType) {  
                auto sizeInBits = 0;
                auto alignInBits = 0;
                auto offsetInBits = 0;

                llvm::SmallVector<LLVM::DINodeAttr> elements;
                for (auto elementType : structType.getBody())
                {
                    sizeInBits = llvmtch.getTypeSizeEstimateInBytes(elementType) * 8; // size of element

                    auto elementDiType = getDIType(elementType, file, line, scope);
                    auto wrapperDiType = LLVM::DIDerivedTypeAttr::get(context, dwarf::DW_TAG_member, StringAttr::get(context, "member"), elementDiType, sizeInBits, alignInBits, offsetInBits);
                    elements.push_back(wrapperDiType);

                    offsetInBits += sizeInBits;
                }

                sizeInBits = offsetInBits;

                diTypeAttr = LLVM::DICompositeTypeAttr::get(context, dwarf::DW_TAG_structure_type, StringAttr::get(context, "struct"), 
                    file, line, scope, LLVM::DITypeAttr(), LLVM::DIFlags::TypePassByValue, sizeInBits, alignInBits, elements);
            })
            .Case<LLVM::LLVMPointerType>([&](auto pointerType) {  
                auto sizeInBits = 64;
                auto alignInBits = 32;
                auto offsetInBits = 0;

                diTypeAttr = LLVM::DIDerivedTypeAttr::get(
                    context, dwarf::DW_TAG_pointer_type, StringAttr::get(context, "pointer"), getDIType(pointerType.getElementType(), file, line, scope), 
                    sizeInBits, alignInBits, offsetInBits);
            })
            .Default([&](auto type) { });

        return diTypeAttr;
    }

  private:
    MLIRContext *context;
    LLVMTypeConverterHelper llvmtch;
};

}

#endif // MLIR_TYPESCRIPT_LOWERTOLLVMLOGIC_LLVMDEBUGINFO_H_