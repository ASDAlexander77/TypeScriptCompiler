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

    LLVM::DITypeAttr getDIType(mlir::Type llvmType, mlir::Type type, LLVM::DIFileAttr file, uint32_t line, LLVM::DIScopeAttr scope)
    {
        // special case
        if (auto anyType = type.dyn_cast_or_null<mlir_ts::AnyType>())
        {
            return getDIType(anyType, file, line, scope);
        }

        if (auto unionType = type.dyn_cast_or_null<mlir_ts::UnionType>())
        {
            MLIRTypeHelper mth(context);
            if (mth.isUnionTypeNeedsTag(unionType))
            {
                return getDIType(unionType, file, line, scope);
            }
        }

        return getDILLVMType(llvmType, type, file, line, scope);
    }

    LLVM::DITypeAttr getDILLVMType(mlir::Type llvmType, mlir::Type type, LLVM::DIFileAttr file, uint32_t line, LLVM::DIScopeAttr scope)
    {
        LLVM::DITypeAttr diTypeAttr;

        mlir::TypeSwitch<mlir::Type>(llvmType)
            .Case<mlir::IntegerType>([&](auto intType) {  
                auto typeCode = dwarf::DW_ATE_signed;
                StringRef typeName = "int";
                auto size = intType.getIntOrFloatBitWidth(); 
                if (type && type.isa<mlir_ts::CharType>())
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
                else if (size == 8)
                {
                    diTypeAttr = LLVM::DIVoidResultTypeAttr::get(context);
                    return;
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
                diTypeAttr = getDILLVMStructType(structType, type, file, line, scope);
            })
            .Case<LLVM::LLVMPointerType>([&](auto llvmPointerType) {  
                MLIRTypeHelper mth(context);
                diTypeAttr = getDIPointerType(llvmPointerType, mth.getElementType(type), file, line, scope);
            })
            .Default([&](auto type) { 
                diTypeAttr = LLVM::DIVoidResultTypeAttr::get(context);
            });

        return diTypeAttr;
    }    

    LLVM::DITypeAttr getDILLVMStructType(LLVM::LLVMStructType structType, mlir::Type type, LLVM::DIFileAttr file, uint32_t line, LLVM::DIScopeAttr scope)
    {
        auto sizeInBits = 0;
        auto alignInBits = 0;
        auto offsetInBits = 0;

        MLIRTypeHelper mth(context);
        llvm::SmallVector<mlir_ts::FieldInfo> destTupleFields;
        auto hasFields = mlir::succeeded(mth.getFields(type, destTupleFields, true));

        llvm::SmallVector<LLVM::DINodeAttr> elements;
        auto index = -1;
        for (auto llvmElementType : structType.getBody())
        {
            index++;

            auto elementSizeInBits = llvmtch.getTypeAllocSizeInBits(llvmElementType); // size of element
            auto elementAlignInBits = llvmtch.getTypeAlignSizeInBits(llvmElementType); // size of element

            // name
            mlir::Type elementType;
            StringAttr name = StringAttr::get(context, std::to_string(index));
            if (hasFields)
            {
                auto fieldId = destTupleFields[index].id;
                if (auto strFieldId = fieldId.dyn_cast_or_null<mlir::StringAttr>())
                {
                    name = strFieldId;
                }

                elementType = destTupleFields[index].type;
            }

            auto elementDiType = getDIType(llvmElementType, elementType, file, line, scope);
            auto wrapperDiType = LLVM::DIDerivedTypeAttr::get(context, dwarf::DW_TAG_member, name, elementDiType, elementSizeInBits, elementAlignInBits, offsetInBits);
            elements.push_back(wrapperDiType);

            offsetInBits += elementSizeInBits;
            if (elementAlignInBits > alignInBits)
            {
                alignInBits = elementAlignInBits;
            }
        }

        sizeInBits = offsetInBits;

        return LLVM::DICompositeTypeAttr::get(context, dwarf::DW_TAG_structure_type, StringAttr::get(context, MLIRHelper::getAnonymousName(structType, "struct")), 
            file, line, scope, LLVM::DITypeAttr(), LLVM::DIFlags::TypePassByValue, sizeInBits, alignInBits, elements);
    }

    LLVM::DITypeAttr getDIType(mlir_ts::AnyType anyType, LLVM::DIFileAttr file, uint32_t line, LLVM::DIScopeAttr scope)
    {
        auto diBodyType = getDIStructType("any", {
            {"size", mlir::IndexType::get(context)},
            {"type", mlir_ts::StringType::get(context)},
        }, file, line, scope);

        return getDIPointerType(diBodyType, file, line, scope);
    } 

    LLVM::DITypeAttr getDIType(mlir_ts::UnionType unionType, LLVM::DIFileAttr file, uint32_t line, LLVM::DIScopeAttr scope)
    {
        auto sizeInBits = 0;
        auto alignInBits = 0;
        auto offsetInBits = 0;

        MLIRTypeHelper mth(context);

        llvm::SmallVector<LLVM::DINodeAttr> elements;
        auto index = -1;
        for (auto elementType : unionType.getTypes())
        {
            index++;

            auto llvmElementType = llvmtch.typeConverter.convertType(elementType);

            auto elementSizeInBits = llvmtch.getTypeAllocSizeInBits(llvmElementType); // size of element
            auto elementAlignInBits = llvmtch.getTypeAlignSizeInBits(llvmElementType); // size of element

            if (elementSizeInBits > sizeInBits) sizeInBits = elementSizeInBits;

            // name
            StringAttr name = mth.getLabelName(elementType);

            auto elementDiType = getDIType(llvmElementType, elementType, file, line, scope);
            auto wrapperDiType = LLVM::DIDerivedTypeAttr::get(context, dwarf::DW_TAG_member, name, elementDiType, elementSizeInBits, elementAlignInBits, offsetInBits);
            elements.push_back(wrapperDiType);

            if (elementAlignInBits > alignInBits)
            {
                alignInBits = elementAlignInBits;
            }             
        }

        sizeInBits = offsetInBits;

        auto diTypeAttrUnion = LLVM::DICompositeTypeAttr::get(context, dwarf::DW_TAG_union_type, StringAttr::get(context, MLIRHelper::getAnonymousName(unionType, "union")), 
            file, line, scope, LLVM::DITypeAttr(), LLVM::DIFlags::TypePassByValue, sizeInBits, alignInBits, elements);

        // top type with tag
        llvm::SmallVector<LLVM::DINodeAttr> unionWithTagElements;

        auto strType = mlir_ts::StringType::get(context);
        auto llvmStrType = llvmtch.typeConverter.convertType(strType);
        auto diStrType = getDIType(llvmStrType, strType, file, line, scope);

        auto wrapperTagDiType = LLVM::DIDerivedTypeAttr::get(context, dwarf::DW_TAG_member, StringAttr::get(context, "type"), 
            diStrType, llvmtch.getPointerBitwidth(0), alignInBits, 0);
        unionWithTagElements.push_back(wrapperTagDiType);

        auto wrapperUnionDiType = LLVM::DIDerivedTypeAttr::get(context, dwarf::DW_TAG_member, StringAttr::get(context, "union"), 
            diTypeAttrUnion, sizeInBits + llvmtch.getPointerBitwidth(0), alignInBits, llvmtch.getPointerBitwidth(0));
        unionWithTagElements.push_back(wrapperUnionDiType);

        auto diTypeAttr = LLVM::DICompositeTypeAttr::get(context, dwarf::DW_TAG_structure_type, StringAttr::get(context, MLIRHelper::getAnonymousName(unionType, "struct")), 
            file, line, scope, LLVM::DITypeAttr(), LLVM::DIFlags::TypePassByValue, sizeInBits, alignInBits, unionWithTagElements);                                    

        return diTypeAttr;
    }    

private:

    LLVM::DITypeAttr getDIPointerType(LLVM::LLVMPointerType llvmPointerType, mlir::Type elementType, LLVM::DIFileAttr file, uint32_t line, LLVM::DIScopeAttr scope)
    {
        return getDIPointerType(getDIType(llvmPointerType.getElementType(), elementType, file, line, scope), file, line, scope);
    }

    LLVM::DITypeAttr getDIPointerType(LLVM::DITypeAttr diElementType, LLVM::DIFileAttr file, uint32_t line, LLVM::DIScopeAttr scope)
    {
        auto sizeInBits = 64;
        auto alignInBits = 32;
        auto offsetInBits = 0;

        return LLVM::DIDerivedTypeAttr::get(
            context, dwarf::DW_TAG_pointer_type, StringAttr::get(context, "pointer"), diElementType, 
            sizeInBits, alignInBits, offsetInBits);
    }

    LLVM::DITypeAttr getDIStructType(StringRef name, ArrayRef<std::pair<StringRef, mlir::Type>> fields, LLVM::DIFileAttr file, uint32_t line, LLVM::DIScopeAttr scope)
    {
        auto sizeInBits = 0;
        auto alignInBits = 0;
        auto offsetInBits = 0;

        MLIRTypeHelper mth(context);

        llvm::SmallVector<LLVM::DINodeAttr> elements;
        auto index = -1;
        for (auto field : fields)
        {
            index++;

            // name
            StringAttr name = StringAttr::get(context, std::get<0>(field));

            mlir::Type elementType = std::get<1>(field);
            auto llvmElementType = llvmtch.typeConverter.convertType(elementType);
            
            auto elementSizeInBits = llvmtch.getTypeAllocSizeInBits(llvmElementType); // size of element
            auto elementAlignInBits = llvmtch.getTypeAlignSizeInBits(llvmElementType); // size of element

            auto elementDiType = getDIType(llvmElementType, elementType, file, line, scope);
            auto wrapperDiType = LLVM::DIDerivedTypeAttr::get(context, dwarf::DW_TAG_member, name, elementDiType, elementSizeInBits, elementAlignInBits, offsetInBits);
            elements.push_back(wrapperDiType);

            offsetInBits += elementSizeInBits;
            if (elementAlignInBits > alignInBits)
            {
                alignInBits = elementAlignInBits;
            }            
        }

        sizeInBits = offsetInBits;

        return LLVM::DICompositeTypeAttr::get(context, dwarf::DW_TAG_structure_type, StringAttr::get(context, name), 
            file, line, scope, LLVM::DITypeAttr(), LLVM::DIFlags::TypePassByValue, sizeInBits, alignInBits, elements);        
    }

  private:
    MLIRContext *context;
    LLVMTypeConverterHelper llvmtch;
};

}

#endif // MLIR_TYPESCRIPT_LOWERTOLLVMLOGIC_LLVMDEBUGINFO_H_