#ifndef MLIR_TYPESCRIPT_LOWERTOLLVMLOGIC_LLVMDEBUGINFO_H_
#define MLIR_TYPESCRIPT_LOWERTOLLVMLOGIC_LLVMDEBUGINFO_H_

#include "TypeScript/LowerToLLVM/LocationHelper.h"

#include "llvm/DebugInfo/DWARF/DWARFUnit.h"

#define DEBUG_TYPE "llvm"

using namespace mlir;
namespace mlir_ts = mlir::typescript;

namespace typescript
{

struct CompositeSizesTrack
{
    CompositeSizesTrack(LLVMTypeConverterHelper &llvmtch) :
        elementSizeInBits(0),
        elementAlignInBits(0),
        sizeInBits(0),
        alignInBits(0),
        offsetInBits(0),
        llvmtch(llvmtch)
    {
    };

    void nextElementType(mlir::Type llvmElementType)
    {
        shiftOffset();
        elementSizeInBits = llvmtch.getTypeAllocSizeInBits(llvmElementType);
        elementAlignInBits = llvmtch.getTypeAlignSizeInBits(llvmElementType);
        calculateSizeAndOffset();
    }

    void nextElementType(LLVM::DITypeAttr diTypeAttr)
    {
        shiftOffset();
        if (auto basicDiTypeAttr = diTypeAttr.dyn_cast<LLVM::DIBasicTypeAttr>())
        {
            elementSizeInBits = elementAlignInBits = basicDiTypeAttr.getSizeInBits();
        }
        else if (auto compDiTypeAttr = diTypeAttr.dyn_cast<LLVM::DICompositeTypeAttr>())
        {
            elementSizeInBits = compDiTypeAttr.getSizeInBits();
            elementAlignInBits = compDiTypeAttr.getAlignInBits();
        }
        else if (auto derivedDiTypeAttr = diTypeAttr.dyn_cast<LLVM::DIDerivedTypeAttr>())
        {
            elementSizeInBits = derivedDiTypeAttr.getSizeInBits();
            elementAlignInBits = derivedDiTypeAttr.getAlignInBits();
        }
        else
        {
            llvm_unreachable("not implemeneted");
        }

        calculateSizeAndOffset();
    }

    void shiftOffset()
    {
        offsetInBits = sizeInBits;
    }

    void calculateSizeAndOffset()
    {
        sizeInBits += elementSizeInBits;
        if (elementAlignInBits > alignInBits)
        {
            alignInBits = elementAlignInBits;
        }        
    }

    uint64_t elementSizeInBits;
    uint64_t elementAlignInBits;

    uint64_t sizeInBits;
    uint64_t alignInBits;
    uint64_t offsetInBits;    

    LLVMTypeConverterHelper &llvmtch;
};

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

#ifdef ENABLE_DEBUGINFO_PATCH_INFO
        if (auto arrayType = type.dyn_cast_or_null<mlir_ts::ArrayType>())
        {
            return getDIType(arrayType, file, line, scope);
        }
#endif        

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

        LLVM_DEBUG(llvm::dbgs() << "DI for llvmType:\t" << llvmType << "\n");

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
                    // TODO: review it
                    //diTypeAttr = LLVM::DIVoidResultTypeAttr::get(context);
                    //return;
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
                diTypeAttr = getDIStructType(structType, type, file, line, scope);
            })
            .Case<LLVM::LLVMPointerType>([&](auto llvmPointerType) {  
                MLIRTypeHelper mth(context);
                diTypeAttr = getDIPointerType(llvmPointerType, mth.getElementType(type), file, line, scope);
            })
            .Default([&](auto type) { 
                // TODO: review it                
                //diTypeAttr = LLVM::DIVoidResultTypeAttr::get(context);
            });

        return diTypeAttr;
    }    

    LLVM::DITypeAttr getDIType(mlir_ts::AnyType anyType, LLVM::DIFileAttr file, uint32_t line, LLVM::DIScopeAttr scope)
    {
        auto diBodyType = getDIStructType("any", {
            {"size", mlir::IndexType::get(context)},
            {"type", mlir_ts::StringType::get(context)},
        }, file, line, scope);

        return getDIPointerType(diBodyType, file, line, scope);
    } 

#ifdef ENABLE_DEBUGINFO_PATCH_INFO
    LLVM::DITypeAttr getDIType(mlir_ts::ArrayType arrayType, LLVM::DIFileAttr file, uint32_t line, LLVM::DIScopeAttr scope)
    {
        llvm::SmallVector<LLVM::DINodeAttr> elements;

        auto sizeElement = LLVM::DISubrangeAttr::get(context, IntegerAttr::get(mlir::IntegerType::get(context, 32), 3), IntegerAttr(), IntegerAttr(), IntegerAttr());
        elements.push_back(sizeElement);

        auto elementType = getDIType(llvmtch.typeConverter.convertType(arrayType.getElementType()), arrayType.getElementType(), file, line, scope);
        return LLVM::DICompositeTypeAttr::get(context, dwarf::DW_TAG_array_type, StringAttr::get(context, MLIRHelper::getAnonymousName(arrayType.getElementType(), "array")), 
            file, line, scope, elementType, LLVM::DIFlags::Zero, 0, 0, elements);        
    } 
#endif    

    LLVM::DITypeAttr getDIType(mlir_ts::UnionType unionType, LLVM::DIFileAttr file, uint32_t line, LLVM::DIScopeAttr scope)
    {
        auto strType = mlir_ts::StringType::get(context);
        auto llvmStrType = llvmtch.typeConverter.convertType(strType);
        auto diStrType = getDIType(llvmStrType, strType, file, line, scope);

        auto diTypeAttrUnion = getDIUnionType(unionType, file, line, scope);

        return getDIStructType(MLIRHelper::getAnonymousName(unionType, "struct"), {
            {"type", diStrType},
            {"union", diTypeAttrUnion},
        }, file, line, scope);        
    }    

private:

    LLVM::DITypeAttr getDIPointerType(LLVM::LLVMPointerType llvmPointerType, mlir::Type elementType, LLVM::DIFileAttr file, uint32_t line, LLVM::DIScopeAttr scope)
    {
        return getDIPointerType(getDIType(llvmPointerType.getElementType(), elementType, file, line, scope), file, line, scope);
    }

    LLVM::DITypeAttr getDIPointerType(LLVM::DITypeAttr diElementType, LLVM::DIFileAttr file, uint32_t line, LLVM::DIScopeAttr scope)
    {
        auto sizeInBits = llvmtch.getPointerBitwidth(0);
        auto alignInBits = sizeInBits;
        auto offsetInBits = 0;

        return LLVM::DIDerivedTypeAttr::get(
            context, dwarf::DW_TAG_pointer_type, StringAttr::get(context, "pointer"), diElementType, 
            sizeInBits, alignInBits, offsetInBits);
    }

    LLVM::DITypeAttr getDIStructType(LLVM::LLVMStructType structType, mlir::Type type, LLVM::DIFileAttr file, uint32_t line, LLVM::DIScopeAttr scope)
    {
        MLIRTypeHelper mth(context);
        llvm::SmallVector<mlir_ts::FieldInfo> destTupleFields;
        auto hasFields = mlir::succeeded(mth.getFields(type, destTupleFields, true));

        CompositeSizesTrack sizesTrack(llvmtch);

        llvm::SmallVector<LLVM::DINodeAttr> elements;
        for (auto [index, llvmElementType] : enumerate(structType.getBody()))
        {
            sizesTrack.nextElementType(llvmElementType);

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
            auto wrapperDiType = LLVM::DIDerivedTypeAttr::get(context, dwarf::DW_TAG_member, name, elementDiType, sizesTrack.elementSizeInBits, sizesTrack.elementAlignInBits, sizesTrack.offsetInBits);
            elements.push_back(wrapperDiType);
        }

        return LLVM::DICompositeTypeAttr::get(context, dwarf::DW_TAG_structure_type, StringAttr::get(context, MLIRHelper::getAnonymousName(structType, "struct")), 
            file, line, scope, LLVM::DITypeAttr(), LLVM::DIFlags::TypePassByValue, sizesTrack.sizeInBits, sizesTrack.alignInBits, elements);
    }

    LLVM::DITypeAttr getDIStructType(StringRef name, ArrayRef<std::pair<StringRef, mlir::Type>> fields, LLVM::DIFileAttr file, uint32_t line, LLVM::DIScopeAttr scope)
    {
        MLIRTypeHelper mth(context);

        CompositeSizesTrack sizesTrack(llvmtch);

        llvm::SmallVector<LLVM::DINodeAttr> elements;
        for (auto field : fields)
        {
            // name
            StringAttr name = StringAttr::get(context, std::get<0>(field));

            mlir::Type elementType = std::get<1>(field);
            auto llvmElementType = llvmtch.typeConverter.convertType(elementType);
            
            sizesTrack.nextElementType(llvmElementType);
            
            auto elementDiType = getDIType(llvmElementType, elementType, file, line, scope);
            auto wrapperDiType = LLVM::DIDerivedTypeAttr::get(context, dwarf::DW_TAG_member, name, elementDiType, 
                sizesTrack.elementSizeInBits, sizesTrack.elementAlignInBits, sizesTrack.offsetInBits);
            elements.push_back(wrapperDiType);
        }

        return LLVM::DICompositeTypeAttr::get(context, dwarf::DW_TAG_structure_type, StringAttr::get(context, name), 
            file, line, scope, LLVM::DITypeAttr(), LLVM::DIFlags::TypePassByValue, sizesTrack.sizeInBits, sizesTrack.alignInBits, elements);        
    }

    LLVM::DITypeAttr getDIStructType(StringRef name, ArrayRef<std::pair<StringRef, LLVM::DITypeAttr>> fields, LLVM::DIFileAttr file, uint32_t line, LLVM::DIScopeAttr scope)
    {
        MLIRTypeHelper mth(context);

        CompositeSizesTrack sizesTrack(llvmtch);

        llvm::SmallVector<LLVM::DINodeAttr> elements;
        for (auto field : fields)
        {
            // name
            StringAttr name = StringAttr::get(context, std::get<0>(field));

            auto elementDiType = std::get<1>(field);

            sizesTrack.nextElementType(elementDiType);
            
            auto wrapperDiType = LLVM::DIDerivedTypeAttr::get(context, dwarf::DW_TAG_member, name, elementDiType, 
                sizesTrack.elementSizeInBits, sizesTrack.elementAlignInBits, sizesTrack.offsetInBits);
            elements.push_back(wrapperDiType);
        }

        return LLVM::DICompositeTypeAttr::get(context, dwarf::DW_TAG_structure_type, StringAttr::get(context, name), 
            file, line, scope, LLVM::DITypeAttr(), LLVM::DIFlags::TypePassByValue, sizesTrack.sizeInBits, sizesTrack.alignInBits, elements);        
    }    

    LLVM::DITypeAttr getDIUnionType(mlir_ts::UnionType unionType, LLVM::DIFileAttr file, uint32_t line, LLVM::DIScopeAttr scope)
    {
        auto sizeInBits = 0;

        MLIRTypeHelper mth(context);

        CompositeSizesTrack sizesTrack(llvmtch);

        llvm::SmallVector<LLVM::DINodeAttr> elements;
        for (auto elementType : unionType.getTypes())
        {
            auto llvmElementType = llvmtch.typeConverter.convertType(elementType);
            sizesTrack.nextElementType(llvmElementType);
            if (sizesTrack.elementSizeInBits > sizeInBits) sizeInBits = sizesTrack.elementSizeInBits;

            // name
            StringAttr name = mth.getLabelName(elementType);

            auto elementDiType = getDIType(llvmElementType, elementType, file, line, scope);
            auto wrapperDiType = LLVM::DIDerivedTypeAttr::get(context, dwarf::DW_TAG_member, name, 
                elementDiType, sizesTrack.elementSizeInBits, sizesTrack.elementAlignInBits, 0);
            elements.push_back(wrapperDiType);
        }

        return LLVM::DICompositeTypeAttr::get(context, dwarf::DW_TAG_union_type, StringAttr::get(context, MLIRHelper::getAnonymousName(unionType, "union")), 
            file, line, scope, LLVM::DITypeAttr(), LLVM::DIFlags::TypePassByValue, sizeInBits, sizesTrack.alignInBits, elements);
    }
  private:
    MLIRContext *context;
    LLVMTypeConverterHelper llvmtch;
};

}

#undef DEBUG_TYPE

#endif // MLIR_TYPESCRIPT_LOWERTOLLVMLOGIC_LLVMDEBUGINFO_H_