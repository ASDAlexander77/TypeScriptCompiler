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
    MLIRContext *context;
    LLVMTypeConverterHelper llvmtch;
    llvm::StringMap<LLVM::DITypeAttr> namedTypes;
    mlir::SmallPtrSet<mlir::Type, 32> usedTypes;

  public:
    LLVMDebugInfoHelper(MLIRContext *context, LLVMTypeConverterHelper llvmtch) : context(context), llvmtch(llvmtch), namedTypes(), usedTypes()
    {
    }

    LLVM::DITypeAttr getDIType(mlir::Type llvmType, mlir::Type type, LLVM::DIFileAttr file, uint32_t line, LLVM::DIScopeAttr scope)
    {
        auto diType = getDITypeScriptType(type, file, line, scope);
        if (diType)
        {
            return diType;
        }

        return getDILLVMType(llvmType, file, line, scope);
    }

    LLVM::DITypeAttr getDITypeScriptType(mlir::Type type, LLVM::DIFileAttr file, uint32_t line, LLVM::DIScopeAttr scope)
    {
        if (!type)
        {
            return {};
        }

        LLVM_DEBUG(llvm::dbgs() << "DI for TS Type:\t" << type << "\n");

        if (auto basicType = getDITypeScriptBasicType(type, file, line, scope))
        {
            return basicType;
        }

        if (auto stringType = type.dyn_cast<mlir_ts::StringType>())
        {
            return getDIType(stringType, file, line);
        }

        if (auto opaqueType = type.dyn_cast<mlir_ts::OpaqueType>())
        {
            return getDIType(opaqueType, file, line);
        }

        // special case
        if (auto anyType = type.dyn_cast<mlir_ts::AnyType>())
        {
            return getDIType(anyType, file, line, scope);
        }

#ifdef ENABLE_DEBUGINFO_PATCH_INFO
        if (auto arrayType = type.dyn_cast_or_null<mlir_ts::ArrayType>())
        {
            return getDIType(arrayType, file, line, scope);
        }
#endif        

        if (auto unionType = type.dyn_cast<mlir_ts::UnionType>())
        {
            MLIRTypeHelper mth(context);
            if (mth.isUnionTypeNeedsTag(unionType))
            {
                return getDIType(unionType, file, line, scope);
            }
        }

        if (auto tupleType = type.dyn_cast<mlir_ts::TupleType>())
        {
            return getDITypeWithFields(tupleType, "tuple", true, file, line, scope);
        }

        if (auto classType = type.dyn_cast_or_null<mlir_ts::ClassType>())
        {
            return getDIType(classType, file, line, scope);
        }

        if (auto classStorageType = type.dyn_cast<mlir_ts::ClassStorageType>())
        {
            return getDITypeWithFields(classStorageType, classStorageType.getName().getValue().str(), false, file, line, scope);
        }

        if (auto enumType = type.dyn_cast<mlir_ts::EnumType>())
        {
            return getDIType(enumType, file, line, scope);
        }

        return getDILLVMType(llvmtch.typeConverter.convertType(type), file, line, scope);
    }    

    LLVM::DITypeAttr getDITypeScriptBasicType(mlir::Type type, LLVM::DIFileAttr file, uint32_t line, LLVM::DIScopeAttr scope)
    {
        LLVM::DITypeAttr diTypeAttr;

        mlir::TypeSwitch<mlir::Type>(type)
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
            .Default([&](auto type) { 
                // TODO: review it                
                //diTypeAttr = LLVM::DIVoidResultTypeAttr::get(context);
            });     

        return diTypeAttr;   
    }    

    LLVM::DITypeAttr getDILLVMType(mlir::Type llvmType, LLVM::DIFileAttr file, uint32_t line, LLVM::DIScopeAttr scope)
    {
        LLVM::DITypeAttr diTypeAttr;

        LLVM_DEBUG(llvm::dbgs() << "DI for llvmType:\t" << llvmType << "\n");

        if (!llvmType)
        {
            return diTypeAttr;
        }

        mlir::TypeSwitch<mlir::Type>(llvmType)
            .Case<mlir::IntegerType>([&](auto intType) {  
                auto typeCode = intType.isSigned() ? dwarf::DW_ATE_signed : dwarf::DW_ATE_unsigned;
                StringRef typeName = "int";
                auto size = intType.getIntOrFloatBitWidth(); 
                if (size == 1)
                {
                    typeName = "bool";
                    typeCode = dwarf::DW_ATE_boolean;
                    size = 8;
                }
                else if (size == 8)
                {
                    typeName = "char";
                    typeCode = dwarf::DW_ATE_signed_char;
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
                diTypeAttr = getDIStructType(structType, file, line, scope);
            })
            .Case<LLVM::LLVMPointerType>([&](auto llvmPointerType) {  
                diTypeAttr = getDIPointerType(getDILLVMType(llvmPointerType.getElementType(), file, line, scope), file, line);
            })
            .Case<LLVM::LLVMFunctionType>([&](auto funcType) {
                diTypeAttr = getDISubroutineType(funcType, file, line, scope);
            })
            .Default([&](auto type) { 
                // TODO: review it                
                //diTypeAttr = LLVM::DIVoidResultTypeAttr::get(context);
            });

        return diTypeAttr;
    }    

    LLVM::DITypeAttr getDIType(mlir_ts::StringType stringType, LLVM::DIFileAttr file, uint32_t line)
    {
        StringRef typeName = "char";
        auto typeCode = dwarf::DW_ATE_signed_char;
        auto size = 8;
        auto diTypeAttr = LLVM::DIBasicTypeAttr::get(context, dwarf::DW_TAG_base_type, StringAttr::get(context, typeName), size, typeCode);
        return getDIPointerType(diTypeAttr, file, line);
    }     

    LLVM::DITypeAttr getDIType(mlir_ts::OpaqueType opaqueType, LLVM::DIFileAttr file, uint32_t line)
    {
        StringRef typeName = "address";
        auto typeCode = dwarf::DW_ATE_address;
        auto size = 0;
        auto diTypeAttr = LLVM::DIBasicTypeAttr::get(context, dwarf::DW_TAG_base_type, StringAttr::get(context, typeName), size, typeCode);
        return getDIPointerType(diTypeAttr, file, line);
    }     

    LLVM::DITypeAttr getDIType(mlir_ts::AnyType anyType, LLVM::DIFileAttr file, uint32_t line, LLVM::DIScopeAttr scope)
    {
        auto diBodyType = getDIStructType("any", {
            {"size", mlir::IndexType::get(context)},
            {"type", mlir_ts::StringType::get(context)},
        }, file, line, scope);

        return getDIPointerType(diBodyType, file, line);
    } 

#ifdef ENABLE_DEBUGINFO_PATCH_INFO
    LLVM::DITypeAttr getDIType(mlir_ts::ArrayType arrayType, LLVM::DIFileAttr file, uint32_t line, LLVM::DIScopeAttr scope)
    {
        llvm::SmallVector<LLVM::DINodeAttr> elements;

        auto sizeElement = LLVM::DISubrangeAttr::get(context, IntegerAttr::get(mlir::IntegerType::get(context, 32), 3), IntegerAttr(), IntegerAttr(), IntegerAttr());
        elements.push_back(sizeElement);

        auto elementType = getDIType(llvmtch.typeConverter.convertType(arrayType.getElementType()), arrayType.getElementType(), file, line);
        return LLVM::DICompositeTypeAttr::get(context, dwarf::DW_TAG_array_type, StringAttr::get(context, MLIRHelper::getAnonymousName(arrayType.getElementType(), "array")), 
            file, line, scope, elementType, LLVM::DIFlags::Zero, 0, 0, elements);        
    } 
#endif    

    LLVM::DITypeAttr getDIType(mlir_ts::UnionType unionType, LLVM::DIFileAttr file, uint32_t line, LLVM::DIScopeAttr scope)
    {
        auto strType = mlir_ts::StringType::get(context);
        auto diStrType = getDITypeScriptType(strType, file, line, scope);

        auto diTypeAttrUnion = getDIUnionType(unionType, file, line, scope);

        return getDIStructType(MLIRHelper::getAnonymousName(unionType, "union"), {
            {"type", diStrType},
            {"data", diTypeAttrUnion},
        }, file, line, scope);        
    }    

    LLVM::DITypeAttr getDIType(mlir_ts::ClassType classType, LLVM::DIFileAttr file, uint32_t line, LLVM::DIScopeAttr scope)
    {
        auto diTypeAttrClassType = getDIPointerType(getDITypeScriptType(classType.getStorageType(), file, line, scope), file, line);
        return diTypeAttrClassType;        
    } 

    LLVM::DITypeAttr getDIType(mlir_ts::EnumType enumType, LLVM::DIFileAttr file, uint32_t line, LLVM::DIScopeAttr scope)
    {
        auto diBaseType = getDITypeScriptType(enumType.getElementType(), file, line, scope);

        //auto enumName = MLIRHelper::getAnonymousName(enumType, "enum");

        // llvm::SmallVector<LLVM::DINodeAttr> elements;
        // auto dictVal = enumType.getValues();
        // for (auto [index, enumValue] : enumerate(dictVal))
        // {
        //     // name
        //     auto name = enumValue.getName();

        //     auto wrapperDiType = LLVM::DIDerivedTypeAttr::get(context, dwarf::DW_TAG_enumeration_type, name, diBaseType, 
        //         0, 0, 0);
        //     elements.push_back(wrapperDiType);
        // }

        // auto compositeType = LLVM::DICompositeTypeAttr::get(context, dwarf::DW_TAG_enumeration_type, 
        //     StringAttr::get(context, enumName), 
        //     file, line, scope, LLVM::DITypeAttr(), LLVM::DIFlags::TypePassByValue, 0, 0, {});

        // return compositeType;

        return diBaseType;
    }

    LLVM::DITypeAttr getDITypeWithFields(mlir::Type typeWithFields, std::string name, bool isNamePrefix, LLVM::DIFileAttr file, uint32_t line, LLVM::DIScopeAttr scope)
    {
        if (!isNamePrefix)
        {
            if (auto diType = namedTypes.lookup(name))
            {
                return diType;
            }    

            if (usedTypes.contains(typeWithFields)) {
                // create forward declaration
                auto fwdCompositeType = LLVM::DICompositeTypeAttr::get(context, dwarf::DW_TAG_structure_type, 
                    StringAttr::get(context, name), 
                    file, line, scope, LLVM::DITypeAttr(), LLVM::DIFlags::FwdDecl, 0, 0, {});

                return fwdCompositeType;
            }

            usedTypes.insert(typeWithFields);            
        }

        MLIRTypeHelper mth(context);
        llvm::SmallVector<mlir_ts::FieldInfo> destTupleFields;
        auto hasFields = mlir::succeeded(mth.getFields(typeWithFields, destTupleFields, true));

        CompositeSizesTrack sizesTrack(llvmtch);

        llvm::SmallVector<LLVM::DINodeAttr> elements;
        for (auto [index, fieldInfo] : enumerate(destTupleFields))
        {
            auto elementType = fieldInfo.type;
            auto llvmElementType = llvmtch.typeConverter.convertType(elementType);
            sizesTrack.nextElementType(llvmElementType);

            // name
            StringAttr name = StringAttr::get(context, std::to_string(index));
            if (hasFields)
            {
                auto fieldId = fieldInfo.id;
                if (auto strFieldId = fieldId.dyn_cast_or_null<mlir::StringAttr>())
                {
                    name = strFieldId;
                }
            }

            auto elementDiType = getDITypeScriptType(elementType, file, line, scope);
            auto wrapperDiType = LLVM::DIDerivedTypeAttr::get(context, dwarf::DW_TAG_member, name, elementDiType, 
                sizesTrack.elementSizeInBits, sizesTrack.elementAlignInBits, sizesTrack.offsetInBits);
            elements.push_back(wrapperDiType);
        }

        auto compositeType = LLVM::DICompositeTypeAttr::get(context, dwarf::DW_TAG_structure_type, 
            StringAttr::get(context, isNamePrefix ? MLIRHelper::getAnonymousName(typeWithFields, name.c_str()) : name), 
            file, line, scope, LLVM::DITypeAttr(), LLVM::DIFlags::TypePassByValue, sizesTrack.sizeInBits, sizesTrack.alignInBits, elements);

        if (!isNamePrefix)
        {
            namedTypes.insert_or_assign(name, compositeType);
        }

        return compositeType;          
    }    

    LLVM::DIDerivedTypeAttr getDIPointerType(LLVM::DITypeAttr diElementType, LLVM::DIFileAttr file, uint32_t line)
    {
        auto sizeInBits = llvmtch.getPointerBitwidth(0);
        auto alignInBits = sizeInBits;
        auto offsetInBits = 0;

        return LLVM::DIDerivedTypeAttr::get(
            context, dwarf::DW_TAG_pointer_type, StringAttr::get(context, "pointer"), diElementType, 
            sizeInBits, alignInBits, offsetInBits);
    }

    LLVM::DISubroutineTypeAttr getDISubroutineType(LLVM::LLVMFunctionType funcType, LLVM::DIFileAttr file, uint32_t line, LLVM::DIScopeAttr scope)
    {
        llvm::SmallVector<LLVM::DITypeAttr> elements;
        for (auto retType : funcType.getReturnTypes())
        {
            elements.push_back(getDITypeScriptType(retType, file, line, scope));  
        }

        if (funcType.getParams().size() > 0 &&  funcType.getReturnTypes().size() == 0)
        {
            // return type is null
            elements.push_back(mlir::LLVM::DINullTypeAttr());
        }

        for (auto paramType : funcType.getParams())
        {
            elements.push_back(getDITypeScriptType(paramType, file, line, scope));  
        }

        auto subroutineType = LLVM::DISubroutineTypeAttr::get(context, elements);
        return subroutineType;
    }

    // Seems LLVM::DIFlags::FwdDecl is resolving issue for me
    LLVM::DITypeAttr getDIStructType(LLVM::LLVMStructType structType, LLVM::DIFileAttr file, uint32_t line, LLVM::DIScopeAttr scope)
    {
        if (structType.isIdentified())
        {
            if (auto diType = namedTypes.lookup(structType.getName()))
            {
                return diType;
            }    

            if (usedTypes.contains(structType)) {
                // create forward declaration
                auto fwdCompositeType = LLVM::DICompositeTypeAttr::get(context, dwarf::DW_TAG_structure_type, 
                    StringAttr::get(context, structType.getName()), 
                    file, line, scope, LLVM::DITypeAttr(), LLVM::DIFlags::FwdDecl, 0, 0, {});

                return fwdCompositeType;
            }

            usedTypes.insert(structType);
        }

        MLIRTypeHelper mth(context);
        CompositeSizesTrack sizesTrack(llvmtch);

        // now we can resolve recursive types
        llvm::SmallVector<LLVM::DINodeAttr> elements;
        for (auto [index, llvmElementType] : enumerate(structType.getBody()))
        {
            sizesTrack.nextElementType(llvmElementType);

            StringAttr elementName = StringAttr::get(context, std::to_string(index));

            auto elementDiType = getDILLVMType(llvmElementType, file, line, scope);
            auto wrapperDiType = LLVM::DIDerivedTypeAttr::get(context, dwarf::DW_TAG_member, 
                elementName, elementDiType, sizesTrack.elementSizeInBits, sizesTrack.elementAlignInBits, sizesTrack.offsetInBits);
                elements.push_back(wrapperDiType);  
        }

        auto name = StringAttr::get(context, structType.isIdentified() 
            ? structType.getName() 
            : MLIRHelper::getAnonymousName(structType, "struct"));
        auto compositeType = LLVM::DICompositeTypeAttr::get(context, dwarf::DW_TAG_structure_type, name, 
            file, line, scope, LLVM::DITypeAttr(), LLVM::DIFlags::TypePassByValue, sizesTrack.sizeInBits, sizesTrack.alignInBits, elements);

        if (structType.isIdentified())
        {
            namedTypes.insert_or_assign(structType.getName(), compositeType);
        }

        return compositeType;
    }

    LLVM::DICompositeTypeAttr getDIStructType(StringRef name, ArrayRef<std::pair<StringRef, mlir::Type>> fields, LLVM::DIFileAttr file, uint32_t line, LLVM::DIScopeAttr scope)
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

    LLVM::DICompositeTypeAttr getDIStructType(StringRef name, ArrayRef<std::pair<StringRef, LLVM::DITypeAttr>> fields, LLVM::DIFileAttr file, uint32_t line, LLVM::DIScopeAttr scope)
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

    LLVM::DICompositeTypeAttr getDIUnionType(mlir_ts::UnionType unionType, LLVM::DIFileAttr file, uint32_t line, LLVM::DIScopeAttr scope)
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
            auto name = mth.getLabelName(elementType);

            auto elementDiType = getDITypeScriptType(elementType, file, line, scope);
            auto wrapperDiType = LLVM::DIDerivedTypeAttr::get(context, dwarf::DW_TAG_member, name, 
                elementDiType, sizesTrack.elementSizeInBits, sizesTrack.elementAlignInBits, 0);
            elements.push_back(wrapperDiType);
        }

        return LLVM::DICompositeTypeAttr::get(context, dwarf::DW_TAG_union_type, StringAttr::get(context, MLIRHelper::getAnonymousName(unionType, "union")), 
            file, line, scope, LLVM::DITypeAttr(), LLVM::DIFlags::TypePassByValue, sizeInBits, sizesTrack.alignInBits, elements);
    }
};

}

#undef DEBUG_TYPE

#endif // MLIR_TYPESCRIPT_LOWERTOLLVMLOGIC_LLVMDEBUGINFO_H_