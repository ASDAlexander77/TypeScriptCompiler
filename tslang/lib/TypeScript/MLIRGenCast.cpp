// Cast-family methods of MLIRGenImpl (see MLIRGenImpl.h).

#include "MLIRGenImpl.h"

namespace typescript
{
namespace mlirgen
{

    // Field list for the clone made when a tuple/object value can't be cast to an
    // interface as-is (field types need coercion, e.g. si32 -> number, or a
    // func-typed field vs a method's funcType). The clone MUST keep the SOURCE
    // tuple's field order: the interface vtable's field-offset slots and any
    // methods compiled against the original object layout (in particular in
    // another module) address fields by byte offset, so reordering to interface
    // member order (methods-first, per InterfaceInfo::getTupleTypeFields)
    // silently reads/writes the wrong slots at runtime. Only the TYPES are taken
    // from the interface; interface-only members are appended after the source
    // fields.
    static mlir::LogicalResult getInterfaceCloneFields(mlir::ArrayRef<mlir_ts::FieldInfo> srcFields,
                                                       InterfaceInfo::TypePtr interfaceInfo, mlir::MLIRContext *context,
                                                       SmallVector<mlir_ts::FieldInfo> &fields)
    {
        SmallVector<mlir_ts::FieldInfo> interfaceFields;
        if (mlir::failed(interfaceInfo->getTupleTypeFields(interfaceFields, context)))
        {
            return mlir::failure();
        }

        for (auto &origField : srcFields)
        {
            auto interfaceField =
                std::find_if(interfaceFields.begin(), interfaceFields.end(),
                             [&](auto &item) { return item.id == origField.id; });
            fields.push_back(interfaceField != interfaceFields.end() ? *interfaceField : origField);
        }

        for (auto &interfaceField : interfaceFields)
        {
            if (std::find_if(fields.begin(), fields.end(),
                             [&](auto &item) { return item.id == interfaceField.id; }) == fields.end())
            {
                // the source has no field of this name at all - only reachable for a
                // conditional (`?`) interface member (getVirtualTable's resolveField
                // already rejects a missing non-conditional one, failing
                // canCastTupleToInterface before this clone path even runs). Match the
                // convention InterfaceSymbolRefOp's `optional` slot-count check already
                // relies on for the non-clone path (see 00interface_optional_cast_order.ts):
                // the field is simply ABSENT from the storage tuple, not present with a
                // placeholder value - so skip it here rather than appending it. Appending it
                // with any value/type would desync the clone's field COUNT from what
                // interfaceVTableNameForObject's slot-index math and the interface's runtime
                // "is this slot within the actual object's tuple size" check expect.
                if (interfaceField.isConditional)
                {
                    continue;
                }

                fields.push_back(interfaceField);
            }
        }

        return mlir::success();
    }

    ValueOrLogicalResult MLIRGenImpl::selectFieldsValues(mlir::Location location, SmallVector<mlir::Value> &values, mlir::Value value,
        ::llvm::ArrayRef<::mlir::typescript::FieldInfo> fields, bool filterSpecialCases, const GenContext &genContext, bool errorAsWarning)
    {
        auto count = 0;
        for (auto [index, fieldInfo] : enumerate(fields))
        {
            LLVM_DEBUG(llvm::dbgs() << "\n!! processing #" << index << " field [" << fieldInfo.id << "]\n";);           

            if (filterSpecialCases)
            {
                // filter out special fields
                if (auto strAttr = dyn_cast_or_null<mlir::StringAttr>(fieldInfo.id)) 
                {
                    if (strAttr.getValue().starts_with(".")) {
                        LLVM_DEBUG(llvm::dbgs() << "\n!! --filtered #" << index << " field [" << fieldInfo.id << "]\n";);           
                        continue;
                    }
                }
            }

            count ++;            

            MLIRPropertyAccessCodeLogic cl(compileOptions, builder, location, value, fieldInfo.id);
            // TODO: implement conditional
            auto propertyAccess = mlirGenPropertyAccessExpressionLogic(location, value, false, cl, genContext); 
            EXIT_IF_FAILED_OR_NO_VALUE(propertyAccess)

            auto value = V(propertyAccess);
            if (value.getType() != fieldInfo.type)
            {
                CAST(value, location, fieldInfo.type, value, genContext)
            }

            values.push_back(value);
        }

        if (count != values.size())
        {
            emitError(location)
                << "count of fields (" << count << ") in destination is not matching to '" << to_print(value.getType()) << "'";            
            return mlir::failure();
        }

        return mlir::success();
    }      

    ValueOrLogicalResult MLIRGenImpl::mapTupleToFields(mlir::Location location, SmallVector<mlir::Value> &values, mlir::Value value, mlir_ts::TupleType srcTupleType, 
        ::llvm::ArrayRef<::mlir::typescript::FieldInfo> fields, bool filterSpecialCases, const GenContext &genContext, bool errorAsWarning)
    {
        auto count = 0;
        for (auto [index, fieldInfo] : enumerate(fields))
        {
            LLVM_DEBUG(llvm::dbgs() << "\n!! processing #" << index << " field [" << fieldInfo.id << "]\n";);           

            if (filterSpecialCases)
            {
                // filter out special fields
                if (auto strAttr = dyn_cast_or_null<mlir::StringAttr>(fieldInfo.id)) 
                {
                    if (strAttr.getValue().starts_with(".")) {
                        LLVM_DEBUG(llvm::dbgs() << "\n!! --filtered #" << index << " field [" << fieldInfo.id << "]\n";);           
                        continue;
                    }
                }
            }

            count ++;
            if (fieldInfo.id == mlir::Attribute() || (index < srcTupleType.size() && srcTupleType.getFieldInfo(index).id == mlir::Attribute()))
            {
                if (index >= srcTupleType.size() && isa<mlir_ts::OptionalType>(fieldInfo.type))
                {
                    // add undefined value
                    auto undefVal = builder.create<mlir_ts::OptionalUndefOp>(location, fieldInfo.type);
                    values.push_back(undefVal);
                    continue;
                }

                MLIRPropertyAccessCodeLogic cl(compileOptions, builder, location, value, builder.getI32IntegerAttr(index));
                auto value = cl.Tuple(srcTupleType, true);
                VALIDATE(value, location)
                values.push_back(value);
            }
            else
            {
                // access by field name
                auto fieldIndex = srcTupleType.getIndex(fieldInfo.id);
                if (fieldIndex < 0)
                {
                    if (isa<mlir_ts::OptionalType>(fieldInfo.type))
                    {
                        // add undefined value
                        auto undefVal = builder.create<mlir_ts::OptionalUndefOp>(location, fieldInfo.type);
                        values.push_back(undefVal);
                        continue;
                    }

                    if (errorAsWarning)
                    {
                        emitWarning(location)
                            << "field " << fieldInfo.id << " can't be found in tuple '" << to_print(srcTupleType) << "'";

                        // add undefined value
                        auto undefVal = builder.create<mlir_ts::UndefOp>(location, fieldInfo.type);
                        values.push_back(undefVal);
                        continue;
                    }
                    
                    emitError(location)
                        << "field " << fieldInfo.id << " can't be found in tuple '" << to_print(srcTupleType) << "'";
                    return mlir::failure();
                }                

                MLIRPropertyAccessCodeLogic cl(compileOptions, builder, location, value, fieldInfo.id);
                // TODO: implement conditional
                auto propertyAccess = mlirGenPropertyAccessExpressionLogic(location, value, false, cl, genContext); 
                EXIT_IF_FAILED_OR_NO_VALUE(propertyAccess)

                auto value = V(propertyAccess);
                if (value.getType() != fieldInfo.type)
                {
                    CAST(value, location, fieldInfo.type, value, genContext)
                }

                values.push_back(value);
            }
        }

        if (count != values.size())
        {
            emitError(location)
                << "count of fields (" << count << ") in destination is not matching to '" << to_print(srcTupleType) << "'";            
            return mlir::failure();
        }

        return mlir::success();
    }    

    ValueOrLogicalResult MLIRGenImpl::castTupleToTuple(mlir::Location location, mlir::Value value, mlir_ts::TupleType srcTupleType, 
        ArrayRef<mlir_ts::FieldInfo> fields, const GenContext &genContext, bool errorAsWarning)
    {
        SmallVector<mlir::Value> values;

        auto result = mapTupleToFields(location, values, value, srcTupleType, fields, false, genContext, errorAsWarning);
        if (mlir::failed(result))
        {
            return mlir::failure();
        }

        SmallVector<::mlir::typescript::FieldInfo> fieldsForTuple;
        fieldsForTuple.append(fields.begin(), fields.end());
        return V(builder.create<mlir_ts::CreateTupleOp>(location, getTupleType(fieldsForTuple), values));
    }    

    ValueOrLogicalResult MLIRGenImpl::NewClassInstanceWithSettingFields(mlir::Location location, mlir_ts::ClassType classType, 
        ArrayRef<mlir_ts::FieldInfo> fields, ArrayRef<mlir::Value> values, const GenContext &genContext)
    {
        SmallVector<mlir::Value, 4> operands;
        auto newInstanceOfClass = NewClassInstance(location, classType, operands, genContext);
        // TODO: assign fields to values
        
        auto valueIndex = 0;
        for (auto fieldInfo : fields)
        {
            // filter out special fields
            if (auto strAttr = dyn_cast_or_null<mlir::StringAttr>(fieldInfo.id)) 
            {
                if (strAttr.getValue().starts_with(".")) {
                    continue;
                }
            }            

            auto value = values[valueIndex];

            MLIRPropertyAccessCodeLogic cl(compileOptions, builder, location, newInstanceOfClass, fieldInfo.id);
            // TODO: implement conditional
            auto propertyAccess = mlirGenPropertyAccessExpressionLogic(location, newInstanceOfClass, false, cl, genContext); 
            EXIT_IF_FAILED_OR_NO_VALUE(propertyAccess)

            auto property = V(propertyAccess);
            if (value.getType() != fieldInfo.type)
            {
                CAST(value, location, fieldInfo.type, value, genContext)
            }

            auto result = mlirGenSaveLogicOneItem(location, property, value, genContext);
            EXIT_IF_FAILED_OR_NO_VALUE(result)

            valueIndex++;
        }

        return newInstanceOfClass;
    }

    ValueOrLogicalResult MLIRGenImpl::castTupleToClass(mlir::Location location, mlir::Value value, mlir_ts::TupleType srcTupleType, 
        ArrayRef<mlir_ts::FieldInfo> fields, mlir_ts::ClassType classType, const GenContext &genContext, bool errorAsWarning)
    {
        SmallVector<mlir::Value> values;
        
        auto result = mapTupleToFields(location, values, value, srcTupleType, fields, true, genContext, errorAsWarning);
        if (mlir::failed(result))
        {
            return mlir::failure();
        }

        return NewClassInstanceWithSettingFields(location, classType, fields, values, genContext);
    }    

    ValueOrLogicalResult MLIRGenImpl::castFieldsToClass(mlir::Location location, mlir::Value value, 
        ::llvm::ArrayRef<::mlir::typescript::FieldInfo> fields, 
        mlir_ts::ClassType classType, const GenContext &genContext, bool errorAsWarning)
    {
        SmallVector<mlir::Value> values;
        
        auto result = selectFieldsValues(location, values, value, fields, true, genContext, errorAsWarning);
        if (mlir::failed(result))
        {
            return mlir::failure();
        }

        return NewClassInstanceWithSettingFields(location, classType, fields, values, genContext);
    }

    ValueOrLogicalResult MLIRGenImpl::castConstArrayToString(mlir::Location location, mlir::Value value, const GenContext &genContext)
    {
        if (auto constArray = dyn_cast<mlir_ts::ConstArrayType>(value.getType()))
        {
            auto stringType = getStringType();
            SmallVector<mlir::Value, 4> strs;

            auto spaceText = " ";
            auto spaceValue = builder.create<mlir_ts::ConstantOp>(location, stringType, getStringAttr(spaceText));

            auto spanText = ",";
            auto spanValue = builder.create<mlir_ts::ConstantOp>(location, stringType, getStringAttr(spanText));

            auto beginText = "[";
            auto beginValue = builder.create<mlir_ts::ConstantOp>(location, stringType, getStringAttr(beginText));

            auto endText = "]";
            auto endValue = builder.create<mlir_ts::ConstantOp>(location, stringType, getStringAttr(endText));

            strs.push_back(beginValue);

            auto constantOp = value.getDefiningOp<mlir_ts::ConstantOp>();
            auto arrayAttr = mlir::cast<mlir::ArrayAttr>(constantOp.getValue());
            for (auto [index, val] : enumerate(arrayAttr))
            {
                if (index > 0) 
                {
                    // text
                    strs.push_back(spanValue);
                }

                // we need to convert it into string
                if (auto typedAttr = dyn_cast<mlir::TypedAttr>(val))
                {
                    strs.push_back(spaceValue);

                    auto itemConstValue = builder.create<mlir_ts::ConstantOp>(location, typedAttr);
                    if (itemConstValue.getType() != stringType)
                    {
                        CAST_A(convertedValue, location, stringType, itemConstValue, genContext);
                        strs.push_back(convertedValue);
                    }
                    else
                    {
                        strs.push_back(itemConstValue);                
                    }                    
                }
                else
                {
                    return mlir::failure();
                }
            }

            if (strs.size() > 1)
            {
                strs.push_back(spaceValue);
            }

            strs.push_back(endValue);

            if (strs.size() <= 0)
            {
                return V(builder.create<mlir_ts::ConstantOp>(location, stringType, getStringAttr("")));
            }

            auto concatValues =
                builder.create<mlir_ts::StringConcatOp>(location, stringType, mlir::ArrayRef<mlir::Value>{strs});

            return V(concatValues);    
        }    

        return mlir::failure();
    }     

    ValueOrLogicalResult MLIRGenImpl::castTupleToString(mlir::Location location, mlir::Value value, mlir_ts::TupleType tupleType,
        ::llvm::ArrayRef<::mlir::typescript::FieldInfo> fields, const GenContext &genContext)
    {
        auto stringType = getStringType();
        SmallVector<mlir::Value, 4> strs;

        auto spaceText = " ";
        auto spaceValue = builder.create<mlir_ts::ConstantOp>(location, stringType, getStringAttr(spaceText));

        auto fieldSepText = ": ";
        auto fieldSepValue = builder.create<mlir_ts::ConstantOp>(location, stringType, getStringAttr(fieldSepText));

        auto spanText = ",";
        auto spanValue = builder.create<mlir_ts::ConstantOp>(location, stringType, getStringAttr(spanText));

        auto quotText = "'";
        auto quotValue = builder.create<mlir_ts::ConstantOp>(location, stringType, getStringAttr(quotText));

        auto beginText = "{";
        auto beginValue = builder.create<mlir_ts::ConstantOp>(location, stringType, getStringAttr(beginText));

        auto endText = "}";
        auto endValue = builder.create<mlir_ts::ConstantOp>(location, stringType, getStringAttr(endText));

        strs.push_back(beginValue);

        for (auto [index, fieldInfo] : enumerate(fields))
        {
            LLVM_DEBUG(llvm::dbgs() << "\n!! processing #" << index << " field [" << fieldInfo.id << "]\n";);           

            if (index > 0) 
            {
                // text
                strs.push_back(spanValue);
            }

            strs.push_back(spaceValue);
            if (fieldInfo.id)
            {
                auto fieldNameValue = builder.create<mlir_ts::ConstantOp>(location, stringType, fieldInfo.id);
                strs.push_back(fieldNameValue);
                strs.push_back(fieldSepValue);
            }

            MLIRPropertyAccessCodeLogic cl(compileOptions, builder, location, value, builder.getI32IntegerAttr(index));
            auto fieldValue = cl.Tuple(tupleType, true);
            VALIDATE(value, location)

            if (fieldValue.getType() != stringType)
            {
                CAST(fieldValue, location, stringType, fieldValue, genContext);
                // expr value
                strs.push_back(fieldValue);
            }
            else
            {
                // expr value
                strs.push_back(quotValue);
                strs.push_back(fieldValue);
                strs.push_back(quotValue);
            }
        }

        if (strs.size() > 1)
        {
            strs.push_back(spaceValue);
        }

        strs.push_back(endValue);

        if (strs.size() <= 0)
        {
            return V(builder.create<mlir_ts::ConstantOp>(location, stringType, getStringAttr("")));
        }

        auto concatValues =
            builder.create<mlir_ts::StringConcatOp>(location, stringType, mlir::ArrayRef<mlir::Value>{strs});

        return V(concatValues);        
    }       

    ValueOrLogicalResult MLIRGenImpl::generatingStaticNewCtorForClass(mlir::Location location, ClassInfo::TypePtr classInfo, int posIndex, const GenContext &genContext)
    {
        if (auto classConstrMethodInfo = classInfo->findMethod(CONSTRUCTOR_NAME))
        {
            auto funcWithReturnClass = getFunctionType(
                classConstrMethodInfo->funcType.getInputs().slice(1) /*to remove this*/, 
                {classInfo->classType}, 
                classConstrMethodInfo->funcType.isVarArg());
            auto foundNewCtoreStaticMethodFullName = generateSynthMethodToCallNewCtor(location, classInfo, classInfo->fullName, posIndex, funcWithReturnClass, 0, genContext);
            if (foundNewCtoreStaticMethodFullName.empty())
            {
                return mlir::failure();
            }

            auto symbOp = builder.create<mlir_ts::SymbolRefOp>(
                location, funcWithReturnClass,
                mlir::FlatSymbolRefAttr::get(builder.getContext(), foundNewCtoreStaticMethodFullName));
        
            return V(symbOp);
        }
        else
        {
            emitError(location) << "constructor can't be found";
            return mlir::failure();
        }
    }

    ValueOrLogicalResult MLIRGenImpl::castClassToTuple(mlir::Location location, mlir::Value value, mlir_ts::ClassType classType, 
        ::llvm::ArrayRef<::mlir::typescript::FieldInfo> fields, const GenContext &genContext)
    {
        auto classInfo = getClassInfoByFullName(classType.getName().getValue());
        assert(classInfo);            

        auto newCtorAttr = MLIRHelper::TupleFieldName(NEW_CTOR_METHOD_NAME, builder.getContext());
        SmallVector<mlir::Value> values;
        for (auto [posIndex, fieldInfo] : enumerate(fields))
        {
            auto foundField = false;                                        
            auto classFieldInfo = classInfo->findField(fieldInfo.id, foundField);
            if (!foundField)
            {
                // TODO: generate method wrapper for calling new/ctor method
                if (fieldInfo.id == newCtorAttr)
                {
                    auto newCtorSymbOp = generatingStaticNewCtorForClass(location, classInfo, posIndex, genContext);
                    EXIT_IF_FAILED_OR_NO_VALUE(newCtorSymbOp)
                    values.push_back(newCtorSymbOp);
                    continue;
                }

                emitError(location)
                    << "field " << fieldInfo.id << " can't be found in class '" << classInfo->fullName << "'";
                return mlir::failure();
            }                

            MLIRPropertyAccessCodeLogic cl(compileOptions, builder, location, value, fieldInfo.id);
            // TODO: implemenet conditional
            mlir::Value propertyAccess = mlirGenPropertyAccessExpressionLogic(location, value, false, cl, genContext); 
            if (propertyAccess)
            {
                values.push_back(propertyAccess);
            }
        }

        if (fields.size() != values.size())
        {
            return mlir::failure();
        }        

        SmallVector<::mlir::typescript::FieldInfo> fieldsForTuple;
        fieldsForTuple.append(fields.begin(), fields.end());
        return V(builder.create<mlir_ts::CreateTupleOp>(location, getTupleType(fieldsForTuple), values));
    }

    ValueOrLogicalResult MLIRGenImpl::castInterfaceToTuple(mlir::Location location, mlir::Value value, mlir_ts::InterfaceType interfaceType, 
        ::llvm::ArrayRef<::mlir::typescript::FieldInfo> fields, const GenContext &genContext)
    {
        auto interfaceInfo = getInterfaceInfoByFullName(interfaceType.getName().getValue());
        assert(interfaceInfo);            

        SmallVector<mlir::Value> values;
        for (auto fieldInfo : fields)
        {
            auto classFieldInfo = interfaceInfo->findField(fieldInfo.id);
            if (!classFieldInfo)
            {
                emitError(location)
                    << "field '" << fieldInfo.id << "' can't be found "
                    << "' in interface '" << interfaceInfo->fullName << "'";
                return mlir::failure();
            }                

            MLIRPropertyAccessCodeLogic cl(compileOptions, builder, location, value, fieldInfo.id);
            // TODO: implemenet conditional
            mlir::Value propertyAccess = mlirGenPropertyAccessExpressionLogic(
                location, value, classFieldInfo->isConditional, cl, genContext); 
            if (propertyAccess)
            {
                values.push_back(propertyAccess);
            }
        }

        if (fields.size() != values.size())
        {
            return mlir::failure();
        }          

        SmallVector<::mlir::typescript::FieldInfo> fieldsForTuple;
        fieldsForTuple.append(fields.begin(), fields.end());
        return V(builder.create<mlir_ts::CreateTupleOp>(location, getTupleType(fieldsForTuple), values));
    }

    ValueOrLogicalResult MLIRGenImpl::cast(mlir::Location location, mlir::Type type, mlir::Value value, const GenContext &genContext, bool disableStrictNullCheck)
    {
        if (!type)
        {
            return mlir::failure();
        }

        if (type == value.getType())
        {
            return value;
        }

        auto valueType = value.getType();

        LLVM_DEBUG(llvm::dbgs() << "\n!! cast " << valueType << "\n -> " << type
                                << "\n";);

        if (mlir::failed(verifyCastPreconditions(location, type, valueType, disableStrictNullCheck)))
        {
            return mlir::failure();
        }

        if (auto enumType = dyn_cast<mlir_ts::EnumType>(valueType))
        {
            value = builder.create<mlir_ts::CastOp>(location, enumType.getElementType(), value);
            valueType = value.getType();
        }        

        if (auto result = castViaToPrimitive(location, type, value, valueType, genContext))
        {
            return *result;
        }

        if (auto result = castToStringSpecialCases(location, type, value, valueType, genContext))
        {
            return *result;
        }

        if (auto result = castToInterfaceSpecialCases(location, type, value, valueType, genContext))
        {
            return *result;
        }

        if (auto result = castTupleLikeVariants(location, type, value, valueType, genContext))
        {
            return *result;
        }

        if (auto result = castToOptionalType(location, type, value, valueType, genContext))
        {
            return *result;
        }

        if (auto result = castToTaggedUnionType(location, type, value, valueType, genContext))
        {
            return *result;
        }

        // const dest: cast via the unwrapped source type instead
        if (auto constType = dyn_cast<mlir_ts::ConstType>(type))
        {
            // TODO: we can't convert array to const array

            auto currType = valueType;
            if (auto refType = dyn_cast<mlir_ts::RefType>(currType))
            {
                type = refType.getElementType();
            }
            else if (auto tupleType = dyn_cast<mlir_ts::TupleType>(currType))
            {
                type = mth.convertTupleTypeToConstTupleType(tupleType);
            }
            else
            {
                return value;
            }
        }

        if (auto result = castFromSourceSpecialCases(location, type, value, valueType, genContext))
        {
            return *result;
        }

        if (mlir::failed(verifyFunctionCastRules(location, type, value, valueType, genContext)))
        {
            return mlir::failure();
        }

        if (auto result = castExtensionFunctionType(location, type, value, valueType))
        {
            return *result;
        }

        if (mlir::failed(verifyCastCompatibility(location, type, valueType)))
        {
            return mlir::failure();
        }

        return V(builder.create<mlir_ts::CastOp>(location, type, value));
    }

    mlir::LogicalResult MLIRGenImpl::verifyCastPreconditions(mlir::Location location, mlir::Type type, mlir::Type valueType, bool disableStrictNullCheck)
    {
        if (auto litType = dyn_cast<mlir_ts::LiteralType>(type))
        {
            if (auto valLitType = dyn_cast<mlir_ts::LiteralType>(valueType))
            {
                if (litType.getValue() != valLitType.getValue())
                {
                    emitError(location, "can't cast from literal type: '") << valLitType.getValue() << "' to '" << litType.getValue() << "'";
                    return mlir::failure();
                }
            }
        }

        // strict null
        if (compileOptions.strictNullChecks && !disableStrictNullCheck)
        {
            auto effectiveType = type;
            if (auto optType = dyn_cast<mlir_ts::OptionalType>(effectiveType)) {
                effectiveType = optType.getElementType();
            }

            if (isa<mlir_ts::NullType>(valueType))
            {
                if (!isa<mlir_ts::NullType>(effectiveType) && !isa<mlir_ts::AnyType>(effectiveType))
                {
                    auto hasNullOrAny = false;
                    if (auto unionType = dyn_cast<mlir_ts::UnionType>(effectiveType))
                    {
                        auto foundType = llvm::find_if(unionType.getTypes(), [&] (auto elementOfUnionType) {
                            return elementOfUnionType == valueType || isa<mlir_ts::AnyType>(elementOfUnionType);
                        });
                        hasNullOrAny |= foundType != unionType.getTypes().end();
                    }

                    if (!hasNullOrAny)
                    {
                        emitError(location, "can't cast from 'null' to '") << to_print(type) << "' in 'strict null mode'";
                        return mlir::failure();
                    }
                }
            }
        }

        return mlir::success();
    }

    std::optional<ValueOrLogicalResult> MLIRGenImpl::castViaToPrimitive(mlir::Location location, mlir::Type type, mlir::Value value, mlir::Type valueType, const GenContext &genContext)
    {
        if ((isa<mlir_ts::StringType>(type)
            || isa<mlir_ts::NumberType>(type)
            || isa<mlir_ts::BigIntType>(type)
            || isa<mlir_ts::BooleanType>(type)
            || isa<mlir_ts::UndefinedType>(type)
            || isa<mlir_ts::SymbolType>(type)
            || isa<mlir_ts::NullType>(type))
            && (isa<mlir_ts::ClassType>(valueType)
                || isa<mlir_ts::ClassStorageType>(valueType)
                || isa<mlir_ts::ObjectType>(valueType)
                || isa<mlir_ts::InterfaceType>(valueType)
                || isa<mlir_ts::TupleType>(valueType)
                || isa<mlir_ts::ConstTupleType>(valueType)))
        {
            // check if we need to call toPrimitive
            if (auto toPrimitiveType = evaluateProperty(location, value, SYMBOL_TO_PRIMITIVE, genContext))
            {
                NodeFactory nf(NodeFactoryFlags::None);
                Expression hint;

                mlir::TypeSwitch<mlir::Type>(type)
                    .template Case<mlir_ts::StringType>([&](auto) {
                        hint = nf.createStringLiteral(S("string"));
                    })
                    .template Case<mlir_ts::NumberType>([&](auto) {
                        hint = nf.createStringLiteral(S("number"));
                    })
                    .template Case<mlir_ts::BigIntType>([&](auto) {
                        hint = nf.createStringLiteral(S("bigint"));
                    })
                    .template Case<mlir_ts::BooleanType>([&](auto) {
                        hint = nf.createStringLiteral(S("boolean"));
                    })
                    .template Case<mlir_ts::UndefinedType>([&](auto) {
                        hint = nf.createStringLiteral(S(UNDEFINED_NAME));
                    })
                    .template Case<mlir_ts::SymbolType>([&](auto) {
                        hint = nf.createStringLiteral(S("symbol"));
                    })
                    .template Case<mlir_ts::NullType>([&](auto) {
                        hint = nf.createStringLiteral(S("null"));
                    })
                    .Default([&](auto type) {});

                auto callResult = mlirGenCallThisMethod(location, value, SYMBOL_TO_PRIMITIVE, undefined, {hint}, genContext);
                EXIT_IF_FAILED(callResult);
                auto callResultValue = V(callResult);
                if (isa<mlir_ts::UnionType>(callResultValue.getType()))
                {
                    return V(builder.create<mlir_ts::GetValueFromUnionOp>(location, type, callResultValue));
                }

                auto castValue = cast(location, type, callResultValue, genContext);
                EXIT_IF_FAILED_OR_NO_VALUE(castValue);
                return castValue;
            }
        }

        return std::nullopt;
    }

    std::optional<ValueOrLogicalResult> MLIRGenImpl::castToStringSpecialCases(mlir::Location location, mlir::Type type, mlir::Value value, mlir::Type valueType, const GenContext &genContext)
    {
        if (auto stringType = dyn_cast<mlir_ts::StringType>(type))
        {
            if (auto classType = dyn_cast<mlir_ts::ClassType>(valueType))
            {
                auto res = mlirGenCallThisMethod(location, value, "get_" SYMBOL_TO_STRING_TAG, undefined, undefined, genContext);
                if (!res.failed_or_no_value())
                {
                    return res;
                }

                return mlirGenCallThisMethod(location, value, TO_STRING, undefined, undefined, genContext);
            }

            if (auto arrayType = dyn_cast<mlir_ts::ConstArrayType>(valueType))
            {
                return castConstArrayToString(location, value, genContext);
            }
            else if (auto arrayType = dyn_cast<mlir_ts::ArrayType>(valueType))
            {
                // we evaluate property to allow to compile code in "generic methods" with "typeof" conditions
                // if we throw error here generic method with "if (false)" condition will generate code which
                // will be removed but because of error, the compilation process will be stopped
                if (auto toStringMethod = evaluateProperty(location, value, TO_STRING, genContext))
                {
                    return mlirGenCallThisMethod(location, value, TO_STRING, undefined, undefined, genContext);
                }
            }

            if (auto srcConstTupleType = dyn_cast<mlir_ts::ConstTupleType>(valueType))
            {
                if (auto toStringMethod = evaluateProperty(location, value, TO_STRING, genContext))
                {
                    return mlirGenCallThisMethod(location, value, TO_STRING, undefined, undefined, genContext);
                }

                return castTupleToString(location, value, mth.convertConstTupleTypeToTupleType(srcConstTupleType),
                    srcConstTupleType.getFields(), genContext);
            }
            else if (auto srcTupleType = dyn_cast<mlir_ts::TupleType>(valueType))
            {
                if (auto toStringMethod = evaluateProperty(location, value, TO_STRING, genContext))
                {
                    return mlirGenCallThisMethod(location, value, TO_STRING, undefined, undefined, genContext);
                }

                return castTupleToString(location, value, srcTupleType, srcTupleType.getFields(), genContext);
            }
        }

        return std::nullopt;
    }

    std::optional<ValueOrLogicalResult> MLIRGenImpl::castToInterfaceSpecialCases(mlir::Location location, mlir::Type type, mlir::Value value, mlir::Type valueType, const GenContext &genContext)
    {
        if (auto interfaceType = dyn_cast<mlir_ts::InterfaceType>(type))
        {
            if (auto classType = dyn_cast<mlir_ts::ClassType>(valueType))
            {
                auto result = mlirGenPropertyAccessExpression(location, value, VTABLE_NAME, genContext);
                auto vtableAccess = V(result);

                auto classInfo = getClassInfoByFullName(classType.getName().getValue());
                assert(classInfo);

                auto implementIndex = classInfo->getImplementIndex(interfaceType.getName().getValue());
                if (implementIndex >= 0)
                {
                    auto interfaceVirtTableIndex = classInfo->implements[implementIndex].virtualIndex;

                    assert(genContext.allowPartialResolve || interfaceVirtTableIndex >= 0);

                    auto interfaceVTablePtr = builder.create<mlir_ts::VTableOffsetRefOp>(
                        location, mth.getInterfaceVTableType(interfaceType), vtableAccess, interfaceVirtTableIndex);

                    auto newInterface = builder.create<mlir_ts::NewInterfaceOp>(
                        location, mlir::TypeRange{interfaceType}, value, interfaceVTablePtr);
                    return V(newInterface);
                }

                // create interface vtable from current class
                auto interfaceInfo = getInterfaceInfoByFullName(interfaceType.getName().getValue());
                assert(interfaceInfo);

                if (auto createdInterfaceVTableForClass =
                        mlirGenCreateInterfaceVTableForClass(location, classInfo, interfaceInfo, genContext))
                {
                    LLVM_DEBUG(llvm::dbgs() << "\n!!"
                                            << "@ created interface:" << V(createdInterfaceVTableForClass) << "\n";);
                    auto newInterface = builder.create<mlir_ts::NewInterfaceOp>(
                        location, mlir::TypeRange{interfaceType}, value, createdInterfaceVTableForClass);

                    return V(newInterface);
                }

                emitError(location) << "type: " << classType.getName() << " missing interface: " << interfaceType.getName();
                return mlir::failure();
            }

            // tuple to interface
            if (auto constTupleType = dyn_cast<mlir_ts::ConstTupleType>(valueType))
            {
                return castTupleToInterface(location, value, constTupleType, interfaceType, genContext);
            }

            if (auto tupleType = dyn_cast<mlir_ts::TupleType>(valueType))
            {
                return castTupleToInterface(location, value, tupleType, interfaceType, genContext);
            }

            // object to interface
            if (auto objectType = dyn_cast<mlir_ts::ObjectType>(valueType))
            {
                return castObjectToInterface(location, value, objectType, interfaceType, genContext);
            }
        }

        return std::nullopt;
    }

    std::optional<ValueOrLogicalResult> MLIRGenImpl::castTupleLikeVariants(mlir::Location location, mlir::Type type, mlir::Value value, mlir::Type valueType, const GenContext &genContext)
    {
        // const tuple to tuple
        if (auto srcConstTupleType = dyn_cast<mlir_ts::ConstTupleType>(valueType))
        {
            ::llvm::ArrayRef<::mlir::typescript::FieldInfo> fields;
            if (auto tupleType = dyn_cast<mlir_ts::TupleType>(type))
            {
                fields = tupleType.getFields();
                return castTupleToTuple(location, value, mth.convertConstTupleTypeToTupleType(srcConstTupleType), fields, genContext);
            }
            else if (auto constTupleType = dyn_cast<mlir_ts::ConstTupleType>(type))
            {
                fields = constTupleType.getFields();
                return castTupleToTuple(location, value, mth.convertConstTupleTypeToTupleType(srcConstTupleType), fields, genContext);
            }
            else if (auto classType = dyn_cast<mlir_ts::ClassType>(type))
            {
                fields = mlir::cast<mlir_ts::ClassStorageType>(classType.getStorageType()).getFields();
                return castTupleToClass(location, value, mth.convertConstTupleTypeToTupleType(srcConstTupleType), fields, classType, genContext);
            }
            else if (auto funcType = dyn_cast<mlir_ts::FunctionType>(type))
            {
                emitError(location, "invalid cast from ") << to_print(valueType) << " to " << to_print(type);
                return mlir::failure();
            }
        }

        // tuple to tuple
        if (auto srcTupleType = dyn_cast<mlir_ts::TupleType>(valueType))
        {
            ::llvm::ArrayRef<::mlir::typescript::FieldInfo> fields;
            if (auto tupleType = dyn_cast<mlir_ts::TupleType>(type))
            {
                fields = tupleType.getFields();
                return castTupleToTuple(location, value, srcTupleType, fields, genContext);
            }
            else if (auto constTupleType = dyn_cast<mlir_ts::ConstTupleType>(type))
            {
                fields = constTupleType.getFields();
                return castTupleToTuple(location, value, srcTupleType, fields, genContext);
            }
            else if (auto classType = dyn_cast<mlir_ts::ClassType>(type))
            {
                emitError(location, "invalid cast from ") << to_print(valueType) << " to " << to_print(type);
                return mlir::failure();
            }
            else if (auto funcType = dyn_cast<mlir_ts::FunctionType>(type))
            {
                emitError(location, "invalid cast from ") << to_print(valueType) << " to " << to_print(type);
                return mlir::failure();
            }
        }

        // class to tuple
        if (auto classType = dyn_cast<mlir_ts::ClassType>(valueType))
        {
            ::llvm::ArrayRef<::mlir::typescript::FieldInfo> fields;
            if (auto tupleType = dyn_cast<mlir_ts::TupleType>(type))
            {
                fields = tupleType.getFields();
                return castClassToTuple(location, value, classType, fields, genContext);
            }
            else if (auto constTupleType = dyn_cast<mlir_ts::ConstTupleType>(type))
            {
                fields = constTupleType.getFields();
                return castClassToTuple(location, value, classType, fields, genContext);
            }
        }

        // interface to tuple
        if (auto interfaceType = dyn_cast<mlir_ts::InterfaceType>(valueType))
        {
            ::llvm::ArrayRef<::mlir::typescript::FieldInfo> fields;
            if (auto tupleType = dyn_cast<mlir_ts::TupleType>(type))
            {
                fields = tupleType.getFields();
                return castInterfaceToTuple(location, value, interfaceType, fields, genContext);
            }
            else if (auto constTupleType = dyn_cast<mlir_ts::ConstTupleType>(type))
            {
                fields = constTupleType.getFields();
                return castInterfaceToTuple(location, value, interfaceType, fields, genContext);
            }
            else if (auto classType = dyn_cast<mlir_ts::ClassType>(type))
            {
                fields = mlir::cast<mlir_ts::ClassStorageType>(classType.getStorageType()).getFields();
                return castFieldsToClass(location, value, fields, classType, genContext);
            }
        }

        // boxed object (see docs/object-literal-boxing-design.md) to tuple: unbox the pointer
        // (reverse of the NewOp+StoreOp+CastOp boxing recipe) then reuse tuple-to-tuple casting.
        // This is a copy -- the destination tuple/const-tuple no longer aliases the boxed object.
        if (auto srcObjectType = dyn_cast<mlir_ts::ObjectType>(valueType))
        {
            if (isa<mlir_ts::TupleType>(type) || isa<mlir_ts::ConstTupleType>(type))
            {
                // storage type is whatever was boxed (tuple/const-tuple); normalize to a
                // mutable TupleType, both for the load and for castTupleToTuple's src param.
                auto srcTupleType = mlir::cast<mlir_ts::TupleType>(
                    mth.convertConstTupleTypeToTupleType(srcObjectType.getStorageType()));

                // RefType (not ValueRefType -- PropertyRefOp's operand is restricted to
                // AnyStructRefLike, which ValueRefType is not part of); mapTupleToFields
                // recovers this ref straight back from the LoadOp below for by-field access.
                auto refType = mlir_ts::RefType::get(srcTupleType);
                auto valueAddr = builder.create<mlir_ts::CastOp>(location, refType, value);
                auto unboxedTuple = builder.create<mlir_ts::LoadOp>(location, srcTupleType, valueAddr);

                ::llvm::ArrayRef<::mlir::typescript::FieldInfo> fields;
                if (auto tupleType = dyn_cast<mlir_ts::TupleType>(type))
                {
                    fields = tupleType.getFields();
                }
                else
                {
                    fields = mlir::cast<mlir_ts::ConstTupleType>(type).getFields();
                }

                return castTupleToTuple(location, unboxedTuple, srcTupleType, fields, genContext);
            }
        }

        return std::nullopt;
    }

    std::optional<ValueOrLogicalResult> MLIRGenImpl::castToOptionalType(mlir::Location location, mlir::Type type, mlir::Value value, mlir::Type valueType, const GenContext &genContext)
    {
        if (auto optType = dyn_cast<mlir_ts::OptionalType>(type))
        {
            if (valueType == getUndefinedType())
            {
                return V(builder.create<mlir_ts::OptionalUndefOp>(location, optType));
            }
            else if (auto optValueType = dyn_cast<mlir_ts::OptionalType>(valueType))
            {
                auto condValue = builder.create<mlir_ts::HasValueOp>(location, getBooleanType(), value);
                return optionalValueOrUndefined(
                    location,
                    condValue,
                    [&](auto genContext)
                    {
                        auto valueFromOptional = builder.create<mlir_ts::ValueOp>(location, optValueType.getElementType(), value);
                        return cast(location, optType.getElementType(), valueFromOptional, genContext);
                    },
                    genContext);
            }
            else
            {
                CAST_A(valueCasted, location, optType.getElementType(), value, genContext);
                return V(builder.create<mlir_ts::OptionalValueOp>(location, optType, valueCasted));
            }
        }

        return std::nullopt;
    }

    std::optional<ValueOrLogicalResult> MLIRGenImpl::castToTaggedUnionType(mlir::Location location, mlir::Type type, mlir::Value value, mlir::Type valueType, const GenContext &genContext)
    {
        if (auto unionType = dyn_cast<mlir_ts::UnionType>(type))
        {
            mlir::Type baseType;
            if (mth.isUnionTypeNeedsTag(location, unionType, baseType))
            {
                auto types = unionType.getTypes();
                if (std::find(types.begin(), types.end(), valueType) == types.end())
                {
                    // find which type we can cast to
                    for (auto subType : types)
                    {
                        if (mth.canCastFromTo(location, valueType, subType))
                        {
                            CAST(value, location, subType, value, genContext);
                            return V(builder.create<mlir_ts::CastOp>(location, type, value));
                        }
                    }
                }
                else
                {
                    return V(builder.create<mlir_ts::CastOp>(location, type, value));
                }
            }
        }

        return std::nullopt;
    }

    std::optional<ValueOrLogicalResult> MLIRGenImpl::castFromSourceSpecialCases(mlir::Location location, mlir::Type type, mlir::Value value, mlir::Type valueType, const GenContext &genContext)
    {
        // union type to <basic type>
        if (auto unionType = dyn_cast<mlir_ts::UnionType>(valueType))
        {
            // union -> any will be done later in CastLogic
            auto toAny = dyn_cast<mlir_ts::AnyType>(type);
            mlir::Type baseType;
            if (!toAny && mth.isUnionTypeNeedsTag(location, unionType, baseType))
            {
                return castFromUnion(location, type, value, genContext);
            }
        }

        // TODO: issue is with casting to Boolean type from union type for example, you need to cast optional type to boolean to check value
        // get rid of using "OptionalType" and use Union for it with "| undefined"
        // unwrapping optional value to work with union inside, we need it as ' | undefined ' is part of union type
        if (auto optType = dyn_cast<mlir_ts::OptionalType>(valueType))
        {
            if (isa<mlir_ts::UnionType>(optType.getElementType()))
            {
                auto val = V(builder.create<mlir_ts::ValueOrDefaultOp>(location, optType.getElementType(), value));
                CAST_A(unwrappedValue, location, type, val, genContext);
                return unwrappedValue;
            }

            // optional to value cast(when we change types)
            auto hasValue = builder.create<mlir_ts::HasValueOp>(location, mlir_ts::BooleanType::get(builder.getContext()), value);

            MLIRCodeLogicHelper mclh(builder, location, compileOptions);
            auto castedVal = mclh.conditionalValue(hasValue,
                [&]() {
                    auto optValue = builder.create<mlir_ts::ValueOp>(location, optType.getElementType(), value);
                    return cast(location, type, optValue, genContext);
                },
                [&](mlir::Type trueType) {
                    if (mlir::isa<mlir_ts::StringType>(type))
                    {
                        auto undefValue = builder.create<mlir_ts::UndefOp>(location, mlir_ts::UndefinedType::get(builder.getContext()));
                        return V(cast(location, type, undefValue, genContext));
                    }

                    if (auto destOptType = mlir::isa<mlir_ts::OptionalType>(type))
                    {
                        auto destOptValue = builder.create<mlir_ts::OptionalUndefOp>(location, type);
                        return V(destOptValue);
                    }

                    auto defValue = builder.create<mlir_ts::DefaultOp>(location, type);
                    return V(defValue);
                });
            return castedVal;
        }

        // unboxing
        if (auto anyType = dyn_cast<mlir_ts::AnyType>(valueType))
        {
            if (isa<mlir_ts::NumberType>(type)
                || isa<mlir_ts::BooleanType>(type)
                || isa<mlir_ts::StringType>(type)
                || isa<mlir_ts::BigIntType>(type)
                || isa<mlir::IntegerType>(type)
                || isa<mlir::FloatType>(type)
                || isa<mlir_ts::ClassType>(type))
            {
                return castPrimitiveTypeFromAny(location, type, value, genContext);
            }
        }

        // opaque to hybrid func
        if (auto opaqueType = dyn_cast<mlir_ts::OpaqueType>(valueType))
        {
            if (auto funcType = dyn_cast<mlir_ts::FunctionType>(type))
            {
                return V(builder.create<mlir_ts::CastOp>(location, type, value));
            }

            if (auto hybridFuncType = dyn_cast<mlir_ts::HybridFunctionType>(type))
            {
                auto funcValue = builder.create<mlir_ts::CastOp>(
                    location,
                    mlir_ts::FunctionType::get(builder.getContext(), hybridFuncType.getInputs(), hybridFuncType.getResults(), hybridFuncType.isVarArg()),
                    value);
                return V(builder.create<mlir_ts::CastOp>(location, type, funcValue));
            }
        }

        return std::nullopt;
    }

    mlir::LogicalResult MLIRGenImpl::verifyFunctionCastRules(mlir::Location location, mlir::Type type, mlir::Value value, mlir::Type valueType, const GenContext &genContext)
    {
        if (mth.isAnyFunctionType(valueType) && mth.isAnyFunctionType(type)) {

            if (mth.isGenericType(valueType))
            {
                // need to instantiate generic method
                auto result = instantiateSpecializedFunction(location, value, type, genContext);
                EXIT_IF_FAILED(result);
            }

            // fall through to finish cast operation
            if (!mth.CanCastFunctionTypeToFunctionType(valueType, type))
            {
                emitError(location, "invalid cast from ") << to_print(valueType) << " to " << to_print(type);
                return mlir::failure();
            }

            if (!mth.isGenericType(type) && !mth.isGenericType(valueType))
            {
                // test fun types
                auto test = mth.TestFunctionTypesMatchWithObjectMethods(location, valueType, type).result == MatchResultType::Match;
                if (!test)
                {
                    emitError(location) << to_print(valueType) << " is not matching type " << to_print(type);
                    return mlir::failure();
                }
            }
        }

        return mlir::success();
    }

    std::optional<ValueOrLogicalResult> MLIRGenImpl::castExtensionFunctionType(mlir::Location location, mlir::Type type, mlir::Value value, mlir::Type valueType)
    {
        if (auto extFuncType = dyn_cast<mlir_ts::ExtensionFunctionType>(valueType))
        {
            if (auto hybridFuncType = dyn_cast<mlir_ts::HybridFunctionType>(type))
            {
                auto boundFunc = createBoundMethodFromExtensionMethod(location, value.getDefiningOp<mlir_ts::CreateExtensionFunctionOp>());
                return V(builder.create<mlir_ts::CastOp>(location, type, boundFunc));
            }

            if (auto boundFuncType = dyn_cast<mlir_ts::BoundFunctionType>(type))
            {
                auto boundFunc = createBoundMethodFromExtensionMethod(location, value.getDefiningOp<mlir_ts::CreateExtensionFunctionOp>());
                return V(builder.create<mlir_ts::CastOp>(location, type, boundFunc));
            }
        }

        return std::nullopt;
    }

    mlir::LogicalResult MLIRGenImpl::verifyCastCompatibility(mlir::Location location, mlir::Type type, mlir::Type valueType)
    {
        if (mth.isAnyFunctionType(valueType) &&
            !mth.isAnyFunctionType(type, true)
            && !isa<mlir_ts::OpaqueType>(type)
            && !isa<mlir_ts::AnyType>(type)
            && !isa<mlir_ts::BooleanType>(type)) {
            emitError(location, "invalid cast from ") << to_print(valueType) << " to " << to_print(type);
            return mlir::failure();
        }

        if (isa<mlir_ts::ArrayType>(type) && isa<mlir_ts::TupleType>(valueType)
            || isa<mlir_ts::TupleType>(type) && isa<mlir_ts::ArrayType>(valueType))
        {
            emitError(location, "invalid cast from ") << to_print(valueType) << " to " << to_print(type);
            return mlir::failure();
        }

        if (auto valueArrayType = dyn_cast<mlir_ts::ArrayType>(valueType))
        {
            if (auto arrayType = dyn_cast<mlir_ts::ArrayType>(type))
            {
                llvm::StringMap<std::pair<ts::TypeParameterDOM::TypePtr,mlir::Type>> typeParamsWithArgs;
                auto extendsResult = mth.extendsType(location, valueArrayType.getElementType(), arrayType.getElementType(), typeParamsWithArgs);
                if (extendsResult != ExtendsResult::True)
                {
                    emitError(location, "invalid cast from ") << to_print(valueType) << " to " << to_print(type)
                        << " as element type " << to_print(arrayType.getElementType()) << " is not base of type "
                        << to_print(valueArrayType.getElementType());
                    return mlir::failure();
                }
            }
        }

        if (isa<mlir_ts::ClassType>(type) || isa<mlir_ts::InterfaceType>(type))
        {
            if (isa<mlir_ts::NumberType>(valueType)
                || isa<mlir_ts::BooleanType>(valueType)
                || isa<mlir_ts::StringType>(valueType)
                || isa<mlir_ts::BigIntType>(valueType)
                || isa<mlir::IntegerType>(valueType)
                || isa<mlir::FloatType>(valueType))
            {
                emitError(location, "invalid cast from ") << to_print(valueType) << " to " << to_print(type);
                return mlir::failure();
            }
        }

        if (isa<mlir_ts::ClassType>(valueType) || isa<mlir_ts::InterfaceType>(valueType))
        {
            if (isa<mlir_ts::NumberType>(type)
                || isa<mlir_ts::BigIntType>(type)
                || isa<mlir::IntegerType>(type)
                || isa<mlir::FloatType>(type))
            {
                emitError(location, "invalid cast from ") << to_print(valueType) << " to " << to_print(type);
                return mlir::failure();
            }
        }

        return mlir::success();
    }

    ValueOrLogicalResult MLIRGenImpl::castPrimitiveTypeFromAny(mlir::Location location, mlir::Type type, mlir::Value value, const GenContext &genContext)
    {
        // info, we add "_" extra as scanner append "_" in front of "__";
        auto funcName = "___unbox";

        // we need to remove current implementation as we have different implementation per union type
        removeGenericFunctionMap(funcName);
        
        // TODO: must be improved
        stringstream ss;

        StringMap<boolean> typeOfs;
        SmallVector<mlir::Type> classInstances;
        ss << S("function __unbox<T>(a: any) : T {\n");
        auto subType = type;
        auto hasUnsupportedType = false;
        mlir::TypeSwitch<mlir::Type>(subType)
            .Case<mlir_ts::BooleanType>([&](auto _) { typeOfs["boolean"] = true; })
            .Case<mlir_ts::TypePredicateType>([&](auto _) { typeOfs["boolean"] = true; })
            .Case<mlir_ts::NumberType>([&](auto _) { typeOfs["number"] = true; })
            .Case<mlir_ts::StringType>([&](auto _) { typeOfs["string"] = true; })
            .Case<mlir_ts::CharType>([&](auto _) { typeOfs["char"] = true; })
            .Case<mlir::IntegerType>([&](auto intType_) {
                if (intType_.isSignless()) typeOfs["i" + std::to_string(intType_.getWidth())] = true; else
                if (intType_.isSigned()) typeOfs["s" + std::to_string(intType_.getWidth())] = true; else
                if (intType_.isUnsigned()) typeOfs["u" + std::to_string(intType_.getWidth())] = true; })
            .Case<mlir::FloatType>([&](auto floatType_) { typeOfs["f" + std::to_string(floatType_.getWidth())] = true; })
            .Case<mlir::IndexType>([&](auto _) { typeOfs["index"] = true; })
            .Case<mlir_ts::BigIntType>([&](auto _) { typeOfs["bigint"] = true; })
            .Case<mlir_ts::FunctionType>([&](auto _) { typeOfs["function"] = true; })
            .Case<mlir_ts::BoundFunctionType>([&](auto _) { typeOfs["function"] = true; })
            .Case<mlir_ts::HybridFunctionType>([&](auto _) { typeOfs["function"] = true; })
            .Case<mlir_ts::ExtensionFunctionType>([&](auto _) { typeOfs["function"] = true; })
            .Case<mlir_ts::ClassType>([&](auto classType_) { typeOfs["class"] = true; classInstances.push_back(classType_); })
            .Case<mlir_ts::InterfaceType>([&](auto _) { typeOfs["interface"] = true; })
            // TODO: we can't use null type here and undefined otherwise code will be cycling 
            // due to issue with TypeOf == 'null' as it should denounce UnionType into Single Type
            // review code to use null in "TypeGuard"
            .Case<mlir_ts::NullType>([&](auto _) { /* TODO: uncomment when finish with TypeGuard and null */ /*typeOfs["null"] = true;*/ })
            .Case<mlir_ts::UndefinedType>([&](auto _) { /* TODO: I don't think we need any code here */ /*typeOfs["undefined"] = true;*/ })
            .Default([&](auto type) {
                LLVM_DEBUG(llvm::dbgs() << "\n\t TypeOf NOT IMPLEMENTED for Type: " << type << "\n";);
                hasUnsupportedType = true;
            });

        if (hasUnsupportedType)
        {
            emitError(location) << "Cast from 'any' to " << to_print(type) << " is not supported";
            return mlir::failure();
        }

        auto next = false;
        for (auto& pair : typeOfs)
        {
            if (next) ss << S(" else ");

            ss << S("if (typeof a == '");
            ss << stows(pair.getKey().str());
            ss << S("') ");
            if (pair.getKey() == "class")
            {
                ss << S("{ \n");

                for (auto [index, _] : enumerate(classInstances))
                {
                    ss << S("if (a instanceof TYPE_INST_ALIAS");
                    ss << index;
                    ss << S(") return a;\n");
                }

                ss << S(" }\n");
            }
            else
            {
                ss << S("return a;\n");
            }

            next = true;
        }

        if (isa<mlir_ts::BooleanType>(type)
            || isa<mlir_ts::TypePredicateType>(type)
            || isa<mlir_ts::NumberType>(type)
            || isa<mlir_ts::StringType>(type)
            || isa<mlir_ts::CharType>(type)
            || isa<mlir_ts::BigIntType>(type)            
            || isa<mlir_ts::NullType>(type)
            || isa<mlir_ts::UndefinedType>(type)
            || isa<mlir::IntegerType>(type)
            || isa<mlir::FloatType>(type)
            || isa<mlir::IndexType>(type))
        {
            // TODO: maybe we need conditional rule here
            ss << "\nif (typeof a == 'number') return a;";
            ss << "\nif (typeof a == 'string') return a;";
            ss << "\nif (typeof a == 'boolean') return a;";
            ss << "\nif (typeof a == 'f32') return a;";
            ss << "\nif (typeof a == 'i32') return a;";
            ss << "\nif (typeof a == 's32') return a;";
            ss << "\nif (typeof a == 'u32') return a;";
            ss << "\nif (typeof a == 'bigint') return a;";
            ss << "\nif (typeof a == 'f64') return a;";
            ss << "\nif (typeof a == 'i64') return a;";
            ss << "\nif (typeof a == 's64') return a;";
            ss << "\nif (typeof a == 'u64') return a;";
            ss << "\nif (typeof a == 'char') return a;";
            ss << "\nif (typeof a == 'index') return a;";
            // TODO: we can't use it without compile_rt(fixtfsi)
            //ss << "\nif (typeof a == 'f128') return a;";
            // TODO: we can't use it without compile_rt(extendhfsf2)
            //ss << "\nif (typeof a == 'f16') return a;";
            ss << "\nif (typeof a == 'i16') return a;";
            ss << "\nif (typeof a == 's16') return a;";
            ss << "\nif (typeof a == 'u16') return a;";
            ss << "\nif (typeof a == 'i8') return a;";
            ss << "\nif (typeof a == 's8') return a;";
            ss << "\nif (typeof a == 'u8') return a;";

            if (mlir::isa<mlir_ts::StringType>(type)) {
                ss << "\nif (typeof a == 'undefined') return 'undefined';";
                ss << "\nif (typeof a == 'null') return 'null';";
            }
        }

        ss << "\nthrow \"Can't cast from any type\";\n";                    
        ss << S("}\n");

        auto src = ss.str();

        {
            MLIRLocationGuard vgLoc(overwriteLoc); 
            overwriteLoc = location;

            if (mlir::failed(parsePartialStatements(src)))
            {
                assert(false);
                return mlir::failure();
            }
        }

        auto funcResult = resolveIdentifier(location, funcName, genContext);

        assert(funcResult);

        GenContext funcCallGenContext(genContext);
        funcCallGenContext.typeAliasMap.insert({".TYPE_ALIAS_T", type});

        for (auto [index, instanceOfType] : enumerate(classInstances))
        {
            funcCallGenContext.typeAliasMap.insert({"TYPE_INST_ALIAS" + std::to_string(index), instanceOfType});
        }

        SmallVector<mlir::Value, 4> operands;
        operands.push_back(value);

        NodeFactory nf(NodeFactoryFlags::None);
        return mlirGenCallExpression(
            location, 
            funcResult, 
            { 
                nf.createTypeReferenceNode(nf.createIdentifier(S(".TYPE_ALIAS_T")).as<Node>()), 
            }, 
            operands, 
            funcCallGenContext);
    }

    ValueOrLogicalResult MLIRGenImpl::castFromUnion(mlir::Location location, mlir::Type type, mlir::Value value, const GenContext &genContext)
    {
        if (auto unionType = dyn_cast<mlir_ts::UnionType>(value.getType()))
        {
            if (auto normalizedUnion = dyn_cast<mlir_ts::UnionType>(mth.getUnionTypeWithMerge(location, unionType.getTypes())))
            {
                // info, we add "_" extra as scanner append "_" in front of "__";
                auto funcName = "___cast";

                // we need to remove current implementation as we have different implementation per union type
                removeGenericFunctionMap(funcName);
                
                // TODO: must be improved
                stringstream ss;

                auto isNullDest = isa<mlir_ts::NullType>(type);

                StringMap<boolean> typeOfs;
                SmallVector<mlir::Type> classInstances;
                SmallVector<mlir::Type> tupleTypes;
                auto hasUnsupportedType = false;
                ss << S("function __cast<T, U>(t: T) : U {\n");
                for (auto subType : normalizedUnion.getTypes())
                {
                /*
                        if (typeof a == 'number') return a; \
                        if (typeof a == 'string') return a; \
                        if (typeof a == 'i32') return a; \
                        if (typeof a == 'class') if (a instanceof U) return a; \
                        return null; \"
                */

                    // true is nullable, false is not
                    mlir::TypeSwitch<mlir::Type>(subType)
                        .Case<mlir_ts::BooleanType>([&](auto _) { typeOfs["boolean"] = false; })
                        .Case<mlir_ts::TypePredicateType>([&](auto _) { typeOfs["boolean"] = false; })
                        .Case<mlir_ts::NumberType>([&](auto _) { typeOfs["number"] = false; })
                        .Case<mlir_ts::StringType>([&](auto _) { typeOfs["string"] = true; })
                        .Case<mlir_ts::CharType>([&](auto _) { typeOfs["char"] = false; })
                        .Case<mlir::IntegerType>([&](auto intType_) {
                            if (intType_.isSignless()) typeOfs["i" + std::to_string(intType_.getWidth())] = false; else
                            if (intType_.isSigned()) typeOfs["s" + std::to_string(intType_.getWidth())] = false; else
                            if (intType_.isUnsigned()) typeOfs["u" + std::to_string(intType_.getWidth())] = false; })
                        .Case<mlir::FloatType>([&](auto floatType_) { typeOfs["f" + std::to_string(floatType_.getWidth())] = false; })
                        .Case<mlir::IndexType>([&](auto _) { typeOfs["index"] = false; })
                        .Case<mlir_ts::BigIntType>([&](auto _) { typeOfs["bigint"] = false; })
                        .Case<mlir_ts::FunctionType>([&](auto _) { typeOfs["function"] = true; })
                        .Case<mlir_ts::BoundFunctionType>([&](auto _) { typeOfs["function"] = true; })
                        .Case<mlir_ts::HybridFunctionType>([&](auto _) { typeOfs["function"] = true; })
                        .Case<mlir_ts::ExtensionFunctionType>([&](auto _) { typeOfs["function"] = true; })
                        .Case<mlir_ts::ClassType>([&](auto classType_) { typeOfs["class"] = true; classInstances.push_back(classType_); })
                        .Case<mlir_ts::InterfaceType>([&](auto _) { typeOfs["interface"] = true; })
                        .Case<mlir_ts::ArrayType>([&](auto _) { typeOfs["array"] = true; })
                        .Case<mlir_ts::ConstArrayType>([&](auto _) { typeOfs["array"] = true; })
                        .Case<mlir_ts::OpaqueType>([&](auto _) { typeOfs["object"] = true; })
                        .Case<mlir_ts::ObjectType>([&](auto _) { typeOfs["object"] = true; })
                        .Case<mlir_ts::NullType>([&](auto _) { typeOfs["null"] = true; })
                        .Case<mlir_ts::UndefinedType>([&](auto _) { typeOfs["undefined"] = false; })
                        .Default([&](auto type) {
                            LLVM_DEBUG(llvm::dbgs() << "\n\t TypeOf NOT IMPLEMENTED for Type: " << type << "\n";);
                            hasUnsupportedType = true;
                        });
                }

                if (hasUnsupportedType)
                {
                    // e.g. a tuple/object-literal-shaped member of the union - see the
                    // "must be improved"/"can't handle types such as 2 tuples in union"
                    // TODO on castFromUnion's declaration; typeof-based dispatch can't
                    // distinguish these today, that's a separate, larger redesign.
                    emitError(location) << "Cast from " << to_print(value.getType()) << " to " << to_print(type) << " is not supported";
                    return mlir::failure();
                }

                if (isNullDest)
                {
                    // to null
                    auto next = false;
                    for (auto& pair : typeOfs)
                    {
                        auto isNullable = pair.getValue();
                        if (next) ss << S(" else ");

                        ss << S("if (typeof t == '");
                        ss << stows(pair.getKey().str());
                        ss << S("') ");
                        if (pair.getKey() == "class")
                        {
                            ss << S("{ \n");

                            for (auto [index, _] : enumerate(classInstances))
                            {
                                ss << S("if (t instanceof TYPE_INST_ALIAS");
                                ss << index;
                                ss << S(") return t;\n");
                            }

                            ss << S(" }\n");
                        }
                        else
                        {
                            if (isNullable)
                                ss << S("return t;\n");
                            else
                                ss << S("return -1;\n");
                        }

                        next = true;
                    }                   
                }
                else
                {
                    // default
                    auto next = false;
                    for (auto& pair : typeOfs)
                    {
                        if (next) ss << S(" else ");

                        ss << S("if (typeof t == '");
                        ss << stows(pair.getKey().str());
                        ss << S("') ");
                        if (pair.getKey() == "class")
                        {
                            ss << S("{ \n");

                            for (auto [index, _] : enumerate(classInstances))
                            {
                                ss << S("if (t instanceof TYPE_INST_ALIAS");
                                ss << index;
                                ss << S(") return t;\n");
                            }

                            ss << S(" }\n");
                        }
                        else
                        {
                            ss << S("return t;\n");
                        }

                        next = true;
                    }
                }

                ss << "\nthrow \"Can't cast from union type\";\n";                    
                ss << S("}\n");

                auto src = ss.str();

                {
                    MLIRLocationGuard vgLoc(overwriteLoc); 
                    overwriteLoc = location;

                    if (mlir::failed(parsePartialStatements(src)))
                    {
                        assert(false);
                        return mlir::failure();
                    }
                }

                auto funcResult = resolveIdentifier(location, funcName, genContext);

                assert(funcResult);

                GenContext funcCallGenContext(genContext);
                funcCallGenContext.typeAliasMap.insert({".TYPE_ALIAS_T", value.getType()});
                funcCallGenContext.typeAliasMap.insert({".TYPE_ALIAS_U", type});

                for (auto [index, instanceOfType] : enumerate(classInstances))
                {
                    funcCallGenContext.typeAliasMap.insert({"TYPE_INST_ALIAS" + std::to_string(index), instanceOfType});
                }

                for (auto [index, tupleType] : enumerate(tupleTypes))
                {
                    funcCallGenContext.typeAliasMap.insert({"TYPE_TUPLE_ALIAS" + std::to_string(index), tupleType});
                }

                SmallVector<mlir::Value, 4> operands;
                operands.push_back(value);

                NodeFactory nf(NodeFactoryFlags::None);
                return mlirGenCallExpression(
                    location, 
                    funcResult, 
                    { 
                        nf.createTypeReferenceNode(nf.createIdentifier(S(".TYPE_ALIAS_T")).as<Node>()), 
                        nf.createTypeReferenceNode(nf.createIdentifier(S(".TYPE_ALIAS_U")).as<Node>()) 
                    }, 
                    operands, 
                    funcCallGenContext);
            }
        }

        return mlir::failure();
    }    

    ValueOrLogicalResult MLIRGenImpl::castTupleToInterface(mlir::Location location, mlir::Value in, mlir::Type tupleTypeIn,
                                     mlir_ts::InterfaceType interfaceType, const GenContext &genContext)
    {

        auto tupleType = mth.convertConstTupleTypeToTupleType(tupleTypeIn);
        auto interfaceInfo = getInterfaceInfoByFullName(interfaceType.getName().getValue());

        auto inEffective = in;

        auto srcTuple = mlir::cast<mlir_ts::TupleType>(tupleType);
        if (mlir::failed(mth.canCastTupleToInterface(location, srcTuple, interfaceInfo, true)))
        {
            SmallVector<mlir_ts::FieldInfo> fields;
            if (mlir::failed(getInterfaceCloneFields(srcTuple.getFields(), interfaceInfo, builder.getContext(), fields)))
            {
                return mlir::failure();
            }

            auto newInterfaceTupleType = getTupleType(fields);
            CAST(inEffective, location, newInterfaceTupleType, inEffective, genContext);
            tupleType = newInterfaceTupleType;

            emitWarning(location, "") << "Cloned object is used. Ensure all types are matching to interface: " << interfaceInfo->fullName;
        }

        // TODO: finish it, what to finish it? maybe optimization not to create extra object?
        // convert Tuple to Object
        auto objType = mlir_ts::ObjectType::get(tupleType);
        auto valueAddr = builder.create<mlir_ts::NewOp>(location, mlir_ts::ValueRefType::get(tupleType), builder.getBoolAttr(false));
        builder.create<mlir_ts::StoreOp>(location, inEffective, valueAddr);
        auto inCasted = builder.create<mlir_ts::CastOp>(location, objType, valueAddr);

        return castObjectToInterface(location, inCasted, objType, interfaceInfo, genContext);
    }

    ValueOrLogicalResult MLIRGenImpl::castObjectToInterface(mlir::Location location, mlir::Value in, mlir_ts::ObjectType objType,
                                    mlir_ts::InterfaceType interfaceType, const GenContext &genContext)
    {
        auto interfaceInfo = getInterfaceInfoByFullName(interfaceType.getName().getValue());
        return castObjectToInterface(location, in, objType, interfaceInfo, genContext);
    }

    ValueOrLogicalResult MLIRGenImpl::castObjectToInterface(mlir::Location location, mlir::Value in, mlir_ts::ObjectType objType,
                                    InterfaceInfo::TypePtr interfaceInfo, const GenContext &genContext)
    {
        auto inEffective = in;
        auto effectiveObjType = objType;

        // same field-type-coercion fallback as castTupleToInterface (e.g. a literal
        // integer field inferred as si32 vs an interface declaring `number`): an
        // object literal with methods is boxed as ObjectType before ever reaching
        // castTupleToInterface (see docs/object-literal-boxing-design.md), so that
        // cast's clone-and-coerce step must be duplicated here rather than relying
        // on falling through to it.
        if (auto storageTuple = dyn_cast<mlir_ts::TupleType>(objType.getStorageType()))
        {
            if (mlir::failed(mth.canCastTupleToInterface(location, storageTuple, interfaceInfo, true)))
            {
                SmallVector<mlir_ts::FieldInfo> fields;
                if (mlir::failed(getInterfaceCloneFields(storageTuple.getFields(), interfaceInfo, builder.getContext(), fields)))
                {
                    return mlir::failure();
                }

                auto newInterfaceTupleType = getTupleType(fields);

                CAST_A(unboxed, location, newInterfaceTupleType, in, genContext);

                auto valueAddr = builder.create<mlir_ts::NewOp>(location, mlir_ts::ValueRefType::get(newInterfaceTupleType), builder.getBoolAttr(false));
                builder.create<mlir_ts::StoreOp>(location, unboxed, valueAddr);
                effectiveObjType = mlir_ts::ObjectType::get(newInterfaceTupleType);
                inEffective = builder.create<mlir_ts::CastOp>(location, effectiveObjType, valueAddr);

                emitWarning(location, "") << "Cloned object is used. Ensure all types are matching to interface: " << interfaceInfo->fullName;
            }
        }

        auto result = mlirGenCreateInterfaceVTableForObject(location, inEffective, effectiveObjType, interfaceInfo, genContext);
        EXIT_IF_FAILED_OR_NO_VALUE(result)
        auto createdInterfaceVTableForObject = V(result);

        LLVM_DEBUG(llvm::dbgs() << "\n!!"
                                << "@ created interface:" << createdInterfaceVTableForObject << "\n";);

        return V(builder.create<mlir_ts::NewInterfaceOp>(location,
            mlir::TypeRange{interfaceInfo->interfaceType}, inEffective, createdInterfaceVTableForObject));
    }

    mlir_ts::CreateBoundFunctionOp MLIRGenImpl::createBoundMethodFromExtensionMethod(mlir::Location location, mlir_ts::CreateExtensionFunctionOp createExtentionFunction)
    {
        auto extFuncType = createExtentionFunction.getType();
        auto boundFuncVal = builder.create<mlir_ts::CreateBoundFunctionOp>(
            location, 
            getBoundFunctionType(
                extFuncType.getInputs(), 
                extFuncType.getResults(), 
                extFuncType.isVarArg()), 
            createExtentionFunction.getThisVal(), createExtentionFunction.getFunc());            

        return boundFuncVal;
    }

} // namespace mlirgen
} // namespace typescript
