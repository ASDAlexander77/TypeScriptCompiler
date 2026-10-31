// Cast-family methods of MLIRGenImpl (see MLIRGenImpl.h).

#include "MLIRGenImpl.h"
#include "TypeScript/MLIRLogic/MLIRRTTIHelperVC.h"

// the RTTI helper undefines it on the way out
#undef DEBUG_TYPE
#define DEBUG_TYPE "mlir"

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
    // fields. A function the interface can't call with the object as its `this`
    // (a closure, see MLIRTypeHelper::isObjectFieldFunctionWithoutThis) keeps its
    // own type: cast to the interface's it would lose its captures. Its slot calls a
    // thunk reading the field instead (mlirGenObjectVirtualTableDefinitionForInterface).
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
            auto keepsOwnType = interfaceField != interfaceFields.end()
                && isa<mlir_ts::FunctionType>(interfaceField->type)
                && MLIRTypeHelper::isObjectFieldFunctionWithoutThis(origField.type);
            fields.push_back(interfaceField != interfaceFields.end() && !keepsOwnType ? *interfaceField : origField);
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

    // the value of a field the source does not have, when the field's type admits undefined: `v?: T`
    // is optional<T>, and `v?: any` stays any (any | undefined is any) - both read as undefined
    mlir::Value MLIRGenImpl::absentFieldValue(mlir::Location location, mlir::Type fieldType, const GenContext &genContext)
    {
        if (isa<mlir_ts::OptionalType>(fieldType))
        {
            return builder.create<mlir_ts::OptionalUndefOp>(location, fieldType);
        }

        if (isa<mlir_ts::AnyType>(fieldType))
        {
            auto undefValue = builder.create<mlir_ts::UndefOp>(location, getUndefinedType());
            auto result = cast(location, fieldType, undefValue, genContext);
            return result.failed() ? mlir::Value() : V(result);
        }

        return mlir::Value();
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
                if (index >= srcTupleType.size())
                {
                    if (auto undefVal = absentFieldValue(location, fieldInfo.type, genContext))
                    {
                        values.push_back(undefVal);
                        continue;
                    }
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
                    if (auto undefVal = absentFieldValue(location, fieldInfo.type, genContext))
                    {
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

    // A tuple literal folded to a constant keeps each array in it as a static header over constant
    // data, so `pop`, `length =` and element writes changed a constant global and `push` reallocated
    // one (#479). It is rebuilt here as a tuple whose arrays are heap copies, made from the
    // literal's attributes the way castConstArrayToArray makes a nested array literal; a nested
    // tuple literal holding an array is rebuilt in turn. Returns no value for anything else.
    ValueOrLogicalResult MLIRGenImpl::copyArraysOfConstTuple(mlir::Location location, mlir::Value value,
        mlir_ts::ConstTupleType constTupleType, const GenContext &genContext)
    {
        auto constOp = value.getDefiningOp<mlir_ts::ConstantOp>();
        auto fieldAttrs = constOp ? dyn_cast<mlir::ArrayAttr>(constOp.getValue()) : mlir::ArrayAttr();
        if (!fieldAttrs || fieldAttrs.size() != constTupleType.size() || !MLIRTypeHelper::constTupleHoldsArray(constTupleType))
        {
            return mlir::Value();
        }

        SmallVector<mlir::Value> values;
        for (auto [index, fieldInfo] : enumerate(constTupleType.getFields()))
        {
            auto fieldAttr = dyn_cast<mlir::ArrayAttr>(fieldAttrs[index]);
            mlir::Value literal;
            if (auto arrayType = dyn_cast<mlir_ts::ArrayType>(fieldInfo.type); arrayType && fieldAttr)
            {
                literal = builder.create<mlir_ts::ConstantOp>(
                    location, getConstArrayType(arrayType.getElementType(), fieldAttr.size()), fieldAttr);
            }
            else if (MLIRTypeHelper::constTupleHoldsArray(fieldInfo.type) && fieldAttr)
            {
                literal = builder.create<mlir_ts::ConstantOp>(
                    location, mth.convertTupleTypeToConstTupleType(fieldInfo.type), fieldAttr);
            }

            if (!literal)
            {
                MLIRPropertyAccessCodeLogic cl(compileOptions, builder, location, value, builder.getI32IntegerAttr(index));
                auto fieldValue = cl.Tuple(constTupleType, true);
                VALIDATE(fieldValue, location)
                values.push_back(fieldValue);
                continue;
            }

            CAST_A(copied, location, mth.convertConstTupleTypeToTupleType(fieldInfo.type), literal, genContext);
            values.push_back(copied);
        }

        auto tupleType = mlir::cast<mlir_ts::TupleType>(mth.convertConstTupleTypeToTupleType(constTupleType));
        return V(builder.create<mlir_ts::CreateTupleOp>(location, tupleType, values));
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

    // `<C>i`, `i as C`, or narrowing by `i instanceof C`: the interface holds an object, and if that
    // object is a C, the result is that very object - asked through slot 0 of the interface's
    // vtable (INTERFACE_VTABLE_HEADER_SLOTS). If it is not (another class, or an object literal),
    // there is no C to hand back, and the cast throws, as `<C>anyValue` does. It used to build a
    // new C out of the interface's fields, which crashed the compiler for any class with a field
    // the interface does not name - `class B implements I` with a field `x` was enough.
    ValueOrLogicalResult MLIRGenImpl::castInterfaceToClass(mlir::Location location, mlir::Value value,
        mlir_ts::ClassType classType, const GenContext &genContext)
    {
        // Where every field of the class is a field of the interface, the old conversion - a new
        // C filled from the interface's fields - still answers for what is not a C, as before.
        auto fields = mlir::cast<mlir_ts::ClassStorageType>(classType.getStorageType()).getFields();
        auto interfaceInfo = getInterfaceInfoByFullName(mlir::cast<mlir_ts::InterfaceType>(value.getType()).getName().getValue());
        auto fieldsInInterface = interfaceInfo && llvm::all_of(fields, [&](auto &field) {
            auto name = dyn_cast_or_null<mlir::StringAttr>(field.id);
            return (name && name.getValue().starts_with(".")) || interfaceInfo->findField(field.id);
        });

        auto classRefVal = builder.create<mlir_ts::ClassRefOp>(
            location, classType, mlir::FlatSymbolRefAttr::get(builder.getContext(), classType.getName().getValue()));
        auto isInstance = mlirGenInstanceOfInterface(location, value, classRefVal, genContext);
        if (isInstance.failed_or_no_value())
        {
            return mlir::failure();
        }

        MLIRCodeLogicHelper mclh(builder, location, compileOptions);
        return mclh.conditionalValue(
            V(isInstance),
            [&]() {
                auto thisPtr = builder.create<mlir_ts::ExtractInterfaceThisOp>(location, getOpaqueType(), value);
                return ValueOrLogicalResult(builder.create<mlir_ts::CastOp>(location, classType, thisPtr));
            },
            [&](mlir::Type) {
                if (fieldsInInterface)
                {
                    return castFieldsToClass(location, value, fields, classType, genContext);
                }

                auto message = builder.create<mlir_ts::ConstantOp>(location, getStringType(),
                                                                   getStringAttr("Can't cast from interface"));
                builder.create<mlir_ts::ThrowOp>(location, message);

                MLIRRTTIHelperVC rtti(builder, theModule, compileOptions);
                setCatchCopyThunkBuilder(rtti);
                if (!rtti.setRTTIForType(location, message.getType(), [&](StringRef classFullName) {
                        return getClassInfoByFullName(classFullName);
                    }))
                {
                    emitError(location, "Not supported type in throw");
                    return ValueOrLogicalResult(mlir::failure());
                }

                // not reached; the branch still needs a value of the class type
                auto nullValue = builder.create<mlir_ts::NullOp>(location, getNullType());
                return ValueOrLogicalResult(builder.create<mlir_ts::CastOp>(location, classType, nullValue));
            });
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
        SmallVector<std::pair<mlir::Attribute, mlir::Value>> fieldValues;
        for (auto [index, fieldInfo] : enumerate(fields))
        {
            LLVM_DEBUG(llvm::dbgs() << "\n!! processing #" << index << " field [" << fieldInfo.id << "]\n";);

            MLIRPropertyAccessCodeLogic cl(compileOptions, builder, location, value, builder.getI32IntegerAttr(index));
            auto fieldValue = cl.Tuple(tupleType, true);
            VALIDATE(value, location)
            fieldValues.push_back({fieldInfo.id, fieldValue});
        }

        return fieldsToString(location, fieldValues, genContext);
    }

    // `{ a: 1, b: 'text' }`: each field's name and its value as a string, a string's in quotes
    ValueOrLogicalResult MLIRGenImpl::fieldsToString(mlir::Location location,
        ::llvm::ArrayRef<std::pair<mlir::Attribute, mlir::Value>> fieldValues, const GenContext &genContext)
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

        for (auto [index, field] : enumerate(fieldValues))
        {
            auto fieldId = field.first;
            auto fieldValue = field.second;
            if (index > 0)
            {
                // text
                strs.push_back(spanValue);
            }

            strs.push_back(spaceValue);
            if (fieldId)
            {
                auto fieldNameValue = builder.create<mlir_ts::ConstantOp>(location, stringType, fieldId);
                strs.push_back(fieldNameValue);
                strs.push_back(fieldSepValue);
            }

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

    // An interface printed: its `toString` when it has one, as a class's is; else its fields, read
    // through it, as an object literal's are (`{ n: 3 }`), those of the interfaces it extends first;
    // and null as "null". An interface whose fields hold its own type again is printed `[Object]`
    // there: the text is made inline, and the type nests without end.
    ValueOrLogicalResult MLIRGenImpl::castInterfaceToString(mlir::Location location, mlir::Value value,
        mlir_ts::InterfaceType interfaceType, const GenContext &genContext)
    {
        auto stringType = getStringType();
        auto interfaceInfo = getInterfaceInfoByFullName(interfaceType.getName().getValue());
        if (!interfaceInfo)
        {
            return mlir::failure();
        }

        if (llvm::is_contained(interfacesBeingPrinted, interfaceType))
        {
            return V(builder.create<mlir_ts::ConstantOp>(location, stringType, getStringAttr("[Object]")));
        }

        interfacesBeingPrinted.push_back(interfaceType);
        auto popPrinted = llvm::make_scope_exit([&]() { interfacesBeingPrinted.pop_back(); });

        auto nullValue = builder.create<mlir_ts::NullOp>(location, getNullType());
        auto isNull = builder.create<mlir_ts::LogicalBinaryOp>(
            location, getBooleanType(), builder.getI32IntegerAttr((int)SyntaxKind::EqualsEqualsEqualsToken), value, nullValue);
        auto ifOp = builder.create<mlir_ts::IfOp>(location, mlir::TypeRange{stringType}, isNull, true);

        builder.setInsertionPointToStart(&ifOp.getThenRegion().front());
        builder.create<mlir_ts::ResultOp>(location, mlir::ValueRange{
            builder.create<mlir_ts::ConstantOp>(location, stringType, builder.getStringAttr(NULL_NAME))});

        builder.setInsertionPointToStart(&ifOp.getElseRegion().front());
        mlir::Value text;
        if (interfaceInfo->findMethod(TO_STRING))
        {
            auto result = mlirGenCallThisMethod(location, value, TO_STRING, undefined, undefined, genContext);
            EXIT_IF_FAILED_OR_NO_VALUE(result)
            text = V(result);
            if (text.getType() != stringType)
            {
                CAST(text, location, stringType, text, genContext);
            }
        }
        else
        {
            llvm::SmallVector<InterfaceFieldInfo> fields;
            std::function<void(InterfaceInfo::TypePtr)> collect = [&](InterfaceInfo::TypePtr info) {
                for (auto &extent : info->extends)
                {
                    collect(std::get<1>(extent));
                }

                fields.append(info->fields.begin(), info->fields.end());
            };
            collect(interfaceInfo);

            SmallVector<std::pair<mlir::Attribute, mlir::Value>> fieldValues;
            for (auto &field : fields)
            {
                auto name = mlir::dyn_cast_or_null<mlir::StringAttr>(field.id);
                if (!name)
                {
                    continue;
                }

                auto fieldValue = mlirGenPropertyAccessExpression(location, value, name.getValue(), genContext);
                EXIT_IF_FAILED_OR_NO_VALUE(fieldValue)
                fieldValues.push_back({field.id, V(fieldValue)});
            }

            auto result = fieldsToString(location, fieldValues, genContext);
            EXIT_IF_FAILED_OR_NO_VALUE(result)
            text = V(result);
        }

        builder.create<mlir_ts::ResultOp>(location, mlir::ValueRange{text});
        builder.setInsertionPointAfter(ifOp);
        return V(ifOp.getResults().front());
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

        // a class given where its value is expected (`make(B)` for `c: typeof A`), or the value of a class
        // where a base class's is: B's value is A's when B extends A
        if (auto classValueType = dyn_cast<mlir_ts::ClassValueType>(type))
        {
            return castToClassValue(location, classValueType, value, genContext);
        }

        // A @value class is nominal. Its value seen as `any`, `unknown`, `Object` or an ordinary
        // interface would be a boxed copy that has lost its class (no vtable to call its methods by);
        // seen as another @value type, a value of a class it is not. With null or undefined it is
        // itself (`V | null`), and so in a union of @value types.
        if (auto valueClassInfo = getValueClassInfo(valueType))
        {
            auto isValueMember = [](mlir::Type memberType) {
                return isa<mlir_ts::ValueType, mlir_ts::NullType, mlir_ts::UndefinedType>(memberType);
            };

            auto unionType = dyn_cast<mlir_ts::UnionType>(type);
            if (isa<mlir_ts::AnyType, mlir_ts::UnknownType, mlir_ts::ObjectType, mlir_ts::InterfaceType, mlir_ts::ValueType>(type)
                || (unionType && !llvm::all_of(unionType.getTypes(), isValueMember)))
            {
                emitError(location) << "the @" VALUE_DECORATOR " class '" << valueClassInfo->name << "' cannot be converted to '"
                                    << to_print(type) << "'";
                return mlir::failure();
            }
        }

        // ...and only its own values are values of it: a literal, a tuple, a @value interface value or
        // an object of the same fields is not one (its constructor never ran). With null or undefined
        // it is `V | null` (the cases below).
        if (auto destClassInfo = getValueClassInfo(type);
            destClassInfo && !isa<mlir_ts::NullType, mlir_ts::UndefinedType, mlir_ts::OptionalType, mlir_ts::UnionType>(valueType))
        {
            // Discovery sees results not resolved yet - a method calling one declared after it gives
            // `void` - and keeps going, as it does for any other cast it cannot make yet. Its IR is
            // thrown away; the final pass checks the real conversion.
            if (genContext.dummyRun || genContext.allowPartialResolve)
            {
                return V(builder.create<mlir_ts::CastOp>(location, type, value));
            }

            emitError(location) << "type '" << to_print(valueType) << "' cannot be converted to the @" VALUE_DECORATOR " class '"
                                << destClassInfo->name << "'";
            return mlir::failure();
        }

        // A @value type is laid out as its storage tuple. Converting to one is converting to that
        // tuple - a field copy in the target's order - and taking the tuple as the value; converting
        // from one is converting its tuple, which boxes a copy for `any`, `Object` or an ordinary
        // interface. So a @value value never reaches the tuple paths below as itself. With null,
        // undefined, an optional or a union on either side it is the cases below (`T | null`).
        if (auto destValueType = dyn_cast<mlir_ts::ValueType>(type))
        {
            // ...and it has no null pointer: a missing value is written `T | null` (an optional
            // value), even where null checks are not strict
            if (isa<mlir_ts::NullType, mlir_ts::UndefinedType>(valueType))
            {
                emitError(location) << "a @" VALUE_DECORATOR " type has no null; write '" << to_print(type) << " | null'";
                return mlir::failure();
            }

            // one that may be missing is read as the value it holds, and throws when it holds none
            // (under `@strict-null false`, where TypeScript would throw only at a later read)
            if (MLIRTypeHelper::getMissingValueElement(valueType))
            {
                auto present = mlirGenPresentValue(location, value, genContext);
                EXIT_IF_FAILED_OR_NO_VALUE(present)
                return cast(location, type, V(present), genContext, disableStrictNullCheck);
            }

            if (!isa<mlir_ts::OptionalType, mlir_ts::UnionType>(valueType))
            {
                CAST_A(storage, location, destValueType.getStorageType(), valueStorage(location, value), genContext);
                return V(builder.create<mlir_ts::CastOp>(location, destValueType, storage));
            }
        }
        else if (isa<mlir_ts::ValueType>(valueType) && !isa<mlir_ts::OptionalType, mlir_ts::UnionType>(type))
        {
            return cast(location, type, valueStorage(location, value), genContext, disableStrictNullCheck);
        }
        // one that may be missing is true when it holds a value, as an object is
        else if (isa<mlir_ts::BooleanType>(type) && MLIRTypeHelper::getMissingValueElement(valueType))
        {
            auto nullValue = builder.create<mlir_ts::NullOp>(location, getNullType());
            return mlirGenBinaryValues(location, SyntaxKind::ExclamationEqualsToken, value, nullValue, genContext);
        }

        // `unknown` holds what `any` does, the same box: a value is boxed into it as into `any`, and
        // read out of it as out of `any`. A value put into `unknown` was the address of a copy of it,
        // which the way back read as the value itself (`(<C>u).n` read the stack slot holding the
        // object pointer as the object).
        if (isa<mlir_ts::UnknownType>(type) && !isa<mlir_ts::AnyType>(valueType))
        {
            CAST_A(boxed, location, getAnyType(), value, genContext);
            return V(builder.create<mlir_ts::CastOp>(location, type, boxed));
        }

        if (isa<mlir_ts::UnknownType>(valueType) && !isa<mlir_ts::AnyType>(type))
        {
            auto boxed = builder.create<mlir_ts::CastOp>(location, getAnyType(), value);
            return cast(location, type, boxed, genContext, disableStrictNullCheck);
        }

        // see isCppVTableClass: `any` is unboxed and an interface made by asking the object
        if ((isa<mlir_ts::AnyType>(valueType) && isCppVTableClass(type))
            || ((isa<mlir_ts::AnyType>(type) || isa<mlir_ts::InterfaceType>(type)) && isCppVTableClass(valueType)))
        {
            emitError(location) << "can't cast '" << to_print(valueType) << "' to '" << to_print(type)
                                << "': a @" CPP_VTABLE " class's objects are C++ objects, with no type information";
            return mlir::failure();
        }

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

        if (auto result = castConstArrayToArray(location, type, value, valueType, genContext))
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

        if (auto result = castToNullableUnionType(location, type, value, valueType, genContext))
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

        // Boxing into `any` makes the box an owner of what it now holds, and it has to take a
        // reference to say so: an `any` releases its payload through the type tag beside it when
        // the box dies (`buildBody`, AnyType). Nothing took one. The payload was copied in, and
        // where it arrived carrying a reference of its own - a closure, whose `__owned_result`
        // is the box of captured variables it was built over - §9.30 gave that reference back at
        // the end of the block and left the `any` pointing at freed memory. Reachable wherever
        // the boxed value outlives the block that boxed it, `fns.push(qux2)` into an `any[]`
        // being the shape that found it (§9.36).
        if (isa<mlir_ts::AnyType>(type))
        {
            mlirGenRetainCaptured(location, mlir::ValueRange{value});
        }

        auto castResult = builder.create<mlir_ts::CastOp>(location, type, value);

        // Printing a number into a string allocates one, and nothing was giving it back: in
        // `"s" + k` the conversion's result is read by the concatenation and then forgotten -
        // not received by anything, so not a receiver's to release, and not marked, so not
        // §9.30's either. It is the plainest allocation in the language and it leaked every
        // time (§9.37).
        if (isa<mlir_ts::StringType>(type) && mlir_ts::castToStringAllocates(valueType))
        {
            markFreshBlockOwned(location, castResult);
        }

        // Boxing into `any` allocates the box, and the cast handed it back with no reference at
        // all: a box a `let` holds was retained by the `let` and freed, but one held by a folded
        // `const`, or passed straight to a call, was never released (1M `const a: any = new C()`:
        // rc 66 MB, gc 6 MB). The box takes the payload's reference above; this is the box's own.
        // An `any` cast to `any` is the same box, and allocates nothing.
        if (isa<mlir_ts::AnyType>(type) && !isa<mlir_ts::AnyType>(valueType))
        {
            markFreshBlockOwned(location, castResult);
        }

        return V(castResult);
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

            if (auto interfaceType = dyn_cast<mlir_ts::InterfaceType>(valueType))
            {
                return castInterfaceToString(location, value, interfaceType, genContext);
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
            // null (or undefined) is an interface over a null `this`, seen through the interface's null
            // vtable, whose methods throw (mlirGenNullInterfaceVTable). It stays the cast the ownership
            // passes know holds nothing; the cast names the vtable, and the lowering puts it in.
            if (isa<mlir_ts::NullType, mlir_ts::UndefinedType>(valueType))
            {
                auto interfaceInfo = getInterfaceInfoByFullName(interfaceType.getName().getValue());
                if (!interfaceInfo)
                {
                    return std::nullopt;
                }

                return mlirGenNullInterfaceVTableCast(location, interfaceType, interfaceInfo, genContext, value);
            }

            // A class object is a pointer that may be null (`C | null` is stored as `C`), and null
            // stays null: an interface over a null `this`. The vtable may be read through the object,
            // so it is chosen by a test for null: a null object is seen through the interface's null
            // vtable, whose methods throw (mlirGenNullInterfaceVTable). The interface is built over
            // the object outside the test, so that the ownership passes see it is the object (a view),
            // not a value merged from branches.
            if (auto classType = dyn_cast<mlir_ts::ClassType>(valueType))
            {
                auto interfaceInfo = getInterfaceInfoByFullName(interfaceType.getName().getValue());
                assert(interfaceInfo);

                auto nullValue = builder.create<mlir_ts::NullOp>(location, getNullType());
                auto isNotNull = builder.create<mlir_ts::LogicalBinaryOp>(
                    location, getBooleanType(), builder.getI32IntegerAttr((int)SyntaxKind::ExclamationEqualsEqualsToken), value, nullValue);

                auto vtableType = mth.getInterfaceVTableType(interfaceType);
                MLIRCodeLogicHelper mclh(builder, location, compileOptions);
                auto interfaceVTable = mclh.conditionalValue(isNotNull,
                    [&]() -> ValueOrLogicalResult {
                        auto classVTable = mlirGenClassInterfaceVTable(location, value, classType, interfaceType, genContext);
                        EXIT_IF_FAILED_OR_NO_VALUE(classVTable)
                        if (V(classVTable).getType() == vtableType)
                        {
                            return classVTable;
                        }

                        return V(builder.create<mlir_ts::CastOp>(location, vtableType, V(classVTable)));
                    },
                    [&](mlir::Type) -> ValueOrLogicalResult {
                        // null, as the ownership passes know it, naming the null vtable the lowering puts in
                        return mlirGenNullInterfaceVTableCast(location, vtableType, interfaceInfo, genContext);
                    });
                EXIT_IF_FAILED_OR_NO_VALUE(interfaceVTable)

                auto newInterface = builder.create<mlir_ts::NewInterfaceOp>(
                    location, mlir::TypeRange{interfaceType}, value, V(interfaceVTable));
                return V(newInterface);
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

            if (auto srcInterfaceType = dyn_cast<mlir_ts::InterfaceType>(valueType))
            {
                return castInterfaceToInterface(location, value, srcInterfaceType, interfaceType, genContext);
            }

            // An `any` holds no interface's name, only its kind: what it holds was read as the
            // interface's (this, vtable) pair, and a class object, or an interface value made for
            // another interface (`<Holder><any>runner`), was called through garbage.
            if (isa<mlir_ts::AnyType>(valueType) && !genContext.allowPartialResolve)
            {
                if (auto interfaceInfo = getInterfaceInfoByFullName(interfaceType.getName().getValue()))
                {
                    return castAnyToInterface(location, interfaceType, interfaceInfo, value, genContext);
                }
            }

            // `J | null` or `C | null` (a `J | undefined` given to `I | undefined` gets here too) is stored
            // as its one member, null a null pointer in it: this is the cast of that member. A plain cast
            // kept J's vtable for I, and lowering could not make a class pointer an interface.
            if (auto srcUnionType = dyn_cast<mlir_ts::UnionType>(valueType))
            {
                mlir::Type srcBaseType;
                if (!mth.isUnionTypeNeedsTag(location, srcUnionType, srcBaseType)
                    && isa_and_nonnull<mlir_ts::ClassType, mlir_ts::InterfaceType>(srcBaseType))
                {
                    auto member = builder.create<mlir_ts::GetValueFromUnionOp>(location, srcBaseType, value);
                    return cast(location, interfaceType, member, genContext);
                }
            }

            // A number, string, boolean or array has no fields an interface's members could be read
            // from: passed for an `{ kind: number }` it compiled, without a word, to a cast that
            // lowering could not make, and the callee read garbage. An interface without members
            // takes any value, as in TypeScript.
            auto baseValueType = mth.stripLiteralType(valueType);
            if (baseValueType.isIntOrIndexOrFloat()
                || isa<mlir_ts::NumberType, mlir_ts::StringType, mlir_ts::BooleanType, mlir_ts::CharType,
                       mlir_ts::BigIntType, mlir_ts::ArrayType, mlir_ts::ConstArrayType>(baseValueType))
            {
                auto interfaceInfo = getInterfaceInfoByFullName(interfaceType.getName().getValue());
                if (interfaceInfo && interfaceInfo->getVTableSize() > 0)
                {
                    emitError(location) << "type '" << to_print(baseValueType) << "' can't be cast to interface '"
                                        << to_print(interfaceType) << "': it has none of the interface's members";
                    return mlir::failure();
                }
            }
        }

        return std::nullopt;
    }

    ValueOrLogicalResult MLIRGenImpl::mlirGenClassInterfaceVTable(mlir::Location location, mlir::Value value, mlir_ts::ClassType classType, mlir_ts::InterfaceType interfaceType, const GenContext &genContext)
    {
        auto classInfo = getClassInfoByFullName(classType.getName().getValue());
        assert(classInfo);

        auto implementIndex = classInfo->getImplementIndex(interfaceType.getName().getValue());
        if (implementIndex >= 0)
        {
            auto result = mlirGenPropertyAccessExpression(location, value, VTABLE_NAME, genContext);
            auto vtableAccess = V(result);

            auto interfaceVirtTableIndex = classInfo->implements[implementIndex].virtualIndex;

            assert(genContext.allowPartialResolve || interfaceVirtTableIndex >= 0);

            return V(builder.create<mlir_ts::VTableOffsetRefOp>(
                location, mth.getInterfaceVTableType(interfaceType), vtableAccess, interfaceVirtTableIndex));
        }

        // create interface vtable from current class
        auto interfaceInfo = getInterfaceInfoByFullName(interfaceType.getName().getValue());
        assert(interfaceInfo);

        // known from the discovery pass on, wherever in the module the cast is (castAnyToInterface)
        classInterfaceVTablesCast.insert(interfaceVTableNameForClass(classInfo, interfaceInfo));

        if (auto createdInterfaceVTableForClass =
                mlirGenCreateInterfaceVTableForClass(location, classInfo, interfaceInfo, genContext))
        {
            LLVM_DEBUG(llvm::dbgs() << "\n!!"
                                    << "@ created interface:" << V(createdInterfaceVTableForClass) << "\n";);
            return createdInterfaceVTableForClass;
        }

        emitError(location) << "type: " << classType.getName() << " missing interface: " << interfaceType.getName();
        return mlir::failure();
    }

    // An interface value keeps the vtable it was made with, laid out for its own interface. Another
    // interface it is cast to reads that vtable by its own slot numbers, which match only when it is the
    // value's interface or one it extends (whose members come first): `interface HasDist { dist }` read
    // `thing`, the first slot of a `Hit { thing; ray; dist }`, and a method slot called whichever method
    // came first. For any other interface the cast carries where each of the target's slots is in the
    // source's vtable, and lowering builds the target's vtable from the value's own one. A slot is copied
    // as it is - a field's offset, a method, a cell, -1 for an absent member - so a member is taken only
    // where both name it alike: a field for a field of the same type, a method for a method. A member the
    // source lacks is absent, as an optional one is: such a cast (`tube = ball`, a Ball having no
    // `length`) was always let through.
    std::optional<ValueOrLogicalResult> MLIRGenImpl::castInterfaceToInterface(mlir::Location location, mlir::Value value,
                                                                             mlir_ts::InterfaceType srcInterfaceType,
                                                                             mlir_ts::InterfaceType interfaceType,
                                                                             const GenContext &genContext)
    {
        if (srcInterfaceType == interfaceType || genContext.allowPartialResolve)
        {
            return std::nullopt;
        }

        auto srcInterfaceInfo = getInterfaceInfoByFullName(srcInterfaceType.getName().getValue());
        auto interfaceInfo = getInterfaceInfoByFullName(interfaceType.getName().getValue());
        if (!srcInterfaceInfo || !interfaceInfo)
        {
            return std::nullopt;
        }

        SmallVector<int32_t> vtableMap(interfaceInfo->getVTableSize() + INTERFACE_VTABLE_HEADER_SLOTS, -1);
        // the header is the value's, whatever interface it is seen through
        vtableMap[INTERFACE_VTABLE_TYPE_SLOT] = INTERFACE_VTABLE_TYPE_SLOT;
        vtableMap[INTERFACE_VTABLE_INSTANCEOF_SLOT] = INTERFACE_VTABLE_INSTANCEOF_SLOT;

        // the first member the cast can't map, reported only when the vtable can't serve as it is
        mlir::Attribute problemMember;
        StringRef problem;
        auto cannotCast = [&](mlir::Attribute memberId, StringRef reason) {
            if (!problemMember)
            {
                problemMember = memberId;
                problem = reason;
            }
        };

        // the slots of the target, in its vtable's order: each `extends` block, then its own methods and
        // fields (InterfaceInfo::getAllMethodsWithVTableOffset)
        std::function<void(InterfaceInfo::TypePtr, int)> mapSlots = [&](InterfaceInfo::TypePtr info, int baseOffset) {
            auto offset = baseOffset;
            for (auto &extent : info->extends)
            {
                mapSlots(std::get<1>(extent), offset);
                offset += std::get<1>(extent)->getVTableSize();
            }

            for (auto &method : info->methods)
            {
                auto slot = method.virtualIndex + baseOffset + INTERFACE_VTABLE_HEADER_SLOTS;
                auto srcOffset = 0;
                if (auto *srcMethod = srcInterfaceInfo->findMethod(method.name, srcOffset))
                {
                    if (srcMethod->funcType != method.funcType)
                    {
                        cannotCast(builder.getStringAttr(method.name), "has another type");
                    }

                    vtableMap[slot] = srcMethod->virtualIndex + srcOffset + INTERFACE_VTABLE_HEADER_SLOTS;
                }
                else if (srcInterfaceInfo->findField(builder.getStringAttr(method.name)))
                {
                    cannotCast(builder.getStringAttr(method.name), "is a method of one and a field of the other");
                }
            }

            for (auto &field : info->fields)
            {
                auto slot = field.virtualIndex + baseOffset + INTERFACE_VTABLE_HEADER_SLOTS;
                auto srcOffset = 0;
                if (auto *srcField = srcInterfaceInfo->findField(field.id, srcOffset))
                {
                    if (mth.stripLiteralType(srcField->type) != mth.stripLiteralType(field.type))
                    {
                        cannotCast(field.id, "has another type");
                    }

                    if (srcField->isConditional && !field.isConditional)
                    {
                        cannotCast(field.id, "is optional");
                    }

                    vtableMap[slot] = srcField->virtualIndex + srcOffset + INTERFACE_VTABLE_HEADER_SLOTS;
                }
                else if (auto fieldName = dyn_cast<mlir::StringAttr>(field.id);
                         fieldName && srcInterfaceInfo->findMethod(fieldName.getValue()))
                {
                    cannotCast(field.id, "is a method of one and a field of the other");
                }
            }
        };

        mapSlots(interfaceInfo, 0);

        // an interface the source extends, or one laid out the same: the vtable serves as it is
        if (llvm::all_of(llvm::enumerate(vtableMap), [](auto slot) { return slot.value() == (int32_t)slot.index(); }))
        {
            return std::nullopt;
        }

        if (problemMember)
        {
            auto diagnostic = emitError(location);
            diagnostic << "interface '" << to_print(srcInterfaceType) << "' can't be cast to interface '"
                       << to_print(interfaceType) << "': member ";
            if (auto memberName = dyn_cast<mlir::StringAttr>(problemMember))
            {
                diagnostic << "'" << memberName.getValue() << "' ";
            }
            else
            {
                diagnostic << problemMember << " ";
            }

            diagnostic << problem;
            return mlir::failure();
        }

        auto castOp = builder.create<mlir_ts::CastOp>(location, interfaceType, value);
        castOp->setAttr(INTERFACE_VTABLE_MAP_ATTR_NAME, builder.getDenseI32ArrayAttr(vtableMap));
        return V(castOp);
    }

    // A constant array keeps the element type its literal was built with: `const c = [1, 2]` is a
    // const_array<si32>, and so is `[1, 2]` meeting a union such as `number[] | string`, where no single
    // array type guides the literal. Lowering turns a const array into an array by copying its data as it
    // is, so for another element type (`number[]`) the elements were read with the wrong layout - garbage,
    // with only a warning. Such an array is built here from its elements, each cast to the element type.
    std::optional<ValueOrLogicalResult> MLIRGenImpl::castConstArrayToArray(mlir::Location location, mlir::Type type, mlir::Value value, mlir::Type valueType, const GenContext &genContext)
    {
        auto arrayType = dyn_cast<mlir_ts::ArrayType>(type);
        if (!arrayType)
        {
            return std::nullopt;
        }

        auto constArrayType = dyn_cast<mlir_ts::ConstArrayType>(valueType);
        if (!constArrayType)
        {
            // a `const c = [1, 2]` read back is already the array its storage widened it to (see
            // processConstRef / adjustLocalVariableType / adjustGlobalVariableType in MLIRGenImpl.h),
            // and may have been changed since: its elements are converted as they are now
            if (auto srcArrayType = dyn_cast<mlir_ts::ArrayType>(valueType))
            {
                return castArrayElementwise(location, arrayType, value, srcArrayType, genContext);
            }

            return std::nullopt;
        }

        // the same element type needs no conversion - unless the elements are arrays: the cast below
        // copies only the outer data to the heap, and the inner arrays kept pointing at the literal's
        // constant data, so `nested[1].push(4)` reallocated a global and `nested[1][0] = 9` wrote one
        if (constArrayType.getElementType() == arrayType.getElementType()
            && !isa<mlir_ts::ArrayType>(arrayType.getElementType())
            && !MLIRTypeHelper::constTupleHoldsArray(arrayType.getElementType()))
        {
            return std::nullopt;
        }

        auto constOp = value.getDefiningOp<mlir_ts::ConstantOp>();
        auto elementAttrs = constOp ? dyn_cast<mlir::ArrayAttr>(constOp.getValue()) : mlir::ArrayAttr();
        if (!elementAttrs)
        {
            // not a literal: left to the cast below
            return std::nullopt;
        }

        // `[[1, 2]]` holds arrays, kept as nested attributes. Their own elements need the same
        // conversion, so each is made a constant array again and cast here in turn.
        auto sourceElementType = constArrayType.getElementType();
        mlir::Type nestedElementType;
        if (auto nestedArrayType = dyn_cast<mlir_ts::ArrayType>(sourceElementType))
        {
            nestedElementType = nestedArrayType.getElementType();
        }
        else if (auto nestedConstArrayType = dyn_cast<mlir_ts::ConstArrayType>(sourceElementType))
        {
            nestedElementType = nestedConstArrayType.getElementType();
        }

        SmallVector<mlir::Value> elements;
        for (auto elementAttr : elementAttrs)
        {
            mlir::Value element;
            if (auto nestedAttrs = dyn_cast<mlir::ArrayAttr>(elementAttr);
                nestedAttrs && MLIRTypeHelper::constTupleHoldsArray(sourceElementType))
            {
                // a tuple holding an array (`[[1, [2]]]`): rebuilt with copies by its cast (#479)
                element = builder.create<mlir_ts::ConstantOp>(
                    location, mth.convertTupleTypeToConstTupleType(sourceElementType), nestedAttrs);
            }
            else if (auto nestedAttrs = dyn_cast<mlir::ArrayAttr>(elementAttr))
            {
                if (!nestedElementType)
                {
                    return std::nullopt;
                }

                element = builder.create<mlir_ts::ConstantOp>(
                    location, getConstArrayType(nestedElementType, nestedAttrs.size()), nestedAttrs);
            }
            else
            {
                element = builder.create<mlir_ts::ConstantOp>(location, sourceElementType, elementAttr);
            }

            CAST_A(castedElement, location, arrayType.getElementType(), element, genContext);
            elements.push_back(castedElement);
        }

        // the data block about to be filled releases every element when it dies
        mlirGenRetainCaptured(location, elements);

        return V(builder.create<mlir_ts::CreateArrayOp>(location, arrayType, elements));
    }

    // An array of numbers held as `s32` (or another integer or float type) keeps that layout, so it is
    // not shared as a `number[]`: a new array is built, each element cast to the element type as it is
    // assigned - nested arrays in turn. The new array is a copy, so later changes to one of them are not
    // seen through the other. An array literal met at a union or a `const` binding gets here with s32
    // elements (see castConstArrayToArray), and so does a variable holding one.
    std::optional<ValueOrLogicalResult> MLIRGenImpl::castArrayElementwise(mlir::Location location, mlir_ts::ArrayType arrayType, mlir::Value value, mlir_ts::ArrayType srcArrayType, const GenContext &genContext)
    {
        auto isNumeric = [](mlir::Type type) {
            return isa<mlir::IntegerType>(type) || isa<mlir::FloatType>(type) || isa<mlir_ts::NumberType>(type);
        };

        std::function<bool(mlir::Type)> isNumericSource = [&](mlir::Type type) {
            if (auto nestedArrayType = dyn_cast<mlir_ts::ArrayType>(type))
            {
                return isNumericSource(nestedArrayType.getElementType());
            }

            return isa<mlir::IntegerType>(type) || isa<mlir::FloatType>(type);
        };

        // what a number can be assigned to: another number type, `any`, or a union holding a number
        std::function<bool(mlir::Type)> isNumericTarget = [&](mlir::Type type) {
            if (auto nestedArrayType = dyn_cast<mlir_ts::ArrayType>(type))
            {
                return isNumericTarget(nestedArrayType.getElementType());
            }

            if (auto unionType = dyn_cast<mlir_ts::UnionType>(type))
            {
                return llvm::any_of(unionType.getTypes(), isNumeric);
            }

            return isNumeric(type) || isa<mlir_ts::AnyType>(type) || isa<mlir_ts::UnknownType>(type);
        };

        auto srcElementType = srcArrayType.getElementType();
        auto elementType = arrayType.getElementType();
        if (srcElementType == elementType || !isNumericSource(srcElementType) || !isNumericTarget(elementType))
        {
            return std::nullopt;
        }

        SymbolTableScopeT varScope(symbolTable);

        auto length = builder.create<mlir_ts::LengthOfOp>(location, builder.getIndexType(), value);
        auto newArray = builder.create<mlir_ts::NewArrayOp>(location, arrayType, length);

        auto srcArrayVarDecl = std::make_shared<VariableDeclarationDOM>(".src_array", srcArrayType, location);
        DECLARE(srcArrayVarDecl, value);

        auto dstArrayVarDecl = std::make_shared<VariableDeclarationDOM>(".dst_array", arrayType, location);
        DECLARE(dstArrayVarDecl, newArray);

        NodeFactory nf(NodeFactoryFlags::None);

        auto _src_array_ident = nf.createIdentifier(S(".src_array"));
        auto _dst_array_ident = nf.createIdentifier(S(".dst_array"));
        auto _i = nf.createIdentifier(S(".i"));

        // for (let .i = 0; .i < .src_array.length; ++.i) .dst_array[.i] = .src_array[.i];
        NodeArray<VariableDeclaration> declarations;
        declarations.push_back(nf.createVariableDeclaration(_i, undefined, undefined, nf.createNumericLiteral(S("0"))));
        auto initVars = nf.createVariableDeclarationList(declarations, NodeFlags::Let);

        auto cond = nf.createBinaryExpression(_i, nf.createToken(SyntaxKind::LessThanToken),
                                              nf.createPropertyAccessExpression(_src_array_ident, nf.createIdentifier(S(LENGTH_FIELD_NAME))));

        auto incr = nf.createPrefixUnaryExpression(nf.createToken(SyntaxKind::PlusPlusToken), _i);

        auto assign = nf.createExpressionStatement(
            nf.createBinaryExpression(nf.createElementAccessExpression(_dst_array_ident, _i), nf.createToken(SyntaxKind::EqualsToken),
                                      nf.createElementAccessExpression(_src_array_ident, _i)));

        if (mlir::failed(mlirGen(nf.createForStatement(initVars, cond, incr, assign), genContext)))
        {
            return mlir::failure();
        }

        return V(newArray);
    }

    std::optional<ValueOrLogicalResult> MLIRGenImpl::castTupleLikeVariants(mlir::Location location, mlir::Type type, mlir::Value value, mlir::Type valueType, const GenContext &genContext)
    {
        // const tuple to tuple
        if (auto srcConstTupleType = dyn_cast<mlir_ts::ConstTupleType>(valueType))
        {
            if (isa<mlir_ts::TupleType, mlir_ts::ConstTupleType, mlir_ts::ClassType>(type))
            {
                // the copy is a tuple of the type the casts below read the constant as
                auto copied = copyArraysOfConstTuple(location, value, srcConstTupleType, genContext);
                EXIT_IF_FAILED(copied)
                if (auto copiedValue = V(copied))
                {
                    if (copiedValue.getType() == type)
                    {
                        return copied;
                    }

                    value = copiedValue;
                }
            }

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
                // a record with a run-time field (`const p: P = { x: 1, items: [i, 2] }`) makes the
                // instance from its fields, as a constant record does (#487)
                fields = mlir::cast<mlir_ts::ClassStorageType>(classType.getStorageType()).getFields();
                return castTupleToClass(location, value, srcTupleType, fields, classType, genContext);
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
                return castInterfaceToClass(location, value, classType, genContext);
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

            // an object literal is boxed for its methods and accessors, which a class instance cannot
            // have of its own: it takes them from its class's vtable. The cast was a reinterpretation
            // of the pointer, so the class read its fields and vtable out of the literal's layout -
            // garbage, and a crash on a method call (#494).
            if (auto classType = dyn_cast<mlir_ts::ClassType>(type))
            {
                emitError(location, "an object literal with methods or accessors cannot be assigned to class '")
                    << to_print(classType)
                    << "': a class instance takes its methods from its class; use 'new', or an interface type";
                return mlir::failure();
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

    // The first interface member of a union whose every required field and method an object literal, a
    // tuple or a class object has, by name; null for another value or when none fits.
    mlir::Type MLIRGenImpl::interfaceMemberFitting(mlir_ts::UnionType unionType, mlir::Type valueType)
    {
        if (!isa<mlir_ts::ObjectType, mlir_ts::TupleType, mlir_ts::ConstTupleType, mlir_ts::ClassType>(valueType))
        {
            return mlir::Type();
        }

        // a field or a method of that name, as `"k" in v` asks of one value (mlirGenInLogic)
        auto has = [&](mlir::Attribute id) { return !!mth.getFieldTypeByFieldName(valueType, id); };

        std::function<bool(InterfaceInfo::TypePtr)> fits = [&](InterfaceInfo::TypePtr interfaceInfo) {
            for (auto &extent : interfaceInfo->extends)
            {
                if (!fits(std::get<1>(extent)))
                {
                    return false;
                }
            }

            for (auto &method : interfaceInfo->methods)
            {
                if (!method.isConditional && !has(builder.getStringAttr(method.name)))
                {
                    return false;
                }
            }

            for (auto &field : interfaceInfo->fields)
            {
                if (!field.isConditional && !has(field.id))
                {
                    return false;
                }
            }

            return true;
        };

        for (auto member : unionType.getTypes())
        {
            auto interfaceType = dyn_cast<mlir_ts::InterfaceType>(member);
            auto interfaceInfo = interfaceType ? getInterfaceInfoByFullName(interfaceType.getName().getValue()) : InterfaceInfo::TypePtr();
            if (interfaceInfo && fits(interfaceInfo))
            {
                return member;
            }
        }

        return mlir::Type();
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

                    // an object literal, a tuple or a class object given to a union of interfaces
                    // (`move({ swim: () => "s" })` for `Fish | Bird`): it becomes the first interface it has every member of, and the
                    // union's tag says which (`"swim" in a`)
                    if (auto memberType = interfaceMemberFitting(unionType, valueType))
                    {
                        CAST(value, location, memberType, value, genContext);
                        return V(builder.create<mlir_ts::CastOp>(location, type, value));
                    }

                    // one member and null (`{ k: number } | null`): what is not null is that member -
                    // an object literal `{ k: 5 }`, whose field is an s32, is converted to it
                    if (!isa<mlir_ts::NullType>(valueType) && types.size() == 2
                        && llvm::any_of(types, [](mlir::Type subType) { return isa<mlir_ts::NullType>(subType); }))
                    {
                        auto memberType = isa<mlir_ts::NullType>(types[0]) ? types[1] : types[0];
                        CAST(value, location, memberType, value, genContext);
                        return V(builder.create<mlir_ts::CastOp>(location, type, value));
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

    // `T | null` without a tag is T's own pointer, and null only there for the checker (strict null
    // checks). A value of another type becomes a T first - an object literal an interface, a const
    // array an array - the way it would be cast to T alone: cast straight to the union, that
    // conversion was left undone for the lowering, which can't do it.
    std::optional<ValueOrLogicalResult> MLIRGenImpl::castToNullableUnionType(mlir::Location location, mlir::Type type, mlir::Value value, mlir::Type valueType, const GenContext &genContext)
    {
        auto unionType = dyn_cast<mlir_ts::UnionType>(type);
        if (!unionType)
        {
            return std::nullopt;
        }

        mlir::Type baseType;
        if (mth.isUnionTypeNeedsTag(location, unionType, baseType) || !baseType || baseType == valueType)
        {
            return std::nullopt;
        }

        // `C | null` given to `I | null`: both are stored as their one member (null is a null pointer
        // in it), so this is the class-to-interface cast of `C` to `I`. A plain cast read the class
        // pointer as an interface. `J | null` given to `I | null` is likewise the cast of `J` to `I`,
        // which gives an `I` that `J` does not extend a vtable of its own (castInterfaceToInterface).
        if (auto srcUnionType = dyn_cast<mlir_ts::UnionType>(valueType); srcUnionType && isa<mlir_ts::InterfaceType>(baseType))
        {
            mlir::Type srcBaseType;
            if (mth.isUnionTypeNeedsTag(location, srcUnionType, srcBaseType)
                || !isa_and_nonnull<mlir_ts::ClassType, mlir_ts::InterfaceType>(srcBaseType))
            {
                return std::nullopt;
            }

            auto member = builder.create<mlir_ts::GetValueFromUnionOp>(location, srcBaseType, value);
            CAST_A(casted, location, baseType, member, genContext);
            return V(builder.create<mlir_ts::CastOp>(location, type, casted));
        }

        // null given to `I | null` is the null interface, as null given to `I` is (castToInterfaceSpecialCases)
        if (isa<mlir_ts::NullType, mlir_ts::UndefinedType>(valueType) && isa<mlir_ts::InterfaceType>(baseType))
        {
            auto casted = cast(location, baseType, value, genContext, /*disableStrictNullCheck=*/true);
            EXIT_IF_FAILED_OR_NO_VALUE(casted)
            return V(builder.create<mlir_ts::CastOp>(location, type, V(casted)));
        }

        if (isa<mlir_ts::NullType>(valueType) || isa<mlir_ts::UndefinedType>(valueType) || isa<mlir_ts::AnyType>(valueType)
            || isa<mlir_ts::OptionalType>(valueType) || isa<mlir_ts::UnionType>(valueType))
        {
            return std::nullopt;
        }

        auto types = unionType.getTypes();
        if (std::find(types.begin(), types.end(), valueType) != types.end())
        {
            return std::nullopt;
        }

        CAST(value, location, baseType, value, genContext);
        return V(builder.create<mlir_ts::CastOp>(location, type, value));
    }

    // `number | null`, `{ k: number } | null`: a value type or null is a union with a tag - one member
    // and null. castFromUnion turns every member into the target type, and null is not a boolean
    // ("can't cast from 'null' to 'boolean'"), nor does it know an object type at all. Tested for
    // truth it is false when it holds null, as text it is "null". Anything else stays with
    // castFromUnion: `let x: number = n` is refused, as TypeScript refuses it - a narrowing or `n!`
    // reads the member.
    std::optional<ValueOrLogicalResult> MLIRGenImpl::castFromNullableUnion(mlir::Location location, mlir::Type type, mlir::Value value, mlir_ts::UnionType unionType, const GenContext &genContext)
    {
        if (!isa<mlir_ts::BooleanType>(type) && !isa<mlir_ts::StringType>(type))
        {
            return std::nullopt;
        }

        mlir::Type memberType;
        auto hasNull = false;
        for (auto subType : unionType.getTypes())
        {
            if (isa<mlir_ts::NullType>(subType))
            {
                hasNull = true;
                continue;
            }

            if (memberType)
            {
                return std::nullopt;
            }

            memberType = subType;
        }

        if (!hasNull || !memberType)
        {
            return std::nullopt;
        }

        auto readMember = [&]() { return V(builder.create<mlir_ts::GetValueFromUnionOp>(location, memberType, value)); };

        auto nullValue = builder.create<mlir_ts::NullOp>(location, getNullType());
        auto isNull = builder.create<mlir_ts::LogicalBinaryOp>(
            location, getBooleanType(), builder.getI32IntegerAttr((int)SyntaxKind::EqualsEqualsEqualsToken), value, nullValue);

        MLIRCodeLogicHelper mclh(builder, location, compileOptions);
        return mclh.conditionalValue(isNull,
            [&]() -> ValueOrLogicalResult {
                if (isa<mlir_ts::BooleanType>(type))
                {
                    return V(builder.create<mlir_ts::ConstantOp>(location, getBooleanType(), builder.getBoolAttr(false)));
                }

                return V(builder.create<mlir_ts::ConstantOp>(location, getStringType(), builder.getStringAttr("null")));
            },
            [&](mlir::Type) -> ValueOrLogicalResult {
                return cast(location, type, readMember(), genContext);
            });
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
                if (auto result = castFromNullableUnion(location, type, value, unionType, genContext))
                {
                    return *result;
                }

                return castFromUnion(location, type, value, genContext);
            }

            // ...and a @value type's `T | null` stored as T (MLIRTypeHelper::getValueNullNiche): null
            // is "null", not T's fields read from a zeroed T
            if (isa<mlir_ts::StringType>(type)
                && isa<mlir_ts::ClassType, mlir_ts::ValueType>(baseType)
                && llvm::any_of(unionType.getTypes(), [](auto subType) { return isa<mlir_ts::NullType>(subType); }))
            {
                return castNullableToString(location, type, value, baseType, genContext);
            }

            // A union of literals, as `c ? 1 : 0` is (`0 | 1`), is stored as its literals' type: cast
            // it to that type first. Cast straight to another type (`index`, a tuple field's type),
            // it reached the lowering as a union, which sign-extended the union itself.
            if (!toAny)
            {
                if (auto literalBaseType = getLiteralUnionBaseType(location, valueType); literalBaseType && literalBaseType != type)
                {
                    CAST_A(baseValue, location, literalBaseType, value, genContext);
                    return cast(location, type, baseValue, genContext);
                }
            }
        }

        // TODO: issue is with casting to Boolean type from union type for example, you need to cast optional type to boolean to check value
        // get rid of using "OptionalType" and use Union for it with "| undefined"
        // unwrapping optional value to work with union inside, we need it as ' | undefined ' is part of union type
        if (auto optType = dyn_cast<mlir_ts::OptionalType>(valueType))
        {
            // ...but not a @value type's that may be missing put into a string: undefined is "undefined",
            // not the default of its union (a zeroed value, read as null or as T's fields) - below
            auto missingValueToString =
                isa<mlir_ts::StringType>(type) && MLIRTypeHelper::getMissingValueElement(optType);
            if (isa<mlir_ts::UnionType>(optType.getElementType()) && !missingValueToString)
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
                || isa<mlir_ts::ClassType>(type)
                // a Shared<T> handle (own spec 23.1) is checked by its `typeof` name, "object"
                || isa<mlir_ts::SharedType>(type))
            {
                return castPrimitiveTypeFromAny(location, type, value, genContext);
            }

            if (isa<mlir_ts::TupleType>(type) && !genContext.allowPartialResolve)
            {
                return castAnyToTuple(location, type, value, genContext);
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
                    mlir_ts::FunctionType::get(builder.getContext(), hybridFuncType.getInputs(), hybridFuncType.getResults(),
                                               hybridFuncType.isVarArg(), hybridFuncType.isCVarArg()),
                    value);
                return V(builder.create<mlir_ts::CastOp>(location, type, funcValue));
            }
        }

        return std::nullopt;
    }

    // Names the part of the signature that does not match, rather than only the two types. The return
    // type is the common one: an integer literal is `s32` here, so a function with an inferred return
    // (`function one() { return 1; }` is `() => s32`) does not match `() => number` - hence the hint to
    // declare the return type.
    void MLIRGenImpl::emitFunctionTypeMismatch(mlir::Location location, mlir::Type valueType, mlir::Type type, MatchResult match)
    {
        auto inFuncType = mth.GetFunctionType(valueType);
        auto resFuncType = mth.GetFunctionType(type);

        auto diag = emitError(location) << "can't assign '" << to_print(valueType) << "' to '" << to_print(type) << "'";
        switch (match.result)
        {
        case MatchResultType::NotMatchArgCount:
            diag << ": it takes " << inFuncType.getNumInputs() << " argument(s), " << resFuncType.getNumInputs()
                 << " expected";
            break;
        case MatchResultType::NotMatchArg:
            diag << ": argument #" << (match.index + 1) << " does not match";
            break;
        case MatchResultType::NotMatchResultCount:
            diag << ": one returns a value and the other does not";
            break;
        case MatchResultType::NotMatchResult:
            if (match.index < inFuncType.getNumResults() && match.index < resFuncType.getNumResults())
            {
                auto inResultType = inFuncType.getResult(match.index);
                auto resResultType = resFuncType.getResult(match.index);
                diag << ": return type '" << to_print(inResultType) << "' is not '" << to_print(resResultType)
                     << "'. Declare the return type to get one, for example 'function f(): "
                     << to_print(resResultType) << "'";
            }
            else
            {
                diag << ": the return types do not match";
            }

            break;
        default:
            break;
        }
    }

    // A function taking fewer parameters than the function type it is given as (`apply(one)` where `apply` takes
    // `(a: number, b: number) => number` and `one` takes `a` only): TypeScript lets it ignore the rest. A named
    // function or an arrow that captures nothing is given as a function of its own that takes them all and calls
    // it with the first ones - called with every argument the type says, it would read them where it does not
    // expect them. An arrow that captures is the same, bound over its captures: the function of its own takes the
    // captures first, as the arrow does, and the closure is bound to it instead. Null where it is not that (a bound
    // method, parameters or results that differ).
    mlir::Value MLIRGenImpl::fewerParametersThunk(mlir::Location location, mlir::Value value, mlir::Type type)
    {
        auto boundOp = value.getDefiningOp<mlir_ts::CreateBoundFunctionOp>();
        auto symbolRefOp = boundOp ? boundOp.getFunc().getDefiningOp<mlir_ts::SymbolRefOp>()
                                   : value.getDefiningOp<mlir_ts::SymbolRefOp>();
        auto inFuncType = symbolRefOp ? dyn_cast<mlir_ts::FunctionType>(symbolRefOp.getType()) : mlir_ts::FunctionType();
        auto resFuncType = mth.GetFunctionType(type);
        // a closure's own value is its function bound over the captures (a method's is bound to its object)
        auto closure = boundOp && boundOp->hasAttr(OWNS_CAPTURE_ATTR_NAME);
        if (!symbolRefOp || !inFuncType || !resFuncType || inFuncType.isVarArg() || inFuncType.isCVarArg()
            || resFuncType.isVarArg() || resFuncType.isCVarArg() || (boundOp && (!closure || inFuncType.getInputs().empty())))
        {
            return mlir::Value();
        }

        // no result and a `void` one are the same (a declared `=> void` keeps its result)
        auto isVoid = [](mlir::TypeRange results) {
            return results.empty() || (results.size() == 1 && isa<mlir_ts::VoidType>(results.front()));
        };

        auto inInputs = inFuncType.getInputs();
        SmallVector<mlir::Type> resInputs;
        if (boundOp)
        {
            resInputs.push_back(inInputs.front());
        }

        resInputs.append(resFuncType.getInputs().begin(), resFuncType.getInputs().end());
        auto sameResults = inFuncType.getResults() == resFuncType.getResults()
            || (isVoid(inFuncType.getResults()) && isVoid(resFuncType.getResults()));
        if (inInputs.size() >= resInputs.size() || !sameResults
            || !llvm::equal(inInputs, llvm::ArrayRef<mlir::Type>(resInputs).take_front(inInputs.size())))
        {
            return mlir::Value();
        }

        auto calleeName = symbolRefOp.getIdentifier();
        auto thunkName = (calleeName + ".args" + std::to_string(resInputs.size())).str();
        auto thunkType = mlir_ts::FunctionType::get(builder.getContext(), resInputs, resFuncType.getResults(), false);
        if (!theModule.lookupSymbol(thunkName))
        {
            mlir::OpBuilder::InsertionGuard guard(builder);

            // compiler glue with no source to step through, as INSTANCEOF_NONE_NAME's
            mlir::Location noLocation = mlir::UnknownLoc::get(builder.getContext());
            auto funcOp = mlir_ts::FuncOp::create(noLocation, thunkName, thunkType);
            funcOp.setPrivate();
            funcOp->setAttr("internal_linkage", builder.getUnitAttr());

            // ...but with --di a function of its own to the debugger, as the catch copy thunk is
            // (mlirGenCatchCopyThunk): its call into the callee needs a location, or LLVM rejects the
            // module ("inlinable function call in a function with a DISubprogram location must have a
            // debug location"). A subprogram for the thunk at the place that first needs it, the body
            // scoped under it.
            DITableScopeT debugFuncScope(debugScope);
            if (compileOptions.generateDebugInfo)
            {
                MLIRDebugInfoHelper mdi(builder, debugScope);
                auto plainLocation = stripMetadata(location);
                funcOp->setLoc(mdi.getSubprogram(plainLocation, thunkName, thunkName, plainLocation));
                noLocation = locFuseWithScope(plainLocation);
            }

            auto &entryBlock = *funcOp.addEntryBlock();
            builder.setInsertionPointToStart(&entryBlock);

            auto results = inFuncType.getResults();
            auto hasResult = !results.empty() && !isa<mlir_ts::VoidType>(results.front());
            auto entryOp = builder.create<mlir_ts::EntryOp>(
                noLocation, hasResult ? mlir::Type(mlir_ts::RefType::get(results.front())) : mlir::Type());

            auto callee = builder.create<mlir_ts::SymbolRefOp>(noLocation, inFuncType,
                                                               mlir::FlatSymbolRefAttr::get(builder.getContext(), calleeName));
            auto callOp = builder.create<mlir_ts::CallIndirectOp>(noLocation, callee,
                                                                  entryBlock.getArguments().take_front(inInputs.size()));
            if (hasResult)
            {
                // the callee's result is the thunk's, handed on as `return f(x)` hands it: owned, and
                // taken over by the return (a string `map` builds came back as a second reference)
                if (mth.ownsHeapMemory(noLocation, results.front()))
                {
                    callOp->setAttr(OWNED_RESULT_ATTR_NAME, builder.getUnitAttr());
                    callOp->setAttr(OWNED_RESULT_CONSUMED_ATTR_NAME, builder.getUnitAttr());
                }

                builder.create<mlir_ts::ReturnValOp>(noLocation, callOp.getResult(0), entryOp.getReference());
            }

            builder.create<mlir_ts::ExitOp>(noLocation, hasResult ? entryOp.getReference() : mlir::Value());

            theModule.push_back(funcOp);
        }

        if (boundOp)
        {
            // the closure itself is bound to it, keeping what owns its captures (its retain and release)
            mlir::OpBuilder::InsertionGuard guard(builder);
            builder.setInsertionPoint(boundOp);
            auto thunkRef = builder.create<mlir_ts::SymbolRefOp>(
                location, thunkType, mlir::FlatSymbolRefAttr::get(builder.getContext(), thunkName));
            boundOp.getFuncMutable().assign(thunkRef);
            boundOp.getResult().setType(mlir_ts::BoundFunctionType::get(builder.getContext(), thunkType));
            return boundOp.getResult();
        }

        return builder.create<mlir_ts::SymbolRefOp>(location, thunkType,
                                                    mlir::FlatSymbolRefAttr::get(builder.getContext(), thunkName));
    }

    mlir::LogicalResult MLIRGenImpl::verifyFunctionCastRules(mlir::Location location, mlir::Type type, mlir::Value &value, mlir::Type &valueType, const GenContext &genContext)
    {
        if (mth.isAnyFunctionType(valueType) && mth.isAnyFunctionType(type)) {

            if (mth.isGenericType(valueType))
            {
                // need to instantiate generic method
                auto result = instantiateSpecializedFunction(location, value, type, genContext);
                EXIT_IF_FAILED(result);

                // An arrow that captures comes back as a new value, the closure bound over its
                // captured variables; the old reference is the bare function and calling it reads
                // the captures through a null pointer (`new W((v) => { seen += 1; })`). Without
                // captures the reference was retyped in place.
                if (auto resultValue = V(result))
                {
                    value = resultValue;
                }

                valueType = value.getType();
            }

            // A C variadic function is called with C's variadic convention and a TypeScript
            // function is not, so neither stands in for the other: through a value of the wrong
            // kind the extra arguments would land in the wrong registers.
            if (isCVarArgFunctionType(valueType) != isCVarArgFunctionType(type))
            {
                emitError(location, "can't assign '")
                    << to_print(valueType) << "' to '" << to_print(type) << "': only one of them is a C variadic function";
                return mlir::failure();
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
                auto test = mth.TestFunctionTypesMatchWithObjectMethods(location, valueType, type);
                if (test.result == MatchResultType::NotMatchArgCount)
                {
                    if (auto thunk = fewerParametersThunk(location, value, type))
                    {
                        value = thunk;
                        valueType = thunk.getType();
                        test = mth.TestFunctionTypesMatchWithObjectMethods(location, valueType, type);
                    }
                }

                if (test.result != MatchResultType::Match)
                {
                    emitFunctionTypeMismatch(location, valueType, type, test);
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
            // a union is what the function was most likely meant for: say which of its function members
            // it does not match, instead of naming the whole union
            if (auto unionType = dyn_cast<mlir_ts::UnionType>(type))
            {
                for (auto memberType : unionType.getTypes())
                {
                    if (!mth.isAnyFunctionType(memberType))
                    {
                        continue;
                    }

                    auto test = mth.TestFunctionTypesMatchWithObjectMethods(location, valueType, memberType);
                    if (test.result != MatchResultType::Match)
                    {
                        emitFunctionTypeMismatch(location, valueType, memberType, test);
                        return mlir::failure();
                    }
                }
            }

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

        // Two specializations of one generic class keep fields of different types in the same
        // place: `const b: Box<string> = new Box<number>(1)` read a number as a string and crashed.
        // One's type arguments have to extend the other's - Box<Derived> as Box<Base>, or back as
        // a downcast - which is TypeScript's rule for a type assertion as well.
        if (auto classType = dyn_cast<mlir_ts::ClassType>(type))
        {
            if (auto valueClassType = dyn_cast<mlir_ts::ClassType>(valueType))
            {
                auto classInfo = getClassInfoByFullName(classType.getName().getValue());
                auto valueClassInfo = getClassInfoByFullName(valueClassType.getName().getValue());
                if (classInfo && valueClassInfo && classInfo->originClassType
                    && classInfo->originClassType == valueClassInfo->originClassType)
                {
                    llvm::StringMap<std::pair<ts::TypeParameterDOM::TypePtr,mlir::Type>> typeParamsWithArgs;
                    if (!isTrue(mth.extendsType(location, valueType, type, typeParamsWithArgs))
                        && !isTrue(mth.extendsType(location, type, valueType, typeParamsWithArgs)))
                    {
                        emitError(location, "type ") << to_print(valueType) << " is not assignable to type " << to_print(type);
                        return mlir::failure();
                    }
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

        if (mlir::failed(verifyUnrelatedClassCast(location, type, valueType)))
        {
            return mlir::failure();
        }

        return mlir::success();
    }

    // Two classes where neither extends the other are compatible by their members, as in
    // TypeScript: `const b: B = new A()` with `A { v: number }` and `B { v: string }` compiled to a
    // plain cast, and reading `b.v` read a number as a string and crashed. Like an assertion, the
    // cast needs one side's fields to be found in the other with types that extend them.
    mlir::LogicalResult MLIRGenImpl::verifyUnrelatedClassCast(mlir::Location location, mlir::Type type, mlir::Type valueType)
    {
        std::string mismatch;
        if (areIncompatibleUnrelatedClasses(location, type, valueType, mismatch))
        {
            emitError(location, "type ") << to_print(valueType) << " is not assignable to type " << to_print(type) << ": " << mismatch;
            return mlir::failure();
        }

        return mlir::success();
    }

    bool MLIRGenImpl::areIncompatibleUnrelatedClasses(mlir::Location location, mlir::Type type, mlir::Type valueType, std::string &mismatch)
    {
        auto classType = dyn_cast<mlir_ts::ClassType>(type);
        auto valueClassType = dyn_cast<mlir_ts::ClassType>(valueType);
        if (!classType || !valueClassType || classType == valueClassType
            || mth.isGenericType(classType) || mth.isGenericType(valueClassType))
        {
            return false;
        }

        auto classInfo = getClassInfoByFullName(classType.getName().getValue());
        auto valueClassInfo = getClassInfoByFullName(valueClassType.getName().getValue());
        if (!classInfo || !valueClassInfo
            || classInfo->hasBase(valueClassType) || valueClassInfo->hasBase(classType)
            // two specializations of one generic compare by their type arguments
            || (classInfo->originClassType && classInfo->originClassType == valueClassInfo->originClassType))
        {
            return false;
        }

        // only an F is an F: a call on a @final class is a direct call to its own method, which
        // would run on the other class's fields
        if (classInfo->isFinal)
        {
            mismatch = "'" + classInfo->name.str() + "' is @" FINAL_DECORATOR;
            return true;
        }

        // the data fields, inherited ones included; not the internal ones (.vtbl) nor the storage
        // a derived class embeds for its base
        std::function<void(ClassInfo::TypePtr, llvm::StringMap<mlir::Type> &)> collectFields =
            [&](ClassInfo::TypePtr info, llvm::StringMap<mlir::Type> &fields) {
                for (auto &base : info->baseClasses)
                {
                    collectFields(base, fields);
                }

                auto storageType = dyn_cast<mlir_ts::ClassStorageType>(info->classType.getStorageType());
                if (!storageType)
                {
                    return;
                }

                for (auto &field : storageType.getFields())
                {
                    auto strId = dyn_cast_or_null<mlir::StringAttr>(field.id);
                    if (!strId || strId.getValue().starts_with(".")
                        || llvm::any_of(info->baseClasses, [&](auto &base) { return strId.getValue() == base->fullName; }))
                    {
                        continue;
                    }

                    fields[strId.getValue()] = field.type;
                }
            };

        llvm::StringMap<mlir::Type> fields;
        llvm::StringMap<mlir::Type> valueFields;
        collectFields(classInfo, fields);
        collectFields(valueClassInfo, valueFields);

        // every field of `to` is in `from`, of a type that extends it
        auto fits = [&](llvm::StringMap<mlir::Type> &from, llvm::StringMap<mlir::Type> &to, std::string &mismatch) {
            for (auto &field : to)
            {
                auto found = from.find(field.getKey());
                if (found == from.end())
                {
                    mismatch = "'" + field.getKey().str() + "' is missing";
                    return false;
                }

                llvm::StringMap<std::pair<ts::TypeParameterDOM::TypePtr,mlir::Type>> typeParamsWithArgs;
                if (!isTrue(mth.extendsType(location, found->getValue(), field.getValue(), typeParamsWithArgs)))
                {
                    mismatch = "'" + field.getKey().str() + "' is " + to_print(found->getValue()) + ", not " + to_print(field.getValue());
                    return false;
                }
            }

            return true;
        };

        std::string reverseMismatch;
        return !fits(valueFields, fields, mismatch) && !fits(fields, valueFields, reverseMismatch);
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
        ss << S("function __unbox<T>(a: any) : T {\nconst b = a;\n");
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
            // A handle's tag names it "object" (TypeOfOpHelper), and there is no `instanceof` for a
            // handle, so the name is all that is checked: an `any` holding a Shared<B> passes for a
            // Shared<A>, as would any other value tagged "object" (own spec 23.7).
            .Case<mlir_ts::SharedType>([&](auto _) { typeOfs["object"] = true; })
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

                // An `any` holding an interface value holds an object behind it, which may be one
                // of these classes: the interface's vtable answers (INTERFACE_VTABLE_HEADER_SLOTS).
                // `b` is `a` before `typeof` narrowed it to the first word of the box.
                if (!typeOfs.contains("interface"))
                {
                    ss << S(" else if (typeof a == 'interface') { const i = <TYPE_ANY_INTERFACE_ALIAS>b;\n");
                    for (auto [index, _] : enumerate(classInstances))
                    {
                        ss << S("if (i instanceof TYPE_INST_ALIAS");
                        ss << index;
                        ss << S(") return <TYPE_INST_ALIAS");
                        ss << index;
                        ss << S(">i;\n");
                    }

                    ss << S(" }\n");
                }
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

            // `if (a)` and `!a` of an `any` holding undefined or null: false, as in JavaScript -
            // it threw "Can't cast from any type"
            if (mlir::isa<mlir_ts::BooleanType>(type) || mlir::isa<mlir_ts::TypePredicateType>(type)) {
                ss << "\nif (typeof a == 'undefined') return false;";
                ss << "\nif (typeof a == 'null') return false;";
            }
        }

        // Every falsy value - 0, NaN, "", false, null, undefined, 0n - is a primitive answered above.
        // What is left - a class instance, an object, an array, a function - is truthy, as in
        // JavaScript: `if (a)` of an `any` holding one threw instead.
        if (mlir::isa<mlir_ts::BooleanType>(type) || mlir::isa<mlir_ts::TypePredicateType>(type))
        {
            ss << "\nreturn true;\n";
        }
        else
        {
            ss << "\nthrow \"Can't cast from any type\";\n";
        }

        ss << S("}\n");

        auto src = ss.str();

        // The generated function's body is generated when the call below instantiates it, not when
        // it is parsed, so the call site has to stay the location until then. Without it a
        // diagnostic there took the generated text's offset as a position in the user's file - an
        // error reported inside a comment on line 1.
        MLIRLocationGuard vgLoc(overwriteLoc);
        overwriteLoc = location;

        if (mlir::failed(parsePartialStatements(src)))
        {
            assert(false);
            return mlir::failure();
        }

        auto funcResult = resolveIdentifier(location, funcName, genContext);

        assert(funcResult);

        GenContext funcCallGenContext(genContext);
        funcCallGenContext.typeAliasMap.insert({".TYPE_ALIAS_T", type});

        for (auto [index, instanceOfType] : enumerate(classInstances))
        {
            funcCallGenContext.typeAliasMap.insert({"TYPE_INST_ALIAS" + std::to_string(index), instanceOfType});
        }

        // Any interface type will do to read an interface value back out of the box: they all
        // lower to one layout, and nothing but the vtable and `this` is read from it. A private
        // name of its own keeps it from being looked up as a declared interface.
        funcCallGenContext.typeAliasMap.insert(
            {"TYPE_ANY_INTERFACE_ALIAS",
             mlir_ts::InterfaceType::get(builder.getContext(), mlir::FlatSymbolRefAttr::get(builder.getContext(), ".any_interface"))});

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

    // `any` to an interface. The box knows the kind of what it holds, not which interface an interface
    // value was made for, so the object behind it is asked what class it is (`instanceof`):
    // - a class with a vtable for the interface - it declares it, or a cast somewhere in the module
    //   makes one (classInterfaceVTablesCast, filled from the discovery pass on, so a cast later in
    //   the module counts) - gives the interface through that vtable, from a class object or an
    //   interface value alike;
    // - any other class of this module can't: no vtable for the interface was made for it, and none
    //   is made here (one made here would be a way to the class's fields that nothing else in the
    //   module shows), so the cast throws. A class object no class answers for throws too;
    // - an object literal boxed as it is (`typeof` 'object' or 'tuple') throws: the box does not say
    //   which fields it has, so nothing can lay out a vtable for it;
    // - null and undefined give the null interface;
    // - anything else - an interface value over an object literal or a class of another module, a
    //   number used as a token - is read as the interface's pair, as before: an `any` holding the
    //   interface it is cast to round-trips.
    ValueOrLogicalResult MLIRGenImpl::castAnyToInterface(mlir::Location location, mlir_ts::InterfaceType interfaceType,
                                                         InterfaceInfo::TypePtr interfaceInfo, mlir::Value value,
                                                         const GenContext &genContext)
    {
        SmallVector<ClassInfo::TypePtr> implementers;
        SmallVector<ClassInfo::TypePtr> others;
        llvm::SmallPtrSet<ClassInfo *, 16> seenClasses;
        llvm::SmallPtrSet<NamespaceInfo *, 16> seenNamespaces;
        std::function<void(NamespaceInfo::TypePtr)> collectClasses = [&](NamespaceInfo::TypePtr namespaceInfo) {
            if (!seenNamespaces.insert(namespaceInfo.get()).second)
            {
                return;
            }

            for (auto &classEntry : namespaceInfo->classesMap)
            {
                auto classInfo = classEntry.getValue();
                // `instanceof` asks the class's `.instanceOf`, which a @value or @cpp class has none of
                if (!classInfo || !classInfo->classType || !classInfo->hasRTTI || classInfo->isValue
                    || classInfo->isCppVTable || !seenClasses.insert(classInfo.get()).second)
                {
                    continue;
                }

                if (classInfo->getImplementIndex(interfaceInfo->fullName) >= 0
                    || classInterfaceVTablesCast.contains(interfaceVTableNameForClass(classInfo, interfaceInfo)))
                {
                    implementers.push_back(classInfo);
                }
                else if (!classInfo->isDeclaration)
                {
                    others.push_back(classInfo);
                }
            }

            for (auto &namespaceEntry : namespaceInfo->namespacesMap)
            {
                collectClasses(namespaceEntry.getValue());
            }
        };

        collectClasses(rootNamespace);

        // info, we add "_" extra as scanner append "_" in front of "__";
        auto funcName = "___unbox_interface";

        // the classes asked differ from one interface to another
        removeGenericFunctionMap(funcName);

        auto throwNotImplemented = [&](stringstream &ss, StringRef className) {
            ss << S("throw \"Can't cast from any: class '") << stows(className.str())
               << S("' does not implement interface '") << stows(interfaceInfo->fullName.str()) << S("'\";\n");
        };

        // `a` narrowed by `typeof` is the first word of the box; `b` is the box
        stringstream ss;
        ss << S("function __unbox_interface<T>(a: any) : T {\nconst b = a;\n");
        ss << S("if (typeof a == 'class') {\n");
        for (auto [index, _] : enumerate(implementers))
        {
            ss << S("if (a instanceof TYPE_INST_ALIAS") << index << S(") return a;\n");
        }

        for (auto [index, other] : enumerate(others))
        {
            ss << S("if (a instanceof TYPE_OTHER_ALIAS") << index << S(") ");
            throwNotImplemented(ss, other->fullName);
        }

        ss << S("throw \"Can't cast from any: the class object has no vtable for interface '")
           << stows(interfaceInfo->fullName.str()) << S("'\";\n}\n");

        ss << S("if (typeof a == 'interface') {\nconst i = <TYPE_ANY_INTERFACE_ALIAS>b;\n");
        for (auto [index, _] : enumerate(implementers))
        {
            ss << S("if (i instanceof TYPE_INST_ALIAS") << index << S(") return <TYPE_INST_ALIAS") << index << S(">i;\n");
        }

        for (auto [index, other] : enumerate(others))
        {
            ss << S("if (i instanceof TYPE_OTHER_ALIAS") << index << S(") ");
            throwNotImplemented(ss, other->fullName);
        }

        ss << S("}\n");

        // an object literal, boxed as it is: the box says only that it is one, not what fields it has, so
        // no vtable can be found for it here, and its block read as the interface's pair called garbage
        ss << S("if (typeof a == 'object' || typeof a == 'tuple') throw \"Can't cast from any: an object literal has no vtable for interface '")
           << stows(interfaceInfo->fullName.str()) << S("'\";\n");
        // null and undefined have a box of their own under the counting models, not the null pair
        ss << S("if (typeof a == 'null' || typeof a == 'undefined') {\nconst n: TYPE_ANY_INTERFACE_ALIAS | null = null;\nreturn <T>n;\n}\n");
        ss << S("return <T><TYPE_ANY_INTERFACE_ALIAS>a;\n}\n");

        auto src = ss.str();

        // see castPrimitiveTypeFromAny: the call site is the location until the call below instantiates
        // the function
        MLIRLocationGuard vgLoc(overwriteLoc);
        overwriteLoc = location;

        if (mlir::failed(parsePartialStatements(src)))
        {
            assert(false);
            return mlir::failure();
        }

        auto funcResult = resolveIdentifier(location, funcName, genContext);

        assert(funcResult);

        GenContext funcCallGenContext(genContext);
        funcCallGenContext.typeAliasMap.insert({".TYPE_ALIAS_T", interfaceType});
        for (auto [index, classInfo] : enumerate(implementers))
        {
            funcCallGenContext.typeAliasMap.insert({"TYPE_INST_ALIAS" + std::to_string(index), classInfo->classType});
        }

        for (auto [index, classInfo] : enumerate(others))
        {
            funcCallGenContext.typeAliasMap.insert({"TYPE_OTHER_ALIAS" + std::to_string(index), classInfo->classType});
        }

        // see castPrimitiveTypeFromAny: any interface type reads an interface value out of the box
        funcCallGenContext.typeAliasMap.insert(
            {"TYPE_ANY_INTERFACE_ALIAS",
             mlir_ts::InterfaceType::get(builder.getContext(), mlir::FlatSymbolRefAttr::get(builder.getContext(), ".any_interface"))});

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

    // `any` to a tuple (an object type, `<{ n: number }>x`). The box holds its value's kind: a tuple is
    // read out of it, and anything else - a class object, a number - throws. It was read as the tuple's
    // layout whatever it held (`(<{ f: F }>x).f = g` with a class object in `x` wrote over its vtable).
    // `___check_tuple` only asks the box (it keeps nothing of it); the read after it is the raw one it
    // was. A tuple of other fields throws too: its descriptor's layout id is not the type's
    // (ts.IsTupleOfLayout) - `{ n: 5 }` holds an s32, which was read as a number.
    ValueOrLogicalResult MLIRGenImpl::castAnyToTuple(mlir::Location location, mlir::Type type, mlir::Value value,
                                                     const GenContext &genContext)
    {
        // info, we add "_" extra as scanner append "_" in front of "__";
        auto funcName = "___check_tuple";

        // the message names the type
        removeGenericFunctionMap(funcName);

        stringstream ss;
        ss << S("function __check_tuple<T>(a: any, sameLayout: boolean) {\n");
        ss << S("if (typeof a != 'tuple') throw \"Can't cast from any: the value is not an object of type '")
           << stows(to_print(type)) << S("'\";\n");
        ss << S("if (!sameLayout) throw \"Can't cast from any: the object's fields are not those of type '")
           << stows(to_print(type)) << S("'\";\n}\n");

        auto src = ss.str();

        // see castPrimitiveTypeFromAny: the call site is the location until the call below instantiates
        // the function
        MLIRLocationGuard vgLoc(overwriteLoc);
        overwriteLoc = location;

        if (mlir::failed(parsePartialStatements(src)))
        {
            assert(false);
            return mlir::failure();
        }

        auto funcResult = resolveIdentifier(location, funcName, genContext);

        assert(funcResult);

        GenContext funcCallGenContext(genContext);
        funcCallGenContext.typeAliasMap.insert({".TYPE_ALIAS_T", type});

        // asked before the call: every box's tag has a descriptor, a tuple's or not
        SmallVector<mlir::Value, 4> operands;
        operands.push_back(value);
        operands.push_back(builder.create<mlir_ts::IsTupleOfLayoutOp>(location, getBooleanType(), value, mlir::TypeAttr::get(type)));

        NodeFactory nf(NodeFactoryFlags::None);
        auto checked = mlirGenCallExpression(
            location,
            funcResult,
            {
                nf.createTypeReferenceNode(nf.createIdentifier(S(".TYPE_ALIAS_T")).as<Node>()),
            },
            operands,
            funcCallGenContext);
        EXIT_IF_FAILED(checked)

        return V(builder.create<mlir_ts::CastOp>(location, type, value));
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

                auto toBoolean = isa<mlir_ts::BooleanType>(type) || isa<mlir_ts::TypePredicateType>(type);

                // A union whose actual type has no null member but whose members all hold null in
                // their own pointer (`A | B | null`) is null when that pointer is null, whatever its
                // tag: asked first, so that no member's test - `instanceof` reads the vtable - meets a
                // null instance. A tag of its own for null is then never the one that tells.
                mlir::Type actualType;
                auto actualUnionType = mth.isUnionTypeNeedsTag(location, unionType, actualType)
                    ? dyn_cast<mlir_ts::UnionType>(actualType) : mlir_ts::UnionType();
                if (actualUnionType && mth.isNullAbsorbingUnion(actualUnionType))
                {
                    typeOfs.erase("null");

                    // false tested for truth, the text "null" as text (`String(null)`), null where the
                    // target takes null, and otherwise no value at all - as the end of this function
                    auto targetTakesNull = isa<mlir_ts::NullType>(type)
                        || (isa<mlir_ts::UnionType>(type)
                            && llvm::any_of(mlir::cast<mlir_ts::UnionType>(type).getTypes(), [](mlir::Type member) { return isa<mlir_ts::NullType>(member); }))
                        || (!compileOptions.strictNullChecks && MLIRTypeHelper::isNullAbsorbingType(type));
                    ss << S("if (t === null) ")
                       << (toBoolean ? S("return false;")
                           : isa<mlir_ts::StringType>(type) ? S("return \"null\";")
                           : targetTakesNull ? S("return null;")
                           : S("throw \"Can't cast from union type\";"))
                       << S("\n");
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
                    // Tested for truth, a null or undefined member is false, as for an `any`
                    // (castFromAny). `return t` there cast null to boolean - "can't cast from 'null'
                    // to 'boolean'" - so a union of null and two or more other types, which is what
                    // `enabled && handle` makes of a boolean and a `Handle | null`, could not be
                    // tested in an `if` at all.

                    // default
                    auto next = false;
                    for (auto& pair : typeOfs)
                    {
                        if (next) ss << S(" else ");

                        ss << S("if (typeof t == '");
                        ss << stows(pair.getKey().str());
                        ss << S("') ");
                        if (toBoolean && (pair.getKey() == "null" || pair.getKey() == "undefined"))
                        {
                            ss << S("return false;\n");
                        }
                        else if (pair.getKey() == "class")
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

                // kept until the call below has instantiated the function - see castPrimitiveTypeFromAny
                MLIRLocationGuard vgLoc(overwriteLoc);
                overwriteLoc = location;

                if (mlir::failed(parsePartialStatements(src)))
                {
                    assert(false);
                    return mlir::failure();
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

    // A union of a class and null needs no tag: it is the object's pointer, null when the union
    // holds null. A plain cast to string read that pointer as the text - the object's bytes, or a
    // crash for null - so test it first, as `??` does, and give the object its toString. (A null
    // `string` stays null when converted; it reads as "null" where it is printed or concatenated,
    // see nullStringAsText in LowerToLLVM.)
    ValueOrLogicalResult MLIRGenImpl::castNullableToString(mlir::Location location, mlir::Type type, mlir::Value value, mlir::Type baseType, const GenContext &genContext)
    {
        // a class is compared as its pointer; a @value type's union as itself, whose null is a pointer
        // in it (the lowering's valueNullCompare)
        mlir::Value compared = value;
        if (!isa<mlir_ts::ValueType>(baseType))
        {
            CAST_A(opaqueValue, location, getOpaqueType(), value, genContext);
            compared = opaqueValue;
        }

        auto nullVal = builder.create<mlir_ts::NullOp>(location, getNullType());
        auto isNull = builder.create<mlir_ts::LogicalBinaryOp>(
            location, getBooleanType(), builder.getI32IntegerAttr((int)SyntaxKind::EqualsEqualsEqualsToken), compared,
            nullVal);

        auto ifOp = builder.create<mlir_ts::IfOp>(location, mlir::TypeRange{type}, isNull, true);

        builder.setInsertionPointToStart(&ifOp.getThenRegion().front());
        auto nullText = builder.create<mlir_ts::ConstantOp>(location, type, builder.getStringAttr(NULL_NAME));
        builder.create<mlir_ts::ResultOp>(location, mlir::ValueRange{nullText});

        builder.setInsertionPointToStart(&ifOp.getElseRegion().front());
        // the union as its base type, now known not to be null: the same pointer, or the same value
        auto baseValue = V(builder.create<mlir_ts::CastOp>(location, baseType, value));
        CAST_A(text, location, type, baseValue, genContext);
        builder.create<mlir_ts::ResultOp>(location, mlir::ValueRange{text});

        builder.setInsertionPointAfter(ifOp);

        return V(ifOp.getResults().front());
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

        // a `?` member held as optional<T> needs a vtable of this object's own (patchAbsentOptionalFieldSlots):
        // it goes in a field after the object's own ones, so the block the interface holds frees it as well
        if (!getOptionalFieldSlots(mlir::cast<mlir_ts::TupleType>(tupleType), interfaceInfo).empty())
        {
            return castTupleWithOwnVTableToInterface(location, inEffective, mlir::cast<mlir_ts::TupleType>(tupleType),
                                                     interfaceInfo, genContext);
        }

        // TODO: finish it, what to finish it? maybe optimization not to create extra object?
        // convert Tuple to Object
        auto objType = mlir_ts::ObjectType::get(tupleType);
        auto valueAddr = builder.create<mlir_ts::NewOp>(location, mlir_ts::ValueRefType::get(tupleType), builder.getBoolAttr(false));

        // this block releases what its fields hold when it dies, so it has to take a reference
        // to each of them first - the same debt an array literal's data block carries (§9.21),
        // and inert until an interface became an owner (§9.31), which is why it went unnoticed
        mlirGenRetainCaptured(location, mlir::ValueRange{inEffective});

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

    // the object is copied into a block with one more field, `.vtable`, after its own: the vtable patched for
    // its `?` members (patchAbsentOptionalFieldSlots) is stored there, so it lives and dies with the block the
    // interface holds. The object's own fields keep their offsets, which are what the vtable holds.
    ValueOrLogicalResult MLIRGenImpl::castTupleWithOwnVTableToInterface(mlir::Location location, mlir::Value in,
                                    mlir_ts::TupleType tupleType, InterfaceInfo::TypePtr interfaceInfo, const GenContext &genContext)
    {
        auto inEffective = in;
        if (inEffective.getType() != tupleType)
        {
            CAST(inEffective, location, tupleType, inEffective, genContext);
        }

        // the vtable's type: the interface's slots typed by this object's fields
        auto plainObjType = mlir_ts::ObjectType::get(tupleType);
        auto vtableName = interfaceVTableNameForObject(plainObjType, interfaceInfo);
        auto globalVTable = resolveFullNameIdentifier(location, vtableName, true, genContext);
        if (!globalVTable)
        {
            if (mlir::failed(mlirGenObjectVirtualTableDefinitionForInterface(location, plainObjType, interfaceInfo, genContext)))
            {
                return mlir::failure();
            }

            globalVTable = resolveFullNameIdentifier(location, vtableName, true, genContext);
            if (!globalVTable)
            {
                return mlir::failure();
            }
        }

        auto vtableType = mlir::cast<mlir_ts::RefType>(globalVTable.getType()).getElementType();

        SmallVector<mlir_ts::FieldInfo> fields(tupleType.getFields().begin(), tupleType.getFields().end());
        fields.push_back({MLIRHelper::TupleFieldName(OWN_INTERFACE_VTABLE_FIELD_NAME, builder.getContext()), vtableType, false, mlir_ts::AccessLevel::Public});
        auto withVTableType = getTupleType(fields);

        SmallVector<mlir::Type> types;
        for (auto &field : tupleType.getFields())
        {
            types.push_back(field.type);
        }

        // as in castTupleToInterface: the block owns what its fields hold
        mlirGenRetainCaptured(location, mlir::ValueRange{inEffective});

        auto parts = builder.create<mlir_ts::DeconstructTupleOp>(location, types, inEffective);
        mlir::Value withVTable = builder.create<mlir_ts::UndefOp>(location, withVTableType);
        for (auto [index, part] : llvm::enumerate(parts.getResults()))
        {
            withVTable = builder.create<mlir_ts::InsertPropertyOp>(location, withVTableType, part, withVTable,
                                                                   MLIRHelper::getStructIndex(builder, index));
        }

        auto objType = mlir_ts::ObjectType::get(withVTableType);
        auto valueAddr = builder.create<mlir_ts::NewOp>(location, mlir_ts::ValueRefType::get(withVTableType), builder.getBoolAttr(false));
        builder.create<mlir_ts::StoreOp>(location, withVTable, valueAddr);
        auto inCasted = builder.create<mlir_ts::CastOp>(location, objType, valueAddr);

        auto vtableStorage = builder.create<mlir_ts::PropertyRefOp>(location, mlir_ts::RefType::get(vtableType), inCasted,
                                                                    (int)fields.size() - 1);

        return castObjectToInterface(location, inCasted, objType, interfaceInfo, genContext, vtableStorage);
    }

    ValueOrLogicalResult MLIRGenImpl::castObjectToInterface(mlir::Location location, mlir::Value in, mlir_ts::ObjectType objType,
                                    InterfaceInfo::TypePtr interfaceInfo, const GenContext &genContext, mlir::Value vtableStorage)
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

                // An object literal's members take this very block for `this`, laid out as it is: given the
                // clone through the interface, a member writing a field the clone widened (`count: 0`, an
                // s32, for `count: number`) writes it as the narrow type into the wider slot, and the
                // interface reads garbage. Such an object cannot be cast; its field has to be the
                // interface's type. One whose members never name that field through `this` can. An object
                // made in another module has no record here: any method of it may.
                auto isMethod = [](mlir::Type type) {
                    return isa<mlir_ts::FunctionType, mlir_ts::HybridFunctionType, mlir_ts::BoundFunctionType>(type);
                };
                auto thisFields = objectLiteralThisFields.find(storageTuple);
                auto recorded = thisFields != objectLiteralThisFields.end();
                auto namedThroughThis = [&](mlir::Attribute id) {
                    auto name = dyn_cast<mlir::StringAttr>(id);
                    return !recorded || !name || thisFields->second.contains(name.getValue()) || thisFields->second.contains("");
                };
                if (recorded || llvm::any_of(storageTuple.getFields(), [&](auto &field) { return isMethod(field.type); }))
                {
                    for (auto &field : storageTuple.getFields())
                    {
                        auto found = llvm::find_if(fields, [&](auto &cloneField) { return cloneField.id == field.id; });
                        if (found == fields.end() || found->type == field.type || isMethod(field.type) ||
                            !namedThroughThis(field.id))
                        {
                            continue;
                        }

                        auto diag = emitError(location) << "field '";
                        if (auto name = dyn_cast<mlir::StringAttr>(field.id))
                        {
                            diag << name.getValue();
                        }
                        else
                        {
                            diag << field.id;
                        }

                        diag << "' of the object is '" << to_print(field.type)
                                            << "' but interface '" << interfaceInfo->fullName << "' declares '"
                                            << to_print(found->type)
                                            << "': the object's methods would not see the converted field. Give the field "
                                               "the interface's type (for a number, a literal such as `0.0`, or `<number>0`)";
                        return mlir::failure();
                    }
                }

                auto newInterfaceTupleType = getTupleType(fields);

                CAST_A(unboxed, location, newInterfaceTupleType, in, genContext);

                auto valueAddr = builder.create<mlir_ts::NewOp>(location, mlir_ts::ValueRefType::get(newInterfaceTupleType), builder.getBoolAttr(false));

                // as in castTupleToInterface: the clone's block owns what it holds
                mlirGenRetainCaptured(location, mlir::ValueRange{unboxed});

                builder.create<mlir_ts::StoreOp>(location, unboxed, valueAddr);
                effectiveObjType = mlir_ts::ObjectType::get(newInterfaceTupleType);
                inEffective = builder.create<mlir_ts::CastOp>(location, effectiveObjType, valueAddr);

                emitWarning(location, "") << "Cloned object is used. Ensure all types are matching to interface: " << interfaceInfo->fullName;
            }
        }

        // a boxed object literal's block is not made by this cast, so it has no room for a vtable of its own:
        // its fields are copied into one that has (castTupleWithOwnVTableToInterface)
        if (!vtableStorage)
        {
            if (auto storageTuple = dyn_cast<mlir_ts::TupleType>(effectiveObjType.getStorageType()))
            {
                if (!getOptionalFieldSlots(storageTuple, interfaceInfo).empty())
                {
                    CAST_A(unboxed, location, storageTuple, inEffective, genContext);
                    return castTupleWithOwnVTableToInterface(location, unboxed, storageTuple, interfaceInfo, genContext);
                }
            }
        }

        auto result = mlirGenCreateInterfaceVTableForObject(location, inEffective, effectiveObjType, interfaceInfo, genContext, vtableStorage);
        EXIT_IF_FAILED_OR_NO_VALUE(result)
        auto createdInterfaceVTableForObject = V(result);

        LLVM_DEBUG(llvm::dbgs() << "\n!!"
                                << "@ created interface:" << createdInterfaceVTableForObject << "\n";);

        mlir::Value interfaceValue = builder.create<mlir_ts::NewInterfaceOp>(location,
            mlir::TypeRange{interfaceInfo->interfaceType}, inEffective, createdInterfaceVTableForObject);

        // An interface made from an object is a reference of its own, and on the paths that
        // reach here the object is very often a block allocated a few lines above for exactly
        // this cast - so nothing else holds it, and nothing else will give it back.
        //
        // Handing the result a reference is what lets every receiver treat it uniformly: a
        // local consumes it (§9.25), a return forwards it, and one that goes to an argument and
        // is then dropped - `trace({ start: pos, dir: rd }, scene)`, which is what this cast is
        // mostly used for - is released at the end of the block that made it (§9.30). Without
        // the +1 that last shape has no owner at all and neither retains nor releases, which is
        // where the whole of raytrace's remaining leak sat.
        builder.create<mlir_ts::RetainOp>(location, interfaceValue);
        interfaceValue.getDefiningOp()->setAttr(OWNED_RESULT_ATTR_NAME, builder.getUnitAttr());

        return V(interfaceValue);
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
