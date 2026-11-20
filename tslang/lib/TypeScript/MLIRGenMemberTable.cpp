// A class's member table (MEMBERS_NAME in Defines.h): what reads a class's members by name sees. Its address
// is the class vtable's slot after `.instanceOf` (CLASS_VTABLE_MEMBERS_SLOT), so an object's table is its
// actual class's.
//
// Every class with a vtable has one, as its vtable - made with the class - names it before the module knows
// whether anything reads the class by name. A class something reads by name (requestMemberTable: boxed into
// `any`, read with a key known only at run time), and every class of the module extending one, has its members
// and its keys adapter in it; an exported class has its members (for an importer's layout check) and no
// adapter; any other has an empty one, which keeps nothing alive.

#include "MLIRGenImpl.h"

#include "TypeScript/MLIRLogic/TypeOfOpHelper.h"

#undef DEBUG_TYPE
#define DEBUG_TYPE "mlir"

namespace typescript
{
namespace mlirgen
{

    // The type a field's entry hashes: what it is stored as, so the library's (from its source) and an importer's
    // (from the declarations it reads) agree - an enum is its values' type (the importer's enum is the one the
    // declarations print), a literal its base type.
    static mlir::Type memberHashType(mlir::Type type)
    {
        if (auto enumType = dyn_cast<mlir_ts::EnumType>(type))
        {
            return memberHashType(enumType.getElementType());
        }

        if (auto literalType = dyn_cast<mlir_ts::LiteralType>(type))
        {
            return memberHashType(literalType.getElementType());
        }

        if (auto optionalType = dyn_cast<mlir_ts::OptionalType>(type))
        {
            return mlir_ts::OptionalType::get(memberHashType(optionalType.getElementType()));
        }

        // an integer by its width: an enum's values are i32 in its module, si32 as declarations print it
        if (auto integerType = dyn_cast<mlir::IntegerType>(type))
        {
            return mlir::IntegerType::get(type.getContext(), integerType.getWidth());
        }

        return type;
    }

    void MLIRGenImpl::requestMemberTable(mlir::Type type)
    {
        if (auto classType = dyn_cast_or_null<mlir_ts::ClassType>(type))
        {
            memberTableRequests.insert(classType.getName().getValue());
        }
    }

    // the entries of a table: of an exported class, as an importer checks its layout against them, and of a class
    // read by name. Not under --export=none, where no class is exported. Only a class read by name gets a keys
    // adapter (makeMemberTables).
    bool MLIRGenImpl::isMemberTableRequested(ClassInfo::TypePtr classInfo)
    {
        return classInfo->isExport || isReadByName(classInfo);
    }

    // read by a key known at run time here (or a subclass of one)
    bool MLIRGenImpl::isReadByName(ClassInfo::TypePtr classInfo)
    {
        if (memberTableRequests.contains(classInfo->fullName))
        {
            return true;
        }

        // a subclass: an object of it may be read through its base
        return llvm::any_of(classInfo->baseClasses, [&](auto &base) { return isReadByName(base); });
    }

    // The entries of a class's table, sorted by name: its public and protected fields, accessors and methods,
    // and its bases'. Never a private one, a static one, nor the compiler's own (`.vtbl`, a base's storage).
    void MLIRGenImpl::collectMemberTableEntries(ClassInfo::TypePtr classInfo, SmallVector<MemberTableEntry> &entries)
    {
        llvm::StringSet<> seen;
        auto add = [&](StringRef name, int kind, mlir_ts::AccessLevel access, bool readOnly, mlir::Type type) {
            if (name.empty() || name.starts_with(".") || name.starts_with("#") || name == CONSTRUCTOR_NAME ||
                access == mlir_ts::AccessLevel::Private || !seen.insert(name).second)
            {
                return;
            }

            entries.push_back({name.str(), kind,
                               access == mlir_ts::AccessLevel::Protected ? MEMBER_ACCESS_PROTECTED : MEMBER_ACCESS_PUBLIC,
                               readOnly, type});
        };

        // the class's own members first: an override hides its base's
        std::function<void(ClassInfo::TypePtr)> walk = [&](ClassInfo::TypePtr info) {
            auto storageType = mlir::cast<mlir_ts::ClassStorageType>(info->classType.getStorageType());
            for (auto &field : storageType.getFields())
            {
                auto name = dyn_cast_or_null<mlir::StringAttr>(field.id);
                if (!name || llvm::any_of(info->baseClasses, [&](auto &base) { return base->fullName == name.getValue(); }))
                {
                    continue;
                }

                add(name.getValue(), MEMBER_KIND_FIELD, field.accessLevel, false, field.type);
            }

            for (auto &accessor : info->accessors)
            {
                if (!accessor.isStatic && accessor.get)
                {
                    add(accessor.name, MEMBER_KIND_ACCESSOR, accessor.getAccessLevel,
                        !(accessor.set && accessor.setAccessLevel == accessor.getAccessLevel), mlir::Type());
                }
            }

            for (auto &method : info->methods)
            {
                // an accessor's functions (`get_g`) are the accessor's entry, not methods of their own
                auto isAccessorFunction = llvm::any_of(info->accessors, [&](auto &accessor) {
                    return (accessor.get && accessor.get.name == method.funcName) ||
                           (accessor.set && accessor.set.name == method.funcName);
                });
                if (!method.isStatic && !isAccessorFunction)
                {
                    add(method.name, MEMBER_KIND_METHOD, method.accessLevel, true, method.funcType);
                }
            }

            for (auto &base : info->baseClasses)
            {
                walk(base);
            }
        };

        walk(classInfo);
        llvm::sort(entries, [](auto &a, auto &b) { return a.name < b.name; });
    }

    mlir_ts::TupleType MLIRGenImpl::getMemberTableHeaderType()
    {
        auto i32Type = builder.getI32Type();
        SmallVector<mlir_ts::FieldInfo> headerFields{
            {MLIRHelper::TupleFieldName("count", builder.getContext()), i32Type, false, mlir_ts::AccessLevel::Public},
            {MLIRHelper::TupleFieldName("flags", builder.getContext()), i32Type, false, mlir_ts::AccessLevel::Public},
            {MLIRHelper::TupleFieldName("keys", builder.getContext()), getOpaqueType(), false, mlir_ts::AccessLevel::Public},
            {MLIRHelper::TupleFieldName("entries", builder.getContext()), getOpaqueType(), false, mlir_ts::AccessLevel::Public}};
        return mlir::cast<mlir_ts::TupleType>(getTupleType(headerFields));
    }

    void MLIRGenImpl::makeMemberTables(const GenContext &genContext)
    {
        // an imported module read as declarations makes no functions: the tables are made at the end of the
        // module that imports it
        if (declarationMode)
        {
            return;
        }

        // the keys adapters of the classes read by name here: a table names its class's (TYPE_DESCR_KEYS). Making
        // one can box another class (a field's), which asks for its table: again until none is new. Not of a class
        // only exported: its adapter would keep all its methods linked (a library's DLL grew by a third to a half)
        // for an importer reading an object of it held as a base by key, which reads it by the base's keys instead
        // (the table's keys are null)
        llvm::SmallPtrSet<ClassInfo *, 8> ensured;
        for (auto changed = true; changed;)
        {
            changed = false;
            for (auto &[location, classInfo] : membersTableClasses)
            {
                if (isReadByName(classInfo) && ensured.insert(classInfo.get()).second)
                {
                    ensureAnyKeysAdapter(location, classInfo->classType, genContext);
                    changed = true;
                }
            }

            makePendingAnyKeysAdapters(genContext);
        }

        for (auto &[location, classInfo] : membersTableClasses)
        {
            makeMemberTable(location, classInfo, genContext);
        }

        membersTableClasses.clear();
    }

    // `expected`: the table of a class of a library this module imports, as this module lays it out
    // (<class>.members_expected), to check against the library's own when it loads (mlirGenImportSharedLib)
    void MLIRGenImpl::makeMemberTable(mlir::Location location, ClassInfo::TypePtr classInfo, const GenContext &genContext,
                                      bool expected)
    {
        auto tableName = concat(classInfo->fullName, expected ? MEMBERS_EXPECTED_NAME : MEMBERS_NAME);
        auto entriesTableName = concat(classInfo->fullName, expected ? MEMBERS_EXPECTED_ENTRIES_NAME : MEMBERS_ENTRIES_NAME);
        if (theModule.lookupSymbol(tableName))
        {
            return;
        }

        mlir::OpBuilder::InsertionGuard insertionGuard(builder);
        builder.setInsertionPointToEnd(theModule.getBody());

        SmallVector<MemberTableEntry> entries;
        if (expected || isMemberTableRequested(classInfo))
        {
            collectMemberTableEntries(classInfo, entries);
        }

        // inside the class: its protected members are read as its own code reads them
        GenContext tableGenContext(genContext);
        tableGenContext.thisClassType = classInfo->classType;

        auto i32Type = builder.getI32Type();
        auto i64Type = builder.getI64Type();
        auto indexType = builder.getIntegerType(compileOptions.sizeBits());
        auto i32 = [&](int value) -> mlir::Value {
            return builder.create<mlir_ts::ConstantOp>(location, i32Type, builder.getI32IntegerAttr(value));
        };
        auto nullOpaque = [&]() -> mlir::Value {
            auto nullValue = builder.create<mlir_ts::NullOp>(location, getNullType());
            return builder.create<mlir_ts::CastOp>(location, getOpaqueType(), nullValue);
        };

        SmallVector<mlir_ts::FieldInfo> entryTypeFields{
            {MLIRHelper::TupleFieldName("name", builder.getContext()), getStringType(), false, mlir_ts::AccessLevel::Public},
            {MLIRHelper::TupleFieldName("kind", builder.getContext()), i32Type, false, mlir_ts::AccessLevel::Public},
            {MLIRHelper::TupleFieldName("access", builder.getContext()), i32Type, false, mlir_ts::AccessLevel::Public},
            {MLIRHelper::TupleFieldName("readonly", builder.getContext()), i32Type, false, mlir_ts::AccessLevel::Public},
            {MLIRHelper::TupleFieldName("offset", builder.getContext()), indexType, false, mlir_ts::AccessLevel::Public},
            {MLIRHelper::TupleFieldName("size", builder.getContext()), indexType, false, mlir_ts::AccessLevel::Public},
            {MLIRHelper::TupleFieldName("hash", builder.getContext()), i64Type, false, mlir_ts::AccessLevel::Public}};
        auto entryType = getTupleType(entryTypeFields);

        VariableClass varClass = VariableType::Var;
        varClass.isExport = classInfo->isExport && !expected;
        varClass.isPublic = classInfo->isPublic && !expected;

        // the entries
        mlir::Type entriesType;
        if (!entries.empty())
        {
            SmallVector<mlir_ts::FieldInfo> entryFields;
            for (auto [index, entry] : llvm::enumerate(entries))
            {
                entryFields.push_back({MLIRHelper::TupleFieldName(std::to_string(index), builder.getContext()), entryType,
                                       false, mlir_ts::AccessLevel::Public});
            }

            entriesType = getTupleType(entryFields);
            // only the header is looked up from another module
            VariableClass entriesVarClass = VariableType::Var;
            registerVariable(
                location, entriesTableName, true, entriesVarClass,
                [&](mlir::Location location, const GenContext &) {
                    MLIRCodeLogic mcl(builder, compileOptions);
                    mlir::Value entriesValue = builder.create<mlir_ts::UndefOp>(location, entriesType);
                    for (auto [index, entry] : llvm::enumerate(entries))
                    {
                        // a field's offset, &(null)->field, as an interface's vtable has it; -1 for the rest
                        mlir::Value offset = builder.create<mlir_ts::ConstantOp>(
                            location, indexType, builder.getIntegerAttr(indexType, -1));
                        mlir::Value size = builder.create<mlir_ts::ConstantOp>(
                            location, indexType, builder.getIntegerAttr(indexType, 0));
                        if (entry.kind == MEMBER_KIND_FIELD)
                        {
                            auto nullObj = builder.create<mlir_ts::NullOp>(location, getNullType());
                            auto classNull = cast(location, classInfo->classType, nullObj, tableGenContext, true);
                            auto fieldValue = mlirGenPropertyAccessExpression(location, classNull, getStringAttr(entry.name),
                                                                              tableGenContext);
                            if (!fieldValue.failed_or_no_value())
                            {
                                if (auto fieldRef = mcl.GetReferenceFromValue(location, slotOfFieldRead(V(fieldValue))))
                                {
                                    offset = builder.create<mlir_ts::CastOp>(location, indexType, fieldRef);
                                }
                            }

                            auto sizeOf = builder.create<mlir_ts::SizeOfOp>(location, builder.getIndexType(), entry.type);
                            size = builder.create<mlir_ts::CastOp>(location, indexType, sizeOf);
                        }

                        auto hash = entry.type ? MLIRHelper::hashOfType(memberHashType(entry.type)) : 0;
                        mlir::Value entryValue = builder.create<mlir_ts::UndefOp>(location, entryType);
                        SmallVector<mlir::Value> values{
                            V(mlirGenStringValue(location, entry.name, true)), i32(entry.kind), i32(entry.access),
                            i32(entry.readOnly ? 1 : 0), offset, size,
                            builder.create<mlir_ts::ConstantOp>(location, i64Type, builder.getI64IntegerAttr((int64_t)hash))};
                        for (auto [fieldIndex, value] : llvm::enumerate(values))
                        {
                            entryValue = builder.create<mlir_ts::InsertPropertyOp>(
                                location, entryType, value, entryValue, MLIRHelper::getStructIndex(builder, fieldIndex));
                        }

                        entriesValue = builder.create<mlir_ts::InsertPropertyOp>(
                            location, entriesType, entryValue, entriesValue, MLIRHelper::getStructIndex(builder, index));
                    }

                    return TypeValueInitType{entriesType, entriesValue, TypeProvided::Yes};
                },
                tableGenContext);
        }

        // the header
        auto headerType = getMemberTableHeaderType();

        registerVariable(
            location, tableName, true, varClass,
            [&](mlir::Location location, const GenContext &) {
                // the class's keys adapter, when one was made (makeAnyKeysAdapter)
                mlir::Value keys = nullOpaque();
                if (!entries.empty() && !expected)
                {
                    auto hash = MLIRHelper::hashOfType(TypeOfOpHelper::typeOfBaseType(classInfo->classType));
                    if (auto adapters = theModule->getAttrOfType<mlir::DictionaryAttr>(ANY_KEYS_ADAPTERS_ATTR_NAME))
                    {
                        if (auto adapterName = dyn_cast_or_null<mlir::StringAttr>(adapters.get(std::to_string(hash))))
                        {
                            if (auto adapterOp = theModule.lookupSymbol<mlir_ts::FuncOp>(adapterName.getValue()))
                            {
                                auto symbolRefOp = builder.create<mlir_ts::SymbolRefOp>(
                                    location, adapterOp.getFunctionType(),
                                    mlir::FlatSymbolRefAttr::get(builder.getContext(), adapterName.getValue()));
                                keys = builder.create<mlir_ts::CastOp>(location, getOpaqueType(), symbolRefOp);
                            }
                        }
                    }
                }

                mlir::Value entriesValue = nullOpaque();
                if (!entries.empty())
                {
                    auto entriesAddress = builder.create<mlir_ts::AddressOfOp>(
                        location, mlir_ts::RefType::get(entriesType), entriesTableName,
                        ::mlir::IntegerAttr());
                    entriesValue = builder.create<mlir_ts::CastOp>(location, getOpaqueType(), entriesAddress);
                }

                mlir::Value headerValue = builder.create<mlir_ts::UndefOp>(location, headerType);
                SmallVector<mlir::Value> values{i32((int)entries.size()), i32(0), keys, entriesValue};
                for (auto [fieldIndex, value] : llvm::enumerate(values))
                {
                    headerValue = builder.create<mlir_ts::InsertPropertyOp>(
                        location, headerType, value, headerValue, MLIRHelper::getStructIndex(builder, fieldIndex));
                }

                return TypeValueInitType{headerType, headerValue, TypeProvided::Yes};
            },
            tableGenContext);
    }

    // `c[k]` with `k` a union of string literals (`keyof C`, `"a" | "b"`) known only at run time: the names
    // it may be, each a member of the class
    bool MLIRGenImpl::classKeyNames(mlir::Value object, mlir::Type keyType, SmallVector<std::string> &names,
                                    const GenContext &genContext)
    {
        SmallVector<mlir::Type> literals;
        if (!stringLiteralsOf(keyType, literals))
        {
            return false;
        }

        for (auto literal : literals)
        {
            auto name = mlir::cast<mlir::StringAttr>(mlir::cast<mlir_ts::LiteralType>(literal).getValue()).getValue();
            if (!evaluateProperty(object.getLoc(), object, name.str(), genContext))
            {
                return false;
            }

            names.push_back(name.str());
        }

        return !names.empty();
    }

    // A switch over the names the key may be: `k == "a" ? c.a : k == "b" ? c.b : ...`, each read as `c.name`
    // reads it (a method bound to the object, its override through the vtable), the result the union of their
    // types. The read is remembered (classKeyReads): a write to it is the same switch of stores.
    ValueOrLogicalResult MLIRGenImpl::mlirGenClassKeyRead(mlir::Location location, mlir::Value object, mlir::Value key,
                                                          ArrayRef<std::string> names, const GenContext &genContext)
    {
        // under own a method read off its object takes the object with it (a bound method moves it)
        if (compileOptions.memoryModel == MemoryModelOwn)
        {
            SmallVector<std::string> values;
            SmallVector<std::string> methods;
            anyKeysOfType(object.getType(), values, methods);
            for (auto &name : names)
            {
                if (llvm::is_contained(methods, name))
                {
                    emitError(location) << "reading a method through a key moves the object under -mm=own; the key set includes '"
                                        << name << "'";
                    return mlir::failure();
                }
            }
        }

        SmallVector<mlir::Type> types;
        for (auto &name : names)
        {
            types.push_back(evaluateProperty(location, object, name, genContext));
        }

        auto resultType = types.size() == 1 ? types.front() : mth.getUnionTypeWithMerge(location, types);

        CAST_A(keyString, location, getStringType(), key, genContext);

        auto readOf = [&](StringRef name) -> ValueOrLogicalResult {
            auto value = mlirGenPropertyAccessExpression(location, object, getStringAttr(name), false, genContext);
            EXIT_IF_FAILED_OR_NO_VALUE(value)
            if (V(value).getType() == resultType)
            {
                return value;
            }

            return cast(location, resultType, V(value), genContext);
        };

        std::function<ValueOrLogicalResult(size_t)> readFrom = [&](size_t index) -> ValueOrLogicalResult {
            if (index + 1 == names.size())
            {
                return readOf(names[index]);
            }

            auto nameValue = V(mlirGenStringValue(location, names[index], true));
            auto isName = builder.create<mlir_ts::LogicalBinaryOp>(
                location, getBooleanType(), builder.getI32IntegerAttr((int)SyntaxKind::EqualsEqualsToken), keyString, nameValue);
            auto ifOp = builder.create<mlir_ts::IfOp>(location, mlir::TypeRange{resultType}, isName, true);

            builder.setInsertionPointToStart(&ifOp.getThenRegion().front());
            auto thenValue = readOf(names[index]);
            EXIT_IF_FAILED_OR_NO_VALUE(thenValue)
            builder.create<mlir_ts::ResultOp>(location, mlir::ValueRange{V(thenValue)});

            builder.setInsertionPointToStart(&ifOp.getElseRegion().front());
            auto elseValue = readFrom(index + 1);
            EXIT_IF_FAILED_OR_NO_VALUE(elseValue)
            builder.create<mlir_ts::ResultOp>(location, mlir::ValueRange{V(elseValue)});

            builder.setInsertionPointAfter(ifOp);
            return V(ifOp.getResults().front());
        };

        auto result = readFrom(0);
        EXIT_IF_FAILED_OR_NO_VALUE(result)
        if (auto definingOp = V(result).getDefiningOp())
        {
            definingOp->setAttr(CLASS_KEY_READ_ATTR_NAME, builder.getUnitAttr());
            classKeyReads[V(result)] = {object, keyString, SmallVector<std::string>(names.begin(), names.end())};
        }

        return result;
    }

    // `c[k] = v` of a read mlirGenClassKeyRead made: the same switch, of stores. The value must be one every
    // named field can hold (tsc: the intersection of their types), and no name a method.
    ValueOrLogicalResult MLIRGenImpl::mlirGenClassKeyWrite(mlir::Location location, mlir::Value object, mlir::Value keyString,
                                                           ArrayRef<std::string> names, mlir::Value value,
                                                           const GenContext &genContext)
    {
        SmallVector<std::string> values;
        SmallVector<std::string> methods;
        anyKeysOfType(object.getType(), values, methods);
        for (auto &name : names)
        {
            if (llvm::is_contained(methods, name))
            {
                emitError(location) << "a method can't be written through a key: the key set includes '" << name << "'";
                return mlir::failure();
            }

            auto fieldType = evaluateProperty(location, object, name, genContext);
            if (mlir::failed(checkStringNumberAssignment(location, value.getType(), fieldType, genContext)))
            {
                return mlir::failure();
            }
        }

        auto storeTo = [&](StringRef name) -> mlir::LogicalResult {
            auto place = mlirGenPropertyAccessExpression(location, object, getStringAttr(name), false, genContext);
            EXIT_IF_FAILED_OR_NO_VALUE(place)
            auto stored = mlirGenSaveLogicOneItem(location, V(place), value, genContext);
            EXIT_IF_FAILED(stored)
            return mlir::success();
        };

        std::function<mlir::LogicalResult(size_t)> storeFrom = [&](size_t index) -> mlir::LogicalResult {
            if (index + 1 == names.size())
            {
                return storeTo(names[index]);
            }

            auto nameValue = V(mlirGenStringValue(location, names[index], true));
            auto isName = builder.create<mlir_ts::LogicalBinaryOp>(
                location, getBooleanType(), builder.getI32IntegerAttr((int)SyntaxKind::EqualsEqualsToken), keyString, nameValue);
            auto ifOp = builder.create<mlir_ts::IfOp>(location, mlir::TypeRange{}, isName, true);

            // the branches have their terminators already: the stores go before them
            builder.setInsertionPoint(ifOp.getThenRegion().front().getTerminator());
            if (mlir::failed(storeTo(names[index])))
            {
                return mlir::failure();
            }

            builder.setInsertionPoint(ifOp.getElseRegion().front().getTerminator());
            if (mlir::failed(storeFrom(index + 1)))
            {
                return mlir::failure();
            }

            builder.setInsertionPointAfter(ifOp);
            return mlir::success();
        };

        if (mlir::failed(storeFrom(0)))
        {
            return mlir::failure();
        }

        return value;
    }

} // namespace mlirgen
} // namespace typescript
