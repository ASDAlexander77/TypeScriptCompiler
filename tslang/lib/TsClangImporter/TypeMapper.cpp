#include "TsClangImporter/TypeMapper.h"

#include "clang/AST/DeclCXX.h"
#include "clang/AST/RecordLayout.h"
#include "clang/Basic/SourceManager.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/Support/MathExtras.h"

namespace tsbindgen
{

namespace
{

bool isIndexTypedef(llvm::StringRef name)
{
    return name == "size_t" || name == "ssize_t" || name == "ptrdiff_t" || name == "intptr_t" ||
           name == "uintptr_t";
}

bool isPlainChar(clang::QualType canonical)
{
    return canonical->isSpecificBuiltinType(clang::BuiltinType::Char_S) ||
           canonical->isSpecificBuiltinType(clang::BuiltinType::Char_U);
}

} // namespace

std::string TypeMapper::tagName(const clang::TagDecl *tag)
{
    if (!tag->getName().empty())
    {
        return tag->getName().str();
    }

    if (auto *typedefDecl = tag->getTypedefNameForAnonDecl())
    {
        return typedefDecl->getName().str();
    }

    return "";
}

std::string TypeMapper::recordKey(const clang::RecordDecl *record)
{
    return "struct:" + tagName(record);
}

std::string TypeMapper::enumKey(const clang::EnumDecl *enumDecl)
{
    return "enum:" + tagName(enumDecl);
}

std::string TypeMapper::typedefKey(const clang::TypedefNameDecl *typedefDecl)
{
    return "typedef:" + typedefDecl->getName().str();
}

bool TypeMapper::namesItsTag(const clang::TypedefNameDecl *typedefDecl)
{
    auto *tag = typedefDecl->getUnderlyingType().getCanonicalType()->getAsTagDecl();
    if (!tag)
    {
        return false;
    }

    if (auto *anonName = tag->getTypedefNameForAnonDecl())
    {
        return anonName->getCanonicalDecl() == typedefDecl->getCanonicalDecl();
    }

    return tag->getName() == typedefDecl->getName();
}

bool TypeMapper::isPlainEnum(const clang::EnumDecl *enumDecl)
{
    auto integer = enumDecl->getIntegerType();
    if (integer.isNull())
    {
        return true;
    }

    auto canonical = integer.getCanonicalType();
    return context.getIntWidth(canonical) == 32 &&
           (!enumDecl->isFixed() || canonical->isSpecificBuiltinType(clang::BuiltinType::Int));
}

std::string TypeMapper::referTo(const clang::TagDecl *tag)
{
    auto key = llvm::isa<clang::EnumDecl>(tag) ? enumKey(llvm::cast<clang::EnumDecl>(tag))
                                                : recordKey(llvm::cast<clang::RecordDecl>(tag));
    if (referencedKeys.insert(key).second)
    {
        referenced.push_back({key, tag});
    }

    return key;
}

bool TypeMapper::isSystem(const clang::Decl *decl)
{
    auto &sourceManager = context.getSourceManager();
    return sourceManager.isInSystemHeader(sourceManager.getExpansionLoc(decl->getLocation()));
}

TsType TypeMapper::map(clang::QualType type, Use use)
{
    type = type.getUnqualifiedType();

    // an array or function parameter, adjusted to a pointer
    if (auto *decayed = type->getAs<clang::DecayedType>())
    {
        return map(decayed->getDecayedType(), use);
    }

    if (auto *typedefType = type->getAs<clang::TypedefType>())
    {
        auto *typedefDecl = typedefType->getDecl();
        if (isIndexTypedef(typedefDecl->getName()))
        {
            return TsType::builtin("index");
        }

        // a system header's typedefs (int32_t, uint8_t, ...) are looked through: their widths are
        // what matter, and the header is not emitted
        if (isSystem(typedefDecl))
        {
            return map(typedefDecl->getUnderlyingType(), use);
        }

        auto aliased = map(typedefDecl->getUnderlyingType(), use);
        if (aliased.skipped() || namesItsTag(typedefDecl) || leadsToOwner(typedefDecl->getUnderlyingType()))
        {
            return aliased;
        }

        return TsType::named(typedefKey(typedefDecl));
    }

    auto canonical = type.getCanonicalType().getUnqualifiedType();
    if (canonical->isVoidType())
    {
        return use == Use::Result ? TsType::builtin("void") : TsType::skip("void");
    }

    if (auto *pointer = type->getAs<clang::PointerType>())
    {
        return mapPointer(pointer, use);
    }

    if (canonical->isBuiltinType())
    {
        return mapBuiltin(canonical, use);
    }

    if (auto *enumType = canonical->getAs<clang::EnumType>())
    {
        auto *enumDecl = enumType->getDecl()->getDefinitionOrSelf();
        if (tagName(enumDecl).empty())
        {
            // an anonymous enum's values are plain integers of its underlying type
            return mapBuiltin(enumDecl->getIntegerType().getCanonicalType(), use);
        }

        return TsType::named(referTo(enumDecl));
    }

    if (auto *record = canonical->getAsRecordDecl())
    {
        if (use == Use::Parameter || use == Use::Result)
        {
            return TsType::skip(record->isUnion() ? "union passed by value" : "struct passed by value");
        }

        auto name = tagName(record);
        if (name.empty())
        {
            return TsType::skip("anonymous struct");
        }

        if (use == Use::Alias)
        {
            return TsType::named(referTo(record));
        }

        auto *definition = record->getDefinition();
        if (!definition)
        {
            return TsType::skip("incomplete struct '" + name + "'");
        }

        auto reason = recordSkipReason(definition);
        if (!reason.empty())
        {
            return TsType::skip("'" + name + "' is skipped (" + reason + ")");
        }

        return TsType::named(referTo(record));
    }

    if (canonical->isArrayType())
    {
        return TsType::skip("array");
    }

    if (canonical->isFunctionType())
    {
        return TsType::skip("function type");
    }

    return TsType::skip("unsupported type '" + type.getAsString() + "'");
}

TsType TypeMapper::mapBuiltin(clang::QualType canonical, Use use)
{
    auto *builtin = canonical->castAs<clang::BuiltinType>();
    switch (builtin->getKind())
    {
    case clang::BuiltinType::Bool:
        return TsType::builtin("boolean");
    case clang::BuiltinType::Float:
        return TsType::builtin("f32");
    case clang::BuiltinType::Double:
        return TsType::builtin("f64");
    case clang::BuiltinType::LongDouble:
        return TsType::skip("long double");
    default:
        break;
    }

    if (canonical->isIntegerType())
    {
        auto width = context.getIntWidth(canonical);
        if (width == 8 || width == 16 || width == 32 || width == 64)
        {
            return TsType::builtin((canonical->isSignedIntegerType() ? "s" : "u") + std::to_string(width));
        }

        return TsType::skip(std::to_string(width) + "-bit integer");
    }

    return TsType::skip("unsupported type '" + canonical.getAsString() + "'");
}

TsType TypeMapper::mapPointer(const clang::PointerType *pointer, Use use)
{
    auto pointee = pointer->getPointeeType();
    auto canonicalPointee = pointee.getCanonicalType().getUnqualifiedType();

    if (isPlainChar(canonicalPointee))
    {
        return TsType::builtin("string");
    }

    if (canonicalPointee->isVoidType())
    {
        return TsType::builtin("Opaque");
    }

    if (canonicalPointee->isFunctionType())
    {
        auto *function = pointee->getAs<clang::FunctionProtoType>();
        auto type = TsType::builtin("Opaque", function ? signatureComment(function) : "(...) => ?");
        type.functionPointer = true;
        return type;
    }

    if (auto *record = canonicalPointee->getAsRecordDecl())
    {
        if (tagName(record).empty())
        {
            return TsType::skip("pointer to an anonymous struct");
        }

        if (leadsToOwner(pointee))
        {
            return TsType::builtin("Opaque", "Reference<" + tagName(record) + ">: a type cannot refer to itself");
        }

        // incomplete, a C++ class, or a struct that is itself skipped: the struct is emitted as
        // `type S = Opaque;`, and a pointer to it is that S
        auto *definition = record->getDefinition();
        if (!definition || !recordSkipReason(definition).empty())
        {
            return TsType::named(referTo(record));
        }

        return TsType::reference(TsType::named(referTo(record)));
    }

    auto inner = map(pointee, Use::Field);
    if (inner.skipped())
    {
        return TsType::skip("pointer to " + inner.skipReason);
    }

    return TsType::reference(inner);
}

std::string TypeMapper::signatureComment(const clang::FunctionProtoType *function)
{
    std::string text = "(";
    for (unsigned index = 0; index < function->getNumParams(); ++index)
    {
        auto type = map(function->getParamType(index), Use::Parameter);
        text += (index ? ", p" : "p") + std::to_string(index) + ": " +
                (type.skipped() ? "?" : renderType(type, nameFromKey));
    }

    if (function->isVariadic())
    {
        text += function->getNumParams() ? ", ..." : "...";
    }

    auto result = map(function->getReturnType(), Use::Result);
    return text + ") => " + (result.skipped() ? "?" : renderType(result, nameFromKey));
}

std::string TypeMapper::recordSkipReason(const clang::RecordDecl *record)
{
    auto found = recordReasons.find(record);
    if (found != recordReasons.end())
    {
        return found->second;
    }

    // a struct that points to itself is mappable while this runs
    recordReasons[record] = "";
    owners.push_back(record);

    auto reason = [&]() -> std::string {
        if (record->isUnion())
        {
            return "union";
        }

        if (auto *cxxRecord = llvm::dyn_cast<clang::CXXRecordDecl>(record); cxxRecord && !cxxRecord->isCLike())
        {
            return "C++ class";
        }

        if (record->field_empty())
        {
            return "empty struct";
        }

        // a named tuple is laid out with natural alignment, field after field; anything else would
        // put the fields where C does not
        const auto &layout = context.getASTRecordLayout(record);
        uint64_t offset = 0;
        uint64_t maxAlign = 8;
        unsigned index = 0;
        for (auto *field : record->fields())
        {
            auto fieldName = field->getName().str();
            if (field->isBitField())
            {
                return "bitfield '" + fieldName + "'";
            }

            if (fieldName.empty())
            {
                return "anonymous member";
            }

            auto type = map(field->getType(), Use::Field);
            if (type.skipped())
            {
                return "field '" + fieldName + "': " + type.skipReason;
            }

            auto align = context.getTypeAlign(field->getType());
            offset = llvm::alignTo(offset, align);
            if (layout.getFieldOffset(index) != offset)
            {
                return "packed or over-aligned layout";
            }

            offset += context.getTypeSize(field->getType());
            maxAlign = std::max<uint64_t>(maxAlign, align);
            ++index;
        }

        if (llvm::alignTo(offset, maxAlign) != static_cast<uint64_t>(context.toBits(layout.getSize())))
        {
            return "packed or over-aligned layout";
        }

        return "";
    }();

    owners.pop_back();
    recordReasons[record] = reason;
    return reason;
}

std::vector<Field> TypeMapper::mapFields(const clang::RecordDecl *definition)
{
    owners.push_back(definition);
    std::vector<Field> fields;
    for (auto *field : definition->fields())
    {
        fields.push_back({escapeReserved(field->getName().str()), map(field->getType(), Use::Field)});
    }

    owners.pop_back();
    return fields;
}

const clang::RecordDecl *TypeMapper::recordBehind(clang::QualType type)
{
    auto canonical = type.getCanonicalType();
    while (true)
    {
        if (auto *pointer = canonical->getAs<clang::PointerType>())
        {
            canonical = pointer->getPointeeType().getCanonicalType();
        }
        else if (auto *array = context.getAsArrayType(canonical))
        {
            canonical = array->getElementType().getCanonicalType();
        }
        else
        {
            break;
        }
    }

    auto *record = canonical->getAsRecordDecl();
    return record ? record->getDefinition() : nullptr;
}

// Whether `type` reaches the struct whose fields are being mapped, through any chain of fields.
bool TypeMapper::leadsToOwner(clang::QualType type)
{
    if (owners.empty())
    {
        return false;
    }

    llvm::SmallPtrSet<const clang::RecordDecl *, 8> visited;
    std::vector<const clang::RecordDecl *> work;
    if (auto *record = recordBehind(type))
    {
        work.push_back(record);
    }

    while (!work.empty())
    {
        auto *record = work.back();
        work.pop_back();
        if (record == owners.back())
        {
            return true;
        }

        if (!visited.insert(record).second)
        {
            continue;
        }

        for (auto *field : record->fields())
        {
            if (auto *next = recordBehind(field->getType()))
            {
                work.push_back(next);
            }
        }
    }

    return false;
}

} // namespace tsbindgen
