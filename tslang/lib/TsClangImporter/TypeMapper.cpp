#include "TsClangImporter/TypeMapper.h"

#include "clang/AST/DeclCXX.h"
#include "clang/AST/DeclTemplate.h"
#include "clang/AST/RecordLayout.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Basic/TargetInfo.h"
#include "clang/Sema/Sema.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/MathExtras.h"

#include <cctype>

namespace tsbindgen
{

namespace
{

bool isIndexTypedef(llvm::StringRef name)
{
    return name == "size_t" || name == "ssize_t" || name == "ptrdiff_t" || name == "intptr_t" ||
           name == "uintptr_t";
}

// an array bigger than this is skipped rather than spelled as that many tuple fields
constexpr uint64_t maxArrayElements = 1024;

bool isPlainChar(clang::QualType canonical)
{
    return canonical->isSpecificBuiltinType(clang::BuiltinType::Char_S) ||
           canonical->isSpecificBuiltinType(clang::BuiltinType::Char_U);
}

} // namespace

namespace
{

// The class is a template's specialization, or inside one (`basic_string<char>::__rep`)
bool inInstantiation(const clang::TagDecl *tag)
{
    for (const clang::DeclContext *context = tag; context; context = context->getParent())
    {
        if (llvm::isa<clang::ClassTemplateSpecializationDecl>(context))
        {
            return true;
        }
    }

    return false;
}

// `static_vector<IBindingSet *, 8>`, `basic_string<char>::__rep`: the classes from the outermost
// enclosing one down, each with its template arguments (the defaulted ones left out); their
// namespaces too with `qualified`
std::string instantiationText(const clang::TagDecl *tag, bool qualified)
{
    auto policy = tag->getASTContext().getPrintingPolicy();
    policy.SuppressScope = !qualified;
    policy.SuppressTagKeyword = true;

    std::vector<const clang::TagDecl *> chain;
    const clang::DeclContext *context = tag;
    for (; context && llvm::isa<clang::TagDecl>(context); context = context->getParent())
    {
        chain.insert(chain.begin(), llvm::cast<clang::TagDecl>(context));
    }

    std::string text;
    if (qualified && context)
    {
        if (auto *scope = llvm::dyn_cast<clang::NamedDecl>(context))
        {
            text = scope->getQualifiedNameAsString();
        }
    }

    for (auto *part : chain)
    {
        text += (text.empty() ? "" : "::") + part->getName().str();
        if (auto *specialization = llvm::dyn_cast<clang::ClassTemplateSpecializationDecl>(part))
        {
            std::string arguments;
            llvm::raw_string_ostream stream(arguments);
            clang::printTemplateArgumentList(stream, specialization->getTemplateArgs().asArray(), policy,
                                             specialization->getSpecializedTemplate()->getTemplateParameters());
            stream.flush();
            // SuppressScope doesn't reach the arguments' types: drop each `scope::` there too
            if (!qualified)
            {
                std::string unscoped;
                for (size_t index = 0; index < arguments.size(); ++index)
                {
                    if (arguments.compare(index, 2, "::") == 0)
                    {
                        while (!unscoped.empty() && (std::isalnum(static_cast<unsigned char>(unscoped.back())) ||
                                                     unscoped.back() == '_'))
                        {
                            unscoped.pop_back();
                        }

                        ++index;
                        continue;
                    }

                    unscoped += arguments[index];
                }

                arguments = unscoped;
            }

            text += arguments;
        }
    }

    return text;
}

// a name of the identifier's characters: `static_vector<IBindingSet *, 8>` -> `static_vector_IBindingSet_8`
std::string identifierOf(const std::string &text)
{
    std::string name;
    auto separated = false;
    for (unsigned char c : text)
    {
        if (std::isalnum(c) || c == '_')
        {
            name += static_cast<char>(c);
            separated = false;
        }
        else if (!separated && !name.empty())
        {
            name += '_';
            separated = true;
        }
    }

    if (separated)
    {
        name.pop_back();
    }

    return name;
}

} // namespace

std::string TypeMapper::tagName(const clang::TagDecl *tag)
{
    if (inInstantiation(tag) && !tag->getName().empty())
    {
        return identifierOf(instantiationText(tag, false));
    }

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

std::string TypeMapper::qualifiedTagName(const clang::TagDecl *tag)
{
    if (inInstantiation(tag) && !tag->getName().empty())
    {
        return instantiationText(tag, true);
    }

    if (!llvm::isa<clang::CXXRecordDecl>(tag) || tag->getDeclContext()->getRedeclContext()->isTranslationUnit())
    {
        return tagName(tag);
    }

    if (!tag->getName().empty())
    {
        return tag->getQualifiedNameAsString();
    }

    if (auto *typedefDecl = tag->getTypedefNameForAnonDecl())
    {
        return typedefDecl->getQualifiedNameAsString();
    }

    return "";
}

std::string TypeMapper::recordKey(const clang::RecordDecl *record)
{
    return "struct:" + qualifiedTagName(record);
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

std::string TypeMapper::classKey(const clang::CXXRecordDecl *record)
{
    return "class:" + record->getQualifiedNameAsString();
}

std::string TypeMapper::classSkipReason(const clang::CXXRecordDecl *record)
{
    auto *definition = record->getDefinition();
    if (!definition)
    {
        return "incomplete class";
    }

    auto found = classReasons.find(definition);
    if (found != classReasons.end())
    {
        return found->second;
    }

    auto reason = [&]() -> std::string {
        if (!definition->isPolymorphic())
        {
            return "not polymorphic";
        }

        if (definition->getDescribedClassTemplate() || llvm::isa<clang::ClassTemplateSpecializationDecl>(definition) ||
            definition->isDependentContext())
        {
            return "class template";
        }

        if (definition->getName().empty() || definition->isLocalClass())
        {
            return "anonymous or local class";
        }

        if (context.getTargetInfo().getPointerWidth(clang::LangAS::Default) != 64)
        {
            return "a 32-bit target";
        }

        if (definition->getNumVBases() > 0)
        {
            return "virtual base class";
        }

        if (definition->getNumBases() > 1)
        {
            return "multiple inheritance";
        }

        if (definition->getNumBases() == 1)
        {
            auto *base = definition->bases_begin()->getType()->getAsCXXRecordDecl();
            if (!base)
            {
                return "base class is a template parameter";
            }

            auto baseReason = classSkipReason(base);
            if (!baseReason.empty())
            {
                return "base class '" + base->getQualifiedNameAsString() + "': " + baseReason;
            }
        }

        return "";
    }();

    classReasons[definition] = reason;
    return reason;
}

// Whether a type holds floating-point values: itself, a record's fields (its bases' too), an
// array's elements
static bool hasFloatingPoint(const clang::ASTContext &context, clang::QualType type)
{
    auto canonical = type.getCanonicalType();
    if (canonical->isRealFloatingType())
    {
        return true;
    }

    if (auto *array = context.getAsArrayType(canonical))
    {
        return hasFloatingPoint(context, array->getElementType());
    }

    auto *record = canonical->getAsRecordDecl();
    if (!record || !record->getDefinition())
    {
        return false;
    }

    auto *definition = record->getDefinition();
    if (auto *cxxRecord = llvm::dyn_cast<clang::CXXRecordDecl>(definition))
    {
        for (auto &base : cxxRecord->bases())
        {
            if (hasFloatingPoint(context, base.getType()))
            {
                return true;
            }
        }
    }

    return llvm::any_of(definition->fields(), [&](const clang::FieldDecl *field) { return hasFloatingPoint(context, field->getType()); });
}

// The targets whose C++ ABI tslang passes records by value with, and returns them in registers
// (TargetInfo::cppRecordAbi): MSVC x64, the Itanium ABI on x86-64 and AArch64
static bool hasCppRecordAbi(const clang::ASTContext &context)
{
    auto &target = context.getTargetInfo();
    auto arch = target.getTriple().getArch();
    return target.getCXXABI().isMicrosoft() ? arch == llvm::Triple::x86_64
                                            : arch == llvm::Triple::x86_64 || arch == llvm::Triple::aarch64;
}

TsType TypeMapper::mapRecordParameter(clang::QualType type)
{
    auto *record = type.getCanonicalType()->getAsRecordDecl();
    if (record->isUnion())
    {
        return TsType::skip("union passed by value");
    }

    // a specialization (std::optional<T>) is instantiated for its layout
    if (!record->getDefinition() && sema)
    {
        sema->isCompleteType(clang::SourceLocation(), type.getCanonicalType());
    }

    auto *definition = record->getDefinition();
    if (!definition)
    {
        return TsType::skip("incomplete struct passed by value");
    }

    // (MSVC passes one with a destructor in registers too, the callee destroying its copy: of a
    // RefCountPtr, a reference the caller never added)
    auto *cxxDefinition = llvm::dyn_cast<clang::CXXRecordDecl>(definition);
    if (!definition->canPassInRegisters() || (cxxDefinition && !cxxDefinition->isTriviallyCopyable()))
    {
        return TsType::skip("struct with a destructor or a copy constructor passed by value");
    }

    if (!hasCppRecordAbi(context))
    {
        return TsType::skip("struct passed by value: not done for this target's C++ ABI");
    }

    if (!context.getTargetInfo().getCXXABI().isMicrosoft() && hasFloatingPoint(context, type))
    {
        return TsType::skip("struct with floating-point fields passed by value");
    }

    return map(type, Use::Field);
}

TsType TypeMapper::mapRecordResult(clang::QualType type, bool &nonTrivial)
{
    auto *record = type.getCanonicalType()->getAsRecordDecl();
    if (!record->getDefinition() && sema)
    {
        sema->isCompleteType(clang::SourceLocation(), type.getCanonicalType());
    }

    auto *definition = record->getDefinition();
    if (!definition)
    {
        return TsType::skip("incomplete record returned by value");
    }

    auto *cxxRecord = llvm::dyn_cast<clang::CXXRecordDecl>(definition);
    nonTrivial = !definition->canPassInRegisters();
    if (definition->isUnion() || (cxxRecord && (cxxRecord->isPolymorphic() || cxxRecord->getNumBases() > 0)))
    {
        return TsType::skip("record returned by value: a union, a polymorphic or a derived class");
    }

    // the fields as a named tuple lays them out: natural alignment, field after field
    const auto &layout = context.getASTRecordLayout(definition);
    std::vector<std::string> names;
    std::vector<TsType> types;
    uint64_t offset = 0;
    uint64_t maxAlign = 8;
    unsigned index = 0;
    for (auto *field : definition->fields())
    {
        auto fieldName = field->getName().str();
        auto fieldType = field->getType().getCanonicalType();

        // a union of 8-byte members, such as nvrhi::Object's `union { uint64_t integer; void* pointer; }`:
        // one 8-byte field, named as its first member
        auto *unionRecord = fieldType->getAsRecordDecl();
        auto pointerUnion = false;
        if (unionRecord && unionRecord->isUnion() && unionRecord->getDefinition() && context.getTypeSize(fieldType) == 64)
        {
            auto members = unionRecord->getDefinition()->fields();
            auto eightBytes = !members.empty() && llvm::all_of(members, [&](const clang::FieldDecl *member) {
                auto memberType = member->getType().getCanonicalType();
                return !member->isBitField() && !member->getName().empty() &&
                       (memberType->isPointerType() || (memberType->isIntegerType() && context.getTypeSize(memberType) == 64));
            });
            if (eightBytes)
            {
                pointerUnion = true;
                if (fieldName.empty())
                {
                    fieldName = (*members.begin())->getName().str();
                }
            }
        }

        if (field->isBitField() || fieldName.empty())
        {
            return TsType::skip("record returned by value: a bitfield or an anonymous member");
        }

        TsType type;
        if (pointerUnion || fieldType->isPointerType() || fieldType->isReferenceType())
        {
            type = TsType::builtin("Opaque | null");
        }
        else
        {
            type = map(field->getType(), Use::Field);
            if (type.skipped())
            {
                return TsType::skip("record returned by value: field '" + fieldName + "': " + type.skipReason);
            }

        }

        auto align = context.getTypeAlign(field->getType());
        offset = llvm::alignTo(offset, align);
        if (layout.getFieldOffset(index) != offset)
        {
            return TsType::skip("record returned by value: packed or over-aligned layout");
        }

        offset += context.getTypeSize(field->getType());
        maxAlign = std::max<uint64_t>(maxAlign, align);
        names.push_back(escapeReserved(fieldName));
        types.push_back(type);
        ++index;
    }

    if (names.empty() || llvm::alignTo(offset, maxAlign) != static_cast<uint64_t>(context.toBits(layout.getSize())))
    {
        return TsType::skip("record returned by value: empty, packed or over-aligned");
    }

    // MSVC returns any record through a hidden pointer, and so does the Itanium ABI one it can't
    // pass in registers or of more than 16 bytes (@cppnontrivial, the pointer before `this`); a
    // trivial one of up to 16 bytes comes back in registers, which tslang reads back as the
    // record for integers and pointers, on x86-64 and AArch64 - not floating-point fields (SSE
    // registers, an HFA's)
    if (!context.getTargetInfo().getCXXABI().isMicrosoft() && !nonTrivial)
    {
        if (!hasCppRecordAbi(context))
        {
            return TsType::skip("record returned by value: not done for this target's C++ ABI");
        }

        if (context.getTypeSize(type) > 128)
        {
            nonTrivial = true;
        }
        else if (hasFloatingPoint(context, type))
        {
            return TsType::skip("trivial record with floating-point fields returned in registers");
        }
    }

    return TsType::tuple(std::move(names), std::move(types));
}

std::string TypeMapper::referTo(const clang::TagDecl *tag)
{
    auto *cxxRecord = llvm::dyn_cast<clang::CXXRecordDecl>(tag);
    auto key = llvm::isa<clang::EnumDecl>(tag)                   ? enumKey(llvm::cast<clang::EnumDecl>(tag))
               : cxxRecord && classSkipReason(cxxRecord).empty() ? classKey(cxxRecord->getDefinition())
                                                                 : recordKey(llvm::cast<clang::RecordDecl>(tag));
    if (referencedKeys.insert(key).second)
    {
        referenced.push_back({key, tag});
    }

    return key;
}

bool TypeMapper::isDeclarable(const clang::RecordDecl *record)
{
    if (!context.getLangOpts().CPlusPlus)
    {
        return true; // in C, a struct declared inside another is still file-scoped
    }

    // a template, and a class inside a function or a lambda's, have no name the bindings could
    // declare; a specialization or a class inside one is declared once a field holds it by value
    if (record->isDependentContext() || record->isInvalidDecl() || record->getParentFunctionOrMethod())
    {
        return false;
    }

    if (isInstantiated(record))
    {
        return heldByValue.count(record->getDefinition()) > 0;
    }

    auto *cxxRecord = llvm::dyn_cast<clang::CXXRecordDecl>(record);
    // (a specialization only named, never instantiated, is TSK_Undeclared, hence the isa)
    return !cxxRecord ||
           (!cxxRecord->getDescribedClassTemplate() && !llvm::isa<clang::ClassTemplateSpecializationDecl>(cxxRecord) &&
            cxxRecord->getTemplateSpecializationKind() == clang::TSK_Undeclared && !cxxRecord->isLambda());
}

bool TypeMapper::isInstantiated(const clang::RecordDecl *record)
{
    auto *definition = record->getDefinition();
    return context.getLangOpts().CPlusPlus && definition && !definition->isDependentContext() &&
           !definition->isInvalidDecl() && inInstantiation(definition) && !definition->getName().empty() &&
           (!llvm::isa<clang::ClassTemplateSpecializationDecl>(definition) ||
            llvm::cast<clang::ClassTemplateSpecializationDecl>(definition)->getSpecializationKind() !=
                clang::TSK_Undeclared);
}

bool TypeMapper::isSystem(const clang::Decl *decl)
{
    auto &sourceManager = context.getSourceManager();
    return sourceManager.isInSystemHeader(sourceManager.getExpansionLoc(decl->getLocation()));
}

TsType TypeMapper::argumentType(clang::QualType argument)
{
    if (auto *cxxRecord = argument.getCanonicalType()->getAsCXXRecordDecl();
        cxxRecord && cxxRecord->getDefinition() && classSkipReason(cxxRecord).empty())
    {
        return TsType::named(referTo(cxxRecord->getDefinition()));
    }

    return map(argument, Use::Field);
}

const clang::RecordDecl *TypeMapper::recordOfKey(const std::string &key)
{
    for (auto &[referencedKey, tag] : referenced)
    {
        if (referencedKey == key)
        {
            return llvm::dyn_cast<clang::RecordDecl>(tag);
        }
    }

    return nullptr;
}

TsType TypeMapper::abstractType(const TsType &type, const std::vector<TsType> &arguments,
                                const std::vector<std::string> &parameters, std::vector<bool> &used, int depth)
{
    // an argument, nullable or not: `T`; `T | null` where the argument itself isn't nullable (a
    // RefCountPtr<IShader>'s `IShader *`)
    for (size_t index = 0; index < arguments.size(); ++index)
    {
        auto nonNull = type;
        nonNull.nullable = arguments[index].nullable;
        if (sameType(nonNull, arguments[index]))
        {
            used[index] = true;
            auto parameter = TsType::builtin(parameters[index]);
            parameter.nullable = type.nullable && !arguments[index].nullable;
            return parameter;
        }
    }

    auto abstracted = type;
    for (auto &pointee : abstracted.pointee)
    {
        pointee = abstractType(pointee, arguments, parameters, used, depth);
    }

    for (auto &argument : abstracted.typeArguments)
    {
        argument = abstractType(argument, arguments, parameters, used, depth);
    }

    // a specialization holding an argument (MSVC's _Compressed_pair<allocator<T>, _Vector_val<...>>):
    // its fields inline, as a tuple, as it has no name of the parameters
    if (type.kind == TsType::Kind::Named && type.typeArguments.empty() && depth < 16)
    {
        auto *record = recordOfKey(type.name);
        if (record && isInstantiated(record) && recordSkipReason(record->getDefinition()).empty())
        {
            std::vector<bool> inner(arguments.size(), false);
            std::vector<std::string> names;
            std::vector<TsType> types;
            for (auto &field : mapFields(record->getDefinition()))
            {
                names.push_back(field.name);
                types.push_back(abstractType(field.type, arguments, parameters, inner, depth + 1));
            }

            if (llvm::is_contained(inner, true))
            {
                for (size_t index = 0; index < used.size(); ++index)
                {
                    used[index] = used[index] || inner[index];
                }

                auto tuple = TsType::tuple(std::move(names), std::move(types));
                tuple.nullable = type.nullable;
                return tuple;
            }
        }
    }

    return abstracted;
}

TsType TypeMapper::genericInstance(const clang::RecordDecl *definition)
{
    auto *specialization = llvm::dyn_cast<clang::ClassTemplateSpecializationDecl>(definition);
    if (!specialization)
    {
        return TsType::skip("not a template's specialization");
    }

    auto found = instances.find(definition);
    if (found != instances.end())
    {
        return found->second;
    }

    // a specialization reached again through its own arguments' layouts
    if (!generalizing.insert(definition).second)
    {
        return TsType::skip("a specialization holding itself");
    }

    auto instance = [&]() -> TsType {
        auto *classTemplate = specialization->getSpecializedTemplate();
        auto *templateParameters = classTemplate->getTemplateParameters();
        auto templateArguments = specialization->getTemplateArgs().asArray();

        // the arguments given (not a defaulted one, std::vector's allocator): the types become the
        // parameters, the integers part of the name (static_vector<T, 8>: `static_vector8`)
        std::vector<TsType> arguments;
        std::string name = classTemplate->getName().str();
        std::string values;
        for (size_t index = 0; index < templateArguments.size(); ++index)
        {
            auto &argument = templateArguments[index];
            auto *parameter = templateParameters->getParam(std::min<size_t>(index, templateParameters->size() - 1));
            if (clang::isSubstitutedDefaultArgument(context, argument, parameter, templateArguments,
                                                    templateParameters->getDepth()))
            {
                continue;
            }

            if (argument.getKind() == clang::TemplateArgument::Type)
            {
                auto type = argumentType(argument.getAsType());
                if (type.skipped())
                {
                    return TsType::skip("type argument: " + type.skipReason);
                }

                arguments.push_back(type);
            }
            else if (argument.getKind() == clang::TemplateArgument::Integral)
            {
                auto value = llvm::toString(argument.getAsIntegral(), 10);
                name += (std::isdigit(static_cast<unsigned char>(name.back())) ? "_" : "") + value;
                values += (values.empty() ? "" : ", ") + value;
            }
            else
            {
                return TsType::skip("a template argument that is neither a type nor an integer");
            }
        }

        if (arguments.empty())
        {
            return TsType::skip("no type argument");
        }

        auto reason = recordSkipReason(definition);
        if (!reason.empty())
        {
            return TsType::skip(reason);
        }

        std::vector<std::string> parameters;
        for (size_t index = 0; index < arguments.size(); ++index)
        {
            parameters.push_back(arguments.size() == 1 ? "T" : "T" + std::to_string(index + 1));
        }

        std::vector<bool> used(arguments.size(), false);
        auto fields = mapFields(definition);
        for (auto &field : fields)
        {
            field.type = abstractType(field.type, arguments, parameters, used, 0);
        }

        if (llvm::is_contained(used, false))
        {
            return TsType::skip("a type argument its layout doesn't show");
        }

        auto qualified = classTemplate->getQualifiedNameAsString();
        Generic generic{"generic:" + qualified + "<" + values + ">", name, qualified, parameters, fields, classTemplate};
        auto known = genericIndex.find(generic.key);
        if (known == genericIndex.end())
        {
            genericIndex[generic.key] = genericList.size();
            genericList.push_back(generic);
        }
        else
        {
            // another specialization's layout, in terms of the parameters, has to be this one's
            auto &other = genericList[known->second];
            auto same = other.fields.size() == fields.size() &&
                        std::equal(fields.begin(), fields.end(), other.fields.begin(),
                                   [](const Field &a, const Field &b) { return a.name == b.name && sameType(a.type, b.type); });
            if (!same)
            {
                return TsType::skip("laid out unlike " + other.name + "'s other specializations");
            }
        }

        auto type = TsType::named(generic.key);
        type.typeArguments = arguments;
        return type;
    }();

    generalizing.erase(definition);
    instances[definition] = instance;
    return instance;
}

TsType TypeMapper::map(clang::QualType type, Use use)
{
    auto mapped = mapType(type, use);
    // C lets a pointer argument or field be null (an optional out-parameter, a missing callback, a
    // desc's unused buffer); a C++ reference can't be. A result isn't marked, which would make every
    // call site check it.
    if ((use == Use::Parameter || use == Use::Field) && !mapped.skipped() && type.getCanonicalType()->isPointerType())
    {
        mapped.nullable = true;
    }

    return mapped;
}

TsType TypeMapper::mapType(clang::QualType type, Use use)
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
        // what matter, and the header is not emitted. So is one named like a tslang type
        // (`typedef int boolean;`): tslang would ignore the alias and use its own type. And so is a
        // C++ one in a namespace or a class (`nvrhi::MipLevel`), which has no declaration here.
        if (isSystem(typedefDecl) || isTsBuiltinTypeName(typedefDecl->getName().str()) ||
            (context.getLangOpts().CPlusPlus && !typedefDecl->getDeclContext()->getRedeclContext()->isTranslationUnit()))
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
        return mapPointer(pointer->getPointeeType(), use);
    }

    // a C++ reference is passed as a pointer
    if (auto *reference = type->getAs<clang::ReferenceType>())
    {
        return mapPointer(reference->getPointeeType(), use);
    }

    if (canonical->isBuiltinType())
    {
        return mapBuiltin(canonical, use);
    }

    if (auto *enumType = canonical->getAs<clang::EnumType>())
    {
        auto *enumDecl = enumType->getDecl()->getDefinitionOrSelf();
        // declared, never defined (`enum E : short;`): no enumerators to emit, only its integer
        if (!enumDecl->getDefinition())
        {
            auto integer = enumDecl->getIntegerType();
            if (integer.isNull())
            {
                return TsType::skip("incomplete enum '" + tagName(enumDecl) + "'");
            }

            return mapBuiltin(integer.getCanonicalType(), use);
        }

        // an anonymous enum's values are plain integers of its underlying type, and so are those of
        // a C++ enum in a namespace or a class, which has no declaration of its own here
        if (tagName(enumDecl).empty() ||
            (context.getLangOpts().CPlusPlus && !enumDecl->getDeclContext()->getRedeclContext()->isTranslationUnit()))
        {
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

        // a field holding a specialization (std::string, nvrhi::static_vector<T, N>) needs its layout:
        // an instance of its template's generic tuple (static_vector8<IBindingSet | null>), or, when
        // it can't be one, declared under a name of its arguments (basic_string_char)
        if (use == Use::Field && isInstantiated(record))
        {
            auto instance = genericInstance(record->getDefinition());
            if (!instance.skipped())
            {
                return instance;
            }

            heldByValue.insert(record->getDefinition());
        }

        if (!isDeclarable(record))
        {
            return TsType::skip("C++ class '" + canonical.getAsString() + "'");
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

    // `T x[N]`: a tuple of its N elements, `[_0: T, _1: T, ...]` (tslang has no fixed-size array
    // type), which lays them out as C does
    if (auto *array = context.getAsConstantArrayType(canonical); array && (use == Use::Field || use == Use::Alias))
    {
        auto count = array->getSize().getZExtValue();
        if (count == 0)
        {
            return TsType::skip("array of no elements");
        }

        if (count > maxArrayElements)
        {
            return TsType::skip("array of " + std::to_string(count) + " elements");
        }

        auto element = map(array->getElementType(), Use::Field);
        if (element.skipped())
        {
            return TsType::skip("array of " + element.skipReason);
        }

        std::vector<std::string> names;
        for (uint64_t index = 0; index < count; ++index)
        {
            names.push_back("_" + std::to_string(index));
        }

        return TsType::tuple(std::move(names), std::vector<TsType>(count, element));
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

TsType TypeMapper::mapPointer(clang::QualType pointee, Use use)
{
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
        // a C++ object of a class bound as a @cpp class: the class, whose values are pointers
        if (auto *cxxRecord = llvm::dyn_cast<clang::CXXRecordDecl>(record); cxxRecord && classSkipReason(cxxRecord).empty())
        {
            return TsType::named(referTo(cxxRecord));
        }

        // there is no declaration a name could refer to, so no name: the C++ type is a comment. A
        // specialization is declared only for a field holding it, whatever the order it's met in.
        if (isInstantiated(record) || !isDeclarable(record))
        {
            return TsType::builtin("Opaque", pointee.getUnqualifiedType().getAsString());
        }

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
    auto reason = spellRecord(record, nullptr);
    owners.pop_back();
    recordReasons[record] = reason;
    return reason;
}

std::vector<Field> TypeMapper::mapFields(const clang::RecordDecl *definition)
{
    owners.push_back(definition);
    std::vector<Field> fields;
    spellRecord(definition, &fields);
    owners.pop_back();
    return fields;
}

TsType TypeMapper::rawWords(uint64_t bytes, uint64_t align)
{
    uint64_t word = 8;
    while (word > 1 && (word > align || bytes % word != 0))
    {
        word /= 2;
    }

    std::vector<std::string> names;
    for (uint64_t index = 0; index < bytes / word; ++index)
    {
        names.push_back("_" + std::to_string(index));
    }

    auto type = TsType::builtin("u" + std::to_string(word * 8));
    return TsType::tuple(std::move(names), std::vector<TsType>(bytes / word, type));
}

TypeMapper::Slot TypeMapper::unionSlot(std::string name, const clang::RecordDecl *unionRecord, uint64_t offset)
{
    const auto &layout = context.getASTRecordLayout(unionRecord);
    auto size = static_cast<uint64_t>(context.toBits(layout.getSize()));
    auto align = static_cast<uint64_t>(context.toBits(layout.getAlignment()));

    // its first member, when that alone has the union's size and alignment (`union { T *p; uint64_t
    // n; }`); raw words otherwise (`union { char buf[16]; char *ptr; }`, a member with bitfields)
    for (auto *member : unionRecord->fields())
    {
        if (member->isZeroSize(context))
        {
            continue;
        }

        if (!member->isBitField() && !member->isAnonymousStructOrUnion() &&
            context.getTypeSize(member->getType()) == size && context.getTypeAlign(member->getType()) == align)
        {
            auto type = map(member->getType(), Use::Field);
            if (!type.skipped())
            {
                return {name.empty() ? escapeReserved(member->getName().str()) : name, type, offset, size, align};
            }
        }

        break;
    }

    return {name.empty() ? "_union" : name, rawWords(size / 8, align / 8), offset, size, align};
}

std::string TypeMapper::collectSlots(const clang::RecordDecl *definition, uint64_t base, std::vector<Slot> &slots,
                                     std::vector<uint64_t> &baseEnds)
{
    const auto &layout = context.getASTRecordLayout(definition);

    // the bases' fields first, where the layout puts each base: C++ member lookup sees them as the
    // struct's own (nvrhi::static_vector's private std::array)
    if (auto *cxxRecord = llvm::dyn_cast<clang::CXXRecordDecl>(definition))
    {
        if (cxxRecord->isPolymorphic())
        {
            return "C++ class with virtual methods";
        }

        if (cxxRecord->getNumVBases() > 0)
        {
            return "virtual base class";
        }

        for (auto &baseSpecifier : cxxRecord->bases())
        {
            auto *baseRecord = baseSpecifier.getType()->getAsCXXRecordDecl();
            if (!baseRecord || !baseRecord->getDefinition())
            {
                return "incomplete base class";
            }

            baseRecord = baseRecord->getDefinition();
            if (baseRecord->isEmpty())
            {
                continue; // no storage (allocator, std::_Compressed_pair's empty base)
            }

            auto baseOffset = base + static_cast<uint64_t>(context.toBits(layout.getBaseClassOffset(baseRecord)));
            auto reason = collectSlots(baseRecord, baseOffset, slots, baseEnds);
            if (!reason.empty())
            {
                return "base class '" + baseRecord->getQualifiedNameAsString() + "': " + reason;
            }

            // MSVC puts the next field after the base's tail padding, the Itanium ABI may put it in
            // there: spellRecord pads up to here when the next field is here
            baseEnds.push_back(baseOffset + static_cast<uint64_t>(context.toBits(context.getASTRecordLayout(baseRecord).getSize())));
        }
    }

    std::vector<const clang::FieldDecl *> fields(definition->field_begin(), definition->field_end());
    for (size_t index = 0; index < fields.size(); ++index)
    {
        auto *field = fields[index];
        auto offset = base + layout.getFieldOffset(field->getFieldIndex());
        auto fieldName = escapeReserved(field->getName().str());

        // [[no_unique_address]] on an empty class (libc++'s std::string allocator): no storage
        if (field->isZeroSize(context))
        {
            continue;
        }

        // a run of bitfields: raw words over the bytes they use, up to the next field
        if (field->isBitField())
        {
            auto start = offset / 8 * 8;
            auto next = index + 1;
            while (next < fields.size() && (fields[next]->isBitField() || fields[next]->isZeroSize(context)))
            {
                ++next;
            }

            auto end = next < fields.size() ? base + layout.getFieldOffset(fields[next]->getFieldIndex())
                                            : base + static_cast<uint64_t>(context.toBits(layout.getDataSize()));
            end = llvm::alignTo(end, 8);
            auto align = static_cast<uint64_t>(context.toBits(layout.getAlignment())) / 8;
            auto words = rawWords((end - start) / 8, align);
            // the words' alignment: their size, `u64` -> 64 bits
            auto wordBits = static_cast<uint64_t>(std::stoi(words.pointee.front().name.substr(1)));
            slots.push_back({fieldName.empty() ? "_bits" : fieldName, words, start, end - start, wordBits});
            index = next - 1;
            continue;
        }

        auto canonical = field->getType().getCanonicalType();
        if (auto *fieldRecord = canonical->getAsRecordDecl())
        {
            auto *fieldDefinition = fieldRecord->getDefinition();
            if (fieldDefinition && fieldDefinition->isUnion())
            {
                slots.push_back(unionSlot(field->isAnonymousStructOrUnion() ? "" : fieldName, fieldDefinition, offset));
                continue;
            }

            // an anonymous struct's fields are the struct's own, as in C11
            if (fieldDefinition && field->isAnonymousStructOrUnion())
            {
                auto reason = collectSlots(fieldDefinition, offset, slots, baseEnds);
                if (!reason.empty())
                {
                    return "anonymous member: " + reason;
                }

                continue;
            }
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

        slots.push_back({fieldName, type, offset, static_cast<uint64_t>(context.getTypeSize(field->getType())),
                         static_cast<uint64_t>(context.getTypeAlign(field->getType()))});
    }

    return "";
}

std::string TypeMapper::spellRecord(const clang::RecordDecl *definition, std::vector<Field> *fields)
{
    if (definition->isUnion())
    {
        return "union";
    }

    std::vector<Slot> slots;
    std::vector<uint64_t> ends;
    auto reason = collectSlots(definition, 0, slots, ends);
    if (!reason.empty())
    {
        return reason;
    }

    if (slots.empty())
    {
        return "empty struct";
    }

    // a named tuple is laid out with natural alignment, field after field; a C++ layout that puts a
    // field elsewhere (packed, over-aligned, the Itanium ABI's base tail padding) isn't one
    uint64_t offset = 0;
    uint64_t maxAlign = 8;
    std::vector<Slot> padded;
    for (auto &slot : slots)
    {
        if (slot.offset > llvm::alignTo(offset, slot.align) && llvm::is_contained(ends, slot.offset))
        {
            // words no bigger than the padding's start is aligned to
            auto start = offset / 8;
            auto words = rawWords((slot.offset - offset) / 8, start ? (start & (~start + 1)) : 8);
            auto wordBits = static_cast<uint64_t>(std::stoi(words.pointee.front().name.substr(1)));
            padded.push_back({"_pad", words, offset, slot.offset - offset, wordBits});
            offset = slot.offset;
        }

        offset = llvm::alignTo(offset, slot.align);
        if (slot.offset != offset)
        {
            return "packed or over-aligned layout";
        }

        padded.push_back(slot);
        offset += slot.size;
        maxAlign = std::max(maxAlign, slot.align);
    }

    slots = std::move(padded);

    // a tuple is aligned as its most aligned field; an alignas() struct (DECLSPEC_ALIGN(16)) more, so
    // a struct holding one would put it elsewhere
    const auto &layout = context.getASTRecordLayout(definition);
    if (llvm::alignTo(offset, maxAlign) != static_cast<uint64_t>(context.toBits(layout.getSize())) ||
        static_cast<uint64_t>(context.toBits(layout.getAlignment())) > maxAlign)
    {
        return "packed or over-aligned layout";
    }

    if (fields)
    {
        // a base's field and the struct's own may have one name: the later gets `_2`, ...
        llvm::StringSet<> names;
        for (auto &slot : slots)
        {
            auto name = slot.name;
            for (auto suffix = 2; names.count(name); ++suffix)
            {
                name = slot.name + "_" + std::to_string(suffix);
            }

            names.insert(name);
            fields->push_back({name, slot.type});
        }
    }

    return "";
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
