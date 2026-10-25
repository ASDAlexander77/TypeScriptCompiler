#include "TsClangImporter/HeaderParser.h"
#include "TsClangImporter/TypeMapper.h"
#include "TypeScript/CppMangler.h"

#include "clang/AST/ASTConsumer.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/DeclTemplate.h"
#include "clang/AST/GlobalDecl.h"
#include "clang/AST/Mangle.h"
#include "clang/AST/RecordLayout.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/AST/VTableBuilder.h"
#include "clang/Basic/TargetInfo.h"
#include "clang/Basic/DiagnosticOptions.h"
#include "clang/Basic/FileManager.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/FrontendAction.h"
#include "clang/Frontend/TextDiagnosticPrinter.h"
#include "clang/Lex/LiteralSupport.h"
#include "clang/Lex/MacroInfo.h"
#include "clang/Lex/PPCallbacks.h"
#include "clang/Lex/Preprocessor.h"
#include "clang/Sema/SemaConsumer.h"
#include "clang/Tooling/Tooling.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/VirtualFileSystem.h"

#include <optional>

namespace tsbindgen
{

namespace
{

std::string escapeString(llvm::StringRef text)
{
    std::string escaped = "\"";
    for (unsigned char c : text)
    {
        switch (c)
        {
        case '\\':
            escaped += "\\\\";
            break;
        case '"':
            escaped += "\\\"";
            break;
        case '\n':
            escaped += "\\n";
            break;
        case '\r':
            escaped += "\\r";
            break;
        case '\t':
            escaped += "\\t";
            break;
        default:
            if (c < 0x20 || c == 0x7f)
            {
                static const char hex[] = "0123456789abcdef";
                escaped += "\\x";
                escaped += hex[c >> 4];
                escaped += hex[c & 0xf];
            }
            else
            {
                escaped += static_cast<char>(c);
            }
        }
    }

    return escaped + "\"";
}

// The name of a record or enum with its namespaces and enclosing classes (`ns::Outer::Inner`), as
// @cppname reads it. An inline namespace is spelled too: both schemes mangle it. Nothing for what
// @cppname can't name: an anonymous namespace or record, a template's specialization or a member of
// one, a function's local type.
std::optional<std::string> cppQualifiedName(const clang::NamedDecl *decl)
{
    std::vector<std::string> parts;
    for (const clang::DeclContext *context = llvm::cast<clang::DeclContext>(decl); !context->isTranslationUnit();
         context = context->getParent())
    {
        if (llvm::isa<clang::LinkageSpecDecl>(context))
        {
            continue; // extern "C++" { ... }
        }

        auto *named = llvm::dyn_cast<clang::NamedDecl>(context);
        if (!named || !named->getIdentifier() || llvm::isa<clang::ClassTemplateSpecializationDecl>(context) ||
            !(llvm::isa<clang::NamespaceDecl, clang::RecordDecl, clang::EnumDecl>(context)))
        {
            return std::nullopt;
        }

        parts.insert(parts.begin(), named->getName().str());
    }

    return llvm::join(parts, "::");
}

std::optional<std::string> cppBuiltinName(const clang::BuiltinType *builtin)
{
    switch (builtin->getKind())
    {
    case clang::BuiltinType::Void:
        return "void";
    case clang::BuiltinType::Bool:
        return "bool";
    case clang::BuiltinType::Char_S:
    case clang::BuiltinType::Char_U:
        return "char";
    case clang::BuiltinType::SChar:
        return "signed char";
    case clang::BuiltinType::UChar:
        return "unsigned char";
    case clang::BuiltinType::WChar_S:
    case clang::BuiltinType::WChar_U:
        return "wchar_t";
    case clang::BuiltinType::Char8:
        return "char8_t";
    case clang::BuiltinType::Char16:
        return "char16_t";
    case clang::BuiltinType::Char32:
        return "char32_t";
    case clang::BuiltinType::Short:
        return "short";
    case clang::BuiltinType::UShort:
        return "unsigned short";
    case clang::BuiltinType::Int:
        return "int";
    case clang::BuiltinType::UInt:
        return "unsigned int";
    case clang::BuiltinType::Long:
        return "long";
    case clang::BuiltinType::ULong:
        return "unsigned long";
    case clang::BuiltinType::LongLong:
        return "long long";
    case clang::BuiltinType::ULongLong:
        return "unsigned long long";
    case clang::BuiltinType::Float:
        return "float";
    case clang::BuiltinType::Double:
        return "double";
    case clang::BuiltinType::LongDouble:
        return "long double";
    default:
        return std::nullopt; // __int128, _Float16, nullptr_t, ...
    }
}

// `type` as @cppname reads it (CppMangler.h): a builtin, a <stdint.h>/<stddef.h> typedef as written
// - it mangles as whatever the target @cppname is compiled for defines it as - or a record or enum by
// its qualified name and tag (`class X`, `union U`, `enum E`; a bare name is a struct), with
// `const`/`volatile`, `*`, `&` and `&&`. Nothing for anything else: an array, a function pointer, a
// template's specialization, ...
std::optional<std::string> cppTypeSpelling(clang::QualType type)
{
    clang::Qualifiers qualifiers;
    std::string base;
    for (;;)
    {
        qualifiers.addQualifiers(type.getLocalQualifiers());
        auto *bare = type.getTypePtr();
        if (auto *typedefType = llvm::dyn_cast<clang::TypedefType>(bare))
        {
            auto name = typedefType->getDecl()->getName();
            if (llvm::is_contained({"int8_t", "uint8_t", "int16_t", "uint16_t", "int32_t", "uint32_t", "int64_t",
                                    "uint64_t", "size_t", "ptrdiff_t", "intptr_t", "uintptr_t"},
                                   name))
            {
                base = name.str();
                break;
            }
        }

        auto desugared = bare->getLocallyUnqualifiedSingleStepDesugaredType();
        if (desugared.getTypePtr() == bare && !desugared.hasLocalQualifiers())
        {
            break;
        }

        type = desugared;
    }

    if (qualifiers.hasRestrict() || qualifiers.hasAddressSpace() || qualifiers.hasObjCLifetime() ||
        qualifiers.hasUnaligned())
    {
        return std::nullopt;
    }

    auto cv = std::string(qualifiers.hasConst() ? "const" : "") +
              (qualifiers.hasConst() && qualifiers.hasVolatile() ? " " : "") +
              (qualifiers.hasVolatile() ? "volatile" : "");
    auto *bare = type.getTypePtr();
    if (base.empty())
    {
        auto pointer = [&](clang::QualType pointee, const char *declarator) -> std::optional<std::string> {
            auto inner = cppTypeSpelling(pointee);
            if (!inner)
            {
                return std::nullopt;
            }

            return *inner + declarator + (cv.empty() ? "" : " " + cv);
        };

        if (auto *pointerType = llvm::dyn_cast<clang::PointerType>(bare))
        {
            return pointer(pointerType->getPointeeType(), "*");
        }

        if (auto *reference = llvm::dyn_cast<clang::LValueReferenceType>(bare))
        {
            return pointer(reference->getPointeeType(), "&");
        }

        if (auto *reference = llvm::dyn_cast<clang::RValueReferenceType>(bare))
        {
            return pointer(reference->getPointeeType(), "&&");
        }

        if (auto *builtin = llvm::dyn_cast<clang::BuiltinType>(bare))
        {
            auto name = cppBuiltinName(builtin);
            if (!name)
            {
                return std::nullopt;
            }

            base = *name;
        }
        else if (auto *tag = llvm::dyn_cast<clang::TagType>(bare))
        {
            auto *decl = tag->getDecl();
            auto name = cppQualifiedName(decl);
            if (!name)
            {
                return std::nullopt;
            }

            switch (decl->getTagKind())
            {
            case clang::TagTypeKind::Struct:
                base = *name;
                break;
            case clang::TagTypeKind::Class:
                base = "class " + *name;
                break;
            case clang::TagTypeKind::Union:
                base = "union " + *name;
                break;
            case clang::TagTypeKind::Enum:
                base = "enum " + *name;
                break;
            default:
                return std::nullopt; // __interface
            }
        }
        else
        {
            return std::nullopt;
        }
    }

    return cv.empty() ? base : cv + " " + base;
}

// A method's declaration as @cppname takes it, its name qualified with its class's (the TS class
// can be named otherwise: --strip-prefix, a namespace, ...), or nothing when a type can't be spelled
std::optional<std::string> cppMethodDeclaration(const clang::CXXMethodDecl *method)
{
    auto *prototype = method->getType()->getAs<clang::FunctionProtoType>();
    auto className = cppQualifiedName(method->getParent());
    auto result = cppTypeSpelling(method->getReturnType());
    if (!prototype || !className || !result || method->getRefQualifier() != clang::RQ_None)
    {
        return std::nullopt;
    }

    std::string parameters;
    for (auto parameter : prototype->getParamTypes())
    {
        auto spelled = cppTypeSpelling(parameter);
        if (!spelled)
        {
            return std::nullopt;
        }

        parameters += (parameters.empty() ? "" : ", ") + *spelled;
    }

    return *result + " " + *className + "::" + method->getName().str() + "(" + parameters + ")" +
           (method->isConst() ? " const" : "") + (method->isVolatile() ? " volatile" : "");
}

// An object-like macro's value as a TS literal: an integer, a float or a string, optionally
// negated and parenthesized. "" for anything else.
std::string literalValue(clang::Preprocessor &preprocessor, llvm::ArrayRef<clang::Token> tokens)
{
    while (tokens.size() >= 2 && tokens.front().is(clang::tok::l_paren) && tokens.back().is(clang::tok::r_paren))
    {
        tokens = tokens.drop_front().drop_back();
    }

    if (!tokens.empty() && llvm::all_of(tokens, [](const clang::Token &token) { return token.is(clang::tok::string_literal); }))
    {
        // no diagnostics engine: a macro body clang never expands is not clang's error
        clang::StringLiteralParser literal(tokens, preprocessor.getSourceManager(), preprocessor.getLangOpts(),
                                           preprocessor.getTargetInfo(), nullptr);
        if (literal.hadError || !literal.isOrdinary())
        {
            return "";
        }

        return escapeString(literal.GetString());
    }

    auto negative = tokens.size() == 2 && tokens.front().is(clang::tok::minus);
    if (negative)
    {
        tokens = tokens.drop_front();
    }

    if (tokens.size() != 1 || !tokens.front().is(clang::tok::numeric_constant))
    {
        return "";
    }

    llvm::SmallString<32> buffer;
    auto invalid = false;
    auto spelling = preprocessor.getSpelling(tokens.front(), buffer, &invalid);
    if (invalid)
    {
        return "";
    }

    // `#define VERSION 1.2.3` is fine C until something expands it; reported into clang's own
    // engine it would fail the whole header, so the parser gets an engine that drops everything
    clang::DiagnosticOptions quietOptions;
    clang::DiagnosticsEngine quiet(clang::DiagnosticIDs::create(), quietOptions, new clang::IgnoringDiagConsumer());
    quiet.setSourceManager(&preprocessor.getSourceManager());
    clang::NumericLiteralParser literal(spelling, tokens.front().getLocation(), preprocessor.getSourceManager(),
                                        preprocessor.getLangOpts(), preprocessor.getTargetInfo(), quiet);
    if (literal.hadError)
    {
        return "";
    }

    std::string sign = negative ? "-" : "";
    if (literal.isIntegerLiteral())
    {
        llvm::APInt value(64, 0);
        if (literal.GetIntegerValue(value))
        {
            return ""; // does not fit in 64 bits
        }

        llvm::SmallString<32> text;
        value.toString(text, 10, /*Signed=*/false);
        return sign + text.str().str();
    }

    if (literal.isFloatingLiteral())
    {
        llvm::APFloat value(llvm::APFloat::IEEEdouble());
        literal.GetFloatValue(value, llvm::RoundingMode::NearestTiesToEven);
        llvm::SmallString<32> text;
        value.toString(text);
        // "2" would be an integer literal in TS
        if (text.find_first_of(".eE") == llvm::StringRef::npos)
        {
            text += ".0";
        }

        return sign + text.str().str();
    }

    return "";
}

class MacroCollector : public clang::PPCallbacks
{
  public:
    MacroCollector(clang::Preprocessor &preprocessor, std::vector<Decl> &macros)
        : preprocessor(preprocessor), macros(macros)
    {
    }

    void MacroDefined(const clang::Token &nameToken, const clang::MacroDirective *directive) override
    {
        auto *info = directive->getMacroInfo();
        auto location = info->getDefinitionLoc();
        auto &sourceManager = preprocessor.getSourceManager();
        if (info->isBuiltinMacro() || location.isInvalid() || sourceManager.isWrittenInBuiltinFile(location) ||
            sourceManager.isWrittenInCommandLineFile(location))
        {
            return;
        }

        auto name = nameToken.getIdentifierInfo()->getName().str();

        Decl macro;
        macro.kind = DeclKind::Macro;
        macro.key = "macro:" + name;
        macro.name = name;
        macro.inMainFile = sourceManager.isInMainFile(location);
        macro.inSystemHeader = sourceManager.isInSystemHeader(location);
        if (info->isFunctionLike())
        {
            macro.skipReason = "function-like macro";
        }
        else if (info->getNumTokens() == 0)
        {
            forget(name); // an include guard or a feature switch: nothing to bind
            return;
        }
        else
        {
            macro.value = literalValue(preprocessor, info->tokens());
            if (macro.value.empty())
            {
                macro.skipReason = "macro is not a literal";
            }
        }

        forget(name);
        macros.push_back(std::move(macro));
    }

    void MacroUndefined(const clang::Token &nameToken, const clang::MacroDefinition &,
                        const clang::MacroDirective *) override
    {
        forget(nameToken.getIdentifierInfo()->getName().str());
    }

  private:
    void forget(const std::string &name)
    {
        llvm::erase_if(macros, [&](const Decl &macro) { return macro.name == name; });
    }

    clang::Preprocessor &preprocessor;
    std::vector<Decl> &macros;
};

class Collector : public clang::RecursiveASTVisitor<Collector>
{
  public:
    Collector(clang::ASTContext &context, HeaderModel &model, clang::Sema *sema)
        : context(context), mapper(context), model(model)
    {
        mapper.setSema(sema);
    }

    bool VisitRecordDecl(clang::RecordDecl *record)
    {
        if (record->isImplicit())
        {
            return true;
        }

        // a polymorphic C++ class, in any namespace: a @cpp class
        if (auto *cxxRecord = llvm::dyn_cast<clang::CXXRecordDecl>(record);
            cxxRecord && cxxRecord->getDefinition() && mapper.classSkipReason(cxxRecord).empty())
        {
            addClass(cxxRecord->getDefinition());
            return true;
        }

        // the mapper spells a pointer to anything else as plain Opaque, so nothing names it
        if (!mapper.isDeclarable(record))
        {
            return true;
        }

        if (TypeMapper::tagName(record).empty())
        {
            return true; // only reachable as a member's type, which skips that member's struct
        }

        auto key = TypeMapper::recordKey(record);
        if (!keys.insert(key).second)
        {
            return true;
        }

        // completed below, once the whole file is seen: its definition may come later. A C++ struct
        // in a namespace or a class is declared under its own name; --filter matches either.
        pendingRecords.push_back({model.decls.size(), record});
        auto &decl = add(DeclKind::OpaqueStruct, key, TypeMapper::tagName(record), record);
        decl.instantiation = mapper.isInstantiated(record);
        auto qualified = TypeMapper::qualifiedTagName(record);
        if (qualified != decl.name)
        {
            decl.qualifiedName = qualified;
        }

        return true;
    }

    bool VisitEnumDecl(clang::EnumDecl *enumDecl)
    {
        auto *definition = enumDecl->getDefinition();
        if (definition != enumDecl || !atFileScope(enumDecl))
        {
            return true;
        }

        auto name = TypeMapper::tagName(enumDecl);
        if (name.empty())
        {
            std::string listed;
            for (auto *enumerator : enumDecl->enumerators())
            {
                listed += (listed.empty() ? "" : ", ") + enumerator->getName().str();
            }

            auto &decl = add(DeclKind::Enum, "enum:{" + listed + "}", "enum { " + listed + " }", enumDecl);
            decl.skipReason = "anonymous enum";
            return true;
        }

        auto key = TypeMapper::enumKey(enumDecl);
        if (!keys.insert(key).second)
        {
            return true;
        }

        auto plain = mapper.isPlainEnum(enumDecl);
        auto &decl = add(plain ? DeclKind::Enum : DeclKind::FixedEnum, key, name, enumDecl);
        decl.scoped = enumDecl->isScoped();
        if (!plain)
        {
            decl.type = mapper.map(enumDecl->getIntegerType(), TypeMapper::Use::Field);
            decl.skipReason = decl.type.skipReason;
        }

        for (auto *enumerator : enumDecl->enumerators())
        {
            llvm::SmallString<32> value;
            enumerator->getInitVal().toString(value, 10);
            decl.enumerators.push_back({enumerator->getName().str(), value.str().str()});
        }

        return true;
    }

    bool VisitTypedefNameDecl(clang::TypedefNameDecl *typedefDecl)
    {
        if (isSystem(typedefDecl) || !atFileScope(typedefDecl) || TypeMapper::namesItsTag(typedefDecl) ||
            isTsBuiltinTypeName(typedefDecl->getName().str()))
        {
            return true; // the mapper looks through all of these
        }

        auto name = typedefDecl->getName();
        if (name == "size_t" || name == "ssize_t" || name == "ptrdiff_t" || name == "intptr_t" ||
            name == "uintptr_t")
        {
            return true;
        }

        auto key = TypeMapper::typedefKey(typedefDecl);
        if (!keys.insert(key).second)
        {
            return true;
        }

        auto &decl = add(DeclKind::Typedef, key, name.str(), typedefDecl);
        decl.type = mapper.map(typedefDecl->getUnderlyingType(), TypeMapper::Use::Alias);
        decl.skipReason = decl.type.skipReason;
        return true;
    }

    bool VisitFunctionDecl(clang::FunctionDecl *function)
    {
        if (llvm::isa<clang::CXXMethodDecl>(function) || function->isImplicit() ||
            function->getTemplatedKind() != clang::FunctionDecl::TK_NonTemplate || !atFileScope(function))
        {
            return true;
        }

        // C++ proper is v2: only what has C linkage has a C symbol. An operator has no name to
        // declare, C linkage or not (guiddef.h's operator== for GUID).
        if (context.getLangOpts().CPlusPlus && (!function->isExternC() || !function->getIdentifier()))
        {
            return true;
        }

        auto name = function->getName().str();
        auto key = "fn:" + name;
        if (!keys.insert(key).second)
        {
            return true;
        }

        auto &decl = add(DeclKind::Function, key, name, function);
        if (function->getStorageClass() == clang::SC_Static)
        {
            decl.skipReason = function->isInlineSpecified() ? "static inline function: no symbol to link against"
                                                            : "static function: no symbol to link against";
            return true;
        }

        auto *prototype = function->getType()->getAs<clang::FunctionProtoType>();
        if (!prototype)
        {
            decl.skipReason = "function without a prototype";
            return true;
        }

        // tslang calls a declaration with the C convention; a callee that cleans its own stack
        // (__stdcall on 32-bit Windows) or takes arguments elsewhere would be called wrongly
        if (prototype->getCallConv() != clang::CC_C)
        {
            decl.skipReason = "calling convention " + clang::FunctionType::getNameForCallConv(prototype->getCallConv()).str();
            return true;
        }

        // `int f(int) __asm__("real_f")` (glibc's __REDIRECT): the symbol is the label
        if (auto *asmLabel = function->getAttr<clang::AsmLabelAttr>())
        {
            llvm::StringRef label = asmLabel->getLabel();
            label.consume_front("\x01"); // the "use this name verbatim" marker
            decl.symbol = label.str();
        }

        decl.varargs = prototype->isVariadic();
        for (unsigned index = 0; index < function->getNumParams(); ++index)
        {
            auto *parameter = function->getParamDecl(index);
            auto parameterName = parameter->getName().empty() ? "p" + std::to_string(index)
                                                              : escapeReserved(parameter->getName().str());
            auto type = mapper.map(parameter->getType(), TypeMapper::Use::Parameter);
            if (type.skipped())
            {
                decl.skipReason = "parameter '" + parameterName + "': " + type.skipReason;
                return true;
            }

            decl.fields.push_back({parameterName, type});
        }

        decl.type = mapper.map(function->getReturnType(), TypeMapper::Use::Result);
        if (decl.type.skipped())
        {
            decl.skipReason = "result: " + decl.type.skipReason;
        }

        return true;
    }

    // After the traversal: complete the structs (a definition may follow the first mention), and
    // add every struct or enum a mapped type names that the traversal did not reach, until both
    // stop changing - completing a struct maps its fields, which can name more.
    void finish()
    {
        size_t completed = 0;
        while (true)
        {
            std::vector<const clang::TagDecl *> missing;
            for (auto &[key, tag] : mapper.referencedTags())
            {
                if (!keys.count(key))
                {
                    missing.push_back(tag);
                }
            }

            for (auto *tag : missing)
            {
                if (auto *record = llvm::dyn_cast<clang::RecordDecl>(tag))
                {
                    VisitRecordDecl(const_cast<clang::RecordDecl *>(record));
                }
                else if (auto *enumDecl = llvm::cast<clang::EnumDecl>(tag)->getDefinition())
                {
                    VisitEnumDecl(enumDecl);
                }
            }

            // the generic tuples the specializations mapped so far are instances of
            for (auto &generic : mapper.generics())
            {
                if (keys.insert(generic.key).second)
                {
                    auto &decl = add(DeclKind::Struct, generic.key, generic.name, generic.at);
                    decl.qualifiedName = generic.qualifiedName;
                    decl.typeParameters = generic.parameters;
                    decl.fields = generic.fields;
                }
            }

            if (completed == pendingRecords.size())
            {
                break;
            }

            for (; completed < pendingRecords.size(); ++completed)
            {
                completeRecord(pendingRecords[completed].first, pendingRecords[completed].second);
            }
        }
    }

  private:
    // A @cpp class: its base, its fields when they can be declared, and the methods it introduces.
    // A method overriding a base's virtual method isn't one: the @cpp class's override keeps the
    // base's slot by name. Destructors, and methods that aren't public, are slots left out (a @cpp
    // class may skip slots).
    void addClass(const clang::CXXRecordDecl *definition)
    {
        auto key = TypeMapper::classKey(definition);
        if (!keys.insert(key).second)
        {
            return;
        }

        Decl decl;
        decl.kind = DeclKind::CppClass;
        decl.key = key;
        decl.name = definition->getName().str();
        decl.qualifiedName = definition->getQualifiedNameAsString();
        locate(decl, definition);

        const clang::CXXRecordDecl *base = nullptr;
        if (definition->getNumBases() == 1)
        {
            base = definition->bases_begin()->getType()->getAsCXXRecordDecl()->getDefinition();
            addClass(base); // declared before its use, which C++ guarantees but a referenced class may not be
            decl.base = TypeMapper::classKey(base);
        }

        llvm::StringSet<> inherited;
        llvm::StringSet<> inheritedMethods;
        if (base)
        {
            inherited = classNames[decl.base];
            inheritedMethods = classMethodNames[decl.base];
        }

        auto &usedNames = classNames[key];
        usedNames = std::move(inherited);
        auto &methodNames = classMethodNames[key];
        methodNames = std::move(inheritedMethods);

        addClassFields(decl, definition, base);
        for (auto &field : decl.fields)
        {
            usedNames.insert(field.name);
        }

        for (auto *method : definition->methods())
        {
            if (method->isImplicit() || method->isDeleted() || method->isStatic() ||
                llvm::isa<clang::CXXConstructorDecl, clang::CXXDestructorDecl, clang::CXXConversionDecl>(method) ||
                method->isOverloadedOperator() || method->getDescribedFunctionTemplate() ||
                method->getAccess() != clang::AS_public ||
                (method->isVirtual() && method->size_overridden_methods() > 0))
            {
                continue;
            }

            // C++ overloads keep the C++ name, each with its own @slot or @linkname: tslang picks one
            // by the arguments. A field of the name makes it `_2`, `_3`, ...
            Method bound;
            bound.cppName = method->getName().str();
            bound.name = escapeReserved(bound.cppName);
            for (auto suffix = 2; usedNames.count(bound.name) && !methodNames.count(bound.name); ++suffix)
            {
                bound.name = escapeReserved(bound.cppName) + "_" + std::to_string(suffix);
            }

            usedNames.insert(bound.name);
            methodNames.insert(bound.name);
            mapMethod(bound, method);
            decl.methods.push_back(std::move(bound));
        }

        model.decls.push_back(std::move(decl));
    }

    // A class's own fields, after the vtable pointer and its base's fields, as a @cpp class lays
    // them out: like a tuple's, the base's storage first, padded to its alignment. Left out (and so
    // in every class derived from it) when C++ puts them elsewhere - the Itanium ABI puts a derived
    // class's fields in its base's tail padding - or one can't be mapped.
    void addClassFields(Decl &decl, const clang::CXXRecordDecl *definition, const clang::CXXRecordDecl *base)
    {
        uint64_t offset = context.getTargetInfo().getPointerWidth(clang::LangAS::Default);
        uint64_t maxAlign = context.getTargetInfo().getPointerAlign(clang::LangAS::Default);
        if (base)
        {
            auto found = fieldsEnd.find(base);
            if (found == fieldsEnd.end())
            {
                decl.fieldsSkipReason = "the fields of the base class aren't declared";
                return;
            }

            offset = found->second.first;
            maxAlign = found->second.second;
        }

        const auto &layout = context.getASTRecordLayout(definition);
        std::vector<Field> fields;
        unsigned index = 0;
        for (auto *field : definition->fields())
        {
            auto fieldName = field->getName().str();
            auto reason = [&]() -> std::string {
                if (field->isBitField() || fieldName.empty())
                {
                    return "a bitfield or an anonymous member";
                }

                auto type = mapper.map(field->getType(), TypeMapper::Use::Field);
                if (type.skipped())
                {
                    return "field '" + fieldName + "': " + type.skipReason;
                }

                auto align = context.getTypeAlign(field->getType());
                offset = llvm::alignTo(offset, align);
                if (layout.getFieldOffset(index) != offset)
                {
                    return "field '" + fieldName + "' isn't where a tuple would put it";
                }

                offset += context.getTypeSize(field->getType());
                maxAlign = std::max<uint64_t>(maxAlign, align);
                fields.push_back({escapeReserved(fieldName), type});
                return "";
            }();

            if (!reason.empty())
            {
                decl.fieldsSkipReason = reason;
                return;
            }

            ++index;
        }

        decl.fields = std::move(fields);
        fieldsEnd[definition] = {llvm::alignTo(offset, maxAlign), maxAlign};
    }

    void mapMethod(Method &bound, const clang::CXXMethodDecl *method)
    {
        if (method->isVariadic())
        {
            bound.skipReason = "variadic method";
            return;
        }

        if (method->isVirtual())
        {
            auto *vtables = context.getVTableContext();
            if (auto *microsoft = llvm::dyn_cast<clang::MicrosoftVTableContext>(vtables))
            {
                auto location = microsoft->getMethodVFTableLocation(clang::GlobalDecl(method));
                if (location.VBase || !location.VFPtrOffset.isZero())
                {
                    bound.skipReason = "its vtable isn't the object's first";
                    return;
                }

                bound.slot = static_cast<int>(location.Index);
            }
            else
            {
                bound.slot = static_cast<int>(
                    llvm::cast<clang::ItaniumVTableContext>(vtables)->getMethodVTableIndex(clang::GlobalDecl(method)));
            }
        }
        else
        {
            // defined in the class (or `inline`): no symbol unless something made one
            if (method->isInlined())
            {
                bound.skipReason = "inline method: no symbol to link against";
                return;
            }

            std::unique_ptr<clang::MangleContext> mangler(context.createMangleContext());
            llvm::raw_string_ostream symbol(bound.symbol);
            mangler->mangleName(clang::GlobalDecl(method), symbol);
            symbol.flush();
        }

        for (unsigned index = 0; index < method->getNumParams(); ++index)
        {
            auto *parameter = method->getParamDecl(index);
            auto parameterName = parameter->getName().empty() ? "p" + std::to_string(index)
                                                              : escapeReserved(parameter->getName().str());
            // a record by value, which a virtual method's call passes as the C++ ABI does
            auto byValueRecord = parameter->getType().getCanonicalType()->getAsRecordDecl() != nullptr;
            if (byValueRecord && !method->isVirtual())
            {
                bound.skipReason = "parameter '" + parameterName + "': a non-virtual method taking a record by value isn't supported yet";
                return;
            }

            auto type = byValueRecord ? mapper.mapRecordParameter(parameter->getType())
                                      : mapper.map(parameter->getType(), TypeMapper::Use::Parameter);
            if (type.skipped())
            {
                bound.skipReason = "parameter '" + parameterName + "': " + type.skipReason;
                return;
            }

            bound.parameters.push_back({parameterName, type});
        }

        auto resultType = method->getReturnType();
        if (auto *record = resultType.getCanonicalType()->getAsRecordDecl())
        {
            if (!method->isVirtual())
            {
                bound.skipReason = "a non-virtual method returning a record by value isn't supported yet";
                return;
            }

            bound.result = mapper.mapRecordResult(resultType, bound.nonTrivialReturn);
        }
        else
        {
            bound.result = mapper.map(resultType, TypeMapper::Use::Result);
        }

        if (bound.result.skipped())
        {
            bound.skipReason = "result: " + bound.result.skipReason;
            return;
        }

        // bound by its declaration when @cppname, mangling it for this target, gives clang's symbol:
        // the same binding then links on every other target too
        if (!method->isVirtual())
        {
            if (auto declaration = cppMethodDeclaration(method))
            {
                CppMangleContext mangleContext;
                mangleContext.isInstanceMethod = true;
                std::string symbol;
                std::string error;
                if (mangleCppName(*declaration, context.getTargetInfo().getTriple(), mangleContext, symbol, error) &&
                    symbol == bound.symbol)
                {
                    bound.cppDeclaration = std::move(*declaration);
                }
            }
        }
    }

    void completeRecord(size_t index, const clang::RecordDecl *record)
    {
        auto *definition = record->getDefinition();
        if (!definition)
        {
            return; // stays OpaqueStruct
        }

        auto &decl = model.decls[index];
        locate(decl, definition);
        decl.kind = DeclKind::Struct;
        decl.skipReason = mapper.recordSkipReason(definition);
        if (!decl.skipReason.empty())
        {
            return;
        }

        decl.fields = mapper.mapFields(definition);
    }

    bool atFileScope(const clang::Decl *decl)
    {
        // in C, a struct declared inside another is still file-scoped
        return !context.getLangOpts().CPlusPlus || decl->getDeclContext()->getRedeclContext()->isTranslationUnit();
    }

    bool isSystem(const clang::Decl *decl)
    {
        auto &sourceManager = context.getSourceManager();
        return sourceManager.isInSystemHeader(sourceManager.getExpansionLoc(decl->getLocation()));
    }

    void locate(Decl &decl, const clang::Decl *at)
    {
        auto &sourceManager = context.getSourceManager();
        auto location = sourceManager.getExpansionLoc(at->getLocation());
        decl.inMainFile = sourceManager.isInMainFile(location);
        decl.inSystemHeader = sourceManager.isInSystemHeader(location);
    }

    Decl &add(DeclKind kind, std::string key, std::string name, const clang::Decl *at)
    {
        Decl decl;
        decl.kind = kind;
        decl.key = std::move(key);
        decl.name = std::move(name);
        locate(decl, at);
        model.decls.push_back(std::move(decl));
        return model.decls.back();
    }

    clang::ASTContext &context;
    TypeMapper mapper;
    HeaderModel &model;
    llvm::StringSet<> keys;
    std::vector<std::pair<size_t, const clang::RecordDecl *>> pendingRecords;
    // a @cpp class's TS member names, its base's included: a method named as a field gets `_2`, ...
    llvm::StringMap<llvm::StringSet<>> classNames;
    // those that are methods: an overload keeps the name
    llvm::StringMap<llvm::StringSet<>> classMethodNames;
    // where a @cpp class's declared fields end, padded to its alignment, and that alignment (bits):
    // a derived class's fields follow
    llvm::DenseMap<const clang::CXXRecordDecl *, std::pair<uint64_t, uint64_t>> fieldsEnd;
};

class ImportConsumer : public clang::SemaConsumer
{
  public:
    explicit ImportConsumer(HeaderModel &model) : model(model)
    {
    }

    void InitializeSema(clang::Sema &value) override
    {
        sema = &value;
    }

    void ForgetSema() override
    {
        sema = nullptr;
    }

    void HandleTranslationUnit(clang::ASTContext &context) override
    {
        Collector collector(context, model, sema);
        collector.TraverseDecl(context.getTranslationUnitDecl());
        collector.finish();
    }

  private:
    HeaderModel &model;
    clang::Sema *sema = nullptr;
};

class ImportAction : public clang::ASTFrontendAction
{
  public:
    ImportAction(HeaderModel &model, std::vector<Decl> &macros) : model(model), macros(macros)
    {
    }

    bool BeginSourceFileAction(clang::CompilerInstance &compiler) override
    {
        auto &preprocessor = compiler.getPreprocessor();
        preprocessor.addPPCallbacks(std::make_unique<MacroCollector>(preprocessor, macros));
        return true;
    }

    std::unique_ptr<clang::ASTConsumer> CreateASTConsumer(clang::CompilerInstance &, llvm::StringRef) override
    {
        return std::make_unique<ImportConsumer>(model);
    }

  private:
    HeaderModel &model;
    std::vector<Decl> &macros;
};

} // namespace

llvm::Expected<HeaderModel> parseHeader(const ParseInput &input)
{
    // What clang::tooling::runToolOnCodeWithArgs does, but with our own diagnostic consumer: an
    // error in the arguments (`-- --bad-flag`) goes to a diagnostics engine whose errors that
    // function never checks, and the run would "succeed" on a command line clang rejected.
    auto overlay = llvm::makeIntrusiveRefCnt<llvm::vfs::OverlayFileSystem>(llvm::vfs::getRealFileSystem());
    auto memory = llvm::makeIntrusiveRefCnt<llvm::vfs::InMemoryFileSystem>();
    overlay->pushOverlay(memory);
    memory->addFile(input.fileName, 0, llvm::MemoryBuffer::getMemBufferCopy(input.code, input.fileName));
    for (auto &[path, content] : input.virtualFiles)
    {
        memory->addFile(path, 0, llvm::MemoryBuffer::getMemBufferCopy(content, path));
    }

    auto files = llvm::makeIntrusiveRefCnt<clang::FileManager>(clang::FileSystemOptions(), overlay);

    std::vector<std::string> commandLine = {"tsbindgen", "-fsyntax-only"};
    commandLine.insert(commandLine.end(), input.args.begin(), input.args.end());
    commandLine.push_back(input.fileName);

    HeaderModel model;
    std::vector<Decl> macros;
    clang::tooling::ToolInvocation invocation(commandLine, std::make_unique<ImportAction>(model, macros), files.get());

    clang::DiagnosticOptions diagnosticOptions;
    clang::TextDiagnosticPrinter diagnostics(llvm::errs(), diagnosticOptions);
    invocation.setDiagnosticConsumer(&diagnostics);

    if (!invocation.run() || diagnostics.getNumErrors() > 0)
    {
        return llvm::createStringError("clang could not parse '" + input.fileName + "'");
    }

    model.decls.insert(model.decls.begin(), macros.begin(), macros.end());
    return model;
}

} // namespace tsbindgen
