#include "TsClangImporter/HeaderParser.h"
#include "TsClangImporter/TypeMapper.h"

#include "clang/AST/ASTConsumer.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/Basic/DiagnosticOptions.h"
#include "clang/Basic/FileManager.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/FrontendAction.h"
#include "clang/Frontend/TextDiagnosticPrinter.h"
#include "clang/Lex/LiteralSupport.h"
#include "clang/Lex/MacroInfo.h"
#include "clang/Lex/PPCallbacks.h"
#include "clang/Lex/Preprocessor.h"
#include "clang/Tooling/Tooling.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/VirtualFileSystem.h"

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
    Collector(clang::ASTContext &context, HeaderModel &model) : context(context), mapper(context), model(model)
    {
    }

    bool VisitRecordDecl(clang::RecordDecl *record)
    {
        // the mapper spells a pointer to anything else as plain Opaque, so nothing names it
        if (record->isImplicit() || !mapper.isDeclarable(record))
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

        // completed below, once the whole file is seen: its definition may come later
        pendingRecords.push_back({model.decls.size(), record});
        add(DeclKind::OpaqueStruct, key, TypeMapper::tagName(record), record);
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

        // C++ proper is v2: only what has C linkage has a C symbol
        if (context.getLangOpts().CPlusPlus && !function->isExternC())
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
};

class ImportConsumer : public clang::ASTConsumer
{
  public:
    explicit ImportConsumer(HeaderModel &model) : model(model)
    {
    }

    void HandleTranslationUnit(clang::ASTContext &context) override
    {
        Collector collector(context, model);
        collector.TraverseDecl(context.getTranslationUnitDecl());
        collector.finish();
    }

  private:
    HeaderModel &model;
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
