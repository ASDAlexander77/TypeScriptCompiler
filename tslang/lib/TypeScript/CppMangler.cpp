#include "TypeScript/CppMangler.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringSwitch.h"

#include <memory>
#include <optional>

// Hand-written rather than clang's MangleContext: tslang doesn't link clang's frontend, and a
// declaration of builtin types, records, pointers and references needs only a small part of either
// scheme. The expected symbols in unittests/MLIRGen/CppMangler.cpp come from clang.

namespace
{

enum class Builtin
{
    Void,
    Bool,
    Char,
    SChar,
    UChar,
    Short,
    UShort,
    Int,
    UInt,
    Long,
    ULong,
    LongLong,
    ULongLong,
    Float,
    Double,
    LongDouble,
    WChar,
    Char8,
    Char16,
    Char32
};

enum class Tag
{
    Struct,
    Class,
    Union,
    Enum
};

struct Type;
using TypePtr = std::shared_ptr<Type>;

struct Type
{
    enum class Kind
    {
        Builtin,
        Named,
        Pointer,
        LValueReference,
        RValueReference
    };

    Kind kind = Kind::Builtin;
    Builtin builtin = Builtin::Void;
    Tag tag = Tag::Struct;
    // a record's or an enum's qualified name, the outermost scope first
    llvm::SmallVector<std::string> name;
    bool isConst = false;
    bool isVolatile = false;
    TypePtr pointee;
};

struct Declaration
{
    TypePtr result;
    // the function's qualified name, the outermost scope first
    llvm::SmallVector<std::string> name;
    llvm::SmallVector<TypePtr> params;
    bool isVariadic = false;
    // a const (volatile) method
    bool isConst = false;
    bool isVolatile = false;
};

// The <stdint.h>/<stddef.h> typedefs that differ between targets, as clang's TargetInfo defines them
struct TargetTypedefs
{
    Builtin int64 = Builtin::LongLong;
    Builtin size = Builtin::ULong;
    Builtin ptrdiff = Builtin::Long;
    Builtin intptr = Builtin::Long;

    explicit TargetTypedefs(const llvm::Triple &triple)
    {
        auto is64 = triple.isArch64Bit();
        if (triple.isOSWindows())
        {
            // LLP64: long stays 32-bit
            size = is64 ? Builtin::ULongLong : Builtin::UInt;
            ptrdiff = intptr = is64 ? Builtin::LongLong : Builtin::Int;
        }
        else if (triple.isOSDarwin())
        {
            ptrdiff = is64 ? Builtin::Long : Builtin::Int;
        }
        else if (triple.isWasm())
        {
            int64 = is64 ? Builtin::Long : Builtin::LongLong;
        }
        else if (is64)
        {
            int64 = Builtin::Long;
        }
        else
        {
            size = Builtin::UInt;
            ptrdiff = intptr = Builtin::Int;
        }
    }
};

Builtin unsignedOf(Builtin builtin)
{
    switch (builtin)
    {
    case Builtin::Int:
        return Builtin::UInt;
    case Builtin::Long:
        return Builtin::ULong;
    case Builtin::LongLong:
        return Builtin::ULongLong;
    default:
        return builtin;
    }
}

std::string nameKey(llvm::ArrayRef<std::string> name)
{
    return llvm::join(name, "::");
}

// a type, for telling whether two are the same: a record is its qualified name, so the key of a
// record and of the scope of that name match
std::string typeKey(const Type &type)
{
    std::string key;
    switch (type.kind)
    {
    case Type::Kind::Builtin:
        key = "#" + std::to_string(static_cast<int>(type.builtin));
        break;
    case Type::Kind::Named:
        key = nameKey(type.name);
        break;
    case Type::Kind::Pointer:
        key = "P(" + typeKey(*type.pointee) + ")";
        break;
    case Type::Kind::LValueReference:
        key = "R(" + typeKey(*type.pointee) + ")";
        break;
    case Type::Kind::RValueReference:
        key = "O(" + typeKey(*type.pointee) + ")";
        break;
    }

    if (type.isConst || type.isVolatile)
    {
        key = std::string(type.isVolatile ? "V" : "") + (type.isConst ? "K" : "") + "(" + key + ")";
    }

    return key;
}

class Parser
{
    llvm::StringRef text;
    size_t pos = 0;
    TargetTypedefs typedefs;
    std::string &error;

    // one token: an identifier, `::`, `&&`, `...`, or one punctuation character; empty at the end
    llvm::StringRef peek()
    {
        while (pos < text.size() && llvm::isSpace(text[pos]))
        {
            ++pos;
        }

        if (pos >= text.size())
        {
            return {};
        }

        auto rest = text.drop_front(pos);
        if (llvm::isAlpha(rest[0]) || rest[0] == '_')
        {
            return rest.take_while([](char c) { return llvm::isAlnum(c) || c == '_'; });
        }

        for (auto multi : {"::", "&&", "..."})
        {
            if (rest.starts_with(multi))
            {
                return rest.take_front(llvm::StringRef(multi).size());
            }
        }

        return rest.take_front(1);
    }

    llvm::StringRef next()
    {
        auto token = peek();
        pos += token.size();
        return token;
    }

    bool accept(llvm::StringRef token)
    {
        if (peek() == token)
        {
            next();
            return true;
        }

        return false;
    }

    static bool isIdentifier(llvm::StringRef token)
    {
        return !token.empty() && (llvm::isAlpha(token[0]) || token[0] == '_');
    }

    bool fail(const llvm::Twine &message)
    {
        if (error.empty())
        {
            auto token = peek();
            error = (message + (token.empty() ? llvm::Twine(" at the end") : " at '" + token + "'")).str();
        }

        return false;
    }

    bool acceptQualifiers(bool &isConst, bool &isVolatile)
    {
        auto any = false;
        for (;;)
        {
            if (accept("const"))
            {
                isConst = any = true;
            }
            else if (accept("volatile"))
            {
                isVolatile = any = true;
            }
            else
            {
                return any;
            }
        }
    }

    bool parseQualifiedName(llvm::SmallVectorImpl<std::string> &name)
    {
        do
        {
            auto token = peek();
            if (token == "operator" || token == "~")
            {
                return fail("operators, constructors and destructors aren't supported");
            }

            if (!isIdentifier(token))
            {
                return fail("expected a name");
            }

            name.push_back(next().str());
        } while (accept("::"));

        if (peek() == "<")
        {
            return fail("templates aren't supported");
        }

        return true;
    }

    std::optional<Builtin> typedefBuiltin(llvm::StringRef token)
    {
        return llvm::StringSwitch<std::optional<Builtin>>(token)
            .Case("int8_t", Builtin::SChar)
            .Case("uint8_t", Builtin::UChar)
            .Case("int16_t", Builtin::Short)
            .Case("uint16_t", Builtin::UShort)
            .Case("int32_t", Builtin::Int)
            .Case("uint32_t", Builtin::UInt)
            .Case("int64_t", typedefs.int64)
            .Case("uint64_t", unsignedOf(typedefs.int64))
            .Case("size_t", typedefs.size)
            .Case("ptrdiff_t", typedefs.ptrdiff)
            .Case("intptr_t", typedefs.intptr)
            .Case("uintptr_t", unsignedOf(typedefs.intptr))
            .Default(std::nullopt);
    }

    // the type specifiers: a builtin type of one or more words, a typedef, or a record's name
    bool parseBaseType(Type &type)
    {
        int isSigned = 0, isUnsigned = 0, chars = 0, shorts = 0, ints = 0, longs = 0, doubles = 0;
        std::optional<Builtin> single;
        auto words = 0;
        for (;;)
        {
            if (acceptQualifiers(type.isConst, type.isVolatile))
            {
                continue;
            }

            auto token = peek();
            auto loneWord = llvm::StringSwitch<std::optional<Builtin>>(token)
                                .Case("void", Builtin::Void)
                                .Case("bool", Builtin::Bool)
                                .Case("float", Builtin::Float)
                                .Case("wchar_t", Builtin::WChar)
                                .Case("char8_t", Builtin::Char8)
                                .Case("char16_t", Builtin::Char16)
                                .Case("char32_t", Builtin::Char32)
                                .Default(std::nullopt);
            auto counter = llvm::StringSwitch<int *>(token)
                               .Case("signed", &isSigned)
                               .Case("unsigned", &isUnsigned)
                               .Case("char", &chars)
                               .Case("short", &shorts)
                               .Case("int", &ints)
                               .Case("long", &longs)
                               .Case("double", &doubles)
                               .Default(nullptr);
            if (loneWord)
            {
                single = loneWord;
            }
            else if (counter)
            {
                ++*counter;
            }
            else
            {
                break;
            }

            next();
            ++words;
        }

        if (words == 0)
        {
            type.kind = Type::Kind::Named;
            auto token = peek();
            auto tag = llvm::StringSwitch<std::optional<Tag>>(token)
                           .Case("struct", Tag::Struct)
                           .Case("class", Tag::Class)
                           .Case("union", Tag::Union)
                           .Case("enum", Tag::Enum)
                           .Default(std::nullopt);
            if (tag)
            {
                next();
                type.tag = *tag;
            }

            if (!isIdentifier(peek()))
            {
                return fail("expected a type");
            }

            if (!parseQualifiedName(type.name))
            {
                return false;
            }

            // `int64_t`, `std::size_t`
            if (!tag && (type.name.size() == 1 || (type.name.size() == 2 && type.name.front() == "std")))
            {
                if (auto builtin = typedefBuiltin(type.name.back()))
                {
                    type.kind = Type::Kind::Builtin;
                    type.builtin = *builtin;
                    type.name.clear();
                }
            }

            acceptQualifiers(type.isConst, type.isVolatile);
            return true;
        }

        type.kind = Type::Kind::Builtin;
        auto sign = isSigned + isUnsigned;
        auto counted = chars + shorts + ints + longs + doubles;
        if (single)
        {
            if (words != 1)
            {
                return fail("invalid combination of type specifiers");
            }

            type.builtin = *single;
        }
        else if (sign > 1 || chars > 1 || shorts > 1 || ints > 1 || longs > 2 || doubles > 1)
        {
            return fail("invalid combination of type specifiers");
        }
        else if (doubles)
        {
            if (sign || counted != 1 + longs || longs > 1)
            {
                return fail("invalid combination of type specifiers");
            }

            type.builtin = longs ? Builtin::LongDouble : Builtin::Double;
        }
        else if (chars)
        {
            if (counted != 1)
            {
                return fail("invalid combination of type specifiers");
            }

            type.builtin = isSigned ? Builtin::SChar : isUnsigned ? Builtin::UChar : Builtin::Char;
        }
        else if (shorts)
        {
            if (longs)
            {
                return fail("invalid combination of type specifiers");
            }

            type.builtin = isUnsigned ? Builtin::UShort : Builtin::Short;
        }
        else if (longs)
        {
            type.builtin = longs == 2 ? Builtin::LongLong : Builtin::Long;
            type.builtin = isUnsigned ? unsignedOf(type.builtin) : type.builtin;
        }
        else
        {
            type.builtin = isUnsigned ? Builtin::UInt : Builtin::Int;
        }

        acceptQualifiers(type.isConst, type.isVolatile);
        return true;
    }

    // the specifiers, then `*` (and its own const), `&`, `&&`
    TypePtr parseType()
    {
        auto type = std::make_shared<Type>();
        if (!parseBaseType(*type))
        {
            return nullptr;
        }

        for (;;)
        {
            auto kind = peek() == "*"    ? Type::Kind::Pointer
                        : peek() == "&"  ? Type::Kind::LValueReference
                        : peek() == "&&" ? Type::Kind::RValueReference
                                         : Type::Kind::Builtin;
            if (kind == Type::Kind::Builtin)
            {
                break;
            }

            next();
            if (type->kind == Type::Kind::LValueReference || type->kind == Type::Kind::RValueReference)
            {
                fail("a pointer or reference to a reference isn't a type");
                return nullptr;
            }

            auto outer = std::make_shared<Type>();
            outer->kind = kind;
            outer->pointee = type;
            if (kind == Type::Kind::Pointer)
            {
                acceptQualifiers(outer->isConst, outer->isVolatile);
            }

            type = outer;
        }

        // `int (*)(int)`, `int[4]`; a `(` alone is a parameter list after a missing name
        auto save = pos;
        auto isFunctionPointer = accept("(") && (peek() == "*" || peek() == "&");
        pos = save;
        if (isFunctionPointer || peek() == "[")
        {
            fail("function pointers and arrays aren't supported");
            return nullptr;
        }

        return type;
    }

    bool parseParams(Declaration &declaration)
    {
        if (accept(")"))
        {
            return true;
        }

        // `f(void)`
        auto save = pos;
        if (accept("void") && accept(")"))
        {
            return true;
        }

        pos = save;
        do
        {
            if (accept("..."))
            {
                declaration.isVariadic = true;
                break;
            }

            auto param = parseType();
            if (!param)
            {
                return false;
            }

            if (param->kind == Type::Kind::Builtin && param->builtin == Builtin::Void)
            {
                return fail("a parameter can't be void");
            }

            declaration.params.push_back(param);
            // the parameter's name, which doesn't change the symbol
            if (isIdentifier(peek()) && peek() != "const" && peek() != "volatile")
            {
                next();
            }
        } while (accept(","));

        if (!accept(")"))
        {
            return fail(peek() == "=" ? "default arguments aren't supported" : "expected ')'");
        }

        return true;
    }

public:
    Parser(llvm::StringRef text, const llvm::Triple &triple, std::string &error) : text(text), typedefs(triple), error(error)
    {
    }

    bool parse(Declaration &declaration)
    {
        // nothing a symbol depends on
        while (accept("inline") || accept("static") || accept("extern"))
        {
        }

        if (peek() == "virtual")
        {
            return fail("a virtual method is called through its vtable slot, @slot(N), not by its symbol");
        }

        declaration.result = parseType();
        if (!declaration.result || !parseQualifiedName(declaration.name))
        {
            return false;
        }

        if (!accept("("))
        {
            return fail("expected '(' after the function's name");
        }

        if (!parseParams(declaration))
        {
            return false;
        }

        acceptQualifiers(declaration.isConst, declaration.isVolatile);
        if (peek() == "&" || peek() == "&&" || peek() == "noexcept" || peek() == "throw")
        {
            return fail("ref-qualifiers and exception specifications aren't supported");
        }

        accept(";");
        if (!peek().empty())
        {
            return fail("unexpected text after the declaration");
        }

        return true;
    }
};

// https://learn.microsoft.com/cpp/build/reference/decorated-names, and clang's MicrosoftMangle.cpp
class MicrosoftMangler
{
    std::string out;
    bool is64;
    bool isX86;
    // the first ten names, and the first ten parameter types longer than a letter (by typeKey): a
    // later one is their index
    llvm::SmallVector<std::string, 10> names;
    llvm::SmallVector<std::string, 10> paramTypes;

    void sourceName(llvm::StringRef name)
    {
        auto it = llvm::find(names, name);
        if (it != names.end())
        {
            out += static_cast<char>('0' + (it - names.begin()));
            return;
        }

        out += name;
        out += '@';
        if (names.size() < 10)
        {
            names.push_back(name.str());
        }
    }

    // innermost first, each ended by `@`, and one more `@` after them
    void qualifiedName(llvm::ArrayRef<std::string> name)
    {
        for (auto &part : llvm::reverse(name))
        {
            sourceName(part);
        }

        out += '@';
    }

    static char qualifiers(bool isConst, bool isVolatile)
    {
        return isConst ? (isVolatile ? 'D' : 'B') : (isVolatile ? 'C' : 'A');
    }

    void builtin(Builtin builtin)
    {
        switch (builtin)
        {
        case Builtin::Void: out += "X"; break;
        case Builtin::Bool: out += "_N"; break;
        case Builtin::Char: out += "D"; break;
        case Builtin::SChar: out += "C"; break;
        case Builtin::UChar: out += "E"; break;
        case Builtin::Short: out += "F"; break;
        case Builtin::UShort: out += "G"; break;
        case Builtin::Int: out += "H"; break;
        case Builtin::UInt: out += "I"; break;
        case Builtin::Long: out += "J"; break;
        case Builtin::ULong: out += "K"; break;
        case Builtin::LongLong: out += "_J"; break;
        case Builtin::ULongLong: out += "_K"; break;
        case Builtin::Float: out += "M"; break;
        case Builtin::Double: out += "N"; break;
        case Builtin::LongDouble: out += "O"; break;
        case Builtin::WChar: out += "_W"; break;
        case Builtin::Char8: out += "_Q"; break;
        case Builtin::Char16: out += "_S"; break;
        case Builtin::Char32: out += "_U"; break;
        }
    }

    void named(const Type &type)
    {
        switch (type.tag)
        {
        case Tag::Struct: out += "U"; break;
        case Tag::Class: out += "V"; break;
        case Tag::Union: out += "T"; break;
        case Tag::Enum: out += "W4"; break;
        }

        qualifiedName(type.name);
    }

    // a type's own const is in the letter of what points at it, or dropped at the top of a parameter;
    // a pointer's own const is in its first letter, `QEAH` for `int *const`
    void type(const Type &type)
    {
        switch (type.kind)
        {
        case Type::Kind::Builtin:
            builtin(type.builtin);
            break;
        case Type::Kind::Named:
            named(type);
            break;
        case Type::Kind::Pointer:
        case Type::Kind::LValueReference:
        case Type::Kind::RValueReference:
            if (type.kind == Type::Kind::Pointer)
            {
                out += type.isConst ? (type.isVolatile ? 'S' : 'Q') : (type.isVolatile ? 'R' : 'P');
            }
            else
            {
                out += type.kind == Type::Kind::LValueReference ? "A" : "$$Q";
            }

            if (is64)
            {
                out += 'E';
            }

            out += qualifiers(type.pointee->isConst, type.pointee->isVolatile);
            this->type(*type.pointee);
            break;
        }
    }

    void resultType(const Type &type)
    {
        if (type.kind == Type::Kind::Named || (type.kind == Type::Kind::Builtin && (type.isConst || type.isVolatile)))
        {
            // a record or an enum returned by value, or a const builtin: `?AUPair@@`, `?BH`
            out += '?';
            out += qualifiers(type.isConst, type.isVolatile);
        }

        this->type(type);
    }

    // by the type, not by its text: a record's name is its index the second time
    void paramType(const Type &type)
    {
        auto unqualified = type;
        if (type.kind == Type::Kind::Builtin || type.kind == Type::Kind::Named)
        {
            unqualified.isConst = unqualified.isVolatile = false;
        }

        auto key = typeKey(unqualified);
        auto it = llvm::find(paramTypes, key);
        if (it != paramTypes.end())
        {
            out += static_cast<char>('0' + (it - paramTypes.begin()));
            return;
        }

        auto start = out.size();
        this->type(type);
        if (out.size() - start > 1 && paramTypes.size() < 10)
        {
            paramTypes.push_back(key);
        }
    }

public:
    explicit MicrosoftMangler(const llvm::Triple &triple)
        : is64(triple.isArch64Bit()), isX86(triple.getArch() == llvm::Triple::x86)
    {
    }

    std::string mangle(const Declaration &declaration, const CppMangleContext &context)
    {
        out = "?";
        qualifiedName(declaration.name);

        if (context.isInstanceMethod)
        {
            // public, not virtual
            out += 'Q';
            if (is64)
            {
                out += 'E';
            }

            out += qualifiers(declaration.isConst, declaration.isVolatile);
            // __thiscall on x86, __cdecl for a variadic method there and everywhere else
            out += isX86 && !declaration.isVariadic ? 'E' : 'A';
        }
        else
        {
            // public static, or a free function; __cdecl
            out += context.isStaticMethod ? "SA" : "YA";
        }

        resultType(*declaration.result);

        for (auto &param : declaration.params)
        {
            paramType(*param);
        }

        if (declaration.params.empty() && !declaration.isVariadic)
        {
            out += 'X';
        }
        else
        {
            out += declaration.isVariadic ? 'Z' : '@';
        }

        // no exception specification
        out += 'Z';
        return out;
    }
};

// https://itanium-cxx-abi.github.io/cxx-abi/abi.html#mangling, and clang's ItaniumMangle.cpp
class ItaniumMangler
{
    std::string out;
    // the substitution candidates seen so far, by key: a scope's or a record's qualified name, or a
    // qualified, pointer or reference type in parentheses
    llvm::SmallVector<std::string> substitutions;

    bool substitute(llvm::StringRef key)
    {
        auto it = llvm::find(substitutions, key);
        if (it == substitutions.end())
        {
            return false;
        }

        // S_, S0_, S1_, ... S9_, SA_, ... SZ_, S10_, ...
        auto index = it - substitutions.begin();
        out += 'S';
        if (index > 0)
        {
            std::string digits;
            for (auto value = index - 1;; value /= 36)
            {
                auto digit = value % 36;
                digits.insert(digits.begin(), static_cast<char>(digit < 10 ? '0' + digit : 'A' + digit - 10));
                if (value < 36)
                {
                    break;
                }
            }

            out += digits;
        }

        out += '_';
        return true;
    }

    void sourceName(llvm::StringRef name)
    {
        out += std::to_string(name.size());
        out += name;
    }

    // the scopes of a nested name: each one a candidate
    void prefix(llvm::ArrayRef<std::string> name)
    {
        if (name.empty() || substitute(nameKey(name)))
        {
            return;
        }

        prefix(name.drop_back());
        sourceName(name.back());
        substitutions.push_back(nameKey(name));
    }

    void recordName(llvm::ArrayRef<std::string> name)
    {
        if (substitute(nameKey(name)))
        {
            return;
        }

        if (name.size() == 1)
        {
            sourceName(name.front());
        }
        else
        {
            out += 'N';
            prefix(name.drop_back());
            sourceName(name.back());
            out += 'E';
        }

        substitutions.push_back(nameKey(name));
    }

    void builtin(Builtin builtin)
    {
        switch (builtin)
        {
        case Builtin::Void: out += "v"; break;
        case Builtin::Bool: out += "b"; break;
        case Builtin::Char: out += "c"; break;
        case Builtin::SChar: out += "a"; break;
        case Builtin::UChar: out += "h"; break;
        case Builtin::Short: out += "s"; break;
        case Builtin::UShort: out += "t"; break;
        case Builtin::Int: out += "i"; break;
        case Builtin::UInt: out += "j"; break;
        case Builtin::Long: out += "l"; break;
        case Builtin::ULong: out += "m"; break;
        case Builtin::LongLong: out += "x"; break;
        case Builtin::ULongLong: out += "y"; break;
        case Builtin::Float: out += "f"; break;
        case Builtin::Double: out += "d"; break;
        case Builtin::LongDouble: out += "e"; break;
        case Builtin::WChar: out += "w"; break;
        case Builtin::Char8: out += "Du"; break;
        case Builtin::Char16: out += "Ds"; break;
        case Builtin::Char32: out += "Di"; break;
        }
    }

    void unqualifiedType(const Type &type)
    {
        switch (type.kind)
        {
        case Type::Kind::Builtin:
            builtin(type.builtin);
            return;
        case Type::Kind::Named:
            recordName(type.name);
            return;
        default:
            break;
        }

        Type unqualified = type;
        unqualified.isConst = unqualified.isVolatile = false;
        auto key = typeKey(unqualified);
        if (substitute(key))
        {
            return;
        }

        out += type.kind == Type::Kind::Pointer ? 'P' : type.kind == Type::Kind::LValueReference ? 'R' : 'O';
        this->type(*type.pointee);
        substitutions.push_back(key);
    }

    void type(const Type &type)
    {
        if (!type.isConst && !type.isVolatile)
        {
            unqualifiedType(type);
            return;
        }

        auto key = typeKey(type);
        if (substitute(key))
        {
            return;
        }

        out += type.isVolatile ? "V" : "";
        out += type.isConst ? "K" : "";
        unqualifiedType(type);
        substitutions.push_back(key);
    }

public:
    std::string mangle(const Declaration &declaration)
    {
        out = "_Z";
        if (declaration.name.size() == 1)
        {
            sourceName(declaration.name.front());
        }
        else
        {
            out += 'N';
            out += declaration.isVolatile ? "V" : "";
            out += declaration.isConst ? "K" : "";
            prefix(llvm::ArrayRef<std::string>(declaration.name).drop_back());
            sourceName(declaration.name.back());
            out += 'E';
        }

        // the result type isn't part of a function's symbol; a parameter's own const isn't either
        for (auto &param : declaration.params)
        {
            auto unqualified = *param;
            unqualified.isConst = unqualified.isVolatile = false;
            type(unqualified);
        }

        if (declaration.isVariadic)
        {
            out += 'z';
        }
        else if (declaration.params.empty())
        {
            out += 'v';
        }

        return out;
    }
};

} // namespace

bool mangleCppName(llvm::StringRef declarationText, const llvm::Triple &triple, const CppMangleContext &context,
                   std::string &symbol, std::string &error)
{
    error.clear();
    Declaration declaration;
    Parser parser(declarationText, triple, error);
    if (!parser.parse(declaration))
    {
        return false;
    }

    if (declaration.name.size() == 1 && !context.className.empty())
    {
        declaration.name.insert(declaration.name.begin(), context.className.str());
    }
    else if (declaration.name.size() == 1 && (context.isInstanceMethod || context.isStaticMethod))
    {
        // it would be a free function's symbol
        error = "the method's class isn't known: qualify the name, Class::" + declaration.name.front();
        return false;
    }

    if (declaration.name.size() >= 2 && declaration.name.back() == declaration.name[declaration.name.size() - 2])
    {
        error = "constructors aren't supported";
        return false;
    }

    if ((declaration.isConst || declaration.isVolatile) && !context.isInstanceMethod)
    {
        error = "only a method with 'this' can be const or volatile";
        return false;
    }

    // MSVC on Windows unless the environment says GNU (MinGW) or Cygwin, as clang decides it
    if (triple.isWindowsMSVCEnvironment())
    {
        symbol = MicrosoftMangler(triple).mangle(declaration, context);
    }
    else
    {
        symbol = ItaniumMangler().mangle(declaration);
    }

    return true;
}
