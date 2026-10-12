#ifndef TSCLANGIMPORTER_BINDINGMODEL_H
#define TSCLANGIMPORTER_BINDINGMODEL_H

#include <functional>
#include <string>
#include <vector>

namespace tsbindgen
{

// A tslang type as the printer spells it. A named type holds the key of the declaration it refers
// to, not a TS name: --strip-prefix renames declarations after the header has been mapped.
struct TsType
{
    enum class Kind
    {
        Builtin,
        Named,
        Reference,
        Tuple // an inline named tuple, `[a: T1, b: T2]`: a C++ record a method returns by value
    };

    Kind kind = Kind::Builtin;
    std::string name;            // Builtin: the tslang spelling; Named: the key of a Decl
    std::vector<TsType> pointee; // Reference: exactly one element; Tuple: the field types
    std::vector<std::string> fieldNames; // Tuple
    std::string comment;         // shown next to the type: a function pointer's C signature, ...
    bool functionPointer = false;
    std::string skipReason;      // non-empty: the C type has no tslang mapping

    bool skipped() const
    {
        return !skipReason.empty();
    }

    static TsType builtin(std::string name, std::string comment = {});
    static TsType named(std::string key);
    static TsType reference(TsType pointee);
    static TsType tuple(std::vector<std::string> names, std::vector<TsType> types);
    static TsType skip(std::string reason);
};

// Spells `type`, asking `nameOf` for the TS name of each named declaration it refers to.
std::string renderType(const TsType &type, const std::function<std::string(const std::string &)> &nameOf);

// The declaration keys `type` refers to, in order, each once.
void collectKeys(const TsType &type, std::vector<std::string> &keys);

// "struct:Point" -> "Point"
std::string nameFromKey(const std::string &key);

// A name that is a TS reserved word gets a `_` suffix: `delete` -> `delete_`.
std::string escapeReserved(std::string name);

// A type tslang itself defines (`boolean`, `string`, `index`, `s32`, ...). tslang ignores a `type`
// alias of such a name and keeps its own type, so a C type of that name must never be declared under it.
bool isTsBuiltinTypeName(const std::string &name);

struct Field
{
    std::string name;
    TsType type;
};

struct Enumerator
{
    std::string name;
    std::string value; // decimal
};

enum class DeclKind
{
    Macro,        // const N = 42;
    OpaqueStruct, // type S = Opaque;
    Struct,       // type S = [a: T1, b: T2];  (type S = Opaque; when skipped)
    Enum,         // enum E { A = 0 }
    FixedEnum,    // type E = u8; const A = 1;
    Typedef,      // type T = ...;
    CppClass,     // @cpp class C extends B { @slot(N) f(...): R; @linkname("...") g(...): R; }
    Function      // declare function f(...): R;
};

// A method of a C++ class bound as a @cpp class.
struct Method
{
    std::string name;        // the TS name: the C++ name, `_2`, `_3`, ... for later overloads
    std::string cppName;     // the C++ name, for messages
    int slot = -1;           // a virtual method's vtable slot; -1 for a non-virtual one
    std::string symbol;      // a non-virtual method's mangled symbol
    std::vector<Field> parameters;
    TsType result;
    bool nonTrivialReturn = false; // returns a record with a destructor or a copy constructor
    std::string skipReason;  // non-empty: left out, with a `// skipped:` line in its place
};

struct Decl
{
    DeclKind kind = DeclKind::Function;
    std::string key;  // unique within the model: "<kind>:<C name>"
    std::string name; // the C name
    std::string symbol; // Function: the symbol it links as, when an asm label makes it differ from the name
    bool inMainFile = false;
    bool inSystemHeader = false;
    std::string skipReason; // non-empty: left out, with a `// skipped:` line in its place

    std::vector<Field> fields;           // Struct, CppClass: fields; Function: parameters
    TsType type;                         // Function: result; Typedef: aliased type; FixedEnum: underlying
    bool varargs = false;                // Function
    std::vector<Enumerator> enumerators; // Enum, FixedEnum
    bool scoped = false;                 // FixedEnum from an `enum class`
    std::string value;                   // Macro: a TS literal

    std::string qualifiedName;           // CppClass: the C++ name with its namespaces (`nvrhi::IDevice`)
    std::string base;                    // CppClass: the key of the base class, or empty
    std::vector<Method> methods;         // CppClass: the methods it introduces, in declaration order
    std::string fieldsSkipReason;        // CppClass: non-empty when its fields aren't declared
};

struct HeaderModel
{
    std::vector<Decl> decls; // in the order the header declares them
};

} // namespace tsbindgen

#endif // TSCLANGIMPORTER_BINDINGMODEL_H
