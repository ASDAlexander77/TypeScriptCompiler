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
        Reference
    };

    Kind kind = Kind::Builtin;
    std::string name;            // Builtin: the tslang spelling; Named: the key of a Decl
    std::vector<TsType> pointee; // Reference: exactly one element
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
    Function      // declare function f(...): R;
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

    std::vector<Field> fields;           // Struct: fields; Function: parameters
    TsType type;                         // Function: result; Typedef: aliased type; FixedEnum: underlying
    bool varargs = false;                // Function
    std::vector<Enumerator> enumerators; // Enum, FixedEnum
    bool scoped = false;                 // FixedEnum from an `enum class`
    std::string value;                   // Macro: a TS literal
};

struct HeaderModel
{
    std::vector<Decl> decls; // in the order the header declares them
};

} // namespace tsbindgen

#endif // TSCLANGIMPORTER_BINDINGMODEL_H
