#ifndef TYPESCRIPT_CPPMANGLER_H_
#define TYPESCRIPT_CPPMANGLER_H_

#include "llvm/ADT/StringRef.h"
#include "llvm/TargetParser/Triple.h"

#include <string>

// @cppname("int Methods::scaled(int)"): a C++ declaration, turned into the symbol the target's C++
// compiler gives it - MSVC's scheme on windows-msvc, the Itanium one of g++ and clang everywhere
// else (MinGW included) - so one declaration binds the same C++ function on every OS, where
// @linkname needs each OS's symbol spelled out.
//
// The declaration is a return type, a name, and parameter types, and `const` after them for a
// const method. Types: the builtin ones (`unsigned long long`, `char16_t`, ...), the <stdint.h>
// and <stddef.h> typedefs (`int64_t`, `size_t`, ...) as the target defines them - `int64_t` is
// `long` on 64-bit Linux and `long long` on Windows and macOS, which mangle differently - and a
// struct, class, union or enum by its qualified name (`ns::Pt`, `class Obj`; a bare name is a
// struct), with `const`/`volatile`, `*`, `&` and `&&`. Templates, operators, constructors,
// destructors and function pointers are not supported.
struct CppMangleContext
{
    // the class an unqualified name is a member of (`int scaled(int)` in a @cpp class `Methods` is
    // `Methods::scaled`); empty outside a class
    llvm::StringRef className;
    // a method with `this` (MSVC encodes it, and its `const`); a static method or a free function
    // has none
    bool isInstanceMethod = false;
    // a static method: MSVC tells it from a free function in a namespace of that name
    bool isStaticMethod = false;
};

// The symbol, or false with `error` saying what is wrong with the declaration.
bool mangleCppName(llvm::StringRef declaration, const llvm::Triple &triple, const CppMangleContext &context,
                   std::string &symbol, std::string &error);

#endif // TYPESCRIPT_CPPMANGLER_H_
