# `tsbindgen` v1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A `tsbindgen` executable that turns a C header (or the `extern "C"` part of a C++ file) into a tslang source file of bindings (`declare function`, named-tuple `type`s, `enum`s, `const`s) that a tslang program includes to call the C library.

**Architecture:** A static library, `TsClangImporter`, runs clang over the input (`clang::tooling::ToolInvocation`, a `RecursiveASTVisitor`, and `PPCallbacks` for macros), maps each C declaration to a tslang one (`TypeMapper`, widths from the target's `ASTContext`), selects what to emit (the input file's own declarations, or `--filter` globs, plus every type those use), and prints the bindings (`BindingPrinter`). `tsbindgen` is a thin command line over it. `tslang.exe` gains no dependency.

**Tech Stack:** C++17, clang 22.1.8 C++ API (Tooling, Frontend, AST, Lex, Basic) from the in-tree LLVM install, gtest/gmock via `add_mlir_unittest`, ctest + `cmake -P` scripts.

**Spec:** `docs/superpowers/specs/2026-09-25-tsbindgen-linkname-design.md`, section "PR 2 — `tsbindgen` v1", as corrected by "Deviations from the spec" below. PR 3 (packaging) is not in this plan.

**Status of the code below:** every file in this plan was built against LLVM 22.1.8 and run on Windows x64 on 2026-09-26: 46 unit tests pass, `test-bindgen-cli` and `test-bindgen-plain` pass under ctest. `test-bindgen-namespace` needs PR 1 and was not run. Nothing was run on Linux.

## Prerequisite: PR 1 (`@linkname`) must be merged first

PR 1 is **not implemented**. Commit `695387d8` ("Implement @linkname as an alias for @dllname…") adds only its plan, `docs/superpowers/plans/2026-09-25-linkname.md`; no code in `tslang/include` or `tslang/lib` mentions `linkname`, and a probe of `namespace Fx { @linkname("fx_add") export declare function add(...) }` still binds the symbol `Fx.add` and marks it `dllexport` (link error: unresolved `Fx.add`).

`--namespace` output depends on it: every function it emits carries `@linkname("<C name>")`. So:

1. Execute `docs/superpowers/plans/2026-09-25-linkname.md` and merge it.
2. Then branch this work from `main`: `git fetch origin && git checkout --no-track -b tsbindgen-v1 origin/main && git push -u origin tsbindgen-v1`.

Tasks 1–4 and the `plain` half of Task 5 do not need PR 1; `test-bindgen-namespace` (Task 5) does, and fails with `unresolved external symbol Fx.add` without it.

## Deviations from the spec (measured 2026-09-26)

Each of these was found by running generated bindings through tslang; the spec's version does not work.

| Spec says | Plan does | Why (measured) |
| --- | --- | --- |
| Output is a `.d.ts` (`-o <out.d.ts>`) | Output is a `.ts` file, included with `/// <reference path="fixture.ts" />`; `-o` defaults to stdout; a `-o x.d.ts` gets a warning | In a `.d.ts` — and in any file pulled in with `import` — a `const` is an ambient declaration with no value: `const FIXTURE_ANSWER = 42;` fails with `LNK2019: unresolved external symbol FIXTURE_ANSWER` (exe) and `Symbols not found: [ FIXTURE_ANSWER ]` (JIT). The same text in a `.ts` included by `/// <reference path>` works in both. `import "./fixture"` is also wrong for another reason: when `fixture.dll` exists it loads that as a tslang library (`MLIRGenModule.cpp`, `mlirGen(ImportDeclaration)`). |
| Signed C integers map to `i8`…`i64` | Signed → `s8`…`s64`, unsigned → `u8`…`u64` | tslang's `iN` are **signless** (`MLIRGenTypes.cpp`: `{"i32", builder.getIntegerType(32)}` vs `{"s32", builder.getIntegerType(32, true)}`): an `int8_t` -5 prints `251`, an `int16_t` -3 prints `65533`, an `int32_t` -5 prints `4294967291`. With `sN` all print correctly. |
| `struct Node { struct Node *next; }` → `next: Reference<Node>` | A pointer field that leads back to its own struct — directly, through another struct, or through a typedef — is `Opaque`, with `/* Reference<Node>: a type cannot refer to itself */` | tslang dies on a self-referencing type alias: `type Node = [value: i32, next: Reference<Node>]; function main() { let n: Node = [1, null]; }` exits 127 with no message, even at `--emit=mlir`. That is a compiler bug to file separately; tsbindgen must not emit a cycle meanwhile. |
| Function-pointer parameters get a trailing `// (a: i32) => i32` | Inline `fn: Opaque /* (p0: s32) => s32 */` for parameters and fields; trailing `//` for a `typedef` | A declaration can have several callbacks; one trailing comment cannot say which is which. Parameter names inside the signature are `p0…`: a C function type carries no names. The file header says once that a callback must have no captures and is passed as `fn as Opaque` (spec open issue "Callback lifetime"). |
| `runToolOnCodeWithArgs`-style parsing (implied) | `ToolInvocation` with our own diagnostic consumer | `runToolOnCodeWithArgs` ignores errors in the clang command line itself: `tsbindgen fixture.h -- --bogus-flag` printed `error: unknown argument` and still exited 0 with output. |
| (silent) | An object-like macro that is not a literal (`#define FLAGS (1 << 3)`, `#define API __declspec(dllexport)`) is skipped with `macro is not a literal`; one with no tokens (include guards) is ignored | The skip list names function-like macros only; a silent drop would hide API. |
| (silent) | `--strip-prefix` never produces a name that is empty, starts with a digit, or is taken: it keeps the C name and warns; a stripped reserved word gets `_` | `fx_len` stripping to `len` when `len` exists would declare two functions `len`. |
| (silent) | A struct whose clang layout is not natural field-after-field (packed, `#pragma pack`, over-aligned) is skipped: `packed or over-aligned layout` | A named tuple is laid out naturally; the spec's measured baseline covers only natural layouts. |
| (silent) | Nothing selected is a warning | `tsbindgen umbrella.h` where `umbrella.h` only `#include`s others emits an empty file; the warning says to use `--filter`. |

Task 8 amends the spec with this table.

## Global Constraints

- Build tree: `I:/TypeScriptCompiler/__build/tslang/windows-msbuild-2026-release` (multi-config MSBuild). After any `CMakeLists.txt` change, reconfigure: `cmake .` in the build tree.
- Build: `cmake --build I:/TypeScriptCompiler/__build/tslang/windows-msbuild-2026-release --config Release -j 16 --target <targets>`.
- `ctest` must be given `-C Release` and run from the build tree; without it every test shows `(Not Run)`.
- A header change followed by a crash in code that did not change (`SEH exception 0xC0000005`, segfault in `tsbindgen`) is a stale object: rebuild with `--clean-first` before debugging.
- clang is **22.1.8**; its resource directory is `<LLVM>/lib/clang/22`. Use `${LLVM_LIBRARY_DIR}/clang/${LLVM_VERSION_MAJOR}` in CMake and `CLANG_VERSION_MAJOR` / `CLANG_VERSION_STRING` (`clang/Basic/Version.h`) in C++; never hard-code 22.
- clang 22 has no `ElaboratedType`, and `TagType::getDecl()` returns the declaration as written: look through sugar with `getAs<...>()`, and take a struct's definition with `getDefinition()`.
- The unit tests build only where `3rdParty/llvm-project/third-party/unittest` exists (the same condition as `MLIRGenTests`). CI coverage comes from the ctest scripts in Task 4 and Task 5, not the unit tests.
- `tslang.exe` must not link any new library. Only `TsClangImporter`, `tsbindgen` and `TsClangImporterTests` link clang's frontend.
- Integer mapping: signed → `s8`/`s16`/`s32`/`s64`, unsigned → `u8`/`u16`/`u32`/`u64`, never `i*`, `int` or `long`.
- The generated file is `.ts`, included with `/// <reference path="..." />`. Never `import` it.
- Exit codes: `0` file written, `1` parse error or I/O failure (clang's diagnostics printed as clang prints them), `2` bad arguments.
- Every commit message ends with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`. Commits are GPG-signed; if signing times out, retry, then hand the commit to the user. Never pass `--no-gpg-sign`.
- Run `gh auth switch --user ASDAlexander77` before any `gh` command.

## Review Focus

1. **Narrow integer and `bool` parameters on Linux x86-64.** clang-compiled callees on the SysV ABI assume the caller sign/zero-extended an `int8_t`, `uint16_t` or `bool` argument to 32 bits; tslang emits `declare` parameters without `signext`/`zeroext`. Expected: `narrow -5 65535 -5 false true` on Linux exactly as on Windows. Pinned by the `narrow` line of `test/bindgen/expected.txt` (Task 5), checked on Linux CI in Task 6. If it fails there, tsbindgen's mapping is right and tslang's call lowering is wrong: file a compiler issue, do not change the mapping.
2. **Hosted system headers from inside ctest.** `fixture.h` includes `<stdbool.h>`, `<stddef.h>`, `<stdint.h>` without `-ffreestanding`, so clang's driver must find the MSVC/Windows SDK headers (or glibc's) with no developer prompt in the environment. Expected: exit 0 on both CI systems. Pinned by `test-bindgen-plain` (Task 5).
3. **A real header, C++ input.** `TypeScriptCompilerDefaultLib/src/wrappers/regex.cpp` is a `.cpp` with an `extern "C"` API over `std::regex`, pulling in the C++ standard library. Expected: exit 0, every `regexp_*` function emitted, matching `lib.native.d.ts` up to type spelling. Checked by hand in Task 7, with the differences written into the PR description.
4. **The same stem as a library.** `import "./fixture"` loads a `fixture.dll`/`libfixture.so` beside the file as a tslang library; the generated file's third line says to include it with `/// <reference path>`. Pinned by `test-bindgen-cli` (the hint) and `test-bindgen-plain` (a C library named `fixture_c`, bindings named `fixture.ts`).
5. **Per-target layout.** A struct's field widths come from the target: `struct Sized { long a; char b; }` is `[a: s32, b: s8]` for `x86_64-pc-windows-msvc` and `[a: s64, b: s8]` for `x86_64-linux-gnu`. Pinned by `TypeMapping.StructFieldsHaveTheTargetsWidths` (Task 2).

## File map

| File | Responsibility |
| --- | --- |
| `tslang/include/TsClangImporter/BindingModel.h`, `tslang/lib/TsClangImporter/BindingModel.cpp` | The mapped declarations (`Decl`, `TsType`, `HeaderModel`) and small helpers: `renderType`, `collectKeys`, `nameFromKey`, `escapeReserved` |
| `tslang/include/TsClangImporter/TypeMapper.h`, `tslang/lib/TsClangImporter/TypeMapper.cpp` | C type → `TsType`, a struct's skip reason (union, bitfield, layout, …), type-cycle breaking |
| `tslang/include/TsClangImporter/HeaderParser.h`, `tslang/lib/TsClangImporter/HeaderParser.cpp` | Runs clang, visits declarations and macros, builds the `HeaderModel` |
| `tslang/include/TsClangImporter/BindingPrinter.h`, `tslang/lib/TsClangImporter/BindingPrinter.cpp` | Selection (`--filter` / main file + used types), TS names (`--strip-prefix`), printing (`--namespace`) |
| `tslang/lib/TsClangImporter/CMakeLists.txt` | The static library, linking clang |
| `tslang/tsbindgen/tsbindgen.cpp`, `tslang/tsbindgen/CMakeLists.txt` | Command line, resource-directory lookup, exit codes, version |
| `tslang/unittests/TsClangImporter/*` | gtest unit tests over `parseHeader` + `printBindings` |
| `tslang/test/bindgen/*` | Fixture C library, programs using the generated bindings, the expected output, and the two `cmake -P` drivers |
| Modified: `tslang/lib/CMakeLists.txt`, `tslang/CMakeLists.txt`, `tslang/unittests/CMakeLists.txt`, `tslang/test/tester/CMakeLists.txt` | `add_subdirectory` lines and three ctest registrations |

---

### Task 1: Confirm clang in the CI packages; the library skeleton and its unit-test target

**Files:**
- Create: `tslang/include/TsClangImporter/BindingModel.h`
- Create: `tslang/lib/TsClangImporter/BindingModel.cpp`
- Create: `tslang/lib/TsClangImporter/CMakeLists.txt`
- Create: `tslang/unittests/TsClangImporter/CMakeLists.txt`
- Create: `tslang/unittests/TsClangImporter/BindingModelTest.cpp`
- Modify: `tslang/lib/CMakeLists.txt` (after `add_subdirectory(TypeScriptMemAllocPass)`)
- Modify: `tslang/unittests/CMakeLists.txt` (after `add_subdirectory(MLIRGen)`)

**Interfaces:**
- Produces: namespace `tsbindgen`; `struct TsType { Kind kind; std::string name; std::vector<TsType> pointee; std::string comment; bool functionPointer; std::string skipReason; bool skipped() const; static TsType builtin(std::string, std::string = {}); static TsType named(std::string key); static TsType reference(TsType); static TsType skip(std::string); }`; `std::string renderType(const TsType &, const std::function<std::string(const std::string &)> &nameOf)`; `void collectKeys(const TsType &, std::vector<std::string> &)`; `std::string nameFromKey(const std::string &)`; `std::string escapeReserved(std::string)`; `struct Field { std::string name; TsType type; }`; `struct Enumerator { std::string name; std::string value; }`; `enum class DeclKind { Macro, OpaqueStruct, Struct, Enum, FixedEnum, Typedef, Function }`; `struct Decl { DeclKind kind; std::string key, name; bool inMainFile, inSystemHeader; std::string skipReason; std::vector<Field> fields; TsType type; bool varargs; std::vector<Enumerator> enumerators; bool scoped; std::string value; }`; `struct HeaderModel { std::vector<Decl> decls; }`. Keys are `"struct:<name>"`, `"enum:<name>"`, `"typedef:<name>"`, `"fn:<name>"`, `"macro:<name>"`.

- [ ] **Step 1: Confirm both CI LLVM packages contain clang's frontend libraries, the `clang` binary and the resource headers**

The spec's open issue "CI LLVM package contents". The URLs are the `LLVM_ZIPFILE` / `LLVM_TARGZFILE` values in `.github/workflows/cmake-test-release-win.yml` and `cmake-test-release-linux.yml`.

```bash
cd "$(mktemp -d)"
curl -sL "$(grep -oP 'LLVM_TARGZFILE: "\K[^"]+' /i/TypeScriptCompiler/.github/workflows/cmake-test-release-linux.yml)" -o llvm.tgz
tar -tzf llvm.tgz | grep -E 'lib/libclang(Tooling|Frontend|AST|Lex|Basic)\.a$|bin/clang$|lib/clang/[0-9]+/include/stddef\.h$'
curl -sL "$(grep -oP 'LLVM_ZIPFILE: "\K[^"]+' /i/TypeScriptCompiler/.github/workflows/cmake-test-release-win.yml)" -o llvm.zip
unzip -l llvm.zip | grep -E 'lib/clang(Tooling|Frontend|AST|Lex|Basic)\.lib$|bin/clang\.exe$|lib/clang/[0-9]+/include/stddef\.h$'
```

Expected: each command lists 7 lines (5 libraries, the binary, `stddef.h`). If a package lacks any of them, stop: it has to be rebuilt with clang and re-uploaded, and `CACHE_VERSION` bumped in both workflows (as for the earlier `/MT` re-upload) — that is a separate change the user makes before this plan continues.

- [ ] **Step 2: Write the failing test**

`tslang/unittests/TsClangImporter/BindingModelTest.cpp`:

```cpp
// The model's helpers: how a mapped type is spelled, which declarations it refers to, and the
// names that must not reach TS unchanged.

#include "TsClangImporter/BindingModel.h"

#include "gmock/gmock.h"

using namespace tsbindgen;

TEST(BindingModel, RendersBuiltinNamedAndReference)
{
    auto nameOf = [](const std::string &key) { return "TS_" + nameFromKey(key); };

    EXPECT_EQ(renderType(TsType::builtin("s32"), nameOf), "s32");
    EXPECT_EQ(renderType(TsType::named("struct:Point"), nameOf), "TS_Point");
    EXPECT_EQ(renderType(TsType::reference(TsType::reference(TsType::named("struct:Point"))), nameOf),
              "Reference<Reference<TS_Point>>");
}

TEST(BindingModel, CollectsEachKeyOnce)
{
    std::vector<std::string> keys;
    collectKeys(TsType::reference(TsType::named("struct:Point")), keys);
    collectKeys(TsType::named("struct:Point"), keys);
    collectKeys(TsType::named("enum:Color"), keys);
    collectKeys(TsType::builtin("string"), keys);

    EXPECT_THAT(keys, testing::ElementsAre("struct:Point", "enum:Color"));
}

TEST(BindingModel, KeysNameTheirDeclaration)
{
    EXPECT_EQ(nameFromKey("typedef:callback_t"), "callback_t");
    EXPECT_EQ(nameFromKey("plain"), "plain");
}

TEST(BindingModel, ReservedWordsGetASuffix)
{
    EXPECT_EQ(escapeReserved("delete"), "delete_");
    EXPECT_EQ(escapeReserved("function"), "function_");
    EXPECT_EQ(escapeReserved("yield"), "yield_");
    EXPECT_EQ(escapeReserved("value"), "value");
    EXPECT_EQ(escapeReserved("len"), "len");
}

TEST(BindingModel, SkippedTypeKeepsItsReason)
{
    auto type = TsType::skip("long double");
    EXPECT_TRUE(type.skipped());
    EXPECT_EQ(type.skipReason, "long double");
    EXPECT_FALSE(TsType::builtin("f64").skipped());
}
```

`tslang/unittests/TsClangImporter/CMakeLists.txt` (Task 2 and Task 3 add files to this list):

```cmake
add_mlir_unittest(TsClangImporterTests
  BindingModelTest.cpp
)

target_link_libraries(TsClangImporterTests
  PRIVATE
  TsClangImporter
)
```

In `tslang/unittests/CMakeLists.txt`, after `add_subdirectory(MLIRGen)`:

```cmake
add_subdirectory(TsClangImporter)
```

- [ ] **Step 3: Run it to verify it fails**

Run: `cd I:/TypeScriptCompiler/__build/tslang/windows-msbuild-2026-release && cmake . && cmake --build . --config Release -j 16 --target TsClangImporterTests`
Expected: FAIL — CMake error that `TsClangImporter` is not a target (or, once it is, `fatal error C1083: Cannot open include file: 'TsClangImporter/BindingModel.h'`).

- [ ] **Step 4: Write the model**

`tslang/include/TsClangImporter/BindingModel.h`:

```cpp
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
```

`tslang/lib/TsClangImporter/BindingModel.cpp`:

```cpp
#include "TsClangImporter/BindingModel.h"

#include <algorithm>

namespace tsbindgen
{

TsType TsType::builtin(std::string name, std::string comment)
{
    TsType type;
    type.kind = Kind::Builtin;
    type.name = std::move(name);
    type.comment = std::move(comment);
    return type;
}

TsType TsType::named(std::string key)
{
    TsType type;
    type.kind = Kind::Named;
    type.name = std::move(key);
    return type;
}

TsType TsType::reference(TsType pointee)
{
    TsType type;
    type.kind = Kind::Reference;
    type.pointee.push_back(std::move(pointee));
    return type;
}

TsType TsType::skip(std::string reason)
{
    TsType type;
    type.skipReason = std::move(reason);
    return type;
}

std::string renderType(const TsType &type, const std::function<std::string(const std::string &)> &nameOf)
{
    switch (type.kind)
    {
    case TsType::Kind::Builtin:
        return type.name;
    case TsType::Kind::Named:
        return nameOf(type.name);
    case TsType::Kind::Reference:
        return "Reference<" + renderType(type.pointee.front(), nameOf) + ">";
    }

    return type.name;
}

void collectKeys(const TsType &type, std::vector<std::string> &keys)
{
    if (type.kind == TsType::Kind::Named && std::find(keys.begin(), keys.end(), type.name) == keys.end())
    {
        keys.push_back(type.name);
    }

    for (auto &pointee : type.pointee)
    {
        collectKeys(pointee, keys);
    }
}

std::string nameFromKey(const std::string &key)
{
    auto colon = key.find(':');
    return colon == std::string::npos ? key : key.substr(colon + 1);
}

std::string escapeReserved(std::string name)
{
    // ECMAScript reserved words, including the strict-mode ones
    static const char *const reserved[] = {
        "await",   "break",    "case",       "catch",     "class",   "const",   "continue", "debugger",
        "default", "delete",   "do",         "else",      "enum",    "export",  "extends",  "false",
        "finally", "for",      "function",   "if",        "implements", "import", "in",     "instanceof",
        "interface", "let",    "new",        "null",      "package", "private", "protected", "public",
        "return",  "static",   "super",      "switch",    "this",    "throw",   "true",     "try",
        "typeof",  "var",      "void",       "while",     "with",    "yield"};

    for (auto *word : reserved)
    {
        if (name == word)
        {
            return name + "_";
        }
    }

    return name;
}

} // namespace tsbindgen
```

`tslang/lib/TsClangImporter/CMakeLists.txt` (Task 2 replaces it with the full version):

```cmake
# C header -> tslang bindings, as a library: tsbindgen is a thin command line over it, and v2's C++
# wrapper generation is meant to reuse it. It links clang's frontend; tslang itself does not.
set(LLVM_LINK_COMPONENTS
  Support
  )

add_llvm_library(TsClangImporter
  BindingModel.cpp
  )
```

In `tslang/lib/CMakeLists.txt`, after `add_subdirectory(TypeScriptMemAllocPass)`:

```cmake
add_subdirectory(TsClangImporter)
```

- [ ] **Step 5: Run it to verify it passes**

Run: `cmake . && cmake --build . --config Release -j 16 --target TsClangImporterTests && ctest -C Release -R unittest-TsClangImporterTests --output-on-failure`
Expected: `100% tests passed, 0 tests failed out of 1` (5 gtest cases inside).

- [ ] **Step 6: Commit**

```bash
git add tslang/include/TsClangImporter/BindingModel.h tslang/lib/TsClangImporter tslang/lib/CMakeLists.txt tslang/unittests/TsClangImporter tslang/unittests/CMakeLists.txt
git commit -m "tsbindgen: TsClangImporter library skeleton and binding model

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 2: Parse a header and print its bindings: the type-mapping table

The whole pipeline lands here, because every test goes header text → clang → model → printed text: `TypeMapper` (C type → `TsType`, struct skip reasons, cycle breaking), `HeaderParser` (clang run, declarations, macros), `BindingPrinter` (selection, names, printing). Task 3 then pins the skip, selection and naming rules with their own tests.

**Files:**
- Create: `tslang/include/TsClangImporter/TypeMapper.h`, `tslang/lib/TsClangImporter/TypeMapper.cpp`
- Create: `tslang/include/TsClangImporter/HeaderParser.h`, `tslang/lib/TsClangImporter/HeaderParser.cpp`
- Create: `tslang/include/TsClangImporter/BindingPrinter.h`, `tslang/lib/TsClangImporter/BindingPrinter.cpp`
- Modify: `tslang/lib/TsClangImporter/CMakeLists.txt` (full version)
- Create: `tslang/unittests/TsClangImporter/ImporterTestHelper.h`, `TypeMappingTest.cpp`, `ParseErrorTest.cpp`
- Modify: `tslang/unittests/TsClangImporter/CMakeLists.txt`

**Interfaces:**
- Consumes: Task 1's model.
- Produces: `struct ParseInput { std::string code, fileName; std::vector<std::string> args; std::vector<std::pair<std::string, std::string>> virtualFiles; }`; `llvm::Expected<HeaderModel> parseHeader(const ParseInput &)` (error: `"clang could not parse '<fileName>'"`, clang's diagnostics already on stderr); `struct PrintOptions { std::vector<std::string> filters; std::string namespaceName, stripPrefix; std::vector<std::string> headerLines; }`; `struct PrintResult { std::string text; std::vector<std::string> warnings; }`; `llvm::Expected<PrintResult> printBindings(const HeaderModel &, const PrintOptions &)` (error only for an invalid glob). `class TypeMapper` is internal to the library.

- [ ] **Step 1: Write the failing tests**

`tslang/unittests/TsClangImporter/ImporterTestHelper.h` — parses in memory, hermetically: `-ffreestanding -nostdlibinc` leave only clang's own headers from the resource directory, so no SDK is needed and one Windows host checks both targets:

```cpp
#ifndef TSCLANGIMPORTER_UNITTESTS_IMPORTERTESTHELPER_H
#define TSCLANGIMPORTER_UNITTESTS_IMPORTERTESTHELPER_H

#include "TsClangImporter/BindingPrinter.h"
#include "TsClangImporter/HeaderParser.h"

#include "gtest/gtest.h"

#include <string>
#include <utility>
#include <vector>

namespace tsbindgen_test
{

inline const char *const windowsTarget = "x86_64-pc-windows-msvc";
inline const char *const linuxTarget = "x86_64-linux-gnu";

struct Generated
{
    std::string text;
    std::vector<std::string> warnings;
};

// Parses `header` as the file input.h and prints its bindings. Hermetic: -ffreestanding and
// -nostdlibinc leave only clang's own headers (stdint.h, stddef.h, ...), from the resource directory
// of the LLVM the tests were built with, so no system SDK is needed and one host checks any target.
// `files` are extra headers input.h can include; a path under "sys/" is a system header.
inline Generated generate(const std::string &header, tsbindgen::PrintOptions options = {},
                          const std::string &target = windowsTarget,
                          std::vector<std::pair<std::string, std::string>> files = {},
                          const std::string &language = "c")
{
    tsbindgen::ParseInput input;
    input.code = header;
    input.fileName = "input.h";
    input.args = {"-x", language, "--target=" + target, "-ffreestanding", "-nostdlibinc", "-isystem", "sys",
                  "-resource-dir", TSBINDGEN_TEST_RESOURCE_DIR};
    input.virtualFiles = std::move(files);

    auto model = tsbindgen::parseHeader(input);
    if (!model)
    {
        ADD_FAILURE() << llvm::toString(model.takeError());
        return {};
    }

    auto printed = tsbindgen::printBindings(*model, options);
    if (!printed)
    {
        ADD_FAILURE() << llvm::toString(printed.takeError());
        return {};
    }

    return {printed->text, printed->warnings};
}

inline std::string text(const std::string &header, const std::string &target = windowsTarget)
{
    return generate(header, {}, target).text;
}

} // namespace tsbindgen_test

#endif // TSCLANGIMPORTER_UNITTESTS_IMPORTERTESTHELPER_H
```

`tslang/unittests/TsClangImporter/TypeMappingTest.cpp` — one test per row of the spec's mapping table, plus the cycle rule, macro redefinition and per-target struct layout:

```cpp
// One test per row of the spec's type-mapping table.

#include "ImporterTestHelper.h"

#include "gmock/gmock.h"

using namespace tsbindgen_test;
using testing::HasSubstr;
using testing::Not;

TEST(TypeMapping, IntegersHaveTheTargetsExactWidth)
{
    auto header = "#include <stdint.h>\n"
                  "int f(signed char a, unsigned short b, long c, unsigned long long d, int8_t e, uint32_t g);\n";

    EXPECT_THAT(text(header, windowsTarget),
                HasSubstr("declare function f(a: s8, b: u16, c: s32, d: u64, e: s8, g: u32): s32;"));
    // C long is 32-bit on Windows and 64-bit on Linux: never `long` in the output
    EXPECT_THAT(text(header, linuxTarget),
                HasSubstr("declare function f(a: s8, b: u16, c: s64, d: u64, e: s8, g: u32): s32;"));
}

TEST(TypeMapping, StructFieldsHaveTheTargetsWidths)
{
    auto header = "struct Sized { long a; char b; };\n";
    EXPECT_THAT(text(header, windowsTarget), HasSubstr("type Sized = [a: s32, b: s8];"));
    EXPECT_THAT(text(header, linuxTarget), HasSubstr("type Sized = [a: s64, b: s8];"));
}

TEST(TypeMapping, SizeTypesAreIndex)
{
    EXPECT_THAT(text("#include <stddef.h>\n#include <stdint.h>\n"
                     "size_t f(ptrdiff_t a, intptr_t b, uintptr_t c);\n"),
                HasSubstr("declare function f(a: index, b: index, c: index): index;"));
}

TEST(TypeMapping, FloatsAndBool)
{
    EXPECT_THAT(text("#include <stdbool.h>\nbool f(float a, double b, _Bool c);\n"),
                HasSubstr("declare function f(a: f32, b: f64, c: boolean): boolean;"));
}

TEST(TypeMapping, CharPointersAreStrings)
{
    EXPECT_THAT(text("const char *f(char *a, const char *b, char **c);\n"),
                HasSubstr("declare function f(a: string, b: string, c: Reference<string>): string;"));
}

TEST(TypeMapping, PointerToScalarOrStructIsReference)
{
    auto out = text("struct Point { int x; int y; };\nvoid f(int *a, struct Point *p, double *d);\n");
    EXPECT_THAT(out, HasSubstr("type Point = [x: s32, y: s32];"));
    EXPECT_THAT(out, HasSubstr("declare function f(a: Reference<s32>, p: Reference<Point>, d: Reference<f64>): void;"));
}

TEST(TypeMapping, VoidPointerAndIncompleteStructAreOpaque)
{
    auto out = text("typedef struct Counter Counter;\nvoid *f(void *a, Counter *c, struct Other *o);\n");
    EXPECT_THAT(out, HasSubstr("type Counter = Opaque;"));
    EXPECT_THAT(out, HasSubstr("type Other = Opaque;"));
    EXPECT_THAT(out, HasSubstr("declare function f(a: Opaque, c: Counter, o: Other): Opaque;"));
}

TEST(TypeMapping, PointerToCxxClassIsOpaque)
{
    auto out = generate("class Engine { public: virtual ~Engine(); int power; };\n"
                        "extern \"C\" int engine_power(Engine *e);\n",
                        {}, windowsTarget, {}, "c++")
                   .text;
    EXPECT_THAT(out, HasSubstr("type Engine = Opaque;"));
    EXPECT_THAT(out, HasSubstr("declare function engine_power(e: Engine): s32;"));
}

TEST(TypeMapping, FunctionPointerIsOpaqueWithItsSignature)
{
    auto out = text("int apply(int (*fn)(int, const char *), int v);\ntypedef void (*callback_t)(void *);\n"
                    "void on(callback_t cb);\n");
    EXPECT_THAT(out, HasSubstr("declare function apply(fn: Opaque /* (p0: s32, p1: string) => s32 */, v: s32): s32;"));
    EXPECT_THAT(out, HasSubstr("type callback_t = Opaque; // (p0: Opaque) => void"));
    EXPECT_THAT(out, HasSubstr("declare function on(cb: callback_t): void;"));
    EXPECT_THAT(out, HasSubstr("pass a function without captures, as `fn as Opaque`"));
}

TEST(TypeMapping, CompleteStructIsANamedTuple)
{
    EXPECT_THAT(text("#include <stdint.h>\ntypedef struct { int8_t a; int32_t b; int64_t c; } Mixed;\n"
                     "struct Node { int value; struct Node *next; };\n"),
                testing::AllOf(HasSubstr("type Mixed = [a: s8, b: s32, c: s64];"),
                               HasSubstr("type Node = [value: s32, next: Opaque /* Reference<Node>: a type cannot refer to itself */];")));
}

// tslang cannot declare a type that refers to itself (the compiler dies on one), so every pointer
// field that leads back to its own struct - directly, through another struct, or through a typedef
// - is Opaque
TEST(TypeMapping, TypeCyclesBecomeOpaque)
{
    auto out = text("struct A { struct B *b; int x; };\nstruct B { struct A *a; };\n"
                    "typedef struct List *ListPtr;\nstruct List { ListPtr next; int v; };\n"
                    "void walk(struct A *a, ListPtr l);\n");
    EXPECT_THAT(out, HasSubstr("type A = [b: Opaque /* Reference<B>: a type cannot refer to itself */, x: s32];"));
    EXPECT_THAT(out, HasSubstr("type B = [a: Opaque /* Reference<A>: a type cannot refer to itself */];"));
    EXPECT_THAT(out, HasSubstr("type List = [next: Opaque /* Reference<List>: a type cannot refer to itself */, v: s32];"));
    EXPECT_THAT(out, HasSubstr("type ListPtr = Reference<List>;"));
    EXPECT_THAT(out, HasSubstr("declare function walk(a: Reference<A>, l: ListPtr): void;"));
    EXPECT_THAT(out, Not(HasSubstr("fn as Opaque")));
}

TEST(TypeMapping, CEnumIsATsEnum)
{
    auto out = text("enum Color { RED, GREEN = 5, BLUE };\nint f(enum Color c);\n");
    EXPECT_THAT(out, HasSubstr("enum Color { RED = 0, GREEN = 5, BLUE = 6 }"));
    EXPECT_THAT(out, HasSubstr("declare function f(c: Color): s32;"));
}

TEST(TypeMapping, FixedNonIntEnumIsATypeAndConstants)
{
    auto out = generate("#include <stdint.h>\n"
                        "enum Mode : uint8_t { Off = 0, On = 1 };\n"
                        "enum class Level : int16_t { Low = -1, High = 2 };\n"
                        "extern \"C\" void set(Mode m, Level l);\n",
                        {}, windowsTarget, {}, "c++")
                   .text;
    EXPECT_THAT(out, HasSubstr("type Mode = u8;\nconst Off = 0;\nconst On = 1;\n"));
    EXPECT_THAT(out, HasSubstr("type Level = s16;\nconst Level_Low = -1;\nconst Level_High = 2;\n"));
    EXPECT_THAT(out, HasSubstr("declare function set(m: Mode, l: Level): void;"));
}

TEST(TypeMapping, VarargsKeepTheFixedParameters)
{
    EXPECT_THAT(text("int print_all(const char *format, ...);\n"),
                HasSubstr("@varargs declare function print_all(format: string): s32;"));
}

TEST(TypeMapping, LiteralMacrosAreConstants)
{
    auto out = text("#define ANSWER 42\n#define NEGATIVE (-7)\n#define MASK 0xFFu\n#define RATIO 1.5\n"
                    "#define WHOLE 2.0\n#define NAME \"fix\\\"ture\"\n");
    EXPECT_THAT(out, HasSubstr("const ANSWER = 42;"));
    EXPECT_THAT(out, HasSubstr("const NEGATIVE = -7;"));
    EXPECT_THAT(out, HasSubstr("const MASK = 255;"));
    EXPECT_THAT(out, HasSubstr("const RATIO = 1.5;"));
    EXPECT_THAT(out, HasSubstr("const WHOLE = 2.0;"));
    EXPECT_THAT(out, HasSubstr("const NAME = \"fix\\\"ture\";"));
}

TEST(TypeMapping, MacrosFollowUndefAndRedefinition)
{
    auto out = text("#define GONE 1\n#undef GONE\n#define LEVEL 1\n#undef LEVEL\n#define LEVEL 2\n");
    EXPECT_THAT(out, Not(HasSubstr("GONE")));
    EXPECT_THAT(out, HasSubstr("const LEVEL = 2;"));
    EXPECT_THAT(out, Not(HasSubstr("const LEVEL = 1;")));
}

TEST(TypeMapping, TypedefIsATypeAliasUnlessItNamesTheStruct)
{
    auto out = text("typedef struct Point { int x; } Point;\ntypedef struct Point Pt;\ntypedef unsigned int handle_t;\n"
                    "void f(Pt *p, handle_t h);\n");
    EXPECT_THAT(out, HasSubstr("type Point = [x: s32];"));
    EXPECT_THAT(out, Not(HasSubstr("type Point = Point")));
    EXPECT_THAT(out, HasSubstr("type Pt = Point;"));
    EXPECT_THAT(out, HasSubstr("type handle_t = u32;"));
    EXPECT_THAT(out, HasSubstr("declare function f(p: Reference<Point>, h: handle_t): void;"));
}

TEST(TypeMapping, ParameterNames)
{
    EXPECT_THAT(text("void f(int, int delete, int function, int value);\n"),
                HasSubstr("declare function f(p0: s32, delete_: s32, function_: s32, value: s32): void;"));
}
```

`tslang/unittests/TsClangImporter/ParseErrorTest.cpp`:

```cpp
// A header clang rejects, or a command line clang rejects, is an error - never an empty success.

#include "ImporterTestHelper.h"

#include "gmock/gmock.h"

namespace
{

llvm::Expected<tsbindgen::HeaderModel> parse(const std::string &code, std::vector<std::string> extraArgs = {})
{
    tsbindgen::ParseInput input;
    input.code = code;
    input.fileName = "input.h";
    input.args = {"-x", "c", "--target=x86_64-pc-windows-msvc", "-ffreestanding", "-nostdlibinc", "-resource-dir",
                  TSBINDGEN_TEST_RESOURCE_DIR};
    input.args.insert(input.args.end(), extraArgs.begin(), extraArgs.end());
    return tsbindgen::parseHeader(input);
}

} // namespace

TEST(ParseError, SyntaxError)
{
    auto model = parse("int broken(\n");
    ASSERT_FALSE(static_cast<bool>(model));
    EXPECT_THAT(llvm::toString(model.takeError()), testing::HasSubstr("clang could not parse 'input.h'"));
}

TEST(ParseError, MissingInclude)
{
    auto model = parse("#include \"nowhere.h\"\nint f(void);\n");
    ASSERT_FALSE(static_cast<bool>(model));
    llvm::consumeError(model.takeError());
}

TEST(ParseError, ArgumentClangRejects)
{
    auto model = parse("int f(void);\n", {"--no-such-clang-flag"});
    ASSERT_FALSE(static_cast<bool>(model));
    llvm::consumeError(model.takeError());
}

TEST(ParseError, WarningsAreNotErrors)
{
    auto model = parse("int f(struct Late *l);\n");
    ASSERT_TRUE(static_cast<bool>(model)) << llvm::toString(model.takeError());
}
```

`tslang/unittests/TsClangImporter/CMakeLists.txt`:

```cmake
add_mlir_unittest(TsClangImporterTests
  BindingModelTest.cpp
  TypeMappingTest.cpp
  ParseErrorTest.cpp
)

target_link_libraries(TsClangImporterTests
  PRIVATE
  TsClangImporter
)

# clang's own headers (stdint.h, stddef.h) for the hermetic -ffreestanding -nostdlibinc parses
target_compile_definitions(TsClangImporterTests PRIVATE
  TSBINDGEN_TEST_RESOURCE_DIR="${LLVM_LIBRARY_DIR}/clang/${LLVM_VERSION_MAJOR}")
```

- [ ] **Step 2: Run them to verify they fail**

Run: `cmake . && cmake --build . --config Release -j 16 --target TsClangImporterTests`
Expected: FAIL — `fatal error C1083: Cannot open include file: 'TsClangImporter/BindingPrinter.h'`.

- [ ] **Step 3: Write the type mapper**

`tslang/include/TsClangImporter/TypeMapper.h`:

```cpp
#ifndef TSCLANGIMPORTER_TYPEMAPPER_H
#define TSCLANGIMPORTER_TYPEMAPPER_H

#include "TsClangImporter/BindingModel.h"

#include "clang/AST/ASTContext.h"
#include "clang/AST/Decl.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/StringSet.h"

#include <utility>
#include <vector>

namespace tsbindgen
{

// C type -> tslang type, for one translation unit. Widths come from the ASTContext, so they are the
// target's: C `long` is i32 for x86_64-pc-windows-msvc and i64 for x86_64-linux-gnu.
class TypeMapper
{
  public:
    enum class Use
    {
        Parameter,
        Result,
        Field,
        Alias // the right-hand side of a typedef
    };

    explicit TypeMapper(clang::ASTContext &context) : context(context)
    {
    }

    TsType map(clang::QualType type, Use use);

    // Why a complete struct cannot be a named tuple ("" if it can). Cached; a struct that refers to
    // itself through a pointer counts as mappable while it is being checked.
    std::string recordSkipReason(const clang::RecordDecl *record);

    // A complete struct's fields. A pointer field that leads back to this struct is Opaque:
    // tslang cannot declare a type that refers to itself.
    std::vector<Field> mapFields(const clang::RecordDecl *definition);

    // The struct/enum's own name, or the typedef that names it when it has none ("" for neither).
    static std::string tagName(const clang::TagDecl *tag);
    static std::string recordKey(const clang::RecordDecl *record);
    static std::string enumKey(const clang::EnumDecl *enumDecl);
    static std::string typedefKey(const clang::TypedefNameDecl *typedefDecl);

    // `typedef struct S S;` and `typedef struct { ... } S;` name the struct itself: no alias is emitted.
    static bool namesItsTag(const clang::TypedefNameDecl *typedefDecl);

    // A C enum is a TS enum only when it is int-sized and not given a fixed underlying type other
    // than int; anything else is a type alias plus constants.
    bool isPlainEnum(const clang::EnumDecl *enumDecl);

    // Every struct and enum a mapped type has named, by key. The AST traversal does not reach a
    // struct first named inside a prototype (`void f(struct S *s);`), so the parser adds these.
    const std::vector<std::pair<std::string, const clang::TagDecl *>> &referencedTags() const
    {
        return referenced;
    }

  private:
    std::string referTo(const clang::TagDecl *tag);
    TsType mapPointer(const clang::PointerType *pointer, Use use);
    TsType mapBuiltin(clang::QualType canonical, Use use);
    std::string signatureComment(const clang::FunctionProtoType *function);
    bool isSystem(const clang::Decl *decl);
    const clang::RecordDecl *recordBehind(clang::QualType type);
    bool leadsToOwner(clang::QualType type);

    clang::ASTContext &context;
    llvm::DenseMap<const clang::RecordDecl *, std::string> recordReasons;
    std::vector<std::pair<std::string, const clang::TagDecl *>> referenced;
    llvm::StringSet<> referencedKeys;
    std::vector<const clang::RecordDecl *> owners; // structs whose fields are being mapped
};

} // namespace tsbindgen

#endif // TSCLANGIMPORTER_TYPEMAPPER_H
```

`tslang/lib/TsClangImporter/TypeMapper.cpp`:

```cpp
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
```

- [ ] **Step 4: Write the parser**

It calls `clang::tooling::ToolInvocation` itself instead of `runToolOnCodeWithArgs`, with a `TextDiagnosticPrinter` whose error count covers the command line too — see "Deviations". A struct first named inside a prototype (`void f(struct S *s);`) is not reached by the AST traversal; `finish()` adds every struct and enum the mapper referred to.

`tslang/include/TsClangImporter/HeaderParser.h`:

```cpp
#ifndef TSCLANGIMPORTER_HEADERPARSER_H
#define TSCLANGIMPORTER_HEADERPARSER_H

#include "TsClangImporter/BindingModel.h"

#include "llvm/Support/Error.h"

#include <string>
#include <utility>
#include <vector>

namespace tsbindgen
{

struct ParseInput
{
    std::string code;             // the input file's contents
    std::string fileName;         // its path; `#include "x.h"` resolves next to it
    std::vector<std::string> args; // clang arguments: -x, --target=, -resource-dir, -I, -D, ...
    // extra files visible to clang only (tests use them in place of real headers)
    std::vector<std::pair<std::string, std::string>> virtualFiles;
};

// Runs clang over the input and maps what it declares. clang's diagnostics go to stderr as clang
// prints them; an error means clang reported an error.
llvm::Expected<HeaderModel> parseHeader(const ParseInput &input);

} // namespace tsbindgen

#endif // TSCLANGIMPORTER_HEADERPARSER_H
```

`tslang/lib/TsClangImporter/HeaderParser.cpp`:

```cpp
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
        clang::StringLiteralParser literal(tokens, preprocessor);
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

    clang::NumericLiteralParser literal(spelling, tokens.front().getLocation(), preprocessor.getSourceManager(),
                                        preprocessor.getLangOpts(), preprocessor.getTargetInfo(),
                                        preprocessor.getDiagnostics());
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
        if (record->isImplicit() || !atFileScope(record))
        {
            return true;
        }

        if (auto *cxxRecord = llvm::dyn_cast<clang::CXXRecordDecl>(record);
            cxxRecord && (cxxRecord->getDescribedClassTemplate() ||
                          llvm::isa<clang::ClassTemplateSpecializationDecl>(cxxRecord)))
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
        if (isSystem(typedefDecl) || !atFileScope(typedefDecl) || TypeMapper::namesItsTag(typedefDecl))
        {
            return true;
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
```

- [ ] **Step 5: Write the printer**

Output order is fixed: constants, then types in declaration order, then functions — so a function never names a type declared after it.

`tslang/include/TsClangImporter/BindingPrinter.h`:

```cpp
#ifndef TSCLANGIMPORTER_BINDINGPRINTER_H
#define TSCLANGIMPORTER_BINDINGPRINTER_H

#include "TsClangImporter/BindingModel.h"

#include "llvm/Support/Error.h"

#include <string>
#include <vector>

namespace tsbindgen
{

struct PrintOptions
{
    // A declaration is selected if its C name matches one of these globs, or, with none, if the
    // input file itself declares it. Types a selected declaration uses are emitted too.
    std::vector<std::string> filters;
    std::string namespaceName; // --namespace: wrap in `namespace N { export ... }`, with @linkname
    std::string stripPrefix;   // --strip-prefix: removed from the TS names (needs a namespace)
    std::vector<std::string> headerLines; // printed first, each as a `//` comment
};

struct PrintResult
{
    std::string text;
    std::vector<std::string> warnings; // one per skipped declaration, and per name kept unstripped
};

// Fails only for a filter that is not a valid glob.
llvm::Expected<PrintResult> printBindings(const HeaderModel &model, const PrintOptions &options);

} // namespace tsbindgen

#endif // TSCLANGIMPORTER_BINDINGPRINTER_H
```

`tslang/lib/TsClangImporter/BindingPrinter.cpp`:

```cpp
#include "TsClangImporter/BindingPrinter.h"

#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/GlobPattern.h"

#include <cctype>

namespace tsbindgen
{

namespace
{

// The selected declarations: every seed, then whatever a non-skipped selected declaration uses.
std::vector<bool> selectDecls(const HeaderModel &model, const std::vector<llvm::GlobPattern> &globs)
{
    llvm::StringMap<size_t> byKey;
    for (size_t index = 0; index < model.decls.size(); ++index)
    {
        byKey[model.decls[index].key] = index;
    }

    std::vector<bool> selected(model.decls.size(), false);
    std::vector<size_t> work;
    auto mark = [&](size_t index) {
        if (!selected[index])
        {
            selected[index] = true;
            work.push_back(index);
        }
    };

    for (size_t index = 0; index < model.decls.size(); ++index)
    {
        auto &decl = model.decls[index];
        auto seed = globs.empty() ? decl.inMainFile
                                  : llvm::any_of(globs, [&](const llvm::GlobPattern &glob) { return glob.match(decl.name); });
        if (seed)
        {
            mark(index);
        }
    }

    while (!work.empty())
    {
        auto &decl = model.decls[work.back()];
        work.pop_back();
        if (!decl.skipReason.empty())
        {
            continue; // printed as Opaque or not at all: it uses nothing
        }

        std::vector<std::string> keys;
        for (auto &field : decl.fields)
        {
            collectKeys(field.type, keys);
        }

        collectKeys(decl.type, keys);
        for (auto &key : keys)
        {
            auto found = byKey.find(key);
            if (found != byKey.end())
            {
                mark(found->second);
            }
        }
    }

    return selected;
}

int group(DeclKind kind)
{
    switch (kind)
    {
    case DeclKind::Macro:
        return 0;
    case DeclKind::Function:
        return 2;
    default:
        return 1; // types, in declaration order, before any function uses them
    }
}

} // namespace

llvm::Expected<PrintResult> printBindings(const HeaderModel &model, const PrintOptions &options)
{
    std::vector<llvm::GlobPattern> globs;
    for (auto &filter : options.filters)
    {
        auto glob = llvm::GlobPattern::create(filter);
        if (!glob)
        {
            return glob.takeError();
        }

        globs.push_back(std::move(*glob));
    }

    auto selected = selectDecls(model, globs);
    PrintResult result;
    if (llvm::none_of(selected, [](bool isSelected) { return isSelected; }))
    {
        result.warnings.push_back(options.filters.empty()
                                      ? "nothing to emit: the input file itself declares nothing; pick "
                                        "declarations from its includes with --filter"
                                      : "nothing to emit: no declaration matches --filter");
    }

    // TS names: the C name, minus --strip-prefix unless that leaves nothing, a leading digit, or a
    // name another declaration already has
    llvm::StringMap<std::string> tsNames;
    llvm::StringMap<std::string> taken;
    for (size_t index = 0; index < model.decls.size(); ++index)
    {
        if (!selected[index])
        {
            continue;
        }

        auto &decl = model.decls[index];
        auto name = decl.name;
        llvm::StringRef stripped(name);
        if (!options.stripPrefix.empty() && stripped.consume_front(options.stripPrefix) && !stripped.empty() &&
            !std::isdigit(static_cast<unsigned char>(stripped.front())))
        {
            auto candidate = escapeReserved(stripped.str());
            if (taken.count(candidate))
            {
                result.warnings.push_back("kept '" + name + "' unstripped: '" + candidate + "' is taken by '" +
                                          taken[candidate] + "'");
            }
            else
            {
                name = candidate;
            }
        }

        taken[name] = decl.name;
        tsNames[decl.key] = name;
    }

    auto nameOf = [&](const std::string &key) {
        auto found = tsNames.find(key);
        return found != tsNames.end() ? found->second : nameFromKey(key);
    };

    auto inNamespace = !options.namespaceName.empty();
    auto indent = std::string(inNamespace ? "    " : "");
    auto exported = std::string(inNamespace ? "export " : "");
    std::string body;
    auto line = [&](const std::string &text) { body += indent + text + "\n"; };
    auto hasFunctionPointer = false;
    auto withComment = [&](const Field &field) {
        auto text = field.name + ": " + renderType(field.type, nameOf);
        if (!field.type.comment.empty())
        {
            hasFunctionPointer |= field.type.functionPointer;
            text += " /* " + field.type.comment + " */";
        }

        return text;
    };

    for (int pass = 0; pass < 3; ++pass)
    {
        for (size_t index = 0; index < model.decls.size(); ++index)
        {
            auto &decl = model.decls[index];
            if (!selected[index] || group(decl.kind) != pass)
            {
                continue;
            }

            auto tsName = nameOf(decl.key);
            if (!decl.skipReason.empty())
            {
                line("// skipped: " + decl.name + " \xE2\x80\x94 " + decl.skipReason);
                result.warnings.push_back("skipped " + decl.name + ": " + decl.skipReason);
                // a skipped struct keeps its users: a pointer to it is Opaque
                if (decl.kind == DeclKind::Struct)
                {
                    line(exported + "type " + tsName + " = Opaque;");
                }

                continue;
            }

            switch (decl.kind)
            {
            case DeclKind::Macro:
                line(exported + "const " + tsName + " = " + decl.value + ";");
                break;
            case DeclKind::OpaqueStruct:
                line(exported + "type " + tsName + " = Opaque;");
                break;
            case DeclKind::Struct: {
                std::string fields;
                for (auto &field : decl.fields)
                {
                    fields += (fields.empty() ? "" : ", ") + withComment(field);
                }

                line(exported + "type " + tsName + " = [" + fields + "];");
                break;
            }
            case DeclKind::Enum: {
                std::string members;
                for (auto &enumerator : decl.enumerators)
                {
                    members += (members.empty() ? "" : ", ") + enumerator.name + " = " + enumerator.value;
                }

                line(exported + "enum " + tsName + " { " + members + " }");
                break;
            }
            case DeclKind::FixedEnum:
                line(exported + "type " + tsName + " = " + renderType(decl.type, nameOf) + ";");
                for (auto &enumerator : decl.enumerators)
                {
                    auto constName = decl.scoped ? tsName + "_" + enumerator.name : enumerator.name;
                    line(exported + "const " + constName + " = " + enumerator.value + ";");
                }

                break;
            case DeclKind::Typedef: {
                auto text = exported + "type " + tsName + " = " + renderType(decl.type, nameOf) + ";";
                if (!decl.type.comment.empty())
                {
                    hasFunctionPointer |= decl.type.functionPointer;
                    text += " // " + decl.type.comment;
                }

                line(text);
                break;
            }
            case DeclKind::Function: {
                std::string parameters;
                for (auto &parameter : decl.fields)
                {
                    parameters += (parameters.empty() ? "" : ", ") + withComment(parameter);
                }

                std::string decorators = decl.varargs ? "@varargs " : "";
                if (inNamespace)
                {
                    decorators += "@linkname(\"" + decl.name + "\") ";
                }

                line(decorators + exported + "declare function " + tsName + "(" + parameters +
                     "): " + renderType(decl.type, nameOf) + ";");
                break;
            }
            }
        }
    }

    for (auto &headerLine : options.headerLines)
    {
        result.text += "// " + headerLine + "\n";
    }

    if (hasFunctionPointer)
    {
        result.text += "// A function pointer is Opaque: pass a function without captures, as `fn as Opaque`.\n";
    }

    if (!result.text.empty())
    {
        result.text += "\n";
    }

    if (inNamespace)
    {
        result.text += "namespace " + options.namespaceName + " {\n" + body + "}\n";
    }
    else
    {
        result.text += body;
    }

    return result;
}

} // namespace tsbindgen
```

`tslang/lib/TsClangImporter/CMakeLists.txt` (full version):

```cmake
# C header -> tslang bindings, as a library: tsbindgen is a thin command line over it, and v2's C++
# wrapper generation is meant to reuse it. It links clang's frontend; tslang itself does not.
set(LLVM_LINK_COMPONENTS
  Support
  )

add_llvm_library(TsClangImporter
  BindingModel.cpp
  TypeMapper.cpp
  HeaderParser.cpp
  BindingPrinter.cpp

  LINK_LIBS
  clangTooling
  clangFrontend
  clangAST
  clangLex
  clangBasic
  )
```

- [ ] **Step 6: Run the tests to verify they pass**

Run: `cmake . && cmake --build . --config Release -j 16 --target TsClangImporterTests && ctest -C Release -R unittest-TsClangImporterTests --output-on-failure`
Expected: `100% tests passed`; the binary itself (`unittests/TsClangImporter/Release/TsClangImporterTests.exe`) prints `[  PASSED  ] 27 tests.` — 5 `BindingModel`, 18 `TypeMapping`, 4 `ParseError`. Clang's own warning `declaration of 'struct Late' will not be visible outside of this function` on stderr is expected (`ParseError.WarningsAreNotErrors`).

- [ ] **Step 7: Commit**

```bash
git add tslang/include/TsClangImporter tslang/lib/TsClangImporter tslang/unittests/TsClangImporter
git commit -m "tsbindgen: parse C headers with clang and print tslang bindings

Signed C integers map to sN (tslang's iN are signless), and a struct
pointer that leads back to its own struct is Opaque: tslang cannot
declare a self-referencing type.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 3: Pin the skip, selection and naming rules

The code is Task 2's; these tests pin the spec's "Skipped declarations", "What gets emitted" and "Naming" sections so a reviewer can check each rule against the spec independently of the mapping table. They are expected to pass as soon as they build; a failure means Task 2's code deviates from the spec — fix the code, not the test.

**Files:**
- Create: `tslang/unittests/TsClangImporter/SkipTest.cpp`, `SelectionTest.cpp`, `NamingTest.cpp`
- Modify: `tslang/unittests/TsClangImporter/CMakeLists.txt`

**Interfaces:**
- Consumes: `generate(header, PrintOptions, target, files, language)` from `ImporterTestHelper.h`; `PrintOptions`, `PrintResult` from Task 2.

- [ ] **Step 1: Write the tests**

`tslang/unittests/TsClangImporter/SkipTest.cpp`:

```cpp
// Every skip reason in the spec: the declaration is left out with a `// skipped:` line and a
// warning, generation carries on, and a skipped struct is still Opaque for its pointer users.

#include "ImporterTestHelper.h"

#include "gmock/gmock.h"

using namespace tsbindgen_test;
using testing::Contains;
using testing::HasSubstr;
using testing::Not;

TEST(Skip, UnionIsSkippedButItsPointerUsersStay)
{
    auto out = generate("union Value { int i; float f; };\nvoid set(union Value *v);\nvoid pass(union Value v);\n");
    EXPECT_THAT(out.text, HasSubstr("// skipped: Value \xE2\x80\x94 union\ntype Value = Opaque;"));
    EXPECT_THAT(out.text, HasSubstr("declare function set(v: Value): void;"));
    EXPECT_THAT(out.text, HasSubstr("// skipped: pass \xE2\x80\x94 parameter 'v': union passed by value"));
    EXPECT_THAT(out.warnings, Contains("skipped Value: union"));
}

TEST(Skip, StructWithABitfieldOrAnUnmappableField)
{
    auto out = text("struct Flags { unsigned a : 1; };\nstruct Wide { long double x; };\n"
                    "struct Grid { int cells[4]; };\n");
    EXPECT_THAT(out, HasSubstr("// skipped: Flags \xE2\x80\x94 bitfield 'a'"));
    EXPECT_THAT(out, HasSubstr("// skipped: Wide \xE2\x80\x94 field 'x': long double"));
    EXPECT_THAT(out, HasSubstr("// skipped: Grid \xE2\x80\x94 field 'cells': array"));
}

TEST(Skip, PackedStruct)
{
    EXPECT_THAT(text("struct __attribute__((packed)) Packed { char c; int i; };\n"),
                HasSubstr("// skipped: Packed \xE2\x80\x94 packed or over-aligned layout"));
    // how real headers usually pack
    EXPECT_THAT(text("#pragma pack(push, 1)\nstruct Wire { char c; int i; };\n#pragma pack(pop)\n"
                     "struct Natural { char c; int i; };\n"),
                testing::AllOf(HasSubstr("// skipped: Wire \xE2\x80\x94 packed or over-aligned layout"),
                               HasSubstr("type Natural = [c: s8, i: s32];")));
}

TEST(Skip, LongDouble)
{
    EXPECT_THAT(text("long double f(long double x);\n"),
                HasSubstr("// skipped: f \xE2\x80\x94 parameter 'x': long double"));
}

TEST(Skip, StructByValue)
{
    auto out = text("struct Point { int x; int y; };\nstruct Point make(int x);\nint sum(struct Point p);\n");
    EXPECT_THAT(out, HasSubstr("type Point = [x: s32, y: s32];"));
    EXPECT_THAT(out, HasSubstr("// skipped: make \xE2\x80\x94 result: struct passed by value"));
    EXPECT_THAT(out, HasSubstr("// skipped: sum \xE2\x80\x94 parameter 'p': struct passed by value"));
}

TEST(Skip, FunctionLikeAndNonLiteralMacros)
{
    auto out = text("#define MAX(a, b) ((a) > (b) ? (a) : (b))\n#define FLAGS (1 << 3)\n#define GUARD_H\n");
    EXPECT_THAT(out, HasSubstr("// skipped: MAX \xE2\x80\x94 function-like macro"));
    EXPECT_THAT(out, HasSubstr("// skipped: FLAGS \xE2\x80\x94 macro is not a literal"));
    EXPECT_THAT(out, Not(HasSubstr("GUARD_H")));
}

TEST(Skip, StaticInlineFunction)
{
    auto out = text("static inline int twice(int x) { return x * 2; }\nstatic int hidden(void);\n");
    EXPECT_THAT(out, HasSubstr("// skipped: twice \xE2\x80\x94 static inline function: no symbol to link against"));
    EXPECT_THAT(out, HasSubstr("// skipped: hidden \xE2\x80\x94 static function: no symbol to link against"));
}

TEST(Skip, OtherTypes)
{
    auto out = text("_Complex double f(void);\nint g();\n");
    EXPECT_THAT(out, HasSubstr("// skipped: f \xE2\x80\x94 result: unsupported type '_Complex double'"));
    EXPECT_THAT(out, HasSubstr("// skipped: g \xE2\x80\x94 function without a prototype"));
}

TEST(Skip, ASkipDoesNotStopGeneration)
{
    auto out = text("union U { int i; };\nint before(void);\nvoid broken(union U u);\nint after(void);\n");
    EXPECT_THAT(out, HasSubstr("declare function before(): s32;"));
    EXPECT_THAT(out, HasSubstr("declare function after(): s32;"));
}
```

`tslang/unittests/TsClangImporter/SelectionTest.cpp`:

```cpp
// What gets emitted: without --filter, what the input file declares; with it, what matches; and in
// both cases the types those use, wherever they are declared.

#include "ImporterTestHelper.h"

#include "gmock/gmock.h"

using namespace tsbindgen_test;
using testing::HasSubstr;
using testing::Not;

namespace
{

const std::vector<std::pair<std::string, std::string>> files = {
    {"other.h", "typedef struct Shared { int id; } Shared;\nint other_unused(void);\n#define OTHER_VALUE 1\n"},
    {"sys/system.h", "struct sys_stat { int size; };\nint sys_call(struct sys_stat *s);\n#define SYS_VALUE 2\n"},
};

} // namespace

TEST(Selection, ByDefaultOnlyTheInputFilesDeclarations)
{
    auto out = generate("#include \"other.h\"\n#include <system.h>\nint mine(void);\n", {}, windowsTarget, files).text;
    EXPECT_THAT(out, HasSubstr("declare function mine(): s32;"));
    EXPECT_THAT(out, Not(HasSubstr("other_unused")));
    EXPECT_THAT(out, Not(HasSubstr("OTHER_VALUE")));
    EXPECT_THAT(out, Not(HasSubstr("sys_call")));
    EXPECT_THAT(out, Not(HasSubstr("SYS_VALUE")));
    EXPECT_THAT(out, Not(HasSubstr("Shared")));
}

TEST(Selection, UsedTypesArePulledInFromAnyHeader)
{
    auto out = generate("#include \"other.h\"\n#include <system.h>\nint use(Shared *s, struct sys_stat *st);\n", {},
                        windowsTarget, files)
                   .text;
    EXPECT_THAT(out, HasSubstr("type Shared = [id: s32];"));
    EXPECT_THAT(out, HasSubstr("type sys_stat = [size: s32];"));
    EXPECT_THAT(out, HasSubstr("declare function use(s: Reference<Shared>, st: Reference<sys_stat>): s32;"));
    EXPECT_THAT(out, Not(HasSubstr("other_unused")));
}

TEST(Selection, FilterSelectsByCNameAnywhere)
{
    tsbindgen::PrintOptions options;
    options.filters = {"sys_*", "mine"};
    auto out = generate("#include \"other.h\"\n#include <system.h>\nint mine(void);\nint yours(void);\n", options,
                        windowsTarget, files)
                   .text;
    EXPECT_THAT(out, HasSubstr("declare function mine(): s32;"));
    EXPECT_THAT(out, HasSubstr("declare function sys_call(s: Reference<sys_stat>): s32;"));
    EXPECT_THAT(out, HasSubstr("type sys_stat = [size: s32];"));
    EXPECT_THAT(out, Not(HasSubstr("yours")));
    EXPECT_THAT(out, Not(HasSubstr("SYS_VALUE")));
}

TEST(Selection, NothingSelectedIsAWarning)
{
    auto out = generate("#include \"other.h\"\n", {}, windowsTarget, files);
    EXPECT_EQ(out.text, "");
    ASSERT_EQ(out.warnings.size(), 1u);
    EXPECT_THAT(out.warnings.front(), HasSubstr("nothing to emit"));

    tsbindgen::PrintOptions options;
    options.filters = {"no_such_*"};
    EXPECT_THAT(generate("int f(void);\n", options).warnings, testing::Contains(HasSubstr("no declaration matches")));
}

TEST(Selection, TypesComeBeforeTheFunctionsThatUseThem)
{
    auto out = text("int first(struct Late *l);\nstruct Late { int x; };\n");
    auto type = out.find("type Late");
    auto function = out.find("declare function first");
    ASSERT_NE(type, std::string::npos);
    ASSERT_NE(function, std::string::npos);
    EXPECT_LT(type, function);
}
```

`tslang/unittests/TsClangImporter/NamingTest.cpp`:

```cpp
// --namespace wraps the output and binds each function by its C name with @linkname;
// --strip-prefix shortens the TS names but never the symbol.

#include "ImporterTestHelper.h"

#include "gmock/gmock.h"

using namespace tsbindgen_test;
using testing::Contains;
using testing::HasSubstr;

TEST(Naming, CNamesAreKept)
{
    auto out = text("#define FX_ANSWER 42\nint fx_add(int a, int b);\n");
    EXPECT_EQ(out, "const FX_ANSWER = 42;\ndeclare function fx_add(a: s32, b: s32): s32;\n");
}

TEST(Naming, NamespaceExportsEverythingAndLinksByCName)
{
    tsbindgen::PrintOptions options;
    options.namespaceName = "Fx";
    auto out = generate("#define FX_ANSWER 42\nstruct fx_point { int x; };\nint fx_add(int a, int b);\n"
                        "int fx_sum(int n, ...);\n",
                        options)
                   .text;
    EXPECT_EQ(out, "namespace Fx {\n"
                   "    export const FX_ANSWER = 42;\n"
                   "    export type fx_point = [x: s32];\n"
                   "    @linkname(\"fx_add\") export declare function fx_add(a: s32, b: s32): s32;\n"
                   "    @varargs @linkname(\"fx_sum\") export declare function fx_sum(n: s32): s32;\n"
                   "}\n");
}

TEST(Naming, StripPrefixRenamesDeclarationsAndTheirUses)
{
    tsbindgen::PrintOptions options;
    options.namespaceName = "Fx";
    options.stripPrefix = "fx_";
    auto out = generate("struct fx_point { int x; };\nint fx_len(struct fx_point *p);\nint other(void);\n", options)
                   .text;
    EXPECT_THAT(out, HasSubstr("    export type point = [x: s32];\n"));
    EXPECT_THAT(out, HasSubstr("    @linkname(\"fx_len\") export declare function len(p: Reference<point>): s32;\n"));
    EXPECT_THAT(out, HasSubstr("    @linkname(\"other\") export declare function other(): s32;\n"));
}

TEST(Naming, StripPrefixKeepsTheCNameWhenStrippingWouldBreakIt)
{
    tsbindgen::PrintOptions options;
    options.namespaceName = "Fx";
    options.stripPrefix = "fx_";
    auto out = generate("int len(void);\nint fx_len(void);\nint fx_2d(void);\nint fx_delete(void);\n", options);
    EXPECT_THAT(out.text, HasSubstr("@linkname(\"len\") export declare function len(): s32;"));
    EXPECT_THAT(out.text, HasSubstr("@linkname(\"fx_len\") export declare function fx_len(): s32;"));
    EXPECT_THAT(out.text, HasSubstr("@linkname(\"fx_2d\") export declare function fx_2d(): s32;"));
    EXPECT_THAT(out.text, HasSubstr("@linkname(\"fx_delete\") export declare function delete_(): s32;"));
    EXPECT_THAT(out.warnings, Contains("kept 'fx_len' unstripped: 'len' is taken by 'len'"));
}

TEST(Naming, HeaderLinesComeFirst)
{
    tsbindgen::PrintOptions options;
    options.headerLines = {"Generated by tsbindgen dev (clang 22.1.8); do not edit.", "tsbindgen input.h"};
    EXPECT_EQ(generate("int f(void);\n", options).text,
              "// Generated by tsbindgen dev (clang 22.1.8); do not edit.\n"
              "// tsbindgen input.h\n"
              "\n"
              "declare function f(): s32;\n");
}
```

`tslang/unittests/TsClangImporter/CMakeLists.txt` (final version):

```cmake
add_mlir_unittest(TsClangImporterTests
  BindingModelTest.cpp
  TypeMappingTest.cpp
  SkipTest.cpp
  SelectionTest.cpp
  NamingTest.cpp
  ParseErrorTest.cpp
)

target_link_libraries(TsClangImporterTests
  PRIVATE
  TsClangImporter
)

# clang's own headers (stdint.h, stddef.h) for the hermetic -ffreestanding -nostdlibinc parses
target_compile_definitions(TsClangImporterTests PRIVATE
  TSBINDGEN_TEST_RESOURCE_DIR="${LLVM_LIBRARY_DIR}/clang/${LLVM_VERSION_MAJOR}")
```

- [ ] **Step 2: Run them**

Run: `cmake . && cmake --build . --config Release -j 16 --target TsClangImporterTests && ctest -C Release -R unittest-TsClangImporterTests --output-on-failure`
Expected: `100% tests passed`; the binary itself prints `[  PASSED  ] 46 tests.` — Task 2's 27 plus 9 `Skip`, 5 `Selection`, 5 `Naming`.

- [ ] **Step 3: Commit**

```bash
git add tslang/unittests/TsClangImporter
git commit -m "tsbindgen: tests for skips, selection and naming

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 4: The `tsbindgen` command line

**Files:**
- Create: `tslang/tsbindgen/tsbindgen.cpp`, `tslang/tsbindgen/CMakeLists.txt`
- Modify: `tslang/CMakeLists.txt` (after `add_subdirectory(tslang)`)
- Create: `tslang/test/bindgen/run-cli-test.cmake`
- Create: `tslang/test/bindgen/fixture.h` (Task 5 uses it too; written here because the CLI test reads it)
- Modify: `tslang/test/tester/CMakeLists.txt` (before `set(TSLANG_OWNERSHIP_SHARDS 8)`)

**Interfaces:**
- Consumes: `parseHeader`, `printBindings` (Task 2).
- Produces: `bin/tsbindgen.exe` (`bin/tsbindgen` on Linux), next to `tslang`. Command line per the spec, with `-o` defaulting to stdout. `--version` prints `tsbindgen <version> (clang <CLANG_VERSION_STRING>)`.

- [ ] **Step 1: Write the fixture header**

`tslang/test/bindgen/fixture.h` — one of each thing v1 binds, and one of each skip that must not break the rest. `FIXTURE_API` exports the functions from the shared library Task 5 builds; without it a Windows DLL exports nothing and the JIT reports `Symbols not found`.

```c
// One of each thing tsbindgen v1 binds, for test/bindgen/run-bindgen-test.cmake. bindgen_test.ts
// calls every function through the generated bindings; expected.txt is what it must print.

#ifndef FIXTURE_H
#define FIXTURE_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#if defined(_WIN32)
#define FIXTURE_API __declspec(dllexport)
#else
#define FIXTURE_API __attribute__((visibility("default")))
#endif

#define FIXTURE_ANSWER 42
#define FIXTURE_RATIO 1.5
#define FIXTURE_NAME "fixture"

typedef struct Counter Counter; // opaque handle

typedef struct Point
{
    int32_t x;
    int32_t y;
} Point;

typedef struct Box
{
    int8_t tag;
    Point corner; // a struct inside a struct
    double scale;
    bool visible;
} Box;

typedef struct Node // points to itself: `next` becomes Opaque
{
    int32_t value;
    struct Node *next;
} Node;

enum Color
{
    RED,
    GREEN = 5,
    BLUE
};

union Value // skipped, but a pointer to it is still Opaque
{
    int32_t i;
    float f;
};

// scalars
FIXTURE_API int32_t fx_add(int32_t a, int32_t b);
FIXTURE_API double fx_scale(double v, float f);
FIXTURE_API int32_t fx_widen_s8(int8_t v);
FIXTURE_API int32_t fx_widen_u16(uint16_t v);
FIXTURE_API int8_t fx_negate_s8(int8_t v);
FIXTURE_API bool fx_not(bool b);

// strings in and out
FIXTURE_API size_t fx_len(const char *s);
FIXTURE_API const char *fx_greet(void);

// out-parameters
FIXTURE_API void fx_divmod(int32_t a, int32_t b, int32_t *quotient, int32_t *remainder);

// an opaque handle
FIXTURE_API Counter *fx_counter_new(void);
FIXTURE_API void fx_counter_inc(Counter *c);
FIXTURE_API int32_t fx_counter_get(Counter *c);
FIXTURE_API void fx_counter_free(Counter *c);

// structs by pointer, read and written by C
FIXTURE_API int32_t fx_point_sum(const Point *p);
FIXTURE_API void fx_box_fill(Box *b);
FIXTURE_API int32_t fx_list_sum(Node *head);

// an enum
FIXTURE_API int32_t fx_color_value(enum Color c);

// a callback
FIXTURE_API int32_t fx_apply(int32_t (*fn)(int32_t), int32_t v);

// varargs
FIXTURE_API int32_t fx_sum(int32_t count, ...);

// skipped declarations
FIXTURE_API void fx_value_clear(union Value *v);
static inline int32_t fx_twice_inline(int32_t x)
{
    return x * 2;
}

#endif // FIXTURE_H
```

- [ ] **Step 2: Write the failing test**

`tslang/test/bindgen/run-cli-test.cmake`:

```cmake
# tsbindgen's command line: 0 when the file was written, 1 for a parse error or I/O failure, 2 for
# bad arguments - and never 0 for a command line clang rejected.

cmake_minimum_required(VERSION 3.17.3)

foreach(var TSBINDGEN SOURCE_DIR WORK_DIR)
    if(NOT DEFINED ${var})
        message(FATAL_ERROR "${var} is required")
    endif()
endforeach()

file(REMOVE_RECURSE "${WORK_DIR}")
file(MAKE_DIRECTORY "${WORK_DIR}")
file(WRITE "${WORK_DIR}/broken.h" "int broken(\n")

# expect(<exit code> <what> <args...>) - leaves stdout + stderr in cli_output
function(expect code what)
    execute_process(COMMAND "${TSBINDGEN}" ${ARGN}
        WORKING_DIRECTORY "${WORK_DIR}"
        OUTPUT_VARIABLE out
        ERROR_VARIABLE err
        RESULT_VARIABLE status)
    if(NOT status EQUAL code)
        message(FATAL_ERROR "${what}: exit ${status}, expected ${code}\n${out}\n${err}")
    endif()
    set(cli_output "${out}${err}" PARENT_SCOPE)
    message(STATUS "${what}: exit ${code}")
endfunction()

set(header "${SOURCE_DIR}/fixture.h")

expect(0 "a header, written to a file" "${header}" -o out.ts)
if(NOT EXISTS "${WORK_DIR}/out.ts")
    message(FATAL_ERROR "exit 0, but out.ts was not written")
endif()

# `import` would load a same-named library instead, so the file says how to include it
file(READ "${WORK_DIR}/out.ts" written)
if(NOT written MATCHES "// Include with: /// <reference path=\"out\\.ts\" />")
    message(FATAL_ERROR "out.ts lacks the include hint:\n${written}")
endif()

expect(0 "a header, to stdout" "${header}" --filter fx_add)
if(NOT cli_output MATCHES "declare function fx_add\\(a: s32, b: s32\\): s32;")
    message(FATAL_ERROR "stdout lacks fx_add:\n${cli_output}")
endif()

expect(0 "--version" --version)
if(NOT cli_output MATCHES "tsbindgen .* \\(clang [0-9]+")
    message(FATAL_ERROR "--version printed:\n${cli_output}")
endif()

expect(0 "a .d.ts output warns" "${header}" -o out.d.ts)
if(NOT cli_output MATCHES "name it \\.ts")
    message(FATAL_ERROR "no .d.ts warning:\n${cli_output}")
endif()

expect(1 "a missing input" missing.h)
expect(1 "a header clang rejects" broken.h)
expect(1 "an argument clang rejects" "${header}" -- --no-such-clang-flag)
expect(1 "an unwritable output" "${header}" -o "${WORK_DIR}/no/such/dir/out.ts")

expect(2 "no input" --filter x)
expect(2 "an unknown option" "${header}" --no-such-option)
expect(2 "--strip-prefix without --namespace" "${header}" --strip-prefix fx_)
expect(2 "an invalid glob" "${header}" --filter "[")
expect(2 "a --resource-dir without clang's headers" "${header}" --resource-dir "${WORK_DIR}")
```

Register it in `tslang/test/tester/CMakeLists.txt`, immediately before `set(TSLANG_OWNERSHIP_SHARDS 8)` (Task 5 adds its own registration next to this one):

```cmake
add_test(NAME test-bindgen-cli
         COMMAND ${CMAKE_COMMAND}
                 "-DTSBINDGEN=$<TARGET_FILE:tsbindgen>"
                 "-DSOURCE_DIR=${PROJECT_SOURCE_DIR}/test/bindgen"
                 "-DWORK_DIR=${CMAKE_CURRENT_BINARY_DIR}/bindgen-cli"
                 -P "${PROJECT_SOURCE_DIR}/test/bindgen/run-cli-test.cmake")
```

- [ ] **Step 3: Run it to verify it fails**

Run: `cmake .`
Expected: FAIL — `Error evaluating generator expression: $<TARGET_FILE:tsbindgen>  No target "tsbindgen"`.

- [ ] **Step 4: Write the command line**

`tslang/tsbindgen/tsbindgen.cpp`:

```cpp
// tsbindgen: C header -> tslang bindings. See docs/superpowers/specs/2026-09-25-tsbindgen-linkname-design.md.

#include "TsClangImporter/BindingPrinter.h"
#include "TsClangImporter/HeaderParser.h"

#include "clang/Basic/Version.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/WithColor.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/TargetParser/Host.h"

#include <optional>

namespace cl = llvm::cl;

namespace
{

cl::OptionCategory category("tsbindgen options");

cl::opt<std::string> inputFile(cl::Positional, cl::desc("<input.h|.c|.cpp>"), cl::Required, cl::cat(category));
cl::list<std::string> includeDirs("I", cl::Prefix, cl::desc("Add an include directory"), cl::value_desc("dir"),
                                  cl::cat(category));
cl::list<std::string> defines("D", cl::Prefix, cl::desc("Define a macro"), cl::value_desc("name[=value]"),
                              cl::cat(category));
cl::opt<std::string> target("target", cl::desc("Target triple (default: the host)"), cl::value_desc("triple"),
                            cl::cat(category));
cl::list<std::string> filters("filter", cl::desc("Emit the declarations whose C name matches this glob"),
                              cl::value_desc("glob"), cl::cat(category));
cl::opt<std::string> namespaceName("namespace", cl::desc("Wrap the output in `namespace <N>`"),
                                   cl::value_desc("N"), cl::cat(category));
cl::opt<std::string> stripPrefix("strip-prefix", cl::desc("Remove <P> from the TS names (needs --namespace)"),
                                 cl::value_desc("P"), cl::cat(category));
cl::opt<std::string> resourceDir("resource-dir", cl::desc("clang's resource directory"), cl::value_desc("dir"),
                                 cl::cat(category));
cl::opt<std::string> outputFile("o", cl::desc("Output file (default: stdout)"), cl::value_desc("out.ts"),
                                cl::init("-"), cl::cat(category));

enum ExitCode
{
    Written = 0,
    ParseOrIOError = 1,
    BadArguments = 2
};

std::string version()
{
#ifdef TSBINDGEN_VERSION
    return TSBINDGEN_VERSION;
#else
    return "dev";
#endif
}

bool hasStddef(llvm::StringRef dir)
{
    llvm::SmallString<256> path(dir);
    llvm::sys::path::append(path, "include", "stddef.h");
    return llvm::sys::fs::exists(path);
}

// clang's own stddef.h/stdint.h live here; without them those headers do not resolve.
std::optional<std::string> findResourceDir(llvm::StringRef exePath, std::vector<std::string> &tried)
{
    auto versioned = [](llvm::StringRef base) {
        llvm::SmallString<256> path(base);
        llvm::sys::path::append(path, "lib", "clang", std::to_string(CLANG_VERSION_MAJOR));
        return std::string(path);
    };

    std::vector<std::string> candidates;
    if (!resourceDir.empty())
    {
        candidates.push_back(resourceDir);
    }
    else
    {
        auto exeDir = llvm::sys::path::parent_path(exePath);
        candidates.push_back(versioned(exeDir));                              // the release package
        candidates.push_back(versioned(llvm::sys::path::parent_path(exeDir))); // an LLVM-style install
#ifdef TSBINDGEN_CONFIGURED_RESOURCE_DIR
        candidates.push_back(TSBINDGEN_CONFIGURED_RESOURCE_DIR); // the LLVM this was built with
#endif
    }

    for (auto &candidate : candidates)
    {
        tried.push_back(candidate);
        if (hasStddef(candidate))
        {
            return candidate;
        }
    }

    return std::nullopt;
}

bool isCxx(llvm::StringRef path)
{
    auto extension = llvm::sys::path::extension(path).lower();
    return extension == ".cpp" || extension == ".cc" || extension == ".cxx" || extension == ".hpp" ||
           extension == ".hh" || extension == ".hxx";
}

} // namespace

int main(int argc, char **argv)
{
    // everything after `--` goes to clang untouched
    std::vector<const char *> ownArgs;
    std::vector<std::string> extraArgs;
    auto afterDashes = false;
    for (int index = 0; index < argc; ++index)
    {
        if (afterDashes)
        {
            extraArgs.push_back(argv[index]);
        }
        else if (index > 0 && llvm::StringRef(argv[index]) == "--")
        {
            afterDashes = true;
        }
        else
        {
            ownArgs.push_back(argv[index]);
        }
    }

    cl::HideUnrelatedOptions(category);
    cl::SetVersionPrinter([](llvm::raw_ostream &os) {
        os << "tsbindgen " << version() << " (clang " << CLANG_VERSION_STRING << ")\n";
    });

    if (!cl::ParseCommandLineOptions(static_cast<int>(ownArgs.size()), ownArgs.data(),
                                     "C header -> tslang bindings\n", &llvm::errs()))
    {
        return BadArguments;
    }

    if (!stripPrefix.empty() && namespaceName.empty())
    {
        llvm::WithColor::error(llvm::errs(), "tsbindgen") << "--strip-prefix needs --namespace\n";
        return BadArguments;
    }

    std::vector<std::string> tried;
    auto exePath = llvm::sys::fs::getMainExecutable(argv[0], reinterpret_cast<void *>(&main));
    auto foundResourceDir = findResourceDir(exePath, tried);
    if (!foundResourceDir)
    {
        auto &error = llvm::WithColor::error(llvm::errs(), "tsbindgen");
        error << "clang's resource directory not found; tried:";
        for (auto &path : tried)
        {
            error << "\n  " << path;
        }

        error << "\n(pass --resource-dir <dir>, the directory with include/stddef.h)\n";
        // a wrong --resource-dir is the caller's argument; a failed lookup is a broken install
        return resourceDir.empty() ? ParseOrIOError : BadArguments;
    }

    auto buffer = llvm::MemoryBuffer::getFile(inputFile);
    if (!buffer)
    {
        llvm::WithColor::error(llvm::errs(), "tsbindgen")
            << "cannot read '" << inputFile << "': " << buffer.getError().message() << "\n";
        return ParseOrIOError;
    }

    llvm::SmallString<256> absoluteInput(inputFile);
    llvm::sys::fs::make_absolute(absoluteInput);

    auto triple = target.empty() ? llvm::sys::getDefaultTargetTriple() : target.getValue();

    tsbindgen::ParseInput input;
    input.code = (*buffer)->getBuffer().str();
    input.fileName = std::string(absoluteInput);
    input.args = {"-x", isCxx(inputFile) ? "c++" : "c", "--target=" + triple, "-resource-dir", *foundResourceDir};
    for (auto &dir : includeDirs)
    {
        input.args.push_back("-I" + dir);
    }

    for (auto &define : defines)
    {
        input.args.push_back("-D" + define);
    }

    input.args.insert(input.args.end(), extraArgs.begin(), extraArgs.end());

    auto model = tsbindgen::parseHeader(input);
    if (!model)
    {
        llvm::WithColor::error(llvm::errs(), "tsbindgen") << llvm::toString(model.takeError()) << "\n";
        return ParseOrIOError;
    }

    std::string commandLine = "tsbindgen";
    for (int index = 1; index < argc; ++index)
    {
        commandLine += " ";
        commandLine += argv[index];
    }

    tsbindgen::PrintOptions options;
    options.filters = filters;
    options.namespaceName = namespaceName;
    options.stripPrefix = stripPrefix;
    options.headerLines = {"Generated by tsbindgen " + version() + " (clang " CLANG_VERSION_STRING "); do not edit.",
                           commandLine};
    // `import` would not do: it loads a same-named .dll/.so as a tslang library if there is one,
    // and includes a source file as declarations only, so its constants would have no value
    if (outputFile != "-")
    {
        options.headerLines.push_back("Include with: /// <reference path=\"" +
                                      llvm::sys::path::filename(outputFile).str() + "\" />");
    }

    auto printed = tsbindgen::printBindings(*model, options);
    if (!printed)
    {
        llvm::WithColor::error(llvm::errs(), "tsbindgen") << llvm::toString(printed.takeError()) << "\n";
        return BadArguments;
    }

    for (auto &warning : printed->warnings)
    {
        llvm::WithColor::warning(llvm::errs(), "tsbindgen") << warning << "\n";
    }

    // A .d.ts is included as declarations only: a `const` in it has no value and fails to link.
    if (llvm::StringRef(outputFile.getValue()).ends_with(".d.ts") &&
        llvm::StringRef(printed->text).contains("const "))
    {
        llvm::WithColor::warning(llvm::errs(), "tsbindgen")
            << "'" << outputFile << "' declares constants, which a .d.ts file cannot define; name it .ts\n";
    }

    std::error_code error;
    llvm::raw_fd_ostream out(outputFile, error, llvm::sys::fs::OF_Text);
    if (error)
    {
        llvm::WithColor::error(llvm::errs(), "tsbindgen")
            << "cannot write '" << outputFile << "': " << error.message() << "\n";
        return ParseOrIOError;
    }

    out << printed->text;
    out.close();
    if (out.has_error())
    {
        llvm::WithColor::error(llvm::errs(), "tsbindgen") << "cannot write '" << outputFile << "'\n";
        out.clear_error();
        return ParseOrIOError;
    }

    return Written;
}
```

`tslang/tsbindgen/CMakeLists.txt`:

```cmake
set_Options()

set(LLVM_LINK_COMPONENTS
  Support
  TargetParser
  )

add_llvm_executable(tsbindgen
  tsbindgen.cpp
  )

llvm_update_compile_flags(tsbindgen)
target_link_libraries(tsbindgen PRIVATE TsClangImporter)

# The last place tsbindgen looks for clang's resource directory (stddef.h, stdint.h, ...): the
# LLVM it was built with. A release package puts one next to tsbindgen instead.
target_compile_definitions(tsbindgen PRIVATE
  TSBINDGEN_CONFIGURED_RESOURCE_DIR="${LLVM_LIBRARY_DIR}/clang/${LLVM_VERSION_MAJOR}")

# the same version tslang reports: the release tag, or the commit a dev build was made from
if(TSLANG_PACKAGE_VERSION)
  set(TSBINDGEN_VERSION "${TSLANG_PACKAGE_VERSION}")
else()
  find_package(Git QUIET)
  if(GIT_EXECUTABLE)
    execute_process(
      COMMAND ${GIT_EXECUTABLE} rev-parse --short HEAD
      WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
      OUTPUT_VARIABLE TSBINDGEN_VERSION
      OUTPUT_STRIP_TRAILING_WHITESPACE
      ERROR_QUIET)
  endif()
endif()

if(TSBINDGEN_VERSION)
  target_compile_definitions(tsbindgen PRIVATE TSBINDGEN_VERSION="${TSBINDGEN_VERSION}")
endif()
```

In `tslang/CMakeLists.txt`, after `add_subdirectory(tslang)`:

```cmake
add_subdirectory(tsbindgen)
```

- [ ] **Step 5: Run it to verify it passes**

Run: `cmake . && cmake --build . --config Release -j 16 --target tsbindgen && ctest -C Release -R test-bindgen-cli --output-on-failure`
Expected: `100% tests passed, 0 tests failed out of 1`. With `-V`, 13 `-- <case>: exit <n>` lines, from `a header, written to a file: exit 0` to `a --resource-dir without clang's headers: exit 2`.

- [ ] **Step 6: Try it by hand**

Run: `I:/TypeScriptCompiler/__build/tslang/windows-msbuild-2026-release/bin/tsbindgen.exe I:/TypeScriptCompiler/tslang/test/bindgen/fixture.h`
Expected on stdout (header lines, then):

```ts
// skipped: FIXTURE_API — macro is not a literal
const FIXTURE_ANSWER = 42;
const FIXTURE_RATIO = 1.5;
const FIXTURE_NAME = "fixture";
type Counter = Opaque;
type Point = [x: s32, y: s32];
type Box = [tag: s8, corner: Point, scale: f64, visible: boolean];
type Node = [value: s32, next: Opaque /* Reference<Node>: a type cannot refer to itself */];
enum Color { RED = 0, GREEN = 5, BLUE = 6 }
// skipped: Value — union
type Value = Opaque;
declare function fx_add(a: s32, b: s32): s32;
```

…through `declare function fx_value_clear(v: Value): void;` and `// skipped: fx_twice_inline — static inline function: no symbol to link against`; on stderr, three `tsbindgen: warning: skipped ...` lines.

- [ ] **Step 7: Commit**

```bash
git add tslang/tsbindgen tslang/CMakeLists.txt tslang/test/bindgen/fixture.h tslang/test/bindgen/run-cli-test.cmake tslang/test/tester/CMakeLists.txt
git commit -m "tsbindgen: command line, with exit codes pinned by test-bindgen-cli

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 5: End to end — call a real C library through generated bindings

**Files:**
- Create: `tslang/test/bindgen/fixture.c`, `bindgen_test.ts`, `bindgen_ns_test.ts`, `expected.txt`, `run-bindgen-test.cmake`
- Modify: `tslang/test/tester/CMakeLists.txt` (next to `test-bindgen-cli`)

**Interfaces:**
- Consumes: `tsbindgen` (Task 4), `tslang`, `fixture.h` (Task 4); `TEST_GC_LIBDIR`, `defaultlib_collector_llvm_lib`, `defaultlib_collector_opt` (already defined above the insertion point in `test/tester/CMakeLists.txt`); `LLVM_TOOLS_BINARY_DIR` (from `LLVMConfig`, the directory with `clang`).

- [ ] **Step 1: Write the programs and the expected output**

`tslang/test/bindgen/bindgen_test.ts` — note `/// <reference path>`, not `import`, and `// @strict-null false` for the `null` list terminator:

```ts
// @strict-null false
/// <reference path="fixture.ts" />

// Calls everything in fixture.h through the bindings tsbindgen generated from it (fixture.ts).
// run-bindgen-test.cmake compares what this prints with expected.txt.

function twice(x: s32): s32 {
    return x * 2;
}

function main() {
    print("macros", FIXTURE_ANSWER, FIXTURE_RATIO, FIXTURE_NAME);
    print("add", fx_add(2, 3));
    print("scale", fx_scale(2.0, 1.5));
    print("narrow", fx_widen_s8(-5), fx_widen_u16(65535), fx_negate_s8(5), fx_not(true), fx_not(false));
    print("len", fx_len("hello"));
    print("greet", fx_greet());

    let quotient: s32 = 0;
    let remainder: s32 = 0;
    fx_divmod(17, 5, ReferenceOf(quotient), ReferenceOf(remainder));
    print("divmod", quotient, remainder);

    const counter = fx_counter_new();
    fx_counter_inc(counter);
    fx_counter_inc(counter);
    print("counter", fx_counter_get(counter));
    fx_counter_free(counter);

    let point: Point = [3, 4];
    print("point", fx_point_sum(ReferenceOf(point)));

    let box: Box = [0, [0, 0], 0.0, false];
    fx_box_fill(ReferenceOf(box));
    print("box", box[0], box[1][0], box[1][1], box[2], box[3]);

    let second: Node = [5, null];
    let first: Node = [3, ReferenceOf(second) as Opaque];
    print("list", fx_list_sum(ReferenceOf(first)));

    print("color", fx_color_value(Color.BLUE));
    print("apply", fx_apply(twice as Opaque, 21));
    print("sum", fx_sum(3, 1, 2, 3));
}
```

`tslang/test/bindgen/bindgen_ns_test.ts` — the same calls through `--namespace Fx --strip-prefix fx_` (needs PR 1):

```ts
// @strict-null false
/// <reference path="fixture_ns.ts" />

// bindgen_test.ts again, against `tsbindgen fixture.h --namespace Fx --strip-prefix fx_`: every
// function is reached through a TS name that is not its C symbol, which @linkname binds.

function twice(x: s32): s32 {
    return x * 2;
}

function main() {
    print("macros", Fx.FIXTURE_ANSWER, Fx.FIXTURE_RATIO, Fx.FIXTURE_NAME);
    print("add", Fx.add(2, 3));
    print("scale", Fx.scale(2.0, 1.5));
    print("narrow", Fx.widen_s8(-5), Fx.widen_u16(65535), Fx.negate_s8(5), Fx.not(true), Fx.not(false));
    print("len", Fx.len("hello"));
    print("greet", Fx.greet());

    let quotient: s32 = 0;
    let remainder: s32 = 0;
    Fx.divmod(17, 5, ReferenceOf(quotient), ReferenceOf(remainder));
    print("divmod", quotient, remainder);

    const counter = Fx.counter_new();
    Fx.counter_inc(counter);
    Fx.counter_inc(counter);
    print("counter", Fx.counter_get(counter));
    Fx.counter_free(counter);

    let point: Fx.Point = [3, 4];
    print("point", Fx.point_sum(ReferenceOf(point)));

    let box: Fx.Box = [0, [0, 0], 0.0, false];
    Fx.box_fill(ReferenceOf(box));
    print("box", box[0], box[1][0], box[1][1], box[2], box[3]);

    let second: Fx.Node = [5, null];
    let first: Fx.Node = [3, ReferenceOf(second) as Opaque];
    print("list", Fx.list_sum(ReferenceOf(first)));

    print("color", Fx.color_value(Fx.Color.BLUE));
    print("apply", Fx.apply(twice as Opaque, 21));
    print("sum", Fx.sum(3, 1, 2, 3));
}
```

`tslang/test/bindgen/expected.txt` — what both programs print, in both modes (the `narrow` line is Review Focus 1):

```text
macros 42 1.5 fixture
add 5
scale 3
narrow -5 65535 -5 false true
len 5
greet hello from C
divmod 3 2
counter 2
point 7
box -7 30 -40 2.5 true
list 8
color 6
apply 42
sum 6
```

- [ ] **Step 2: Write the driver and register it**

`tslang/test/bindgen/run-bindgen-test.cmake`:

```cmake
# tsbindgen end to end: generate bindings for a real C library and call it through them.
#
# fixture.c is compiled with the in-tree clang, to an object and to a shared library; tsbindgen
# turns fixture.h into fixture.ts; bindgen_test.ts is compiled against it and linked with the
# object (--emit=exe), then run under the JIT against the shared library (--emit=jit). Both must
# print expected.txt. MODE=namespace does the same through `--namespace Fx --strip-prefix fx_`,
# which only links because @linkname binds each renamed function to its C symbol.

cmake_minimum_required(VERSION 3.17.3)

foreach(var MODE TSLANG TSBINDGEN CLANG SOURCE_DIR WORK_DIR GC_LIB LLVM_LIB TSLANG_LIB TSLANG_BIN OPT)
    if(NOT DEFINED ${var})
        message(FATAL_ERROR "${var} is required")
    endif()
endforeach()

if(WIN32)
    set(exe_suffix ".exe")
    set(obj_suffix ".obj")
    set(fixture_library "${WORK_DIR}/fixture_c.dll")
    set(runtime "${TSLANG_BIN}/TypeScriptRuntime.dll")
    set(pic "")
    set(clang_pic "")
    set(run_env "PATH=${WORK_DIR};$ENV{PATH}")
else()
    set(exe_suffix "")
    set(obj_suffix ".o")
    set(fixture_library "${WORK_DIR}/libfixture_c.so")
    set(runtime "${TSLANG_LIB}/libTypeScriptRuntime.so")
    set(pic "-relocation-model=pic")
    set(clang_pic "-fPIC")
    set(run_env "LD_LIBRARY_PATH=${WORK_DIR}:$ENV{LD_LIBRARY_PATH}")
endif()

# the C library's stem differs from the bindings' on purpose: `import "./fixture"` next to a
# fixture.dll would load that as a tslang library
if(MODE STREQUAL "namespace")
    set(bindings "fixture_ns.ts")
    set(program "bindgen_ns_test.ts")
    set(bindgen_args --namespace Fx --strip-prefix fx_)
else()
    set(bindings "fixture.ts")
    set(program "bindgen_test.ts")
    set(bindgen_args "")
endif()

file(REMOVE_RECURSE "${WORK_DIR}")
file(MAKE_DIRECTORY "${WORK_DIR}")
file(COPY "${SOURCE_DIR}/fixture.h" "${SOURCE_DIR}/fixture.c" "${SOURCE_DIR}/${program}" DESTINATION "${WORK_DIR}")
file(READ "${SOURCE_DIR}/expected.txt" expected)
string(REPLACE "\r\n" "\n" expected "${expected}")

set(common --no-default-lib ${OPT} ${pic} "--gc-lib-path=${GC_LIB}" "--llvm-lib-path=${LLVM_LIB}"
    "--tslang-lib-path=${TSLANG_LIB}")

# run(<what> <command...>) - fails unless it exits 0; leaves stdout in run_output
function(run what)
    execute_process(COMMAND ${CMAKE_COMMAND} -E env "${run_env}" ${ARGN}
        WORKING_DIRECTORY "${WORK_DIR}"
        OUTPUT_VARIABLE out
        ERROR_VARIABLE err
        RESULT_VARIABLE status)
    if(NOT status EQUAL 0)
        message(FATAL_ERROR "${what}: exit ${status}\n${out}\n${err}")
    endif()
    string(REPLACE "\r\n" "\n" out "${out}")
    set(run_output "${out}" PARENT_SCOPE)
endfunction()

function(expect_output what)
    if(NOT run_output STREQUAL expected)
        message(FATAL_ERROR "${what} printed:\n${run_output}\nexpected:\n${expected}")
    endif()
    message(STATUS "${what}: as expected")
endfunction()

run("clang -c fixture.c" "${CLANG}" -c ${clang_pic} fixture.c -o "fixture${obj_suffix}")
run("clang -shared fixture.c" "${CLANG}" -shared ${clang_pic} fixture.c -o "${fixture_library}")
run("tsbindgen" "${TSBINDGEN}" fixture.h ${bindgen_args} -o "${bindings}")

run("--emit=exe" "${TSLANG}" --emit=exe ${common} "${program}" "--obj=fixture${obj_suffix}" -o "bindgen_test${exe_suffix}")
run("the program" "${WORK_DIR}/bindgen_test${exe_suffix}")
expect_output("--emit=exe")

run("--emit=jit" "${TSLANG}" --emit=jit ${common} "--shared-libs=${runtime}" "--shared-libs=${fixture_library}" "${program}")
expect_output("--emit=jit")
```

In `tslang/test/tester/CMakeLists.txt`, immediately before the `add_test(NAME test-bindgen-cli ...)` from Task 4, with this comment heading both:

```cmake
# tsbindgen: bindings generated from test/bindgen/fixture.h call the C library compiled from
# fixture.c, linked (--emit=exe) and under the JIT; `namespace` does it again through
# --namespace/--strip-prefix, which @linkname binds. And the command line's exit codes.
foreach(bindgen_mode plain namespace)
    add_test(NAME test-bindgen-${bindgen_mode}
             COMMAND ${CMAKE_COMMAND}
                     "-DMODE=${bindgen_mode}"
                     "-DTSLANG=$<TARGET_FILE:tslang>"
                     "-DTSBINDGEN=$<TARGET_FILE:tsbindgen>"
                     "-DCLANG=${LLVM_TOOLS_BINARY_DIR}/clang${CMAKE_EXECUTABLE_SUFFIX}"
                     "-DSOURCE_DIR=${PROJECT_SOURCE_DIR}/test/bindgen"
                     "-DWORK_DIR=${CMAKE_CURRENT_BINARY_DIR}/bindgen-${bindgen_mode}"
                     "-DGC_LIB=${TEST_GC_LIBDIR}"
                     "-DLLVM_LIB=${defaultlib_collector_llvm_lib}"
                     "-DTSLANG_LIB=${CMAKE_BINARY_DIR}/lib"
                     "-DTSLANG_BIN=${CMAKE_BINARY_DIR}/bin"
                     "-DOPT=${defaultlib_collector_opt}"
                     -P "${PROJECT_SOURCE_DIR}/test/bindgen/run-bindgen-test.cmake")
endforeach()
```

- [ ] **Step 3: Run it to verify it fails**

Run: `cmake . && ctest -C Release -R test-bindgen-plain --output-on-failure`
Expected: FAIL — `file COPY cannot find ".../tslang/test/bindgen/fixture.c"` (the C file is written in the next step).

- [ ] **Step 4: Write the C library**

`tslang/test/bindgen/fixture.c`:

```c
#include "fixture.h"

#include <stdarg.h>
#include <stdlib.h>

struct Counter
{
    int32_t n;
};

int32_t fx_add(int32_t a, int32_t b)
{
    return a + b;
}

double fx_scale(double v, float f)
{
    return v * f;
}

int32_t fx_widen_s8(int8_t v)
{
    return v;
}

int32_t fx_widen_u16(uint16_t v)
{
    return v;
}

int8_t fx_negate_s8(int8_t v)
{
    return (int8_t)-v;
}

bool fx_not(bool b)
{
    return !b;
}

size_t fx_len(const char *s)
{
    size_t n = 0;
    while (s[n])
    {
        n++;
    }

    return n;
}

const char *fx_greet(void)
{
    return "hello from C";
}

void fx_divmod(int32_t a, int32_t b, int32_t *quotient, int32_t *remainder)
{
    *quotient = a / b;
    *remainder = a % b;
}

Counter *fx_counter_new(void)
{
    Counter *c = malloc(sizeof *c);
    c->n = 0;
    return c;
}

void fx_counter_inc(Counter *c)
{
    c->n++;
}

int32_t fx_counter_get(Counter *c)
{
    return c->n;
}

void fx_counter_free(Counter *c)
{
    free(c);
}

int32_t fx_point_sum(const Point *p)
{
    return p->x + p->y;
}

void fx_box_fill(Box *b)
{
    b->tag = -7;
    b->corner.x = 30;
    b->corner.y = -40;
    b->scale = 2.5;
    b->visible = true;
}

int32_t fx_list_sum(Node *head)
{
    int32_t sum = 0;
    for (; head; head = head->next)
    {
        sum += head->value;
    }

    return sum;
}

int32_t fx_color_value(enum Color c)
{
    return (int32_t)c;
}

int32_t fx_apply(int32_t (*fn)(int32_t), int32_t v)
{
    return fn(v);
}

int32_t fx_sum(int32_t count, ...)
{
    va_list args;
    va_start(args, count);
    int32_t sum = 0;
    for (int32_t i = 0; i < count; i++)
    {
        sum += va_arg(args, int32_t);
    }

    va_end(args);
    return sum;
}

void fx_value_clear(union Value *v)
{
    v->i = 0;
}
```

- [ ] **Step 5: Run both modes**

Run: `cmake --build . --config Release -j 16 --target tslang tsbindgen && ctest -C Release -R "test-bindgen-(plain|namespace)" -V`
Expected: both pass, each printing `-- --emit=exe: as expected` and `-- --emit=jit: as expected`. `test-bindgen-namespace` failing with `unresolved external symbol Fx.add` means PR 1 is not in this tree — see the Prerequisite. Failing with **only** `Fx.sum` unresolved means PR 1 mishandles `@varargs` and `@linkname` on the same declaration: a PR 1 bug to fix there, not a tsbindgen one.

- [ ] **Step 6: Run the whole suite**

Run: `ctest -j 16 -C Release`
Expected: `100% tests passed` — the previous total (2887 on 2026-09-26) plus 4 (`test-bindgen-plain`, `test-bindgen-namespace`, `test-bindgen-cli`, `unittest-TsClangImporterTests`).

- [ ] **Step 7: Commit**

```bash
git add tslang/test/bindgen tslang/test/tester/CMakeLists.txt
git commit -m "tsbindgen: end-to-end test against a C library, linked and under the JIT

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 6: Linux

Nothing in this plan was run on Linux. The code has no Windows-only path except `FIXTURE_API` and the `WIN32` branches of the two scripts, but Review Focus 1 (narrow integers on SysV) can only be checked there.

**Files:** none, unless CI fails.

- [ ] **Step 1: Push and open the PR as a draft; watch both CI workflows**

```bash
gh auth switch --user ASDAlexander77
git push
gh pr create --draft --title "tsbindgen v1: C headers to tslang bindings" --body-file <(printf 'Draft: CI check before review.\n\n🤖 Generated with [Claude Code](https://claude.com/claude-code)\n')
gh pr checks --watch
```

Expected: both `cmake-test-release-win` and `cmake-test-release-linux` green, with `test-bindgen-plain`, `test-bindgen-namespace`, `test-bindgen-cli` in the ctest log (`unittest-TsClangImporterTests` only if the runner has gtest's sources).

- [ ] **Step 2: If a Linux bindgen test fails, classify before touching code**

- `narrow` line differs (e.g. `narrow 251 65535 251 ...`): Review Focus 1 — tslang's call lowering. File a compiler issue with the failing line and the generated `fixture.ts`; mark the PR description with it; do not change the mapping.
- `clang: error: unknown ...` / `cannot find -l...`: the fixture build on Linux — fix `run-bindgen-test.cmake`'s non-`WIN32` branch.
- `tsbindgen` exit 1 on `fixture.h`: system headers not found — check the `LLVM_TARGZFILE` package's `lib/clang/<ver>/include` (Task 1 Step 1).

Rerun a flaky async test once before investigating it (see the CI flaky-test notes); these bindgen tests have no async code.

---

### Task 7: Default-lib check (manual, reported in the PR)

Spec "PR 2 tests" item 3, and Review Focus 3.

**Files:** none.

- [ ] **Step 1: Generate bindings for the two wrappers**

```bash
B=I:/TypeScriptCompiler/__build/tslang/windows-msbuild-2026-release/bin
L=I:/TypeScriptCompilerDefaultLib/src
"$B/tsbindgen.exe" "$L/wrappers/regex.cpp" --filter "regexp_*" -o regex.ts
echo "regex exit=$?"
"$B/tsbindgen.exe" "$L/wrappers/http.cpp" --filter "http_*" -o http.ts
echo "http exit=$?"
```

Expected: both `exit=0`. If one fails on an include, pass the directories the default lib's own build uses with `-I` (see `TypeScriptCompilerDefaultLib`'s build scripts) and note them in the PR.

- [ ] **Step 2: Compare with the hand-written declarations**

```bash
grep -E "declare function (regexp|http)_" "$L/native/lib.native.d.ts" | sed 's/  */ /g' | sort > hand.txt
grep -hE "declare function (regexp|http)_" regex.ts http.ts | sed 's/  */ /g' | sort > generated.txt
diff hand.txt generated.txt
```

Expected differences, which are not bugs: `int`/`s32`, `double`/`f64`, `long`/`s64` or `s32`, `index` vs `u64`/`index`, and parameter names where the header has none. Any other difference is either a generator bug (fix it, with a unit test) or a mistake in `lib.native.d.ts` — write each one into the PR description saying which.

---

### Task 8: Amend the spec; finish the PR

**Files:**
- Modify: `docs/superpowers/specs/2026-09-25-tsbindgen-linkname-design.md`
- Modify: `tslang/docs/reference-counting-evaluation.md` (append a section)

- [ ] **Step 1: Record the C-string open issue with the rc model's**

The spec's open issue "C-owned `char*` returned as `string`" says to record it there. Append after the last `### 9.<n>` section (9.77 on 2026-09-26) a section numbered one higher:

```markdown
### 9.78 A C string returned through generated bindings (open)

tsbindgen maps a C function's `char *` / `const char *` result to `string`, as the default library
already does by hand (`regexp_match_results_format`). A tslang string carries an 8-byte header before
its characters; a C-owned `char *` has none. Under `-mm=gc` that is harmless - nothing reads the
header. Under `-mm=rc` a release of such a value reads a count that is not there, which is undefined.
Not fixed: tsbindgen v1 targets `gc`. Revisit if rc becomes a supported model for bindings - the
likely shape is a distinct C-string type in the compiler that is never released.
```

- [ ] **Step 2: Amend the spec**

In the spec: change the mapping table's integer row to `s8`…`s64` / `u8`…`u64`; change `-o <out.d.ts>` to `-o <out.ts>` (default stdout) and every "`.d.ts`" that names tsbindgen's output to "`.ts`"; replace the function-pointer row's comment form with the inline `/* ... */` one; and add, after "Skipped declarations", a subsection **"As built (2026-09-26)"** containing this plan's "Deviations from the spec" table verbatim. Change the Status line to "PR 2 implemented; PR 3 (packaging) not started."

- [ ] **Step 3: Commit, then mark the PR ready**

```bash
git add docs/superpowers/specs/2026-09-25-tsbindgen-linkname-design.md tslang/docs/reference-counting-evaluation.md
git commit -m "Spec: tsbindgen as built - .ts output, signed sN, type cycles

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
git push
```

Write the PR description (via `gh api` — `gh pr edit` is broken on this repo): what tsbindgen does, the deviations table, Task 7's comparison, and any Linux finding from Task 6. End it with `🤖 Generated with [Claude Code](https://claude.com/claude-code)`. Then `gh pr ready`.
