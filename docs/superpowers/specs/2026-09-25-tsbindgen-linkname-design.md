# `@linkname` and `tsbindgen` v1

Design for binding C (and `extern "C"` C++) code from tslang without writing
`.d.ts` declarations by hand. It covers two items: a `@linkname` decorator for
declaring a C function under a different TS name, and `tsbindgen`, a generator
that turns C headers into tslang binding files (`.ts`).

Status: PR 1 and PR 2 implemented; PR 3 (packaging) not started. Refines Option A of
`docs/c-cpp-header-import.md`; C++ wrapper generation (that doc's "C++ 2"
route) is v2 and gets its own spec once v1 has merged.

## Why

The default lib already has a hand-written C layer:
`TypeScriptCompilerDefaultLib/src/wrappers/*.cpp` wrap C++ (`std::regex`, HTTP,
threads) behind `extern "C" TSLANG_EXPORT` functions, and
`src/native/lib.native.d.ts` repeats each signature by hand
(`regexp_match_results_size(cm: Opaque): index`). Every new binding costs two
edits that must agree, and nothing checks that they do. A generator removes
the second edit; `@linkname` lets a binding use a TS name other than the C
symbol.

## Measured baseline

Probed with `__build/tslang/windows-msbuild-2026-release` (built 2026-09-25),
`--no-default-lib`, target `x86_64-pc-windows-msvc`.

| Case | Result |
| --- | --- |
| `@dllname("strlen") declare function cStrLen(s: string): index;` | Works. IR declares and calls `@strlen`; `--emit=jit` and `--emit=exe` both print `5`. |
| `@dllname("?foo@@YAHH@Z") declare function foo(x: i32): i32;` | Works. IR declares `@"?foo@@YAHH@Z"(i32)`. |
| Two declarations with `@dllname("strlen")`, plus a plain `declare function strlen` | **Broken.** IR has `@strlen`, `@strlen.1`, `@strlen.2`. JIT: `Symbols not found: [ strlen.1, strlen.2 ]`. Exe: `LNK2019: unresolved external symbol strlen.1`. |
| `namespace C { @dllname("strlen") export declare function len(...) }` | Links and runs, but IR is `declare dllexport i64 @strlen(ptr)`: the exe re-exports the CRT's `strlen` and the linker writes `p4.lib`/`p4.exp`. |
| `namespace C { export declare function strlen(...) }` | IR is `declare dllexport i64 @C.strlen(ptr)`: the symbol is the namespaced name, and it is marked `dllexport` too. |
| `enum Color {...}` parameter of a `declare function` | Lowers to `i32`. |
| `(a: Opaque, b: Opaque) => i32` parameter | Lowers to `{ ptr, ptr, ptr }` (hybrid function), not a C function pointer. |
| `cmp as Opaque` passed to an `Opaque` parameter | Lowers to bare `ptr @cmp`, a valid C function pointer. |
| `@varargs declare function printf(format: string);` | C varargs (existing test `02funcs_vararg.ts`). `...args: any[]` instead passes an array struct. |
| `type S = [a: i8, b: i32, c: i64]`, `f(ReferenceOf(s))` to `declare function f(s: Reference<S>)` | Storage is `alloca { i8, i32, i64 }`, fields in declared order (C's layout). `f` gets the variable's own storage, and a later `s[1]` reads from it, so C writes are visible. |

### Two findings that reduced scope

**`@linkname` already exists as `@dllname`.** `docs/c-cpp-header-import.md`
(#197) lists a `@linkname("?foo@@YAHH@Z")` attribute as a needed compiler
addition. `@dllname` has since gained exactly that behaviour, for
definitions, declarations, globals, methods and accessors
(`test/tester/tests/export_dllname.ts`). What is missing is a name that fits
static linking, and two defects in the declaration case (below).

**C callbacks need no compiler work for v1.** A tslang function type is
always a `HybridFunctionType` (`MLIRGenTypes.cpp`, `getFunctionType`), which
is ABI-incompatible with a C function pointer. But casting a plain function to
`Opaque` produces the bare symbol address, so generated bindings type callback
parameters as `Opaque` and callers write `fn as Opaque`.

## PR 1 — `@linkname`

### Decorator

`@linkname("sym")` is a second spelling of `@dllname("sym")`, with identical
meaning everywhere `@dllname` is read today: `MLIRGenFunctions.cpp`
(`mlirGenFunctionLikeDeclaration`), `MLIRGenImpl.h`
(`processFunctionAttributes`), `MLIRGenVariables.cpp`, `MLIRGenClasses.cpp`.
Add `#define LINK_NAME "linkname"` to `Defines.h` and one predicate
`isLinkNameDecorator(name)` (true for `DLL_NAME` or `LINK_NAME`) used at each
of those sites. The attribute stored on the op stays `DLL_NAME`, so lowering,
`ExportFixPass` and `DeclarationPrinter` need no change; `__decls` keeps
printing `@dllname(...)`, which the importer reads identically.

Both decorators on one declaration with different values is an error:
`conflicting @dllname/@linkname: 'a' vs 'b'`. Same value is accepted.

`@dllname` is not deprecated.

### Fix: bindings that share a symbol

Today the function rename happens at LLVM level, in `ExportFixPass`
(`F.setName(dllName)`), and LLVM silently makes a taken name unique. Move the
function rename to MLIR, into `LowerToLLVM.cpp` next to the existing global
rename (the `renamedGlobals` loop, which already uses
`SymbolTable::replaceAllSymbolUses` + `setSymbolName`). For each
`mlir_ts::FuncOp` with a `DLL_NAME` attribute, remove the attribute, then:

| Symbol `sym` already in the module? | Action |
| --- | --- |
| No | Rename the op to `sym` and update all uses. |
| Yes, same function type, at least one of the two is a declaration | Point all uses of the renamed op at the existing symbol, keep whichever is a definition, erase the other. |
| Yes, same function type, both are definitions | Error: `'a' and 'b' both define symbol 'sym'`. |
| Yes, different function type | Error: `'a' binds symbol 'sym' with type T1, but 'b' declares it with type T2`. |

"Same function type" compares the `mlir_ts::FunctionType` of the two ops,
including the varargs flag. It is deliberately stricter than comparing the
LLVM types (`string` and `Opaque` both lower to `ptr`): at the point of the
rename the call ops are still TS-typed, so redirecting a call to a callee with
a different TS signature would fail the verifier unless casts were inserted.

`ExportFixPass`'s `DLL_NAME` branch stays as a fallback for any function
attribute still set after MLIR lowering; once the MLIR rename runs, it never
fires.

### Fix: no `dllexport` on a declaration

In `processFunctionAttributes` (`MLIRGenImpl.h`), `dllExport` is currently
`getExportModifier(...) || InternalFlags::DllExport`. Require a body as well:
a declaration without a body never gets the `export` attribute. `isPublic`
(which keeps the symbol alive through SymbolDCE) is computed separately and
is unchanged.

The `-shared` cross-module tests import re-parsed declarations that carry
`@dllimport`; after this change those get `import` only instead of both
`export` and `import` (where `import` won anyway, being applied last in
`ExportFixPass`). The full ctest suite verifies this.

### PR 1 tests

- `test/tester/tests/declare_linkname.ts`, registered with
  `tslang_add_test` as compile and JIT tests:
  - `@linkname("strlen") declare function a(...)` and
    `@dllname("strlen") declare function b(...)`, plus
    `declare function strlen(...)` in the same file, all called and all
    returning the right length;
  - the same binding inside a namespace with `export declare`;
  - `@linkname` on a TS-defined function called through a `declare` of the
    target name.
- A negative test for each new error (type mismatch, two definitions,
  conflicting decorators) in `test/tester/linkname/`, registered like the
  `call-arity` cases in `test/tester/CMakeLists.txt`: `--emit=obj`,
  `PASS_REGULAR_EXPRESSION` on the message, `FAIL_REGULAR_EXPRESSION` on
  `Stack dump|Assertion failed`.
- A `DeclarationPrinter` unit test: a `@linkname` function prints as
  `@dllname("...")`.
- Manual check, recorded in the PR: `--emit=llvm` of the namespace case shows
  no `dllexport` on `@strlen`, and `--emit=exe` writes no `.lib`/`.exp`.

## PR 2 — `tsbindgen` v1

### Structure

- `tslang/lib/TsClangImporter/` — static library, no dependency on MLIRGen:
  - `HeaderParser`: runs clang (`clang::tooling`, a `RecursiveASTVisitor`
    plus `PPCallbacks` for macros) and returns plain records: functions,
    records, enums, typedefs, macros, each with its source location.
  - `TypeMapper`: C type → tslang type text, or a skip reason.
  - `BindingPrinter`: records → `.ts` text (see "As built" below: not `.d.ts`).
- `tslang/tsbindgen/` — CLI executable over the library.
- Links against clang libraries from the in-tree LLVM install
  (`clangTooling`, `clangFrontend`, `clangAST`, `clangSema`, `clangLex`,
  `clangBasic`, …). `find_package(Clang)` is already in
  `tslang/CMakeLists.txt`; `tslang.exe` itself gains no new dependency.

### Command line

```text
tsbindgen <input.h|.c|.cpp> [-I<dir>]... [-D<name>[=v]]... [--target <triple>]
          [--filter <glob>]... [--namespace <N> [--strip-prefix <P>]]
          [--resource-dir <dir>] [-o <out.ts>] [-- <extra clang args>...]
```

- `--target` defaults to the host triple. Type widths come from clang's
  `ASTContext` for that target.
- A `.cpp` input is parsed as C++; only functions with C language linkage are
  taken.
- clang's resource directory (its `stddef.h`/`stdint.h`; without it they fail
  to resolve) is found in this order: `--resource-dir <dir>` if given;
  `<exe dir>/lib/clang/<ver>` (the release package layout, where `tsbindgen`
  sits at the package root next to `tslang`); `<exe dir>/../lib/clang/<ver>`
  (an LLVM-style install); the directory the build was configured with. None
  found is an error naming the paths tried.

### What gets emitted

A declaration is **selected** if its name matches any `--filter` glob, or,
with no `--filter`, if it is declared in the input file itself (not in an
included header). Types used by a selected declaration are emitted too,
wherever they are declared. Real headers pull in thousands of system
declarations, so this rule is essential.

### Type mapping

| C | tslang |
| --- | --- |
| integer types | exact width, signed `s8`…`s64` / unsigned `u8`…`u64`, from the target (tslang's `i8`…`i64` are signless). Never `int`/`long`: tslang `long` is 64-bit, but C `long` is 32-bit on Windows. |
| `size_t`, `ssize_t`, `ptrdiff_t`, `intptr_t`, `uintptr_t` | `index` |
| `float` / `double` / `_Bool`, `bool` | `f32` / `f64` / `boolean` |
| `char*`, `const char*` | `string` |
| `char**` | `Reference<string>` |
| `T*`, T a scalar or complete struct | `Reference<T>` |
| `void*`; pointer to an incomplete struct or a C++ class | `Opaque`. A named incomplete struct `S` also gets `type S = Opaque;` and parameters use `S`. |
| function pointer | `Opaque`, with the C signature in a comment: `fn: Opaque /* (p0: s32) => s32 */` for a parameter or field, a trailing `// ...` for a `typedef`. |
| complete struct of mappable fields | named tuple `type S = [a: T1, b: T2];`, as `tm` in `core.os.d.ts` |
| C `enum` | TS `enum` with the same names and values |
| C++ `enum : u8` (fixed non-`int` underlying type) | `type E = u8;` plus `const A = 1;` per enumerator, using the enumerator's own name (`A`, as C code spells it for an unscoped enum; `E_A` for an `enum class`, which has no unqualified name) |
| `...` | fixed parameters only, plus `@varargs` |
| `#define N 42` / `1.5` / `"s"` | `const N = 42;` (integer, float and string literal object-like macros only) |
| `typedef` | `type` alias, unless it names a struct already emitted under the same name |

Parameter names are taken from the header; an unnamed parameter becomes
`p<index>`. A name that is a TS reserved word gets a `_` suffix.

### Skipped declarations

Each of these is left out of the output, with
`// skipped: <name> — <reason>` in its place and a warning on stderr:

- a union, or a struct with a bitfield or an unmappable field;
- `long double`;
- a function that takes or returns a struct by value (its ABI is unverified;
  v2's generated wrappers fix this);
- a function-like macro;
- a `static inline` function (no symbol to link against);
- any other type the mapper does not handle.

A skipped struct does not take its users with it. It is still emitted as
`type S = Opaque;`, so a function that takes or returns `S*` is emitted with
an `S` (that is, `Opaque`) parameter, exactly like an incomplete struct. Only
a function that uses a skipped type **by value** is skipped itself. Without
this rule a single union in a header would remove most of its API.

A skip never stops generation. The exit code is `0` if the file was written,
`1` for a clang parse error or I/O failure (clang's diagnostics are printed
as-is), `2` for bad arguments.

### As built (2026-09-26)

Measured while implementing PR 2 (see `docs/superpowers/plans/2026-09-26-tsbindgen.md`); where this
section and the text above disagree, this section is what was built.

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

### Naming

C names are kept. `--namespace N` wraps the output in
`namespace N { export declare function ... }` and adds `@linkname("orig")` to
each function (without it, `namespace C { export declare function strlen }`
binds the symbol `C.strlen`, measured). `--strip-prefix P` removes `P` from the TS name of every
declaration that starts with it; the `@linkname` keeps the C name.
`--strip-prefix` without `--namespace` is an argument error.

### Output header

```ts
// Generated by tsbindgen <version> (clang <version>); do not edit.
// tsbindgen sqlite3.h --filter sqlite3_* --target x86_64-pc-windows-msvc
```

### PR 2 tests

1. **Unit tests** in `tslang/unittests/TsClangImporter/` (gtest, like
   `unittests/MLIRGen/DeclarationPrinter.cpp`). Each test passes a short
   header string to the importer and checks the printed text. Coverage: every
   row of the mapping table, every skip reason, filtering (selected vs
   pulled-in types, system declarations excluded), `--namespace`,
   `--strip-prefix`, reserved-word parameters. Tests parse with
   `-ffreestanding` and clang's own `stdint.h`/`stddef.h`, so they need no
   system SDK, and one host checks both targets: `long` maps to `i32` for
   `x86_64-pc-windows-msvc` and to `i64` for `x86_64-linux-gnu`.
2. **End-to-end test** in `tslang/test/bindgen/`, run by ctest on the Windows
   and Linux CI:
   - `fixture.h` + `fixture.c`: one of each feature — scalars, a string in and
     out, an out-parameter, an opaque handle, a struct passed by pointer, an
     enum, a literal macro, a callback, a varargs function.
   - At test time: compile `fixture.c` with the in-tree clang to an object and
     a shared library, run `tsbindgen fixture.h -o fixture.ts`, compile
     `bindgen_test.ts` against it, link with `-obj=` and run it; then run the
     same program under `--emit=jit` with `-shared-libs=` pointing at the
     shared library. Both runs must print the expected output.
3. **Default-lib check** (manual, reported in the PR description):
   `tsbindgen src/wrappers/regex.cpp --filter "regexp_*"` and the same for
   `http.cpp`, compared with the matching lines of `lib.native.d.ts`. Known,
   expected differences: `int` vs `i32` and `double` vs `f64` spelling. Any
   other difference is either a generator bug or a hand-written mistake, and
   the PR says which.

## PR 3 — Packaging

- `create-release.yml`: add `tsbindgen.exe` and
  `3rdParty/llvm/x64/release/lib/clang/<ver>/include` (placed at
  `lib/clang/<ver>/include` in the package, next to `tsbindgen` at the
  package root — the second lookup location above) to the Windows
  `Compress-Archive` list; the equivalent `cp` lines for each Linux package.
- A smoke step in each packaging job: from the unpacked package, run
  `tsbindgen --version` and generate the fixture header from PR 2's test.

## Open issues

- **CI LLVM package contents.** The LLVM packages used by both the Windows
  and Linux CI must contain the clang Frontend/Sema/Tooling libraries and
  `lib/clang/<ver>/include`. The local Windows install does; confirm both CI
  packages do at the start of PR 2. If one does not, re-upload it and bump
  `CACHE_VERSION` (see the earlier `/MT` re-upload).
- **C-owned `char*` returned as `string`.** A tslang string carries an 8-byte
  header before its characters (string constants in the IR start with
  `\FF` × 8, and call sites pass `ptr + 8`); a `char*` returned by C has
  none. The default lib already returns C strings as `string`
  (`regexp_match_results_format`) and this works under `-mm=gc`, so v1 maps
  them the same way. Under `-mm=rc` a release of such a pointer is undefined;
  record it with the rc model's open items, and revisit if rc becomes a
  supported target for bindings.
- **Struct by value.** Skipped in v1. Before v2 generates wrappers for them,
  measure whether tslang's lowering of a named tuple parameter already
  matches clang's ABI for struct sizes 1/2/4/8/9/16/17 on both targets
  (section 5 of `docs/c-cpp-header-import.md`); if it does, v1's skip can be
  lifted without wrappers.
- **Callback lifetime.** `fn as Opaque` is only safe for a function with no
  captures. Nothing in the type system enforces this; the generated comment
  says so. A thin function-pointer type in the compiler would, and is a
  separate proposal.
- **Enum width.** The probe used `int`-sized enums only. The unit tests
  cover the C++ fixed-type fallback; an `enum` under `-fshort-enums` is not
  handled and is out of scope.

## Out of scope

- C++ classes, methods, overloads, templates, and wrapper `.cpp` generation
  (v2, separate spec).
- `import "x.h"` inside the compiler (Option B of the import doc).
- Converting the default lib to generated bindings.
- Unions, bitfields, function-like macros, `static inline` functions.

## Decisions taken

- `@linkname` is an alias of `@dllname`, not a new mechanism.
- Function renames move to MLIR, where a symbol clash can be detected; the
  LLVM-level rename is left only as a fallback.
- `tsbindgen` is a separate executable over a reusable `TsClangImporter`
  library, shipped next to `tslang.exe`; `tslang.exe` does not link clang's
  frontend.
- v1 covers C and `extern "C"` C++ only; C++ proper is v2 on the same core.
- Parsing uses clang's C++ API, not libclang or `-ast-dump=json`: record
  layout, macro values and (for v2) mangling and CodeGen are only available
  there.
- Generated names are the C names; `--namespace` and `--strip-prefix` are
  opt-in.
- Callbacks are `Opaque`; struct-by-value functions are skipped until
  measured.
