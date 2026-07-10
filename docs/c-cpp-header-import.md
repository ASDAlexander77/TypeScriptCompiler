# C/C++ Library Integration — Converting Headers into tslang Declarations

Estimate and research: how to consume existing C/C++ libraries (`.h`/`.hpp` +
`.lib`/`.a`/`.dll`/`.so`) from tslang by converting header declarations into the
compiler's own declaration form, instead of writing `.d.ts` bindings by hand.

---

## 1. What already works today (manual FFI)

The compiler already has a complete *manual* C FFI; every mechanism needed to call
into a C library exists — only the declaration authoring is manual:

| Mechanism | Where | Notes |
| --- | --- | --- |
| `declare function name(args): T` in `.d.ts` | `MLIRGen.cpp` (ambient declarations) | Emits an external `mlir_ts::FuncOp` declaration; the symbol resolves at link/JIT time. The default lib binds dozens of libc functions this way (`strtol`, `strstr`, `toupper`, `localtime`, … in `TypeScriptCompilerDefaultLib/src/native/lib.native.d.ts`, `core/core.os.d.ts`). |
| `@dllimport` / `@dllexport` decorators | `MLIRGen.cpp` (`InternalFlags::DllImport`), `Defines.h` | Marks a declaration as imported from a DLL (`@dllimport('.')` path form supported). |
| Struct-by-value via named tuples | e.g. `type tm = [tm_sec: i32, tm_min: i32, ...]` | Tuples lower to LLVM structs; used today for `struct tm` returned by value. |
| `Reference<T>` | default lib native decls | Out-parameters / `T*` arguments (`strtol(str, end: Reference<string>, ...)`). |
| `Opaque` type | `TypeScriptTypes.td` (`TypeScript_Opaque`) | Untyped pointer (`void*`) that round-trips through casts. |
| C varargs | `func.varargs` attribute (`MLIRGen.cpp`) | `declare function printf(format: string, ...args)`-style declarations. |
| Linking | `tslang.cpp` driver: `-lib=` (static libs, `--emit=exe`), `-obj=` (object files), `-shared-libs=` (JIT), plus `*-lib-path` options | Both AOT (lld) and JIT (ORC symbol resolution) paths exist. |

So "integration with C libraries" reduces to one missing piece: **automatically
generating the `.d.ts` (or in-memory) declarations from headers** — a bindings
generator, in the family of Rust `bindgen`, Zig `translate-c`/`@cImport`,
Swift's ClangImporter, and D's ImportC.

A decisive advantage for this project: **clang is already built in-tree.** The
3rdParty LLVM build uses `-DLLVM_ENABLE_PROJECTS="clang;lld;mlir"`
(`scripts/config_llvm_*.bat/sh`), so the clang frontend libraries (and libclang)
of exactly the matching LLVM version are available with zero new dependencies.
Parsing C/C++ headers correctly (preprocessor, GNU/MSVC extensions, target ABI
knowledge) is only realistic with clang; hand-rolled header parsers fail on real
SDK headers immediately.

## 2. Type mapping: C → tslang

The mapping the generator must implement (all target types already exist in the
compiler):

| C declaration | tslang declaration | Notes |
| --- | --- | --- |
| `int8_t`/`char` (numeric use) | `i8` / `char` | |
| `uint8_t` … `uint64_t` | `u8`/`u16`/`u32`/`u64` | |
| `int16_t`/`short`, `int`/`int32_t`, `long long`/`int64_t` | `i16`, `i32` (`int`), `i64` | |
| `long` | `i32` on Windows, `i64` on Linux | Generator must resolve per target triple — clang's `ASTContext` gives exact widths, never guess from the name. |
| `size_t`/`ptrdiff_t`/`intptr_t` | `index` | |
| `float` / `double` | `f32` / `number` (`double`) | |
| `_Bool`/`bool` | `boolean` | |
| `const char*` (string semantics) | `string` | tslang `string` lowers to a char pointer (`StringType` → LLVM ptr in `LowerToLLVM.cpp`); zero-terminated, compatible with C. |
| `T*` (out-param / in-out) | `Reference<T>` | |
| `void*`, forward-declared `struct S*` (opaque handles) | `Opaque` or a branded `type SHandle = Opaque` | Branding preserves some type safety between different handle types. |
| `struct S { ... }` (complete, POD) | named tuple `type S = [field1: T1, field2: T2, ...]` | Field order/layout preserved; matches the existing `tm` pattern. |
| `union` | tuple of the largest member + accessor casts, or `Opaque` | No union type in tslang; acceptable first-cut: expose as opaque bytes. |
| `enum E { A = 1 }` | `enum E { A = 1 }` | Underlying width from clang. |
| `#define NAME 42` (object-like macro, literal) | `const NAME = 42` | Via clang's preprocessing record; only literal/`constexpr`-foldable macros; skip function-like macros (report them in a comment). |
| `typedef` | `type` alias | |
| Function pointers | function type `(a: T) => R` | Already representable; C callbacks into tslang functions work if the closure has no captured environment (thick-vs-thin function pointer distinction must be checked per site). |
| `...` varargs | `...args` + `func.varargs` | Already supported. |
| Bitfields | not representable | Emit the containing struct as opaque + generated accessor functions (phase 2), or skip with a diagnostic. |

## 3. Integration approaches — estimate

### Option A — standalone generator tool `tsbindgen` (recommended first step)

A separate executable (new target under `tslang/`, linked against clang libs from
3rdParty) that does:

```text
tsbindgen curl/curl.h -I<include paths> [--filter curl_*] -o curl.d.ts
```

1. Run clang frontend (`clang::tooling` on the compiler AST level, or the
   C `libclang` API) over the header with the right target triple/includes.
2. Walk top-level decls: functions, records, enums, typedefs, macros.
3. Apply the section-2 mapping; emit a `.d.ts` file (optionally wrapped in a
   `namespace`/module per header) that users consume with the existing
   `-lib=curl.lib` linking flags.

- `clang::tooling` + `ASTVisitor` (C++ API) is preferred over `libclang` here:
  we already build the exact clang version in-tree, so the C++ API's version
  instability doesn't apply, and it exposes everything (mangled names, exact
  record layout via `ASTRecordLayout`, macro expansion) that libclang truncates.
- Output as `.d.ts` text (not binary) so users can inspect/patch it, matching how
  the default lib is written today.

**Effort: ~3–5 weeks** for production-quality C support (functions, structs,
enums, typedefs, opaque handles, literal macros, per-target `long`/`size_t`,
allowlist filtering — real headers pull in thousands of transitive decls, so
name filtering is a required feature, not a nicety). This yields immediate value:
bind libcurl, sqlite3, zlib, SDL, most of Win32 `extern "C"` surfaces.

### Option B — in-compiler importer (`import "sqlite3.h"` / `@cImport`) — phase 2

Same mapping logic, but invoked inside the compiler: an import of a `.h` file (or
a `/// <reference header="..."/>` form) runs clang in-process and materializes
declarations directly into MLIRGen's symbol tables, skipping `.d.ts` text.

- Best UX (Zig-like), always in sync with the header.
- Costs: links clang into `tslang.exe` (binary size +~60–100 MB static, mitigable
  by making it a lazily-loaded shared component or keeping the parse in a
  `tsbindgen` subprocess with an IPC/temp-file handoff); compile-time cost of
  parsing headers per build (needs a cache keyed on header + flags hash, like the
  existing per-build defaultlib layout).
- **Effort: +3–4 weeks on top of Option A** if A's mapper is written as a reusable
  library (`TsClangImporter`) from the start — the recommended structure.

### Option C — SWIG / existing generators — not recommended

SWIG, dstep, ctypes-gen etc. target their own runtime models and would need a
custom backend anyway; writing the clang-based mapper directly (Option A) is less
work than maintaining a SWIG module, and keeps the exact-version clang advantage.

### Option D — in-process clang → LLVM IR module with declarations (ABI layer)

Idea: run clang inside the compiler over the `.h` file, have it **emit an LLVM IR
module containing the declarations**, and merge that module into the tslang-produced
`llvm::Module` (via `llvm::Linker::linkModules`, after MLIR→LLVM translation — the
transformer hook in `tslang/tslang/transform.cpp` is the natural place).

Two facts shape how this must be done:

1. **A header alone emits nothing.** LLVM IR only contains declarations that are
   *referenced*; `clang -emit-llvm header.h` produces an empty module. The importer
   must synthesize a translation unit that references each wanted symbol.
2. **IR is ABI-lowered — that is both the strength and the limitation.** The
   clang-emitted declaration for
   `struct Big f(struct Small s)` is not `Big @f(Small)`; it is e.g.
   `void @f(ptr sret(%struct.Big), i64)` on SysV or `ptr @f(ptr, ptr)` shapes on
   Win64, with `sret`/`byval`/coercion attributes. This is exactly the C ABI
   knowledge we do **not** want to reimplement in the tslang lowering — clang
   computes it for us, per target, correctly, forever. But the same lowering
   strips everything the TypeScript **type checker** needs: parameter names,
   signedness (`unsigned` vs `int` are both `i32`), `char*`-as-string vs
   byte-buffer, enums, typedefs, macros, struct field names. So an IR-only
   pipeline cannot produce the `declare function` surface — MLIRGen needs
   declarations *before* lowering, at type-check time.

Therefore Option D is not an alternative to the AST-based mapping (Options A/B);
it is the **ABI back-half of the same importer**, and the combination is stronger
than either half:

```text
            ┌── AST walk ──► TS declarations (.d.ts or in-memory)  → type checker
header.h ──►│  in-process clang (one parse)
            └── CodeGen  ──► LLVM IR module: wrappers + decls      → llvm::Linker
```

The most robust form of the CodeGen half — and the recommended one — is **wrapper
emission** rather than raw declaration emission. For each imported function whose
signature involves anything ABI-sensitive (struct by value, unions, bitfields,
`long double`, small-struct register splitting), synthesize into the clang TU:

```c
// generated TU, compiled by in-process clang with the real header
#include "curl/curl.h"
__attribute__((used)) CURLcode __ts_curl_easy_setopt(CURL *h, CURLoption o, void *p)
{ return curl_easy_setopt(h, o, p); }
// struct-by-value example: tslang always passes/returns via pointer
__attribute__((used)) void __ts_localtime(const time_t *t, struct tm *out)
{ *out = *localtime(t); }
```

tslang then declares and calls only the `__ts_*` wrappers, whose signatures use
nothing but scalars and pointers — shapes the existing tslang lowering already
gets right on every target. After `linkModules`, LLVM's inliner collapses the
wrapper at `-O1+`, so the cost is zero in optimized builds; even at `-O0` it is
one direct call. Functions with plain scalar/pointer signatures don't need a
wrapper at all — the AST-side importer declares them directly, as today.

Additional properties of this design:

- **Works in the JIT identically** — the merged module goes through the same ORC
  path; no extra `.obj` files, no separate compile step for users.
- **Kills spike-list item 1** (section 5): struct-by-value ABI correctness stops
  being tslang's problem entirely.
- **Static inline functions and function-like macros become bindable** — a wrapper
  around `static inline` code or a macro invocation gives it a real symbol; this
  is something no declaration-only generator can do, and real headers (Win32,
  glib, stb_*) are full of both.
- **C++ phase 1 collapses into the same mechanism**: the wrapper TU is compiled
  as C++, wrappers call methods/overloads/small templates through ordinary C++
  code, and export flat `extern "C"`-shaped symbols — no `@linkname`/mangling
  support needed in the tslang compiler at all (section 4's C++ 2 route becomes
  the default C++ route, but in-memory, without shipping wrapper `.cpp` files).
- Cost: links clang CodeGen libraries into the importer host (`tsbindgen` first;
  the compiler itself only if/when Option B lands), and IR-level linking requires
  matching target triple + data layout between the clang invocation and the tslang
  module (drive both from the same `TargetMachine` settings).

**Effort: ~1–2 weeks on top of Option A's AST mapper** (the clang driver setup is
shared; the additions are wrapper-TU synthesis, `EmitLLVMOnlyAction`, and the
`linkModules` merge + a flag to dump the merged IR for debugging).

## 4. C++ support — separate, phased problem

C is a 90%-mechanical translation; C++ is not. Realistic phasing:

| Phase | Scope | Approach | Effort |
| --- | --- | --- | --- |
| C++ 0 | `extern "C"` blocks in C++ headers | Falls out of Option A for free (clang parses the header as C++, importer takes only C-linkage decls) | included in A |
| C++ 1 | Free functions, non-template classes with methods (no virtuals crossing the boundary), overloads | Bind by **mangled name**: clang provides Itanium/MSVC mangled symbols (`clang::MangleContext`); emit `declare` per overload with the mangled name attached (needs a small compiler addition: a `@linkname("?foo@@YAHH@Z")` attribute on declarations so the emitted `FuncOp` symbol differs from the TS-visible name). Methods become functions taking `this: Reference<S>` first. Constructors/destructors bound explicitly; caller manages object memory (`GC_malloc_atomic` for PODs / explicit new-delete pair bindings). | 2–4 weeks after A |
| C++ 2 | Templates, virtual dispatch, exceptions, STL types | **Do not bind directly.** Generate a flattened `extern "C"` wrapper `.cpp` (tsbindgen emits both the wrapper and its `.d.ts`), compiled with the system C++ compiler and linked via existing `-obj=`. Explicit template instantiations only; exceptions caught at the wrapper boundary and converted to error codes (C++ exceptions must never unwind into tslang frames — different personality functions, especially in JIT where Win64 unwind info is custom, see `docs/jit-gc-static-roots.md` context). | 3–6 weeks, per-need |

The C++ ABI reality (mangling differences, MSVC vs Itanium, RTTI, std::string
layout) is why every successful system (Swift, Rust `cxx` crate, Zig) either uses
clang as the importer *and* restricts the surface, or generates C wrappers. The
phase C++ 2 wrapper route is the honest long-term answer for arbitrary C++.

## 5. ABI risk areas to validate early (spike list)

These decide whether generated declarations actually work, and should be a 2–3 day
spike before committing to the full tool:

1. **Struct-by-value calls** — Win64 passes structs > 8 bytes by hidden pointer,
   SysV splits into registers. The existing `tm`-by-value binding suggests the
   MLIR→LLVM lowering already produces plain LLVM struct types and lets LLVM's
   backend apply C ABI rules — verify against clang-compiled callees for sizes
   1/2/4/8/9/16/17 bytes on both platforms. Any mismatch means tsbindgen must
   emit `sret`/byval-shaped signatures itself (clang's `CodeGen` ABI info can be
   queried for ground truth).
2. **`long`/`size_t`/`wchar_t` width matrix** per target triple (wasm32 included).
3. **Callbacks** — passing a tslang function as a C function pointer: confirm a
   non-capturing function lowers to a bare pointer (captures need trampolines —
   `TRAMPOLINE_SIZE` in `Defines.h` hints this exists; verify GC interaction, a
   trampoline allocated by `GC_malloc` handed to a C library outliving the caller
   is invisible-root territory).
4. **GC vs C-owned memory** — pointers returned by the library are not GC-scanned
   objects (fine, Boehm ignores unknown ranges), but a GC pointer stored *into*
   C-owned memory is invisible to the collector → collected while alive. Rule for
   generated bindings: parameters that retain pointers must be documented/annotated;
   provide `GC_add_roots`-style pinning helper in the default lib. (Same class of
   bug as the JIT statics issue; see `docs/jit-gc-static-roots.md`.)
5. **JIT path** — `-shared-libs=libcurl.dll` symbol resolution already works for
   the runtime; confirm generated decls resolve identically under `--emit=jit`
   (Win32 JIT has custom symbol binding, see `jit.cpp`).

## 6. Recommended plan

1. **Spike (2–3 days):** hand-write `.d.ts` bindings for one real library
   (sqlite3 is ideal: pure C, opaque handles, callbacks, strings, int64) and run
   its smoke test AOT + JIT on Windows/Linux. This validates section 5 with zero
   new code and produces the golden file for the generator.
2. **`tsbindgen` v1 (3–5 weeks):** clang-tooling-based, C only, allowlist
   filtering, per-target type resolution, `.d.ts` output; structured as a reusable
   `TsClangImporter` library + thin CLI. Ship with generated bindings for
   sqlite3/zlib/curl as tests (curl is already a runtime dependency on Linux).
3. **ABI layer via clang CodeGen (1–2 weeks, Option D):** wrapper-TU synthesis +
   `EmitLLVMOnlyAction` + `llvm::Linker::linkModules` merge into the tslang
   module — removes the struct-by-value ABI risk entirely and makes
   `static inline` functions and function-like macros bindable.
4. **C++ phase 1 (2–4 weeks):** simple classes/overloads through the same wrapper
   mechanism (wrapper TU compiled as C++ exporting flat symbols); the
   `@linkname` mangled-name route stays as a fallback for wrapper-free binding.
5. **In-compiler import (3–4 weeks, optional):** `import "header.h"` reusing
   `TsClangImporter`, with a parse cache.

Total to a genuinely usable "point at a C library and call it" experience:
**~1–1.5 months** (steps 1–2, with step 3 folded in if struct-heavy APIs matter
early); full C++ story is incremental on top.
