# 32-bit compilation

Design for compiling TypeScript to 32-bit targets, with `i686-pc-windows-msvc`
as the proof target and width-correctness that generalizes to any 32-bit triple.

Status: approved design. Phases 1-4 implemented.

## Why now

Until #333, every compiled program linked `LLVMSupport.lib` and
`LLVMDemangle.lib`. Those come from the prebuilt LLVM in `3rdParty/llvm/x64`,
which exists only as x64 — so a 32-bit link was blocked by a library we could
not rebuild cheaply. #333 removed that dependency. Every remaining x64 blocker
is a library this repository builds from source.

## Measured baseline

Probed with the Sep-18 build (`__build/tslang/windows-msbuild-2026-release`),
target `i686-pc-windows-msvc`. These are measurements, not estimates.

| Case | Result |
| --- | --- |
| `--emit=obj -mm=none --no-default-lib` | Works. `hello.obj` machine type `0x014c` (`IMAGE_FILE_MACHINE_I386`). |
| `--emit=exe -mm=none --no-default-lib` | Works. `hello.exe` PE machine `0x014c`; runs and prints. |
| `LLVMSupport.lib` in the link | Gone, as of #333. (The older `__build/tslang/msbuild/x64/release` build still drags it in.) |
| Exceptions | `unresolved external symbol __CxxThrowException`; linker hints `__CxxThrowException@8`. |
| `-mm=gc` | `gc.lib : warning LNK4272: library machine type 'x64' conflicts with target machine type 'x86'`, then unresolved `_GC_init`, `_GC_enable_threads`. |
| `TypeScriptAsyncRuntime.lib` | Same x64/x86 machine-type conflict. |

clang's driver locates the x86 CRT and Windows SDK import libraries on its own.
No toolchain-discovery work is needed.

### Two findings that reduced scope

Both were assumptions that probing disproved. They are recorded because the
obvious reading of the emitted IR suggests the opposite.

**The async runtime ABI is not a width bug.** Emitted IR declares
`mlirAsyncRuntimeAddRef(ptr, i64)`, `mlirAsyncRuntimeCreateValue(i64)` and
similar regardless of triple, which looks like a 64-bit-shaped ABI. It is not:
`lib/AsyncRuntimeCommon.inc` declares these parameters as `int64_t`, not
`size_t`. `i64` is correct on both arches. Nothing to change in the runtime API.
(Phase 4b found the one real width bug on this path elsewhere: the coroutine
frame allocator, `aligned_alloc`, *is* `size_t`, and upstream declares it
`(i64, i64)`.)

**The Win32 EH struct layouts are already correct for x86.** `ThrowInfo` is
emitted as `{i32 x 4}` and `CatchableType` as `{i32 x 7}`
(`include/TypeScript/LowerToLLVM/LLVMRTTIHelperVCWin32.h`). On x64 those i32
fields hold image-base-relative RVAs. On 32-bit MSVC the corresponding fields
hold absolute pointers — which are also 4 bytes. The layouts coincide. What
differs is only the *value* stored: x86 must not subtract the image base.

## Architecture

One new type, `TargetInfo`, constructed once in `prepareOptions()` from
`llvm::Triple` and carried on `CompileOptions`. The triple itself stays on
`CompileOptions` only as the existing `moduleTargetTriple` string. `TargetInfo`
holds two kinds of fact:

- **Widths** — `pointerBits`, used for pointers, sizes and indices alike.
- **Policy** — `usesImageBaseRelativeEH`, `stdcallDecoratesCxxThrow`,
  `supportsInProcessJit`.

`TypeHelper` gains `getSizeType()` and `getPointerIntType()` backed by it.
`TypeHelper` took only an `MLIRContext*`; it now takes an optional
`const CompileOptions *`, defaulting to null. Most of its ~150 construction
sites use only width-independent helpers (`getI8Type`, `getPtrType`,
`getVoidType`) and are unchanged; only the sites that need a target width pass
the options. `getSizeType()` called on a `TypeHelper` built without them fails
via `report_fatal_error` in every build configuration, rather than guessing a
width.

### Naming policy predicates

Policy predicates are named for what they decide, not for the architecture that
currently needs them: `usesImageBaseRelativeEH`, never `isX64`. This is what
keeps phase 3 from smearing `arch == x86_64` across three RTTI helpers, and it
is what makes the answer reviewable in one place next to the comment explaining
it.

### Replacing the arch list

`tslang/opts.cpp` currently derives `sizeBits` from a hand-maintained list of
64-bit architectures. The list is wrong: it marks `aarch64_32` as 64-bit, but
ARM64_32 is an ILP32 target whose pointers are 32 bits. It is replaced by
`llvm::Triple::getArchPointerBitWidth()`, which is what makes "any 32-bit
triple" true rather than "the 32-bit triples someone remembered".

### Arch in the library layout

`x86` and `x64`, matching the existing `3rdParty/gc/x64`. Arch is the outermost
dimension under each component, so a whole arch tree ships or is deleted as one
unit and existing x64 paths keep their shape:

- `3rdParty/gc/x86/release/`, `3rdParty/gcdll/x86/release/`
- `defaultlib/{lib,dll}/x86/{debug,release}/{gc,rc,none}/`

Those are the build trees. The compiler finds the x86 libraries in the `x86`
subdirectory of the lib path it is given, so the gc scripts also copy their
output under the x64 lib directories (see phase 2, "As built").

Strict, no fallback — the rule the default-lib layout already follows. A
missing arch tree is an error that names the arch and the script that builds
it. It never falls back to another arch: that link succeeds and then corrupts
memory, which is far harder to diagnose than a directory that is not there.

## Phase 1 — Target width correctness

`TargetInfo` lands. `TypeHelper` takes an optional `CompileOptions`. The arch
list is replaced.

Eighteen `getI64Type()` call sites are in the two Win32 RTTI helpers
(`include/TypeScript/LowerToLLVM/LLVMRTTIHelperVCWin32.h` and
`include/TypeScript/MLIRLogic/MLIRRTTIHelperVCWin32.h`, nine each); those are
phase 3's, since phase 3 rewrites that arithmetic anyway. Outside them there are
18 call sites across `lib/` and `include/`. A grep for `getI64Type()` returns
21 lines there, but three are not call sites: the two definitions of
`getI64Type` (`TypeHelper` and `MLIRTypeHelper`) and a comment mentioning it.
Phase 1 audits the 18, and each is classified in a comment as one of:

- **genuinely 64-bit** — the language's `number` is f64; runtime ABI
  parameters declared `int64_t`; anything whose width is fixed by a contract
  outside the target.
- **target-width** — pointer arithmetic, object sizes, GEP indices.

Result: 13 are target-width and 5 genuinely 64-bit. The audit's value is not
the count of changes but that the classification becomes explicit and survives
future edits. It also found two latent 32-bit bugs, both fixed:

- The `-1` sentinel for an optional interface member the object does not
  provide was produced at `i64` in MLIRGen while the lowering that tests for it
  compared at a different width. Producer and consumer now both use the target
  width.
- A pointer converted to a float went through `sitofp`. With the intermediate
  integer now as wide as the pointer, an address at or above `0x80000000` on a
  32-bit target would read back as a negative float. It is now `uitofp`.

`target datalayout` is set at IR emission rather than only later in
`tslang/obj.cpp`, so `--emit=llvm` output is self-describing.

**Gate.** `--emit=llvm` for `i686-pc-windows-msvc`, `wasm32-unknown-unknown`
and `x86_64-pc-windows-msvc` emits the correct datalayout. Verifiable with no
native builds.

The gate originally also required that no 32-bit target emit 64-bit-shaped size
or pointer arithmetic. Phase 1 does not meet that, and the requirement moves to
phase 2. The lowering's `LLVMTypeConverter` is built with LLVM's default data
layout for every non-wasm target (`TypeScriptToLLVMLoweringPass::runOnOperation`
in `lib/TypeScript/LowerToLLVM.cpp`, ~7255-7268), so on i686 `getIntPtrType`,
struct layout, union sizing and alignment still use 8-byte pointers. Phase 1
fixes the widths MLIRGen and the lowering choose explicitly; it does not change
what the type converter derives.

## Phase 2 — 32-bit native build matrix

The first task, and a gate on any native 32-bit test, is to build the type
converter's data layout from the module's `llvm.data_layout` instead of LLVM's
default. This changes x64 output — the default layout aligns `i64` at 32 bits,
x64's real layout at 64 — so it needs its own x64 diff and review rather than
riding along with the build-matrix work. It also removes the hardcoded wasm32
layout string, which has `f128:64` and lacks `i128:128` and so matches neither
of LLVM's derivations.

The compiler itself stays x64 and cross-compiles. Building `tslang.exe` as a
32-bit binary is out of scope.

**As built.** The lookup rule is one path for both machines: for an x86
target, the compiler takes each library from the `x86` subdirectory of the lib
path it is given (`--gc-lib-path`, `--gc-shared-lib-path`, `--tslang-lib-path`
or their environment variables). The x64 layout stays flat and unchanged.

| Library | Built by | Installed to | Copied to (what the compiler reads) | Flag value, for x64 and x86 alike |
| --- | --- | --- | --- | --- |
| static Boehm `gc.lib` | `scripts\build_gc_<cfg>_vs_x86.bat` | `3rdParty/gc/x86/<cfg>/` | `3rdParty/gc/x64/<cfg>/lib/x86/gc.lib` | `--gc-lib-path=3rdParty/gc/x64/<cfg>/lib` |
| Boehm DLL: `gc.lib` + `gc.dll` | `scripts\build_gc_<cfg>_shared_vs_x86.bat` | `3rdParty/gcdll/x86/<cfg>/` (`lib/`, `bin/`) | `3rdParty/gcdll/x64/<cfg>/lib/x86/` (both files) | `--gc-shared-lib-path=3rdParty/gcdll/x64/<cfg>/lib` |
| `TypeScriptAsyncRuntime.lib` | `scripts\build_tslang_runtime_<cfg>_x86.bat` | `__build/tslang-runtime/<cfg>/x86/` | (installed in place) | `--tslang-lib-path=__build/tslang-runtime/<cfg>` |

`<cfg>` is `release` for `--opt` and `debug` otherwise, the same choice that
picks the CRT. Each gc script installs into its own x86 tree, then copies into
the x86 subdirectory of the matching x64 lib directory. It adds only that
subdirectory; the x64 files are not touched. The shared build's `gc.dll` goes
beside `gc.lib` because the compiler looks for `gc.dll` next to the import
library first and in `../bin` second. From `lib/x86`, `../bin` would be
`lib/bin`, which does not exist. So the value users already pass for x64
(the `lib` directory, as the test suite does) works for x86 unchanged.
`prepare_3rdParty.bat <cfg> x86` checks the copied location, so an x86 build
made before the copy existed is re-run and gets the copy.

There is no x86 `TypeScriptRuntime` (the JIT DLL): `--emit=jit` refuses a
non-host arch, so nothing would load it.

`tslang/exe.cpp` resolves the three paths for the target, not the host. For
Windows x86 and x64 it reads the COFF machine of the library the link will use
(`Dump::coffMachine`), and of the `gc.dll` it picked. A mismatch is a hard error
naming the file, its machine and the target's, rather than LNK4272 plus a list
of unresolved symbols. A missing x86 build is a hard error naming the script
and the directory to pass. `tslang/test/check-x86-libs.sh` covers the error
paths. `tslang/test/check-x86-run.sh` links and runs 32-bit programs from the
real layout under `-mm=gc`, `rc` and `none`.

**Gate.** `-mm=gc` and `-mm=rc` link and run a 32-bit hello world.

## Phase 3 — Win32 x86 exception handling

Two changes, both narrow because of the finding above.

1. `_CxxThrowException` is `__stdcall` on x86, so its symbol is
   `__CxxThrowException@8`. **As built:** a module pass,
   `CxxThrowCallingConvPass` (`lib/TypeScriptExceptionPass`), sets
   `CallingConv::X86_StdCall` on the declaration and on every call and invoke
   of it. `transform.cpp` adds it after `Win32ExceptionPass` and before the
   optimization pipeline, gated on `stdcallDecoratesCxxThrow`. It is one pass
   rather than a fix at each declaration because three places create calls:
   the MLIR lowering (`ThrowLogic.h`), the rethrow `Win32ExceptionPass`
   synthesizes, and its `ToInvoke`. A call whose convention differs from its
   callee's is undefined behaviour, which InstCombine turns into `unreachable`.
   Any use other than as a direct callee is a hard error. The commented-out
   block that used to sit in `Win32ExceptionPass::getThrowFn` is gone.
2. The cross-references in `ThrowInfo`, `CatchableType` and
   `CatchableTypeArray` are built by one `ehReference` function in each Win32
   RTTI helper. When `usesImageBaseRelativeEH` holds it emits
   `trunc(ptrtoint - ptrtoint __ImageBase)` at `sizeBits()` width, which is
   exactly the old x64 output; otherwise it emits `ptrtoint` to i32, and
   `__ImageBase` is not declared. Only the MLIR-level helper
   (`MLIRRTTIHelperVCWin32.h`) emits these globals; the LLVM-dialect helper's
   table emitters (`LLVMRTTIHelperVCWin32::setRTTIForType` and what it calls)
   have no callers, and were changed only so neither helper can compute RVAs
   on x86. `usesImageBaseRelativeEH` is decided by pointer width, as clang's
   `MicrosoftCXXABI::isImageRelative()` does, so 32-bit ARM Windows is absolute
   as well.

**Testing.** Tests assert that control flow reaches the catch, and on
unwinding order. They do not assert on the catch variable's contents: reading a
catch variable's value is a known-unreliable area independent of this work, and
a test written on it would fail for reasons that have nothing to do with 32-bit
support.

`tslang/test/check-x86-eh.sh` (hand-run, needs the phase 2 x86 libraries,
release only) checks the IR and object form of both changes, then links and
runs `tslang/test/x86/eh_order.ts` (an exact unwinding trace) and eleven
exception corpus files as 32-bit exes under gc, rc and none. Debug x86 builds,
the `using`/dispose corpus files and a throw crossing an x86 `--emit=dll`
boundary into a C++ `catch (int)` were verified by hand.

**Gate.** throw/catch, nested catch, try/finally and rethrow behave correctly
as 32-bit binaries. Met: no compiler change beyond the two above was needed.

## Phase 4 — Default library and the suite

`getDefaultLibSubDir` in `include/TypeScript/Defines.h` gains an arch segment,
giving `defaultlib/{lib,dll}/x86/{debug,release}/{gc,rc,none}/`. The
default-lib build scripts and the test runner gain an arch flag.

Async/await and generators are measured here rather than in an earlier phase.
They depend on phases 2 and 3 and cannot run before both are in place. Nothing
in the probing suggests they are separately broken; if they are, that is a
finding of this phase.


**Gate.** The same suite that runs at 64 bits runs at 32 bits, green. Genuine
failures are triaged and recorded the way the Debug-suite failures are, not
silently excluded.

**Split.** A corpus probe (`tslang/test/probe-corpus.sh`: every corpus file as
a single-file exe, x64 and i686 × gc/rc/none) found 15 files failing at i686
only, so phase 4 is three PRs:

- **4a — union storage as bytes (done).** A tagged union stored its value as
  its largest member's struct; bytes in that struct's padding were not carried
  by a value copy, so another member's field there was lost — at x64 for some
  layouts, at i686 for common ones (a pointer followed by an 8-aligned `f64`).
  The value field is now a packed `<{[N/P x iP], [N%P x i8]}>` (P the pointer
  size), with the unused tail zeroed when a smaller member is stored. Also
  fixed: `castLLVMTypes` copied small aggregates as one pointer (4 bytes at
  i686), and a module-level union initialized with a constant now gets a
  global constructor instead of failing to compile. 29 probe rows newly pass,
  none regress; x64 `.text` is unchanged within 0.03%.
- **4b — async at i686 (done).** Upstream `ConvertAsyncToLLVM` assumes 64 bits
  twice. It declares the coroutine frame allocator `aligned_alloc(i64, i64)`
  whatever the target; at i686 the callee reads `size = 0` and the frame
  overflows its block (the crash: `mlirAsyncRuntimeEmplaceToken(null)` on a
  worker thread). And its 64-bit-index converter meets tslang's 32-bit one,
  leaving `i32 -> index -> i64` casts into `mlirAsyncRuntimeCreateGroup` that
  LLVM translation rejects (`00for_await`). The upstream pass is prebuilt, so
  two tslang passes repair its output when `sizeBits() < 64`:
  `AsyncTargetWidthPass` (after `ConvertAsyncToLLVM`) retypes `aligned_alloc`
  to pointer width, and `AsyncIndexCastPass` (after `LowerToLLVM`) folds the
  cast chains into `sext`/`trunc`. x64 never runs them. All six async corpus
  files run at i686 under gc (async needs Boehm's thread API, so under rc and
  none it does not link at x64 either); wasm32 had the same allocator bug.
- **4c — the suite at i686 (done, without the x86 default library).**
  `test-runner -x86` builds and runs a test as a 32-bit program (i686 triple,
  the x86 library directories, `/machine:x86`). Configure with
  `-DTSLANG_TEST_X86=ON`, then run `ctest -C Release -L x86`; the option is off
  by default, so the default suite and CI keep their 2769 test names (plus the
  two foreign-target import tests below). With it on, every `test-runner`
  registration without `-jit` gets a `test-x86-...` twin: 1398 twins. The
  1357 `-jit` registrations get none,
  because `--emit=jit` runs in the x64 compiler process and refuses a foreign
  arch. Only one test file, `02funcs_vararg.ts`, is JIT-only at the file level;
  every other file has a compile registration and so a twin. The cmake-script
  tests get no twin via `tslang_add_test` either. There were 12 of them:
  `rc-debug-info`, `gc-shared-auto`, the two default-library collector tests
  and the 8 ownership-verifier shards. The two foreign-target import tests
  below make 14.
  1395 twins pass; 3 are registered `DISABLED` in
  `tslang/test/tester/x86-exclusions.cmake`: `internals` under gc, rc and
  none, because its `inline_asm<i64>` with an `=r` constraint needs a 64-bit
  register that i686 does not have. Importing a DLL now reads `__decls` from
  the file when the target's arch or OS is not the host's
  (`targetInfo.supportsInProcessJit` is false); the host path, which loads the
  DLL, is unchanged. Such an import must be a PE DLL for the target's machine,
  or it is an error. Two cmake-script tests in the default suite,
  `test-compile-foreign-target-import-read` and `-errors`
  (`foreign-target-import.cmake`), cover that path without x86 libraries: an
  x64 DLL imported by a program compiled for x86_64 Linux, then a non-PE file
  and an x64 DLL imported into an x86 program.
  The last three failures were not a compiler defect but Windows installer
  detection: a 32-bit exe with no `requestedExecutionLevel` manifest, whose
  name contains `setup`, `install`, `update` or `patch`, needs elevation and,
  unelevated, never starts (`...-abstract-virtual-dispatch`: "dispatch" holds
  "patch"). x64 exes are exempt. tslang now links an asInvoker `RT_MANIFEST`
  resource object into every Windows x86 `--emit=exe` (not DLLs, which are not
  launched, and not x64, whose output stays unchanged); it builds the object
  itself because `-manifest:embed` makes MSVC's link.exe run `rc.exe`, which is
  on `PATH` only in a developer prompt. The runner passes lld-link
  `/manifest:embed /manifestuac:...` for x86 exes. `check-x86-run.sh` builds
  hello as `my_setup_patch.exe` and runs it.

**As built — the x86 default library (done; was deferred from 4c).**
`getDefaultLibSubDir` (`include/TypeScript/Defines.h`) gained a required `arch`
parameter and `DEFAULT_LIB_ARCH_X86`, giving the layout named above:
`defaultlib/{lib,dll}/x86/{debug,release}/{gc,rc,none}/`, with no `x86`
segment (and no change) for every other target, x64 included. `exe.cpp`
passes `DEFAULT_LIB_ARCH_X86` only for a Windows x86 target, names the arch
and the `TSLANG_ARCH=x86` build step in the missing-directory error, and
checks the COFF machine of `TypeScriptDefaultLib.lib` the same way it already
checked the collector and the runtime (an x64 default library in an x86
program is refused before the link, not left to become unresolved symbols).
`jit.cpp` and `defaultlib.cpp` pass `""` (host-only paths). In the separate
`TypeScriptCompilerDefaultLib` repository, `TSLANG_ARCH=x86` (set before
`build.bat`) switches `vcvarsall.bat` to `x86`, adds
`-mtriple=i686-pc-windows-msvc` to every `tslang.exe` invocation, and routes
every build and staging path through the same layout; `TSLANG_ARCH` unset or
`x64` produces the same output tree as before. That repository's own 150-test
suite passes at 150/150, release and debug, as i686 executables against the
static x86 tree (`tests.ps1`).

One probe-only failure needed a real fix: `paragraph.replaceAll(/Dog/gi,
"ferret")` segfaulted after its first `console.log` at i686 (149/150; x64
passed). Root cause: `RegExp.replaceAll` in `lib.ts` called the native
`declare function regexp_replace` with 3 of its 4 parameters; tslang does not
check declare-function call arity (TypeScript's TS2554), so the omitted
argument read whatever a register or a stack slot happened to hold - luck
that held at x64 and did not at i686. Fixed in the default library, not in
the compiler: `RegExp.replaceAll`'s regex branch in `lib.ts` now passes the
replacement string as the native call's 4th argument instead of omitting it.

Compiler-side coverage: `check-x86-libs.sh` gained the default-library
error-path cases (no x86 tree; an x64 `TypeScriptDefaultLib.lib` staged where
the x86 one belongs; an x64 control case). `check-x86-run.sh` gained a
default-library section, skipped (not failed) when
`TypeScriptCompilerDefaultLib`'s x86 tree is absent: `defaultlib_smoke.ts`
(Array, Map, string, Date and RegExp, including `replaceAll(regex, ...)`)
under gc, rc and none release, the gc case again against the debug tree
(`--di`), and a gc `--emit=dll` library linking the default library plus an
exe that imports it (the shape of `defaultlib-collector.cmake`'s compile
mode, reused in bash), with `TypeScriptDefaultLib.dll` and the x86 `gc.dll`
copied beside the importing exe. `tester/CMakeLists.txt` hand-registers
`test-x86-compile-gc-defaultlib-collector` (labelled `x86`) as the x86 twin
of the existing compile-mode default-library collector test, extending
`defaultlib-collector.cmake`'s own skip test to look under the `x86` tree
when an `X86` flag is passed; there is still no JIT twin (the JIT refuses an
x86 target, unrelated to the default library). With `-DTSLANG_TEST_X86=ON`:
1396 of the 1399 x86-labelled tests pass (the 1395 `tslang_add_test` twins
plus the new collector twin), 3 stay `DISABLED`. With it `OFF`,
`CTestTestfile.cmake` is unchanged except for `_BACKTRACE_TRIPLES` line
numbers (this file added lines above existing `add_test` calls), and the
default suite stays 2771/2771.

Remaining i686-only probe failures after 4b: `internals` (an `inline_asm<i64>` with an `=r` constraint, which no single
i686 register can satisfy); `02funcs_vararg` (unresolved `_printf`; it is
commented out of the suite).

## Error handling

Three failures get explicit, actionable messages instead of downstream
confusion:

1. **Missing per-arch library tree** — names the arch and the script that
   builds it. Not a fallback to another arch.
2. **Arch mismatch in a library that is present** — reported against the
   offending path at resolution time, via the machine-type check, rather than
   as a linker warning buried among unresolved symbols.
3. **`--emit=jit` with a non-host arch** — `tslang.exe` is an x64 process and
   cannot execute i386 code in-process. Refused up front with a message saying
   so. What the combination does today must be confirmed before this guard is
   written; the guard is `supportsInProcessJit`.

## Open issues

- **ABI-by-environment triples.** For `x86_64-…-gnux32`, mips64 `gnuabin32` and
  aarch64 `ilp32`, `getArchPointerBitWidth()` says 64 while the TargetMachine's
  data layout says `p:32`. Nothing asserts the two agree. Proposed fix: a check
  in `lib/TypeScript/MLIRGenModule.cpp` comparing
  `createDataLayout().getPointerSizeInBits(0)` with `compileOptions.sizeBits()`.
- **Importing an x86 DLL (phase 4) — closed in 4c.** Compiling a program that
  imported an x86 DLL failed with `./lib.dll: Can't open: Unknown error (0xC1)`:
  `MLIRGenImpl::mlirGenImportSharedLib` loaded the DLL into the x64 compiler to
  read its `__decls`. For a foreign target (arch or OS) it now reads the string from
  the file (`Dump::readExportedCString`); the x86 `-shared` twins pass.
- **The x86 default library — closed (phase 4, "As built" note above).**
  `getDefaultLibSubDir` has the arch segment, the `TypeScriptCompilerDefaultLib`
  repository builds the x86 tree under `TSLANG_ARCH=x86`, and
  `test-x86-compile-gc-defaultlib-collector` gives the compile-mode collector
  test its x86 twin. x86 programs no longer need `--no-default-lib`.
- **tslang does not check declare-function call arity (TypeScript's TS2554).**
  Found while root-causing the x86 `replaceAll(regex, ...)` segfault above: a
  call to a `declare function` with fewer arguments than it declares compiles
  regardless, and the callee reads whatever a register or stack slot happens
  to hold for the missing ones. That hid the `RegExp.replaceAll` bug at x64
  (the garbage argument happened to be usable) until the x86 build read a
  garbage stack slot instead. No arity check exists for `declare function`
  calls generally, only for the specific case this one bug surfaced.
- **`"xxx".replaceAll("", "_")` gives `___` instead of `_x_x_x_` (pre-existing,
  both architectures).** An empty-pattern `replaceAll` should insert the
  replacement between every character; it currently inserts once per
  distinct-looking match instead. Confirmed unchanged at x64 and i686; not
  investigated further here.
- **Neither CI nor the release workflows build an x86 default library.** Both
  build x64 only, so `-DTSLANG_TEST_X86=ON` and `check-x86-run.sh`'s
  default-library section are local-only checks today; the x86-labelled ctest
  run and the default suite's x64-only default-library collector tests never
  run together in the same CI job. The `test-runner -x86` twins (the 1395
  `tslang_add_test` twins) still pass `--no-default-lib` unconditionally, the
  same as their x64 counterparts - `test-runner.cpp` was not changed by this
  work - so they cover the arch/ABI matrix, not the default library; only
  `test-x86-compile-gc-defaultlib-collector` and `check-x86-run.sh` exercise
  the x86 default library itself.
- **Per-test flags miss the entry file of multi-file tests (not 32-bit;
  pre-existing).** Found in phase 4c: in `test-runner`, `-mm`, `-fast-math`
  and `--gctors-as-method` reach only the non-entry files of a multi-file test,
  so the entry file compiles with the defaults.
- **`mlirAsyncRuntimGetNumWorkerThreads` is declared `() -> i32` at 32 bits**
  (upstream declares it returning `index`) while the runtime returns `int64_t`.
  Harmless at i686 cdecl (the low half is in EAX); at wasm32 it would be a
  signature mismatch. Unreachable from TypeScript today: only upstream's
  async-parallel-for creates the op.
- **wasm32 `-mm=none` async crashes the compiler (pre-existing).** Compiling an
  async program for wasm32 without GC usually fails with 0xC0000005, before and
  after phase 4b. Suspected: `MemAllocFixPass` erasing the `free` declaration
  that the coroutine frame's `free` calls still use.
- **x86 EH type names use the 64-bit mangling (phase 4, interop only).** The
  hardcoded names in `LLVMRTTIHelperVCWin32Const.h` (`??_R0PEAD@8`,
  `_CT??_R0PEAD@88`) carry the `__ptr64` qualifier and 64-bit sizes. tslang's
  throw and catch share them, so tslang programs are unaffected, and a C++
  `catch (int)` across a DLL boundary works; catching a *pointer* type thrown
  by MSVC-compiled x86 code would not match.
- **`00catch_value*` at i686.** Both exit 0 under gc, rc and none, release and
  debug, as of phase 3.
- **Untyped `catch (e)` does not catch a thrown number (not 32-bit; x64 too).**
  The `ThrowInfo` follows the thrown value's static type
  (`MLIRGenStatements.cpp` ~1049-1067) while an untyped catch variable gets a
  `void*` handler (`??_R0PEAX@8`), so `throw 1` / `throw "s"` escape it and the
  process exits 127 with its earlier output unflushed. `throw <any>x` and class
  objects are caught. Pre-existing; `eh_order.ts` uses `catch (e: TypeOf<1>)`.

- **A global initialized from a DLL-exported global reads garbage (not 32-bit;
  x64 too).** Found in phase 4a review: in an importer, `const copiedN: number
  = cN` where `cN` is exported by a `-shared` module prints a denormal (e.g.
  3.04e-312) with both the old and new compiler; a union copy crashes silently.
  Likely the importer's global constructor runs before the import is bound.

## Out of scope

- `-m32` shorthand. `-mtriple=i686-pc-windows-msvc` already works and is the
  mechanism; a second spelling earns nothing.
- Building `tslang.exe` itself as a 32-bit binary.
- 32-bit targets other than `i686-pc-windows-msvc` as *proof* targets. Phase 1
  makes width-correctness general, and `wasm32` continues to be exercised, but
  only `i686-pc-windows-msvc` gets a full native build matrix and suite run.

## Decisions taken

| Decision | Choice | Alternative rejected because |
| --- | --- | --- |
| Where target facts live | `TargetInfo` on `CompileOptions`, derived once from `llvm::Triple` | Querying the `Triple` at each site scatters policy across every width-dependent call site and relocates the arch-list rot rather than fixing it. Querying MLIR's `DataLayout` cannot express EH policy at all, and much width-dependent code runs in MLIRGen where no `DataLayout` is in scope. |
| Arch naming | `x86` / `x64` | Matches `3rdParty/gc/x64`, which already exists. |
| Arch position in paths | Outermost under each component | A whole arch tree ships or is deleted as one unit; x64 paths keep their shape. |
| Missing arch tree | Hard error | A fallback links and then corrupts memory. |
| Suite scope at 32 bits | The same suite as 64 bits | A declared subset hides exactly the failures this work exists to find. |
