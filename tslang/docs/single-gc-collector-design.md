# One collector per process: design proposal

Status: **PR 1 merged** (steps 1, 2, 5 - Windows); **PR 2** on branch `gc-shared-lib-auto`: step 3,
step 4 (by fingerprint, not a marker), step 6, the CMake template, the debug default library, and
Linux. Open: suite tests that keep the default library. See [Progress](#progress) at the end.

## Problem

Under `-mm=gc`, every tslang binary that links Boehm statically carries its **own collector**:
its own heap, its own roots, its own mark bits. When two such binaries share a process and one
of them holds a pointer to an object the other allocated, the owning collector never sees that
reference. So it frees the object, the memory is reused, and the holder reads a plausible wrong
value. Nothing crashes; the data is silently wrong.

Item 5ao (reference-counting-evaluation.md §9.76) fixed this for **exe + user DLL** by linking
both against a Boehm DLL. It missed the binary that is in almost every process: the **default
library DLL**.

### Evidence (2026-09-12)

`TypeScriptDefaultLib.dll` (`defaultlib/dll/release/gc`) imports only `ntdll`, `WINHTTP` and
`KERNEL32`, yet contains Boehm's internals (`GC_mark_some`, `GC_stop_world`, `GC_thr_init`).
It was linked with `-lgc` against the static `gc.lib`.

Repro: hold 2000 strings built by the default library (`padStart`) in an array the program
allocated, then churn strings of **different** content (`repeat`) so a collection runs. Each
case was also run with `GC_INITIAL_HEAP_SIZE=1GB` as a no-collection control.

| Process | Collectors | Held strings wrong | Control |
| --- | --- | --- | --- |
| AOT exe, static default lib | 1 | 0 / 2000 | - |
| JIT, default lib (the normal `--emit=jit` path) | 2: `TypeScriptRuntime.dll` + default-lib DLL | **2000 / 2000** | 0 |
| AOT exe + user `-shared` DLL, both already on `gc.dll` | 2: `gc.dll` + default-lib DLL | **1984 / 2000** | 0 |

The suite never saw this because `test-runner` passes `--no-default-lib` to every test.

## Where Boehm is linked today

| Binary | How it gets Boehm | Loaded together with |
| --- | --- | --- |
| user exe (`--emit=exe`) | `-lgc` from `GC_LIB_PATH`: static, or `gcdll` if pointed there | user DLLs it imports |
| user DLL (`--emit=dll`) | same `-lgc` | the exe; the default-lib DLL (links `dll/` import lib) |
| `TypeScriptDefaultLib.dll` | `-lgc`, built with the **static** `gc.lib` | JIT runs; every user DLL |
| `TypeScriptDefaultLib.lib` (`lib/`) | none: an archive, the exe links Boehm | nothing (merged into the exe) |
| `TypeScriptRuntime.dll` | `BDWgc::gc` static (`GC_NOT_DLL` in `gc.cpp`); re-exports `GC_*` via `.def` | JIT code; the default-lib DLL |

The JIT resolves `GC_malloc`/`GC_add_roots`/... by name through `SearchForAddressOfSymbol`,
which finds `TypeScriptRuntime.dll`'s exports. The default-lib DLL's calls were bound at link
time to its own copy. `jit.cpp` already guards against a *second `TypeScriptRuntime` copy* for
exactly this reason, but not against the default library.

## The rule

> **One process, one Boehm.** A `gc` binary that can share a process with another `gc` binary
> must take its collector from `gc.dll`. A binary that is guaranteed to be alone may keep the
> static `gc.lib`.

"Alone" means only one case: an AOT exe that links the static default library and imports no
tslang shared library. That is the common deployment, and it keeps shipping a single file.

## Proposed design

### 1. `TypeScriptRuntime.dll` takes Boehm from `gc.dll`

Link `TypeScriptRuntime` against the shared BDWgc package (`3rdParty/gcdll/...`) and drop
`GC_NOT_DLL` on Windows in `gc.cpp`. The `.def` re-exports stay as they are; they now forward to
`gc.dll`. Install `gc.dll` into `bin/` next to `TypeScriptRuntime.dll`, so the loader finds it
for JIT runs.

Result: JIT code, the runtime's `GC_add_roots` for JIT sections, the async runtime's thread
registration, and anything else in the JIT process share one collector, as long as step 2 also
lands.

### 2. The default-library DLL takes Boehm from `gc.dll`

In the default library's build scripts, build the **`gc` DLL** with `GC_LIB_PATH` pointing at
the `gcdll` import library. Keep the static `lib/` archive unchanged; it does not link Boehm.
Only the `gc` model is affected, because `rc` and `none` have no collector.

Result: JIT (`TypeScriptRuntime.dll` + default-lib DLL) and exe + user DLL + default-lib DLL all
share `gc.dll`.

### 3. `tslang` chooses the GC flavour itself

Today `--gc-lib-path` is one directory and the user must know to swap it. Instead:

- Add `--gc-shared-lib-path` / `GC_SHARED_LIB_PATH` for the import library and `gc.dll`. The
  release package already ships it as `gcdll/`, so the default is `<gc-lib-path>/gcdll`.
- `exe.cpp` links the **shared** flavour when:
  - `--emit=dll`, or
  - `--emit=exe` and the module imports a tslang shared library. MLIRGen already knows this: it
    emits `LoadLibraryPermanentlyOp` for `import './x'`, and it reads the imported library's
    symbols at compile time. Record the fact in `CompileOptions` or a module attribute.
- Otherwise it links the static `gc.lib`, as today.
- When the shared flavour is used, copy `gc.dll` next to the output, or print exactly which file
  to ship. A missing `gc.dll` is a loader error at startup, which is loud; the bug being fixed is
  silent.

### 4. Refuse or warn on a mixed process at compile time

MLIRGen already reads a marker symbol from an imported DLL (`__tsmm_<model>_<file>_<hash>`) and
warns when the memory models differ. Add a second marker, **`__tsgc_<static|shared>_...`**,
emitted by every `gc` binary. When importing a `gc` library that linked Boehm statically, emit
an error, or at least a warning naming 5ao:

```text
error: shared library 'foo.dll' links its own garbage collector (static gc.lib).
Objects crossing between it and this module can be freed while still in use.
Rebuild it with tslang --emit=dll (which uses gc.dll).
```

A library with no marker predates this change. Warn, don't error, so existing binaries still
load.

### 5. Tests that include the default library

Add both repros to the suite **without** `--no-default-lib`:

- `test-jit-gc-defaultlib-collector`: JIT, held default-lib strings + different-content churn.
- `test-compile-gc-shared-defaultlib-collector`: exe + user DLL + default-lib DLL.

Teeth rules from the RC work apply:

- The churn must allocate content **different** from what is held; same-content churn cannot
  fail (§9.76).
- Build the expected value with the same method as the held one. A hand-built
  `"#".repeat(n) + ...` compared unequal to a correct value during this investigation.
- Confirm each test fails on today's binaries before landing the fix.

This needs a `test-runner` switch to keep the default library for named tests, plus the
per-test working directory the shared path already has (`gc.dll` and `TypeScriptDefaultLib.dll`
must sit beside the binaries **at compile time too**, because importing a DLL loads it).

### 6. Packaging

- Windows zip: `gc.dll` in the root beside `tslang.exe` and `TypeScriptRuntime.dll`, for JIT.
  Keep `gcdll/gc.lib` for linking user programs.
- The default-lib `dll/` tree is built against `gcdll` (step 2).
- `docs/memory-models.md`: a program that loads a tslang DLL ships `gc.dll` **and**
  `TypeScriptDefaultLib.dll` beside the exe.

## Alternatives considered

**A. Always `gc.dll`, drop the static `gc.lib`.** Simplest to reason about: one collector by
construction. Rejected as the default because every standalone exe, the common case, would have
to ship a second file for no benefit. It stays a one-line fallback if step 3's detection turns
out to be unreliable.

**B. DLLs import `GC_*` from their host instead of from `gc.dll`.** Windows binds an import to a
DLL *name*, and the host differs: the exe under AOT, `TypeScriptRuntime.dll` under JIT. Doing this
needs a runtime-resolved function table (`GetModuleHandle(NULL)` / `SearchForAddressOfSymbol`)
consulted on every allocation, which means new codegen, an extra indirection, and a new failure
mode if the table is missing. It buys nothing over a shared `gc.dll`.

**C. Make the collectors see each other's roots** (`GC_add_roots` across heaps). Each collector
would still own its heap and free lists. It would have to register every other heap's live
ranges dynamically as they grow, and marking through a foreign heap does not stop the owner from
sweeping. Fragile at best; rejected.

## Risks and open questions

- **CRT.** `gc.dll` is built `/MT` (static CRT), so it has its own CRT heap. Boehm allocates its
  heap with `VirtualAlloc` and does not hand CRT memory to callers, so this should be safe, but
  verify `GC_win32_free_heap` (called from `destroy_gcruntime`) and teardown under the JIT's
  unconditional `TerminateProcess` exit in `jit.cpp`.
- **Threads.** `GC_enable_threads` / `GC_allow_register_threads` must be called on the collector
  in `gc.dll`. After step 1 the runtime's export forwards there, but check the JIT stand-in in
  `jit.cpp` (`jitEnableGCThreads`) resolves to the same collector.
- **Linux.** Not measured. ELF symbol interposition may already route every module's public
  `GC_*` calls to the first-loaded copy, which would hide the bug. Boehm's hidden internals make
  that unreliable either way. Run both repros on Linux before changing anything there.
- **Debug trees.** Need the debug shared Boehm (`scripts/build_gc_debug_shared_vs.bat`) and a
  debug `dll/debug/gc` default lib built against it.
- **Performance.** An allocation becomes one indirect call through the import table. Expected to
  be negligible next to the allocation itself; confirm with `raytrace` under the AOT harness.
- **Old binaries.** Default-lib DLLs and user DLLs built before this change keep their private
  collector until rebuilt. Step 4's marker is how they get found.

## Validation plan

1. Build `TypeScriptRuntime` and the `gc` default-lib DLL against `gcdll`.
2. Re-run both repros: expect **0 / 2000** with collection on. Also re-run each with the old
   binaries to confirm they still fail, so the repro keeps its teeth.
3. Full `ctest` on Windows, release and debug; the Linux suite unchanged.
4. The new default-library tests from step 5 in CI.
5. `raytrace` memory and time, before and after, under `gc`.

## Suggested PR split

1. Runtime + default-lib DLL on `gc.dll`, plus the two regression tests (steps 1, 2, 5). This
   closes the bug for JIT, the most exposed path.
2. `tslang` picks the flavour and copies `gc.dll` (step 3), plus packaging and docs (step 6).
3. The `__tsgc_` marker and the import-time diagnostic (step 4).

## Progress

### PR 1 - runtime and default-lib DLL on `gc.dll` (Windows)

- **Step 1.** The top-level CMakeLists defines an imported `tslang_gc_shared` target from
  `3rdParty/gcdll/x64/<build>` (`TSLANG_GC_SHARED_PREFIX`, with `GC_DLL` so the headers declare the
  API `dllimport`). Configuring fails on Windows if it is missing. `TypeScriptRuntime` links it and
  copies `gc.dll` into `bin/`; `gc.cpp` no longer forces `GC_NOT_DLL` when `GC_DLL` is set.
  `prepare_3rdParty.bat` builds the shared Boehm, and the release zip ships `bin/gc.dll` in its
  root. `TypeScriptRuntime.dll` now imports `gc.dll`.
- **Step 2.** In the default library's `scripts/build_core.bat`, the DLL step links
  `--gc-lib-path=%GC_SHARED_LIB_PATH%` (default `..\TypeScriptCompiler\3rdParty\gcdll\x64\<build>\lib`;
  the release workflow sets it). `TypeScriptDefaultLib.dll` now imports `gc.dll`. Separate repo,
  same branch name.
- **Step 5.** `import_gc_single_collector.ts` / `export_gc_single_collector.ts`, registered as
  `test-jit-shared-export-import-gc-single-collector` and
  `test-compile-shared-export-import-gc-single-collector`. It needs no default library: the
  library side builds the held strings **and** churns, so the library's collector is the one that
  must run. The existing owned-returns test churned in the importer, which is why its JIT variant
  passed with two collectors.

Measured (release, before → after, each "before" failing and passing again with collection
suppressed):

| Case | Before | After |
| --- | --- | --- |
| new test, `-jit -shared` | assertion failed | 0 bad |
| new test, AOT `-shared` | 0 bad | 0 bad |
| JIT + default-lib DLL repro | 2000 / 2000 bad | 0 bad |
| exe + user DLL + default-lib DLL repro | 1984 / 2000 bad | 0 bad |

Merged as #309 (compiler) and TypeScriptCompilerDefaultLib #6.

### PR 2 - `tslang` chooses the collector's linkage (step 3, Windows)

- `CompileOptions::importsSharedLibrary`, set in `tslang.cpp` by walking the generated module for
  `LoadLibraryPermanentlyOp` before the passes lower it. Every `import` that resolves to a DLL -
  dynamic or `@static` - goes through `mlirGenImportSharedLib`, which emits that op.
- `exe.cpp`: under `-mm=gc` on Windows, `--emit=dll` or an importing `--emit=exe` links
  `-L<shared>` instead of the static directory, and after a successful link copies `gc.dll` beside
  the output (a warning names the file to ship if it cannot).
- The shared directory: `--gc-shared-lib-path`, else `GC_SHARED_LIB_PATH`, else
  `<gc-lib-path>/gcdll` (the release package), else `--gc-lib-path` itself when `gc.dll` sits
  beside it or in `../bin` (so the default library's `build_core.bat` keeps working). Nothing found
  is an **error**, not a fallback to the static `gc.lib`.
- A lone program is unchanged: static `gc.lib`, no `gc.dll` copied.
- `test-compile-gc-shared-auto` (`shared-collector-auto.cmake`) drives `tslang --emit=dll/exe` itself -
  `test-runner` links with lld directly and never exercised the compiler's choice. Five cases: the
  library gets `gc.dll`; its importer runs clean; `--gc-lib-path` at a shared build alone is
  enough; a lone exe runs with no `gc.dll` near it and gets none copied; a library with only a
  static collector fails and names `gc.dll`.
- Teeth, with the previous compiler: a library and its importer built with only
  `--gc-lib-path=<static>` link two collectors and fail the gc_single_collector assertion; with
  collection suppressed the same binary prints 0 bad.
- The VS Code template's `--emit=dll` task passes `--gc-lib-path=<package root>`, so it resolves
  `<package>/gcdll` without a change.

### PR 2, continued - step 4, step 6, template, debug default library, Linux

**Step 4 reads the library, not a marker.** The proposed `__tsgc_<static|shared>_...` marker would
be written at compile time, and the compile does not know how the binary gets linked: `test-runner`
builds its shared libraries with `--emit=obj` and links them with lld, the default library's DLL is
built with `--embed-declarations=false` (so it carries no markers at all), and libraries from before
the change have none. Instead `Dump::containsGarbageCollector` (ObjDumper.cpp) looks for the string
`GC_INITIAL_HEAP_SIZE` in the binary's data sections. Boehm's `GC_init` reads that environment
variable, so the name is present in every binary that contains the collector and in none that only
calls it. Checked against: stale debug `TypeScriptDefaultLib.dll` (static gc) and `gc.dll` - present;
release `gc`/`rc` default-lib DLLs and `TypeScriptRuntime.dll` (both flavours) - absent.

- `mlirGenImportSharedLib`: importing, under `-mm=gc` on Windows, a `gc` library that contains a
  collector is an **error** (AOT and JIT). Because the fingerprint is evidence rather than a
  missing marker, an old library gets the error too; it is broken, not merely old.
- `jit.cpp`: the same check on `TypeScriptDefaultLib.dll` before loading it - this is what a stale
  default library looks like.
- Test: `test-compile-gc-shared-auto` case 6 builds a library against the static `gc.lib` (through a
  stand-in shared directory) and expects both `--emit=exe` and `--emit=jit` importing it to fail
  naming the collector.

**Linux, measured** (WSL Ubuntu, release package v0.0-pre-alpha81, the gc_single_collector pair):

| Case | Result | No-collection control |
| --- | --- | --- |
| exe + user `.so`, as linked today | assertion failed | 0 bad |
| JIT + user `.so` | 0 bad | 0 bad |
| exe linked with `--whole-archive libgc.a` + `--export-dynamic-symbol=GC_*` | 0 bad | 0 bad |

The user `.so` does link its own `libgc.a` copy. Under the JIT it is harmless because
`libTypeScriptRuntime.so` exports `GC_*` (482 symbols) and ELF symbol interposition sends the `.so`'s
calls there. A plain exe exports none, so the `.so` ran its own collector. The default library's
`.so` links no collector (`GC_*` undefined) and already binds to the host. So Linux needs no
`gc.so` and no fingerprint check: `exe.cpp` now links an importing `--emit=exe` with the whole
collector exported. Whole archive because the `.so` may call `GC_*` functions the program does not
(without it 478 of 486 were exported, and the repro happened to pass). Not exercised on Linux beyond
the hand link: no Linux build of this branch was available locally, so the Linux CI run is its test.

**Step 6.** The zip already ships `gc.dll` in its root and `gcdll/` (PR 1). `docs/memory-models.md`
now lists what to ship beside a program that loads a tslang DLL (`gc.dll`,
`TypeScriptDefaultLib.dll`), the step 4 error, and the Linux behaviour.

**CMake template** (`docs/how/cmake_tslang`): `TSLANG_SHARED_GC` (default OFF). On Windows it links
`gcdll/gc.lib` by full path (same file name as the static one in the package root) and copies
`gc.dll` beside the target; on Linux it adds the two export options. The template still builds no
shared libraries itself - those come from `tslang --emit=dll`, which chooses on its own.

**Debug default library.** `dll/debug/gc/TypeScriptDefaultLib.dll` was still the static-gc build from
before PR 1, and the local debug `TypeScriptRuntime.dll` predated it too. Rebuilt both: the debug
compiler (reconfigured, so it picks up `TSLANG_GC_SHARED_PREFIX`) and `scripts\build_vs.bat debug gc`,
which already linked `3rdParty/gcdll/x64/debug/lib`. Both now import `gc.dll`; no script change
needed. The release workflow builds the default library with release `GC_SHARED_LIB_PATH` for every
flavour; a debug DLL imports `gc.dll` by name and binds to the one already in the process.

Still open: tests that include the default library itself (the suite still passes
`--no-default-lib`).
