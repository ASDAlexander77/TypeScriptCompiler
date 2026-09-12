# One collector per process: design proposal

Status: **PR 1 implemented** (steps 1, 2, 5 - Windows) on branch `fix-single-gc-collector`;
steps 3, 4, 6 and Linux still open. See [Progress](#progress) at the end.

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

Still open: the tests that would include the default library itself (the suite still passes
`--no-default-lib`), steps 3, 4 and 6, the debug default-lib build, and all of Linux.
