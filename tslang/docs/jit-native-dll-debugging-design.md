# Debugging `--emit=jit` in WinDbg and Visual Studio: design proposal

Status: **implemented** on branch `jit-native-dll` (2026-10-20) as `--emit=jit --jit-native`. The
approach was first proven by hand on a two-module program; see
[Feasibility](#feasibility-2026-10-20). Where the implementation differs from the plan below, see
[As built](#as-built).

## Problem

On Linux a JIT run can be debugged in gdb or lldb. On Windows, WinDbg and Visual Studio see
nothing of the program: no function names, no source lines, no locals.

The JIT's only debugger hook is the GDB JIT interface (`createGDBRegistrationListener`,
jit.cpp `createJit`, and only without `--opt`). gdb and lldb read `__jit_debug_descriptor`.
WinDbg and Visual Studio do not. They learn about code from a module load event and a PDB, and
RTDyld code has neither: it sits in plain `VirtualAlloc` memory.

The objects are no help either. On windows-msvc the module carries `!"CodeView", i32 1`, so the
objects hold CodeView. No debugger reads CodeView out of a JIT registration. `--lldb`, which
switches to DWARF, is applied only in `dumpLLVMIR` and never on the JIT path.

Stack walking already works (`k` goes through JIT frames), because `JitSectionMemoryManager`
registers `.pdata` with `RtlAddFunctionTable`. What's missing is symbols and source.

## Proposal

When asked to, the JIT on Windows links the objects the JIT cache already keeps in `__jit` into
**one DLL with a PDB** and loads it with `LoadLibrary` instead of RTDyld. Then:

- the debugger gets a real module load event and finds the PDB beside the DLL. Breakpoints by
  name or by line, source stepping, locals and stacks work in WinDbg, cdb and Visual Studio;
- the loader registers `.pdata`, and `RtlPcToFileHeader` finds the image. That makes the
  `_CxxThrowException` shim, the RVA-0 image-base workaround and the read-only-as-code mapping
  unnecessary in this mode;
- the program runs exactly as it would linked AOT as `--emit=dll`. It uses the same link line
  and the same runtime libraries.

### One DLL, not one per unit

All units go into a single image, the way AOT links a program and its imported modules into one
exe. A DLL per unit would need every cross-module **data** reference to go through `__imp_`
indirection: globals, vtables, RTTI, a class's `.size`. Codegen emits no `dllimport` for those.
One image resolves them at link time, as AOT does.

### Trigger

The first version uses an explicit option, e.g. `--jit-native` (name to be decided):
`--emit=jit` + Windows + the option → DLL mode. It implies `--di` unless `--opt` is given.

Possible follow-up: switch on automatically when `IsDebuggerPresent()` and `--di`. That doesn't
cover attaching a debugger later, so the explicit option stays.

## Work items

1. **A reentrant link function.** `buildExe` (exe.cpp) runs `llvm::InitLLVM` and reads the
   globals `emitAction` and `outputFilename`. Split the link-line builder out of it so it takes
   the object list, the output path, "shared" and extra linker arguments as parameters.
   `buildExe` keeps its behaviour and calls the new function. The JIT calls it mid-run without a
   second `InitLLVM`.
2. **Exports.** `-Wl,/EXPORT:` (or a generated `.def`) for `__tslang_jit_main`, every unit's
   `__tslang_jit_init_*`, `__mlir_gctors` when the program has it, and the hook from item 3.
3. **A host hook in the image.** Add one more function to the entry-thunk module, e.g.
   `__tslang_jit_host_hook(int what)`. It is compiled into the DLL, so it reaches **the DLL's**
   CRT:
   - *flush*: `fflush(NULL)` before `runJitProgram`'s `TerminateProcess`. The DLL links the
     static CRT, so it owns the program's stdout buffer. The existing flush reaches only
     ucrtbase, and only under `_DEBUG`. Without this hook the output is **lost**. This was
     measured: the hand-built DLL printed only when it was unloaded.
   - *assert mode*: `_CrtSetReportMode` / `_set_error_mode`, so a failing `assert` in a debug-CRT
     DLL doesn't open a modal box. This is the same hole PR #276 closed for AOT; in the JIT today
     `_assert` is bound to `jitAssertFailed`.
4. **The run path.** In `runJitCached`: after `buildJitUnits`, collect the units' object paths
   and link them. Then `LoadLibraryExW` with `LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR |
   LOAD_LIBRARY_SEARCH_DEFAULT_DIRS`, plus `AddDllDirectory` for the default library's folder:
   the DLL imports `TypeScriptDefaultLib.dll`, and under gc it imports `gc.dll` (the link step
   already copies `gc.dll` beside the output). Then call the same sequence `runJitProgram`
   calls (init functions in order, `__mlir_gctors`, the entry thunk) through `GetProcAddress`.
   `runJitProgram` is factored so the two loaders share argv building, `callEntryThunk` and the
   teardown.
5. **Caching the DLL.** Name it from a hash of the unit object hashes plus the link options,
   e.g. `__jit/<main>.<hash>.dll` with its `.pdb`. Reuse it when it already exists. Link to a
   temporary name and rename, as `writeFileAtomically` does. A loaded DLL can't be replaced,
   which is fine: a new hash is a new name. Link with `/INCREMENTAL:NO`, or link.exe writes a
   14 MB `.ilk` every time.
6. **What isn't loaded.** In DLL mode the image is self-contained, like `--emit=dll`: it links
   `TypeScriptAsyncRuntime.lib` itself. So `loadJitSharedLibraries` skips
   `TypeScriptRuntime.dll` and the default library. Otherwise the process runs two async
   runtimes, and under gc possibly a second collector. `--shared-libs` still load: a program
   resolves those at run time through `tslang_load_library_permanently`.
7. **Paths that can't do it.** `--jit-cache=false` and stdin (`-`) compile in memory and have no
   objects on disk. Report an error ("--jit-native needs the JIT cache"). Don't write temporary
   objects. Off Windows the option is ignored, with a warning: gdb and lldb already work there.
8. **Tests.** Two of them:
   - a cmake `-P` script in the style of `jit-throw-without-runtime.cmake`, running
     `tslang --emit=jit --jit-native` on a two-module program. It checks the output and its
     order, the exit code, an uncaught throw, and a second run that reuses the DLL. test-runner
     can't cover this: it always adds `--shared-libs=TypeScriptRuntime.dll` and never sees
     tslang's own link line.
   - a cdb check where cdb is installed: `bp <dll>!<method>`, `kn`, `lsa .`. Skip otherwise.

## Behaviour that changes in this mode (documented, not fixed)

- **A linker is needed**: link.exe from Visual Studio, or lld-link. The in-memory JIT needs none.
- **Symbols resolve at link time.** A missing symbol is a link error, not a JIT materialization
  error. Today `--shared-libs` also feed symbol resolution, through DLL export tables; in this
  mode the AOT rules apply.
- **The CRT follows `--opt`, not the tslang build.** With `--opt` it links `libcmt`, without it
  `libcmtd`, as `buildExe` does today. A release tslang with `--di` gets the debug CRT.
- **Size and time.** Measured on the probe: a 0.9 MB DLL and 5.5 MB PDB, linked in 138 ms. A
  rebuilt unit costs one link; an unchanged run costs none.

## Feasibility (2026-10-20)

Done by hand with the 2026-10-20 release build (`windows-msbuild-2026-release`) under `-mm=rc`.
`m.ts` exports a class with a method, an exported `let`, and has a top-level `print`. `p.ts`
imports it, throws and catches a string, and prints.

1. `tslang --emit=jit --di` put `m.ts.<hash>.obj` (init `__tslang_jit_init_65a6b283138c1d5b`)
   and `p.ts.<hash>.obj` in `__jit`.
2. The `--emit=dll --verbose` link line (link.exe), with those two objects instead of its own,
   plus `-incremental:no -export:__tslang_jit_main -export:__tslang_jit_init_…`, **linked
   cleanly in 138 ms**. The JIT-flavoured main object links as a DLL, and the weakened COMDAT
   duplicates cause no conflicts.
3. A host process that called `LoadLibraryW` → init → `__tslang_jit_main(1, argv)` printed
   `m init`, `caught`, `42` with exit code 0. The C++ exception was caught with no shim. The
   output appeared only when the DLL was unloaded (see item 3).
4. Under cdb, `bp prog!Counter.inc` stopped in the **imported** module. `kn` showed
   `prog!Counter.inc [m.ts @ 3]` ← `prog!main [p.ts @ 4]` ← `prog!_tslang_jit_main`, and `lsa .`
   listed `m.ts` with the current line marked.

Locals showed as `<value unavailable>` at the method's first instruction. That is debug-info
quality for TS code, which AOT shares, not something this mode adds.

## As built

- **The link** is `linkBinary` (exe.cpp, declared in link.h): the link line `buildExe` built,
  taking its objects, output, "shared", extra arguments and a quiet flag as parameters. `buildExe`
  keeps `InitLLVM` and the command-line globals and calls it. With `quiet`, the linker's output
  goes to a temporary file and is shown only if the link fails.
- **The host functions** (`__tslang_jit_native_init`, `__tslang_jit_native_flush`) are in an object
  of their own, generated at link time (jitnative.cpp), not in the entry-thunk module. The cached
  objects are also the ones the in-memory JIT loads, and a call to the debug C runtime's
  `_CrtSetReportMode` would leave the in-memory JIT an unresolved symbol. Init calls
  `_set_error_mode(_OUT_TO_STDERR)` and `_set_abort_behavior(0, _WRITE_ABORT_MSG |
  _CALL_REPORTFAULT)`, sets a SIGABRT handler that flushes (a failing assert ends in `abort()`,
  which flushes nothing), and with the debug C runtime sends `_CRT_ASSERT` and `_CRT_ERROR`
  reports to stderr.
- **The static default library**, not the DLL's import library (`LinkRequest::staticDefaultLib`).
  With the import library the link fails as soon as the program reads data the default library
  defines. A `Map` reads `PrimeHelpers.hashCollisionThreshold` by its own name, and only an
  `__imp_` reference resolves a DLL's data export. The in-memory JIT binds it from the export
  table. `--emit=dll` has the same failure today (a separate bug). So in this mode tslang loads
  neither the default library's DLL nor `TypeScriptRuntime.dll`.
- **The DLL and its PDB** are `__jit/<file>.<hash>.dll` and `.pdb`, linked under a temporary name,
  `/PDB:` the final name with `/PDBALTPATH:%_PDB%`, then renamed. `.lib` and `.exp` are removed.
  The DLLs of earlier runs of the same program are removed, and so are temporary files more than an
  hour old, which a run killed mid-link leaves behind.
- **In WinDbg** the module is named after the file with its dots made underscores:
  `p_ts_<hash>`. Break by function (`bm *!Counter.inc`, `bu <module>!name`) or by source line
  (`` bu `m.ts:6` ``). A source line that maps to more than one address, such as a module's line 1
  when its initializer is there too, is an "ambiguous symbol" error.
- **Assert output** is the C runtime's (`Assertion failed: <message>, file <f>, line <n>`), not
  the in-memory JIT's `<f>:<n>: assertion failed: <message>`. The exit code (3) is the same.
- **Test**: `test-jit-native-dll` (test/tester/jit-native.cmake), plus the cdb step where cdb is
  installed. It covers rc only. gc needs gc.dll's import library (`GC_SHARED_LIB_PATH`) and a
  runtime library built with the C runtime the configuration links. It works by hand with
  `--opt`, and with `--di` against a debug runtime library.

## Estimate

About **3–4 days**: item 1 is a careful refactor of a 600-line function, items 2–7 are plumbing
in jit.cpp and jitcache.cpp, and item 8 is the cmake test plus a cdb probe. The risk is in
item 1 (the link line has many target branches) and item 6 (making sure nothing else loads the
runtime DLL).
