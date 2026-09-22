# 32-bit x86 Default Library Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A 32-bit Windows program (`-mtriple=i686-pc-windows-msvc`) links and runs against the default library, for every model (gc, rc, none), debug and release, static and DLL. It no longer needs `--no-default-lib`.

**Architecture:** Two repos. The compiler (`I:\TypeScriptCompiler`) gains an arch segment in `getDefaultLibSubDir`. That gives `defaultlib/{lib,dll}/x86/{debug,release}/{gc,rc,none}/` for Windows x86, while every other target keeps today's path. The compiler also checks the COFF machine of the default library it links. The default-library repo (`I:\TypeScriptCompilerDefaultLib`) gets `TSLANG_ARCH=x86`: its build scripts produce that tree, and `tests.ps1` runs the repo's 150 tests as 32-bit exes. One x86-only bug the probe found (`string_replaceAll`) is root-caused and fixed.

**Tech Stack:** C++17 (tslang driver), Windows batch, PowerShell, CMake/CTest, bash check scripts.

**Spec:** `docs/superpowers/specs/2026-09-20-32-bit-compilation-design.md`: "Arch in the library layout", "Phase 4 — Default library and the suite", and the open issue "The x86 default library (deferred from phase 4)".

## Global Constraints

- Layout (spec, verbatim): `defaultlib/{lib,dll}/x86/{debug,release}/{gc,rc,none}/`. The x64 paths keep their shape. There is no `x64` segment.
- Strict, no fallback (spec): a missing arch tree is an error that names the arch and the script that builds it. It never falls back to another arch.
- `getDefaultLibSubDir` stays the ONE place the path is composed. Every consumer calls it and never spells the path out.
- The arch segment applies to Windows x86 only, the same rule as `getTargetLibDir` / `resolveWindowsLibPath` in `tslang/tslang/exe.cpp`. The JIT refuses x86, so `jit.cpp` always passes the host (no segment).
- x64 behaviour must be byte-for-byte unchanged. Proof: the default suite `ctest -j 16 -C Release` stays green (2771), and `tslang/test/compare-ir.sh` shows noise only.
- Commit trailer: `Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>`. GPG signing: retry, pause, `gpgconf --kill scdaemon`/`gpg-agent`, then hand the staged commit to the user with a message file. Never bypass signing.
- Branches: compiler `x86-default-lib`, from `origin/main` with `--no-track`. For DefaultLib, use `x86-default-lib` from `origin/main` if PR #7 (`debug-crt-for-native-wrappers`) has merged; otherwise branch from `debug-crt-for-native-wrappers` and say so in the PR (it edits the same `build_core.bat`).
- Use Edit/Write for any file containing backslashes (batch, PowerShell). The Bash heredoc mangles `\1` etc. Scan new files for control characters.

## Probe results (2026-09-22, compiler main b277622c)

- `lib.ts` and `lib.win32.ts` compile for i686 under gc, rc and none with no source change.
- The wrappers compile with `vcvarsall.bat x86` + `cl /MT`. `tslang --emit=dll -mtriple=i686-pc-windows-msvc --gc-lib-path=3rdParty\gcdll\x64\release\lib` links (the compiler appends `x86`). `lib.exe` archives the static lib.
- The 150 DefaultLib tests, release/gc, as x86 exes against that static lib: 149 pass. `string_replaceAll` segfaults (exit 139) after its first line; x64 passes. The failing call is `paragraph.replaceAll(/Dog/gi, "ferret")`. The string-pattern `replaceAll` works.
- Not probed: debug, rc, none, DLL linking, jit (refused, by design).

## File map

Compiler repo:
- `tslang/include/TypeScript/Defines.h`: `getDefaultLibSubDir` gains a required `arch` parameter, plus `DEFAULT_LIB_ARCH_X86`.
- `tslang/tslang/exe.cpp`: passes the arch, gives an x86-specific missing-tree error, and checks the machine of `TypeScriptDefaultLib.lib`.
- `tslang/tslang/jit.cpp`, `tslang/tslang/defaultlib.cpp`: pass `""` (host / x64).
- `tslang/test/check-x86-libs.sh`: default-lib error-path cases.
- `tslang/test/check-x86-run.sh`: default-lib run cases (skipped if no x86 default lib).
- `tslang/test/tester/CMakeLists.txt`: x86 twin of `test-compile-gc-defaultlib-collector` under `TSLANG_TEST_X86`.
- `docs/superpowers/specs/2026-09-20-32-bit-compilation-design.md`: record as built, close the open issue.
- The `string_replaceAll` fix location is decided by Task 4's root cause.

DefaultLib repo:
- `scripts/build_core.bat`: `TSLANG_ARCH` (x64 default | x86), vcvarsall arch, `-mtriple`, output and staging paths.
- `scripts/build_llvm.bat`: `-m32` for clang-cl under x86.
- `build.bat`: documents `TSLANG_ARCH`.
- `tests.ps1`: `TSLANG_ARCH=x86` runs compile mode only, as 32-bit exes.
- `README.md`: how to build and test x86.

---

### Task 1: Compiler looks for the default library in the x86 tree

**Files:**
- Modify: `tslang/include/TypeScript/Defines.h:240-265`
- Modify: `tslang/tslang/exe.cpp` (default-lib block around line 660-715; `isWindowsX86Target`/`checkWindowsBinaryMachine` exist at ~135 and ~340)
- Modify: `tslang/tslang/jit.cpp:~400`, `tslang/tslang/defaultlib.cpp:~146`
- Test: `tslang/test/check-x86-libs.sh`

**Interfaces:**
- Produces: `std::string getDefaultLibSubDir(bool shared, bool debugBuild, const char *memoryModel, const char *arch)`. `arch` is `""` for no segment or `DEFAULT_LIB_ARCH_X86` (`"x86"`). The result is `defaultlib/<lib|dll>/[x86/]<debug|release>/<model>`.
- Produces: the error text `no x86 default library built for -mm=<m>: <dir> does not exist` (Task 5 greps it).

- [ ] **Step 1: Write the failing checks.** Add these to `check-x86-libs.sh`, following its existing case style (read the script first; it builds fake lib trees in a temp dir). Compile `tslang/test/x86/await_order.ts` or any hello-world `--emit=exe -mtriple=i686-pc-windows-msvc` WITHOUT `--no-default-lib`, and pass a `--default-lib-path` pointing at:
  1. a tree that has only x64 `defaultlib/lib/release/gc/TypeScriptDefaultLib.lib` (copy from `I:\TypeScriptCompilerDefaultLib\__build\defaultlib`). Expect a nonzero exit and stderr containing `no x86 default library built for -mm=gc` and the path `defaultlib/lib/x86/release/gc`.
  2. a tree with that x64 `.lib` copied into `defaultlib/lib/x86/release/gc/`. Expect a nonzero exit and stderr containing `is built for x64` (the `checkWindowsBinaryMachine` wording; confirm it against `Dump::coffMachineName`).
  3. x64 control: the same program without `-mtriple`, against tree 1, still compiles (exit 0).

- [ ] **Step 2: Run to verify it fails.** Run `bash tslang/test/check-x86-libs.sh`. Expected: cases 1 and 2 FAIL (today the compiler takes `defaultlib/lib/release/gc` and the link fails with LNK4272/unresolved symbols, not our message). Case 3 passes.

- [ ] **Step 3: Implement.** In `Defines.h`:

```cpp
// ...and, for a Windows x86 target, per arch, outermost under lib/ or dll/ like every other x86
// library (spec "Arch in the library layout"). Every other target, x64 included, has no arch
// segment, so existing trees keep their shape. There is no fallback to the x64 tree: an x64
// library links into an x86 program only to fail on every symbol.
#define DEFAULT_LIB_ARCH_X86 "x86"

inline std::string getDefaultLibSubDir(bool shared, bool debugBuild, const char *memoryModel, const char *arch)
{
    std::string subDir = std::string(DEFAULT_LIB_DIR "/") + (shared ? DEFAULT_LIB_KIND_SHARED : DEFAULT_LIB_KIND_STATIC) + "/";
    if (arch && *arch)
    {
        subDir += std::string(arch) + "/";
    }

    return subDir + (debugBuild ? DEFAULT_LIB_BUILD_DIR_DEBUG : DEFAULT_LIB_BUILD_DIR_RELEASE) + "/" + memoryModel;
}
```

Update the layout comment above it to `defaultlib/{lib,dll}/[x86/]{debug,release}/{gc,rc,none}/`.

In `exe.cpp`, in the `!compileOptions.noDefaultLib` block, compute the arch as `win && arch == llvm::Triple::x86 ? DEFAULT_LIB_ARCH_X86 : ""` (the local `arch`/`win` variables already exist in that function; rename locally if they shadow). Pass it to `getDefaultLibSubDir`. The missing-directory error becomes arch-aware:

```cpp
llvm::errs() << "error: no " << (x86DefaultLib ? "x86 " : "") << "default library built for -mm="
             << memoryModelName(compileOptions.memoryModel) << ": " << defaultLibDir << " does not exist. "
             << (x86DefaultLib ? "Build it in TypeScriptCompilerDefaultLib with TSLANG_ARCH=x86 (see its build.bat), "
                               : "Build it (see the default-lib build scripts), ")
             << "or compile with --no-default-lib.\n";
```

After the existence check and before the collector check, add the machine check for Windows x86/x64 (`winX86OrX64` exists in the function):

```cpp
// Refused here for the reason resolveWindowsLibPath refuses the other libraries: an x64 default
// library in an x86 program (or the reverse) is a linker warning and then an unresolved symbol
// for every call into it, which names neither the library nor the fix.
if (winX86OrX64 && !defaultLibDir.empty())
{
    llvm::SmallString<256> defaultLibFile(defaultLibDir);
    llvm::sys::path::append(defaultLibFile, DEFAULT_LIB_NAME ".lib");
    if (llvm::sys::fs::exists(defaultLibFile) && !checkWindowsBinaryMachine(TheTriple, defaultLibFile))
    {
        return 1;
    }
}
```

(`dll/` trees hold the DLL's import library `TypeScriptDefaultLib.lib` next to the DLL. Verify with `ls`. If it is absent there, check the `.dll` instead.)

In `jit.cpp` and `defaultlib.cpp`, add the argument `/*arch=*/""` with a one-line comment: JIT is host-only / install checks the host build.

- [ ] **Step 4: Run to verify it passes.** Build the compiler (the Release build dir the suite uses; see memory `build-and-test-invocation-gotchas`). Then run `bash tslang/test/check-x86-libs.sh`: all ok, including the existing cases.

- [ ] **Step 5: x64 unchanged.** Run `ctest -j 16 -C Release` in the build dir (2771/2771) and `bash tslang/test/compare-ir.sh` (noise only). Also compile one DefaultLib test at x64 WITH the default lib (`--default-lib-path=I:\TypeScriptCompilerDefaultLib\__build`) and run it.

- [ ] **Step 6: Commit.** Message: "Look for the Windows x86 default library in its own arch tree".

---

### Task 2: DefaultLib builds the x86 tree

**Files:** (DefaultLib repo)
- Modify: `scripts/build_core.bat`, `scripts/build_llvm.bat`, `build.bat`, `README.md`

**Interfaces:**
- Consumes: the Task 1 layout.
- Produces: `set TSLANG_ARCH=x86 && build.bat [release|debug] [model]` stages `__build\defaultlib\{lib,dll}\x86\<mode>\<model>\` (`TypeScriptDefaultLib.lib`, and under `dll\` also `TypeScriptDefaultLib.dll` + its import `.lib`). With `TSLANG_ARCH` unset or `x64`, the output is identical to today.

- [ ] **Step 1: Implement `build_core.bat`.** Read the whole file first. Changes:
  - `set ARCH=x64` becomes `if "%TSLANG_ARCH%"=="" set TSLANG_ARCH=x64` then `set ARCH=%TSLANG_ARCH%`. Reject anything but `x64`/`x86` with the script's XXX banner and `exit /b 1`.
  - `set ARCH_DIR=` for x64 and `set ARCH_DIR=\x86` for x86. Also `set TRIPLE_OPT=` for x64 and `set TRIPLE_OPT=-mtriple=i686-pc-windows-msvc` for x86.
  - Every output path `lib\%BUILD%\%MM%` / `dll\%BUILD%\%MM%`, and the `%BUILD_LIB_PATH%\{dll,lib}\...` staging paths, become `lib%ARCH_DIR%\%BUILD%\%MM%` etc. Do this as ONE variable per tree to avoid missing one: `set LIB_OUT=lib%ARCH_DIR%\%BUILD%\%MM%` and `set DLL_OUT=dll%ARCH_DIR%\%BUILD%\%MM%`, then use those everywhere (rd, md, /Fo, -o, --obj, del, xcopy, the final exist check).
  - Every `tslang.exe` invocation gets `%TRIPLE_OPT%`.
  - The vcvars line: `vcvars64.bat` becomes `vcvarsall.bat` with the argument `x64` or `x86` (both live in `VC\Auxiliary\Build`). Keep the existing quoting pattern.
  - The `GC_SHARED_LIB_PATH` default stays `..\TypeScriptCompiler\3rdParty\gcdll\x64\%BUILD%\lib`: the compiler appends `x86` itself. Add a comment saying so.
  - `COMPILER_VERSION.txt`: append `arch: %ARCH%`.
- [ ] **Step 2: `build_llvm.bat`.** clang-cl targets x64 regardless of vcvars. For x86, `TSLANG_CC` must be `clang-cl -m32`. Set it in `build_core.bat` after the arch is known (`if "%TSLANG_TOOLCHAIN%"=="llvm" if "%ARCH%"=="x86" set "TSLANG_CC=%TSLANG_CC% -m32"`), and make the `where %TSLANG_CC%` check use the bare tool name (set a `TSLANG_CC_TOOL` before appending). `llvm-lib` infers the machine from its inputs.
- [ ] **Step 3: `build.bat`** header comment documents `set TSLANG_ARCH=x86`. It is not a positional argument: the positional pair `mode model` stays as is, in the same idiom as `TSLANG_TOOLCHAIN`. `README.md` gets the x86 build line plus the prerequisite (x86 Boehm: `prepare_3rdParty.bat <cfg> x86` in the compiler repo).
- [ ] **Step 4: Build and inspect.** Run `set TSLANG_ARCH=x86 && build.bat` via `cmd //c` from the repo root (all 6 combos). Then run `set TSLANG_ARCH=x64 && build.bat release gc`, and `build.bat release gc` with the variable unset. Verify:
  - every `__build\defaultlib\{lib,dll}\x86\{debug,release}\{gc,rc,none}\TypeScriptDefaultLib.lib` is machine `0x14c` (use `dumpbin /headers` or `llvm-objdump --file-headers`, whichever is on PATH);
  - `dll\x86\...\TypeScriptDefaultLib.dll` is PE machine `0x14c`, and under gc imports `gc.dll` (`dumpbin /imports`);
  - the x64 tree is still machine `0x8664`, with the same file list as before (`git status` in `__build` is irrelevant because it's untracked; compare `ls -R` before/after).
- [ ] **Step 5: Commit** in DefaultLib. Message: "Build the default library for 32-bit Windows with TSLANG_ARCH=x86".

---

### Task 3: DefaultLib tests run as 32-bit exes

**Files:** (DefaultLib repo)
- Modify: `tests.ps1`, `README.md`

**Interfaces:**
- Consumes: the Task 2 tree and the Task 1 compiler.
- Produces: `$Env:TSLANG_ARCH="x86"; .\tests.ps1` runs release/compile and debug/compile for 32 bits, and skips jit with a printed note ("jit is host-only; skipped for x86").

- [ ] **Step 1: Implement.** In `Test`, when `$Env:TSLANG_ARCH -eq "x86"`:
  - add `-mtriple=i686-pc-windows-msvc` to the compile arguments;
  - drop `--shared-libs=...TypeScriptRuntime.dll` (an x64 JIT DLL; exe linking doesn't need it — verify by reading `exe.cpp`'s use of shared libs, and keep it for x64);
  - point `--gc-lib-path` at `..\TypeScriptCompiler\3rdParty\gc\x64\<build>\lib` and `--tslang-lib-path` at `..\TypeScriptCompiler\__build\tslang-runtime\<build>`. The compiler appends `x86` to both. Pass them as explicit flags, not via the `$Env:` defaults, so an x64 value already in the environment can't leak in.

  The run step is unchanged: the x86 exe runs under WOW64. At the bottom, x86 runs only the two compile passes.
- [ ] **Step 2: Run** `$Env:TSLANG_ARCH="x86"; powershell -File tests.ps1` (PowerShell tool). Expected: 149/150 per config, with `string_replaceAll` failing (the probe). Record the exact debug/compile results; debug was not probed. Any failure besides `string_replaceAll` gets root-caused in Task 4 (same procedure), not papered over.
- [ ] **Step 3: x64 unchanged.** Run `tests.ps1` with `TSLANG_ARCH` unset: same results as before the change (run it once on the unchanged script first to have the baseline).
- [ ] **Step 4: Commit.** Message: "Run the default-library tests as 32-bit programs with TSLANG_ARCH=x86".

---

### Task 4: Root-cause and fix `string_replaceAll` at i686

**REQUIRED SUB-SKILL:** superpowers:systematic-debugging. No fix before the root cause is stated with evidence.

**Files:** determined by the root cause. The regression test goes in the compiler repo's suite if the bug is in the compiler, so every x86 twin covers it. If the bug is in `lib.ts`/wrappers, the DefaultLib test already covers it.

- [ ] **Step 1: Reproduce minimal.** Start from `tests/string_replaceAll.ts`. Cut it to the smallest program that still crashes at x86 and passes at x64, e.g. just `console.log("I think Ruth's dog".replaceAll(/Dog/gi, "ferret"))`. Try release and debug, and gc/rc/none. Build commands as in the probe:

```
tslang --opt --opt_level=3 --nowarn -mm=gc -mtriple=i686-pc-windows-msvc --emit=exe --default-lib-path=I:\TypeScriptCompilerDefaultLib\__build --gc-lib-path=I:\TypeScriptCompiler\3rdParty\gc\x64\release\lib --tslang-lib-path=I:\TypeScriptCompiler\__build\tslang-runtime\release t.ts
```

- [ ] **Step 2: Locate.** Read `replaceAll`'s regex path in `src/lib.ts` and the regex wrapper (`src/wrappers/regex.cpp`). Look first for width assumptions: `index`/`i64`/`long` parameters crossing into C++ `size_t`/`int`, and struct layouts shared between TS and C++ (e.g. match-result arrays), which differ at 32 bits. If reading doesn't settle it, get a crash dump (memory `windbg-available-for-debugging`: `procdump -ma -e`, DbgX.Shell with a `$$<` script, `!wow64exts.sw` for the 32-bit stack).
- [ ] **Step 3: Failing test first.** If the bug is in the compiler, add a minimal compiler-suite test (`tslang/test/tester/tests/`, registered like its neighbours; x86 twin via `tslang_add_test`) that does NOT need the default library, i.e. reproduces the width bug in pure TS. If it is in the default library, the DefaultLib test is the failing test.
- [ ] **Step 4: Fix, then verify:**
  - the x86 DefaultLib tests are 150/150 in both configs;
  - x64 `tests.ps1` is unchanged;
  - if the compiler changed: `ctest -C Release` is green, and the x86 twins are green with `-DTSLANG_TEST_X86=ON` (`ctest -C Release -L x86`; re-configure OFF afterwards).
- [ ] **Step 5: Commit** in whichever repo holds the fix, with a message naming the root cause.

---

### Task 5: Compiler-side run coverage and docs

**Files:**
- Modify: `tslang/test/check-x86-run.sh`, `tslang/test/tester/CMakeLists.txt` (~line 2089-2120), `docs/superpowers/specs/2026-09-20-32-bit-compilation-design.md`

- [ ] **Step 1: `check-x86-run.sh`.** Add a default-lib section. It finds an x86 default library the way `defaultlib-collector.cmake` finds one (the `DEFAULT_LIB_CANDIDATES` pair `../TypeScriptCompilerDefaultLib/__build`), and SKIPS with a message if `defaultlib/lib/x86` is absent. It builds and runs as 32-bit exes, checking output:
  - one program using Array/Map/string/Date/RegExp, including the `replaceAll(regex, …)` from Task 4, under gc, rc and none (release);
  - the gc one again with `--di` (debug tree);
  - a gc `--emit=dll` library that uses the default library, plus an exe that imports it. Copy the shape from the existing shared-library cases in the script, and copy `TypeScriptDefaultLib.dll` and x86 `gc.dll` next to the exe.
- [ ] **Step 2: x86 collector twin.** In `CMakeLists.txt`, under `if(TSLANG_TEST_X86)`, register `test-x86-compile-gc-defaultlib-collector`, labelled `x86`. It uses the same script with an x86 flag, read `defaultlib-collector.cmake` first. It passes `-mtriple=i686-pc-windows-msvc` through `OPT`, and the gc lib path whose `x86` subdir exists, and it keeps `SKIP_REGULAR_EXPRESSION "SKIPPED:"`. Extend the script's skip test so that when the x86 flag is on it checks the `x86` tree. With `TSLANG_TEST_X86=OFF`, `CTestTestfile.cmake` must be identical to before.
- [ ] **Step 3: Run.** `bash tslang/test/check-x86-run.sh` (all ok, default-lib section ran, not skipped). Then `cmake -DTSLANG_TEST_X86=ON .` + `ctest -C Release -L x86`: 1395 pass + the collector twin passes, 3 disabled. Then re-configure OFF, and run `ctest -j 16 -C Release`: 2771/2771.
- [ ] **Step 4: Spec.**
  - Status line: phases 1-4 implemented, no exception.
  - Phase 4 gets an "As built" note for the default library: the layout, `TSLANG_ARCH`, the machine check, test counts, and the Task 4 root cause.
  - Close the open issue "The x86 default library", and update the 4c bullet's "(the default library has no x86 build)" parenthetical.
  - Note what is still not covered: CI and the release workflows build x64 only, and the x86 `test-runner` twins still pass `--no-default-lib` like x64.
- [ ] **Step 5: Commit.** Message: "Cover the x86 default library in the x86 checks and record it in the spec".

---

## Finish

Two PRs, compiler first: its layout is what the DefaultLib output must match, and the DefaultLib tests need the Task 1 compiler. Each description states the pairing and the counts. Neither CI builds x86, so each PR's test plan lists what was run locally.
