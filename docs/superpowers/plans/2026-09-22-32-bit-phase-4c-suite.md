# 32-bit Phase 4c: The Suite at i686 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** The ctest suite that runs at 64 bits also runs as 32-bit executables and is green. Every test that cannot run at 32 bits is excluded explicitly, with a recorded reason.

**Architecture:**
- `test-runner` gains a `-x86` flag. It compiles with `-mtriple=i686-pc-windows-msvc` and links with `lld -flavor link /machine:x86` against the x86 VC, SDK, UCRT, GC and runtime libraries.
- A CMake option, `TSLANG_TEST_X86` (off by default), registers an x86 twin of every non-JIT test, named `test-x86-…` and labelled `x86`. So `ctest -L x86` runs the 32-bit suite, and the default suite is unchanged.
- The `-shared` tests need the compiler to read an imported DLL's `__decls` declarations from the file, because it cannot load an x86 DLL into its x64 process.
- The remaining failures are then triaged: fixed if they are 32-bit defects, recorded otherwise.

**Tech Stack:** CMake/CTest, C++17 (`test-runner.cpp`, `llvm::object` for PE reading), lld-link, and the MSVC x86 CRT and Windows SDK.

**Spec:** `docs/superpowers/specs/2026-09-20-32-bit-compilation-design.md`, Phase 4 ("4c") and its gate: *the same suite that runs at 64 bits runs at 32 bits, green; genuine failures are triaged and recorded, not silently excluded.*

## Global Constraints

- **The default suite is unchanged.** With `TSLANG_TEST_X86` off, `ctest -j 16 -C Release` registers and passes exactly the tests it does today (2769), and the x64 `test-*` names do not change. CI is untouched.
- **JIT tests have no x86 twin.** `--emit=jit` refuses a non-host arch (phase 1), so every `-jit` registration is skipped when making twins. The skipped count is recorded in the plan's result, not hidden.
- **x86 twins use the phase 2 library layout:**
  - gc: `3rdParty/gc/x64/<cfg>/lib`, with its `x86/` subdirectory.
  - gc DLL: `3rdParty/gcdll/x64/<cfg>/lib`.
  - Runtime: `__build/tslang-runtime/<cfg>`, with its `x86/` subdirectory.
  - With the option on, a missing x86 library is a configure-time `FATAL_ERROR` that names the script to run, not a wall of failing tests.
- **Compiler behaviour at x64 does not change.** The DLL-declaration change reads from the file only when the target arch differs from the host's. `compare-ir.sh` x64 shows noise only.
- **Exclusion is explicit.** A test that cannot run at x86 is listed in one place, `tslang/test/tester/x86-exclusions.cmake`, with a one-line reason. Its twin is registered with the `DISABLED` property, so ctest reports it as "Not Run (Disabled)", not as absent.
- Hard errors, not silent fallbacks. Commits are GPG-signed; never bypass signing. The trailer is `Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>`.

## Measured starting point (main e0039a62)

- `tslang/test/tester/CMakeLists.txt` registers 1208 tests: 580 `test-compile-*`, 623 `test-jit-*` and 1 `test-ownership-*`. That is 2769 counting the extra files. The flag mixes are:
  - plain, `-jit`, `-shared`;
  - `-mm=rc` and `-mm=none` of each;
  - `-compile-time`, `-gctors-as-method`, `-noopt`, `-fast-math`.
- `test-runner.cpp` writes a cached `.bat` per flag variant (`compileBatName()` etc.). It compiles with `tslang --emit=obj --entry-point --opt --opt_level=3 --no-default-lib` and links with `%LLVMEXEPATH%\lld.exe -flavor link` against the libpaths that CMake bakes in as defines: `TEST_LIBPATH` (VC x64), `TEST_SDKPATH`, `TEST_UCRTPATH`, `TEST_GCPATH` and `TEST_TSLANG_LIBPATH`.
- Only one test uses the real default library: the collector test near `CMakeLists.txt:1942`.
- **Corpus probe at i686** (single-file exes): the only failures left at i686 alone are `internals` (`inline_asm<i64>` with `=r`) and `02funcs_vararg` (not in the suite).
- **Importing an x86 DLL** fails at compile time with `./lib.dll: Can't open: Unknown error (0xC1)`. `MLIRGenImpl::mlirGenImportSharedLib` (`lib/TypeScript/MLIRGenModule.cpp` ~983) calls `llvm::sys::DynamicLibrary::getPermanentLibrary`, then `getAddressOfSymbol(<__decls symbol>)`, which is a `const char*` pointing to the declaration text.

## File structure

| File | Change | Task |
| --- | --- | --- |
| `tslang/test/tester/test-runner.cpp` | `-x86` flag: triple, x86 libpaths, `/machine:x86`, separate cached script names. It refuses `-jit`. | 1 |
| `tslang/test/tester/CMakeLists.txt` | Adds the x86 libpath defines, the `TSLANG_TEST_X86` option and the twin registration. | 1, 2 |
| `tslang/test/tester/x86-exclusions.cmake` | New. The recorded exclusions. | 2, 4 |
| `tslang/include/TypeScript/ObjDumper.h`, `tslang/lib/TypeScript/ObjDumper.cpp` | `Dump::readExportedCString(path, symbol)` for PE files. | 3 |
| `tslang/lib/TypeScript/MLIRGenModule.cpp` (`mlirGenImportSharedLib`) | For a foreign-arch target, read `__decls` from the file and skip `getPermanentLibrary`. | 3 |
| Whatever the triage finds | Fixes for real 32-bit defects. | 4 |

---

### Task 1: `test-runner -x86`

**Files:**
- Modify: `tslang/test/tester/test-runner.cpp`, `tslang/test/tester/CMakeLists.txt`

**Interfaces:**
- Produces:
  - A `test-runner` option, `-x86`, accepted anywhere among the flags.
  - CMake defines `TEST_LIBPATH_X86`, `TEST_SDKPATH_X86`, `TEST_UCRTPATH_X86`, `TEST_GCPATH_X86`, `TEST_GC_SHARED_LIBPATH_X86` and `TEST_TSLANG_LIBPATH_X86`. Task 2 relies on these names.

- [ ] **Step 1: The x86 libpaths in CMake.** Next to the existing Windows-only discovery in `tslang/test/tester/CMakeLists.txt` (lines 1-60), derive:
  - VC: the same `VC_LIB_DIR` computation with the arch segment `x86` instead of `${CMAKE_VS_PLATFORM_TOOLSET_HOST_ARCHITECTURE}`.
  - SDK and UCRT: `…/um/x86` and `…/ucrt/x86`.
  - gc: `${TEST_GC_LIBDIR}`. The runner points `/libpath` at its `x86` subdirectory directly, because lld has no tslang lookup rule.
  - gc DLL: `TEST_GC_SHARED_PREFIX/lib`, and its `x86` subdirectory.
  - Runtime: `${PROJECT_SOURCE_DIR}/../__build/tslang-runtime/<cfg>`, and its `x86` subdirectory.

  Pass them as `target_compile_definitions(test-runner …)` beside the existing ones. Print them with `message(STATUS …)` as the file does for the x64 ones.

- [ ] **Step 2: The runner flag.** In `main`'s argument loop (~723), add `-x86` and set `auto x86 = false;` from it. When `x86` is set:
  - `tslang_opt_ext` gains ` -mtriple=i686-pc-windows-msvc`.
  - The compile, shared and multi-file script writers use the `_X86` libpaths and add `/machine:x86` to the lld line.
  - `optVariantSuffix()` gains `x86`, so an x86 script never reuses an x64 one. The scripts are cached by name; see the comment above `optVariantSuffix`.
  - `-jit` combined with `-x86` prints `test-runner: -jit cannot run x86 code in the x64 compiler process` and exits non-zero.

  Also check the non-script paths in the runner, such as the gc DLL copy and any `TEST_*` use, for x64-only assumptions, and make each one arch-aware.

- [ ] **Step 3: Try it by hand.** From the ctest working directory of an existing test (see how ctest invokes `test-runner`), run `test-runner -x86 <tests>/00try_catch.ts`, then again with `-mm=rc` and with `-mm=none`. All must pass.
  - Check that the exe the script builds is I386. Temporarily comment out its `del`, or build the same command by hand.
  - Run `test-runner -x86 -shared <a -shared test>` too. Record its failure; Task 3 fixes it.

- [ ] **Step 4: Verify** that `ctest -j 16 -C Release` is still 2769/2769. **Commit.**

---

### Task 2: The x86 test set

**Files:**
- Modify: `tslang/test/tester/CMakeLists.txt`
- Create: `tslang/test/tester/x86-exclusions.cmake`

**Interfaces:**
- Consumes: Task 1's `-x86` flag and defines.
- Produces:
  - `option(TSLANG_TEST_X86 "Also register the 32-bit (i686) twin of every non-JIT test" OFF)`.
  - Tests named `test-x86-<rest>`, where `<rest>` is the x64 name with its leading `test-` removed, labelled `x86`.
  - `x86-exclusions.cmake` defines `TSLANG_X86_EXCLUDED`, a list of x64 test names, and one `set(TSLANG_X86_EXCLUDED_REASON_<name> "…")` per entry.

- [ ] **Step 1: Registration without duplicating 1208 lines.**
  - Define a function `tslang_add_test(NAME <name> COMMAND test-runner <args…>)` that calls `add_test` unchanged, and replace every `add_test(` in this file with it. The edit is mechanical; check the count before and after.
  - When `TSLANG_TEST_X86` is ON and the args contain no `-jit`, the function also calls `add_test(NAME test-x86-… COMMAND test-runner -x86 <args…>)` and `set_tests_properties(… PROPERTIES LABELS x86)`.
  - If the x64 name is in `TSLANG_X86_EXCLUDED`, it also sets `DISABLED TRUE`. A `message(STATUS …)` prints the reason at configure time.
  - Keep any `set_tests_properties` that the file applies to x64 tests: timeouts, `RUN_SERIAL`, environment. Apply the same to the twin by making the function return the twin name, or with a small helper. Grep the file for `set_tests_properties` and handle each.
  - The one `test-ownership-*` and the default-library collector test are not `test-runner` tests. Leave them without twins, and say so in a comment.

- [ ] **Step 2: Configure-time prerequisites.** When the option is ON, check that the x86 gc lib, the gc DLL files and the runtime lib exist at the paths from Task 1. If one is missing, stop with `FATAL_ERROR` naming the phase 2 script that builds it: `prepare_3rdParty.bat <cfg> x86`, or `scripts\build_tslang_runtime_<cfg>_x86.bat`.

- [ ] **Step 3: Configure and run.**
  - `cmake --preset <the release preset> -DTSLANG_TEST_X86=ON`, then build.
  - `ctest -N` shows 2769 plus the x86 twins.
  - `ctest -N -L x86 | tail -1` gives the twin count. Record it, together with the number of `-jit` tests skipped.
  - `ctest -j 16 -C Release -L x86 --output-on-failure > x86-first-run.txt`. Record the pass count and the list of failing tests, grouped by flag mix (plain, `-shared`, `-mm=rc`, `-compile-time`, …). This list is Task 4's input.
  - Do not exclude anything yet, except `internals`: add its entry to `x86-exclusions.cmake` with the reason "`inline_asm<i64>` with an `=r` constraint: no single i686 register holds 64 bits", if it fails as the probe predicts.

- [ ] **Step 4: Verify.**
  - With the option OFF (reconfigure), `ctest -j 16 -C Release` is 2769/2769, and the names are unchanged: diff `ctest -N` before and after this task.
  - **Commit.**

---

### Task 3: Read `__decls` from the file for a foreign-arch target

**Files:**
- Modify: `tslang/include/TypeScript/ObjDumper.h`, `tslang/lib/TypeScript/ObjDumper.cpp`, `tslang/lib/TypeScript/MLIRGenModule.cpp`

**Interfaces:**
- Produces: `namespace Dump { std::optional<std::string> readExportedCString(llvm::StringRef path, llvm::StringRef symbol); }`. It reads the string that the exported pointer variable `symbol` points to in a PE DLL, without loading the DLL.

- [ ] **Step 1: A failing check.** Build a two-file `-shared` pair at x86 by hand: `lib.ts` exporting a function, with `--emit=dll`, and `main.ts` importing it, with `--emit=exe`. Today it fails with `Can't open: Unknown error (0xC1)`. Also run the `-shared` twins from Task 2's run: all of them fail with this error.

- [ ] **Step 2: `readExportedCString`.** Use `llvm::object::COFFObjectFile`:
  1. Find `symbol` in the export table: `export_directories()`, `getSymbolName`, `getExportRVA`.
  2. Read a pointer-sized value at that RVA: 4 bytes for PE32, 8 for PE32+. It is the string's virtual address at the preferred image base.
  3. Convert it to an RVA by subtracting `getImageBase()`.
  4. Read the NUL-terminated string at that RVA from the section that contains it (`getRvaPtr`).

  Any failure returns `std::nullopt`, and every `llvm::Error` is consumed. Put it beside `coffMachine` and `getSymbols`.

- [ ] **Step 3: Use it for a foreign arch.** In `mlirGenImportSharedLib`, when the target is not the host arch (`!compileOptions.targetInfo.supportsInProcessJit` is exactly that predicate), make these changes:
  - Skip `getPermanentLibrary`, because the DLL cannot be loaded.
  - Get each `__decls` text from `Dump::readExportedCString` instead of `dynLib.getAddressOfSymbol`.
  - Keep everything else the same: the `dynamic` `@dllimport` rewrite, the parsing, and the global constructor that loads the library at *run* time. That constructor runs in the x86 program, where loading works.
  - A missing symbol keeps today's warning. A symbol that is present but can't be read is an error.

  Leave the host-arch path exactly as it is.

- [ ] **Step 4: Verify.**
  - The hand-built pair builds and runs at x86 under gc.
  - Rerun the `-shared` x86 twins and record the new result.
  - `compare-ir.sh` x64 shows noise only. `ctest -j 16 -C Release`, with the option OFF, is 2769/2769, and so is the x64 `-shared` subset.
  - Add a unit test in `tslang/unittests` if one fits the existing `ObjDumper` coverage. Otherwise the `-shared` twins cover it.
  - **Commit.**

---

### Task 4: Triage to green (Phase 4 gate)

**Files:**
- Modify: `tslang/test/tester/x86-exclusions.cmake`, and wherever the fixes land.

- [ ] **Step 1: Rerun** `ctest -C Release -L x86`. Group the failures by cause, and start with the largest group. For each group, first compare the x64 twin (does it pass?) and the corpus probe result for the same file. Then find the first point of divergence, using the WinDbg recipe in memory for crashes.

- [ ] **Step 2: Fix or record.**
  - **Fix** anything that is a 32-bit defect in the compiler, the runner or the library layout: a wrong width, a wrong ABI, or a missing x86 path. Put each fix in its own commit, with a test where the twin does not already cover it.
  - **Record** anything that can never work at x86. Add it to `x86-exclusions.cmake` with a reason that names the cause, for example "`inline_asm<i64>` register constraint".
  - **Stop** on anything that fails the same way at x64, or whose cause belongs to another project (the default-lib repo, upstream LLVM). Record it with the diagnosis and report it. Do not work around it.
  - A fix that changes x64 output must show `compare-ir.sh` evidence that the change is intended, and the x64 suite must stay green.

- [ ] **Step 3: The gate.**
  - `ctest -j 16 -C Release -L x86` passes every twin that is not excluded.
  - The number of excluded twins, and each reason, is listed in the report and in the spec.
  - The default suite, with the option OFF, is 2769/2769.
  - `check-x86-run.sh`, `check-x86-eh.sh` and `check-x86-libs.sh` still pass.

- [ ] **Step 4: Update the spec.** Mark 4c as implemented, and record the twin count, the number of JIT tests skipped, the exclusions with their reasons, and how to run the suite (`-DTSLANG_TEST_X86=ON`, then `ctest -L x86`). Commit.

---

## Phase 4c gate

- [ ] `ctest -L x86` is green, and every exclusion is recorded with a reason.
- [ ] The default suite is unchanged: 2769/2769, with the same test names.
- [ ] An x86 `-shared` program builds, links and runs.
- [ ] The spec records the result.

## Rulings made while writing this plan

- **Twins, not a mode switch** (the user's choice). The option is off by default, so CI and the default suite are untouched.
- **No x86 JIT twins.** The JIT runs in the x64 compiler process; phase 1 refuses a foreign-arch JIT.
- **The default library for x86 is out of scope here.** The suite compiles with `--no-default-lib`. The one collector test that uses the real default library has no twin, because the default library is built in a separate repository (`TypeScriptCompilerDefaultLib`) that would need its own x86 build. `getDefaultLibSubDir`'s arch segment goes with that work, as a follow-up.
- **`__decls` is read from the file only for a foreign arch.** The host path, which loads the DLL, is unchanged, so x64 behaviour cannot change.
