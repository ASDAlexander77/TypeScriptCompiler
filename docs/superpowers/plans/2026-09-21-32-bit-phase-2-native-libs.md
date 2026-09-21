# 32-bit Phase 2: Target Layout and Native x86 Libraries — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make lowering size pointers and structs by the target's real data layout, and build the x86 libraries a compiled program links, so that `-mm=gc` and `-mm=rc` programs link and run as 32-bit Windows executables.

**Architecture:** Lowering's `LLVMTypeConverter` reads the data layout that `MLIRGenModule` already puts on the module, instead of LLVM's default. Boehm and `TypeScriptAsyncRuntime` get x86 builds. The async runtime has no real LLVM/MLIR dependency, so a small standalone CMake project builds it for any arch. The compiler looks for x86 libraries in an `x86` subdirectory of each library path, and checks each library's COFF machine type before linking.

**Tech Stack:** C++17, MLIR/LLVM (prebuilt x64 in `3rdParty/llvm/x64`), CMake + MSVC (Visual Studio 18 2026, `-A Win32` for x86), Boehm GC 8.2.12, GoogleTest, bash test scripts.

**Spec:** `docs/superpowers/specs/2026-09-20-32-bit-compilation-design.md` — Phase 2 section, plus its "Open issues".

## Global Constraints

- Proof target is `i686-pc-windows-msvc`. The compiler itself stays x64 and cross-compiles; never build `tslang.exe` as 32-bit.
- Arch naming is `x86` / `x64`. x86 libraries live in an `x86` subdirectory of the given library path. x64 paths keep their existing flat layout, unchanged.
- A missing x86 library, or a library of the wrong machine type, is a hard error that names the arch and how to build it. Never fall back to another arch.
- Anything linked into a compiled program uses the static CRT: `/MT` for Release, `/MTd` for Debug.
- Task 1 is the **only** task allowed to change x64 output, and only where the type converter's layout feeds it. Every x64 difference it causes must be explained. All other tasks change no x64 output.
- `--emit=jit` refuses x86 (Phase 1), so `TypeScriptRuntime.dll` needs no x86 build.
- No exceptions (Phase 3), default library (Phase 4) or async (Phase 4) at x86 in this phase. Test programs use `--no-default-lib` and no `try`/`throw`/`async`.
- Run the suite with `ctest -j 16 -C Release`. Without `-C Release`, every test shows "(Not Run)".

## Rulings made while writing this plan

- **x86 library location: an `x86` subdirectory of the existing path** (`<GC_LIB_PATH>/x86/gc.lib`), not new flags or environment variables. Existing scripts, packages and CI keep working unchanged, and the spec's "arch is the outermost dimension under each component" is honored.
- **x86 builds are opt-in.** `prepare_3rdParty.bat <build> x86` builds them; the default does not. CI and everyone's setup stay the same speed.
- **The pointer-width agreement guard (spec open issue) is folded in as Task 2.** It is the other half of "width sources agree".
- **Building a DLL at x86 is measured, not gated.** The spec's Phase 2 gate is `-mm=gc` and `-mm=rc` executables; x86 symbol decoration for DLL exports is untested ground, so Task 6 attempts it and records the result without blocking on it.

## Baseline (local only)

`.superpowers/sdd/2026-09-21-32-bit-phase-2-native-libs/baseline/` is git-ignored scratch. It holds:

- masked `--emit=llvm -mm=gc --no-default-lib` output of all 519 files in `tslang/test/tester/tests`, for x64 (`x64/`) and i686 (`i686/`), from the compiler at `main` 6eac25f4;
- the unmasked output (`x64-raw/`, `i686-raw/`);
- `../mask.sh`.

Rebuild it on another machine by running the pre-change release `tslang.exe` over the corpus and masking with `mask.sh`.

**Emitted IR is not reproducible run to run.** 289 of 519 files differ between two runs of the same binary. The cause is hash-like IDs inside names and constants. `mask.sh` removes that noise completely except in one file, `typeGuardOfFormTypeOfBoolean`. That file differs for a real reason: when two union members are the same size, `findMaxSizeType` picks the storage type in hash order. A file hit by that is noisy by nature; compare it across two runs before calling a difference real.

The baseline is captured at 6eac25f4: 4 x64 files and 6 i686 files fail to compile, and 288 of 513 i686 modules contain `ptrtoint ptr … to i64`.

## File Structure

| File | Responsibility |
| --- | --- |
| `tslang/lib/TypeScript/LowerToLLVM.cpp` (modify) | Type converter takes the module's data layout; wasm32 hardcode deleted. |
| `tslang/lib/TypeScript/MLIRGenModule.cpp` (modify) | Refuses a triple whose layout pointer width disagrees with its arch. |
| `tslang/test/check-datalayout.sh` (modify) | Adds the wasm32 `i128:128` and gnux32 checks. |
| `tslang/test/compare-ir.sh` (create) | Masked corpus IR diff between two compilers or against a stored baseline. |
| `scripts/build_gc_{debug,release}_vs_x86.bat`, `scripts/build_gc_{debug,release}_shared_vs_x86.bat` (create) | Boehm static and DLL for Win32. |
| `prepare_3rdParty.bat` (modify) | Optional second argument `x86`. |
| `tslang/runtime/CMakeLists.txt` (create) | Standalone build of `TypeScriptAsyncRuntime` for any arch, no MLIR/LLVM link. |
| `scripts/build_tslang_runtime_{debug,release}_x86.bat` (create) | Configures and installs the x86 runtime. |
| `tslang/lib/TypeScript/ObjDumper.cpp`, `tslang/include/TypeScript/ObjDumper.h` (modify) | `Dump::coffMachine(path)` for objects, archives and import libraries. |
| `tslang/tslang/exe.cpp` (modify) | Per-target-arch library path resolution and machine-type check. |
| `tslang/test/check-x86-libs.sh`, `tslang/test/check-x86-run.sh` (create) | Error-path and end-to-end x86 checks. |

## Build and test commands

```bash
# compiler (Release) and unit tests (Debug), from i:/TypeScriptCompiler/tslang
cmake --build --preset build-windows-msbuild-2026-release
cmake --build --preset build-windows-msbuild-2026-debug --target MLIRTypeScriptUnitTests
# suite
cd i:/TypeScriptCompiler/__build/tslang/windows-msbuild-2026-release && ctest -j 16 -C Release
# unit tests
i:/TypeScriptCompiler/__build/tslang/windows-msbuild-2026-debug/unittests/MLIRGen/Debug/MLIRGenTests.exe
```

`TSLANG=i:/TypeScriptCompiler/__build/tslang/windows-msbuild-2026-release/bin/tslang.exe` below.
`READOBJ=i:/TypeScriptCompiler/3rdParty/llvm/x64/release/bin/llvm-readobj.exe`.

---

### Task 1: Lowering uses the target's data layout

**Files:**
- Modify: `tslang/lib/TypeScript/LowerToLLVM.cpp` — `TypeScriptToLLVMLoweringPass::runOnOperation`, currently ~7255-7268
- Modify: `tslang/test/check-datalayout.sh`
- Create: `tslang/test/compare-ir.sh`

**Interfaces:**
- Consumes: the module attribute `llvm.data_layout` (`mlir::LLVM::LLVMDialect::getDataLayoutAttrName()`). `MLIRGenModule` sets it from the `TargetMachine` for every compilation that has a triple, which `prepareOptions()` always supplies.
- Produces: `test/compare-ir.sh <tslang.exe> <triple> <baseline-dir> <out-dir>`, used again in Tasks 2 and 5.

Today `options.dataLayout` is LLVM's default (8-byte pointers, `i64` aligned at 4 bytes) for every target except wasm32, which gets a hardcoded string that is itself wrong (`f128:64`, missing `i128:128`). The type converter's layout drives `getIntPtrType`, `getTypeAllocSizeInBytes`, `getStructLayout`, union storage selection (`findMaxSizeType`) and debug-info field offsets. Allocation sizes are **not** affected: `SizeOfOpLowering` uses the `getelementptr null, 1` idiom, which LLVM resolves with the real layout at codegen.

- [ ] **Step 1: Write the comparison script**

Create `tslang/test/compare-ir.sh`:

```bash
#!/usr/bin/env bash
# Compiles every test in tslang/test/tester/tests to LLVM IR with the given compiler and triple,
# masks tslang's run-to-run naming noise, and diffs each file against a stored masked baseline.
# Prints one line per file that differs or that newly fails/succeeds, then a summary.
#   compare-ir.sh <tslang.exe> <triple> <baseline-dir> <out-dir>
set -u
TSLANG="${1:?tslang.exe}"; TRIPLE="${2:?triple}"; BASE="${3:?baseline dir}"; OUT="${4:?out dir}"
HERE="$(cd "$(dirname "$0")" && pwd)"
TESTS="$HERE/tester/tests"
MASK="${MASK:-$(cd "$BASE/.." && pwd)/mask.sh}"
mkdir -p "$OUT"
same=0; differ=0; newfail=0; newpass=0
for f in "$TESTS"/*.ts; do
    b="$(basename "$f" .ts)"
    if "$TSLANG" --emit=llvm -mm=gc --no-default-lib -mtriple="$TRIPLE" "$f" -o "$OUT/$b.raw.ll" >/dev/null 2>&1; then
        "$MASK" "$OUT/$b.raw.ll" > "$OUT/$b.ll"
        if [ ! -f "$BASE/$b.ll" ]; then echo "NEWPASS $b"; newpass=$((newpass+1))
        elif diff -q "$BASE/$b.ll" "$OUT/$b.ll" >/dev/null; then same=$((same+1))
        else echo "DIFF    $b"; differ=$((differ+1)); fi
    else
        if [ -f "$BASE/$b.ll" ]; then echo "NEWFAIL $b"; newfail=$((newfail+1)); fi
    fi
done
echo "same=$same differ=$differ newfail=$newfail newpass=$newpass"
```

`chmod +x` it and commit it with mode 100755 (`git update-index --chmod=+x`).

- [ ] **Step 2: Add the failing checks**

In `tslang/test/check-datalayout.sh`, add a wasm32 check that the layout is LLVM's real one:

```bash
# wasm32's layout used to be a hardcoded string with f128:64 and no i128:128, matching neither of
# LLVM's own wasm32 derivations; lowering now reads the TargetMachine's layout from the module.
check "wasm32-unknown-unknown"  "i128:128"
```

Run `test/check-datalayout.sh $TSLANG`. Expected: FAIL for the new wasm32 line. The old hardcode wins later in the pipeline, so it has no `i128:128`.

Run `test/compare-ir.sh $TSLANG i686-pc-windows-msvc <baseline>/i686 /tmp/p2t1-i686-before` and record `grep -l 'ptrtoint ptr [^ ]* to i64' /tmp/p2t1-i686-before/*.raw.ll | wc -l` (expected 288).

- [ ] **Step 3: Implement**

Replace the block that begins `mlir::DataLayout dl(m);` and ends with the wasm32 `if`:

```cpp
    mlir::DataLayout dl(m);
    LowerToLLVMOptions options(&getContext(), dl);
    // The type converter sizes pointers, structs and unions and computes alignment with this layout,
    // so it has to be the target's. MLIRGenModule has already put the TargetMachine's layout on the
    // module; without it LLVM's default applies, which has 8-byte pointers and a 4-byte-aligned i64 -
    // wrong for i686 in the first and for x64 in the second. Allocation sizes never came from here
    // (SizeOfOp uses getelementptr null, 1), but union storage selection, getIntPtrType and debug-info
    // offsets do.
    if (auto dataLayoutAttr = m->getAttrOfType<mlir::StringAttr>(mlir::LLVM::LLVMDialect::getDataLayoutAttrName()))
    {
        options.dataLayout = llvm::DataLayout(dataLayoutAttr.getValue());
    }
```

The wasm32 hardcoded block (both `options.dataLayout = …` and the `m->setAttr(...)`) is deleted. If `llvm::DataLayout`'s string constructor is not available in this LLVM version, use `llvm::DataLayout::parse(...)` and report the error through `m.emitError` + `signalPassFailure()` rather than ignoring it.

- [ ] **Step 4: Verify i686 and wasm32**

`test/check-datalayout.sh $TSLANG` → all ok, including the new wasm32 line.

`test/compare-ir.sh $TSLANG i686-pc-windows-msvc <baseline>/i686 /tmp/p2t1-i686` →
- `grep -l 'ptrtoint ptr [^ ]* to i64' /tmp/p2t1-i686/*.raw.ll | wc -l` must fall from 288 to a small number. Explain every file that remains: each must be genuinely 64-bit (e.g. a `bigint`), not a pointer converted at the wrong width.
- `NEWFAIL` must be 0. `NEWPASS` is welcome; list them.

- [ ] **Step 5: Explain every x64 difference**

`test/compare-ir.sh $TSLANG x86_64-pc-windows-msvc <baseline>/x64 /tmp/p2t1-x64`.

For **each** `DIFF` file, write one line in the report: what changed and why the new form is the right one for x64's real layout (e.g. a union's storage type changed because `{i32, i64}` is 16 bytes, not 12). A difference in `typeGuardOfFormTypeOfBoolean` alone is the known hash-order noise — rerun it to confirm. An unexplained difference is a finding, not noise. `NEWFAIL` must be 0.

- [ ] **Step 6: Suite and unit tests**

`ctest -j 16 -C Release` → 2765/2765. `MLIRGenTests.exe` → all pass.

- [ ] **Step 7: Commit**

```
Lower with the target's data layout, not LLVM's default

The type converter sized pointers at 8 bytes and aligned i64 at 4 for every target but wasm32,
because it was built with LLVM's default layout. MLIRGenModule already puts the TargetMachine's
layout on the module; lowering now reads it. On i686 that ends 64-bit pointer arithmetic
(<N> of 513 corpus modules had ptrtoint ptr to i64; now <M>). On x64 <K> corpus files change,
each listed in the PR: <one-line summary>.

The hardcoded wasm32 layout is gone. It carried f128:64 and lacked i128:128, matching neither of
LLVM's wasm32 derivations.
```

---

### Task 2: Refuse a triple whose pointer widths disagree

**Files:**
- Modify: `tslang/lib/TypeScript/MLIRGenModule.cpp` — the `createTargetMachine` block added in Phase 1 (~275-306)
- Modify: `tslang/test/check-datalayout.sh`

**Interfaces:**
- Consumes: `compileOptions.sizeBits()` (Phase 1), and the `llvm::TargetMachine` created there.
- Produces: no API.

For ABI-by-environment triples (`x86_64-pc-linux-gnux32`, `mips64-linux-gnuabin32`, `aarch64-linux-gnu_ilp32`), `Triple::getArchPointerBitWidth()` says 64 while the TargetMachine's layout says 32. MLIRGen would then build 64-bit sizes while lowering (after Task 1) uses 32-bit pointers. No such target is supported, so refuse it rather than miscompile it.

- [ ] **Step 1: Failing check** — in `check-datalayout.sh`, add a bad-triple case using the script's existing `check_bad_triple` helper:

```bash
# x32: the arch is 64-bit but the ABI's pointers are 32. MLIRGen and lowering would disagree.
check_bad_triple "x86_64-pc-linux-gnux32"
```

Run it → FAIL (today it compiles).

- [ ] **Step 2: Implement** — right after the layout is created:

```cpp
                auto dataLayout = machine->createDataLayout();
                // MLIRGen sizes things with the arch's pointer width (TargetInfo) and lowering with
                // this layout. For ABI-by-environment triples (gnux32, gnuabin32, ilp32) the two
                // disagree, and nothing downstream can reconcile them, so refuse the target.
                if (dataLayout.getPointerSizeInBits(0) != static_cast<unsigned>(compileOptions.sizeBits()))
                {
                    emitError(location) << "target triple '" << compileOptions.moduleTargetTriple << "' has "
                                        << dataLayout.getPointerSizeInBits(0) << "-bit pointers but a "
                                        << compileOptions.sizeBits() << "-bit architecture; this ABI is not supported";
                    return mlir::failure();
                }
```

(Use the `emitError`/`location` spelling already used by the neighboring Phase 1 error in that block.)

- [ ] **Step 3: Verify** — `check-datalayout.sh` all ok; the three supported triples still pass; `compare-ir.sh` for x64 against Task 1's output shows no differences; `ctest -j 16 -C Release` 2765/2765.

- [ ] **Step 4: Commit.**

---

### Task 3: Boehm for x86

**Files:**
- Create: `scripts/build_gc_release_vs_x86.bat`, `scripts/build_gc_debug_vs_x86.bat`, `scripts/build_gc_release_shared_vs_x86.bat`, `scripts/build_gc_debug_shared_vs_x86.bat`
- Modify: `prepare_3rdParty.bat`

**Interfaces:**
- Produces: `3rdParty/gc/x86/{release,debug}/{include,lib/gc.lib}` and `3rdParty/gcdll/x86/{release,debug}/{include,lib/gc.lib,bin/gc.dll}`. Tasks 4-6 depend on exactly these paths.

- [ ] **Step 1** — Create each x86 script as a copy of its x64 twin (`build_gc_release_vs.bat` etc.) with exactly three changes: `-A x64` → `-A Win32`; `x64` → `x86` in the `__build\...` directory; `x64` → `x86` in `CMAKE_INSTALL_PREFIX`. Keep the `-DCMAKE_MSVC_RUNTIME_LIBRARY` value, threads and cplusplus settings identical.

- [ ] **Step 2** — In `prepare_3rdParty.bat`, read an optional second argument: `set ARCH=x64` then `if not "%2"=="" set ARCH=%2`. After the existing x64 gc/gcdll blocks, add:

```bat
rem x86 (Win32) Boehm, for compiling 32-bit programs (-mtriple=i686-pc-windows-msvc). Opt-in:
rem   prepare_3rdParty.bat release x86
IF "%ARCH%"=="x86" (
  IF EXIST ".\3rdParty\gc\x86\%BUILD%\lib\gc.lib" (
    echo "No need to build x86 GC (%BUILD%)"
  ) ELSE (
    cd %p%
    @call scripts\build_gc_%BUILD%_%TOOL%_x86.bat
  )
  IF EXIST ".\3rdParty\gcdll\x86\%BUILD%\lib\gc.lib" (
    echo "No need to build x86 shared GC (%BUILD%)"
  ) ELSE (
    cd %p%
    @call scripts\build_gc_%BUILD%_shared_%TOOL%_x86.bat
  )
)
```

The script names must match that pattern. It assumes `3rdParty/gc-8.2.12` is already extracted, which the x64 block does.

- [ ] **Step 3** — Run the four x86 scripts (or `prepare_3rdParty.bat release x86` and `… debug x86`). Verify the machine type:

```bash
"$READOBJ" --file-headers 3rdParty/gc/x86/release/lib/gc.lib | grep -m1 -i "Machine"      # IMAGE_FILE_MACHINE_I386
"$READOBJ" --file-headers 3rdParty/gcdll/x86/release/bin/gc.dll | grep -m1 -i "Machine"  # IMAGE_FILE_MACHINE_I386
```

Do the same for the debug builds. Confirm the x64 trees are untouched (`git status` shows only the scripts; the `3rdParty` outputs are not tracked).

- [ ] **Step 4: Commit** the scripts and `prepare_3rdParty.bat`.

---

### Task 4: `TypeScriptAsyncRuntime` for x86, built standalone

**Files:**
- Create: `tslang/runtime/CMakeLists.txt`
- Create: `scripts/build_tslang_runtime_release_x86.bat`, `scripts/build_tslang_runtime_debug_x86.bat`

**Interfaces:**
- Consumes: `3rdParty/gc/x86/<cfg>/include` (Task 3).
- Produces: `__build/tslang-runtime/<cfg>/x86/TypeScriptAsyncRuntime.lib`. Tasks 5-6 point `--tslang-lib-path` at `__build/tslang-runtime/<cfg>`.

The in-tree `lib/TypeScriptAsyncRuntime` target is an `add_mlir_library` inside a project that needs the x64-only MLIR and Clang packages, so it cannot be configured as Win32. The library itself has no LLVM/MLIR link dependency: the only `llvm::` mentions are in comments, and its one MLIR header (`mlir/ExecutionEngine/AsyncRuntime.h`) includes only `<cstddef>` and `<stdint.h>`. `AsyncRuntime.cpp` defines `MLIR_ASYNC_RUNTIME_EXPORT` empty itself, so no dllimport issue arises.

- [ ] **Step 1** — Create `tslang/runtime/CMakeLists.txt`:

```cmake
# Builds only the library compiled programs link - TypeScriptAsyncRuntime - for any target arch,
# without MLIR or LLVM. The main project cannot do this for x86: it needs the x64-only MLIR and Clang
# packages. The runtime needs neither: it includes one header-only MLIR file for the C ABI it
# implements, and gc.h. Sources are shared with lib/TypeScriptAsyncRuntime, so the two cannot drift.
#
#   cmake -S tslang/runtime -B <dir> -A Win32 -DTSLANG_GC_INCLUDE=<gc>/include -DTSLANG_MLIR_INCLUDE=<llvm>/include
cmake_minimum_required(VERSION 3.20)
project(TypeScriptRuntimeLibs CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
# Static CRT, like everything else linked into a compiled program (see exe.cpp).
set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")

set(TSLANG_GC_INCLUDE "" CACHE PATH "Boehm include directory for the target arch")
set(TSLANG_MLIR_INCLUDE "" CACHE PATH "Directory holding mlir/ExecutionEngine/AsyncRuntime.h")
if (NOT EXISTS "${TSLANG_GC_INCLUDE}/gc.h")
  message(FATAL_ERROR "TSLANG_GC_INCLUDE must point at a directory with gc.h, got '${TSLANG_GC_INCLUDE}'")
endif()
if (NOT EXISTS "${TSLANG_MLIR_INCLUDE}/mlir/ExecutionEngine/AsyncRuntime.h")
  message(FATAL_ERROR "TSLANG_MLIR_INCLUDE must hold mlir/ExecutionEngine/AsyncRuntime.h, got '${TSLANG_MLIR_INCLUDE}'")
endif()

set(SRC "${CMAKE_CURRENT_SOURCE_DIR}/../lib/TypeScriptAsyncRuntime")
add_library(TypeScriptAsyncRuntime STATIC "${SRC}/AsyncRuntime.cpp" "${SRC}/DynamicRuntime.cpp")
target_include_directories(TypeScriptAsyncRuntime PRIVATE
  "${CMAKE_CURRENT_SOURCE_DIR}/../include"
  "${CMAKE_CURRENT_SOURCE_DIR}/../lib"
  "${TSLANG_GC_INCLUDE}"
  "${TSLANG_MLIR_INCLUDE}")
if (MSVC)
  # The same definitions the in-tree add_mlir_library build compiles with.
  target_compile_definitions(TypeScriptAsyncRuntime PRIVATE
    WIN32 _WINDOWS UNICODE _UNICODE
    _CRT_SECURE_NO_DEPRECATE _CRT_SECURE_NO_WARNINGS _CRT_NONSTDC_NO_DEPRECATE _CRT_NONSTDC_NO_WARNINGS
    _SCL_SECURE_NO_DEPRECATE _SCL_SECURE_NO_WARNINGS
    __STDC_CONSTANT_MACROS __STDC_FORMAT_MACROS __STDC_LIMIT_MACROS)
  target_compile_options(TypeScriptAsyncRuntime PRIVATE /GR /EHsc)
endif()

install(TARGETS TypeScriptAsyncRuntime ARCHIVE DESTINATION .)
```

Adjust include directories if a source includes something not covered, but add no link dependency. If a source genuinely needs an LLVM symbol, stop and report it — that contradicts the premise.

- [ ] **Step 2** — Create `scripts/build_tslang_runtime_release_x86.bat` (and a Debug twin, `Debug`/`debug`):

```bat
@rem TypeScriptAsyncRuntime for x86, linked into 32-bit programs (-mtriple=i686-pc-windows-msvc).
@rem Needs 3rdParty\gc\x86\release (scripts\build_gc_release_vs_x86.bat). Installs into
@rem __build\tslang-runtime\release\x86; pass --tslang-lib-path=__build\tslang-runtime\release.
pushd
cmake -S tslang\runtime -B __build\tslang-runtime\msbuild\x86\release -G "Visual Studio 18 2026" -A Win32 -Wno-dev -DTSLANG_GC_INCLUDE=%cd%\3rdParty\gc\x86\release\include -DTSLANG_MLIR_INCLUDE=%cd%\3rdParty\llvm\x64\release\include -DCMAKE_INSTALL_PREFIX=%cd%\__build\tslang-runtime\release\x86
cmake --build __build\tslang-runtime\msbuild\x86\release --config Release -j 8
cmake --install __build\tslang-runtime\msbuild\x86\release --config Release
popd
```

- [ ] **Step 3: Prove the standalone build is equivalent, at x64 first.** Build the same project for x64 (`-A x64`, gc include from `3rdParty/gc/x64/release/include`, prefix `__build/tslang-runtime/release`, so the lib lands flat at x64's location). Then compile and run three async suite tests at x64 against it, e.g. `00async_await.ts` plus two more `async`/`await` tests from `tslang/test/tester/tests`: `$TSLANG --emit=exe -mm=gc --no-default-lib --tslang-lib-path=__build/tslang-runtime/release <test>` and run it. Each must produce the same output as the in-tree-runtime build of the same test. This proves the standalone build before anything relies on it at x86.

- [ ] **Step 4** — Run both x86 scripts. Check `"$READOBJ" --file-headers __build/tslang-runtime/release/x86/TypeScriptAsyncRuntime.lib | grep -m1 Machine` reports `IMAGE_FILE_MACHINE_I386`, and the same for debug.

- [ ] **Step 5: Commit** the CMake project and the two scripts. `__build` is not tracked.

---

### Task 5: The compiler finds x86 libraries and checks their machine type

**Files:**
- Modify: `tslang/include/TypeScript/ObjDumper.h`, `tslang/lib/TypeScript/ObjDumper.cpp`
- Modify: `tslang/tslang/exe.cpp` — the link-argument assembly (~480-560)
- Create: `tslang/test/check-x86-libs.sh`

**Interfaces:**
- Produces: `namespace Dump { uint16_t coffMachine(llvm::StringRef path); }` — the COFF machine of a `.obj`, a static `.lib`, or an import `.lib` (its first member that has one), or `0` if none can be determined.
- Consumes: `compileOptions.moduleTargetTriple`; Task 3/4 outputs.

- [ ] **Step 1: Failing error-path checks.** Create `tslang/test/check-x86-libs.sh <tslang.exe>`. It compiles a `print("x")` file with `--emit=exe -mm=gc --no-default-lib -mtriple=i686-pc-windows-msvc` and asserts, for each case, a non-zero exit and a message containing the given substrings:
  1. `--gc-lib-path=<a temp dir with no x86 subdir>` → message contains `x86` and `build_gc_` (the script to run).
  2. `--gc-lib-path=<temp dir>` where `<temp>/x86/gc.lib` is a **copy of the x64** `gc.lib` → message contains `gc.lib`, `x64` (found) and `x86` (expected).
  3. The same two cases for `--tslang-lib-path` and `TypeScriptAsyncRuntime.lib` (script name `build_tslang_runtime_`).

  Follow `check-datalayout.sh`'s style (ok/FAIL lines, exit status). Run it now → FAIL (today's message is the linker's LNK4272 plus unresolved symbols).

- [ ] **Step 2: `Dump::coffMachine`.** In `ObjDumper.cpp`, using `llvm::object::createBinary`: for a `COFFObjectFile` return `getMachine()`; for an `Archive`, iterate children and return the machine of the first child that is a `COFFObjectFile` or a `COFFImportFile` (import libraries hold short import members, which carry a machine too); otherwise `0`. Consume every `llvm::Error`; this is a query, not a hard failure. Declare it in `ObjDumper.h` next to `containsGarbageCollector`, in the same namespace.

- [ ] **Step 3: Resolution in `exe.cpp`.** Where `gcLibPathOpt`, the shared-gc path and `tslangLibPathOpt` are built, and only when the target is Windows and `TheTriple.getArch() == llvm::Triple::x86`:
  - replace each non-empty path `P` with `P/x86`;
  - if `P/x86` does not exist: error `no x86 build of <component> in <P>/x86. Build it with scripts\<script>.bat, or point <flag> at a directory holding an x86 subdirectory.` and return 1. `<script>` is `build_gc_<cfg>_vs_x86` for gc, `build_gc_<cfg>_shared_vs_x86` for gcdll, and `build_tslang_runtime_<cfg>_x86` for the runtime; `<cfg>` follows the same debug/release choice the link already makes;
  - for the library file the link will use (`gc.lib`, `TypeScriptAsyncRuntime.lib`), check `Dump::coffMachine(file)`: if it is non-zero and not `COFF::IMAGE_FILE_MACHINE_I386`, error `<file> is built for <found arch>, but this program targets x86.` and return 1.
  - Also check x64 targets: a non-zero machine other than `IMAGE_FILE_MACHINE_AMD64` is the mirror error. This catches an x86 library placed in the flat x64 path.
  - Map machine to name with a small helper: `0x14c` → `x86`, `0x8664` → `x64`, `0xAA64` → `arm64`, else the hex value.

  Keep the helper small and local to `exe.cpp`. Do not change behaviour for non-Windows or wasm targets.

  **Existing checks that would misfire.** `getGCLibPath()`, `getGCSharedLibPath()` and `getTslangLibPath()` call `checkGCLibPath`/`checkTslangLibPath` on the *flat* path. Those print `path: '…' is not pointing to file 'gc.lib'` when the library is absent there, which it is by design for an x86-only directory. Make those existence checks target-aware — look in `P/x86` when the target is x86 — or move them into the new resolution so each path is checked exactly once, at the location actually used. A correct x86 build must print no error about the flat path. Note these getters also run for targets that never use the path; keep that behaviour unchanged for x64.

- [ ] **Step 4: Verify.** `check-x86-libs.sh $TSLANG` → all ok. `ctest -j 16 -C Release` → 2765/2765. The x64 flat-path check must not fire in the suite. `compare-ir.sh` is not relevant here (link-only change).

- [ ] **Step 5: Commit.**

---

### Task 6: 32-bit programs link and run (Phase 2 gate)

**Files:**
- Create: `tslang/test/check-x86-run.sh`
- Create: `tslang/test/x86/hello.ts`, `tslang/test/x86/gc_stress.ts`

**Interfaces:**
- Consumes: everything above. `--gc-lib-path=i:/TypeScriptCompiler/3rdParty/gc/<cfg>` would not work (x86 lives in `3rdParty/gc/x86/<cfg>`, not `<path>/x86`). Point the flags at a directory whose `x86` subdirectory holds the library. For gc that means a small staging directory: the script copies `3rdParty/gc/x86/release/lib/gc.lib` into `<tmp>/gc/x86/gc.lib`. For the runtime, `__build/tslang-runtime/release` already has that shape.

- [ ] **Step 1: Test programs.**

`tslang/test/x86/hello.ts`:
```ts
print("hello from 32-bit");
```

`tslang/test/x86/gc_stress.ts` — must force many collections and prove nothing live was freed:
```ts
class Node { constructor(public value: number, public next: Node | null) {} }

function build(n: number): Node | null {
    let head: Node | null = null;
    for (let i = 0; i < n; i++) head = new Node(i, head);
    return head;
}

function sum(head: Node | null): number {
    let s = 0;
    for (let p = head; p; p = p.next) s += p.value;
    return s;
}

// A long-lived list that must survive every collection triggered by the garbage below.
const keep = build(1000);
let strings = 0;
for (let round = 0; round < 200; round++) {
    build(5000);                                  // garbage
    const text = "round " + round + " of " + 200; // string garbage
    strings += text.length;
    const arr: number[] = [];
    for (let i = 0; i < 100; i++) arr.push(i);    // array growth garbage
}
print(sum(keep));   // 499500 - wrong or crashing if any of `keep` was collected
print(strings);
```

Compute the expected second value (sum of lengths of `"round " + r + " of 200"`) and hard-code it in the script's expected output. If tslang rejects any syntax here, adapt it while keeping the three properties: a long-lived structure that must survive, enough garbage to force many collections, and output that is wrong if the long-lived structure was freed. Confirm the program first as an x64 build — it must print the expected values there before it means anything at x86.

- [ ] **Step 2: Script.** `check-x86-run.sh <tslang.exe>` builds each program at `-mtriple=i686-pc-windows-msvc --no-default-lib` for `-mm=gc`, `-mm=rc` and `-mm=none`, runs it, and checks:
  - the exe's PE machine is `0x014c` (`"$READOBJ" --file-headers x.exe | grep Machine` → `IMAGE_FILE_MACHINE_I386`);
  - exit code 0;
  - exact expected stdout.

  `-mm=rc` does not link Boehm, so it needs only the runtime path. Prints ok/FAIL per case and exits non-zero on any FAIL.

- [ ] **Step 3: Run it.** All nine combinations (3 programs × 3 models, with hello/stress as defined) pass. **This is the Phase 2 gate.**

- [ ] **Step 4: Measure (not gated): an x86 DLL.** Compile a two-file program at x86 `-mm=gc`: `lib.ts` with `export function add(a: number, b: number) { return a + b; }` built `--emit=dll`, and `main.ts` importing it built `--emit=exe`. This uses `gcdll/x86` through Task 5's resolution. Record exactly what happens — success, a link error, a load failure, or a wrong result — and the first error text. If it fails, do **not** fix it in this task. It is a Phase 4 input.

- [ ] **Step 5: Suite.** `ctest -j 16 -C Release` → 2765/2765.

- [ ] **Step 6: Commit** the script and programs.

---

## Phase 2 gate

- [ ] `check-x86-run.sh`: hello and gc_stress run correctly as 32-bit executables under `-mm=gc`, `-mm=rc` and `-mm=none`; the PE machine is `0x014c`.
- [ ] `check-x86-libs.sh`: a missing x86 directory and a wrong-arch library each fail with a message naming the arch and the fix.
- [ ] `check-datalayout.sh`: wasm32 layout has `i128:128`; `x86_64-pc-linux-gnux32` is refused.
- [ ] i686 corpus: `ptrtoint ptr … to i64` is gone except where explained as genuinely 64-bit; no new compile failures.
- [ ] x64 corpus: every difference from the 6eac25f4 baseline is explained in Task 1's report; Tasks 2-6 add none.
- [ ] `ctest -j 16 -C Release` 2765/2765; `MLIRGenTests` all pass.
- [ ] The x86 DLL result is recorded (pass or first error) for Phase 4.
