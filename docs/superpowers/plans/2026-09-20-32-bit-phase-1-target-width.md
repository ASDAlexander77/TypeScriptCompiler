# 32-bit Phase 1: Target Width Correctness — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the compiler derive pointer/size width and target policy from the target triple rather than from a hand-maintained architecture list, so that any 32-bit triple is handled correctly.

**Architecture:** A new `TargetInfo` struct is derived once from `llvm::Triple` in `prepareOptions()` and carried on `CompileOptions` alongside the parsed `Triple`. It holds widths (`pointerBits`) and policy predicates named for the decision they answer (`usesImageBaseRelativeEH`, `stdcallDecoratesCxxThrow`, `supportsInProcessJit`). `TypeHelper` gains an optional `CompileOptions` pointer and target-width accessors; only call sites that need a width pass it.

**Tech Stack:** C++17, MLIR/LLVM (prebuilt in `3rdParty/llvm/x64`), CMake + MSVC (Visual Studio 18 2026), GoogleTest via `add_mlir_unittest`.

**Spec:** `docs/superpowers/specs/2026-09-20-32-bit-compilation-design.md`

## Global Constraints

- Proof target is `i686-pc-windows-msvc`. Width correctness must generalize to any 32-bit triple; only this one gets a native build matrix (later phases).
- The compiler itself stays x64 and cross-compiles. Never build `tslang.exe` as 32-bit.
- Policy predicates are named for what they decide, never for an architecture. `usesImageBaseRelativeEH`, never `isX64`.
- Arch naming, where it appears, is `x86` / `x64`.
- Missing per-arch resources are hard errors, never a fallback to another arch.
- This phase changes **no** behavior for existing 64-bit targets. Any diff in x64 output is a bug in this phase.
- The 18 `getI64Type()` sites in `LLVMRTTIHelperVCWin32.h` and `MLIRRTTIHelperVCWin32.h` belong to Phase 3. Do not touch those call sites here. The files themselves are not frozen: Task 2 changes `compileOptions.sizeBits` to `sizeBits()` in both, which is required.

## Deviation from the spec

The spec says `TypeHelper` "grows a `CompileOptions` parameter" and calls it "a wide but mechanical diff". Counting found 152 `TypeHelper` construction sites across 22 files, nearly all of which only use width-independent helpers (`getI8Type`, `getPtrType`, `getVoidType`). This plan instead gives `TypeHelper` an **optional** `const CompileOptions *`, defaulting to `nullptr`, and migrates only the sites that call a target-width accessor. `getSizeType()` asserts the pointer is set, so a site that needs a width and did not pass options fails loudly in debug rather than silently picking a wrong one. Net effect: same design, ~20 sites touched instead of 152.

## File Structure

| File | Responsibility |
| --- | --- |
| `include/TypeScript/TargetInfo.h` (create) | The `TargetInfo` struct and its derivation from a target/host `llvm::Triple` pair. Header-only; pure, no MLIR dependency, so it unit-tests without a compilation context. |
| `include/TypeScript/DataStructs.h` (modify) | `CompileOptions` gains `TargetInfo targetInfo`. `sizeBits` becomes a deprecated accessor over `targetInfo.pointerBits` so existing readers keep working during migration. |
| `tslang/opts.cpp` (modify) | `prepareOptions()` builds `TargetInfo` from the target and host triples. The 23-entry 64-bit arch list is deleted. |
| `include/TypeScript/LowerToLLVM/TypeHelper.h` (modify) | Optional `const CompileOptions *`; adds `getSizeType()` and `getPointerIntType()`. |
| `lib/TypeScript/MLIRGenModule.cpp` (modify) | Emits `target datalayout` on the module, not just the triple. |
| `unittests/MLIRGen/TargetInfo.cpp` (create) | GoogleTest coverage of `TargetInfo::fromTriple` across triples. |
| `unittests/MLIRGen/CMakeLists.txt` (modify) | Registers the new test file. |

## Build and test commands

Configure and build (once, if not already done):

```bash
cmake --preset windows-msbuild-2026-debug
cmake --build --preset build-windows-msbuild-2026-debug --target MLIRTypeScriptUnitTests
```

Run the unit tests:

```bash
i:/TypeScriptCompiler/__build/tslang/windows-msbuild-2026-debug/unittests/MLIRGen/Debug/MLIRGenTests.exe --gtest_filter='TargetInfoTest.*'
```

---

### Task 1: `TargetInfo` and its derivation from a triple

**Files:**
- Create: `i:/TypeScriptCompiler/tslang/include/TypeScript/TargetInfo.h`
- Create: `i:/TypeScriptCompiler/tslang/unittests/MLIRGen/TargetInfo.cpp`
- Modify: `i:/TypeScriptCompiler/tslang/unittests/MLIRGen/CMakeLists.txt`

**Interfaces:**
- Consumes: `llvm::Triple` from `llvm/TargetParser/Triple.h`.
- Produces: `struct TargetInfo` with fields `unsigned pointerBits`, `bool usesImageBaseRelativeEH`, `bool stdcallDecoratesCxxThrow`, `bool supportsInProcessJit`, and `static TargetInfo fromTriple(const llvm::Triple &target, const llvm::Triple &host)`. Tasks 2-5 and Phase 3 depend on exactly these names.

- [ ] **Step 1: Write the failing test**

Create `unittests/MLIRGen/TargetInfo.cpp`:

```cpp
#include "TypeScript/TargetInfo.h"

#include "llvm/TargetParser/Triple.h"

#include "gmock/gmock.h"

// TargetInfo replaces a hand-maintained list of 64-bit architectures in opts.cpp. That list had
// already rotted - loongarch64 was on it, riscv32 appeared nowhere - so these tests deliberately
// cover arches nobody has built for, to prove the answer comes from the triple rather than from
// somebody remembering to add a line.
namespace
{

llvm::Triple t(const char *str)
{
    return llvm::Triple(llvm::Triple::normalize(str));
}

const llvm::Triple &hostX64()
{
    static llvm::Triple host = t("x86_64-pc-windows-msvc");
    return host;
}

TargetInfo infoFor(const char *target)
{
    return TargetInfo::fromTriple(t(target), hostX64());
}

TEST(TargetInfoTest, PointerWidthComesFromTheTriple)
{
    EXPECT_EQ(infoFor("x86_64-pc-windows-msvc").pointerBits, 64u);
    EXPECT_EQ(infoFor("i686-pc-windows-msvc").pointerBits, 32u);
    EXPECT_EQ(infoFor("wasm32-unknown-unknown").pointerBits, 32u);
    EXPECT_EQ(infoFor("aarch64-unknown-linux-gnu").pointerBits, 64u);
}

// The arches the old list got wrong, in both directions.
TEST(TargetInfoTest, PointerWidthIsRightForArchesTheOldListMissed)
{
    EXPECT_EQ(infoFor("riscv32-unknown-elf").pointerBits, 32u);
    EXPECT_EQ(infoFor("armv7-unknown-linux-gnueabihf").pointerBits, 32u);
    EXPECT_EQ(infoFor("loongarch64-unknown-linux-gnu").pointerBits, 64u);
}

// An unknown arch has no width of its own; falling back to 0 would make every size computation
// degenerate, so it takes the host's.
TEST(TargetInfoTest, UnknownArchFallsBackToHostWidth)
{
    EXPECT_EQ(infoFor("unknown-unknown-unknown").pointerBits, 64u);
}

// MSVC C++ EH cross-references are image-base-relative RVAs on 64-bit Windows and absolute
// pointers on 32-bit x86. Non-Windows targets use the Itanium scheme and neither applies.
TEST(TargetInfoTest, ImageBaseRelativeEHIsWin64Only)
{
    EXPECT_TRUE(infoFor("x86_64-pc-windows-msvc").usesImageBaseRelativeEH);
    EXPECT_FALSE(infoFor("i686-pc-windows-msvc").usesImageBaseRelativeEH);
    EXPECT_FALSE(infoFor("x86_64-unknown-linux-gnu").usesImageBaseRelativeEH);
}

// _CxxThrowException is __stdcall on 32-bit x86 only, which is what puts the @8 on the symbol.
TEST(TargetInfoTest, StdcallThrowIsWin32X86Only)
{
    EXPECT_TRUE(infoFor("i686-pc-windows-msvc").stdcallDecoratesCxxThrow);
    EXPECT_FALSE(infoFor("x86_64-pc-windows-msvc").stdcallDecoratesCxxThrow);
    EXPECT_FALSE(infoFor("i686-unknown-linux-gnu").stdcallDecoratesCxxThrow);
}

// tslang.exe is an x64 process and cannot execute i386 or wasm in-process.
TEST(TargetInfoTest, InProcessJitNeedsAMatchingHost)
{
    EXPECT_TRUE(infoFor("x86_64-pc-windows-msvc").supportsInProcessJit);
    EXPECT_FALSE(infoFor("i686-pc-windows-msvc").supportsInProcessJit);
    EXPECT_FALSE(infoFor("wasm32-unknown-unknown").supportsInProcessJit);
    EXPECT_FALSE(infoFor("x86_64-unknown-linux-gnu").supportsInProcessJit);
}

} // namespace
```

Register it by adding `TargetInfo.cpp` to `unittests/MLIRGen/CMakeLists.txt`:

```cmake
add_mlir_unittest(MLIRGenTests
  TypeToString.cpp
  DeclarationPrinter.cpp
  ExportFilter.cpp
  TypeHelper.cpp
  TargetInfo.cpp
)
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
cmake --build --preset build-windows-msbuild-2026-debug --target MLIRTypeScriptUnitTests
```

Expected: compile error, `Cannot open include file: 'TypeScript/TargetInfo.h'`.

- [ ] **Step 3: Write the implementation**

Create `include/TypeScript/TargetInfo.h`:

```cpp
#ifndef TYPESCRIPT_TARGETINFO_H_
#define TYPESCRIPT_TARGETINFO_H_

#include "llvm/TargetParser/Triple.h"

// Everything the compiler needs to know about the target beyond the triple string itself,
// derived once in prepareOptions() and carried on CompileOptions.
//
// Predicates are named for the decision they answer, never for the architecture that currently
// needs them - `usesImageBaseRelativeEH`, not `isX64`. A new target then answers the questions,
// instead of every call site growing another arch comparison. This replaces a hand-maintained
// list of 64-bit arches in opts.cpp that had already rotted.
struct TargetInfo
{
    // Width of a pointer, and of the integer type used for sizes and indices.
    unsigned pointerBits = 64;

    // MSVC C++ EH stores the cross-references inside ThrowInfo and CatchableType as
    // image-base-relative RVAs on 64-bit Windows, and as absolute pointers on 32-bit x86. Both
    // are 4 bytes, so the struct layouts coincide and only the stored value differs: x86 must
    // not subtract the image base. See Phase 3.
    bool usesImageBaseRelativeEH = true;

    // _CxxThrowException is __stdcall on 32-bit x86, so the emitted symbol carries the @8
    // suffix that the linker asks for.
    bool stdcallDecoratesCxxThrow = false;

    // Whether tslang.exe - an x64 process - can execute this target's code in-process.
    bool supportsInProcessJit = true;

    static TargetInfo fromTriple(const llvm::Triple &target, const llvm::Triple &host)
    {
        TargetInfo info;

        info.pointerBits = target.getArchPointerBitWidth();
        if (info.pointerBits == 0)
        {
            // An unknown arch has no width of its own. Zero would make every size computation
            // degenerate, so take the host's - the same assumption the compiler already makes
            // when no triple is given at all.
            info.pointerBits = host.getArchPointerBitWidth();
        }

        const bool msvc = target.isKnownWindowsMSVCEnvironment();
        const bool x86_32 = target.getArch() == llvm::Triple::x86;

        info.usesImageBaseRelativeEH = msvc && !x86_32;
        info.stdcallDecoratesCxxThrow = msvc && x86_32;
        info.supportsInProcessJit =
            target.getArch() == host.getArch() && target.getOS() == host.getOS();

        return info;
    }
};

#endif // TYPESCRIPT_TARGETINFO_H_
```

- [ ] **Step 4: Run the test to verify it passes**

```bash
cmake --build --preset build-windows-msbuild-2026-debug --target MLIRTypeScriptUnitTests
i:/TypeScriptCompiler/__build/tslang/windows-msbuild-2026-debug/unittests/MLIRGen/Debug/MLIRGenTests.exe --gtest_filter='TargetInfoTest.*'
```

Expected: 6 tests, all PASS.

- [ ] **Step 5: Commit**

```bash
git add tslang/include/TypeScript/TargetInfo.h tslang/unittests/MLIRGen/TargetInfo.cpp tslang/unittests/MLIRGen/CMakeLists.txt
git commit -m "Derive target width and policy from the triple

TargetInfo answers what the compiler needs to know about a target:
pointer width, and the policy questions Win32 EH and the JIT ask. Named
for the decisions rather than the arches, so a new target answers them
instead of each call site growing an arch comparison.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 2: Carry `TargetInfo` on `CompileOptions` and delete the arch list

**Files:**
- Modify: `i:/TypeScriptCompiler/tslang/include/TypeScript/DataStructs.h:9-24`
- Modify: `i:/TypeScriptCompiler/tslang/tslang/opts.cpp:36-91`

**Interfaces:**
- Consumes: `TargetInfo::fromTriple` from Task 1.
- Produces: `CompileOptions::targetInfo` (a `TargetInfo`) and `CompileOptions::sizeBits` retained as an accessor returning `targetInfo.pointerBits`. Every existing reader of `compileOptions.sizeBits` keeps compiling unchanged — that is the point of keeping it.

- [ ] **Step 1: Write the failing test**

Append to `unittests/MLIRGen/TargetInfo.cpp`, inside the anonymous namespace:

```cpp
// sizeBits is what ~20 existing call sites read. It must keep answering, and it must now answer
// from TargetInfo rather than from a separate field that could drift away from it.
TEST(TargetInfoTest, CompileOptionsSizeBitsFollowsTargetInfo)
{
    CompileOptions options;

    options.targetInfo = infoFor("i686-pc-windows-msvc");
    EXPECT_EQ(options.sizeBits(), 32);

    options.targetInfo = infoFor("x86_64-pc-windows-msvc");
    EXPECT_EQ(options.sizeBits(), 64);
}
```

Add the include at the top of the file, below the `TargetInfo.h` include:

```cpp
#include "TypeScript/DataStructs.h"
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
cmake --build --preset build-windows-msbuild-2026-debug --target MLIRTypeScriptUnitTests
```

Expected: compile error — `CompileOptions` has no member `targetInfo`, and `sizeBits` is an `int` data member, not callable.

- [ ] **Step 3: Write the implementation**

In `include/TypeScript/DataStructs.h`, add the include at the top:

```cpp
#include "TypeScript/TargetInfo.h"
```

Then replace the `int sizeBits;` member (line 20) with:

```cpp
    TargetInfo targetInfo;

    // Pointer/size width in bits. Kept as an accessor rather than a field so it cannot drift
    // away from targetInfo, which is the single place the target's width is decided.
    int sizeBits() const
    {
        return static_cast<int>(targetInfo.pointerBits);
    }
```

In `tslang/opts.cpp`, add the include:

```cpp
#include "TypeScript/TargetInfo.h"
```

Replace `compileOptions.sizeBits = 32;` (line 57) with:

```cpp
    compileOptions.targetInfo = TargetInfo::fromTriple(
        TheTriple, llvm::Triple(llvm::sys::getDefaultTargetTriple()));
```

Delete the entire `if (TheTriple.getArch() == llvm::Triple::UnknownArch || ...) { compileOptions.sizeBits = 64; }` block (lines 67-91). It is replaced by `getArchPointerBitWidth()` inside `TargetInfo::fromTriple`.

Then update every reader of `compileOptions.sizeBits` to call it. The sites, from the spec's audit:

- `include/TypeScript/LowerToLLVM/LLVMCodeHelperBase.h:279`
- `include/TypeScript/LowerToLLVM/LLVMRTTIHelperVCWin32.h:118`
- `include/TypeScript/LowerToLLVM/ThrowLogic.h:149`
- `include/TypeScript/MLIRLogic/MLIRRTTIHelperVCWin32.h:143`
- `include/TypeScript/MLIRLogic/MLIRTypeHelper.h:1646, 1651, 1866, 1915, 1920`
- `lib/TypeScript/LowerToLLVM.cpp:5250, 5293, 7257`
- `lib/TypeScript/MLIRGenModule.cpp:263`
- `tslang/transform.cpp:347, 348`
- `unittests/MLIRGen/TypeHelper.cpp:36`

Each is a mechanical `compileOptions.sizeBits` → `compileOptions.sizeBits()`. The one exception is `unittests/MLIRGen/TypeHelper.cpp:36`, which *assigns* it; change that line from `compileOptions.sizeBits = 64;` to:

```cpp
        compileOptions.targetInfo = TargetInfo::fromTriple(
            llvm::Triple("x86_64-pc-windows-msvc"), llvm::Triple("x86_64-pc-windows-msvc"));
```

- [ ] **Step 4: Run the tests to verify they pass**

```bash
cmake --build --preset build-windows-msbuild-2026-debug --target MLIRTypeScriptUnitTests
i:/TypeScriptCompiler/__build/tslang/windows-msbuild-2026-debug/unittests/MLIRGen/Debug/MLIRGenTests.exe
```

Expected: all tests PASS, including the pre-existing `MLIRTypeHelperTest` suite. If any `MLIRTypeHelperTest` test now fails, the width default changed for x64 — that is a regression in this task, not an acceptable diff.

- [ ] **Step 5: Verify x64 output is byte-identical**

This phase must not change 64-bit behavior. Build the compiler and diff its IR against the current one:

```bash
cmake --build --preset build-windows-msbuild-2026-release
cd "$TMPDIR" && printf 'print("x");\n' > w.ts
i:/TypeScriptCompiler/__build/tslang/windows-msbuild-2026-release/bin/tslang.exe \
  --emit=llvm -mm=gc --no-default-lib w.ts && mv w.ll after.ll
git stash && cmake --build --preset build-windows-msbuild-2026-release
i:/TypeScriptCompiler/__build/tslang/windows-msbuild-2026-release/bin/tslang.exe \
  --emit=llvm -mm=gc --no-default-lib w.ts && mv w.ll before.ll
git stash pop
diff before.ll after.ll
```

Expected: no differences.

- [ ] **Step 6: Commit**

```bash
git add tslang/include/TypeScript/DataStructs.h tslang/tslang/opts.cpp tslang/unittests/MLIRGen/TargetInfo.cpp tslang/unittests/MLIRGen/TypeHelper.cpp
git commit -m "Take pointer width from the triple, not from an arch list

opts.cpp listed 23 architectures it considered 64-bit. The list had
already rotted: loongarch64 was on it, riscv32 appeared nowhere. Triple
already knows, so ask it.

sizeBits stays, as an accessor over TargetInfo, so it cannot drift from
the one place the width is decided.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 3: Target-width accessors on `TypeHelper`

**Files:**
- Modify: `i:/TypeScriptCompiler/tslang/include/TypeScript/LowerToLLVM/TypeHelper.h:19-30,62-66`
- Modify: `i:/TypeScriptCompiler/tslang/unittests/MLIRGen/TargetInfo.cpp`

**Interfaces:**
- Consumes: `CompileOptions::targetInfo` from Task 2.
- Produces: `TypeHelper::getSizeType()` and `TypeHelper::getPointerIntType()`, both returning `mlir::Type`; and constructors `TypeHelper(OpBuilder &, const CompileOptions &)` and `TypeHelper(MLIRContext *, const CompileOptions &)`. The existing two constructors stay, so the 152 existing sites are untouched.

- [ ] **Step 1: Write the failing test**

Append to `unittests/MLIRGen/TargetInfo.cpp`, inside the anonymous namespace:

```cpp
// TypeHelper is constructed at 152 sites, nearly all of which only ask for width-independent
// types. Rather than thread CompileOptions through all of them, the options are optional and
// only the sites that need a width pass them - so getSizeType() has to answer from the target,
// and has to refuse rather than guess when nobody supplied one.
TEST(TargetInfoTest, TypeHelperSizeTypeFollowsTheTarget)
{
    mlir::MLIRContext context;

    CompileOptions options32;
    options32.targetInfo = infoFor("i686-pc-windows-msvc");
    EXPECT_EQ(typescript::TypeHelper(&context, options32).getSizeType(),
              mlir::IntegerType::get(&context, 32));

    CompileOptions options64;
    options64.targetInfo = infoFor("x86_64-pc-windows-msvc");
    EXPECT_EQ(typescript::TypeHelper(&context, options64).getSizeType(),
              mlir::IntegerType::get(&context, 64));
}
```

Add these includes at the top of the file:

```cpp
#include "TypeScript/LowerToLLVM/TypeHelper.h"

#include "mlir/IR/MLIRContext.h"
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
cmake --build --preset build-windows-msbuild-2026-debug --target MLIRTypeScriptUnitTests
```

Expected: compile error — no `TypeHelper` constructor taking `CompileOptions`, and no member `getSizeType`.

- [ ] **Step 3: Write the implementation**

In `include/TypeScript/LowerToLLVM/TypeHelper.h`, add the include:

```cpp
#include "TypeScript/DataStructs.h"
```

Replace the class's member and constructor block (lines 19-30) with:

```cpp
class TypeHelper
{
    MLIRContext *context;

    // Optional on purpose. TypeHelper is constructed at ~152 sites, nearly all of which only ask
    // for width-independent types (getI8Type, getPtrType, getVoidType). Threading CompileOptions
    // through all of them would be a large diff for no benefit, so only the sites that ask for a
    // target width pass it - and getSizeType() asserts rather than guessing when they have not.
    const CompileOptions *compileOptions = nullptr;

  public:
    TypeHelper(OpBuilder &rewriter) : context(rewriter.getContext())
    {
    }

    TypeHelper(MLIRContext *context) : context(context)
    {
    }

    TypeHelper(OpBuilder &rewriter, const CompileOptions &compileOptions)
        : context(rewriter.getContext()), compileOptions(&compileOptions)
    {
    }

    TypeHelper(MLIRContext *context, const CompileOptions &compileOptions)
        : context(context), compileOptions(&compileOptions)
    {
    }
```

Then add the accessors immediately after `getI64Type()` (line 66):

```cpp
    // An integer as wide as the target's pointer: object sizes, byte counts, GEP indices.
    // Distinct from getI64Type(), which is for values that are 64-bit whatever the target is -
    // the language's own 64-bit integers and runtime ABI parameters declared int64_t.
    mlir::Type getSizeType()
    {
        assert(compileOptions && "getSizeType() needs the target: construct TypeHelper with CompileOptions");
        return mlir::IntegerType::get(context, compileOptions->targetInfo.pointerBits);
    }

    // The integer a pointer converts to under ptrtoint on this target.
    mlir::Type getPointerIntType()
    {
        return getSizeType();
    }
```

- [ ] **Step 4: Run the test to verify it passes**

```bash
cmake --build --preset build-windows-msbuild-2026-debug --target MLIRTypeScriptUnitTests
i:/TypeScriptCompiler/__build/tslang/windows-msbuild-2026-debug/unittests/MLIRGen/Debug/MLIRGenTests.exe --gtest_filter='TargetInfoTest.*'
```

Expected: 8 tests, all PASS.

- [ ] **Step 5: Commit**

```bash
git add tslang/include/TypeScript/LowerToLLVM/TypeHelper.h tslang/unittests/MLIRGen/TargetInfo.cpp
git commit -m "Give TypeHelper target-width accessors

getSizeType() is for things that are as wide as a pointer; getI64Type()
stays for things that are 64-bit whatever the target is. Keeping both
means the audit in the next commit can record which each site meant.

The options are optional because 152 sites construct TypeHelper and
almost none need a width. getSizeType() asserts rather than guessing.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 4: Emit `target datalayout` on the module

**Files:**
- Modify: `i:/TypeScriptCompiler/tslang/lib/TypeScript/MLIRGenModule.cpp:254-264`

**Interfaces:**
- Consumes: `CompileOptions::sizeBits()` from Task 2.
- Produces: no new API. The emitted LLVM module gains a `target datalayout` line.

Today the module carries only `llvm.target_triple`, and the data layout is set much later, in `tslang/obj.cpp:248`, from the `TargetMachine`. So `--emit=llvm` output is not self-describing: it names a 32-bit target while carrying LLVM's default layout. Anything reasoning about size between MLIRGen and `obj.cpp` is guessing.

- [ ] **Step 1: Write the failing test**

This is an end-to-end property of emitted IR, and the repo has no lit/FileCheck harness — the test is a shell check. Create `i:/TypeScriptCompiler/tslang/test/check-datalayout.sh`:

```bash
#!/usr/bin/env bash
# Phase 1 gate: emitted IR must name a data layout that matches its triple, so that --emit=llvm
# output is self-describing rather than carrying LLVM's default layout under a 32-bit triple.
set -u
TSLANG="${1:?usage: check-datalayout.sh <path to tslang.exe>}"
work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT
printf 'print("x");\n' > "$work/dl.ts"

fail=0
check() {
    local triple="$1" expect="$2"
    "$TSLANG" --emit=llvm -mm=none --no-default-lib -mtriple="$triple" "$work/dl.ts" \
        -o "$work/dl.ll" >/dev/null 2>&1
    local got
    got="$(grep -m1 '^target datalayout' "$work/dl.ll" || true)"
    if [ -z "$got" ]; then
        echo "FAIL $triple: no 'target datalayout' in emitted IR"
        fail=1
    elif ! printf '%s' "$got" | grep -q -- "$expect"; then
        echo "FAIL $triple: expected a layout containing '$expect', got: $got"
        fail=1
    else
        echo "ok   $triple"
    fi
}

# p:32: and p:64: are the pointer-size entries; they are what this phase is about.
check "i686-pc-windows-msvc"    "p:32:"
check "wasm32-unknown-unknown"  "p:32:"
check "x86_64-pc-windows-msvc"  "-p270:" # x86-64 layouts carry the AS270/271/272 entries

exit "$fail"
```

- [ ] **Step 2: Run it to verify it fails**

```bash
chmod +x i:/TypeScriptCompiler/tslang/test/check-datalayout.sh
i:/TypeScriptCompiler/tslang/test/check-datalayout.sh \
  i:/TypeScriptCompiler/__build/tslang/windows-msbuild-2026-release/bin/tslang.exe
```

Expected: all three FAIL with "no 'target datalayout' in emitted IR".

- [ ] **Step 3: Write the implementation**

In `lib/TypeScript/MLIRGenModule.cpp`, inside the `if (!compileOptions.moduleTargetTriple.empty())` block, after the `dlti.dl_spec` assignment, add:

```cpp
            // The data layout is otherwise only set in obj.cpp, from the TargetMachine, which is
            // after --emit=llvm has already printed the module. Setting it here makes the emitted
            // IR self-describing: a 32-bit triple no longer prints alongside LLVM's default
            // 64-bit layout. Derived from the triple via LLVM's own target registry so it matches
            // exactly what obj.cpp will later set, rather than being a second hand-built string.
            std::string errorMessage;
            if (auto *target = llvm::TargetRegistry::lookupTarget(
                    compileOptions.moduleTargetTriple, errorMessage))
            {
                std::unique_ptr<llvm::TargetMachine> machine(target->createTargetMachine(
                    llvm::Triple(compileOptions.moduleTargetTriple), "generic", "",
                    llvm::TargetOptions(), std::nullopt));
                if (machine)
                {
                    theModule->setAttr(
                        mlir::LLVM::LLVMDialect::getDataLayoutAttrName(),
                        builder.getStringAttr(machine->createDataLayout().getStringRepresentation()));
                }
            }
```

Add the includes at the top of `MLIRGenModule.cpp`:

```cpp
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Target/TargetMachine.h"
```

- [ ] **Step 4: Run it to verify it passes**

```bash
cmake --build --preset build-windows-msbuild-2026-release
i:/TypeScriptCompiler/tslang/test/check-datalayout.sh \
  i:/TypeScriptCompiler/__build/tslang/windows-msbuild-2026-release/bin/tslang.exe
```

Expected: three `ok` lines, exit 0.

Target registration is already handled and needs no work here: `tslang.cpp:337` calls `llvm::InitializeAllTargets()` in `main`, before any compilation, and that function calls `InitializeAllTargetInfos()` internally (see `llvm/Support/TargetSelect.h`). So every configured backend, wasm included, is in the registry by the time `MLIRGenModule` runs. If `lookupTarget` nevertheless fails for a triple, that backend is not in this LLVM build's `Targets.def` — check before assuming a registration bug.

- [ ] **Step 5: Commit**

```bash
git add tslang/lib/TypeScript/MLIRGenModule.cpp tslang/test/check-datalayout.sh
git commit -m "Set the data layout on the emitted module

The layout was only set in obj.cpp, from the TargetMachine, which is
after --emit=llvm has printed. So IR emitted for a 32-bit triple carried
LLVM's default 64-bit layout, and anything reading sizes between MLIRGen
and obj.cpp was guessing.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 5: Audit and classify the 19 non-EH `getI64Type()` sites

**Files:**
- Modify (audit, some will change): `lib/TypeScript/GCPass.cpp` (1), `lib/TypeScript/LowerToLLVM.cpp` (2), `lib/TypeScript/MLIRGenAccessCall.cpp` (1), `lib/TypeScript/MLIRGenClasses.cpp` (4), `lib/TypeScript/MLIRGenExpressions.cpp` (1), `lib/TypeScript/MLIRGenImpl.h` (1), `lib/TypeScript/MLIRGenInterfaces.cpp` (2), `include/TypeScript/LowerToLLVM/CastLogicHelper.h` (2), `include/TypeScript/LowerToLLVM/CodeLogicHelper.h` (1), `include/TypeScript/LowerToLLVM/ConvertLogic.h` (1), `include/TypeScript/LowerToLLVM/ThrowLogic.h` (1), `include/TypeScript/LowerToLLVM/TypeHelper.h` (1, the definition), `include/TypeScript/MLIRLogic/MLIRTypeHelper.h` (2)

**Interfaces:**
- Consumes: `TypeHelper::getSizeType()` from Task 3.
- Produces: no new API. Each audited site carries a one-line comment recording its classification.

**Do not touch** `include/TypeScript/LowerToLLVM/LLVMRTTIHelperVCWin32.h` (9 sites) or `include/TypeScript/MLIRLogic/MLIRRTTIHelperVCWin32.h` (9 sites). Those are Phase 3's; it rewrites that arithmetic anyway, and changing them here would collide.

- [ ] **Step 1: Enumerate the sites**

```bash
cd i:/TypeScriptCompiler/tslang
grep -rn 'getI64Type()' lib include \
  | grep -v 'RTTIHelperVCWin32.h' \
  > /tmp/i64-audit.txt
wc -l /tmp/i64-audit.txt
```

Expected: 20 lines. **One of them is not a call site:** `include/TypeScript/LowerToLLVM/TypeHelper.h:62` is the *definition* of `getI64Type()` itself. Leave it alone. That leaves **19 call sites** to classify.

- [ ] **Step 2: Classify each site**

For each line, read the surrounding code and decide which of two things the `i64` means:

- **genuinely 64-bit** — the value is 64 bits whatever the target is. The language's own 64-bit integer types; runtime ABI parameters declared `int64_t` in `lib/AsyncRuntimeCommon.inc` (`mlirAsyncRuntimeAddRef`, `mlirAsyncRuntimeCreateValue`, `mlirAsyncRuntimeAddTokenToGroup`); anything whose width is fixed by a contract outside the target. **Leave the code as is** and add a comment saying why.
- **target-width** — a pointer converted with `ptrtoint`, an object size, a byte count, a GEP index. **Change to `getSizeType()`.**

Add a one-line comment at each site recording the decision, in the form:

```cpp
// 64-bit whatever the target: mlirAsyncRuntimeAddRef takes int64_t, not size_t.
```

or

```cpp
// target-width: this is a byte count for the allocation below.
```

The comment is the deliverable here as much as any code change. The point of the audit is that the classification becomes explicit and survives future edits — a bare `getI64Type()` gives a later reader nothing to check against.

- [ ] **Step 3: Run the unit tests**

```bash
cmake --build --preset build-windows-msbuild-2026-debug --target MLIRTypeScriptUnitTests
i:/TypeScriptCompiler/__build/tslang/windows-msbuild-2026-debug/unittests/MLIRGen/Debug/MLIRGenTests.exe
```

Expected: all PASS.

- [ ] **Step 4: Verify x64 output is unchanged**

Any site correctly classified as target-width emits `i64` on x64 either way, so x64 IR must be byte-identical. A diff here means a site was misclassified.

```bash
cmake --build --preset build-windows-msbuild-2026-release
cd "$TMPDIR" && printf 'class C { x: number; f() { return this.x; } }\nconst c = new C();\nc.x = 1;\nprint(c.f());\n' > a.ts
i:/TypeScriptCompiler/__build/tslang/windows-msbuild-2026-release/bin/tslang.exe \
  --emit=llvm -mm=gc --no-default-lib a.ts && mv a.ll after.ll
git stash && cmake --build --preset build-windows-msbuild-2026-release
i:/TypeScriptCompiler/__build/tslang/windows-msbuild-2026-release/bin/tslang.exe \
  --emit=llvm -mm=gc --no-default-lib a.ts && mv a.ll before.ll
git stash pop
diff before.ll after.ll
```

Expected: no differences.

- [ ] **Step 5: Run the full test suite**

The suite is the real check that no 64-bit behavior moved. Run it exactly as for a normal change, Release:

```bash
cmake --build --preset build-windows-msbuild-2026-release
cd i:/TypeScriptCompiler/__build/tslang/windows-msbuild-2026-release && ctest -j 16
```

Expected: the same result as `main` — 2765/2765 as of 2026-09-16. Any new failure is this task's, not a pre-existing one; confirm by re-running that test on `main` before concluding otherwise.

- [ ] **Step 6: Commit**

```bash
git add -A tslang/lib tslang/include
git commit -m "Classify every i64 in lowering as fixed-width or target-width

Each of the 19 non-EH getI64Type() call sites now says which it meant. Most
are genuinely 64-bit - the async runtime's parameters are int64_t, not
size_t, so i64 is right on x86 too. The rest are pointer and size
arithmetic and now ask the target.

x64 IR is unchanged, which is the check that the classification is
right: a target-width site emits i64 on x64 either way.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 6: Refuse `--emit=jit` for a non-host target

**Files:**
- Modify: `i:/TypeScriptCompiler/tslang/tslang/jit.cpp:357` (the first statement of `runJit`)

**Interfaces:**
- Consumes: `CompileOptions::targetInfo.supportsInProcessJit` from Task 1.
- Produces: no new API.

`tslang.exe` is an x64 process and cannot execute i386 or wasm code in-process. The spec requires this be refused up front with a message that says so.

- [ ] **Step 1: Find out what it does today**

```bash
cd "$TMPDIR" && printf 'print("x");\n' > j.ts
i:/TypeScriptCompiler/__build/tslang/windows-msbuild-2026-release/bin/tslang.exe \
  --emit=jit -mm=none --no-default-lib -mtriple=i686-pc-windows-msvc j.ts; echo "exit=$?"
```

Record the observed behavior in the commit message. If it already refuses with a clear message, close this task as unnecessary and say so rather than adding a second check.

- [ ] **Step 2: Write the guard**

The guard goes at the **top of `runJit`**, as the first statement of the function body at `tslang/jit.cpp:359` — before `PrintStackTraceOnErrorSignal`, `registerMLIRDialects` and `InitializeNativeTarget`. Nothing should be initialized for a run that cannot happen. `compileOptions` is a parameter of `runJit` (line 357), so it is in scope.

Note there is a *second* `llvm::Triple TheTriple` block further down at line 554; that one is inside the `dumpToObjectFile` path and is **not** where this guard belongs.

```cpp
int runJit(int argc, char **argv, mlir::ModuleOp module, CompileOptions &compileOptions)
{
    // tslang.exe is an x64 process: it cannot execute i386 or wasm code in-process. Refused up
    // front rather than left to LLJIT, whose failure for this is a relocation or "symbol not
    // found" error deep in the session that never mentions the triple as the cause.
    if (!compileOptions.targetInfo.supportsInProcessJit)
    {
        llvm::WithColor::error(llvm::errs(), "tslang")
            << "--emit=jit runs the code in this process, which is "
            << llvm::sys::getDefaultTargetTriple() << ", so it cannot run code built for "
            << compileOptions.moduleTargetTriple
            << ". Build it instead with --emit=exe or --emit=obj.\n";
        return -1;
    }

    // to avoid false positive memory leak reports in release builds
    // ... existing body continues unchanged ...
```

- [ ] **Step 3: Verify the guard fires**

```bash
i:/TypeScriptCompiler/__build/tslang/windows-msbuild-2026-release/bin/tslang.exe \
  --emit=jit -mm=none --no-default-lib -mtriple=i686-pc-windows-msvc j.ts; echo "exit=$?"
```

Expected: the new error message, non-zero exit.

- [ ] **Step 4: Verify the host case still works**

```bash
i:/TypeScriptCompiler/__build/tslang/windows-msbuild-2026-release/bin/tslang.exe \
  --emit=jit -mm=gc j.ts; echo "exit=$?"
```

Expected: prints `x`, exit 0.

- [ ] **Step 5: Run the full suite**

```bash
cd i:/TypeScriptCompiler/__build/tslang/windows-msbuild-2026-release && ctest -j 16
```

Expected: unchanged from Task 5.

- [ ] **Step 6: Commit**

```bash
git add tslang/tslang/jit.cpp
git commit -m "Refuse --emit=jit for a target this process cannot run

tslang.exe is x64 and cannot execute i386 or wasm in-process. Left to
LLJIT this surfaces as a relocation failure deep in the session that
never mentions the triple.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

## Phase 1 gate

All of the following, before Phase 2 is planned:

- [ ] `MLIRGenTests.exe` passes, including the new `TargetInfoTest` suite.
- [ ] `test/check-datalayout.sh` passes for `i686-pc-windows-msvc`, `wasm32-unknown-unknown` and `x86_64-pc-windows-msvc`.
- [ ] `ctest` matches `main` — 2765/2765.
- [ ] x64 `--emit=llvm` output is byte-identical to `main` for a class-and-method sample.
- [ ] `--emit=jit -mtriple=i686-pc-windows-msvc` refuses with a message naming both triples.
- [ ] Every one of the 19 non-EH `getI64Type()` call sites carries a classification comment (the 20th grep hit is the definition in `TypeHelper.h:62`).
- [ ] The 18 EH `getI64Type()` sites are untouched. The two RTTI helper files are *not* otherwise frozen: Task 2 legitimately changes `compileOptions.sizeBits` to `sizeBits()` at `LLVMRTTIHelperVCWin32.h:118` and `MLIRRTTIHelperVCWin32.h:143`. Verify with `git diff -- <the two helpers> | grep getI64Type`, which must be empty — not with `git diff --stat`, which will legitimately be non-empty.
