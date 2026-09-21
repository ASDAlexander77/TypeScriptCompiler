# 32-bit Phase 4b: Async at i686 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** async/await programs, including `for await`, compile and run correctly as 32-bit executables.

**Architecture:** Both i686 async failures come from the upstream `ConvertAsyncToLLVMPass` (`3rdParty/llvm-project/mlir/lib/Conversion/AsyncToLLVM/AsyncToLLVM.cpp`), which assumes 64 bits in two places. The pass is part of the prebuilt LLVM, so it can't be patched here. Instead, one small tslang MLIR pass repairs its output for targets whose pointer width is below 64:

1. **The coroutine frame allocator.** `CoroBeginOpConversion` calls `lookupOrCreateAlignedAllocFn(…, rewriter.getI64Type())` and computes the size with `llvm.coro.size.i64`. That declares `aligned_alloc(i64, i64)`. `GCPass` renames it to `GC_memalign`; without GC, it resolves to the runtime's `aligned_alloc` shim. Both are `(size_t, size_t)`, which is 32-bit at i686. The callee therefore reads `alignment = 4, size = 0`, and the 16-byte frame overflows the tiny block it gets back.
   - Evidence: a crash dump of `await g()` at i686. The worker thread faults reading `0x1c` in `mlirAsyncRuntimeEmplaceToken`, which `async_execute_fn` calls with a null token.
2. **Index values crossing into `i64` runtime parameters.** `async.runtime.create_group(%size : index)` and similar operations are lowered with a converter whose index is 64-bit. The runtime really does take `int64_t` there (`lib/AsyncRuntimeCommon.inc`), so `i64` is correct. Our i686 `LLVMTypeConverter` makes `index` an `i32`, which leaves `unrealized_conversion_cast %x : i32 to index` followed by `… : index to i64`. LLVM translation rejects that. This is `00for_await`'s "LLVM Translation failed for operation: builtin.unrealized_conversion_cast".

**Tech Stack:** C++17, the MLIR LLVM dialect, tslang's MLIR pass pipeline (`tslang/tslang/transform.cpp`), and bash test scripts.

**Spec:** `docs/superpowers/specs/2026-09-20-32-bit-compilation-design.md`, Phase 4 ("4b — async at i686") and "Open issues".

## Global Constraints

- **x64 output is unchanged.** The new pass is added only when `compileOptions.sizeBits() < 64`. Prove it with `tslang/test/compare-ir.sh` against a baseline from this plan's base commit. The only allowed differences are the noise files listed in `tslang/test/mask-ir.sh`.
- **wasm32 has pointers below 64 bits too.** The pass is target-generic (pointer width), not i686-specific. wasm32 async output may change; report it. `check-datalayout.sh` must stay green.
- Hard errors, not silent fallbacks. If the pass finds an `aligned_alloc` whose shape it doesn't expect (other than two integer parameters and a `ptr` result), or a cast chain it can't lower, it emits an error.
- **Suite:** `ctest -j 16 -C Release` stays all-green. **Corpus:** `tslang/test/probe-corpus.sh --compare` against a baseline from the plan's base commit shows no row that passed before and fails after. Two rows are known to flip between runs: `00array8_tuple_spread` and `arrayLiterals2ES5` at x86 under rc.
- **x86 lib flags:** as in `tslang/test/check-x86-run.sh`: `--gc-lib-path=<repo>/3rdParty/gc/x64/release/lib`, `--tslang-lib-path=<repo>/__build/tslang-runtime/release`, and unset the `*_LIB_PATH` env vars.
- **The async runtime needs Boehm's thread API**, so async programs link only under `-mm=gc`, at x64 as well: under rc and none the link fails with unresolved `GC_allow_register_threads`. Changing that is out of scope. The gate is `-mm=gc`.
- Commits are GPG-signed; never bypass signing. The trailer is `Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>`.

## Measured starting point (main 9a521c31)

At i686 under `-mm=gc`:
- `00async_await`, `00async_gc_threading`, `00async_result_types` and `00owned_async` crash with exit 139.
- `00for_await` and `00for_await_yield` fail to compile.

At x64 all six pass. A minimal repro, `async function g() { print("in g"); } function main() { print("start"); await g(); print("after await"); }`, crashes at i686 and passes at x64.

## File structure

| File | Change | Task |
| --- | --- | --- |
| `tslang/test/check-x86-run.sh` | Add the async corpus cases (gc) and an IR check on the frame allocator's width. | 1 |
| `tslang/include/TypeScript/Pass/AsyncTargetWidthPass.h`, `tslang/lib/TypeScript/AsyncTargetWidthPass.cpp` | New MLIR pass that fixes both defects. | 2, 3 |
| `tslang/lib/TypeScript/CMakeLists.txt` | Add the source. | 2 |
| `tslang/tslang/transform.cpp` (~163-178) | Schedule the pass. | 2, 3 |

---

### Task 1: Failing checks

**Files:**
- Modify: `tslang/test/check-x86-run.sh`

**Interfaces:**
- Produces: async cases in `check-x86-run.sh`, which Tasks 2 and 3 turn green.

- [ ] **Step 1: Capture the baselines before any compiler change.**
  - Copy the current Release `tslang.exe` to the SDD workspace as `baseline/tslang-base.exe`. It is built from main's sources.
  - Run `tslang/test/compare-ir.sh` for `x86_64-pc-windows-msvc` into `baseline/x64`.
  - Run `probe-corpus.sh` into `baseline/probe`.

- [ ] **Step 2: Add the cases.** In `check-x86-run.sh`, add a loop over the corpus files `00async_await`, `00async_gc_threading`, `00async_result_types`, `00owned_async`, `00for_await` and `00for_await_yield` (from `tslang/test/tester/tests`). Build each at `-mtriple=i686-pc-windows-msvc --opt --no-default-lib --entry-point -mm=gc`, check that the PE machine is I386, and run it with a 30 s timeout; it must exit 0.
  - Use `--entry-point`: the suite builds with it, and these files define `main` without `export`. Check how the existing cases pass flags and follow them.
  - Add the minimal repro above as `tslang/test/x86/await_order.ts`, with its exact expected output. First find out what x64 prints: an `await` inside a non-async `main` may print "after await" before "in g", and that order is the current semantics. Compare against x64 rather than assuming.

- [ ] **Step 3: IR check.** Emit `--emit=llvm` for the repro at i686, gc. Assert that the frame allocator is declared with pointer-width parameters: `declare ptr @GC_memalign(i32, i32)`, and not `(i64, i64)`. Also emit at `-mm=none` and assert `@aligned_alloc(i32, i32)`.

- [ ] **Step 4: Run it.** Every new case FAILs, and every existing case still passes.

- [ ] **Step 5: Commit.** The suite is unaffected, because `check-x86-run.sh` is hand-run.

---

### Task 2: Pointer-width frame allocator

**Files:**
- Create: `tslang/include/TypeScript/Pass/AsyncTargetWidthPass.h`, `tslang/lib/TypeScript/AsyncTargetWidthPass.cpp`
- Modify: `tslang/lib/TypeScript/CMakeLists.txt`, `tslang/tslang/transform.cpp`

**Interfaces:**
- Produces: `std::unique_ptr<mlir::Pass> mlir::typescript::createAsyncTargetWidthPass(unsigned pointerBits)`, a module pass. Task 3 extends the same pass.

- [ ] **Step 1: The pass.** Follow the existing tslang MLIR passes for the declaration and registration style: grep `createGCPass` and its header. Its job:

```cpp
// Upstream ConvertAsyncToLLVM assumes 64-bit sizes: it allocates coroutine frames with
// aligned_alloc(i64, i64) whatever the target. aligned_alloc takes size_t, which is pointer
// width, so on a 32-bit target the callee reads (alignment, 0) and the frame overflows its
// block. Retype the declaration to the pointer width and truncate the arguments at each call.
```

  - Find `LLVM::LLVMFuncOp` named `aligned_alloc`. If none exists, do nothing.
  - If its parameters are already `i<pointerBits>`, do nothing.
  - If they are `(i64, i64) -> ptr`, change the function type to `(iP, iP) -> ptr`. For every `LLVM::CallOp` whose callee is `aligned_alloc`, insert `LLVM::TruncOp` on both operands. The values are coroutine frame sizes and alignments, which fit in 32 bits.
  - Any other shape is a hard error (`emitError` plus `signalPassFailure`).

- [ ] **Step 2: Schedule it** in `transform.cpp`, directly after `createConvertAsyncToLLVMPass()`, under the same `#ifdef ENABLE_ASYNC`, and only when `compileOptions.sizeBits() < 64`. That is before `GCPass`, so the rename to `GC_memalign` keeps the corrected signature.

- [ ] **Step 3: Verify.**
  - The Task 1 IR check passes for gc and none.
  - `00async_await`, `00async_gc_threading`, `00async_result_types`, `00owned_async` and the repro now run at i686. If one still fails, diagnose it before moving on: compare with x64 and find the crash site. Report what you find; don't widen the pass to cover it without understanding it.
  - `00for_await` still fails to compile; that is Task 3.
  - `compare-ir.sh` x64: noise only.
  - `ctest -j 16 -C Release`: all green.

- [ ] **Step 4: Commit.**

---

### Task 3: Index values into `i64` runtime parameters

**Files:**
- Modify: `tslang/lib/TypeScript/AsyncTargetWidthPass.cpp`, `tslang/tslang/transform.cpp`

**Interfaces:**
- Consumes: Task 2's pass.
- Produces: the same pass gains a second phase. Alternatively, it becomes a second pass factory, `createAsyncIndexCastPass(unsigned pointerBits)`, if it has to run at a different point in the pipeline; decide from where the casts first exist.

- [ ] **Step 1: Locate the casts.** Emit `--emit=mlir-llvm` for `00for_await` at i686. Today the output holds `unrealized_conversion_cast %x : i32 to index` followed by `… : index to i64`, feeding `llvm.call @mlirAsyncRuntimeCreateGroup(%7) : (i64)`.
  - Find which pass leaves this chain. The async pass converts `index` to `i64`, and LowerToLLVM later converts the `index` producer to `i32`. The fix has to run after both, so after `createLowerToLLVMPass` and before the LLVM IR translation. `GCPass` runs there too.
  - Also look for the reverse direction, `i64 → index → i32`. For example, `mlirAsyncRuntimGetNumWorkerThreads` returns `index` upstream, and `async.runtime.add_to_group` returns an index.

- [ ] **Step 2: Fold the chains.** Walk `UnrealizedConversionCastOp`s whose single operand is itself the result of an `UnrealizedConversionCastOp`, where the chain is `iA → index → iB` and both A and B are integer widths.
  - Replace the outer cast with `LLVM::SExtOp` when A < B, `LLVM::TruncOp` when A > B, or the original value when A == B.
  - Erase the inner cast if it has no uses left.
  - Use sign extension because `index` is signed in MLIR's arith semantics. The values involved (group sizes, counts) are non-negative, so sign and zero extension agree.
  - Any other `unrealized_conversion_cast` is left alone. LLVM translation will still report it, and it is not in scope.
  - Put a comment at the fold explaining why the chain exists: two type converters with different index widths meet here.

- [ ] **Step 3: Verify.**
  - `00for_await` and `00for_await_yield` compile and run at i686 under gc.
  - The full `check-x86-run.sh` passes, including Task 1's cases.
  - `check-x86-eh.sh` and `check-x86-libs.sh` still pass.
  - `compare-ir.sh` x64: noise only.
  - `ctest -j 16 -C Release`: all green.
  - `probe-corpus.sh`, then `--compare` against `baseline/probe`: no regressions. At x86 gc, the six async files are newly passing. Also report wasm32: run `--emit=llvm` for the six async files with `-mtriple=wasm32-unknown-unknown`, before and after, and report the diff and any compile errors (wasm async may not be supported at all; just record it).

- [ ] **Step 4: Commit.**

---

## Phase 4b gate

- [ ] The six async corpus files and `await_order.ts` run correctly as 32-bit exes under gc.
- [ ] The frame allocator is declared with pointer-width parameters at i686, under gc and none.
- [ ] x64 IR is unchanged apart from known noise, and ctest is all green.
- [ ] `probe-corpus.sh --compare`: no regressions at either arch.

## Rulings made while writing this plan

- **A repair pass, not a vendored copy of `AsyncToLLVM.cpp`.** The frame allocator width is hard-coded to `i64` upstream, whatever `LowerToLLVMOptions` says. So a copy would need the same edit, and it would also mean carrying about 1000 lines of upstream code. The repair pass is small and states each assumption it relies on.
- **The runtime ABI stays `int64_t`.** `mlirAsyncRuntimeCreateGroup(int64_t)` and similar are correct at every arch; phase 1's spec says so. Only the `index` bridge into them is fixed.
- **Async under rc and none stays out of scope.** It fails to link at x64 as well.
