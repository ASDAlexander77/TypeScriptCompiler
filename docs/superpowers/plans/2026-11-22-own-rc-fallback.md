# `-mm=own` counted fallback Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Under `-mm=own --own-fallback=count`, a value whose ownership inference fails (tier 1)
keeps rc's retains and releases and is counted at run time. Every own release honors that count,
always.

**Architecture:** Own's header word gains an *extra owners* field beside the weak count. Every own
release tests `word == 0` first, and decrements the extra owners when there are any. In
`OwnershipInferencePass`, the repair after a failed dry run (today `makeStringCopies`) marks each
refused non-copyable value counted. The real run exempts those values and keeps all their rc ops,
and the lowering turns their retains into extra-owner increments. Own's share routine sets the shared
bit on counted blocks so their counts go atomic.

**Tech Stack:** C++17, MLIR (`ts` dialect), LLVM lowering, CMake/ctest, test-runner, TypeScript test files.

**Spec:** `docs/superpowers/specs/2026-11-22-own-rc-fallback-design.md` (read it first; section
numbers below refer to it).

## Global Constraints

- Releases honor the count **always under own**, with or without the flag (spec "Decisions taken").
- Without `--own-fallback=count`, own accepts and refuses exactly what it does today, and the IR before lowering is unchanged, with two exceptions: a field taken out of a call's own result is a `ts.TakeOutOfField` in every own build, since a callee built with the flag can return a counted object (Task 4 review ruling, spec section 4); and an `@atomic` class's `new` and field stores emit `ts.MarkShared` under own, its instances no longer stack promoted (Task 5, spec section 6).
- Flag: `--own-fallback=error|count`, default `error`.
- The unit counted is a refused value with **all** its rc ops (spec 3). A site that cannot leave them whole stays an error.
- Values own copies today (strings, `any`, `@copies` arrays) stay copied (spec 3).
- A counted object may only live *longer* than inference assumes, never shorter.
- Overflow of either count makes the block immortal (spec 1).
- Every kept count is a warning: `-mm=own counts this object here (--own-fallback=count); a cycle through it is never freed`, with the refused error and its notes as notes (spec 5).
- Threads: own's share routine sets the shared bit on blocks with extra owners; slow-path inc/dec are atomic on a shared block (spec 6).
- Work in the worktree `I:\TypeScriptCompiler-wt-fallback`, branch `own-rc-fallback` (from `origin/main` 09de814f). Never touch `I:\TypeScriptCompiler` (another session's checkout), never switch branches, never push until the batch is done. One PR for the batch. Its `3rdParty\{llvm,gc,gcdll,gc-8.2.12,libatomic_ops-7.10.0}` and `3rdParty\llvm-project\third-party` are junctions: never `rm -rf` or `git clean` anything under `3rdParty`.
- Build (from the worktree): `cmake --build __build/tslang/windows-msbuild-2026-release --config Release --target tslang -- /m:4`; tests: `ctest -C Release -R <regex> -j 8` from `I:\TypeScriptCompiler-wt-fallback\__build\tslang\windows-msbuild-2026-release`. After editing `TypeScriptOps.td`, build with `/m:3`.
- The release change alters the default library's own archives: rebuild the DefaultLib own build (clean clone, lib env vars cleared, per the DefaultLib build notes) before running its tests.

## Review Focus

1. A counted object held in an array that default-library code releases (`pop`, `splice`, `length = 0`, the array's release): must decrement, not free. Test in Task 3 (`own_fallback_array_release.ts`).
2. A counted object that is also a `WeakRef` target / `@weak` field target: `deref()` stays valid until the last owner, counted or not, lets go. Test in Task 3 (`own_fallback_weak_target.ts`).
3. An exception thrown out of a function holding a kept count: `UnwindReleasePass` copies the kept releases, no leak, no double free. Written in Task 2, run in Task 3 (`own_fallback_unwind.ts`).
4. A counted object pushed into a keeper (made immortal): never freed, no crash. Test in Task 4 (`own_fallback_keeper.ts`).
5. A generator that holds a counted object across a `yield` and is dropped half-way: released once. Written in Task 2, run in Task 3 (`own_fallback_generator.ts`).

---

### Task 1: The flag

**Files:**
- Modify: `tslang/include/TypeScript/DataStructs.h` (`CompileOptions`)
- Modify: `tslang/tslang/tslang.cpp` (beside `verifyOwnership`), `tslang/tslang/opts.cpp` (beside `atomicRC`), `tslang/tslang/jitcache.cpp` (cache key, beside `own-skip-inference`)

**Interfaces:**
- Produces: `enum class OwnFallback { Error, Count };` and `OwnFallback ownFallback = OwnFallback::Error;` in `CompileOptions`, plus `bool ownCountsFallback() const { return memoryModel == MemoryModelOwn && ownFallback == OwnFallback::Count; }`.

- [ ] **Step 1: Add** `cl::opt<OwnFallback> ownFallbackOpt("own-fallback", ...)` with values `error` and `count`, description: "-mm=own: where inference cannot prove a move or a borrow, report an error (error) or keep rc's count for that object, with a warning (count)". Copy it into `compileOptions` in `opts.cpp` and add it to the JIT cache key.
- [ ] **Step 2: Check:** `tslang --own-fallback=bogus x.ts` fails with an option error. `tslang -mm=own --own-fallback=count --emit=jit tslang/test/tester/own/own_err_field_read_overwritten.ts` still reports its error.
- [ ] **Step 3: Commit** `own: --own-fallback=error|count option`.

### Task 2: Inference keeps the count (no lowering yet), and the go/no-go sweep

Inference errors surface at `--emit=mlir-affine`, before the lowering, so this task measures the gain
before anything changes the release or the default library.

**Files:**
- Modify: `tslang/lib/TypeScript/OwnershipInferencePass.h` (state: `countedValues`, `dryRunDiagnostics`)
- Modify: `tslang/lib/TypeScript/OwnershipInferencePass.cpp` (`runOnFunction`, `reporting`, `makeStringCopies`, `decideRetains`)
- Modify: `tslang/lib/TypeScript/OwnershipInferenceMoves.cpp` (`checkValueMoves`: skip counted values)
- Modify: `tslang/lib/TypeScript/OwnershipInferenceDiagnostics.cpp` (record instead of emit in the dry run; the warning)
- Modify: `tslang/include/TypeScript/Defines.h` (`OWN_COUNTED_ATTR_NAME "__own_counted"`)
- Create: the test programs below, in `tslang/test/tester/own/`
- Create: `tslang/test/tester/own-fallback.cmake`
- Create: `scripts/own_corpus_sweep.sh`, from the spec's sweep. Every compile passes `-o` into a scratch folder: `--emit=llvm` without `-o` writes `<source>.ll` beside the test and overwrites the tracked `.ll` files in `tslang/test/tester/tests`. It runs gc, own, and own with `--own-fallback=count`, each at a chosen `--emit` (`mlir-affine` here, `llvm` later), and prints the counts and each refused file's first error.

**Interfaces:**
- Consumes: `CompileOptions::ownCountsFallback()` (Task 1).
- Produces:
  - `llvm::DenseSet<mlir::Value> countedValues` on the pass.
  - `struct RecordedDiagnostic { mlir::Location loc; std::string message; llvm::SmallVector<std::pair<mlir::Location, std::string>> notes; };` and `llvm::DenseMap<mlir::Value, RecordedDiagnostic> dryRunDiagnostics`.
  - The attribute `__own_counted` on every kept `ts.Retain` / `ts.RetainSlot`.
  - The warning text exactly: `-mm=own counts this object here (--own-fallback=count); a cycle through it is never freed`.
  - `own-fallback.cmake` with `MODE` = `IR` (compile `--emit=mlir-affine`; assert exit 0, the warning, and at least one `__own_counted`), `RUN` (Task 3), or `EXPECT_ERROR` (assert `error:` with the flag).

- [ ] **Step 1: Write the test programs.** Each prints its result and `done.`. Each uses runtime values (`"a" + i`, `new Node(i)` in a loop of 1000), so a double free or a use after free shows once it runs (Task 3). Each is refused under own today:
  - `own_fallback_second_owner.ts`: `h.node = n; g.node = n;` then both read.
  - `own_fallback_move_overwrite.ts`: `h.node = n; h.node = other; g.node = n; print(g.node.v)`. This is the case where keeping only the later retain would free too early.
  - `own_fallback_fresh_second_owner.ts`: `const n = new Node(i); h.node = n; g.node = n;` (born at rc count 0: the first owner's retain stays erased).
  - `own_fallback_call_second_owner.ts`: `const n = mk(i); h.node = n; g.node = n;` (arrives owning a count: all retains kept).
  - `own_fallback_swap.ts`: `let t = a[i]; a[i] = a[j]; a[j] = t` over `Node[]`.
  - `own_fallback_global_lent.ts`: a global's field lent to a call that replaces the global.
  - `own_fallback_unwind.ts`: the second-owner shape inside a function that throws after the second store, caught in `main`, 1000 times.
  - `own_fallback_generator.ts`: a generator storing a counted node in two holders across a `yield`, iterated half-way and dropped.
  - `own_fallback_result_borrow_stays_error.ts`: `g.node = f(h)` with `f` returning `h.node` (tier 2). Registered `EXPECT_ERROR`.
- [ ] **Step 2: Register** each as `test-own-fallback-ir-<name>` (`MODE=IR`, or `EXPECT_ERROR` for the last one). Run `ctest -C Release -R own-fallback-ir -j 8`. Expected: all FAIL except the `EXPECT_ERROR` one.
- [ ] **Step 3: Record in the dry run.** Where `reporting()` returns false in the dry run, the diagnostic sites store their error and notes in `dryRunDiagnostics` under the value they concern. A site that reports on a pair (use, move) records under the moved value.
- [ ] **Step 4: Mark in the repair.** With `ownCountsFallback()`, in the repair branch of `runOnFunction` after a failed dry run:
  - Every retain `makeStringCopies` walks whose value is not `isCopyableValueAt(...)`, and every retain `copyForRetain` refuses, puts its value in `countedValues`.
  - Each counted value keeps all its rc retains, marked `__own_counted`, **except** the first owner's retain of a value rc creates at count 0 in this function (spec 3, "the word is rc's count minus one").
  - `callCopies` of a non-copyable value takes the handle branch (retain marked `__own_counted`, and a release after the call unless the callee keeps it).
- [ ] **Step 5: Exempt in the real run.** `checkValueMoves` and the borrow-store checks skip values in `countedValues`. `decideRetains` never erases a marked retain, and no move erases a release of a counted value. For each counted value, emit the warning at its recorded error's location, with the recorded error and notes as notes.
- [ ] **Step 6: Run** `ctest -C Release -R own-fallback-ir -j 8`. Expected: PASS. Run the own set (`-R "own"`). Expected: unchanged; every `own_err_*` test still refused without the flag.
- [ ] **Step 7: Sweep.** Run `bash scripts/own_corpus_sweep.sh mlir-affine`. Record own without the flag (expect 863, or the current `main` figure) and own with it, and the remaining first errors.
- [ ] **Step 8: Commit** `own: --own-fallback=count marks the values inference refuses (inference only)`.
- [ ] **Step 9: STOP for go/no-go.** Report the sweep figures to the user and ask whether to go on to Task 3, which changes the release ABI and needs the default library rebuilt. Do not start Task 3 without their yes.

### Task 3: The header word, the release under own, and running counted programs

**Files:**
- Modify: `tslang/include/TypeScript/Defines.h` (beside `HEAP_BLOCK_IMMORTAL`, the own header comment)
- Modify: `tslang/include/TypeScript/LowerToLLVM/OwnershipRoutineLogic.h` (`emitIfLastReference`, `emitIsMortal` callers, `weakCountMask`, `emitWeakIncRef`, `emitFreeOrBury`; new helpers)
- Modify: `tslang/lib/TypeScript/LowerToLLVM.cpp` (`RetainOpLowering`, `RetainSlotOpLowering`)
- Modify: the runtime's C/C++ element-release paths under own, found by `grep -rn "IMMORTAL\|header" tslang/lib/TypeScriptRuntime` (list them in the commit message)
- Modify: `tslang/test/fuzz/mm_fuzz.py` (`--own-fallback` adds the flag to the own leg; `leaks` accepts files)
- Create: `tslang/test/tester/own-release-shape.cmake`, `tslang/test/tester/own/own_fallback_array_release.ts`, `tslang/test/tester/own/own_fallback_weak_target.ts`

**Interfaces:**
- Consumes: `__own_counted` (Task 2), `own-fallback.cmake` (Task 2).
- Produces, in `OwnershipRoutineLogic`:
  - `llvm::APInt extraOwnersMask()`, `llvm::APInt extraOwnerOne()`. The extra-owner field is bits 31-61 on a 64-bit word and 15-29 on a 32-bit one, from `compileOptions.sizeBits()`. `weakCountMask()` narrows to bits 0-30 / 0-14.
  - `void emitExtraIncRef(mlir::Value payloadPtr)`: no-op on immortal. Atomic add when the shared bit is set, plain add otherwise. Overflow makes the block immortal (`emitMarkImmortal`).
  - `mlir::Value emitOwnReleaseIsLast(mlir::Value payloadPtr)`: returns i1 "destroy now". It decrements extra owners when there are any (atomic if shared). A shared block with no extra owners is the last owner.
  - `RetainOpLowering` / `RetainSlotOpLowering` lower a `__own_counted` retain to `emitExtraIncRef` (for the slot form, of the value it holds). An unmarked retain keeps today's error.

- [ ] **Step 1: Write the IR-shape test** `own-release-shape.cmake`. Compile `own_fallback_array_release.ts` with `--emit=llvm -mm=own --no-default-lib`, and find the release routine of class `Node`. Assert that its first `icmp` on the loaded header word compares with `0` (regex `icmp eq i64 %[0-9a-z.]+, 0`). Register it as `test-own-release-shape` with `-P`, as `own-string-copy-block.cmake` is.
- [ ] **Step 2: Write `own_fallback_array_release.ts`** (no fallback in it). `main` pushes 1000 `new Node(i)`, `pop`s 10, `splice(0, 10)`, sets `a.length = 0`, refills, and prints the sum and `done.`. Register it JIT and exe for `gc rc own`. Then write `own_fallback_weak_target.ts`: a `Node` with `@weak parent` and a `WeakRef<Node>` to a counted child (second owner, built with the flag). `deref()` before and after the child's last owner drops it prints `alive`, `gone`, then `done.`.
- [ ] **Step 3: Register the run tests.** Each Task 2 program except the `EXPECT_ERROR` one gets `test-own-fallback-run-<name>` (`MODE=RUN`: `--emit=jit -mm=own --own-fallback=count`, exit 0, output equal to `-mm=gc`, the warning present). `own_fallback_move_overwrite.ts` and `own_fallback_unwind.ts` also get an exe build. Run `ctest -C Release -R "own-release-shape|own-fallback-run|own_fallback_(array_release|weak_target)" -j 8`. Expected: the shape test and the run tests FAIL ("ownership inference left a retain behind" for counted retains); the array test PASSes.
- [ ] **Step 4: Implement spec 1-2.** The new constants go in `Defines.h`. `emitIfLastReference` under own and not `counted` follows spec 2 through `emitOwnReleaseIsLast`. `emitFreeOrBury` is reached only when extra owners are 0. The weak helpers use the narrowed mask. The runtime element-release paths do the same test. The retain lowering is as in Interfaces.
- [ ] **Step 5: Run** the tests of Step 3. Expected: PASS. Then run the whole own set: `ctest -C Release -R "own|-own-" -j 8`. Expected: the same count as on `origin/main`.
- [ ] **Step 6: Leak and heap gate.**
  - `python tslang/test/fuzz/mm_fuzz.py leaks --own-fallback tslang/test/tester/own/own_fallback_*.ts`. Expected: no peak growth beyond rc's. A kept retain too many shows here as growth; output equal to gc does not show it.
  - One run of the exe tests under the quarantine heap (the guard-page recipe), or ASan on WSL. Expected: no fault.
- [ ] **Step 7: Rebuild the DefaultLib own archives and run its own tests**, per the DefaultLib build notes (clean clone, lib env vars cleared, archive timestamps checked). Expected: as before (2 skips by design).
- [ ] **Step 8: Time the ray tracer** under own, ahead of time, from `tslang/benchmarks/raytrace`: best of 5, before (`origin/main` binary) and after. Expected: within ±3%. Record both numbers in the commit message.
- [ ] **Step 9: Commit** `own: the release tests the header word against 0 and honors extra owners`.

### Task 4: Report sites that commit erasures, `TakeOutOfField`, stack promotion, keepers

**Files:**
- Modify: `tslang/lib/TypeScript/OwnershipInferenceClosures.cpp` (the array-view report, `fieldBorrowRetains`), every other site that adds to `fieldBorrowRetains` or `captureRetains` on a report (find with `grep -n "fieldBorrowRetains.insert\|captureRetains.insert" tslang/lib/TypeScript/OwnershipInference*.cpp`)
- Modify: `tslang/lib/TypeScript/LowerToLLVM.cpp` (`TakeOutOfFieldOpLowering`), `tslang/include/TypeScript/LowerToLLVM/OwnershipRoutineLogic.h` (`emitTakeOutOfField`)
- Modify: `tslang/lib/TypeScript/StackPromotionPass.cpp`
- Create: `tslang/test/tester/own/own_fallback_view_stays_error.ts`, `own_fallback_take_out_of_field.ts`, `own_fallback_not_promoted.ts`, `own_fallback_keeper.ts`

**Interfaces:**
- Consumes: `countedValues`, `__own_counted` (Task 2); `emitExtraIncRef`, `extraOwnersMask` (Task 3).
- Produces: nothing new for later tasks.

- [ ] **Step 1: Write the tests.**
  - `own_fallback_view_stays_error.ts`: the array-view shape from `own_err_*view*` tests. With the flag it is still an `error:` (`own-fallback.cmake`, `MODE=EXPECT_ERROR`).
  - `own_fallback_take_out_of_field.ts`: a counted holder (second owner) whose other owner is released while a field is taken out of it (the own list-pop shape `this.head = this.head.next`, with the list also held a second time). The remaining owner still reads the field.
  - `own_fallback_not_promoted.ts`: a local `new Node` that falls back. `--emit=mlir-affine` shows its `ts.New` with `stackAlloc = false` (a check in `own-fallback.cmake`).
  - `own_fallback_keeper.ts`: a counted closure's object pushed into a module-level keeper. It runs, prints, `done.`.
  - Each run test is also added to the Task 3 leak gate (`mm_fuzz.py leaks --own-fallback`).
- [ ] **Step 2: Run them to see them fail.** Expected: the view test compiles (wrong), and take-out reads null or crashes.
- [ ] **Step 3: Audit the report sites.** Each one that commits erasures for the value it reports either leaves the value's ops whole when the flag is on (then it is tier 1) or does not add the value to `countedValues` (it stays an error). Write the list and the choice for each in the commit message.
- [ ] **Step 4: `TakeOutOfField`.** It empties the field only when the object's extra owners are 0. Otherwise it keeps the field and `emitExtraIncRef`s what the field holds. Immortal is unchanged.
- [ ] **Step 5: Stack promotion.** A value with a `__own_counted` retain, or stored into one, counts as escaping.
- [ ] **Step 6: Run** the four tests and the own set. Expected: PASS.
- [ ] **Step 7: Commit** `own fallback: what stays an error, take-out and stack promotion of counted objects`.

### Task 5: Threads

**Files:**
- Modify: `tslang/include/TypeScript/LowerToLLVM/OwnershipRoutineLogic.h` (`getOrCreateShareRoutine` and its walk)
- Create: `tslang/test/tester/own/own_fallback_thread.ts`
- Modify: `tslang/test/tester/CMakeLists.txt`

**Interfaces:**
- Consumes: `extraOwnersMask`, `emitExtraIncRef`, `emitOwnReleaseIsLast` (Task 3), the shared bit (`Defines.h`).
- Produces: under own, `getOrCreateShareRoutine` is non-empty for types that own heap memory. As rc's does, its walk sets the shared bit on every mortal block it reaches, and stops at an immortal block or one already shared (spec 6: a one-owner child of a counted block is reachable from both threads).

- [ ] **Step 1: Write `own_fallback_thread.ts`.** A counted `Node` (second owner in a holder kept on the main thread) holds a one-owner child. The `Node` is captured by a `Thread.spawn` closure. Both threads read the node and its child 100000 times; the spawned thread takes a fallback count on the child. Both drop their references; join; `done.`. Run it under `own-fallback.cmake` and as an exe, 20 runs in a loop in the test. Expected: output equal to gc, exit 0 every run.
- [ ] **Step 2: Run it to see it fail** (or flake). Expected: a crash or mismatch in some runs, or the share routine absent under own (check `--emit=llvm` for `tsshr_`).
- [ ] **Step 3: Implement** the share routine under own as in Interfaces. `emitExtraIncRef` and `emitOwnReleaseIsLast` already take the atomic path on a shared block, and treat a shared block with no extra owners as the last owner (Task 3). Check they do.
- [ ] **Step 4: Run** it and the workers tests (`ctest -C Release -R "thread|worker|atomics" -j 4`). Time `test-jit-own-*thread*` before and after. Expected: PASS, and timing within noise.
- [ ] **Step 5: Commit** `own: the share routine marks counted blocks shared`.

### Task 6: Measurement and documentation

**Files:**
- Modify: `docs/own-memory-model.md` (a section "Counted fallback" after "Shared ownership", and the warning in "Reading the errors"), `docs/own-readiness.md` (a measured line)

- [ ] **Step 1: Sweep.** Run `bash scripts/own_corpus_sweep.sh llvm` (script from Task 2). Expected: own without the flag accepts 863 (as on 2026-11-22, or the current `main` figure); with the flag, more. Record both, plus the top refusals that remain.
- [ ] **Step 2: Fuzz.** Run `python tslang/test/fuzz/mm_fuzz.py --jobs 4 run --count 400 --seed 1122000 --own-fallback`, then `leaks` against rc. Expected: no mismatch, no crash, no peak growth beyond rc's. Repeat on Linux (WSL, `-j 4`).
- [ ] **Step 3: Real code.** Build `native_ts_graphics` at the commit before its 55-file own port, under own with the flag, and run its 525 tests. Record pass count and warnings. (Ask the user for the checkout path if it is not known.)
- [ ] **Step 4: Docs.** In `own-memory-model.md`, cover what the flag does, the warning, cycles leak, and what is still an error (tiers 2-3). In `own-readiness.md`, add a dated line with the sweep and fuzz figures.
- [ ] **Step 5: Commit** `own fallback: corpus sweep, fuzzer mode, docs`. Push, open the PR, follow it to green.
