# `WeakRef<T>` under `-mm=own` Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A built-in `WeakRef<T>` that works under own, gc and none: own keeps a weak count in the target's header word, freed targets become tombstones, and `deref()` gives a compile-time borrow.

**Architecture:** A new type `!ts.weak_ref<T>` with ops `ts.WeakRefNew` / `ts.WeakRefDeref`, resolved by name before any declaration. Under own a weak reference is a counted handle like `Shared<T>` (own leaves rc's retains and releases, lowering counts on the target's header). The class release routine checks the count before freeing. Inference treats a deref result as a borrow of an unknown place (TSLANG 84's `anyPlace` chain).

**Tech Stack:** C++17, MLIR (TableGen ops/types), LLVM lowering, the tslang test-runner and CTest.

**Spec:** `tslang/docs/superpowers/specs/2026-11-16-own-weakref-design.md`

Worktree: `I:\TypeScriptCompiler\.claude\worktrees\own-items-60`, branch `own-weakref`. Build: `cmake --build __build/tslang/windows-msbuild-2026-release --config Release --target tslang --parallel 8` (all targets before ctest); tests: `ctest -C Release -R <regex> -j 8` from that build dir. JIT probe: `bin/tslang.exe -mm=<m> --emit=jit file.ts`.

## Global Constraints

- Models: own, gc, none have `WeakRef`; under rc any use is the error `WeakRef is not supported under -mm=rc; use -mm=gc, -mm=own or -mm=none`.
- Under own and none, `T` and the argument must be a class object; else `WeakRef needs a class object`. Under gc any object (`T extends object`) is accepted, as today.
- Own header word of a mortal class object: bits 0..61 weak count, bit 62 dead (tombstone), bit 63 unused, all bits set = immortal (tested first). On a 4-byte word (wasm32, 32-bit targets) the dead bit is bit 30.
- `deref()` returns `T | undefined`: the object, or `undefined` once freed (own: tombstone; gc: link cleared; none: never).
- Releasing a `WeakRef` is no drop: it does not remove `__own_no_drops` and ends no borrow.
- No object grows; a program without `WeakRef` gains one header test per free of a class object, nothing else.
- `WeakRef` resolves to the built-in type even where a class named `WeakRef` is declared (the default library declares one; it is removed in Task 7).
- Commit messages end with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`; commits are GPG-signed (never `--no-gpg-sign`).

## Review Focus

1. A `WeakRef` copied (`b.parent = a.parent`) and both released: the count goes back to zero, no early free of the tombstone, no leak. → Task 2 test `00weakref` step "copies".
2. A `WeakRef` field reassigned after its target died: the old tombstone is freed by that release. → Task 2, `00weakref` "retarget".
3. An object holding a `WeakRef` to itself, freed: its field release takes the count to zero before the free path tests it, so the block is freed once. → Task 2, `00weakref` "self".
4. `WeakRef`s held in an array that is popped, spliced and dropped. → Task 2, `00weakref` "elements".
5. A `WeakRef` to an immortal object (a global's, kept for good): count untouched, `deref()` always gives it. → Task 2, `00weakref` "immortal".

---

### Task 1: The type, the surface, gc and none

**Files:**
- Modify: `tslang/include/TypeScript/TypeScriptTypes.td` (after `TypeScript_Shared`, ~line 133): `TypeScript_WeakRef`, mnemonic `weak_ref`, one parameter `elementType`.
- Modify: `tslang/include/TypeScript/TypeScriptOps.td` (after `TypeScript_SharedCountOp`, ~line 2531): `TypeScript_WeakRefNewOp` (`ins AnyType:$target`, `outs TypeScript_WeakRef:$result`), `TypeScript_WeakRefDerefOp` (`ins TypeScript_WeakRef:$weakRef`, `outs AnyType:$result`).
- Modify: `tslang/lib/TypeScript/MLIRGenTypes.cpp` (`isEmbededTypeWithBuiltins`/`NoBuiltins` ~949/1043, `getEmbeddedTypeWithParam*` ~1207/1314, `getTypeByTypeReference` ~804): `WeakRef` checked before declared generics.
- Modify: `tslang/lib/TypeScript/MLIRGenExpressions.cpp` (`new` ~1666, beside `mlirGenNewShared` ~1851): `mlirGenNewWeakRef`, taken before normal resolution.
- Modify: `tslang/lib/TypeScript/MLIRGenAccessCall.cpp` (beside `s.value` ~350): `w.deref()` → `ts.WeakRefDeref`.
- Modify: `tslang/lib/TypeScript/LowerToLLVM.cpp` (beside Shared lowering ~1385, registration ~9990): type converter (a pointer), `WeakRefNew`, `WeakRefDeref` for gc and none.
- Test: `tslang/test/tester/tests/00weakref_basic.ts`, `tslang/test/tester/own/own_err_weakref_rc.ts`, `own_err_weakref_not_class.ts`; `tslang/test/tester/CMakeLists.txt`.

**Interfaces:**
- Produces: `mlir_ts::WeakRefType` (`getElementType()`), `mlir_ts::WeakRefNewOp`, `mlir_ts::WeakRefDerefOp`; `MLIRTypeHelper::isWeakRefType(mlir::Type)` (also true for an optional of one, and a union of one with members owning no heap, mirroring `isSharedHandleType`).

- [ ] **Step 1: Write `00weakref_basic.ts`**: a `Node` with `parent: WeakRef<Node> | null`; `add(c)` sets `c.parent = new WeakRef(this)`; `depth()` via `this.parent?.deref()`; asserts `leaf.depth() == 2`, `(new WeakRef(root)).deref() === root`; prints `done.`. Register it for `gc none` (jit + compile) beside the `shared_models` block (~line 5079).
- [ ] **Step 2: Write the errors**: `own_err_weakref_rc.ts` (any `new WeakRef(new C())`) expecting `WeakRef is not supported under -mm=rc` under `-mm=rc`; `own_err_weakref_not_class.ts` (`new WeakRef([1, 2])`) expecting `WeakRef needs a class object` under own. Register in the own error-cases list (rc case needs its own `add_test` with `-mm=rc`).
- [ ] **Step 3: Build and run `ctest -C Release -R "weakref"`**: expected FAIL (unknown type / GC_ call refused).
- [ ] **Step 4: Implement the type, ops, MLIRGen and the gc/none lowering.** gc: allocate an atomic (unscanned) gc block of one word, store `~ptr`, call `GC_general_register_disappearing_link(block, target)`; deref loads the word, 0 → `undefined`, else `~word`. none: the value is the target pointer; deref returns it. `deref()`'s result type is what a method declared `deref(): T | undefined` gets. Under own, until Task 2, emit the error `WeakRef under -mm=own is not implemented yet`.
- [ ] **Step 5: Run `ctest -C Release -R "weakref|shared"`**: expected PASS; also the default library's gc `WeakMap`/`WeakSet` tests still pass (they now use the built-in type): JIT-run `I:\dl_release_check\tests\weakmap_weakset.ts` and `weakref_basic.ts` under gc.
- [ ] **Step 6: Commit** `WeakRef<T>: built-in type; gc and none`.

### Task 2: Own run time: counts, tombstones, handles

**Files:**
- Modify: `tslang/include/TypeScript/Defines.h` (~line 583): `HEAP_BLOCK_DEAD_BIT` comment and value (word's bit `bits - 2`), `HEAP_BLOCK_WEAK_COUNT_MASK`.
- Modify: `tslang/include/TypeScript/LowerToLLVM/OwnershipRoutineLogic.h`: `buildBody` class branch (~1707-1718) — after `releaseFields`, if `word & WEAK_COUNT_MASK` then `word |= DEAD` else `emitFreeBlock`; new `emitWeakIncRef(target)` / `emitWeakDecRef(target)` (skip immortal; dec frees the block when the result is exactly `DEAD`); the release routine of a `weak_ref` type calls `emitWeakDecRef`.
- Modify: `tslang/lib/TypeScript/LowerToLLVM.cpp`: own lowering of `WeakRefNew` (`emitWeakIncRef`), `WeakRefDeref` (immortal → target; dead → `undefined`; else target), Retain/Release of a `weak_ref` (allowed under own like Shared handles, ~510/746/834).
- Modify: `tslang/lib/TypeScript/OwnershipFacts.h` (`isHandle`/`touchesHandle` ~1041): also `isWeakRefType`; new `inline bool isWeakHandle(mlir::Value)`.
- Modify: `tslang/lib/TypeScript/OwnershipSignaturePass.cpp` (`dropsInBody` ~2136): a release of a weak handle is no drop.
- Modify: `tslang/lib/TypeScript/OwnershipInferencePlaceReads.cpp` / `OwnershipInferencePass.cpp`: `ts.WeakRefDeref`'s result is a borrowing read with `Chain{anyPlace = true}`, no roots; ended as TSLANG 84's chains are (`dropsChain`, `typeMayHold`).
- Test: `tslang/test/tester/tests/00weakref.ts`; CMakeLists (own, gc, none; `-noopt` under own in the debug-own list ~6686; not in `own_test`, whose no-counting check forbids counts).

**Interfaces:**
- Consumes: Task 1's type, ops and `isWeakRefType`.
- Produces: `own_facts::isWeakHandle(mlir::Value) -> bool`; `OwnershipRoutineLogic::emitWeakIncRef(mlir::Value)`, `emitWeakDecRef(mlir::Value)`.

- [ ] **Step 1: Write `00weakref.ts`** with sections, each in a 2000-iteration loop with same-size `new Node` churn between steps:
  - "tree": parents, children; `leaf.depth() == 3`.
  - "dropped": `const w = new WeakRef(sub)` kept while `root.children.pop()` frees `sub`; then `w.deref() === undefined`.
  - "copies": `b.parent = a.parent`; free the parent; both `deref()` undefined; drop both.
  - "retarget": a field's target dies, the field gets `new WeakRef(other)`; `deref() === other`.
  - "self": `n.me = new WeakRef(n)`, `n` dropped.
  - "elements": an array of `WeakRef`s, `pop`, `splice(0, 1)`, the array dropped.
  - "immortal": `const G = new Node(7)` global; `new WeakRef(G).deref() === G` after locals churn.
  - Ends `print("done.")`; enable `00weakref_basic` under own too.
- [ ] **Step 2: Run** `ctest -C Release -R weakref`: FAIL under own (Task 1's not-implemented error).
- [ ] **Step 3: Implement** the files above.
- [ ] **Step 4: Run** `ctest -C Release -R "weakref|shared|linked_list"`: PASS. `--verify-ownership` quiet on `00weakref` under own.
- [ ] **Step 5: Leak and double free**: AOT churn (`--emit=exe --no-default-lib`, 20M iterations of the "dropped" and "copies" sections, expected total as a float) prints `done.` with flat private memory under own; on WSL `~/tsc-size` (build `__build/tslang/ninja/release`) `GLIBC_TUNABLES=glibc.malloc.tcache_count=0 MALLOC_PERTURB_=165 ctest -R "weakref|own"` passes.
- [ ] **Step 6: Commit** `-mm=own: WeakRef counts in the target's header; freed targets are tombstones`.

### Task 3: Own: what a deref result may not do

**Files:**
- Modify: `tslang/lib/TypeScript/OwnershipInferenceDiagnostics.cpp` (`describePlace` and the place-borrow reports): a deref result is described as `the object a WeakRef refers to`.
- Test: `tslang/test/tester/own/own_err_weakref_stored.ts`, `_returned.ts`, `_captured.ts`, `_after_call.ts`, `_after_release.ts`; `own_weakref_borrow.ts` (positive); CMakeLists.

**Interfaces:**
- Consumes: Task 2's borrow chain for `ts.WeakRefDeref`.

- [ ] **Step 1: Write the tests.** Errors (own error-cases list, exact line:col filled from the first run):
  - `_stored`: `h.n = w.deref()!` → `borrows the object a WeakRef refers to and cannot be stored, returned or captured`.
  - `_returned`: `return w.deref()`, same message.
  - `_captured`: `const p = w.deref(); fns.push(() => p!.v)`, same message.
  - `_after_call`: `const p = w.deref(); dropAll(); print(p!.v)` where `dropAll` releases a `Node` field → `borrows the object a WeakRef refers to but is used here after it may be freed`.
  - `_after_release`: `const p = w.deref(); holder.child = null; print(p!.v)`, same message.
  - Positive `own_weakref_borrow.ts`: `const p = this.parent?.deref(); if (p) return p.v + p.depth();` with `@final class Node` (own, gc, none).
- [ ] **Step 2: Run** `ctest -C Release -R weakref`: the messages differ (generic place text) → FAIL.
- [ ] **Step 3: Implement** the description; keep the existing message forms.
- [ ] **Step 4: Run** `ctest -C Release -R "weakref|own-err"`: PASS.
- [ ] **Step 5: Commit** `-mm=own: a WeakRef's object is a borrow`.

### Task 4: No stack promotion of a weak target

**Files:**
- Modify: `tslang/lib/TypeScript/StackPromotionPass.cpp` (`moduleUsesShared` ~85, `decide` ~390): `weakTargets` — class names `T` of every `weak_ref<T>` type in the module; a `ts.New` of such a class or of one deriving from it gets reason `"weak-target"`, in every model.
- Test: `tslang/test/tester/tests/00weakref_frame.ts` (own, gc, none, and `-noopt` own).

- [ ] **Step 1: Write the test**: `function make(l: Holder) { const n = new Node(5); l.w = new WeakRef(n); return n.v; }` called in a loop; after each call `l.w!.deref()` is `undefined` under own, and under gc/none either `undefined` or a node whose `v == 5`; churn between calls; `done.`
- [ ] **Step 2: Run**: FAIL (garbage or crash under own/none when `n` is promoted).
- [ ] **Step 3: Implement** the refusal; check the dump option the pass already has for reasons (`"shared-handle"` shows how) prints `weak-target`.
- [ ] **Step 4: Run** `ctest -C Release -R "weakref|stack-promotion"`: PASS.
- [ ] **Step 5: Commit** `WeakRef: a weak target stays on the heap`.

### Task 5: Own: nothing a WeakRef module holds crosses threads

The design refuses a weak-target object crossing threads. Thread hand-off reaches the compiler only as `Shared.share(this.start)` (`ts.MarkShared`) inside the default library's `ThreadHandle`/worker constructors, where a closure's captures are not visible. So the rule is: under own, in a module whose `weakTargets` is non-empty, a `ts.MarkShared` whose operand's type may hold a weak target (`typeMayHold`, which answers yes for a function's captures) is an error at the spawn's source position: `-mm=own cannot share this value with another thread: it may hold a 'Node', which a WeakRef refers to`.

**Files:**
- Modify: `tslang/lib/TypeScript/OwnershipSignaturePass.cpp` (or inference, where `ts.MarkShared` is visible with the call stack position): the check.
- Test: `tslang/test/tester/own/own_err_weakref_thread.ts` (spawn of a closure capturing a `Node` in a module using `WeakRef<Node>`), `own_weakref_thread_ok.ts` (a module without `WeakRef` still spawns as before; reuse an existing threads test if one covers it).

- [ ] **Step 1: Write the tests**; register the error with its line:col.
- [ ] **Step 2: Run**: the error test FAILs (compiles).
- [ ] **Step 3: Implement**; the position is the user's `Thread.spawn` call (the MarkShared is in an instantiated library generic: report at the outermost location in the op's call-site `CallSiteLoc`/`FusedLoc` chain that is in the user's file).
- [ ] **Step 4: Run** `ctest -C Release -R "weakref|thread|worker"`: PASS.
- [ ] **Step 5: Commit** `-mm=own: a module using WeakRef shares nothing that may hold a weak target`.

### Task 6: Docs and validation, then the PR

**Files:**
- Modify: `tslang/docs/superpowers/specs/2026-09-24-own-memory-model-design.md` (append `## 84. WeakRef<T>`), `docs/own-memory-model.md` (Current limits: back-references), `docs/memory-models.md` (~line 92, "the fourth option" now exists for own/gc/none), `docs/own-readiness.md` (shapes table rows "Parent link" and "`new WeakRef(a)`"; What is left item 1).

- [ ] **Step 1: Write the docs**, including Task 5's coarse thread rule and the cross-module note (objects from an older own compiler's object files are unsupported).
- [ ] **Step 2: Full Windows ctest**: `ctest -C Release -j 8` — all pass.
- [ ] **Step 3: Linux**: apply the branch diff to WSL `~/tsc-size` (its `origin` is stale: fetch commits by sha from `/mnt/i/TypeScriptCompiler`), `ninja -j 4`, run own and weakref tests under `MALLOC_PERTURB_` as in Task 2 — all pass.
- [ ] **Step 4: Corpus**: compile every `tslang/test/tester/tests/*.ts` and `own/*.ts` under own with origin/main's compiler and this branch's; no file newly refused, no crash.
- [ ] **Step 5: Fuzzer**: `python tslang/test/fuzz/mm_fuzz.py --jobs 6 --out <scratch> run --count 500 --seed 918000` — 0 failures.
- [ ] **Step 6: Default library**: build `I:\dl_release_check` with this compiler and run its tests for gc and own (scripts as in the previous round): no new failure; `weakref_basic`, `weakmap_weakset` pass under gc.
- [ ] **Step 7: Commit, push `own-weakref`, open the PR** (`gh auth switch --user ASDAlexander77` first), body: what works under which model, the header layout, the thread rule, the validation numbers.

### Task 7: Default library follow-up (separate repository)

In a clean clone of TypeScriptCompilerDefaultLib (`I:\dl_release_check`, never the user's checkout), branch `weakref-builtin`:

- [ ] **Step 1**: remove `class WeakRef<T extends object>` (`src/generics/lib.generics.ts` ~3061-3081); give `WeakMapEntry` a private `class __GcWeakLink<T extends object>` with the same body, so `WeakMap`/`WeakSet` stay gc only.
- [ ] **Step 2**: build and run the library's tests under gc, own, rc and none: `weakref_basic` passes under gc (built-in), `weakmap_weakset` passes under gc and stays skipped elsewhere.
- [ ] **Step 3**: commit, push, open the PR, linked from Task 6's PR; it merges after it.
