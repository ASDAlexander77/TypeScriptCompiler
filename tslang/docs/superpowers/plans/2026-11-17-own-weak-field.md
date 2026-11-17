# `@weak` fields under `-mm=own` Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** `@weak parent: Node | null` on a class field: read and written like any field. Under own it is a `WeakRef` whose every read is checked and stops the program on a freed object. Under gc and none it is an ordinary field.

**Architecture:** MLIRGen records `@weak` fields on the class (declared type kept for the program, storage type `!ts.weak_ref<C>` under own, the same way redeclared fields keep a view type: `ClassInfo::redeclaredFieldTypes`). Under own a read becomes `ts.WeakRefDeref` marked `checked` plus a cast to the declared type, a write becomes `ts.WeakRefNew` plus a store, and an identity comparison reads the raw pointer through a new op `ts.WeakRefPointer`. Everything else (count, tombstone, handle retain/release, `"weak-target"` stack refusal, thread refusal) is spec 84's code, reached because the ops are the same. Inference adds two rules: a fresh value written to a `@weak` field, and a `@weak` read given to a call that may drop.

**Tech Stack:** C++17, MLIR (TableGen ops), LLVM lowering, the tslang test-runner and CTest.

**Spec:** `tslang/docs/superpowers/specs/2026-11-17-own-weak-field-design.md` (builds on `2026-11-16-own-weakref-design.md`, implemented).

Worktree `I:\TypeScriptCompiler\.claude\worktrees\own-items-60`, branch `own-weak-field`. Build: `cmake --build __build/tslang/windows-msbuild-2026-release --config Release --parallel 8`; tests: `ctest -C Release -R <regex> -j 8` from that build dir. JIT probe: `bin/tslang.exe -mm=<m> --emit=jit file.ts`.

## Global Constraints

- Decorator name `weak` (`WEAK_DECORATOR "weak"` in `include/TypeScript/Defines.h`, beside `ATOMIC_DECORATOR`).
- Allowed: instance fields of a class, typed `C`, `C | null`, `C | undefined`, or optional `p?: C`, `C` a class (not `@value`). Generic `@weak p: T` is checked per instance.
- Elsewhere: `'@weak' applies only to a field of a class`. Wrong type: `a @weak field needs a class type`.
- rc: `'@weak' is not supported under -mm=rc; use -mm=gc, -mm=own or -mm=none`.
- gc and none: an ordinary field, no new code at run time.
- Own read of a tombstone: to standard error `error: the @weak field 'Node.parent' was read after its object was freed` (class and field named), then exit with a non-zero code; no `finally`, not catchable, no dialog on Windows; works in exe, JIT and wasm.
- `===`/`!==` against a `@weak` read compare the stored pointer and never stop; truthiness, `??`, `?.` are reads and stop.
- Fresh write: `the @weak field 'parent' cannot own a new object; it would be freed at once`.
- Call through a read: `'parent' is a @weak field and 'removeChild' may free objects; the call could free the object it runs on` (suggest `@nodrops` on the method or `@final` on the class).
- Interface view: `'Node' cannot be seen as 'HasParent': its field 'parent' is @weak`.
- Commit messages end with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`; commits are GPG-signed (never `--no-gpg-sign`). Error expectation strings in `own_error_cases` contain no `;` and no unbalanced `[` (use `.`).

## Review Focus

1. A node whose parent is itself (`n.parent = n`), then dropped: the field's release lowers the count before the free tests it, block freed once. → Task 2, `00weak_field` "self".
2. A subclass inheriting a `@weak` field from its base, read through the subclass and through a base-typed variable. → Task 2, `00weak_field` "subclass".
3. Chained reads `leaf.parent.parent.v` and `leaf.parent?.parent?.v`: each link checked. → Task 2, `00weak_field` "chain".
4. A generic `class TreeNode<T> { @weak parent: TreeNode<T> | null }` instantiated twice. → Task 2, `00weak_field` "generic".
5. An optional `@weak up?: Node` never assigned: reads `undefined`, the holder freed without touching any count. → Task 2, `00weak_field` "optional".

---

### Task 1: The decorator, its checks, gc and none

**Files:**
- Modify: `tslang/include/TypeScript/Defines.h` (`WEAK_DECORATOR`).
- Modify: `tslang/include/TypeScript/MLIRLogic/MLIRGenStore.h` (~837, beside `redeclaredFieldTypes`): `llvm::StringMap<mlir::Type> weakFieldTypes;` (field name → declared type) on `ClassInfo`, and `bool isWeakField(StringRef name)` that searches bases too.
- Modify: `tslang/lib/TypeScript/MLIRGenClasses.cpp` (field declaration processing, ~2340-2360 and ~2772): read `@weak` with `MLIRHelper::hasDecorator(classMember, WEAK_DECORATOR)`, check position and type, record it. `MLIRGenImpl.h` / `MLIRGenFunctions.cpp`: `@weak` on a local, parameter (property), static field or interface member is the position error.
- Test: `tslang/test/tester/tests/00weak_field.ts`, `tslang/test/tester/own/own_err_weak_field_position.ts`, `own_err_weak_field_type.ts`, `own_err_weak_field_generic.ts`, `own_err_weak_field_rc.ts`; `tslang/test/tester/CMakeLists.txt`.

**Interfaces:**
- Produces: `WEAK_DECORATOR`; `ClassInfo::weakFieldTypes`, `ClassInfo::isWeakField(StringRef) -> bool`.

- [ ] **Step 1: Write `00weak_field.ts`**: the spec's `Node` (`@weak parent: Node | null = null`, `children`, `add`, `depth`) plus the Review Focus sections, each printing a line and asserting with `if (...) throw`:
  - "tree": root/mid/leaf; `leaf.depth() == 2`; `leaf.parent.parent.v == root.v` (chain, Focus 3) and `leaf.parent?.parent?.v`;
  - "move": `mid.children.pop()` moves `leaf`, `other.add(leaf)`; `leaf.parent === other`;
  - "drop": a subtree dropped; its children's fields never read afterwards; churn `for 1000: new Node(i)` between steps;
  - "self" (Focus 1), "subclass" (`class Leaf extends Node`, Focus 2), "generic" (`TreeNode<number>` and `TreeNode<string>`, Focus 4), "optional" (`@weak up?: Node`, read `=== undefined`, Focus 5);
  - last line `done.`.
  Register it for `gc none` (jit + compile) in a `foreach(weak_field_mm gc none)` beside the weakref block (~5107).
- [ ] **Step 2: Write the errors**: position (`@weak` on a static field, and on a local), type (`@weak names: string[]`, `@weak v: Point` with `@value class Point`, `@weak x: A | B`), generic (`class G<T> { @weak p: T | null = null }` used as `G<string>`), rc (`-mm=rc` add_test like `own_err_weakref_rc`). Add the `name|:line:col: error: <text>` entries to `own_error_cases` from the Global Constraints texts; run once to read the real line:column.
- [ ] **Step 3: Build, `ctest -C Release -R "weak_field"`**: expected FAIL (unknown decorator ignored, so the errors do not fire; gc/none may already pass, which is fine).
- [ ] **Step 4: Implement** the decorator, `weakFieldTypes`, the checks and the rc error. Under own, until Task 2, a `@weak` field is the error `@weak fields under -mm=own are not implemented yet`.
- [ ] **Step 5: `ctest -C Release -R "weak_field|weakref|own_err"`**: expected PASS.
- [ ] **Step 6: Commit** `@weak fields: the decorator, its checks, gc and none`.

### Task 2: Own storage, checked reads, writes, comparisons

**Files:**
- Modify: `tslang/include/TypeScript/TypeScriptOps.td` (~2527-2540): `WeakRefDerefOp` gets optional `UnitAttr:$checked` and `OptionalAttr<StrAttr>:$weakField` (`"Node.parent"`); `WeakRefNewOp` gets `OptionalAttr<StrAttr>:$weakField`; new `TypeScript_WeakRefPointerOp` (`ins TypeScript_WeakRef:$weakRef`, `outs AnyType:$result`), the stored pointer without any check.
- Modify: `tslang/lib/TypeScript/MLIRGenClasses.cpp`: under own the field's storage type in the class's storage is `!ts.weak_ref<C>` (a null pointer is null/undefined).
- Modify: `tslang/lib/TypeScript/MLIRGenImpl.h` (class field `PropertyRefOp` read/assign paths) and `MLIRGenExpressions.cpp` (`===`/`!==` operands, beside `mlirGenWeakRefDeref` ~1993): read = load + `WeakRefDeref{checked, weakField}` + cast to the declared type; write = `WeakRefNew{weakField}` of the value (null stays null) + store; a `@weak` read that is an operand of `===`/`!==` = load + `WeakRefPointer`. Field initializers go through the write path.
- Modify: `tslang/lib/TypeScript/LowerToLLVM.cpp` (`WeakRefDerefOpLowering` ~1509, registration ~10116): when `checked`, a non-null stored pointer whose target is not alive (`emitWeakTargetAliveOrNull` gives null) branches to a block that writes the message to standard error and exits non-zero; `WeakRefPointerOpLowering` returns the pointer (gc never sees these ops: Task 1 keeps gc fields ordinary).
- Test: `00weak_field.ts` registered for own (jit + compile, and `-noopt`); `tslang/test/tester/own/own_weak_field_compare_dead.ts`; `tslang/test/tester/own/own_weak_field_freed_read.ts` with a new `tslang/test/tester/own-weak-field-stop.cmake` (modelled on `own-exe-run.cmake`).

**Interfaces:**
- Consumes: `ClassInfo::isWeakField`, `weakFieldTypes` (Task 1).
- Produces: `WeakRefDerefOp::getChecked()`, `getWeakField()`; `WeakRefNewOp::getWeakField()`; `mlir_ts::WeakRefPointerOp`. Task 3 finds `@weak` reads and writes by these attributes.

- [ ] **Step 1: Write `own_weak_field_compare_dead.ts`**: a child kept, its parent dropped, then `child.parent === p` is `false` for a fresh `p`, `child.parent !== null` is `true`, `child.parent = null` works, then churn and `done.`. Register it under own, gc and none.
- [ ] **Step 2: Write `own_weak_field_freed_read.ts`** (a child kept, parent dropped, `print(child.parent.v)`, then `print("after")`) and `own-weak-field-stop.cmake`: builds with `--emit=exe -mm=own`, runs it, requires a non-zero exit, the stderr text `the @weak field 'Node.parent' was read after its object was freed`, and no `after` in the output. Same source registered under gc and none as ordinary tests (they print the value then `after`; the file ends with `done.`).
- [ ] **Step 3: Register `00weak_field` under own**; build; `ctest -C Release -R "weak_field"`: expected FAIL (Task 1's not-implemented error).
- [ ] **Step 4: Implement** storage, read, write, comparison and the lowering. Remove Task 1's not-implemented error.
- [ ] **Step 5: `ctest -C Release -R "weak_field|weakref"`**: expected PASS. Leak check: a churn of 2000 vs 20000 re-parentings built with `--emit=exe -mm=own`, peak working set flat (within 1 MB).
- [ ] **Step 6: Commit** `@weak fields under own: checked reads, counted writes, raw comparisons`.

### Task 3: Own inference rules and messages

**Files:**
- Modify: `tslang/lib/TypeScript/OwnershipInferencePass.cpp` / `OwnershipInferenceMoves.cpp`: a `WeakRefNewOp` with `weakField` whose operand `isFresh` (and not immortal) is the fresh-write error.
- Modify: `tslang/lib/TypeScript/OwnershipInferencePlaceReads.cpp` (the `WeakRefDerefOp` cases ~291, ~730, ~1330): a `checked` deref result (through casts) used as the receiver or an argument of a call that may drop is the call-through error. "May drop" is what `dropsChain` already decides for a call (no `__own_no_drops`, a virtual call whose overrides are not all known, a call through a function value).
- Modify: `tslang/lib/TypeScript/OwnershipInferenceDiagnostics.cpp` (~16): a `checked` deref is named `the object the @weak field 'parent' refers to`.
- Test: `own_err_weak_field_fresh.ts`, `own_err_weak_field_call_drops.ts` (receiver of a method that drops), `own_err_weak_field_arg_drops.ts` (argument of such a call, through a non-`@final` class whose override drops), `own_err_weak_field_returned.ts`, `own_err_weak_field_stored.ts`, `own_err_weak_field_captured.ts`, `own_err_weak_field_held.ts` (`const p = this.parent; drop(); p.v`), `own_err_weak_field_thread.ts` (a weak-target object captured by `Thread.spawn`); and `own_weak_field_reread.ts` (runtime, own/gc/none: `this.parent.v; drop(); this.parent.v` compiles and runs, `this.parent.depth()` with a `depth` that frees nothing compiles).

**Interfaces:**
- Consumes: the `checked` / `weakField` attributes (Task 2).

- [ ] **Step 1: Write the tests above**; add the error entries (texts from Global Constraints; the returned/stored/captured/held texts follow spec 84's with `the object the @weak field 'parent' refers to`).
- [ ] **Step 2: Build, `ctest -C Release -R "weak_field"`**: expected FAIL on the new error cases (fresh writes compile, the call-through runs).
- [ ] **Step 3: Implement** the two rules and the naming.
- [ ] **Step 4: Full suite** `ctest -C Release -j 8` and the own corpus sweep (`msgs.py`, session scratchpad `corpus`): no new failures, corpus count not lower than 740.
- [ ] **Step 5: Commit** `@weak fields: fresh writes and calls that may free their object are errors`.

### Task 4: Interface views and other modules

**Files:**
- Modify: `tslang/lib/TypeScript/MLIRGenCast.cpp` / `MLIRGenInterfaces.cpp`: converting a class (or subclass) that has a `@weak` field to an interface or object type with a member of that name is the interface-view error. An `any` box cast back to the class is unaffected.
- Modify: `tslang/lib/TypeScript/DeclarationPrinter.cpp` (beside `redeclaredFieldTypes` ~645): the exported class declaration prints `@weak` on the field, so an importing module (and `-shared` export/import) records it again.
- Test: `own_err_weak_field_interface.ts`, `own_err_weak_field_object_type.ts`; `own/export_weak_field.ts` + `own/import_weak_field.ts` (the importer builds a tree, re-parents, reads `depth()`, prints `done.`), registered like `import_own_no_drops` (jit + compile, `-shared`), and an import-side freed read checked by `own-weak-field-stop.cmake` with the two sources.

**Interfaces:**
- Consumes: `ClassInfo::isWeakField` (Task 1), Task 2's ops.

- [ ] **Step 1: Write the tests**; add the error entries.
- [ ] **Step 2: Build, run `ctest -C Release -R "weak_field"`**: expected FAIL (cast accepted; importer reads an ordinary field layout or fails to compile).
- [ ] **Step 3: Implement** the cast refusal and the printed decorator.
- [ ] **Step 4: Validate the batch**: Windows `ctest -C Release -j 8`; Linux (WSL, linux.sh per branch: ninja -j 4, ctest -j 8, `MALLOC_PERTURB_`); `mm_fuzz.py` seed 931000 x500; the default library on all four models from the clean clone `I:\dl_release_check`. Expected: all pass.
- [ ] **Step 5: Commit** `@weak fields: refused through interfaces, carried across modules`.

### Task 5: Documentation

**Files:**
- Modify: `tslang/docs/superpowers/specs/2026-09-24-own-memory-model-design.md` (a section after 92: spec 93, `@weak` fields, the spec's rules and the user decisions), `docs/own-memory-model.md` (a "Parent links" subsection next to `WeakRef`, the `Node` example), `docs/memory-models.md` (one row), `docs/own-readiness.md` (the back-reference item: done, checked at run time; the compile-time proof stays open as an optimisation).

- [ ] **Step 1: Write the four documents**; every message quoted matches the Global Constraints text.
- [ ] **Step 2: Commit** `Docs: @weak fields (own spec 93)`.
