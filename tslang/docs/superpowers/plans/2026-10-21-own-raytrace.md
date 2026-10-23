# `-mm=own` compiles the ray tracer - Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** the ray tracer, with two small source edits, compiles and runs correctly under `-mm=own`, by closing nine compiler gaps as general rules.

**Architecture:** every change is in the own analysis: cross-function facts in
`OwnershipSignaturePass.cpp`, per-function moves and borrows in `OwnershipInferencePass.cpp`,
shared predicates in `OwnershipFacts.h`, and release routines in
`include/TypeScript/LowerToLLVM/OwnershipRoutineLogic.h`. Each task is one PR off `main`, small
and squash-merged, in the order below.

**Tech Stack:** C++17, MLIR/LLVM 22, tslang `ts` dialect, CMake/CTest, MSVC (Windows) and GCC (WSL).

**Spec:** `tslang/docs/superpowers/specs/2026-10-21-own-raytrace-design.md` (approved 2026-10-21).
Background: `tslang/docs/superpowers/specs/2026-09-24-own-memory-model-design.md` (§2 semantics,
§15 signature pass, §23 `Shared<T>`, §24 assignment after a move).

## Global Constraints

- No unsound acceptance: when a rule cannot prove the block outlives every use, keep the error.
- No run-time counting under own outside `Shared<T>`: every positive own test is also in the
  `own_test` list, whose `test-own-no-counting-*` check fails on `__tslang_inc_ref|__tslang_dec_ref`.
- gc, rc and none compile and print exactly what they did; every positive test runs under gc, rc
  and own (and none unless it allocates in a loop).
- Diagnostics keep their existing texts; new rejections reuse an existing text where one fits.
- Build: `cmake --build __build/tslang/windows-msbuild-2026-release --config Release --parallel 8`
  (drop to `--parallel 4` if a `.td` changes). Tests: `cd __build/tslang/windows-msbuild-2026-release && ctest -j 16 -C Release`
  (`-C Release` is required; without it nothing runs). Before a PR the full suite has no new
  failures against `main`.
- Any change to what is released or freed also runs the own tests on Linux (the WSL build):
  the Windows heap hides double frees.
- Teeth for every new acceptance rule: a temporary `getenv("OWN_TEETH")` switch that skips the
  rule's safety condition must make the new positive test crash or print wrong under AOT
  (`test-compile-own-*`), then the switch is removed. Commit before checking out anything.
- Test files go in `tslang/test/tester/own/`; positives join the `own_test` foreach in
  `tslang/test/tester/CMakeLists.txt` (~line 3460) and get a gc/rc(/none) foreach of their own;
  negatives join `own_error_cases` (~line 3624) as `"name|regex"` (no `;` in the regex).
- Each task appends a results section to `2026-09-24-own-memory-model-design.md` (§25, §26, ...).
- Commits are GPG-signed; on `gpg: signing failed: Timeout` run `gpg-connect-agent "scd killscd" /bye`
  and retry. Never `--no-gpg-sign`.

## Review Focus

- An immortal global's field overwritten (`K.w.child = new C()`) must still free the old child,
  and a borrow of `K.w.child` must still end there - Task 1, `own_err_immortal_field_overwritten`.
- A read after a move where the new owner is overwritten on one path only must stay an error -
  Task 5, `own_err_read_after_move_overwritten`.
- A borrowing record stored into a field of a longer-lived object must stay an error - Task 7,
  `own_err_borrowing_record_stored`.
- A literal argument given to a callee that keeps it must move (the caller must not free it) -
  Task 3, `own_literal_moves` (`keepRay` case).
- An interface with an exported implementer must stay open (no facts) - Task 6,
  `own_err_interface_exported_impl`.

## Shared helpers

Each repro below is minimal and is the test. `V` in the tests is:

```ts
class V {
    constructor(public x: number) {}
    static plus(a: V, b: V) { return new V(a.x + b.x); }
}
```

Debugging recipe (not committed): `tslang --emit=mlir-affine --opt --opt_level=3 --no-default-lib -mm=own --mlir-print-ir-after-all --mlir-disable-threading file.ts 2> ir.txt`
shows the IR each pass gets; `--stack-promotion-report` is unrelated and stays off.

---

### Task 1: Immortal globals (G1)

**Files:**
- Modify: `tslang/lib/TypeScript/OwnershipFacts.h` (attr name + predicate)
- Modify: `tslang/lib/TypeScript/OwnershipSignaturePass.cpp` (`moveIntoGlobals`, ~line 118; `writtenGlobals`, ~line 160)
- Modify: `tslang/lib/TypeScript/OwnershipInferencePass.cpp` (`isImmortalLiteral`, ~line 3840; `reportShapeOfSecondReference`, ~line 4690)
- Modify: the own global-initializer lowering in `tslang/lib/TypeScript/LowerToLLVM.cpp` (`GlobalOpLowering`)
- Test: `tslang/test/tester/own/own_immortal_global.ts`, `own_err_global_returned.ts` (rewritten), `own_err_immortal_borrowed_init.ts`, `own_err_immortal_field_overwritten.ts`

**Interfaces:**
- Produces: `#define OWN_IMMORTAL_GLOBAL_ATTR_NAME "__own_immortal"` (unit attr on `ts.Global`), and
  `bool own_facts::isImmortalGlobalRead(mlir::Value value)` - true for a `ts.Load` of a
  `ts.AddressOf` whose global carries the attr (looking through views as `rootOf` does).
  Tasks 7 and 8 treat such a value as owning nothing.

- [ ] **Step 1: Write the positive test** `own_immortal_global.ts`:

```ts
// V as in the plan's shared helpers
class K {
    static w = new V(1);
    static d = K.w;                       // an immortal global from another
    static f() { return K.w; }            // returned
}
class H { v: V; constructor() { this.v = K.w; } }   // stored into a field
function g(n: number) { return n > 1 ? K.w : new V(n); }   // immortal or fresh
function main() {
    const h = new H();
    print(K.f().x, K.d.x, h.v.x, g(2).x, g(0).x);   // 1 1 1 1 0
    let s = 0;
    for (let i = 0; i < 100000; i++) { s += g(i % 3).x; }   // fresh ones are freed
    print(s);                                              // 66666
    assert(s == 66666);
    print("done.");
}
```

- [ ] **Step 2: Write the negatives.**
  - Rewrite `own_err_global_returned.ts`: keep its body and add `function reset() { V.def = new V(2); }` called from `main` after the print. A reassigned global stays mortal. Regex unchanged: `borrows the global 'V.def' and cannot be stored, returned or captured`.
  - `own_err_immortal_borrowed_init.ts`: `class K { static list = [new V(1)]; static first = K.list[0]; }` and `function main() { print(K.first.x); }`. Regex: `the initializer of global 'K.first' takes a second reference`.
  - `own_err_immortal_field_overwritten.ts`: `class N { constructor(public c: V) {} }`, `class K { static n = new N(new V(1)); }`, `function main() { const c = K.n.c; K.n.c = new V(2); print(c.x); }`. Regex: `borrows .* but is used here after .* released or overwritten`.
- [ ] **Step 3: Register** `own_immortal_global` in the `own_test` list and in a `foreach(mm rc gc none)` pair; add the three negatives to `own_error_cases`.
- [ ] **Step 4: Build and run** `ctest -C Release -R "immortal|global_returned"`. Expected: `own_immortal_global` FAILS under own with `borrows the global 'K.w'`; gc/rc/none PASS; the negatives PASS.
- [ ] **Step 5: Implement.**
  - In `OwnershipSignaturePass`, after `writtenGlobals` is filled: a global gets `__own_immortal` when it is not in `writtenGlobals`, not exported, and its initializer region's stored value is fresh, holds no block, or is a load of another `__own_immortal` global (fixpoint over globals in module order). Run this before `moveIntoGlobals`, which treats a load of an immortal global like `isConstantData`.
  - In inference, `isImmortalLiteral(value)` also returns `isImmortalGlobalRead(value)`; `reportShapeOfSecondReference`'s global branch is then unreachable for immortal globals.
  - Lowering, own only: when a `ts.Global` with `__own_immortal` stores a heap block, write `HEAP_BLOCK_IMMORTAL` into its header (the same header word `LLVMCodeHelperBase` writes for literals, ~line 290), so every release skips it at run time.
- [ ] **Step 6: Run** `ctest -C Release -R "immortal|global_returned|own"`. Expected: all PASS, `test-own-no-counting-own_immortal_global` PASS.
- [ ] **Step 7: Teeth:** with `OWN_TEETH` set, skip the header store; `test-compile-own-own_immortal_global` must fail (the first release frees `K.w`). Remove the switch.
- [ ] **Step 8: Full suite, WSL own tests, spec §25, commit, PR** ("-mm=own: a global never reassigned is immortal").

### Task 2: `Shared<I>` of an object literal (G4)

**Files:**
- Modify: `tslang/lib/TypeScript/OwnershipSignaturePass.cpp` (`moveIntoGlobals`: `touchesHandle`, `isFresh`)
- Modify: `tslang/lib/TypeScript/OwnershipInferencePass.cpp` (the handle construction path of §23.2)
- Test: `tslang/test/tester/own/own_shared_literal.ts`

**Interfaces:**
- Consumes: nothing new. Produces: none.

- [ ] **Step 1: Write the test** `own_shared_literal.ts`:

```ts
interface S { f: (x: number) => number; roughness: number; }
let g = new Shared<S>({ f: function (x: number) { return x * 2; }, roughness: 2 });
function main() {
    let l = new Shared<S>({ f: function (x: number) { return x + 1; }, roughness: 3 });
    const m = l;
    print(g.value.f(3), g.value.roughness, l.value.f(3), m.value.roughness);   // 6 2 4 3
    print("done.");
}
```

- [ ] **Step 2: Register** in `own_test` and a `foreach(mm rc gc none)`. Run `ctest -C Release -R own_shared_literal`. Expected: own FAILS (`the initializer of global 'g' takes a second reference`, and the local's `borrows a field`); gc/rc/none PASS.
- [ ] **Step 3: Implement:** a boxed object literal (the `ts.NewInterface`/box of a folded literal, which §18.2 treats as a view) given to the `Shared` constructor moves in exactly as `new C()` does - in `moveIntoGlobals` (the retain is the handle's own) and in the inference's handle construction.
- [ ] **Step 4: Run** `ctest -C Release -R "own_shared|shared"`. Expected: PASS, including the existing `Shared` tests.
- [ ] **Step 5: Teeth** (skip the move: the literal is freed with the frame while the handle points at it), full suite, WSL, spec §26, commit, PR.

### Task 3: Owning literals moved anywhere (G3)

**Files:**
- Modify: `tslang/lib/TypeScript/OwnershipInferencePass.cpp` (`recordRetainOf`, `checkValueMoves` ~line 3996, the literal branch of `reportShapeOfSecondReference` ~line 4792)
- Modify: `tslang/lib/TypeScript/OwnershipFacts.h` if the "owning record" predicate moves there
- Test: `own_literal_moves.ts`; `own_err_record_literal_given.ts` becomes the positive `own_record_literal_given.ts`; new negative `own_err_literal_moved_twice.ts`

**Interfaces:**
- Produces: `bool own_facts::isOwningRecord(mlir::Value value)` - an object literal (or its interface view) whose fields each are fresh, a moved local, or hold no block. Task 7 builds the borrowing variant beside it.

- [ ] **Step 1: Write the positive test** `own_literal_moves.ts`:

```ts
// V as in the plan's shared helpers
interface Ray { start: V; dir: V; }
interface L { pos: V; }
interface Sc { ls: V[]; c: V; }
class Keeper { rays: Ray[] = []; }
function len(r: Ray) { return r.start.x + r.dir.x; }
function keepRay(k: Keeper, r: Ray) { k.rays.push(r); }
function mk(): Sc { return { ls: [new V(1), new V(2)], c: new V(3) }; }
function main() {
    const p = new V(1);
    const d = new V(2);
    print(len({ start: p, dir: d }));                          // 3   (p, d move in)
    const q = new V(1);
    print(len({ start: V.plus(q, q), dir: q }));               // 3   (fresh + moved)
    const ls: L[] = [{ pos: new V(1) }, { pos: new V(2) }];
    const r: Ray = { start: new V(4), dir: new V(5) };
    const s = mk();
    print(ls[1].pos.x, r.dir.x, s.c.x, s.ls[1].x);             // 2 5 3 2
    const k = new Keeper();
    for (let i = 0; i < 1000; i++) keepRay(k, { start: new V(i), dir: new V(1) });
    print(k.rays.length, k.rays[999].start.x);                 // 1000 999
    print("done.");
}
```

- [ ] **Step 2: Negatives.** Move `own_err_record_literal_given.ts` to `own_record_literal_given.ts` (positive, prints `1`) and drop its `own_error_cases` line. Add `own_err_literal_moved_twice.ts`: `const p = new V(1); const k = new Keeper(); keepRay(k, { start: p, dir: new V(2) }); keepRay(k, { start: p, dir: new V(3) });`. Regex: `is used here after its value was moved`.
- [ ] **Step 3: Register; run** `ctest -C Release -R "literal_moves|record_literal|literal_moved_twice"`. Expected: positives FAIL under own with `an object literal that holds a value it owns`; gc/rc/none PASS.
- [ ] **Step 4: Implement:** an owning record is a value like a fresh object - it may move into a call argument (the caller releases it after a borrowing callee; a keeping callee takes it), an array literal element, a nested literal field, a local through an interface view, or the result. A moved local inside it is moved at the literal (use after = existing use-after-move error). A field read (`this.pos`) inside it is left to Task 7: until then such a literal is accepted only as a call-argument temporary.
- [ ] **Step 5: Run** the same filter plus `-R own`. Expected: PASS. Teeth (skip releasing the argument record: rc/own leak shows nothing - use the double-free direction instead: release `p` after the call too; AOT must crash). Full suite, WSL, spec §27, commit, PR.

### Task 4: A field read into a `let` (G6)

**Files:**
- Modify: `tslang/lib/TypeScript/OwnershipInferencePass.cpp` (`decideSlot` ~line 1887 and the "takes a second reference" path through `reportSecondReference` ~line 4655)
- Test: `own_field_chain_let.ts`

**Interfaces:** none new.

- [ ] **Step 1: Write the test** `own_field_chain_let.ts`:

```ts
// V as in the plan's shared helpers
interface R { dir: V; }
interface Is { ray: R; dist: number; }
function f(r: R) { let d = r.dir; let p = V.plus(d, d); return V.plus(p, d); }
function sh(i: Is) { let d = i.ray.dir; let pos = V.plus(d, i.ray.dir); return V.plus(d, pos); }
function main() {
    const r: R = { dir: new V(1) };
    const i: Is = { ray: { dir: new V(2) }, dist: 3 };
    print(f(r).x, sh(i).x);   // 3 6
    print("done.");
}
```

- [ ] **Step 2: Register; run.** Expected under own: FAIL with `takes a second reference` (after Task 3, the literals themselves pass).
- [ ] **Step 3: Implement:** a `let` initialized from a field read of a parameter (through any chain of field reads and interface views) borrows, bounded by that parameter, and a call ends it only when the call's facts say it may drop (as `own_err_param_read_call` already pins).
- [ ] **Step 4: Run** `ctest -C Release -R "field_chain_let|field_let|param_read_call"`. Expected: PASS (the existing negatives still fail as they should). Teeth, full suite, spec §28, commit, PR.

### Task 5: Read after a move while the new owner holds it (G7)

**Files:**
- Modify: `tslang/lib/TypeScript/OwnershipInferencePass.cpp` (`checkSlotMove` ~line 4400, `reportUseAfterMove` ~line 4983)
- Test: `own_read_after_move.ts`; `own_err_field_alias.ts` and `own_err_union_moved.ts` rewritten; new `own_err_read_after_move_overwritten.ts`

**Interfaces:**
- Produces: in the inference, a use after a move is a **read** when it is a field/element load, a
  call argument the callee borrows (signature facts), a comparison, or a method call through it
  whose facts do not keep it. Any other use (a store, a `let` binding, a keeping call, a return) is
  still a second owner.

- [ ] **Step 1: Write the positive test** `own_read_after_move.ts`:

```ts
// V as in the plan's shared helpers
class N { v: number[] = []; }
class C { x: number = 5; }
class H { c: C | null = null; }
function main() {
    const a: number[] = [1];
    const n = new N();
    n.v = a;
    print(a.length);                       // 1
    const h = new H();
    const c = new C();
    h.c = c;
    print(c.x);                            // 5
    const vs = [new V(1), new V(3), new V(2)];
    let best: V = undefined;
    let m = 0;
    for (let i in vs) {
        let v = V.plus(vs[i], vs[i]);
        if (v.x > m) { best = v; m = v.x; }   // read after the move into best
    }
    print(best.x);                         // 6
    print("done.");
}
```

- [ ] **Step 2: Negatives.** `own_err_field_alias.ts`: insert `n.v = [2];` between `n.v = a;` and `print(a.length);`. `own_err_union_moved.ts`: insert `h.c = null;` before `print(c.x);`. Regex for both becomes `is used here after its value was moved`. New `own_err_read_after_move_overwritten.ts` (overwritten on one path only):

```ts
class C { x: number = 5; }
class H { c: C | null = null; }
function main() {
    const h = new H();
    const c = new C();
    h.c = c;
    if (c.x > 1) { h.c = null; }
    print(c.x);
}
```

Regex: `is used here after its value was moved`. Unchanged and still failing: `own_err_param_kept_used`, `own_err_push_let`, `own_err_two_owners`, `own_err_literal_two_records`.
- [ ] **Step 3: Register; run.** Expected: positive FAILS under own (`is used here after its value was moved`).
- [ ] **Step 4: Implement:** in `checkSlotMove`/the move check, a read (definition above) after a move into a destination that is a local slot, a field of a local or of `this`, or an element store in this function is a borrow through that destination, sound when no path from the move to the read overwrites, releases or moves the destination (reuse the kill walk `readUsedAfter` does for borrows). A move into a call (kept parameter) or `push` is not such a destination.
- [ ] **Step 5: Run** `ctest -C Release -R "read_after_move|field_alias|union_moved|param_kept_used|push_let|two_owners|literal_two_records"`. Expected: PASS. Teeth (drop the kill walk: the overwritten negative compiles and its AOT run reads freed memory). Full suite, WSL, spec §29, commit, PR.

### Task 6: Facts through a module-private interface (G9)

**Files:**
- Modify: `tslang/lib/TypeScript/OwnershipSignaturePass.cpp` (`collectClassVTables` ~line 1127, `families`, `computeFacts` ~line 1741, `meetFacts` ~line 1790)
- Test: `own_interface_facts.ts`, `own_err_interface_exported_impl.ts`

**Interfaces:**
- Produces: a call through an interface vtable slot (`<Class>.<Iface>..vtbl`) is resolved to the
  family of every implementation of that slot when the interface is closed; it then carries the
  meet of their facts (`__own_no_drops`, `__own_params`, `__own_result_borrows`) like a private
  virtual call. Task 7 relies on `__own_result_borrows` arriving on such calls.

- [ ] **Step 1: Write the test** `own_interface_facts.ts`:

```ts
// V as in the plan's shared helpers
interface Th { area(v: V): number; }
class A implements Th { area(v: V) { return v.x; } }
class B implements Th { area(v: V) { return v.x * 2; } }
class Sc { ts: Th[] = [new A(), new B()]; }
function tot(sc: Sc, r: V) {
    let best = 0;
    for (let i in sc.ts) { let a = sc.ts[i].area(r); if (a > best) best = a; }
    return best + r.x;
}
function main() { print(tot(new Sc(), new V(3))); print("done."); }   // 9
```

- [ ] **Step 2: Negative** `own_err_interface_exported_impl.ts`: the same program with `export class B`. Regex: `borrows .* but is used here after it may be released or overwritten`.
- [ ] **Step 3: Register; run.** Expected: positive FAILS under own with `borrows an element but is used here after it may be released or overwritten`.
- [ ] **Step 4: Implement:** an interface is closed when it is not exported, and every class vtable `<C>.<I>..vtbl` and every object literal implementing it is in this module with no exported implementer. Group interface vtable entries into families by `(interface, position)`; a call through such a slot gets the family meet; if any member is open, the call stays open.
- [ ] **Step 5: Run** `ctest -C Release -R "interface"`. Expected: PASS, existing interface tests unchanged. Teeth (give the call `__own_no_drops` when one implementation drops: an implementation that releases its argument's field must crash the AOT run). Full suite, WSL, spec §30, commit, PR.

### Task 7: Borrowing records (G2)

**Files:**
- Modify: `tslang/lib/TypeScript/OwnershipSignaturePass.cpp` (result facts; `__own_result_borrows` becomes a set)
- Modify: `tslang/lib/TypeScript/OwnershipFacts.h` (`resultBorrows`, `borrowedParam` ~line 752-808)
- Modify: `tslang/lib/TypeScript/OwnershipInferencePass.cpp` (records with borrowed fields; bounded uses)
- Modify: `tslang/include/TypeScript/LowerToLLVM/OwnershipRoutineLogic.h` (release of a record skips its borrowed fields, chosen at the creation site)
- Test: `own_borrowing_record.ts`, `own_err_borrowing_record_stored.ts`, `own_err_borrowing_record_outlives.ts`

**Interfaces:**
- Consumes: Task 3 `isOwningRecord`; Task 6 facts on interface calls; Task 1 `isImmortalGlobalRead`.
- Produces: `__own_result_borrows` as an `ArrayAttr` of argument indices (receiver = 0 for
  methods); `llvm::SmallVector<int> own_facts::resultBorrowsAll(mlir::Operation *call)`, with the
  existing single-index `resultBorrows` kept as a wrapper returning the first or -1 until no caller
  needs it. `@lends` accepts several names.

- [ ] **Step 1: Write the positive test** `own_borrowing_record.ts`:

```ts
// V as in the plan's shared helpers
interface Ray { start: V; dir: V; }
interface Hit { thing: Th; ray: Ray; dist: number; }
interface Th { hit(ray: Ray): Hit; k: number; }
class A implements Th { k = 1; hit(ray: Ray): Hit { return { thing: this, ray: ray, dist: ray.dir.x }; } }
class B implements Th { k = 2; hit(ray: Ray): Hit { return ray.dir.x > 5 ? null : { thing: this, ray: ray, dist: ray.dir.x * 2 }; } }
class Sc { ts: Th[] = [new A(), new B()]; }
function closest(ray: Ray, sc: Sc) {
    let best = +Infinity;
    let bestHit: Hit = undefined;
    for (let i in sc.ts) {
        let h = sc.ts[i].hit(ray);
        if (h != null && h.dist < best) { bestHit = h; best = h.dist; }
    }
    return bestHit;
}
function main() {
    const sc = new Sc();
    let s = 0;
    for (let i = 0; i < 1000; i++) {
        const h = closest({ start: new V(0), dir: new V(i % 10) }, sc);
        s += h.thing.k + h.dist;
    }
    print(s);        // 5500: A always wins (dist d against B's 2d), k 1
    assert(s == 5500);
    print("done.");
}
```

(Before implementing, confirm 5500 under rc and gc; if it differs, fix the expected value from the
rc run, not the code.)

- [ ] **Step 2: Negatives.** `own_err_borrowing_record_stored.ts`: `class Keep { h: Hit; }` and `keep.h = sc.ts[0].hit(ray)` where `keep` is a global. Regex: `borrows the result of .* and cannot be stored, returned or captured`. `own_err_borrowing_record_outlives.ts`: `let h = a.hit(ray); a = new A(); print(h.thing.k);` with `a` a `let`. Regex: `borrows the result of .* but is used here after it may be released or overwritten`.
  Add a two-name case to `own_lends.ts`: an exported `@lends("a", "b")` function returning `{ a: a, b: b }`, used while both live.
- [ ] **Step 3: Register; run.** Expected: positive FAILS under own (`an object literal that holds a value it owns` at the `return`).
- [ ] **Step 4: Implement:** a returned literal whose fields are `this`, a parameter, or field reads of them (and possibly fresh values) makes the function's result borrow that set; the record's release frees only its fresh fields (per creation site: the box carries its own routine). At the caller a borrowing result is a borrow of each argument in the set, merged and returned as plain borrows are (`closest` returns a borrow of `ray` and `sc`).
- [ ] **Step 5: Run** `ctest -C Release -R "borrowing_record|borrowed_result|lends|own"`. Expected: PASS. Teeth (release the borrowed fields too: AOT double free). Full suite, WSL, spec §31, commit, PR.

### Task 8: `reduce`, and a parameter returned on some paths (G5)

**Files:**
- Modify: `tslang/lib/TypeScript/OwnershipSignaturePass.cpp` (a parameter returned on some paths and replaced by a fresh value on others is kept: `__own_params`)
- Modify: `tslang/lib/TypeScript/OwnershipInferencePass.cpp` (`otherReturn`, the mixed-return report ~line 4696)
- Modify: the `___array_reduce` generator in MLIRGen if its accumulator is not already a moved local (find it with `grep -rn "___array_reduce" tslang/lib`)
- Test: `own_reduce.ts`; `own_err_mixed_return.ts` rewritten

**Interfaces:**
- Consumes: Task 1 (immortal start value `Color.defaultColor`).

- [ ] **Step 1: Write the test** `own_reduce.ts`:

```ts
// V as in the plan's shared helpers
class K { static zero = new V(0); }
function pick(acc: V, v: V) { return v.x > 2 ? acc : V.plus(acc, v); }   // returns acc on one path
function main() {
    const a = [new V(1), new V(2), new V(3)];
    const r = a.reduce((acc: V, v: V) => V.plus(acc, v), new V(0));
    const q = a.reduce(pick, K.zero);
    print(r.x, q.x);    // 6 3
    print("done.");
}
```

- [ ] **Step 2: Negative.** `own_err_mixed_return.ts`: change `main` to `const c = new C("a"); print(pick(c, true).n, c.n);` (the caller still uses `c`, so it cannot give it up). Regex: `is used here after its value was moved`.
- [ ] **Step 3: Register; run.** Expected: positive FAILS under own (`cannot make the callers of '___array_reduce<...>' give it up`).
- [ ] **Step 4: Implement:** when a function returns a parameter on some paths and another value on others, and every caller can move the argument in, the parameter is kept (`__own_params`): the returning path moves it out, the others release it. `___array_reduce` moves the initial value into its accumulator, gives the accumulator to the callback (kept), and takes the result as the new accumulator.
- [ ] **Step 5: Run** `ctest -C Release -R "reduce|mixed_return|param_kept"`. Expected: PASS. Teeth, full suite, WSL, spec §32, commit, PR.

### Task 9: The ray tracer under own (G8 and the deliverable)

**Files:**
- Create: `tslang/test/tester/own/own_raytrace.ts` (the own copy at 64x64)
- Create: `tslang/benchmarks/raytrace/rtbench_own.ts` (after PR #634 merges)
- Modify: `tslang/benchmarks/raytrace/bench.ps1`, `tslang/benchmarks/raytrace/README.md`, `README.md` (Performance)

- [ ] **Step 1: Write the test** `own_raytrace.ts`: `tslang/benchmarks/raytrace/rtbench.ts` (from PR #634) with the two source edits of the spec (`Shared<Surface>`; `Plane.normal` returns `new Vector(this.norm.x, this.norm.y, this.norm.z)`), rendering 64x64, and `exec` ending in `print("checksum:", sum); assert(sum == 606318);`.
- [ ] **Step 2: Register** in `own_test` and `foreach(mm rc gc)`; run. Expected: gc/rc PASS. Under own, list every remaining error.
- [ ] **Step 3: For each remaining error:** reduce it to a minimal repro in `own/`, decide whether an earlier task's rule should have covered it (fix that rule) or it is a new gap (add the rule here with its own test and negative). No source edits beyond the two in the spec.
- [ ] **Step 4: Run** `ctest -C Release -R own_raytrace`. Expected: PASS under own, gc and rc; `test-own-no-counting-own_raytrace` PASS.
- [ ] **Step 5: Benchmark:** `rtbench_own.ts` = the test at 512x512 with `low digits: 369`; `bench.ps1` builds it with `--mm=own` (with and without stack promotion) and replaces the `tslang own does not compile` rows; record peak working set as the README does. Update both READMEs' tables and drop "does not compile".
- [ ] **Step 6: Full suite, WSL, spec §33 (results: errors left 0, time, memory), commit, PR.**
