# `-mm=own`: a single-owner memory model

Date: 2026-09-24. Status: design approved in conversation; written review 2026-09-28 (§10),
amendments folded in. Phase 0 merged as #399 (results §11); phase 1
(moves by reachability) merged as #401 (results §12); phase 2 (borrows for locals) merged as #402
(results §13); phase 3 (containers and unions) merged as #403 (results §14); phase 4 (function
signatures and `any`) merged as #406-#410 (results §15); phase 5a (closures that do not escape)
merged as #436 (results §16); phase 5b (closures that escape) merged as #437 (results §17); phase
7a (generators, async) merged as #439 (results §18); phase 7b (generators that borrow) merged as
#441 (results §19); phase 6 (the corpus report) merged as #444, #445 and #449 (results
§20, §21); strings as values merged as #452 (§22); phase 8 (`Shared<T>`) designed in §23. Plans
in `docs/superpowers/plans/`.

## 1. Purpose

A fourth memory model beside `gc`, `rc` and `none`, spelled `-mm=own`, in which compiled
code reclaims heap memory with **no runtime reference counting**. Ownership is inferred at
compile time the way Rust's borrow checker checks it: every mortal heap block has exactly one
owner, ownership moves, other uses are borrows bounded by the owner's lifetime, and a program
whose ownership cannot be proven is **rejected with a compile error** rather than compiled with
a fallback.

What it gives that `-mm=rc` does not: no `__tslang_inc_ref`/`__tslang_dec_ref` traffic at all,
destruction at a statically known point, and a diagnostic instead of a leak when ownership is
unclear. What it costs: some valid TypeScript does not compile under this model. That is the
point, and it is why it is a separate model rather than an optimisation of `rc`.

Decisions taken in the design conversation (do not re-litigate):

| Question | Decision |
| --- | --- |
| Goal | No runtime counting |
| Shape | Separate model `-mm=own`, not an elision stage inside `rc` |
| When ownership cannot be proven | Compile error (Rust), not warning+leak, not per-value rc |
| Shared ownership | `Shared<T>`, explicit and counted, as a **later phase** |
| Where the analysis lives | A pass over the ownership ops MLIRGen already emits (approach A), not a borrow checker inside MLIRGen, not user-written annotations |
| Aliasing inside a loop of a value declared outside it | Borrow only; never a move |

## 2. Semantics

### 2.1 Invariant

Every **mortal** heap block - string, array, class instance, capture box, captured-variable
cell, `any` box, tagged-union payload, generator state object - has exactly one owner at every
program point: a local slot, a field, an element, a cell, a global, or a temporary in flight.
There is no count. The owner destroys the block when it gives the block up: scope exit,
overwrite, `delete`, `pop`/`shift`/`splice`.

**Immortal** blocks - string literals, `typeof` tags, objects that crossed in from a module
built under another model - have no owner and may be aliased freely. The free path keeps its
single header read to skip them; that read is the only header access left in this model.

### 2.2 The three verdicts

At every point where `rc` would take a second reference, the compiler reaches one of three
verdicts, tried in this order:

1. **Move.** The source place has no use reachable after the aliasing point. The source gives
   the value up: its scope-exit release disappears statically. A use reachable after a move is
   a *use-after-move* error citing both points. The moved-from slot is not nulled at runtime;
   "moved" is a compile-time fact.
2. **Borrow.** The receiver is a local slot or a temporary, and on every path its last use
   precedes every release and every *dropping mutation* of the owner, and the borrowed value is
   never stored into a longer-lived place, returned, or captured by an escaping closure. The
   receiver takes no reference and releases nothing. Borrows chain: a borrow of a borrow is
   bounded by the outer tenure.
3. **Error.** Neither applies.

A *dropping mutation* of a place is anything that can destroy the value it holds: assignment
to a variable, field or captured variable; `pop`, `shift`, `splice`, `length =` on an array;
`delete`.

### 2.3 Containers own

A store into a field, element, cell or global is a **move into the container**: the source must
be fresh (a `new`, a literal, a call result) or dead afterwards. A read out of a container is a
borrow bounded by the container's own tenure. Boxing into `any` or a tagged union moves into the
box; unboxing borrows.

### 2.4 Calls

Arguments are borrowed, as today. A return is a move: the caller owns the result (today's
"+1 from every call", §9.24 of the rc document, read as "you own it").

Two Rust rules are ported as *inference*, never as annotation:

- **Owned-by-callee parameter.** A callee that stores a parameter into a container or returns
  it makes that parameter owned-by-callee. The caller must reach a *move* verdict for the
  argument.
- **Borrowed-from-argument result.** A callee that returns a borrow of a parameter or of `this`
  (a getter returning `this.field`) marks its result borrowed-from-argument *i*. At the call site
  the result's tenure is the argument's, so the result can only be borrowed. This is Rust's
  lifetime-elision rule.

Both are intra-module facts. A callee without facts - declared, imported from a module that
does not export them, a virtual call whose candidates disagree - defaults to borrowed parameters
and an owned result, the leak-side default `OwnedReturnConsumptionPass` already uses. A module
built under `own` exports one fact, `__own_no_drops`, beside the existing `__tsmm_own_*` marker
(§15.8); the two above stay intra-module.

### 2.5 Closures

A closure that does not escape borrows what it captures; the frame keeps ownership of the
cells. A closure that escapes - stored into a container, returned, boxed, or passed to an
owned-by-callee parameter - takes its cells with it, and any use of those variables in the
enclosing frame after the closure's creation is an error.

### 2.6 Loops

An acquisition inside a loop whose source was declared outside the loop can only be a borrow.
A move there would leave the next iteration reading a moved slot.

### 2.7 Out of scope for the first version

- Generators and async functions: the state object as owner. Own phase (§7, phase 7).
- `Shared<T>`: an explicit, counted generic for graphs that a tree of owners cannot express,
  reusing `rc`'s retain/release routines for that type only. Own phase (§7, phase 8). It rides
  the generics machinery like the settled `WeakRef<T>` design (rc document §9.8); no parser
  change.
- `&mut`-style exclusivity beyond "the owner is not mutated while borrowed". Single-threaded,
  so read/read aliasing needs no rule.
- Lifetime annotations of any kind.

## 3. Architecture

Five pieces; MLIRGen is not one of them. MLIRGen keeps emitting exactly the ownership ops it
emits today (`ts.Retain`/`ts.RetainSlot`/`ts.RetainCell`, their releases, `__owned_consumed`
declarations) and those ops keep erasing under `gc` and `none`, so no existing model's output
changes by a single instruction.

### 3.1 The model

- `MemoryModelOwn` in `include/TypeScript/TypeScriptCompiler/Defines.h`; `memoryModelName()`
  returns `"own"`, so `-mm=own` and the shared-library marker `__tsmm_own_<file>_<hash>` agree
  by construction.
- `CompileOptions` (`include/TypeScript/DataStructs.h`) gains `tracksOwnership()` = rc **or**
  own ("ownership ops are live and lower to code") beside the existing `isRefCounted()` = rc
  only ("a count is maintained in the block header"). Every current `isRefCounted()` site
  (21 across `LowerToLLVM.cpp`, `MLIRGenImpl.h`, `MLIRGenExpressions.cpp`, `LLVMCodeHelper.h`,
  `LLVMCodeHelperBase.h`, `OwnershipRoutineLogic.h`, `CastLogicHelper.h`, `DataStructs.h`) is
  audited into one of the two.
- `needsGCRuntime()` stays false for own: allocation goes through `malloc` as under rc. The block
  header stays (uniform ABI across models, immortal marker); `_MemoryAlloc` stores no count.
- `tslang.cpp`'s `-mm=` option, `opts.cpp`, `transform.cpp`, `exe.cpp`, `jit.cpp` accept the
  new spelling wherever they enumerate models.

### 3.2 `OwnershipSignaturePass` (module pass)

Computes the interprocedural facts of §2.4 and pins them on each `FuncOp` as attributes:

- `__own_params`: the parameter indices the body stores into a container or returns.
- `__own_result_borrows`: the parameter index (or `this`) the result borrows from, if any.

Runs to a fixpoint over the call graph (a function that returns what a callee returns depends
on the callee's fact). Unknown callees default as in §2.4. A library built under own exports
`__own_no_drops` into a symbol beside the marker, and the importer reads it during the symbol
enumeration it already does for `__tsmm_` (§15.8). The other two facts are not exported: a module
linking the library statically sees none, and a caller that does not know them frees twice.

### 3.3 `OwnershipInferencePass` (per `ts.FuncOp`)

Scheduled in `transform.cpp` where `OwnershipVerifierPass` is scheduled - after
`LowerToAffine*`, before the optimiser - and only under `own`. That level is the right one:
control flow is an ordinary CFG with unwind edges as edges, so liveness is standard; yet
`ts.VariableOp` slots and every ownership op are still present and still carry source
locations. Its output is IR with **zero** retain ops left, or errors.

### 3.4 Lowering under own

- `RetainOpLowering`, `RetainSlotOpLowering`, `RetainCellOpLowering`: unreachable under own. If
  one survives, `emitError` ("ownership inference left a retain behind") and fail - never
  silently erase. This is the backstop only; the gate is `OwnershipInferencePass` (§3.3), which
  in phase 0 is a stub that reports every retain at its source location and fails (§10.2).
- `Release*` lowerings call `OwnershipRoutineLogic` as today; `OwnershipRoutineLogic` gains an
  own flavour where `emitIfLastReference` becomes "load header; if not immortal, run the body"
  - one load, no read-modify-write - and the per-type `tsrel_`/`tsrelv_` routines are
  destroy-shaped. `__tslang_inc_ref` and `__tslang_dec_ref` are never referenced.
- `_MemoryAlloc` under own: header word left as the allocator returned it (zero after the
  memset), no count store.

### 3.5 Verifier and test runner

- `--verify-ownership` runs after inference and learns one rule: a slot the inference marked
  `__own_borrowed` or `__own_moved` has neither an acquisition nor a release. Everything else it
  checks today still holds (moved sources lose both halves of their pair; borrowers never had
  either).
- `test/tester/test-runner.cpp` gains the `-mm=own` variant using the existing
  `optVariantSuffix()` cached-script mechanism (`jitown`, `compileown`), and the CMake lists gain
  `test-jit-own-*`/`test-compile-own-*` targets.
- The runner gains an **expected-compile-error mode** (it has none today: `00union_errors.ts`
  is a runtime test). A test file whose first line is `// @error: <text>` must fail to compile
  and its diagnostics must contain `<text>`; a compile that succeeds, or fails with a different
  message, fails the test.

### 3.6 Data flow

```text
MLIRGen (unchanged)
  -> OwnedReturnConsumptionPass          (every model, as today)
  -> LowerToAffine*                      (scopes become CFG, unwind edges become edges)
  -> OwnershipSignaturePass              (own only, module)
  -> OwnershipInferencePass              (own only, per function; deletes retains or errors)
  -> OwnershipVerifierPass               (if --verify-ownership)
  -> optimiser
  -> LowerToLLVM                         (own flavour of OwnershipRoutineLogic)
```

## 4. The inference

Everything the pass decides is a property of a **slot** (a `ts.VariableOp` result, a parameter,
or a heap-owning temporary SSA value) and of the **places** a value can be stored into.

### 4.1 Inputs, all present at affine level today

- Acquisitions: `ts.RetainSlot`, `ts.Retain`, `ts.RetainCell`, and `ts.VariableOp` carrying
  `OWNED_LOCAL_CONSUMED_ATTR_NAME` (owns a fresh value: never an alias, never an error).
- Releases: `ts.ReleaseSlot`, `ts.Release`, `ts.ReleaseCell` - where each owner's tenure ends.
- Stores into places: `ts.StoreOp` whose reference is a field, element, cell or global;
  `ts.ReturnVal`; boxing casts; capture-box construction (`__capture_box`).
- Dropping mutations on a place (§2.2): assignment stores, the array-mutating ops of rc step 5f,
  `ts.DeleteOp`.
- Callee facts (§3.2) at every call, direct, virtual or through a value.

### 4.2 Algorithm, per function

1. **Liveness.** `mlir::Liveness` over the CFG; loads and stores through a slot's ref count as
   uses of the slot.
2. **Tenures.** For each owner: from its acquisition to each of its releases, plus every
   dropping mutation of the place inside that range.
3. **Verdicts**, for each acquisition in program order:
   - *Move* if no use of the source is reachable from the aliasing point (its own releases
     excluded). Delete the retain and the source's releases on paths from here; record
     `__own_moved` on the source slot. A reachable use is a use-after-move error.
   - Else *borrow* if the receiver is a local slot or temporary and, on every path, its last use
     precedes every release and dropping mutation of the owner, and the receiver never reaches a
     store into a longer-lived place, a return, or an escaping capture. Delete the retain and the
     receiver's releases; record `__own_borrowed` on the receiver with the owner as its bound.
   - Else error (§5).
4. **Loops.** An acquisition inside a loop of a source declared outside: borrow only (§2.6).
5. **Containers.** A store into a container is an acquisition whose receiver is the container's
   place; only a move verdict is acceptable.
6. **Closures.** A capture box whose closure value escapes moves its cells in; frame uses of
   those variables after the creation point are errors. A non-escaping closure borrows.
7. **Calls.** An argument to an owned-by-callee parameter goes through step 3 as a move. A
   result flagged borrowed-from-argument gets the argument's tenure.

### 4.3 Precision escape valve

If a shape proves unreadable at the ops level, MLIRGen tags the op with an attribute (precedent:
`__owns_capture`, `__capture_box`) so the pass can read the source-level fact. Never a branch
on the memory model in MLIRGen: that is what keeps the erase-under-gc property, and with it the
structural immunity of the ~2,800 gc tests to this work.

## 5. Diagnostics

One `emitError` at the aliasing point, `attachNote`s at the reason, in the language of the rules:

| Rule | Error | Notes |
| --- | --- | --- |
| use-after-move | `'x' is used here after its value was moved` | `value moved here` |
| borrow outlives owner | `'y' borrows 'x' but is used here after 'x' is released` | `'x' released here` |
| borrow across dropping mutation | `'e' borrows an element of 'arr' but 'arr.pop()' here may destroy it` | `borrowed here` |
| borrow escapes | `'y' borrows 'x' and cannot be stored into 'node.next' / returned / captured` | `borrowed here` |
| container needs a move | `'x' is stored into 'node.next' but still used here` | `stored here` |
| escaping closure | `'x' is captured by a closure that escapes here and is still used here` | `closure created here` |
| callee owns parameter | `argument 'x' is given to 'f', which keeps it, but 'x' is still used here` | `'f' stores its parameter here` |

Errors are collected per function and all reported, then the pass signals failure so the
pipeline stops before lowering. Locations are the ones MLIRGen attached; the verifier already
reports through them at this level.

## 6. Testing

1. **The promise, mechanically.** `--emit=llvm -mm=own` output of every own test contains
   neither `__tslang_inc_ref` nor `__tslang_dec_ref`.
2. **Positive tests, one per verdict shape**: move, borrow, container move, callee-owned
   parameter, borrowed-from-argument result, non-escaping closure, `for...of`, `any` and union
   boxing, `using`/unwind. Each runs as `jitown` and `compileown`, asserts behaviour, and is
   measured: a 1M-iteration allocation loop with the AOT measuring script (rc's
   `measure.ps1`, promoted from the scratchpad into `test/tester/tools/`) must hold flat like
   rc's 3.8 MB, not climb like `none`'s 172.8 MB.
3. **Negative tests, one per row of §5**, through the `// @error:` mode of §3.5, asserting the
   message text.
4. **Teeth**, rc's method adapted: for each positive test, temporarily re-emit the moved
   source's release (a double free) and confirm the test fails. Proves each verdict is
   load-bearing.
5. **Corpus under `-mm=own` as a report, not a gate**: files compiling clean, and a histogram of
   §5 rules hit. This is the model's usability metric and decides which shapes phase 6 refines.
6. `--verify-ownership` in every own test; a silent verifier is a regression.
7. Windows and Linux (WSL) suites green at every phase; rc, gc and none suites unchanged.

## 7. Phasing

Each phase is a PR series leaving `main` green.

0. **Flag and lowering.** `MemoryModelOwn`, the `tracksOwnership()` audit, own flavour of
   `OwnershipRoutineLogic`, retain lowerings that error, runner variant, `// @error:` mode,
   measuring script checked in. Result: every alias is an error, but alias-free programs
   compile, run and reclaim with no counting. First measured numbers.
1. **Move inference** (§4.2 step 3, first verdict) and use-after-move.
2. **Borrow inference** for locals and temporaries: tenures, dropping mutations, loop rule.
3. **Containers**: stores as moves, reads as borrows; `any`/union boxing. As built (§14): unions
   and reads; `any` moved to phase 4.
4. **Signature pass**: owned-by-callee parameters, borrowed-from-argument results, export beside
   the marker and import on the other side. As built (§15): the facts, drop-free callees and
   `any`; only `__own_no_drops` is exported and imported (§15.8).
5. **Closures**: escaping vs non-escaping, cells.
6. **Corpus report and diagnostics polish**: refine the shapes the histogram says matter.
7. **Generators and async**: the state object as owner.
8. **`Shared<T>`**: counted opt-in reusing rc's routines; the verifier learns it.

Phase 0 is shippable behind the flag; 1-3 make the model usable; 4-5 make real programs compile.

## 8. Prerequisites and open items shared with `rc`

- **Foreign blocks are not marked immortal at runtime.** The rc design decided that an object
  crossing from a module built under another model is marked immortal so the counted side never
  frees it (mixed links allowed, leak rather than corrupt). Only *static* blocks carry the
  marker today (`LLVMCodeHelper.h` string and array literals); no import-boundary write exists.
  Consequence for own as for rc: a GC-built default library's objects would be freed with
  `free()`; the ctest suites run `--no-default-lib`. Phase 0 inherits the restriction;
  closing it is one piece of work that serves both models and is tracked here, not in own's
  phases.
- `WeakRef<T>` (rc document §9.8) is designed and not implemented; own needs it no more and no
  less than rc does, and only once `Shared<T>` exists (a tree of single owners cannot form a
  cycle).
- The WASM allocator path is unvalidated under rc and will be equally so under own.

## 9. Non-goals

- Replacing or changing `gc`, `rc` or `none`.
- Making every TypeScript program compile under `own`.
- Any new syntax. `Shared<T>` and `WeakRef<T>` are ordinary generic types.
- Thread safety of ownership; revisit with threading, not before.

## 10. Written review, 2026-09-28

Checked against `main` at #397 (`6cb9eecf`). The references in §3 still hold: the six
`ts.Retain*`/`ts.Release*` ops, `OWNED_LOCAL_CONSUMED_ATTR_NAME`, the `__tsmm_` marker,
`memoryModelName()`, `optVariantSuffix()`, the verifier's slot in `transform.cpp`, and
21 `isRefCounted()` call sites plus its definition. Six amendments:

### 10.1 A lowering error does not fail the compile (prerequisite, every model)

`runMLIRPasses` (`tslang/transform.cpp`) collects every diagnostic and prints it, but its
result reflects only `pm.run()`. A pattern that calls `emitError` and still returns success -
`CastLogicHelper.h`'s "cast from ... is not supported" is the example that was found - prints
`error:` and the program is then emitted and run anyway: `<string><any>(number[] | null)`
under `--emit=jit` prints the error and crashes with 0xC0000005. §3.4's backstop would
inherit the hole: a surviving retain would print an error and ship a double free. Fix,
before phase 0 and in every model: an error-severity diagnostic fails the compile. It may
surface latent errors the suite tolerates today; each is a real bug.

### 10.2 The phase-0 gate is a pass, not the lowering

Phase 0 schedules `OwnershipInferencePass` as a stub: under own it walks each `ts.FuncOp` at
the verifier's slot, reports every surviving `ts.Retain*` at the location MLIRGen gave it
("... takes a second reference; ownership inference cannot prove a move or borrow yet"), and
signals failure. Lowering has lost the slot names by then; the pass has not. Phase 1 grows the
same pass. The lowering `emitError` of §3.4 stays as the backstop.

### 10.3 Default library

`exe.cpp` and `jit.cpp` resolve the default library directory by `memoryModelName()`, so
`-mm=own` without `--no-default-lib` fails with "no default library built for -mm=own" rather
than linking a GC-built library whose objects `free()` would destroy. That is the right
answer until §8's import-boundary marking exists; own tests run `--no-default-lib` as rc's do.

### 10.4 The header word still needs its store

§3.4 says `_MemoryAlloc` under own leaves the header "zero after the memset". The memset runs
on wasm only (`useCalloc`/`zeroing`); a native block's header is whatever `malloc` returned.
The own flavour of the free path reads the header to skip immortal blocks, so `_MemoryAlloc`
keeps its store of 0 under own - the same single store rc makes, meaning "mortal" rather than
"count 0". What own drops is the read-modify-write on every retain and release.

### 10.5 Expected-error tests use the existing ctest pattern

§3.5's `// @error:` runner mode is not needed: `test/tester/CMakeLists.txt` already tests
compile errors with plain `add_test` + `PASS_REGULAR_EXPRESSION` + `FAIL_REGULAR_EXPRESSION`
(`call_arity_cases`). The runner's compile scripts do not capture the compiler's diagnostics,
so a runner mode would have to re-invoke tslang itself; own's negative tests follow the
call-arity pattern instead.

### 10.6 Almost every retain is a birth reference, not an alias

Blocks are born unowned (count 0, rc §9.24), so the first owner of *every* fresh value - a
local's declaration, a temporary passed to `print`, a `new` - takes it with a `ts.Retain` or
`ts.RetainSlot`. Measured on `main` with `--emit=mlir-affine -mm=rc --no-default-lib`:
`print(i)` for an `s32` is `Cast {__owned_result}` + `Retain` + `Print` + `Release`;
`const s = "item " + i` is two retains and a release; only programs that allocate nothing are
retain-free. A phase-0 gate that rejects every retain therefore compiles only heap-free
programs, and §7's "alias-free programs compile, run and reclaim" is not reachable without
the move verdict for a fresh value's single acquisition - which depends on each producer's
+0/+1 convention (`__owned_result`, the retaining return of rc §9.24/§9.25, pop/shift). How
phase 0 and phase 1 split around this is decided in the plan.

## 11. Phase 0 results, 2026-09-29

Branch `own-phase-0`, stacked on PR #398. `-mm=own` compiles a program in which every heap
block has one owner and rejects every other ownership shape at its source location. Its LLVM
output references neither `__tslang_inc_ref` nor `__tslang_dec_ref`: a release is one header
load, an immortal check and the destroy.

### 11.1 What phase 0 accepts

Owners are counted from the uses of a heap value, not from rc's retains. rc also transfers
ownership by *consuming* a value, which leaves no retain behind: a declaration marked
`__owned_consumed`, a push, a field store. The first implementation counted retain ops, and the
whole-branch review found programs that compiled and then double-freed: `const c = new C()`
pushed in a loop, pushed and then held by a `let`, or taken by two `let`s. `delete` was a
further problem, because it destroys a value that its slot releases again at scope exit.

The rule as built:

- A **taking use** of a value V is one of these: a declaration marked `__owned` or
  `__owned_consumed` whose initializer is V; a `ts.Store` of V into any reference (a field, an
  element, a global or a return slot); an insertion of V by push, unshift or splice; a return;
  and any use not on a short borrow list. The borrow list is print, calls, property and element
  references, `length`, method and `this` references, arithmetic, comparison, concatenation, a
  non-owning declaration, `--di`'s `ts.DebugVariable`, and the retain/release bookkeeping.
- **Owners** of V are its taking uses, plus one if V is released as a temporary. There must be
  at most one owner. Zero is a leak, as under rc.
- **A move out of a temporary.** One taking use and one temporary release count as one owner
  when the release comes after the taker in the same block and rc gave the taker a reference of
  its own (a `ts.Retain` of V in front of it, or the `ts.RetainSlot` of an owning `let`). The
  value moves into the taker, and the release is erased with the retains. This is the shape of
  a folded `const` handed on, `const a = new C(); let b = a;`, since rc stopped letting a
  receiver take a named `const`'s reference over (that let one reference have several holders
  under rc: `00owned_named_consts.ts`). A
  taker rc gives no reference, such as `ts.NewInterface`, is still a second owner.
- **Locality**: the taking use is in V's defining block, and nothing uses V after it. That one
  check rejects a move inside a loop (§2.6), a move on one branch, and use after move, with no
  liveness analysis. Phase 1 relaxes it.
- A **fresh** V is made by `ts.New`, `ts.CreateArray`, `ts.NewArray`, `ts.StringConcat` or
  `ts.CharToString`, by a non-call op marked `__owned_result`, by a cast of a constant, or by a
  direct call of a function defined in this module. Every function returns its result retained
  (rc §9.24), and the call's own `__owned_result` mark does not survive the affine lowering.
  Only a fresh V's retains are erased. A retain of anything else is an error, and so is a
  retain of a variable hoisted in front of a `try`.
- A cast of a constant is the immortal global, which a release skips, or a copy with one owner,
  which a release destroys. It was added because `let s = "abc"` and `v: number[] = []` were
  rejected.
- `delete`, `ts.RetainCell` (any captured variable, even a number) and a captured variable
  holding a heap value are errors in phase 0. A retain that reaches LLVM lowering is a
  backstop error.

Interface and `any` boxing count as a second owner beside the temporary, so they are rejected.
That is phase 3.

The planned "released but never owned" rule was dropped on the premise that no program reaches
it. The premise was wrong: rc's latent over-releases are reachable through consumption and
`delete`, as shown above. What closes them is counting consumptions as owners and rejecting
`delete`, not that rule. The ownership verifier now runs *before* inference under own, so
`--verify-ownership` checks rc's pairing on the IR as MLIRGen made it.

Every probe from the review (32 programs) is now either rejected or produces gc's output under
own AOT. Several of them also mis-run under `-mm=rc`, so the rc model has latent
over-releases of its own; they are listed in the final report for a separate fix.

### 11.2 Measured

`test/tester/tools/measure.ps1` builds each program AOT at `-O3` and reads the peak working set
from the process handle after exit:

| Program | gc | rc | none | own |
| --- | --- | --- | --- | --- |
| `own_fresh_string` (1M strings) | 5.8 | 4.2 | 80.9 | 4.2 |
| `own_return_new` (1M instances via a factory) | 5.8 | 4.2 | 50.2 | 4.2 |
| `own_try_local` (100k throws holding an array) | 5.9 | 6.5 | 6.5 | 6.5 |
| `own_fresh_array` | 5.6 | 4.2 | 4.2 | 4.2 |
| `own_literals` | 5.6 | 4.1 | 4.1 | 4.1 |

All figures are MB. Own holds flat exactly where rc does. In `own_fresh_array` and
`own_literals`, LLVM removes the allocations at `-O3`, so none does not grow and the rows show
nothing about reclamation. `own_try_local`'s throw path does not release its local under any
model, which is rc's leak too.

### 11.3 The count is load-bearing

Tested both ways:

- With every retain kept, all 15 positive checks fail at the backstop.
- With the rule loosened to two acquisitions, `own_err_two_owners.ts` (a fresh array stored
  into a field and a local) compiles and segfaults under own, while rc runs it correctly. It
  is now a negative test.

### 11.4 Known limits (the input to phase 1)

- An owned local declared inside a `try` body is rejected, because its storage is hoisted.
- A class field initialized from a parameter, an alias through `let`, and a store of a local
  into a field are all rejected (a retain of a load).
- Closures are rejected, whatever they capture: every captured variable's cell gets a
  `ts.RetainCell`.
- Interface and `any` boxing, `delete`, a callee that keeps its parameter (`g = c` retains a
  value the callee does not own; spec §2.4's owned-by-callee parameter is phase 4), and a move
  out of the value's own block are all rejected.
- Results of runtime helpers and `declare`d functions are rejected.
- A field initialized from an array literal of eight numbers is rejected ("takes a second
  reference" at the literal); four are accepted. Not yet diagnosed.

## 12. Phase 1 results, 2026-09-29

Plan: `docs/superpowers/plans/2026-09-29-own-phase-1.md`. Only `OwnershipInferencePass.cpp`
changed; MLIRGen and the lowering did not.

### 12.1 What phase 1 accepts

A move (§2.2, verdict 1) is decided by reachability on the affine CFG, where structured
statements are already `cf.br`/`cf.cond_br` between blocks.

- **Sources.** A fresh SSA value, as in phase 0, or an **owning local slot**: a `ts.Variable`
  marked `__owned`, not captured, read with `ts.Load` through casts that keep the same block (a
  class widened to a union with null, an upcast). Parameters, fields and boxing casts are not
  sources yet.
- **The move point** is where rc takes the receiver's reference: its `ts.Retain` of the value
  when that comes first, else the taker. A `return` at this level is a `ts.Store` into the return
  slot, with the retain *before* the source's scope-exit `ts.ReleaseSlot`, so measuring from the
  store would leave that release alive.
- **Use after move.** Any other use of the source reachable from the taker, without passing the
  source's definition (its producer, or the slot's `ts.Variable`), is an error: `'a' is used here
  after its value was moved`, with a note at the move. For a slot that includes an assignment, and
  every use of what *any* read of the slot returned: `const b = a` is folded into a `ts.Load` of
  `a`'s slot, so `b.x` after `a` moves reads the moved value through a load that came before the
  move (found by the final review: it compiled and read freed memory).
- **Loops (§2.6).** A taker control comes back to without the value being made again is `moved
  inside a loop but was made outside it`. A move out of a `let` declared in the loop body is fine:
  each iteration passes the declaration.
- **Releases.** Each release of the source the move can reach must be dominated by the move, and
  is erased. A release the move cannot reach stays: it is the owner on the paths that did not
  move (`if (x) { let b = a; return ...; } use(a)`). A reachable release the move does not
  dominate is `moved here on some paths only`.
- **A taker rc gives no reference** (`ts.NewInterface`) with the temporary also released is still
  a second owner (§11.1).

Also changed: a cast of `null` or `undefined` counts as a literal, so `c: C | null = null`
compiles; under `--di` the errors name the variable, read from MLIRGen's `NameLoc`; and only a
value whose type owns heap memory is a candidate, while a string literal (the immortal global)
may be held by any number of places. Under `--opt`, and so under the JIT, CSE merges identical
literals before inference; one merged `0` stored into a variable inside a loop and before it was
reported as a move inside a loop.

### 12.2 Measured

1M iterations, AOT, `measure.ps1`, both `-O3` and `-O1` (identical within 0.1 MB):

| program | gc | rc | none | own |
| --- | --- | --- | --- | --- |
| `own_move_let` (`let b = a`) | 6.6 | 4.8 | 546.8 | 4.8 |
| `own_move_return` (`return a`) | 6.6 | 4.8 | 546.8 | 4.8 |
| `own_move_field` (`h.c = a`) | 6.5 | 4.8 | 546.8 | 4.8 |

### 12.3 Teeth

With the moved releases kept (both `toErase.insert(moved...)` lines disabled), 8 of the 10
move checks fail. `own_move_loop_body` passed, because it read the value before either release
ran and allocated nothing over it. It now keeps every value in an array past the loop and reads
them after `churn()`, and fails too (JIT exit 127).

### 12.4 Changed negative tests

Two takers of one value are now a move followed by a use, so `own_err_const_twice`,
`own_err_push_let`, `own_err_two_owners`, `own_err_use_after_move`, `own_err_field_alias` and
`own_err_alias` report `is used here after its value was moved` instead of `takes a second
reference`. `own_err_alias` was a bare `let b = a`, which phase 1 accepts; it now reads `a`
after the move. The loop cases report `is moved inside a loop`.

### 12.5 Known limits (the input to phase 2)

- Borrows (§2.2 verdict 2): a value read after it was stored into a field or a local, a value
  made outside a loop and used as an owner inside it, and a `const` alias of a `let` held across
  the `let`'s move (`const b = a; let c = a; ... b.x`).
- A `const` alias of a `let` held across an *assignment* of the `let` (`const b = a; a = new C();
  b.x`) is not caught: the assignment is not a move, and it destroys what `b` reads. rc has the
  same bug today (it prints garbage); own inherits it until phase 2's dropping-mutation rule.
- A move on some paths only: needs drop elaboration, releasing on the paths that did not move.
- Assigning a moved `let`: rc's assignment releases what the slot holds, the moved value.
- Moves out of fields, parameters, and through boxing casts; moves into a nested region's op
  (compared by the statement that holds them, which answers "reachable").
- Everything in §11.4 not listed above as accepted: owned locals in a `try` body, closures,
  interface and `any` boxing, `delete`, callee-owned parameters, runtime helpers and `declare`d
  functions, the eight-element field literal.

## 13. Phase 2 results, 2026-09-29

Plan: `docs/superpowers/plans/2026-09-29-own-phase-2.md`. Only `OwnershipInferencePass.cpp`
changed.

### 13.1 What phase 2 accepts

The verdicts run in §2.2's order. Phase 1's move check runs first, quietly. When it fails, a
receiver that is a `let` borrows (verdict 2). That means an owning `ts.Variable`, declared from
the value, that took its own reference (a `ts.RetainSlot`, not `__owned_consumed`) and is not
captured. Its retain and every `ts.ReleaseSlot` of its slot are erased, and the owner keeps all
of its releases.

A borrow is sound while both of these hold:

- **No use of the borrower can run after any end of the owner.** The ends are a local owner's
  `ts.ReleaseSlot`s and assignments (`ts.Store` into its slot), or a temporary owner's
  `ts.Release`s. The uses are every read of the borrower's slot and every use of anything derived
  from a read that can still point into the block: casts that keep the block, field and element
  references and values loaded through them, bound methods, and a catch or finally clause's
  non-owning local. The walk stops at a number or a boolean. The final review found the first
  version, which followed only casts, compiling `const f = b.m; a = new C(); f()`, a `for...of`
  over `b.v`, and a catch local of `b` into reads of freed memory. A path back to a use through
  the borrower's own declaration is a new borrow: `for (...) { let b = a; ...; a = new C(); }` is
  fine.
- **Nothing the borrower holds is kept.** It must not be retained, taken by a non-borrowing use
  (stored, pushed, returned, declared into another owning `let`), or assigned.

Where an owner has **more than one receiver**, every one of them has to borrow: a borrow pins
its owner. A field store beside a borrowing `let` is `'a' is moved here while 'b' borrows it`.

The errors follow §5. Under `--di` they name both variables, and a folded `const` owner is named
through its `ts.DebugVariable`:

- `'b' borrows 'a' but is used here after 'a' is released or overwritten`, with a note at the end;
- `'b' borrows 'a' and cannot be stored, returned or captured`;
- `'b' borrows 'a' and cannot be assigned`.

### 13.2 Measured

1M iterations (100K calls), AOT, `measure.ps1`, at `-O3` and `-O1` (identical within 1.5%):

| program | gc | rc | none | own |
| --- | --- | --- | --- | --- |
| `own_borrow_inner_let` | 6.6 | 4.8 | 1088.3 | 4.8 |
| `own_borrow_loop` | 6.4 | 4.7 | 113.4 | 4.7 |
| `own_borrow_two_lets` | 6.6 | 4.8 | 546.8 | 4.8 |

### 13.3 Teeth

With each borrower's releases kept (only its retain erased), 10 of 10 borrow checks fail
(`own_borrow_nullable` included). The borrower's release destroys what the owner still holds.

### 13.4 Changed tests

These phase-1 negatives are legal now, and positives replace them:

- `own_err_use_after_move`, `own_err_const_twice` → `own_borrow_inner_let`;
- `own_err_alias`, `own_err_let_twice` → `own_borrow_two_lets`;
- `own_err_let_loop` → `own_borrow_loop`;
- `own_err_move_some_paths` → `own_borrow_branch`.

Other negatives:

- `own_err_move_some_paths_field` keeps "some paths" covered (a field store cannot borrow).
- `own_err_assign_after_move` and `own_err_const_alias_move` now report the borrow's error.
- New: `own_err_borrow_escape`, `own_err_borrow_and_move`, and from the final review
  `own_err_borrow_bound_method`, `own_err_borrow_catch_local`, `own_err_borrow_finally_local`,
  `own_err_borrow_derived_ref`, with `own_borrow_nullable` (`if (b)` on a nullable borrower was
  a false escape).

### 13.5 Known limits (the input to phase 3)

- Reads out of containers as borrows (§2.3): `const e = arr[i]`, `const c = h.c`, bounded by the
  container's tenure and its dropping mutations (`pop`, `shift`, `splice`, `length =`, an
  overwrite).
- Borrow chains (`let c = b` where `b` borrows) and borrowers that are assigned.
- Borrows of parameters and of `this`, which need callee facts (phase 4).
- A move after the last use of every borrower. It is rejected today, since any borrower pins
  its owner.
- A temporary taken by a `let` and also consumed elsewhere (`let b = h.c = new C()`): with no
  temporary release there is nothing to bound a borrow, so it is still an error.
- From §12.5: moves on some paths only, and a `const` alias of a `let` held across an assignment
  of the `let` (rc has the same bug).

## 14. Phase 3 results, 2026-09-29

Plan: `docs/superpowers/plans/2026-09-29-own-phase-3.md`. Only `OwnershipInferencePass.cpp`
changed.

### 14.1 What phase 3 accepts

**Views.** An op whose result is the block its operand holds is transparent. The pass strips it to
its root wherever it names a value, and looks through it wherever it lists a value's uses. The
views are:

- a `ts.Cast` between class, union and optional types;
- `ts.CreateUnionInstance` and `ts.GetValueFromUnionOp`;
- `ts.OptionalValue` and `ts.Value`.

So `let u: C | null = new C(i)`, `let u: C | string = c` and `let u: C | undefined = ...` are one
owner, not a second reference. An empty optional (`u = undefined`, `ts.OptionalUndef`) holds no
block, like `null` widened. Testing a value is a read, not a taker: `if (d)`, `d?.x`, `typeof u`,
and a cast to `boolean`.

**Reads out of containers are borrows (§2.3).** `const c = h.c`, `arr[i]`, and every temporary
read of a heap value through a `ts.PropertyRef` or `ts.ElementRef` own nothing. Before phase 3
nothing bounded them, and `const c = h.c; h.c = new C(); c.x` compiled into a read of freed
memory. rc has the same hole.

A read may not be kept: storing, pushing, returning or retaining it is
`'c' borrows 'h.c' and cannot be stored, returned or captured`. None of its uses, and no use of
anything it produces, may be reachable from a *drop*:

- an overwrite (`ts.ReleaseSlot`) of any place on the way from the read to its root. Places are
  matched by position and reference type, which two references to one field of one object always
  share; an unrelated field that happens to match only adds an error;
- a `pop`, `shift`, `splice` or `length =` of an array whose element type is on the way;
- an end of the root: a local's releases, assignments, and every declaration or retain that may
  move it away; a made value's releases and every use that takes it; a global's assignments;
- a call. A call drops the read if it is given something derived from the root that the read did
  not produce (`reset(h)` drops `h.c`; `use(c)` does not). Where the root is a parameter, `this`
  or a global, **any** call drops it: the callee may reach the owner through a global.

A path back to a use through the read itself is a new borrow, so reading `h.c` at the top of a
loop body and overwriting it at the bottom is fine. The error is
`'c' borrows 'h.c' but is used here after it may be released or overwritten`, with a note at the
drop.

A borrow is followed through what hands the same value on. That covers casts and views, the
argument of a block a branch passes it to (`n.c ?? m.c`), and the reads of a local that owns
nothing and is assigned it (`for (x of arr)` into an outer `let x` declared without a value).
Past a merge or such a local, the value is made again where that is replaced: the merge's block,
or the local's assignments. The branch or store that hands it on is itself a use, bounded by the
read. The final review found the first version, which kept the read as the only point where the
value is made again, wrong both ways:

- It compiled a local assigned on one iteration and read on the next, after the overwrite, into a
  read of freed memory.
- It rejected `c ? h.c : h.d` after any earlier call, through the other branch.

A container reached through a merge (`(i > 0 ? a : b).c`) has each merged value as a root, and
each root's ends bound the read. The first version saw one root it did not own, with no ends, and
compiled `a = z` between the read and its use into a read of freed memory.

**A `let` declared from a read borrows it**, as phase 2's borrowers do: `let c = h.c`,
`let e = arr[j]`. Its `ts.RetainSlot` and `ts.ReleaseSlot`s go, and its reads join the read's
uses. Assigning it is `'c' borrows 'h.c' and cannot be assigned`.

### 14.2 Measured

1M iterations (100K calls), AOT, `measure.ps1`, at `-O3` and `-O1` (identical within 4%):

| program | gc | rc | none | own |
| --- | --- | --- | --- | --- |
| `own_field_borrow` | 6.5 | 4.8 | 562.2 | 4.8 |
| `own_field_let_borrow` | 6.5 | 4.8 | 546.6 | 4.8 |
| `own_union_nullable` | 6.6 | 4.8 | 546.6 | 4.8 |
| `own_union_tagged` | 6.6 | 4.8 | 546.6 | 4.8 |
| `own_element_borrow` (10K) | 6.5 | 4.7 | 11.0 | 4.7 |

### 14.3 Teeth

- With a move's releases kept, the three union positives fail with heap corruption (6 of 6 runs).
- With a place-read `let`'s releases kept, both `let` positives fail the same way (4 of 4).
- A folded `const` read erases nothing, so it has nothing to disable. Its teeth are the negatives,
  which compiled before this phase: five of them read freed memory under rc and own (`999` for
  `0`).

### 14.4 Tests and the corpus

- **Positives:** `own_union_nullable`, `own_union_tagged`, `own_union_optional`,
  `own_optional_access`, `own_field_borrow`, `own_element_borrow`, `own_container_loop`,
  `own_borrow_merge`, `own_field_let_borrow`, `own_element_let_borrow`, and from the final review
  `own_borrow_merge_after_call` (also a push onto the array a `for...of` walks).
- **Negatives:** `own_err_union_moved`, `own_err_field_read_overwritten`,
  `own_err_element_read_overwritten`, `own_err_element_read_popped`,
  `own_err_element_read_loop_pop`, `own_err_field_read_owner_assigned`,
  `own_err_field_read_owner_moved`, `own_err_field_read_call`, `own_err_param_read_call`,
  `own_err_global_read_assigned`, `own_err_merge_read_overwritten`, `own_err_alias_read_popped`,
  `own_err_container_read_stored`, `own_err_field_let_overwritten`, `own_err_field_let_assigned`,
  and from the final review `own_err_alias_read_stale`, `own_err_merge_root_assigned`.

Corpus (`test/tester/tests/*.ts` under `-mm=own --no-default-lib`): 252 of 564 compiled before,
272 after. 23 files now compile. 3 stopped:

- `export_class_abstract`;
- `export_class_abstract_virtual_dispatch`;
- `export_class_implements_interface_abstract`.

Each has ``${this.color} area=${this.area()}``, a read of a field of `this` that is still used
after a call: the any-call rule above.

### 14.5 Deferred: `any`

§7 put `any` boxing in phase 3; it moves to phase 4. Unboxing is a call to a generated
`___unbox<T>`, which retains the payload inside and returns it +1. `___cast<U, T>` allocates on one
branch and returns the payload on the other. Treating either as a borrow is phase 4's
borrowed-from-argument result.

An rc bug found here belongs with it: `const a: any = new C(i)` never releases the box
(`measure.ps1`: rc 66.2 MB, gc 6.1 MB). The boxing cast is +0, and a folded `const` takes no
reference.

### 14.6 Known limits (the input to phase 4)

- Reads of a parameter's or `this`'s fields across any call, including
  ``${this.color} ${this.area()}``. Callee facts decide which calls may reach the owner.
- Returning a field (`get c() { return this.c; }`): a borrowed-from-argument result.
- `any` boxing and unboxing (§14.5).
- A read out of a container reached through an interface (`ts.InterfaceSymbolRef`), which is not a
  `ts.PropertyRef` and is not bounded yet.
- From §13.5: borrow chains through assigned borrowers, moves after the last use of every
  borrower, and a temporary taken by a `let` and consumed elsewhere.

## 15. Phase 4 results, 2026-09-29

Plan: `docs/superpowers/plans/2026-09-29-own-phase-4.md`. New: `OwnershipSignaturePass.cpp` and
`OwnershipFacts.h` (the helpers both passes read the ops with). Changed:
`OwnershipInferencePass.cpp`, and two MLIRGen sites for `any` (§15.4).

### 15.1 The signature pass

A module pass, own only, just before inference. It resolves every call it can, computes three
facts per function, and pins on each resolved call the facts its candidates agree on:

- `__own_no_drops`: the callee destroys nothing its caller can reach. That means no overwrite of a
  field or element whose root it did not make, no global assignment, no removal from an array it
  did not make, no `delete`, and no call that may drop. It is a least fixpoint. A constructor
  filling its own `this` is not a drop.
- `__own_result_borrows = K`: every heap value the function returns is parameter `K`'s. That is
  seen through views, `!ts.opaque` casts, `ts.Unbox`, field and element reads, the object a method
  is called on, and results that borrow. `null` agrees with any.
- `__own_params`: the parameters the body keeps. A kept parameter is stored into a field, element
  or global, pushed, or passed to a kept parameter.

**Resolving a call.** The candidates are found as follows:

- a direct call has one candidate;
- a virtual call reaches every entry at its index under its method name in any class `..vtbl`.
  This is its family, and every member must be private and defined here;
- a call through the vtable's first slot is only built to reach a generated `..instanceOf`,
  which drops nothing (§15.7 has the bug this found);
- anything else (a closure, a function read from a field, an interface) is unknown.

**The closed world.** `__own_no_drops` goes one way: an unknown call may drop, which can only add
an error. The other two facts are relied on by the callee's body. A caller that does not know them
borrows the argument and releases it, or releases a result it does not own, a double free either
way. So a function has them only if all of these hold:

- it is private;
- every use of its symbol is a direct call, a class vtable entry, or a method reference used only
  as a callee;
- every family it is in agrees.

A function that loses them keeps rc's convention. The error in its body gets a note saying why:
`'firstOf' could return a borrow or keep a parameter, but it is used other than by a call` (or
`it can be called from another module`, `an override in its class family disagrees`,
`an override may be defined in another module`). The facts are recomputed with the calls pinned
until nothing changes, so a method returning another's borrow, and a function forwarding a
parameter to one that keeps it, get the fact too.

`mlir::SymbolTable::getSymbolUses(module)` does not look inside the module, which is a symbol
table. The first version saw no uses, so every function was closed. `own_err_borrowed_result_escapes`
compiled until the walk used the module's body region.

### 15.2 What the inference accepts

- **Calls that drop.** A call drops a borrow only if its callee may drop (`__own_no_drops` absent).
  `` `${this.color} area=${this.area()}` `` compiles again. Phase 3 skipped a call's own arguments
  ("read before it runs"). That was unsound: `f(h.c)` compiled where `f` resets `h.c` through a
  global and then reads its parameter. Now a call given a borrow uses it for as long as the callee
  runs. Two things are exempt, because they are read as the call starts: the function value an
  indirect call goes through, and the object a field-held method is bound to (`o.m()`).
- **Results that borrow.** A call with `__own_result_borrows` is a borrow read like a field read.
  Its places are a wildcard, so any overwrite of any field or element, and any removal, drops it.
  Its roots are the argument's. Its temporary release and a consuming `let`'s releases are erased.
  In the callee, the retain rc made for the caller goes, and the return is a use, not an escape.
- **Kept parameters.** In the callee, the take is a move out of the parameter's slot. It must be
  the last use, not loop, and dominate every return. The callee has no release for the parameter,
  so `if (f) h.c = c` is an error. In the caller, the call is a taker of each kept argument, and rc's
  retain for it is in the callee. It can move a fresh value, an owning `let` or a kept parameter. A
  parameter, a global or a field it does not own is
  `argument 'x' is given to 'keep', which keeps it, but -mm=own cannot move it here`.
- **A thrown value read by a catch's copy thunk** (`.eh.copy.*`) is a move. A throw hands the
  thrower's reference to the exception object, and nothing gives it back (rc leaks it the same
  way). A TypeScript rethrow is a new throw, so each exception is copied into one catch.
  `___unbox`'s failure path throws a string, so every unboxing needs this.

A bug the positive tests found: a move whose acquisition is the call itself was "erased with its
retain". The call to `forward(h, a)` vanished, and the program read uninitialised memory.

### 15.3 Measured

AOT, `measure.ps1`, at `-O3` and `-O1` (identical within 1%), in MB:

| program | gc | rc | none | own |
| --- | --- | --- | --- | --- |
| `own_param_kept` | 6.5 | 4.8 | 1657.7 | 4.8 |
| `own_getter_borrow` | 6.5 | 4.8 (was 12.6) | 1653.3 | 4.8 |
| `own_any_box` | 6.5 | 4.8 | 558.9 | 4.8 |
| `own_call_no_drops` | 6.5 | 4.8 | 1100.7 | 4.8 |

rc climbed in `own_getter_borrow`: it never released a getter's result used as a temporary
(`h.cc.x`). The getter retains its result like any function, but `OwnedReturnConsumptionPass`
only settled `ts.CallIndirect`, and a getter read is an accessor op until the affine lowering.
The pass now classifies accessor reads the way it classifies calls (`rc_getter_temporary.ts`).
Own has no reference to give back.
§14.5's `const a: any = new C(i)` loop reads 4.6 MB under all four models, at both levels. `none`
does not grow, so LLVM removes the allocation and that program shows nothing about reclamation.

### 15.4 `any`

The boxing cast now carries its birth reference (`__owned_result` + `ts.Retain`), as a printed
number does (`markFreshStringOwned` became `markFreshBlockOwned`). That is rc's §14.5 leak fix:
a box held by a folded `const`, or passed straight to a call, was never released. Under own it
makes the box a fresh value with one owner. An `any` cast to `any` allocates nothing and is not
marked. The fix broke rc's catch copy thunks at first: the thunk's plain store into the catch slot
let the box's new reference go back at the end of the thunk, so the catch read a freed box (10
suite failures). The store now takes it over. The catch slot is runtime memory.

`<C>a` is `___unbox<C>(a)`. With `___unbox`'s borrowed result (through `ts.Unbox`) and its
drop-free `.instanceOf` call, the payload is borrowed for as long as the box lives.

### 15.5 Teeth

- With the releases a kept argument's move erases kept, `own_param_kept` fails under JIT and AOT.
- With a borrowed result's temporary release kept, `own_getter_borrow` and `own_any_box` fail.
- Keeping the callee's retain for a borrowed return makes the same two fail at compile time: the
  retain reads as an escape. So it proves nothing about the runtime. At runtime it would be a leak,
  not a double free.
- `own_call_no_drops` erases nothing. Its teeth are the negatives.

### 15.6 Tests and the corpus

- **Positives:** `own_call_no_drops`, `own_getter_borrow` (getter, method, free function, a method
  forwarding another's borrow, a field-or-`null` result, a returned parameter), `own_param_kept`
  (field, forwarding, method into `this`, push, constructor parameter property), `own_any_box`.
- **Negatives:**
  - calls that drop: `own_err_call_drops_indirectly`, `own_err_call_given_borrow`,
    `own_err_call_this_borrow`, `own_err_exported_virtual_call`;
  - borrowed results: `own_err_borrowed_result_stored`, `own_err_borrowed_result_outlives`,
    `own_err_borrowed_result_let_outlives`, `own_err_borrowed_result_escapes`,
    `own_err_borrowed_result_family`;
  - kept parameters: `own_err_param_kept_used`, `own_err_param_kept_some_paths`,
    `own_err_param_kept_loop`, `own_err_param_kept_not_owned`;
  - `any`: `own_err_any_unboxed_outlives`.

Corpus (`test/tester/tests/*.ts` under `-mm=own --no-default-lib`): 272 of 564 compiled before,
303 after, and none stopped. "Takes a second reference" fell from 189 files to 149. About half of
the new ones throw or catch (the copy thunks); several more cast out of `any`. Release suite: 3226
of 3226.

### 15.8 Export and import of `__own_no_drops`

A library built under own writes, beside its marker `__tsmm_own_<module>`, a second exported string
`__tsown_<module>`. It lists the exported functions that destroy nothing a caller can reach, one
name per line, by the name they are exported under. The signature pass writes it: it is a copy of
the marker global under another name and value. So it is exported exactly as the marker is on
every platform, and MLIRGen emits the same under own as under rc (`test-own-mlirgen-matches-rc`).

The importer reads it in `mlirGenImportSharedLib`, beside the marker, from the loaded library or
from the file. It keeps the names on its module as `ts.own_imported_no_drops`. That happens in every
model, for the same reason, and only own's signature pass reads it. A call resolves to such a
function in either of two forms:

- a declaration linked through the import library;
- `ts.Load(ts.AddressOf @f)`, where the global `f` is `SearchForAddressOfSymbol("<name>")` (a
  library loaded at run time).

The call is then known and drop-free, like the `.instanceOf` slot. A virtual call to a method of
an imported class stays unknown, because an override may be defined in the importer.

Only this fact crosses, and only in one direction of trust:

- A missing `__own_no_drops` only makes the importer report more. So a module that links the
  library statically is sound without it: it re-parses the library's source and sees no facts.
- `__own_params` and `__own_result_borrows` change the callee's body: its retains are gone. An
  importer that does not know them releases a moved argument or frees a borrowed result. The static
  importer above is such an importer, and under own there is no counting to adapt a call with.

Tests: `test-jit-own-shared-no-drops` and `test-compile-own-shared-no-drops` run
`import_own_no_drops.ts` against `export_own_no_drops.ts` as a `-shared` pair. The importer holds a
borrow of a parameter's field across calls to `total` and `M.first`. On Windows,
`own-shared-no-drops.cmake` also checks two things. `import_own_err_imported_drops.ts` is still
rejected, because `shrink` removes an element and is not listed. And the same program is rejected
against the library built under rc, which lists nothing: that is the fact's teeth.

Until the fix below, test-runner's `-shared` mode passed `-mm=` to the library only, and the
program was built under gc. Every `-shared -mm=rc`, `-mm=none` and `-mm=own` test ran a mixed
link. So, before it, the two runner tests above showed only that the pair works; the Windows
script was the one checking the fact. The flag now goes to every file, and all 302 `-shared`
tests pass with it.

**Objects of an imported class.** `new H()` of a class from a library builds its object with the
library's `H..new` and runs `H.constructor` through a method bound to it:
`ts.CreateBoundFunction(ts.Cast(h to !ts.opaque), ctor)`, split back into `ts.GetThis` and
`ts.GetMethod` for the call. The cast read as taking `h`, so every later use of `h` was "used after
its value was moved". The cast is now a borrow when it has only that use, and when the object comes
out again only as a call's argument, as `ts.ThisSymbolRef` is for a method of this module.

The object is also fresh, and owned here, when the library was built under own. Its `..new` is
listed among the library's `__own_no_drops`, and the signature pass marks such a call
`__own_fresh_result`, which `isFresh` reads. Nothing else makes an imported call's result fresh.
A library built under rc lists nothing, and its blocks carry a count their own module holds. There,
a borrow under the object still ends at any unknown call. `import_own_class.ts` and
`export_own_class.ts` run as a `-shared` pair under every model, AOT and JIT. The Windows script
checks that the program is rejected against the rc library.

### 15.7 Known limits (the input to phase 5 and later)

- An exported function, or a method of an exported class, gets no owned or borrowed facts (§15.8
  says why they cannot be exported), and a virtual call on one gets no `__own_no_drops`. That is
  sound and restrictive for `-shared` modules. The exception is a call on an object of another
  module's `@final` class. It still goes through the vtable, but no class can extend the class, so
  the slot holds the method the call names. MLIRGen marks its `ts.ThisVirtualSymbolRef` (an
  accessor's `ts.VirtualSymbolRef`) `__final_receiver`, and the signature pass resolves the call
  as one to that imported function, so its `__own_no_drops` applies. A record (§32) can then be
  read across such a call (`import_own_record.ts`), but not across one the library does not list
  (`import_own_record_final_drops.ts`), nor across a method call on a class that is not `@final`
  (`import_own_record_open_method.ts`).
- A value an importer under own receives from a library built under rc (a call's result, now an
  object it builds with `new` too) is destroyed at the end of its owner's scope. Own's release
  skips only immortal blocks. So if the library kept a reference of its own, that is a
  use-after-free, where the mixed-link policy promises a leak.
- A parameter kept on some paths only (needs drop elaboration: a release on the others).
- A borrowed result's places are a wildcard, so any field overwrite between the call and the use
  drops it, even of an unrelated object.
- Interface calls (`ts.InterfaceSymbolRef`) are unresolved: no facts, and they drop.
- A virtual family is matched by method name and index, so an unrelated class's method at the
  same index can only take facts away.
- The three corpus files §14.4 lost (`export_class_abstract*`) stay rejected. Their class is
  exported, so the virtual call to `area` has candidates this module cannot see, and it may drop
  (`own_err_exported_virtual_call` is the same shape). The error does not yet say why: the note
  on lost facts is attached to escapes and second references, not to drops.
- From §14.6: a read reached through an interface, borrow chains through assigned borrowers,
  moves after the last use of every borrower, and a temporary taken by a `let` and consumed
  elsewhere.

## 16. Phase 5a results, 2026-10-02

Plan: `docs/superpowers/plans/2026-10-02-own-phase-5.md`. Changed: `OwnershipInferencePass.cpp`,
`OwnershipSignaturePass.cpp`, `OwnershipFacts.h`, and in the lowering the capture box's
descriptor and `ts.ReleaseCell`. MLIRGen does not change.

### 16.1 What phase 5a accepts

A closure is `ts.CreateBoundFunction(box, @f) {__owns_capture}`. Its box holds a cell for each
variable captured by reference and a copy of each `const` captured by value. The pass decides each
closure over at least one cell before anything else, because the verdict changes what the box's
stores are.

- **Escape.** The closure value is followed through its views (a bound function and a hybrid one are
  the same `{func, this, tag}`, so the cast between them is now a view) and through the locals that
  hold it. It does not escape when every use is rc's bookkeeping, a call given it (not kept), or a
  call through it (`ts.GetThis`/`ts.GetMethod`, or the devirtualized `ts.SymbolCallInternal(box)` of
  a folded `const f`). Anything else is an escape: a return, a store into a field, element, global or
  another box, a kept argument. An escape is `a closure that captures a variable and escapes here is
  not supported by -mm=own yet` (5b).
- **A closure that does not escape borrows everything its box holds.** rc's `ts.RetainCell` of each
  cell and `ts.Retain` of each copy go. The box's stores of copies are reads, not takers
  (`__own_capture_borrow`, this pass's own mark). The closure carries `__own_borrows_captures`, and
  its tag names a routine that frees the box and nothing else (`tsfrecb_`).
- **The closure's uses** are every call given it, called through it or through its box. None may run
  after an end of what it borrows: the `ts.ReleaseCell`s of a cell this function owns, and the
  releases and takers of a copy's owner (rootEnds of a fresh value or an owning local). A cell an
  enclosing closure holds, and a parameter, end nowhere here. A copy of a read out of a container is
  bounded by the read's own check: walkBorrowed adds the closure's uses to the read's.
- **A local that owns nothing** (`let f: () => number; f = () => ...`) is an alias of the closure.
  rc's birth reference reads to the move logic as the store's, which erased the closure's only
  release. Now the closure keeps its releases, and nothing may run it after them. That holds for a
  closure over copies only too, whose box otherwise owns its copies as in phase 4.
- **A captured parameter's cell** (`this` included) borrows the caller's value. Its `ts.ReleaseCell`
  carries `__own_cell_borrows` and frees the cell only. Nothing may assign the variable: not the
  frame, and not a closure body. The signature pass pins `__own_assigns_captures` on each closure:
  the box fields whose cell its body assigns, followed through the closures it builds over the same
  cell. The inference reads it, because a per-function pass may not look into another function. A
  non-owning cell with a heap value that is not a parameter's is
  `'x' is captured by a closure, but -mm=own cannot tell who owns its value here`.
- **Cells are places.** `ts.ReleaseSlot` of a cell, a captured `ts.Variable` in the frame or a cell
  read out of a box in a closure body, is a drop of every chain whose root this function does not own.
  A read of a cell is such a root. In the signature pass the closure-body form makes a function drop.
- **A call given a closure may run it.** A call whose operand is a closure value, a capture box, or a
  closure's `this` drops every chain when its callee may drop: the box may hold anything. Before, a
  call given only the box was not "given" what the box held (`derivedFrom` follows results, and a
  store has none).

### 16.2 Measured

AOT, `measure.ps1`, in MB: `own_closure_borrow` (thirteen shapes, 20000 rounds) reads gc 6.1, rc 4.3,
none 1879.7, own 4.3.

### 16.3 Teeth

- With the owning routine for a borrowing box, `own_closure_borrow` fails under AOT (0xC0000005): the
  closure in `innerBlock` is a `let` released at its block's end, and its box then destroys the cell
  the frame still reads.
- With `__own_cell_borrows` ignored, it fails under AOT (0xC0000374): the cell of `param`'s `c` and of
  `D.read`'s `this` free the caller's objects, which `main` reads after a churn.
- Under the JIT both pass: LLVM folds the reads of freed memory. The AOT runs are the evidence.
- Each new negative compiles with its rule switched off. `own_err_closure_param_assigned` then
  crashes (0xC0000374).

### 16.4 Tests and the corpus

- **Positive:** `own_closure_borrow` (a local, a copy beside a cell, an argument, a loop, nested, an
  inner block, `this`, a parameter, the frame and the closure assigning the variable, a `let` holding
  the closure, a `let` aliasing it, a `let` aliasing a closure over copies), under JIT, AOT and the
  no-counting check.
- **Negatives:** `own_err_closure` (now: escapes), `own_err_closure_outlives_cell`,
  `own_err_closure_copy_moved`, `own_err_closure_param_assigned`, `own_err_closure_assigns_param`,
  `own_err_cell_assigned_in_frame`, `own_err_cell_assigned_in_closure`,
  `own_err_closure_assigns_captured_field`.
- The capture box's descriptor has no retain routine under own, as no other descriptor does: it would
  bring `__tslang_inc_ref` in.

Corpus under `-mm=own --no-default-lib`: 310 of 591 before, 319 after, none lost. The nine new ones
(`00arrow_generic`, `00capture_in_new_arguments`, `00funcs_expression_generic`,
`00funcs_generic_arrow`, `00funcs_nesting_capture`, `00stack_test`, `22lambdas`, `241arrayforeach`,
`25lamdacapture`) pass under test-runner `-mm=own`, AOT and JIT. Of the 25 files the old error
stopped first: 9 compile, 4 escape (5b), 8 still meet a cell no closure claims (generators, which
are phase 7, and object literals whose methods capture), 4 stop at an earlier limit.

### 16.5 Known limits (the input to phase 5b)

- A closure over a cell that escapes (§2.5's second half): 5b. A closure over copies only already
  owns them, as in phase 4, escaping or not.
- A copy of a borrowed value captured by a closure that does not escape works only when the borrow's
  own check passes; a copy of a borrow into an escaping closure is an error.
- A captured local cannot be borrowed by a `let` or moved out of (slotLoadOf and borrowerOf exclude
  captured variables).
- A non-escaping closure always borrows: one that could own its captures instead (called after the
  cell's block ends, the variable not used again) is rejected.
- A call given any function value drops every chain when its callee may drop.

## 17. Phase 5b results, 2026-10-02

Plan: `docs/superpowers/plans/2026-10-02-own-phase-5.md`, task 4. Changed: `OwnershipInferencePass.cpp`,
and, for the bug in §17.2, `OwnershipFacts.h` and `OwnershipSignaturePass.cpp`.

### 17.1 What phase 5b accepts

A closure over a cell that escapes (§16.1's classification) owns what its box holds. Its box keeps
today's routine (`tsrelcb_`), which destroys each cell with what is in it, and each copy.

- **Copies** move in as in phase 4: the box's store is a taker, and the copy's owner gives it up there.
- **Each cell moves in** at the store that fills the box: rc's `ts.RetainCell` for it goes, and so
  does each of the frame's `ts.ReleaseCell`s the move reaches. The move must dominate them:
  `'a' is moved here on some paths only` otherwise (a closure pushed in one branch of an `if`).
- **Nothing in the frame may use the variable after the move.** That covers a read, an assignment
  and another capture: `'a' is captured by a closure that escapes, and is still used here`. Two
  closures that escape cannot share a variable. This is §2.5's "any use after the closure's
  creation", measured from the box's store.
- **A move round a loop** of a cell made outside it is the loop rule (§2.6).
- **No move is possible** for a parameter's cell with a heap value, whose value is the caller's, or
  for a cell an enclosing closure holds: `a closure that escapes owns what it captures, but -mm=own
  cannot move this variable into it: ...`. A parameter's cell holding a number moves: it owns
  nothing but itself (`makeScaler(k)`).
- **A borrowing closure over the same cell** must be done with it before the move: the move into an
  escaping box is an end of the cell, beside its `ts.ReleaseCell`s.
- **A call through the box** (a folded `const f` called before `return f`) may not come after the
  closure escapes or is released.

### 17.2 A call inside a try body was not a call

A call in a try body lowers to `ts.Invoke`, or to `ts.InvokeHybrid` through a hybrid function: a
terminator whose normal and unwind edges are its successors. Neither pass knew the op. So such a
call dropped no borrow: a read of `h.c` held across `reset(h)` inside a `try`, with `reset`
overwriting `h.c`, compiled and printed 999 (gc: 1). It also resolved to no callee and returned nothing
fresh. Found when `51exceptions.ts` reported its on-the-spot closure call in a `try` as an escape.
`directCallee`, `calleeValue` and `callArgs` now take every call form apart, and `isCall` and
`isFresh` know both ops. Test: `own_err_call_in_try_drops`.

### 17.3 Measured

AOT, `measure.ps1`, in MB: `own_closure_escape` reads gc 5.9, rc 4.3, none 234.6, own 4.3.

### 17.4 Teeth

With the frame's `ts.ReleaseCell`s kept after a move, `own_closure_escape` fails under AOT
(0xC0000005). The JIT folds the reads of freed memory, as in §16.3.

### 17.5 Tests and the corpus

- **Positive:** `own_closure_escape` (over an object local, over a number parameter, a counter, a
  copy beside a cell, called through the box before it is returned, kept by a constructor, pushed
  into a global).
- **Negatives:** `own_err_closure_escape_used`, `own_err_closure_escape_twice`,
  `own_err_closure_escape_param`, `own_err_closure_escape_loop`, `own_err_closure_escape_inherited`,
  `own_err_closure_escape_some_paths`, `own_err_call_in_try_drops`. `own_err_closure` (a counter that
  escapes) compiles now and is gone.

Corpus: 319 of 591 before, 325 after, none lost. `13actions` comes from the escapes. `51exceptions`
and four `using` files (`00disposable`, `00try_catch_return_dispose`, `00using_nested_scopes`,
`01try_catch_return_dispose`) come from §17.2: a call in a try body returns a fresh value now. All six
pass under test-runner `-mm=own`, AOT and JIT, and their JIT output matches gc's.

Found, not fixed (every model, MLIRGen): pushing a closure that returns `s32` into a
`(() => number)[]` segfaults the compiler. For example, `let n = 0; h.push(() => n)`, since an
integer literal is `s32`.

### 17.6 Known limits

- The capture is the move, so a closure created on one path only, with the variable alive on the
  other, is an error (no drop elaboration).
- A frame use between the capture and the escape is rejected, though the box is still alive there.
- A closure that escapes from a closure body with a cell it inherited, and a closure over a
  parameter's object, are errors. `this` captured by an escaping callback is the common case.
- From §16.5: a non-escaping closure always borrows; a captured local is never `let`-borrowed.

## 18. Phase 7a results, 2026-10-02

Plan: `docs/superpowers/plans/2026-10-02-own-phase-7.md`. Changed: `OwnershipFacts.h`,
`OwnershipInferencePass.cpp`, `OwnershipSignaturePass.cpp`, and `tslang/transform.cpp` (the
inference runs on `func.func` too). MLIRGen does not change.

### 18.1 What phase 7a accepts

A generator's maker builds its initial state as a `ts.Constant` tuple in a local, copies it into a
block `ts.New` makes, and returns that block cast to `!ts.object`. Its `next` keeps the
generator's locals in the state object's fields and returns a `{value, done}` record. The caller
calls `next` through the state object.

- **The state object is a fresh block.** The cast of a `ts.New`'d `value_ref` to its `!ts.object`
  is a view. Inside the pass such a block owns (`ownsHeapMemory` says no for any `value_ref`), so
  its root is checked, and the store that fills it is a borrow, not a taker.
- **The initial state holds nothing.** rc's retain of the tuple is erased when its owning fields
  hold what the constant put there: null, a number, an immortal literal.
- **Locals are fields.** A store into one is a move into the container, a read a borrow bounded
  by it, and reassignment releases the old value (rc's `ts.ReleaseSlot` of the field). Nothing new
  was needed.
- **A yield moves its value to the caller.** A record built to be returned (a local a
  `ts.Constant` initializes, read once, the read only returned) takes the fresh values stored into
  its fields: each store is the value's taker, and rc's retain of the filled record is where the
  value moves in. A plain function returning `{ value: new C(), done: false }` is the same shape.
- **The caller's `next` resolves.** The signature pass resolves a function read out of an
  object's field when the field only ever holds that function: every `ts.Constant` of the
  object's tuple type holds it there, nothing stores into the field (a method sees its object as
  `object_storage`, whose fields count too), and no other op makes a value of the type. The call
  gets `__own_fresh_result`. `next` is open (its symbol sits in a constant), so its body may not
  return a borrow.
- **What a generator captures is an escaping closure's box.** The maker puts the cells of its
  parameters into a box and the box into the state's `.captured` field. That store is the box's
  escape, and phase 5b's rules apply as they are. A cell holding a number moves in
  (`range(from, to)`), and a parameter's object is 5b's "its value is the caller's" error. An
  object literal whose methods capture keeps its box the same way.
- **Async.** The async lowering outlines a `for await` body and each `await`'s continuation into a
  `func.func`, which the inference did not visit. A retain there was neither erased nor reported,
  and the lowering failed with "ownership inference left a retain behind" (`00for_await_yield`).
  The inference now runs on every function. An outlined body's arguments are the values it
  captures: a retain of one is an error. The signature pass collects the calls in those bodies
  too. It did not: a callee that keeps its argument (`stash(c)` pushing into a global) moved it
  in its own body, while the `for await` body that called it kept its release of `c`, a use after
  free on `main` as well. Verified in the IR only: the release is gone. A run cannot show it,
  because `await f()` returns before `f`'s `for await` body has run (§18.5).

### 18.2 An interface over an object is that object

The view above showed that `ts.NewInterface` gave a block a second owner. The object-literal
interface cast marks it `__owned_result`, and `isFresh` took the mark at its word.
`let raw = {...}; let i = <I>raw` freed `raw`'s block twice. On `main` already, a folded object
literal given twice as an interface (`use(o); use(o)`) is freed twice: it crashes there. A first
fix, fresh only when the object has no other use, lost 12 corpus files that compile on `main`;
the view below gets them back.

`ts.NewInterface` is now a view of its object, as a union or an optional is, and a view is as fresh
as its root:

- `let i = <I>raw` borrows `raw`. rc's `ts.Retain` of the interface is the reference of the `let`
  that consumes it, and a borrow erases it.
- A class instance behind an interface is one block, so the phase-0 placeholder
  `own_err_interface` compiles and is gone.
- Two releases of one block, one reachable after the other, are two owners: the error at the
  second view (`own_err_interface_two_views`).

### 18.3 Measured

AOT, `measure.ps1`, in MB:

| test | gc | rc | none | own |
|---|---|---|---|---|
| `own_generator` | 6.1 | 11.2 | 1548.6 | 4.3 |
| `own_record_return` | 5.9 | 4.2 | 224.0 | 4.2 |
| `own_interface_view` | 5.9 | 4.3 | 551.1 | 4.3 |
| `own_async` | 6.1 | 5.7 | 334.7 | 5.8 |

rc is above gc on `own_generator` because it leaks yielded objects (§18.5).

### 18.4 Teeth

Each made with a temporary `getenv` switch:

- Keep the frame's `ts.ReleaseCell` of a cell after it moves into a generator's box:
  `own_generator` fails, AOT (0xC0000374) and JIT.
- Keep a value's temporary release after it moves into a returned record: `own_record_return`
  fails, AOT (0xC0000374) and JIT. In a generator's `next` there is no such release to keep
  (§18.5).
- Skip the released-twice check: `own_err_interface_two_views` compiles and crashes.
- Let a field that is assigned another function resolve: `own_err_field_function_reassigned`
  compiles.

The JIT's object cache (`__jit` beside the source) keys on the compiler binary and the options, not
on the environment. Every JIT run with the teeth build after the first reused the first one's
object, so the cache was cleared between them.

### 18.5 Tests and the corpus

- **Positives:**
  - `own_generator`: numbers, a class local, a local reassigned between yields, a generator given
    up after one value, yields of fresh objects and strings through `for...of` and by hand, number
    parameters, an object literal with a method;
  - `own_record_return`;
  - `own_interface_view`: a class instance as an interface, a `let` borrowed through one, a
    literal given once, a literal made straight into one;
  - `own_async`: a heap local across an `await`, a fresh result, a heap parameter after an
    `await`, a string, a `for await` body.
- **Negatives:** `own_err_interface_two_views`, `own_err_field_function_reassigned`,
  `own_err_generator_param_object`.
- The no-counting check's failure pattern is `error:` now: `error` alone matched the async
  runtime's own message, "Awaited async operand is in error state".

Corpus: 325 of 591 before, 359 after. Every file that newly compiles runs under test-runner
`-mm=own`, AOT and JIT, and so does every `-shared` pair whose two halves both compile. The
JIT output of each matches gc's, except the two async files (`00for_await`, `00for_await_yield`),
which the JIT runs only with the async runtime test-runner links: they pass under test-runner. Two files that compiled on `main` are errors now, both correctly:

- `00object` stores one object literal into two tuples, which rc leaks.
- `00interface_object_array` reads an object after it moved into an array.

Found, not fixed:

- **A temporary made in front of a `yield`** in the generator's own body is never released, under
  rc as well. `OwnedReturnConsumptionPass` does not release past a resume point, so
  `yield "s" + i` leaks the number's string (50 MB over 200000 runs). *Fixed (#619 and its
  follow-up):* a temporary is released in front of the first yield, resume point or statement
  holding one after it.
- **rc leaks yielded objects.** The call's reference and the record's retain are two, and the
  caller releases one. *Fixed (follow-up to #619):* under rc a yield retains its record the way a
  return does, `next` is classified as returning owned, and its caller takes the record over
  through the state object's field that only ever holds `next` (`own_facts::FieldFunctions`).
- **Reading a record out of a call or a folded `const`** (`print(f().value.x)`,
  `const r = f(); r.value.x`) is an error. A `let` works.
- **A method of an object literal held in a `let`**, returning a block, called and kept, is a
  borrow error: `let o = {make(): C {...}}; const c = o.make()`. A `const o` works.
- **`measure.ps1` reads `compile.bat`**, which only a test-runner run without `-mm` writes.
- **An `await` of a function that returns nothing never awaits** (#440, every model): MLIRGen
  exits before building the `async.await` when the awaited call has no value, so `await f()`
  continues at once and `f` may never run. That is what made the `for await` facts above
  impossible to test at run time.

### 18.6 Known limits (the input to 7b)

- A generator over a parameter's object needs its result to borrow the argument, and a box that
  frees its cells but not their values.
- A nested generator over an outer local inherits the cell: 5b's "captured from an enclosing
  function" error.
- `.map`/`.filter`, which MLIRGen builds as generators over the source array and the callback, stop
  on "borrows a field and cannot be stored".
- A yielded borrow (a parameter's element, a state field's value) is an error.

## 19. Phase 7b results, 2026-10-02

Plan: `docs/superpowers/plans/2026-10-02-own-phase-7b.md`. Changed: `OwnershipSignaturePass.cpp`,
`OwnershipInferencePass.cpp`, `OwnershipFacts.h`, `Defines.h`, and `OwnershipRoutineLogic.h`
(the lowering).

### 19.1 What phase 7b accepts

A generator over a parameter that holds a block, `function* each(a: number[])`, borrows it: Rust's
`fn each<'a>(a: &'a [f64]) -> impl Iterator + 'a`. The caller owns the state object, which may not
outlive the argument, and whose release gives the argument back to nobody.

- **The decision is the signature pass's.** The release routine is built once per type, and the
  inference is per function and parallel, so neither can make it. A state type is unique to its
  generator: its `next`'s type names a per-site `object_storage`. For a state type whose maker is
  closed, and is the type's only `ts.New`:
  - the `.captured` box borrows when every fill is a parameter's cell, or a copy read out of a
    parameter's field (`.map`'s maker is a closure body that copies its box);
  - a state field borrows when everything stored into it, in the whole module, is a read of a
    borrowing cell or field (a least fixpoint). The `for...of` copy of the array is such a field.

  It writes `ts.own_borrowing_fields` on the module (per type, the box's index first), and pins
  `__own_result_bounded`, the parameters whose blocks the state borrows, on the maker and its calls.
  A maker is closed when it is private and its symbol is used only by direct calls, and by closures
  that are only retained and released (`.map` builds one and calls its body through the box).
- **The lowering**, under own, skips a borrowing field in the state object's release routine. A
  borrowing box gets `tsrelcbc_`, which frees each cell and none of what the cells or by-value
  fields hold.
- **In the maker**, a parameter's cell moves into a borrowing box: no 5b error, and the frame's
  `ts.ReleaseCell` goes. A copy stored into a borrowing box is a borrow.
- **In `next`**:
  - a store into a borrowing field takes nothing: rc's retain goes, and so does its release when
    the field is overwritten;
  - what is read out of a borrowing cell, field or box field is the caller's, and may only be
    borrowed. Yielding it is an error (`own_err_generator_yields_borrow`): it would give the
    caller's value a second owner, and rc retains nothing there for anything else to see;
  - a read of a borrowing box's field is not a place read: only the maker fills the box, so no call
    overwrites it.
- **At the caller**, the generator is owned as any fresh value, and bounded:
  - no use of it may come after an end of a bounding argument. Its uses are followed through the
    locals that hold it and the `next` it is called through. An end is the argument's root's
    releases, assignments and moves, or, for a read out of a place, what may destroy the place.
    Its releases are not uses: the routine gives back nothing it borrows;
  - it may not be stored, returned or captured (`own_err_generator_bounded_escapes`);
  - a bounding capture box (`.map`) ends where the closures on it are released, and where what it
    was filled from ends (`own_err_generator_map_outlives`).
- **`.map` and `.filter`.** The caller's closure box holds a copy of the array. A box of copies
  that does not escape, is given to a bounded maker, and holds nothing fresh borrows its copies, so
  two `.map`s of one array no longer move it twice. A callback that captures is a fresh closure
  made for the box: the box then owns its copies, and the array moves in. rc reads the array twice
  for one copy and retains the first read: the second is the same value, both where a taker is
  looked for and where a box field's retain is.

### 19.2 Measured

AOT, `measure.ps1`, in MB:

| test | gc | rc | none | own |
|---|---|---|---|---|
| `own_generator_borrow`, 20000 iterations | 6.1 | 4.3 | 1441.5 | 4.3 |
| the same, 200000 iterations | 6.1 | 4.3 | 14376.5 | 4.3 |

The ten-times run is the leak check. It found one in this phase's own work: the first form of the
copies rule borrowed a fresh capturing callback, which nobody then freed (own 10.5 MB, rc 4.3).

### 19.3 Teeth

Each made with a temporary `getenv` switch, the JIT cache cleared between runs:

- **The owning box routine for a borrowing box:** `own_generator_borrow` fails, AOT
  (0xC0000374) and JIT.
- **A borrowing field released:** the same.
- **The bounded check skipped:** `own_err_generator_outlives_arg`,
  `own_err_generator_bounded_escapes` and `own_err_generator_map_outlives` compile.
- **The borrowed-read check skipped:** `own_err_generator_yields_borrow` compiles.

### 19.4 Tests and the corpus

- **Positive:** `own_generator_borrow`. It covers an array, a class and a string parameter, a
  generator given up early, two generators over one argument, an argument read out of a field,
  `map`, `filter`, and a capturing `filter` last.
- **Negatives:** `own_err_generator_outlives_arg`, `own_err_generator_bounded_escapes`,
  `own_err_generator_yields_borrow`, `own_err_generator_map_outlives`,
  `own_err_generator_place_overwritten` (the field the argument was read from is assigned).
- **Removed:** `own_err_generator_param_object` compiles now and is gone.

Corpus: 359 of 591 before, 364 after, none lost: `00funcs_generic_iterator`, `01disposable`,
`02disposable` (an object literal whose methods capture a parameter is the same shape),
`00filter`, `00class_iterator_super`. Each runs under test-runner `-mm=own`, AOT and JIT.

Found, not fixed (every model): `a.map(f).map(g)` fails to compile ("the condition has no value").

### 19.5 Known limits

- **A yielded borrow** (an element of a parameter's array, the parameter itself) is an error. The
  result record type `{value: C, done}` is shared between generators, so a fact per type cannot
  say that a record borrows.
- **A capturing `.map`/`.filter` callback** moves the array into the closure's box: a later use of
  the array is a use after move.
- **A bounded generator cannot be returned** past the argument it borrows, even when the
  argument is the returning function's own parameter (the fact does not propagate).
- **A copy a borrowing field takes of a sub-field** (`const v = c.v` in the generator) is not
  borrowing, so it is the existing "cannot be stored" error.
- **Without `--di`**, the bounded errors name "this value".

## 20. Phase 6 results, 2026-10-02 (first PR)

Plan: `docs/superpowers/plans/2026-10-02-own-phase-6.md`. Changed: `MLIRGenImpl.h` (#444, every
counting model), `OwnershipSignaturePass.cpp`, `OwnershipInferencePass.cpp`, `OwnershipFacts.h`.

### 20.1 The report (spec §6.5)

The tooling stays out of the tree:
- a script that compiles every corpus file under own and keeps the first error;
- a temporary dump, under `OWN_SHAPES`, at the four places that report the catch-all. It prints the
  retained value's definition, its root, its users, its type, whether it owns a block, the source
  location, and for a read of a slot, the slot's kind and what takes the value.

On ad8ea908 (7b merged), 364 of 591 compile. The first error of the 227 that do not:

| first error | files |
|---|---|
| `takes a second reference` (the catch-all, which names no rule) | 137 |
| used after its value was moved | 22 |
| borrows a field and cannot be stored, returned or captured | 21 |
| borrows an element and cannot be stored, returned or captured | 9 |
| `ownership inference left a retain behind` (the lowering's guard) | 8 |
| the rest: other named rules, unsupported, frontend | 30 |

The 137, by the dump: 447 sites. The last column counts the files where the bucket is the only
blocker:

| shape | sites | files | only blocker |
|---|---|---|---|
| a read of a slot (mixed) | 97 | 39 | 17 |
| a call's result stored: script-mode globals | 59 | 28 | 14 |
| tuple literals | 52 | 23 | 13 |
| the value owns no block (`number \| null`) | 35 | 23 | 9 |
| a block argument (`c ? a : b`) | 18 | 11 | 7 |
| a function literal | 24 | 13 | 6 |
| a read of a global | 22 | 8 | 4 |
| a number, bool or bigint converted to a string | 42 | 11 | 0 |
| other | 55 | 34 | 18 |

### 20.2 What this PR accepts

- **A global declared in top-level code (#444, rc).** `const t = new C()` there stored into the
  global with no reference, and the end-of-block release gave the value's back: 5ai's bug, in
  the declaration's store, which 5ai's fix (the assignment's) did not reach.
  - The fix: the declaration's store takes the owning-slot path, as an assignment does.
  - It is not observable under rc today. Nothing runs after a program's top-level code, and a
    module with top-level code cannot be imported (§20.5).
  - Under own, the store read as a move and the release as a second reference.
- **A global's initializer** runs in the global's region, which no function's inference sees,
  so rc's retain for its `ts.GlobalResult` reached the lowering's guard. The signature pass now
  erases such a retain when:
  - the value is the only reference to its block: retained once, released nowhere in the region;
  - it is fresh, constant data, or a record of constants;
  - every use of it ends in the `ts.GlobalResult`: directly, through views, as an element of an
    array literal, or copied into an object literal's box.

  Any other retain there is reported by the global's name.
- **Data nothing owns** (`isConstantData`, `OwnershipFacts.h`): what holds no block (a number or
  null in a union), a `ts.Constant` or a field of one, a function named by a `ts.SymbolRef` and
  its cast to a function value, and a tuple of such.
  - A retain of it is erased.
  - A `ts.RetainSlot` is decided on its slot's initializer, so only while nothing is stored into
    the slot or a field of it, and no closure captures it.
  - Such data has no use after a move: under `--opt`, CSE merges equal literals (the left-out
    optional field of two tuple literals), and one value goes into several places.
  - Two exclusions, from a type-only reading. The header's `ownsHeap` reads the type, and the
    inference pass overrides it for a `ts.New` seen as its object; the two disagree by design.
    - A `ts.New` is excluded: its `value_ref` owns nothing by type, but it is the object. Without
      this, `00object` compiled again: one object literal stored into two records, which 7a
      rightly rejects.
    - A value a `ts.NewInterface` view boxes is excluded: each view is a block of its own.
- **A record literal made for an owning local**, `let a = { item: new C() }`, and its assignment
  form. MLIRGen builds it in a local a `ts.Constant` starts, stores the fresh value into the
  field, and reads the record once into `a`. That is 7a's record built to be returned, with an
  owning local in place of the result slot: the fresh value moves into the record, the record
  into `a`, and `a`'s `ts.RetainSlot` is the reference for what moved in.

### 20.3 Measured and teeth

AOT, `measure.ps1`, a loop that builds and drops `let a = { item: new Leaf(i) }`, in MB:

| iterations | gc | rc | none | own |
|---|---|---|---|---|
| 20000 | 5.9 | 4.2 | 7.7 | 4.2 |
| 200000 | 5.9 | 4.2 | 35.2 | 4.2 |

Teeth, each a temporary switch, with the JIT cache cleared:
- **The global declaration's retain reverted** (#444): `own_global_script` fails under own (AOT,
  JIT, no-counting). rc passes either way, as §20.2 says.
- **The global initializer's erase skipped:** `own_global_init` fails.
- **The constant-data erase skipped:** `own_constant_data` fails.
- **Its no-use-after-move skipped:** `own_constant_data` fails under the runner's `--opt`.
- **The `ts.New` exclusion dropped:** `own_err_literal_two_records` compiles.
- **The boxing exclusion dropped:** `own_err_interface_two_views` still fails. It compiles only with
  both exclusions dropped, so for that test each exclusion is a second guard for the other.
- **The record rule's two acceptances dropped:** `own_record_local` fails.

### 20.4 Tests and the corpus

- **Positive:** `own_global_script` (also under rc, none and gc), `own_global_init`,
  `own_constant_data`, `own_record_local`.
- **Negative:** `own_err_literal_two_records`.

Corpus: 364 → 402 of 591, none lost. It is 402 with the runner's `--opt --opt_level=3` as well;
the plain count can overstate (`01optional` compiled plain and failed under `--opt` before the
no-use-after-move rule). Each of the 38 passes test-runner `-mm=own`, AOT and JIT, with gc's
output, re-gated on the final build. The first errors after:

| first error | files |
|---|---|
| `takes a second reference` | 101 |
| used after its value was moved | 24 |
| borrows a field and cannot be stored, returned or captured | 21 |
| borrows an element and cannot be stored, returned or captured | 9 |
| `left a retain behind` | 0 |

Remaining shapes, by the dump, as the files where each is the only blocker:
- slot reads, 25. They split into:
  - a record a `ts.Constant` starts, stored or taken;
  - a parameter's value stored, pushed or cast;
- other, 24;
- a conditional's result (`return a > b ? a : b`), 7;
- tuples, 7;
- nested record literals (`{ inner: { item: new C() } }`) and arrays of record literals.

These are the next PR's input: named messages for what the dump recognises, and the sampling
of the use-after-move and borrows-a-field files for false positives.

### 20.5 Found, not fixed (every model)

- **A global initialized from another global reads garbage** (#446). `let a: number[] = [1, 2]; let b = a;`
  at module level gives `b.length` = 140711081380496 under gc. With a class, the program faults.
- **A module with a top-level statement, imported as source,** (#447) fails with "Global is referenced
  by parentless instruction! @puts" (invalid LLVM IR).
- **With `-shared`, an exported `const` from such a module** (#448) cannot be resolved by the importer.

### 20.6 Known limits

- **A borrow of data nothing owns past its owner's overwrite** is still an error:
  `let m = n; n = null; use(m)`, with `n: number | null`.
- **Nested record literals, and arrays of record literals,** are the catch-all.
- **A conditional's result** is the catch-all.

## 21. Phase 6 results, 2026-10-02 (second PR): named messages

Plan: `docs/superpowers/plans/2026-10-02-own-phase-6.md`, task 5. Changed: `OwnershipFacts.h`,
`OwnershipInferencePass.cpp`. Nothing outside own's two passes changes, so rc, gc and none emit
what they did.

### 21.1 The report

The same tooling as §20.1, rebuilt: the corpus script (now also run with the runner's
`--opt --opt_level=3`), and the `OWN_SHAPES` dump, applied only inside the catch-all branch.

On cd7c2840 (#445 merged), 402 of 592 compile, plain and with `--opt`. The catch-all is the first
error of 100 files. After this PR, 416 compile plain and 415 with `--opt`, none lost either way:

| first error | before | after |
|---|---|---|
| `takes a second reference` (the catch-all) | 100 | 25 |
| used after its value was moved | 24 | 23 |
| borrows a field and cannot be stored, returned or captured | 21 | 20 |
| borrows an element and cannot be stored, returned or captured | 10 | 10 |

The 25 left are a long tail: no shape is the only blocker of more than three files. Phase 6's exit
criterion (every bucket that is the only blocker of six or more files is accepted or named) holds.

The 13 files gained under `--opt` each pass test-runner `-mm=own`, AOT and JIT, with the runner's
default flags and with `-noopt` (`--di --opt_level=0`). `00interface_object3` compiles plain only
(§21.6), so it is not counted.

### 21.2 Named messages

`reportSecondReference` now classifies what it is given, from the retained value's root, in one
place, so all four places that report it get the same names. Each message has a negative test.

| shape | message | test |
|---|---|---|
| a function returns a borrow of a parameter on one path and something else on another | `'f' returns a borrow of its argument on some paths and another value on others; -mm=own cannot tell its callers whether they own the result`, with a note at the other return | `own_err_mixed_return`; `own_err_unbox_string` until §22, which copies the string and deleted it |
| a parameter's value, returned, where the callers cannot learn the fact (§15) | `'x' is a parameter, returned here, but -mm=own cannot tell the callers of 'f' that the result borrows it`, with §15's facts-lost note | `own_err_param_returned_exported` |
| a parameter's value, kept (stored, pushed, given away) | `'x' is a parameter, kept here, but -mm=own cannot make the callers of 'f' give it up`, with the note | `own_err_param_kept_exported` |
| a store into a parameter's slot | `'x' is a parameter and cannot be assigned under -mm=own yet` | `own_err_param_assigned` |
| a store into a local that owns nothing | `'x' owns nothing under -mm=own (it is declared without a value, or in a catch or finally clause), so it cannot be given a new value here` | `own_err_local_owns_nothing` |
| a value merged from branches or loop iterations (a conditional, `&&`, `\|\|`, `??`, a loop) | 'x' is merged from several branches (the operators, a loop); -mm=own cannot prove which one owns it yet | `own_err_merged_result` |
| a global's value, stored, returned or boxed | `'x' borrows the global 'g' and cannot be stored, returned or captured` | `own_err_global_returned` |
| a global's value read into a local | `'x' borrows the global 'g', but -mm=own cannot prove the global keeps its value while 'x' is used: a call or an assignment may overwrite it` | `own_err_global_let_call` |
| a captured variable's value, through its cell | `'x' borrows a captured variable and cannot be stored, returned or captured` | `own_err_captured_returned` |
| an object literal holding a value it does not own outright: a literal nested by value, or one holding a borrowed or `let` field when it is not seen as an interface (§32; since §27, before which any literal given anywhere but an owning local or the result was one) | `an object literal that holds a value it owns is moved only into a local or returned as it is; -mm=own cannot move it here yet` | `own_err_record_literal_nested` (nested by value), `own_err_record_literal_let_field` (a `let`'s value, not an interface); the interface-typed shapes are borrowing records (§32): `own_err_borrowing_record_*`, `own_literal_moves`, `own_borrowing_record_arg`; `own_err_record_literal_let` (§27) is gone |

The mixed-return message is the largest bucket (17 files). Most are two synthesized helpers:
- `___unbox<string>` (`<string>` of an `any`) returns the box's own string when it holds one, and a
  string made from its number, boolean or bigint otherwise;
- `___cast<A, string>` of a union does the same with the union's payload.

The others are user functions such as `return name` on one path and `return "Sorry, " + name` on
another.

The parameter messages name the parameter from its slot, which carries a name under `--di` only,
like the other messages (`test-own-err-names-the-variable`).

### 21.3 What this PR accepts

- **A string or an array widened to a union of it and null or undefined, or narrowed back**
  (`isView`). Such a cast is the same block: the lowering passes the pointer through, and an
  optional array narrowed out of its optional is the array or an empty one. A union with other
  members is not a view, since it can convert on the way out (a number printed into a string).
- **A view of a value that owns no block holds none** (`holdsNoBlock`). A union's number payload
  made into another union is a view, by `rootOf`, of the union it came from, but it is a number
  all the way. A `value_ref` is not taken as such a type: by type it owns nothing, but one
  `ts.New` made is the object (§20.2). A `ts.Retain` of such a value is erased. Without this, the
  number path of `widen(u: number | string): boolean | number | string` left its retain behind
  for the lowering's guard: a loud failure, not a hole.
  - The signature pass reads it too. A function that returns a number widened into a union
    (`mkN(v: number): number | P { return v; }`) no longer has a result that "borrows" its
    argument, so `00union_global_init` stops at a later error, a global boxed into `any`.
- **A local's declaration retain of constant data, whatever is stored later**
  (`slotHoldsInitializerAt`, replacing §20.2's "nothing is ever stored into the slot").
  - MLIRGen emits `ts.RetainSlot` only at a declaration (`takeOwnershipOfLocal`), right after the
    initializer. The rule checks that nothing between the two can write the slot: in their block,
    no op is given the slot, no call runs, and no op has regions.
  - So the retain takes only the initializer, and each later store is decided on its own.
  - `let y: number | null = null; y = 0` compiles now.
- **A field of constant data is constant data** (`isConstantData`): a tuple of literals re-made as
  a named tuple, `let pair: [name: string, age: number] = ["user", 10]`.
- **An array a spread builds** (`isBuiltArrayRead`). `const b = [...a, ...a]`:
  - MLIRGen makes a fresh array in a local of its own that owns nothing, pushes into it, and reads
    it once, when it is done, into the owning local it is made for.
  - The fresh array moves into the builder and then into that local, whose `ts.RetainSlot` is the
    reference.
  - The builder may only be filled (`push`, `unshift`), and only before the read.
- **A function-typed field of an interface, called** (`isBoundForCall`). `i.toString()` with
  `toString: () => string` binds `ts.ExtractInterfaceThis(i)` to the function read out of the field
  and splits the bound function straight back for the call. It is #410's shape with the interface's
  `this` in place of a cast to `opaque`: the call borrows the object.

Two rules were tried and dropped, because no corpus file and no program written for them needed
them: `ts.SafeCast` as a view, and a number cast to a union as holding no block.

### 21.4 Tests and teeth

- **Positive:** `own_nullable_view`, `own_declared_constant`, `own_spread_build`,
  `own_interface_field_call`. Each passes test-runner under own, rc, none and gc, AOT and JIT, and
  under own with `-noopt`.
- **Negative:** the eleven in the table of §21.2.

Teeth, each a temporary switch: the positive stops compiling, plain, with `--opt` and with `--di`.

| rule turned off | fails |
|---|---|
| the nullable view | `own_nullable_view` |
| a view of a value that owns no block | `own_nullable_view` |
| the erase of a retain of such a value | `own_nullable_view` |
| a field of constant data | `own_declared_constant` |
| the declaration retain | `own_declared_constant`, `own_constant_data` |
| the spread builder | `own_spread_build` |
| the interface's `this` bound for a call | `own_interface_field_call` |

### 21.5 Sampled: use-after-move (24) and borrows-a-field (21)

Both buckets were read with `--di`.

- **One false positive, fixed:** the interface field call of §21.3 (`00interface_function_typed_field`).
- **The rest are true positives of single ownership.** Two shapes cover most of them, and each is a
  question for the model, not a fix:
  - **A value moved into a place that still holds it, then read**, in about 14 of the 24
    use-after-move files. Examples: `h.item = kept; return kept.n`, `(x = f()) !== null`,
    `(x += "ba") == ...`, `constructor(public what: string) { print(what) }`. The read could be a
    borrow of the destination while nothing overwrites or releases it.
  - **A string stored or returned as a second owner**, in most of the 21 borrows-a-field files.
    Examples: `testrec.str = testrec.str2`, `return held.text`, `caught = e.m`. Strings are
    immutable, so a second reference could be a copy.
- **A value given to an `any` parameter moves into its box** (`isA(union)` in `00union_to_any`). The
  box owns what it holds (§15), so the use after the call is a true use after a move. A box that
  borrows, for a callee that does not keep it, would accept it.

### 21.6 Known limits

- **A synthesized helper's error** (`___unbox<string>`, `___cast<...>`) is reported at the
  location of the first use that instantiated it, which may be a comment line at the top of the
  file.
- **An interface's `this` bound for a call is decided per read.** Under `--opt`, CSE merges the
  reads of one interface value, and a read that is also stored (`const mt: () => void = a.print`)
  makes the merged one a taker. `00interface_object3` compiles plain and fails with `--opt` and
  `--di`.
- **`instanceof` of a class value held in a union or a nullable local** is false under every
  model, even narrowed (`let a: C | null = new C(); a instanceof C`). It is a frontend or lowering
  bug outside this phase. `own_declared_constant` tests the value with `typeof`.

## 22. Strings as values (design, 2026-10-02)

A design to build after phase 6, ahead of phase 8 (`Shared<T>`). §21.5 found that most of the
corpus files rejected with "borrows a field and cannot be stored, returned or captured" are a
string given a second owner (`testrec.str = testrec.str2`, `return held.text`, `caught = e.m`). So
are the 17 files rejected with "returns a borrow of its argument on some paths and another value
on others" (`___unbox<string>`, `___cast<A, string>`, `return name` beside
`return "Sorry, " + name`). A string is immutable in TypeScript, so a copy cannot be told apart
from sharing. MLIRGen already treats it as a value: `s.length = n` makes a new string rather than
resizing in place.

### 22.1 The rule

Under `-mm=own`, a second reference to a string that own cannot prove to be a move or a borrow
becomes a copy.

- **A string** is a value whose type is `string`, a string literal type, or a union or an optional
  of `string` with only `null` or `undefined` beside it. A union with `null` is one pointer, the
  same as `string`; an optional (`string | undefined`, an optional parameter) is `{ptr, i1}`, the
  pointer meaningful only when the flag is set. A union with other members (`string | number`),
  and an object or array holding strings, are not covered.
- **Silent.** A copy makes no diagnostic.
- **Only where the program would fail.** Copies are made only in a function the analysis rejects,
  and only for the string retains it could not erase. A function that compiles today compiles to
  the same IR.
- **A write in place.** tslang accepts `t[0] = <char>65` on a string, which writes into its bytes;
  standard TypeScript rejects it. After `let t = s`, if `t` is a copy, the write changes `t` only.
  Under gc both names see it. The difference is accepted and tested.

### 22.2 The copy

`ts.StringCopy %s : !ts.string -> !ts.string`, its result a new allocation (`MemAlloc`). It lowers
to: null stays null; otherwise the length plus one byte, through the allocator `ts.StringConcat`
uses, and a `memcpy`. An optional is copied only when it holds a value.

- Only the own inference pass creates it, but it lowers under every model.
- `isFresh` counts it, beside `ts.StringConcat`.
- A string literal is never copied: it is constant data (§20.2), whose retains are already erased.

### 22.3 Where the copies go

The inference pass decides each function in up to two runs.

1. **A dry run.** The whole analysis runs, but it reports nothing and changes nothing:
   - nothing is erased;
   - the signature facts are not stripped;
   - the attributes the analysis sets are set and tracked, and taken off again after the copies
     (step 2), before the real run;
   - every error, including those that do not go through the `quiet` reporters today (the
     `RetainCell` and `delete` errors), is counted as a failure, not emitted.

   Without a failure, the function is decided by the real run alone, exactly as today.
2. **The copies.** After a failing dry run, each string retain the dry run did not erase is
   rewritten where its use can be found:
   - `ts.Retain(v)` for one use, the one `takerOf` names (a store, a push, a call that keeps it, a
     store into the result slot): `c = ts.StringCopy(v)` is inserted before that use, the use
     takes `c` in place of `v`, and the retain is erased. `v` keeps its owner, which still
     releases it; the use owns `c`.
   - `ts.RetainSlot` of `let x = v`: the local starts from `ts.StringCopy(v)` instead, and the
     retain is erased. `x`'s release at scope exit frees the copy.
3. **The real run.** The analysis runs on the rewritten function as it does today. A copy is fresh,
   and moves into its use by the existing rules. Anything still unproven is reported.

### 22.4 No copy

The error stays where a copy would be wrong:

- **No single use.** The retain's use cannot be pinned to one op, or the use took the value
  without a retain (rc consumed a fresh value). Copying there would leave the original with no
  owner: a leak.
- **A fact the callers rely on.** The value is a parameter, or read out of one, that the signature
  pass told this function's callers it keeps (`__own_params`) or returns a borrow of
  (`__own_result_borrows`). Facts lost (§15, an exported function) are not such facts. There the
  callers keep their own, so a copy is right.
- **Not a string** in the sense of §22.1.

### 22.5 What it should accept

- A field's string stored into another field, returned, pushed, or read out of a caught object
  (`caught = e.m`).
- A string moved into a field and used after (`h.text = kept; print(kept)`).
- A function that returns a borrow of its argument on some paths and something else on others:
  the borrowed return is copied, so the result is always fresh. That includes
  `___unbox<string>` and `___cast<A, string>`.
- An exported constructor storing its string parameter.

### 22.6 Tests and measures

- **Positive** (test-runner under own, AOT and JIT; each also under rc, none and gc, where no copy
  is made): `own_string_copy_field`, `own_string_copy_after_move`,
  `own_string_copy_mixed_return`, `own_string_copy_nullable`, `own_string_copy_exported`,
  `own_string_copy_branch` (added after the final review).
- **Own only:** `own_string_copy_written`, `let t = s; t[0] = <char>65`, which changes `t` only.
- **Negative:**
  - a kept parameter whose callers know the fact;
  - a second reference to a value that is not a string;
  - a taker that consumed without a retain.
- **Teeth:** with the copies turned off, every positive fails. With the fact guard turned off, its
  negative compiles.
- **Measured:** a loop that copies a field's string 200,000 times stays flat under own, as under
  rc (`measure.ps1`).
- **Unchanged where it compiles:** `--emit=mlir-affine` with the copies on and off is the same for
  every corpus file that compiles on main.
- **Corpus:** none lost, plain and with `--opt`. Each gained file passes test-runner under own,
  AOT and JIT, with the default flags and with `-noopt`.

### 22.7 Results, 2026-10-02

Plan: `docs/superpowers/plans/2026-10-02-own-strings-as-values.md`. Changed: `TypeScriptOps.td`,
`LowerToAffineLoops.cpp`, `LowerToLLVM.cpp` (the op, lowered under every model), `OwnershipFacts.h`,
`OwnershipInferencePass.cpp`. MLIRGen and the signature pass do not change, so rc, gc and none
emit what they did.

#### What was built

- **`ts.StringCopy`** (§22.2) and its lowering:
  - null stays null, and an optional is copied only when its flag is set;
  - otherwise the length plus one byte, from the allocator `ts.StringConcat` uses, and a `memcpy`;
  - an operand that is not a pointer must be exactly `{ptr, i1}`, or the pattern does not match.
- **`isCopyableString`** (`OwnershipFacts.h`) is §22.1's string: `string`, a string literal type, an
  optional of one, or a union of one with only `null` or `undefined` beside it that needs no tag.
  Measured under gc: `string | null` and `"a" | "b" | null` are a `ptr`, `string | undefined` is
  `{ptr, i1}`, and `string | number` is tagged and not covered.
- **The dry run** (§22.3). `analyze` runs once quietly. Every report goes through `reporting()`,
  which counts a failure instead of emitting it, the `RetainCell` and `delete` errors included.
  Nothing is erased or stripped. The attributes it sets (`setAttrTracked`) are tracked and taken
  off after the copies, before the real run: a store into a borrowing closure's box or into a
  borrowing field is a borrow only by its `__own_capture_borrow`, and the copies must see it. The
  copies erase only retains, which carry none of them. On the corpus the dry run changed nothing,
  plain and with `--opt`.
- **The copies** (`makeStringCopies`, `copyForRetain`, `borrowsKnownParam`). After a failing dry
  run, each string retain it did not erase is rewritten:
  - a `ts.Retain` whose one use is in the retain's block: the use takes `ts.StringCopy(v)`;
  - a `ts.RetainSlot`: the local starts from the copy;
  - no copy for: no single use in the retain's block, one operand taken twice (`several`), not a
    string, or a parameter whose callers were told it is kept or borrowed.
- **Two facts about the copy**, both needed by the positives:
  - It is a borrow of its operand (`isBorrow`, beside `StringConcat` and `StringResize`). Without
    it, the place check saw a copy of a field as keeping the field.
  - The borrow walk (`walkBorrowed`) stops at it: its result points into nothing. Without it, in
    `b.s = a.s` the store of the copy counted as a use of `a.s` after `ReleaseSlot(b.s)`. The two
    fields have the same position and type, so `dropsChain` cannot tell them apart.

#### Where the copy goes (§22.3 as built)

- **At rc's retain, not in front of the use,** when the use's operand is the retained value, or is
  defined before the retain in its block.
  - A store releases what it overwrites before it stores: in `b.s = b.s`, a copy in front of the
    store would read freed memory.
  - A return comes after rc releases the locals: in `return made.str2`, the copy would read the
    field of a released object.
  - With the copy in front of the use, the place check rejects both rather than miscompiling
    (teeth (c)).
- **Only for a use in the retain's block.** rc's birth retain sits right after the producer, which
  may be in another block. Copied there, the copy has no owner on a path that does not reach the
  use (`if (c) { h.a = s; }`): a leak. rc's retain beside the store makes the copy in the store's
  block instead, and the real run erases the birth retain once the value is proven.
- **A loop makes a copy per pass.** A string made outside a loop and stored inside it compiles with
  a copy in the loop body. Each store's `ReleaseSlot` frees the previous copy, the field owns the
  last one, and the original is released after the loop. `own_err_field_loop`, that shape with a
  string, compiles and runs now; it uses a class instead, and its message is unchanged.

#### Corpus

`--emit=llvm -mm=own --no-default-lib`, every file of `test/tester/tests`, against main, plain and
with the runner's `--opt --opt_level=3`:

| | main | this branch | gained | lost |
|---|---|---|---|---|
| plain | 417 / 593 | 453 / 593 | 36 | 0 |
| `--opt` | 416 / 593 | 451 / 593 | 35 | 0 |

Two of the files gained compile plain only:
- `00interface_object3`, §21.6;
- `05strings`: under `--opt`, CSE merges the two `"X" + true` (one printed, one compared) into one
  fresh string taken twice, and the catch-all reports it. A three-line program of that shape fails
  the same way on main.

The first errors, plain (`--opt` in brackets where it differs):

| first error | main | this branch |
|---|---|---|
| `takes a second reference` (the catch-all) | 25 (27) | 19 (22) |
| used after its value was moved | 23 (25) | 24 (25) |
| borrows a field and cannot be stored, returned or captured | 20 | 11 |
| returns a borrow of its argument on some paths and another value on others | 17 | 1 |
| borrows an element and cannot be stored, returned or captured | 10 | 3 |
| borrows a field but is used here after it may be released or overwritten | 6 | 13 |
| borrows an element but is used here after it may be released or overwritten | 0 | 2 |
| merged from several branches | 8 (7) | 5 (4) |
| borrows the global and cannot be stored, returned or captured | 8 | 7 |
| the rest | 59 (57) | 55 (54) |

- **The mixed-return bucket, 17 → 1.** The one left is `conditionalTypes2`'s
  `___cast<union<hybrid_func, string>, hybrid_func>`, which returns a function value, not a string
  (§22.4).
- **The use-after-release buckets, 6 → 15,** are files whose earlier error is gone.
- **No error is new.** For every corpus file, plain, with `--opt` and with `--di`, each error this
  branch reports is one main reports too. The whole error list was compared, not only the first.

#### Gated

The 35 files gained under `--opt`:
- **30 run alone.** Each passes test-runner `-mm=own`, AOT and JIT, with the default flags and with
  `-noopt`.
- **5 are halves of four import/export pairs:** `export_class_extends`, `export_class_indexer`,
  `export_owned_returns` with `import_owned_returns`, and `export_type_alias`. Each pair was run
  the way the suite runs it, under own: AOT, `-shared` AOT, and `-shared` JIT. A JIT pair without
  `-shared` fails under gc too.
  - `class_extends`, `class_indexer`, `owned_returns`: pass, with the default flags and `-noopt`.
  - `type_alias`: the export half compiles, but the importer is rejected at
    `M.labelOf({ label: "l" })` with "borrows a field and cannot be stored, returned or captured".
    Main reports the same error behind a `___cast` mixed return. The literal is re-made into
    `M.Options` by a `ts.CreateTuple` of its field's read, with no retain, so there is nothing to
    copy (§22.4).
  - The export half's own copy (`return o.label ?? "none"` in an exported function, whose facts
    are lost) was gated with a one-module program holding the same function: it passes under
    own, AOT and JIT, with the default flags and with `-noopt`.

#### Tests and teeth

- **Positive.** Each runs under own (AOT, JIT, and no counting) and also under rc, none and gc, AOT
  and JIT; each passes under own with `-noopt`:
  - `own_string_copy_field`: a field into a field, a field returned, a field pushed;
  - `own_string_copy_after_move`: a string moved into a field and used after, stored on one branch
    only, stored in a loop, and stored beside a closure that borrows it (the closure also captures
    a `let`, so it borrows all it captures). Before the dry run's attributes were kept for the
    copies, the store into the closure's box counted as a second taker, and that last shape was
    rejected;
  - `own_string_copy_nullable`: four copies (`string | null` holding a pointer and null, an optional
    holding a value and none), so the lowering's three paths run;
  - `own_string_copy_exported`: an exported constructor storing its string parameter;
  - `own_string_copy_mixed_return`: `greet` returning `name` on one path and `"Sorry, " + name` on
    the other, `<string>` of an `any` (`___unbox<string>`), and `<string>` of a `number | string`
    (`___cast`). Its own IR has eight copies; main rejects it with eight errors. It needed no new
    code;
  - `own_string_copy_branch`: a string stored into a field on one branch only and used after,
    alone, for the IR check below.
- **Own only:** `own_string_copy_written`, `let t = s; t[0] = <char>65` changes `t` only.
- **The copy's block** (`test-own-string-copy-in-store-block`, `own-string-copy-block.cmake`): the
  own IR of `own_string_copy_branch.ts` has one `ts.StringCopy` in `stored`, after the entry
  block's `cf.cond_br`. Copied at the birth retain instead, it would leak on the path that skips
  the store, which no run can see, so the IR is read. The file is also a positive, under every
  model; it has no loop, so with the same-block rule off it still compiles, and only this check
  fails (teeth (e)). `own_string_copy_after_move` has the same `stored`, but with the rule off its
  `storedInLoop` is rejected first.
- **Negative:**

| test | message |
|---|---|
| `own_err_string_kept_param_known` | is used here after its value was moved |
| `own_err_string_call_keeps_twice` | borrows a field and cannot be stored, returned or captured |
| `own_err_string_pushed_twice` | is used here after its value was moved |

- **Teeth**, each a temporary switch, with the JIT cache cleared:

| switch | result |
|---|---|
| (a) `makeStringCopies` returns at once | the 15 own runs of the five positives then written fail |
| (b) `borrowsKnownParam` returns false | `own_err_string_kept_param_known` compiles, and `keep` never releases the `v` its callers gave up: the leak the guard is for |
| (c) the copy always in front of the use | `own_string_copy_field` and `own_string_copy_nullable` fail (6 tests); `r.s = r.s` is rejected by the place check |
| (d) the `several` guard off | `own_err_string_pushed_twice` compiles, and is correct: each retain copies a different operand |
| (e) the same-block rule off | `test-own-string-copy-in-store-block` fails: `stored`'s copy is in the entry block |

- **Step 7 of the plan.** Five string negatives compiled once strings were copied:
  - `own_err_mixed_return`, `own_err_param_returned_exported`, `own_err_param_kept_exported` and
    `own_err_merged_result` use a class now, with their messages unchanged (`pick` for the first);
  - `own_err_unbox_string` was deleted (§21.2's table says so).

#### Measured

AOT, `measure.ps1`, `r.s = "v" + i; r.t = r.s` in a loop, in MB:

| iterations | gc | rc | none | own |
|---|---|---|---|---|
| 20000 | 5.9 | 4.4 | 6.6 | 4.4 |
| 200000 | 5.9 | 4.4 | 25.9 | 4.4 |

Main rejects the program under own (`r.t = r.s`, "borrows a field").

#### Unchanged where it compiles

The 417 files that compile on main were compiled with main's binary and this branch's,
`--emit=llvm -mm=own --no-default-lib`. §22.6 named `--emit=mlir-affine`; the plan's step and this
check use the LLVM IR, which is what runs. The output is identical once the names that change from
run to run are normalised. Main against main needed the same normalisations, as `sed -E`
substitutions:
- the hashes, `s/FH[0-9]+/FH/g`;
- pointer-derived suffixes (`@a_<n>`, `@td_<n>_number`, `@tsrel_<n>`),
  `s/([A-Za-z])_[0-9]{4,}/\1_N/g`;
- class numbers (`<Class>.<n>..vtbl`), `s/\.[0-9]{6,}\./.N./g`;
- the size of a string constant that holds an `FH` name, on its line only,
  `s/\[[0-9]+ x i8\]( c".*FH)/[K x i8]\1/`.

One file, `00union_null_undefined_nonstrict`, can differ in the order of a union's members in a
synthesized `___cast<union<...>>` name. The order flips between runs of either binary (main gave
one order in 9 runs of 12, this branch in 10 of 12). With the same order, the IR is identical.

#### Linux

GCC, in WSL, the branch at `efd252b1`: the build is clean, and
`ctest -R "own-err|own-no-counting|own-verify"` passes 140 of 140. `ctest -R own_string_copy` (the
positives under own, rc, none and gc, AOT and JIT) passes 48 of 48. After the final review's fixes
(the borrow attributes kept for the copies, the retain-before-use check, `dominance` cleared, and
`own_string_copy_branch`), the branch at `2345904d` builds clean there, and
`ctest -R "own-err|own-no-counting|own-verify|own_string_copy|own_err_string"` passes 191 of 191.

#### Outside the rule

- **A returned field of a parameter.** `function textOf(r) { return r.str2; }` stays a borrowed
  return (`__own_result_borrows`), and a caller that uses the result after overwriting `r.str2` is
  still rejected (§22.4). The accepted returned field is one of an object the function owns, of a
  global, or of an exported function's parameter (facts lost).
- **The guard's negative.** `borrowsKnownParam` matters only when a known parameter's retain
  survives the dry run, as in `keep(v) { holder.s = v; print(v); }`. The plan's version, `main`
  passing `h.s` to `keep`, never reaches the guard: `main` has no retain to copy.
- **A field's string given to a known keeper** (`keep(h.s)`) is that shape: no retain in the caller,
  so it stays an error.
- **No consumed-twice shape.** The plan's `own_err_string_consumed_twice` compiles: the const lambda
  becomes a direct call, and the two locals are views of the const. A parameter called, a field
  holding a function, and an indirect result pushed twice and stored twice all had rc's retains.
  The test was dropped; `own_err_string_call_keeps_twice` covers a taker with no retain in the
  caller.
- **An omitted optional argument** is a `ts.Undef : !ts.optional<!ts.string>`. Given to a callee
  that keeps its parameter, it is reported "not owned", because `holdsNoBlock` does not cover it.
  This predates the phase.
- **One string taken twice by one op** (`list.push(s, s)`) is rejected, by the `several` guard. With
  the guard off it compiles and is correct, so the rule is conservative there; widened, the
  negative becomes a positive.
- **A string captured by a closure that owns its copies, and also stored** (`const f = () =>
  s.length; r.t = s; print(s)`). A closure that captures only copies owns them (phase 4) unless it
  does not escape, is given to a maker whose result is bounded, and none of its copies is fresh.
  Here the string is fresh, so the store into the box is a taker, the string has two, and the
  `several` guard keeps the error.
- **A use whose operand is a view defined after the retain** gets its copy in front of the use. A
  same-position, same-type `ReleaseSlot` in between would make the place check reject the program:
  a false error, not a miscompile. No test has it.
- **An object literal re-made into its annotated type** (a field missing from the literal) reads
  the literal's fields into a `ts.CreateTuple` with no retain, so a string there is not copied
  (`import_type_alias`).
- **A fresh string merged by CSE** under `--opt` (`05strings`): the two `"X" + true` become one
  value with two retains and two releases, for a print and a comparison. Neither retain has a taker
  to copy for, so the catch-all reports it, on main too.
- **A copy made at a birth retain, before a call that may throw** (`const t = "t" + n; boom(n);
  h.text = t`): the copy is made right after `t`, before `boom`, and leaks when `boom` throws.
  `t` itself leaks the same way already, since its release is on the normal path only.
- **The tag of a union.** `isCopyableString` asks `isUnionTypeNeedsTag` with a default
  `CompileOptions`, and the lowering asks with the real ones (`strictNullChecks` can change the
  answer). Both disagreements are safe: where the pass sees a tag there is no copy, and the error
  stays; where only the lowering sees one, `ts.StringCopy` fails to match (`notifyMatchFailure`),
  and the build fails loudly.

## 23. `Shared<T>` (design, 2026-10-03)

Phase 8 (§7). §2.7 deferred it: an explicit, counted generic for graphs that a tree of single
owners cannot express (a parent link, a node in two lists, a DAG), reusing rc's retain and release
routines for that type only. Everything else stays single-owner and uncounted.

Decisions taken in the design conversation:

| Question | Decision |
| --- | --- |
| Form | A wrapper: a handle to a block that holds one `T` and the count. Not a marker on `T` itself |
| Where it is declared | Built into the compiler, so `--no-default-lib` programs and every model have it |
| Members | `.value` read and write, `===`/`!==` on handles, `Shared.count(s)` |
| Handles under own | Own steps aside: every rc retain and release of a handle stays and is counted |
| Cycles | Leak, as under rc, and documented; `WeakRef<T>` under own is a later phase |

### 23.1 The type and its members

```ts
class Node {
    v = 0;
    next: Shared<Node> | null = null;
}

const a = new Shared(new Node());
const b = a;                 // a second handle: the count is 2
a.value.v = 1;
print(b.value.v);            // 1
b.value = new Node();        // the old Node is released, the new one moves in
print(Shared.count(a), a === b);
```

- **The type.** `Shared<T>` is resolved by name in `MLIRGenTypes` beside `Reference<T>` and
  `BoxedObject<T>`, to a new `!ts.shared<T>`. A program's own declaration of a type named `Shared`
  is found first and wins, as for every embedded name.
- **At run time** a handle is a pointer to a heap block: the header word is the count, and the
  payload is one `T` in its storage type.
- **`new Shared(x)`** is `ts.SharedNew`: it allocates the block and stores `x` into it, which is a
  store into a container (§2.3): `x` moves in.
- **`s.value`** is a `ts.Load` of the place `ts.SharedValueRef(s)`, and **`s.value = x`** stores
  into that place the way a field store does (MLIRGen's retain, `ts.ReleaseSlot` and `ts.Store`),
  so `x` moves in and the old `T` is released. One op instead of two lets the place-read and store
  machinery of MLIRGen and own apply unchanged.
- **`a === b` and `a !== b`** compare the pointers.
- **`Shared.count(s)`** reads the header: the number of handles under rc and own, `-1` under gc
  and none, which keep no count.
- **A program means the same under every model.** gc and none get a wrapper that is never
  counted. rc counts it as it counts every block. Own counts this type and no other.
- **Other uses of the type.** A handle can be a field, an element, a union member (`Shared<T> |
  null` is one pointer), a capture, a parameter, a result, or boxed into `any`. It goes into a
  module's `__decls` as `Shared<T>`, so a module that imports it sees the same type.

### 23.2 Handles under own: the analysis steps aside

- **MLIRGen emits rc's ops for handles under own too.** It emits the same `ts.Retain`,
  `ts.RetainSlot`, releases and `__owned_consumed` declarations for a `!ts.shared` value under rc
  and own, because both track ownership (§3.1). That is unchanged.
- **The inference pass leaves every retain and release of a handle in place.** A handle is a value of
  one of these types (`MLIRTypeHelper::isSharedHandleType`):
  - `!ts.shared<T>`, or an optional of a handle;
  - a union with at least one `Shared<T>` member whose other members own no heap memory: `null`,
    `undefined`, a number, a boolean, a literal, another `Shared<U>`. `Shared<T> | null` is one
    pointer (§22.1). `Shared<T> | number` is tagged: its routines dispatch on the tag through
    descriptors, a `Shared` member's counting and the others' doing nothing.

  The pass never erases such a retain or release, and never reports one. A handle's count is
  therefore exactly rc's, which rc's own suite tests.
- **A `Shared<T>` in a union with a member that owns a block** (`Shared<T> | string`,
  `Shared<T> | Node`, `Shared<T>[] | Shared<T>`) is rejected under own where a handle is put into
  one: `a Shared<T> in a union with '<type>', which -mm=own does not count, is not supported`.
  Counting the union would count a single-owner block, and not counting it would lose the
  handle's count. gc, none and rc accept it.
- **Lowering.** Under own, a surviving retain of a `!ts.shared` value no longer reports "left a
  retain behind"; it lowers to `__tslang_inc_ref`, as under rc.
- **The release routine** of a `!ts.shared` value is the same in every model that tracks
  ownership: `__tslang_dec_ref`, and at zero, release the payload `T` and free the block.
- **The `T` is released by its model's routine.** Under own, that is a single-owner drop: free
  unless immortal, after releasing the `T`'s own fields.
- **Inside a container.** Everything outside the handle is ordinary own analysis. A handle stored
  in a field, element, union or closure is a value the container owns, and the container's drop
  releases it through the routine above.
- **The type descriptor.** An `any` box, a union or an interface releases what it holds through the
  type's descriptor, so `!ts.shared` gets a descriptor whose release slot is its routine.
- **Not in this phase.** Eliding the count for a handle own could prove moved or borrowed. That
  would need a birth convention for handles that differs from rc's (rc counts the birth with a
  retain own would erase), so it is left for a measured later step.

### 23.3 Reading through a handle

Under own, no other path reaches a block while its owner holds it. That is why a borrow out of a
container ends only at the container's own release or overwrite. A `Shared` breaks that: two
handles can reach one block.

```ts
const n = a.value;           // a borrow of the Node
b.value = new Node();        // if b is a, this frees the Node n borrows
print(n.v);                  // rejected
```

- **The rule.** A borrow read through a handle, at any depth (`s.value`, `s.value.name`,
  `s.value.items[0]`), ends at the first of:
  - the handle's own release or overwrite (the existing rule);
  - a write into any place reached through any `Shared`: `x.value = …`, `x.value.f = …`, an
    element store, `push` or `pop` on an array reached through one;
  - a call that is not known to drop nothing (`__own_no_drops`, §15).
- **A handle copied out is a counted retain, not a borrow.** `cur = cur.value.next` gives `cur` a
  handle of its own, which no write ends. That is how a graph is walked, so walking is not
  restricted.
- **The error** is the existing "borrows a field but is used here after it may be released or
  overwritten".
- **Conservative by design.** It does not check whether two handles could be the same: any write
  through any handle, and any call that may drop, ends every such borrow. The alternative is a
  runtime borrow flag, Rust's `RefCell`, which costs at run time; it is not taken.
- **A string read through a handle** is a borrow too. Where one of rc's retains survives, §22
  makes it a copy, as for any field.

### 23.4 Cycles

`a.value.next = b; b.value.next = a` leaks both blocks, as under rc. The test
`own_shared_cycle` shows that it compiles and runs, and §23.7 (the results) records it.

### 23.5 The verifier

`--verify-ownership` (§9.18 of the rc document) runs before the own passes. It checks that every
slot that takes a reference gives it back on every path. Handles keep rc's pairing, so it needs no
change. It runs on every `own_shared_*` test, under own and rc.

### 23.6 Tests and measures

- **Positive**, under own, rc, none and gc, AOT and JIT:
  - `own_shared_basic`: two handles, `.value` read and write, `===`.
  - `own_shared_user_named`: a program's own class named `Shared`, and its static `count`, win over
    the built-in (§23.1).
  - `own_shared_graph`: a DAG (one item under two parents) and a singly linked list walked with
    `cur = cur.value.next`.
  - `own_shared_containers`: one node in two arrays, a `Shared<T> | null` field, a closure
    capturing a handle, a handle boxed into `any` and back, a handle returned from a function.
  - `own_shared_cycle`: a cycle that is never freed, which still runs: a doubly linked list and a
    node with a parent link, both cycles that leak (§23.4).
  - `own_shared_union`: a handle in a `Shared<T> | number` local, field and element, each switched
    to a number and back, with rc's counts under own (§23.2).

  Each frees every handle but one, then allocates over the freed memory (the churn pattern)
  before reading through the one left. An early free then reads reused memory.
- **Under own and rc only:** `own_shared_count`, which asserts `Shared.count` after each copy and
  drop.
- **Negative:**
  - `own_err_shared_borrow_write`: the §23.3 example;
  - `own_err_shared_borrow_call`: a borrow through a handle used after a call that may drop;
  - `own_err_shared_moved_in`: a class value given to `new Shared(x)` and then used, which is a
    use after move.
  - `own_err_shared_union_owned`: a handle put into a `Shared<T> | string` (§23.2).
- **Teeth:**
  - with the inference pass erasing a handle's retains as for any value, the positives fail;
  - with §23.3's rule off, its negatives compile.
- **Verifier:** `--verify-ownership` is quiet on every `own_shared_*` file.
- **Measured:** a loop that builds and drops a small graph of handles 200,000 times stays flat
  under own and rc (`measure.ps1`), and grows under none.
- **Unchanged:** no corpus file uses `Shared`, so the corpus (453 / 593 plain, 451 with `--opt`)
  must be unchanged, and its LLVM IR identical for every file that compiles, as in §22.7.

### 23.7 Results, 2026-10-03

Plan: `docs/superpowers/plans/2026-10-03-own-shared.md`. Changed: `TypeScriptTypes.td`,
`TypeScriptOps.td`, `MLIRTypeCore.h`, `MLIRTypeHelper.h`, `MLIRTypeIterator.h`, `MLIRPrinter.h`,
`TypeOfOpHelper.h` (the type), `MLIRGenTypes.cpp`, `MLIRGenExpressions.cpp`,
`MLIRGenAccessCall.cpp`, `MLIRGenCast.cpp`, `MLIRGenImpl.h` (MLIRGen), `OwnershipRoutineLogic.h`,
`CastLogicHelper.h`, `LowerToAffineLoops.cpp`, `LowerToLLVM.cpp` (the routines and lowering),
`OwnershipFacts.h`, `OwnershipInferencePass.cpp`, `OwnershipSignaturePass.cpp` (own),
`OwnedReturnConsumptionPass.cpp` (a handle global's initializer, every model). Every new
branch keys on `!ts.shared`, a `ts.SharedValueRef`, or `isSharedHandleType`, so a program without
`Shared` compiles as before (see "Unchanged where it compiles").

#### What was built

- **The type** `!ts.shared<T>` (`mlir_ts::SharedType`), lowered to `ptr`. `Shared` is an embedded
  name in `MLIRGenTypes`, so a program's own `Shared` is found first and wins
  (`own_shared_user_named`). It is nullable (`===`, `null`), its `typeof` is `"object"`, and it
  prints as `Shared<T>`, in messages and in `__decls`.
- **The ops**, three, not the two of §23.1's first draft:
  - `ts.SharedNew` (`MemAlloc`): allocates the block with the allocator every model's `new` uses,
    and stores the value;
  - `ts.SharedValueRef`: the place `!ts.ref<T>` of the payload, the identity in lowering.
    **`s.value` is a `ts.Load` of it**, and `s.value = x` stores through it (§23.1), so the place
    machinery of MLIRGen and own applies unchanged;
  - `ts.SharedCount`: `-1` where nothing is counted (gc, none), otherwise the header word as a
    `number`.
- **MLIRGen.**
  - `new Shared(x)` is reached only when `Shared` does not resolve. `T` comes from a type argument,
    then a `Shared<T>` receiver, then the value: **`new Shared(1)` is a `Shared<s32>`**, as
    `let x = 1` is an `s32`. `new Shared()`, two arguments and two type arguments are errors.
  - `Shared.count(s)` is intercepted at the call when `Shared` does not resolve. It takes a bare
    handle, and a narrowed `Shared<T> | null` or optional one (`if (s)` keeps the union's type),
    which it unwraps as `.value` does.
  - The payload goes in with `mlirGenRetainCaptured` (a move, §2.3), and `ts.SharedNew`'s result is
    a fresh owned reference (`markFreshBlockOwned`). A `.value` place is an owning slot
    (`isOwnedSharedValueSlot`), so `s.value = x` is rc's `Retain`, `ReleaseSlot`, `Store`.
  - **A `const` of a handle gets identity storage** (`needsIdentityStorage`, as a `const` array
    does), so `const b = a` is a second, counted handle (§23.1). That is any handle
    (`isSharedHandleType`): a bare `Shared<T>`, and a union or optional one (`const c = p.child`,
    `const c: Shared<T> | null = a`). A narrowed view (`if (c)`) stays a view. Until the final
    review's fix round, only a bare `Shared<T>` had storage: a union `const` was folded into the
    place it was read from, held no count, and got no check, so unlinking a node
    (`const nxt = head.value.next; head.value.next = null`) left `nxt` reading freed memory under
    rc and own.
  - **A handle global initialised by a value that carries a reference is born holding it.** A
    value given to `ts.GlobalResult` straight keeps its reference (a terminator is no use §9.30
    releases after). One given through a view of the global's type - `ts.Cast` to
    `Shared<T> | null`, `ts.OptionalValue` to `Shared<T> | undefined`, `ts.CreateUnionInstance` -
    was a discarded temporary, given back at the end of the initializer, so the global held a block
    at count 0, under rc as well. `OwnedReturnConsumptionPass` now claims it through any such view
    (`claimHandleGlobalInitializers`), from `new Shared(..)` or from a call: a call's result is
    marked as carrying a reference only there, once its callee is classified, which is why
    MLIRGen cannot see it. Keyed on the global's type. The first fix round claimed only a cast of
    `new Shared(..)`, in MLIRGen; `Shared<T> | undefined = new Shared(..)` and
    `Shared<T> | null = mk()` stayed at count 0 until the second (`own_shared_global_init`).
    **A handle read out of a field or an element carries no reference**, and since the third fix
    round the claim gives it one: a `ts.Retain` right after the read, before the end of the
    initializer gives back a temporary object or array (`own_shared_global_field`:
    `holder.child`, `new P().child`, `mkP().child`, `mkArr()[0]`). Until then such a global was
    born at count 0 under rc and own alike and crashed when read (the re-review's `globfield`,
    `tmpfield`, `callfield`: `count 0`, then 0xC0000005). #446 did not hide that: it hides only a
    global initialised from another global. The class equivalents (`let g: Node = new P().child`)
    read freed memory under rc on main too (`cls_globfield`, `cls_tmpfield`); they are not changed.
    **Not covered:** a handle global initialised from another global (`let h = root`) takes no
    count of its own; #446 makes such an initializer read null in every model (the review's
    `glob.ts` reads null for `d` and `h`), which hides it. A global object literal holding a handle
    from a call (`let o = { s: mkS(4) }`) gives the call's reference back and crashes under rc and
    own (`objonly`), as its class equivalent reads freed memory under rc on main (`cls_obj`).
- **The routines.** `ownsHeapMemory` is true for a handle. Its release routine is
  `__tslang_dec_ref`, and at zero the payload's routine and `__tslang_free_block`; its retain is
  `__tslang_inc_ref`. `emitIfLastReference(counted)` makes own count a handle where it tests
  mortality for everything else. A tag-free `Shared<T> | null` needs no routine of its own.
- **Under own, the analysis steps aside** (§23.2):
  - the facts `isHandle` and `touchesHandle`; `isPlace` includes `ts.SharedValueRef`;
  - the inference walk returns early for every op that touches a handle, except a `ReleaseSlot` of
    a field or element holding one, which stays a drop (a handle overwritten there may be the last
    one). A handle never becomes a candidate, by any of the four ways in, and a safety net takes
    any handle op out of the erase list;
  - the signature pass gives no parameter or result fact to a handle, and `moveIntoGlobals` leaves
    a handle's retain in a global's initializer in place: it is the global's count. Before the fix
    round it erased it as a move, so a global handle was born at count 0 and freed while still
    held, and a `Shared<T> | null` global was rejected as "takes a second reference";
  - a generator over a handle parameter is not a borrowing state (`findBorrowingStates`): the
    parameter's cell holds a count of its own, which only the box's release gives back, so its
    box owns what it holds;
  - the lowering emits a surviving handle retain under own (`__tslang_inc_ref`), without "left a
    retain behind";
  - a cell holding a handle (a captured handle parameter) is released by `ReleaseCell`, and a box
    may take it (`decideCells`, `decideEscaping`): the cell holds its own count.
- **The handle, widened** (Ruling 6, §23.2): a `Shared<T>`, an optional of one, or a union with a
  `Shared` member whose other members own no heap. `Shared<T> | number` and
  `Shared<A> | Shared<B> | null | number` are counted under own exactly as rc counts them.
  `Shared<T> | string`, `Shared<T> | Node` and `Shared<T>[] | Shared<T>` are rejected under own
  where a handle is put into one, with a named error (`reportHandleInOwningUnion`).
- **Reading through a handle** (§23.3), as built:
  - `chainOf` treats a base read through by `ts.SharedValueRef` as a `NotOwned` root and records
    it as a handle; a handle loaded out of a place walks on up, so `p.child.value` sees `p.child`;
  - `rootEnds`: a borrow through a handle also ends at that handle's own `ReleaseSlot` or `Store`
    (a local) or `ts.Release` (a temporary). A copy of the handle does not end it;
  - `dropsChain`: a `ReleaseSlot` of a `.value` place ends every borrow read through any handle,
    **whatever the element type** (Ruling 4);
  - a `push` or `unshift` on an array reached through a handle (`reachesSharedValue`) is a drop,
    inline and inside a callee (`dropsInBody`);
  - a function that stores a parameter into a handle's payload keeps that parameter (`keeps()`
    sees `s.value = x` as a store into a place);
  - **a handle read out of a place is a borrow of the place** (`placeReadOf`), checked as a field
    read is (`checkPlaceRead`), unless every use of it is one rc counts (`isCountedHandleRead`,
    `handleReadUses`, Rulings 7 and 9). **Every use is classified, and a use not known to count
    is a borrow** (fourth fix round, Ruling 9). A counted use is a receiver with rc's retain for
    that very use (`isCountedHandleReceiver`): a store into a field, an element, a global or an
    owning local (`countedStoreRetain`: `ts.Retain` of the value, then the store, or the
    `ts.ReleaseSlot` of the old value and then the store); a return, which after the affine
    lowering is a store into the result slot after the scope exit's releases (`ts.Retain`, then
    `ts.Release` and `ts.ReleaseSlot` of the frame's locals, then the store); a push, an unshift,
    `new Shared`, an array literal's element and an `any` box, each right after the row of
    retains MLIRGen puts before what it fills (`retainedRightBefore`); or an owning local's slot
    (`ts.RetainSlot`). A cast or view that is still a handle, a merge and a local that owns
    nothing pass the value on, and their uses are classified in turn. Everything else is a
    borrow: a use `isBorrow` lists, and every use that is neither of the above - a tuple's element,
    an object literal's field, a capture, a `yield`, a cast to what is not a handle, a receiver
    with no retain of its own, any op not named here; the borrow walk reports one that keeps the
    value as a keep. rc's retains count for nothing on their own: each is the count of the
    receiver right after it. A handle argument is always a borrow, never a kept one, since a
    handle gets no parameter facts: `keep(k, q.child = p.child)`, `keep` storing it, is rejected.
    `cur = cur.value.next`
    is counted only, so walking stays legal. `f(p, p.child)`, where `f` overwrites `p.child`, is
    rejected (`own_err_shared_place_arg`). Before the first fix round every handle read was
    exempt, and that program read freed memory under own. It still does under rc, which retains
    no argument;
  - **the exemption is per use, not per read** (second fix round). One read with a counted use and
    one that reads through the place (`f(p, q, q.child = p.child)`) is a borrow: the counted uses
    are a copy of their own in the borrow walk (`walkBorrowed`'s `countedHandle`), and the other
    use is tenure-checked, so that program, where `f` overwrites both fields, is rejected
    (`own_err_shared_place_two_uses`). The release of the old value a counted store makes right
    before storing the retained read (`ts.Retain`, `ts.ReleaseSlot`, `ts.Store`) is not a drop
    of the read (`collectCountedStoreReleases`): the block holds the retain's count by then, even
    when the two fields are one. So `Shared.count(q.child = p.child)` and
    `look(q.child = p.child)` stay legal (`own_shared_place_two_uses`). Until this round any one
    counted use exempted the whole read, and that program read freed memory under own;
  - **a store counts only when rc retains for it** (third fix round). `isCountedHandleReceiver`
    took every `ts.Store` of the read as counted, and the borrow walk skipped it as a copy. In
    `const o = { s: (q.child = p.child) }` the read has two stores: the one into `q.child`, which
    rc retains for, and the one into the object literal's field, which rc makes no retain for. The
    second kept the read with no count, so after `dropBoth(p, q)` own read freed memory and crashed
    (the re-review's `e2_objlit`). The second round had put the remaining gap in "an op outside
    `isBorrow`'s list"; this one was the store receiver. That program is now rejected with the
    plain `{ s: p.child }`'s message, `borrows a field and cannot be stored, returned or captured`
    (`own_err_shared_place_literal`). The array literal, return, constructor argument and `any`
    shapes (`e3`, `e6`, `e7`, `e9`) are retained by rc, compile, and count as rc does;
  - **a use not known to count is a borrow** (fourth fix round, Ruling 9). `handleReadUses` still
    let a use that was neither a counted receiver nor in `isBorrow`'s list fall through, so in
    `const t: [Shared<T>, number] = [(q.child = p.child), 1]` the counted store into `q.child`
    exempted the read and the tuple kept it with no count: after `dropBoth(p, q)` own read freed
    memory and crashed (the re-review's `c7`, and `as const` and `[Shared<T> | null, number]`,
    `d4` and `d10`). The default is now inverted: only a receiver with rc's retain for that use
    counts, and every other use makes the read a borrow, which the borrow walk reports at the tuple
    (`own_err_shared_place_tuple`, the plain `[p.child, 1]`'s message). Push, unshift,
    `new Shared` and a return now need their retain too, where before they counted on sight; an
    array literal's element is now a receiver, with its retain (before, its bare retain alone
    counted the read). A return and an `any` box are classified before `passesOn` would hand
    them on (`isHandOff`), in `handleReadUses` and in the borrow walk alike, and
    `ts.ReturnInternal` is no receiver any more: it is reached only through the result slot.
    **The pairing is exact in effect.** `countedStoreRetain` and `retainedRightBefore` match a
    retain by position, and a retain has one next op, so no two receivers count the same retain.
    A retain made for one receiver but sitting right before another, uncounted, store of the value
    would be taken by that store; the receiver it was made for then finds no retain of its own and
    makes the read a borrow, so the read is never exempt with fewer retains than counted
    receivers. No MLIRGen site emits that order (each emits its retain right before its own op),
    and nothing more was built for it;
  - `ts.SharedValueRef` and `ts.SharedCount` read a handle and keep nothing (`isBorrow`), so
    `s.value` and `Shared.count(s)` are borrows of whatever `s` borrows.
- **`any`.** A handle boxes under its own descriptor, named `"object"`, whose release and retain
  slots are the handle's routines, kept under own. A tag-free `Shared<T> | null` boxes under the
  handle's descriptor, so a null in it is `typeof` `"object"`. `<Shared<T>>a` goes through
  `___unbox<Shared<T>>`, which checks the name and throws otherwise. The counts round-trip
  exactly under rc and own.

#### Tests and teeth

- **Positive**, under gc, none, rc and own, AOT and JIT (88 runs): `own_shared_basic`,
  `own_shared_user_named`, `own_shared_graph`, `own_shared_containers`, `own_shared_cycle`,
  `own_shared_union`, two from the final review's fix round, two from its second, and one from its
  third:
  - `own_shared_global_field`: handle globals initialised from `holder.child` (born at 2: the
    field keeps its count), `new P().child`, `mkP(..).child` and `mkArr(..)[0]` (born at 1: the
    temporary gave its count back). Each drops the other handle where there is one, reuses the
    memory, and reads through the global;
  - `own_shared_global_init`: `Shared<T> | undefined` from `new Shared(...)`, `let` and `const`
    `Shared<T> | null` from a call, an explicit cast of a call's result, `Shared<T> | undefined`
    and `Shared<T> | number` from a call. Each asserts the global's count of 1, copies it, drops
    the global's handle (the copy's, for the `const`), reuses the memory, and reads what is left;
  - `own_shared_place_two_uses`: one read of `p.child` stored into `q.child` and read in the same
    expression (`Shared.count(q.child = p.child)`, `look(q.child = p.child)`), and the same field
    on both sides; `p.child` is then replaced, and `q.child` is read after the memory is reused;
  - `own_shared_global`: a `const`, a `let` and a `Shared<T> | null` global, each initialised by
    `new Shared(...)` at module level and born with one count. Each `let` global's handle is
    dropped while a copy holds the block, the memory is reused, and the copy is read;
  - `own_shared_unlink`: a list node unlinked from its head (§23.1's `Node`), a parent's child
    taken into a `const` before the field is cleared, and a `Shared<T> | null` const of a `let`
    that is then given another block. In each the `const` is the only handle left when it is read.

  **Under rc and own** (4 runs): `own_shared_count`, with every count after a copy, a drop, an
  outer block's death, a `.value` write, a narrowed union, an `any` round trip, a captured handle
  parameter (two calls), an escaping closure over one (100 iterations), a generator over a handle
  parameter (100 calls), handles copied inside a generator, and (second fix round) a generator
  that reads its handle again after the caller has replaced its own and the memory is reused
  (`2`, `42`, then `1`). The fix round also asserts the
  counts of a `const` union (`own_shared_union`: 4, not 3) and of a `const` read out of a field
  (`own_shared_graph`: 3, under rc and own). **`--verify-ownership`**, under own and rc, is quiet
  on the eleven counted files (22 runs).
- **`-O0`**: test-runner's `-noopt` does nothing in a Release build, so each positive was also run
  by hand with `--opt_level=0` under own and rc, AOT and JIT.
- **Negative** (12). Messages with `--di`; without it the borrower is `'this value'` and the place
  "a field" or "an element", as for every own error:

| test | message |
|---|---|
| `own_err_shared_borrow_write` | `'n' borrows 'a.value' but is used here after it may be released or overwritten` (the §23.3 example) |
| `own_err_shared_borrow_call` | the same, after a call that may drop |
| `own_err_shared_borrow_push` | `'first' borrows 'a.value.items[]' but is used here after …` (a push through another handle) |
| `own_err_shared_borrow_push_call` | `… borrows 'a.value' but is used here after …` (the push inside a callee) |
| `own_err_shared_borrow_other_type` | `'n' borrows 'a.value' but is used here after …` (a write through a `Shared<Base>` alias) |
| `own_err_shared_borrow_reassigned` | `'n' borrows 'a.value' but is used here after …` (the handle itself reassigned) |
| `own_err_shared_moved_in` | `is used here after its value was moved` |
| `own_err_shared_union_owned` | `a Shared<Node> in a union with 'string', which -mm=own does not count, is not supported` |
| `own_err_shared_place_arg` | `'this value' borrows 'p.child' but is used here after …`, at `f(p, p.child)`, where `f` overwrites `p.child` |
| `own_err_shared_place_two_uses` | `'this value' borrows a field but is used here after …`, at `f(p, q, q.child = p.child)`, where `f` overwrites both fields; the same with `--opt` |
| `own_err_shared_place_literal` | `'this value' borrows a field and cannot be stored, returned or captured`, at `{ s: (q.child = p.child) }` (31:15), then `dropBoth(p, q)`; the same with `--opt` |
| `own_err_shared_place_tuple` | `'this value' borrows a field and cannot be stored, returned or captured`, at `[(q.child = p.child), 1]` (31:39), then `dropBoth(p, q)`; the same with `--opt` |

- **Suites:** the Shared set is 126 tests, `ctest -R own` 616 and `ctest -R "own|rc"` 1570, all
  passing on this branch's final build (72, 562 and 1516 before the final review's fix round;
  93, 583 and 1537 after it; 114, 604 and 1558 after the second; 125, 615 and 1569 after the
  third).
- **Teeth**, each a temporary switch, with the JIT cache cleared:

| switch | result |
|---|---|
| Task 1: before the type existed (its RED) | `own_shared_basic` fails under gc and none, `can't resolve name: Shared` |
| Task 2: the payload's release taken out of the handle routine | `own_shared_count` fails under rc and own: `the payload released when the outer block died` |
| (a) the safety net and the retain loop's handle skip removed | nothing fails: the walk's early return keeps handles out first |
| (a) extended: the walk's early return removed too | `own_shared_basic`, `own_shared_graph` and `own_shared_count` under own (AOT and JIT), and `own_shared_count`'s own verifier run, fail to compile |
| (b) a handle root `Owned`, not `NotOwned` | `own_err_shared_borrow_call` compiles |
| (c) the `.value` match removed from `dropsChain` | `own_err_shared_borrow_write` compiles |
| (d) the inline push rule off | `own_err_shared_borrow_push` compiles |
| the element-type comparison restored for `.value` places | `own_err_shared_borrow_other_type` compiles |
| the push rule in `dropsInBody` off | `own_err_shared_borrow_push_call` compiles |
| the `rootEnds` handle clause off | `own_err_shared_borrow_reassigned` compiles |
| the `decideCells` handle return removed | `own_shared_count` under own: `a captured parameter's count given back` (a count leaked per call) |
| the widened handle off | `own_shared_union` is rejected by the named rule |
| the widened handle and the named rejection off (a tagged union is single-owner) | `own_shared_union`: `is used here after its value was moved`; a field loop crashes; the counts drop below rc's |
| fix round: `moveIntoGlobals` erases a handle's retain again | `own_shared_global` under own (AOT, JIT, verifier): `the initializer of global 'maybe' takes a second reference`; the reviewer's `gl1` prints `start 0 0` and crashes |
| fix round: a union global no longer takes over its initializer's reference | `own_shared_global` under rc and own: `a nullable global is born with one count; copied` |
| fix round: identity storage for a bare `Shared<T>` only | `own_shared_unlink` under rc: `the field and the const`; under own it is rejected (`borrows a field but is used here after …`: the place-read check now sees the folded `const`); `own_shared_graph` and `own_shared_union` fail their counts under rc and own |
| fix round: the `placeReadOf` exemption for every handle restored | `own_err_shared_place_arg` compiles |
| fix round: `ts.SharedValueRef` and `ts.SharedCount` out of `isBorrow` | under own, `own_shared_count`: `borrows a field and cannot be stored, returned or captured` (handles copied inside a generator); `own_shared_containers`: `borrows an element and cannot be stored …` (`right[0].value.v`) |
| fix round: a generator over a handle parameter a borrowing state again | `own_shared_count` under own: `the generator's count given back` (a count leaked per call) |
| second fix round: the global claim back to the first round's (a cast of `new Shared` only) | `own_shared_global_init` under rc and own, JIT and AOT: `an optional global from new Shared is born with one count`; the review's `v1` and `v3` print 0, `opt_new_min` 0, `null_call_min` garbage |
| second fix round: any one counted use exempts the whole read again | `own_err_shared_place_two_uses` compiles; the review's `hole.ts` crashes under own (`count 0`) |
| second fix round: a counted store's release of the old value a drop again | `own_shared_place_two_uses` under own (JIT, AOT, verifier): `borrows a field but is used here after …` |
| second fix round: the counted uses not a copy in the borrow walk | `own_shared_place_two_uses` under own (JIT, AOT, verifier) and `own_err_shared_place_two_uses`: `borrows a field and cannot be stored, returned or captured` |
| third fix round: any store of the read counted again (`isCountedHandleReceiver`) | `own_err_shared_place_literal` compiles (`--emit=obj` exits 0); the re-review's `e2_objlit` crashes under own (0xC0000005) |
| third fix round: a store with no retain not marked uncounted (`isUncountedStoreOf` off, the predicate kept) | the same: the store falls through both lists, the read stays exempt, `own_err_shared_place_literal` compiles and `e2_objlit` crashes |
| third fix round: the field-read retain in the global claim off | `own_shared_global_field` under rc and own, JIT and AOT: `a global from a live object's field takes a count of its own`; `globfield`, `tmpfield`, `callfield` crash again (`count 0`) |
| third fix round: the element read left out of the claim | `own_shared_global_field` under rc and own, JIT and AOT: `a global from a temporary array's element is born with one count` |
| fourth fix round: the fall-through restored (a use neither counted nor in `isBorrow`'s list, nor a store, is neither) | `own_err_shared_place_tuple` fails, the only one of the 126 (`--emit=obj` exits 0); the re-review's `c7`, `d4` and `d10` crash under own (0xC0000005) |
| fourth fix round: a return and an `any` box no longer classified before `passesOn` (`isHandOff` off) | `own_shared_containers` under own (JIT, AOT, verifier) fails to compile, at its `any` box (`const boxed: any = h.s`); the re-review's `e6`, `e9`, `b9`, `c2` and this round's return and `any` probes are rejected with `borrows a field and cannot be stored, returned or captured` |

- **What (d) showed:** no existing rule ends an element borrow across a push through another
  handle; the push rule is the only one. Each other §23.3 rule also has its own negative.
- **The containers test is weak as teeth:** several owners keep its node alive, so an early free
  there may read live memory. The counts are asserted in `own_shared_count` and `own_shared_union`.

#### Measured

AOT, `measure.ps1`, in MB. The first program is the plan's: per iteration, three handles, one of
them in two arrays, every block given back at the end of the iteration. The second builds a
two-node cycle (`a.value.next = b; b.value.next = a`) per iteration.

| program | iterations | gc | rc | none | own |
|---|---|---|---|---|---|
| a leaf in two arrays | 20000 | 5.9 | 4.4 | 10.6 | 4.4 |
| a leaf in two arrays | 200000 | 5.9 | 4.4 | 65.9 | 4.4 |
| a two-node cycle | 200000 | 5.9 | 35.2 | 35.2 | 35.2 |

Own and rc are flat and equal; none grows. A cycle leaks under rc and own exactly as under none,
and gc collects it (§23.4).

#### Corpus

`--emit=llvm -mm=own --no-default-lib`, every file of `test/tester/tests`, main's binary against
this branch's, plain and with `--opt`: 453 / 593 and 451 / 593 on both, each file with the same
first error. Two messages (`00types`, `callWithSpread`) name a function by its run-to-run hash
(`FH<n>`); with it normalised, the two lists are identical.

#### Unchanged where it compiles

The 453 files that compile on main were compiled with main's binary and this branch's,
`--emit=llvm -mm=own --no-default-lib`, with §22.7's normalisations; the `[n x i8]` size is
normalised on any line holding an `FH` name. 451 are identical. The other two differ only as
main differs from itself between runs, in the order of a union's members in a synthesized name,
and so in the order of the functions and constants that follow it:

| file | main, 12 runs | this branch, 12 runs |
|---|---|---|
| `00union_null_undefined_nonstrict` (§22.7) | 11 one order, 1 the other | 10, 2 |
| `00union_bin_ops2` (`___bin_op_plus<union<…>>`) | 8, 4 | 7, 5 |

Each binary gives exactly the same two normalised outputs, byte for byte. After the final review's
fix round, the corpus is again 453 / 593 and 451 / 593 with the same first errors, and all 453
files compile to the same normalised IR as main's (in that run both flipping files came out in
main's recorded order). Under `-mm=rc`, 50 of
those files (every ninth) were compared the same way: 49 are identical, and `00union_bin_ops2`
flips the same way (main 8 and 4, this branch 11 and 1, the same two outputs).

After the second fix round the corpus is again 453 / 593 and 451 / 593 with the same first
errors. Under own, 451 of the 453 are identical. `00union_null_undefined_nonstrict` came out in
main's other order, and so did `00funcs`, a third file of the same kind that had not flipped in
the earlier runs: the order of a union's members in `___cast<!ts.union<si32, !ts.boolean>,…>`.
Over 12 runs, main gives 11 and 1 of two outputs and this branch 10 and 2, the same two. Under rc,
49 of the 50 are identical, and `00union_bin_ops2` is main's other output again.

After the third fix round, a spot check of 40 corpus files (every fifteenth, and one more),
`--emit=llvm --no-default-lib` against main's binary under own and rc, normalised as above: all
70 that compile are identical (40 under rc, 30 under own); the other ten fail under own on both
binaries with the same first error. None of the three flippers was in the sample.

After the fourth fix round, another 40 (every fifteenth from the seventh), the same way: 66 of
the 67 that compile are identical; the other is `00union_bin_ops2`, main's other output, under
rc in one run and under own in the rerun on the final build (main gives it 1 time in 10 here
too). The 13 that fail fail on both binaries with the same first error.

#### The final review and its fix round

The final review of the branch found three critical defects and one important one. Every test
then kept a second owner alive, which is how all three critical ones got through. The fix round
fixed all four, each with a test that drops the only other owner before the read, and a tooth
(above):

- **Critical 1, a global handle born at count 0.** `moveIntoGlobals` erased a global's handle
  retain as a move: the global was freed while held (gl1), and a `Shared<T> | null` global was
  rejected. Under rc too, a `Shared<T> | null` global gave back its initializer's reference
  through the cast. Both are fixed (`own_shared_global`).
- **Critical 2, a union `const` folded.** `const c = p.child` and `const c: Shared<T> | null = a`
  held no count and got no check: §23.1's own unlink read freed memory under rc and own. Every
  handle `const` now has storage (`own_shared_unlink`).
- **Critical 3, a handle argument neither counted nor checked.** `f(p, p.child)`, where `f`
  overwrites `p.child`, freed the block `f` was reading. A handle read out of a place is now a
  borrow unless rc counts it, and that program is rejected (`own_err_shared_place_arg`).
- **Important 4, a generator over a handle falsely rejected.** `isBorrow` did not list
  `ts.SharedValueRef` and `ts.SharedCount`. With them listed, such a generator compiled but leaked
  a count per call under own, because its state borrowed the box that holds the handle's cell. A
  handle parameter now makes the box owned (`own_shared_count`).

The re-review found Critical 1 fixed only in part, and one new important defect. A second fix
round, approved by the user, fixed both:

- **Critical 1, the rest.** The first round's claim covered only a cast of `new Shared(..)`. A
  `Shared<T> | undefined` global from `new Shared(..)` (through `ts.OptionalValue`), and a
  `Shared<T> | null` global, `let` or `const`, from a call or a cast of one, were still born at
  count 0: own read garbage and crashed, rc printed count 0. The claim moved to
  `OwnedReturnConsumptionPass`, which sees through every view and knows a call's result
  (`own_shared_global_init`).
- **Important, a counted use exempted its read.** `isCountedHandleRead` exempted a read when any
  one of its uses was counted, so `f(p, q, q.child = p.child)`, where `f` overwrites both fields,
  compiled under own and freed the block `f` read. The exemption is now per use: a read with an
  uncounted use is a borrow, and its counted uses are a copy (`own_err_shared_place_two_uses`,
  `own_shared_place_two_uses`).
- Also: a generator that outlives its caller's handle is asserted (`own_shared_count`).

The second re-review found one more important defect and a gap in this section. A third fix
round, approved by the user, fixed both:

- **Important, a store with no count exempted its read.** Every `ts.Store` of a handle read was
  a counted receiver, so in `{ s: (q.child = p.child) }` the object literal's field, which rc
  makes no retain for, kept the read uncounted, and own crashed after both fields were replaced.
  A store now counts only when rc retained for it (`countedStoreRetain`), and one it did not
  makes the read a borrow (`own_err_shared_place_literal`).
- **The global gap was understated.** A handle global initialised from a field read was born at
  count 0 and crashed under rc and own; #446 hides only the global-from-global case. The claim now
  retains a handle read out of a field or an element (`own_shared_global_field`).

The third re-review found one more important defect, of the same kind: a use outside both lists
fell through. A fourth fix round, approved by the user, inverted the default (Ruling 9):

- **Important, a tuple's element exempted its read.** `[(q.child = p.child), 1]`, in a `const`
  tuple (`as const`, or with a `Shared<T> | null` element), kept the read with no count beside a
  counted store; own crashed after both fields were replaced. Every use of a handle read is now
  classified, a receiver counts only with rc's retain for that very use, and anything else is a
  borrow (`own_err_shared_place_tuple`).

#### Linux

GCC, in WSL, the branch at `bbac9ba8`: it builds with no errors, and
`ctest -R "own_shared|own-err|own-no-counting|own-verify|verify-ownership-own_shared"` passes 213
of 213, the same 213 tests the Windows build lists for that pattern. After the final review's fix
round, at `fad98c53` (the later commits change only documents), it builds with no errors and the
same pattern passes 234 of 234, again the Windows build's count. After the second fix round
(its changes applied to `476c4d8f` as a patch), it builds with
no errors and the same pattern passes 255 of 255, the Windows build's count. After the third fix
round, at `3a86e20d`, it builds with no errors and the same pattern passes 266 of 266, the Windows
build's count. After the fourth (its changes applied to `ceedafa5` as a patch), it builds with no
errors and the same pattern passes 267 of 267, the Windows build's count.

#### Outside the rule

- **The covariance hole.** `Shared<Derived>` is assignable to `Shared<Base>`, so
  `b.value = new Base()` through the `Base` view leaves a `Base` in a `Derived` block. That is
  TypeScript's own unsoundness (mutable covariance) and is not fixed: the `Derived` view then
  reads fields a `Base` does not have (p2c, under gc and rc). Own lets no borrow survive such a
  write: any write through any handle ends every borrow read through a handle (Ruling 4).
- **Conservative rejections**, false errors rather than miscompiles:
  - a function that gives back a handle (a `ts.Release`, or a `ReleaseSlot` of a local holding
    one) loses `__own_no_drops`, so a call to it ends borrows its caller holds;
  - a getter returning `s.value` is rejected (`borrows 's.value' and cannot be stored, returned or
    captured`): `borrowedParam` does not see through `ts.SharedValueRef`;
  - a write through a handle of an unrelated type ends borrows through every handle (p1d);
  - a push or unshift through a handle ends element borrows, inline and in a callee alike;
  - a handle read out of a place and given straight to a call that may drop is a borrow (Ruling
    7), so the call ends it even where the callee retains the handle first. With `p` a parameter,
    `keep(k, p.child)` is rejected when `keep` stores into a field. Copying it into a local first
    (`const c = p.child; keep(k, c)`) is accepted, and counts;
  - a handle read whose count goes into a tuple, beside a counted use, is rejected whether or not
    anything drops it (Ruling 9): `[(q.child = p.child), 1]` with nothing dropped after it (the
    re-review's `c7c`, `c7d`), and `const [a, n] = [(q.child = p.child), 1]` (`d3`, which counted
    1 under rc and own). All three compiled before this round; the plain `[p.child, 1]` was
    already rejected. `const x = p.child; const t = [x, 1]` is accepted, and counts;
  - a return whose retain is not followed by only the scope exit's `ts.Release` and
    `ts.ReleaseSlot` ops (in any order) before the store into the result slot is rejected. One
    such shape was found: a scope exit split across blocks, as a `using` disposal makes it
    (`using r = …; return p.child`, in a function or an arrow), compiled and counted 1 before this
    round and is now rejected ("borrows a field and cannot be stored, returned or captured").
    `return p.child` after `const` and `let` locals, after a single `let`, inside a loop, inside
    `try`/`finally`, from an arrow or a method, into an optional or a union result all compile
    and count as rc does;
  - so is one read of a handle with a counted use and a use given to a call that may drop: with
    `k` an owning local, `f(p, k = p.child)`, where `f` overwrites `p.child`, is rejected even
    though `k` keeps the block alive (the review's `hole2.ts`). `k = p.child; f(p, k)` is
    accepted;
  - a generator over a handle parameter and another parameter that holds a block, that reads the
    handle (`function* g(s: Shared<T>, xs: number[])`), is rejected: the handle's cell makes the
    box own what it holds, and the other parameter is the caller's. With the handle unread it
    compiles (the review's `gen3.ts`; `gen3b.ts` reads it);
  - a generator function expression capturing a local handle (`const it = function* () { yield
    Shared.count(a); }`) is rejected as an escaping closure that cannot move the variable, as for
    any local.
- **`any` is checked by name only.** `<Shared<A>>a` accepts any value tagged `"object"`, so a
  `Shared<B>` passes as a `Shared<A>` and its fields are read as `A`'s. Literals, numbers and
  class instances are rejected. A wrong-type unbox cannot be asserted in a test, because a `let`
  in `try` whose initializer throws is not caught (#456).
- **Through null.** `Shared.count(u)` and `u.value` on an un-narrowed `Shared<T> | null` read
  through null when it is null; neither is a compile error.
- **Misused names.** `Shared.foo(s)` and a bare `Shared` value report only "failed statement",
  with no cause, as `BoxedObject` and `Reference` do. A member other than `value` gets the generic
  "Can't resolve property". When a program has its own `Shared`, the `Shared.count` guard leaves
  one dead op behind.
- **A rejected mixed union** (`Shared<T> | string`) also reports own's ordinary error on the same
  function after the named one (`borrows … and cannot be assigned`).
- **The union check's options.** `ownsHeapMemoryOfUnionMember` builds its `MLIRTypeHelper` with a
  default `CompileOptions`, as `isCopyableString` (§22.7) and `OwnershipFacts`' `ownsHeap` do.
- **rc is no oracle for borrows.** rc takes no retain for `const n = obj.f`, so it runs the
  use-after-free shapes own rejects (`own_err_shared_borrow_write`, p2c, r1) and prints whatever
  the freed memory holds. Own and rc differ there by design.
- **No count is elided.** Own counts every handle as rc does (§23.2's last bullet); a handle own
  could prove moved or borrowed is still counted.
- **A bare `{ }` block** does not end a folded `const boxed: any = h`: §9.30 releases a fresh box
  at the end of its MLIR block, and a bare block in `main` is not one. Loops and functions are
  exact.
- **A handle returned from an imported function is counted one extra**, under rc and own alike. The
  importer's `ts.CallIndirect` through the DLL's global carries no `__owned_result`, so its
  reference is never given back: `const a = make(3)` is at 2. A class leaks the same way through
  an import, before this phase as after (final review, `xm/import_w`). There is no test.
- **Not tested:** a generic `function f<T>(s: Shared<T>)`; nothing rebuilds a `SharedType` with a
  substituted element.
- **Resolved during the phase:** an escaping closure capturing a handle parameter, rejected at
  first, is accepted (its cell holds a count) and asserted in `own_shared_count`; `Shared.count`
  on a narrowed `Shared<T> | null` const, an error at first, is accepted; a tagged
  `Shared<T> | number`, single-owner at first, is counted (Ruling 6).

#### Found, not fixed (every model)

- #453: a push onto an array parameter is lost to the caller (gc sees the old length, rc frees the
  data block the caller still holds). `own_shared_count` keeps its handles in a class for this.
- #454: under rc, `const b = a` of a `let` class value takes no reference, so `a = …` frees what
  `b` reads. A `const` handle has its own storage and is not affected: a bare one from the start,
  a union or optional one since the final review's fix round. Before it, `const c: Shared<T> | null
  = a` was folded, and `a = …` freed what `c` read, under rc and own.
- #455: a `Node | null` cast to `any` crashes the compiler (a null tag). Fixed for
  `Shared<T> | null` only.
- #456: a `let` inside `try` whose initializer throws is not caught.
- #457: a throw in a `try` that declares a local crashes: the final review's t1, t6 and t10 under
  rc and own, and t8 and t9 under rc (own rejects those two).
- #459: a union global of a class (`let G: Node | null = mk()`) gives back its initializer's
  reference through the cast, as a handle one did, so it is freed at once: under own `G.v` reads
  0, under rc it reads reused memory. Fixed for handle globals only, through any view of the
  global's type (`claimHandleGlobalInitializers`), since a general fix changes the IR of programs
  without `Shared`.
- #458: an unbraced `else` whose body is a `for … of` over a generator, inside a loop, fails with
  "operand does not dominate this use" (the re-review's `gen2.ts`; braced, `gen2b.ts` compiles and
  counts as rc does).
- Not filed: an owning local initialised through a view of its type
  (`let b: Shared<T> | undefined = new Shared(..)`, `let e: Shared<T> | null = mk()`) holds two
  counts until the end of its block, where the initializer's reference is given back as a
  discarded temporary (§9.30): `Shared.count` reads one high there (`gl/loc.ts`: 2, under rc and
  own alike). It is balanced, not a leak.
- Not filed: a global whose initializer is a conditional (`let x = on() ? 1 : 2`) is a compile
  error in every model (`'ts.Global' op expects region #0 to have 0 or 1 blocks`), so a handle
  global cannot be initialised by one.
- Not filed: `<Shared<T> | string>a`, an `any` unboxed to a union holding a string, is a
  compile error in every model (`'ts.Cast' op type ['!ts.any'] can't be stored in …`; the
  final review's u1).
- Not filed: a global object literal initialised with a call's result (`let o = { s: mk(4) }`)
  gives the call's reference back at the end of the initializer: the class version reads freed
  memory under rc on main (`cls_obj`), the handle version crashes under rc and own (`objonly`).
- Not filed: a discarded `arr.pop()` does not give back the element it removes, so a handle
  popped out of an array keeps its count (2 after the pop of a two-handle block, under rc and own
  alike). The re-review's `arronly` (a `const` read out of a global array, then `arr = []`)
  prints 2 under rc and 1 under own, as before this round.

## 24. Assignment after a move (#528), 2026-10-06

Phase 1 (§12.1) counted an assignment to a moved-from `let` as a use after the move: rc's
assignment releases what the slot holds before it stores (`ts.ReleaseSlot` right before
`ts.Store`), and after a move that is the receiver's value. So `let b = a; a = new C()` and
`h.c = a; a = new C()` were errors, although Rust accepts both. Now the moved-from slot owns the
new value:

- **An assignment is a new value.** A path from the move through an assignment of the slot
  reaches a new value, as a path through the declaration does. Uses only such paths reach are
  not uses after the move. The assignment's own release goes with the move, since what the slot
  holds is the receiver's. A release only a path through an assignment reaches stays, and
  destroys the new value.
- **Some paths.** A release that the move reaches both holding the moved value and holding a
  newer one is `moved here on some paths only`. For example, `h.c = a; if (x) { a = new C(); }`
  reaches the scope exit both ways (`own_err_reassign_some_paths`). A read on such a path is
  still a use after the move (`own_err_assign_after_move`, now this shape).
- **Receivers per value.** Receivers of one slot that each read a value of their own are
  decided one at a time: `h.c = a; a = new C(); g.c = a` moves twice. Two receivers read values
  of their own when an assignment comes between their reads on every path between them.
  Otherwise the receivers are decided together, as before (§13).
- **Loops.** `keep.push(a); a = new C()` in a loop body moves each iteration's value: the move's
  read is met again only through the assignment. A path from the move back to a use of a read
  that passes through the read again uses the value read again (`readUsedAfter`).

`own_err_borrow_bound_method`, `own_err_borrow_catch_local` and `own_err_borrow_finally_local`
used `let b = a; … a = new C()` to make `b` a borrow, but that is a move now. Each reads `a` once
after `let b = a`, which keeps `b` a borrow. `own_move_reassign` runs the shapes under own, rc
and gc: own holds 4.4 MB, as rc's 4.5 MB. On a quarantine heap it runs clean; with the
assignment's release kept, it is a double free. Corpus: 467 of 628 compile under own, unchanged.

## 25. Immortal globals (G1 of the ray tracer plan), 2026-10-21

A read of a global was a borrow that could not be stored, returned or captured (§21.2): the global
owns its value, and a call or an assignment may give it another. The ray tracer's `Color.white`,
`Color.black` and `Color.background = Color.black` are never given another value, and are returned
and stored everywhere. Such a global is now **immortal**, like a string literal: its block's header
is `HEAP_BLOCK_IMMORTAL`, so every release skips it, and a read of it owns nothing and may go
anywhere - returned, stored into a field or another global, given to a call that keeps it, merged
with a fresh value (`n > 1 ? K.w : new V(n)` is an owned result on both paths; the caller's release
of `K.w` does nothing).

- **Which globals.** OwnershipSignaturePass (`findImmortalGlobals`) marks a `ts.Global`
  `__own_immortal` when it is of a class type that owns a heap block, is not exported (no `export`,
  `import`, `dllexport` or `dllimport`), and is written once, by its initializer. The initializer is
  either the global's own region (`const c = new V(1)`: every `ts.AddressOf` of it only loaded), or,
  for a class static, its region yields only the default and the one `ts.Store` through a
  `ts.AddressOf` of it is in a static constructor a `ts.GlobalConstructor` runs. Every symbol use of
  the global is checked (`SymbolTable::getSymbolUses`), not only `ts.AddressOf`: a static is also
  reached through its class's vtable, and `k.w = x` with `k: typeof K` writes it through a
  `ts.VirtualSymbolRef` that `writtenGlobals` never sees. When any vtable reference is written
  through, no static is immortal. A module-level `export let`, an exported class's static, and a
  global assigned anywhere else stay mortal. Top-level script code is itself a function a
  `ts.GlobalConstructor` runs, so its one straight-line store into a global (`let c: V;` then
  `c = new V(1)` at the top level, or `K.w = new V(1)`) is accepted like a static constructor's.
  The store must run once (fix-up of 2026-10-23, below): one in a loop op, or in a block on a cycle
  of its function's CFG, leaves the global mortal.
- **Which values.** The initializer gives it an object made right there (`ts.New`, an owning
  literal; none it gives back with a `ts.Release`), nothing, or another immortal global's value
  (found to a fixpoint). A call's result never qualifies, whatever its callee: the pass decides
  before it knows which callees return a borrow, and a global's region is no function the
  inference checks. So `static w = id(new V(1))`, `const c = id(new V(1))` and
  `const c = firstOf(list)` stay mortal and keep today's errors (`own_err_immortal_region_call`,
  `own_err_immortal_region_borrow`), as does a borrowed initializer (`static first = K.list[0]`,
  `own_err_immortal_borrowed_init`). Marking a block someone else owns immortal would turn its
  owner's free into a silent leak; marking a temporary the region frees would write the header
  into a freed block, and the global would hold it. Since main's #638 (merged into this branch
  after T8) a global's region calls get their callees' facts (`ownGlobalInitializerCalls`), and a
  region that holds a call's borrowing result is rejected there first ("the initializer of global
  'c' holds the result of 'id', which borrows its argument"); `own_err_immortal_region_call` now
  pins that message. They stay mortal all the same: `findImmortalGlobals` runs first and still
  refuses them. `own_err_immortal_region_borrow` (since the fix-up of 2026-10-23) calls through a
  function value instead, `const c = pick()(list)`, whose callee #638 cannot know, so it again shows
  this refusal: `f`'s `return c` is "borrows the global 'c' and cannot be stored, returned or
  captured".
  Main's two other global checks step aside for an immortal global's read: a global may be given
  one (#636, `let e = c`), and one may be lent to a call that may drop (#639; a static's one store
  in its static constructor is in `writtenGlobals`, so `lend(b, K.w)` was rejected after the merge
  - `own_immortal_global` now runs it).
- **Only the global's own block.** `K.n.c = x` still destroys the old `c`, and a borrow of `K.n.c`
  stays bounded (`own_err_immortal_field_overwritten`).
- **Inference.** `own_facts::isImmortalGlobalRead` is a `ts.Load` of a `ts.AddressOf` the pass
  marked, seen through views. `isImmortalLiteral` includes it, and its retains (a `ts.Retain`, or a
  `ts.RetainSlot` where the slot still holds it) are erased, as a constant's are. `moveIntoGlobals`
  erases a global region's retain of one.
- **Lowering.** The initializing `ts.Store` carries `__own_immortal`; under own its lowering writes
  `HEAP_BLOCK_IMMORTAL` into the block's header when the value is not null. A region that gives
  the global its value is always made a `__cctor` (it would otherwise become a constant
  initializer, with no store), and the store the `__cctor` makes is marked from the global's
  attribute. A static's region, which yields only the default, is left alone: a `__cctor` of it
  would run after the static constructor and put the default back. So every block the global ever
  holds is one no release frees, whatever reads of it are taken to own; the value test makes it a
  block that was the global's to keep.

Not fixed here, and older than this section: a mortal global whose region stores a call's result
that borrows a temporary the region frees (`export const c = id(new V(1))`, or `const c = ...`
read only in place) holds a freed block, and a read of it in place is accepted. The region is no
function, and the call's facts are not pinned there.

`own_immortal_global` runs the static shapes (from another global, returned, stored into a field,
merged with a fresh value, in a loop of 100000) and `own_immortal_global_region` the module-level
ones (`const c = new V(1)`, `let e = c`), under own, rc, gc and none; own emits no counting.
`own_err_global_returned` and `own_err_param_kept_not_owned` now also give the global another
value, which keeps them errors: their programs as they were are accepted. (Since the merge of main,
`own_err_param_kept_not_owned`'s `keep` pushes instead of storing into a field: a keeper that
drops is rejected at the call by #639 before phase 4's message.) The brief's text for
`own_err_immortal_borrowed_init` (`the initializer of global 'K.first' takes a second reference`)
is moveIntoGlobals', which only a region initializer reaches; a static's borrowed initializer is
reported by inference in its static constructor, as before (`borrows 'K.list[]' and cannot be
stored, returned or captured`). Teeth: with the header store skipped, the AOT and JIT runs of both
fail with `0xC0000374` (heap corruption: the first release frees `K.w`, and later reads and
releases use the freed block). With the loop at 30 million (20 million fresh `V`s freed), own
peaks at 4.3 MB, as rc does.

**Fix-up, 2026-10-23: the store must run once.** "Written once" counted store ops, not how often
they run. A static's one store in a loop of its static block (`static { for (1e6) K.w = new
V(i + 1); }`), or a global's in a loop of top-level script code, marked every block it was given
immortal, and the `ts.ReleaseSlot` before each store skipped the old one: own peaked at 589 MB for
a million `V`s of 64 numbers (static block), 602 MB (`let c: V;` written in a top-level loop) and
600 MB (`K.w` written in a top-level loop), rc at 4.4-4.5 MB. A leak, not a use-after-free: no block
was freed early. `findImmortalGlobals` now refuses a store that may run again (`mayRunAgain`: an
enclosing loop op, kept for safety, or a block on a cycle of its region's CFG, which is what the
affine lowering leaves of every loop). Such a global is mortal: each store releases the value it
replaces, and the three programs peak at 4.4 MB under own. A store in an `if` still runs once and
is still accepted. What only compiled because the global was immortal is now an error, as for any
mortal global: `let e = c` after the top-level loop ("borrows the global 'c' and cannot be stored,
returned or captured", #636), and `let e = K.w` after the static block loop ("the initializer of
global 'e' reads a value another owner keeps").

`own_global_loop_store` writes `K.w` in a loop of its static block, `S.w` in a `while` of
top-level code and `T.w` once in an `if` of its static block (returned from `t()`, which only an
immortal global allows), 10000 rounds, under own, rc, gc and none, AOT and JIT; own emits no
counting. `test-own-global-loop-store-mortal` checks its `--emit=mlir-affine`: `T.w`'s
`ts.AddressOf` is marked `__own_immortal`, `K.w`'s and `S.w`'s are not. The module-level form
`let c: V;` is not in the test: a `let` with no initializer warns. Negatives
`own_err_global_loop_store_static` (`return K.w` after a static block loop) and
`own_err_global_loop_store_script` (`let e = K.w` after a top-level loop), "borrows the global
'K.w' and cannot be stored, returned or captured". Teeth (a temporary `OWN_TEETH` switch skipping
`mayRunAgain`, removed): the mortal check fails (`K.w`/`S.w` marked) and both negatives compile.

**The other write guards (final review I4, 2026-10-23).** A global wrongly marked immortal and
later written frees what its reads were taken to own nothing of - a use-after-free, not a leak -
so each guard of "written once, by its initializer" has a program of its own, checked the same
way (`test-own-global-mortal-*`, `--emit=mlir-affine`: `M`'s `ts.AddressOf` is not marked
`__own_immortal`, `I`'s is): `own_global_mortal_export` (`export let M` and an exported class's
static `MK.m`), `own_global_mortal_typeof` (`k.w = I` with `k: typeof M`: the vtable-write guard,
the only one that reads a `ts.VirtualSymbolRef`'s users; a write through `typeof K` is that route,
not another), `own_global_mortal_static_twice` (two stores in one static block) and
`own_global_mortal_static_method` (`this.w = new V(3)` in a static method, the static's only
store: a function no `ts.GlobalConstructor` runs). Each program has one guard between `M` and
immortality. Teeth (one temporary `OWN_TEETH` switch per guard, removed): skipping the `export`
test, `vtableWritten`, the second-store test (`initStore`) or the static-constructor test each
makes its own check fail (`M`, `MK.m` / `M.w` marked) and leaves the other three passing. The
typeof write of a fresh value (`k.w = new V(3)`) is not used: it is an error today (`'this value'
takes a second reference`), and `k.w = M.w` reaches checkGlobalRead's keep backstop.

**A `const` read of a mortal global ends at its assignment (fix-up review C1, 2026-10-23).** Making
those globals mortal exposed an older hole, on main too: `const c = G` names the read itself (a
`ts.Load` of the global's `ts.AddressOf`, `__owned_result_named`, with no retain), which no check
bounded - `let c = G` is a borrowing local, rejected, but the `const` was neither a place read nor
a local. So `const c = K.w; K.w = new V(i + 1); print(c.x)` in a static block's or top-level code's
loop compiled and, now that the store frees the old block, read freed memory (exit 139 on Windows
and Linux); before the fix-up it leaked instead. The same in any function (`const c = G; G = new
V(1); print(c.x)`) read freed memory on main. OwnershipInferencePass now checks each read of a
mortal global (`isGlobalRead`: not immortal, owns a heap block, not a handle; `checkGlobalRead`):
its uses, and the uses of what it produces (walkBorrowed, made again at the read), must not come
after a `ts.Store` or `ts.ReleaseSlot` of that global in the same function ("'this value' borrows
'G' but is used here after 'G' is released or overwritten": the name is the read's, as in the
other own diagnostics, and `ownerName` finds a variable's name only with `--di`). Where the walk
meets a keep, the read is followed into what holds it or rejected (both below). Not here: a call that assigns the global (`const c = G; reset(); print(c.x)`, still accepted; #639
covers such a call only when the read is its argument), and the same with a local
(`g(n, n = mk(2))`, still accepted, and wrong at run time under own before and after). A `const`
of an owning local (`const c = n; n = mk(5); print(c.x)`) is a move, which is sound. Tests:
`own_err_global_const_store_static`, `own_err_global_const_store_script` and
`own_err_global_const_store` (runnable programs; gc prints `1 64 ... 5`). Tooth (a temporary
`OWN_TEETH` switch skipping the check, removed): the three compile, and the built programs exit
139 under `MALLOC_PERTURB_` on Linux. Newly rejected: only programs of this shape; the corpus
keeps its 526 files.

**An object literal that holds such a read is bounded by the global too (fix-up re-review N1,
2026-10-23).** An object literal built in a local (`const o = { v: c }`, `{ v: K.w }`,
`{ p: { v: K.w } }`, `const { v } = { v: G }`) takes the read with no retain: a `ts.Store` into a
`ts.PropertyRef` of the local's value tuple, or the local's initializer. walkBorrowed took that
store for a keep, and checkGlobalRead ended there, so `o.v.x` after `K.w = new V(i + 1)` read the
freed block (exit 139 on Windows and Linux; BASE leaked instead, and the module-global form
`const o = { v: G }; G = new V(1)` read freed memory on main). The literal holds the borrow, not
a second owner, so checkGlobalRead now follows it (`recordLocalHolding`): every use of the local -
its fields' reads, a copy (`const o = b`), a spread (`{ ...b }`, through `ts.DeconstructTuple`),
a destructuring - and what those lead to are uses of the read, made again where the local is
declared. A value made from the borrow and stored into another such local makes that local a
holder too. Tests:
`own_err_global_record_static` (the static block's loop) and `own_err_global_record` (a module
global in a function), runnable, gc printing `1 64 ... 5` and `0 64 ... 5`. Tooth (a temporary
`OWN_TEETH` switch ending the follow at once, removed): both compile, and the built programs exit
139 on Windows and under `MALLOC_PERTURB_` on Linux. The ray tracer is unchanged: the globals its
records read (`Color.*`) are immortal and its `Surfaces.*` are handles, which this check skips.

**Every keep is followed or rejected (fix-up re-review N2, 2026-10-23).** A `let` that takes the
value back out of such a literal (`const o = { v: K.w }; let l = o.v`, `let l: V = o.v`,
`let l = o["v"]`, `let l = <V>o.v`, an interface-typed `let`, `const p = o.v; let l = p`,
`let { v } = o`, `let { v } = { v: K.w }`) was still a keep that ended the walk in silence, and the
`let` itself passes as a borrowing `let` of `o`'s field, bounded by `o`, not by the global: `l.x`
after `K.w = new V(i + 1)` read the freed block (exit 139 on Windows and Linux; the module-global
form on main too). So checkGlobalRead no longer lets a keep go. Each keep the walk meets is
either:
- followed as a holder (`borrowHolderOf`): a local - a `let`, a `const`, an object literal built in
  place, a destructured local - declared from the read, assigned it, or given it in a field of its
  value tuple at any depth. Its uses under the field path that holds the borrow, and what they lead
  to, are uses of the read, made again only where the local is given a new value (its declaration
  when it has an initializer, and a store into that path or a prefix of it; the paragraphs
  below); its own retain and releases
  are its bookkeeping, which the owner checks decide. A sibling field never given the borrow
  (`{ v: K.w, z: Z.z }` with only `o.z` read later) is not followed. A tuple taken apart
  (`ts.DeconstructTuple`) is followed field by field, and a value made from the borrow that
  declares or is stored into another local makes it a holder too. rc's temporary release of a
  call's result that borrows an argument goes with that result, as in checkPlaceRead;
- or rejected: "the value read from 'G' is kept where its end at an assignment of 'G' cannot be
  checked; -mm=own cannot prove it outlives the global's value" - a field of an object, an array,
  a global, a callee that keeps it, a capture, a return. A string read is copied where it is kept
  (§22), so its keeps are not this check's. This is a backstop: almost every keep fails an earlier
  rule first. One shape is known to reach it alone, `k.w = M.w` with `k: typeof M` (a static kept
  through a vtable slot), pinned by `own_err_global_kept_typeof`.

Tests: `own_err_global_let_holder_static` (a static block's loop), `own_err_global_let_destructure`
(`let { v } = { v: K.w }` in a top-level loop) and `own_err_global_let_holder` (a module global),
runnable, gc printing `1 64 ... 5` and `0 64 ... 5`. Tooth (a temporary `OWN_TEETH` switch that
lets a keep go and follows no plain local, removed): the three compile, and the built programs
exit 139 on Windows and under `MALLOC_PERTURB_` on Linux. Newly rejected beside them: the unsafe
shapes, and one correct one - a spread of a holder (`const o = { v: K.w, z: Z.z }; const q =
{ ...o }`, then `q.z` read after the store): `ts.DeconstructTuple` is followed field by field, so
every field of the copy holds the borrow and the copy loses the field path. That is a sound
over-rejection, kept. The corpus keeps 526 files, the ray tracer has no error, and the sibling
field of round 2's over-rejection is accepted again.

**A holder is renewed where it is stored, never at the read (fix-up re-review N3, 2026-10-23).**
The follow above made a holder's uses again at the read as well as at the local's declaration
(the kills the walk reached it with). For a holder declared before the loop that is wrong: `const
o = { v: Z.z }; for (...) { const c = G; print(o.v.x); o.v = c; G = new V(i + 1); churn(); }`
reads the read's block in the next round - the path back from the store goes through
`const c = G`, which renews `c`, not `o.v` - and own accepted it (exit 139 on Windows and Linux;
the module-global form on main too, the static-block and top-level forms since the loop-store
fix-up). So a holder's uses are now made again only where the holder is given a new value
(`holderRenewals`): the local's declaration when it has an initializer (N4, below), and every
store into the field path that holds the borrow or into a prefix of it (the whole local, `o.p` for
`o.p.v`); a store into a sibling field or below the path is not one. Those depend on the holder
alone, not on the way the walk reached it, so a holder is followed once. A value loaded out of the holder is made again only by a path
back through that load, which is itself a use bounded by the renewals; and a store that is a
renewal is also a use of the value it stores, bounded by that value's own renewals, so a store of
a stale borrow into the path is rejected at the store. Reachability goes around the back edge: a
store in one round reaches a use in the next. Accepted by this: the field stored before each
round's use (`o.v = c; print(o.v.x); G = ...`), a second read re-stored after the store
(`G = ...; const d = G; o.v = d; print(o.v.x)`), a fresh value stored into the path or a prefix
of it after the store (`o.v = Z.z`, `o.p = { v: Z.z }`), and a sibling read. Rejected: the use
before the store in the next round (`for`, `while`, `do`, `for-of`, `continue`, a labelled
`continue`, nested loops, a holder in an enclosing block or an outer loop, nested paths, two
globals in one holder), a store on one arm only, the holder used after the loop, and a stale value
stored back (`const t = o.v; G = ...; o.v = t`). Tests: `own_err_global_outer_holder` (a module
global, the shape above) and `own_err_global_outer_holder_static` (a static block's loop, the
store in one round and the use in the later ones), runnable, gc printing `42 64 0 64 1 64 2 64 4`
and `1 64 1 64 1 64 5`. Tooth (a temporary `OWN_TEETH` switch making the read and the
declaration the renewals again, removed): both compile, and the built programs exit 139 on
Windows and under `MALLOC_PERTURB_` on Linux. Newly rejected beside them: the unsafe shapes, and
two correct ones, both sound over-rejections - a holder stored on one arm of an inner loop before
every use in practice (`for (j ...) { if (j == 0) { o.v = c; } print(o.v.x); }`), which the CFG
cannot order, and a holder copied into another holder, renewed and copied again
(`h.v = c; outer.h = h; G = ...; h.v = Z.z; outer.h = h; print(outer.h.v.x)`); the corpus keeps
526 files and the ray tracer has no error.

**A declaration without an initializer renews nothing (fix-up re-review N4, 2026-10-23).** `let
l!: V` (or `let l: V`, `let o: { v: V }`) declared inside the loop body stores nothing: tslang
does not reset the slot each round, so it still holds the last round's value. The renewal rule
above counted every declaration, so `for (...) { let l!: V; if (i > 1) { print(l.x); } const c =
K.w; if (i == 1) { l = c; } K.w = new V(i + 1); churn(); }` was accepted and read the freed block
in round 2 (exit 139; the module-global form on main too, the static-block and top-level forms
since the loop-store fix-up). Now a declaration is a renewal only when it has an initializer
(`holderRenewals`, and walkBorrowed's `passesOn` says the same for the non-owning local it hands a
value to - a narrowing for every borrow check, which can only reject more: a place read into such
a `let` in a loop was accepted the same way, `const c = p.v; if (i == 1) { l = c; } p.v = new
V(i + 1)`, and is rejected now). `let l: V | undefined` has an initializer and is a renewal. Still
accepted: the same `let` assigned before each round's use, a `const o = { v: Z.z }` declared in the
loop (its initializer re-stores every round), and `for (x of arr)` into an outer `let x!: V`
(each round's element is stored first). Tests: `own_err_global_uninit_let_static` (the shape
above, in a static block whose only store of `K.w` is the loop's) and `own_err_global_uninit_let`
(a module global in a function), runnable, gc printing `1 64 1 64 4`. Tooth (a temporary
`OWN_TEETH` switch counting every declaration again, removed): both compile, and the built
programs exit 139 on Windows and under `MALLOC_PERTURB_` on Linux. Newly rejected beside them:
only the unsafe shapes; the corpus keeps 526 files and the ray tracer has no error.

## 26. `Shared<I>` of an object literal (G4 of the ray tracer plan), 2026-10-21

The ray tracer's surfaces are module-level `new Shared<Surface>({ ...function fields... })`. The
literal is a boxed object (`ts.New` of its record) seen as the interface through `ts.NewInterface`,
which §18.2 makes a view of the box. MLIRGen gives the payload to `ts.SharedNew` with rc's retain
right before it (`mlirGenRetainCaptured`): a `new C()` arrives carrying its reference and has no
such retain, but the interface view is retained.

- **In a function** the inference already took that retain as the move into the handle (§23.2):
  `let l = new Shared<S>({...})` compiled before this section, and needed no change.
- **In a global's initializer** `moveIntoGlobals` judged the retain and found its value going into
  `ts.SharedNew` rather than the global's `ts.GlobalResult`: `the initializer of global 'g' takes a
  second reference`. `movesIntoResult` now takes `new Shared(x)`'s payload operand as a move: the
  handle's block owns `x` from then on, whatever becomes of the handle, which rc counts. The other
  conditions stay: the value is fresh (the box, seen through its views), retained once, and given
  back by nothing in the region. So `let b = new Shared<S>(a.value)`, a second owner of `a`'s
  payload, keeps the error (`own_err_shared_literal_borrowed`).
- **Not here: calling a function field through the handle.** `g.value.f(3)` and `l.value.f(3)` are
  still `'this value' borrows a field but is used here after it may be released or overwritten`. A
  borrow read through a handle ends at any call that may drop (§23.3), and a call through an
  interface slot has no facts: the signature pass does not name what it reaches. That is not the
  literal's: `new Shared<S>(new K())` with a class `K`, and `p.s.f(3)` on a parameter `p` with no
  handle at all, fail the same way, and a direct method call through a handle (`l.value.m(3)` on a
  `Shared<C>`) compiles. It is G9 (facts through a module-private interface).

`own_shared_literal` makes a global and a local `Shared<S>` of a literal with function fields,
copies the local's handle, reuses freed memory, and reads `roughness` through all three handles,
under own, rc, gc and none, AOT and JIT; `--verify-ownership` is quiet on it under own and rc. Teeth:
with the global's move replaced by a release of the literal (freed while the handle points at it),
the AOT run fails its assertion that the global's literal is alive: its block was freed and reused.

## 27. Owning literals moved anywhere (G3 of the ray tracer plan), 2026-10-22

An object literal was moved only into an owning local or the result (§18.1, §20.2): `len({ start:
p, dir: d })`, `[{ pos: new V(1) }, { pos: new V(2) }]`, `const r: Ray = {...}` and `return { ls:
[...], c: new V(3) }` through an interface type were `an object literal that holds a value it owns
is moved only into a local or returned as it is`, 13 of the ray tracer's 44 errors. Such a literal is
now a value like a fresh object.

- **The box.** A literal given as an object or an interface is built in a local a `ts.Constant`
  starts, read once, and the read copied into a block of its own: a `ts.New` of a `value_ref`
  filled by that one store, seen as the object (`ts.Cast`) and the interface (`ts.NewInterface`),
  which are views of it (§18.2). `recordRetainOf` now takes that copy as the record's owner, beside
  the result slot and an owning local, when the read goes nowhere else. rc's `ts.Retain` of the
  read is then where the fresh values stored into the fields move in, as for a record built to be
  returned: each value's own check erases its release, which the retain dominates, and reports a use
  of it after the literal (`const p` moved into two literals, `own_err_literal_moved_twice`: `is used
  here after its value was moved`; one moved in a loop it was made outside of: `is moved inside a
  loop`).
- **Where the box goes** is the box's own check, as for any `ts.New`: a call argument is released
  after a callee that borrows it, and moves into one that keeps it (`__own_params`: `keepRay` pushes
  it, and the caller's release goes); an array literal element moves into the `ts.CreateArray`; a
  `const r: Ray = {...}` is the box, released at the end of its scope; a literal returned is the
  result. A literal in an interface-typed field of another literal (`{ inner: { start: ... } }` with
  `inner: Ray`) is a fresh box stored into the outer record, which moves in like any fresh value.
- **`own_facts::isOwningRecord`** names such a record - its read, its box, or a view of the box -
  when every field that owns a block holds what the constant put there, nothing, or a fresh value: a
  new object, a call's result that borrows nothing, a literal, a `const` local folded into the value
  it holds. `holdsConstantsOrMoves` reads its fields through it. A field that holds no block (`null`
  in a `V | null`) is now accepted too; it moves nothing.

Not here, and still `an object literal that holds a value it owns ...`:
- **A `let` read into a field** (`own_err_record_literal_let`). Its slot keeps its release at the end
  of its scope, which nothing cancels: accepting it would free the block twice. A sound narrowing of
  the plan's "a moved local": a `const` local is folded into its value and moves in.
- **A borrowed field**: a parameter (`{ start: a }`) or a field read (`{ start: this.pos }`, also
  `borrows a field and cannot be stored`). The box's release would free the borrowed block. That is
  G2's borrowing record, whose release skips borrowed fields; the ray tracer's `{ start: pos, dir:
  rd }` is one.
- **A literal nested by value** (`const o = { inner: { item: new C() } }` with no interface type),
  §20.6's catch-all: the inner record is copied into the outer one's field, not boxed.

A literal stored into a field of a nullable interface type (`k.r = {...}` with `r: Ray | null`) is
`takes a second reference`: in the IR the pass sees, the box is retained once for the interface and
again for the union, and released nowhere in the function (the program compiles and runs under rc).
Into an `r: Ray` field it compiles.

**A method that keeps its argument or returns a borrow, reached through an interface.** A call
through an interface (`s.take(x)` with `s: Sink`, a `ts.InterfaceSymbolRef` and an indirect call)
is not resolved, so it gets none of the method's facts. The caller released what the method kept,
and released a result that borrows (an argument, `this.v`, `this.items[0]`) as its own: a double
free under own either way, older than this section (`s.take(new V(1))`, `g.get(a)`), which a
literal argument now reached too. The interface's vtable for a class (`KS.Sink..vtbl`) holds the
class's `..instanceOf`, so the signature pass took it for a class vtable and its entries for calls
it resolves. Now `collectInterfaceReached` tells it apart by its name (`<class>.<interface>..vtbl`,
where the class's own `..instanceOf` is `<class>..instanceOf`), and a method it holds that keeps a
parameter or returns a borrow, by its body or by `@keeps` / `@lends`, loses its facts. Its body is
then the error the path that finds it gives - `'<param>' is a parameter, kept here, but -mm=own
cannot make the callers of 'KS.take' give it up`, `... is a parameter, returned here, but -mm=own
cannot tell the callers of 'G.get' that the result borrows it`, `borrows a field (an element) and
cannot be stored, returned or captured` - noted `it is reached through an interface, whose calls are
not given its facts`, which wins over an exported method's `'@lends' or '@keeps' would say it`
(`own_err_interface_keeps_literal`, `own_err_interface_keeps_param`,
`own_err_interface_borrow_param`, `own_err_interface_borrow_element`,
`own_err_interface_borrow_field`). A direct call of such a method (`ks.take(x)`) is refused too
whenever the class implements the interface: its interface vtable is made for the `implements`
clause, cast or not. G9 (facts through a module-private interface) relaxes this for closed
interfaces by giving the interface call the meet of these facts.

`own_literal_moves` runs the shapes (two moved `const`s; a fresh call result beside a moved one; an
array of literals; a `const r: Ray`; a returned interface record holding an array; 1000 literals
given to a callee that keeps them) under own, rc, gc and none, AOT and JIT, and on Linux;
`own_record_literal_given` (was `own_err_record_literal_given`) returns `{ v: new Vec(n) }` as a
`Holder`. Own emits no counting. Teeth: with the moved-in values' releases kept, both AOT and JIT
runs of both fail with `0xC0000374` (heap corruption: the box's release frees the field, then the
value's own release frees it again). A loop of 2 million iterations of every shape (about 10
blocks each) peaks at 4.8 MB under own, as under rc (gc 6.7 MB): the box's release frees its fields.

## 28. A field read through an interface (G6 of the ray tracer plan), 2026-10-22

`let d = r.dir` with `r: R` an interface was `takes a second reference; -mm=own cannot prove a move
or a borrow here yet`, while the same read of a class field borrows (§2.3, phase 3). The field of an
interface is a `ts.InterfaceSymbolRef` - the object's field at the offset the interface's vtable
holds - and `isPlace` named only `ts.PropertyRef`, `ts.ElementRef` and `s.value`. So the read was no
borrow at all: a `let` from it fell through to the second-reference error, and a `const` from it, or
a read used in place, was not checked against anything. That was a use after free older than this
section: `const c = h.c; h.c = new C(9); print(c.x)` with `h: HI` printed a freed block's value. The
overwrite was invisible too: `hi.c = new C(9)` through an interface view of `h` did not end a borrow
of the class field `h.c`, and the signature pass gave `function set(hi: HI) { hi.c = new C(9); }`
`__own_no_drops`, so its callers kept their borrows of `hi.c` across the call.

- **An interface field is a place** (`own_facts::isInterfaceField`: an `InterfaceSymbolRef` whose
  result is a reference; a method is a bound function, not a place). Its read borrows, as a class
  field's does: tenure-checked against the ends of what it was read through and against the drops
  (`checkPlaceRead`), through any chain of field reads and interface views (`i.ray.dir`: the chain
  `dir`, `ray`, parameter `i`). A call ends it only when the call's facts say it may drop - `V.plus`
  does not - and any call does for a root the function does not own, as `own_err_param_read_call`
  pins for a class field. `borrowedParam`, `reachesSharedValue` and the signature pass's
  `reachesOutside` walk through it, so `function dirOf(r: R) { return r.dir; }` returns a borrow of
  `r`, bounded by the caller's `r`, and an overwrite through an interface parameter is a drop. A
  parameter stored through an interface field (`function keep(r: R, v: V) { r.dir = v; }`) is kept
  (`__own_params`), as one stored into a class field is: the caller moves it in. Before, that was
  `is a parameter, kept here, but -mm=own cannot make the callers of 'keep' give it up`.
- **Which field it is** is not known where it is read: the interface's index is not the class's
  position, and a class may declare the field narrower than the interface. So the chain records it
  as `Chain::InterfaceField`, and `dropsChain` matches it with any field overwrite of any position
  and type, and an overwrite through an interface with every field on a chain. Elements and `s.value`
  keep their own matching.

`own_field_chain_let` runs `let d = r.dir` and `let d = i.ray.dir` (a nested interface literal, as
§27 accepts it) with `V.plus` between the reads, a class implementing the interface passed as it,
and `dirOf`'s borrowed result, 100,000 times, then `keep` moving a `const` into a field through an
interface and reading it 1000 times, under own, rc, gc and none, AOT and JIT, and on Linux;
own emits no counting. Negatives: `own_err_iface_field_overwritten` (overwritten through the
parameter), `own_err_iface_field_view_overwritten` (a class field's borrow, overwritten through an
interface view), `own_err_iface_field_call` (a callee given the interface overwrites it: no
`__own_no_drops`), `own_err_iface_field_result` (`dirOf`'s result read after the field is
overwritten), all `borrows ... but is used here after it may be released or overwritten`. Teeth:
with the borrowing `let`'s releases kept (only its retain erased), the AOT run of
`own_field_chain_let` fails with `0xC0000374` (heap corruption: the `let`'s release frees `r.dir`),
and so does the JIT run; with the interface clause of `dropsChain` skipped,
`own_err_iface_field_view_overwritten` compiles and prints a freed block's value. With the call's
`__own_result_borrows` dropped for a callee that reads through an interface field (the caller then
owns `dirOf`'s result and releases `r.dir` while the record holds it), and with the call's
`__own_params` dropped for such a callee (the caller still releases `nv` after `keep` stored it in
`k.dir`), the AOT run fails with `0xC0000374` both times. A loop of 2 million iterations (about 6
blocks each) peaks at 4.3 MB under own, as under rc.

Only fields: `borrowedParam`, `reachesSharedValue` and `reachesOutside` walk through an
`InterfaceSymbolRef` only when it is a field (`isInterfaceField`). A method read through an
interface (`function getM(m: M) { return m.get; }`) is what it was before this section.

Corpus (`test/tester/tests/*.ts` under `-mm=own --no-default-lib`, `--emit=obj`): 519 of 693
compile on Linux before this section and 519 after, the same files; none lost, none gained. On
Windows 522 after (`00catch_value`, `00cppvtable_hidden_return` and `00throw_in_block_debug_info`
compile there and not on Linux, before and after). One file that fails either way fails earlier,
with a more precise error: `00array_assignment5` was `takes a second reference` and is now `borrows
'Test.lineTokens.tokens' and cannot be stored, returned or captured`.

Not here: the ray tracer's `shade` still fails at `let d = isect.ray.dir`, now as `borrows a field
but is used here after it may be released or overwritten`, because `isect.thing.normal(pos)` is a
call through an interface, which has no facts (§27) and so may drop anything the function does not
own. Checking reads through an interface also reports the reads the ray tracer used in place across
such calls (`scene.camera.pos` beside the `getPoint` closure, `scene.lights` given to `reduce` with
a closure, the rest of `shade`): 24 errors before, 33 now, none of them a new kind.

They are G9's (facts through a module-private interface), G5's (`reduce`) and G2's (borrowing
records).

Seen on the way, and rc's own, older than this section: under rc, `const c = h.c; h.c = new C(9);
churn(); print(c.x)` with a class `H` prints 999 (gc prints 0): the `const` takes no reference, so
the overwrite frees what it reads. A `let` is retained.

## 29. A read after a move, through the new owner (G7 of the ray tracer plan), 2026-10-22

`h.c = c; print(c.x)`, `n.v = a; print(a.length)` and `best = v; s += v.x` were `is used here after
its value was moved`: a move (§2.2) allowed no use of the old name after it. Now a use after a move
that only reads the value is a borrow of the place the value moved into, as a read of `h.c` itself
would be (§2.3), and is accepted while that place still holds it (USER decision).

- **What a read is** (`isReadAfterMove`): a field, element or length read through the value, a
  method reached through it, a comparison or arithmetic, `print`, `assert`, a test, or an argument
  the callee borrows (`!isKeptArgument`). What the read produces - a field's reference, a bound
  method - is walked as a borrow's uses are (`walkBorrowed`, starting not borrowed); a call result
  that borrows its argument is the value again and may not be kept. A method reached through it
  whose callee keeps its `this` is not a read. A store through a field read (`h.c = c; c.x = 7`)
  is a use of the borrow, not a keep: the block stays the place's. Anything else after the move -
  a store of the value, an owning `let`, a keeping call, a return, rc's retain for a receiver - is
  still a second owner and keeps the error.
- **Where it moved** (`destinationChain`): an owning local (a declaration, or an assignment), or a
  field or element of a local or a parameter (`this`), described as the `Chain` `checkPlaceRead`
  reads (`chainOf` from the store's place). A move into a call that keeps the argument, a `push`, a
  return, a global's field, a captured variable or a handle's payload is not such a destination.
- **The kill walk** is `checkPlaceRead`'s: the ends of the chain's roots (`rootEnds`) and the drops
  that may destroy what the place holds (`dropsChain`, with the moved value standing in for the
  read's result): an overwrite of the place, matched by position and type, through an interface
  view too (§28); a call whose facts may drop (no `__own_no_drops`) given something that reaches the
  place's root, or any such call for a root the function does not own; a call given a closure; and
  an assignment to a global, for a root the function does not own (below). The read fails when
  some end or drop E has a path move -> E (not through the value's definition, the local's
  declaration or its assignments) and a path E -> the read or anything it produced (not through
  those either, nor through the move again). Together the two are exactly one path from
  the move through E to the read on which the value is neither made again nor moved again; a path
  that moves the same value again without making it again is the `moved inside a loop` error
  already. A read of the old name again after E is no new value - the move leaves the slot as it
  was - so, unlike `checkPlaceRead`, the read itself is no kill. In a loop that makes the value in
  each iteration (`let v = ...; best = v; s += v.x`), the next iteration's `best = v` overwrite is
  reached from the move only through `let v`, so the read in the same iteration stands.
- `movesOf` (a fresh SSA value, a folded `const`) asks it of each read after a taker;
  `checkSlotMove` (an owning local) of every use of a slot read the move reaches, and of each late
  use `readUsedAfter` finds.

`own_read_after_move` runs the straight-line moves into a field (`n.v = a`, `h.c = c`), into an
element (`arr[0] = ce`) and into a field of `this` (`this.f = c` in a method), each read after
same-size allocations; a move into an owning local in a loop that makes the value each iteration;
100,000 iterations of a move into a field and into a local each read after it; and a declaration
that takes a value over (`let q = p; print(p.x); p = mkv(8)`: `q` keeps the first value, §24).
Then it allocates same-size blocks, reads the moved values again and asserts them, under own, rc,
gc and none, AOT and JIT, and on Linux; own emits no counting. Its values come from factories
(`mkc(n)`), so stack promotion leaves them heap blocks. Negatives, all `is used here after its
value was moved`: `own_err_field_alias` (`n.v = [2]` after `n.v = a`), `own_err_union_moved`
(`h.c = null`), `own_err_read_after_move_overwritten` (overwritten on one path),
`own_err_read_after_move_view` (overwritten through an interface view `hv.c = null`),
`own_err_read_after_move_call` (`clear(h)`, which may drop),
`own_err_read_after_move_loop` (the read at the top of a loop reached around the back edge after
the iteration's `best = new V(i)` freed the moved value), `own_err_read_after_move_after_loop`
(a read after a loop that overwrote the destination). `own_err_param_kept_used`, `own_err_push_let`,
`own_err_two_owners` and `own_err_literal_two_records` still fail as before.

`own_err_borrow_bound_method`, `own_err_borrow_catch_local` and `own_err_borrow_finally_local`
(phase 2) made `let b = a` a borrow by reading `a` after it. That read now goes through `b`, which
takes the value over, and `a = new C(2)` frees nothing `b` holds: the programs are correct (they
print what rc prints, on Windows and Linux). They now give `a` to a second `let`, so that each
borrows it, as before.

**An assignment to a global is a drop** for a chain with a root this function does not own (found
in review). `G = mkH()` is `ts.ReleaseSlot(ts.AddressOf @G)` before its store, and the drops took
only a `ts.ReleaseSlot` of a place or a cell. So `function run(h: H) { const c = mk(); h.c = c; G =
mkH(); churn(); print(c.x); }`, called as `run(G)`, compiled and printed 999 (a freed block that
the allocations reused; rc prints 5), and so did the same through `this` in a method of the
global's class. A parameter, `this`, or a value read out of a field or a global may be the global's
old value, or under it, and the assignment frees it. The plain place read had it too, older than
this section: `h.c = mk(); const x = h.c; G = mkH(); print(x.x)`. Now such a `ts.ReleaseSlot` is a
drop, and `dropsChain` takes it for every chain that is not owned (`!chain.owned()`); a local or a
value this function made cannot be the global's. An immortal global's (§25) one store is its
initializer, whose old value is empty, so its `ts.ReleaseSlot` drops nothing. A callee that assigns
a global was already no `__own_no_drops` (`dropsInBody`), so a call of it ends such chains.
Negatives: `own_err_read_after_move_global` (the parameter, `this`, and the assignment in a callee:
`is used here after its value was moved`) and `own_err_field_let_global_overwritten` (the place
read: `borrows a field but is used here after it may be released or overwritten`). Teeth: with the
global's `ts.ReleaseSlot` dropping nothing, both compile but the callee's case; the first prints
garbage (`9.88131e-324`) for the parameter and 999 for `this`, the second `9.88131e-324`, instead
of 5. Still open, every model (rc too): a parameter itself read after the global it came from is
assigned (`function run(h: C) { G = mk(); print(h.x); }` called as `run(G)`): no place is read,
so there is no chain to end. Under own main's #639 (merged into this branch after T8) closes that
one at the caller: a value read out of a global something assigns, lent to a call that may drop,
is an error ("a value read out of the global 'G' is lent to a call that may release or overwrite
it"). That error comes first for `run(G)`, `GK.run()` and `runCallee(G)` too, so since the merge
both negatives give their callees fresh values (`run(mkH())`, `new K().run()`) and pin this
section's callee-side rule alone; the teeth above were measured with the global passed, before
the merge. The original shapes are #639 negatives of their own (fix-up of 2026-10-23), so a
narrower #639 would still meet them: `own_err_global_argument_transitive` (`run(G)`, `run` calls
`reset`, which assigns `G`), `own_err_global_method_transitive` (`G.m()`, `m` calls `reset`),
`own_err_global_this_field_move` (`GK.run()` moves into `this.f`, then assigns `GK`),
`own_err_global_argument_field_let` (`run(G)`, `const x = h.c`, then `G` assigned) and
`own_err_global_kept_field` (`keep(h, g)` stores `c` into `h.c`, `g` assigned by a `reset`
elsewhere), each "a value read out of the global '...' is lent to a call that may release or
overwrite it".

Teeth: with the kill walk skipped (`readsThroughOwner` accepting every read), all seven negatives
compile, and AOT runs with same-size allocations before the read print a freed block's value:
`c.x` prints 0 (overwritten on one path, `clear`, set to null) and 999 (through the view) instead
of 5; `a.length, a[0]` prints `64 6.4649e-312` instead of `1 1`; the loop's sum is 1999 instead of
3; the read after the loop prints 999 instead of 1. The positive cannot show it: by design nothing
in it overwrites a destination before a read. With the rule off, the positive fails to
compile with `is used here after its value was moved`, at every read after a move. A loop of 2
million iterations peaks at 4.5 MB under own, as under rc.

Not here: an assignment that moves on some paths only (`if (v.x > m) { best = v; }`) is `is moved
here on some paths only; -mm=own cannot release it on the others yet` - the local's one release at
the join is on a path that moved it and one that did not. The use-after-move error came first and
hid it. It is G10 (Task 5b, static drop elaboration), not a follow-on of G7; the ray tracer's
`intersections` has this shape, and the `moved here on some paths only` of design G8 is it too.
Matching an overwrite by position and type also matches another class's field of the same type at
the same position (`dropsChain`, as for any place read): `own_read_after_move`'s loop holder keeps
its field elsewhere for that reason.

Corpus (`test/tester/tests/*.ts` under `-mm=own --no-default-lib`, `--emit=obj`, Windows): 522 of
693 before this section, 527 after; none lost. Gained: `00class_recursive`,
`00generator_manual_next2`, `00globals_entry_point`, `00owned_locals`, `00union_value_equality`,
each accepted by this rule. Each prints under own what it prints under rc (Windows AOT) and runs
clean under own on Linux.

The ray tracer (`rtbench_own.ts`) has 33 errors, as at §28, but none of them is `is used here
after its value was moved` any more. Two are `is moved here on some paths only` (G10), which the
use-after-move errors used to hide. The rest are G9's, G5's and G2's (§28).

## 30. A move on some paths only (G10 of the ray tracer plan), 2026-10-22

`let v = ...; if (v.x > m) { best = v; }` was `is moved here on some paths only; -mm=own cannot
release it on the others yet`: where the paths join, `v`'s one `ts.ReleaseSlot` is on a path that
moved the value and on one that did not, and no run-time record of the move is kept (§2.2). Now the
release is placed statically (static drop elaboration, controller ruling; a run-time drop flag was
ruled out): on each edge into the join that holds the value and did not move it, and nowhere on
the paths that moved it. The releases after the join go. Each path frees the value once - through
the new owner where it moved, on the edge where it did not.

- **The state of the slot** (`placeOnUnmovedEdges`): a forward walk of the function's CFG gives,
  before each op of the body, what the local's slot may hold - nothing (before its `ts.Variable`,
  after a release), its value (after the `ts.Variable` or an assignment), or a moved value (after
  the move's acquire). A point that some paths reach holding the value and others having moved it
  is *joined*. Every `ts.ReleaseSlot` that may hold a moved value goes; one that some path also
  reaches with nothing in the slot keeps the error. An assignment's own release before its store
  is one of them: an assignment over a value that may have moved frees the old value only where it
  did not move (`if (k % 3 == 0) { kept = r; } r = new V(k);` in a loop).
- **Where the release goes**: on every edge from a block that ends holding the value, and only the
  value, into a joined block - at the end of the block when it has no other successor (a `then` or
  `else` branch, a loop's latch), else in a new block on the edge. A joined block that some path
  reaches with nothing in the slot, an edge whose branch is not `cf.br`/`cf.cond_br` (an
  `Invoke`), and an edge out of an exception handler (from a landing pad to its
  `EndCatch`/`EndCleanup`) keep the error.
- **The safety condition**: the release moves earlier, from the end of the local's scope to where
  the paths join, so nothing may use the value at a joined point: no read of the slot, and nothing
  read out of it - followed forward through every op's results (a field's reference, a view, a
  bound method, a block argument), through a store into a local (every read of that local), and
  out of a region (a closure, an async body) through the op that holds it. A store of such a value
  into a place or into a captured local is not followed and keeps the error. A call's result is
  followed only when its facts say it borrows that argument (`__own_result_borrows`); anything
  else a call returns is its own block. The move hands the value to its new owner, whose reads are
  its own, and rc's retain for it is erased. A value the slot is given (its initializer, an
  assignment's) is the slot's only once stored, so a use of it before that - the constructor's, or
  one around a loop before the value is made again - is no use of the slot's value.
- **...and past the joined points** (found in review): what was read out of the value before an
  edge released it - a bound method `v.get`, an interface view `const i: I = v` - may not be used
  anywhere reachable from the start of a block the edge released it into, before the read it came
  from (or the op that made the value) runs again: not after a new value is stored into the local
  (`if (c) { b = v; } v = mkv(2); m()`), where the state is no longer joined, and not after the
  scope ends. A read of the slot itself after a new value is stored reads the new one, and stays
  accepted. The error then carries a note at the use: `used here, after the paths join, so it
  cannot be released before they do`.
- A `string` read out of the value counts as a pointer into it, though own copies strings in some
  places: too strict, but sound.
- **One receiver**: the rule applies when the move is the local's one receiver (`decideSlot`).
  With two, `if (c) { a = v; } else { b = v; }`, each move's other paths include the other's, and
  the error stays.
- **Reads after the join**: a read of `v` after the paths join (`if (c) { b = v; } print(v.x);`)
  goes through `b` on the path that moved it (§29), but on the other `v` must free the value after
  the read, where only a run-time record could tell the paths apart: the error stays. So does a
  `let` declared outside a loop, moved on some iterations and read in the next without a new value
  (`moved inside a loop`).
- **Unwind paths** are paths of the CFG (`ts.Invoke`'s unwind edge to a landing pad), and get the
  same treatment: a release on the normal edge into a joined `try` body comes before the call that
  may throw, and a cleanup's `__unwind_only` release of a joined slot goes with the others.
- The releases are recorded during the analysis (which runs twice, dry and real, and tries a move
  quietly before a borrow) and put in only after it, before the erasures (`placeEdgeReleases`).

`own_conditional_move` runs: the brief's loop that keeps the largest value (`best = v` on some
iterations, `v` declared in the loop); a straight-line move taken and one not taken; 100,000
iterations of a move into a local on even iterations, read through the new owner inside the branch
(§29) and freed by the local on odd ones; an assignment in a loop over a value that may have moved;
a value moved into a field on every path and given a new value on some, which that path frees
before the paths join (the old `own_err_reassign_some_paths`); a move into a call that keeps its
argument on one arm of `?:` (the ray tracer's `shade`), and on the right of `&&`, whose skipping
edge carries the `&&`'s value, so the release goes in a block of its own that passes the value on;
and a move in a `try` whose call throws on
half of 20,000 calls, with a `finally`. It prints the same under own, rc, gc and none, AOT and JIT,
and on Linux; own emits no counting. Negatives: `own_err_conditional_move_read` (`b = undefined`
after the join, then `v.x`: `is used here after its value was moved`),
`own_err_conditional_move_outer` (`moved inside a loop but was made outside it`),
`own_err_conditional_move_read_after_join` (`v.x` after the join),
`own_err_conditional_move_field_borrow` (a field of `v` read after the join),
`own_err_conditional_move_catch` (the move inside a `catch`), `own_err_conditional_move_bound_method`
(`const m = v.get` called after the join), `own_err_conditional_move_view` (`const i: I = v` read
after the join) and `own_err_conditional_move_old_view` (both, used after `v` is given a new value
past the join), the last six `is moved here on some paths only`.

The brief's positive read `t` after the join (`if (k % 2 == 0) { b = t; } s += t.x;`). That is the
shape the rule leaves out, so the read moved into the branch (`b = t; s += t.x;`) and before the
`if`, and `s` is 199999 instead of 149999.

`own_err_reassign_some_paths` (a value moved, then given a new value on some paths) and
`own_err_borrow_derived_ref` (phase 2: `let b = a`, then `a = new C(2)` in a loop) became correct
programs: the path that gave `a` a new value frees it before the paths join, and the move into `b`
needs no borrow. They print what rc prints. The first now reads `a` after the join, and the second
gives `a` to a second `let`, so that each borrows it; both fail as before.

Teeth: with the release placed on the edges that moved the value as well, `own_conditional_move`
(AOT, `test-compile-own-`) exits 0xC0000374 (heap corruption), and so do the brief's loop, the `?:`
move into a keeping call and the `try` with a throwing call, each alone; the straight-line
`if (v.x > 0) { best = v; } print(best.x)` prints 0 instead of 1. With the safety condition off,
`own_err_conditional_move_read_after_join` compiles, and a copy with same-size allocations before
the read prints 0 for `v.x` on the path that did not move it, instead of 1; no other check covers
it, since §29 accepts the read on the path that moved. With what is read out of `v` not followed
(only the slot's reads), `own_err_conditional_move_bound_method` and `_view` compile - no older
check sees them - and their AOT runs crash (0xC0000005), where rc prints `1 1`. With the walk
checking only joined points (as first written), both forms of `own_err_conditional_move_old_view`
compile and read the freed old value: the bound method printed `3 4` on one run and crashed
(0xC0000005) on the other, instead of `3 3`; the view printed `3 1002` and `3 1004`. A field of
`v` read after the join, a `const` alias of `v` and a call result that borrows `v` are each also
rejected by an older check (the move ends a place borrow of `v`; two receivers; `borrows the
result of`). Still open, every model (rc prints garbage too): a bound method or a view of a
local used after the local is given a new value, with no move at all (`const m = v.get; v =
mkv(2); m()`).

Corpus (`test/tester/tests/*.ts` under `-mm=own --no-default-lib`, `--emit=obj`, Windows): 527 of
693 before this section, 528 after; none lost. Gained: `00nullish_values`; it prints under own what
it prints under rc (Windows AOT) and runs clean under own on Linux.

The ray tracer (`rtbench_own.ts`) has 31 errors, from 33: both `is moved here on some paths only`
are gone - `intersections` (`closestInter = inter` on some iterations) and `shade` (`pos` and
`reflectDir` moved into `getReflectionColor`, whose facts keep them, on one arm of `?:`). The rest
are G9's, G5's and G2's (§28): 21 `borrows a field`, 5 object-literal moves, 2 `borrows an
element`, 1 mixed return, 1 kept parameter (the `reduce`), 1 `borrows a field and cannot be
stored`.

## 31. Facts through a closed interface (G9 of the ray tracer plan), 2026-10-22

A call through an interface (`ts.InterfaceSymbolRef`, then an indirect call) was not resolved: it
had no facts, so it might destroy anything, and since §27 every method held in an interface vtable
lost its keep and borrow facts, called directly or not. `sc.ts[i].area(r)` ended the borrow of
`sc.ts` it was made through (`borrows an element but is used here after it may be released or
overwritten`), and `s.take(x)` / `g.get(a)` / `i.first()` / `h.hold()` were the §27 errors in the
method's body. Now a call through an interface only this module can see is resolved like a call
of a private virtual family.

- **Closed.** An interface is open when another module can see it - its name is a word of the
  module's exported declarations (`__decls`), or `!ts.iface<@I>` is in the type of an exported or
  declared function or of a global that is not private or is exported or imported (a class's
  fields included) - or when a value of it may be made other than by `ts.NewInterface` over a
  vtable the signature pass reads: the global, a heap copy of it, or the one an object's class
  vtable holds for it. A cast into the interface from another interface opens both (the other's
  members are reached through it); one from `any` or an opaque pointer opens every interface boxed
  into one; a `ts.InterfaceRef` used as a value, a vtable read other than to make a value, an
  exported class whose vtable holds one of its vtables, and a vtable named for it that no value is
  made with all open it. An untraceable `ts.NewInterface` opens every interface.
- **Families.** Each slot of a closed interface (its vtable position) has the functions its
  vtables hold there - a class's method, an object literal's method, never a declaration, never one
  that says `@keeps` or `@lends`. A call through the slot gets `__own_no_drops` when every member
  destroys nothing, and the kept parameters and borrowed result its members agree on: `meetFacts`
  demotes a family that disagrees, as it does a class family, so the borrowing or keeping member's
  body is the error it was (`own_err_interface_family_disagrees`). That is a sound narrowing of the
  plan's "a parameter kept by any member is kept by the call": a member that does not keep a moved-in
  argument has no release for it, which would leak. A `public` method is open (§15) and gives its
  family no facts; an object literal's method is open too (its symbol is a value in the literal).
- **Members keep their facts.** §27's stripping (`interfaceReached`) applies only to the members of
  open families and to vtables no value is made with, so a direct call of a method of a closed
  interface keeps its facts (`ks.take(x)`, `f.first()`).
- **A result that borrows `this`** through the interface is bounded by the interface value:
  `boundThis` sees `ts.GetThis` of an interface-bound method as the object the interface value is a
  view of (`own_err_interface_result_outlives`: the local holding it is given a new value before the
  read).
- **Function-typed fields** (`f: (x: number) => number`, the ray tracer's `Surface`): the slot holds
  the offset of an object literal's field, and the call goes to whatever function the field holds.
  When the interface is closed and every function ever stored into those fields is one this module
  defines - the literal's own (`{ f: function ... }`, in its `ts.Constant`, a closure over locals
  included: its captures are a field of the object), or one stored later through the object or
  through the interface - the call gets their facts. A store of anything else, the field reached
  other than to be read or stored, an object of the literal's type made other than by the literal,
  a function-typed store through an open interface, or a class implementer (its slot is a thunk's
  cell) keeps the call open. In the language as it is, a function stored into such a field is
  always the literal's: `s.f = fn` through the interface is `saving to constant object`, and a
  function without `this` (a parameter, a named function) is held through a thunk, which is open
  (`own_err_interface_fn_field_unknown`).

`own_interface_facts` runs the plan's loop over `sc.ts` (prints 9), a field borrowed across a call
through two class members and an object-literal member, the closed forms of the five §27 negatives
(a kept literal, a kept `const`, a kept value, through the interface and directly; a borrowed
argument, element and field, through the interface and directly), each read after churn, under own,
rc, gc and none, AOT and JIT, and on Linux; own emits no counting. `own_interface_fn_field` calls a
function field through a global and a local `Shared<S>` of a literal, through a parameter with no
handle, and through a handle given as a parameter, each after churn, under the four models (counted,
so not in `own_test`), and `--verify-ownership` is quiet under own and rc. The five §27 negatives
now export their interface (the open form, same texts). New negatives: `own_err_interface_exported_impl`
(the plan's program with `export class B`), `own_err_interface_family_disagrees`,
`own_err_interface_member_drops`, `own_err_interface_from_interface`,
`own_err_interface_result_outlives`, `own_err_interface_fn_field_drops`,
`own_err_interface_fn_field_unknown`, `own_err_interface_fn_field_exported`.

Teeth: with the call's `__own_params` dropped for a call through a closed interface, and with its
`__own_result_borrows` dropped, the AOT and JIT runs of `own_interface_facts` fail with `0xC0000374`
(heap corruption: the caller releases what `KS.take` kept, and what `G.get` lent). With
`__own_no_drops` given to a call through a family one of whose members overwrites a field of its
argument, `own_err_interface_member_drops` (with same-size allocations before the read) compiles and
prints 77777 instead of 5 (gc prints 5); the same for a function field whose function overwrites
`this.v` (`own_err_interface_fn_field_drops`): 77777 instead of 5. The store-of-unknown-origin
condition has no program that reaches it (above), so it has no tooth.

Measured: 2 million iterations of a kept argument, a borrowed result and a function-field call
through closed interfaces peak at 4.4 MB under own, as under rc (gc 6.4 MB).

Corpus (`test/tester/tests/*.ts` under `-mm=own --no-default-lib`, `--emit=obj`, Windows): 528 of
693 before this section and 528 after, the same files.

The ray tracer (`rtbench_own.ts`) has 31 errors, as at §30: none of them is cleared, for two reasons
outside this rule. `Sphere.normal` and `Sphere.intersect` call `Math.sqrt`, whose `sqrt` is declared
without `@nodrops`, so they may destroy anything and the `Thing` families get no `__own_no_drops`;
`checkerboard`'s functions call `Math.floor` the same way. And `Surface` is exported:
`export let shiny` in `module Surfaces` puts the globals and `interface Surface` into `__decls`, so
another module may store its own function into a surface. Measured on copies: with the four C
declarations `@nodrops`, 22 errors before this section and 18 after (cleared: both errors at
`scene.things[i].intersect(ray)` in `intersections`, `isect.thing.normal(pos)` and the
`Vector.minus(d, ...)` after it in `shade`); with the `export`s dropped too, 22 and 14 (also
`thing.surface.value.reflect(pos)`, `diffuse(pos)`, `specular(pos)` and the `roughness` read); with
only the `export`s dropped, 31 and 30 (`specular(pos)`). Neither is one of the plan's two source
edits.

Not here:
- **Kept parameters do not take the union** across members (above): a family whose members
  disagree is demoted, as a class family is.
- **Class families group by (position, name)** across every vtable that holds an `.instanceOf`,
  interface vtables included (unchanged): two closed interfaces with a same-named method at the same
  position demote each other when they disagree. Too strict, but sound.
- A class implementing a function-typed field (`f = (x) => ...`) is held through a thunk's cell and
  stays open.
- **A `public` method** is open (§15: a function that is not private may be called from another
  module), so its effective facts are empty, and a family it is in can agree only on none: a
  `public get(v) { return v; }` beside a private one returning the same borrow demotes both (the
  ray tracer's `Plane.normal` and `Plane.intersect` are `public`). For type exposure this section
  reads only exported and declared functions; `findOpen` does not, yet.
- **`export` inside a namespace** exports the declaration from the module (`__decls` lists `namespace
  NS { interface Th ... }`), so an interface declared `export` in a namespace for use outside it is
  open; one used only inside its namespace is closed.

**After review** (2026-10-22). The two "not here" items above on `public` methods and namespace
exports are now handled as follows, and `own_interface_facts`' churn now keeps 2000 blocks alive, so
a block freed too early is reused before it is read.

- **A function value reinterpreted** carried an interface value into a closed family unseen (found
  in review, a new acceptance that crashed under own): `f2 as any as (o: Op, a: V) => number` with
  `f2` taking an `Op2` let an `Op` value reach `Op2`'s calls, and so did `(() => Op)` cast through
  `any` to `(() => Op2)`. Now a cast that makes a function out of `any` or `unknown`, or sees one
  function as a function of another signature, opens every interface either signature mentions (a
  class's fields included). A function boxed into `any` puts its signature's interfaces among the
  boxed ones, which any unboxing opens. A bound function's `this` and an opaque `this` are not part
  of the comparison: a closure seen as a plain function of the same parameters is the same.
  Negatives `own_err_interface_fn_any_param`, `_result`, `_drops`.
- **C declarations of numbers** (controller ruling): a body-less `declare function` with no `export`
  or import marker (another TypeScript module's declarations carry `export`), not variadic, whose
  parameters and result are numbers, booleans, characters, integers or enums of them, destroys
  nothing: it is given no block of this module. `sqrt`, `floor`, `pow` and `fabs` in the ray tracer.
  The risk, as for `@nodrops`, is a C function calling back into the module: one registered through
  another declaration (`signal(15, onSig as Opaque)`, then `raise(15)`; a window procedure and
  `SendMessage`), or an exported function reached by name. The first is answered by type below (the
  rule is off in a module that casts a function-holding value to a type that holds none, or has a
  call that is not resolved to functions of this module with an argument whose type holds a
  function); the second remains.
  `own_declared_scalar_no_drops` holds a field's
  borrow across `sqrt`, across a function that calls `sqrt` and `floor`, and across a call through a
  closed interface whose member does; `own_err_declared_drops` (`strlen(s: string)`) is unchanged.
- **A `public` method of a class no other module can name** (controller ruling) is not callable from
  another module: the class is not exported (its vtable is private), not a word of `__decls`, and not
  in the type of an exported or declared function or of an exported or imported global (a derived
  class's storage holds its base, so a nameable class makes its bases nameable). `findOpen` and class
  families use the same test (`callableFromOutside`). `own_err_interface_public_exported` (an exported
  class) and `own_err_interface_public_reachable` (a class an exported function returns) keep the
  error; `own_interface_facts` calls `public` methods that return a borrow through a closed interface.
  `own_err_generator_method_this` exports its class to keep what it tests.
- **An executable's exports** (USER decision): a module built into a program - `--emit=exe`, or the
  module the JIT enters (`isExecutable || isJit`; a module the JIT cache compiles beside it has
  `isJit` off) - is imported by no other module, so its `__decls`, exported functions and globals, and
  exported classes open no interface and make no class nameable. What it imports still does: a
  declared function, an imported global, a global with no initializer (an import of a `.ts` module).
  An object file (test-runner's AOT build is `--emit=obj --entry-point`), a DLL and a `-shared`
  library stay as they were; the pass takes the kind as an option, a library when not told.
  `own_interface_exe_exports` (an exported interface and an exported global of it) runs under the
  JIT and as `tslang --emit=exe`, and is the error on `G.get` as `--emit=obj` and `--emit=dll`
  (`own-exe-exports.cmake`). The rule covers interfaces and classes, not functions:
  `callableFromOutside` ignores the kind, so a program's `export function` is still taken as one
  another module calls - never a closed function, a callback-family member or a family member with
  facts of its own. A sound over-rejection (final review M3): the function is treated as a
  library's exported function is.
- **An iterator `next()` through a closed interface** (ruling) gets the family's facts like any other
  call; it is no longer assumed fresh.

Teeth: with function reinterpretations not opening anything, `own_err_interface_fn_any_param` and
`_result` compile and their AOT runs fail with `0xC0000374` (with and without `--opt`), and `_drops`
prints 77777 instead of 5 (gc 5). With every declaration without `export` taken as destroying
nothing, `own_err_declared_drops` compiles, and so does a program that lends a field and passes the
field's object to a declared `free(p: V)`; its AOT run fails with `0xC0000374`. With a `public`
method's class never nameable, `own_err_interface_public_reachable` compiles; with every module taken
for an executable, `own-exe-exports.cmake` fails (the object file and the DLL compile), and so do
`own_err_interface_exported_impl` and `own_err_interface_keeps_param`. The last two rules protect
callers in another module, which no single-module run can show.

Corpus (`--emit=obj`, not an executable): 528 of 693 before this section and after, the same files.

The ray tracer built as a program, `tslang --emit=exe --opt --opt_level=3 --no-default-lib -mm=own
rtbench_own.ts`: 31 errors before this section, 15 after. Cleared: `Camera`'s constructor (three,
`Vector.norm` reaching `sqrt`), `Sphere.normal`, `this` in `Sphere.intersect`, both errors at
`scene.things[i].intersect(ray)`, `isect.thing.normal(pos)` and the `Vector.minus(d, ...)` after it
in `shade`, `thing.surface.value.reflect(pos)`, `.diffuse(pos)`, `.specular(pos)` and the
`roughness` read, the three `camera.forward/right/up` reads in `getPoint`, and one of the two at
`traceRay({ start: scene.camera.pos, ... })`. New in their place: `scene.things[i].intersect(ray)`
is `moved inside a loop but was made outside it` - the call now has the family's facts, and
`Sphere.intersect` and `Plane.intersect` keep `this` and `ray` in the record they return; that is
G2's borrowing record. As an object file (`--emit=obj`) it has 19: the `Surface` calls stay open.

Not here, also:
- A member that says `@keeps` or `@lends` opens its family (above): a sound narrowing of the plan,
  which carried declared facts through closed interfaces too (controller ruling: accepted). Such an
  implementer through an interface stays the error it was.
- An abstract class's interface vtable is always open: too strict, but sound.
- A class value reinterpreted through `any` (`new C() as any as C2`, `C2`'s field an `Op2` where
  `C`'s is an `Op`) is not followed; the program stops at run time under every model, before any
  call through the field.
- A function value retyped inside a `ts.CreateTuple` or `ts.CreateArray` (a `HasF2` element put
  into a `HasF` field) is not seen as a cast, by the member closure or by the C-callback test; no
  program that compiles today reaches it (every such program is stopped by another ownership
  error first).

**After re-review** (2026-10-22):

- **An interface made out of another one hands over its members' arguments.** A `HasF2`
  (`f: (o: Op2, h: H) => number`) seen as a `HasF` (`f: (o: Op, h: H) => number`) - by
  assignment, through `any`, as a parameter - lets a caller hand an `Op` to a function that calls
  through `Op2`; so does a `Runner2` (`go(o: Op2, ...)`) seen as a `Runner`. Now a cast that makes an
  interface value out of another interface or out of `any` opens, with the two interfaces, every
  interface their members mention - the types of their vtables' slots and of what is read out of
  their values, function signatures and class fields included - and theirs, recursively. Null seen
  as an interface makes nothing (it has no vtable), so `return null` from a method typed `Intersection`
  opens nothing. Negatives `own_err_interface_member_types`, `own_err_interface_member_methods`.
- **An override of a method of a class another module can name, or defines,** is called from that
  module through the base's vtable without its facts, whatever the override's own class is: a
  `public keep` of an unexported `Der extends Base`, with `Base` exported and handed out by
  `export function mk(): Base`, kept its argument while an importer released it; and an importer's
  `Der extends Base` of an imported `Base` (abstract or with a body) was called by the library the
  same way. Now a method of a class whose base chain (the first field of a derived class's storage)
  holds a nameable class, or one with no own vtable here, is callable from outside, private or not -
  the private form was accepted before this section too. Negatives
  `own_err_override_nameable_base` (library side), `own_err_override_imported_base` and
  `own_err_override_imported_class` (importer side, the latter a private override of a concrete
  base); the library-and-importer pairs run under rc, gc and none (`test-runner -shared`).
- **C callbacks.** A C function the module declares may call a function of the module back during
  the call, and that function may destroy what a caller borrows (`signal(15, onSig as Opaque)`,
  then `const c = gh.c; raise(15); ... c.x` with `onSig` overwriting `gh.c`). Foreign code can only
  learn of a function of this module through the module's foreign surface, or through a function
  turned into data; so the C-scalar rule holds only in a module whose **entire foreign surface is
  scalar** and that **turns no function into data by the means the pass sees**: a `ts.Cast`, an
  `inline_asm`, a `call_intrinsic` (controller ruling, fix round 5; a closed-world test, in place
  of the three earlier rounds' list of the routes a function value takes out, which each review
  extended). A type is scalar when it is a number, a boolean, a character, an integer or
  float kind, or an enum or a literal of one. Not scalar: an opaque pointer, a reference, a string,
  a class, an interface, an object, a tuple, an array, a union with such a member, a function type,
  `any`, `unknown`, a rest parameter. The foreign surface, every item of which must be scalar:
  - every body-less function - each parameter and the result (void is fine): a C declaration,
    another module's function (declared here with `import`, or a DLL's), a `declare class`'s
    method, a `@cpp` class's `@slot` declaration, and the compiler's own members of a `declare
    class` (`..instanceOf`, `..new`), which take the class;
  - every global declared with no initializer or imported: a `declare var`, a `declare class`'s
    vtable and type-name string (so a `declare class` always turns the rule off: a sound narrowing
    of the ruling's "any member");
  - every `@cpp` class's fields and its bases' (a base is a `@cpp` class of its own), with no method
    of its own with a body: C++ calls one through the C++ vtable;
  - every function found in a library at run time (`SearchForAddressOfSymbol`): the function type
    its address is cast to.

  What turns a function into data, or makes an address of foreign memory:
  - a `ts.Cast` from a type that holds a function to a type that holds none (`any`, `unknown`, an
    opaque pointer, an integer, a reference, anything else but a boolean, which cannot carry one);
    a class or object input counts its fields for every output but a class (its base) or an
    interface (a view of it, read through its vtable only) - `<Reference<u64>>k` with `K { f:
    (s: s32) => void }`, then a load of the field, is counted (final patch; before it only
    `any`, `unknown` and opaque outputs counted the fields, review6r5 g1-g4, f7); a vtable's
    address is the compiler's own and does not count;
  - a `ts.Cast` to an opaque pointer or a reference from a type that is not itself an address of
    memory this module made (`<Opaque>0x1234`, then `Deref(<Reference<fn>>p) = onSig`): a number,
    `any`, a union;
  - an `inline_asm<T>(...)` or `call_intrinsic<T>(...)` (`ts.InlineAsm`, `ts.CallIntrinsicOp`)
    given an operand whose type holds a function, fields counted (final patch; review6r5 h1, h2:
    `inline_asm<u64>("mov $1, $0", "=r,r", onSig)`, an identity intrinsic on `onSig`).

  A type holds a function when it is one, or a union's or optional's member, a tuple's field, an
  array's element or a reference's pointee does, recursively; a class or object does when a field
  does; an interface does when a slot of a vtable its values are made with, past the header, holds
  a method or a function-typed field, or a field's type holds one (recursively: an interface field
  whose interface has methods, a class field with a function field, an array of them) - the header
  `.instanceOf`, the compiler's own, does not count - and an interface no vtable of this module
  makes values with (one a declared function returns) holds one. The test runs after the class
  storages and the interface families are known. No ordinary call is examined: a function value
  given to one can only reach foreign code through a declared symbol, which the surface test
  already judged by its type; the two builtins that are calls of no declared symbol, `inline_asm`
  and `call_intrinsic`, are examined by their operands (above). Not here: a function reached by
  name - an exported function of a library
  module, which another module may hand to C - stays accepted, as it was with `@nodrops` (an
  executable's exports are visible to no module, above). Negatives
  `own_err_declared_callback` (`signal(15, onSig as Opaque)`; `onSig` overwrites `gh.c`, which a
  caller of `raise(15)` borrows), `_global` (`let cb = onSig`), `_method` (`k.m as Opaque`),
  `_tuple` (`reg([onSig, 1])`, compiled only), `_indirect` (`let sg = signal; sg(15, onSig)`),
  `_wrapper` (`reg(signal)`, `reg` calling its parameter with `onSig`), `_class_method`
  (`w.reg(onSig)` on a `declare class`, compiled only), `_interface_field` (a data-only `Box`
  whose field is a `CB` with a method, given to a declared function through a function value,
  compiled only), `_stored_global` (`declare var hook: (s: s32) => void; hook = onSig`, compiled
  only), `_stored_reference` (`Deref(getSlot()) = onSig` through a declared
  `getSlot(): Reference<(s: s32) => void>`, compiled only), `_cpp_class` (`addListener(new
  MyL())`, `MyL` a `@cpp` class whose `on` has a body, compiled only), `_field_reference`
  (`<Reference<u64>>k`, the field loaded as an integer into `signal(s32, u64)`) and `_intrinsic`
  (`call_intrinsic<Opaque>("llvm.launder.invariant.group.p0", onSig)`, then `<u64>`). Cost: a module with any
  non-scalar declaration - a C function taking a string, a struct or a pointer, a `declare class`,
  a `@cpp` class with methods, an imported function - loses the rule for its scalar declarations
  too (explicit `@nodrops` still works); a data-only interface handed to a declared function
  (review6r4 b9) is such a declaration. The ray tracer declares four scalar functions and nothing
  else.

Teeth: with the member closure skipped, `own_err_interface_member_types` and `_methods` compile and
print `5 77777` (gc `5 5`). With the C rule kept on despite an escaping function,
`own_err_declared_callback` prints `77777 9` (gc `5 9`). With a base chain never making a method
callable from outside, every override negative compiles, and the pairs under `-mm=own` fail their
assert (`kept value still there`): the library side AOT, the importer side AOT and under the JIT
(gc passes all of them).

The ray tracer built as a program stays at 15 errors (the same ones); as an object file, 19.
Corpus: 528 before and after, the same files.

**Fix round 3** (2026-10-22). The member types of an interface whose values are all made from class
objects (`const r2: Runner2 = rr`, the vtable read through the object's class) were not read: the
class vtables joined the interface's vtables only after the member closure. They join before it now
(negatives `own_err_interface_member_class`, `own_err_interface_member_any`). The C-callback test is
by type (the bullet above). Teeth: with the class vtables left out of the closure, both new member
negatives compile and print `5 77777` (gc `5 5`); with the type test off, `_global` and `_method`
print `77777 9` (gc `5 9`). The ray tracer built as a program stays at 15 errors; corpus 528, the
same files.

**Fix round 4** (2026-10-22). The C-callback test looked only at direct calls of declarations, so a
declared function called through a value (`let sg = signal; sg(15, onSig)`, a global holding it, a
wrapper's parameter, a C-returned pointer cast to a function), a `declare class` or `@cpp` method
and a foreign interface's member left the rule on (own printed `77777 9`, gc `5 9`). A call is now
judged by what it resolves to (the bullet above), interfaces count by what their vtables hold, and
a class or object cast to `any`/`unknown`/opaque takes its fields along. "An interface holds a
function" was first taken as every interface (controller ruling, option (a)): that turned the rule
off in the ray tracer through `reduce`'s helper, `___array_reduce<Light, Color>`, whose
`f(r, v)` calls its function parameter with a `Light` - `{ pos; color }`, no method, no function
field - and the ray tracer went to 31 errors, the 16 cleared by the C-scalar rule back. Option (b),
by the vtable's slots past the header, recursively through field types, keeps it at the same 15 and
rejects every probe of the review (n7b, n7d, n7e, n7g, n8b, n8c, n9, n1b, n11, n2b, and the earlier
k1-k10, c3, n6, n10, n13, n16, n17, n18). Negatives `_indirect`, `_wrapper`, `_class_method`,
`_interface_field`. Teeth: with the call test back to direct calls of declarations, `_indirect` and
`_wrapper` compile and print `77777 9` (gc `5 9`), and `_class_method` and `_interface_field`
compile. The ray tracer built as a program stays at 15 errors (the same ones); as an object file,
19. Corpus 528, the same files.

**Fix round 5** (2026-10-22). Round 4's test still listed the routes a function takes out of the
module - a cast, a call into foreign code - and the re-review found two it did not list: a
function stored into memory C reads (`declare var hook: fn; hook = onSig`; `Deref(getSlot()) =
onSig` through a declared `Reference<fn>`; an `Opaque` a C function returned seen as a class with
a function field, then a store; a `declare class` with a function field) and a `@cpp` object handed
to C++, whose C++ vtable slots are this module's methods (`addListener(new MyL())`, also as
`Opaque`, also through a C++ method). The test is inverted (the bullet above): instead of naming
what takes a function out, the rule holds only where nothing can - every declared, imported,
`@cpp` or run-time-found symbol is scalar, and no cast turns a function into data or makes an
address out of a number. The enumeration it replaces is gone: no call is resolved for it, no
argument's type is walked. Also from the re-review: a cast to a boolean (`if (s)` on an interface
with methods, `if (cb)` on an optional callback, `cb?.(1)`) no longer turns the rule off; the
earlier text's "no route of the value matters" was false, and the `.instanceOf` slot and a field
holding only defined functions are no longer part of any call test. Negatives `_stored_global`,
`_stored_reference`, `_cpp_class`. Teeth: with the surface test skipped, `own_err_declared_callback`
(`signal(15, onSig as Opaque)`) compiles and prints `77777 9` (gc `5 9`), and the three new
negatives, review6r4 s8 and the `@cpp`-through-an-integer probe (`addListener(<u64>(new MyL() as
Opaque))`, rejected only by the `@cpp` clause) compile. Every probe of rounds 1-4 is still
rejected (review6r1 c3, review6r2 k1-k10, review6r3 n1b n2b n6 n7b n7d n7e n7g n8b n8c n9 n10 n11
n13 n16 n17 n18, review6r4 a1-a8, b2-b8, e1-e3, s1 s3 s5 s6 s8 s9, c1 c2 c3); d1, d3, d4, d5, d6, d6c
keep the rule; so does a program declaring only scalar functions. Newly rejected by the
whitelist, as its cost: review6r3 n0 (an unused `declare function signal(sig: s32, handler:
Opaque)`), review6r4 b1, b1b, b5 and b9 (a class or a data-only interface given to a declared
function). The ray tracer built as a program stays at 15 errors (the same ones). Corpus 528, the
same files.

**Final patch** (2026-10-22, user decision: keep the implicit C-scalar rule with these two fixes,
then stop). Re-review 5 found two ways a function became data that the cast test did not see: a
class or array with a function field cast to an integer or a reference (`<Reference<u64>>k`,
`<Reference<Reference<Reference<u64>>>>ks`, `<u64>ks`; the field then loaded as an integer into
the all-scalar `signal(s32, u64)`; own printed `77777 9`, gc `5 9`) - the fields of a class or
object input were counted only into `any`, `unknown` and opaque outputs, a narrowing left from
round 3 - and `inline_asm<T>(...)` / `call_intrinsic<T>(...)`, calls of no declared symbol that
take any operand (`inline_asm<u64>("mov $1, $0", "=r,r", onSig)`; an identity intrinsic on
`onSig`). Now a cast counts the input's fields for every output but a class or an interface, and
an `inline_asm` or `call_intrinsic` with an operand whose type holds a function (fields counted)
turns the rule off. Negatives `own_err_declared_callback_field_reference` (review6r5 g2) and
`_intrinsic` (h2), both runnable. Teeth, one switch per fix: with the fields counted only into
`any`/`unknown`/opaque again, `_field_reference` compiles and prints `77777 9` under own (gc `5
9`) while `_intrinsic` stays rejected; with the `inline_asm`/`call_intrinsic` clause off,
`_intrinsic` prints `77777 9` while `_field_reference` stays rejected. g1-g5, f7, f9, f10, h1, h2
rejected; d1-d9 and the scalar-only control keep the rule. The ray tracer built as a program stays
at 15 errors (the same ones); as an object file, 19. Corpus 528, the same files.

## 32. Borrowing records (G2 of the ray tracer plan), 2026-10-22

`{ thing: this, ray: ray, dist: d }` returned as an `Intersection` was `an object literal that holds
a value it owns is moved only into a local or returned as it is` (§27): the box's release would free
`this` and `ray`, which are the caller's. So were the ray tracer's call-argument temporaries `{
start: pos, dir: rd }`, `{ start: pos, dir: livec }` and `{ start: scene.camera.pos, dir:
getPoint(...) }`, and `const r: Ray = { start: this.pos, dir: new V(1) }`. Such a literal is now a
**borrowing record** (USER decision): Rust's `struct Hit<'a> { thing: &'a Th, ray: &'a Ray, dist }`.
Its box is owned like any fresh block - by the local that holds it, the temporary, or, returned, the
caller - and its release frees the box and the fields it owns and leaves the borrowed ones to their
owners. What it borrows bounds every use of it. (The plan calls the result "a borrow" of its sources;
a caller that owned nothing would leave every box and its fresh fields to nobody, and the plan's
own interplay - a record moved on some paths only, read through its new owner, released by a
routine that frees only its fresh fields - is an owner's. The bound is a borrow's: the same chain
and the same errors as a call's result that borrows its arguments.)

- **Which literals.** The signature pass classifies each object literal box seen only as interfaces
  (`classifyRecordBoxes`, in every round of the facts fixpoint, since a field's call may turn out to
  return a borrow): each field that holds a block is *owned* - a fresh value, which moves in as in
  §27 - or *borrowed* (`recordFieldSource`): `this` or a parameter, seen through views; a read out
  of a place, or a call's result that borrows an argument; an owning local's value (`let d`); a
  captured variable's value (a closure body's read of a cell); an immortal global or literal. A
  field is never both, and never a string (strings are copied, §22). A box with a borrowed field is
  marked, and so is every interface made of it (`__own_record_borrowed_fields`, the fields' tuple
  positions); a store into a borrowed field keeps nothing (`keeps`, so `this` and `ray` are not
  kept parameters of `hit` any more).
- **The release travels with the value** (the design's builder check). The interface's tag names a
  descriptor of its own (variant `r<i>...`) whose release routine (`tsrelrec_<type>_<fields>`) frees
  the box and the fields it owns and skips the marked ones; a non-escaping closure's box
  (`__own_borrows_captures`) is the precedent. Every release of the record - a temporary's, a
  local's at the end of its scope, §30's edge release, an overwrite of a local, the caller's - reads
  the tag, so records made at
  different sites (`Sphere.intersect`, `Plane.intersect`) may meet in one local. A box that is seen
  as anything but interfaces is not classified (its static object type's routine would free every
  field), and stack promotion refuses a marked box (its `ts.ReleaseContents` would give back every
  field); today's escape analysis refuses every box seen as an interface anyway ("converted").
- **The facts.** `__own_result_borrows` is a set (`DenseI32ArrayAttr` of argument indices, the
  receiver 0 for a method; `own_facts::resultBorrowsAll`). Alone it is a plain borrow and names one
  argument, as before (`resultBorrows` keeps that index; a declared `@lends` of several is a plain
  borrow of each). With `__own_result_record` the result is a borrowing record: the caller owns the
  box, and it may hold a borrow of each argument named. A function's result is a record when every
  heap value it returns is one - a box made there, or a record call's result - through views,
  merges and the owning locals that hold it (`recordParamsOf`), and every field it borrows is a
  parameter's, `this`'s, or a read out of one: those parameters are the set. A record on one path
  and anything else on another (a plain borrow, a fresh object) gives no facts. `closest` returns
  `bestHit`, which holds `sc.ts[i].hit(ray)`'s records: its result borrows `ray` and `sc`.
  `@lends` may name several parameters: `@lends("a", "b") @record pairOf(a, b): Pair { return { a: a,
  b: b } }` (`own_lends`).
- **`@record` (USER decision 2026-10-23).** A declaration seen from another module has no body to
  say whether the result is a record, so `@record` beside `@lends` says it
  (`OWN_DECLARED_RECORD_ATTR_NAME`, a unit attribute). The declaration printer emits it after
  `@lends`; the importer's declaration gets it, and its calls get `__own_result_record`
  (`meetFacts`, and `calleesDeclare` for a call no function resolves). Under own the importer's
  OwnedReturnConsumptionPass counts such a call's result owned (`calleeDeclaresRecord`: by name,
  through a dynamic import's global, or as a method), so the box is released where the caller
  lets it go. Before, the importer took the result for a plain borrow and the box leaked; nothing
  was freed twice. Guessing record-ness from the result type, or from `@lends` naming several
  parameters, is not safe: a plain borrow of an interface argument looks the same, and a caller
  told it owns a record would free that argument. The defining module checks the marker against
  its body (`checkDeclared`): `@record` without `@lends` is an error, `@record` on a body that
  returns a borrow as it is is an error, and a function another module can call (exported, in a
  module that is not an executable) whose body returns a record has to say `@record`. A function
  no other module calls need not: its callers go by its body. An override says the same
  `@record` as what it overrides. Tests: `export_own_record` / `import_own_record` (-shared, every
  model), `import_own_record_outlives`, `own-shared-record.cmake` (the importer releases each box),
  `own_err_record_without_lends`, `own_err_record_not_record`, `own_err_record_exported_unsaid`.
- **Families** meet by agreement, as before (`meetFacts`): members whose sets or record-ness differ
  are demoted. The union of the sets would be sound for records (the caller would bound the result by
  more arguments), but not across a record and a plain borrow, whose callers release different
  things; agreement is what the ray tracer needs (`Sphere.intersect` and `Plane.intersect` both lend
  `this` and `ray`).
- **Where the record is made** (`classifyRecords`, `checkBorrowingRecords`): every use of it and of
  what is read out of it, through its views, merges and the locals that hold it (`walkRecord`), is a
  use of each borrowed field's source. A read out of a place has them as its own uses
  (`walkBorrowed`, through the field's store, now a borrow), so a drop of the place before a use is
  `borrows a field but is used here after ...`. An owning local's value is bounded by the local's
  ends (`borrows 'd' but is used here after 'd' is released or overwritten`), a captured variable's
  by whatever may assign it (an assignment through a cell, a call that may drop), a parameter the
  function keeps by its moves. `this` and other parameters are the caller's for the whole call.
  rc's retain of the record read goes (`holdsConstantsOrMoves`); its fresh fields move in against it
  as an owning record's do. The record stored anywhere but a local (a field, an element, a global, a
  capture), given to a callee that keeps it, or written through (`r.start = ...`: the overwrite would
  release a borrowed value, and the release would leave the one written) is `... and cannot be
  stored, returned or captured`. Returned, every field it borrows must be a parameter's that the
  function's facts lend; a local's value, or a function whose facts are lost (an exported one
  without `@lends`), is the same error.
- **Where a call returns one** (`checkRecordResults`): the result is fresh (`isFresh`, whatever the
  call goes through), so its moves and releases are the ordinary ones - `let h = ...`, §30's move
  into `bestHit` on some paths with its release on the other edges, §29's reads through the new
  owner - and each release is the per-site routine. It may hold a borrow of each named argument,
  somewhere under it, so the chain `checkPlaceRead` builds for a borrowing call, now over all of
  them (`chainOf`), bounds every use: `borrows the result of 'A.hit' but is used here after it may
  be released or overwritten`. Kept, it is `borrows the result of ... and cannot be stored`;
  returned, its arguments must be parameters this function lends. rc's retain of a fresh argument
  right before the call is no end there: the box's own check decides it. A call's result that
  records a value is followed by `derivedFrom` and by §30's taint. A method reached through an
  interface is named `<interface>.<method>` in these errors (`own_err_interface_result_outlives`
  reads `the result of 'First.first'`).
- **Iterators** (§31): no `next()` or field function whose result is a record is marked
  `__own_fresh_result`; a record's bound applies.

`own_borrowing_record` is the brief's program: `A.hit` and `B.hit` (`null` on some paths) return
records through the closed interface `Th`, and `closest` keeps the best in a local and returns it,
1000 times with churn between, under own, rc, gc and none, AOT and JIT, and on Linux; own emits no
counting. `own_borrowing_record_arg` covers the addendum: call-argument temporaries with two
parameters, a parameter and a fresh value, a field read and a fresh value, an owning local's value
and a captured parameter (the ray tracer's `addLight`), a `let` read into a field, `const` and `let`
locals holding a record, and a method returning one; the borrowed values are read again after the
records are gone. The brief's program as written does not compile in any model (`k = 1` is an
`s32`, which does not match the interface's `number`, and `cond ? null : {...}` cannot be cast to
`Hit`), so the test declares `k: number` and returns `null` from an `if`.

Negatives: `own_err_borrowing_record_stored` (`keep.h = sc.ts[0].hit(ray)`, a global's field),
`_outlives` (`a = new A()` before `h.thing.k`), `_kept_arg` (a temporary given to a callee that keeps
it), `_let_outlives`, `_cell`, `_kept_param`, `_place_overwritten`, `_returned_local`, `_open`,
`_field_written`, `_borrowed_again`. `own_err_record_literal_let` (§27: a `let` read into a field) is
a correct program now - the record borrows the `let`'s value, and the `let` keeps its release - and
is gone; its shape is in `own_borrowing_record_arg`. `own_err_lends_two_params` now reads `'@lends'
of 'pick' names 'a', 'b', but its body returns a borrow of 'a'`. A method's `@lends` reaches its
class's declaration as a 64-bit mask, so since the final review (M4, 2026-10-23) an index past 63 is
an error rather than a borrow an importer silently never learns of: `own_err_lends_index_64` (a
method of 64 parameters lending its last, argument 64), "'@lends' of 'K.pick' names 'p63', but a
method's '@lends' can name only its first 64 arguments, the receiver among them" (the parameter by
its name since the follow-up; it read "names its argument 64" before). The compiler before it accepted
the program; the 63rd parameter (argument 63) is still accepted, under own, rc and gc.

Teeth (temporary `OWN_TEETH` switches, removed): with the per-site routine dropped (every record's
interface given its type's descriptor), the AOT runs of `own_borrowing_record` and
`own_borrowing_record_arg` fail with `0xC0000005` and `0xC0000374`; with only the boxes of records
that are not returned (the call-argument temporaries) given their type's routine,
`own_borrowing_record_arg` fails with `0xC0000374` and `own_borrowing_record` passes (its only
temporary owns both fields). The rules each have a negative that reads a freed block with the rule
off: the caller's bound skipped, `_outlives` prints 77777
and `_stored` 6.95159e-310 (gc 1 and 1); a local's, a captured variable's and a kept parameter's
ends skipped, `_let_outlives`, `_cell` and `_kept_param` print 77779 (gc 3); the return check and
the local's ends skipped, `_returned_local` prints 77777 (gc 1), and `_open` compiles; the kept
check skipped, `_kept_arg` prints 77777 (gc 3), and with the place read's hook skipped too,
`_field_written` exits `0xC0000374` (gc `8 3`); the hook alone skipped, `_place_overwritten`
prints 77778 (gc 4). `_borrowed_again` is the hole review of this rule found: `const h2 = id(h)`
was bounded by `h`'s ends, not by what `h` borrows; before the fix it compiled and printed 77777
(gc 1). With stack promotion's refusal skipped nothing changes: its escape analysis refuses every
box seen as an interface already ("converted").

Measured: 2 million iterations of the brief's loop and a call-argument temporary (about 7 blocks
each) peak at 4.2 MB under own, as under rc (gc 6.2 MB), AOT with `--opt`.

Corpus (`test/tester/tests/*.ts` under `-mm=own --no-default-lib`, `--emit=obj`, Windows): 528 of
693 before this section and 528 after, the same files.

The ray tracer built as a program (`--emit=exe`): 15 errors before this section, 10 after.
Cleared: the records `Sphere.intersect` and `Plane.intersect` return (148, 163); `moved inside a
loop` at `scene.things[i].intersect(ray)` (203: `ray` and the receiver are borrowed now, not kept);
`getReflectionColor`'s `{ start: pos, dir: rd }` (243) and `addLight`'s `{ start: pos, dir: livec }`
(250); `scene.camera.pos` stored into `render`'s literal (279:43). That literal (279:29) is now
`borrows a field but is used here after it may be released or overwritten`: the record holds
`scene.camera.pos` across `traceRay`, which may drop because `getNaturalColor` calls `reduce` with
a closure. New for the same reason: `traceRay`'s `this.shade(isect, ...)`, where `isect` borrows
the result of `intersections` (227). The rest are G5's (`reduce`: 237-239 and 266 across
`getNaturalColor`, the generated `___array_reduce`, `addLight` returning its parameter on one path)
and the `getPoint` read (279:75), present before this section (G8). With `reduce` replaced by a
loop, 2 errors remain: `addLight`'s mixed return and the `getPoint` read.

Not here:
- A record whose box is seen as an object rather than an interface (`const o = { a: this.p }` with
  no interface type) is not classified: its static type's routine would free the borrowed field.
  §27's error stands.
- A field written through a borrowing record is an error even when it is a field the record owns.
- A record's borrowed field may not hold a string.
- Plain borrows are still of one argument (`c ? a : b` is no borrow), and do not flow through an
  owning local assigned in a loop; only records do.

## 33. `reduce` takes its accumulator (G5 of the ray tracer plan), 2026-10-22

`a.reduce((acc: V, v: V) => V.plus(acc, v), new V(0))` was `'init' is a parameter, kept here, but
-mm=own cannot make the callers of '___array_reduce<...>' give it up` (the generated helper's `let r =
init`, then `r = f(r, v)`), and `borrows an element but is used here after it may be released`: the
call through `f` went to an unknown function, which may drop anything. A callback that returns its
accumulator on some paths (the ray tracer's `addLight`, `return col` when in shadow) was `returns a
borrow of its argument on some paths and another value on others`. Now the initial value moves in,
each callback is given the accumulator to keep, and its result is the next one, like Rust's
`fold(init, |acc, x| ...)`, whose closure takes `acc` by value.

- **A parameter returned on some paths only is kept** (`mixedReturnParam`, OwnershipSignaturePass).
  When every value a function returns that holds a block is either one parameter itself (its slot's
  read, through views and merges) or one the function owns, and both occur, that parameter is kept
  (`__own_params`): the callers move the argument in, the path that returns it moves it out, and
  every other path releases it. Not when the parameter is returned on every path (a borrowed result,
  as before), two parameters are returned, a part of one is (`acc.next`), a borrowing record is, or
  the value is a string (copied, spec 22) or a handle. A caller that still uses the argument after
  the call is `is used here after its value was moved` (`own_err_mixed_return`, rewritten:
  `print(pick(c, true).n, c.n)`). A function whose facts are lost (an exported one, one used as a
  value) keeps today's error in its body.
- **Its releases are placed statically** (OwnershipInferencePass, `releaseKeptParamsAtExits`). Such
  a parameter is named in `__own_params_released`: a kept parameter its body never takes into a
  field, an element, a global, a keeping call or an assigned owning local (one it does take there
  keeps the old rule, the move must reach every return: `own_err_param_kept_some_paths`). MLIRGen
  makes no owner of a parameter, so the inference puts a release of its slot before each
  `ts.ReturnInternal` and each `ts.ThrowCall` that leaves the function (not one whose thrown value
  is read out of the parameter), as MLIRGen puts an owning local's. The move's own check
  (`checkSlotMove` with §30's placement) then erases every release the moved value reaches and
  places one on each edge into the join from a path that did not move it; nothing may use the value
  where the paths have joined (§30's safety condition, followed through what was read out of it).
  `pick`'s `V.plus(acc, v)` arm releases `acc` before the arms join; an early return, a `break` out
  of a loop, a return from inside a loop or from a `try` with a `finally`, a `throw`, each release
  it once. No run-time flag, no pointer comparison. A parameter kept this way and moved by two
  receivers (`if (n > 5) return a; if (n < 0) return a;`) is still `used here after its value was
  moved`.
- **A parameter read into an owning `let` that is given another value is kept** (`keeps`): `let r =
  init; ... r = f(r, v)` cannot borrow `init`, since the assignment releases what `r` holds, so
  `init` moves into `r` (the move dominates every return, as for any kept parameter, and rc's
  `ts.RetainSlot` goes with it). Not a string: the local copies it. `___array_reduce`'s source is
  unchanged; this rule makes its accumulator a moved local.
- **Callbacks** (`collectCallbackFamilies`). A function-typed parameter of a closed function (private,
  every use of its symbol a direct call) that is only ever called, and to which every caller passes a
  function this module defines - its symbol, through casts between function types, or a closure over
  a capture box built of it - reaches exactly those functions: the parameter's family. Its members
  are closed when every other use of their symbols is a direct call (`callbackPassing` in `findOpen`),
  and a call through the parameter is resolved to them (`resolveCallback`): it gets the facts they
  agree on, and `__own_no_drops` when none of them drops. A closure's body takes its capture box
  first, so its parameter k is the call's argument k - 1 (`shiftFacts`); a fact about the box itself
  agrees with nothing. `__own_params = [0]` on `___array_reduce`'s `f(r, v)` moves `r` in; the
  assignment's release of the old `r` goes with the move (#528), and `r` owns the result.
- **Agreement by promotion** (`meetCallbacks`). `own_reduce` passes `(acc, v) => V.plus(acc, v)`,
  which only borrows `acc`, and `pick`, which keeps it, to the one `___array_reduce<V, V>`. A kept
  parameter of some member that every other member only borrows - a plain slot, not a string or a
  handle, not lent by the result - is given to all of them to keep; the others release it on every
  path (`__own_params_released`). Any other disagreement demotes every member, as a class family's
  are, and their bodies fail with the note `a function passed in its place disagrees`
  (`own_err_callback_disagrees`: `pick` beside `(acc, v) => acc`). A member a callback family demoted
  is never promoted again, so the meet settles. Class and interface families still agree or demote
  (addendum): promotion is only for callbacks; a closed interface whose implementations both return
  their accumulator on some paths agrees and keeps it, one beside a plain borrow demotes both. A
  promoted member called directly moves its argument in too (`own_reduce`'s `add`;
  `own_err_callback_promoted_used` reads the argument after).
- **The immortal start value.** `a.reduce(pick, K.zero)` moves the immortal `K.zero` (§25) in: its
  retains are erased, and the releases `pick` and `main` make of it are skipped by its header.
  `own_reduce` reduces `[new V(5), new V(7)]` with `pick` from `K.zero`, so `pick` returns `K.zero`
  itself and `main` releases it at its end; after churn `K.zero.x` reads 0, and is asserted. No path
  writes it: `V.plus` makes a new object.
- `___array_reduce` has no foreign surface, and the C-scalar rule (§31) does not read it: nothing
  changed there.

`own_reduce` runs the brief's program (`r.x, q.x` is `6 3`), a reduce that returns the immortal
start value, a named callback `add` that only borrows its accumulator beside `pick` (promoted), and
20,000 iterations of an early return, a return from inside a loop, a return from a `try` with a
`finally`, a `throw` that the caller catches, and a direct call of `add`, each read after churn,
under own, rc, gc and none, AOT and JIT, and on Linux; own emits no counting, and
`--verify-ownership` is quiet on it under own and rc. Negatives: `own_err_mixed_return`
(rewritten), `own_err_mixed_return_join_use` (a field read out of the kept parameter before the join
and used after it: `is moved here on some paths only`), `own_err_callback_escapes` (`pick` also
called through a `let`: `cannot make the callers of 'pick' give it up`), `own_err_callback_disagrees`,
`own_err_callback_promoted_used`. `own_err_borrowed_result_escapes` passed `firstOf` straight to
`apply`; the call through `g` now reaches only `firstOf` and gets its borrowed result, so the program
is correct (it prints what gc prints). It now passes `firstOf` through a `let`, and fails as before.

Teeth (temporary `OWN_TEETH` switches, removed): with the exit releases kept on the paths that
return the parameter (the returning path releases it too), the AOT run of `own_reduce` exits
`0xC0000374` (heap corruption), and so do the brief's two reduces alone and the early-return, loop
and `try` returns alone; with the call through the callback given no kept parameter
(`___array_reduce` releases the accumulator it handed to a callback that keeps it), the AOT run of
`own_reduce` and of the brief's reduces alone exit `0xC0000374`, and the direct returns alone pass.

Measured: 2 million iterations of both reduces and an early return (about 9 blocks each) peak at
4.9 MB under own, as under rc (gc 6.5 MB), AOT.

Corpus (`test/tester/tests/*.ts` under `-mm=own --no-default-lib`, `--emit=obj`, Windows): 528 of
693 before this section and 528 after, the same files.

The ray tracer built as a program (`--emit=exe`): 10 errors before this section, 1 after. Cleared:
`addLight`'s mixed return (253) and the two in the generated `___array_reduce` (its kept `init`, and
the element its call through `f` borrows); and, since the call through `f` now has the closure's
facts and destroys nothing, so does `getNaturalColor`, and the reads held across it or across what
calls it: `scene.lights` given to `reduce` (266), `shade`'s three (237-239), `traceRay`'s
`this.shade(isect, ...)` (227) and `scene.camera.pos` in `render`'s literal across `traceRay`
(279:29). Left: the `scene.camera` argument of `getPoint` in `render` (279:75, G8), now reported
against `Color.toDrawingColor(color)` (280:25), which builds and calls its `legalize` closure and so
may drop. It was there before §32. It does not reproduce in three smaller programs (a closure over
`camera` called in a loop beside a call that may drop, inside a record argument, in a method), and
a method in place of the closure fails the same way, so it is not the closure's call.

On a scratch copy whose only other change is `legalize` made a module-level function (so
`toDrawingColor` destroys nothing; not one of the plan's source edits), the program compiles under
own as an executable and runs: checksum 38837369 as under rc, exit 0, peak 4.9 MB as rc's 4.9 MB on
Windows, and 4.3 MB (rc 4.4 MB) on Linux, where glibc reports no bad free.

**After review** (2026-10-22). The `let` rule above accepted a read of freed memory (found in review):
`const inner = p.v; let r = p; r = new N(new V(2)); ... inner.x` moved `p` into `r`, and the
assignment freed `p` and `p.v` while `inner` still read them (own printed 779000, gc 501500; BASE
rejected the program, since `p` was not kept). The same shape with a field as the new owner, `const
inner = p.v; h.n = p; h.n = new N(...); inner.x`, was wrong before this section too (BASE and HEAD
both printed 779000): a borrow read out of a kept parameter had no ends at all - the parameter is the
caller's for the whole call, unless the function keeps it. Now a borrow read out of a parameter the
function keeps, before the parameter moves, is bounded by the new owner from the move on, as a read
through that owner would be (§29's destination chain, `keptParamOutlives` in `checkPlaceRead`): the
owner's ends and the drops of its place after the move end it. A move into anything that is not such
a place (a call that keeps it, a return, a push) ends it at the move, and so do the parameter slot's
releases (the exit releases above). A borrow read before the move and used while the new owner
still holds it is accepted (`h.n = p; churn(); inner.x`, and `let r = p; inner.x + r.v.x` before `r`
is given another value). Negatives `own_err_kept_param_let_overwritten` and
`own_err_kept_param_field_overwritten` (each built through a `@noinline` factory, with churn: an
argument made at the call site can be stack-promoted, and its release then frees nothing): `borrows
a field but is used here after it may be released or overwritten`. Tooth: with the bound skipped, both
compile, print 779000 instead of 501500 (gc), and fail their assertion. `own_reduce`, the ray tracer
(1 error, the same) and the corpus (528, the same files) are unchanged. It narrows one shape that
was correct: a borrow read out of `p` and used in a `finally` after `return p` (the return ends it at
the move, though the caller then holds the value) is now rejected; no suite test or corpus file has
it.

**After re-review** (2026-10-22). A call that keeps one argument and is given the same value again
through another operand let the callee free what it still borrowed: `f(p, q) { const inner = q.v;
let r = p; r = new N(...); inner.x }` called as `f(n, n)` printed 631370 (gc 501500), a variant
crashed, and the same with `n` an owning `let` printed 639140 - BASE rejected them, since `p` was not
kept, and the `let` rule above made them reachable. The same through a field keep (`f(h, n, n)` with
`h.n = p; h.n = new N(...)`, then `q.v`) was wrong before this section too. The caller's
use-after-move checks skipped the taker itself as a use: `movesOf` for a fresh or folded value,
`checkSlotMove`'s reads for an owning slot. Now (`callGivenAgain`) a call that keeps an argument is a
use after the move when another of its operands - the value again, a view of it, something read out
of it (`f(h, n, n.v)`), another read of the same slot, its receiver - may point into the moved block:
`is used here after its value was moved`. `___array_reduce`'s `f(r, v)` passes the accumulator once.
Negatives `own_err_kept_and_borrowed_arg` (a folded `const`), `_slot` (an owning `let`) and `_field`
(the field keep), each through a `@noinline` factory with churn and an assert. Tooth: with the check
skipped, the first two print 779000 instead of 501500 and fail their assertion, and the third crashes
(0xC0000005). `own_reduce`, the ray tracer (1 error, the same) and the corpus (528, the same files)
are unchanged.

**After the second re-review** (2026-10-22). One value at two kept positions of one call (`f(n, n)`
with both `p` and `q` kept: two `let`s given other values, a field and a `let`, two fields, two
pushes) is two owners of one block, each of which frees it; `callGivenAgain` passed over every kept
operand rooted at the moved value. Now it passes over one - the move itself - and a second is a use
after it (`is used here after its value was moved`). The `let` forms printed wrong or crashed only
since this section's `let` rule (BASE rejected them); two fields and two pushes were wrong at BASE
too. Negatives `own_err_kept_twice` (two `let`s), `_field` (a field and a `let`), `_push` (two
pushes), each through a `@noinline` factory with churn and an assert. Tooth: with every kept position
passed over again, they exit `0xC0000374`, `0xC0000005` and `0xC0000374`. Distinct objects (`f(n,
m)`, `f(h, n, m)`) stay accepted. Also, for an owning slot the other reads counted are only those
that reach the call without crossing an assignment of the slot (or its declaration): `const arr =
[n]; n = mk(5); f(n, arr)`, which reads the old value into `arr`, is accepted again, as before the
first re-review. That is not "holds the moved value" by itself; it is, once the move's own read is
held to the same condition (the third re-review, below).

**After the third re-review** (2026-10-22). The read filter above dropped every read of the old value
when the slot was assigned inside the call's own argument list, the move's read among them:
`let n = mk(i); f(n, n, (n = mk(5), 0))` with `p` kept and `q` borrowed was accepted and crashed
(0xC0000005; the previous round rejected it). Underneath, the assignment alone is wrong, with no
second operand: `f(n, (n = mk(7), 0))` with `p` kept in a `let` (BASE rejected it, since `p` was not
kept) or in a field (wrong at BASE too) frees the old `n` at the assignment's release and then hands
it to `f`. Now `checkSlotMove` rejects a move whose own read does not reach the taker without
crossing an assignment of the slot (`!reachableAfter(load, taker, kills)`; the taker itself may be
the slot's assignment, `n = n`, which is the destination, not an assignment between), reporting the
assignment between them as `is used here after its value was moved` (the call is "value moved here"). That
test asks whether every path from the read to the call is assigned; it misses an assignment on one
arm (the fourth re-review, below), and the read filter is not made exact by it: a read made after
the assignment on the assigning arm holds a new value and still passes (an over-rejection, harmless).
`f(n, (n = mk(i), 0), n)` (the old `n` kept, the new one borrowed) stays rejected,
by this check. Negatives `own_err_kept_arg_assigned_in_call` (`f(n, (n = mk(7), 0))`, `p` kept in a
`let`) and `own_err_kept_and_borrowed_arg_assigned_in_call` (`f(n, n, (n = mk(5), 0))`, `q`
borrowed), each through a `@noinline` factory with churn and an assert; gc and none print 501500.
Tooth: with the check skipped, both, and the field-keep form, compile and exit `0xC0000005` with no
output. `const arr = [n]; n = mk(5); f(n, arr)`, distinct objects at two kept positions, `own_reduce`,
the ray tracer (1 error, the same) and the corpus (528, the same files) are unchanged.

**After the fourth re-review** (2026-10-22). The check above was an all-paths test: with the
assignment on one arm of a conditional in the argument list - `f(n, i % 2 == 0 ? (n = mk(7), 0) :
0)`, the same through `&&` or `||`, or with an `await` on the other arm - the other arm is a path
from the read to the call that crosses no assignment, so the move was accepted, and on the assigning
arm the old `n` was freed before `f` kept it (exit 139; BASE rejected the `let` forms, and the
field-keep form was wrong at BASE too). Now the test is existential: the move is rejected when some
assignment of the slot other than the taker lies on a path from the move's read to the taker
(`reachableAfter(load, a, varOp) && reachableAfter(a, taker, {load, varOp})`), reported at that
assignment with the same text; the all-paths test stays as a fallback (reported at the taker), and a
read and taker `reachableAfter` cannot order (one statement kept as a region, or outside the body;
unreachable today, since the moved-in-loop check takes those first and no op of a body holds a
region by this pass) now refuse instead of passing. Shapes that stay accepted, and why the predicate
leaves them alone: an assignment before the read, with the read between it and the call (`n = mk(i);
f(n, 0)` in a loop over a slot declared outside it, `f((n = mk(i), n), 0)`, `const arr = [n]; n =
mk(5); f(n, arr)`); an assignment after the call, conditional or not (the way back to the taker
crosses the read, or the declaration); `n = n`, `n = (n = mk(5), n)`, `n = acc(n, i)`. Negatives
`own_err_kept_arg_assigned_on_arm` (`i >= 0 ? (n = mk(7), 0) : 0`) and `own_err_kept_arg_assigned_in_and`
(`(i % 2 == 0 && (n = mk(7), true)) ? 0 : 0`), each through a `@noinline` factory with churn and an
assert; gc and none print 501500. Tooth: with the check skipped, both, and the field-keep form with
the conditional assignment, compile and exit `0xC0000005` with no output. `own_reduce`, the ray
tracer (1 error, the same) and the corpus (528, the same files) are unchanged.

Not here:
- A slot assigned inside the argument list of a call that keeps it (`f(n, (n = mk(7), 0))`,
  with or without a second operand, unconditionally or on one arm) crashes under rc as well (exit
  139): rc releases the slot's old value before the store, ahead of the callee that keeps it. That is
  rc's codegen, out of scope here; own rejects these shapes since the third and fourth re-reviews.
- Unwinding out of the function - a call that throws outside a `try` - before a kept parameter is
  moved leaks it, as it leaks an owning local; so does a `throw` of any value computed from it, even
  a number (`throw acc.x`): the release before a leaving throw is skipped when the thrown value is
  computed from the parameter at all, where only one that can hold its block needs it.
- After `r = f(r, v)` moves `r` into the call, `r`'s slot still holds the moved pointer until the
  call's result is stored into it. If `f` throws, the slot dangles. That is benign only because
  owning locals are not released on unwind; a change that releases them there must not release a
  slot whose value a call took.
- A callback passed through a `let`, a field or a global, a member whose symbol is used anywhere
  else (`a.reduce(V.plus, ...)`: a static method is a vtable entry), a parameter that is passed on
  rather than called, an exported or `@keeps`/`@lends` function: no family, as before.
- The default library's `reduce` (design G5: a follow-up).

## 34. A direct call's result, and the ray tracer (G8 of the ray tracer plan), 2026-10-23

After §25-§33 and the merge of main, the ray tracer's own copy
(`benchmarks/raytrace/rtbench_own.ts`: `rtbench.ts` with the plan's two source edits,
`Shared<Surface>` surfaces and a fresh vector from `Plane.normal`) had one error left, built as a
program: `'this value' borrows a field but is used here after it may be released or overwritten` at
`render`'s `getPoint(x, y, scene.camera)`, with `Color.toDrawingColor(color)` on the next line as
the place it may be released. G8's other items (`Camera`'s constructor, `Sphere.normal`, `shade`'s
ternary, `let d = isect.ray.dir`) were cleared by §28-§33.

The cause was not the closure. `walkBorrowed` follows a borrow through every value made from it, and
took the result of any call it is given to as one of them, since a result may point into an
argument. So the vector `getPoint` makes was walked as part of the borrow `scene.camera`; its uses
(through the record given to `traceRay`) were uses of that borrow; and `toDrawingColor` - which
builds and calls its `legalize` closure, so it has no `__own_no_drops` - may destroy anything
reachable from a parameter, `scene.camera` among it. The minimal program is `const p = gp(x,
sc.cam); const c = drop(p); use(p)` with any function `gp` that returns a new object and any `drop`
that may destroy. It compiles when nothing after the result may destroy: with `const legalize` in
place of the ray tracer's `let legalize`, `toDrawingColor` has `__own_no_drops`.

- **A direct call of a function this module defines ends the walk at the call** (`walkBorrowed`,
  OwnershipInferencePass), as a call marked `__own_fresh_result` already did (an iterator's
  `next()`, another own module's `..new`). The test is `isFresh` of its one result - a
  `ts.SymbolCallInternal` or `ts.Invoke` naming a `ts.Func` with a body, with no
  `__own_result_borrows` - and the call has neither `__own_result_bounded` (a generator's state that
  borrows its arguments) nor `__own_result_record` (a borrowing record, §32): those are still
  walked. The call itself is still a use of the borrow for as long as it runs. Sound because such a
  body hands back nothing that points into what it was given but what its facts name: returning a
  borrow of a parameter, of a part of one, or a closure, bound method, array or record over one is
  either a fact its callers see or an error in the body (`borrows a field and cannot be stored,
  returned or captured`, or, when its callers cannot all know the fact, §31's body errors). A string
  is no exception: one read out of a parameter and returned is not copied but is a borrow fact like
  any other (`function nm(c: Cam) { return c.name; }` gets `__own_result_borrows`, so `nm(sc.cam)`
  is still walked, and its result read after `drop(sc)` is "borrows the result of 'nm' but is used
  here after it may be released or overwritten"). Since the fix-up of 2026-10-23 this test and the
  `__own_fresh_result` one are one helper, `callResultIsOwn`.
- **Not widened:** a call through a value, through an interface, through a callback family or to a
  declared function is still walked (isFresh is false), and so is a method call: a class method is
  called through `ts.ThisVirtualSymbolRef`, so `sc.cam.copy()` and `this.point(x, sc.cam)` keep the
  error. That is a sound over-rejection: the signature pass resolves those calls, but the inference
  does not learn that a resolved call's result is owned (a follow-up). The ray tracer's `getPoint`
  is a closure, whose call is a direct call of its lifted function.

`own_fresh_call_result` reads `scene.cam` into a closure's call, a function's call and a record
given to a method (`{ start: scene.cam.f, dir: getPoint(x, scene.cam) }`), then calls a `toDrawing`
that may destroy, churns, and uses all three results, in a loop; it prints 124 under own, rc, gc and
none, AOT and JIT, with no counting under own. Before this section it had three errors (54:26,
54:32, 51:61). Negatives `own_err_call_result_bounded` (a generator's state made from `s.cam`,
iterated after `replace(s)`) and `own_err_call_result_record` (a borrowing record `{ v: c.f, n: 1 }`
made from `s.cam`, read after `replace(s)`), each through a `@noinline` factory with churn and an
assert; gc, rc and none print 5 and 6. Tooth (a temporary `OWN_TEETH` switch, removed): with every
call's result ending the walk, `own_err_call_result_bounded` compiles and exits 139 (gc 5);
`own_err_call_result_record` stays an error through the record's own bound (`borrows the result of
'hit'`). The positive cannot fail with the rule widened: its results really are fresh.

Probes (own as an executable with `--opt`, before the rule / after / with the tooth): a lifted
closure returning `sc.cam.f` or a captured borrowing `let`, an exported function returning `c.f`,
one returning a mortal global's field, a callback family with a member returning its argument, a
method returning `this` or `this.f`, a virtual override returning `c.f`, `id(c).f`, a `V | null` of
`c.f`, an `any` holding `c.f`, an element of an array parameter, a function that makes an unrelated
object and returns `c.f`, a `Shared<V>` parameter's `s.value`, an async function returning `c.f`
or, after an `await`, `new V(c.f.x)`: rejected before and after. An async function that reads
`c.f.x` after an `await`, awaited after the drop: accepted before and after, printing what gc
prints (async is outside this rule either way). A generator over `c.f.x`: rejected before and
after; with the tooth it compiles and crashes. Newly accepted, and printing what gc prints: a
function returning `new V(c.f.x)` or `[c.f.x]`, the same called inside a `try` (a `ts.Invoke`), and
a closure called with `sc.cam` in a loop across a call that may destroy.

**The ray tracer** (`test/tester/own/own_raytrace.ts`: `rtbench_own.ts` at 64x64, ending
`print("checksum:", sum); assert(sum == 606318)`) compiles under own with no errors and prints
checksum 606318, as under gc, rc and none. Built as a program only: it exports `Surfaces.shiny` and
`Surfaces.checkerboard`, which an object file or a DLL opens to other modules (§31, USER decision),
so test-runner's AOT build (an object file it links itself) cannot build it under own. Registered as
`test-jit-own-own_raytrace` (the JIT's entry module counts as a program) and
`test-compile-own-own_raytrace` (`own-exe-run.cmake`: `tslang --emit=exe`, run, match the checksum),
and under gc, none and rc through test-runner, JIT and AOT. Not in `own_test`: its `Shared<Surface>`
handles count (`__tslang_inc_ref`/`__tslang_dec_ref` appear only in the handle's retain and release
routines), as for `own_shared_literal` (§26); for the same reason, and because an
`--emit=mlir-affine` build is not a program, no `--verify-ownership` run.

Measured (`bench.ps1`, 512x512, Windows 11 x64, the machine otherwise idle; two runs of the script,
median of 7 runs each after a warm-up): own 627.5 ms both times (`--no-stack-promotion` 626.4 and
626.8 ms), rc 640.0 and 637.9 ms in the same runs (gc 1295-1299, none 662-666, Rust idiomatic 58,
Rust `Rc` 515, C++ idiomatic clang 54 ms). Peak working set (`PeakWorkingSet64` polled while the
process runs, three runs each): own 4.5 MB, own without stack promotion 4.5 MB, rc 4.5-4.9 MB, gc
6.5 MB, none 840-846 MB. Linux (WSL, ninja Release): own 0.13 s and 4.5 MB maximum resident set, rc
0.15 s and 4.3 MB, checksum ...369, exit 0, three runs each. Errors left in the own copy: 0.

Corpus (`test/tester/tests/*.ts` under `-mm=own --no-default-lib`, `--emit=obj`, Windows): 525 of
702 before this section, 526 after: `00owned_any_boxing` gained (`(<string>boxed[0]).length`: the
generated `___unbox<string>`'s result, a copy, was walked as a borrow of the array's element, which
the array's release at the return ends; built under own it prints what gc prints). None lost.

Not here:
- A method call's result (`o.m(borrow)`, through `ts.ThisVirtualSymbolRef`) is still walked as a
  borrow of its arguments, as is an indirect call's.
- Known limit of the teeth (fix-up, 2026-10-23): with the `isFresh` test alone removed,
  `own_err_field_function_reassigned` (a conservative negative) compiles and prints gc's output;
  with the borrowing-record exclusion alone removed, `own_err_call_result_record` stays an error
  through the record's own bound (`borrows the result of 'hit'`). No probe (review9's 95, T9's,
  the merge review's, every `own_err_*`) was accepted unsafely with either removed, so no negative
  shows a use-after-free that only those two tests prevent.
- The benchmark keeps its two source edits; the unedited `rtbench.ts` does not compile under own (a
  non-goal of the plan).

## 35. A merge of locals made before a loop, bound in it, 2026-10-23

`const each: C = i % 2 == 0 ? h : h2` inside a loop, with `h` and `h2` made before it, was
`'this value' is moved inside a loop but was made outside it` for every type (found by #671, whose
`00interface_cast_unrelated` loop case was rewritten around it). The ternary lowers to an
`arith.select` of the two values (or, when a side has a call, to a `cf` branch passing each as a
block argument), and neither was a read: the merge took the value, and `takerLoops` saw the taker
come round without the value made again.

- **A merge whose value is only read is a borrow of each value it chooses** (`borrowsThroughMerge`,
  OwnershipInferenceMoves). `movesOf` walks what the select's result or the block argument leads to
  (`walkBorrowed`, which now passes through a select as it does through a branch); when nothing
  stores, returns, captures or retains it, and none of it runs after one of the value's releases,
  the merge is a read, and so is each use the walk found: a use of the merged value after a move of
  `h` is a use after that move. A `let` declared from a merge of values the function made (`const h =
  new C()`) borrows it as `let b = a` does: its
  reads are walked, its `ts.RetainSlot` and `ts.ReleaseSlot`s go (`mergeBorrowRetains`, which
  `decideRetains` then skips). The verdict-2 path (`checkValueMoves`'s borrowing `let`s) leaves such
  a merge out of the takers too.
- **Not widened:** a merge whose value is kept is still a taker, decided as before: pushed, stored
  into a field, returned or captured by a closure that escapes inside the loop it is still `moved
  inside a loop`, and a merged value assigned to an outer `let` and read after one side's release is
  still rejected (`takes a second reference`). A `let each` that is assigned again is not a borrower.

`own_ternary_outer_loop` binds `c ? h : h2` in a loop as a `const` and a `let`, of a class and of an
interface, as a branch (`i % 4 == 0 ? h : (odd(i) ? h2 : h)`), and with one side made in the loop,
churns, and reads `h` and `h2` after the loop; it prints `done.` under own, rc, gc and none, JIT and
AOT, with no counting under own. `00interface_cast_unrelated` gets its loop case back. Negatives
`own_err_ternary_loop_stored` (pushed), `own_err_ternary_loop_returned` (a `let` returned) and
`own_err_ternary_loop_captured` (captured by an escaping closure). Tooth: built without the change,
each of the run test's six functions fails (`moved inside a loop`, and for the `let`s `takes a
second reference`), and so does the interface test's loop.

Not here:
- `let each = c ? h : h2` where `h` and `h2` are themselves `let`s: the merge is of their reads, which
  are not values `movesOf` checks, so `each`'s `ts.RetainSlot` is still `merged from several
  branches` (and `moved inside a loop`). A `const each` of the same reads compiled before this section
  and still does.

## 36. Records with no box: tuples that hold borrows (`@value` records), 2026-10-24

With `@value` (docs/superpowers/specs/2026-10-24-value-types-design.md) the ray tracer's
`Intersection` is a tuple, not a box: `{ thing: this, ray: ray, dist: dist }` returned as
`Intersection | null` through the interface `Thing`, and `closestInter` given the nearer one inside
`intersections`' loop. Section 32's records are boxes, found where `ts.New` is filled from the
literal's local; a tuple has none, so the store of `this` into the literal's field was a keep, the
facts were lost (`'S.intersect' could return a borrow or keep a parameter, but it is reached through
an interface`), and every use downstream was an error. The same held for a type literal (`type Hit =
{ thing: Thing; d: number }`). The marked benchmark had 13 own errors.

**The rule.** A tuple that holds borrows and owns nothing is a borrowing record with no box: Rust's
`struct Hit<'a> { thing: &'a Th, .. }` passed by value. What it borrows bounds every use of it, as in
section 32; there is nothing to give back, so no copy of it is counted and no release of it, or of a
local that holds only such records, frees anything.

- **Which literals** (OwnershipSignatureRecords, `classifyUnboxedRecord`, in every round of the facts
  fixpoint beside `classifyRecordBoxes`). An object literal's local (`isConstantRecordLocal`) whose
  read is never boxed (`recordBoxOfLocal`), read whole (a literal whose fields are read one by one, to
  be re-made as another tuple, is not one), where every field that holds a block borrows it - `this` or
  a parameter, a read out of a place, a call's result that borrows, an owning local's value, a
  captured variable's, an immortal value - and at least one does. A fresh value counts only in a
  *temporary*, a literal whose read is only given to calls that do not keep it and read
  (`len({ start: new Vec(1, 2), dir: new Vec(3, 4) })`): the new vectors keep their own releases, and
  the record borrows them too. Anywhere else a fresh field is one the tuple would own (an owning tuple,
  #679's rules). Never a handle (counted, section 23), a string (copied, section 22), or a field of a
  generator's state (whether that read moves the value out is the generator's rule, decided after the
  facts: `findYieldedFields`). The local is marked `__own_record_unboxed_fields` (and
  `__own_record_temporary`); each read of it is a record in `recordBoxes`.
- **The facts.** A function whose returned values are all such records (through views, merges and
  owning locals: `recordParamsOf`) has record facts as in section 32, plus `__own_result_record_unboxed`
  on the function and its calls (`Facts.unboxed`). Boxes and unboxed records in one value, or on two
  return paths, give no facts: their callers release them differently. Families agree on it as on the
  rest. A function with `@lends` and `@record` that another module can call and whose body returns an
  unboxed record is an error: `@record` tells the importer it owns a box, and it cannot be told this.
- **Where the record is made and where a call returns one** (OwnershipInferenceRecords). The unboxed
  record joins `records` (no box; its reads are the starts of `walkRecord`), and a call that returns
  one goes through `checkRecordResults` as before. `walkRecord` follows what a tuple adds: a field
  taken out of the record (`ts.ExtractProperty`) that may point into a block is the record still - kept,
  it is the record kept - and a field read in place out of a local that holds the record
  (`ts.PropertyRef` of the local, or of the local seen as its storage tuple) is a read of the record,
  whose loaded value is the record's, and a write of that field (`r.s = x`) is a keep. A temporary's
  fresh field is bounded by that value's own release.
- **No owner to find.** A value whose sources are only unboxed records and values that hold no block
  (`unboxedRecordFlow`: through views, merges and owning locals) has no owner check
  (`checkOwnershipOps` skips it: copies are free), its retains go (`decideRetains`), and after the
  moves are placed `dropUnboxedRecordCounts` erases every retain and release of it, and every
  `ts.RetainSlot` and `ts.ReleaseSlot` of a local that holds only such values - the edge releases of
  section 30 included. A local or a merge that holds one beside a value that owns a block is `a value
  that borrows what it holds and owns nothing (a tuple that holds a borrow) meets a value that owns a
  block here; -mm=own cannot release them alike`. A handle's counts are left alone. A `@value` local
  is reached through a cast of its reference to its storage tuple, and every one of these walks
  looks through it (`slotVariableOf`, `valuesStoredInto`).

Fixes on the way, each needed by a test of this section:

- `isView`: a `ts.Cast` between a `@value` type and its storage tuple is the same value (it lowers to
  nothing). The type walkers of the signature pass (`valueInterfaces`, `anyInterfaces`,
  `functionValues`, `castMakesInterface`, `holdsFunction`, `storageTupleOf`) read a `@value` type as
  its tuple (`asStorageTuple`): `castMakesInterface(value, tuple)` answered yes and opened `Thing`, which
  is why the facts above were "reached through an interface".
- `recordRetainOf` follows the record's read through its views (a `@value` literal is read, then cast
  to its type, then retained and returned).
- `fieldTakersOf` and `fieldOnlyRead`: a field taken out of a record that holds no block takes nothing
  (`print(h.dist)` converted the number to a string, which counted as a taker of the record).
- A closure that does not escape and captures, by reference, storage of this frame or one a parameter
  points to that holds no block (`this` of a `@value` constructor whose fields are numbers) needs no
  owner: `'this value' is captured by a closure, but -mm=own cannot tell who owns its value here` before.

Tests: `own_value_record` (the brief's `S` with a ternary and `P` with an `if`, through `Thing`, 1000
rays with churn), `own_value_loop` (the `intersections` loop: `closest` starts undefined, takes the
nearer `t.hit(ray)`, is returned and shaded through `thing`), `own_value_temporaries` (call-argument
temporaries of a `@value` interface and of a type literal, a `const` record borrowing a `let` twice
read in place, a `@value` constructor's closure), under own (no counting), rc, gc and none, AOT and
JIT; `value_iface_cycle` and `value_ternary_null` under own too. Negatives:
`own_err_value_record_place_dropped` (the record borrows `p.child`, which a call overwrites before
`o.s` is read in place), `_stored` (pushed into an array), `_let_reassigned`, `_mixed_local`,
`_field_written`, `_param_field_kept` (a callee pushes a record parameter's borrowed field), and the
exits of a record that borrows a `let`: `_returned`, `_captured` (by a closure that escapes),
`_field_stored` (a class field), `_global`. `own_err_value_refs_copy` pins `value_refs`' copy
shape (a copy out of an array read after the element is overwritten); own rejects it at the read
even without the overwrite, an over-rejection that predates this section. `value_interface` runs
under own; its two gaps (below) are in `value_interface_own_gaps`, which every other model runs. `own_err_record_literal_let_field` now reads `borrows ... and cannot be stored` at
the push (the literal borrows the `let`); `own_err_shared_place_literal` pins the handle exclusion (with
it missing, it compiled and crashed with 0xC0000005).

Teeth (temporary switches, removed): with `dropUnboxedRecordCounts` off, the AOT builds of
`own_value_record` exit 0xC0000005 and `own_value_loop` and `own_value_temporaries` 0xC0000374; with the
in-place field read of `walkRecord` off, `own_err_value_record_place_dropped` compiles and prints 11
(gc 5; before the storage cast was followed it printed 77777).

Corpus (`test/tester/tests/*.ts` under `-mm=own --no-default-lib`, `--emit=obj`, Windows): 542
before, 543 after: `00spread` compiles and prints what gc prints, AOT and JIT.

The marked benchmark (`rtbench.ts` with the value-types edits and `rtbench_own.ts`'s `Shared<Surface>`,
`Plane.normal` returning `this.norm`), `--emit=exe --opt --opt_level=3 -mm=own`: 13 errors before, none
after; it prints the checksum 38837369.

Not here:
- A literal nested in another whose fields own new values (`{ ray: { dir: new Vec(..), .. }, thing:
  new T1() }`), and an owning literal pushed into an array or stored into a field: #679's owning
  tuples move only into a local or the result (`value_interface_own_gaps`, off under own).
- An unboxed record converted to an ordinary interface (`const s: HasStart = mk(new Vec(6, 0), ..)`):
  the box made from the value would own the fields the record borrows. It needs section 32's box
  classification for a box filled from a record value, not from a literal (`value_interface_own_gaps`).
- A closure inside a closure in a `@value` method that captures `this` (`value_refs`' `Box.nested`):
  the inner capture of the outer box's copy is a keep.
- `value_refs` as a whole: it copies values that hold owned Nodes out of arrays, fields, a generic and
  closures and reads the copies after the originals' Nodes are overwritten. Under single ownership
  the copy is a borrow of the original, and those programs read freed blocks; own rejects each one
  (`borrows an element and cannot be stored`, `... but is used here after ...`), correctly.
- An exported function returning an unboxed record (see the facts above).
- `const ts: Th[] = [new A(), new B()]; const h = ts[1].hit(ray); print(h.dist)` outside a loop is
  `borrows an element but is used here after it may be released or overwritten` for a record with a
  box (section 32) as much as for one without; the same call inside a loop over `ts` compiles.

## 37. A `for...of` over an iterator read out of a place borrows it, 2026-10-27

MLIRGen desugars `for (const v of e)` over an iterator to `for (let .b = e, .c = .b.next(); !.c.done;
.c = .b.next())`. Under own a `let` initialized from a parameter is a second owner of it, so
`function forEach(iter: TIter, f) { for (const v of iter) f(v); }` was "'iter' is a parameter, kept
here", and so was every iterator helper of the default library that walks its source (`forEach`,
`every`, `some`, `find`, `toArray`). `.b` is never assigned. Under own, when `e` is a `ts.Load` - a
parameter, a variable, a field, a global - `.b` is now a `const` declared just before the loop, in the
same scope: it names the value, a borrow the inference checks like `const it = iter`. A fresh `e`
(`for (const v of gen())`) keeps the `let`, which owns it. The other models keep the `let` in every
case: its count keeps the iterator alive should the body assign the variable it came from.

A body that assigns the variable is safe under own: a parameter's slot releases nothing when assigned
(the frame never owned it), and a local `let`'s loop compiles to the same IR as before (`.b` gets a
slot, and the inference moves the value into it as it did the `let`). A labelled loop keeps its label:
the `const` statement is emitted before the `ts.For`, not wrapped around it.

`own_for_of_borrowed_iterator` runs a parameter, a labelled loop with `continue outer` and `break
outer`, a parameter the body assigns, a field and a global, churning between steps, under own, rc,
gc and none; own emits no counting. Before the change own rejected it with four errors. DefaultLib
under own: `iterator_every`, `iterator_find`, `iterator_forEach`, `iterator_some` and
`iterator_toArray` compile and pass.

Not here: the lazy helpers `map`, `filter`, `take` (generators whose state borrows the source) and
`drop` (which returns it) still give a result that borrows a temporary, so `const seq =
fibonacci().map(f)` is an error; that needs a generator state that owns a parameter. `reduce` with a
string accumulator starts its `let` from the `initial` parameter.
## 38. A generator that keeps its arguments (`@keeps` on a generator), 2026-10-27

A generator over a parameter that holds a block borrows it (§19): the caller owns the state, which may
not be stored, returned or captured past the argument. The default library's lazy iterator helpers
(`map`, `filter`, `take`, `drop`) are generators over their source, so `const seq =
fibonacci().map(f)` was "borrows its generator's arguments and cannot be stored" - in JavaScript a
helper takes its source over. `@keeps("p")` on a generator now says that: the state owns `p`.

- **The signature pass.** `findBorrowingStates` still takes a `@keeps` maker's state, marked `owns`:
  no bound (`__own_result_bounded`), nothing in `borrowsFound`, and in `ts.own_borrowing_fields` the
  box's position is written as -1, so the release routine frees the box like any escaping closure's -
  its cells and what they hold - while the fields that hold only what is read out of it (a `for...of`
  cursor over `iter`, §37) still borrow, from the box, which the state frees with it.
  `localOwnedParams` counts, for a function that says `@keeps`, each parameter owning a block whose
  cell the maker stores into the state's `.captured` box (`paramsCapturedIntoState`), so the callers
  move the arguments in and `checkDeclared` holds the body to the decorator. A box owns every cell it
  holds: `@keeps` on `map` has to name the callback too ("names 'iter', but its body keeps 'iter',
  'f'"), and a capturing closure given to two helpers is used after its move (a plain function
  holds no block, and is not moved).
- **The inference.** `decideEscaping` lets the cell of a parameter the function keeps move into an
  escaping box, as an owning local's does (it was "a closure that escapes owns what it captures, but
  -mm=own cannot move this variable into it: its value is the caller's").
- **The callers.** An argument a callee keeps moves in only when it is fresh or this function's to
  give. Two shapes the helpers meet were neither: `isFresh` now takes a `ts.CallIndirect` of a
  `ts.SymbolRef` to a function defined here (a generator called at a script's top level, an
  extension call such as `g().take(2)`), as it took a direct call; and `holdsNoBlock` a plain function
  cast to a hybrid one (`f(double)`, `f((x) => x * 2)`), whose `this` is null.

`own_generator_keeps` runs a class argument, chained iterator generators, a lambda and a named
callback, a generator returned past its argument's expression, one given up after a step, and a
module-level one, churning between steps, under own, rc, gc and none; own emits no counting. AOT under
own, 200000 iterations peak at 4.4 MB, as 1000 do (4.3 MB). Negatives: `own_err_generator_keeps_missing`
(the callback left out), `own_err_generator_keeps_used_after` (the argument read after it moved),
`own_err_generator_keeps_bounded` (a generator that borrows its argument given to one that keeps it).

DefaultLib under own: `iterator_map`, `iterator_filter` and `iterator_toArray` pass with `@keeps` on
`map`, `filter` and `take`, and `drop` made a generator.

Not here: a generator that borrows its container given to a keeper (`s.values().take(2)`) - §39. A
`for...of` at a script's top level over a
module-level generator whose `next` calls a callback: the call may assign the global, so the loop's
borrow of it ends. `reduce` with a string accumulator, as before.

## 39. A keeper given a bounded generator is bounded in turn, 2026-10-27

`new Set([1, 2, 3]).values().take(2)` (DefaultLib `iterator_take`, `iterator_drop`): `values()` is a
generator whose state borrows the set (§19, bounded by it), and `take` says `@keeps` (§38), so its
state owns the iterator it is given. Moving a bounded result into a kept argument was "borrows its
generator's arguments and cannot be stored, returned or captured" - the keeper might outlive the set.

It need not: the keeper's state owns the bounded state, and so is bounded by the same arguments. When
`boundedUses` meets the bounded result as a kept argument of a call whose result is a keeper's state
(`ownsKeptState`: -1 in place of the box in `ts.own_borrowing_fields`), it follows that result as it
follows a local holding the bounded one: its uses - a `for...of` over it, a `next()` in the
statement, a local holding it, another keeper given it (`.drop(1).take(2)`) - are the bounded
result's, checked against the ends of its arguments as before, and returning, storing or capturing
it is still refused. The release needs nothing new: the keeper's state frees the bounded state, whose
release gives back nothing it borrows, so the order against the container's release does not matter.

Only a bounded *result* is followed so: what `boundedUses` walks is the call's result and the locals
holding it, each this function's to move. A keeper given a borrow of something else (an element, a
parameter) is the argument move's business, unchanged.

`own_generator_keeps_bounded` walks keepers over a generator that borrows a temporary, two keepers
deep, one step in the statement that made it, and a chain over a named source held in a local, churning
between steps, under own, rc, gc and none; AOT under own, 200000 iterations peak at 4.4 MB, as 1000
do. Negatives: `own_err_generator_keeps_bounded` (now: returned past its source - the §38 shape,
walked in place, is accepted) and `own_err_generator_keeps_bounded_outlives` (used after the source
is overwritten).

DefaultLib: `iterator_take` and `iterator_drop` pass under own. Their `limit` is now a number, as in
JavaScript: `take(Infinity)` converted `Infinity` to an `index`, which is no particular count: the same
line took all three under rc and none under own. `reduce`'s string accumulator, listed in
§38 as open, is DefaultLib #45 (its accumulator typed `V`, not the `any` that `?? iter.next().value`
gives).
