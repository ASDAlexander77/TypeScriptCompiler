# `-mm=own`: a single-owner memory model

Date: 2026-09-24. Status: design approved in conversation; written review 2026-09-28 (§10),
amendments folded in. Phase 0 merged as #399 (results §11); phase 1
(moves by reachability) merged as #401 (results §12); phase 2 (borrows for locals) merged as #402
(results §13); phase 3 (containers and unions) merged as #403 (results §14); phase 4 (function
signatures and `any`) on branch `own-phase-4` (results §15). Plans in `docs/superpowers/plans/`.

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
built under `own` exports its facts beside the existing `__tsmm_own_*` marker.

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
on the callee's fact). Unknown callees default as in §2.4. For a `-shared` build the facts are
serialised into an exported symbol beside the marker; the importer reads them during the symbol
enumeration it already does for `__tsmm_` and applies them to the declared `FuncOp`s.

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
   `any`; export and import deferred.
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

### 15.7 Known limits (the input to phase 5 and later)

- Export and import of the facts (§3.2). An exported function, or a method of an exported class,
  gets no owned or borrowed facts, and a virtual call on one gets no `__own_no_drops`. That is sound
  and restrictive for `-shared` modules.
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
- rc: a getter's result used as a temporary is never released (§15.3).
- Every model: `<B>anyValue` segfaults when `B` implements an interface.
  `mlirGenInstanceOfOpaque` calls vtable slot 0 as `..instanceOf`, but such a class keeps the
  interface's vtable there (`B..vtbl = {I, .instanceOf, ...}`). Fixing it changes the vtable
  layout, so it is left for its own PR.
- From §14.6: a read reached through an interface, borrow chains through assigned borrowers,
  moves after the last use of every borrower, and a temporary taken by a `let` and consumed
  elsewhere.
