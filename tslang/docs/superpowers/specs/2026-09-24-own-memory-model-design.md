# `-mm=own`: a single-owner memory model

Date: 2026-09-24. Status: design approved in conversation, awaiting written review.

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
  silently erase.
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
3. **Containers**: stores as moves, reads as borrows; `any`/union boxing.
4. **Signature pass**: owned-by-callee parameters, borrowed-from-argument results, export beside
   the marker and import on the other side.
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
