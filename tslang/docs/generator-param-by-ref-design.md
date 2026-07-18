# Generator (bound-method-typed) function parameters: pass-by-reference design

Status: **REJECTED — superseded by `docs/generator-object-wrapper-design.md`.**
The RefType-parameter approach documented below was analyzed and dropped: the
user proposed the better alternative of making the generator wrapper itself a
reference type (`ObjectType`) instead of a value tuple, which fixes parameter
aliasing, const storage, and closure capture at the root with none of the
function-type-equality blast radius described in §3 below. This document is
kept as the record of why the RefType path was not taken.

Branch: `generator-param-by-ref`. See `docs/const-let-storage-design.md` §3b
for the bug this addresses and the [[generator-param-value-semantics-bug]]
memory for the original repro.

## 1. The bug, restated precisely

`const-let-storage-rework` (merged, PR #244) gave `const` bindings whose type
has a bound-method field (currently: generator wrapper objects) real storage
at declaration time — fixing state loss for `.next()` calls made *within the
same function*. It explicitly did not fix the case where such a value is
passed as a function **parameter**: `.next()` calls made inside the callee
still don't advance the caller's binding, because the value is copied at the
call boundary, not aliased.

## 2. Why there is no way to avoid touching the function's ABI type

Confirmed by tracing the pipeline (agent research, not yet re-verified line
by line at implementation time — recheck before coding):

- `mlirGenFunctionParams` (`MLIRGenFunctions.cpp:1026-1070`) wraps every
  incoming block argument in `mlir_ts::ParamOp`, unconditionally built as
  `RefType::get(param->getType())` seeded from `arguments[index]`
  (`MLIRGenFunctions.cpp:1057-1058`).
- `ParamOpLowering` (`LowerToAffineLoops.cpp:183-193`) lowers `ParamOp` to a
  **fresh** `mlir_ts::VariableOp` (a new alloca), always — it does not matter
  what `arguments[index]`'s type is; a new box is minted and the incoming
  value is stored into it as the initializer.
- `arguments` (the entry block's arguments) are not built independently —
  they come from `FunctionOpInterface::addEntryBlock()`, which derives one
  block argument per input type in the `FuncOp`'s **current `FunctionType`**
  (`MLIRGenFunctions.cpp:1208`, `:1317`). Confirmed no separate/parallel
  argument-list construction exists.

Consequence: **the only way a genuine pointer can arrive in the callee's
block argument is for the function's declared `FunctionType` to say the
parameter's type is `RefType<T>`, not `T`.** There is no codegen-only trick
(no "pass the ref through some side channel") that bypasses this — the ABI is
fixed by the type the `FuncOp` was built with. This is different from the
const-storage fix, which only had to change a *local declaration's* storage
decision and never touched a cross-function-boundary type.

## 3. The blast radius this creates

Once a parameter's registered type becomes `RefType<T>` instead of `T`
(for `T` = a `TupleType`/`ConstTupleType` with a bound-method field, per the
existing `MLIRTypeHelper::hasBoundMethodField` predicate from the merged
const-storage fix), every place that compares function *types* structurally
for equality now sees a mismatch against "the same" function's other
appearances (e.g. a `let` variable of function type, a generic instantiation,
a `ReturnType<typeof gen>` query, an assignability check at a call site
against a differently-sourced signature of the same shape). Found so far:

- `MLIRTypeHelper::TestFunctionTypesMatch` (`MLIRTypeHelper.h:905-934`): raw
  `inInputs[i] != resInputs[i]` per-parameter equality (line 922). Callers at
  `MLIRTypeHelper.h:1136, 1368, 1411, 1474, 1723` — assignability/overload/
  generic-instantiation matching all funnel through this one function, so a
  fix localized here (e.g. treat `T` and `RefType<T>` as equal specifically
  when `hasBoundMethodField(T)`) would cover all of these callers uniformly.
- A **second**, separate comparator exists nearby (`MLIRTypeHelper.h` around
  line 1209-1216) using a recursive `canMatch(location, ...)` instead of raw
  `!=` for a different function-type-matching path (different call sites,
  different `startParam`/opaque-`this`-skipping logic). Not yet confirmed
  whether `canMatch` already tolerates a `RefType` wrapper or would also need
  a fix. **This must be checked before implementation** — if `canMatch`
  already unwraps refs generically (plausible, since it's used for broader
  structural compatibility, e.g. object/unknown per its inline comment),
  this path may already be fine; if not, it needs the same treatment as
  `TestFunctionTypesMatch`.
- Not yet audited: generic type-parameter inference/matching (`MLIRGenGenerics.cpp`,
  `tryInferTupleFields`-family in `MLIRTypeHelper.h`), and whatever backs
  `ReturnType<typeof gen>`-style type queries — these may do their own
  independent structural comparison rather than funneling through either
  matcher above. Must be enumerated before implementation, not discovered
  reactively via ctest failures.

This is a materially bigger blast radius than the const-storage fix, which
touched exactly 3 files and ~70 lines. The precedent that this *kind* of
special-casing is tractable exists — `OptionalType` is special-cased at
similar density throughout this file (e.g. `MLIRGenImpl.h:1690`) — but the
touch points here are scattered across a type-matching subsystem that has no
single choke point guaranteeing full coverage the way `registerVariable` was
a single choke point for the storage decision.

## 4. Proposed approach (once implementation starts)

1. **Enumerate every structural function-type comparison site first**,
   exhaustively, before writing the parameter-type change — grep for
   `FunctionType` type-equality comparisons (`!=`, `==`) and every
   `canMatch`/`TestFunctionTypesMatch` caller, not just the ones surfaced by
   one research pass. Build a checklist; don't rely on ctest to find the
   rest, since the original const-storage fix already took 3 rounds of full
   suite runs to surface all issues on a *much smaller* change.
2. Add a single normalization helper, e.g.
   `MLIRTypeHelper::stripIdentityRef(mlir::Type t)` — returns `t`'s element
   type if `t` is `RefType<U>` and `hasBoundMethodField(U)`, else `t`
   unchanged. Use it to normalize both sides immediately before every
   structural comparison found in step 1, rather than special-casing each
   comparator's internals differently.
3. Change parameter type registration (`mlirGenFunctionSignaturePrototype`,
   `MLIRGenImpl.h:1660-1701`, feeding `getFunctionType` via
   `mlirGenFunctionPrototype`, `MLIRGenFunctions.cpp:249-330`) to wrap a
   bound-method-bearing parameter's type in `RefType` when building `argTypes`
   for the `FuncOp`'s `FunctionType` — mirroring how optional params are
   special-cased at `MLIRGenImpl.h:1690-1693`.
4. Change `mlirGenFunctionParams` (`MLIRGenFunctions.cpp:1026-1070`): when
   `param->getType()` is already the caller-visible `RefType` (i.e. this
   parameter took the new path), do NOT wrap it in another `ParamOp`→fresh
   `VariableOp`; bind the variable directly to the incoming block argument
   (which is already the caller's storage pointer) instead of allocating and
   copying.
5. Change call-site operand building
   (`mlirGenAdjustOperandTypes`, `MLIRGenImpl.h:6394-6455`, specifically the
   `value.getType() != argTypeDestFuncType` branch around line 6445): when the
   destination type is `RefType<T>` for a `hasBoundMethodField` `T`, obtain
   the operand's *reference* instead of loading — `MLIRCodeLogic::GetReferenceFromValue`
   (`MLIRCodeLogic.h:122-155`) already unwraps a `LoadOp` back to its
   `.getReference()`, and `resolveIdentifierAsVariable`
   (`MLIRGenVariables.cpp:919`) already emits that `LoadOp`, so the ref is
   recoverable at this point without restructuring identifier resolution.
6. Test incrementally, per the const-storage fix's proven workflow: target
   repro first (`00generator_manual_next2.ts`'s `drainTwo` case, extended to
   continue driving `it` after the call — this is exactly the case that file
   currently deliberately avoids exercising), then the same previously-fragile
   set (`00disposable`/`01disposable`/`02disposable`, `00spread`, `01symbol`,
   any test exercising function-type assignability or generics with
   function-typed values), then full suite. Expect multiple rounds.

## 5. Open questions to resolve before coding starts

- Does `canMatch` (§3, second comparator) already tolerate `RefType`
  wrappers generically? If yes, step 1's checklist shrinks by one entry.
- Are there other constructs beyond generators that already have
  `hasBoundMethodField(T) == true` today, or could soon (e.g. an object
  literal with a bound method, not just the generator wrapper)? If so, this
  fix benefits them for free, but the checklist in step 1 must consider
  those call shapes too, not just `function* gen(){}`.
- Is there a case where a bound-method-bearing value is passed *by value on
  purpose* (e.g. intentionally snapshotting a generator's current state into
  a helper that must not mutate the caller's copy)? TypeScript itself has no
  such distinction (objects are always reference semantics at the language
  level) — but confirm no existing test relies on the current (arguably
  accidental) copy-by-value behavior as if it were a feature.

## 6. Non-goals (unchanged from the merged fix)

- No change to `const`/`let` reassignment semantics.
- No storage changes for classes/arrays/plain objects — still unaffected,
  still already pointer-like.
- No `DominanceInfo` introduction.
