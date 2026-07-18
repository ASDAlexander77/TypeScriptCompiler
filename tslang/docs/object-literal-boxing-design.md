# Object literals with methods as reference types (`ObjectType`): design

Status: **PR A merged (#248, main@9a7bd71e); PR B merged (#249,
main@9f0d1a6a); PR C implemented (narrower than planned — see §5), full
suite green (353/353 JIT + 357/357 AOT)** — generalizes the generator-wrapper
boxing of PR #245 (`docs/generator-object-wrapper-design.md`) from "literals
carrying the `BoxAsObject` flag" to **every object literal that has at least
one method or accessor**. Pure-data literals (`{ x: 1, y: 2 }`) deliberately
stay value tuples. Follows the same investigate-first format; all anchors
below verified by code inspection on main@172e013b.

## PR B implementation notes (the flip + fallout)

The flip itself (§3 gap 4) was a 2-line change
(`MLIRGenExpressions.cpp:1343-1345`), but it exercises `ObjectType` through
every code path that previously only ever saw tuples for a literal with
methods — each missing `ObjectType` case in those paths surfaced as a crash,
a wrong-answer, or a `never`-typed variable. Five fixes were needed beyond
the flip itself, all following the same shape as PR A's gaps (an existing
tuple-shaped dispatch missing an `ObjectType` case that looks through to
`getStorageType()`):

1. **Boxing seed value**: routing the `boxAsObject` branch's initial
   const-tuple-to-mutable-tuple step through `mlirGenCreateTuple` (which
   allocates + stores directly) instead of the generic `CAST_A` pipeline
   (which re-reads every field, including method fields, individually) —
   the latter triggered a spurious-but-harmless "losing this reference"
   warning on every boxed literal, since it read a `this`-bound method value
   off the source const-tuple only to immediately discard it.
   (`MLIRGenExpressions.cpp:1352-1360`.)
2. **Accessor get/set on a pointer operand**: `TupleGetSetAccessor`
   (`MLIRCodeLogic.h`, shared by `Tuple()` for value tuples and `RefLogic()`
   for pointers/objects) unconditionally used `ExtractPropertyOp`, whose
   operand is restricted to value-typed `AnyStructLike` — not
   `ObjectType`/`RefType`. Needed a `PropertyRefOp`+`LoadOp` branch for
   ref-like operands, mirroring `RefLogic`'s own direct-field-access code
   right below it. Root-caused via MLIR dump comparison, not guesswork: the
   verifier error named the exact operand-type constraint.
3. **`castObjectToInterface` missing the field-coercion fallback**:
   `castTupleToInterface` has a "clone with interface's field types then box"
   fallback for e.g. an inferred `si32` field vs. an interface declaring
   `number` — necessary because that fallback only fires from
   `castTupleToInterface`, which a literal already boxed to `ObjectType`
   never reaches (PR A's dispatcher routes `ObjectType` straight to
   `castObjectToInterface`). Duplicated the same clone-and-coerce logic
   there, operating on the object's storage type. (`MLIRGenCast.cpp`.)
4. **Element access** (`obj[Symbol.x]`, `obj["field"]`) had no `ObjectType`
   case in `mlirGenElementAccess`'s dispatcher at all (`llvm_unreachable`
   fallback) — added one that delegates to `mlirGenPropertyAccessExpression`
   for a constant string/symbol key, same as the `ClassType`/`InterfaceType`
   cases already there. (`MLIRGenAccessCall.cpp`.)
5. **Generic intersection-type return values** (`D & M` where `M` infers to
   a method-bearing literal's `ObjectType`): the tuple-merge loop inside
   `getIntersectionType(IntersectionTypeNode, ...)` had no `ObjectType` case
   and silently returned `NeverType` for one, breaking any generic function
   returning an intersection that includes a boxed argument type. Also
   generalized `tryInferTuple` (generic parameter inference) the same way,
   though that one turned out not to be on the failing path for the actual
   regression test — kept anyway since it's the same latent gap.
   (`MLIRGenTypes.cpp`, `MLIRGenGenerics.cpp`.)

None of these were guessed — each was root-caused via a crash/error message
or an MLIR/LLVM IR dump diff against the pre-flip baseline (`git stash`).
One red herring debugged and ruled out: an accessor (`get`/`set`) whose
backing field infers `si32` but whose declared parameter type is `number`
corrupts the write — reproduces identically on unmodified pre-PR-B `main`,
so it's a pre-existing, unrelated bug, not something this change caused.
Documented as a known limitation in `00object_ref_semantics.ts` (worked
around there by using a float literal) rather than fixed, since it's out of
scope for the boxing flip.

## 1. Goal and rationale

A method can only mutate its object through `this`, and the methods of an
object literal *already* take `ObjectType` as `this`
(`oli.objThis`, `MLIRGenExpressions.cpp:1301`) while the produced value is a
tuple — the value/`this` representation mismatch behind the whole
`boundRefMaterializedCache` / `needsIdentityStorage` bug family (PRs
#244–#246). Predicting *which* methods actually mutate is interprocedural and
effectively undecidable; its sound conservative approximation degenerates to
"has any bound-method field", which is exactly `hasBoundMethodField`
(`MLIRTypeHelper.h:1622`). So: box on the structural condition, not on a
mutation analysis.

Out of scope (deliberately): method-less literals keep value semantics
(`let b = a; b.x = 2` does not affect `a` — documented divergence from JS).
If full reference semantics is ever wanted, the right mechanism is
reference-by-default + escape-analysis *unboxing*, a separate project.

## 2. Verified current state (file:line)

- **Boxing path already exists and is proven**: `mlirGen(ObjectLiteralExpression)`
  boxes when `InternalFlags::BoxAsObject` is set — `NewOp(ValueRefType<tuple>)`
  + `StoreOp` + `CastOp` → anonymous `ObjectType::get(tupleType)`
  (`MLIRGenExpressions.cpp:1336-1371`). Only setter today:
  `buildGeneratorWrapperDeclaration` (`MLIRGenFunctions.cpp:622`).
- **Method presence is already recorded**: `addObjectFuncFieldInfo` fills
  `oli.methodInfos` / `oli.methodInfosWithCaptures`
  (`MLIRGenImpl.h:7406-7429`); accessors route through the same
  `processObjectFunctionLikeProto` path (`MLIRGenImpl.h:7491`, dispatched at
  `:7628`). So the box condition needs no new analysis.
- **Property access / calls / for...of on `ObjectType` are correct** — proven
  end-to-end by PR #245 (350/350 JIT, 354/354 compile with the generator
  wrapper boxed; `evaluateProperty` flows through the generic
  `ObjectType`-aware machinery).
- **Globals need no new lowering**: `GlobalOpLowering` walks the initializer
  region and routes anything containing `NewOp` through the
  global-constructor path (`LowerToLLVM.cpp:3703-3716`). A boxed global
  literal is therefore handled by existing code; PR #246's
  bound-method-tuple-constant special case (`LowerToLLVM.cpp:3684-3731`)
  becomes redundant for these literals (cleanup, stage 3).
- **`ObjectType` → interface cast exists**: `castObjectToInterface`
  (`MLIRGenCast.cpp:822-825`, `:1584+`), so literals assigned to
  interface-typed targets keep working — and stop being *double*-represented
  (today the tuple is boxed a second time by `castTupleToInterface`,
  `MLIRGenCast.cpp:1574-1579`, so the interface view never aliases the
  original value; after this change the same heap object backs both).
- **`typeof` already says `"object"`** for `ObjectType`
  (`MLIRGenCast.cpp:1405`).

## 3. Gaps to fill (the actual work)

1. **`MLIRTypeHelper::getFields` has no `ObjectType` case**
   (`MLIRTypeHelper.h:2163-2229+`: const-tuple, tuple, interface, class,
   class-storage, array, string — no object). Add a look-through:
   `ObjectType` → `getStorageType()` fields (mirror the `ClassType` case).
   `getFieldTypeByIndexType` (`:2141`) and every other `getFields` consumer
   inherit the fix.
2. **Object spread `{...obj}` crashes on `ObjectType`**: the `TypeSwitch` in
   `mlirGenObjectLiteralFields` handles Tuple/ConstTuple/Interface/Class and
   hits `llvm_unreachable` in `.Default` (`MLIRGenImpl.h:7670-7725`). Add an
   `ObjectType` case modeled on the `ClassType` case (per-field
   `mlirGenPropertyAccessExpressionLogic`, **not** `DeconstructTupleOp`,
   which needs a by-value tuple). Reachable *today* with a generator object
   (`const g = gen(); const o = {...g}`), so this can land and be tested
   before the flip.
3. **No `ObjectType` → tuple cast**: `castTupleLikeVariants`
   (`MLIRGenCast.cpp:831-923`) covers ConstTuple/Tuple/Class/Interface
   sources only. Needed for explicitly-annotated declarations
   (`const x: { m: () => void } = { m() {} }`) where the declared type
   resolves to a tuple type. Implement as: `CastOp` object →
   `ValueRefType<storage>` + `LoadOp` (the boxing recipe in reverse), then
   existing `castTupleToTuple`. This is a *copy* — value semantics at that
   annotation boundary, same as today, no regression (see §5 open item).
4. **The flip itself**: in `mlirGen(ObjectLiteralExpression)`
   (`MLIRGenExpressions.cpp:1336-1339`) change the box condition from the
   flag alone to
   `boxAsObject || !oli.methodInfos.empty() || !oli.methodInfosWithCaptures.empty()`.
   Keep the flag: synthetic wrappers set it explicitly and it documents
   intent. Keep boxing to the anonymous structural
   `ObjectType::get(tupleType)` exactly as #245 shipped (not the named
   `oli.objThis`) — structural identity means two same-shape literals get the
   same value type, which conditional expressions / reassignment / arrays of
   literals rely on.

## 4. Verification checklist (cheap checks before/while flipping)

Things that consume the literal's *value type* and must tolerate
`ObjectType`; each is a targeted look-or-test, not necessarily code:

- `==` / `===` between two boxed literals → should lower as pointer compare
  (reference equality — closer to JS than today's tuple compare; behavior
  change to note in tests).
- Destructuring `const { a } = objWithMethods`
  (`processDeclarationObjectBindingPatternSubPath` — property access based,
  expected fine; covered by `00object_deconst.ts`, `39objectdestructuring.ts`).
- Optional/union wrapping (`{m(){}} | undefined`) — pointer in an optional is
  the easy case, but confirm `castToOptionalType` round-trips.
- Debug info for `ObjectType`-typed locals (`generateDebugInfo` path,
  `MLIRGenVariables.cpp:64-74`).
- **Cross-module export/import** of a literal with methods
  (`export_object_literal_with_interface.ts` + the import twins): the
  declaration/type must round-trip as `ObjectType`. This is the least-proven
  area — treat these four tests as first-class targets.
- `main()`-return / top-level: `00object_global.ts`,
  `00global_const_object_method.ts` (the #246 repro — must now take the
  NewOp-triggered constructor path instead of the bound-method special case).

## 5. Staged PR plan (small, squash-merged, per the established workflow)

1. **PR A — infrastructure, no behavior change** (#248, merged): gaps 1–3.
   Test via generator objects, which are already boxed: spread a generator
   object, `getFields`-driven paths on it, annotated-tuple assignment from
   one. 352/352 JIT + 356/356 AOT green.
2. **PR B — the flip** (implemented): gap 4 + regression tests, plus five
   more `ObjectType`-look-through gaps discovered by actually running the
   flip (see "PR B implementation notes" above — accessor get/set, interface
   cast field coercion, element access, generic intersection-type merge,
   generic parameter inference). New test `00object_ref_semantics.ts`:
   alias-mutation via second binding (`const b = a; b.inc(); a.get()`),
   parameter passing, closure capture, array element, object nested in
   object, accessor mutating `this`, global `const` literal with mutating
   method (generalizes the #246 repro), conditional-expression merge of two
   same-shape literals. Full suite green: 353/353 JIT + 357/357 AOT.
3. **PR C — cleanup, narrower than originally planned** (implemented): the
   assumption that `needsIdentityStorage` and `boundRefMaterializedCache`
   become fully dead after the flip was **wrong** — verified empirically
   (not by reasoning) via temporary `report_fatal_error` probes at
   `hasBoundMethodField` and both `isBoundRef` call sites
   (`MLIRCodeLogic.h`'s `Tuple()`/`TupleNoError()`), full suite rerun: 8 of
   10 originally-regressed-then-fixed tests plus 2 more (`00question_question`,
   `00type_aliases_in_generics`) still hit them. Root cause: the **annotated
   tuple-type boundary** noted below in §6 is not just a documented
   value-semantics limitation — the code path that makes it *work correctly*
   (rather than crash) is exactly `isBoundReference`/`boundRefMaterializedCache`
   in `Tuple()`. Whenever a boxed `ObjectType` literal is unboxed back into a
   plain tuple to match a declared/inferred type (an annotated `const`, a
   function parameter, an interface-cast fallback, a generic
   intersection/parameter-inference result — anywhere PR A's gap-3 unboxing
   cast or an equivalent fires), the resulting tuple's method field is still
   *shaped* like a bound method (first input `ObjectType`) even though the
   tuple itself is now a plain value. `Tuple()`'s `isBoundRef` branch is what
   correctly materializes `this` for a call through that field. **Kept as
   permanent, still-necessary machinery, not scheduled for removal.**

   The one piece confirmed genuinely dead: the #246
   `isBoundMethodTupleConstant` check in `GlobalOpLowering`
   (`LowerToLLVM.cpp`, formerly ~3696-3731) — removed. Every boxed literal's
   initializer region unconditionally contains a `NewOp` (the boxing
   recipe), which the walk's existing `isa<mlir_ts::NewOp>(op)` case already
   forces onto the global-constructor path; the bound-method-field check
   never changed the outcome. Verified by running every test file directly
   through the JIT with a diagnostic (non-fatal) probe at that specific call
   site — zero hits across the full ~369-file test corpus — before removing
   it for real and reconfirming 353/353 JIT + 357/357 AOT green.

## 6. Risks / open items

- **FIXED (2026-07-19): global interface from a method-bearing literal**. A
  top-level binding whose declared type is an interface, initialized from an
  object literal with a method (`const c: Counter = { count: 0, inc(){...} }`),
  crashed the JIT (0xC0000005) on first access. Casting the literal to the
  interface (`mlirGenCreateInterfaceVTableForObject`) builds a *per-object*
  vtable patched with the method's function pointer; that vtable was a stack
  `VariableOp` (alloca). A local binding's alloca outlives its uses, but a
  global binding's initializer lowers to a `__cctor` function, so the alloca
  dangled once `__cctor` returned and the interface's vtable pointer
  referenced freed stack memory. Fix: heap-allocate the patched vtable
  (`NewOp` + `StoreOp` + `CastOp`), same footing as the object it describes.
  Regression test `00interface_global_method.ts`. Note the method-*less*
  interface case (`Point`) was always fine — it points `NewInterface`
  directly at the shared global vtable, no per-object patched copy.
- **STILL OPEN: cross-module (`-shared`) method-bearing interface**. Exporting
  such a global and reading it from an importing module still crashes the JIT
  (null function-pointer call in the importer's `__mlir_gctors`); the
  `-shared` AOT path has its own separate issues too. This is a distinct
  problem in the shared-lib DLL-load / `gctors-as-method` symbol-resolution
  subsystem, not the vtable-lifetime bug fixed above (which was
  single-module). Reproduces via the `export/import_object_literal_with_interface.ts`
  pair with an added method-bearing `export var`. Not yet root-caused.
- **Annotated tuple types re-open a small value-semantics window**: after
  gap 3, `const x: { m(): void } = { m() {} }` copies the object into a
  tuple at the annotation boundary, losing aliasing for that binding. The
  eventual fix is in *type resolution* (map method-bearing type literals to
  `ObjectType`), which is a bigger, separate change — out of scope; document
  the limitation in the test. **This is also why `needsIdentityStorage` /
  `boundRefMaterializedCache` can't be removed (§5 PR C)**: the unboxed
  tuple's method field still has a bound-method *shape*, and that machinery
  is what makes calling through it work correctly rather than crash.
- **Function-expression / arrow fields** (`{ f: () => ... }`,
  `{ f: function () {...} }`) are plain fields, not `methodInfos` entries —
  they do not trigger boxing. Matches the "methods only" rule; TS's
  contextual `this` for function-expression properties is not modeled today
  anyway.
- **Structurally identical literals still differ in method-field types**:
  each literal's methods embed its own anonymous named `objThis` in their
  `this` parameter. Pre-existing (tuple types differed the same way);
  `isAnyOrUnknownOrObjectType` loose matching (`MLIRTypeHelper.h:1100-1116`)
  papers over it. Watch for it in the conditional-expression test.
- **Perf**: one GC alloc per literal-with-methods instantiation — what JS
  engines pay; pure-data literals (the benchmark-hot ones) are untouched.
- **Compile-output tests** that print types containing such literals will
  change text; expected, update expectations.
- Class-scoped and namespace-scoped literals share the same
  `mlirGen(ObjectLiteralExpression)` entry, so they change uniformly, but
  the `fixThisReference` interplay noted in the generator design (§5) rates
  a test pass.
