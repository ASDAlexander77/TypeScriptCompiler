# Object literals with methods as reference types (`ObjectType`): design

Status: **proposed** — generalizes the generator-wrapper boxing of PR #245
(`docs/generator-object-wrapper-design.md`) from "literals carrying the
`BoxAsObject` flag" to **every object literal that has at least one method or
accessor**. Pure-data literals (`{ x: 1, y: 2 }`) deliberately stay value
tuples. Follows the same investigate-first format; all anchors below verified
by code inspection on main@172e013b.

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

1. **PR A — infrastructure, no behavior change**: gaps 1–3. Test via
   generator objects, which are already boxed: spread a generator object,
   `getFields`-driven paths on it, annotated-tuple assignment from one.
   Full suite must stay 350/354 green.
2. **PR B — the flip**: gap 4 + regression tests. New test
   `00object_ref_semantics.ts`: alias-mutation via second binding
   (`const b = a; b.inc(); a.get()`), parameter passing, closure capture,
   array element, object nested in object, accessor mutating `this`,
   global `const` literal with mutating method (generalizes the #246 repro),
   conditional-expression merge of two same-shape literals. Then the known
   fragile set (`00disposable*`, `00spread`, `01symbol`, generator suite) and
   full ctest.
3. **PR C — cleanup** (only after B is green on main): remove
   `needsIdentityStorage` (`MLIRGenImpl.h:788-794`,
   `MLIRGenVariables.cpp:102-113`), `boundRefMaterializedCache`
   (`MLIRGenImpl.h:10797+`), the #246 `isBoundMethodTupleConstant` check in
   `GlobalOpLowering` (`LowerToLLVM.cpp:3696-3731`), and stale comments —
   each removal proven dead by the flip, verified by full suite. Keeping
   them through B preserves bisectability.

## 6. Risks / open items

- **Annotated tuple types re-open a small value-semantics window**: after
  gap 3, `const x: { m(): void } = { m() {} }` copies the object into a
  tuple at the annotation boundary, losing aliasing for that binding. The
  eventual fix is in *type resolution* (map method-bearing type literals to
  `ObjectType`), which is a bigger, separate change — out of scope; document
  the limitation in the test.
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
