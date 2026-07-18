# Generator wrapper as a reference type (`ObjectType`): design

Status: **implemented and verified** — 350/350 JIT + 354/354 compile on the
first full-suite round, plus new aliasing regression coverage (parameter,
closure capture, plain assignment) added to
`test/tester/tests/00generator_manual_next2.ts`. The chronic "losing this
reference" warning on generator tests is gone, confirming the value/`this`
representation mismatch theory (§3). Implementation matched the plan in §4
exactly; step 4 (for...of) required no changes — iteration discovers `next`
via `evaluateProperty`, which flows through the generic property-access
machinery that already handles `ObjectType`.
Supersedes the rejected `docs/generator-param-by-ref-design.md` (RefType
parameters). Proposed by the user; verified feasible by code inspection.

## 1. The idea

Stop representing the generator wrapper (`{ step, next() {...} }`, built by
`buildGeneratorWrapperDeclaration`, `MLIRGenFunctions.cpp:531-678`) as a
value-typed tuple. Make it a **reference type** — `mlir_ts::ObjectType`, a
pointer to heap storage — the same representation class instances already
have. Anything with identity in JavaScript is a reference type; the wrapper
has mutable identity (`step` advanced by `next()`), so it should be one too.

## 2. Why this beats the RefType-parameter approach

| Concern | RefType params (rejected) | ObjectType wrapper (this) |
| --- | --- | --- |
| Parameter aliasing | fixed only at fn boundaries, via ABI change | fixed — callee copies the *pointer* |
| `const g = gen(); g.next()` | needs PR #244's storage fix | works even for a bare SSA const (pointer has identity) |
| Closure capture | needs by-ref capture flag | pointer copied by value is already correct |
| Function-type equality (`TestFunctionTypesMatch`, `canMatch`, generics, `ReturnType<>`) | must treat `T`/`RefType<T>` as equal at every comparison site — scattered, unbounded checklist | untouched — the wrapper is *one* type everywhere |
| Cost | none | one GC heap alloc per generator instantiation (semantically correct; it's what JS engines do) |

## 3. Verified enabling facts (file:line)

- **Boxing recipe already exists**: `castTupleToInterface`
  (`MLIRGenCast.cpp:1574-1579`) boxes a tuple with exactly
  `NewOp(ValueRefType<tuple>)` + `StoreOp` + `CastOp` → `ObjectType`. `NewOp`
  heap allocation is GC-managed (same as class instances) — solves the
  escapes-`gen()`-frame lifetime question.
- **Property access on `ObjectType` is already correct**: the TypeSwitch case
  (`MLIRGenAccessCall.cpp:302`) → `MLIRPropertyAccessCodeLogic::Object`
  → `RefLogic` (`MLIRCodeLogic.h:1623-1671`) emits `PropertyRefOp` **directly
  on the pointer** — a real field address into shared storage. No fresh
  alloca, no pristine-copy seeding, no `boundRefMaterializedCache`. The
  entire bug family lives only in the tuple access path (`Tuple`/
  `TupleNoError`), which `ObjectType` never enters.
- **Methods already expect an `ObjectType` `this`**: object-literal codegen
  builds `oli.objThis = getObjectType(objectStorageType)`
  (`MLIRGenExpressions.cpp:1298-1301`) as the `this` type for the literal's
  methods (the `USE_BOUND_FUNCTION_FOR_OBJECTS` mechanism;
  `isBoundReference`, `MLIRTypeHelper.h:203-219`, recognizes first-param
  `ObjectType` as bound). Only the produced *value* is currently a tuple
  (`MLIRGenExpressions.cpp:1330-1342`) — the value/this representation
  mismatch is likely also the source of the persistent "losing this
  reference" warnings on every generator test.
- **Marker mechanism exists**: synthetic AST already carries codegen hints in
  `internalFlags` (`VarsInObjectContext` set at `MLIRGenFunctions.cpp:598`,
  `ThisArgAlias` at `:607`; enum at `ts-new-parser/enums.h:544-560`, bits
  free from `1 << 11`).
- **Return-type inference needs no help**: the wrapper function has no
  explicit return-type node; `discoverFunctionReturnTypeAndCapturedVars`
  takes the type of the returned value. Box the literal → the wrapper's
  return type *becomes* `ObjectType` automatically, consistently, everywhere
  (including pass-2 method-prototype registration from PR #243, which runs
  the same discovery).
- **`ObjectType` → interface casts already exist** (`castObjectToInterface`,
  `MLIRGenCast.cpp:1584+`), so iterable-as-interface usage keeps working.

## 4. Implementation plan

1. Add `InternalFlags::BoxAsObject = 1 << 11` (`ts-new-parser/enums.h`).
2. `buildGeneratorWrapperDeclaration` (`MLIRGenFunctions.cpp:618`): set the
   flag on the synthetic `generatorObject` literal — one line.
3. `mlirGen(ObjectLiteralExpression)` (`MLIRGenExpressions.cpp:1270-1343`):
   when the flag is set, after building the (const-)tuple value: cast
   const-tuple → mutable tuple (`convertConstTupleTypeToTupleType` — `step`
   must be mutable at runtime), then box via `NewOp` + `StoreOp` + `CastOp`
   to `oli.objThis` (the *named* `ObjectType<objectStorageType>` the methods'
   `this` already expects — not an anonymous `ObjectType::get(tupleType)`),
   and return the pointer value.
4. Check `for...of` / iterator-protocol lowering: find where `next` is
   discovered on the iterated expression's type (search `ITERATOR_NEXT`
   consumers); if it inspects tuple fields directly, teach it to look through
   `ObjectType::getStorageType()`. Same check for `MLIRTypeHelper::getFields`
   (`MLIRTypeHelper.h:2133-2220`) if anything getFields-based touches the
   wrapper.
5. Build + test rounds (proven workflow): target repros first
   (`00generator_manual_next.ts`, `00generator_manual_next2.ts`,
   `00generator7.ts`, plus **extend** `00generator_manual_next2.ts`'s `main2`
   to finally drive `it` past `drainTwo` — the case it deliberately avoided —
   and a new closure-capture-mutation case), then the fragile set
   (`00disposable`/`01`/`02`, `00spread`, `01symbol`), then full ctest suite.
6. If green: the `[value, done]` result tuple stays a value tuple
   (deliberately — it has no mutable identity); PR #244's
   `needsIdentityStorage` machinery and the `boundRefMaterializedCache`
   become dead for generators — leave both in place, remove in a later
   cleanup PR (same bisectability reasoning as before).

## 5. Risks / open items

- `for...of`/spread lowering shape (step 4) is the main unknown — not yet
  read.
- Generator methods in classes (`class C { *gen(){} }`) and object literals
  (`{ *gen(){} }`) share `buildGeneratorWrapperDeclaration`, so they change
  uniformly — but class-method `this` interplay (`fixThisReference` path,
  `MLIRGenFunctions.cpp:534-546, 604-614, 628-637`) needs a test pass.
- Compile-output tests that print the wrapper's type will change text.
- Async generators / `for await` (`ForAwait` flag) — check whether that path
  builds its own wrapper or shares this one.
