# `isObjectShapedTuple` structural ambiguity: design

Status: **proposed, not implemented** — written for later review. Triggered by
a code-review comment on `MLIRPrinter.h:232` during the cross-module
class-extends work (`fix/cross-module-class-extends-crash` branch). A narrow
symptom of the problem (numeric field ids) was already patched (see §4); this
document is about the deeper, still-open ambiguity the reviewer flagged, not
about that patch.

## 1. Problem statement

`MLIRPrinter::printTupleOrObjectType` (`include/TypeScript/MLIRLogic/MLIRPrinter.h:236-246`)
decides, purely at print time, whether to render a `TupleType`/`ConstTupleType`
value as a TS tuple (`[...]`) or a TS object type (`{...}`):

```cpp
template <typename TPL>
bool isObjectShapedTuple(TPL t)
{
    auto fields = t.getFields();
    return !fields.empty() && llvm::all_of(fields, [](auto &field) {
        return field.id && (isa<mlir::StringAttr>(field.id) || isa<mlir::FlatSymbolRefAttr>(field.id));
    });
}
```

The heuristic: "every field has a name-like id" ⇒ object. This is not just an
incomplete check (the numeric-id gap fixed in §4) — it is **unsound in
principle**, because both a genuine TS tuple type and a genuine TS object type
can legally produce a field list where every `id` is a `StringAttr`. The
printer has no signal left, by the time it sees the MLIR type, that would let
it tell those two cases apart.

## 2. Root cause: three source forms collapse into one representation

`mlir_ts::FieldInfo` (`include/TypeScript/TypeScriptOps.h:36-54`) is `{Attribute
id; Type type; bool isConditional; AccessLevel accessLevel;}`, and exactly the
same `TupleType`/`ConstTupleType` (a `SmallVector<FieldInfo>`) is built for
three distinct TS surface forms, verified at these call sites:

1. **Positional tuple, named elements** — TS `[name: string, age: number]`.
   `MLIRGenImpl::getTupleFieldInfo(TupleTypeNode, ...)`
   (`lib/TypeScript/MLIRGenTypes.cpp:2512-2572`) pushes
   `{TupleFieldName(namedTupleMember->name, ...), type, ...}` for a
   `NamedTupleMember` (line 2531) — a real `StringAttr` id, exactly like an
   object field's id. Consumed by `getTupleType`/`getConstTupleType`
   (2668-2700) into a plain `TupleType`/`ConstTupleType`. This type is
   accessed positionally (`00tuple_named.ts` assigns via `["user", 10.0]`) but
   its fields also support name access (`a.name`) — the language deliberately
   blurs value-level access, which is fine; the problem is purely that the
   *declared type text* must still say `[...]` to round-trip as a
   length-checked, order-checked tuple rather than an unordered object type.
2. **Object-type-literal annotation** — TS `{ name: string; age: number }` as
   a *type*. `MLIRGenImpl::getTupleFieldInfo(TypeLiteralNode, ...)`
   (`lib/TypeScript/MLIRGenTypes.cpp:2574-2666`) pushes
   `{TupleFieldName(propertySignature->name, ...), type, ...}` (line 2594) —
   the identical `StringAttr`-via-`TupleFieldName` shape as case 1. Consumed
   by `getTupleType(TypeLiteralNode, ...)` into the same `TupleType`.
3. **Object literal value type** — TS `{ name: "a", age: 1 }`.
   `addObjectFieldInfo` builds `oli.fieldInfos` the same way
   (`TupleFieldName`-derived ids), and the literal's value type is
   `getConstTupleType(oli.fieldInfos)`
   (`lib/TypeScript/MLIRGenExpressions.cpp:1383`) — again a `ConstTupleType`
   with all-`StringAttr` ids.

Nothing survives in the `FieldInfo`/type itself that records *which of these
three AST forms* produced it — the distinction is discarded during lowering.
So `isObjectShapedTuple`, looking only at field-id kind, cannot in principle
distinguish case 1 from cases 2/3: they are structurally identical MLIR
values. Today it resolves the ambiguity by guessing "object" for anything with
all-named fields, which is simply wrong for case 1.

## 3. Concrete failure case (not yet covered by any test)

```ts
type Pair = [name: string, age: number];
```

lowers to a `TupleType` whose two fields both carry `StringAttr` ids ("name",
"age"). `isObjectShapedTuple` returns `true` for it (both before and after the
§4 fix — that fix only excludes *numeric* ids, and these are strings), so
`printTupleOrObjectType` prints it as:

```
{name: string, age: number}
```

instead of the correct

```
[name: string, age: number]
```

This is not just cosmetic: a tuple type is length-checked and
positionally-typed; an object type is structurally matched (any object with
at least those properties satisfies it, order-independent, no `.length`
tuple semantics). Reimporting the printed text (the whole reason this printer
exists — cross-module `@dllimport` declaration round-tripping, per the
comment at `MLIRPrinter.h:221-227`) would silently widen the type. No test
currently exercises a named-tuple-typed value through this printer path
(`00tuple_named.ts` only checks runtime field access, never a printed/emitted
type), which is why this has not surfaced as a failure yet.

## 4. What the just-applied fix does and does not cover

A narrower bug was fixed alongside this review: `MLIRGenImpl::getTupleFieldInfo(TupleTypeNode, ...)`
also produces `IntegerAttr` ids for array-mode literal-type tuple elements
(e.g. `type T = [1, 2, 3]`, `lib/TypeScript/MLIRGenTypes.cpp:2551`), and the
original `(bool)field.id` check treated those as "named" too, printing
`{0: 1, 1: 2, 2: 3}` instead of `[1, 2, 3]`. The fix (uncommitted,
`include/TypeScript/MLIRLogic/MLIRPrinter.h:229-239`) narrows the check to
`isa<StringAttr>(id) || isa<FlatSymbolRefAttr>(id)`, which correctly excludes
numeric ids. It does **not** and cannot resolve §3, since named-tuple-element
ids are genuinely `StringAttr` — the same fix that closed the numeric case is
structurally unable to close the string case, because the ambiguity is not
about attribute *kind* at all; it is about missing provenance.

## 5. Options

**Option A — carry an explicit discriminator on the type.** Add a bit (e.g. a
`bool isTuple`/`isObjectShape` parameter, or reuse an existing spare slot) on
`TupleType`/`ConstTupleType`'s storage, set once at construction by
`getTupleType(TupleTypeNode/TypeLiteralNode, ...)` and by the object-literal
path, and read directly by the printer instead of re-deriving it from field
ids. Smallest change: the printer becomes `t.getIsObjectShape()` instead of
`isObjectShapedTuple(t)`. Cost: every `TupleType::get`/`ConstTupleType::get`
call site (there are many — captures, array/string/optional synthetic tuples
in `MLIRTypeHelper.h`, etc., §2 of `object-literal-boxing-design.md` lists
several) must pick a value; a wrong default silently reintroduces this same
bug for a new call site instead of a compile error, so this only helps if the
default is chosen deliberately (positional/tuple, since that is the
type's primary purpose) and every "this is actually object-shaped" producer
is updated explicitly.

**Option B — separate MLIR types for "positional tuple" vs "object shape".**
Stop overloading `TupleType`/`ConstTupleType` for object-literal/type-literal
values; route those through `ObjectStorageType` (which already exists and is
used for the object-literal's `this`-type, `MLIRGenExpressions.cpp:1300`) or a
new named/anonymous record type, uniformly. This is the architecturally clean
fix — it removes the ambiguity at the source instead of threading a flag
through it — but it is a large, invasive change: every consumer that
currently does `TypeSwitch<...>().Case<TupleType>/.Case<ConstTupleType>(...)`
for "structural object value" (casts, property access, spread, `getFields`,
interface/class field-clone logic — the same call sites enumerated across
`object-literal-boxing-design.md` §2-3 and
`interface-vtable-simplification-design.md`) would need an additional case or
a look-through, mirroring the `ObjectType` rollout's fallout (5 extra fixes
found by *running* the flip, not by inspection, per that doc's "PR B
implementation notes"). Given that history, Option B should not be attempted
without the same investigate-first + staged-PR discipline.

**Option C — printer-only provenance, no type-system change.** Thread a
`bool isObjectShape` through the two `getTupleFieldInfo`/`getTupleType`
call-site families as a constructor argument used *only* to select a print
mode (e.g. stash it as an extra always-null-valued sentinel `FieldInfo`, or a
side-table keyed by the type). Cheaper than Option A in that it touches only
the type-building call sites already named in §2 (three producers, not every
`TupleType::get` in the codebase), but it is a narrower patch than Option A
that still leaves the same "new producer forgets to set it" failure mode, and
is arguably more surprising than a first-class type parameter.

## 6. Recommendation (for discussion, not decided)

Option A, scoped narrowly: add the discriminator only to
`TupleType`/`ConstTupleType` (not a new type), default it to "positional
tuple" everywhere, and audit the handful of producers in §2 plus
`MLIRTypeHelper.h`'s synthetic tuples (array/string/optional destructuring,
`:2230-2247`) to set "object shape" explicitly where warranted. This fixes the
printer without the blast radius of Option B, at the cost of an audit pass
over existing `TupleType::get`/`ConstTupleType::get` call sites — needed once,
not ongoing, since new call sites get a compile-visible constructor argument
rather than a silently-wrong default. Option B is the more correct long-term
shape (same direction as the `ObjectType` boxing precedent) but should be a
separate, later project given its proven-large fallout surface.

## 7. Non-goals / out of scope

- Not fixing §3's concrete failure case in this document — it is a design
  writeup, not a patch. No test for it exists yet either; add
  `00tuple_named_print.ts` (or extend `00tuple_named.ts`) once a fix direction
  is chosen, asserting the printed/declaration text for a named-tuple-typed
  cross-module export.
- Not re-litigating `ObjectType` boxing (`object-literal-boxing-design.md`) —
  that already gives *non-tuple* object literals (ones with methods) a
  distinct type. This document is about the remaining *value* (non-boxed)
  object literals and type-literal annotations, which still share
  `TupleType`/`ConstTupleType` with positional tuples.
- Not auditing every `printType`/`TypeSwitch` consumer for correctness beyond
  the printer — Option B's fallout audit (§5) is real work for whoever picks
  that option up, not pre-solved here.
