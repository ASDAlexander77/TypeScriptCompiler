# Interface vtable simplification: design

Status: **PR 1 (§4 + §5) and PR 2 (§3) both implemented, full suite green
(718/718)**. Follow-up to PR #251 (heap-allocated patched vtable) and PR #252
(canonical slot numbering, `fix-interface-vtable-index-mismatch@be2c9620`).
All anchors below verified by code inspection at the time each PR started;
see each section's "Implementation notes" subsection for what changed, or was
found to be wrong, while building it. Goal: remove the last *runtime-patched*
vtable path for the common case, make slot numbering single-sourced, and fix
latent bugs found during review and implementation.

## 1. Current architecture: what a vtable slot means

A vtable global exists per (implementer type, interface) pair —
`interfaceVTableNameForClass` / `interfaceVTableNameForObject`. Layout is
extends-recursive, then the interface's own methods in declaration order, then
its own fields in declaration order (`assignCanonicalVirtualIndexes`,
`MLIRGenStore.h`, added in #252). A slot holds one of three things:

1. **Class method** → constant function-pointer symbol. The class builder
   resolves the implementing method via `classInfo->findMethod(name)` (which
   knows its `funcName`) and emits `SymbolRefOp(funcName)` straight into the
   global's initializer (`MLIRGenClasses.cpp:1590-1597`). Fully constant, zero
   runtime work.
2. **Field** (class or object) → byte offset of the field within the
   implementer, encoded as a pointer via the GEP-on-null trick: cast `NullOp`
   to the implementer type, take `&(null)->field`
   (`MLIRGenClasses.cpp:1562-1586`, `MLIRGenInterfaces.cpp:304-343`).
   Position-independent and constant.
3. **Object-literal method** → the odd one out, detailed in §2.

Access site (`InterfaceSymbolRefOpLowering`, `LowerToLLVM.cpp:4851-4939`):
`VTableOffsetRefOp` **loads the slot value** (`LowerToLLVM.cpp:4746-4770`);
the `BoundFunctionType` branch then uses that value directly as the method's
function pointer (4881-4891), while the field branch computes
`ptrtoint(thisVal) + ptrtoint(slotValue)` (`calcFieldTotalAddrFunc`, 4894-4904).
Optional members are handled at *runtime* by comparing the slot value against
`-1` (4907-4927); the `-1` sentinel is written into the slot by the builder for
members missing on the cast target (`MLIRGenInterfaces.cpp:348-351`).

Consequence worth stating explicitly: because the same interface access site
must serve both class-backed and literal-backed values, **slot semantics must
be uniform per slot across all implementers** — a method slot must always hold
a callable function pointer. This rules out "store the method-field's offset
and dereference at the access site" as a unification strategy (classes don't
store methods in instances), and it is why §3 keeps the class convention and
brings object literals to it, not the other way round.

## 2. Why object-literal methods are special (and expensive)

For a boxed literal (`docs/object-literal-boxing-design.md`), methods live *in
the object* as func-typed fields. For capture-free methods the field's
initializer is already a compile-time symbol — `addObjectFuncFieldInfo` pushes
`FlatSymbolRefAttr(funcName)` into the literal's const-tuple values
(`MLIRGenImpl.h:7411-7429`, name obtained from the lifted `funcOp` in
`processObjectFunctionLikeProto`, 7517). Methods **with captures** instead go
through `methodInfosWithCaptures` → `fieldsToSet` (7419-7426): the field holds
a per-instance closure/trampoline pointer that does not exist at compile time.

The vtable builder for objects, however, only sees the implementer's *type*.
A tuple field of `FunctionType` carries no symbol name (unlike
`classInfo->findMethod`), so the builder cannot emit `SymbolRefOp` — its
method branch is literally `llvm_unreachable("not implemented yet")`
(`MLIRGenInterfaces.cpp:359`) and methods are resolved *as fields*
(`methodsAsFields = true`, `MLIRTypeHelper.h:1350-1439`). The shared global
therefore gets an **offset** in each method slot — which the `BoundFunctionType`
access branch would misinterpret as a function pointer. That is papered over at
every cast by `mlirGenCreateInterfaceVTableForObject`
(`MLIRGenInterfaces.cpp:183-262`): clone the global onto the GC heap
(`NewOp`+`LoadOp`+`StoreOp`, the #251 fix), then for each interface method
`LoadSaveOp` the function-pointer *value* out of the object's field into the
heap vtable's slot (229-252; `LoadSaveOp` = load-src-store-dst,
`LowerToLLVM.cpp:4137-4148`). So the patch is load-bearing: the unpatched
global is never valid for a method-bearing interface.

Cost and bug tally of this path: a heap allocation plus O(methods) code per
cast; PR #251 (patched vtable was a stack alloca dangling out of a global's
`__cctor`); PR #252 (a module that never casts never runs the patch machinery's
index-assignment side effects — see §4); and the latent optional-member bug
(§5).

## 3. Change 1: constant vtables for capture-free literal methods

Key observation: for capture-free methods the patched content is a
**per-type constant**. Every instance of a given literal type carries the same
lifted-method pointers (the lifted function is per-literal-expression, and each
literal expression gets its own location-hashed `ObjectStorageType`). The
per-cast patch recomputes the same values every time. Fix: record the symbol at
literal-creation time and emit it like the class path does.

- Add a side table on `MLIRGenImpl` (parallel to `fullNameClassesMap`):
  `objectStorageName → (fieldId → funcName)`. Populate it in
  `addObjectFuncFieldInfo` — it has all three in hand (`oli.objThis`'s storage
  name, `fieldId`, `funcName`) — for the capture-free branch only.
- In `mlirGenObjectVirtualTableDefinitionForInterface`'s initializer lambda:
  when a vtable member is func-typed and the (storage name, fieldId) lookup
  hits, emit `SymbolRefOp(funcName)` exactly as
  `MLIRGenClasses.cpp:1590-1597` does, instead of the offset placeholder.
- In `mlirGenCreateInterfaceVTableForObject`: if **all** interface methods were
  emitted as symbols, `return globalVTableRefValue` unconditionally — the same
  path method-less interfaces already take (line 258). The clone+patch block
  (215-255) is skipped entirely; the vtable global becomes genuinely constant.

The runtime-patch path **stays as fallback** for the two cases where the
constant is unknowable:

| case | why | behavior |
|---|---|---|
| method with captures | field holds a per-instance closure pointer | keep per-object heap clone + patch (current, and genuinely required — a shared global would be wrong here) |
| cast of an *imported* object-typed value | type reconstructed from `@dllimport` declaration text (`mlirGenImportSharedLib`, `MLIRGenModule.cpp`); no local `funcOp`, side-table lookup misses | keep runtime patch — it loads the pointer out of the object itself, which is position-independent and works cross-module today |

Semantic note to accept consciously: the constant vtable snapshots the method
at **compile time**, the current patch snapshots at **cast time**. They differ
only if a method-typed field is reassigned between object creation and the
cast — aligning with class semantics (where the vtable is always compile-time)
seems right, but it is a decision, not an accident.

Wins: casts become O(1) with no heap allocation (less GC pressure, less code);
the #251 heap machinery becomes dead on the main path (kept for the fallback);
the vtable global can be const-qualified; cross-module behavior stops depending
on which module happened to run a cast.

### Implementation notes (PR 2, as landed)

Two corrections to the plan above, both found by testing rather than
inspection - the "captures need a fallback" premise was actually wrong in the
*helpful* direction (one fewer case to handle), and a second, unrelated
distinction turned out to be load-bearing that the plan didn't mention at all.

**Captures don't need the fallback.** Rereading `addObjectFuncFieldInfo`
(`MLIRGenImpl.h`) shows the func-typed field's value is
`FlatSymbolRefAttr(funcName)` unconditionally - captures-with-methods land in
`methodInfosWithCaptures` *in addition to*, not *instead of*, the same
symbol-valued field. What captures actually add is a separate `.captured`
data field the (single, shared) lifted function reads via `this` at call time
(`mlirGenObjectLiteralCaptures`) - not a different function per instance. So
the side table (`objectLiteralMethodSymbolsMap`, keyed on the object literal's
own `ObjectStorageType`, populated in `addObjectFuncFieldInfo` for every
method, captures or not) needed no captures-awareness at all. The imported-
object-type fallback (side-table miss, since there's no local `funcOp` to
name) is the only case actually exercised in practice today - no test yet
covers it explicitly (see §7's test matrix, still open).

**A vtable-identity subtlety this plan didn't anticipate, but had to solve
to key the side table at all:** vtable globals are named by
`hash_value(objectType)` (`interfaceVTableNameForObject`) - structural, not
nominal. The worry this raises - two differently-implemented, same-shape
literals colliding on one vtable global and a baked-in constant only being
correct for one of them - does not happen, verified by compiling a
counterexample (`{count:0,inc(){...+1}}` vs `{count:0,inc(){...+100}}`):
they get different vtable globals. Why: a method field's `FunctionType`
carries `this` as an explicit first parameter typed `object<object_storage<@literal's-own-symbol>>`,
and that symbol embeds the literal's own source location - so the "same
shape" premise never actually holds once the method's own signature is
counted. This is also why the side table has to be keyed on that *nested*
`ObjectStorageType` (recovered at the vtable-build site by reading
`FunctionType.getInputs().front()` off the object's own field, cast to
`ObjectType`, `.getStorageType()`), not on the boxed `ObjectType` directly,
since the boxed type wraps a converted plain `TupleType` (`mlirGen(ObjectLiteralExpression)`,
`convertConstTupleTypeToTupleType`), a *different* MLIR `Type` value than
`oli.objThis` captured when the field was added, even though one is derived
from the other.

**The bug that actually broke the regression suite (22 failures on the first
pass): `PropertySignature` with a function type is not a `MethodSignature`.**
`interface Something { toString: () => string; }` categorizes `toString` as
an interface **field** (`InterfaceInfo::fields`), not a method
(`InterfaceInfo::methods`) - `getVirtualTable`'s `methodsAsFields=true` only
governs how the *object's own* func-typed field gets resolved into the local
`virtualTable` snapshot; it collapses every entry to `isField=true`
regardless of which list the *interface* actually declared the member in. The
access site, however, dispatches on the interface's real categorization
(`InterfaceMembers` tries `findField` before `findMethod`) - so
`toString`'s access always goes through `InterfaceFieldAccess`
(`thisVal + slotValue`, offset semantics), never `InterfaceMethodAccess`
(raw function-pointer slot, `BoundFunctionType`), no matter how the object
stores it. The first version of this change substituted a constant
`SymbolRefOp` into *any* func-typed slot, including `toString`'s - handing a
raw function pointer to code that adds it to `thisVal` expecting an offset,
producing a garbage call target. Crash confirmed via WinDbg
(`WinDbgX.exe -g -c ".dump /ma ..."` then offline `!analyze -v`; see
[[windbg-available-for-debugging]]): faulting instruction `call qword ptr
[rax+rdx]`, both registers holding `thisVal`/slot-sum garbage. Fix: gate the
symbol substitution on `newInterfacePtr->findMethod(fieldName)` actually
finding the member in the interface's own (extends-inclusive) method list,
not merely on the vtable entry being func-typed
(`mlirGenObjectVirtualTableDefinitionForInterface`). Regression test:
`00interface_function_typed_field.ts` - `00interface_object.ts` (pre-existing)
already covered this pattern and is what surfaced the bug, but a minimal,
purpose-labeled test makes the invariant explicit for future readers.

718/718 tests green (716 from PR 1 + this section's new regression test +1
already counted, plus the incidental fix above).

## 4. Change 2: a single writer for `virtualIndex`

Slot numbering is currently embodied in four places:

1. registration-time assignment via `getNextVTableMemberIndex()` — raw
   interleaved declaration order, i.e. the **wrong** layout
   (`mlirGenInterfaceAddFieldMember`, `MLIRGenInterfaces.cpp:620`, and
   `addInterfaceMethod`); since #252 it is dead weight, immediately
   overwritten;
2. `getVirtualTable()` — re-assigns `virtualIndex` as a *side effect* of
   building a vtable for one particular cast target
   (`MLIRGenStore.h:346,356,375,385,404,414`);
3. `getVTableSize()` — implicit count whose contract is documented only by the
   comment *"as I remember methods are first in interfaces"*
   (`MLIRGenStore.h:538`);
4. `assignCanonicalVirtualIndexes()` — the canonical pass added by #252.

Plan: make (4) the only writer. Registration initializes `virtualIndex = -1`;
`getVirtualTable()` becomes read-only with respect to member infos (it builds
its rows by iterating the same canonical order — factor a shared
`forEachVTableSlot(callback)` used by both `assignCanonicalVirtualIndexes` and
`getVirtualTable` so the two can never diverge again); `getVTableSize` derives
from the same helper and the narrative comment goes away. An assert that the
canonical pass ran (any member with `-1` outside the optional-missing case)
catches ordering mistakes early.

### Implementation notes (PR 1, as landed)

The plan above undersold the blast radius by one layer: `getVirtualTable()`'s
per-cast mutation wasn't only correcting for "no cast happened yet" (§4's
framing) - for `interface D extends A, B { ... }`, it was also the *only*
mechanism computing each BASE interface's slot position **within the derived
interface's combined vtable**. `A`'s fields have one `virtualIndex` when `A`
is used standalone (0, 1, ...) and a *different* one when accessed through
`D` (offset by however many slots `D`'s other extends-parents contribute
first) - a single mutable field on the shared `InterfaceFieldInfo` cannot
hold both, and the old code "solved" it by last-writer-wins mutation, correct
only for whichever root interface was cast most recently (`00interface_object4.ts`,
`00interface_conjunction.ts`'s `t2 extends F1, F2` both regressed on the first
version of this change until this was accounted for).

Fix actually shipped: `findField`/`findMethod` (MLIRGenStore.h) grew an
offset-accumulating overload - `findField(id, int &vtableOffset)` - that walks
the `extends` chain and sums each hop's `recalcOffsets()`-computed
`std::get<0>(extent)` (the base interface's slot-block start within *its*
parent) as it unwinds the recursion. `InterfaceFieldAccess`/
`InterfaceMethodAccess` (MLIRGenImpl.h) gained a `vtableOffset` parameter,
added to the member's own (now genuinely standalone-canonical) `virtualIndex`.
Every call site that resolves a field/method for an actual access
(`InterfaceMembers`'s dispatcher, plus the get/set method lookups inside
`InterfaceAccessorAccess` and `InterfaceIndexAccess`) was updated to capture
and pass the offset. Call sites that only need the field/method for a type
check (`castInterfaceToTuple`, safe-cast narrowing, etc.) keep using the
plain no-offset overload - they never build an `InterfaceSymbolRefOp`.

Second, unrelated bug surfaced by the same removal: `mergeInterfaces()`
(MLIRGenTypes.cpp, used only by intersection-type synthesis - `type t = A & B
& { c: number }`) had a pre-existing mismatched aggregate initializer -
`{id, type, isConditional, getNextVTableMemberIndex()}` against
`InterfaceFieldInfo{id, type, isConditional, interfacePosIndex, virtualIndex}`,
silently landing the computed index in `interfacePosIndex` (read only for
methods, never for fields, so inert) and leaving `virtualIndex`
zero-initialized. Masked for years by the same getVirtualTable() mutation.
Fixed as part of this PR; the intersection-type synthesis path
(`getIntersectionType`, MLIRGenTypes.cpp) also needed its own
`assignCanonicalVirtualIndexes()` call - it builds `InterfaceInfo` directly
rather than through `mlirGen(InterfaceDeclaration)`'s AST walk, so it never
picked up §4's fix at its original call site.

Net: two pre-existing bugs (both papered over by the mutation this PR
removes) had to be fixed to keep the regression suite green, beyond what §4's
original text anticipated. Regression coverage:
`00interface_optional_cast_order.ts` (§5), plus the pre-existing
`00interface_object4.ts` (single-level `extends`) and
`00interface_conjunction.ts` (intersection types, both the `interface t2
extends F1, F2` and `type t = F1 & F2 & {...}` forms) now exercise the
extends-offset path that had none before.

## 5. Change 3: fix the latent optional-member cast-order miscompile

`getVirtualTable()` writes `virtualIndex = -1` into the **shared**
`InterfaceInfo` whenever the particular cast target lacks an optional member
(`MLIRGenStore.h:346,375,404`). `InterfaceFieldAccess` then branches on that at
*compile time* and emits `OptionalUndefOp` (`MLIRGenImpl.h:5507-5520`).
Sequence that miscompiles today (untested, by inspection):

```ts
interface I { a: number; m?: number; }
let x: I = { a: 1 };            // cast target lacks m -> m.virtualIndex = -1
let y: I = { a: 2, m: 5 };
print(y.m);                      // compiled AFTER the x cast: sees -1,
                                 // emits OptionalUndef -> undefined, silently
```

Same disease family as #252 (per-cast mutation of interface-wide state), just
silent instead of crashing. The robust mechanism already exists and is fully
sufficient: the *slot value* is `-1` for a missing member
(`MLIRGenInterfaces.cpp:348-351`) and the lowering checks it at runtime
(`LowerToLLVM.cpp:4907-4927`). Fix falls out of §4: stop writing `-1` to
`InterfaceInfo`; `InterfaceFieldAccess` keys the optional-typed load off
`isConditional` alone and always emits the `InterfaceSymbolRefOp`. Write the
regression test *first* (e.g. `00interface_optional_cast_order.ts`) to confirm
the analysis before changing behavior.

## 6. Non-goals / rejected

- **Offsets for method slots** (unify with fields, dereference at access):
  rejected — see §1's uniformity constraint; class-backed values share the
  access site.
- **Patch-once-into-a-shared-global** (memoized runtime patch): superseded by
  §3, which gets a stronger result (true constant) for the same cases; the
  captures case can't use a shared global anyway (per-instance pointers).
- **Deduplicating diamond-extends slots**: `Point3d extends Point, Point`
  currently yields a `{x,y,x,y,z}` vtable (observed in `--emit=mlir`).
  Harmless — layout is deterministic on both sides and `findField` resolves to
  the first copy — but wasteful. Deferred: it changes the vtable ABI, so it
  must ship with cross-module tests, and the payoff is small.

## 7. Implementation order

1. **PR 1 (correctness, small)**: §5 regression test, then §4 single-writer
   refactor which removes the compile-time `-1` path. Pure `MLIRGenStore.h` /
   `MLIRGenInterfaces.cpp` change, no lowering changes.
2. **PR 2 (simplification, medium)**: §3 side table + symbol-emitting builder
   + collapse of the patch path to the fallback cases. Expect a net-negative
   diff.
3. **PR 3 (optional)**: revisit whether the #251 `NewOp` is still needed on
   the fallback path (it is, whenever a captures-bearing literal is cast
   inside a global initializer — keep unless proven otherwise).

Test matrix per PR: full suite (714 at time of writing); the
`export-import-object-literal-with-interface` pair; a **new** cross-module test
casting an *imported object-typed value* to an interface in the importer
(exercises the §3 fallback — currently uncovered); a captures-bearing literal
cast to an interface (check whether covered; add if not); the §5 cast-order
test.

Known risks: two-pass compilation (`Stages::Discovering`) — the side table
must be populated consistently in whichever pass builds the vtable initializer;
and `@dllimport` type reconstruction must keep *missing* from the side table
(never a stale hit) so imported types deterministically take the fallback.

### Test matrix follow-up (post-PR2, 2026-07-19)

**Captures-bearing literal cast to an interface**: was genuinely uncovered:
neither `00funcs_capture.ts` (captures an outer var in an object-literal
method, but never cast to an interface) nor any of `00interface_object.ts` /
`00interface_global_method.ts` / `00interface_object5.ts` (cast to an
interface, but capture-free) combined both properties. Added
`00interface_captures.ts`. Works correctly, confirming §3's PR2
implementation note that captures need no fallback: `--emit=mlir` shows the
vtable slot gets the constant `SymbolRefOp` (one lifted function shared by
every `make(x)` call) and exactly one `ts.New` (boxing the literal, which
carries the per-instance `.captured` field the shared function reads via
`this`) — no second heap allocation for a patched vtable. 720/720 with this
test added.

**The imported-object-type fallback is not just untested - it's currently
broken, two different ways, independent of this arc's changes.** Attempting
to actually write the cross-module test surfaced two separate PRE-EXISTING
bugs (reproduced on `main@1db740c6`, i.e. before this arc's changes were ever
written into that call path in a way that matters here — both crash sites
predate PR2 and are unrelated to `objectLiteralMethodSymbolsMap`):

1. `export var counterObj = { count: 0, inc() {...} };` (no explicit type
   annotation, inferred boxed `ObjectType`) exports its declaration as `let
   counterObj : object;` — the cross-module declaration-serialization
   mechanism has no way to write out an inferred object-literal's structural
   shape, degrading to the bare `object` type. Reimporting and casting that
   to an interface in the importer hits `mlir::cast<mlir_ts::TupleType>`
   asserting false at `MLIRGenInterfaces.cpp:312`
   (`mlirGenObjectVirtualTableDefinitionForInterface`), because
   `objectType.getStorageType()` for the reconstructed bare-`object` type
   isn't a `TupleType`/`ObjectStorageType` at all. Stack confirmed via
   WinDbgX (`WinDbgX.exe -pv -p <pid> -c ".dump /ma <path>" -c "qd"` attached
   to the process while it was blocked on the assert's message box, then
   `~*k` on the dump) - crash path: `MLIRGenCast.cpp:1621/1672`
   (`castObjectToInterface`) → `MLIRGenInterfaces.cpp:193`
   (`mlirGenCreateInterfaceVTableForObject`) → `:312`.
2. Giving the export an explicit structural type annotation
   (`export var counterObj: { count: number; inc(): void } = {...}`) avoids
   the `object`-degradation (the declaration now serializes as a real tuple
   type, `let counterObj : [count:number, inc:() => void];`) but the literal
   no longer gets boxed as `ObjectType` at all (plain tuple instead) — and
   casting *that* cross-module tuple value to a method-bearing interface in
   the importer hits `llvm_unreachable("review usage")` at
   `CastLogicHelper.h:765`, a pre-existing dead/unimplemented branch in the
   LOWERING-level cast dispatcher (`castLLVMTypes`'s "value to ref of value"
   case).

Neither is caused by or related to `objectLiteralMethodSymbolsMap`/§3's
side table — both crash before any vtable-slot content is decided, in the
object-type reconstruction and generic-cast layers respectively. This means
casting a cross-module method-bearing object VALUE to an interface inside
the importing module does not currently work at all, regardless of how it's
exported. **Test not added** (a test that's expected to crash doesn't
belong in the regression suite); the attempt and both crash sites are
recorded here instead. This is a distinct, deeper bug area (object-literal
type export/reimport across `@dllimport` boundaries, and the generic
cross-type-system cast lowering) than the vtable-slot work in §3-§5, and
is a candidate for its own separate investigation if it becomes worth
fixing - not part of this arc. `export_object_literal_with_interface.ts`'s
existing pattern (declare the export *already typed as the interface* at
its definition site, `export var counter: Counter = {...}`, so the cast
happens in the exporting module where a local `funcOp` genuinely exists)
remains the only currently-working way to share a method-bearing object
across modules, and is unaffected by any of the above.

### Bug 2 crash fixed, but a third bug blocks full correctness (2026-07-19)

The `CastLogicHelper.h:765` `llvm_unreachable("review usage")` from bug 2
above is fixed: `castTypeScriptTypes`'s `resFuncType`/`BoundFunctionType`
branch (the "losing this reference" case, used when a bound method value is
narrowed to a plain function pointer for a vtable slot) was missing the
equivalent `HybridFunctionType` case - the actual type of a method field
read off a cross-module reconstructed tuple. Added the missing branch,
mirroring the existing `BoundFunctionType` one exactly (same `GetMethodOp`
extraction, same "losing this reference" warning, same rationale: an
interface vtable call re-supplies its own `thisVal`, so the value's own
captured/bound `this` is safe to drop).

This alone does **not** make the scenario work end-to-end. Verifying it
with a real repro (`counterObj: {count,inc}` exported untyped-as-interface,
cast to `Counter` in the importer, called via `-shared` compile+link+run,
confirmed with WinDbg breakpoints on the actual call site and live
register/memory inspection) surfaced a **third, independent, pre-existing
bug**: the object clone built when casting the cross-module tuple VALUE to
an interface (the one behind the "Cloned object is used" warning) writes
its two fields in the wrong order - `inc`'s function pointer lands at
offset 0 and `count` at offset 8, reversed from the tuple's actual declared
layout (`struct<(f64, ptr)>`, count@0/inc@8) that the vtable's field-offset
slot for `count` is computed against. Net effect at runtime: `c.inc()`
doesn't crash, but silently corrupts the `inc` funcptr slot instead of
incrementing `count` (confirmed: `rax`/`rcx` and `dq rcx L2` at the call
site show `[thisVal+0] = <inc's real address>`, `[thisVal+8] = 0`, exactly
backwards from what the vtable's `count` offset assumes). This is a
different code path than `CastLogicHelper.h` - almost certainly in
`MLIRGenCast.cpp`'s `castObjectToInterface` object-cloning logic (wherever
the "Cloned object is used" warning is emitted) - and is a separate,
not-yet-investigated bug. No regression test was added for the
now-fixed crash (a full end-to-end test would still fail, on this new bug,
not the old one); this section records the fix and the newly-found blocker
for whoever picks up the clone-bug investigation next.

### Clone field-order bug FIXED (2026-07-19, follow-up to the above)

Root cause of the reversed clone: both clone-building blocks (in
`castTupleToInterface` and `castObjectToInterface`, MLIRGenCast.cpp) built
the clone's field list from `InterfaceInfo::getTupleTypeFields`, which
emits **methods first, then fields** (MLIRGenStore.h) - so for
`Counter {count; inc()}` the clone's layout became `[inc@0, count@8]`,
reversed from the source tuple's `[count@0, inc@8]`. Two consumers still
addressed fields by the source layout: the interface vtable's field-offset
slots and - fatally, unfixable from the importer - the exporting module's
already-compiled `inc` body, which hard-codes `count` at offset 0. Hence
`c.count` read the funcptr slot and `inc()` incremented it.

Fix: new file-local helper `getInterfaceCloneFields` (MLIRGenCast.cpp),
used by both clone sites - the clone now preserves the SOURCE tuple's field
order, taking only the field TYPES from the interface (the type-coercion
purpose of the clone, e.g. si32 -> number or func-typed field vs method
funcType), with interface-only members appended after the source fields.
All member lookups downstream are name-based, so only the byte layout was
at stake.

With this plus PR #256, the full cross-module scenario finally works:
regression test pair added -
`export_object_literal_structural_typed.ts` (structurally-typed
method-bearing export, NOT interface-typed at the definition site) /
`import_object_literal_structural_typed.ts` (casts the imported value to
the interface in the importer, asserts count 0 -> 1 -> 2 through the
interface), wired as
`test-{compile,jit}-shared-export-import-object-literal-structural-typed`.
Note the cast still clones (warning stands): mutations through the
interface don't write back to the exporter's global - that's the
documented value-semantics of this cast, not a bug in this arc. Known
remaining sharp edge (pre-existing, unchanged): a clone whose field TYPES
are size-changing coercions (e.g. si32 -> f64 number) still shifts offsets
relative to already-compiled method bodies expecting the original layout;
that can only bite literals whose inferred field types differ in size from
the interface's, and is out of scope here. 722/722 suite (720 + 2 new).

### Bug 1 (untyped export -> bare `object`) - printer fix implemented, PAUSED before PR (2026-07-19)

Attempted the fix [[imported-object-interface-cast-bugs]] flagged as
"not yet investigated": `MLIRPrinter::printType`'s `ObjectType` case
(include/TypeScript/MLIRLogic/MLIRPrinter.h) unconditionally printed the
literal string `"object"`, discarding the storage type entirely - the root
cause of the `let counterObj : object;` degradation. Fix (implemented,
**uncommitted on branch `fix-untyped-object-export-degradation`, not
pushed, no PR**): print the structural shape via the already-existing
(previously dead-code) `printObjectType` helper when the storage type is a
`TupleType`/`ConstTupleType`/`ObjectStorageType`, falling back to bare
`object` only when it's genuinely opaque. Confirmed this alone fixes the
originally-described crash: the declaration now round-trips as
`let counterObj : {count:number, inc:() => void};` instead of
`object;`, and no longer hits the `mlir::cast<mlir_ts::TupleType>` assert
at `MLIRGenInterfaces.cpp:312`.

**But this does not make the untyped-export scenario work end-to-end -
verifying surfaced a fourth, deeper bug**: an untyped, method-bearing
object literal (`export var counterObj = {...}`, no annotation) is
auto-boxed as `ObjectType` (pointer-indirected) in the EXPORTING module,
per the object-literal-boxing rule from the #248/#249 arc. The printed
`{...}` text is a plain TS structural type-literal syntax with no way to
say "and this should be boxed" - there is no such TS source syntax, boxing
is purely an expression-shape heuristic applied to object LITERALS, never
to type ANNOTATIONS. So the IMPORTER, parsing that declaration as an
ordinary type annotation, reconstructs an UNBOXED `TupleType` - a
representation mismatch against the exporting module's actual boxed
global. Confirmed via WinDbg (same technique as the earlier bugs in this
file): crashes with a null-funcptr `call rax` DURING the
`<A.Counter>A.counterObj` cast construction itself, before any method call
or field read - earlier/more fundamental than the clone-field-order or
width-coercion bugs above.

Also independently reconfirmed via a same-module (no cross-module import
at all) repro that the width-coercion sharp edge noted above is real and
pre-existing: `{ count: 0, ... }` (integer literal, infers `s32`) cast to
an interface declaring `count: number` reads back
`4.94066e-324` (= the bit pattern of small int `1` misread as an 8-byte
double) after one `inc()` - not a NaN or a crash, silently wrong. Using a
float literal (`count: 0.0`, infers `number`/f64 directly, no
width-narrowing) avoids this specific bug and cast/mutate/read correctly
same-module - but does NOT avoid the boxing-mismatch crash above when
cross-module, since that one triggers before any field is ever touched.

**Not yet investigated**: how to recover "boxed-ness" across the
`@dllimport` boundary for an untyped export. Two directions worth
comparing before picking one: (a) teach the declaration-EXPORT side to
emit some signal distinguishing "this var's type should reconstruct as
boxed `ObjectType`" (the printer alone can't invent new TS syntax for
this - would need either a compiler-internal-only declaration extension or
piggybacking on existing syntax in a way the ordinary parser still
accepts), or (b) teach `@dllimport` type RECONSTRUCTION (whichever code
turns a parsed type annotation into the `var`/`let`'s MLIR type during
declaration-file processing) to apply the same "structural type with
method members -> box as ObjectType" rule that literal EXPRESSIONS already
get, at least for declaration-only (no initializer) `@dllimport` bindings.
Direction (b) seems lower-risk (touches reconstruction, not the
established literal-boxing heuristic or wire format) but wasn't explored.
Sequencing note for whoever picks this up: this bug must be fixed BEFORE
the width-coercion one matters for the untyped-export cross-module case,
since it crashes strictly earlier in the same call path.

**Follow-up (2026-07-20): confirmed the two directions are NOT symmetric.**
User hypothesized declaring an explicit intermediate `object`-typed
binding (`export let counterObj2: object = counterObj;`) might sidestep
the mismatch. Tested: it does not - it regresses to the ORIGINAL
`mlir::cast<mlir_ts::TupleType>` assert (MLIRGenInterfaces.cpp:312)
instead of the boxing-mismatch crash. Reason: `counterObj2`'s own MLIR
storage type is genuinely opaque (widened away, not just a printer
omission - its declaration prints as bare `object` even WITH the printer
fix applied, confirming the concrete shape is erased at assignment, not
just at print time), so it has strictly LESS structural information than
`counterObj` itself, not more. This rules out "introduce an explicit
`object`-typed alias" as a workaround and confirms direction (b) above
(teach `@dllimport` reconstruction to box a method-bearing structural
type annotation, matching the literal-expression boxing rule) is the
right next thing to try - direction (a) (encode boxed-ness in the
declaration syntax itself) has no natural TS syntax to piggyback on,
as an `object`-typed annotation turns out to mean "opaque", not "boxed
structural".

### Newly found: multi-method cross-module vtable slot bug (2026-07-19)

Found while extending test coverage beyond this arc's fixes - every prior
test/fix in #256-#258 only ever exercised a **single-method** interface
cast cross-module (`Counter {count; inc()}`). Trying a genuinely
multi-method interface (`Accumulator {total; add(n); addTwice(n);
scaled(factor): number}`, canonical vtable order after
`assignCanonicalVirtualIndexes` = methods-first-in-declaration-order then
fields = `add`@0, `addTwice`@1, `scaled`@2, `total`@3) surfaced a clean,
reproducible pattern when casting a cross-module structurally-typed VALUE
to it and calling each method **in isolation** (bisected one at a time via
a temporary `test-runner.cpp` stdout-surfacing patch, same technique as
earlier bugs in this file - reverted before commit):

| method (canonical slot) | isolated result |
|---|---|
| `add(n)` (slot 0) | correct - mutates `total` as expected |
| `addTwice(n)` (slot 1) | WRONG VALUE, no crash - `total` ends up incorrect but the process completes and reports the mismatch cleanly |
| `scaled(factor)` (slot 2) | CRASH - silent, no assert/error text reaches output at all (raw access violation with buffered stdout lost, unlike the controlled assert failures elsewhere in this file) |

Slot 0 works, slot 1 is wrong-but-survives, slot 2 crashes outright -
consistent with SOMETHING going wrong specifically in how slots beyond 0
are constructed or addressed for a cross-module structurally-typed cast
(as opposed to the field-order bug from earlier in this file, which was a
uniform reversal affecting all slots equally and is already fixed). Not
yet root-caused - candidates worth checking first: whether
`getInterfaceCloneFields`/the vtable-patch loop in
`mlirGenCreateInterfaceVTableForObject` (MLIRGenInterfaces.cpp) iterates
methods needing patching in the right order relative to the CANONICAL
`virtualIndex` for interfaces with >1 method (an off-by-one or
wrong-iteration-source bug would explain "slot 0 fine, slot 1+ broken");
or whether the heap-cloned vtable's allocated SIZE is computed from a
stale/undersized type (a 2-slot allocation for a >2-slot vtable would also
match this exact crash-only-past-slot-N shape).

**Not fixed.** The regression test actually added for this session
(`export/import_object_literal_structural_typed_params.ts`) deliberately
stays within the single-method shape that's known to work, to avoid
committing a failing test; it extends coverage only along the
"zero-arg -> parameterized method" axis, not the "single-method ->
multi-method" axis. A genuinely multi-method cross-module test is blocked
on this bug and is the natural next thing to add once it's fixed.

Also worth noting for whoever investigates: a completely SEPARATE,
same-module-only finding surfaced while building the initial (broken)
version of this test - a value-returning method cannot `return
this.siblingMethod(...)` (using a sibling call's return value directly in
a `return` statement) within the same type-literal-annotated object
literal; calling the sibling as a bare statement (discarding its return
value) works fine. Confirmed same-module, unrelated to cross-module
casting at all. Avoided in the committed tests
(`00object_annotated_method_params.ts` uses `setBase`/re-`scale`, not a
chained-return pattern). See the next section for the full mechanism,
found in a follow-up session.

### Chained-return inference gap: mechanism found (2+ of 3 layers fixed), root cause remains (2026-07-19)

Root-caused via `--debug-only=mlir` (traced `discovering 'return type' &
'captured variables'` LLVM_DEBUG output). The mechanism, precisely:

1. An enclosing function with no explicit return type (e.g. `main()`) runs
   a speculative **discovery pass** first
   (`discoverFunctionReturnTypeAndCapturedVars`, MLIRGenFunctions.cpp) to
   infer its return type and captured variables, via a dummy `FuncOp` and
   `GenContext{dummyRun=true, allowPartialResolve=true}`.
2. That pass walks the WHOLE body, including unrelated statements like
   `let calc = {...}` - which recurses into the object literal's methods
   (`processObjectFunctionLikeProto`/`processObjectFunctionLike`,
   MLIRGenImpl.h), each of which ALSO runs the same discovery machinery
   for ITS OWN return type if not explicit on the method's own literal
   syntax (or, it turns out, even when it IS explicit - see below).
3. Methods are discovered in declaration order, but a method's prototype
   is only registered into the literal's (mutable) `ObjectStorageType`
   AFTER its OWN discovery completes - so when `scaleAndAdd`'s body (being
   walked to infer test ITS return type, or just to find captured vars)
   references `this.scale(...)`, `scale`'s prototype may not be in the
   storage type yet EVEN THOUGH `scale` is declared earlier in the source
   and even when `scaleAndAdd` HAS an explicit return type on its own
   literal syntax (confirmed: adding `: number` to `scaleAndAdd` itself
   did not avoid the failure - the discovery machinery runs regardless,
   apparently also needed for captured-variable discovery).
4. Property access on `this` for `scale` then fails to resolve
   (`mlirGenPropertyAccessExpressionBaseLogic`, MLIRGenAccessCall.cpp) -
   but ONLY when the call's result is USED (assigned, returned, or part of
   a larger expression). A bare-statement call (`this.scale(factor);`,
   result discarded) does not hit this: statement calls don't need the
   property access to produce a typed VALUE the way an expression context
   does, so they don't trip the same resolution path during discovery.

**Fixed** (two gates, both matching the SAME `dummyRun`/`allowPartialResolve`
tolerance idiom already used elsewhere in both files - e.g.
`mlirGenCallExpression`'s `!result.value && genContext.allowPartialResolve`
case in MLIRGenAccessCall.cpp):
- `mlirGenPropertyAccessExpressionBaseLogic`'s final
  `emitError("Can't resolve property...")` now returns `mlir::success()`
  (no value, not a hard failure) when `genContext.dummyRun ||
  genContext.allowPartialResolve`.
- `discoverFunctionReturnTypeAndCapturedVars`'s
  `emitError("'return' is not found...")` now skips the diagnostic (still
  returns `mlir::failure()` to signal "this discovery attempt didn't
  converge", but without recording a user-facing error) under the same
  condition, checked against the OUTER `genContext` (the nested
  discovery's caller) rather than the freshly-constructed
  `genContextWithPassResult` (which is unconditionally
  `dummyRun=true`/`allowPartialResolve=true` for ANY discovery call,
  nested or not, and so can't distinguish "nested inside another dummy
  run" from "top-level, real discovery" on its own).

Both verified individually and together: no change in behavior for any
existing test, full suite stayed at 730/730 with both fixes applied.

**NOT fixed - a third, more fundamental layer remains.** Even with both
gates, the original repro still fails, now with a bare
`error: failed statement` for `main()` and no more specific diagnostic.
Cause: `mlir::failure()` - even without an emitted diagnostic - still
propagates as a hard abort through ordinary SEQUENTIAL STATEMENT
processing (`mlirGenFunctionBody` and friends don't distinguish "a nested
speculative sub-discovery legitimately didn't converge, continue as if
this statement is fine" from "a real error occurred, stop everything").
So `scaleAndAdd`'s nested discovery failure - now silent, thanks to the
two gates above - still bubbles up through processing the (entirely
unrelated to `main`'s return type) `let calc = {...}` statement, aborting
`main`'s WHOLE discovery pass, which is where the final `failed statement`
error comes from. A real fix needs the statement-processing layer (or
whichever call site turns "one nested dummy sub-discovery didn't
converge" into "abort the entire enclosing discovery") to tolerate this
too - and it's not yet known how many such call sites exist, since this
is the third layer found and each one so far needed its own targeted
gate. Deliberately not chased further this session - the two committed
gates are real, idiom-consistent, and non-regressing improvements on their
own, but do not by themselves fix the user-visible repro. No regression
test added (the repro this was investigating still fails); the two
partial fixes are covered only by "full suite still green," not by a
test proving the original bug is fixed.
