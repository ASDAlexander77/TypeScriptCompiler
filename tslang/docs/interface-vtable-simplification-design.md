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
