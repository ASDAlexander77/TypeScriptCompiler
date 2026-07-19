# Interface vtable simplification: design

Status: **PR 1 (§4 + §5) implemented, full suite green (716/716)**. §3
(constant vtables for capture-free literal methods) not yet started. Follow-up
to PR #251 (heap-allocated patched vtable) and PR #252 (canonical slot
numbering, `fix-interface-vtable-index-mismatch@be2c9620`). All anchors below
verified by code inspection on that branch, except where PR 1's implementation
notes (end of §4) correct them. Goal: remove the last *runtime-patched* vtable
path for the common case, make slot numbering single-sourced, and fix one
latent cast-order miscompile found during review.

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
