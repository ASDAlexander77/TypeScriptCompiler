# Arrays as references

Date: 2026-10-03. Status: design approved in conversation (sections 1-4); written spec awaiting
review. Fixes #453 (a push onto an array parameter is lost to the caller) and #477 (a
destructuring rest element aliases its source). Plans go in `docs/superpowers/plans/`.

## 1. Purpose

TypeScript arrays are references: every variable, parameter, field and element that holds an
array holds the same array, and `push`, `pop`, `splice` and `length =` through any of them are
seen through all of them. tslang lowers an array to a two-word value `{ data, length }`. Copies
share the elements until the first change of length, and then part ways: the copy that changed
has a new length and, because `push` reallocates on every call, often a new data block, while
every other copy keeps the old length and a pointer the reallocation may have freed.

This design makes an array value one pointer to a heap header that never moves.

Decisions taken in the design conversation (do not re-litigate):

| Question | Decision |
|---|---|
| Fix the parameter case only (pass mutated array parameters by reference)? | No. That leaves local aliases, record fields, element aliases and arrays stored into fields broken (§2), and needs a signature marker carried across modules and function types. |
| Header contents | `{ data, length, capacity }`. |
| Growth | Amortised doubling (PR 3). PR 2 keeps capacity equal to length. |
| Rollout | Three PRs: a behaviour-free refactor through one layout helper, the switch, then growth (§7). |
| Where the spec lives | Its own file: the change is not specific to `-mm=own`. |

## 2. The evidence

Fourteen shapes, `tslang --emit=jit --no-default-lib`, main 9a5cc05b with #476's change (global
constructor order only). "Works" means the change made through the second name is seen through
the first. Under own the shapes were one file, which the four compile errors stopped; "-" means
not run separately, and shapes 2 and 13 were run on their own.

| # | Shape | gc / rc / none | own |
|---|---|---|---|
| 1 | element write through a parameter | works | - |
| 2 | `push` through a parameter | length stale | crash (exit 127) |
| 3 | `pop` through a parameter | length stale | - |
| 4 | `length = 0` through a parameter | length stale | - |
| 5 | local alias `const b = a; b.push(..)` | length stale | compile error (borrow) |
| 6 | class field in place, `h.items.push(..)` through a parameter | works | - |
| 7 | field alias `const it = h.items; it.push(..)` | length stale | compile error (borrow) |
| 8 | object-literal field through a parameter | length stale | - |
| 9 | closure capture | works | - |
| 10 | `aa[0].push(..)` | works | - |
| 11 | element alias `const x = aa[0]; x.push(..)` | length stale | compile error (borrow) |
| 12 | `a === b` after `b = a` | works | - |
| 13 | four pushes through a parameter, then read `a[0]` | gc reads 0 (old block); rc crashes | crash |
| 14 | array stored into a field, then pushed through the original | length stale | compile error (move) |

The shapes that work are the ones where the struct itself lives in memory both sides address (a
class instance, a capture cell, the outer array's data block). Every shape that copies the
struct fails. Under rc a parameter push exits with 127 and prints nothing.

## 3. What an array is

- An array value (`T[]`, the typed-array aliases `Int8Array` .. `BigUint64Array`, `string[]`) is
  one pointer to a heap **header** `{ data: ptr, length: index, capacity: index }`, in that
  order. The header is allocated when the array is made and never moves; only `data` is
  reallocated.
- Assigning, passing, returning, storing into a field or element, capturing and boxing into `any`
  or a union copy the pointer. `===` and `!==` compare header pointers.
- An empty array is a header with `data = null`, `length = 0`, `capacity = 0`. There is no shared
  empty header: each `[]` is a distinct array.
- A null header (zeroed memory: an element of `new Array<T[]>(n)`, a field with no initializer,
  `undefined` cast to an array) reads as an empty array, and an op that changes an array through
  its slot stores a fresh empty header into a slot holding null first. This keeps today's
  behaviour, where a zeroed `{ data, length }` is an empty array (plan ruling R1).
- A null slot passed by value to a parameter (an unset field, an element made by `length =`):
  a push through the parameter materialises a header in the parameter's copy, so the slot's
  owner does not see the change and, under rc and own, that header leaks. Accepted: such a slot
  holds `undefined` in TypeScript terms, where `.push` would throw (ruling T4-D).
- `ConstArray` (a literal's static data) is unchanged. The cast from `ConstArray` to `T[]` makes a
  header and copies the data into a fresh block on both of its paths; the path that today points
  the struct at the static data (`byValue = false`) copies as well, since `push` would otherwise
  reallocate static memory.
- `[a, ...rest] = src` gives `rest` a new array holding a copy of the slice (#477). MLIRGen builds
  it from a new empty array and a loop of pushes instead of emitting `ts.ArrayView`, so the
  ownership passes see a fresh array (plan ruling R3).
- Layout: an array field, element, tuple slot or union payload shrinks from two words to one
  pointer; `T[] | undefined` is the pointer and its tag.
- ABI: every function that takes or returns an array changes. Objects built before the switch do
  not link with objects built after it. The default library has no native code that reads an
  array (its C++ wrappers take none), so its sources are unchanged; it is rebuilt with the
  switch, for every model and both build types.

## 4. Ownership in each model

Only the lowering of the array case in the retain/release routines
(`OwnershipRoutineLogic.h`) changes. The MLIR-level ownership passes work by type and are
unchanged.

- **gc.** Header and data block come from `GC_malloc`; the header keeps the data block alive.
  The data block keeps today's allocation call.
- **none.** As today: nothing is freed. The header is one more allocation.
- **rc.** The counted block is the header.
  - Retain increments the header's count (today: the data block's).
  - Release, on the last reference: release elements `[0, length)`, free the data block, free
    the header. The data block's own count is never read: the header holds its only reference.
  - Slots `[length, capacity)` are never released (and, from PR 3, hold zero).
  - `pop` and `shift` hand the removed element's reference to the caller, as today.
- **own.** The header is a single-owned block, like a class instance.
  - A borrowed array parameter is the owner's header pointer: `push` through it changes the
    owner's array, as `h.items.push` through a borrowed object already does.
  - The rules do not change. Local and field aliases (shapes 5, 7, 11, 14) stay compile
    errors; moves, element moves and borrows are as today.
- **`Shared<T[]>`** holds the header pointer as its payload. Nothing specific.
- **Cycles.** `a.push(a)` is a reference cycle: under rc and own it leaks like a class cycle, under
  gc it is collected.

## 5. The code that changes

- **`include/TypeScript/LowerToLLVM/ArrayLayout.h` (new)** is the only code that knows the
  layout. It reads and writes `data`, `length` and `capacity`, gives an element's address, and
  makes an array from a data pointer and a length (and, from PR 2, a capacity). PR 1 implements it
  over today's struct; PR 2 changes its insides to the header.
- **Op signatures stay.** `ArrayPush`, `ArrayPop`, `ArrayShift`, `ArrayUnshift`, `ArraySplice` and
  `SetLengthOf` keep taking a reference to the slot; in PR 2 they load the header pointer from it.
  Narrowing them to take the value is out of scope.
- **Where arrays are made:** `NewEmptyArray`, `NewArray`, `CreateArray`, the `ConstArray` cast
  (`CastLogicHelper.h`, both paths), the spread builder, `ArrayView` (a copy, §3).
- **Where arrays are read:** `LengthOf`, `ElementRef` and element access, for-of, printing, the
  array casts in `CastLogicHelper.h` (to string, to an opaque pointer, `any` boxing and unboxing),
  the retain/release routines (§4).
- **Debug info** (`LLVMDebugInfo.h`): an array's debug type becomes a pointer to a
  `{ data, length, capacity }` struct, so a debugger still shows the elements.
- **`main(argc, argv: string[])`:** MLIRGen gives `main` a `Ref<string>` second parameter, so the
  entry point takes C's `char **`, and binds `argv` to a new op `ts.ArrayFromCStrings(argc, argv)`
  that makes the array with the model's allocator, copying each string (plan ruling R2). Today
  the struct is read from C's `argv` and `envp`, which is the open `argv.length` = envp bug; this
  fixes it. A `Ref<string>` argv still receives C's `char **` directly. Both AOT and JIT.
- **32-bit x86:** the header is three pointer-sized words; nothing is specific to x86.

## 6. Growth (PR 3)

- `push`, `unshift` and `splice` that need more room than `capacity` grow it to
  `max(4, 2 * capacity, needed)`; otherwise they write into the existing block.
- `pop`, `shift` and a smaller `length =` keep the block. They zero each slot they vacate, so gc
  does not keep a removed element alive and a later growth reads zero.
- `length = n` larger than `length` zeroes `[length, n)` in every model, gc included: after a
  `pop` the slots past `length` are no longer fresh memory. Today's behaviour (new slots read as
  zero; the default library's `Set`, `Map` and `Array.map` rely on it, see the comment in
  `SetLengthOfOpLowering`) is kept.

## 7. Rollout and testing

### PR 1: the layout helper (no behaviour change)

- Every layout access in lowering goes through `ArrayLayout`: the 29 `ARRAY_DATA_INDEX` /
  `ARRAY_SIZE_INDEX` uses in `LowerToLLVM.cpp`, `CastLogicHelper.h`, `LLVMCodeHelper.h` and
  `OwnershipRoutineLogic.h`, and every place that builds or takes apart the struct with raw
  indexes.
- Gate: `--emit=llvm` output byte-identical for every suite test before and after; the full
  Windows Release suite; no `ARRAY_DATA_INDEX` / `ARRAY_SIZE_INDEX` outside the helper.

### PR 2: the switch

- §3, §4 and §5: the representation, the creation and read sites, the retain/release routines,
  debug info, the `main` adapter, the rest copy. Capacity equals length.
- Tests:
  - `00array_reference.ts`: shapes 1-14 of §2 as asserts; compile and JIT, rc and none corpora.
  - An own test with the shapes own accepts, including a push through a borrowed parameter.
  - `00array_rest_copy.ts` (#477).
  - A `main(argc, argv: string[])` test for AOT and JIT.
- Gates: the full Windows Release suite; the full Linux (WSL) suite, which is required because
  the Windows heap hides double frees; the own corpus with no file lost; own's memory measurement
  flat; the default library rebuilt and its suite run under every model.

### PR 3: growth

- §6.
- Tests: many pushes then reads, through aliases; `pop` then `length =` larger reads zero;
  `shift`/`unshift` around a growth; a push-in-a-loop timing check against PR 2.

### Done when

- #453 and #477 are closed.
- Shapes 1-14 work under gc, rc and none, and under own where own accepts them.
- No `ARRAY_*_INDEX` outside `ArrayLayout.h`.
- Windows and Linux suites are green and the default library is rebuilt.

## 8. Out of scope

- Op signatures taking the array value instead of a slot reference.
- Shrinking the data block.
- A shared empty-array sentinel.
- Collecting reference cycles under rc and own (`WeakRef<T>` is a separate item).
