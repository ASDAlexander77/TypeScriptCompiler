# Class member table: one by-name description of a class, for `c[k]`, `k in c`, `any[k]` and shared-library imports

Date: 2026-11-20. Status: draft for approval.

## Intent and why

A class's run-time identity today is a set of separate pieces: the class value
(`.vtable`, `.new`, `.constructor`, `.rtti`, `MLIRTypeHelper::getClassValueStorageType`),
the vtable (methods and, with `ADD_STATIC_MEMBERS_TO_VTABLE`, statics, by slot index),
and - on main since #940 - a per-type generated key adapter in the type descriptor's
`TYPE_DESCR_KEYS` slot that answers `box[key]`, `key in box`, `box[key] = v` for an `any`.

What is missing is **field offsets and member names as data**. That gap shows up in two places:

1. **Shared libraries (`@dllimport` / `ClassInfo::isDynamicImport`).** The importer re-parses the
   library's `__decls_<module>` text and recomputes the class layout itself; field offsets become
   constants in the importer's code. Methods, `.vtable` and `.rtti` are already looked up by symbol
   at run time - fields are the one part of the layout fixed at compile time. Any mismatch (library
   rebuilt with a new private field, a layout-relevant decorator the printer forgot) corrupts memory
   silently. `DeclarationPrinter.cpp` carries several "layout-relevant" comments for exactly this.
2. **Access by a run-time name on a typed class** (`c[k]`, `k in c`) is not supported; on an `any` it
   goes through a generated switch per type, which covers public fields only and is refused under
   `-mm=own`.

The user wants one table per class that describes every member by name - kind, offset or address,
type - and that every by-name feature reads.

## Goals

- One **member table** per class (and one for its statics), emitted as read-only data.
- `c[k]` and `k in c` on a typed class instance with a run-time key (rules below).
- `C[k]` / `k in C` on a class value with `k: keyof typeof C`, through the statics table.
- `x[k]`, `k in x`, `x[k] = v` on an `any` holding a class instance read the same table (the class's
  `TYPE_DESCR_KEYS` entry becomes a thin wrapper over the shared table walker).
- Methods reachable by name: `c["m"]` / `c[k]` yields the method bound to `c`.
- Shared-library imports **check at load**: when a library loads, the importer compares the offsets and
  member types it compiled against with the library's tables and stops with a clear message on any
  mismatch.

## Non-goals

- **Resilient (bind-at-load) access.** Importer field accesses keep compile-time offsets; libraries
  still cannot change layout without importers being rebuilt. The table makes that possible later
  (user chose check-at-load, 2026-11-20).
- JS `import()` expressions (still a clear error, ruling 11-18).
- Reflection APIs (enumerating members from user code, `Object.keys` on a class instance) - the data
  will be there, but no new surface is added here.
- Tuples, object literals, interfaces and arrays keep their generated `TYPE_DESCR_KEYS` adapters
  (builder's call to unify later; see "Left to the builder").
- Writes through `string` keys.

## Anti-goals

- Objects must not grow. The table hangs off the class/type descriptor, never off an instance.
- A program that never reads a class by name must not pay for its table (size or link-time keeping
  of otherwise dead methods).
- No silent wrong values: a member a key does not name is `undefined` (reads) or a TypeError (writes);
  a library mismatch is a load-time error, never a misread.
- No change in behaviour of existing `any[k]` programs beyond what is listed (methods now reachable).

## Constraints

- **Cross-module descriptor contract** (`Defines.h`, "Runtime type descriptor"): slots may be
  added, never reordered or resized; a new slot goes immediately before `TYPE_DESCR_BLOCK_HEADER`,
  which stays last. Adding a slot is a layout change: rebuild the default library (all models, debug
  and release - see memory notes on `build.bat` env), the wasm and Android archives, and invalidate
  the JIT cache. Coordinate a DefaultLib PR.
- Works under `-mm=gc`, `rc`, `own`, `none`; on Windows/Linux/wasm32/wasm64/Android; exe, JIT, DLL.
- `-mm=own`: reads through `keyof C` keys only (typed union result, no `any` box). Anything that
  would box into `any` keeps today's clear error (the own `any[k]` variant was reverted, #942 notes).
- Offsets must be target-correct (32-bit targets, wasm32 accessor tag bit 30 - reuse the interface
  slot tag macros, `INTERFACE_OPTIONAL_STORAGE_TAG`, `INTERFACE_METHOD_CELL_TAG`,
  `INTERFACE_ACCESSOR_CELL_TAG`).
- Library `__decls` text keeps being the source of types for the importer; the table does not replace it.

## Approaches considered

- **A. One data table; generated per-type functions become thin wrappers over a shared walker.**
  Chosen (user, 2026-11-20). One description of each class serves `c[k]`, `k in c`, `any[k]`, statics
  and the import check. Costs one descriptor layout change and a slightly slower lookup than a
  per-class switch.
- B. Keep generated adapters, add the table for imports only. Smallest change, but two descriptions of
  each class to keep in sync, and methods-by-name would mean extending every adapter.
- C. Generated adapters only, plus an "import" op. No data, but the importer can only probe one name
  at a time and cannot list or report what is missing.

## Design (as decided)

### The table

Per class, read-only, internal linkage like the type descriptor (`<class>.members`), plus
`<class>.static_members`. Shape:

```
header:  { count: u32, flags: u32, class rtti: ptr }
entries: { name: ptr to NUL-terminated name (pooled per module),
           kind: u8   (field | optional field | method | accessor | static field | static method),
           access: u8 (public | protected),
           readonly: u8,
           slot: word  (field: offset, with the interface-slot tag bits for optional / accessor-cell /
                        method-cell; method: function address; static field: address of the global),
           type: ptr   (the member's type descriptor: kind, layout id, release/retain/share,
                        so the walker can box, retain and compare types without per-class code) }
```

Entries sorted by name so lookup is a binary search over interned strings (builder may pick a hash
instead). Private members are omitted: `c["priv"]` with a literal key stays a compile-time access, as
in tsc; `keyof C` never contains private or protected names. Protected entries exist for a subclass
in another module and for the import check; a run-time key never reaches them.

Reached from: a new type-descriptor slot `TYPE_DESCR_MEMBERS` (before the block header) for instances,
and a new class-value field `.members` beside `.rtti` for the class value (statics table pointer;
the instance table is reachable through `.rtti`/descriptor).

### When a class gets a table

Only when something needs it (my call, unopposed): the class is exported (`--emit=dll`, `-shared`,
`export` in a library build), or a value of it is boxed into `any`, or the module reads it by a
run-time key (`c[k]`, `k in c`, `C[k]`). Made at module end like `makePendingAnyKeysAdapters`. A
plain executable that does none of these emits nothing.

### `c[k]` on a typed class instance (user decisions 2026-11-20)

1. `k: keyof C` (or a union of literal names): result type is the union of those members' types; a
   method's type is its bound function type. Lookup through the table, value returned unboxed into the
   union. Works under every model including `own`.
2. `k: string` (C has no index signature): result is `any`, `undefined` for a name C does not have -
   same as `(c as any)[k]` (ruling 11-19). Refused under `own` with a clear error.
3. Writes `c[k] = v`: only with `keyof C` keys, only to non-readonly fields; a run-time check rejects a
   value whose type does not match the named field (TypeError), e.g. union member mismatch. Not
   through `string` keys (error).
4. `k in c`: true when the table has a public entry named `k` (an optional field counts only while
   set, as #940 does).
5. Statics: `C[k]` with `k: keyof typeof C` reads the statics table, same rules.

### `any[k]` on a class instance

The class's `TYPE_DESCR_KEYS` entry keeps its `(box, op, key, value)` contract; for a class it is a
small wrapper calling the shared walker with the class's table. Methods are now reachable (bound to
the boxed object). Everything else is unchanged.

### Shared-library imports: check at load (user decision 2026-11-20)

The library exports its classes' tables by symbol (`<fullName>.members`, like `.vtable` / `.rtti`).
For each imported class the importer emits, into the existing library-load global constructor
(`MLIRGenModule.cpp`, `LoadLibraryPermanentlyOp` block), a compact expected list: for every member it
accesses through compile-time layout - name, kind, offset, type layout id. At load a runtime routine
looks each up in the library's table and compares. Mismatch → print
`library '<path>': <Class>.<member> <what changed>; rebuild this program against the new library`
and exit non-zero. Missing table (old library) → today's behaviour plus a warning at compile time.
Also check the instance size the importer assumes when it subclasses a library class.

## Actual class for string keys (user decision 2026-11-20)

`c[k]` with `k: string`, `k in c`, and `x[k]` on an `any` holding a class instance read the table of the
object's **actual** class, as JavaScript does: a `D` held as `C` finds `D`'s own public fields. To reach
that table from an instance, the class vtable gets a header slot holding the member table pointer,
right after `.instanceOf` (`CLASS_VTABLE_INSTANCEOF_SLOT`). Every class that has a table therefore gets
a vtable, even one that would otherwise have none. `@value` classes have no subclasses, so their static
type is their actual class, and they need no slot. This shifts every class vtable slot by one: a
cross-module layout change with the same rebuild cost as the descriptor slot. `@cpp` vtables are not
touched. `keyof C` keys may still read the static type's table, because their result type comes from `C`
and field offsets are the same in a subclass.

## Calls made after the builder check (author's calls, open to the user's correction)

- **Overrides.** A method entry for a virtual method holds its **vtable slot index**, not an address, and is read
  through the object's own vtable. `c[k]` on a `D` held as `C` therefore runs `D`'s override. Field offsets
  are the same in a subclass (the base storage is embedded first), so reading the static type's table
  for fields is correct. Non-virtual and `@final` methods hold addresses.
- **`-mm=own` and methods.** A bound method moves its object (ruling 11-16), so under own `c[k]` is refused
  with a clear error when `keyof C` (or the given key union) contains a method. Unions of field types
  only are allowed. A result union holding several classes uses the existing instanceof dispatch
  (#884-#890).
- **Writes follow tsc.** `c[k] = v` with `k: keyof C` requires `v` to be assignable to the
  **intersection** of the named fields' types, as tsc does. That needs no run-time type check. A key that
  names a readonly field, or a method, is a compile error. (This replaces "run-time check" in
  decision 3. The result is stricter, and it is checked at compile time.)
- **Type identity in the table.** Each entry also holds `MLIRHelper::hashOfType` of the member's declared
  type (computed from the type text, so it matches across modules). The import check and nothing else
  compares it. `layoutIdOf` is not used for this, so the type descriptor contract is unchanged beyond the
  new `TYPE_DESCR_MEMBERS` slot.
- **Import check scope.** Only `@dllimport` libraries loaded at run time (`isDynamicImport`) are covered.
  Statically linked `.lib`/`-shared` imports and the default library are not. A library built with tables
  adds the marker line `// @members` to each `__decls_<module>`. An importer that sees no marker warns
  at compile time ("library predates layout checks") and skips the check.
- **Cost.** Tables in a static library (the default library) are not exported and are referenced only by
  the code that needs them, so `--gc-sections` drops unused ones. In a DLL the tables are exported and
  keep their methods alive, as their exported symbols already do. Boxing a class into `any` makes its
  table, so methods of boxed classes stay linked. That cost is accepted. The PR reports the size
  change of the default library and the test executables.

## Left to the builder

- Hash vs sorted-array lookup; name interning across modules (compare by string, not pointer).
- Exact encoding of `kind`/`flags`; whether accessors get one entry with a cell or two.
- Whether the expected-layout check list is emitted for all members of an imported class or only the
  ones the importer touches.
- Whether tuples/interfaces later move onto the same table (out of scope here).
- Test plan: per model (gc/rc/own/none), exe + JIT + DLL, wasm; library-mismatch test that rebuilds a
  library with a reordered field and expects the load-time error.
