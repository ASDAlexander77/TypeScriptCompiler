# `@weak` fields under `-mm=own`: design

Date: 2026-11-17. Status: draft for approval.

## Why

Spec 84 gave `-mm=own` a `WeakRef<T>`, so a tree can point back at its parent without a second owner.
Every read of such a link goes through `deref()` and handles `undefined`:

```ts
const p = this.parent?.deref();
return p ? 1 + p.depth() : 0;
```

The user wants a back-reference that reads like any other field: `this.parent.v`. The reason is
ergonomics only. Run-time cost is not the concern, so the field may be checked at run time.

```ts
class Node {
    @weak parent: Node | null = null;
    children: Node[] = [];
    constructor(public v: number) {}

    add(c: Node) {
        c.parent = this;
        this.children.push(c);
    }

    depth(): number {
        return this.parent ? 1 + this.parent.depth() : 0;
    }
}
```

## Decisions taken (user, 2026-11-17)

| Question | Decision |
| --- | --- |
| Why not `WeakRef` | `deref()` and `undefined` on every read. Wanted: plain `this.parent.v` |
| A call through a read | Refused when the call may free things: a `@weak` read may not be the receiver or an argument of a call that may drop. Not a run-time pin held for the call |
| Approach | B, a checked field: a `WeakRef` underneath, with the `deref()` written by the compiler on every read. A read of a freed parent stops the program with a clear error instead of giving `undefined`. Not A (a compile-time proof, which refuses re-parenting and removal) and not C (unchecked, can read freed memory) |

## Goals

- `@weak` on a class field makes it a reference that does not own its object. Reads and writes look
  like reads and writes of any other field.
- It compiles and runs under own, gc and none, with the same output when no freed parent is read.
- Under own, no accepted program reads freed memory or leaks. A read of a field whose object was
  freed stops the program with a message naming the field, never returns garbage and never
  continues.
- Trees with parent links can be edited normally: re-parenting a child, removing one from its parent,
  dropping a subtree.
- No new run-time machinery: the weak count, tombstone, stack-promotion and thread rules are spec
  84's, reused as they are.

## Non-goals

- `-mm=rc`, for the same reason as `WeakRef` (no second header word). `@weak` there is a compile
  error naming the models that have it.
- Dropping the run-time check where the compiler could prove the object alive (approach A). It may
  come later as an optimisation. It changes no source.
- `@weak` anywhere but an instance field of a class: not on locals, parameters, parameter properties
  (`constructor(@weak public p: Node)`), static fields, interface members, object literal fields or
  array elements. Each is an error: "'@weak' applies only to a field of a class". *(My call: the
  parent-link case needs only the class field. The others can come later.)*
- A field type other than a class type, `C | null`, `C | undefined` or an optional field `p?: C`.
  Unions of several classes, interfaces, arrays, strings, `@value` classes and `Shared<T>` are errors:
  "a @weak field needs a class type".
- Catching the failure. A read of a freed object is a bug in the program, like an out-of-bounds
  write, not an expected outcome. *(My call: see "A read of a freed object".)*

## Anti-goals

- A read that returns a freed object, or one that is freed before the read value's last use.
- A field that silently becomes owning, or a write that leaks or frees an object it does not own.
- A program whose output under gc or none differs from own's, except that own stops on a read of a
  freed object.

## Constraints from the code as it is

- **Spec 84 (`WeakRef`).** Under own it is a pointer to a class object. The object's header word
  counts `WeakRef`s and marks a tombstone. Freeing an object with a non-zero count releases its fields
  and keeps the block; the last `WeakRef` frees it. Ops `ts.WeakRefNew` / `ts.WeakRefDeref`
  (`TypeScriptOps.td:2527-2540`) and type `!ts.weak_ref<T>`. A `WeakRef` value is a handle (`isHandle`)
  whose retain and release are the target's count + 1 and - 1. Under gc it is a disappearing link,
  under none a plain pointer.
- **What `deref()` returns** under own is a borrow with an unknown place (the `anyPlace` chain). It may
  be used until anything that could free an object of its type (a release of a value whose type may
  hold `T`, a call that may drop). It may not be stored, returned, captured or given to a call that
  keeps it.
- **Weak targets.** An object of a class that is the `T` of any `WeakRef<T>` in the module is never
  stack-promoted (`"weak-target"`), and under own may not cross threads.
- **Field decorators.** Class members' decorators are read in `MLIRGenClasses.cpp`
  (`iterateDecorators(classMember, ...)`, `MLIRHelper::hasDecorator(classMember, ...)`). Decorator
  names live in `include/TypeScript/Defines.h` (`*_DECORATOR`).

## Approaches considered

1. **A field proven safe at compile time.** Costs nothing at run time. It refuses any program that
   takes a child out of its parent (`pop`, `splice`, re-parenting, returning a child). Rejected by the
   user: ordinary tree editing stops compiling.
2. **A checked field, a `WeakRef` underneath (chosen).** Every shape works. Each read costs one header
   load and a branch, plus the weak count. Most of the run time already exists in spec 84.
3. **An unchecked field.** Never released, never checked. Rejected: a freed parent is read silently.

## The design

### Surface

- `@weak` is a decorator on an instance field of a class: `@weak parent: Node | null = null;`. The
  declared type is the type the program reads and writes. *(My call: a decorator, not a type such as
  `Weak<Node>`, so that `this.parent` has type `Node | null` and type-checks with no new rules. The
  name `weak` is my call too.)*
- Allowed field types: a class type `C`, `C | null`, `C | undefined`, or optional `p?: C`. For a
  generic class, the check runs on each instance: `@weak p: T` with `T` given a class is fine, with
  `T` given anything else it is the type error above.
- Under rc: "'@weak' is not supported under -mm=rc; use -mm=gc, -mm=own or -mm=none".

### At run time, by model

| | the field | a read `x.f` | a write `x.f = v` | `x` freed |
| --- | --- | --- | --- | --- |
| own | `!ts.weak_ref<C>` (a pointer to the object, its header counting it) | null/undefined as stored; else the object if alive; else stop (below) | count of `v` + 1 (unless null/immortal), old value's count - 1 (freeing a tombstone at 0) | the field's value - 1, as a `WeakRef` field |
| gc | an ordinary field of type `C` | ordinary read | ordinary write | nothing |
| none | an ordinary field of type `C` | ordinary read | ordinary write | never |

Under gc the field is a plain strong reference. A parent link then keeps its parent alive while the
child is reachable, which is what JavaScript does with a plain field. Since gc never frees a reachable
object, a gc program never reaches the "read of a freed object" case. *(My call: a disappearing link
under gc would add a case where gc stops and JavaScript does not. The field only differs from own in
the case where own stops.)*

### A read of a freed object (own)

A read of a `@weak` field whose object is a tombstone writes, to standard error:

```
error: the @weak field 'Node.parent' was read after its object was freed
```

and ends the program with a non-zero exit code, without running `finally` blocks or other cleanup.
It is not an exception and cannot be caught. *(My call: a catchable throw would make every `@weak`
read a call that may throw, which changes how own frees values in every function that reads one, and
the case is a bug in the program, not a condition to handle. Swift's `unowned` traps the same way.)*
The exact wording is the builder's, in the style of the run-time's other fatal messages, but it names
the class and the field.

### What a read gives (own)

A read `x.f` is the field's `deref()`, checked as above. Its value follows spec 84's rule for a
`deref()` result: a borrow with an unknown place.

- It may be used: fields read, methods called on it, passed to calls that do not keep it, compared,
  narrowed (`if (this.parent)`). Until anything that could free an object of its class, as for
  `deref()`.
- Each read is checked again. So `this.parent.v; drop(); this.parent.v` compiles: the second read is a
  new borrow, checked at run time. Only a value held across something that may free it is refused:
  `const p = this.parent; drop(); p.v` is the spec 84 error.
- It may not be the receiver or an argument of a call that may drop (no `__own_no_drops` on the
  callee, a virtual call whose overrides are not all known to drop nothing, a call through a function
  value). The callee could free the object through its owner while running on it, and the per-read
  check cannot see that. *(User decision.)* This is stricter than spec 84's rule for a `deref()`
  result, which refuses only uses after such a call. `this.parent.depth()` compiles when `depth`
  frees nothing; otherwise the message says so and suggests `@nodrops` on the method or `@final` on
  the class: "'parent' is a @weak field and 'removeChild' may free objects; the call could free the
  object it runs on".
- It may not be stored in an owning place, returned, captured or given to a call that keeps it. The
  messages follow spec 84's, naming the field: "'p' borrows the object the @weak field 'parent'
  refers to and cannot be returned". To keep it, write it to another `@weak` field, or make a
  `new WeakRef(this.parent)`.

### Comparisons and null tests (own)

- `===` / `!==` against an object, `null` or `undefined` compare the stored pointer and never stop.
  A tombstone's block stays allocated until its last weak reference goes, so its address cannot be
  reused by another object, and `if (c.parent === this) c.parent = null` is always safe.
  *(My call, builder check.)*
- A truthiness test (`if (c.parent)`, `c.parent ?? d`, `c.parent?.v`) is a read: on a tombstone it
  stops like any other read. A tombstone is not null, and the program learns that it kept a stale
  link. *(My call: it keeps the user's "stops instead of giving undefined" decision. A program that
  wants to clear a link without reading it compares with `===` or assigns `null`.)*

### Seen through an interface or object type (own)

A class whose `@weak` field would be read through an interface, an object type or a structural cast
(`<{ parent: Node }>n`, an object type parameter) would read and write the raw storage with no check
and no count. So a class with a `@weak` field cannot be converted to an interface or object type that
has a member of the same name: "'Node' cannot be seen as 'HasParent': its field 'parent' is @weak".
An `any` round trip back to the class itself is fine, since only the class reads the field.
*(My call, builder check: an accessor cell in the interface vtable would allow it, at a much larger
cost. It can come later.)*

### What a write takes (own)

- A write takes any value that has an owner elsewhere: `this`, a borrow, a parameter, a read of another
  field, a read of another `@weak` field. It does not move it. It raises the object's weak count and
  lowers the old value's.
- A write of a value nobody else owns (a `new C()`, a call that returns a fresh object, a value moved
  out of its owner) is an error: "the @weak field 'parent' cannot own a new object; it would be freed at
  once". *(My call: storing it would free the object straight away and leave a tombstone. That compiles
  but is always a bug.)*
- An initializer (`= null`) and a write of `null`/`undefined` are ordinary writes.
- A `@weak` field is no owner: releasing its holder lowers a count and frees at most a tombstone,
  never a live object. Like a `WeakRef` field, that release is not a drop
  (`__own_no_drops` holds, no borrow ends).

### Weak targets, threads, stack promotion

The class of every `@weak` field is a weak target exactly as if the module had a `WeakRef` of it. Its
objects are not stack-promoted (every model, `"weak-target"`) and under own may not cross threads.
Nothing new is needed beyond counting `@weak` fields where `WeakRef<T>`s are counted.

### Across modules

The field's representation under own is `!ts.weak_ref<C>`, so a class with `@weak` fields compiled in
one module and used in another reads the same layout. A class's `@weak` marks travel with the class
declaration (the exported declaration text and the class info), so the importing module checks reads
and writes too.

## Tests to write

Each test is an own test unless noted, and the runtime ones run under own, gc and none (and own
`-noopt`), printing `done.` last.

- `00weak_field`: the `Node` tree above built, `depth()` from a leaf, a child moved to another parent,
  a child removed with `pop()`, a subtree dropped while its children's `parent` fields remain (no read
  of them afterwards). Churn between steps so that a freed block read by mistake shows. Same output in
  all three models.
- `own_weak_field_freed_read`: a child kept after its parent is dropped, then `child.parent.v` read.
  Under own: the message names `Node.parent`, the exit code is non-zero, nothing after it prints.
  Under gc and none: the program prints the value (gc kept the parent; none never frees).
- `own_weak_field_compare_dead`: after the parent is freed, `child.parent === p`, `!== null` and
  `child.parent = null` run without stopping; then the program prints `done.`.
- Leak and double-free checks: an ahead-of-time churn loop of re-parenting with flat memory; the own
  tests on Linux under `MALLOC_PERTURB_`; a tombstone freed when the last `@weak` field pointing at it
  goes.
- Errors (`own_err_weak_field_*`): a read as the receiver of a method that may drop, and as an
  argument of such a call (a non-`@final` class with an override that drops); a class with a `@weak` field converted to an
  interface naming the field; returning a read; storing a read into an owning field; capturing
  one; holding one across a call that may drop; writing a fresh object; `@weak` on a local, a
  parameter property, a static field, an interface member; `@weak` on an array, string, `@value`,
  `Shared<T>` or a two-class union field; `@weak p: T` instantiated with a string; `@weak` under rc;
  a weak-target object captured by `Thread.spawn`.
- Across modules: a class with a `@weak` field declared in one module, read and written in another,
  including the freed-read stop.

## Left to the builder

- Where the field is turned into `!ts.weak_ref<C>` and where reads and writes become
  `WeakRefDeref` + check and `WeakRefNew`/retain, and whether the check is a new op, a flag on
  `WeakRefDeref`, or a call into the runtime.
- How the field's declared type (`C | null`) and its storage type (`!ts.weak_ref<C>`, nullable) meet
  in the class's field list, and how the export text carries `@weak`.
- The exact text of the messages.
- Documentation: a section in the own spec (`tslang/docs/superpowers/specs/2026-09-24-own-memory-model-design.md`),
  the user guide (`docs/own-memory-model.md`), `docs/memory-models.md` and the readiness page.
