# `-mm=own` compiles the ray tracer

Date: 2026-10-21. Status: design, approved decisions marked **USER**.

## Why

The ray tracer benchmark (`tslang/benchmarks/raytrace`, PR #634) runs under gc, rc and none, and
is the program the README quotes for performance. Under `-mm=own` it does not compile: 44 errors.
`own` is meant to accept ordinary object-oriented TypeScript that has a single owner for each
value, and this program almost does. Its few genuinely shared objects are the two surfaces.

**USER:** small deliberate source edits like `Shared<T>` for genuinely shared objects are fine; every
other error is a compiler gap and gets fixed in the compiler. `own` gets its own copy of the
benchmark with the edit; gc, rc and none keep running the unedited source.

## Goals

- `rtbench_own.ts` (the benchmark with the two source edits below, nothing else) compiles
  under `-mm=own --opt --opt_level=3 --no-default-lib`, prints checksum 38837369, and frees what it
  allocates (peak working set comparable to rc's 4 MB).
- Each gap below is fixed as a general rule, with minimal positive tests that run under gc, rc and
  own, teeth (a deliberately broken variant that frees too early must fail), and negative tests for
  the cases that must stay errors.
- The benchmark table gains an own row.

## Non-goals

- Making the unedited `rtbench.ts` compile under own. The surfaces are shared; that is `Shared<T>`.
- Stack promotion of the ray tracer's temporaries (a separate item: caller-provided result slot).
- Performance work beyond what falls out.
- Annotations. Everything is inferred, as `__own_result_borrows` is today.

## Anti-goals

- **No unsound acceptance.** A rule that accepts a program and then frees a block someone still
  reads, or frees it twice, is worse than the error it removes. When unsure, keep the error.
- No runtime counting under own outside `Shared<T>`.
- No change in what gc, rc or none compile or print, except where a fix in shared MLIRGen code is
  also a fix there (say so in the PR).

## The source edits (the only two)

`rtbench_own.ts` differs from `rtbench.ts` by two edits.

**1. `Plane.normal` returns a fresh copy (USER).** `Plane.normal` returned `this.norm`, a borrow of
its own field, while `Sphere.normal` returns a fresh vector. Through the `Thing` interface a caller
cannot know whether it owns the result, and a run-time flag is ruled out. So:

```ts
public normal(pos: Vector): Vector { return new Vector(this.norm.x, this.norm.y, this.norm.z); }
```

The compiler does not insert such copies itself: for class objects a copy changes identity.

**2. The surfaces are `Shared<Surface>`:**

```ts
interface Thing { ...; surface: Shared<Surface>; }
class Sphere { constructor(public center: Vector, radius: number, public surface: Shared<Surface>) ... }
class Plane  { constructor(public norm: Vector, public offset: number, public surface: Shared<Surface>) ... }
module Surfaces {
    export let shiny = new Shared<Surface>({ ... });
    export let checkerboard = new Shared<Surface>({ ... });
}
// and every thing.surface.X becomes thing.surface.value.X
```

With both edits it compiles and prints the right checksum under rc today.

## The gaps

Each has a minimal repro (kept as the test). "Today" is main 49b5f6bb. Error texts are the
current ones.

### G1. Immortal globals (USER decision)

A global that is never reassigned after its initializer is **immortal**: like a string literal, a
release skips it, so a read of it may be returned, stored into a field, element or another global,
captured, or given to a call that keeps it, the way Rust hands out a `'static`.

```ts
class K { static w = new V(1); static d = K.w; static f() { return K.w; } }
function g(n: number) { return n > 1 ? K.w : new V(n); }     // mixed: immortal or fresh
```

Today: `'this value' borrows the global 'K.w' and cannot be stored, returned or captured`.

- **Never reassigned** (decided here, correct me): a class static or module-level `let`/`const`
  whose only store is its initializer, in this module. An exported global stays mortal unless the
  module is the program's entry and nothing imports it - other modules may assign it. (In the ray
  tracer, `Color.*` statics are not exported; `Surfaces.*` are `Shared` anyway.)
- **Which values** (builder check, my call): only a global whose initializer is a fresh value
  (`new C()`, a fresh call result, an owning literal) or another immortal global. A borrowed
  initializer (`static x = arr[0]`, a call that returns a borrow) stays mortal and keeps today's
  error: marking a block someone else owns immortal would turn its later free into a silent leak.
- **Mechanism:** the existing `HEAP_BLOCK_IMMORTAL` header (OwnershipRoutineLogic.h): the global's
  initializer marks the block it stores immortal; inference treats a load of such a global as it
  treats `isImmortalLiteral` - owns nothing, may go anywhere, its release erased or skipped.
- **A mixed return** (`n > 1 ? K.w : new V(n)`) is then an owned result on every path: the caller
  "owns" the immortal one too, and its release is a no-op at run time. Today's error
  `returns a borrow of its argument on some paths and another value on others` must not fire for an
  immortal path.
- **Only the global's own value is immortal.** Its fields are not: `K.w.child = x` still frees
  the old child, so a borrow of `K.w.child` stays bounded as today.
- Under gc/rc/none nothing changes.

### G2. Borrowing records (USER decision)

A function that returns an object literal built from borrowed values returns a **borrow** of
those sources (inferred, like a Rust struct with a lifetime):

```ts
interface I { t: T; d: number; }
class T { k = 1; hit(): I { return { t: this, d: 2 }; } }   // result borrows `this`
```

Today: `an object literal that holds a value it owns is moved only into a local or returned as it
is` and `argument 'this value' is given to 'T.hit', which keeps it`.

- The result is `__own_result_borrows` of the receiver/parameters it holds (directly, or a field
  read of one). Several sources (ray tracer: `{ thing: this, ray: ray, dist }` borrows `this` and
  `ray`) means several - today's fact names one argument; it becomes a set.
- **Mixed fields:** a record may hold owned fields (`new V()`) and borrowed ones. Its release frees
  the owned fields and leaves the borrowed ones (a per-record routine skipping borrowed fields, as
  `ts.own_borrowing_fields` does for generator states in phase 7b).
- **The release travels with the value** (builder check, my call): which fields a record's release
  skips is decided where the record is made, not from the static type where it is released. Then
  records from different creation sites (`Sphere.intersect` and `Plane.intersect`) may meet in one
  local, as `closestInter` does.
- **Flow:** borrow-ness flows through returns, merges and locals as for plain borrows today:
  `intersections` keeps the best of several such records in a local and returns it - its result
  borrows its own `ray` and `scene`.
- **Stays an error:** storing a borrowing record where it can outlive a source (field, global,
  element of a longer-lived array, closure that escapes).
- Across modules: an open (exported) function has to say it with `@lends`, which today names one
  parameter; it may name several. (The ray tracer is one module, so this can come last.)

### G3. Owning literals moved anywhere

An object literal that owns its fields (fresh values, or locals moved in) may move into a call
argument, an array literal element, a nested literal, or be returned through an interface type:

```ts
len({ start: p, dir: d });                         // p and d move in; the callee borrows the record
const ls: L[] = [{ pos: new V(1) }, { pos: new V(2) }];
return { ls: [new V(1), new V(2)], c: new V(3) };
```

Today: `an object literal that holds a value it owns is moved only into a local or returned as it
is; -mm=own cannot move it here yet` (13 of the 44).

- This includes a literal converted to an interface type even when it goes into a local:
  `const r: Ray = { start: new V(1), dir: new V(2) }` fails today with the same error.
- An argument record is a temporary: the caller owns it and releases it after the call, as it does
  a fresh `new C()` argument. Whether the callee borrows or keeps it comes from the callee's facts,
  as for any argument.
- When a field is a local moved in (`{ start: p }`), `p` is moved (use after = error, G7 applies).
  When it is a borrow (`{ start: this.pos }`), the record is a borrowing record (G2) bounded by the
  source, and as a call-argument temporary that is always fine.

### G4. `Shared<I>` of an object literal

```ts
let g = new Shared<S>({ f: function (x: number) { return x; }, roughness: 2 });   // global or local
```

Today: global `the initializer of global 'g' takes a second reference`; local `borrows a field
but is used here after it may be released or overwritten`. `new Shared<C>(new C())` already works:
the literal must move into the handle the same way.

### G5. `reduce` takes its accumulator

```ts
const r = a.reduce((acc: V, v: V) => V.plus(acc, v), new V(0));
```

Today: `is a parameter, kept here, but -mm=own cannot make the callers of '___array_reduce<...>' give
it up` and `borrows an element but is used here after it may be released`. The initial value moves
in, each callback result replaces the accumulator (the previous one released unless the callback
returned it - `addLight` returns `col` on one path), elements are borrowed by the callback, and the
final accumulator is returned owned. Under `--no-default-lib` this is the generated
`___array_reduce`.

- **A callback that returns its accumulator on some paths** (builder check, my call): the
  callback's accumulator parameter is inferred as owned (kept), like a Rust `fn(acc: Color) ->
  Color` by value. Returning it moves it out; a path that returns something else releases it. All
  static: no run-time pointer comparison (anti-goal: no run-time bookkeeping). The same rule serves
  any function that returns a parameter on some paths and a fresh value on others, when every
  caller can give the argument up.
- **The default library's `reduce`** is a follow-up, not this series: the benchmark builds with
  `--no-default-lib`, and stating "the callback's result replaces the accumulator" in `lib.d.ts`
  needs a marker that does not exist yet.

### G6. A field read into a `let`

```ts
function f(r: R) { let d = r.dir; let p = V.plus(d, d); return V.plus(p, d); }
```

Today: `takes a second reference; -mm=own cannot prove a move or a borrow here yet`. `d` is a borrow
of a parameter's field, bounded by the parameter; calls that cannot drop it (signature facts) must
not end it.

### G7. Read after a move while the new owner holds it (USER decision)

```ts
if (v.x > m) { best = v; m = v.x; }
```

Today: `is used here after its value was moved`. The read is a borrow through the new owner: sound
while that owner is not overwritten, released or moved on between the move and the read (and the
value is not mutated through a path that would free it). Otherwise it stays the error.

### G9. Facts through a module-private interface (builder check)

`OwnershipSignaturePass` treats a call through an interface as open: it does not name what it
reaches, so it gets no facts. The ray tracer's hot calls all go through interfaces
(`scene.things[i].intersect(ray)`, `isect.thing.normal(pos)`, `surface.value.diffuse(pos)`), so
G2's borrowing result on `Sphere.intersect` would never reach `intersections`.

- **Closed world for interfaces** (my call): when an interface is not exported and every class or
  object literal that implements it is in this module (and none is exported as that interface), a
  call through it gets the meet of its implementations' facts, as a private virtual family already
  does.
- If the implementations disagree in a way the meet cannot express, the call stays open and the
  existing errors stand.
- Its own PR, before G2.

### G8. Not yet reproduced in isolation

The ray tracer also has `borrows a field but is used here after it may be released or overwritten`
at `Camera`'s constructor, `Sphere.normal` and `render`'s `getPoint` (`this.pos`, `this.center`,
`camera.forward`), one `moved here on some paths only` on `shade`'s ternary, and one `takes a
second reference` at `let d = isect.ray.dir`. Minimal versions of each compile: the `Camera`
constructor with chained calls over `this.pos`, `normal()` through the `Thing` interface, the
`getPoint` closure reading `camera.forward`, and the `depth >= maxDepth ? … : this.call()` ternary
(probes h1-h4). `let d = i.ray.dir` through two interface fields reproduces, and is G6. The rest
are therefore expected to be follow-ons of G1-G7 inside the full program. Each is
re-checked after G1-G7 land; any that remains gets its own repro and fix, in this same series.

## Order of work

Each item is one PR, small and squash-merged, with its tests (gc, rc, own), teeth, negatives, a
corpus run (`own` corpus count may only go up) and the full suite on Windows; Linux (WSL) for any
change to what is released, since the Windows heap hides double frees.

1. G1 immortal globals.
2. G4 `Shared<I>` of a literal.
3. G3 owning literals moved anywhere.
4. G6 field read into a let.
5. G7 read after a move.
6. G9 facts through a module-private interface.
7. G2 borrowing records (the largest; needs G3 and G9).
8. G5 `reduce`.
9. G8 leftovers; then `tests/raytrace_own.ts` (the own copy at a test size) under own in the suite,
   and `benchmarks/raytrace/rtbench_own.ts` + an own row in both READMEs (after #634 merges).

## Left to the builder

- Exact attribute names and which pass decides each fact (signature pass vs inference), following
  the existing split: cross-function facts in `OwnershipSignaturePass`, per-function in
  `OwnershipInferencePass`, routines in `OwnershipRoutineLogic.h`.
- Test names and CMake anchors (use distinct anchors per PR - parallel PRs conflicted before).
- Whether a step splits into two PRs.
- The results of each step go in `2026-09-24-own-memory-model-design.md` as a new section, as
  earlier phases did.
