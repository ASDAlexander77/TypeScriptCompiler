# `@value` classes and interfaces

Date: 2026-10-24. Status: design, awaiting approval.

## Why

The ray tracer benchmark (`benchmarks/raytrace`) runs 8x slower than idiomatic Rust (own 444 ms,
none 434 ms with PR #683, Rust 54 ms). The user's goal is 1.1x of Rust, about 60 ms. Almost all of
the gap is heap allocation of short-lived `Vector`, `Color`, `Ray` and `Intersection` objects:
Rust's port allocating the same objects behind `Rc` takes 490 ms. Stack promotion cannot remove
them: the `Vector`/`Color` results are "returned", and the `Ray`/`Intersection` literals are
"converted" (boxed when they become interface values, `castTupleToInterface`).

TypeScript has no way to say "this is a plain value". The user's ruling: the benchmark may opt in
with a `@value` marker, and a `@value` type behaves like a value object. Its fields are readonly
and it has no identity, so copying it is invisible, and a program means the same as it does in
ordinary TypeScript. Only speed and memory change.

## Goals

- `@value` on a class declaration: the class's instances are values (a struct of its fields), not
  heap objects. `new C(...)` makes a value; passing, returning, assigning and storing copy it.
- `@value` on an interface declaration: the interface type is a value of its fields, so an object
  literal of that type stays the stack tuple tslang already builds for it, with no box and no
  vtable.
- The benchmark keeps its object literals exactly as written. `rtbench.ts` and `rtbench_own.ts`
  mark `Vector` and `Color` as `@value` classes and `Ray` and `Intersection` as `@value`
  interfaces (see "Benchmark edits").
- Every memory model (gc, rc, own, none) compiles the marked benchmark and prints the same checksum
  (38837369). Own included: if own rejects a `@value` value holding a borrow, fixing the ownership
  passes is part of this project (the user's ruling).
- Target: the marked benchmark runs within 1.1x of Rust idiomatic, about 60 ms, under own and
  none. rc and gc are measured and reported too. The project is done when `@value` works and is
  measured, whether or not it reaches 60 ms: the README then says where the remaining time goes,
  and closing the rest is a separate project (the user's ruling).

## Non-goals

- Inferring value-ness for unmarked classes or interfaces (escape analysis, stack promotion v2).
- Generic `@value` classes, `@value` classes with `extends` or `implements`, and converting a
  `@value` class value to `any`, `unknown`, `Object`, a union with non-value members, or a
  non-`@value` interface. Each is a compile error in this version (boxing them is a follow-up).
- Removing the remaining indirect calls: `Thing.intersect`/`normal` through the interface, the
  `Surface` function fields, the `addLight` closure and the virtual `RayTracer` methods stay as
  they are. The first three are indirect in Rust too.
- Fixing the existing handling of `null` assigned to a plain type-literal tuple (see "Found on the
  way").

## Anti-goals

- A program whose output differs from ordinary TypeScript. Where a `@value` value would let the
  difference show (identity, mutation, `instanceof`), the compiler rejects the program instead.
- Changing anything for code without `@value`.
- Making `@value` a hint the compiler may ignore. Marked types are always values; anything that
  cannot be compiled as a value is an error that names the reason.

## Design

### The marker

- `#define VALUE_DECORATOR "value"` in `Defines.h`, beside `FINAL_DECORATOR`.
- Classes: `ClassInfo` (`MLIRGenStore.h`) gets `bool isValue`, set in the class decorator loop in
  `MLIRGenClasses.cpp` (about 382-420, where `@cpp` and `@final` are read), on declarations as
  well as definitions.
- Interfaces: `InterfaceInfo` (`MLIRGenStore.h` about 283-315) gets `bool isValue`. Interfaces read
  no decorators today; `mlirGenInterfaceInfo` (`MLIRGenInterfaces.cpp` about 1118) reads them with
  `iterateDecorators`, as classes do. The parser already accepts decorators on interfaces.
- `DeclarationPrinter` prints `@value` before both, so `-shared` importers (`__decls`) see it.
- `@value` with `@cpp`, `abstract`, `@final` (implied anyway), `extends`, `implements` or type
  parameters is an error.

`@value interface` is not valid TypeScript for `tsc`, which rejects decorators on interfaces. That
is accepted: the benchmark already depends on tslang-only declarations.

### One marked type for both

Both kinds of `@value` declaration get one new type, `ValueType`
(`!ts.value<@Name, !ts.tuple<...>>`): the declaration's name and its fields as a tuple in
declaration order. It lowers to the tuple's LLVM struct. The marker is what the rules below key on
(`T | null` as an optional, the `===` error, the declaration printer), so plain tuples and type
literals keep exactly their current behaviour, including the bug under "Found on the way".
*(Designer's call, after the builder check.)*

- A `@value` class's type is nominal: only that class's values convert to it.
- A `@value` interface's type is structural, as TypeScript says: a tuple, an object literal, a
  type-literal value or another `@value` interface value with the same fields converts to it by
  copying the fields into its order, and it converts to a type literal with the same fields the
  same way.

### `@value` interfaces

A `@value` interface value is laid out as its type literal already is: the tuple of its fields in
declaration order (`type Ray = { start: Vector; dir: Vector }` is passed by value today). So:

- A `@value` interface may have only fields, all `readonly`, none optional (`?`). Methods, index
  signatures, call signatures and `extends` of a non-`@value` interface are errors.
- An object literal whose contextual type is a `@value` interface is built as the tuple in the
  interface's field order (the existing reordering in `mlirGen(ObjectLiteralExpression)`), with no
  `castTupleToInterface`. A literal with extra or missing fields is the usual type error.
- `x.f` is a field of the tuple: `PropertyRef`/`ExtractProperty`, not `InterfaceSymbolRef`.
- Converting to a non-`@value` interface, `any` or `Object` boxes a copy, through the existing
  tuple-to-interface and tuple-to-any paths. Readonly fields and no identity make the copy
  invisible.
### `@value` classes

A `@value` class's `ValueType` carries the class's name, for member lookup and type checks, and its
instance fields in declaration order (parameter properties included). A plain `TupleType` is not
enough: two `@value` classes with the same fields would be one type, and a method call on a value
would not know its class.

- Fields: every instance field is `readonly`; a field without it is an error. Field reads are tuple
  field reads.
- `new C(args)` reserves a stack slot for the tuple, runs the constructor with `this` pointing at
  the slot, and loads the value. In the constructor `this` may be read and its fields assigned, as
  `readonly` allows. `this` there is the slot being filled, not a value yet, so it may only be the
  object of a member access: `this.f` reads, `this.f = x` and `this.f++` writes, `this.m()` (the
  method sees the fields stored so far), `this["f"]`, in the constructor or in an arrow function
  inside it (a closure that captures `this` may only be called there: "a closure that captures
  'this' of the @value class 'V' can only be called inside its constructor"). Any use
  of `this` as a whole value - `const self = this`, `let s = this`, `f(this)`, `arr.push(this)`,
  `return this`, `[this]`, `{ v: this }`, `...this`, `() => this` - is an error: "'this' of the
  @value class 'V' is not a value until its constructor returns; read 'this.f'". A copy taken
  there would miss the fields stored after it, where TypeScript's object has them. An instance
  field's initializer runs in the constructor (in the one made for it when none is written), so the
  same rule holds there: `readonly b = show(this)` is the error, `readonly b = this.a * 2` is not.
- Instance methods and getters take `this` as a value (or a pointer to a caller copy: builder's
  choice) and are called directly; there is no vtable and no `.instanceOf`. Setters are an error.
- Static members are unchanged. A static field of the class's own type (`static white = new
  Color(...)`) is a global holding the value.
- `instanceof C` is an error (no type information at run time).
- The class may not be extended, and may not extend or implement anything (non-goals).

### Rules for both

- `===`, `!==`, `==`, `!=` between two `@value` values is an error ("a @value type has no identity;
  compare its fields"). Comparing with `null` or `undefined` is allowed (see "Missing values").
- `typeof v` is `"object"`.
- Arrays (`Vector[]`) hold the values inline, as arrays of tuples do today.
- A generic function or class instantiated with a `@value` type works on copies, as it does with a
  tuple. The benchmark needs this: `scene.lights.reduce(addLight, ...)` instantiates
  `__array_reduce` with `R = Color`. Only a `@value` class with type parameters of its own is a
  non-goal.
- A `@value` value may hold references (`Intersection.thing: Thing`). Copying it copies them as a
  tuple copy does today: rc retains each owning field (`buildRetainBody` falls through to
  `retainFields`), gc scans the stack, and none does nothing. Under own, a value holding a borrowed
  reference is a record (own spec 32, the tuple rules of #679). Today's own support for the ray
  tracer rests on borrowing records that live in a box the caller owns (ruling 32 in
  `.superpowers/own-raytrace-rulings.md`); with `@value` there is no box. If own rejects
  `{ thing: this, ... }` returned through the `Thing` interface, or `closestInter` reassigned in
  the loop (ruling 10), the builder fixes the ownership passes properly, with a minimal repro and
  an own test for each imprecision. A workaround that bends a pass only for these shapes is not a
  fix. If a rejection turns out to be correct (the code really is unsafe), it is reported and the
  benchmark change for it is decided separately.
- Stack promotion and the escape facts need no new reason codes: there is no allocation to
  report.
- A `@value` type cannot contain itself, directly or through other `@value` types: it is the tuple
  of its fields, held in place, so `next: L | null` (or `A.b: B | null` with `B.a: A`) would make it
  infinitely large. The error names the type and the fields from it back to itself: "@value type 'L'
  contains itself through 'next'; a value cannot hold itself", "@value type 'A' contains itself
  through 'b.a'; ..." (each type of a longer cycle is reported, from its own field). An array of
  itself in a field (`next: L[]`) or a function type naming it is the same error: the field's type is
  made from the `ValueType` being made, which names its whole tuple, so it cannot refer to itself.
  An ordinary class or interface between them (`box: Box` with `Box.l: L | null`) is a reference,
  and works. A generic one specialized with the type itself (`box: Box<L>`) is not: the specialized
  class's type is made from L's, so it is "@value type 'L' reaches itself through 'box' in the type
  arguments of 'Box<L>'; a @value type cannot hold a generic type made from it". (The unspecialized
  `Box` was taken in its place before, and the compiler crashed at a literal of L.) The same holds
  through another @value type (`A.p: Box<B>` with `B.a: A`). A generic of another @value type
  declared later (`box: Box<K>`) is typed once K is laid out, wherever the generic is declared;
  while a @value type's field is typed, a failed instantiation is never replaced by the
  unspecialized generic, so `Box<Nope>` is "can't find type by name: Nope".

### Missing values

TypeScript lets `null` and `undefined` stand for an `Intersection`. A value has no null pointer.

- Every way of writing that a `@value` type `T` may be missing - `T | null`, `T | undefined`,
  `T | null | undefined`, an optional parameter `v?: T`, an optional field `f?: T` - is one
  encoding with three states: `OptionalType<UnionType<T, null>>`
  (`MLIRTypeHelper::getMissingValueType`). undefined is the empty optional, null is the union's
  null member, and a value is the union holding `T`. Converting between the spellings changes
  nothing, so null stays null and undefined stays undefined, and converting a plain `T` to one
  makes a present value.
- As emitted, that is a tagged union inside an optional. `isUnionTypeNeedsTag` answers yes for
  the union (`isMissingValueUnion`: one `ValueType` member and `null`), where merging its members
  would give the optional back, so it is laid out as any tagged union is: a packed
  `<{ ptr, payload }>`. The tag is a pointer to the member's name in a type-descriptor constant
  (`@td_<n>_object` for the value, since `typeof` of a value is `"object"`; `@td_<n>_null` for
  null), pointing past the descriptor's 32-byte header at the name string. The payload is an
  array of `i64` as large as the largest member (`<{ [1 x i64] }>` for `{ dist: number }`), and
  holds the value's storage tuple. The optional adds its has-value `i1`, so `Hit | null` of
  `@value interface Hit { readonly dist: number }` is passed and returned as
  `{ <{ ptr, <{ [1 x i64] }> }>, i1 }`. undefined is all zeroes (`i1` false); null is `i1` true
  with the `null` descriptor's tag and a zero payload; a value is `i1` true with the
  `object` tag and the tuple. `=== undefined` reads the `i1`; `=== null` and `== null` read it and
  then compare the tag with the string `"null"` (`strcmp` when the tag is not a null pointer).
  This is the cost the benchmark README measures for `Intersection | null`.
- Strict comparisons with `null` and `undefined` are exact, as in TypeScript: `x === undefined` is
  false for `null`, and `x === null` false for `undefined`. Loose `x == null` and `x != null` ask
  whether `x` holds a value. Comparing one with anything else (another value, a number) is the
  identity error above.
- Reading a field of a missing value, calling a method or getter on it, or converting it to a
  plain `T` throws at run time, as a method of a null interface does (the same throwing stub,
  with its own message).
- Under `// @strict-null false`, a conversion such as `const h: Hit = find(-1)` therefore throws at
  the conversion, where TypeScript would throw only at a later read of `h`. This divergence is
  accepted (controller's ruling).
- Assigning, returning or initializing `null`/`undefined` into a plain `T` is an error, even in a
  `// @strict-null false` file: "a @value type has no null; write `T | null`". This is the user's
  ruling: it costs four type annotations in the benchmark and keeps every other `@value` slot free
  of a flag. In a non-strict file `T | null` is written out for the compiler's sake; TypeScript
  reads it as `T`, so the program means the same.
- The optional storage applies only to `@value` types (`ValueType`). How unions of plain tuples
  with `null` are built today (`getUnionTypeMergeTypes`) does not change.

### Benchmark edits

Both `rtbench.ts` and `rtbench_own.ts`, comments updated. No object literal changes.

- `@value class Vector`, `@value class Color`, their constructor parameters `public readonly`.
- `@value interface Ray`, `@value interface Intersection`, fields `readonly`.
- `Thing.intersect`, `Sphere.intersect`, `Plane.intersect`: `Intersection | null`.
- `RayTracer.intersections`: `let closestInter: Intersection | undefined = undefined`.
- `rtbench_own.ts`: `Plane.normal` returns `this.norm` again, since returning a value copies it.
- `Color.toDrawingColor` and `Camera`, `Light`, `Scene` are unchanged.

The Rust and C++ ports are unchanged. README gains a results section with and without the markers.

## Testing

- `test/tester` positive tests, each AOT and JIT, under every model where the suite runs them:
  - a `@value` class with a constructor, static helpers, an instance method, a getter and a static
    field of its own type; values passed, returned, stored in a class field and an array;
  - a `@value` interface built from literals in a different field order, read, passed and nested
    in another `@value` interface and in a `@value` class;
  - `T | null` / `T | undefined` returned and compared, and a field read after a check;
  - a `@value` value holding a class reference, copied many times under rc with a `churn()` of
    same-size objects (the use-after-free rule from the rc work);
  - a `@value` interface value converted to an ordinary interface and to `any`, then read.
- Error tests: a non-`readonly` field, a setter, `extends`/`implements`, generic, `instanceof`,
  `===` between values, `null` into a plain `@value` type, a method or `?` field in a `@value`
  interface, `@value @cpp`, converting a `@value` class to `any`.
- An IR check that a function taking and returning `@value` values and building a literal of a
  `@value` interface has no allocation call in its optimized LLVM IR (like
  `fresh-objects-folded.cmake`).
- A `-shared` pair: an exporting module with a `@value` class and a `@value` interface, used from
  an importer.
- The full suite stays green (5525 on Windows Release with PR #683).
- `bench.ps1`: the checksum under all four models, and the timings.

## Found on the way

On main, `return null` from a function returning a type-literal tuple compiles with
`warning: invalid cast from '!llvm.ptr' to '!llvm.struct<(ptr, ptr)>'`, and the result compares
`false` with both `null` and `undefined`; a tuple local initialized with `undefined` compares
`false` with `undefined`. That is a bug in plain TypeScript code, separate from this project.

## Left to the builder

- How `ValueType` is threaded through MLIRGen: where a class reference resolves to it, and
  which class paths (`NewClassInstance`, `ClassMembersAccess`, field access) branch on it.
- Passing `this` to instance methods by value or by pointer to a copy, and how large values are
  passed in calls (the LLVM ABI decides registers versus memory).
- The encoding of `T | null | undefined`.
- Error message wording.
- Branch layout: on main, or stacked on `process-heap-malloc-attrs` (PR #683) so the benchmark
  numbers include it.
