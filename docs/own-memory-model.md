# The `-mm=own` memory model

`-mm=own` frees memory the way Rust does: every heap block has exactly one owner, and the
compiler works out at compile time where each block is freed. There is no collector and no
reference count at run time. When the compiler cannot prove a single owner, it rejects the
program instead of compiling it with a leak or a double free. With `--own-fallback=count` it counts
that one object instead, with a warning ([Counted fallback](#counted-fallback---own-fallbackcount)).

```bash
tslang --emit=exe -mm=own hello.ts
```

There are no lifetime annotations. Inside a module the compiler infers everything. A few
decorators state, in a function's or a class's signature, what another module needs to know
about it, and `Shared<T>` gives opt-in shared ownership.

- [Owners, moves and borrows](#owners-moves-and-borrows)
- [Reading the errors](#reading-the-errors)
- [Strings](#strings)
- [Closures](#closures)
- [Shared ownership: `Shared<T>`](#shared-ownership-sharedt)
- [Counted fallback: `--own-fallback=count`](#counted-fallback---own-fallbackcount)
- [Decorators](#decorators)
  - [`@keeps`](#keeps) - the parameters a function keeps
  - [`@lends`](#lends) - the arguments a result borrows
  - [`@record`](#record) - a result that is a box of borrows
  - [`@borrows`](#borrows) - the arguments a generator borrows
  - [`@immortalizes`](#immortalizes) - the parameters a function keeps for good
  - [`@nodrops`](#nodrops) - a C function that frees nothing
  - [`@final`](#final) - a class nobody extends, a method nobody overrides
  - [`@value`](#value) - a type that is copied, not shared
- [Separate compilation](#separate-compilation)
- [Code for one memory model](#code-for-one-memory-model)
- [Current limits](#current-limits)

Every example below compiles (or fails, where it says so) with
`tslang -mm=own --emit=jit example.ts`, and the output shown is the compiler's.

## Owners, moves and borrows

The owner of a block is a local, a field, an array element or a global. The owner frees the
block when it gives it up: when its scope ends, when it is overwritten, or when the block is
removed from its container (`pop`, `shift`, `splice`, `delete`).

Where a second reference would be made, the compiler decides one of three things:

1. **Move.** The source is not used afterwards, so it gives the block up. "Moved" is a
   compile-time fact; nothing happens at run time.
2. **Borrow.** The new reference owns nothing. It is valid only while the owner keeps the block:
   it may not be used after the owner frees or overwrites it, and it may not be stored in a
   longer-lived place, returned or captured by a closure that escapes.
3. **Error.** Neither is possible.

Three rules follow:

- **Arguments are borrowed.** A function lends its arguments to the function it calls.
- **Results are owned.** The caller owns what a call returns.
- **Containers own.** A store into a field, an element or a global moves the value in: the value
  must be fresh (a `new`, a literal, a call's result) or not used again. A read out of a
  container is a borrow.

```typescript
class Node { constructor(public value: number) {} }
class Holder { node: Node = new Node(0); }

function sum(n: Node, m: Node) { return n.value + m.value; }   // arguments are borrowed
function make(v: number) { return new Node(v); }              // the caller owns the result

function main() {
    const h = new Holder();
    const n = make(1);
    print(sum(n, h.node));      // n and h.node are lent to sum, and still owned here
    const first = h.node;       // a borrow of the field: first owns nothing
    print(first.value);
    h.node = n;                 // n moves into h; the old Node is freed here
    const list: Node[] = [];
    list.push(make(2));         // a fresh value moves into the array
    list.push(new Node(3));
    print(list[0].value + list[1].value, h.node.value);
}                               // h, list and what they own are freed here
```

```text
1
0
5 1
```

A `let` that is only given other locals' values, or fields', and is assigned again (`let cur = a;
... cur = b;`) borrows each value until its next assignment: none of them moves into it. Walking a
list from a local head is that case (`let node = head; while (node) { ...; node = node.next; }`).
Giving the head up, or cutting the list, while the walk still reads it is an error.

A parameter a function keeps can go into an object literal it returns (`return { key, prev }`): the
literal owns it, as it owns a new object, and reading the parameter after that is an error. A
parameter given a default by `??=` or `||=` can be copied into a `let` (`let a = <number[]>results`),
which borrows it.

A catch clause owns what was thrown: the value is freed where the clause ends, unless the clause
gives it on (`return e`, `throw e`, storing it), also with a `finally` or inside another `try`.

Inside a module the compiler also infers each function's facts from its body: a function that
stores a parameter keeps it, so its callers have to move the argument in; a function that
returns a field of its argument returns a borrow, so its callers may only borrow the result.
Callers in other modules cannot see the body, which is what the [decorators](#decorators) are for.

## Assigned parameters and `delete`

A parameter is lent by the caller. A function that assigns one keeps it instead: the parameter
then owns each value it is given, so the caller moves the argument in and may not use it after
the call.

```typescript
class B { constructor(public b: number) {} }

function pick(x: B, fresh: boolean) {
    if (fresh) {
        x = new B(7);       // pick assigns x, so it keeps it
    }

    return x.b;
}

function main() {
    print(pick(new B(1), true));    // a new object moves in

    const b = new B(2);
    print(pick(b, false));          // b moves in
    // print(b.b);                  // error: 'b' is used here after its value was moved
}
```

An exported function, or a method reached through an interface, says so in its signature with
[`@keeps("x")`](#keeps). A parameter captured or assigned by a closure, one a generator assigns,
and `x ??= value` are not accepted yet.

`delete` ends an owner. `delete c` of a local or a field that owns its value frees the value now
and leaves null there, so reading it afterwards fails as a null read does. If the value was moved
out before (`h.child = c; delete c`), nothing is freed. A `const` holding a new object is freed
where its block ends. `delete` of a parameter, or of anything else the function does not own, is
an error:

```text
error: 'delete' of a value this function does not own: under -mm=own only a local or a field that owns its value can be deleted
```

## Reading the errors

A value is named `'this value'` in the messages below; the position points at it.

**Used after a move.** The `Node` moved into `h.node`, so `n` no longer owns it:

```typescript
class Node { constructor(public value: number) {} }
class Holder { node: Node; }

function main() {
    const h = new Holder();
    const g = new Holder();
    const n = new Node(1);
    h.node = n;     // the Node moves into h
    g.node = n;     // a second owner: compile error
    print(h.node.value);
}
```

```text
e1_move.ts:9:5: error: 'this value' is used here after its value was moved
    g.node = n;     // a second owner: compile error
    ^
e1_move.ts:8:5: note: value moved here
    h.node = n;     // the Node moves into h
    ^
```

The next two use the second example's classes, with `node` initialized.

**A borrow stored.** `h.node` is still `h`'s, so it cannot become `g`'s too:

```typescript
function main() {
    const h = new Holder();
    const g = new Holder();
    g.node = h.node;    // a borrow of h.node stored in g: compile error
    print(g.node.value);
}
```

```text
e3_field_borrow.ts:7:5: error: 'this value' borrows a field and cannot be stored, returned or captured
    g.node = h.node;    // a borrow of h.node stored in g: compile error
    ^
```

**A borrow used after its owner let go.** Overwriting `h.node` frees the `Node` that `first`
borrows:

```typescript
function main() {
    const h = new Holder();
    const first = h.node;       // borrows the Node h owns
    h.node = new Node(1);       // frees that Node while first still names it: compile error
    print(first.value);
}
```

```text
e4_overwrite.ts:8:11: error: 'this value' borrows a field but is used here after it may be released or overwritten
    print(first.value);
          ^
e4_overwrite.ts:7:5: note: it may be released or overwritten here
    h.node = new Node(1);       // frees that Node while first still names it: compile error
    ^
```

A call can free things too. A call of a function that may overwrite or free what the caller
borrows ends the borrow, and the note then points at the call. A call of a function the compiler
knows nothing about - a C function, a virtual method another module may override - may free
anything (see [`@nodrops`](#nodrops), [`@final`](#final) and
[Separate compilation](#separate-compilation)).

**A `readonly` field is not freed by a call.** Only the constructor of the class that declares a
`readonly` field writes it, and arrows it calls where they are written
(`(() => { this.a = 1; })()`) - in every memory model, as TypeScript has it: anywhere else
`o.f = v`, `o.f++`, `[o.f] = ...`, `o[k] = v` and `delete o.f` are compile errors. So no call
overwrites one, and a borrow of it lasts as long as its object does. That holds when the object
is a parameter, `this`, or a local the function owns, and down a chain of `readonly` fields
(`o.mid.inner`):

```typescript
class Inner { constructor(public x: number) {} }
class Outer { constructor(readonly inner: Inner) {} }
declare function unknown(o: Outer): void;

function f(o: Outer) {
    const i = o.inner;  // borrows o.inner
    unknown(o);         // may free anything, but cannot write o.inner
    print(i.x);         // compiles; without `readonly` this is the error above
}
```

The borrow still ends where the object does: the local is assigned, moved away, or assigned
by a closure, or the object is a global some call may replace. A `private` field nothing writes
after the constructor counts the same. A subclass that declares the field again without
`readonly` writes the same slot, so the field is then not counted. A class another module may
extend is counted only in a program (`--emit=exe` or the JIT), not in a library.

The usual ways out are the Rust ones: move instead of borrowing (give the value up), store a
fresh value (`new`, a copy), keep the borrow shorter than its owner, mark the field `readonly`,
or share the object with [`Shared<T>`](#shared-ownership-sharedt).

**A warning in place of the error.** With `--own-fallback=count`, some of these errors become a
warning: the compiler keeps a reference count for that one object instead (see
[Counted fallback](#counted-fallback---own-fallbackcount)). The warning names the place, and its
note is the error it replaces:

```text
w.ts:9:5: warning: -mm=own counts this object here (--own-fallback=count); a cycle through it is never freed
    g.node = n;     // a second owner: counted
    ^
w.ts:9:5: note: a value that borrows 'h.node' cannot be stored, returned or captured
```

## Strings

A string is immutable, so a copy cannot be told apart from sharing. Where the compiler cannot
prove a move or a borrow of a string, it copies it instead of reporting an error:

```typescript
class Person { constructor(public name: string) {} }

function main() {
    const p = new Person("Ann" + 1);
    const q = new Person("");
    q.name = p.name;                    // a string is copied, not shared
    p.name = "Bob";
    print(p.name, q.name);
}
```

```text
Bob Ann1
```

An array literal no variable holds (`["k" + i, s][1]`, `[a, b].length`, `f([mk(1)])`, `for (const x
of ["a" + i])`) is freed at the end of its block, or after the call it is given to, with what it owns. A
`let` given a string element of one holds a copy of it, so it can be given another value later, in a
loop too:

```typescript
let st: string = ["s" + i, name][1];   // a copy: the literal is freed at the end of the block
for (let j = 0; j < 3; j++) {
    st = [st + j, "x"][0];             // a copy again
}
```

An object element of such a literal is a borrow of it, as an element of a named array is: it can be
read while the literal lives, and keeping it past that is an error. The same holds for:
- a literal nested in another (`[[mk(i)], [mk(2)]][0][0]`): it moves into the outer one and is freed
  with it;
- a literal given a local's object as well as new ones (`f([mk(i), o])`), or only locals' objects and
  read (`[o].length`): the local's object moves into it (using `o` afterwards is an error) and is freed
  with it. A literal of locals given straight to a call (`f([o])`) still borrows them;
- with `--own-fallback=count`, a literal given a new object and a counted one (`f([mk(i), GA[0]])`):
  freeing it gives the count back;
- a literal whose element a `?:` branch yields (`const o = c ? [mk(i)][0] : null`): it is held past the
  conditional, as a `const` before it would be, and freed at the end of the block the conditional is
  in (the outermost one, when it is inside another's branch), and on the way to a `catch` of the same
  function. `o` borrows it, so it can be read in that block - by another conditional too
  (`d ? o : null`) - but not stored, returned or captured. When the other branch makes a new object
  (`c ? [mk(i)][0] : mk(2)`), the result would own its value on one path and borrow it on the other:
  that is refused, as before.

A string passed to a parameter the callee keeps is not copied, because the callers rely on that
fact: `new Person(p.name)` is an error. Pass a string you give up, or a fresh one
(`new Person(p.name + "")`). String literals are constant data and are never freed or copied.

## Closures

A closure that does not escape borrows what it captures. A closure that escapes - returned,
stored, or given to a function that keeps it - takes its captured variables with it, and the
frame that made it may not use them afterwards.

```typescript
function counter() {
    let n = 0;
    return () => ++n;                   // the closure escapes and takes n's cell with it
}

function main() {
    const values = [1, 2, 3];
    let total = 0;
    values.forEach(v => { total += v; });   // a closure that does not escape borrows
    const next = counter();
    next();
    print(total, next());
}
```

```text
6 2
```

A closure kept - stored into a field, a global or an array, or given to a function that keeps it -
owns the parameters it captures, so the function keeps them and its callers give them up, as for a
parameter stored into a field. A closure stored into a field of the object it captures is a
back-reference instead: it borrows that object.

```typescript
class Holder {
    run: () => number;

    constructor(body: (n: number) => number) {
        this.run = () => body(2);      // the closure is kept, and owns body
    }
}

function main() {
    const h = new Holder((n: number) => n * 21);
    print(h.run());
}
```

```text
42
```

A closure kept for good, in an array that is only ever pushed onto, is covered by
[`@immortalizes`](#immortalizes).

## Shared ownership: `Shared<T>`

A graph that a tree of owners cannot express - two objects that both need a third, a list that
hands out its nodes - uses `Shared<T>`. A `Shared<T>` is a counted handle: copying it makes a
second handle to the same object, and the object is freed when the last handle goes. `.value`
reads or writes the object inside.

```typescript
class Node { constructor(public value: number) {} }
class Holder { node: Shared<Node> | null = null; }

function main() {
    const a = new Shared(new Node(1));
    const b = a;                        // a second handle to the same Node
    const h = new Holder();
    h.node = a;                         // and a third
    b.value.value = 5;                  // .value is the object inside
    print(a.value.value, a === b, Shared.count(a));
    b.value = new Node(7);              // replaces the Node all handles share
    if (h.node) print(h.node.value.value);
}
```

```text
5 true 3
7
```

- `===` and `!==` compare handles: two handles are equal when they hold the same object.
- `Shared.count(s)` is the number of handles. Under `-mm=gc` and `-mm=none`, where `Shared<T>`
  is a plain reference, it is `-1`.
- Only the handles are counted; the rest of the program keeps single owners.
- **Cycles are never freed**, as under `-mm=rc`: two objects whose `Shared` fields point at each
  other keep each other alive. The program still runs correctly.

`Shared<T>` is also the answer to an array method that would give its elements a second owner.
`filter` over class instances keeps each element it selects in the result while the source
still owns it, which is an error; over handles it is not:

```typescript
class Node { constructor(public value: number) {} }

function main() {
    const all = [new Shared(new Node(1)), new Shared(new Node(2))];
    const big = all.filter(n => n.value.value > 1);   // copies of the handles
    print(big.length, Shared.count(big[0]));
}
```

```text
1 2
```

## Counted fallback: `--own-fallback=count`

Where the compiler cannot prove a move or a borrow, `-mm=own` reports an error. With
`--own-fallback=count` it keeps `-mm=rc`'s reference count for the object involved instead, and
says so with a warning. That object is counted from then on: each owner beyond the first adds one
to a count in its header, and the last owner to let go frees it. Everything else in the program
keeps its single owner and pays nothing.

```typescript
class Node { constructor(public value: number) {} }
class Holder { node: Node; }

function main() {
    const h = new Holder();
    const g = new Holder();
    const n = new Node(1);
    h.node = n;     // the Node moves into h
    g.node = n;     // a second owner: counted
    g.node.value = 2;
    print(h.node.value, g.node.value);
}
```

```text
w.ts:9:5: warning: -mm=own counts this object here (--own-fallback=count); a cycle through it is never freed
    g.node = n;     // a second owner: counted
    ^
w.ts:9:5: note: a value that borrows 'h.node' cannot be stored, returned or captured
2 2
```

Without the flag (or with `--own-fallback=error`, the default) the same program is the error in the
note. There is one warning for each place the compiler counts, and no summary. No option turns
these warnings into errors; leave the flag out for that. Like every warning, they are printed when the
file is compiled: `--nowarn` hides them, and a JIT run the JIT cache serves without compiling prints none.

A program `-mm=own` accepts without the flag is decided the same way with it: nothing in it is
counted. A `let` that borrows a field or an element and is then given a new value, which own
repairs (its new values go into a holder of their own), is repaired with the flag too.

**What it covers.** The errors of inference where rc would keep a retain: a second owner, a value
used after it was moved, a borrow stored, returned or captured, an element's borrow stored (a swap
through a local, `let t = a[i]; a[i] = a[j]; a[j] = t`, compiles with two warnings), and a field
read lent to a call that may free it. Values `-mm=own` already copies (strings, `any`, arrays in
`@copies` functions) are still copied, not counted.

**What is still an error.**

- A borrow used after its owner let go where no variable holds a count of its own
  (`const first = h.node; h.node = new Node(1); print(first.value)`), and a borrow of a call's
  result: rc takes that count inside the callee.
- A closure that escapes and whose captured variable is still used.
- A local given a literal and then a second owner.
- An object that may have a field held through a type descriptor: a tagged union (`Node | string`)
  or a closure. Such a field cannot be given a second owner under own.
- A value put into a field of an object literal built by value (`f({ v: n, items: [n] })` after
  the array took `n`): rc counts that field with a retain of the whole literal, which own does not
  keep.
- A value whose store gives back the field's old value with a release another value's decision
  already took (`p.v = p.v; p.v = p.v`), and a captured `const` given a counted value in a loop.
  These are refused at their place, with own's error.
- Everything that is not an inference error: errors that point at `Shared<T>` (`new Map<string,
  Node>()`, `new Array<Node>()`, `filter` over class instances, `reduce` with no initial value),
  a missing `@keeps` or `@lends` on an exported function, parameters assigned or kept where the
  callers cannot give them up, a value read out of a global and lent to a call that may replace the
  global, element access of `any`, and the other errors of this page.

**Every own build honors a count.** A release under `-mm=own` tests the header word against 0
(one owner: free it), then for an immortal block (do nothing), then for extra owners (one fewer),
with or without the flag. A field taken out of a call's result does the same test before it
empties the field. So modules built with the flag, modules built without it and the default library
mix freely: a module that never counts still releases a counted object correctly.

**Threads.** A counted object reached from two threads needs its count changed atomically. As under
rc, the compiler marks what crosses at the points where it is shared: `Shared.share`,
`Thread.spawn` of a closure, a worker and its messages, and a store into a field of an `@atomic`
class and its `new`. These mark every block they reach, and a counted block so marked changes its
count atomically. Two threads that reach a counted object through a global, with no share point
between them, race on its count, as they would under rc; such a program is not rejected. The
`@atomic` store sees `obj.field = v` only, not a push into an array held by an `@atomic` object, as
under rc.

**What it costs.**

- A release of a block with one owner is a test of one word, as before.
- A release of a shared block (one that crossed a share point) takes an out-of-line path with an
  atomic operation: about 4 ns more per block, +27% on a probe that shares and drops lists of 2000
  nodes (the share walk and the releases together).
- Every type carries a share routine (`tsshr_...`), also in a program that never shares: about
  1.5 KB in the default library's DLL.
- A field taken out of a call's result costs a header load and two branches.

**Limits.**

- A cycle through counted objects is never freed, as under rc and as with `Shared<T>`.
- An object of a class defined in another module, with a tagged-union or closure field, that this
  module counts while it sees the object only as something else: a base class, an interface, `any`,
  `unknown` or `Opaque`. The compiler tells whether an object may have such a field from the classes
  of this module only.
- Such a field taken out of a call's result is emptied, by a module built with the flag or without
  it, even when the object has other owners: they read the field as empty later (never freed memory).

## Decorators

Inside one module the compiler infers what each function does with its parameters. A function
another module can call - an exported function, a method of an exported class - is called from
code that sees only its declaration, so what its callers rely on has to be in its signature.
The compiler says so when it is missing, and checks the body against what the signature says.

The other memory models ignore `@keeps`, `@lends`, `@record`, `@borrows`, `@immortalizes` and
`@nodrops`, so the same source compiles under all of them. A method that overrides another has to
say the same `@keeps`, `@borrows`, `@lends` and `@record` as the method it overrides.

Each takes the names of parameters, as strings; `"this"` names the receiver.

### `@keeps`

`@keeps("p")`: the function keeps parameter `p` - stores it in a field, an element or a
global - so its callers move the argument in and may not use it afterwards.

Without it, an exported constructor that stores its parameter is an error:

```typescript
export class Tag { constructor(public name: string) {} }

export class Holder {
    tag: Tag;
    constructor(tag: Tag) {
        this.tag = tag;
    }
}
```

```text
m1_keeps_err.ts:6:9: error: 'this value' is a parameter, kept here, but -mm=own cannot make the callers of 'Holder.constructor' give it up
        this.tag = tag;
        ^
m1_keeps_err.ts:5:5: note: 'Holder.constructor' could return a borrow or keep a parameter, but it can be called from another module; '@lends' or '@keeps' would say it in its signature
    constructor(tag: Tag) {
    ^
```

With it:

```typescript
export class Tag { constructor(public name: string) {} }

export class Holder {
    tag: Tag;

    @keeps("tag")
    constructor(tag: Tag) {
        this.tag = tag;
    }

    @keeps("tag")
    replace(tag: Tag) {
        this.tag = tag;
    }
}

function main() {
    const h = new Holder(new Tag("first"));
    const t = new Tag("second");
    h.replace(t);               // t moves into h: t may not be used after this
    print(h.tag.name);
}
```

```text
second
```

`@keeps` also goes on a setter that stores its value (`@keeps("n") set rebase(n: Node)`), and on a
generator: its state then owns the arguments it holds, instead of borrowing them (see
[`@borrows`](#borrows)), so the generator can be stored and returned. The state owns every argument
it holds, so `@keeps` names each of them:

```typescript
@keeps("iter", "f")
function* mapped<T extends { next(): { value: any, done: boolean } }>(iter: T, f: (x: number) => number) {
    for (const v of iter) yield f(v);
}

function* count(n: number) {
    for (let i = 0; i < n; i++) yield i;
}

function make() {
    return mapped(count(3), (x) => x * 10);   // count(3) moves into the state
}

function main() {
    let t = 0;
    for (const v of make()) t += v;
    print(t);
}
```

```text
30
```

Given an iterator that borrows its container (`s.values()` of a `Set`, a generator over a
parameter), such a generator owns that iterator but is bounded by the container in turn: it can be
walked in place while the container lives (`for (const v of s.values().take(2))`), but not stored,
returned or captured past it.

### `@lends`

`@lends("p")`: the result is a borrow of parameter `p` (or of `"this"`): the caller owns
nothing, and may use the result only while `p`'s owner keeps it.

Without it, returning a field of an argument from an exported function is an error:

```typescript
export class Node { constructor(public v: number) {} }
export class List { first: Node = new Node(1); }

export function firstOf(l: List): Node {
    return l.first;
}
```

```text
m2_lends_err.ts:5:5: error: 'this value' borrows a field and cannot be stored, returned or captured
    return l.first;
    ^
m2_lends_err.ts:4:1: note: 'firstOf' could return a borrow or keep a parameter, but it can be called from another module; '@lends' or '@keeps' would say it in its signature
export function firstOf(l: List): Node {
^
```

With it:

```typescript
export class Node { constructor(public v: number) {} }

@final
export class List {
    first: Node = new Node(1);

    @lends("this")
    peek(): Node {
        return this.first;
    }
}

@lends("l")
export function firstOf(l: List): Node {
    return l.first;
}

function main() {
    const l = new List();
    const a = firstOf(l);       // a borrow of l's Node: valid while l is
    const b = l.peek();
    print(a.v + b.v);
}
```

```text
2
```

`@lends` with one name is a plain borrow. With several, the result is a record that borrows each
of them, and it needs [`@record`](#record).

### A method that returns `this`

A method whose result type is `this` returns the object it is called on, so calls chain. It lends
`this` without saying `@lends("this")`, and the compiler checks that the body returns `this` on every
path (one that returns another object of the type is an error under own). Called on an object the
caller owns no more after the call - a new one - the result is that object, the caller's to keep:
a chain from a new value can be returned, stored or given to a function that keeps it. A method of
a class only this module uses may say the class instead of `this`; its body shows the same.

```typescript
export class Bag {
    items: number[] = [];

    add(n: number): this {
        this.items.push(n);
        return this;
    }
}

function filled(): Bag {
    return new Bag().add(1).add(2);     // the new Bag, given back
}

function main() {
    const b = filled();
    b.add(3);                           // on b: a borrow of it, used in place
    print(b.items.length);
}
```

```text
3
```

### `@record`

`@record`, always beside `@lends`: the result is a *borrowing record* - a new object the caller
owns and frees, whose fields borrow the arguments `@lends` names. The record may not outlive any
of them.

```typescript
export class Node { constructor(public v: number) {} }

export interface Pair {
    a: Node;
    b: Node;
}

@lends("a", "b")
@record
export function pairOf(a: Node, b: Node): Pair {
    return { a: a, b: b };
}

function main() {
    const x = new Node(1);
    const y = new Node(2);
    const p = pairOf(x, y);     // the caller owns the record; p.a and p.b borrow x and y
    print(p.a.v + p.b.v);
}
```

```text
3
```

Without `@record`, a caller in another module would take the result for a plain borrow and never
free the record, so leaving it out of an exported function is an error:

```text
m4_record_err.ts:8:1: error: 'pairOf' returns a record that borrows 'a', 'b', which another module can learn only from '@record' beside its '@lends'
@lends("a", "b")
^
```

### `@borrows`

`@borrows("p")`, on a generator: the generator's state borrows `p` (or `"this"`). The caller owns
the generator, but it may not outlive `p`. It also goes on a function that returns such a
generator. On anything else it is an error.

A generator borrows its parameters without saying so, `this` among them: an extension
generator such as `function* keys<T>(this: T[])`, called `a.keys()`, may not outlive `a`. A function
that returns one says `@borrows("this")`, as `[Symbol.iterator]` of an array does.

```typescript
@final
export class Word {
    constructor(public text: string) {}

    @borrows("this")
    *letters() {
        for (const c of this.text) yield <string>c;
    }
}

@borrows("w")
export function *lettersOf(w: Word) {
    for (const c of w.text) yield <string>c;
}

function main() {
    const w = new Word("abc");
    let s = "";
    for (const c of lettersOf(w)) s += c;    // the generator may not outlive w
    for (const c of w.letters()) s += c;
    print(s);
}
```

```text
abcabc
```

### `@immortalizes`

A module-level array that is only ever pushed onto - a *keeper*, such as a list of registered
callbacks - holds what it is given until the program ends. A value pushed into one is made
*immortal*: its owner's release frees nothing, and neither does the release of anything a closure
or a bound method in it relies on.

Inside a module the compiler finds the keepers itself. `@immortalizes("p")` says, for callers in
other modules, that the function keeps parameter `p` in one; each caller then makes its argument
immortal.

```typescript
type Callback = (n: number) => void;
let handlers: Callback[] = [];          // only ever pushed onto: a keeper

@immortalizes("h")
export function onTick(h: Callback): void {
    handlers.push(h);
}

function setup(): void {
    const total = [0.0];
    onTick((n: number) => { total[0] += n; print(total[0]); });
}                                       // total outlives setup: the closure keeps it

function main() {
    setup();
    for (const h of handlers) h(1);
    for (const h of handlers) h(2);
}
```

```text
1
3
```

Without it, the exported function is an error:

```text
m5_immortal_err.ts:4:1: error: 'onTick' keeps '#0' for good, but -mm=own cannot tell its callers in other modules; '@immortalizes' would say it in its signature
```

On a `declare function` it is trusted as written: say it on a C function that keeps a callback.

### `@nodrops`

`@nodrops declare function f(...)`: a function defined outside TypeScript - a C function - that
frees nothing a TypeScript caller can reach. A call of a declared function without it may free
anything, so a borrow cannot be lent to it.

```typescript
@nodrops declare function strlen(s: string): index;

class W {
    constructor(private text: string) {}

    size(): int {
        return <int>strlen(this.text);   // lends a field to a C function
    }
}

function main() {
    print(new W("abc").size());
}
```

```text
3
```

Without `@nodrops`:

```text
m6_nodrops_err.ts:7:21: error: 'this value' borrows a field but is used here after it may be released or overwritten
        return <int>strlen(this.text);   // lends a field to a C function
                    ^
```

A declaration has no body to check, so this is the declaration's word. Never put it on a function
that can run TypeScript code (one that takes a callback).

### `@final`

`@final class C`: no class may extend `C`. It works under every memory model: a call on a value
of type `C` is a direct call, with no vtable lookup
([design](../tslang/docs/superpowers/specs/2026-10-19-final-classes-design.md)).

`@final interface I`: only classes of this module implement `I`. A value of type `I` is made from
a class object or null and nothing else: a cast from an object literal, from another interface's
value or from `any` is an error, as are an interface that extends `I`, `export` on `I`, and an
exported or declared class that implements it. In return, with `--opt_level=3` a call of one of
`I`'s methods that at most four functions can be is those functions called directly, with no call
through a pointer left.

Under `-mm=own` it matters more. A method of an exported class that is not `@final` may be
overridden in another module, by a method that frees anything; the compiler cannot see that
method, so a call of it ends every borrow the caller holds, and its result is not known to be
fresh:

```typescript
export class S {
    color: string = "red";
    w: number = 2;
    area() {
        return this.w * 2;
    }

    describe() {
        return `${this.color} area=${this.area()}`;
    }
}

function main() {
    print(new S().describe());
}
```

```text
m8_virtual_err.ts:11:16: error: 'this value' borrows a field but is used here after it may be released or overwritten
        return `${this.color} area=${this.area()}`;
               ^
m8_virtual_err.ts:11:38: note: it may be released or overwritten here
        return `${this.color} area=${this.area()}`;
                                     ^
m8_virtual_err.ts:11:38: note: 'S.area' may be overridden in another module by a method that does; '@final' on the method, or on its class, says it is not
```

With `@final export class S` it compiles and prints `red area=4`. Mark exported classes `@final`
unless another module has to extend them.

A class that other modules extend can mark single methods instead: `@final area()` may not be
overridden (a subclass that does is an error, and an abstract method can't be `@final`), so a call
of it runs that method, and `-mm=own` goes by what its body does. That holds for a class imported
from another module too: the declaration it is imported by says `@final`.

### `@value`

`@value class C` and `@value interface R`: a type whose values are copied, like numbers, not
shared. It has no identity, no vtable and no base class, and its values are not heap blocks, so
they need no owner at all
([design](../tslang/docs/superpowers/specs/2026-10-24-value-types-design.md)). It suits small
immutable types such as vectors and colors, and it works under every memory model.

```typescript
@value
class Vector {
    constructor(public readonly x: number, public readonly y: number) {}

    plus(v: Vector) { return new Vector(this.x + v.x, this.y + v.y); }
}

@value
interface Ray {
    readonly start: Vector;
    readonly dir: Vector;
}

function main() {
    let p = new Vector(0, 0);
    const r: Ray = { start: p, dir: new Vector(1, 2) };
    for (let i = 0; i < 3; i++) p = p.plus(r.dir);   // no allocation: Vectors are copied
    print(p.x, p.y, r.start.x);
}
```

```text
3 6 0
```

A `@value` record may hold a reference to a class object; under `-mm=own` that field is a
borrow, so the record may not outlive the object it names (the ray tracer's `Intersection`,
which names the object it hit).

## Separate compilation

A module imported as source (`import './lib'`) is read as declarations: the importer sees the
decorators above, and nothing of the bodies. Everything else about another module's functions is
unknown, so a call of one may free anything.

A module compiled under `-mm=own` also records, in its object file, what each exported function
frees: nothing, or only certain fields, and whether its result is fresh. Give the objects of the
modules a program imports to `--own-facts`, and calls into them keep the caller's borrows they do
not touch:

```bash
tslang --emit=obj -mm=own camera.ts -o camera.obj
tslang --emit=exe -mm=own --own-facts=camera.obj --obj=camera.obj main.ts -o main.exe
```

A shared library built under `-mm=own` carries the same facts, which an importer reads from the
loaded library. A module built under another memory model is unknown to `-mm=own`, and objects
that cross from it are never freed (see [Mixing modules](memory-models.md#mixing-modules)).
Modules built with and without `--own-fallback=count` mix freely: every own release honors a
count ([Counted fallback](#counted-fallback---own-fallbackcount)).

## Code for one memory model

`__MEMORY_MODEL__` is the memory model a module is compiled for, as a string. A comparison with it
is decided at compile time, and only the branch taken is compiled:

```typescript
function main() {
    if (__MEMORY_MODEL__ === "own") {
        print("single owner");
    } else {
        print("model:", __MEMORY_MODEL__);
    }
    print(__OS__, __ARCH__);
}
```

```text
single owner
win32 x64
```

`__OS__` and `__ARCH__` are the target's OS and architecture, named as Node's `process.platform`
and `process.arch` name them.

`--verify-ownership` is a debugging aid for the compiler itself: it checks that every slot taking
a reference gives it back on every path out of a function.

## Current limits

`-mm=own` is still in development, and some valid TypeScript does not compile under it yet.
[own-readiness.md](own-readiness.md) says how far it is, with measurements.

- The default library has an `-mm=own` build, and arrays, strings and their methods work. `Map`
  and `Set` work with number, boolean and string keys and values (`new Map<string, number>()`,
  `new Set([1, 2])`): their iterators borrow the container, and strings go in as copies. An
  object goes in as a `Shared<T>` handle (`new Map<string, Shared<Node>>()`); a plain object or
  `any` key or value is a compile error that points at `Shared<T>`, which includes a `new Map()`
  with no type arguments and nothing to infer them from.
- Array methods that keep elements of the source in a new array (`filter` over class instances)
  give each such element a second owner, which is an error by design. Use `Shared<T>` elements,
  or build the result from copies.
- `reverse` and `sort` work on arrays of numbers, strings and class instances. They exchange
  elements with `[a[i], a[j]] = [a[j], a[i]]`, which moves nothing; swap two elements or two fields
  the same way in your own code. To take a value out of a place, exchange it with a local:
  `let v: T | undefined = undefined; [v, this.f] = [this.f, v]` leaves the place holding what the
  local held and the local owning what the place held. A swap through a local (`let t = a[i];
  a[i] = a[j]; a[j] = t`) gives an object in `t` a second owner, which is an error. A callback your function only calls,
  tests (`compare && compare(a, b)`), hands on to another function (as `sort` hands its comparison
  to a helper), names again (`const g = f`) or captures in a closure that calls it (`a.forEach((v)
  => f(v))`, also kept in a local) is known by the functions its callers pass. So is one stored in
  a class's field (`constructor(private f: ...)`, `this.f = f`, also from a class extending it)
  and one a function returns, also when the class's objects are seen through an interface. Not
  when one of the class's objects, or of a class extending it, becomes `any` or a union, or another
  module may build or extend it: then its fields' callbacks may be any function. Nor is a callback
  parameter of a method an interface reaches: a call through the interface may pass any function,
  from any module. A callback may assign the numbers and strings it
  captures (`total += v.n`); one stored in an object may not capture locals, which would escape
  with it. `copyWithin` works on
  numbers and strings (a string is copied where it moves); over class instances it would leave one
  object in two elements, an error by design.
- Iterator helpers work, the lazy ones (`map`, `filter`, `take`, `drop`) taking their source over
  as in JavaScript: `const seq = fibonacci().map(f)`. Over an iterator that borrows its container
  (`s.values()` of a `Map` or a `Set`) a helper is bounded by the container too:
  `for (const v of s.values().take(2))` works, but the result cannot be returned or kept past `s`.
  An array's `values()`, `keys()`, `entries()` and `[Symbol.iterator]()` borrow it in the same
  way. A closure that captures, given to two of them, is moved into the first. What a generator
  yields is the caller's: `for (const v of g()) a.push(v)` and `[...s.matchAll(re)]` take each value
  over, as long as nothing reads it after it is pushed.
- Arguments given to a rest parameter (`a.concat(b)`) are borrowed for the call, so `b` is still
  yours after it.
- `reduce` and `reduceRight` over an array of arrays or objects are errors: with no initial value
  they start from an element, and may return it as it is, a second owner of it.
- The `Array<T>` class holds values, strings or `Shared<T>` handles, as `Map` does: `new
  Array<number>()`, `new Array<string>()` and `new Array<Shared<Node>>()` work, and `new
  Array<Node>()` is a compile error that points at `Shared<T>`. A string stored again and again
  (`fill`) is copied each time.
- A parameter with a default (`constructor(private items: T[] = [])`) is kept like any other
  parameter, whether the caller passes it or not.
- A linked list whose nodes own their `next` works: a local that steps down it (`for (let c =
  this.head; c; c = c.next)`) borrows the nodes, a new node goes in at the head (`n.next = this.head;
  this.head = n`) or at the tail (`c.next = n`), and `this.head = this.head.next` or `c.next =
  c.next.next` takes a node out. The node taken out is freed right there, so read what you need of
  it first: `const h = this.head; const v = h.v; this.head = h.next; return v;`, not `return h.v`
  after the assignment. A node can also be made with the old head as an argument (`this.head = new
  Node(v, this.head)`), and a tree grown by a function that recurses on a field and gives it back
  (`t.left = insert(t.left, v)`): the field is given up to the call, and its result refills it. That
  takes a call to a function of the same module that never throws (no `throw` and no `await` in it or
  in anything it calls), whose other arguments are values with no heap block or the object `new` made;
  anything else is an error. A walk may assign its
  parameter (`while (n) { k++; n = n.next; }`) when every value it gives it is read out of the one it
  holds: the parameter then borrows the caller's list. A parameter given anything else (`n = new
  Node(0)` on some path) is kept, and the caller gives it up.
- A back-reference is a `WeakRef<T>` (`parent: WeakRef<Node> | null`, `c.parent = new WeakRef(this)`,
  `this.parent?.deref()`). It does not own its target: when the target's owner frees it, `deref()`
  gives `undefined` from then on. The object `deref()` gives is a borrow: it cannot be stored,
  returned or captured, and anything that may free an object of its type (a release of a holder of
  that type, a call that may drop) ends it, so read what you need right away. A `@final` target class
  keeps more of those borrows. The target is a class object. A module that makes WeakRefs may not
  share with another thread (`Thread.spawn`, a worker, `Shared.share`) anything that may hold their
  target, since the count is not atomic; in practice no `Thread.spawn` of a closure in such a module.
  Each object carries the count in the header word it already has: no object grows.
- A back-reference can also be a `@weak` field, read and written like any other field:

  ```ts
  class Node {
      @weak parent: Node | null = null;
      children: Node[] = [];
      add(c: Node) { c.parent = this; this.children.push(c); }
      depth(): number { return this.parent ? 1 + this.parent.depth() : 0; }
  }
  ```

  It is a `WeakRef` underneath, so it owns nothing and has the same limits (a class object, no
  threads). Each read is checked: reading it after its object was freed stops the program with "the
  @weak field 'Node.parent' was read after its object was freed". `===`, `!==`, `== null` and
  `switch (c.parent)` compare the stored pointer and never stop, so `if (c.parent === this) c.parent = null` is always safe. A read
  borrows, as `deref()` does: it cannot be stored, returned or captured, and it cannot be the object or
  an argument of a call that may free objects (`@nodrops` on the method or `@final` on its class says
  it frees nothing). Writing a new object into it is an error, in either arm of a `?:` too: nothing would own it. A
  field that redeclares a base class's `@weak` field is `@weak` as well, and an abstract field cannot be. A class with a
  `@weak` field cannot be seen through an interface that names the field. Under gc and none it is an
  ordinary field; under rc it is an error.
- `Shared<T>` cycles are never freed; break them with a `WeakRef` where one side only refers back.
- There are no lifetime annotations: a borrow that the compiler cannot follow is an error, even
  when the program would be correct. The fix is usually a move, a fresh value or `Shared<T>`; for
  some of these errors, `--own-fallback=count` counts the object instead.

The ray tracer benchmark ([tslang/benchmarks/raytrace](../tslang/benchmarks/raytrace/README.md))
compiles under `-mm=own`; its timings against the other memory models are in the
[performance page](performance.md).
