# The `-mm=own` memory model

`-mm=own` frees memory the way Rust does: every heap block has exactly one owner, and the
compiler works out at compile time where each block is freed. There is no collector and no
reference count at run time. When the compiler cannot prove a single owner, it rejects the
program instead of compiling it with a leak or a double free.

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
- [Decorators](#decorators)
  - [`@keeps`](#keeps) - the parameters a function keeps
  - [`@lends`](#lends) - the arguments a result borrows
  - [`@record`](#record) - a result that is a box of borrows
  - [`@borrows`](#borrows) - the arguments a generator borrows
  - [`@immortalizes`](#immortalizes) - the parameters a function keeps for good
  - [`@nodrops`](#nodrops) - a C function that frees nothing
  - [`@final`](#final) - a class nobody extends
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

Inside a module the compiler also infers each function's facts from its body: a function that
stores a parameter keeps it, so its callers have to move the argument in; a function that
returns a field of its argument returns a borrow, so its callers may only borrow the result.
Callers in other modules cannot see the body, which is what the [decorators](#decorators) are for.

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

The usual ways out are the Rust ones: move instead of borrowing (give the value up), store a
fresh value (`new`, a copy), keep the borrow shorter than its owner, or share the object with
[`Shared<T>`](#shared-ownership-sharedt).

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
```

With `@final export class S` it compiles and prints `red area=4`. Mark exported classes `@final`
unless another module has to extend them.

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
  the same way in your own code. A swap through a local (`let t = a[i]; a[i] = a[j]; a[j] = t`)
  gives an object in `t` a second owner, which is an error. A callback your function only calls,
  tests or hands on to another function (as `sort` hands its comparison to a helper) is known by
  the functions its callers pass; one stored, captured or returned is not. `copyWithin` works on
  numbers and strings (a string is copied where it moves); over class instances it would leave one
  object in two elements, an error by design.
- Iterator helpers work, the lazy ones (`map`, `filter`, `take`, `drop`) taking their source over
  as in JavaScript: `const seq = fibonacci().map(f)`. Over an iterator that borrows its container
  (`s.values()` of a `Map` or a `Set`) a helper is bounded by the container too:
  `for (const v of s.values().take(2))` works, but the result cannot be returned or kept past `s`.
  An array's `values()`, `keys()`, `entries()` and `[Symbol.iterator]()` borrow it in the same
  way. A closure that captures, given to two of them, is moved into the first. A `for...of` over a
  generator, or a spread of one (`[...s.matchAll(re)]`), only borrows what each step yields: keeping
  a yielded object or array (pushing it into another array) is an error for now.
- The `Array<T>` class holds values, strings or `Shared<T>` handles, as `Map` does: `new
  Array<number>()` works, and `new Array<Node>()` is a compile error that points at `Shared<T>`.
  `Array<string>` and `Array<Shared<T>>` do not compile yet.
- A parameter with a default (`constructor(private items: T[] = [])`) is kept like any other
  parameter, whether the caller passes it or not.
- `Shared<T>` cycles are never freed.
- There are no lifetime annotations: a borrow that the compiler cannot follow is an error, even
  when the program would be correct. The fix is usually a move, a fresh value or `Shared<T>`.

The ray tracer benchmark ([tslang/benchmarks/raytrace](../tslang/benchmarks/raytrace/README.md))
compiles under `-mm=own`; its timings against the other memory models are in the
[README](../README.md#performance).
