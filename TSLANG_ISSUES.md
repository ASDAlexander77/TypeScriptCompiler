# tslang issues found while working on native_ts_graphics

Issues hit while porting https://github.com/ASDAlexander77/native_ts_graphics to tslang. Every repro
should be a complete file that was run (`tslang file.ts`, JIT, Windows x64), with the tslang version.

## Open

Items 1-30 (collected 2026-10-12 to pre-alpha119) are fixed or answered; their write-ups are in this
file's git history. 24-26 were fixed by #775 (regression tests
`tslang/test/tester/tests/00tslang_issues_24_26.ts` and `ref_element_layout_error.ts`): 24 (an array
literal holding `const X = 1 << n` widened to 64-bit elements: shifts and bitwise operators now give an
`s32`, as `int` is); 25 (`cond ? [1] : []` crashed when iterated: an empty array literal now takes the
other branch's array type); 26 (`Ref(x)` of a `number` was accepted for a `Ref<f32>` parameter: a
reference to a number laid out otherwise is now an error). 27 (`-mm=rc`: a template literal passed
straight to a call reached the callee with no reference behind it, so `console.log(text)` freed it
before `this.message = text` stored it): a template, and an object or literal array printed into a
string, now carry a reference of their own as `"a" + b` does, which the caller gives back after the
call. Under `-mm=own` and rc the same strings also leaked (3M template arguments: ~570 MB, now ~50 MB).
Regression test `tslang/test/tester/tests/00tslang_issue_27.ts`.
28 (a `readonly` constructor parameter without `public`, `private` or `protected` made no parameter
property, so `t.width` didn't resolve): `readonly` alone now makes a public one, as in TypeScript.
Regression test `tslang/test/tester/tests/00tslang_issue_28.ts`.
29 (a function-typed parameter of a `declare function` was a TypeScript function value, `{ ptr, ptr, ptr }`,
so `_beginthreadex(..., entry, arg, ...)` shifted `arg` and every argument after the callback): a
function declared `@cdecl` takes each function-typed parameter as a C function pointer. A function or
an arrow that captures nothing passes; a capturing arrow or a bound method passed straight to it is a
compile error ("a closure or a bound method can't be used as a plain function pointer"). Not checked
yet: a closure already stored in a variable of function type, passed on to a `@cdecl` parameter, still
loses its captures at run time. Regression tests `tslang/test/tester/tests/00tslang_issue_29.ts`,
`00tslang_issue_29_thread.ts` (Windows), `00tslang_issue_29_pthread.ts` (Linux) and
`cdecl_closure_error.ts`.
30 (a generic's body, compiled where it is first instantiated, resolved a name to a local of the
instantiating function: `const total = 5` in `main` hid the module's `total()` from `Bag<T>.sum()`, and
a local `values` broke the default library's `Array<T>`): a generic declared outside every function
now sees no local of another function; one declared in a function still reads that function's locals,
and lambdas in a generic still capture the generic's own. Regression test
`tslang/test/tester/tests/00tslang_issue_30.ts`.
Items 34-36, found in the default library on 2026-11-08, are fixed or answered too: 34 (a string
literal starting with U+FEFF lost it, so its `.length` was 0): LLVM's UTF-16 to UTF-8
conversion took a leading U+FEFF for a byte order mark and skipped it, and would have byte-swapped a
literal starting with U+FFFE. Literals are now converted as they are. 35 (`"\\n"` read as a line feed)
did not reproduce: on the same build the repro prints `2 6 4 2 2 3` and `92 2`, as expected. The repro
file was most likely written through a shell that turns `\\` into `\`, which gives exactly the output
reported. Both are covered by `tslang/test/tester/tests/00string_literal_escapes.ts`. 36 (an older
runtime on `TSLANG_LIB_PATH` failed every `--emit=dll` with a bare LNK2001 for `__tslang_heap_shared`):
tslang now checks the runtime's `TypeScriptAsyncRuntime.lib` before linking a library and says it is
older than the compiler. `-v` is an alias for `--verbose`, which prints the linker's command line, so
the linker's "use -v" hint works.

Items 31-33, `-mm=own` limits found in the default library on 2026-11-08, are fixed too, and
`thread_spawn.ts` and `atomics_wait.ts` there now run under every model. 31 (two closures couldn't
capture the same array): a closure that escapes owns what it captures, so that stays an error for a
plain object, and `Shared<T>` is the way to share one. A `const` `Shared<T>` is now captured by copy, so
any number of closures hold a counted handle each, in a loop too; under `-mm=own` a handle's count is
always atomic, since handles are what threads share. 32 (the error said `'this value'` and pointed at
the arrow): the error about a captured variable now names it, with or without `--di`, points at its
first use in the closure, and says to use a `const Shared<T>`. 33 (`this.result = this.fn()` was refused
when `R` is a class): the result of a call through a callback is the caller's when every function that
can reach the call is defined in the module and none returns a borrow of its argument. A callback that
gives back its argument or what it captured is still refused. Regression tests
`tslang/test/tester/tests/00tslang_issues_31_33.ts` and `tslang/test/tester/own/own_err_capture_*.ts`,
`own_err_callback_result_*.ts`.

Items 37 and 38, found on 2026-11-08 while trying to add JavaScript's `Symbol` to the default library:
37 (a function named `Symbol` crashed the compiler, even when nothing called it): the default library's
`[Symbol.toPrimitive]` and the like then named a member the function hasn't got, and the failed computed
name was read as if it had a value. A failed computed name is now the error it is, and see 38. 38 (a
declared `Symbol` hid the built-in one): `[Symbol.iterator]` and the other well-known names are recognized
in a computed name before `Symbol` is resolved, so a `Symbol` declared as a function or a class leaves them
working. As a value, `Symbol.iterator` stays the built-in one unless the declared `Symbol` has that member.
`typeof Symbol.iterator` gives `"symbol"`. Printing a symbol, or storing one in a variable that is used at
run time, is now a compile error that says so; it was an internal error before. A library can now declare
`Symbol`, so `Symbol("x")`, `Symbol.for`, `Symbol.keyFor`, `description` and `toString` are library work.
Regression tests `tslang/test/tester/tests/00tslang_issue_37.ts`, `00tslang_issue_38.ts`,
`symbol_value_error.ts` and `computed_name_failed_error.ts`.

Still open from 38: a symbol made at run time can't be a property key (`obj[sym]`). Fields are laid out at
compile time, so a symbol needs a run-time value and a way to look a key up. It has not been tried.

### `-mm=own` limits found in the default library, 2026-11-09 (items 39-42)

Found while adding `Worker` and making `JSON.stringify` work under `-mm=own` in the default library
(TypeScriptCompilerDefaultLib PRs #80-#85). The library works around each one, and its README names
them as `-mm=own` limits; each needs a compiler change to go away. Every repro below is a complete
file, run with `tslang --nowarn -mm=own --emit=jit file.ts` and again with `-mm=gc`, on Windows x64,
tslang commit 2fffc119 (release build of 2026-11-09 17:35), against the default library at
TypeScriptCompilerDefaultLib 916cd07. Each compiles and runs under `-mm=gc`, with the output given.

#### 39. A method can't say it gives its receiver back owned, so a chain from a new value can't be returned or kept by a callee

A method that returns `this` so calls can be chained (`JsonValue.set`, `JsonValue.push`) can only be
declared `@lends("this")` for another module: its result borrows the value it was called on. A chain
that starts from a new value (`JsonValue.newObject().set(...)`) has no owner but that borrow. Kept in a
local it compiles now; returned, or given to a parameter that is kept, it is an error.

```typescript
// JsonValue is in the default library, another module: its set() and push() are declared
// @lends("this") (they return `this`, so a chain can go on), and that is all a caller knows of them.

// returning a chain from a new value: under -mm=own "'this value' borrows the result of
// 'JsonValue.set' and cannot be stored, returned or captured"
function person(): JsonValue {
    return JsonValue.newObject().set("name", JsonValue.ofString("Ann")).set("age", JsonValue.ofNumber(30));
}

function main() {
    // compiles under -mm=own: a chain kept in a local
    const a = JsonValue.newObject().set("a", JsonValue.ofNumber(1));
    print(a.stringify());

    print(person().stringify());

    // a chain given to a parameter that is kept (set's `value`): the same error
    const b = JsonValue.newObject();
    b.set("list", JsonValue.newArray().push(JsonValue.ofNumber(1)).push(JsonValue.ofNumber(2)));
    print(b.stringify());
}
```

`-mm=own`:

```text
39_chain_receiver.ts:7:5: error: 'this value' borrows the result of 'JsonValue.set' and cannot be stored, returned or captured
    return JsonValue.newObject().set("name", JsonValue.ofString("Ann")).set("age", JsonValue.ofNumber(30));
    ^
39_chain_receiver.ts:19:5: error: 'this value' borrows the result of 'JsonValue.push' and cannot be stored, returned or captured
    b.set("list", JsonValue.newArray().push(JsonValue.ofNumber(1)).push(JsonValue.ofNumber(2)));
    ^
```

`-mm=gc`: `{"a":1}`, `{"name":"Ann","age":30}`, `{"list":[1,2]}`.

To do: a way to declare that a method returns its receiver (`return this` on every path), so that
called on a value the caller owns, or on a temporary, the result is that value, owned by whoever owned
the receiver, rather than a borrow of it. Something like `@returnsThis` / `@lends("this")` that a
temporary receiver turns into a move. Then the default library's `tests/json.ts` can drop its
`__MEMORY_MODEL__ != "own"` branch (`chained()`), and `toJSON()` can return a chain.

#### 40. Once an arrow captures anything, an arrow inside it can't capture the outer arrow's parameter

An arrow given a parameter (`self`) stores a second arrow that captures `self`. That compiles under
`-mm=own`, until the outer arrow also captures something from its own enclosing function. Then the
inner arrow's capture of `self` is refused. It is the shape of the default library's `Worker`:
`new Worker((self) => { self.onmessage = (e) => self.postMessage(...); })`. A worker whose function
uses state shared with the creating thread can't also post from its handler (the library's README and
`docs/worker-design.md` name the limit; `tests/worker_terminate.ts` tests the two shapes apart).

```typescript
class Scope {
    handler: () => void;
    count: int;

    constructor() {
        this.handler = () => {};
        this.count = 0;
    }
}

function run(body: (self: Scope) => void) {
    const scope = new Scope();
    body(scope);
    scope.handler();
    print(scope.count);
}

function main() {
    const shared = new Shared(new Int32Array(1));

    // compiles under -mm=own and prints 1: the outer arrow captures nothing
    run((self: Scope) => {
        self.handler = () => {
            self.count++;
        };
    });

    // the same, once the outer arrow captures something: under -mm=own
    // "a closure that escapes owns what it captures, but -mm=own cannot move this variable into it:
    // its value is the caller's", at the inner arrow
    run((self: Scope) => {
        const counter = shared;
        Atomics.add(counter.value, 0, 1);
        self.handler = () => {
            self.count++;
        };
    });
}
```

`-mm=own`:

```text
40_closure_param_capture.ts:34:24: error: a closure that escapes owns what it captures, but -mm=own cannot move this variable into it: its value is the caller's
        self.handler = () => {
                       ^
```

`-mm=gc`: `1`, `1`.

To do: the inner arrow captures `self` the same way in both calls, and `self` is the outer arrow's
parameter in both; what the outer arrow captures should not change whose `self` is. The error also
does not name the variable (item 32 made it name a captured variable in other cases).

#### 41. An arrow kept in a field can't capture a parameter, and `@keeps` can't say that it does

A constructor that keeps an arrow capturing one of its parameters is refused ("its value is the
caller's"), and `@keeps("body")`, which would make the caller give `body` up, is refused too: the
compiler finds that the constructor keeps nothing, although the arrow it stores takes `body` with it.
The default library hit this starting a `Worker`'s thread with `Thread.spawn(() => run(body, ...))` in
its constructor; it now keeps a `WorkerStart` object holding `body` (as `ThreadHandle` keeps its
`ThreadStart`) and starts the thread natively.

```typescript
class Holder {
    run: () => int;

    constructor(body: (n: int) => int) {
        // the arrow is kept in a field and captures the parameter `body`
        this.run = (): int => body(2);
    }
}

function main() {
    const h = new Holder((n: int): int => n * 21);
    print(h.run());
}
```

`-mm=own`:

```text
41_keeps_closure_capture.ts:6:20: error: a closure that escapes owns what it captures, but -mm=own cannot move this variable into it: its value is the caller's
        this.run = (): int => body(2);
                   ^
41_keeps_closure_capture.ts:6:9: note: the closure escapes here
        this.run = (): int => body(2);
        ^
```

With `@keeps("body")` on the constructor (and `@final export class Holder`):

```text
41b_keeps_rejected.ts:6:5: error: '@keeps' of 'Holder.constructor' names 'body', but its body keeps nothing
    @keeps("body")
    ^
```

`-mm=gc`: `42`.

To do: count a parameter captured by an escaping closure as kept, both when inferring what a function
keeps and when checking `@keeps`, so the closure owns it (the caller moves the argument in).

#### 42. A class instance can't be taken out of a `T | undefined` field

Exchanging a field with a local (`[data, this.data] = [this.data, data]`), the way the own-model guide
says to take a value out of a place, works for a string but not for a class instance in a field typed
`T | undefined`. The default library wanted a queue item holding either a message or an error; it now
queues the event objects themselves (`MessageEvent<Out> | ErrorEvent`) so nothing has to come out of a
field.

```typescript
class Item {
    constructor(public n: number) {}
}

// a slot a value is put into and later taken back out of, by exchanging it with a local
class Slot<T> {
    constructor(public data: T | undefined) {}

    take(): T {
        let data: T | undefined = undefined;
        [data, this.data] = [this.data, data];
        return <T>data;
    }
}

function main() {
    // compiles under -mm=own and prints "a"
    const text = new Slot<string>("a");
    print(text.take());

    // the same for a class instance: under -mm=own
    // "'this value' borrows a field and cannot be stored, returned or captured"
    const item = new Slot<Item>(new Item(1));
    print(item.take().n);
}
```

`-mm=own`:

```text
42_move_out_of_union_field.ts:11:29: error: 'this value' borrows a field and cannot be stored, returned or captured
        [data, this.data] = [this.data, data];
                            ^
42_move_out_of_union_field.ts:11:9: error: 'this value' borrows an element and cannot be stored, returned or captured
        [data, this.data] = [this.data, data];
        ^
42_move_out_of_union_field.ts:12:9: error: 'this value' is merged from several branches (`?:`, `&&`, `||`, `??`, a loop); -mm=own cannot prove which one owns it yet
        return <T>data;
        ^
```

The same file with only the `Slot<string>` lines compiles under `-mm=own` and prints `a`.
`-mm=gc`: `a`, `1`.

To do: the exchange of a union-typed field with a local should move the value out for a class
instance as it does for a string, and the `<T>` cast of a `T | undefined` local that holds it should
not count as a merge of branches.
