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

Items 31-33 are `-mm=own` limits found in the default library (TypeScriptCompilerDefaultLib), not in
native_ts_graphics. There the tests `thread_spawn.ts` and `atomics_wait.ts` are marked `// not own`
because of them. Each repro was compiled with `--emit=exe` (the JIT's default-library DLL didn't
link against this build) and runs under `-mm=gc` with the output shown.

### 31. `-mm=own`: two closures can't capture the same array

tslang (commit 94b8bdb2, built with uncommitted local changes). Found 2026-11-08.

A closure passed to `Thread.spawn` moves what it captures, so a second closure capturing the same
`Int32Array` is a use after move. Threads can't share memory under `-mm=own`, which is what
`Atomics` is for. A single capture is fine: in the third repro the main thread still reads the
array after the thread captured it. The error names `'this value'`, not `counter`, and points at the
second arrow function rather than at `counter` inside it.

Repro (`share_two.ts`):
```ts
// -mm=own: two threads that capture the same Int32Array
function main() {
    const counter = new Int32Array([0]);
    const a = Thread.spawn((): int => { Atomics.add(counter, 0, 1); return 0; });
    const b = Thread.spawn((): int => { Atomics.add(counter, 0, 1); return 0; });
    a.join();
    b.join();
    print(Atomics.load(counter, 0));
}
```

`-mm=gc` prints `2`. `-mm=own`:
```
share_two.ts:5:28: error: 'this value' is used here after its value was moved
    const b = Thread.spawn((): int => { Atomics.add(counter, 0, 1); return 0; });
                           ^
share_two.ts:4:28: note: value moved here
    const a = Thread.spawn((): int => { Atomics.add(counter, 0, 1); return 0; });
                           ^
```

The same in a loop (`share_loop.ts`), the usual way to start workers, gives a different error:
```ts
// -mm=own: threads spawned in a loop, each capturing the same Int32Array
function main() {
    const counter = new Int32Array([0]);
    const workers: ThreadHandle<int>[] = [];
    for (let t = 0; t < 4; t++) {
        workers.push(Thread.spawn((): int => { Atomics.add(counter, 0, 1); return 0; }));
    }

    for (const w of workers) w.join();
    print(Atomics.load(counter, 0));
}
```

`-mm=gc` prints `4`. `-mm=own`:
```
share_loop.ts:6:35: error: 'this value' is moved inside a loop but was made outside it; only a borrow could do that, and -mm=own cannot prove one yet
        workers.push(Thread.spawn((): int => { Atomics.add(counter, 0, 1); return 0; }));
                                  ^
```

One capture compiles and runs under both models, printing `1` (`share_after.ts`):
```ts
// -mm=own: the main thread uses the Int32Array after a thread captured it
function main() {
    const counter = new Int32Array([0]);
    const a = Thread.spawn((): int => { Atomics.add(counter, 0, 1); return 0; });
    a.join();
    print(Atomics.load(counter, 0));
}
```

Expected: a way to share one array between threads under `-mm=own`, such as `Shared<T>` captured by
each closure, or a capture that borrows when the closure can't outlive the array.

### 32. `-mm=own`: the error for a move in a loop doesn't say what was moved

tslang (commit 94b8bdb2, built with uncommitted local changes). Found 2026-11-08.

Both errors in 31 call the captured variable `'this value'`. In a program with several captures,
nothing says which one is moved. The error should name `counter` and point at its use inside the
arrow function.

### 33. `-mm=own`: a thread can't return an object, and the error points into the default library

tslang (commit 94b8bdb2, built with uncommitted local changes). Found 2026-11-08.

`Thread.spawn` keeps the function's result in a field of `ThreadStart<R>` for `join()` to return.
When `R` is a class, storing it is rejected. The error is reported in the default library's
`lib.generics.ts` (`this.result = result;` in `ThreadStart.run`) with no line in the user's program,
so nothing points at the `Thread.spawn` call that caused it. A string or a number works.

Repro (`return_object.ts`):
```ts
// -mm=own: a thread whose function returns an object
class Box {
    constructor(public value: int) {}
}

function main() {
    const t = Thread.spawn((): Box => new Box(42));
    print(t.join().value);
}
```

`-mm=gc` prints `42`. `-mm=own`:
```
.\__build\defaultlib\generics/lib.generics.ts:857:13: error: 'this value' takes a second reference; -mm=own cannot prove a move or a borrow here yet
```

`const result = this.fn(); this.result = result;` moves a value the function just returned, which
the field should be able to take. Expected: the move is accepted. Failing that, the error should
also name the instantiating call in the user's file.
