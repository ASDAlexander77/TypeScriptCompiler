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
compile error ("a closure or a bound method can't be used as a plain function pointer"). A function
value (a `const` of function type, a callback parameter) passed on to a `@cdecl` parameter is checked at
run time: a closure throws that error there, instead of the C function running it without its captures
later (an `atexit` handler crashed at exit). A function named directly skips the check. Also, a
capturing arrow can now be assigned to a function type whose leading parameters are `Opaque`
(`(a: Opaque, b: Opaque) => s32`, a C comparator). The type match took a leading `Opaque` for a receiver
and refused it; the parameters are now first matched as they are. Regression tests
`00tslang_issue_29_function_value.ts` and `00tslang_issue_29_comparator.ts`, and `tslang/test/tester/tests/00tslang_issue_29.ts`,
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

Items 39-42, `-mm=own` limits found in the default library on 2026-11-09 (TypeScriptCompilerDefaultLib
PRs #80-#85), are fixed. 39 (a chain from a new value, `JsonValue.newObject().set(...).set(...)`, could
not be returned or kept): a method whose result type is `this` returns the object it is called on, and
called on a temporary the caller owns, its result is that object, the caller's to keep. `: this` says
it, with no decorator, and implies `@lends("this")`; under `-mm=own` a body that returns another object
of the type is an error. A method typed with its class that returns `this` on every path works the same
within its module; for another module, `: this` is how a declaration says it (the `-shared` declaration
printer writes it). The default library's `set` and `push` return `this` now, and `tests/json.ts` runs
its chains under every model. 40 (an arrow that captures something could not store a closure over its
object parameter into a field of that object): it is a back-reference as it is from an arrow that
captures nothing. 41 (a closure kept in a field could not capture a parameter, and `@keeps` could not
say it did): a closure that is kept owns the parameters it captures, so the function keeps them and its
callers give them up; `@keeps("body")` is accepted. The error for a capture that still cannot move names
the variable. 42 (`[data, this.data] = [this.data, data]` did not move a class instance out of a
`T | undefined` field): an exchange of a local and a field or an element hands each value over under
every model, and `<T>data` of such a local is a view of it. Regression tests
`tslang/test/tester/own/own_chain_returns_this.ts`, `own_closure_keeps_param.ts`,
`own_swap_local_field.ts`, `own_err_this_result_other.ts` and `own_err_closure_kept_param_used.ts`.

Item 43, found on 2026-11-09 while giving the default library's `JsonValue` an indexer
(TypeScriptCompilerDefaultLib PR #86), is fixed. 43 (`bag[key] = value` on a class imported from a
library, whose indexer's `set` is `@keeps("value")`: the caller freed the value `set` kept, so it was
freed twice and the program hung at exit under `-mm=own`): the indexer's accessors of such a class are
looked up by name, and the ownership passes now take that call for a call of `set`, with its declared
`@keeps` and `@lends`. The crash of tslang.exe with `JsonValue`'s indexer was another bug, under every
model: the call made for `obj[key] = value` had no result even when `set` returns one, as `JsonValue.set`
does, so it disagreed with the function it calls and lowering to LLVM crashed now and then; it has the
setter's result now. Also, a method returning `this` that says what it keeps (`@keeps`) is no longer
told to say `@lends` too. Regression tests `tslang/test/tester/tests/import_class_keeps_indexer.ts`
(with `export_class_keeps_indexer.ts`) and `00class_indexer_set_result.ts`.

Items 44 and 45, found on 2026-11-10 in native_ts_graphics' `examples/nri/variable_shading_nri.ts`, are
fixed. 44 (`flag && a && b` of two classes in a namespace threw "Can't cast from union type"): the helpers
a cast makes from source (`___cast`, `___unbox`, `___unbox_interface`, `___check_tuple`, `___bin_op_*`,
`___union_eq_*`) are declared in the root namespace, but the one made for the cast before was erased
from the namespace of the code asking for the new one, and from the root only when that code was outside
every namespace. In a namespace the old `___cast`, made for `null | boolean | A`, stayed and was
instantiated for `null | boolean | A | B`, so a `B` reached its `throw`. The old helper is now erased from
the root namespace, where the new one is declared. 45 (`-mm=own` refused `flag && a && b` of two classes,
in a namespace or not: "borrows the result of '___cast<...>'"): the second `&&` widened `null | boolean |
A` to `null | boolean | A | B` through a `___cast` call, whose result own took for a value stored. A union
widened to one that has all its members is now a plain cast, the value keeping its tag and its bytes, as
the lowering already did for it. With that, `!this.flag && texture && descriptor` in a method of an
exported class (every class in a namespace is) was still refused, "borrows a field but is used here
after it may be released or overwritten": `___cast` tests an instance with its class's `..instanceOf`,
called through the vtable, and for a class another module may extend own took that call for one that
may destroy anything. A class's `..instanceOf` is the compiler's own, which destroys nothing, called
through the vtable as through the `.instanceOf` slot. Regression tests
`tslang/test/tester/tests/00tslang_issue_44.ts` and `00tslang_issue_45.ts`, under every model.

Item 46, found on 2026-11-10 while testing 45, is fixed. 46 (`-mm=own`: a function that returns its
union parameter, given a new object, freed it twice; the exe hung or crashed at exit, while the JIT run
passed). `function same(v: A | null): A | null { return v; }` called as `same(new A())`, and as much
with `boolean | A`, `null | boolean | A` widened to `null | boolean | A | B`, or a named object no longer
used. On v0.0-pre-alpha121 too: its exe exits with 0xC0000374 (heap corruption). A call whose result is
its argument itself (spec 78) takes a temporary over: the temporary's releases go and the result is the
caller's own. The argument here is `<A | null>new A()`, a view, and the check looked at the view's users,
which hold no release: the result was made the caller's own while the object's release stayed. The
temporary is the object behind its views now. And the object's release may be in another block of the
function (a condition after the call, `x instanceof A`, splits it): one the call leads to, where the
result is released too, counts as after it. With the result left a borrow when the object is used after
the call, `x instanceof A` of the borrowed union was refused ("borrows ... and cannot be stored"), as it
was of a field read (`const x = this.f; x instanceof A`) before: the object seen as an opaque pointer,
whose vtable is read for its class's `.instanceOf`, is a test that keeps nothing, as for an owned value.
Regression test `tslang/test/tester/tests/00tslang_issue_46.ts`, JIT and compiled, under every model.

Items 47 and 48, found on 2026-11-13 porting powder_sandbox and native_ts_graphics to `-mm=own`, are
fixed. 47 (a lazily cached field could not be returned through a local: `let result = this.cached; if
(!result) { result = new Q(1); this.cached = result; } return result;` was refused three ways): two
changes. First, before the facts, a read of a local where it holds what a field holds is read as the
field. That is a `let` or a `const` declared from the field and not assigned since, or a local given a
value that was just moved into the field. So `return result` is `return this.cached`, a borrow of
`this`, and the local's first value, read by nothing now, is dropped. The check-then-return form
(`const hit = this.cached; if (hit) return hit; ... return result;`) and `r = n` after
`this.cached = n` work the same. Second, a store into a field that was tested empty on the way (`!f`,
`f == null`, `f === undefined`, or the else branch of `if (f)`, reading the field or such a local) frees
nothing. A method whose only overwrites are such fills drops nothing, so a second `d.getQ()` no longer
ends the borrow the first one returned, and `main` prints `1 1 true`. A loop between the test and the
store, a test of another field, or a call in between that may write the field leaves the store a drop.
A call in between is judged by facts that count every fill as a drop, since a method that only fills
may still fill this field. An exported method still says what it returns (`@lends("this")`), and
an importer calling it twice needs it `@final` (48). The diagnostics name the variable now: a folded
`const` by its debug name under `--di`, and "while another read of it still borrows it" where no local
does. Regression tests `tslang/test/tester/tests/00tslang_issue_47.ts` (every model),
`own/own_lazy_cache_main.ts` with `own_lazy_cache_lib.ts` (an exported class in a namespace, imported,
every model), and `own/own_err_fill_after_call.ts`, `own_err_fill_in_loop.ts`, `own_err_fill_other_field.ts`,
`own_err_fill_after_fill_call.ts`.

48 (in an exported class that isn't `@final`, any `this.method()` may overwrite any field): a method can
be `@final` now, as a class can. No subclass may override it ("method 'x' overrides the @final method of
'Base'"), and it cannot be abstract. A call of it runs that method: directly when its class is defined in
the module, through the vtable with the one callee known when it is imported. The `-shared` declaration
printer writes `@final`, so an importer knows it too. Under `-mm=own` the call goes by the method's own
facts: `this.factor()` reading a number drops nothing, and a borrow of `this.items[1]` held across it
compiles. The error for a borrow ended by such a call has a note: "'Base.factor' may be overridden in
another module by a method that does; '@final' on the method, or on its class, says it is not". Also
fixed: the store into a field whose type holds no block was still counted a drop when the body was
checked for drops (`!releasesNothing(...) && a || b` read as `(... && a) || b`). Regression tests
`tslang/test/tester/tests/00tslang_issue_48.ts` (every model), `own/own_final_method_main.ts` with
`own_final_method_lib.ts` (a `@final` method of an imported class), `own/own_err_overridable_call.ts`,
`tests/final_method_override_error.ts` and `final_method_abstract_error.ts`.

Still open from 48:
- Whole-program facts: a method that no module of the program overrides is still taken as overridable
  when its class is exported. `--own-facts` reads the facts of the modules a module imports, not of the
  ones that import it, so the base's module can't know its subclasses. `@final` is the way to say it.
- `@final` on an accessor (`get`/`set`) works now, as on a method: a direct call, an error for an
  override ("accessor 'x' overrides the @final accessor of 'Base'"), and the declaration printer writes
  it.
- The other errors seen in powder_sandbox's own build were item 49 (fixed). The one in `levels.ts`, `'x' is
  moved inside a loop but was made outside it` (the same `box` array kept by several `LevelStroke`s),
  is the game's to fix.

Item 49, three `-mm=own` errors powder_sandbox reduced on 2026-11-14 from what was left in its build,
is fixed. Before the fix, `startA`, `startB` and `moveAside` below were refused:

```ts
class Level { name: string = "a"; }
export class App {
    level: Level | null = null;
    other: Level | null = null;
    read(n: string): Level | null { return n == "" ? null : new Level(); }
    startA(n: string) { const l = this.read(n); if (l) { this.level = l; } }
    startB(n: string) { const l = this.read(n); this.level = l; }
    moveAside() { this.other = this.level; this.level = null; }
}
```

- `startA` ("moved here on some paths only"): a value moved on some ways only is fine when, on every
  other way to its release, a branch tested it empty (`if (l)`, `if (!l) return;`, `l !== null`). There
  it holds nothing, so the release frees nothing and goes (`emptyWhereNotMoved`). `if (l && n == "y")`
  is still an error.
- `startB` ("takes a second reference"): `this.read(n)` goes through the vtable, since another module
  may override `read`. The result of such a call is the caller's when no method of the family says it
  borrows (`@lends`, `@borrows`, `@record`, a `this` result) and each one defined here returns its own.
  An override in another module may not say it borrows when the method it overrides does not. Under
  own that is an error now: "method 'read' returns a borrow ..., but the method of 'Base' it overrides
  does not".
- `moveAside` ("borrows a field and cannot be stored"): a field read stored into another place and the
  source field refilled right after is a move, as the exchange `[a, this.f] = [this.f, a]` (spec 77)
  already was. Between the store and the refill there may be only reads, constants and rc's counts of
  `null`. A call in between is still an error.

Regression tests `tslang/test/tester/tests/00tslang_issue_49.ts` (every model),
`own/own_err_moved_when_tested_other.ts`, `own_err_field_moved_call_between.ts` and
`own_err_override_lends.ts`. powder_sandbox's exchange workaround still works.

Items 50 and 51, found on 2026-11-14 by compiling every module of powder_sandbox's `-mm=own` build
(native_ts_graphics fc6fe6c as its submodule) with the compiler from main, are fixed.

50 (a value of the function's own moved into a field, then used: `const layout =
device.createPipelineLayoutFromDesc(desc); this.layout = layout; if (!layout) return false;
device.createPipelineFromDesc(layout, ...)`, in native_ts_graphics' `core/nri_donut_blit.ts`, was
refused, "'this value' is used here after its value was moved"). Item 47 read a local as the field it
was moved into, but a `const` given a call's result is that result itself, with no local to read. Now
each use the store dominates reads the field instead, read right after the store, so it is a borrow
of the field. The checks then follow that borrow through the calls in between with their facts. This
also covers a store inside an `if`, and an object (`new Layout()`) stored into a `Layout | null`
field, which is read back narrowed. A string stored into a field is a copy and a `Shared<T>` handle a
count, so those stay the function's own, as before. A call in between that may overwrite the field is
still an error, now "borrows a field but is used here after it may be released or overwritten".
`nri_donut_blit.ts` compiles under own now. It also closes a hole: with a value moved into a field of
a parameter (`h.c = c`), and `h` then given away into a field that is cleared, alpha127 took the later
`c.x` and printed 0 from freed memory. That is refused now. Still open: a value merged from a `?:`
(`const l = ok ? make() : null; this.f = l; use(l)`) stays "merged from several branches", as before.

51 (`--own-facts=<file>` naming a file that does not exist was passed over in silence): every call into
that module was then taken to release or overwrite anything, so the errors that followed read as the
program's own. powder_sandbox's `build-own` had 27 of the 58 facts objects its compile lines name
(`own-facts/core/donut.ts.obj` was never built there). With them all built, the sweep's own failures
went from 210 files to 149, and "borrows a field but is used here after it may be released or
overwritten" from 724 errors to 275. A missing file still gives no facts, but tslang warns now:
"--own-facts: '<file>' does not exist; calls into its module are taken to release or overwrite
anything".

Regression tests `tslang/test/tester/tests/00tslang_issue_50.ts` (every model),
`own/own_err_moved_field_overwritten.ts`, `own/own_err_moved_param_field_dropped.ts` and the ctest
`test-own-facts-missing-file`. `own/own_err_read_after_move_global.ts` is still refused at all three
sites, now as a borrow of the field that the global's assignment ends.

52 (a value of the function's own pushed into an array that is a field, then used: `this.textures.push(
texture); this.srgb.push(srgb); this.views.push(texture.createView(...))` in `core/nri_donut_textures.ts`,
refused "used here after its value was moved") is fixed the same way. After the push the value is the
array's last element, so each use the push dominates reads `array[newLength - 1]`, read right after the
push: a borrow of the element. Pushing into other arrays, or more into the same one, keeps it. A `pop`,
an element stored, the array replaced, or a call that may do any of these between the push and the use
is still refused ("borrows an element but is used here after it may be released or overwritten"). In
the sweep, push-then-use sites went from 91 to 72. The rest move the value into a constructor whose
result is pushed (`this.queuedFrames.push(new QueuedFrame(viewSet, ...))`, then `viewSet` used): 53. Regression tests `tslang/test/tester/tests/00tslang_issue_52.ts`
(every model) and `own/own_err_pushed_then_popped.ts`.

53 (a value of the function's own given to a constructor that keeps it in a field, then used: each
NRI example's `main`, `const sample = NriSample.create(); const app = new App(sample);
sample.addExampleOption(...)`, and `this.queuedFrames.push(new QueuedFrame(viewSet, ...)); viewSet.update(...)`,
refused "used here after its value was moved") is fixed. A constructor defined in the module keeps a
parameter when it stores it, as it is, into a field nothing else in it writes, and `this` does not
escape it: it is only read for its fields, or has an array field pushed into or popped. After the call,
each use of the value reads that field of the object. If the object was moved on, the read goes through
the place it was moved to last, as 50 and 52 made it (`this.queuedFrames[newLength - 1].viewSet`). A
borrow of a field, so a method that overwrites it, the object given away, or the object replaced before
the use is still refused. A constructor that writes the field again, or passes `this` on (a method
call, `super(...)`), keeps the old error. In the sweep, use-after-move errors went from 178 to 94: those
of a constructor's argument from 93 to 11, and push-then-use from 72 to 13. Six examples compile under
own now (`bokeh`, `calibrated_timestamps`, `shader_debugprintf`, `simple_compute`, `simple_hdr`,
`simple_pbr`), 143 of the 276 sources still fail. Regression tests `tslang/test/tester/tests/00tslang_issue_53.ts`
(every model) and `own/own_err_kept_by_constructor_reset.ts`.

54 (a value merged from a `?:` whose branches each bring a fresh value or null: `const pipeline =
loaded ? device.createPipelineFromDesc(layout, desc) : null; ... this.blitPipelines.push(pipeline)` in
`core/nri_donut_textures.ts`, refused "is merged from several branches; -mm=own cannot prove which one
owns it yet") is fixed. Such a merge is the function's own when every way into it brings a fresh value
that nothing else uses (a call's result, a `new`, another such merge) or null or undefined: rc gave it the
one reference it holds. It moves like the call's result itself, and 50, 52 and 53 follow it after the
move. In a loop's body each iteration's merge is a value of its own. Two checks took a block argument,
which has no defining op, for a value made before the loop: the release at the end of one iteration
looked like a second release of the next iteration's value, and a move in the body looked like a move
of a value made outside the loop. They now take the argument as made where its block starts. A merge at
a loop's head (a value carried from one iteration into the next) is not covered. A merge that takes a
local (`c ? local : null`), a merge moved twice, and one made before a loop and moved in it are still
refused. `nri_donut_textures.ts` is left with one error, which is its own: `new MipBlit(texture, ...)`
keeps `this.textures[t]` in a second object, a second owner (an index would do). Merge errors left in
the sweep: 3. Regression tests `tslang/test/tester/tests/00tslang_issue_54.ts` (every model) and
`own/own_err_merge_moved_twice.ts`.

55 (an array literal of fields: `const directions = [this.dir, this.dir, this.right, this.right, this.up,
this.up]` in `core/donut_first_person_camera.ts`, `const pipelines = [this.a, this.b, this.c]` and
`for (const p of [this.a, this.b])` in the examples, refused "borrows a field and cannot be stored") is
fixed. An array literal that is only read (its length and its elements), held in a local or used as it
is, and whose elements are each a read out of a field or an element or a parameter, borrows its
elements while it is used, as the array a rest parameter is given already did (spec 49). Its release
gives back the array and not the elements, so the same field twice is fine. Every use of the array,
and of what is read out of it, is a use of each element: a field written, or a call that may write
it, before such a use is still refused, "borrows a field but is used here after it may be released
or overwritten". So is an array written, stored, returned, captured or given to a call, one with a
fresh element, or an element read out of it and stored. Spec section 80 has the rules. In the sweep,
array literals of fields refused went from 18 to 1 (`this.options = [...]` stores it: refused, rightly),
and `donut_first_person_camera.ts` is left with the two errors of a `let` that borrows a field and is
given a new array (`let dir = this.dir; ... dir = rotateAround(...)`). Regression tests
`tslang/test/tester/tests/00tslang_issue_55.ts` (every model, and as an exe) and
`own/own_err_borrowing_array_field_reset.ts`.

56 (a `let` given only fields: `let layout = this.computeLayout; ... if (this.fp16Enabled) { layout =
this.computeLayoutFP16; ... }` in `examples/nri/16bit_arithmetic_nri.ts`, refused "borrows a field and
cannot be assigned; -mm=own cannot prove that yet") is fixed. An owning `let` whose declaration and
every assignment give it a read of a field or an element, or a parameter, borrows each value it is
given until the next assignment. Each read's uses are the `let`'s reads its own store reaches before
another assignment, and the assignments end it, so a field written once the `let` holds another field
is fine, in a loop too. rc's references for the `let` (its retain, its releases, and the retain for each
assignment) go. Still refused: the field the `let` holds written (or a call that may write it) before a
use, the `let` stored or returned, and a fresh value assigned (`dir = rotateAround(...)` in
`core/donut_first_person_camera.ts`, which would make the `let` own its value on some ways and borrow
it on others). One shape is refused that is sound: in a loop that reads the `let`, assigns it another
field and then writes the first, a read after the loop is taken to see the first field. In the sweep
the only `cannot be assigned` errors left are the camera's two. Regression tests
`tslang/test/tester/tests/00tslang_issue_56.ts` (every model, and as an exe) and
`own/own_err_let_borrows_reassigned_overwritten.ts`.

57 (a `let` that only holds parameters: `let src = order; let dst = scratch; ... const swap = src; src =
dst; dst = swap;` in `core/octomap_common.ts`'s exported `sortDescending`, refused "a parameter, kept
here, but -mm=own cannot make the callers of 'sortDescending' give it up") is fixed. Even `let s = a;
return s.length;` was refused: MLIRGen makes every `let` an owning local, so storing a parameter into
one kept the parameter. A `let` whose declaration and every assignment give it a parameter (read out of
a slot nothing assigns) or the value of another such `let` holds the caller's values for the whole
call, so before the facts the signature pass makes it a plain local that owns nothing and drops rc's
counting for it. What is done with its reads is then what is done with the parameter: stored into a
field or a global, captured, or given a fresh value (`s = [9]`) it is still refused. A `Shared<T>`
handle (counted under own) and a string (copied) are left as they were. `octomap_common.ts` compiles under
own now, and with it `examples/render_octomap.ts`. Six of ntg's `core/` modules are left that do not.
Regression tests `tslang/test/tester/tests/00tslang_issue_57.ts` (every model, and as an exe) and
`own/own_err_parameter_let_stored.ts`.

58 (a `const` declared from a merge of owning locals: `const selected = x2py2gez2pw2 ? (x2gey2 ? tensor0 :
tensor1) : (z2gew2 ? tensor2 : tensor3)` in `core/directx_math.ts`'s quaternion-from-matrix, refused "is
merged from several branches; -mm=own cannot prove which one owns it yet") is fixed. Even `const s = c ?
t0 : t1` was refused: each local keeps its value, and is released at its scope's end, so neither can move
into the merge. The `const` (or `let`) now borrows each of them, as a `let` given one local while that
local is still used already did: rc's retain and release for it go, and each local's own releases and
assignments end the borrow. One of them assigned or moved while the `const` is still used, or the
`const` stored, returned or assigned, is still refused. null or undefined may be one of the ways in. A
merge made in a loop's body was also refused, "moved inside a loop but was made outside it" (the merge's
own declaration was taken as reached again around the loop before the merge is remade), which also hit
54's `const p = c ? mk() : null` in a loop. With `directx_math.ts` compiling, its facts object exists, so
`core/simple_camera.ts` compiles too (it had been refused for calling into a module with no facts), and
the sweep went from 141 failing sources to 126. Regression tests `tslang/test/tester/tests/00tslang_issue_58.ts`
(every model) and `own/own_err_merged_locals_overwritten.ts`, `own/own_err_merged_locals_moved.ts`.

59 (a value moved into a field of an object read out of an array, then used: `const queuedFrame =
this.queuedFrames[i]; ... queuedFrame.aabbAttributeBuffer = buffer; const view = buffer ?
buffer.createStructuredView(0) : null;` in `examples/nri/procedural_geometry_nri.ts`, refused "used here
after its value was moved") is fixed. 50 and 52 follow a value moved into a field, or pushed into a field
array, by reading the field right after the move; they took only an object read out of a local or
parameter nothing assigns. A `const` given an element or a field is no local by then (it is the read
itself), so `frame.view = view` with `const frame = frames[i]` was left a move. The read right after the
move now takes an object read out of an element or a field too: nothing runs between, so it is the same
object, and the checks then follow the borrow through that chain. The element or the field it hangs from
replaced, popped or reset before a use is still refused. In the
sweep, "used here after its value was moved" went from 49 sites to 37, and `core/ray_tracing_extended_scene.ts`'s
own errors are left (its `@keeps` and `Shared<T>`, below). One site of this shape is left,
`procedural_geometry_nri.ts:425` (`!aabbAttributeView` after the `||` chain), which a reduced copy of it
does not reproduce. Regression tests `tslang/test/tester/tests/00tslang_issue_59.ts` (every model) and
`own/own_err_element_field_moved_replaced.ts`.

60 (the error at a borrow a call ends did not say why, when the callee is a function defined here: `const
name = this.name; for (...) this.props[i].tick(this.gameTime); ... name` in `core/rtxpt_game.ts`, "borrows a
field but is used here after it may be released or overwritten", with only "it may be released or
overwritten here" at the call) is fixed. `Prop.tick` calls `setNodeTransform`, which calls
`world.setGameNodeTransform(...)`, a method of the exported `GameWorld`. Another module may extend
`GameWorld` and override it with a method that destroys anything, so `Prop.tick` may too, and the borrows
of `this.props[i]` and `this.name` end at it. 48 named the method only when the call itself was the
virtual one. The signature pass now finds, for a call whose callees may destroy anything, the overridable
call they reach, breadth first through calls that may destroy anything, and the error notes it: "'Prop.tick'
calls 'setNodeTransform', which calls 'GameWorld.setGameNodeTransform'; that method may be overridden in
another module by a method that does; '@final' on the method, or on its class, says it is not". Nothing
compiles that did not: it is the diagnostic. The note is given only when that method is the one
reason (no other overridable method, unknown or imported callee, or body that destroys anything on the
way), so `@final` on it is enough. In the sweep, 48 errors carry it, beside the 51 at a direct call of
such a method. They name native_ts_graphics' `@final`s to add: `MsvcRandom` (41, through the
examples' `randomFloat`) and `GameWorld` (7). Regression tests
`tslang/test/tester/tests/00tslang_issue_60.ts` (every model; the shape with `@final`, which compiles) and
`own/own_err_overridable_reached.ts` (the note).

61 (a module or namespace array nothing assigns, read where a borrow cannot go: `[CUBE_VERTICES,
PLANE_VERTICES, CUBE_VERTICES]` and `[[MAT_RED, MAT_GREEN, ...], [MAT_GREY]]` in
`examples/nri/ray_tracing_reflection_nri.ts`, `return GREEN;` in `examples/nri/collision_nri.ts`, `const p
= CAMERA_POSITION;` held across calls in `examples/nri/async_compute_bloom_nri.ts`, refused "the value read
from 'X' is kept where its end at an assignment of 'X' cannot be checked" and "borrows the global 'X' and
cannot be stored, returned or captured") is fixed. Even a plain `const A = [1.0, 2.0]` at the top level was
refused: only a global of a class type could be immortal (own spec 25). An array global is now one on the
same terms - not exported, written once by its initializer, given a fresh array, nothing, or another
immortal global's value - and a read of it owns nothing and may go anywhere. The block made immortal is the
array's header, which every array release tests first, so neither its data nor its elements are released
through it; an element overwritten, popped or spliced out still frees the old element, and a borrow of
`G[i]` stays bounded. A global pushed to, or written in any other way through its address, stays mortal.
The initializing store marks the header only when it is mortal: a constant array's header is in read-only
data, immortal already. Spec 25 has the amendment. In the sweep, 126 sources failing went to 119
(`ptlas.ts`, `ray_tracing_reflection.ts`, `collision.ts`, `nvrhi_direct.ts`, `async_compute_bloom.ts`,
`msaa.ts`, `texture_compression_comparison.ts` now compile, and `ray_tracing_reflection_nri.ts` went from
29 errors to 1), and errors from 907 to 811. Regression tests `tslang/test/tester/tests/00tslang_issue_61.ts`
(every model) and `own/own_err_immortal_array_element_overwritten.ts`.

62 (an array changed in place through a local that borrows the field holding it: `const model =
this.refractionModel; ... model.push(0.0);` in `examples/ray_tracing_extended.ts` and its NRI ports, refused
"borrows a field and cannot be stored, returned or captured" at the push) is fixed. `push`, `pop`,
`unshift`, `shift` and `splice` take the slot that holds the array's header, and the borrowing local's
slot was taken to keep the field's value. The header is the field's array, so the change is the field's
too: each such op through the local is now a use of the borrow, bounded as a read is. An element read
before a `pop` and used after, the field given another array before a use, and the local returned or
stored are still refused. A field that holds a null header (never given an array) gets one in the local's
slot only, which nothing frees: that is `undefined` in TypeScript terms, where `.push` throws, as the
array spec's ruling T4-D accepts for a parameter. In the sweep, 4 errors less (807); the two NRI ports
keep others. Regression tests `tslang/test/tester/tests/00tslang_issue_62.ts` (every model) and
`own/own_err_borrowed_array_pushed_returned.ts`.

63 (a lazy getter that fills an empty field with a C call in between: `if (!this.cachedVertexBuffer)
this.cachedVertexBuffer = new NriBuffer(Nri_GetGltfMeshVertexBuffer(this.handle));` in `core/nri.ts`, which
made every call of such a getter one that "may release or overwrite" whatever the caller borrowed) is fixed.
A fill of an empty field (TSLANG 47) may not run a call that can reach the field in between, and an imported
C function could reach anything. One declared `@nodrops`, with no export, import or dllimport, not variadic,
and given and returning only numbers, strings, `null`/`undefined` or `@cpp` handles (or unions of them), runs
no function of this module and gets nothing that leads to one: the fill now holds across it. Regression test
`tslang/test/tester/tests/00tslang_issue_63.ts` (every model; it also holds 64 and 65).

64 (a value moved into a field of an object already pushed into an array, then used: `const queuedFrame =
new QueuedFrame(); this.queuedFrames.push(queuedFrame); queuedFrame.pbrView = pbrView; if (!pbrView) ...` in
`examples/nri/simple_pbr_nri.ts`, refused "is used here after its value was moved") is fixed. TSLANG 50's
borrow of the field the value went into is now also taken after the push, and for a value built by `new`
in one arm of a `?:` (its construction's own uses are not uses of the merge). Popping or splicing the
object out before the use, or overwriting the field, is still refused. Regression test
`own/own_err_pushed_object_field_popped.ts`.

65 (a value moved into an element, or a local's value moved into a field or an element, then used: `let
pipeline = null; if (loaded) pipeline = create(); this.baselinePipelines[logicOp] = pipeline; return
pipeline;` in `examples/nri/logic_op_dynamic_state_nri.ts`) is fixed. The use borrows the element (or field)
from the store on, as TSLANG 50 does for a fresh value, when the local's slot is not assigned again after the
store and the store does not run again (a loop). The element reset or popped before the use, or the store in
a loop, is still refused. Regression test `own/own_err_local_stored_element_reset.ts`.

With 63-65 in, errors in the sweep went from 807 to 767 (21 sources have fewer; `simple_pbr_nri.ts` 6 to 1),
and no source has more; 119 sources still fail, each for other errors as well.

66 (a `let` borrowing a field, then given a fresh value on some paths: `let dir = this.dir; ... if (yaw != 0.0
|| pitch != 0.0) { dir = rotateAround(...); } ... this.dir = normalize(dir);` in
`core/donut_first_person_camera.ts`'s `animate`, refused "borrows a field and cannot be assigned; -mm=own
cannot prove that yet") is fixed. The `let` would own its value on one path and borrow it on the other, and
-mm=own has no flag to release it on one only. Where the dry run finds such a `let` (initialized from a
place, only read, assigned and released), each fresh value it is given goes into a slot of its own, released
where the `let` is (an assignment of another fresh value, the scope's end, an unwind), and the `let` is given
a borrow of that slot's value; it then holds only borrows, as TSLANG 56's do. Each borrow of the slot is
checked like any other: it may not be stored, returned or captured, nor used after the slot's value is
released. The `let` given its field on the other paths stays bounded by the field, so the field overwritten
before a use is still refused. In the sweep, the camera module compiles, and with its facts the examples that
call it do too: errors went from 767 to 664, and failing sources from 119 to 118 (`async_compute_bloom_nri.ts`,
`multi_draw_indirect_nri.ts` and `terrain_tessellation_nri.ts` 8 errors to 1, both `ray_tracing_*_nri.ts`
ports 25 to 6), and no source has more. Regression tests `tslang/test/tester/tests/00tslang_issue_66.ts`
(every model; loops, a class-typed `let`, a throw), `own/own_err_split_let_stored.ts` and
`own/own_err_split_let_field_overwritten.ts`.

67 (a value pushed only where it is not null, then used: `const view = constantBuffer.createConstantView(offset,
size); if (view) this.constantViews.push(view); return view;` in `examples/nri/rt_shadows_nri.ts` and its
NRI ports, refused "is used here after its value was moved") is fixed. TSLANG 52 made each use the push
dominates a read of the array's last element; one after the paths meet again is now one too. The block the
test's false edge goes to, entered otherwise only from blocks the push dominates, takes the element read
back on those paths and null on the false edge, and the uses there read it. A function returning it returns
a borrow of `this`, as one returning the element read does. The array popped before the use is still
refused, and so is a push under a test that is not the value's own truth (`if (view && keep)`), where the
value is not null on the other path. A `let` pushed this way (`let pipeline = null; if (loaded) pipeline =
create(); if (pipeline) this.pipelines.push(pipeline); return pipeline;` in `msaa_nri.ts`) is not handled
yet. In the sweep, 6 errors less; their sources fail for others too. Regression tests
`tslang/test/tester/tests/00tslang_issue_67.ts` (every model; it holds 68 too) and
`own/own_err_pushed_or_null_popped.ts`.

68 (a union of string literals read as a `string`: `ImGui.text(`primitiveUnderestimation: ${this.yesNo(
p[3])}`)` in `examples/conservative_rasterization.ts`, with `yesNo` returning `value != 0.0 ? "yes" : "no"`,
refused "takes a second reference") is fixed. `"no" | "yes"` is one pointer, like a `string`, and the cast
between them lowers to nothing, but -mm=own took it for a conversion making a new string, so the call
result it was cast from was released while the string was used. A cast between a string, a string literal
and a union of them (and null) that needs no tag is now a view. In the sweep, 2 sources more compile
(`conservative_rasterization.ts`, `multi_draw_indirect.ts`) and 20 errors less. A union of string literals
also allows only its literals now, under every model: `let a: "no" | "yes" = "maybe"`, a `string` given
to it, or `a = "maybe"` after `let a = yesNo(v)`, compiled and held "maybe"; each is now refused, as in
TypeScript (TS2322, and TS2345 for an argument). The check found a parameter property typed by its
default left unwidened: `constructor(private source: string, private flags = "")` in the default library's
`RegExp` made a field of type `""`; it is a `string` now, as a declared field and a parameter already were.
Regression tests
`tslang/test/tester/tests/assign_string_literal_union_error.ts`, `assign_string_to_literal_union_error.ts`
and `argument_string_literal_union_error.ts`.

69 (a value moved on every path, by a different move on each: `const node = new SceneNode(...); if
(SCENE_ORDER[o] == TESSELLATED) this.tessNodes.push(node); else { ... this.baselineNodes.push(node); }` in
`examples/nri/extended_dynamic_state2_nri.ts`, refused "is moved here on some paths only; -mm=own cannot
release it on the others yet") is fixed. Each move was judged alone: the value's release where the paths meet
was past this move on some paths, and on the others past a move it did not look at. A release no path reaches
from the value's making without passing one of its moves is past a move on each, and goes. A path that moves
it nowhere (`if ... else if ...` with no `else`) is still refused, and so is a use after the moves. In the
sweep, 4 errors less. Regression tests `tslang/test/tester/tests/00tslang_issue_69.ts` (every model; it holds
70 too) and `own/own_err_moved_on_two_of_three_paths.ts`.

70 (a constant global of a string literal's type given to a call that keeps it: `this.loadTexture(device,
FLAME_TEXTURE_PATH)` in `examples/nri/ray_tracing_extended_nri.ts`, with `const FLAME_TEXTURE_PATH = "..."` in
its namespace and `loadTexture` pushing the path into `textureNames`, refused "is given to ..., which keeps
it, but -mm=own cannot move it here: this function does not own it") is fixed. The global is a constant whose
type is the literal, so it holds that literal and nothing else, in read-only data: given to a keeper, it is the
literal itself, as `"..."` written in place is. 69 exposed it: `loadTexture` is now found to keep its path. In the sweep, the 2 errors it gave
(in `ray_tracing_extended_nri.ts` and `ray_tracing_invocation_reorder_nri.ts`) are gone.

71 (a value narrowed from `T | null`, kept by a constructor, then used: `const texture =
device.createStorageTexture(...); if (!texture) return null; const target = new Target(texture, ...);
target.readView = texture.createView(format);` in `examples/nri/raytracing_ao_nri.ts`, refused "is used here
after its value was moved") is fixed. TSLANG 53 reads the field the constructor keeps a value in for each use
after the call, but only for a value of the field's type, or a class value kept in a union field. A value of
the field's class or null, narrowed by the test before, is now one too: the field read back is widened to
the value's type again. The object popped before the use is still refused. Regression test
`tslang/test/tester/tests/00tslang_issue_71.ts` (every model; it holds 72 too).

72 (a local, rather than a value, kept by a constructor or pushed, then read: `const t: number[] = []; for (let
i = 0; i < 16; i++) t.push(transform[i]); const node = new SceneNode(..., t, ...); ... this.movingX = t[12];`
in a loop in `examples/nri/extended_dynamic_state2_nri.ts`, and `let pipeline = ...; ...;
this.pipelines.push(pipeline); return pipeline.n;`) is fixed. TSLANG 65's reads of a local after a store (the
local assigned nowhere after it, its address taken nowhere) now serve the constructor and the push too. A
store in a loop is taken when the local is declared in that loop, so each time round starts it afresh, and an
array filled in place through the local is taken when the filling comes before the move. The local pushed to
after the move, declared outside the loop that moves it, or its new owner popped before the read, is still
refused. In the sweep, 71 and 72 take 20 errors, and `extended_dynamic_state2.ts` compiles. Regression test
`own/own_err_kept_local_array_popped.ts`.

73 (a private field read off a parameter or `this` around a call: `constructor(private sample: NriSample)`, then
`this.sample.run(() => ...)` and a later `this.sample` read in the examples' `App` classes, refused "borrows a
field but is used here after it may be released or overwritten") is fixed. Any call could have overwritten the
field (a callback that writes it, say), so the read before it was not trusted after it. Only the class's own
methods can write a private field, and they are all in the module. So the signature pass now marks each private
field that nothing but its class's constructor writes, and a call no longer drops such a field read off a
parameter (`this` included) the function does not assign. The object is the caller's, so the field lives as long
as the call does. A private field a method writes, called back from the call, is still refused. Regression tests
`tslang/test/tester/tests/00tslang_issue_73.ts` (every model) and `own/own_err_private_field_written_callback.ts`.
`own/own_err_declared_drops.ts` lends a public field now: the private one it had is a field 73 keeps.

74 (a value of `cond ? null : f(...)`, with `f` another module's, pushed where it is not null, then used: `const
texture = textureName == "" ? null : device.loadTexture(...); if (!texture) ... else { this.materialTextures.push(texture);
this.materialViews.push(texture.createView(texture.getFormat())); }` in `examples/nri/mouse_cursor_nri.ts`, refused
"is used here after its value was moved") is fixed. TSLANG 65's push step reads such a value back from the array
when the value is the function's own. A call's result alone counts, whoever defines the function. A merge of a
call's result and null (TSLANG 54) counted only when the call's result was known to be fresh, which, for another
module's function, the signature pass decides only after this step. A merge now counts when each way into it
brings null or a call's result, as the call alone does. Regression test: TSLANG 74's step in
`tslang/test/tester/own-facts-objects.cmake` (`tests/own_facts_tex_lib.ts`, `tests/own_facts_tex_main.ts`).

75 (a call through a constant that holds a function: `const f = Math.fround;` in `examples/clustered_geometry.ts`,
then `centroidY = f(centroidY + c.positions[v])` with `c` a cluster of a parameter's mesh, refused "borrows a field
but is used here after it may be released or overwritten" at each later use of `c`) is fixed. The signature pass
took a call through a global's value as one through a callback it knows nothing of, so the call could destroy
anything. A global that no store writes, and that its initializer gives a function's symbol, now resolves to that
function: `f(...)` is a call of `Math.fround`, which destroys nothing. A global something assigns is still
unknown. In the sweep, `clustered_geometry.ts` goes from 44 errors to 16. Regression tests
`tslang/test/tester/tests/00tslang_issue_75.ts` (every model) and `own/own_err_const_alias_drops.ts`.

76 (a parameter kept by a constructor that also gives `this` to a call: `constructor(private sample: NriSample) {
this.reset(); }`, then `sample.addExampleOption(...)` after `new App(sample)` in `examples/nri/simple_compute_nri.ts`,
refused "is used here after its value was moved") is fixed. TSLANG 53 reads the field a constructor keeps a
parameter in, for each use after the call. It took only a constructor that reads `this` for its fields and
nothing else, so that no call it makes can reach the field. A constructor that gives `this` to a call now keeps a
parameter too, when the field is private and no other function writes it: as in 73, no call can then overwrite
it. A field a method writes is still refused. In the sweep, the 4 examples with this shape lose their errors.
Regression tests `tslang/test/tester/tests/00tslang_issue_76.ts` (every model) and `own/own_err_kept_field_rebound.ts`.

77 (an array kept by a constructor, then filled through the local: `const sceneSets: NriDescriptorSet[] = [];
this.queuedFrames.push(new QueuedFrame(sceneView, sceneSets)); for (...) sceneSets.push(sceneSet);` in
`examples/nri/msaa_nri.ts`, refused "is used here after its value was moved") is fixed. An array is a reference,
so after the move the local is the object's array, and each push fills it. TSLANG 72 took an array filled in
place before the move. One changed in place after it now changes the field, reached the way the other uses after
the move are: through the object, or the place it moved to last. The object popped before the push is still
refused. Regression tests `tslang/test/tester/tests/00tslang_issue_77.ts` (every model) and
`own/own_err_kept_array_filled_after_pop.ts`.

78 (a value pushed on a field of a local object, then used: `const queuedFrame = new QueuedFrame(...); for (...) {
const faceAllocator = ...; queuedFrame.faceAllocators.push(faceAllocator);
queuedFrame.faceCommandBuffers.push(faceAllocator.createCommandBuffer()); }` in
`examples/nri/threaded_rendering_nri.ts`, refused "is used here after its value was moved") is fixed. TSLANG 65's
push step and the field store step read the value back through the object right after the move. They took an
object read out of a local or an element, but not a `const` holding the object itself, which is a value and
needs no read. That value now serves, as it is. The element popped before the use is still refused.
Regression tests `tslang/test/tester/tests/00tslang_issue_78.ts` (every model) and
`own/own_err_local_object_field_popped.ts`.

79 (a local pushed only where it is not null, then used: `let pipeline: NriPipeline | null = null; if (loaded)
pipeline = device.createPipelineFromDesc(...); if (pipeline) { this.pipelineKeys.push(key);
this.pipelines.push(pipeline); } return pipeline;` in `examples/nri/msaa_nri.ts`, refused "is used here after its
value was moved") is fixed. TSLANG 67 did this for a value. For a local, the test and the uses are reads of
their own. Now, when nothing assigns the local from the test on, each read where the paths join takes the merge:
the element read back after the push, or the null the test found. The local is then not read after the join, so
its release goes on the edge that did not push it. The method returns a borrow of `this` on every way, so the
early `return this.pipelines[i]` is no longer refused either. `msaa_nri.ts` and
`dynamic_multisample_rasterization_nri.ts` compile. Regression tests
`tslang/test/tester/tests/00tslang_issue_79.ts` (every model) and `own/own_err_pushed_let_popped.ts`.

80 (a parameter a constructor stores into a field, then reads in a loop that calls: `constructor(world:
GameWorld, public name: string, ...)`, then `${name}` in `core/rtxpt_game.ts`, refused "is used here after its
value was moved") is fixed. The pairing of a local with the field it was stored into reads the field for
the local's later reads, but it stopped at any call. A constructor that reads `this` only to reach its fields
never gives the object to anyone, so no call can write them, and the pairing now sees past calls there. Copying
the parameter at the store instead leaks it: the caller gave it up, and nothing releases the copy's original
(`own/own_err_string_kept_param_known.ts`). A first fix that copied grew 3M calls to 2.7 GB, and was backed out
before #856. A constructor that gives `this` away is still refused. Regression tests
`tslang/test/tester/tests/00tslang_issue_80.ts` (every model) and `own/own_err_escaped_this_field_read.ts`.

81 (a value moved on some paths only: `const controller = new LightController(); if (controller.read(...)) {
...; this.lights.push(controller); }` in `core/rtxpt_game.ts`, refused "is moved here on some paths only; -mm=own
cannot release it on the others yet") is fixed. Spec 30 placed a local's release on the edges that did not move
it; a value, like the `const` here, had no such rule. Now a value one taker moves gets its release on each edge
into what follows the move, from where the value is held and was not moved. The release after the move goes. A
use after the paths join is still refused, a copyable value's read included (`own_string_copy_branch.ts`). Regression tests `tslang/test/tester/tests/00tslang_issue_81.ts`
(every model) and `own/own_err_moved_on_some_paths_used.ts`.

84 (a borrow of a call's result across a store of an unrelated type: `const texture = this.swapChain.texture(index);
...; this.recordTriangle(commandBuffer); ...; commandBuffer.textureBarrier(texture, ...)` in
`examples/nri/color_write_enable_nri.ts`, where `recordTriangle` stores a pipeline into
`this.trianglePipelines[enabled]`, refused "borrows the result of 'NriSwapChain.texture' but is used here after
it may be released or overwritten") is fixed. A call's result borrows a place not known, so any release was
taken to free it. A release frees the released value and what it holds, which is what its type can hold. When
that type cannot hold the borrowed value's type, it cannot free it: the release, or a call whose drops are
known stores, is no longer taken to. Only an object whose class is known is followed: a @final class's, or a
declared @cpp one, which holds nothing of ours (MLIRGen lists them on the module, OWN_FINAL_CLASSES_ATTR_NAME).
Any other class may be a subclass's object, holding anything, and an interface, `any` or a function may hold
anything too. The borrowed value too must be of a @final class, or an array. In the sweep,
`color_write_enable_nri.ts` and `logic_op_dynamic_state_nri.ts` compile. Regression tests
`tslang/test/tester/tests/00tslang_issue_84.ts` (every model) and `own/own_err_borrowed_result_holder_stored.ts`.

For native_ts_graphics, from the same sweep (its code, not the compiler):
- `core/ray_tracing_extended_scene.ts`: the exported classes' constructors keep their array parameters
  (`this.origin = origin`), which an exported constructor has to say: `@keeps("origin", "direction")`
  on `FlameParticleGenerator`'s and `@keeps("vertices", "triangles")` on `Model`'s. `@keeps` on a
  constructor works (checked). Its `this.particles.filter(...)` needs `Shared<FlameParticle>` elements.
- `examples/dynamic_primitive_clipping.ts:223`, `desc.setDepthClip(clip)` with `clip` the loop's `int`,
  fails under every model since #846 (a number is not a `boolean` argument, TS2345). alpha127 still
  takes it; the next release won't. `desc.setDepthClip(clip != 0)` fixes it.
- `core/nri_donut_textures.ts`: `MipBlit` held the texture it blits, and `recordBlits` read it back after the
  blits array kept it, so the module and the 9 examples importing it failed under `-mm=own` (199 errors). A
  `MipBlit` now holds the texture's index into `this.textures`. `MsvcRandom` (`core/random.ts`) and `GameWorld`
  (`core/rtxpt_game.ts`) are now `@final`: a class that can be derived from may have its methods overridden, so
  `-mm=own` cannot read their facts (about 48 errors). So are `AtomOrbitals`, `OrbitalsPass`, `Pose` and
  `RtxptGame` now; a class that can be derived from turns each call of its methods into one that may
  destroy anything.
- The `*_nri.ts` examples' `QueuedFrame` kept `this.frames.commandBuffer(i)`, a borrow of the frame ring's,
  which a field cannot hold: it is read from `this.frames` where it is recorded (16 examples). The
  `atom_orbitals` examples set two fields to one `Orbital` (`this.orbitalA = this.orbitalB`), a second owner:
  each field gets its own. These are on ntg branch `own-sweep-fixes`.
- Left in `examples/clustered_geometry.ts` (ntg's call): `Cluster.clone` and `SceneObject.copy` share the
  vertex, index and material arrays on purpose ("which nothing changes"). That gives each array two owners,
  so either the copies copy them (`slice()`) or the arrays are `Shared`. The `add` closure keeps its mesh and
  position and returns the object it pushed; a function, or the caller pushing, would say that.
- Under `-mm=own`, 2 `core/` modules do not compile yet (`ray_tracing_extended_scene.ts`, `rtxpt_game.ts`), and
  nor do many examples, which #310 did not port: with every facts object built, 50-74 in and the changes above,
  48 of the 276 sources fail (194 errors). The largest groups left are "borrows a field but is used here after
  it may be released or overwritten" (36), "borrows a field and cannot be stored, returned or captured" (31),
  "used here after its value was moved" (27), and an element borrowed and stored (21) or used after it may be
  released (20).

82 (fixed; found 2026-11-14 on head 5aa1cdb0; native_ts_graphics `examples/raytracing_mini_engine.ts`
`computeAO`): a closure that calls another closure it captured makes its method count as freeing
anything, so every caller's borrows end at the call. Run with `-mm=own --emit=obj`:

```typescript
@nodrops declare function C_Rcp(x: number): number;
class Holder { n: number = 1.0; }
class P {
    holder: Holder | null = new Holder();
    values: number[] = [2.0];
    calc(h: Holder | null, x: number): number {
        const f = (y: number): number => Math.fround(y);
        const rcp = (y: number): number => {
            const r = f(C_Rcp(y));      // f, a captured closure, called from a closure
            return f(r + r);
        };
        return h ? rcp(x * h.n) : 0.0;
    }
    run() {
        const a = this.values;
        this.calc(this.holder, 1.0);    // "it may be released or overwritten here"
        print(a[0]);                    // error: 'this value' borrows a field but is used here after ...
    }
}
function main() { new P().run(); }
```

Refused twice: `print(a[0])` ("borrows a field but is used here after it may be released or
overwritten", note at the `this.calc` call) and the call itself (`this.holder` lent to it). The C call
plays no part: `const r = f(y)` in `rcp` gives the same two errors, while `rcp` calling `Math.fround`
directly instead of `f` gives none, and so does calling `f` straight from `calc` (outside a closure).
So a call through a function value a closure captured seems to be taken as an unknown call (may free
anything) when the enclosing method's facts are worked out, although the value is a local closure whose
body is known. Suspect (not checked in the source yet): how OwnershipSignaturePass's per-function
FieldDrops resolves a call made inside a lambda through a captured value. Workaround in
native_ts_graphics: the helper is a module-level function (`refinedReciprocal`) instead of a closure
capturing `f`.

Fixed: the signature pass took a call through a field of the closure's capture box (its first argument) as a
callback it knows nothing of. A captured function value now resolves to the functions stored into that
field, when every store into it, in each function that makes or calls the closure with a box, is a
function's symbol, and the closures given the box write none of its fields. `const f = ...` captured by
`rcp` is then a call of `f`'s function. A captured `let` another closure assigns stays unknown. Regression
tests `tslang/test/tester/tests/00tslang_issue_82.ts` (every model) and
`own/own_err_closure_reassigned_captured_call.ts`.

83 (fixed; found 2026-11-14 on head 5aa1cdb0; native_ts_graphics `hdr_display`, `hdr_display_nri`):
under `-mm=own`, a row of a module-level array of arrays bound to a local (`const meta = POOL[idx];`)
is freed at the local's end, so the second call frees it again: the program hangs (here, and in
hdr_display at exit in mimalloc's `mi_process_done` -> `_mi_page_free_collect`, a cyclic free list) or
crashes. It compiles without a diagnostic. Built with `--emit=exe` (gc prints `750000 1000 500`):

```typescript
const POOL = [[1000.0, 0.001], [500.0, 0.002]];
function build(idx: int): number {
    const meta = POOL[idx];          // dropped at the end of build: a double free from the 2nd call
    return meta[0];
}
function main() {
    let total = 0.0;
    for (let i = 0; i < 1000; i++) {
        total += build(i % 2);
    }
    print(total, POOL[0][0], POOL[1][0]);
}
```

`-mm=own`: hangs. Fine: `return POOL[idx][0];` (no local), the same with the array a local passed in
(`pool[idx]`), and a global string array's element bound to a local (`const name = NAMES[idx]`). So a
borrow of a global's element seems to be taken as an owned value when the element is itself an object.
Workaround in native_ts_graphics: `.slice()` (a copy) at the three such sites (`hdr_display.ts`,
`hdr_display_nri.ts`, `raytracing_mini_engine.ts`; native_ts_graphics #322). 61 (immortal array globals,
after 5aa1cdb0) may cover this; not rechecked on a head with it.

Fixed (it still hung after 61 and #856): `POOL` passed for a keeper, an array global whose elements are
immortal, because its literal's elements are constants. Its rows are constant arrays, though, each a block
the literal makes, and nothing makes them immortal. A keeper's element read is no borrow, so `meta` owned the
row and freed it on each call. A global whose literal has constant arrays (or tuples) for elements is no
keeper now, so a read of its row is a borrow, and the local releases nothing. Scalar and string constants
still make one. Regression test `tslang/test/tester/tests/00tslang_issue_83.ts` (every model; built with the
old compiler, its exe hangs at exit).

native_ts_graphics: with #321 and #322 (2026-11-14, on 5aa1cdb0 with workarounds) every source compiles
under `-mm=own` and its ctest passes 525/525, as under gc and rc.

85 (fixed; found 2026-11-14 on pre-alpha128 (0c8439cb); native_ts_graphics `examples/sakura_forest_ser.ts`
`LIBRARY_EXPORTS`): under `-mm=own`, a module-level `const` whose initializer concatenates three or more
string literals is refused at the initializer. Run with `-mm=own --emit=jit` (gc prints
`Raygen,Miss,Hit,Group`):

```typescript
namespace M {
    const EXPORTS = "Raygen,Miss," +
        "Hit," + "Group";       // error: the initializer of global 'M.EXPORTS' takes a second reference;
                                // -mm=own cannot prove a move or a borrow here yet
    function show(names: string) {
        print(names);
    }
    export function run() {
        show(EXPORTS);
    }
}
function main() {
    M.run();
}
```

Fine: two literals (`"Raygen,Miss," + "HitGroup"`), and one literal. So `a + b` seems to be taken as a
fresh string the global owns, while `(a + b) + c` reads the inner sum as a second reference to a value
something else holds. The whole initializer is a constant (literals only), so it could fold to one
literal, as 70 made a literal-typed constant global its literal. Suspect (not checked in the source yet):
the global-initializer check in the own pass treating the nested `+`'s left operand as a borrow. Workaround
in native_ts_graphics: a function returning the concatenation (`libraryExports()`).

Fixed: the inner sum is a temporary the initializer makes, reads and gives back itself. rc's retain on it is
the reference it arrives with and its release gives that back, as in a function body, but the check that
moves a value into a global took every retain in the initializer for the global's: one retain, a release,
and no move into the result read as a second reference. A fresh string retained once and released once,
that only a concatenation reads, now keeps its release and loses the retain. Four literals, a number in
the chain, a template literal, and other globals, a field or an element read into the chain work too
(a call that returns a borrow of its argument is still refused). Still open: a string
built this way and given to a constructor in a global's initializer (`const H = new Holder("x" + "y")`,
two literals are enough) is refused the same way, as before. Regression test
`tslang/test/tester/tests/00tslang_issue_85.ts` (every model).
