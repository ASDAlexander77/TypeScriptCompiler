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
(a call that returns a borrow of its argument is still refused). So does a sum given to a call, which
was refused even of two strings (`const H = new Holder("x" + "y")`): the global-initializer step after the
facts moves it into a callee that keeps it, so its release goes, and leaves the release after a callee
that only reads it. A callee whose result borrows it is refused, "holds the result of 'id', which borrows
its argument". A chain returning `this` on a new object (`const B = new Bag().add("m").add("n" + "o")`,
with `add(s: string): this` or a `return this` the body infers) was refused too, "holds the result of
'Bag.add', which borrows its argument", with a plain literal as well. A function body takes the object
over (39, spec 78), but only for calls in functions. The global step now does the same: a call whose result
is its `this` itself, on an object the initializer made (a `new`, or the call before it in the chain)
and gives to nothing after but its releases, returns that object, the global's own, and its releases go.
A chain on another global's object (`const A = SHARED.add("q")`) is still refused: `A` would be a second
owner. So is a function returning its parameter (`const c = id(new V(1))`, #638): only a method's
`this` is taken over. Regression tests `tslang/test/tester/tests/00tslang_issue_85.ts` (every model),
`own/own_err_global_concat_borrowed.ts` and `own/own_err_global_chain_other_global.ts`.

## TypeScript compiler tests sweep (2026-11-17)

Items 86-125 were found on 2026-11-17 by compiling TypeScript's own compiler tests
(`testdata/tests/cases/compiler` of the tsc repo, 7,107 files) with a release build of 03aaab9d, as
`tslang --emit=obj <test> -o x.o` (no `main`: each test's code is module-level). Only the 3,597 tests
tsc compiles without errors (no `.errors.txt` baseline) count here. tslang compiles 1,575 of them.
731 more are multi-file tests (`// @filename:` sections that tsc's harness splits) and aren't tslang
bugs. That leaves 1,291 single-file failures. The items below are the ones worth fixing, grouped by
cause, with the number of tests each one fails. Names in "Tests:" are files in that folder. The
by-design limits seen in the sweep are not listed: overloads of a declared function, `import x = A.B`
and `import x = require()`, non-constant `obj[key]`, generic interface methods with `T[]`
parameters, recursive type aliases, `null` into `string`, `@noCheck` tests and JSX tests that need
`/.lib/react16.d.ts`. Crash stacks came from the debug build (built 2026-11-13). Repros marked "run"
were run as written with `tslang --emit=jit` on the release build. The others are the tsc test file
itself, which failed in the sweep.

The first batch of fixes (86-91, 98, 99, 103, 109, 111, 113, 116, 117, 122) took the tsc-clean tests that
compile from 1,575 to 1,633, and the crashes and hangs over all 7,107 tests from 361 to 287, with no test that
compiled before failing after. Each fixed item says what is still open in it.

The second batch (92, 95, 96, 110, 114, 115, 124, and part of 121) took them to 1,659, and the crashes and hangs to
258. One test that compiled before fails after, `classMemberInitializerScoping.ts`, which tsc rejects too: its
constructor, with an untyped parameter, was an implicit generic that was never generated, so `this.y = ''` into a
`number` went unchecked.

### Crashes (137 tests) and hangs (3)

86 (a `var` in a namespace, used by a statement in that namespace, crashes the compiler): 8+ tests.
Tests: `compoundVarDecl1.ts`, `structural1.ts`, `sourceMapValidationModule.ts`,
`namedFunctionExpressionInModule.ts`, `internalAliasUninitializedModule*.ts`. Run, crashes:

```typescript
namespace M { var a = 10; a++; }
function main() { print("done."); }
```

Also crashes: `namespace M { export interface I { salt: number; } export function f(i: I) {} f({ salt: 2 }); }`
and `namespace M { var x = function bar(a: number) { return a; }; print(x(1)); }`.

Fixed: a namespace at the module level holds back its code statements and runs them from a global
constructor of its own, as an imported module's top level runs (`generateModuleInitCode`); its declarations
stay where they were. All 8 tests compile. A nested namespace's code runs before its parent's code, not where
the nested namespace stands in it.

87 (an arrow with an untyped parameter in an object literal crashes in `getNameOfFunction`, from
`registerGenericFunctionLike`): 5 tests. Tests: `commentsBeforeFunctionExpression1.ts`,
`commentsAfterFunctionExpression1.ts`, `contextualComputedNonBindablePropertyType.ts`,
`mappedTypeContextualTypesApplied.ts`, `mappedTypeGenericIndexedAccess.ts`. Run, crashes (with
`(a: number) => 0` it prints `0`):

```typescript
var v = {
    f: (a) => 0
};
function main() { print(v.f(1)); }
```

Fixed: a function that is a field of an object literal (`{ f: (a) => 0 }`, `{ f(a) {} }`, `{ f: function (a) {} }`)
takes an untyped parameter as `any`, with the warning a global variable's function gets. It was made generic,
which a field can't hold: it crashed, or failed with "can't find type by name: TGenParam_a". A parameter the
receiver types (`const o: I = { f: (a) => a + 1 }`) still takes the receiver's type. The 5 tests no longer
crash; 4 stop at other items (106, 119, 120, and a field read of an `any`).

88 (`for (a in expr)` with `expr: any`): 3 crashes in `mlirGen` from `processStatements`, plus 7
tests that fail with "Can't resolve property 'length' of type any". In an async function it crashes
too (`es5-asyncFunctionForInStatements.ts`). Tests: `forInStatement5.ts`, `forInStatement6.ts`,
`forStatementInnerComments.ts`, `forInStatement1.ts`, `controlFlowArrays.ts`,
`es5-asyncFunctionHoisting.ts`, `yieldInForInInDownlevelGenerator.ts`. Run, crashes:

```typescript
var a: any;
var expr: any;
for (a in expr) {
}
function main() { print("done."); }
```

Fixed: `for (k in obj)` assigns a variable declared before the loop (it read the initializer as a
declaration list and crashed). `for...in` over an `any` or `unknown` is an error that says so: "'for...in'
over a value of type 'any' is not supported: its keys are not known; give it an object, a tuple or an array
type", in place of "Can't resolve property 'length' of type any". Regression test `for_in_any_error.ts`.

89 (`<Foo> new Object()`, with `Foo` an empty interface, crashes in
`mlirGenObjectVirtualTableDefinitionForInterface` from `castObjectToInterface`): `castNewObjectBug.ts`.
Run, crashes:

```typescript
interface Foo { }
var xx = <Foo> new Object();
function main() { print("done."); }
```

Fixed: `new Object()` with no arguments is `{}`; `Object` names the type `object`, which has no constructor
of its own (`ts.New` of it failed verification). An object with no known fields cast to an interface reads as an
empty tuple, so an interface with no members takes it, and any other is told what it misses.

90 (a module-level object binding pattern with a default crashes): 8 tests, all
`sourceMapValidationDestructuring*DefaultValues.ts` and `*ObjectBindingPattern3/4.ts`, and
`noImplicitAnyDestructuringVarDeclaration2.ts`, `parameterInitializerBeforeDestructuringEmit.ts`. Run,
crashes:

```typescript
var { x = 500 } = { x: 20 };
function main() { print(x); }
```

Fixed: a default applies to a field that may be missing (an optional one); one the value always has keeps its
value, as in TypeScript, where the default is then never used (`var { x = 500 } = { x: 20 }` cast the field to an
optional type it isn't, and crashed). `sourceMapValidationDestructuringForOfObjectBindingPattern.ts` still
crashes: a `for...of` over an array of objects with a nested pattern, not looked at yet.

91 (a destructuring assignment whose value is another assignment crashes in `mlirGenSaveLogicOneItem`):
3 tests. Tests: `destructuringTempOccursAfterPrologue.ts`,
`classFieldsNamedEvaluationDestructuringAssignment.ts`, `shorthandPropertyAssignmentNoCrash.ts`. Run,
crashes:

```typescript
function test(p: any) {
    p = { prop: p } = p;
}
function main() { print("done."); }
```

Fixed (the crash): a field the value hasn't got is an error, where its missing value was saved. The three
tests now fail with "Can't resolve property 'prop' of type any": reading a field of an `any` is not supported.

92 (tagged templates): a tagged template in a function crashes (`taggedTemplatesInDifferentScopes.ts`,
`taggedTemplateNestedInvalidEscape.ts`). A tag typed `(strs: string[], ...)` given `any[]` fails with
"cast from array type '!ts.array<!ts.string>' to array type '!ts.array<!ts.any>' is not supported"
(7 tests: `taggedTemplateStringsWithMultilineTemplate*.ts`, `taggedTemplateStringsWithWhitespaceEscapes*.ts`,
`arrayConcat2.ts`, `arrayConstructors1.ts`, `regexpOptionalCaptureStringArray.ts`). Run, crashes:

```typescript
export function tag(parts: TemplateStringsArray, ...values: any[]) {
    return parts[0];
}
function foo() {
    tag `foo`;
}
function main() { foo(); print("done."); }
```

Fixed (the crash): a tagged template is a call like any other, so a tag with a rest parameter gets its rest array.
It was given one operand for its two inputs, whatever the scope, when there were no `${}` parts, and the verifier
read past the operands. The verifier checks the count first now. Still open: a tag typed `(strs: string[], ...)`
given `any[]`.

93 (the JSX parser crashes): 21 `.tsx` tests. These fail in the parser, before any JSX support is
needed. An access violation in `tagNamesAreEquivalent` (the closing tag of a nested element or a
fragment), and fail-fast in `parseJsxElementOrSelfClosingElementOrFragment` (a tag after a `@jsx`
pragma comment, a nested child). Tests: `jsxNestedIndentation.tsx`, `jsxPartialSpread.tsx`,
`jsxPragmaAfterTags.tsx`, `jsxPragmaAfterTokenSameLine.tsx`, `contextuallyTypedJsxAttribute2.tsx`,
`expandoContextualTypes.tsx`, `contextuallyTypedJsxChildren*.tsx`, `jsxFunctionTypeChildren*.tsx`,
`jsxEmptyExpressionNotCountedAsChild.tsx`, `jsxIntrinsicUnions.tsx`, `booleanLiteralsContextuallyTypedFromUnion.tsx`.
Smallest:

```typescript
declare var React: any;
declare function Child(props: { children?: any }): any;
function Test() {
    return <Child>
        <Child>
            <Child></Child>
        </Child>
    </Child>
}
```

94 (generic function specialization crashes in `instantiateSpecializedFunction`): 11 tests, 6 from
`cast`/`verifyFunctionCastRules` and 5 from `mlirGenSpecialized`/`mlirGenCallExpression`. Tests:
`nestedGenericSpreadInference.ts`, `returnInfiniteIntersection.ts`, `genericFunctionsNotContextSensitive.ts`,
`conditionalPromptInference.ts`, `genericCallInferenceConditionalType2.ts`,
`genericTypeParameterEquivalence2.ts`, `mixinIntersectionIsValidbaseType.ts`,
`genericIndexedAccessMethodIntersectionCanBeAccessed.ts`, `declarationEmitArrowFunctionNoRenaming.ts`.
Smallest:

```typescript
declare function wrap<X>(x: X): { x: X };
declare function call<A extends unknown[], T>(x: { x: (...args: A) => T }, ...args: A): T;
const leak = call(wrap(<T>(x: T) => x), 1);
```

95 (a property read in a contextually typed callback crashes in
`mlirGenPropertyAccessExpressionBaseLogic`): 5 tests. Tests: `objectBindingPatternContextuallyTypesArgument.ts`,
`destructureOfVariableSameAsShorthand.ts`, `dependentDestructuringDeferredCallback.ts`,
`regexMatchAll.ts`, `regexMatchAll-esnext.ts`. Smallest:

```typescript
declare function id<T>(x: T): T;
const { f = (x: string) => x.length } = id({ f: x => x.charAt });
```

Fixed (the crash): an object pattern whose value failed stops there; each of its fields read from no value.
The example is now an error, "Can't resolve property 'charAt' of type any", since the arrow's `x` is not
typed from the default.

96 (`instanceof` narrowing of a union that holds an intersection crashes in
`isSafeTypeTheSameAndNoNeedToCast`): 3 tests. Tests: `inKeywordAndIntersection.ts`, `narrowByInstanceof.ts`,
`controlFlowInstanceofWithSymbolHasInstance.ts`. Smallest:

```typescript
class A { a = 0 }
class B { b = 0 }
function f10(obj: A & { x: string } | B) {
    if (obj instanceof Object) {
        obj;
    }
}
```

Fixed: in the release build it was not the intersection. `Object` is only a type to tslang, and naming it as a
value failed with no error ("failed statement"). `x instanceof Object` is now decided from the type of `x`: true
for an object of any kind (each type of a union one), false for a primitive, and an error for a type that may be
either. A type name used as a value elsewhere is an error that says so, "'Object' is a type here, not a value"
(regression test `type_as_value_error.ts`).

97 (a type parameter constrained by its own type overflows the stack): 5 tests. Tests: `recursiveTypes1.ts`
(`getTypeByTypeReference` from `zipTypeParametersWithArguments`), `instantiatedBaseTypeConstraints.ts`
(`loc2Fuse`), `checkInfiniteExpansionTermination.ts`, `checkInfiniteExpansionTermination2.ts`,
`inferredReturnTypeIncorrectReuse1.ts`, and `stableTypeMapperOrdering.ts`,
`varianceProblingAndZeroOrderIndexSignatureRelationsAlign*.ts`. Smallest:

```typescript
interface Entity<T extends Entity<T>> {
   X: T;
}
interface Person<U extends Person<U>> extends Entity<U> {
    n: number;
}
interface Customer extends Person<Customer> {
    s: string;
}
```

98 (source files tslang can't read): a UTF-16 file with a BOM (`bom-utf16le.ts`, `bom-utf16be.ts`) and
invalid UTF-8 in a regex literal (`regexInvalidUtf8WithUnicodeFlag.ts`) fail-fast in `loadSourceBuf`.
A JSDoc comment whose line is 100+ spaces fail-fasts in `Parser::extractPragmas`
(`commentWithUnreasonableIndentationLevel01.ts`). UTF-16 should be converted. Invalid UTF-8 should
be a diagnostic, not an abort.

Fixed: a source file is read as UTF-16 when it starts with a UTF-16 byte order mark (FF FE or FE FF), and
otherwise as UTF-8, where a byte that is not valid UTF-8 becomes U+FFFD. A comment with no `@` is not run
through the pragma regular expressions, whose `\s*` recursed once per character over a long run of spaces until
the stack overflowed. All 4 tests compile. Regression test `00utf16_source.ts`, a UTF-16 file.

99 (a class member with a string-literal or numeric name): `static "foo" = 0` is "redefinition of
symbol named 'C.foo'", and `staticMemberWithStringAndNumberNames.ts` (`static 0 = 1`, `C['0']`, `C[0]`)
crashes. Run:

```typescript
class C {
    static "foo" = 0;               // error: redefinition of symbol named 'C.foo'
    x = C['foo'];
}
function main() { print(new C().x); }
```

Fixed: the cause was not the string-literal name. An instance field reading a static of its own class
(`x = C.foo`, `x = C["foo"]`) made the class be processed again, and the static's global was made a second time.
The global made the first time is now the one it has. A numeric static name (`static 0 = 1`) is the string of
its digits, as a property key is (it crashed); `C["0"]` reads it, but `C[0]` still looks for an indexer
("Class member '.index' can't be found").

100 (`import M = Z.M` inside a namespace also named `M`, and merged namespaces): crashes in the debug
build, "Failed to emit LLVM IR" in release. Tests: `moduleSharesNameWithImportDeclarationInsideIt.ts`,
`...It2.ts`, `...It4.ts`, `mergedNamespaceDeclarationMemberReferences.ts`, `es6ModuleFunctionDeclaration.ts`,
`importOnAliasedIdentifiers.ts`, `privacyGloImport.ts`, `collisionExportsRequireAndInternalModuleAliasInGlobalFile.ts`,
`nameCollisionWithBlockScopedVariable1.ts` (a block-scoped `let M` in namespace `M`, then `new C()`).

101 (a built-in error class used as a value, `x = RangeError`, crashes in `globalVariableAccess`):
`errorConstructorSubtypes.ts`, `discriminantOrderIndependence.ts`.

102 (other crashes; these `abort()` in the debug build, so no stack was taken): `unionTypeWithIndexedLiteralType.ts`
(`const u: U = { x: "lit" }` with `U = Idx | I | "lit"`), `typeParameterConstrainedToOuterTypeParameter2.ts`,
`recursiveTypeInference.ts`, `recursiveTypeInference2.ts`, `recursiveTypeAliasWithSpreadConditionalReturnNotCircular.ts`,
`reverseMappedTypeInferenceWidening1.ts`, `reverseMappedTypePrivateMembers.ts`, `promiseTypeInference.ts`,
`inferenceContextualReturnTypeUnion2.ts`, `lazyInheritedMemberLookup.ts`, `circularDeclaredPropertyAccess.ts`
(`MLIRTypeHelper::extendsType` from `testConstraint`), `crashInGetTextOfComputedPropertyName.ts`,
`crossProductUnionIntersectionRepeatedPropDeclarationsNoOOM1.ts`, `contextualGenericReturnToOptionalClassField.ts`,
`controlFlowPropertyDeclarations.ts`, `discriminantNarrowingCouldBeCircular.ts`,
`expandoFunctionNestedAssigmentsDeclared.ts`, `implicitAnyInCatch.ts`, `metadataOfUnionWithNull.ts`,
`narrowByClauseExpressionInSwitchTrue5.ts`, `genericFunctionsAndConditionalInference.ts` (`tryInferInterface`),
`narrowingUnionToUnion.ts` (`resolveIdentifierAsVariable`), `noUsedBeforeDefinedErrorInTypeContext.ts`
(`getInterfaceVirtualTableForObject` from `castTupleToInterface`: `{} as IThing<typeof foo>` in `foo`'s
own initializer), `stringMappingSpecialCasing.ts`, `templateLiteralInferenceSupplementarySplit.ts`,
`templateLiteralIntersection4.ts`, `superPropertyAccessInComputedPropertiesOfNestedType_ES5/ES6.ts`,
`staticFieldWithInterfaceContext.ts`, `typePartameterConstraintInstantiatedWithDefaultWhenCheckingDefault.ts`,
`unionOfEnumInference.ts`, `unusedLocalsAndParametersDeferred.ts`, `unusedLocalsAndParametersTypeAliases.ts`,
`workerSubtypeHandlerThis.ts`.

103 (the compiler doesn't finish in 60 s): `declFileEnumUsedAsValue.ts`, `declFileTypeofEnum.ts`,
`declarationEmitClassMemberNameConflict2.ts`. All three store an enum itself as a value: `enum e { a, b, c }
var x = e;`, or a field `Hello = Hello`.

Fixed (the hang): the lowering's clean-up of conversion casts replaced a value's uses with itself when two
casts took each other's result, which asserts in a debug build and never finished in a release build. An enum
named as a value is its members at compile time, with nothing a variable could hold at run time: `const x = E`
stands for it, as before, and a `var` or `let` is now an error, "variable 'x' can't hold the enum 'e' itself:
use 'const', or its members". `declarationEmitClassMemberNameConflict2.ts` (a field `Hello = Hello`) fails with
"operand #0 does not dominate this use" instead of hanging. Regression test `enum_variable_error.ts`.

### Internal IR and codegen errors (about 140 tests)

104 (a generic `declare function` called with an arrow: "'llvm.func' op external functions must have
'external' or 'extern_weak' linkage"): 33 tests. Tests: `typeInferenceWithTypeAnnotation.ts`,
`inferTupleFromBindingPattern.ts`, `assertionFunctionsCanNarrowByDiscriminant.ts`,
`asyncFunctionsAndStrictNullChecks.ts`, `bindingPatternContextualTypeDoesNotCauseWidening.ts`,
`circularMappedTypeConstraint.ts`. Run (also with `--emit=exe`):

```typescript
declare function f<T>(p: (t: T) => T): T;
function main() { f((n: number) => n); }
```

Fixed (third batch): a specialization of a `declare function` has no body, and the lowering gave every
specialization `linkonce_odr` linkage, which LLVM allows only on a definition. A body-less one now stays an
external declaration. Test: `obj-only/declared_generic_call.ts` (compiled to an object; nothing defines `f`).

105 (a closure over a module-level `let` declared in a `for` header or loop body: "'ts.Load' op using
value defined outside the region"): 21 tests. Tests: `nestedBlockScopedBindings*.ts`,
`blockScopedBindingsReassignedInLoop2/3/5.ts`, `capturedLetConstInLoop11.ts`. The same loop in a
function compiles. `nestedBlockScopedBindings7.ts`, run as a file:

```typescript
for (let x; false;) {
    () => x;
}
```

Narrowed in the second batch: at the module level, an arrow held by a `const` in the loop (`const f = () => x`)
works; held by a `let` (`let a = () => x`, as the tests do) or left unused, it fails. Not fixed yet.
Narrowed again in the third batch: no loop is needed, a module-level block is enough
(`{ let x = 1; let a = () => x; }`). The arrow is made without capturing `x`, so its body loads `.main`'s
local directly; the same block in a function, or with `const a`, captures `x`.

Fixed (fourth batch). The module's discovery pass is a dummy run of the top-level statements outside any
function, and in it the block's `x` is not a local of anything. That pass discovered the arrow and
registered its type, so where the arrow was really generated it was not discovered again: it captured
nothing. An arrow or a function expression met by that pass is no longer registered.
`asyncSuperDefaultParameters.ts` (`super` in an async arrow's default parameter) is a different failure and
is still open.

106 (a module-level arrow: "'llvm.mlir.addressof' op must reference a global defined by
'llvm.mlir.global'"): 21 tests. Only `--emit=obj` and `--emit=exe` fail. The JIT, with its cache, runs
them. Tests: `parenthesizedAsyncArrowFunction.ts` (`let foo = (async bar => bar);`),
`thisExpressionInIndexExpression.ts` (`function f() { return r => r[this]; }`),
`collisionRestParameterUnderscoreIUsage.ts`, `collisionThisExpressionAndNameResolution.ts`,
`declarationEmitIndexTypeArray.ts`, `declarationEmitTypeParameterNameInOuterScope.ts`.

Fixed (third batch). There were two causes.
- An arrow with an untyped parameter is generic in tslang. Held by a `let`/`var`, returned from a function
  with no declared return type, or set as an object literal field, it stayed generic. The name then pointed
  at a function that was never made, and a local `let` crashed. Each of these now holds one function, with
  its untyped parameters typed `any`, as TypeScript types them (`instantiateGenericFunctionWithAny`). Only
  a `const` keeps the function generic. A `var` hoisted from a nested block gets that instance's type up
  front.
- `declare var c` in a module that has code of its own was made a local of the function that code runs in
  (`.main`), so functions reading `c` named a global nobody declared. An ambient variable is always a
  global now. A second `declare var c` uses the global the first one declared.
  - In such a file, an ambient name that is also a C library function (`declare const log: (s: string) =>
    void;` in `downlevelIterationDeprecated.ts`) now collides with it, as any global by that name does.
    Before, that file compiled only because `log` was a local.

Related fixes:
- A returned closure (`return m => m + n`) was wrapped in a bound-method adapter that read its captures
  from the wrong box (`boundMethodWiderTypeAdapter` now skips a value already of the type).
- Instantiating a closure against its own type matched its captures parameter to the first declared
  parameter.
- The crash guard in `instantiateSpecializedFunction` from the `var` batch stays.

Tests: `00generic_arrow_any.ts`, `obj-only/declare_var_module_code.ts`.

Still open:
- `return r => r[this]` (`thisExpressionInIndexExpression.ts`, "not supported").
- A generic arrow returning a generic interface (`<p>(a: p): W<p> => ({ a })`) given a function type crashed
  in `castTupleToInterface`: `W<any>` has no interface info where the cast is. It is now the error "can't
  find type by name: p" (it crashed for a typed `const` before this batch too).

107 (a namespace used as a value: "failed to legalize operation 'ts.NamespaceRef'"): 18 tests.
Tests: `moduleWithTryStatement1.ts`, `structuralTypeInDeclareFileForModule.ts`, `cloduleWithRecursiveReference.ts`,
`commentsClass.ts`, `declFileModuleAssignmentInObjectLiteralProperty.ts`. Run:

```typescript
namespace M { export var x = 1; }
var m = M;                      // error: failed to legalize operation 'ts.NamespaceRef'
function main() { print(m.x); }
```

108 (a class used as a value, `this` in a static member or a class named in an arrow: "failed to legalize
operation 'ts.ClassRef' that was explicitly marked illegal"): 11 tests. Tests: `staticGetter1.ts`,
`noCollisionThisExpressionAndClassInGlobal.ts`, `abstractClassInLocalScope.ts`, `complexNarrowingWithAny.ts`,
`controlFlowForCatchAndFinally.ts`, `declFileClassWithStaticMethodReturningConstructor.ts`. Run:

```typescript
class C {
    static n = 1;
    static get x() { return this; }     // error: failed to legalize operation 'ts.ClassRef'
}
function main() { print(C.x.n); }
```

109 (a call to a function expression or arrow that is defined after it, or inside a method: "'ts.Call'
op '...' does not reference a valid function"): 6 tests. Tests: `2dArrays.ts`,
`defineVariables_useDefineForClassFields.ts` (module-level `const a = () => b()` then `const b = ...`),
`checkSuperCallBeforeThisAccessing6.ts` (`super()` to `Base.constructor`), `argumentsAsPropertyName.ts`.
Run:

```typescript
class Board {
    ships: number[] = [1, 2];
    allPositive() {
        return this.ships.every(function (val) { return val > 0; });   // error: 'ts.Call' op does not reference a valid function
    }
}
function main() { print(new Board().allPositive()); }
```

Fixed for function expressions: one whose untyped parameters take their types from where it goes (a
callback, `this.ships.every(function (val) { ... })`) is generic, as an arrow function is, and is now passed as
such, so it is instantiated for its receiver. `2dArrays.ts` and `argumentsAsPropertyName.ts` compile. Still open:
a module-level arrow calling one declared after it (`defineVariables_useDefineForClassFields.ts`) and `super()`
to a base constructor (`checkSuperCallBeforeThisAccessing6.ts`).

110 (`new` on a plain function: "'ts.New' op result #0 must be  or , but got '!ts.func<...>'"): 5 tests,
plus `classFunctionMerging.ts` and `classFunctionMerging2.ts` (a class merged with a function). Tests:
`avoid.ts`, `contextualReturnTypeOfIIFE.ts`, `anyAsReturnTypeForNewOnCall.ts`. If constructor functions
stay unsupported, this should be an error that says so. Run:

```typescript
function Point(x: number) { }
function main() { var o = new Point(3); print("done."); }
```

Fixed (the error): constructor functions stay unsupported, and `new` on a function is now the error "'new' on a
function is not supported: declare a class with a constructor" (regression test `new_function_error.ts`).

111 (`++`, `--` and compound assignment on an untyped `var x;`: "Binary operation is not supported for
type: '!llvm.ptr'"): 7 tests. Tests: `systemModule8.ts`, `assignmentCompatBug3.ts`,
`commentsArgumentsOfCallExpression2.ts`, `nestedBlockScopedBindings3.ts`, `nestedBlockScopedBindings4.ts`.
Run:

```typescript
var x;
function main() { x = 1; x++; print(x); }
```

Fixed: an `any` in arithmetic, a bitwise operator or an ordering comparison is the number it holds, or,
beside a string in `+`, the string. `++` and `--` on an `any` add or subtract one from the number it holds and
store the result back boxed; the value is a number, the old one for a postfix operator. `+` of two `any`s may
add or join, which is known only at run time, so it is an error that asks for a cast: "'+' of two values of type
'any' is not supported: cast one of them to 'number' or 'string'" (2 of the tests). Regression test
`any_plus_any_error.ts`.

112 (narrowing that leaves an `unrealized_conversion_cast`: "LLVM Translation failed for operation:
builtin.unrealized_conversion_cast"): 4 tests, `--emit=obj` and `--emit=exe`. Tests:
`correlatedLabelNullishFallback.ts`, `narrowingInCaseClauseAfterCaseClauseWithReturn.ts`,
`narrowingTypeofUndefined1.ts`, `dependentDestructuringCrossFilePosition.ts`. Smallest:

```typescript
function label(options: { label: string; alias: string } | { label: object; alias?: string }) {
    if (typeof options.label === "string") return options.alias ?? options.label;
}
```

### Common language gaps

113 (`this` at module level, in an arrow: "can't resolve name: this"): 22 tests, e.g.
`collisionThisExpressionAndVarInGlobal.ts`, `asyncArrowStaticFieldThis.ts`, `castExpressionParentheses.ts`,
`classNameReferencesInStaticElements.ts`. Module-level `this` is `undefined` in an ES module. Run:

```typescript
var f = () => this;             // error: can't resolve name: this
function main() { print("done."); }
```

Fixed: `this` where nothing gives it one (the module's top level, an arrow there, a plain function) is
`undefined`, as in an ES module and in strict mode. Inside a class it is left as it was: a static field's arrow
(`static i = async () => this.h`, `asyncArrowStaticFieldThis.ts`, `classNameReferencesInStaticElements.ts`)
still fails, since `this` there is the class.

114 (`arguments`: "can't resolve name: arguments"): 18 tests, e.g. `argsInScope.ts`,
`argumentsObjectIterator01_ES6.ts`, `argumentsObjectIterator03_ES6.ts`, `modularizeLibrary_UsingES5LibAndES6ArrayLib.ts`.
Run:

```typescript
function f(x: number, y: number) {
    return arguments.length;    // error: can't resolve name: arguments
}
function main() { print(f(1, 2)); }
```

Fixed for functions with plain parameters: `arguments` is an `any[]` of the values of the function's parameters
(a method's object is not one of them). Still errors that say so: `arguments` in an arrow (it would be the
enclosing function's, regression test `arguments_arrow_error.ts`), and in a function with a rest or destructured
parameter. More arguments than parameters are not seen.

115 (`super` in an arrow in a method, or in a default parameter: "can't resolve name: super"): 6 tests.
Tests: `superAccessInFatArrow1.ts`, `captureSuperPropertyAccessInSuperCall01.ts`
(`super(() => { return super.blah(); })`), `asyncSuperDefaultParameters.ts` (`async (b = super.m()) => b`),
`collisionThisExpressionAndLocalVarWithSuperExperssion.ts`.

Fixed for an arrow in a method: `super` there is the base of the method's `this`, which the arrow captures
(`() => super.m()`). Still open: `super` in a default parameter, and in an arrow given to `super(...)`.

116 (a parameter's default that reads an earlier parameter: "can't resolve name: x"): 5 tests,
`optionalParamReferencingOtherParams1.ts`, `inferParameterWithMethodCallInitializer.ts`. Run:

```typescript
function strange(x: number, y = x * 2) { return y; }   // error: can't resolve name: x
function main() { print(strange(2)); }
```

Fixed: each parameter is known by name while the parameters after it are read, so a default can read the
ones before it (it is evaluated for its type before the function exists). `inferParameterWithMethodCallInitializer.ts`
still fails, on a return type `typeof a`.

117 (`toString` on a number literal, which is an `s32`: "Can't resolve property 'toString' of type s32"):
9 tests, e.g. `toStringOnPrimitives.ts`, `destructuringInVariableDeclarations1/2/5/6.ts`
(`export let { toString } = 1;`). Run:

```typescript
function main() { print(1..toString()); }   // error: Can't resolve property 'toString' of type s32
```

Fixed: an integer value with no member of the name has the members of `number` (the value converted), so
`1..toString()`, `i.toFixed(1)` and `(255).toString(16)` work. Still open: a method taken as a value from a
number, `export let { toString } = 1;` ("failed to legalize operation 'ts.CreateExtensionFunction'").

118 (redeclarations JavaScript allows: "redefinition of symbol named ..."): 47 tests. `var v = 1; var v = 2;`
(`duplicateVariableDeclaration1.ts`, `redeclarationOfVarWithGenericType.ts`), and a class declared again
in a nested block, which should shadow the outer one: "redefinition of symbol named 'C..instanceOf'"
(`classDeclarationBlockScoping1.ts`, `classDeclarationBlockScoping2.ts`). Run:

```typescript
class C { }
{
    class C { }                 // error: redefinition of symbol named 'C..instanceOf'
}
function main() { print("done."); }
```

Found in the second batch: most of the 48 tests are not redeclarations at all. A `var` in a function is kept
in a module global named by the variable alone (`VariableDeclarationInfo::detectFlags`: `isGlobal` for every
`var`), so any two functions, methods or constructors that each declare `var x` fail with "redefinition of symbol
named 'x'". Worse, as that is one static, a recursive function's `var` is shared by every call (and thread):

```typescript
function f(n: number): number { var x = n; if (n > 0) f(n - 1); return x; }
function main() { print(f(3)); }   // prints 0, not 3
```

Giving the global a per-function name is not enough: the variable is looked up in the namespace's globals map by
its name alone (`registerVariableDeclaration`), first one wins, so the second function read the first one's
global. The fix is a `var` that lives in its function (hoisted to the function's top), which changes what owns
it under rc and own (a global slot is a root; a local is released at the function's exit): a design decision of
its own.

Fixed (third batch, the MDN `var` rules): a `var` written in a function, method, accessor, constructor, arrow,
generator or async function is a local of that function. Before the body is generated, every `var` in it (in
blocks, `if`, loops and their headers, `try`/`catch`/`finally`, `switch`, labels; not in nested functions) is
declared once at the function's top, so it is hoisted, block-free (`if (c) { var x = 1 } return x`), the same
variable when declared again (`var v = 1; var v = 2;` assigns), and one per call (`f(3)` above prints 3). A name
that is also a parameter is that parameter. Its type is the first declaration's type, else its initializer's
widened type, else `any`. Under rc and own the value it holds is released when the function returns, like a
`let`'s. Tests: `00var_function_scope.ts` (gc/rc/none), `own_var_function_scope.ts` (all four models);
`00globals3.ts` read `main`'s `var a` from another function, which JavaScript does not allow, so its `a` is now
declared at module level.

Still open:
- `var v = 1; var v = 2;` at module or namespace level is still "redefinition of symbol" (module `var`s stay
  globals; the second one's assignment would have to run in the module's init code). Fixed in the fourth
  batch:
  - A module `var` declared again with no value is nothing, and with a value it is an assignment.
  - A repeated `var` with a value counts as module code (`hasGlobalCode`), so the file's top level runs in
    its entry function, in order: `var v = 1; var w = v + 1; var v = 5;` gives v 5 and w 2.
  - Test: `00module_var_redeclared.ts`.
  - Still open: a namespace `var` declared in two `for` headers (`duplicateVariablesByScope.ts` compiles, but
    a function of the namespace reading it does not resolve it), and names that clash between namespace
    members (`importInTypePosition.ts`, `declarationEmitNameConflicts3.ts`).
- A `var` read before its declaration gets its hoisted slot only when its type is known before the body runs
  (a type annotation, or a first declaration inside a block, loop or `if`); `x = 7; var x = one();` in one body is "can't resolve
  name: x".
- A `var` holding a destructuring pattern (`var [a, b] = t;`) is block-scoped like a `let`.
- A generic function value in a `let` or `var` with no declared type (`var f = <U>(u?: U) => { };`) crashed
  in `instantiateSpecializedFunction` (it did for `let` before; `methodContainingLocalFunction.ts` reached it
  through `var`). It now fails without crashing ("can't find type by name: U", as for 119); keeping such a value
  generic, as a `const` does, is still open.
- -mm=own: a closure that escapes in a loop and captures a `var` the loop shares is refused ("'i' is moved
  inside a loop but was made outside it"), as it is for a `let` declared before the loop.

119 (a type parameter used in a mapped type or a callback type of a generic declaration: "can't find type
by name: K" / "T" / "V" / "A"): about 50 tests. Tests: `reverseMappedContravariantInference.ts`
(`declare function conforms<T>(source: { [K in keyof T]: (val: T[K]) => boolean }): ...`),
`comparisonAnonymousMappedTypes.ts`, `conditionalFunctionUnionInference.ts`, `typeInferenceFixEarly.ts`,
`contextualTypingTwoInstancesOfSameTypeParameter.ts`, `inferentialTypingWithFunctionType.ts`,
`inferentialTypingWithFunctionType2.ts`, `contextualSigInstantiationRestParams.ts`, `enumIndexer.ts`.

120 ("not all types could be inferred"): 29 tests, e.g. `simpleArrowFunctionParameterReferencedInObjectLiteral1.ts`,
`arrayFlatMap.ts`, `contextualSignatureInstantiation1.ts`. Run:

```typescript
function main() {
    const r = [1].map(() => [2].map(p => ({ X: p })));   // error: not all types could be inferred
    print(r.length);
}
```

121 ("failed statement" with no other error): 67 tests. The error that caused it is lost, so these
can't be grouped further. Fix the reporting first, then sort them. Tests:
`restElementWithNumberPropertyName.ts`, `asConstReadonlyTupleInferenceThroughNestedGenericCall.ts`,
`asyncYieldStarContextualType.ts`, `autoAsiForStaticsInClassDeclaration.ts`. Run:

```typescript
const { 0: a, ...b } = [0, 1, 2];   // error: failed statement
function main() { print(a); }
```

Second batch, 2026-11-17: making the silent failures speak (a probe build) sorted the 65 tests into the items
above and these causes. Fixed here:
- a numeric name in an object pattern over an array, `const { 1: b } = [10, 20, 30]`, reads the element; the rest
  of an array in an object pattern (`...b` above) is an error that says so.
- a built-in type name used as a value (`Object.freeze(...)`, `x = Object`, 11 tests): "'Object' is a type here,
  not a value". `Object.keys` and the like are not in the default library.
- an untyped `static x` and an untyped parameter property, `constructor(public x, public y)`, are `any`s, as an
  untyped instance field is (`autoAsiForStaticsInClassDeclaration.ts`, `classExpressionWithStaticProperties2.ts`,
  `topLevel.ts`). A constructor's untyped parameters are `any`s; it was made an implicit generic, which no
  constructor can be.

Still silent: an object given to an interface with an index signature (`const m: IMap = {}`, `objectIndexer.ts`),
a variable whose type has a generic method (`var a1: { fn<T>(x: T): T }`), and about 15 others.

122 (an untyped rest parameter with a binding pattern: "can't resolve type for binding pattern, provide
default initializer"): 9 tests, `restParameterWithBindingPattern1.ts` (`function a(...{a, b}) { }`),
`restParameterWithBindingPattern2.ts` (`function a(...[a, b]) { }`), `commentsAfterSpread.ts`,
`asyncFunctionTempVariableScoping.ts`. An untyped rest parameter is `any[]`, so its pattern can take
`any` (with `...[x, y]: number[]` it works).

Fixed for array patterns: an untyped rest parameter is an `any[]`, and `...[a, b]` reads that
(`restParameterWithBindingPattern2.ts`). An untyped object pattern (`...{ a, b }`, or a plain parameter
`({ foo, bar })`) would read fields of an `any`, which is not supported, and still fails.

123 (calling a value of type `any`: "not a function to call"): part of 47 tests, e.g.
`castFunctionExpressionShouldBeParenthesized.ts` (`(function a() { } as any)().foo()`), `anonterface.ts`,
`callbackAssignedNonNull.ts`. The rest are `Symbol(...)`/`Symbol.for`, which are default-library
work (see 38).

124 (a computed property name in a class: "not supported 'Computed Property Name' expression"): 9 tests,
`declarationEmitMultipleComputedNamesSameDomain.ts` (`[x] = 10;`), `declarationEmitComputedNameWithQuestionToken.ts`
(`[dataSomething]?() {}`), `classExpressionWithComputedPropertyInLoop.ts`, `capturedLetConstInLoop13.ts`.

Fixed for constant names: a module-level `const` that holds a literal is a constant name (it is read from its
global, whose type still says what it holds), and a computed field initializes its field. Before, the
constructor's field initializers refused any computed name and the failure was dropped, so every field of such a
class kept its zero value, with no error: `class C { ["k"] = 10; j = 5 }` gave `k` and `j` 0. A name that is not a
constant (a `let`, a parameter, an `unique symbol`) is still an error, which now says what a name must be.

125 (missing default-library types and type arguments): "generic type Iterable can't be found" (6:
`forAwaitForUnion.ts`, `mapConstructor.ts`, `contextualTypeIterableUnions.ts`), "generic type Generator
can't be found" (4: `contextualTypeOnYield1.ts`, `contextualTypeOnYield2.ts`), "can't resolve name:
Uint8ClampedArray" (5: `typedArrays.ts`, `typedArraysSubarray.ts`, `arrayToLocaleString*.ts`), and
`new WeakMap()` / `new SS()` with no type arguments, which tsc infers or defaults: "Generic class
'WeakMap' is missing type arguments" (5: `defaultWeakMapInferredPropertyAssignment.ts`,
`genericObjectCreationWithoutTypeArgs.ts`, `inheritanceOfGenericConstructorMethod1.ts`).

126 (a parameter of type `never`: "'Never' can't be used as parameter type"): 18 tests, e.g.
`discriminantsAndNullOrUndefined.ts`, `enumPropertyExhaustiveNullableReceiver.ts` (`function assertNever(x:
never)`). Fixed (third batch): a parameter written `never` takes whatever it is given, as `any`. Only a value
narrowed to nothing reaches it. A `never` that tslang computed itself (from an intersection it can't express,
`callableWeakMetadata.ts`, `jsEmitIntersectionProperty.ts`) is still the error. Under -mm=own, passing a
borrowed parameter on to it is refused ("kept here"), as for any `any` parameter. Test: `00import_equals_value.ts`.

127 (`import b = a.member` of an enum, a function or a variable: "this import-equals target is not
supported"): 25 tests, e.g. `internalAliasEnum*.ts`, `es6ModuleInternalImport.ts`,
`exportImportCanSubstituteConstEnumForValue.ts`. Fixed (third batch): `b` names that reference, generated
again where `b` is used. As a type it is the referenced value's type, so an enum alias works as one. Test:
`00import_equals_value.ts`.

108 fixed (third batch, USER DECISION 11-17: a class referenced as a value is resolved as an interface-like
value whose members are the class's own `new`, statics and `instanceof`; namespaces unchanged). A class named
as a value with no declared type is now the class's value, `typeof C`. That is the `ClassValueType` that
`let k: typeof C = C` already had (#742, #889), with statics, `new` and `instanceof` through the vtable of the
class it holds. This applies when the class is:
- held by a `let`/`var` (module, local or hoisted);
- returned (`return C`, `return this` in a static method, `() => C`);
- put in an object literal field;
- boxed as `any`/`unknown`/`Object`;
- compared (`token === C`).

A `const` still names the class itself. Test: `00class_value_inferred.ts` (gc/rc/none, and own JIT).

Still open:
- `this` in a static method is the class that declares the method, also when the method is called through a
  subclass (`Derived.me()` returns `Base`). In JavaScript it is `Derived`; static methods are not given the
  class they were called through. It was already the case before this change. Fixed by 162.
- `class X extends k` where `k` is a class value ("its class is known at run time only").

Not fixed in the third batch (design calls):
- 107 (a namespace as a run-time value): left as it is, by the decision above.

128 (found in the fourth batch: a closure made in `for (let i ...)` saw the one `i` the loop had moved past;
every closure returned 3, where JavaScript gives each iteration its own `i`, 0 1 2). Fixed: when the body
makes a closure and nothing in it assigns the loop's variables, the body's `i` is a variable of its own,
made from the loop's at the start of each iteration (CreatePerIterationEnvironment). This applies to
value-type variables (numbers, booleans...). Under -mm=own the loop now compiles: each iteration's `i` is
made inside the loop and moved into the closure, which was "'i' is moved inside a loop but was made outside
it" before.
- Still open:
  - A body that assigns the variable keeps the one binding. Its value would have to be copied back, which
    `continue` skips.
  - A string or object loop variable keeps the one binding.
- Test: `00for_let_per_iteration.ts`.

129 (a parameter in the types after it: `function f1(a: number, b: typeof a)`, `(a: number): typeof a`, `{ a:
typeof a }`; "can't resolve name: a"). Fixed (fourth batch): the parameters are declared, by type, for a
later parameter's `typeof` type, as they were for its default, and while the return type is evaluated.
Tests: `unusedParameterUsedInTypeOf.ts`, `newLineInTypeofInstantiation.ts`,
`indirectTypeParameterReferences.ts`; `00module_var_redeclared.ts`.

130 (`e satisfies T`: "unsupported expression"): `parenthesizedSatisfiesExpressionWithComments.ts`,
`spreadParameterTupleType.ts`, `declarationEmitAsConstSatisfiesNonReadonlyResult.ts`. Fixed (fifth batch):
`e satisfies T` is `e` and keeps its own type. When `e` is a function, it takes its untyped parameters' types
from `T`. `spreadParameterTupleType.ts` then reaches a crash of its own, which happens without `satisfies`
too: a rest parameter typed as a spread tuple, `function fn(...args: [...A, ...A])`.

131 (`Object` as a value: "'Object' is a type here, not a value", 27 tests). Fixed (fifth batch), unless the
program declares its own `Object`:
- `Object.freeze`, `Object.seal` and `Object.preventExtensions` give their argument back (tslang's objects
  have no frozen state).
- `Object.keys`, `Object.values` and `Object.entries` of an object or a tuple are made of its fields at
  compile time, the keys `for...in` walks. The result is a fresh, mutable array.

Still open:
- `any`, `object` or record arguments ("its keys are not known").
- `Object.assign`, `Object.fromEntries`, `Object.defineProperty`, `Object.create`.
- `push` on the call result itself (`Object.keys(o).push(...)`: "Can't get reference of the array").

132 (a type parameter nothing gives a type: "not all types could be inferred", 31 tests). Fixed (fifth batch):
- An uninferred type parameter is its constraint, else `any`, as TypeScript falls back to the constraint and
  as tslang's erased instantiation does. A declared default was already used.
- 6 of those tests compile now; most of the rest stop at a later, unrelated error.
- Still open:
  - `tupleTypeInference2.ts` now reaches a crash that `g<void[], any>([[]] as [void[]])` already had: a union of
    tuples holding `void[]`.
  - Passing a generic class as a constructor type (`fruitFactory1(Banana)`) fails on the class reference.
  - A function given for `() => Generator<T>` (no `Generator` type, 125) crashed in the type printer while
    reporting the mismatch; this already happened with explicit type arguments. A type that could not be
    made now prints as `<unresolved>`.
  - `contextualGenericReturnToOptionalClassField.ts` crashes about every other run, with this batch's
    compiler and the one before it: a nondeterministic crash, not caused by this batch. Fixed (sixth batch,
    133).

133 (crashes, sixth batch). Fixed:
- `contextualGenericReturnToOptionalClassField.ts` crashed in about half its runs. The debug build asserts
  every time: casting a boxed object to a tuple read its storage as a tuple, and the storage of `object`
  itself (`lookup<T extends object = object>()`) is not one. Now it is the error "can't cast 'object' to
  '{label:string}': the object's fields are not known".
- `[...A, ...A]` of a tuple `A` crashed in getRestType, which took every rest element for an array. A
  spread tuple now has its elements in place. A rest element of any other type that is not an array is
  an error.
- `[[]] as [void[]]` crashed making an LLVM array of `void`. A byte stands in for an element that LLVM
  can't hold, wherever the lowering makes an array type.
- `[undefined, ''] as [never, string]` (`tupleTypeInference2.ts`) crashed lowering a value of type `never`
  (an LLVM `void`) into a tuple. Converting a value to `never` is now an error ("a value of type 'undefined'
  can't be converted to 'never'"). A `never` element in a tuple's LLVM struct takes a byte, and a
  `void`/`never` array element does too. A rest element of a type parameter (`[...TArgs, TLast]`) stays
  generic.
- A class field holding a class (`static self = X;`, `cls = B;`) holds the class's value, `typeof X`, as a
  `let` does (108). It failed with "failed to legalize operation 'ts.ClassRef'".
- Still open: a rest parameter typed as a tuple (`fn(...args: [...A, number])`) cannot be called with
  separate arguments ("'ts.CreateArray' op result #0 must be ..."), and a class given where a constructor
  type is expected (`reg.register(B)`).

Test: `00class_field_values_tuple_spread.ts` (all four models).

134 (a value of a generic function type: `declare const p: <T>(fn: () => T) => T;`, `let b: <T>(x: T) => T`,
`var b: { <T>(x: T): T }`; about 15 of the 55 "failed statement" tests, with nothing else said). Fixed (seventh
batch):
- A function type, a constructor type or a call signature with type parameters, written as the type of a
  value, holds one function. Its type parameters are erased, each to its constraint or else `any`, as an
  erased instantiation's are.
- Where an enclosing generic is still unresolved, the type stays as before.
- Most of the other "failed statement" tests now report the error that stopped them.
- Assigning an arrow to a call-signature object type (`let c: { (x: number): number } = (x) => x`) still
  fails, without generics too.

Also fixed: an interface extending `Tag<Self, Type>` where a generic function is also named `Tag` read a null
value's type (a crash; the name as a value is the function). Its base is now taken by the name as a type.
Still open: in that program the extending interface's own members are missing ("Interface member 'key'
can't be found"); with the function renamed, it works. Test: `00generic_function_type_values.ts`.

Two crashes that already happened before, which more tests reach through erasure, are fixed too:
- `merge<any, any>` of a conditional type (`keyof base & keyof props extends never ? ...`,
  `longObjectInstantiationChain3.ts`) read a non-parameter inferred type as a type parameter in the false
  branch. Now it is checked, as in the true branch.
- A declared variable of an intersection tslang can't resolve to one type (`typeof ErrImpl & (() => number)`,
  `aliasInstantiationExpressionGenericIntersectionNoCrash1.ts`) aborted the lowering. Now it is an error:
  "the intersection has no single type".

135 (crashes, eighth batch). Stacks of the 52 tsc-clean non-JSX crashes were taken with the debug build and
grouped. Fixed, 23 tests: each read a type or info that could not be made, now each is checked:
- a template literal type span (`getTemplateLiteralSpan`, 4);
- `Record<K, V>` of keys or values not made (`RecordType`, 2);
- `for...in` over a value with no type (`forInFieldKeys`, 2);
- an index signature of no function type;
- a constraint met by an interface not declared yet (`extendsType`, 3);
- `super` in a class that extends nothing ("'super' can only be used in a class that extends another", 3);
- a base class entry that is null (`getHasVirtualTable`);
- inference from an interface not declared yet (`tryInferInterface`, 2);
- narrowing to a type that could not be made (`isSafeTypeTheSameAndNoNeedToCast`, 3).

Still crashing (29):
- a builder inserting into a block that is gone (`globalVariableAccess`: `errorConstructorSubtypes.ts`,
  `discriminantOrderIndependence.ts`);
- recursion with no bound in the type printer and in type instantiation (`checkInfiniteExpansionTermination*.ts`,
  `recursiveTypes1.ts`, `stableTypeMapperOrdering.ts`, `varianceProbling*.ts`, `subclassThisTypeAssignable02.ts`);
- `parsingDeepParenthensizedExpression.ts` (the parser's recursion);
- the rest, each its own.

Test: `00satisfies_object_builtins.ts` (all four models).

136 (ninth batch; with #908 merged, 26 tsc-clean tests still crash, listed under 135). Fixed:
- A mapped type over `keyof T` of a T not known yet (`declare function h<T>(): { [K in keyof T]: T[K] }`,
  `combineReducers<S>(r: { [K in keyof S]: ... })`): K was not declared while its template was read ("can't
  find type by name: K", about 25 tests). K stands for `keyof T` there, and is the key when there is no `as`.
- A generic function given as an argument (`map(xs, identity)`, `apply(identity)`, `xs.filter(isDefined)`):
  the callee's parameter was rebuilt with its uninferred type parameters missing ("can't find type by name:
  U"), and the argument's own type parameters were read in the callee's scope. Now an uninferred one stays
  itself, and the argument's are inferred from the parameter's type, as a call infers them.
- `typeof this.x` in a method's signature left a class reference in the module's body ("failed to legalize
  operation 'ts.ClassRef'"); a qualified name is evaluated in the temporary scope other `typeof`s use.
- A method of a primitive stored in a variable (`let f = n.toFixed`, `let { toFixed } = 1`, a module `const`):
  in a function the call through it was dropped with no error (`print(f(1))` printed an empty line); at the
  module level it failed in the lowering. Now it is an error; a local `const` still names it.
- A namespace stored in a variable with no declared type (`var v = M`): "a namespace is not a value", where
  the lowering failed (TSLANG 107 stays: namespaces are not values).

Still open (counts of tsc-clean single-file tests): calling a value of type `any` (51), `'export ...' is not
supported` (49, by design since #908), overloads of a declared function (23), JSX tests that need react.d.ts
(31), an arrow or method with type parameters of its own in an object literal or a `var` with a declared
generic function type (`{ id<T>(v: T) {} }`, `var f: <T>(x: T) => T = <T>(x: T) => x`), a callback with fewer
parameters than the function type it is given to (`xs.filter(isDefined)`), inference through reverse mapped
and recursive conditional types, `await` in `try`/`finally` of an async function ("'ts.Try' marked illegal"),
a class passed as a constructor type (`f(Banana)` for `new (...args) => T`).

Tests: `00generic_function_arguments.ts` (all four models); `diagnostic-location/namespace_as_value.ts`.

137 (fixed in #884, except the open items at the end; found 2026-11-17 on main 363b13ce; TypeScript's compiler test
`tests/cases/compiler/abstractClassUnionInstantiation.ts`): `tslang abstractClassUnionInstantiation.ts`
failed with "Can't resolve property 'map' of type [ConcreteA, AbstractA, AbstractB]". The file:

```typescript
class ConcreteA {}
class ConcreteB {}
abstract class AbstractA { a: string; }
abstract class AbstractB { b: string; }

type Abstracts = typeof AbstractA | typeof AbstractB;
type Concretes = typeof ConcreteA | typeof ConcreteB;
type ConcretesOrAbstracts = Concretes | Abstracts;

declare const cls1: ConcretesOrAbstracts;
declare const cls2: Abstracts;
declare const cls3: Concretes;

new cls1(); // should error
new cls2(); // should error
new cls3(); // should work

[ConcreteA, AbstractA, AbstractB].map(cls => new cls()); // should error
[AbstractA, AbstractB, ConcreteA].map(cls => new cls()); // should error
[ConcreteA, ConcreteB].map(cls => new cls()); // should work
[AbstractA, AbstractB].map(cls => new cls()); // should error
```

Five gaps behind it, each fixed:

- A tuple had no Array members. An array literal of mixed element types (`[1, "a"]`, `[A, B]`) is a tuple,
  so `.map` did not resolve. TypeScript types it `(number | string)[]`. A tuple of unnamed elements now has
  Array's members that read it (map, filter, forEach, every, some, reduce, find, join, slice, at, ...), from
  an array of its elements of the union of their types; `t.length` is its size. Mutators (`t.reverse()`,
  `t.push(x)`) stay refused: they would change that copy, not the tuple. Under `-mm=own`, a tuple read
  from a local whose elements own memory is an error (each element would get a second owner).
- `new c()` with `c: typeof A | typeof B` built a `ts.New` of the union type: "'ts.New' op result #0 must
  be ..., but got '!ts.union<!ts.class_value<...>, ...>'", then "module verification error". It now makes
  the class the value holds: `.new` and the constructor come from the value. The arguments are generated
  once, so every member's constructor must take the same ones. A union holding an abstract class is
  TypeScript's error: "Cannot create an instance of an abstract class through a value of '...'". A class
  named as an array element (`[A, B]`) or a ternary branch (`c ? A : B`) was a class reference, not its
  value: given to `typeof A | typeof B` it was "'ts.Cast' op type ... can't be stored in ...", and even
  `c ? A : A` given to `typeof A` was refused. Both now give the class's value.
- A method called on a union of classes called the first member's: `o.name()` with `o: A | B` holding a B
  returned A's result. A union's tag cannot tell classes apart: `IsTypeTagOf` compares a layout id, and
  every class (and every class value) has the same one, a pointer. A union of classes laid out alike has
  no tag at all. A call of a method every member class has, with the same parameters, now evaluates the
  object and the arguments once, asks the object its class as `instanceof` does (a subclass before its
  bases) and calls that class's method.
- `-mm=own`: `(c: typeof A): A | B => new c()` was "a value takes a second reference" (also into `A | null`,
  a union local, an `(A | B)[]`). MLIRGen marks the `.new` call through a class value as giving a fresh
  instance (`__own_fresh_result`), but that call is indirect, and LowerToAffineLoops turned it into a
  `CallInternal` before the ownership inference read it, keeping only the ABI attributes. Without the mark
  the result had no owner, and its cast to the union could not be a move. The mark now goes onto the
  replacement call.
- `-mm=own`: a tagged union was taken to own memory whatever its members. One whose members all hold no
  memory (numbers, booleans, null, class values: `number | boolean`, `typeof A | typeof B`) owns none now,
  so `map` over an `(typeof A | typeof B)[]` or a `(number | boolean)[]` lends its elements instead of
  failing in the default library ("a value that borrows 'this[]' is used here after it may be released").

The file now fails only where TypeScript reports errors, lines 15, 16, 19, 20 and 22, with the abstract
class error. Lines 17 and 21 compile and run once `cls3` has a value. As written it is declared only, so
`tslang file.ts` runs `new cls3()` through an empty class value and crashes; that is the test's
`declare const`, not a compiler fault. Fixed in #884 and #885. Regression tests
`tslang/test/tester/tests/00tuple_array_methods.ts`, `00union_method_dispatch.ts` (every model),
`class_value_union_abstract_error.ts`, `own_tuple_as_array_error.ts` and
`own/own_new_through_value_union.ts`.

Still open, found along the way (each on main before #884):
- Fixed after: `for (const x of t)` over a tuple was "Element access with a non-constant index is not
  supported on this type" (before #884, "Can't resolve property 'length'"). `t[i]` of a tuple of unnamed
  elements at a run-time index now reads each element where it is, as the union of their types, chosen
  by the index (an index past the end reads the last element), and for...of reads the tuple so. Under
  `-mm=own` the elements are borrowed, not copied. Still refused under own: for...of straight over a
  literal of objects it makes (`for (const o of [new A(), new B()])`, "an object literal that holds a
  value it owns is moved only into a local ..."); kept in a `const` first it works. Fixed in #887.
  Regression test `tslang/test/tester/tests/00tuple_for_of.ts` (every model).
- Fixed after: `const c: typeof A | typeof B | null = B; new c(1)` gave a `typeof A` ("'n' through a value
  of 'typeof A': only a static member can be reached through a class value"). Four gaps behind it:
  `new` of a union of class values with `| null` built a `ts.New` of the union type, as #884 fixed
  without the null (now: the class it holds, made after the arguments, and "a class value is null or
  undefined: there is no class to make" thrown when it holds none); `typeof A | null` was stored as
  `typeof A`, its null dropped as for a type that holds null in its own pointer, so assigning null was an
  invalid cast (a class value is four pointers: it keeps null as a member now, as a number does); a class
  value as a condition (`if (k)`, `!k`) was an invalid cast, and of a union of them "Cast ... to boolean
  is not supported" (now true, or not null); and a method of `A | B | null` called A's for any object
  (the dispatch of #884 now passes over null and undefined). Fixed in #889. Regression test
  `tslang/test/tester/tests/00class_value_union_null.ts` (every model).
- Fixed after: `-mm=own --no-default-lib`: `map` over a union array holding strings
  (`const a: (number | string)[] = [1, "a"]; for (const k of a.map(...))`) was "a value that borrows an
  element cannot be stored, returned or captured" (`filter` too, and with the default library "'filter'
  would give an element of type 'number | string' a second owner"). The built-in generator keeps each
  element in its state, which for a `string[]` takes a copy of the string (spec 50); a tagged union was
  not copyable. A union of strings and data nothing owns (numbers, booleans, null, undefined) is now:
  `ts.StringCopy` copies its payload when its tag says string, and `@sharesElements` passes it as it
  passes a string. Fixed in #890. Regression test `tslang/test/tester/tests/00string_union_elements.ts`
  (every model).

138 (tenth batch of the compiler-tests sweep, 2026-11-20; internal errors that reached the user). Fixed:
- `new` on a value that is no class - an `any` (`var A; A = class {}; new A()`, `new (<any>a).b`), a namespace
  (`new Promise(...)` where Promise is only a namespace) or `undefined` - built a `ts.New` of that type, which
  failed verification ("'ts.New' op result #0 must be ..."). Now an error at the `new`: "'new' on a value of
  type 'any' is not supported: its class is known at run time only; give it a class type ('typeof C')".
- A tagged template whose tag is a generic function (`id\`hello\`` with `function id<T>(x: T)`) called the
  generic itself ("'ts.Call' op 'id' does not reference a valid function"). It is now the call
  `id(["hello"], ...values)`, which infers the type arguments. The tag is evaluated before the values, as in
  JavaScript.
- An arrow with untyped parameters that nothing gives types to, as an array element (`[x => x]`) or given as
  `any` (`g(r => r)` with `g(a: any)`), stayed generic and the address of a function never made was stored
  ("'llvm.mlir.addressof' op must reference a global"). It is instantiated with `any`, as a field's or a
  `let`'s is (TSLANG 106); boxed in an `any` it gets its call adapter, so `a(5)` through the `any` works.
- A namespace in a field of an object literal (`{ m: M }`): "a namespace is not a value", where the lowering
  failed ("failed to legalize operation 'ts.NamespaceRef'").
- A class given to a constructor type (`register(B)` for `register(c: new (...args: any[]) => A)`): "a value of
  the constructor type '...' is not supported: use the class's type, 'typeof C'", where a bare class
  reference failed in the lowering ("failed to legalize operation 'ts.ClassRef'").

- "not a function to call" (the error now names the type): `b && f || g` was `boolean | F` - `&&` of a boolean
  is `false | F` now, as in TypeScript, and `||` drops the `false` its left side may hold, so it is `F | G`;
  `c ? f : g` and `o.f ?? g` of two different function types made a union that can't be called - they are the
  one taking the most parameters, when the other's are the first of them and the results match, the other
  cast to it (a named function or an arrow given fewer parameters goes through its thunk).
- A bound method - a class's, or an interface's (`val.something`) - given as a function type with more
  parameters after its own (`o.something ?? val.something`, `apply(c.something)` for `(a, b) => void`):
  refused ("it takes 1 argument(s), 2 expected", the `this` counted). It goes into a box with an adapter bound
  to it that calls the method with the first arguments (boundMethodWiderTypeAdapter, which did this only for
  extra parameters first). Two bugs on the way: given as a bound method type, the adapter kept that type's
  `this` as a parameter after the box (gc ran by luck, rc read a missing argument); and made in a branch it was
  released at the branch's end under rc (OwnedReturnConsumptionPass did not follow the cast to the method
  type), so the call read freed memory. Under -mm=own a method given as a function value stays refused; the
  message now counts the parameters a caller gives and says so.
- An object literal whose methods use variables of the enclosing function, given for a type literal with
  methods (`run({ something() { n = 5; } })` for `run(o: { something(): void })`), crashed at run time in
  every memory model (main too). Such a type is a tuple, copied field by field; the literal's methods reach
  their variables through a hidden field of the literal (`.captured`, read off `this`), which the copy does not
  have, and were called with the copy as `this`. Now a compile error that says to declare the type as an
  interface, which takes the object itself (and works). A function-typed field (`something?: () => void`) and
  a literal whose methods use no outer variables were not affected.
- A declared union of function types (`type H = (() => void) | (() => number)`, `x()`): "not a function to call",
  and no call could tell its members apart (each is tagged "function"). It is now one type
  (getMergedFunctionType): the widest parameters, when each member's are the first of them, returning `void`
  when one member does (with the same results, the widest member). `c ? f : g` / `a ?? g` merge the same way.
  For that, a function returning a value is given as one returning `void`, as TypeScript allows
  (`let f: () => void = n`, refused before: "one returns a value and the other does not"): a named function or a
  closure through its thunk, a bound method through its boxed adapter, each dropping (releasing) the result.
  Under -mm=own a method given so stays refused, and the message says so. Churn of 3M dropped strings: no leak
  (rc 5.4 MB peak, own 5.4 MB). A declared union whose widest member's extra parameters are optional
  (`((v) => void) | ((v, i?) => void)`) stays a union: it is given as its narrower member's type, which the
  merged type (taking `i`) could not be.
- Not yet: a function union with `| null`/`| undefined`, or one whose members return different non-void values
  (`(() => string) | (() => number)`), calling a declared union whose extra parameters are optional, call signatures through conditional types, generic arrows
  returning generic arrows.

Still open: values of constructor types (`new (...) => T` as a parameter type; `typeof C` works); a class
declared in a nested block with the name of an outer one ("redefinition of symbol named 'C..instanceOf'":
block-scoped declarations share the function's namespace); `x.constructor === Symbol` (the built-in `Symbol`
as a value, "'llvm.mlir.addressof' op must reference a global").

Tests: `00untyped_arrows_and_tags.ts` (compile and JIT; not -mm=own, where calling `any` is refused);
`00call_logical_of_functions.ts` (compile, JIT, JIT -mm=own); `00bound_method_fewer_parameters.ts` (compile,
JIT, JIT -mm=rc); `00function_union_call.ts` (compile, JIT, JIT -mm=rc, JIT -mm=own);
`diagnostic-location/new_on_any.ts`, `namespace_in_object_field.ts`, `constructor_type_value.ts`,
`captured_literal_method_copy.ts`.

139 (the open items of 138, 2026-11-20). Fixed:
- A function value read at run time - a parameter, a variable, a bound method - given as a function type it does
  not match exactly: only a named function or an arrow had a thunk (fewerParametersThunk), so `let u: H = cb`
  with `cb: () => number` and `H = (() => void) | (() => number)` became an error with 138's merged unions (before
  138 it was stored in the union, which could not be called). Such a value now goes into a box with an adapter bound
  to it (functionValueAdapter), as 138's bound method adapter did. Both adapters follow one rule
  (adaptableSignature): fewer parameters ignore the rest of the arguments; more are given `undefined` when each
  extra one is optional (`(v, i?) => void` given as `(v) => void`); a result is dropped where the type returns
  `void`, and widened where it returns a union holding it (`() => string` given as `() => string | number`, which
  was "return type 'string' is not 'number | string'"). The parameters both have must be the same. A thunk's name
  is now its type's hash: two targets with one parameter count but different results shared the first one made.
  Under rc the widened result is retained for the return and the call's own reference given back, as in
  `return s()` of a `string | number` function; taken over as is, the thunk's return was not one
  OwnedReturnConsumptionPass reads as owned, and one such function left every call through a function value in
  the module unreleased (2M calls: 45 MB, now 5.5 MB). Under -mm=own a value read at run time is not given so
  yet (a box would hold a second reference to it); the error says so.
- Declared unions of function types merge further: members returning different values are one type returning
  their union (`(() => string) | (() => number)` is `() => string | number`), `| null` / `| undefined` stay
  beside the merged functions (`if (f) f()`, `g?.()`), and a union whose widest member's extra parameters are
  optional merges too (138 kept it a union: it is given as the narrower member's type through the adapter).
- A class declared in a nested block of a function is that block's, as a `let` is: the block gets a namespace of
  its own inside the function's. It was the function's: after `{ class C {} }` a `new C()` made the block's (the
  module's `C` hidden), and two blocks declaring one name were "redefinition of symbol named 'C..instanceOf'".
- `new (...) => A` with `A` a class is the class value `typeof A`: `make(B)` for `make(c: new () => A)` and
  `register(c: new (...args: any[]) => A)` take the class (or a subclass), read at run time, and `new c()` makes
  one of it. Its constructor is called as `A`'s is, as through `typeof A`: a subclass whose constructor takes
  other parameters is called with `A`'s (wasm traps on the signature). A constructor type of an interface stays
  one and is still refused at the value given to it.
  In type checks a class value is matched as its class: a generic constraint (`T extends Constructor<Point>`
  given `Point`) and a conditional's `infer` (`T extends Constructor<V, infer P> ? P : never`) work as before.
- A key of an `any` read as a function of the same name found in a namespace: `foo.get = class {}` with
  `foo: any` took the default library's new generic `Reflect.get<T, K>(target: T, ...)`, whose `T` takes
  anything, as an extension function of `foo` ("can't assign to a value of type '(p0: T, p1: K) => any'"). A
  key of an `any` stays a run-time read unless the function takes an `any`.
- The built-in `Symbol` read as a value (`x.constructor === Symbol`): an error at its place, "the built-in 'Symbol'
  is not a value", instead of "'llvm.mlir.addressof' op must reference a global" with no location.

Still open: a function value read at run time given as another function type under -mm=own; constructor types
of interfaces; `Symbol` as a value; call signatures through conditional types; generic arrows returning generic
arrows. Found along the way (not fixed): under rc with the installed default library a call through a function
value does not release its result when a library function returns heap memory unclassified (stale library in
the build environment; not seen with `--no-default-lib`).

Tests: `00function_union_call.ts` (+ `| null`, different results, optional extras; compile, JIT, rc, own);
`00function_value_adapter.ts` (compile, JIT, JIT -mm=rc); `00block_scoped_class.ts`,
`00constructor_type_value.ts` (compile, JIT, rc, own); `diagnostic-location/builtin_symbol_value.ts`,
`own_function_value_adapter.ts`; `constructor_type_value.ts` now an interface's constructor type.

140 (twelfth batch of the compiler-tests sweep, 2026-11-20; picked from the failing tests tsc compiles cleanly). Fixed:
- A class declared in a block at the module's or a namespace's top level (`{ class C {} }` with a module `C`) is the
  block's, as 139 made one in a function's block: "redefinition of symbol named 'C..instanceOf'".
- `interface X extends Object` / `extends Function`: the built-in types, which the program does not declare, give
  the interface nothing and are left out ("'Object' is a type here, not a value").
- `export default Foo` of an interface or a type alias of the file exports nothing at run time: it was a constant
  given the type ("failed to legalize operation 'ts.InterfaceRef'").
- An `await` in a `finally` block: an error at the `await` ("failed to legalize operation 'ts.Try' that was
  explicitly marked illegal"). One in the `try` or a `catch` works.
- A type parameter constrained by `keyof` keeps the literal it is given (`P extends keyof T` given `"minimizer"`), as
  in TypeScript: widened to `string`, `T[P]` was "indexing a type by 'string' is not supported". A key whose type is
  a string literal (`obj[key]`, `key: K`) reads the field it names ("indexer is not declared").
- A crash from 138 (#958): a function type with a type that failed to resolve (the `T[P]` above, after its error)
  matched against an adapter read the null type (`contextualTypeCaching.ts`).

Not yet: generic methods and arrows in an object literal (`{ id<T>(v: T) {} }`, "can't find type by name: T"); a
non-exported class in each of two declarations of one namespace; an `await` in a `finally`; a callable interface
(`interface Cb extends Function`) given a function; overloads in a `declare class` that is used ("redefinition of
symbol named 'C.constructor'"; refusing them as a declared function's overloads are refused also refused a `@cpp`
class's C++ overloads, and declared classes nothing uses, which compile).

Tests: `00block_scoped_class.ts` (+ module-level blocks), `00interface_extends_builtin.ts`,
`00keyof_literal_key.ts` (compile, JIT, rc, own); diagnostic-location `await_in_finally.ts`.

141 (thirteenth batch of the compiler-tests sweep, 2026-11-20). Fixed:
- A branded primitive (`string & { __brand: "UserId" }`, `number & { readonly _unit: "m" }`, `{ _foo: number } &
  number`): a primitive and object types that only tag it. Its values are the primitive's, as in TypeScript; it was
  `never` ("'Never' can't be used as parameter type", "a value of type 'string' can't be converted to 'never'").
- A set accessor whose parameter type is not the get accessor's (divergent accessor types, `get foo(): number` with
  `set foo(x: string | number)`): an assignment is checked against the setter's type, as it was cast to it already
  ("type '"32"' is not assignable to type 'number'").
- A parameter's destructuring pattern that nothing gives types to (`function f({ a, b })`, `function g([x, y])`,
  `({ a }) => a` with no type to take it from) has `any` elements, as TypeScript's implicit any: "can't resolve
  type for binding pattern 'a', provide default initializer". A skipped element (`[, z]`) holds its place. The
  array pattern's tuple has unnamed elements, as `[any, any]`'s: with generated names an array literal given to it
  leaked under -mm=own (1M calls: 77 MB, now 5.7).
- A branch of `c ? f : g` that is a function of untyped parameters (`var f = c ? function (n) {...} : ...`), with
  nothing receiving it to type them, is instantiated with `any` as a `let` given one is - in the branch that runs, and
  for the type of one a constant condition skips: "variable 'f' is referencing generic type".
- 140's `export default` scan read a declaration's name through another node type: a crash under gcc (every
  Linux CI leg), not under MSVC.

Not yet: a name written with two leading underscores (`__property2`) is kept escaped with a third, as the parser
escapes it, so a string key of it ("__property2", `keyof`) does not match. Unescaping it where names are read
broke the compiler's own escaped names; every place a member name becomes a field or a key would have to.

Tests: `00branded_primitives.ts`, `00divergent_accessors.ts`, `00untyped_binding_params.ts` (compile, JIT, rc, own);
`00conditional_untyped_functions.ts` (compile, JIT, rc).

142 (fourteenth batch of the compiler-tests sweep, 2026-11-21). Fixed:
- `Array.isArray(x)`: "Class member 'isArray' can't be found" (the default library's Array<T> has no such
  static). It is `typeof x === "array"`, answered from x's type at compile time or by a union's or an `any`'s tag,
  and it narrows x in both branches and after an `if` that returns, as TypeScript's type guard. (The narrowing of a
  condition's `false` was read from the expression as written, not the one tested: it now reads the same one.)

- A member of another declaration of an enum named in a later one's initializer (`enum E { a } enum E { c = a }`):
  "can't resolve name: a". The earlier declarations' members are names there, as the same declaration's are.

Still open, looked at: a module-level arrow calling a `const` arrow declared after it, with top-level code in
the module (the code runs in a function of its own, where a later `const` is no name yet - as in any function:
`const a = () => b(); const b = ...` in a function is "can't resolve name: b"); an `async function*` given a
type naming `AsyncGenerator` (not in the default library) crashes after the error, on Windows only.

- A module-level variable given a value as `any` or `unknown`, or one read from it (`let r: any = 5;`, `const a = r
  as number`): LLVM refused the global ("ops with side effects not allowed in global initializers") - boxing
  allocates, and a read from it checks the type, and both were left in the global's constant initializer. It is set
  by a global constructor, as an array or a string concatenation is.
- A function declared in a block of the module's top-level code (`{ let x = 1; function foo() { x } }`,
  `for (let x; ;) { function foo() { x } }`), and a method of an object literal there: "'ts.Load' op using value
  defined outside the region", and a call of it crashed. The pass over the module's declarations met it first,
  where the block's locals are not known, and registered it with no captures; it is discovered where it is
  generated, as an arrow there is (105).
- A parameter's default value that reads what the closure captures (`(b = super.m()) => b`, `(b = this.m()) =>
  b`, `(b = y) => b`): read across the closure's region, as above. A closure's captures are named before its
  parameters now (an object literal's method reaches them through its `this`, which is a parameter, and keeps
  the old order).
- A parameter whose type holds `undefined` and has a default (`foo: number | string | undefined = 2`): "'ts.Value'
  op failed to verify that type of 'value' matches element type of 'optional'". The argument is optional of the
  type without `undefined`, and is stored as the parameter's own type.

Still open, looked at: under -mm=rc a module-level `let u: unknown = 7` is read as freed memory ("Can't cast
from any type"): `unknown` is boxed as `any` is, but the box is given back at the end of the statement and an
`unknown` variable owns nothing. Counting `unknown` as an owner, as `any` is, broke a value that is no box - a
construct signature with no return type (`new(x: number)`) returns `unknown`, and the class object comes through
it as it is (rc corpus newWithSpread). `x == "a"` with x of `number | string | undefined` (an optional of a union) crashes the
compiler in lowering - also before this batch; a type literal's divergent accessors (`type T = { get foo(): number;
set foo(v: number | string) }`) still take the getter's type for an assignment (an interface's take the setter's,
141).

Tests: `00array_is_array.ts` (compile, JIT, rc, own), `00merged_enum_members.ts` (compile, JIT),
`00module_block_functions.ts`, `00any_global_initializer.ts` (compile, JIT, rc, own),
`00default_param_captures.ts` (compile, JIT, rc; -mm=own refuses an escaping closure over a caller's value, as before).

143 (fifteenth batch of the compiler-tests sweep, 2026-11-21). Fixed:
- `x == "a"` with x of `number | string | undefined` (an optional of a union with a tag) crashed the compiler in
  lowering (142's open item). It is compared as a union is with a value it has members that can't be compared
  with: by a helper that asks the member it holds, and an empty one is never equal.
- A union of function types whose member has a parameter or a result type not resolved yet (the React
  declarations of the JSX tests) crashed the compiler where 139's union merges function members.
- A destructuring `var` of the module reading a global declared before it (`var r = {...}; var { a } = r;`): its
  constructor ran before the global's own initializer and read nothing - a crash at run time. Such a constructor now
  runs after the globals before it, in the module's order.
- A destructuring `var` of the module all of whose names were declared before (`var { a } = r; var { a } = s;`):
  "redefinition of symbol named 'a'". It is an assignment to them, as `var v = 1; var v = 2;` is.
- An assignment pattern's defaults: `({ a: x = 1 } = o)` failed ("only a variable, a property or an element can be
  assigned to"), and `[x = 1] = arr` assigned the default before the element; a shorthand's default (`{ a = 1 }`)
  was ignored. The default is taken when the field is empty, or for an element past the end of the array (where a
  read of it was whatever follows the array's data); a hole (`[a, , b]`) takes nothing; a target that is a pattern
  itself (`[[a, b] = [1, 2]] = [[2, 3]]`, `{ q: [a, b] } = o`) is destructured from its element - it was evaluated
  as an expression, which gave `1 2` for that one, or "only a variable ... can be assigned to".

Tests: `00optional_union_equality.ts` (compile, JIT, rc), `00destructuring_redeclare_defaults.ts` (compile, JIT, rc,
own).

144 (sixteenth batch of the compiler-tests sweep, 2026-11-21). Statements that failed with nothing said ("failed
statement" and no reason), and `this` in static arrows. Fixed:
- `return g()` of a call that returns nothing (`function h() { return g(); }`, `return super.foo()`, in an arrow):
  the missing value failed the function's discovery with no message. It is the call and a `return;`.
- An element past the end of a tuple a literal is taken as (`var a = [1, "x"]; var [x, y, z] = a;`) is undefined.
- `this` in an arrow made in a static member (`static foo = () => this.n`, an arrow in a static method): "can't
  resolve name: this". An arrow captures `this`, and a static member has none to capture; its `this` is the class,
  as the code around it has it (GenContext::staticThis).
- A `let`/`const` of a function's body read by a function made before it (`const f = () => n; const n = 1;`,
  `const a = () => b(); const b = () => 10;`): "can't resolve name: n". It is one of the function's variables,
  declared before the body as a `var` is (declareFunctionVars), and its declaration gives it its value. Not yet: such
  a later function that itself reads a variable of the body declared with a value (its type is known only where it is
  reached); under -mm=own a captured variable holding a closure is refused, as before.
- Clear errors where there were none: an object given as an interface with an index signature (`[s: string]: T`
  - not supported yet: a Map, or an interface of named members); an object without an optional method of an
  interface (`b?(): number` - an absent one has no absent mark in its slot yet, so a test of it would read as present;
  an optional property of a function type, `b?: () => number`, works); a generic method in a type literal (an
  interface can have one).

Still open, looked at: `this` in an arrow in a default parameter value of an instance method or a constructor
(`bar(f = () => this)`) - the default is evaluated for the signature before `this` exists; `static y = this` fails
in lowering (ClassRef); `(Foo as any) = null`.

Tests: `00void_call_return.ts`, `00static_this_arrows.ts` (compile, JIT, rc, own), `00forward_closure_refs.ts`
(compile, JIT, rc); diagnostic-location
`index_signature_object.ts`, `absent_optional_method.ts`, `generic_method_type_literal.ts`.

145 (seventeenth batch of the compiler-tests sweep, 2026-11-21). Fixed:
- `static y = this`: the class's value, as `static self = P` is - `this` in a static field is the class (an instance
  field's `this` is the instance, which 133 excluded for both). It failed in lowering (`ts.ClassRef`).
- `this` in a parameter's default value of a method or a constructor, in an arrow too (`bar(f = () => this)`,
  `constructor(z = 1, zzz = (p = this) => this)`): "can't resolve name: this". The method's own `this` is named while
  the defaults are evaluated for their types, as an explicit `this:` was; an arrow met there is discovered again where
  it is generated (GenContext::evaluatingParameterDefault - applied to every evaluated expression, an instance field's
  `() => () => this` crashed the compiler). A parameter typed by a function given as its default is a hybrid
  function: a closure held as a plain one lost its captures ("invalid cast" in lowering).
- A class merged with a namespace of its name (`class C {} namespace C { export var x }`): `new C()` was "'C' is a
  namespace, not a class". The name is the class; a member it hasn't got is the namespace's (`C.x`), and in a type the
  name's qualified part is the namespace (`C.Shape`, also for a class expression named as a namespace is).

Still open, looked at: an arrow nested in an arrow given as a default (`f = () => () => this`) - its captures are
those of the evaluation; a namespace function named as an instance method of the class it merges with (`C.foo`:
"redefinition of symbol").

Tests: `00this_in_defaults.ts` (compile, JIT, rc), `00class_namespace_merge.ts` (compile, JIT, rc, own).

146 (eighteenth batch of the compiler-tests sweep, 2026-11-21). Fixed:
- An implementation with fewer parameters than an overload signature of it (`f(a): T; f(a, b): T; f(a) {...}`,
  methods and constructors too), which TypeScript allows - the further arguments are ignored: a call through the
  wider overload was "Expected 1 arguments, but got 2". The implementation is given the overload's further parameters,
  optional and of type `any` (widenOverloadImplementations, run over each file's statements and class members - a walk
  of every node overflowed the stack on a deep expression, binderBinaryExpressionStress).
- A generic function held by an object literal's field (`{ test: <T>(value: T): T => value }`, `{ m<T>(x: T) {} }`):
  "can't find type by name: T". The field holds one function, its type parameters `any`, as its untyped parameters
  are (106).

Tests: `00overload_more_params.ts`, `00object_generic_members.ts` (compile, JIT, rc, own).

147 (nineteenth batch of the compiler-tests sweep, 2026-11-21). Fixed:
- `new SS()` of a generic class with no constructor and no base class, given no arguments: "Generic class 'SS' is
  missing type arguments". A type parameter with no default is `any` (TypeScript's `unknown`), as an untyped parameter
  nothing gives a type is (106). Narrow on purpose: applied to every `new` without arguments, the default library's
  classes were instantiated anew (the typed arrays failed), and a generic class's own `new this()` (its `..new`) made
  a `Box<any>` whose declaration for a shared library named `!ts.any`. A class whose constructor or base class could
  give the type arguments, and `new this()`, are as before.

Still open, looked at: a computed property name that is not a constant in an object literal (`{ [m]: i }`) needs
keys known at run time; an identifier starting with an extended Unicode escape (`const \u{61} = 1`) fails in the
parser ("Hexadecimal digit expected"), though the escape inside one and in a string scans - some re-scan of the token (fixed by 162);
the same non-exported class in two declarations of one namespace needs each declaration's own scope.

Tests: `00generic_new_without_args.ts` (compile, JIT, rc, own).

148 (twentieth batch of the compiler-tests sweep, 2026-11-21). Fixed:
- `typeof this.foo` as a field's type (`bar: typeof this.foo`): "Class member 'foo' can't be found" - while the class
  is first declared its fields are not members yet. The member's type is read from the class's declaration (its
  annotation, or its initializer's type); the declaration is now known in that first pass too (processingDeclaration).
- `C[0]` of a class member named `0` (`static 0 = 1`, `1 = "one"`): the member, as `C["0"]` is - read as an
  indexer's key it was "can't resolve property/field/base '.index'".

Still open, looked at: an object literal's accessor reading the `const` it initializes (`const a = { get self() {
return a; } }`) - its type is circular (TypeScript's `any`); a getter returning itself (`get p() { return this.p; }`).

Tests: `00typeof_this_field.ts`, `00numeric_member_keys.ts` (compile, JIT, rc, own).

149 (twenty-first batch of the compiler-tests sweep, 2026-11-21). Fixed:
- `this: void` (`function f(this: void, x: number)`), which says a function uses no `this`: no parameter at all -
  "'Void' can't be used as parameter type".
- `globalThis.x` where no `globalThis` is declared: the global `x`, named from the module's root, as a value and in a
  type query (`typeof globalThis.isNaN`) - "can't resolve name: globalThis". (`globalThis.Symbol` stays an error: the
  built-in `Symbol` is not a value, 140.)

Looked at and kept: a `never` parameter type tslang computed (an intersection it can't express) stays an error
(126).

Tests: `00this_void_param.ts`, `00global_this_members.ts` (compile, JIT, rc, own).

150 (twenty-second batch of the compiler-tests sweep, 2026-11-21). Fixed:
- A method implementing an interface's numeric method without a return type of its own (`clean(input: number) {
  return 123; }` for `clean(input: T): T` of Field<number>) returns the interface's type, as an override returns its
  base method's: the integer literal's s32 was "method signature not matching ... => s32".
- A class method an interface calls otherwise than it is declared - with a result the interface's `void` drops
  (`foo(): void` of `foo() { return "x"; }`), or without optional parameters the method has further (`method()` of
  `method(a = 4)`): "method signature not matching". The interface's slot holds an adapter calling
  `this.<name>(p0..pn)` as the class declares it (generateSynthMethodToCallField), made in each module that lays out
  the vtable. Only for a result the interface drops or one both have: a method returning nothing for a type
  predicate's result crashed the compiler (typePredicateInherit), and a result of another type is not cast.

Tests: `00interface_numeric_return.ts`, `00interface_method_adapters.ts` (compile, JIT, rc, own).

151 (twenty-third batch of the compiler-tests sweep, 2026-11-21). Fixed:
- A generic construct or call signature in a type literal (`declare var X: { new <T>(): number }`) failed as "failed
  statement" with nothing said: a clear error, as a generic method there is (144) - an interface can have one.

What remains of the tsc-clean failures is mostly the default library's (`Object.create`/`assign`/`fromEntries`,
`Record` and `Function` as values, `Iterable`/`Generator`/`AsyncGenerator` types, `Promise.resolve()`/`reject()` and
`"s".slice()` with no arguments, `toLocaleString(locales)`) and design questions (keys known only at run time in an
object literal, properties added to a function, a `never` tslang computes, declared-function overloads).

Tests: diagnostic-location `generic_construct_signature_type_literal.ts`.

152 (keys known only at run time, 2026-11-21; USER DECISIONS: a string index signature is a dictionary, an absent key
reads `T | undefined`, under -mm=own the dictionary owns its values). Fixed:
- `{ [key: string]: T }`, a string index signature and nothing else, is the default library's `ObjectDictionary<T>`
  (`@dictionary`, TypeScriptCompilerDefaultLib #114): entries by key known at run time, in the order they were added.
  Before, it was a record of `get`/`set` functions nothing could fill: `const d: { [k: string]: number } = { a: 1 }`
  was "field "get" can't be found in tuple '{a:s32}'".
- `d[k]`, `d["name"]` and `d.name` of a dictionary, outside its own class, are its entries - never its members, so
  `d.get` and `d.size` are entries too - read as `T | undefined` (undefined for a key it does not hold) and written as
  T. `k in d`, `for...in`, `Object.keys(d)` and `delete d[k]` / `delete d.name` go by its entries.
- An object literal with a key known only at run time (`{ [m]: i }` with `m: string`) makes a dictionary of its values'
  type, and so does one given where a dictionary is wanted (`const d: { [k: string]: T } = { a: 1 }`, an argument, a
  return). A spread into one copies a record's fields, or a dictionary's entries (the library's
  `objectDictionaryAssign`). Before: "a computed property name must be a constant (a literal, or a `const` holding
  one), or a well-known symbol". Whether a key is a constant is decided as before: its value's constant, or its
  literal type's (an imported `const`, `"a" + "b"` folded).
- A dictionary type given from elsewhere (a `return`'s receiver, a result type found in an earlier pass) is
  instantiated again where the literal is made: it may have been instantiated only in an evaluation, and then its
  members were generated nowhere ("'ObjectDictionary<si32>.constructor' does not reference a valid function").
- -mm=own: a parameter moved on paths that exclude each other - `if (f) a.push(v); else a[0] = v;`, a move then
  `return` and another move after it, a `T | undefined` narrowed once and moved on two paths - was "'v' is used here
  after its value was moved" (only one move of a parameter was allowed). Each move empties the parameter, so the
  release at the exit, kept for a path that moves nothing, frees nothing on the others; one the function keeps on
  every path is reached at each exit only through one of the moves (movesOnExclusivePaths). The dictionary's `set`
  needs it. Leak probes, flat at 33 MB: 20M calls moving a parameter on exclusive paths (own), 3M writes,
  overwrites and deletes of a dictionary of objects (rc and own).

- Found on the way: a module importing another and using a Map (`import "./a"; new Map<string, boolean>()`)
  failed with "'ts.Func' op symbol declaration cannot have public visibility": the keys adapter of the RangeError
  the library's Map throws was made while the imported file was read as declarations, a declaration, then made
  public. It is made in full there now, as a specialization is (makeAnyKeysAdapter).

- Later: an interface of a string index signature and nothing else (`interface Dictionary<T> { [x: string]: T }`;
  nothing extended, no decorator) is the same dictionary - `type Dictionary<T> = { [x: string]: T }`
  (isStringIndexSignatureOnly). It was a record of `get`/`set` functions nothing could fill. A class may still
  `implements` it (nothing to implement); without the default library's dictionary it stays the interface it
  was. One extending it and adding nothing (`interface More extends Counts {}`) is that
  dictionary too (`type More = Counts`); one adding members of its own stays an interface.
- Later: a generic class, interface or type alias instantiated inside a generic function with a type parameter of
  the same name as one of its own (`Box<S>` in `f<T, S>`, Box's parameter named T) took the function's: the
  instance's context starts from the function's type parameters and aliases, a type parameter as an argument did
  not replace the T there, and an alias is looked up first - `Box<S>` was made as `Box<T>`, so
  `f(Box<number>, Box<string>)` inferred T = number | string, S = any, and read its second argument with the wrong
  layout (a crash). The declaration's own parameters shadow the enclosing ones now (shadowTypeParameters); a
  type alias's arguments are read before. Test `00generic_shadowed_type_param.ts`.

Still open, looked at:
- -mm=own: a local moved inside two `if`s one after the other (`if (a) keep(k); if (b) keep(k);`) is "moved here on
  some paths only" - so the dictionary copies its keys rather than taking the caller's; a spread of a record into a
  dictionary is refused (its fields would have two owners).
- Later: a spread of a dictionary into one of another value type (`{ ...ints }` of `int`s where numbers are wanted)
  calls `__dictionary_assign_values<T, S>`, made from source once a module, each value cast (it was an error: the
  library's objectDictionaryAssign takes two of one type). A call, not a loop in place: under -mm=own the dictionary
  made there is lent to it. (Under -mm=own a dictionary of a union holding strings is refused anyway: "a value that
  borrows the result of 'ObjectDictionary<number | string>.get' cannot be stored", open.)
- `a[a.length] = x` on an array did not grow it (gc wrote past the end; rc and own crashed). Fixed under gc, rc and
  none: the element store grows the array to `i + 1` first when `i` is at or past the length
  (mlirGenGrowArrayForWrite, SetLengthOf: the slots in between read as the element type's default; a negative
  index grows nothing). Test `00array_write_grows.ts`. Later under -mm=own too: there the growth is a
  SetLengthOf in place, unconditional and marked as only growing (ARRAY_GROWS_ONLY_ATTR_NAME: its lowering keeps the
  longer length, a negative index grows nothing), which -mm=own's passes read as a push (growsOnly). As the call
  the other models make, they read the array's slot given to it as kept ("'src' is a parameter, kept here") and
  the elements a borrow reads as dropped; in a branch between the value stored and its store, the store as a
  second reference. Not through a local over a parameter's array (`let src = order; src[i] = v`): changed in place
  through it - a push too - the local owns what it holds and the parameter is kept, so there the write is as before
  (open). Nor through a handle's payload (`s.value[i] = v`): a push there may move the elements another handle's
  borrow reads (spec 23.3), so the library's TypedArray.set (and MessageQueue.close, the workers) were taken for
  methods that release what their callers borrow, and 8 library tests stopped compiling under -mm=own ("a value that
  borrows 'this.elements' is used here after it may be released"); the write there is as before (open). Test
  `00array_write_shared.ts`. A parameter's array, a field's of a parameter, arrays of objects and of arrays: flat at 6 MB over 200000
  rounds (gc, rc, own), clean on the wasm checked heap (own, rc). When the right side may give the place another
  array - itself (`a[1] = (a = [5], 9)`), another place (`h.a[5] = (h.a = [7], 8)`), a call when the place is not a
  local - the store goes into the array the left side read, and that one grows, held in a slot of its own (it grew
  the new one, and the store went past the old one's end: an out-of-bounds write, garbage under rc). Under -mm=rc
  that array is retained from its read to the store when the right side stores an array (that store had released
  it: a freed block written to and grown, wasm's "realloc of ... not a block the heap gave out") - not for a call
  alone: the release after the store is not run when the right side throws, and `k.nums[i] = f(i)` of an `f` that
  throws leaked the array once a throw (so a call that gives the place another array, releasing the one read, stays
  open under rc). -mm=own grows through the place as before and refuses the left side's array released by the right
  side. Test `00array_write_reassigned.ts` (compile, JIT, rc); flat over 20M rounds, and over 1M rounds of a right
  side that throws (rc).

Tests: `00object_dictionary.ts`, `00own_param_exclusive_moves.ts` (compile, JIT, rc, own).

153 (iterables and generators, the compiler side of the default library's iteration types - TypeScriptCompilerDefaultLib
#113, whose `Iterable<T>`, `Generator<T, TReturn, TNext>` and the rest are objects with `next`, 2026-11-21). Fixed:
- An array or a string given where an iterator interface is wanted (`sum([1.5, 2])` of `sum(v: Iterable<number>)`,
  `cat("abc")` of `cat(v: Iterable<string>)`): "type 'number[]' can't be cast to interface 'Iterable<number>': it has
  none of the interface's members". A generator over its elements is given - `__array_values<T>` /
  `__string_values<T>`, made from source once a module, generic so that a pass whose module is thrown away does not
  leave a call to nothing - its element the interface's (`next`'s result's `value`): an `s32[]` given to an
  `Iterable<number>` is converted first (castIterableToIterator). Under -mm=own the generator borrows the array.
- A generator typed `Generator<T>` (`Iterator<T>`, `Iterable<T>`, `IterableIterator<T>`) yields T: its next() is
  declared `{ value: T; done: boolean }`, so `yield 1` is a number. It was typed by what it yields, the literal's
  s32: "can't assign '() => {value:s32, done:boolean}' to '() => {value:number, done:boolean}'".
- A value returned by a generator typed `Generator<T, TReturn>` with a TReturn other than T (and not void,
  undefined or any) is an error at the `return`: next() gives `{ value: T }`, and `return "end"` in a
  `Generator<number, string>` was cast to a number without a word (it read 0).

- next(value) of a generator typed `Generator<T, TReturn, TNext>`: its next() takes `value?: TNext`, and the
  `yield` it resumes evaluates to the value, a TNext (TNext's default - 0, null - when next() is given none: the
  first one, a for...of). It was undefined.
- An object-literal method given as an interface's function member of another arity - fewer parameters (`next()`
  for `next: (value?: number) => ...`, the rest ignored, as JavaScript does) or more optional ones (`next(v?)` for
  `next: () => ...`): "can't assign '() => ...' to '(p1: number | undefined) => ...': it takes 0 argument(s), 2
  expected". The vtable's slot holds a thunk giving the method as many arguments as it takes
  (isArityAdaptableObjectMethod, generateSynthMethodToCallField), as for a field holding a closure. So a generator
  whose next() takes no value is a Generator whose next takes one, and the other way round. The library's
  `Generator` declares `next: (value?: TNext)` once a compiler with this is released (a released one without it
  can't give a plain generator as one).
- return(value) and throw(e) of a generator typed `Generator<T, ...>`, as JavaScript's: paused at a yield, it
  resumes there finishing (`return`, so its finally blocks run) or throwing e (its own try/catch may catch it, and
  it goes on to its next yield); one not started yet, or finished, does not run - return(value) gives `{ value,
  done: true }`, throw(e) throws e. The generator object holds a mode its yields check where they resume
  (addGeneratorResumeMethods, mlirGenGeneratorResumeChecks). Under -mm=own return() only for a T that owns
  nothing (a number, a boolean): its result would hold the value the caller lends. The library's `Generator`
  declares them optional (`return?`, `throw?`: a generator declared otherwise has neither), and its next() takes no
  value: declared `next: (value?: TNext) =>`, a Generator given to an Iterable<T> (`next: () =>`) was let through
  as it is - castInterfaceToInterface reports a member of another type only when the vtable must be remapped, and
  `next` is the first slot of both - so an Iterable's next() called a function of another signature: wasm's
  "function signature mismatch" (the library's iteration_types.ts), natively a yield reading a value never given
  - fixed: such a cast is refused when a member function is called with another signature - another number of
  parameters, or a parameter or result passed otherwise (`(value?: number) =>` for `(value?: any) =>`: a
  Generator<T, R, number> given to a Generator<T> trapped too) - castInterfaceToInterface, refusedMember,
  passedAlike: members of other types passed alike, `mutrec`'s `x: A` and `x: B`, a class, an array, a string, any,
  are read as before; and an
  iterator given as another whose `next` is of another type is iterated by a generator over its values
  (castIteratorToIterator, `__iterator_values`: a while over next(), not a for...of - under -mm=own the for...of's
  read of the kept iterator across the yield was "a value that borrows a field ... may be released"). The library
  declares `next: (value?: TNext) =>` again. Tests `00generator_iterable_adapter.ts` (compile, JIT, rc, own),
  diagnostic-location `interface_member_other_type.ts`. Under -mm=own a call of an optional function-typed member of an interface (`g.return(v)`)
  was read as moving the object: the method value bound to it is wrapped in an optional, given out of the If that
  tests the member (lowered by then: a branch's block argument), tested and unwrapped for the call - all of which
  only calls it now lends the object (isBoundForCall, boundOnlyCalled), as a call of a member that is not optional
  does. Test `00interface_optional_member_call.ts` (compile, JIT, rc, own).
- -mm=own: an object literal's field of `<any>undefined` or `<any>null` (`{ t: <any>undefined, n: 1 }`) was
  released twice: own_facts::isConstantData took null or undefined seen through its casts for data nothing owns,
  a box into `any` included - so the store of the box into the object was not checked as a move, and the
  temporary's release gave the box back as well as the object's. wasm's heap caught it ("free of ...: not a block
  the heap gave out"), and Linux under -mm=own; Windows did not. A cast into `any` or `unknown` is a block of its
  own. (The generator object's `.thrown` holds `<any>0`, from before this was found.) Test `00any_undefined_field.ts`.
- Found on the way, generators and `try` (all memory models):
  - A throw or a call after a resumed yield inside a `try` went past it - its catch did not catch, its finally did
    not run (`try { yield 1; throw "x"; } catch (e) {...}`: "uncaught exception"). A yield splits the body at its
    label, and only the body's first block got unwind edges (TryOpLowering: `continue` in a `do ... while
    (false)`).
  - A throw after a yield in a loop in a `try` did not compile ("operand #0 does not dominate this use" at the
    `try`): the catch-type constants the landing pad reads were made at the start of the body, which a resumed
    turn does not pass - in a generator they go to the function's entry.
  - A `let` in a `try` read after a yield did not compile, the same error: its storage was put in front of the
    try, not of the generator's switch (GenContext::generatorSwitchOperation).
  - A catch variable of a generator was a field of the generator object, and its use crashed: the catch handler
    copies the exception into a slot of the frame. It is a local of the frame now, living in one turn.
  - A yield in a catch block is an error at the yield: resuming there is resuming inside the exception handler
    (it crashed in LLVM's instruction selection; before the fixes above the catch did not catch).
- A yield in a finally block was skipped (`try { yield 1; } finally { print("f"); yield 2; }` gave 1 and not 2,
  and "f" was not printed): TryOpLowering copies the finally for each way out of the try (the end of its body, the
  unwind, each return/break/continue), each copy with the yield's resume label, and the generator's switch
  dispatches by position - the copies shifted every label after them, so the resume landed after the yield in
  another copy. First an error (#980); then, USER DECISION, made to work as JavaScript's
  (mlirGenTryWithYieldingFinally): the try and its catch clause in a labelled block with a catch-all keeping the
  exception; each `return` in them (and return()'s at a resume point), and each `break`/`continue` out of them,
  records its way out in a mode variable and leaves the block (YieldingFinally); the finally once after it, as code
  a yield resumes in; then the way out taken. A return's value is kept in a variable of the generator; one through
  two such finallys is handed on with the same one. return(v) gives what the finally yields and finishes with v
  (kept in `.thrown`). A repeated resume label is an error in the lowering rather than a resume somewhere else.
  return() while paused in a yield* finishes through the finally too (the delegate is left where it is: its own
  finally blocks do not run). Under -mm=own, open: an object thrown through such a finally stops with "cannot copy
  an 'any' that holds an object" (the exception kept is an `any`; numbers and strings work); a value that owns
  memory returned through it is an error at the return (what is returned would be a borrow of the generator's
  field); return() in a yield* is as before (and a yield* of a Generator<T> is refused under own already: "a value
  that borrows a field is used here after it may be released", before these changes too). Also: a `let` of a nested block hiding one around
  it in a generator shared its field of the generator object (the outer `x` read the inner's value after the
  block): a field of its own now. Tests `00generator_yield_in_finally.ts` (results node's), `00generator_finally.ts`.

- Later: a generator is its own iterator - `g[Symbol.iterator]()` (`[Symbol.asyncIterator]` of an async one) is g
  itself, made at the call: a generator's object has a step and no member of that name. It was "Can't resolve
  property 'iterator'". A method `[Symbol.iterator]() { return this; }` on the generator object was tried before
  and backed out: a generator made inside another's machinery (an iterator's builtin map/filter) resolved that
  `this` through the outer one's alias, and under -mm=own a method returning `this` is refused. Test
  `00generator_symbol_iterator.ts` (compile, JIT, rc, own).

Still open, looked at:
- -mm=own: next(value) of a generator sending it an object (`Generator<T, void, Box>`): the generator keeps what it
  is sent (`const b = yield x` is a variable of the generator), and next() is called through the Generator
  interface, whose callers can't be made to give it up. Sent as a Shared<T> it works; one of another type is now an
  error at the yield that says so (it was "'this value' is a parameter, kept here"). Tests
  `00generator_next_shared.ts`, diagnostic-location `generator_next_object_own.ts`.
- -mm=own: a generator typed as an interface of the program's own (an Iterable<T> with next) works now. An array
  given as an Iterable<T> was "a value takes a second reference": the generator made over it would hold a borrow of
  the array for as long as it lives, which -mm=own can't say. USER DECISION 2026-11-21: under -mm=own it iterates a
  copy - `__array_values_owned<T, I>` borrows the array, pushes its elements into a local and moves that into
  `__array_values_of_copy<T, I>` (@keeps), declared to return the interface: a generator object becomes an interface
  only at its own function's return under -mm=own. A string the same way (`__string_values_owned`). An array of
  objects can't be copied there (refused); a for...of over such an interface of strings was still "a second
  reference" (fixed: 158). Test `00iterable_array_copy.ts` (compile, JIT, rc, own). (A generator typed as such an interface
  with yields of integer literals is typed by them, not by T, in every model: only the library's names give T.)

Tests: `00iterables.ts`, `00generator_typed_yield.ts` (compile, JIT, rc),
diagnostic-location `generator_return_of_other_type.ts`; `00generator_next_value.ts`, `00object_method_arity.ts`,
`00generator_return_throw.ts` (compile, JIT, rc, own), diagnostic-location `generator_yield_in_catch.ts`.

154 (compile time of a function of many `any` operations, 2026-11-21). The tsc sweep's
binaryArithmeticControlFlowGraphNotTooLarge - 1300 lines of `any` arithmetic in one function - timed out now and
then: not flaky, a steady 45 s (gc) against the sweep's 60 s with 8 jobs, and ~27 min under rc.
- gc (#976): a discovery pass made a generic instance again for every call (`___any_key<any>` for each `x[k]`), a
  module walk after each `any` access, OwnedReturnConsumptionPass and UnwindReleasePass walking the rest of a
  block, or a handler, once for each temporary: 45 s -> 4.7 s.
- rc: OwnedReturnConsumptionPass gave a discarded temporary back where its stretch of the block ends, so every
  temporary was live across every call after it, and each call that may throw got a cleanup pad giving all of
  them back - quadratic code (UnwindReleasePass ~8-11 min, LLVM's optimizer ~14 min). Now a temporary is given
  back right after the op of its block that uses it last (lastUseInBlock: its users' results unless they come with
  a reference of their own, views followed even then, what a conditional hands on, loads of a local it is stored
  into; a use that can't be followed keeps the old place). Not a named `const` (a local: `Shared.count`, a
  WeakRef see it let go of) nor a view (its release is the block it shows: an interface of an object literal's
  copy freed the literal before its next use). UnwindReleasePass skips a value whose release dominates the call.
  ~27 min -> 10.6 s; the cleanup releases grow linearly. rc only: own's passes read where the releases are.

Verified: ctest (8781), the wasm32 rc tester suite debug and release (763 each), mm_fuzz run (551 programs) and
leaks (80): no failures.

155 (a `T | undefined` that has no value, put into a string or into arithmetic, 2026-11-22). A `string | number |
undefined` put into a string - print, `"" + u`, `u + "!"`, a template, `<string>u` - gave its union's default, "0"
(and a "casting from optional" warning with no location), not "undefined": a union that may be missing was
unwrapped to its default before the cast (castToOptionalType's union path, and unwrapForBinaryOp for `+` with a
string). Now the cast of it to a string tests for a value, as a `number | undefined`'s did. Arithmetic on a `number
| undefined` that has none (`m + 1`, `-`, `*`, `/`, `%`, `**`) gave the default too, 0 - now NaN, as JavaScript
(unwrapOptionalOperand); a bitwise or shift operator reads undefined as 0 there as well, so it keeps the default,
without the warning. Tests `00optional_union_to_string.ts`, `00optional_number_arithmetic.ts` (compile, JIT, rc, own).

Verified: ctest (8842), the wasm32 tester and default-library suites (rc, none, own; the library with `next(value?)`
back), the library's tests.ps1 release (gc, rc, none, own), mm_fuzz run (551 programs) and leaks (80): no failures.
The tsc sweep: 3261, two fewer - identicalTypesNoDifferByCheckOrder and
nongenericPartialInstantiationsRelatedInBothDirections cast an interface whose call (construct) signature takes a
tuple, passed by value, to one whose takes an interface: a call through it would pass the argument otherwise, now
refused (castInterfaceToInterface, passedAlike).

156 (return() of a generator paused in a `yield*`, 2026-11-22). A `yield*` is a loop over its delegate whose yield
return() and throw() resume at; return() finished the generator there and left the delegate where it was, so the
delegate's finally blocks never ran (JavaScript calls its return()). Now a delegate whose type has `return` (a
Generator) is held in a variable (`.yd`, mlirGenYieldStar), and the resume check at that yield calls
`.yd.return?.(value)` with the value the generator's return() was given before it finishes
(mlirGenGeneratorResumeChecks, GenContext::yieldStarDelegate): the delegate's finally blocks run, then the
generator's own. Open: throw() still throws at the yield rather than handing the exception to the delegate's
throw() (which may catch it and go on yielding); -mm=own, where a yield* of a Generator<T> is refused already. Test
`00generator_yield_star_return.ts` (compile, JIT, rc).

157 (-mm=own: an object thrown through a finally that yields, 2026-11-22). The exception is kept in an `any` for
after the finally (mlirGenTryWithYieldingFinally), and an `any` that holds an object can't be copied under -mm=own,
so a class instance thrown there stopped at run time ("cannot copy an 'any' that holds an object"). USER RULING
2026-11-22: a clear compile error. It is said at a `throw` whose value's type is an object (owns memory and is not a
string, a number or an `any`) when it gets to such a finally - not caught first by the try's own catch clause or a
try inside it (GenContext::throwCaughtBeforeYieldingFinally): "-mm=own: an object of type 'Failure' thrown through a
finally block that yields is not supported: catch it inside the try statement, or throw a string". Open: an object
thrown by a called function, or a rethrown `any` holding one, still stops at run time; the complete fix would move
the caught value into the kept slot rather than copy it. Tests `00generator_throw_through_yielding_finally.ts` (all
models) and diagnostic-location `generator_throw_object_through_yielding_finally_own.ts`.

158 (-mm=own: a for...of over an Iterable<string>, a dictionary of `number | string` read and stored, 2026-11-22;
USER RULING: both are to work, not to be errors).
- `function cat(v: Iterable<string>) { for (const c of v) s += c; }` was "a value takes a second reference" at the
  `for`, for an array, a string or a generator given: next() called through the interface was not known to hand
  over the `{value, done}` it returns - a closed interface's members keep their own facts (spec 31), and a
  generator's next reads as borrowing its state. Every function behind such a slot that is a generator's own next
  (isGeneratorNext: named `next`, its first parameter a generator's state object, its result the record) moves what
  it yields into the record - a yield of a borrow is an error in its body - so the call's result is fresh, as a
  direct call's is (isIteratorNext). `let r = v.next(); r = v.next()` by hand works the same way.
- `{ [k: string]: number | string }`: `e["z"] = d["c"]`, `arr.push(d["c"])`, `bx.v = d["c"]`, `l = d["c"]` were "a
  value that borrows the result of 'ObjectDictionary<...>.get' cannot be stored, returned or captured". A read
  borrows the entry, and a string's is copied where it is kept; this one is an optional of a tagged union,
  `(number | string) | undefined`, which isCopyableString left out. Now ts.StringCopy copies it (the union when the
  flag says it holds one, its string when the tag says string), and `ts.ValueOrDefault` of text - what a store into
  a `number | string` reads the optional through - is a view of the read, so the store copies it. A union that
  holds nothing (the zero one ValueOrDefault gives for a key the dictionary does not hold) has no tag: the copy
  tests for it (it read the descriptor of a null tag: a crash in an executable).
- Found by the 1M-round check, leaks under -mm=own before these changes too: an array literal of new values given
  straight to a call - `f(["x" + i, "y" + i])`, 95 MB a million calls - was given back by nothing, with its
  strings (a literal of numbers was, and one of values borrowed for the call is given back shallow, spec 49): now
  owned when each element that owns memory is a new value moved into it (holdsOnlyMovedInElements), and released
  after the call or moved into a callee that keeps it. And the array a conversion makes for a call (`f(a)` of an
  `int[]` given as a `number[]`, or as an Iterable<number>): a `ts.NewArray` filled by a loop, which the "its one
  use is the call" test did not see (isConvertedArrayFor).

Still open, looked at: `x == 1.5` of an `x?: number | string` (a dictionary's read compared with a number) is "a
value is merged from several branches" under -mm=own - `___union_eq_equals` over the optional merges a new string
with a literal; compared through a local of the union it works. A generator that yields its object's own strings
(`*each() { for (const s of this.items) yield s; }`) is refused under -mm=own as before ("a value that borrows a
field cannot be stored, returned or captured"): it does not copy what it yields. Leaks -mm=own still has for an
array literal given straight to a call (as before these changes, a million calls each): one of new objects
(`f([new C(), new C()])`, 140 MB - an object's other uses, its constructor's call, are not followed), one mixing a
new value with a named one (`f(["x" + i, s])`, 95 MB), and any of them when the call throws (220 MB: the array is
released after the call, not on the way out of it).

Tests `00iterable_string_for_of.ts`, `00dictionary_union_values.ts` (compile, JIT, rc, own).

Verified: ctest (8855), the wasm32 tester and default-library suites (rc, none, own), the library's tests release
under -mm=own (236), a million rounds of each shape at a flat 5.5 MB: no failures.

159 (-mm=own: `yield*` of a Generator<T>, 2026-11-22; USER RULING: to be supported, with the delegate kept).
`function* outer(): Generator<number> { yield 100; yield* inner(3); yield 200; }` was "a value that borrows a field
is used here after it may be released or overwritten" at the `yield*`, and so was any `d.next()` on a generator held
in a field (`this.d.next()` of a class).
- The call was not resolved: a `Generator<T>`'s `next` is a member of function type, and what implements it for a
  generator is not a field of its object but a thunk (`next(value?)` over the generator's own `next()`), kept in a
  method cell `{method, flagOffset, fieldOffset}` whose tagged address the vtable slot holds. The scan of interface
  vtables reads such a cell now (slotMethodCellFunction: one always present and never written through), and a call
  of a function-typed member looks in the method families when no field family answers. `this.d.next()` of a class
  field compiles with that alone. Such a thunk counts as a generator's own next for what the call hands over
  (forwardsGeneratorNext, with 158's isGeneratorNext).
- In a generator that is not enough: its variables are fields of its object, the `yield*` loop called next() on a
  delegate borrowed from one, and what that next() runs may, as far as its type says, be this same generator
  function's next() - a generator delegating to itself - which gives that field a new value. Fields are told apart
  by position and type, not by object. So under -mm=own a `yield*` of a delegate a call makes, seen through an
  interface, keeps it in a variable of its own and exchanges it with a local of next() for each step
  (`[.yl, .yd] = [.yd, .yl]`, the documented way to take a value out of a place): the local owns the delegate for
  the call, nothing is borrowed from the generator, and a `finally` exchanges it back - also when next() throws
  (it leaked there). The local is a real local of next() (InternalFlags::StackLocal on its declaration), in a
  block that ends before the yield. mlirGenYieldStar; the other models, and a delegate that is an array or a
  string, are iterated as before.
- return() and throw() of the generator are checked for at the yield of such a `yield*` too (they were not, under
  -mm=own), and return() calls the delegate's return() first, exchanged the same way: its finally blocks run, as
  156 has it for the other models.

Still open, as before these changes: a `yield*` of a delegate read out of a parameter or a variable (`yield* it`,
which the generator does not own), and of a generator made by a method of `this` in a class's generator method
(`yield* this.range()`: it borrows `this`, and can't be kept in a variable of the generator); a `d.next()` written
by hand on a generator a generator's variable holds; a Generator<T> of objects (156's return() is for a T that
owns nothing). In every model: a generator function that names itself (`yield* deep(d - 1)` in `deep`: "can't
resolve name"), a method likewise; one that goes through another function works.

Tests `00generator_yield_star_delegate.ts` (compile, JIT, rc, own), `00generator_yield_star_return.ts` now under own too.

Verified: ctest (8867), the wasm32 tester and default-library suites (rc, none, own), the library's tests release
under -mm=own (236), mm_fuzz run (914 programs), a million rounds of each shape at a flat 5.7 MB: no failures.

160 (a generator that names itself, 2026-11-22). `function* deep(d: number): Generator<number> { yield d; if (d > 0)
yield* deep(d - 1); }` was "can't resolve name: deep", and so was the call kept in a variable or iterated by a
for...of; a method's `yield* this.left.walk()` was "Class member 'walk' can't be found" - in every memory model. A
generator is rewritten into a wrapper function that returns the generator's object, whose next() is the body; the
variables of the body are fields of that object, so the body is discovered in full, with every type, before the
wrapper has a type of its own - and a function is entered under its name only once its type is found (an ordinary
function's call of itself is let go unresolved while its type is discovered; a for...of over nothing can't be).
- A wrapper with a declared type (`: Generator<number>`, InternalFlags::GeneratorWrapper) has its type before its
  body: mlirGenFunctionPrototype enters it by its name - a method as a method of its class - before the body is
  discovered. The entry stays when the body can't be generated yet, so two generators that name each other
  (`ping` and `pong`, a class's `up` and `down`) find each other on the next try; a method's is replaced by the
  method's own registration (mlirGenClassMethodMember, GENERATOR_SELF_METHOD_ORDER).
- One with no declared type is refused as before ("can't resolve name"): the type of the call is what is being
  found. tsc refuses it too under noImplicitAny (TS7023).

Tests `00generator_names_itself.ts` (compile, JIT, rc, own), `00generator_names_itself_methods.ts` (compile, JIT, rc:
the call in a for...of and the methods, which -mm=own does not take yet - 159).

161 (the items 158-160 left open, 2026-11-22). Fixed:
- -mm=own: `x == 1.5` of an `x?: number | string` (a dictionary's read compared with a value) was "a value is merged
  from several branches". The helper that compares a union member by member (`___union_eq_*`) narrowed the optional
  itself, and its string member compared with a number made, for an empty one, a text beside the converted one. Under
  -mm=own the empty case is tested where the comparison is (not equal to any value), and the union it holds is
  compared as any union is (binaryOpLogicForUnionAndValue).
- A call that throws into a `catch` of its own function gave back nothing its frame held in a loop's body - the
  argument's temporary string, an array literal - under -mm=rc and -mm=own (`try { f("x" + i); } catch {}`: 80 bytes
  a turn). UnwindReleasePass releases at an invoke's pad only what "nothing reachable from the pad uses again",
  and the handler goes back round the loop, where the same op makes the next turn's value and uses that: counted
  as a use. Now a path that passes the value's definition again is not one (usedAgainFrom).
- A generator's exception through a frame with no `try` gave back nothing either (`function run() { const g = gen();
  return sum(g); }`, 80 bytes): a call that reaches a generator's next() was taken not to throw - next() is a field
  of the generator's object, its address in the object's constant, and only `ts.SymbolRef`s counted as functions an
  indirect call may reach. The functions an object constant holds count too (computeMayThrow).
- -mm=own: an array literal given straight to a call was given back only when every element was a new value used by
  nothing else (158). Also now: one of new objects (`f([new C("a"), new C("b")])`, 140 MB a million calls - the
  object's constructor call and the like come before the literal, isMovedIn), and one that mixes a new value with
  text something else owns (`f(["x" + i, s])`, 95 MB: the literal is given a copy of `s`, copyBorrowedTextElements).
- -mm=own: `yield* it` of a parameter of the generator (a Generator<T>, an Iterable<T>) was "a value that borrows a
  field is used here after it may be released or overwritten": the loop called next() on a copy of the parameter kept
  in a variable of the generator. next() is called on the name itself now, as a loop written by hand does; return()
  of the generator calls the delegate's return() first (mlirGenYieldStar).
- -mm=own: a `Generator<P>` of objects did not compile ("a value that borrows the result of '___unbox<...>' cannot be
  stored, returned or captured"): return(value: P) keeps its value in an `any` for the yield it resumes at, which an
  object can't be read back out of as its owner. A generator whose T owns memory and is not text has no return() and
  throw() under -mm=own (the interface's are optional); it is given up by being released.

Still open, looked at:
- -mm=own: a `yield*` over a Generator of objects (`yield* people(k)`: the value is read out of the step's record
  and yielded - an object can't be copied out of it); `yield* this.range()` in a class's generator method and a
  for...of over `this.items` in one (what is made borrows `this`, and can't be kept in a variable of the generator -
  the consumer may change `this` between two steps; read by index, `this.items[i]`, it works); a `d.next()` written
  by hand on a generator that a variable of a generator holds (fields are told apart by position and type, not by
  object: 159). An array literal with an object something else owns (`f([new C(), c])`) moves `c` in, as before, and
  is not given back.
- every model: a generator that names itself with no declared type ("can't resolve name": the type of the call is
  what is being found; tsc refuses it too, TS7023). `"1.50" === 1.5` of a `number | string` member is true (the
  text is converted), as before (fixed by 162).

Tests `00optional_union_compare.ts`, `00array_literal_argument.ts`, `00generator_yield_star_parameter.ts` (compile,
JIT, rc, own).

162 (leftovers of 108, 147 and 161, 2026-11-23). Fixed:
- `this` in a static method is the class it is called through: `Derived.me()` of `Base`'s `static me() { return this; }`
  is `Derived`, `this.label` reads Derived's static, and `new this()` makes a Derived. A static method is compiled
  once, with `this` the class that declares it, so a subclass now gets a copy of its own of each base class static
  method that reads `this`: the base's declaration compiled as if the subclass declared it (an override in its
  vtable too, so a call through a `typeof Base` holding Derived reaches it; addInheritedStaticsReadingThis). Not
  one that reads `super` (that would be the base's base), one of a generic base, one declared in another file or
  namespace, or one the subclass declares itself: those keep the base's method, as before.
- `const \u{61} = 1` scans: `tokenValue = scanExtendedUnicodeEscape() + scanIdentifierParts()` left the order of
  the two calls to the compiler, and MSVC scanned the rest of the name first, which took the escape's hex digits.
  Also, plain characters between two braced escapes in one name (`\u{61}x\u{62}`) were dropped. Still open: a
  private name with any escape (`#\u{78}`) is not found again by `this.#x`.
- `===` and `!==` between a union of primitives and a value (`u === 1.5` with `u: number | string`) converted the
  member it held to the value's type first, so `"1.50" === 1.5` was true. A member of another kind is now not
  equal (isComparableForEquality, strict), `switch` cases of a union value compare the same way, and `==` still
  converts.

Tests `00tslang_issue_108.ts`, `00tslang_issue_147.ts`, `00tslang_issue_161.ts` (compile, JIT, gc, rc, none, own).

163 (leftovers of 133 and 114, and three looked at, 2026-11-23). Fixed:
- A rest parameter typed as a tuple (`fn(...args: [string, number])`) is called with separate arguments
  (`fn("a", 1)`): they make the tuple, each as its element's type, and a tuple spread in (`fn(...t)`) is passed
  as it is. A wrong count says how many the rest parameter takes. It failed with "'ts.CreateArray' op result #0
  must be ...".
- `arguments` in a function with a rest parameter is the other arguments followed by the rest's elements, and in
  one with a destructured parameter that parameter's argument. `arguments` is now held by a local of its own: read
  as a temporary (`for (const v of arguments)`), the array leaked under -mm=own (300 MB for 2M calls) and, with a
  rest parameter, under rc, where a spread literal read as a temporary is never given back (`for (const v of
  [...xs])` leaks the same way, open). Not in a loop's own condition (`i < arguments.length`), which has no scope
  to give a local back: there it stays a temporary, as before.

Looked at, still open:
- 112 (`typeof options.label === "string"` of a union of two object types left an `unrealized_conversion_cast`):
  `{ label: string; alias: string } | { label: object; alias?: string }` has no tag - both are laid out as
  pointers - so nothing at run time says which one a value is, and `typeof options.label` can't be answered. The
  fix is a tag for a union of object types whose fields of one name differ in type, and a per-member conversion
  where a property of the union is read.
- 148 (`const a = { get self(): any { return a; } }`): every function in an initializer that reads the variable
  being declared fails ("can't resolve name: a"), with a written type too
  (`const a: { m(): number } = { m() { return a.v; } }`). Untyped, tsc refuses most of these as well (TS7022);
  with a type written, the variable has to be declared before its initializer is evaluated.
- 103 (`Hello2 = Hello1`, a field given an enum, "operand does not dominate"): the field is typed as the enum's
  member type, not the enum object, so `f.Hello2.World1` doesn't resolve either; `const h = Hello1` works.

Tests `00tslang_issue_133.ts`, `00tslang_issue_114.ts` (compile, JIT, gc, rc, none, own).

164 (112 and unions of object shapes, 2026-11-23). USER RULING: keep a tag where nothing else tells the members
apart; none where the first field or the class does. Fixed:
- A field of a tagged union of shapes, classes or interfaces is read from the member the value holds (the tag; a
  class asked by instanceof). It was read as the first member: a field at another position read garbage silently
  (`s.kind` of `{ size; kind: "sq" } | { r; name; kind: "ci" }` printed null).
- 112: shapes with the same field names whose fields are stored differently (`label: string` vs `label: object`)
  were merged into one shape stored as neither; they stay a tagged union. An object literal without an optional
  field goes to the member it fits.
- `v instanceof A` of `IA | IB` asks the interface it holds (it said false), the narrowed class is that interface's
  object, `<IB>v` is the interface (the compiler crashed).
- No tag where the data tells: shapes whose first field, of one name and type in each, is a literal of its own are
  stored as the largest shape (DiscriminatedUnionLogic reads the field); a union of classes (null perhaps among
  them) is the object's pointer, the class asked of `.instanceOf` (tsunm_ routine), unless a @cpp, @value or static
  class is in it (CLASSES_WITHOUT_RTTI_ATTR_NAME). A layout change: the default library is built again
  (`DataView.source` is `ArrayBuffer | SharedArrayBuffer`; an old library's `DataView..size` clashes at link).

Still open:
- `<object>v` of a class or an object literal is the address of a stack slot holding the value
  (CastLogicHelper::castToOpaqueType): kept past its function under rc or own (`return l` of `object | string`) it
  crashes. 00tslang_issue_112.ts runs under gc and none for that.
- -mm=own: an object field read through a union of shapes can't be returned (a value merged from branches); an
  object literal without an optional field given to a union of shapes is refused (its fields borrowed from the
  literal's record); an interface narrowed to a class is refused ("kept here"), as for one interface.
- A subclass object given to a union of its bases (`return new Big()` for `Sq | Ci`, Big extends Ci) is refused:
  "none of the union's types".
- `<Shape>a` of an `any` holding a union of shapes: "can't be stored".

Tests `00union_shape_reads.ts`, `00union_first_field_tag.ts`, `00union_classes_no_tag.ts` (gc, rc, none, own),
`00union_interfaces_instanceof.ts` (gc, rc, none), `00tslang_issue_112.ts` (gc, none).

165 (leftovers of 103, 162 and 164, 2026-11-23). Fixed:
- A subclass object given to a union of its bases (`return new Big()` for `Sq | Ci`, Big extends Ci) is the
  nearest base the union has; `instanceof` asks the object's own class. It was "none of the union's types".
- `<Shape>a` of an `any` holding a member of a union of object shapes is the shape whose layout the box's tuple has
  (ts.IsTupleOfLayout, one shape after another; the last is checked as `<{...}>a` is, and throws if it is none).
  It was "'ts.Cast' op type ['!ts.any'] can't be stored". Not under -mm=own: each check reads the box again, which
  own counts as a second reference (one shape, `<{ kind: "ci"; r: number }>a`, works there).
- `<object>` of an object literal was the address of a stack slot holding it (CastLogicHelper::castToOpaqueType):
  returned from its function it pointed at a dead frame, and -mm=rc counted it as a heap block (`id(o)` of
  `<object>{}` crashed). It is a block of its own now, freed alone: an `object`'s fields are never read back,
  so the block takes no reference to what they hold. A class object, and an `object` of a known shape, cast to
  `object` is the same pointer (a class object was a stack slot's address too: `<object>l` returned and kept
  read a dead frame under rc). 00tslang_issue_112.ts runs under rc too. Under -mm=own `<object>` of a literal
  is as before (refused where it is returned: "a second reference").
- A private name written with an escape (`#\u{78}`, `#y\u0031`) is the name it spells: the parser made it of the
  token's source text, not of its scanned value, and `this.#x` did not find it.
- 103: a field given an enum itself (`Hello2 = Hello1`) is an error that says so, as for a `let` ("field 'Hello2'
  can't hold the enum 'Hello1' itself"); it was typed as one of the enum's members, and `f.Hello2.World1` was
  "Can't resolve property".

Still open, looked at:
- 148: a function in a `const`'s initializer reading that `const`, with its type written
  (`const a: { v: number; m(): number } = { v: 5, m() { return a.v + 1; } }`): "can't resolve name: a". The
  variable is registered once its initializer is generated (registerVariable); a written type would let it be
  declared first.
- `for (const v of [...xs])` leaks the spread array under -mm=rc (163).
- -mm=rc: a class object whose last reference is an `object` is freed without its fields (`<object>new L(i)` of
  an `L` with a string field, kept only as `object`: 72 MB at 1M, 134 MB at 3M). Nothing in an `object` says it
  is a class (releaseFields of `any` storage releases nothing).

Tests `00tslang_issue_165.ts` (compile, JIT, gc, rc, none, own), `00tslang_issue_165_any.ts` and
`00tslang_issue_112.ts` (gc, rc, none), crash-errors `class_field_enum_itself.ts`.

166 (leftovers of 142, 148, 153 and 165, 2026-11-23). Fixed:
- 148: a module variable with a type written is that type while its own initializer is generated, so a method or
  arrow there can read it (`const a: { v: number; m(): number } = { v: 5, m() { return a.v + 1; } }`, a `let` of an
  interface type the same): it was "can't resolve name: a". The name is registered as the global of the written
  type for that time (GenContext::globalWrittenType, createGlobalVariableInitialization) and given up after; the
  variable is registered as it ends up, as before. Untyped it stays an error (tsc's TS7022 too).
- -mm=rc: a module variable of type `unknown` given a value (`let u: unknown = 7`) was read as freed memory ("Can't
  cast from any type"): its initializer's `any` box was given back at the end of the statement, and an `unknown`
  owns nothing. The global takes a reference of its own to the box (a local still does not: it is read within
  the statement's scope; counting every `unknown` as an owner broke a construct signature's `unknown` result).
- `async function* g(): AsyncGenerator<number> { yield 1; }` (also AsyncIterator, AsyncIterable,
  AsyncIterableIterator): next()'s result is typed by T, as a sync generator's is, so an integer literal is yielded
  as a number. It was "can't assign '() => {value:s32, done:boolean}'".
- Looked at again, no longer failing: `for (const v of [...xs])` (rc and own flat at 1M and 3M), `x == "a"` of an
  `x?: number | string` (it crashed in lowering; a warning with no location is left), `static y = this`,
  `(Foo as any) = null`.

Still open, looked at:
- 148 for a local: `const a: T = { m() { return a.v; } }` in a function is "can't resolve name: a". Declared first
  (`let a: T; a = { ... }`) it is refused as an object literal whose methods capture the function's variables
  copied to a type literal; with an interface type that works.
- A type literal's accessors (`type T = { get foo(): number; set foo(v: number | string) }`) are a field `foo`: an
  object literal's or a class's accessors given to it are "field "foo" can't be found". A copy of a class object
  could not call its setter anyway; an interface keeps the object.
- -mm=own: a module variable of type `unknown` with a value is refused ("the initializer of global 'u' takes a
  second reference"); an arrow in a typed module `const` reading another of its fields (`who: () => self.name`)
  is "a value that borrows a field is used here after it may be released".
- -mm=rc: a class object whose last reference is an `object` is freed without its fields (165).

Tests `00tslang_issue_166.ts` (compile, JIT, gc, rc, none, own), `00tslang_issue_166_unknown.ts` (gc, rc, none).

167 (leftovers of 136, 142 and 166, 2026-11-23). Fixed:
- Narrowing an optional union (`typeof x === "number"`, `v instanceof A` of an `x?: number | string`, `v?: A | B`)
  went through a cast of the whole optional, which warned "casting from optional (undefined) value" with no
  location (`<unknown>:0`), the narrowed value read as the union's default where it had none. The test says there is
  a value: the union is taken out of the optional as it is and narrowed (addSafeCastStatement). `x == "a"` of such
  an `x` (the `___union_eq_*` helper) no longer warns either.
- A type guard given as a function value: `xs.filter(isDefined)` was "can't assign '(p0: number | undefined) =>
  number' to '(p0: number | undefined, p1: number) => boolean'", and `const p: (v: T) => boolean = isDef` compiled
  but a call through `p` failed ("'ts.Call' op result type mismatch"): the guard's result (`v is T`, a
  TypePredicateType) is a boolean to the adapter now (resultWidensTo), and one is made for a guard given as a
  function returning boolean.
- 136: a function taking a number given where integers are passed (`[1, -2, 3].filter(pos)` of `pos(v: number)`,
  the literal's elements s32): "can't assign '(p0: number) => boolean' to '(p0: s32, p1: number) => boolean'". The
  adapter converts the argument (adaptableSignature).
- An object or a class object with accessors given to a type literal's accessors (`type T = { get foo(): number;
  set foo(v: number) }`), which are a field: an error that says so and points at an interface ("an object whose
  'foo' is an accessor can't be given to a type literal ..."), instead of "field "foo" can't be found". A copy of
  it could not call the setter (`t.foo = 3` would not reach it); an interface's accessors are called.
- Looked at again, no longer failing: an object literal's generic method (`{ id<T>(v: T) { return v; } }`), a
  class declared in two blocks of one function under one name, `await` in a `try`/`finally` of an async function.

Still open, looked at:
- A constructor type as a parameter (`make(Banana, 3)` of `make<T>(ctor: new (n: number) => T, n)`): "'new' on a
  value of type 'new (p0: number) => any' is not supported: it is not a class" (a clear error).
- 148 for a local, a module `unknown` under -mm=own, and -mm=rc's class object freed through `object` without its
  fields, as 166 says.

Tests `00tslang_issue_167.ts` (compile, JIT, gc, rc, none, own), crash-errors `type_literal_accessor_object.ts`,
`type_literal_accessor_class.ts`.

168 (leftovers of 148, 166 and 167, 2026-11-23). Fixed:
- A class given to a generic constructor type (`make(Banana, 3)` of `make<T>(ctor: new (n: number) => T, n)`,
  `create<T>(c: new () => T)`, `T extends Animal`): T is the class's instance type (inferType). With T known, the
  parameter is the class value `typeof Banana` (getConstructorType), which `new ctor(n)` makes one of, a subclass
  too. It was "'new' on a value of type 'new (p0: number) => any' is not supported: it is not a class". An object
  given for it (`make(new Banana(1), 3)`) is an error, as before.
- 148 for a local: a local with a type written whose initializer has a function reading it (`const c: I = { m() {
  return c.v + 1; } }`, an arrow field `next: () => n.name`) is declared first and then given its initializer, so
  the function captures the variable. It was "can't resolve name: c". A function declaring the name itself (a
  parameter, a local) reads its own and changes nothing. Such a `const` refuses another assignment ("can't assign
  to 'c': it is a constant"). A type literal's type still says to use an interface (the object's methods would
  lose what they capture). Under -mm=own it is refused, as `let c: I; c = { ... }` is: the object would hold a
  closure that holds the object. Under -mm=rc that cycle is never freed (166 MB at 3M), as for the declared form.
- -mm=own: a module variable of type `unknown` with a value (`let u: unknown = 7.5`, `= "a" + 1`, `= { a: 1 }`)
  was "the initializer of global 'u' takes a second reference". The initializer takes its `any` box once and gives
  it back once; the two cancel and the box moves into the global (moveIntoGlobals).

Still open, looked at:
- An `unknown` variable owns nothing, under rc and own, so a value stored into one after its declaration (`u = "x"
  + i`) is not kept: under -mm=rc the box is given back at the end of the statement's scope and the variable reads
  freed memory (a module `let u: unknown;` assigned in one function and read in another after other allocations
  crashes); under -mm=own the old box is never given back (160 MB at 3M assignments). `any` is fine in both.
  Counting `unknown` as an owner needs the one value that is no box handled first: a class given to a construct
  signature with no return type (`f: { new(x: number) }` of `{ f: B }`, newWithSpread.ts) returns B's object as
  `unknown` as it is, with no adapter that boxes it.
- `typeof` of an `unknown` or `any` holding an integer literal is "s32" (integer literals are s32), not "number".

Tests `00tslang_issue_168.ts` (compile, JIT, gc, rc, none, own), `00tslang_issue_168_self.ts` (gc, rc, none),
crash-errors `const_declared_first_assigned.ts`.

169 (the `unknown` left open by 166 and 168, 2026-11-23). Fixed: an `unknown` owns the `any` box it holds, under rc
and own, as `any` does (ownsHeapMemory; its retain, release and share routines are `any`'s). It owned nothing, so a
value stored into one after its declaration was not kept:
- -mm=rc read it freed: a module `let g: unknown` assigned in one function and read in another after other
  allocations crashed.
- -mm=own leaked the box it replaced (160 MB at 3M assignments; a local `unknown` in a loop the same). Now flat at
  5 MB in both models.
- A box made for `unknown` is fresh: the cast to `unknown` carries its reference to the store (MLIRGenCast), and
  -mm=own sees `any` and `unknown` as one box both ways (isView), so `<B>u` of an `unknown` const borrows as of an
  `any` one. 166's -mm=rc retain on an `unknown` global's initializer and 168's -mm=own special case for it are gone
  (moveIntoGlobals follows the cast).
- A class given to a construct signature with no result type (`f: { new(x: number) }` of `{ f: B }`, which returns
  `unknown`) went through a wrapper returning B's object as it is, typed `unknown` with no box, which an owner of
  `unknown` would release as a box: the wrapper returns the field's `unknown` (or `any`), so the object is boxed
  (generatingStaticNewCtorForClass).

Still open, looked at:
- A generic class given to a constructor type straight from an instantiation expression (`make(Box<number>, 3)`):
  "Can't find class Box". The function is instantiated for `Box<number>` before the argument makes that class; the
  same class named first (`const k = Box<number>; make(k, 3)`) or given as the type argument (`make<Box<number>>(...)`)
  works. `make(Box, 3)` (the class's own type parameters from the constructor type) is not inferred.
- -mm=own: `typeof a === "string"` of a `const` holding an `any` (or `unknown`) a function returned is "a value
  takes a second reference" (a `let` works).

Tests `00tslang_issue_169.ts` (compile, JIT, gc, rc, none, own).

170 (the -mm=rc `object` left open by 165 and 169, 2026-11-23; USER RULING: under -mm=rc a class object given to
`object` is kept in a box). Fixed:
- -mm=rc: a class object whose last reference is an `object` was freed without its fields (165; 72 MB at 1M). An
  `object` said nothing of what it pointed at. Under rc an `object` is now an `any` box: a value given to one is
  boxed (a class object, an object literal, an array...), `any` and `unknown` are that box either way, its release
  is the box's, which frees the class object with its fields, and `===` with an `object` compares as `any` (by what
  the boxes hold: `<object>c === c`). Flat at 5.4 MB at 1M and 3M for a class object, an `object[]`, an `object`
  field, `object | string` and a boxed literal with run-time strings. gc, none and own keep the pointer.
- null or undefined as an `object` (`let n: object | null = null`) was the address of a stack slot holding null, so
  `n === null` was false in every model, and -mm=rc counted the slot (a crash): it is the null pointer.
- `a === u` of an `any` and an `unknown` holding the same box was false: the `unknown` was boxed again. An `unknown`
  is compared as the `any` it is.
- -mm=rc: an object literal with methods held by a `const` (`const user = { ..., sayHi() {...} }`) was freed by the
  first retain and release another holder made: `takeAny(user)` (a box of it) gave its block back, and `user` read freed
  memory (00bound_method_wider_callback.ts crashed once an `object` was a box). The block was made with a count of 0,
  and a `const` takes the reference a fresh value carries; it now carries one, as `new C()`'s does (a `let` took a
  reference of its own, which is why it worked). Flat at 3M with `const`, `let`, a returned literal and one given straight
  to a call.

Fixed by 171 for gc and none.

Tests `00tslang_issue_170.ts` (gc, rc, none); `00tslang_issue_169.ts` checks the null and `unknown` comparisons in every model.

171 (leftovers of 169 and 170, 2026-11-23). Fixed:
- gc and none box an `object` as rc does (170), so `object` reads the same in every model but own: an `any` taken
  from an `object` is what the `object` holds, so `o === a` is true and `<L>a` is the class object (it threw "Can't
  cast from any type"); 165's separate block for `<object>{...}` under gc and none is gone, the box replaces it.
- A generic class given to a constructor type with no type arguments (`make(Box, 4)` of
  `make<T>(ctor: new (n: number) => T, n)`, `makePair(Pair)` for `new (s: string, n: number) => T`): its type
  arguments come from the constructor type's parameters (inferClassTypeFromConstructorArguments, as `new Box(4)`
  takes them from its arguments), and the class given is that specialization (castToClassValue). It was "Can't find
  class Box". `make(Box<number>, 3)` already worked (169's note was about `make(Box, 3)`).
- Under --di, `const k = Box<number>` (a `const` naming a class, an interface, a namespace or a type) had a debug record of
  a reference with nothing to lower to: "failed to legalize operation 'ts.ClassRef'". It has none now.

Still open, looked at:
- -mm=own: `typeof a === "string"` of a `const a` holding the `any` a call returned is "a value takes a second
  reference" at the unboxed string's retain (a `let` works): own's inference does not take the string read out of a
  call's box held by a folded `const` as a borrow of that box.

Tests `00tslang_issue_171.ts` (compile, JIT, gc, rc, none, own); `00tslang_issue_170.ts` checks `a === o`, `u === o` and
`<L>a` again.

172 (older items looked at again, 2026-11-23). Fixed:
- A module `const` naming a class (`const k = B;`) failed the whole module with "failed to legalize operation
  'ts.ClassRef'", even unused: a module variable is a global, and a class's name can't be stored in one. It holds
  the class's value, as `let k = B` does (a local `const` still names the class itself), so `new k()` works. `class D
  extends k` is the clear error for a class value ("its class is known at run time only"); naming the class through a
  module `const` at compile time would need the name kept as an alias of the class.
- A `var` used before its declaration in a function (`x = 7; var x = one();`, 1384): "can't resolve name: x". It is
  declared before the body with the type its initializer gives, as a `var` read by an earlier function already was.
  Not under -mm=own, which refuses a variable declared without a value that is given one later.

Looked at, still open:
- -mm=own: `typeof a === "string"` of a `const a` holding the `any` a call returned (171): the IR own's inference sees
  is the box, an unbox in the branch and one release, the same as a cast in a branch, which compiles; the narrowing's
  own shape is refused inside the inference ("a value takes a second reference"). A `let` works.
- A generic arrow returning a generic interface given to a typed `const` (`const f: <p>(a: p) => W<p> = <p>(a: p): W<p>
  => ({ a })`, 1188): "can't find type by name: p".
- A namespace `var` declared in two `for` headers read by a function of the namespace (1392): "can't resolve name".

Tests `00tslang_issue_172.ts` (gc, rc, none).

173 (172's open items, 2026-11-23). Fixed:
- A generic function given to a function type whose parameter is `any`: a typed `const` holding a generic arrow
  (`const f: <p>(a: p) => W<p> = <p>(a: p): W<p> => ({ a })`, 1188, whose type is erased to `(a: any) => W<any>`)
  and `const g: (a: any) => any = id` failed with "can't find type by name: p". A type parameter inferred as `any`
  was left to the instantiation, which never gave it one; it is now `any`.
- A namespace `var` declared in a block of the namespace's code (`for (var i = 0; ...)`, `if (c) { var k = 1; }`,
  1392) and read by a function of the namespace: "can't resolve name: i". The namespace's declarations were
  generated before its code, which declares those variables. When they fail, the code is generated first and they
  are tried again (once: the code's function is not generated twice).
- `class D extends k` where `k` is a module `const` set from a class (`const k = B`): the const is a constant global
  set from that class, so the base is the class itself, known at compile time. A const given a subclass through
  its type (`const k: typeof B = C`) is still the clear error for a class value.

Looked at, still open:
- -mm=own: `typeof a === "string"` of a `const a` holding the `any` a call returned (171, 172).
- -mm=own: the typed `const` holding a generic arrow that stores its parameter (`({ a })`) is refused: "a value
  that borrows 'a' cannot be stored" (the function value may keep a parameter), as the untyped form is.

Tests `00tslang_issue_173.ts` (gc, rc, none).

174 (the quick items of the open-item inventory, 2026-11-23). Every "still open" list above was checked again against
the build: a repro of each item was run. Fixed:
- A closure reading a `const` declared after it whose value reads another `const` declared later (`const f = () =>
  g(); const k = 5; const g = () => k;`, 144): "can't resolve name: g". g was declared before the body (an earlier
  function reads it) with the type of its value, which needs k: what a declaration made before the body reads is
  declared before it too. Under -mm=own it is own's clear error for a closure's capture.
- A namespace `var` declared again (`var p: A.P; var p = {...}`, `var n = 1; var n = 7;` in a namespace, 118):
  "redefinition of symbol named 'C.p'". The second, with a value, is an assignment, which the namespace's code runs
  (isRepeatedVarStatement), as the module's code does for the module's.
- An object literal given a class whose constructor takes arguments (`class Point { constructor(public x: number,
  public y: number) {} }`, `const p: Point = { x: 0, y: 0 }`): the compiler crashed. The literal sets the fields; the
  constructor, which was called with no arguments and failed unseen, is not called for it (one with no parameters
  still is, as before).
- A function pushed into an array of functions (`fs.push(() => i)`, `fs.unshift(...)` of a `(() => number)[]`):
  "can't assign '() => s32' to '() => number'". It takes its types from the array's element, as one assigned to an
  element does.

Checked again and working now (the lists above can be read as closed for these): 49 and 54 (-mm=own), 90, 92, 93
(JSX is a clear error, per the ruling), 97, 100, 101, 105, 106, 109 (`this` arrow before `super()`), 115, 116, 118
(an untyped generic `var`), 119, 120, 123, 131 (`Object.keys` of an `any`), 134 (an arrow given to a call-signature
type), 135, 136, 138/139, 142 (`AsyncGenerator`, with the default library), 166 (-mm=own).

Not quick after all, still open:
- A module `const a = () => b()` in a root that has top-level code, read by a function (109/142): the root's code runs
  in a function of its own, where an arrow `const` stays a local of it and is not the module function its name means.
- `var [a, b] = t` in a block read after it (118): the names of a pattern need their types before the body.
- `push` on an array no variable holds (`Object.keys(o).push("z")`, `mk().push(3)`, 131): it needs a reference to
  the array, and under rc and own the array's count.
- `typeof` of an `unknown` or `any` holding an integer literal is "s32" (168): integer literals are s32 (ruling).
- Not reached in this batch: `for (var x in this)` in a method (102), an interface extending `Tag<...>` beside a
  function `Tag` (134), a callable interface given an arrow (140), `m(f = () => () => this)` (145), a namespace
  function named as the merged class's method (145), a getter returning itself (148), a generator typed as the
  program's own iterator interface with integer yields (153).

Tests `00tslang_issue_174.ts` (gc, rc, none).

175 (the quick items 174 did not reach, 2026-11-23). Fixed:
- A getter that reads itself (`get p(): number { return this.p; }`, 148): "Class member 'p' can't be found". A
  getter with a declared type is known as its class's accessor while its body is discovered, as a generator method
  is (mlirGenFunctionPrototype); addAccessor then registers it as what it is (virtual, abstract).
- An interface of one call signature (`interface Cb extends Function { (x: number): number }`, 140): a function given
  to it was "invalid cast from (p0: number) => number to Cb". With nothing else in it or what it extends, it is that
  function type, as the type literal `{ (x: number): number }` is (callSignatureOnlyType).
- An interface extending a generic interface beside a function of that name (`interface K extends Tag<K, number>`
  with `function Tag<T>(x: T): T`, 134): "interface cannot extend type: (p0: K) => K". The name given type arguments
  as a value is the function; the base is the type of the name.
- `for (var x in this)` in a method (102): "Class member 'length' can't be found". A class instance's keys are its
  fields, its bases' first, known from its type, as an object literal's are; methods are not walked.

Still open:
- `m(f = () => () => this)` (145): "'ts.Load' op using value defined outside the region"; `function g(a = 5, f = ()
  => () => a)` crashes the compiler. An arrow inside an arrow of a parameter's default captures what the function
  has not made yet.
- A namespace function named as an instance method of the class it merges with (`C.foo`, 145): both are the symbol
  `C.foo`.
- A generator typed as the program's own iterator interface (`function* g(): It<number>`) with integer yields
  (153): next()'s result type is given for Generator, Iterator, Iterable and IterableIterator only.

Tests `00tslang_issue_175.ts` (gc, rc, none).

176 (larger items of the open-item inventory, first batch, 2026-11-24). Fixed:
- A `for (let i ...)` body that assigns `i` (128): every closure made in it saw the loop's last value (3 3 3, not
  0 1 2). The body has an `i` of its own each iteration now even when it assigns it, and gives the value back to
  the loop's at the end of the body and at a `continue` to the loop, before the copy is given back
  (GenContext::loopCopyBack), as ECMAScript copies the last iteration's bindings into the next. Not under -mm=own,
  which keeps the one variable there (its capture rule refuses the copy a closure takes and the body writes).
- A `for (let s = "a"; ...)` of a reference type (128): one copy per iteration under gc and none, the reference
  copied as JavaScript copies it (a a a -> a aa aaa). Under rc and own a copy would be one more owner: still the
  loop's one variable.
- A closure in a parameter's default value that reads an earlier parameter (`function g(a = 5, f = () => a)`,
  `f = () => () => a`, 145): the program crashed. Which parameters a closure takes as copies was decided after the
  default values were made, so the closure there took `a` as a cell its function did not expect. It is decided
  before them now, and a default value that assigns a parameter counts.

Looked at, still open:
- -mm=rc: a closure made in a parameter's default value is given back at the end of the default value's region, and
  the parameter then holds what was freed (`g()` above prints 0 under rc; `m(f = () => this)` too). Who owns a
  defaulted parameter's value is the question: the caller does not, and the function releases no parameter.
- `m(f = () => () => this)` in a method (145): "'ts.Load' op using value defined outside the region".
- Names that start with `__` (141): the parser keeps them escaped (`__p` is `___p`), and the compiler is not one
  way about it - a field is declared under the escaped name, `o.__p` looks for the plain one, and an identifier the
  compiler makes from an escaped name is escaped again (`____p`): `class A { __p = 7 }` with `o.__p` fails. The
  compiler's own helpers are named by their escaped names (`___unbox`, `___cast`, ...). One convention (the plain
  name, TypeScript's idText) needs those helper names and the default library's symbols looked at together.
- A non-exported class of one name in two declarations of a namespace (140/147): "redefinition of symbol named
  'N.C.constructor'". Each declaration's non-exported members need names of their own.

Tests `00tslang_issue_176.ts` (gc, none).

177 (names that start with `__`, 141 and 176; user ruling 2026-11-24: the compiler's own names use `.`, not `__`).
Fixed:
- A name that starts with `__` (`class A { __p = 7 }`, `o.__p`, `keyof A`, `for...in`, `{ __p } = o`): the parser keeps
  it escaped (`___p`, TypeScript's escapedText), and the compiler was not one way about it - a field was declared as
  `___p` and read as `__p` ("can't resolve property"), an identifier made again from the escaped name was escaped once
  more (`____p`), and `keyof A` gave `"___p"` ("can't cast from literal type '"__p"' to '"___p"'"). A name is read
  as written now (MLIRHelper::getName undoes the escape), wherever it is declared or used.
- The helpers the compiler makes from source (`function __unbox<T>(a: any): T`, `__cast`, `__any_key`,
  `__bin_op_<op>`, ...) are named with a `.` (`.unbox`, `.cast`, ...): parsePartialStatements renames the `__x`
  names of a helper's source, a name no program can write, so a program's own `function __unbox` is not the helper.
- The default library's `__` names (`__String`, `__is`, `__Iterator`, ...) are read as written too: their symbols are
  `__String.split`, no longer `___String.split`, so a default library built before this does not link with it (its
  archives are made again by this compiler, as CI does).

Tests `00tslang_issue_177.ts` (gc, rc, none; it passes under own too).

178 (items of the open-item inventory: 109, 118, 131, 145, 153).
Fixed:
- A module arrow const calling one declared after it, beside top-level code (`const a = () => b(); const b = () =>
  10; print(a());`, 109): the root's code is a function of its own, where the const was a local alias of an arrow
  named by its place, while the module's functions called it by its name ("'ts.Call' op 'b' does not reference a
  valid function"). Such a const is the module's there too (GenContext::rootFunctionConst), a function named by it.
- A destructuring `var` in a block read after it (`if (c) { var [a, b] = t; } return a + b;`, 118): it was a `let` of
  the block ("can't resolve name: a"). Its names are the function's variables now: the pattern is taken apart into
  `let`s of the block named `.var.a`, ..., and each name is given its value. Not under -mm=own, which refuses a
  variable declared without a value that is given one later.
- push, unshift, splice, pop and shift on an array a call made (`Object.keys(o).push("z")`, `make().push(7)`, 131):
  "Can't get reference of the array, ensure const array is not used". The array is given a place for the call.
- A method of a class and a function of the namespace merged with it, with one name (`class C { foo() }`,
  `namespace C { export function foo() }`, 145): both were the symbol `C.foo` ("redefinition of symbol named
  'C.foo'"). The method is `C..foo` now, and `C.foo` through the class is the namespace's function.
- A generator typed by a program's own iterator interface (`interface It<T> { next(): { value: T; done: boolean } }`,
  `function* g(): It<number> { yield 1; }`, 153): "can't assign '() => {value:s32, ...}'". Its next() returns
  `ReturnType<It<number>["next"]>`, so a yield is given the interface's value type, as for `Generator<T>`.
Still open:
- An exported arrow const beside top-level code that a function calls (`export const f = () => ...; print(f());
  function main() { f(); }`): "'llvm.mlir.addressof' op must reference a global".
- A module const set from a call whose type is known only later (`const a = () => b(); const b = () => 10; const k
  = a();`, no top-level code): a function reading k gets it as `none` ("failed to legalize operation 'ts.Cast'").
- `m(f = () => () => this)`: a closure in a closure of a default value reading `this` ("'ts.Load' op using value
  defined outside the region").

Tests `00tslang_issue_178.ts` (gc, rc, none).

179 (the items 176-178 left open, 2026-11-24). Fixed:
- An exported arrow const beside top-level code that a function calls (`export const f = () => 41; print(f());
  function main() { f(); }`): "'llvm.mlir.addressof' op must reference a global". The root's code is a function of
  its own, where the const was a local, while `main` read the module's global `f`. An exported const there is the
  module's global, as one is without top-level code: a `let` given its value (exported, other modules read it).
- A module const set from a call whose type is known only later (`const a = () => b(); const b = () => 10; const k =
  a();`): a function reading `k` got it as `none` ("failed to legalize operation 'ts.Cast'"). A global whose value
  has no type yet is not declared: the statement is tried again once the call's type is known. One that never has a
  type (`const p = print`) is a clear error: "the type of 'p' can't be resolved from its value".
- A closure in a closure of a parameter's default value (`m(f = () => () => this)`, `p(x = 2, f = () => () => x)` in
  a method): "'ts.Load' op using value defined outside the region". The closures met while a default value is
  evaluated for its type are not registered: neither its arrow nor one in it (their captures there are not what they
  are where the value is made), and the type the arrow had there is not taken as its own (it returned the inner
  closure as a plain function, "losing this reference").
- -mm=rc: a closure, or another value that owns memory, made by a parameter's default value (`g(a = 5, f = () => a)`,
  `m(f = () => this)`, `s = "x" + a`) was given back at the end of the default value: the parameter held what was
  freed (`g()` printed 0). The value is owned by a slot of the frame, empty unless the default is taken, given back
  at the function's exit; the parameter borrows it. (A string default leaked before; it does not now.)
- A non-exported name declared in two blocks of one namespace (`namespace N { class C {} } namespace N { class C {}
  }`, functions, `let`s, enums, types too, 140/147): "redefinition of symbol named 'N.C.constructor'". What a block
  does not export is its own: such a name of a later block is renamed in that block (`.C.<pos>`, a name no program
  can write), its declaration and the names reading it, not a property or member named so; `{ count }` keeps its key.

Looked at, still open:
- -mm=own: an exported `const s = "x" + k` beside top-level code read by `main` across a call of an exported function
  is "a value that borrows 's' is used here after 's' is released or overwritten" (it was before this, too): the
  const is a `let` of the module, which the call may write.

Tests `00tslang_issue_179.ts` (gc, rc, none; own refuses the closures capturing `this` in default values, as it does
`m(f = () => this)`).

180 (-mm=own items of the open-item inventory: 69, 157, 169, 2026-11-24). Fixed:
- A value moved on some ways of an `if ... else if` with no `else`, or of an `if` inside an `else` (`if (i == 0)
  s.a.push(n); else if (i == 1) s.b.push(n);`, 69): "'n' is moved here on some paths only; -mm=own cannot release it
  on the others yet". One move was released on the edges that did not move it (TSLANG 81), several moves only when
  they covered every way. The release goes on the edges from the ways that moved it on none, past any of the moves
  (valueUnmovedEdges takes every move); no move may follow another.
- `typeof a === "string"` of a `const a` holding a call's `any`, then `print(a)` (169): "a value takes a second
  reference". What an `any` holds read out of it (ts.Unbox), and a narrowed name's view of a value (ts.SafeCast), are
  borrows of it while what uses them only reads (user ruling 2026-11-16: an `any` cast back borrows). A `let` read
  through its slot already was.
- An object thrown through a finally block that yields by a function the try calls (`try { yield 1; boom(); }
  finally { yield 2; }`, 157, user ruling 2026-11-22): it stopped at run time ("cannot copy an 'any' that holds an
  object"). It is the compile error a `throw` written there is (#986), at the call: "'boom' throws an object of type
  'Failure', and thrown through a finally block that yields it is not supported". A function that throws an object
  out of itself is remembered, and so are the calls a function makes outside a catch clause of its own; a call in
  such a try is checked through them, again once the module is generated (the callee may be generated after it).
  A function that catches what it throws, a try/catch inside the yielding try, and a string thrown are not refused.

Looked at, still open:
- -mm=own: a call through a method or a function value (`o.m()`, `f()` of a variable) that throws an object through
  a yielding finally still stops at run time: only calls of functions by their name are followed.
- -mm=own: a field read through a union of object shapes returned (`getBox(s: Sq | Ci) { return s.box; }`, 164): "a
  value is merged from several branches". Reading the member out of the union (GetValueFromUnionOp) as a borrow of
  the union in borrowedParam was not enough; the merge of the shapes' reads is refused before it.
- -mm=own: `if (a) h.x = k; if (b) h.y = k;` (152) is refused as a use after a move, rightly: both may run.
- `arguments.length` counts an optional parameter not given (`h(1)` of `h(a, b?)` is 2, JavaScript says 1, 114).

Tests `00own_partial_moves_any_narrowing.ts` (gc, rc, own), diagnostic-location
`generator_call_throwing_object_through_yielding_finally_own.ts`.

181 (-mm=own items 180 left: 164 and methods through a yielding finally, 2026-11-24). Fixed:
- A field read through a union of object shapes, returned (`function getBox(s: Sq | Ci): Box { return s.box; }`,
  164): "a value is merged from several branches". The field is read in a branch for each shape, out of the member the
  union holds (ts.GetValueFromUnionOp, ts.ExtractProperty of it), and that read was not seen as a part of the
  argument: borrowedParam follows both now, so the function returns a borrow of its argument, as with one shape. Its
  callers read the result and may not store it, as for one shape.
- A method, in a try whose finally yields, that throws an object out of itself (`try { yield 1; w.boom(); } finally
  { yield 2; }`): it stopped at run time. Methods are followed as functions are (180): the method a call names
  (ts.ThisSymbolRef, a static's ts.SymbolRef, a virtual one by the class the call goes through).

Looked at, still open:
- -mm=own: a method overridden in a subclass that throws an object, called through the base class in such a try, is
  not seen (the base's method is the one followed); a call of a function value either.
- -mm=own: `for (const s of this.items) yield s` in a generator method (158): the loop keeps the field's array in the
  generator's state across its yields, a borrow of a field stored. Reading the field again at each step
  (`for (let i ...) yield this.items[i]`, which works) would not be JavaScript's for-of when the field is assigned
  during the loop: a ruling is needed.

Tests `00own_union_shape_field_borrow.ts` (gc, rc, own), diagnostic-location
`generator_method_throwing_object_through_yielding_finally_own.ts`.

182 (158, USER RULING 2026-11-24). Fixed:
- -mm=own: `for (const s of this.items) yield s` in a generator method (158): "a value that borrows a field cannot be
  stored, returned or captured". The loop kept the field's array in the generator's state across its yields. Under
  -mm=own, a for-of in a generator over an array field reached from `this` (`this.items`, `this.inner.words`) reads
  the field again at each step: `for (let .i = 0; .i < this.items.length; .i++) { const s = this.items[.i]; ... }`,
  with `break`, `continue`, a destructuring element and a loop variable declared before the loop as before. By the
  ruling, a field assigned during the loop is followed (`this.items = [...]` in the body ends a loop over the new
  array), which JavaScript's for-of would not do; under the other memory models the loop is as before.

Looked at, still open:
- -mm=own: an element that is an object (`for (const o of this.objs) yield o.n`), and the same loop written with an
  index, is "a value that borrows an element cannot be stored": a generator's local is a field of its state, and a
  borrow of an element cannot be one.

Tests `00own_generator_for_of_this_field.ts` (gc, rc, own).

183 (114, 94, 156, 2026-11-24). Fixed:
- `arguments.length` (114) counted the parameters the function declares, not the arguments given: `h(1)` of
  `h(a, b?)` was 2. An optional parameter, or one with a default value, that the caller left out is not counted now
  (mlirGenArguments: the count is worked out from the parameters given, and the `arguments` array is that long), in
  a loop condition and a for-of over `arguments` too. A function with a rest parameter is as before.
- A generic arrow given to a parameter typed by a type parameter (94: `wrap(<T>(x: T) => x)`, `apply(<T>(x: T) =>
  x)`): the arrow is instantiated for `any` and its type inferred from that (resolveGenericParamFromFunctionCall). A
  rest parameter typed by a type parameter (`(...args: A) => T`) is inferred from the function given, `A` an array
  of its parameters' type (tryInferFunction), and a function that takes its parameters one by one can be given as a
  function type whose rest parameter is an array: an adapter reads them out of the array (fewerParametersThunk for a
  named function or an arrow, functionValueAdapter for one read at run time, `{ x: (n: number) => n + 1 }.x`). 94's
  example now compiles and runs (gc, rc).
- throw() of a generator paused in a `yield*` (156): the exception is handed to the delegate's throw(), as in
  JavaScript. The delegate may catch it and go on yielding (the generator gives its next value), or finish (the
  `yield*` is done and the generator goes on), or throw it on (its finally blocks run, then the generator's own try
  statements see it). The loop over a delegate that has `throw` calls next() itself, and the resume check at its
  yield puts what throw() gives in place of the next step (mlirGenYieldStar, mlirGenGeneratorResumeChecks).

Looked at, still open:
- `h(1, undefined)` of `h(a, b?)` gives 1 for `arguments.length` (JavaScript gives 2): an `undefined` given and one
  left out are the same to the function.
- A function whose parameters have different types given to `(...args: A) => T` (`call2((a: number, b: string) =>
  b + a, 1, "x")`): `A` is inferred as `(number | string)[]`, not the tuple `[number, string]`, and the function is
  refused.
- -mm=own: the function given as a rest-array function type when it is read at run time (an object literal's field,
  94's example): the own rule for adapters of such values, with that error. throw() at a `yield*` under -mm=own still
  throws at the yield (the delegate is exchanged into a local for each call there, 156).

Tests `00arguments_length_given.ts` (gc, rc, own), `00generic_arrow_through_generic.ts` (gc, rc),
`00generator_yield_star_throw.ts` (gc, rc).

184 (183's leftovers, 2026-11-24). Fixed:
- A function whose parameters have different types given to `(...args: A) => T` (`call2((a: number, b: string) =>
  b + a, 1, "x")`): `A` is inferred as the tuple of them, `[number, string]`, as tsc infers it (tryInferFunction),
  and the arguments given for `...args: A` are taken as that tuple, not inferred again as an array of their union. A
  tuple of unnamed elements extends an array whose element each element extends (`[number, string] extends
  unknown[]`, extendsType). The adapters read the function's parameters out of a rest tuple as out of a rest array
  (restArrayOf, expandRestArguments). Parameters of one type are an array, as before. Works in every model for a
  function or an arrow given directly.
- -mm=own: throw() of a generator paused in a `yield*` of a generator made there (`yield* inner()`) is handed to the
  delegate's throw() (183 for the other models), for elements that own nothing (a number, a boolean), as the
  generator's return() under -mm=own. The delegate is exchanged into a local of next() for the call and back, and
  what was thrown is exchanged out of the generator's field into a local first: given from the field, the call was
  taken for one that may overwrite the field it borrows.

Looked at, still open:
- -mm=own: throw() at a `yield*` of strings or objects throws at the yield, as before: the step's result would be one
  of two calls' ("a value is merged from several branches"). And a delegate named by a parameter (`yield* it`).
- A delegate whose type has `throw` but not `return` is not handed the exception (the library's Generator has both).

Tests `00generic_arrow_through_generic.ts` (gc, rc; tuple cases added), `00generator_yield_star_throw.ts` (own added).

185 (181's leftover, 2026-11-24). Fixed:
- -mm=own: a method overridden in a subclass that throws an object, called through the base class in a try whose
  finally yields (`w.boom()` of a `W` holding a `V` whose `boom` throws): it stopped at run time ("cannot copy an
  'any' that holds an object"). The method a call names is followed together with its overrides
  (ownOverridesOf: the same method of each class deriving from the class the call goes through, among the functions
  known to throw or to call others), so the call is the error a throw there is.

Looked at, still open:
- -mm=own: throw() at a `yield*` of strings or objects (184). The step's result is a call of the delegate's optional
  `throw` or a default; reading the method unchecked once the `if` has seen it makes the call direct, but own then
  still takes the result for a second reference (the call's freshness through the interface is not established).
  Not done; the yield* throws at the yield there, as before.
- -mm=own: a call of a function value (`f()` of a variable) that throws an object through a yielding finally still
  stops at run time: which function it is is known at run time only.

Decided: -mm=own, a generator's for-of over a field of `this` whose elements are objects (`for (const o of
this.objs) yield o.n`, 182's leftover) stays refused, with its error. USER RULING 2026-11-24: not moving each element
out of the array (which would empty it, unlike JavaScript).

Tests diagnostic-location `generator_override_throwing_object_through_yielding_finally_own.ts`.

186 (2026-11-24). Fixed:
- `arguments.length` of `h(1, undefined)` for `h(a, b?)` was 1 (JavaScript says 2, 114/183): the function can't tell
  an `undefined` given from a parameter left out. A function declaration whose body reads `arguments` and has optional
  parameters (not a method, not generic, not exported, no rest parameter) takes a parameter of its own after the
  others, `.argc` (ARGUMENT_COUNT_NAME, mlirGenParameters), and a call by its name gives it the count of its
  arguments (mlirGenCall); the length is that count, no more than the parameters (the arguments past them are not
  passed). Called as a function value it is left out, and the length is worked out from the parameters as before.
- An optional value given to an `any` (`const x: any = b` of a `b?: string`): one holding nothing was boxed as the
  type's value with no text, which crashed where it was read (`"" + x`); it is boxed as undefined now, and one holding
  a value as that value (as an optional `any` was).
- A `yield*` over a generator whose type has throw() but not return(): the delegate was not held, so a throw() of the
  generator was not handed to it. It is held for its throw(), and no return() is called that the type does not have.
- -mm=own: `yield* it` of a generator a parameter holds hands a throw() to the delegate's throw() as well (184 did a
  delegate made there), for elements that own nothing.

Looked at, still open:
- `arguments` of a method, a generic function or one called as a function value: the length is worked out from the
  parameters, as before (an `undefined` given counts as left out).

Tests `00arguments_length_given.ts` (cases added), `00generator_yield_star_throw_only.ts` and
`00generator_yield_star_throw_param.ts` (gc, rc, own).

187 (2026-11-24). Fixed:
- `arguments.length` of an `undefined` given in a class's methods, static methods and constructors, and in generic
  functions, as 186 did for plain functions: they take the count parameter too, and a call by the method's symbol
  (direct, through the vtable, static, `new`) gives it. An override with or without `arguments`, or with fewer
  parameters, is adapted to the slot it overrides, as one with more optional parameters is.
- -mm=own: throw() at a `yield*` of strings is handed to the delegate's throw() (184/186 did elements that own
  nothing), for a delegate made there, one a parameter holds and one whose type has throw() only. The delegate's
  optional `throw`, once its `if` has tested it, is read out unchecked (InternalFlags::UncheckedNonNull,
  uncheckedOptionalMember): a call of the optional method gave the call's result or a default, "a value is merged
  from several branches". Its result is fresh as an iterator's `next()` is (OwnershipSignaturePass: a generator's own
  `throw` returns what its `next` gives; any other is reached through calls the pass can't all see).
- `f!(x)` of an optional function (`o.f!()` of an `f?: () => T`, `g!()` of a `(() => T) | undefined`) calls it and
  throws a TypeError where it is missing, as JavaScript does; the call was skipped and its result a default. Under
  -mm=own that result, one of the call and a default, could not be kept (`r = g.throw!(e)`: "a value is merged from
  several branches"); it is the call's now (InternalFlags::CalleeNonNull, mlirGenFunctionValuePresent).

Looked at, still open:
- -mm=own: a generator of strings given up before it finishes leaks (three `next()` of a generator of three
  yields, in a loop, grows the heap; the compiler before 187 too). Its throw() is not the cause.
- -mm=own: a `yield*` of a Generator of objects is refused, as before ("a value that borrows a field cannot be
  stored").
- `arguments` read through a function value (`const f = obj.m; f(1, undefined)`): the count is not given, as before.

Tests `00arguments_length_methods.ts`, `00generator_yield_star_throw_strings.ts` and `00callee_non_null.ts` (gc, rc, own).

188 (187's leftover, 2026-11-24). Fixed:
- -mm=own: a generator's `next()` (or `throw()`) called by hand through Generator<T> (`const r = g.next()`,
  `g.next().value.length`) left what it yielded unreleased: a generator of strings grew the heap with every call (the
  compiler before 187 too; a for-of over it did not). Its `{ value, done }` is the caller's - OwnershipSignaturePass
  takes it for fresh (isIteratorNext) - but MLIRGen marked only a plain call's result owned, so nothing gave it back.
  The call through an interface's function-typed field `next` or `throw` returning such a record is marked owned
  now (mlirGenCall), under -mm=own. Checked: a yielded string kept past the record (`last = g.next().value`,
  `keep.push(g.next().value)`) runs and stays flat; a hand-written `next()` returning a field of its object is refused
  at compile time, as before.
- -mm=own: a function read at run time given as another function type (`call({ x: (n: number) => n + 1 }, 4)` for
  `(...args: A) => T`, a callback with fewer parameters through a variable): it was refused ("a function value read at
  run time is not given as another function type yet"). It is held by an adapter in a box as in the other models
  (functionValueAdapter); a closure, whose captures the box would hold a second reference to, and a value the
  adapter would carry out of its caller (`function reg(cb): H { let u: H = cb; return u; }`) are refused by the own
  rules, now at the place the adapter is made (it had no location: `<unknown>:0`).

Looked at, still open:
- -mm=own: a generator that yields objects it got from another iterator is refused in every shape - `yield*`,
  `for (const b of inner()) yield b`, and `next()` by hand (`for (;;) { const r = it.next(); if (r.done) break; yield
  r.value; }`): "a value that borrows a field is used here after it may be released or overwritten". The value is
  the caller's to move, but own takes what the delegate's result holds for a borrow of the generator's state.
- -mm=own: an adapted function value in a module `const` initializer (`const kept = call(wrap(<T>(x: T) => x), 1)`):
  "the initializer of global 'kept' takes a second reference"; in a function it works.

Tests `00own_generator_next_by_hand.ts`, `00function_value_rest_adapter.ts` (gc, rc, own), `00function_value_adapter.ts`
(own added), diagnostic-location `own_function_value_adapter.ts` (now the escaping case).

189 (188's leftover, 2026-11-24). Fixed:
- -mm=own: a generator that yields objects it got from another iterator was refused in every shape - `yield*`,
  `for (const b of inner()) yield b`, `next()` by hand (`for (;;) { const r = it.next(); if (r.done) break; yield
  r.value; }`, `let r = it.next(); while (!r.done) { yield r.value; r = it.next(); }`): "a value that borrows a field
  is used here after it may be released or overwritten". Two causes:
  - the `{ value, done }` the iterator gave is kept in a field of the generator's state (for-of's `.c`, yield*'s
    `.yc`, the source's `r`), and the yield gave the caller a borrow of its `value`, which the next `next()` released.
    Such a read now moves the value out and empties the field right after it (OwnershipSignaturePass::
    takeYieldedResultValues), when the record's one field that owns memory is read only to be yielded, or kept in a
    field that only yields (findYieldedFields), and from each such read on the record is stored anew before its value
    is read again: along the blocks a loop is made of by then, and from a yield on to any place the generator may go
    on at. Other reads of the value before it is yielded borrow it, as before (`if (r.value.s.length > 3) continue`).
  - for-of, and an iterator a generator keeps in a local (`const it = inner(k); ... it.next()`), even of numbers: the
    `next()` through the `Generator` interface may be any generator's `next()`, this one's too, which writes this
    generator's state fields, so own took the call for one that may release the iterator it runs on. Only a
    generator's own methods write its state, and none runs again while one does: a field of the state read off the
    state its method runs on is not dropped by a call (OwnershipInferencePass::readsRunningGeneratorState). Not
    `next`, `return`, `throw`, which the interface lets anyone assign, nor the captured box.
  Checked: values right in gc, rc and own; a caller that leaves the generator early; 1M runs of each shape stay at
  ~50 MB under own. Refused still: a value yielded twice without a new `next()`, read again after it is yielded, or
  kept in a global.

Looked at, still open:
- -mm=own: a for-of element kept in a `let` declared outside the loop (`let kept: Box | null = null; for (const b of
  g) kept = b;`) in a plain function: "'b' borrows a field and cannot be stored, returned or captured"; `push` works.
- -mm=own: a generator of objects has no `throw()` and `return()` (MLIRGenFunctions, by design: what return() keeps
  in an `any` can't be read back out as its owner), so `g.throw!(e)` on one is "a function that is not there is called".

Tests `00own_generator_objects_from_iterator.ts` (gc, rc, own), diagnostic-location
`own_generator_value_yielded_twice.ts`.

190 (189's and #1027's leftovers, 2026-11-25). Fixed:
- -mm=own: a for-of element kept in a `let` declared outside the loop (`let kept: Box | null = null; for (const b of
  g) kept = b;`) was refused ("'b' borrows a field and cannot be stored, returned or captured"); `push` worked. The
  element moves out of the loop's `{ value, done }` record into the `let`, whose old value is given back
  (OwnershipInferencePass::fieldTakeOf takes an assignment to an owning local declared with a value: rc's retain,
  the release of the old value, the store). Refused still: `b` read after the store, an array's element (a
  borrow, as before), a captured `let`.
- -mm=own: a module `const` whose initializer gives a function value read at run time as another function type
  (`const kept = call(wrap(<T>(x: T) => x), 1)`), or values boxed into a rest parameter's array (`const n =
  count(1)` of a `count(...a: any[])`): "the initializer of global takes a second reference"; in a function it
  worked. The adapter's box given only to calls and given back after them is the initializer's temporary
  (moveIntoGlobals); a global initializer's rest array is owned and given back at the end of the region, as in a
  function (ownNewArrayArguments; it was given back by nothing, under rc too). A callee returning its rest array
  stays refused in a global, as `id(new V(1))` does.
- A function value read out of an `any` whose box holds a smaller value (`<((n: number) => number) | null>a` of an
  `a: any = null`) loaded the whole `{func, this, tag}`, 16 bytes past the end of the box: 00any_call under -mm=rc
  with --opt=false faulted on a guard-page heap (an optimized build loaded only the first pointer). UnboxAny copies
  only the bytes the box has into a zeroed slot for a value wider than a pointer.
- -mm=rc: a callee with no body in the module (another module's, a non-generic library function) was taken to
  write nothing, so the argument hold of #1027 missed `clobber(arr[0], h)` from another module setting
  `h.items[0]`: the store freed the box the callee then read (the program spun in a corrupted heap). Such a
  callee may write any place of an object it is given (writtenKinds "*"); one given only numbers, booleans and
  strings writes nothing.

Tests `00own_forof_keep_in_outer_let.ts`, `00own_global_initializer_adapter.ts`, `00any_unbox_wider_than_box.ts`
(gc, rc, own), `import_bodiless_writes.ts` + `export_bodiless_writes.ts` (gc, rc).

191 (open items of 187-190, 2026-11-25). Fixed:
- -mm=own: a generator object's own `next()` called by hand (`const g = gen(k); g.next()`, the object typed by its
  shape, not Generator<T>), its `{ value, done }` dropped or kept in a local: what it yielded was never freed - a
  generator of strings, stepped and then left (or drained), grew the heap with every call (50 MB for 2M calls, rc
  4 MB). 188 marked the call through Generator<T>'s `next` owned; the call through the object's own `next`/`throw`
  field is marked so too (mlirGenCall), which OwnershipSignaturePass already took for fresh (isIteratorNext).
- -mm=own: a module `const` beside top-level code read in a function across a call of a function value another
  such const holds (`export const s = "x" + k; export const g = function () {...}; ... print(s, g())`): "a value that
  borrows 's' is used here after 's' is released or overwritten". The root's code stores such a const once, where
  it is declared, but the store made it a global "written", so `g()` had no known callee and might assign any
  global. Such a global is marked (WRITTEN_ONCE_GLOBAL_ATTR_NAME, GenContext::rootStatementWasConst), and its one
  store, outside a loop, is taken for its initializer (OwnershipSignaturePass::writtenOnceStores): no call writes
  it, and a function value it holds is that function (globalFunction). A `let` such a function writes still ends
  what was read of it.
- A non-exported `const h = function (n) {...}` beside top-level code, called (`h(1)`): "'ts.Call' op 'h' does not
  reference a valid function", in every model. A function expression is the module's function there, as an arrow
  is (TSLANG 178).

Looked at, still open:
- `arguments` read through a function value (`const f = obj.m; f(1, undefined)`): a function value's type has no
  place for the count, so a call through one can't give it; every call through a function value would have to.
- -mm=own: a call of a function value that throws an object through a yielding finally stops at run time: which
  function it is is known at run time only.
Checked: throw() at a `yield*` of strings is handed to the delegate (187); of objects there is no throw() under
-mm=own, by design (189).

Tests `00own_generator_next_discarded.ts`, `00own_module_const_beside_top_level.ts` (gc, rc, own).

191, continued: TypeScript's compiler tests that passed on 11-23 and failed on 11-25 (sweep 3263 -> 3291 -> 3300 pass, no
new failures). Fixed:
- An interface extending an interface of one call signature (`interface I7 extends I6 {}` of `interface I6 { ():
  void }`, 175 made such an interface its function type): "interface cannot extend type: () => void". What extends
  it extends the interface (MLIRGenInterfaces). An interface that declares the call signature of the one it
  extends again is still that function type (callSignatureOnlyType: one signature, however often declared).
- Interfaces of one call signature naming each other (`interface A { (): B }`, `interface B { (): C }`, ...): each
  became a function type when it was done while the others held the interface ("invalid cast from () => A to
  B"). One whose signature names a callable interface, or one not done yet, stays an interface.
- An interface of one call signature in an intersection (`x: F & { foo: number }`): "'Never' can't be used as
  parameter type". A part of an intersection that names it is the interface, which the intersection adds members
  to (keepCallableInterface).
- A field read through a union of shapes, one of whose fields is `unknown` (`s.value !== undefined` of `{ value:
  string } | { value: unknown } | { value: undefined }`): "'ts.Cast' op type ['!ts.any'] can't be stored in". The
  `unknown` read is the union's `unknown` member (mlirGenUnionFieldRead).
- `x.slice(1)` of `[number, string] | [number, string, string]`: "calling an extension function given as a value is
  not supported". An array method of a tuple is not a field to read out of each member (mlirGenUnionFieldRead);
  and a tuple literal given to a union of tuples (`count([1, "a"])`) is the member of as many elements
  (tupleMemberFitting), which 112's tags made a verifier error.

Still open from the sweep: `{ foo?: number }` given to `{ foo: number } | { foo: undefined }` (which member is
known at run time only), `x.slice(1) as readonly string[]` (an array of a union is not an array of strings), two
functions returning each other through interfaces of one call signature ("can't resolve name"), and
contextualOuterTypeParameters, contextualParamTypeVsNestedReturnTypeInference1,
declarationEmitTypeParameterNameInOuterScope, genericClassesRedeclaration, genericFunctionsAndConditionalInference.

Test `00tslang_issue_191.ts` (gc, rc, own, none).

191, continued (sweep 3300 -> 3303 pass, no new failures). Fixed:
- `new A()` in `<A,>(x: A) => new A()` of a module with a class `A`: "'new' on a value of type 'any'". A type
  parameter is no value: the name is the class of this namespace or one around it, unless the parameter is given a
  class (resolveIdentifierInNamespace).
- `const fn1: <T>(x: T) => void = t => {...}` (a module const of a generic function type given an arrow):
  "redefinition of symbol named 'fn1'". Its value is the cast of the arrow named after it, and the global was kept
  beside that function (isGlobalConstLambda sees through the cast).
- A function of one signature declared again in a later block of a `declare namespace` (genericClassesRedeclaration):
  "declared again with another signature". 179's renaming of a block's own names took the second block's
  `IIndexable` for one; a `declare namespace` exports its members without `export` (renameNamespaceBlockLocals).
- A type parameter whose constraint is literals (`F extends Target`, `type Target = keyof Targets<any>`) given a
  literal (`Ops<"left">` for `Ops<F>`) keeps it, as TypeScript does: it was widened to `string`, and `Targets<A>[F]`
  of `Ops<string>` could not be read.

Still open: genericFunctionsAndConditionalInference (a generic method `lr<I, O>(a: Targets<I>[F])` of `Ops<"left">`:
`Targets<I>["left"]` with `I` not given yet is not resolved), contextualParamTypeVsNestedReturnTypeInference1 (type
arguments from the contextual type of a nested call).

Also open: a function given to an interface of one call signature that stays an interface (`interface A { (): B }`,
`interface B { (): A }`, `function mb(): A { return ma; }`): "invalid cast from () => B to A". Such interfaces name each
other, so neither can be a function type; a function given to one would need an object whose call slot holds it.

191, continued: module syntax (USER RULING 2026-11-25, which reverses 11-18's "exports are clear errors"; sweep
3303 -> 3565 pass, no new failures). Fixed:
- `export { a, b as c }` of a file's own declarations: each is exported as `export` written on it would make it
  (lowerDefaultExports adds the modifier), and `c` is another name for `b` (an import alias, registered where the
  statement is generated: mlirGenExportDeclaration). `export { a as default }` makes `a` the default export.
- `export { a, b as c } from "./m"`, `export * from "./m"`, `export { default as x } from "./m"`: m is generated
  as an import of it is (mlirGenImportModule), and the renames are aliases; `export * as ns from "./m"` is ns as
  `import * as ns` makes it. Tslang's modules are one flat program, so what m declares reaches this module's
  importers through it.
- `export = X`: a declaration of the file named so is exported, an expression is kept in an exported constant, and
  the file's `exportEqualsByFile` (and default export) is that name. `import x = require("./m")` generates m and
  makes x another name for it; a module without `export =` is x as `import * as x` makes it. In a `declare module
  "m"` block `export =` compiles to nothing; elsewhere (a namespace) it is still an error, as in TypeScript.
- A value given for a namespace's type (`var m: typeof moduleA = i` of an imported module) crashed the lowering:
  it is "a value of type 'I' can't be given for 'moduleA': a namespace has no run-time value".

Tests `import_export_lists.ts` + `export_lists_r.ts` + `export_lists_m.ts`, `import_export_equals.ts` +
`export_equals_ns.ts` + `export_equals_class.ts` (gc, rc, own, compiled separately); crash-errors export_list,
export_from and export_equals now compile.

191, continued (sweep 3565 -> 3594 pass, no new failures). Fixed:
- A JSON file imported as a module (`import b = require("./b.json")`, `import b from "./b.json"`): it was loaded as
  a shared library ("Can't open: Unknown error (0xC1)"). Its one expression is its `export =` (lowerDefaultExports),
  and a specifier with a source extension is not a library (importLibraryPath).
- A module exporting `var y;` (no type, no value), imported: "failed statement". In declaration mode it is an
  `any`, as where the module is compiled itself.

Tests `import_json_module.ts` + `json_module_data.json` (gc, rc, own; compiled separately and JIT).

191, continued (sweep 3594 -> 3614 pass, no new failures). Fixed:
- A namespace merged with a function (`function f() {} namespace f { export let x = 5; }`): `f.x` was "Can't
  resolve property 'x' of type () => number". A member not otherwise found on a function's symbol is the
  namespace's of that name (mlirGenPropertyAccessExpressionLogic).
- Properties a function is given after it (`foo.bar = 12`, `foo["baz"] = "ok"`, `foo[strConst] = ...`, a const
  arrow or function expression too): TypeScript's expando declarations. The first assignment of each property at
  the file's top level is `namespace foo { export let bar = 12; }` (lowerExpandoAssignments), a later one assigns
  it. They were "access expression is not applicable to () => void". Not done: a symbol key, or a property given
  inside a function body.

Test `00expando_function_properties.ts` (gc, rc, own, none).

191, continued (sweep 3614 -> 3632 pass, no new failures). Fixed:
- Imports of a `declare module "m" { ... }` the program declares - `import x = require("m")`, `import * as x from
  "m"`, a default import, `import { a, b as c } from "m"`: they looked for a file ("can't open file: m.ts"). Each
  name is the module's, as `import x = m.a` makes it (bindAmbientModuleImport), through what the module's
  `export =` names (an `import X = C` of its block followed); an interface or a type alias of it is that type.
  `declare module "./a" { ... }` adds to a file's module (an augmentation) and is not one; `declare module "foo";`
  (no body) gives `any`s - it crashed the compiler.
- `h == null` of a `typeof f | null` (a plain function's pointer): it was constant false; it compares the pointer.

Test `00ambient_module_import.ts` (gc, rc, own, none).

191, continued (sweep 3632 -> 3640 pass, no new failures). Fixed:
- An import alias of a class merged with a namespace (`import A = M.C` of `class C {} namespace C {}`, an ambient
  module's `export = C`): `new A()` was "'M.C' is a namespace, not a class". The name is the class, whose members a
  namespace member is found after.
- Two ambient modules requiring each other (`declare module "sub" { import Main = require("main"); ... }` before
  `declare module "main"`): ambient modules are recorded when their file is read, and an import of one binds by its
  full name (a local named as the module found the local). A file of the module's name wins over the declaration.
- `import * as x from "./m"` of a module whose `export = X`: x is X, as with require.
- A namespace, or a module imported as one, given as a value (`f(foo)` of `import * as foo`): it failed in the
  lowering ("failed to legalize operation 'ts.NamespaceRef'"); it is "the namespace 'foo' is not a value". Making
  module namespaces run-time objects is not done (namespaces have no run-time value, a decision of 11-17).

191, continued: computed keys known only at run time (USER 2026-11-25: ease the constant-key rule where the object
holds run-time keys - a dictionary, a class, an `any`). Fixed:
- A method of an object literal with a key known only at run time (`{ [fieldName]() { ... } }`): the literal is a
  dictionary (it was for properties, 152), and a method is an entry holding the function (a function expression:
  its `this` is not the dictionary). It was "can hold only properties and spreads".
- Such a dictionary given to an `any` (`const box: any = { [k]: 7 }`, `box[k]`): the run-time key read threw "Can't
  cast from any type" - a dictionary's keys adapter reads its entries (`for...in`, `k in`, write), not its class's
  members (makeAnyKeysAdapter). Not under -mm=own, where `any[k]` is refused.
Next: a class member named at run time (USER: one dynamic table per class).

Test `00computed_runtime_keys.ts` (gc, rc, none; the methods part works under own too).

191, continued: a class member named at run time (USER 2026-11-25: "Per-class dynamic table"). Fixed:
- `class Holder { [methodName]() {...}; static [staticName] = 42; [declaredName]: string; }` with names known only
  at run time: "a computed property name must be a constant". Such methods and static valued properties go into one
  dictionary per class (static `.dyn`, filled by a static block); `h[k]` through the class's keys adapter reads it
  after the fixed members. `h.f1` with a constant name does not find them. A field with no value is type-only and is
  dropped. Instance fields with values and accessors named at run time keep the error. Not under -mm=own.
- Found alongside: `const x = pick(1); class C { static y = x; }` - y was null: class statics were initialized before
  the module's globals. A global reserves its initialization order when it is created (ts.ctor_priority), so code
  runs in source order. Globals inside namespaces keep the old order.

Tests `00class_runtime_keys.ts` (gc, rc, none), `00static_init_order.ts` (all).

191, continued (sweep: 3 more pass). Fixed:
- A computed name keyed by a union of literals (`const k = cond ? "f1" : "f2"; { [k]() {...} }`, a class member of
  it): "a computed property name must be a constant" - such a key is known at run time like a string.
- An object pattern with a computed key (`const { [itemId]: item } = items`): it read the binding's name ("Class
  member 'item' can't be found"). A constant key is the field's name (a parameter's pattern type too:
  `({ [`key`]: renamed }) => renamed`); one known at run time reads `items[k]`. A dictionary's entries are read by
  key (`const { a } = dict`).
- `const { a: { x }, b } = o`: the elements after a nested object pattern were not declared ("can't resolve name: b").

Test `00destructure_computed_keys.ts` (gc, rc, none).

192. Bugs found by the tsc tests sweep (crashes and internal errors). Fixed:
- A getter with a declared type whose body fails, beside a setter, then any use of the property: the compiler
  crashed. The getter, known as the accessor before its body is made (175), is taken back when its body fails, and a
  virtual accessor whose getter has no method yet reads the function itself (a null method's slot was read).
- `instanceof` on a union one class of which failed (`x: Base | Derived1` of a Derived1 missing Base's field): the
  union's cast helper instantiated itself until the stack overflowed; the same union cast met again while its helper
  is made is an error.
- `typeof` did not narrow a field: `typeof this._x === "number" ? this._x : this._x.bar` was "Can't resolve property
  'bar' of type number", in an `if` too. A field is narrowed as a name is (getterSetterSubtypeAssignment crashed on it).
- `var before = new C()` above `class C`: "failed to legalize operation 'ts.AddressOf'" - the class's vtable global
  was not lowered yet; its address is taken from its ts.Global.
- A class as a parameter's default (`static bar(zz = this)`, `f(zz = C)`): "failed to legalize operation
  'ts.ClassRef'"; the parameter is the class's value, as `var k = C` is.
- `break` / `continue` with no loop (or switch, for break) around it in its function: "failed to legalize operation
  'ts.Break'"; now TypeScript's error (a second, internal one still follows it).
- `any` to a union of primitives (`const v: number | string = a`, `null as any` given to one): "'ts.Cast' op type
  ['!ts.any'] can't be stored". The member whose tag the box has is read; a number's from any numeric tag, as a
  literal is boxed as an s32. A union two members of which share a tag is not done this way.
- A class's box holding no object (`c ? new P() : null` is a P, given to an `any`) read back as the class: its
  `instanceof` read the vtable of nothing (a trap); it is null.

Tests `00tslang_issue_192.ts` (all models), `accessor_failed_getter_error.ts`, `instanceof_failed_class_error.ts`,
`break_outside_loop_error.ts`, `continue_outside_loop_error.ts`, `00any_to_union.ts` (gc, rc, none).

193. More bugs found by the tsc tests sweep (internal errors and a silent failure). Fixed:
- A static and an instance field of one name (`static p = 3; p = 4;`, also named by a literal or a symbol): `C.p`
  read the instance field off the class ("failed to legalize operation 'ts.ClassRef'"). Through the class itself
  only its statics are looked up.
- A static class as a value (`{ s: Array }`, `export default Array`): "failed to legalize operation 'ts.AddressOf'" -
  a static class (and a @value class) has no `..rtti` global; its value holds none. A static method read through such
  a value was "Expected 0 arguments, but got 1"; it is read as through the class's name.
- A class given for an array (`g(A)` with `g(x: { new (): A }[])`, an interface method's last overload taking one):
  "failed to legalize operation 'ts.ClassRef'"; now an error.
- A type in a module variable (`const K = Foo` of `type Foo = number`): "failed to legalize operation 'ts.TypeRef'";
  now an error. `export default Foo` of an imported type exports no value, as TypeScript emits none.
- Object literals given for `Int<T, U>` and then `Int<U, T>` (`foo("", { method(p1) { return p1.length } },
  { method(p2) { ... } })`): U, given by the first literal's method, did not type the second's parameter, which was
  the type parameter itself, and the call failed with "failed statement" alone. The literals are typed in order, each
  by what those before it give.
- `map` of a `never[] | string[]`: the union was read as its `never[]`, whose `map` took `item: never` ("'Never'
  can't be used as parameter type"). `never`, and an array of it, merged with another type is that type. An untyped
  parameter's errors were at `<unknown>:0`; they are at the parameter.
- `unknown | null` (`a ? b : null` of `b: unknown`), stored as the box alone: its null was a null pointer read as a
  box, and `u === null` compared the box's pointer - false when it held null, and the JIT crashed. Null is boxed, and
  the box is compared as an `unknown` is.
- `any` given to a union with `unknown` in it (`(a ? b ?? null : null) ?? 0` is `0 | unknown`), and an object literal
  given to `Object | string`: "'ts.Cast' op type ... can't be stored". The member that takes anything takes it.
- Two `any`s compared in a module variable's initializer, with the default library: "redefinition of symbol named
  'memcmp'" (the library declares memcmp, not lowered yet there); the library's is called.
- An object literal's method given to an object type's method signature (`const o: { method(x: string): number } =
  { method(a) { return a.length } }`): its parameter took the signature's `this` ("Can't resolve property 'length' of
  type Opaque"). Given to a union of such shapes told apart by a literal field (`{ kind: "a", method(a) {...} }`),
  the literal is typed by the member its literal fields name, and cast to it ("can't be stored"). Not yet: a
  discriminant read from a module `const` (widened to string), a template literal or a missing one. (The module
  `const` and the template literal work since 211; an exported `const` was the one widened.)
- An array literal given to a union with a tuple in it (`f(["a"])` for `["a"] | I`): "can't be stored"; the union's
  array takes it.
- A negated test narrowed neither branch: `if (!Array.isArray(d)) d.v`, `if (!(typeof d === "array"))`, `if (!(d
  instanceof F)) return; d.v` read the union. `!(a === b)` is `a !== b`; `!(x instanceof C)` narrows its then branch by
  what the test's being false gives and its else branch to C.
- A variable a test narrowed, then assigned in the branch, read the value the test saw: `if (typeof x === "number")
  { x = "s" + x; print(x) }` printed the number (wrong result), also after an `if` inside the branch that assigned it.
  After an assignment it is narrowed to what it was given (a member of its union), else it is its declared type,
  read again.
- After `if (test) x = e;` (no else) the variable is what the test's being false leaves it or what the branch gave
  it: `if (!Array.isArray(d)) d = [d];` makes it an array after the `if` (it read the `F | F[]`). Only for a variable
  the branch assigns once, by a statement of its own, in a branch that does not leave early.
- A function returning `never` given for one returning a type parameter that another argument made an `s32[]`
  (`withFew([1, 2, 3], id, fail)`): "can't assign '(p0: string) => never' to '(p0: string) => s32[]'" (it compiled
  before `never` gave way in the merge). It is called through an adapter, which never returns either.
- An ambient destructuring declaration (`declare var { a, ...x } : Foo;`, in a `declare namespace` too): "failed
  statement" alone. Each name is declared as `declare var a: Foo["a"]` is; a rest element declares nothing.
- `any` given to a union with a tuple in it (`string | [string, string]`): "can't be stored"; the box's tag says
  which member it is.
- `[n, ...rest] = robot` of a tuple: the rest was its last element cast to an array ("cast from type '!ts.string' to
  array type", at `<unknown>:0`); it is an array of the elements from there on, or a tuple of them for a tuple target.

Tests `00tslang_issue_193.ts` (all models), `00never_array_union.ts`, `00unknown_null_union.ts` (gc, rc, none),
`00narrowing_after_assign.ts` (all models), `class_not_array_error.ts`, `type_not_value_error.ts`, and the `export default` test (`export-default.cmake`).

194. Bugs found by the third sweep of TypeScript's compiler tests. Fixed:
- An object literal `as const` given to a type (`const c: Candidate = { mode: "a", output: x } as const`): typed by
  `const`, its `mode` was a string, which no member of `{ mode: "a"; ... } | { mode: "b" }` takes ("can't be stored").
  It is typed by the type it is given.
- A field `msg?: undefined` (an optional `undefined` is `undefined`) left out of an object literal: "field "msg" can't
  be found in tuple". It is absent, as an optional field is.
- A union of object shapes (`b ? { valid: true } : { valid: false, msg: "m" }`) given to another union of shapes:
  "Cast from ... is not supported" - `typeof` says "tuple" for each. The tag's layout tells them apart; each member is
  tested for and cast as it is.
- A union in a union (a function returning a `U` and an object literal: `{ valid } | (A | B)`) is its members.
- A function returning a union and then a value of one of its members: "failed statement" alone.
  Not yet: such a function returning a union and then a value of no member (`{ valid: true }` after a `U` of shapes with
  `msg`); reading a field some member has not (`v.msg` of `{ valid } | { valid; msg }`, an error in TypeScript) reads
  the value as the first member that has it. (Both settled in 211: the first works; the second reads the member the
  value holds.)

Tests `00tslang_issue_194.ts` (all models).

195. Bugs found by the third sweep of TypeScript's compiler tests (continued). Fixed:
- A union with an object shape tested for truth (`if (u)` of a `number | { x: number }`, `a[k] && a[k].x`): "Cast from
  ... to boolean is not supported". A shape is true. The empty shape a `never` field is (`string | []`, the `value` of
  `{ value: string } | { value?: never }` after the other member was ruled out) is no value of the type it is cast to.
- A field tested for truth (`if (o.bar)` of a `bar: number | null`) was not narrowed, as `o.bar !== null` is: "can't
  cast from 'null' to 'number'" where it was read as a number in the branch.
- An object shape thrown (`throw { code: 1 }`, an intersection of shapes, an array, null): "Not supported type in throw".
  It is thrown as an `any`, what a catch variable is (literals widened first).
- A function returning a union of shapes and then an object literal a member takes by leaving out a field it may
  (`{ valid: true }` for `{ valid: boolean; msg?: undefined }`), or one no member takes: "failed statement" alone. The
  union of them keeps its members as they are (merged, `{ valid; msg?: undefined }` went into `{ valid; msg: string }`).
- ...and when the function returning the union is declared after its caller: the return type was taken from the other
  `return` alone, and the union cast to it ("failed statement"). A function whose discovery could not make a returned
  value yet is discovered again when it is made.
  (`const { bar } = o` in a branch where `o.bar` is narrowed: TSLANG 204.) Not yet: an `any`
  holding `{ code: 7 }` (an s32 field) can't be cast to `{ code: number }` ("the object's fields are not those of type").

Tests `00tslang_issue_195.ts` (all models).

196. Bugs found by the third sweep of TypeScript's compiler tests (continued). Fixed:
- `m! && m[0]` of a `string[] | null`: `m!` tested did not narrow `m`, as `m` tested does; and `m[0]` of such a union
  (one pointer) gave ElementRef the union ("operand #0 must be ... array"). It is indexed as the array.
- A shape with an optional field (`{ foo?: number }`) given to a union with a member for each case of it (`{ foo:
  number } | { foo?: undefined }`): "'ts.Cast' op type ... can't be stored". Whether the field has a value tells which.
- An object literal given to a union of shapes, told apart by a field it gives as `undefined` (`{ disc?: false }`),
  leaves out, gives a template of literals for (`` `${"s"}` ``), gives a const for (`{ kind, ... }` with `const kind =
  "a"`: a shorthand field was not typed by the member), or gives where another member has `never` (`href?: never`):
  no member was found ("can't be stored"), and the literal's functions had untyped parameters.
- A union of shapes with the same fields tested for truth (`if (u)`, the test of `u?.kind`): the members merge into
  one type, and the cast failed with nothing said ("failed statement"). An object is true; a cast from such a union to
  a type it can't be says so.
  Not yet: under -mm=own, an object literal leaving out an optional field (`const o: { disc?: false; cb } = { cb }`)
  is "a value that borrows a field cannot be stored".

Tests `00tslang_issue_196.ts` (all models).

197. Bugs found by the third sweep of TypeScript's compiler tests (continued): a class reading a local it can't capture
gave invalid IR ("'ts.Load' op using value defined outside the region"). Fixed:
- A class declared in a block of the module's code (`for (let i...) { class C { f = i; } }`, `{ let x; class C { m() {
  return x; } } }`), or a class expression of the module's code: its members were discovered by the pass over the
  module's declarations, which does not know the block's locals, and not again. They are discovered where they are
  made, as a function of such a block is (TSLANG 105); one reading a local is the error a class in a function gets
  ("a class can't capture yet"), and one reading none works as before.
- A static block of a class declared in a function (`static { H.x = v; }`), and the table its run-time named members are
  put in (`[name]() {}` with `const name = ...` in the function), read the function's locals: a static block runs at
  the start of the program. It is an error.

- `strtod` declared (as the default library declares it) beside a string converted to a number: where the declaration
  was not lowered yet, the conversion's own `strtod` was a second symbol of that name ("redefinition of symbol named
  'strtod'", or a crash). It is called by name, as `memcmp` is (TSLANG 193).

Tests `00tslang_issue_197.ts` (all models), `class_reads_block_local_error.ts`, `class_static_block_local_error.ts`.

198. Bugs found by the third sweep of TypeScript's compiler tests (continued). Fixed:
- A rest element in an object destructuring assignment (`({ a: x, ...r } = o)`): "rest destructuring in an object
  destructuring assignment is not supported". It is an object of the fields the pattern does not name, as in a
  declaration.
- The rest of a `T | undefined` in a declaration (`const { ...rest } = u`): "field ... can't be found in tuple". It is
  T's fields, with their defaults where it has no value, as `{ ...u }` spreads it.
- ...and the same in an assignment (`({ ...r } = u)`, and `var { ...rest } = u` of a `var rest` declared before it,
  which is one): "field "value" can't be found".
  Not yet: the rest of a `T | null` ("the rest of a value of type 'null | T' can't be taken").

Tests `00tslang_issue_198.ts` (all models).

199. An enum indexed by a key known at run time (`Size[mySize]`, `Size[k]` of a string): "Enum reverse lookup by index
is not supported". A number gives the name of the member with that value (the reverse mapping of a numeric enum, ""
when no member has it; a constant number too, which was read as a member's name), a string the value of the member with
that name (0 when none has it); each member is compared in turn.

Tests `00tslang_issue_199.ts` (all models).

200. A destructuring `var` of the module declaring some names again and some for the first time (`var { n: nb, p: pb }
= r; var { n: nc, p: pb } = s;`): "redefinition of symbol named 'pb'" - only a pattern all of whose names were declared
before was an assignment. The names declared before are taken as hidden variables of the pattern (`.var.pb.<pos>`)
and assigned to after it, in the module's code or a constructor of their own.

Tests `00tslang_issue_200.ts` (all models).

201. Function values (sweep, continued). Fixed:
- A function called where it is made (`(function () { return f; })()`) whose value is given a type (`const g: (v: any) =>
  string = ...`): the type was taken as the called function's own ("invalid cast from (p0: string) => string to
  string"). The callee is made without it.
- A function value given as one whose parameter is `any` (`(v: string) => string` as `(v: any) => string`, as
  TypeScript allows): "argument #1 does not match". An adapter converts the argument (checked at run time). A function
  returning an integer given as one returning a number (`() => 0` as `(v: any) => number`): "it takes 0 argument(s)".
  The adapter widens the result.
  A result of type `any` given as another type (`(v: any) => any` as `(v: any) => string`) is cast by the adapter too
  (TSLANG 204).

Tests `00tslang_issue_201.ts` (all models).

202. A named function expression held by a variable of its name in a namespace (`namespace N { export let f = function
f() {} }`, or an expando `example.isFoo = function isFoo() {}`, which is one): "redefinition of symbol named 'N.f'" - the
variable is that symbol. The function's symbol is one of its own (`N..fn.f`); its name, by which it calls itself, stays.

Tests `00tslang_issue_202.ts` (all models).

203. A function of the program named as a C function the default library declares (`function log(n: number) {}` beside
its `declare function log(x: number): number`): "redefinition of symbol named 'log'". It is a symbol of its own
(`.u.log`), and the program's `log` is it; the library's own code keeps the C function.

Tests: `defaultlib/defaultlib_collector.ts` (with the default library, jit and exe).

204. Bugs found by the tsc compiler-tests sweep (continued). Fixed:
- A generic function destructuring a parameter of a generic type (`function f<TX>({ f }: TG<TX>)`, a type alias): "Can't
  resolve property 'f' of type TG<TX>"; with a rest (`{ f, ...rest }: IG<TX>`), "variable 'rest' is referencing generic
  type". The pattern was taken apart in the discovery of the function's parameters, before TX is known; that discovery
  reads no body, nor the names a pattern binds.
- `const { bar } = o` in a branch where `o.bar` is narrowed (`if (o.bar)`): `bar` had the field's declared type.
- A function returning `any` given as one returning another type: the adapter casts the result (checked at run time).
- A destructuring pattern as a catch clause's variable (`catch ({ x })`): the clause failed without an error. What was
  caught is held by a hidden variable and destructured at the start of the clause. An array pattern of it, a field of it
  under -mm=own and its rest (`catch ({ ...rest })`, which was an empty object) are errors: it is an `any`.
- A call tested for a narrowing (`o() === undefined && x`, `typeof f() === "string"`) was made again by the narrowing -
  twice, three times in an `if`. Only a name, `this`, and a property or an element of one are narrowed.
- A `?:` with a branch of no value (`c ? f() : 0` with `f(): void`, `c ? super() : 0`): "'ts.If' op expects a non-empty
  block". The branch is undefined, as in JavaScript.
- Clear errors where these failed with "failed statement" alone: a generic class expression (`const m = class C<X> {}`,
  not a value); too few type arguments (`new D<number>()` of `D<T, U>`); an optional or rest index signature parameter;
  an enum member with a numeric or computed name; a destructuring declaration with no initializer (`var [a], { b };`);
  an element past a tuple's end, in a pattern (`{ 3: d }`) or an assignment (`[a, b, c] = t` of a pair).
- Clear errors where these were internal errors or named hidden members: the class itself given as an interface
  (`const c: Constructor<A> = A`, "failed to legalize operation 'ts.ClassRef'"; `typeof A` and a constructor type
  `new (...args: any[]) => A` work); the class itself converted to a string (`f += ""`, "Class member 'get_toStringTag'
  can't be found"); a method with a numeric name (`static 1() {}`, read as `C[1]()`: "can't resolve property/field/base '.index'"); an index signature keyed by `boolean` or by literals (`[u: "foo" | "bar"]`).

The local default library at I:\tslang was rebuilt (11-26): the one from 11-22 predates the compiler naming change, and
an exe using `Math` failed to link ("DataView..size already defined").

Tests `00tslang_issue_204.ts` (all models), `00tslang_issue_204_catch.ts` (gc, rc, none), and the error tests
`too_few_type_args_error.ts`, `destructuring_no_initializer_error.ts`, `tuple_index_pattern_error.ts`,
`index_signature_optional_error.ts`, `enum_numeric_name_error.ts`, `generic_class_expression_error.ts`,
`catch_rest_any_error.ts`, `class_itself_interface_error.ts`, `class_itself_string_error.ts`, `index_signature_key_error.ts`.

205. Bugs found by the tsc compiler-tests sweep (continued). Fixed:
- A generic function whose result is a generic function type using its own type parameter (`function f<Out>(): <R>(cb:
  (d: Out) => R) => R`, declared or not): "failed statement" with nothing said. While Out is unknown the inner type
  parameters stay named generics (zipped with no type arguments, one with no default or constraint failed).
- A generic arrow returned (`return <R>(cb: () => R) => cb()`, with a declared return type or none): "can't find type
  by name: R". Instantiated with `any`, a type parameter only inside a parameter's function type (`cb: () => R`, given
  `any`) is `any`.
- An arrow returned from an arrow instantiated for a function type (`(cb) => (x: Out) => R`): the inner arrow took the
  outer one's whole type as its result ("can't assign ... return type '(p0: number) => number' is not 'number'", or a
  wrong result at run time). The type a function is given to no longer reaches the functions written in its body.
- An untyped parameter of an arrow given to an interface's field of a function type (`const h: H = { consume: (d) =>
  ... }` for `consume: (d: number) => void`): typed Opaque - the hidden `this` the field's type is stored with - so
  "can't assign '(p1: Opaque) => void' ... it takes 1 argument(s), 2 expected".
- A generic call given an object literal whose members give the type parameter to each other (`{ produce: () => 42,
  consume: (data) => ... }` for `Options<Data>`, fields or methods): "failed statement". The fields' types are read
  without the hidden `this`, a declared result is the member's (`(): number => 1.5`), and members whose parameters
  wait for the type parameter are left out until the others have given it.
- `X & any` was `never` (`{ x: string } & T` with T `any`: "'Never' can't be used as parameter type"); it is `any`, as in
  TypeScript (`never & any` stays `never`).
- An arrow member of an object literal that says `this` or `super` was a method of the literal: `{ a: () => this.m() }`
  read the literal, `{ a: () => super.m() }` was "can't resolve name: .this". It reads the code around it, as in
  JavaScript (user ruling 11-26): a closure, whatever the literal is made for. Under -mm=own, one that escapes (the
  literal returned) is refused, as any closure capturing `this` that escapes is. A literal's method keeps the literal as
  its `this`.
- A closure given for an interface's field of a function type through an adapter (`() => s32` for `f: () => number`,
  `return { f: () => this.k }` with `k = 7`): the adapter lost the closure's captures, and the call read a wrong value.
  It is adapted to the field's type without its `this`, and stays a closure.
- An untyped parameter of an arrow in a literal made in a class field's initializer (`lit = { d: (cb) => cb(this) }`):
  typed as the closure's captures, read from the type the literal was first made with ("not a function to call: a value
  of type 'Reference<{this:C1}>'").

Not yet:
- A literal given to a generic call whose field returns an integer where the type parameter is inferred as a number
  from elsewhere (`takeG({ produce: () => 44 }, 1.5)`): "can't assign '() => s32' to '() => number'", as is a literal
  variable given to the interface (`const o2: Options<number> = t`).
- `{ x: string } & any` is `never` ("'Never' can't be used as parameter type"); TypeScript has `any`. Making it `any`
  sends `conditionalTypeDoesntSpinForever.ts` (tsc) into a loop, so it stays for now. (Fixed in 211, no loop.)
- The debug JIT crashes in LLVM translation on any program (local debug build, 11-26); release is fine.

Tests `00tslang_issue_205.ts` and `00tslang_issue_205_this.ts` (gc, rc, none: under -mm=own the returned closure over a parameter is refused by design).

206. The array `splice` gives back, under -mm=rc, was freed while it was being returned (release run of 2026-11-26,
array_class_splice.ts crashing in the debug JIT on Ubuntu 22.04/24.04 with no output). Fixed:
- `const removed = this.data.splice(start, count); ...; return removed;` (the default library's Array<T>.splice): the
  result is marked owned (mlirGenArraySplice), so `removed` takes its reference over and does not retain, but the
  lowering made the header with `ArrayLayout::make`, whose block is born with no reference. The return retained it to
  1, the local's release took it back to 0 and freed it, and the caller read a freed array: the next allocations wrote
  over it and mimalloc's free list (a crash in `mi_malloc` at a later `splice`). It has been so since allocations are
  born with no reference; on Windows and in a warm JIT cache the freed block happened to survive. Under rc the header
  is now born with the reference, as an owned `new C()`'s is.

Tests `00rc_splice_result_owned.ts` (gc, rc, none; under own the test's re-insertion of the removed elements is
refused).

207. Leftovers of 195, 196, 198 and 205. Fixed:
- A literal given to a generic call whose function member returns an integer, where the type parameter is a number
  from another argument (`takeG({ produce: () => 44 }, 1.5)` for `takeG<D>(o: Options<D>, d: D)`): "can't assign
  '() => s32' to '() => number'", and with the arguments the other way round `D` was `1.5 | s32`. The literal is typed
  by the parameter once the call's other arguments give D (inferObjectLiteralArgumentReceivers, as a literal whose
  members' parameters wait for D already was); its members are read one by one for the inference, without making them -
  a member made then kept the `() => s32` it was made with.
- An `any` holding an object literal written with integers (`{ code: 7 }`, an s32 field) cast to an object type of
  numbers (`<{ code: number }>a`): it threw "the object's fields are not those of type". A layout with s32 in place of
  some of the number fields (three at most) is read as itself and converted (castAnyToTupleOfNumbers). Not under
  -mm=own, where the value merged from those reads is a second reference: there the cast still throws.
- Under -mm=own, an object literal leaving out an optional field of the tuple it is given to (`const o: { disc?: false;
  cb } = { cb }`): "a value that borrows a field cannot be stored". The literal is made with the missing optional
  fields, undefined, and so is the receiver's tuple as it is, where it was converted by reading its fields out of its
  slot.
- The rest of a `T | null` (`const { a, ...rest } = u`, and `({ a, ...rest } = u)`): "the rest of a value of type
  'null | T' can't be taken". It is T's, read as the pattern's other names read T's fields.

Not yet (both fixed in 208):
- Under -mm=own, a tuple literal given to a callee that keeps it (`k.keepOne(["a" + i, i])`): "this function does not
  own it". The same tuple in a local first (`const t: [string, number] = ["a" + i, i]; k.keepOne(t)`) works.
- A literal held by a variable first (`const t = { produce: () => 45 }; const o: Options<number> = t`): `t`'s member is
  `() => s32`, which is not `() => number` (integer literals are s32, user ruling).

Tests `00tslang_issue_207.ts` (all models) and `00tslang_issue_207_any.ts` (gc, rc, none).

208. The "not yet" of 207. Fixed:
- A literal held by a variable first, whose function member returns an integer, given as an interface whose member
  returns a number (`const t = { produce: () => 45 }; const o: Options<number> = t`, a `let` as well, and
  `readOptions(t)` for a parameter `o: Options<number>`): "field "produce" not matching type: '() => number' and
  '() => s32'". Integer literals stay s32 (user ruling); the function is adapted, as it already was given as a
  `() => number` value (`let f: () => number = one`). The interface's slot calls the field through a thunk, as it calls
  a closure field, and the thunk converts the result (isResultWideningFieldFunction): the object is given as it is,
  with no copy, and no box for -mm=own to refuse. Where the object is copied for another field, the copy keeps the
  function as it is for the same thunk; a field of a constant object read as a function value is its function, named
  (fewerParametersThunk).
- Under -mm=own, a tuple or object literal given to a callee that keeps it (`k.keepOne(["a" + i, i])`,
  `k.keepObj({ s: "b" + i, n: i })`): "this function does not own it". The literal is built in a slot nothing owns; as
  an argument it now goes into a local of its own, as `const t = [...]; k.keepOne(t)` puts it (ownedLiteralArgument):
  the local gives it to a callee that keeps it, and gives it back where the callee only reads it. 2M calls of either
  stay at about 5 MB. Only a literal of new values or string constants (one holding what it borrows, or a closure, is a
  borrowing record, as it was), and not inside `&&`/`?:`, whose region does not reach the scope's end.
- Under -mm=own, the same directly in a `try` body, and a declared local there (`try { const t = ["u" + n, 1];
  k.keepOne(t); }`): "an object literal that holds a value it owns is moved only into a local". The local is made in
  front of the `try` and given its value by a store right before its retain; that retain is now the literal's, as an
  initializer's is (recordRetainOf).

Not yet:
- Under -mm=own, a value moved in a `try` body that other statements of the body can throw before
  (`try { f(); const t = ["b" + i, i]; k.keepOne(t); g(); }`): "moved here on some paths only". Fixed in 210.
- Under -mm=own, a callee that keeps a parameter and throws before keeping it leaked the parameter: fixed in 209.

Tests `00tslang_issue_208.ts` (all models).

209. Under -mm=own, a method that keeps its parameter on every way it returns by (`this.items.push(c)`) and throws
before keeping it leaked the parameter: the caller gave it up, and the exception took the method out holding it. 1M
throws of `k.keep(new C(i))` grew to about 60 MB, for a `throw` of the method's own (`if (c.n % 3 == 0) throw "no";`)
and for a call that throws (`check(c.n);`, `checkC(c);`). Fixed:
- A `throw` the move cannot have run before releases the parameter first (decideParam), as a parameter the body
  releases on the paths that do not move it already was (releaseKeptParamsAtExits) - not where the thrown value is made
  of it.
- A call that may throw before the move gives it back on its unwind (UnwindReleasePass): the inference marks the
  parameter's slot and the op that moves it (OWN_KEPT_PARAM_ATTR_NAME, OWN_KEPT_PARAM_TAKER_ATTR_NAME), and the call's
  cleanup pad releases the slot where no move may have run before the call. Not for the call that is the move, nor
  where a handler of the same function the call unwinds to uses the parameter again (`try { check(c.n); } catch {}
  this.items.push(c);`, which keeps it).
Each probe now stays at about 5 MB.

Tests `00tslang_issue_209.ts` (all models).

210. Under -mm=own, a value moved into a keeping call in a `try` body with statements that may throw before and after
it (`try { mayFail(i); const t = ["b" + i, i]; k.keepOne(t); mayFail(i + 1); }`, and a literal given straight to the
call): "moved here on some paths only". The `try`'s cleanup releases the local, and is reached from before the move
and after it. The local's slot is made in front of the `try` and starts as null, so the ways before its store find
nothing; the slot is now emptied right before the keeping call, the value already read (clearsOnUnwind), so the ways
past the call find nothing either. A call that keeps its argument has taken it whether it returns or throws (its own
unwind gives it back, 209); the empty slot was refused for a call taker before. 3M turns throwing before, between and
after the moves stay at about 6 MB.

Tests `00tslang_issue_210.ts` (all models).

211. Leftovers of 193, 194 and 205, and old "still open" notes a triage re-ran (2026-11-27; repros and the full table
in the triage report). Fixed:
- 193: a discriminant read from an exported module `const` (`export const KA = "a"; run({ kind: KA, method(a) {...} })`
  for a union told apart by `kind`): "can't be stored". An exported `const` is stored as a `let` (user ruling 09-13),
  and its type was widened to `string` with it; never assigned, it keeps its string literal type, as in TypeScript.
- 194: `v.msg` of `{ valid } | { valid; msg }` read the value as the first member that has `msg`, whichever member it
  was - past the end of a smaller one. It is read from the member the value holds, and is the type's empty value for
  one that has not the field (TypeScript has `undefined`, and makes the read an error; tslang reads it where it is
  narrowed in ways it does not follow yet - `switch (true)` clauses, a truthy discriminant - which tsc-clean tests do, so
  making it an error failed three of them). `"msg" in v`, which tells the members apart, works for a union of object
  shapes (it was "only a union of interfaces is"), narrowing as for interfaces. A function returning a union and then
  a value of no member works already.
- 205: `{ x: string } & any` is `any` (`never & any` stays `never`, `X & unknown` is X). The tsc test that looped with
  the first attempt (`conditionalTypeDoesntSpinForever.ts`) stops at once on an unrelated error.
- Under -mm=own, a variable holding a string converted to another object type (`const u = { n: 1, name: "x" + i };
  const p: { n: number; name: string } = u`, `n` an s32): "a value that borrows a field cannot be stored". The new
  object takes a copy of each string read out of the source (strings are values), is built in a slot as an own literal
  of new values is, and goes into a local of its own, so it can be stored, kept, or only read. Only when every other
  field holding a block is new or constant: one with an object, a nested object or an optional string read out of the
  source stays a borrow, refused where it is stored as before. 2M conversions stay at about 6 MB.
- 137: under -mm=own, `for (const o of [new A(), new B()])`: "an object literal that holds a value it owns is moved
  only into a local". The literal goes into a local of its own, as `const items = [...]; for (const o of items)`.
- 152, -mm=own: an element stored past the end through a local over a parameter's array (`let src = order; src[n] = v`)
  or through a handle's payload (`s.value[i] = v`) did not grow the array (deliberately, as -mm=own refuses the growth
  there) and was written past the end of its block: a corrupted heap, crashing later. It is a RangeError now (an assert
  without the default library); a store inside the array is unchanged. Not JavaScript's behaviour, which grows the
  array: growing through such a local needs -mm=own to take it for a borrower of the parameter that changes it in place,
  as it does for a field's (`const model = this.model; model.push(0)`, TSLANG 62) - a push through it is refused too.

Not yet:
- Under -mm=own, `<{ code: number }>a` of an `any` holding `{ code: 7 }` (an s32 field) still throws. Each layout is a
  cast of the `any` of its own, and -mm=own takes a named `any` cast to a tuple more than once for a second reference
  of its box: `const c = <T>a; const d = <T>a` is refused as well, before this.
- `{ x: string } & T` of a generic function is `never` ("'Never' can't be used as parameter type"), even given T.
  Leaving it to be resolved once T is given made inference fail elsewhere (`SetupImageRefs<K> & {...}`, tsc
  objectRestBindingContextualInference and others: "can't instantiate function"), so it was taken out again.
- 131 `Record<string, number>` is a record of no fields (a warning), and a literal given to it loses its own. Made a
  dictionary, as `{ [key: string]: number }` is, the tsc tests using `T extends Record<string, X>` as a constraint
  failed (an object literal does not meet a dictionary), so it was taken out again: it needs the constraint check to
  take a literal for a dictionary.
- 140 overloads in a declared class: "redefinition of symbol named 'DC.constructor'" for constructors; an error for
  every declared method overload failed tsc-clean tests whose method overloads compiled, so it was taken out.
- Still failing from the triage, not taken here: 132 a generic class given for a constructor type (`fruitFactory1(Banana)`
  of `Banana<T extends string>`: "Can't find class Banana", T to be instantiated from its constraint); 152 a record
  spread into a dictionary under -mm=own; 117 a primitive's method as a value (`let f = n.toFixed`); 131 `Object.keys`
  of `object`, `Object.assign({a: 1}, {b: 2})`, `Object.defineProperty`, `Object.create`; 191 `x.slice(1) as readonly
  string[]` and callable interfaces naming each other; 139 a constructor type of an interface; 140 `await` in
  `finally`; 108 `class X extends k` of a `let`; the built-in `Symbol` as a value; recursive conditional types; the
  tsc tests genericFunctionsAndConditionalInference and contextualParamTypeVsNestedReturnTypeInference1; and under
  -mm=own 56 (a loop shape), 128 (a `for` body assigning its `let`), 144 (a closure capturing a later const holding a
  closure), 164 (an interface narrowed to a class), 153 (an object sent by `next()`), 159 (`yield* this.range()`).
  Design questions without a ruling: overloads of a declared function (136) and of a declared class's
  constructor (140).

Tests `00tslang_issue_211.ts` (all models), `own_grow_past_end_check.ts` (-mm=own).

212. Leftovers of 211's triage (2026-11-27). Fixed:
- 132: a generic class given for a constructor type (`fruitFactory1(Banana)` of `new (...args: any[]) => TFruit`,
  `class Banana<T extends string>`): "Can't find class Banana". Its type arguments come from the constructor type's
  parameters, as `new Banana(...)`'s come from its arguments (171), and a rest among them gives its element type to each
  constructor parameter the others do not reach, the constructor's own rest included: `any` makes T `any` (TypeScript's
  `Banana<any>`), `...args: "foo"[]` makes it a string. The tsc test contextualSignatureInstantiation4 compiles.
- 164: under -mm=own an interface narrowed to a class by `instanceof` (`if (v instanceof A) return v.a` of `v: IA` or
  `IA | IB`): "'v' is a parameter, kept here". The narrowed value is the object the interface holds, a borrow as the
  interface is; the cast also had a way for what is not an A (a new A of the interface's fields), never taken there,
  which made the result a reference of its own. Not for a generic class, which `instanceof` answers for every G<T>.
  00union_interfaces_instanceof.ts runs under -mm=own too.
- `v instanceof G` of a generic class narrowed to the generic itself, no type a value has: an interface `v` crashed the
  compiler, a union "Can't resolve property". It narrows a union to its members that are a G<T> (or derive from one),
  else not at all.
- 152: under -mm=own a record's strings spread into a dictionary (`const d: { [k: string]: string } = { ...rec }`) or
  given to `Object.assign(d, rec)`: "a value that borrows a field cannot be stored". Each string read out of the record
  is copied (strings are values). A literal given to Object.assign goes into a local of its own (ownedLiteralArgument),
  which gives back what it holds: left in a slot nothing owned, it leaked once its strings were copied.
- 144: under -mm=own a `const` a function before it reads, holding a closure or an object (`const f = () => g(); const g
  = () => 10;`): "'g' is captured by a closure, but -mm=own cannot tell who owns its value here". Made one of the
  function's variables (declared with no value, TSLANG 172/174), it is marked an owned local before anything captures
  it, and its declaration gives it the value it owns. One whose value is a borrow (`const q = p`, `const g = this.c`)
  stays refused, as before ("kept here", "borrows 'this.c'").
- 191: an array of a union said to be an array of members of it (`x.slice(1) as readonly string[]` of `[number,
  string] | [number, string, string]`, tsc sliceResultCast): "cast from array type ... is not supported" with no
  location, at lowering. It is a new array, each element read as the member it is said to be (castArrayElementwise),
  for an assertion (`as`, `<T>`) of a union with a tag only: TypeScript refuses `const s: string[] = m`, which stays
  as it was, and a union with no tag (classes) is cast as it is, the same array.
- 139: a class given to an interface of a construct signature only (`interface PtCtor { new (x: number): Pt }`,
  `make(P)`; `const c: Constructor<A> = A`): "the class 'P' itself can't be given as interface" (TSLANG 204's clear
  error for an internal one). It is made over an object of one field, `new`, a function calling the class's
  constructor and returning its object as the signature's result. An interface with other members as well is still
  refused (class_itself_interface_error.ts). Not under -mm=own, which refuses a construct signature's result, through a
  type literal (`{ new (x: number): Pt }`) as well: "a value takes a second reference".
- A leak under -mm=rc of the result of every call through a function value (a callback, an object literal's method,
  an interface method an object literal's function field backs), in every program linking the default library since
  DefaultLib #108: its `Temporal.Now` functions lacked `@owned`, the only heap results there the compiler could not
  classify, and one is enough to turn off the rule that such a call hands its result over (DefaultLib #122). A call of
  a function an object literal holds (`o.make(i)`, the literal a constant of the functions' names) names its callee
  now, so its result is classified whatever else the program links, under -mm=own too (it leaked there).

Not yet:
- Under -mm=own an interface method an object literal's function field backs (`const m: Mk = { make: (x) => new
  P(x) }; m.make(i)`) leaks its result: the thunk calling the field cannot know whose result it is, and -mm=own never
  takes a call through a value for one handing its result over.
- Under -mm=rc `new c(i)` of a type literal's construct signature (`c: { new (x: number): Pt }`) given a class crashes
  at run time (before this batch too).
- Under -mm=own a tuple literal holding a new string given to a union of tuples (`const x: [number, string] | [number,
  string, string] = [i, "a" + i, "b"]`): "a value that borrows a field cannot be stored"; `x.slice(1)` of such a
  parameter: "an element ... would have two owners".
- Under -mm=own `make(L)` of `class L<T> { constructor(...items: T[]) }` for `new (...args: number[]) => X`: "'items'
  is a parameter, kept here" (gc runs it); and a `let` captured by a closure and assigned after (`let g = () => 10;
  const f = () => g(); g = () => 11;`): "takes a second reference".
- The `any` dictionary of a spread record (`const d: { [k: string]: any } = { ...rec }`) leaks under -mm=own (before
  this batch too).
- Still from the triage: 117, 131 (Object.keys of `object`, Object.defineProperty, Object.create), 128, 56, 140 `await`
  in `finally`, 108, 191's callable interfaces naming each other, recursive conditional types, `Symbol` as a value, own
  153 and 159; design questions without a ruling: overloads of a declared function (136) and class constructor (140).

Tests `00tslang_issue_212.ts`, `00tslang_issue_212_dictionary.ts` (all models), `00tslang_issue_212_ctor_interface.ts`
(gc, rc, none); `00union_interfaces_instanceof.ts` under -mm=own; `class_itself_interface_error.ts` changed.

213. Leftovers of 212 (2026-11-27). Fixed:
- A class given to a type literal's construct signature of another result type (`make(P)` for `c: { new (x: number):
  Pt }`, Pt an interface): `new c(i)` printed garbage under every memory model, or crashed. The function made for the
  class returned the class, and the field holding it promised the interface: the caller read a class reference as an
  interface. It returns the signature's result type now, the object cast to it. Under -mm=own `new c(i)` was "a value
  takes a second reference" (here and through an interface of a construct signature, 139): every function such a field
  holds is one the compiler makes, which returns the object it makes, so the call hands its result over. An object's
  vtable holds a method as a reference to its function (`ts.Cast` of the symbol), which the owned-result pass did not
  read as that function: the slot was unknown, and a call through it (an interface method an object literal's function
  backs, `m.make(i)` of `const m: Mk = { make: (x) => new P(x) }`) never gave its result back under -mm=own (212).
- `v === undefined` (and `==`, `!==`, `!=`) of a `v: any` boxed `undefined` on the heap for every comparison, a box
  nothing gave back under -mm=rc and -mm=own (32 bytes a comparison; a dictionary of `any` compares each value it is
  given, so `d["a"] = i` leaked one each store - the "any dictionary of a spread record" leak of 212). `undefined` is a
  null `any` to the comparison, which takes one for undefined.
- 140: an `await` in a `finally` block ("not supported yet"). The finally is code of its own after the try, as a
  finally that yields is (mlirGenTryWithYieldingFinally): run once on every way out of the try, its end, a `return`, a
  `break` or `continue`, a throw (caught into an `any` and thrown again after it). A `return`'s value is kept in a local
  of the function declared in front of the try (a generator keeps it in a field); in a function returning nothing,
  `return f()` calls f. Async functions, arrows and methods, with or without a declared result type, nested finally
  blocks, loops. Under -mm=own an object thrown through such a finally is refused, as through one that yields; the
  error says "awaits". Building it, the lookup of the function an `await`'s body is in walked into the async.execute op
  being built, attached to nothing yet, and crashed under -mm=own.
- Under -mm=rc a `for (let s ...)` of a string (or another reference) that a closure captures shared one variable: each
  closure saw the loop's last string (TSLANG 176 copied value types only). The iteration's copy takes a count of its
  own, as a `let` does, and one the body assigns gives its value back to the loop's variable.
- 117 (`let f = n.toFixed; f(1)`) stays a compile error (user ruling 2026-11-27): tsc accepts it, but JavaScript throws
  a TypeError at the call (the method loses its `this`), so the error says so where the program is written.
- `let x: any;` with no initializer is undefined under every memory model (user ruling 2026-11-27), with no "'let'
  does not have initializer" warning: it is `let x: any = undefined;`. Outside -mm=rc a local's slot was never
  written: a read took whatever was on the stack, and `x === null` dereferenced it, which crashed at --opt_level=0 (the
  debug-programs CI legs) and was hidden by O3. A local in a loop, a `try` or a generator starts each turn undefined.
  A module-level one (and -mm=rc's local) was a null `any`, which `===` reads as undefined but typeof, print, `+` and
  the casts dereferenced.
- A null `any` is undefined wherever it comes from: a class field of type `any` with no initializer (zeroed memory)
  crashed on typeof, print, `+` and the casts. Every read of a box (AnyLogic) takes a constant block of zero bytes
  for a null pointer, and its tag is undefined's descriptor. A number read out of an `any` holding undefined is NaN
  and out of one holding null 0 (`<number>u` threw "Can't cast from any type"); an integer of either is 0, as
  JavaScript's integer conversion gives (`undefined | 0`).
  An `any` compared for equality with a number, string, boolean or bigint is compared as an `any`, whichever side it
  is on: `u == 0` threw, `"undefined" == u` and `u == false` were true, and `false == a` did not compile ("'llvm.icmp'
  op requires all operands to have the same type"). Loosely, null equals undefined: a value that may be null
  (`object | null`, `C | null`, `string | null`) is `== undefined` when it is null - it was false whatever it held.
  Test `00any_null_is_undefined.ts` (all models).
- `00tslang_issue_205` failed to compile now and then (about 1 run in 100, more with the change above): "Can't
  resolve property 'cb' of type number", "has more parameters than another declaration of the same name". The
  scratch function evaluations run in (tempFuncOp) was erased at the end of the discovery pass and its handle kept;
  a function op made later at the same address was taken for it, so what a closure read of the function around it
  was not captured. The handles are cleared with the ops.

Not yet:
- Under -mm=rc an async function with a `catch` clause and a `finally` that awaits leaks about 32 bytes a call when a
  string is returned or thrown through it; a plain async function returning `name + (await v())` leaks too (before
  this batch), and more under -mm=own.
- Under -mm=own a `break` out of a `catch` clause leaks the caught value (before this batch).
- Under -mm=own a construct signature's function cannot give the constructor a parameter it keeps (`R(r: string)`:
  "this function does not own it"), and `Constructor<T>` of `new (...args: any[]) => T` is "a value takes a second
  reference".
- Still from the triage and 212: own 56, 128 (a captured `for` variable the body assigns, a string one), 153, 159; own
  `make(L)` of a rest constructor, a `let` captured and assigned after, a tuple literal with a new string given to a
  union of tuples; 131 (Object.keys of `object`, Object.defineProperty, Object.create, Record<string, X>), 108, 191's
  callable interfaces naming each other, recursive conditional types, `Symbol` as a value; design questions without a
  ruling: overloads of a declared function (136) and class constructor (140).

Tests `00tslang_issue_213.ts`, `00tslang_issue_213_await_finally.ts`, `00let_any_starts_undefined.ts` (all models),
`00tslang_issue_213_loop.ts` (gc, rc, none); diagnostic-location `await_in_finally.ts` removed (it compiles now).

214. Leftovers of 213 (2026-11-27). Fixed:
- Under -mm=own a `break` (or `continue`) out of a catch clause leaked what the clause made for a statement before it
  (`s = "c" + e; break L;`: the caught string unboxed): a temporary is given back where its block ends, and a jump of
  the block itself was not taken for a way out of it - the release went after the jump, never run. It goes in front of
  the jump now, as for a `return` or a `throw`.
- Under -mm=own a class whose constructor keeps a string given to a construct signature (`R(r: string)` for `{ new (r:
  string): R }` or an interface of one): "an argument is given to 'R.constructor', which keeps it, but -mm=own cannot
  move it here". The function made for the class is called through a value, which lends it its arguments; it gives the
  constructor a copy of each string, held by a local of its own until then. An object kept so stays refused (it can't
  be copied).
- Under -mm=own a tuple literal holding a new string given to a union of tuples (`const x: [number, string] | [number,
  string, string] = [i, "a" + i, "b"]`): "a value that borrows a field cannot be stored". The literal is built as the
  member of its length (a tuple of new values), not as its own type read out of its slot; a `const` of the union made
  of a literal's slot has storage of its own, as one of a tuple has, or what the literal holds was never given back.
- 213's "async leaks" are not leaks: a string an async function returns is made on the runtime's worker thread and
  given back on the caller's, and memory levels off (about 97 MB at 4M and 8M calls under rc and -mm=own alike).

Not yet:
- 128 under -mm=own (a captured `for` variable the body assigns): the iteration's copy is written back to the loop's
  variable after the closure took it, which own refuses ("captured by a closure that escapes, and is still used here",
  user ruling 2026-11-13). It needs a capture by value for value types.
- Under -mm=own `Constructor<T>` of `new (...args: any[]) => T`: the function made for the class takes no `any[]`, and
  the arity adapter around it is refused where it is boxed ("a value takes a second reference").
- A union of tuples asserted to one of them (`<[number, string, string]>x`) fails with nothing said ("failed
  statement": the other member is cast to it too), and indexed (`x[1]`) is "not applicable" (before this batch).
- Under -mm=own `x.slice(1)` of a union of tuples: "an element ... would have two owners".
- Still from 213: 56, 153, 159 under own; 131, 108, 191's callable interfaces, recursive conditional types, `Symbol`
  as a value; design questions without a ruling: overloads of a declared function (136) and class constructor (140).

Tests `00tslang_issue_214.ts` (all models).

215. Leftovers of 214 (2026-11-27). Fixed:
- `Record<string, T>` was an object with no fields in every model ("mapped type is empty for constrain: string"): the
  literal given to it lost its fields, `Object.keys` gave none and `r["a"]` read nothing (131). It is `{ [key: string]:
  T }`, the dictionary of T a string index signature is, both as the built-in Record (no default library) and as the
  mapped type `{ [P in K]: T }` of K = string (the default library's Record). Without the dictionary class, as before.
- A union of tuples cast to one of them (`<[number, string, string]>x` of `[number, string] | [number, string,
  string]`) failed with nothing said ("failed statement"): each member is tested for and cast, and the shorter one
  could not be made the longer - its cast read past its end. The member the value holds is cast; one lacking an
  element of the target throws "Can't cast from [number, string] to [number, string, string]" there, as a cast from
  an interface that does not hold the class throws (TypeScript reads the element as undefined). A tuple cast to one
  with more elements (not of a union) is a compile error that says so (tsc refuses it too).
- `x[1]` of such a union, "access expression is not applicable": an element at a constant index every member has,
  after the same element types, is read where the first member has it.
- A member of Array of a union of tuples (`x.slice(1)`, `x.join(",")`, `forEach`, `map`) read the value as the first
  member, in every model: a `[0, "c"]` held by `[number, string] | [number, string, string]` joined "0,c,null". It is
  the array of the member the value holds, of the union of all members' element types.
- Under -mm=own a tuple read from a variable made an array (`t.slice(1)`, `t.forEach(f)` of `t: [number, string]`)
  was refused ("an element ... would have two owners as an element of an array"): its strings are copied for the array,
  as a tuple cast to another copies them (211). Another element owning memory stays refused.
- A union of object shapes or tuples cast to one of its members boxed the union in an `any` to read its tag, and
  nothing gave the box back: 48 bytes a cast under rc and -mm=own. The union's own tag is read.

Not yet:
- Under -mm=rc an array made for a call and given as `this` to a method (`([i, s] as (number | string)[]).slice(1)`,
  `t.slice(1)` of a tuple) is never given back: about 100 bytes a call (before this batch).
- Under -mm=own `a.join(",")` of a `(number | string)[]` leaks about 120 bytes a call (default library; before this
  batch).
- Still from 214: 128 under own (a capture by value for value types), own `Constructor<T>` of `...args: any[]`; from
  213: 56, 153, 159 under own; 131 (`Object.keys` of `object`, `Object.assign`, `defineProperty`, `create`), 108, 191's
  callable interfaces, recursive conditional types, `Symbol` as a value; design questions without a ruling: overloads
  of a declared function (136) and of a `declare class` constructor (140).

Tests `00tslang_issue_215.ts` (all models).
