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

For native_ts_graphics, from the same sweep (its code, not the compiler):
- `examples/dynamic_primitive_clipping.ts:223`, `desc.setDepthClip(clip)` with `clip` the loop's `int`,
  fails under every model since #846 (a number is not a `boolean` argument, TS2345). alpha127 still
  takes it; the next release won't. `desc.setDepthClip(clip != 0)` fixes it.
- Under `-mm=own`, 7 `core/` modules do not compile yet (`directx_math.ts`, `donut_first_person_camera.ts`,
  `nri_donut_textures.ts`, `octomap_common.ts`, `ray_tracing_extended_scene.ts`, `simple_camera.ts`,
  `rtxpt_game.ts`), and nor do most examples, which #310 did not port: with every facts object built
  and 50-53 in, 143 of the 276 sources fail. The largest group left is "borrows a field but is used
  here after it may be released or overwritten" (about 280 sites).
