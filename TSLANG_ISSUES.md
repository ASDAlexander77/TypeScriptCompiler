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
