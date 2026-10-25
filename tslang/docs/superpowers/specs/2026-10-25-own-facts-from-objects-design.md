# -mm=own: what a call overwrites, and facts read from the objects of imported modules

Status: implemented 2026-10-25 (branch `own-call-field-drops`).

## 1. Why

native_ts_graphics, a graphics sample collection compiled with tslang, builds every example from
separately compiled modules - the example, `core/donut.ts` (wrapper classes over a C++ library),
`core/simple_camera.ts` and so on - each into an object of its own, linked into one program. Under
`-mm=own` about 1200 errors remained after the project's own fixes (`@cpp` handle classes, `@value`
wrapper classes, `@nodrops` C declarations), nearly all `'x' borrows a field but is used here after it
may be released or overwritten`, for two reasons:

1. **A callee that destroys something destroys everything.** A method that replaces an array field of
   its own object (`this.position = [x, y]`), called on an object held in a field
   (`this.camera.reset()`), dropped the caller's borrow of `this.camera`: the call was not
   `__own_no_drops`, so it might destroy anything, though it can only destroy the old `position`.
2. **Nothing crosses between modules linked statically.** An imported `.ts` module is re-parsed as
   declarations; its functions have no bodies there, so a call of one may destroy anything. Facts cross
   only from a shared library (`__tsown_`, read through the loaded library) and, for stack promotion,
   from static libraries the program links (`__tsborrow_`).

## 2. What a call overwrites (`__own_drop_fields`)

`OwnershipSignaturePass::computeDrops` keeps `noDrops` as it was and adds, for every other function,
its `FieldDrops`: the places it may overwrite when that is all it destroys that a caller can reach -
each place's position and slot type, the same pair a store in the caller's own body is judged by
(`OwnershipInferencePass::overwritesChain`, factored out of `dropsChain`'s `ts.ReleaseSlot` branch):

- a field's position (`OWN_INTERFACE_FIELD_POSITION` for one reached through an interface, which then
  matches any field), -1 for an element, and the slot's reference type;
- collected from the body's overwrites `dropsInBody` counts (`fieldDropsInBody`); anything else it
  counts - a global's assignment, a captured variable's, a handle's payload (`x.value =`), a popped,
  shifted or spliced element, `delete`, a handle released - makes the function `any`;
- joined with its callees' to a fixed point; a call whose callees are not all known, or one of them
  `any`, makes it `any`.

A call whose callees all have `FieldDrops` gets `__own_drop_fields` = `[[position, type]...]`
(`calleesFieldDrops`). `dropsChain` judges such a call as those stores would be judged in the caller,
and only narrows what it said before: a chain the function owns is still dropped only when the call is
given something derived from its root, and both must hold. Not for a chain whose places are not all
known (`chain.anyPlace`, a borrowing call's result), nor a call given a closure.

This is as sound as the store rule it reuses: inlining the callee's stores would give the same answer.

## 3. Facts from objects (`__tsownfacts_`, `--own-facts`)

- **Export.** Every module built under own whose object another module may link (not the JIT's, nor
  the JIT cache's) publishes `__tsownfacts_<file>_<hash>`, named like `__tsborrow_` after the main
  source file, whatever it exports: the default library's static `lib.obj` is compiled with
  `--export=none`, and has no `__tsmm_own_` marker, yet a program links it and calls `Math.abs`. It
  is a NUL-terminated char array, so it can be read from an object without relocations (`__tsown_` is
  a pointer, read through a loaded library), made by the LLVM lowering from the module attribute
  `ts.own_object_facts` as `__tsborrow_` is. One line per non-private function: its name alone when
  it is `__own_no_drops`, else its name and a tab-separated `<position> <slot type>` per place, when
  it has `FieldDrops` with at least one place. Types are printed MLIR types and parsed back by the
  importer.
- **Fresh results.** A line ends with a `fresh` field when the function's one result is its caller's
  to own (`returnsFresh`): its body borrows no argument for it (`factsLocal` and `factsEffective`
  have no `borrows`, and no record), and it says no `@lends`, `@borrows` or `@record`. That is what
  `isFresh` already assumes of a call of a function defined in the same module: every function
  returns its result retained (rc 9.24). A body that returns a borrow - a parameter, a field of one,
  a mortal global - and can be called from another module is an error in its own module, so a
  function in the facts text never returns one silently. The importer puts these names in
  `ts.own_imported_fresh_results`, and their calls get `__own_fresh_result`: `this.eye =
  transform3(this.eye, m)` may release the old array. An older compiler reading `name\tfresh` fails
  to parse the place and gives the function no facts: safe.
- **Import.** `--own-facts=<object or static library>` (repeatable, comma-separated; only under own)
  names the objects of the modules this one imports. The driver reads their `__tsownfacts_` texts
  (`readOwnFacts`, beside `readBorrowFacts`); MLIRGen puts the names on the module as imported no-drops
  functions (`ts.own_imported_no_drops`, as a shared library's are) and the places as
  `ts.own_imported_drop_fields`. A line whose type does not parse gives that function no facts.
- **Use.** `OwnershipSignaturePass::addImported` resolves a declaration by its name, as it does an
  imported no-drops function; one with places sets `Callees::importedOverwrites`, so the call gets
  `__own_drop_fields` instead of `__own_no_drops`, and `callMayDrop` counts it. A `..new` gets
  `__own_fresh_result` from a no-drops library, as before, and any function whose line says `fresh`.
- **The default library.** Without `--no-default-lib`, and not under the JIT, the driver also reads
  the facts of the static default library the link will use (`getDefaultLibBinary`, the build and
  model of this compile; never its DLL, which a program may load another copy of).

The facts must describe the code linked: the build passes the objects of the modules it links (or
objects compiled from the same sources with the same flags). tslang cannot check that, as it cannot
for `__tsborrow_`; a stale object is the build's error.

Only calls by name benefit. A virtual call of an imported class's method cannot be resolved here (its
vtable is the other module's), so a class meant to be called across modules is best `@final`.

## 4. Tests

- `own_call_field_drops` (run under own, gc, rc and none: `3 7 4`): a method replacing its own array
  field, called on a field-held object, directly, through a method, through an interface.
- `own_err_call_field_drops_owner`, `_under`: a method overwriting the borrowed field, and a field
  under the borrowed object, still drop the borrow.
- `own-facts-objects.cmake`: `own_facts_lib.ts` compiled to an object; `own_facts_main.ts` fails without
  `--own-facts` and compiles, links and prints `11` with it (`shifted`'s fresh result replaces the
  array it was given); `own_facts_main_drops.ts`, whose imported callee overwrites the borrowed
  place, fails with it.

## 5. Compile-time constants (`__MEMORY_MODEL__`, `__OS__`, `__ARCH__`)

String literals of the memory model (`gc`, `rc`, `own`, `none`) and of the target's OS and
architecture, named as Node's `process.platform` and `process.arch` name them (`win32`, `linux`,
`android`, `darwin`; `x64`, `ia32`, `arm64`, `arm`, `wasm32`). A compare of one with a string literal
(`===`, `!==`, `==`, `!=`) is decided while generating (`foldLiteralCompare`), so an `if` on it
generates the branch taken only: under own a generated wrapper does not keep a callback that only
gc's collector needs kept. Only these constants' ops are folded (`ts.compile_time_constant`): other
values' literal types - a narrowed field's, `~1`'s - need not be what they hold at run time, and
folding every literal compare broke 46 tests. `00memory_model_constant.ts` runs under all four models.
