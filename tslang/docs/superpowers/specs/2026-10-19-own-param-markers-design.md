# `@keeps` and `@borrows`: ownership facts in a function's signature

Date: 2026-10-19. Status: implemented (branch `own-param-markers`), with the own generator fixes
`matchAll` needed.

## Why

After `@nodrops` and `@final` (PR #611), the default library still did not build under `-mm=own`.
The remaining errors had one cause, "decision B": an exported function keeps or borrows a
parameter. OwnershipSignaturePass computes these facts from the body (`__own_params`,
`__own_result_bounded`). It drops them for any function a caller outside the module may reach (it
is "open"), because a caller that does not know them borrows an argument the callee keeps, or
releases a result the callee still points into. Either way that frees twice. The sites were
`Response.constructor` (keeps `headers`), `___String.matchAll` and `String[Symbol.iterator]`
(generators whose state borrows `this` and `expr`), and the `String.matchAll` wrapper.

The user chose markers: the facts become part of the function's declared signature, so every
caller knows them, including one in another module, which re-parses the declaration.

## The markers

Both go on a function, method or constructor, and name its parameters. `"this"` names the receiver.

- `@keeps("headers", "body")`: the function keeps these arguments. Every caller moves them in.
- `@borrows("this", "expr")`: the caller owns the result, but the result may not outlive these
  arguments. The result is either the state of a generator over them, or what a call with such a
  bounded result returns (`return this.value.matchAll(expr)`, a "forwarder").
- `@lends("l")` (added 2026-10-20): the result is a borrow of the one argument `l`
  (`__own_result_borrows`). The caller owns nothing of it, and it may not outlive `l`. It names
  exactly one parameter, and it cannot be combined with `@borrows` (a result is either the
  caller's or lent). It can be combined with `@keeps`.

Under gc, rc and none they mean nothing. MLIRGen still records them, and the LLVM lowering drops
the attributes.

## How it works

- **MLIRGen** (`processOwnParamMarkers`) turns the names into argument indices (FuncOp attributes
  `ts.own_declared_keeps` and `ts.own_declared_borrows`). It also records the argument names
  (`ts.own_declared_param_names`) for the errors. A name that is not a parameter is an error. The
  generator wrapper copies the method's decorators, so a marked generator's maker carries them.
- **OwnershipSignaturePass:**
  - A marked function's `@keeps` and `@lends` facts stand even when it is open (`meetFacts`). A
    family does not erase them. A call this pass cannot resolve gets them from its callees'
    signatures (`computeFacts`).
  - A call gets what its callees' signatures say, whether or not it is resolved
    (`declaredByCallees`, over `Callees::named`). The named callees are the direct callee, every
    member of a virtual slot's family, or a dynamic import's address global.
  - A call whose callees say `@borrows` gets `__own_result_bounded` and `__own_fresh_result`: the
    caller owns the state.
  - `findBorrowingStates` accepts a marked maker even though it is public, and a method reached
    through its vtable. It also accepts `this` held by value (`.this`, read from the parameter's
    slot).
  - `findBoundedForwarders` gives a marked non-generator what its returned bounded calls borrow,
    mapped back to its own parameters (directly, or a field read of one).
  - `checkDeclared` holds each body to its signature, exactly. It also checks that the overrides in
    a marked method's family say the same. A function that says `@keeps` or `@lends` must say both
    facts its body has: a body that returns a borrow needs `@lends`, one that keeps an argument
    needs `@keeps`.
  - The error for an open function that returns a borrow or keeps a parameter says that `@lends`
    or `@keeps` would put it in its signature.
- **OwnershipInferencePass:** a marked function may return a bounded result (`boundedUses`).
- **Across modules:**
  - `__decls` prints the markers with the printed parameter names (`p1`, and `this` for a
    receiver). `MethodInfo::ownKeeps` and `ownBorrows` carry them for methods;
    `addFunctionDeclarationToExport` gets them from the FuncOp.
  - A dynamic import's function is a global holding its address, so
    `mlirGenFunctionLikeDeclarationDynamicImport` copies the declaration's markers onto that global.
  - DefaultLib's `lib.d.ts` says the same as `lib.ts`.

## Found on the way and fixed here

- **A silent miscompile.** A public generator method over `this` with no marker compiled under own
  (release), and the state's release freed the caller's object. The caller then freed it a second
  time, and the program exited 127 with no output. `classifyStateBoxes` let a non-borrowing box
  take rc's reference for a parameter the function does not own. It now leaves that reference
  alone, so the existing "is a parameter, kept here" error fires (`own_err_generator_method_this`).
  `--di` already errored there, by accident: its `ts.DebugVariable` use broke the one-use check
  that paired the reference.
- **`--di`.** `retainBefore` and `twinReadAfter` now ignore `ts.DebugVariable` uses (`soleUser`),
  so a marked generator over `this` builds with debug info too.

## Generator locals under own

`matchAll`'s body needed three fixes to own generators that markers could not give. Each is
covered by `own_generator_locals.ts`, under gc, rc and own.

- **Narrowing.** A generator's locals are fields of its state object, so narrowing one
  (`if (result == null) break;`) made a second field, filled from the first: a second reference.
  `addSafeCastStatement` now declares a generator's narrowed local as a view of the local's own
  storage (`VariableDeclarationDOM::narrowedType`). Each read loads the local and casts it, so a
  read after a resume sees the current value. An SSA copy would not: it read stale values after a
  yield, and gc printed 3 where 9 is right.
- **A spread in a generator.** `[...a]` is a `for...of` that MLIRGen writes, and its `.a` alias of
  the source array became a field holding a second reference. The loops of the spread, `forEach`,
  `every` and `some` hold no yield, so their locals stay in the frame (`allocateVarsInContextThis`
  and `allocateVarsOutsideOfOperation` off). `isBuiltArrayReturned` also accepts the built array
  moved into a field, `h.c = [...a]` or a generator's local.
- **A yielded local.** `yield groups` read the field into the `{value, done}` record, a borrow of
  the state. OwnershipSignaturePass now finds the state fields that are read for nothing but
  yields, where each stored value is yielded later in its block (`findYieldedFields`). The yield's
  read is fresh (the value moves to the caller), an overwrite gives nothing back, and the state's
  release routine skips the field (`OWN_YIELDED_FIELDS_ATTR_NAME`). `matchAll` computes its
  empty-match skip from `result` before the yield, so `groups` is only yielded.
- **The owned-return pass took a generator's `next` for a class method.** OwnedReturnConsumptionPass
  matches a virtual call's candidates by member name. A class's `next()` was grouped with every
  generator's `next`, whose record does not classify as owned, so `src.next()` lost
  `__owned_result` and own reported a second reference. Only class methods (a class first) are
  candidates now.

With these, `lib.ts` compiles under gc, rc, none and own, release and `--di`. DefaultLib's
`build.bat` and `build.sh` build own as a fourth model.

## Not done (follow-ups)

- **Iterating another module's generator under own** works now. The user ruled that what an
  iterator's `next()` returns is the caller's, through the `Iterator` interface or into another
  module, and only that, not every unresolved call. OwnershipSignaturePass marks such calls fresh
  (`isIteratorNext`: a callee named `next` returning `{value, done}`), and so is a call that
  returns a generator's state object (`makesGeneratorState`), which a generator function always
  makes. `isFresh` reads that mark before it rejects a declared callee, and `walkBorrowed` does not
  follow a fresh result that is not bounded. Another module's generator state is released here
  without its fields (`findImportedStates`): only the box's cells, and their values unless the
  maker says `@borrows`. What its locals still hold when it is given up early leaks. A release
  made from the shape freed the borrowed argument a second time (exit 127). `import_final_class`
  and `import_own_markers` now run under own too. `own_iterator_next.ts` covers the
  interface-shaped call in one module, which already worked.
- **An own DefaultLib returned corrupted regex strings.** Fixed 2026-10-20. It was not a borrowed
  temporary. `resize` assigns its `this` parameter (`this.length = n`), and under own a string
  parameter's slot did not own what was assigned to it: that string leaked, and `return this`
  returned a copy of it, read up to its first NUL - one byte, which the caller filled with `len`
  characters. Under own a string parameter the body assigns (including `p.length = n`) now starts
  a local of its own, made from a copy, as rc already did for every assigned parameter
  (`mlirGenFunctionOwnAssignedParams`). Covered by `own_string_param_assigned.ts`.
- **A generator local's previous value leaks** under rc and own (on main too): a `const` declared in
  a loop inside a generator is stored into its state field again each iteration, and nothing gives
  back what the last iteration stored. Releasing it there fixes rc, but under own it makes `next`
  a call that may drop, and a caller's `for...of` over such a generator is then rejected (its
  loop variable borrows the record). The fix wants the caller-side `for...of` move as well.
- **A generator local's type from a function declared later** is `none` in the state object, and
  the lowering fails ("failed to legalize operation 'ts.Cast'") under every model. `matchAll`
  annotates `const more: boolean`.
- **An ordinary function whose result points into an argument** (`__own_result_borrows`, where
  the caller owns nothing) says so with `@lends`, added 2026-10-20. Covered by `own_lends.ts`, the
  `own_err_lends_*` cases, and `import_own_markers` / `import_own_markers_lent_outlives` across
  modules.
- **A ternary that picks an array literal or `null`** (`return i < 3 ? ["a"] : null`) fails MLIR
  verification ("operand #0 does not dominate this use") under rc and own. This is unrelated.
- **Android** still builds gc, rc and none only.
