# -mm=own: what an append-only global keeps is immortal

## 1. Why

A callback a C library calls later has to stay alive after the call that registered it. Under gc a
TypeScript wrapper keeps it in a module-level array that is only ever pushed to, so Boehm sees it:

```ts
let retained: Callback[] = [];
setRenderCallback(handler: Callback) { retained.push(handler); Donut_SetRenderCallback(this.handle, handler); }
```

Under own that push is a move: the wrapper keeps `handler`, so its caller must give it up, and a
bound method `this.onRender` cannot be given up (its object is the caller's `this`, owned by
`main`). A closure that captures a local fails the same way ("captured by a closure that escapes,
and is still used here"). Since `__MEMORY_MODEL__` the generated wrappers skip the push under own,
and the lifetime is the C++ one: the pass objects outlive `App.run()`. This design lets own prove it
instead: what such an array keeps lives to the end of the program, so it is made immortal where it
is kept, as an immortal global is (spec 25).

## 2. Keepers

A **keeper** is a `ts.Global` that:

- is not exported or imported (no `export`, `import`, `dllexport`, `dllimport`), and is not reached
  through a vtable;
- is an array whose element type is a class, a bound function, a function value (`hybrid_func`), a
  string or an array;
- is given its value only by its region, which yields a new array (`[]`, `ts.Constant` of an array,
  `ts.CreateArray`, `ts.NewArray`);
- is used only by `ts.AddressOf`s whose users are a `ts.ArrayPush` (the array operand), and
  `ts.Load`s whose users read it: `ts.LengthOf`, and `ts.ElementRef`s that are only loaded.

Nothing ever takes an element out of a keeper or overwrites one (no `pop`, `shift`, `splice`,
`length =`, element store, nor the array given to anything), and nothing releases a global. So
what it holds is never released: it lives to the end of the program.

`findKeepers` (OwnershipSignaturePass) marks each `ts.AddressOf` of a keeper
`__own_keeper_global`, and each `ts.Load` of one of its elements `__own_keeper_read`.

## 3. Immortal at the push

A value pushed into a keeper is made immortal there: its block's header becomes
`HEAP_BLOCK_IMMORTAL`, so every release skips it, its fields included (a release of an immortal
block runs no destroy body). Writing the header only leaks; it frees nothing. So it is sound wherever
the block is a heap block with a header, and every block the value relies on is made immortal too.
Two blocks need more than the value's own:

- **A bound method** (`this.onRender`) holds its `this` without owning it, so the object is
  marked.
- **A closure that does not escape** borrows its cells and the values it captured by value from the
  frame that built it, which releases them. Its box, each cell and each value captured by value
  are marked.

Only the function that builds a value knows these blocks: a callee given a `hybrid_func` does not
know whether its context is an object, a box (whose layout it does not know) or a C++ object with
no header at all. So the mark is made by the function that knows the value's origin, and a function
that pushes a parameter passes the duty to its callers:

- **The value is a parameter** (a load of the parameter's slot, through casts): the function
  **immortalizes** that parameter (`__own_immortalizes`, argument indices). Nothing is marked
  there; every call of it marks its argument, by these same rules.
- **The value is built here, or read out of an owning local whose initializer it is**: a bound
  method (`ts.ThisSymbolRef`, `ts.ThisVirtualSymbolRef` of a class that owns its block, not
  `@cpp`), a closure (`ts.CreateBoundFunction` of a capture box), an allocation (`ts.New`,
  `ts.CreateArray`, `ts.NewArray`), or a string. It is marked here.
- **It is immortal already**: a literal, an immortal global's read, a keeper's element.
- **Anything else** (a field's value, a call's result, a merge): an error at the push, "-mm=own
  cannot make the value kept in 'retained' immortal here".

A call passes an argument to a parameter its callee immortalizes. That argument is decided by the
same rules: a parameter of the caller propagates (a fixpoint over the calls, as `@keeps` facts
are), and anything else is marked at the call.

## 4. In the IR

- `%i = ts.MakeImmortal(%v)`: the same value, now immortal. Inserted right after `%v`'s definition
  (the root, before casts); every other use of `%v` is given `%i`. With `assumed` it marks nothing:
  it wraps a parameter its callers made immortal.
- `ts.MarkImmortalCell(%cell)`: marks a captured variable's cell.

To inference, `ts.MakeImmortal`'s result is immortal (`isImmortalLiteral`): its retains go, and it
may be pushed, stored, returned or given to a call that keeps it, as a string literal may. Its
operand is only borrowed, so the frame still owns what it built and releases it at the end, which
the header makes a no-op. A keeper's element read is immortal too, and is no place read. A keeper's
`ts.AddressOf` is no global read either: the array is never replaced, only appended to. A push of
an immortal value keeps nothing, so a parameter pushed through `ts.MakeImmortal` is not
`__own_params`. That is why the caller's `this` can go to `setRenderCallback`.

The lowering, under own only:

- a class or object instance, a string: the block, when not null and not already immortal (a stack
  object and a literal have immortal headers, and a literal's may be read-only);
- an array: its header block, when there is one;
- a bound function: its `this`, an object or a capture box;
- `assumed`: nothing.

## 5. Across modules: `@immortalizes`

A function another module may call (open: exported, or its address taken) cannot tell callers it
does not see to mark their arguments. Like `@keeps`, an open function that immortalizes a parameter
must say so: `@immortalizes("handler")`. It is checked against the body (`checkDeclared`): a
parameter it names that the body does not immortalize, or one the body immortalizes that it does
not name, is an error. A bodyless declaration's `@immortalizes` is trusted, as `@nodrops` is: a C
function that keeps a callback for good. The decorator is carried in an importer's re-parsed
declaration and in a shared library's declarations, so every caller marks its arguments.

A call through a class family or an interface immortalizes a parameter only when every callee does.
When the callees disagree, the function that immortalizes is an error: some calls of it would not
mark.

## 6. Not here

- A keeper of optionals, unions or tuples: their payloads would need marking through the tag.
- A function value of unknown origin (a field's `hybrid_func`): its context may be a bound method's
  object, which only its type tells.
- Removing elements from a keeper: that is an ordinary array, and the push is the move it is
  today.

## 7. As built

- `OwnershipSignatureKeepers.cpp`: `findKeepers`, `classifyKept`, `decideKeptForGood` (runs after
  `findOpen`, before `computeFacts`: an immortal value keeps nothing, so a parameter pushed through
  `ts.MakeImmortal` is not `__own_params`), `makeImmortal`. A call's callees are its resolved
  functions, or, for a call the pass knows nothing else of (another module's function with no
  facts), the declarations of the symbols it names, whose `@immortalizes` is read.
- Inference: `isImmortalLiteral` and the retain decision accept `isKeptForGood`; a keeper's element
  read is neither a place read nor a global read; `ts.MakeImmortal` and `ts.MarkImmortalCell` are
  borrows; `followClosure` takes `ts.MakeImmortal` as a use and follows it no further, so a closure
  kept for good borrows its cells from the frame (which then may still use the captured variables),
  and whatever is done with the immortal value afterwards bounds nothing.
- Stack promotion (`StackEscapeFacts`): `ts.MakeImmortal` and `ts.MarkImmortalCell` are an escape,
  reason "kept for good", so nothing kept is ever a stack object.
- The decorator is carried like `@keeps`: `processOwnParamMarkers`, the global a declaration's calls
  go through, a shared library's printed declarations (a 32-bit mask in a class's method info, which
  fits where a 64-bit one would grow the vtable's `SmallVector` past its inline limit).

Existing programs change only where an array was a keeper: `own_err_closure_escape_twice`, `_loop`
and `_some_paths` pushed closures into a global that is now one, so they were accepted (soundly);
they now empty it once (`handlers.length = 0`), which keeps their errors. `own_closure_escape` and
the stack-promotion test `pass_to_keeper` push into keepers and pass, the latter with the reason
"kept for good" under own.

## 8. Tests

- `own_immortal_keepers` (own, gc, rc, none; compile and JIT): a registry pushing a parameter into
  a keeper and using it after; a constructor registering `this.onTick` from a function that returns;
  a closure over a local the frame still uses; a forwarding function, and an exported one that says
  `@immortalizes`; objects pushed directly, fresh and from a local; a keeper's element stored into a
  field; churn between, so a freed block would be handed out again.
- `test-own-keepers-objects` (own-keepers-objects.cmake): the keeper in a library object, an
  importer registering a bound method and a closure from functions that return; `2 102 2`.
- `own_err_keeper_*`: a value of unknown origin pushed (`unknown_value`), an exported function
  without `@immortalizes` (`exported`), an `@immortalizes` the body disagrees with (`declared`), a
  function that keeps for good passed on as a value (`called_through_value`), and an array that is
  also popped, which is no keeper, so the move error stays (`popped`).
