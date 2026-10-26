# -mm=own: generator methods over `this`, and the default library's `Map` and `Set`

Status: implemented 2026-10-26 (branch `own-generator-this-borrows`; DefaultLib branch
`own-generic-iterators`).

## 1. Why

`lib.generics.ts` - `Map`, `Set`, the array generators - is compiled into each program, with its
bodies, so it was never part of the own build of the default library (`lib.ts`). Under `-mm=own`
every `map_*` and `set_*` test of the DefaultLib suite failed to compile. The errors came in a few
kinds:

- `private *iter()` keeps `this` in its state, and `entries()`, `keys()`, `values()`, `forEach`,
  `[Symbol.iterator]` and every set operation call it on `this`, which they do not own.
- `@borrows("this")` (spec of the param markers) helped only for an exported or `@final` class: a
  method of any other class is called through its vtable (`ts.GetThis(ts.ThisVirtualSymbolRef(o))`),
  which the forwarder check and the bound check did not see through. And a generator method with
  the marker still kept `this`: a call moved the object into the generator, so a caller could not
  use it after the loop (`for (const c of w.iter()) ...; w.x` was "used after its value was moved").
- The insert paths store a key and a value that are the caller's: a string read out of an element
  (`for (const [k, v] of values) this.tryInsert(k, v)`), a borrowed string put into a yielded tuple
  (`yield [key, value]`), a string lent to a callback (`forEach`'s `f(entry.value, entry.key)`).
- `resizeHelper` moved entries out of one array into another, which nothing under own can do.

## 2. Rules

1. **A generator method's state borrows `this`.** When the signature pass finds a borrowing state
   (phase 7b), the box's store of `this` (`.this`) keeps nothing (`OWN_STATE_BORROW_ATTR_NAME`), so
   the method does not keep `this` and its callers keep their object. The states are found before
   the facts now; they need none of them.
2. **Through the vtable.** `parameterOf` (the forwarder check) and `checkBoundedResults` see the
   object a method is called on through `ts.GetThis` (`boundThis`). A reassignment of the object
   while its generator is read is the error a function generator's argument gets
   (`own_err_generator_method_reassigned`).
3. **A yielded tuple.** A state field read only into a tuple that is yielded - also cast to a named
   tuple, which reads the fields out into a new one - moves into the yield, as a field read straight
   into it does (`onlyIntoYield`).
4. **Strings as values, three more places (spec 22).** A `ts.CreateTuple` of borrowed strings that
   rc retains starts from copies (`copyIntoTuple`). A string read out of a place and given to a call
   that may free it is copied for the call, and the copy released after it, when the call is a
   callback (a function value nothing can mark `@nodrops`); a call that keeps it takes the copy. A
   direct call that may free it stays an error, so `@nodrops` keeps its meaning
   (`own_err_declared_drops`). A string given twice is copied twice (it was
   `own_err_string_call_keeps_twice`, an error for want of a retain to turn into a copy).
5. **A parameter kept on some paths only** is released on the others (spec 33's exit releases and
   spec 30's edge releases): `takenParams` is the parameters a move reaches every return from, not
   any kept one. It was `own_err_param_kept_some_paths`. The move must be in the function's top
   region (`placeOnUnmovedEdges`): a move inside a loop is still the error.
6. **A read is read again around a loop.** `findBetween` does not search past the read's own block:
   a path back into it (a `continue`) reads again, and the call is given that read
   (`for (const v of it) { if (other.has(v)) continue; result.add(v); }`).
7. **Length writes drop elements of their type.** A call's drops for `length =`, `pop`, `shift`,
   `splice` on an array it reaches are the elements of arrays of that element type (`-1`, the
   element's ref type), as `dropsChain` judges them in a body, not anything.
8. **Callbacks of methods.** A method no other method overrides gets a callback family (spec 33)
   from its calls through the vtable and through its bound method: its users may be its class's
   vtable and `ts.ThisVirtualSymbolRef`/`ts.ThisSymbolRef` whose methods are only called with the
   object first. A family holds every method of a slot's name, of any class; another member
   overrides this one when its class extends this one's (a base is the first field of a class's
   storage). A closed family with no members - a method no one calls but through the vtable - calls
   nothing through the parameter.

## 3. The library (DefaultLib `own-generic-iterators`)

- `@borrows("this")` on `iter`, `iterKey`, `iterValue` and their forwarders; the iterators read
  `this._entries[i]` at each step instead of keeping the array in their state, so an insert during
  iteration that gives the map a new array is not a dangling borrow. (A state that keeps a field of
  `this` is "borrows a field and cannot be stored": phase 7b's borrowing fields are read straight
  out of the box.)
- `forEach` is a loop over the entries, not over `iter()`.
- `resizeHelper` grows `_entries` in place (`length =`) instead of moving each entry into a new
  array.
- An entry is stored as a tuple cast to the entry type, not an object literal (a literal that holds
  what it owns moves only into a local, spec 36), in one place after the search loop (rule 5).
  `Set`'s overwrite branch was dead (a set never overwrites an equal value) and is gone.
- `hashCodeGeneral` hashes the parameter's own bytes (`Ref(key)`) instead of a local copy, which
  took the key under own.
- Bug fix, every model: `size` was `count`, which still counts deleted entries on the free list.

## 4. Not here

- `any` or object keys and values (settled in section 6: through `Shared<T>`). The constructor from an array (`new Map([[k, v]])`) cannot take
  the elements out of the caller's array, and moving an object parameter into a tuple
  (`Retain` of a tuple made of kept parameters) is "takes a second reference". This waits on the
  ruling on containers of owned values (two sets sharing an element by `union` is two owners).
  Tried and dropped: a parameter kept through a tuple field (`entries[i] = <Entry>[key, value]`)
  counted as kept. It made `set` and `add` keeping calls, and a string given to two of them
  (`m1.set(k, 1); m2.set(k, 2)`) has no retain to copy (spec 22.4): an error that the copy into the
  tuple (rule 4) avoids, while no object case compiled with it either.
- `new Set([1, 2])` instantiated `Set<any>` beside `Set<si32>` (MLIRGen specialized by the default
  type arguments before inferring them, in every model), so an untyped container failed under own
  for the `any` case. Fixed after: the defaults are taken only when neither the constructor's
  arguments nor the receiver give the type arguments.
- A callback that assigns a captured variable (`(s) => { r += s.n; }`) has no family, for a function
  as much as a method: what it lends is still an error.
- A module-level `Date` with a method that changes it (`event.setDate(24)`) is "a value read out of
  the global is lent to a call that may release or overwrite it", though `setDate` writes a number.

## 5. Tests

Positive, under own (AOT and JIT): `own_generator_method_this`, `own_string_copy_call`,
`own_param_kept_some_paths`, `own_method_callbacks`, `own_loop_read_again`,
`own_length_drops_elements`, `own_tuple_param_kept`; each also prints the same under gc, rc and none.
Errors: `own_err_generator_method_reassigned`, `own_err_callback_method_overridden`.
Retired: `own_err_param_kept_some_paths` (rule 5), `own_err_string_call_keeps_twice` (rule 4).

## 6. Objects in `Map` and `Set`: `Shared<T>`

Ruling (2026-10-26): under own an object goes into a `Map` or `Set` as a `Shared<T>` handle; a plain
object or `any` element is a compile error at the user's code that points at `Shared<T>`. A container
keeps what it is given and hands out borrows of it (`get`, the iterators), and `union` or two maps
holding one value would be two owners of a block with one owner. A handle is counted (spec 23.2),
so it can be in any number of entries; values and strings (copied in, rule 4) need nothing.

- **`@sharedElements`** (`SHARED_ELEMENTS_DECORATOR`), on a generic class: under own each type
  argument has to be a value (owns no heap memory), a string, or a handle (`isSharedHandleType`, so
  `Shared<T> | null` too). `checkSharedElements` runs where the class is specialized
  (`instantiateSpecializedClassType`, both overloads), with the location of the user's `new` or type
  annotation, before the members are generated, whose errors would point into the library:
  `-mm=own: 'Map' cannot hold 'Node' (its 'V'): an element has to be a value, a string or a Shared<T>
  handle; use 'Shared<Node>'`, and for `any` `give it one of those types`. Keys are checked as values
  are. A `new` whose inferred type arguments were refused does not go on to the defaults
  (`sharedElementsRefused`), which would report `any` too. The other models ignore it. DefaultLib puts
  it on `Map` and `Set`; nothing in the compiler names them.
- **A handle in a tuple entry** (`copyIntoTuple`): a tuple made of a handle beside borrowed strings
  retains the handle for the tuple, as it copies the strings (`<MapEntry<K, V>>[hash, next, key, value]`).
- **A handle a yield takes out of the state.** A field read only to be yielded
  (`OWN_YIELDED_FIELDS_ATTR_NAME`) gives its value to the caller, who owns the record `next` returns
  and releases it; the state's release of the old value is erased. A handle's counting is rc's and
  stays (spec 23.2), and that kept this release too: each step released the previous handle twice,
  once in the state and once in the caller (a `Map` of two handles walked by `for...of` freed the
  first). The release of a yielded field is now erased for a handle as for anything else.
- **The owned result of a virtual call** (`OwnedReturnConsumptionPass`): whether every method a
  vtable call can reach returns its result owned was decided over every method of that name in the
  module, so with the default library loaded `A.get` stood beside `Headers.get` and the like, the
  call was not +1, and an rc handle it returned leaked. Now only methods of a class on the receiver's
  chain (its bases and the classes extending it) count (`onReceiverChain`).

Tests: `own_shared_elements` (a local `@sharedElements` container of numbers, strings and handles,
under every model), `own_shared_yield_count` (two entries walked, a pair yielded, a `break`; counts
under rc and own), `own_err_shared_elements_object`, `_any`, `_inferred`.
