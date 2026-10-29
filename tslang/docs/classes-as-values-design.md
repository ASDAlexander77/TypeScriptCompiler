# Classes as values, and local classes that capture: design

Status: **proposed, not implemented**. Written 2026-10-29, to be picked up later.

## 1. What works and what fails today

**Works:** a class used under another name at compile time.

```ts
const C = A;              // an alias: `new C(3)` is `new A(3)`
const D = class { v = 7; };
```

**Fails:** a class passed or stored as a run-time value.

```ts
function make(c: typeof A, n: number) { return new c(n); }
make(A, 1); make(B, 1);   // B extends A
```

This fails with "failed to legalize operation 'ts.ClassRef' that was explicitly marked illegal". `ts.ClassRef` names a class at compile time only. No lowering turns it into a value, so a class given to a parameter, stored in a variable, an array or a field, or returned, has nothing to lower to.

**Fails:** a class declared in a function whose members read the function's locals.

```ts
function counter(start: number) {
    class Counter { n = start; inc() { this.n += start; return this.n; } }
    return new Counter();
}
counter(2).inc();         // "Expected 1 arguments, but got 0."
```

The members are lifted to module functions that take the captured variables as an extra parameter, the way an arrow function's body does. A call through the object (`c.inc()`), or the field initializers run by `new`, never pass it.

## 2. Classes as values

### 2.1 Representation

A class value is a pointer to the class's vtable. Every vtable already has the pieces a value needs: `.instanceOf`, `.new` (`NEW_METHOD_NAME`, allocation), `.rtti` and the methods. `typeof A` becomes a type, `ClassValueType<A>`, lowered to that pointer.

- **Assignability:** `B`'s value is assignable to `typeof A` when `B` extends `A` and its constructor accepts `A`'s arguments. TypeScript checks the construct signatures structurally; nominal "extends" plus a parameter check is enough here.
- **`ts.ClassRef` used as a value:** it becomes the address of the class's vtable (`ts.AddressOf @A..vtbl`), as the object constructor already stores it.

### 2.2 Construction through a value

`new c(n)` with `c: typeof A` takes three steps:
1. **Allocate:** call the vtable's `.new` slot.
2. **Construct:** call the constructor, which must be reachable from the vtable too. A new slot, `.constructor`, typed as `A`'s constructor; a subclass fills it with its own constructor, which takes the same arguments for an assignable class.
3. **Result:** the result type is `A`.

Static members (`c.create()`, `c.count`) need slots of their own, or a second table the vtable points to. The first version can leave them out: report "static member through a class value is not supported".

### 2.3 Ownership

- **The class value owns nothing:** it points at static data, like a string literal. It is immortal under rc and own.
- **The constructed object:** it is a fresh owned result, like `new A(n)`.

### 2.4 Cross-module

A vtable of a class from a `-shared` library is already addressable (`isDynamicImport`), so a class value can refer to it. The new `.constructor` slot changes the vtable layout. Classes compiled by an older compiler would not have it, so the slot goes at the end, and the layout contract in `docs/interface-vtable-simplification-design.md` gains one entry.

### 2.5 Tests

- a factory that takes `typeof A` and is given `A` and a subclass;
- an array of class values;
- `instanceof` against a class value;
- under gc, rc, none and own.

## 3. Local classes that capture

### 3.1 Options

**A. Captures in the instance.** The class's storage gains a hidden field, `.captured`, holding the capture box (as an object literal's `.captured` field does, `MLIRGenExpressions.cpp`).
- **Filling it:** `new Counter()` in the declaring function fills it from the current box.
- **Reading it:** members read the captured variables through `this..captured` instead of an extra parameter, so a call through the object passes nothing new.
- **Pros:** works for methods, accessors, field initializers and the constructor, and for an object that outlives the function (`return new Counter()`).

**B. One class per declaring call.** Instantiate the class's functions again per capture box. That's impossible without run-time code generation.

**C. Reject.** Report an error for a member of a local class that names a local of the enclosing function, instead of the misleading arity error.

### 3.2 Recommendation

1. **Now:** do C. It replaces "Expected 1 arguments, but got 0" with an error that names the captured variable.
2. **Then:** A.
   - **Boxes are reused:** they're the same capture boxes arrow functions use (`ts.Capture`, `__capture_box`).
   - **Under own:** the box is owned by the instance and freed with it. The instance's release routine releases `.captured`, the way a closure's box routine does (spec 2.5 in the own design document).
   - **Under rc:** the box is counted like any other field.

### 3.3 Interaction with classes as values

A local class used as a value (`return Counter`) carries its captures. Its value then has to be a pair, `{vtable, box}`, not the vtable alone. Keep the two features apart at first: class values only for module-level classes, and an error for a local class that captures and is used as a value.

## 4. Non-goals

- `abstract new () => A` construct signatures and mixins (`function Mixin<T extends new (...a: any[]) => {}>(Base: T)`): these build on class values but need generic construct signatures. A follow-up.
- `typeof` on a class value returning `"function"` as in JavaScript: tslang's `typeof` reports its own kind names. Decide when class values land.
