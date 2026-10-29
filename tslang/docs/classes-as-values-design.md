# Classes as values, and local classes that capture: design

Status: **class values implemented for module-level classes; local classes that capture are an error
(option C)** (2026-10-29); see §5. Captures in the instance (option A) are not done.

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

## 5. What is implemented

**The value.** `ClassValueType<A>` (`class_value<...>`, printed `typeof A`) is what `typeof A` of a class
gives (`typeOfQueried`). It is laid out as a tuple of four opaque pointers
(`MLIRTypeHelper::getClassValueStorageType`), not as a vtable pointer alone:
- **`.vtable`:** the class's vtable. An abstract class has one too, though no instance carries it: its
  statics are read through it, and its abstract methods' and interfaces' slots are null.
- **`.new`:** the class's `.new`, which allocates an instance and sets its vtable; null for an abstract class.
- **`.constructor`:** the class's constructor, or null for none.
- **`.rtti`:** the class's rtti, which every class has and no two share.

A pair-or-more layout needs no new vtable slot, so no class's vtable changes (§2.4 is moot). `ts.Cast`
between the value and that tuple, or to the value of a base class, lowers to nothing.

**Making one** (`castToClassValue`, `mlirGenClassValue`):
- **Where:** a class name becomes a value only where a `typeof A` is expected - a parameter, a typed
  variable or field, an array element, a return. `const C = A` stays a compile-time alias.
- **Assignable:** `B` is a value of `typeof A` when it extends `A` and its constructor takes the same
  parameters (fewer is allowed).
- **Another module's class** is a value too. A shared library's (-shared) has its `.new`, constructor and
  rtti behind the pointers the importer resolves at load time, and its vtable is looked up in the library
  by name. A class without a constructor of its own, whose base is a library's, takes the base's through
  that pointer. Through the value, a library class's static field is read where the library's vtable slot
  points: the field itself, not the importer's pointer to it.
- **A @cpp class implemented here** (its methods have bodies, `isCppVTableImplementation`) is a value: its
  vtable (C++'s layout), `.new` and constructor are as any class's. It has no rtti, so its identity, the
  value's last part, is its vtable, which each such class has of its own. A @cpp class has no statics, so
  nothing but `new` and `==` is reached through its value. One whose objects C++ makes has no `.new` or
  constructor here, and can't be a value (an error).
- **Not yet:** a @value class can't be a value.

**Using one:**
- **`new c(args)`** (`NewClassInstanceFromClassValue`): the arguments are cast to `A`'s constructor's
  parameters, and `.new` is called through the value, its result marked owned and fresh. Then the
  constructor is called with the instance first, when there is one, as any call is made (spread
  arguments, a rest parameter, optional parameters). A value of an abstract class's type is an error.
- **`==` / `!=`:** compare the rtti; a class named beside a value is its value.
- **`typeof`** of a class value is `"function"`.
- **A static method** is called through the value's vtable (ADD_STATIC_MEMBERS_TO_VTABLE). A subclass's
  static method of the same name takes over the slot, so `c.label()` is the label of the class `c` holds.
- **A static field** is read through the vtable too. One that a subclass declares again, of the same
  type, takes over the base's slot (`ClassInfo::getVirtualTable`), so it is the field of the class the
  value holds; the compiler's own (`.rtti`, `.size`) keep a slot each.
- Any other member through a value is an error.

**Ownership:** the value owns nothing (its parts are static data and functions). The instance `new` makes
through it is a fresh owned result.

**Local classes (option C).** A method, constructor or accessor of a class declared in a function that reads
one of the function's locals is an error naming the variable, in place of "Expected 1 arguments, but got
0". So is a field initializer's: the discovery pass of a constructor, implicit or not, is given the field
initializers (`generatedStatements`) with its body, so what they read is found as the body's reads are.

**Tests:**
- **`00class_values`:** a factory, an array, a return, a field, a class without a constructor, a subclass
  without its own constructor, equality, abstract classes, `typeof`, static fields and methods through
  the value (a subclass's, an inherited one, an abstract class's own); under
  gc, rc, none and own.
- **`import_class_values`** + **`export_class_values`:** a library's classes, and one of this module extending
  one, as values both ways; linked, -shared and -jit -shared, under every memory model.
- **`00cpp_class_values`:** @cpp classes implemented here, made through a value and called through their
  C++ slots (one by hand, as C++ would), in an array, compared; under gc, rc, none and own.
- **`class_value_constructor_error`**, **`class_local_capture_error`**, **`class_field_capture_error`** and
  **`cpp_class_value_error`:** the errors.
