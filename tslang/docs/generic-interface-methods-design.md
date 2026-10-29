# Generic methods in interfaces (and virtual generic class methods): design

Status: **proposed, not implemented**. Written 2026-10-29, to be picked up later.

## 1. What fails today

```ts
interface Mapper { map<T>(x: T): T; }
class Id implements Mapper { map<T>(x: T): T { return x; } }
const m: Mapper = new Id();
m.map<number>(5);
```

- **The interface:** compiling it fails with "Generic method 'map' in the interface is not allowed". `getInterfaceMethodNameAndType` (`lib/TypeScript/MLIRGenImpl.h`) refuses any method signature with type parameters.
- **A class hierarchy, worse because it's silent:**

```ts
class Id { map<T>(x: T): T { return x; } }
class Twice extends Id { map<T>(x: T): T { print("twice"); return x; } }
const a: Id = new Twice();
a.map<number>(5);   // calls Id.map: "twice" is never printed
```

A generic class method is not virtual. Each call instantiates the method of the *static* type's class (`staticGenericMethods`, `MLIRGenClasses.cpp`), so an override in a subclass is never reached through a base reference. No error or warning says so.

## 2. Why it is not simply a vtable slot

A vtable slot holds one function pointer. A generic method has no single function, only one instantiation per type argument, and each is produced where a call names it (monomorphization).

- **Through an interface or a base class:** the caller does not know which class implements the method, so it cannot instantiate the right body.
- **In the implementer:** the class does not know which type arguments callers will use (other modules included), so it cannot fill a slot per instantiation ahead of time.

## 3. Options

**A. Erased slot.** The interface (or base-class) vtable gets one slot per generic method, holding an instantiation in which each type parameter is `any`.
- **At the call site:** box the arguments whose type is a type parameter, call through the slot, and unbox the result to the call's type argument. The unbox is checked, as `<T>anyValue` is.
- **In the implementer:** the class provides the `any` instantiation of its own method for the slot. Direct calls on a known class keep the monomorphized fast path they use today.
- **Pros:** works across modules and with separately compiled implementers.
- **Cons:** boxing costs, and the body runs on `any`. A body that relies on the type parameter's real type (`sizeof<T>()`, `new T()` through a constraint, arithmetic on `T extends number`) changes behaviour in the erased form. Such methods would have to be rejected for virtual dispatch with a clear error.

**B. Closed-world instantiation table.** Collect every (method, type argument) pair called through an interface or base type in the program. Give each pair a slot, appended to every implementer's vtable.
- **Pros:** no boxing; the bodies run at their real types.
- **Cons:** needs whole-program knowledge. It breaks for `-shared` libraries and the JIT cache (modules compiled apart), and the slot layout changes as programs grow. Unfit for this compiler's cross-module model (`docs/interface-vtable-simplification-design.md`).

**C. Dictionary passing.** Instantiate a generic method once, with a hidden argument that carries type information (size, descriptor, retain/release routines). This is what Swift does for unspecialized generics.
- **Pros:** no boxing of values; one body per method.
- **Cons:** a large change. Every generic body would need to lower against a runtime type description instead of concrete MLIR types, and the ownership passes (`-mm=own`, `-mm=rc`) reason about concrete types throughout.

**D. Keep rejecting, but say so.** Report an error for a generic method overridden in a subclass, instead of silently calling the base method. Interfaces keep their current error.

## 4. Recommendation

1. **Now:** do D for classes. It turns a silent wrong call into an error. That's cheap, and correct regardless of what follows.
2. **Then:** A for interfaces and virtual class methods, limited to methods whose bodies the erased form can run unchanged:
   - type parameters used only as values: stored, passed on, returned;
   - no `sizeof<T>`, no `new` through a constraint, no numeric operators on `T`.

   A method outside that set gets a clear error at its declaration.
3. **C** is the long-term general answer, but it is a project of its own.

## 5. Work involved for A

- **Interface declaration:** for each generic method, a slot whose function type has every type parameter replaced by `any` (`getInterfaceMethodNameAndType`, `InterfaceInfo::methods`).
- **Implementer:** when a class's vtable is built for the interface (`getVirtualTable`, `mlirGenObjectVirtualTableDefinitionForInterface`), instantiate the class method with `any` for each type parameter. Then put that symbol in the slot.
- **Call site:** `m.map<number>(5)` through an interface or a base type becomes three steps: cast the arguments to `any`, make an indirect call through the slot, and cast the result to `number` (`castFromAny`, checked).
- **Base classes:** the class vtable gains the same erased slot for each generic method, and an override fills it. A call through a base-class reference uses it; a call on the exact class keeps today's direct instantiation.
- **Ownership under -mm=own:** a boxed argument is a fresh `any` the callee borrows. The result is unboxed into a value the caller owns. Both are existing patterns (spec 59, casts to `any`).
- **Tests:**
  - an interface call with two type arguments;
  - a subclass override reached through a base reference;
  - a cross-module implementer (`-shared`);
  - the rejection of a body that uses `sizeof<T>`;
  - each case under gc, rc, none and own.

## 6. Non-goals

- Generic *interfaces* (`interface Box<T>`) already work: they are instantiated per type argument. This document is only about generic *methods*.
- Making generic class methods virtual where nothing overrides them: a direct instantiation stays faster.
