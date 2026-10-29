# Generic methods in interfaces (and virtual generic class methods): design

Status: **option A implemented for interfaces, D for classes** (2026-10-29); see §7. Virtual dispatch of
generic class methods through a base class is not done.

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

## 7. What is implemented

**Interfaces (option A).**
- **The slot** (`getInterfaceMethodNameAndType`): a generic method's signature is generated with each type
  parameter bound to `any`, and `InterfaceMethodInfo::genericSignature` keeps the AST for call sites.
  A type parameter must be the whole type of a parameter or of the result: `T[]`, `[A, B]`, `A | B` or
  `Box<T>` erased are not laid out as their instantiations are, so they are an error at the declaration.
- **The implementer** (`findErasedGenericMethod`, `instantiateErasedGenericFunction`): when a class's
  vtable for the interface finds no plain method of the name, the class's generic method (or its
  nearest base's) is instantiated with `any` and fills the slot. Kept in `ClassInfo::erasedGenericMethods`,
  and made again when the symbol is not in the current module (the discovery pass drops its module).
  An object literal with an untyped arrow (`{ map: (x) => x }`) fills the slot as for any method.
- **The call** (`mlirGenCallExpression`, `instantiateErasedGenericResultType`): an interface symbol ref of a
  generic method is remembered (`erasedGenericCallees`); its call infers the type arguments from the
  arguments when none are given (the signature with each type parameter as a named generic type, matched
  by `inferType`), makes the call through the slot, and casts the `any` result to the instantiated
  result type - the checked `<T>anyValue` cast.
- **Cross-module**: an exported interface with a generic method is exported as its source
  (`addInterfaceSourceToExport`), as a generic interface is, so the importer has the generic signature
  and not only the erased slot.

**Classes (option D).** A generic method declared in a class whose base has a generic method of that name
is an error: a call through the base class would not reach it.

**Limits.**
- A body that depends on its type parameter's real type (`sizeof<T>()`) runs with `any` through an
  interface; it is not rejected.
- Under own, a class object unboxed from the `any` result is a borrow of the box: it can be read, not
  given on as another `any` argument. An object literal and a class implementing the same interface
  method that returns its parameter disagree on the family's result under own, as a non-generic
  `map(x: any): any` does.

**Tests.** `00interface_generic_method` (explicit and inferred type arguments, a string, a class object,
two implementers, an inherited generic method, a direct call), `own_interface_generic_method` (under own,
rc, gc and none, with churn), `export_interface_generic_method` / `import_interface_generic_method` (an
exported interface, class and function used from another module, plain and `-shared`).
