# `@final` classes

Date: 2026-10-19. Status: design, awaiting approval.

## Why

The default library does not build under `-mm=own`. After the `@nodrops` work (compiler branch
`own-nodrops-declarations`, DefaultLib branch `own-default-lib`) 14 errors remain in `src/lib.ts`.
Seven of them have one cause: a method call on an exported class goes through the vtable, and
another module could subclass that class and override the method. So own cannot resolve the call
(`OwnershipSignaturePass::resolveFamily` requires every family member to be private and defined in
this module), and an unresolved call:

- may destroy anything, so a borrow of a field cannot be lent to it. This affects the `String`
  wrapper's `match`, `replace`, `replaceAll`, `search` and `split`: each one hands `this.value` to a
  `___String` function that calls a `RegExp` method.
- returns a value that is not fresh (spec 2026-09-24 section on freshness: only a direct call of a
  function defined in this module is fresh), so it cannot be moved into a union or compared and
  then returned. This affects `Date[Symbol.toPrimitive]`'s `return this.toString()` (lib.ts
  432/435) and, in the debug build only, `___String.match`'s `if (result == null)` after
  `expr.exec(this)`.

The user's ruling: DefaultLib classes are final in practice, and nobody overrides them. A class
should be able to say so with a `@final` decorator.

## Goals

- `@final` on a class declaration: no class may extend it. This is enforced as a compile error, in
  the module that defines the class and in every module that sees its declaration.
- A method or accessor call on a value whose static type is a `@final` class is a direct call to
  the implementation that type has, not a vtable call. Own then resolves it with the rules it
  already has for direct calls: no-drops from the body, a fresh result, result-borrows facts, and
  stack promotion and borrow facts the same way.
- Mark the DefaultLib classes `@final` in `lib.ts` and in their `lib.d.ts` declarations: `RegExp`,
  `MatchResults`, `MatchIndicesResults`, `String`, `Number`, `Boolean`, `BigInt`, `Date`, `Headers`
  and `Response`. `ArrayBuffer` has no `lib.d.ts` declaration, so it is marked in `lib.ts` only.
  `console` and `Math` are `static` classes with no instances and stay unmarked.
- The seven errors listed above go away. The `Headers` constructor's `k + ""` copies, a workaround
  for the same cause, are reverted.

## Non-goals

- The other remaining errors are a separate decision ("an exported function cannot keep or borrow
  a parameter": `matchAll`, `String.iterator`, `Response.constructor`, and the mislocated
  `lib.ts:151` error, which comes from `matchAll`'s generated `next`). They stay.
- Own facts crossing a static link. A user program under own that calls `re.exec(s.value)` still
  sees the library's methods as declarations without facts. That is a follow-up project (the same
  mechanism as the borrow facts of 2026-10-18).
- `@final` on individual methods. It is class-level only.
- Adding `-mm=own` to DefaultLib's `build.bat` / `build.sh` loops. That waits until `lib.ts`
  compiles under own with no errors.

## Anti-goals

- Changing a program's behaviour under gc, rc or none. A direct call reaches the same function
  the vtable would have reached, because no subclass can exist. Only the dispatch changes.
- Dropping the vtable of a `@final` class. Interfaces the class implements, `instanceof`, and calls
  through a non-final base type still use it.
- Breaking the cross-module paths. A `@final` class from a DLL loaded at run time (`isDynamicImport`,
  no import library) must keep its vtable calls, because a direct call there would be a symbol that
  cannot be linked.

## Design

### Parsing and storage

- `#define FINAL_DECORATOR "final"` in `Defines.h`, beside `CPP_VTABLE`.
- `ClassInfo` gets a `bool isFinal` (in `MLIRGenStore.h`, default false). It is set in the class
  decorator loop in `MLIRGenClasses.cpp` (the `iterateDecorators(classDeclarationAST, ...)` block at
  about line 361), on declarations (`declare class`, `lib.d.ts`) as well as definitions.
- `DeclarationPrinter` prints `@final` before a final class, as it prints `@cpp` (about line 436),
  so `-shared` importers (`__decls`) see it.

### Enforcement

- Where a class's `extends` clause is resolved (the `heritageClauses` loops in
  `MLIRGenClasses.cpp`), a base class with `isFinal` is an error at the `extends` expression:
  `class 'X' cannot extend 'RegExp': it is @final`. It applies in every memory model.
- `@final` together with `abstract` is an error, since an abstract class must be extended.
  `@final` with `@cpp` is also an error: a C++ binding is dispatched through the C++ vtable.

### Dispatch

A method or accessor reference for a class instance is virtual at two sites:
`ClassMethodAccess` in `MLIRGenImpl.h` (about line 8045,
`if (methodInfo.isAbstract || methodInfo.isVirtual && !isStorageType)`, creating
`ThisVirtualSymbolRefOp` at about 8067) and `ClassAccessorAccess` in `MLIRGenAccessCall.cpp`
(about line 705). Each gains one exception: when the *receiver's* static type is a `@final`
class that is not `isDynamicImport`, it emits the direct `ThisSymbolRefOp` to `funcName`.

The `classInfo` these functions see is not always the receiver's. For an inherited member,
`ClassMembersAccess` recurses through `ClassBaseClassAccess` with the base's `ClassInfo`
(`MLIRGenAccessCall.cpp` about 947/1012). So the receiver's finality has to be passed down
through `ClassMembersAccess` -> `ClassBaseClassAccess` -> `ClassMethodAccess` /
`ClassAccessorAccess`, as a flag or as the receiver's `ClassInfo`. The decision then reads:

- A method the final class defines or overrides is called directly.
- A method it inherits from a non-final base without overriding it is called directly too, as
  `Base.m`. With no subclass, that is the only implementation a value of this static type can have.
- A call through a variable typed as the non-final base, or as an interface, is unchanged.

The index accessor (`ClassIndexAccess`, as in `MatchResults[i]`) already emits a direct
`ThisIndexAccessorOp`, and `super.` calls are already direct (their `this` is a storage type), so
neither changes. The builder should confirm that no other `VirtualSymbolRefOp` /
`ThisVirtualSymbolRefOp` creation applies to a class instance.

Direct calls name the method by symbol, so for user programs they also get past a known
`lib.d.ts` / `lib.ts` mismatch. `lib.d.ts`'s `RegExp` leaves out `split`, which `lib.ts` defines,
so a user program's vtable index for `[Symbol.dispose]` does not match the library's. Public
methods and accessors of exported classes are already in the import library, so the direct
calls link, compiled or JIT. Under rc, `OwnedReturnConsumptionPass` treats a direct call to a
`lib.d.ts` declaration as it treats a virtual one.

**Amended after the final review (2026-10-19).**

- **Direct calls only into this module.** A member is called directly only when the class that has it is defined in this module, not declared there (`isDeclaration`, `isImport`, `isDynamicImport`). A `@static import` links by name against the import library, which has no symbol for a member of a base the library does not export, so inheriting such a member broke the link. A user program's calls into the default library's `@final` classes therefore stay vtable calls. That costs nothing the goals need: own facts do not cross a static link anyway. The extends error still applies to declared classes.
- **No other class can pose as a `@final` one.** An unrelated class that fits a `@final` class member by member is not assignable to it (`type Y is not assignable to type F: 'F' is @final`), because the direct call would run F's body on Y's fields.

### Own

No change in the ownership passes is planned. A `ThisSymbolRef` call is what `resolveValue`
already resolves (`addDefined`, or `addImported` for an own library over a DLL import), and it
canonicalizes to the `SymbolCallInternal` that `isFresh` accepts. `noDrops` does not depend on a
function being open, so the public `RegExp.exec` called directly gets its body's facts.

**Amended 2026-10-24.** A call on an object of a `@final` class another module defines stays a vtable call (see "Direct calls only into this module"). MLIRGen marks its `ts.ThisVirtualSymbolRef` `__final_receiver`, and `resolveValue` resolves it as a call to the method it names, not to the method's family. An accessor's `ts.VirtualSymbolRef` is marked and resolved the same way. So `addImported` applies, and an own library's `__own_no_drops` covers the call.

The builder checks this first, on the three lib.ts shapes (`re.exec(s)` returning
`MatchResults | null`, `this.toString()` returned as `string | number`, and
`this.value.match(expr)`, whose callee destroys its own local `MatchResults` after `[...result]`).
If one still fails because of a real imprecision in a pass, fixing it properly is part of this
project, with a minimal repro, a fix and an own test, as the `@nodrops` branch did for four such
imprecisions. *(Builder's call, made by the designer: the user left scope open.)* A workaround
that only bends a pass for these shapes is not. If a remaining error turns out to be correct
(the code really is unsafe under own), it is reported, and the library change for it is decided
separately.

## Testing

- Compiler `test/tester`:
  - Positive own tests (`own/`, run AOT and JIT): a `@final` class with a virtual method whose
    result is returned through a union and compared with null, and a wrapper class that lends a
    field to a `@final` class's method. Each comes with a `churn()` of same-size objects (the
    use-after-free rule from the rc work).
  - Error cases: `extends` of a `@final` class, `@final abstract`, `@final @cpp`.
  - An IR check that a call on a `@final` static type emits no `ThisVirtualSymbolRef`, while one
    through its interface still does.
  - A `-shared` pair: an exporting module with a `@final` class and an importer that calls it and
    fails to extend it.
- The full suite stays green (4682/4682 on the base branch).
- DefaultLib:
  - Release builds of gc and rc pass `tests.ps1`, compiled and JIT.
  - `lib.ts` under own: the seven listed errors are gone (release and `--di`), and the remaining
    count is only the non-goal sites.

## Left to the builder

- The error message wording, and whether to give the user a fix-it hint.
- Whether `isFinal` also skips emitting `.instanceOf` overrides. It should not: `instanceof`
  still needs them.
- Branch layout. Stack this on `own-nodrops-declarations` (whose commit is still waiting on GPG
  signing), or rebase once that lands.
