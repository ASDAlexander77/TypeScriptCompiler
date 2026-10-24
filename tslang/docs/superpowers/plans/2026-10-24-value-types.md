# `@value` Classes and Interfaces Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A `@value` class or interface is a value of its readonly fields, not a heap object, so the ray tracer's `Vector`/`Color`/`Ray`/`Intersection` temporaries stop allocating, under every memory model, with the benchmark's object literals unchanged.

**Architecture:** One new dialect type, `!ts.value<@Name, !ts.tuple<...>>` (`mlir_ts::ValueType`), names a `@value` declaration and carries its fields as a tuple. It lowers to that tuple's LLVM struct. MLIRGen reaches the fields through a `ts.Cast` to the storage tuple, which lowers to nothing, so the existing tuple ops and lowerings do the work. Every type walker sees a `ValueType` as its storage tuple through one helper. The rules that make a value invisible as a copy (readonly, no `===`, no null in a plain slot, no identity conversions) are MLIRGen errors keyed on `ValueType`.

**Tech Stack:** C++17, MLIR (ts dialect, TableGen types), CMake/CTest, gtest (`unittests/MLIRGen`), the tslang test-runner, PowerShell (`bench.ps1`).

**Spec:** `tslang/docs/superpowers/specs/2026-10-24-value-types-design.md`

## Global Constraints

- Branch `value-types`, stacked on `process-heap-malloc-attrs` (PR #683). The design commit is a98a9b7e. When #683 merges, merge `origin/main` into `value-types`. Never rebase, never force-push.
- The sources use CRLF. Edit them with the Edit tool or a Python script that writes `newline=''`. Never use `sed -i`, which rewrites the file to LF.
- Never bypass commit signing. On `gpg: signing failed: Timeout`: retry, run `gpg-connect-agent "scd killscd" /bye` and retry, then leave the change staged and hand the exact command to the user.
- Build: `cmake --build I:/TypeScriptCompiler/__build/tslang/windows-msbuild-2026-release --config Release --target tslang --parallel 8`. Reconfigure (`cmake .` in that folder) after adding tests. Tests: `ctest -C Release -j 16 --timeout 300` there. Without `-C Release` nothing runs. Full suite before this work: 5525 passed.
- Code without `@value` compiles exactly as before. That includes plain tuples and type literals, so the `return null` tuple bug in the spec's "Found on the way" stays as it is.
- `@value` is never a hint. A marked type is always a value, and anything that can't be compiled as one is an error naming the reason.
- Error texts, pinned by the tests (`X`, `V`, `R`, `f`, `T` stand for the names in each case):
  - `class 'V' cannot be both @value and @cpp` (also `abstract`, `@final`)
  - `@value class 'V' cannot extend 'B'`
  - `class 'X' cannot extend 'V': it is @value`
  - `@value class 'V' cannot implement an interface`
  - `@value class 'V' cannot have type parameters`
  - `field 'f' of @value class 'V' must be readonly`
  - `@value class 'V' cannot have a setter`
  - `field 'f' of @value interface 'R' must be readonly`
  - `field 'f' of @value interface 'R' cannot be optional`
  - `@value interface 'R' can only have fields`
  - `@value interface 'R' cannot extend 'I': it is not @value`
  - `instanceof can't be used with the @value class 'V': its values have no type information`
  - `a @value type has no identity; compare its fields`
  - `a @value type has no null; write 'T | null'`
  - `the @value class 'V' cannot be converted to 'T'`
- Benchmark checksum: 38837369, under gc, rc, none and own.

## Review Focus

- A closure that captures a `@value` local reads the variable, not a snapshot. `let v = new V(1); const f = () => v.x; v = new V(2);` then `f() == 2`, as in TypeScript, because the capture is of the binding. Pinned in Task 5's `value_refs.ts`.
- A `@value` value passed through a generic (`[a, b].reduce((acc, x) => V.plus(acc, x), V.zero)`, the benchmark's `lights.reduce` shape) is copied correctly. Pinned in Task 5.
- A literal whose fields are written in a different order from the interface, nested two levels (`{ ray: { dir: d, start: s }, dist: 1, thing: t }`), lands each field in the right slot. Pinned in Task 2.
- A `@value` value stored in a field of an ordinary class (`Camera.pos`) and then read after the class's other fields change keeps its own fields. Pinned in Task 3.
- An optional parameter of a `@value` type (`f(v?: V)`), called with no argument and with one, and `x?.dist` on `T | null`. Pinned in Task 4.

---

### Task 1: `@value` is parsed, stored, enforced and printed

**Files:**
- Modify: `tslang/include/TypeScript/Defines.h` (beside `#define FINAL_DECORATOR "final"`, about line 327)
- Modify: `tslang/include/TypeScript/MLIRLogic/MLIRGenStore.h` (`InterfaceInfo` about 283-315, `ClassInfo` about 753-820, and their initializer lists)
- Modify: `tslang/lib/TypeScript/MLIRGenClasses.cpp` (decorator loop about 382-420; the combination checks beside the `@final` ones about line 101; the `ExtendsKeyword`/`ImplementsKeyword` heritage branches)
- Modify: `tslang/lib/TypeScript/MLIRGenInterfaces.cpp` (`mlirGenInterfaceInfo` about 1118-1160, and the interface member and heritage loops)
- Modify: `tslang/lib/TypeScript/DeclarationPrinter.cpp` (class printer about 484-491, and the interface printer)
- Test: `tslang/unittests/MLIRGen/DeclarationPrinter.cpp`
- Test: `tslang/test/tester/value/*.ts` (one file per error case below)
- Modify: `tslang/test/tester/CMakeLists.txt` (a `value_error_cases` block after `final_error_cases`, about line 3040, in the same shape)

**Interfaces:**
- Produces:
  - `#define VALUE_DECORATOR "value"`
  - `bool ClassInfo::isValue` and `bool InterfaceInfo::isValue`, default `false`, set for definitions and `declare` declarations alike
  - `DeclarationPrinter` writes `@value` on its own line before a `@value` class or interface

- [ ] **Step 1: Write the failing printer unit tests.** Next to `final_class_prints_final_decorator`, add:
  - `value_class_prints_value_decorator` (`ci->isValue = true`; `HasSubstr("@value")`)
  - `value_interface_prints_value_decorator` (the same for an `InterfaceInfo`)
  - `plain_interface_has_no_value_decorator` (`Not(HasSubstr("@value"))`)
- [ ] **Step 2: Write the error-case files** in `test/tester/value/`. Each has one offence and nothing else that fails:
  - `value_cpp.ts` (`@value @cpp declare class V { m(): int; }`)
  - `value_abstract.ts`
  - `value_final.ts`
  - `value_extends.ts` (`class B {}` and `@value class V extends B {}`)
  - `extends_value.ts` (`@value class V { constructor(public readonly x: number) {} }` and `class X extends V {}`)
  - `value_implements.ts`
  - `value_generic.ts`
  - `value_iface_extends.ts` (`interface I { readonly a: number; }` and `@value interface R extends I { readonly b: number; }`)
- [ ] **Step 3: Register them** as `value_error_cases` `"name|message"` entries with the messages from Global Constraints. Each runs `tslang --emit=obj --no-default-lib -mm=none` and has `PASS_REGULAR_EXPRESSION` set to the message and `FAIL_REGULAR_EXPRESSION "Stack dump|Assertion failed"`. Test names are `test-compile-value-<name>`.
- [ ] **Step 4: Confirm they fail.** Reconfigure and build `tslang` and `MLIRTypeScriptUnitTests`, then run `ctest -C Release -R "value-|unittest-MLIRGen"`. Expected: the new tests fail.
- [ ] **Step 5: Implement.**
  - Read the decorator into `isValue` in both loops. Interfaces get their own `iterateDecorators` call.
  - Emit the combination and heritage errors at the declaration (or `extends`/`implements`) location and fail the declaration, as `@final`'s do.
  - `extends_value` is checked where `@final`'s `extends` check is.
  - Print the decorator.
- [ ] **Step 6: Run the tests.** `ctest -C Release -R "value-|final-|unittest-MLIRGen"` should all pass. Then the full suite: 0 failed.
- [ ] **Step 7: Commit.** `@value: parse, store, reject invalid combinations, print`.

### Task 2: `ValueType`, and `@value` interfaces

**Files:**
- Modify: `tslang/include/TypeScript/TypeScriptTypes.td` (a `TypeScript_Value` def after `TypeScript_Class`, about line 682)
- Modify: `tslang/include/TypeScript/MLIRLogic/MLIRTypeHelper.h`:
  - the new helper
  - `isValueType` (about 206)
  - `ownsHeapMemory` (body about 3952)
  - `getOwnershipFieldTypes` (about 2271)
  - type equality and assignability helpers that switch on `TupleType`
- Modify: `tslang/lib/TypeScript/LowerToLLVM.cpp` (the `TupleType` conversion at about 8173; `CastOp` lowering for `ValueType` to and from its storage tuple)
- Modify: `tslang/include/TypeScript/LowerToLLVM/OwnershipRoutineLogic.h` (`buildRetainBody` about 1309, `buildBody` about 1109)
- Modify: `tslang/lib/TypeScript/StackEscapeFacts.h` (where it classifies tuple values)
- Modify: `tslang/lib/TypeScript/MLIRGenVariables.cpp` (`resolveTypeByNameInNamespace`, interface branch at about 1316)
- Modify: `tslang/lib/TypeScript/MLIRGenExpressions.cpp` (`mlirGen(ObjectLiteralExpression)` about 1891-2124)
- Modify: `tslang/lib/TypeScript/MLIRGenAccessCall.cpp` (the member-access `TypeSwitch` at about 203-384: a `Case<mlir_ts::ValueType>`)
- Modify: `tslang/lib/TypeScript/MLIRGenCast.cpp` (`cast` about 691, `castToInterfaceSpecialCases` about 1010)
- Modify: `tslang/lib/TypeScript/MLIRGenImpl.h` (the binary-operator path for `===`/`!==`/`==`/`!=`)
- Test: `tslang/test/tester/value/value_interface.ts`
- Test: `tslang/test/tester/value/value_no_alloc.ts`, `tslang/test/tester/value-no-alloc.cmake`
- Test: error files `value_iface_mutable.ts`, `value_iface_optional.ts`, `value_iface_method.ts`, `value_iface_identity.ts`
- Modify: `tslang/test/tester/CMakeLists.txt`

**Interfaces:**
- Consumes: `InterfaceInfo::isValue` (Task 1).
- Produces:
  - `mlir_ts::ValueType`, mnemonic `value`, parameters `(FlatSymbolRefAttr name, Type storageType)`, builder `ValueType::get(name, storageType)`. `storageType` is always a `mlir_ts::TupleType`.
  - `mlir::Type MLIRTypeHelper::stripValueType(mlir::Type)` returns the storage tuple of a `ValueType` and anything else unchanged. Every type walker that must treat a value as its fields calls it.
  - `mlir_ts::ValueType MLIRGenImpl::getValueTypeOfInterface(InterfaceInfo::TypePtr)` gives the fields in declaration order.
  - `mlir::Value MLIRGenImpl::valueStorage(mlir::Location, mlir::Value)` emits the `ts.Cast` from a `ValueType` value to its tuple.
  - In the IR, `ts.Cast` between a `ValueType` and its exact storage tuple, in either direction, lowers to nothing.

- [ ] **Step 1: Write `value_interface.ts`.** It runs, asserts, and ends with `print("done.")`, like the `final/` tests:
  - `class Vec { constructor(public x: number, public y: number) {} }`, `interface Thing { id(): int; }`, `class T1 implements Thing { id(): int { return 7; } }`
  - `@value interface Ray { readonly start: Vec; readonly dir: Vec; }`
  - `@value interface Hit { readonly thing: Thing; readonly ray: Ray; readonly dist: number; }`
  - `function mk(s: Vec, d: Vec): Ray { return { start: s, dir: d }; }`
  - `function hit(t: Thing, r: Ray, d: number): Hit { return { dist: d, ray: r, thing: t }; }` (fields out of order)
  - `function len(r: Ray) { return r.start.x + r.dir.y; }`
  - `interface HasStart { start: Vec; }`

  Asserts:
  - `len({ start: new Vec(1, 2), dir: new Vec(3, 4) }) == 5`
  - `const h = hit(new T1(), mk(new Vec(1, 0), new Vec(0, 2)), 9); h.dist == 9 && h.thing.id() == 7 && h.ray.dir.y == 2`
  - the nested literal `{ ray: { dir: new Vec(0, 3), start: new Vec(5, 0) }, dist: 1, thing: new T1() }` typed `Hit`: `.ray.start.x == 5 && .ray.dir.y == 3`
  - `const a: any = mk(new Vec(8, 0), new Vec(0, 0)); a.start.x == 8`
  - `const s: HasStart = mk(new Vec(6, 0), new Vec(0, 0)); s.start.x == 6`
  - `const tl: { start: Vec; dir: Vec } = mk(...)` and back to `Ray`: fields equal
  - `typeof mk(new Vec(0, 0), new Vec(0, 0)) == "object"`
  - a loop of 10000 `len(mk(...))` summing to an exact total
- [ ] **Step 2: Write the no-allocation IR check.**
  - `value_no_alloc.ts` declares `@value interface Vec { readonly x: number; readonly y: number; }` and a `@value interface Ray` of two `Vec`s. It exports `function len2(a: number, b: number): number`, which builds `{ start: { x: a, y: 0 }, dir: { x: 0, y: b } }`, passes it to a non-exported `len` and returns the result.
  - `value-no-alloc.cmake` is modelled on `fresh-objects-folded.cmake`. For each of `gc rc none own`, it compiles with `--emit=llvm --opt --opt_level=3 --no-default-lib` and fails if the body of `@len2(` matches `[mc]alloc|GC_malloc|__tslang_heap`. The pattern deliberately doesn't match `alloca`: a stack slot is fine. `malloc`/`calloc` still catch Linux, where ProcessHeapPass doesn't rename them.
- [ ] **Step 3: Write the error files.**
  - `value_iface_mutable.ts`: a field without `readonly`.
  - `value_iface_optional.ts`: `readonly f?: number`.
  - `value_iface_method.ts`: `m(): int;`.
  - `value_iface_identity.ts`: `r1 === r2` on two `Ray` values.
- [ ] **Step 4: Register.**
  - `value_interface` runs AOT and JIT in `foreach(value_mm gc rc none own)`, as `final_dispatch` does: `test-compile-value-interface-${mm}`, `test-jit-value-interface-${mm}`.
  - `test-value-no-alloc` uses `add_test(... -P value-no-alloc.cmake)`, as `test-fresh-objects-folded` does.
  - The four error files are added to `value_error_cases`.
- [ ] **Step 5: Confirm they fail.** `ctest -C Release -R "value-"`. Expected: the new tests fail (no `@value` interface semantics yet).
- [ ] **Step 6: Implement `ValueType`.** Add the TableGen def and the `stripValueType` helper, and lower `ValueType` to the converted storage tuple. Then sort every `mlir_ts::TupleType` hit in `MLIRTypeHelper.h`, `LowerToLLVM/*.h`, `StackEscapeFacts.h`, the `Ownership*Pass.cpp` files and `MLIRGen*.cpp` into two lists:
  - **Walkers that descend into fields: apply `stripValueType`.** These are `isValueType`, `ownsHeapMemory`, `getOwnershipFieldTypes`, `buildRetainBody`/`buildBody`, `StackEscapeFacts.h`, type equality and assignability, and the LLVM type converter. Any other hit that reads the fields' types or sizes belongs here too.
  - **Sites that key on "this is a tuple" to box or re-represent it: do NOT apply.** A `ValueType` must not reach them. They are:
    - the tuple branches of `castToInterfaceSpecialCases` (`MLIRGenCast.cpp` about 1068-1076, into `castTupleToInterface`);
    - `NewClassInstance`'s `isValueType` → `ValueRefType` heap box (`MLIRGenAccessCall.cpp` about 2388-2391);
    - the object literal's `BoxAsObject` path;
    - `isNullAbsorbingType`;
    - the tuple-is-a-value-type branch of `getUnionTypeMergeTypes` (about 3456-3460) that makes `UnionType{tuple, null}`.

    A `ValueType` reaches the boxing paths only through an explicit `valueStorage` cast at a conversion the spec allows (Step 7).
- [ ] **Step 7: Implement `@value` interfaces in MLIRGen.**
  - The interface name resolves to `getValueTypeOfInterface`.
  - Declaration rules give the three member errors.
  - An object literal whose contextual type is a `ValueType` is built as its storage tuple (the existing receiver-order path) and cast to the `ValueType`. There is no `castTupleToInterface`.
  - Member access on a `ValueType` goes to `cl.Tuple` on `valueStorage(...)`.
  - `cast` converts:
    - between a `ValueType` interface and a tuple, type literal or another `ValueType` interface with the same field names, by field copy in the target's order;
    - a `ValueType` interface to `any`, `Object` or a non-`@value` interface by casting it to its storage tuple and taking the existing tuple paths.
  - `===`/`!==`/`==`/`!=` with a `ValueType` on both sides is the identity error. With `null`/`undefined` on one side it is left to Task 4.
- [ ] **Step 8: Run.** `ctest -C Release -R "value-"` should all pass. Then the full suite: 0 failed (no change without `@value`).
- [ ] **Step 9: Commit.** `@value interfaces: a literal of a @value interface type is a value`.

### Task 3: `@value` classes

**Files:**
- Modify: `tslang/lib/TypeScript/MLIRGenClasses.cpp` (class storage creation: no vtable field and no `.instanceOf`/`..new`/vtable global for a `@value` class; field and accessor rules)
- Modify: `tslang/lib/TypeScript/MLIRGenVariables.cpp` (`resolveTypeByNameInNamespace`, class branch at about 1297)
- Modify: `tslang/lib/TypeScript/MLIRGenAccessCall.cpp`:
  - `NewClassInstance` about 2377, `NewClassInstanceLogicAsOp` about 2503
  - the `ValueType` case added in Task 2: method and getter lookup through the class
- Modify: `tslang/lib/TypeScript/MLIRGenImpl.h` (`instanceof`, about 6226; `this` in methods and constructors of a `@value` class)
- Modify: where a method's function type and its `this` parameter are built (member generation in `MLIRGenClasses.cpp`, and `MLIRGenFunctions.cpp`). For instance members of a `@value` class only, `this` is the `ValueType` rather than `classInfo->classType`; statics are unchanged.
- Modify: `tslang/lib/TypeScript/MLIRGenCast.cpp` (conversion errors)
- Test: `tslang/test/tester/value/value_class.ts`
- Test: error files `value_mutable_field.ts`, `value_setter.ts`, `value_instanceof.ts`, `value_to_any.ts`, `value_identity.ts`
- Modify: `tslang/test/tester/CMakeLists.txt`

**Interfaces:**
- Consumes: `ValueType`, `stripValueType`, `valueStorage`, the `ValueType` member-access case and the identity error (Task 2); `ClassInfo::isValue` (Task 1).
- Produces:
  - `mlir_ts::ValueType MLIRGenImpl::getValueTypeOfClass(ClassInfo::TypePtr)`, with instance fields (parameter properties included) in declaration order.
  - The class's name used as a type resolves to it. Used as a value (`V.plus`, `V.white`), it still resolves to the class, so static access is unchanged.
  - Instance methods and getters take `this` as the class's `ValueType`. The plan picks by value (the spec leaves value or pointer-to-copy to the builder); LLVM's ABI lowering decides registers or memory.
  - In the constructor, `this` is a `!ts.ref<storage tuple>` to a stack slot. `new V(...)` makes the slot, calls the constructor and loads the value.

- [ ] **Step 1: Write `value_class.ts`.** It runs, asserts and prints `done.`:
  - `@value class V { constructor(public readonly x: number, public readonly y: number, public readonly z: number) {} static plus(a: V, b: V) { return new V(a.x + b.x, a.y + b.y, a.z + b.z); } static zero = new V(0, 0, 0); dot(o: V) { return this.x * o.x + this.y * o.y + this.z * o.z; } get len2() { return this.dot(this); } }`
  - `@value class P { readonly a: number; readonly b: number; constructor(a: number) { this.a = a; this.b = a * 2; } }` (fields assigned in the constructor)
  - `class Cam { pos: V; n = 0; constructor(p: V) { this.pos = p; } }`

  Asserts:
  - `V.plus(new V(1, 2, 3), new V(4, 5, 6)).y == 7`
  - `new V(1, 2, 2).len2 == 9`
  - `V.zero.x == 0`
  - `new P(3).b == 6`
  - `const c = new Cam(new V(1, 1, 1)); c.n = 5; c.pos = V.plus(c.pos, c.pos); c.pos.x == 2`
  - `const arr: V[] = [new V(1, 0, 0), new V(2, 0, 0)]; arr[1].x == 2`
  - `let v = new V(1, 0, 0); const w = v; v = new V(9, 0, 0); w.x == 1`
  - a 10000-step loop `acc = V.plus(acc, new V(1, 0, 0))` ending at `acc.x == 10000`
- [ ] **Step 2: Write the error files.**
  - `value_mutable_field.ts`: `@value class V { x: number = 0; }`.
  - `value_setter.ts`.
  - `value_instanceof.ts`.
  - `value_to_any.ts`: `const a: any = new V(...)`. Expect `the @value class 'V' cannot be converted to 'any'`.
  - `value_identity.ts`: `new V(...) === new V(...)`.
- [ ] **Step 3: Register.**
  - `value_class` runs AOT and JIT under `gc rc none own`, as in Task 2.
  - The error files are added to `value_error_cases`.
  - Extend `value_no_alloc.ts` with `export function plus2(a: number): number { return V.plus(new V(a, 0, 0), new V(a, 0, 0)).x; }`, using a `@value class V`.
  - `value-no-alloc.cmake` also checks `@plus2(`.
- [ ] **Step 4: Confirm they fail.** `ctest -C Release -R "value-"`.
- [ ] **Step 5: Implement.**
  - The `@value` class type is `getValueTypeOfClass`.
  - Class creation:
    - skips the vtable, `.instanceOf`, `..new` and the RTTI global;
    - applies the readonly-field and setter errors.
  - `new` and constructors follow the Interfaces block. In `NewClassInstance`, the `ValueType` branch (stack slot, constructor, load) comes before the existing `isValueType` check, which would otherwise box the value as a heap `ValueRefType`. `value-no-alloc` catches a box.
  - Method and getter calls on a `ValueType` of a class are direct calls with the value as `this`.
  - The conversion error covers `any`, `unknown`, `Object`, unions with non-value members and non-`@value` interfaces.
  - `instanceof` with a `@value` class on the right is the error.
- [ ] **Step 6: Run.** `ctest -C Release -R "value-"` should all pass, then the full suite: 0 failed.
- [ ] **Step 7: Commit.** `@value classes: new makes a value, methods take this by value`.

### Task 4: missing values

**Files:**
- Modify: `tslang/include/TypeScript/MLIRLogic/MLIRTypeHelper.h` (`getUnionTypeMergeTypes` about 3445-3515). `ValueType | null` becomes `OptionalType<ValueType>`, through a rule placed before the tuple-is-a-value-type branch (about 3456-3460). `ValueType | undefined` already becomes an optional the way a tuple's does.
  - Probed on main under `// @strict-null false`: for `type R = { a: number }`, `R | null` stays `!ts.union<!ts.null, !ts.tuple<...>>` all the way to MLIR, and `R | undefined` is `!ts.optional<!ts.tuple<...>>`. Both compare correctly. So the pragma does not strip the union, and the null rule belongs in `getUnionTypeMergeTypes`.
- Modify: `tslang/lib/TypeScript/MLIRGenCast.cpp` (`null`/`undefined` into a plain `ValueType` is the error; into `OptionalType<ValueType>` it is the empty optional)
- Modify: `tslang/include/TypeScript/LowerToLLVM/OptionalLogicHelper.h` only if a comparison shape is missing
- Test: `tslang/test/tester/value/value_missing.ts`, error file `value_null_plain.ts`
- Modify: `tslang/test/tester/CMakeLists.txt`

**Interfaces:**
- Consumes: Tasks 2-3.
- Produces:
  - `T | null` and `T | undefined` for a `ValueType` `T` are `OptionalType<T>`, and so is a `f?: T` parameter or field.
  - `T | null | undefined` is a `UnionType` of `T`, `null` and `undefined` (the existing tagged union). So `x === undefined` is false when `x` holds `null`. This is the plan's call on the spec's open encoding: it reuses what exists.

- [ ] **Step 1: Write `value_missing.ts`.** It has `// @strict-null false` at the top, as the benchmark does, and:
  - `@value interface Hit { readonly dist: number; }`
  - `function find(d: number): Hit | null { if (d < 0) return null; return { dist: d }; }`
  - `function first(xs: number[]): Hit | undefined { let best: Hit | undefined = undefined; for (const x of xs) { const h = find(x); if (h != null && (best === undefined || h.dist < best.dist)) best = h; } return best; }`
  - `function opt(v?: Hit) { return v === undefined ? -1 : v.dist; }`

  Asserts:
  - `find(-1) === null`, `find(-1) != null` is false, `find(2).dist == 2`
  - `first([3, -1, 2]).dist == 2`, `first([-1]) === undefined`
  - `find(5)?.dist == 5`, `find(-5)?.dist === undefined`
  - `opt() == -1`, `opt({ dist: 4 }) == 4`
  - `let u: Hit | null | undefined = null; u === undefined` is false and `u === null` is true
- [ ] **Step 2: Write `value_null_plain.ts`:** `@value interface Hit { readonly dist: number; } function f(): Hit { return null; }`. Expect `a @value type has no null; write 'Hit | null'`. Add it to `value_error_cases`.
- [ ] **Step 3: Register** `value_missing` under `gc rc none own`, AOT and JIT. Confirm it fails.
- [ ] **Step 4: Implement** the union merge, the casts and the error. Check that `x.dist` on an `OptionalType<ValueType>` after a `!= null` test reads the field (the `OptionalType` member-access case at `MLIRGenAccessCall.cpp` about 371).
- [ ] **Step 5: Run.** `ctest -C Release -R "value-"` should all pass, then the full suite: 0 failed.
- [ ] **Step 6: Commit.** `@value: T | null is an optional value; null into a plain @value type is an error`.

### Task 5: values holding references, generics and closures (gc, rc, none)

**Files:**
- Modify only where the tests fail: the ownership routines and walkers touched in Task 2, and capture/closure code in MLIRGen (`mlirGenRetainCaptured`, capture analysis)
- Test: `tslang/test/tester/value/value_refs.ts`
- Modify: `tslang/test/tester/CMakeLists.txt`

**Interfaces:**
- Consumes: Tasks 2-4.

- [ ] **Step 1: Write `value_refs.ts`.** It runs, asserts and prints `done.`:
  - `class Node { constructor(public n: int) {} }`
  - `@value interface Rec { readonly node: Node; readonly w: number; }`
  - `@value class V { constructor(public readonly x: number) {} static plus(a: V, b: V) { return new V(a.x + b.x); } }`
  - `function churn() { let s = 0; for (let i = 0; i < 64; i++) { const t = new Node(i); s += t.n; } return s; }` (same-size objects reuse freed blocks: the rc use-after-free rule)

  Asserts:
  - Build 1000 `Rec`s from fresh `Node`s into an array. Copy each into a local twice, drop the array's reference by overwriting the slot with a new `Rec`, `churn()`, then read `copy.node.n`. Sum the reads and assert the exact total.
  - `[new V(1), new V(2), new V(3)].reduce((acc, x) => V.plus(acc, x), new V(0)).x == 6`
  - `let v = new V(1); const f = () => v.x; v = new V(2); f() == 2` (the capture is of the binding)
  - A closure that captures a `Rec` and is called after the original's `Node` was overwritten elsewhere still reads its own `node.n`.
- [ ] **Step 2: Register** under `gc rc none` (own is Task 6), AOT and JIT, and also run with `--verify-ownership` under rc, as the corpus verifier does. Run it.
- [ ] **Step 3: Fix what fails.**
  - A retain or release missing for a reference field of a value goes in the routine that handles tuples, through `stripValueType`, not a value-only path.
  - A closure that copies instead of capturing the binding is a capture-analysis bug: the capture must box the variable as it does for any other local.
- [ ] **Step 4: Run.** `ctest -C Release -R "value-"` should all pass, then the full suite: 0 failed.
- [ ] **Step 5: Commit.** `@value: values holding references, through generics and closures`.

### Task 6: own

**Files:**
- Test: `tslang/test/tester/own/own_value_record.ts`, `tslang/test/tester/own/own_value_loop.ts`
- Modify: `tslang/test/tester/CMakeLists.txt` (append to the `foreach(own_test ...)` list at about line 3613; add `value_refs` under own)
- Modify, only as Step 3 finds: `tslang/lib/TypeScript/OwnershipInferencePass.cpp`, `OwnershipSignaturePass.cpp`, `OwnedReturnConsumptionPass.cpp`, `OwnershipFacts.h`

**Interfaces:**
- Consumes: Tasks 2-5. Background: own spec 32 (borrowing records), #679 (a tuple of owned values is a value), rulings 10 and 32 in `.superpowers/own-raytrace-rulings.md`.

- [ ] **Step 1: Write the own tests.** These are the benchmark's shapes, each with `churn()` and exact totals.
  - `own_value_record.ts`: `interface Thing { hit(r: Ray): Hit | null; id(): int; }`, `@value interface Ray { ... }`, `@value interface Hit { readonly thing: Thing; readonly ray: Ray; readonly dist: number; }`. `class S implements Thing { hit(r: Ray): Hit | null { return r.dir.x > 0 ? { thing: this, ray: r, dist: r.dir.x } : null; } }`. The caller reads `h.thing.id()` and `h.dist` through the `Thing` interface.
  - `own_value_loop.ts`: the `intersections` loop. `let closest: Hit | undefined = undefined;` over an array of `Thing`s, reassigned from `t.hit(ray)` inside the loop when nearer. Then the result is used after the loop, and passed to a function that reads `thing` and calls through it.
- [ ] **Step 2: Run them** with `test-runner -mm=own` and `-jit -mm=own`. Also run `value_interface`, `value_class`, `value_missing` and `value_refs` under own.
- [ ] **Step 3: Handle each rejection or crash.** Reduce it to a minimal `--no-default-lib` repro, find the cause in the pass, and decide:
  - **Real imprecision:** fix it properly, and add a positive own test, plus an `own_err_*` case if the fix narrows an error.
  - **Correct error** (the code really is unsafe): stop and report it with the repro. The spec says the benchmark change for it is decided separately.

  Never special-case `ValueType` inside an ownership pass to get a shape through. A value is a tuple to the passes.
- [ ] **Step 4: Run.** `ctest -C Release -R "own|value-"` should all pass, then the full suite: 0 failed.
- [ ] **Step 5: Commit.** `-mm=own: @value records` (plus each fix, described in the message).

### Task 7: `@value` across a module boundary

**Files:**
- Test: `tslang/test/tester/tests/export_value_types.ts`, `tests/import_value_types.ts`
- Modify: `tslang/test/tester/CMakeLists.txt` (beside `test-compile-shared-export-import-final-class`)

**Interfaces:**
- Consumes: Tasks 1-4. `__decls` carries `@value` because the printer prints it.

- [ ] **Step 1: Write the pair.**
  - `export_value_types.ts`: `@value export class V { constructor(public readonly x: number) {} static plus(a: V, b: V) { return new V(a.x + b.x); } get twice() { return this.x * 2; } }`, `@value export interface R { readonly a: V; readonly n: number; }`, and `export function mkR(n: number): R { return { a: new V(n), n: n }; }`.
  - `import_value_types.ts` sums `V.plus(mkR(i).a, new V(1)).twice` over 1000 iterations, asserts the exact total, builds an `R` literal locally and passes it back, then prints `done.`.
- [ ] **Step 2: Register.**
  - `test-compile-shared-export-import-value-types` (`test-runner -shared`) and `test-jit-shared-export-import-value-types` (`-jit -shared`), for `gc rc` through `foreach(mm gc rc)`.
  - The same two under `own`, registered the way `import_own_record` is. own is the primary target.
- [ ] **Step 3: Run, full suite, commit.** `@value: -shared import`.

### Task 8: the benchmark

**Files:**
- Modify: `tslang/benchmarks/raytrace/rtbench.ts`, `rtbench_own.ts` (header comments and the edits below)
- Modify: `tslang/benchmarks/raytrace/README.md` (a new results section above "Results 2026-10-24")
- Modify: `C:\Users\duzha\.claude\projects\I--TypeScriptCompiler\memory\raytrace-alloc-spike-2026-10-24.md` and `MEMORY.md`

**Interfaces:**
- Consumes: everything above.

- [ ] **Step 1: Edit both benchmark files** exactly as the spec's "Benchmark edits" lists. Leave every object literal unchanged.
  - `@value class Vector`, `@value class Color`, with constructor parameters `public readonly`
  - `@value interface Ray`, `@value interface Intersection`, with fields `readonly`
  - `intersect(...)`: `Intersection | null` in `Thing`, `Sphere` and `Plane`
  - `let closestInter: Intersection | undefined = undefined;`
  - `rtbench_own.ts`: `Plane.normal` returns `this.norm`

  Check that `git diff` shows no change to any `{ ... }` literal line.
- [ ] **Step 2: Check the checksum.** For each model, compile and run as `bench.ps1` does (`--emit=exe --opt --opt_level=3 --no-default-lib --mm=<model>`; own builds `rtbench_own.ts`). Each prints 38837369. A difference is a bug in Tasks 2-6: go back there with a reduced test.
- [ ] **Step 3: Check that the hot path is allocation-free.** Run `--emit=mlir-affine --stack-promotion-report` under rc on `rtbench.ts`. No remaining heap site may be a `Vector`, `Color`, `Ray` or `Intersection`. Record which heap sites remain (expected: the `addLight` closure and setup-time objects).
- [ ] **Step 4: Measure.** Run `pwsh tslang/benchmarks/raytrace/bench.ps1` twice and take the medians.
- [ ] **Step 5: Write the README section** "Results <date>: `@value`". It needs:
  - the table: every row of the 2026-10-24 table, with a `@value` column;
  - the same machine notes;
  - the stack-promotion finding;
  - "Where the time goes" for what is left, if own and none are above about 60 ms (1.1x of Rust's 54 ms). The spec makes that a report, not more work in this project.
- [ ] **Step 6: Update memory.** Record the numbers, the branch and its state in `raytrace-alloc-spike-2026-10-24.md`, and its `MEMORY.md` line.
- [ ] **Step 7: Full suite, commit.** `Ray tracer: Vector/Color/Ray/Intersection are @value; results`.
