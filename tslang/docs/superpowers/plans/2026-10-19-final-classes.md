# `@final` Classes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a `@final` class decorator. No class may extend a `@final` class, and calls on a value whose static type is a `@final` class are direct, not through the vtable. Then mark the DefaultLib classes, so that seven `-mm=own` errors in `lib.ts` go away.

**Architecture:** MLIRGen reads `@final` into `ClassInfo::isFinal`. It rejects `extends` of such a class and `@final` combined with `abstract` or `@cpp`, and `DeclarationPrinter` prints the decorator. The member-access chain (`ClassMembersAccess` -> `ClassBaseClassAccess` -> `ClassMethodAccess` / `ClassAccessorAccess`) carries a `finalReceiver` flag, which turns the virtual reference into the direct `ThisSymbolRefOp`. No ownership pass is expected to change. Task 3 fixes a real pass imprecision if one shows up.

**Tech Stack:** C++17, MLIR (ts dialect), CMake/CTest, gtest (`unittests/MLIRGen`), the tslang test-runner, TypeScript sources for the DefaultLib.

**Spec:** `tslang/docs/superpowers/specs/2026-10-19-final-classes-design.md`

## Global Constraints

- Base: the staged `own-nodrops-declarations` commit must land first, and the user runs it (GPG): `git commit -F C:\Users\duzha\AppData\Local\Temp\claude\I--TypeScriptCompiler\fa46240b-10d4-4a2a-8a46-50b5c5be8c2f\scratchpad\msg_own_nodrops.txt`. Then branch `own-final-classes` from it with `git checkout -b own-final-classes` (in `I:\TypeScriptCompiler`).
- DefaultLib work continues on its existing branch `own-default-lib` in `I:\TypeScriptCompilerDefaultLib`.
- The sources use CRLF. Edit them with the Edit tool or a Python script that writes `newline=''`. Never use `sed -i`, which rewrites the file to LF.
- Never bypass commit signing (`--no-gpg-sign`, `-c commit.gpgsign=false`). On `gpg: signing failed: Timeout`, retry from PowerShell, pause about 30s and retry, then leave the change staged and hand the exact command to the user.
- Build: `cmake --build I:/TypeScriptCompiler/__build/tslang/windows-msbuild-2026-release --config Release --target tslang --parallel 8`. Tests: `ctest -C Release -j 16 --timeout 300` in that folder. Without `-C Release` nothing runs.
- Behaviour under gc, rc and none must not change. Only the dispatch changes.
- A `@final` class keeps its vtable, its `.instanceOf` and its interface vtables.
- The direct call applies only when the method's owning class is neither `isDynamicImport` nor `isCppVTable`, and the method is not `isAbstract`.
- Error texts, pinned by the tests:
  - `class 'X' cannot extend 'B': it is @final`
  - `class 'X' cannot be both @final and abstract`
  - `class 'X' cannot be both @final and @cpp`

## Review Focus

- A generic `@final class Box<T>`: each specialization is `@final` too, and its method calls are direct. This is pinned in Task 2's `final_dispatch.ts`.
- A method read as a value (`const fn = f.m; fn()`): the direct `ThisSymbolRef` still binds `this`. Pinned in Task 2.
- A call through an optional or nullable `F | null` (`g?.m()`): it still runs the right method. Pinned in Task 2.
- `this.m()` inside the `@final` class's own constructor and methods is direct and correct. Pinned in Task 2's IR check.
- A `@final` class used from a DLL loaded at run time (JIT `-shared`): it keeps vtable calls and still links. Pinned in Task 4's JIT run.

---

### Task 1: `@final` is parsed, stored, enforced and printed

**Files:**
- Modify: `tslang/include/TypeScript/Defines.h` (beside `#define CPP_VTABLE "cpp"`, about line 241)
- Modify: `tslang/include/TypeScript/MLIRLogic/MLIRGenStore.h` (`ClassInfo` fields about 785-800, and the constructor initializer list about 815)
- Modify: `tslang/lib/TypeScript/MLIRGenClasses.cpp` (class decorator loop about 361; the `ExtendsKeyword` branch of `mlirGenClassHeritageClause` about 1224-1245)
- Modify: `tslang/lib/TypeScript/DeclarationPrinter.cpp` (`print(ClassInfo::TypePtr)`, about 436)
- Test: `tslang/unittests/MLIRGen/DeclarationPrinter.cpp`
- Test: `tslang/test/tester/final/extends_final.ts`, `extends_declared_final.ts`, `final_abstract.ts`, `final_cpp.ts`
- Modify: `tslang/test/tester/CMakeLists.txt` (a new `final_error_cases` block after the `call_arity_cases` block, about line 2775, in the same shape)

**Interfaces:**
- Produces:
  - `#define FINAL_DECORATOR "final"`
  - `bool ClassInfo::isFinal` (default `false`), set for definitions and for `declare class` declarations alike
  - `DeclarationPrinter` writes `@final` on its own line before a final class

- [ ] **Step 1: Write the failing printer unit tests.** Add `final_class_prints_final_decorator` (`ci->isFinal = true`; `EXPECT_THAT(text, testing::HasSubstr("@final"))` and `HasSubstr("class Shape")`) and `non_final_class_has_no_final_decorator` (`Not(HasSubstr("@final"))`), next to `abstract_class_prints_abstract_keyword`.
- [ ] **Step 2: Write the error-case files.** Each has one offence and nothing else that fails:
  - `extends_final.ts`: `@final class B { m() { return 1; } }` and `class X extends B {}`. Expect `class 'X' cannot extend 'B': it is @final`.
  - `extends_declared_final.ts`: `@final declare class D { m(): int; }` and `class E extends D {}`. Expect `class 'E' cannot extend 'D': it is @final`. This is the lib.d.ts shape.
  - `final_abstract.ts`: `@final abstract class A { abstract m(): int; }`. Expect `class 'A' cannot be both @final and abstract`.
  - `final_cpp.ts`: `@final @cpp declare class P { m(): int; }`. Expect `class 'P' cannot be both @final and @cpp`.
- [ ] **Step 3: Register them.** Add `final_error_cases` as `"name|regex"` entries. Run each with `$<TARGET_FILE:tslang> --emit=obj --no-default-lib -mm=none <file> -o ${CMAKE_CURRENT_BINARY_DIR}/final-<name>.obj`. Use `PASS_REGULAR_EXPRESSION` set to the message and `FAIL_REGULAR_EXPRESSION "Stack dump|Assertion failed"`, exactly as `call_arity_cases` does. The test names are `test-compile-final-<name>`.
- [ ] **Step 4: Confirm the new tests fail.** Reconfigure (`cmake .`), build `tslang` and `MLIRTypeScriptUnitTests`, then run `ctest -C Release -R "final-|unittest-MLIRGen"`. Expected: the four `test-compile-final-*` tests and the two new unit tests fail.
- [ ] **Step 5: Implement.**
  - Add the define and the field.
  - In the decorator loop, `if (name == FINAL_DECORATOR) newClassPtr->isFinal = true;`. After the loop, emit the two combination errors at the class location. Use `newClassPtr->isAbstract` and `newClassPtr->isCppVTable` and fail the declaration the way the loop's other errors do.
  - In the `ExtendsKeyword` branch, right after `auto classInfo = getClassInfoByFullName(baseName);`, emit the extends error at the `extendingType` location when `classInfo && classInfo->isFinal`, and fail.
  - In the printer, print `"@" FINAL_DECORATOR` and a newline when `classType->isFinal`, in the same place as `@cpp`.
- [ ] **Step 6: Run the tests.** `ctest -C Release -R "final-|unittest-MLIRGen"` should all pass. Then run the full suite: expected 0 failed.
- [ ] **Step 7: Commit.** `@final classes: parse, store, reject extends/abstract/@cpp, print`.

### Task 2: direct dispatch on a `@final` receiver

**Files:**
- Modify: `tslang/lib/TypeScript/MLIRGenImpl.h`:
  - declarations about 8153-8164
  - `ClassMethodAccess` about 7958-8075, with the virtual branch at about 8045
- Modify: `tslang/lib/TypeScript/MLIRGenAccessCall.cpp`:
  - `ClassAccessorAccess` about 583-811, with the virtual reference at about 705
  - `ClassBaseClassAccess` about 935-950
  - `ClassMembersAccess` (the `ClassInfo` overload) about 985-1050
  - the overload re-resolution call at about 1983
- Test: `tslang/test/tester/final/final_dispatch.ts`, `final_ir_direct.ts`, `final_ir_base.ts`
- Modify: `tslang/test/tester/CMakeLists.txt` (after `final_error_cases`)

**Interfaces:**
- Consumes: `ClassInfo::isFinal` (Task 1).
- Produces: a trailing parameter `bool finalReceiver = false`, after `const GenContext &genContext`, on these four functions. Declarations and definitions must match.
  - `ClassMembersAccess(mlir::Location, mlir::Value thisValue, ClassInfo::TypePtr, mlir::StringRef name, bool isSuperClass, mlir::Value argument, mlir_ts::AccessLevel, const GenContext &, bool finalReceiver = false)`
  - `ClassBaseClassAccess(ClassInfo::TypePtr classInfo, ClassInfo::TypePtr baseClass, int index, mlir::Location, mlir::Value thisValue, StringRef name, mlir::Value argument, mlir_ts::AccessLevel, const GenContext &, bool finalReceiver = false)`
  - `ClassMethodAccess(ClassInfo::TypePtr, mlir::Location, mlir::Value thisValue, int methodIndex, bool isSuperClass, mlir_ts::AccessLevel, const GenContext &, bool finalReceiver = false)`
  - `ClassAccessorAccess(ClassInfo::TypePtr, mlir::Location, mlir::Value thisValue, int accessorIndex, bool isSuperClass, mlir_ts::AccessLevel, const GenContext &, bool finalReceiver = false)`

- [ ] **Step 1: Write `final_dispatch.ts`.** It runs and asserts, and ends with `print("done.")`:
  - a non-final `class B { m(): int { return 1; } n(): int { return 10; } get g(): int { return 100; } }`
  - `interface I { m(): int; }`
  - `@final class F extends B implements I { m(): int { return 2; } get g(): int { return 200; } k(): int { return this.m() + this.n(); } }`
  - `@final class Box<T> { constructor(public v: T) {} get(): T { return this.v; } }`

  Asserts, in `main`:
  - `f.m() == 2`, `f.n() == 10`, `f.g == 200`, `f.k() == 12`
  - `(<B>f).m() == 2`, `(<I>f).m() == 2`, `f instanceof B`
  - `const fn = f.m; fn() == 2`
  - `let o: F | null = f; o?.m() == 2`
  - `new Box<int>(7).get() == 7`
- [ ] **Step 2: Write the IR checks.**
  - `final_ir_direct.ts` holds the classes above and only calls on `F`- and `Box<int>`-typed values, including `this.m()` inside `F.k` and a call in `F`'s constructor.
  - `final_ir_base.ts` holds the same classes and only calls `const b: B = new F(); b.m();`.
- [ ] **Step 3: Register them.**
  - `final_dispatch` with `tslang_add_test(NAME test-compile-final-dispatch-${mm} COMMAND test-runner -mm=${mm} ...)` and `test-jit-final-dispatch-${mm}`, in a `foreach(mm gc rc none own)`.
  - `final_ir_direct` with `add_test` running `tslang --emit=mlir --no-default-lib`. Use `PASS_REGULAR_EXPRESSION "ThisSymbolRef"` and `FAIL_REGULAR_EXPRESSION "ThisVirtualSymbolRef|error|Stack dump"`.
  - `final_ir_base` the same way, with `PASS_REGULAR_EXPRESSION "ThisVirtualSymbolRef"` and `FAIL_REGULAR_EXPRESSION "error|Stack dump"`.
- [ ] **Step 4: Confirm the expected results before the change.** Reconfigure, build, then `ctest -C Release -R "final-dispatch|final-ir"`. Expected: `final-ir-direct` fails (it has `ThisVirtualSymbolRef`), `final-ir-base` passes, and the `final-dispatch-*` runs pass. If `own` fails here, note the error for Task 3 and continue.
- [ ] **Step 5: Implement.**
  - At the top of the `ClassInfo` overload of `ClassMembersAccess`, compute `finalReceiver = finalReceiver || (!isSuperClass && classInfo->isFinal);`.
  - Pass that flag into `ClassMethodAccess`, `ClassAccessorAccess` and `ClassBaseClassAccess`. `ClassBaseClassAccess` passes it on into its recursive `ClassMembersAccess`.
  - At about 1983, pass `classInfo->isFinal`.
  - In `ClassMethodAccess` and `ClassAccessorAccess`, compute `direct = finalReceiver && !methodInfo.isAbstract && !classInfo->isDynamicImport && !classInfo->isCppVTable` (for the accessor, from the accessor's get/set method info). When `direct` is true, skip the virtual branch and fall through to the existing `ThisSymbolRefOp` path that non-virtual methods already take.
  - Check every other `VirtualSymbolRefOp` / `ThisVirtualSymbolRefOp` creation in these two files. Either apply the rule or add a one-line comment saying why it does not apply. Interface calls stay unchanged.
- [ ] **Step 6: Run the tests.** `ctest -C Release -R "final-"` should all pass. Then run the full suite: expected 0 failed.
- [ ] **Step 7: Commit.** `@final classes: direct calls on a @final receiver`.

### Task 3: own resolves calls on `@final` classes

**Files:**
- Test: `tslang/test/tester/own/own_final_fresh_result.ts`, `own/own_final_field_lend.ts`
- Modify: `tslang/test/tester/CMakeLists.txt` (append both to the `foreach(own_test ...)` list after `own_string_concat_element`)
- Modify, only if Step 2 fails: `tslang/lib/TypeScript/OwnershipInferencePass.cpp` / `OwnershipSignaturePass.cpp` / `OwnershipFacts.h`

**Interfaces:**
- Consumes: the direct calls from Task 2.

- [ ] **Step 1: Write the own tests.** Both are exported classes, the lib.ts shape, with a `churn()` of same-size objects between the call and the use, and `print("done.")`. Use 10000 iterations and assert the exact totals.
  - `own_final_fresh_result.ts`: `@final export class Re { exec(s: string): Res | null { ... } }` returning `new Res(...)` or `null`. Also `@final export class D { t = 1; toString() { return "x" + this.t; } prim(h: string): string | number { if (h === "number") return <number>this.t; return this.toString(); } }`. Use both: `const r = re.exec(s); if (r != null) ...`, and `d.prim("string")`.
  - `own_final_field_lend.ts`: `@final export class Re { test(s: string): boolean { return s.length > 2; } }` and `export class W { constructor(private value: string) {} has(re: Re) { return re.test(this.value); } }`.
- [ ] **Step 2: Run them under own.** Run `Release/test-runner.exe -mm=own <file>` and `-jit -mm=own <file>` in `__build/.../test/tester`. Expected: exit 0.
- [ ] **Step 3: Handle a gap.** If either fails, reduce it to a minimal repro under `--no-default-lib` and find the cause in the pass. Then decide:
  - **Real imprecision:** fix it, and add a positive own test plus, if the fix narrows an error, an `own_err_*` case.
  - **Correct error:** stop and report it with the repro. The spec says such a case is decided separately.

  Never special-case `@final` inside an ownership pass. A direct call is the whole mechanism.
- [ ] **Step 4: Register and run.** `ctest -C Release -R "own"` should all pass. Then run the full suite: expected 0 failed.
- [ ] **Step 5: Commit.** `-mm=own: tests for calls on @final classes` (plus any fix, described in the message).

### Task 4: `@final` across a module boundary

**Files:**
- Test: `tslang/test/tester/tests/export_final_class.ts`, `tests/import_final_class.ts`
- Modify: `tslang/test/tester/CMakeLists.txt` (beside `test-compile-shared-export-import-class-interface`, about line 1490)

**Interfaces:**
- Consumes: Tasks 1-2. `__decls` carries `@final` because the printer prints it.

- [ ] **Step 1: Write the pair.**
  - `export_final_class.ts`: `@final export class Counter { n = 0; inc(): int { return ++this.n; } get value(): int { return this.n; } }`.
  - `import_final_class.ts` imports it, calls `inc()` 1000 times and asserts `c.value == 1000`, then prints `done.`.
- [ ] **Step 2: Register them.** Add `test-compile-shared-export-import-final-class` (`test-runner -shared`) and `test-jit-shared-export-import-final-class` (`test-runner -jit -shared`, the dynamic import) for `gc` and `rc`, through a `foreach(mm gc rc)` that passes `-mm=${mm}`.
- [ ] **Step 3: Run.** `ctest -C Release -R "import-final-class"` should all pass. A JIT link failure means the dynamic-import exclusion from Task 2 is not taking effect: fix it there.
- [ ] **Step 4: Full suite, then commit.** Run the full suite: expected 0 failed. Commit message: `@final classes: -shared import, compiled and dynamic`.

### Task 5: DefaultLib classes are `@final`

**Files (in `I:\TypeScriptCompilerDefaultLib`):**
- Modify: `src/lib.ts`: add `@final` on its own line before `export class` (the codebase's decorator placement, as with `@cpp`) for `Boolean`, `Number`, `BigInt`, `Date`, `MatchIndicesResults`, `MatchResults`, `RegExp`, `String`, `ArrayBuffer`, `Headers` and `Response`.
- Modify: `src/lib.d.ts`: add `@final` on its own line before `declare class` for the same classes, except `ArrayBuffer`, which has no declaration.
- Modify: `src/lib.ts`, `Headers` constructor: revert the `k + ""` / `v + ""` copies and their comment to `for (const [k, v] of init) this.set(k, v);`.

**Interfaces:**
- Consumes: a `tslang.exe` built from Tasks 1-3.

- [ ] **Step 1: Record the baseline.** Run `scratchpad/ownlib.sh` and the same with `OPTS="--di --opt_level=0"`, and keep the error lists. Expected: 14 (release) and 18 (debug) on the current branch heads.
- [ ] **Step 2: Mark the classes and revert the Headers copies.**
- [ ] **Step 3: Compile under own again.** Re-run both `ownlib.sh` variants. Expected: no errors at the String wrapper `match`/`replace`/`replaceAll`/`search`/`split` lines, at `Date[Symbol.toPrimitive]`, at `___String.match`'s `if (result == null)` (debug), or in the `Headers` constructor. What remains is only `matchAll` (its function line, its `next`, and the mislocated `lib.ts:151`), `String.iterator` (debug) and `Response.constructor`. Any other error goes back through Task 3, Step 3.
- [ ] **Step 4: Check gc and rc.**
  - Build: `cmd /c "cd /d I:\TypeScriptCompilerDefaultLib && .\build.bat release gc"`, then the same with `rc`. Both should exit 0.
  - Test: `pwsh -NoProfile -File .\tests.ps1 -Model gc`, then `-Model rc`. Expected: `Finished release, compile` and `Finished release, jit` all passing (gc 167/167; rc 166/167 with 1 skipped). The debug passes link against a stale debug runtime and are not a criterion.
- [ ] **Step 5: Run the compiler's DefaultLib tests.** `ctest -C Release -R "defaultlib"` in the compiler build should pass.
- [ ] **Step 6: Commit (DefaultLib, unsigned repo).** Message: `@final on the library's classes: own resolves their methods; Headers no longer copies`.
