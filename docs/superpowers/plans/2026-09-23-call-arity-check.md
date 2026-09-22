# Call Arity Check Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A call that passes fewer arguments than the callee's required parameters is a compile error, like TypeScript's TS2554 ("Expected N arguments, but got M."). Today tslang silently pads the missing arguments with `undef`. That hid a real bug: `RegExp.replaceAll` in the default library called the native `regexp_replace` with 3 of its 4 arguments. It worked at x64 by register luck and crashed at i686.

**Architecture:** Missing arguments are padded in one place, `mlirGenPrepareCallOperands` in `tslang/lib/TypeScript/MLIRGenImpl.h`. The padding stays for parameters that may be omitted: `!ts.optional<...>` (optional `b?: T` and default-valued `b = 5` parameters are both typed optional), `!ts.undefined` and `!ts.void`. For any other parameter it becomes an error. A probe with exactly that rule (2026-09-23, main 6900ee67) failed 34 ctest registrations, from 5 test files and one compiler-internal call:
- **Compiler-internal:** `mlirGen(ClassExpression)` (`MLIRGenExpressions.cpp` ~1703) turns a class expression into a class-typed value by calling `NewClassInstance(..., undefined, ...)`. That is a constructor call with NO arguments. This is itself a bug: the constructor runs at the class definition with garbage arguments (`00interface_new` prints an extra `Call ctor : 0, 0`). The fix is to allocate the object and set its vtable WITHOUT calling the constructor: `NewClassInstanceLogicAsOp(location, classInfo, false, genContext)`. An experiment showed it passes 00interface_new, 00class_expression, 00class_expression2, 00class_expression3 and 00union_to_any under gc and rc. The extra ctor line disappears. A typed `ts.Undef` does NOT work: `00interface_new` casts the value to an interface, which needs a real object and vtable. `ts.ClassRef` alone has no lowering.
- **Tests written against the silent padding:**
  - `01types_utility.ts:178` `new S()` for `constructor(s: string, n: number)`;
  - `00funcs_generic_with_typeof.ts:20` `gen<string[]>()` for `gen<T>(t: T)`.

  TypeScript rejects both.

**Tech Stack:** C++17 (MLIRGen), CMake/CTest.

**Spec:** none separate. This plan is the design record. Semantics follow TypeScript's TS2554 for the "too few arguments" direction only.

## Global Constraints

- **Omittable parameters, and only these:** `mlir_ts::OptionalType`, `mlir_ts::UndefinedType`, `mlir_ts::VoidType`. Every other missing parameter is an error. Varargs (`isVarArg`, the last parameter) keep their current handling.
- **Message:** `Expected {N} arguments, but got {M}.`. When the callee has omittable trailing parameters, use `Expected {min}-{max} arguments, but got {M}.`, as TypeScript does. N, min, max and M count only what the USER sees:
  - exclude compiler-inserted leading operands (`this`, bound-method `this`, capture/closure refs);
  - a vararg parameter counts as not required and is left out of max, so for varargs the message is `Expected at least {min} arguments, but got {M}.`.

  The numbers must be verified by tests for a plain function, a method, a constructor (`new`), and a closure that captures a variable.
- **Too many arguments is out of scope.** Do not add that check. Record it as a follow-up in the final report.
- **Emission:** use `emitError(location)` with the call's location, and return failure. No warnings, no flag to turn it off.
- **Test totals:** the default suite (`ctest -j 16 -C Release`) must end green. Its count grows only by the new negative tests.
- **DefaultLib:** its sources must still compile under the new check. Build from DefaultLib `main`, which has the replaceAll fix (merged with DefaultLib #7 on 2026-09-23).
- **Commits:** end with `Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>`. GPG signing: retry, pause, `gpgconf --kill scdaemon` / `gpg-agent`, then leave the change staged with a message file for the user. Never bypass signing.
- **Branch:** `fix-call-arity` (already created from origin/main 6900ee67).
- **Build and test commands:**
  - build: `cmake --build I:/TypeScriptCompiler/__build/tslang/windows-msbuild-2026-release --config Release --target tslang -j 8`;
  - tests: `cd` into that build dir, then `ctest -j 16 -C Release`. `-C Release` is mandatory.
  - The machine's env has stale `TSLANG_LIB_PATH`/`GC_LIB_PATH=I:\tslang`. Pass lib paths explicitly for any manual `--emit=exe`.

---

### Task 1: Class expressions stop calling the constructor

**Files:**
- Modify: `tslang/lib/TypeScript/MLIRGenExpressions.cpp` (`mlirGen(ClassExpression)`, ~1698-1703)
- Test: a new suite test `tslang/test/tester/tests/00class_expression_no_ctor.ts`, registered like its neighbours (read `tslang/test/tester/CMakeLists.txt` for how `00class_expression3.ts` is registered and copy that shape).

- [ ] **Step 1: Failing test.** The class's constructor prints, and the test asserts the constructor ran exactly once: only for the explicit `new`, not at the class definition. Use a module-level counter:

```ts
let ctorCalls = 0;

let Point = class {
    constructor(public x: number, public y: number) { ctorCalls++; }
};

function main() {
    assert(ctorCalls == 0, "class expression must not call the constructor");
    const p = new Point(3, 4);
    assert(ctorCalls == 1);
    assert(p.x == 3 && p.y == 4);
    print("done.");
}
```

Build and run it: `--emit=jit` with `--no-default-lib` and `--shared-libs=<bin>/TypeScriptRuntime.dll`, under gc and rc. Expected before the fix: the first assert fails.
- [ ] **Step 2: Fix.** Replace the `NewClassInstance(location, classValue, undefined, undefined, false, genContext)` return with an allocation that does not call the constructor:

```cpp
                // A class expression's value only has to carry the class (its type and vtable), so
                // that `new X(...)`, casts to an interface and the like work on it. Allocate it and
                // set the vtable, but do not run the constructor: that would execute user code at
                // the class definition, with no arguments for its parameters.
                return V(NewClassInstanceLogicAsOp(location, classInfo, false, genContext));
```

Drop the now-unused `classValue` / `ClassRefOp` creation if nothing else uses it.
- [ ] **Step 3: Verify.**
  - The new test passes under gc and rc.
  - `00interface_new`, `00class_expression`, `00class_expression2`, `00class_expression3` and `00union_to_any` still pass. `00interface_new` no longer prints `Call ctor : 0, 0`: check its registered expected output, if it has one.
  - Full `ctest -j 16 -C Release` is green, with 3 new registrations or however many the neighbour pattern creates.
- [ ] **Step 4: Commit.** Message: "Do not run the constructor when a class expression is evaluated".

### Task 2: The arity check

**Files:**
- Modify: `tslang/lib/TypeScript/MLIRGenImpl.h` (`mlirGenPrepareCallOperands`, and whatever passes it the count of compiler-inserted leading operands)
- Modify: `tslang/test/tester/tests/01types_utility.ts:178` (`new S()` becomes `new S("s", 1)`, keeping what the test checks), `tslang/test/tester/tests/00funcs_generic_with_typeof.ts:20` (`gen<string[]>()` becomes `gen<string[]>(undefined)`; the file is `@strict-null false`, so this is valid TypeScript and keeps the "no" result)
- Create: `tslang/test/tester/call-arity/*.ts` negative cases, registered in `tslang/test/tester/CMakeLists.txt` as plain `add_test(NAME test-compile-call-arity-<case> COMMAND <tslang> --emit=obj --no-default-lib -mm=none <file> -o <tmp>)` with `PASS_REGULAR_EXPRESSION "<exact message>"`. It must not be matched by the `tslang_add_test` twin machinery.

**Interfaces:**
- Consumes: Task 1, which removed the one compiler-internal short call.
- Produces: the error texts from Global Constraints.

- [ ] **Step 1: Failing tests.** Negative cases, each a file with one bad call and the message it must produce:
  - plain function `req(a: number, b: number)` called `req(1)`: `Expected 2 arguments, but got 1.`
  - `opt(a: number, b: number, c?: number)` called `opt(1)`: `Expected 2-3 arguments, but got 1.`
  - a method `o.m(a: number, b: number)` called `o.m(1)`: `Expected 2 arguments, but got 1.` (the `this` operand must not be counted)
  - a constructor `new C(1)` for `constructor(a: number, b: number)`: `Expected 2 arguments, but got 1.`
  - a closure capturing an outer variable, `(a: number, b: number) => a + b + captured`, called with one argument: `Expected 2 arguments, but got 1.`
  - a `declare function nat(a: string, b: string): void;` called `nat("x")`: `Expected 2 arguments, but got 1.`
  - varargs `va(a: number, ...rest: number[])` called `va()`: `Expected at least 1 arguments, but got 0.`

  Positive (in one file that must compile and run): `opt(1, 2)`, `def(1)` with `b = 5`, `va(1)`, `va(1, 2, 3)`, and a call omitting a `void` parameter.

  Register the negative tests and run them. Expected: all FAIL, because the compile succeeds today.
- [ ] **Step 2: Implement.**
  - In `mlirGenPrepareCallOperands`, before padding, check each missing parameter against the Global Constraints rule. On the first non-omittable one, emit the TS2554-style message and return failure.
  - To report user-visible counts, the function needs the number of compiler-inserted leading operands. Find how `this` / capture operands are prepended on each call path (`mlirGenCallFunction` inserts `thisValue`; constructor calls pass `this` as operand 0; bound functions and captures: trace them). Pass the offset in explicitly, as a new parameter with every caller updated, or derive it from the function type if the type marks them. Do not guess from operand types.
  - Compute min = the number of user-visible parameters that are neither omittable nor varargs, and max = the user-visible parameter count excluding varargs.
- [ ] **Step 3: Fix the two tests** (`01types_utility.ts`, `00funcs_generic_with_typeof.ts`) as listed under Files.
- [ ] **Step 4: Verify.**
  - The negative tests pass and the positive file compiles and runs.
  - `ctest -j 16 -C Release` is fully green.
  - DefaultLib compiles under the new check. From `I:\TypeScriptCompilerDefaultLib`, on `main`, pointed at the new compiler (`build.bat` uses `..\TypeScriptCompiler\__build\tslang\windows-msbuild-2026-release\bin`), run `build.bat release`. All 3 models must build.
  - Then run DefaultLib `tests.ps1` and compare against a pre-change baseline. Compile mode fails on this machine because of the stale env, so compare failure lists; jit must not regress.
  - Report any other DefaultLib or corpus file the check rejects. If one is a real missing-argument bug in DefaultLib, list it; do not fix it in this task.
- [ ] **Step 5: Commit.** Message: "Report a call with too few arguments (TS2554) instead of padding it with undef".
