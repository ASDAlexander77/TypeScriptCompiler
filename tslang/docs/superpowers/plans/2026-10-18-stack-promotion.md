# Stack Promotion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Allocate class instances, boxed object literals, array headers, fixed arrays and capture boxes on the stack when an escape analysis proves they never outlive their function, under every memory model, silently falling back to the heap otherwise.

**Architecture:** A new module pass, `StackPromotionPass`, runs after the affine lowering (and after own's inference), computes per-function "parameter may escape" facts over the call graph, then marks allocation ops whose results provably do not escape with a `stackAlloc` attribute. LLVM lowering turns a marked allocation into an entry-block `alloca` laid out exactly like a heap block (header word = `HEAP_BLOCK_IMMORTAL`, then payload). Under rc and own, the pass inserts a new `ts.ReleaseContents` op so what the object owns is given back without freeing the object.

**Tech Stack:** C++17, MLIR (TypeScript dialect, LLVM dialect), LLVM, CMake/ctest, `test-runner`.

**Spec:** `tslang/docs/superpowers/specs/2026-10-18-stack-promotion-design.md`

## Global Constraints

- Default limits: at most **256 bytes** per promoted object, at most **4096 bytes** promoted per function. Only compile-time-known sizes qualify.
- On by default at every optimisation level, including `-O0`; `--no-stack-promotion` turns it off.
- Unknown callee (declared, imported, indirect, virtual with any candidate not defined in this module) = escape. Anything the analysis cannot classify = escape.
- No promotion inside async functions or generators (version 1).
- No promotion in a function on a direct call-graph cycle in its module (recursion).
- No clone, no copy, no warning, no error: a site that escapes is an ordinary heap site.
- `Class()` means exactly `new Class()`.
- Output of every existing test must be identical with and without `--no-stack-promotion`, under gc, rc, own and none.
- All paths in this plan are relative to `tslang/` unless absolute.
- Sources are CRLF; edit with the Edit tool, never `sed -i`.

## Review Focus

1. **An object stored into a local that outlives a loop iteration** (`let keep; for (...) { const p = new P(); if (c) keep = p; }`) - must go to the heap, because the one entry-block slot is reused next iteration. Test: Task 6 `loop_carried.ts`.
2. **A method that stores `this`** (`register() { registry.push(this); }`) called on a local - `this` is a parameter; the call is an escape. Test: Task 4 `method_keeps_this.ts`.
3. **A virtual call where one override keeps its argument** - the base method borrows, the derived one stores. Must be heap. Test: Task 4 `virtual_one_keeps.ts`.
4. **rc/own: an object whose field is overwritten several times, then the function returns** - every old field value released once, the last one released at exit, nothing released twice. Test: Task 5 `field_overwrite.ts` under rc and own with `--verify-ownership`.
5. **A promoted object compared, `instanceof`-tested, or printed through its own `toString`** - none of these escape; output identical to the heap version. Test: Task 4 `borrow_uses.ts`.

---

## File Structure

| file | responsibility |
| --- | --- |
| Create `lib/TypeScript/StackPromotionPass.cpp` | the pass: escape facts, per-site decision, limits, report remarks, `ReleaseContents` insertion |
| Create `lib/TypeScript/StackEscapeFacts.h` | the per-function "parameter may escape" fact computation (module fixpoint), separate so it can later be read from declarations (spec §8.2) |
| Modify `include/TypeScript/Passes.h`, `lib/TypeScript/CMakeLists.txt` | declare and build `createStackPromotionPass(const CompileOptions &, StackPromotionLimits)` |
| Modify `include/TypeScript/TypeScriptOps.td` | `stackAlloc` on `NewArrayOp`/`CreateArrayOp`; new `ReleaseContentsOp` |
| Modify `lib/TypeScript/LowerToLLVM.cpp` | stack lowering of `NewOp`, array ops, captured `VariableOp`; `ReleaseContentsOpLowering` |
| Modify `include/TypeScript/LowerToLLVM/LLVMCodeHelperBase.h` | `StackBlockAlloc(mlir::Type storageType)` returning a payload pointer |
| Modify `include/TypeScript/LowerToLLVM/OwnershipRoutineLogic.h` | `emitReleaseContents(mlir::Type storageType, mlir::Value payloadPtr)` |
| Modify `lib/TypeScript/MLIRGenAccessCall.cpp` | `Class()` = `new Class()` |
| Modify `tslang/tslang.cpp`, `tslang/transform.cpp`, `tslang/jitcache.cpp` | flags, pipeline, JIT cache key |
| Modify `test/tester/test-runner.cpp` | `-no-stack-promotion` pass-through |
| Create `test/tester/stack-promotion/*.ts` | decision and behaviour tests |
| Modify `test/tester/CMakeLists.txt` | register them |
| Modify `docs/memory-models.md` | user-facing section |

## Test conventions used below

- **Run test** = `tslang_add_test(NAME test-jit-<mm>-stack-<name> COMMAND test-runner -jit -mm=<mm> <file>)` and the `test-compile-` twin without `-jit`, for each `mm` in `gc rc own none`, in one `foreach`. A program asserts with `assert(...)` and prints; exit code 0 is a pass.
- **Decision test** = `add_test(... COMMAND $<TARGET_FILE:tslang> --emit=mlir-affine --no-default-lib -mm=<mm> --stack-promotion-report <file>)` with `PASS_REGULAR_EXPRESSION` on the remark and `FAIL_REGULAR_EXPRESSION "Stack dump|Assertion failed"`.
- Remark format (pinned here, used by every decision test): `<file>:<line>:<col>: remark: stack-promotion: stack` or `... remark: stack-promotion: heap: <reason>`, where `<reason>` is one of `returned`, `stored`, `captured`, `argument`, `unknown-callee`, `converted`, `thrown`, `yielded`, `loop-carried`, `too-big`, `frame-budget`, `recursive`, `coroutine`, `try-region`, `dynamic-size`, `growable`, `unclassified`.
- Build: `cmake --build __build/tslang/windows-msbuild-2026-release --config Release --target tslang test-runner`; run: `ctest --test-dir __build/tslang/windows-msbuild-2026-release -C Release -R <regex> --output-on-failure` (from `I:\TypeScriptCompiler`).

---

### Task 1: `Class()` is `new Class()`

**Files:**
- Modify: `lib/TypeScript/MLIRGenAccessCall.cpp` (`NewClassInstance`, the `Class()` caller near line 2063)
- Create: `test/tester/stack-promotion/class_call_same_as_new.ts`
- Modify: `test/tester/CMakeLists.txt`

**Interfaces:**
- Produces: nothing new; `NewClassInstance(..., bool onStack)` loses `onStack` (all callers pass the heap path).

- [ ] **Step 1: Write the failing test** `class_call_same_as_new.ts`: a class `P { x: number; y: number; constructor(x, y) }` with a `toString()`; `main` builds `P(1, 2)` and `new P(3, 4)`, a function `make()` returning `P(5, 6)`, and asserts `a.x + a.y == 3`, `make().y == 6`, `a instanceof P`. Register as a run test for all four models.
- [ ] **Step 2: Run, expect FAIL** with `can't instantiate new instance of P which has 'virtual table' on stack`.
- [ ] **Step 3: Implement**: `Class()` goes through the same path as `new Class()`; delete the "virtual table on stack" error and the `TODO`. `NewClassInstanceLogicAsOp`'s `stackAlloc` parameter stays (the pass sets the attribute later, not MLIRGen) but every MLIRGen caller passes `false`.
- [ ] **Step 4: Run, expect PASS** under all four models.
- [ ] **Step 5: Commit** `Class() is new Class()`.

### Task 2: Flags, pipeline slot, and an empty pass

**Files:**
- Create: `lib/TypeScript/StackPromotionPass.cpp`
- Modify: `include/TypeScript/Passes.h`, `lib/TypeScript/CMakeLists.txt`, `tslang/tslang.cpp`, `tslang/transform.cpp`, `tslang/jitcache.cpp`, `test/tester/test-runner.cpp`, `test/tester/CMakeLists.txt`

**Interfaces:**
- Produces:
  - `struct StackPromotionLimits { unsigned maxObjectBytes = 256; unsigned maxFrameBytes = 4096; bool report = false; };` in `Passes.h`
  - `std::unique_ptr<mlir::Pass> mlir::typescript::createStackPromotionPass(CompileOptions &compileOptions, StackPromotionLimits limits);` - a `ModuleOp` pass.
  - `cl::opt<bool> noStackPromotion("no-stack-promotion")`, `cl::opt<unsigned> stackPromotionMaxObject("stack-promotion-max-object", init 256)`, `cl::opt<unsigned> stackPromotionMaxFrame("stack-promotion-max-frame", init 4096)`, `cl::opt<bool> stackPromotionReport("stack-promotion-report")`, all in `TypeScriptCompilerCategory`.
  - test-runner accepts `-no-stack-promotion` and appends `--no-stack-promotion` to `tslang_opt` (same mechanism as `-mm=`).

- [ ] **Step 1: Write the failing test**: `add_test(test-stack-promotion-flag-accepted ... tslang --emit=mlir-affine --no-default-lib -mm=none --no-stack-promotion --stack-promotion-report class_call_same_as_new.ts)` with `PASS_REGULAR_EXPRESSION "ts\\.Func"`; and a run test `test-jit-gc-stack-off-class-call` using `test-runner -jit -mm=gc -no-stack-promotion`.
- [ ] **Step 2: Run, expect FAIL** (unknown option).
- [ ] **Step 3: Implement**: flags; the pass with an empty `runOnOperation`; schedule it in `transform.cpp` right after the `MemoryModelOwn` inference block and before `#ifdef ENABLE_OPT_PASSES`, unless `noStackPromotion`; add the four flags to the JIT cache key line in `jitcache.cpp` (next to `verify-ownership=`).
- [ ] **Step 4: Run, expect PASS.**
- [ ] **Step 5: Commit** `Stack promotion: flags and an empty pass`.

### Task 3: Stack lowering of `ts.New`, and promoting leaf objects with no uses but field access

Scope: class instances and boxed object literals (`ts.New` of a `ClassType` or `ValueRefType`) whose storage owns no heap memory (`OwnershipRoutineLogic::ownsHeapMemory(storageType) == false`), whose only uses are as the base of field/property reads and writes, and retain/release ops. Everything else: heap, reason `unclassified`. This makes the mechanism work end to end before the analysis grows.

**Files:**
- Modify: `include/TypeScript/LowerToLLVM/LLVMCodeHelperBase.h`, `lib/TypeScript/LowerToLLVM.cpp` (`NewOpLowering`), `lib/TypeScript/StackPromotionPass.cpp`
- Create: `test/tester/stack-promotion/leaf_local.ts`, `leaf_returned.ts`, `literal_boxed_local.ts`

**Interfaces:**
- Produces:
  - `mlir::Value LLVMCodeHelperBase::StackBlockAlloc(mlir::Type storageType)` - entry-block `alloca` of `headerSize + sizeof(storage)` bytes, aligned to the pointer size; stores `HEAP_BLOCK_IMMORTAL` in the header and zero-fills the payload **at the allocation site** (not the entry block, so a loop re-zeroes); returns the payload pointer.
  - In the pass: `bool isStackCandidate(mlir::Operation *allocOp, StringRef &reason)`; on success sets `stackAlloc = true` on the op. When `limits.report`, `mlir::emitRemark(op->getLoc())` with the pinned format.

- [ ] **Step 1: Write the failing tests**: `leaf_local.ts` (a class with two `number` fields, built and summed in a loop of 1000, asserts the sum) - decision test expects `stack` under all four models, run test passes under all four. `leaf_returned.ts` (`return new P(1, 2)`) - decision `heap: returned`, run test passes. `literal_boxed_local.ts` - an object literal with a method, used locally - decision `stack`.
- [ ] **Step 2: Run, expect FAIL** (no remark).
- [ ] **Step 3: Implement** `StackBlockAlloc`, the `stackAlloc` branch in `NewOpLowering` calling it (replacing the header-less `ch.Alloca`), and the candidate check. Vtable store (`mlirGenSetVTableToInstance`) is unchanged - it stores through the payload pointer.
- [ ] **Step 4: Run, expect PASS**; also run the whole suite under `-mm=gc` once to confirm no output change (`ctest -C Release -R "test-(jit|compile)-" -j 12`), and compile `leaf_local.ts` with `--di` to confirm debug info still verifies. `GCNewExplicitlyTypedOp` is not handled: it is only emitted under `ENABLE_TYPED_GC`, which is retired (`LLVMCodeHelperBase.h`, heap block header comment).
- [ ] **Step 5: Commit** `Stack promotion: leaf objects with field-only uses`.

### Task 4: Escape facts for calls, methods and `this`

**Files:**
- Create: `lib/TypeScript/StackEscapeFacts.h`
- Modify: `lib/TypeScript/StackPromotionPass.cpp`
- Create tests: `borrow_uses.ts`, `pass_to_reader.ts`, `pass_to_keeper.ts`, `method_keeps_this.ts`, `virtual_one_keeps.ts`, `ctor_keeps_this.ts`, `to_interface.ts`, `to_any.ts`, `pass_to_print.ts`

**Interfaces:**
- Consumes: `own_facts::directCallee(Operation*)`, `own_facts::calleeValue(Operation*)` from `lib/TypeScript/OwnershipFacts.h`; the virtual-candidate resolution `OwnershipSignaturePass` uses (move it into `OwnershipFacts.h` if it is private to that pass).
- Produces:
  - `class StackEscapeFacts { public: explicit StackEscapeFacts(mlir::ModuleOp); bool paramMayEscape(mlir::Operation *callOp, unsigned operandIndex) const; bool isRecursive(mlir::FunctionOpInterface) const; };`
  - A parameter (including `this`) "may escape" if its value, or a value derived from it, reaches any escape in spec §5.2 inside the callee body, or is passed to a parameter that may escape. Fixpoint over the module call graph, starting from "no escape" for defined functions and "escape" for every unknown callee. Not the closed-world own facts: this is a property of the callee body, valid for any caller.
  - Conversions to `any`, interface, union or `unknown` are escapes in version 1 (stricter than spec §5.2 allows; our call - simpler and safe).

- [ ] **Step 1: Write the failing tests** (decision + run, all four models):
  - `borrow_uses.ts`: method call on the object, `===` between two locals, `instanceof`, `print(p.toString())` → `stack`.
  - `pass_to_reader.ts`: a module function `len(p: P) { return p.x * p.x + p.y * p.y; }` → `stack`.
  - `pass_to_keeper.ts`: a function storing its parameter in a global array → `heap: argument`.
  - `method_keeps_this.ts`: `register() { registry.push(this); }` → `heap: argument`.
  - `virtual_one_keeps.ts`: base `visit(x)` reads, override stores `x` in a field of a global → `heap: argument`.
  - `ctor_keeps_this.ts`: constructor stores `this` in a global → `heap: argument`.
  - `to_interface.ts`, `to_any.ts`: assigned to an interface / `any` local → `heap: converted`.
  - `pass_to_print.ts`: `print(p)` where `P` has `toString` → `heap: unknown-callee` (print is a runtime call; this pins today's behaviour so a future declaration attribute shows up as a test change).
- [ ] **Step 2: Run, expect FAIL** (all `unclassified`).
- [ ] **Step 3: Implement** `StackEscapeFacts` and extend `isStackCandidate` to accept call operands whose `paramMayEscape` is false, comparisons, and `instanceof`/type-tag reads; follow the value through local `ts.Variable` slots (store into a non-captured local slot, loads from it) with the same rules applied to every load.
- [ ] **Step 4: Run, expect PASS.**
- [ ] **Step 5: Commit** `Stack promotion: call and method escape facts`.

### Task 5: Objects that own heap memory under rc and own

**Files:**
- Modify: `include/TypeScript/TypeScriptOps.td` (`ReleaseContentsOp`), `include/TypeScript/LowerToLLVM/OwnershipRoutineLogic.h`, `lib/TypeScript/LowerToLLVM.cpp`, `lib/TypeScript/StackPromotionPass.cpp`
- Create tests: `fields_string.ts`, `fields_object.ts`, `field_overwrite.ts`, `try_region.ts`

**Interfaces:**
- Produces:
  - `def TypeScript_ReleaseContentsOp : TypeScript_Op<"ReleaseContents"> { let arguments = (ins TypeScript_AnyRefLike:$instance, TypeAttr:$storageType); }`
  - `void OwnershipRoutineLogic::emitReleaseContents(mlir::Type storageType, mlir::Value payloadPtr)` = `releaseFields(storageType, payloadPtr)` then zero the payload's owned fields (so a second call, or the next iteration's call, releases nothing). Lowers to nothing under gc and none.
- Rules (the pass, rc and own only):
  - insert `ts.ReleaseContents` immediately **before** each promoted allocation op (frees the previous iteration's contents; the zero-filled entry state makes the first one a no-op) and before every function return;
  - erase `ts.Retain`/`ts.Release`/`ts.RetainSlot`/`ts.ReleaseSlot` whose operand is the promoted object itself (they are no-ops on an immortal header; erasing them is the performance point);
  - a function containing a try/catch/finally region promotes nothing: reason `try-region` (version 1; unwind paths would need their own `ReleaseContents`).
  - Under own, the `--verify-ownership` verifier must accept the result; if it rejects, teach it that `ReleaseContents` gives back what the object owns.

- [ ] **Step 1: Write the failing tests** (run tests under all four models, plus for rc and own an `add_test` on `tslang --emit=mlir-affine --no-default-lib -mm=<mm> --verify-ownership <file>` with `FAIL_REGULAR_EXPRESSION "error:|Stack dump"`):
  - `fields_string.ts`: class with a `string` field built from concatenation in a loop of 10000 → `stack`; output identical.
  - `fields_object.ts`: field holds a heap object → `stack`; the inner object stays heap (`heap: stored`).
  - `field_overwrite.ts`: field assigned three times from fresh strings, then return → `stack`; each old value is read through a second local alias before the overwrite and printed after, so a double release shows up as a wrong value or crash.
  - `try_region.ts`: a local inside a function with `try` → `heap: try-region`.
- [ ] **Step 2: Run, expect FAIL.**
- [ ] **Step 3: Implement** the op, its lowering, `emitReleaseContents`, the insertion rules, and drop the `ownsHeapMemory == false` restriction from Task 3.
- [ ] **Step 4: Run, expect PASS**; full suite under `-mm=rc` and `-mm=own`, compare with `-no-stack-promotion` (Task 8 step 2 recipe).
- [ ] **Step 5: Commit** `Stack promotion: objects that own memory, under rc and own`.

### Task 6: Limits, loops, recursion, coroutines

**Files:**
- Modify: `lib/TypeScript/StackPromotionPass.cpp`
- Create tests: `loop_carried.ts`, `too_big.ts`, `frame_budget.ts`, `recursive.ts`, `async_local.ts`, `generator_local.ts`, `thrown.ts`

**Interfaces:**
- Consumes: `StackEscapeFacts::isRecursive`, `StackPromotionLimits`.
- Rules: an object whose value may be live at the allocation op's next execution (stored into a slot defined outside the innermost loop containing the allocation, or loaded after the loop's back edge) → `heap: loop-carried`; storage size > `maxObjectBytes` → `too-big`; sites past `maxFrameBytes` in the function, sites inside loops kept first, then source order → `frame-budget`; `isRecursive` → `recursive`; any function marked async or generator, and any function outlined from one (`func::FuncOp` continuations) → `coroutine`; operand of `ts.Throw` → `thrown`.

- [ ] **Step 1: Write the failing tests**: one per rule above, each a decision test for the expected reason plus a run test; `too_big.ts` uses a class of 40 `number` fields (320 bytes) and also checks `--stack-promotion-max-object=512` gives `stack`; `frame_budget.ts` has 20 locals of 32-field classes; `recursive.ts` is a recursive tree sum building a temporary per call, run with depth 100000 under all four models (must not overflow the stack).
- [ ] **Step 2: Run, expect FAIL.**
- [ ] **Step 3: Implement** the rules.
- [ ] **Step 4: Run, expect PASS.**
- [ ] **Step 5: Commit** `Stack promotion: limits, loops, recursion, coroutines`.

### Task 7: Arrays and capture boxes

**Files:**
- Modify: `include/TypeScript/TypeScriptOps.td` (`stackAlloc` OptionalAttr<BoolAttr> on `CreateArrayOp`, `NewArrayOp`), `include/TypeScript/LowerToLLVM/ArrayLayout.h`, `lib/TypeScript/LowerToLLVM.cpp` (`CreateArrayOpLowering`, `NewArrayOpLowering`, `VariableOpLowering` for captured variables), `lib/TypeScript/StackPromotionPass.cpp`, `OwnershipRoutineLogic.h`
- Create tests: `array_local_fixed.ts`, `array_local_push.ts`, `array_returned.ts`, `array_dynamic_length.ts`, `closure_local.ts`, `closure_escapes.ts`

**Interfaces:**
- Array header `{data, length, capacity}` goes in a `StackBlockAlloc` block. The data block goes on the stack only if the length is a compile-time constant, its byte size fits the object limit, and no use may grow it (`push`, `unshift`, `splice`, `length` store, or any call argument whose `paramMayEscape` is true); otherwise the data block is heap (reason on the remark for the data: `growable` or `dynamic-size`) while the header can still be stack.
- `emitReleaseContents` for an array header: release elements `[0, length)`, then free the data block if it is a heap block (check its header: `emitIsMortal`), then zero the header.
- Capture box (`ts.Variable` with `CAPTURE_BOX_ATTR_NAME`): stack when every closure built over it (`ts.CreateBoundFunction` / `ts.CreateClosure` users) satisfies Task 4's rules as a value; reuse the closure-escape walk in `OwnershipInferencePass` (`followClosure`) by moving it into `OwnershipFacts.h` if needed.

- [ ] **Step 1: Write the failing tests**: `array_local_fixed.ts` (`const v = [x, y, z]` summed in a loop) → header `stack`, data `stack`; `array_local_push.ts` → header `stack`, data `heap: growable`; `array_returned.ts` → `heap: returned`; `array_dynamic_length.ts` (`new Array(n)`) → data `heap: dynamic-size`; `closure_local.ts` (`[1,2,3].forEach` style with a local arrow capturing a counter, closure passed to a module function that only calls it) → box `stack`; `closure_escapes.ts` → box `heap: captured`.
- [ ] **Step 2: Run, expect FAIL.**
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run, expect PASS**; full suite under all four models.
- [ ] **Step 5: Commit** `Stack promotion: arrays and capture boxes`.

### Task 8: Whole-suite equivalence, benchmarks, docs

**Files:**
- Modify: `docs/memory-models.md`, `test/tester/CMakeLists.txt`
- Create: `test/tester/stack-promotion/bench_small_class.ts`, `bench_vector_math.ts`

- [ ] **Step 1: Equivalence run**: full ctest suite under each of gc, rc, own, none, once as is and once with the runner's `-no-stack-promotion` (add a CMake option `TSLANG_TEST_NO_STACK_PROMOTION` that appends it to every `tslang_add_test`, same mechanism as the x86 twins, default OFF). Expected: identical pass/fail lists; compare against the known-failure list for the Debug suite.
- [ ] **Step 2: Teeth**: temporarily make `paramMayEscape` return `false` always; confirm `pass_to_keeper.ts` or `method_keeps_this.ts` fails at run time (a dangling read). Revert; commit nothing from this step.
- [ ] **Step 3: Benchmarks**: `bench_small_class.ts` (10^7 iterations building and summing a 2-field class), `bench_vector_math.ts` (3-component vector temporaries), plus the ray tracer and n-body used in `docs/memory-models.md`; `--emit=exe --opt --opt_level=3`, each model, with and without `--no-stack-promotion`; record time, peak working set, allocation count in a table.
- [ ] **Step 4: Docs**: a "Stack promotion" section in `docs/memory-models.md`: what it does, that it is automatic and silent, the two limits and their flags, `--no-stack-promotion`, `--stack-promotion-report`, the version-1 exclusions (calls into other modules, async/generators, recursion, try regions), and the benchmark table.
- [ ] **Step 5: Commit** `Stack promotion: suite equivalence, benchmarks, docs`.
