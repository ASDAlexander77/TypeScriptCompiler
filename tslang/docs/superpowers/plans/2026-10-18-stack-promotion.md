# Stack Promotion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Allocate class instances, boxed object literals, array headers, fixed arrays and capture boxes on the stack when an escape analysis proves they never outlive their function, under every memory model, silently falling back to the heap otherwise.

**Architecture:** A new module pass, `StackPromotionPass`, runs after the affine lowering (and after own's inference), computes per-function "parameter may escape" facts over the call graph, then marks allocation sites whose results provably do not escape with a `stackAlloc` attribute. An allocation site is a `ts.New` (boxed object literal, class expression) or, for `new C()`, the call of the class's generated `C..new` method (see "How a class allocation site looks" below); a promoted `C..new` call is replaced by an inline `ts.New stackAlloc` plus the vtable store. LLVM lowering turns a marked allocation into an entry-block `alloca` laid out exactly like a heap block (header word = `HEAP_BLOCK_IMMORTAL`, then payload). Under rc and own, the pass inserts a new `ts.ReleaseContents` op so what the object owns is given back without freeing the object.

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
- All paths in this plan are relative to `tslang/` unless absolute. One exception: the user docs live at the repository root, `I:\TypeScriptCompiler\docs\memory-models.md` (`../docs/memory-models.md` from `tslang/`); there is no `tslang/docs/memory-models.md`.
- Sources are CRLF; edit with the Edit tool, never `sed -i`.
- `--verify-ownership` runs **before** own's inference and therefore before this pass (`tslang/transform.cpp:136-150`: the verifier pairs rc's acquisitions with their releases, and inference erases the acquisitions it proves are the one owner's, so the verifier cannot run after inference). The verifier never sees promoted IR. No test in this plan claims it does; teaching it `ReleaseContents` and scheduling a second run is a follow-up.

## How a class allocation site looks

`new C()` does **not** put a `ts.New` in the caller. With `USE_NEW_AS_METHOD` (`include/TypeScript/Config.h:28`, always on), `NewClassInstanceAsMethodCallOp` (`lib/TypeScript/MLIRGenAccessCall.cpp:2554-2584`) emits a call of the class's generated static method `C..new` (`NEW_METHOD_NAME` = `.new`, built in `MLIRGenClasses.cpp:1795-1840`), whose body is `return new this` with the constructor call suppressed: a `ts.New`, the vtable store (`mlirGenSetVTableToInstance`: `ts.AddressOf` of the class vtable + field ref + `ts.Store`), then the return. The caller then calls the constructor with the result as `this`. The call result carries `__owned_result` (`OWNED_RESULT_ATTR_NAME`): `C..new`'s return path retains, so the instance arrives owned, and `OwnedReturnConsumptionPass` (before the affine lowering) has already decided who consumed it by the time this pass runs. After Task 1 the old `Class()` path is gone, so the only inline class `ts.New` in a user function is a class expression (`MLIRGenExpressions.cpp:2145`); boxed object literals (`ts.New` of a `ValueRefType`) are inline. The pass runs before the inliner, so `--opt` does not change this.

The pass therefore recognises **two kinds of site**:

1. a `ts.New` op (object literal box, class expression);
2. a zero-operand `ts.SymbolCallInternal` (or `ts.Invoke`, which only occurs in a try region and is excluded anyway) whose callee, via `own_facts::directCallee`, is a `ts.Func` named `<class>..new` **defined in this module** with a body (an imported class has a declaration only: unknown callee, heap).

The value under analysis is the call result; the constructor call that follows is an ordinary call with the object as `this`, judged by Task 4's facts. When a kind-2 site is promoted, the pass replaces the call: it creates `ts.New stackAlloc = true` of the class type at the call, clones the vtable-store ops from the first block of `C..new`'s body (everything between its `ts.New` and the store into its return slot; the body has no arguments, so the mapping is the cloned `ts.New` only), copies every attribute of the call op onto the new `ts.New` (so `__owned_result` and the `__owned_consumed` on the receiving slot stay consistent; no LLVM lowering reads `__owned_result`), and replaces the call result. If the clone turns out brittle (the `.new` body shape changes), the fallback is MLIRGen emitting `ts.New` + vtable store inline for a class defined in the current module and the `C..new` call only for a declaration; that is a larger, rc-visible change and is not the plan's choice.

## Calls this plan makes that differ from the spec's wording

- Functions with a try/catch/finally region promote nothing in version 1 (spec §6 has the scope-exit action run on unwind paths too).
- A separate `StackEscapeFacts` analysis instead of running `OwnershipSignaturePass` under every model (spec §5.1 left the choice to the builder; own's facts assume a closed world and retain ops, which gc IR does not have).
- Every conversion to `any`, an interface, a union or `unknown` is an escape (spec §5.2 only counts it when the converted value escapes).

## Review Focus

1. **An object stored into a local that outlives a loop iteration** (`let keep; for (...) { const p = new P(); if (c) keep = p; }`) - must go to the heap, because the one entry-block slot is reused next iteration. Test: Task 4 `loop_carried.ts`.
2. **A method that stores `this`** (`register() { registry.push(this); }`) called on a local - `this` is a parameter; the call is an escape. Test: Task 4 `method_keeps_this.ts`.
3. **A virtual call where one override keeps its argument** - the base method borrows, the derived one stores. Must be heap. Test: Task 4 `virtual_one_keeps.ts`.
4. **rc/own: an object whose field is overwritten several times, then the function returns** - every old field value released once, the last one released at exit, nothing released twice. Test: Task 5 `field_overwrite.ts` under rc and own (run tests; the verifier cannot check this, see Global Constraints).
5. **A promoted object compared, `instanceof`-tested, or printed through its own `toString`** - none of these escape; output identical to the heap version. Test: Task 4 `borrow_uses.ts`.
6. **A `new C()` site is a `C..new` call, not a `ts.New`** - the pass must recognise the call (see "How a class allocation site looks") and the replacement must leave rc/own bookkeeping balanced. Test: Task 3 `leaf_local.ts` under all four models.
7. **`return p.s` from a promoted object under rc and own** - the returned string must be retained before the object's contents are released. Test: Task 5 `return_field.ts`.

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
| Modify `lib/TypeScript/OwnershipFacts.h` | virtual-candidate resolution and `followClosure` moved here if they are private to the own passes (Tasks 4, 7) |
| Modify `tslang/tslang.cpp`, `tslang/transform.cpp`, `tslang/jitcache.cpp` | flags, pipeline, JIT cache key |
| Modify `test/tester/test-runner.cpp` | `-no-stack-promotion` pass-through, with its own cached-script suffix |
| Create `test/tester/stack-promotion/*.ts` | decision and behaviour tests |
| Modify `test/tester/CMakeLists.txt` | register them |
| Modify `../docs/memory-models.md` (repository root) | user-facing section |

## Test conventions used below

- **Run test** = `tslang_add_test(NAME test-jit-<mm>-stack-<name> COMMAND test-runner -jit -mm=<mm> <file>)` and the `test-compile-` twin without `-jit`, for each `mm` in `gc rc own none`, in one `foreach`. A program asserts with `assert(...)` and prints; exit code 0 is a pass. test-runner always passes `--no-default-lib` (`test-runner.cpp:91,96`), so `-mm=own` needs no own DefaultLib build; test programs must not import the default library.
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
- [ ] **Step 3: Implement**: `Class()` goes through the same path as `new Class()` - `NewClassInstanceOnStack` (`MLIRGenAccessCall.cpp:2056-2064`) becomes a call of `NewClassInstance(..., onStack = false)`, i.e. the `C..new` call plus the constructor call; delete the "virtual table on stack" error, the `onStack` parameter and the `TODO`. `NewClassInstanceLogicAsOp`'s `stackAlloc` parameter stays (the pass sets the attribute later, not MLIRGen) but every MLIRGen caller passes `false`.
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
  - test-runner accepts `-no-stack-promotion` and appends `--no-stack-promotion` to `tslang_opt` (same mechanism as `-mm=`, `test-runner.cpp:790-799`), **and** adds `nosp` to `optVariantSuffix()` (`test-runner.cpp:155-158`) so the cached `jit*.bat`/`compile*.bat` scripts of a `-no-stack-promotion` run are their own files. The scripts embed `tslang_opt` at creation time and are shared by every parallel runner with the same name; without the suffix, whichever runner created `jitgc.bat` first would decide for everyone and Task 8's equivalence run would compare promotion-on with promotion-on. Delete stale `__build/.../compile*.bat` and `jit*.bat` after changing the runner.

- [ ] **Step 1: Write the failing tests**:
  - `test-stack-promotion-report-reaches-stderr`: `tslang --emit=mlir-affine --no-default-lib -mm=none --stack-promotion-report class_call_same_as_new.ts`, `PASS_REGULAR_EXPRESSION "remark: stack-promotion: pass-ran"`. This proves both that remarks print and that the pass runs before `--emit=mlir-affine` stops the pipeline; every later decision test depends on it. If the pass is scheduled after that emit point, move the decision tests to whichever `--emit=` level does run it and update the convention above.
  - `test-stack-promotion-off`: same command with `--no-stack-promotion` added, `FAIL_REGULAR_EXPRESSION "pass-ran"`, `PASS_REGULAR_EXPRESSION "ts\\.Func"`.
  - a run test `test-jit-gc-stack-off-class-call` using `test-runner -jit -mm=gc -no-stack-promotion`.
- [ ] **Step 2: Run, expect FAIL** (unknown option).
- [ ] **Step 3: Implement**: flags; the pass, whose `runOnOperation` for now only emits `remark: stack-promotion: pass-ran` on each `ts.Func` when `limits.report` (removed in Task 3); schedule it in `transform.cpp` right after the `MemoryModelOwn` inference block and before `#ifdef ENABLE_OPT_PASSES`, unless `noStackPromotion` (that slot is inside `if (isLoweringToAffine)`, which `--emit=mlir-affine` enters, so the decision tests see the pass; remarks reach stderr through the postponed-diagnostics handler in `runMLIRPasses`, which keeps every severity); add the four flags to the JIT cache key line in `jitcache.cpp` (next to `verify-ownership=`); the runner flag and suffix above.
- [ ] **Step 4: Run, expect PASS.**
- [ ] **Step 5: Commit** `Stack promotion: flags and an empty pass`.

### Task 3: Stack lowering of `ts.New`, `C..new` sites, and promoting leaf objects with no uses but field access

Scope, in this order: first boxed object literals (`ts.New` of a `ValueRefType`, inline in the user function), then class instances through the `C..new` call site (see "How a class allocation site looks"). Both only when the storage owns no heap memory (`OwnershipRoutineLogic::ownsHeapMemory(storageType) == false`) and the only uses of the object are as the base of field/property reads and writes, and retain/release ops. A derived reference (the result of a property/field ref op on the object) is itself checked with the same whitelist: consumed only as the address of a load or store, never a call operand, never the stored value - otherwise `p.inner.method()` or a ref passed as `this` slips through as "field access". Everything else: heap, reason `unclassified`. In particular a call operand is `unclassified` here, so a class **with a constructor** (the constructor call takes the object as `this`) cannot be promoted until Task 4's call facts exist; this task's class test uses a class with no constructor and no field initializers (`mlirGenCallConstructor`, `MLIRGenAccessCall.cpp:2330-2365`, emits no call for one). This makes the mechanism work end to end before the analysis grows.

**Files:**
- Modify: `include/TypeScript/LowerToLLVM/LLVMCodeHelperBase.h`, `lib/TypeScript/LowerToLLVM.cpp` (`NewOpLowering`), `lib/TypeScript/StackPromotionPass.cpp`
- Create: `test/tester/stack-promotion/literal_boxed_local.ts`, `leaf_local.ts`, `leaf_returned.ts`

**Interfaces:**
- Produces:
  - `mlir::Value LLVMCodeHelperBase::StackBlockAlloc(mlir::Type storageType)` - entry-block `alloca` of `headerSize + sizeof(storage)` bytes, aligned to the pointer size, created through the existing `Alloca(mlir::Type, int)` helper (it inserts at the front of the entry block, which `verifyAlloca` in `LowerToLLVM.cpp:7956` requires of every constant-size alloca outside a stacksave/restore pair); stores `HEAP_BLOCK_IMMORTAL` in the header and zero-fills the payload **at the allocation site** (not the entry block, so a loop re-zeroes); returns the payload pointer.
  - In the pass: `bool isStackCandidate(mlir::Operation *siteOp, StringRef &reason)` over both site kinds; on success sets `stackAlloc = true` on a `ts.New`, or performs the `C..new` call replacement and sets it on the new `ts.New`. When `limits.report`, `mlir::emitRemark(siteOp->getLoc())` with the pinned format, at the location of the site in the user function (the call's location, not `.new`'s body).

- [ ] **Step 1: Write the failing tests**: `literal_boxed_local.ts` - an object literal with a method and two `number` fields, used locally (fields read and written, no method call yet) - decision `stack` under all four models, run test passes under all four. `leaf_local.ts` (a class `P { x: number; y: number; }` with **no constructor**, `const p = new P(); p.x = i; p.y = i + 1;` summed in a loop of 1000, asserts the sum) - decision `stack`, run test passes under all four; confirm first with `--emit=mlir-affine` that the site is a `ts.SymbolCallInternal @P..new` with no constructor call after it (if a constructor call is emitted anyway, this test moves to Task 4 and only the literal test stays here). `leaf_returned.ts` (`return new P()`) - decision `heap: returned`, run test passes.
- [ ] **Step 2: Run, expect FAIL** (no remark).
- [ ] **Step 3: Implement** `StackBlockAlloc`, the `stackAlloc` branch in `NewOpLowering` calling it (replacing the header-less `ch.Alloca`), the candidate check for a `ts.New` site, then the `C..new` site: recognition, the clone of the vtable store from `.new`'s body, the attribute copy, the replacement. The cloned vtable store writes through the payload pointer, as it does for a heap block. Under rc and own the replaced call result's retains/releases are untouched in this task (they run against the immortal header; Task 5 erases them).
- [ ] **Step 4: Run, expect PASS**; also run the whole suite under `-mm=gc` once to confirm no output change (`ctest -C Release -R "test-(jit|compile)-" -j 12`), and compile `leaf_local.ts` with `--di` to confirm debug info still verifies. `GCNewExplicitlyTypedOp` is not handled: it is only emitted under `ENABLE_TYPED_GC`, which is `false` (`include/TypeScript/Config.h:62`; `LLVMCodeHelperBase.h` heap block header comment).
- [ ] **Step 5: Commit** `Stack promotion: leaf objects with field-only uses`.

### Task 4: Escape facts for calls, methods and `this`

**Files:**
- Create: `lib/TypeScript/StackEscapeFacts.h`
- Modify: `lib/TypeScript/StackPromotionPass.cpp`
- Create tests: `ctor_local.ts`, `borrow_uses.ts`, `pass_to_reader.ts`, `pass_to_keeper.ts`, `method_keeps_this.ts`, `virtual_one_keeps.ts`, `ctor_keeps_this.ts`, `to_interface.ts`, `to_any.ts`, `pass_to_print.ts`, `loop_carried.ts`

**Interfaces:**
- Consumes: `own_facts::directCallee(Operation*)`, `own_facts::calleeValue(Operation*)` from `lib/TypeScript/OwnershipFacts.h`; the virtual-candidate resolution `OwnershipSignaturePass` uses (move it into `OwnershipFacts.h` if it is private to that pass).
- Produces:
  - `class StackEscapeFacts { public: explicit StackEscapeFacts(mlir::ModuleOp); bool paramMayEscape(mlir::Operation *callOp, unsigned operandIndex) const; bool isRecursive(mlir::FunctionOpInterface) const; };`
  - A parameter (including `this`) "may escape" if its value, or a value derived from it, reaches any escape in spec §5.2 inside the callee body, or is passed to a parameter that may escape. Fixpoint over the module call graph, starting from "no escape" for defined functions and "escape" for every unknown callee, iterating until no function's facts change (mutual recursion included). Not the closed-world own facts: this is a property of the callee body, valid for any caller.
  - Conversions to `any`, interface, union or `unknown` are escapes in version 1 (stricter than spec §5.2 allows; our call - simpler and safe).

- [ ] **Step 1: Write the failing tests** (decision + run, all four models):
  - `ctor_local.ts`: the class from Task 3 with a `constructor(x, y)` that only assigns fields, built in a loop → `stack` (the constructor is a call with `this` as its first operand; `paramMayEscape` is false for it).
  - `borrow_uses.ts`: method call on the object, `===` between two locals, `instanceof`, `print(p.toString())` → `stack`. (`===` on two class values lowers to a raw pointer compare, `UnaryBinLogicalOrHelper.h:240-270`, `isNullableTypeNoUnion` branch: `PtrToInt` + `icmp`, no conversion to `any`; so the comparison is a borrow, not `converted`.)
  - `pass_to_reader.ts`: a module function `len(p: P) { return p.x * p.x + p.y * p.y; }` → `stack`.
  - `pass_to_keeper.ts`: a function storing its parameter in a global array → `heap: argument`.
  - `method_keeps_this.ts`: `register() { registry.push(this); }` → `heap: argument`.
  - `virtual_one_keeps.ts`: base `visit(x)` reads, override stores `x` in a field of a global → `heap: argument`.
  - `ctor_keeps_this.ts`: constructor stores `this` in a global → `heap: argument`.
  - `to_interface.ts`, `to_any.ts`: assigned to an interface / `any` local → `heap: converted`.
  - `pass_to_print.ts`: `print(p)` where `P` has `toString` → `stack`. `print` of a class instance is not a runtime call with the object: MLIRGen casts every `print` operand to `string` first, and for a class that is a call of `get_[Symbol.toStringTag]` or `toString` with the object as `this` (`MLIRGenCast.cpp:952-962`, `castToStringSpecialCases`); `PrintOpLowering` asserts every item is already a string pointer (`LowerToLLVM.cpp:162-195`). So the object only reaches its own method, which borrows `this`. The test pins this so a change in how `print` takes a class shows up as a test change.
  - `loop_carried.ts`: `let keep: P; for (...) { const p = new P(i); if (i == 3) keep = p; }` then reads `keep.x` after the loop and asserts 3 → `heap: loop-carried`. It lands here, not in Task 6, because following values through local slots (this task) is what would otherwise promote it and reuse its slot.
- [ ] **Step 2: Run, expect FAIL** (all `unclassified`).
- [ ] **Step 3: Implement** `StackEscapeFacts` and extend `isStackCandidate` to accept call operands whose `paramMayEscape` is false, comparisons, and `instanceof`/type-tag reads; follow the value through local `ts.Variable` slots (store into a non-captured local slot, loads from it) with the same rules applied to every load. With slot-following comes the loop-carried rule: an object whose value may be live at the allocation op's next execution (stored into a slot defined outside the innermost loop containing the allocation, or loaded after the loop's back edge) → `heap: loop-carried`. Loops are a CFG matter after the affine lowering: use `mlir::CFGLoopInfo` (`mlir/Analysis/CFGLoopInfo.h`, present in this LLVM) over the function's dominator tree to find the innermost loop containing the site and its back edges. A value that reaches a block argument (`cf.br ^bb(%p)`) is `unclassified` in this version.
- [ ] **Step 4: Run, expect PASS.**
- [ ] **Step 5: Commit** `Stack promotion: call and method escape facts`.

### Task 5: Objects that own heap memory under rc and own

**Files:**
- Modify: `include/TypeScript/TypeScriptOps.td` (`ReleaseContentsOp`), `include/TypeScript/LowerToLLVM/OwnershipRoutineLogic.h`, `lib/TypeScript/LowerToLLVM.cpp`, `lib/TypeScript/StackPromotionPass.cpp`
- Create tests: `fields_string.ts`, `fields_object.ts`, `field_overwrite.ts`, `return_field.ts`, `try_region.ts`, `conditional_alloc.ts`

**Interfaces:**
- Produces:
  - `def TypeScript_ReleaseContentsOp : TypeScript_Op<"ReleaseContents"> { let arguments = (ins TypeScript_AnyRefLike:$instance, TypeAttr:$storageType); }`
  - `void OwnershipRoutineLogic::emitReleaseContents(mlir::Type storageType, mlir::Value payloadPtr)` = `releaseFields(storageType, payloadPtr)` then zero the payload's owned fields (so a second call, or the next iteration's call, releases nothing). Lowers to nothing under gc and none.
- Rules (the pass, rc and own only):
  - insert `ts.ReleaseContents` immediately **before** each promoted allocation op (frees the previous iteration's contents) and once at function exit: the affine lowering gives every function a single return block (`LowerToAffineLoops.cpp:55-80`; each `ts.ReturnVal` stores into the return slot and branches there), so the exit insertion point is **immediately before the single `ts.ReturnInternal`** of that block, one per promoted site. The retain of a returned value happens at the `ts.ReturnVal` site, before the branch into the return block, so `return p.s` has retained the string before the object's contents are released; `return_field.ts` pins that order. A function whose return block is reached through a `cleanup` block (finally) has a try region and is excluded anyway;
  - **zero the promoted block's payload in the entry block as well** (in addition to the allocation-site zero-fill from Task 3). Without it, an allocation inside an `if` that was not taken leaves stack garbage in the slot, and the `ReleaseContents` at the return releases garbage pointers. Test: `conditional_alloc.ts` below. Confirm `GCPass` does not strip these memsets (it only strips memsets that zero a GC allocation, `GCPass.cpp:94-131`).
  - erase `ts.Retain`/`ts.Release`/`ts.RetainSlot`/`ts.ReleaseSlot` whose operand is the promoted object itself (they are no-ops on an immortal header; erasing them is the performance point);
  - a function containing a try/catch/finally region promotes nothing: reason `try-region` (version 1; unwind paths would need their own `ReleaseContents`).
  - `--verify-ownership` cannot check any of this (it runs before own's inference and before this pass; see Global Constraints). The checks are run tests under rc and own, where a missed release is a leak the test cannot see but a double release or a release of garbage is a wrong value or a crash; the Debug build's CRT asserts (`_CrtSetReportMode`, PR #276) catch heap corruption on the compile twin.

- [ ] **Step 1: Write the failing tests** (decision tests, and run tests under all four models):
  - `fields_string.ts`: class with a `string` field built from concatenation in a loop of 10000 → `stack`; output identical.
  - `fields_object.ts`: field holds a heap object → `stack`; the inner object stays heap (`heap: stored`).
  - `field_overwrite.ts`: field assigned three times from fresh strings (`p.s = "v" + i`), then return → `stack`; before each overwrite the current value is printed **directly** (`print(p.s)`), never read through a second local after the overwrite: under own a local that took `p.s` borrows a field the overwrite has just destroyed, so such a read would be wrong by own's own rules, not because of promotion. A double release shows up as a wrong value or a crash on the next concatenation.
  - `return_field.ts`: `function f(i: number) { const p = new P("v" + i); return p.s; }` called 1000 times, the results concatenated and printed → `stack`; run tests under rc and own especially (the string must be retained before `ReleaseContents` runs; see the insertion rule).
  - `try_region.ts`: a local inside a function with `try` → `heap: try-region`.
  - `conditional_alloc.ts`: `function f(c: boolean) { if (c) { const p = new P("s" + 1); print(p.s); } return 1; }` called with `false` then `true`, 1000 times each → `stack`; no crash under rc and own.
- [ ] **Step 2: Run, expect FAIL.**
- [ ] **Step 3: Implement** the op, its lowering, `emitReleaseContents`, the insertion rules, and drop the `ownsHeapMemory == false` restriction from Task 3.
- [ ] **Step 4: Run, expect PASS**; full suite under `-mm=rc` and `-mm=own`, compare with `-no-stack-promotion` (Task 8 step 1 recipe).
- [ ] **Step 5: Commit** `Stack promotion: objects that own memory, under rc and own`.

### Task 6: Limits, recursion, coroutines, throw

**Files:**
- Modify: `lib/TypeScript/StackPromotionPass.cpp`
- Create tests: `too_big.ts`, `frame_budget.ts`, `recursive.ts`, `async_local.ts`, `generator_local.ts`, `thrown.ts`

**Interfaces:**
- Consumes: `StackEscapeFacts::isRecursive`, `StackPromotionLimits`.
- Rules: storage size > `maxObjectBytes` → `too-big`; sites past `maxFrameBytes` in the function (the budget counts header + payload per site), sites inside loops kept first (a site is "inside a loop" when `mlir::CFGLoopInfo` puts its block in any loop), then source order → `frame-budget`; `isRecursive` → `recursive`; coroutines → `coroutine`; operand of `ts.Throw` → `thrown`.
- Coroutine detection (there is no "async" or "generator" attribute on a function to read): a function is a coroutine when (a) it is a `ts.Func` whose body contains an `async.execute` op - that is what an `await` expands to (`MLIRGenExpressions.cpp:447`), and the async lowering later outlines those regions; (b) it is any `func::FuncOp` - the only `func.func`s in the module at this point are the continuations the async lowering outlined, which is why own's inference nests on `func::FuncOp` as well as `ts.Func` (`transform.cpp:147-150`); (c) it is the `.step` method (`GENERATOR_STEP`, `Defines.h:175`) that `buildGeneratorWrapperDeclaration` (`MLIRGenFunctions.cpp:860-910`) makes out of a generator body - the body's locals that live across a `yield` are captured cells of the generator's state object, which the `captured` rule already rejects, and the rest sit in a method whose frame is re-entered per `next()`. Confirm the `.step` symbol name once with `--emit=mlir` on a generator before pinning the check.

- [ ] **Step 1: Write the failing tests**: one per rule above, each a decision test for the expected reason plus a run test; `too_big.ts` uses a class of 40 `number` fields (320 bytes) and also checks `--stack-promotion-max-object=512` gives `stack`; `frame_budget.ts` has 20 locals of 32-field classes (256-byte payload each, under the object limit; 20 of them exceed the 4096-byte frame budget); `recursive.ts` is a recursive tree sum building a temporary per call → `heap: recursive`, run under all four models at a depth that passes **today**: no stack-size setting exists anywhere in tslang, the runner or the test CMake (Windows' default main-thread stack is 1 MB), so first run the test on `main` with `--no-stack-promotion` at depth 10000 and lower the depth until it passes there; the promoted build must pass at the same depth. The test exists to show promotion adds no frame to a recursive function, not to find the platform's limit.
- [ ] **Step 2: Run, expect FAIL.**
- [ ] **Step 3: Implement** the rules.
- [ ] **Step 4: Run, expect PASS.**
- [ ] **Step 5: Commit** `Stack promotion: limits, recursion, coroutines, throw`.

### Task 7: Arrays and capture boxes

**Files:**
- Modify: `include/TypeScript/TypeScriptOps.td` (`stackAlloc` OptionalAttr<BoolAttr> on `CreateArrayOp`, `NewArrayOp`), `include/TypeScript/LowerToLLVM/ArrayLayout.h`, `lib/TypeScript/LowerToLLVM.cpp` (`CreateArrayOpLowering`, `NewArrayOpLowering`, `VariableOpLowering` for captured variables), `lib/TypeScript/StackPromotionPass.cpp`, `OwnershipRoutineLogic.h`
- Create tests: `array_local_fixed.ts`, `array_local_push.ts`, `array_returned.ts`, `array_dynamic_length.ts`, `closure_local.ts`, `closure_escapes.ts`

**Interfaces:**
- Array header `{data, length, capacity}` goes in a `StackBlockAlloc` block. The data block goes on the stack only if the length is a compile-time constant, its byte size fits the object limit, and no use may grow it (`push`, `unshift`, `splice`, `length` store, or any call argument whose `paramMayEscape` is true); otherwise the data block is heap (reason on the remark for the data: `growable` or `dynamic-size`) while the header can still be stack.
- `emitReleaseContents` for an array header: release elements `[0, length)`, then free the data block if it is a heap block (check its header: `emitIsMortal`), then zero the header.
- Capture box (`ts.Variable` with `CAPTURE_BOX_ATTR_NAME`): stack when every closure built over it (`ts.CreateBoundFunction` / `ts.CreateClosure` users) satisfies Task 4's rules as a value; reuse the closure-escape walk in `OwnershipInferencePass` (`followClosure`) by moving it into `OwnershipFacts.h` if needed.

- [ ] **Step 1: Write the failing tests**: `array_local_fixed.ts` (`const v = [x, y, z]` where `x`, `y`, `z` are the function's parameters, summed in a loop) → header `stack`, data `stack` - the elements must **not** be constants: an array literal of constants is a static literal global since #479 (no allocation, no site, no remark); `array_local_push.ts` → header `stack`, data `heap: growable`; `array_returned.ts` → `heap: returned`; `array_dynamic_length.ts` (`new Array(n)`) → data `heap: dynamic-size`; `closure_local.ts` (`[1,2,3].forEach` style with a local arrow capturing a counter, closure passed to a module function that only calls it) → box `stack`; `closure_escapes.ts` → box `heap: captured`.
- [ ] **Step 2: Run, expect FAIL.**
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run, expect PASS**; full suite under all four models.
- [ ] **Step 5: Commit** `Stack promotion: arrays and capture boxes`.

### Task 8: Whole-suite equivalence, benchmarks, docs

**Files:**
- Modify: `../docs/memory-models.md` (repository root: `I:\TypeScriptCompiler\docs\memory-models.md`), `test/tester/CMakeLists.txt`
- Create: `test/tester/stack-promotion/bench_small_class.ts`, `bench_vector_math.ts`

- [ ] **Step 1: Equivalence run**: full ctest suite under each of gc, rc, own, none, once as is and once with the runner's `-no-stack-promotion` (add a CMake option `TSLANG_TEST_NO_STACK_PROMOTION` that appends it to every `tslang_add_test`, same mechanism as the x86 twins, default OFF). The `nosp` script suffix from Task 2 is what makes the two runs really differ; check one generated `jitgcnosp.bat` contains `--no-stack-promotion` before trusting the comparison. Expected: identical pass/fail lists; compare against the known-failure list for the Debug suite.
- [ ] **Step 2: Teeth**: temporarily make `paramMayEscape` return `false` always; confirm `pass_to_keeper.ts` or `method_keeps_this.ts` fails at run time. A dangling stack read is not reliably wrong - the dead frame keeps its bytes until something reuses them - so the teeth test must **overwrite the frame first**: after the keeper function returns, call another module function that allocates two or three promoted objects of the same class with different field values and writes them, and only then read the kept object's fields and assert the original values. Revert; commit nothing from this step.
- [ ] **Step 3: Benchmarks**: `bench_small_class.ts` (10^7 iterations building and summing a 2-field class), `bench_vector_math.ts` (3-component vector temporaries), plus the ray tracer and n-body used in `../docs/memory-models.md`; `--emit=exe --opt --opt_level=3`, each model, with and without `--no-stack-promotion`; record time, peak working set, allocation count in a table.
- [ ] **Step 4: Docs**: a "Stack promotion" section in `../docs/memory-models.md`: what it does, that it is automatic and silent, the two limits and their flags, `--no-stack-promotion`, `--stack-promotion-report`, the version-1 exclusions (calls into other modules, async/generators, recursion, try regions), and the benchmark table.
- [ ] **Step 5: Commit** `Stack promotion: suite equivalence, benchmarks, docs`.
