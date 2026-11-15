# Production readiness: Phase 0 results

Working notes for Phase 0 of the [production-readiness plan](production-readiness-plan.md).
Compiler under test: `main` at `3392fef6`, 2026-11-13, Windows x64.

## 1. Debug-built compiler suite (Windows)

`ctest -C Debug -j 8 --timeout 300` on a Debug build of `3392fef6` (no fixes from this document):
**60 of 7,085 tests failed**, none by timeout, in 28 minutes. The library environment variables
(`DEFAULT_LIB_PATH` and the others) were cleared, as in CI, and other builds ran on the machine at
the same time. The Release build of the same commit passes all of them.

The 60 failures fall into 14 groups. None has been investigated yet.

| Group | Tests | Variants failing |
| --- | --- | --- |
| `00parse_number_range` | 6 | exe, JIT, rc and none corpus |
| `00class_to_interface_null` | 8 | exe, JIT, rc and none corpus, own |
| `00throw_bool_bigint` | 6 | exe, JIT, rc and none corpus |
| `00throw_f32` | 6 | exe, JIT, rc and none corpus |
| `00cast_to_f32` | 6 | exe, JIT, rc and none corpus |
| `value_missing_niche`, `value_missing_niche_one_field`, `value_missing_ops` | 18 | exe and JIT under gc, rc, none |
| `gc-opt-uncaught-throw_*` (string, class, number, number_shortest) | 4 | exe |
| `symbol-value-error` | 1 | exe |
| `foreign-thread-gc` | 1 | exe |
| `atomic-rc` | 1 | exe |
| `gc-defaultlib-collector` | 1 | exe; may be the cleared library path |
| `borrow-facts-defaultlib` | 1 | may be the cleared library path |
| `test-bindgen-cli` | 1 | |

`00throw_f32`, `00throw_bool_bigint`, `00cast_to_f32`, `gc-opt-uncaught-throw_*`, `test-bindgen-cli`
and `gc-defaultlib-collector` were already on the 2026-10-01 list. The rest are new since then.
The full log is not kept in the repository.

## 2. Fatal-error sites

A search for `llvm_unreachable` and `report_fatal_error` under `tslang/` finds 42 matches in 23
files. 8 are inside comments, which leaves 34 live sites. This is a first-pass sort by kind; none
has been proven reachable or dead yet.

In a Release build without assertions, LLVM normally compiles `llvm_unreachable` to an optimizer
hint, not a trap. If the LLVM used here is built that way, a reachable site in groups B and C does
not crash the released compiler: it continues with undefined behaviour. That is worse than a crash
and is the reason to triage these. Not verified for the prebuilt LLVM package.

### A. Deliberate guards that also fire in Release (8)

These use `report_fatal_error` with a message. They are internal-consistency or environment checks.

| Site | Message |
| --- | --- |
| `tslang/tslang/vscode.cpp:209` | error emitting data to file |
| `tslang/include/TypeScript/TypeScriptPassContext.h:35` | lowering-context entry left by a previous function |
| `tslang/lib/TypeScriptExceptionPass/Win32ExceptionPass.cpp:378` | no end found for an exception handling region |
| `tslang/lib/TypeScriptExceptionPass/WasmExceptionPass.cpp:307` | a landing pad reached other than by an invoke |
| `tslang/lib/TypeScriptExceptionPass/CxxThrowCallingConvPass.cpp:173` | `_CxxThrowException` used other than as a direct callee |
| `tslang/include/TypeScript/MLIRLogic/MLIRRTTIHelperVCWin32.h:560` | absolute EH references need 32-bit pointers |
| `tslang/include/TypeScript/LowerToLLVM/LLVMRTTIHelperVCWin32.h:352` | absolute EH references need 32-bit pointers |
| `tslang/include/TypeScript/LowerToLLVM/TypeHelper.h:97` | `getSizeType()` needs the target |

Proposed handling: keep them, but `vscode.cpp:209` is a user-reachable I/O failure (read-only folder,
full disk) and should be an ordinary error with exit code 1.

### B. Defaults of switches meant to be exhaustive (6)

| Site | Message |
| --- | --- |
| `tslang/tslang/TextDiagnostic.cpp:33`, `:55` | invalid diagnostic type |
| `tslang/lib/TypeScript/MLIRGenStatements.cpp:82` | unknown body type |
| `tslang/lib/TypeScript/MLIRGenStatements.cpp:495` | unknown statement type |
| `tslang/lib/TypeScript/MLIRGenImpl.h:4130`, `:4161` | unknown statement |

Proposed handling: compare the parser's statement kinds with the handled list. Any statement kind
the parser can produce and the switch does not handle is reachable from user code.

### C. Type and lowering gaps, reachability unknown (20)

| Site | Message |
| --- | --- |
| `tslang/include/TypeScript/MLIRLogic/MLIRTypeHelper.h:3845` | wrong type |
| `tslang/include/TypeScript/LowerToLLVM/CastLogicHelper.h:1046`, `:1253` | review usage |
| `tslang/include/TypeScript/LowerToLLVM/CodeLogicHelper.h:323` | return op. can't be found |
| `tslang/include/TypeScript/LowerToLLVM/LLVMCodeHelper.h:492` | should be processed in constant with denseattr value |
| `tslang/include/TypeScript/LowerToLLVM/LLVMCodeHelper.h:523`, `:540` | ConstArrayType / ConstTupleType must not be used in array |
| `tslang/include/TypeScript/LowerToLLVM/LLVMCodeHelper.h:762` | not used |
| `tslang/include/TypeScript/LowerToLLVM/LLVMDebugInfo.h:56` | not implemented |
| `tslang/include/TypeScript/LowerToLLVM/UndefLogicHelper.h:200` | bug |
| `tslang/include/TypeScript/LowerToLLVM/UnaryBinLogicalOrHelper.h:47` | not implemented (after an `emitError`) |
| `tslang/lib/TypeScript/LowerToLLVM.cpp:4170` | type mismatch |
| `tslang/lib/TypeScript/LowerToLLVM.cpp:4336`, `:4481`, `:4723` | cast must happen earlier |
| `tslang/lib/TypeScript/LowerToLLVM.cpp:9120` | type usage (IntersectionType) is not implemented |
| `tslang/lib/TypeScript/LowerToAffineLoops.cpp:2609` | missing catch data |
| `tslang/lib/TypeScript/MLIRGenImpl.h:1599` | can't find local variable |
| `tslang/lib/TypeScript/MLIRGenImpl.h:5194` | type predicate can't find parameter index |
| `tslang/lib/TypeScript/MLIRGenImpl.h:10709` | capture or this ref is not resolved |

Proposed handling, per site: a short repro attempt on the Debug compiler, then a caller trace if
the repro misses. Reachable sites become `emitError` plus failure. `UnaryBinLogicalOrHelper.h:47`
already emits an error first, so only the `llvm_unreachable` after it needs to go.

### D. Inside comments (8)

`Win32ExceptionPass.cpp:331`, `:1185`; `MLIRTypeHelper.h:1093`; `CastLogicHelper.h:1265`;
`CodeLogicHelper.h:310`; `MLIRGenTypes.cpp:3116`; `MLIRGenFunctions.cpp:2424`;
`MLIRGenGenerics.cpp:1562`. No action, other than deleting the dead comments when the file is next
touched.

## 3. Known items not in the tracker

Candidates came from earlier session notes. Each one marked "reproduced" was re-run today on the
Release compiler (JIT, default options). The rest still need a repro before an issue is filed.

### Reproduced today

| # | Item | Result today | Kind |
| --- | --- | --- | --- |
| 1 | A class field typed `(() => number) \| null` cannot be assigned a function: `this.f = t` | `error: invalid cast from () => number to () => number \| null`, then `not a function to call` | Valid TypeScript rejected |
| 2 | `n = b` and `s = b`, where `b` is a `boolean`, `n` a `number`, `s` a `string` | Compiles and prints `1 true`. `tsc` rejects both (TS2322) | Invalid TypeScript accepted |
| 3 | `Symbol("key")` at run time | `error: not a function to call` | Unsupported feature, unclear message |
| 4 | An optional interface field (`label?: string`) implemented by a get/set pair | `error: ... an optional field can't be filled by an accessor` | Unsupported feature, clear message |

Checked and no longer a bug: an abstract field overridden by a get/set pair, read through the base
class, now returns the right value.

Items 1 and 2 are fixed on branch `fix-fn-or-null-and-boolean-assign` (uncommitted):

- **Item 1.** `(() => T) | null` is stored as the function value itself, with no tag: null is a null
  code pointer in its first word. A function is now stored in it, `if (f)` and `f !== null` narrow
  it, and it is called, under gc, rc, none and own. Test: `00function_or_null.ts`.
- **Item 2.** A boolean given to a number or a string, or a number or string given to a boolean, is
  an error in assignments, declarations and arguments. Tests: `assign_boolean_to_number_error.ts`,
  `assign_boolean_to_string_error.ts`, `assign_number_to_boolean_error.ts`,
  `argument_boolean_to_number_error.ts`.

Found while fixing them:

| # | Item | Result today | Kind |
| --- | --- | --- | --- |
| 20 | `-mm=own`: a closure that captures the object it is stored in (`c.onTick = () => { c.seen++ }`) compiled, then crashed with an access violation once the field was cleared. **Fixed on the same branch**: the closure borrows the object, as one capturing `this` in a method does. Test: `00closure_back_reference_local.ts` | Was a crash at run time | Fixed |
| 23 | `-mm=own`: a closure stored in a field owns a *different* object it captured, and frees it when the field is cleared (`c.onTick = null`). A later use of that object compiled and read freed memory. **Fixed on the same branch** by the owner's ruling (2026-11-13): any use of the object after the capture is an error, as a use of a captured variable already was. `00tslang_issues_31_33.ts` lost one assertion that read a captured object. Test: `own_escaping_capture_used_after_error.ts` | Was a silent use-after-free | Fixed |
| 21 | A `fn \| null` value can be called without a null check, as a `C \| null` object's methods can. `tsc` rejects it (TS2721) | Compiles | Invalid TypeScript accepted |
| 22 | The default library has not been compiled against the boolean check | Not run | To verify |

### Not yet re-verified

| # | Item | Source | Kind |
| --- | --- | --- | --- |
| 5 | A closure stored in a variable of function type, then passed to a `@cdecl` parameter, loses its captures at run time | `TSLANG_ISSUES.md` | Silent miscompile |
| 6 | `-mm=rc`: fuzz seed 4487 crashes (argument assignment plus try/finally on a global field) | notes, 2026-11-10 | Crash |
| 7 | `-mm=rc`: seeds `review6r1__v1` / `v1b` crash through `f2 as any as (o: Op, ...)` | notes, 2026-10-31 | Crash |
| 8 | Memory-model fuzzer findings R1-R10, O4 and G1 (a gc closure bug) | notes, 2026-10-23 | Mixed; several were fixed in later rounds |
| 9 | With a class declared later in the file: built-in `arr.filter(arrow)` under `--no-default-lib` gives "the condition has no value" | notes | Valid TypeScript rejected |
| 10 | With a class declared later in the file: `assert(g.particles[0].duration == 3.0)` gives "Can't resolve property 'duration' of type any" | notes | Valid TypeScript rejected |
| 11 | `<Walker>unknownVal` crashes under gc; a cast from `any` to an object type followed by a field write crashes | notes, 2026-10-28 | Crash; may be fixed by #723 |
| 12 | `-mm=own`: a union local narrowed by `instanceof` and walked twice gives a spurious use-after-move | notes, 2026-10-28 | Valid program rejected |
| 13 | `-mm=own`: lazy iterator helpers, `const seq = fibonacci().map(f)`, are refused | notes | Unsupported |
| 14 | `-mm=own`: `f && f(x)` with a callback, and a callback that is stored or captured | notes | Unsupported |
| 15 | `-mm=own`: `const o = {w: (x: number) => x}; apply(o)` | notes | Unsupported |
| 16 | Two-pass lambda parameter inference | notes, 2026-10-28 | Unsupported |
| 17 | `check_declarations` test is flaky: union member order depends on pointer values | notes, 2026-11-10 | Nondeterministic compiler output |
| 18 | `test-compile-00-for-await-yield` (Windows) and `test-compile-00-for-await` (Linux) fail now and then on hosted runners | notes | Flaky test |
| 19 | gc without stack promotion is twice as slow on the ray tracer (136 ms against 71 ms), unexplained | notes, 2026-11-10 | Performance |

Item 17 matters beyond the test: output that depends on pointer values means builds are not
reproducible.

## 4. Uncommitted edits on `main`

| File | What the edit is | Suggested |
| --- | --- | --- |
| `tslang/CMakeLists.txt` | Adds an opt-in `TSLANG_ENABLE_LTCG` option, off by default | Owner's call; earlier measurement found link-time code generation gave no size win |
| `docs/how/wasm/test/test_wasm.ps1` | Replaces hard-coded `C:\dev\...` paths with relative ones | Looks safe to commit |

Neither was touched.
