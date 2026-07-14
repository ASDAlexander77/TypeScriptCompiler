# LowerToAffineLoops.cpp / LowerToLLVM.cpp — bug review (2026-07-13)

Static correctness audit of the two MLIR lowering passes
(`tslang/lib/TypeScript/LowerToAffineLoops.cpp`, TypeScript dialect → Affine/SCF/CF,
2325 lines; `tslang/lib/TypeScript/LowerToLLVM.cpp`, → LLVM IR, 6585 lines).
Neither file had uncommitted changes or recent history at review time (last
real touch was PR #198) — this was a full-file static audit, not a diff
review, done via 8 parallel finder passes + direct verification.

Status key: **CONFIRMED** = verified by reading the exact lines and comparing
against sibling code/behavior. **PLAUSIBLE** = strong evidence, not
independently reproduced with a running compiler.

## Fixed in this pass (items 1–3)

Verified: full ctest suite green after all three fixes (339/339 JIT, 342/342
AOT, `ctest --test-dir __build/tslang/windows-msbuild-2026-debug -C Debug -j8`).
Items 1–2 additionally verified by direct JIT execution of new test cases;
item 3 verified via `--emit=mlir-affine --mtriple=x86_64-pc-linux-gnu` IR
inspection (this Windows JIT host can't execute the Itanium exception ABI the
fix targets, since `isWindows` is derived from the target triple and this
build's SEH path never sets `cmpValue` at all — see the method note at the
bottom).

### 1. `ForOp` lowering drops loop-carried results for `for(;;)` — CONFIRMED, FIXED, VERIFIED

`LowerToAffineLoops.cpp:611-696` (`ForOpLowering`). `ValueRange args;` is
default-constructed and only assigned from `condOp.getArgs()` in the
`ConditionOp` branch (line 669). The `NoConditionOp` branch (infinite
`for(;;)`, lines 674-678) never assigns `args`, so `rewriter.replaceOp(forOp,
args)` at line 692 replaces all of `forOp`'s results with an empty list.

**Failure scenario:** a `for(;;) { ... }` loop with loop-carried values
(`ForOp`/`NoConditionOp` support `Variadic<AnyType>:$args`) hits the
`replaceOp` result-count mismatch — an assert/crash in `applyPartialConversion`,
or malformed IR in release builds.

**Fix:** assign `args = noCondOp.getArgs()` in the `else` branch, mirroring the
`ConditionOp` branch.

**Verified:** JIT-executed a `for(;;) { if (i>=5) break; sum += i; i++; }`
returning `sum` — correctly returns 10 (0+1+2+3+4), no crash.

### 2. `ArrayShiftOp` uses wrong GEP element type — CONFIRMED, FIXED, VERIFIED

`LowerToLLVM.cpp:2623-2624` (`ArrayShiftOpLowering`). The GEP computing the
"rest of the array" source pointer for the post-shift `MemoryMoveOp` was built
with `th.getI32Type()` as the GEP element type:

```cpp
auto offset1 =
    rewriter.create<LLVM::GEPOp>(loc, th.getPtrType(), th.getI32Type(), currentPtr, ValueRange{incSize});
```

Every sibling array-mutation pattern (`ArrayPushOp`, `ArrayUnshiftOp`,
`ArraySpliceOp`) and even the immediately preceding GEP in the same function
(line 2620, `offset0`) correctly use `llvmElementType` as the GEP element type.

**Failure scenario:** `Array<number>.shift()` (element type `f64`, 8 bytes)
computed the shift's source offset using a 4-byte stride instead of 8,
corrupting array contents or reading out of bounds for arrays of
wider-than-i32 elements.

**Fix:** changed the GEP element type from `th.getI32Type()` to
`llvmElementType`.

**Verified:** JIT-executed `[10.5,20.5,30.5,40.5].shift()` — `first === 10.5`,
remaining array is exactly `[20.5, 30.5, 40.5]` in order.

### 3. Mismatched typed `catch` silently falls through instead of rethrowing — CONFIRMED, FIXED, VERIFIED

`LowerToAffineLoops.cpp:1620-1633` (`TryOpLowering`, non-Windows/Itanium path).
When the RTTI comparison (`cmpValue`) for a typed `catch` clause is false (the
thrown value's type doesn't match the declared catch type), the conditional
branch's false edge went to `continuation` — ordinary post-try control flow —
instead of propagating/rethrowing the exception. The code carried its own
admission of this: `// TODO: when catch not matching - should go into result
(rethrow)`.

**Failure scenario:** `try { throw new TypeError() } catch (e: RangeError) {}`
on the Itanium/Linux path silently swallowed the mismatched exception and
continued normal execution instead of propagating it to an enclosing handler
or terminating the process.

**Fix:** on RTTI mismatch, branch to a new block that creates a `NullOp` +
`ThrowOp` (the existing "rethrow current exception" idiom used elsewhere in
this function, e.g. the finally-rethrow path at line ~1605) instead of
branching to `continuation`. If a parent `TryOp`'s landing pad exists
(`parentTryOpLandingPad`), `tsContext->unwind[rethrowOp]` is set so
`ThrowOpLowering` emits `ThrowUnwindOp` targeting it directly; otherwise
`ThrowOpLowering` falls back to `ThrowCallOp` (process-level rethrow). No
`EndCatchOp` is needed before the rethrow, matching the existing precedent's
comment ("we do not need EndCatch as throw will redirect execution anyway").

**Verified:** could not execute the Itanium ABI path natively on this Windows
JIT host (see method note), so verified via `--emit=mlir-affine
--mtriple=x86_64-pc-linux-gnu` IR dump instead of JIT execution:

- Single-level mismatch (`try{throw TypeErrorX}catch(e:RangeErrorX){}`, no
  parent try): false branch now emits `ts.Null` + `ts.ThrowCall` instead of
  branching to the continuation block.
- Nested case (`try{try{throw TypeErrorX}catch(e:RangeErrorX){}}catch(e:TypeErrorX){}`):
  inner mismatch emits `ts.Null` + `ts.ThrowUnwind(...)[^parentLandingPad]`,
  which correctly re-enters the outer try's landing pad, re-evaluates its RTTI
  compare, matches, and the outer catch runs — confirmed by reading the full
  block chain in the IR dump.
- Matching-catch happy path re-verified unaffected (JIT-executed, returns the
  caught value correctly).
- Full ctest suite (339 JIT + 342 AOT) green with this change in place.

## Second pass (2026-07-14) — items 4, 6, 7, 9, 10 fixed

Fixed on branch `fix/lowering-passes-bugs-2`, verified via full ctest suite
(688/688 passed, JIT + AOT) plus two research-agent-verified deep dives (items
4 and 7) that confirmed the failure mechanism with file:line evidence before
the fix was written.

4. **`LowerToAffineLoops.cpp:494,567,635,717`** (While/DoWhile/For/Label
   lowering) — FIXED. Confirmed via agent investigation: the old
   `visitorBreakContinue`/`walk()` recursed into nested loop/label/switch
   regions, and an unlabeled `break`/`continue` matched *any* enclosing
   scope's walk (`MLIRHelper::matchLabelOrNotSet` returns `true` whenever the
   op has no label of its own). Correctness depended entirely on undocumented
   MLIR dialect-conversion worklist ordering (last write to
   `tsContext->jumps[op]` wins, and the innermost loop's pattern happened to
   run last) — not a real guarantee, and confirmed to have zero explicit
   ordering protection in the code. Replaced with a shared
   `visitBreakContinueInScope` helper (manual recursive region walk, added
   just above `WhileOpLowering`) that threads independent
   `eligibleForUnlabeledBreak`/`eligibleForUnlabeledContinue` flags through
   the descent: both flags clear on crossing a nested While/DoWhile/For/Label
   boundary, and the break-only flag additionally clears on crossing a nested
   `SwitchOp` (switch owns unlabeled `break` but not `continue`, matching JS
   scoping). Labeled break/continue matching the scope's own label is still
   found no matter how many boundaries are crossed. Added
   `test_triple_nested_unlabeled` to `00break_continue.ts` (3 levels of
   nested `for`, innermost unlabeled `break`/`continue`), JIT-executed and
   checked by value.

6. **`LowerToLLVM.cpp:1271`** (`SymbolCallInternalOpLowering`) — FIXED.
   Replaced the debug-only `assert(llvmFuncType)` with
   `return rewriter.notifyMatchFailure(...)` when the callee symbol doesn't
   resolve to a known function-like op, matching the idiomatic `return
   failure()` pattern used elsewhere in this file.

7. **`LowerToLLVM.cpp:2043`** (`NewOpLowering`, stack-allocation path) —
   FIXED. Confirmed via agent investigation: `resultType = tch.convertType(newOp.getType())`
   converts a class type to an opaque LLVM pointer
   (`LowerToLLVM.cpp:5967-5969`, `ClassType → LLVM::LLVMPointerType`), so the
   old `ch.Alloca(resultType, 1)` allocated 8 bytes (`sizeof(ptr)`) on the
   stack regardless of the class's actual field layout — a stack
   under-allocation/overflow for any stack-`new`'d class with fields. Fixed
   by sizing off `tch.convertType(storageType)` (the actual converted
   `ClassStorageType`/struct), mirroring both the sibling heap-allocation path
   a few lines below and the `AllocaOpLowering` pattern earlier in the file.
   The now-unused `resultType` local was removed. No test added: the one
   test targeting this path (`00class_stack.ts`) has the stack-alloc call
   form commented out with a pre-existing, unrelated `TODO: ERROR can't
   create class with vtable on stack` blocker.

9. **`LowerToAffineLoops.cpp:1666`** (`TryOpLowering`, `linuxHasCleanups`) —
   FIXED. A cleanup-only try (no catch, no finally — e.g. the `TryOp`
   synthesized for a `using` declaration) hit the `linuxHasCleanups` block
   with neither `catchHasOps` nor `finallyHasOps`, so `cleanupBlockLast`'s
   `ResultOp` terminator was erased with nothing replacing it — malformed IR.
   Fixed by adding the missing branch: build a Linux cleanup landing pad
   (`LandingPadOp` cleanup=true + `BeginCleanupOp`) at the start of
   `cleanupBlock` and replace the terminator with `EndCleanupOp` (targeting
   `parentTryOpLandingPad` if set), mirroring the existing Windows
   unconditional cleanup-landing-pad setup and the Linux
   `finallyHasOps`-with-no-catch precedent already in this function.
   `EndCleanupOp` always lowers to `LLVM::ResumeOp` (confirmed by reading
   `EndCleanupOpLowering`), so this only affects the exception-unwind path,
   consistent with Itanium cleanup-landingpad semantics. Exercised indirectly
   by the existing `00disposable.ts`/`01disposable.ts`/`02disposable.ts`
   JIT+AOT tests (a bare `using` block with no explicit try/catch/finally is
   exactly this code path), all passing.

10. **`LowerToAffineLoops.cpp:763`** (`LabelOpLowering`) and the same pattern
    in `SwitchOpLowering` (~854) — FIXED. Both `assert(false)` fallbacks
    (silent no-op in release builds, leaving a malformed CFG) replaced with
    `return rewriter.notifyMatchFailure(...)`.

## Third pass (2026-07-14) — item 5 fixed, plus two bugs found while testing it

Fixed on branch `fix/lowering-passes-bugs-3`, verified via full ctest suite
(690/690 passed, JIT + AOT).

5. **`LowerToAffineLoops.cpp:1351,1358,1370`** (`TryOpLowering`, `finally`
   region cloning) — FIXED. Confirmed: `cloneRegionBefore` produces brand-new
   `Operation*` instances (verified against MLIR's `Region::cloneInto`/
   `IRMapping` machinery), so `tsContext->jumps`/`parentTryOp` entries
   populated by walking the *original* `finally` region (line ~1260, and by
   an enclosing loop's break/continue-scope walk that runs before this try
   is lowered) were silently absent for both of the two cloned copies
   (normal-exit and return-cleanup) — only the third, *inlined* copy (which
   preserves original `Operation*` identity) kept valid wiring. Fixed by
   passing an explicit `mlir::IRMapping` to both `cloneRegionBefore` calls
   and re-propagating any `jumps`/`parentTryOp` entry found for an old op
   onto its corresponding new op via the mapping's `getOperationMap()`,
   immediately after each clone.

   While building a regression test for this, found and fixed two more bugs
   in the same area:

   - **Pre-existing, unrelated to this fix**: a `try` nested directly inside
     a `finally` block failed to compile at all (`error: failed to legalize
     operation 'ts.Try' that was explicitly marked illegal`), regardless of
     break/continue. Root cause: `cloneRegionBefore` correctly notifies the
     conversion driver of the cloned nested `TryOp`, which then legalizes it
     *recursively while the outer TryOp's own pattern application is still on
     the stack* (`legalizePatternCreatedOperations`) — but MLIR's
     `OperationLegalizer::canApplyPattern` refuses to reapply the same shared
     `Pattern` instance recursively unless it opts in via
     `setHasBoundedRewriteRecursion()`, which no `TsPattern` did. Fixed by
     giving `TryOpLowering` its own constructor that calls
     `setHasBoundedRewriteRecursion()` — safe here since each recursive
     application strictly consumes one `TryOp`.
   - **Pre-existing, separate, NOT fixed — tracked here for follow-up**:
     `break`/`continue` inside a `try` *body* (as opposed to directly inside
     `finally`) does not run the enclosing `finally` block before jumping —
     `jumps[breakOp]`/`jumps[continueOp]` are set by the loop's own
     break/continue-scope walk (`visitBreakContinueInScope`) to point
     *directly* at the loop's continuation/increment block, with no
     awareness that a `try/finally` sits in between. Reproduces a JIT crash
     (`0x80000003`) on unmodified `main` (i.e. predates all three passes of
     this review) with a minimal `for { try { if (...) continue; } finally
     { ... } }`. Not fixed in this pass — would require threading
     finally-block awareness into the jump-target logic for every loop
     lowering, larger in scope than this pass's fixes. The new regression
     test (`00try_finally_break_continue.ts`) deliberately only covers
     break/continue placed directly *inside* `finally` (which does work
     correctly) to avoid this separate gap.

   Added `00try_finally_break_continue.ts` (break in finally, continue in
   finally, nested try/catch in finally, each inside a loop, each run 3-5
   times to catch wrong-copy-used-once-vs-every-iteration bugs), JIT +
   AOT green.

## Remaining known issues (not yet fixed, tracked here for follow-up)

5b. **`LowerToAffineLoops.cpp`** (loop `jumps[]` target computation) —
    `break`/`continue` inside a `try` body does not run an enclosing
    `finally` block before jumping to the loop's continuation/increment.
    See "Third pass" above for the repro and root cause. Needs the
    break/continue lowering (or the `TryOp` lowering) to detect that the
    jump target crosses a `finally` boundary and route through it first.

8. **`LowerToLLVM.cpp:4159`** (`LandingPadOpLowering`, Windows path) —
   cleanup-only branch reuses the typed-catch filter value instead of an
   empty/undef cleanup clause; the code has its own `// BUG: in LLVM landing
   pad is not fully implemented` comment. PLAUSIBLE, not yet fixed —
   deliberately left alone in the second pass: this is deep Windows SEH
   landing-pad construction with the original author's own admission it's
   incomplete, and a wrong guess here risks a worse regression than the
   status quo. Needs someone with direct SEH/landingpad-clause expertise, or
   a way to single-step actual Windows exception dispatch through it.

## Other observations (not correctness bugs, lower priority)

- **Duplication**: the `visitorBreakContinue` lambda is copy-pasted verbatim
  in `WhileOpLowering`, `DoWhileOpLowering`, `ForOpLowering`, `LabelOpLowering`
  (~14 lines × 4). Six near-identical accessor-lowering structs in
  `LowerToAffineLoops.cpp` (905-1205, `ThisAccessorOp` through
  `BoundIndirectIndexAccessorOp`) could collapse to one templated helper.
  `CallHybridInternalOpLowering`/`InvokeHybridOpLowering` in `LowerToLLVM.cpp`
  duplicate ~100 lines of this-pointer/vtable dispatch logic.
- **Altitude**: GC-root registration (`LowerToLLVM.cpp:1911`) and
  side-effect detection for global initializers (`LowerToLLVM.cpp:3492`, with
  a commented-out `MemoryEffectOpInterface` version right below it) both
  enumerate specific op/type kinds by hand instead of querying a trait —
  same class of issue as the `jit-globals-not-gc-roots` GC memory. A new type
  or op added elsewhere silently isn't covered.
- No `CLAUDE.md` exists anywhere in this repo, so no conventions check
  applied.

## Method note

MLIR's dialect-conversion driver pre-collects a pre-order worklist (verified
against `DialectConversion.cpp`), so parent ops always convert before their
*original* nested children — ruling out a naive "child converted before
parent sets up its context" race for `tsContext` side-tables. The real hazard
in this file is region **cloning** (item 5): clones are new `Operation*`
instances not covered by side-table entries populated before the clone was
made.
