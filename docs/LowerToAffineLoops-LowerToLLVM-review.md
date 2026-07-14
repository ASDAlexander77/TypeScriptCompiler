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

## Remaining known issues (not yet fixed, tracked here for follow-up)

4. **`LowerToAffineLoops.cpp:494,567,635,717`** (While/DoWhile/For/Label
   lowering) — the `visitorBreakContinue` walk recurses into nested
   loop/label regions, not just the immediate body. An unlabeled
   `break`/`continue` inside a nested loop can have its `tsContext->jumps[op]`
   entry set by either the inner or outer loop's walk, with the last one to
   run winning — order-dependent, so a nested unlabeled `break` could jump out
   of both loops instead of just the inner one. PLAUSIBLE, not yet fixed.

5. **`LowerToAffineLoops.cpp:1351,1358,1370`** (`TryOpLowering`, `finally`
   region cloning) — the `finally` region is cloned twice (normal-exit and
   return-cleanup copies) *after* the `parentTryOp`/`jumps` side-tables were
   populated by walking only the original region. A nested `try` or a
   `break`/`continue` inside a `finally` block loses correct wiring in the
   cloned copies. PLAUSIBLE, not yet fixed — needs a test case (`grep` of
   `test/` found no try/finally-nesting coverage).

6. **`LowerToLLVM.cpp:1271`** (`SymbolCallInternalOpLowering`) — `llvmFuncType`
   can remain null if `moduleOp.lookupSymbol` fails to resolve or the symbol is
   an unexpected kind; only guarded by a debug-only `assert`, no `return
   failure()` guard. PLAUSIBLE, not yet fixed.

7. **`LowerToLLVM.cpp:2043`** (`NewOpLowering`, stack-allocation path) —
   allocates using `resultType` (converted pointer type) instead of
   `storageType` (actual class layout) used by the sibling heap-allocation
   path; asymmetry could under-allocate the stack slot for `new` with
   `getStackAlloc()` set. PLAUSIBLE, needs confirming what `resultType`
   actually converts to for class types before treating as certain.

8. **`LowerToLLVM.cpp:4159`** (`LandingPadOpLowering`, Windows path) —
   cleanup-only branch reuses the typed-catch filter value instead of an
   empty/undef cleanup clause; the code has its own `// BUG: in LLVM landing
   pad is not fully implemented` comment. PLAUSIBLE, not yet fixed.

9. **`LowerToAffineLoops.cpp:1660`** (`TryOpLowering`, `linuxHasCleanups`) —
   merges `catchesBlock`/`finallyBlock` into `cleanupBlockLast` only when
   `catchHasOps`/`finallyHasOps`, but `linuxHasCleanups` doesn't imply either;
   a cleanup-only try with no catch/finally content could erase
   `cleanupBlockLast`'s terminator with nothing replacing it. PLAUSIBLE, not
   yet fixed.

10. **`LowerToAffineLoops.cpp:763`** (`LabelOpLowering`) — falls back to
    `assert(false)` if the label region's terminator block isn't a `MergeOp`;
    in release builds (asserts stripped) this is a silent no-op leaving a
    malformed CFG if the label region's live last block doesn't end in
    `MergeOp` (e.g. dead code after a break-only path). PLAUSIBLE, not yet
    fixed. Same pattern exists in `SwitchOpLowering` at line ~854.

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
