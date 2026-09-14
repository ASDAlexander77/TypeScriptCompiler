# Optimization: direct call when a hybrid function has a statically-null `this`

Status: implemented as option D below (`SimplifyIndirectCallWithKnownCallee`, test `00funcs_hybrid_null_this.ts`).

## Problem

Casting a plain function value (e.g. an address from `GetProcAddress`) to a function type and
calling it produces a runtime `this` check whose outcome is known at compile time.

Repro (`--emit=llvm --opt_level=0`):

```ts
declare function GetProcAddress(library: Opaque, functionName: Opaque): Opaque;
const f = GetProcAddress(module, "wglCreateContext");
let hdc: Opaque = 0;
const r = (<(p: Opaque) => Opaque>f)(hdc);
```

MLIR:

```mlir
%162 = "ts.Cast"(%145) : (!ts.opaque) -> !ts.func<!ts.opaque, !ts.opaque, false>
%163 = "ts.Cast"(%162) : (!ts.func<...>) -> !ts.hybrid_func<!ts.opaque, !ts.opaque, false>
%165 = "ts.CallIndirect"(%163, %164) : (!ts.hybrid_func<...>, !ts.opaque) -> !ts.opaque
```

LLVM IR:

```llvm
%39 = insertvalue { ptr, ptr, ptr } undef, ptr %24, 0
%40 = insertvalue { ptr, ptr, ptr } %39, ptr null, 1     ; this = null (constant)
%41 = insertvalue { ptr, ptr, ptr } %40, ptr null, 2
%42 = load ptr, ptr %1
%43 = extractvalue { ptr, ptr, ptr } %41, 1
%44 = ptrtoint ptr %43 to i64
%45 = icmp ne i64 %44, 0                                 ; always false
br i1 %45, label %46, label %49
46: %48 = call ptr %47(ptr %43, ptr %42)                 ; dead
49: %51 = call ptr %50(ptr %42)                          ; the only reachable call
52: %53 = phi ptr [ %51, %49 ], [ %48, %46 ]
```

The result is correct, but every such call gets a dead branch, a phi and an extra call site.
At `--opt` LLVM probably folds it. At `--opt_level=0` / `--di` it stays, which makes debug
builds bigger and stepping through the code confusing.

## Where it comes from

1. `MLIRGenCast.cpp` (~L1169, "opaque to hybrid func"): `Opaque -> FunctionType -> HybridFunctionType`.
   A function type in a cast expression becomes `HybridFunctionType`.
2. `include/TypeScript/LowerToLLVM/CastLogicHelper.h` (~L273): `FunctionType -> HybridFunctionType`
   is lowered to `CreateBoundFunctionOp(NullOp, func)`, so `this` is a literal null.
   The same applies to `NullType -> HybridFunctionType` (~L313).
3. `lib/TypeScript/LowerToLLVM.cpp`:
   - `CallInternalOpLowering` (~L1720) rewrites any call whose callee is `HybridFunctionType`
     into `CallHybridInternalOp`.
   - `CallHybridInternalOpLowering` (~L1751) always emits `GetThisOp` -> `Cast` to boolean ->
     `conditionalBlocksLowering` with a with-this call and a no-this call.
   - `InvokeOpLowering` (~L1865) and `InvokeHybridOpLowering` (~L1902) have the same shape for
     calls inside `try`.

## Proposed fix (pick one; A is the smallest)

### A. Fold at call lowering when the callee was built from a plain function

In `CallInternalOpLowering`, before rewriting to `CallHybridInternalOp`, look at the *original*
callee `op.getOperand(0)` (not `transformed`):

- If it is defined by `mlir_ts::CastOp` whose input type is `mlir_ts::FunctionType`, the value
  has no `this`. Emit a plain indirect `LLVM::CallOp` using the cast's input, converted with
  `rewriter.getRemappedValue(castOp.getIn())` or a `DialectCastOp`, and skip the hybrid path.
- If it is defined by `mlir_ts::CreateBoundFunctionOp` whose `this` operand comes from
  `mlir_ts::NullOp`, do the same with the function operand.

Apply the same check in `InvokeOpLowering` before it creates `InvokeHybridOp`.

Watch out: this is a dialect conversion, so the defining op may already be scheduled for
replacement. Read the original op's operands and map them through the rewriter, and don't rely
on the op still being there after the call. If the cast result has no other uses it becomes dead
and should be erased by DCE, or explicitly.

### B. Only the no-this branch in `CallHybridInternalOpLowering`

Keep the rewrite to `CallHybridInternalOp`, but in `CallHybridInternalOpLowering` (and
`InvokeHybridOpLowering`) check whether `this` is provably null, using the same defining-op test
as in A. If it is, emit only the body of the "no this" lambda (L1824-L1838) and skip
`conditionalBlocksLowering`.

### C. At MLIR generation

In `MLIRGenCast.cpp`, when the target of an `Opaque`/`FunctionType` cast is a function type that
has no `this`, produce `FunctionType` instead of `HybridFunctionType`. This fixes it earliest,
but it changes what type the expression has. Check assignment compatibility: a variable of that
function type may later receive a bound method, and that is exactly why `HybridFunctionType`
exists. Only do this if the cast result is consumed directly by a call, so A/B is safer.

### D. Fold in the `CallIndirectOp` canonicalizer (implemented)

`SimplifyIndirectCallWithKnownCallee` (`lib/TypeScript/TypeScriptOps.cpp`) already folds indirect
calls with a known callee. It now also handles a callee defined by `ts.Cast` from `FunctionType` to
`HybridFunctionType`: the call's callee operand is replaced by the cast's input, and the cast is
erased if it has no other uses.

Why here rather than A/B/C:

- `mlir::createCanonicalizerPass()` runs unconditionally (not gated by `--opt`), so the fold applies
  at `--opt_level=0` / `--di`.
- It runs before `LowerToAffine` turns calls inside `try` into `ts.Invoke`, so one pattern covers
  both `CallInternalOp` and `InvokeOp`; the four LLVM lowerings are untouched.
- It works on plain TS dialect ops, so there is no dialect-conversion replacement hazard.
- `CallIndirectOp` accepts a `FunctionType` callee, with the same inputs and results.

Limits: only a cast consumed directly by the call is folded. A function-typed variable
(`const g = <fn>f; g()`) keeps the runtime check, which is correct since such a variable may hold a
bound method. The `null -> HybridFunctionType` case is only materialized during LLVM lowering and is
not folded (calling it crashes anyway).

Before the fix, the dead branch even called the plain function with an extra `ptr null` argument
(`call double @add1(ptr null, double 4.1e+01)`), which was only harmless because it was unreachable.

## Expected output after the fix

```llvm
%42 = load ptr, ptr %1
%51 = call ptr %24(ptr %42)
```

## Tests to add / run

- The repro above: check with `--emit=llvm --opt_level=0` that there is no `icmp`/`br`/`phi` around the call.
- The same call inside `try { ... }` (goes through `InvokeOp` / `InvokeHybridOp`).
- Regression: a real bound method stored in a variable of function type, then called. It must
  still pass `this`, e.g.
  `class C { x = 1; m() { return this.x; } } const c = new C(); const g: () => number = c.m; g();`
- Regression: a `null` value cast to a function type, then called through the hybrid path
  (CastLogicHelper.h ~L313).
- Run the existing test suite, `-mm=gc` and `-mm=rc` at least.
