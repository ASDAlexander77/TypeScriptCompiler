# `const`/`let` storage model: design and fix

Branch: `const-let-storage-rework`. Status: **implemented and verified** (350/350
JIT + 354/354 compile, 0 failures).

## 1. The bug this addresses

Three related generator bugs (see prior memory: `generator-const-manual-next-bug`,
`object-literal-generator-method-type-bug`, `generator-param-value-semantics-bug`)
all trace back to one design assumption in `registerVariable`:

> `const` bindings never need real storage (an alloca); only `let`/`var` do.

That assumption is correct for `const` bindings that hold plain values (numbers,
booleans, plain object-literal tuples with no methods). It is **wrong** for
`const` bindings that hold a value-typed aggregate with internal mutable state
accessed through a bound method — today, in practice, that means **generator
wrapper objects** (`{ step, next() {...} }`, built by
`buildGeneratorWrapperDeclaration` in `MLIRGenFunctions.cpp`). Calling `.next()`
mutates the `step` field in place; if the binding has no address, every call
site that needs a ref to mutate through has to invent one from scratch, seeded
from the pristine, never-mutated original value — so state never advances.

## 2. Why this is narrower than "rework const/let for all types"

Initial framing (see conversation) assumed the fix was general: decide storage
need by "does the type have mutable identity" across all TS constructs. A
survey of how each construct actually lowers to LLVM shows this is unnecessary
for everything except tuples:

| TS construct | MLIR type | LLVM representation | Pointer-like? | Mutation-through-alias works today regardless of const/let? |
|---|---|---|---|---|
| `class` instance | `!ts.class<...>` | `NewOp` → alloca/heap, type maps to `LLVMPointerType` | Yes | Yes — SSA value already is a pointer |
| Array (`T[]`) | `!ts.array<T>` | `{ptr, len}` header over heap buffer | Yes | Yes |
| Const/fixed array | `!ts.const_array<T,N>` | `LLVMPointerType` | Yes | Yes |
| Object needing bound methods | `!ts.object<storage>` | `LLVMPointerType` to named struct | Yes | Yes |
| Plain object literal `{a,b}` (no methods) | `!ts.tuple<...>` / `!ts.const_tuple<...>` | bare LLVM struct **by value** | No | N/A — no mutation surface, so it's fine that it's value-like |
| **Generator wrapper `{step, next()}`** | `!ts.tuple<...>` / `!ts.const_tuple<...>` | bare LLVM struct **by value** | **No** | **No — this is the bug** |

Classes, arrays, and true "object" types are already pointer-like at the LLVM
level, so aliasing/mutation already works no matter what `registerVariable`
does with `const`. The only gap is: **a value-typed tuple aggregate that
contains a bound-method field** (first field/param typed `OpaqueType` /
`ObjectType`, i.e. carries a `this`) needs an address, but today only gets one
if the codebase falls back to the ad hoc `boundRefMaterializedCache`.

This narrows the fix from "rearchitect storage for every type" to: **detect
"this variable's type has a bound-method field" at `registerVariable` time and
force real storage then, for both `const` and `let`, instead of only
discovering the need lazily/locally at the first property access.**

## 3. Two independent bugs, two fixes

### 3a. Local const-storage bug (same function body) — fix now

Today: `const g = gen(); g.next(); g.next();` relies on
`boundRefMaterializedCache` (`MLIRCodeLogic.h:1153` +
`MLIRGenImpl.h:10775`) — a `ScopedHashTable<mlir::Value, mlir::Value>` consulted
only inside `MLIRPropertyAccessCodeLogic::Tuple`/`TupleNoError`, gated on
`!dummyRun && !allowPartialResolve`, and restricted to same-block reuse because
there's no real dominance check. This works but is a workaround bolted onto a
storage decision that was made too early and too coarsely (by `const`-ness
alone, ignoring the type).

**Fix as implemented:** `detectFlags` runs before the initializer's type is
resolved (`func(...)` — which re-invokes `mlirGen` on the initializer
expression and has real side effects, so it can only run once — is only
called later, inside `getVariableTypeAndInit`), so the predicate can't be
decided that early. Instead:

- `MLIRTypeHelper::hasBoundMethodField(type)` (`MLIRTypeHelper.h`, next to
  `hasUndefines`) mirrors the existing `hasUndefinesLogic` pattern
  (`llvm::any_of(type.getFields(), ...)`), testing each field via
  `isBoundReference(fieldType, isBound)` instead of undefined/null — true for
  a `TupleType`/`ConstTupleType` with at least one bound-method field.
- `VariableDeclarationInfo::processConstRef` (`MLIRGenImpl.h`) calls
  `getVariableTypeAndInit` (as before), then computes
  `needsIdentityStorage = mth.hasBoundMethodField(type)` and returns early
  (skipping the value-only `ConstRef` path) when true.
- `getVariableTypeAndInit` gained a `typeAndInitResolved` guard so it can
  safely be called a second time (a no-op reusing the already-resolved
  type/initial) — needed because `registerVariable` now calls
  `createLocalVariable` (which also calls `getVariableTypeAndInit` at its top)
  *after* `processConstRef` already resolved everything, for the
  `needsIdentityStorage` case. Without the guard, the initializer expression
  (e.g. the `gen(...)` call) would be code-generated twice.
- `registerVariable` (`MLIRGenVariables.cpp`) branches:

```
if (variableDeclarationInfo.isConst) {
    processConstRef(...);
    if (variableDeclarationInfo.needsIdentityStorage)
        createLocalVariable(...);   // real alloca, same as `let`
} else {
    createLocalVariable(...);
}
```

`adjustLocalVariableType` (`MLIRGenImpl.h`) still early-returns for any
`isConst` type without widening it, so a `const` generator's type is
unaffected by this second pass through `createLocalVariable` — only the
storage decision changes.

**A second bug surfaced by this change, also fixed:** `resolveIdentifierAsVariable`
(`MLIRGenVariables.cpp:910`) only emits a `LoadOp` through a variable's
`RefType` storage when `varDecl->getReadWriteAccess()` is true; otherwise it
returns the raw ref value unloaded. `createVariableDeclaration`
(`MLIRGenImpl.h`) previously only set read-write-access for `!isConst` (i.e.
`let`/`var`) — a plain `const` never had storage, so this didn't matter for
it. Once a `const` generator gets real storage, it hit this gate too: the
first test run (`00generator_manual_next2.ts`, `drainTwo(it)`) crashed
(0xC0000005) because every use of `it` yielded the bare `RefType`/pointer
instead of the loaded struct value, and a cast to the expected
`!llvm.struct<(i32,i32,ptr,ptr)>` from `!llvm.ptr` failed. Fix:
`createVariableDeclaration` now also sets read-write-access when
`needsIdentityStorage` is true. This is correct for the *other* consumer of
this flag too — `MLIRCodeLogic::CaptureTypeStorage` (`MLIRCodeLogic.h:254`)
decides by-value vs. by-ref closure capture from the same flag, and a
`const` generator captured by a closure should be captured by reference for
the same reason it needs storage in the first place. Note this flag's name
is misleading: despite being called "read-write access," it has nothing to
do with TypeScript-level const-reassignment (this codebase does not enforce
reassignment-to-const as a diagnostic anywhere in MLIRGen) — its only two
consumers are the load-through-ref gate and the capture-by-ref decision.

Once this lands, the `boundRefMaterializedCache` machinery becomes dead for
the case it was built for (a `const` generator already has real storage
before the first `.next()` call, so the property-access fallback in
`MLIRPropertyAccessCodeLogic::Tuple`/`TupleNoError` finds `refValue` already
set via the normal load-ref path and never reaches the cache branch). Verified:
full suite passes with the cache code still in place and untouched. Decision:
leave the cache in place rather than remove it in this change — removing it
in the same change as the storage fix would make it harder to bisect if
something unrelated regresses, and there may be shapes (e.g. object-literal
generator methods, closures) that still exercise it. Revisit removal as a
separate, later cleanup once more confidently redundant.

### 3b. Cross-function parameter aliasing bug — NOT fixed by 3a, tracked separately

`generator-param-value-semantics-bug`: passing a `const` (or `let`) generator
as a function parameter still copies it, because:

- The call lowers a tuple-typed argument as a bare LLVM struct **by value**
  (`TupleType`/`ConstTupleType` → unnamed packed `LLVMStructType`,
  `LowerToLLVM.cpp:6063-6081`) — standard call ABI, no hidden indirection.
- The callee's `mlirGenFunctionParams` (`MLIRGenFunctions.cpp:1026-1070`) wraps
  every incoming argument in `ParamOp`, which lowers
  (`LowerToAffineLoops.cpp:183-193`, `ParamOpLowering`) to a **fresh**
  `VariableOp` seeded from that by-value argument — a new box, unconditionally,
  regardless of const/let on either side.

Fixing 3a does not touch this path at all: the caller having real storage for
its own `const g` doesn't change what gets copied into the callee's block
argument. This needs a distinct fix at the ABI boundary — e.g. detecting
"parameter type has a bound-method field" (the same predicate from 3a) and
passing such parameters by reference (pointer to the caller's storage) instead
of by value, analogous to how `class`/`array`/`object` types are already
pointer-like and thus already "by reference" in effect. This is deferred:
broader blast radius (touches call-site argument lowering and the function
ABI/calling convention, not just declaration codegen), and out of scope for
this change. Tracked by the existing `generator-param-value-semantics-bug`
memory; revisit after 3a is verified stable.

## 4. Testing — done, results

1. Build Debug via PowerShell (per workflow memory) — clean build, all TUs
   recompiled.
2. First run of `00generator_manual_next2.ts` crashed 0xC0000005 — not the bug
   this change targets, but a second bug this change *surfaced* (see §3a's
   `getReadWriteAccess` discovery above). Fixed by also setting
   read-write-access when `needsIdentityStorage`. After that fix, this test
   and `00generator_manual_next.ts` / `00generator7.ts` all pass cleanly (only
   pre-existing benign "using casting to undefined value" / "losing this
   reference" warnings).
3. Full suite: **350/350 JIT + 354/354 compile, 0 failures** (`ctest --test-dir
   ... -R "^test-jit-" -j8` / `-R "^test-compile-" -j8`), covering the
   previously-fragile set from the original 3-round const-manual-next fix
   (`00disposable`/`01disposable`/`02disposable`, `00spread`,
   `01symbol`, etc.) with no regressions.
4. Confirmed 3b (parameter aliasing) is still present and unchanged: a
   throwaway repro (`gen`/`drainTwo` calling `.next()` twice inside the callee,
   then asserting the caller's `it.next()` returns the *next* value after
   those two calls) hit the JIT-executed program's own `assert()` failing at
   runtime — visible only as an apparent "hang" that was actually an invisible
   Windows CRT assert dialog (confirmed via `Get-Process` showing a
   `tslang.exe` with `MainWindowTitle` "Microsoft Visual C++ Runtime Library";
   see `windows-assert-dialog-looks-like-hang` memory). This is the expected,
   pre-existing failure mode for 3b — not newly introduced, not masked.

## 5. Non-goals

- Not changing what `const`/`let` mean for reassignment checking.
- Not adding storage for classes/arrays/plain objects — they don't need it.
- Not fixing the parameter-aliasing bug (3b) in this change.
- Not introducing real MLIR `DominanceInfo` — out of scope; the existing
  same-block conservatism in the (soon-to-be-mostly-dead)
  `boundRefMaterializedCache` path is untouched by this change.
