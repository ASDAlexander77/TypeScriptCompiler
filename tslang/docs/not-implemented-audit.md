# `llvm_unreachable("not implemented")` audit

Status: **2 confirmed crashes fixed (this pass), ~120 markers still uninvestigated** —
written as a roadmap for continuing this audit, not a claim that the sweep is
complete. Triggered by a user request to review every "not implemented" marker
in the codebase and see which ones can be implemented.

## 1. Scope and method

A repo-wide search for `not implemented`/`llvm_unreachable` across
`lib/TypeScript/*.cpp` and `include/TypeScript/{MLIRLogic,LowerToLLVM}/*.h`
turns up **~130 raw matches** (some are two-line sites: an `LLVM_DEBUG` print
immediately followed by the `llvm_unreachable`, counted here as one site).
Grep command used:

```
grep -rn "not implemented\|Not implemented\|NOT IMPLEMENTED\|not yet implemented\|NotImplemented" \
  --include=*.cpp --include=*.h lib include tslang
```

These markers span three very different situations that look identical in a
grep, and need to be told apart before deciding what "implement" even means
for each:

1. **Legitimate diagnostics.** A handful are `emitError(...)` calls for
   genuinely invalid source (e.g. `MLIRGenClasses.cpp:1919`, "Abstract method
   'X' is not implemented in 'Y'" — a normal missing-override error;
   `MLIRGenVariables.cpp:285/324`, array binding pattern spread/type
   mismatches). These already produce clean compiler errors. Not gaps.
2. **Generic exhaustiveness fallbacks.** The majority (~90+ of the raw
   matches) are `.Default([&](auto type) { llvm_unreachable("not
   implemented"); })` at the bottom of an MLIR `TypeSwitch` chain, mostly in
   low-level LLVM-lowering, RTTI, and cast-helper code
   (`LowerToLLVM.cpp`, `CastLogicHelper.h`, `LLVMRTTIHelperVC*.h`,
   `MLIRRTTIHelperVC*.h`, `MLIRTypeHelper.h`'s `funcRef` family, etc.). These
   fire only if a `mlir::Type` value reaches that specific conversion/lowering
   stage with a kind the author never handled there. Some are real gaps; many
   are defensive guards for states the type system already rules out earlier
   in the pipeline (see §3 for a proven example) — **reachability is unknown
   without testing each one**, and that is genuinely the expensive part.
3. **Named, specific gaps.** ~40 markers carry a message naming the exact
   scenario ("SpreadAssignment not implemented for type: X", "TypeOf NOT
   IMPLEMENTED for Type: X", "not implemented (index)", "not implemented
   (ElementAccessExpression)", …). These are much cheaper to turn into a
   reproduction: the message plus its surrounding `if`/`TypeSwitch` branches
   usually tells you exactly what source-level construct is missing.

Of these, only **6 sites were actually tested** this pass (2 named markers
confirmed reachable and fixed, 1 generic fallback confirmed dead, 3 more
named markers read but not yet reproduced). Everything else in §5 is an
inventory, not a verdict.

## 2. Reproduction recipe used

For each candidate: read the surrounding code to infer what TS source
pattern would make execution reach that branch, write a minimal `.ts` file
exercising it, and run it through the actual compiler:

```
test-runner.exe <path-to-repro.ts>
```

(no `-jit`/`-shared` needed for a single-file reachability check). A crash
looks like:

```
not implemented
UNREACHABLE executed at I:\...\MLIRGenImpl.h:5797!
```

If it crashes, the marker is real and reachable. If it compiles/runs, either
the type is resolved away earlier in the pipeline (like §3), or the guess
about the trigger was wrong and needs another attempt.

## 3. Confirmed dead code (proof-of-concept for triaging fallbacks)

`LowerToLLVM.cpp:6267`:

```cpp
converter.addConversion([&](mlir_ts::IntersectionType type) {
    llvm_unreachable("type usage (IntersectionType) is not implemented");
    return mlir::Type();
});
```

Intersection types (`A & B`) are exercised by two existing, currently-passing
tests (`00intersection_type.ts`, `00intersection_type_generic.ts`,
`test-compile-00-intersection-type[-generic]` / `test-jit-...` in
`test/tester/CMakeLists.txt`), so an `IntersectionType` MLIR value clearly
*can* exist. It just never survives to LLVM type conversion — it must be
resolved into its concrete merged/flattened type earlier in MLIRGen. This
one marker is confirmed unreachable for any currently-expressible source
program. **This is the template for triaging the remaining ~90 generic
fallbacks**: find an existing test that plausibly produces the type in
question, confirm it passes, and if so the fallback is very likely dead for
today's feature set (not proof for all future features, but proof for now).

## 4. Fixed this pass

Both fixes are in `lib/TypeScript/MLIRGenImpl.h`, both converted a hard
`llvm_unreachable` crash into an `emitError` + graceful failure, matching the
pattern already established at `MLIRGenVariables.cpp:285` (Array Binding
Pattern spread) — nothing new was invented, this is the existing
"fail loud with a message, don't crash" convention applied to two spots that
hadn't gotten it yet.

### 4.1 `obj[dynamicKey]` — non-constant index on a tuple/object-literal value

`mlirGenElementAccessTuple` (`MLIRGenImpl.h`, was line 5797):

```ts
function main() {
    const obj = { a: 1, b: 2 };
    let key = "a";
    print(obj[key]);   // key is a runtime variable, not a literal
}
```

crashed with `UNREACHABLE executed at MLIRGenImpl.h:5797`. Root cause:
tuples/object-literals in this compiler lower to a **fixed-layout struct**
(each field resolved to a specific byte offset at compile time), not a
dynamic hash map. When the index expression is a compile-time constant, the
existing code resolves it to a field the normal way; the crash was the `else`
branch, hit whenever the index is a genuine runtime value. This is not
missing code so much as **a real limitation of the current object
representation** — properly "implementing" `obj[runtimeKey]` in general would
need a dynamic property-bag runtime representation (a different data
structure entirely, not a small patch). Converted the crash to:

```cpp
emitError(location) << "Element access with a non-constant index is not supported on this type; "
                        "only array types and constant keys (obj[\"literal\"]) can be indexed";
return ValueOrLogicalResult(mlir::failure());
```

### 4.2 Spreading a non-struct-like value into an object literal

The `SpreadAssignment` `TypeSwitch` inside object-literal codegen
(`MLIRGenImpl.h`, was line 7863) only handles spreading a
`TupleType`/`ConstTupleType`/`InterfaceType`/`ClassType`/`ObjectType` into
`{...expr}`. Anything else hit the `Default` branch:

```ts
function main() {
    const arr = [1, 2, 3];
    const obj = { ...arr };            // crash
}
function main2() {
    let x: {a: number} | number[] = { a: 1 };
    const obj = { ...x };              // crash, same site
}
```

Unlike §4.1, this genuinely **is** a missing feature (array spread would need
to synthesize numeric-string-keyed fields `"0"`, `"1"`, …; union spread would
need a runtime type-tag dispatch) rather than an architectural wall — it just
wasn't scoped/implemented for this pass. Converted to:

```cpp
emitError(location) << "Spread in an object literal is not supported for type: " << to_print(type);
return mlir::failure();
```

Both verified individually (clean diagnostic, no crash) and via the full
suite (`ctest -C Debug -j8`: 829/829, no regressions — these `Default`
branches were never reached by any existing passing test).

## 5. Inventory of remaining markers (untested this pass)

Grouped by file. "Shape" is a guess from reading the surrounding code, not a
verified verdict — see §2 for how to actually check one.

### 5.1 Named/specific (cheapest to investigate next — read the message + local branch, write a 5-line repro)

| Site | Message | Shape (unverified guess) |
|---|---|---|
| `MLIRGenAccessCall.cpp:1159` | not implemented (ElementAccessExpression) | `boxedObj[computedNonStringExpr]` — cousin of the fixed §4.1 site, one level up in the dispatch (only reached before deciding it's a tuple) |
| `MLIRGenAccessCall.cpp:1219` | not implemented (ElementAccessExpression) | `enumValue[computedExpr]` with a non-constant index |
| `MLIRGenAccessCall.cpp:1524` | not implemented | unread this pass |
| `MLIRGenCast.cpp:1321-1322` | TypeOf NOT IMPLEMENTED for Type | inside a generated `__unbox<T>` helper (generic type-parameter unboxing from `any`); `.Default` for a type kind not in its explicit list (Tuple/Array/Enum/Union/Optional are plausible candidates) |
| `MLIRGenCast.cpp:1498-1499` | TypeOf NOT IMPLEMENTED for Type | second, near-identical site — check if it's reachable via a different call path than 1321 |
| `MLIRGenImpl.h:5330` | not implemented | unread |
| `MLIRGenImpl.h:6732` | not implemented | unread |
| `MLIRGenImpl.h:7314` | not implemented | unread |
| `MLIRGenImpl.h:7418` | not implemented | unread |
| `MLIRGenImpl.h:8164` | not implemented | unread |
| `MLIRGenImpl.h:8382` / `:8400` / `:8426` | not implemented | unread, three sites close together — likely related |
| `MLIRGenImpl.h:9342` | not implemented | unread |
| `MLIRGenInterfaces.cpp:475` | not implemented yet | unread |
| `MLIRGenInterfaces.cpp:932` / `:954` | not implemented | unread |
| `MLIRGenTypes.cpp:183` | not implemented type declaration | unread |
| `MLIRGenTypes.cpp:1474` / `:1567` / `:1876` / `:1910` / `:2001` / `:2204` / `:2210` / `:2661` / `:3401` | not implemented | unread, largest single-file cluster after MLIRGenImpl.h |
| `LLVMCodeHelper.h:452` | array literal is not implemented(1) | likely the LLVM-lowering-side twin of the (confirmed-dead) `MLIRGenImpl.h:7875` "object literal is not implemented(1)" `else` branch — check the same way (is there any other `SyntaxKind` an array-literal element list can produce?) |
| `UnaryBinLogicalOrHelper.h:42-43` | "Not implemented operator for type 1: 'X'" (`emitError`) then `llvm_unreachable` | **worth a quick look on its own**: this one already calls `emitError` (like §4's fix) but *still* falls through to `llvm_unreachable` right after — likely a copy-paste of the crash-then-message pattern that never got the `return` that would make the error actually graceful. If so, this is a one-line fix (drop the `llvm_unreachable`, return failure), no new investigation needed beyond confirming what emits it currently crashes instead of erroring cleanly. |

### 5.2 Generic `TypeSwitch::Default` exhaustiveness fallbacks (likely mostly dead, per §3's precedent — verify by checking whether an existing passing test already produces the relevant `mlir::Type` at that pipeline stage)

`MLIRGenClasses.cpp:603,635,1802,2269,2306` · `MLIRGenExpressions.cpp:530,552,988` ·
`MLIRGenGenerics.cpp:424,542,911,1342` · `MLIRGenImpl.h:3203,3477,3803,4402,4518,6379,7080,7094` ·
`MLIRGenInterfaces.cpp:656` · `CastLogicHelper.h:338,344,353,359,460,487,1002` (four of these say
"must be processed at MLIR pass" — suggests a *specific*, documented reason
they should be unreachable at the LLVM-lowering stage, worth reading before
assuming they're arbitrary) · `CodeLogicHelper.h:241` · `OptionalLogicHelper.h:143,213` ·
`UndefLogicHelper.h:74,107` · `MLIRCodeLogic.h:1218,1660,1680,1722` ·
`MLIRPrinter.h:302-303,534` (type-name printing — a `Default` here would show
up immediately as a printer test failure, so likely easy to check against
`unittests/MLIRGen/TypeToString.cpp`'s existing coverage) · `MLIRTypeIterator.h:403-404` ·
`Win32ExceptionPass.cpp:584`.

### 5.3 RTTI type-switch fallbacks (Windows/Linux variants — the Linux ones can't be exercised from this Windows dev box without a Linux/WSL build)

`LLVMRTTIHelperVCWin32.h:141,156,169` · `LLVMRTTIHelperVCLinux.h:113,128,142` ·
`MLIRRTTIHelperVC.h:108` · `MLIRRTTIHelperVCWin32.h:216,226,241` ·
`MLIRRTTIHelperVCLinux.h:146,161,182,202,217,230,399`.

### 5.4 `MLIRTypeHelper.h`'s `funcRef` family — interesting because directly unit-testable

`getReturnTypeFromFuncRef` (:732-733), `getParamFromFuncRef` (:755-756),
`getFirstParamFromFuncRef` (:779-780), `getParamsFromFuncRef` (:805-806),
`getParamsTupleTypeFromFuncRef` (:840, `llvm_unreachable` already commented
out at :843 — someone deliberately silenced this one, worth understanding
why before re-enabling), `getVarArgFromFuncRef` (:863-864), plus :410, :420,
:899, :2108, :2256-2257, :2290, :2307, :2685, :2709. Unlike most of §5.2,
these are **pure functions taking an `mlir::Type` and returning a piece of
it** — exactly the shape `unittests/MLIRGen/TypeHelper.cpp` (added this
session, see `declaration-printer-unit-tests`-style memory entries) already
tests other `MLIRTypeHelper.h` functions with. Reachability here is testable
*without* writing a `.ts` repro at all: construct the "wrong" `mlir::Type`
input directly in a unit test and see what real callers expect the sensible
behavior to be, the same way the existing `canWideTypeWithoutDataLoss` tests
work.

## 6. Suggested next steps, in cost order

1. `UnaryBinLogicalOrHelper.h:42-43` — likely a one-line fix, already has the
   error message, just needs the crash removed.
2. The rest of §5.1 (named/specific) — cheap repro-and-check, same recipe as
   §4.
3. §5.4 (`funcRef` family) — extend `unittests/MLIRGen/TypeHelper.cpp` with
   direct calls instead of writing `.ts` repros; fast to iterate.
4. §5.2 (generic fallbacks) — triage a handful against existing passing
   tests using §3's method before assuming any individual one is live.
5. §5.3 (RTTI) — lowest priority from this (Windows) machine; the Linux
   variants need a WSL/Linux build to exercise at all, and even the Windows
   ones are deep in a code path (RTTI/exception typeinfo generation) that's
   hard to reach without a specific class-hierarchy-plus-exception scenario.

## 7. Non-goals / out of scope

- This document does not claim every remaining marker is a "real bug" —
  §3 demonstrates at least some meaningful fraction are dead code, and the
  true ratio across all ~120 is unknown.
- Not attempting §4.2's actual missing feature (array/union spread into an
  object literal) in this pass — only converting its crash into a clean
  error. Implementing real array-spread semantics (numeric-string-keyed
  field synthesis) is a separate, scoped follow-up if ever prioritized.
- Not attempting a dynamic-property-bag runtime representation for §4.1 —
  that is a different object model entirely, out of scope for a
  crash-to-error pass.
