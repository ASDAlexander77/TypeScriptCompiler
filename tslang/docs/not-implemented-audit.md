# `llvm_unreachable("not implemented")` audit

Status: **5 confirmed crashes fixed across two passes, ~115 markers still
uninvestigated** — written as a roadmap for continuing this audit, not a
claim that the sweep is complete. Triggered by a user request to review
every "not implemented" marker in the codebase and see which ones can be
implemented. Second pass (§4.3-4.5) worked through this document's own §6
priority list while waiting on the first pass's PR to merge.

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

A second, different flavor of dead code: `UnaryBinLogicalOrHelper.h:42-43`'s
`UnaryOp<>` template (originally flagged in §6 as a likely one-line fix —
that guess was wrong) has **zero instantiations anywhere in the codebase**
(`grep -rn "UnaryOp<" lib include tslang` finds nothing outside its own
definition). Unlike `IntersectionType`, this isn't "a reachable type that
gets resolved away" — the whole function template is orphaned; unary-minus
lowering actually goes through a separate, standalone `NegativeOpValue`/
`NegativeOpBin` pair (`LowerToLLVM.cpp:3023,3047`). Its sibling `BinOp<>` in
the same file, by contrast, is live (12 call sites in `LowerToLLVM.cpp` for
`+ - * / % ** >> >>> << & | ^`) and — worth noting — already uses the
graceful `emitError` + `return failure()` pattern with no `llvm_unreachable`
at all, so there was nothing to fix there either. **Lesson for triaging
`.Default` fallbacks generally: check for call sites before assuming a
marker is reachable, the same way you'd check for a producing test before
assuming a type is reachable.**

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

### 4.3 `obj[dynamicKey]` on a boxed (method-bearing) object literal

`MLIRGenAccessCall.cpp` (was line 1159), the `ObjectType` branch of
`mlirGenElementAccess`. Same root cause as §4.1, different runtime
representation (boxed `ObjectType`, used when the object literal has
methods, vs. the plain value-`TupleType` §4.1 covers):

```ts
function main() {
    const obj = { a: 1, greet() { return "hi"; } };
    let n = 42;
    print(obj[n]);   // crash - only constant string keys work here
}
```

Note `obj[n]: any` (the *result* typed `any`, e.g. via `const obj: any =
{...}`) does **not** hit this crash — it's caught earlier by a different,
already-graceful check. The crash needs the object's own inferred type to
stay a concrete boxed `ObjectType`. Same fix as §4.1: `emitError` +
`ValueOrLogicalResult(mlir::failure())`.

### 4.4 `Color[n]` — numeric-enum reverse mapping by a non-constant index

`MLIRGenAccessCall.cpp` (was line 1219), the `EnumType` branch of the same
function:

```ts
enum Color { Red, Green, Blue }
function main() {
    let n = 1;
    print(Color[n]);   // crash
}
```

Real TypeScript numeric enums support reverse mapping (`Color[1] ===
"Green"`). Investigating this turned up something more significant than the
crash itself: **the constant-index case doesn't work either.**
`Color[1]` (a literal, not a variable) fails with `error: Enum member '' can't
be found` — the constant-index branch just forwards the raw integer
attribute into a *string-keyed* property lookup, which was never going to
resolve. So reverse enum mapping is not implemented **at all**, for any
index form; the crash on the non-constant path is a symptom, not the actual
gap. Implementing it for real needs a reverse lookup table generated
alongside the enum (or a `switch` over values) — out of scope for this pass.
Converted the crash to `emitError(location) << "Enum reverse lookup by index
is not supported";` and left the deeper constant-index bug undocumented in
code (documented here) since fixing the crash without fixing the underlying
feature would just swap one wrong-but-silent-ish failure for another
wrong-but-clean one — worth a dedicated follow-up rather than a half fix.

### 4.5 `super(...)` call target with no resolvable reference

`MLIRGenAccessCall.cpp` (was line 1535), inside the `ClassStorageType` case
of the call-expression dispatch (`mlirGenCallExpressionCases`-style
`TypeSwitch`, the "seems we are calling type constructor for super()"
branch). When `MLIRCodeLogic::GetReferenceFromValue` fails to produce a
reference for the call target, the code crashed instead of falling through
to the same graceful `.Default` two cases below it in the exact same
`TypeSwitch` (`emitError(location, "not supported function type"); value =
mlir::Value();`). No standalone repro was found for this one — it requires
whatever unusual expression shape makes a `ClassStorageType`-typed call
target fail `GetReferenceFromValue`, which wasn't reachable from the "obvious"
`super()` shapes tried. Fixed by literally duplicating the neighboring
`.Default` body, on the reasoning that whatever situation reaches this
branch deserves the same treatment its sibling already gives every other
unsupported call-target shape — lowest-confidence fix of the five in this
document (untested against a live repro), but also the lowest-risk, since it
only changes behavior for a path that was already 100% fatal.

All three (§4.3-4.5) verified individually where a repro was found, and via
the full suite (`ctest -C Debug -j8`: 829/829, no regressions).

## 5. Inventory of remaining markers (untested this pass)

Grouped by file. "Shape" is a guess from reading the surrounding code, not a
verified verdict — see §2 for how to actually check one.

### 5.1 Named/specific (cheapest to investigate next — read the message + local branch, write a 5-line repro)

**Fixed this pass**: `MLIRGenAccessCall.cpp`'s three sites (was lines
1159/1219/1535) — see §4.3-4.5.

| Site | Message | Shape (unverified guess) |
| --- | --- | --- |
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

**Attempted this pass, not reached**: tried two plausible `.ts`-level
triggers — a callback parameter typed as a union of function signatures
(`type Cb = ((x: number) => void) | ((x: string) => void)`, hoping arrow
function param-type inference would call `getParamFromFuncRef`/
`getFirstParamFromFuncRef` with a `UnionType`) — neither crashed; both hit
earlier, already-graceful type-mismatch errors instead. The `.ts`-level
approach that worked for §4.1-4.5 doesn't seem to reach this family easily;
confirms the §5.4 recommendation itself (skip writing more `.ts` repros,
go straight to unit-testing the functions directly with a constructed
`mlir::Type` the way `TypeHelper.cpp` already does for other
`MLIRTypeHelper.h` functions) rather than a reason to deprioritize it.

## 6. Suggested next steps, in cost order

1. ~~`UnaryBinLogicalOrHelper.h:42-43`~~ — turned out to be dead code
   (`UnaryOp<>` has zero call sites), not a live one-line fix; see the new
   §3 entry. No action needed.
2. ~~The rest of §5.1 (named/specific)~~ — done this pass: found and fixed
   3 more real crashes (§4.3-4.5); only `MLIRGenCast.cpp`'s two `TypeOf`
   sites and the large `MLIRGenImpl.h`/`MLIRGenInterfaces.cpp`/
   `MLIRGenTypes.cpp` cluster remain from the original list.
3. §5.4 (`funcRef` family) — extend `unittests/MLIRGen/TypeHelper.cpp` with
   direct calls instead of writing `.ts` repros; fast to iterate. Two `.ts`
   attempts this pass didn't reach it (see above) — go straight to unit
   tests next time.
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
