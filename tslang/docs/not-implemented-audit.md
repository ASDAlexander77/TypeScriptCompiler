# `llvm_unreachable("not implemented")` audit

Status: **§5.1, §5.2, §5.3, and §5.4 are all now fully closed - every
marker this document's own inventory ever named has been triaged, AND a
fresh, corrected repo-wide grep (§1's command was missing `.td` files, and
`LowerToLLVM.cpp` had never actually been triaged despite being mentioned
in passing) found and closed 14 more real live sites (§4.17, twelfth
pass) - as of this pass, `grep -rn 'llvm_unreachable("not implemented")'`
across every `.cpp`/`.h`/`.td` file in the repo returns zero live hits.**
What remains: the Linux RTTI execution-verification gap (fixes applied and
compile-verified, not run-verified - no Linux/WSL build stood up this
session), and 2 bugs found-but-out-of-scope (§6 item 9). This is still not
a claim that every conceivable crash is gone (§7's caveats about other
`llvm_unreachable` messages - "type mismatch", "cast must happen earlier" -
and other diagnostic-message patterns this specific grep pattern doesn't
catch), but the specific `"not implemented"` marker this whole document
tracks is now fully accounted for. Triggered by a user request to
review every "not implemented" marker in the codebase and see which ones
can be implemented. Second pass (§4.3-4.5) worked through this document's
own §6 priority list while waiting on the first pass's PR to merge. Third
pass (§4.6-4.8) closed out §5.4's `MLIRTypeHelper.h` `funcRef` family after
that PR merged. Fourth pass (§4.9) closed out `MLIRGenCast.cpp`'s two
`TypeOf` sites, the last item left from the original §5.1 named/specific
list. Fifth pass (§4.10) started the `MLIRGenImpl.h` cluster (§5.1's last
remaining block), closing 4 of its sites. Sixth pass (§4.11) closed 3 more
sites in the same cluster, including two real, easily-reachable
`import X = ...` crashes. Seventh pass (§4.12) closed the rest of §5.1:
`MLIRGenImpl.h`'s last 2 sites, `MLIRGenInterfaces.cpp`'s remaining 2, the
entire `MLIRGenTypes.cpp` cluster (10 sites), and `LLVMCodeHelper.h:452` -
plus 2 bonus fixes found along the
way that weren't in the original grep (a `return nullptr;`-as-`std::string`
UB crash, and the `MLIRGenImpl.h:7907` object-literal twin this doc had left
un-patched after calling it dead). Eighth pass (§4.13) closed all of §5.2 -
every generic `TypeSwitch::Default`/switch fallback this doc had inventoried
across 12 files - finding several more real, easily-reachable crashes along
the way (array-to-`any[]` widening, `class`/`interface extends`ing the wrong
kind of type, accessor `++`/`--`, generic-call type-argument inference
failure, destructuring assignment gaps) plus one gap found by diffing
against a type list rather than reproducing a crash (`MLIRTypeIterator.h`
was missing `NamespaceType` entirely). Ninth pass (§4.14) went back and
fixed the two bugs §4.13 had explicitly found-but-not-fixed as out of
scope - both turned out to share one root cause in `TupleFieldName()`
(a null `mlir::Attribute` reaching an unguarded `dyn_cast`/`cast`), found
by tracing code after a live debugger proved unable to catch either crash
(a plain `assert()`/`abort()` raises no Win32 exception for ProcDump to
see, and live-attaching WinDbg made the same assert pop a blocking GUI
dialog instead, since `IsDebuggerPresent()` becomes true). Tenth pass
(§4.15) closed §5.3 (RTTI) and the remaining §5.4 stray lines - see §4.15
for the full writeup, including the one real bug found (a Linux-only
int64-width asymmetry between two `setType` overloads) and the honest
caveat about it not being execution-verified. Eleventh pass (§4.16), a new
session after PR #305 merged, closed the very last named site in this
document, `MLIRTypeHelper.h:410/420` (`convertAttrIntoType`) - see §4.16.
Twelfth pass (§4.17), same session, prompted by a user spotting a still-
live site in `TypeScriptOps.td` (a file this audit's own grep had never
covered): found and closed 14 more sites across `TypeScriptOps.td`,
`DiagnosticHelper.cpp`, and `LowerToLLVM.cpp` (which the doc's §1 overview
had mentioned but §5 never actually inventoried), including one real crash
(`~` on a raw `f64`/`f32` value) - see §4.17.

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

**Correction from §4.17 (twelfth pass)**: this command has a real gap -
`--include=*.td` was never added, so TableGen files (which can embed live
C++ in `extraClassDeclaration`/`builders`/etc., like
`TypeScriptOps.td:725`) were invisible to every pass before the twelfth.
Also, being *mentioned* in this doc's prose (like `LowerToLLVM.cpp` was,
in the paragraph below) is not the same as being *triaged* - always
cross-check against §5's per-file lists, which is the actual bookkeeping;
`LowerToLLVM.cpp` sat un-triaged for 11 passes despite being named here
from the start. Add `--include=*.td` before trusting this command's
output as complete again.

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

### 4.6-4.8 `MLIRTypeHelper.h`'s `funcRef` family — three more live crashes, found via §5.4's own recommendation

§5.4 (below) suggested checking this family's reachability by tracing real
callers instead of writing more `.ts` repros, since two `.ts`-level attempts
in pass two didn't reach it. Tracing every caller of
`getReturnTypeFromFuncRef`/`getParamFromFuncRef`/`getFirstParamFromFuncRef`/
`getParamsFromFuncRef`/`getVarArgFromFuncRef` found that **every** call site
outside `MLIRTypeHelper.h` itself is already guarded by an `isAnyFunctionType`
check (or is structurally guaranteed a function type by C++'s own type
system, e.g. a parameter statically typed `mlir_ts::FunctionType`) — the
same "guarded, therefore dead" shape as §3's `IntersectionType` proof, just
proven by call-site inspection instead of an existing test. The internal
recursive uses inside `MLIRTypeHelper.h` (`equalFunctionTypes`,
`mergeFuncTypes`, `extendsTypeFuncTypes`) are equally guarded at their own
single call sites.

**Except one cluster that isn't guarded at all**: `MLIRGenTypes.cpp`'s
`getEmbeddedTypeWithParamBuiltins` — the code that implements the built-in
utility types `ReturnType<T>`, `Parameters<T>`/`ConstructorParameters<T>`,
`ThisParameterType<T>`, and `OmitThisParameter<T>` (recognized by name, no
declaration needed, matching real TypeScript's lib.es5.d.ts versions) — calls
straight into this family with whatever type argument the user wrote, with
no `isAnyFunctionType` check first. Real TypeScript rejects a non-function
type argument to any of these with a constraint error at the call site;
tslang has no such constraint check, so the type argument flows unchecked
into the `funcRef` helpers. Confirmed by direct repro:

```ts
function main() {
    let x: ReturnType<number>;        // crash: MLIRTypeHelper.h:733 (getReturnsFromFuncRef)
}
```

```ts
function main() {
    let x: ThisParameterType<number>; // crash: MLIRTypeHelper.h:780 (getFirstParamFromFuncRef)
}
```

```ts
function main() {
    let x: OmitThisParameter<number>; // crash: MLIRTypeHelper.h:899 (getOmitThisFunctionTypeFromFuncRef)
}
```

`Parameters<number>`/`ConstructorParameters<number>` does **not** crash —
its `getParamsTupleTypeFromFuncRef` backend already had its `llvm_unreachable`
commented out (see §5.4's note on this), so it silently returns a null type,
and the embedded-type dispatcher already treats that as "generic type
Parameters can't be found" - a real, if slightly misleading, clean error.
That existing behavior is the template the fix below follows for its three
crashing siblings.

None of these six functions take a `mlir::Location`, so none can call
`emitError` directly the way §4.1-4.5's fixes did. Instead, each `Default`
case was changed to do nothing and fall through to the function's existing
default-constructed return value (null `Type`/empty `ArrayRef`/`false`) —
exactly what `getParamsTupleTypeFromFuncRef` already did. The caller
(`getEmbeddedTypeWithParamBuiltins`) already treats that null result as
"generic type X can't be found", so no caller-side changes were needed to
get a clean diagnostic. Fixed in `getReturnsFromFuncRef` (formerly
`getReturnTypeFromFuncRef`'s helper - also dropped its now-pointless
`noError` parameter, since both branches behaved identically once the crash
was removed), `getParamFromFuncRef`, `getFirstParamFromFuncRef`,
`getParamsFromFuncRef`, `getVarArgFromFuncRef`, and
`getOmitThisFunctionTypeFromFuncRef`. The last three of those were proven
dead by the call-site trace above (not reachable from any real caller today)
but fixed anyway for consistency with their three now-fixed siblings, since
leaving some `Default` branches in the same six-function family crashing and
others not would just be a landmine for the next caller who doesn't happen
to add the same guard.

Verified individually (all four repros above give a clean "generic type X
can't be found" instead of crashing) and via the full suite (`ctest -C Debug
-j8`: 829/829, no regressions). Also spot-checked the non-crash path: the
existing tests exercising these utility types with a real function argument
(`test/tester/tests/00types_utility.ts`, `01types_utility.ts`) still pass.

### 4.9 `MLIRGenCast.cpp`'s two `TypeOf` sites — one dead, one a real, easily-reachable crash

Both are `.Default` branches of a `TypeSwitch` that builds up a synthetic
`typeof t == '...'` dispatch function as TS source text, then parses and
calls it - the compiler's mechanism for runtime type discrimination when
casting away from a type whose concrete shape isn't known until runtime
(`any`, or a union that can't be merged into one storage type).

**`castPrimitiveTypeFromAny` (was :1320-1322, the `__unbox<T>` helper for
generic type-parameter unboxing from `any`, as guessed in the original
§5.1 entry): dead code.** Its one call site (`MLIRGenCast.cpp:1140`, inside
`castFromSourceSpecialCases`-family cast dispatch) only invokes it when
`type` (the cast destination) is one of `{NumberType, BooleanType,
StringType, BigIntType, IntegerType, FloatType, ClassType}` - a strict
subset of what the `TypeSwitch` inside already handles (`{Boolean,
TypePredicate, Number, String, Char, Integer, Float, Index, BigInt,
Function×4, Class, Interface, Null, Undefined}`). Same "guarded, therefore
dead" shape as §3's `IntersectionType` and §4.6-4.8's `funcRef` family.
Fixed anyway (crash → set a flag, `emitError` + `return failure()` after
the switch) since `location` was already in scope here and leaving a live
`llvm_unreachable` behind is a landmine for the next caller.

**`castFromUnion` (was :1498-1499): a real, easily-reachable crash.** Called
from `castFromSourceSpecialCases` whenever casting *from* a union-typed
value whose members can't be merged into one storage representation
(`mth.isUnionTypeNeedsTag`) to anything other than `any`. It loops over
each union member type building the same kind of `typeof`-dispatch
function, and the `TypeSwitch` per member is missing `TupleType`/
`ConstTupleType` entirely - i.e. **any union with an object-literal-shaped
member hits this the moment it needs a runtime cast**, which is a very
ordinary shape (not an obscure corner case like §4.4's enum reverse-mapping
or §4.5's `super()` edge case):

```ts
function main() {
    let x: number | { a: number };
    x = 5;
    let y = <number>x;   // crash: UNREACHABLE at MLIRGenCast.cpp:1499
}
```

The function's own forward declaration in `MLIRGenImpl.h` already carries a
`// TODO: remove using typeof for Union types as it can't handle types such
as 2 tuples in union etc` - confirming this is a known, **genuinely missing
feature** (like §4.2), not just an unreached architectural corner: even two
*different* tuple-shaped union members couldn't be told apart by `typeof`
alone (both report `"object"`), so a real fix needs a structural redesign
(a runtime shape tag, not `typeof` string dispatch), out of scope here. A
partial start at this is visible in the code - a `tupleTypes`
`SmallVector` and `TYPE_TUPLE_ALIAS` templating exist and are wired up at
the end of the function, but nothing ever pushes into `tupleTypes` because
no `.Case<mlir_ts::TupleType>` was ever added to populate it; that
half-finished thread was left as-is rather than completed, since finishing
it properly means solving the "2 tuples in union" ambiguity the TODO
already flags, not just adding one more `.Case`. Converted the crash to
`emitError(location) << "Cast from " << to_print(value.getType()) << " to "
<< to_print(type) << " is not supported"; return mlir::failure();` after the
member loop, gated by the same kind of flag used for `castPrimitiveTypeFromAny`
(a per-subtype lambda can't `return` the enclosing function directly).

Verified individually (clean diagnostic, no crash) and via the full suite
(`ctest -C Debug -j8`: 829/829, no regressions).

### 4.10 `MLIRGenImpl.h` cluster, first 4 sites — 1 real crash, 2 dead, 1 untested-feature-path

Started §5.1's last remaining block (was: 5330, 6732, 7314, 7418, 8164,
8382/8400/8426, 9342 — this pass covers the first four).

**`getIntTypeAttribute` (was :6737, inside the integer-literal-to-attribute
helper): a real, confirmed crash**, and a good illustration of why a
"hanging" test isn't always an infinite loop. A plain integer literal
needing more than 128 bits (e.g. a 62-digit literal) crashes here - but on
this dev machine the `test-runner.exe` process appeared to *hang* rather
than crash: `llvm_unreachable`'s internal `abort()` popped an invisible
"Microsoft Visual C++ Runtime Library" dialog (confirmed via `Get-Process | Select MainWindowTitle` showing
"Microsoft Visual C++ Runtime Library" while CPU usage stayed near zero -
blocked, not busy-looping; the same "idle-CPU hang is actually an invisible
blocking dialog" pattern applies to `abort()`, not just `assert()`/
`_CrtDbgReport`). No
`mlir::Location` is available inside `getIntTypeAttribute` itself (it only
takes the literal's raw text), so the fix returns a null `Attribute` on
overflow instead of crashing, and the actual caller two frames up
(`MLIRGenImpl::mlirGen(NumericLiteral ...)` in `MLIRGenExpressions.cpp`,
which already has `loc(numericLiteral)` in scope) now checks for null and
emits `"integer literal is too large to represent (more than 128 bits)"`
cleanly. Verified: the same repro now fails instantly with a clean
diagnostic instead of hanging behind an invisible dialog.

Note: `BigIntLiteral` (the `123n` suffix form, a separate AST node handled
a few lines below in the same file) has its own independent, *unguarded*
`APSInt`-to-`int64` conversion with no width check at all - a likely
silent-truncation bug for a large `...n` literal, not a crash. Different
failure mode (wrong answer vs. crash), out of scope for this "not
implemented" audit; left as a note for a future pass.

**`createConstArrayOrTuple` (was :7319, the `TypeData::NotSet` fallthrough
after the `Tuple`/`Array` checks): dead code.** `ArrayInfo::adjustArrayType`
(called immediately before an `ArrayInfo` is handed to this function)
unconditionally normalizes `TypeData::NotSet` to `TypeData::Array` - so by
the time this function runs, `dataType` can only ever be `Tuple` or
`Array`. Same "producer already guarantees it" shape as the `funcRef`
family fix in §4.6-4.8, just proven through a normalization step instead of
a caller-side type check. Fixed anyway (`emitError` + `return failure()`,
`location` was already in scope) for the same "don't leave a landmine"
reasoning as the rest of this audit.

**The sibling array-spread `.Default` (was :7423, "array spread value
type... not implemented", inside the array-literal-into-a-dynamic-array
builder): also dead code, proven by enumerating every producer.** Every
site that constructs an `ArrayElement{value, isSpread=true,
isVariableSizeOfSpreadElement=false}` (the only combination that reaches
this branch) does so exclusively from the `TupleType` case a few lines
above the crash - so `val.value.getType()` is always already a `TupleType`
by construction, meaning the `dyn_cast<mlir_ts::TupleType>` right before
this `else` always succeeds and this branch is never taken. Fixed anyway.

**`ClassStaticFieldAccess` (was :5330, the `classInfo->isDynamicImport`
branch): an untested feature path, not a normal-program crash.**
`isDynamicImport` is only set on a `ClassInfo` by a class-level
`@dllimport("path")` decorator *with an argument* (the "load this class
from an external DLL by path at runtime" feature) - grepping the entire
test suite for `@dllimport(` with any argument found zero matches. This
whole feature path is untested anywhere, and its sibling variable-level
version (`MLIRGenVariables.cpp`) carries its own explicit `// TODO: finish
it, look at mlirGenCustomRTTIDynamicImport as example` - so this reads as
an intentionally-incomplete feature, not a hidden gap in otherwise-working
code. Still converted the crash to `emitError` + `return mlir::Value()`
(matching this function's own established local convention two lines
above, for the "field not accessible" case - this function returns a bare
`mlir::Value`, not a `LogicalResult`, so there's no `mlir::failure()` to
return here), since crashing is strictly worse than a clean "not supported"
error even for an unfinished feature.

All four verified (repro for the real crash, full-suite for regressions):
`ctest -C Debug -j8` → 829/829, no regressions.

### 4.11 `MLIRGenImpl.h` cluster, 3 more sites — 2 real `import X = ...` crashes, 1 defensive fix

Continues §4.10's sweep of the same cluster (was: 8164, 8382/8400/8426,
9342 remaining after the previous pass; this pass covers 3 of those 4
entries).

**`mlirGenModuleReference` (was :8432): a real, easily-reachable crash.**
`import X = require("module")` - the classic CommonJS/Node-interop form of
TypeScript's import-equals syntax, still common in real-world legacy TS -
crashed, because the function only handled `SyntaxKind::QualifiedName`
(`import X = A.B.C`) and `SyntaxKind::Identifier` (`import X = Foo`), never
`SyntaxKind::ExternalModuleReference` (confirmed via the parser: `ts-new-
parser` does have a dedicated `ExternalModuleReference` node and
`isExternalModuleReference` check, so this genuinely parses). Since tslang
compiles as one flat program with no Node.js-style dynamic `require()` at
runtime, there's nothing sensible to resolve this to - converted to a clean
`emitError` instead ("'import X = require(...)' is not supported").

**`mlirGen(ImportEqualsDeclaration...)` (was :8458): also a real,
easily-reachable crash**, once the target resolves to anything other than
a namespace, class, or interface - e.g. `import X = SomeEnum` or `import X
= someFunction`. Confirmed via repro (`enum Color {...}; import C =
Color;`). Converted to a clean `emitError`.

Fixing the first crash uncovered a **second, latent bug** in the exact same
function: after making `mlirGenModuleReference` return a graceful
`mlir::failure()` instead of crashing, the `require()` repro immediately
hit a *different* crash - `Assertion failed: detail::isPresent(Val) &&
"dyn_cast on a non-existent value"` - because the caller
(`mlirGen(ImportEqualsDeclaration...)`) unwraps the callee's result with
`V(result)` without ever checking `result.failed()` first. That check was
never needed before, because the callee previously only ever *crashed* or
*succeeded* - a graceful failure return was a code path this caller had
never had to handle. Added the missing `EXIT_IF_FAILED(result)` guard
before the `V(result)` unwrap. **Lesson for the rest of this audit**:
converting a `llvm_unreachable` into `return mlir::failure()` can surface a
second bug one level up, in a caller that assumed its callee could never
fail gracefully - always re-run the specific repro after the fix, not just
the full test suite, to catch this.

**`addInterfaceMethod` (was :9342, the `methodName.empty()` guard): fixed,
but unlike every other fix in this document, reachability was *not*
verified with a repro this time** - this one already had `llvm_unreachable`
immediately followed by a dead `return mlir::failure();` (the exact
"crash-then-already-written-graceful-return" shape flagged early in this
audit for `UnaryBinLogicalOrHelper.h:42-43`, which turned out to be fully
dead code). `location` was already in scope, so the fix was a trivial
one-line swap (`emitError(location, "interface method name cannot be
empty"); return mlir::failure();`), cheap enough to apply defensively
without spending time on a repro attempt (an interface method with a
computed name is one plausible trigger, not investigated).

Still open from this cluster: `MLIRGenImpl.h:8164` (`processTypeParameter`'s
empty-name branch) and one of the "three close together" sites - the
`TypeAliasDeclaration` empty-name `else` (was :8414, the fourth of the
original 8382/8400/8426 trio once read in full) - both look parser-
grammar-guaranteed unreachable (a type alias or type parameter without a
name shouldn't parse at all) but neither was actually verified against the
parser; left as genuinely unread rather than assumed dead.

Verified via repro (both real crashes) and the full suite: `ctest -C Debug
-j8` → 829/829, no regressions.

### 4.12 Seventh pass — closes §5.1 entirely: `MLIRGenImpl.h`'s last 2 sites, `MLIRGenInterfaces.cpp`'s remaining 2, all of `MLIRGenTypes.cpp` (10 sites), `LLVMCodeHelper.h:452`, plus 2 bonus finds

**`MLIRGenImpl.h:8164` (`processTypeParameter`) and `:8382`
(`mlirGen(TypeAliasDeclaration...)`)**: the two sites left open at the end
of §4.11. Both are the "name came back empty" branch for a construct whose
grammar requires an `Identifier` (`<T>` / `type X = ...`), so - like
`addInterfaceMethod` before them - genuinely look parser-guaranteed
unreachable and neither was reproduced. Fixed anyway per this audit's
running "don't leave a landmine" policy: `mlirGen(TypeAliasDeclaration...)`
got a trivial `emitError` swap (a `mlir::failure()` return already existed
one line below); `processTypeParameter` needed a real code change since its
only path back to callers is a `TypeParameterDOM::TypePtr` - callers
(`processTypeParameters` et al.) push whatever it returns into a vector and
later dereference it, so returning `nullptr` for "fail" would just move the
crash one level up. Fixed by emitting the error and returning
`std::make_shared<TypeParameterDOM>("")` - a valid, if semantically empty,
object the caller can hold without crashing.

**`MLIRGenInterfaces.cpp`'s two remaining sites** (this doc's original
:475/:932/:954 - line numbers drifted from PR #302, which had already fixed
the middle one, "unsupported interface member", before this pass started):

- **The `extends` heritage-clause `TypeSwitch::Default` (interface
  extending a non-InterfaceType/TupleType target): a real, confirmed
  crash.** `interface X extends SomeClass {}` - real, documented TypeScript
  (extending an interface from a class's instance shape) - hit this
  immediately (confirmed via the "idle-CPU-but-actually-blocked-on-an-
  invisible-abort-dialog" pattern from [[windows-assert-dialog-looks-like-hang]],
  same as §4.10's `getIntTypeAttribute` finding). Root-caused further:
  `mth.getFields` (the same helper the `TupleType` `Case` above it already
  called) *already* handles `ClassType`, `ConstTupleType`, `ObjectType`,
  `ArrayType`, and `StringType` internally - so the fix was to make
  `Default` call `mth.getFields` too (generalizing, not hand-rolling a new
  per-type `Case`) and only `emitError` if that itself fails. This
  incidentally fixes `ConstTupleType`/`ObjectType`/`ArrayType`/`StringType`
  extends-targets as a side effect, not just the `ClassType` one that was
  reproduced.
- **`getNameForMethod`'s non-`StringAttr` computed-name branch: a real,
  confirmed crash.** `interface X { [1](): void; }` (a constant-foldable
  non-string computed method name) reaches this - `1` folds to a
  `ConstantOp` with an `IntegerAttr`, which isn't a `StringAttr`. Converted
  to `emitError` + `return {"", false}` (the same failure shape the
  function already returns one branch above for the `mlir::failed(result)`
  case).

**Bonus find, not in the original "not implemented" grep**:
investigating the numeric-computed-method-name repro above first hit a
*different*, more severe crash before reaching `getNameForMethod` at all -
`getNameWithArguments` (`MLIRGenImpl.h`, the general
name-or-anonymous-name-synthesis helper used by every function-like
declaration, not just interface methods) had `return nullptr;` inside a
function returning `std::string`. That's UB: `nullptr` converts to `const
char*`, and `std::string(const char*)` calls `strlen` on it, which
segfaults. WinDbg/ProcDump confirmed the crash (`strlen` →
`basic_string::basic_string<char*>` → `getNameWithArguments:9277` in the
stack). Worse, this early return also *skipped* the anonymous-name
synthesis fallback every other empty-name case in the same function falls
through to - so the fix was to delete the early return entirely (not just
guard it), letting the computed-name failure fall through to that existing
fallback. This bug was reachable for *any* function-like declaration whose
computed name fails to resolve (not just the specific interface-method
repro that found it), making it plausibly the most broadly-reachable single
bug found in this entire audit.

**All 10 sites in `MLIRGenTypes.cpp`** (this doc's :183/1474/1567/1876/
1910/2001/2204/2210/2661/3401 list, closing §5.1's last file):

- **`getType(Node, ...)` master dispatcher (:183) and `getTypeByTypeName`
  (:277, an already-exhaustive `Identifier`/`QualifiedName` EntityName
  dispatch) and `getResolveTypeParameter(TypeParameterDeclaration...)`
  (:250, another empty-type-parameter-name case, same shape as
  `processTypeParameter` above)**: converted to `emitError` +
  `mlir::Type()`/graceful-return without a repro; `getType`'s only
  identified-but-unconfirmed gap is `ImportType` (`import("mod").Foo`,
  grepped and confirmed unhandled anywhere in the codebase) - a real gap,
  just not cheap to reproduce (needs a second module) so left undemonstrated.
- **`RecordType`'s and `OmitTypes`' non-literal-key `Default`/`else`
  branches (:1474, :1567): confirmed dead code**, by the same call-site-
  trace method §4.6-4.8 used for the `funcRef` family. `Record<K,T>`,
  `Pick<T,K>`, `Omit<T,K>`, `Exclude<T,U>`, `Extract<T,U>` are *all*
  declared as ordinary generic type aliases in the default lib
  (`lib.generics.ts`, e.g. `type Record<K extends string|number|symbol, T> =
  { [P in K]: T }`), and `getTypeByTypeReference` always tries
  `resolveGenericType` before ever reaching the builtin-fallback family
  these two functions belong to (`getEmbeddedTypeWithManyParamsBuiltins`) -
  confirmed live by testing `Record<string,V>`/`Record<number,V>` and
  watching them route through the mapped-type machinery (a *different*,
  unrelated "mapped type is empty for constrain" warning) instead of
  crashing. Fixed to no-op (matching `PickTypes`' sibling
  `pickTypesProcessKey`, which already silently ignores an unrecognized key
  shape) rather than crash, for the "don't leave a landmine" reason, not
  because either is expected to ever actually run.
- **`getMappedType`'s two `Default`/`else` branches for `as`-remapped
  mapped-type keys (:2204, :2210): unconfirmed, converted defensively.**
  These fire only when a mapped type's `as` name-remapping clause produces
  a shape (non-literal, or a union/non-union mismatch against the value
  type) that couldn't be constructed from a simple repro attempt in the
  time available; left as genuinely untested rather than claimed dead,
  same honesty standard as the rest of this document.
- **`getTypeOperator`'s final fallthrough (:1876): grammar-exhaustive**
  (`unique`/`keyof`/`readonly` are the only three TS type-operator
  keywords, all three handled above) - converted defensively, no repro
  expected to exist.
- **`getIndexedAccessType`'s `StringType`-index branch (:1910): a real,
  confirmed crash.** `type X = Foo[string]` (indexing a type by the general
  `string` type rather than a specific literal key - invalid in real TS
  unless the target has a string index signature, which this compiler's
  indexed-access resolution doesn't model) crashed instantly. This function
  had no `mlir::Location` parameter at all (nor did its 3 recursive
  self-calls or its `IndexedAccessTypeNode` entry point) - threaded one
  through the whole call chain (5 call sites total, all in this file) rather
  than falling back to some contextless diagnostic.
- **`getIndexedAccessType`'s final fallthrough (:2001), same function,
  same location threading applies**: unconfirmed (nothing tried reached
  it), converted defensively using the same threaded `location`.
- **`getTupleFieldInfo(TypeLiteralNode...)`'s member-kind `else` (:2661): a
  real, confirmed, easily-reachable crash.** A `TypeLiteral`
  (`type X = { ... }`) resolves to this compiler's plain-value `TupleType`
  representation, and this function only handled
  `PropertySignature`/`MethodSignature`/`ConstructSignature`/
  `IndexSignature`/`CallSignature` - missing `GetAccessor`/`SetAccessor`,
  which real TS type literals do support (`type X = { get foo(): number }`).
  Confirmed via repro. Unlike `InterfaceDeclaration` (which has its own
  accessor/vtable machinery in `MLIRGenInterfaces.cpp`), a `TupleType` has
  no getter/setter dispatch representation to plug an accessor into at all
  - implementing this for real is a separate, scoped feature (same
  "real gap, not dead code, out of scope for a crash-audit pass" category
  as §4.2's array/union spread and §4.9's union-tuple `TypeOf` cast).
  Converted to a clean `emitError` instead. The final catch-all `else`
  (any *other* unhandled member kind) got the same treatment.
- **`getLiteralType(LiteralTypeNode...)`'s final fallthrough (:3401):
  unconfirmed.** Tried negative numeric literals (`-1`) and `BigIntLiteral`
  (`100n`) as plausible triggers - both already handled cleanly (fold to a
  `ConstantOp` before reaching this point). Converted defensively without a
  confirmed repro.

**`LLVMCodeHelper.h:452` (`getArrayValue`'s final fallthrough, LLVM-lowering
stage): unconfirmed, likely dead per this doc's own earlier hypothesis.**
Tried an array of `null` elements and a mixed `number|string` union array as
plausible triggers for an unhandled const-array-literal element type -
neither reached this branch (both already handled by earlier cases in the
same function). This file has zero prior `emitError`-style diagnostics
anywhere in it (this pass of the audit is LLVM-lowering-stage code, past
the point normal MLIRGen diagnostics are emitted) - used the `op->emitError`
idiom instead (confirmed as a live pattern via `LowerToLLVM.cpp`'s own
debug-assert error calls) plus a `return mlir::Value()`, rather than
`llvm_unreachable`.

**Bonus fix, the twin this document itself had already flagged**:
`MLIRGenImpl.h:7907` ("object literal is not implemented(1)") - §5.1's table
entry for `LLVMCodeHelper.h:452` speculated this was its already-confirmed-
dead MLIRGen-level twin, but the `llvm_unreachable` itself had never
actually been replaced. Verified the grammar-exhaustiveness claim by reading
`mlirGenObjectLiteralFields`'s full dispatch (`PropertyAssignment`|
`ShorthandPropertyAssignment`|`SpreadAssignment`|`MethodDeclaration`|
`GetAccessor`|`SetAccessor` - the complete `ObjectLiteralElementLike`
grammar) and converted it to the standard `emitError` + graceful-failure
shape while here, rather than leave the confirmed-dead crash in place.

Verified: every confirmed-crash repro re-run clean after the fix (no
crashes, clean diagnostics), plus the full suite: `ctest -C Debug -j8` →
**829/829, no regressions**.

### 4.13 Eighth pass — closes §5.2 entirely (12 files), several more real crashes found

Worked through every site in §5.2's inventory list, file by file, using the
same repro-when-possible / defensive-fix-when-not approach as every prior
pass. Rebuilt and ran the full suite after each file (not just once at the
end) specifically so a bad fix would be caught close to its cause; 829/829
green after every single rebuild in this pass, no regression ever surfaced.

**`CastLogicHelper.h` (LLVM-lowering-stage `cast()` helper), 7 sites**:

- **`castToArrayType`'s final `else` (was :1002): a real, easily-reachable
  crash.** Widening `number[]` to an `any[]` function parameter (an entirely
  ordinary pattern) crashed - the function only ever handled a source that
  was itself `ConstArrayType`/`NullType`/`UndefinedType`, never a plain
  dynamic `ArrayType`. Properly supporting it needs a runtime per-element
  boxing loop (`number` and boxed `any` aren't bit-compatible) - a separate,
  scoped feature, not a quick fix. Converted to a clean `emitError`.
- **The other 6 sites (array-to-string, tuple-to-interface ×2, union-to-
  boolean-with-RTTI-tag, optional-to-other-type), all marked "must be
  processed at MLIR pass": confirmed dead or blocked upstream, not
  reproduced.** Array-to-string and tuple-to-interface casts were confirmed
  to already work end-to-end via direct JIT repros without ever reaching
  these branches (they're resolved by a dedicated MLIR-level pass/pattern
  before LLVM lowering, exactly as their own comment says). The union-to-
  boolean case is blocked one stage earlier by MLIRGen's own union-to-
  boolean rejection (`castFromUnion`, §4.9) - confirmed via two repro
  attempts (`if (x)` and `!!x` where `x: number | {a: number}`), both
  already fail cleanly before LLVM lowering runs. The optional-to-other-type
  case wasn't reproduced despite two attempts (widening to a broader union,
  casting through `any`). All 6 converted to `emitError` defensively.

**`MLIRGenClasses.cpp`, 5 sites**:

- **The `extends` and `implements` heritage-clause `TypeSwitch::Default`s
  (was :603, :635, plus :1802's mirror of the `implements` one): both real,
  confirmed crashes.** `class Sub extends SomeInterface {}` (real TS itself
  rejects extending an interface, but this compiler didn't check) and
  `class Square implements SomeTypeAlias` where the alias resolves to an
  object-shape `TupleType` rather than a declared interface (real, valid,
  common TypeScript - this compiler's `implements` only accepts a genuine
  `InterfaceType`, so this is a real gap, not an error case). Fixing the
  `extends` one surfaced a **second bug in the same spot**: the original
  code had no failure-tracking at all in that loop (unconditionally
  `return mlir::success()` regardless of which `TypeSwitch` case fired), so
  the first attempt's `emitError` was silently swallowed - the repro
  compiled "successfully" with the class heritage just quietly dropped.
  Caught by testing the exact repro after the fix (not just the full
  suite), per §4.11's own lesson about second bugs one level up. Added a
  `success` bool matching the sibling `implements` branch's existing
  pattern.
- **Two empty-member-name guards (was :2269, :2306): unconfirmed.** Tried a
  computed class-property initializer (`class X { [key] = 42; }`) and a
  parameter-property combined with a destructuring pattern
  (`constructor(public {x,y}: T)`, itself invalid real TS - TS2369) as
  plausible triggers; both hit a *different*, undocumented crash first (an
  LLVM `Casting.h` `dyn_cast on a non-existent value` assertion, in
  `getNameWithArguments`'s computed-property-name path or nearby - not
  chased further, out of this audit's "not implemented" scope, noted here
  as a discovered-but-unfixed bug for a future pass). Converted both
  defensively regardless.

**`MLIRGenExpressions.cpp`, bonus find + 3 sites**:

- **Bonus, not in the original grep**: the master `mlirGen(Expression...)`
  dispatcher's final `llvm_unreachable("unknown expression")` (a different
  message, missed by the original `"not implemented"` grep) - found while
  investigating the ctor-destructuring-parameter-property repro above.
  Reachable via that exact same invalid-but-unchecked pattern (a
  `BindingPattern` name node, not an `Expression` kind, reaching this
  dispatcher). Converted to `emitError`.
- **`mlirGen(DeleteExpression...)`'s final `else` (was :996): a confirmed
  real crash on first repro, but not reproducible afterward.** `delete
  obj.a` on a plain mutable-tuple object literal crashed on the first
  attempt (`UNREACHABLE executed at ...:996!`, exact repro, exact line) -
  but re-running the identical repro after the fix (and several variants:
  `const` instead of `let`, an rvalue tuple from a function call) never
  reached the new code path at all; property access on a tuple value
  apparently always yields a `RefType`, which this branch's guarding `if`
  already excludes. Genuinely unclear why the first run differed - noted
  honestly rather than claimed as a confirmed live bug. Fixed anyway
  (real TS itself rejects deleting a non-optional property, TS2790 - this
  compiler doesn't check that either, and there's no way to "delete" a
  fixed-layout tuple field regardless): `emitError` instead of crash.
- **Prefix/postfix unary operator `default` branches (was :538, :560):
  grammar-exhaustive** (`+ - ~ ! ++ --` prefix, `++ --` postfix are the
  complete TS operator sets, all handled above) - converted defensively,
  no repro expected to exist.

**`MLIRGenGenerics.cpp`, 4 sites**:

- **`instantiateSpecializedFunction`'s type-argument-resolution `else`
  (was :911, in the middle of the file's numbering): a real, confirmed
  crash.** `function foo<T>(): T { ... }` called as `foo()` - no explicit
  type argument, and nothing to infer `T` from (zero parameters, zero call
  operands). Real TS rejects this too ("type argument cannot be inferred
  from usage"). Converted to a clean `emitError`.
- **Three more (function-ref-shape dispatch ×2, a `CreateBoundFunctionOp`
  result-type switch): unconfirmed or construction-guaranteed.** The
  `CreateBoundFunctionOp` one is dead by construction (that op's result can
  only ever be `BoundFunctionType`/`HybridFunctionType`, both handled).
  The other two (a generic function reference that's neither wrapped in
  `CreateBoundFunctionOp` nor a bare `SymbolRefOp`/`ThisSymbolRefOp`) each
  had one plausible trigger tried (a generic function assigned to a
  variable then called; `super.genericMethod(...)`) and neither reached the
  crash - both resolve through the already-handled paths. Converted
  defensively.

**`MLIRGenImpl.h`'s §5.2 cluster, 7 sites**:

- **`mlirGenSaveLogicArray`'s and `mlirGenSaveLogicObject`'s
  `TypeSwitch`/dispatch `Default`s (destructuring-assignment codegen,
  2 sites): both real, confirmed crashes.** `[a, b] = "hi"` (array-
  destructuring from a string - this compiler's destructuring-assignment
  codegen only positionally indexes Array/ConstArray/Tuple/ConstTuple, never
  gained the iterator-protocol handling `for...of` has) and `({ a, ...rest
  } = obj)` (rest-destructuring in an object-destructuring assignment - a
  real, separate missing feature, same shape as §4.2's object-literal
  spread gap: synthesizing `rest` needs enumerating every not-yet-
  destructured field). Both converted to clean `emitError`s; the array one
  needed an added `hasUnsupportedType` flag since the original code had no
  failure path out of that `TypeSwitch` either (same "silently swallowed"
  risk as the `MLIRGenClasses.cpp` extends fix above - caught this time
  before testing, not after).
- **The switch-case jump-op dispatch, the prefix-unary-on-constant switch,
  and a nested nested-`isDynamicImport` RTTI branch (3 sites): construction-
  or call-site-guaranteed dead**, all converted defensively without a
  repro attempt (the jump-op one only ever receives a `CondBranchOp`/
  `BranchOp` by construction two lines below; the constant-unary switch's
  only caller already guards with the same 4-operator check; the RTTI one
  is nested inside `isDynamicImport`, itself an untested feature path per
  §4.10).
- **The spread-iterator-protocol `TypeSwitch`/`else` pair (`hasIterator`
  and `processArrayValuesSpreadElement`, 2 sites each = the remaining
  entries): the array-spread one is a real, confirmed crash.**
  `[...new MyIter()]` where `MyIter`'s `next(): any` (a custom iterator
  whose `next()` return type isn't a `Tuple`/`ConstTuple`) crashed in
  `processArrayValuesSpreadElement`. The `for...of` sibling (`hasIterator`)
  wasn't reproduced with the identical class (a `for (let x of
  new MyIter())` repro hit an unrelated, pre-existing "any"-property-
  resolution error first) but shares the exact same `TypeSwitch` shape, so
  fixed identically. Both `Default` branches now treat an unrecognized
  `next()` return shape as "not a well-formed iterator" (return
  `false`/`emitError`) rather than crash.

**`CodeLogicHelper.h:241` (`saveResult`, LLVM-lowering-stage `++`/`--`
codegen): a real, confirmed crash, and the oldest-documented gap in this
whole audit** - the function's own comment already said `// TODO: finish it
for field access` before this pass touched it. `b.v++` where `v` is a
get/set accessor pair (not a plain field) crashed: a plain variable/field
increment loads through a `LoadOp` and stores straight back to its
reference, but an accessor has no reference to store to - it would need
calling the setter with the incremented value instead, a real, separate
feature. Plain `obj.field++` (the common case) was confirmed to already
work fine first, to isolate exactly what the TODO was still missing.
Converted to `mlir::emitError` (this file had no prior diagnostic-emission
precedent at all, unlike the sibling LLVM-lowering helpers).

**`OptionalLogicHelper.h` and `UndefLogicHelper.h`, 4 sites total
(2 each): confirmed dead by call-site trace, same method as the `funcRef`
family (§4.6-4.8).** Both files' `switch (opCmpCode) { ... default:
llvm_unreachable }` pairs handle exactly the 8 comparison `SyntaxKind`s
(`== === != !== > >= < <=`); traced every path `opCmpCode` can take back to
`LowerToLLVM.cpp`'s `LogicalBinaryOpLowering`, which dispatches via a
`switch` over the identical 8-way set before ever calling into either
helper - the `default` branches are unreachable for as long as those two
switches stay in sync. Converted to `mlir::emitError`/`op->emitError`
defensively rather than left crashing.

**`MLIRCodeLogic.h`, 4 sites: unconfirmed, converted defensively.**
An enum-member constant-attribute dispatch (`StringAttr`/`IntegerAttr`/
`FloatAttr`/`BoolAttr` handled, matching the string/numeric-literal-only
grammar for TS enum members) and three property-access helpers
(`Ref`/`Object`/`Class`) whose non-tuple/non-`ClassStorageType` branches
look reachable only through deep, already-extensively-tested class/object/
array property-access machinery; no repro attempt succeeded in the time
available. All four converted to `emitError` + a null-value return (the
enum one additionally needed an early-return guard added after the
`TypeSwitch`, since constructing a `LiteralType` from a null base type
afterward would just move the crash one line down).

**`MLIRPrinter.h:303` (`printAttribute`'s `Default`): likely a real gap,
not reproduced.** Handles `StringAttr`/`FlatSymbolRefAttr`/`IntegerAttr`/
`FloatAttr` but not `mlir::BoolAttr` - and `MLIRCodeLogic.h`'s own enum-
attribute dispatch (just above) treats `BoolAttr` as a *separate* case from
`IntegerAttr` in this codebase, suggesting a boolean-valued attribute
wouldn't already be caught by the `IntegerAttr` case here either. No
repro found in the time available. Changed to fall back to the
attribute's own default printer (`out << a;`), matching the sibling
`printType`'s own `Default` two dozen lines below (already non-crashing,
`out << t;` - was mis-flagged as still-crashing in this doc's §5.2
inventory; it wasn't).

**`MLIRTypeIterator.h:404`: not a "convert the crash" fix - a genuine,
previously-undetected missing `Case`, found by diffing against
`TypeScriptTypes.td`'s full type list rather than by reproducing a
crash.** This recursive type-walker (used by `isGenericType`/
`hasInferType`/`forEachTypes`/`getAllInferTypes` - i.e. called on nearly
every resolved type throughout the whole compiler) had `Case`s for every
one of this compiler's ~45 `mlir_ts::*Type` kinds *except*
`NamespaceType`. Grepped every `TypeScript_*` type def in
`TypeScriptTypes.td` and diffed against every `Case<mlir_ts::...Type>` in
this switch to find the gap - the same technique §4.6-4.8 used for the
`funcRef` family, just applied to a type-list instead of a call-site list.
Added the missing `Case` (a leaf type - just a name, nothing to recurse
into) rather than merely guarding the `Default`; also softened the
`Default` itself to a debug log instead of a crash, in case some other
type is still missing. Not confirmed via a live crash repro (a namespace-
qualified generic class instantiation, `NS.Box<T>`, was tried but the fix
was already in place by the time it ran) - documented honestly as a
static-analysis fix, not a reproduced one.

**`Win32ExceptionPass.cpp:584` (`getCallBundleFromCatchRegion`'s final
`else`): confirmed dead by construction, matching this file's own existing
convention for the identical invariant.** Every `CatchRegion` gets exactly
one of `catchPad`/`cleanupPad` set, at the exact site that first identifies
it (two `CatchPadInst::Create` call sites vs. one `CleanupPadInst::Create`,
mutually exclusive) - and this same file already asserts precisely this
invariant twice elsewhere (lines 393, 426) rather than treating it as a
diagnosable error. Matched that existing convention (`assert(...)`) instead
of introducing a new diagnostic pattern in a raw LLVM-IR pass that had
never used one.

Verified: every confirmed-crash repro re-run clean after its fix, plus the
full suite after *every* file's fixes in this pass (not just once at the
end): `ctest -C Debug -j8` → **829/829, no regressions**, every single time.

### 4.14 Ninth pass — fixes the two bugs §4.13 found but deliberately left unfixed

§4.13 explicitly deferred two crashes as "a different category, out of this
audit's `llvm_unreachable` scope." This pass root-caused and fixed both,
via careful code tracing rather than a live debugger - ProcDump couldn't
catch either crash (a plain `assert()` failure calls `abort()`, which exits
cleanly via `_exit(3)` with no Win32 exception raised at all, so neither
`-e` exception-monitor mode nor `-t` terminate-monitor mode captured a
useful stack; live-attaching WinDbg to set a breakpoint made the *same*
assert instead pop a blocking "Debug Error!" GUI dialog, since
`IsDebuggerPresent()` becomes true - a live debugger changes this specific
crash's behavior rather than just observing it). Abandoned the debugger
and traced both by reading code instead - both turned out to share one
root cause, `TupleFieldName()` in `MLIRGenTypes.cpp`.

**Root cause**: `TupleFieldName(Node name, ...)` returns a null
`mlir::Attribute` in two situations its callers didn't guard against:

1. `getNameFromComputedPropertyName` fails to extract a compile-time
   constant from a computed name (e.g. `[key]` where `key` is a `const`
   whose value isn't directly a `ConstantOp` - referencing it takes an
   extra symbol-resolution layer this extraction logic doesn't see
   through) - it already emits its own clean diagnostic before returning
   the null/failure pair, but `TupleFieldName` propagates the null
   `Attribute` onward with no failure signal for a non-computed-name caller
   to check.
2. `name` is a `BindingPattern` (`ObjectBindingPattern`/`ArrayBindingPattern`)
   rather than an `Identifier`/`ComputedPropertyName` - `MLIRHelper::getName`
   correctly returns empty for it (patterns have no simple name), but
   `getNameFromComputedPropertyName` only special-cases
   `SyntaxKind::ComputedPropertyName`, so a `BindingPattern` falls through
   to `TupleFieldName`'s own fallback path, which unconditionally does
   `mlirGen(name.as<Expression>(), genContext)` - `.as<Expression>()` on a
   node that isn't an `Expression` subtype at all, an invalid downcast.

Both failure modes manifest identically at the call sites: a null
`mlir::Attribute` (or an invalid AST-node cast) reaching an unguarded
`dyn_cast<mlir::StringAttr>(...)`/`mlir::cast<mlir::StringAttr>(...)` a few
lines later, which crashes with `Assertion failed: detail::isPresent(Val)
&& "dyn_cast on a non-existent value"` - `dyn_cast` (unlike
`dyn_cast_or_null`) requires its input to already be non-null/valid; none
of the crash sites were using the `_or_null` variant.

**Fix 1 - computed class-field name** (`class X { [key] = 42; }` where
`key` doesn't fold to a `ConstantOp`): added a null-check on `TupleFieldName`'s
result immediately in all 4 call sites inside `MLIRGenClasses.cpp`
(`mlirGenClassDataFieldMember`, `mlirGenClassStaticFieldMember`,
`mlirGenClassStaticFieldMemberDynamicImport`,
`mlirGenClassConstructorPublicDataFieldMembers` - the last one's trigger is
believed unreachable in practice since constructor parameter names can't
actually be computed property names, but guarded anyway for consistency and
because `dyn_cast` crashes on null regardless of how unlikely the null is).
Each returns `mlir::failure()` early, matching `getNameFromComputedPropertyName`'s
own already-emitted diagnostic rather than adding a second, redundant one.

**Fix 2 - parameter property + destructuring pattern**
(`constructor(public {x, y}: T) {}` - itself invalid real TypeScript,
TS2369, but this compiler didn't check that and crashed instead of
erroring): added a `SyntaxKind::ObjectBindingPattern`/
`SyntaxKind::ArrayBindingPattern` check directly in `TupleFieldName` itself,
before the unconditional `.as<Expression>()` cast - this fixes the root
function once, benefiting all ~13 call sites across the codebase (only the
`MLIRGenClasses.cpp` constructor-parameter path was confirmed reachable via
repro, but the fix is at the one shared choke point rather than duplicated
per caller).

Verified: both original repros give a clean diagnostic instead of crashing
(`"not supported 'Computed Property Name' expression"` for fix 1,
`"a binding pattern cannot be used as a field name"` for fix 2), plus the
full suite: `ctest -C Debug -j8` → **829/829, no regressions**.

### 4.15 Tenth pass — closes §5.3 (RTTI) and §5.4's remaining stray `MLIRTypeHelper.h` lines

Started with the stray lines first (cheap, same file already open from prior
passes), then moved to §5.3 (RTTI), the doc's own last untriaged pool.

**`MLIRTypeHelper.h` strays**:

- `getAttributeType` (was :2109) - `Default` for a field-id attribute that's
  neither `StringAttr`/`FloatAttr`/`IntegerAttr`/`TypedAttr`. Two `.ts`-level
  repro attempts (numeric interface keys `interface X { 1: string }` used via
  `keyof`; a computed boolean property key) both resolved cleanly without
  reaching this branch - field ids in this compiler are apparently always one
  of the four handled kinds. Left **unconfirmed** (not proven dead, no
  reachable trigger found either) but hardened anyway: returns
  `UnknownType` instead of crashing, mirroring the function's own `!attr`
  branch immediately above it.
- `getFields`'s final `Default` (was :2258) - every real caller (`grep`'d
  across the whole tree) already calls it via `mlir::succeeded`/`mlir::failed`,
  so the function's contract already assumes failure is a normal, handled
  outcome; the `noError=true` parameter already returns `failure()` for this
  exact branch, so making the `noError=false` path do the same (instead of
  crashing) is a zero-risk consistency fix, not a behavior change for any
  caller that checks the result.
- `getFieldIndexByFieldName`/`getFieldInfoByIndex` (was :2291/:2308) -
  **confirmed dead**: their only callers anywhere in the tree
  (`MLIRGenInterfaces.cpp`, vtable-patching logic) always construct the
  argument as `mlir_ts::TupleType` first (via `TupleType::get(...)` or a
  `cast<RefType>(...).getElementType()` that's itself always a `TupleType`),
  never any other shape. Hardened anyway (return `-1` / default `FieldInfo`)
  for consistency with sibling functions in the same file.
- `extendsType`'s two `// TODO: get it by function` sites (was :2685/:2709,
  the `TupleType`/`ConstTupleType` branches of a tuple-shaped `extends`
  target with an **unnamed** field, e.g. `T extends [number, string]`) -
  **confirmed real, reachable crash**, via:

  ```ts
  type IsPair<T> = T extends [number, string] ? true : false;
  let a: IsPair<{ 0: number, 1: string }> = true;
  ```

  `UNREACHABLE executed at MLIRTypeHelper.h:2686!`. Root cause: the
  field-matching loop only knew how to look up `srcType`'s corresponding
  field **by name** (`item.id`); a plain positional tuple element (`number`,
  `string` with no `x:`/`y:` label) has `item.id == nullptr`, and the branch
  for that case was still a stub. Fixed by falling back to a **positional**
  lookup: collect `srcType`'s fields once via the existing `getFields()`
  dispatcher (which already supports far more shapes than just
  tuple/const-tuple - interfaces, classes, arrays, strings, optionals - so
  this also generalizes past the tuple-vs-tuple case in the original repro),
  then index into that list by the unnamed field's position. Applied
  identically to both the `TupleType` and `ConstTupleType` branches (same
  TODO, same bug, same fix - the const-tuple sibling wasn't independently
  reproduced but is structurally identical code).

**§5.3 (RTTI)**: the doc flagged this as hardest-to-reach ("deep in a code
path that's hard to reach without a specific class-hierarchy-plus-exception
scenario") and lowest priority. Traced instead of guessed-and-reproed, since
the whole cluster is generic `TypeSwitch::Default`/width-mismatch fallbacks
with no named trigger:

- **Windows side** (`LLVMRTTIHelperVCWin32.h:141,156,169`,
  `MLIRRTTIHelperVCWin32.h:216,226,241`, plus `MLIRRTTIHelperVC.h:108`'s
  `getLandingPadType` fallback) - **confirmed dead** by caller trace. There
  are exactly **two** places in the entire codebase that ever create a
  `mlir_ts::ThrowOp`: the user-facing `throw` statement
  (`MLIRGenStatements.cpp`) and a mismatch-rethrow synthesized during
  `TryOp` lowering (`LowerToAffineLoops.cpp:1766`) - and that second one is
  gated to `!compileOptions.isWindows` (Windows's own `__CxxFrameHandler3`
  matches catch types at the OS level, so this compiler only needs the
  synthetic rethrow on non-Windows targets). Both the `throw` and
  `catch (e: T)` code paths already call a **graceful** twin of these
  functions at MLIRGen time (`setType(type, resolveClassInfo)` /
  `setRTTIForType`, which cleanly rejects unsupported types with
  `"Not supported type in throw"`/`"...in catch"` instead of crashing) before
  any TS-dialect value carrying that type can reach the lowering-stage
  `llvm_unreachable` versions - and the two versions accept exactly the same
  type set. Verified empirically too, not just by reading: `throw true`,
  `throw` a 64-bit int, `throw`/`catch` a class with inheritance, and
  `throw` an enum value all either get the clean MLIRGen-time error or
  compile+run successfully; none reach the crash sites. Hardened anyway
  (graceful `false`/no-op fallback instead of crash) for consistency and to
  fully close the item rather than leave it an asterisk.
- **Linux side** (`LLVMRTTIHelperVCLinux.h:113,128,142`,
  `MLIRRTTIHelperVCLinux.h:146,161,182,202,217,230`) - same dead-by-caller-
  trace reasoning applies to the `Default` branches (the synthetic-rethrow
  path here is real, since it isn't Windows-gated, but it already produces
  `mlir_ts::NullType`, which this file's `LLVMRTTIHelperVCLinux::setType` was
  already explicitly handling via a dedicated `.Case<mlir_ts::NullType>`
  before this pass touched it - not a crash). **But** found one genuine,
  well-evidenced real bug via a different route - not a repro, since no
  Linux/WSL build exists in this session to execute one (see below): the
  file has **two** `setType` overloads, one taking a `resolveClassInfo`
  callback (used by the MLIRGen-time graceful gate) and one without (used by
  `LowerToAffineLoops.cpp`'s catch-arg-type lowering). The
  `resolveClassInfo` overload already special-cased 64-bit integers
  (`setI64AsCatchType()`) alongside 32-bit; the other overload only had the
  32-bit case, falling through to `llvm_unreachable` for 64-bit ints. Since
  MLIRGen's graceful gate (which uses the `resolveClassInfo` overload)
  already accepts a `catch (e: i64)`/`throw` of a 64-bit int on Linux, that
  construct would reach the *other* overload during lowering and crash -
  a genuine asymmetry between two copies of what should be the same type
  list. Fixed by adding the matching `else if (width == 64)` case; also
  hardened the remaining `Default`/width-mismatch branches in both
  overloads for consistency. **`MLIRRTTIHelperVCLinux.h`'s
  `getClassInfoName` `default:` case (the doc's stray `:399`, actually at
  :418 by the time this pass reached it - the file had drifted since the
  doc was last updated)** - confirmed dead the same way: its 4th enum value
  (`TypeInfo::Value`) is filtered out by every caller's own switch before
  ever reaching this function (routed to `typeInfoValue()` instead of
  `typeInfoClass()`/`getClassInfoName()`); hardened anyway.
  **Honesty note**: none of the Linux-side code was execution-verified in
  this session - WSL Ubuntu *is* available in this environment (confirmed
  via `wsl --list`), but no Linux build of this project exists yet, and
  standing one up (LLVM/MLIR from scratch) was judged out of proportion for
  4 lines whose trigger condition is already well-understood via static
  trace and cross-checked against the file's own graceful-gate sibling. The
  Windows build **does** compile both the Win32 and Linux RTTI classes
  unconditionally (selected at runtime via `compileOptions.isWindows`, not
  `#ifdef`), so all of these edits at least compile cleanly and were
  exercised by the full Windows test suite - just not through an actual
  Linux/`__gxx_personality_v0` exception unwind.
  One incidental fix along the way: `LLVMRTTIHelperVCWin32.h` needed a
  local `#define DEBUG_TYPE "llvm"` / `#undef` bracket added (matching the
  pattern already used by `LLVMRTTIHelperVCLinux.h` and `MLIRTypeHelper.h`)
  since it had never used `LLVM_DEBUG` before this pass and several of its
  including `.cpp` files don't define `DEBUG_TYPE` themselves.

Verified: the `extendsType` repro above now compiles and runs cleanly
instead of crashing (plus a const-tuple variant of the same repro, which
resolves - to `false`, arguably a separate correctness question about
const-tuple-vs-tuple structural matching, out of this audit's crash-fixing
scope, same as other precedents in §7); `throw`/`catch` repros across bool,
64-bit int, enum, and class-with-inheritance all still compile+run
correctly on Windows with no behavior change; full suite
`ctest -C Debug -j8` → **829/829, no regressions**.

### 4.16 Eleventh pass — closes the last named site, `MLIRTypeHelper.h:410/420` (`convertAttrIntoType`)

New session, after PR #305 (tenth pass) merged. `convertAttrIntoType(attr,
destType, builder)` const-folds a constant attribute (from a literal array
element) into a target element type; its only caller is
`createConstArrayOrTuple` (`MLIRGenImpl.h:7343`), used when a const
array/tuple literal is being cast to a different element type than the one
inferred from its own literals (`arrayInfo.applyCast`) - e.g. a receiver
type annotation that disagrees with what the literal values would infer on
their own.

Three `.ts`-level repro attempts (`let arr: boolean[] = [1, 0]`,
`let arr: i32[] = [true, false]`, `let arr: string[] = [1, 2, 3]`) all
compiled and ran with no crash and no diagnostic - none reached either
`llvm_unreachable`. Read (not guessed) why: this compiler represents
boolean literals as plain builtin `i1` `IntegerAttr`, which already
satisfies `isIntOrIndex()` at the top of the int-handling branch, so a
bool-vs-int mismatch never reaches the `else` at :410 in the first place -
it's silently treated as an int/int conversion instead (a separate,
pre-existing type-looseness issue, out of this audit's scope). The
`string[] = [1,2,3]` case should exercise the final `Default` at :420
(`StringType` destination isn't `NumberType`/int/float), but three misses
is the audit's own established stopping point for blind repro attempts -
matching the pattern from the `funcRef` family (§4.6-4.8) - so this was
converted to a **static hardening fix** rather than pursued with a fourth
guess: an unconfirmed trigger is still worth defusing, especially since the
existing failure mode here (a null `mlir::Attribute` silently entering an
`ArrayAttr`) is worse than the audit's usual `llvm_unreachable` - it
wouldn't even give a "not implemented" message, just a delayed, confusing
crash somewhere downstream in the verifier or lowering.

**Fix**: both `llvm_unreachable` sites in `convertAttrIntoType` now return
`mlir::Attribute()` (null), matching the established convention for
`mlir::Type`-in/`mlir::Type`-out-style helpers with no `Location` in scope
(§2's fix-convention note) - and matching this exact function's own
sibling `convertFromFloatAttrIntoType`, which already returns null for its
own unsupported case a few lines above. Since the caller
(`createConstArrayOrTuple`) previously pushed whatever `convertAttrIntoType`
returned straight into the `ArrayAttr` with no check, added a null-check at
the call site that emits `"can't cast array literal element to '<type>'"`
and returns `mlir::failure()` cleanly instead - this is the actual
crash-preventing half of the fix, since the helper alone would just move
the problem one frame later.

Verified: the 3 original (non-crashing) repros still compile/run
unchanged; full suite `ctest -C Debug -j8` → **829/829, no regressions**.
This closes the last named/known-location site in this document - see §6
for what (if anything) is still open.

### 4.17 Twelfth pass — the audit's own scope had a hole: `LowerToLLVM.cpp` was never actually triaged, and `.td` files were never grepped at all

Prompted by a user spotting `TypeScriptOps.td:725` still crashing - a file
this audit's own grep command (§1) never covered, since `--include=*.cpp
--include=*.h` silently excludes TableGen `.td` files even though they can
embed live C++ (`extraClassDeclaration`, `builders`, etc.). Re-running the
grep with `--include=*.td` added, and separately re-checking `lib/TypeScript/
LowerToLLVM.cpp` specifically, turned up **14 more live sites this
document's own §1 overview had mentioned in passing ("mostly...
low-level LLVM-lowering... code") but §5's actual per-file inventory never
listed for investigation** - a real gap in this audit's own bookkeeping,
not just an unlucky miss. Breakdown: `TypeScriptOps.td:725` (1),
`DiagnosticHelper.cpp:104` (1), `LowerToLLVM.cpp` (12; `:6267`'s
IntersectionType site was already correctly triaged as dead in §3, so
excluded from this count).

**One real, reachable crash found**: `~x` (bitwise NOT) on a value with an
explicit raw float type (`f64`/`f32` - this compiler's low-level typed
variables distinct from the wrapped `number` type) crashed at
`LowerToLLVM.cpp:3070` (`NegativeOpBin`'s else branch). Root cause was at
MLIRGen time, not lowering: `mlirGen(PrefixUnaryExpression)`'s `~` case
(`MLIRGenExpressions.cpp`) only cast the operand to `i32` when it *wasn't*
already `isIntOrIndexOrFloat()` - but a raw float type already satisfies
that check, so it silently skipped the truncation every other numeric
representation (including the wrapped `number` type) already got. JS
`~x` always yields a 32-bit int (`ToInt32` semantics) regardless of input
type, so this wasn't just a missing cast, it was wrong for any raw-float
input. Fixed by tightening the condition to `!isIntOrIndex()` (cast unless
already int/index, matching real `~` semantics) so raw floats get the same
i32 truncation as everything else. Verified: `~3.5` on an `f64` variable
now compiles, runs, and yields `-4` (`ToInt32` semantics, printed as
unsigned `4294967292` = `0xFFFFFFFC`) instead of crashing.

**Everything else in this pass was confirmed dead** via caller trace, two
different strengths of proof:

- **TableGen-verifier-enforced exhaustiveness** (the strongest kind seen
  anywhere in this audit - not just "every caller I found," but "the
  verifier physically rejects anything else"): `LoadOp`'s `reference`
  operand is constrained to `TypeScript_RefOrBoundRefOrValueRef`
  (`RefType|BoundRefType|ValueRefType`) and `GetMethodOp`'s `boundFunc` to
  `TypeScript_BoundFunctionLike` (`BoundFunctionType|HybridFunctionType`) -
  both lowering patterns already handled every case the verifier permits.
- **Caller trace, C++-type-guaranteed**: `ArithmeticBinaryOp`,
  `ArithmeticUnaryOp`, `LogicalBinaryOp` (and by extension `StringCompareOp`/
  `AnyCompareOp`, which receive the same opcode `LogicalBinaryOp` already
  validated) are only ever constructed with opcodes from a small closed set
  - traced every construction site across the whole codebase (not just the
  obvious ones; `binaryOpLogic`'s generic `default:` dispatch in
  `MLIRGenExpressions.cpp` looked the riskiest since it forwards *any*
  leftover `SyntaxKind` into `ArithmeticBinaryOp`, but tracing back through
  `isNeededToSaveData`'s compound-assignment normalization and the
  `&&`/`||`/`??`/`in`/`instanceof`/`=` guards above it showed the reachable
  set is an exact match for the lowering switch's cases). `VirtualSymbolRefOp`/
  `ThisVirtualSymbolRefOp` were the interesting edge case: their TableGen
  result-type constraint (`TypeScript_AnyRefOrCallable`) is genuinely
  *broader* than what the lowering code handles (allows `ValueRefType`/
  `BoundFunctionType`/`HybridFunctionType`, not just `RefType`/`FunctionType`)
  - looked like a real gap until every actual construction site turned out
  to pass a C++-strongly-typed `mlir_ts::FunctionType`/`RefType` field
  (`FunctionEntry::funcType`, `MethodInfo::funcType`, or an explicit
  `RefType::get(...)`), never anything the broader constraint would also
  allow. `TypeScriptOps.td:725` (`CallIndirectOp`'s type-inferring builder)
  and `DiagnosticHelper.cpp:104` (`printLocation`'s `Location` TypeSwitch,
  exhaustive over all 6 of MLIR's builtin location kinds, no custom one
  registered anywhere in this dialect) were both dead the same way.

All were hardened to fail gracefully (`emitError`+`return failure()` where
a `Location`/op is in scope, matching this audit's established
convention) rather than left as bare crashes, even the TableGen-verifier-
proven ones - consistent with how this audit has always treated confirmed-
dead sites (§4.6-4.8, §4.15's Windows RTTI, etc.).

Verified: `~x`-on-raw-float repro now runs correctly (was crashing);
`ctest -C Debug -j8` → **829/829, no regressions**. This pass's own
lesson for whoever re-runs this audit's grep next: **re-check the grep
command against `--include=*.td` too**, and don't trust a file being
*mentioned* in this doc's prose as proof it was ever actually *triaged* -
cross-check against §5's per-file lists, which is where the real
bookkeeping lives.

## 5. Inventory of remaining markers (untested this pass)

Grouped by file. "Shape" is a guess from reading the surrounding code, not a
verified verdict — see §2 for how to actually check one.

### 5.1 Named/specific (cheapest to investigate next — read the message + local branch, write a 5-line repro)

**Fixed this pass**: `MLIRGenAccessCall.cpp`'s three sites (was lines
1159/1219/1535) — see §4.3-4.5. **Fixed a previous pass** (§4.9):
`MLIRGenCast.cpp`'s two `TypeOf` sites (was lines 1321-1322/1498-1499) — one
dead (guarded), one a real crash (union with a tuple-shaped member).
**Fixed this pass** (§4.10): `MLIRGenImpl.h`'s first four sites (was
5330/6732/7314/7418) — 1 real crash, 2 dead, 1 untested-feature-path.
`UnaryBinLogicalOrHelper.h:42-43` row removed - already resolved back in §3
(dead code, `UnaryOp<>` has zero call sites; see §6 item 1). **Fixed a
previous pass** (§4.11): `MLIRGenImpl.h`'s next three sites (was
8400/8426/9342 - `import X = require(...)`, `import X = <non-namespace/
class/interface>`, and `addInterfaceMethod`'s empty-name guard) — 2 real
crashes, 1 defensive/unverified. **§5.1 is now fully closed** (§4.12,
seventh pass): `MLIRGenImpl.h`'s last 2 sites, `MLIRGenInterfaces.cpp`'s
remaining 2 (the middle one, :932, had already been fixed by PR #302 before
this pass started), the entire `MLIRGenTypes.cpp` cluster (10 sites), and
`LLVMCodeHelper.h:452` (plus its `MLIRGenImpl.h:7907` MLIRGen-level twin,
which turned out to still be an un-patched crash despite this doc's own
earlier "confirmed-dead" note about it). No `.cpp`/`.h` named/specific
marker with a message naming its trigger remains as `llvm_unreachable`
today - what's left is exclusively §5.2's generic `TypeSwitch::Default`
fallbacks and §5.3's RTTI fallbacks, both harder to triage because they
carry no clue about their trigger beyond the enclosing type switch.

The table this section used to carry (unread sites) is gone - all of it is
now covered by §4.10-§4.12 above. See §4.12 for real-vs-dead-vs-unconfirmed
verdicts on every site, and the `MLIRTypeHelper.h` stray lines noted in §5.4
below for the one still-unread pocket adjacent to this list.

### 5.2 Generic `TypeSwitch::Default` exhaustiveness fallbacks — CLOSED this pass, see §4.13

Every site originally listed here (`MLIRGenClasses.cpp:603,635,1802,2269,2306` ·
`MLIRGenExpressions.cpp:530,552,988` · `MLIRGenGenerics.cpp:424,542,911,1342` ·
`MLIRGenImpl.h:3203,3477,3803,4402,4518,6379,7080,7094` ·
`MLIRGenInterfaces.cpp:656` · `CastLogicHelper.h:338,344,353,359,460,487,1002` ·
`CodeLogicHelper.h:241` · `OptionalLogicHelper.h:143,213` ·
`UndefLogicHelper.h:74,107` · `MLIRCodeLogic.h:1218,1660,1680,1722` ·
`MLIRPrinter.h:302-303,534` · `MLIRTypeIterator.h:403-404` ·
`Win32ExceptionPass.cpp:584`) is now fixed - see §4.13 for the full
real/dead/unconfirmed breakdown per site. `MLIRGenInterfaces.cpp:656` turned
out to already be resolved as a side effect of §4.12's interface-extends
fix, before this pass even started touching it directly.

This was this doc's own inventory of §5.2, not an exhaustive repo grep run
fresh - a new sweep might turn up more generic fallbacks this list never
included (see §7's caveat).

### 5.3 RTTI type-switch fallbacks — CLOSED this pass (tenth), see §4.15

`LLVMRTTIHelperVCWin32.h:141,156,169` · `LLVMRTTIHelperVCLinux.h:113,128,142` ·
`MLIRRTTIHelperVC.h:108` · `MLIRRTTIHelperVCWin32.h:216,226,241` ·
`MLIRRTTIHelperVCLinux.h:146,161,182,202,217,230,399` (the `:399` had drifted
to `:418` by the time this pass reached it). Windows side confirmed dead by
caller trace (only 2 `ThrowOp` creation sites in the whole codebase, both
already gated by a graceful MLIRGen-time twin check); Linux side mostly the
same, plus one genuine real bug (an int64-width asymmetry between two
`setType` overloads) fixed but not execution-verified - no Linux/WSL build
exists in this session. See §4.15 for the full writeup.

### 5.4 `MLIRTypeHelper.h`'s `funcRef` family — CLOSED this pass, see §4.6-4.8

`getReturnTypeFromFuncRef`, `getParamFromFuncRef`, `getFirstParamFromFuncRef`,
`getParamsFromFuncRef`, `getParamsTupleTypeFromFuncRef`, `getVarArgFromFuncRef`,
and `getOmitThisFunctionTypeFromFuncRef` (was :899, previously miscounted in
this list as an unrelated stray line) were all traced to their real callers
instead of unit-tested directly — call-site inspection turned out to be
faster than writing unit tests once it was clear every internal caller was
already `isAnyFunctionType`-guarded. Three were confirmed live crashes via
`ReturnType<T>`/`ThisParameterType<T>`/`OmitThisParameter<T>` with a
non-function `T`; all six are now fixed. See §4.6-4.8 for the full writeup.

Note `:410` and `:420`, also swept up in this line-number cluster originally,
turned out to be an unrelated numeric-attribute-conversion helper (constant
folding between int/float attrs), not part of the `funcRef` family - still
unread, left in the general backlog (a fresh repro/trace attempt would be
needed to triage it, same as any other never-investigated site in this
doc). `:2108, :2256-2257, :2290, :2307, :2685, :2709` (also confirmed not
`funcRef`-family) are now all resolved too - see §4.15 (tenth pass).

**Attempted this pass** (before switching to the call-site-trace approach
that actually worked): two plausible `.ts`-level triggers — a callback
parameter typed as a union of function signatures
(`type Cb = ((x: number) => void) | ((x: string) => void)`) — neither
crashed; both hit earlier, already-graceful type-mismatch errors instead.
Confirms `.ts`-level repro attempts weren't going to find this family's real
bug (the built-in utility types); tracing real callers is what worked.

## 6. Suggested next steps, in cost order

1. ~~`UnaryBinLogicalOrHelper.h:42-43`~~ — turned out to be dead code
   (`UnaryOp<>` has zero call sites), not a live one-line fix; see the new
   §3 entry. No action needed.
2. ~~The rest of §5.1 (named/specific)~~ — done this pass: found and fixed
   3 more real crashes (§4.3-4.5); only `MLIRGenCast.cpp`'s two `TypeOf`
   sites and the large `MLIRGenImpl.h`/`MLIRGenInterfaces.cpp`/
   `MLIRGenTypes.cpp` cluster remain from the original list.
3. ~~§5.4 (`funcRef` family)~~ — done this pass: traced every real caller
   (faster than the originally-planned unit-test approach), found and fixed
   3 more live crashes (§4.6-4.8); the other 3 functions in the family were
   fixed too even though proven dead, for consistency within the family.
4. ~~`MLIRGenCast.cpp`'s two `TypeOf` sites~~ — done this pass (§4.9): one
   dead (guarded), one a real crash (union with a tuple-shaped member,
   `<number>x` where `x: number | {a: number}`) — fixed. That was the last
   item from the original §5.1 named/specific list; only the large
   `MLIRGenImpl.h`/`MLIRGenInterfaces.cpp`/`MLIRGenTypes.cpp` cluster remains
   from §5.1, plus the stray
   `MLIRTypeHelper.h:410/420/2108/2256-2257/2290/2307/2685/2709` sites
   (confirmed *not* part of the `funcRef` family, see §5.4).
5. ~~The `MLIRGenImpl.h`/`MLIRGenInterfaces.cpp`/`MLIRGenTypes.cpp` cluster
   (§5.1's last remaining block)~~ — **done, §5.1 is now fully closed**.
   §4.10 closed the first 4 `MLIRGenImpl.h` sites (5330/6732/7314/7418);
   §4.11 closed 3 more (8400/8426/9342, 2 of them real `import X = ...`
   crashes); §4.12 (seventh pass) closed the rest: `MLIRGenImpl.h:8164/8382`,
   `MLIRGenInterfaces.cpp`'s 2 remaining sites, all 10 of `MLIRGenTypes.cpp`,
   `LLVMCodeHelper.h:452`, plus 2 bonus finds (a `getNameWithArguments`
   `return nullptr;`-as-`std::string` UB crash, and `MLIRGenImpl.h:7907`'s
   still-unpatched "confirmed dead" object-literal twin). 829/829, no
   regressions.
6. ~~§5.2 (generic fallbacks)~~ — **done, §5.2 is now fully closed** (§4.13,
   eighth pass): all 12 files' sites fixed, several real crashes found
   (array-to-`any[]` widening, `class`/`interface` heritage-clause type
   errors, accessor `++`/`--`, generic zero-arg inference failure,
   destructuring-assignment gaps) plus one genuine missing `Case`
   (`MLIRTypeIterator.h`'s `NamespaceType`) found by type-list diffing
   rather than a repro. 829/829, no regressions, checked after every file.
7. ~~§5.3 (RTTI)~~ — **done, §5.3 is now fully closed** (§4.15, tenth pass):
   Windows side confirmed dead by caller trace (verified empirically too);
   Linux side fixed a real int64-width asymmetry bug between two `setType`
   overloads, plus hardened the rest for consistency - not
   execution-verified (no Linux/WSL build stood up this session, judged out
   of proportion for 4 lines already well-understood via static trace).
8. ~~The stray `MLIRTypeHelper.h:2108/2256-2257/2290/2307/2685/2709`
   sites~~ — **done** (§4.15): one confirmed real crash (`extendsType`'s
   unnamed-tuple-field lookup, fixed with a positional fallback), two
   confirmed dead (`getFieldIndexByFieldName`/`getFieldInfoByIndex`,
   hardened for consistency), one left unconfirmed but hardened
   (`getAttributeType`), one zero-risk consistency fix (`getFields`).
9. ~~`MLIRTypeHelper.h:410/420` (`convertAttrIntoType`)~~ — **done**
   (§4.16, eleventh pass, new session after PR #305 merged): 3 repro
   attempts missed (this compiler represents booleans as plain `i1`, so a
   bool-vs-int mismatch never reaches the crash branch at all), so
   converted to a static hardening fix instead of a 4th guess - both sites
   now return null, and the one caller (`createConstArrayOrTuple`) now
   checks for that null and emits a clean diagnostic instead of building a
   malformed `ArrayAttr`. This was the **last named/known-location site in
   the entire document**.
10. Two bugs discovered but *not* fixed, out of this document's "not
    implemented" scope but worth a dedicated look: (a) an LLVM `Casting.h`
    `dyn_cast on a non-existent value` assertion, reachable via a computed
    class-property initializer (`class X { [key] = 42; }`) - see §4.13's
    `MLIRGenClasses.cpp` writeup; (b) `MLIRGenExpressions.cpp`'s
    delete-crash reproduced exactly once and then never again across
    several retries with the identical source - if it resurfaces, start
    from §4.13's honest note about it rather than assuming it's fixed.
11. The Linux RTTI execution-verification gap (§4.15) - fixes are applied
    and compile-verified but not run-verified, since no Linux/WSL build
    exists in this environment. The only way to fully close this would be
    standing up a Linux build of the project and exercising the C++
    exception-unwind path directly.
12. ~~`TypeScriptOps.td`/`DiagnosticHelper.cpp`/`LowerToLLVM.cpp`'s 14
    never-triaged sites~~ — **done** (§4.17, twelfth pass): found via a
    user spotting a still-live crash this audit's own grep command never
    covered (missing `--include=*.td`). One real crash fixed (`~` on a raw
    `f64`/`f32` value); the other 13 confirmed dead, several via a
    TableGen-verifier-enforced type constraint - the strongest proof of
    dead code found anywhere in this audit.

With items 1-9 and 12 all closed, **every named/known-location marker this
document ever tracked has been triaged, using a corrected grep command
that now also covers `.td` files.** What's left (items 10-11) is a
Linux-only verification gap and two out-of-scope bugs, plus the standing
caveat in §7 that a fresh repo-wide grep (for message patterns other than
this document's own `"not implemented"`, e.g. `llvm_unreachable`'s other
messages like "type mismatch" or "cast must happen earlier") might surface
markers this document's own inventory never included.

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
