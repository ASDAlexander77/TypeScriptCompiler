# Cross-module `.instanceOf` resolution under `-shared`: design

Status: **investigated, NOT implemented** — three fix attempts this session, each
found to cause real regressions, all reverted. Written up for a future attempt.
Follow-up to PR #274 (`Fix cross-module class extends crash and add regression
coverage`), whose commit message names this exact gap as a known, disabled issue.
All anchors below verified by code inspection and live `ctest` runs on
`main@89eb9869` during this investigation.

## 1. The problem

`export_class_extends.ts`/`import_class_extends.ts` and its multilevel/diamond
siblings (`test/tester/CMakeLists.txt`, search "KNOWN ISSUE") are disabled for
the `-shared` (AOT and JIT) variants. A derived class in one module extending a
base class from another, real DLL-boundary module fails to compile:

```
error: Class member 'M.Animal..instanceOf' can't be resolved (dynamic import)
```

from `ClassMethodAccess`'s `isDynamicImport` branch (`MLIRGenImpl.h:5447-5481`),
while the non-`-shared` variant of the same test (`test-compile-export-import-class-extends`)
passes today.

## 2. Root cause, part 1: `.instanceOf` never gets the decorator regular methods get

Two entirely different import mechanisms exist, selected by whether a compiled
DLL for the import target already exists on disk (`mlirGen(ImportDeclaration)`,
`MLIRGenModule.cpp:929-948`):

- **`mlirGenInclude`** (`MLIRGenModule.cpp:757-778`, used when no DLL exists,
  i.e. the non-`-shared` multi-file case): re-parses the *original source* of
  the imported file, `declarationMode = true`, and runs a full nested
  discover+codegen pair (`mlirDiscoverAllDependencies` + `mlirCodeGenModule`) on
  it *inside the importer's own process*.
- **`mlirGenImportSharedLib`** (`MLIRGenModule.cpp:780-900`, used when the DLL
  exists, i.e. `-shared`): does **not** re-parse source. It loads a *printed
  declaration string* embedded in the compiled DLL as a data symbol
  (`SHARED_LIB_DECLARATIONS`, produced by `DeclarationPrinter.cpp` when the
  exporting module was compiled), textually substitutes bare `@dllimport` for
  `@dllimport('.')` (line 875, only when `dynamic`), and re-parses *that* text
  via `parsePartialStatements`.

`DeclarationPrinter::printBeforeDeclaration` (`DeclarationPrinter.cpp:18-22`)
prints `@dllimport` exactly once, before the *class declaration* — never per
member. Regular instance methods (e.g. `speak()`) get no per-member decorator
either; they still work today only because nothing in the currently-enabled
tests calls `super.speak()` across the module boundary (see §5).

Class-level `@dllimport(...)` sets `newClassPtr->isDynamicImport = true`
(`MLIRGenClasses.cpp:273-282`, gated on `args.size() > 0` — the textually
substituted `'.'` argument). `mlirGenClassInstanceOfMethod`
(`MLIRGenClasses.cpp:1350-1445`) synthesizes `.instanceOf` fresh via
`NodeFactory` for *every* class regardless of source content (RTTI is never
written by the user) — but never attaches a decorator to the node it builds
(`modifiers` only ever gets a bare `PublicKeyword`, `MLIRGenClasses.cpp:1417-1421`
before this session's reverted edit). So when `mlirGenClassMethodMember`
processes it (`MLIRGenClasses.cpp:1868-1874` only special-cases
static/constructor/`.new` for `isDynamicImport` classes — not plain instance
methods), it falls through to the ordinary `mlirGenFunctionLikeDeclaration`
path, which checks the *method's own* decorators
(`MLIRGenFunctions.cpp:803-809`) — finds none — and emits a real (bodyless)
`FuncOp` instead of the "dlsym-style global variable" trampoline that
`mlirGenFunctionLikeDeclarationDynamicImport` builds for genuinely
`@dllimport`-decorated members. This asymmetry is already called out in a
comment at `MLIRGenImpl.h:5451-5461` ("Not every method... is actually
registered as a dlsym-style global variable... Try that first" — the
`theModule.lookupSymbol<FuncOp>` fallback PR #274 added).

## 3. Root cause, part 2: that FuncOp is structurally unresolvable anyway

Even granting part 1's asymmetry, the ordinary-FuncOp path *should* still work
via PR #274's `theModule.lookupSymbol<mlir_ts::FuncOp>(funcName)` fallback —
except it can't, for `-shared` specifically:

- `mlirDiscoverAllDependencies` (`MLIRGenModule.cpp:543-595`) wraps its own
  `DiscoveryModuleScope` (`MLIRGenImpl.h:9347-9369`), which redirects
  `theModule` to a **throwaway** `discovery_module` and erases it on scope
  exit. `mlirGenFunctionLikeDeclaration`'s `theModule.push_back(funcOp)`
  (`MLIRGenFunctions.cpp:915-918`) is additionally gated on
  `!funcDeclGenContext.dummyRun` — during discovery, `dummyRun = true`, so the
  FuncOp isn't even pushed into the (already-throwaway) discovery module.
- `mlirGenSourceFile` (`MLIRGenModule.cpp:193-235`) only runs the **real**
  codegen pass (`mlirCodeGenModule`, `dummyRun = false`, where the FuncOp
  *would* survive) `if (mlir::succeeded(mlirDiscoverAllDependencies(...)))` —
  i.e. discovery must fully converge (`processStatements`'s do-while loop,
  `MLIRGenModule.cpp:378-422`, must reach `notResolved == 0`) before the real
  pass ever starts.
- `import './export_class_extends'`'s own statement is processed once, inside
  discovery, via `mlirGenImportSharedLib` → `parsePartialStatements` — and
  because that succeeds as a language construct (`mlirGen(ImportDeclaration)`
  returns success), the statement is marked `processed = true` and never
  retried within discovery's own do-while loop. But `Dog`'s `.instanceOf` body
  (which does `super.instanceOf(...)`) is processed in the *same* discovery
  pass and needs `M.Animal..instanceOf`'s FuncOp *right then* — which, per the
  point above, was never pushed anywhere durable. `Dog`'s statement fails,
  `notResolved > 0` persists across retries (nothing about the situation ever
  changes since `import`'s statement isn't retried), discovery's do-while gives
  up, `mlirDiscoverAllDependencies` returns failure, and `mlirGenSourceFile`
  never reaches the real pass at all. Confirmed live: a
  `[DBG codegen] BEGIN module=...]` trace print in `mlirCodeGenModule` never
  fired in the failing repro.

This differs fundamentally from `mlirGenInclude`'s own nested discover+codegen
pair, which *does* reach a real (`dummyRun=false`) pass for the include file —
because that pair is self-contained and always runs both phases regardless of
convergence perfection *within its own recursive call*, whereas the **outer**
`mlirGenSourceFile`'s gate (discovery must fully succeed before real codegen
runs) is what actually blocks things for the main file being compiled.

## 4. Fix attempt 1 (this session): give `.instanceOf` the same decorator

Synthesize a `@dllimport('.')`-equivalent decorator on `instanceOfMethod` when
`newClassPtr->isDynamicImport`, mirroring what `mlirGenImportSharedLib`'s
textual substitution gives every other exported member — routing `.instanceOf`
through `mlirGenFunctionLikeDeclarationDynamicImport`'s `registerVariable`
(`fullNameGlobalsMap`, a persistent `MLIRGenImpl` member, **not** tied to
`theModule`/`DiscoveryModuleScope`) instead of the dummyRun-fragile FuncOp path.

This uncovered a **second, independent bug** en route: the generic
`dynamicImport` branch in `mlirGenFunctionLikeDeclaration`
(`MLIRGenFunctions.cpp:811-818`, pre-existing, previously only ever exercised
by plain top-level `@dllimport` functions) registers under
`funcProto->getNameWithoutNamespace()` with `isFullNamespaceName=false` — which
re-qualifies using only the *current namespace*, with no knowledge of an
enclosing *class* scope. For a class method this silently drops the class
segment (`"M.Animal..instanceOf"` → registered as `"M..instanceOf"`), while
`ClassMethodAccess`'s lookup uses `methodInfo.funcName`, the correctly
class-qualified name — a guaranteed miss. Fixed by using `funcProto->getName()`
(already correctly qualified) instead, matching the working sibling
`mlirGenClassMethodMemberDynamicImport` (`MLIRGenClasses.cpp:2038-2063`), which
already passes `funcOp.getName()` with `isFullNamespaceName`'s `true` default.

**Result: still failed**, with the *same* "can't be resolved" error — plus a
new `Assertion failed: HT.TopLevelMap[ThisEntry->getKey()] == ThisEntry &&
"Scope imbalance!"` crash (`llvm/ADT/ScopedHashTable.h:244`) on the failure
path. Traced to a **third bug**, below.

## 5. Root cause, part 3: `fullNameGlobalsMap`'s scoping is itself fragile

`fullNameGlobalsMap` (`MLIRGenImpl.h`, `llvm::ScopedHashTable<StringRef,
VariableDeclarationDOM::TypePtr>`) gets a `ScopedHashTableScope` pushed in
three places, unlike sibling maps (`fullNameClassesMap`,
`fullNameInterfacesMap`, etc.) which only ever get **one**, in
`mlirGenSourceFile` (`MLIRGenModule.cpp:207-218`, whole-compile lifetime):

1. `mlirGenSourceFile` (`MLIRGenModule.cpp:208-209`) — whole-compile, opened once.
2. `mlirDiscoverAllDependencies` (`MLIRGenModule.cpp:550-552`) — once per
   discovery invocation (including recursive ones via `mlirGenInclude`).
3. `discoverFunctionReturnTypeAndCapturedVars`'s **"simulate scope"**
   (`MLIRGenFunctions.cpp:409-414`, comment literally says "simulate scope") —
   once per speculative function-body discovery (`detectReturnType`,
   `MLIRGenFunctions.cpp:278-293`, which fires for essentially every function,
   even ones with an explicit return type, "due to captured vars").

`llvm::ScopedHashTable::insert` (`ScopedHashTable.h:193-195`) always inserts
into `CurScope` — whichever scope is topmost *at the moment of the call* —
regardless of which logical owner "should" hold the entry. Live tracing
(temporary `llvm::errs()` instrumentation, since removed) showed
`M.Animal..instanceOf`'s registration landing, and then vanishing
(`fullNameGlobalsMap.count(...)` dropping from 1 to 0), immediately after a
**sibling** method's (`M.Animal.constructor`'s) own "simulate scope" closed —
i.e. the registration was inserted while nested inside a transient,
already-slated-for-teardown scope that has nothing to do with `.instanceOf`
itself, and was discarded with it. In one run this same mechanism produced the
"Scope imbalance!" crash outright (a stricter LIFO violation, not just a quiet
disappearance) rather than a silent vanish — same root cause, worse symptom,
timing-dependent.

This is exactly why the PR #274 author's own two prior attempts ("deferring
synthesis until a non-speculative pass") caused an *infinite loop* in
unrelated same-module tests instead: deferring resolution under
`allowPartialResolve` only helps if some *later*, non-partial pass is
guaranteed to retry it — but per §3, a same-module class's own discovery can
also run entirely under `allowPartialResolve` with no such later retry ever
scheduled, so deferring indefinitely just spins the `notResolved` retry loop
forever.

## 6. Fix attempt 2: make `isFullName` registrations survive on a root scope

Captured the `mlirGenSourceFile`-owned scope in a new `rootGlobalsScope`
member, changed `registerVariableDeclaration`'s `isFullName` branch
(`MLIRGenVariables.cpp:76-79`) to `insertIntoScope(rootGlobalsScope, ...)`
instead of plain `insert(...)`, reasoning that a fully-qualified-name
registration represents a real, whole-compile-lifetime symbol (matching how
`fullNameClassesMap` et al. already behave).

**Result: catastrophic regression** — 70+ unrelated `ctest` failures (plain
non-shared JIT tests, interfaces, object literals, enums, vars — nothing
class-extends-related). Reverted immediately. Conclusion: **many other
`isFullName` registrations legitimately rely on being torn down when a merely
speculative/discovery scope ends** — e.g. presumably things registered
tentatively during discovery that a genuinely different, correct value must
replace during the real pass. Making *all* of them permanent broke that.
`mlirGenClassVirtualTableDefinition`'s own vtable registration
(`MLIRGenClasses.cpp:1695-1791`) already defends against the "does this
survive" question itself, live, via `if (fullNameGlobalsMap.count(...)) return
success();` (line 1710) re-checked fresh on every call — not a cached flag —
which is presumably why *it* doesn't exhibit this bug: it doesn't assume
persistence, it re-verifies.

## 7. Fix attempt 3: keep only the name-qualification fix (§4), drop the decorator

Reasoning: isolate which half of attempt 1 caused the regression. Reverted the
decorator synthesis (§4) but kept the `getName()` fix in the generic
`dynamicImport` branch alone, on the theory that it's a strict correctness fix
for an existing (if rarely-exercised) code path.

**Result: still regressed** — 10 of the original 20 failing tests (enums,
interfaces, a generic "component" test — nothing involving classes at all).
This means the generic `dynamicImport` branch's `getNameWithoutNamespace()` +
`isFullNamespaceName=false` combination is *not*, in fact, equivalent to
`getName()` + default-`true` for at least one other existing caller shape
(plausibly a namespaced non-class `@dllimport` declaration where
`currentNamespace` at that call site does *not* already match the name's own
namespace prefix — not yet root-caused). Reverted in full.

## 8. State at end of session

All three attempts reverted; `main`'s behavior is unchanged (the four tests
named in §1 remain disabled, exactly as PR #274 left them). Full `ctest`
(761/761) verified green on the reverted tree. The one change kept from this
session is unrelated: `tslang.cpp`'s `_CrtSetReportMode(_CRT_ASSERT,
_CRTDBG_MODE_FILE)` fix, which was itself missing (only `_CrtSetReportFile` was
set, which has no effect without also setting the mode) and caused every
`assert()` failure hit while iterating on this investigation to pop a blocking
"Assertion failed" dialog rather than just print to stderr — a friction-only
fix, not a behavior change to the compiler's actual TS semantics.

## 9. Recommendation for a future attempt

Given §6 and §7 both show that "just make registrations more persistent" and
"just fix the obviously-wrong name" each have *unexamined* dependents
elsewhere in the codebase, the next attempt should not touch
`fullNameGlobalsMap`'s scoping model or the generic `dynamicImport` branch at
all. Instead, consider a narrower target: give `.instanceOf` *its own*,
dedicated resolution path (not shared with either the ordinary-FuncOp
mechanism §2 exposes as broken, nor the generic dynamicImport/registerVariable
mechanism §5-7 show is fragile in ways not yet fully mapped) — e.g. a small,
purpose-built global (keyed and inserted exactly like
`mlirGenClassVirtualTableDefinition`'s vtable global, §6, which is the one
`fullNameGlobalsMap` consumer already proven immune to this class of bug
because it re-checks liveness on every call instead of assuming persistence).
Whatever the mechanism, verify with the *full* `ctest` suite (not just the
class-extends tests) before considering it done — both attempt-2 and
attempt-3's regressions were invisible from the class-extends tests alone and
only surfaced project-wide.
