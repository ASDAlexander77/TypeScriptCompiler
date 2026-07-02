# MLIRGen.cpp — Code Review & Refactoring Plan

*Reviewed: 2026-07-01. File: `tslang/lib/TypeScript/MLIRGen.cpp` (26,702 lines).*

## Snapshot

| Metric | Value | Note |
|---|---|---|
| Lines | 26,702 | 68% of all `lib/TypeScript` code (39,017 total) |
| Classes | 1 god class: `MLIRGenImpl` (lines 158–26,623) | plus a few small structs |
| Methods (approx.) | ~650 | all in one class body |
| Largest method | `cast(...)` — ~590 lines | next: `mlirGen(ObjectLiteralExpression)` ~520 |
| `TODO` comments | 206 | 22 mention "hack" |
| `const_cast` uses | 38 | mostly around `SourceMgr` and `GenContext` |
| `#if`/`#ifdef` | ~50 | WIN32 / platform / feature toggles |
| `genContext` mentions | 1,832 | threaded through every call |
| Error-handling macros (`EXIT_IF_FAILED`, `VALIDATE*`) | 156 | |

The file works, is battle-tested, and encodes a lot of subtle knowledge. The problem is not quality of individual logic but **monolith structure**: compile time (single TU), reviewability, merge conflicts, and the ease of introducing cross-cutting state bugs (e.g. the recent nested-discovery `theModule.getBody()->clear()` bug — exactly the class of bug a 26k-line stateful class invites).

---

## 1. Split the translation unit (highest value, lowest risk)

`MLIRGenImpl` has natural seams already visible in method naming. Split into partial-class-style units — same class, methods distributed across files (C++ allows defining member functions in any TU), or better, extract collaborator classes:

| Proposed file | Contents (existing methods) | Rough size |
|---|---|---|
| `MLIRGenModule.cpp` | module/discovery driver: `mlirGenSourceFile`, `mlirDiscoverAllDependencies`, `mlirCodeGenModule`, include/import loading (`mlirGenInclude`, `loadSourceBuf`, `loadIncludeFile`), shared-lib import | ~1,500 |
| `MLIRGenStatements.cpp` | statements: if/for/while/switch/try/labels | ~3,000 |
| `MLIRGenExpressions.cpp` | expressions: binary/unary/call/property access/object literal/array | ~6,000 |
| `MLIRGenFunctions.cpp` | function-like declarations, parameters, generics instantiation, capture | ~4,000 |
| `MLIRGenClasses.cpp` | class/interface/enum decl gen, vtables, RTTI, `.size`/`.rtti` statics | ~5,000 |
| `MLIRGenTypes.cpp` | `getType(...)` family, union/intersection/conditional types, `inferType` | ~4,000 |
| `MLIRGenCast.cpp` | `cast(...)`, `castFromUnion`, `castPrimitiveTypeFromAny`, safe-cast logic | ~1,500 |
| `MLIRGenConst.cpp` | const-eval: `isConstValue`, `evaluateBinaryOp` on constants | ~800 |

Benefits: parallel builds (this single TU dominates incremental build time), smaller review diffs, and the option to unit-test pieces (there is already a `unittests/` tree with `MLIRGenTests`).

Mechanically safest path: keep `MLIRGenImpl` as-is, move method *bodies* out with no signature changes, verify with the existing test suite after each batch.

## 2. Kill the dangling-`FuncOp` hazard in the symbol maps

`functionMap` (per-namespace `llvm::StringMap<mlir_ts::FuncOp>`) and `GenericFunctionInfo::funcOp` cache **raw op handles**. Discovery passes create ops and then erase them (`clearTempModule`, and until recently `theModule.getBody()->clear()`), so any cached handle from a discarded pass dangles. Today the code survives because consumers only read the *type* early — but this is exactly what made the nested-import bug subtle.

Refactor: store what's actually consumed — name + `mlir_ts::FunctionType` (+ flags) — in the map instead of the op:

```cpp
struct FunctionEntry {
    std::string fullName;
    mlir_ts::FunctionType type;
    bool isGeneric, hasBody, isPublic;
    // resolve to FuncOp lazily via SymbolTable when needed
};
```

Ops should be looked up through `mlir::SymbolTable` when a real reference is needed. This decouples "what we know about a function" from "an op that may no longer exist."

## 3. Replace mutable-member state + guards with an explicit context

The class mixes two kinds of state:

- **Session state** (fine as members): `builder`, `sourceMgr`, `compileOptions`, `theModule`, symbol maps.
- **Traversal state mutated and restored via `MLIRValueGuard`** (20 uses) or manual save/restore: `declarationMode`, `sourceFile`, `mainSourceFileName`, `stage`, `label`, `overwriteLoc`. Manual save/restore (e.g. `declarationModeStore` in `mlirGenClassSizeStaticField`) is not exception/early-return safe.

Refactor: move traversal state into `GenContext` (it is already threaded through all 1,832 call sites) or a small `TraversalState` struct passed by reference. Rule of thumb: *if a method needs to restore it before returning, it should not be a member.* This also removes most of the 38 `const_cast`s (many exist precisely because `GenContext` is passed `const` but the code wants to mutate traversal state).

## 4. Discovery pass: run once, not "dummy run then real run"

`mlirGenSourceFile` runs the whole AST twice: a `dummyRun/allowPartialResolve` discovery pass whose ops are thrown away, then real codegen. Costs: 2× walk, op churn, and cleanup fragility (see §2; the nested-import bug lived here).

Shorter-term hardening (cheap):
- Build discovery ops into a **dedicated throwaway `ModuleOp`** with its own builder instead of the real `theModule` — then cleanup is `discoveryModule.erase()` and can never touch real content. (This supersedes the current snapshot-and-erase in `mlirDiscoverAllDependencies`; the snapshot fix is correct but a separate module makes the invariant structural.) Note: requires routing the builder's insertion point for the whole discovery call tree, so do it as a deliberate change with tests, not a drive-by.

Longer term: replace the discovery codegen pass with a symbol-collection pass over the AST that populates the maps *without creating IR at all*.

## 5. Break up the giant methods

Worst offenders (line counts approximate):

| Method | Lines | Suggested decomposition |
|---|---|---|
| `cast(...)` (21169) | ~590 | dispatch table keyed on (srcKind, dstKind); each conversion a named helper — many already exist (`castFromUnion`, `castPrimitiveTypeFromAny`), finish the job |
| `mlirGen(ObjectLiteralExpression)` (15157) | ~520 | split: field collection, method/accessor gen, spread handling, const-object fast path |
| `inferType(...)` (1900) | ~300 | one helper per template-type kind (union, array, function, class, conditional) |
| `mlirGenClassVirtualTableDefinition` (19167) | ~280 | vtable-entry builder + interface-slot resolution helpers |
| `mlirGenPropertyAccessExpressionBaseLogic` (11118) | ~250 | per-receiver-type handlers (class, interface, enum, union, namespace, tuple) |

Pattern for all of them: they are long `if/else`-on-type chains; convert to `llvm::TypeSwitch` with one member function per case. This is mechanical and each extraction is independently testable/revertible.

## 6. Error-handling consistency

Three styles coexist: `EXIT_IF_FAILED*`/`VALIDATE*` macros (156 uses), explicit `if (mlir::failed(...)) return ...`, and `emitError(...) << ...` with ad-hoc returns. Also duplicated diagnostic literals (e.g. `"can't do binary operation on constants: "` ×6).

- Pick the macro family as canonical (it's the majority), document it at the top, convert stragglers opportunistically.
- Move repeated message literals into named helpers/constants so wording stays consistent (`emitBinaryConstOpError(...)`).
- The postponed-diagnostics machinery (`postponedMessages` in both discovery and codegen) is copy-pasted between `mlirDiscoverAllDependencies` and `mlirCodeGenModule` — extract a small `PostponedDiags` RAII helper.

## 7. TODO debt triage

206 TODOs / 22 "hack"s. Not all equal; the ones worth scheduled work:

- `// TODO: temp hack` in `prepareDefaultLib` path interplay (tslang.cpp) — default-lib resolution logic is duplicated in 4 places (`tslang.cpp`, `jit.cpp`, `exe.cpp`, `defaultlib.cpp`); centralize in one `DefaultLibLocator`.
- `// TODO: review usage of SizeOf in code, as size of class pointer is not size of data struct` (line ~18156) — correctness-adjacent, deserves a test.
- `// TODO: no need to clean up here as whole module will be removed` (discovery) — stale after the nested-discovery fix; update comments to match the new invariant.

Suggested policy: new TODOs must reference an issue number; do a one-time sweep to file the top ~20 as issues and delete the stale ones.

## 8. Smaller cleanups

- **`const_cast<llvm::SourceMgr &>(sourceMgr).setIncludeDirs(...)`** in the constructor: take `SourceMgr&` non-const in the constructor signature instead.
- **Platform `#ifdef`s** for shared-lib naming (`WIN_LOADSHAREDLIBS`/`LINUX_LOADSHAREDLIBS` in `mlirGen(ImportDeclaration)`): extract `resolveSharedLibName(StringRef) -> std::string` — currently the same dance appears in import handling and in `jit.cpp`.
- **`std::stringstream declExports` member**: accumulates exports globally and is reset with `declExports.str(""); declExports.clear();` — wrap in a small `DeclExportCollector` so reset/emission is one call and nesting behavior is explicit.
- **Header hygiene**: the implementation includes 59 headers; after the TU split, each unit should include only what it needs (biggest single lever on rebuild time after the split itself).
- **`isStatement(SyntaxKind)` / long switch tables**: several of these duplicate knowledge the parser already has; consider generating them or moving next to `SyntaxKind`.

---

## Suggested order of execution

1. **§4a** — discovery into a separate throwaway `ModuleOp` (removes a whole bug class; small, testable).
2. **§2** — de-op-ify `functionMap`/`GenericFunctionInfo` (prerequisite-free after 1, prevents dangling handles forever).
3. **§1** — mechanical TU split (no behavior change; do after 1–2 so moved code is the fixed code).
4. **§5** — giant-method decomposition, one method per PR.
5. **§3** — traversal state into `GenContext` (touches many lines; easier after the split).
6. **§6–§8** — opportunistic, alongside other work.

Each step is independently shippable and verifiable with the existing `test/tester` suite plus `unittests/MLIRGenTests`.
