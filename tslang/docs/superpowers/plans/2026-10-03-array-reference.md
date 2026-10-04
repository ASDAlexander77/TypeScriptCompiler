# Arrays as References Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make a tslang array value one pointer to a heap header `{ data, length, capacity }` that never moves, so every alias of an array sees `push`/`pop`/`splice`/`length =` (#453), and make a destructuring rest element a copy (#477).

**Architecture:** Three PRs on branch `array-reference`. PR 1 routes every access to the array layout in lowering through one new helper, `ArrayLayout`, with no change to the generated IR. PR 2 changes the helper's insides and the type conversion to the header, plus the sites that need more than the helper (rc/own routines, static arrays, the `main` argv, the rest copy, debug info). PR 3 adds capacity growth.

**Tech Stack:** C++17, MLIR/LLVM dialect conversion patterns (`TsLlvmPattern`), the tslang test runner (ctest), Git Bash + WSL.

**Spec:** `tslang/docs/superpowers/specs/2026-10-03-array-reference-design.md` (approved 2026-10-03).

## Global Constraints

- Header field order: `data` (ptr), `length` (index), `capacity` (index) - indexes `ARRAY_DATA_INDEX 0`, `ARRAY_SIZE_INDEX 1`, `ARRAY_CAPACITY_INDEX 2` (`tslang/include/TypeScript/Defines.h`).
- After PR 1, no `ARRAY_DATA_INDEX` / `ARRAY_SIZE_INDEX` / `ARRAY_CAPACITY_INDEX` outside `tslang/include/TypeScript/LowerToLLVM/ArrayLayout.h` (and `Defines.h`).
- Growth (PR 3): new capacity = `max(4, 2 * capacity, needed)`.
- Every memory model: `gc`, `rc`, `none`, `own`. own stays count-free: no `__tslang_inc_ref`/`__tslang_dec_ref` in what own emits (`test-own-no-counting-*`).
- Commits are GPG-signed; on a signing timeout retry once, then hand the user the exact `git commit -F <file>` command. Never bypass signing. Commit messages end with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.
- Source files are CRLF: edit with the Edit tool or Python with `newline=''`; never `sed -i` (it strips CRLF).
- Build: `cmake --build __build/tslang/windows-msbuild-2026-release --config Release --parallel 8 --target tslang`; full suite: `cd __build/tslang/windows-msbuild-2026-release && ctest -C Release -j 12 --timeout 300`. Compiler binary: `__build/tslang/windows-msbuild-2026-release/bin/tslang.exe`.
- Scratch paths used by the steps: `S="C:/Users/duzha/AppData/Local/Temp/claude/I--TypeScriptCompiler/f5d6099f-c9ad-473f-a7eb-6f10e2a42c61/scratchpad/arrref"`, `R="C:/Users/duzha/AppData/Local/Temp/claude/I--TypeScriptCompiler/f5d6099f-c9ad-473f-a7eb-6f10e2a42c61/scratchpad/reinsert.py"` (registers a test: `python "$R" <name-with-dashes> <file.ts> <compile-anchor.ts> <corpus-anchor.ts>`, run from `I:/TypeScriptCompiler`).
- Known unrelated failure: `test-compile-gc-defaultlib-collector` (stale local default lib in `I:\tslang`, `Error..size` 32 vs 48) until the default lib is rebuilt in Task 7.

## Rulings (spec gaps decided while planning)

- **R1, a null header is an empty array.** Today zeroed memory is a valid empty array: elements of `new Array<T[]>(n)`, class fields of array type with no initializer, and `undefined` cast to an array (`CastLogicHelper::castToArrayType`, `isUndef`). So: reading `data`/`length` of a null header gives null/0; an op that changes an array through its slot (`push`, `pop`, `shift`, `unshift`, `splice`, `length =`) first stores a fresh empty header into a slot that holds null; rc/own release of null does nothing (`emitIfLastReference` already skips null). Cost if wrong: one compare and branch per array read.
- **R2 (replaced 2026-10-04), `main`'s argv is `Ref<string>` only.** `main`'s argv is `Ref<string>` only (C's `char **`); a `string[]` argv is a compile error that names the `Ref<string>` form (owner's ruling 2026-10-04; replaces plan ruling R2). Spec section 5 has the same text. The original Task 5 (a `ts.ArrayFromCStrings` op) is dropped.
- **R3, the rest copy is built in MLIRGen.** Spec §3 says `ts.ArrayView` becomes a copy. A copy must own its elements under rc/own; MLIRGen's existing "fresh array + synthesized loop" idiom (`MLIRGenCast.cpp`, numeric array widening) gets that from the passes for free. So MLIRGen stops emitting `ts.ArrayView` for a rest element and builds `const .rest: T[] = []; for (let .i = index; .i < .src.length; .i++) .rest.push(.src[.i]);` instead. `ArrayViewOpLowering` stays for other users (none known; see Task 3 step 1).

## Review Focus

1. **Zeroed arrays (R1):** `const aa = new Array<number[]>(2); aa[0].push(1)` and a class field `items: number[];` pushed in a method must keep working - tests in Task 4 (`00array_zero_slots.ts`).
2. **Truthiness and null:** `[]` is truthy and `[] === null` is false after the switch (today both are wrong: an empty array has a null data pointer); `let a: number[] | null = null; if (a)` stays falsy - tests in Task 4.
3. **Constant arrays inside global data:** `const t: [number, number[]] = [1, [2, 3]]` and `const aa: number[][] = [[1], [2]]` at module level must still read correctly under rc (immortal static header) - tests in Task 4 (`00array_static_nested.ts`).
4. **Arrays crossing `any` and unions:** `const x: any = arr; (x as number[]).push(4)` is seen through `arr` - test in Task 4.
5. **Rest of a shorter array:** `const [a, b, ...rest] = [1]` gives `rest.length === 0` (no negative count) - test in Task 3.

---

## PR 1: the layout helper (no behaviour change)

### Task 1: `ArrayLayout` and the sites in `LowerToLLVM.cpp`

**Files:**
- Create: `tslang/include/TypeScript/LowerToLLVM/ArrayLayout.h`
- Modify: `tslang/lib/TypeScript/LowerToLLVM.cpp` - `LengthOfOpLowering` (~line 618), `SetLengthOfOpLowering` (~638), `CreateArrayOpLowering` (~3000), `NewEmptyArrayOpLowering` (~3070), `NewArrayOpLowering` (~3113), `ArrayPushOpLowering` (~3170), `ArrayPopOpLowering` (~3256), `ArrayUnshiftOpLowering` (~3305), `ArrayShiftOpLowering` (~3400), `ArraySpliceOpLowering` (~3465), `ArrayViewOpLowering` (~3650)
- Create (scratchpad, never committed): `ir-snapshot.sh`

**Interfaces:**
- Produces (used by every later task):

```cpp
class ArrayLayout : public LLVMCodeHelperBase
{
  public:
    ArrayLayout(mlir::Operation *op, PatternRewriter &rewriter, const TypeConverter *typeConverter, CompileOptions &compileOptions);

    // what the field GEPs are based on, for an op that changes the array held in `slot`
    mlir::Value headerForUpdate(mlir_ts::ArrayType arrayType, mlir::Value slot);
    // the same, for a routine that only reads (never materialises: R1)
    mlir::Value headerForRead(mlir_ts::ArrayType arrayType, mlir::Value slot);
    mlir::Value dataAddress(mlir_ts::ArrayType arrayType, mlir::Value header);
    mlir::Value lengthAddress(mlir_ts::ArrayType arrayType, mlir::Value header);
    // of an array value
    mlir::Value data(mlir_ts::ArrayType arrayType, mlir::Value array);
    mlir::Value length(mlir_ts::ArrayType arrayType, mlir::Value array);
    // what `===` and truthiness compare
    mlir::Value identity(mlir_ts::ArrayType arrayType, mlir::Value array);
    // a new array over `data` (owned by the array) with `length` elements
    mlir::Value make(mlir_ts::ArrayType arrayType, mlir::Value data, mlir::Value length);
    // a constant array inside a global's initializer: `data` is static, nothing is allocated
    // (Task 4 replaces this with makeStatic(arrayType, dataGlobalName, dataOffsetBytes, length))
    mlir::Value makeStatic(mlir_ts::ArrayType arrayType, mlir::Value data, mlir::Value length);
    mlir::Value zero(mlir_ts::ArrayType arrayType);
    mlir::Value undef(mlir_ts::ArrayType arrayType);
};
```

- [ ] **Step 1: Record the IR baseline.** Build the branch base (before any change) and save the compiler:

```bash
cd /i/TypeScriptCompiler && git log --oneline -1
cmake --build __build/tslang/windows-msbuild-2026-release --config Release --parallel 8 --target tslang
S="C:/Users/duzha/AppData/Local/Temp/claude/I--TypeScriptCompiler/f5d6099f-c9ad-473f-a7eb-6f10e2a42c61/scratchpad/arrref"
mkdir -p "$S" && cp __build/tslang/windows-msbuild-2026-release/bin/tslang.exe "$S/tslang-base.exe"
```

Write `$S/ir-snapshot.sh`:

```bash
#!/bin/bash
# ir-snapshot.sh <tslang.exe> <out-dir>: --emit=llvm of every suite test under gc, rc and own
T="$1"; OUT="$2"; mkdir -p "$OUT"
ls /i/TypeScriptCompiler/tslang/test/tester/tests/*.ts /i/TypeScriptCompiler/tslang/test/tester/own/*.ts |
  xargs -P 12 -I{} bash -c '
    f="{}"; b=$(basename "$f" .ts); d=$(basename $(dirname "$f"))
    for mm in gc rc own; do
      "'"$T"'" --emit=llvm --no-default-lib -mm=$mm "$f" -o "'"$OUT"'/$d-$b-$mm.ll" > "'"$OUT"'/$d-$b-$mm.err" 2>&1
    done'
```

Run it: `bash "$S/ir-snapshot.sh" "$S/tslang-base.exe" "$S/base"`. Expected: about 3 x 1,100 `.ll` files (some empty with an error in `.err`; that is fine, the comparison includes the errors).

- [ ] **Step 2: Create `ArrayLayout.h`** with the PR 1 implementation. Follow the include guard and namespace style of `LLVMCodeHelperBase.h`.

```cpp
#ifndef MLIR_TYPESCRIPT_LOWERTOLLVM_ARRAYLAYOUT_H_
#define MLIR_TYPESCRIPT_LOWERTOLLVM_ARRAYLAYOUT_H_

#include "TypeScript/Defines.h"
#include "TypeScript/TypeScriptOps.h"
#include "TypeScript/MLIRLogic/MLIRHelper.h"
#include "TypeScript/LowerToLLVM/LLVMCodeHelperBase.h"

// The one place that knows how an array value is laid out
// (docs/superpowers/specs/2026-10-03-array-reference-design.md). An array value is a
// { data, length } struct; a `slot` is a pointer to one.
class ArrayLayout : public LLVMCodeHelperBase
{
  public:
    ArrayLayout(mlir::Operation *op, PatternRewriter &rewriter, const TypeConverter *typeConverter,
                CompileOptions &compileOptions)
        : LLVMCodeHelperBase(op, rewriter, typeConverter, compileOptions)
    {
    }

    mlir::Value headerForUpdate(mlir_ts::ArrayType, mlir::Value slot)
    {
        return slot;
    }

    mlir::Value headerForRead(mlir_ts::ArrayType, mlir::Value slot)
    {
        return slot;
    }

    mlir::Value dataAddress(mlir_ts::ArrayType arrayType, mlir::Value header)
    {
        return fieldAddress(arrayType, header, ARRAY_DATA_INDEX);
    }

    mlir::Value lengthAddress(mlir_ts::ArrayType arrayType, mlir::Value header)
    {
        return fieldAddress(arrayType, header, ARRAY_SIZE_INDEX);
    }

    mlir::Value data(mlir_ts::ArrayType, mlir::Value array)
    {
        TypeHelper th(rewriter);
        return rewriter.create<LLVM::ExtractValueOp>(op->getLoc(), th.getPtrType(), array,
                                                     MLIRHelper::getStructIndex(rewriter, ARRAY_DATA_INDEX));
    }

    mlir::Value length(mlir_ts::ArrayType, mlir::Value array)
    {
        TypeHelper th(rewriter);
        return rewriter.create<LLVM::ExtractValueOp>(op->getLoc(), typeConverter->convertType(th.getIndexType()), array,
                                                     MLIRHelper::getStructIndex(rewriter, ARRAY_SIZE_INDEX));
    }

    mlir::Value identity(mlir_ts::ArrayType arrayType, mlir::Value array)
    {
        return data(arrayType, array);
    }

    mlir::Value make(mlir_ts::ArrayType arrayType, mlir::Value data, mlir::Value length)
    {
        auto loc = op->getLoc();
        auto llvmArrayType = typeConverter->convertType(arrayType);
        mlir::Value value = rewriter.create<LLVM::UndefOp>(loc, llvmArrayType);
        value = rewriter.create<LLVM::InsertValueOp>(loc, llvmArrayType, value, data,
                                                     MLIRHelper::getStructIndex(rewriter, ARRAY_DATA_INDEX));
        return rewriter.create<LLVM::InsertValueOp>(loc, llvmArrayType, value, length,
                                                    MLIRHelper::getStructIndex(rewriter, ARRAY_SIZE_INDEX));
    }

    mlir::Value makeStatic(mlir_ts::ArrayType arrayType, mlir::Value data, mlir::Value length)
    {
        return make(arrayType, data, length);
    }

    mlir::Value zero(mlir_ts::ArrayType arrayType)
    {
        return rewriter.create<LLVM::ZeroOp>(op->getLoc(), typeConverter->convertType(arrayType));
    }

    mlir::Value undef(mlir_ts::ArrayType arrayType)
    {
        return rewriter.create<LLVM::UndefOp>(op->getLoc(), typeConverter->convertType(arrayType));
    }

  private:
    mlir::Value fieldAddress(mlir_ts::ArrayType arrayType, mlir::Value header, int32_t index)
    {
        TypeHelper th(rewriter);
        return rewriter.create<LLVM::GEPOp>(op->getLoc(), th.getPtrType(), typeConverter->convertType(arrayType), header,
                                            ArrayRef<LLVM::GEPArg>{0, index});
    }
};

#endif // MLIR_TYPESCRIPT_LOWERTOLLVM_ARRAYLAYOUT_H_
```

If `LLVMCodeHelperBase.h` puts its class in a namespace or needs `using namespace` lines, mirror them exactly.

- [ ] **Step 3: Route the `LowerToLLVM.cpp` sites through it.** Add `#include "TypeScript/LowerToLLVM/ArrayLayout.h"` next to the other `LowerToLLVM/` includes. In each pattern, construct `ArrayLayout layout(op, rewriter, getTypeConverter(), tsLlvmContext->compileOptions);` and replace, keeping every other line and the order of the remaining instructions:
  - `LengthOfOpLowering`: `rewriter.replaceOp(op, layout.length(arrayType, transformed.getOp()))` where `arrayType = cast<mlir_ts::ArrayType>(op.getOp().getType())`. If `LengthOf` also accepts strings or const arrays (`TypeScript_ArrayLike`), keep the existing code for those types and use the layout only for `ArrayType`.
  - `SetLengthOf`, `ArrayPush`, `ArrayPop`, `ArrayUnshift`, `ArrayShift`, `ArraySplice`: replace the two GEPs `currentPtrPtr = GEP(llvmArrayType, transformed.getOp(), {0, ARRAY_DATA_INDEX})` and `countAsIndexTypePtr = GEP(..., {0, ARRAY_SIZE_INDEX})` with

    ```cpp
    auto header = layout.headerForUpdate(arrayType, transformed.getOp());
    auto currentPtrPtr = layout.dataAddress(arrayType, header);
    // ... (the load of currentPtr stays where it is)
    auto countAsIndexTypePtr = layout.lengthAddress(arrayType, header);
    ```
  - `CreateArray`, `NewEmptyArray`, `NewArray`: replace the `UndefOp` + two `InsertValueOp` with `layout.make(arrayType, allocated, <the count value they insert today>)`. `NewEmptyArray` inserts `size0` with a raw index 1 - same call.
  - `ArrayViewOpLowering`: `arrayPtr = layout.data(arrayType, transformed.getOp())`, result `layout.make(arrayType, arrayOffset, transformed.getCount())`.

- [ ] **Step 4: Build and compare the IR.**

```bash
cmake --build __build/tslang/windows-msbuild-2026-release --config Release --parallel 8 --target tslang 2>&1 | grep -E "error C|error:" | head
bash "$S/ir-snapshot.sh" /i/TypeScriptCompiler/__build/tslang/windows-msbuild-2026-release/bin/tslang.exe "$S/t1"
diff -rq "$S/base" "$S/t1" | head
```

Expected: no build error; `diff -rq` prints nothing. Any difference is a refactoring mistake: open the two files with `diff` and fix the site (usually an instruction order change).

- [ ] **Step 5: Commit.**

```bash
git add tslang/include/TypeScript/LowerToLLVM/ArrayLayout.h tslang/lib/TypeScript/LowerToLLVM.cpp
git commit -m "Lowering: one helper knows the array layout (array ops in LowerToLLVM.cpp)

ArrayLayout reads and writes an array's data and length, and makes one.
No change to the generated IR: --emit=llvm identical for every suite test
under gc, rc and own.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

### Task 2: the header sites, then PR 1

**Files:**
- Modify: `tslang/include/TypeScript/LowerToLLVM/CastLogicHelper.h` (array to `Ref`/opaque ~223-240; `castToArrayType` ~1095-1150; truthiness ~475; `extractArrayPtr` ~1293)
- Modify: `tslang/include/TypeScript/LowerToLLVM/LLVMCodeHelper.h` (`getReadOnlyRTArray` ~436; `GetAddressOfArrayElement` ~849)
- Modify: `tslang/include/TypeScript/LowerToLLVM/UnaryBinLogicalOrHelper.h` (array `===` ~271)
- Modify: `tslang/include/TypeScript/LowerToLLVM/OwnershipRoutineLogic.h` (retain ~1162; `buildArrayBody` ~1280)

**Interfaces:**
- Consumes: `ArrayLayout` from Task 1.
- Produces: `CastLogicHelper::extractArrayPtr` now returns `layout.identity(...)` (it is only used for identity: truthiness and `===`); a new `CastLogicHelper::extractArrayData` returns `layout.data(...)` for the `Ref`/opaque casts.

- [ ] **Step 1: Include and route.** Include `ArrayLayout.h` in each of the four headers (check for an include cycle: `ArrayLayout.h` must not include `LLVMCodeHelper.h` or `CastLogicHelper.h`). Then:
  - `CastLogicHelper.h` ~223-240: both `ExtractValueOp ... ARRAY_DATA_INDEX` become `ArrayLayout(op, rewriter, tch.typeConverter, compileOptions).data(arrayType, in)`. Use whatever `TypeConverter` pointer `CastLogicHelper` already holds (it constructs `LLVMCodeHelper` somewhere - copy that).
  - `castToArrayType`: `isUndef` returns `layout.zero(arrayType)` / `layout.undef(arrayType)` in its two branches; the final `UndefOp`+`InsertValueOp`x2 becomes `layout.make(arrayType, arrayPtr, sizeValue)`.
  - `extractArrayPtr` (~1293): keep the `DialectCastOp`, then `return layout.identity(arrayType, inAsLLVMType);`.
  - `LLVMCodeHelper::getReadOnlyRTArray`: `return ArrayLayout(op, rewriter, typeConverter, compileOptions).makeStatic(originalArrayType, itemValArrayPtr, sizeValue);` (keep `itemValArrayPtr` and `sizeValue` computed as now).
  - `LLVMCodeHelper::GetAddressOfArrayElement`: for `ArrayType`, `dataPtr = ArrayLayout(...).data(arrayType, arrayOrStringOrTuple)`.
  - `UnaryBinLogicalOrHelper.h`: nothing to change if it only calls `castLogic.extractArrayPtr` (it does) - verify.
  - `OwnershipRoutineLogic.h` retain (~1162): `emitIncRef(LoadOp(ptr, layout.dataAddress(arrayType, layout.headerForRead(arrayType, slotPtr))))`; `buildArrayBody`: `dataSlot = layout.dataAddress(arrayType, layout.headerForRead(arrayType, slotPtr))`, `sizeSlot = layout.lengthAddress(arrayType, <same header>)`. Compute the header once, before `emitIfLastReference`.

- [ ] **Step 2: Check none are left.**

```bash
cd /i/TypeScriptCompiler/tslang && grep -rn "ARRAY_DATA_INDEX\|ARRAY_SIZE_INDEX" lib include | grep -v "ArrayLayout.h\|Defines.h"
```

Expected: no output. Also check `grep -n "getStructIndex(rewriter, [01])" include/TypeScript/LowerToLLVM/LLVMCodeHelper.h lib/TypeScript/LowerToLLVM.cpp` and confirm none of the remaining hits builds or reads an array value.

- [ ] **Step 3: Build, compare the IR, run the suite.**

```bash
cmake --build __build/tslang/windows-msbuild-2026-release --config Release --parallel 8 --target tslang 2>&1 | grep -E "error C|error:" | head
bash "$S/ir-snapshot.sh" /i/TypeScriptCompiler/__build/tslang/windows-msbuild-2026-release/bin/tslang.exe "$S/t2"
diff -rq "$S/base" "$S/t2" | head
cmake --build __build/tslang/windows-msbuild-2026-release --config Release --parallel 8
cd __build/tslang/windows-msbuild-2026-release && ctest -C Release -j 12 --timeout 300 2>&1 | grep -E "tests passed|^\s+[0-9]+ - " | head
```

Expected: `diff` prints nothing; ctest shows only the known `test-compile-gc-defaultlib-collector` failure.

- [ ] **Step 4: Commit, push, open PR 1.**

```bash
git add tslang/include/TypeScript/LowerToLLVM/CastLogicHelper.h tslang/include/TypeScript/LowerToLLVM/LLVMCodeHelper.h tslang/include/TypeScript/LowerToLLVM/UnaryBinLogicalOrHelper.h tslang/include/TypeScript/LowerToLLVM/OwnershipRoutineLogic.h
git commit -m "Lowering: casts, constants and ownership routines read arrays through ArrayLayout

The last places outside the helper that knew the { data, length } layout.
No change to the generated IR.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
gh auth switch --user ASDAlexander77
git push -u origin array-reference
gh pr create --base main --head array-reference --title "Lowering: one helper knows the array layout (#453 step 1)" --body-file <pr-body file>
```

The PR body states: refactor only, IR byte-identical for every suite test under gc/rc/own, step 1 of the spec, ends with `🤖 Generated with [Claude Code](https://claude.com/claude-code)`. After PR 1 merges, PR 2's work continues on a new branch `array-reference-switch` from `origin/main` (`git checkout --no-track -b array-reference-switch origin/main`).

---

## PR 2: the switch

### Task 3: a rest element is a copy (#477)

**Files:**
- Modify: `tslang/lib/TypeScript/MLIRGenVariables.cpp` (~lines 330-395, the two `isDotDotDot` branches)
- Create: `tslang/test/tester/tests/00array_rest_copy.ts`
- Modify: `tslang/test/tester/CMakeLists.txt` (register; rc and none corpora)

**Interfaces:**
- Consumes: nothing new (old layout still in place - this task is layout-independent).
- Produces: MLIRGen no longer emits `ts.ArrayView` for a rest element.

- [ ] **Step 1: Write the failing test** `00array_rest_copy.ts` (CRLF line endings, like the other tests):

```ts
// A destructuring rest element is a new array, not a view into the source (#477).
class Box {
    constructor(public v: number) {}
}

function main() {
    const src: number[] = [1, 2, 3];
    const [first, ...rest] = src;
    assert(first == 1, "first");
    rest[0] = 20;
    assert(src[1] == 2, "a write to the rest leaks into the source");
    rest.push(4);
    assert(rest.length == 3 && rest[2] == 4, "push onto the rest");
    assert(src.length == 3 && src[2] == 3, "the source is unchanged");

    const [a, b, ...none] = [1];
    assert(none.length == 0, "the rest of a shorter array is empty");

    const boxes: Box[] = [new Box(1), new Box(2), new Box(3)];
    const [head, ...tail] = boxes;
    tail.push(new Box(4));
    assert(head.v == 1 && tail.length == 3 && tail[0].v == 2 && tail[2].v == 4, "class elements");

    const words: string[] = ["a", "b", "c"];
    const [w0, ...ws] = words;
    ws.push("d");
    assert(w0 == "a" && ws.length == 3 && ws[2] == "d" && words.length == 3, "string elements");

    print("done.");
}
```

Run: `__build/tslang/windows-msbuild-2026-release/bin/tslang.exe --emit=jit --no-default-lib -mm=gc tslang/test/tester/tests/00array_rest_copy.ts`. Expected: assertion failure "a write to the rest leaks into the source". Under `-mm=rc`: exit 127.

Also check `ts.ArrayView` has no other creator: `grep -rn "create<mlir_ts::ArrayViewOp>" tslang/lib`. Expected: only the two in `MLIRGenVariables.cpp`.

- [ ] **Step 2: Build the copy in MLIRGen.** In both `isDotDotDot` branches replace the `LengthOfOp`/`ArithmeticBinaryOp`/`ArrayViewOp` lines with a call to a new `MLIRGenImpl` helper, declared in `MLIRGenImpl.h` next to the other `mlirGen*` helpers and defined in `MLIRGenVariables.cpp`:

```cpp
// `[a, ...rest] = src`: rest is a new array holding src[index..] (#477) - built as
//   const .rest: T[] = []; for (let .i = index; .i < .src.length; .i++) .rest.push(.src[.i]);
// so the ownership passes see a fresh array and element pushes, as for any other array
ValueOrLogicalResult MLIRGenImpl::mlirGenArrayRestCopy(mlir::Location location, mlir_ts::ArrayType arrayType,
                                                       mlir::Value source, int64_t index, const GenContext &genContext)
{
    SymbolTableScopeT varScope(symbolTable);

    auto srcVarDecl = std::make_shared<VariableDeclarationDOM>(".rest_src", arrayType, location);
    DECLARE(srcVarDecl, source);

    auto empty = builder.create<mlir_ts::NewEmptyArrayOp>(location, arrayType);
    auto restVarDecl = std::make_shared<VariableDeclarationDOM>(".rest", arrayType, location);
    DECLARE(restVarDecl, empty);

    NodeFactory nf(NodeFactoryFlags::None);
    auto _src = nf.createIdentifier(S(".rest_src"));
    auto _rest = nf.createIdentifier(S(".rest"));
    auto _i = nf.createIdentifier(S(".rest_i"));

    NodeArray<VariableDeclaration> declarations;
    declarations.push_back(nf.createVariableDeclaration(
        _i, undefined, undefined, nf.createNumericLiteral(S(std::to_string(index).c_str()))));
    auto initVars = nf.createVariableDeclarationList(declarations, NodeFlags::Let);
    auto cond = nf.createBinaryExpression(_i, nf.createToken(SyntaxKind::LessThanToken),
                                          nf.createPropertyAccessExpression(_src, nf.createIdentifier(S(LENGTH_FIELD_NAME))));
    auto incr = nf.createPrefixUnaryExpression(nf.createToken(SyntaxKind::PlusPlusToken), _i);
    NodeArray<Expression> pushArgs;
    pushArgs.push_back(nf.createElementAccessExpression(_src, _i));
    auto push = nf.createExpressionStatement(nf.createCallExpression(
        nf.createPropertyAccessExpression(_rest, nf.createIdentifier(S("push"))), undefined, pushArgs));

    if (mlir::failed(mlirGen(nf.createForStatement(initVars, cond, incr, push), genContext)))
    {
        return mlir::failure();
    }

    return resolveIdentifier(location, ".rest", genContext);
}
```

Copy the exact `DECLARE`, `NodeFactory`, `S(...)` and `resolveIdentifier` usage from the numeric-widening code in `MLIRGenCast.cpp` (~line 1140-1185): match how it returns `.dst_array` after the loop (it may read the variable differently from `resolveIdentifier`; use the same call it uses). In the `ConstArrayType` branch pass the already-cast `arrayValue`; in the `ArrayType` branch pass `init`.

- [ ] **Step 3: Run the test under every model.**

```bash
for mm in gc rc none own; do __build/tslang/windows-msbuild-2026-release/bin/tslang.exe --emit=jit --no-default-lib -mm=$mm tslang/test/tester/tests/00array_rest_copy.ts; echo "$mm exit=$?"; done
```

Expected: `done.` and exit 0 for gc, rc, none. Under own, an ownership error on the class-element or string section is acceptable (own's rules); the number section must compile - if own rejects the whole file, keep the test out of own and say so in the report.

- [ ] **Step 4: Register the test** with the scratchpad script (it adds `test-compile-00-<name>` and `test-jit-00-<name>` after the anchor's lines and the file to the rc/none corpus list after the corpus anchor, keeping CRLF):

```bash
cd /i/TypeScriptCompiler && python "$R" array-rest-copy 00array_rest_copy.ts 00array5_deconst.ts 00array5_deconst.ts
```

where `R="C:/Users/duzha/AppData/Local/Temp/claude/I--TypeScriptCompiler/f5d6099f-c9ad-473f-a7eb-6f10e2a42c61/scratchpad/reinsert.py"`. Anchors used by this plan (each occurs once in its list): `00array5_deconst.ts` (rest copy), `00array2.ts` (reference), `00array3.ts` (zero slots), `00array6.ts` (static nested), `00array7.ts` (main argv), `00class_new.ts` (growth).

- [ ] **Step 5: Full suite, commit.**

```bash
cmake --build __build/tslang/windows-msbuild-2026-release --config Release --parallel 8
cd __build/tslang/windows-msbuild-2026-release && ctest -C Release -j 12 --timeout 300 2>&1 | grep -E "tests passed|^\s+[0-9]+ - " | head
cd /i/TypeScriptCompiler
git add tslang/lib/TypeScript/MLIRGenVariables.cpp tslang/lib/TypeScript/MLIRGenImpl.h tslang/test/tester/tests/00array_rest_copy.ts tslang/test/tester/CMakeLists.txt
git commit -m "A destructuring rest element is a new array (#477)

Closes #477

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

### Task 4: the switch

**Files:**
- Modify: `tslang/include/TypeScript/Defines.h` (add `#define ARRAY_CAPACITY_INDEX 2` after `ARRAY_SIZE_INDEX`)
- Modify: `tslang/include/TypeScript/LowerToLLVM/ArrayLayout.h` (the header implementation)
- Modify: `tslang/lib/TypeScript/LowerToLLVM.cpp` (type conversion ~7229: `ArrayType` -> `ptr`)
- Modify: `tslang/include/TypeScript/LowerToLLVM/OwnershipRoutineLogic.h` (retain and `buildArrayBody`)
- Modify: `tslang/include/TypeScript/LowerToLLVM/CastLogicHelper.h` (`castToArrayType` `byValue = false` path copies)
- Modify: `tslang/tslang/jit.cpp` (`addEntryThunk`: drop the `{ptr, iN}` struct branch)
- Create: `tslang/test/tester/tests/00array_reference.ts`, `00array_zero_slots.ts`, `00array_static_nested.ts`
- Modify: `tslang/test/tester/CMakeLists.txt`

**Interfaces:**
- Consumes: `ArrayLayout` API from Task 1 (unchanged signatures).
- Produces: `ArrayLayout::headerType()` returning `LLVM::LLVMStructType` `{ptr, index, index}`; `ArrayLayout::capacityAddress(arrayType, header)`; an array's LLVM type is `ptr`.

- [ ] **Step 1: Write the failing tests.** `00array_reference.ts` - the 14 shapes of spec §2 as asserts:

```ts
// An array is a reference: a change of length through any name is seen through every other (#453).
class Holder { items: number[] = [1]; }
type Rec = { items: number[] };

function pushParam(into: number[]) { into.push(2); }
function pushField(h: Holder) { h.items.push(2); }
function pushRec(r: Rec) { r.items.push(2); }
function grow(into: number[]) { into.push(2); into.push(3); into.push(4); into.push(5); }
function setLen(into: number[]) { into.length = 0; }
function popParam(into: number[]) { into.pop(); }
function setElem(into: number[]) { into[0] = 9; }

function main() {
    const a1: number[] = [1]; setElem(a1); assert(a1[0] == 9, "1 element write through a parameter");
    const a2: number[] = [1]; pushParam(a2); assert(a2.length == 2 && a2[1] == 2, "2 push through a parameter");
    const a3: number[] = [1, 2]; popParam(a3); assert(a3.length == 1, "3 pop through a parameter");
    const a4: number[] = [1, 2]; setLen(a4); assert(a4.length == 0, "4 length= through a parameter");
    const a5: number[] = [1]; const b5 = a5; b5.push(2); assert(a5.length == 2, "5 local alias");
    const h6 = new Holder(); pushField(h6); assert(h6.items.length == 2, "6 class field");
    const h7 = new Holder(); const it7 = h7.items; it7.push(2); assert(h7.items.length == 2, "7 field alias");
    const r8: Rec = { items: [1] }; pushRec(r8); assert(r8.items.length == 2, "8 record field");
    const a9: number[] = [1]; const f9 = () => { a9.push(2); }; f9(); assert(a9.length == 2, "9 closure");
    const aa10: number[][] = [[1]]; aa10[0].push(2); assert(aa10[0].length == 2, "10 nested");
    const aa11: number[][] = [[1]]; const in11 = aa11[0]; in11.push(2); assert(aa11[0].length == 2, "11 nested alias");
    const a12: number[] = [1]; const b12 = a12; assert(a12 === b12, "12 identity");
    const a13: number[] = [7]; grow(a13); assert(a13.length == 5 && a13[0] == 7 && a13[4] == 5, "13 growth");
    const a14: number[] = [1]; const h14 = new Holder(); h14.items = a14; a14.push(2); assert(h14.items.length == 2, "14 stored into a field");

    const a15: number[] = [1]; const x15: any = a15; (x15 as number[]).push(2); assert(a15.length == 2, "15 through any");
    const c16: number[] = []; const d16: number[] = [];
    assert(!(c16 === d16), "16 two empty arrays are different arrays");
    let truthy = false; if (c16) { truthy = true; } assert(truthy, "16 an empty array is truthy");
    let n16: number[] | null = null; let falsy = true; if (n16) { falsy = false; } assert(falsy, "16 null is falsy");

    print("done.");
}
```

`00array_zero_slots.ts` (R1):

```ts
// Zeroed memory is an empty array: nested arrays made by length, and a field with no initializer.
class Bag {
    items: number[];
    add(v: number) { this.items.push(v); }
}

function main() {
    const aa = new Array<number[]>(2);
    assert(aa[0].length == 0, "a zeroed element reads as empty");
    aa[0].push(1);
    aa[0].push(2);
    assert(aa[0].length == 2 && aa[0][1] == 2 && aa[1].length == 0, "push into a zeroed element");

    const b = new Bag();
    b.add(5);
    assert(b.items.length == 1 && b.items[0] == 5, "push into a field with no initializer");

    print("done.");
}
```

`00array_static_nested.ts` (Review Focus 3):

```ts
// Constant arrays inside global data keep their elements, and are never freed.
const aa: number[][] = [[1], [2, 3]];
const t: [number, number[]] = [1, [4, 5]];

function main() {
    assert(aa.length == 2 && aa[1].length == 2 && aa[1][1] == 3, "nested constant array");
    assert(t[1].length == 2 && t[1][0] == 4, "array in a constant tuple");
    for (let i = 0; i < 3; i++) {
        const inner = aa[1];
        assert(inner[0] == 2, "read through a local, repeatedly");
    }
    print("done.");
}
```

Run each with `--emit=jit --no-default-lib` under gc and rc. Expected before the switch: `00array_reference.ts` fails at "2 push through a parameter" (gc) / exit 127 (rc); `00array_zero_slots.ts` and `00array_static_nested.ts` pass (they pin today's behaviour that the switch must keep). If `new Array<number[]>(2)` or a field with no initializer does not compile today, drop that line and note it in the report.

- [ ] **Step 2: Switch the representation in `ArrayLayout.h`.** Replace the PR 1 bodies:

```cpp
    // { data, length, capacity }: what an array value points to
    LLVM::LLVMStructType headerType()
    {
        TypeHelper th(rewriter);
        auto llvmIndexType = typeConverter->convertType(th.getIndexType());
        return LLVM::LLVMStructType::getLiteral(rewriter.getContext(), {th.getPtrType(), llvmIndexType, llvmIndexType},
                                                false);
    }

    // a slot holding null - a zeroed element or field, `undefined` as an array - gets an empty
    // header first, so a change made through the slot is kept (spec ruling R1)
    mlir::Value headerForUpdate(mlir_ts::ArrayType arrayType, mlir::Value slot)
    {
        TypeHelper th(rewriter);
        CodeLogicHelper clh(op, rewriter);
        auto loc = op->getLoc();
        auto ptrType = th.getPtrType();
        mlir::Value header = rewriter.create<LLVM::LoadOp>(loc, ptrType, slot);
        auto isNull = rewriter.create<LLVM::ICmpOp>(loc, LLVM::ICmpPredicate::eq, header,
                                                    rewriter.create<LLVM::ZeroOp>(loc, ptrType));
        return clh.conditionalExpressionLowering(
            loc, ptrType, isNull,
            [&](OpBuilder &, Location) -> mlir::Value {
                auto fresh = MemoryAlloc(headerType(), MemoryAllocSet::Zero);
                rewriter.create<LLVM::StoreOp>(loc, fresh, slot);
                return fresh;
            },
            [&](OpBuilder &, Location) -> mlir::Value { return header; });
    }

    mlir::Value headerForRead(mlir_ts::ArrayType, mlir::Value slot)
    {
        TypeHelper th(rewriter);
        return rewriter.create<LLVM::LoadOp>(op->getLoc(), th.getPtrType(), slot);
    }

    mlir::Value capacityAddress(mlir_ts::ArrayType arrayType, mlir::Value header)
    {
        return fieldAddress(arrayType, header, ARRAY_CAPACITY_INDEX);
    }

    // a null header reads as an empty array (R1)
    mlir::Value data(mlir_ts::ArrayType arrayType, mlir::Value array)
    {
        TypeHelper th(rewriter);
        return loadOrZero(array, th.getPtrType(), ARRAY_DATA_INDEX);
    }

    mlir::Value length(mlir_ts::ArrayType arrayType, mlir::Value array)
    {
        TypeHelper th(rewriter);
        return loadOrZero(array, typeConverter->convertType(th.getIndexType()), ARRAY_SIZE_INDEX);
    }

    mlir::Value identity(mlir_ts::ArrayType, mlir::Value array)
    {
        return array;
    }

    mlir::Value make(mlir_ts::ArrayType arrayType, mlir::Value data, mlir::Value length)
    {
        auto loc = op->getLoc();
        auto header = MemoryAlloc(headerType());
        rewriter.create<LLVM::StoreOp>(loc, data, fieldAddress(arrayType, header, ARRAY_DATA_INDEX));
        rewriter.create<LLVM::StoreOp>(loc, length, fieldAddress(arrayType, header, ARRAY_SIZE_INDEX));
        rewriter.create<LLVM::StoreOp>(loc, length, fieldAddress(arrayType, header, ARRAY_CAPACITY_INDEX));
        return header;
    }

    mlir::Value zero(mlir_ts::ArrayType)
    {
        TypeHelper th(rewriter);
        return rewriter.create<LLVM::ZeroOp>(op->getLoc(), th.getPtrType());
    }

    mlir::Value undef(mlir_ts::ArrayType arrayType)
    {
        return zero(arrayType);
    }

  private:
    mlir::Value fieldAddress(mlir_ts::ArrayType, mlir::Value header, int32_t index)
    {
        TypeHelper th(rewriter);
        return rewriter.create<LLVM::GEPOp>(op->getLoc(), th.getPtrType(), headerType(), header,
                                            ArrayRef<LLVM::GEPArg>{0, index});
    }

    mlir::Value loadOrZero(mlir::Value header, mlir::Type fieldType, int32_t index)
    {
        TypeHelper th(rewriter);
        CodeLogicHelper clh(op, rewriter);
        auto loc = op->getLoc();
        auto isSet = rewriter.create<LLVM::ICmpOp>(loc, LLVM::ICmpPredicate::ne, header,
                                                   rewriter.create<LLVM::ZeroOp>(loc, th.getPtrType()));
        return clh.conditionalExpressionLowering(
            loc, fieldType, isSet,
            [&](OpBuilder &, Location) -> mlir::Value {
                return rewriter.create<LLVM::LoadOp>(
                    loc, fieldType,
                    rewriter.create<LLVM::GEPOp>(loc, th.getPtrType(), headerType(), header, ArrayRef<LLVM::GEPArg>{0, index}));
            },
            [&](OpBuilder &, Location) -> mlir::Value { return rewriter.create<LLVM::ZeroOp>(loc, fieldType); });
    }
```

`CodeLogicHelper::conditionalExpressionLowering` is used the same way by `StringCopyOpLowering` (`LowerToLLVM.cpp` ~920); if `ArrayLayout.h` cannot include `CodeLogicHelper.h` without a cycle, build the two blocks by hand as `SetLengthOfOpLowering` does (~700-720). `MemoryAlloc(mlir::Type, MemoryAllocSet)` is in `LLVMCodeHelperBase`.

**Constant arrays in global data.** `makeStatic(arrayType, data, length)` from Task 1 cannot work with a header: `getReadOnlyRTArray` runs inside a global's initializer region, where nothing can be allocated, and a header global's own initializer cannot use a value computed in another region. Replace it with:

```cpp
    // a constant array in global data: a header global "ah_<data global>" holding
    // { data, length, capacity = length }, preceded under rc and own by an immortal block word (as
    // getOrCreateGlobalArray does for the data, LLVMCodeHelper.h ~555). Returns its address.
    mlir::Value makeStatic(mlir_ts::ArrayType arrayType, StringRef dataGlobalName, int64_t dataOffsetBytes,
                           int64_t length)
    {
        auto loc = op->getLoc();
        auto parentModule = op->getParentOfType<mlir::ModuleOp>();
        TypeHelper th(rewriter);
        auto llvmIndexType = typeConverter->convertType(th.getIndexType());
        auto withWord = compileOptions.tracksOwnership();
        auto headerName = ("ah_" + dataGlobalName).str();

        auto global = parentModule.lookupSymbol<LLVM::GlobalOp>(headerName);
        if (!global)
        {
            OpBuilder::InsertionGuard guard(rewriter);
            rewriter.setInsertionPointToStart(parentModule.getBody());
            mlir::Type globalType = headerType();
            if (withWord)
            {
                globalType = LLVM::LLVMStructType::getLiteral(rewriter.getContext(), {llvmIndexType, headerType()},
                                                              /*isPacked=*/true);
            }

            global = rewriter.create<LLVM::GlobalOp>(loc, globalType, /*isConstant=*/true, LLVM::Linkage::Internal,
                                                     headerName, mlir::Attribute{});
            global.setAlignment(getHeapBlockHeaderSize());

            auto &region = global.getInitializerRegion();
            rewriter.setInsertionPointToStart(rewriter.createBlock(&region));
            auto llvmLength = rewriter.create<LLVM::ConstantOp>(loc, llvmIndexType,
                                                                rewriter.getIntegerAttr(llvmIndexType, length));
            mlir::Value dataPtr = rewriter.create<LLVM::AddressOfOp>(loc, th.getPtrType(), dataGlobalName);
            if (dataOffsetBytes != 0)
            {
                dataPtr = rewriter.create<LLVM::GEPOp>(
                    loc, th.getPtrType(), th.getI8Type(), dataPtr,
                    ValueRange{rewriter.create<LLVM::ConstantOp>(loc, llvmIndexType,
                                                                 rewriter.getIntegerAttr(llvmIndexType, dataOffsetBytes))});
            }

            mlir::Value headerVal = rewriter.create<LLVM::UndefOp>(loc, headerType());
            headerVal = rewriter.create<LLVM::InsertValueOp>(loc, headerVal, dataPtr, MLIRHelper::getStructIndex(rewriter, ARRAY_DATA_INDEX));
            headerVal = rewriter.create<LLVM::InsertValueOp>(loc, headerVal, llvmLength, MLIRHelper::getStructIndex(rewriter, ARRAY_SIZE_INDEX));
            headerVal = rewriter.create<LLVM::InsertValueOp>(loc, headerVal, llvmLength, MLIRHelper::getStructIndex(rewriter, ARRAY_CAPACITY_INDEX));
            mlir::Value globalVal = headerVal;
            if (withWord)
            {
                globalVal = rewriter.create<LLVM::UndefOp>(loc, globalType);
                globalVal = rewriter.create<LLVM::InsertValueOp>(
                    loc, globalVal,
                    rewriter.create<LLVM::ConstantOp>(loc, llvmIndexType, rewriter.getIntegerAttr(llvmIndexType, HEAP_BLOCK_IMMORTAL)),
                    MLIRHelper::getStructIndex(rewriter, 0));
                globalVal = rewriter.create<LLVM::InsertValueOp>(loc, globalVal, headerVal, MLIRHelper::getStructIndex(rewriter, 1));
            }

            rewriter.create<LLVM::ReturnOp>(loc, ValueRange{globalVal});
        }

        mlir::Value address = rewriter.create<LLVM::AddressOfOp>(loc, global);
        return withWord ? getPayloadPtrFromBlockPtr(loc, address, llvmIndexType) : address;
    }
```

In `getReadOnlyRTArray` (`LLVMCodeHelper.h` ~436), call `makeStatic(originalArrayType, <the data global's name>, <the payload offset>, arrayValue.size())`. The data global's name is the `vecVarName` `getOrCreateGlobalArray(originalElementType, size, arrayAttr)` computes - split that overload so the name is returned too - and the payload offset is `getHeapBlockHeaderSize()` when `tracksOwnership()`, else 0 (what `getOrCreateGlobalArray` adds to the global's address). Match how `getOrCreateGlobalArray` sets the insertion point for a new global (`seekLast`) if globals must stay ordered.

- [ ] **Step 3: Type conversion.** `LowerToLLVM.cpp` ~7229:

```cpp
    // an array is a reference to its header { data, length, capacity } (ArrayLayout.h)
    converter.addConversion([&](mlir_ts::ArrayType type) {
        return LLVM::LLVMPointerType::get(m.getContext());
    });
```

- [ ] **Step 4: rc and own routines** (`OwnershipRoutineLogic.h`):
  - retain (~1162): `emitIncRef(layout.headerForRead(arrayType, slotPtr));` (the header is the counted block).
  - `buildArrayBody`: `header = layout.headerForRead(arrayType, slotPtr)`; `emitIfLastReference(header, ...)` whose body: release elements `[0, length)` reading `data`/`length` through `layout.dataAddress(arrayType, header)` / `layout.lengthAddress(arrayType, header)` (the header is non-null inside), then `emitFreeBlock(<loaded data>)`, then `emitFreeBlock(header)`. Read the current body first: if it already frees the data block, keep that and add the header's free after it. Update the two comments ("an array value is { data, length }...") to describe the header.

- [ ] **Step 5: The const-array cast copies on both paths** (`CastLogicHelper.h` `castToArrayType`, the `else` of `if (byValue)`): make it allocate and `MemoryCopyOp` exactly as the `byValue` branch does (delete the "copy ptr only" branch and its TODO), so `push` never reallocates static data.

- [ ] **Step 6: JIT thunk** (`tslang/tslang/jit.cpp` `addEntryThunk`): delete the `index == 1 && arrayType && ...` struct branch and the `arrayType` local; update the comment above the function (an array is now a pointer; `string[]` argv arrives in Task 5). Until Task 5, a `main(argc, argv: string[])` is broken - no suite test uses one (`grep -rln "argv: string\[\]" tslang/test` is empty).

- [ ] **Step 7: Build and run the new tests under every model.**

```bash
cmake --build __build/tslang/windows-msbuild-2026-release --config Release --parallel 8 --target tslang 2>&1 | grep -E "error C|error:" | head
for t in 00array_reference 00array_zero_slots 00array_static_nested 00array_rest_copy; do for mm in gc rc none; do
  __build/tslang/windows-msbuild-2026-release/bin/tslang.exe --emit=jit --no-default-lib -mm=$mm tslang/test/tester/tests/$t.ts > "$S/o.txt" 2>&1; echo "$t $mm exit=$? $(tail -1 "$S/o.txt")"; done; done
```

Expected: every line `exit=0 done.`

- [ ] **Step 8: Register the three tests** (Task 3 step 4's script; anchors `00array2.ts`, `00array3.ts`, `00array6.ts`):

```bash
python "$R" array-reference 00array_reference.ts 00array2.ts 00array2.ts
python "$R" array-zero-slots 00array_zero_slots.ts 00array3.ts 00array3.ts
python "$R" array-static-nested 00array_static_nested.ts 00array6.ts 00array6.ts
```

Then run the full suite:

```bash
cmake --build __build/tslang/windows-msbuild-2026-release --config Release --parallel 8
cd __build/tslang/windows-msbuild-2026-release && ctest -C Release -j 12 --timeout 300 2>&1 | grep -E "tests passed|^\s+[0-9]+ - " | head -40
```

Expected: only `test-compile-gc-defaultlib-collector` fails. Any other failure is a site the switch missed: reproduce it with `--emit=jit --no-default-lib` and the runner's flags, read its `--emit=llvm`, find the array access that still assumes a struct (an `extractvalue`/`insertvalue` on a `ptr`, or a two-word union payload), and route it through `ArrayLayout`. Record each in the report.

- [ ] **Step 9: Commit.**

```bash
git add -A tslang/include tslang/lib tslang/tslang/jit.cpp tslang/test/tester/tests/00array_reference.ts tslang/test/tester/tests/00array_zero_slots.ts tslang/test/tester/tests/00array_static_nested.ts tslang/test/tester/CMakeLists.txt
git commit -m "An array is a reference to a header that never moves (#453)

An array value is one pointer to { data, length, capacity }. Assigning,
passing and storing an array copy the pointer, so a push, pop, splice or
length= through any name is seen through every other. Under rc the header
is the counted block; a null header reads as an empty array and gets a
header before a change through its slot.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

### Task 5: `main`'s argv is `Ref<string>` only

Revised 2026-10-04 (owner's ruling; the `ts.ArrayFromCStrings` op is dropped). A `main` whose second parameter is `string[]` would read C's `char **` as an array header, so it becomes a compile error:

- `LowerToAffineLoops.cpp`: in every build but a DLL (JIT and AOT) an array argv of `main` is an error with a source location, with the message `'main' takes argv as Ref<string> (C's char **), not string[]; read an argument with Deref(argv[i])`.
- `jit.cpp` `addEntryThunk`: comment and `unsupported()` message name `Ref<string>` only; no behaviour change.
- Test `tslang/test/tester/lowering-errors/main_argv_string_array.ts`, registered in `tslang/test/tester/CMakeLists.txt` for `--emit=jit`, `--emit=exe` and `--emit=obj` (and as a no-error case for `--emit=dll`), passes only when the compile output contains `takes argv as Ref<string>`. `tests/00main_argc_argv.ts` (the `Ref<string>` form) keeps passing.

### Task 6: debug info

**Files:**
- Modify: `tslang/include/TypeScript/LowerToLLVM/LLVMDebugInfo.h` (~197: `ArrayType`)
- Modify: `tslang/include/TypeScript/MLIRLogic/MLIRTypeHelper.h` (~2430: `getFields` for `ArrayType`)

- [ ] **Step 1:** `getFields` for `ArrayType` adds a third field `capacity` (`mlir::IndexType`) after `length`, keeping the order (the TODO there says debug info depends on it). Check its other callers first: `grep -rn "getFields(" tslang/lib tslang/include | head -30` - if any caller indexes array fields by position for something other than debug info, keep `getFields` as is and build the three fields inside `LLVMDebugInfo.h` instead.
- [ ] **Step 2:** In `getDITypeScriptType`, return `getDIPointerType(getDITypeWithFields(location, arrayType, to_print(arrayType), false, file, line, scope), file, line)` for an `ArrayType` - the same shape as a class (~line 455).
- [ ] **Step 3:** Run the debug-info tests: `cd __build/tslang/windows-msbuild-2026-release && ctest -C Release -R "debug-info|--di|-di-" --timeout 300`. Expected: all pass. Compile `00array_reference.ts` with `--di` (`tslang.exe --emit=exe --di ...` as the `debug-info-*.cmake` tests do) and check it links and runs.
- [ ] **Step 4:** Commit ("Debug info: an array is a pointer to { data, length, capacity }").

### Task 7: own, the gates, the default library, PR 2

**Files:**
- Create: `tslang/test/tester/own/own_array_param_push.ts`; Modify: `tslang/test/tester/CMakeLists.txt` (own test list ~2591)

- [ ] **Step 1: own test** `own_array_param_push.ts`:

```ts
// A borrowed array parameter is the owner's array: a push through it is the owner's (#453).
function keep(into: number[]) {
    into.push(2);
    into.push(3);
}

function main() {
    let a: number[] = [7];
    keep(a);
    assert(a.length == 3 && a[0] == 7 && a[2] == 3, "push through a borrowed parameter");
    print("done.");
}
```

Add `own_array_param_push` to the `foreach(own_test ...)` list. Run `ctest -C Release -R "own-array-param-push"`: the JIT, compile and no-counting tests must pass.

- [ ] **Step 2: own corpus.** With the corpus scripts from the own work (`C:/Users/duzha/AppData/Local/Temp/claude/I--TypeScriptCompiler/f5d6099f-c9ad-473f-a7eb-6f10e2a42c61/scratchpad/corpus465/run.sh <out.tsv>`; it reads `OPTS` from the environment and compiles with the current build), first take a baseline with the PR base's compiler (`main.tsv`/`mainopt.tsv` there are from main 9a5cc05b and may be reused if main has not changed lowering since), then run plain and `OPTS=--opt` on the branch and compare with `join -t "$(printf '\t')" main.tsv new.tsv | awk -F'\t' '($2=="ok") != ($3=="ok")'`: no file may go from ok to failing. Files that go from failing to ok are expected (shapes own rejected for a struct reason).
- [ ] **Step 3: own memory measurement.** Run the own measurement (`measure.ps1`, after one plain test-runner run) and compare with the numbers in the own spec's last results section: flat (within noise) is the gate.
- [ ] **Step 4: Linux.** Apply the branch's diff on a WSL clone of `origin/main` and run the full suite there (`git diff origin/main > fix.patch`; a script like `scratchpad/wsl446.sh` that applies it, builds with ninja, runs `ctest -j 8 --timeout 300`, then restores the clone). Expected: 0 failures. Windows' heap hides double frees; this run is required.
- [ ] **Step 5: Default library.** Rebuild `I:\TypeScriptCompilerDefaultLib` with the new compiler for every model and both build types, into the layout `defaultlib/{dll,lib}/{debug,release}/{gc,rc,none}` (memory: set the `*_LIB_PATH` env vars per mode; they point at `I:\tslang`). Then run the default-library tests (`ctest -C Release -R defaultlib`) - `test-compile-gc-defaultlib-collector` must now pass - and the default library's own suite under every model.
- [ ] **Step 6: Commit the own test, push, open PR 2.** The PR body lists: the representation, R1-R3, the tests, the gates with their numbers, "closes #453, closes #477", and that the default library must be rebuilt after merge (CI's default-lib artifact too). Ends with `🤖 Generated with [Claude Code](https://claude.com/claude-code)`.

---

## PR 3: growth

### Task 8: capacity growth

**Files:**
- Modify: `tslang/lib/TypeScript/LowerToLLVM.cpp` (`ArrayPush`, `ArrayUnshift`, `ArraySplice`, `ArrayPop`, `ArrayShift`, `SetLengthOf` lowerings)
- Modify: `tslang/include/TypeScript/LowerToLLVM/ArrayLayout.h` (`ensureCapacity`)
- Create: `tslang/test/tester/tests/00array_growth.ts`; Modify: `tslang/test/tester/CMakeLists.txt`

**Interfaces:**
- Produces: `mlir::Value ArrayLayout::ensureCapacity(mlir_ts::ArrayType arrayType, mlir::Value header, mlir::Value needed)` - returns the (possibly new) data pointer, already stored in the header, with `capacity >= needed`; new slots past the old capacity are zero.

- [ ] **Step 1: Write the failing test** `00array_growth.ts`:

```ts
// Arrays grow by doubling; what pop and shift vacate reads as zero when length grows again.
class Box { constructor(public v: number) {} }

function fill(into: number[], n: number) { for (let i = 0; i < n; i++) into.push(i); }

function main() {
    const a: number[] = [];
    const alias = a;
    fill(a, 1000);
    assert(alias.length == 1000 && alias[0] == 0 && alias[999] == 999, "1000 pushes through an alias");

    a.pop();
    a.length = 1000;
    assert(a[999] == 0, "a popped slot reads zero when length grows again");

    const s: number[] = [1, 2, 3];
    s.shift();
    s.unshift(9);
    s.push(4);
    assert(s.length == 4 && s[0] == 9 && s[1] == 2 && s[3] == 4, "shift, unshift and push around a growth");

    const boxes: Box[] = [];
    for (let i = 0; i < 100; i++) boxes.push(new Box(i));
    for (let i = 0; i < 50; i++) boxes.pop();
    assert(boxes.length == 50 && boxes[49].v == 49, "class elements across growth and pops");

    print("done.");
}
```

Run under gc and rc. Expected before the change: passes (behaviour is already right with exact reallocation), except possibly "a popped slot reads zero" under gc - the test pins the behaviour PR 3 must keep. Then write a timing check (not committed) in the scratchpad: 1,000,000 pushes onto an array, measured with `--emit=exe` before and after.

- [ ] **Step 2: `ensureCapacity`** in `ArrayLayout.h`: load `capacity`; if `needed > capacity`: `newCapacity = max(4, 2 * capacity, needed)`, `data = MemoryRealloc(data, newCapacity * sizeof(element))`, zero `[capacity, newCapacity)` with `LLVM::MemsetOp`, store `data` and `newCapacity`; return the data pointer. Element size comes from `mlir_ts::SizeOfOp` as in the existing lowerings.
- [ ] **Step 3: Use it.** `push`/`unshift`/`splice` call `ensureCapacity(header, newLength)` instead of `MemoryRealloc(currentPtr, newLength * size)`. `pop`/`shift`/`splice` (removal) no longer reallocate; they zero each slot they vacate (`MemsetOp` of one element, or of the tail for `shift`'s move). `SetLengthOf`: growing calls `ensureCapacity` and zeroes `[length, n)` in every model, gc included (delete the `needsGCRuntime()` exception and rewrite its comment: after a `pop` the slots past `length` are no longer fresh memory); shrinking zeroes `[n, length)` and keeps the block.
- [ ] **Step 4: Register and run.** `python "$R" array-growth 00array_growth.ts 00class_new.ts 00class_new.ts`. Run `00array_growth.ts` under gc, rc, none (and the own corpus check from Task 7 step 2), the timing check (expect a large drop for 1,000,000 pushes), then the full Windows suite and the WSL suite.
- [ ] **Step 5: Commit, push, open PR 3** ("Arrays grow by doubling (#453 step 3)").

---

## Self-review notes

- Spec coverage: §3 (representation, empty array, const cast both paths, rest copy, layout, ABI) - Tasks 3, 4; §4 (ownership per model) - Task 4 step 4, Task 7; §5 (helper, op signatures kept, creation/read sites, DI, main, x86) - Tasks 1, 2, 4, 5, 6 (x86: no specific code; the WSL/Windows suites include the x86 tests that exist); §6 growth - Task 8; §7 gates - Tasks 1, 2, 4, 7, 8; §8 out of scope respected.
- R2 and R3 change how §5's adapter and §3's view-as-copy are built, not what they do; the spec is amended alongside this plan.
