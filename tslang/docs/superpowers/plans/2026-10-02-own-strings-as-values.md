# `-mm=own` Strings as Values Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development
> (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use
> checkbox (`- [ ]`) syntax for tracking.

**Goal:** Under `-mm=own`, a second reference to a string that own cannot prove a move or a borrow
becomes a copy (`ts.StringCopy`), so string-heavy programs compile instead of failing.

**Architecture:** The ownership inference pass decides each function in up to two runs.
- A dry run of today's analysis reports nothing and changes nothing.
- If the dry run fails, each string retain it could not erase is rewritten: the one use rc made
  that retain for takes a fresh `ts.StringCopy` of the value, and the retain goes.
- The real run then verifies the rewritten function as it does today.

A function that compiles today takes exactly today's path.

**Tech Stack:** C++17, MLIR/LLVM (tablegen ops, conversion patterns), the tslang test runner,
CMake/CTest.

**Spec:** `tslang/docs/superpowers/specs/2026-09-24-own-memory-model-design.md`, section 22.
Sections 15 (signature facts), 20.2 (constant data) and 21 (named messages, nullable views) are
background.

## Global Constraints

- Branch `own-strings-as-values` (spec §22 is commit 0cd3dc23, on main 9d92dc84).
- Copies are silent: no diagnostic, no flag (§22.1).
- Copies only in a function the dry run rejects, and only for string retains the dry run did not
  erase. A function that compiles today must compile to the same IR (§22.1, §22.6).
- "A string" (§22.1, as corrected in Task 1):
  - `!ts.string`;
  - a string literal type;
  - a `!ts.union` of string, string literals and `!ts.null` only, which needs no tag (one pointer);
  - an `!ts.optional` of one of those (`string | undefined`, an optional parameter), which is
    `{ptr, i1}`.
- No copy (§22.4) when:
  - the retain's use is not exactly one op, or there is no retain (rc consumed the value);
  - the value borrows a parameter the function's callers know it keeps (`__own_params` on the
    function) or returns a borrow of (`__own_result_borrows`);
  - the retain is a closure capture's (`captureRetains`) or a borrowing field's
    (`fieldBorrowRetains`).
- `ts.StringCopy`: null stays null and allocates nothing; otherwise `strlen + 1` bytes through
  `ch.MemoryAlloc` (the allocator `ts.StringConcat` uses) and a `memcpy`. It lowers under every
  model.
- Never branch on the memory model in MLIRGen (§4.3). This plan does not touch MLIRGen.
- Sources are CRLF. Do not edit them with Git Bash `sed -i`, which rewrites them to LF. Use the
  Edit tool, or Python with `newline=''`.
- Build: `cmake --build __build/tslang/windows-msbuild-2026-release --config Release --target tslang --parallel 8`.
  For tests: `ctest -C Release` (without `-C`, every test is "Not Run").
  - Editing `TypeScriptOps.td` regenerates the dialect. If MSVC fails with C1060, rebuild with
    `--parallel 3`.
- Commits are GPG-signed. Never pass `--no-gpg-sign`. If signing times out, stage everything and
  hand the commit to the user with the message in a file.
- End commit messages with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.

## Review Focus

1. **A copy that is not made, while its retain is erased.** The use then shares the original, and
   the original's owner frees it, which is a use after free. The tests must free the original and
   allocate over it before reading the copy (the `churn` pattern), or a missing copy passes on
   stale memory. Task 3 owns this.
2. **The dry run emitting or changing anything.** A diagnostic printed twice, an attribute left
   set, or a retain erased by the dry run. Task 2 checks the error count and that the IR is
   unchanged.
3. **An optional string whose flag says "no value".** Its pointer may be garbage, and copying it
   would crash. The lowering must test the flag before the pointer, and Task 3's
   `own_string_copy_nullable` passes an `undefined` through a copy.
4. **A parameter the callers were told is kept.** Copying it leaks, because the caller gave its
   reference up. Task 3's negative `own_err_string_kept_param_known` pins the guard.
5. **The same retain chosen twice, or a use that takes the value twice** (`f(s, s)`, where one
   call keeps both arguments). `takerOf` returns null for two takers, which means no copy. Task 3
   tests that a call keeping two arguments still errors rather than copying once.

---

### Task 1: the `ts.StringCopy` op, its lowering, and the facts

**Files:**
- Modify: `tslang/include/TypeScript/TypeScriptOps.td` (after `TypeScript_StringConcatOp`, about
  line 2374)
- Modify: `tslang/lib/TypeScript/LowerToLLVM.cpp`:
  - a new pattern after `StringConcatOpLowering` (about line 880);
  - its registration in the pattern list (about line 7966, beside `StringConcatOpLowering`).
- Modify: `tslang/lib/TypeScript/LowerToAffineLoops.cpp`: the legal-op list, about line 2660,
  beside `mlir_ts::StringConcatOp`.
- Modify: `tslang/lib/TypeScript/OwnershipFacts.h`:
  - `isFresh`, about line 297;
  - a new `isCopyableString` after `holdsNoBlock`.
- Modify: `tslang/docs/superpowers/specs/2026-09-24-own-memory-model-design.md`, §22.1 and §22.2:
  correct the representation of the optional.

**Interfaces:**
- Produces:
  - `mlir_ts::StringCopyOp`, built as `builder.create<mlir_ts::StringCopyOp>(loc, value.getType(), value)`,
    accessor `getIn()`, result type equal to the operand type;
  - `inline bool isCopyableString(mlir::Type type)` in `OwnershipFacts.h`;
  - `isFresh(value)` true for a `StringCopyOp` result.

- [x] **Step 1: Add the op.** In `TypeScriptOps.td`, after `TypeScript_StringConcatOp`:

```tablegen
// A copy of a string: a new block holding the same text, null when `in` is null (or, for an
// optional, when it holds no value). Made by -mm=own's inference pass only, where a second
// reference to a string cannot be proven a move or a borrow (spec 22); it lowers under every
// model.
def TypeScript_StringCopyOp : TypeScript_Op<"StringCopy", [AllTypesMatch<["in", "result"]>]> {
  let arguments = (ins AnyType:$in);
  let results = (outs Res<AnyType, "", [MemAlloc]>:$result);
}
```

- [x] **Step 2: Mark it legal for the affine lowering.** In `LowerToAffineLoops.cpp`, in the
  `target.addLegalOp<...>` list, change `mlir_ts::StringConcatOp,mlir_ts::StringCompareOp` to
  `mlir_ts::StringConcatOp, mlir_ts::StringCopyOp, mlir_ts::StringCompareOp`.

- [x] **Step 3: Write the LLVM lowering.** In `LowerToLLVM.cpp`, after `StringConcatOpLowering`:

```cpp
// A copy of a string: `strlen + 1` bytes, from the allocator `ts.StringConcat` uses. A null string
// stays null and allocates nothing. An optional `{ptr, i1}` is copied only when it holds a value:
// a pointer whose flag says "none" may be anything.
class StringCopyOpLowering : public TsLlvmPattern<mlir_ts::StringCopyOp>
{
  public:
    using TsLlvmPattern<mlir_ts::StringCopyOp>::TsLlvmPattern;

    LogicalResult matchAndRewrite(mlir_ts::StringCopyOp op, Adaptor transformed,
                                  ConversionPatternRewriter &rewriter) const final
    {
        TypeHelper th(rewriter);
        LLVMCodeHelper ch(op, rewriter, getTypeConverter(), tsLlvmContext->compileOptions);
        TypeConverterHelper tch(getTypeConverter());
        CodeLogicHelper clh(op, rewriter);

        auto loc = op->getLoc();
        auto i8PtrTy = th.getPtrType();
        auto llvmIndexType = tch.convertType(th.getIndexType());
        auto llvmBoolType = th.getLLVMBoolType();
        auto strlenFuncOp = ch.getOrInsertFunction("strlen", th.getFunctionType(llvmIndexType, {i8PtrTy}));

        auto copyOf = [&](mlir::Value source, mlir::Value copyIt) {
            return clh.conditionalExpressionLowering(
                loc, i8PtrTy, copyIt,
                [&](OpBuilder &, Location) -> mlir::Value {
                    mlir::Value bytes = rewriter.create<LLVM::CallOp>(loc, strlenFuncOp, ValueRange{source}).getResult();
                    bytes = rewriter.create<LLVM::AddOp>(
                        loc, llvmIndexType,
                        ValueRange{bytes, rewriter.create<LLVM::ConstantOp>(loc, llvmIndexType, rewriter.getIntegerAttr(llvmIndexType, 1))});
                    auto copy = ch.MemoryAlloc(bytes);
                    rewriter.create<LLVM::MemcpyOp>(loc, copy, source, bytes, /*isVolatile=*/false);
                    return copy;
                },
                [&](OpBuilder &, Location) -> mlir::Value { return source; });
        };

        mlir::Value in = transformed.getIn();
        auto isSet = [&](mlir::Value pointer) -> mlir::Value {
            return rewriter.create<LLVM::ICmpOp>(loc, LLVM::ICmpPredicate::ne, pointer, rewriter.create<LLVM::ZeroOp>(loc, i8PtrTy));
        };

        if (mlir::isa<LLVM::LLVMPointerType>(in.getType()))
        {
            rewriter.replaceOp(op, ValueRange{copyOf(in, isSet(in))});
            return success();
        }

        // an optional: `{ptr, i1}`
        auto pointer = rewriter.create<LLVM::ExtractValueOp>(loc, in, ArrayRef<int64_t>{0});
        auto hasValue = rewriter.create<LLVM::ExtractValueOp>(loc, in, ArrayRef<int64_t>{1});
        auto copyIt = rewriter.create<LLVM::AndOp>(loc, llvmBoolType, hasValue, isSet(pointer));
        auto copied = copyOf(pointer, copyIt);
        rewriter.replaceOp(op, ValueRange{rewriter.create<LLVM::InsertValueOp>(loc, in, copied, ArrayRef<int64_t>{0})});
        return success();
    }
};
```

  Check the helper names against `StringResizeOpLowering`/`StringConcatOpLowering` in the same
  file (`th.getLLVMBoolType`, `LLVM::ZeroOp`, `ch.MemoryAlloc`). If `getLLVMBoolType` is not on
  `TypeHelper`, use `rewriter.getI1Type()`.

  Register it: in the pattern list (the line with `StringConcatOpLowering,`), add
  `StringCopyOpLowering,` after it.

- [x] **Step 4: Teach the facts.** In `OwnershipFacts.h`, `isFresh`: add `mlir_ts::StringCopyOp`
  to the `mlir::isa<mlir_ts::NewOp, mlir_ts::CreateArrayOp, mlir_ts::NewArrayOp, mlir_ts::StringConcatOp, ...>`
  list. After `holdsNoBlock`, add:

```cpp
// A value -mm=own copies rather than give a second owner (spec 22.1): a string, a string literal,
// a union of strings and null that needs no tag (one pointer), or an optional of one of those
// (`{ptr, i1}`).
inline bool isCopyableString(mlir::Type type)
{
    auto isText = [](mlir::Type member) {
        if (auto literalType = mlir::dyn_cast<mlir_ts::LiteralType>(member))
        {
            return mlir::isa<mlir_ts::StringType>(literalType.getElementType());
        }

        return mlir::isa<mlir_ts::StringType>(member);
    };

    if (auto optionalType = mlir::dyn_cast<mlir_ts::OptionalType>(type))
    {
        return !mlir::isa<mlir_ts::OptionalType>(optionalType.getElementType()) &&
               isCopyableString(optionalType.getElementType());
    }

    if (auto unionType = mlir::dyn_cast<mlir_ts::UnionType>(type))
    {
        return llvm::any_of(unionType.getTypes(), isText) &&
               llvm::all_of(unionType.getTypes(), [&](mlir::Type member) {
                   return isText(member) || mlir::isa<mlir_ts::NullType>(member);
               });
    }

    return isText(type);
}
```

  A union of strings and null must need no tag. Confirm on a `string | null` local that
  `MLIRTypeHelper::isUnionTypeNeedsTag` answers false (the LLVM type is `ptr`; see the `@n` and
  `@w` functions in §21's checks). If some string union does need a tag, add that condition here.

- [x] **Step 5: Correct the spec.** In §22.1, change "Those are one pointer, the same as `string`
  (the nullable views of §21.3)" to:

```
A union with `null` is one pointer, the same as `string`; an optional (`string | undefined`, an
optional parameter) is `{ptr, i1}`, the pointer meaningful only when the flag is set.
```

  In §22.2, change "A nullable string is copied through its pointer." to "An optional is copied
  only when it holds a value."

- [x] **Step 6: Build, and confirm nothing else changed.** Build (Global Constraints). Run
  `ctest -C Release -R "own" -j 8 --timeout 300` and expect 100% passed. Nothing creates the op
  yet, so this is a no-op change for every program.

- [x] **Step 7: Commit.**

```bash
git add tslang/include/TypeScript/TypeScriptOps.td tslang/lib/TypeScript/LowerToLLVM.cpp tslang/lib/TypeScript/LowerToAffineLoops.cpp tslang/lib/TypeScript/OwnershipFacts.h tslang/docs/superpowers/specs/2026-09-24-own-memory-model-design.md
git commit -m "-mm=own strings: ts.StringCopy and the facts that know it (spec 22.2)"
```

---

### Task 2: the dry run (no behaviour change)

**Files:**
- Modify: `tslang/lib/TypeScript/OwnershipInferencePass.cpp`

**Interfaces:**
- Consumes: nothing from Task 1.
- Produces, members of `OwnershipInferencePass`:
  - `bool dryRun = false;`
  - `unsigned failures = 0;`
  - `bool reporting();`
  - `void setAttrTracked(mlir::Operation *op, llvm::StringRef name);`
  - `bool analyze(llvm::SetVector<mlir::Operation *> &toErase);`, which returns true when the
    function passed. In a dry run it erases nothing and strips no facts; otherwise it does both,
    as `runOnFunction` does today.

- [x] **Step 1: Record the baseline.** Build, then, in the scratchpad, save the first error of
  every corpus file (Appendix A, `run.sh`) as `base.tsv`, plain and with
  `OPTS="--opt --opt_level=3"` as `baseopt.tsv`. Also save the full stderr of three negative tests:

```bash
T=__build/tslang/windows-msbuild-2026-release/bin/tslang.exe
for f in own_err_two_owners own_err_closure_escape_twice own_err_delete; do $T --emit=obj --no-default-lib -mm=own tslang/test/tester/own/$f.ts -o NUL > $SCRATCH/$f.before.txt 2>&1; done
```

- [x] **Step 2: Add the reporting switch.** Add the members beside `unsigned quiet = 0;`:

```cpp
    // The dry run (spec 22.3): the whole analysis, but nothing reported and nothing changed. A
    // report it would make is counted in `failures`.
    bool dryRun = false;
    unsigned failures = 0;
    // attributes the dry run set on ops that did not have them, taken off again after it
    llvm::SmallVector<std::pair<mlir::Operation *, mlir::StringAttr>> dryRunAttrs;

    // Whether a report is to be emitted: not under `quietly`'s trial, and not in the dry run, which
    // only counts it.
    bool reporting()
    {
        if (quiet)
        {
            return false;
        }

        if (dryRun)
        {
            ++failures;
            return false;
        }

        return true;
    }

    // `op->setAttr(name, unit)`, undone after the dry run when the op did not have it
    void setAttrTracked(mlir::Operation *op, llvm::StringRef name)
    {
        auto attrName = mlir::StringAttr::get(&getContext(), name);
        if (dryRun && !op->hasAttr(attrName))
        {
            dryRunAttrs.push_back({op, attrName});
        }

        op->setAttr(attrName, mlir::UnitAttr::get(&getContext()));
    }
```

- [x] **Step 3: Route every report through it.**
  - In every report function (`grep -n "if (quiet)" OwnershipInferencePass.cpp`, about 12 of
    them), replace
    ```cpp
            if (quiet)
            {
                return;
            }
    ```
    with
    ```cpp
            if (!reporting())
            {
                return;
            }
    ```
  - Then find every `emitError` that is not inside such a function. Today: the `RetainCell` and
    `DeleteOp` errors in the walk at the top of `runOnFunction`, and any others
    `grep -n "emitError" ... | grep -v report` shows. Wrap each with its `signalPassFailure()`:
    ```cpp
                if (reporting())
                {
                    op->emitError("...");
                    signalPassFailure();
                }
    ```
  - Every `signalPassFailure()` must now sit behind a `reporting()` that returned true. Confirm with
    `grep -n "signalPassFailure" OwnershipInferencePass.cpp` and read each one.

- [x] **Step 4: Track the attributes.** Replace each `x->setAttr(NAME, mlir::UnitAttr::get(&getContext()))`
  in the pass with `setAttrTracked(x, NAME)` (`grep -n "setAttr(" OwnershipInferencePass.cpp`: the
  `OWN_CAPTURE_BORROW_ATTR_NAME`, `OWN_BORROWS_CAPTURES_ATTR_NAME` and
  `OWN_CELL_BORROWS_ATTR_NAME` sites).

- [x] **Step 5: Split `runOnFunction` into `analyze`.**
  - Move the body of `runOnFunction` into `bool analyze(llvm::SetVector<mlir::Operation *> &toErase)`,
    with `toErase` declared by the caller, not inside.
  - Its last part becomes:
    ```cpp
            if (dryRun)
            {
                return failures == 0;
            }

            for (auto *op : toErase)
            {
                op->erase();
            }

            // the facts were for this pass only
            returnedParams.clear();
            f.walk([](mlir::Operation *op) { /* the existing removeAttr loop, unchanged */ });
            return true;
    ```
  - Every member the body uses must start each run clean. `analyze` already clears `drops`,
    `derivedCache`, `closures`, `closureOf`, `stateBoxes`, `fieldBorrowRetains`,
    `fieldBorrowReleases`, `captureStores` and `captureRetains`, and resets `returnsBorrowOf`. Also
    clear `returnedParams` at its start. Read the private members once and clear any other
    per-function state the same way.
  - `runOnFunction` becomes:
    ```cpp
        void runOnFunction()
        {
            // the dry run (spec 22.3): does today's analysis pass as it is?
            {
                llvm::SetVector<mlir::Operation *> toErase;
                dryRun = true;
                failures = 0;
                auto passed = analyze(toErase);
                dryRun = false;
                for (auto &[op, name] : dryRunAttrs)
                {
                    op->removeAttr(name);
                }

                dryRunAttrs.clear();
                (void)passed; // Task 3 makes the copies here when it did not
            }

            llvm::SetVector<mlir::Operation *> toErase;
            analyze(toErase);
        }
    ```

- [x] **Step 6: Confirm nothing changed.**
  - Build. Run `ctest -C Release -R "own|mlirgen-matches" -j 8 --timeout 300`: expect 100%.
  - Rerun the corpus (`run.sh`) plain and `--opt`, and `diff` against `base.tsv`/`baseopt.tsv`,
    ignoring the generated `FH<digits>` hashes (`sed -E 's/FH[0-9]+/FH/g'` on both). Expect no
    differences.
  - Rerun Step 1's three negatives and `diff` their stderr against the `.before.txt` files. They
    must be identical: each error once, with the same notes.

- [x] **Step 7: Commit.**

```bash
git add tslang/lib/TypeScript/OwnershipInferencePass.cpp
git commit -m "-mm=own strings: the inference pass's dry run (spec 22.3), no change yet"
```

---

### Task 3: the copies, their guards, and the tests

**Files:**
- Modify: `tslang/lib/TypeScript/OwnershipInferencePass.cpp`
- Create (positive): in `tslang/test/tester/own/`, `own_string_copy_field.ts`,
  `own_string_copy_after_move.ts`, `own_string_copy_nullable.ts`, `own_string_copy_exported.ts`,
  `own_string_copy_written.ts`
- Create (negative): in `tslang/test/tester/own/`, `own_err_string_kept_param_known.ts`,
  `own_err_string_call_keeps_twice.ts`, `own_err_string_consumed_twice.ts`
- Modify: `tslang/test/tester/CMakeLists.txt`:
  - the `foreach(own_test ...)` list;
  - `own_error_cases`;
  - a foreach running the positives under rc, none and gc;
  - fix up the existing string negatives (Step 7).

**Interfaces:**
- Consumes: `mlir_ts::StringCopyOp`, `isCopyableString` (Task 1); `dryRun`, `analyze`,
  `captureRetains`, `fieldBorrowRetains` (Task 2).
- Produces: `bool makeStringCopies(const llvm::SetVector<mlir::Operation *> &erased)`, which returns
  whether it rewrote anything.

- [x] **Step 1: Write the positive tests.** Each frees the original, or overwrites its owner, and
  allocates over it with `churn()` before reading the copy. A missing copy then reads freed memory.

`own_string_copy_field.ts`:

```ts
// A field's string given a second owner is copied (spec 22): stored into another field, returned,
// pushed, and read out of a caught object. Each original is then overwritten and its memory reused
// before the copy is read.
class Rec {
    str = "";
    str2 = "";
}

class E {
    constructor(public m: string) {}
}

function churn() {
    let keep: string[] = [];
    for (let i = 0; i < 1000; i++) keep.push("k" + i);
    return keep.length;
}

function textOf(r: Rec) {
    return r.str2;
}

function main() {
    const r = new Rec();
    r.str2 = "two" + 2;
    r.str = r.str2;
    r.str2 = "other" + 3;
    assert(churn() == 1000);
    assert(r.str == "two2" && r.str2 == "other3");

    const got = textOf(r);
    r.str2 = "again" + 4;
    assert(churn() == 1000);
    assert(got == "other3");

    const list: string[] = [];
    list.push(r.str2);
    r.str2 = "last" + 5;
    assert(churn() == 1000);
    assert(list[0] == "again4");

    let caught = "";
    try {
        throw new E("thrown" + 6);
    } catch (e) {
        caught = (<E>e).m;
    }

    assert(churn() == 1000);
    assert(caught == "thrown6");

    print("done.");
}
```

`own_string_copy_after_move.ts`:

```ts
// A string moved into a field and used after: the field takes a copy, the local keeps its own.
class H {
    text = "";
}

function churn() {
    let keep: string[] = [];
    for (let i = 0; i < 1000; i++) keep.push("k" + i);
    return keep.length;
}

function main() {
    const h = new H();
    let kept = "kept" + 1;
    h.text = kept;
    print(kept);
    h.text = "other" + 2;
    assert(churn() == 1000);
    assert(kept == "kept1" && h.text == "other2");

    print("done.");
}
```

`own_string_copy_nullable.ts`:

```ts
// Nullable strings copied: a `string | null` holding a string and holding null, and an optional
// (`string | undefined`, an optional parameter) holding a string and holding nothing - whose
// pointer must not be read.
class Box {
    s: string | null = null;
    o: string | undefined = undefined;
}

function churn() {
    let keep: string[] = [];
    for (let i = 0; i < 1000; i++) keep.push("k" + i);
    return keep.length;
}

function keepOpt(b: Box, v?: string) {
    b.o = v;
}

function main() {
    const a = new Box();
    const b = new Box();
    a.s = "nn" + 1;
    b.s = a.s;
    a.s = null;
    assert(churn() == 1000);
    assert(b.s == "nn1");

    b.s = a.s;
    assert(b.s == null);

    a.o = "opt" + 2;
    b.o = a.o;
    a.o = undefined;
    assert(churn() == 1000);
    assert(b.o == "opt2");

    b.o = a.o;
    assert(b.o === undefined);

    keepOpt(a, b.o);
    keepOpt(a);
    assert(a.o === undefined);

    print("done.");
}
```

`own_string_copy_exported.ts`:

```ts
// An exported class's constructor keeps its string parameter. Its callers cannot be told so (it can
// be called from another module), so they keep their own, and the field takes a copy.
export class Animal {
    name: string;
    constructor(name: string) {
        this.name = name;
    }
}

function churn() {
    let keep: string[] = [];
    for (let i = 0; i < 1000; i++) keep.push("k" + i);
    return keep.length;
}

function main() {
    let n = "cat" + 1;
    const a = new Animal(n);
    n = "dog" + 2;
    assert(churn() == 1000);
    assert(a.name == "cat1" && n == "dog2");

    print("done.");
}
```

`own_string_copy_written.ts`, run under own only (gc shares the block, so `s` would change too):

```ts
// A write into a copy's bytes changes the copy only (spec 22.1): strings are values under own.
class R {
    s = "";
    t = "";
}

function main() {
    const r = new R();
    r.s = "abc" + "";
    r.t = r.s;
    r.t[0] = <char>65;
    assert(r.t == "Abc", "the copy was written");
    assert(r.s == "abc", "the original was not");

    print("done.");
}
```

- [x] **Step 2: Write the negative tests.**

`own_err_string_kept_param_known.ts`:

```ts
// -mm=own rejects, and does not copy: `keep` is private and called only directly, so the
// signature pass tells its callers it keeps its parameter (`__own_params`) and they give theirs up.
// A copy in `keep` would leak the one the caller gave. `main` passes a field's string, which it
// does not own.
class H {
    s = "";
}

const holder = new H();

function keep(v: string) {
    holder.s = v;
}

function main() {
    const h = new H();
    h.s = "x" + 1;
    keep(h.s);
    print(holder.s);
}
```

  Expected message: `argument .* is given to 'keep', which keeps it, but -mm=own cannot move it here`.
  If the actual first error differs, keep a regex that names the kept-parameter rule, and say so in
  the commit message.

`own_err_string_call_keeps_twice.ts`:

```ts
// -mm=own rejects, and does not copy: one string given twice to a call that keeps both arguments.
// The retains have no single use each, so no copy is made.
class P {
    a = "";
    b = "";
}

const p = new P();

function both(x: string, y: string) {
    p.a = x;
    p.b = y;
}

function main() {
    const h = new P();
    h.a = "s" + 1;
    both(h.a, h.a);
    print(p.a, p.b);
}
```

  Expected message: whatever the current build reports at `both(h.a, h.a)`. Record it as the regex.
  It must not be "compiles".

`own_err_string_consumed_twice.ts`:

```ts
// -mm=own rejects, and does not copy: rc hands the result of an indirect call to two consuming
// declarations with no retain at all, so there is no retain to turn into a copy (spec 22.4).
function main() {
    const make = (n: number) => "s" + n;
    const a = make(1);
    let b = a;
    let c = a;
    print(b, c);
}
```

  Expected message: the current build's first error for it, recorded as the regex. If this program
  compiles before Task 3 (rc retained after all), find another value rc consumes twice without a
  retain (spec §11, phase 0's notes), or record in §22.7 that no such source shape was found, and
  drop the test.

  The negative for a value that is not a string is the existing class negatives
  (`own_err_two_owners`, `own_err_push_let`, ...). Step 5's full `own` run keeps them passing.

- [x] **Step 3: Register the tests and watch them fail.** In `CMakeLists.txt`:
  - add `own_string_copy_field own_string_copy_after_move own_string_copy_nullable own_string_copy_exported own_string_copy_written`
    to the `foreach(own_test ...)` list;
  - add the three negatives to `own_error_cases` with the regexes from Step 2;
  - after the `own_global_script` foreach, add:

```cmake
# Strings as values (spec 22): no model but own makes a copy, and each program means the same
# under every model.
foreach(strings_mm rc none gc)
    foreach(strings_test own_string_copy_field own_string_copy_after_move own_string_copy_nullable own_string_copy_exported)
        tslang_add_test(NAME test-jit-${strings_mm}-${strings_test} COMMAND test-runner -jit -mm=${strings_mm} "${PROJECT_SOURCE_DIR}/test/tester/own/${strings_test}.ts")
        tslang_add_test(NAME test-compile-${strings_mm}-${strings_test} COMMAND test-runner -mm=${strings_mm} "${PROJECT_SOURCE_DIR}/test/tester/own/${strings_test}.ts")
    endforeach()
endforeach()
```

  Build. Run `ctest -C Release -R "own_string_copy|own_err_string" -j 8`. Expected: every
  `-own-` positive fails (own rejects them today), the rc/none/gc ones pass, and the negatives
  pass. If an rc/none/gc run fails, that is a bug outside this phase. Stop and report it.

- [x] **Step 4: Write the rewrite.** In `OwnershipInferencePass.cpp`, add:

```cpp
    // Strings as values (spec 22.3): each string retain the dry run did not erase is rewritten, where
    // its one use can be found, into a copy for that use. True when anything was rewritten.
    bool makeStringCopies(const llvm::SetVector<mlir::Operation *> &erased)
    {
        llvm::SmallVector<mlir::Operation *> retains;
        getFunction()->walk([&](mlir::Operation *op) {
            if (mlir::isa<mlir_ts::RetainOp, mlir_ts::RetainSlotOp>(op) && !erased.contains(op) &&
                !captureRetains.contains(op) && !fieldBorrowRetains.contains(op))
            {
                retains.push_back(op);
            }
        });

        auto copied = false;
        for (auto *retain : retains)
        {
            copied = copyForRetain(retain) || copied;
        }

        return copied;
    }

    // The value a parameter of this function is, when its callers were told this function keeps it
    // or returns a borrow of it (spec 15): copying it would leak what the caller gave up.
    bool borrowsKnownParam(mlir::Value value)
    {
        auto param = borrowedParam(value);
        return param >= 0 && (llvm::is_contained(ownedParams(getFunction()), param) || param == returnsBorrowOf);
    }

    bool copyForRetain(mlir::Operation *retain)
    {
        // `let x = v`: the local starts from a copy
        if (auto retainSlotOp = mlir::dyn_cast<mlir_ts::RetainSlotOp>(retain))
        {
            auto varOp = retainSlotOp.getSlot().getDefiningOp<mlir_ts::VariableOp>();
            auto init = varOp ? varOp.getInitializer() : mlir::Value();
            if (!init || !isCopyableString(init.getType()) || borrowsKnownParam(init) || isCellVariable(varOp.getResult()))
            {
                return false;
            }

            mlir::OpBuilder builder(varOp);
            auto copy = builder.create<mlir_ts::StringCopyOp>(varOp.getLoc(), init.getType(), init);
            varOp->setOperand(0, copy);
            retain->erase();
            return true;
        }

        auto retainOp = mlir::cast<mlir_ts::RetainOp>(retain);
        auto value = retainOp.getReference();
        if (!isCopyableString(value.getType()) || borrowsKnownParam(value))
        {
            return false;
        }

        // the one use rc retained it for
        mlir::OpOperand *taking = nullptr;
        auto several = false;
        forEachUse(value, [&](mlir::Operation *user, mlir::Value used) {
            if (mlir::isa<mlir_ts::RetainOp, mlir_ts::ReleaseOp>(user) || isBorrow(user, used))
            {
                return;
            }

            for (auto &operand : user->getOpOperands())
            {
                if (operand.get() == used)
                {
                    several = several || taking;
                    taking = &operand;
                }
            }
        });

        if (!taking || several || !isCopyableString(taking->get().getType()))
        {
            return false;
        }

        mlir::OpBuilder builder(taking->getOwner());
        auto copy = builder.create<mlir_ts::StringCopyOp>(retain->getLoc(), taking->get().getType(), taking->get());
        taking->set(copy);
        retain->erase();
        return true;
    }
```

  Check whether `VariableOp`'s initializer is operand 0 (`ODS`: `Optional<AnyType>:$initializer`
  is its only operand). If there is a generated `getInitializerMutable()`, use
  `varOp.getInitializerMutable().assign(copy)` instead.

  In `runOnFunction`, replace `(void)passed; // Task 3 ...` with:

```cpp
                if (!passed)
                {
                    makeStringCopies(toErase);
                }
```

  This needs `captureRetains` and `fieldBorrowRetains` as the dry run left them. They are members
  and are cleared only at the start of `analyze`, so they still hold the dry run's sets here.

- [x] **Step 5: Run the tests.** Build. Run
  `ctest -C Release -R "own_string_copy|own_err_string|own" -j 8 --timeout 300`. Expected: all pass.
  - If a positive still fails, read its error:
    - one named in §22.4 means that shape is outside the rule; adjust the test only if the spec
      says so;
    - otherwise look for why `copyForRetain` returned false (add a temporary `llvm::errs()` line,
      never committed).
  - Also run each positive under own with the runner's `-noopt` (`--di --opt_level=0`):
    `test-runner -noopt -mm=own <file>` and `test-runner -noopt -jit -mm=own <file>`.

- [x] **Step 6: Teeth.** Each change is temporary, made after this task's code is committed or
  backed up.
  - (a) `makeStringCopies` returns false at its top: every `own_string_copy_*` positive must fail.
  - (b) `borrowsKnownParam` returns false: `own_err_string_kept_param_known` must compile, or
    fail at run time under AOT. If it still errors the same way, the guard is a second guard there;
    record that in §22.7.
  - Revert both. Clear `tslang/test/tester/own/__jit` and the build's `own/__jit` between JIT runs
    (the JIT cache does not key on such a switch).

- [x] **Step 7: The string negatives that now compile.**
  - Run `ctest -C Release -R "test-own-err" -j 8`. Phase 6's named-message negatives used strings
    in some cases: `own_err_mixed_return`, `own_err_unbox_string`,
    `own_err_param_returned_exported`, `own_err_param_kept_exported`, `own_err_merged_result`, and
    perhaps older ones.
  - For each that now fails because the program compiles, rewrite it with a class in place of the
    string, so that it still exercises its message. For example, in `own_err_mixed_return.ts`:
    ```ts
    class C { constructor(public n: string) {} }
    function pick(c: C, other: boolean) {
        if (!other) return c;
        return new C("new");
    }
    function main() {
        const c = new C("a");
        print(pick(c, true).n);
    }
    ```
    Update its regex's function name (`'pick' returns a borrow ...`).
  - `own_err_unbox_string` cannot become a class: `<C>anyValue` does not take the string path.
    Delete it from `own_error_cases` and from the tree. Step 1's `own_string_copy_*` tests and
    Task 4's mixed-return test cover the copy now.
  - Rerun until `test-own-err` is 100%.

- [x] **Step 8: Commit.**

```bash
git add tslang/lib/TypeScript/OwnershipInferencePass.cpp tslang/test/tester/CMakeLists.txt tslang/test/tester/own/
git commit -m "-mm=own strings: a second reference to a string own cannot prove becomes a copy (spec 22.3, 22.4)"
```

---

### Task 4: mixed returns, and the corpus

**Files:**
- Create: `tslang/test/tester/own/own_string_copy_mixed_return.ts`
- Modify: `tslang/test/tester/CMakeLists.txt` (the `own_test` list and the rc/none/gc foreach)
- Modify: `tslang/docs/superpowers/specs/2026-09-24-own-memory-model-design.md`: a new §22.7
  "Results"
- Modify: `tslang/docs/superpowers/plans/2026-10-02-own-strings-as-values.md`: tick the boxes

**Interfaces:**
- Consumes: Task 3's rewrite (no new code is expected here; if the mixed-return shape needs one, it
  goes into `copyForRetain` with its own test).

- [x] **Step 1: Write the test.** `own_string_copy_mixed_return.ts`:

```ts
// A function that returns a borrow of its argument on one path and a new string on another: the
// borrowed return is copied, so the result is always the caller's (spec 22.5). Also the
// synthesized `<string>` of an `any`, and a union cast to a string.
function greet(name: string) {
    if (name === "Honda") return name;
    return "Sorry, " + name;
}

function churn() {
    let keep: string[] = [];
    for (let i = 0; i < 1000; i++) keep.push("k" + i);
    return keep.length;
}

function main() {
    let who = "Hon" + "da";
    const g = greet(who);
    who = "x" + 1;
    assert(churn() == 1000);
    assert(g == "Honda");
    assert(greet("Bob") == "Sorry, Bob");

    const a = <any>("boxed" + 1);
    const back = <string>a;
    assert(back == "boxed1");

    let u: number | string = "un" + 1;
    const s = <string>u;
    u = 3;
    assert(churn() == 1000);
    assert(s == "un1");

    print("done.");
}
```

  Add it to the `own_test` list and to the rc/none/gc foreach. Build, and run its six runner
  variants plus `-noopt -mm=own`. Expected: all pass. If own still rejects it, read which retain
  had no single use (Task 3, Step 5), fix `copyForRetain` with a test-first change, and record the
  shape in §22.7.

- [x] **Step 2: Corpus.** Rerun `run.sh` plain and `--opt` (Appendix A). Then:
  - **None lost:** `join -t $'\t' base.tsv new.tsv | awk -F'\t' '$2=="ok" && $3!="ok"'` prints
    nothing, plain and `--opt`.
  - **Gate every gained file:** list the files ok under `--opt` now and not in `baseopt.tsv`, then
    run `gate1.sh` on each, with and without `RFLAGS=-noopt` (Appendix A). Every line must read
    `aot=ok jit=ok`. A failure is a bug: stop and fix it, with a test, before going on.
  - Record the first-error histogram before and after
    (`cut -f2 | sed -E "s/'[^']*'/'X'/g" | sort | uniq -c | sort -rn`).

- [x] **Step 3: Unchanged where it compiles.** Before building this branch's final binary, copy
  main's `tslang.exe` aside (build main once, or reuse a saved one: `tslang-main.exe`).
  - For every file ok in `base.tsv`, compare
    `--emit=llvm -mm=own --no-default-lib <file> -o <out>.ll` from both binaries, with
    `FH<digits>` hashes normalised.
  - Expect no difference. Any difference means a function that compiled on main took the copy
    path, or the dry run changed something. Find it and fix it.

- [x] **Step 4: Measure.** Write `$SCRATCH/copy_loop.ts`:

```ts
class R {
    s = "";
    t = "";
}

function main() {
    const r = new R();
    for (let i = 0; i < 200000; i++) {
        r.s = "v" + i;
        r.t = r.s;
    }

    print(r.t);
}
```

  Run `pwsh tslang/test/tester/tools/measure.ps1 -Source $SCRATCH/copy_loop.ts` (once after a plain
  test-runner run, so `compile.bat` exists). Then run it again at 20000 iterations. Expected:
  - own's peak is the same at both counts, within 0.5 MB, so every copy is freed;
  - rc is flat as well;
  - none grows.

  Record both rows in §22.7.

- [x] **Step 5: Spec results.** Append §22.7 "Results" to the spec, in the style of §20/§21:
  - what was built;
  - the corpus table, before and after, plain and `--opt`;
  - the gated files;
  - the teeth (Task 3, Step 6);
  - the measure rows;
  - the IR-unchanged check;
  - every shape found outside the rule.

  Tick this plan's boxes.

- [ ] **Step 6: Linux.** Build the branch with GCC in WSL and run
  `ctest -R "own-err|own-no-counting|own-verify"` there (recipe: the WSL clone at
  `~/ts/TypeScriptCompiler`, build dir `__build/tslang/ninja/release`, fetch this branch from the
  Windows repo).

- [ ] **Step 7: Commit, and open the PR.**

```bash
git add tslang/test/tester/own/own_string_copy_mixed_return.ts tslang/test/tester/CMakeLists.txt tslang/docs/superpowers/
git commit -m "-mm=own strings: mixed returns, corpus and results (spec 22.7)"
gh auth switch --user ASDAlexander77
git push -u origin own-strings-as-values
```

  The PR body carries the §22.7 corpus table, and ends with
  `🤖 Generated with [Claude Code](https://claude.com/claude-code)`.

---

## Appendix A: the corpus tooling (scratchpad only, never committed)

`one.sh`, the first error of one corpus file:

```sh
#!/bin/sh
f="$1"; T=/i/TypeScriptCompiler/__build/tslang/windows-msbuild-2026-release/bin/tslang.exe
out=$(cd /i/TypeScriptCompiler/tslang/test/tester/tests && timeout 120 "$T" --emit=llvm -mm=own --no-default-lib $OPTS "$(basename "$f")" -o NUL 2>&1)
rc=$?
first=$(printf '%s\n' "$out" | grep -m1 -E "error" | sed -E 's/^.*error: //' | tr -d '\r')
if [ $rc -eq 0 ]; then echo "$(basename "$f")	ok"; else echo "$(basename "$f")	${first:-rc=$rc}"; fi
```

`run.sh`, every file into `$1`:

```sh
#!/bin/sh
cd "$(dirname "$0")"
ls /i/TypeScriptCompiler/tslang/test/tester/tests/*.ts | xargs -P 12 -n 1 sh ./one.sh | sort > "${1:-results.tsv}"
echo "$(grep -c '	ok$' "${1:-results.tsv}") / $(wc -l < "${1:-results.tsv}") ok"
```

`gate1.sh`, test-runner under own for one file, AOT and JIT (`RFLAGS=-noopt` for the `--di` build):

```sh
#!/bin/sh
f="$1"; R=/i/TypeScriptCompiler/__build/tslang/windows-msbuild-2026-release/test/tester/Release/test-runner.exe
cd /i/TypeScriptCompiler/__build/tslang/windows-msbuild-2026-release/test/tester || exit 1
src=/i/TypeScriptCompiler/tslang/test/tester/tests/$f
a=$(timeout 300 "$R" $RFLAGS -mm=own "$src" >/dev/null 2>&1 && echo ok || echo FAIL)
j=$(timeout 300 "$R" $RFLAGS -jit -mm=own "$src" >/dev/null 2>&1 && echo ok || echo FAIL)
echo "$f aot=$a jit=$j"
```
