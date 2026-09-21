# 32-bit Phase 4a: Union Byte Storage Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Every member of a tagged union reads back exactly the value it was stored with, on x64 and on i686, under every memory model.

**Architecture:** A tagged union lowers to a packed `{ptr tag, S}`, where `S` is the LLVM type of its largest member (`findMaxSizeType`). A member goes in and out through memory: `castLLVMTypes` stores it, `CopyStructOp` copies `min(src, dst)` bytes, and the result is loaded back. When that memory is loaded as `S`, any bytes in `S`'s *padding* are not part of the loaded value. So a smaller member's field that sits in that padding is lost the next time the union is copied as a value.

At x64 this happens only for some layouts; the old test `00bigint_struct_layout` documents one and cannot assert its field `a`. At i686 it happens for common ones, because a pointer is 4 bytes and an `f64` after it is 8-aligned. `02union_type` then prints `Error 9.11037e-305 downloading` instead of `Error 1 downloading`.

The fix is to make `S` an `[N x i8]` array, where N is the largest member's alloc size. A byte array has no padding, so every byte is carried. The in and out paths already go through memory, so they need no new logic. Each consumer of the union's value field must be checked to still reinterpret correctly.

**Tech Stack:** C++17, the MLIR LLVM dialect, the TypeScript dialect lowering (`LowerToLLVM.cpp`, `CastLogicHelper.h`, `OwnershipRoutineLogic.h`, `LLVMDebugInfo.h`), and bash test scripts.

**Spec:** `docs/superpowers/specs/2026-09-20-32-bit-compilation-design.md`: the Phase 4 section, and "Open issues". The union defect was deferred in phase 2 (ruling 1). A corpus probe of phase 4 found that it breaks real tests at i686.

## Global Constraints

- **The x64 output changes by design:** the storage type of every tagged union becomes `[N x i8]`. So the phase 2/3 rule "x64 IR unchanged" does **not** apply to this plan. It is replaced by the following:
  - `ctest -j 16 -C Release` from `__build/tslang/windows-msbuild-2026-release` stays at 2766/2766, plus any tests this plan adds.
  - The corpus probe (below) shows no file that passed before and fails after, at either arch and under any memory model.
  - x64 code size is measured and reported, not gated.
- **Union size and tag layout are unchanged** apart from the storage type. `N` is the same alloc size the old storage type had, so `sizeof` of every union stays the same. Check this.
- **Ownership (`-mm=rc`) must stay exact.** The rc verifier `--verify-ownership` must stay silent wherever it is silent today.
- Hard errors, not silent fallbacks.
- **Corpus probe:** `tslang/test/probe-corpus.sh`, added in Task 1. It compiles and runs every file in `tslang/test/tester/tests` as a single-file exe with the suite's flags (`--opt --opt_level=3 --no-default-lib --entry-point`), for x64 and i686 × gc, rc and none, and writes a TSV. The baseline TSV is taken from the plan's base commit. About 100 files per model fail at x64 too; those are multi-file tests that need the test runner, and "no new failures" is judged against the baseline, not against zero.
- **x86 library flags:**
  - i686: `--gc-lib-path=<repo>/3rdParty/gc/x64/release/lib`, `--tslang-lib-path=<repo>/__build/tslang-runtime/release`.
  - x64: `--tslang-lib-path=<repo>/__build/tslang/windows-msbuild-2026-release/lib` and the same gc path.
  - Unset `GC_LIB_PATH`, `TSLANG_LIB_PATH`, `GC_SHARED_LIB_PATH` and `DEFAULT_LIB_PATH`.
- Commits are GPG-signed. Never bypass signing; if it times out, leave the change staged and report. The trailer is `Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>`.
- Tests never read a catch variable's value.

## Measured starting point (main b61e6c6c, corpus probe)

| | x64 pass | i686 pass |
| --- | --- | --- |
| gc | 423 | 410 |
| rc | 418 | 409 |
| none | 418 | 411 |

Fifteen files fail only at i686. Of those, `02union_type` is confirmed to be this defect, and `03union_type` and `03union_type_case_order` show the same `types have different sizes` warnings. `internals`, `00typeof_static_fold_conditions`, `parser`, `arrayLiterals` and `arrayLiterals2ES5` (rc) are not diagnosed; some may be this defect. The async files belong to phase 4b.

## File structure

| File | Change | Task |
| --- | --- | --- |
| `tslang/test/probe-corpus.sh` | New. Runs the whole corpus as single-file exes and writes a TSV. | 1 |
| `tslang/test/tester/tests/00union_member_padding.ts` | New. Round-trips union members whose fields fall in another member's padding. | 1 |
| `tslang/test/tester/tests/00bigint_struct_layout.ts` | Asserts `a` again. | 1 |
| `tslang/test/tester/CMakeLists.txt` | Registers the new test. | 1 |
| `tslang/lib/TypeScript/LowerToLLVM.cpp` (~6690, the `UnionType` conversion) | Storage becomes `[N x i8]`. | 2 |
| `tslang/include/TypeScript/LowerToLLVM/LLVMTypeConverterHelper.h` (`findMaxSizeType`) | Adds a storage-size query. | 2 |
| `tslang/include/TypeScript/LowerToLLVM/CastLogicHelper.h`, `OwnershipRoutineLogic.h`, `UnaryBinLogicalOrHelper.h`, `LLVMDebugInfo.h`, `LowerToLLVM.cpp` (union op lowerings ~2123-2230) | Each reader of the union value is audited and fixed where needed. | 2 |

---

### Task 1: Failing tests and the corpus probe

**Files:**
- Create: `tslang/test/probe-corpus.sh`, `tslang/test/tester/tests/00union_member_padding.ts`
- Modify: `tslang/test/tester/tests/00bigint_struct_layout.ts`, `tslang/test/tester/CMakeLists.txt`

**Interfaces:**
- Produces:
  - `tslang/test/probe-corpus.sh <tslang.exe> <out-dir> [jobs]`. It writes `<out-dir>/results.tsv` with the columns `name arch mm stage code`, where `stage` is one of `pass`, `compile`, `link`, `run` or `timeout`.
  - `tslang/test/probe-corpus.sh --compare <before.tsv> <after.tsv>`. It prints every row that passed before and does not pass after, then a summary line. It exits non-zero if there is any such row.

- [ ] **Step 1: The probe script.**
  - One compile-and-run per file, arch (`x64`: `x86_64-pc-windows-msvc`, `x86`: `i686-pc-windows-msvc`) and memory model.
  - It runs with `xargs -P <jobs>` (default 12), a 30 s `timeout` per run, and in a per-case directory under the output dir. Exes and objects are deleted after each run.
  - It unsets the lib env vars and passes the lib flags from Global Constraints. It fails loudly (`FAIL missing prerequisite`) if `tslang.exe`, the libs or `timeout` are missing.
  - The classification: compile exit ≠ 0 with `LNK` in the output is `link`; any other non-zero compile exit is `compile`; a run exit of 124 is `timeout`; any other non-zero run exit is `run`.
  - Keep it hand-run, like the other `check-*.sh` scripts.
  - Run it at this plan's base commit and keep `results.tsv` as the baseline, outside the repo, in the SDD workspace. Report the per-arch and per-model counts; they should match the table above within a file or two.

- [ ] **Step 2: The padding test.** Create `tslang/test/tester/tests/00union_member_padding.ts`. The shapes below are chosen so that one member's field falls in another member's padding: at x64 for the first union, and at i686 for the second.

```ts
// A union is stored as one byte buffer. Each member must read back exactly what it was
// stored with, even when its fields sit where another member's struct has padding.

type Three = { flag: boolean; a: s32; b: s32 };
type Big = { flag: boolean; v: number };
type U1 = Three | Big;

function makeThree(): U1 { return { flag: true, a: 7, b: 9 }; }
function makeBig(): U1 { return { flag: false, v: 1.5 }; }

type Loading = { state: string };
type Failed = { state: string; code: number };
type Success = { state: string; response: { title: string; duration: number; summary: string } };
type Net = Loading | Failed | Success;

function describe(s: Net): string {
    if (s.state == "failed") return `code ${(<Failed><any>s).code}`;
    if (s.state == "success") return `title ${(<Success><any>s).response.title}`;
    return "loading";
}

function main() {
    const t = makeThree();
    const t2 = t;                       // copy the union as a value
    const three = <Three><any>t2;
    assert(three.a == 7);
    assert(three.b == 9);

    const b = makeBig();
    const b2 = b;
    assert((<Big><any>b2).v == 1.5);

    const f: Net = { state: "failed", code: 1.0 };
    const f2 = f;
    assert(describe(f2) == "code 1");
    assert(describe({ state: "success", response: { title: "t", duration: 2, summary: "s" } }) == "title t");
    assert(describe({ state: "loading" }) == "loading");

    print("done.");
}
```

  - Use `<T><any>u` to get a member back. Phase 2 found that `u.field` narrowing of object unions reads through the first member.
  - If a construct is rejected, adapt it, keeping two things: some union member's field lies in another member's padding at x64, and some lies in padding at i686. Put a comment next to each union saying which member's field lies in which padding, at which arch.
  - Register it in `tslang/test/tester/CMakeLists.txt` next to `00bigint_struct_layout`, following that entry's pattern.

- [ ] **Step 3: Re-arm `00bigint_struct_layout`.** It currently documents that it cannot assert `a`. Find that comment and add the assert back.

- [ ] **Step 4: Confirm the tests fail.**
  - Build them at x64 and i686 under gc, rc and none with the current compiler. At least one assert must fail at x64 (the `a` field), and `describe(f2)` must fail at i686.
  - Record which assert fails where. If nothing fails at x64, the first union shape is wrong for x64: change it until it fails, and explain the layout in the comment.
  - `ctest -j 16 -C Release` now has the new failures, and nothing else.

- [ ] **Step 5: Commit.** Commit the script and tests as they are. The suite is red on the two union tests until Task 2; say so in the commit message.

---

### Task 2: Byte-array union storage

**Files:**
- Modify: `tslang/lib/TypeScript/LowerToLLVM.cpp`:
  - the `UnionType` conversion, near line 6690;
  - `CreateUnionInstanceOpLowering` and `GetValueFromUnionOpLowering`, near lines 2123-2230.
- Modify: `tslang/include/TypeScript/LowerToLLVM/LLVMTypeConverterHelper.h`
- Audit, and modify where needed:
  - `tslang/include/TypeScript/LowerToLLVM/CastLogicHelper.h` (lines ~570, ~623, ~1121: the three `findMaxSizeType` callers, and the struct-to-struct path ~790-815);
  - `OwnershipRoutineLogic.h` (GEPs to `UNION_VALUE_INDEX` at ~941 and ~1095);
  - `UnaryBinLogicalOrHelper.h`;
  - `LLVMDebugInfo.h`;
  - every other use of `UNION_VALUE_INDEX` or `findMaxSizeType`.

**Interfaces:**
- Consumes: Task 1's tests and `probe-corpus.sh`.
- Produces:
  - `LLVMTypeConverterHelper::getUnionStorageSize(mlir_ts::UnionType) -> unsigned`: the largest member's alloc size, in bytes.
  - The union's LLVM type becomes packed `{ptr, [N x i8]}` when it needs a tag. When it doesn't, it is the merged base type, unchanged: `isUnionTypeNeedsTag` false means there is no real union.

- [ ] **Step 1: Storage size.** In `LLVMTypeConverterHelper`, add:

```cpp
    // Bytes a union's value field must hold: the largest alloc size among its members.
    unsigned getUnionStorageSize(mlir_ts::UnionType unionType)
    {
        unsigned size = 0;
        for (auto subType : unionType.getTypes())
        {
            size = std::max(size, (unsigned)getTypeAllocSizeInBytes(typeConverter->convertType(subType)));
        }

        return size;
    }
```

Keep `findMaxSizeType` for now. Step 3 decides whether any caller still needs a member *type*, not only a size.

- [ ] **Step 2: The conversion.** In the `UnionType` conversion, when a tag is needed, use `LLVM::LLVMArrayType::get(th.getI8Type(), ltch.getUnionStorageSize(type))` as the value field instead of `selectedType`. Put a short comment above it that states the reason: the padding of a member struct is not carried by a value copy, and at 32-bit a member's `f64` sits in the storage member's padding. The no-tag path stays as it is.

- [ ] **Step 3: Audit every reader.** For each use of `UNION_VALUE_INDEX`, `findMaxSizeType` and the union's converted type, write down in your report what it does and whether a byte array breaks it.
  - **Expected safe.**
    - `CreateUnionInstanceOp` and `GetValueFromUnionOp` build a packed `{ptr, member}` and move it through memory with `castLLVMTypes`, and `CopyStructOp` copies `min(src, dst)` bytes.
    - Ownership GEPs to the value field and then load or store the member type through that pointer.
    - These are memory reinterpretations, which are correct for any storage type of the same size. Confirm each one.
  - **Expected to need a change.**
    - The union→union cast in `CastLogicHelper.h` (~623) passes `findMaxSizeType(inUnionType)` as a value type. It would now carry the source's *member* type into a different union. Use the source union's byte-array type instead. Check that `castLLVMTypes` handles array→array and array→struct through memory. Its struct-to-struct branch only matches `LLVMStructType` on both sides: extend it to cover `LLVMArrayType` as well, using the same memory path.
    - The ~1121 caller: read it and decide.
    - `LLVMDebugInfo.h`: describe the value field as a byte array, or as the members over the same offset. Keep `--di` builds compiling and running; the suite runs a `--di` variant.
  - **Anything else** that extracts the value field *as a value* and uses it as a member struct without going through memory is a bug under this change. Route it through memory as `GetValueFromUnionOp` does.

- [ ] **Step 4: Build and run the Task 1 tests.** `00union_member_padding` and `00bigint_struct_layout` must pass at x64 and i686 under gc, rc and none. Build them at i686 with `--emit=llvm` too, and confirm there are no `types have different sizes` warnings for those unions. The warning compares struct sizes, which are now equal by construction, or arrays, which are no longer compared.

- [ ] **Step 5: Verify everything.**
  - `ctest -j 16 -C Release` passes everything: 2766 plus the new test.
  - `probe-corpus.sh` after the change, then `--compare <baseline> <after>`: nothing that passed before fails after. Report the files that pass *now* and did not before; `02union_type`, `03union_type` and `03union_type_case_order` at i686 are expected among them. List the remaining i686-only failures from the phase 4 measurement that still fail, with one line each on whether they look union-related.
  - rc: run `--verify-ownership` on the rc corpus files that use tagged unions. The union files in the corpus are the ones whose `--emit=mlir` output contains `ts.CreateUnionInstance`. The verifier must stay silent where it was silent at the base commit.
  - **x64 code size:** total `.text` size of the corpus exes before and after (`llvm-readobj --sections`), or the byte count of `--emit=obj` objects if that's simpler. Report it; it's not gated.

- [ ] **Step 6: Commit.** Use one commit for the storage change and its audit fixes, with a message explaining the defect and the fix.

---

## Phase 4a gate

- [ ] `00union_member_padding` and `00bigint_struct_layout` (with `a` asserted) pass at x64 and i686 under gc, rc and none.
- [ ] `ctest -j 16 -C Release` passes everything.
- [ ] `probe-corpus.sh --compare` against the base-commit baseline: no new failures at either arch.
- [ ] `02union_type` passes at i686. The fate of every other i686-only failure is reported.
- [ ] x64 code-size change reported.

## Rulings made while writing this plan

- **`[N x i8]`, not an array of words.** Alignment is irrelevant because the union struct is already packed. A byte array is the one storage type with no padding at every size. If the measured code size shows a real regression, an `[N/8 x i64]`-plus-tail layout can follow in a separate change.
- **`findMaxSizeType` stays** until the audit shows no caller needs a member type. Deleting it is not the goal.
- **The phase 2 union ruling is reversed.** Phase 2 deferred this defect as x64-only and rare. At i686 it breaks corpus tests, so it blocks phase 4's gate.
