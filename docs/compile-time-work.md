# Compile-time work: state and how to carry on

Working notes for the compile-time optimisation of tslang itself, started 2026-11-15. Read this to
pick the work up in a new session. User-facing numbers are in [performance.md](performance.md#compile-time).

## 1. Where it stands

The compiler took a time that grew much faster than the source for several kinds of program. Each
case had the same root: something done once per global, call or class that scanned the whole
module. The fixes replace the scan with a table, and every table is checked against the scan it
replaces (section 4).

Merged:

| PR | What |
| --- | --- |
| #862 | The last global is found by scanning the module's top level, not every function body. `LLVMGlobalsIndex` for the LLVM lowering pass: names of LLVM globals and the last global of each kind. |
| #863 | The comdat table is first in the module. `--mlir-timing` works. |
| #864 | MLIR generation remembers each module's last global (`MLIRGenImpl::lastGlobalOfModule`). |

| #865 | `StackEscapeFacts` and `LLVMGlobalsIndex::lookupSymbol` look symbols up in tables; an enum's member dictionary is made once; bounded comdat tables; the symbols the lowering makes are tabled (`createSymbol`, `lookupOwnSymbolAs`: a miss is final only for names the lowering itself invents - the check caught `puts`, which MLIR's `cf.assert` lowering declares); `tslang/benchmarks/compile-time/` and this document. |

Not merged yet, branch `perf-compile-time-5`:

- **MLIR generation tables each module's symbols** (`symbolsOfModule`, `lookupModuleSymbol`,
  `pushFunction` in `MLIRGenImpl.h`). Every `theModule.lookupSymbol` of MLIR generation goes through
  it; a name that is not in the table is not in the module.
- **An exception's records are made once per type and module** (`mlirGenRttiForType`): a `throw`
  of a type already seen no longer has `MLIRRTTIHelperVC` look each record up by name and scan
  for the last global.
- MLIR generation of 2,000 interfaces: 3.1 s -> 1.0 s, and it now doubles with the source.

Figures, object file, no optimisation, `--no-default-lib`, Windows x64, Release compiler:

| Input | At the start | Now |
| --- | --- | --- |
| 2,000 classes of 5 lines | 99.9 s | 2.5 s |
| 8,000 classes | not measured (4,000: 19.7 s after #862) | 11.9 s |
| 2,000 instances of a generic class | 16.4 s | 6.0-7.0 s |
| 2,000 interfaces with a class and a use each | 10.9 s | 5.0 s |
| 2,000 classes overriding a base method | 3.3 s | 2.4 s |
| an enum of 2,000 members and a switch | 1.8 s | 0.3 s |
| 2,000 functions of 5 lines | 1.8 s | 1.7 s |

## 2. How to measure

**`--mlir-timing`** prints the time of each stage (MLIR generation, every MLIR pass, translation
to LLVM IR, LLVM optimisation, code emission) for any `--emit`. This is the number to trust.

**`tslang/benchmarks/compile-time/shapes.ps1`** generates 14 program shapes at growing sizes and
prints the time and the growth from one size to the next. Linear is about 2.0x for twice the
source; 3x and more means something quadratic.

```
pwsh tslang/benchmarks/compile-time/shapes.ps1                         # all shapes, n = 500, 1000, 2000
pwsh tslang/benchmarks/compile-time/shapes.ps1 -Only interfaces -Stages   # which stage grows
```

**`tslang/benchmarks/compile-time/profile.ps1 <file.ts>`** samples call stacks of the Release
compiler (relinked with symbols into `out/`, the build directory untouched) and prints the hot
functions. Use it to find the code inside a stage, not to compare stages: it takes fewer samples
where stacks are deep, and it once showed a lookup at 31% that was 5% of the wall time. A
per-function pass runs on a worker thread; pass `-Flags --mlir-disable-threading` to see it.

The suite's largest test is 1,500 lines and shows none of this. Only generated inputs do.

## 3. What is left, with the cause of each

All 14 shapes of `shapes.ps1` now grow 1.7-2.6x from n = 1,000 to 2,000. Timings of the larger
inputs move by 10-15% between runs on the development machine, so read up to about 2.5x as linear.
What stands out:

| Shape | Total | Growth | Where |
| --- | --- | --- | --- |
| one function of n `if` statements | 0.66 -> 1.7 s | 2.6x | `TypeScriptToAffineLoweringTSFuncPass` 0.23 -> 0.89 s (3.9x) |
| interfaces | 2.2 -> 5.0 s | 2.3x | Canonicalizer 0.28 -> 0.73 s (2.6x); `OwnedReturnConsumptionPass` and `StackPromotionPass` were 4-5x but small |

**One very large function.** The `IfOp` lowering in `LowerToAffineLoops.cpp` calls
`rewriter.splitBlock`. Under a conversion rewriter that moves the rest of the block one operation
at a time, each with a recorded rewrite, so n sequential `if`s move n^2/2 operations. Options, none
tried: lower the `if`s of a block last to first, so each split moves little; or build the
continuation block once per region instead of splitting at every `if`. Upstream's SCF-to-CF has
the same shape.

**Not looked at:** `OwnedReturnConsumptionPass` and `BorrowFactsExportPass` showed 4-5x growth on
some shapes but small absolute times (0.2 s); the canonicalizer runs grow 2.3-2.6x; `-mm=rc` and
`-mm=own` were only timed on classes; no shape uses the default library; nothing was measured on
Linux.

## 4. The rules that keep the tables honest

Every table answers a question a scan used to answer, and must give the same answer.

- **`TSLANG_VERIFY_LLVM_GLOBALS_INDEX=<file>`** makes the compiler ask the scan too, append any
  difference to the file and use the scan's answer. It covers `LLVMGlobalsIndex` (global lookup,
  last global, symbol lookup), `lastGlobalOfModule`, `symbolsOfModule` and `mlirGenRttiForType`. After a change to any of them: run the
  whole suite with it set, and the file must not exist afterwards. It has caught two wrong
  versions that the plain suite's IR comparison would also have shown, and one (`puts`) that it
  would not have.
- **LLVM globals** made during the conversion go through `LLVMCodeHelperBase::createGlobal`;
  other symbols through `LLVMGlobalsIndex::createSymbol`. A grep for `create<LLVM::GlobalOp>`,
  `create<LLVM::LLVMFuncOp>` and `create<func::FuncOp>` under `LowerToLLVM` should find only the
  wrappers and the code that runs after `applyFullConversion`.
- **Rollback.** The conversion driver erases what a failed pattern created. The index is the
  conversion's listener (`notifyPatternEnd` arrives when a pattern fails, in MLIR 22.1.8) and
  forgets its pointers when a pattern that created something fails. Pointers to symbols that
  existed before the conversion are safe until it ends.
- **A miss is final only for the lowering's own names.** MLIR's patterns create symbols too.
- **In MLIR generation** the remembered global of a module is dropped where a global or a module
  is erased, for dummy-run globals, and around `MLIRRTTIHelperVC` (`RttiGlobalsScope`), which
  makes globals itself.
- **The table of a module's symbols** (`symbolsOfModule`) hears of a function at `pushFunction`,
  the only way one enters a module, and of a global at `createGlobalVariable`. A name with no
  operation in it is looked up by the scan: a global of a dummy run (erased unseen), and two
  symbols of one name. The globals `MLIRRTTIHelperVC` makes are not heard of, so the table is
  dropped, and made again by a scan, when a call of the helper added to the module
  (`mlirGenRttiForType`, the one place it is called from). Under the check the helper is still
  called for a type it was called for, and must add nothing.
- A cheaper comparison inside a scan does not help: the cost is walking the operation list.

## 5. Verifying a change

1. Build Release: `cmake --build . --config Release --target tslang -- /m:6` in
   `__build/tslang/windows-msbuild-2026-release`.
2. `shapes.ps1` for the shapes touched, before and after.
3. The suite with the check: clear `DEFAULT_LIB_PATH`, `GC_LIB_PATH`, `LLVM_LIB_PATH`,
   `TSLANG_LIB_PATH`, `TSLANG_HOME`; set `TSLANG_VERIFY_LLVM_GLOBALS_INDEX`; `ctest -C Release -j 10
   --timeout 300`. Then the suite without it.
4. Compare LLVM IR with a compiler from before the change (`--emit=llvm`, default options,
   `--opt --opt_level=3`, `--mm=rc`) on the largest tests. `00param_assigned_owned.ts` differs
   from run to run with any compiler (the same lines in another order) and is not a finding.

Known on the development machine: `test-borrow-facts-*` (5 tests) fail because the build cache's
`CMAKE_AR` names an MSVC toolset that is no longer installed. They pass when their commands are
rerun with the installed `lib.exe`; a CMake reconfigure fixes it.

Not done for any of this work: a Linux build, and a run of the suite with a Debug-built compiler.
The test workflows do not run on pull requests, so nothing checks Linux until one is started by
hand.
