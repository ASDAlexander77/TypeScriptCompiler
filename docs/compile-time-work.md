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

Open: PR #865, branch `perf-compile-time-4`.

| Commit | What |
| --- | --- |
| `8f303db6` | `StackEscapeFacts` looks functions up in a table. |
| `5254b577` | `LLVMGlobalsIndex::lookupSymbol`: the callee of a call is found from a table of the symbols the module has before the conversion. |
| `fea78525` | An enum's member dictionary is made once, not after every member. |

Also on `perf-compile-time-4`, after those three (see `git log`; all went through section 5):

- **Bounded comdat tables** (`addComdatSelector`, `COMDAT_TABLE_SELECTORS` in `LowerToLLVM.cpp`;
  `comdatTable` in `LLVMGlobalsIndex.h`). Translation to LLVM IR for 2,000 generic instances:
  1.8 s -> 0.5-0.7 s.
- **Symbols the lowering makes are tabled** (`createSymbol`, `lookupOwnSymbolAs`, `createdSymbols`
  in `LLVMGlobalsIndex.h`; 14 lookups and 17 creations routed in `OwnershipRoutineLogic.h`,
  `ConvertLogic.h`, `LLVMCodeHelperBase.h`, `LLVMCodeHelper.h`, `LowerToLLVM.cpp`). A first version
  treated every name missing from the tables as absent from the module; the check caught `puts`,
  which MLIR's own `cf.assert` lowering declares. The rule is now: a miss is final only for names
  the lowering itself invents (`ownName`), otherwise the scan decides. LLVM lowering of 2,000
  interfaces: 2.0 s -> 1.4 s.
- `tslang/benchmarks/compile-time/` (the scripts of section 2) and this document.

Figures, object file, no optimisation, `--no-default-lib`, Windows x64, Release compiler:

| Input | At the start | Now |
| --- | --- | --- |
| 2,000 classes of 5 lines | 99.9 s | 2.5 s |
| 8,000 classes | not measured (4,000: 19.7 s after #862) | 11.9 s |
| 2,000 instances of a generic class | 16.4 s | 6.0-7.0 s |
| 2,000 interfaces with a class and a use each | 10.9 s | 7.3-8.7 s |
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

Measured with all of the above, n = 1,000 -> 2,000. Timings of the larger inputs moved by 10-15%
between runs on the development machine, and at n = 2,000 even code emission (LLVM's, linear)
came out at 2.8x on these two shapes, so read 2.5-2.9x here as "about linear, plus memory":

| Shape | Total | Growth | Where |
| --- | --- | --- | --- |
| interfaces | 2.8 -> 8.7 s | 3.1x | MLIR generation 0.9 -> 3.2 s (3.6x) is the one clear case; LLVM lowering 0.46 -> 1.36 s (2.9x); `OwnedReturnConsumptionPass` and `StackPromotionPass` 0.04 -> 0.2 s (4-5x, small) |
| one function of n `if` statements | 0.65 -> 1.7 s | 2.7x | `TypeScriptToAffineLoweringTSFuncPass` 0.23 -> 0.89 s (3.9x) |
| generics | 2.8 -> 7.0 s | 2.5x | nothing stands out; every stage 2.1-2.8x |
| inheritance, classes | | 2.3x | nothing stands out |

**Interfaces, MLIR generation.** `mlirGenNullInterfaceVTable` (`MLIRGenInterfaces.cpp`) makes one
throwing stub per interface method through `mlirGenThrowingMethodStub`, which asks
`theModule.lookupSymbol<FuncOp>(stubName)` before and after: the first is a miss and scans the
module. `INSTANCEOF_NONE_NAME` is looked up for every vtable and the function is pushed at the
*end* of the module, so each lookup scans all of it. `createGlobalVariable` still scans after a
dummy run made a global (the entry of `lastGlobalOfModule` is dropped then).

The sites were mapped on 2026-11-15 (branch `perf-compile-time-5`, cut from `perf-compile-time-4`,
nothing committed on it yet), and the plan below follows from them. Nothing of it is written.

- Only `mlir_ts::FuncOp` and `mlir_ts::GlobalOp` are symbols. A function enters a module through
  `theModule.push_back(funcOp)` and nowhere else: 8 sites (`MLIRGenFunctions.cpp` 1427 and 2310,
  `MLIRGenStatements.cpp` 1630, `MLIRGenInterfaces.cpp` 500, `MLIRGenClasses.cpp` 3484 and 3587,
  `MLIRGenCast.cpp` 2494 and 2650). A global enters through `createGlobalVariable`, and through
  `MLIRRTTIHelperVC`, which MLIR generation cannot see into.
- A function already in a module is erased in one place, `existing->erase()` in
  `mlirGenThrowingMethodStub`. The other `funcOp.erase()` calls (`MLIRGenStatements.cpp` 1616,
  `MLIRGenClasses.cpp` 3462 and 3547, `MLIRGenCast.cpp` 2624) erase a function that failed before
  it was pushed. `GenContext::clean` erases blocks and the globals of a dummy run, not functions.
  The temp and discovery modules are erased whole (where `forgetLastGlobalOf` is already called).
- Plan: per module, a table of symbol names, with the operation for a function and no operation
  for a global (a dummy run's global can be erased unseen). A name not in the table is not in the
  module, which is the answer the guards need fast: `mlirGenThrowingMethodStub` (the first
  lookup), `INSTANCEOF_NONE_NAME`, the thunk guards in `MLIRGenClasses.cpp` 3416 and 3502 and
  `MLIRGenCast.cpp` 2443 and 2576, `MLIRGenFunctions.cpp` 2228. A function in the table is
  returned without a scan (this is what fixes `INSTANCEOF_NONE_NAME`, found at the end of the
  module today); a global in it is looked up by the scan. Fill the table at the 8 `push_back`
  sites and in `createGlobalVariable`; build it by one scan for a module first seen with
  operations in it; drop a function's entry at the one erase, and a module's table where the
  module is erased.
- The risk is a symbol entering a module by a route the table does not hear of: its name would be
  reported absent and made twice. Names from `MLIRRTTIHelperVC` are only ever looked up by that
  helper, with its own scan. Extend the checking mode to this table (compare every answer with
  `theModule.lookupSymbol`) and run the suite with it before trusting it, as for the other tables.

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
  last global, symbol lookup) and `lastGlobalOfModule`. After a change to any of them: run the
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
