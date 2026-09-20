# 32-bit compilation

Design for compiling TypeScript to 32-bit targets, with `i686-pc-windows-msvc`
as the proof target and width-correctness that generalizes to any 32-bit triple.

Status: approved design, not yet implemented.

## Why now

Until #333, every compiled program linked `LLVMSupport.lib` and
`LLVMDemangle.lib`. Those come from the prebuilt LLVM in `3rdParty/llvm/x64`,
which exists only as x64 — so a 32-bit link was blocked by a library we could
not rebuild cheaply. #333 removed that dependency. Every remaining x64 blocker
is a library this repository builds from source.

## Measured baseline

Probed with the Sep-18 build (`__build/tslang/windows-msbuild-2026-release`),
target `i686-pc-windows-msvc`. These are measurements, not estimates.

| Case | Result |
| --- | --- |
| `--emit=obj -mm=none --no-default-lib` | Works. `hello.obj` machine type `0x014c` (`IMAGE_FILE_MACHINE_I386`). |
| `--emit=exe -mm=none --no-default-lib` | Works. `hello.exe` PE machine `0x014c`; runs and prints. |
| `LLVMSupport.lib` in the link | Gone, as of #333. (The older `__build/tslang/msbuild/x64/release` build still drags it in.) |
| Exceptions | `unresolved external symbol __CxxThrowException`; linker hints `__CxxThrowException@8`. |
| `-mm=gc` | `gc.lib : warning LNK4272: library machine type 'x64' conflicts with target machine type 'x86'`, then unresolved `_GC_init`, `_GC_enable_threads`. |
| `TypeScriptAsyncRuntime.lib` | Same x64/x86 machine-type conflict. |

clang's driver locates the x86 CRT and Windows SDK import libraries on its own.
No toolchain-discovery work is needed.

### Two findings that reduced scope

Both were assumptions that probing disproved. They are recorded because the
obvious reading of the emitted IR suggests the opposite.

**The async runtime ABI is not a width bug.** Emitted IR declares
`mlirAsyncRuntimeAddRef(ptr, i64)`, `mlirAsyncRuntimeCreateValue(i64)` and
similar regardless of triple, which looks like a 64-bit-shaped ABI. It is not:
`lib/AsyncRuntimeCommon.inc` declares these parameters as `int64_t`, not
`size_t`. `i64` is correct on both arches. Nothing to change.

**The Win32 EH struct layouts are already correct for x86.** `ThrowInfo` is
emitted as `{i32 x 4}` and `CatchableType` as `{i32 x 7}`
(`include/TypeScript/LowerToLLVM/LLVMRTTIHelperVCWin32.h`). On x64 those i32
fields hold image-base-relative RVAs. On 32-bit MSVC the corresponding fields
hold absolute pointers — which are also 4 bytes. The layouts coincide. What
differs is only the *value* stored: x86 must not subtract the image base.

## Architecture

One new type, `TargetInfo`, constructed once in `prepareOptions()` from
`llvm::Triple` and carried on `CompileOptions` alongside the parsed `Triple`
itself. It holds two kinds of fact:

- **Widths** — `pointerBits`, `indexBits`.
- **Policy** — `usesImageBaseRelativeEH`, `stdcallDecoratesCxxThrow`,
  `supportsInProcessJit`.

`TypeHelper` gains `getSizeType()` and `getPointerIntType()` backed by it.
`TypeHelper` currently takes only an `MLIRContext*`, so it grows a
`CompileOptions` parameter. That is a wide but mechanical diff and the single
riskiest mechanical change in this plan.

Keeping the parsed `llvm::Triple` on `CompileOptions` as well means one-off
queries do not each require a new `TargetInfo` field.

### Naming policy predicates

Policy predicates are named for what they decide, not for the architecture that
currently needs them: `usesImageBaseRelativeEH`, never `isX64`. This is what
keeps phase 3 from smearing `arch == x86_64` across three RTTI helpers, and it
is what makes the answer reviewable in one place next to the comment explaining
it.

### Replacing the arch list

`tslang/opts.cpp` currently derives `sizeBits` from a hand-maintained list of
64-bit architectures. The list is wrong: it marks `aarch64_32` as 64-bit, but
ARM64_32 is an ILP32 target whose pointers are 32 bits. It is replaced by
`llvm::Triple::getArchPointerBitWidth()`, which is what makes "any 32-bit
triple" true rather than "the 32-bit triples someone remembered".

### Arch in the library layout

`x86` and `x64`, matching the existing `3rdParty/gc/x64`. Arch is the outermost
dimension under each component, so a whole arch tree ships or is deleted as one
unit and existing x64 paths keep their shape:

- `3rdParty/gc/x86/release/`, `3rdParty/gcdll/x86/release/`
- `defaultlib/{lib,dll}/x86/{debug,release}/{gc,rc,none}/`

Strict, no fallback — the rule the default-lib layout already follows. A
missing arch tree is an error that names the arch and the script that builds
it. It never falls back to another arch: that link succeeds and then corrupts
memory, which is far harder to diagnose than a directory that is not there.

## Phase 1 — Target width correctness

`TargetInfo` lands. `TypeHelper` takes `CompileOptions`. The arch list is
replaced.

There are 38 `getI64Type()` call sites across `lib/` and `include/`. Eighteen
of them are in the two Win32 RTTI helpers
(`include/TypeScript/LowerToLLVM/LLVMRTTIHelperVCWin32.h` and
`include/TypeScript/MLIRLogic/MLIRRTTIHelperVCWin32.h`, nine each); those are
phase 3's, since phase 3 rewrites that arithmetic anyway. Phase 1 audits the
remaining 20, and each is classified in a comment as one of:

- **genuinely 64-bit** — the language's `number` is f64; runtime ABI
  parameters declared `int64_t`; anything whose width is fixed by a contract
  outside the target.
- **target-width** — pointer arithmetic, object sizes, GEP indices.

Most are expected to be genuinely 64-bit. The audit's value is not the count of
changes but that the classification becomes explicit and survives future edits.

`target datalayout` is set at IR emission rather than only later in
`tslang/obj.cpp`, so `--emit=llvm` output is self-describing.

**Gate.** `--emit=llvm` for `i686-pc-windows-msvc`, `wasm32-unknown-unknown`
and `x86_64-pc-windows-msvc` emits the correct datalayout, and no 32-bit target
emits 64-bit-shaped size or pointer arithmetic. Verifiable with no native
builds.

## Phase 2 — 32-bit native build matrix

- `3rdParty/gc/x86` and `3rdParty/gcdll/x86`: Boehm built for x86, static and
  DLL, debug and release.
- `TypeScriptRuntime` and `TypeScriptAsyncRuntime` cross-built for x86.
- `tslang/exe.cpp` resolves `GC_LIB_PATH`, `GC_SHARED_LIB_PATH` and
  `TSLANG_LIB_PATH` per target arch rather than per host.

The compiler itself stays x64 and cross-compiles. Building `tslang.exe` as a
32-bit binary is out of scope.

`checkGCLibPath` and `checkTslangLibPath` grow a COFF machine-type check, so an
arch mismatch is reported against the path that is wrong rather than surfacing
as LNK4272 plus a list of unresolved symbols.

**Gate.** `-mm=gc` and `-mm=rc` link and run a 32-bit hello world.

## Phase 3 — Win32 x86 exception handling

Two changes, both narrow because of the finding above.

1. `_CxxThrowException` is `__stdcall` on x86. The block already present but
   commented out at `lib/TypeScriptExceptionPass/Win32ExceptionPass.cpp:696`
   sets `CallingConv::X86_StdCall`, which makes LLVM emit
   `__CxxThrowException@8`. Gated on `stdcallDecoratesCxxThrow`.
2. The image-base subtraction in `LLVMRTTIHelperVCWin32.h` (three sites) and
   the corresponding logic in `MLIRRTTIHelperVCWin32.h` is gated on
   `usesImageBaseRelativeEH`. On x86 the field takes the `ptrtoint` result
   directly. The `ptrtoint` and `sub` widths come from `TargetInfo` rather than
   being hardcoded `i64`.

**Testing.** Tests assert that control flow reaches the catch, and on
unwinding order. They do not assert on the catch variable's contents: reading a
catch variable's value is a known-unreliable area independent of this work, and
a test written on it would fail for reasons that have nothing to do with 32-bit
support.

**Gate.** throw/catch, nested catch, try/finally and rethrow behave correctly
as 32-bit binaries.

## Phase 4 — Default library and the suite

`getDefaultLibSubDir` in `include/TypeScript/Defines.h` gains an arch segment,
giving `defaultlib/{lib,dll}/x86/{debug,release}/{gc,rc,none}/`. The
default-lib build scripts and the test runner gain an arch flag.

Async/await and generators are measured here rather than in an earlier phase.
They depend on phases 2 and 3 and cannot run before both are in place. Nothing
in the probing suggests they are separately broken; if they are, that is a
finding of this phase.

**Gate.** The same suite that runs at 64 bits runs at 32 bits, green. Genuine
failures are triaged and recorded the way the Debug-suite failures are, not
silently excluded.

## Error handling

Three failures get explicit, actionable messages instead of downstream
confusion:

1. **Missing per-arch library tree** — names the arch and the script that
   builds it. Not a fallback to another arch.
2. **Arch mismatch in a library that is present** — reported against the
   offending path at resolution time, via the machine-type check, rather than
   as a linker warning buried among unresolved symbols.
3. **`--emit=jit` with a non-host arch** — `tslang.exe` is an x64 process and
   cannot execute i386 code in-process. Refused up front with a message saying
   so. What the combination does today must be confirmed before this guard is
   written; the guard is `supportsInProcessJit`.

## Out of scope

- `-m32` shorthand. `-mtriple=i686-pc-windows-msvc` already works and is the
  mechanism; a second spelling earns nothing.
- Building `tslang.exe` itself as a 32-bit binary.
- 32-bit targets other than `i686-pc-windows-msvc` as *proof* targets. Phase 1
  makes width-correctness general, and `wasm32` continues to be exercised, but
  only `i686-pc-windows-msvc` gets a full native build matrix and suite run.

## Decisions taken

| Decision | Choice | Alternative rejected because |
| --- | --- | --- |
| Where target facts live | `TargetInfo` on `CompileOptions`, derived once from `llvm::Triple` | Querying the `Triple` at each site scatters policy across 38 call sites and relocates the arch-list rot rather than fixing it. Querying MLIR's `DataLayout` cannot express EH policy at all, and much width-dependent code runs in MLIRGen where no `DataLayout` is in scope. |
| Arch naming | `x86` / `x64` | Matches `3rdParty/gc/x64`, which already exists. |
| Arch position in paths | Outermost under each component | A whole arch tree ships or is deleted as one unit; x64 paths keep their shape. |
| Missing arch tree | Hard error | A fallback links and then corrupts memory. |
| Suite scope at 32 bits | The same suite as 64 bits | A declared subset hides exactly the failures this work exists to find. |
