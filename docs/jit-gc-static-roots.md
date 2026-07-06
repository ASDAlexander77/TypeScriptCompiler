# JIT crash: GC collects objects referenced only by globals (test-jit-Raytrace)

Status: **fixed** (PR #184, 2026-07-05). Files: `tslang/jit.cpp`,
`lib/TypeScriptRuntime/gc.cpp`, `include/TypeScript/gcwrapper.h`,
`lib/TypeScriptRuntime/TypeScriptRuntime.def`.

## Symptom

`test-jit-Raytrace` (raytrace.ts) crashed with `0xC0000005` when run under
`--emit=jit`, most reliably at `--opt_level=0` (the debug tester configuration).
The test had been disabled since 2023 with the comment
`# TODO: find out why it is crashing`.

Characteristic behaviour that pointed at the garbage collector:

| Configuration                  | Result  |
|--------------------------------|---------|
| AOT (`--emit=exe`) at O0       | passes  |
| JIT at O0, 1x1 or 16x16 render | passes  |
| JIT at O0, 48x48 and larger    | crashes |
| JIT at O0 with `--nogc`        | passes  |

The crash appears exactly when the Boehm heap grows enough to trigger the
first collection cycle.

## Root cause 1: JIT data sections are not GC roots

The Boehm GC discovers static roots automatically only for *loader-mapped
modules*: on Windows it walks the address space and registers writable
`MEM_IMAGE` regions (the `.data`/`.bss` of the exe and every DLL). That is why
AOT binaries work out of the box.

JIT'd code is different. The MLIR `ExecutionEngine` links objects with
RTDyld, whose `SectionMemoryManager` obtains section memory from plain
`VirtualAlloc`/`mmap` (`MEM_PRIVATE`). The globals of the JIT'd module —
including every static class member — live in those sections, and **the GC
never scans them**.

Consequence: an object whose only reference is a global (`Color.background`,
`Surfaces.shiny`, … in raytrace.ts) is unreachable from the GC's point of
view. On the first collection it is freed, its block is recycled by a later
allocation, and the next access through the stale pointer reads garbage or
faults (in raytrace the fault came from calling `thing.surface.reflect()`
through a recycled object).

### Minimal reproduction

```ts
class Vec {
    constructor(public x: number, public y: number, public z: number) {}
}

class Statics {
    static background = new Vec(11, 22, 33);
}

function main() {
    for (let i = 0; i < 3000000; i++) {
        let v = new Vec(i, i, i);   // force GC cycles
    }
    print(Statics.background.x);    // expected 11
}
```

Before the fix this printed `2.9751e+006` — i.e. the static's memory had been
recycled as `Vec(i, i, i)` with `i ≈ 2975100`, an allocation from the last
collection cycle. With `--nogc` it printed `11`.

### Fix

`tslang/jit.cpp` installs a custom `llvm::SectionMemoryManager::MemoryMapper`
(`GCRootsSectionMemoryMapper`) through
`mlir::ExecutionEngineOptions::sectionMemoryMapper`. Every **RWData** section
allocation is registered with the collector:

```
GC_add_roots(base, base + size);   // on allocate (RWData only)
GC_remove_roots(base, base + size); // on release
```

`GC_add_roots`/`GC_remove_roots` are resolved at run time with
`llvm::sys::DynamicLibrary::SearchForAddressOfSymbol` from the already-loaded
`TypeScriptRuntime` library, so the mapper is inert under `--nogc` or when no
GC runtime is present. The exports were added as thin wrappers
(`_mlir__GC_add_roots`, `_mlir__GC_remove_roots`) in
`lib/TypeScriptRuntime/gc.cpp` and re-exported under the plain names in
`TypeScriptRuntime.def`, following the existing pattern for `GC_malloc` etc.

Notes:

- `GC_add_roots` auto-initializes the collector (bdwgc 8.2), so registering
  sections *before* the JIT'd `__mlir_gctors` calls `GC_init()` is safe.
- `SectionMemoryManager` sub-allocates several object sections from one mapped
  page, so registering the whole mapped block covers `.data`, `.bss` and any
  later RW sections placed in it.
- The number of RW blocks per module is tiny compared to bdwgc's
  `MAX_ROOT_SETS` (~8192).

## Root cause 2: two TypeScriptRuntime copies = two GC instances

The first fix initially appeared not to work, which exposed a second bug.

`runJit()` unconditionally appended a fallback `TypeScriptRuntime` path
(resolved through `TSLANG_LIB_PATH` / the tslang lib directory) to the shared
library list — *even when the caller had already passed one via
`--shared-libs`*. When the two paths point at different files, Windows loads
**two copies of TypeScriptRuntime.dll**, each with its own statically linked
Boehm GC: two heaps, two root sets, two sets of collector state.

The host registered roots in the copy found first by
`SearchForAddressOfSymbol`, while the JIT'd code allocated from the other
copy — so the registration had no effect. Beyond this bug, the double load is
a general hazard: any pointer that crosses the two heaps (realloc/free of a
block owned by the other instance) corrupts memory.

This is easy to hit locally: the test runner passes
`--shared-libs=<build>/bin/TypeScriptRuntime.dll` while `TSLANG_LIB_PATH`
points at an installed copy (e.g. `I:\tslang`).

### Fix

`runJit()` now skips the fallback lookup when the `--shared-libs` list already
names a `TypeScriptRuntime` library (case-insensitive stem match). Exactly one
copy of the runtime — and therefore one GC — is loaded per JIT process.

## How to diagnose this class of bug

1. `--nogc` passing while the normal run crashes ⇒ collector involvement.
2. Crash threshold scaling with allocation volume ⇒ first collection cycle.
3. AOT passing while JIT fails ⇒ suspect JIT-only differences: static roots,
   symbol resolution, unwind info.
4. Print a static's fields after an allocation-heavy loop (repro above): a
   value tracking the loop counter proves the block was recycled.
5. Check for duplicate runtime copies:
   `Get-Process tslang | % Modules | ? ModuleName -match TypeScriptRuntime` —
   more than one entry (or one from an unexpected path) means two GC
   instances.
6. `tslang --emit=jit --dump-object-file --object-filename=out.obj …` dumps
   the JIT'd object; `llvm-nm`/`llvm-readobj` show which section a global
   lives in (`B`/`D` symbols ⇒ `.bss`/`.data` ⇒ must be a GC root range).

## Related

- The same PR fixed `00for_await_yield` at `-O3`: `print()`'s string-concat
  lowering emitted a variable-size stack `alloca` inside `async_execute_fn`
  coroutines, which LLVM's CoroSplit rejects ("Coroutines cannot handle non
  static allocas yet"); `StringConcatOpLowering` now heap-allocates when the
  enclosing function carries the `presplitcoroutine` attribute.
- Known, still open: a tight multi-million-iteration `new` loop at
  `--opt --opt_level=3` under JIT dies with `0xC00000FD` (stack overflow);
  reproduces on pre-fix binaries, unrelated to the GC roots work.
