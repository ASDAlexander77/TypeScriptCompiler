# LLVM GC Infrastructure — Integration Estimate for the TypeScript Compiler

Reference: [LLVM Garbage Collection docs](https://llvm.org/docs/GarbageCollection.html)

This document estimates which parts of LLVM's garbage-collection support framework
(`gc` function attribute, `GCStrategy`, `llvm.gcroot`, `llvm.gcread`/`llvm.gcwrite`,
`gc.statepoint`, stack maps / `GCMetadataPrinter`) are applicable to the compiler's
current GC implementation, what each would buy us, and a phased implementation plan.

---

## 1. Current GC implementation (baseline)

The compiler uses **Boehm-Demers-Weiser GC 8.2.12** (`gc-8.2.12.tar.gz` in the repo root)
as a **conservative, non-moving** collector. Integration today is entirely at the
allocation-call level — no LLVM GC infrastructure is used:

| Piece | Location | What it does |
| --- | --- | --- |
| `GCPass` (MLIR, LLVM dialect level) | `tslang/lib/TypeScript/GCPass.cpp` | Renames `malloc`/`calloc`/`realloc`/`free`/`aligned_alloc` → `GC_malloc`/`GC_malloc_atomic`/`GC_realloc`/`GC_free`/`GC_memalign`; injects `GC_init()` into `main` or the global-ctors function; attaches `allockind("alloc")` + memory-effects attributes so GVN/EarlyCSE at `-O3` do not CSE two allocations into one; removes `memset` after `GC_malloc` (already zeroed). |
| Runtime wrapper | `tslang/lib/TypeScriptRuntime/gc.cpp`, `TypeScriptGC.cpp`, `tslang/include/TypeScript/gcwrapper.h` | `_mlir__GC_*` wrappers over Boehm: `GC_init`, `GC_malloc[_atomic]`, `GC_memalign`, `GC_realloc`, `GC_free`, `GC_add_roots`/`GC_remove_roots`, typed allocation (`GC_malloc_explicitly_typed`, `GC_make_descriptor`), disappearing links, `GC_gcollect`. |
| Typed (precise-heap) allocation | `ENABLE_TYPED_GC` in `tslang/include/TypeScript/Config.h` (on by default); `MLIRGen.cpp` (`NewClassInstanceLogicAsOp`), `LowerToLLVM.cpp` (`GCMakeDescriptorOpLowering`, `GCNewExplicitlyTypedOpLowering`) | Class instances are allocated with `GC_malloc_explicitly_typed` using a per-class Boehm descriptor: a generated type-bitmap method feeds `GC_make_descriptor`, and the resulting descriptor is cached in a global so it is built once per class. This gives the heap scan pointer/non-pointer precision for class fields (less false retention); stacks/registers/globals remain conservatively scanned. |
| JIT static roots | `tslang/tslang/jit.cpp` | JIT-allocated RW data sections are registered via `GC_add_roots` (see `docs/jit-gc-static-roots.md`) because Boehm cannot discover JIT'd sections on its own. |

Properties that follow from Boehm being conservative and non-moving:

- **No stack maps needed** — Boehm scans thread stacks and registers conservatively.
- **No object relocation** — pointers never move, so no `gc.relocate` semantics are needed.
- **No read/write barriers required** in the default (stop-the-world, non-incremental) mode.
- Known weaknesses: false retention (integers that look like pointers), and the risk
  the LLVM docs explicitly call out — *aggressive optimization can hide the only live
  copy of a pointer* (kept as an interior/derived value or re-materialized arithmetic),
  making the object collectable while still in use.

## 2. What LLVM's GC framework provides, piece by piece

| LLVM facility | Purpose | Needed by Boehm? |
| --- | --- | --- |
| `gc "name"` attribute + `GCStrategy` plugin | Declares per-function GC strategy; hook for custom lowering and metadata | Not required, but a cheap formalization hook |
| `llvm.gcroot` | Marks stack slots holding heap references; forces them to be spillable/visible | Not required (conservative stack scan), useful as an anti-"hidden pointer" pin and for shadow-stack targets |
| `llvm.gcwrite` / `llvm.gcread` | Write/read barrier insertion points, lowered by the strategy | Only if we enable Boehm **incremental/generational** mode with manual dirty-marking |
| `gc.statepoint` / `gc.relocate` / `gc.result` + `RewriteStatepointsForGC` | Precise safepoints with relocation semantics for **moving** collectors | Not usable with Boehm (non-moving); only relevant if we ever swap the collector |
| Stack maps + `GCMetadataPrinter` | Emit binary root-location tables per safepoint for the runtime to crawl | Not consumable by Boehm as-is; relevant only for a precise-GC future |
| `shadow-stack` built-in strategy | Maintains an explicit linked list of roots, no target-specific codegen needed | **Yes for WASM** — the one environment where conservative native-stack scanning is unavailable |

Key framing from the LLVM docs: LLVM ships **no collector**; the framework exists to
make *accurate* (precise, possibly moving) collection possible. A conservative collector
like Boehm deliberately needs almost none of it. So the honest estimate is: **most of the
statepoint/stack-map machinery is not applicable today**, but four pieces are genuinely
useful, in increasing order of effort:

## 3. Applicability estimate

### 3.1 Custom no-op `GCStrategy` ("tsboehm") — low effort, foundational

Register a strategy and tag every generated function with it:

```cpp
// in a compiler-linked .cpp (e.g. next to transform.cpp's pipeline setup)
#include "llvm/IR/GCStrategy.h"
class TSBoehmGC : public llvm::GCStrategy {
public:
    TSBoehmGC() {
        UseStatepoints = false;
        UsesMetadata = false;      // no stack maps: Boehm scans conservatively
        NeededSafePoints = false;
    }
};
static llvm::GCRegistry::Add<TSBoehmGC> X("tsboehm", "Boehm conservative GC for tslang");
```

and after MLIR→LLVM-IR translation (in `tslang/tslang/transform.cpp`, where the
`PassBuilder` pipeline is set up):

```cpp
for (auto &F : *llvmModule)
    if (!F.isDeclaration())
        F.setGC("tsboehm");
```

Benefit: purely declarative today (default strategy behavior = lower intrinsics to
plain loads/stores, no safepoints, empty stack map), but it is the prerequisite for
every later step, makes GC-managed functions self-describing in the IR, and lets us
attach custom lowering later without touching the frontend again.

Estimated effort: **~1 day**, including plumbing the `setGC` loop behind a
`CompileOptions` flag.

### 3.2 `llvm.gcroot` pinning as mis-optimization insurance — low/medium effort

The one *correctness* risk of Boehm + `-O3` is pointer hiding. A targeted use of
`llvm.gcroot` (emitted from `LowerToLLVM.cpp` for reference-typed locals, or as an
LLVM IR pass over the module) forces each GC pointer to live in a real stack slot
that the conservative scan will see, across every call site. With the default
(no-op-lowering) strategy in 3.1 this costs a stack spill per rooted value and
nothing else — no runtime change at all, because Boehm finds the slots by scanning.

Trade-off: defeats `mem2reg`/register promotion for rooted values → measurable
slowdown in hot loops. Recommendation: implement it, but gate behind
`--gc-safe-roots` (off by default) and use it as a debugging/hardening tool for
crashes like the historical JIT statics collection bug rather than as an
always-on mode.

Estimated effort: **2–4 days** (choosing which MLIR values are reference-like is the
work; the mechanical part — `alloca` in entry block + `llvm.gcroot` call — is small).

### 3.3 `llvm.gcwrite` → Boehm incremental/generational dirty-marking — medium effort, perf feature

Boehm supports incremental & generational collection (`GC_enable_incremental()`),
normally driven by OS page protection (slow/fragile on Windows and unavailable under
some JIT memory managers). Boehm ≥ 8.x offers manual dirty-bit mode
(`MANUAL_VDB` / `GC_set_manual_vdb_allowed(1)`), where the mutator must call
`GC_ptr_store_and_dirty(dst, src)` (or `GC_end_stubborn_change`) after pointer stores.

This is exactly what `llvm.gcwrite` + a custom lowering pass is for:

1. Frontend emits `llvm.gcwrite(value, object, slot)` for every store of a reference
   into a heap object (we know these sites precisely in `LowerToLLVM.cpp` where
   property/element stores are lowered).
2. `TSBoehmGC` grows `CustomWriteBarriers = true` plus a lowering pass that rewrites
   each `gcwrite` to `store` + `call @GC_dirty(slot)` when incremental mode is on,
   or a plain `store` when it is not (one flag, zero cost in the default build).
3. Runtime wrapper gains `_mlir__GC_dirty`, plus init-time `GC_set_manual_vdb_allowed(1)`
   and `GC_enable_incremental()` in `gc.cpp`, exported through
   `TypeScriptRuntime.def` and `init_gcruntime` like the existing symbols.

Benefit: bounded pause times for large heaps (games/servers), and generational
collection speedup for allocation-heavy TS code. `llvm.gcread` is **not** needed —
Boehm's incremental mode requires no read barrier.

Estimated effort: **1–2 weeks** including tests (pause-time test, correctness under
`-O3`, JIT + AOT, and the shared-lib path via `TypeScriptRuntime.def`).

### 3.4 `shadow-stack` strategy for WASM — medium effort, unblocks a target

On `wasm32` there is no way to conservatively scan the WebAssembly value stack;
Boehm on WASM only sees the linear-memory "C stack", and misses locals held in wasm
locals/registers. LLVM's built-in `shadow-stack` strategy solves precisely this:
each function pushes a frame of its `llvm.gcroot` slots onto an explicit
`@llvm_gc_root_chain` list — fully portable, no codegen support needed.

Integration sketch (builds on 3.1 + 3.2's root emission):

- For the wasm target, `F.setGC("shadow-stack")` instead of `"tsboehm"`.
- Emit `llvm.gcroot` for all reference locals (the 3.2 machinery, non-optional here).
- Runtime: register a Boehm *custom roots* callback (`GC_set_push_other_roots`)
  that walks `llvm_gc_root_chain` and `GC_push_all`-es each frame's root array.

Estimated effort: **2–3 weeks**, dominated by wasm build/runtime plumbing rather
than the LLVM side. Only worth doing when the wasm target
(`prepare_3rdParty_wasm_debug.sh`) becomes a priority.

### 3.5 `gc.statepoint` / stack maps / `GCMetadataPrinter` — not applicable now

These exist to support **precise, moving** collectors (relocation of live references
across safepoints). Boehm can never consume relocations, and feeding it precise stack
maps buys little while the heap scan stays conservative. This tier only becomes
relevant if the project ever replaces Boehm with a precise collector (e.g. MMTk or a
custom semispace/generational GC). In that world the migration path is:
`PlaceSafepoints` + `RewriteStatepointsForGC` + a `UsesMetadata`/`UseStatepoints`
strategy + a `GCMetadataPrinter` emitting our stack-map format, and a new runtime.
Estimated effort: **multiple months** — record as long-term option only.

Note the LLVM docs' own caveat here: JIT support for the stack-map path is limited,
and this project relies heavily on the JIT (`jit.cpp` with custom section handling),
which further weighs against the statepoint route while on Boehm.

## 4. Summary matrix

| Option | LLVM facility | Effort | Benefit | Recommendation |
| --- | --- | --- | --- | --- |
| 3.1 `tsboehm` GCStrategy + `setGC` | `GCStrategy`, `gc` attr | ~1 day | Foundation; self-describing IR | Do first |
| 3.2 `gcroot` pinning (`--gc-safe-roots`) | `llvm.gcroot` | 2–4 days | Correctness insurance vs. pointer-hiding at `-O3`; debugging tool | Do, off by default |
| 3.3 Write barriers → incremental Boehm | `llvm.gcwrite` + custom lowering | 1–2 weeks | Bounded pauses, generational perf | Do when heap sizes justify it |
| 3.4 Shadow stack for WASM | `shadow-stack` strategy | 2–3 weeks | Makes GC correct on wasm at all | Tie to wasm-target milestone |
| 3.5 Statepoints / stack maps | `gc.statepoint`, `GCMetadataPrinter` | months | Only with a moving precise GC | Long-term note only |

Independent of the LLVM framework, heap-side precision is already largely in place:
with `ENABLE_TYPED_GC` (on by default in `Config.h`), class instances are allocated via
`GC_malloc_explicitly_typed` with cached per-class `GC_make_descriptor` descriptors, so
non-pointer class fields are excluded from the scan. Remaining conservative surfaces are
stacks/registers (addressed by 3.2/3.4), globals, and non-class allocations (arrays,
strings, closures — strings and other pointer-free data already use `GC_malloc_atomic`
where the frontend marks them).

## 5. Suggested implementation order

1. **Phase 0 (done)** — typed-allocation descriptors for class instances
   (`ENABLE_TYPED_GC`, on by default); possible extension: typed descriptors for
   non-class aggregates (tuples, closures, arrays of references).
2. **Phase 1** — 3.1 strategy registration + `setGC` plumbing behind a compile option.
3. **Phase 2** — 3.2 `llvm.gcroot` emission under `--gc-safe-roots`; use it to
   re-verify the historical JIT/`-O3` GC crashes.
4. **Phase 3** — 3.3 incremental-mode write barriers, benchmarked on the tester suite
   (watch the existing `gc_malloc_cse_o3.ts` style tests for optimizer interactions).
5. **Phase 4 (conditional)** — 3.4 shadow stack when wasm becomes active.
