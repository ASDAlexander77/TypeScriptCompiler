# Reference Counting as a Memory-Model Option

Status: **evaluation only, nothing implemented.** Written 2026-09-02 against `main`,
revised the same day after the framing correction below.

> **Framing.** RC is evaluated here as a **selectable memory model alongside GC**
> (`-mm=rc`), not as a replacement for it. GC stays the default. This is the right
> framing, and it changes the conclusion: the cycle problem stops being a blocker and
> becomes an opt-in tradeoff, and delivery can be incremental. It also introduces one
> problem a replacement never had — two models must coexist in one compiler and, worse,
> in one link.

## Verdict

**Viable as an option, and the option framing is what makes it viable.** Three things
follow, in priority order:

1. **The ABI decision must be made before any code is written.** It is the only decision
   here that cannot be retrofitted. See §4 — this is the new central risk and it did not
   exist under the replacement framing.
2. **The per-gap engineering cost does not go down.** Everything in §3 is still required
   in full for `-mm=rc` to work at all. What changes is who bears the risk, and that the
   work can ship in stages behind a flag instead of landing complete.
3. **The permanent cost is two memory models in MLIRGen forever.** Not a one-time build
   cost — a standing tax on every future language feature. That is the real thing to
   weigh, and it is a judgment about project capacity, not a technical blocker.

---

## 1. What the GC integration actually is today

This matters because the current design is what makes the option look cheap when it is not.

GC is wired in by **name substitution at the very end of the pipeline**. Nothing in
the IR, the type system, or MLIRGen knows a collector exists.

| Piece | Location | Role |
| --- | --- | --- |
| `GCPass` | `lib/TypeScript/GCPass.cpp` (274 lines) | Runs *after* `LowerToLLVMPass`. Renames `malloc`/`calloc`/`realloc`/`free`/`aligned_alloc` to `GC_malloc`/`GC_malloc_atomic`/`GC_realloc`/`GC_free`/`GC_memalign`; injects `GC_init()`; attaches `allockind("alloc")` so `-O3` GVN does not CSE two allocations into one; drops the `memset` after `GC_malloc`. |
| Allocation funnel | `LLVMCodeHelperBase.h:253` `_MemoryAlloc` | Every heap allocation in the compiler goes through here and emits a plain `malloc` call. **Twelve** call sites total. |
| Typed heap | `MLIRGenClasses.cpp:1322` `mlirGenClassTypeBitmap` | Emits a per-class pointer/non-pointer bitmap, cached in a global, fed to `GC_make_descriptor` / `GC_malloc_explicitly_typed`. |
| Pipeline order | `tslang/transform.cpp:161` | `GCPass` is the last pass, gated on `!disableGC`. |

The consequence for an *option*: GC's selectability is nearly free because GC needs no
program knowledge, so its entire branch point is one late pass. RC needs the most program
knowledge of anything in the compiler and must branch in MLIRGen, before lowering discards
type and ownership information. **The two models cannot be made selectable at the same
place in the pipeline.** That asymmetry is what the rest of this document is about.

### 1.1 The option mechanism already exists

`CompileOptions` (`include/TypeScript/DataStructs.h`) is a plain struct threaded through
MLIRGen, both lowering passes, and `GCPass`. `disableGC` already rides it end to end
(`opts.cpp:45` → `transform.cpp:161`). Adding `memoryModel` is mechanically identical.

One cleanup this should force: `-nogc` today means *leak everything* — `malloc` with no
`free`. With RC added there are three models, so the flag should become
`-mm={gc,rc,none}` with `-nogc` kept as an alias, rather than two independent booleans
that can contradict each other. **Done 2026-09-03 — see §9.6.**

## 2. What already exists to build on

Four assets. They are why a staged approach is viable rather than a standing start.

- **Pointer-layout metadata per class.** `mlirGenClassTypeBitmap` already computes which
  fields of a class are pointers, by generating code that takes field addresses off a null
  base. Today it produces a Boehm descriptor word; the same data is exactly what a
  recursive release routine needs.
- **A scope-exit walker.** `mlirGenDisposable` (`MLIRGenImpl.h:511`) already walks out of
  scopes calling `Symbol.dispose` on `using` variables, with `CurrentScope` / `LoopScope` /
  `FullStack` depth semantics for `break`/`continue`/`return`. That is the shape release
  calls need.
- **RC precedent in-pipeline.** `transform.cpp:142` already runs MLIR's
  `createAsyncRuntimeRefCountingPass()` (plus its `Opt` variant under `-O`). Liveness-based
  automatic RC on async values runs in this compiler today. The technique is proven here.
- **A test-runner pattern for option variants.** `-fast-math` tests already get their own
  cached script names (`jitfm` / `compilefm`) because the plain `jit`/`compile` scripts are
  shared across parallel single-file tests and embed the flag string at creation time
  (`test-runner.cpp:85-108`). `-mm=rc` reuses that pattern directly.

## 3. What RC requires that does not exist

Six gaps. **The option framing reduces none of them** — each is still required in full
before `-mm=rc` produces a correct program. Ordered roughly by cost.

### 3.1 Object headers (the ABI decision, see §4)

Nothing on the heap has a header. There is nowhere to put a count.

- `string` lowers to a bare pointer (`LowerToLLVM.cpp:6101`) and is handed **straight to
  libc**: `strlen` (`:485`, `:572`), `strcpy`/`strcat` (`:573`, `:574`), `strcmp`
  (`:635`, `:849`, `:895`, `:1004`), `puts` (`:164`). A header can live *before* the
  returned pointer so libc still works, but every release site must then recover
  `ptr - sizeof(header)`, and every pointer from elsewhere must not.
- `array` lowers to a by-value `{dataPtr, length}` struct (`LowerToLLVM.cpp:6115`).
- A class instance is a raw pointer to its storage struct, field 0 being the vtable.

### 3.2 Literal-versus-heap discrimination

String literals and const arrays are LLVM globals, not heap blocks, but they flow into the
same SSA values as heap results:

```ts
let s = cond ? "literal" : a + b;   // sometimes a global, sometimes heap
```

Releasing a global is a crash. Needs a saturating "immortal" count the globals also carry,
or a pointer tag. Boehm needs neither — it ignores addresses outside its heap.

### 3.3 An ownership model in MLIRGen

The bulk of the work, with no shortcut. Values are plain `mlir::Value` with no
owned/borrowed distinction. Retain and release decisions are needed at every assignment,
field store, element store, argument pass, return, capture, and box-into-`any`, across the
largest and most intricate part of the codebase.

**This is also where the two models permanently diverge.** GC mode needs none of it. Every
future language feature has to be correct under both.

> **Started 2026-09-03 (§9.12).** Locals now own what they hold. The "correct under both"
> tax turned out smaller than written here: ownership is stated once, in ops that erase under
> a collector, so MLIRGen carries no second model - only a second lowering does.

### 3.4 Type-erased release

`any` boxes as `{size, typeNamePtr, payload}` (`AnyLogic.h:48`) where the type tag is a
**type-name string**, under a standing `// TODO: add type id to track data type`. To
release an `any`'s payload you must know whether it holds a pointer and which routine
frees it. There is no id-to-release-function table. Tagged unions have the same problem.

> **Addressed 2026-09-03 (§9.3).** Every tag now points into a per-concrete-type descriptor
> carrying a kind id and a reserved release slot. The table step 4 needs has somewhere to
> live; nothing fills it in yet.

### 3.5 Cleanup landing pads

`ENABLE_EXCEPTIONS` is on. Every throw path must release the live owned values in each
frame it unwinds. Existing landing pads (`LowerToLLVM.cpp:4402`) do catch dispatch only.
This is the classic source of RC bugs that surface only under exceptions.

### 3.6 Interior references

`BoundRefType` lowers to `{ptr, ptr}` and `GetReferenceFromValue` hands out references to
object *fields*. An interior reference must keep its owner alive. Boehm handles this free
via interior-pointer scanning; RC would need those values widened to carry and retain the
owner.

## 4. The new central problem: two models in one link

> **Decided 2026-09-03: allow mixed links, treating what crosses as immortal** — leak rather
> than double-free, chosen over a hard error because an error forces a per-model default lib.
> The marker this needs landed in §9.7; the marking itself lands with ownership insertion.

This risk **does not exist under the replacement framing** and is the single most important
finding of the revision.

There are 72 cross-module tests (`import_*` / `export_*`), and heap objects cross module
boundaries in both directions: a consumer allocates instances of an imported class through
its own synthesized `.new`, while an exporting module's own code allocates objects the
consumer then holds and mutates. The declaration mechanism is source-text re-print and
re-parse (`declExports`, `MLIRGenImpl.h:11304`), so **each side compiles its own view under
its own `CompileOptions`.**

Nothing today prevents a GC-built shared library from being linked against an RC-built
consumer. If RC adds a header and GC does not, then:

- RC-side code computes `ptr - sizeof(header)` on an object a GC-built module allocated
  without one, and decrements whatever precedes it in the heap.
- GC-side code hands out objects that RC-side scope exits will release and free.

Both are **silent memory corruption**, not a link error. Given how much of this project's
recent history is cross-module work, this would be a persistent, hard-to-diagnose class of
bug.

There are exactly two acceptable answers, and the choice must be made before any code is
written because it is not retrofittable:

**(a) Emit the header in both modes — recommended.** GC builds pay one word per heap object
and ignore it. The ABI becomes uniform, mixed linking is safe, and the ABI change lands and
is tested *under GC*, where a wrong count is harmless. This also makes §3.1 a mode-neutral
change that can ship long before any RC semantics exist.

**(b) Forbid mixed linking and fail loudly.** Emit a memory-model marker into `declExports`
(it is text and re-parsed, so this is cheap) and additionally reference a mode-specific
sentinel symbol so a mismatch fails at link time rather than at runtime.

Doing neither is the worst outcome. (a) and (b) are not exclusive; (a) plus the marker from
(b) is the strongest position.

## 5. Cycles: a blocker under replacement, a documented tradeoff as an option

> **Decided 2026-09-03: weak references in the language, spelled `WeakRef<T>`.** Not
> leak-and-document. Representation settled in §9.8 — a weak count in front of the strong one,
> so `-mm=gc` builds keep their single header word.

Plain RC leaks cycles, and here the cycles are not exotic:

- **Recursive closures are a compiler-generated cycle.** Capture records are heap allocated
  (`LowerToAffineLoops.cpp:2106`, `ALLOC_CAPTURE_IN_HEAP`) and hold the captured values. A
  self-referential arrow function stores its own `HybridFunction` `{funcPtr, captureBoxPtr}`
  *into the very box that pointer names*. The compiler emits this, not an unusual program.
- Ordinary user cycles: `class Node { parent: Node; children: Node[] }`, a generator holding
  `this` while `this` holds the generator, mutually referencing objects.

Of the 453 tests in `test/tester/tests`: 159 use classes, 86 use arrow functions, 22 use
generators.

**As an option this is acceptable and has direct precedent.** Swift ships ARC as its only
model and leaks cycles by design, mitigated by `weak`/`unowned` and documentation. Here GC
remains the default, so a user selecting `-mm=rc` is making the same informed trade Swift
users make, and the safe model is one flag away. What this requires is honesty rather than
a solution:

- Document cycle leakage as a defined property of the mode, not a bug.
- Decide whether to add a weak-reference annotation. TypeScript has no surface syntax for
  it, so this is a language extension and should be a separate decision, not a prerequisite.
- A trial-deletion cycle collector (Bacon-Rajan) remains available later and is a second
  collector. As an *option* that is at least coherent, where under replacement it defeated
  the purpose.

*Not* a cycle, worth recording because it looks like one: object-literal method fields
store an **unbound** function pointer. `getEffectiveFunctionTypeForTupleField`
(`MLIRCodeLogic.h:158`) strips the bound-ness for storage, and `this` is re-bound at load
time (`LowerToLLVM.cpp:5157`). A method-bearing object does not hold a pointer to itself.

## 6. Standing costs of carrying two models

Distinct from build cost. These do not end when the feature ships.

| Cost | Detail |
| --- | --- |
| **MLIRGen carries two models** | Every future language feature must be correct under both, or explicitly unsupported under RC. Given this project's cadence of interface/generator/cross-module fixes, this compounds indefinitely. |
| **Test matrix roughly doubles** | 453 tests, for whatever subset RC claims to support. The `jitfm`/`compilefm` pattern (`test-runner.cpp:85-108`) extends to `jitrc`/`compilerc`, so the mechanism exists; the CI time is the cost. |
| **Mixed-link surface** | Permanent, per §4, unless the uniform-header answer is taken. |
| **Flag surface** | `-mm={gc,rc,none}` must be coherent across JIT, executable, DLL and shared-import paths, all of which read `CompileOptions` independently. |

## 7. Performance is not the argument

The honest case for RC is **determinism and memory footprint**, not throughput. A naive
implementation puts a retain/release on every array fat-pointer copy and every string
assignment — and strings are the highest-churn allocation in the compiler, since every
concat and every number-to-string allocates. Non-atomic counts are cheap, but
`ENABLE_ASYNC` is on, so any value crossing a coroutine suspension point needs atomics or a
thread-confinement proof that does not exist today.

## 8. Cost by tier

Tiers A and B are mode-neutral and improve the GC default immediately. C and D are the RC
option proper.

| Tier | Scope | Size | Effect |
| --- | --- | --- | --- |
| A | Escape analysis: promote non-escaping `MemoryAlloc` to `alloca` | small | Pure win, both modes |
| H | Uniform object header in both modes (§4a) | medium | Mode-neutral; unblocks everything below |
| B | Extend `mlirGenDisposable` to free provably scope-bound temporaries | small | Pure win |
| C | `-mm=rc` supporting `string` only, other types still GC-allocated | medium | Shippable increment |
| D | `-mm=rc` across the heap | multi-month | Leaks cycles, by documented design |
| D+ | D plus a cycle collector | D plus a second collector | Parity with GC |

Tier A is worth more under RC than under GC: every heap object elided is retain/release
traffic elided, not merely collector pressure.

## 9. Recommended order

The ordering point: **steps 1-4 are useful on their own, land under GC where mistakes are
harmless, and commit to nothing.** Step 5 is the commitment.

1. **Settle the ABI question (§4)** and write it down. Nothing else should start first.
2. **Uniform object header in both modes**, GC still running. A wrong count is inert here,
   which makes the widest-blast-radius change the safest to land and the easiest to test.
   Add the memory-model marker to `declExports` at the same time. **Split this in two — see
   §9.1, the halves are not equally hard.**

### 9.1 The header has two allocation paths, and only one is easy

This is the detail that decides how big step 2 is.

> **Status: path 1 landed 2026-09-02 and the full release suite passes.** The header is
> reserved in `_MemoryAlloc` / `_MemoryRealloc` / `_MemoryFree` and the word is never read.
> All three helpers were genuinely exercised: 19 tests drive array `push`/`splice`/`unshift`
> through realloc, 37 exercise string allocation and `SetStringLength`, and 9 use `delete`
> (including `00new_delete.ts`) through free. **The provenance worry did not materialise** —
> nothing reaches realloc or free holding a pointer that did not come from the allocator, so
> the base adjustment is safe in practice, not merely in principle. The 72 cross-module tests
> pass, which is the result that matters most for §4.
>
> Still unvalidated by this run: the WASM allocator path (`ts_malloc`/`ts_realloc`/`ts_free`),
> which is built and tested separately.
>
> **Status: path 2 resolved 2026-09-03 by removing it, not by shifting it.** Investigating the
> descriptor shift showed the bitmap it would shift was never correct, so `ENABLE_TYPED_GC` is
> now `false` and class instances take the generic path like everything else. See §9.2.

### 9.2 The typed path was retired rather than adapted

The plan in §9.1 was to shift every descriptor bit by the header size. Reading
`mlirGenClassTypeBitmap` (`MLIRGenClasses.cpp:1322`) first showed there was nothing sound to
shift. Three defects, each confirmed against the code:

1. **Shift direction inverted.** Line 1427 passes `GreaterThanGreaterThanToken`, which maps to
   `rightShift` (`MLIRGenImpl.h:4996`), sitting directly under a comment reading
   `// 1 << index_mod`. `1 >> bitIndex` is zero for every bit position but zero, so no bit
   above position zero could ever be set.
2. **Wrong array index.** Line 1412 indexes the bitmap with `calcIndex`, the field's word index
   *within the object*, where it needs `calcIndex / bitsPerWord`. The array holds only
   `ceil(N/64)` elements, so any class with pointer fields past word zero also read and wrote
   past the end of the stack allocation.
3. **Never zeroed.** `AllocaOpLowering` emits a bare alloca under an explicit
   `// TODO: call MemSet` (`LowerToLLVM.cpp:2223`), and the generator only ORs bits in. The
   descriptor was therefore derived from uninitialized stack memory.

Net effect: `GC_make_descriptor` received a garbage bitmap, and any pointer field whose bit
read as zero went untraced, so a reachable object could be collected. Latent because short
tests seldom trigger a collection cycle.

Since the precision was fictitious, the cheaper and safer resolution was to stop using the
typed path rather than repair and then shift it. Class instances now lower through
`NewOp` → `NewOpLowering` → `MemoryAlloc` (`LowerToLLVM.cpp:2261`), which means they are
conservatively scanned — what they effectively got anyway — and they pick up the block header
uniformly, which is what §4 needed. The `#else` branches for this already existed; only the
`Config.h` flag changed.

This closes the ABI question: **every heap allocation now carries the header.** The bitmap
generator's three defects remain in the tree, unused and documented, and are worth their own
fix if precise class scanning is ever wanted back.

**Path 1 — the generic helpers (easy).** `_MemoryAlloc`, `_MemoryRealloc` and `_MemoryFree`
all live in `LLVMCodeHelperBase.h:253/300/330`. Eleven of the twelve allocation sites route
through them, and so does the single `free` site (`DeleteOpLowering`, `LowerToLLVM.cpp:3022`,
the `delete` operator). Prepending a word means: allocate `size + H` and return `ptr + H`;
pass `ptr - H` on realloc and free. **Everything else is unaffected**, because every other
consumer — `strlen`, `strcpy`, `strcat`, `strcmp`, GEPs, the array `{ptr,len}` pair — operates
on the payload pointer and never sees the block base. Under GC the word is never read, so the
change is inert and the existing suite is a complete oracle. This is one file and three
functions.

**Path 2 — the typed-GC class path (the hard half).** `GCNewExplicitlyTypedOpLowering`
(`LowerToLLVM.cpp:5879`) does **not** go through those helpers. It calls
`GC_malloc_explicitly_typed(sizeof(storageType), typeDescr)` directly. And the descriptor
collides with a header: `mlirGenClassTypeBitmap` computes each bit index as *field address off
a null base, divided by word size* (`MLIRGenClasses.cpp:1400-1420`), so bit positions are
**object-base-relative**. Prepend a header and the object base no longer coincides with the
block base Boehm scans, so every bit in the descriptor is off by `H/wordsize`. Boehm then
traces the wrong words — silent false retention or, worse, premature collection of live
objects, under the *default* configuration.

So path 2 requires shifting every bitmap bit by the header size and reserving the leading
word as non-pointer, and it perturbs machinery that is live and load-bearing today. Land
path 1 first and alone; treat path 2 as its own change with its own verification.
3. **Real type ids in `any`/union boxes**, replacing the type-name string (§3.4).
   Independently useful — `any` comparison already pays for stringly-typed tags.
   **Done 2026-09-03, see §9.3.**
4. **Generate per-type release routines** from the existing bitmap machinery, initially
   unreferenced and verifiable in isolation. **Done 2026-09-03, see §9.4** — built fresh
   rather than from the bitmap machinery, which §9.2 had already retired as unsound.
4a. **Give static string literals the block header, with an immortal marker.** Inserted by
   §9.4's finding: a `string` field can hold a pointer into a read-only global, so releasing
   strings is impossible until heap and static strings are distinguishable.
   **Done 2026-09-03, see §9.5** — which also closed a second hole, `typeof` results pointing
   into descriptors.
4b. **`-mm={gc,rc,none}`, and maintain the count.** The flag step 5 hangs off, plus
   initialising the header at allocation and turning §9.4 destroy routines into real
   reference drops. Still inert. **Done 2026-09-03, see §9.6.**
4c. **Memory-model marker in `declExports`.** §4's last outstanding piece, and a prerequisite
   for marking foreign objects immortal. **Done 2026-09-03, see §9.7.**
4d. **`WeakRef<T>` representation.** Settled on paper before any code, because the header
   layout it implies is ABI. **Designed 2026-09-03, see §9.8; not implemented.**
4e. **`ts.Retain` / `ts.Release` in the dialect**, with retain routines to match the release
   ones, so that ownership can be *stated* before deciding where. Still inert — nothing emits
   them. **Done 2026-09-03, see §9.10.**
4f. **`using` disposes on the unwind path, for the case ownership tracking will lean on.**
   A prerequisite for step 5, not step 5 itself: confirms the scope-exit machinery ownership
   insertion will reuse actually runs on `throw`, for at least one real shape. Narrowly scoped
   after surfacing several independent pre-existing gaps in the same machinery. **Done
   2026-09-03, see §9.11.**
5. **Ownership tracking in MLIRGen behind `-mm=rc`**, checked by a verifier that flags any
   owned value without a matching release on every path, unwind paths included. **Verifier done
   2026-09-04, see §9.18.** *Point of
   no return* — and the first step where a mistake is not inert: a missing retain frees live
   memory, an extra one leaks. Narrowed by §9.10: the mistake can only reach `-mm=rc`.
5a. **Locals own what they hold.** The first slice of step 5 and the one that builds the
   mechanism the rest reuses. Deliberately balanced by construction, so it cannot
   over-release. **Done 2026-09-03, see §9.12.**
5b. **Owned storage is hoisted out of the `TryOp`, and the unwind leg releases.** **Done
   2026-09-04, see §9.15.**
5c. **Fields own what they hold.** The first insertion point beyond locals, and the first the
   verifier guarded rather than followed. **Done 2026-09-04, see §9.19.**
5d. **Elements own what they hold.** `arr[i] = x`, the direct sibling of 5c. Exposed the first
   *over*-release: an array literal stores its elements without retaining them.
   **Done 2026-09-04, see §9.20.**
5e. **Literal construction retains what it captures.** Array literals and boxed object literals,
   which fill an owning block in one go rather than through an assignment. Taken ahead of the
   rest because it turned out not to be latent at all — two holders and two overwrites freed a
   live value on the compiler as it stood. **Done 2026-09-04, see §9.21.** The spread form
   (`[...xs, y]`) goes through `ts.ArrayPush` and waits for 5f; the unboxed object literal is the
   inline-record case and waits for 5g.
5f. **The array-mutating ops.** `push`/`unshift`/`splice` now take a reference; `pop`/`shift`
   correctly need none — the block transfers its reference to the result rather than releasing
   it, which is the existing "+1 nobody consumed" convention. Also closes 5e's spread-literal
   hole, since `[...xs]` is built out of `push`. **Done 2026-09-04, see §9.22.** What `splice`
   *deletes* still drops references without releasing them (a leak, and the first item needing
   emission from `LowerToLLVM` rather than MLIRGen).
5g. **Inline records** — an assignment through a field of a record held inline now retains and
   releases, conditionally on the storage under it owning. Arguments turned out to need nothing
   (a parameter's slot is not owned, so they are already borrowed at +0) and returns likewise
   (the scope-exit release balances the declaration's retain, and the caller receives the birth
   reference). 5e's unboxed object literal was never broken either — construction balances
   through the owned local's `RetainSlot`. **Done 2026-09-04, see §9.23.**
5h. **Remove 5a's slack**: consume a freshly allocated value's birth reference. The point where a
   mistake stops being an inert leak, and where every test written since 5a gains teeth.
   **First half done 2026-09-04, see §9.24**: allocations are born unowned, and a `return`
   retains its value so the scope exit cannot free it on the way out — which makes the
   convention uniform at *every function returns +1*. A real removal for arrays, strings and
   boxed object literals; still neutral for class instances, because `new C()` is a call to
   `C..new` and that return retain hands back the same +1 the birth reference did (verified by
   the release-before-retain swap, not assumed).
5i. **Consume the +1 at the receiving sites** — all four (declaration, store, literal capture,
   push) now take an already-owned value over instead of retaining it again. **Done 2026-09-04,
   see §9.25.** Only `new C()` is marked as producing one, at the site that knows the callee is
   the generated `C..new`; nothing is inferred from an operation being a call, because a runtime
   helper or a pre-convention import returns a heap value with no retain behind it and consuming
   one of those frees live memory. **`let x = new C()` is now genuinely freed, and the
   release-before-retain swap finally fails three of the ownership tests** — the experiment
   §9.19 asked to be re-run once the slack went.
5j. **`pop`/`shift` transfers consumed** — the compiler's own operations, so nothing to
   classify. **Done 2026-09-04, see §9.26.**
5k. **An ordinary call's result** (`let y = f()`) — done via a module pass after MLIRGen that
   inspects each function's returns instead of predicting them. **Done 2026-09-04, see §9.27.**
   469 call sites marked across the suite.
5m. **Retain the value a return actually returns.** The retain landed on the value the return
   statement evaluated, but `mlirGenReturnValue` then cast it to the declared return type, so a
   return needing a cast retained the wrong value. **Done 2026-09-04, see §9.31**, where it had
   to be: with a cast that allocates - a literal returned as an interface - the block the caller
   receives got no reference at all, so this was not only the leak it looked like.
5l. **Discarded temporaries** — `f();`, and every call result used as an argument without being
   bound to anything, which is what expression-shaped code is made of. **Done 2026-09-04, see
   §9.30**: consumption is recorded explicitly now, and what nothing consumed is released at the
   end of the block that produced it. Closing it also closed a §9.27 gap that kept it from firing
   at all — `return new C()` forwards a reference rather than retaining one, so those functions
   were never classified as returning owned. `raytrace.ts` 129.5 MB → 79.3 MB, below `none` for
   the first time; the nested-call shape on its own is flat.
5n. **An interface owns what it was made from.** **Done 2026-09-04, see §9.31.** The type
   carries only a name, so the value now carries the runtime type tag of its `this` and releases
   through the concrete type's own routines, exactly as an `any` box does. Both interface shapes
   are flat: a boxed literal bound in a loop, and one passed as an argument and dropped. Making
   an interface an owner turned two dormant omissions into live over-releases - a boxed literal
   that never retained its fields, and a pushed owned result never marked consumed - so
   `raytrace`'s figure went **up**, 79.3 MB to 114.2: part of §9.30's number was memory freed
   while still referenced.
5o. **Classify an instance method's callee.** **Investigated 2026-09-04, mostly NOT done, see
   §9.32.** `calleeNameOf` now looks through the bound-function chains the dialect's own
   canonicalizer already resolves, which is safe and worth almost nothing: every non-virtual
   method reference in `raytrace` is a constructor. The value is all in virtual dispatch, and
   `private` does not make that single-target here - this compiler accepts a subclass
   redeclaring a private method and dispatches to the override, where TypeScript rejects the
   program. Doing it properly needs the callee's override set, which the pass cannot see and
   MLIRGen cannot close cross-module, and it buys 2.6% of `raytrace`. Left open deliberately.
5p. **A closure owns its capture box.** **Done 2026-09-04, see §9.33.** A bound or hybrid
   function value carries the tag of its `this` beside the pointer, as an interface does, and
   only a closure over captured variables is marked as owning it - a bound method must not take
   ownership of its receiver. Measuring it turned up the larger of the two: **§9.30's releases
   were being placed after `ts.ReturnVal`, which is not a terminator, and the affine lowering was
   dropping them** - 36 of `raytrace`'s 47. `raytrace` 113.8 -> 103.4 MB.
5q. **A captured variable's cell is owned, not borrowed.** **Done 2026-09-04, see §9.34.** A
   variable a closure captures by reference gets a heap block of its own - a *cell* - and it had
   no owner at all: the frame released the *value* at scope exit while the box still pointed at
   the cell, and the cell itself was never freed. The cell now has owners - the frame that
   declared the variable, and each box that captured it - and the box has routines of its own
   that give the cells back. Two further over-releases fell out of writing the cases: a `const`
   captured by value was not retained by the box, and assigning to a captured variable *from
   inside the closure* stored a value nothing had taken, which §9.30 then freed as a discarded
   temporary. `raytrace` 103.4 -> **2.6 MB**, below `gc`'s 4.2.
5r. **A cell is given back whether or not the frame owns what is in it.** **Done 2026-09-05, see
   §9.35.** The frame released a cell only where §9.34's list already held the slot, and that
   list is the list of locals whose *value* the frame owns - so a captured parameter, whose
   argument belongs to the caller, and a captured local of a type that owns nothing, such as a
   `number`, both leaked the cell. Every local and every parameter is now listed, and scope exit
   asks each what it turned out to be; and a cell takes a reference to the value it is
   initialised with unless the frame already took one, which is what makes it safe for the
   cell's release to release its contents whoever put them there. Closure over a parameter at
   `-O0`: **22.6 -> 3.7 MB**, against `gc`'s 2.6 and `none`'s 22.0; closure over a `number`
   local, 51.7 -> 3.7 against `none`'s 113.0.
5s. **Boxing into `any` takes no reference to what the box then owns.** **Done 2026-09-05, see
   §9.36.** An `any` releases its payload through the type tag when the box dies, and nothing had
   taken that reference: the payload was copied in and, where it arrived carrying one of its own
   - a closure, whose reference is the box of captured variables it was built over - §9.30 gave
   that reference back at the end of the block and left the `any` pointing at freed memory. This
   was `22lambdas.ts`'s intermittent failure under `rc`, one run in four, which is what a
   use-after-free looks like from outside. The boxing cast now consumes or retains like every
   other owning receiver. A second bug fell out of the corpus sweep and belongs to 5r: a captured
   declaration with **no initializer** has a cell holding whatever the allocator last left there,
   and scope exit releases it - cells are now zeroed at birth, as owned locals already were.
5t. **A freshly built string carries no reference for its receiver.** **Done 2026-09-05, see
   §9.37.** Printing a number into a string allocates one, and so does concatenating; neither
   said so, so a receiver added a reference of its own - balanced, and it worked - while an
   intermediate no receiver ever took was left with none and leaked. `"s" + k` leaks twice over:
   the conversion's result feeds the concatenation and is then forgotten. Both now hand back a
   reference the way `new` and a call do. `let s = "s" + k` at `-O0`, 500k iterations:
   **34.5 -> 3.7 MB**, below `gc`'s 4.1; and the `any`-boxing benchmark, whose remaining leak
   this turned out to be, 39.9 -> 3.7. A second bug came out of the same measurements and is
   older than any of this: **growing an array does not zero the slots it exposes**, and the
   store into one releases what the slot held - which is `Array.map` in the default library, and
   was `arrS.map(e => e + "_")` crashing 6 runs in 20 under `rc`.
5u. **Read the `-mm=rc` corpus sweep through.** **Done 2026-09-05, see §9.38.** Of the 117
   non-zero exits, **101 fail identically under `gc`** - 79 are one half of a two-file
   `export_*`/`import_*` test that cannot be JITed alone, 20 are the default library colliding
   with a test written for `--no-default-lib`, 2 fail under both models. The sweep had also
   never run the configuration the suite runs (`--opt --opt_level=3 --no-default-lib`), which is
   what manufactured those 20 and hid 13 real ones. Sweeping both configurations under both
   models leaves **29 files that fault under `rc` and pass under `gc`**, of which **exactly one
   is a file ctest ever runs under `rc`**. **18 of the 29 fault in only some of the four
   `{-O0,-O3} x {default library, none}` configurations**, which is what a use-after-free looks
   like from outside: the columns vary the heap layout, not the ownership. Three reductions came
   out of it, and they are the next three slices.
5v. **A generator that iterates another generator faults at `-O3`.** `function* g() { for (const
   o of inner()) yield o; }` faults before printing anything, `rc` only, 10 runs in 10, with or
   without the default library; `yield* inner()` is the same bug, one generator alone is fine,
   and `-O0` is fine. Covers `00generator4/5/6`, `00funcs_expression_iterator`, `01iterator`,
   `00iterator_bug`. **Next slice** - the largest cluster, and it fails in the configuration the
   suite ships.
5w. **A `for...of` loop variable that holds a reference is not retained.** `const a = [[1],[2]];
   for (const v of a) print(v.length)` faults with the default library at either optimisation
   level, `rc` only, 10 runs in 10. A single-level `for...of` is fine and `a.length` on the same
   array is fine, so it is the loop variable taking an element that owns something without
   taking a reference to it. `--no-default-lib` hides it by routing array `for...of` through the
   built-in intrinsic loop instead of the library's iterator protocol - which is also why
   nothing has ever caught it. Covers `00array3`, `00for_of`, `19forof`, `01map`,
   `00tuple_with_array`, `arrayLiterals`.
5x. **`await` frees something twice.** `async function f() { return 1; }` and then `const r =
   await f()` prints the right answer and dies of heap corruption on the way out, in all four
   configurations, 10 runs in 10. Calling `f()` without awaiting it is fine. Covers
   `00async_await` and `00for_await`. The coroutine frame is the obvious suspect, and nothing in
   this work has ever looked at one.
5y. **The `rc` tier of ctest is 39 files of 475.** Every ownership measurement in §9.12-§9.37 was
   taken inside that 39, and 28 of the 29 known faults are outside it. Once 5v-5x are closed,
   registering the rest of the corpus under `rc` - as individual tests or as one sweep target -
   is what stops this recurring.
6. **Flip the allocator under the flag.** **Done 2026-09-04, see §9.28.** `needsGCRuntime()` now
   names only `gc`; `rc` allocates from `malloc`, frees through `free` and links no libgc, so a
   memory measurement under it finally means something — a million-iteration allocation loop stays
   flat at 3.8 MB where `none` reaches 172.8 MB. Flushed out a Win64 crash that was never RC's:
   allocating inside a catch funclet, latent under `-mm=none` since long before this work.

**Scope the first shipping mode narrowly.** Two candidates, and they are compatible:

- **`string` only (Tier C).** Strings are leaves — a string never points to another heap
  object, so release is a single free with no recursive traversal and **no cycle is
  representable**. Strings are also the highest allocation-rate type. Highest benefit, zero
  cycle risk, bounded blast radius. Step 4a (§9.5) cleared the static-string blocker; what is
  left before this scope is reachable is maintaining the count itself.
- **WASM target.** The strongest driver for RC existing at all. WASM is the one environment
  where conservative native-stack scanning is unavailable, which is the assumption Boehm
  rests on (`docs/llvm-gc-integration.md`), and the compiler already forks its allocation
  path there (`ts_malloc`/`ts_realloc`/`ts_free`, `LLVMCodeHelperBase.h:265/312/340`, patched
  back by `MemAllocFixPass.cpp`). Scoping the first RC mode to WASM rides a split that
  already exists.

The other drivers that would justify Tier D: hard real-time latency budgets, and shipping
without a runtime dependency on libgc.

### 9.3 Step 3: the tag now points into a per-type descriptor

Landed 2026-09-03, full release suite green (829/829, 106 of them cross-module).

The obstacle was that the tag is not merely *a* string, it is the **`typeof` result itself**:
`GetTypeInfoFromUnionOp` returns it straight to `typeof`, `MLIRGenImpl.h:3895` `strcmp`s it
against `"class"` to implement `instanceof` over `any`, and the generated union operator
helpers compare `typeof(r) == "class"` in source text. Anything that stops the tag being a
readable `char*` breaks all three.

So the tag stays a `char*` and the record moves in front of it. Each distinct type gets one
static global `{ { i32 kind, i32 reserved, ptr release }, [N x i8] name }`, and the tag is
the address of `name`. Every existing consumer keeps reading a NUL-terminated type name and
is untouched; anything wanting the record takes `tag - sizeof(record)`, which the emitted IR
constant-folds to `getelementptr i8, ptr @td_..., i64 -16`. The trailing name is a byte
array, so nothing is padded in front of it and that offset equals the record size on every
target. This is the same header-in-front-of-payload shape as step 2, deliberately.

Three consequences worth recording:

- **The descriptor is keyed by the concrete type, not by the name.** Two classes both report
  `"class"` and now get two records — which is the entire point, since step 4 needs somewhere
  per-class to hang a release routine, and the name erases exactly that distinction.
  `typeOfBaseType` strips the wrappers `typeOfAsString` already sees through, so all the
  string literal types still share one `"string"` record rather than minting their own.
- **`TYPE_DESCR_*` is a cross-module contract**, even though every record has internal
  linkage. A tag produced by one module is read back by another, so the reader applies *its*
  idea of the record size to *the producer's* record. Same §4 hazard as the heap header, same
  answer: pin the layout once. The release slot is reserved now for that reason, not because
  anything calls it.
- **`any` comparison stopped paying for stringly-typed tags.** Asking "is this operand
  numeric" ran nine `strcmp`s per operand, because `typeOfAsString` reports `"s32"`/`"f64"`
  and not `"number"` for anything but a float. It is now one load and one compare against
  `TYPE_KIND_NUMBER`, and it covers every numeric width instead of the nine that happened to
  be listed. The width dispatch in `unboxNumericAsF64` stays name-based on purpose: the kind
  says *numeric*, and it is the width that decides how many bytes to read back.

### 9.4 Step 4: per-type release routines

Landed 2026-09-03, full release suite green (829/829). Nothing calls them; the only reference
is the descriptor slot from §9.3, which is also what keeps them from being dead-stripped.

The doc originally said "from the existing bitmap machinery". That machinery turned out to be
the unsound generator §9.2 retired, so this is built fresh — and built the other way round.
The old bitmap was *computed at run time*, with shifts and ORs into a stack array, which is
the root of all three of its defects. The pointer layout of a type is knowable at compile
time, so the routines are emitted as straight-line code with the offsets baked in.

**Calling convention:** a routine takes a pointer to the *storage holding* a value, not the
value. That is uniform across value categories — a class field, an `any` payload slot and a
local all address the same way — and it makes releasing a field a plain GEP plus a call.

**What each shape does:**

| type | owns | routine |
| --- | --- | --- |
| `string` | its own block | null check, free |
| `array<E>` | data block + elements | loop `0..length` calling E's routine, then free data |
| class / object | the instance block | release storage fields, free instance |
| `any` | its own box | read the tag's descriptor, call *its* release on the payload slot, free box |
| tagged union | nothing (payload inline) | same descriptor dispatch, no free |
| `optional<T>` | nothing | release the value slot when the flag is set |
| tuple, class storage | nothing | release the fields, free nothing |

The `any` and union rows are the payoff of §9.3: a value whose type is known only at run time
still resolves to a release routine, through the tag.

Recursion works because the symbol is created before its body: `class Node { next: Node }`
emits a routine that calls itself. That also means a cyclic *object* graph would recurse
forever, which is the cycle problem of §5 showing up in concrete form rather than a new one.

**Deliberately not released**, each for a stated reason: `InterfaceType` carries only a name,
so the layout behind its `this` pointer is not recoverable from the type and needs an RTTI
lookup rather than a static walk — *this one changed at §9.31: the value now carries the runtime
type tag of its `this`, which makes the lookup a tag read, so an interface is an owner*;
function types do not mention their capture box, so there is
nothing to walk even though the box is heap-allocated; `RefType`/`ValueRefType` point at
storage the value does not own; `ConstArrayType` and `ConstTupleType` are static data. A null
release slot says "nothing to release" positively — it is not an "unknown".

#### The finding: static strings block releasing strings

Writing the string routine surfaced a prerequisite that reorders the plan. A string literal
compiles to `store ptr @s_..., ...` — a `string` field can hold a pointer directly into a
read-only global that no allocator produced. `free(@s_... - headerSize)` corrupts the heap.

This lands squarely on §9's recommended first shipping scope, which is **strings only**,
chosen because strings are leaves with no representable cycle. That scope is not reachable
until heap strings and static strings are distinguishable at run time.

The consistent answer is the same one used twice already: give static string globals the same
block header, with an immortal marker in the count, so `__tslang_free_block` can test it and
skip. Every heap string already has that header from step 1, and every string pointer is
already `&bytes` of something — this only changes what precedes those bytes. It is deliberately
*not* part of this change: it touches every string literal in every module, on a hot path, and
deserves its own verification.

So the order from here is: **static-string immortality first, then ownership tracking (step 5)** —
not straight to step 5 as originally written.

**Cost note.** These routines are emitted under GC, where they are pure dead weight, so that
their construction and module verification are exercised by every test. That is the "verifiable
in isolation" the plan asked for, paid for in a few small internal functions per module.

### 9.5 Step 4a: static blocks carry the header too

Landed 2026-09-03, full release suite green (829/829). This is the prerequisite §9.4 turned up,
done straight away because it changes the layout of globals and so cannot be retrofitted.

A string literal compiles to `store ptr @s_..., ...`. Before this, a `string` value was two
different shapes depending on where it came from — a heap payload with a header in front, or a
raw pointer into a read-only global — and nothing at run time could tell them apart. Every
global string now carries the same header word as a heap block, set to `HEAP_BLOCK_IMMORTAL`,
and `__tslang_free_block` skips a block that says it is immortal.

The encoding is all-ones bytes, so the marker reads as `-1` whatever the word size or
endianness, and it stays a plain `[N x i8]` global with a `StringAttr` initializer — no
initializer region, and the existing `seekLast<StringAttr>` placement still works. The global
is aligned to the header size so the word can be read as a word. Deliberately not zero: a
zeroed word is what a fresh heap block reads.

**Every global string gets it, not just the ones that could be released.** Deciding
per-call-site which `getOrCreateGlobalString` produces a TypeScript string — as opposed to a
printf format or a symbol name — would be an audit whose failure mode is silent corruption in
exchange for saving eight bytes per constant. Uniformity is the same call made in step 2, for
the same reason. The `"true"`/`"false"` globals from a boolean cast are a good example of a
site that is not obviously a string value but is one.

#### The tag was the second hole

`typeof x` returns a pointer into a type descriptor, and `let s: string = typeof x` is ordinary
TypeScript — so a tag is a string value that can be released like any other. The descriptor's
name had nothing in front of it but the `release` field, which would have read as a very
mortal-looking count.

The record therefore ends with the block header, immediately before the name:
`{ i32 kind, i32 reserved, ptr release, index blockHeader }`. Both reads now work off the same
pointer — `tag - sizeof(header)` is the immortal marker, `tag - sizeof(record)` the record —
and a tag is simultaneously a descriptor's name and a well-formed immortal payload.

#### What this does not do

Nothing writes the header on allocation, because nothing maintains a count yet. So the immortal
test is meaningful for static blocks, where the marker is baked into the initializer, and says
nothing useful about a heap block, whose word is whatever the allocator left. That half belongs
with maintaining the count, in steps 5 and 6 — the static half is separated out here only
because it is the half that changes an ABI.

### 9.6 Step 5, part one: the flag exists and the count is maintained

Landed 2026-09-03, 847/847 green — the suite plus 17 new `-mm=rc` variants and one `-mm=none`.
**This is not step 5.** Step 5 is ownership tracking in MLIRGen, and it is still the point of no
return; what this does is build the two things step 5 needs to exist first, both of which are
still inert.

**`-mm={gc,rc,none}` replaces `-nogc`.** The flag cleanup this document has called for since the
first draft: there were always three models — `-nogc` meant "leak everything", not "collect
differently" — spelled as a single boolean. `-nogc` stays as a deprecated alias for `-mm=none`,
and `CompileOptions` grew `needsGCRuntime()` and `isRefCounted()` so no caller reads the model
enum directly.

`-mm=rc` at this point means *counts are maintained and the release machinery is generated*; the
collector still runs and is still what frees. That is deliberately an intermediate: it makes the
header word real without anything depending on it being right. It held until step 6 (§9.28), which
took the collector out from under `rc` entirely.

**Allocation initialises the count.** `_MemoryAlloc` stores 1 into the block header, after any
memset, so a block starts owned by exactly the reference being returned. **Only under
`-mm=rc`** — under `gc` nothing reads the word, and a store per allocation on the hot path is not
worth paying for dead code. Confirmed in the emitted IR: zero such stores under `gc`, one per
allocation site under `rc`.

**The generated routines became real releases.** §9.4's routines destroyed unconditionally,
which is a destructor, not a release. Each one now drops a reference and only destroys when it
was the last:

```
if (p != null && __tslang_dec_ref(p)) { release fields; __tslang_free_block(p); }
```

`__tslang_dec_ref` is where the immortal marker from §9.5 does its work: an immortal block is
neither decremented nor ever the last, so a string literal and a `typeof` result survive being
released like any other string, without a write to read-only memory. `__tslang_free_block` is
now a plain free, since it is only reachable behind that test.

The routines are reference-counting shaped in *every* model, because they are dead code in all
but `rc` and one shape is simpler than two. Only `rc` initialises the count they read.

**Coverage.** The test runner gained the `-mm=` variant alongside `-fast-math`, using the same
per-variant cached-script trick, and 17 tests now run under `-mm=rc`: strings, arrays and their
elements, `any`, tagged unions, tuples, classes, interfaces, generators, closures, `delete`, and
unwind paths. These prove the model compiles and runs correctly across the shapes the routines
walk — **not** that counting is correct, which nothing yet exercises. `-mm=none` also picked up
its first test ever, since the old `-nogc` had none and the rename would otherwise have been
unguarded.

**What is still ahead of step 5 proper.** Nothing calls a release, and nothing retains. Adding
those is the ownership tracking, and it is where a mistake stops being inert: a missing retain
frees live memory, an extra one leaks. That still wants the verifier the plan describes — every
owned value with a matching release on every path, unwind paths included — built alongside it
rather than after.

> **Superseded in part 2026-09-03 (§9.12).** Locals retain and release. The verifier is still
> outstanding, and so is everything that is not a local: fields, elements, arguments, returns
> and temporaries.

### 9.7 The memory-model marker

Landed 2026-09-03, 847/847 green. The last outstanding piece of §4.

A shared library records the model it was built under as an exported data symbol
`__tsmm_<model>_<file>_<hash>`. The model is in the **name**, so an importer reads it during the
symbol enumeration it already performs and never loads the data. Deliberately *not*
`__decls`-prefixed, so it can never reach the declaration re-parser — see
`decls-cross-module-declaration-mechanism` for why that enumeration is prefix-driven. A library
with no marker predates this, and everything collected back then, so a missing marker reads as
`gc`.

Both spellings come from one `memoryModelName()`, so the `-mm=` flag and the marker cannot
disagree about what a model is called.

Verified end to end: a DLL built `-mm=gc` carries `__tsmm_gc_export_vars_<hash>`; importing it
`-mm=gc` is silent, importing it `-mm=rc` reports

> shared library './export_vars.dll' was built with -mm=gc, this module with -mm=rc. Objects
> crossing between them are never reclaimed.

and still runs, which is the agreed policy: allow the link, treat what crosses as immortal, leak
rather than double-free.

**Two things this does not yet do.** Nothing marks crossing objects immortal — that lands with
ownership insertion, and until a release actually frees, a mixed link is harmless anyway. And
the mismatch path has no automated test: the 106 cross-module tests all build both sides the
same way, and giving the runner a per-side model would be more plumbing than the one warning is
worth. The marker's *presence* is covered by all of them, which is the part that could break
something.

**The consequence to keep in view:** the default lib is GC-built. Under `-mm=rc` everything it
allocates crosses a boundary and therefore leaks. Avoiding a per-model default lib is what the
allow-and-leak policy bought — this is the price of it, and it means `-mm=rc` will not be
leak-free for real programs until the default lib can be built per model.

### 9.8 Weak references: `WeakRef<T>`

The decision on cycles is **weak references in the language**, rather than leak-and-document.
This section settles their representation, because it is ABI-shaped and this arc has been
sequenced around making those decisions before writing code.

#### Surface: `WeakRef<T>`, not a `weak` keyword

JavaScript already has `WeakRef<T>` with `.deref(): T | undefined` (lib.es2021.weakref). Using
that spelling costs no change to the vendored `ts-new-parser`, rides the generics machinery that
is already cross-module-complete, and is a shape TypeScript programmers know.

The semantics come out *stronger* than JavaScript's, compatibly: `deref()` returns undefined
exactly when the last strong reference went, deterministically, rather than "whenever the
collector felt like it". Under `-mm=gc` it can be backed by a plain strong reference that never
returns undefined — a legal implementation of the JS contract, and one that keeps both models
working. `WeakMap`/`WeakSet` are out of scope.

#### Representation: a weak count, and where to put it

Something has to outlive the object to answer "is it dead". Three ways: a weak count beside the
strong one, a side table keyed by address, or a per-object indirection cell (which needs a
header slot or a table to be found, so it collapses into one of the other two).

The objection to a weak count was that §9.5's uniform-header requirement would force it on
`-mm=gc` builds too — doubling a header that is already dead weight there. **That objection
dissolves once the header grows downwards.** Put the strong count immediately before the
payload and the weak count before *that*:

```
    [ weak ] [ strong ] | payload
                        ^ the pointer everything holds
```

`strong` is at `payload - wordSize` in **every** model. That is the only field a cross-model
write touches — marking a foreign object immortal — so the uniformity §9.7 needs is preserved
while `weak` exists only under `-mm=rc`. GC builds keep the single word they have today.

This does split one constant in two: the *block* size, used for allocation and free, and the
*strong offset*, used by the count operations. `getBlockPtrFromPayloadPtr` currently serves
both, and the count paths would move to the strong offset.

Taking a weak reference to an immortal object — a string literal, or anything from a
differently-managed module — never touches the weak word: immortal means never dies, so the
reference is trivially always valid. That keeps a `-mm=rc` module from reading a second header
word that a `-mm=gc` module never wrote.

#### Lifecycle

Strong zero destroys, weak zero frees. When the last strong reference goes, the fields are
released as they are today, but the block itself survives while any weak reference remains — a
tombstone, distinguished by `strong == 0 && weak > 0`. `deref()` checks `strong > 0`, and if so
increments it and returns the object, so the referent cannot die between the check and the use.

`WeakRef<T>` is itself an owned type with its own release routine — decrement `weak`, free the
block if both counts are zero — which makes it one more shape for `ReleaseRoutineLogic` rather
than anything new.

None of the count operations are atomic. That matches the rest of the compiler today and should
be revisited with threading, not before.

#### What this does not solve

An accidental cycle still leaks silently; weak references let a programmer break one they know
about. The natural follow-on is a debug-mode leak report at exit — every block whose strong
count never reached zero — which is cheap once counts are maintained, and is a far better answer
than a cycle collector for a language whose users can switch to `-mm=gc` with one flag.

### 9.9 The first real caller: `delete`

Landed 2026-09-03, 847/847 green.

Everything before this generated release machinery that nothing called. `delete` is the one
place a reference is dropped that the language already spells out, so it makes the natural first
caller — and unlike ownership tracking it is a single lowering site, not a whole-program
analysis.

Under `-mm=rc`, `DeleteOp` now drops a reference instead of freeing outright: the object goes
only if this was the last reference, and what it owns is released with it. Under `gc` and
`none` it still frees directly, so the default is untouched. Two things fall out that a bare
free did not give: an object's fields are released rather than leaked to the collector, and
`delete` can no longer free an immortal block.

`ReleaseRoutineLogic::emitReleaseValue` is the entry point ownership tracking will reuse. The
per-type routines address storage rather than values, so it goes through a small value-taking
wrapper whose alloca sits in the wrapper own entry block — which keeps every caller from having
to find a safe place for one, since a release inside a loop must not grow the frame, and LLVM
inlines and promotes it away.

Verified under both models: a class owning a string and an array of strings releases correctly,
and `delete` on a string literal leaves the static block untouched, which is the immortal marker
doing its job. `delete` on a plain string local emits no `DeleteOp` in either model —
pre-existing behaviour, unchanged here.

**This is the first change that is not inert.** It only affects `-mm=rc`, and only `delete`, but
a release now actually frees. With no retains inserted yet every block still has a count of one,
so a released object is always the last reference — which is exactly the case ownership tracking
will complicate.


### 9.10 Step 4e: `ts.Retain` and `ts.Release`

Ownership is now sayable in the dialect. `ts.Retain` records that a further owner holds a
value; `ts.Release` gives one owner's claim up, destroying the value and freeing its block when
it was the last. Nothing emits either yet, so this step is still inert.

**The ops erase under any model that is not reference counting.** This is the design decision
the rest of the step follows from, and it is what makes "RC is an option" hold at the level of
the code rather than as an aspiration. MLIRGen can state ownership once, unconditionally, with
no `isRefCounted()` branching through it; the ops carry the intent and the lowering decides
whether it costs anything.

It also reshapes the risk of step 5 considerably. Ownership insertion is where a mistake stops
being inert — a missing retain frees live memory, an extra one leaks — but a misplaced op is
*erased* in a collected build. The ~830 GC tests are therefore structurally immune to
insertion bugs, not merely expected to pass. Only the 17 `-mm=rc` tests can break, which is a
blast radius small enough to reason about.

**Retain is not the mirror image of release, and the asymmetry is the whole difficulty.**
Retaining a *reference* stops at the block it names: a second reference to an object does not
duplicate that object's own references to its fields. Release does walk the fields, but only
inside `emitIfLastReference` — that is, only when the block is about to die and its fields'
references die with it. What does propagate a retain inwards is a value held *inline* — a
tuple, an optional, a tagged union — because copying one really does duplicate every reference
it holds. Getting this backwards leaks (retaining fields that were never released) or
double-frees (releasing fields that were never retained), and neither shows up until a count
is wrong much later, so the two builders sit next to each other in one file with the reasoning
written between them. `ReleaseRoutineLogic` became `OwnershipRoutineLogic` for that reason.

`__tslang_inc_ref` skips a block marked `HEAP_BLOCK_IMMORTAL`, which is not an optimisation:
incrementing all-ones gives zero, and the next release would read that as "last reference" and
free a string literal.

The descriptor record grew a retain slot beside the release one (`TYPE_DESCR_RETAIN`), for the
same reason the release slot exists — a tagged union carries its payload inline, so copying one
has to retain a value whose type is only known at run time, and the tag is what knows it. The
block header stays last, immediately in front of the name bytes, so a tag still reads as an
immortal string payload; the name simply moved from offset 24 to 32.

Verified by reading the emitted IR under both models. A retain routine loads the reference and
calls `__tslang_inc_ref`, with no field walk, confirming the asymmetry holds in the generated
code and not just in intent. Temporarily emitting both ops at the `delete` site showed
`tsretv_`/`tsrelv_` calls under `-mm=rc` and *nothing at all* under `-mm=gc`, where only the
collector's `GC_free` remains; the hook was then reverted. Full release suite green: 847/847.

### 9.11 Step 4f: `using` disposes when an exception unwinds with no enclosing `try`

The reported gap: `using r = new Res(); throw 1;` at a function's top level, with no `try`
anywhere in that function, never ran `[Symbol.dispose]()` on the way out — confirmed at both
`-O0` and `-O3` before any fix. `mlirGen(Block)` disposed a `using` only on the block's *normal*
exit path; nothing gave it a landing pad to run from on `throw`.

**The fix synthesizes a catch-less `TryOp` around a block that declares `using`.** Mirrors
`mlirGen(TryStatement)`'s own try-body/cleanup handling almost exactly - a real
`try { using x = ...; } finally {}` already goes through that path and already disposes
correctly on throw, so the synthetic version reuses it rather than inventing a second mechanism.
Catches and finally stay empty; `TryOpLowering` erases an empty catches region and wires the
cleanup block as a plain cleanup landing pad, so the exception is never caught, only cleaned up
after.

**Building this surfaced four independent pre-existing bugs in the `TryOp`/dispose machinery,
none caused by this session's other changes.** Each was confirmed with 100% hand-written source
- an explicit `try`/`catch`/`finally`, no synthesis involved - before being treated as
out-of-scope for this step:

1. **A `TryOp` with cleanup but no catch and no finally crashed the lowering.** Every TypeScript
   `try` statement had always had at least one of catch/finally, so
   `unwindDests.push_back(catchesBlock ? catchesBlock : finallyBlock)` in
   `LowerToAffineLoops.cpp`'s `TryOpLowering` had never had to handle both being null. The
   synthetic wrapper is the first thing to build a cleanup-only `TryOp` at all, so it's the
   first thing to hit this. **This one is fixed, not just avoided** - a null `Block*` doesn't
   belong in `unwindDests` in the first place, and the Linux side of the same function already
   had the correct three-way fallback (`catchesBlock -> finallyBlock -> parentTryOpLandingPad ->
   empty (resume)`) sitting right next to the broken Windows one, comment already anticipating
   exactly this case. The Windows site now matches it.
2. **`TryOp` nested inside another `TryOp`'s body crashes the LLVM translation** with an LLVM
   assertion (`Cannot assign a name to void values!`), reproduced by hand:
   `try { try { using x=...; throw; } finally {} } catch {}`. Not fixed - guarded against:
   `blockIsFunctionRootBody` restricts synthesis to a function's own top-level body, which by
   construction can never be nested inside anything.

   > **Update (§9.17).** Fixed. `blockIsFunctionRootBody` was already gone by §9.13; what was
   > left of this was a `using` one scope deeper than a hand-written `try`'s body, and it was
   > `Win32ExceptionPass::ToInvoke` mangling an operation that was already an invoke. A
   > `using` in a catch or finally *clause* is still guarded, by
   > `blockIsInsideCatchOrFinally`, and was re-checked against the fix: a different cause.
3. **A block with its own `using` nested inside a `TryOp` that already has other `using`s
   breaks MLIR verification** (`ts.PropertyRef` gets the wrong ref type for the inner
   `using`'s dispose method), reproduced by hand:
   `try { using a=...; { using c=...; } } finally {}`. Not fixed - guarded against:
   `blockHasNestedUsing` scans (skipping into neither a nested function nor class) for a
   `using` anywhere below the block's own top level.

   > **Update (§9.17).** Fixed, and `blockHasNestedUsing` is deleted. Same `ToInvoke` cause as
   > item 2. The guard's cost was that the *outer* `using` stood down from being wrapped so
   > the inner one could be, so it never disposed on unwind at all - the row in §9.13's table
   > reading "outer skipped" in both columns.
4. **`using` plus `return` inside a `try` body is broken independent of throw entirely**,
   reproduced by hand with the simplest possible shape: `try { using a=...; return; } finally
   {}`. `mlirGenDisposable`'s `FullStack` walk at the return site and the try-body's own tail
   dispose both try to dispose the same var. Not fixed - guarded against: `blockHasReturn` scans
   the whole function body for any `return`.
5. **A separate, still-unexplained hang** (not a compile failure) turned up disposing an
   *object-literal* `using` (`{ [Symbol.dispose]() {...} }`, as opposed to a class instance)
   across an unwind with no enclosing try - reproduced with the exact same shape as the fixed
   case, swapping only `new Res()` for a `loggy()`-style object literal, and confirmed the
   synthesis correctly declined to wrap it (no `ts.Try` in the emitted MLIR) before the hang was
   traced to the untouched pre-existing plain-dispose path. Guarded against the same way as the
   others: `blockUsingInitializersAreAllNewExpr` restricts synthesis to `using x = new
   SomeClass(...)` - every case actually verified working is written exactly this way.

Since none of these four are cheaply detectable from a resolved *type* before generation (the
type isn't known yet - see `blockDeclaresUsing`'s own comment on why the check has to be
syntactic), each guard is a syntactic proxy for "would this hit the known-broken shape,"
checked before deciding whether to wrap: `blockDeclaresUsing`, `blockIsFunctionRootBody`,
`blockUsingInitializersAreAllNewExpr`, `!blockHasNestedUsing`, `!blockHasReturn`. Failing any
of them falls back to the exact pre-existing plain-dispose path, byte-for-byte - a function that
doesn't qualify is no worse off than before this step, just not newly fixed either. The net
result is narrow: `using x = new SomeClass(...)` declared directly in a function's own
top-level body, with no other `using`-bearing scope and no `return` anywhere in that function,
now disposes correctly on `throw` with no enclosing `try`. Everything else - object-literal
disposables, `using` plus `return`, nested `using` scopes - is exactly as before: not fixed,
not worse.

New test: `test/tester/tests/03disposable.ts` (`test-compile-03-disposable`,
`test-jit-03-disposable`) - the originally reported shape, now asserting dispose actually ran.
Full release suite green: 849/849 (847 existing + the 2 new).

> **Re-audited 2026-09-03 (§9.13). Two of the four guards were already stale when written and
> have been deleted; two of the "pre-existing bugs" above do not reproduce.** Read §9.13 rather
> than this list for the current state.

### 9.12 Step 5a: locals own what they hold

The first slice of step 5, and the first time anything in the compiler calls a retain or a
release on its own account rather than because the program said `delete`. Full release suite
green: 852/852 (849 existing plus 3 new).

**The rule.** A local variable declaration whose type owns heap memory takes a reference when
it is declared and gives it back at every exit from its scope — the block's end, a `return`
from anywhere inside it, a `break` or `continue` that leaves it. Assigning through such a local
hands the count over: the incoming value gains this scope as an owner and the outgoing one
loses it.

**Stated unconditionally, and the collected build shows no trace of it.** MLIRGen never asks
which memory model is in force; it emits `ts.RetainSlot` / `ts.ReleaseSlot`, and the lowering
decides. Confirmed by reading the emitted LLVM for the same file under both models: under `rc`
the retain sits immediately after the initialising store and the releases sit in reverse
declaration order at each exit; under `gc` the two functions are **instruction-for-instruction
what they were before this step** — not a dead load left for a later pass to remove, because
the slot-addressed ops erase whole and take the access with them. That is what the new
`ts.RetainSlot`/`ts.ReleaseSlot` pair buys over the value-addressed `ts.Retain`/`ts.Release`
from §9.10, which would have needed a load kept alive under a collector to have an operand.

**Balanced by construction, which is the property that makes this safe to land first.** The
reference an allocation is born with (§9.6) is never given up here. Every release this step
emits is therefore paired with a retain this step emitted, so no release can outnumber its
retains and nothing can be freed early. What it can do is leak — and under `-mm=rc` the
collector is still what reclaims, so the leak is inert. That direction is deliberate: an
over-release is a use-after-free that surfaces far from its cause, and a leak is not. Removing
the slack is later work, and each piece of it is a separate decision: consuming the +1 when the
initialiser is a fresh allocation, retaining on field and element stores, and releasing
temporaries.

**Where a local is *not* made an owner**, each because the frame borrows the reference rather
than owning it, and releasing one would drop a count nobody took:

- globals, which outlive every scope;
- parameters — only variable declarations reach the hook, so a parameter's slot is never
  marked, and assigning to a parameter neither retains nor releases;
- captured variables held in the `this` context, whose slot belongs to the context;
- `const` bindings with no storage, which have no slot to release from;
- **declarations with no initialiser.** This one was found the hard way and is the single bug
  this step produced: a `catch (v: string)` variable is declared like any other `let` but
  written by the landing pad, not by an initialiser, so retaining at the declaration read an
  uninitialised slot as a live reference and trapped. `00try_catch.ts` under `-mm=rc` was the
  only test in 849 that failed, which is exactly the blast radius §9.10 predicted. The
  consequence is that a `let s: string;` assigned later never becomes an owner either —
  correct rather than merely safe, since the assignment path only fires on a slot the
  declaration marked, so that stays balanced too.

**The unwind leg is skipped, on purpose.** An owned local's storage is allocated inside the
`TryOp` body region, which does not dominate the cleanup region, so a release emitted there
would not verify. Disposal still runs on that leg (§9.11); the release does not, which leaks
the reference when an exception passes through. Fixing it means hoisting owned storage out of
the operation the way `using` variables already are (`allocateUsingVarsOutsideOfOperation`) —
tractable, and left for the step that also brings the verifier.

> **Update (§9.15).** Done. The hoisting landed, the dominance problem is gone, and the
> release now runs on the unwind leg too, so the leak described here no longer happens.

**Where it hooks in.** Three points, all of them ones that already existed:

- `takeOwnershipOfLocal` (`MLIRGenVariables.cpp`), called from `registerVariable` right where
  `usingVars` is collected, marks the storage with `__owned` and emits the retain.
- `mlirGenScopeExit` (`MLIRGenImpl.h`) wraps `mlirGenDisposable` and the new
  `mlirGenReleaseOwned`, so all eleven existing scope-exit call sites — block end, `return`,
  `break`, `continue`, try body — got the releases for free. Disposal runs first: a disposable
  is still usable while its `[Symbol.dispose]()` runs, and dropping the last reference first
  could have freed it.
- `mlirGenSaveLogicOneItem` (`MLIRGenImpl.h`) is the single choke point every assignment form
  passes through — plain, compound and destructuring alike. Retain-then-release, in that
  order, is what makes `x = x` safe: releasing first could drop the last reference and free the
  value about to be stored back.

`ownsHeapMemory` moved from `OwnershipRoutineLogic` to `MLIRTypeHelper` so that both sides ask
one function. The two disagreeing about which types own memory would place retains that never
pair with a release, which is the failure mode with no local symptom.

**Coverage.** `test/tester/tests/00owned_locals.ts`, run under all three models
(`test-compile-00-owned-locals`, `test-jit-00-owned-locals`, `test-jit-rc-owned-locals`).
Beyond one local of each owning shape, it covers the paths that reach a slot *without* going
through an assignment expression, since that is where a missing retain would turn into a
release of a reference nobody took: `for…of` bindings, destructured declarations and
destructured assignment (`[a, b] = [b, a]`), a captured local, `break`/`continue` out of a
loop, a `return` out of a nested block, and returning a value the caller is about to own. A
2000-iteration churn loop makes an early free likely to be handed straight back out rather than
silently tolerated.

### 9.13 Re-auditing the `using` guards: half of them were already unnecessary

§9.11 added four conditions, each meant to keep the synthesized `TryOp` away from a shape that
crashed. Each was real when observed. But they were all observed *before* §9.11's own
`unwindDests` fix landed, and that fix — the cleanup-only `TryOp` that pushed a null `Block *`
— turned out to be the cause of more of them than the notes credited. Re-running every guarded
shape against the current build:

| shape | before | after |
|---|---|---|
| `using` in an `if` block, throw | dispose skipped | **disposes** |
| `using` in a loop body, throw | dispose skipped | **disposes** |
| `using` two scopes deep, throw | dispose skipped | **disposes** |
| `using` sharing a function with `return`, throw | dispose skipped | **disposes** |
| `using` inside a hand-written `try` | worked | works |
| object-literal `using`, throw | dispose skipped | dispose skipped |
| outer `using` with a nested `using` scope, throw | outer skipped | outer skipped (**both dispose since §9.17**) |

**`blockIsFunctionRootBody` and `blockHasReturn` are deleted.** Both were guarding shapes that
now work. Dropping the root-body condition is the one that matters: synthesis is no longer
confined to a function's own top-level body, so a `using` in an `if` branch, a loop body, or a
block nested inside a hand-written `try` all dispose on the way out. Nested `TryOp`s, which
§9.11 recorded as crashing LLVM translation, compose correctly — `try/catch` inside
`try/catch`, and a synthesized cleanup inside a hand-written `try`, both verified.

**`blockUsingInitializersAreAllNewExpr` and `blockHasNestedUsing` stay, and each was confirmed
individually necessary** by dropping it alone and rebuilding: without the first, an
object-literal disposable fails the build; without the second, an outer `using` whose block
also contains a nested `using` scope segfaults the compiler. Those are the two genuinely open
bugs, and they are now stated in terms of what was actually reproduced rather than what was
inferred.

> **Update (§9.17).** `blockHasNestedUsing` is now deleted too — the segfault it was standing
> in front of was `Win32ExceptionPass::ToInvoke`, not anything about nesting. The method here
> is what made that possible to check: confirming a guard is *individually* necessary is what
> turns it from folklore into a one-line experiment to redo after any fix in the area.
> `blockUsingInitializersAreAllNewExpr` was re-checked and stays.

Method worth repeating: the gate was made maskable by an environment variable for the duration
of the experiment, so one build could test all sixteen combinations. Four rebuilds' worth of
bisection in a single compile, and the mask made "necessary individually" a question that could
be asked directly instead of argued from a combined result.

**Separately, a genuinely new pre-existing bug, unrelated to any of this.** Throwing from
inside a `catch` clause crashes the LLVM backend (`X86 Assembly Printer`, access violation) —
reduced to `try { throw 1; } catch (e: int) { throw 2; }` with no `using`, no locals and no
heap types anywhere in it, so neither ownership insertion nor the `using` machinery can be
involved. Recorded here because it surfaced while building the matrix above; not fixed, and no
test asserts it, which is why nothing caught it before.

> **Fixed 2026-09-03, see §9.14.**

New test: `test/tester/tests/04disposable.ts` (`test-compile-04-disposable`,
`test-jit-04-disposable`), covering the four newly-working shapes plus the two exact-count
cases that would catch a double dispose — a function that throws past a `using` on one path and
returns past it on the other, and a synthesized cleanup nested inside a hand-written `try`.
Full release suite green: 854/854.

### 9.14 Throwing out of a `catch` clause

`try { throw 1; } catch (e: int) { throw 2; }` crashed the compiler. The cause is one missing
line, and the shape of it is worth keeping.

`ThrowOpLowering` ends with `clh.CutBlock()`, which drops everything after the throw in its
block — including the `EndCatchOp` that `TryOpLowering` had placed just before the region's
terminator. `Win32ExceptionPass` then finds a catch region with no end marker, picks one for
itself by splitting the block *ahead* of the throw, and emits the `catchret` there. The result
is a `catchret` followed by a call that still carries `"funclet"(token %catchpad)` — a bundle
naming a funclet it has already returned from. That reaches the backend and crashes it.

`ReturnOpLowering`, `BreakOpLowering` and `ContinueOpLowering` all emit an `EndCatchOp` before
leaving a catch. `ThrowOpLowering` was the only abrupt exit that did not.

**It needed a new side table rather than the existing one.** The other three record "I am
leaving a catch" by having `tsContext->unwind[op]` set. A throw cannot: for a throw that map
already means its invoke destination, and the finally handling writes exactly that into it. So
`leavesCatch` is its own set, populated by the same walk over the catches region that already
marks returns.

**And only when there is no `finally`.** With one, the throw becomes an invoke into the finally
block and *the finally* ends the catch; ending it at the throw as well runs it twice and breaks
the unwind. `51exceptions.ts` — `catch (e: number) { … if (k >= 10) throw e } finally { … }` —
is the case that proves it, and it caught the first version of this fix.

**Still open, and each confirmed independent of this fix:**

- **An exception escaping a catch clause is lost under AOT**, and always was. A *call* in a
  catch that throws (`catch (e) { thrower(); }`) loses it too, with no `throw` statement
  involved anywhere and nothing in this change able to affect it. The IR is well-formed at
  both `-O0` and `-O3`; the gap is in the AOT exception tables. `00throw_in_catch.ts` is
  therefore registered JIT-only.

  > **Update.** Both wrong, and differently wrong. The first was the
  > `CatchableType::sizeOrOffset` miscompile (§9.15) and went away with it; this file's tests
  > now run under AOT as well, as `test-compile-00-throw-in-catch`. The second was neither
  > AOT-specific nor in the exception tables: the MLIR inliner was **erasing the throw**
  > (§9.16). "The IR is well-formed" was checked on the callee, which is exactly the function
  > that survives intact — the deletion happens at the call site.

- **A call inside a catch followed by a throw out of it** (`catch (e) { new Res(); throw 2; }`)
  crashes at run time, AOT and JIT alike, at every optimisation level and memory model. Its IR
  is well-formed too. Unrelated to ending the catch.

  > **Update.** Also the `CatchableType::sizeOrOffset` miscompile (§9.15); fixed there, and
  > covered now by `00try_using_catch.ts`. Correctly identified as unrelated to ending the
  > catch — it just wasn't an EH bug at all. Three of these entries had one cause between
  > them, and the thing they had in common was a *call in a catch*: that is the shape whose
  > frame layout the overflow reached.

- **Throwing from a `finally`** (`try { throw 1; } finally { throw 2; }`) segfaults, from the
  same `CutBlock` cause — `ts.BeginCleanup` with no `ts.EndCleanup`. Not fixed here because
  `EndCleanupOp` is a terminator taking a landing pad and unwind destinations rather than a
  marker, and the finally region is cloned once per exit path, so each copy would need its own.

**A regression in §9.13 turned up while testing this, and is fixed here too.** Dropping
`blockIsFunctionRootBody` also stopped excluding *catch and finally regions*, and synthesizing
a cleanup `TryOp` in one crashes the compiler — `catch (e: int) { using r = new Res(); }`
segfaults with the wrapping and compiles without it. §9.13's matrix checked nesting inside a
try *body* and never inside a catch region. `blockIsInsideCatchOrFinally` restores exactly that
half; the four shapes §9.13 fixed all still work.

The same predicate also excludes those clauses from ownership (§9.12): under `-mm=rc` a release
in a catch clause is a call inside a funclet, which is the fragile construct above, and
`catch (e: int) { let r = new Res(); }` segfaulted. Locals there are simply not owned now — they
leak, the trade every other exclusion in §9.12 makes. (That leak was covered by the collector when
this was written; since step 6 (§9.28) it is a real one under `-mm=rc`.)
Both holes existed because no test had a `using` or a heap local inside a catch clause;
`04disposable.ts` now has both, and `03disposable.ts`/`04disposable.ts` gained `-mm=rc`
variants, which is what would have caught the ownership half.

New test: `test/tester/tests/00throw_in_catch.ts` (`test-jit-00-throw-in-catch`,
`test-jit-rc-throw-in-catch`). Full release suite green: 858/858.

### 9.15 Step 5b: owned storage is hoisted out of the `TryOp`, and the unwind leg releases

§9.12 left one hole on purpose: an owned local's storage was allocated inside the `TryOp` body
region, which does not dominate the cleanup region, so the release could not be emitted on the
unwind leg and the reference leaked when an exception passed through. This step closes it — and
turned up a miscompile of our own on the way, which is the more valuable half of the result.

**What landed.** Owned storage is hoisted out in front of the `TryOp`, exactly the way `using`
storage already was. `allocateUsingVarsOutsideOfOperation` is renamed
`allocateScopeOwnedVarsOutsideOfOperation` because it now serves both, and the hoist decision
for an owning local cannot be made in `detectFlags` with the rest — it needs the variable's
type, which is not known until `createLocalVariable`. Verified in the emitted LLVM: the
`alloca` moves to the function entry and the initialising store stays at the declaration, in
both memory models. Nothing else about a collected build changes.

**One predicate, two callers.** `localTakesOwnership` is the single test for "does this
declaration make its scope the owner", shared by the hoisting decision and by
`takeOwnershipOfLocal`. They must agree: a local that is hoisted but not owned only wastes a
move, but one that is owned and *not* hoisted puts a release in a region its slot does not
dominate and the module stops verifying. This is the same lesson as moving `ownsHeapMemory`
into `MLIRTypeHelper` in §9.12 — two sides asking the same question separately is the failure
mode with no local symptom.

**Hoisted storage starts null, under `-mm=rc` only.** A hoisted slot's initialising store stays
behind at the declaration, and the unwind edge can reach the cleanup region before that store
runs — the allocation in `let r = new Res()` is itself an `invoke` whose unwind destination is
that region. A release there would read whatever the frame happened to hold, which is precisely
how the catch-variable bug in §9.12 trapped. Null is the one value every release routine treats
as nothing to do (`emitIfLastReference` null-checks first), so `VariableOpLowering` zero-fills a
hoisted owned slot. Gated on `isRefCounted()` in the *lowering*, not in MLIRGen: no other model
reads the slot before its store, and a collected build is meant to come out of this step
byte-identical.


**The unwind leg releases.** The cleanup region now calls `mlirGenScopeExit` rather than only
`mlirGenDisposable`, so an exception passing through a scope gives back the references that
scope took. Confirmed in the emitted LLVM: under `rc` the cleanup funclet holds one `tsrel_` per
owned local, in reverse declaration order, each carrying the funclet bundle; under `gc` the same
region is empty, because the slot-addressed ops erase whole. Step 5's local half is now complete
on every path.

#### The detour: a miscompile of our own, found because this step tripped it

Turning the release on broke exactly one test, and chasing it turned up a bug that had nothing
to do with reference counting and had been in the tree the whole time.

The shape, which needs no ownership at all and fails under `-mm=gc`:

```ts
function f() {
    try { using r = new Res(); throw 1; }
    catch (e: TypeOf<1>) { print("a"); print("b"); }
}
```

`main` keeps a pointer in `rsi` across the call to `f` — legal, `rsi` is callee-saved — and gets
it back with its **low 32 bits zeroed**.

**Root cause: `CatchableType::sizeOrOffset` said a caught `int` was 8 bytes.** Both RTTI helpers
(`MLIRRTTIHelperVCWin32.h` and `LLVMRTTIHelperVCWin32.h`) hardcoded `8` for every catchable
type. The CRT copies exactly that many bytes into the catch variable's frame slot, so catching a
4-byte `int` wrote 8 and clobbered whatever sat above the slot. The symbol name we emit had been
saying so all along: `_CT??_R0H@8` **4** — the trailing digit is the size, and it disagreed with
the record it named.

**Why it hid for so long.** What sits above the catch slot is a question of frame layout. Ahead
of time it was padding, so the overflow was invisible. The JIT compiles with the **large code
model**, where every call materialises a 64-bit address into a register; that pressure makes a
catch funclet use a callee-saved register, which makes the parent save it, which puts a saved
register exactly where the overflow lands. Hence: JIT-only in practice, sensitive to unrelated
code changes, and not reproducible with clang — clang emits `4`.

**How it was found**, because the route generalises. `llc -code-model=large` on the same IR
reproduced it ahead of time, which exonerated the JIT's unwind-table registration and turned a
compiler-rebuild loop into a seconds-long one. clang's C++ equivalent at `-mcmodel=large` did
*not* reproduce, which said the defect was in our IR rather than the backend. Deleting the
cleanup funclet still reproduced, which said the `using` was a red herring. A hardware
write-breakpoint on the saved-register slot then named the writer: an 8-byte store from inside
the CRT's EH machinery, of the value `1`, at establisher+52 — the catch object, one word wide
for a four-byte `int`.

**The fix** gives each catchable type its real size: `int` is 4 and `double` is 8 on every
target, while the pointer-shaped ones (string, opaque pointer, class reference) take
`compileOptions.sizeBits / 8`, since those genuinely do follow the architecture flag. Both
helpers were wrong identically and both are fixed; leaving one behind is the classic trap with a
duplicated table. Worth recording while in there: the whole name table in
`LLVMRTTIHelperVCWin32Const.h` is 64-bit MSVC mangling (`PEA` is a `__ptr64` pointer, and
pointer entries bake `@88` into the symbol), so a 32-bit target needs its own table, not just a
different size.

New test: `test/tester/tests/00try_using_catch.ts`, run under all three models
(`test-compile-00-try-using-catch`, `test-jit-00-try-using-catch`,
`test-jit-rc-try-using-catch`, `test-jit-none-try-using-catch`). It covers the caught-`int` case
that was broken and a caught `number`, which is genuinely eight bytes and has to keep working
now that `int` narrowed. Writing it turned up one more thing worth recording: moving the `using`
one scope deeper, into an `if` inside the try body, crashes the *compiler* in every memory
model. That is §9.11's second item — a synthesized cleanup `TryOp` nested inside a real
`TryOp`'s body — still open, and it is why the test covers only the flat shape.

> **Update (§9.17).** Fixed, and `00using_nested_scopes.ts` now covers the deeper shape.

Full release suite green: 862/862.

### 9.16 The inliner was deleting throws

Not an RC bug at all, and not an EH bug either — a silent wrong-code bug in the ordinary
optimised build, found by re-testing §9.14's open list after the §9.15 fix and asking why one
entry survived. Two defects, one behind the other.

**A function whose body ends in a throw inlined down to nothing.** This:

```ts
function thrower() { throw 5; }
function callsIt() { thrower(); }
```

compiled, under `--opt`, to a `callsIt` that does nothing but return:

```mlir
ts.Func @callsIt !ts.func<, , false> {
    "ts.ReturnInternal"() : () -> ()
}
```

MLIR's inliner has a fast path for a single-block callee (`inlineRegionImpl`, the
`singleBlockFastPath` branch): it offers the block's terminator to the dialect's
`handleTerminator` hook and then calls `firstBlockTerminator->erase()` **unconditionally**. The
assumption is that a terminator is return-like and its operands are all the block had left to
say. `ts.ThrowCall` is a terminator too, and `TypeScriptInlinerInterface::handleTerminator` only
ever did anything for `ReturnInternalOp` — so the throw was handed over, ignored, and erased.
The multi-block path has no such erase, which is why a *conditional* throw was always fine and
only the throw-only helper was hit.

The fix is the hook MLIR provides for exactly this, `allowSingleBlockOptimization`: decline the
fast path unless the terminator is a return. The multi-block path then leaves the throw in place
as the block's terminator and puts the code after the call site in an unreachable block, which
is what it should have been all along.

**Then the same throw inlined into a `catch` clause crashed the backend** — the case that had
been recorded as "lost under AOT, so the gap is in the AOT exception tables". It was neither.
`Win32ExceptionPass` ends a catch region at a `_CxxThrowException` call by splitting the block
*ahead* of it and emitting the `catchret` there, which leaves the throw outside the funclet; but
it also collected that same call into `catchRegion.calls`, which is what stamps
`"funclet"(token %catchpad)` on. So the throw named a pad it had already returned from — the
identical malformed shape §9.14 describes, reached by a different route. An `__cxa_end_catch`
marker is what normally keeps the two apart, by closing the region before the throw is reached,
and a throw the inliner brought in has no marker: the `EndCatchOp` that followed the call it
replaced went with the rest of the now-unreachable code after it.

**The first attempt at that overreached, and 00try_catch.ts caught it.** Closing the region on
the throw, the way the marker does, broke three tests. That scan walks `instructions(F)` in
order rather than by region, so once inlining has merged several functions into one, a throw
belonging to one catch turns up while another is still open — and closing there strands the rest
of that catch's calls with no bundle at all. Skipping the call is all that is needed; the region
stays open. The `isCatch()` guard matters too: a cleanup region gets no `catchret`, so its throw
stays inside the funclet and does still need the bundle.

**What this says about the earlier diagnosis.** Three entries on §9.14's open list had two
causes between them, and both diagnoses pointed at the runtime — "the AOT exception tables", "it
is the runtime side that drops it" — on the strength of the IR being well-formed. It was: the IR
of the *callee*, which is the one function the bug leaves intact. The deletion happens at the
call site, and the call site was never looked at. The cheap check that would have settled it in
minutes is the one that eventually did — dump `--emit=mlir-affine` with and without `--opt` and
diff, which is a much smaller step than reasoning about exception tables.

New test: `test/tester/tests/00throw_inlined.ts`, run under all three models
(`test-compile-00-throw-inlined`, `test-jit-00-throw-inlined`, `test-jit-rc-throw-inlined`,
`test-jit-none-throw-inlined`). It covers the plain call, the call from inside a catch clause,
and a conditional throw as the control that always worked. `00throw_in_catch.ts` picks up its
AOT variant here as well, now that nothing on its header's list is true any more.

Full release suite green: 867/867.

### 9.17 A nested `using` scope, and the guards that were standing in for one bug

Two of §9.11's guards turned out to be avoiding the same defect, in a place neither of them
named. Fixing it retires one guard outright and closes the last two `using`-on-unwind gaps.

**The shapes.** Both crashed the compiler, in every memory model:

```ts
try { if (flag) { using r = new Res(); throw 1; } } catch (e: int) { }   // one scope deeper
using a = new Res(); { using c = new Res(); } throw 1;                   // outer plus inner
```

The first was §9.11's item 2, guarded by `blockIsFunctionRootBody` and, once that went in
§9.13, by nothing — it simply crashed. The second was item 3, guarded by
`blockHasNestedUsing`, whose cost was that the *outer* `using` stood down from being wrapped so
that the inner one could be, and therefore never disposed on unwind at all.

**One cause: `Win32ExceptionPass::ToInvoke`.** The helper exists to turn a call into an invoke
with a given unwind destination, so it splits the block at the call to make room for the new
terminator. But two of its callers hand it an operation that is *already* an invoke — the
"fix incorrect landing pad" loop that redirects an invoke whose unwind destination is wrong. An
invoke already ends its block, so splitting at it puts it alone in the new continuation block,
and every caller erases it immediately afterwards. What is left is an empty block with no
terminator, and the real continuation stranded with no predecessors:

```llvm
  %invoke = invoke void %24(ptr %23) [ "funclet"(token %cleanuppad) ]
          to label %invoke.cont unwind label %26
invoke.cont:                                      ; preds = %15
                                                  ; <- empty, no terminator
25:                                               ; No predecessors!
  cleanupret from %cleanuppad unwind label %26
```

That reaches `AlwaysInlinerPass`, which walks the empty block and dies. An invoke needs its
unwind edge redirected and the bundle added, not a block of its own; cloning it in place with
`CallBase::Create` and calling `setUnwindDest` is what the funclet-bundle loop a few hundred
lines above already does.

**Then the guards were re-tested, one at a time.** This is the payoff and the reason §9.13 was
careful to establish that each guard was *individually* necessary — that turns "is this still
needed?" into a one-line experiment rather than an argument.

- `blockHasNestedUsing` — **deleted.** The outer and inner `using` now both dispose on unwind,
  innermost first.
- `blockIsInsideCatchOrFinally` — **stays.** Dropped alone, `catch (e: int) { using r = new
  Res(); }` still crashes. A different cause, still open. It is worth naming its second cost:
  `localTakesOwnership` consults the same predicate, so a heap local declared in a catch or
  finally clause is not owned and leaks under `-mm=rc`.
- `blockUsingInitializersAreAllNewExpr` — stays, re-checked, unchanged.

**Also still open, and confirmed independent:** throwing from a `finally` still crashes the
compiler, in both memory models. That is the `ts.BeginCleanup`-with-no-`ts.EndCleanup` shape
§9.14 describes, and this fix does not touch it.

New test: `test/tester/tests/00using_nested_scopes.ts`, run under all three models
(`test-compile-00-using-nested-scopes`, `test-jit-00-using-nested-scopes`,
`test-jit-rc-using-nested-scopes`, `test-jit-none-using-nested-scopes`). It covers a `using` in
an `if` and in a bare block inside a try body, and the outer/inner pair both with the inner
scope already closed and with both still live, asserting disposal *order* rather than just that
it happened.

Full release suite green: 871/871.

### 9.18 The verifier, and the first thing it found

Step 5's plan named a verifier from the start — "every owned value with a matching release on
every path, unwind paths included" — and said to build it alongside the insertion rather than
after. This is that, one step ahead of the work it exists to guard.

**Where it runs, and in which model.** `OwnershipVerifierPass`, at the affine level, behind
`--verify-ownership`. Affine because that is the first point where the unwind paths are ordinary
CFG edges and can be walked like any other; before `TryOpLowering` they are regions, and after
`LowerToLLVM` the ops are gone. And in **every** memory model, not just `-mm=rc`:
`ts.RetainSlot` and `ts.ReleaseSlot` survive to there regardless of model and are only erased on
the way to LLVM, so a collected build checks the same invariant a counted one does. That matters
more than it sounds — most of the suite, and most of CI, is collected.

**What it checks.** For each `ts.RetainSlot`, a backward must-analysis over the function's
blocks: is there a path from the retain to a function exit that passes no `ts.ReleaseSlot` on
the same slot? A block releases if it does so directly or inside a region of one of its own
operations — counting nested regions as releasing rather than as opaque, because a verifier that
reports a leak the IR does pay somewhere the walk does not follow is a verifier that gets
switched off. Being a must-analysis it starts optimistic and is driven down to a fixed point,
which leaves a loop with no exit reading as satisfied — correctly, as it has no path to an exit
to leak on. The cheap structural half of the other direction is there too: a release naming a
slot that is never retained.

It checks the direction that leaks rather than the direction that frees live memory, on purpose.
Step 5a's insertion is balanced by construction, so an unmatched release cannot currently be
generated; what an extension to fields, elements, arguments or returns will get wrong first is a
path out that nobody released on.

**Confirmed to fire before being trusted.** A verifier that has never failed is a verifier that
might not work. The unwind-leg release from §9.15 was reverse-applied, and it reported the leak
at the right declaration, in all three memory models; restored, it went quiet again.

**What it found on its first run.** 460 test files, two with findings, both real.

1. **A `break` or `continue` written inside another block skipped every scope between itself and
   the loop** — the disposals a `using` declared *and* the references those scopes' locals took.
   Not an RC bug: `using` had it too, and that half is user-visible. Three iterations of
   `for (…) { using r = new Res(); if (i == 1) continue; }` disposed twice.

   The walk outwards stopped at the first scope that was not itself a loop. `isLoop` is set by a
   loop on the context it hands its body and then inherited by every context copied from it, so
   it answers "somewhere inside a loop", not "is the loop" — and the very first step of the walk
   thought it had already arrived. Written directly in the loop body it happened to be right,
   which is why that shape always worked and hid this one. Fixed by splitting the two meanings:
   `isLoopBodyScope` is taken by the block that becomes the loop's body and cleared for anything
   nested further in.

   Two attempts either side of it were wrong and are worth recording. Carrying the target label
   into the recursion instead of the empty one looks obviously right and breaks
   `02disposable.ts`: the loop sites clear `label` before storing it, so a labelled loop's
   context holds an empty label too, and `continue cont1` relies on the outer loop matching the
   empty label the recursion passes down. And moving the recursion out of the
   `ownedVars != nullptr` guard — on the reasonable theory that a scope owning nothing says
   nothing about its parents — broke `Path.ts`. The `isLoop` fix made it unnecessary anyway.

2. **A `[Symbol.dispose]()` that itself throws during unwind skips the release that follows it.**
   The cleanup region invokes dispose, and its unwind edge goes to the enclosing catch without
   passing the `ts.ReleaseSlot` on the far side. Real, and left alone: releasing before disposing
   would fix the path and break the ordering §9.12 chose deliberately — a disposable is still
   usable while its `[Symbol.dispose]()` runs, and dropping the last reference first could have
   freed it. `00using_nested_scopes.ts` is the one file that still reports.

New tests: `test/tester/tests/00break_continue_scope_exit.ts`, all three models
(`test-compile-00-break-continue-scope-exit`, `test-jit-00-break-continue-scope-exit`,
`test-jit-rc-break-continue-scope-exit`, `test-jit-none-break-continue-scope-exit`). It covers
`continue` and `break` from inside an `if`, two levels of nesting, a `using` in the intermediate
scope as well, the labelled form, and the shape that always worked as a control. Confirmed to
fail with the fix reverse-applied.

Full release suite green: 875/875.

### 9.19 Step 5c: fields own what they hold

The first piece of step 5 beyond locals, and the one the verifier from §9.18 was built ahead of.

**The gap.** A field store was a bare `ts.Store`. The runtime half had been in place since §9.4 —
`releaseFields` in `OwnershipRoutineLogic` walks an instance's fields when its release routine
runs — but nothing ever took the reference that routine was giving up, and overwriting a field
dropped the outgoing value on the floor without releasing it.

**The fix** is the one already written for locals, applied to a second kind of storage.
`isOwnedLocalSlot` becomes one arm of `isOwningSlot`; the other is `isOwnedFieldSlot` — a
`ts.PropertyRef` whose base is a class or object instance, and whose field type owns heap memory.
Retain the incoming value, release what the slot still holds, then store. Retaining first is what
makes `h.item = h.item` safe.

**Scoped deliberately.** A field of a record held *inline* — a tuple in a local, a parameter's
slot — is not covered. Its fields are released by whatever owns the record, which is only tracked
when that is an owned local, and retaining into a record nothing releases would leak. That is the
same question arguments and elements ask, and it gets one answer, later, not three.

**The counting stays balanced by construction.** A freshly allocated value's birth reference is
still unconsumed, so every count sits one above the truth, uniformly, now on fields as well as
locals. Nothing can reach zero on a live value, which is the property §9.12 chose and this keeps.

**Which means the new tests have no teeth yet, and that was checked rather than assumed.**
Swapping the store to release-before-retain — the classic way to free the value you are about to
store back — leaves every case in `00owned_fields.ts` passing, because a release cannot reach
zero while the slack is there. They are written as aliasing cases anyway, and run in every model,
because that is exactly what gives them teeth the moment the slack goes.

**One thing this broke in the verifier, worth recording.** The structural half of §9.18 —
"released but never retained" — went from zero findings to **49 files**. All false. An overwrite
hands the count over with `ts.Retain` on the *value* coming in and `ts.ReleaseSlot` on the *slot*,
so the slot never appears in a `ts.RetainSlot` and every field store in the suite looked
unmatched. The check now recognises the hand-over by the store that follows the release. The
lesson is about verifiers rather than about fields: a check that pairs acquisitions and releases
has to know every shape the pairing takes, and adding an insertion point adds a shape.

After that, the verifier reports **two** functions across the whole suite, and both are the same
throwing-`[Symbol.dispose]()` path §9.18 already documented — the cleanup region's own dispose
invoke unwinding past the release that follows it. No new findings from this step.

New test: `test/tester/tests/00owned_fields.ts`, all three models
(`test-compile-00-owned-fields`, `test-jit-00-owned-fields`, `test-jit-rc-owned-fields`,
`test-jit-none-owned-fields`). Repeated overwrite, an alias that outlives the field's reference,
self-assignment, one value shared between two holders, and a field assigned from another field.

Full release suite green: 879/879.

### 9.20 Step 5d: elements own what they hold — and the literal that does not

The direct sibling of §9.19. A `T[]` value is `{ data, length }`, and its release routine walks
the elements of the data block before freeing it (`buildArrayBody` in `OwnershipRoutineLogic`) —
the exact mirror of what `releaseFields` does for an instance. So `arr[i] = x` carried the same
debt `obj.f = x` did, and was likewise a bare `ts.Store`.

**The fix** is a third arm on `isOwningSlot`: `isOwnedElementSlot` — a `ts.ElementRef` whose base
is an `ArrayType` and whose element type owns heap memory. Only `ArrayType`; `ts.ElementRef` also
addresses a `ConstArrayType`, whose data is a static literal nothing releases, and a `StringType`,
whose characters are not references at all. Element access already produces `ts.Load` on a
`ts.ElementRef`, so the store flows through the same assignment path fields do and needed no new
emission code — only the predicate.

**Scoped deliberately.** `push`, `unshift` and `splice` put a value into that same data block
through their own ops rather than through an assignment, and `pop` and `shift` take one back out.
The taking-out half asks the same question a `return` does — give up a reference to a value the
caller is about to hold — so those belong together in one later slice rather than half here.

**These tests do have teeth, unlike §9.19's, and that is the interesting part.** The same
release-before-retain swap that left every field case passing makes `test-jit-rc-owned-elements`
fail outright: the element self-assignment reads back `0` where the field self-assignment still
reads `5`. Reduced to two five-line programs, that asymmetry is not about elements at all.

**What it exposes: an array literal stores its elements without retaining them.** The IR for
`let arr = [kept];` is a `ts.CreateArray(%kept)` with no `ts.Retain` anywhere near it, while the
`ts.ReleaseSlot` at scope exit runs the array's release routine, which releases every element.
The array gives up a reference it never took. A field filled through the assignment path holds
birth + field = 2 and survives a stray release; an element seeded by a literal holds only its
birth reference, so releasing first drops it to zero and frees a live value.

Object literals construct the same way and have the same hole. That makes literal construction,
not arguments or returns, the next thing to take.

> **Correction, made while implementing §9.21.** This section originally called the gap an
> over-release *in waiting*, masked entirely by the slack, and illustrated it with an array going
> out of scope and releasing an element it never retained. That mechanism is wrong: the data
> block has an unconsumed birth reference of its own, so it does not die at scope exit and never
> reaches its elements at all. The real mechanism is that the element is simply one count below
> an equivalent field, and it is *each explicit overwrite* that spends the missing reference —
> the first cancelled by the birth slack, the second going past zero. Which means it was never
> latent: it frees live memory today. §9.21 has the reduced case.

The verifier is unchanged by this step: still the same two files and six retain sites, all the
known throwing-`[Symbol.dispose]()` path, and no new "released but never retained" — the
hand-over recognition added in §9.19 generalised to elements without modification.

New test: `test/tester/tests/00owned_elements.ts`, all three models. Repeated overwrite, an alias
outliving the element's reference, self-assignment, one leaf shared between two arrays, an element
assigned from another array's element, one value reaching two slots of the same array, and
overwriting a single slot inside a loop.

Full release suite green: 883/883.

### 9.21 Step 5e: literal construction, and the first over-release that was already live

§9.20 ended by predicting that array and object literals capture without retaining, and filed it
as a latent problem for after the slack came out. Writing the fix meant reducing the case
properly, and the reduction said something different: it frees live memory now.

**The reduced case.** Two array literals holding one value, each overwritten once:

```ts
let kept = new Leaf(7);
let a = [kept];
let b = [kept];
print("A", kept.n);   // 7
a[0] = new Leaf(1);
print("C", kept.n);   // 7
b[0] = new Leaf(2);
print("D", kept.n);   // 0   <- freed while `kept` still holds it
```

**Why §9.20's account of it was wrong.** That section said the array dies at scope exit and
releases an element it never retained. It does not: the data block carries an unconsumed birth
reference of its own, so its count never reaches zero and its release routine never runs. The
elements are not reached that way at all.

What actually happens is quieter and worse. An element seeded by a literal sits at **one** —
its birth reference only — where a field filled through the assignment path sits at two. Every
`arr[i] = x` releases what the slot held. The first such release is exactly cancelled by the
birth slack, which is why one overwrite looks fine and why §9.19's and §9.20's tests pass. The
**second** release of the same value, through a different literal, has nothing left to spend and
takes it past zero. Two holders and two overwrites is the whole recipe, and it needs no future
change to become reachable.

So the slack was never masking this. It was masking exactly one release of it.

**The fix** is one helper, `mlirGenRetainCaptured`, used at the two places that fill an owning
block in one go instead of through an assignment: the array literal's `ts.CreateArray`, and the
boxed object literal's `ts.New` + `ts.Store`. Both blocks release what they hold when they die,
so both must take a reference to it. A record-shaped value retains through its own routine, which
walks its owning fields, so the boxed case needs one `ts.Retain` on the whole tuple rather than
one per field.

**Still open, and now precisely bounded.** The spread form of an array literal (`[...xs, y]`)
builds its array through `ts.ArrayPush` rather than `ts.CreateArray`, so it keeps the same hole
until §5f takes the mutating ops. An unboxed object literal — one with no methods — stays an
inline const-tuple or tuple, which is the inline-record case §9.19 deferred and §5g will answer.

**The test does have teeth, and each case was checked rather than the file as a whole.** Against
the compiler as it stood, `00owned_literals.ts` returns 3 where 10 is due, 1 where 7 is, 7 where 8
is — six of its seven cases wrong, the seventh being a deliberate control that must pass either
way. A single overwrite is not enough to bite; what bites is one value reaching two slots that are
both later overwritten, whether that is two literals sharing it or one literal holding it twice.

The verifier is again unchanged — same two files, same six sites. It tracks `ts.RetainSlot` and
`ts.ReleaseSlot`, and this step adds neither; the value-form `ts.Retain` is outside what it pairs.
That is a real limit rather than a clean bill of health, and it is worth saying plainly: the check
that would have caught this bug is not the one that exists. A verifier that pairs a construction
site's retain against the owning block's eventual release needs to reason about the block, not
about a slot in a frame, and nothing here does that yet.

Full release suite green: 887/887.

### 9.22 Step 5f: the array-mutating ops

The last of the insertion points that fill an array's data block. `push`, `unshift` and `splice`
put a value in through their own ops rather than through an assignment, so like the literal in
§9.21 none of them took a reference to what they inserted, while the block goes on releasing
every element it holds when it dies. Same bug, same recipe to expose it — one value reaching two
slots that are both later overwritten — and the same fix: retain each inserted value, in
`MLIRCustomMethods` where the three ops are built.

**This also closes the spread literal §9.21 left open.** `[...xs, y]` is not built by
`ts.CreateArray` at all: `mlirGenAppendArrayByEachElement` synthesises a `for..of` calling
`push`, so it inherits push's retain rather than needing one of its own. That is the whole of
what §9.21 deferred on the array side.

**`pop` and `shift` get no counterpart, and that is a decision rather than an omission.** The
block does not release the element it gives up — the size shrinks past the slot, so the release
routine (`buildArrayBody`, which loops to `size`) never reaches it. The reference the block held
simply transfers to the returned value. That leaves the result carrying the same "+1 nobody has
consumed" that every freshly produced value already carries, which is the convention §9.12 chose
and §5h removes wholesale. Pairing a release here instead would free a value the caller is about
to use. So the question §9.20 flagged — what a `pop` and a `return` owe each other — turns out to
be already answered by the existing convention, and needs nothing of its own until the slack goes.

**Still open, and bounded.** What `splice` *deletes* is memmoved over and its references dropped
without a release. That leaks rather than over-releases, so it is inert; and it cannot be fixed at
this level anyway, because the number of elements to release is only known inside the lowering.
It is the first item in this arc that will need a retain or release emitted from `LowerToLLVM`
rather than from MLIRGen, which also puts it outside what the verifier can see.

**One test case had to be strengthened, and only running each case separately found it.**
`spreadLiteralSharesValue` passed on the unfixed compiler with two overwrites: the source array is
itself a literal, so under §9.21 it already holds a legitimate retained reference, and that one
extra absorbed the second release. The case was worthless as written and looked fine. Overwriting
the source as well spends the literal's own reference and puts the two spread copies back on the
hook for theirs — 6 where 13 is due, on the compiler as it stood. The habit that caught it is the
one from §9.21: check each case against the unfixed compiler, not the file as a whole.

The verifier is unchanged again — same two files, same six sites — and for the same structural
reason as §9.21: these are value-form `ts.Retain`s against a block's eventual release, which is
not the pairing it tracks.

New test: `test/tester/tests/00owned_array_ops.ts`, all three models. push, unshift and
splice-insert each sharing a value between two arrays; one array pushed twice with the same value;
the spread literal; `pop` and `shift` as run-path coverage of the transfer; and a single overwrite
after a push as a control.

Full release suite green: 891/891.

### 9.23 Step 5g: inline records — and why arguments and returns needed nothing

Three things were queued for this step: arguments, returns, and the inline-record cases §9.19 and
§9.21 deferred. Checking each before writing anything turned two of the three into no-ops, and
the third into a live over-release.

**Arguments are already borrowed, and that is the right convention.** A parameter's slot is not
marked owned, so passing a heap value neither retains nor releases: the callee borrows for the
duration of the call and the caller's own reference keeps it alive. The hazard worth testing is a
callee that drops every holder of what it was handed, so the sharpest available case was written —
a function passed a value plus the class field, the second holder and the array that all point at
it, dropping all three before reading it. It reads correctly. It has to: every holder that drops
also retained when it took, so the count cannot fall below the number of live holders. Nothing to
do here now; the convention becomes load-bearing at 5h.

**Returns already work, for a reason worth naming.** `return x` releases `x`'s owned slot on the
way out, which balances the retain at its declaration — and what the caller receives is the birth
reference, unconsumed. That is exactly the +1 transfer `pop` and `shift` perform in §9.22, arrived
at from the other direction. Verified through two frames. This is also precisely what 5h has to be
careful about: once the birth reference is consumed by the local's retain, that scope-exit release
becomes the last one and would free the value before it is returned.

**The inline-record case, however, was an over-release, and the reasoning that deferred it was
half wrong.** §9.19 excluded a field of a record held inline because "retaining into a record
nothing releases would leak". The half that does not hold is that an owned local holding a record
*does* release its fields: `ts.RetainSlot` and `ts.ReleaseSlot` on a record-shaped slot go through
the type's own routines, and those walk the fields. So the local retained the field's original
value and released whatever the field held at scope exit, while an assignment in between swapped
that value taking and giving nothing:

```ts
let x = new Leaf(1);
{
    let a = { item: new Leaf(9) };
    let b = { item: new Leaf(9) };
    a.item = x;     // no retain
    b.item = x;     // no retain
}                   // both locals release x at scope exit: 2 -> 1 -> 0, freed
print(x.n);         // 0
```

**The rule is conditional, unlike the class one.** A class or object field always owns, because
the instance is a heap block whose release routine always runs over its fields. An inline record's
field owns exactly when the storage under it owns — so `isOwnedFieldSlot` now recurses through a
`RefType` base into `isOwningSlot`. A parameter's slot answers no, and so does the scratch storage
a literal is built in, which is what keeps construction from leaking.

**Construction needed nothing, and that too was checked rather than assumed.** A literal is built
in scratch storage nobody owns and then copied into the owned local, whose `RetainSlot` retains
the fields on the way in. That balances, which also closes the unboxed-object-literal item §9.21
left open — it was never broken, only unexamined.

**Two of the six test cases had to be reshaped after failing to bite**, the same way §9.22's
spread case did. `recordsInsideAnArray` cannot bite yet at all: the releases would come from the
array's own release routine, and that never runs while its data block still carries an unconsumed
birth reference. It is kept, labelled as coverage of the predicate's element/record recursion
rather than as a counting test. The other needed a third record, because an array holding the same
value retains it legitimately and that reference has to be spent first. Both were found by running
each case against the unfixed compiler individually — three slices running, three times this has
caught a case that passed either way.

The verifier is unchanged once more, same two files and six sites.

New test: `test/tester/tests/00owned_inline_records.ts`, all three models.

Full release suite green: 895/895.

### 9.24 Step 5h, first half: allocations are born unowned, and every function returns +1

Two things happened here. One is the change; the other is that two comments in the tree were
wrong and cost real time before the change could even be designed, which is worth recording
because both were the kind of stale note that reads as authoritative.

**The wrong comments.** `Defines.h` said of the header word "the word is not yet initialized on
allocation - nothing maintains a count". That stopped being true at §9.6. Reading it, and the
sibling note in `getHeapBlockHeaderSize` claiming class instances bypass the header through
`GC_malloc_explicitly_typed`, led to an hour of reasoning from a model in which blocks were born
at zero and none of §9.12-§9.23's arithmetic held. `_MemoryAlloc` settles it in one line — it
stored `1`, with the comment "the block starts owned by exactly one reference" — and the typed
path that would have bypassed the header sits behind `ENABLE_TYPED_GC` and was retired in §9.2.
Both comments are now corrected. The lesson is narrow and practical: in this area, read the
emitting code, not the note describing it.

**The change.** `_MemoryAlloc` now writes **0**. A block starts unowned; whoever first takes it -
a local's declaration, a field or element store, a literal capturing it, a push - is what brings
the count to one, and that owner's release is what takes it back to zero and frees it. That is
the slack §9.12 deliberately left, and every insertion point that had to exist before it could
come out now does (§9.19-§9.23).

Being born at zero also gives the remaining mistakes a benign shape at the boundary: a release of
a block nobody ever took underflows to all-ones, which is `HEAP_BLOCK_IMMORTAL`, so the block
leaks instead of being freed out from under a live reference.

**The companion change, which is not optional.** The scope exit at a `return` releases every
owned local in the frame, and the returned value is very often held by one of them. Once
allocations are born unowned that release is the last one, so `return x` after `let x = new C()`
would free the value on the way out. The value is therefore retained before the scope exit.
Retaining the value rather than trying to identify which local holds it is what makes this work
for `return h.item`, `return arr[0]` and `return cond ? a : b` alike — and it establishes a
uniform convention: **every function returns +1**, the same transfer `pop` and `shift` perform.

**What this does and does not achieve, stated exactly.** For arrays, strings and boxed object
literals it is a real removal of the slack: `let a = [1, 2, 3]` now takes its data block to one
and back to zero, and the block is freed. For **class instances it is currently neutral**, and
that was verified rather than assumed. `new C()` is a call to a compiler-generated `C..new`, so
the return retain applies to it too and hands back exactly the +1 the birth reference used to
provide. The check was the release-before-retain swap from §9.20: if class instances had lost
their slack, that swap would now free live memory and the owned-* tests would fail. All 35 still
pass, so they have not gained teeth yet.

**Which names the second half precisely.** The convention is now uniform - calls hand out +1 - so
what remains is for the receiving sites to *consume* it: a local declaration, a field or element
store, a literal capture or a push whose incoming value is already +1 should not retain again.
The classification fails safe in the direction of not knowing: an unrecognised producer is treated
as +0, retained, and leaks.

> **Correction, made while implementing §9.25.** This paragraph originally listed
> `ts.CreateArray`, `ts.New`, `ts.ArrayPop` and `ts.ArrayShift` as +1 producers alongside calls.
> That was carried over from the model in which allocations were born at one. They are not:
> once a block starts unowned, `ts.CreateArray` and `ts.New` hand back a value at **zero**, and
> their receiver's retain is exactly right. Only a call that retained on the way out, and
> `pop`/`shift` transferring a reference the data block held, are genuinely +1.

The dangerous direction is the opposite one, and it has a specific name: a call that returns a
heap value **without** passing through the return path patched here - a runtime or builtin helper
such as string concatenation, or a function imported from a module built before this convention.
Treating those as +1 would skip a retain that was never performed and free live memory. So the
second half cannot simply say "calls are +1"; it has to distinguish a user function with a
generated return from an external one. That is the next slice, and it is the first in this arc
where the failure mode is a premature free rather than a leak.

Full release suite green: 895/895.

### 9.25 Step 5i: consuming the transferred reference, and the tests finally bite

§9.24 left the convention uniform - every function returns +1 - and the leak that came with it:
a receiver that retains an already-owned value is one owner above the truth. This closes that for
the case that was still leaking on every program, and in doing so it is the first slice where the
counting is load-bearing rather than slack.

**First, a correction to §9.24's own list of producers.** That section named `ts.CreateArray`,
`ts.New`, `ts.ArrayPop` and `ts.ArrayShift` as +1 alongside calls. That was written from the old
model. Once allocations are born unowned, a freshly allocated block is at **zero**, so
`ts.CreateArray` and `ts.New` produce +0 and their receiver's retain is exactly right. What is
genuinely +1 is narrower: a call that retained its result on the way out, and `pop`/`shift`, which
hand over a reference the data block was holding.

**What is marked, and what deliberately is not.** Only `new C()` is marked here, at the one place
that builds the call and therefore knows the callee is the generated `C..new` - which goes through
the retaining return path. Nothing infers ownership from an operation merely being a call. That
restraint is the whole safety argument: a runtime or builtin helper, or a function imported from a
module built before this convention, hands back a heap value with no retain behind it, and
consuming one of those would skip a retain nobody performed and free live memory. Answering "not
owned" for something that was in fact owned only leaks, so the unknown case falls the safe way.

**The four receivers all consume**: a local declaration, a field or element store, a literal
capturing a value, and `push`/`unshift`/`splice`. The release side is untouched in every case -
what a slot was holding still has to be given up, whoever the incoming reference came from.

**The declaration case needed the verifier extended, and the first attempt silenced it instead.**
A consumed local has no `ts.RetainSlot`; the declaration itself is the acquisition. Adding the
slot to the "was it ever retained" set stopped the false "released but never retained" reports -
but the every-path check iterated the `ts.RetainSlot` list, so it quietly stopped running for
exactly the locals whose release now matters most. The sweep went from two findings to zero, which
looked like an improvement and was a regression. The pass now collects *acquisitions* - a slot
paired with the operation to blame - from both shapes, and the two known
throwing-`[Symbol.dispose]()` findings are back. Third time this arc that a new insertion point
taught the verifier a new shape, and the first time the symptom was silence rather than noise.

**The tests have teeth now, and this is the milestone §9.19 was waiting for.** Re-running the
release-before-retain swap: before this slice it failed nothing; now it fails
`test-jit-rc-owned-locals`, `test-jit-rc-owned-fields` and `test-jit-rc-owned-elements`.
`00owned_fields.ts` was written at §9.19 with a header explaining that it guarded shape and run
path but not counting, and asking for exactly this experiment to be re-run once the slack went.
It now catches the bug it was written for.

**Still leaking, and now the whole of what is left.** Every +1 that is not consumed: the result of
an ordinary function call assigned anywhere (`let y = f()` retains a value `f` already retained),
a discarded `pop`, and a returned value the caller drops. Closing those needs the producer
classification to extend past `new`, which is the risky work this slice deliberately did not do.

Full release suite green: 895/895. Verifier: two files, six sites, unchanged.

### 9.26 Step 5j: the transfers that can be settled, and the call that cannot

5j was meant to extend the producer classification past `new` to ordinary calls. Half of it is
here; the other half turned out to need a different shape than "one more marking site", and this
section records why rather than shipping a heuristic for it.

**What landed: `pop` and `shift`.** These are the compiler's own operations with known semantics,
so there is nothing to classify. The data block gives up the element without releasing it - the
size shrinks past the slot, so its release routine never reaches it again - which hands the
block's reference to whoever receives the result. Marking those two results owned lets a receiver
take that reference over instead of adding one, so `let x = arr.pop()` is now one owner rather
than two.

**What did not, and the specific reason.** Every function retains its result on the way out
(§9.24), so `let y = f()` is one owner above the truth as well - and that is the dominant
remaining leak. Marking it needs to know that *this* callee retains, and three separate things
stop that being answerable where the call is generated:

- **Not every return path retains.** The retain sits in the return *statement*, but a concise
  arrow body (`() => expr`) reaches `mlirGenReturnValue` down a different path, and so does
  `yield`. Marking a function whose body takes one of those would consume a reference nobody
  took.
- **The callee may not exist yet.** MLIRGen emits `ts.CallIndirect` on a symbol reference; the
  callee's `FuncOp` need not have been created when the call site is generated, so a lookup would
  answer differently depending on declaration order. Always-safe, since the unknown case falls to
  +0 and leaks — but silently order-dependent, which is worse than not doing it.
- **External callees look identical.** A `declare`d function, one imported through `__decls`, or
  a runtime helper has no retaining return at all. These are the cases where being wrong frees
  live memory.

The shape that answers all three is a pass after MLIRGen, when every `FuncOp` is present and each
one's return paths can be inspected rather than predicted. That is a different piece of work from
the marking sites of §9.25, and it is the right place to stop this slice.

**A test that was worthless, caught by the habit rather than by luck.** The first version of
`00owned_transfer.ts` passed with a deliberate over-release injected into `pop` — because a freed
block keeps its contents until something else claims them, so reading through the receiver read
the right answer out of freed memory. It caught nothing the existing tests did not already catch.
Each case now calls a `churn()` helper between the transfer and the read, allocating enough
same-shaped blocks to land on the freed one, and it then fails against that injection as it
should. This is the same trap as §9.24's memory measurement: an experiment that confirms what you
expected is worth less than one you tried to break.

Worth recording separately: injecting the *opposite* mistake - treating every `ts.Load` result as
already-owned, so receivers stop retaining - fails six of the ownership tests. The suite does
detect premature frees broadly now, which is the property that matters most from here on.

Full release suite green: 899/899. Verifier: two files, unchanged.

### 9.27 Step 5k: consuming an ordinary call's result

The dominant remaining leak, and the first piece of this arc that is a pass rather than a marking
site. §9.26 gave the reason: deciding whether *this* callee retains its result cannot be settled
where MLIRGen builds the call. All three obstacles it named dissolve once every function exists,
so the work moves to a module pass that runs straight after MLIRGen.

**It looks rather than predicts.** A function counts as returning owned only when every
`ts.ReturnVal` of a heap-owning value in it is preceded by a `ts.Retain` of *that same value* in
the same block. Anything else is left alone: a callee with no body, a call through a function
value with no single callee to inspect, a generator (vetoed outright, since what its caller
receives is the generator object rather than anything those returns produce). Those callers keep
retaining, which leaks rather than freeing something live. The whole design puts the uncertain
case on the leaking side.

Each exclusion was checked on emitted IR rather than assumed: a `declare`d callee keeps its
`ts.RetainSlot`, an indirect call through a parameter keeps its own, and a local function's call
is marked and its receiver's retain removed - with the declaration marked as the acquisition so
the verifier can still pair the release that follows.

**A prerequisite that had to land first.** A concise arrow body (`() => expr`) returns without
going through the return statement, so it never got §9.24's retain and would have been excluded
from the convention entirely. It has one now. This is not fixing a dangling read - there is no
scope exit there to free anything - it is the convention itself: callers cannot be told "calls
return owned" while one shape of function quietly returns borrowed.

**What the check actually excludes, measured rather than guessed.** Removing it raises the number
of marked call sites across the suite from **469 to 497**, so it is not a formality - it excludes
28 real calls, reaching 92 distinct functions. Following one of them to its IR explains all of
them: the retain is emitted on the value the return statement *evaluated*, while
`mlirGenReturnValue` then casts that value to the declared return type, and it is the cast result
that `ts.ReturnVal` carries. So a return needing a cast retains the wrong value. It is benign -
the reference lands on a value nobody releases, which leaks - and the pass is right to exclude
those functions, but the fix is to apply the return-type cast before the retain rather than
inside the return. That is a separate slice.

**An honest coverage note.** Disabling the retain check entirely - marking every function with an
owning return, sound or not - still leaves the suite at 903/903. The guard is reasoned rather than
test-validated, because no test currently exercises a callee that returns a heap value without
retaining it. The *other* direction is covered: over-consuming a call result, by removing every
receiver retain instead of one, segfaults `00owned_call_results.ts` outright.

New test: `test/tester/tests/00owned_call_results.ts`, all three models - a result outliving the
local that received it, shared between two locals, stored into a field, captured by an array
literal and by `push`, forwarded through two frames, returned from a method, and returned from an
arrow function. Each calls `churn()` between the last release and the read, for the reason §9.26
learned the hard way.

Full release suite green: 903/903. Verifier: two files, unchanged.

### 9.28 Step 6: the allocator flips, and the first measurement that means anything

`needsGCRuntime()` returned true for `rc` from §9.6 onward, so under `-mm=rc` Boehm was still
allocating and still collecting behind the counts. That was the right call while the insertion
points were being built one at a time - a missing release stayed an inert leak - but it meant no
memory number taken under `rc` said anything about reference counting. The predicate now names
exactly one model, `gc`. Under `rc` the program allocates from `malloc`, frees through `free`, and
links no libgc; what the counts miss now leaks, and shows.

**What the flip cost: one crash, and it was not RC's.** `test-jit-rc-disposable-scopes` failed
immediately - and the same file failed under `-mm=none` too, on a build with none of this work in
it. A `using` inside a `catch` clause, at `-O3`, on Win64. Narrowed to a bare `new` inside a
handler whose result is used there.

The chain: a handler is its own funclet, and every call inside one has to carry a `funclet`
operand bundle naming its pad. `Win32ExceptionPass` stamps them correctly - verified on emitted IR
at `--opt --opt_level=0`, where the allocation and its zero-fill both carry the bundle. LLVM then
rewrites `malloc` + `memset(0)` into `calloc`, and builds the replacement **without carrying the
operand bundles over**. WinEHPrepare stops seeing the instructions after it as part of the
funclet, and the handler is emitted as a bare prologue: no body, no `catchret`. It faults the
moment it runs.

Each step was checked rather than reasoned about. Stock `opt -O3` on our own pre-optimisation IR
reproduces the dropped bundle, so the defect is LLVM's, not the pass's. `llc` on that output shows
the empty funclet directly; hand-adding the bundle back to the `calloc` restores the full handler
body and its `catchret`. `-print-after-all` names the pass: **DSE's `tryFoldIntoCalloc`**, not
SimplifyLibCalls, which is why emitting the zero-fill as `llvm.memset` rather than a `memset` call
fixed `none` but left `rc` still crashing.

**The fix is to ask for what we mean.** A zeroed block is now requested as `calloc` outright, so
there is no pair left for that fold to rewrite; GCPass maps it onto `GC_malloc` - rewritten rather
than renamed, since the arity differs - and drops the now-unreferenced declaration. The wasm fork
has no `ts_calloc` and no Win64 funclets, so it keeps the two-step form, as the intrinsic.

Only `gc` was ever safe here, and by accident: GCPass deletes the zero-fill, so the pattern the
fold looks for never survived to LLVM. That is why the new test needs its non-`gc` variants to be
worth anything, and the teeth were confirmed the usual way - with the fix disabled and rebuilt,
`00alloc_in_catch.ts` faults under `-mm=rc`.

**Found and left alone:** a try/catch nested *inside* a catch clause crashes with no allocation in
it at all, in every memory model and at every optimisation level. Unrelated to this, and older
than it; the case is called out in `00alloc_in_catch.ts` rather than covered by it.

**The measurement, at last.** Peak working set, AOT executables:

| program | gc | rc | none |
|---|---|---|---|
| allocation churn, 1M iterations, `-O0` | 4.2 MB | **3.8 MB** | 172.8 MB |
| `raytrace.ts`, `-O3` | 4.1 MB | **129.5 MB** | 106.3 MB |

The first line is what this whole arc was for: a value bound to an owned local is allocated,
released and reclaimed a million times over, flat, without a collector - marginally below Boehm.
(At `-O3` that loop vanishes in all three models, allocations and release calls together, which is
its own small piece of good news about the generated code.)

The second line is the honest other half. `raytrace.ts` reclaims **nothing**: it is built almost
entirely out of `return new Vector(...)` used inline - `Vector.plus(Vector.times(k, a), b)` - so
every intermediate is a call result passed straight as an argument and never bound to any slot.
Each carries the +1 its return retained (§9.24) with no owner to give it back. That is item 5l,
discarded temporaries, and this measurement reclassifies it: not a nicety at the end of the list
but the dominant leak in ordinary expression-shaped code. `rc` sitting *above* `none` is the same
story seen from the other side - the release calls are uses, so fewer dead allocations get
optimised away, and nothing is reclaimed to pay for it.

New tests: `00alloc_in_catch.ts` in all four variants, plus `-mm=none` variants of `03disposable.ts`
and `04disposable.ts` - the file that caught this had no non-`gc` coverage of its own.

Full release suite green: 909/909. Verifier: two files, unchanged.

### 9.29 A `try`/`catch` inside a `catch` clause

Not RC's, and older than any of this - §9.28 only found it because a test written for that step
tried to allocate inside a nested handler. It crashed in every memory model, at every optimisation
level, with nothing allocated in it at all. A try nested in a try *body* or in a `finally` always
worked; only the catch clause was affected, which is why nothing had caught it.

Two independent bugs, the second visible only once the first was fixed. Both are confirmed
individually load-bearing by disabling each alone and rebuilding.

**1. The catch-variable search descended into the nested try.** `TryOpLowering` finds its clause's
`ts.CatchOp` by walking the catches region, and the walk went straight through a nested `ts.TryOp`
into that try's own catches. It picked up the *inner* clause's catch, so the outer try's landing
pad got its RTTI type filter from the wrong clause. The debug build says this outright - the
`assert(!catchOpPtr)` on the second catch found - which is worth remembering: the release build
faulted with no diagnostic at all and no usable stack, and the debug build named the line in one
run without a debugger. The walk is now pre-order and skips a nested try, because `skip()` only
prunes regions still to come and the default post-order has already visited them.

**2. A catch clause can be ended twice over.** §9.14 has a `throw` leaving a catch clause end that
catch ahead of itself. A nested `try`'s throw is such a throw, so the enclosing clause is already
ended by the time the outer try emits its own end-of-catch marker - and the surplus marker became
the region's `end` instruction, which is where the catchret goes, so it survived into the emitted
code. `__cxa_end_catch` is an Itanium marker with no Win64 counterpart, so it failed to link
(`Symbols not found: [ __cxa_end_catch ]`). Win32ExceptionPass now skips past end-of-catch markers
while looking for a region's end and removes them, which handles any number of them, and removes
an unclaimed one found with no region open at all.

New test `test/tester/tests/00nested_catch.ts`, four variants: a catch in a catch, three levels
deep, an inner clause that is never taken, a nested try with a `finally` of its own, the whole
thing inside a loop, an inner clause that throws past the outer one, and the try-in-body and
try-in-finally shapes that always worked, kept alongside so a fix here cannot quietly break them.
`00alloc_in_catch.ts` regained the nested case it had to leave out.

**The direct test of bug 1 is a type test, not a value test.** `outerFilterIsItsOwn` throws a
string caught by the outer clause and an int caught by the inner, so an outer pad carrying the
inner clause's filter would not catch the string at all.

**Found while writing these, and deliberately NOT fixed: reading a catch variable's value is
broken on its own.** No nesting involved. `try { throw 2 } catch (v: int) { t = v }` reads 0 rather
than 2 - but only in a module that throws just that one type; adding the other clauses of
`00try_catch.ts` to the same file makes it read correctly, which is why that test passes and this
went unnoticed. Reproduced in every model, and at `-O3` a separate variant of the same shape reads
0 where `-O0` reads 3. `00nested_catch.ts` therefore checks which clause runs and in what order and
never reads a catch value; nothing there should be made to depend on a broken feature. A third bug,
in the same subsystem, still open.

Full release suite green: 913/913.

### 9.30 Step 5l: giving back the temporaries

§9.28's measurement made this the priority: `raytrace.ts` reclaimed **nothing** under `-mm=rc`.
It is built almost entirely out of `Vector.plus(Vector.times(k, a), b)`, so nearly every
allocation it makes is an intermediate passed straight as an argument and never bound to
anything - carrying the +1 its return retained (§9.24) with no owner to give it back.

**Consumption is now recorded rather than implied.** Every receiving site (§9.25) answered
`producesOwnedReference` by *not* emitting a retain, which left no trace: after §9.27 erased a
receiver's retain, a consumed call and a call nobody received looked identical. A second
attribute, `OWNED_RESULT_CONSUMED_ATTR_NAME`, is set wherever a receiver takes a reference over,
and its absence is what identifies a discarded temporary.

**The release goes at the end of the producer's own block.** That is a temporary's natural
lifetime - the enclosing statement, or one iteration of a loop body - and, more to the point, it
is unconditionally after every use in that block. Placing it after the last *user* looks tighter
and is wrong: the receiver of `let x = <T>f()` retains the result of the **cast**, not of the
call, so the call's last user is the cast and a release put there runs before that retain and
frees the value out from under it. Two shapes are refused and left leaking as before: a user
outside the producer's block, and a user that is a terminator (a value handed to a successor as a
block argument is still live past the release point).

*Amended at §9.33: "the end of the block" was taken literally, and `ts.ReturnVal` is not a
terminator - so in any function ending with a `return` the release landed after it, and the
affine lowering dropped it as unreachable. 36 of `raytrace`'s 47 releases never ran. The rule is
now "before the first op that ends the block's execution", which is what MLIRGen's own scope exit
had been doing all along.*

**A second gap had to close before any of this fired.** §9.27 classified a function as returning
owned only when every return was preceded by a `ts.Retain`. But there are two ways to hand back a
reference: retain one, or forward one already held - and `return new C()` consumes the instance's
own +1 rather than adding a second (§9.25), so there is no retain to find. Every
`static times(...) { return new Vector(...) }` was therefore unclassified, which is most of what
expression-shaped code is built from. A return whose value comes from an `OWNED_RESULT` operation
now counts too.

**What it reclaims**, peak working set, AOT:

| program | gc | rc before | rc after | none |
|---|---|---|---|---|
| `raytrace.ts`, `-O3` | 4.1 MB | 129.5 MB | **79.3 MB** | 106.7 MB |
| nested call temporaries, 500k iterations, `-O0` | 4.2 MB | — | **3.8 MB** | 188.0 MB |
| object literal returned as an interface, `-O0` | 4.2 MB | — | 41.5 MB | 42.3 MB |

The second line is the shape this step is about, and it is now flat. The third is what `raytrace`
still leaks: an **object literal returned through an interface** reclaims essentially nothing.
Arrays and strings were checked the same way and both reclaim (3.9 MB against 42.6, and 3.8 MB
against 27.2), so the remaining leak is specific to the literal/interface path - the boxed literal
or the clone an interface cast makes - and is the next thing to look at, not another temporaries
problem.

*Read with §9.31: it was the boxed literal, an interface owned nothing at all - and the 79.3 MB
above was partly memory freed while still referenced, so it is not a figure to compare against.*

New test `test/tester/tests/00owned_temporaries.ts`, four variants. **Teeth, measured per case
with two separate probes** rather than assumed:

- releasing consumed results as well as discarded ones is caught only by the loop case, because
  an end-of-block release lands after every read in that block - a useful reminder that this
  particular perturbation cannot reach most of the file;
- releasing immediately after the producer instead of at end of block - the ordering the whole
  design rests on - is caught by 4 of the 8 cases at `-O3` and by 6 of 8 at `-O0`.

Two cases cannot fail loudly and are kept knowingly: `discardedResult` has nothing that reads the
discarded values, and `temporaryKeptByCallee` is balanced by push's own retain either way. The
first version of `usedAsArgumentAndBound` passed under both probes for the wrong reason - a freed
block still held its old value - so `argumentReadAfterCalleeAllocates` was added, where the callee
allocates before reading its arguments and a block freed early is overwritten before the read.

Full release suite green: 917/917.

### 9.31 Step 5n: an interface is an owner

`MLIRTypeHelper::ownsHeapMemory` answered no for `InterfaceType`, with a reason that was true as
far as it went: an interface type carries only a name, so the layout behind its `this` pointer is
not recoverable from the type. Nothing an interface held was ever released - not the block a
literal is boxed into, not the class instance behind a cast - which is where `raytrace`'s
remaining leak sat (§9.30).

The layout does not have to be recoverable from the type, because **an interface has exactly the
problem an `any` box has, and the same answer was already in the tree**. An interface value grows
a third word holding the runtime type tag of whatever `this` points at (`INTERFACE_TYPE_INDEX`),
and its release and retain routines read that tag's descriptor and call the concrete type's own
routines - `releaseViaDescriptor`/`retainViaDescriptor` unchanged from what §9.6 built for `any`
and for tagged unions. Two properties make the tag safe to rely on:

- **every interface value in the program is built by one op.** A cast from a class, a cast from
  an object literal, and even `null`/`undefined` as an interface all reach `ts.NewInterface`
  (`CastLogicHelper`), so there is no path that leaves the slot undefined. Where `this` owns
  nothing the tag is null, which states "nothing to release" positively, exactly as a null
  descriptor slot does.
- **descriptors are keyed by the concrete type, not by the `typeof` name.** Every object literal
  reports the name "object"; `getOrCreateTypeDescriptorName` already hashes the type into the
  symbol, so two literals of different shapes get their own records.

The interface value going from 16 to 24 bytes is why `-mm=none`'s numbers in this section are
about 8% above §9.30's.

**Two dormant omissions became live over-releases the moment an interface became an owner, and
both were freeing memory that was still in use.** Neither is a consequence of this slice; both
were reachable before it and simply had nothing that walked the block.

1. **`castTupleToInterface` allocates a block and fills it from a literal without retaining what
   the literal holds** - the debt §9.21 describes, which the object-literal boxing path
   (`MLIRGenExpressions`) pays and this one did not. `{ start: pos, dir: rd }` cast to an
   interface produced a block holding two `Vector` references it had never taken.
2. **`retainInsertedElements` skipped its retain for an already-owned value without recording the
   consumption** - the one receiving site that did not. Once §9.30 began releasing what nothing
   consumed, `arr.push(new C())` released the instance at the end of the pushing block, freeing
   an element the array still held.

Both are invisible for as long as the block that builds is also the block that reads, since
§9.30's release goes at the end of the producing block - which is why §9.30's own tests missed
them. `function add() { store.push({ v: new Vec(42) }) }` read from `main` prints `999` on the
commit before this one and `42` after it.

**Item 5m, cast before retain, is done here** because without it this slice does not fire. A
return retains the value the return expression evaluated, and `mlirGenReturnValue` then casts -
so `return { start: p, dir: d }` against a declared interface retained the *tuple*, and the heap
block the cast allocates got no reference at all. The cast now happens first
(`castToDeclaredReturnType`), before both the retain and the scope exit.

That surfaced a **placement bug in §9.30 that had nothing to do with interfaces**: a discarded
temporary is released at the end of its block, and in a generator that block can contain a
`ts.StateLabel` - a resume point the state machine re-enters. The end of the block is then
reachable on a path that never ran the op producing the value, which appears as a dominance
failure in the affine lowering (`00iterator_bug.ts`) rather than as anything the pass could see
while the generator was still one block. `allUsesReleasableInOwnBlock` now refuses a value with a
state label after its definition.

**Measured, AOT peak working set:**

| shape | `gc` | `rc` | `none` |
| --- | --- | --- | --- |
| literal boxed as an interface, bound in a loop | 4.2 | **4.2** | 42.6 |
| interface temporary passed as an argument (`raytrace`'s shape) | 4.6 | **3.8** | 114.6 |
| instance method returning a class | 4.2 | 31.9 | 31.9 |
| `raytrace.ts`, `-O3` | 4.6 | 114.2 | 114.5 |

The shapes this slice is about are now flat. **`raytrace` is not, and its number is worse than
§9.30's 79.3 MB - which was not a real number.** The over-release in (1) above was freeing
`Vector`s that boxed `Ray` literals still pointed at, and memory that is handed back while still
referenced counts as reclaimed. Correctness cost 35 MB here, and the honest reading of §9.30's
figure is that part of it was never earned.

What `raytrace` leaks now is item 5o below: it is built out of **instance-method** calls, and
§9.27 classifies a callee only when the call names it through a `ts.SymbolRef`. A method reached
through `GetMethod`/a vtable answers empty and is left alone, so its result's +1 is never
consumed - visible on its own as `rc` and `none` both at 31.9 MB in the table above.

New test `test/tester/tests/00owned_interfaces.ts`, four variants, eight cases, each building in
one block and reading in another with `churn()` between - the arrangement §9.30's tests lacked
and the reason those two over-releases survived it. **Teeth measured per fix**: releasing the
interface immediately after construction, dropping the boxing retain, and dropping the push
consumption marking each break the test at both `-O0` and `-O3`.

Full release suite green: 921/921. Ownership verifier unchanged at its two standing findings.

**Found and not fixed:** `--di --opt_level=0` fails to emit LLVM IR for any reference-counted
program ("DISubprogram attached to more than one function") - the generated `tsrel_`/`tsret_`
routines inherit the debug scope current when they were generated. Pre-existing, reproduces on
`00owned_temporaries.ts` and `00interface.ts`, and on no test-suite variant.

### 9.32 Step 5o: what a method call names, and what it does not

§9.31 measured an instance method's result reclaiming nothing and filed 5o to fix it. The fix
that landed is much smaller than the item, and the reason is worth recording, as is the thing
that turned out to be the actual leak.

**The half that is safe.** `calleeNameOf` resolved a callee only through `ts.SymbolRef`, so it saw
a plain function and nothing else. A method call is not shaped like that: `obj.m(x)` builds a
bound function and splits it apart again, `GetMethod` for the code and `GetThis` for the receiver,
so the callee sits a step further back. It now looks through that, for exactly the chains the
dialect's own canonicalizer (`SimplifyIndirectCallWithKnownCallee`) already rewrites into direct
calls — `ts.ThisSymbolRef`, and `GetMethod` over either a `ts.ThisSymbolRef` or a
`CreateBoundFunctionOp` naming a function. That those chains name one callee is not a new
judgement; it is the one the canonicalizer has been making all along.

**The half that pays, and does not hold.** That safe half is worth almost nothing here, because
an ordinary instance method does *not* take it. Of `raytrace`'s method references, 32 are
`ts.ThisSymbolRef` — every one a constructor, so `void`, so nothing to own — and the 9 that
return values are all `ts.ThisVirtualSymbolRef`. A method on a class with a vtable is dispatched
through it whether or not anything overrides it.

A virtual call's identifier names the declaration the call was written against, not what the
runtime class put in the slot, so reading it as the callee would consume a reference an override
may never have taken — the one mistake in this arc that frees live memory rather than leaking.
`private` looks like it settles that: TypeScript forbids overriding a private member, all 9 of
`raytrace`'s sites are private, and marking them classified took the file from 114.2 MB to
111.2 MB with the suite green.

**It was reverted, because the guarantee is not true of this compiler.** This program:

```ts
class Base { private tag(): number { return 1; } public get(): number { return this.tag(); } }
class Derived extends Base { private tag(): number { return 2; } }
```

is rejected by TypeScript and accepted here, and `new Derived().get()` prints `2` — the
redeclared private method takes the base's vtable slot and overrides it. So `private` is not a
single-target property in tslang, and a rule resting on it would have been sound only by
accident. Making it sound needs the callee's whole override set, which is a hierarchy question
the pass cannot see and MLIRGen cannot close cross-module; that is 5o's real cost, and it buys
2.6% of `raytrace`.

**What is actually left is not method dispatch at all.** With interfaces owning (§9.31) and
static calls classified (§9.27), the remaining allocation in `raytrace` is the **capture box of a
closure**. `ownsHeapMemory` names function types among the deliberate exclusions, for a reason as
true as the interface one was: the box is heap-allocated but its type does not appear in the
function type, so there is nothing in the type to walk. `getNaturalColor` builds `addLight` per
ray, and nothing ever gives that box back.

| shape | `gc` | `rc` | `none` |
| --- | --- | --- | --- |
| closure created per call, in a loop | 4.2 | **76.8** | 77.8 |
| private instance method returning a class | 4.2 | 31.9 | 31.9 |
| `raytrace.ts`, `-O3` | 4.6 | 113.8 | 114.5 |

That is the same shape of problem §9.31 solved, and the same answer is available: an interface
could not name what was behind its `this` either, and now carries a tag beside it. A closure
value is a pair the same way an interface is. Filed as 5p, and it is where the next slice should
go rather than 5o.

Full release suite green: 921/921.

### 9.33 Step 5p: a closure owns its capture box — and the release that never ran

Two things landed together here. The second was found while measuring the first, is much larger
than it, and is not about closures at all.

**A closure's capture box.** `ownsHeapMemory` excluded function types for a reason with exactly
the shape the interface exclusion had (§9.31): the box is heap-allocated but its type appears
nowhere in the function type. So the same answer applies. A bound or hybrid function value grows
a third word carrying the runtime type tag of its `this` (`CLOSURE_TYPE_INDEX`), and release and
retain go through it — `releaseViaTagBesideThis` is now shared by both cases, which differ only
in which slot the tag sits in.

The tag is what separates the two kinds of value that share this representation. **A closure owns
its capture box; a bound method's `this` is an object that belongs to somebody else**, and
`obj.m` must not take a reference to `obj`. Only the closure built over captured variables is
marked (`OWNS_CAPTURE_ATTR_NAME`, set where MLIRGen knows the difference); everything else carries
a null tag and its release costs a call that does nothing. As with `NewInterface`, one op builds
every such value, so no path leaves the slot undefined.

**The release that never ran.** Measuring that change showed no movement at all, and the reason
was not in the closure work:

> **`ts.ReturnVal` is not a terminator.** The scope-exit releases and `ts.Exit` follow it in the
> same block. But the affine lowering turns it into a branch to the exit block, so anything
> appended *after* it is dropped as unreachable — and §9.30 appends its releases at the end of the
> block.

Any function whose block ends with a `return` has that shape, which is most of them. **In
`raytrace.ts`, 36 of 47 releases were being discarded on the way to affine**: §9.30 had been
largely inert for exactly the code it was written for, and the tests never saw it because a
release that vanishes only leaks. MLIRGen's own scope-exit releases never had the problem — it
emits them before building the return, which is the model the pass now follows: insert before the
first `ts.ReturnVal`/`ts.Return`/`ts.Exit` after the definition, else at end of block. 46 of 47
survive now.

Releasing before the return is right for a discarded temporary by definition — what is being
returned was consumed by the return's own retain, so it is not in this set.

That fix also had to be paid for: it made releases survive in **generators**, where a position
plainly after a definition may not be on the path that reaches it once the state machine cuts the
block at every resume point. Three generator tests failed with dominance errors. §9.30's narrower
`ts.StateLabel` guard is not enough, so whole generators are now excluded from the discarded
release, the same way `functionReturnsOwned` already excludes them and for the same reason. Their
temporaries leak.

| shape | `gc` | `rc` | `none` |
| --- | --- | --- | --- |
| closure created per call, in a loop | 4.2 | **46.9** | 75.1 |
| interface temporary as an argument | 4.2 | 3.8 | 114.6 |
| `raytrace.ts`, `-O3` | 4.2 | **103.4** | 114.5 |

`raytrace` 113.8 → 103.4, of which the release placement is 4.3 MB and the capture box 6.1 MB.

**What the closure benchmark still holds is not the box.** It is down from 77.8 MB but not flat,
and the rest is the *captured variables themselves*: a parameter or local that a closure captures
is given a heap cell of its own so the box can point at it, and nothing frees those. Two cells per
call in that benchmark against one box, which is the ratio the number shows.

The same allocation is behind a **use-after-free that predates all of this**: a captured object
carried out of its frame is released by that frame's own scope exit.

```ts
function makeAdder(k: number) { let bump = new Vec(k); return (v: Vec) => v.x + bump.x; }
```

`makeAdder`'s scope exit releases `bump`, and the returned closure's box still points at it —
`ts.ReleaseSlot` on the captured local, emitted by MLIRGen, nothing to do with this slice. Filed
as 5q. `00owned_closures.ts` deliberately captures a number in its two escaping cases so that this
older bug does not mask what they are there to check. (Closed in §9.34, which also revised what
the cells cost: measured against a benchmark that survives `-O3`, they were the whole of
`raytrace`'s remaining leak, not a share of it.)

New test `test/tester/tests/00owned_closures.ts`, four variants, six cases. **Teeth**: releasing
the closure immediately after construction breaks it at both `-O0` and `-O3`. Note that
*disabling* the tag does not break it and cannot — that is the leaking direction, which no test
can see.

Full release suite green: 925/925. Ownership verifier unchanged at its two standing findings.

### 9.34 Step 5q: a captured variable's cell is owned

A variable a closure captures by reference does not live in its frame. Its storage is a heap
block of its own — call it a **cell** — so that the frame and every closure over it read and
write the one variable (`ALLOC_CAPTURE_IN_HEAP`, `VariableOpLowering`). The box holds the cell's
address; the frame holds the same address in the slot the variable's name resolves to.

Nothing owned that cell. The frame's scope exit released the *value in it* and the cell itself
was never freed, which is two failures in one:

```ts
function makeAdder(k: number) { let bump = new Vec(k); return (v: Vec) => v.x + bump.x; }
```

`makeAdder`'s exit freed the `Vec` while the returned closure still pointed at the cell holding
it — a use-after-free older than any of this work — and leaked the cell on the way out.

**The cell has owners now**, and there are exactly two kinds: the frame that declared the
variable, and each capture box that captured it. So:

- A cell is **born owned** (`VariableOpLowering`, under `-mm=rc`). That is the opposite of a
  value block, which is born unowned because a receiver is about to take it (§9.24) — a cell has
  no receiver, and its creator is its first owner. The one heap variable this must *not* apply to
  is the box itself, whose `captured = true` means only "allocate in the heap"; `CaptureOpLowering`
  marks it `CAPTURE_BOX_ATTR_NAME` so the two cannot be confused.
- **The box takes one too**, at the capture site — and only where a `ts.Variable`/`ts.Param` was
  actually marked captured, because that marking is what makes storage a cell. Retaining a stack
  slot would write a count into the frame word in front of it.
- **A scope exit releasing a captured local emits `ts.ReleaseCell`**, not `ts.ReleaseSlot`. The
  frame is giving up the cell; the value inside it goes when the last owner does.

`ts.RetainCell` / `ts.ReleaseCell` are new ops for that difference — the existing pair addresses
storage in order to reach the value in it, these address the block *as* the value. Both erase
under a model that is not reference counting, like every other ownership op.

**The box needed routines of its own.** Its release had been the generic `object<tuple<..>>` one,
and that skips a `RefType` field — correctly, everywhere else in the compiler, since a reference
field points at storage its holder does not own. A capture-by-reference field is the exception:
it holds a cell address, and the box co-owns that cell. `getOrCreateCaptureBoxReleaseRoutine`
walks the fields itself, releasing a cell through the field and anything else in place, and is
keyed by the capture's own `ref<tuple<..>>` type rather than the box's `object<tuple<..>>` so
that neither the routines nor the descriptor can be shared with a plain object of the same shape.

Writing the cases turned up two more over-releases, both older than this slice and both confirmed
against a rebuilt baseline rather than assumed:

- **A `const` captured by value** goes into the box as a copy of the reference, and a copy of a
  reference is a further owner. Nothing retained it, so `const held = new Vec(15); return () =>
  held.x` read freed memory. `mlirGenRetainCaptured` at the capture site.
- **Assigning to a captured variable from inside the closure.** Inside the closure the variable
  is a load of the box's field, which none of `isOwningSlot`'s cases recognised, so `cur = new
  Vec(..)` stored a value nothing had taken — and §9.30 then released it at the end of the block
  as a discarded temporary, freeing what the assignment had just set. `isCapturedCellSlot`
  recognises the shape: a tuple field whose type is a reference is what a capture by reference
  is, and nothing else builds one.

The ownership verifier had to be told that `ts.ReleaseCell` discharges a slot too — without that
it reported every captured local in the suite as a leak, which is the verifier being right about
its own model and wrong about the program.

| shape | `gc` | `rc` | `none` |
| --- | --- | --- | --- |
| `raytrace.ts`, `-O3` | 4.2 | **2.6** | 114.5 |
| closure over a local, per call, `-O0` | 2.6 | **2.6** | 31.9 |
| closure over a parameter, per call, `-O0` | 2.6 | **22.6** | 2.6 |

`raytrace` 103.4 → **2.6 MB**, below `gc`'s own 4.2, and the size of that step says what the
cells were: its per-pixel closures (`addLight`, `recenterX`/`recenterY`) declare their captured
variables inside functions called once per pixel, so the leak was a cell per capture per pixel.

The third row is the remaining hole and is filed as 5r. A parameter is borrowed, so nothing in
the frame releases it, and a captured parameter's cell therefore has an owner that never lets go.
That is a leak and only a leak — the cell outliving everything is exactly what stops the box from
releasing an argument the caller still owns — and it is why the `none` column there is the small
one: with no ownership calls to keep it alive, the whole allocation is optimised away in the other
two models. (§9.35 closes it, and finds that the second row's benchmark was flattering: a
captured *local* leaks its cell too whenever the local's type owns nothing.)

Five new cases in `00owned_closures.ts`. **Teeth**, each checked by disabling the fix and
rebuilding: removing the box's retain of the cell, and removing the cell's birth reference, each
abort the test. Disabling the box's *release* of the cells does not and cannot — that is the
leaking direction.

Full release suite green: 925/925. Ownership verifier unchanged at its two standing findings.

### 9.35 Step 5r: a cell is given back whether or not the frame owns what is in it

§9.34 gave a cell owners, and released the frame's one at scope exit — but only for the slots
that were already on the scope's list, and that list is `ownedVars`, the list of locals whose
*value* the frame owns. Which of a function's cells got a release therefore turned on a question
that has nothing to do with cells at all. Two shapes fell outside it, and the measurements are
the whole argument:

| shape, `-O0` | `gc` | `rc` before | `rc` after | `none` |
| --- | --- | --- | --- | --- |
| closure over a `Vec` local | 4.1 | 3.7 | 3.7 | 161.3 |
| closure over a `number` local | 4.1 | **51.7** | **3.7** | 113.0 |
| closure over a `number` parameter | 4.1 | **51.7** | **3.7** | 122.6 |
| closure over a `Vec` parameter | 4.1 | **65.6** | **3.7** | 167.3 |

Only the first row worked, and only because a `Vec` local is one the frame owns the value of. A
captured `number` leaks its cell because a `number` owns no heap memory and so the local was
never listed; a captured parameter leaks its cell because a parameter is the caller's and is
never listed either. The cell is the same 24-byte block in all four rows.

So the listing and the ownership question are separated. `trackPossibleCell` lists every local
that has storage, owned or not, and `mlirGenFunctionBody` lists every parameter; scope exit then
asks each slot what it turned out to be — a cell gets `ts.ReleaseCell`, a slot the frame owns the
value of gets `ts.ReleaseSlot`, and a slot that is neither has nothing emitted for it and is on
the list only because it might have become a cell and did not.

That it can only be asked at scope exit is the reason for the two-step. Whether a variable is
captured is not known at its declaration: the closure that captures it is written afterwards, and
marks the storage when it is generated. Scope exit is generated after both.

A parameter's list is the function's rather than the body block's, which is the scope a parameter
actually has, and wiring it in exposed something that had always been wrong and was merely
unreachable. A scope exit walks outwards through `parentBlockContext`, and a function's context
inherits that pointer from whatever context the function was generated under — for a nested
function, the enclosing function's blocks. The walk used to stop at the first context without a
list, and a function's context never had one; give it one and a `return` inside a lambda starts
releasing the *enclosing* frame's locals, which crashes the compiler. `parentBlockContext` is now
cleared at the function boundary, where the walk always should have ended.

**The other half is what the cell holds.** A cell's release releases its contents, which is right
when the frame put an owned value there and an over-release when it did not — and a captured
parameter's cell starts out holding the caller's argument. Rather than teach the release to skip
those, the cell takes a reference to the value it is initialised with unless the frame already
took one (`OWNED_LOCAL_ATTR_NAME` says so). Then **a cell owns what it holds**, unconditionally,
and every reader can rely on it:

- the cell's release gives that reference back;
- `isCapturedCellSlot` (§9.34) already made a store *through the box* hand the count over;
- `isOwningSlot` now recognises the cell from the declaring frame's side too, so `v = new Vec(..)`
  written where the parameter is in scope takes the new value and gives up the old.

Without the last of those, the store takes nothing and §9.30 frees the value as a discarded
temporary — the same failure §9.34 found on the box side, arriving from the other direction.

Three new cases in `00owned_closures.ts`, each with **teeth** confirmed by disabling one fix and
rebuilding:

| case | fails without |
| --- | --- |
| `capturedParameterEscapes` | the cell's retain of the argument |
| `mutateCapturedParameter` | the cell's retain of the argument |
| `capturedParameterReassignedInFrame` | the cell in `isOwningSlot` |

Each fails only for its own fix and passes for the other, and the **release-before-retain swap** —
emitting the cell's retain ahead of the store rather than after it, so it retains whatever the
block happened to hold — crashes all three. The release side cannot be given teeth by a test, as
always: it leaks, and the measurements above are what stand in for it.

`capturedParameterEscapes` is two frames deep on purpose. `new Vec(50)` is a discarded temporary
of the *middle* frame, released at the end of that block, so a cell that had not taken a reference
loses the value before the outermost frame ever reads it. Written one frame shallower the release
lands after the read and the case passes with the bug present — the same trap §9.30 set, and the
reason an ownership case has to build in one block and read in another.

Two exclusions, both about where a release would land rather than about the variable: a
declaration directly inside a try body (its release is repeated in the cleanup region, which
cannot see storage declared in the body — `localTakesOwnership` pairs its answer with hoisting for
exactly that reason, and hoisting every local of every try body is not a trade this makes), and
one in a catch or finally clause, excluded on the same terms §9.24 excludes it. A captured
variable declared in either keeps leaking its cell.

`raytrace` at `-O3` is unchanged at **2.6 MB** against `gc`'s 4.2 and `none`'s 104.8 — §9.34 had
already taken its cells, which are all captured locals holding objects.

Full release suite green: 925/925. Ownership verifier unchanged at its two standing findings.

A corpus sweep under `-mm=rc` — every test file JIT'd, exit codes compared against the same sweep
on §9.34's commit — turned up one difference, `22lambdas.ts`, and it is **not this slice's**: it
fails about one run in four under `rc` on both commits (3 bad in 15 before, 4 in 15 after) and
never under `gc` or `none`. Filed as 5s. The sweep is worth keeping as a tool, and worth fixing
that flake first, since a flaky instrument is a poor way to measure the next slice.

### 9.36 Step 5s: boxing into `any`, and the flake that led there

§9.35 left a corpus sweep behind — every test file JIT'd under `-mm=rc`, exit codes diffed
against the same sweep on the previous commit — and one file, `22lambdas.ts`, failed about one
run in four under `rc` and never under `gc` or `none`. Nondeterministic and model-specific is
what a use-after-free looks like from outside: some run frees something still referenced, and
whether the reused block is fatal to read depends on what was allocated over it.

Halving the file three times reached this, which fails **5 runs in 5**:

```ts
function build() {
    let fns: any[] = [];
    for (let k = 0; k < 3; k++) {
        const kk = k;
        fns.push(qux2);
        function qux2() { glb1 += kk; }
    }
}
```

with `STATUS_HEAP_CORRUPTION`. Three of the four ingredients turned out to be required and one
did not. Pushing the *same* closure with no captures is clean; pushing strings is clean; a typed
`(() => void)[]` instead of `any[]` is clean; the loop is not needed at all. So: a closure, with
a capture box, boxed into `any`.

The IR says the rest:

```mlir
%11 = "ts.CreateBoundFunction"(%10, %9) {__owned_result, __owns_capture}
"ts.Retain"(%11)                            // the closure takes its capture box
%12 = "ts.Cast"(%11) : (!ts.bound_func<..>) -> !ts.any
"ts.Retain"(%12)                            // the array takes the box
"ts.ArrayPush"(%2, %12)
"ts.Release"(%11)                           // §9.30: nothing received the closure
```

Boxing allocates a block and copies the value into it, and the box **owns** what it holds: when
it dies it releases the payload through the type tag beside it (`buildBody`, AnyType). Nothing
took that reference. The closure arrived carrying one — `__owned_result`, the reference
`resolveFunctionWithCapture` takes on the capture box — and, no receiver having claimed it,
§9.30 gave it back at the end of the block. The capture box was freed while the `any` in the
array still pointed at it, and the array's own release then released it a second time.

The fix is one line in `cast()`: where the destination is `any`, `mlirGenRetainCaptured` — the
same consume-or-retain every other owning receiver uses. Consume where the payload already
carries a reference, retain where it does not.

The two halves of that matter separately, and the cases prove it separately. A payload that
carries a reference and is *not* consumed is the crash above. A payload that carries none and is
not retained is a plain leak — `out.push(makeName())` where the frame also still holds the
string — and `stringBoxedAndStillHeld` is the case for it. Four new cases in a new file,
`00owned_any_boxing.ts`, and **all four abort with the fix disabled**.

They run as `--emit=exe`, or through `test-runner`, but not as a bare `tslang --emit=jit`. That
is worth writing down because it cost an hour: `___unbox` throws on a type mismatch, which pulls
in the CRT's `type_info` vftable, and only `--shared-libs=TypeScriptRuntime.dll` makes that
symbol resolvable — which `test-runner` passes and a hand-written sweep does not. The sweep
script was wrong in exactly that way, and fixing it dropped its failure count from 161 to 118 of
473 files. **A sweep that has not loaded the runtime is measuring the harness.**

**A second bug, and it is §9.35's.** With the sweep fixed, the before/after diff showed
`15references.ts` newly crashing, and it has no `any` in it at all:

```ts
let x: string;
const f = () => { if (1 > 1) x = "foo"; };
f();
```

A captured declaration with **no initializer**. Its cell is a *heap* block, so what it holds
before the first store is whatever the allocator last left there — and §9.35 lists every local,
initialised or not, so scope exit releases the cell and the cell releases its contents. An owned
local can never be in this position: it is only owned when it has an initializer, which is why
§9.24's null-store was written for the hoisted-out-of-a-`TryOp` case alone. Cells are now zeroed
at birth on the same terms.

Its two cases needed care to make deterministic. `churn()` before the declaration caught it about
one run in ten, because the block has to be the right size *and* still hold something fatal to
release. Running a *cell of exactly this shape* first — `writtenCell`, which allocates one, frees
the string in it and then frees the cell — hands the next declaration a block holding an
already-freed string pointer, and that is 10 runs in 10.

929/929 with the four new tests registered. Ownership verifier unchanged at its two standing
findings. The final sweep diff against §9.35's commit is two lines: `22lambdas.ts` 3 bad runs →
0, and the new file.

**What the measurements then said, which is not about `any`.** Boxing in a loop still leaks under
`rc` — 39.9 MB against `gc`'s 4.1 — and taking it apart puts the leak somewhere else entirely:

| 500k iterations, `-O0` | `gc` | `rc` | `none` |
| --- | --- | --- | --- |
| `push(k)` into `number[]` | 2.6 | 2.6 | 2.6 |
| `push(k)` into `any[]` | 2.6 | 3.7 | 32.3 |
| `let s = "s" + k`, no array at all | 4.1 | **34.5** | 42.8 |
| `push("s" + k)` into `string[]` | 4.1 | 32.3 | 52.2 |

The array machinery is flat, the `any` box is flat, and a concatenated string held by a local
leaks on its own. Filed as 5t, and it is the one to take next: strings are Tier C, the narrow
first shipping scope this evaluation recommends.

### 9.37 Step 5t: a string that nothing owns

The `any`-boxing benchmark from §9.36 still leaked after that fix — 39.9 MB against `gc`'s 4.1 —
and taking it apart moved the leak out of `any` entirely, then out of arrays:

| 500k iterations, `-O0` | `gc` | `rc` before | `rc` after | `none` |
| --- | --- | --- | --- | --- |
| `push(k)` into `number[]` | 2.6 | 2.6 | 2.6 | 2.6 |
| `push(k)` into `any[]` | 2.6 | 3.7 | 3.7 | 32.3 |
| `let s = "s" + k`, no array, no `any` | 4.1 | **34.5** | **3.7** | 50.6 |
| `push("s" + k)` into `string[]` | 4.1 | 32.3 | **3.7** | 72.6 |
| `push(new Vec(k))` and `push("s" + k)` into `any[]` | 4.1 | 39.9 | **3.7** | 130.1 |

The third row is the whole story and it has no container in it at all. `("s" + k)` allocates
twice — the conversion prints `k` into a fresh buffer (`ConvertLogic`'s itoa/f64ToString), and
the concatenation builds another (`ts.StringConcat`) — and neither said it was handing anyone a
reference:

```mlir
%5 = "ts.Cast"(%3) : (!ts.number) -> !ts.string       // allocates; unmarked, unreferenced
%6 = "ts.ArithmeticBinary"(%4, %5) ...                // allocates; the local retains it
%7 = "ts.Variable"(%6) {__owned}
```

`%6` was fine by accident: it is born unowned, the local retains it, the local's scope exit
releases it. `%5` is the leak. Nothing received it — being an operand of a concatenation is not
receiving — so no receiver retained it, and §9.30 only releases what carries `OWNED_RESULT`, so
it did not release it either. Every number ever printed into a string leaked.

Both now do what `new` and a call do: **a `ts.Retain` making the reference real, and
`OWNED_RESULT` saying a receiver may take it over rather than adding one.** The two halves have
to travel together, and that is the whole design point — the mark alone hands a receiver a
reference nobody made, and the retain alone is a leak with extra steps. It is also what makes
this safe to be generous with: a value wrongly counted as fresh gains a reference and a release
for it, which is balanced.

Marked: the plain cast to `string` where the source is a number, an integer, an index or a char
(the printing conversions — a boolean, `undefined` and a string literal all hand back a global,
which is immortal), and `+` where the result is a string, which is the one arithmetic operator
that allocates. Class, array and tuple `toString` are ordinary calls and were already handled by
§9.27.

**The second bug, and it is much older.** Measuring the array rows turned up `00map.ts` failing
under `rc` — 6 runs in 20, and 0 in 20 under `gc` and `none`. It reduces to:

```ts
arrS.map((e) => e + "_")
```

and the default library's `map` is `result.length = this.length; for (..) result[i] = func(..)`.
Growing an array is `MemoryRealloc` and nothing zeroes the tail, so the new slots hold whatever
was last in that memory — and an element store gives up what the slot held first
(`isOwnedElementSlot`), so the release reads it. `SetLengthOfOpLowering` now zeroes the exposed
tail, under `-mm=rc` and only where an element owns something.

It was tempting to read this as §9.37's own regression, because the sweep diff showed `00map.ts`
newly crashing. Rebuilding §9.36's commit and running it twenty times is what settled it: 6 bad
there too. **One clean sweep run is not evidence about a file that fails a third of the time** —
the same trap §9.36 walked into from the other side, where a single lucky run hid it.

Four new cases in `00owned_strings.ts`, and the two halves have separate teeth:

| probe | fails |
| --- | --- |
| mark the fresh string without retaining it | the three string cases, 10 of 10 |
| do not zero a grown array's new slots | `grownArrayHoldsItsStrings`, 12 of 12 |

Neither probe touches the other's cases. The string cases have no teeth against the *leak* —
nothing can, a leak being invisible — so what they guard is the over-release the fix could
introduce, which is why "mark without retain" is the probe that matters for them.

Two things were needed to make them read a freed buffer reliably. **Compare the whole string,
not its length**: a freed buffer keeps its length long after its bytes are written over, and the
length-comparing versions passed 8 runs in 8 with the bug present. And, for the array case,
**eight rounds of a same-shape scratch array** rather than a general `churn()`: what makes an
unwritten slot fatal is holding something that looks like a string, and one round got three runs
in four where eight gets all of them.

933/933. Ownership verifier unchanged at its two standing findings. The sweep diff against
§9.36's commit is one line, the new file. `raytrace` at `-O3` is unchanged at 2.6 MB against
`gc`'s 4.2 and `none`'s 109.0.

### 9.38 Step 5u: reading the sweep, and the configurations nobody had swept

The `-mm=rc` corpus sweep had sat at 117 non-zero exits of 475 for three slices running. It was
stable enough to be useful as a regression detector - a diff against the previous commit found
§9.36, and half of §9.37 - and it had never once been read. Reading it needed one question:
**what does `gc` do with the same file and the same command line?** Whatever fails under both is
not the memory model.

| bare `--emit=jit -mm=<model> --shared-libs=...`, 475 files | non-zero |
| --- | --- |
| `gc` | 101 |
| `rc` | 117 |
| fails under `rc`, passes under `gc` | **16** |
| fails under `gc`, passes under `rc` | 0 |

So 101 of the 117 have nothing to do with reference counting, and every one of them is the
sweep's own doing:

- **79 are one half of a two-file test.** Every `export_*`, `import_*`, `decl_*` and `emit_*`
  file, plus `component`/`service` and `shared`/`use_shared`. The exporting half has no `main`
  (`Symbols not found: [ main ]`); the importing half needs its partner built into a DLL first
  (`Symbols not found: [ M.Animal..new, ... ]`). JITing either one standalone cannot work by
  construction.
- **20 are the default library colliding with the test.** `redefinition of symbol named
  'Math.PI'`, `'sqrt'`, `'Number..instanceOf'`, `'Error..instanceOf'` - tests that declare their
  own and are meant to be compiled without the default library - plus a few type errors
  (`conditionalTypes1`, `path`, `raytrace-0`) that are compiler limitations, identical under both
  models.
- **2 fail identically under both models**: `arrayLiterals2ES5` faults under `gc` too, and
  `import_object_literal_untyped_multi_method` fails an MLIR verifier in both.

That second group is the interesting one, because it says the sweep had never been running these
tests the way the suite runs them. `test-runner` compiles with **`--opt --opt_level=3
--no-default-lib`**; the sweep compiled bare - `-O0`, with the default library. Those are not the
same program, and neither configuration dominates the other:

| | `gc` | `rc` | rc-only |
| --- | --- | --- | --- |
| bare, `-O0`, with the default library | 101 | 117 | 16 |
| `--opt --opt_level=3 --no-default-lib` - what the suite runs | 87 | 109 | **22** |
| union of the two | | | **29** |

Only 9 files are in both lists. `gc` stays clean in the other direction in both configurations:
nothing passes under `rc` that fails under `gc`.

**Exactly one of the 29 is a file ctest ever runs under `rc`.** 39 of the 475 test files are
registered as `test-jit-rc-*`; the other 436 have never been compiled under `-mm=rc` by anything
but this sweep, and everything the ownership work has been measured against lives inside that
39.

The ownership verifier reports nothing for any of the 29, and that is structural rather than a
gap. It checks that an acquired slot is given back on every path out of a function, which is the
*leak* direction. All 29 are faults - `0xC0000005`, `0xC0000374`, a breakpoint - which is the
over-release direction, and it cannot see that at all.

#### The configuration columns are not four bugs

Running all 29 under `rc` in all four combinations of `{-O0, -O3} x {default library, none}`:

| how many configurations fault | files |
| --- | --- |
| all four | 11 |
| some but not all | 18 |

The 18 are the point. A use-after-free that passes is a use-after-free that got away with it, and
what the columns vary is the heap layout, not the program's ownership. `00array3` faults with the
default library at both optimisation levels and passes without it; `01map` is the exact opposite
of `00spread`; `00generator4` faults at `-O3` only, and on one run it faulted, on the next it
hung, and on a third it came back with a failed `assert` - the same bug wearing three symptoms.
So the honest count is not 16, or 22, or 29: it is that **29 files break the memory model and 18
of them only break it sometimes**, which is what this class of bug looks like from outside.

#### Three of them reduce

The 29 are not 29 bugs. Three reductions, each `rc`-only, each 10 runs in 10:

**A generator that iterates another generator**, at `-O3`, with or without the default library:

```ts
function* inner() { yield 1; yield 2; }
function* g() { for (const o of inner()) yield o; }
function main() { for (const x of g()) print(x); print("done."); }
```

Faults before printing anything. `yield* inner()` and an inline `(function*(){...})()` fault the
same way; one generator on its own is fine, and `-O0` is fine.

**A `for...of` element that is itself a reference**, with the default library, at either
optimisation level:

```ts
const a = [[1], [2]];
for (const v of a) print(v.length);
```

Also faults before printing. A single-level `for...of` is fine and `a.length` on the same array
is fine, so it is the loop variable taking an element that owns something. `--no-default-lib`
hides it because it routes `for...of` over an array through the built-in intrinsic loop instead
of the library's iterator protocol.

**`await`**, in all four configurations, and the only one of the three that gets the answer
right first:

```ts
async function f() { return 1; }
function main() { const r = await f(); print(r); print("done."); }
```

Prints `1` and `done.` and then dies of heap corruption on the way out - a double free at
teardown. Calling `f()` without awaiting it is fine, so the `await` is what frees something
twice.

Between them these cover the generator/iterator cluster (`00generator4/5/6`,
`00funcs_expression_iterator`, `01iterator`, `00iterator_bug`), the array-iteration cluster
(`00array3`, `00for_of`, `19forof`, `01map`, `00tuple_with_array`, `arrayLiterals`) and the
coroutine pair (`00async_await`, `00for_await`). The rest - object destructuring, union
narrowing, class statics, and the two whole programs `nbody` and `raytrace` - has not been
reduced.

#### Two things about measuring this

**Sweep the configuration the suite uses, not a configuration.** A missing `--no-default-lib`
manufactured 20 failures; a missing `--opt --opt_level=3` hid 13 real ones. The recipe worth
keeping is: sweep both configurations under both models, diff, and read only the rc-only column.
That is what turns the sweep into a to-do list. This is the second time the sweep's own command
line was the bug - §9.36 lost 43 files to a missing `--shared-libs`.

**Read exit codes in PowerShell, not in bash.** A Windows fault code does not survive bash's
8-bit exit status. The `await` reduction reported `0/10` failures through a bash `||` and
`10/10` - every one of them `-1073740940` - through `Start-Process`'s `ExitCode`. The one that
reads clean is the wrong one.

Nothing was changed in this slice. 933/933, verifier unchanged at its two standing findings.
