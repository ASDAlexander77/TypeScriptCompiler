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
that can contradict each other.

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
   strings is impossible until heap and static strings are distinguishable. Blocks the
   strings-first scope below.
5. **Ownership tracking in MLIRGen behind `-mm=rc`**, checked by a verifier that flags any
   owned value without a matching release on every path, unwind paths included. *Point of
   no return.*
6. **Flip the allocator under the flag.** GC stays the default.

**Scope the first shipping mode narrowly.** Two candidates, and they are compatible:

- **`string` only (Tier C).** Strings are leaves — a string never points to another heap
  object, so release is a single free with no recursive traversal and **no cycle is
  representable**. Strings are also the highest allocation-rate type. Highest benefit, zero
  cycle risk, bounded blast radius. **Blocked on step 4a** — see §9.4.
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
lookup rather than a static walk; function types do not mention their capture box, so there is
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
