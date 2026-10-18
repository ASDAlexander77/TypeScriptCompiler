# Stack promotion: short-lived objects on the stack

Status: design, 2026-10-18. Not implemented.

## 1. Intent

Pure performance. Short-lived objects - a temporary class instance, a small array, a boxed
object literal, a closure's capture box - should cost a stack slot, not a heap allocation, a
collector's attention, or a retain/release pair. The motivating case: code like a number-to-string
conversion that builds a helper object, uses it, and drops it should never touch the heap.

The program's meaning does not change. Only speed and memory use do.

## 2. Decisions (the user's)

- **The compiler decides where an object lives.** An escape analysis looks at every allocation
  site. A site whose object provably never outlives its function, and is small enough, allocates on
  the stack. Every other site allocates on the heap exactly as today.
- **Escapes are silent.** No error, no warning: a site that escapes is simply a heap site. There is
  no clone-on-escape - an object is never copied, so reference identity always holds.
- **`Class()` and `new Class()` are the same.** `Class()` stays accepted, as today, and means
  exactly `new Class()`: same constructor call, same analysis. It is no longer a request for the
  stack.
- **Every kind of heap allocation the compiler makes for a value is covered**, not only classes:
  class instances, arrays, boxed object literals, closure capture boxes.
- **Too big goes to the heap.**
- **All memory models**: `gc`, `rc`, `own`, `none`.

## 3. Goals, non-goals, anti-goals

Goals
- Fewer heap allocations for short-lived objects, measured (§9).
- No behavioural change under any memory model; the existing suite passes unchanged under all four.

Non-goals
- A user-visible annotation or keyword for stack allocation.
- Copying or moving objects between stack and heap at run time.
- Stack promotion of objects allocated in one function and dying in another (§8.1).

Anti-goals (a failure even if it "works")
- Any dangling stack pointer. When the analysis is unsure, the answer is heap.
- Stack overflow in programs that run today (deep recursion with promoted frames).
- A noticeable compile-time cost.

## 4. Feasibility notes that shaped the design

### 4.1 Why not LLVM address spaces

The first idea was to tag stack pointers with an LLVM address space. Rejected as the safety
mechanism (builder may still use one as an internal aid, see §7):

- An address space is a static tag on the pointer type. LLVM diagnoses nothing; an
  `addrspacecast` to address space 0 is just a cast. "Protection" would be a pass we write that
  treats the cast as an escape - i.e. the escape analysis itself, run after lowering has erased
  the facts it needs (field store vs call argument vs capture vs return).
- Every method's `this`, every runtime and DefaultLib function takes an address-space-0 pointer,
  so under "cast = escape" `a.method()` and `print(a)` are escapes. Any useful analysis needs
  parameter borrow facts - which `-mm=own` already computes in MLIR.
- `alloca` returns address space 0 on x86/ARM/wasm; wasm gives meaning to address spaces 1, 10,
  20 and x86 to 256-258 and 270-272; every pattern lowering a class pointer would have to accept a
  second pointer type.
- Prior art (Julia's address spaces 10-13, LLVM statepoint GC's address space 1) uses address
  spaces as optimizer bookkeeping stripped before codegen, not as the safety check.

### 4.2 What exists today

- `ts.New` has a `stackAlloc` attribute; `Class()` sets it
  (`MLIRGenAccessCall.cpp`, `NewClassInstance`) and `NewOpLowering` emits an `alloca`
  (`LowerToLLVM.cpp`). The `alloca` has **no block header**, and `Class()` is rejected for any
  class with a virtual table - which is now every class - so the path is dead. There is a `TODO`
  on it.
- `LLVMCodeHelperBase::Alloca` already places allocas in the function's entry block.
- Heap blocks carry a leading header word; `HEAP_BLOCK_IMMORTAL` in it makes rc's retain/release
  and own's destroy skip the block (literals and `typeof` tags use this).
- `OwnershipSignaturePass` (module pass) computes per-function facts, notably `__own_params`
  (parameters the body stores into a container or returns) and `__own_result_borrows`.
  `OwnershipInferencePass` follows def-use chains per function, including closure escapes. Both run
  only under `-mm=own`, after `LowerToAffine*` and before the optimiser (`tslang/transform.cpp`).

## 5. The analysis

### 5.1 Where it runs

A new pass, `StackPromotionPass`, per function, under **every** memory model, at the same level
as the ownership passes (after `LowerToAffine*`, before the optimiser): control flow is a plain CFG
with unwind edges, and every allocation and ownership op is still present.

It needs `OwnershipSignaturePass`'s per-function parameter facts, so that pass runs under every
model when promotion is on (it is analysis only; it pins attributes). Under own it already runs
first. The builder decides whether the fact computation is shared or factored out.

### 5.2 What counts as an escape

A site escapes - and stays on the heap - if its value, or anything derived from it, may be:

- returned, yielded, or thrown;
- stored into a field, array element, global, `captured` variable, or any slot not itself a local
  of this function that dies with it;
- captured by a closure that escapes; (a non-escaping closure capturing it is fine);
- passed to a parameter the callee's facts do not show as borrowed. **Unknown callee = escape**:
  declared/external functions, imports without facts, virtual calls whose candidates disagree,
  indirect calls through values. (Note this is the *opposite* default from own's
  "unknown callee borrows" - own's default is leak-safe, not stack-safe.)
- stored as `this` by its own constructor (the constructor is a call with `this` as an argument;
  it follows the rule above);
- converted to `any`, an interface, a union, or `unknown` where the converted value escapes by
  the same rules (an interface value carries the object pointer);
- live across an `await` or `yield` in a coroutine. Simplest correct rule (builder's choice to
  refine): **no promotion inside async functions and generators** in the first version.
- returned through a call whose result borrows from it (`__own_result_borrows`) where that result
  escapes.

Anything the analysis cannot classify is an escape.

### 5.3 What is covered

| allocation | stack when |
| --- | --- |
| class instance (`ts.New`, and the gc typed `GCNewExplicitlyTypedOp`) | does not escape, size ≤ limit |
| array header | does not escape |
| array data block | header on stack **and** length is a compile-time constant **and** nothing may grow it (`push`, `unshift`, `splice`, `length =`, anything passed the array where the callee's facts do not rule it out). Otherwise heap even when the header is on the stack. |
| boxed object literal | does not escape, size ≤ limit. Object literals that are plain tuple values are unchanged. |
| closure capture box | the closure does not escape |

Strings are not in the first version (they are produced by runtime/DefaultLib calls, see §8.1).

### 5.4 Size limits

- Only compile-time-known sizes qualify.
- Default: at most 256 bytes for one object, at most 4 KB promoted per function. (Our call;
  Windows' default stack is 1 MB and recursion multiplies a frame. Go allows 64 KB implicit.)
- Both adjustable by flag. The builder picks names; suggested `--stack-promotion-max-object=` and
  `--stack-promotion-max-frame=`.
- When the per-function budget is exceeded, sites are kept in order of expected benefit (sites in
  loops first); the builder decides the heuristic.
- **Recursion (our call).** A function on a direct call-graph cycle in its module (computed from
  the call graph `OwnershipSignaturePass` already walks) gets no promotion. Cycles only through
  virtual or indirect calls are not detected; for those the per-function budget is the guard.
  Promoted slots live in the entry block, so every call pays for them even on a cold path - one
  more reason the defaults are small.

### 5.5 Loops

All promoted allocas go in the entry block (already what `Alloca` does). A site inside a loop
reuses one slot per iteration, which is correct only if the object does not live into the next
iteration: a value from iteration *n* still reachable in iteration *n+1* (stored in a local
declared outside the loop, for instance) is an escape for this purpose. A use outside the loop
from which the site cannot run again - a `return` or a `break` out of a search - sees the last
iteration's object and is not.

## 6. Per memory model

The stack object has the same layout as a heap block: header word, then payload. The pointer the
rest of the compiler sees addresses the payload, as for heap blocks. The header is set to
`HEAP_BLOCK_IMMORTAL`, so all existing code that inspects headers treats it as not-to-free.

| model | allocation | at scope exit |
| --- | --- | --- |
| gc | alloca; vtable store as today; no type descriptor | nothing. Boehm scans the stack conservatively, so heap objects held by the stack object's fields stay alive; it ignores pointers into non-heap memory. |
| rc | alloca, header = immortal | **release the object's fields** (and array elements) without freeing the object. Today's release routine does "release fields + free" in one; it needs a "release contents" half callable on its own. Retains/releases of the object itself become no-ops (immortal) - the builder should delete them in the pass rather than leave the runtime check. |
| own | alloca, header = immortal | destroy contents without free; the scope is the owner. `--verify-ownership` must accept promoted sites. |
| none | alloca | nothing. |

Exceptions: the scope-exit action must also run on the unwind path, using the same machinery
that releases locals during unwind today.

## 7. Flags and diagnostics

- On by default at every optimisation level, including `-O0` (our call): one behaviour everywhere,
  and the Debug suite exercises it. Debugger display of promoted objects must keep working (the
  payload pointer is what debug info describes, as for heap objects).
- `--no-stack-promotion` turns it off - for bisecting miscompiles and A/B measurement.
- A report flag (name: builder) prints, per allocation site, stack or heap, and for heap the
  first escape reason. Off by default. Tests use it to assert decisions.
- Optional, builder's call: in debug builds, tag promoted pointers with a reserved address space
  in the LLVM IR and verify no `ptrtoint`/store of one reaches memory not on the stack - a
  checker, never the mechanism.

## 8. Scope decisions (the user's, 2026-10-18)

### 8.1 Allocations in a callee that die in the caller: not in version 1

Promotion is per allocation site, and a site can only use its own function's stack. An object a
callee allocates and returns - number-to-string inside the DefaultLib, for instance - stays on the
heap in version 1. Inlining (same-module only) and a caller-provided result slot (like C++
return-value optimisation / LLVM `sret`) were considered and deferred.

### 8.2 Calls into other modules are escapes in version 1

"Unknown callee = escape" stands: every call into DefaultLib or a `-shared` library is an escape
for its arguments. Future direction, not version 1: describe callee behaviour in declarations -
per-parameter "borrowed, not kept" facts, either exported automatically with the cross-module
declarations or written as attributes on declarations - so an object can stay on the stack across
such a call. Version 1 should keep its parameter facts in a form that can later be read from a
declaration as easily as computed from a body.

### 8.3 On by default

The feature ships on by default. The §9 benchmarks are reported, not a gate.

## 9. Testing and measurement

- Suite: the full ctest suite under gc, rc, own and none with promotion on; the same with
  `--no-stack-promotion` must stay identical in output.
- Decision tests: small programs plus the report flag, one per escape rule in §5.2, each asserting
  stack or heap. Escape rules are exactly where a bug becomes a dangling pointer, so every rule
  gets a positive and a negative test.
- Teeth: per the project's habit, temporarily disable one escape rule and confirm a test fails
  (the dangling case should read a value overwritten by a later call).
- Benchmarks, before/after, each model: a tight loop building and dropping a small class; vector
  math with temporaries; the ray tracer and n-body from `docs/memory-models.md`. Report time,
  peak working set, and allocation count.

## 10. Left to the builder

Pass structure and sharing with the ownership passes; the exact op list each rule matches; flag
names; the budget heuristic; how `GCNewExplicitlyTypedOp` and `ts.New` converge; removing the old
`stackAlloc` path and the "virtual table on stack" error; phasing into PRs.
