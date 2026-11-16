# `WeakRef<T>` under `-mm=own`: design

Date: 2026-11-16. Status: draft for approval.

## Why

Under `-mm=own` every heap object has one owner, and the owner frees it. A child cannot point back
at its parent: the field would be a second owner. Today such a program either does not compile or
uses `Shared<T>`, whose cycles leak. The readiness page (`docs/own-readiness.md`) lists this first
among what is left: "Without it a tree with parent links either does not compile or leaks."

The goal is a reference that does not own what it points to, that cannot read freed memory, and that
a TypeScript programmer already knows how to write.

```ts
class Node {
    parent: WeakRef<Node> | null = null;
    children: Node[] = [];
    constructor(public v: number) {}

    add(c: Node) {
        c.parent = new WeakRef(this);
        this.children.push(c);
    }

    depth(): number {
        const p = this.parent?.deref();
        return p ? 1 + p.depth() : 0;
    }
}
```

## Decisions taken (user, 2026-11-16)

| Question | Decision |
| --- | --- |
| Form | The standard `WeakRef<T>`, with `new WeakRef(x)` and `deref(): T \| undefined`. Checked at run time. Not a compile-time "borrowing field" |
| What `deref()` gives | A plain borrow, checked at compile time like every other borrow under own. Not a pin count held at run time |

## Goals

- `WeakRef<T>` of a class object compiles and runs under own, gc and none, with the same output.
- `deref()` returns the object while it lives, and `undefined` once its owner freed it, at once and
  every time (stronger than JavaScript's "some time after", and compatible with it).
- No accepted program reads freed memory or leaks. Every block is freed exactly once.
- No object gets bigger. Programs that do not use `WeakRef` run no new code beyond one test per free
  of a class object.

## Non-goals

- `-mm=rc`. Its header word is the strong count; a weak count needs a second word (rc evaluation
  §9.8), which is an ABI change of its own. Under rc, `WeakRef` stays a compile error with a message
  that names the models that have it.
- `WeakMap` and `WeakSet` under own (they stay gc only).
- Under own and none, a `WeakRef` to anything but a class object: arrays, strings, object literals,
  closures, `Shared<T>` handles, `@value` classes. These are compile errors there. Under gc,
  `WeakRef<T extends object>` keeps accepting any object, as the library class does today, so no gc
  program breaks. *(Builder-check call.)*
- Atomic weak counts. Instead, under own, an object of a weak-target class may not cross threads (see
  "Threads" below).
- `FinalizationRegistry`.

## Anti-goals

- A `deref()` that can return an object which is freed before the borrow's last use. That is the one
  failure that makes the feature worse than not having it.
- A tombstone or weak count that leaks: every object freed while weak references remain must have its
  block freed when the last of them goes.

## Constraints from the code as it is

- **The header word.** Every heap block has one word in front of the payload. Under own,
  `_MemoryAlloc` writes 0 there, meaning "mortal" (`LLVMCodeHelperBase.h:516-543`); an immortal
  block holds `HEAP_BLOCK_IMMORTAL` (-1, all bits set), which `emitIsMortal` tests. A class object's
  word under own holds nothing else today: no count, never the shared bit (only `Shared<T>` blocks are
  counted). Under none and gc nothing writes the word, and none never frees.
- **The free path.** The release routine of a class (`OwnershipRoutineLogic.h:1707-1718`) is: if the
  object is mortal, release its fields, then free the block. Routines are generated per module, per
  type.
- **Stack promotion** (`StackPromotionPass.cpp`) runs after own inference under every model and may
  put an object on the stack, writing `HEAP_BLOCK_IMMORTAL` into its header. A `WeakRef` to it would
  then never see it die, and read the dead frame. There is a precedent: under rc and own an object is
  not promoted when the module uses `Shared` and the storage owns heap (`"shared-handle"`,
  `:390-397`).
- **Name resolution.** The default library declares `class WeakRef<T extends object>` (gc only, built
  on `GC_general_register_disappearing_link`, `lib.generics.ts:3061-3081`), and `WeakMap`/`WeakSet` on
  it. A declared class is found before an embedded type (`MLIRGenTypes.cpp:804-814`), so a built-in
  `WeakRef` is shadowed while that class exists.
- **`Shared<T>` handles** are the model for a value own does not own but counts: `isHandle` values
  keep rc's retains and releases, and lowering counts them. A `WeakRef` is that kind of value.
- **Borrows of an unknown place** already exist: TSLANG 84's chain with `anyPlace`, ended by any
  release whose type may hold the borrowed value's type (`typeMayHold`) and by any call that may drop.

## Approaches considered

1. **`WeakRef<T>`, weak count in the target's header word (chosen).** No new block, no growth.
   Freeing an object that weak references still point to releases its fields and keeps the block as
   a tombstone; the last weak reference frees it.
2. **`WeakRef<T>` as a separate cell block** (the object points at a cell that outlives it). Needs a
   header slot or a side table to find the cell from the object, which collapses into 1 or a table
   lookup on every free. Rejected: more memory and a lookup.
3. **A declared borrowing field, proven at compile time.** No run-time cost, but the compiler must
   prove the target outlives the holder, which fails as soon as a child is taken out of its parent.
   Rejected by the user in favour of 1; it may come later as an opt-in for shapes that can be proven.

For `deref()`, a run-time pin count (`deref()` raises it, the borrow's end lowers it, a free while
pinned waits) was considered and rejected by the user in favour of a compile-time borrow.

## The design

### Surface

- `WeakRef<T>` is a type built into the compiler, as `Shared<T>` is (`!ts.weak_ref<T>`), so it works
  with `--no-default-lib` and in every model that has it. `new WeakRef(x)` and `w.deref()` are
  recognised as `Shared`'s members are.
- `T` must be a class type, and the argument a class object (not `null`, not a union). Anything else
  is an error: "WeakRef needs a class object".
- Under rc: "WeakRef is not supported under -mm=rc; use -mm=gc, -mm=own or -mm=none".
- The default library's `class WeakRef` is removed. `WeakMap`/`WeakSet` stay gc only: their entries
  keep the disappearing link through a private library class (today's `WeakRef` body under another
  name), so under the other models they fail exactly as they do now, on the `GC_*` call. The built-in
  type under gc lowers to the same link. The library and the compiler are released together (the
  library's own build cannot compile a `WeakRef` otherwise). *(Builder-check call.)*

### At run time, by model

| | `new WeakRef(x)` | `w.deref()` | a `WeakRef` released | `x` freed |
| --- | --- | --- | --- | --- |
| own | one pointer to `x`; `x`'s header count + 1 unless immortal | `x`, or `undefined` if `x`'s block is a tombstone | count - 1 unless immortal; if it reaches 0 on a tombstone, free the block | if the count is non-zero: release the fields, mark the tombstone, keep the block; else as today |
| gc | a block holding the hidden pointer, registered as a disappearing link (today's class, inlined) | `undefined` once the collector cleared the link | nothing | the collector clears the link |
| none | one pointer to `x` | always `x` (none never frees) | nothing | never |

Under own the header word of a mortal class object becomes:

```
  bit 63 (top)   unused for class objects (the shared bit of Shared blocks)
  bit 62         dead: the object was freed and the block is a tombstone
  bits 0..61     the number of WeakRefs to it
  all bits set   immortal, as today (tested first)
```

The word size is 4 bytes on wasm32 and 32-bit targets; the dead bit is then bit 30. The count cannot
reach the dead bit in any program that fits in memory, since every `WeakRef` is at least a pointer.

`deref()` on a tombstone returns `undefined`. Reading any field of a tombstone is impossible, because
`deref()` is the only way to it.

### Ownership of the `WeakRef` value itself (own)

A `WeakRef` is a handle in own's sense, like `Shared<T>`: own steps aside, rc's retains and releases
of it stay, and lowering turns them into the target's count + 1 and - 1 (`isHandle` covers
`weak_ref`). Copying a `WeakRef` (`b.parent = a.parent`) is one more weak reference. So a `WeakRef`
field, element or local behaves exactly as a `Shared` handle does, with the target's header as the
count and no block of its own.

One difference from `Shared`: releasing a `WeakRef` never frees anything but a tombstone, which nothing
can reach. So it is no drop: it does not take `__own_no_drops` from a function (`dropsInBody` counts a
handle release as one today), and it ends no borrow. *(Builder-check call.)*

### What `deref()` returns (own)

The result is a **borrow with an unknown place** (TSLANG 84's `anyPlace` chain), not tied to the
`WeakRef` it came from:

- it may be used — fields read, methods called, passed to calls that do not keep it — until anything
  that could free the object: a release of a value whose type may hold `T` (`typeMayHold`), a call that
  may drop (`__own_no_drops` absent, or `__own_drop_fields` of a field whose type may hold `T`);
- it may not be stored, returned, captured or given to a call that keeps it: "'p' borrows the object a
  WeakRef refers to and cannot be stored, returned or captured";
- a use after something that could free it: "'p' borrows the object a WeakRef refers to but is used
  here after it may be freed".

So `const p = this.parent?.deref(); if (p) return p.v + p.depth();` compiles when `depth` frees
nothing (a method of a class nobody else extends, or `@final`), and the error explains when it does
not. `this.parent?.deref()?.v` is the same rule with one use.

Under gc and none the result is an ordinary value; the borrow rule is own's alone.

### Where an object may live

An object of a class that is the `T` of any `WeakRef<T>` in the module (or a subclass of one) is never
stack-promoted, under every model (under none a promoted target would be read through the `WeakRef`
after its frame ended): a new refusal reason, `"weak-target"`, beside `"shared-handle"`.
*(Builder-check call: every model, not own only.)* An object a
`WeakRef` refers to that is immortal (a global's, kept for good) is valid forever and needs no count.

### Threads (own)

Weak counts and tombstones are not atomic. If an object reached another thread and was freed there, a
`deref()` on the first thread could race with the free. So under own an object of a weak-target class
(or anything that can hold one, by `typeMayHold`) may not cross threads: it may not be captured by a
`Thread.spawn` callback, returned from one, or be (or be held by) the payload of a `Shared<T>` that is
shared with `Shared.share`. Each is a compile error naming the class. *(Builder-check call: the goals
rule out atomics in every free and the anti-goals rule out leaving it open, so refusing is the one
consistent choice.)*

### Across modules

The free path changes for every own class object, and routines are generated per module. A module
compiled by an older compiler frees a block without looking at the count, which would free a block
`WeakRef`s still point at. The default library is rebuilt with the compiler, as for every release;
an object file from an older own compiler linked into a program that uses `WeakRef` is unsupported,
and nothing detects it.

## Tests to write

- `00weakref` (own, gc, none; and `-noopt` under own): a tree with parent links built, walked up from
  a leaf, a child moved to another parent, a subtree dropped while a `WeakRef` into it remains and
  `deref()` is `undefined` afterwards; churn between steps so that a freed block read by mistake shows.
- Leak and double-free checks: an ahead-of-time churn loop with flat memory; the own tests on Linux
  under `MALLOC_PERTURB_` with tcache off; a tombstone freed when its last `WeakRef` goes (count the
  frees, or a churn that would grow if it were not).
- Errors: a weak-target object captured by `Thread.spawn`, returned from it, or in a shared `Shared`;
  `deref()` result stored, returned, captured; used after a call that may drop and after a
  release of a holder of `T`; `WeakRef` of a string, array, `@value`, `Shared`; `WeakRef` under rc.
- Stack promotion: an object of a weak-target class made locally and weakly referenced through a call
  is not promoted (the test reads it after the frame through the `WeakRef` and gets `undefined` or the
  live object, never garbage).
- gc: the existing `WeakRef`, `WeakMap` and `WeakSet` tests of the default library still pass.

## Left to the builder

- The op set (`ts.WeakRefNew`, `ts.WeakRefDeref`, or members on an existing op family), the lowering
  of the count, the tombstone and the gc link, and where `isHandle` is extended.
- How the release routine of a class tests the count (one load and branch before the free).
- The exact text of the messages, in the style of the existing own messages.
- Whether the borrow of a `deref()` result reuses `anyPlace` chains as they are or needs a new chain
  kind; the rule above is what it must enforce.
- Spec section 84 in `tslang/docs/superpowers/specs/2026-09-24-own-memory-model-design.md`, the user
  guide (`docs/own-memory-model.md`), `docs/memory-models.md` and the readiness page.
