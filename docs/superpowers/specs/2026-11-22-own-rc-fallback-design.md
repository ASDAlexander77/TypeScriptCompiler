# `-mm=own` with a counted fallback: keep rc's count where single ownership cannot be proven

Date: 2026-11-22. Status: draft for approval.

## Intent and why

`-mm=own` infers one owner per heap block (move, else borrow, else compile error) and counts nothing
at run time. Single ownership works, but extending inference is slow work: inside function bodies
every code shape needs its own rule, and what is still refused is a long tail (about 186 corpus
files, and a 55-file source change before `native_ts_graphics` compiled). The evaluation of
2026-11-22 kept rc and own on the same IR ([[rc-own-sharing-is-structural]]). It found that the
biggest lever left is to fall back to rc's count for the one object inference cannot place, in
place of an error.

Own is already rc's IR with the retains erased that inference proves belong to the one owner
(`tslang/tslang/transform.cpp`: the verifier, then `OwnershipSignaturePass`, then
`OwnershipInferencePass`). Where inference cannot erase a retain, it reports today. With this
design it can instead **leave that retain in place**, and the object is counted from then on.

What this changes in own's promise: "no counting at run time" becomes "no counting except where the
compiler says it kept a count". The fallback is opt-in per program and every place it is used is a
warning, so a program that does not use it compiles as before.

## Decisions taken

- **Releases honor a count always under own** (user, 2026-11-22). Every release under own checks the
  block's header word for extra owners, whether or not the program uses the fallback. There is one
  ABI under own and one default-library build. A program, a library and the default library can be
  mixed freely, whatever each was compiled with.
- Not re-opened: rc and own stay on shared IR; `ReadonlyArray` and `as const` are not ownership facts.

## Goals

- A program compiled with `-mm=own --own-fallback=count` compiles where own's inference reports one
  of the tier 1 errors below. The object involved is counted, the program runs correctly, and a
  warning says where.
- Without the flag, own reports exactly what it reports today: same errors, same accepted programs,
  and the same erased retains (the IR before lowering is unchanged).
- Own's release keeps its cost: a block with one owner is freed by a release that tests a single
  word against zero.
- Measurable gain: the corpus files accepted under own with the flag rise from the current figure
  (re-measured as step 1 of the plan), and `native_ts_graphics` builds from its source *before* the
  55-file own port.

## Non-goals

- Making the fallback the default. Own without the flag stays the strict model.
- The refusals that have no retain to keep: errors from MLIRGen, from `OwnershipSignaturePass`, and
  the by-design refusals that point at `Shared<T>` (`new Map<string, Node>()`, `new Array<Node>()`,
  `filter` over class instances, `reduce` without an initial value). Tier 3 below lists them; each
  needs its own design.
- Cycle collection. A cycle through counted objects leaks, as under rc.
- Changing rc. The fallback reuses rc's ops and placement; rc's lowering is not touched.

## Anti-goals (a failure even if it "works")

- Any change to the code or the decisions of a program that does not pass the flag, beyond the
  release's new test of the header word.
- A counted object freed earlier than own's analysis already assumes. The fallback may only make an
  object live **longer** than inference thinks, never shorter. This invariant is what keeps every
  fact other functions and modules rely on (frees nothing, fresh result, borrow of a field) sound
  without re-deriving it.
- A silent fallback: every counted place is a warning, and a warning can be made an error.
- Measurable slowdown of the ray tracer under own without the flag.

## Constraints found in the code

1. **Counting belongs to the block, and its releases are everywhere.** Under own a release frees
   unless the block is immortal (`OwnershipRoutineLogic::emitIfLastReference` → `emitIsMortal`), and
   a type has no retain routine (`TypeOfOpLowering` leaves it empty except for `Shared<T>` handles).
   A kept retain is only safe if every release of that block - in other functions, other modules,
   type release routines and the default library - decrements instead of freeing. That is why the
   release check is always on.
2. **The header word is in use under own.** It holds 0 (one owner), `HEAP_BLOCK_IMMORTAL` (all bits),
   or the WeakRef weak count in the low bits with the dead bit at `bits - 2`
   (`Defines.h`, own spec 84). The top bit is the shared bit of a counted block. A count of extra
   owners needs bits of its own.
3. **rc counts every owner; own's first owner holds an implicit share.** rc's `__tslang_dec_ref`
   frees when the count goes from 1 to 0. Own always erases the retain of a block's first owner
   (that is what inference proves), so the retains a dry run leaves are second owners. The word
   under own therefore counts **extra** owners: 0 is "one owner", which is what every block holds
   today.
4. **The precedent exists.** `OwnershipInferencePass::runOnFunction` runs a dry run; where it fails,
   `makeStringCopies` repairs the function (a string copy for each retain the dry run could not
   erase, and for each field read lent to a call that may free it), then the real run decides. For a
   `Shared<T>` handle the same repair already **keeps a retain** (and a release after the call)
   instead of copying. The fallback widens that branch from handles to every heap object.
5. **Not every refusal has a retain to keep.** Probed on 2026-11-22 with `--own-skip-inference`
   against `-mm=rc`, both at `--emit=mlir-affine`:
   - `h.node = n; g.node = n` (a second owner) and `let t = a[i]; a[i] = a[j]; a[j] = t` (a swap
     through a local): own's IR before inference has exactly rc's retains and releases.
   - `const first = h.node; h.node = new Node(1); print(first.value)` (a borrow used after its owner
     let go): under own MLIRGen makes no local and no retain (`__owned_result_named` load), where rc
     makes a local with `ts.RetainSlot`. There is nothing to keep. The same shape is refused for a
     string too (`const first = h.name; h.name = "b" + 2; print(first)`), while a `Shared<T>` field
     in that shape compiles.
6. **Ops written for a block that dies at its release.** `TakeOutOfFieldOpLowering` empties a field of
   an object about to be released, assuming the release frees it. `StackPromotionPass` marks a
   promoted block immortal and erases its retains and releases, assuming it dies with the frame.
   `RetainCellOpLowering` is an error under own. The share routine (threads) is empty except under
   rc.

## Approaches considered

| | How a release knows the block is counted | Trade-off |
| --- | --- | --- |
| **A. Count in the header, checked by every own release (chosen)** | Run time: the word is 0 for one owner, more for extra owners | One ABI; any mix of modules is safe; per object, not per type. Costs a changed release shape everywhere (expected to be neutral, to be measured). |
| B. The same, only in programs built with the flag | Run time, but only in flagged builds | Unflagged builds byte-identical. Needs a second default-library build per target, and a module built without the flag that releases a counted object frees it while it is shared. |
| C. Counted types, decided at compile time (as `Shared<T>` is) | Static: every release of a counted class decrements | No run-time test for other types. A type counted in one module is not counted in another that releases it inline, so it needs the fact exported, and it is coarse: one fallback counts every object of the class. |

The user chose A.

## Design

### 1. The header word under own

Under own the word becomes three fields plus the existing markers. Exact widths are the builder's
call; the proposal is:

| Bits (64-bit word) | Bits (32-bit word) | Meaning |
| --- | --- | --- |
| all set | all set | `HEAP_BLOCK_IMMORTAL`, tested first (unchanged) |
| 63 | 31 | shared bit (unchanged; set only by the share routine, see 6) |
| 62 | 30 | dead bit: a tombstone kept for WeakRefs (unchanged) |
| 31-61 | 15-29 | **extra owners**: owners beyond the first, kept by the fallback |
| 0-30 | 0-14 | weak count (narrowed from 0-61 / 0-29) |

An increment that would overflow either count makes the block immortal instead (a leak, never a
double free). The builder checks the 32-bit limits (32767 extra owners or WeakRefs) against the
default library's tests. *(My call: equal halves. Narrower weak counts are acceptable because WeakRef
use is rare.)*

### 2. The release under own

`emitIfLastReference` under own (for a block that is not a `Shared<T>` handle) becomes:

```
word = load header
if word == 0:                    # one owner, no WeakRef: the common case
    destroy (release fields) and free
else if word == IMMORTAL:        # literal, keeper, stack block
    nothing
else if extra(word) > 0:         # counted: one owner fewer
    store word - (1 << extraShift)
else:                            # last owner, WeakRefs remain
    destroy and bury (emitFreeOrBury, unchanged)
```

Today's path tests `word != IMMORTAL`, destroys, and then `emitFreeOrBury` loads the word again for
the weak count. The new fast path tests `word == 0` once and frees without the second test. The ray
tracer timing under own, before and after, is the check that this is neutral or better. All
releases under own take this shape, with or without the flag, including type release routines, the
array and `any` routines in the runtime that release elements, and the default library's own
build. The plan lists them by grepping for `emitIsMortal` and the runtime's free paths.

The retain under own, emitted only for a retain the fallback kept (marked, see 3), adds
`1 << extraShift` unless the block is immortal. `RetainOpLowering` and `RetainSlotOpLowering` keep
their error ("ownership inference left a retain behind") for an unmarked retain.

### 3. Where inference keeps a retain (tier 1)

Inside `OwnershipInferencePass::runOnFunction`, in the step that repairs a function after a failed
dry run, with `--own-fallback=count`:

- **A retain the dry run could not erase** (the set `makeStringCopies` walks: not erased, not a
  capture retain, not a field-borrow retain) of a value that is not a string:
  kept, marked `__own_counted`. The releases rc placed for the same value are kept too, and the
  real run must neither erase nor move them. A string keeps its copy as today.
- **A field read lent to a call that may free it** (`callCopies`) of a value that is not a string:
  a retain before the call and a release after it, unless the callee keeps the argument. This is
  the existing handle branch, widened.

A function that passes the dry run is decided exactly as now.

**The unit is a refused value, with all of its rc ops.** *(My call, from the builder check.)* What is
counted is the value whose check failed, and it keeps every retain and release rc placed for it,
not only the retains the dry run left over. Its birth reference has no retain (rc hands the first
taker the value's own reference, `OwnershipInferenceMoves.cpp`), so keeping the rest leaves a word
that counts extra owners. Keeping only a later retain is not enough. In `h.node = n; h.node = other;
g.node = n`, the move's retain and the source's releases are what keep `n` alive until `g.node`
retains it.

- `checkValueMoves` already commits a value's move erasures only when its check passes, so a refused
  fresh value keeps all its ops. That is the model.
- Some report sites commit erasures for a value they report: the array-view report puts the
  reported retains into `fieldBorrowRetains`, which `decideRetains` erases and `makeStringCopies`
  skips (`OwnershipInferenceClosures.cpp`). The builder audits every tier 1 report site, including
  every site that records into `fieldBorrowRetains` or `captureRetains`. A site that cannot leave its
  value's rc ops whole stays an error and is out of tier 1. The prototype's measured figure replaces
  "63 for sure".
- A refusal with no rc op to keep, such as a value rc hands to two consuming declarations with no
  retain (the comment in `OwnershipInferencePass::decideRetains`), stays an error.

**The word is rc's count minus one.** rc's allocation writes 0 into the word, and its first owner
takes a retain. Own's word counts owners beyond the first, so for each counted value the kept
retains must come to rc's minus exactly one. A value rc creates at count 0 in this function (a
`new`, an `__owned_result` temporary) keeps its first owner's retain erased, as own does today. A
value that arrives already owning a count (a call's result, a kept parameter) keeps all of its
retains. Keeping one too many leaks the block silently, so Task 3 of the plan checks the balance
for each counted value in a debug build. Its tests watch for leaks, not only output.

**How the real run knows.** *(My call.)* The dry run records each failing value with the diagnostic
it would have reported: the error and its notes. `reporting()` today only counts failures. The
repair marks those values counted (`__own_counted` on their retains, and the value in a set the real
run reads). The real run then exempts a counted value from its own checks (moves, uses after a move,
borrow stores) and leaves all its rc ops in place. Every other value is checked as today. An error
the real run raises on a value that was not counted is still an error, so in a function with
several errors, one kept count suppresses only its own value's. The warning carries the recorded
diagnostic as its notes.

**Values own copies today stay copied.** *(My call.)* `isCopyableValue` covers strings and `any`,
and `isCopyableValueAt` covers arrays in `@copies` functions. The repair keeps copying those as
today, so nothing an accepted program can observe changes. The fallback counts only where the repair
would otherwise leave an error. That includes a string whose retain `copyForRetain` refuses (a
different block, several uses, a borrow of a known parameter): today an error, with the flag
counted.

Why this is sound (to be stated in the code as the invariant the builder protects): rc's IR is
balanced, and every erasure own makes is a balanced one (a second owner's retain with the releases
it pairs with, or a move that erases the source's releases). Keeping a value's retains and releases
as rc placed them leaves that value's ops balanced, with the first owner's share implicit (word 0).
Every kept retain makes the block live longer than own's analysis assumes, never shorter, so every
fact derived elsewhere stays true.

### 4. Ops that assume a block dies at its release

- **`TakeOutOfField`**: lowering empties the field only when the object's word is 0 (it will be
  freed). When the object has extra owners, its field stays and what it holds gains one extra owner
  (it is now held by the field and by the place it was handed to). The immortal case is unchanged.
- **Stack promotion**: a value with a marked retain, or reachable from one, is not promoted. A kept
  retain is an escape.
- **Keepers / `MakeImmortal`**: unchanged. Marking a counted block immortal discards its counts,
  which is safe because nothing frees an immortal block.
- **WeakRef / `@weak`**: unchanged in behavior. A tombstone appears only when the last owner, counted
  or not, lets go.
- **Cells** (`RetainCellOp`): not in tier 1. The closure refusals stay errors (tier 2 candidate).

### 5. Diagnostics

Each kept retain is a warning at the location of the error own would have reported, with that
error's text as a note, so the reason is the one users already know:

```
w.ts:8:5: warning: -mm=own counts this object here (--own-fallback=count); a cycle through it is never freed
w.ts:8:5: note: 'n' is used here after its value was moved
w.ts:7:5: note: value moved here
```

One warning per kept retain; no summary line. `--own-fallback=error` (the default) keeps today's
errors. *(My call: the flag takes `error|count` so a later `copy`-like mode or a `-Werror`
equivalent fits; a warning can already be made fatal with the existing warnings-as-errors option,
if there is one - the builder checks.)*

### 6. Threads

A counted object that crosses to another thread needs atomic counts, and a rule per module cannot
see it: a module with no threads can hand a counted object to another module, or to the default
library, which shares it. **Decision (user, 2026-11-22): check at share time.**

- Own's share routine (`getOrCreateShareRoutine`, empty today except under rc) is built under own
  too. It walks what is being shared, as rc's does, and sets the existing shared bit on **every
  mortal block it reaches**, as rc's does. Marking only the counted ones would not do: a one-owner
  child of a shared counted block is reachable from both threads, and either can take a count on
  it. It runs at `Thread.spawn` of a closure, a worker's message and `Shared.share`. The walk stops
  at an immortal block or one already shared.
- The release's slow path (2) tests the shared bit first. With extra owners, it does an atomic
  subtract, as in rc's `__tslang_dec_ref`. With none, the caller is the last owner and destroys the
  block (bury if WeakRefs remain). The retain kept by the fallback does the same, as `emitIncRef`
  does. The fast path `word == 0` is unchanged, since a shared block is never 0.
- The bit is set only by the thread that holds the block before any other can reach it, the rule
  rc's design already relies on.
- Cost: the walk at each share, in every own program, with or without the flag. Releases pay
  nothing more. The plan measures a spawn-heavy test (the workers tests) before and after.
- A weak count on a shared block stays non-atomic, as today: a module that makes WeakRefs may still
  not share their targets (spec 84).

### 7. Tiers

| Tier | Refusals | What the fallback needs |
| --- | --- | --- |
| 1 (this design) | Inference errors where the dry run leaves a retain or a lent field read: a second owner, use after move, a borrow stored, returned or captured, an element borrow stored, a global lent to a call that may release it | Keep the retain (3) |
| 2 (next design) | A borrow used after its owner let go, where MLIRGen under own elides the local (`const first = h.node`); closures that escape and are still used (cells) | MLIRGen keeps rc's local and `RetainSlot` for the shape, so inference can erase it or keep it; counted cells |
| 3 (out of scope) | MLIRGen and signature-pass errors; by-design refusals pointing at `Shared<T>`; missing `@keeps`/`@lends` on exported functions | Each its own design |

Tier 2 fixes the string case of the same shape as well, which is refused today.

Corpus sweep of 2026-11-22 (Release compiler on `readonly-ownership-facts` at `ab5f34ac`; each file
of `tslang/test/tester/tests` compiled `--emit=llvm --no-default-lib --opt --opt_level=3`): 1072
files, 1032 compile under gc, **863 under own, 169 refused**. By the first error of each refused file:

| Files | First message (names elided) | Tier |
| --- | --- | --- |
| 21 | `'X' is used here after its value was moved` | 1 |
| 26 | `... borrows (a field / an element / a captured variable / 'X') and cannot be stored, returned or captured` (7 wordings) | 1 |
| 7 | `... borrows the result of 'X' and cannot be stored, returned or captured` | 2: no retain in the caller (below) |
| 5 | `a value read out of the global 'X' is lent to a call that may release or overwrite it` | 1 |
| 4 | `a value takes a second reference; -mm=own cannot prove a move or a borrow here yet` | 1 |
| 28 | `... borrows 'X' but is used here after it may be released or overwritten` (4 wordings) | 1 where the borrow is a lent field read or has a retain; 2 where MLIRGen elided the local |
| 4 | `'X' borrows the result of 'X' but is used here after ...` | 2 |
| 9 | closures: `captured by a closure that escapes, and is still used`, `cannot move 'X' into it` | 2 |
| 65 | everything else: 30 messages - parameters assigned or kept, `@keeps`-shaped, object literals, `any` element access, globals' initializers, function values of another type, `Object.keys` | 3, mostly |

So on the first error tier 1 reaches **at most 56 files, and up to 84** if the 28 borrow-after-release
files all have a retain. These are upper bounds. Some report sites erase what they report (3, the
audit), and a file whose first error falls back may still be refused by a later one. The plan
measures the real figure with an inference-only prototype before any change to the release.

**A borrow of a call's result has nothing to keep in the caller.** Probed with `g.node = f(h)`
where `f` returns `h.node`: under rc and under own before inference, the caller has no retain for
the result; rc's count for it is taken inside `f`, which own erases because `f` lends. Counting that
shape needs `f` itself to keep its retain, a decision across functions, so it is tier 2.

## Measuring

- **Corpus**: every file of `tslang/test/tester/tests` with `--emit=llvm --no-default-lib --opt
  --opt_level=3`, under gc, own, and own with `--own-fallback=count`. Report accepted counts, and
  the warnings per file.
- **Soundness**: `tslang/test/fuzz/mm_fuzz.py` with a mode that adds the flag, comparing output with
  gc; the `leaks` command against rc for peak growth. Run on Linux too (glibc reports double frees
  the Windows heap hides).
- **Real code**: `native_ts_graphics` at the commit before its own port, built under own with the
  flag, against its 525 tests; count the warnings.
- **Cost**: the ray tracer under own without the flag, before and after the release change (ahead of
  time, best of several runs), and the binary size of a default-library test.
- **No change without the flag**: the full own test set and the default library's own tests pass
  unchanged; the own-errors tests (`tslang/test/tester/own`, about 359 expected errors) still fail
  as expected.

## Left to the builder

- The exact bit widths, and where `extraShift` lives (`Defines.h` beside the existing markers).
- Which runtime (C/C++) routines release elements under own, and making each honor the word.
- How the real run is told a retain is kept (an attribute, `__own_counted`, is the proposal) and
  that the value's releases are fixed.
- Whether the warning location is the retain's or the original error's when they differ.
