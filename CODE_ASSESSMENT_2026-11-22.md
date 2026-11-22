# Code assessment: recent work on tslang (2026-11-22)

This covers the code changed in the recent runs of compiler PRs: the run-time-keys dictionary,
Iterable/Generator support, yield in finally, `a[i] = v` growth, and the `-mm=own` follow-ups (TSLANG 152-160,
PRs #972-#989, DefaultLib #114-#117).

**Summary:** the code works and is well tested, but it is getting harder to change safely, chiefly in the `-mm=own`
ownership checker.

---

## 1. What is good

### 1.1 Validation
Each change goes through:

| Check | Scope |
|---|---|
| ctest | about 8,850 tests (compile, JIT, rc, own, diagnostic locations) |
| wasm suites | tester and DefaultLib in wasm32/wasm64, debug and release, rc/none/own |
| DefaultLib native suite | four memory models |
| tsc compiler-tests sweep | about 3,260 TypeScript compiler tests |
| differential fuzzer | `tslang/test/fuzz/mm_fuzz.py`: `run` (results compared across models) and `leaks` |
| leak probes | 1M-20M rounds, peak memory must stay flat |

This is why regressions were caught within hours: #982's growth change broke 8 DefaultLib own tests, which #983
fixed. #984's interface cast let a `next` of another signature through and trapped on wasm.

### 1.2 Error messages
Own errors say what is wrong and what to do, for example:
- "catch it inside the try statement, or throw a string"
- "give the object a Shared<T>"
- "return it after the try statement, or yield it"

Diagnostic-location tests under `tslang/test/tester/diagnostic-location/` pin their positions.

### 1.3 Lowering to TypeScript where it fits
Generic helpers are written as TypeScript source and compiled once per module:
- `__iterator_values`, `__array_values`, `__string_values`
- `__dictionary_assign_values`
- the `.array_grow.<type>` functions

They are easy to read, and they get the ownership checks for free.

---

## 2. What worries me

### 2.1 The own checker is a growing list of special cases
Each gap found gets its own pattern exemption, for example:
- `growsOnly` SetLengthOf
- `own_facts::reachesSharedValue`
- `isBoundForCall`
- the exemption for kept params (`@keeps`)
- the delegate exchanged into a local of `next()` around each `yield*` step (#988)

Each is reasonable on its own, but they interact in ways that are hard to predict. #982 broke 8 DefaultLib tests
through one such interaction: growth through a `Shared` payload read as a push.

There is no small set of rules that explains why a program is accepted or refused, so every new feature finds a new
gap. Item 3 of the 2026-11-22 rulings shows it best: `this.d.next()` on a generator held in a field is refused
everywhere, in a generator or a plain class. #988 works around that rule for `yield*` instead of fixing the rule.

### 2.2 Desugaring through made-up AST nodes with magic names
MLIRGen builds TypeScript syntax trees with `NodeFactory` and generates them again, using names that carry meaning:
- `.yd` (the `yield*` delegate)
- `.r`, `.v`
- `$fin<pos>_m`, `$fin<pos>_e`, `$fin<pos>_x` (yielding finally)
- `.thrown`, `.mode` (generator resume)

This is quick to write, but fragile:
- **The `$fin` check.** Item 1 (#986) decides whether a catch clause is the compiler-made catch-all of a yielding
  finally by testing whether its variable name starts with `"$fin"` (`MLIRGenStatements.cpp`, `mlirGen(TryStatement)`).
  That is a hack I added.
- **Locations.** The made-up nodes get copied positions, which is why errors sometimes point at the wrong place or
  carry an odd "called from" note.

### 2.3 `GenContext` keeps gaining flags
`tslang/include/TypeScript/MLIRLogic/MLIRGenContext.h` keeps gaining fields:
- `yieldingFinally`
- `yieldStarDelegate`
- `throwCaughtBeforeYieldingFinally`
- `finallyGuardScope`, `finallyEscapingJumps`
- `isUnwindCleanupScope`, `unwindCleanupOp`, and dozens more

Each one is a bit of state that every copy of the context carries, and it is easy to forget to reset one at a
function or try boundary. `yieldingFinallyOf()` needs its `funcOp` check for exactly that reason: the pointer leaks
into the bodies of nested functions.

### 2.4 Some fixes are partial
These are written up in `TSLANG_ISSUES.md`, but the list grows faster than it shrinks:

| Item | What is left |
|---|---|
| TSLANG 157 (#986) | Only a `throw` written in the `try` is caught at compile time. An object thrown by a called function still stops at run time. The complete fix moves the caught value instead of copying it. |
| TSLANG 156 (#985) | `throw()` is not handed on to the `yield*` delegate. |
| own `yield*` (#988) | Still refused: a variable or parameter as the delegate, `yield* this.range()` in a class method, a hand-written `d.next()` on a generator in a generator variable, `Generator<T>` of objects. |
| leaks | When a call that was passed a temporary argument throws, the argument leaks (for example `sum(overBoom())`). |

### 2.5 File size
`tslang/lib/TypeScript/MLIRGenImpl.h` is over 20,000 lines in one header. A small edit there rebuilds most of the
compiler, which slows every fix.

### 2.6 Comment style
The comments explain why, which is good, but many run to paragraphs of history ("it was..., now..."). That history
belongs in commit messages and `TSLANG_ISSUES.md`; in the code it buries the current rule.

---

## 3. What I would do about it

In order of value:

1. **Write down the own rules.** A short spec of what may borrow what, and when a call may drop something. Then
   check each pattern exemption in `Ownership*.cpp` / `OwnershipFacts.h` against it. Several would likely collapse
   into one rule, for example "a value the frame owns can't be dropped by an unknown call". This would also settle
   the general case behind item 3: calling an unknown function while a field of `this` is borrowed.
2. **Mark generated nodes with attributes, not name prefixes.** For example, an `InternalFlags` bit on the made-up
   catch clause instead of the `$fin` string test, and the same for `.yd` and `.thrown`.
3. **Split `MLIRGenImpl.h`** into the `MLIRGen*.cpp` files that already exist, to cut rebuild time.
4. **Trim comments to the current rule**, and leave the history to commits and `TSLANG_ISSUES.md`.
5. **Burn down `TSLANG_ISSUES.md`'s open items** before adding new features in the same area.

None of this blocks the next release, and the current batch is well validated. But the `-mm=own` model especially
needs item 1 before it grows much further.
