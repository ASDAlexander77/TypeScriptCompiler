# `-mm=own`: readiness

Where the single-owner memory model stands, measured on `main` at `363b13ce`, 2026-11-15, with the
Release compiler on Windows x64. The linked-list rows of [Shapes that matter](#shapes-that-matter)
were rejected in that measurement and are fixed by the change that adds this page; the table shows
them as they are with it. The model itself is described in
[own-memory-model.md](own-memory-model.md); the tiers and their exit criteria are in the
[production-readiness plan](production-readiness-plan.md).

## Verdict

**`-mm=own` is not ready for production. It is usable for new code written for it, and it is a
preview for porting existing TypeScript.**

- Everything its design set out to build is built, and the tracker has no open defect against it.
- Nothing unsound was found in this round of measurement.
- It still rejects about one in five programs the other models accept. A back-reference is a
  `WeakRef<T>` (spec 84), which a module that also spawns threads with closures cannot use.
- Memory-safety bugs were still being found in real code in the three days before this was written.
- None of the three tier 3 exit criteria in the plan is met.

It stays in tier 3 ("experimental") in the plan's terms.

## Scorecard

| Area | Status | What it rests on |
| --- | --- | --- |
| Design | Complete | Spec phases 0-8 and the later designs (parameter decorators, facts from object files, immortal keepers, generic containers, `WeakRef<T>`, `@weak` fields) are merged. |
| Tracker | Clean | No open GitHub issue. The own items of `TSLANG_ISSUES.md` (31-33, 39-42, 47-85) are fixed, with one remnant: whole-program facts (item 48). |
| Soundness | No failure found, not proven | See [Soundness](#soundness). |
| Coverage of TypeScript | Partial | 707 of 882 corpus files (80%) compile. See [Coverage](#coverage). |
| Default library | Nearly complete | Its `-mm=own` build passes its tests; 2 of 230 are skipped as refused by design (`array_flatMap_shared`, `map_groupBy`). |
| Separate compilation | Works, with a cost | `--own-facts` and shared libraries carry facts. A call of a method of an exported class that is not `@final` still ends every borrow. |
| Platforms | In the release matrix | Windows, Linux, wasm32/wasm64 and Android run the own legs in CI. Only Windows was run for this document. |
| Tests | Broad | 560 files in `tslang/test/tester/own` (359 of them expected errors), and the general corpus runs under own in CI. |
| Real code | One program | native_ts_graphics compiles and passes 525/525 under own, after a 55-file source change. See [Real code](#real-code). |
| Tier 3 exit criteria | 0 of 3 | See [Exit criteria](#exit-criteria). |

## Soundness

The claim of the model is that an accepted program never frees a block twice, never reads a freed
one and never leaks. What was checked:

- **Differential fuzzing.** `tslang/test/fuzz/mm_fuzz.py`, 400 seeds from 915000: 738 programs
  checked, 353 accepted by own, each built ahead of time and compared with `-mm=gc`. No mismatch,
  no crash, under own or rc.
- **Churn loops.** Four shapes that older notes or the guide called refused, and that compile today
  (a parameter a generator assigns, a value merged from `?:` stored into a field and then used, a
  new object passed through a function that returns its argument, a reassigned `Shared<T>`
  parameter): one to three million iterations each, ahead of time. Output equal to gc's, and peak
  memory flat at 5.7 MB.
- **The corpus.** No file made the compiler crash under own.

This is weak evidence, for three reasons. Half of what the fuzzer generates is rejected, so about
350 programs ran. It was Windows only, and the Windows heap hides double frees that glibc reports.
And compiling one real codebase turned up, between 2026-11-13 and 2026-11-15, a read of freed
memory (item 50), a double free (item 83), a fix that leaked 2.7 GB and was backed out (item 80),
and a compiler crash in about one run in 500 (#867). All are fixed. The rate says more are there:
the change that added this page accepted a walk that cut the list ahead of itself (`const nx =
n.next; n.next = null; n = nx`) and read the freed node, found the same day by a probe written for
the next change (spec 83).

## Coverage

Every file of `tslang/test/tester/tests` was compiled with `--no-default-lib --opt --opt_level=3`
under gc and under own. Of 915 files, 882 compile under gc; of those, 707 compile under own and 175
are rejected. The corpus holds own's own regression tests, which raises the figure, and a few
multi-file tests compiled alone without their facts, which lowers it.

The rejections, by message:

| Count | Message | Kind |
| --- | --- | --- |
| 23 | takes a second reference; cannot prove a move or a borrow here yet | Not yet |
| 19 | borrows a field but is used after it may be released or overwritten | Rule, often conservative |
| 19 | used after its value was moved | Rule |
| 15 | borrows a field and cannot be stored, returned or captured | Rule |
| 9 | an object literal that holds a value it owns is moved only into a local or returned | Not yet |
| 8 | a parameter, kept here, but the callers cannot be made to give it up | Needs `@keeps` |
| 7 | merged from several branches (`?:`, `&&`, `\|\|`, `??`, a loop) | Not yet |
| 7 + 6 | borrows an element (stored, or used after a release) | Rule |
| 6 + 4 | borrows the result of a call | Rule |
| 5 | captured by a closure that escapes, and still used | By design |
| 5 | a global's value lent to a call that may release it | Conservative |
| 42 | 22 other messages, one to five files each | Mixed |

Measured again on 2026-11-16 after spec 86 (values merged by `?:`, constant merges, an optional record's
field read): of 890 files that compile under gc, 724 compile under own; "takes a second reference"
dropped from 24 files to 19 and "merged from several branches" from 7 to 5. No file went the other way.

After spec 88 (a literal that takes a `let`, `?:` of two fresh values, nested and global records,
literals tagged into a union): of 892 files that compile under gc, 728 compile under own; "an object literal that holds a value
it owns" dropped from 9 files to 6 and "takes a second reference" from 19 to 18. No file went the other way.

After spec 89 (a method value takes its object): 729 compile under own; `00bound_method_count` runs.

After spec 90 (a `let` in a `try`, a `switch`'s narrowed parameter, `??=` / `||=` / `&&=` assign only
where needed): 735 compile under own (`logicalAssignment5*`, `03union_type*`, `00try_cleanup_then_catch`).
No file went the other way.

After specs 91-92 (a read-only `view` borrows its elements; a value boxed into `any` only to be cast
back borrows; `<Opaque>x` borrows): 740 compile under own (`00any_types`, `00any_unbox_interface_class`,
`00opaque_nullable_compare`, `00union_global_init`, `00union_member_padding`). No file went the other way.

After spec 94 (a `let` reassigned from other locals borrows each value, a list walked from a local head, a `?:` of `@value` literals holding new objects, one object literal given to interface parameters more than once): 751 of 900 compile under own (`00for_optional_class_condition`, `00interface_object`, `00conditional_nested_record_literal_temporary`). No file went the other way.

After spec 95 (`return` inside `using`, a parameter's `let` after `??=` or `||=`, a `?:` of literals given to an interface or held by a `const`, a kept parameter in a returned literal): 758 of 944 compile under own (`00array7`, `00using_unwind_outer_locals`, and five more from the corpus growing and from other fixes on main: `00generator6`, `00owned_unions`, `00safe_cast_bug`, `00string_union_elements`, `00tslang_issues_92_124`). No file went the other way.

About 50 of the 175 say in the message that the compiler cannot prove something yet. The rest are
the model's rules; some of those are the single-owner design working as intended, and some are the
analysis being more careful than the program needs.

### Shapes that matter

Probed one by one under the JIT:

| Shape | Result |
| --- | --- |
| Walking a list: `for (let c = this.head; c; c = c.next)` | Compiles and runs |
| Walking a list from a local head: `let node = head; while (node) node = node.next` | Compiles and runs (spec 94); giving the head up or cutting the list during the walk is rejected |
| Reassigning a `let` from other locals: `let b = q; ... b = r` | Compiles and runs (spec 94): it borrows each |
| `return s` inside a `using` block | Compiles and runs (spec 95), also when dispose or the constructor throws |
| A literal holding a kept parameter, returned: `return { k }` | Compiles and runs (spec 95); not from an exported function, whose parameters are not kept |
| A `?:` of literals holding new objects, as an interface or in a `const` | Compiles and runs (spec 95) |
| List `push`: `n.next = this.head; this.head = n` | Compiles and runs |
| Appending at the tail, cutting the tail: `c.next = n`, `c.next = null` after the walk | Compiles and runs |
| List `pop`: `this.head = this.head.next`; unlinking: `c.next = c.next.next` | Compiles and runs |
| Using the popped node afterwards: `this.head = h.next; return h.v` | Rejected: the node is freed by the assignment; read it first |
| Push through a constructor: `this.head = new Node(v, this.head)` | Compiles and runs, when the constructor is in the same module and never throws |
| Walking by assigning a parameter: `n = n.next` | Compiles and runs |
| Tree insert by recursion on a parameter: `t.l = ins(t.l, v)` | Compiles and runs, when the function is in the same module and never throws |
| Cutting while walking: `const nx = n.next; n.next = null; n = nx` | Rejected: the cut frees what `nx` names (accepted by mistake from 2026-11-16 until spec 83) |
| Tree insert and sum as methods recursing on `this` | Compiles and runs |
| Parent link: `k.parent = new WeakRef(this)` | Compiles and runs (spec 84); `deref()` gives `undefined` once the parent is freed |
| `new WeakRef(a)` | Compiles and runs under own, gc and none; rc refuses it |
| Parent link as a field: `@weak parent: Node \| null`, `c.parent = this`, `this.parent.v` | Compiles and runs (spec 93); a read after the parent was freed stops the program |
| `new Map<string, Node>()`, `filter` over class instances | Rejected by design; `Shared<Node>` |
| Swap through a local: `let t = a[i]; a[i] = a[j]; a[j] = t` | Rejected by design; `[a[i], a[j]] = [a[j], a[i]]` |
| `x ??= new B()` on a parameter, a local or a field | Compiles and runs (spec 90): a plain assignment where `x` is null; `??=` of an accessor is still rejected |
| `let t = p` inside `try`, `switch (s.kind)` over a union parameter | Compiles and runs (spec 90): borrows, as outside `try` and under `if` |
| `const w = a.view(0, 1)` only read | Compiles and runs (spec 91): borrows the elements; a call or `pop` before a read of it is rejected |
| `(<T><any>s).f`, `const box = a as any; (box as T[]).push(x)` | Compiles and runs (spec 92): the box borrows |
| `<Opaque>x` of a local or a field; `<Opaque>new Item()` | Compiles and runs: borrows (spec 92); a temporary is rejected - keep it in a variable first |
| `return set ? n : undefined`, `h.n = set ? n : null` of an owned local | Compiles and runs: `n` moves on the way that chooses it, and is released on the other |
| `return first ? shared : other` of globals never reassigned, `c ? 1 : "one"` | Compiles and runs: the merge owns nothing |
| `let p: Node \| null = on() ? mk(3) : null` at the top level | Compiles and runs |
| `return set ? n : m` of two owned locals | Compiles and runs (spec 88): both move into what is selected, and the other is released there |
| `return { s, n }`, `const o = { s, node }; return o` with `let`s | Compiles and runs (spec 88): the record takes them |
| Method values: `h.f = v.get`, `return v.get`, `let f = v.get` | Compiles and runs (spec 89): stored or returned it takes `v`; kept beside `v` it borrows |
| A method read through an interface, stored or returned (`return o.fn`) | Rejected, with what to do instead |
| A closure that assigns a parameter it captures | Rejected |
| `m.forEach((v, k) => { total += v; })` on a `Map` or a `Set` | Compiles and runs |
| `m.forEach((v, k) => { names.push(k); })` | Compiles and runs: a callback declaring fewer parameters than `forEach` passes is given a copy of a string it keeps |
| try/catch/finally, unions, `any`, async functions (11 corpus files) | Compile; async was not run for this document |

## Real code

native_ts_graphics (the `build-own` of powder_sandbox) is the one outside codebase compiled under
own. The sweep that got it there produced compiler items 50-85 in three days, and the source needed
a 55-file change (+432 / -351): getters that return an object of the caller's own instead of a lent
one, statements reordered so a value is used before it is given away, global rows copied where
kept, `@final` on classes, and swaps in place of a sort through a temporary. With those it passes
its 525 tests under own, as under gc and rc.

That is the realistic cost of a port today: the program works, and a noticeable share of it is
rewritten to say who owns what.

## Performance

Covered in [performance.md](performance.md) and [memory-models.md](memory-models.md). In short: the
ray tracer under own runs within about 1.4x of the Rust version with `@value` vectors, memory stays
at the working set, and there is no collector or count at run time. Performance is not what holds
the model back.

## Exit criteria

From the plan's Phase 5 row for `-mm=own`:

| Criterion | Met | Notes |
| --- | --- | --- |
| Close the open items: optional accessor fields, capturing callbacks | Half | An optional interface field filled by an accessor is a clean error under every model, not supported. Capturing callbacks over `Map` and `Set` work, keys and values pushed into a captured array included; an error found in such a callback points at its source. |
| A written specification of the ownership rules | No | The rules exist as a design log of about 5,000 lines (`tslang/docs/superpowers/specs/2026-09-24-own-memory-model-design.md`) and as the user guide. Neither is a specification someone could implement or check against. |
| Error messages reviewed by someone new to the model | No | Not done. Messages name the variable with or without `--di` (`'item' is used here after its value was moved`, `'v' borrows 'r.value'`). A value no variable holds - a temporary, a call's argument read from a field - is described by what it borrows (`a value that borrows 'p.n' is used here after it may be released or overwritten`); no corpus message says `'this value'` any more. |

The tier also repeats the Phase 2 gate for its own scope: no open crash or miscompile, an
assertions-enabled CI leg, and a soak run. The first holds today; the other two do not exist for own.

## What is left, in order

1. **Back-references beyond WeakRef.** `WeakRef<T>` works (spec 84): the target's header word holds
   the weak count, and a freed target is a tombstone until its last WeakRef goes. A `@weak` field reads
   it as a plain field, checked on every read (spec 93). Left: a module that makes WeakRefs cannot spawn
   threads with closures, and dropping the check where the compiler can prove the parent alive.
2. **The rest of linked structures.** A list, a node built from the old head, a walk that assigns a
   parameter and the recursive tree insert work (spec 83). A call that may throw, or one into another
   module, still cannot take a field this way, and the popped node cannot be used after the assignment that frees it.
3. **The "not yet" messages.** Second references the compiler cannot classify, values merged from
   branches, object literals that hold owned values.
4. **Whole-program facts**, so a class nobody extends does not need `@final` to keep its callers'
   borrows.
5. **The exit criteria**: a rules specification and an outside review of the messages. Variable names
   no longer need `--di`, and a value no variable holds is described by what it borrows.
6. **Soundness evidence at scale**: fuzzing on Linux with a perturbing allocator, an AddressSanitizer
   run of compiled own programs, a second and third outside codebase.

## What it is ready for

| Use | Ready |
| --- | --- |
| New code written for own: trees of owners, values, arrays, strings, `Shared<T>` where shared | Yes, as a preview |
| Numeric and allocation-heavy kernels where a collector or counting costs too much | Yes, as a preview |
| Porting an existing TypeScript program unchanged | No |
| Programs built on graphs with back-references | As a preview, with `WeakRef<T>`; not with threads in the same module |
| A release that promises memory safety to outside users | No |

## How this was measured

- Compiler: `main` at `363b13ce`, Release, Windows 11 x64. Default library built the same day.
- Corpus sweep: each file of `tslang/test/tester/tests` with `--emit=llvm --no-default-lib --opt
  --opt_level=3`, under `-mm=gc` and `-mm=own`, compile only.
- Fuzzer: `python mm_fuzz.py --jobs 4 run --count 400 --seed 915000`.
- Probes and churn loops: small files, JIT for accept or reject, ahead-of-time builds for output and
  peak working set against gc.
- Not run: the full ctest suite, the default library's tests, anything on Linux, wasm or Android.
  Those lines of the scorecard come from the CI configuration and the repository's records.
