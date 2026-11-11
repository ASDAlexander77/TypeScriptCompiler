# The ray tracer against Rust: what is left

`tslang/benchmarks/raytrace/rtbench_static.ts` is the ray tracer written without interfaces and
closures; `rust/src/bin/static_dispatch.rs` beside it is the Rust program it mirrors (`enum Thing`,
`enum Surface`, `match`). This note says where the two stand, what was done to get there, and what
is still between them. The benchmark's [README](../tslang/benchmarks/raytrace/README.md) has every
measurement.

## Where it stands

Ubuntu 24.04 x64, a 4-core cloud VM, 2026-11-11, `bench.sh` with 21 rounds, median wall time. A
few percent either way is noise on this machine.

| Build | Time | vs Rust |
|---|---|---|
| tslang gc, none, own | 53 ms | 1.06x |
| tslang rc | 57 ms | 1.14x |
| Rust, enums and `match` | 50 ms | 1x |

Instructions executed (callgrind), under `-mm=none`: tslang 650 million, Rust 604 million. Without
the math library, whose `floor` and `fmod` Rust compiles in and tslang calls in libm, the
program's own code is about 540 million in tslang against 480 million in Rust with its SLP
vectorizer off (which does not change Rust's time).

Not measured: Windows, and any machine but this one.

## What was done

Five changes to the optimizer, all in `tslang/tslang/transform.cpp` except the first
(`include/TypeScript/LowerToLLVM/ArrayLayout.h`). Together they took the benchmark from 60 to
53 ms.

| Change | Why |
|---|---|
| An array's length and data are loaded through a select between the header and a constant empty header | The branch on a null array kept both loads inside the loop |
| A call of a C math function is the LLVM intrinsic | `sqrt` could write `errno`: a call, a branch to libm, and no hoisting over it |
| `--emit=exe` with `--opt` makes functions that are not exported internal | The inliner inlines a function with one call left whatever its size |
| The inliner's threshold at `--opt_level=3` is 325 | `intersections` cost 285 against 250 and stayed a call in every ray's loop |
| SROA and jump threading run before the first SimplifyCFG (levels 2 and 3) | A miss's `dist === 0 ? Infinity : dist` became a select that jump threading does not unfold |

Open questions about these, none of them checked:

- The threshold of 325 applies to every program at `--opt_level=3`. Its effect on code size and
  compile time of a large program was not measured. With 250 the benchmark takes about 57 ms.
- The early SROA and jump threading run on every function at levels 2 and 3. Compile time was not
  measured.
- Internal functions: the rule is "not `export`ed, not the entry point, no section, in an
  executable". The test suite passes; a program whose other objects call a function of the main
  module by name without `export` would no longer link.

## What is left

Counting executed instructions per block against the Rust build (how: below) puts nearly all of
the difference in the loop over the things, which runs 9 million times a frame: 13 instructions a
thing in the shadow loop and 15 in the primary loop, against 8 and 7 in Rust. The rest is a few
instructions per shadow ray (54 against 44) and per shaded point (127 against 113). None of the
parts below was measured by itself, and the estimates of what each is worth are instruction
counts, not times.

### 1. The caller's minimum still runs on a miss

`intersections` does `if (dist < closest) { index = i; closest = dist; }`. SimplifyCFG turns it
into selects (`minsd`, and `ucomisd` + `cmova` for the index) before `Thing.intersect` is inlined
into the loop. After inlining, a miss arrives as the constant `Infinity`, for which the test is
false, but jump threading does not thread a constant through a select whose other operand is not
a constant. So a miss still runs the minimum, reloads the constant `Infinity` and copies a
register: 3 to 6 instructions a thing, about 30 million a frame, and `closest` is carried from one
iteration to the next through `minsd`.

What it would take, one of:

- keep the caller's `if` a branch until its callee is inlined (SimplifyCFG's folding of a
  two-entry phi, off in the pipeline's first run), then let jump threading and the later
  SimplifyCFG decide. `--two-entry-phi-node-folding-threshold=0` on the command line did not
  change the code, so the select is made somewhere else; which pass makes it was not found;
- a pass of our own, after inlining, that unfolds a select over a phi with a floating-point
  constant among its values.

Writing a miss as `number | undefined` in the benchmark (Rust's `Option<f64>`) does not avoid
this: the flag becomes a phi of `true`/`false` feeding the same selects.

### 2. Things are pointers in the array

`Thing[]` holds pointers to class instances, so each thing costs one load more than Rust's
`Vec<Thing>`, and the hit thing is loaded again for shading. Rust with `Vec<Box<Thing>>` is 3 ms
slower than with `Vec<Thing>`, which is the size of what is left.

`@value class Thing` puts the things in the array itself and compiles today, but measured no
faster: each element is then loaded as one aggregate, all six fields, before the shape is tested.
What it would take: an element of an array of `@value` records read field by field where only
fields are used (`things[i].intersect(ray)` with `intersect` inlined), instead of copied whole.

### 3. `const light = lights[i]` copies the light

The same copy, seen in the shadow loop: all six numbers of the light are loaded at the top of the
iteration and the three of its color are stored on the stack, where Rust reads the color only for
a point the light reaches. About 9 instructions a shadow ray, 14 million a frame. The fix is the
one of part 2: a `const` bound to an element of a `@value` array that nothing writes can be read
in place.

### 4. Two of the ray's numbers live on the stack in the shadow loop

x86-64 has 16 vector registers and none of them survives a call. The shadow loop keeps the point,
the direction, the normal, the reflected direction and the color so far alive, and two numbers end
up on the stack (`subsd 0x10(%rsp), %xmm7` in the sphere test). Rust's SLP vectorizer packs two of
a vector's three components into one register, which leaves it room. In tslang SLP does not fire
on this code at all (no packed instruction in the binary). Why was not looked into; a guess is
that a `@value` argument arrives as separate scalars (`{ double, double, double }` by value) where
Rust's arrives as a pointer whose two first fields load as one vector.

### 5. `-mm=rc` is 4 ms behind the other models

Not looked into. The program allocates nothing per ray, so it is counting: probably the things
read out of the array and the scene's arrays.

### Not a cost: the 32-bit loop counter

`let i = 0` is an `s32`, sign-extended beside a 64-bit induction variable (2 instructions a
thing). With `let i: index = 0` the program executes more instructions (661 million against 650
million) and takes the same time.

## How to measure

```bash
TSLANG=<path to tslang> RUNS=21 bash tslang/benchmarks/raytrace/bench.sh
```

Wall time on a shared machine moves by a millisecond between runs, which is a third of what is
left, so compare instruction counts first:

```bash
valgrind --tool=callgrind --dump-instr=yes --compress-pos=no --callgrind-out-file=cg.out ./rtbs_none
```

Everything is inlined into `traceRay` (`trace_ray` in Rust), so the function totals say nothing.
Join the per-address counts of `cg.out` with `objdump -d` and group the instructions by how often
they ran: blocks of the two builds that ran the same number of times are the same block of the
source (6,353,088 is a thing in the shadow loop, 4,764,816 a sphere there, 1,588,272 a shadow ray,
397,068 a shaded point), and the number of instructions in each is what to compare.

The inliner's decisions:

```bash
tslang --emit=llvm --opt --opt_level=3 --no-default-lib --mm=none rtbench_static.ts -o s.ll
opt -passes='cgscc(inline)' -pass-remarks-missed=inline -disable-output s.ll
```

LLVM's hidden options are accepted on tslang's command line (`--help-hidden`), which is the quick
way to try a threshold.

One comparison to keep in mind: the Rust port written as `rtbench_static.ts` is, a miss as
`f64::INFINITY` and the thing by its index, takes 73 ms. Part of what is left is the form of the
source, which costs Rust more than it costs tslang.
