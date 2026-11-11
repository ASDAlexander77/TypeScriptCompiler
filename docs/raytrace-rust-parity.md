# The ray tracer against Rust: where it stands and what is left

`tslang/benchmarks/raytrace/rtbench_static.ts` is the ray tracer written without interfaces and
closures; `rust/src/bin/static_dispatch.rs` beside it is the Rust program it mirrors (`enum Thing`,
`enum Surface`, `match`). This note says where the two stand, what was done to get there, and what
is left. The benchmark's [README](../tslang/benchmarks/raytrace/README.md) has every measurement.

## Where it stands

Ubuntu 24.04 x64, a 4-core cloud VM, 2026-11-11, `bench.sh` with 21 rounds, median wall time. A
few percent either way is noise on this machine.

| Build | Time | vs Rust |
|---|---|---|
| tslang gc | 50.4 ms | 1.00x |
| tslang none, own | 50.0 ms | 0.99x |
| tslang rc | 53.0 ms | 1.05x |
| Rust, enums and `match` | 50.4 ms | 1x |

Instructions executed (callgrind), under `-mm=none`: tslang 606 million, Rust 604 million.

Under gc, none and own the program takes the time of the Rust one. Under rc it does not (part 1
below). Not measured: Windows, and any machine but this one.

## What was done

Six changes to the optimizer, all in `tslang/tslang/transform.cpp` except the first
(`include/TypeScript/LowerToLLVM/ArrayLayout.h`). Together they took the benchmark from 60 to
50 ms.

| Change | Why |
|---|---|
| An array's length and data are loaded through a select between the header and a constant empty header | The branch on a null array kept both loads inside the loop |
| A call of a C math function is the LLVM intrinsic | `sqrt` could write `errno`: a call, a branch to libm, and no hoisting over it |
| `--emit=exe` with `--opt` makes functions that are not exported internal | The inliner inlines a function with one call left whatever its size |
| The inliner's threshold at `--opt_level=3` is 325 | `intersections` cost 285 against 250 and stayed a call in every ray's loop |
| SROA and jump threading run before the first SimplifyCFG (levels 2 and 3) | A miss's `dist === 0 ? Infinity : dist` became a select that jump threading does not unfold |
| A select whose condition compares a phi and is decided for one of the phi's constants is made a branch, then jump threading (levels 2 and 3, after a function's callees are inlined) | The caller's `if (dist < closest)` was selects before `intersect` was inlined, so a miss still ran the minimum |

Open questions about these, none of them checked:

- The threshold of 325 applies to every program at `--opt_level=3`. Its effect on code size and
  compile time of a large program was not measured.
- The early SROA and jump threading, and the select pass with its jump threading, run on every
  function at levels 2 and 3. Compile time was not measured.
- Internal functions: the rule is "not `export`ed, not the entry point, no section, in an
  executable". The test suite passes; a program whose other objects call a function of the main
  module by name without `export` would no longer link.
- The select pass makes a branch of a select that the processor may have predicted worse than it
  computed the select. It fires only where a predecessor's constant decides the condition, and the
  branch that is left for the other predecessors is given back to SimplifyCFG. No program but
  these two benchmarks was timed with it.

## What is left

### 1. `-mm=rc`: the array of things is retained on every ray

`const things = scene.things` in `intersections` takes a reference: the array's count goes up at
the top of the function and down at the end, through a release routine that is not inlined, 2.2
million times a frame. That is the 3 ms rc is behind (711 million instructions against 606
million). Using `scene.things` in the loop instead retains it on every use and is worse (934
million, 65 ms).

The retain is there because a call in the loop could assign `scene.things` and so free the array
under the alias. Here nothing can: once `Thing.intersect` is inlined the loop is arithmetic. What
it would take: dropping a retain and its release when nothing between them can release or store
(a pass over the LLVM IR after inlining, as Objective-C's ARC optimizer does), or the same proof
in the ownership inference, from the callee's facts ("writes no field"), before lowering.

### 2. Differences that are still there and cost nothing measured

With the instruction counts level, these are no longer a gap, but the code still differs from
Rust's in them:

- `Thing[]` holds pointers, one load more a thing than Rust's `Vec<Thing>`. `@value class Thing`
  puts the things in the array itself and measured no faster: each element is loaded whole, all
  six fields, before the shape is tested.
- `const light = lights[i]` loads all six numbers of the light at the top of the iteration, where
  Rust reads the color only for a point the light reaches. Sinking the loads with LLVM's sink pass
  changed nothing.
- LLVM's SLP vectorizer packs two of a vector's three components in the Rust build and does not
  fire in tslang's. Turning it off in Rust does not change Rust's time.
- `let i = 0` is a 32-bit counter beside a 64-bit induction variable. With `let i: index = 0` the
  program executes more instructions and takes the same time.

### 3. The benchmark with interfaces

`rtbench.ts` (interfaces, closures) is unchanged by all this at 84 ms under gc, none and own, 1.7x
the static version. Its costs are the calls through `Thing` and `Surface` and the `addLight`
closure; see the benchmark's README. The Rust port written the same way (`dyn Thing`) takes
112 ms here.

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
`f64::INFINITY` and the thing by its index, takes 73 ms: the form costs Rust more than it costs
tslang.
