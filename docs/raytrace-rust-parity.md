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
| tslang gc | 50.5 ms | 1.02x |
| tslang rc | 50.6 ms | 1.02x |
| tslang none | 49.6 ms | 1.00x |
| tslang own | 49.5 ms | 1.00x |
| Rust, enums and `match` | 49.6 ms | 1x |

Instructions executed (callgrind): tslang 606 million under `-mm=none` and 622 million under
`-mm=rc`, Rust 604 million.

Under every memory model the program takes the time of the Rust one, within this machine's noise.
Not measured: Windows, and any machine but this one.

## What was done

Seven changes to the optimizer, all in `tslang/tslang/transform.cpp` except the first
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
| `-mm=rc`: a retain of a local and its release on every path are removed when the function and everything it calls store only into locals and can release nothing (before the inliner) | `const things = scene.things` was counted up and down on every ray |

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

- The rc pass trusts its own reading of what a function can do: a store is local when LLVM's
  `getUnderlyingObject` finds an alloca or a parameter, a callee is safe when it is defined in the
  module and passes the same test, or is a declaration that only reads memory. It finds the pair
  by the names `tsret_` and `tsrel_`. The test suite passes and
  `00rc_local_alias_no_count.ts` covers a callee that releases the array; nothing else checks it,
  and an object shared between threads and released by another thread while the alias is in use
  is not considered (the function's caller holds a reference to the owner in the case measured).

## What is left

Nothing in time, on this benchmark and this machine. What follows is what still differs.

### 1. rc counts that the alias pass does not reach

The pass removes a pair only around code that can release nothing. `rtbench.ts` (interfaces,
closures) under rc is unchanged at 119 ms against 84 ms for the other models: its counts are on
interface values and closure cells, and its loop calls through `Thing`, which the pass cannot see
through. The benchmark's README has what those counts are.

### 2. Differences that are still there and cost nothing measured

With the instruction counts level, these are no gap, but the code still differs from Rust's in
them:

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
the static version, and 119 ms under rc. Its costs are the calls through `Thing` and `Surface` and the `addLight`
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
