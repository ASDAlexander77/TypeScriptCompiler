# Performance

Benchmarks of tslang under each memory model against Rust and C++ ports of the same code.
See [memory-models.md](memory-models.md) for what the models are.

**Compared with Rust:** under `-mm=own`, tslang matches Rust on numeric array loops and runs
within about 1.35x of idiomatic Rust on an object-heavy ray tracer, about 6.6x faster than the
same program written with Rust's `Rc` for every object. Written without interfaces and closures,
as Rust code with enums is, the ray tracer takes the same time as that Rust code under every
memory model (measured on Linux x64). Put shortly: within 1.1-1.4x of idiomatic
Rust, and on par for tight array loops. These are two benchmarks of our own, measured on Windows
x64; on the ray tracer the remaining gap is in code generation (calls through interfaces, structs
returned through memory), not in memory management, since `-mm=own` runs within a few percent of
`-mm=none`, which frees nothing.

The ray tracer benchmark in [tslang/benchmarks/raytrace](../tslang/benchmarks/raytrace/README.md)
(`raytrace.ts` at 512x512; release build, Windows 11 x64, median of 9 runs, measured 2026-11-10)
against Rust and C++ ports of the same code. The benchmark's copy marks `Vector` and `Color` as
`@value` classes and `Ray` and `Intersection` as `@value` interfaces, so they are structs passed by
value rather than heap objects; `-mm=own` builds `rtbench_own.ts`, the same program with the
shared surfaces as `Shared<T>` handles.

| build | time | vs Rust (values) |
| --- | --- | --- |
| `-mm=gc` | 73 ms | 1.3x slower |
| `-mm=none` | 73 ms | 1.3x slower |
| `-mm=own` | 76 ms | 1.35x slower |
| `-mm=rc` | 96 ms | 1.7x slower |
| Rust, values (`Copy` vectors and colors) | 56 ms | |
| C++, values (clang -O3 / MSVC /O2) | 52 / 70 ms | |
| Rust, every object in an `Rc` | 500 ms | 8.9x slower |
| C++, every object in a `shared_ptr` (clang / MSVC) | 834 / 908 ms | 15x slower |

On 2026-10-21, before `@value`, every vector and color the program computed was a heap object and
the same benchmark took 1298 ms under `-mm=gc` and 599 ms under `-mm=rc`; `@value` took it to 255
and 172 ms. The rest of the gain came from capturing parameters by copy instead of in heap cells,
calling methods nothing overrides directly, and testing a union for null without a tag. `-mm=rc`
still pays for retains and releases in the inner loops. The benchmark's README has every step and
the earlier results. `-mm=rc` went from 110 to 96 ms when retains and releases through an
interface began changing the count inline; what it still pays over the other models is counting
itself, not thread safety (atomics are only used for blocks shared between threads).

The same ray tracer written without interfaces and closures, the way a Rust program with enums
and `match` is (`rtbench_static.ts`: a thing is one `@final` class with a shape tag and a
`switch`, the rest `@value` classes and free functions), against that Rust program
(`rust/src/bin/static_dispatch.rs`). Ubuntu 24.04 x64, a 4-core cloud VM, release build, median of
21 runs, measured 2026-11-11:

| build | time | vs Rust (enums) |
| --- | --- | --- |
| `-mm=gc` | 50 ms | the same |
| `-mm=none` | 50 ms | the same |
| `-mm=own` | 50 ms | the same |
| `-mm=rc` | 51 ms | the same |
| Rust, enums and `match` | 50 ms | |
| Rust, `dyn Thing` (the port above) | 113 ms | |

On the same machine the version with interfaces above (`rtbench.ts`) takes 77 ms under `-mm=gc`,
78 under `-mm=none`, 82 under `-mm=own` and 111 under `-mm=rc`, against 116 ms for the Rust port
with `dyn Thing` and 70 ms for the C++ port: a call through an interface or a callback now first
tries, as direct calls, the few functions of the module it can be. With `@final interface Thing`
(only classes of this module implement it) no call through a pointer is left, and it takes 71 ms
under `-mm=gc` and `-mm=none`, the C++ port's time, and 106 ms under `-mm=rc`.

With nothing called through a vtable or a function value the four memory models run alike, and
tslang takes the time of the Rust program (under `-mm=none` it executes 606 million instructions
against Rust's 604 million). It was 1.2x (60 ms) before seven changes to the optimizer made for
this benchmark: an array's length and data are read without a branch on null, a C math function
is the LLVM intrinsic (TypeScript has no `errno`), `--emit=exe` makes functions that are not
exported internal, the inliner's threshold at `--opt_level=3` is 325, jump threading runs before
the first SimplifyCFG, a select over a phi with a constant is made a branch again once its
function's callees are inlined, and under `-mm=rc` a local alias is not counted where nothing can
release what it names. The steps, their open questions and what is left are in
[docs/raytrace-rust-parity.md](raytrace-rust-parity.md).

Loops over arrays are measured by [tslang/benchmarks/numeric](../tslang/benchmarks/numeric/README.md):
six kernels over flat arrays of 4096 elements, with Rust and C++ ports. On Windows 11 x64 (release
build, median of 15 runs, ms, measured 2026-11-10):

| build | saxpy | triad | stencil | dot | matmul | hash |
| --- | --- | --- | --- | --- | --- | --- |
| `-mm=gc` | 87.8 | 107.9 | 92.5 | 91.6 | 53.8 | 64.6 |
| `-mm=rc` | 86.2 | 114.4 | 93.9 | 94.8 | 56.7 | 65.5 |
| `-mm=none` | 88.1 | 112.6 | 95.1 | 92.6 | 53.4 | 62.7 |
| `-mm=own` | 85.6 | 111.8 | 94.1 | 91.2 | 53.8 | 62.3 |
| Rust (`Vec<f64>`, iterators) | 77.0 | 93.5 | 94.7 | 93.5 | 52.0 | 65.4 |

On Ubuntu 24.04 x64 (a 4-core cloud VM, median of 9 runs, ms, measured 2026-11-10):

| build | saxpy | triad | stencil | dot | matmul | hash |
| --- | --- | --- | --- | --- | --- | --- |
| `-mm=gc` | 85.5 | 93.3 | 97.5 | 93.2 | 23.2 | 44.6 |
| `-mm=rc` | 97.9 | 95.0 | 98.8 | 98.1 | 28.7 | 44.7 |
| `-mm=none` | 88.7 | 93.8 | 97.3 | 93.6 | 23.2 | 43.8 |
| `-mm=own` | 87.1 | 94.7 | 97.3 | 96.5 | 26.7 | 48.0 |
| Rust (`Vec<f64>`, iterators) | 68.2 | 76.0 | 103.4 | 94.9 | 20.6 | 44.4 |
| C++ (`std::vector`, clang -O3) | 69.0 | 75.2 | 107.4 | 93.5 | 30.1 | 42.8 |

Here the memory model does not matter: the loops allocate nothing, and all four models vectorize
them. tslang is level with Rust and C++ on four of the kernels and takes about 1.25x as long on
saxpy and triad. Under `--emit=jit` the loops are vectorized for the host's CPU (AVX2 on that
machine) rather than the baseline x86-64 an executable targets, which made saxpy, triad, matmul
and hash 15-25% faster than the JIT of the release before it. The same ray tracer on that Linux
machine takes 128 ms under `-mm=gc`, 123 under `-mm=none`, 130 under `-mm=own` and 238 under
`-mm=rc`, against 103 ms for the C++ port; the benchmarks' READMEs have the full tables.
