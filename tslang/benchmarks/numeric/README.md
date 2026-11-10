# Numeric loops over arrays: memory models vs Rust and C++

`nbench.ts` is six tight loops over flat arrays, the code a vectorizer is for. The ray tracer in
`../raytrace` is scalar code with calls through interfaces, so it cannot show whether loops are
vectorized; this one can. `rust/` and `cpp/` hold ports of the same kernels, written the way each
language's programmer would (`Vec<f64>` with slices and iterators, `std::vector<double>`).

| Kernel | Loop | Notes |
|---|---|---|
| saxpy | `y[i] += a * x[i]` | element-wise, two arrays |
| triad | `z[i] = x[i] * y[i] + c` | element-wise, three arrays |
| stencil | `dst[i] = (src[i-1] + 2 src[i] + src[i+1]) / 4` | neighbours of one array |
| dot | `s += x[i] * y[i]` | a floating-point reduction: strict IEEE order keeps it scalar in every language without fast-math |
| matmul | `c[i][j] += a[i][k] * b[k][j]` | 64x64, flat arrays, i-k-j order |
| hash | `h[i] = (h[i] * 31 + g[i]) & 0xffff`, summed | `int[]`, with an integer reduction |

Arrays hold 4096 elements (32 KB of doubles), so the loops measure the generated code and not the
memory bus. Every kernel ends in a check value, and every build prints the same six. No kernel needs
floating-point operations reordered; the C++ port is built with `-ffp-contract=off` so that clang
does not fuse `a * b + c`, which changes the low digits.

Run (needs a Release tslang, and cargo and clang++ for the ports):

```
TSLANG=<path to tslang> bash tslang/benchmarks/numeric/bench.sh
```

- tslang: `--emit=exe --opt --opt_level=3 --no-default-lib --mm=<model>` (`--no-default-lib`
  because the file declares its own `floor`/`fmod`)
- `JIT_LIB=<libTypeScriptRuntime.so>` adds a row per model under `--emit=jit`, the JIT cache warm
- Rust: `opt-level=3`, `lto="fat"`, `codegen-units=1`, `panic="abort"`
- C++: `clang++ -std=c++20 -O3 -ffp-contract=off`

With a kernel's name as its argument a program runs only that kernel, which is how `bench.sh` times
them one by one. A JIT figure includes the compiler's start-up, about 13 ms here.

## Results 2026-11-10: Windows

Release, Windows 11 x64; main d64978ed, and the same tree with the heap's release and resize
declared not to capture their block (below). `bench.sh` runs under Git Bash; the lib paths
(`TSLANG_LIB_PATH`, `GC_LIB_PATH`, `LLVM_LIB_PATH`) point at the build tree. Median of 15 runs
after a warm-up, ms; every build prints the same check values. The "after" rows are one run, in
which Rust too was about 10% slower than in the "before" run: compare a row with the Rust row of
its own run.

| Build | saxpy | triad | stencil | dot | matmul | hash |
|---|---|---|---|---|---|---|
| tslang gc | 87.3 | 112.3 | 97.3 | 96.7 | 57.3 | 66.8 |
| tslang rc | 130.9 | 150.4 | 146.7 | 94.5 | 84.9 | 157.7 |
| tslang none | 88.1 | 112.7 | 92.2 | 94.2 | 60.9 | 67.0 |
| tslang own | 88.7 | 114.2 | 93.3 | 94.7 | 59.4 | 160.0 |
| Rust | 85.9 | 98.4 | 97.1 | 95.0 | 55.5 | 66.8 |
| C++ (clang) | 133.4 | 145.9 | 143.5 | 93.3 | 75.2 | 157.8 |
| tslang rc, release does not capture | 92.3 | 117.8 | 102.2 | 103.6 | 68.5 | 75.1 |
| tslang own, release does not capture | 99.9 | 121.4 | 108.2 | 103.2 | 68.7 | 75.0 |
| Rust, the same run | 93.8 | 106.6 | 107.1 | 109.4 | 64.0 | 76.1 |

- On Windows, under every memory model but gc, `malloc`, `realloc` and `free` become the process
  heap's `__tslang_heap_*` helpers (ProcessHeapPass). LLVM declares the C `free` and `realloc` not
  to capture the block they are given; the helpers did not say so. A header that a function may
  release on any path then counted as escaped, so a store to the elements might have written it,
  and the loop loaded the elements' pointer again on every iteration. The rc executable had 8
  packed floating-point and integer instructions, against 234-282 for gc, none, own and Rust, and
  own's `hash` loop was not vectorized either. Linux does not run the pass. Declared `captures(none)`, rc and
  own are level with Rust on every kernel.
- `noalias` on the resize's result, which LLVM also says of `realloc`, vectorizes a few more loops
  under none, but crashes five array tests (0xC0000409); it is left out.
- The C++ port is about twice as slow here as on Linux (82 packed instructions): clang with the
  MSVC standard library does not vectorize most of its loops. That is the port's build, not a
  baseline to compare with.

## Results 2026-11-10: Linux

Release, Ubuntu 24.04 x64, a 4-core cloud VM (Xeon 2.1 GHz, AVX2); main d78c735 with the collector's
allocations declared `noalias` (below). Median of 9 runs after a warm-up, ms; every build prints
the same check values. The machine is shared, so a few percent either way is noise.

| Build | saxpy | triad | stencil | dot | matmul | hash |
|---|---|---|---|---|---|---|
| tslang gc | 85.5 | 93.3 | 97.5 | 93.2 | 23.2 | 44.6 |
| tslang rc | 97.9 | 95.0 | 98.8 | 98.1 | 28.7 | 44.7 |
| tslang none | 88.7 | 93.8 | 97.3 | 93.6 | 23.2 | 43.8 |
| tslang own | 87.1 | 94.7 | 97.3 | 96.5 | 26.7 | 48.0 |
| tslang gc (jit) | 101.2 | 93.4 | 118.6 | 106.9 | 32.6 | 57.2 |
| tslang rc (jit) | 91.4 | 86.9 | 96.4 | 108.2 | 33.3 | 42.0 |
| tslang none (jit) | 84.0 | 88.7 | 104.0 | 104.3 | 22.4 | 46.1 |
| tslang own (jit) | 90.9 | 91.0 | 108.0 | 103.3 | 25.6 | 48.5 |
| Rust | 68.2 | 76.0 | 103.4 | 94.9 | 20.6 | 44.4 |
| C++ (clang) | 69.0 | 75.2 | 107.4 | 93.5 | 30.1 | 42.8 |

- The four memory models are level with each other, and level with Rust and C++ on stencil, dot,
  matmul and hash. On saxpy and triad tslang takes about 1.25x as long as the two ports. All of
  them vectorize these loops for SSE2: an executable targets the baseline x86-64, as the Rust and
  C++ builds do.
- dot is level across all ten rows: a strict floating-point reduction runs one addition at a time
  everywhere. `--fast-math` is what would change it.
- The JIT rows generate the same loops for the host's CPU (below) and carry the compiler's
  start-up. The gc and none objects the JIT caches hold the same number of vector instructions;
  what makes the gc row slower than the none row here was not looked into.

### gc before its allocations were `noalias`

An array is a header holding the pointer to its elements. Under rc, none and own the header and
the elements come from `malloc` and `realloc`, whose results LLVM knows by name to be blocks no
other pointer reaches. Under gc they come from `GC_malloc` and `GC_realloc`, which were declared
without that, so a store to an element might have written the header: every loop loaded the
elements' pointer again on each iteration, and none of them was vectorized. GCPass now declares
the results of the collector's allocation functions `noalias`.

| Build | saxpy | triad | stencil | dot | matmul | hash |
|---|---|---|---|---|---|---|
| tslang gc, d78c735 | 190.5 | 225.3 | 206.9 | 94.1 | 91.2 | 220.4 |
| tslang gc, with `noalias` | 85.5 | 93.3 | 97.5 | 93.2 | 23.2 | 44.6 |

Before the change the gc executable had no packed floating-point instruction; after it, its
optimized LLVM IR has as many vector operations as the none build's. The ray tracer in
`../raytrace` is unchanged by it (gc 123 ms before, 128 ms after, within that run's noise).

### The JIT before and after the optimizer was given the host's CPU

The JIT generates code for the host CPU. Before d78c735 the optimizer's cost model was for a
generic x86-64, so it vectorized for SSE2 and the code generator had nothing wider to emit. The
same benchmark under the release before that change (v0.0-pre-alpha122), same machine and run:

| Build (jit) | saxpy | triad | stencil | dot | matmul | hash |
|---|---|---|---|---|---|---|
| none, alpha122 | 111.5 | 106.8 | 106.6 | 106.4 | 29.9 | 58.6 |
| none, d78c735 | 82.4 | 85.3 | 111.7 | 105.6 | 23.0 | 45.2 |
| rc, alpha122 | 103.9 | 104.0 | 107.5 | 105.1 | 38.4 | 52.6 |
| rc, d78c735 | 86.7 | 84.0 | 97.0 | 104.5 | 32.8 | 50.4 |
| own, alpha122 | 98.4 | 105.0 | 112.1 | 102.3 | 31.7 | 54.3 |
| own, d78c735 | 86.2 | 92.8 | 103.0 | 102.9 | 22.4 | 44.2 |

In the object the JIT caches for `-mm=none`, alpha122 has no instruction on a 256-bit register and
d78c735 has 720. saxpy, triad, matmul and hash are 15-25% faster; stencil and dot do not move
(dot is not vectorized, and stencil's figures are within the noise of each other). Executables are
unchanged by it: they target the baseline CPU unless `-mcpu` says otherwise.
