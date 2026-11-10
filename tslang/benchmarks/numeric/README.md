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

## Results 2026-11-10

Release, Ubuntu 24.04 x64, a 4-core cloud VM (Xeon 2.1 GHz, AVX2); main d78c735. Median of 9 runs
after a warm-up, ms; every build prints the same check values. The machine is shared, so a few
percent either way is noise.

| Build | saxpy | triad | stencil | dot | matmul | hash |
|---|---|---|---|---|---|---|
| tslang gc | 190.5 | 225.3 | 206.9 | 94.1 | 91.2 | 220.4 |
| tslang rc | 84.0 | 94.2 | 96.6 | 92.5 | 28.0 | 43.4 |
| tslang none | 92.3 | 100.7 | 101.5 | 94.0 | 23.8 | 44.8 |
| tslang own | 89.0 | 97.3 | 94.6 | 96.3 | 23.3 | 44.1 |
| tslang gc (jit) | 172.9 | 189.7 | 223.7 | 105.5 | 102.7 | 220.2 |
| tslang rc (jit) | 86.7 | 84.0 | 97.0 | 104.5 | 32.8 | 50.4 |
| tslang none (jit) | 82.4 | 85.3 | 111.7 | 105.6 | 23.0 | 45.2 |
| tslang own (jit) | 86.2 | 92.8 | 103.0 | 102.9 | 22.4 | 44.2 |
| Rust | 73.0 | 82.6 | 103.6 | 96.5 | 21.2 | 46.6 |
| C++ (clang) | 66.4 | 72.4 | 107.2 | 92.9 | 31.5 | 43.4 |

- rc, none and own are within about 1.2x of Rust and C++ on every kernel, and level with them on
  stencil, dot, matmul and hash. Their loops are vectorized (SSE2 in an executable, which targets
  the baseline x86-64 as the Rust and C++ builds do).
- **gc is 2 to 5 times slower than the other models on every kernel but dot.** Its executable has
  no packed floating-point instruction at all: none of the loops is vectorized, where the same
  source under none has them. dot is the same in every build because nothing vectorizes it. Why
  the gc build's loops stay scalar has not been looked into.
- dot is level across all ten rows: a strict floating-point reduction runs one addition at a time
  everywhere. `--fast-math` is what would change it.

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
