# Ray tracer: memory models vs Rust and C++

`rtbench.ts` is `test/tester/tests/raytrace.ts` at 512x512 with a pixel checksum (see its header).
`rust/` and `cpp/` each hold two ports of the same algorithm:

- `rust/src/bin/idiomatic.rs`, `cpp/idiomatic.cpp`: `Vector`/`Color` are plain values, things are
  `Box<dyn Thing>` / `std::unique_ptr<Thing>`, surfaces hold function pointers. This is what a
  Rust or C++ programmer would write.
- `rust/src/bin/boxed.rs`, `cpp/shared.cpp`: every `Vector`, `Color`, `Ray` and `Intersection` is
  a fresh reference-counted heap object (`Rc` / `std::make_shared`), things are shared too, and
  surface members are `Rc<dyn Fn>` / `std::function`. This is roughly the allocation pattern
  tslang produces, so the gap between a language's two ports is the cost of allocation alone.
  `Rc` counts are plain integers; `shared_ptr` counts are atomic.

Run (needs a Release tslang and cargo; clang++ on PATH and Visual Studio are used when present):

```
pwsh tslang/benchmarks/raytrace/bench.ps1
```

- tslang: `--emit=exe --opt --opt_level=3 --no-default-lib --mm=<model>` (`--no-default-lib`
  because the file declares its own `sqrt`/`floor`/`pow`/`fabs`)
- Rust: `opt-level=3`, `lto="fat"`, `codegen-units=1`, `panic="abort"`
- C++: `clang++ -std=c++20 -O3`, and `cl /std:c++20 /O2 /EHsc /MT`

## Results 2026-10-21

main 49b5f6bb, Release, Windows 11 x64; Rust 1.99.0, clang 21.1.4, MSVC from Visual Studio 18;
median of 7 runs after a warm-up. Every build prints the same checksum.

| Build | median | peak working set | vs C++ idiomatic (clang) |
|---|---|---|---|
| tslang gc | 1298 ms | 6 MB | 26x |
| tslang gc, `--no-stack-promotion` | 1341 ms | | |
| tslang rc | 599 ms | 4 MB | 12x |
| tslang rc, `--no-stack-promotion` | 600 ms | | |
| tslang none | 703 ms | 845 MB | 14x |
| tslang none, `--no-stack-promotion` | 727 ms | | |
| tslang own | does not compile | | |
| Rust idiomatic | 55 ms | 4 MB | 1.1x |
| Rust Rc-boxed | 492 ms | 4 MB | 9.7x |
| C++ idiomatic (clang) | 51 ms | | 1x |
| C++ idiomatic (MSVC) | 69 ms | | 1.4x |
| C++ shared_ptr (clang) | 830 ms | | 16x |
| C++ shared_ptr (MSVC) | 899 ms | | 18x |

- Almost all of the gap to Rust and C++ is allocation: their heap-allocating ports are 9-16x
  slower than their idiomatic ones. rc is 1.2x slower than Rust with `Rc`, and 1.4x faster than
  C++ with `shared_ptr` (atomic counts, `std::function` calls).
- `none` never frees, so it pays for fresh pages and ends up slower than `rc`.
- Stack promotion has no measurable effect here. `--emit=mlir-affine --stack-promotion-report`
  under rc keeps every hot-path allocation on the heap: `Vector.*`/`Color.*` results are
  "returned", the `Ray`/`Intersection` literals are "converted" (to the interface) or
  "recursive", and the `addLight`/`getPoint` closures are "counted-capture".
- `-mm=own` reports about 44 errors: the program shares the `Surfaces` globals, keeps owned values
  in object literals, and aliases `Color.black` from two statics.
