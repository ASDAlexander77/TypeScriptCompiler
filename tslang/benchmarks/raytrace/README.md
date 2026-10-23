# Ray tracer: memory models vs Rust and C++

`rtbench.ts` is `test/tester/tests/raytrace.ts` at 512x512 with a pixel checksum (see its header).
`rtbench_own.ts` is the same program for `-mm=own`, with two source edits: the surfaces, which every
thing shares, are `Shared<Surface>` handles, and `Plane.normal` returns a fresh vector instead of
its own field.
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
  because the file declares its own `sqrt`/`floor`/`pow`/`fabs`); `-mm=own` builds `rtbench_own.ts`
- Rust: `opt-level=3`, `lto="fat"`, `codegen-units=1`, `panic="abort"`
- C++: `clang++ -std=c++20 -O3`, and `cl /std:c++20 /O2 /EHsc /MT`

## Results 2026-10-21 (own rows 2026-10-23)

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
| tslang own (2026-10-23) | 628 ms | 4.5 MB | 12x |
| tslang own, `--no-stack-promotion` (2026-10-23) | 627 ms | | |
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
- The own rows were measured on 2026-10-23 (branch own-raytrace, the same machine, two runs of
  `bench.ps1`, median of 7 each; both gave 627.5 ms), the rest of the table on 2026-10-21. In the
  same session rc took 638-640 ms and peaked at 4.5 MB as well, so own runs as fast as rc and frees
  as promptly, with no counts: only the two `Shared<Surface>` handles count. Peak working set is the
  process's `PeakWorkingSet64`, polled while it runs.
- `-mm=own` needs `rtbench_own.ts`'s two edits: the surfaces are shared by every thing, and
  `Plane.normal` returned a borrow of its own field where `Sphere.normal` returns a fresh vector, so
  a caller through the `Thing` interface could not know whether it owns the result.
