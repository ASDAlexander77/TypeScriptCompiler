# Ray tracer: memory models vs Rust and C++

`rtbench.ts` is `test/tester/tests/raytrace.ts` at 512x512 with a pixel checksum (see its header).
`rtbench_own.ts` is the same program for `-mm=own`, with one source edit: the surfaces, which every
thing shares, are `Shared<Surface>` handles. In both files `Vector` and `Color` are `@value` classes
and `Ray` and `Intersection` `@value` interfaces (see "Results 2026-10-24: `@value`").
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

## Results 2026-10-31: parameters captured by copy

Release, Windows 11 x64, the same machine; main 1cd17c33 against the same tree with this change.
Median of 7 runs after a warm-up; every build prints the same checksum. Neither benchmark file
changed.

A closure used to move every parameter it captured into a heap cell, shared by the frame and the
closure's box, so `getNaturalColor` made six cells on every call for `addLight`. Now a parameter that
nothing assigns (neither the body nor a closure in it) is captured by copy, as `this` and a `const`
already were. Under `-mm=own`, a closure that does not escape borrows the copy, as it borrowed the
cell. Under own, a generator's parameters, callbacks and `Shared<T>` handles stay in cells.

| Build | main | parameters by copy | vs Rust idiomatic (55 ms) |
|---|---|---|---|
| tslang gc | 273 ms | 97 ms | 1.8x |
| tslang rc | 171 ms | 134 ms | 2.4x |
| tslang none | 158 ms | 97 ms | 1.8x |
| tslang own | 143 ms | 104 ms | 1.9x |

Under none, the optimized `RayTracer.getNaturalColor` allocates nothing on the heap any more: the
closure's box is on its stack. Variant A below (`addLight` rewritten as a loop) measured 89 ms
under none and own. Both now stay 8-15 ms above that, which this run does not explain.

## Results 2026-10-24: `@value`

Release, Windows 11 x64, the same machine and toolchains as below; branch `value-types` (98ec37ff,
stacked on the "allocator as malloc" change). Two runs of `bench.ps1`, median of 7 runs after a
warm-up each; the table gives the mean of the two medians. For the tslang builds in the table the
two medians were within 1.5 ms of each other; for the C++ `shared_ptr` rows within 6 ms, and for
the `--no-stack-promotion` builds within 7 ms (gc: 256.2 and 249.4). Every build prints the same
checksum.

`rtbench.ts` and `rtbench_own.ts` mark `Vector` and `Color` as `@value` classes (constructor
parameters `public readonly`) and `Ray` and `Intersection` as `@value` interfaces (fields
`readonly`). Their values are structs passed and returned by value, not heap objects. A `@value`
type has no null of its own, so `intersect` returns `Intersection | null` and `closestInter` is
`Intersection | undefined`. No object literal changed. `rtbench_own.ts`'s `Plane.normal` returns
`this.norm` again, since returning a value copies it.

| Build | allocator as malloc | `@value` | vs Rust idiomatic |
|---|---|---|---|
| tslang gc | 1199 ms | 255 ms | 4.7x |
| tslang rc | 630 ms | 172 ms | 3.1x |
| tslang none | 434 ms | 165 ms | 3.0x |
| tslang own | 444 ms | 148 ms | 2.7x |
| Rust idiomatic | 54 ms | 55 ms | 1x |
| Rust Rc-boxed | 490 ms | 492 ms | 9.0x |
| C++ idiomatic (clang) | 50 ms | 50 ms | 0.9x |
| C++ idiomatic (MSVC) | 68 ms | 68 ms | 1.2x |
| C++ shared_ptr (clang) | 829 ms | 903 ms | 16.5x |
| C++ shared_ptr (MSVC) | 896 ms | 977 ms | 18x |

The "allocator as malloc" column is the earlier run below; its Rust and C++ values are that run's
(its "main" column, since those ports do not use tslang's allocator). The Rust and C++ ports did
not change; their `@value` column is this run, for comparison. "vs Rust idiomatic" divides the
`@value` column by this run's Rust idiomatic (54.6 ms). The goal was 1.1x of Rust idiomatic, about 60 ms. own is 2.7x, about 90 ms short of it.

`--no-stack-promotion` changes gc, rc and own by 2% or less; none takes 180 ms without it (9%
slower). This run does not explain the difference for none.

Stack promotion: `--emit=mlir-affine --stack-promotion-report` under rc keeps 17 allocations on
the heap and none of them is a `Vector`, `Color`, `Ray` or `Intersection`. They are the four
closures that capture (`addLight`, `getPoint`, `recenterX`, `recenterY`: "counted-capture") and
the scene built once in `defaultScene` and `exec` (the things, the lights, the camera, the scene
literal, the `RayTracer`). In the optimized LLVM IR, `render` allocates nothing per pixel under
none (LLVM inlines `getPoint` and its two closures and deletes their cells; rc allocates two cells
once per `render`). The one allocation left on the hot path is `addLight`'s: `getNaturalColor`
puts each of the six variables the closure captures in a heap cell (rc adds a seventh block), on
every call.

Where the time goes. A 512x512 render calls `intersections` 2.24 million times (9 million
`Thing.intersect` calls through the interface), `shade` 397 thousand times and `testRay` 1.59
million times. Each row below edits a scratch copy of the benchmark (`rtbench.ts` for none and
rc, `rtbench_own.ts` for own) to remove one cost and keeps the checksum; none of these edits is in
the benchmark. The edits:

- A: `getNaturalColor` does not build `addLight` and call `scene.lights.reduce(addLight,
  Color.defaultColor)`; it starts from `let col = Color.defaultColor`, runs `addLight`'s body in a
  `for (const light of scene.lights)` loop (a light in shadow is skipped, otherwise
  `col = Color.plus(col, ...)`), and returns `col`.
- B: `intersect` returns `Intersection` in `Thing`, `Sphere` and `Plane`; a miss returns
  `{ thing: this, ray: ray, dist: 0 }` instead of `null`. `closestInter` is an `Intersection` that
  starts as `{ thing: scene.things[0], ray: ray, dist: 0 }`, and the tests `inter != null`,
  `isect != null` (`testRay`) and `isect === undefined` (`traceRay`) test `dist` against 0.
- F: `@final` before `class RayTracer`, so calls to its methods are direct.

Median of 7 runs, ms:

| Variant | none | own | rc |
|---|---|---|---|
| the benchmark | 165 | 148 | 171 |
| A: `addLight` written as a loop over the lights, no closure | 89 | 89 | 108 |
| B: `intersect` returns `dist: 0` for a miss, no `\| null` | 148 | 131 | 177 |
| F: `@final class RayTracer` | 147 | 129 | 161 |
| A and B | 73 | 72 | 117 |
| A, B and F | 58 | 58 | 107 |

- A: the `addLight` closure costs 59-76 ms (A alone: 76 ms under none, 59 under own; the same
  after B). The IR shows where: 2.4 million cell allocations per render (six per `shade`), which
  none never frees and own and rc free again. Closures are a non-goal of `@value`.
- B: `Intersection | null` costs 16-17 ms under none and own (B alone, and B after A). The IR shows
  two likely parts, not measured separately: the result is an optional holding a union, 88 bytes,
  returned from every `Thing.intersect` call; and every test for null compares the union's type
  tag with the string `"null"` (`strcmp`), once per `intersect` result, per `traceRay` and per
  `testRay`. Under rc B is 5 ms slower, probably because the miss now carries a `thing` that is
  retained and released (not measured separately).
- F: the virtual `RayTracer` methods cost 18-19 ms alone and 14-15 ms after A and B (none and
  own). `traceRay` and `testRay` call `intersections` and `shade` through the vtable, so LLVM
  inlines none of them and the 88-byte result goes through memory. `@final` makes the calls
  direct. The `Thing` and `Surface` calls stay indirect, as in Rust.
- With all three removed, none and own run in 58 ms, 1.06x Rust: these three account for the
  whole gap. rc stays at 107 ms. The IR shows that copying an `Intersection` retains and releases
  its `thing` through the type's retain and release functions, which are indirect calls; that
  this is what keeps rc at 107 ms is a reading of the IR, not measured.
- Checks for a missing value: of the three narrowed reads looked at in the optimized IR, one has
  no check left and two keep one. In `RayTracer.intersections` under none, `inter.dist` after
  `inter != null` reads the field with no check. In `RayTracer.traceRay` (none and rc),
  `isect === undefined` rules out only `undefined` (null and undefined are one encoding), so
  passing `isect` to `shade`, which takes a plain `Intersection`, keeps a test for null and its
  throw ("a @value value is null or undefined"). Under rc, the `inter.dist` read in
  `intersections` keeps one too, because the retain and release calls of `closestInter = inter`
  come between the test and the read. Only `traceRay`'s check was measured: writing
  `isect == null` there, which folds it, gives the same times. The rc `intersections` re-check was
  not isolated.

## Results 2026-10-24

Release, Windows 11 x64, the same machine and toolchains as below; median of 7 runs after a
warm-up, `bench.ps1`. Every build prints the same checksum. "main" is 63712f3b; "allocator as
malloc" adds the change that declares the process-heap allocator to LLVM as LLVM declares `malloc`
(ProcessHeapPass): its result is a block no other pointer reaches (`noalias`), and it touches only
the allocator's own state.

| Build | main | allocator as malloc | vs Rust idiomatic |
|---|---|---|---|
| tslang gc | 1189 ms | 1199 ms | 22x |
| tslang rc | 606 ms | 630 ms | 11.6x |
| tslang none | 642 ms | 434 ms | 8.0x |
| tslang own | 597 ms | 444 ms | 8.2x |
| Rust idiomatic | 54 ms | | 1x |
| Rust Rc-boxed | 490 ms | | 9.0x |
| C++ idiomatic (clang) | 50 ms | | 0.9x |
| C++ idiomatic (MSVC) | 68 ms | | 1.2x |
| C++ shared_ptr (clang) | 829 ms | | 15x |
| C++ shared_ptr (MSVC) | 896 ms | | 16x |

`--no-stack-promotion` changes rc and own by 2% or less in either column.

Where the time goes, from the optimized LLVM IR of `RayTracer.shade` and its neighbours:

- LLVM inlines `Vector.plus`, `Color.scale` and the rest, but each inlined copy still allocates.
  Before this change the allocator was declared as writing other memory, so every `new` clobbered
  the fields just stored into the previous object: they were loaded back from the heap instead of
  reused, and the object could not be deleted. With the `malloc` declaration, own and none drop
  about a quarter of their time. gc uses its own allocator (GCPass) and is unchanged; rc is 4%
  slower, which this run does not explain.
- Under own, a release first reads the block's header to skip immortal objects (globals such as
  `Color.black`). On an object the function just made, that read can only see a mortal header, but
  LLVM does not prove it here, and the one load keeps the block alive. Removing it by hand from the
  optimized IR (a simulation, not a compiler change) takes own to about 295 ms.
- What is left escapes in ways no local rewrite removes: vectors passed to calls through the `Thing`
  interface (`isect.thing.normal(pos)`), the surface functions and the virtual `RayTracer` methods;
  the `Ray` and `Intersection` literals, boxed as interfaces; and the `addLight` closure. Those need
  objects that live on the caller's stack across a call, or values that are not heap objects at
  all.

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
