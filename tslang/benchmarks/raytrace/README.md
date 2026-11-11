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

`rtbench_static.ts` and `rust/src/bin/static_dispatch.rs` are the same ray tracer with static
dispatch only, the TypeScript written the way the Rust is (see "Results 2026-11-11"):

- Rust: `enum Thing` and `enum Surface` with a `match` in each method, no `dyn Thing` and no
  function pointers; free functions, and a `for` loop over the lights.
- tslang: no interfaces and no closures. A thing is one `@final` class with a `shape` tag and a
  `switch`, a surface is an enum, `Vector`, `Color`, `Ray`, `Light` and `Hit` are `@value` classes.
  One file builds under every memory model.

Run (needs a Release tslang and cargo; clang++ on PATH and Visual Studio are used when present):

```
pwsh tslang/benchmarks/raytrace/bench.ps1
TSLANG=<path to tslang> bash tslang/benchmarks/raytrace/bench.sh
```

- tslang: `--emit=exe --opt --opt_level=3 --no-default-lib --mm=<model>` (`--no-default-lib`
  because the file declares its own `sqrt`/`floor`/`pow`/`fabs`); `-mm=own` builds `rtbench_own.ts`
- Rust: `opt-level=3`, `lto="fat"`, `codegen-units=1`, `panic="abort"`
- C++: `clang++ -std=c++20 -O3`, and `cl /std:c++20 /O2 /EHsc /MT`

## Results 2026-11-11: static dispatch (Linux)

Release, Ubuntu 24.04 x64, a 4-core cloud VM (Xeon 2.3 GHz); main 864dab4, built against the
official LLVM 22.1.8 package; Rust 1.97.0, clang 18.1.3. `bench.sh`: a warm-up, then 15 rounds in
which every build runs once; median wall time. Every build prints the same checksum. The machine is
shared, so a few percent either way is noise.

| Build | `rtbench.ts` | `rtbench_static.ts` | vs Rust static |
|---|---|---|---|
| tslang gc | 87 ms | 60 ms | 1.2x |
| tslang rc | 123 ms | 62 ms | 1.2x |
| tslang none | 89 ms | 61 ms | 1.2x |
| tslang own | 91 ms | 60 ms | 1.2x |
| Rust static (`static_dispatch.rs`) | | 51 ms | 1x |
| Rust idiomatic (`dyn Thing`) | 116 ms | | 2.3x |
| C++ idiomatic (clang) | 70 ms | | 1.4x |

- Without interfaces and closures the four memory models run alike: nothing is counted or
  allocated per ray, so rc loses its extra time (123 to 62 ms).
- tslang is within 1.2x of the Rust port written the same way, and 1.4x faster than its own
  interface version.
- Under `-mm=own` a `@value` class cannot hold a borrowed element of an array (`new Hit(thing,
  dist)` is "borrows an element and cannot be stored, returned or captured"), so `Hit` names the
  thing by its index in `scene.things`. Under none, holding the reference instead measured 2 ms
  faster.
- Making `Thing` a `@value` class (the things inline in the array, as in Rust's `Vec<Thing>`) or
  reading `things.length` once before the loop changed nothing.

### The optimizer: array reads, math calls, internal functions, the inliner

callgrind (804 million instructions against Rust's 598 million), the optimized IR and the
inliner's remarks put the gap in four things, each now changed in the compiler:

- An array's length and data were read as `header == null ? 0 : header->field`, a branch around
  the load, so the loop in `intersections` tested the array for null and loaded both again on every
  iteration. They are now loaded through a select between the header and one constant empty
  header, with no branch, and a loop hoists them.
- `sqrt` was libm's, which may write `errno`: a call that kept loads from being hoisted over it and
  a branch to the library for a negative argument. A call of a C math function (`sqrt`, `pow`,
  `floor`, ...) is now the LLVM intrinsic, as under clang's `-fno-math-errno` and in Rust.
- Every function had external linkage. With `--opt`, `--emit=exe` now makes a function that is not
  exported internal, so one with a single call left is inlined whatever its size: `shade`,
  `getNaturalColor` and `getReflectionColor` end up inside `traceRay`, as in the Rust build.
- `intersections` cost the inliner 285 against a threshold of 250 and stayed a call in each ray's
  loop, with every live number spilled around it (its Rust twin costs 245 against rustc's 275).
  The threshold at `--opt_level=3` is now 325, the one LLVM keeps for functions with an inline
  hint.

`bench.sh`, 21 rounds, the same machine:

| Build | before | after | vs Rust static (51 ms) |
|---|---|---|---|
| tslang gc, `rtbench_static.ts` | 60 ms | 55 ms | 1.07x |
| tslang rc, `rtbench_static.ts` | 62 ms | 60 ms | 1.18x |
| tslang none, `rtbench_static.ts` | 61 ms | 55 ms | 1.08x |
| tslang own, `rtbench_static.ts` | 60 ms | 55 ms | 1.08x |
| tslang gc, none, own, `rtbench.ts` | 87, 89, 91 ms | 85, 84, 87 ms | |
| tslang rc, `rtbench.ts` | 123 ms | 126 ms | |

- One at a time, under none (medians of 21 runs): the array reads, the math calls and internal
  functions together took 59 ms to 57; the inliner's threshold on top of them, to 54.
- `../numeric`'s six kernels take the same time with the old and the new compiler (within 1%), and
  the test suite passes.

### Jump threading before the first SimplifyCFG

Counting executed instructions per block (callgrind) against the Rust build put most of what was
left on one path: a sphere the ray misses. `intersect` ends `return dist === 0 ? Infinity : dist`
with `dist` a phi of constants on the miss paths; the pipeline's first SimplifyCFG makes the test a
select, and LLVM's jump threading later unfolds a select only over integer constants. A miss ran 8
instructions of select and then the caller's comparison, where Rust's goes straight to the next
thing. The pipeline now starts with SROA and jump threading (at `--opt_level` 2 and 3), so such a
path is threaded to its result before the select exists.

| Build, `rtbench_static.ts` | before | after | vs Rust static (50 ms) |
|---|---|---|---|
| tslang gc | 55 ms | 53 ms | 1.06x |
| tslang rc | 60 ms | 57 ms | 1.14x |
| tslang none | 55 ms | 53 ms | 1.06x |
| tslang own | 55 ms | 53 ms | 1.06x |

- Instructions executed under none: 679 million to 650 million (Rust 604 million, 647 million with
  its SLP vectorizer off, which does not change its time). `rtbench.ts` and `../numeric` take the
  same time as before, and the test suite passes.
- The 32-bit loop counter named above as a cost is not one: with `let i: index = 0` the program
  executes more instructions (661 million) and takes the same time.
- The Rust port written as this file is, a miss as `f64::INFINITY` and the thing by its index,
  takes 73 ms: the form costs Rust more than it costs tslang.
- [docs/raytrace-rust-parity.md](../../../docs/raytrace-rust-parity.md) lists what is left and what
  each part would take.
- What is left is spread out, a few instructions in every block: the caller's `minsd` still runs
  on a miss; the things are pointers in the array, one load more each; `const light = lights[i]`
  loads all six numbers of the light, where Rust reads the color only when the point is lit; and
  two of the ray's numbers live on the stack in the shadow loop. None was measured by itself.

### Selects over a phi made branches again

Part 1 of [docs/raytrace-rust-parity.md](../../../docs/raytrace-rust-parity.md): `intersections`
does `if (dist < closest) { index = i; closest = dist; }`, which SimplifyCFG makes selects before
`Thing.intersect` is inlined into the loop. A miss then arrives as the constant `Infinity`, for
which the test is false, and still ran the minimum. A pass of our own, after a function's callees
are inlined, turns a select back into a branch when its condition compares a phi of the same block
and is decided for one of the phi's constants; jump threading, run next, sends those predecessors
past it. A miss now goes to the next thing, as in the Rust build.

`bench.sh`, 21 rounds, the same machine:

| Build, `rtbench_static.ts` | before | after | vs Rust static (50.4 ms) |
|---|---|---|---|
| tslang gc | 53 ms | 50.4 ms | 1.00x |
| tslang rc | 57 ms | 53.0 ms | 1.05x |
| tslang none | 53 ms | 50.0 ms | 0.99x |
| tslang own | 53 ms | 50.0 ms | 0.99x |

- Instructions executed under none: 650 million to 606 million; Rust executes 604 million.
- `rtbench.ts` and `../numeric` take the same time as before, and the test suite passes.
- Sinking the loads of `const light = lights[i]` to where the color is used (LLVM's sink pass at
  the end of the pipeline) changed neither the instruction count nor the time, and is not in.
- rc: `const things = scene.things` retains the array and releases it at the end of every
  `intersections` call, 2.2 million times a frame (711 million instructions against 606 million).
  Without the alias it is worse: every `scene.things` in the loop is retained (934 million, 65 ms).

### rc: a local alias that needs no count

`const things = scene.things` retained the array and released it at the end of every
`intersections` call. The count keeps the array alive should something release it under the alias;
here nothing can. Under `-mm=rc`, before the inliner runs, a retain of a local and its release on
every path are now removed when the function, and every function it calls, stores only into its
own locals (or what a parameter given a local points to) and calls nothing that could release:
`tsret_<type>(slot)` and `tsrel_<type>(slot)` on a slot stored to once. A callee that assigns the
field keeps the pair (`test/tester/tests/00rc_local_alias_no_count.ts`).

`bench.sh`, 21 rounds, the same machine:

| Build, `rtbench_static.ts` | before | after | vs Rust static (49.6 ms) |
|---|---|---|---|
| tslang gc | 50.4 ms | 50.5 ms | 1.02x |
| tslang rc | 53.0 ms | 50.6 ms | 1.02x |
| tslang none | 50.0 ms | 49.6 ms | 1.00x |
| tslang own | 50.0 ms | 49.5 ms | 1.00x |

Instructions executed under rc: 711 million to 622 million. `rtbench.ts` under rc is unchanged
(119 ms): its counts are on interfaces and closures, which this does not reach.

## Results 2026-11-10: Windows

Release, Windows 11 x64, the machine of the 10-31 run; main d64978ed, and the same tree with the
heap's release and resize declared not to capture their block (see `../numeric`). `bench.ps1`,
median of 9 runs after a warm-up; every build prints the same checksum.

| Build | main | release does not capture | vs Rust idiomatic (57 ms) |
|---|---|---|---|
| tslang gc | 74.4 ms | 74.1 ms | 1.3x |
| tslang gc, `--no-stack-promotion` | 159.3 ms | 155.2 ms | |
| tslang rc | 117.5 ms | 114.2 ms | 2.0x |
| tslang none | 78.2 ms | 73.2 ms | 1.3x |
| tslang own | 77.5 ms | 76.4 ms | 1.3x |
| Rust idiomatic | 57.2 ms | 57.2 ms | 1x |
| Rust Rc-boxed | 528.7 ms | 502.6 ms | 8.8x |
| C++ idiomatic (clang) | 52.7 ms | 51.6 ms | 0.9x |
| C++ idiomatic (MSVC) | 71.6 ms | 71.2 ms | 1.2x |
| C++ shared_ptr (clang) | 858.4 ms | 844.0 ms | 14.8x |
| C++ shared_ptr (MSVC) | 911.3 ms | 910.2 ms | 16x |

- Against 10-31 (gc 71, rc 110, none 70, own 77) the four models are within a few percent; Rust
  and C++ also measured 4-5% slower than then.
- gc now depends on stack promotion: without it gc takes twice as long, where on 10-24 the flag
  changed it by 2% or less. Under rc and own the flag changes nothing, under none 10 ms. Not
  looked into.
- The heap change is for loops over arrays; here it moves the models by no more than the noise.

### rc: an interface's count changed inline

A sampling profile of the rc build (the program's main thread sampled every 0.2 ms, 20 runs,
symbols from a `--di` build), with each function's share scaled by `pow_fma`'s, which does the same
work under every model, put rc's extra time in `Sphere.intersect` (+17 ms against none), the inlined
`addLight` closure that holds `testRay` and `intersections` (+16 ms), the out-of-line retain and
release routines (+17 ms) and `traceRay` (+7 ms). Most of it was one thing: an `Intersection`
holds an interface, `thing`, and an interface is retained and released through the type descriptor
its vtable names - vtable, tag, descriptor, then an indirect call LLVM can neither inline nor pair
with its counterpart. `Sphere.intersect` retained `thing` that way on every hit, and spilled the
80-byte result to the stack to give the routine its address.

Every type whose value is a reference to a block counted the plain way (class and object literal
instances, strings, `any` boxes, `Shared<T>`) now shares one retain routine, and under rc a retain
through an interface whose descriptor names that routine adds one to the count inline; a release
takes one off inline when it is not the last reference on a block only this thread holds. The last
reference, shared and immortal blocks and every other kind still go through the descriptor.

| Build | before | interface count inline |
|---|---|---|
| tslang rc | 110-114 ms | 95.9 ms |
| tslang gc, none, own | 71-76 ms | 73.1, 73.3, 75.8 ms |

What is left of rc's extra time is counting itself: `scene.things` is retained on every iteration
of `intersections` (the read of an interface's field is owned, since the slot may hold a getter's
cell), and `closestInter = inter` copies a reference that `inter` gives up at the end of the
iteration, where a move would do. Hoisting `scene.things` out of the loop by hand saves about 8 ms.
Cancelling `return x`'s retain against the local's release, tried as a pass, measured nothing: with
both halves inline LLVM already folds them.

## Results 2026-11-10: Linux

Release, Ubuntu 24.04 x64, a 4-core cloud VM (Xeon 2.1 GHz); main d78c735 with the collector's
allocations declared `noalias`. `bench.ps1` is for Windows, so the same builds were made and timed
by hand: median of 15 runs, taken in turn so that the machine's drift falls on every build alike.
Every build prints the same checksum. The machine is shared, so a few percent either way is noise.

| Build | Median | vs C++ idiomatic |
|---|---|---|
| tslang gc | 128 ms | 1.2x |
| tslang rc | 238 ms | 2.3x |
| tslang none | 123 ms | 1.2x |
| tslang own | 130 ms | 1.3x |
| Rust idiomatic | 164 ms | 1.6x |
| Rust Rc-boxed | 398 ms | 3.8x |
| C++ idiomatic (clang) | 103 ms | 1x |
| C++ shared_ptr (clang) | 571 ms | 5.5x |

The ratios are against the C++ port here, because the Rust idiomatic port is 1.6x slower than it
on this machine, where on Windows the two are within 10% of each other. Built with
`-C target-cpu=native` the Rust port takes 109 ms, and the same flag does not change the C++ port;
what the baseline x86-64 Rust build loses the time to was not looked into. The tslang builds
target the baseline CPU too.

Under `--emit=jit` with the JIT cache warm, the times are the executables' plus the compiler's
start-up (about 13 ms): the JIT's code for this benchmark is the executable's, and the same before
and after the optimizer was given the host's CPU (d78c735) - there are no loops here for it to
vectorize. `../numeric` is the benchmark that shows that change.

## Results 2026-10-31: parameters captured by copy, direct method calls, null without a tag

Release, Windows 11 x64, the same machine; main 1cd17c33 against the same tree with each change
added in turn. Median of 7 runs after a warm-up; every build prints the same checksum. Neither
benchmark file changed.

- Parameters by copy: a closure used to move every parameter it captured into a heap cell, shared
  by the frame and the closure's box, so `getNaturalColor` made six cells on every call for
  `addLight`. Now a parameter that nothing assigns (neither the body nor a closure in it) is
  captured by copy, as `this` and a `const` already were. Under `-mm=own`, a closure that does not
  escape borrows the copy, as it borrowed the cell. Under own, a generator's parameters, callbacks
  and `Shared<T>` handles stay in cells. Under none, the optimized `RayTracer.getNaturalColor` no
  longer allocates on the heap: the closure's box is on its stack.
- Direct method calls: a call through a class's vtable to a method that no class in the module
  overrides is made a direct call, as a call on a `@final` class's value already was, so LLVM can
  inline it. A class that code outside the module may extend (exported, declared or imported, a
  C++ binding, or a base of one of these) keeps its methods virtual. That takes in `RayTracer`'s
  methods: variant F below, now without the source edit.
- Null tested by kind: a union's tag points into its type's descriptor, so whether it holds null
  is the descriptor's kind (`TYPE_KIND_NULL`), one load and one compare. It was a `strcmp` of the
  type's name with `"null"`, on every `intersect` result and every `traceRay` and `testRay`.
- Null in the reference: `Intersection` holds an interface, `thing`, and an interface is null when
  the object in it is (`{vtable, this}`, `this` null). So `Intersection | null` needs no tag: it is
  stored as an `Intersection`, null being one whose `thing` holds no object, and the 88-byte
  tagged result of `intersect` is the value itself with the optional's flag. A `@value` type with
  only numbers keeps its tag.

| Build | main | parameters by copy | and direct method calls | and null by kind | and null in the reference | vs Rust idiomatic (55 ms) |
|---|---|---|---|---|---|---|
| tslang gc | 266 ms | 97 ms | 79 ms | 75 ms | 71 ms | 1.3x |
| tslang rc | 171 ms | 133 ms | 120 ms | 115 ms | 110 ms | 2.0x |
| tslang none | 157 ms | 96 ms | 79 ms | 74 ms | 70 ms | 1.3x |
| tslang own | 143 ms | 104 ms | 86 ms | 82 ms | 77 ms | 1.4x |

Of the three costs measured below, A's cells, F's virtual calls and B's union are now gone without
editing the source: variant B (`intersect` returns `dist: 0` for a miss, no union) runs in 69.9 ms
under none, as the benchmark itself now does.

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
