# Ray tracer: memory models vs Rust

`rtbench.ts` is `test/tester/tests/raytrace.ts` at 512x512 with a pixel checksum (see its header).
`rust/` holds two ports of the same algorithm:

- `idiomatic.rs`: `Vector`/`Color` are `Copy` values, things are `Box<dyn Thing>`, surfaces hold
  `fn` pointers. This is what a Rust programmer would write.
- `boxed.rs`: every `Vector`, `Color`, `Ray` and `Intersection` is a fresh `Rc` heap object, things
  are `Rc<dyn Thing>`, surface members are `Rc<dyn Fn>` closures. This is roughly the allocation
  pattern tslang produces, so the gap between the two ports is the cost of allocation alone.

Run (needs a Release tslang and cargo):

```
pwsh tslang/benchmarks/raytrace/bench.ps1
```

tslang builds use `--emit=exe --opt --opt_level=3 --no-default-lib --mm=<model>`
(`--no-default-lib` because the file declares its own `sqrt`/`floor`/`pow`/`fabs`); Rust uses
`opt-level=3`, `lto="fat"`, `codegen-units=1`, `panic="abort"`.

## Results 2026-10-21

main 49b5f6bb, Release, Windows 11 x64, Rust 1.99.0; 7 runs after a warm-up.

| Build | median | peak working set |
|---|---|---|
| tslang gc | 1276 ms | 6 MB |
| tslang gc, `--no-stack-promotion` | 1335 ms | |
| tslang rc | 624 ms | 4 MB |
| tslang rc, `--no-stack-promotion` | 617 ms | |
| tslang none | 690 ms | 845 MB |
| tslang none, `--no-stack-promotion` | 716 ms | |
| tslang own | does not compile | |
| Rust idiomatic | 56 ms | 4 MB |
| Rust Rc-boxed | 504 ms | 4 MB |

- Almost all of the gap to Rust is allocation: the Rc-boxed port is 9x slower than the idiomatic
  one, and rc is 1.24x slower than the Rc-boxed port.
- `none` never frees, so it pays for fresh pages and ends up slower than `rc`.
- Stack promotion has no measurable effect here. `--emit=mlir-affine --stack-promotion-report`
  under rc keeps every hot-path allocation on the heap: `Vector.*`/`Color.*` results are
  "returned", the `Ray`/`Intersection` literals are "converted" (to the interface) or
  "recursive", and the `addLight`/`getPoint` closures are "counted-capture".
- `-mm=own` reports about 44 errors: the program shares the `Surfaces` globals, keeps owned values
  in object literals, and aliases `Color.black` from two statics.
