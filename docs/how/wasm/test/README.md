# Running tslang programs and tests as WebAssembly

`wasm32-unknown-unknown` and `wasm64-unknown-unknown` have no C library. tslang links a wasm program
with `--allow-undefined`, so whatever the C library and the default library's C++ wrappers provide on
other targets (`malloc`, `snprintf`, `puts`, `regexp_*`, `stdio_write`, the time functions, ...) is an
import from the `env` module. [host.mjs](host.mjs) implements them in JavaScript, for Node and for
the browser.

## The default library

Built by the default library's `scripts\build_wasm.bat` (Windows) or `scripts/build_wasm.sh` (Linux)
for `-mm=none`, `-mm=rc` and `-mm=own`, release and debug:

    __build\defaultlib\lib\<wasm32|wasm64>\unknown\unknown\unknown\<release|debug>\<model>\libTypeScriptDefaultLib.a

It is `lib.ts` plus the OS layer `lib.wasm.ts`, with no C++ wrappers. `-mm=gc` (tslang's default) is
an error for a wasm triple, because the collector has no wasm build. Pass `-mm` explicitly.

## One program

    tslang --emit=exe -mm=rc --opt -mtriple=wasm32-unknown-unknown --default-lib-path=<...>\__build -o=hello.wasm hello.ts
    node run.mjs hello.wasm

The exit code is the program's own (`main`'s result, or `exit()`'s argument), or 1 for a trap or an
uncaught exception.

wasm64 needs Node 24 or later. LLVM gives a wasm64 module 64-bit tables, which older V8 cannot
instantiate.

## The tests

    node run_tests.mjs [--suite=tester,defaultlib] [--triple=wasm32,wasm64] [--mm=none,rc,own] [--debug] [-j<n>] [filter]

For each triple and model, this compiles the tests and runs each program under Node. The summary
and every failure are written to `out/results.txt`.

- `tester`: every single-file test the tester's CMakeLists.txt runs, built as test-runner builds them
  (`--no-default-lib`). A test passes when it exits with 0 and prints `done.`.
- `defaultlib`: the default library's own tests (`TypeScriptCompilerDefaultLib/tests`), linked with
  the wasm default library and run in that repository, with `tests/<name>.stdin` as standard input.
  A test passes when it prints `ALL DONE`, writes nothing to stderr and exits with 0, as `tests.ps1`
  checks.

Some tests are skipped, each with its reason:

- tests that catch an exception (`try`/`catch` does not catch on wasm yet; a throw ends the program)
- async tests (the async runtime has no wasm build)
- tests that use threads
- default-library tests marked `// gc only`, `// linux only`, or (under own) `// not own`
- the tests in [wasm_skips.txt](wasm_skips.txt)

Under `-mm=own`, a test that doesn't compile for wasm is compiled again natively. If that fails too,
the test is listed as NATIVE and not counted as a failure: it is something `-mm=own` cannot prove
yet. Native ctest runs only a few tests under own, but the whole corpus under none and rc, so under
those two models every wasm compile error counts as a failure.

## In a browser

Put the `.wasm` files in `wasm\`, serve this folder over HTTP (e.g. `npx http-server .`) and open
[index.html](index.html). It runs each program with the same host.
