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

Under Node, `stdio_write` and `stdio_write_bytes` write the program's bytes as they are, so binary
output (a Native Messaging host's length prefix, say) comes out unchanged, and `stdio_read` reads
stdin's bytes. A host made with a `write` of its own and no `writeBytes` (the browser page) gets
the bytes as UTF-8 text.

wasm64 needs Node 24 or later. LLVM gives a wasm64 module 64-bit tables, which older V8 cannot
instantiate.

## Exceptions

`try`/`catch` and `finally` are wasm exception handling (the `exception-handling` feature, in every
current browser and Node). The tslang compiler's `WasmExceptionPass` turns each landing pad into a
catch of the C++ exception tag, `__cpp_exception`, which the module imports. The host creates the
tag, so instantiate through `host.instantiate(bytes)`, which reads the tag's parameter type from the
module: a pointer, i32 or i64. The host also keeps what a C++ runtime would: `__cxa_throw` throws a
`WebAssembly.Exception` of that tag, and it tracks each exception's type and the exceptions being
handled. A JS exception (`exit()`, a failed assert) is not of that tag, so no `catch` sees it.

## Async

`async`/`await` and `Promise` lower to MLIR's async runtime (`mlirAsyncRuntime*`). The host
implements it on each thread: an `async` call queues its task, and a blocking
`await` runs queued tasks until what it waits for is ready. `run()` runs the tasks still queued
when `main` returns. `sleep(ms)` blocks: `Atomics.wait` in Node, a busy wait on a browser's main
thread.

## Threads

`Thread.spawn` and `Worker` run under Node. A program that starts a thread (one that calls
`tslang_thread_spawn`) is linked with `--shared-memory --import-memory`: the host makes the shared
memory, and starts each thread in a Node worker ([thread.mjs](thread.mjs)) that instantiates the same
compiled module on it, with a stack and thread-local storage of its own. `join` waits on the
thread's control block with `Atomics.wait`; `Atomics.wait` and `Atomics.notify` work across
threads. A program that starts no thread keeps an unshared memory, so it still runs in a browser
that is not cross-origin isolated.

What this needs from the compiler and the default library:

- Every wasm object is compiled with the `atomics` and `bulk-memory` features, which a shared memory
  requires of everything linked into it. Rebuild the default library's wasm archives after updating.
- The heap lives in the module's memory behind a lock, since one thread frees what another allocated.
- On wasm32 a heap block's header is 8 bytes, not 4, so a payload of 64-bit elements is 8-aligned:
  a real 64-bit atomic traps on an address that is not.

Not in a browser: a browser does not let its main thread block in `join`, and shares memory only on
a cross-origin-isolated page. Each thread has its own JS host, so what the host keeps (open regular
expression matches, async tasks, C++ exceptions in flight) stays on the thread that made it.

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

- default-library tests marked `// gc only`, `// linux only`, or (under own) `// not own`
- the tests in [wasm_skips.txt](wasm_skips.txt)

Under `-mm=own`, a test that doesn't compile for wasm is compiled again natively. If that fails too,
the test is listed as NATIVE and not counted as a failure: it is something `-mm=own` cannot prove
yet. Native ctest runs only a few tests under own, but the whole corpus under none and rc, so under
those two models every wasm compile error counts as a failure.

## In CI

Every pull request runs both suites on wasm32 and wasm64 under none, rc and own, on Windows and
Linux: the exe leg of each build job, with its kind of program (release, or `--debug`), against a
wasm default library that job builds from TypeScriptCompilerDefaultLib's main with the compiler it
just built (`.github/workflows/cmake-test-release-*.yml`). A release builds the wasm default library
into each package and tests the package's compiler, `wasm-ld` and library the same way, release and
debug (`test_wasm_*` in `create-release.yml`). CI passes `--require-run`: on Node 24, which wasm64
needs, a test that is only compiled fails.

## In a browser

Put the `.wasm` files in `wasm\`, serve this folder over HTTP (e.g. `npx http-server .`) and open
[index.html](index.html). It runs each program with the same host.
