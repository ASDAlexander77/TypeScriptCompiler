// Compiles tests for wasm and runs each under Node with the JavaScript host (host.mjs).
//
//   node run_tests.mjs [options] [filter]
//
//   --suite=tester,defaultlib  which tests (default: both)
//                                tester      the tester's tests (tslang/test/tester/tests), as
//                                            test-runner builds them: --no-default-lib. A test passes
//                                            when its program exits with 0 and prints "done.".
//                                defaultlib  the default library's tests (TypeScriptCompilerDefaultLib/tests),
//                                            linked with the wasm default library, as its tests.ps1 runs
//                                            them: a test passes when it prints "ALL DONE", writes
//                                            nothing to stderr and exits with 0.
//   --triple=wasm32,wasm64     targets (default: both)
//   --mm=none,rc,own           memory models (default: all three; gc has no wasm build)
//   --debug                    --di --opt_level=0 programs and the debug default library (default: --opt)
//   --tslang=<path>            tslang (default: the release build tree, else tslang on PATH)
//   --default-lib-repo=<dir>   the default library's repository (default: ../TypeScriptCompilerDefaultLib
//                              next to this one); its __build holds the wasm default library
//   -j<n>                      tests at a time (default: the number of CPUs)
//   --out=<dir>                where the .wasm files and results.txt go (default: ./out)
//   --timeout=<seconds>        per program run (default: 60)
//   --require-run              a test compiled but not run (wasm64 before Node 24) fails (CI)
//   filter                     a regular expression the test's file name has to match
//
// The tester's tests are the single-file ones its CMakeLists.txt runs (`test-compile-*`): the others
// in the folder are multi-file tests, tests that expect a compile error, or not run at all.
//
// Skipped, and listed as such: the default library's tests marked "// gc only", "// linux only" or
// (under own) "// not own", and the tests in wasm_skips.txt, each with its reason.
//
// Under -mm=own, a test that does not compile for wasm is compiled once more for the host
// (--emit=obj): if that fails too, the failure is not wasm's - what -mm=own cannot prove yet - and it
// is listed apart (NATIVE), not counted as a failure. ctest runs only a few tests under own natively;
// it runs the whole corpus under none and rc, so there a wasm compile error is always a failure.
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import process from "node:process";
import { fileURLToPath } from "node:url";
import { repo, makeSuites, loadSkips, skipReason, runProcess, lastLines, firstLines, report, runJobs } from "../suites.mjs";

const here = path.dirname(fileURLToPath(import.meta.url));

const options = {
    suite: ["tester", "defaultlib"], triple: ["wasm32", "wasm64"], mm: ["none", "rc", "own"], debug: false,
    jobs: os.cpus().length, out: path.join(here, "out"), timeout: 60, filter: null,
};
for (const arg of process.argv.slice(2))
{
    let m;
    if ((m = /^--suite=(.+)$/.exec(arg))) options.suite = m[1].split(",");
    else if ((m = /^--triple=(.+)$/.exec(arg))) options.triple = m[1].split(",");
    else if ((m = /^--mm=(.+)$/.exec(arg))) options.mm = m[1].split(",");
    else if (arg === "--debug") options.debug = true;
    else if ((m = /^--tslang=(.+)$/.exec(arg))) options.tslang = m[1];
    else if ((m = /^--default-lib-repo=(.+)$/.exec(arg))) options.defaultLibRepo = path.resolve(m[1]);
    else if ((m = /^-j(\d+)$/.exec(arg))) options.jobs = Number(m[1]);
    else if ((m = /^--out=(.+)$/.exec(arg))) options.out = path.resolve(m[1]);
    else if ((m = /^--timeout=(\d+)$/.exec(arg))) options.timeout = Number(m[1]);
    else if (arg === "--require-run") options.requireRun = true;
    else if (!arg.startsWith("-")) options.filter = new RegExp(arg);
    else { console.error(`unknown option ${arg}`); process.exit(2); }
}

if (options.mm.includes("gc"))
{
    console.error("-mm=gc has no wasm build (the garbage collector is not built for wasm): use none, rc or own");
    process.exit(2);
}

const exe = process.platform === "win32" ? "tslang.exe" : "tslang";
options.tslang ??= [
    path.join(repo, "__build/tslang/windows-msbuild-2026-release/bin", exe),
    path.join(repo, "__build/tslang/ninja/release/bin", exe),
].find((p) => fs.existsSync(p)) ?? exe;
options.defaultLibRepo ??= path.resolve(repo, "../TypeScriptCompilerDefaultLib");
const defaultLibPath = path.join(options.defaultLibRepo, "__build");

const skips = loadSkips(path.join(here, "wasm_skips.txt"), (x) => x.startsWith("wasm"));
const suites = makeSuites(options.defaultLibRepo);

// the environment the compiler looks the default library up in, and nothing of the user's own lib paths
const compileEnv = { DEFAULT_LIB_PATH: defaultLibPath, GC_LIB_PATH: "", LLVM_LIB_PATH: "", TSLANG_LIB_PATH: "" };
const buildArgs = options.debug ? ["--di", "--opt_level=0"] : ["--opt"];

const legs = [];
for (const suiteName of options.suite)
{
    if (!suites[suiteName]) { console.error(`unknown suite ${suiteName}: tester or defaultlib`); process.exit(2); }
    for (const triple of options.triple)
        for (const mm of options.mm)
            legs.push({ suiteName, suite: suites[suiteName], target: triple, triple, mm, linux: false,
                        name: `${suiteName}-${triple}-${mm}${options.debug ? "-debug" : ""}` });
}

const jobs = [];
const results = [];
for (const leg of legs)
{
    const outDir = path.join(options.out, leg.name);
    fs.mkdirSync(outDir, { recursive: true });
    const files = leg.suite.files().filter((f) => !options.filter || options.filter.test(f)).sort();
    for (const file of files)
    {
        const source = fs.readFileSync(path.join(leg.suite.dir, file), "utf8");
        const reason = skipReason(skips, leg.suite, file, source, leg);
        if (reason) { results.push({ leg: leg.name, file, status: "SKIP", detail: reason }); continue; }
        jobs.push({ leg, file, outDir });
    }
}

// wasm64: LLVM gives a wasm64 module 64-bit tables (table64), which V8 runs from Node 24 on; Node 22
// has memory64 behind a flag, but not the tables. Before 24 a wasm64 test is compiled, not run.
const nodeMajor = Number(process.versions.node.split(".")[0]);
const nodeFlags = nodeMajor < 24 ? ["--experimental-wasm-memory64"] : [];
const canRun = (leg) => leg.triple !== "wasm64" || nodeMajor >= 24;
const runner = path.join(here, "run.mjs");

const runOne = async ({ leg, file, outDir }) =>
{
    const source = path.join(leg.suite.dir, file);
    const wasm = path.join(outDir, file.replace(/\.ts$/, ".wasm"));
    fs.rmSync(wasm, { force: true });
    const compiled = await runProcess(options.tslang, ["--emit=exe", `-mm=${leg.mm}`, `-mtriple=${leg.triple}-unknown-unknown`,
        ...buildArgs, ...leg.suite.compileArgs, `-o=${wasm}`, source], { env: compileEnv, timeout: 300 });
    if (compiled.code !== 0 || !fs.existsSync(wasm))
    {
        const detail = firstLines(compiled.stderr + compiled.stdout, 6);
        if (leg.mm !== "own") return { leg: leg.name, file, status: "COMPILE", detail };

        const obj = path.join(outDir, file.replace(/\.ts$/, ".native.o"));
        const native = await runProcess(options.tslang, ["--emit=obj", `-mm=${leg.mm}`, ...buildArgs, ...leg.suite.compileArgs,
            `-o=${obj}`, source], { env: compileEnv, timeout: 300 });
        fs.rmSync(obj, { force: true });
        if (native.code === 0) return { leg: leg.name, file, status: "COMPILE", detail };
        return { leg: leg.name, file, status: "NATIVE", detail: "native: " + firstLines(native.stderr + native.stdout, 4) };
    }

    if (!canRun(leg))
    {
        return { leg: leg.name, file, status: options.requireRun ? "FAIL" : "BUILT",
                 detail: `compiled; not run: ${leg.triple} needs Node 24 or later (this is ${process.version})` };
    }

    const stdinFile = source.replace(/\.ts$/, ".stdin");
    const input = fs.existsSync(stdinFile) ? fs.readFileSync(stdinFile) : undefined;
    const run = await runProcess(process.execPath, ["--no-warnings", ...nodeFlags, runner, wasm],
        { timeout: options.timeout, cwd: leg.suite.cwd, input });
    if (run.timedOut) return { leg: leg.name, file, status: "TIMEOUT", detail: lastLines(run.stdout + run.stderr, 6) };
    const failure = leg.suite.passed(run);
    if (failure) return { leg: leg.name, file, status: "FAIL", detail: `${failure}\n` + lastLines(run.stdout + run.stderr, 8) };
    return { leg: leg.name, file, status: "PASS" };
};

console.log(`tslang: ${options.tslang}\ndefault lib: ${defaultLibPath}\nlegs: ${legs.map((l) => l.name).join(", ")}; ${jobs.length} runs, ${results.length} skips`);
await runJobs(jobs, options.jobs, runOne, results);
process.exitCode = report(legs.map((l) => l.name), results, options.out);
