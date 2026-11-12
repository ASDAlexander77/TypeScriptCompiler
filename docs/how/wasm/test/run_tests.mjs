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
import { spawn } from "node:child_process";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import process from "node:process";
import { fileURLToPath } from "node:url";

const here = path.dirname(fileURLToPath(import.meta.url));
const repo = path.resolve(here, "../../../..");

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

// wasm_skips.txt: one test a line, `<file> [<triple or model>,...] # <reason>`. Triples and models
// given limit it to those (wasm32,own: wasm32 under own); none given, every leg.
const skips = new Map();
const skipsFile = path.join(here, "wasm_skips.txt");
if (fs.existsSync(skipsFile))
{
    for (const line of fs.readFileSync(skipsFile, "utf8").split(/\r?\n/))
    {
        const m = /^\s*([^#\s]+)(?:\s+([\w,]+))?\s*(?:#\s*(.*))?$/.exec(line);
        if (!m) continue;
        const only = m[2] ? m[2].split(",") : [];
        skips.set(m[1], {
            triples: only.filter((x) => x.startsWith("wasm")),
            models: only.filter((x) => !x.startsWith("wasm")),
            reason: m[3] ?? "listed in wasm_skips.txt",
        });
    }
}

// the single-file tests ctest runs: `tslang_add_test(NAME test-compile-... COMMAND test-runner [options] "<one file>")`
const testerFiles = () =>
{
    const names = new Set();
    const cmake = fs.readFileSync(path.join(repo, "tslang/test/tester/CMakeLists.txt"), "utf8");
    const re = /tslang_add_test\(NAME test-compile-\S+ COMMAND test-runner((?: -[\w=-]+)*) ((?:"[^"]+"\s*)+)\)/g;
    let m;
    while ((m = re.exec(cmake)) !== null)
    {
        const testFiles = m[2].match(/"[^"]+"/g);
        if (testFiles.length !== 1 || / -shared\b/.test(m[1])) continue;
        names.add(testFiles[0].slice(1, -1).replace(/^.*\//, ""));
    }

    // a name built in CMake (`${arc_test}.ts`) is not a file
    return [...names].filter((name) => fs.existsSync(path.join(repo, "tslang/test/tester/tests", name)));
};

const suites = {
    tester: {
        dir: path.join(repo, "tslang/test/tester/tests"),
        files: testerFiles,
        compileArgs: ["--no-default-lib"],
        passed: (run) => run.code === 0 && (run.stdout + run.stderr).includes("done.") ? null
            : run.code !== 0 ? `exit code ${run.code}` : "no 'done.' msg.",
    },
    defaultlib: {
        dir: path.join(options.defaultLibRepo, "tests"),
        files: () => fs.readdirSync(path.join(options.defaultLibRepo, "tests")).filter((f) => f.endsWith(".ts")),
        compileArgs: ["--nowarn"],
        cwd: options.defaultLibRepo,
        marker: (source, leg) =>
        {
            const first = source.split(/\r?\n/, 1)[0];
            if (first.startsWith("// gc only")) return "marked '// gc only'";
            if (first.startsWith("// linux only")) return "marked '// linux only'";
            if (first.startsWith("// not own") && leg.mm === "own") return "marked '// not own'";
            return null;
        },
        passed: (run) => run.code !== 0 ? `exit code ${run.code}`
            : !run.stdout.includes("ALL DONE") ? "no 'ALL DONE' marker"
            : run.stderr.trim() ? "wrote to stderr" : null,
    },
};

const skipReason = (suite, file, source, leg) =>
{
    const skip = skips.get(file);
    if (skip && (!skip.triples.length || skip.triples.includes(leg.triple)) && (!skip.models.length || skip.models.includes(leg.mm)))
    {
        return skip.reason;
    }

    const marked = suite.marker?.(source, leg);
    if (marked) return marked;
    return null;
};

const runProcess = (command, args, { env, timeout, cwd, input } = {}) => new Promise((resolve) =>
{
    const child = spawn(command, args, { env: { ...process.env, ...env }, cwd, stdio: [input === undefined ? "ignore" : "pipe", "pipe", "pipe"] });
    let stdout = "", stderr = "", timedOut = false;
    child.stdout.on("data", (d) => (stdout += d));
    child.stderr.on("data", (d) => (stderr += d));
    if (input !== undefined) { child.stdin.on("error", () => { }); child.stdin.end(input); }
    const timer = timeout ? setTimeout(() => { timedOut = true; child.kill(); }, timeout * 1000) : null;
    child.on("error", (e) => { stderr += String(e); });
    child.on("close", (code) => { if (timer) clearTimeout(timer); resolve({ code, stdout, stderr, timedOut }); });
});

// the environment the compiler looks the default library up in, and nothing of the user's own lib paths
const compileEnv = { DEFAULT_LIB_PATH: defaultLibPath, GC_LIB_PATH: "", LLVM_LIB_PATH: "", TSLANG_LIB_PATH: "" };
const buildArgs = options.debug ? ["--di", "--opt_level=0"] : ["--opt"];

const legs = [];
for (const suiteName of options.suite)
{
    if (!suites[suiteName]) { console.error(`unknown suite ${suiteName}: tester or defaultlib`); process.exit(2); }
    for (const triple of options.triple)
        for (const mm of options.mm)
            legs.push({ suiteName, suite: suites[suiteName], triple, mm, name: `${suiteName}-${triple}-${mm}${options.debug ? "-debug" : ""}` });
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
        const reason = skipReason(leg.suite, file, source, leg);
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
const lastLines = (text, n) => text.trim().split(/\r?\n/).slice(-n).join("\n");

const runOne = async ({ leg, file, outDir }) =>
{
    const source = path.join(leg.suite.dir, file);
    const wasm = path.join(outDir, file.replace(/\.ts$/, ".wasm"));
    fs.rmSync(wasm, { force: true });
    const compiled = await runProcess(options.tslang, ["--emit=exe", `-mm=${leg.mm}`, `-mtriple=${leg.triple}-unknown-unknown`,
        ...buildArgs, ...leg.suite.compileArgs, `-o=${wasm}`, source], { env: compileEnv, timeout: 300 });
    if (compiled.code !== 0 || !fs.existsSync(wasm))
    {
        let detail = (compiled.stderr + compiled.stdout).trim().split(/\r?\n/).slice(0, 6).join("\n");
        if (leg.mm !== "own") return { leg: leg.name, file, status: "COMPILE", detail };

        const obj = path.join(outDir, file.replace(/\.ts$/, ".native.o"));
        const native = await runProcess(options.tslang, ["--emit=obj", `-mm=${leg.mm}`, ...buildArgs, ...leg.suite.compileArgs,
            `-o=${obj}`, source], { env: compileEnv, timeout: 300 });
        fs.rmSync(obj, { force: true });
        if (native.code === 0) return { leg: leg.name, file, status: "COMPILE", detail };
        return { leg: leg.name, file, status: "NATIVE", detail: "native: " + (native.stderr + native.stdout).trim().split(/\r?\n/).slice(0, 4).join("\n") };
    }

    if (!canRun(leg))
    {
        return { leg: leg.name, file, status: "BUILT", detail: `compiled; not run: ${leg.triple} needs Node 24 or later (this is ${process.version})` };
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

let next = 0, finished = 0;
const total = jobs.length;
const worker = async () =>
{
    while (next < jobs.length)
    {
        const job = jobs[next++];
        const result = await runOne(job);
        results.push(result);
        finished++;
        if (["FAIL", "COMPILE", "TIMEOUT"].includes(result.status)) console.log(`[${finished}/${total}] ${result.status} ${result.leg} ${result.file}`);
        else if (finished % 100 === 0) console.log(`[${finished}/${total}] ...`);
    }
};

console.log(`tslang: ${options.tslang}\ndefault lib: ${defaultLibPath}\nlegs: ${legs.map((l) => l.name).join(", ")}; ${total} runs, ${results.length} skips`);
await Promise.all(Array.from({ length: Math.max(1, options.jobs) }, worker));

results.sort((a, b) => a.leg.localeCompare(b.leg) || a.file.localeCompare(b.file));
const lines = [];
for (const leg of legs)
{
    const mine = results.filter((r) => r.leg === leg.name);
    const count = (s) => mine.filter((r) => r.status === s).length;
    lines.push(`== ${leg.name}: ${count("PASS")} passed, ${count("FAIL") + count("COMPILE") + count("TIMEOUT")} failed (${count("COMPILE")} to compile, ${count("TIMEOUT")} timed out), ${count("SKIP")} skipped`
        + (count("NATIVE") ? `, ${count("NATIVE")} do not compile natively under own either` : "")
        + (count("BUILT") ? `, ${count("BUILT")} compiled but not run` : ""));
}

const report = [...lines, ""];
for (const r of results.filter((r) => r.status !== "PASS"))
{
    report.push(`${r.status} ${r.leg} ${r.file}${r.detail ? "\n    " + r.detail.replace(/\n/g, "\n    ") : ""}`);
}

fs.mkdirSync(options.out, { recursive: true });
fs.writeFileSync(path.join(options.out, "results.txt"), report.join("\n") + "\n");
console.log(lines.join("\n"));
console.log(`details: ${path.join(options.out, "results.txt")}`);
process.exitCode = results.some((r) => ["FAIL", "COMPILE", "TIMEOUT"].includes(r.status)) ? 1 : 0;
