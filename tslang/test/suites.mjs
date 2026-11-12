// What the wasm and Android test runners (wasm/run_tests.mjs, android/run_tests.mjs) share: which
// tests there are, which are skipped, when one passes, and the report.
//
// tester      the tester's tests (tslang/test/tester/tests), as test-runner builds them:
//             --no-default-lib. A test passes when its program exits with 0 and prints "done.".
// defaultlib  the default library's tests (TypeScriptCompilerDefaultLib/tests), linked with the
//             default library, as its tests.ps1 runs them, in that repository, with
//             tests/<name>.stdin as standard input: a test passes when it prints "ALL DONE",
//             writes nothing to stderr and exits with 0.
import { spawn } from "node:child_process";
import fs from "node:fs";
import path from "node:path";
import process from "node:process";
import { fileURLToPath } from "node:url";

export const repo = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "../..");

// the single-file tests ctest runs: `tslang_add_test(NAME test-compile-... COMMAND test-runner [options] "<one file>")`.
// The others in the folder are multi-file tests, tests that expect a compile error, or not run at all.
export const testerFiles = () =>
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

// leg: { mm, linux } - linux: the target is Linux (Android is), so "// linux only" tests run
export const makeSuites = (defaultLibRepo) => ({
    tester: {
        dir: path.join(repo, "tslang/test/tester/tests"),
        files: testerFiles,
        compileArgs: ["--no-default-lib"],
        passed: (run) => run.code === 0 && (run.stdout + run.stderr).includes("done.") ? null
            : run.code !== 0 ? `exit code ${run.code}` : "no 'done.' msg.",
    },
    defaultlib: {
        dir: path.join(defaultLibRepo, "tests"),
        files: () => fs.readdirSync(path.join(defaultLibRepo, "tests")).filter((f) => f.endsWith(".ts")),
        compileArgs: ["--nowarn"],
        cwd: defaultLibRepo,
        marker: (source, leg) =>
        {
            const first = source.split(/\r?\n/, 1)[0];
            if (first.startsWith("// gc only") && leg.mm !== "gc") return "marked '// gc only'";
            if (first.startsWith("// linux only") && !leg.linux) return "marked '// linux only'";
            if (first.startsWith("// not own") && leg.mm === "own") return "marked '// not own'";
            return null;
        },
        passed: (run) => run.code !== 0 ? `exit code ${run.code}`
            : !run.stdout.includes("ALL DONE") ? "no 'ALL DONE' marker"
            : run.stderr.trim() ? "wrote to stderr" : null,
    },
});

// a skips file: one test a line, `<file> [<target or model>,...] # <reason>`. Targets and models
// given limit it to those (wasm32,own: wasm32 under own); none given, every leg. isTarget tells a
// target from a model.
export const loadSkips = (file, isTarget) =>
{
    const skips = new Map();
    if (!fs.existsSync(file)) return skips;
    for (const line of fs.readFileSync(file, "utf8").split(/\r?\n/))
    {
        const m = /^\s*([^#\s]+)(?:\s+([\w,-]+))?\s*(?:#\s*(.*))?$/.exec(line);
        if (!m) continue;
        const only = m[2] ? m[2].split(",") : [];
        skips.set(m[1], {
            targets: only.filter(isTarget),
            models: only.filter((x) => !isTarget(x)),
            reason: m[3] ?? `listed in ${path.basename(file)}`,
        });
    }

    return skips;
};

// leg: { target, mm, ... }
export const skipReason = (skips, suite, file, source, leg) =>
{
    const skip = skips.get(file);
    if (skip && (!skip.targets.length || skip.targets.includes(leg.target)) && (!skip.models.length || skip.models.includes(leg.mm)))
    {
        return skip.reason;
    }

    return suite.marker?.(source, leg) ?? null;
};

export const runProcess = (command, args, { env, timeout, cwd, input } = {}) => new Promise((resolve) =>
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

export const lastLines = (text, n) => text.trim().split(/\r?\n/).slice(-n).join("\n");
export const firstLines = (text, n) => text.trim().split(/\r?\n/).slice(0, n).join("\n");

export const FAILED = ["FAIL", "COMPILE", "TIMEOUT"];

// results: { leg, file, status, detail }; status PASS, FAIL, COMPILE, TIMEOUT, SKIP, NATIVE (-mm=own
// cannot compile it natively either) or BUILT (compiled, not run). Writes <out>/results.txt, prints
// the summary, and returns the exit code: 1 when anything failed.
export const report = (legNames, results, out) =>
{
    results.sort((a, b) => a.leg.localeCompare(b.leg) || a.file.localeCompare(b.file));
    const lines = [];
    for (const leg of legNames)
    {
        const mine = results.filter((r) => r.leg === leg);
        const count = (s) => mine.filter((r) => r.status === s).length;
        lines.push(`== ${leg}: ${count("PASS")} passed, ${count("FAIL") + count("COMPILE") + count("TIMEOUT")} failed (${count("COMPILE")} to compile, ${count("TIMEOUT")} timed out), ${count("SKIP")} skipped`
            + (count("NATIVE") ? `, ${count("NATIVE")} do not compile natively under own either` : "")
            + (count("BUILT") ? `, ${count("BUILT")} compiled but not run` : ""));
    }

    const text = [...lines, ""];
    for (const r of results.filter((r) => r.status !== "PASS"))
    {
        text.push(`${r.status} ${r.leg} ${r.file}${r.detail ? "\n    " + r.detail.replace(/\n/g, "\n    ") : ""}`);
    }

    fs.mkdirSync(out, { recursive: true });
    fs.writeFileSync(path.join(out, "results.txt"), text.join("\n") + "\n");
    console.log(lines.join("\n"));
    console.log(`details: ${path.join(out, "results.txt")}`);
    return results.some((r) => FAILED.includes(r.status)) ? 1 : 0;
};

// runs each job through work(job) on `jobs` workers, printing failures and progress
export const runJobs = async (jobs, jobCount, work, results) =>
{
    let next = 0, finished = 0;
    const total = jobs.length;
    const worker = async () =>
    {
        while (next < jobs.length)
        {
            const result = await work(jobs[next++]);
            results.push(result);
            finished++;
            if (FAILED.includes(result.status)) console.log(`[${finished}/${total}] ${result.status} ${result.leg} ${result.file}`);
            else if (finished % 100 === 0) console.log(`[${finished}/${total}] ...`);
        }
    };
    await Promise.all(Array.from({ length: Math.max(1, jobCount) }, worker));
};
