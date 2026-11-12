// Compiles tests for Android and runs them on a device or emulator through adb.
//
//   node run_tests.mjs [options] [filter]
//
//   --build                    only compile: the programs and a manifest go to --out
//   --run                      only run what --build left in --out (needs adb, not the compiler)
//                              (neither: both)
//   --suite=tester,defaultlib  which tests (default: both) - see ../suites.mjs
//   --mm=gc,rc,none,own        memory models (default: all four)
//   --abi=x86_64               arm64-v8a, armeabi-v7a or x86_64 (default: x86_64, which an x86_64
//                              emulator runs natively)
//   --tslang=<path>            tslang (default: the release build tree, else tslang on PATH)
//   --default-lib-repo=<dir>   the default library's repository, for its tests (default:
//                              ../TypeScriptCompilerDefaultLib next to this one)
//   --default-lib-path=<dir>   the folder holding defaultlib/ (default: <default-lib-repo>/__build)
//   --android-lib=<dir>        the folder holding the Android libgc.a and libTypeScriptAsyncRuntime.a,
//                              as a release package has them: android/<abi>/lib (default: this
//                              repository's 3rdParty/gc/android/<abi>/release/lib and
//                              __build/tslang-runtime/release/android/<abi>)
//   --android-ndk-path=<dir>   the NDK tslang links with (default: ANDROID_NDK_HOME)
//   -j<n>                      compiles at a time (default: the number of CPUs)
//   --device-jobs=<n>          programs at a time on the device (default: 4)
//   --timeout=<seconds>        per program run (default: 120)
//   --out=<dir>                programs, manifest.json and results.txt (default: ./out)
//   --adb=<path>               adb (default: adb on PATH)
//   filter                     a regular expression the test's file name has to match
//
// A test passes as ../suites.mjs says. Skipped, and listed as such: the tests in android_skips.txt
// and the default library's tests marked "// gc only" (but under gc) or (under own) "// not own".
// Under -mm=own, a test that does not compile for Android and does not compile for the host either is
// listed apart (NATIVE), not counted as a failure - what -mm=own cannot prove yet.
//
// On the device each program runs in /data/local/tmp/tslang-tests under `timeout` and a memory
// limit (an out-of-memory adb-shell process can take the emulator down), with its output in files;
// what comes back is its exit code, whether it printed "done." / "ALL DONE", how much it wrote to
// stderr and the last lines of each.
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import process from "node:process";
import { fileURLToPath } from "node:url";
import { repo, makeSuites, loadSkips, skipReason, runProcess, lastLines, firstLines, report, runJobs } from "../suites.mjs";

const here = path.dirname(fileURLToPath(import.meta.url));
const ABIS = {
    "arm64-v8a": "aarch64-linux-android29",
    "armeabi-v7a": "armv7a-linux-androideabi29",
    "x86_64": "x86_64-linux-android29",
};

const options = {
    suite: ["tester", "defaultlib"], mm: ["gc", "rc", "none", "own"], abi: "x86_64", build: false, run: false,
    jobs: os.cpus().length, deviceJobs: 4, timeout: 120, out: path.join(here, "out"), adb: "adb", filter: null,
};
for (const arg of process.argv.slice(2))
{
    let m;
    if (arg === "--build") options.build = true;
    else if (arg === "--run") options.run = true;
    else if ((m = /^--suite=(.+)$/.exec(arg))) options.suite = m[1].split(",");
    else if ((m = /^--mm=(.+)$/.exec(arg))) options.mm = m[1].split(",");
    else if ((m = /^--abi=(.+)$/.exec(arg))) options.abi = m[1];
    else if ((m = /^--tslang=(.+)$/.exec(arg))) options.tslang = m[1];
    else if ((m = /^--default-lib-repo=(.+)$/.exec(arg))) options.defaultLibRepo = path.resolve(m[1]);
    else if ((m = /^--default-lib-path=(.+)$/.exec(arg))) options.defaultLibPath = path.resolve(m[1]);
    else if ((m = /^--android-lib=(.+)$/.exec(arg))) options.androidLib = path.resolve(m[1]);
    else if ((m = /^--android-ndk-path=(.+)$/.exec(arg))) options.ndk = path.resolve(m[1]);
    else if ((m = /^-j(\d+)$/.exec(arg))) options.jobs = Number(m[1]);
    else if ((m = /^--device-jobs=(\d+)$/.exec(arg))) options.deviceJobs = Number(m[1]);
    else if ((m = /^--timeout=(\d+)$/.exec(arg))) options.timeout = Number(m[1]);
    else if ((m = /^--out=(.+)$/.exec(arg))) options.out = path.resolve(m[1]);
    else if ((m = /^--adb=(.+)$/.exec(arg))) options.adb = m[1];
    else if (!arg.startsWith("-")) options.filter = new RegExp(arg);
    else { console.error(`unknown option ${arg}`); process.exit(2); }
}

if (!options.build && !options.run) options.build = options.run = true;
const triple = ABIS[options.abi];
if (!triple) { console.error(`unknown ABI ${options.abi}: ${Object.keys(ABIS).join(", ")}`); process.exit(2); }

const manifestFile = path.join(options.out, "manifest.json");

// ---- build: compile every test, write the manifest ----
const build = async () =>
{
    const exe = process.platform === "win32" ? "tslang.exe" : "tslang";
    options.tslang ??= [
        path.join(repo, "__build/tslang/windows-msbuild-2026-release/bin", exe),
        path.join(repo, "__build/tslang/ninja/release/bin", exe),
    ].find((p) => fs.existsSync(p)) ?? exe;
    options.defaultLibRepo ??= path.resolve(repo, "../TypeScriptCompilerDefaultLib");
    options.defaultLibPath ??= path.join(options.defaultLibRepo, "__build");
    const gcLib = options.androidLib ?? path.join(repo, "3rdParty/gc/android", options.abi, "release/lib");
    const runtimeLib = options.androidLib ?? path.join(repo, "__build/tslang-runtime/release/android", options.abi);
    const ndk = options.ndk ?? process.env.ANDROID_NDK_HOME;
    if (!ndk) { console.error("no Android NDK: pass --android-ndk-path or set ANDROID_NDK_HOME"); process.exit(2); }
    // the programs carry the debug information of the collector, libc++ and the default library's
    // C++ wrappers: four times their size, more than an artifact and an emulator should take
    const host = { win32: "windows-x86_64", linux: "linux-x86_64", darwin: "darwin-x86_64" }[process.platform];
    const strip = path.join(ndk, "toolchains/llvm/prebuilt", host, "bin", process.platform === "win32" ? "llvm-strip.exe" : "llvm-strip");
    if (!fs.existsSync(strip)) { console.error(`no llvm-strip in the NDK: ${strip}`); process.exit(2); }

    const skips = loadSkips(path.join(here, "android_skips.txt"), (x) => x in ABIS);
    const suites = makeSuites(options.defaultLibRepo);
    // the environment the compiler looks the libraries up in, and nothing of the user's own lib paths
    const compileEnv = { DEFAULT_LIB_PATH: "", GC_LIB_PATH: "", LLVM_LIB_PATH: "", TSLANG_LIB_PATH: "", ANDROID_NDK_HOME: ndk };
    const libArgs = [`--default-lib-path=${options.defaultLibPath}`, `--gc-lib-path=${gcLib}`, `--tslang-lib-path=${runtimeLib}`];

    const legs = [];
    for (const suiteName of options.suite)
    {
        if (!suites[suiteName]) { console.error(`unknown suite ${suiteName}: tester or defaultlib`); process.exit(2); }
        for (const mm of options.mm)
            legs.push({ suiteName, suite: suites[suiteName], target: options.abi, mm, linux: true, name: `${suiteName}-${options.abi}-${mm}` });
    }

    fs.rmSync(options.out, { recursive: true, force: true });
    const jobs = [];
    const results = [];
    for (const leg of legs)
    {
        fs.mkdirSync(path.join(options.out, leg.name), { recursive: true });
        for (const file of leg.suite.files().filter((f) => !options.filter || options.filter.test(f)).sort())
        {
            const source = fs.readFileSync(path.join(leg.suite.dir, file), "utf8");
            const reason = skipReason(skips, leg.suite, file, source, leg);
            if (reason) { results.push({ leg: leg.name, file, status: "SKIP", detail: reason }); continue; }
            jobs.push({ leg, file });
        }
    }

    const built = [];
    const compile = async ({ leg, file }) =>
    {
        const source = path.join(leg.suite.dir, file);
        const name = file.replace(/\.ts$/, "");
        const binary = path.join(options.out, leg.name, name);
        const compiled = await runProcess(options.tslang, ["--emit=exe", `-mm=${leg.mm}`, `-mtriple=${triple}`, "--opt",
            ...leg.suite.compileArgs, ...libArgs, `-o=${binary}`, source], { env: compileEnv, timeout: 300 });
        if (compiled.code !== 0 || !fs.existsSync(binary))
        {
            const detail = firstLines(compiled.stderr + compiled.stdout, 6);
            if (leg.mm !== "own") return { leg: leg.name, file, status: "COMPILE", detail };

            const obj = `${binary}.native.o`;
            const native = await runProcess(options.tslang, ["--emit=obj", `-mm=${leg.mm}`, "--opt", ...leg.suite.compileArgs,
                `-o=${obj}`, source], { env: compileEnv, timeout: 300 });
            fs.rmSync(obj, { force: true });
            if (native.code === 0) return { leg: leg.name, file, status: "COMPILE", detail };
            return { leg: leg.name, file, status: "NATIVE", detail: "native: " + firstLines(native.stderr + native.stdout, 4) };
        }

        const stripped = await runProcess(strip, [binary], { timeout: 120 });
        if (stripped.code !== 0) return { leg: leg.name, file, status: "COMPILE", detail: `llvm-strip: ${firstLines(stripped.stderr + stripped.stdout, 4)}` };

        const stdin = source.replace(/\.ts$/, ".stdin");
        if (fs.existsSync(stdin)) fs.copyFileSync(stdin, `${binary}.stdin`);
        built.push({ leg: leg.name, file, name, cwd: leg.suiteName === "defaultlib" ? "dl" : leg.name, stdin: fs.existsSync(stdin) });
        return { leg: leg.name, file, status: "BUILT" };
    };

    console.log(`tslang: ${options.tslang}\ntarget: ${triple}\ndefault lib: ${options.defaultLibPath}\nlegs: ${legs.map((l) => l.name).join(", ")}; ${jobs.length} to build, ${results.length} skips`);
    await runJobs(jobs, options.jobs, compile, results);

    // what is not to run: skipped, not compiled, NATIVE; the run step reports it with its own results
    const manifest = { triple, legs: legs.map((l) => l.name), results: results.filter((r) => r.status !== "BUILT"), programs: built };
    fs.writeFileSync(manifestFile, JSON.stringify(manifest, null, 1));
    console.log(`built ${built.length} programs into ${options.out}`);
    return manifest;
};

// ---- run: push the programs, run them on the device, judge what comes back ----
const DEVICE = "/data/local/tmp/tslang-tests";

// one line a program: <id> <leg>/<name> <cwd> <stdin: 0/1>. Each of the jobs takes every jobs-th line.
const deviceScript = (timeout, jobs) => `#!/system/bin/sh
D=${DEVICE}
cd $D || exit 1
mkdir -p res
runone() {
  id=$1; bin=$2; wd=$3; in=/dev/null
  [ "$4" = 1 ] && in=$D/$bin.stdin
  cd $D/$wd
  # an adb-shell process the kernel kills for memory takes the whole emulator down: limit it
  # toybox's timeout (API 29) exits with 124 for a program a signal ended too, not only for one it
  # timed out: the seconds it ran tell them apart
  start=$(date +%s)
  ( ulimit -v 2097152; timeout ${timeout} $D/$bin < $in > $D/res/$id.out 2> $D/res/$id.err ); code=$?
  secs=$(( $(date +%s) - start ))
  done=0; grep -q 'done\\.' $D/res/$id.out $D/res/$id.err 2>/dev/null && done=1
  all=0; grep -q 'ALL DONE' $D/res/$id.out 2>/dev/null && all=1
  errbytes=$(wc -c < $D/res/$id.err)
  tail -c 2000 $D/res/$id.out > $D/res/$id.otail; tail -c 2000 $D/res/$id.err > $D/res/$id.etail
  rm -f $D/res/$id.out $D/res/$id.err
  echo "$code $done $all $errbytes $secs" > $D/res/$id.sum
}
worker() {
  n=0
  while read id bin wd in; do
    [ $((n % ${jobs})) = $1 ] && runone $id $bin $wd $in
    n=$((n + 1))
  done < $D/programs.txt
}
k=0
while [ $k -lt ${jobs} ]; do worker $k & k=$((k + 1)); done
wait
echo finished
`;

const adb = (args, opts = {}) => runProcess(options.adb, args, { timeout: opts.timeout ?? 600, input: opts.input });

const run = async (manifest) =>
{
    const state = await adb(["get-state"], { timeout: 30 });
    if (state.code !== 0 || state.stdout.trim() !== "device") { console.error(`no device: adb get-state says ${(state.stdout + state.stderr).trim()}`); process.exit(2); }

    const programs = manifest.programs;
    fs.writeFileSync(path.join(options.out, "run.sh"), deviceScript(options.timeout, options.deviceJobs));
    const step = async (what, args, timeout) =>
    {
        const r = await adb(args, { timeout });
        if (r.code !== 0) { console.error(`${what} failed: ${(r.stdout + r.stderr).trim()}`); process.exit(1); }
        return r;
    };
    await step("clear", ["shell", `rm -rf ${DEVICE} && mkdir -p ${DEVICE}/dl/tests`]);
    await step("push run.sh", ["push", path.join(options.out, "run.sh"), `${DEVICE}/run.sh`]);

    // a leg at a time: all of them would not fit on an emulator's data partition
    const res = path.join(options.out, "res");
    fs.rmSync(res, { recursive: true, force: true });
    for (const leg of manifest.legs)
    {
        const mine = programs.map((p, i) => ({ p, i })).filter(({ p }) => p.leg === leg);
        if (!mine.length) continue;
        fs.writeFileSync(path.join(options.out, "programs.txt"), mine.map(({ p, i }) => `${i} ${p.leg}/${p.name} ${p.cwd} ${p.stdin ? 1 : 0}`).join("\n") + "\n");
        await step(`push ${leg}`, ["push", path.join(options.out, leg), `${DEVICE}/`], 1800);
        await step("push programs.txt", ["push", path.join(options.out, "programs.txt"), `${DEVICE}/programs.txt`]);
        // pushed from Windows or out of a CI artifact, a program has lost its executable bit
        await step("chmod", ["shell", `chmod -R 755 ${DEVICE}/${leg}`]);

        const started = Date.now();
        const ran = await adb(["shell", `sh ${DEVICE}/run.sh`], { timeout: 3 * 3600 });
        console.log(`${leg}: ${mine.length} programs in ${Math.round((Date.now() - started) / 1000)} s, ${options.deviceJobs} at a time: ${(ran.stdout + ran.stderr).trim().split(/\r?\n/).slice(-1)[0]}`);
        await step("pull results", ["pull", `${DEVICE}/res`, options.out], 1800);
        await step("clean up", ["shell", `rm -rf ${DEVICE}/${leg} ${DEVICE}/res`]);
    }

    const suites = makeSuites(options.defaultLibRepo ?? path.resolve(repo, "../TypeScriptCompilerDefaultLib"));
    const results = [...manifest.results];
    const read = (f) => (fs.existsSync(f) ? fs.readFileSync(f, "utf8") : "");
    programs.forEach((p, i) =>
    {
        const sum = read(path.join(res, `${i}.sum`)).trim();
        if (!sum)
        {
            results.push({ leg: p.leg, file: p.file, status: "FAIL", detail: "no result: the device went away, or the run did not reach it" });
            return;
        }

        const [code, done, all, errBytes, secs] = sum.split(/\s+/).map(Number);
        const out = read(path.join(res, `${i}.otail`)), err = read(path.join(res, `${i}.etail`));
        const detail = lastLines(out + err, 8);
        if (code === 124 && secs >= options.timeout - 1) { results.push({ leg: p.leg, file: p.file, status: "TIMEOUT", detail }); return; }
        if (code === 124)
        {
            results.push({ leg: p.leg, file: p.file, status: "FAIL", detail: `ended by a signal after ${secs} s (crash, abort or out of memory)\n${detail}` });
            return;
        }

        // what the suite judges, from what came back: the markers as found in the whole output
        const suite = p.leg.startsWith("defaultlib-") ? suites.defaultlib : suites.tester;
        const failure = suite.passed({
            code,
            stdout: out + (done ? "\n[done.]" : "") + (all ? "\n[ALL DONE]" : ""),
            stderr: errBytes > 0 && !err.trim() ? "" : err,
        });
        results.push(failure ? { leg: p.leg, file: p.file, status: "FAIL", detail: `${failure}${code > 128 ? ` (signal ${code - 128})` : ""}\n${detail}` }
                             : { leg: p.leg, file: p.file, status: "PASS" });
    });

    process.exitCode = report(manifest.legs, results, options.out);
};

const manifest = options.build ? await build() : JSON.parse(fs.readFileSync(manifestFile, "utf8"));
if (options.run) await run(manifest);
else process.exitCode = manifest.results.some((r) => r.status === "COMPILE") ? 1 : 0;
