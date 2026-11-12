// Runs one tslang wasm program under Node: node run.mjs program.wasm [args...]
//
// The exit code is the program's: what main returned or exit() was given, 1 for a trap or an
// uncaught exception. A wasm64 program needs Node's memory64 support, on by default from Node 24
// and behind --experimental-wasm-memory64 before it (run_tests.mjs passes it).
import fs from "node:fs";
import os from "node:os";
import process from "node:process";
import worker_threads from "node:worker_threads";
import { createHost, UncaughtThrow } from "./host.mjs";

const file = process.argv[2];
if (!file)
{
    console.error("usage: node run.mjs program.wasm [args...]");
    process.exit(2);
}

const host = createHost({
    args: process.argv.slice(2),
    node: { fs, process, worker_threads, os },
    write: (fd, text) => fs.writeSync(fd, text),
});

let code;
try
{
    const instance = await host.instantiate(fs.readFileSync(file));
    code = host.run(instance);
}
catch (e)
{
    if (e instanceof UncaughtThrow)
    {
        fs.writeSync(2, "Uncaught exception\n");
    }
    else
    {
        fs.writeSync(2, `${e?.stack ?? e}\n`);
    }

    code = 1;
}

process.exitCode = code & 0xff;
