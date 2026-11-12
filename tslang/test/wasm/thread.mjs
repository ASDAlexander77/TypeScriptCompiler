// One thread of a tslang wasm program, in a Node worker: host.mjs's tslang_thread_spawn starts it
// with the program's compiled module, its shared memory and what the thread runs (see runThread).
// The thread writes as the program does, straight to the file descriptors, so its output is not
// held back while the main thread waits in join.
import fs from "node:fs";
import os from "node:os";
import process from "node:process";
import worker_threads from "node:worker_threads";
import { createHost, writeAllSync } from "./host.mjs";

const { module, memory, memoryType, args, ...thread } = worker_threads.workerData;

const host = createHost({
    args,
    node: { fs, process, worker_threads, os },
    write: (fd, text) => writeAllSync(fs, fd, text),
});

try
{
    host.runThread(await host.instantiate(module, { memory, memoryType }), thread);
}
catch (e)
{
    // the thread did not start (runThread reports what ends a started one): join must still wake
    writeAllSync(fs, 2, `${e?.stack ?? e}\n`);
    const control = new Int32Array(memory.buffer, thread.control, 3);
    control[1] = 2;
    Atomics.store(control, 0, 1);
    Atomics.notify(control, 0);
}
