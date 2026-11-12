// One thread of a tslang wasm program, in a Node worker: host.mjs's tslang_thread_spawn starts it
// with the program's compiled module, its shared memory and what the thread runs (see runThread).
// The thread writes to the file descriptors itself (bufferedOutput, flushed before it blocks and
// before join returns), so its output is not held back by the main thread.
import fs from "node:fs";
import os from "node:os";
import process from "node:process";
import worker_threads from "node:worker_threads";
import { createHost, bufferedOutput } from "./host.mjs";

const { module, memory, memoryType, args, ...thread } = worker_threads.workerData;

const output = bufferedOutput(fs);
const host = createHost({
    args,
    node: { fs, process, worker_threads, os },
    write: output.write,
    flush: output.flush,
});

try
{
    host.runThread(await host.instantiate(module, { memory, memoryType }), thread);
}
catch (e)
{
    // the thread did not start (runThread reports what ends a started one): join must still wake
    output.write(2, `${e?.stack ?? e}\n`);
    const control = new Int32Array(memory.buffer, thread.control, 3);
    control[1] = 2;
    Atomics.store(control, 0, 1);
    Atomics.notify(control, 0);
}
