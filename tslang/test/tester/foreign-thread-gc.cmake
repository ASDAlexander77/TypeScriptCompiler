# A `-mm=gc` shared library called from threads its host made (foreign-thread-gc-host.cpp).
#
# The collector scans, and stops during a collection, only the threads registered with it. A tslang
# program registers its own: GC_init the one it starts on, the async runtime its pool's. A host
# that is not a tslang program - an Android app's JNI threads, a C program's - calls in on threads
# nobody registered, and the first collection one of them started aborted ("Collecting from unknown
# thread"). An exported function of a gc shared library now registers the thread that calls it
# (__tslang_gc_enter, AsyncGCThreadsCommon.inc). The library allocates on every call, so a few
# thousand calls collect many times, on every thread.
#
# Also with no top-level code: then nothing ran GC_init at load, and the first call does.
#
# And with exported functions called while the library loads - a static constructor of an
# exported class, top-level code calling an export - which on Windows runs inside DllMain, under
# the loader lock. The first call there enabled threads, which started the collector's parallel
# marker threads and waited for them; they could not start until the load finished, and the load
# never did (the default library hung every JIT run). The allocations escape into globals, so the
# optimizer cannot fold the calls away.
#
# The host also checks that the library enabled the collector's threads on its loading thread's
# calls, before another thread came in (#523): enabling them starts the markers before the
# allocator takes its lock, so a newcomer enabling them raced a registered thread's allocation.
#
# And with the work done on the library's own async pool (#522), which collects there. The pool's
# threads register themselves through hooks that only the library's GC_enable_threads sets (each
# module has its own copy of the scheduler); when only a registered thread called in, they never
# were, and the collection a pool thread started aborted ("Collecting from unknown thread").

cmake_minimum_required(VERSION 3.17.3)

foreach(var TSLANG HOST WORK_DIR GC_LIB TSLANG_LIB LLVM_LIB)
    if(NOT DEFINED ${var})
        message(FATAL_ERROR "${var} is required")
    endif()
endforeach()

file(REMOVE_RECURSE "${WORK_DIR}")
file(MAKE_DIRECTORY "${WORK_DIR}")

set(ENV{GC_LIB_PATH} "")
set(ENV{GC_SHARED_LIB_PATH} "")
set(ENV{TSLANG_LIB_PATH} "")

set(work [=[
class Node {
    constructor(public value: number, public next: Node | undefined) {}
}

// a 200-node list per call: a loop of calls collects many times
export function work(seed: number): number {
    let head: Node | undefined = undefined;
    for (let i = 0; i < 200; i++) {
        head = new Node(seed + i, head);
    }

    let sum: number = 0;
    for (let node = head; node !== undefined; node = node.next) {
        sum += node.value;
    }

    return sum;
}
]=])

file(WRITE "${WORK_DIR}/with-top-level.ts" "${work}\nlet loaded: number = 1;\n")
file(WRITE "${WORK_DIR}/no-top-level.ts" "${work}")
file(WRITE "${WORK_DIR}/static-constructor-at-load.ts" "${work}
export class Registry {
    static first: Node = new Node(1, undefined);
}
")
file(WRITE "${WORK_DIR}/export-called-at-load.ts" "${work}
export function makeNode(value: number): Node {
    return new Node(value, undefined);
}

export let kept: Node = makeNode(1);
")
file(WRITE "${WORK_DIR}/async-pool.ts" [=[
declare function GC_gcollect(): void;

class Node {
    constructor(public value: number, public next: Node | undefined) {}
}

// runs on one of the library's pool threads, and now and then collects there
async function sumOnThePool(seed: number): number {
    let head: Node | undefined = undefined;
    for (let i = 0; i < 200; i++) {
        head = new Node(seed + i, head);
    }

    if (seed % 500 == 0) {
        GC_gcollect();
    }

    let sum: number = 0;
    for (let node = head; node !== undefined; node = node.next) {
        sum += node.value;
    }

    return sum;
}

export function work(seed: number): number {
    return await sumOnThePool(seed);
}
]=])

set(libs "--gc-lib-path=${GC_LIB}" "--tslang-lib-path=${TSLANG_LIB}" "--llvm-lib-path=${LLVM_LIB}")
if(DEFINED GC_SHARED_LIB)
    list(APPEND libs "--gc-shared-lib-path=${GC_SHARED_LIB}")
endif()

if(WIN32)
    set(prefix "")
    set(suffix ".dll")
    set(pic "")
else()
    set(prefix "lib")
    set(suffix ".so")
    # a shared object's code has to be position independent (R_X86_64_32S otherwise)
    set(pic "-relocation-model=pic")
endif()

foreach(name with-top-level no-top-level static-constructor-at-load export-called-at-load async-pool)
    set(library "${WORK_DIR}/${prefix}${name}${suffix}")
    execute_process(COMMAND "${TSLANG}" --emit=dll --opt -mm=gc --no-default-lib ${pic} ${libs} ${name}.ts -o "${library}"
        WORKING_DIRECTORY "${WORK_DIR}"
        OUTPUT_VARIABLE out
        ERROR_VARIABLE err
        RESULT_VARIABLE status)
    if(NOT status EQUAL 0)
        message(FATAL_ERROR "--emit=dll ${name}: exit ${status}\n${out}\n${err}")
    endif()

    execute_process(COMMAND "${HOST}" "${library}" 8 2000
        WORKING_DIRECTORY "${WORK_DIR}"
        OUTPUT_VARIABLE out
        ERROR_VARIABLE err
        RESULT_VARIABLE status
        TIMEOUT 240)
    if(NOT status EQUAL 0)
        message(FATAL_ERROR "${name}: the host's threads calling into the library: exit ${status}\n${out}\n${err}")
    endif()
    message(STATUS "${name}: ${out}")
endforeach()
