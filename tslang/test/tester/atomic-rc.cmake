# --atomic-rc: a `-mm=rc` shared library called from several threads at once (atomic-rc-host.cpp),
# every one of them taking and dropping references to the same object.
#
# Under plain `rc` a count change is a load, an add and a store, so two threads changing the same
# count at once lose one of the changes: the count drifts, and the object is freed while it is
# still held or never freed at all. --atomic-rc makes each change one atomic read-modify-write.
# With it the count after the threads are done has to be the count before, exactly.
#
# The same library built without the flag is run too, but only reported: losing an update is a
# race, so its absence proves nothing, and a drift there is what the flag is for, not a failure.
# Seeing it drift is what shows this test can see the race at all.

cmake_minimum_required(VERSION 3.17.3)

foreach(var TSLANG HOST WORK_DIR TSLANG_LIB LLVM_LIB)
    if(NOT DEFINED ${var})
        message(FATAL_ERROR "${var} is required")
    endif()
endforeach()

file(REMOVE_RECURSE "${WORK_DIR}")
file(MAKE_DIRECTORY "${WORK_DIR}")

set(ENV{TSLANG_LIB_PATH} "")

file(WRITE "${WORK_DIR}/shared_churn.ts" [=[
class Node {
    v = 1;
}

class Box {
    s: Shared<Node>;

    constructor(s: Shared<Node>) {
        this.s = s;
    }
}

// written once, here, and only read after: the threads share the block, not the global
const G = new Shared(new Node());

// every Box takes a reference to G's block and gives it back when it goes
export function churn(n: number): number {
    let sum = 0;
    for (let i = 0; i < n; i++) {
        const box = new Box(G);
        sum += box.s.value.v;
    }

    return sum;
}

export function count(): number {
    return Shared.count(G);
}
]=])

set(libs "--tslang-lib-path=${TSLANG_LIB}" "--llvm-lib-path=${LLVM_LIB}")

if(WIN32)
    set(prefix "")
    set(suffix ".dll")
    set(pic "")
else()
    set(prefix "lib")
    set(suffix ".so")
    set(pic "-relocation-model=pic")
endif()

function(build_and_run name expect_exact)
    set(library "${WORK_DIR}/${prefix}${name}${suffix}")
    execute_process(COMMAND "${TSLANG}" --emit=dll --opt -mm=rc ${ARGN} --no-default-lib ${pic} ${libs}
                            shared_churn.ts -o "${library}"
        WORKING_DIRECTORY "${WORK_DIR}"
        OUTPUT_VARIABLE out
        ERROR_VARIABLE err
        RESULT_VARIABLE status)
    if(NOT status EQUAL 0)
        message(FATAL_ERROR "--emit=dll ${name}: exit ${status}\n${out}\n${err}")
    endif()

    execute_process(COMMAND "${HOST}" "${library}" 8 200000
        WORKING_DIRECTORY "${WORK_DIR}"
        OUTPUT_VARIABLE out
        ERROR_VARIABLE err
        RESULT_VARIABLE status
        TIMEOUT 240)
    if(expect_exact)
        if(NOT status EQUAL 0)
            message(FATAL_ERROR "${name}: the count after the threads is not the count before: exit ${status}\n${out}\n${err}")
        endif()
        message(STATUS "${name}: ${out}")
    else()
        message(STATUS "${name} (reported only, exit ${status}): ${out}")
    endif()
endfunction()

build_and_run(atomic TRUE --atomic-rc)
build_and_run(plain FALSE)
