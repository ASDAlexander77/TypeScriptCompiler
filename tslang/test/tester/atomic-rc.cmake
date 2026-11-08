# A `-mm=rc` shared library called from several threads at once (atomic-rc-host.cpp), every one of
# them taking and dropping references to the same objects: built with --atomic-rc, or with the
# objects marked shared by `Shared.share` once at load.
#
# Under plain `rc` a count change is a load, an add and a store, so two threads changing the same
# count at once lose one of the changes: the count drifts, and the object is freed while it is
# still held or never freed at all. --atomic-rc makes each change one atomic read-modify-write;
# `Shared.share(v)` sets a bit in the header of every block v reaches, and only those blocks'
# counts change atomically. Either way the count after the threads are done has to be the count
# before, exactly.
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
interface HasV {
    v: number;
}

class Child {
    v: number = 1;
}

function makeReader(c: Child) {
    return () => c.v;
}

class Node {
    v: number = 1;
    // a literal: immortal, which marking has to leave alone (its header may be read-only)
    tag = "node";
    child = new Child();
    children: Child[] = [new Child(), new Child()];
    // reached only through a descriptor: what an `any`, an interface, a union and a closure hold
    boxed: any = new Child();
    viewed: HasV = new Child();
    either: Child | string = new Child();
    reader = makeReader(new Child());
}

class Box {
    s: Shared<Node>;
    child: Child;
    children: Child[];
    tag: string;
    boxed: any;
    viewed: HasV;
    either: Child | string;
    reader: () => number;

    constructor(s: Shared<Node>) {
        const node = s.value;
        this.s = s;
        this.child = node.child;
        this.children = node.children;
        this.tag = node.tag;
        this.boxed = node.boxed;
        this.viewed = node.viewed;
        this.either = node.either;
        this.reader = node.reader;
    }
}

// The root every thread starts from: written here, while one thread runs, and only read after.
// Without @atomic or Shared.share its blocks are counted plainly.
ATOMIC_IT class Holder {
    s = new Shared(new Node());
    late: Child | undefined;
}

const H = new Holder();
// stored after the constructor: an @atomic class's field barrier marks it
H.late = new Child();
SHARE_IT

// every Box takes a reference to each of them, and gives it back when it goes
export function churn(n: number): number {
    let sum: number = 0;
    for (let i = 0; i < n; i++) {
        const box = new Box(H.s);
        const late = H.late;
        sum += box.s.value.v * box.child.v * box.children[1].v * box.viewed.v * box.reader() * (late ? late.v : 0);
    }

    return sum;
}

export function count(): number {
    return Shared.count(H.s);
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

file(READ "${WORK_DIR}/shared_churn.ts" churn_source)
string(REPLACE "ATOMIC_IT" "" churn_source_plain "${churn_source}")
string(REPLACE "SHARE_IT" "" unmarked "${churn_source_plain}")
string(REPLACE "SHARE_IT" "Shared.share(H);" marked "${churn_source_plain}")
string(REPLACE "ATOMIC_IT" "@atomic" churn_source_atomic "${churn_source}")
string(REPLACE "SHARE_IT" "" atomic_class "${churn_source_atomic}")
file(WRITE "${WORK_DIR}/unmarked.ts" "${unmarked}")
file(WRITE "${WORK_DIR}/marked.ts" "${marked}")
file(WRITE "${WORK_DIR}/atomic_class.ts" "${atomic_class}")

function(build_and_run name source expect_exact)
    set(library "${WORK_DIR}/${prefix}${name}${suffix}")
    execute_process(COMMAND "${TSLANG}" --emit=dll --opt -mm=rc ${ARGN} --no-default-lib ${pic} ${libs}
                            ${source} -o "${library}"
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

# every count atomic
build_and_run(atomic unmarked.ts TRUE --atomic-rc)
# Shared.share(H) at load: everything H reaches is marked, and only those counts change atomically
build_and_run(marked marked.ts TRUE)
# `@atomic class Holder`: marked once its constructor has run, and `H.late = ...` marked by the
# field barrier
build_and_run(atomic-class atomic_class.ts TRUE)
# both: a block with the shared bit under --atomic-rc still frees at the right count
build_and_run(marked-atomic marked.ts TRUE --atomic-rc)
# neither: reported only
build_and_run(plain unmarked.ts FALSE)
