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
#
# Then, when a default library has been built (DEFAULT_LIB_CANDIDATES, DEFAULT_LIB_PATH), a library
# linked against its DLL: the shared objects are a RegExp and strings the default library made, and
# RegExp.test runs in its DLL on all threads at once. The descriptor whose share routine marks them
# is the default library's, so this is where a library built before the share slot would show.
# Under --atomic-rc alone the default library's own code still counts an unmarked block plainly,
# so that build is only reported.

cmake_minimum_required(VERSION 3.17.3)

foreach(var TSLANG HOST WORK_DIR TSLANG_LIB LLVM_LIB DEFAULT_LIB_CANDIDATES)
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
set(default_lib_option --no-default-lib)

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
    execute_process(COMMAND "${TSLANG}" --emit=dll --opt -mm=rc ${ARGN} ${default_lib_option} ${pic} ${libs}
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

# Against the default library's DLL, when one has been built for this target and model
execute_process(COMMAND "${TSLANG}" --print-default-lib-dir=dll --opt -mm=rc
    OUTPUT_VARIABLE default_lib_dll_subdir
    OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE status)
if(NOT status EQUAL 0 OR default_lib_dll_subdir STREQUAL "")
    message(FATAL_ERROR "tslang --print-default-lib-dir=dll --opt -mm=rc: exit ${status}")
endif()

set(default_lib "")
foreach(candidate "$ENV{DEFAULT_LIB_PATH}" ${DEFAULT_LIB_CANDIDATES})
    if(NOT candidate STREQUAL "" AND IS_DIRECTORY "${candidate}/${default_lib_dll_subdir}")
        set(default_lib "${candidate}")
        break()
    endif()
endforeach()

if(default_lib STREQUAL "")
    message(STATUS "no default library built in ${default_lib_dll_subdir}: its DLL is not checked")
    return()
endif()

message(STATUS "default library: ${default_lib}")
set(ENV{DEFAULT_LIB_PATH} "")
set(default_lib_option "--default-lib-path=${default_lib}")
# the host loads a library's dependencies from the library's folder
file(COPY "${default_lib}/${default_lib_dll_subdir}/${prefix}TypeScriptDefaultLib${suffix}" DESTINATION "${WORK_DIR}")

file(WRITE "${WORK_DIR}/defaultlib_churn.ts" [=[
class Child {
    v: number = 1;
}

class Box {
    s: Shared<Child>;
    rx: Shared<RegExp>;
    words: string[];

    constructor(s: Shared<Child>, rx: Shared<RegExp>, words: string[]) {
        this.s = s;
        this.rx = rx;
        this.words = words;
    }
}

// objects of the default library's: a RegExp, and strings split made
ATOMIC_IT class Holder {
    s = new Shared(new Child());
    rx = new Shared(new RegExp("ab"));
    words: string[] = "x,ab,y".split(",");
}

const H = new Holder();
SHARE_IT

export function churn(n: number): number {
    let sum: number = 0;
    for (let i = 0; i < n; i++) {
        const box = new Box(H.s, H.rx, H.words);
        // in the default library's DLL, on the shared RegExp and a shared string
        if (box.rx.value.test(box.words[1])) {
            sum += box.s.value.v;
        }
    }

    return sum;
}

export function count(): number {
    return Shared.count(H.s) * 1000 + Shared.count(H.rx);
}
]=])

file(READ "${WORK_DIR}/defaultlib_churn.ts" churn_source)
string(REPLACE "ATOMIC_IT " "" churn_source_plain "${churn_source}")
string(REPLACE "SHARE_IT" "" unmarked "${churn_source_plain}")
string(REPLACE "SHARE_IT" "Shared.share(H);" marked "${churn_source_plain}")
string(REPLACE "ATOMIC_IT " "@atomic " churn_source_atomic "${churn_source}")
string(REPLACE "SHARE_IT" "" atomic_class "${churn_source_atomic}")
file(WRITE "${WORK_DIR}/defaultlib_unmarked.ts" "${unmarked}")
file(WRITE "${WORK_DIR}/defaultlib_marked.ts" "${marked}")
file(WRITE "${WORK_DIR}/defaultlib_atomic_class.ts" "${atomic_class}")

build_and_run(defaultlib-marked defaultlib_marked.ts TRUE)
build_and_run(defaultlib-atomic-class defaultlib_atomic_class.ts TRUE)
build_and_run(defaultlib-marked-atomic defaultlib_marked.ts TRUE --atomic-rc)
# the default library's own count changes on these blocks stay plain: reported only
build_and_run(defaultlib-atomic defaultlib_unmarked.ts FALSE --atomic-rc)
build_and_run(defaultlib-plain defaultlib_unmarked.ts FALSE)
