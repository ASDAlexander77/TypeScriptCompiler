# A program reads a shared library's class fields at the offsets it was compiled with. Built again with
# another layout, the library was read wrongly and silently (a field moved: the program printed its
# neighbour). Now the program checks each class's member table in the library as it loads it
# (ts.CheckMembers, MemberTableCheck.h) and stops, naming what changed.
#
# The program is built against v1 of the library and run with v1 (it works), then with v2 (a field
# added in front: `a` moved) and v3 (a private field added at the end: the offsets stand, the instance
# size changed - a class of the program extending Box would be laid out over it). Under the JIT the
# program is compiled again against the library it loads, and reads it right.

cmake_minimum_required(VERSION 3.17.3)

foreach(var TSLANG WORK_DIR GC_LIB TSLANG_LIB LLVM_LIB)
    if(NOT DEFINED ${var})
        message(FATAL_ERROR "${var} is required")
    endif()
endforeach()

file(REMOVE_RECURSE "${WORK_DIR}")
file(MAKE_DIRECTORY "${WORK_DIR}")

set(ENV{GC_LIB_PATH} "")
set(ENV{GC_SHARED_LIB_PATH} "")
set(ENV{TSLANG_LIB_PATH} "")

set(make [=[

export function make(): Box {
    return new Box();
}
]=])

file(WRITE "${WORK_DIR}/v1.ts" "export class Box {\n    a: number = 1;\n    b: number = 2;\n}\n${make}")
file(WRITE "${WORK_DIR}/v2.ts" "export class Box {\n    z: number = 0;\n    a: number = 1;\n    b: number = 2;\n}\n${make}")
file(WRITE "${WORK_DIR}/v3.ts" "export class Box {\n    a: number = 1;\n    b: number = 2;\n    private c: number = 3;\n}\n${make}")
file(WRITE "${WORK_DIR}/program.ts" [=[
import "./lib";

class Mine extends Box {
    m: number = 4;
}

function main() {
    const box = make();
    print("b =", box.b);
    const mine = new Mine();
    print("m =", mine.m);
    print("done.");
}
]=])

set(libs "--gc-lib-path=${GC_LIB}" "--tslang-lib-path=${TSLANG_LIB}" "--llvm-lib-path=${LLVM_LIB}")
if(WIN32)
    set(library "lib.dll")
    set(program "program.exe")
    set(pic "")
else()
    set(library "liblib.so")
    set(program "program")
    set(pic "-relocation-model=pic")
endif()

function(run what)
    execute_process(COMMAND ${ARGN} WORKING_DIRECTORY "${WORK_DIR}" OUTPUT_VARIABLE out ERROR_VARIABLE err RESULT_VARIABLE status)
    set(run_out "${out}${err}" PARENT_SCOPE)
    set(run_status "${status}" PARENT_SCOPE)
endfunction()

function(build_library version)
    file(COPY_FILE "${WORK_DIR}/${version}.ts" "${WORK_DIR}/lib.ts")
    run("lib" "${TSLANG}" --emit=dll -mm=none --no-default-lib ${pic} ${libs} lib.ts -o "${library}")
    if(NOT run_status EQUAL 0)
        message(FATAL_ERROR "--emit=dll ${version}: exit ${run_status}\n${run_out}")
    endif()
endfunction()

function(expect_runs what)
    foreach(mode exe jit)
        if(mode STREQUAL "exe")
            run(${what} "${WORK_DIR}/${program}")
        else()
            run(${what} "${TSLANG}" --emit=jit -mm=none --no-default-lib ${libs} program.ts)
        endif()
        if(NOT run_status EQUAL 0 OR NOT run_out MATCHES "b = 2" OR NOT run_out MATCHES "done\\.")
            message(FATAL_ERROR "${what} (${mode}): expected to run, exit ${run_status}\n${run_out}")
        endif()
    endforeach()
endfunction()

# the executable was built against v1 and stops; the JIT compiles the program again against the library
# in front of it, so it reads it right
function(expect_stops what message)
    run(${what} "${WORK_DIR}/${program}")
    if(run_status EQUAL 0 OR NOT run_out MATCHES "${message}" OR run_out MATCHES "b = ")
        message(FATAL_ERROR "${what} (exe): expected '${message}' before main, exit ${run_status}
${run_out}")
    endif()

    run(${what} "${TSLANG}" --emit=jit -mm=none --no-default-lib ${libs} program.ts)
    if(NOT run_status EQUAL 0 OR NOT run_out MATCHES "b = 2" OR NOT run_out MATCHES "m = 4" OR NOT run_out MATCHES "done\\.")
        message(FATAL_ERROR "${what} (jit): expected to run against it, exit ${run_status}
${run_out}")
    endif()
endfunction()

build_library(v1)
run("program" "${TSLANG}" --emit=exe -mm=none --no-default-lib ${libs} program.ts -o "${program}")
if(NOT run_status EQUAL 0)
    message(FATAL_ERROR "--emit=exe program: exit ${run_status}\n${run_out}")
endif()

expect_runs("v1")

build_library(v2)
expect_stops("v2" "library '[^']*': Box\\.a moved from offset [0-9]+ to [0-9]+; rebuild this program against the new library")

build_library(v3)
expect_stops("v3" "library '[^']*': Box changed size from [0-9]+ to [0-9]+ bytes; rebuild this program against the new library")

# Fields of every kind round-trip: the program, built against the library's declarations, has each field's type
# as the library has it (its type hash), so a library unchanged loads - under -mm=none and -mm=rc.
file(WRITE "${WORK_DIR}/mixed.ts" [=[
export enum Color {
    Red,
    Green,
}

export class Other {
    n: number = 1;
}

export class Mixed {
    c: Color = Color.Green;
    o: Other = new Other();
    arr: number[] = [1, 2];
    opt?: string;
    f: (x: number) => number = (x: number) => x + 1;
    s: string = "s";
    i: int = 3;
    flag: boolean = true;
}

export function makeMixed(): Mixed {
    return new Mixed();
}
]=])
file(WRITE "${WORK_DIR}/mixed_program.ts" [=[
import "./mixed";

function main() {
    const m = makeMixed();
    print("mixed", m.c == Color.Green, m.o.n, m.arr.length, m.f(1), m.s, m.i, m.flag);
    print("done.");
}
]=])
foreach(mm none rc)
    if(WIN32)
        set(mixed_library "mixed.dll")
        set(mixed_program "mixed_program.exe")
    else()
        set(mixed_library "libmixed.so")
        set(mixed_program "mixed_program")
    endif()
    run("mixed" "${TSLANG}" --emit=dll -mm=${mm} --no-default-lib ${pic} ${libs} mixed.ts -o "${mixed_library}")
    if(NOT run_status EQUAL 0)
        message(FATAL_ERROR "--emit=dll mixed -mm=${mm}: exit ${run_status}
${run_out}")
    endif()
    run("mixed" "${TSLANG}" --emit=exe -mm=${mm} --no-default-lib ${libs} mixed_program.ts -o "${mixed_program}")
    if(NOT run_status EQUAL 0)
        message(FATAL_ERROR "--emit=exe mixed_program -mm=${mm}: exit ${run_status}
${run_out}")
    endif()
    run("mixed" "${WORK_DIR}/${mixed_program}")
    if(NOT run_status EQUAL 0 OR NOT run_out MATCHES "mixed true 1 2 2 s 3 true" OR NOT run_out MATCHES "done\\.")
        message(FATAL_ERROR "mixed -mm=${mm}: expected an unchanged library to load, exit ${run_status}
${run_out}")
    endif()
endforeach()

# A library built by an older tslang has no member tables (no `// @members` in its declarations) and no slot for
# one in its class vtables: importing it is an error, as every virtual call into it would land one slot off.
file(READ "${WORK_DIR}/lib.d.ts" declarations)
string(REGEX REPLACE "// @members\r?\n" "" declarations "${declarations}")
file(WRITE "${WORK_DIR}/lib.d.ts" "${declarations}")
run("old" "${TSLANG}" --emit=exe -mm=none --no-default-lib ${libs} program.ts -o old_program.exe)
if(run_status EQUAL 0 OR NOT run_out MATCHES "was built by an older tslang \\(its classes have no member tables\\): build it again with this tslang")
    message(FATAL_ERROR "a library without member tables: expected an error, exit ${run_status}
${run_out}")
endif()

message(STATUS "a library's class layout is checked as it loads")
