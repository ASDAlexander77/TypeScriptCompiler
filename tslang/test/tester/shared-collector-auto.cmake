# tslang chooses the garbage collector's linkage by itself (Windows, -mm=gc).
#
# A shared library, and a program that imports one, must take Boehm from gc.dll: two statically
# linked collectors in one process each free what only the other's memory references (see
# docs/single-gc-collector-design.md). A program that is alone keeps the static gc.lib.
#
# This drives `tslang --emit=dll` / `--emit=exe` itself, which `test-runner` does not: its shared
# tests link with lld directly and pick the collector in the test harness, so they never exercise
# the compiler's own choice.
#
# The pair is the gc_single_collector one: the library builds the strings the program holds, then
# churns different ones, so a second collector in the library frees them.

cmake_minimum_required(VERSION 3.17.3)

foreach(var TSLANG TESTS_DIR WORK_DIR GC_LIB GC_SHARED_LIB LLVM_LIB TSLANG_LIB OPT)
    if(NOT DEFINED ${var})
        message(FATAL_ERROR "${var} is required")
    endif()
endforeach()

# Only what the command line says: an inherited GC_SHARED_LIB_PATH / GC_LIB_PATH would hide
# whether the compiler found the collector itself.
set(ENV{GC_SHARED_LIB_PATH} "")
set(ENV{GC_LIB_PATH} "")

file(REMOVE_RECURSE "${WORK_DIR}")
file(MAKE_DIRECTORY "${WORK_DIR}/lone")

set(common --no-default-lib ${OPT} "--llvm-lib-path=${LLVM_LIB}" "--tslang-lib-path=${TSLANG_LIB}")

# run(<what> <dir> <expect-success> <command...>) - leaves the output in run_output
function(run what dir expect_success)
    execute_process(COMMAND ${ARGN}
        WORKING_DIRECTORY "${dir}"
        OUTPUT_VARIABLE out
        ERROR_VARIABLE err
        RESULT_VARIABLE status)
    set(run_output "${out}${err}" PARENT_SCOPE)
    if(expect_success AND NOT status EQUAL 0)
        message(FATAL_ERROR "${what}: exit ${status}\n${out}\n${err}")
    endif()
    if(NOT expect_success AND status EQUAL 0)
        message(FATAL_ERROR "${what}: expected to fail, but succeeded\n${out}\n${err}")
    endif()
endfunction()

set(library "${TESTS_DIR}/export_gc_single_collector.ts")
set(program "${TESTS_DIR}/import_gc_single_collector.ts")

# 1. A shared library links gc.dll and gets gc.dll copied beside it.
run("--emit=dll" "${WORK_DIR}" TRUE
    "${TSLANG}" --emit=dll ${common} "--gc-lib-path=${GC_LIB}" "--gc-shared-lib-path=${GC_SHARED_LIB}"
    "${library}" -o export_gc_single_collector.dll)
if(NOT EXISTS "${WORK_DIR}/gc.dll")
    message(FATAL_ERROR "--emit=dll did not put gc.dll beside the library")
endif()

# 2. A program that imports it links gc.dll too, and the two share one collector.
run("--emit=exe importing a shared library" "${WORK_DIR}" TRUE
    "${TSLANG}" --emit=exe ${common} "--gc-lib-path=${GC_LIB}" "--gc-shared-lib-path=${GC_SHARED_LIB}"
    "${program}" -o main.exe)
run("the program" "${WORK_DIR}" TRUE "${WORK_DIR}/main.exe")
if(NOT run_output MATCHES "done\\.")
    message(FATAL_ERROR "the program did not finish:\n${run_output}")
endif()

# 3. --gc-lib-path naming a shared build is enough on its own (the default library's build script).
file(REMOVE "${WORK_DIR}/export_gc_single_collector.dll")
run("--emit=dll with --gc-lib-path at the shared build" "${WORK_DIR}" TRUE
    "${TSLANG}" --emit=dll ${common} "--gc-lib-path=${GC_SHARED_LIB}"
    "${library}" -o export_gc_single_collector.dll)
run("the program, against that library" "${WORK_DIR}" TRUE "${WORK_DIR}/main.exe")
if(NOT run_output MATCHES "done\\.")
    message(FATAL_ERROR "the program did not finish against the rebuilt library:\n${run_output}")
endif()

# 4. A program that is alone keeps the static collector: it runs with no gc.dll anywhere near it.
run("--emit=exe alone" "${WORK_DIR}/lone" TRUE
    "${TSLANG}" --emit=exe ${common} "--gc-lib-path=${GC_LIB}" "--gc-shared-lib-path=${GC_SHARED_LIB}"
    "${TESTS_DIR}/00funcs.ts" -o lone.exe)
if(EXISTS "${WORK_DIR}/lone/gc.dll")
    message(FATAL_ERROR "--emit=exe copied gc.dll beside a program that imports no shared library")
endif()
run("the lone program" "${WORK_DIR}/lone" TRUE "${WORK_DIR}/lone/lone.exe")

# 5. With no shared collector to be found, a shared library does not link rather than silently
#    getting a collector of its own.
run("--emit=dll with only a static collector" "${WORK_DIR}/lone" FALSE
    "${TSLANG}" --emit=dll ${common} "--gc-lib-path=${GC_LIB}"
    "${library}" -o nope.dll)
if(NOT run_output MATCHES "gc\\.dll")
    message(FATAL_ERROR "--emit=dll failed without naming gc.dll:\n${run_output}")
endif()

message(STATUS "tslang picks the shared collector for a library and its importer, and the static one for a lone program")
