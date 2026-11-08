# The heap a Windows program allocates from (every memory model but gc): mimalloc
# (TypeScriptMimalloc.lib) in an executable tslang links, and under the JIT - unless the program
# loads a library tslang built, which allocates from the process heap and frees what the program
# allocated, or --process-heap asks for the process heap.
#
# An executable that has mimalloc prints its statistics at exit when MIMALLOC_SHOW_STATS is set;
# one that allocates from the process heap prints nothing. The JIT says which heap it gave the
# program under --verbose.

cmake_minimum_required(VERSION 3.17.3)

foreach(var TSLANG TESTS_DIR WORK_DIR GC_LIB LLVM_LIB TSLANG_LIB OPT)
    if(NOT DEFINED ${var})
        message(FATAL_ERROR "${var} is required")
    endif()
endforeach()

set(ENV{TSLANG_LIB_PATH} "")
set(ENV{GC_LIB_PATH} "")

file(REMOVE_RECURSE "${WORK_DIR}")
file(MAKE_DIRECTORY "${WORK_DIR}")

set(common --no-default-lib ${OPT} "--llvm-lib-path=${LLVM_LIB}" "--tslang-lib-path=${TSLANG_LIB}"
    "--gc-lib-path=${GC_LIB}")
set(program "${TESTS_DIR}/00mimalloc_heap.ts")

# run(<what> <command...>) - fails unless it succeeds; leaves the output in run_output
function(run what)
    execute_process(COMMAND ${ARGN}
        WORKING_DIRECTORY "${WORK_DIR}"
        OUTPUT_VARIABLE out
        ERROR_VARIABLE err
        RESULT_VARIABLE status)
    set(run_output "${out}${err}" PARENT_SCOPE)
    if(NOT status EQUAL 0 OR "${out}${err}" MATCHES "Stack dump")
        message(FATAL_ERROR "${what}: exit ${status}\n${out}\n${err}")
    endif()
endfunction()

# build(<exe> <expect mimalloc> <tslang options...>) - links <exe> with --verbose and checks whether
# TypeScriptMimalloc.lib is on the link line, then runs it and checks whether mimalloc is what it has
function(build exe expect_mimalloc)
    run("--emit=exe ${exe}" "${TSLANG}" --emit=exe --verbose ${common} ${ARGN} -o "${exe}")
    if(expect_mimalloc AND NOT run_output MATCHES "TypeScriptMimalloc")
        message(FATAL_ERROR "${exe}: TypeScriptMimalloc.lib is not linked\n${run_output}")
    endif()
    if(NOT expect_mimalloc AND run_output MATCHES "TypeScriptMimalloc")
        message(FATAL_ERROR "${exe}: TypeScriptMimalloc.lib is linked, expected the process heap\n${run_output}")
    endif()

    set(ENV{MIMALLOC_SHOW_STATS} "1")
    run("${exe}" "${WORK_DIR}/${exe}")
    unset(ENV{MIMALLOC_SHOW_STATS})
    if(NOT run_output MATCHES "done\\.")
        message(FATAL_ERROR "${exe} did not finish:\n${run_output}")
    endif()
    if(expect_mimalloc AND NOT run_output MATCHES "heap stats")
        message(FATAL_ERROR "${exe} does not allocate from mimalloc (no statistics at exit):\n${run_output}")
    endif()
    if(NOT expect_mimalloc AND run_output MATCHES "heap stats")
        message(FATAL_ERROR "${exe} has mimalloc, expected the process heap:\n${run_output}")
    endif()
endfunction()

# jit(<expected heap> <tslang options...>)
function(jit expected)
    run("--emit=jit ${ARGN}" "${TSLANG}" --emit=jit --verbose ${common} ${ARGN})
    if(NOT run_output MATCHES "done\\.")
        message(FATAL_ERROR "--emit=jit ${ARGN} did not finish:\n${run_output}")
    endif()
    if(NOT run_output MATCHES "JIT heap: ${expected}")
        message(FATAL_ERROR "--emit=jit ${ARGN}: expected 'JIT heap: ${expected}'\n${run_output}")
    endif()
endfunction()

# 1. An executable alone allocates from mimalloc, under own, rc and none; not under gc, nor with
#    --process-heap.
build(own.exe TRUE -mm=own "${program}")
build(rc.exe TRUE -mm=rc "${program}")
build(none.exe TRUE -mm=none "${program}")
build(gc.exe FALSE -mm=gc "${program}")
build(process.exe FALSE -mm=own --process-heap "${program}")

# 2. So does the program the JIT runs, from the cache or not; with --process-heap it does not.
jit("mimalloc" -mm=own "${program}")
jit("mimalloc" -mm=rc --jit-cache=false "${program}")
jit("process heap" -mm=own --process-heap "${program}")

# 3. A program importing a library tslang built shares that library's process heap: the importer
#    makes objects of the library's class, which the library's code releases.
run("--emit=dll" "${TSLANG}" --emit=dll ${common} -mm=rc "${TESTS_DIR}/export_class_interface.ts"
    -o export_class_interface.dll)
# `import './export_class_interface'` finds the DLL in the working directory before the source
build(import.exe FALSE -mm=rc "${TESTS_DIR}/import_class_interface.ts")
jit("process heap" -mm=rc "${TESTS_DIR}/import_class_interface.ts")
jit("process heap" -mm=rc --jit-cache=false "${TESTS_DIR}/import_class_interface.ts")
