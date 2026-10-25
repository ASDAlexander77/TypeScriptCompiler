# -mm=own's keepers across two modules compiled into objects of their own and linked into one program
# (docs/superpowers/specs/2026-10-25-own-immortal-keepers-design.md): own_keepers_lib.ts keeps the
# callbacks it is given for good, and says so with `@immortalizes`; own_keepers_main.ts registers a
# bound method and a closure from functions that return before the callbacks run. The importer reads
# the marker from the library's declaration and makes what it passes immortal, so the program runs
# with the right numbers after the memory its locals released is handed out again.

cmake_minimum_required(VERSION 3.17.3)

foreach(var TSLANG SOURCE_DIR WORK_DIR LLVM_LIB TSLANG_LIB)
    if(NOT DEFINED ${var})
        message(FATAL_ERROR "${var} is required")
    endif()
endforeach()

set(ENV{GC_LIB_PATH} "")
set(ENV{TSLANG_LIB_PATH} "")

file(REMOVE_RECURSE "${WORK_DIR}")
file(MAKE_DIRECTORY "${WORK_DIR}")
# the importer finds `./own_keepers_lib` beside it, as a source to read declarations from
foreach(file own_keepers_lib.ts own_keepers_main.ts)
    file(COPY "${SOURCE_DIR}/${file}" DESTINATION "${WORK_DIR}")
endforeach()

function(tslang_run what)
    execute_process(COMMAND "${TSLANG}" ${ARGN}
        WORKING_DIRECTORY "${WORK_DIR}"
        OUTPUT_VARIABLE out
        ERROR_VARIABLE err
        RESULT_VARIABLE status)
    set(status "${status}" PARENT_SCOPE)
    set(output "${out}${err}" PARENT_SCOPE)
endfunction()

tslang_run(lib --emit=obj -mm=own own_keepers_lib.ts -o own_keepers_lib.obj)
if(NOT status EQUAL 0 OR output MATCHES "error:|Stack dump")
    message(FATAL_ERROR "own_keepers_lib: exit ${status}\n${output}")
endif()

tslang_run(link --emit=exe -mm=own "--llvm-lib-path=${LLVM_LIB}" "--tslang-lib-path=${TSLANG_LIB}"
           --own-facts=own_keepers_lib.obj --obj=own_keepers_lib.obj own_keepers_main.ts -o own_keepers.exe)
if(NOT status EQUAL 0 OR output MATCHES "error:|Stack dump")
    message(FATAL_ERROR "linking own_keepers: exit ${status}\n${output}")
endif()

execute_process(COMMAND "${WORK_DIR}/own_keepers.exe"
    WORKING_DIRECTORY "${WORK_DIR}"
    OUTPUT_VARIABLE out
    ERROR_VARIABLE err
    RESULT_VARIABLE status)
if(NOT status EQUAL 0 OR NOT "${out}" MATCHES "^2 102 2[\r\n]+done[.]")
    message(FATAL_ERROR "own_keepers.exe: exit ${status}\n${out}${err}")
endif()
