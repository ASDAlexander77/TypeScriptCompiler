# -mm=own across two modules compiled into objects of their own and linked into one program
# (docs/superpowers/specs/2026-10-25-own-facts-from-objects-design.md): own_facts_main.ts holds a
# borrow of a field across calls of own_facts_lib.ts's methods. Given the library's object with
# --own-facts it compiles, links and runs; without, the calls may destroy anything and the borrow is
# an error; and a call whose facts say it overwrites the borrowed place (own_facts_main_drops.ts)
# stays an error with them.

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
# the importer finds `./own_facts_lib` beside it, as a source to read declarations from
foreach(file own_facts_lib.ts own_facts_main.ts own_facts_main_drops.ts)
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

tslang_run(lib --emit=obj -mm=own own_facts_lib.ts -o own_facts_lib.obj)
if(NOT status EQUAL 0 OR output MATCHES "error:|Stack dump")
    message(FATAL_ERROR "own_facts_lib: exit ${status}\n${output}")
endif()

# without the library's facts: Cam.reset may destroy anything, the borrow of this.cam among it
tslang_run(blind --emit=obj -mm=own own_facts_main.ts -o own_facts_main_blind.obj)
if(status EQUAL 0 OR NOT output MATCHES "borrows a field but is used here after it may be released or overwritten")
    message(FATAL_ERROR "own_facts_main without --own-facts compiled, or failed otherwise: exit ${status}\n${output}")
endif()

tslang_run(main --emit=obj -mm=own --own-facts=own_facts_lib.obj own_facts_main.ts -o own_facts_main.obj)
if(NOT status EQUAL 0 OR output MATCHES "error:|Stack dump")
    message(FATAL_ERROR "own_facts_main with --own-facts: exit ${status}\n${output}")
endif()

tslang_run(drops --emit=obj -mm=own --own-facts=own_facts_lib.obj own_facts_main_drops.ts -o own_facts_main_drops.obj)
if(status EQUAL 0 OR NOT output MATCHES ":10:11: error: '[^']*' borrows a field but is used here after it may be released or overwritten")
    message(FATAL_ERROR "own_facts_main_drops compiled, or failed otherwise: exit ${status}\n${output}")
endif()

tslang_run(link --emit=exe -mm=own "--llvm-lib-path=${LLVM_LIB}" "--tslang-lib-path=${TSLANG_LIB}"
           --own-facts=own_facts_lib.obj --obj=own_facts_lib.obj own_facts_main.ts -o own_facts.exe)
if(NOT status EQUAL 0)
    message(FATAL_ERROR "linking own_facts: exit ${status}\n${output}")
endif()

execute_process(COMMAND "${WORK_DIR}/own_facts.exe"
    WORKING_DIRECTORY "${WORK_DIR}"
    OUTPUT_VARIABLE out
    ERROR_VARIABLE err
    RESULT_VARIABLE status)
if(NOT status EQUAL 0 OR NOT "${out}" MATCHES "^11[\r\n]+done[.]")
    message(FATAL_ERROR "own_facts.exe: exit ${status}\n${out}${err}")
endif()
