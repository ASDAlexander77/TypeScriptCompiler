# An async program built by `tslang --emit=exe` under every model but gc - which links no collector.
#
# The async runtime registers its pool threads with the collector under gc (AsyncGCThreads.h). That
# code used to sit in the scheduler's own object file, so every program that awaits anything pulled
# in references to GC_allow_register_threads, GC_register_my_thread, GC_unregister_my_thread and
# GC_get_stack_base - and under rc, none and own the compiler links no gc.lib, so none of them
# linked: "unresolved external symbol GC_allow_register_threads". It is an object file of its own
# now, pulled in only by the call to GC_enable_threads that the GC pass puts in a gc program.
#
# This drives `tslang --emit=exe` itself, which `test-runner` does not: it links with lld directly,
# and always with gc.lib, so the rc and none corpus runs of the same file never saw it.

cmake_minimum_required(VERSION 3.17.3)

foreach(var TSLANG TESTS_DIR WORK_DIR LLVM_LIB TSLANG_LIB OPT)
    if(NOT DEFINED ${var})
        message(FATAL_ERROR "${var} is required")
    endif()
endforeach()

# Only what the command line says: an inherited GC_LIB_PATH would put the collector back in reach.
set(ENV{GC_LIB_PATH} "")

file(REMOVE_RECURSE "${WORK_DIR}")
file(MAKE_DIRECTORY "${WORK_DIR}")

set(name 00async_await)

set(failures "")
foreach(model rc none own)
    execute_process(
        COMMAND "${TSLANG}" --emit=exe -mm=${model} --no-default-lib ${OPT}
                "--llvm-lib-path=${LLVM_LIB}" "--tslang-lib-path=${TSLANG_LIB}"
                "${TESTS_DIR}/${name}.ts" -o=${name}.${model}.exe
        WORKING_DIRECTORY "${WORK_DIR}"
        OUTPUT_VARIABLE out
        ERROR_VARIABLE err
        RESULT_VARIABLE status)

    if(NOT status EQUAL 0)
        # the first error, not a warning printed ahead of it
        if("${out}${err}" MATCHES "[^\n]*error[^\n]*")
            set(first_line "${CMAKE_MATCH_0}")
        else()
            string(REGEX REPLACE "\n.*" "" first_line "${err}")
        endif()
        list(APPEND failures "  -mm=${model}: --emit=exe: exit ${status}: ${first_line}")
        continue()
    endif()

    execute_process(
        COMMAND "${WORK_DIR}/${name}.${model}.exe"
        WORKING_DIRECTORY "${WORK_DIR}"
        OUTPUT_VARIABLE out
        ERROR_VARIABLE err
        RESULT_VARIABLE status)

    if(NOT status EQUAL 0 OR NOT out MATCHES "done\\.")
        list(APPEND failures "  -mm=${model}: the program: exit ${status}")
    endif()
endforeach()

if(failures)
    string(REPLACE ";" "\n" report "${failures}")
    message(FATAL_ERROR "an async program without the collector:\n${report}")
endif()

message(STATUS "an async program links and runs without the collector under rc, none and own")
