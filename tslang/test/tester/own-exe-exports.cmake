# An executable's exports are seen by no other module (own spec 31, USER decision): built with
# `tslang --emit=exe`, own_interface_exe_exports.ts - an exported interface and an exported global
# holding a value of it - compiles under -mm=own and runs; built as an object file or a DLL, which
# another module may import, the interface is open and `G.get` stays an error. test-runner builds an
# object file with --entry-point and links it itself, so it cannot show the first half.

cmake_minimum_required(VERSION 3.17.3)

foreach(var TSLANG SOURCE WORK_DIR LLVM_LIB TSLANG_LIB)
    if(NOT DEFINED ${var})
        message(FATAL_ERROR "${var} is required")
    endif()
endforeach()

file(REMOVE_RECURSE "${WORK_DIR}")
file(MAKE_DIRECTORY "${WORK_DIR}")

set(failures "")

execute_process(
    COMMAND "${TSLANG}" --emit=exe -mm=own --no-default-lib --opt --opt_level=3
            "--llvm-lib-path=${LLVM_LIB}" "--tslang-lib-path=${TSLANG_LIB}"
            "${SOURCE}" -o=program.exe
    WORKING_DIRECTORY "${WORK_DIR}"
    OUTPUT_VARIABLE out
    ERROR_VARIABLE err
    RESULT_VARIABLE status)
if(NOT status EQUAL 0)
    list(APPEND failures "  --emit=exe: exit ${status}: ${out}${err}")
else()
    execute_process(
        COMMAND "${WORK_DIR}/program.exe"
        WORKING_DIRECTORY "${WORK_DIR}"
        OUTPUT_VARIABLE out
        ERROR_VARIABLE err
        RESULT_VARIABLE status)
    if(NOT status EQUAL 0 OR NOT out MATCHES "5\n.*done\.")
        list(APPEND failures "  the program: exit ${status}: ${out}${err}")
    endif()
endif()

foreach(emit obj dll)
    execute_process(
        COMMAND "${TSLANG}" --emit=${emit} -mm=own --no-default-lib
                "--llvm-lib-path=${LLVM_LIB}" "--tslang-lib-path=${TSLANG_LIB}"
                "${SOURCE}" -o=library.${emit}
        WORKING_DIRECTORY "${WORK_DIR}"
        OUTPUT_VARIABLE out
        ERROR_VARIABLE err
        RESULT_VARIABLE status)
    if(status EQUAL 0 OR NOT "${out}${err}" MATCHES "cannot tell the callers of 'G.get' that the result borrows it")
        list(APPEND failures "  --emit=${emit}: exit ${status}, expected the error on 'G.get': ${out}${err}")
    endif()
endforeach()

if(failures)
    string(REPLACE ";" "\n" report "${failures}")
    message(FATAL_ERROR "an executable's exports under -mm=own:\n${report}")
endif()

message(STATUS "an executable's exports leave its interfaces closed; an object's and a DLL's do not")
