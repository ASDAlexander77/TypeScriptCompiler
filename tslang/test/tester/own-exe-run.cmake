# Builds a program under -mm=own with `tslang --emit=exe` and runs it: the output must match EXPECT
# and end with "done.". For a program that exports from its root, whose exports only an executable
# keeps from other modules (own spec 31, USER decision): test-runner builds an object file with
# --entry-point and links it itself, which another module could import, so it cannot build one.

cmake_minimum_required(VERSION 3.17.3)

foreach(var TSLANG SOURCE EXPECT WORK_DIR LLVM_LIB TSLANG_LIB)
    if(NOT DEFINED ${var})
        message(FATAL_ERROR "${var} is required")
    endif()
endforeach()

file(REMOVE_RECURSE "${WORK_DIR}")
file(MAKE_DIRECTORY "${WORK_DIR}")

execute_process(
    COMMAND "${TSLANG}" --emit=exe -mm=own --no-default-lib --opt --opt_level=3
            "--llvm-lib-path=${LLVM_LIB}" "--tslang-lib-path=${TSLANG_LIB}"
            "${SOURCE}" -o=program.exe
    WORKING_DIRECTORY "${WORK_DIR}"
    OUTPUT_VARIABLE out
    ERROR_VARIABLE err
    RESULT_VARIABLE status)
if(NOT status EQUAL 0)
    message(FATAL_ERROR "--emit=exe -mm=own: exit ${status}: ${out}${err}")
endif()

execute_process(
    COMMAND "${WORK_DIR}/program.exe"
    WORKING_DIRECTORY "${WORK_DIR}"
    OUTPUT_VARIABLE out
    ERROR_VARIABLE err
    RESULT_VARIABLE status)
if(NOT status EQUAL 0 OR NOT out MATCHES "${EXPECT}" OR NOT out MATCHES "done\\.")
    message(FATAL_ERROR "the program: exit ${status}, expected '${EXPECT}' and 'done.': ${out}${err}")
endif()

message(STATUS "${out}")
