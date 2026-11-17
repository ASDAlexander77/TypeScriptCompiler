# own spec 93: a read of a @weak field whose object was freed stops the program. Builds SOURCE under
# -mm=own with `tslang --emit=exe` and runs it: the run must fail, print the message naming the
# field, and print nothing after it ("after").

cmake_minimum_required(VERSION 3.17.3)

foreach(var TSLANG SOURCE FIELD WORK_DIR LLVM_LIB TSLANG_LIB)
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
set(message "the @weak field '${FIELD}' was read after its object was freed")
string(FIND "${err}${out}" "${message}" found)
if(status EQUAL 0 OR found EQUAL -1 OR out MATCHES "(^|\n)after\r?\n" OR NOT out MATCHES "(^|\n)before\r?\n")
    message(FATAL_ERROR "the program: exit ${status}, expected a failure with '${message}' after 'before' and no 'after': ${out}${err}")
endif()

message(STATUS "exit ${status}: ${err}")
