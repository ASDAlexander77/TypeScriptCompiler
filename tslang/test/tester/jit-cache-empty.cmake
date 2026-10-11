# Runs a program under the JIT cache from an empty cache, so the run compiles the program and every
# module it imports - several modules in one process, each in an MLIRContext of its own. (A test
# run by test-runner keeps its cache in __jit beside the source: from the second run on, it
# compiles nothing.)
#
# TSLANG, SOURCE (the program), WORK_DIR (the cache, emptied first), EXPECTED (its output).
# -mm=none and --no-default-lib keep the run to the compiler alone.

cmake_minimum_required(VERSION 3.17.3)

foreach(var TSLANG SOURCE WORK_DIR EXPECTED)
    if(NOT DEFINED ${var})
        message(FATAL_ERROR "${var} is required")
    endif()
endforeach()

file(REMOVE_RECURSE "${WORK_DIR}")
file(MAKE_DIRECTORY "${WORK_DIR}")

execute_process(COMMAND "${TSLANG}" --emit=jit -mm=none --no-default-lib "--jit-cache-dir=${WORK_DIR}" "${SOURCE}"
    OUTPUT_VARIABLE out
    ERROR_VARIABLE err
    RESULT_VARIABLE status)
string(STRIP "${out}" out)
if(NOT status EQUAL 0 OR NOT out STREQUAL EXPECTED)
    message(FATAL_ERROR "expected '${EXPECTED}': exit ${status}\n${out}\n${err}")
endif()

file(GLOB objects "${WORK_DIR}/*.o" "${WORK_DIR}/*.obj")
list(LENGTH objects count)
if(count LESS 2)
    message(FATAL_ERROR "expected an object for the program and one for each module it imports, found: ${objects}")
endif()
