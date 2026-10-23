# -mm=own across a DLL boundary: a function another module defines that says `@record` returns a box
# its caller owns (own spec 32), which the importer learns only from the declaration it re-parses.
# test-runner's -shared mode runs the pair; this checks what a run cannot see - that the importer
# releases the box. Without `@record` it took the result for a plain borrow and never freed it.

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

execute_process(COMMAND "${TSLANG}" --emit=dll -mm=own --no-default-lib
                        "--llvm-lib-path=${LLVM_LIB}" "--tslang-lib-path=${TSLANG_LIB}"
                        "${SOURCE_DIR}/export_own_record.ts" -o export_own_record.dll
    WORKING_DIRECTORY "${WORK_DIR}"
    OUTPUT_VARIABLE out
    ERROR_VARIABLE err
    RESULT_VARIABLE status)
if(NOT status EQUAL 0)
    message(FATAL_ERROR "--emit=dll export_own_record: exit ${status}\n${out}\n${err}")
endif()

# `import './export_own_record'` finds the DLL in the working directory before the source
execute_process(COMMAND "${TSLANG}" --emit=mlir-affine -mm=own --no-default-lib "${SOURCE_DIR}/import_own_record.ts"
    WORKING_DIRECTORY "${WORK_DIR}"
    OUTPUT_VARIABLE out
    ERROR_VARIABLE err
    RESULT_VARIABLE status)
if(NOT status EQUAL 0 OR "${out}${err}" MATCHES "error:|Stack dump")
    message(FATAL_ERROR "import_own_record: exit ${status}\n${out}\n${err}")
endif()

# one release of a Pair box in each loop
string(REGEX MATCHALL "\"ts.Release\"[(]%[0-9]+[)] : [(]!ts.iface<@Pair>[)]" released "${out}${err}")
list(LENGTH released count)
if(NOT count EQUAL 2)
    message(FATAL_ERROR "import_own_record: ${count} releases of a Pair box, expected 2\n${out}${err}")
endif()
