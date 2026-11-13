# Builds a program that throws what nothing catches with `tslang --emit=exe` and runs it. It must end with
# exit code 1, its output so far ("before") must not be lost, and stderr must say what was thrown, as the
# JIT says it (EXPECT). Before TypeScriptAsyncRuntime/UncaughtException.cpp an executable called abort():
# exit code 3, or 0xC0000409 with --opt, and not a word - not even the "before" still in stdout's buffer.

cmake_minimum_required(VERSION 3.17.3)

foreach(var TSLANG SOURCE MODEL EXPECT WORK_DIR GC_LIB LLVM_LIB TSLANG_LIB OPT)
    if(NOT DEFINED ${var})
        message(FATAL_ERROR "${var} is required")
    endif()
endforeach()

# only the paths given here
set(ENV{TSLANG_LIB_PATH} "")
set(ENV{GC_LIB_PATH} "")
set(ENV{LLVM_LIB_PATH} "")
set(ENV{DEFAULT_LIB_PATH} "")

file(REMOVE_RECURSE "${WORK_DIR}")
file(MAKE_DIRECTORY "${WORK_DIR}")

execute_process(
    COMMAND "${TSLANG}" --emit=exe -mm=${MODEL} --no-default-lib ${OPT}
            "--llvm-lib-path=${LLVM_LIB}" "--tslang-lib-path=${TSLANG_LIB}" "--gc-lib-path=${GC_LIB}"
            "${SOURCE}" -o=program.exe
    WORKING_DIRECTORY "${WORK_DIR}"
    OUTPUT_VARIABLE out
    ERROR_VARIABLE err
    RESULT_VARIABLE status)
if(NOT status EQUAL 0)
    message(FATAL_ERROR "--emit=exe -mm=${MODEL}: exit ${status}: ${out}${err}")
endif()

execute_process(
    COMMAND "${WORK_DIR}/program.exe"
    WORKING_DIRECTORY "${WORK_DIR}"
    OUTPUT_VARIABLE out
    ERROR_VARIABLE err
    RESULT_VARIABLE status)
if(NOT status EQUAL 1 OR NOT out MATCHES "^before\r?\n" OR NOT err MATCHES "${EXPECT}")
    message(FATAL_ERROR "the program: exit ${status} (expected 1), stdout '${out}' (expected 'before'), "
                        "stderr '${err}' (expected '${EXPECT}')")
endif()

# and, when given, the frames it was thrown from (TypeScript/UncaughtStackTrace.h)
if(DEFINED EXPECT_TRACE AND NOT err MATCHES "${EXPECT_TRACE}")
    message(FATAL_ERROR "the program's stack trace: stderr '${err}' (expected '${EXPECT_TRACE}')")
endif()

message(STATUS "${err}")
