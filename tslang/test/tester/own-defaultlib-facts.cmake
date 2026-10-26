# -mm=own with the real default library: a field lent to a library method that frees nothing.
# The library's own facts say so - read from its static library for an exe, and from the DLL the
# JIT loads under the JIT (borrowfacts.cpp, readOwnFacts). Without them every library call may
# destroy anything, and the program does not compile.
#
#   MODE=jit     run it under the JIT
#   MODE=compile build an exe and run it
#
# Every test-runner test passes --no-default-lib, so this drives tslang itself. With no default
# library built the test prints SKIPPED and CTest reports it skipped.

cmake_minimum_required(VERSION 3.17.3)

foreach(var MODE TSLANG TSLANG_BIN TEST WORK_DIR GC_LIB LLVM_LIB TSLANG_LIB OPT DEFAULT_LIB_CANDIDATES)
    if(NOT DEFINED ${var})
        message(FATAL_ERROR "${var} is required")
    endif()
endforeach()

if(WIN32)
    set(exe_suffix ".exe")
    set(runtime "${TSLANG_BIN}/TypeScriptRuntime.dll")
else()
    set(exe_suffix "")
    set(runtime "${TSLANG_LIB}/libTypeScriptRuntime.so")
endif()

# the folder of the library the run uses: the DLL for the JIT, the static library for an exe
if(MODE STREQUAL "jit")
    set(kind "dll")
else()
    set(kind "lib")
endif()

execute_process(COMMAND "${TSLANG}" --print-default-lib-dir=${kind} ${OPT} -mm=own
    OUTPUT_VARIABLE default_lib_subdir
    OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE status)
if(NOT status EQUAL 0 OR default_lib_subdir STREQUAL "")
    message(FATAL_ERROR "tslang --print-default-lib-dir=${kind} ${OPT} -mm=own: exit ${status}")
endif()

set(default_lib "")
foreach(candidate "$ENV{DEFAULT_LIB_PATH}" ${DEFAULT_LIB_CANDIDATES})
    if(NOT candidate STREQUAL "" AND IS_DIRECTORY "${candidate}/${default_lib_subdir}")
        set(default_lib "${candidate}")
        break()
    endif()
endforeach()

if(default_lib STREQUAL "")
    if("$ENV{TSLANG_REQUIRE_DEFAULT_LIB}" STREQUAL "1")
        message(FATAL_ERROR "no own default library built in ${default_lib_subdir} under DEFAULT_LIB_PATH='$ENV{DEFAULT_LIB_PATH}'")
    endif()
    message("SKIPPED: no own default library built in ${default_lib_subdir} (set DEFAULT_LIB_PATH to the folder holding 'defaultlib')")
    return()
endif()

set(ENV{DEFAULT_LIB_PATH} "")
file(REMOVE_RECURSE "${WORK_DIR}")
file(MAKE_DIRECTORY "${WORK_DIR}")

set(common ${OPT} -mm=own "--default-lib-path=${default_lib}" "--llvm-lib-path=${LLVM_LIB}"
    "--tslang-lib-path=${TSLANG_LIB}" "--gc-lib-path=${GC_LIB}")

if(WIN32)
    set(run_env "PATH=${default_lib}/${default_lib_subdir};$ENV{PATH}")
else()
    set(run_env "LD_LIBRARY_PATH=${default_lib}/${default_lib_subdir}:$ENV{LD_LIBRARY_PATH}")
endif()

if(MODE STREQUAL "jit")
    set(command "${TSLANG}" --emit=jit ${common} "--shared-libs=${runtime}" "${TEST}")
else()
    execute_process(COMMAND "${TSLANG}" --emit=exe ${common} "${TEST}" -o "${WORK_DIR}/test${exe_suffix}"
        OUTPUT_VARIABLE out ERROR_VARIABLE err RESULT_VARIABLE status)
    if(NOT status EQUAL 0)
        message(FATAL_ERROR "--emit=exe -mm=own: exit ${status}\n${out}\n${err}")
    endif()

    set(command "${WORK_DIR}/test${exe_suffix}")
endif()

execute_process(COMMAND ${CMAKE_COMMAND} -E env "${run_env}" ${command}
    WORKING_DIRECTORY "${WORK_DIR}"
    OUTPUT_VARIABLE out ERROR_VARIABLE err RESULT_VARIABLE status)
if(NOT status EQUAL 0 OR NOT out MATCHES "done\.")
    message(FATAL_ERROR "${MODE} -mm=own: exit ${status}\n${out}\n${err}")
endif()

message(STATUS "${MODE} -mm=own: done.")
