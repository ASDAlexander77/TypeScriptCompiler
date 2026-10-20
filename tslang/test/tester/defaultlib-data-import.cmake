# A shared library that reads data defined in the default library (-mm=rc).
#
# `tslang --emit=dll` links the default library's DLL, and a DLL exports data only as
# `__imp_<name>`: a plain external global does not reach it. So a library whose Map read
# PrimeHelpers.hashCollisionThreshold failed to link ("unresolved external symbol
# PrimeHelpers.hashCollisionThreshold referenced in function Map<!ts.string,!ts.number>.tryInsert"),
# while --emit=exe (the static default library) and --emit=jit (bound from the DLL's exports) did
# not show it. Every test-runner test passes --no-default-lib, so none had the default library's
# data in a DLL. This links export_defaultlib_data.ts as a DLL with the real default library and
# runs a program that imports it.
#
# The default library is a separate build (TypeScriptCompilerDefaultLib). With none found the test
# prints SKIPPED and CTest reports it skipped.

cmake_minimum_required(VERSION 3.17.3)

foreach(var TSLANG TESTS_DIR WORK_DIR LLVM_LIB TSLANG_LIB OPT DEFAULT_LIB_CANDIDATES)
    if(NOT DEFINED ${var})
        message(FATAL_ERROR "${var} is required")
    endif()
endforeach()

if(WIN32)
    set(exe_suffix ".exe")
    set(dll_prefix "")
    set(dll_suffix ".dll")
    set(pic "")
else()
    set(exe_suffix "")
    set(dll_prefix "lib")
    set(dll_suffix ".so")
    set(pic "-relocation-model=pic")
endif()

# The folder of the shared default library for this target, build and model, from tslang itself
execute_process(COMMAND "${TSLANG}" --print-default-lib-dir=dll ${OPT} -mm=rc
    OUTPUT_VARIABLE default_lib_dll_subdir
    OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE status)
if(NOT status EQUAL 0 OR default_lib_dll_subdir STREQUAL "")
    message(FATAL_ERROR "tslang --print-default-lib-dir=dll ${OPT} -mm=rc: exit ${status}")
endif()

# DEFAULT_LIB_PATH at run time wins, so a workflow can point at the library it just built
set(default_lib "")
foreach(candidate "$ENV{DEFAULT_LIB_PATH}" ${DEFAULT_LIB_CANDIDATES})
    if(NOT candidate STREQUAL "" AND IS_DIRECTORY "${candidate}/${default_lib_dll_subdir}")
        set(default_lib "${candidate}")
        break()
    endif()
endforeach()

if(default_lib STREQUAL "")
    if("$ENV{TSLANG_REQUIRE_DEFAULT_LIB}" STREQUAL "1")
        message(FATAL_ERROR "no default library built in ${default_lib_dll_subdir} under DEFAULT_LIB_PATH='$ENV{DEFAULT_LIB_PATH}'")
    endif()
    message("SKIPPED: no default library built in ${default_lib_dll_subdir} (set DEFAULT_LIB_PATH to the folder holding 'defaultlib')")
    return()
endif()

set(default_lib_dll_dir "${default_lib}/${default_lib_dll_subdir}")
message(STATUS "default library: ${default_lib}")

set(ENV{GC_LIB_PATH} "")
set(ENV{DEFAULT_LIB_PATH} "")

file(REMOVE_RECURSE "${WORK_DIR}")
file(MAKE_DIRECTORY "${WORK_DIR}")

set(common -mm=rc ${OPT} ${pic} "--default-lib-path=${default_lib}" "--llvm-lib-path=${LLVM_LIB}" "--tslang-lib-path=${TSLANG_LIB}")

# the loader has to find the library and the default library's DLL
if(WIN32)
    set(run_env "PATH=${WORK_DIR};$ENV{PATH}")
else()
    set(run_env "LD_LIBRARY_PATH=${WORK_DIR}:${default_lib_dll_dir}:$ENV{LD_LIBRARY_PATH}")
endif()

# run(<what> <command...>) - fails unless it exits 0
function(run what)
    execute_process(COMMAND ${CMAKE_COMMAND} -E env "${run_env}" ${ARGN}
        WORKING_DIRECTORY "${WORK_DIR}"
        OUTPUT_VARIABLE out
        ERROR_VARIABLE err
        RESULT_VARIABLE status)
    if(NOT status EQUAL 0)
        message(FATAL_ERROR "${what}: exit ${status}\n${out}\n${err}")
    endif()
    set(out "${out}" PARENT_SCOPE)
endfunction()

run("--emit=dll reading the default library's data"
    "${TSLANG}" --emit=dll ${common} "${TESTS_DIR}/export_defaultlib_data.ts"
    -o "${dll_prefix}export_defaultlib_data${dll_suffix}")
file(COPY "${default_lib_dll_dir}/${dll_prefix}TypeScriptDefaultLib${dll_suffix}" DESTINATION "${WORK_DIR}")

run("--emit=exe importing it"
    "${TSLANG}" --emit=exe ${common} "${TESTS_DIR}/import_defaultlib_data.ts" -o "main${exe_suffix}")

run("the program" "${WORK_DIR}/main${exe_suffix}")
if(NOT out MATCHES "size: 300" OR NOT out MATCHES "done\\.")
    message(FATAL_ERROR "the program:\n${out}")
endif()

message(STATUS "a shared library reads the default library's data from its DLL")
