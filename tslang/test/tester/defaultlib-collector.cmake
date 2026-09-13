# One garbage collector per process, with the default library in the process (-mm=gc).
#
# Every test-runner test passes --no-default-lib, so the binary that is in almost every real
# process - the default library - was never part of one. A default library whose DLL links its own
# collector frees the strings it hands out: 2000 of 2000 under the JIT, 1984 of 2000 for an exe with
# a user shared library (docs/single-gc-collector-design.md). This drives tslang with the real
# default library instead:
#
#   MODE=jit     the JIT, the program alone and importing a shared library
#   MODE=compile an exe alone, and an exe importing a shared library that links the default-library DLL
#
# The default library is a separate build (TypeScriptCompilerDefaultLib). With none found the test
# prints SKIPPED and CTest reports it skipped; the release workflows run it after building one.

cmake_minimum_required(VERSION 3.17.3)

foreach(var MODE TSLANG TSLANG_BIN TESTS_DIR WORK_DIR GC_LIB LLVM_LIB TSLANG_LIB OPT DEFAULT_LIB_CANDIDATES)
    if(NOT DEFINED ${var})
        message(FATAL_ERROR "${var} is required")
    endif()
endforeach()

if(WIN32)
    set(exe_suffix ".exe")
    set(dll_prefix "")
    set(dll_suffix ".dll")
    set(runtime "${TSLANG_BIN}/TypeScriptRuntime.dll")
    set(pic "")
else()
    set(exe_suffix "")
    set(dll_prefix "lib")
    set(dll_suffix ".so")
    set(runtime "${TSLANG_LIB}/libTypeScriptRuntime.so")
    set(pic "-relocation-model=pic")
endif()

if("--di" IN_LIST OPT)
    set(build "debug")
else()
    set(build "release")
endif()

# DEFAULT_LIB_PATH at run time wins, so a workflow can point at the library it just built
set(default_lib "")
foreach(candidate "$ENV{DEFAULT_LIB_PATH}" ${DEFAULT_LIB_CANDIDATES})
    if(NOT candidate STREQUAL "" AND IS_DIRECTORY "${candidate}/defaultlib/dll/${build}/gc")
        set(default_lib "${candidate}")
        break()
    endif()
endforeach()

if(default_lib STREQUAL "")
    # a workflow that has just built the library sets this, so a wrong path fails instead of skipping
    if("$ENV{TSLANG_REQUIRE_DEFAULT_LIB}" STREQUAL "1")
        message(FATAL_ERROR "no default library built for ${build}/gc under DEFAULT_LIB_PATH='$ENV{DEFAULT_LIB_PATH}'")
    endif()
    message("SKIPPED: no default library built for ${build}/gc (set DEFAULT_LIB_PATH to the folder holding 'defaultlib')")
    return()
endif()

set(default_lib_dll_dir "${default_lib}/defaultlib/dll/${build}/gc")
message(STATUS "default library: ${default_lib}")

set(ENV{GC_SHARED_LIB_PATH} "")
set(ENV{GC_LIB_PATH} "")
set(ENV{DEFAULT_LIB_PATH} "")

file(REMOVE_RECURSE "${WORK_DIR}")
file(MAKE_DIRECTORY "${WORK_DIR}/alone")

set(gc_opts "--gc-lib-path=${GC_LIB}")
if(DEFINED GC_SHARED_LIB)
    list(APPEND gc_opts "--gc-shared-lib-path=${GC_SHARED_LIB}")
endif()

set(common ${OPT} ${pic} "--default-lib-path=${default_lib}" "--llvm-lib-path=${LLVM_LIB}" "--tslang-lib-path=${TSLANG_LIB}" ${gc_opts})

# the loader has to find the default-library shared object (Linux) and gc.dll / the library (both)
if(WIN32)
    set(run_env "PATH=${WORK_DIR};$ENV{PATH}")
else()
    set(run_env "LD_LIBRARY_PATH=${WORK_DIR}:${default_lib_dll_dir}:$ENV{LD_LIBRARY_PATH}")
endif()

# run(<what> <dir> <command...>) - fails unless it exits 0 and prints "bad: 0" and "done."
function(run what dir)
    execute_process(COMMAND ${CMAKE_COMMAND} -E env "${run_env}" ${ARGN}
        WORKING_DIRECTORY "${dir}"
        OUTPUT_VARIABLE out
        ERROR_VARIABLE err
        RESULT_VARIABLE status)
    if(NOT status EQUAL 0)
        message(FATAL_ERROR "${what}: exit ${status}\n${out}\n${err}")
    endif()
endfunction()

function(run_program what dir)
    execute_process(COMMAND ${CMAKE_COMMAND} -E env "${run_env}" ${ARGN}
        WORKING_DIRECTORY "${dir}"
        OUTPUT_VARIABLE out
        ERROR_VARIABLE err
        RESULT_VARIABLE status)
    if(NOT status EQUAL 0 OR NOT out MATCHES "bad: 0" OR NOT out MATCHES "done\\.")
        message(FATAL_ERROR "${what}: exit ${status}\n${out}\n${err}")
    endif()
    message(STATUS "${what}: bad: 0")
endfunction()

set(alone "${TESTS_DIR}/defaultlib_collector.ts")
set(library "${TESTS_DIR}/export_defaultlib_collector.ts")
set(program "${TESTS_DIR}/import_defaultlib_collector.ts")
set(library_file "${dll_prefix}export_defaultlib_collector${dll_suffix}")

# The shared library links the default-library DLL; both modes import it.
run("--emit=dll with the default library" "${WORK_DIR}"
    "${TSLANG}" --emit=dll ${common} "${library}" -o "${library_file}")
file(COPY "${default_lib_dll_dir}/${dll_prefix}TypeScriptDefaultLib${dll_suffix}" DESTINATION "${WORK_DIR}")

if(MODE STREQUAL "jit")
    run_program("JIT, default library" "${WORK_DIR}/alone"
        "${TSLANG}" --emit=jit ${common} "--shared-libs=${runtime}" "${alone}")
    run_program("JIT importing a shared library, default library" "${WORK_DIR}"
        "${TSLANG}" --emit=jit ${common} "--shared-libs=${runtime}" "${program}")
elseif(MODE STREQUAL "compile")
    run("--emit=exe alone, default library" "${WORK_DIR}/alone"
        "${TSLANG}" --emit=exe ${common} "${alone}" -o "alone${exe_suffix}")
    run_program("exe, default library" "${WORK_DIR}/alone" "${WORK_DIR}/alone/alone${exe_suffix}")

    run("--emit=exe importing a shared library, default library" "${WORK_DIR}"
        "${TSLANG}" --emit=exe ${common} "${program}" -o "main${exe_suffix}")
    run_program("exe importing a shared library, default library" "${WORK_DIR}" "${WORK_DIR}/main${exe_suffix}")
else()
    message(FATAL_ERROR "MODE must be jit or compile, not '${MODE}'")
endif()
