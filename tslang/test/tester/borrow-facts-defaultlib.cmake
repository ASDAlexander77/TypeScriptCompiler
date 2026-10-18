# Borrow facts from the real default library (docs/superpowers/specs/2026-10-18-borrow-facts-across-modules-design.md):
# `Math.max(a, b, c)` builds its rest array at the call, and the default library lists that parameter
# as only read, so the array goes on the stack - read from the static library an exe links, and from
# the DLL the JIT loads. A JIT cache entry depends on that DLL.
#
# The default library is a separate build (TypeScriptCompilerDefaultLib), checked per model: one
# built before borrow facts existed publishes none and is passed over. With none to check the test
# prints SKIPPED.

cmake_minimum_required(VERSION 3.17.3)

foreach(var TSLANG SOURCE_DIR WORK_DIR GC_LIB LLVM_LIB TSLANG_LIB DEFAULT_LIB_CANDIDATES)
    if(NOT DEFINED ${var})
        message(FATAL_ERROR "${var} is required")
    endif()
endforeach()

set(ENV{GC_LIB_PATH} "")
set(ENV{TSLANG_LIB_PATH} "")

set(default_lib "")
foreach(candidate "$ENV{DEFAULT_LIB_PATH}" ${DEFAULT_LIB_CANDIDATES})
    if(NOT candidate STREQUAL "" AND IS_DIRECTORY "${candidate}/defaultlib")
        set(default_lib "${candidate}")
        break()
    endif()
endforeach()
set(ENV{DEFAULT_LIB_PATH} "")

if(default_lib STREQUAL "")
    message("SKIPPED: no default library built (set DEFAULT_LIB_PATH to the folder holding 'defaultlib')")
    return()
endif()

if(WIN32)
    set(static_name "TypeScriptDefaultLib.lib")
    set(shared_name "TypeScriptDefaultLib.dll")
    set(run_env "PATH=${GC_LIB}/../bin;$ENV{PATH}")
else()
    set(static_name "libTypeScriptDefaultLib.a")
    set(shared_name "libTypeScriptDefaultLib.so")
    set(run_env "LD_LIBRARY_PATH=$ENV{LD_LIBRARY_PATH}")
endif()

# run(<dir> <what> <out-var> <command...>) - fails unless it exits 0
function(run dir what out_var)
    execute_process(COMMAND ${CMAKE_COMMAND} -E env "${run_env}" ${ARGN}
        WORKING_DIRECTORY "${dir}"
        OUTPUT_VARIABLE out
        ERROR_VARIABLE err
        RESULT_VARIABLE status)
    if(NOT status EQUAL 0 OR "${out}${err}" MATCHES "Stack dump|Assertion failed")
        message(FATAL_ERROR "${what}: exit ${status}\n${out}\n${err}")
    endif()
    set(${out_var} "${out}${err}" PARENT_SCOPE)
endfunction()

set(source "${SOURCE_DIR}/facts_defaultlib.ts")
set(promoted "facts_defaultlib[.]ts:5:21: remark: stack-promotion: stack")
set(checked 0)

foreach(mm gc rc own none)
    execute_process(COMMAND "${TSLANG}" --print-default-lib-dir=lib -mm=${mm}
        OUTPUT_VARIABLE lib_subdir OUTPUT_STRIP_TRAILING_WHITESPACE)
    execute_process(COMMAND "${TSLANG}" --print-default-lib-dir=dll -mm=${mm}
        OUTPUT_VARIABLE dll_subdir OUTPUT_STRIP_TRAILING_WHITESPACE)
    set(static_lib "${default_lib}/${lib_subdir}/${static_name}")
    set(shared_lib "${default_lib}/${dll_subdir}/${shared_name}")
    if(NOT EXISTS "${static_lib}" OR NOT EXISTS "${shared_lib}")
        continue()
    endif()

    execute_process(COMMAND "${TSLANG}" --print-borrow-facts=${static_lib} OUTPUT_VARIABLE static_facts)
    if(NOT static_facts MATCHES "\nMath[.]max 0\n")
        message(STATUS "-mm=${mm}: the default library publishes no facts for Math.max (built before borrow facts)")
        continue()
    endif()

    set(dir "${WORK_DIR}/${mm}")
    file(REMOVE_RECURSE "${dir}")
    file(MAKE_DIRECTORY "${dir}")
    set(common -mm=${mm} "--default-lib-path=${default_lib}" --stack-promotion-report)

    run("${dir}" "-mm=${mm} --emit=exe" decided
        "${TSLANG}" --emit=exe ${common} "--gc-lib-path=${GC_LIB}" "--llvm-lib-path=${LLVM_LIB}" "--tslang-lib-path=${TSLANG_LIB}"
        "${source}" -o use.exe)
    if(NOT decided MATCHES "${promoted}")
        message(FATAL_ERROR "-mm=${mm} --emit=exe: expected '${promoted}'\n${decided}")
    endif()
    run("${dir}" "-mm=${mm} run" out "${dir}/use.exe")
    if(NOT out MATCHES "^6\n")
        message(FATAL_ERROR "-mm=${mm} run: expected 6\n${out}")
    endif()

    # the JIT reads the DLL it loads; its cache entry depends on it
    run("${dir}" "-mm=${mm} --emit=jit" decided
        "${TSLANG}" --emit=jit ${common} "--jit-cache-dir=${dir}/cache" "${source}")
    if(NOT decided MATCHES "${promoted}" OR NOT decided MATCHES "(^|\n)6\n")
        message(FATAL_ERROR "-mm=${mm} --emit=jit: expected '${promoted}' and 6\n${decided}")
    endif()
    file(GLOB manifests "${dir}/cache/*.deps")
    if(manifests STREQUAL "")
        message(FATAL_ERROR "-mm=${mm} --emit=jit: no cache manifest in ${dir}/cache")
    endif()
    file(READ "${manifests}" manifest)
    file(TO_NATIVE_PATH "${shared_lib}" shared_lib_native)
    string(FIND "${manifest}" "${shared_name}" at)
    if(at EQUAL -1)
        message(FATAL_ERROR "-mm=${mm} --emit=jit: the cache manifest does not name ${shared_lib}\n${manifest}")
    endif()

    math(EXPR checked "${checked} + 1")
endforeach()

if(checked EQUAL 0)
    message("SKIPPED: no default library under ${default_lib} publishes borrow facts (rebuild it)")
endif()
