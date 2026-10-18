# Borrow facts across modules (docs/superpowers/specs/2026-10-18-borrow-facts-across-modules-design.md):
# a library's __tsborrow_ facts read back from its object file and from a static library holding it,
# and nothing from an import library.

cmake_minimum_required(VERSION 3.17.3)

foreach(var TSLANG SOURCE_DIR WORK_DIR AR MM LLVM_LIB TSLANG_LIB)
    if(NOT DEFINED ${var})
        message(FATAL_ERROR "${var} is required")
    endif()
endforeach()

set(ENV{GC_LIB_PATH} "")
set(ENV{TSLANG_LIB_PATH} "")

# run(<dir> <what> <expect> <out-var> <command...>) - <expect> is "ok" or a regex the output must match
function(run dir what expect out_var)
    execute_process(COMMAND ${ARGN}
        WORKING_DIRECTORY "${dir}"
        OUTPUT_VARIABLE out
        ERROR_VARIABLE err
        RESULT_VARIABLE status)
    if("${out}${err}" MATCHES "Stack dump|Assertion failed")
        message(FATAL_ERROR "${what}: crashed\n${out}\n${err}")
    endif()
    if(NOT status EQUAL 0)
        message(FATAL_ERROR "${what}: exit ${status}\n${out}\n${err}")
    endif()
    if(NOT expect STREQUAL "ok" AND NOT "${out}${err}" MATCHES "${expect}")
        message(FATAL_ERROR "${what}: expected '${expect}'\n${out}\n${err}")
    endif()
    set(${out_var} "${out}${err}" PARENT_SCOPE)
endfunction()

if(MM STREQUAL "own")
    set(lib_source facts_lib_own)
    set(listed "Geo[.]len 0\n.*sumOf 0\n")
else()
    set(lib_source facts_lib)
    set(listed "Geo[.]len 0\n.*mixed 0\n.*sumOf 0\n")
endif()

set(dir "${WORK_DIR}/${MM}")
file(REMOVE_RECURSE "${dir}")
file(MAKE_DIRECTORY "${dir}")

if(WIN32)
    set(obj facts_lib.obj)
    set(archive facts_lib.lib)
else()
    set(obj facts_lib.o)
    set(archive libfacts_lib.a)
endif()

run("${dir}" "--emit=obj ${lib_source}" ok out
    "${TSLANG}" --emit=obj --no-default-lib -mm=${MM} "${SOURCE_DIR}/${lib_source}.ts" -o ${obj})
if(WIN32)
    run("${dir}" "archive" ok out "${AR}" /NOLOGO /OUT:${archive} ${obj})
else()
    run("${dir}" "archive" ok out "${AR}" rcs ${archive} ${obj})
endif()

run("${dir}" "facts of the object" "__tsborrow_${lib_source}_[0-9]+\n[^\n]*${listed}" out
    "${TSLANG}" --print-borrow-facts=${obj})
run("${dir}" "facts of the static library" "__tsborrow_${lib_source}_[0-9]+\n[^\n]*${listed}" out
    "${TSLANG}" --print-borrow-facts=${archive})

# A DLL's import library: import members and descriptor objects, never facts. (A gc DLL needs the
# shared collector, which these tests are not given.)
if(WIN32 AND NOT MM STREQUAL "gc")
    file(MAKE_DIRECTORY "${dir}/dll")
    # self-contained: the test libraries call a function they only declare
    file(WRITE "${dir}/dll/facts_dll.ts" "export function sumOf(a: number[]): number { return a.length; }\n")
    run("${dir}/dll" "--emit=dll facts_dll" ok out
        "${TSLANG}" --emit=dll --no-default-lib -mm=${MM}
        "--llvm-lib-path=${LLVM_LIB}" "--tslang-lib-path=${TSLANG_LIB}"
        facts_dll.ts -o facts_lib.dll)
    run("${dir}/dll" "--emit=obj facts_dll" ok out
        "${TSLANG}" --emit=obj --no-default-lib -mm=${MM} facts_dll.ts -o facts_dll.obj)
    run("${dir}/dll" "facts of the DLL's object" "sumOf 0" out
        "${TSLANG}" --print-borrow-facts=facts_dll.obj)
    run("${dir}/dll" "facts of the import library" ok out
        "${TSLANG}" --print-borrow-facts=facts_lib.lib)
    if(NOT out STREQUAL "")
        message(FATAL_ERROR "facts of the import library: expected none\n${out}")
    endif()
endif()

# no such file: no facts, no error
run("${dir}" "facts of a missing file" ok out "${TSLANG}" --print-borrow-facts=missing.lib)
if(NOT out STREQUAL "")
    message(FATAL_ERROR "facts of a missing file: expected none\n${out}")
endif()
