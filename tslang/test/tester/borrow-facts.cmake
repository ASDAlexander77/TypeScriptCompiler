# Borrow facts across modules (docs/superpowers/specs/2026-10-18-borrow-facts-across-modules-design.md):
# a library's __tsborrow_ facts read back from its object file and from a static library holding it,
# and nothing from an import library; then a program linked against that static library.

cmake_minimum_required(VERSION 3.17.3)

foreach(var TSLANG SOURCE_DIR WORK_DIR AR MM GC_LIB LLVM_LIB TSLANG_LIB)
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
    # what the JIT reads from the default library's DLL: its exports, from the file
    run("${dir}/dll" "facts of the DLL" "__tsborrow_facts_dll_[0-9]+\nsumOf 0\n" out
        "${TSLANG}" --print-borrow-facts=facts_lib.dll)
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

# A program linked statically against the library reads its facts: an array passed where the library
# listed the parameter goes on the stack, every other stays on the heap. -mm=own refuses the keeping
# functions, so it has a program of its own.
if(MM STREQUAL "own")
    set(use_source facts_use_own)
    set(decisions
        "facts_use_own[.]ts:13:15: remark: stack-promotion: stack"
        "facts_use_own[.]ts:18:15: remark: stack-promotion: heap: unknown-callee"
        "facts_use_own[.]ts:23:15: remark: stack-promotion: stack")
else()
    set(use_source facts_use)
    set(decisions
        "facts_use[.]ts:20:15: remark: stack-promotion: stack"
        "facts_use[.]ts:25:15: remark: stack-promotion: heap: unknown-callee"
        "facts_use[.]ts:30:15: remark: stack-promotion: heap: unknown-callee"
        "facts_use[.]ts:35:15: remark: stack-promotion: heap: unknown-callee"
        "facts_use[.]ts:42:15: remark: stack-promotion: stack"
        "facts_use[.]ts:43:15: remark: stack-promotion: heap: unknown-callee"
        "facts_use[.]ts:48:15: remark: stack-promotion: heap: unknown-callee"
        "facts_use[.]ts:53:15: remark: stack-promotion: stack")
endif()

set(link_libs "--gc-lib-path=${GC_LIB}" "--llvm-lib-path=${LLVM_LIB}" "--tslang-lib-path=${TSLANG_LIB}")

run("${dir}" "--emit=exe ${use_source} --lib=facts_lib" ok decided
    "${TSLANG}" --emit=exe --no-default-lib -mm=${MM} --stack-promotion-report ${link_libs} --lib=facts_lib
    "${SOURCE_DIR}/${use_source}.ts" -o use.exe)
foreach(decision ${decisions})
    if(NOT decided MATCHES "${decision}")
        message(FATAL_ERROR "--lib=facts_lib: expected '${decision}'\n${decided}")
    endif()
endforeach()

# a declaration bound to another symbol (@linkname) takes no fact by its own name
if(NOT MM STREQUAL "own")
    run("${dir}" "--emit=exe facts_linkname --lib=facts_lib" ok decided
        "${TSLANG}" --emit=exe --no-default-lib -mm=${MM} --stack-promotion-report ${link_libs} --lib=facts_lib
        "${SOURCE_DIR}/facts_linkname.ts" -o linkname.exe)
    if(NOT decided MATCHES "facts_linkname[.]ts:12:15: remark: stack-promotion: heap: unknown-callee")
        message(FATAL_ERROR "@linkname declaration: expected heap\n${decided}")
    endif()
endif()

# the same program, nothing promoted: the same output
run("${dir}" "--emit=exe ${use_source} --no-stack-promotion" ok out
    "${TSLANG}" --emit=exe --no-default-lib -mm=${MM} --no-stack-promotion ${link_libs} --lib=facts_lib
    "${SOURCE_DIR}/${use_source}.ts" -o use_heap.exe)
run("${dir}" "run use" ok promoted_output "${dir}/use.exe")
run("${dir}" "run use_heap" ok heap_output "${dir}/use_heap.exe")
if(NOT promoted_output STREQUAL heap_output)
    message(FATAL_ERROR "promoted program printed\n${promoted_output}\nunpromoted\n${heap_output}")
endif()

# --emit=obj does not link, so it reads nothing
run("${dir}" "--emit=obj ${use_source}" ok decided
    "${TSLANG}" --emit=obj --no-default-lib -mm=${MM} --stack-promotion-report --lib=facts_lib
    "${SOURCE_DIR}/${use_source}.ts" -o use_obj.obj)
if(decided MATCHES "remark: stack-promotion: stack")
    message(FATAL_ERROR "--emit=obj read facts\n${decided}")
endif()

# a library built before facts existed: links, no warning, nothing promoted through it
file(MAKE_DIRECTORY "${dir}/old")
run("${dir}/old" "--emit=obj ${lib_source} --no-borrow-facts-export" ok out
    "${TSLANG}" --emit=obj --no-default-lib -mm=${MM} --no-borrow-facts-export "${SOURCE_DIR}/${lib_source}.ts" -o ${obj})
if(WIN32)
    run("${dir}/old" "archive" ok out "${AR}" /NOLOGO /OUT:${archive} ${obj})
else()
    run("${dir}/old" "archive" ok out "${AR}" rcs ${archive} ${obj})
endif()
run("${dir}/old" "--emit=exe ${use_source} against the old library" ok decided
    "${TSLANG}" --emit=exe --no-default-lib -mm=${MM} --stack-promotion-report ${link_libs} --lib=facts_lib
    "${SOURCE_DIR}/${use_source}.ts" -o use.exe)
if(decided MATCHES "remark: stack-promotion: stack|warning")
    message(FATAL_ERROR "against the old library\n${decided}")
endif()
