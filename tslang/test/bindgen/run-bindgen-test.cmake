# tsbindgen end to end: generate bindings for a real C library and call it through them.
#
# fixture.c is compiled with the in-tree clang, to an object and to a shared library; tsbindgen
# turns fixture.h into fixture.ts; bindgen_test.ts is compiled against it and linked with the
# object (--emit=exe), then run under the JIT against the shared library (--emit=jit). Both must
# print expected.txt. MODE=namespace does the same through `--namespace Fx --strip-prefix fx_`,
# which only links because @linkname binds each renamed function to its C symbol.

cmake_minimum_required(VERSION 3.17.3)

foreach(var MODE TSLANG TSBINDGEN CLANG SOURCE_DIR WORK_DIR GC_LIB LLVM_LIB TSLANG_LIB TSLANG_BIN OPT)
    if(NOT DEFINED ${var})
        message(FATAL_ERROR "${var} is required")
    endif()
endforeach()

if(WIN32)
    set(exe_suffix ".exe")
    set(obj_suffix ".obj")
    set(fixture_library "${WORK_DIR}/fixture_c.dll")
    set(runtime "${TSLANG_BIN}/TypeScriptRuntime.dll")
    set(pic "")
    set(clang_pic "")
    set(run_env "PATH=${WORK_DIR};$ENV{PATH}")
else()
    set(exe_suffix "")
    set(obj_suffix ".o")
    set(fixture_library "${WORK_DIR}/libfixture_c.so")
    set(runtime "${TSLANG_LIB}/libTypeScriptRuntime.so")
    set(pic "-relocation-model=pic")
    set(clang_pic "-fPIC")
    set(run_env "LD_LIBRARY_PATH=${WORK_DIR}:$ENV{LD_LIBRARY_PATH}")
endif()

# the C library's stem differs from the bindings' on purpose: `import "./fixture"` next to a
# fixture.dll would load that as a tslang library
if(MODE STREQUAL "namespace")
    set(bindings "fixture_ns.ts")
    set(program "bindgen_ns_test.ts")
    set(bindgen_args --namespace Fx --strip-prefix fx_)
else()
    set(bindings "fixture.ts")
    set(program "bindgen_test.ts")
    set(bindgen_args "")
endif()

file(REMOVE_RECURSE "${WORK_DIR}")
file(MAKE_DIRECTORY "${WORK_DIR}")
file(COPY "${SOURCE_DIR}/fixture.h" "${SOURCE_DIR}/fixture.c" "${SOURCE_DIR}/${program}" DESTINATION "${WORK_DIR}")
file(READ "${SOURCE_DIR}/expected.txt" expected)
string(REPLACE "\r\n" "\n" expected "${expected}")

set(common --no-default-lib ${OPT} ${pic} "--gc-lib-path=${GC_LIB}" "--llvm-lib-path=${LLVM_LIB}"
    "--tslang-lib-path=${TSLANG_LIB}")

# run(<what> <command...>) - fails unless it exits 0; leaves stdout in run_output
function(run what)
    execute_process(COMMAND ${CMAKE_COMMAND} -E env "${run_env}" ${ARGN}
        WORKING_DIRECTORY "${WORK_DIR}"
        OUTPUT_VARIABLE out
        ERROR_VARIABLE err
        RESULT_VARIABLE status)
    if(NOT status EQUAL 0)
        message(FATAL_ERROR "${what}: exit ${status}\n${out}\n${err}")
    endif()
    string(REPLACE "\r\n" "\n" out "${out}")
    set(run_output "${out}" PARENT_SCOPE)
endfunction()

function(expect_output what)
    if(NOT run_output STREQUAL expected)
        message(FATAL_ERROR "${what} printed:\n${run_output}\nexpected:\n${expected}")
    endif()
    message(STATUS "${what}: as expected")
endfunction()

# -O2 as a real library would be built: at -O0 clang re-extends a narrow parameter inside the callee,
# which would hide a caller that does not extend it (the `narrow` line, on the SysV ABI)
run("clang -c fixture.c" "${CLANG}" -O2 -c ${clang_pic} fixture.c -o "fixture${obj_suffix}")
run("clang -shared fixture.c" "${CLANG}" -O2 -shared ${clang_pic} fixture.c -o "${fixture_library}")
run("tsbindgen" "${TSBINDGEN}" fixture.h ${bindgen_args} -o "${bindings}")

run("--emit=exe" "${TSLANG}" --emit=exe ${common} "${program}" "--obj=fixture${obj_suffix}" -o "bindgen_test${exe_suffix}")
run("the program" "${WORK_DIR}/bindgen_test${exe_suffix}")
expect_output("--emit=exe")

run("--emit=jit" "${TSLANG}" --emit=jit ${common} "--shared-libs=${runtime}" "--shared-libs=${fixture_library}" "${program}")
expect_output("--emit=jit")
