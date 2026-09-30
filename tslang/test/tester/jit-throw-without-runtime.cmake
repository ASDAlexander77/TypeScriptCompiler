# Runs a program that throws under the JIT with NO --shared-libs, under every model but gc.
#
# `test-runner` always passes `--shared-libs=TypeScriptRuntime.dll`, and that DLL exports type_info's
# vtable (??_7type_info@@6B@), which every type descriptor a throw emits points at. A JIT run loads
# the DLL by itself only under gc; under rc, none and own without it, nothing in the process
# exported the vtable, and any program with a `throw` failed to materialize: "Symbols not found:
# [ ??_7type_info@@6B@ ]". The JIT now binds it itself (see jitTypeInfoVftable in jit.cpp).

if(NOT DEFINED TSLANG OR NOT DEFINED TESTS_DIR)
    message(FATAL_ERROR "TSLANG and TESTS_DIR are both required")
endif()

set(name 00jit_throw_without_runtime.ts)

set(failures "")
foreach(model rc none own)
    execute_process(
        COMMAND "${TSLANG}" --emit=jit -mm=${model} --no-default-lib "${TESTS_DIR}/${name}"
        OUTPUT_VARIABLE output
        ERROR_VARIABLE diagnostics
        RESULT_VARIABLE status)

    if(NOT status EQUAL 0 OR NOT output MATCHES "ALL DONE")
        # the first error, not a warning printed ahead of it
        if(diagnostics MATCHES "[^\n]*error:[^\n]*")
            set(first_line "${CMAKE_MATCH_0}")
        else()
            string(REGEX REPLACE "\n.*" "" first_line "${diagnostics}")
        endif()
        list(APPEND failures "  -mm=${model}: exit ${status}: ${first_line}")
    endif()
endforeach()

if(failures)
    string(REPLACE ";" "\n" report "${failures}")
    message(FATAL_ERROR "a throw under the JIT without TypeScriptRuntime:\n${report}")
endif()

message(STATUS "a throw under the JIT without TypeScriptRuntime runs under rc, none and own")
