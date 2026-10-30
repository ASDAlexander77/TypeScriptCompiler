# Emits LLVM IR WITH debug info and no optimisation, as a debug build compiles (`--di --opt_level=0`),
# under every memory model, for programs that make functions of the compiler's own: glue with no source
# of its own, which still has to come out with a debug location on everything that calls.
#
# `test-runner` in a release build compiles with `--opt` and no `--di`, so it never sees these; the
# release workflow's debug default library does. A function taking fewer parameters than its function
# type is called through a thunk (fewerParametersThunk), made with no location: with `--di` it got a
# subprogram while its call had none, and LLVM rejected the module - "inlinable function call in a
# function with a DISubprogram location must have a debug location". Array callbacks take the
# element's index too, so `a.every((v) => ...)` made one in every debug program that used them.
#
# A compile, not a run, for the same reason as debug-info-rc.cmake: the failure was a module that
# would not come out.

if(NOT DEFINED TSLANG OR NOT DEFINED TESTS_DIR)
    message(FATAL_ERROR "TSLANG and TESTS_DIR are both required")
endif()

set(files
    # a named function and an arrow given as a function type with more parameters
    00function_fewer_parameters.ts
    # an arrow that captures, bound over its captures to the thunk
    00closure_fewer_parameters.ts)

set(failures "")
set(count 0)
foreach(model gc rc none own)
    foreach(name ${files})
        execute_process(
            COMMAND "${TSLANG}" --emit=llvm --di --opt_level=0 -mm=${model} --no-default-lib
                    "${TESTS_DIR}/${name}" -o=${name}.${model}.ll
            OUTPUT_QUIET
            ERROR_VARIABLE diagnostics
            RESULT_VARIABLE status)

        if(NOT status EQUAL 0)
            # the first error, not a warning printed ahead of it
            if(diagnostics MATCHES "[^\n]*error:[^\n]*")
                set(first_line "${CMAKE_MATCH_0}")
            else()
                string(REGEX REPLACE "\n.*" "" first_line "${diagnostics}")
            endif()
            list(APPEND failures "  ${name} -mm=${model}: exit ${status}: ${first_line}")
        endif()

        file(REMOVE "${name}.${model}.ll")
        math(EXPR count "${count} + 1")
    endforeach()
endforeach()

if(failures)
    string(REPLACE ";" "\n" report "${failures}")
    message(FATAL_ERROR "debug info without optimisation failed:\n${report}")
endif()

message(STATUS "debug info without optimisation: ${count} compile(s) emitted")
