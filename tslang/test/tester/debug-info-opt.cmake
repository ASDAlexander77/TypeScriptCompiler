# Emits LLVM IR for a few programs WITH debug info AND optimisation, under every memory model.
#
# `test-runner` picks `--opt` or `--di` from the build configuration, never both, so nothing else
# here asks for the pair - and it was broken three ways, all of them before any code was generated:
#
# - `--opt` stripped every location ahead of the inliner while `--di` still gave every function a
#   subprogram, so a call the inliner left had no location in a function that needed one. Under rc
#   that is every program (the `tsrel_`/`tsret_` helpers are made after the inliner runs): "inlinable
#   function call in a function with a DISubprogram location must have a debug location".
# - With the locations kept, DIScopeForLLVMFuncOpPass crashed on two shapes the inliner leaves: a
#   call site whose callee has no file (an op made without a location, inlined), and a call site on
#   an op outside any function (a string literal's global, made for an inlined op). 18enums.ts has
#   both.
#
# A compile, not a run, for the same reason as debug-info-rc.cmake: every failure was a module that
# would not come out.

if(NOT DEFINED TSLANG OR NOT DEFINED TESTS_DIR)
    message(FATAL_ERROR "TSLANG and TESTS_DIR are both required")
endif()

set(files
    # inlined string literals and an op with no location of its own
    18enums.ts
    # a string local, an owning field, an array, a closure, a class
    00owned_debug_info.ts
    # classes, interfaces and closures, which reach the reference-counting helpers
    00interface.ts
    00owned_closures.ts
    # the largest program here
    raytrace.ts)

set(failures "")
set(count 0)
foreach(model gc rc none)
    foreach(name ${files})
        execute_process(
            COMMAND "${TSLANG}" --emit=llvm --di --opt -mm=${model} --no-default-lib
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
    message(FATAL_ERROR "debug info with --opt:\n${report}")
endif()

message(STATUS "debug info with --opt clean over ${count} compiles")
