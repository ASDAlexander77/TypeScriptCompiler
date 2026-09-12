# Emits LLVM IR for a few reference-counted programs WITH debug info.
#
# This combination has no other coverage in the suite. `test-runner` decides between `--opt` and
# `--di` from the build configuration rather than a flag, so a `-mm=rc` test registered through
# it gets whichever the current build uses and never both. That is why the failure of section
# 9.31 survived to section 9.69 as a one-line note: `--di --opt_level=0 -mm=rc` emitted no IR at
# all, for any reference-counted program, and nothing ran it.
#
# A compile, not a run: the bug was a module that would not translate, so reaching the end of
# emission is the whole assertion. Kept to a handful of files rather than the corpus because the
# routines that carry the bug - the generated `tsrel_`/`tsret_` pair and the `__tslang_*`
# helpers - are all reached by the shapes below, and a corpus-wide sweep would cost the suite
# thirty seconds to say the same thing.

if(NOT DEFINED TSLANG OR NOT DEFINED TESTS_DIR)
    message(FATAL_ERROR "TSLANG and TESTS_DIR are both required")
endif()

set(files
    # written for this: a string local, an owning field, an array, a closure, a class
    00owned_debug_info.ts
    # the two section 9.31 named when it found the failure
    00owned_temporaries.ts
    00interface.ts
    # `any` boxing and the closure/capture routines, which reach the other helpers
    00any.ts
    00owned_closures.ts
    # the largest reference-counted program here
    raytrace.ts)

set(failures "")
foreach(name ${files})
    execute_process(
        COMMAND "${TSLANG}" --emit=llvm --di --opt_level=0 -mm=rc --no-default-lib
                "${TESTS_DIR}/${name}" -o=${name}.ll
        OUTPUT_QUIET
        ERROR_VARIABLE diagnostics
        RESULT_VARIABLE status)

    if(NOT status EQUAL 0)
        string(REGEX REPLACE "\n.*" "" first_line "${diagnostics}")
        list(APPEND failures "  ${name}: exit ${status}: ${first_line}")
    endif()

    file(REMOVE "${name}.ll")
endforeach()

if(failures)
    string(REPLACE ";" "\n" report "${failures}")
    message(FATAL_ERROR "debug info under -mm=rc:\n${report}")
endif()

message(STATUS "debug info clean under -mm=rc over ${CMAKE_MATCH_COUNT} files")
