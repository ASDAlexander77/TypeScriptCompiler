# Runs the ownership verifier over one shard of the corpus.
#
# `--verify-ownership` is an affine-level pass, so this is a compile and not a run: about 80ms a
# file, no linker, no JIT. It asks one question of every function - does a slot that takes a
# reference give it back on every path out, unwind paths included - and it has answered "no"
# twice for real: the break/continue scope-exit bug of section 9.18, and the two nested `using`
# scopes of section 9.62. Both times it was run by hand, and nothing repeated the run, which is
# why the second pair sat open for as long as it did. This is that sweep, repeated.
#
# One memory model is enough. The ownership operations survive to this level whatever the model
# is - they are only erased on the way to LLVM - so `-mm=gc` and `-mm=none` report exactly what
# `-mm=rc` reports here; the model is named only because the pass has to pick one.
#
# Sharded purely so ctest can spread the cost; the shards are one sweep, not one test each.

# Run with `cmake -P`, so no policy version is inherited from the project. Without this,
# CMP0057 defaults to OLD and `IN_LIST` below is not an operator.
cmake_minimum_required(VERSION 3.17.3)

if(NOT DEFINED TSLANG OR NOT DEFINED TESTS_DIR OR NOT DEFINED SHARD OR NOT DEFINED SHARDS)
    message(FATAL_ERROR "TSLANG, TESTS_DIR, SHARD and SHARDS are all required")
endif()

if(NOT DEFINED MODEL)
    set(MODEL "rc")
endif()

# Files that do not compile under these flags for reasons of their own, and so have nothing to
# report. Kept as a list rather than by ignoring the exit code, so that a file which stops
# compiling for a NEW reason - a crash, most of all - fails this test instead of passing it
# silently.
set(not_compilable_alone
    # needs a companion module that is not on this command line
    00switch_state.ts
    import_vars.ts
    # written for `-nostrictnull`
    raytrace-0.ts)

file(GLOB corpus "${TESTS_DIR}/*.ts")
list(SORT corpus)

set(failures "")
set(checked 0)
set(index 0)
foreach(file ${corpus})
    math(EXPR bucket "${index} % ${SHARDS}")
    math(EXPR index "${index} + 1")
    if(NOT bucket EQUAL SHARD)
        continue()
    endif()

    get_filename_component(name "${file}" NAME)
    if(name IN_LIST not_compilable_alone)
        continue()
    endif()

    execute_process(
        COMMAND "${TSLANG}" --emit=mlir-affine "-mm=${MODEL}" --verify-ownership --no-default-lib "${file}"
        OUTPUT_QUIET
        ERROR_VARIABLE diagnostics
        RESULT_VARIABLE status)

    math(EXPR checked "${checked} + 1")

    if(diagnostics MATCHES "error: ownership:")
        string(REGEX MATCHALL "[^\n]*error: ownership:[^\n]*" reported "${diagnostics}")
        string(REPLACE ";" "\n    " reported "${reported}")
        list(APPEND failures "  ${name}\n    ${reported}")
    elseif(NOT status EQUAL 0)
        list(APPEND failures "  ${name}\n    did not compile (exit ${status}), so nothing was checked")
    endif()
endforeach()

if(failures)
    string(REPLACE ";" "\n" report "${failures}")
    message(FATAL_ERROR "ownership verifier, shard ${SHARD} of ${SHARDS}:\n${report}")
endif()

message(STATUS "ownership verifier clean over ${checked} files (shard ${SHARD} of ${SHARDS}, -mm=${MODEL})")
