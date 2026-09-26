# tsbindgen's command line: 0 when the file was written, 1 for a parse error or I/O failure, 2 for
# bad arguments - and never 0 for a command line clang rejected.

cmake_minimum_required(VERSION 3.17.3)

foreach(var TSBINDGEN SOURCE_DIR WORK_DIR)
    if(NOT DEFINED ${var})
        message(FATAL_ERROR "${var} is required")
    endif()
endforeach()

file(REMOVE_RECURSE "${WORK_DIR}")
file(MAKE_DIRECTORY "${WORK_DIR}")
file(WRITE "${WORK_DIR}/broken.h" "int broken(\n")

# expect(<exit code> <what> <args...>) - leaves stdout + stderr in cli_output
function(expect code what)
    execute_process(COMMAND "${TSBINDGEN}" ${ARGN}
        WORKING_DIRECTORY "${WORK_DIR}"
        OUTPUT_VARIABLE out
        ERROR_VARIABLE err
        RESULT_VARIABLE status)
    if(NOT status EQUAL code)
        message(FATAL_ERROR "${what}: exit ${status}, expected ${code}\n${out}\n${err}")
    endif()
    set(cli_output "${out}${err}" PARENT_SCOPE)
    message(STATUS "${what}: exit ${code}")
endfunction()

set(header "${SOURCE_DIR}/fixture.h")

expect(0 "a header, written to a file" "${header}" -o out.ts)
if(NOT EXISTS "${WORK_DIR}/out.ts")
    message(FATAL_ERROR "exit 0, but out.ts was not written")
endif()

# `import` would load a same-named library instead, so the file says how to include it
file(READ "${WORK_DIR}/out.ts" written)
if(NOT written MATCHES "// Include with: /// <reference path=\"out\\.ts\" />")
    message(FATAL_ERROR "out.ts lacks the include hint:\n${written}")
endif()

expect(0 "a header, to stdout" "${header}" --filter fx_add)
if(NOT cli_output MATCHES "declare function fx_add\\(a: s32, b: s32\\): s32;")
    message(FATAL_ERROR "stdout lacks fx_add:\n${cli_output}")
endif()

expect(0 "--version" --version)
if(NOT cli_output MATCHES "tsbindgen .* \\(clang [0-9]+")
    message(FATAL_ERROR "--version printed:\n${cli_output}")
endif()

expect(0 "a .d.ts output warns" "${header}" -o out.d.ts)
if(NOT cli_output MATCHES "name it \\.ts")
    message(FATAL_ERROR "no .d.ts warning:\n${cli_output}")
endif()

expect(1 "a missing input" missing.h)
expect(1 "a header clang rejects" broken.h)
expect(1 "an argument clang rejects" "${header}" -- --no-such-clang-flag)
expect(1 "an unwritable output" "${header}" -o "${WORK_DIR}/no/such/dir/out.ts")

expect(2 "no input" --filter x)
expect(2 "an unknown option" "${header}" --no-such-option)
expect(2 "--strip-prefix without --namespace" "${header}" --strip-prefix fx_)
expect(2 "an invalid glob" "${header}" --filter "[")
expect(2 "a --resource-dir without clang's headers" "${header}" --resource-dir "${WORK_DIR}")
