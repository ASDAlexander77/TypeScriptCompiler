# A new array only read where it is made (temp-array-literal/temp_array_literal.ts) - `[a, b].length`,
# `[a, b][1]`, `[...xs, b].length` - is held and given back at the end of its block under rc: in
# `reads` each one is retained once and released once. Before, nothing gave it back. No run shows the
# leak, so the IR is read, after OwnedReturnConsumptionPass (--emit=mlir).
cmake_minimum_required(VERSION 3.17.3)
execute_process(COMMAND "${TSLANG}" --emit=mlir --no-default-lib -mm=rc "${FILE}"
                OUTPUT_VARIABLE out ERROR_VARIABLE err RESULT_VARIABLE result)
# the IR is printed to stderr
string(APPEND out "${err}")
if(NOT result EQUAL 0)
    message(FATAL_ERROR "tslang failed (${result}):\n${out}")
endif()

string(FIND "${out}" "ts.Func @reads " begin)
if(begin EQUAL -1)
    message(FATAL_ERROR "no function `reads` in:\n${out}")
endif()
string(SUBSTRING "${out}" ${begin} -1 body)
string(SUBSTRING "${body}" 1 -1 rest)
string(FIND "${rest}" "ts.Func @" end)
if(NOT end EQUAL -1)
    math(EXPR end "${end} + 1")
    string(SUBSTRING "${body}" 0 ${end} body)
endif()

# the arrays the literals make: each non-empty CreateArray, and the read of the spread literal's slot
string(REGEX MATCHALL "%[0-9]+ = \"ts[.]CreateArray\"[(]%[^\n]*|%[0-9]+ = \"ts[.]Load\"[^\n]*__spread_array_literal" made "${body}")
list(LENGTH made count)
# [a, b, "x"], [a, "q" + i], [a] and the spread [...[a], b], [new P, new P]
if(NOT count EQUAL 5)
    message(FATAL_ERROR "expected 5 new arrays in `reads`, found ${count}:\n${body}")
endif()

foreach(line ${made})
    string(REGEX MATCH "^(%[0-9]+)" array "${line}")
    string(REGEX MATCHALL "\"ts[.]Retain\"[(]${array}[)]" retains "${body}")
    string(REGEX MATCHALL "\"ts[.]Release\"[(]${array}[)]" releases "${body}")
    list(LENGTH retains r)
    list(LENGTH releases l)
    if(NOT r EQUAL 1 OR NOT l EQUAL 1)
        message(FATAL_ERROR "${array} is retained ${r} and released ${l} times, not once each:\n${body}")
    endif()
endforeach()
