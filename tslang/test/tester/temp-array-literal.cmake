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

# An element of a new array handed out of a `?:` branch (`pick`, `pickOrNull`): the branch retains it for
# the result, right before it yields it, and gives the array back - once - before that.
foreach(function pick pickOrNull)
    string(FIND "${out}" "ts.Func @${function} " begin)
    if(begin EQUAL -1)
        message(FATAL_ERROR "no function `${function}` in:\n${out}")
    endif()
    string(SUBSTRING "${out}" ${begin} -1 body)
    string(SUBSTRING "${body}" 1 -1 rest)
    string(FIND "${rest}" "ts.Func @" end)
    if(NOT end EQUAL -1)
        math(EXPR end "${end} + 1")
        string(SUBSTRING "${body}" 0 ${end} body)
    endif()

    string(REGEX MATCH "(%[0-9]+) = \"ts[.]CreateArray\"[(]%" made "${body}")
    if(NOT made)
        message(FATAL_ERROR "no new array in `${function}`:\n${body}")
    endif()
    set(array "${CMAKE_MATCH_1}")
    string(REGEX MATCHALL "\"ts[.]Retain\"[(]${array}[)]" retains "${body}")
    string(REGEX MATCHALL "\"ts[.]Release\"[(]${array}[)]" releases "${body}")
    list(LENGTH retains r)
    list(LENGTH releases l)
    if(NOT r EQUAL 1 OR NOT l EQUAL 1)
        message(FATAL_ERROR "${array} in `${function}` is retained ${r} and released ${l} times, not once each:\n${body}")
    endif()

    if(NOT body MATCHES "\"ts[.]Release\"[(]${array}[)][^\n]*\n[^\n]*\"ts[.]Result\"[(](%[0-9]+)[)]")
        message(FATAL_ERROR "`${function}` does not give the array back right before it yields:\n${body}")
    endif()
    set(yielded "${CMAKE_MATCH_1}")
    string(FIND "${body}" "\"ts.Retain\"(${yielded})" retained)
    string(FIND "${body}" "\"ts.Release\"(${array})" released)
    if(retained EQUAL -1 OR NOT retained LESS released)
        message(FATAL_ERROR "`${function}` does not retain what it yields (${yielded}) before it gives the array back:\n${body}")
    endif()
endforeach()
