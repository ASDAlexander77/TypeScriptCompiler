# A new array given to an optional parameter (optional-array-argument/optional_array_argument.ts): the call
# takes `ts.OptionalValue` of it, and nothing gave the array back (`new Set([a, b])` leaked every call). No run
# shows the leak, so the IR is read after OwnedReturnConsumptionPass (--emit=mlir): under rc each new array in
# `reads` is retained once and released once around its call; under own each is an owned result that is given
# back (released) after it.
cmake_minimum_required(VERSION 3.17.3)
execute_process(COMMAND "${TSLANG}" --emit=mlir --no-default-lib -mm=${MM} "${FILE}"
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

string(REGEX MATCHALL "%[0-9]+ = \"ts[.]CreateArray\"[(]%[^\n]*" made "${body}")
list(LENGTH made count)
if(NOT count EQUAL 3)
    message(FATAL_ERROR "expected 3 new arrays in `reads`, found ${count}:\n${body}")
endif()

foreach(line ${made})
    string(REGEX MATCH "^(%[0-9]+)" array "${line}")
    string(REGEX MATCHALL "\"ts[.]Release\"[(]${array}[)]" releases "${body}")
    list(LENGTH releases l)
    if(NOT l EQUAL 1)
        message(FATAL_ERROR "${array} is released ${l} times, not once:\n${body}")
    endif()

    if(MM STREQUAL "rc")
        string(REGEX MATCHALL "\"ts[.]Retain\"[(]${array}[)]" retains "${body}")
        list(LENGTH retains r)
        if(NOT r EQUAL 1)
            message(FATAL_ERROR "${array} is retained ${r} times, not once:\n${body}")
        endif()
    endif()
endforeach()
