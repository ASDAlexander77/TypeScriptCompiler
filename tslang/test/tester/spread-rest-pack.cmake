# A spread of run-time length given to a rest parameter (spread-rest-pack/spread_rest_pack.ts:
# `cat(...parts)`) packs a new array, born uncounted, read out of the slot MLIRGen filled it in
# (`ts.Load` marked __spread_array_literal). Under rc the call holds it - a retain right before the
# call, a release right after - as it holds the array `cat(a, b)` packs. Before, nothing gave it back:
# no run shows the leak (the default library's Array.join lost the array and a count of every
# element on each call), so the IR is read, after OwnedReturnConsumptionPass (--emit=mlir).
cmake_minimum_required(VERSION 3.17.3)
execute_process(COMMAND "${TSLANG}" --emit=mlir --no-default-lib -mm=rc "${FILE}"
                OUTPUT_VARIABLE out ERROR_VARIABLE err RESULT_VARIABLE result)
# the IR is printed to stderr
string(APPEND out "${err}")
if(NOT result EQUAL 0)
    message(FATAL_ERROR "tslang failed (${result}):\n${out}")
endif()

string(FIND "${out}" "ts.Func @joined " begin)
if(begin EQUAL -1)
    message(FATAL_ERROR "no function `joined` in:\n${out}")
endif()
string(SUBSTRING "${out}" ${begin} -1 body)
string(SUBSTRING "${body}" 1 -1 rest)
string(FIND "${rest}" "ts.Func @" end)
if(NOT end EQUAL -1)
    math(EXPR end "${end} + 1")
    string(SUBSTRING "${body}" 0 ${end} body)
endif()

string(REGEX MATCH "(%[0-9]+) = \"ts[.]Load\"[^\n]*__spread_array_literal" load "${body}")
if(NOT load)
    message(FATAL_ERROR "no packed spread (a ts.Load marked __spread_array_literal) in `joined`:\n${body}")
endif()
set(array "${CMAKE_MATCH_1}")

if(NOT body MATCHES "\"ts[.]Retain\"[(]${array}[)][^\n]*\n[^\n]*\"ts[.]CallIndirect\"[(][^\n]*${array}[^\n]*\n[^\n]*\"ts[.]Release\"[(]${array}[)]")
    message(FATAL_ERROR "the packed spread ${array} is not held across the call (retain, call, release):\n${body}")
endif()
