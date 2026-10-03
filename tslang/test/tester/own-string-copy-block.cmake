# Under own a string stored on one branch only gets its copy in the store's block, never at rc's
# birth retain after the producer: copied there, the copy has no owner on the path that skips the
# store, a leak no run can see (spec 22.7, "only for a use in the retain's block").
# Checked on `stored` of own_string_copy_branch.ts: its one `ts.StringCopy` comes after the entry
# block's `cf.cond_br`. That file has no loop, so without the rule it still compiles.
cmake_minimum_required(VERSION 3.17.3)
execute_process(COMMAND "${TSLANG}" --emit=mlir-affine --no-default-lib -mm=own "${FILE}"
                OUTPUT_VARIABLE out ERROR_VARIABLE err RESULT_VARIABLE result)
# the IR is printed to stderr
string(APPEND out "${err}")
if(NOT result EQUAL 0)
    message(FATAL_ERROR "tslang failed (${result}):\n${out}")
endif()
string(FIND "${out}" "ts.Func @stored " begin)
if(begin EQUAL -1)
    message(FATAL_ERROR "no function `stored` in:\n${out}")
endif()
string(SUBSTRING "${out}" ${begin} -1 body)
string(SUBSTRING "${body}" 1 -1 rest)
string(FIND "${rest}" "ts.Func @" end)
if(NOT end EQUAL -1)
    math(EXPR end "${end} + 1")
    string(SUBSTRING "${body}" 0 ${end} body)
endif()
string(REGEX MATCHALL "\"ts[.]StringCopy\"" copies "${body}")
list(LENGTH copies count)
if(NOT count EQUAL 1)
    message(FATAL_ERROR "`stored` has ${count} ts.StringCopy, not 1:\n${body}")
endif()
string(FIND "${body}" "cf.cond_br" branch)
string(FIND "${body}" "\"ts.StringCopy\"" copy)
if(branch EQUAL -1 OR copy LESS branch)
    message(FATAL_ERROR "`stored`'s copy is not in the store's block:\n${body}")
endif()
