# A folded `const` made by `new` and chosen by a conditional expression is released at the end of
# its block (00conditional_named_const_release.ts). The branch yields it through `ts.Result`, a
# terminator, and that kept it from any release under rc (and own, which reads rc's IR). No run
# shows the leak, so the IR is read, after OwnedReturnConsumptionPass (--emit=mlir). Each `new`
# bound to a `const` in a function is released exactly once - except, in `assignedOutside`, the
# one chosen into a local that owns nothing and outlives the `if` - and in `chosenOrNew` the
# `ts.If`, which owns the side it made, is released once as well.
cmake_minimum_required(VERSION 3.17.3)
execute_process(COMMAND "${TSLANG}" --emit=mlir --no-default-lib -mm=rc "${FILE}"
                OUTPUT_VARIABLE out ERROR_VARIABLE err RESULT_VARIABLE result)
# the IR is printed to stderr
string(APPEND out "${err}")
if(NOT result EQUAL 0)
    message(FATAL_ERROR "tslang failed (${result}):\n${out}")
endif()

# the body of a function, from its `ts.Func` to the next one
function(func_body function outVar)
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
    set(${outVar} "${body}" PARENT_SCOPE)
endfunction()

function(releases value body outVar)
    string(REGEX MATCHALL "\"ts[.]Release\"[(]%${value}[)]" matches "${body}")
    list(LENGTH matches n)
    set(${outVar} ${n} PARENT_SCOPE)
endfunction()

# function name, then how many times each of its named `new`s is released, in order
set(expected
    "constChosen:1:1"
    "letChosen:1:1"
    "chosenOrNew:1"
    "nestedChosen:1:1"
    "chosenInLoop:1:1"
    "returnChosen:1:1"
    "assignedOutside:1:0")
foreach(entry IN LISTS expected)
    string(REPLACE ":" ";" entry "${entry}")
    list(POP_FRONT entry function)
    func_body("${function}" body)
    string(REGEX MATCHALL "%[0-9]+ = \"ts[.]New\"[^\r\n]*__owned_result_named" news "${body}")
    list(LENGTH news count)
    list(LENGTH entry wanted)
    if(NOT count EQUAL wanted)
        message(FATAL_ERROR "`${function}` has ${count} named `new`s, not ${wanted}:\n${body}")
    endif()
    foreach(new IN LISTS news)
        list(POP_FRONT entry times)
        string(REGEX MATCH "^%([0-9]+)" ignored "${new}")
        releases("${CMAKE_MATCH_1}" "${body}" n)
        if(NOT n EQUAL times)
            message(FATAL_ERROR "`${function}` releases %${CMAKE_MATCH_1} ${n} times, not ${times}:\n${body}")
        endif()
    endforeach()
endforeach()

func_body("chosenOrNew" body)
string(REGEX MATCH "%([0-9]+) = \"ts[.]If\"[^\r\n]*[\r\n]+ *\"ts[.]Retain\"" ignored "${body}")
if(NOT CMAKE_MATCH_1)
    message(FATAL_ERROR "`chosenOrNew` does not retain the borrowed side of its `ts.If`:\n${body}")
endif()
releases("${CMAKE_MATCH_1}" "${body}" n)
if(NOT n EQUAL 1)
    message(FATAL_ERROR "`chosenOrNew` releases its `ts.If` %${CMAKE_MATCH_1} ${n} times, not once:\n${body}")
endif()
