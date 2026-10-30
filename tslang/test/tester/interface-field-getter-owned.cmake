# Under rc a read through an interface's field is owned (00interface_field_getter_owned.ts): the field may be
# filled by a getter, which returns its result retained, so the lowering retains what a plain slot holds too.
# No run shows a leak, so the IR is read, after OwnedReturnConsumptionPass (--emit=mlir). Every read through
# an interface field in these functions is marked owned, and is taken over by its receiver or released exactly
# once; an assignment leaves no read of the place it wrote.
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

# the reads through an interface field in `body`: the lines loading an InterfaceSymbolRef's result
function(interface_reads body outVar)
    set(reads)
    string(REGEX MATCHALL "%[0-9]+ = \"ts[.]InterfaceSymbolRef\"" refs "${body}")
    foreach(ref IN LISTS refs)
        string(REGEX MATCH "^%([0-9]+)" ignored "${ref}")
        string(REGEX MATCHALL "%[0-9]+ = \"ts[.]Load\"[(]%${CMAKE_MATCH_1}[)][^\r\n]*" loads "${body}")
        list(APPEND reads ${loads})
    endforeach()
    set(${outVar} "${reads}" PARENT_SCOPE)
endfunction()

# function name, then how many reads through an interface field it has
set(expected
    "intoLocal:1"
    "asTemporary:2"
    "asArgument:1"
    "returned:1"
    "inLoop:1"
    "storedLocal:1"
    "stored:0")
foreach(entry IN LISTS expected)
    string(REPLACE ":" ";" entry "${entry}")
    list(GET entry 0 function)
    list(GET entry 1 wanted)
    func_body("${function}" body)
    interface_reads("${body}" reads)
    list(LENGTH reads count)
    if(NOT count EQUAL wanted)
        message(FATAL_ERROR "`${function}` has ${count} reads through an interface field, not ${wanted}:\n${body}")
    endif()
    foreach(read IN LISTS reads)
        if(NOT read MATCHES "__owned_result")
            message(FATAL_ERROR "`${function}` reads an interface field unowned:\n${read}\n${body}")
        endif()
        if(read MATCHES "__owned_result_consumed")
            continue()
        endif()
        string(REGEX MATCH "^%([0-9]+)" ignored "${read}")
        string(REGEX MATCHALL "\"ts[.]Release\"[(]%${CMAKE_MATCH_1}[)]" releases "${body}")
        list(LENGTH releases n)
        if(NOT n EQUAL 1)
            message(FATAL_ERROR "`${function}` releases the read %${CMAKE_MATCH_1} ${n} times, not once:\n${body}")
        endif()
    endforeach()
endforeach()
