# `@owned` on a declaration (owned-declarations/owned_declarations.ts): the function it covers is
# marked ts.declared_owned_result - said on the function, on its `declare class` (methods and `..new`)
# or on its `declare namespace` - and, since nothing else in the file returns a heap value it did not
# retain, the call through a parameter in `viaValue` has its result consumed (__owned_result, not
# MLIRGen's __owned_result_named).
# The same file with every `@owned` taken out is the other side: a declaration may be C, or hand back a
# pointer into a buffer, so nothing is marked and the indirect call keeps retaining - closed world
# does not hold. Read after OwnedReturnConsumptionPass (--emit=mlir), under rc.
cmake_minimum_required(VERSION 3.17.3)

function(compile file outVar)
    execute_process(COMMAND "${TSLANG}" --emit=mlir --no-default-lib -mm=rc "${file}"
                    OUTPUT_VARIABLE out ERROR_VARIABLE err RESULT_VARIABLE result)
    # the IR is printed to stderr
    string(APPEND out "${err}")
    if(NOT result EQUAL 0)
        message(FATAL_ERROR "tslang failed (${result}) on ${file}:\n${out}")
    endif()
    set(${outVar} "${out}" PARENT_SCOPE)
endfunction()

# the text of a function, from its `ts.Func` to the next one
function(func_text out function outVar)
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

set(declared lib_text LibBox..new LibBox.describe LibNS.name)

compile("${FILE}" marked)
foreach(function ${declared})
    func_text("${marked}" ${function} text)
    if(NOT text MATCHES "ts[.]declared_owned_result")
        message(FATAL_ERROR "`${function}` is not marked ts.declared_owned_result:\n${text}")
    endif()
endforeach()
func_text("${marked}" viaValue text)
if(NOT text MATCHES "\"ts[.]CallIndirect\"[^\n]*__owned_result[,}]")
    message(FATAL_ERROR "the call through a function value in `viaValue` is not consumed:\n${text}")
endif()

file(READ "${FILE}" source)
string(REPLACE "@owned" "" source "${source}")
set(unmarkedFile "${WORK_DIR}/owned_declarations_unmarked.ts")
file(WRITE "${unmarkedFile}" "${source}")
compile("${unmarkedFile}" unmarked)
foreach(function ${declared})
    func_text("${unmarked}" ${function} text)
    if(text MATCHES "ts[.]declared_owned_result")
        message(FATAL_ERROR "`${function}` says nothing, yet is marked ts.declared_owned_result:\n${text}")
    endif()
endforeach()
func_text("${unmarked}" viaValue text)
if(text MATCHES "\"ts[.]CallIndirect\"[^\n]*__owned_result[,}]")
    message(FATAL_ERROR "with unmarked declarations the call in `viaValue` must keep retaining:\n${text}")
endif()
