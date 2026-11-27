# `@owned` on a declaration (owned-declarations/owned_declarations.ts): the function it covers is
# marked ts.declared_owned_result - said on the function, on its `declare class` (methods and `..new`)
# or on its `declare namespace` - and the result of a call of it, read and dropped (`direct`), is given
# back (__owned_result). Under rc, since nothing else in the file returns a heap value it did not
# retain, the call through a parameter in `viaValue` has its result consumed as well (__owned_result,
# not MLIRGen's __owned_result_named) - and so is `lent`'s, a `@lends` method's, which rc's body
# retains for its caller. Under own a `@lends` method's result is a borrow: `lent`'s call is not
# marked, and with it among the declarations closed world does not hold.
# The same file with every `@owned` taken out is the other side: a declaration may be C, or hand back a
# pointer into a buffer, so nothing is marked and the calls keep retaining - closed world does not
# hold. Read after OwnedReturnConsumptionPass (--emit=mlir), under rc and under own (MM).
cmake_minimum_required(VERSION 3.17.3)

if(NOT MM)
    set(MM rc)
endif()

function(compile file outVar)
    execute_process(COMMAND "${TSLANG}" --emit=mlir --no-default-lib -mm=${MM} "${file}"
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

set(declared lib_text LibBox..new LibBox.describe LibBox.child LibNS.name)

compile("${FILE}" marked)
foreach(function ${declared})
    func_text("${marked}" ${function} text)
    if(NOT text MATCHES "ts[.]declared_owned_result")
        message(FATAL_ERROR "`${function}` is not marked ts.declared_owned_result:\n${text}")
    endif()
endforeach()
# the result of a call given back: `"ts.CallIndirect"(...) {__owned_result...}`, not `__owned_result_named` alone
set(consumed "\"ts[.]CallIndirect\"[^\n]*[{ ]__owned_result[,}]")
func_text("${marked}" direct text)
if(NOT text MATCHES "${consumed}")
    message(FATAL_ERROR "the result of the call of `lib_text` in `direct` is not given back (${MM}):\n${text}")
endif()
func_text("${marked}" viaValue text)
if(MM STREQUAL "rc" AND NOT text MATCHES "${consumed}")
    message(FATAL_ERROR "the call through a function value in `viaValue` is not consumed:\n${text}")
endif()
func_text("${marked}" lent text)
if(MM STREQUAL "rc" AND NOT text MATCHES "${consumed}")
    message(FATAL_ERROR "under rc the result of `label` in `lent` is the caller's:\n${text}")
endif()
if(MM STREQUAL "own" AND text MATCHES "${consumed}")
    message(FATAL_ERROR "under own `label` lends its result (`@lends`): `lent` must not give it back:\n${text}")
endif()

# ...and own's inference takes an `@owned` result for the caller's own, not a part of what it was called on
# (`ofChild`): checked past the affine lowering, where inference runs
if(MM STREQUAL "own")
    execute_process(COMMAND "${TSLANG}" --emit=mlir-affine --no-default-lib -mm=own "${FILE}"
                    OUTPUT_VARIABLE out ERROR_VARIABLE err RESULT_VARIABLE result)
    if(NOT result EQUAL 0)
        message(FATAL_ERROR "-mm=own refused ${FILE} (${result}):\n${err}")
    endif()
endif()

file(READ "${FILE}" source)
string(REPLACE "@owned" "" source "${source}")
set(unmarkedFile "${WORK_DIR}/owned_declarations_unmarked_${MM}.ts")
file(WRITE "${unmarkedFile}" "${source}")
compile("${unmarkedFile}" unmarked)
foreach(function ${declared})
    func_text("${unmarked}" ${function} text)
    if(text MATCHES "ts[.]declared_owned_result")
        message(FATAL_ERROR "`${function}` says nothing, yet is marked ts.declared_owned_result:\n${text}")
    endif()
endforeach()
foreach(function viaValue direct)
    func_text("${unmarked}" ${function} text)
    if(text MATCHES "${consumed}")
        message(FATAL_ERROR "with unmarked declarations the call in `${function}` must keep retaining (${MM}):\n${text}")
    endif()
endforeach()
