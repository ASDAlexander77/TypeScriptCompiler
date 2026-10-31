# A call through a class's vtable to a method no class overrides is made direct
# (MLIRGenImpl::devirtualizeClosedMethods); one an override exists for stays virtual
# (tests/00class_method_devirtualized.ts). The IR is read (--emit=mlir): `nameOf` calls `Shape.name`
# directly, and `areaOf` still calls `Shape.area` through the vtable.
cmake_minimum_required(VERSION 3.17.3)
execute_process(COMMAND "${TSLANG}" --emit=mlir --no-default-lib "${FILE}"
                OUTPUT_VARIABLE out ERROR_VARIABLE err RESULT_VARIABLE result)
# the IR is printed to stderr
string(APPEND out "${err}")
if(NOT result EQUAL 0)
    message(FATAL_ERROR "tslang failed (${result}):\n${out}")
endif()

# the body of a function, from its `ts.Func` to the next one
function(body_of name var)
    string(FIND "${out}" "ts.Func @${name} " begin)
    if(begin EQUAL -1)
        message(FATAL_ERROR "no function `${name}` in:\n${out}")
    endif()
    string(SUBSTRING "${out}" ${begin} -1 body)
    string(SUBSTRING "${body}" 1 -1 rest)
    string(FIND "${rest}" "ts.Func @" end)
    if(NOT end EQUAL -1)
        math(EXPR end "${end} + 1")
        string(SUBSTRING "${body}" 0 ${end} body)
    endif()
    set(${var} "${body}" PARENT_SCOPE)
endfunction()

body_of(nameOf name_body)
if(name_body MATCHES "ThisVirtualSymbolRef" OR NOT name_body MATCHES "\"ts[.]ThisSymbolRef\"[(]%[0-9]+[)] <{identifier = @Shape[.]name}>")
    message(FATAL_ERROR "`nameOf` does not call `Shape.name` directly:\n${name_body}")
endif()

body_of(areaOf area_body)
if(NOT area_body MATCHES "ThisVirtualSymbolRef")
    message(FATAL_ERROR "`areaOf` does not call `Shape.area` through the vtable:\n${area_body}")
endif()
