# A `const` record literal built at run time is a variable whose slot is released (#485): under rc
# each of the seven records in 00const_record_owned_fields.ts has a `ts.ReleaseSlot` of its tuple
# slot. Folded, as before, nothing counted an array or tuple built in the literal, and no run shows
# the leak that left, so the IR is read.
cmake_minimum_required(VERSION 3.17.3)
execute_process(COMMAND "${TSLANG}" --emit=mlir-affine --no-default-lib -mm=rc "${FILE}"
                OUTPUT_VARIABLE out ERROR_VARIABLE err RESULT_VARIABLE result)
# the IR is printed to stderr
string(APPEND out "${err}")
if(NOT result EQUAL 0)
    message(FATAL_ERROR "tslang failed (${result}):\n${out}")
endif()
foreach(function arrayField stringField callField newField tupleField newArrayField nestedField)
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
    if(NOT body MATCHES "\"ts[.]ReleaseSlot\"[(]%[0-9]+[)] : [(]!ts[.]ref<!ts[.]tuple<")
        message(FATAL_ERROR "`${function}` does not release its record's slot:\n${body}")
    endif()
endforeach()
