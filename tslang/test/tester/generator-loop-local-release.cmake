# A local declared in a generator's loop is a field of the state object, stored again on every turn:
# under rc the generator's `next` releases what the field held before the store. Without it every
# turn but the last leaked, and no run shows that, so the IR is read: each generator in
# own_generator_loop_local.ts has a `ts.ReleaseSlot` of its `string[]` field.
cmake_minimum_required(VERSION 3.17.3)
execute_process(COMMAND "${TSLANG}" --emit=mlir-affine --no-default-lib -mm=rc "${FILE}"
                OUTPUT_VARIABLE out ERROR_VARIABLE err RESULT_VARIABLE result)
# the IR is printed to stderr
string(APPEND out "${err}")
if(NOT result EQUAL 0)
    message(FATAL_ERROR "tslang failed (${result}):\n${out}")
endif()
foreach(generator rows untilBreak pairs)
    string(FIND "${out}" "ts.Func @\".f_${generator}..f_${generator}..obj" begin)
    if(begin EQUAL -1)
        message(FATAL_ERROR "no `next` of generator `${generator}` in:\n${out}")
    endif()
    string(SUBSTRING "${out}" ${begin} -1 body)
    string(SUBSTRING "${body}" 1 -1 rest)
    string(FIND "${rest}" "ts.Func @" end)
    if(NOT end EQUAL -1)
        math(EXPR end "${end} + 1")
        string(SUBSTRING "${body}" 0 ${end} body)
    endif()
    if(NOT body MATCHES "\"ts[.]ReleaseSlot\"[(]%[0-9]+[)] : [(]!ts[.]ref<!ts[.]array<!ts[.]string>>[)]")
        message(FATAL_ERROR "`${generator}` does not release its loop local's old value:\n${body}")
    endif()
endforeach()
