# Under own a rest parameter its function says it keeps (`@keeps("args")`, own_rest_keeps_declared.ts) is
# released at the function's exit once its elements are moved out, as an inferred keep's is (spec 72). No run
# shows a leak, so the IR is read (--emit=mlir-affine): `List.push` releases its rest array before it returns.
cmake_minimum_required(VERSION 3.17.3)
execute_process(COMMAND "${TSLANG}" --emit=mlir-affine --no-default-lib -mm=own "${FILE}"
                OUTPUT_VARIABLE out ERROR_VARIABLE err RESULT_VARIABLE result)
# the IR is printed to stderr
string(APPEND out "${err}")
if(NOT result EQUAL 0)
    message(FATAL_ERROR "tslang failed (${result}):\n${out}")
endif()

# the body of `List.push`, from its `ts.Func` to the next one
string(FIND "${out}" "ts.Func @List.push " begin)
if(begin EQUAL -1)
    message(FATAL_ERROR "no function `List.push` in:\n${out}")
endif()
string(SUBSTRING "${out}" ${begin} -1 body)
string(SUBSTRING "${body}" 1 -1 rest)
string(FIND "${rest}" "ts.Func @" end)
if(NOT end EQUAL -1)
    math(EXPR end "${end} + 1")
    string(SUBSTRING "${body}" 0 ${end} body)
endif()

if(NOT body MATCHES "\"ts[.]ReleaseSlot\"[(]%[0-9]+[)] : [(]!ts[.]ref<!ts[.]array<!ts[.]class<@Item")
    message(FATAL_ERROR "`List.push` does not release its rest array:\n${body}")
endif()
