# Under rc, a field of a @value value in an ordinary class's field, read into a const
# (`const s = h.r.start`) or into a literal of the @value interface (`{ start: h.r.start, dir:
# h.r.dir }`), takes a reference exactly as the same read of a type literal does
# (value/value_rc_field_retain.ts): each pair of functions, @value and type literal, has the same
# ownership operations, and at least one retain. A read that does not see the place it reads from
# (MLIRCodeLogic::GetReferenceFromValue through the value's storage tuple) is taken out of a copy of
# the record and not retained, so the const is freed under it once the field is overwritten.
cmake_minimum_required(VERSION 3.17.3)
function(ownership_ops ir fn out)
    string(FIND "${ir}" "ts.Func @${fn} " begin)
    if(begin EQUAL -1)
        message(FATAL_ERROR "no function `${fn}` in:
${ir}")
    endif()
    string(SUBSTRING "${ir}" ${begin} -1 body)
    string(FIND "${body}" "
  }" end)
    string(SUBSTRING "${body}" 0 ${end} body)
    string(REGEX MATCHALL "\"ts[.](Retain|Release)[A-Za-z]*\"" ops "${body}")
    set(${out} "${ops}" PARENT_SCOPE)
endfunction()
execute_process(COMMAND "${TSLANG}" --emit=mlir --no-default-lib -mm=rc "${FILE}"
                OUTPUT_VARIABLE out ERROR_VARIABLE err RESULT_VARIABLE result)
if(NOT result EQUAL 0)
    message(FATAL_ERROR "tslang failed (${result}):
${out}${err}")
endif()
# the module is printed to stderr
set(ir "${out}${err}")
foreach(pair "rayFromHit|rayTFromHit" "startFromHit|startTFromHit")
    string(REPLACE "|" ";" pair "${pair}")
    list(GET pair 0 value_fn)
    list(GET pair 1 literal_fn)
    ownership_ops("${ir}" ${value_fn} value_ops)
    ownership_ops("${ir}" ${literal_fn} literal_ops)
    if(NOT value_ops STREQUAL literal_ops)
        message(FATAL_ERROR "`${value_fn}` (@value) has other ownership operations than `${literal_fn}` (type literal):
${value_fn}: ${value_ops}
${literal_fn}: ${literal_ops}")
    endif()
    if(NOT value_ops MATCHES "Retain")
        message(FATAL_ERROR "no retain in `${value_fn}` or `${literal_fn}`: ${value_ops}")
    endif()
endforeach()
