# -mm=own without --own-fallback=count: a field taken out of a call's own result is a
# ts.TakeOutOfField (own rc fallback design, section 4: in every own build, as releases honor extra
# owners in every own build), not an empty store - a callee built with the flag may return a counted
# object. FILE compiles under plain -mm=own and its IR has the op.
cmake_minimum_required(VERSION 3.17.3)
execute_process(COMMAND "${TSLANG}" --emit=mlir-affine --no-default-lib -mm=own "${FILE}"
                OUTPUT_VARIABLE out ERROR_VARIABLE err RESULT_VARIABLE result)
string(APPEND out "${err}")
if(NOT result EQUAL 0)
    message(FATAL_ERROR "tslang failed (${result}):\n${out}")
endif()
string(FIND "${out}" "\"ts.TakeOutOfField\"" take)
if(take EQUAL -1)
    message(FATAL_ERROR "no ts.TakeOutOfField without the flag:\n${out}")
endif()
