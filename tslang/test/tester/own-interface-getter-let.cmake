# Under own a read through an interface field put in a `let` that is only read (own_interface_getter_let.ts) is
# released at the function's returns on the getter's path (localHeldToExits). No run
# shows a leak, so the IR is read (--emit=mlir-affine): `heldInLet` releases the read before it returns.
cmake_minimum_required(VERSION 3.17.3)
execute_process(COMMAND "${TSLANG}" --emit=mlir-affine --no-default-lib -mm=own "${FILE}"
                OUTPUT_VARIABLE out ERROR_VARIABLE err RESULT_VARIABLE result)
# the IR is printed to stderr
string(APPEND out "${err}")
if(NOT result EQUAL 0)
    message(FATAL_ERROR "tslang failed (${result}):\n${out}")
endif()

# the body of `heldInLet`, from its `ts.Func` to the next one
string(FIND "${out}" "ts.Func @heldInLet " begin)
if(begin EQUAL -1)
    message(FATAL_ERROR "no function `heldInLet` in:\n${out}")
endif()
string(SUBSTRING "${out}" ${begin} -1 body)
string(SUBSTRING "${body}" 1 -1 rest)
string(FIND "${rest}" "ts.Func @" end)
if(NOT end EQUAL -1)
    math(EXPR end "${end} + 1")
    string(SUBSTRING "${body}" 0 ${end} body)
endif()

if(NOT body MATCHES "\"ts[.]Release\"[(]%[0-9]+[)] {__own_cell_read}")
    message(FATAL_ERROR "`heldInLet` does not release the read through the interface field:\n${body}")
endif()
