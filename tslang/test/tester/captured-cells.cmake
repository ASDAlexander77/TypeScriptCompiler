# A captured variable's cell is given back on every path out of its scope under rc
# (captured-cells/captured_cells.ts). No run shows the leak, so the IR is read, after MLIRGen (--emit=mlir):
#  - loopCapture: the body's per-iteration `i` (a captured `ts.Variable` inside the loop) is released as a
#    cell - at the end of the body and at its `continue` and `break`;
#  - earlyReturn: the parameter's copy is captured, and each of the function's two returns releases it as a
#    cell (`ts.ReleaseCell`), the early one too; the only `ts.ReleaseSlot` of it left is its assignment's.
cmake_minimum_required(VERSION 3.17.3)
execute_process(COMMAND "${TSLANG}" --emit=mlir --no-default-lib -mm=rc "${FILE}"
                OUTPUT_VARIABLE out ERROR_VARIABLE err RESULT_VARIABLE result)
# the IR is printed to stderr
string(APPEND out "${err}")
if(NOT result EQUAL 0)
    message(FATAL_ERROR "tslang failed (${result}):\n${out}")
endif()

function(func_text function outVar)
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

function(count pattern body outVar)
    string(REGEX MATCHALL "${pattern}" matches "${body}")
    list(LENGTH matches n)
    set(${outVar} ${n} PARENT_SCOPE)
endfunction()

# the per-iteration `i`: a captured variable, made inside the loop's body region
func_text(loopCapture body)
string(REGEX MATCH "(%[0-9]+) = \"ts[.]Variable\"[(]%[0-9]+[)] <[{]captured = true" cell "${body}")
if(NOT cell)
    message(FATAL_ERROR "no captured per-iteration variable in `loopCapture`:\n${body}")
endif()
set(cell "${CMAKE_MATCH_1}")
count("\"ts[.]ReleaseCell\"[(]${cell}[)]" "${body}" released)
# the end of the body, the `continue` and the `break`
if(released LESS 3)
    message(FATAL_ERROR "the per-iteration cell ${cell} is released ${released} times, not on every way out of the body:\n${body}")
endif()

func_text(earlyReturn body)
string(REGEX MATCH "(%[0-9]+) = \"ts[.]Variable\"[(]%[0-9]+[)] <[{]captured = true" cell "${body}")
if(NOT cell)
    message(FATAL_ERROR "no captured copy of the parameter in `earlyReturn`:\n${body}")
endif()
set(cell "${CMAKE_MATCH_1}")
# the early return, the last one and the end of the body each give the cell back
count("\"ts[.]ReleaseCell\"[(]${cell}[)]" "${body}" cellReleases)
if(cellReleases LESS 3)
    message(FATAL_ERROR "the cell ${cell} of `earlyReturn` is released as a cell ${cellReleases} times, not on every way out:\n${body}")
endif()
count("\"ts[.]ReleaseSlot\"[(]${cell}[)][^\n]*\n[^\n]*\"ts[.]ReturnVal\"" "${body}" slotReturns)
if(NOT slotReturns EQUAL 0)
    message(FATAL_ERROR "a return of `earlyReturn` releases the cell ${cell} as a slot:\n${body}")
endif()
