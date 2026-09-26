# A shared library imported twice into one module: the program imports plain_module.dll, and
# imports via_module.ts as source, which imports plain_module.dll again. The second import used to
# generate the library's declarations again, and the module verifier rejected the copy
# ("redefinition of symbol named 'plain_fn'"). test-runner cannot set this up: its -shared mode
# links every module into one DLL.
#
# The program is only compiled: via_module has no body anywhere here, so it would not link.
#
# -mm=none keeps the collector out of it: the DLL then links no gc.dll.

cmake_minimum_required(VERSION 3.17.3)

foreach(var TSLANG SOURCE_DIR WORK_DIR LLVM_LIB TSLANG_LIB)
    if(NOT DEFINED ${var})
        message(FATAL_ERROR "${var} is required")
    endif()
endforeach()

set(ENV{GC_LIB_PATH} "")
set(ENV{TSLANG_LIB_PATH} "")

file(REMOVE_RECURSE "${WORK_DIR}")
file(MAKE_DIRECTORY "${WORK_DIR}")

set(common -mm=none --no-default-lib)

# run(<what> <command...>) - fails the test unless the command succeeds
function(run what)
    execute_process(COMMAND ${ARGN}
        WORKING_DIRECTORY "${WORK_DIR}"
        OUTPUT_VARIABLE out
        ERROR_VARIABLE err
        RESULT_VARIABLE status)
    if(NOT status EQUAL 0 OR "${out}${err}" MATCHES "error:")
        message(FATAL_ERROR "${what}: exit ${status}\n${out}\n${err}")
    endif()
endfunction()

run("--emit=dll plain_module"
    "${TSLANG}" --emit=dll ${common} "--llvm-lib-path=${LLVM_LIB}" "--tslang-lib-path=${TSLANG_LIB}"
    "${SOURCE_DIR}/plain_module.ts" -o plain_module.dll)

# `import './plain_module'` finds the DLL in the working directory; './via_module' has no DLL, so
# it is imported as source, from beside the program
run("--emit=obj import_diamond, importing plain_module.dll directly and through via_module.ts"
    "${TSLANG}" --emit=obj ${common} "${SOURCE_DIR}/import_diamond.ts" -o import_diamond.obj)

message(STATUS "a DLL imported directly and through a source module is declared once")
