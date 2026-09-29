# -mm=own across a DLL boundary: the library exports which of its functions destroy nothing
# (`__tsown_<module>`), and an importer under own relies on it. test-runner's -shared mode runs the
# pair; this checks what it cannot - that an imported function the library does not list still
# drops, and that the listed ones are accepted because of the list: against the same library
# built under rc, which says nothing, the program that passes above is an error.

cmake_minimum_required(VERSION 3.17.3)

foreach(var TSLANG SOURCE_DIR WORK_DIR LLVM_LIB TSLANG_LIB)
    if(NOT DEFINED ${var})
        message(FATAL_ERROR "${var} is required")
    endif()
endforeach()

set(ENV{GC_LIB_PATH} "")
set(ENV{TSLANG_LIB_PATH} "")

set(borrow_ended "borrows (a field|an element) but is used here after it may be released or overwritten")

# compile(<dir> <what> <expect> <command...>) - <expect> is "ok" or a regex the output must match
function(compile dir what expect)
    execute_process(COMMAND ${ARGN}
        WORKING_DIRECTORY "${dir}"
        OUTPUT_VARIABLE out
        ERROR_VARIABLE err
        RESULT_VARIABLE status)
    if("${out}${err}" MATCHES "Stack dump|Assertion failed")
        message(FATAL_ERROR "${what}: crashed\n${out}\n${err}")
    endif()
    if(expect STREQUAL "ok")
        if(NOT status EQUAL 0 OR "${out}${err}" MATCHES "error:")
            message(FATAL_ERROR "${what}: exit ${status}\n${out}\n${err}")
        endif()
    elseif(status EQUAL 0 OR NOT "${out}${err}" MATCHES "${expect}")
        message(FATAL_ERROR "${what}: expected '${expect}', exit ${status}\n${out}\n${err}")
    endif()
endfunction()

foreach(library_model own rc)
    set(dir "${WORK_DIR}/${library_model}")
    file(REMOVE_RECURSE "${dir}")
    file(MAKE_DIRECTORY "${dir}")

    compile("${dir}" "--emit=dll -mm=${library_model} export_own_no_drops" ok
        "${TSLANG}" --emit=dll -mm=${library_model} --no-default-lib
        "--llvm-lib-path=${LLVM_LIB}" "--tslang-lib-path=${TSLANG_LIB}"
        "${SOURCE_DIR}/export_own_no_drops.ts" -o export_own_no_drops.dll)
endforeach()

# `import './export_own_no_drops'` finds the DLL in the working directory before the source
compile("${WORK_DIR}/own" "import_own_no_drops against the own library" ok
    "${TSLANG}" --emit=obj -mm=own --no-default-lib "${SOURCE_DIR}/import_own_no_drops.ts" -o import.obj)

compile("${WORK_DIR}/own" "import_own_err_imported_drops against the own library" "${borrow_ended}"
    "${TSLANG}" --emit=obj -mm=own --no-default-lib "${SOURCE_DIR}/import_own_err_imported_drops.ts" -o import.obj)

compile("${WORK_DIR}/rc" "import_own_no_drops against the rc library" "${borrow_ended}"
    "${TSLANG}" --emit=obj -mm=own --no-default-lib "${SOURCE_DIR}/import_own_no_drops.ts" -o import.obj)

message(STATUS "an own library's __own_no_drops reaches its importer, and only what it lists")
