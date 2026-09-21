# Importing a DLL when compiling for a target other than the host (Windows).
#
# tslang cannot load such a DLL into its own process, so it reads the declarations the DLL
# exports (__decls_*) from the file instead (Dump::readExportedCString). The x86 twins
# (TSLANG_TEST_X86) cover that path for i686, but they are off by default. This test covers it in
# the default suite, with no x86 libraries: the DLL is an ordinary x64 one, and the importer is
# compiled for x86_64 Linux. That target has the host's arch but another OS, so the DLL is read as
# a file, as a PE32+ image. The importer is only compiled, never linked or run.
#
# MODE=read    the declarations are read: the importer compiles, and uses what they declare.
# MODE=errors  a file that is not a PE image, and a DLL for another machine, are refused with an
#              error that says so.
#
# -mm=none keeps the collector out of it: the DLL then links no gc.dll.

cmake_minimum_required(VERSION 3.17.3)

foreach(var MODE TSLANG TESTS_DIR WORK_DIR LLVM_LIB TSLANG_LIB)
    if(NOT DEFINED ${var})
        message(FATAL_ERROR "${var} is required")
    endif()
endforeach()

set(ENV{GC_LIB_PATH} "")
set(ENV{TSLANG_LIB_PATH} "")

file(REMOVE_RECURSE "${WORK_DIR}")
file(MAKE_DIRECTORY "${WORK_DIR}")

set(common -mm=none --no-default-lib)

# run(<what> <dir> <expect-success> <command...>) - leaves the output in run_output
function(run what dir expect_success)
    execute_process(COMMAND ${ARGN}
        WORKING_DIRECTORY "${dir}"
        OUTPUT_VARIABLE out
        ERROR_VARIABLE err
        RESULT_VARIABLE status)
    set(run_output "${out}${err}" PARENT_SCOPE)
    if(expect_success AND NOT status EQUAL 0)
        message(FATAL_ERROR "${what}: exit ${status}\n${out}\n${err}")
    endif()
    if(NOT expect_success AND status EQUAL 0)
        message(FATAL_ERROR "${what}: expected to fail, but succeeded\n${out}\n${err}")
    endif()
endfunction()

set(library "${TESTS_DIR}/shared.ts")
set(program "${TESTS_DIR}/use_shared.ts")
set(foreign -mtriple=x86_64-unknown-linux-gnu)

if(MODE STREQUAL "read")
    run("--emit=dll" "${WORK_DIR}" TRUE
        "${TSLANG}" --emit=dll ${common} "--llvm-lib-path=${LLVM_LIB}" "--tslang-lib-path=${TSLANG_LIB}"
        "${library}" -o shared.dll)

    # use_shared.ts calls test1/test2 and prints val_str, which only the DLL's declarations
    # declare: without them it does not compile ("can't resolve name").
    run("--emit=obj for x86_64 Linux, importing an x64 DLL" "${WORK_DIR}" TRUE
        "${TSLANG}" --emit=obj ${common} ${foreign} "${program}" -o use_shared.o)
    if(run_output MATCHES "missing information about shared library")
        message(FATAL_ERROR "the DLL's declarations were not found:\n${run_output}")
    endif()

    # The program looks the imported symbols up by name at run time, so their names are in it.
    run("--emit=llvm for x86_64 Linux, importing an x64 DLL" "${WORK_DIR}" TRUE
        "${TSLANG}" --emit=llvm ${common} ${foreign} "${program}" -o use_shared.ll)
    file(READ "${WORK_DIR}/use_shared.ll" ir)
    foreach(name test1 test2 val_str)
        if(NOT ir MATCHES "${name}\\\\00\"")
            message(FATAL_ERROR "the program does not look up '${name}', which the DLL declares:\n${ir}")
        endif()
    endforeach()

    message(STATUS "declarations read from an x64 DLL's file for a foreign target")
elseif(MODE STREQUAL "errors")
    # 1. A file that is not a PE image.
    file(MAKE_DIRECTORY "${WORK_DIR}/not-pe")
    file(WRITE "${WORK_DIR}/not-pe/foo.dll" "not a DLL\n")
    file(WRITE "${WORK_DIR}/not-pe/main.ts" "import './foo'\n\nfunction main() {\n    print(\"done.\");\n}\n")
    run("importing a non-PE file for a foreign target" "${WORK_DIR}/not-pe" FALSE
        "${TSLANG}" --emit=obj ${common} ${foreign} main.ts -o main.o)
    if(NOT run_output MATCHES "cannot read declarations from '[^']*foo\\.dll': for a target other than the host, only PE DLLs can be imported")
        message(FATAL_ERROR "importing a non-PE file failed for another reason:\n${run_output}")
    endif()

    # 2. An x64 DLL imported into an x86 program.
    file(MAKE_DIRECTORY "${WORK_DIR}/machine")
    run("--emit=dll" "${WORK_DIR}/machine" TRUE
        "${TSLANG}" --emit=dll ${common} "--llvm-lib-path=${LLVM_LIB}" "--tslang-lib-path=${TSLANG_LIB}"
        "${library}" -o shared.dll)
    run("importing an x64 DLL into an x86 program" "${WORK_DIR}/machine" FALSE
        "${TSLANG}" --emit=obj ${common} -mtriple=i686-pc-windows-msvc "${program}" -o use_shared.obj)
    if(NOT run_output MATCHES "shared library '[^']*shared\\.dll' is built for x64, but this program targets x86")
        message(FATAL_ERROR "importing an x64 DLL into an x86 program failed for another reason:\n${run_output}")
    endif()

    message(STATUS "a non-PE file and a DLL for another machine are refused for a foreign target")
else()
    message(FATAL_ERROR "MODE must be read or errors, not '${MODE}'")
endif()
