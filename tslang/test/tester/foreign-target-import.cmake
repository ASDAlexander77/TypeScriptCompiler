# Importing a shared library when compiling for a target other than the host (Windows).
#
# tslang cannot load such a library into its own process. It never needs to: what a library
# declares is in the file tslang wrote beside it, `<library stem>.d.ts` (TypeScript/
# LibraryDeclarations.h), read the same for every target. The x86 twins (TSLANG_TEST_X86) cover
# i686, but they are off by default. This test covers foreign targets in the default suite, with no
# x86 or Android libraries:
#
#   - an ordinary x64 DLL imported by a program compiled for x86_64 Linux (the host's arch, another
#     OS). An import is looked for under the target's name for a library, which is also what the
#     program loads at run time: `import './shared'` is ./libshared.so there, so the x64 DLL is
#     written as libshared.so; it is still a PE image inside.
#   - an Android library: an aarch64 ELF object of the library's module stands in for the `.so`
#     (linking one needs the NDK), with the declarations --emit=obj wrote beside it. Before the
#     declarations moved out of the binary, a library that is not a PE image could not be imported
#     on Windows at all.
#
# The importers are only compiled, never linked or run.
#
# MODE=read    the declarations are read: the importer compiles, and uses what they declare.
# MODE=errors  a library with no declarations beside it, declarations that are not tslang's, the
#              declarations of a library that is not there, and a DLL for another machine are
#              refused with an error that says so.
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

# check_imports(<what> <dir> <triple> <library file>) - use_shared.ts calls test1/test2 and prints
# val_str, which only the library's declarations declare: without them it does not compile ("can't
# resolve name"). The program looks them up by name at run time, so their names are in it, and it
# loads the library under the target's name for it.
function(check_imports what dir triple library_file)
    run("--emit=llvm for ${what}" "${dir}" TRUE
        "${TSLANG}" --emit=llvm ${common} -mtriple=${triple} "${program}" -o use_shared.ll)
    file(READ "${dir}/use_shared.ll" ir)
    foreach(name test1 test2 val_str)
        if(NOT ir MATCHES "${name}\\\\00\"")
            message(FATAL_ERROR "${what}: the program does not look up '${name}', which the library declares:\n${ir}")
        endif()
    endforeach()

    string(REPLACE "." "\\." library_pattern "${library_file}")
    if(NOT ir MATCHES "${library_pattern}\\\\00\"")
        message(FATAL_ERROR "${what}: the program does not load ${library_file}:\n${ir}")
    endif()

    run("--emit=obj for ${what}" "${dir}" TRUE
        "${TSLANG}" --emit=obj ${common} -mtriple=${triple} "${program}" -o use_shared.o)
    message(STATUS "${what}: declarations read")
endfunction()

if(MODE STREQUAL "read")
    # 1. An x64 DLL, for x86_64 Linux.
    set(dir "${WORK_DIR}/pe")
    file(MAKE_DIRECTORY "${dir}")
    run("--emit=dll" "${dir}" TRUE
        "${TSLANG}" --emit=dll ${common} "--llvm-lib-path=${LLVM_LIB}" "--tslang-lib-path=${TSLANG_LIB}"
        "${library}" -o libshared.so)
    if(NOT EXISTS "${dir}/libshared.d.ts")
        message(FATAL_ERROR "--emit=dll wrote no declarations beside libshared.so")
    endif()
    check_imports("x86_64 Linux importing an x64 DLL" "${dir}" x86_64-unknown-linux-gnu libshared.so)

    # 2. An Android library, an ELF image: its object stands in for it, its declarations beside it.
    set(dir "${WORK_DIR}/android")
    file(MAKE_DIRECTORY "${dir}")
    run("--emit=obj for Android" "${dir}" TRUE
        "${TSLANG}" --emit=obj ${common} -mtriple=aarch64-linux-android29 -relocation-model=pic
        "${library}" -o shared.o)
    if(NOT EXISTS "${dir}/shared.d.ts")
        message(FATAL_ERROR "--emit=obj wrote no declarations beside shared.o")
    endif()
    file(RENAME "${dir}/shared.o" "${dir}/libshared.so")
    file(RENAME "${dir}/shared.d.ts" "${dir}/libshared.d.ts")
    check_imports("Android importing an Android library" "${dir}" aarch64-linux-android29 libshared.so)
elseif(MODE STREQUAL "errors")
    # 1. A library with no declarations beside it (one built before they moved out of the binary).
    file(MAKE_DIRECTORY "${WORK_DIR}/no-decls")
    file(WRITE "${WORK_DIR}/no-decls/libfoo.so" "not a library\n")
    file(WRITE "${WORK_DIR}/no-decls/main.ts" "import './foo'\n\nfunction main() {\n    print(\"done.\");\n}\n")
    run("importing a library with no declarations beside it" "${WORK_DIR}/no-decls" FALSE
        "${TSLANG}" --emit=obj ${common} ${foreign} main.ts -o main.o)
    if(NOT run_output MATCHES "shared library '[^']*libfoo\\.so' has no declarations beside it \\('[^']*libfoo\\.d\\.ts'\\)")
        message(FATAL_ERROR "importing a library with no declarations failed for another reason:\n${run_output}")
    endif()

    # 2. A .d.ts beside it that tslang did not write.
    file(WRITE "${WORK_DIR}/no-decls/libfoo.d.ts" "declare function foo(): void;\n")
    run("importing a library beside a hand-written .d.ts" "${WORK_DIR}/no-decls" FALSE
        "${TSLANG}" --emit=obj ${common} ${foreign} main.ts -o main.o)
    if(NOT run_output MATCHES "'[^']*libfoo\\.d\\.ts' is not the declarations of shared library")
        message(FATAL_ERROR "importing a library beside a hand-written .d.ts failed for another reason:\n${run_output}")
    endif()

    # 3. An x64 DLL imported into an x86 program.
    file(MAKE_DIRECTORY "${WORK_DIR}/machine")
    run("--emit=dll" "${WORK_DIR}/machine" TRUE
        "${TSLANG}" --emit=dll ${common} "--llvm-lib-path=${LLVM_LIB}" "--tslang-lib-path=${TSLANG_LIB}"
        "${library}" -o shared.dll)
    run("importing an x64 DLL into an x86 program" "${WORK_DIR}/machine" FALSE
        "${TSLANG}" --emit=obj ${common} -mtriple=i686-pc-windows-msvc "${program}" -o use_shared.obj)
    if(NOT run_output MATCHES "shared library '[^']*shared\\.dll' is built for x64, but this program targets x86")
        message(FATAL_ERROR "importing an x64 DLL into an x86 program failed for another reason:\n${run_output}")
    endif()

    # 4. A library's declarations read as a source file (`import './shared'` looks for shared.ts,
    #    but a reference names the file itself): they would declare functions nothing defines and
    #    nothing loads.
    file(REMOVE "${WORK_DIR}/machine/shared.dll")
    file(WRITE "${WORK_DIR}/machine/main.ts" "/// <reference path=\"shared.d.ts\" />\n\nfunction main() {\n    test1();\n    print(\"done.\");\n}\n")
    run("importing the declarations of a missing DLL" "${WORK_DIR}/machine" FALSE
        "${TSLANG}" --emit=obj ${common} -mtriple=x86_64-pc-windows-msvc main.ts -o main.obj)
    if(NOT run_output MATCHES "holds the declarations of a shared library, which `import` reads beside the library; the library itself is not there")
        message(FATAL_ERROR "importing the declarations of a missing DLL failed for another reason:\n${run_output}")
    endif()

    message(STATUS "libraries with no, foreign or orphaned declarations, and a DLL for another machine, are refused")
else()
    message(FATAL_ERROR "MODE must be read or errors, not '${MODE}'")
endif()
