# The default library's layout, as tslang --print-default-lib-dir names it: the build scripts of
# TypeScriptCompilerDefaultLib stage into what it prints and tslang links from the same function
# (getDefaultLibSubDir), so a change to the layout shows up here, as a changed literal, before a
# default library built one way is looked for the other way.

cmake_minimum_required(VERSION 3.17.3)

if(NOT DEFINED TSLANG)
    message(FATAL_ERROR "TSLANG is required")
endif()

# check(<expected> <tslang arguments...>)
function(check expected)
    execute_process(COMMAND "${TSLANG}" ${ARGN}
        OUTPUT_VARIABLE out
        ERROR_VARIABLE err
        OUTPUT_STRIP_TRAILING_WHITESPACE
        RESULT_VARIABLE status)
    if(NOT status EQUAL 0 OR NOT out STREQUAL "${expected}")
        message(FATAL_ERROR "tslang ${ARGN}: exit ${status}\nexpected: ${expected}\n     got: ${out}\n${err}")
    endif()
endfunction()

check("defaultlib/lib/x86_64/pc/windows/msvc/release/gc" --print-default-lib-dir=lib -mtriple=x86_64-pc-windows-msvc)
check("defaultlib/dll/x86_64/pc/windows/msvc/debug/rc" --print-default-lib-dir=dll --di -mm=rc -mtriple=x86_64-pc-windows-msvc)
check("defaultlib/lib/x86_64/unknown/linux/gnu/release/none" --print-default-lib-dir=lib -mm=none -mtriple=x86_64-unknown-linux-gnu)
check("defaultlib/lib/aarch64/unknown/linux/android/release/gc" --print-default-lib-dir=lib -mtriple=aarch64-linux-android29)

# the spellings of one target share its tree: the canonical arch name, no API level or version
check("defaultlib/lib/i386/pc/windows/msvc/release/gc" --print-default-lib-dir=lib -mtriple=i686-pc-windows-msvc)
check("defaultlib/lib/i386/pc/windows/msvc/release/gc" --print-default-lib-dir=lib -mtriple=i386-pc-windows-msvc19.40.0)
check("defaultlib/lib/x86_64/unknown/linux/android/release/gc" --print-default-lib-dir=lib -mtriple=x86_64-linux-android34)

message(STATUS "--print-default-lib-dir names the triple/build/model layout")
