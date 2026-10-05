# The C runtime a program calls into is the target's, not the host's.
#
# A failing assert, a number turned into a string and a class's comdat were once chosen with
# `#ifdef WIN32` in the compiler, so a Windows-hosted tslang gave an Android or Linux object the
# MSVC CRT's `_assert`/`sprintf_s` and an `exactmatch` comdat, which ELF cannot hold. Each triple
# here is compiled to LLVM IR only (never linked or run), from any host, and checked for its own:
#
#   x86_64-pc-windows-msvc    _assert(msg, file, line), sprintf_s, comdat exactmatch
#   x86_64-unknown-linux-gnu  __assert_fail(msg, file, line, func), snprintf, comdat any
#   x86_64-linux-android      __assert(file, line, msg) - Bionic has no __assert_fail -, snprintf,
#                             comdat any

cmake_minimum_required(VERSION 3.17.3)

foreach(var TSLANG WORK_DIR)
    if(NOT DEFINED ${var})
        message(FATAL_ERROR "${var} is required")
    endif()
endforeach()

file(REMOVE_RECURSE "${WORK_DIR}")
file(MAKE_DIRECTORY "${WORK_DIR}")

file(WRITE "${WORK_DIR}/crt.ts" [=[
class Foo {
    x = 1;
}

function main() {
    const f = new Foo();
    const n = 3.5;
    const s = "v" + n;
    assert(s.length > f.x, "too short");
    print(s);
}
]=])

# check(<triple> <assert decl regex> <sprintf name> <comdat kind> <names that must not be declared...>)
function(check triple assert_decl sprintf_name comdat_kind)
    execute_process(COMMAND "${TSLANG}" --emit=llvm -mm=none --no-default-lib -mtriple=${triple} crt.ts -o ${triple}.ll
        WORKING_DIRECTORY "${WORK_DIR}"
        OUTPUT_VARIABLE out
        ERROR_VARIABLE err
        RESULT_VARIABLE status)
    if(NOT status EQUAL 0)
        message(FATAL_ERROR "--emit=llvm for ${triple}: exit ${status}\n${out}\n${err}")
    endif()

    file(READ "${WORK_DIR}/${triple}.ll" ir)
    if(NOT ir MATCHES "declare void @${assert_decl}")
        message(FATAL_ERROR "${triple}: a failing assert does not call '${assert_decl}':\n${ir}")
    endif()
    if(NOT ir MATCHES "declare i32 @${sprintf_name}\\(")
        message(FATAL_ERROR "${triple}: a number is not turned into a string with '${sprintf_name}':\n${ir}")
    endif()
    if(NOT ir MATCHES "\\$Foo\\.\\.size = comdat ${comdat_kind}")
        message(FATAL_ERROR "${triple}: the class's comdat is not '${comdat_kind}':\n${ir}")
    endif()
    foreach(name ${ARGN})
        if(ir MATCHES "declare [^\n]*@${name}\\(")
            message(FATAL_ERROR "${triple}: declares '${name}', which is another CRT's:\n${ir}")
        endif()
    endforeach()
endfunction()

check(x86_64-pc-windows-msvc "_assert\\(ptr, ptr, i32\\)" sprintf_s exactmatch __assert_fail __assert snprintf)
check(x86_64-unknown-linux-gnu "__assert_fail\\(ptr, ptr, i32, ptr\\)" snprintf any _assert __assert sprintf_s)
check(x86_64-linux-android "__assert\\(ptr, i32, ptr\\)" snprintf any _assert __assert_fail sprintf_s)

message(STATUS "assert, number-to-string and class comdats follow the target's CRT")
