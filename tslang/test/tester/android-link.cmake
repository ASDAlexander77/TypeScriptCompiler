# Compiling and linking for Android (-mtriple=<arch>-linux-android<api>).
#
# MODE=codegen  runs everywhere, with no NDK:
#               - objects for Android are position independent without -relocation-model=pic
#                 (Android executables must be PIE): a constant's address is RIP-relative, where
#                 the static default for x86_64 Linux takes it as an absolute immediate;
#               - a triple without an API level, and no NDK, are refused with an error that says so.
# MODE=link     links an executable and a shared library for x86_64 Android with the real default
#               library, collector and async runtime, through the NDK. Prints SKIPPED (CTest then
#               reports it skipped) unless ANDROID_NDK_HOME and the Android builds of those are
#               there: scripts/build_gc_release_android.bat, scripts/build_tslang_runtime_release_
#               android.bat, and TypeScriptCompilerDefaultLib's scripts/build_android.bat (its
#               __build next to this repository, or TSLANG_ANDROID_DEFAULT_LIB_PATH). A shared library
#               links with --no-undefined, so a symbol Bionic lacks fails here, not in the app.

cmake_minimum_required(VERSION 3.17.3)

foreach(var MODE TSLANG WORK_DIR REPO_DIR)
    if(NOT DEFINED ${var})
        message(FATAL_ERROR "${var} is required")
    endif()
endforeach()

file(REMOVE_RECURSE "${WORK_DIR}")
file(MAKE_DIRECTORY "${WORK_DIR}")

file(WRITE "${WORK_DIR}/app.ts" [=[
class Acc {
    total = 0;
    add(n: number) { this.total += n; return this; }
}

export function sum3(a: number, b: number, c: number): number {
    const acc = new Acc().add(a).add(b).add(c);
    assert(acc.total >= 0, "negative");
    print(`sum3 = ${acc.total}`);
    return acc.total;
}

function main() {
    print(sum3(1, 2, 3));
}
]=])

# run(<what> <expect-success> <command...>) - leaves the output in run_output
function(run what expect_success)
    execute_process(COMMAND ${ARGN}
        WORKING_DIRECTORY "${WORK_DIR}"
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

if(MODE STREQUAL "codegen")
    foreach(triple x86_64-linux-android29 x86_64-unknown-linux-gnu)
        run("--emit=asm for ${triple}" TRUE
            "${TSLANG}" --emit=asm --opt -mm=none --no-default-lib -mtriple=${triple} app.ts -o ${triple}.s)
        file(READ "${WORK_DIR}/${triple}.s" asm)
        if(asm MATCHES "\\$frmt_")
            set(${triple}_absolute TRUE)
        endif()
    endforeach()
    if(x86_64-linux-android29_absolute)
        file(READ "${WORK_DIR}/x86_64-linux-android29.s" asm)
        message(FATAL_ERROR "the code for Android takes an absolute address, it is not position independent:\n${asm}")
    endif()
    if(NOT x86_64-unknown-linux-gnu_absolute)
        message(FATAL_ERROR "the check cannot tell: x86_64 Linux code without -relocation-model=pic takes no absolute address either")
    endif()

    run("--emit=exe for Android with no API level" FALSE
        "${TSLANG}" --emit=exe -mm=none --no-default-lib -mtriple=x86_64-linux-android app.ts -o app)
    if(NOT run_output MATCHES "an Android target needs its API level in the triple, e.g. -mtriple=x86_64-linux-android29")
        message(FATAL_ERROR "a triple without an API level failed for another reason:\n${run_output}")
    endif()

    set(ENV{ANDROID_NDK_HOME} "")
    run("--emit=exe for Android with no NDK" FALSE
        "${TSLANG}" --emit=exe -mm=none --no-default-lib -mtriple=x86_64-linux-android29 app.ts -o app)
    if(NOT run_output MATCHES "linking for Android needs the Android NDK: pass --android-ndk-path or set ANDROID_NDK_HOME")
        message(FATAL_ERROR "linking with no NDK failed for another reason:\n${run_output}")
    endif()

    message(STATUS "Android code is position independent; a missing API level or NDK is refused")
elseif(MODE STREQUAL "link")
    set(abi x86_64)
    set(triple x86_64-linux-android29)
    set(gc_lib "${REPO_DIR}/3rdParty/gc/android/${abi}/release/lib")
    set(runtime_lib "${REPO_DIR}/__build/tslang-runtime/release/android/${abi}")
    if(DEFINED ENV{TSLANG_ANDROID_DEFAULT_LIB_PATH})
        set(default_lib "$ENV{TSLANG_ANDROID_DEFAULT_LIB_PATH}")
    else()
        set(default_lib "${REPO_DIR}/../TypeScriptCompilerDefaultLib/__build")
    endif()
    # the one default-library tree, every target in it; tslang names this one's folder
    execute_process(COMMAND "${TSLANG}" --print-default-lib-dir=lib -mm=gc -mtriple=${triple}
        OUTPUT_VARIABLE default_lib_subdir
        OUTPUT_STRIP_TRAILING_WHITESPACE
        COMMAND_ERROR_IS_FATAL ANY)

    if("$ENV{ANDROID_NDK_HOME}" STREQUAL "")
        message("SKIPPED: ANDROID_NDK_HOME is not set")
        return()
    endif()
    foreach(file "${gc_lib}/libgc.a" "${runtime_lib}/libTypeScriptAsyncRuntime.a"
                 "${default_lib}/${default_lib_subdir}/libTypeScriptDefaultLib.a")
        if(NOT EXISTS "${file}")
            message("SKIPPED: ${file} is not built")
            return()
        endif()
    endforeach()

    set(libs "--default-lib-path=${default_lib}" "--gc-lib-path=${gc_lib}" "--tslang-lib-path=${runtime_lib}")
    run("--emit=exe for ${triple}" TRUE
        "${TSLANG}" --emit=exe --opt -mm=gc -mtriple=${triple} ${libs} app.ts -o app)
    run("--emit=dll for ${triple}" TRUE
        "${TSLANG}" --emit=dll --opt -mm=gc -mtriple=${triple} ${libs} app.ts -o libapp.so)

    foreach(binary app libapp.so)
        file(READ "${WORK_DIR}/${binary}" magic LIMIT 4 HEX)
        if(NOT magic STREQUAL "7f454c46")
            message(FATAL_ERROR "${binary} is not an ELF file")
        endif()
    endforeach()

    message(STATUS "an executable and a shared library linked for ${triple}")
else()
    message(FATAL_ERROR "MODE must be codegen or link, not '${MODE}'")
endif()
