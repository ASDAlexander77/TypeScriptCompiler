# --jit-native (Windows): `tslang --emit=jit` links the JIT cache's objects into one DLL with its PDB
# and loads it with the system loader, so a debugger sees the program as a module with symbols and
# source. See docs/jit-native-dll-debugging-design.md.
#
# 1. A program importing a module runs, and leaves one DLL and its PDB in __jit.
# 2. The second run uses that DLL: it is not linked again.
# 3. The imported module changes: the program runs the new code, from a new DLL, and the old one is
#    removed.
# 4. The program's exit code, an uncaught exception and a failing assert come out as they do from
#    the in-memory JIT: the DLL has a C runtime of its own, whose output tslang has to flush and
#    whose assert would otherwise put up a dialog.
# 5. Without the JIT cache there is nothing to link: an error, not a run.
# 6. If cdb is there (CDB), a breakpoint in the imported module is hit, at its source line.
#
# This drives `tslang` itself, which `test-runner` does not: it links nothing for the JIT. -mm=rc and
# --no-default-lib keep the link to the compiler's own libraries.

cmake_minimum_required(VERSION 3.17.3)

foreach(var TSLANG SOURCE_DIR WORK_DIR LLVM_LIB TSLANG_LIB OPT)
    if(NOT DEFINED ${var})
        message(FATAL_ERROR "${var} is required")
    endif()
endforeach()

file(REMOVE_RECURSE "${WORK_DIR}")
file(MAKE_DIRECTORY "${WORK_DIR}")
file(COPY "${SOURCE_DIR}/greeting.ts" "${SOURCE_DIR}/greeting_module.ts" DESTINATION "${WORK_DIR}")
file(WRITE "${WORK_DIR}/exit_code.ts" "function main() {\n    print(\"before\");\n    return 7;\n}\n")
file(WRITE "${WORK_DIR}/uncaught.ts" "function main() {\n    print(\"before\");\n    throw \"boom\";\n}\n")
file(WRITE "${WORK_DIR}/assert_fails.ts" "function main() {\n    print(\"before\");\n    assert(1 == 2, \"one is not two\");\n    print(\"after\");\n}\n")

set(tslang_args --emit=jit --jit-native -mm=rc --no-default-lib ${OPT}
                "--llvm-lib-path=${LLVM_LIB}" "--tslang-lib-path=${TSLANG_LIB}")

# run(<source> <exit code> <stdout> <stderr regex>) - fails the test unless the program ends with that
# exit code, prints exactly that, and its diagnostics match (an empty regex matches anything)
function(run source expected_status expected_out expected_err)
    execute_process(COMMAND "${TSLANG}" ${tslang_args} ${source}
        WORKING_DIRECTORY "${WORK_DIR}"
        OUTPUT_VARIABLE out
        ERROR_VARIABLE err
        RESULT_VARIABLE status)
    string(STRIP "${out}" out)
    if(NOT status EQUAL expected_status OR NOT out STREQUAL expected_out OR NOT err MATCHES "${expected_err}")
        message(FATAL_ERROR "${source}: expected exit ${expected_status} and '${expected_out}', got exit ${status}\n"
                            "stdout:\n${out}\nstderr:\n${err}")
    endif()
endfunction()

# the program's DLLs in the cache, and when each was written; each must have its PDB
function(native_dlls result source)
    file(GLOB dlls "${WORK_DIR}/__jit/${source}.*.dll")
    list(SORT dlls)
    set(stamps "")
    foreach(dll ${dlls})
        string(REGEX REPLACE "\\.dll$" ".pdb" pdb "${dll}")
        if(NOT EXISTS "${pdb}")
            message(FATAL_ERROR "${dll} has no PDB beside it")
        endif()
        file(TIMESTAMP "${dll}" stamp "%Y%m%d%H%M%S")
        list(APPEND stamps "${dll}@${stamp}")
    endforeach()
    set(${result} "${stamps}" PARENT_SCOPE)
endfunction()

run(greeting.ts 0 "one" "")
native_dlls(first greeting.ts)
list(LENGTH first count)
if(NOT count EQUAL 1)
    message(FATAL_ERROR "expected one DLL for greeting.ts in __jit, found: ${first}")
endif()

# a second apart, so a DLL linked again has another time
execute_process(COMMAND "${CMAKE_COMMAND}" -E sleep 1.1)

run(greeting.ts 0 "one" "")
native_dlls(second greeting.ts)
if(NOT first STREQUAL second)
    message(FATAL_ERROR "the second run linked again instead of using the DLL:\n${first}\n${second}")
endif()

file(WRITE "${WORK_DIR}/greeting_module.ts" "export function greeting() {\n    return \"two\";\n}\n")

run(greeting.ts 0 "two" "")
native_dlls(third greeting.ts)
list(LENGTH third count)
if(NOT count EQUAL 1 OR third STREQUAL second)
    message(FATAL_ERROR "expected the old DLL replaced by a new one after greeting_module.ts changed:\n${second}\n${third}")
endif()

run(exit_code.ts 7 "before" "")
run(uncaught.ts 1 "before" "uncaught exception")
run(assert_fails.ts 3 "before" "Assertion failed: one is not two")

execute_process(COMMAND "${TSLANG}" ${tslang_args} --jit-cache=false greeting.ts
    WORKING_DIRECTORY "${WORK_DIR}"
    OUTPUT_VARIABLE out
    ERROR_VARIABLE err
    RESULT_VARIABLE status)
if(status EQUAL 0 OR NOT err MATCHES "--jit-native links the objects of the JIT cache")
    message(FATAL_ERROR "--jit-native --jit-cache=false: expected an error, got exit ${status}\n${out}\n${err}")
endif()

if(DEFINED CDB AND EXISTS "${CDB}")
    # The DLL linked above, by name: the debugger stops when it loads, loads its PDB, and then stops
    # in the imported module's `greeting`, whose frame the PDB places on line 1 of
    # greeting_module.ts. A breakpoint on the line itself would be ambiguous - the module's
    # initializer is on line 1 too - and optimized, `greeting` has no code on any other. dbgeng
    # names the module after the file with its dots made underscores. Symbols only from __jit, so
    # nothing is fetched from a symbol server. `q` ends tslang too.
    list(GET third 0 dll)
    string(REGEX REPLACE "@.*$" "" dll "${dll}")
    get_filename_component(dll_name "${dll}" NAME)
    string(REGEX REPLACE "\\.dll$" "" module "${dll_name}")
    string(REPLACE "." "_" module "${module}")
    execute_process(COMMAND "${CDB}" -y "${WORK_DIR}/__jit"
                            -c "sxe ld:${dll_name}; g; .lines -e; ld ${module}; bu ${module}!greeting; g; kn 3; q"
                            "${TSLANG}" ${tslang_args} greeting.ts
        WORKING_DIRECTORY "${WORK_DIR}"
        OUTPUT_VARIABLE out
        ERROR_VARIABLE err
        RESULT_VARIABLE status
        TIMEOUT 120)
    if(NOT out MATCHES "Breakpoint 0 hit" OR NOT out MATCHES "${module}!greeting [^\n]*greeting_module\\.ts @ 1")
        message(FATAL_ERROR "cdb did not stop in ${module}!greeting at greeting_module.ts line 1:\n${out}\n${err}")
    endif()
    message(STATUS "cdb stops in a function of the imported module, at its source line")
endif()

message(STATUS "--jit-native links the JIT cache into a DLL with a PDB, reuses it, and runs the program as the JIT does")
