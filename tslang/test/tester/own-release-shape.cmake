# The release under own (own rc fallback design, section 2) tests the header word against 0 first:
# 0 is one owner and no WeakRef, the common case, freed at once; any other word (immortal, extra
# owners kept by --own-fallback=count, WeakRefs) takes the slow path. Read in the release routine
# of class `Node` (the routine that addresses a `%Node` field) of FILE, compiled to LLVM IR under
# -mm=own without the flag: its first `icmp` on an i64 compares the word with 0.
cmake_minimum_required(VERSION 3.17.3)
get_filename_component(outName "${FILE}" NAME_WE)
set(llFile "${CMAKE_CURRENT_BINARY_DIR}/${outName}.release-shape.ll")
execute_process(COMMAND "${TSLANG}" --emit=llvm --no-default-lib -mm=own "${FILE}" -o "${llFile}"
                OUTPUT_VARIABLE out ERROR_VARIABLE err RESULT_VARIABLE result)
if(NOT result EQUAL 0)
    message(FATAL_ERROR "tslang failed (${result}):\n${out}${err}")
endif()
file(READ "${llFile}" ir)
# out of a CMake list's way: `;` separates its items, and `[` `]` keep a `;` from separating
string(REPLACE ";" "," ir "${ir}")
string(REPLACE "[" "(" ir "${ir}")
string(REPLACE "]" ")" ir "${ir}")

# each function definition, up to the next one
string(REGEX MATCHALL "define [^\n]*@tsrel_[0-9]+\\(ptr %0\\) [^\n]*\n([^}][^\n]*\n|\n)*}" routines "${ir}")
set(found FALSE)
foreach(routine IN LISTS routines)
    string(FIND "${routine}" "getelementptr %Node, ptr" nodeField)
    if(nodeField EQUAL -1)
        continue()
    endif()
    set(found TRUE)
    string(REGEX MATCH "icmp [a-z]+ i64 [^\n]*" firstCompare "${routine}")
    if(NOT firstCompare MATCHES "^icmp eq i64 %[0-9a-z.]+, 0$")
        message(FATAL_ERROR "the release of Node first compares its header word with `${firstCompare}`, not `eq ..., 0`:\n${routine}")
    endif()
endforeach()
if(NOT found)
    message(FATAL_ERROR "no release routine of Node in:\n${ir}")
endif()
