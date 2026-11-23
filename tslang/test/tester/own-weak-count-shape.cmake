# A WeakRef's count on its target's header word under own (Defines.h, OwnershipRoutineLogic.h
# emitWeakIncRef / emitWeakDecRef): an increment that would overflow the weak count makes a live
# target immortal, but leaves a tombstone's word as it is (immortal would read as alive to deref(),
# with the fields released), and a decrement leaves a tombstone whose count is full as it is (the
# count stuck there may be short of the WeakRefs left: the tombstone leaks, never is freed under one).
# A run cannot reach 2^31 (64-bit) or 2^15 (32-bit) WeakRefs, so FILE is read as LLVM IR, for the
# host (64-bit) and for wasm32 (32-bit word): each routine that tests the weak count for full
#   - and adds one: picks the word itself, not immortal (-1), when the dead bit is set;
#   - and takes one: keeps the word when the count is full and the dead bit is set.
cmake_minimum_required(VERSION 3.17.3)
get_filename_component(outName "${FILE}" NAME_WE)

function(check_width triple width mask dead)
    set(llFile "${CMAKE_CURRENT_BINARY_DIR}/${outName}.weak-count-${width}.ll")
    set(tripleArg)
    if(NOT triple STREQUAL "")
        set(tripleArg "--mtriple=${triple}")
    endif()
    execute_process(COMMAND "${TSLANG}" --emit=llvm --no-default-lib -mm=own ${tripleArg} "${FILE}" -o "${llFile}"
                    OUTPUT_VARIABLE out ERROR_VARIABLE err RESULT_VARIABLE result)
    if(NOT result EQUAL 0)
        message(FATAL_ERROR "tslang failed (${result}):\n${out}${err}")
    endif()
    file(READ "${llFile}" ir)
    string(REPLACE ";" "," ir "${ir}")
    string(REPLACE "[" "(" ir "${ir}")
    string(REPLACE "]" ")" ir "${ir}")

    string(REGEX MATCHALL "define [^\n]*\n([^}][^\n]*\n|\n)*}" routines "${ir}")
    set(increments 0)
    set(decrements 0)
    foreach(routine IN LISTS routines)
        if(NOT routine MATCHES "icmp eq i${width} %[0-9]+, ${mask}\n")
            continue()
        endif()
        if(NOT routine MATCHES "and i${width} %[0-9]+, ${dead}\n")
            message(FATAL_ERROR "a weak count test without the dead bit (${width}-bit):\n${routine}")
        endif()
        if(routine MATCHES "add i${width} %[0-9]+, 1\n")
            math(EXPR increments "${increments} + 1")
            # the overflow: the word on a tombstone, immortal (-1) otherwise - never immortal first
            if(NOT routine MATCHES "select i1 %[0-9]+, i${width} %[0-9]+, i${width} -1\n")
                message(FATAL_ERROR "the weak increment does not keep a tombstone's word (${width}-bit):\n${routine}")
            endif()
            if(routine MATCHES "select i1 %[0-9]+, i${width} -1, ")
                message(FATAL_ERROR "the weak increment makes the target immortal whatever it is (${width}-bit):\n${routine}")
            endif()
        endif()
        if(routine MATCHES "sub i${width} %[0-9]+, 1\n")
            math(EXPR decrements "${decrements} + 1")
            if(NOT routine MATCHES "and i1 %[0-9]+, %[0-9]+\n[^\n]*sub i${width} %[0-9]+, 1\n[^\n]*select i1 %[0-9]+, i${width} %[0-9]+, i${width} %[0-9]+\n")
                message(FATAL_ERROR "the weak decrement takes a full tombstone's count down (${width}-bit):\n${routine}")
            endif()
        endif()
    endforeach()
    if(increments EQUAL 0 OR decrements EQUAL 0)
        message(FATAL_ERROR "no weak increment (${increments}) or decrement (${decrements}) found (${width}-bit) in:\n${ir}")
    endif()
endfunction()

check_width("" 64 2147483647 4611686018427387904)
check_width("wasm32-unknown-unknown" 32 32767 1073741824)
