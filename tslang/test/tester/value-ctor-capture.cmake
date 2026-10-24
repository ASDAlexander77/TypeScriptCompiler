# A closure in a @value constructor that reads `this` (value/value_ctor_capture.ts) captures the
# address of the slot `new` made for the value. Under rc its box borrows that slot: the box's release
# routine (`tsrelcb_`) frees the box and nothing else. Released as a capture cell, the slot had a
# count read and decremented in the caller's frame and was handed to the allocator's free.
cmake_minimum_required(VERSION 3.17.3)
set(ll "${WORK_DIR}/value_ctor_capture.ll")
execute_process(COMMAND "${TSLANG}" --emit=llvm --no-default-lib -mm=rc "${FILE}" -o "${ll}"
                OUTPUT_VARIABLE out ERROR_VARIABLE err RESULT_VARIABLE result)
if(NOT result EQUAL 0)
    message(FATAL_ERROR "tslang failed (${result}):
${out}${err}")
endif()
file(READ "${ll}" ir)
# ...and no function of the program's own (not a `ts*` ownership routine or a `__tslang_*` helper)
# counts a block directly: the only one such a count could be on here is the slot, which a nested
# closure's capture took as a cell (RetainCell, a count written into the caller's frame)
set(rest "${ir}")
while(TRUE)
    string(FIND "${rest}" "
define " begin)
    if(begin EQUAL -1)
        break()
    endif()
    math(EXPR begin "${begin} + 1")
    string(SUBSTRING "${rest}" ${begin} -1 rest)
    string(FIND "${rest}" "
}" end)
    string(SUBSTRING "${rest}" 0 ${end} body)
    string(SUBSTRING "${rest}" ${end} -1 rest)
    if(body MATCHES "^define [^@]*@(ts|__tslang)")
        continue()
    endif()
    if(body MATCHES "@__tslang_(inc|dec)_ref[(]")
        message(FATAL_ERROR "a function counts a block directly:
${body}")
    endif()
endwhile()
set(routines 0)
while(TRUE)
    string(FIND "${ir}" "define internal void @tsrelcb_" begin)
    if(begin EQUAL -1)
        break()
    endif()
    string(SUBSTRING "${ir}" ${begin} -1 ir)
    string(FIND "${ir}" "
}" end)
    string(SUBSTRING "${ir}" 0 ${end} body)
    string(SUBSTRING "${ir}" ${end} -1 ir)
    math(EXPR routines "${routines} + 1")
    string(REGEX MATCHALL "__tslang_free_block[(]" frees "${body}")
    list(LENGTH frees free_count)
    if(NOT free_count EQUAL 1)
        message(FATAL_ERROR "a capture box's release routine frees ${free_count} blocks, not just the box:
${body}")
    endif()
endwhile()
if(routines EQUAL 0)
    message(FATAL_ERROR "no capture box release routine in:
${ll}")
endif()
