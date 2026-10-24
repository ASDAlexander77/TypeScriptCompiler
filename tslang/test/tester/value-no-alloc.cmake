# A literal of a @value interface is a value, not a heap object (value/value_no_alloc.ts): the
# optimized LLVM IR of `len2`, which builds a nested @value literal and passes it on, calls no
# allocator, under every model. `alloca` is a stack slot and is fine; malloc/calloc are listed for
# Linux, where ProcessHeapPass does not rename them.
cmake_minimum_required(VERSION 3.17.3)
foreach(mm gc rc none own)
    set(ll "${WORK_DIR}/value_no_alloc_${mm}.ll")
    execute_process(COMMAND "${TSLANG}" --emit=llvm --opt --opt_level=3 --no-default-lib -mm=${mm} "${FILE}" -o "${ll}"
                    OUTPUT_VARIABLE out ERROR_VARIABLE err RESULT_VARIABLE result)
    if(NOT result EQUAL 0)
        message(FATAL_ERROR "-mm=${mm}: tslang failed (${result}):\n${out}${err}")
    endif()
    file(READ "${ll}" ir)
    string(FIND "${ir}" "@len2(" begin)
    if(begin EQUAL -1)
        message(FATAL_ERROR "-mm=${mm}: no function `len2` in:\n${ir}")
    endif()
    string(SUBSTRING "${ir}" ${begin} -1 body)
    string(FIND "${body}" "\n}" end)
    string(SUBSTRING "${body}" 0 ${end} body)
    if(body MATCHES "[mc]alloc|GC_malloc|__tslang_heap")
        message(FATAL_ERROR "-mm=${mm}: `len2` allocates:\n${body}")
    endif()
endforeach()
