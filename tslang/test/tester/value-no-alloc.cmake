# A @value value is not a heap object (value/value_no_alloc.ts): the optimized LLVM IR of `len2`,
# which builds a nested literal of a @value interface and passes it on, and of `plus2`, which makes
# @value class values with `new` and passes them to a static helper, calls no allocator, under every
# model. `alloca` is a stack slot and is fine; malloc/calloc are listed for Linux, where
# ProcessHeapPass does not rename them. LLVM can fold away the allocation of an object that does not
# escape, so the unoptimized IR of `V.plus`, whose `new V` is returned, is read as well.
cmake_minimum_required(VERSION 3.17.3)
function(check_no_alloc mm ir fn)
    string(FIND "${ir}" "@${fn}(" begin)
    if(begin EQUAL -1)
        message(FATAL_ERROR "-mm=${mm}: no function `${fn}` in:
${ir}")
    endif()
    string(SUBSTRING "${ir}" ${begin} -1 body)
    string(FIND "${body}" "
}" end)
    string(SUBSTRING "${body}" 0 ${end} body)
    if(body MATCHES "[mc]alloc[(]|GC_malloc|__tslang_heap")
        message(FATAL_ERROR "-mm=${mm}: `${fn}` allocates:
${body}")
    endif()
endfunction()
foreach(mm gc rc none own)
    foreach(opt opt noopt)
        set(ll "${WORK_DIR}/value_no_alloc_${mm}_${opt}.ll")
        if(opt STREQUAL "opt")
            set(opt_args --opt --opt_level=3)
        else()
            set(opt_args)
        endif()
        execute_process(COMMAND "${TSLANG}" --emit=llvm ${opt_args} --no-default-lib -mm=${mm} "${FILE}" -o "${ll}"
                        OUTPUT_VARIABLE out ERROR_VARIABLE err RESULT_VARIABLE result)
        if(NOT result EQUAL 0)
            message(FATAL_ERROR "-mm=${mm} ${opt}: tslang failed (${result}):
${out}${err}")
        endif()
        file(READ "${ll}" ir)
        if(opt STREQUAL "opt")
            check_no_alloc(${mm} "${ir}" len2)
            check_no_alloc(${mm} "${ir}" plus2)
        else()
            check_no_alloc(${mm} "${ir}" V.plus)
        endif()
    endforeach()
endforeach()
