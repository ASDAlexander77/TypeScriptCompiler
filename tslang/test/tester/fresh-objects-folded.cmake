# A fresh object's field read after another allocation is the value just stored
# (00fresh_objects_folded.ts): the optimized LLVM IR of `keep` loads no double, under every model
# that frees. The allocator helpers are declared as LLVM declares malloc (ProcessHeapPass); declared
# as writing other memory, the second allocation clobbered the first object's field.
cmake_minimum_required(VERSION 3.17.3)
foreach(mm none rc own)
    set(ll "${WORK_DIR}/fresh_objects_folded_${mm}.ll")
    execute_process(COMMAND "${TSLANG}" --emit=llvm --opt --opt_level=3 --no-default-lib -mm=${mm} "${FILE}" -o "${ll}"
                    OUTPUT_VARIABLE out ERROR_VARIABLE err RESULT_VARIABLE result)
    if(NOT result EQUAL 0)
        message(FATAL_ERROR "-mm=${mm}: tslang failed (${result}):\n${out}${err}")
    endif()
    file(READ "${ll}" ir)
    string(FIND "${ir}" "@keep(" begin)
    if(begin EQUAL -1)
        message(FATAL_ERROR "-mm=${mm}: no function `keep` in:\n${ir}")
    endif()
    string(SUBSTRING "${ir}" ${begin} -1 body)
    string(FIND "${body}" "\n}" end)
    string(SUBSTRING "${body}" 0 ${end} body)
    if(body MATCHES "load double")
        message(FATAL_ERROR "-mm=${mm}: `keep` loads a field it just stored:\n${body}")
    endif()
endforeach()
