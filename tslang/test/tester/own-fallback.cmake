# -mm=own --own-fallback=count (own rc fallback design): a value inference cannot prove keeps rc's
# count, with a warning where own would have reported its error.
#   MODE=IR            compiled to --emit=mlir-affine with the flag: compiles, warns, and keeps at least
#                      one retain marked `__own_counted` - exactly COUNTED of them when it is given (a
#                      retain too few frees the object early); with FUNC, RELEASED and RELEASES, also
#                      exactly RELEASES releases naming RELEASED in function FUNC; with HEAP_AT (the
#                      line:column of a `new`), also that StackPromotionPass keeps that object on the
#                      heap because it is counted (--stack-promotion-report)
#   MODE=RUN           run with the flag (JIT, or built ahead of time with EXE=1): exits 0, warns,
#                      prints `done.` and no error, and prints what it prints under -mm=gc
#   MODE=EXPECT_ERROR  still refused with the flag: `error:`
#   MODE=SAME          the other way round: a file own accepts without the flag is accepted with it
#                      too, and run with it (JIT) prints what it prints under -mm=gc, with no error -
#                      the flag only ever turns an error into a count
# Every other mode also checks that the file is refused under own without the flag: each test
# exercises the fallback, never a shape own already accepts.
cmake_minimum_required(VERSION 3.17.3)

if(NOT DEFINED MODE)
    message(FATAL_ERROR "MODE is not set (IR, RUN, EXPECT_ERROR or SAME)")
endif()

execute_process(COMMAND "${TSLANG}" --emit=mlir-affine --no-default-lib -mm=own "${FILE}"
                OUTPUT_VARIABLE plainOut ERROR_VARIABLE plainErr RESULT_VARIABLE plainResult)
if(MODE STREQUAL "SAME")
    if(NOT plainResult EQUAL 0)
        message(FATAL_ERROR "${FILE} is refused under -mm=own without --own-fallback=count:
${plainOut}${plainErr}")
    endif()
elseif(plainResult EQUAL 0)
    message(FATAL_ERROR "${FILE} compiles under -mm=own without --own-fallback=count: it does not exercise the fallback")
endif()

if(MODE STREQUAL "IR")
    set(report)
    if(DEFINED HEAP_AT)
        set(report --stack-promotion-report)
    endif()
    execute_process(COMMAND "${TSLANG}" --emit=mlir-affine --no-default-lib -mm=own --own-fallback=count ${report} "${FILE}"
                    OUTPUT_VARIABLE out ERROR_VARIABLE err RESULT_VARIABLE result)
    # the IR is printed to stderr, with the diagnostics
    string(APPEND out "${err}")
    if(NOT result EQUAL 0)
        message(FATAL_ERROR "tslang failed (${result}):\n${out}")
    endif()
    string(FIND "${out}" "warning: -mm=own counts this object here (--own-fallback=count); a cycle through it is never freed" warning)
    if(warning EQUAL -1)
        message(FATAL_ERROR "no fallback warning:\n${out}")
    endif()
    string(REGEX MATCHALL "__own_counted" marked "${out}")
    list(LENGTH marked markedCount)
    if(markedCount EQUAL 0)
        message(FATAL_ERROR "no retain marked __own_counted:\n${out}")
    endif()
    if(DEFINED COUNTED AND NOT markedCount EQUAL COUNTED)
        message(FATAL_ERROR "${markedCount} retains marked __own_counted, expected ${COUNTED}:\n${out}")
    endif()
    if(DEFINED HEAP_AT)
        string(FIND "${out}" ":${HEAP_AT}: remark: stack-promotion: heap: counted" heap)
        if(heap EQUAL -1)
            message(FATAL_ERROR "the object made at ${HEAP_AT} is not kept on the heap as counted:\n${out}")
        endif()
    endif()
    # the releases rc placed stay, and none is added but a kept parameter's at the exits: in function
    # FUNC, exactly RELEASES `ts.Release` / `ts.ReleaseSlot` whose line names RELEASED (`@Node`)
    if(DEFINED RELEASES)
        string(FIND "${out}" "ts.Func @${FUNC} " begin)
        if(begin EQUAL -1)
            message(FATAL_ERROR "no function `${FUNC}` in:\n${out}")
        endif()
        string(SUBSTRING "${out}" ${begin} -1 body)
        string(SUBSTRING "${body}" 1 -1 rest)
        string(FIND "${rest}" "ts.Func @" end)
        if(NOT end EQUAL -1)
            math(EXPR end "${end} + 1")
            string(SUBSTRING "${body}" 0 ${end} body)
        endif()
        string(REGEX MATCHALL "\"ts[.]Release(Slot)?\"[^\n]*${RELEASED}" releases "${body}")
        list(LENGTH releases releaseCount)
        if(NOT releaseCount EQUAL RELEASES)
            message(FATAL_ERROR "`${FUNC}` has ${releaseCount} releases of ${RELEASED}, expected ${RELEASES}:\n${body}")
        endif()
    endif()
elseif(MODE STREQUAL "RUN")
    # what the program prints under gc is what it must print counted
    execute_process(COMMAND "${TSLANG}" --emit=jit --jit-cache=false --no-default-lib -mm=gc "${FILE}"
                    OUTPUT_VARIABLE gcOut ERROR_VARIABLE gcErr RESULT_VARIABLE gcResult)
    if(NOT gcResult EQUAL 0)
        message(FATAL_ERROR "tslang -mm=gc failed (${gcResult}):\n${gcOut}${gcErr}")
    endif()
    # EXE: built ahead of time, against this build's runtime (TSLANG_LIB), and run, rather than run
    # by the JIT
    if(EXE)
        get_filename_component(exeName "${FILE}" NAME_WE)
        set(exeFile "${CMAKE_CURRENT_BINARY_DIR}/${exeName}.own-fallback.exe")
        file(REMOVE "${exeFile}")
        execute_process(COMMAND "${TSLANG}" --emit=exe --no-default-lib -mm=own --own-fallback=count
                                "--tslang-lib-path=${TSLANG_LIB}" "${FILE}" -o "${exeFile}"
                        OUTPUT_VARIABLE buildOut ERROR_VARIABLE buildErr RESULT_VARIABLE result)
        set(warnings "${buildOut}${buildErr}")
        if(NOT result EQUAL 0)
            message(FATAL_ERROR "tslang failed (${result}):\n${warnings}")
        endif()
        execute_process(COMMAND "${exeFile}" OUTPUT_VARIABLE out ERROR_VARIABLE err RESULT_VARIABLE result)
    else()
        # compiled every run (no JIT cache): a program loaded from the cache is not compiled, and
        # gives no warning
        execute_process(COMMAND "${TSLANG}" --emit=jit --jit-cache=false --no-default-lib -mm=own --own-fallback=count "${FILE}"
                        OUTPUT_VARIABLE out ERROR_VARIABLE err RESULT_VARIABLE result)
        set(warnings "${err}")
    endif()
    if(NOT result EQUAL 0)
        message(FATAL_ERROR "the counted program failed (${result}):\n${out}${err}")
    endif()
    string(FIND "${warnings}" "warning: -mm=own counts this object here (--own-fallback=count)" warning)
    if(warning EQUAL -1)
        message(FATAL_ERROR "no fallback warning:\n${warnings}")
    endif()
    string(FIND "${out}" "done." done)
    if(done EQUAL -1)
        message(FATAL_ERROR "the program did not finish (no `done.`):\n${out}${err}")
    endif()
    string(FIND "${out}${err}" "error:" error)
    if(NOT error EQUAL -1)
        message(FATAL_ERROR "an error was reported:\n${out}${err}")
    endif()
    if(NOT out STREQUAL gcOut)
        message(FATAL_ERROR "counted, the program prints\n${out}\nwhere under gc it prints\n${gcOut}")
    endif()
elseif(MODE STREQUAL "SAME")
    execute_process(COMMAND "${TSLANG}" --emit=jit --jit-cache=false --no-default-lib -mm=gc "${FILE}"
                    OUTPUT_VARIABLE gcOut ERROR_VARIABLE gcErr RESULT_VARIABLE gcResult)
    execute_process(COMMAND "${TSLANG}" --emit=jit --jit-cache=false --no-default-lib -mm=own --own-fallback=count "${FILE}"
                    OUTPUT_VARIABLE out ERROR_VARIABLE err RESULT_VARIABLE result)
    if(NOT result EQUAL 0 OR NOT gcResult EQUAL 0)
        message(FATAL_ERROR "with --own-fallback=count (${result}) or under gc (${gcResult}) it fails:
${out}${err}${gcOut}${gcErr}")
    endif()
    string(FIND "${out}${err}" "error:" error)
    if(NOT error EQUAL -1 OR NOT out STREQUAL gcOut)
        message(FATAL_ERROR "with --own-fallback=count it prints
${out}${err}
where under gc it prints
${gcOut}")
    endif()
elseif(MODE STREQUAL "EXPECT_ERROR")
    execute_process(COMMAND "${TSLANG}" --emit=mlir-affine --no-default-lib -mm=own --own-fallback=count "${FILE}"
                    OUTPUT_VARIABLE out ERROR_VARIABLE err RESULT_VARIABLE result)
    string(APPEND out "${err}")
    string(FIND "${out}" "error:" error)
    if(result EQUAL 0 OR error EQUAL -1)
        message(FATAL_ERROR "compiled with --own-fallback=count, expected an error (${result}):\n${out}")
    endif()
else()
    message(FATAL_ERROR "unknown MODE '${MODE}' (IR, RUN, EXPECT_ERROR or SAME)")
endif()
