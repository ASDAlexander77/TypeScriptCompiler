# -mm=own --own-fallback=count (own rc fallback design): a value inference cannot prove keeps rc's
# count, with a warning where own would have reported its error.
#   MODE=IR            compiled to --emit=mlir-affine with the flag: compiles, warns, and keeps at least
#                      one retain marked `__own_counted` - exactly COUNTED of them when it is given (a
#                      retain too few frees the object early); with FUNC, RELEASED and RELEASES, also
#                      exactly RELEASES releases naming RELEASED in function FUNC
#   MODE=RUN           run with the flag (JIT): prints `done.` and no error
#   MODE=EXPECT_ERROR  still refused with the flag: `error:`
# Every mode also checks that the file is refused under own without the flag: each test exercises
# the fallback, never a shape own already accepts.
cmake_minimum_required(VERSION 3.17.3)

if(NOT DEFINED MODE)
    message(FATAL_ERROR "MODE is not set (IR, RUN or EXPECT_ERROR)")
endif()

execute_process(COMMAND "${TSLANG}" --emit=mlir-affine --no-default-lib -mm=own "${FILE}"
                OUTPUT_VARIABLE plainOut ERROR_VARIABLE plainErr RESULT_VARIABLE plainResult)
if(plainResult EQUAL 0)
    message(FATAL_ERROR "${FILE} compiles under -mm=own without --own-fallback=count: it does not exercise the fallback")
endif()

if(MODE STREQUAL "IR")
    execute_process(COMMAND "${TSLANG}" --emit=mlir-affine --no-default-lib -mm=own --own-fallback=count "${FILE}"
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
    execute_process(COMMAND "${TSLANG}" --emit=jit -mm=own --own-fallback=count "${FILE}"
                    OUTPUT_VARIABLE out ERROR_VARIABLE err RESULT_VARIABLE result)
    string(APPEND out "${err}")
    if(NOT result EQUAL 0)
        message(FATAL_ERROR "tslang failed (${result}):\n${out}")
    endif()
    string(FIND "${out}" "done." done)
    if(done EQUAL -1)
        message(FATAL_ERROR "the program did not finish (no `done.`):\n${out}")
    endif()
    string(FIND "${out}" "error:" error)
    if(NOT error EQUAL -1)
        message(FATAL_ERROR "an error was reported:\n${out}")
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
    message(FATAL_ERROR "unknown MODE '${MODE}' (IR, RUN or EXPECT_ERROR)")
endif()
