# A temporary inside a generator body is released before the yield that ends the call, and only
# where no resume point comes between (00generator_temporaries.ts). No run shows which ones leak,
# so the IR is read, after OwnedReturnConsumptionPass (--emit=mlir), under rc:
#  - lengths:     `yield ("a" + i).length` releases the number's string and the concatenation;
#  - strings:     `yield "s" + i` releases the number's string but not the concatenation, which
#                 the yielded record holds;
#  - branchYield: a temporary in front of an `if` with a yield inside is not released;
#  - twoSegments: each of the two yields has its own two releases in front of it.
# Under own the file must also pass the ownership verifier.
cmake_minimum_required(VERSION 3.17.3)
execute_process(COMMAND "${TSLANG}" --emit=mlir --no-default-lib -mm=rc "${FILE}"
                OUTPUT_VARIABLE out ERROR_VARIABLE err RESULT_VARIABLE result)
# the IR is printed to stderr
string(APPEND out "${err}")
if(NOT result EQUAL 0)
    message(FATAL_ERROR "tslang failed (${result}):\n${out}")
endif()

# the body of a generator's `next`, from its `ts.Func` to the next one
function(next_body name outVar)
    string(FIND "${out}" "ts.Func @\".f_${name}.." begin)
    if(begin EQUAL -1)
        message(FATAL_ERROR "no generator `${name}` in:\n${out}")
    endif()
    string(SUBSTRING "${out}" ${begin} -1 body)
    string(SUBSTRING "${body}" 1 -1 rest)
    string(FIND "${rest}" "ts.Func @" end)
    if(NOT end EQUAL -1)
        math(EXPR end "${end} + 1")
        string(SUBSTRING "${body}" 0 ${end} body)
    endif()
    set(${outVar} "${body}" PARENT_SCOPE)
endfunction()

function(count_releases body outVar)
    string(REGEX MATCHALL "\"ts[.]Release\"[(]%[0-9]+[)] : [(]!ts[.]string[)]" releases "${body}")
    list(LENGTH releases count)
    set(${outVar} ${count} PARENT_SCOPE)
endfunction()

# (CMake's regex has no {n})
set(release "\"ts[.]Release\"[(]%[0-9]+[)] : [(]!ts[.]string[)] -> [(][)][\r\n]+ *")
set(releasesBeforeYield "${release}${release}\"ts[.]YieldReturnVal\"")

next_body(lengths body)
if(NOT body MATCHES "${releasesBeforeYield}")
    message(FATAL_ERROR "`lengths` does not release its two temporaries before the yield:\n${body}")
endif()

next_body(strings body)
count_releases("${body}" count)
string(REGEX MATCH "%([0-9]+) = \"ts[.]ArithmeticBinary\"[^\r\n]*-> !ts[.]string" concat "${body}")
if(NOT count EQUAL 1 OR NOT concat OR body MATCHES "\"ts[.]Release\"[(]%${CMAKE_MATCH_1}[)]")
    message(FATAL_ERROR "`strings` must release the number's string and keep the yielded one:\n${body}")
endif()

next_body(branchYield body)
count_releases("${body}" count)
if(NOT count EQUAL 0)
    message(FATAL_ERROR "`branchYield` releases a temporary a resume point cuts off:\n${body}")
endif()

next_body(twoSegments body)
string(REGEX MATCHALL "${releasesBeforeYield}" segments "${body}")
list(LENGTH segments count)
if(NOT count EQUAL 2)
    message(FATAL_ERROR "`twoSegments` does not release two temporaries before each yield:\n${body}")
endif()

execute_process(COMMAND "${TSLANG}" --emit=mlir-affine --verify-ownership --no-default-lib -mm=own "${FILE}"
                OUTPUT_QUIET ERROR_VARIABLE err RESULT_VARIABLE result)
if(NOT result EQUAL 0)
    message(FATAL_ERROR "own rejects it (${result}):\n${err}")
endif()
