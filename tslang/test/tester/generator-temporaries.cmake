# What a generator body makes and what it yields is given back (00generator_temporaries.ts). No
# run shows which references leak, so the IR is read, after OwnedReturnConsumptionPass
# (--emit=mlir). Under rc:
#  - lengths:     `yield ("a" + i).length` releases the number's string and the concatenation
#                 before the yield;
#  - strings:     `yield "s" + i` retains the record before the yield and then releases both
#                 strings: the record holds its own reference to the concatenation;
#  - items:       `yield new Item(i)` retains the record and releases the instance;
#  - branchYield: a temporary in front of an `if` with a yield inside is released before the `if`;
#  - twoSegments: each of the two yields has its own two releases in front of it;
#  - main:        every call of a generator's `next` takes over the record it returns.
# Under own, `strings` keeps the concatenation (it moves into the record) and releases only the
# number's string, and the file must pass the ownership verifier.
cmake_minimum_required(VERSION 3.17.3)

function(emit model outVar)
    execute_process(COMMAND "${TSLANG}" --emit=mlir --no-default-lib -mm=${model} "${FILE}"
                    OUTPUT_VARIABLE out ERROR_VARIABLE err RESULT_VARIABLE result)
    # the IR is printed to stderr
    string(APPEND out "${err}")
    if(NOT result EQUAL 0)
        message(FATAL_ERROR "tslang -mm=${model} failed (${result}):\n${out}")
    endif()
    set(${outVar} "${out}" PARENT_SCOPE)
endfunction()

# the body of a function, from its `ts.Func` to the next one
function(func_body out prefix outVar)
    string(FIND "${out}" "ts.Func @${prefix}" begin)
    if(begin EQUAL -1)
        message(FATAL_ERROR "no function `${prefix}` in:\n${out}")
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

function(count pattern body outVar)
    string(REGEX MATCHALL "${pattern}" matches "${body}")
    list(LENGTH matches n)
    set(${outVar} ${n} PARENT_SCOPE)
endfunction()

# (CMake's regex has no {n})
set(release "\"ts[.]Release\"[(]%[0-9]+[)] : [(]!ts[.](string|class<@Item[^)]*)[)] -> [(][)][\r\n]+ *")
set(recordRetain "\"ts[.]Retain\"[(]%[0-9]+[)] : [(]!ts[.]tuple<[^\r\n]*[\r\n]+ *")
set(yield "\"ts[.]YieldReturnVal\"")
set(stringRelease "\"ts[.]Release\"[(]%[0-9]+[)] : [(]!ts[.]string[)]")

emit(rc out)

func_body("${out}" "\".f_lengths.." body)
if(NOT body MATCHES "${release}${release}${yield}")
    message(FATAL_ERROR "`lengths` does not release its two temporaries before the yield:\n${body}")
endif()

func_body("${out}" "\".f_strings.." body)
if(NOT body MATCHES "${recordRetain}${release}${release}${yield}")
    message(FATAL_ERROR "`strings` does not retain its record and release both strings before the yield:\n${body}")
endif()

func_body("${out}" "\".f_items.." body)
if(NOT body MATCHES "${recordRetain}${release}${yield}")
    message(FATAL_ERROR "`items` does not retain its record and release the instance before the yield:\n${body}")
endif()

func_body("${out}" "\".f_branchYield.." body)
if(NOT body MATCHES "${release}${release}\"ts[.]If\"")
    message(FATAL_ERROR "`branchYield` does not release its temporaries before the `if` holding the yield:\n${body}")
endif()

func_body("${out}" "\".f_twoSegments.." body)
count("${release}${release}${yield}" "${body}" n)
if(NOT n EQUAL 2)
    message(FATAL_ERROR "`twoSegments` does not release two temporaries before each yield:\n${body}")
endif()

# `next` is called with the state object as its only argument: `ts.CallIndirect(%next, %state)`.
# Only a record that holds a string or an instance has anything to take over; a `const` holding
# the record names it (`__owned_result_named`) and releases it at scope exit.
set(heapRecord "[^\r\n]*-> !ts[.]tuple<{\"value\",!ts[.](string|class)")
func_body("${out}" "main " body)
count("\"ts[.]CallIndirect\"[(]%[0-9]+, %[0-9]+[)] {__owned_result, __owned_result_(consumed|named)}${heapRecord}" "${body}" taken)
count("\"ts[.]CallIndirect\"[(]%[0-9]+, %[0-9]+[)]${heapRecord}" "${body}" calls)
if(calls EQUAL 0 OR NOT taken EQUAL calls)
    message(FATAL_ERROR "`main` does not take over every record `next` returns (${taken} of ${calls}):\n${body}")
endif()

emit(own out)
func_body("${out}" "\".f_strings.." body)
count("${stringRelease}" "${body}" n)
string(REGEX MATCH "%([0-9]+) = \"ts[.]ArithmeticBinary\"[^\r\n]*-> !ts[.]string" concat "${body}")
if(NOT n EQUAL 1 OR NOT concat OR body MATCHES "\"ts[.]Release\"[(]%${CMAKE_MATCH_1}[)]")
    message(FATAL_ERROR "under own `strings` must release the number's string and keep the yielded one:\n${body}")
endif()

execute_process(COMMAND "${TSLANG}" --emit=mlir-affine --verify-ownership --no-default-lib -mm=own "${FILE}"
                OUTPUT_QUIET ERROR_VARIABLE err RESULT_VARIABLE result)
if(NOT result EQUAL 0)
    message(FATAL_ERROR "own rejects it (${result}):\n${err}")
endif()
