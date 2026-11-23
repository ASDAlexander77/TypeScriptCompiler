# -mm=own: the number of releases of one shape in what a file compiles to (--emit=mlir-affine), at least MIN:
#   -DTSLANG=... -DFILE=... -DPATTERN=<regex of the release line> -DMIN=<n> [-DFLAGS="--own-fallback=count"]
# Used where a test needs a release on more than one path (the normal one and an exception's), which one
# PASS_REGULAR_EXPRESSION can't count.
cmake_minimum_required(VERSION 3.17.3)

set(flags)
if(DEFINED FLAGS)
    separate_arguments(flags NATIVE_COMMAND "${FLAGS}")
endif()

execute_process(COMMAND "${TSLANG}" --emit=mlir-affine --no-default-lib -mm=own ${flags} "${FILE}"
                OUTPUT_VARIABLE out ERROR_VARIABLE err RESULT_VARIABLE result)
# the IR is printed to stderr, with the diagnostics
string(APPEND out "${err}")
if(NOT result EQUAL 0)
    message(FATAL_ERROR "tslang failed (${result}):\n${out}")
endif()

string(REGEX MATCHALL "${PATTERN}" found "${out}")
list(LENGTH found count)
if(count LESS MIN)
    message(FATAL_ERROR "${count} releases matching `${PATTERN}`, expected at least ${MIN}")
endif()
message(STATUS "${count} releases matching `${PATTERN}`")
