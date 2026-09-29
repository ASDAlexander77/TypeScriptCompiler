# The JIT cache: `tslang --emit=jit` compiles the program and each .ts module it imports into an
# object of its own, kept in `__jit` beside the source, and loads the objects from there while
# nothing they were compiled from changes.
#
# 1. The first run compiles both files and leaves an object and a manifest for each in __jit.
# 2. The second run finds them: the program runs, and neither object is written again.
# 3. The imported module changes: the program runs the new code, and both objects are compiled
#    again - the program's too, which was compiled against the module's declarations.
#
# The sources are copied into the working directory, so the cache and the edit stay out of the
# source tree. -mm=none and --no-default-lib keep the run to the compiler alone.

cmake_minimum_required(VERSION 3.17.3)

foreach(var TSLANG SOURCE_DIR WORK_DIR)
    if(NOT DEFINED ${var})
        message(FATAL_ERROR "${var} is required")
    endif()
endforeach()

file(REMOVE_RECURSE "${WORK_DIR}")
file(MAKE_DIRECTORY "${WORK_DIR}")
file(COPY "${SOURCE_DIR}/greeting.ts" "${SOURCE_DIR}/greeting_module.ts" DESTINATION "${WORK_DIR}")

# run(<expected output>) - fails the test unless the program runs and prints what is expected
function(run expected)
    execute_process(COMMAND "${TSLANG}" --emit=jit -mm=none --no-default-lib greeting.ts
        WORKING_DIRECTORY "${WORK_DIR}"
        OUTPUT_VARIABLE out
        ERROR_VARIABLE err
        RESULT_VARIABLE status)
    string(STRIP "${out}" out)
    if(NOT status EQUAL 0 OR NOT out STREQUAL expected)
        message(FATAL_ERROR "expected '${expected}': exit ${status}\n${out}\n${err}")
    endif()
endfunction()

# the objects in the cache, and when each was written
function(cached_objects result)
    file(GLOB objects "${WORK_DIR}/__jit/*.o" "${WORK_DIR}/__jit/*.obj")
    list(SORT objects)
    set(stamps "")
    foreach(object ${objects})
        file(TIMESTAMP "${object}" stamp "%Y%m%d%H%M%S")
        list(APPEND stamps "${object}@${stamp}")
    endforeach()
    set(${result} "${stamps}" PARENT_SCOPE)
endfunction()

run("one")
cached_objects(first)
list(LENGTH first count)
if(NOT count EQUAL 2)
    message(FATAL_ERROR "expected an object for greeting.ts and one for greeting_module.ts in __jit, found: ${first}")
endif()

# a second apart, so an object written again has another time
execute_process(COMMAND "${CMAKE_COMMAND}" -E sleep 1.1)

run("one")
cached_objects(second)
if(NOT first STREQUAL second)
    message(FATAL_ERROR "the second run compiled again instead of using the cache:\n${first}\n${second}")
endif()

file(WRITE "${WORK_DIR}/greeting_module.ts" "export function greeting() {\n    return \"two\";\n}\n")

run("two")
cached_objects(third)
foreach(object ${third})
    if(object IN_LIST second)
        message(FATAL_ERROR "${object} was not compiled again after greeting_module.ts changed")
    endif()
endforeach()

message(STATUS "the JIT cache is used while the sources are the same, and compiled again when they change")
