# `export default <expression>` (an exported constant named for the module, lowerDefaultExports) and
# `import x from "./m"` binding x to a module's default export - an expression, an object, a function by
# name, an unnamed function, a named function imported under another name, an unnamed class. Run under the
# JIT with its cache (each module an object of its own, the importer reading the others as declarations)
# and without it, under rc and none, and linked from separately compiled objects.

cmake_minimum_required(VERSION 3.17.3)

foreach(var TSLANG WORK_DIR)
    if(NOT DEFINED ${var})
        message(FATAL_ERROR "${var} is required")
    endif()
endforeach()

file(REMOVE_RECURSE "${WORK_DIR}")
file(MAKE_DIRECTORY "${WORK_DIR}")

file(WRITE "${WORK_DIR}/m1.ts" "export default 2 + 2;\n")
file(WRITE "${WORK_DIR}/m2.ts" "export default { a: 1, b: \"x\" };\n")
file(WRITE "${WORK_DIR}/m3.ts" "function helper(n: number) { return n * 3; }\nexport default helper;\n")
file(WRITE "${WORK_DIR}/m4.ts" "export default function () { return 7; }\n")
file(WRITE "${WORK_DIR}/m5.ts" "export default function named() { return 8; }\n")
file(WRITE "${WORK_DIR}/m6.ts" "export default class { v = 9; }\n")
# the default export of an imported type: no value, as TypeScript emits none (it was "failed to legalize
# operation 'ts.TypeRef'", TSLANG 193)
file(WRITE "${WORK_DIR}/m7.ts" "type Ten = number;\nexport { Ten };\n")
file(WRITE "${WORK_DIR}/m8.ts" "import { Ten } from \"./m7\";\nexport default Ten;\nexport const ten: Ten = 10;\n")
file(WRITE "${WORK_DIR}/main.ts" [=[
import four from "./m1";
import obj from "./m2";
import triple from "./m3";
import seven from "./m4";
import eight from "./m5";
import Nine from "./m6";
import { ten } from "./m8";

function main() {
    print(four, obj.a, obj.b, triple(5), seven(), eight(), new Nine().v, ten);
}
]=])

set(expected "4 1 x 15 7 8 9 10")

# run(<what> <tslang arguments...>)
function(run what)
    execute_process(COMMAND "${TSLANG}" ${ARGN}
        WORKING_DIRECTORY "${WORK_DIR}"
        OUTPUT_VARIABLE out ERROR_VARIABLE err RESULT_VARIABLE status TIMEOUT 120
        OUTPUT_STRIP_TRAILING_WHITESPACE)
    if(NOT status EQUAL 0 OR NOT out STREQUAL "${expected}")
        message(FATAL_ERROR "${what}: exit ${status}\nexpected: ${expected}\n     got: ${out}\n${err}")
    endif()
endfunction()

run("JIT with its cache" --emit=jit --no-default-lib -mm=none "${WORK_DIR}/main.ts")
run("JIT without its cache" --emit=jit --jit-cache=false --no-default-lib -mm=none "${WORK_DIR}/main.ts")
run("JIT under rc" --emit=jit --no-default-lib -mm=rc "${WORK_DIR}/main.ts")

# each module its own object, linked with the program's: the default exports are the modules' symbols
if(DEFINED TSLANG_LIB)
    set(objects "")
    foreach(module m1 m2 m3 m4 m5 m6 m7 m8)
        execute_process(COMMAND "${TSLANG}" --emit=obj --no-default-lib -mm=none "${WORK_DIR}/${module}.ts" -o "${WORK_DIR}/${module}.obj"
            WORKING_DIRECTORY "${WORK_DIR}" ERROR_VARIABLE err RESULT_VARIABLE status)
        if(NOT status EQUAL 0)
            message(FATAL_ERROR "${module}.ts --emit=obj: exit ${status}\n${err}")
        endif()
        list(APPEND objects "${WORK_DIR}/${module}.obj")
    endforeach()

    list(JOIN objects "," objectList)
    execute_process(COMMAND "${TSLANG}" --emit=exe --no-default-lib -mm=none "--tslang-lib-path=${TSLANG_LIB}"
                            "--obj=${objectList}" "${WORK_DIR}/main.ts" -o "${WORK_DIR}/main${CMAKE_EXECUTABLE_SUFFIX}"
        WORKING_DIRECTORY "${WORK_DIR}" OUTPUT_VARIABLE out ERROR_VARIABLE err RESULT_VARIABLE status)
    if(NOT status EQUAL 0)
        message(FATAL_ERROR "main.ts --emit=exe with the modules' objects: exit ${status}\n${out}${err}")
    endif()

    execute_process(COMMAND "${WORK_DIR}/main${CMAKE_EXECUTABLE_SUFFIX}"
        WORKING_DIRECTORY "${WORK_DIR}" OUTPUT_VARIABLE out ERROR_VARIABLE err RESULT_VARIABLE status
        OUTPUT_STRIP_TRAILING_WHITESPACE)
    if(NOT status EQUAL 0 OR NOT out STREQUAL "${expected}")
        message(FATAL_ERROR "the linked program: exit ${status}\nexpected: ${expected}\n     got: ${out}\n${err}")
    endif()
endif()

message(STATUS "export default <expression> and default imports")
