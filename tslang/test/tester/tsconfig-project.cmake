# tsconfig.json as the project file (project.cpp): -p and ./tsconfig.json, the 'tslang' section,
# compilerOptions, extends, relative paths, the command line winning, and the errors.
#
# --print-default-lib-dir prints what -mm, --di and -mtriple came to, so it shows which options a
# project file set without compiling anything. The JIT runs use --no-default-lib, so no default
# library is needed.

cmake_minimum_required(VERSION 3.17.3)

foreach(var TSLANG WORK_DIR)
    if(NOT DEFINED ${var})
        message(FATAL_ERROR "${var} is required")
    endif()
endforeach()

file(REMOVE_RECURSE "${WORK_DIR}")
file(MAKE_DIRECTORY "${WORK_DIR}/app" "${WORK_DIR}/editor")

# run(<dir> <tslang arguments...>): out, err and status in the caller
macro(run dir)
    execute_process(COMMAND "${TSLANG}" ${ARGN}
        WORKING_DIRECTORY "${dir}"
        OUTPUT_VARIABLE out
        ERROR_VARIABLE err
        OUTPUT_STRIP_TRAILING_WHITESPACE
        RESULT_VARIABLE status)
endmacro()

# check(<expected stdout> <dir> <tslang arguments...>)
function(check expected dir)
    run("${dir}" ${ARGN})
    if(NOT status EQUAL 0 OR NOT out STREQUAL "${expected}")
        message(FATAL_ERROR "tslang ${ARGN} (in ${dir}): exit ${status}\nexpected: ${expected}\n     got: ${out}\n${err}")
    endif()
endfunction()

# fails(<text the error must hold> <dir> <tslang arguments...>)
function(fails expected dir)
    run("${dir}" ${ARGN})
    if(status EQUAL 0)
        message(FATAL_ERROR "tslang ${ARGN} (in ${dir}) succeeded; expected an error with: ${expected}\n${out}")
    endif()
    string(FIND "${err}" "${expected}" at)
    if(at EQUAL -1)
        message(FATAL_ERROR "tslang ${ARGN} (in ${dir}): the error does not say '${expected}':\n${err}")
    endif()
endfunction()

# --- the 'tslang' section, extends, and the command line winning -------------------------------

# JSONC: comments and trailing commas
file(WRITE "${WORK_DIR}/base.json" [=[
// shared by the projects
{
    /* the model and a debug build */
    "tslang": { "mm": "rc", "di": true, },
}
]=])

file(WRITE "${WORK_DIR}/app/tsconfig.json" [=[
{
    "extends": "../base.json",
    "compilerOptions": { "strict": true, "noEmit": true, "target": "esnext" },
    "include": ["main.ts"],
    "tslang": {
        "mtriple": "x86_64-unknown-linux-gnu",
        "no-default-lib": true,
        "jit-cache-dir": "cache"
    }
}
]=])

file(WRITE "${WORK_DIR}/app/main.ts" [=[
function main() {
    print("hello from the project");
}
]=])

set(linux "defaultlib/lib/x86_64/unknown/linux/gnu")

# -p takes the folder or the file; --project is the same
check("${linux}/debug/rc" "${WORK_DIR}" -p app --print-default-lib-dir=lib)
check("${linux}/debug/rc" "${WORK_DIR}" -p app/tsconfig.json --print-default-lib-dir=lib)
check("${linux}/debug/rc" "${WORK_DIR}" --project=app --print-default-lib-dir=lib)
# the command line wins
check("${linux}/debug/own" "${WORK_DIR}" -p app --print-default-lib-dir=lib -mm=own)
check("${linux}/release/rc" "${WORK_DIR}" -p app --print-default-lib-dir=lib --di=false)
# a later file in 'extends' wins over an earlier, and the file itself over both
file(WRITE "${WORK_DIR}/none.json" [=[{ "tslang": { "mm": "none" } }]=])
file(WRITE "${WORK_DIR}/two.json" [=[
{ "extends": ["./base.json", "./none.json"], "files": ["app/main.ts"], "tslang": { "mtriple": "x86_64-unknown-linux-gnu" } }
]=])
check("${linux}/debug/none" "${WORK_DIR}" -p two.json --print-default-lib-dir=lib)
# ./tsconfig.json is not looked for by an action that compiles nothing
check("defaultlib/lib/x86_64/unknown/linux/gnu/release/gc" "${WORK_DIR}/app" --print-default-lib-dir=lib -mtriple=x86_64-unknown-linux-gnu)

# --- running the project ---------------------------------------------------------------------

# no input file: ./tsconfig.json, its 'include' naming the entry; the JIT on this host
check("hello from the project" "${WORK_DIR}/app" --emit=jit -mtriple=)
file(GLOB objects "${WORK_DIR}/app/cache/*")
list(LENGTH objects objectsBefore)
if(objectsBefore EQUAL 0)
    message(FATAL_ERROR "the JIT cache is not in 'cache' beside the project file")
endif()

# a changed project file is not served the object of the old options
file(READ "${WORK_DIR}/app/tsconfig.json" config)
string(REPLACE "\"no-default-lib\": true" "\"no-default-lib\": true, \"mm\": \"none\"" config "${config}")
file(WRITE "${WORK_DIR}/app/tsconfig.json" "${config}")
check("hello from the project" "${WORK_DIR}/app" --emit=jit -mtriple=)
file(GLOB objects "${WORK_DIR}/app/cache/*")
list(LENGTH objects objectsAfter)
if(NOT objectsAfter GREATER objectsBefore)
    message(FATAL_ERROR "the JIT cache served the object compiled with the old project options")
endif()

# outDir: the default output name in that folder, made beside the project file
file(WRITE "${WORK_DIR}/out.json" [=[
{ "compilerOptions": { "outDir": "build/obj" }, "tslang": { "entry": "app/main.ts", "emit": "obj", "no-default-lib": true, "mm": "none" } }
]=])
check("" "${WORK_DIR}/editor" -p ../out.json)
file(GLOB built "${WORK_DIR}/build/obj/main.o*")
if(NOT built)
    message(FATAL_ERROR "outDir: no main.o(bj) in ${WORK_DIR}/build/obj")
endif()

# an input file on the command line: no tsconfig.json is read, even a broken one
file(WRITE "${WORK_DIR}/editor/tsconfig.json" "{ \"tslang\": { \"not-an-option\": 1 }, \"include\": [\"main.ts\"] }")
check("" "${WORK_DIR}/editor" --emit=mlir --no-default-lib "${WORK_DIR}/app/main.ts" -o=main.mlir)

# a tsconfig.json for the editor only (no entry file): tslang reads stdin as before
file(WRITE "${WORK_DIR}/editor/tsconfig.json" [=[{ "compilerOptions": { "strict": true }, "include": ["src/**/*.ts"] }]=])
execute_process(COMMAND "${TSLANG}" --emit=mlir --no-default-lib
    INPUT_FILE "${WORK_DIR}/app/main.ts"
    WORKING_DIRECTORY "${WORK_DIR}/editor"
    OUTPUT_VARIABLE out ERROR_VARIABLE err RESULT_VARIABLE status)
if(NOT status EQUAL 0)
    message(FATAL_ERROR "a tsconfig.json without an entry file changed reading stdin: exit ${status}\n${err}")
endif()

# --- errors ----------------------------------------------------------------------------------

file(WRITE "${WORK_DIR}/unknown.json" [=[{ "files": ["app/main.ts"], "tslang": { "mmm": "rc" } }]=])
fails("'mmm' is not a tslang option" "${WORK_DIR}" -p unknown.json)
file(WRITE "${WORK_DIR}/refused.json" [=[{ "files": ["app/main.ts"], "tslang": { "new": true } }]=])
fails("'new' is not an option a project file can set" "${WORK_DIR}" -p refused.json)
file(WRITE "${WORK_DIR}/value.json" [=[{ "files": ["app/main.ts"], "tslang": { "mm": "xx" } }]=])
fails("set in value.json" "${WORK_DIR}" -p value.json)
file(WRITE "${WORK_DIR}/array.json" [=[{ "files": ["app/main.ts"], "tslang": { "mm": ["rc", "own"] } }]=])
fails("'mm' takes one value, not an array" "${WORK_DIR}" -p array.json)
file(WRITE "${WORK_DIR}/syntax.json" [=[{ "tslang": { "mm" "rc" } }]=])
fails("Expected : after object key" "${WORK_DIR}" -p syntax.json)
file(WRITE "${WORK_DIR}/noentry.json" [=[{ "include": ["src/**/*.ts"] }]=])
fails("no file to compile" "${WORK_DIR}" -p noentry.json)
file(WRITE "${WORK_DIR}/package.json" [=[{ "extends": "@tsconfig/node20" }]=])
fails("names a package" "${WORK_DIR}" -p package.json)
file(WRITE "${WORK_DIR}/cycle1.json" [=[{ "extends": "./cycle2.json" }]=])
file(WRITE "${WORK_DIR}/cycle2.json" [=[{ "extends": "./cycle1.json" }]=])
fails("comes back to this file" "${WORK_DIR}" -p cycle1.json)
fails("no project file" "${WORK_DIR}" -p missing)

message(STATUS "tsconfig.json: -p, ./tsconfig.json, the tslang section, extends, outDir, precedence and errors")
