# How `import` finds a module (MLIRGenImpl::includeFilePath / resolveModuleFile), as TypeScript does:
# `<name>.ts` (or `<name>.<os>.ts`) first, then `<name>.tsx`, `<name>.d.ts`, a folder (its package.json's
# `types`/`typings`/`main`, or `index`), and for a name that is not a path `node_modules/<name>` and
# `node_modules/@types/<name>` in the importing file's folder and those above; `./m.js` is m.ts.

cmake_minimum_required(VERSION 3.17.3)

foreach(var TSLANG WORK_DIR)
    if(NOT DEFINED ${var})
        message(FATAL_ERROR "${var} is required")
    endif()
endforeach()

file(REMOVE_RECURSE "${WORK_DIR}")
file(MAKE_DIRECTORY "${WORK_DIR}")

# compiles(<file> [<what it checks>]): the file compiles to an object
function(compiles file what)
    execute_process(COMMAND "${TSLANG}" --emit=obj --no-default-lib -mm=none "${WORK_DIR}/${file}" -o "${WORK_DIR}/out.obj"
        WORKING_DIRECTORY "${WORK_DIR}"
        OUTPUT_VARIABLE out ERROR_VARIABLE err RESULT_VARIABLE status TIMEOUT 60)
    if(NOT status EQUAL 0)
        message(FATAL_ERROR "${what}: ${file} does not compile (exit ${status})\n${out}${err}")
    endif()
endfunction()

# a declaration file only
file(WRITE "${WORK_DIR}/decl.d.ts" "export declare function fromDecl(): number;\n")
file(WRITE "${WORK_DIR}/use_decl.ts" "import { fromDecl } from \"./decl\";\nfunction main() { let f = fromDecl; }\n")
compiles(use_decl.ts "./decl is decl.d.ts")

# a folder with an index
file(WRITE "${WORK_DIR}/lib/index.ts" "export function fromIndex() { return 1; }\n")
file(WRITE "${WORK_DIR}/use_index.ts" "import { fromIndex } from \"./lib\";\nfunction main() { print(fromIndex()); }\n")
compiles(use_index.ts "./lib is lib/index.ts")

# a package in node_modules, by its package.json's `types`, from a folder below
file(WRITE "${WORK_DIR}/node_modules/pkg/package.json" "{ \"name\": \"pkg\", \"types\": \"dist/pkg.d.ts\", \"main\": \"dist/pkg.js\" }\n")
file(WRITE "${WORK_DIR}/node_modules/pkg/dist/pkg.d.ts" "export declare const fromPkg: number;\n")
file(WRITE "${WORK_DIR}/src/deep/use_pkg.ts" "import { fromPkg } from \"pkg\";\nfunction main() { let n = 0; }\n")
compiles(src/deep/use_pkg.ts "pkg from node_modules two folders up")

# `main` naming a .js: the declarations beside it
file(WRITE "${WORK_DIR}/node_modules/mainjs/package.json" "{ \"name\": \"mainjs\", \"main\": \"lib/entry.js\" }\n")
file(WRITE "${WORK_DIR}/node_modules/mainjs/lib/entry.d.ts" "export declare const fromMain: number;\n")
file(WRITE "${WORK_DIR}/use_mainjs.ts" "import { fromMain } from \"mainjs\";\nfunction main() { let n = 0; }\n")
compiles(use_mainjs.ts "main: lib/entry.js is lib/entry.d.ts")

# @types, a scoped name too
file(WRITE "${WORK_DIR}/node_modules/@types/typed/index.d.ts" "export declare const fromTypes: number;\n")
file(WRITE "${WORK_DIR}/node_modules/@types/scope__name/index.d.ts" "export declare const fromScoped: number;\n")
file(WRITE "${WORK_DIR}/use_types.ts" "import { fromTypes } from \"typed\";\nimport { fromScoped } from \"@scope/name\";\nfunction main() { let n = 0; }\n")
compiles(use_types.ts "node_modules/@types/typed and @types/scope__name")

# a package whose `main` names itself (`"main": ""`): its index, not a loop
file(WRITE "${WORK_DIR}/node_modules/self/package.json" "{ \"name\": \"self\", \"main\": \"\" }\n")
file(WRITE "${WORK_DIR}/node_modules/self/index.d.ts" "export declare const fromSelf: number;\n")
file(WRITE "${WORK_DIR}/use_self.ts" "import { fromSelf } from \"self\";\nfunction main() { let n = 0; }\n")
compiles(use_self.ts "main: \"\" falls back to the index")

# `./util.js` is util.ts: run under the JIT, whose cache compiles util.ts on its own
file(WRITE "${WORK_DIR}/util.ts" "export function twice(n: number) { return n * 2; }\n")
file(WRITE "${WORK_DIR}/use_js.ts" "import { twice } from \"./util.js\";\nfunction main() { print(twice(21)); }\n")
execute_process(COMMAND "${TSLANG}" --emit=jit --no-default-lib -mm=none "${WORK_DIR}/use_js.ts"
    WORKING_DIRECTORY "${WORK_DIR}"
    OUTPUT_VARIABLE out ERROR_VARIABLE err RESULT_VARIABLE status TIMEOUT 60
    OUTPUT_STRIP_TRAILING_WHITESPACE)
if(NOT status EQUAL 0 OR NOT out STREQUAL "42")
    message(FATAL_ERROR "./util.js is util.ts: exit ${status}, printed '${out}'\n${err}")
endif()

# a name found nowhere still says which file it looked for
file(WRITE "${WORK_DIR}/use_missing.ts" "import { x } from \"nowhere\";\nfunction main() { }\n")
execute_process(COMMAND "${TSLANG}" --emit=obj --no-default-lib -mm=none "${WORK_DIR}/use_missing.ts" -o "${WORK_DIR}/out.obj"
    WORKING_DIRECTORY "${WORK_DIR}" OUTPUT_VARIABLE out ERROR_VARIABLE err RESULT_VARIABLE status)
string(FIND "${err}" "can't open file:" at)
if(status EQUAL 0 OR at EQUAL -1)
    message(FATAL_ERROR "an import found nowhere: exit ${status}\n${err}")
endif()

message(STATUS "import finds .d.ts, folders, package.json, node_modules, @types and .js names")
