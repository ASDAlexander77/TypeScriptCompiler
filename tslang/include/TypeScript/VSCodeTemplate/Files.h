#define NODE_MODULE_TSLANG_PATH "node_modules/tslang"
#define DOT_VSCODE_PATH ".vscode"

const auto TSCONFIG_JSON_DATA = R"raw(
{
    "compilerOptions": {
      "target": "es2017",
      "lib": ["dom", "dom.iterable", "esnext"],
      "allowJs": true,
      "skipLibCheck": true,
      "strict": true,
      "noEmit": true,
      "esModuleInterop": true,
      "module": "esnext",
      "moduleResolution": "bundler",
      "resolveJsonModule": true,
      "isolatedModules": true,
      "jsx": "preserve",
      "incremental": true,
      "types": ["tslang"]
    },
    "include": ["<<PROJECT>>.ts"],
    "exclude": ["node_modules"]
}
)raw";

const auto TSLANG_INDEX_D_TS = R"raw(
declare function print(...args: any[]) : void;
declare function assert(cond: boolean, msg?: string) : void;
declare type byte = any;
declare type short = any;
declare type ushort = any;
declare type int = any;
declare type uint = any;
declare type index = any;
declare type long = any;
declare type ulong = any;
declare type char = any;
declare type i8 = any;
declare type i16 = any;
declare type i32 = any;
declare type i64 = any;
declare type u8 = any;
declare type u16 = any;
declare type u32 = any;
declare type u64 = any;
declare type s8 = any;
declare type s16 = any;
declare type s32 = any;
declare type s64 = any;
declare type f16 = any;
declare type f32 = any;
declare type f64 = any;
declare type f128 = any;
declare type half = any;
declare type float = any;
declare type double = any;
declare type Opaque = any;

type Ref<T> = any
type Reference<T> = Ref<T> // deprecated alias of Ref

declare function Ref<T>(r: T): Ref<T>;
declare function Deref<T>(r: Ref<T>): T;

// deprecated aliases of Ref / Deref
declare function ReferenceOf<T>(r: T): Ref<T>;
declare function LoadReference<T>(r: Ref<T>): T;

declare function sizeof<T>(v?: T): index;
)raw";

const auto TASKS_JSON_DATA = R"raw(
{
    // See https://go.microsoft.com/fwlink/?LinkId=733558
    // for the documentation about the tasks.json format
    "version": "2.0.0",
    "tasks": [
        {
            "label": "build EXE (Debug)",
            "command": "<<TSLANG_CMD>>",
            "type": "shell",
            "args": [
                "--gc-lib-path=<<GC_LIB_PATH>>",
                "--llvm-lib-path=<<LLVM_LIB_PATH>>",
                "--tslang-lib-path=<<TSLANG_LIB_PATH>>",
                "--default-lib-path=<<DEFAULT_LIB_PATH>>",
                "--no-default-lib",
                "--di",
                "--opt_level=0",
                "--emit=exe",
                "${file}"
            ],
            "group": {
                "kind": "build",
                "isDefault": true
            },
            "presentation": {
                "reveal": "silent"
            },
            "problemMatcher": "$msCompile"
        },
        {
            "label": "build EXE (Release)",
            "command": "<<TSLANG_CMD>>",
            "type": "shell",
            "args": [
                "--gc-lib-path=<<GC_LIB_PATH>>",
                "--llvm-lib-path=<<LLVM_LIB_PATH>>",
                "--tslang-lib-path=<<TSLANG_LIB_PATH>>",
                "--default-lib-path=<<DEFAULT_LIB_PATH>>",
                "--no-default-lib",
                "--opt",
                "--opt_level=3",
                "--emit=exe",
                "${file}"
            ],
            "group": {
                "kind": "build",
                "isDefault": false
            },
            "presentation": {
                "reveal": "silent"
            },
            "problemMatcher": "$msCompile"
        },
        {
            "label": "build DLL (Debug)",
            "command": "<<TSLANG_CMD>>",
            "type": "shell",
            "args": [
                "--gc-lib-path=<<GC_LIB_PATH>>",
                "--llvm-lib-path=<<LLVM_LIB_PATH>>",
                "--tslang-lib-path=<<TSLANG_LIB_PATH>>",
                "--default-lib-path=<<DEFAULT_LIB_PATH>>",
                "--no-default-lib",
                "--di",
                "--opt_level=0",
                "--emit=exe",
                "${file}"
            ],
            "group": {
                "kind": "build",
                "isDefault": false
            },
            "presentation": {
                "reveal": "silent"
            },
            "problemMatcher": "$msCompile"
        },
        {
            "label": "build DLL (Release)",
            "command": "<<TSLANG_CMD>>",
            "type": "shell",
            "args": [
                "--gc-lib-path=<<GC_LIB_PATH>>",
                "--llvm-lib-path=<<LLVM_LIB_PATH>>",
                "--tslang-lib-path=<<TSLANG_LIB_PATH>>",
                "--default-lib-path=<<DEFAULT_LIB_PATH>>",
                "--no-default-lib",
                "--opt",
                "--opt_level=3",
                "--emit=exe",
                "${file}"
            ],
            "group": {
                "kind": "build",
                "isDefault": false
            },
            "presentation": {
                "reveal": "silent"
            },
            "problemMatcher": "$msCompile"
        }        
    ]
}
)raw";

const auto LAUNCH_JSON_DATA_WIN32 = R"raw(
{
    // Use IntelliSense to learn about possible attributes.
    // Hover to view descriptions of existing attributes.
    // For more information, visit: https://go.microsoft.com/fwlink/?linkid=830387
    "version": "0.2.0",
    "configurations": [
        {
            "name": "Current File - EXE (Debug)",
            "type": "cppvsdbg",
            "preLaunchTask": "build EXE (Debug)",
            "request": "launch",
            "program": "${workspaceFolder}/${fileBasenameNoExtension}.exe",
            "args": [
            ],
            "stopAtEntry": false,
            "cwd": "${workspaceFolder}",
            "symbolSearchPath": "${workspaceFolder}",
            "environment": [
            ],
            "visualizerFile": "${workspaceFolder}/tslang.natvis"
        },         
        {
            "name": "Current File - EXE (Release)",
            "type": "cppvsdbg",
            "preLaunchTask": "build EXE (Release)",
            "request": "launch",
            "program": "${fileBasenameNoExtension}.exe",
            "args": [
            ],
            "stopAtEntry": false,
            "cwd": "${workspaceFolder}",
            "symbolSearchPath": "${workspaceFolder}",
            "environment": [
            ],
            "visualizerFile": "${workspaceFolder}/tslang.natvis"
        },         
        {
            "name": "Current File - JIT",
            "type": "cppvsdbg",
            "request": "launch",
            "program": "<<TSLANG_CMD>>",
            "args": [
                "--shared-libs=<<TSLANG_LIB_PATH>>\\TypeScriptRuntime.dll",
                "--gc-lib-path=<<GC_LIB_PATH>>",
                "--llvm-lib-path=<<LLVM_LIB_PATH>>",
                "--tslang-lib-path=<<TSLANG_LIB_PATH>>",
                "--default-lib-path=<<DEFAULT_LIB_PATH>>",
                "--no-default-lib",
                "--opt",
                "--opt_level=3",                
                "--emit=jit",
                "${file}"                
            ],
            "stopAtEntry": false,
            "cwd": "${workspaceFolder}",
            "symbolSearchPath": "${workspaceFolder}",
            "environment": [
            ]
        }
    ]
}
)raw";

const auto LAUNCH_JSON_DATA_LINUX = R"raw(
{
    // Use IntelliSense to learn about possible attributes.
    // Hover to view descriptions of existing attributes.
    // For more information, visit: https://go.microsoft.com/fwlink/?linkid=830387
    "version": "0.2.0",
    "configurations": [
        {
            "name": "Current File - EXE (Debug)",
            "type": "cppdbg",
            "preLaunchTask": "build EXE (Debug)",
            "request": "launch",
            "program": "${workspaceFolder}/${fileBasenameNoExtension}",
            "args": [
            ],
            "stopAtEntry": false,
            "cwd": "${workspaceFolder}",
            "environment": [
            ],
            "visualizerFile": "${workspaceFolder}/tslang.natvis"
        },         
        {
            "name": "Current File - EXE (Release)",
            "type": "cppdbg",
            "preLaunchTask": "build EXE (Release)",
            "request": "launch",
            "program": "${fileBasenameNoExtension}",
            "args": [
            ],
            "stopAtEntry": false,
            "cwd": "${workspaceFolder}",
            "environment": [
            ],
            "visualizerFile": "${workspaceFolder}/tslang.natvis"
        },         
        {
            "name": "Current File - JIT",
            "type": "cppdbg",
            "request": "launch",
            "program": "<<TSLANG_CMD>>",
            "args": [
                "--shared-libs=<<TSLANG_LIB_PATH>>/libTypeScriptRuntime.so",
                "--gc-lib-path=<<GC_LIB_PATH>>",
                "--llvm-lib-path=<<LLVM_LIB_PATH>>",
                "--tslang-lib-path=<<TSLANG_LIB_PATH>>",
                "--default-lib-path=<<DEFAULT_LIB_PATH>>",
                "--no-default-lib",
                "--opt",
                "--opt_level=3",                
                "--emit=jit",
                "${file}"                
            ],
            "stopAtEntry": false,
            "cwd": "${workspaceFolder}",
            "environment": [
            ]
        }
    ]
}
)raw";

#define CMAKE_FOLDER_PATH "cmake"

const auto CMAKE_LISTS_TXT_DATA = R"raw(cmake_minimum_required(VERSION 3.20)

# Make CMake find ts-language modules
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

project(<<PROJECT>> CXX)

# Enable TS-language
enable_language(TSLANG)

# Include folders
include_directories(${CMAKE_TSLANG_DIR}/defaultlib)

# Lib folders
link_directories(${CMAKE_TSLANG_DIR} ${CMAKE_TSLANG_DIR}/defaultlib/lib)

# set options
if (CMAKE_BUILD_TYPE STREQUAL "Release")
	set(CMAKE_TSLANG_FLAGS "--opt --opt_level=3") # global
else()
	set(CMAKE_TSLANG_FLAGS "--di --opt_level=0") # global
endif()

if(WIN32)
else()
    set(CMAKE_TSLANG_FLAGS "${CMAKE_TSLANG_FLAGS} -relocation-model=pic") # global
endif()

# .ts files compile with TSLANG command; .cpp with the C++ compiler.
add_executable(${PROJECT_NAME}
    main.cpp
    mycode.ts
    adder.ts
)

# required libs
set(TSLANG_LINK_LIBS "TypeScriptDefaultLib" "TypeScriptAsyncRuntime" "gc" "LLVMSupport")

# ntdll provides RtlGetLastNtStatus (pulled in by LLVMSupport) on Windows
if(WIN32)
    list(APPEND TSLANG_LINK_LIBS "ntdll")
else()
    list(APPEND TSLANG_LINK_LIBS "LLVMDemangle" "stdc++" "m" "pthread" "tinfo" "dl" "rt")
endif()

target_link_libraries(${PROJECT_NAME} ${TSLANG_LINK_LIBS})
)raw";

const auto CMAKE_PRESETS_JSON_DATA = R"raw({
  "version": 3,
  "cmakeMinimumRequired": {
    "major": 3,
    "minor": 20,
    "patch": 0
  },
  "configurePresets": [
    {
      "name": "default",
      "displayName": "Default (Ninja)",
      "description": "TSLANG custom language requires the Ninja generator (the Visual Studio generator ignores custom languages).",
      "generator": "Ninja",
      "binaryDir": "${sourceDir}/build"
    }
  ],
  "buildPresets": [
    {
      "name": "default",
      "configurePreset": "default"
    }
  ]
}
)raw";

const auto CMAKE_MAIN_CPP_DATA = R"raw(// Declare the symbols your .foo object exports.
// Use extern "C" so names match your compiler's output (no C++ mangling).
extern "C" int  foo_add(int a, int b);
extern "C" void foo_hello(void);

#include <cstdio>

int main() {
    foo_hello();
    std::printf("foo_add(2,3) = %d\n", foo_add(2, 3));
    return 0;
}
)raw";

const auto CMAKE_MYCODE_TS_DATA = R"raw(// Example source in TypeScript language.
// `tslang` compiler turns this into mycode.obj, which CMake links
// with main.cpp. Replace with real TypeScript syntax; the symbols exported
// must match the extern "C" declarations in main.cpp.

import './adder'

export function foo_add(a: int, b: int): int {
    const adder = new Adder(a, b);
    return adder.result;
}

export function foo_hello() {
    console.log("hello from foo");
}
)raw";

const auto CMAKE_ADDER_TS_DATA = R"raw(async function adder(a = 0, b = 0) {
    return a + b;
}

export class Adder {
	#a: int;
	#b: int;

	constructor(a: int, b: int) {
		this.#a = a;
		this.#b = b;
	}

	get result() { return await adder(this.#a, this.#b); }
}
)raw";

const auto CMAKE_README_MD_DATA = R"raw(# Custom language with your own extension and compile command in CMake

This registers TypeScript CMake *language* (`TSLANG`) that:

- owns its own source extension (`.ts`),
- is built with **tslang --emit=obj**,
- produces `.obj` files that CMake links automatically alongside regular C++.

CMake then handles dependency tracking and incremental builds for `.ts`
sources the same way it does for `.cpp`.

## Layout

```
custom_lang/
├── CMakeLists.txt
├── cmake/
│   ├── CMakeDetermineTSLANGCompiler.cmake
│   ├── CMakeTSLANGCompiler.cmake.in
│   ├── CMakeTSLANGInformation.cmake
│   └── CMakeTestTSLANGCompiler.cmake
├── main.cpp
└── mycode.ts
```

## How it works

- `enable_language(TSLANG)` runs `CMakeDetermineTSLANGCompiler.cmake`, which finds
  `tslangc`, then loads `CMakeTSLANGInformation.cmake`, which registers the compile
  rule (`CMAKE_TSLANG_COMPILE_OBJECT` — *tslang --emit=obj*).
- Because `CMAKE_TSLANG_SOURCE_FILE_EXTENSIONS` contains `tslang`, any `.ts` source
  is routed to your command and compiled to a `.obj`.
- That `.obj` is added to the target and linked with `main.cpp`'s object using
  the C++ linker (`CMAKE_TSLANG_LINK_EXECUTABLE`).
- Change `mycode.ts` and only it recompiles.
- Compile: `cmake --preset default && cmake --build --preset default`

## Passing flags

```cmake
set(CMAKE_TSLANG_FLAGS "--opt_level=3")                       # global
set_source_files_properties(mycode.ts PROPERTIES
    COMPILE_OPTIONS "--define;TSLANG=1")                      # per-file
```

## Minimal alternative

If you don't need a first-class language, either:

- Mark a file as an already-built object and just link it:
  `set_source_files_properties(mycode.obj PROPERTIES EXTERNAL_OBJECT TRUE GENERATED TRUE)`
  and add it to `add_executable`, producing it with `add_custom_command`; or
- Compile a differently-named file *as C++*:
  `set_source_files_properties(mycode.tslang PROPERTIES LANGUAGE CXX)`.

Use the typescript-language setup below when `.ts` is a source type you
compile often and want CMake to treat as first-class.
)raw";

const auto CMAKE_DETERMINE_TSLANG_COMPILER_DATA = R"raw(# Locate your custom compiler
find_program(CMAKE_TSLANG_COMPILER
    NAMES tslang tslang.exe
    HINTS "${CMAKE_SOURCE_DIR}/tools"
    DOC "TSLANG compiler")

cmake_path(GET CMAKE_TSLANG_COMPILER PARENT_PATH CMAKE_TSLANG_DIR)

mark_as_advanced(CMAKE_TSLANG_COMPILER)
mark_as_advanced(CMAKE_TSLANG_DIR)

# Which source extensions belong to TSLANG, and the object suffix
set(CMAKE_TSLANG_SOURCE_FILE_EXTENSIONS ts)
if (NOT WIN32)
	set(CMAKE_TSLANG_OUTPUT_EXTENSION .o)
else()
	set(CMAKE_TSLANG_OUTPUT_EXTENSION .obj)   # .o on Linux
endif()
set(CMAKE_TSLANG_COMPILER_ENV_VAR "TSLANG")

# Emit the compiler-id config file CMake expects
configure_file(
    ${CMAKE_CURRENT_LIST_DIR}/CMakeTSLANGCompiler.cmake.in
    ${CMAKE_PLATFORM_INFO_DIR}/CMakeTSLANGCompiler.cmake @ONLY)
)raw";

const auto CMAKE_TEST_TSLANG_COMPILER_DATA = R"raw(# Skip the actual test compile; assume the compiler works.
set(CMAKE_TSLANG_COMPILER_WORKS TRUE)
)raw";

const auto CMAKE_TSLANG_COMPILER_IN_DATA = R"raw(set(CMAKE_TSLANG_COMPILER "@CMAKE_TSLANG_COMPILER@")
set(CMAKE_TSLANG_DIR "@CMAKE_TSLANG_DIR@")
set(CMAKE_TSLANG_SOURCE_FILE_EXTENSIONS @CMAKE_TSLANG_SOURCE_FILE_EXTENSIONS@)
set(CMAKE_TSLANG_OUTPUT_EXTENSION @CMAKE_TSLANG_OUTPUT_EXTENSION@)
set(CMAKE_TSLANG_COMPILER_LOADED 1)
set(CMAKE_TSLANG_COMPILER_WORKS TRUE)
)raw";

const auto CMAKE_TSLANG_INFORMATION_DATA = R"raw(# The actual compile command.
# Placeholders CMake substitutes:
#   <CMAKE_TSLANG_COMPILER>  the binary
#   <FLAGS>               per-target flags
#   <SOURCE>              input .ts
#   <OBJECT>              output .obj
#   <DEFINES> <INCLUDES>  optional
if(NOT CMAKE_TSLANG_COMPILE_OBJECT)
    set(CMAKE_TSLANG_COMPILE_OBJECT
        "<CMAKE_TSLANG_COMPILER> <FLAGS> --default-lib-path=${CMAKE_TSLANG_DIR} --emit=obj --export=none -o=<OBJECT> <SOURCE>")
endif()

# How CMake links TSLANG objects into an executable/library.
# Reuse the C++ linker so linking with .cpp works out of the box.
if(NOT CMAKE_TSLANG_LINK_EXECUTABLE)
    set(CMAKE_TSLANG_LINK_EXECUTABLE
        "<CMAKE_CXX_COMPILER> <FLAGS> <OBJECTS> -o <TARGET> <LINK_LIBRARIES>")
endif()

set(CMAKE_TSLANG_INFORMATION_LOADED 1)
)raw";

const auto TSLANG_NATVIS = R"raw(<?xml version="1.0" encoding="utf-8"?>
<AutoVisualizer xmlns="http://schemas.microsoft.com/vstudio/debugger/natvis/2010">

  <!-- TypeScript -->

  <Type Name="array&lt;*&gt;">
    <DisplayString>array of {length} elements</DisplayString>
    <Expand>
      <Item Name="[length]">length</Item>
      <ArrayItems>
        <Size>length</Size>
        <ValuePointer>data</ValuePointer>
      </ArrayItems>
    </Expand>
  </Type>  

</AutoVisualizer>

)raw";
