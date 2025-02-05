#define NODE_MODULE_TSNC_PATH "node_modules/tsnc"
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
      "types": ["tsnc"]
    },
    "include": ["<<PROJECT>>.ts"],
    "exclude": ["node_modules"]
}
)raw";

const auto TSNC_INDEX_D_TS = R"raw(
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

type Reference<T> = any

declare function ReferenceOf<T>(r: T): Reference<T>;
declare function LoadReference<T>(r: Reference<T>): T;

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
            "command": "<<TSC_CMD>>",
            "type": "shell",
            "args": [
                "--gc-lib-path=<<GC_LIB_PATH>>",
                "--llvm-lib-path=<<LLVM_LIB_PATH>>",
                "--tsc-lib-path=<<TSC_LIB_PATH>>",
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
            "command": "<<TSC_CMD>>",
            "type": "shell",
            "args": [
                "--gc-lib-path=<<GC_LIB_PATH>>",
                "--llvm-lib-path=<<LLVM_LIB_PATH>>",
                "--tsc-lib-path=<<TSC_LIB_PATH>>",
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
            "command": "<<TSC_CMD>>",
            "type": "shell",
            "args": [
                "--gc-lib-path=<<GC_LIB_PATH>>",
                "--llvm-lib-path=<<LLVM_LIB_PATH>>",
                "--tsc-lib-path=<<TSC_LIB_PATH>>",
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
            "command": "<<TSC_CMD>>",
            "type": "shell",
            "args": [
                "--gc-lib-path=<<GC_LIB_PATH>>",
                "--llvm-lib-path=<<LLVM_LIB_PATH>>",
                "--tsc-lib-path=<<TSC_LIB_PATH>>",
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
            "visualizerFile": "${workspaceFolder}/tsnc.natvis"
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
            "visualizerFile": "${workspaceFolder}/tsnc.natvis"
        },         
        {
            "name": "Current File - JIT",
            "type": "cppvsdbg",
            "request": "launch",
            "program": "<<TSC_CMD>>",
            "args": [
                "--shared-libs=<<TSC_LIB_PATH>>\\TypeScriptRuntime.dll",
                "--gc-lib-path=<<GC_LIB_PATH>>",
                "--llvm-lib-path=<<LLVM_LIB_PATH>>",
                "--tsc-lib-path=<<TSC_LIB_PATH>>",
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
            "visualizerFile": "${workspaceFolder}/tsnc.natvis"
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
            "visualizerFile": "${workspaceFolder}/tsnc.natvis"
        },         
        {
            "name": "Current File - JIT",
            "type": "cppdbg",
            "request": "launch",
            "program": "<<TSC_CMD>>",
            "args": [
                "--shared-libs=<<TSC_LIB_PATH>>/libTypeScriptRuntime.so",
                "--gc-lib-path=<<GC_LIB_PATH>>",
                "--llvm-lib-path=<<LLVM_LIB_PATH>>",
                "--tsc-lib-path=<<TSC_LIB_PATH>>",
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

const auto TSNC_NATVIS = R"raw(<?xml version="1.0" encoding="utf-8"?>
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
