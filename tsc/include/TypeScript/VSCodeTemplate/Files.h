#define NODE_MODULE_TSNC_PATH "node_modules/tsnc"
#define DOT_VSCODE_PATH ".vscode"

#define TSCONFIG_JSON_DATA "{\r\n    \"compilerOptions\": {\r\n      \"target\": \"es2017\",\r\n      \"lib\": [\"dom\", \"dom.iterable\", \"esnext\"],\r\n      \"allowJs\": true,\r\n      \"skipLibCheck\": true,\r\n      \"strict\": true,\r\n      \"noEmit\": true,\r\n      \"esModuleInterop\": true,\r\n      \"module\": \"esnext\",\r\n      \"moduleResolution\": \"bundler\",\r\n      \"resolveJsonModule\": true,\r\n      \"isolatedModules\": true,\r\n      \"jsx\": \"preserve\",\r\n      \"incremental\": true,\r\n      \"types\": [\"tsnc\"]\r\n    },\r\n    \"include\": [\"<<PROJECT>>.ts\"],\r\n    \"exclude\": [\"node_modules\"]\r\n  }"
#define TSNC_INDEX_D_TS "declare function print(...args: (string | number)[]) : void;\r\ndeclare function assert(cond: boolean, msg?: string) : void;\r\ndeclare type int = any;"
#define TASKS_JSON_DATA "{\r\n    // See https://go.microsoft.com/fwlink/?LinkId=733558\r\n    // for the documentation about the tasks.json format\r\n    \"version\": \"2.0.0\",\r\n    \"tasks\": [\r\n        {\r\n            \"label\": \"build EXE (Debug)\",\r\n            \"command\": \"<<TSC_CMD>>\",\r\n            \"type\": \"shell\",\r\n            \"args\": [\r\n                \"--gc-lib-path=<<GC_LIB_PATH>>\",\r\n                \"--llvm-lib-path=<<LLVM_LIB_PATH>>\",\r\n                \"--tsc-lib-path=<<TSC_LIB_PATH>>\",\r\n                \"--default-lib-path=<<DEFAULT_LIB_PATH>>\",\r\n                \"--no-default-lib\",\r\n                \"--di\",\r\n                \"--opt_level=0\",\r\n                \"--emit=exe\",\r\n                \"${file}\"\r\n            ],\r\n            \"group\": {\r\n                \"kind\": \"build\",\r\n                \"isDefault\": true\r\n            },\r\n            \"presentation\": {\r\n                \"reveal\": \"silent\"\r\n            },\r\n            \"problemMatcher\": \"$msCompile\"\r\n        },\r\n        {\r\n            \"label\": \"build EXE (Release)\",\r\n            \"command\": \"<<TSC_CMD>>\",\r\n            \"type\": \"shell\",\r\n            \"args\": [\r\n                \"--gc-lib-path=<<GC_LIB_PATH>>\",\r\n                \"--llvm-lib-path=<<LLVM_LIB_PATH>>\",\r\n                \"--tsc-lib-path=<<TSC_LIB_PATH>>\",\r\n                \"--default-lib-path=<<DEFAULT_LIB_PATH>>\",\r\n                \"--no-default-lib\",\r\n                \"--opt\",\r\n                \"--opt_level=3\",\r\n                \"--emit=exe\",\r\n                \"${file}\"\r\n            ],\r\n            \"group\": {\r\n                \"kind\": \"build\",\r\n                \"isDefault\": false\r\n            },\r\n            \"presentation\": {\r\n                \"reveal\": \"silent\"\r\n            },\r\n            \"problemMatcher\": \"$msCompile\"\r\n        },\r\n        {\r\n            \"label\": \"build DLL (Debug)\",\r\n            \"command\": \"<<TSC_CMD>>\",\r\n            \"type\": \"shell\",\r\n            \"args\": [\r\n                \"--gc-lib-path=<<GC_LIB_PATH>>\",\r\n                \"--llvm-lib-path=<<LLVM_LIB_PATH>>\",\r\n                \"--tsc-lib-path=<<TSC_LIB_PATH>>\",\r\n                \"--default-lib-path=<<DEFAULT_LIB_PATH>>\",\r\n                \"--no-default-lib\",\r\n                \"--di\",\r\n                \"--opt_level=0\",\r\n                \"--emit=exe\",\r\n                \"${file}\"\r\n            ],\r\n            \"group\": {\r\n                \"kind\": \"build\",\r\n                \"isDefault\": false\r\n            },\r\n            \"presentation\": {\r\n                \"reveal\": \"silent\"\r\n            },\r\n            \"problemMatcher\": \"$msCompile\"\r\n        },\r\n        {\r\n            \"label\": \"build DLL (Release)\",\r\n            \"command\": \"<<TSC_CMD>>\",\r\n            \"type\": \"shell\",\r\n            \"args\": [\r\n                \"--gc-lib-path=<<GC_LIB_PATH>>\",\r\n                \"--llvm-lib-path=<<LLVM_LIB_PATH>>\",\r\n                \"--tsc-lib-path=<<TSC_LIB_PATH>>\",\r\n                \"--default-lib-path=<<DEFAULT_LIB_PATH>>\",\r\n                \"--no-default-lib\",\r\n                \"--opt\",\r\n                \"--opt_level=3\",\r\n                \"--emit=exe\",\r\n                \"${file}\"\r\n            ],\r\n            \"group\": {\r\n                \"kind\": \"build\",\r\n                \"isDefault\": false\r\n            },\r\n            \"presentation\": {\r\n                \"reveal\": \"silent\"\r\n            },\r\n            \"problemMatcher\": \"$msCompile\"\r\n        }        \r\n    ]\r\n}"
#define LAUNCH_JSON_DATA_WIN32 "{\r\n    // Use IntelliSense to learn about possible attributes.\r\n    // Hover to view descriptions of existing attributes.\r\n    // For more information, visit: https://go.microsoft.com/fwlink/?linkid=830387\r\n    \"version\": \"0.2.0\",\r\n    \"configurations\": [\r\n        {\r\n            \"name\": \"Current File - EXE (Debug)\",\r\n            \"type\": \"cppvsdbg\",\r\n            \"preLaunchTask\": \"build EXE (Debug)\",\r\n            \"request\": \"launch\",\r\n            \"program\": \"${fileBasenameNoExtension}.exe\",\r\n            \"args\": [\r\n            ],\r\n            \"stopAtEntry\": false,\r\n            \"cwd\": \"${workspaceFolder}\",\r\n            \"symbolSearchPath\": \"${workspaceFolder}\",\r\n            \"environment\": [\r\n            ],\r\n            \"visualizerFile\": \"${workspaceFolder}/tsnc.natvis\"\r\n        },         \r\n        {\r\n            \"name\": \"Current File - EXE (Release)\",\r\n            \"type\": \"cppvsdbg\",\r\n            \"preLaunchTask\": \"build EXE (Release)\",\r\n            \"request\": \"launch\",\r\n            \"program\": \"${fileBasenameNoExtension}.exe\",\r\n            \"args\": [\r\n            ],\r\n            \"stopAtEntry\": false,\r\n            \"cwd\": \"${workspaceFolder}\",\r\n            \"symbolSearchPath\": \"${workspaceFolder}\",\r\n            \"environment\": [\r\n            ],\r\n            \"visualizerFile\": \"${workspaceFolder}/tsnc.natvis\"\r\n        },         \r\n        {\r\n            \"name\": \"Current File - JIT\",\r\n            \"type\": \"cppvsdbg\",\r\n            \"request\": \"launch\",\r\n            \"program\": \"<<TSC_CMD>>\",\r\n            \"args\": [\r\n                \"--shared-libs=<<TSC_LIB_PATH>>\\\\TypeScriptRuntime.dll\",\r\n                \"--gc-lib-path=<<GC_LIB_PATH>>\",\r\n                \"--llvm-lib-path=<<LLVM_LIB_PATH>>\",\r\n                \"--tsc-lib-path=<<TSC_LIB_PATH>>\",\r\n                \"--default-lib-path=<<DEFAULT_LIB_PATH>>\",\r\n                \"--no-default-lib\",\r\n                \"--opt\",\r\n                \"--opt_level=3\",                \r\n                \"--emit=jit\",\r\n                \"${file}\"                \r\n            ],\r\n            \"stopAtEntry\": false,\r\n            \"cwd\": \"${workspaceFolder}\",\r\n            \"symbolSearchPath\": \"${workspaceFolder}\",\r\n            \"environment\": [\r\n            ]\r\n        }\r\n    ]\r\n}"
#define LAUNCH_JSON_DATA_LINUX "{\r\n    // Use IntelliSense to learn about possible attributes.\r\n    // Hover to view descriptions of existing attributes.\r\n    // For more information, visit: https://go.microsoft.com/fwlink/?linkid=830387\r\n    \"version\": \"0.2.0\",\r\n    \"configurations\": [\r\n        {\r\n            \"name\": \"Current File - EXE (Debug)\",\r\n            \"type\": \"cppdbg\",\r\n            \"preLaunchTask\": \"build EXE (Debug)\",\r\n            \"request\": \"launch\",\r\n            \"program\": \"${fileBasenameNoExtension}\",\r\n            \"args\": [\r\n            ],\r\n            \"stopAtEntry\": false,\r\n            \"cwd\": \"${workspaceFolder}\",\r\n            \"environment\": [\r\n            ],\r\n            \"visualizerFile\": \"${workspaceFolder}/tsnc.natvis\"\r\n        },         \r\n        {\r\n            \"name\": \"Current File - EXE (Release)\",\r\n            \"type\": \"cppdbg\",\r\n            \"preLaunchTask\": \"build EXE (Release)\",\r\n            \"request\": \"launch\",\r\n            \"program\": \"${fileBasenameNoExtension}\",\r\n            \"args\": [\r\n            ],\r\n            \"stopAtEntry\": false,\r\n            \"cwd\": \"${workspaceFolder}\",\r\n            \"environment\": [\r\n            ],\r\n            \"visualizerFile\": \"${workspaceFolder}/tsnc.natvis\"\r\n        },         \r\n        {\r\n            \"name\": \"Current File - JIT\",\r\n            \"type\": \"cppdbg\",\r\n            \"request\": \"launch\",\r\n            \"program\": \"<<TSC_CMD>>\",\r\n            \"args\": [\r\n                \"--shared-libs=<<TSC_LIB_PATH>>/libTypeScriptRuntime.so\",\r\n                \"--gc-lib-path=<<GC_LIB_PATH>>\",\r\n                \"--llvm-lib-path=<<LLVM_LIB_PATH>>\",\r\n                \"--tsc-lib-path=<<TSC_LIB_PATH>>\",\r\n                \"--default-lib-path=<<DEFAULT_LIB_PATH>>\",\r\n                \"--no-default-lib\",\r\n                \"--opt\",\r\n                \"--opt_level=3\",                \r\n                \"--emit=jit\",\r\n                \"${file}\"                \r\n            ],\r\n            \"stopAtEntry\": false,\r\n            \"cwd\": \"${workspaceFolder}\",\r\n            \"environment\": [\r\n            ]\r\n        }\r\n    ]\r\n}"