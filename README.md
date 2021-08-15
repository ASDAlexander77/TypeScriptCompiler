# TypeScript Native Compiler
###### Powered by [![LLVM|MLIR](https://llvm.org/img/LLVM-Logo-Derivative-1.png)](https://llvm.org/)

<p align="left">
  <a aria-label="License" href="https://github.com/ASDAlexander77/TypeScriptCompiler/blob/main/LICENSE">
    <img alt="" src="https://img.shields.io/npm/l/next.svg?style=for-the-badge&labelColor=000000">
  </a>
</p>

# Demo

[![Demo](https://asdalexander77.github.io/img/tsc_emit.gif)](https://github.com/ASDAlexander77/TypeScriptCompiler/releases/)

Chat Room
---------

Want to chat with other members of the TypeScriptCompiler community?

[![Join the chat at https://gitter.im/ASDAlexander77/TypeScriptCompiler](https://badges.gitter.im/Join%20Chat.svg)](https://gitter.im/ASDAlexander77/TypeScriptCompiler?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge)
[![Test Build (Windows)](https://github.com/ASDAlexander77/TypeScriptCompiler/actions/workflows/cmake-release.yml/badge.svg)](https://github.com/ASDAlexander77/TypeScriptCompiler/actions/workflows/cmake-release.yml)

# Example

```TypeScript
abstract class Department {
    constructor(public name: string) {}

    printName(): void {
        print("Department name: " + this.name);
    }

    abstract printMeeting(): void; // must be implemented in derived classes
}

class AccountingDepartment extends Department {
    constructor() {
        super("Accounting and Auditing"); // constructors in derived classes must call super()
    }

    printMeeting(): void {
        print("The Accounting Department meets each Monday at 10am.");
    }

    generateReports(): void {
        print("Generating accounting reports...");
    }
}

function main() {
    let department: Department; // ok to create a reference to an abstract type
    department = new AccountingDepartment(); // ok to create and assign a non-abstract subclass
    department.printName();
    department.printMeeting();
    //department.generateReports(); // error: department is not of type AccountingDepartment, cannot access generateReports
}
```

Run
```cmd
tsc --emit=jit example.ts
```

Result
```
Department name: Accounting and Auditing
The Accounting Department meets each Monday at 10am.
```

## Compile as JIT

```cmd
tsc --emit=jit hello.ts
```
File ``hello.ts``

```TypeScript
function main() {
    print("Hello World!");
}
```
Result
```
Hello World!
```

## Compile as Binary Executable

### On Windows
File ``tsc-compile.bat``
```cmd
rem set %LLVM% and %TSCBIN%
set LLVMPATH=%LLVM%\llvm\release\bin
set TSCPATH=%TSCBIN%\tsc\bin
set VCFLD=C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\SDK\ScopeCppSDK\vc15
set LIBPATH=%VCFLD%\VC\lib
set SDKPATH=%VCFLD%\SDK\lib
%TSCPATH%\tsc.exe --emit=jit -dump-object-file -object-filename=%FILENAME%.o %FILENAME%.ts
%LLVMPATH%\lld.exe -flavor link %FILENAME%.o /libpath:"%LIBPATH%" /libpath:"%SDKPATH%" "%LIBPATH%\libcmt.lib" "%LIBPATH%\libvcruntime.lib" "%SDKPATH%\kernel32.lib" "%SDKPATH%\libucrt.lib" "%SDKPATH%\uuid.lib"
```
Compile 
```cmd
tsc-compile.bat hello
```

Run
```
hello.exe
```

Result
```
Hello World!
```

### On Linux (Ubuntu 20.04)
File ``tsc-compile.sh``
```cmd
./tsc --emit=jit -dump-object-file -object-filename=$1.o $1.ts
gcc -o $1 $1.o
```
Compile 
```cmd
sh -f tsc-compile.sh hello
```

Run
```
./hello
```

Result
```
Hello World!
```

### Compiling as WASM
### On Windows
File ``tsc-compile-wasm.bat``
```cmd
rem set %LLVM% and %TSCBIN%
set LLVMPATH=%LLVM%\llvm\release\bin
set TSCPATH=%TSCBIN%\tsc\bin
%TSCPATH%\tsc.exe --emit=llvm %FILENAME%.ts 2>%FILENAME%.ll
%LLVMPATH%\llc.exe -mtriple=wasm32-unknown-unknown -O3 --filetype=obj -o=%FILENAME%.o %FILENAME%.ll
%LLVMPATH%\wasm-ld.exe %FILENAME%.o -o %FILENAME%.wasm --no-entry --export-all --allow-undefined
```
Compile 
```cmd
tsc-compile-wasm.bat hello
```

Run ``run.html``
```
<!DOCTYPE html>
<html>
  <head></head>
  <body>
    <script type="module">
let buffer;

const config = {
    env: {
        memory_base: 0,
        table_base: 0,
        memory : new WebAssembly.Memory({ initial: 256}),
        table: new WebAssembly.Table({
            initial: 0,
            element: 'anyfunc',
        })
    }
};

fetch("./hello.wasm")
    .then(response =>{
        return response.arrayBuffer();
    })
    .then(bytes => {
        return WebAssembly.instantiate(bytes, config); 
    })
    .then(results => { 
       let { main } =  results.instance.exports;
       buffer = new Uint8Array(results.instance.exports.memory.buffer);
       main();
    });
    </script>
  </body>
</html>
```

## Build
### On Windows

First, precompile dependencies

```
prepare_3rdParty.bat 
```

To build TSC binaries:

```
cd tsc
config_tsc_debug.bat
build_tsc_debug.bat
```

### On Linux (Ubuntu 20.04)

First, precompile dependencies

```
sh -f prepare_3rdParty.sh
```

To build ``TSC`` binaries:

```
cd tsc
sh -f config_tsc_debug.sh
sh -f build_tsc_debug.sh
```
