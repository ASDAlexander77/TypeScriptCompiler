# TypeScript Native Compiler
###### Powered by [![LLVM|MLIR](https://llvm.org/img/LLVM-Logo-Derivative-1.png)](https://llvm.org/)

[![Donate](https://img.shields.io/badge/Donate-PayPal-green.svg)](https://www.paypal.com/donate/?hosted_button_id=BBJ4SQYLA6D2L)

# Build

[![Test Build (Windows)](https://github.com/ASDAlexander77/TypeScriptCompiler/actions/workflows/cmake-test-release-win.yml/badge.svg)](https://github.com/ASDAlexander77/TypeScriptCompiler/actions/workflows/cmake-test-release-win.yml)
[![Test Build (Linux)](https://github.com/ASDAlexander77/TypeScriptCompiler/actions/workflows/cmake-test-release-linux.yml/badge.svg)](https://github.com/ASDAlexander77/TypeScriptCompiler/actions/workflows/cmake-test-release-linux.yml)

# Demo [release](https://github.com/ASDAlexander77/TypeScriptCompiler/releases/)

[![Demo](https://asdalexander77.github.io/img/tsc_emit.gif)](https://github.com/ASDAlexander77/TypeScriptCompiler/releases/)


# Try it [here](https://godbolt.org/#z:OYLghAFBqd5QCxAYwPYBMCmBRdBLAF1QCcAaPECAMzwBtMA7AQwFtMQByARg9KtQYEAysib0QXAGx8BBAKoBnTAAUAHpwAMvAFYTStJg1AEAngAdMC5MTxmCpJfWQE8Ayo3QBhVLQCuLBhAAJlIPABk8BkwAOX8AI0xiEAAWUjNUBUJXBm8/AOC0jKyBCKjYlgSk1MdMZ2yhAiZiAlz/QJCauoEGpoJSmPjElIdG5tb8jtG%2ByIGKoeSASgdUX2Jkdg4qXwYuhgBqFiZIiAWAUg0AQVOAdgAhc4u9p72zG0EIU6CggAlMWlpUHsAOokWjoMBgT5BM4AZnulxuABEOEtaJwAKy8QIcLSkVCcAAq5kwIhsdj2ChWa0we0%2BMJ4pAImhRSwA1sF0QA6IIADiCMJ5AE5klxeVwYYLBfpOMkscy8ZxeAoQBpGcylnBYEg8GQKBA0CwzHREtFWBsiRZSbYCHsNJy7a9MEENHsAFIASQJewgAHFPAteJh8EQbOg9PxBCIxOwuKkI/IlGp5bouA4/rUXG4IB5xoFU%2BEZuVKnp0plMzkfG0S0Vy/0i0NU51yz0xpX8o307sW9MyoMko2prm9AopnW%2BxIlgRiJhMHgw9wUdKOJjSNjcfiOGEwgA1ACye3dACU9tu8JgAO6JPYWknWa22%2B2cx3Ot2e71%2Bhbe3CEEi0/mpvZvENY1iD/ekAzVHEWVIBBMCYLAkhOUh2RFTlJUldEYXRWMNCCdF%2BSXWVV3lDclRVSCtA1GAoGoiAkEwVRal8ENyEoJpgAUZRDD%2BIQEFQc9sQZA0jQMcsuKiWheP4tdeGEkCQAIKwAH0NBUpTnSU4BkFIOT6GIAB5ZipIE%2BUGNqC5iA4xVQkY5AGnwbFeHjKNxFjGRBEUFR1Cg0gU30QxjGJKwyXsWg8DiZVICWVA7GyZUOCVKl1mHAgHPEni%2BJMhdGWnDYGXPYgmDMTgeFRDE5R8jdsFs5jfxvK1yTtB1pxfD0vV9f1vXqu87FIPZvxDMCuAgpkoIWNkQBhIJOWuLhrkkdFrg0a56UkZapTRDgiJYEBrnRYjKus5VVVGyjSE1RAUFQYC9NY/VrpEoZFOQNTVI0rS%2BDoAhEmVCA4nlOJIiaEwStk662EEfSGFoEGfKwQ4jHEOGdQzPAADdLFMmrvtB8hBD%2BeUwriQriBMbwsFxqc8B2hclioAwONPC99IsQSnNkFyYzjWRPKTHy/IMIwFKCnrQvCyKkJi8t4qVTty3cBgvDbPNQkVsc5n7Qoy2yIdU1LYoGHV4sOycZtB2V4c5fqUdC3HAdel1kZeiNhslkpVZktTKcZ1Bsrlwq9dCWJBqbSap8WpdNr306iBupCvqBt/OkAKAx7QOTkb1SWWD4KGJDNu2jkDsDhKHHI07oPZKaZrmhalpWqR1qXGEA94UiKOgzaglbhVS4r8bSAx4hMjcZIgA%3D%3D)

[![Compiler Explorer](https://asdalexander77.github.io/img/god_bolt_tsc_native.jpg)](https://godbolt.org/#z:OYLghAFBqd5QCxAYwPYBMCmBRdBLAF1QCcAaPECAMzwBtMA7AQwFtMQByARg9KtQYEAysib0QXAGx8BBAKoBnTAAUAHpwAMvAFYTStJg1AEAngAdMC5MTxmCpJfWQE8Ayo3QBhVLQCuLBhAAJlIPABk8BkwAOX8AI0xiEAAWUjNUBUJXBm8/AOC0jKyBCKjYlgSk1MdMZ2yhAiZiAlz/QJCauoEGpoJSmPjElIdG5tb8jtG%2ByIGKoeSASgdUX2Jkdg4qXwYuhgBqFiZIiAWAUg0AQVOAdgAhc4u9p72zG0EIU6CggAlMWlpUHsAOokWjoMBgT5BM4AZnulxuABEOEtaJwAKy8QIcLSkVCcAAq5kwIhsdj2ChWa0we0%2BMJ4pAImhRSwA1sF0QA6IIADiCMJ5AE5klxeVwYYLBfpOMkscy8ZxeAoQBpGcylnBYEg8GQKBA0CwzHREtFWBsiRZSbYCHsNJy7a9MEENHsAFIASQJewgAHFPAteJh8EQbOg9PxBCIxOwuKkI/IlGp5bouA4/rUXG4IB5xoFU%2BEZuVKnp0plMzkfG0S0Vy/0i0NU51yz0xpX8o307sW9MyoMko2prm9AopnW%2BxIlgRiJhMHgw9wUdKOJjSNjcfiOGEwgA1ACye3dACU9tu8JgAO6JPYWknWa22%2B2cx3Ot2e71%2Bhbe3CEEi0/mpvZvENY1iD/ekAzVHEWVIBBMCYLAkhOUh2RFTlJUldEYXRWMNCCdF%2BSXWVV3lDclRVSCtA1GAoGoiAkEwVRal8ENyEoJpgAUZRDD%2BIQEFQc9sQZA0jQMcsuKiWheP4tdeGEkCQAIKwAH0NBUpTnSU4BkFIOT6GIAB5ZipIE%2BUGNqC5iA4xVQkY5AGnwbFeHjKNxFjGRBEUFR1Cg0gU30QxjGJKwyXsWg8DiZVICWVA7GyZUOCVKl1mHAgHPEni%2BJMhdGWnDYGXPYgmDMTgeFRDE5R8jdsFs5jfxvK1yTtB1pxfD0vV9f1vXqu87FIPZvxDMCuAgpkoIWNkQBhIJOWuLhrkkdFrg0a56UkZapTRDgiJYEBrnRYjKus5VVVGyjSE1RAUFQYC9NY/VrpEoZFOQNTVI0rS%2BDoAhEmVCA4nlOJIiaEwStk662EEfSGFoEGfKwQ4jHEOGdQzPAADdLFMmrvtB8hBD%2BeUwriQriBMbwsFxqc8B2hclioAwONPC99IsQSnNkFyYzjWRPKTHy/IMIwFKCnrQvCyKkJi8t4qVTty3cBgvDbPNQkVsc5n7Qoy2yIdU1LYoGHV4sOycZtB2V4c5fqUdC3HAdel1kZeiNhslkpVZktTKcZ1Bsrlwq9dCWJBqbSap8WpdNr306iBupCvqBt/OkAKAx7QOTkb1SWWD4KGJDNu2jkDsDhKHHI07oPZKaZrmhalpWqR1qXGEA94UiKOgzaglbhVS4r8bSAx4hMjcZIgA%3D%3D)

Chat Room
---------

Want to chat with other members of the TypeScriptCompiler community?

[![Join the chat at https://gitter.im/ASDAlexander77/TypeScriptCompiler](https://badges.gitter.im/Join%20Chat.svg)](https://gitter.im/ASDAlexander77/TypeScriptCompiler?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge)

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
set FILENAME=%1
set LLVMPATH=C:\TypeScriptCompiler\3rdParty\llvm\release\bin
set TSCPATH=C:\TypeScriptCompiler\__build\tsc\bin
set LIBPATH="C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\VC\Tools\MSVC\14.29.30037\lib\x64"
set SDKPATH="C:\Program Files (x86)\Windows Kits\10\Lib\10.0.18362.0\um\x64"
set UCRTPATH="C:\Program Files (x86)\Windows Kits\10\Lib\10.0.18362.0\ucrt\x64"
%TSCPATH%\tsc.exe --emit=jit -nogc -dump-object-file -object-filename=%FILENAME%.o %FILENAME%.ts
%LLVMPATH%\lld.exe -flavor link %FILENAME%.o /libpath:%LIBPATH% /libpath:%SDKPATH% /libpath:%UCRTPATH% /defaultlib:libcmt.lib libvcruntime.lib
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
./tsc --emit=jit -nogc -dump-object-file -object-filename=$1.o $1.ts
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
%TSCPATH%\tsc.exe --emit=llvm -nogc %FILENAME%.ts 2>%FILENAME%.ll
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
