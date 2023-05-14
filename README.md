# TypeScript Native Compiler
###### Powered by [![LLVM|MLIR](https://llvm.org/img/LLVM-Logo-Derivative-1.png)](https://llvm.org/)

[![Donate](https://img.shields.io/badge/Donate-PayPal-green.svg)](https://www.paypal.com/donate/?hosted_button_id=BBJ4SQYLA6D2L)

# Build

[![Test Build (Windows)](https://github.com/ASDAlexander77/TypeScriptCompiler/actions/workflows/cmake-test-release-win.yml/badge.svg)](https://github.com/ASDAlexander77/TypeScriptCompiler/actions/workflows/cmake-test-release-win.yml)
[![Test Build (Linux)](https://github.com/ASDAlexander77/TypeScriptCompiler/actions/workflows/cmake-test-release-linux.yml/badge.svg)](https://github.com/ASDAlexander77/TypeScriptCompiler/actions/workflows/cmake-test-release-linux.yml)

# What's new

- Generic methods
```TypeScript
class Lib {
    static max<T>(a: T, b: T): T {
        return a > b ? a : b;
    }
}

function main() {
    assert(Lib.max(10, 20) == 20);
    assert(Lib.max("a", "b") == "b");
    assert(Lib.max<number>(20, 30) == 30);
    print("done.");
}
```

# Planning
- Migrating to LLVM 16.0.3
- JavaScript Built-in classes library

# Demo 
[(click here)](https://github.com/ASDAlexander77/TypeScriptCompiler/releases/)

[![Demo](https://asdalexander77.github.io/img/tsc_emit.gif)](https://github.com/ASDAlexander77/TypeScriptCompiler/releases/)


# Try it 
[(click here)](https://godbolt.org/#z:OYLghAFBqd5QCxAYwPYBMCmBRdBLAF1QCcAaPECAMzwBtMA7AQwFtMQByARg9KtQYEAysib0QXABx8BBAKoBnTAAUAHpwAMvAFYTStJg1AEAngAdMC5MTxmCpJfWQE8Ayo3QBhVLQCuLBglpDwAZPAZMADl/ACNMYgkuUjNUBUJXBm8/AKDk1PSBMIjoljiEriTHTGcMoQImYgIs/0CpB0wnFwE6hoIiqNj4xId6xuactoVRvvCB0qGKgEoHVF9iZHYOKl8GGoEAKgBqYAhFw4BSAHYAIXONAEFDw5M8DvQj6h29hg%2Bzq9uHk8ni83ocuOcAMwAx5AkG0dCHABMkOhQOer3hhwhKLuMKuABFFqccQ8Cbjcdtdl0GIcWExwqcLjdcWj%2BMRDhA0AwpodVIdUFRjqc/szAWigWYbIIIKpFiSYU8yaTLviOMtaJwAKy8QIcLSkVCcAAq5kwIhsdkOkSYLgAbphDgpVusHedERCeKQCJo1csANYgCEaAB0ABYNBpJKGIZrLqHLpd3VH9JxQ7wWBII6RdfrDRxeAoQBovT7lnBYEg8GQKJzUCwzHR4ta2MbTebbAQrTa8PbDt56432SGQ5LMIjLocAFIASSNHIA4p5FrxMPgiDZ0Hp%2BIIRGJ2FxQzJBIoVOo9To9FVvu4GF4fC09KFZiUyiBpCk0tTxq08p%2BMv0XyGaQr2pHoxnvCZ2k6WppgAwYEmA6Zv0vWDn3gt9lgIYhMEwPBNw0dUtR1H0DU4EIQgANQAWUOacACVDgo14AHd4kOE0LHbS1rTtB1%2Bwbegh2DEdsPHKdZwXJcOVwQgSAud0kj7OsBLYt0PWXEtz0WZYEEwJgsASU5SADUMuGDcdJERTVEXjABOCEADYNFDaQNQ4NNSAzLgsxzXg8wLItNK0MsYCgUKICQTBVGqXx13ISgGmABRlEMDohAQVBmN1T00AHAxqRSiJaHSzLfNIXKVISAgrAAfQ0OqavHGrgGQcrlMHZtNl4CrBwAeVikqspIqLqnuYgks4FdouQOp8F1Xht2EURxAPI95CUNQSN0JIDCMEBTAsKwLXsWg8BiQtIGWVA7AyQt8xWNYNhQubCrSjKhu4XgsJwzhPWY4gmDMX61RTDhtWzEi82wabYrkjizWsDsu14pS8rY4dg1HMSZznCBFzOCB4a4zseJ7TBSEOGT13k9SvtLHS9IMyh/RARFJGDCFEQ0WyNAhWzQ01SQHMuLgIUuUGPIzWNgy4TVNVshyISDcdrMRWyIfPUj7sLYtvS00hy0QFB2sE%2BLazRqravq%2Bqmpa0gsFtPANiYzBmN6ixsoWugCHiQsIBiEiYnCBoTGB0gg%2BYYgTF6mJtGqPWcrrNhBF6hhaFDzWsDpIxxEzqt47Ju79RG5BYs2T1wh9tz9VOmIAaj7wsDDrC8AzT7lioAwkpdt2PbDxbdxWw9FpPTbNe2/RDGMU0jo7fQzouozrupO6Cw6Au3AgDxkKSJ9inQpIPwKTIIJ/I/qTg%2BZyigjeGDAppT8vdfvnvy/X0qJDH4/3o34WZYnUevuQiYNiKazzOxNsiNuLdl7PxQchwMZYwnDjSSBMqZyTUlwDSetgoM30kMIyblJYgFDIiYMUZwyOQcrZRM8YRYa1zJNBwgUcG%2BmMqzdmnNua835oLYWotxZuQhKAxh91WHaVBoiERfkmHiOWPaYgaQ3ChiAA)

[![Compiler Explorer](https://asdalexander77.github.io/img/god_bolt_tsc_native.jpg)](https://godbolt.org/#z:OYLghAFBqd5QCxAYwPYBMCmBRdBLAF1QCcAaPECAMzwBtMA7AQwFtMQByARg9KtQYEAysib0QXABx8BBAKoBnTAAUAHpwAMvAFYTStJg1AEAngAdMC5MTxmCpJfWQE8Ayo3QBhVLQCuLBglpDwAZPAZMADl/ACNMYgkuUjNUBUJXBm8/AKDk1PSBMIjoljiEriTHTGcMoQImYgIs/0CpB0wnFwE6hoIiqNj4xId6xuactoVRvvCB0qGKgEoHVF9iZHYOKl8GGoEAKgBqYAhFw4BSAHYAIXONAEFDw5M8DvQj6h29hg%2Bzq9uHk8ni83ocuOcAMwAx5AkG0dCHABMkOhQOer3hhwhKLuMKuABFFqccQ8Cbjcdtdl0GIcWExwqcLjdcWj%2BMRDhA0AwpodVIdUFRjqc/szAWigWYbIIIKpFiSYU8yaTLviOMtaJwAKy8QIcLSkVCcAAq5kwIhsdkOkSYLgAbphDgpVusHedERCeKQCJo1csANYgCEaAB0ABYNBpJKGIZrLqHLpd3VH9JxQ7wWBII6RdfrDRxeAoQBovT7lnBYEg8GQKJzUCwzHR4ta2MbTebbAQrTa8PbDt56432SGQ5LMIjLocAFIASSNHIA4p5FrxMPgiDZ0Hp%2BIIRGJ2FxQzJBIoVOo9To9FVvu4GF4fC09KFZiUyiBpCk0tTxq08p%2BMv0XyGaQr2pHoxnvCZ2k6WppgAwYEmA6Zv0vWDn3gt9lgIYhMEwPBNw0dUtR1H0DU4EIQgANQAWUOacACVDgo14AHd4kOE0LHbS1rTtB1%2Bwbegh2DEdsPHKdZwXJcOVwQgSAud0kj7OsBLYt0PWXEtz0WZYEEwJgsASU5SADUMuGDcdJERTVEXjABOCEADYNFDaQNQ4NNSAzLgsxzXg8wLItNK0MsYCgUKICQTBVGqXx13ISgGmABRlEMDohAQVBmN1T00AHAxqRSiJaHSzLfNIXKVISAgrAAfQ0OqavHGrgGQcrlMHZtNl4CrBwAeVikqspIqLqnuYgks4FdouQOp8F1Xht2EURxAPI95CUNQSN0JIDCMEBTAsKwLXsWg8BiQtIGWVA7AyQt8xWNYNhQubCrSjKhu4XgsJwzhPWY4gmDMX61RTDhtWzEi82wabYrkjizWsDsu14pS8rY4dg1HMSZznCBFzOCB4a4zseJ7TBSEOGT13k9SvtLHS9IMyh/RARFJGDCFEQ0WyNAhWzQ01SQHMuLgIUuUGPIzWNgy4TVNVshyISDcdrMRWyIfPUj7sLYtvS00hy0QFB2sE%2BLazRqravq%2Bqmpa0gsFtPANiYzBmN6ixsoWugCHiQsIBiEiYnCBoTGB0gg%2BYYgTF6mJtGqPWcrrNhBF6hhaFDzWsDpIxxEzqt47Ju79RG5BYs2T1wh9tz9VOmIAaj7wsDDrC8AzT7lioAwkpdt2PbDxbdxWw9FpPTbNe2/RDGMU0jo7fQzouozrupO6Cw6Au3AgDxkKSJ9inQpIPwKTIIJ/I/qTg%2BZyigjeGDAppT8vdfvnvy/X0qJDH4/3o34WZYnUevuQiYNiKazzOxNsiNuLdl7PxQchwMZYwnDjSSBMqZyTUlwDSetgoM30kMIyblJYgFDIiYMUZwyOQcrZRM8YRYa1zJNBwgUcG%2BmMqzdmnNua835oLYWotxZuQhKAxh91WHaVBoiERfkmHiOWPaYgaQ3ChiAA)

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
tsc --emit=jit --opt --shared-libs=TypeScriptRuntime.dll example.ts
```

Result
```
Department name: Accounting and Auditing
The Accounting Department meets each Monday at 10am.
```

## Run as JIT

- with Garbage collection
```cmd
tsc --emit=jit --emit=jit --opt --shared-libs=TypeScriptRuntime.dll hello.ts
```

- without Garbage collection
```cmd
tsc --emit=jit --emit=jit --nogc hello.ts
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
- with Garbage collection

File ``tsc-compile-gc.bat``
```cmd
set FILENAME=%1
set LLVMEXEPATH=C:/dev/TypeScriptCompiler/3rdParty/llvm/release/bin
set LLVMLIBPATH=C:/dev/TypeScriptCompiler/3rdParty/llvm/release/lib
set TSCLIBPATH=C:/dev/TypeScriptCompiler/__build/tsc-release/lib
set TSCEXEPATH=C:/dev/TypeScriptCompiler/__build/tsc-release/bin
set GCLIBPATH=C:/dev/TypeScriptCompiler/3rdParty/gc/Release
set LIBPATH="C:/Program Files/Microsoft Visual Studio/2022/Professional/VC/Tools/MSVC/14.35.32215/lib/x64"
set SDKPATH="C:/Program Files (x86)/Windows Kits/10/Lib/10.0.22000.0/um/x64"
set UCRTPATH="C:/Program Files (x86)/Windows Kits/10/Lib/10.0.22000.0/ucrt/x64"
%TSCEXEPATH%\tsc.exe --opt --emit=llvm C:\temp\%FILENAME%.ts 2>%FILENAME%.ll
%LLVMEXEPATH%\llc.exe -O3 --filetype=obj -o=%FILENAME%.o %FILENAME%.ll
%LLVMEXEPATH%\lld.exe -flavor link %FILENAME%.o /libpath:%LIBPATH% /libpath:%SDKPATH% /libpath:%UCRTPATH% /libpath:%LLVMLIBPATH% /libpath:%GCLIBPATH% /libpath:%TSCLIBPATH% msvcrt.lib ucrt.lib kernel32.lib user32.lib gcmt-lib.lib TypeScriptAsyncRuntime.lib LLVMSupport.lib
```
Compile 
```cmd
tsc-compile-gc.bat hello
```

- without Garbage collection

File ``tsc-compile-nogc.bat``
```cmd
set FILENAME=%1
set LLVMEXEPATH=C:/dev/TypeScriptCompiler/3rdParty/llvm/release/bin
set LLVMLIBPATH=C:/dev/TypeScriptCompiler/3rdParty/llvm/release/lib
set TSCLIBPATH=C:/dev/TypeScriptCompiler/__build/tsc-release/lib
set TSCEXEPATH=C:/dev/TypeScriptCompiler/__build/tsc-release/bin
set LIBPATH="C:/Program Files/Microsoft Visual Studio/2022/Professional/VC/Tools/MSVC/14.35.32215/lib/x64"
set SDKPATH="C:/Program Files (x86)/Windows Kits/10/Lib/10.0.22000.0/um/x64"
set UCRTPATH="C:/Program Files (x86)/Windows Kits/10/Lib/10.0.22000.0/ucrt/x64"
%TSCEXEPATH%\tsc.exe --opt -nogc --emit=llvm C:\temp\%FILENAME%.ts 2>%FILENAME%.ll
%LLVMEXEPATH%\llc.exe -O3 --filetype=obj -o=%FILENAME%.o %FILENAME%.ll
%LLVMEXEPATH%\lld.exe -flavor link %FILENAME%.o /libpath:%LIBPATH% /libpath:%SDKPATH% /libpath:%UCRTPATH% /libpath:%LLVMLIBPATH% /libpath:%TSCLIBPATH% msvcrt.lib ucrt.lib kernel32.lib user32.lib TypeScriptAsyncRuntime.lib LLVMSupport.lib
```
Compile 
```cmd
tsc-compile-gc.bat hello
```


Run
```
hello.exe
```

Result
```
Hello World!
```

### On Linux (Ubuntu 20.04 and 22.04)
- with Garbage collection

File ``tsc-compile-gc.sh``
```cmd
FILENAME=$1
TSCEXEPATH=/home/dev/TypeScriptCompiler/__build/tsc-ninja-release/bin
TSCLIBPATH=/home/alex/TypeScriptCompiler/__build/tsc-ninja-release/lib
LLVMEXEPATH=/home/alex/TypeScriptCompiler/3rdParty/llvm-ninja/release/bin
LLVMLIBPATH=/home/alex/TypeScriptCompiler/3rdParty/llvm-ninja/release/lib
GCLIBPATH=/home/alex/TypeScriptCompiler/3rdParty/gc/release
$TSCEXEPATH/tsc --emit=llvm --opt $FILENAME.ts 2>$FILENAME.il
$LLVMEXEPATH/llc -relocation-model=pic --filetype=obj -o=$FILENAME.o $FILENAME.il
gcc -o $FILENAME -L$LLVMLIBPATH -L$GCLIBPATH -L$TSCLIBPATH $FILENAME.o -lgcmt-lib -lTypeScriptAsyncRuntime -lLLVMSupport -lLLVMDemangle -frtti -fexceptions -lstdc++ -lm -lpthread -ltinfo -ldl
```
Compile 
```cmd
sh -f tsc-compile-gc.sh hello
```

- without Garbage collection

File ``tsc-compile-nogc.sh``
```cmd
FILENAME=$1
TSCEXEPATH=/home/dev/TypeScriptCompiler/__build/tsc-ninja-release/bin
TSCLIBPATH=/home/alex/TypeScriptCompiler/__build/tsc-ninja-release/lib
LLVMEXEPATH=/home/alex/TypeScriptCompiler/3rdParty/llvm-ninja/release/bin
LLVMLIBPATH=/home/alex/TypeScriptCompiler/3rdParty/llvm-ninja/release/lib
$TSCEXEPATH/tsc --emit=llvm --opt -nogc $FILENAME.ts 2>$FILENAME.il
$LLVMEXEPATH/llc -relocation-model=pic --filetype=obj -o=$FILENAME.o $FILENAME.il
gcc -o $FILENAME -L$LLVMLIBPATH -L$GCLIBPATH -L$TSCLIBPATH $FILENAME.o -lTypeScriptAsyncRuntime -lLLVMSupport -lLLVMDemangle -frtti -fexceptions -lstdc++ -lm -lpthread -ltinfo -ldl
```
Compile 
```cmd
sh -f tsc-compile-nogc.sh hello
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
echo "check if your LLC support WebAssembly by running command: llc.exe --version --triple"
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

To build ``TSC`` binaries:

```
cd tsc
config_tsc_debug.bat
build_tsc_debug.bat
```

### On Linux (Ubuntu 20.04 and 22.04)

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
