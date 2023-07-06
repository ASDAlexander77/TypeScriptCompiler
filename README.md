# TypeScript Native Compiler
###### Powered by [![LLVM|MLIR](https://llvm.org/img/LLVM-Logo-Derivative-1.png)](https://llvm.org/)

[![Donate](https://img.shields.io/badge/Donate-PayPal-green.svg)](https://www.paypal.com/donate/?hosted_button_id=BBJ4SQYLA6D2L)

# Build

[![Test Build (Windows)](https://github.com/ASDAlexander77/TypeScriptCompiler/actions/workflows/cmake-test-release-win.yml/badge.svg)](https://github.com/ASDAlexander77/TypeScriptCompiler/actions/workflows/cmake-test-release-win.yml)
[![Test Build (Linux)](https://github.com/ASDAlexander77/TypeScriptCompiler/actions/workflows/cmake-test-release-linux.yml/badge.svg)](https://github.com/ASDAlexander77/TypeScriptCompiler/actions/workflows/cmake-test-release-linux.yml)

# What's new 
- Shared libraries
``shared.ts`` - shared library file: 
```TypeScript
export const val_num = 2.5;
export const val_str = "Hello World! - val";

export function test1()
{
	print("Hello World! test 1");
}

export function test2()
{
	print("Hello World! test 2");
}
```

- Load shared library
```TypeScript
import './shared'

function main()
{
	test1();
	test2();

	print(val_str);

	print(val_num);

	print("done.");
}

```

- Debug information: option `--di` in `tsc`
```cmd
tsc --opt_level=0 --di --emit=obj <file>.ts
```
- [more...](https://github.com/ASDAlexander77/TypeScriptCompiler/wiki/What's-new)

# Planning
- [x] Migrating to LLVM 16.0.3
- [ ] Shared libraries (work in progress)
- [ ] JavaScript Built-in classes library

# Demo 
[(click here)](https://github.com/ASDAlexander77/TypeScriptCompiler/releases/)

[![Demo](https://asdalexander77.github.io/img/tsc_emit.gif)](https://github.com/ASDAlexander77/TypeScriptCompiler/releases/)


# Try it 
[(click here)](https://godbolt.org/#z:OYLghAFBqd5TKALEBjA9gEwKYFFMCWALugE4A0BIEAZgQDbYB2AhgLbYgDkAjF%2BTXRMiAZVQtGIHgA4BQogFUAztgAKAD24AGfgCsp5eiyagiATwAO2JalIELRcisaoiBIdWaYAwunoBXNiYQACZyLwAZAiZsADlAgCNsUhAAZh5yC3QlYncmXwCg0Mzs3KEomPi2JJT0p2wXNyERIhZSIgLA4LDnbFc8lraiCrjE5LSMpVb2zqKe6eHo0erx9IBKJ3R/UlROLhYEqdIWVwBqVCMlJVOAEWwLIY5hU4BSAHYAIRetAEFT//OQiO/lcZAgFn8CXoBFQp1YHBApyO0WAa1en3eN2%2BP2xANOFjswli7GwEDWiIAbugCJh0V9fni8QTokQIC8QiE7g92k8iHCSYj2SFXiEPqciEgCEoAHTw7BrF6pel/AGY7G4gEHI4nPnM4QAWWw2DcJjJlOpmEVYoA9NbTmx/FNTklTgQ2BZGLzsLToqccHYKd7zpcVEp1W8sb9sRcWFdTj9UBh/MIUVzHsw%2Bdh1EQvNc0zyM3SNf8MExgaDSGSiwzGf8lP4rJWhQmkymTKdjLSfv5CCbgEKFUrTrbAWWiKQQSRSNdff6CIHaTGrtZ7Y6%2BeJ6PQkQ3kmTi%2BjIzia/89URDcaUWbTlSadWVbXT2yOQAVJDYeOJrZt4C3e7p54cMa1zYCcSCnPqQiYCwZgdnyPBaOw0oDla%2B5qlGx6nMAzDJCwOYAEr3GQRBKFeN60u8yq1iehKskKADi2HHH2HafsmzGkIR7QytKSEcoOlGqhG4aHtiNDJv0Qj2iw0RVhR%2B6MHyODckQvKIvmKkZlaw52ugADW4roOcHG4e%2BLCnBxNDJMwuwGR2TAdoc446uKljYPuSn/nyio3HC2AAO4fq2fbqbye5DiOem2bYIE5nZtKxjkwD2WZTBCAAtFqTlnPWCRLmGGEeQWwjSqexIcGFAn/IVGnFae559hV%2B62tVvLSlhMRMdgBFZFxFXaacySkGQiItYWUpwugfLoDQLlWIFX7BX%2BRWOOcxipXyJy7HG7U4fhnHEcJXAbPQ3AAKz8MEXA6OQ6DcM%2BrliHYDinMSbiBkiWw7O%2B7LpPwRDaEdGy6WkqTSqk4MQ5DkMAGyGNwAAs/BsCAbzw2DqTSJjqTw/DbzQ1oWjw%2BQl3XbdXD8EoIBaOQ/1XUd5BwLAKBZn0/hTpQ1BtMASiqMYDQiEg6B%2BZdfDkBg7oMLheS8zE9AC0LJP8OLFgMOMxGoAA%2BloWsa6kp0a8AqBi%2BgEuMKQZV7ErJsq2bADybPy8LAPhOofQ/KQ3PcPwLOoC0hCXfwgjCGIEicDwRNB4oKgaM7%2BgZEYJggOYVg2E9jjQoc8AbOgDh5JT5M3YGpB2DglOQBsSifbsBhTP7Mv84LTu8PwfnHBY3B8MdZ0Xc7ZO4K7qBs2Qpz3VYj32Hyr3zu%2Bvim8kpxaNKi8EtgeunAAUgAks%2BpwQHR3hohAo/YOPz1T4G5CnPgxDDz9PBrH9ANrBsb4sP61Bd1wiPkMjISndK0NoY8GAdDU6p0QipAAJyQKAcTXuXsnBUxpk/IGaR/4Yz1uAngeNIHYLeG8OGXBUg9zpjdBBtMdAbEZsgNA1tVYUCoBAZW9Ck42B1trPWBsjY4ApDCbAAA1Ag/lbZWBFoHBgOZpzUASM7BI0Q2hmA7vwORrBSBmFtgkXQfRaai3FryW2TB6CKNITgBI/hgDeAkPQfOoscBsGMMASQJiCAcX6IGfO10fZsz2KLFkDRnYZ2OGo3wOBnbjjdEojYNAjDc0EcI0RSi5DBw3GHCO8hlBqE0KQuOhgHFJ1cqnCehgCCZ3LjdXOQIEHoCLiXawWd6iNDyJ4JgPg/BdAMJEJYVQagGCyDkJo%2BQ2lFAyH0soTARjdPGJMBo2iBgLFmMEaZjTmgLAmWMFIkx5lDMWU4VZXT1lSArlXMOn9zpwNIWTEeD1bATxelLd6s8bbz0XsvDia8t47z3gfXe18pwinWI/Omz9yCv3fmScgwN0jShCNIdI4d4bSEgajRFWhZAnS/iQ0mCDKbUwofTahEAUAuI5kwuhZsLZ3WuWnO5b0Z5kueUvEqbzTob23rvfeD9wi9jIDSAwkcQ6SHDkkqOmTY41xmRJYIEAvALI6S0tZKwNklH6XkWVIzSgDIVT0pZsyVlDDVQ03VTBBjtC1VM3Z%2Brtk1z2ZUA598aYcWwLyrQpzMX8DJhECI/D9SnE3nhU4cS/Lz2PqfSe9y6Vz1IAvRlK93lsq%2BYfX5t8IEZFOI8%2Bh/z76Asoag8G6MoYFtSIQ7%2BisyEFxxcgoFn8QhurLdmwG5Ai45A8PDIAA%3D%3D)

[![Compiler Explorer](https://asdalexander77.github.io/img/god_bolt_tsc_native.jpg)](https://godbolt.org/#z:OYLghAFBqd5TKALEBjA9gEwKYFFMCWALugE4A0BIEAZgQDbYB2AhgLbYgDkAjF%2BTXRMiAZVQtGIHgA4BQogFUAztgAKAD24AGfgCsp5eiyagiATwAO2JalIELRcisaoiBIdWaYAwunoBXNiYQACZyLwAZAiZsADlAgCNsUhAAZh5yC3QlYncmXwCg0Mzs3KEomPi2JJT0p2wXNyERIhZSIgLA4LDnbFc8lraiCrjE5LSMpVb2zqKe6eHo0erx9IBKJ3R/UlROLhYEqdIWVwBqVCMlJVOAEWwLIY5hU4BSAHYAIRetAEFT//OQiO/lcZAgFn8CXoBFQp1YHBApyO0WAa1en3eN2%2BP2xANOFjswli7GwEDWiIAbugCJh0V9fni8QTokQIC8QiE7g92k8iHCSYj2SFXiEPqciEgCEoAHTw7BrF6pel/AGY7G4gEHI4nPnM4QAWWw2DcJjJlOpmEVYoA9NbTmx/FNTklTgQ2BZGLzsLToqccHYKd7zpcVEp1W8sb9sRcWFdTj9UBh/MIUVzHsw%2Bdh1EQvNc0zyM3SNf8MExgaDSGSiwzGf8lP4rJWhQmkymTKdjLSfv5CCbgEKFUrTrbAWWiKQQSRSNdff6CIHaTGrtZ7Y6%2BeJ6PQkQ3kmTi%2BjIzia/89URDcaUWbTlSadWVbXT2yOQAVJDYeOJrZt4C3e7p54cMa1zYCcSCnPqQiYCwZgdnyPBaOw0oDla%2B5qlGx6nMAzDJCwOYAEr3GQRBKFeN60u8yq1iehKskKADi2HHH2HafsmzGkIR7QytKSEcoOlGqhG4aHtiNDJv0Qj2iw0RVhR%2B6MHyODckQvKIvmKkZlaw52ugADW4roOcHG4e%2BLCnBxNDJMwuwGR2TAdoc446uKljYPuSn/nyio3HC2AAO4fq2fbqbye5DiOem2bYIE5nZtKxjkwD2WZTBCAAtFqTlnPWCRLmGGEeQWwjSqexIcGFAn/IVGnFae559hV%2B62tVvLSlhMRMdgBFZFxFXaacySkGQiItYWUpwugfLoDQLlWIFX7BX%2BRWOOcxipXyJy7HG7U4fhnHEcJXAbPQ3AAKz8MEXA6OQ6DcM%2BrliHYDinMSbiBkiWw7O%2B7LpPwRDaEdGy6WkqTSqk4MQ5DkMAGyGNwAAs/BsCAbzw2DqTSJjqTw/DbzQ1oWjw%2BQl3XbdXD8EoIBaOQ/1XUd5BwLAKBZn0/hTpQ1BtMASiqMYDQiEg6B%2BZdfDkBg7oMLheS8zE9AC0LJP8OLFgMOMxGoAA%2BloWsa6kp0a8AqBi%2BgEuMKQZV7ErJsq2bADybPy8LAPhOofQ/KQ3PcPwLOoC0hCXfwgjCGIEicDwRNB4oKgaM7%2BgZEYJggOYVg2E9jjQoc8AbOgDh5JT5M3YGpB2DglOQBsSifbsBhTP7Mv84LTu8PwfnHBY3B8MdZ0Xc7ZO4K7qBs2Qpz3VYj32Hyr3zu%2Bvim8kpxaNKi8EtgeunAAUgAks%2BpwQHR3hohAo/YOPz1T4G5CnPgxDDz9PBrH9ANrBsb4sP61Bd1wiPkMjISndK0NoY8GAdDU6p0QipAAJyQKAcTXuXsnBUxpk/IGaR/4Yz1uAngeNIHYLeG8OGXBUg9zpjdBBtMdAbEZsgNA1tVYUCoBAZW9Ck42B1trPWBsjY4ApDCbAAA1Ag/lbZWBFoHBgOZpzUASM7BI0Q2hmA7vwORrBSBmFtgkXQfRaai3FryW2TB6CKNITgBI/hgDeAkPQfOoscBsGMMASQJiCAcX6IGfO10fZsz2KLFkDRnYZ2OGo3wOBnbjjdEojYNAjDc0EcI0RSi5DBw3GHCO8hlBqE0KQuOhgHFJ1cqnCehgCCZ3LjdXOQIEHoCLiXawWd6iNDyJ4JgPg/BdAMJEJYVQagGCyDkJo%2BQ2lFAyH0soTARjdPGJMBo2iBgLFmMEaZjTmgLAmWMFIkx5lDMWU4VZXT1lSArlXMOn9zpwNIWTEeD1bATxelLd6s8bbz0XsvDia8t47z3gfXe18pwinWI/Omz9yCv3fmScgwN0jShCNIdI4d4bSEgajRFWhZAnS/iQ0mCDKbUwofTahEAUAuI5kwuhZsLZ3WuWnO5b0Z5kueUvEqbzTob23rvfeD9wi9jIDSAwkcQ6SHDkkqOmTY41xmRJYIEAvALI6S0tZKwNklH6XkWVIzSgDIVT0pZsyVlDDVQ03VTBBjtC1VM3Z%2Brtk1z2ZUA598aYcWwLyrQpzMX8DJhECI/D9SnE3nhU4cS/Lz2PqfSe9y6Vz1IAvRlK93lsq%2BYfX5t8IEZFOI8%2Bh/z76Asoag8G6MoYFtSIQ7%2BisyEFxxcgoFn8QhurLdmwG5Ai45A8PDIAA%3D%3D)

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
tsc --emit=jit --opt --shared-libs=TypeScriptRuntime.dll hello.ts
```

- without Garbage collection
```cmd
tsc --emit=jit --nogc hello.ts
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
%TSCEXEPATH%\tsc.exe --opt --emit=obj C:\temp\%FILENAME%.ts -o=%FILENAME%.o
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
%TSCEXEPATH%\tsc.exe --opt -nogc --emit=obj C:\temp\%FILENAME%.ts -o=%FILENAME%.o
%LLVMEXEPATH%\lld.exe -flavor link %FILENAME%.o /libpath:%LIBPATH% /libpath:%SDKPATH% /libpath:%UCRTPATH% /libpath:%LLVMLIBPATH% /libpath:%TSCLIBPATH% msvcrt.lib ucrt.lib kernel32.lib user32.lib TypeScriptAsyncRuntime.lib LLVMSupport.lib
```
Compile 
```cmd
tsc-compile-nogc.bat hello
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
TSCLIBPATH=/home/dev/TypeScriptCompiler/__build/tsc-ninja-release/lib
LLVMLIBPATH=/home/dev/TypeScriptCompiler/3rdParty/llvm-ninja/release/lib
GCLIBPATH=/home/dev/TypeScriptCompiler/3rdParty/gc/release
$TSCEXEPATH/tsc --emit=obj --opt -relocation-model=pic $FILENAME.ts -o=$FILENAME.o
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
TSCLIBPATH=/home/dev/TypeScriptCompiler/__build/tsc-ninja-release/lib
LLVMEXEPATH=/home/dev/TypeScriptCompiler/3rdParty/llvm-ninja/release/bin
LLVMLIBPATH=/home/dev/TypeScriptCompiler/3rdParty/llvm-ninja/release/lib
$TSCEXEPATH/tsc --emit=obj --opt -nogc -relocation-model=pic $FILENAME.ts -o=$FILENAME.o
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
%TSCPATH%\tsc.exe -nogc --emit=obj --opt --mtriple=wasm32-unknown-unknown %FILENAME%.ts -o=%FILENAME%.ll
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
chmod +x *.sh
./prepare_3rdParty.sh
```

To build ``TSC`` binaries:

```
cd tsc
chmod +x *.sh
./config_tsc_debug.sh
./build_tsc_debug.sh
```
