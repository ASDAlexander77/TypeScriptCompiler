# TypeScript Native Compiler
###### Powered by [![LLVM|MLIR](https://llvm.org/img/LLVM-Logo-Derivative-1.png)](https://llvm.org/)

[![Donate](https://img.shields.io/badge/Donate-PayPal-green.svg)](https://www.paypal.com/donate/?hosted_button_id=BBJ4SQYLA6D2L)

# Build

[![Test Build (Windows)](https://github.com/ASDAlexander77/TypeScriptCompiler/actions/workflows/cmake-test-release-win.yml/badge.svg)](https://github.com/ASDAlexander77/TypeScriptCompiler/actions/workflows/cmake-test-release-win.yml)
[![Test Build (Linux)](https://github.com/ASDAlexander77/TypeScriptCompiler/actions/workflows/cmake-test-release-linux.yml/badge.svg)](https://github.com/ASDAlexander77/TypeScriptCompiler/actions/workflows/cmake-test-release-linux.yml)

# What's new

- ```union type``` in ```yield```

```TypeScript
function* g() {
  yield* (function* () {
    yield 1.0;
    yield 2.0;
    yield "3.0";
    yield 4.0;
  })();
}

function main() {
    for (const x of g())
        if (typeof x == "string")
            print("string: ", x);
        else if (typeof x == "number")
            print("number: ", x);
}
```

- Well-Known Symbols

* ```toPrimitive```

```TypeScript
    const object1 = {

        [Symbol.toPrimitive](hint: string) : string | number | boolean {
            if (hint === "number") {
                return 10;
            }
            if (hint === "string") {
                return "hello";
            }
            return true;
        }

    };

    print(+object1); // 10        hint is "number"
    print(`${object1}`); // "hello"   hint is "string"
    print(object1 + ""); // "true"    hint is "default"
```    

* ```interator```

```TypeScript
class StringIterator {
    next() {
        return {
            done: true,
            value: ""
        };
    }
    [Symbol.iterator]() {
        return this;
    }
}

function main() {
    for (const v of new StringIterator) { }
}
```

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
- [x] Migrating to LLVM 16.0.3
- [ ] JavaScript Built-in classes library

# Demo 
[(click here)](https://github.com/ASDAlexander77/TypeScriptCompiler/releases/)

[![Demo](https://asdalexander77.github.io/img/tsc_emit.gif)](https://github.com/ASDAlexander77/TypeScriptCompiler/releases/)


# Try it 
[(click here)](https://godbolt.org/#z:OYLghAFBqd5QCxAYwPYBMCmBRdBLAF1QCcAaPECAMzwBtMA7AQwFtMQByARg9KtQYEAysib0QXABx8BBAKoBnTAAUAHpwAMvAFYTStJg1AEAngAdMC5MTxmCpJfWQE8Ayo3QBhVLQCuLBhAAJlIPABk8BkwAOX8AI0xiCQBmUjNUBUJXBm8/AOC0jKyBCKjYlgSkrlTHTGdsoQImYgJc/0CQ2vqBRuaCUpj4xJSHJpa2/M6x/sjBiuHqgEoHVF9iZHYOKl8GboYAKgBqYAhFw4BSAHYAIXONAEFDw5M8TFp0I%2Bodvc%2Bzq9uHk8ni83uhDlxzskAY8gSD3ocgpDoUDnq94ckkXcYVcACKLU6Yh64rFY7a7FwCQ4sJiRU4XG5YlH8YiHCBoBgKAiHVSHVBUY6nP4MwEooFmGyCCCqRaEmFPYlEy44jjLWicACsvECHC0pFQnAAKuZMCIbHZDtEmC4AG6YQ4KVbrO3nILJHikAiaFXLADWIGSGgAdAAWDQaSTB5Lqy7By6XV0R/ScYO8FgSMOkbW6/UcXgKEAaD1e5ZwWBIPBkChs1AsMx0RKWtiG42m2xcy02u3eWv1llBoPizDJZKHABSAEkDayAOKeRa8TD4Ig2dB6fiCERidhcYMyQSKFTqHU6PRdCmBCAeCaBLihBjoAblSogaTpTLn696N/FBiPoZJaQzwaaZP1vICemmP95gA0Y%2BlA2CWig59JGWAhiEwTA8FXDRVQ1LUvT1TgwjCAA1ABZQ5xwAJUOEjXgAd0SQ4jQsVtzQ7PBbUObs63oPtAwHdDhzHScZznVlcEIEgLldW9uJrXimJdN15yLY9FmWBBMCYLAklOUg/WDLhAyCS5JCCdUgljABOZIADYNGDaQ1Q4FNSDTLgMyzXgczzAs1K0EsYCgYKICQTBVDqXxl3IShmmABRlEMN4hAQVB6O1d00B7AxzySqJaFS9LvNIbLFKSAgrAAfQ0GqquHKrgGQUqFN7RtNl4MrewAeWioqMoIiK6nuYgEs4BdIuQRp8G1Xh12EURxB3Pd5CUNQCN0W8DCMEBTAsKwzXsWg8DifNIGWVA7GyfNcz1W1iBXSx4GWB01g2U8CBm/KUrSgbuF4NCMM4d16OIJgzGBlUkw4TVMwInNsEm6LpJYk1rDbC0rU4rtWr4w5%2B0DQdhInKcIFnM4IFRtj2yx21SEOSTlxklSAeLTTtN0yhfWCSRA2SIINGsjRkms4N1UkOzLmqS5obctNLjsgSheVwXVes28St8hx/M9dTSFLRAUFxxJYurHLhkq5A6tqhqmtILBrTwDY6MwejuosTK5roAhEnzCA4gIuJImaExIdIIPmGIExuribQ6l1rKazYQRuoYWhQ%2BPe3MDiXxgE8MRaBu90sGpIxxEz/B0PqW0bt1IbkGizZ3UiH2XN1Y64jBqPvCwMO0LwNN/uWKgDASl23Y9sP5s3Jbd3mg91szzb9EMYxjQOtt9BOs79Mu88bp8%2B7Hp3l63nj7J3HveDwlmJ8FkKd9sng79zyQ%2B/wIYXpxh8dpTzPvYv4zDKP%2BD6cEf75DApBW%2BICuAvUdO9WB0NYaa04MxFs6N2K0xxubfigkhwjhJmJCmjNpLKVgazdS7MdLDH0i5OWwRkiBhVswtWy0UG3XzIWXWgVuZBF5vzVWIsxYSylskGWLlkj4Uzlrbh3poZBCkdmcaAU5H3UyG4YMQA%3D%3D%3D)

[![Compiler Explorer](https://asdalexander77.github.io/img/god_bolt_tsc_native.jpg)](https://godbolt.org/#z:OYLghAFBqd5QCxAYwPYBMCmBRdBLAF1QCcAaPECAMzwBtMA7AQwFtMQByARg9KtQYEAysib0QXABx8BBAKoBnTAAUAHpwAMvAFYTStJg1AEAngAdMC5MTxmCpJfWQE8Ayo3QBhVLQCuLBhAAJlIPABk8BkwAOX8AI0xiCQBmUjNUBUJXBm8/AOC0jKyBCKjYlgSkrlTHTGdsoQImYgJc/0CQ2vqBRuaCUpj4xJSHJpa2/M6x/sjBiuHqgEoHVF9iZHYOKl8GboYAKgBqYAhFw4BSAHYAIXONAEFDw5M8TFp0I%2Bodvc%2Bzq9uHk8ni83uhDlxzskAY8gSD3ocgpDoUDnq94ckkXcYVcACKLU6Yh64rFY7a7FwCQ4sJiRU4XG5YlH8YiHCBoBgKAiHVSHVBUY6nP4MwEooFmGyCCCqRaEmFPYlEy44jjLWicACsvECHC0pFQnAAKuZMCIbHZDtEmC4AG6YQ4KVbrO3nILJHikAiaFXLADWIGSGgAdAAWDQaSTB5Lqy7By6XV0R/ScYO8FgSMOkbW6/UcXgKEAaD1e5ZwWBIPBkChs1AsMx0RKWtiG42m2xcy02u3eWv1llBoPizDJZKHABSAEkDayAOKeRa8TD4Ig2dB6fiCERidhcYMyQSKFTqHU6PRdCmBCAeCaBLihBjoAblSogaTpTLn696N/FBiPoZJaQzwaaZP1vICemmP95gA0Y%2BlA2CWig59JGWAhiEwTA8FXDRVQ1LUvT1TgwjCAA1ABZQ5xwAJUOEjXgAd0SQ4jQsVtzQ7PBbUObs63oPtAwHdDhzHScZznVlcEIEgLldW9uJrXimJdN15yLY9FmWBBMCYLAklOUg/WDLhAyCS5JCCdUgljABOZIADYNGDaQ1Q4FNSDTLgMyzXgczzAs1K0EsYCgYKICQTBVDqXxl3IShmmABRlEMN4hAQVB6O1d00B7AxzySqJaFS9LvNIbLFKSAgrAAfQ0GqquHKrgGQUqFN7RtNl4MrewAeWioqMoIiK6nuYgEs4BdIuQRp8G1Xh12EURxB3Pd5CUNQCN0W8DCMEBTAsKwzXsWg8DifNIGWVA7GyfNcz1W1iBXSx4GWB01g2U8CBm/KUrSgbuF4NCMM4d16OIJgzGBlUkw4TVMwInNsEm6LpJYk1rDbC0rU4rtWr4w5%2B0DQdhInKcIFnM4IFRtj2yx21SEOSTlxklSAeLTTtN0yhfWCSRA2SIINGsjRkms4N1UkOzLmqS5obctNLjsgSheVwXVes28St8hx/M9dTSFLRAUFxxJYurHLhkq5A6tqhqmtILBrTwDY6MwejuosTK5roAhEnzCA4gIuJImaExIdIIPmGIExuribQ6l1rKazYQRuoYWhQ%2BPe3MDiXxgE8MRaBu90sGpIxxEz/B0PqW0bt1IbkGizZ3UiH2XN1Y64jBqPvCwMO0LwNN/uWKgDASl23Y9sP5s3Jbd3mg91szzb9EMYxjQOtt9BOs79Mu88bp8%2B7Hp3l63nj7J3HveDwlmJ8FkKd9sng79zyQ%2B/wIYXpxh8dpTzPvYv4zDKP%2BD6cEf75DApBW%2BICuAvUdO9WB0NYaa04MxFs6N2K0xxubfigkhwjhJmJCmjNpLKVgazdS7MdLDH0i5OWwRkiBhVswtWy0UG3XzIWXWgVuZBF5vzVWIsxYSylskGWLlkj4Uzlrbh3poZBCkdmcaAU5H3UyG4YMQA%3D%3D%3D)

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
sh -f prepare_3rdParty.sh
```

To build ``TSC`` binaries:

```
cd tsc
sh -f config_tsc_debug.sh
sh -f build_tsc_debug.sh
```
