# TypeScript Native Compiler
###### Powered by [![LLVM|MLIR](https://llvm.org/img/LLVM-Logo-Derivative-1.png)](https://llvm.org/)

[![Donate](https://img.shields.io/badge/Donate-PayPal-green.svg)](https://www.paypal.com/donate/?hosted_button_id=BBJ4SQYLA6D2L)

# Build

[![Test Build (Windows)](https://github.com/ASDAlexander77/TypeScriptCompiler/actions/workflows/cmake-test-release-win.yml/badge.svg)](https://github.com/ASDAlexander77/TypeScriptCompiler/actions/workflows/cmake-test-release-win.yml)
[![Test Build (Linux)](https://github.com/ASDAlexander77/TypeScriptCompiler/actions/workflows/cmake-test-release-linux.yml/badge.svg)](https://github.com/ASDAlexander77/TypeScriptCompiler/actions/workflows/cmake-test-release-linux.yml)

# What's new
- JavaScript Built-in objects library [[Default Library repo](https://github.com/ASDAlexander77/TypeScriptCompilerDefaultLib/)]

- Visual Studio Code project
```cmd
tsc --new Test1
```

- Strict null checks
```TypeScript
let sn: string | null = null; // Ok
let s: string = null; // error
```

- improved `Template Literal Types`
```TypeScript
type Color = "red" | "green" | "blue";
type HexColor<T extends Color> = `#${string}`;
```

- Public, private, and protected modifiers
```TypeScript
class Point {
    private x: number;
    #y: number;
}

const p = new Point();
p.x // access error
p.#y // error
```

- Class from Tuple
```TypeScript
class Point {
    x: number;
    y: number;
}

class Line {
    constructor(public start: Point, public end: Point) { }
}

const l = new Line({ x: 0, y: 1 }, { x: 1.0, y: 2.0 });
```

- Compile-time `if`s
```TypeScript
function isArray<T extends unknown[]>(value: T): value is T {
    return true;
}

function gen<T>(t: T)
{
    if (isArray(t))
    {
        return t.length.toString();
    }

    return "int";
}

const v1 = gen<i32>(23); // result: int
const v2 = gen<string[]>([]); // result: 0
```

- Migrated to LLVM 19.1.3

- improved ```generating debug information``` more info here: [Wiki:How-To](https://github.com/ASDAlexander77/TypeScriptCompiler/wiki/How-To#compile-and-debug-with-visual-studio-code)
```cmd
tsc --di --opt_level=0 --emit=exe example.ts
```
- [more...](https://github.com/ASDAlexander77/TypeScriptCompiler/wiki/What's-new)

# Planning
- [x] Migrating to LLVM 19.1.3
- [x] Shared libraries
- [ ] JavaScript Built-in classes library

# Demo 
[(click here)](https://github.com/ASDAlexander77/TypeScriptCompiler/releases/)

[![Demo](https://raw.githubusercontent.com/ASDAlexander77/ASDAlexander77.github.io/main/img/tsc_emit.gif)](https://github.com/ASDAlexander77/TypeScriptCompiler/releases/)


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

File ``tsc-compile.bat``
```cmd
set FILENAME=%1
set GC_LIB_PATH=C:\dev\TypeScriptCompiler\__build\gc\msbuild\x64\release\Release
set LLVM_LIB_PATH=C:\dev\TypeScriptCompiler\__build\llvm\msbuild\x64\release\Release\lib
set TSC_LIB_PATH=C:\dev\TypeScriptCompiler\__build\tsc\windows-msbuild-release\lib
set TSCEXEPATH=C:\dev\TypeScriptCompiler\__build\tsc\windows-msbuild-release\bin
%TSCEXEPATH%\tsc.exe --opt --emit=exe %FILENAME%.ts
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

### On Linux (Ubuntu 20.04 and 22.04)

File ``tsc-compile.sh``
```bash
FILENAME=$1
export TSC_LIB_PATH=~/dev/TypeScriptCompiler/__build/tsc/linux-ninja-gcc-release/lib
export LLVM_LIB_PATH=~/dev/TypeScriptCompiler/3rdParty/llvm/release/lib
export GC_LIB_PATH=~/dev/TypeScriptCompiler/3rdParty/gc/release
TSCEXEPATH=~/dev/TypeScriptCompiler/__build/tsc/linux-ninja-gcc-release/bin
$TSCEXEPATH/tsc --emit=exe $FILENAME.ts --relocation-model=pic
```
Compile 
```bash
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
set FILENAME=%1
set GC_LIB_PATH=C:\dev\TypeScriptCompiler\__build\gc\msbuild\x64\release\Release
set LLVM_LIB_PATH=C:\dev\TypeScriptCompiler\__build\llvm\msbuild\x64\release\Release\lib
set TSC_LIB_PATH=C:\dev\TypeScriptCompiler\__build\tsc\windows-msbuild-release\lib
C:\dev\TypeScriptCompiler\__build\tsc\windows-msbuild-release\bin\tsc.exe --emit=exe --nogc -mtriple=wasm32-unknown-unknown %FILENAME%.ts
```
Compile 
```cmd
tsc-compile-wasm.bat hello
```

Run ``run.html``
```html
<!DOCTYPE html>
<html>

<head></head>

<body>
    <script type="module">
        let buffer;
        let buffer32;
        let buffer64;
        let bufferF64;
        let heap;

        let heap_base, heap_end, stack_low, stack_high;

        const allocated = [];

        const allocatedSize = (addr) => {
            return allocated["" + addr];
        };

        const setAllocatedSize = (addr, newSize) => {
            allocated["" + addr] = newSize;
        };

        const expand = (addr, newSize) => {

            const aligned_newSize = newSize + (4 - (newSize % 4))

            const end = addr + allocatedSize(addr);
            const newEnd = addr + aligned_newSize;

            for (const allocatedAddr in allocated) {
                const beginAllocatedAddr = parseInt(allocatedAddr);
                const endAllocatedAddr = beginAllocatedAddr + allocated[allocatedAddr];
                if (beginAllocatedAddr != addr && addr < endAllocatedAddr && newEnd > beginAllocatedAddr) {
                    return false;
                }
            }

            setAllocatedSize(addr, aligned_newSize);
            if (addr + aligned_newSize > heap) heap = addr + aligned_newSize;
            return true;
        };

        const endOf = (addr) => { while (buffer[addr] != 0) { addr++; if (addr > heap_end) throw "out of memory boundary"; }; return addr; };
        const strOf = (addr) => String.fromCharCode(...buffer.slice(addr, endOf(addr)));
        const copyStr = (dst, src) => { while (buffer[src] != 0) buffer[dst++] = buffer[src++]; buffer[dst] = 0; return dst; };
        const ncopy = (dst, src, count) => { while (count-- > 0) buffer[dst++] = buffer[src++]; return dst; };
        const append = (dst, src) => copyStr(endOf(dst), src);
        const cmp = (addrL, addrR) => { while (buffer[addrL] != 0) { if (buffer[addrL] != buffer[addrR]) break; addrL++; addrR++; } return buffer[addrL] - buffer[addrR]; };
        const prn = (str, addr) => { for (let i = 0; i < str.length; i++) buffer[addr++] = str.charCodeAt(i); buffer[addr] = 0; return addr; };
        const clear = (addr, size, val) => { for (let i = 0; i < size; i++) buffer[addr++] = val; };
        const aligned_alloc = (size) => { 
            const aligned_size = size + (4 - (size % 4)); 
            if ((heap + aligned_size) > heap_end) throw "out of memory"; 
            setAllocatedSize(heap, aligned_size); 
            const heapCurrent = heap; 
            heap += aligned_size; 
            return heapCurrent; 
        };
        const free = (addr) => delete allocated["" + addr];
        const realloc = (addr, size) => {
            if (!expand(addr, size)) {
                const newAddr = aligned_alloc(size);
                ncopy(newAddr, addr, allocatedSize(addr));
                free(addr);
                return newAddr;
            }

            return addr;
        }

        const envObj = {
            memory: new WebAssembly.Memory({ initial: 256 }),
            table: new WebAssembly.Table({
                initial: 0,
                element: 'anyfunc',
            }),
            fmod: (arg1, arg2) => arg1 % arg2,
            sqrt: (arg1) => Math.sqrt(arg1),
            floor: (arg1) => Math.floor(arg1),
            pow: (arg1, arg2) => Math.pow(arg1, arg2),
            fabs: (arg1) => Math.abs(arg1),
            _assert: (msg, file, line) => console.assert(false, strOf(msg), "| file:", strOf(file), "| line:", line, " DBG:", path),
            puts: (arg) => output += strOf(arg) + '\n',
            strcpy: copyStr,
            strcat: append,
            strcmp: cmp,
            strlen: (addr) => endOf(addr) - addr,
            malloc: aligned_alloc,
            realloc: realloc,
            free: free,
            memset: (addr, size, val) => clear(addr, size, val),
            atoi: (addr, rdx) => parseInt(strOf(addr), rdx),
            atof: (addr) => parseFloat(strOf(addr)),
            sprintf_s: (addr, sizeOfBuffer, format, ...args) => {
                const formatStr = strOf(format);
                switch (formatStr) {
                    case "%d": prn(buffer32[args[0] >> 2].toString(), addr); break;
                    case "%g": prn(bufferF64[args[0] >> 3].toString(), addr); break;
                    case "%llu": prn(buffer64[args[0] >> 3].toString(), addr); break;
                    default: throw "not implemented"; 
                }

                return 0;
            },
        }

        const config = {
            env: envObj,
        };

        WebAssembly.instantiateStreaming(fetch("./hello.wasm"), config)
            .then(results => {
                const { main, __wasm_call_ctors, __heap_base, __heap_end, __stack_low, __stack_high } = results.instance.exports;
                buffer = new Uint8Array(results.instance.exports.memory.buffer);
                buffer32 = new Uint32Array(results.instance.exports.memory.buffer);
                buffer64 = new BigUint64Array(results.instance.exports.memory.buffer);
                bufferF64 = new Float64Array(results.instance.exports.memory.buffer);
                heap = heap_base = __heap_base, heap_end = __heap_end, stack_low = __stack_low, stack_high = __stack_high;
                try
                {
                    if (__wasm_call_ctors) __wasm_call_ctors();
                    main();
                }
                catch (e)
                {
                    console.error(e);
                }
            });
    </script>
</body>

</html>
```

## Build

### On Windows

#### Requirements:

- ``Visual Studio 2022``
- 512GB of free space on disk


First, precompile dependencies

```cmd
cd TypeScriptCompiler
prepare_3rdParty.bat 
```

To build ``TSC`` binaries:

```cmd
cd TypeScriptCompiler\tsc
config_tsc_release.bat
build_tsc_release.bat
```

### On Linux (Ubuntu 20.04 and 22.04)
#### Requirements:

- `gcc` or `clang`
- `cmake`
- `ninja-build`
- 512GB of free space on disk
- sudo apt-get install ``libtinfo-dev``

First, precompile dependencies

```bash
chmod +x *.sh
cd ~/TypeScriptCompiler
./prepare_3rdParty.sh
```

To build ``TSC`` binaries:

```bash
cd ~/TypeScriptCompiler/tsc
chmod +x *.sh
./config_tsc_release.sh
./build_tsc_release.sh
```
