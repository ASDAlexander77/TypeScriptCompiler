# TypeScript Native Compiler

Powered by [![LLVM|MLIR](https://llvm.org/img/LLVM-Logo-Derivative-1.png)](https://llvm.org/)

Developed by [![Claude Code](https://img.shields.io/badge/Claude-Code-D97757?logo=claude&logoColor=white)](https://claude.com/claude-code)

[![Donate](https://img.shields.io/badge/Donate-PayPal-green.svg)](https://www.paypal.com/donate/?hosted_button_id=BBJ4SQYLA6D2L)

A native ahead-of-time (AOT) and JIT compiler for **TypeScript**, built on **LLVM/MLIR**.
It compiles `.ts` files directly to native executables, WebAssembly, or runs them
on the fly via a built-in JIT — no Node.js or JavaScript runtime required.

## CI Status

[![Test Build (Windows)](https://github.com/ASDAlexander77/TypeScriptCompiler/actions/workflows/cmake-test-release-win.yml/badge.svg)](https://github.com/ASDAlexander77/TypeScriptCompiler/actions/workflows/cmake-test-release-win.yml)
[![Test Build (Linux)](https://github.com/ASDAlexander77/TypeScriptCompiler/actions/workflows/cmake-test-release-linux.yml/badge.svg)](https://github.com/ASDAlexander77/TypeScriptCompiler/actions/workflows/cmake-test-release-linux.yml)

## Showcase: native_ts_graphics

**[native_ts_graphics](https://github.com/ASDAlexander77/native_ts_graphics)** is a set of real-time 3D graphics samples written in TypeScript and compiled to native code with tslang.

- **25 samples**, TypeScript ports of NVIDIA's Donut samples on the [Donut](https://github.com/NVIDIA-RTX/Donut) rendering framework, on **Direct3D 12** and **Vulkan**: from `basic_triangle` and `deferred_shading` to ray tracing (`rt_reflections`, `rt_shadows`, `rt_particles`), meshlets, compute n-body, order-independent transparency, terrain tessellation, variable-rate shading and work graphs
- **Mixed C++/TypeScript** through CMake: `.ts` files are a first-class CMake language (`TSLANG`), compiled with `tslang --emit=obj` and linked with the C++ code, with incremental rebuilds
- Runs on **Windows** and **Android** (Vulkan, `arm64-v8a` and `x86_64`): the Vulkan samples are packaged as APKs, with a script that builds the samples, starts an emulator and tests them
- Choose the memory model at configure time: `-DTSLANG_MEMORY_MODEL=gc|rc|none`

It is a good place to see how TypeScript code drives a native C++ framework: the samples call Donut through a C interop layer (`donut_interop.d.ts`) wrapped in TypeScript classes.

## Contents

- [Showcase: native_ts_graphics](#showcase-native_ts_graphics)
- [What's new](#whats-new)
- [Roadmap](#roadmap)
- [Demo](#demo)
- [Try it online](#try-it)
- [Example](#example)
- [Running your code](#run-as-jit)
  - [As JIT](#run-as-jit)
  - [Debugging JIT code with GDB (Linux)](#debugging-jit-code-with-gdb-linux)
  - [As a native executable](#compile-as-binary-executable)
  - [For Android](#compile-for-android)
  - [As WebAssembly](#compiling-as-wasm)
- [C bindings with tsbindgen](#c-bindings-with-tsbindgen)
- [Memory models](#memory-models)
- [Building from source](#build)
  - [For Android](#build-for-android)
- [Community](#chat-room)
- [License](#license)

## What's new

- New memory model **`-mm=own`**: single ownership inferred at compile time, like Rust's borrow checker. There is no collector and no reference counting, and a program whose ownership cannot be proven is a compile error. Shared ownership is opt-in with `Shared<T>`. See [Memory models](#memory-models)
- **Android** targets (`arm64-v8a`, `x86_64`, API 29+): executables and shared libraries linked through the Android NDK. The release zip carries the collector, async runtime and default library for both ABIs. See [Compile for Android](#compile-for-android)
- Implemented `try/catch` exception handling in the **JIT** on Windows (x64 SEH unwinding via a custom LLJIT runtime)
- Migrared to **LLVM 22.1.8**
- Migrated to **Visual Studio 2026** (Windows build chain)
- JavaScript Built-in objects library [[Default Library repo](https://github.com/ASDAlexander77/TypeScriptCompilerDefaultLib/)]

- Visual Studio Code project

```bat
tslang --new Test1
```

- CMake project (mixed C++/TypeScript, `TSLANG` registered as a first-class CMake language)

```bat
tslang --cmake Test1
```

Generates a ready-to-build project (`CMakeLists.txt`, `CMakePresets.json`, `main.cpp`, sample `.ts` sources and the `cmake/` TSLANG language modules). Build it with:

```bat
cd Test1
cmake --preset default && cmake --build --preset default
```

- Strict null checks

```typescript
let sn: string | null = null; // Ok
let s: string = null; // error
```

- Improved `Template Literal Types`

```typescript
type Color = "red" | "green" | "blue";
type HexColor<T extends Color> = `#${string}`;
```

- Public, private, and protected modifiers

```typescript
class Point {
    private x: number;
    #y: number;
}

const p = new Point();
p.x // access error
p.#y // error
```

- Class from Tuple

```typescript
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

```typescript
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

- Improved `generating debug information` — more info here: [Wiki:How-To](https://github.com/ASDAlexander77/TypeScriptCompiler/wiki/How-To#compile-and-debug-with-visual-studio-code)

```bat
tslang --di --opt_level=0 --emit=exe example.ts
```

- [More What's New entries on the wiki](https://github.com/ASDAlexander77/TypeScriptCompiler/wiki/What's-new)

## Roadmap

- [x] Migrating to LLVM 22.1.8
- [x] Shared libraries
- [x] JavaScript Built-in classes library

## Demo

[Releases](https://github.com/ASDAlexander77/TypeScriptCompiler/releases/)

See also [native_ts_graphics](#showcase-native_ts_graphics): D3D12/Vulkan graphics samples, including ray tracing, written in TypeScript and compiled with tslang.

## Try it

[Open the native example on Compiler Explorer](https://godbolt.org/#z:OYLghAFBqd5TKALEBjA9gEwKYFFMCWALugE4A0BIEAZgQDbYB2AhgLbYgDkAjF%2BTXRMiAZVQtGIHgA4BQogFUAztgAKAD24AGfgCsp5eiyagiATwAO2JalIELRcisaoiBIdWaYAwunoBXNiYQACZyLwAZAiZsADlAgCNsUhAAZh5yC3QlYncmXwCg0Mzs3KEomPi2JJT0p2wXNyERIhZSIgLA4LDnbFc8lraiCrjE5LSMpVb2zqKe6eHo0erx9IBKJ3R/UlROLhYEqdIWVwBqVCMlJVOAEWwLIY5hU4BSAHYAIRetAEFT//OQiO/lcZAgFn8CXoBFQp1YHBApyO0WAa1en3eN2%2BP2xANOFjswli7GwEDWiIAbugCJh0V9fni8QTokQIC8QiE7g92k8iHCSYj2SFXiEPqciEgCEoAHTw7BrF6pel/AGY7G4gEHI4nPnM4QAWWw2DcJjJlOpmEVYoA9NbTmx/FNTklTgQ2BZGLzsLToqccHYKd7zpcVEp1W8sb9sRcWFdTj9UBh/MIUVzHsw%2Bdh1EQvNc0zyM3SNf8MExgaDSGSiwzGf8lP4rJWhQmkymTKdjLSfv5CCbgEKFUrTrbAWWiKQQSRSNdff6CIHaTGrtZ7Y6%2BeJ6PQkQ3kmTi%2BjIzia/89URDcaUWbTlSadWVbXT2yOQAVJDYeOJrZt4C3e7p54cMa1zYCcSCnPqQiYCwZgdnyPBaOw0oDla%2B5qlGx6nMAzDJCwOYAEr3GQRBKFeN60u8yq1iehKskKADi2HHH2HafsmzGkIR7QytKSEcoOlGqhG4aHtiNDJv0Qj2iw0RVhR%2B6MHyODckQvKIvmKkZlaw52ugADW4roOcHG4e%2BLCnBxNDJMwuwGR2TAdoc446uKljYPuSn/nyio3HC2AAO4fq2fbqbye5DiOem2bYIE5nZtKxjkwD2WZTBCAAtFqTlnPWCRLmGGEeQWwjSqexIcGFAn/IVGnFae559hV%2B62tVvLSlhMRMdgBFZFxFXaacySkGQiItYWUpwugfLoDQLlWIFX7BX%2BRWOOcxipXyJy7HG7U4fhnHEcJXAbPQ3AAKz8MEXA6OQ6DcM%2BrliHYDinMSbiBkiWw7O%2B7LpPwRDaEdGy6WkqTSqk4MQ5DkMAGyGNwAAs/BsCAbzw2DqTSJjqTw/DbzQ1oWjw%2BQl3XbdXD8EoIBaOQ/1XUd5BwLAKBZn0/hTpQ1BtMASiqMYDQiEg6B%2BZdfDkBg7oMLheS8zE9AC0LJP8OLFgMOMxGoAA%2BloWsa6kp0a8AqBi%2BgEuMKQZV7ErJsq2bADybPy8LAPhOofQ/KQ3PcPwLOoC0hCXfwgjCGIEicDwRNB4oKgaM7%2BgZEYJggOYVg2E9jjQoc8AbOgDh5JT5M3YGpB2DglOQBsSifbsBhTP7Mv84LTu8PwfnHBY3B8MdZ0Xc7ZO4K7qBs2Qpz3VYj32Hyr3zu%2Bvim8kpxaNKi8EtgeunAAUgAks%2BpwQHR3hohAo/YOPz1T4G5CnPgxDDz9PBrH9ANrBsb4sP61Bd1wiPkMjISndK0NoY8GAdDU6p0QipAAJyQKAcTXuXsnBUxpk/IGaR/4Yz1uAngeNIHYLeG8OGXBUg9zpjdBBtMdAbEZsgNA1tVYUCoBAZW9Ck42B1trPWBsjY4ApDCbAAA1Ag/lbZWBFoHBgOZpzUASM7BI0Q2hmA7vwORrBSBmFtgkXQfRaai3FryW2TB6CKNITgBI/hgDeAkPQfOoscBsGMMASQJiCAcX6IGfO10fZsz2KLFkDRnYZ2OGo3wOBnbjjdEojYNAjDc0EcI0RSi5DBw3GHCO8hlBqE0KQuOhgHFJ1cqnCehgCCZ3LjdXOQIEHoCLiXawWd6iNDyJ4JgPg/BdAMJEJYVQagGCyDkJo%2BQ2lFAyH0soTARjdPGJMBo2iBgLFmMEaZjTmgLAmWMFIkx5lDMWU4VZXT1lSArlXMOn9zpwNIWTEeD1bATxelLd6s8bbz0XsvDia8t47z3gfXe18pwinWI/Omz9yCv3fmScgwN0jShCNIdI4d4bSEgajRFWhZAnS/iQ0mCDKbUwofTahEAUAuI5kwuhZsLZ3WuWnO5b0Z5kueUvEqbzTob23rvfeD9wi9jIDSAwkcQ6SHDkkqOmTY41xmRJYIEAvALI6S0tZKwNklH6XkWVIzSgDIVT0pZsyVlDDVQ03VTBBjtC1VM3Z%2Brtk1z2ZUA598aYcWwLyrQpzMX8DJhECI/D9SnE3nhU4cS/Lz2PqfSe9y6Vz1IAvRlK93lsq%2BYfX5t8IEZFOI8%2Bh/z76Asoag8G6MoYFtSIQ7%2BisyEFxxcgoFn8QhurLdmwG5Ai45A8PDIAA%3D%3D)

## Chat Room

Want to chat with other members of the TypeScriptCompiler community?

- [GitHub Discussions](https://github.com/ASDAlexander77/TypeScriptCompiler/discussions) (preferred)
- [![Join the chat at https://gitter.im/ASDAlexander77/TypeScriptCompiler](https://badges.gitter.im/Join%20Chat.svg)](https://gitter.im/ASDAlexander77/TypeScriptCompiler?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge) (legacy)

## Example

```typescript
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

```bat
tslang --emit=jit --opt --shared-libs=TypeScriptRuntime.dll example.ts
```

Result

```text
Department name: Accounting and Auditing
The Accounting Department meets each Monday at 10am.
```

## Run as JIT

File ``hello.ts``

```typescript
function main() {
    print("Hello World!");
}
```

Build

```bat
tslang hello.ts
```

Result

```text
Hello World!
```

### JIT cache

The JIT compiles the program and each `.ts` module it imports into an object file of its own and
keeps it in a `__jit` folder next to the source file. On the next run it loads those objects
instead of compiling the files again, as long as nothing they were compiled from has changed: the
file itself, the files it references or imports, `lib.d.ts`, the options, or the `tslang` build.
An edited module is recompiled together with every file that imports it.

```text
hello.ts
__jit/hello.ts.<hash>.o        the object
__jit/hello.ts.<hash>.o.deps   what it was compiled from
```

- `--jit-cache-dir=<folder>` keeps all the objects in one folder instead;
- `--jit-cache=false` turns the cache off: everything is compiled into one module on every run.

A program whose imports form a cycle is compiled into one object. Its modules can't be compiled
one by one, which is also true for `--emit=obj`.

## Debugging JIT code with GDB (Linux)

JIT-compiled TypeScript can be debugged at source level with GDB — breakpoints on `.ts` lines,
stepping, and backtraces all work. The JIT registers every compiled object with the debugger via
the standard [GDB JIT interface](https://sourceware.org/gdb/current/onlinedocs/gdb.html/JIT-Interface.html),
so an attached GDB picks up the DWARF debug info at runtime.

Two things are required:

- pass `--di` so the compiler emits debug information;
- run **without** `--opt` (JIT debug registration is only enabled when optimizations are off).

```bash
gdb --args tslang --di hello.ts
(gdb) break hello.ts:2
Make breakpoint pending on future shared library load? (y or [n]) y
(gdb) run
```

The breakpoint stays *pending* until the JIT compiles and registers your code, then binds and stops
with full source context:

```text
Thread 1 "tslang" hit Breakpoint 1, main () at hello.ts:2
2           print("Hello World!");
```

From there the usual GDB commands work on the JIT'd frames: `next`, `step`, `info locals`, `bt`.

Notes:

- The `(No debugging symbols found in tslang)` warning at startup refers to the compiler binary
  itself and is harmless — the debug info for *your* code arrives when the JIT registers it.
- The `debugger;` TypeScript statement is also supported: it compiles to a debug trap instruction,
  so execution stops exactly there when a debugger is attached. Without a debugger attached it
  terminates the process with `SIGTRAP`, so remove it when you are done.
- For LLDB, enable its JIT loader first: `settings set plugin.jit-loader.gdb enable`.

## Compile as Binary Executable

> Make sure you have download `tslang` from releases first.

The compile process may use a few environment variables so the linker can find the
runtime libraries, then invoke `tslang` with `--emit=exe`. **Edit the paths to match your
checkout** — the `C:\dev\...` / `~/dev/...` values are only examples.

| Variable | Points to |
| --- | --- |
| `GC_LIB_PATH` | Boehm GC library (the garbage collector) |
| `LLVM_LIB_PATH` | Not needed any more: programs no longer link an LLVM library. Still accepted so older scripts keep working |
| `TSLANG_LIB_PATH` | TSLANG runtime library |
| `DEFAULT_LIB_PATH` | Default library |

### Compile on Windows

Build

```bat
tslang --emit=exe hello.ts
```

Run

```text
hello.exe
```

Result

```text
Hello World!
```

### Compile on Linux (Ubuntu 20.04 and 22.04)

Build

```bash
./tslang --emit=exe hello.ts --relocation-model=pic
```

Run

```text
./hello
```

Result

```text
Hello World!
```

## Compile for Android

`tslang` cross-compiles for Android from a Windows host. It targets `arm64-v8a`
(`-mtriple=aarch64-linux-android29`) and `x86_64` (`-mtriple=x86_64-linux-android29`), and links
through the Android NDK (tested with r29 and r30). The triple has to carry the API level, and 29
is the lowest the default library supports. Point `--android-ndk-path` (or `ANDROID_NDK_HOME`) at
the NDK.

The Windows release zip carries the Android libraries for both ABIs:

| Library | Option |
| --- | --- |
| `android\<abi>\lib\libgc.a` | `--gc-lib-path=<zip>\android\<abi>\lib` (only under `-mm=gc`) |
| `android\<abi>\lib\libTypeScriptAsyncRuntime.a` | `--tslang-lib-path=<zip>\android\<abi>\lib` |
| `defaultlib\...` (the default library, beside the Windows one) | `--default-lib-path=<zip>` |

To build them from source instead, see [Build for Android](#build-for-android).

Build an executable for arm64-v8a:

```bat
tslang --emit=exe -mtriple=aarch64-linux-android29 --opt ^
    --android-ndk-path=C:\Android\android-ndk-r30 ^
    --default-lib-path=<zip> ^
    --gc-lib-path=<zip>\android\arm64-v8a\lib ^
    --tslang-lib-path=<zip>\android\arm64-v8a\lib ^
    hello.ts -o hello
```

`--emit=dll` takes the same options and builds a shared library (`-o libhello.so`) for an app to
load. Either way the output is one self-contained, position-independent binary: the default
library, collector and libc++ are linked in statically, so it needs only Bionic's libc, libm and
libdl. Android has no libcurl, so `fetch()` throws.

Run it on a device or emulator. Run `chmod +x` because a file pushed from Windows is not executable:

```bat
adb push hello /data/local/tmp/
adb shell "chmod +x /data/local/tmp/hello && /data/local/tmp/hello"
```

Result

```text
Hello World!
```

## Compiling as WASM

### WASM build on Windows

Build

```bat
tslang.exe --emit=exe -mm=none -mtriple=wasm32-unknown-unknown hello.ts
```

Run ``run.html``

<details>
<summary>Click to expand the full <code>run.html</code> WebAssembly loader</summary>

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

</details>

## C bindings with tsbindgen

`tsbindgen` reads a C header (or `.c` file) with clang and prints the matching tslang declarations.

```text
tsbindgen <input.h|.c|.cpp> [options] [-- extra clang args]

  -o out.ts          write to a file (default: stdout)
  -I dir, -D name=v  include directories and macros, as for clang
  --filter <glob>    emit the declarations whose C name matches (repeatable)
  --namespace N      wrap the output in `namespace N`
  --strip-prefix P   remove P from the TS names (needs --namespace)
  --target triple    target triple (default: the host)
```

Given `simple.c`:

```c
#include <stdint.h>
#define MAX_ITEMS 16
typedef struct { int32_t x; int32_t y; } Point;
int add(int a, int b) { return a + b; }
static int helper(int a) { return a; }
double dist(const Point *p);
```

`tsbindgen simple.c -o simple.ts` writes:

```typescript
const MAX_ITEMS = 16;
type Point = [x: s32, y: s32];
declare function add(a: s32, b: s32): s32;
// skipped: helper — static function: no symbol to link against
declare function dist(p: Reference<Point>): f64;
```

Include the result with `/// <reference path="simple.ts" />`, not `import`. Name it `.ts`, not `.d.ts`: a `.d.ts` file is read as declarations only, so its constants would have no value.

### When nothing is emitted

- **Declarations from `#include`d headers are emitted only with `--filter`.** Without it, only what the input file itself declares is printed; a file that only includes headers prints nothing but the warning `nothing to emit`. Pick what you need by name:

  ```bat
  tsbindgen inc.c --filter puts --filter printf
  ```

  ```typescript
  declare function puts(_Buffer: string): s32;
  @varargs declare function printf(_Format: string): s32;
  ```

- **`static` and `static inline` functions are skipped**: there is no symbol to link against.
- **Function-like macros and macros that are not literals are skipped.**

### "function without a prototype"

```c
extern int test_func();
```

gives `// skipped: test_func — function without a prototype`. Before C23, empty parentheses do not mean "no arguments": they leave the arguments unspecified, and clang parses C17 by default. Either write `(void)`:

```c
extern int test_func(void);
```

or, for a header you cannot change, parse it as C23 (everything after `--` goes to clang):

```bat
tsbindgen test.c -o test.ts -- -std=c23
```

Both give:

```typescript
declare function test_func(): s32;
```

### C++ headers

`.cpp`, `.cc`, `.cxx`, `.hpp`, `.hh` and `.hxx` files are parsed as C++; a `.h` file is parsed as C, so a C++ header named `.h` needs `-x c++` after `--`:

```bat
tsbindgen mylib.h -o mylib.ts -- -x c++
```

Given `mylib.h`:

```cpp
#include <cstdint>
namespace lib { class Widget { public: int size() const; }; int helper(int); }
struct Point { int32_t x; int32_t y; };
enum class Mode : int { A = 1, B = 2 };
#ifdef __cplusplus
extern "C" {
#endif
int add(int a, int b);
double dist(const Point *p);
#ifdef __cplusplus
}
#endif
int cxx_only(int a);
```

it writes:

```typescript
type Point = [x: s32, y: s32];
enum Mode { A = 1, B = 2 }
declare function add(a: s32, b: s32): s32;
declare function dist(p: Reference<Point>): f64;
```

Only plain structs, enums and `extern "C"` functions are emitted. Classes, namespace members (`lib::helper`) and functions with C++ linkage (`cxx_only`) are left out, without a warning: their symbol names are mangled, so tslang cannot link to them. To bind a C++ API, expose it through `extern "C"` wrapper functions (with an opaque handle for each class) and run tsbindgen on the header that declares them.

## Memory models

How heap memory is managed is selected with `-mm=`:

| flag | | |
| --- | --- | --- |
| `-mm=gc` | garbage collection (Boehm) | the default |
| `-mm=rc` | reference counting - freed as soon as the last reference goes, no collector, no `libgc` | **does not collect reference cycles** |
| `-mm=none` | nothing is ever freed | short-lived programs |
| `-mm=own` | single ownership inferred at compile time - no collector, no counting | **in development**; ownership that cannot be proven is a compile error |

`-mm=gc` is the default and needs no thought. `-mm=rc` reclaims memory deterministically and
holds close to the working set - a ray tracer that reaches 114 MB under `-mm=none` holds 4.1 MB,
against garbage collection's 5.8 - but **objects that refer to each other in a cycle are never
freed under it**, which is the same trade Swift makes with ARC.

See **[docs/memory-models.md](docs/memory-models.md)** for which shapes leak, which do not, and
what to do about it.

### `-mm=own`

`-mm=own` works the way Rust's borrow checker does. Every heap block has exactly one owner: a
local, a field, an array element or a global. When a value is aliased, the compiler either moves
it, so the source gives it up, or borrows it for as long as the owner lives. The owner frees the
block at a point known at compile time: when its scope ends, when it is overwritten, or when it is
removed from its container. No reference counts are kept at run time. When the compiler cannot
prove a single owner, it rejects the program instead of compiling it with a leak:

```typescript
class Node { constructor(public value: number) {} }
class Holder { node: Node; }

function main() {
    const h = new Holder();
    const g = new Holder();
    const n = new Node(1);
    h.node = n;     // the Node moves into h
    g.node = n;     // a second owner: compile error
    print(h.node.value);
}
```

```text
owners.ts:9:5: error: 'this value' is used here after its value was moved
    g.node = n;     // a second owner: compile error
    ^
owners.ts:8:5: note: value moved here
    h.node = n;     // the Node moves into h
    ^
```

Shared ownership is opt-in and explicit. `Shared<T>` keeps a count, and `.value` reads or writes
the object inside it:

```typescript
const a = new Shared<Node>(new Node(1));
const b = a;                                          // both share the Node
print(a.value.value, b.value.value, Shared.count(a)); // 1 1 2
```

`-mm=own` is still in development: some valid TypeScript does not compile under it yet. The
default library has no `-mm=own` build yet, so compile with `--no-default-lib`.

## Build

### Build on Windows

#### Windows Requirements

- ``Visual Studio 2026``
- ~50 GB of free disk space (LLVM/MLIR build dominates this)

> The hardcoded `C:\dev\...` paths in the scripts below are examples — adjust them to your checkout location.

First, precompile dependencies

```bat
cd TypeScriptCompiler
prepare_3rdParty.bat
```

To build ``TSLANG`` binaries:

```bat
cd TypeScriptCompiler\tslang
config_tslang_release.bat
build_tslang_release.bat
```

### Build on Linux (Ubuntu 20.04 and 22.04)

#### Linux Requirements

- `gcc` or `clang`
- `cmake`
- `ninja-build`
- ~50 GB of free disk space (LLVM/MLIR build dominates this)
- sudo apt-get install ``libtinfo-dev``

First, precompile dependencies

```bash
chmod +x *.sh
cd ~/TypeScriptCompiler
./prepare_3rdParty.sh
```

To build ``TSLANG`` binaries:

```bash
cd ~/TypeScriptCompiler/tslang
chmod +x *.sh
./config_tslang_release.sh
./build_tslang_release.sh
```

### Build for Android

The release zip already carries these libraries. To build them yourself, use a Windows host with
the Android NDK (r29 or r30) in `ANDROID_NDK_HOME` and Ninja on `PATH`. Run these after
`prepare_3rdParty.bat`, which provides the gc sources and the LLVM headers they use. Each script builds both
`arm64-v8a` and `x86_64` for API level 29, or only the ABI you pass:

```bat
cd TypeScriptCompiler
scripts\build_gc_release_android.bat
scripts\build_tslang_runtime_release_android.bat
```

| Script | Builds | Pass it as |
| --- | --- | --- |
| `build_gc_release_android.bat` | Boehm GC, static, parallel marking off ([bdwgc#980](https://github.com/bdwgc/bdwgc/issues/980)) | `--gc-lib-path=3rdParty\gc\android\<abi>\release\lib` |
| `build_tslang_runtime_release_android.bat` | `libTypeScriptAsyncRuntime.a` | `--tslang-lib-path=__build\tslang-runtime\release\android\<abi>` |

Build the default library for Android with `scripts\build_android.bat` in the
[Default Library repo](https://github.com/ASDAlexander77/TypeScriptCompilerDefaultLib/). It
builds the release and debug libraries under gc, rc and none into the same `__build\defaultlib` tree as
every other target, so `--default-lib-path=<TypeScriptCompilerDefaultLib>\__build` finds them.

## License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

## Contributing

Issues and pull requests are welcome. See the
[Wiki](https://github.com/ASDAlexander77/TypeScriptCompiler/wiki) for documentation,
build notes, and how-to guides.
