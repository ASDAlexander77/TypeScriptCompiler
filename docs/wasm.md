# WebAssembly

tslang targets `wasm32-unknown-unknown` and `wasm64-unknown-unknown`. The current way to run a
program is the JavaScript host in [tslang/test/wasm](../tslang/test/wasm/README.md) (`host.mjs`,
`run.mjs`), which provides the C library imports, exceptions, async and threads, for Node and the
browser:

```bat
tslang --emit=exe -mm=rc --opt -mtriple=wasm32-unknown-unknown --default-lib-path=<...>\__build -o=hello.wasm hello.ts
node run.mjs hello.wasm
```

`-mm=gc` is not available for wasm: pass `-mm=rc`, `-mm=own` or `-mm=none`. The default library for
wasm is built by `scripts\build_wasm.bat` / `scripts/build_wasm.sh` in the
[Default Library repo](https://github.com/ASDAlexander77/TypeScriptCompilerDefaultLib/).

## Minimal loader without the default library (older recipe)

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
