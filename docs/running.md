# Running and compiling programs

- [Example](#example)
- [Run as JIT](#run-as-jit) ([JIT cache](#jit-cache))
- [Debugging JIT code with GDB (Linux)](#debugging-jit-code-with-gdb-linux)
- [Compile as Binary Executable](#compile-as-binary-executable)

Other targets: [Android](android.md), [WebAssembly](wasm.md).

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
