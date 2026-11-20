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

## Project file (tsconfig.json)

The options and the file to compile can live in `tsconfig.json`, beside `compilerOptions`, in a
section of tslang's own. tsc and VS Code ignore it, as tslang ignores what is tsc's.

```jsonc
{
    "extends": "./base.json",
    "compilerOptions": { "strict": true, "outDir": "out" },
    "files": ["main.ts"],
    "tslang": {
        "emit": "exe",
        "mm": "rc",
        "opt": true,
        "lib": ["mylib.lib"]
    }
}
```

```bat
tslang -p .
tslang -p path\to\tsconfig.json
tslang
```

- `-p` (or `--project`) takes the file or its folder. With no `-p` and no input file, tslang reads
  `./tsconfig.json` when that names a file to compile; otherwise it reads the program from stdin as
  before. With an input file and no `-p`, no project file is read.
- The keys of `tslang` are the command-line options without dashes (`"mm": "rc"`,
  `"opt_level": 2`, `"di": true`). A list option takes an array. A key that is not an option is an
  error.
- The file to compile is `tslang.entry`, else the only name in `files`, else the only `.ts` named
  in `include` (no wildcards). An input file on the command line wins over all of them. tslang
  compiles that file and what it imports, not a set of files.
- Of `compilerOptions`, tslang reads `strictNullChecks` (or `strict`), `outFile` (as `-o`) and
  `outDir` (the default output name in that folder). The rest are tsc's and ignored, `noEmit`
  among them, since the `tsconfig.json` that `tslang --new` writes sets it for the editor.
- An option given on the command line wins over the file, for a list too (the two are not merged).
  An option in `tslang` wins over the `compilerOptions` one that means the same.
- `extends` takes a path (or an array of them, a later one winning), relative to the file that
  names it. A package name (`"@tsconfig/node20"`) is an error: tslang does not look in
  `node_modules`.
- Relative paths are relative to the file that names them: the entry, `files`, `outDir`, `-o`,
  the `*-lib-path` options, `jit-cache-dir`, `obj`, `own-facts`, and `lib` or `shared-libs` when
  that file is there (otherwise it is a name the linker looks up).
- The file may hold comments and trailing commas.
- `--verbose` prints which project file was read.

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
