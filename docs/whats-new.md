# What's new

- A class can be given to a generic constructor type (`make<T>(ctor: new (n: number) => T, n)` called as `make(Banana, 3)`); a typed local whose initializer has a method or arrow reading it (`const c: I = { m() { return c.v; } }`) compiles; under -mm=own a module `let u: unknown = 7.5` compiles
- Type guards can be passed as callbacks (`xs.filter(isDefined)`) or stored as functions returning boolean; a function taking a number can be given integers through a callback (`[1, -2, 3].filter(pos)`); narrowing `x?: number | string` with `typeof` no longer warns "casting from optional (undefined) value"; an object with accessors given to a type literal says to use an interface
- A module variable with a written type can be read by a method in its own initializer (`const a: { v: number; m(): number } = { v: 5, m() { return a.v + 1; } }`); `AsyncGenerator<number>` takes `yield 1`; a module `unknown` variable keeps its value under -mm=rc
- A subclass object can be given to a union of its base classes; `<Shape>a` reads a union of object shapes out of an `any`; `<object>{ ... }` is an object of its own (it pointed into the stack, and crashed under -mm=rc); private names written with escapes (`#\u{78}`) are found; a class field given a whole enum is a clear error
- A field of a union of object shapes or classes is read from the member the value holds: `s.kind` of `{ size: number; kind: "sq" } | { r: number; kind: "ci" }` read the first member's field whatever the value was. Shapes whose first field is a literal of their own (`{ kind: "sq"; ... } | { kind: "ci"; ... }`) and unions of classes (`A | B | null`) are stored without a type tag: the first field, or the object's class, says which member it is. Unions with such fields or classes change layout, so the default library has to be built again with this tslang
- `readonly` fields are enforced, as tsc enforces them: only the constructor of the class that declares one (its field initializers, and arrows it calls where they are written, `(() => { this.a = 1; })()`) writes it. `o.f = v`, `o.f += v`, `o.f++`, `[o.f] = ...`, `o[k] = v` with `k` naming it, and `delete o.f` elsewhere are compile errors in every memory model; until now they compiled. Under `-mm=own` a borrow of a `readonly` field then lasts as long as its object, across calls that may free other things. See [`-mm=own` guide](own-memory-model.md#reading-the-errors)
- `export default <expression>` (`export default 2 + 2`, `export default { ... }`, `export default helper`) and default imports: `import x from "./m"` names m's default export, whatever its name (`export default function f`, `export default class`, unnamed ones too). `import` also finds `<name>.d.ts`, a folder's `index` or `package.json` entry, and packages in `node_modules` / `node_modules/@types`
- A class can be read by a key known only at run time, as JavaScript reads it: `c[k]` with `k: string` gives `any` (`undefined` for a name the object does not have) and sees the object's **actual** class (a `D` held as a `C` has `D`'s fields, when `D`'s own module reads it by a run-time key or boxes it into `any`; a class of another module that is only exported is read by the keys of the class it is held as); `k in c` too. With `k: keyof C` (or a union of member names) the result is the union of those members' types, a method bound to its object, and `c[k] = v` writes when `v` fits every named field. `keyof C` now lists a class's methods and accessors as tsc does, and `keyof typeof C` its statics. Through `any`, a method now reads bound to its object instead of throwing. Each class carries a member table for this (the vtable's slot after `.instanceOf`, so the default library has to be built again with this tslang)
- A program checks the class layout of a shared library it imports as the library loads: built against one version of a library and run with another whose class fields moved, changed type or whose instance size changed, it stops before `main` with `library '<path>': <Class>.<field> moved from offset 8 to 16; rebuild this program against the new library` instead of reading the wrong fields
- `tsconfig.json` as the project file: `tslang -p .` (or `tslang` with no input file) reads the file to compile and the options from it, tslang's own in a `"tslang"` section (`"mm": "rc"`, `"emit": "exe"`), plus `strict`, `outFile` and `outDir` from `compilerOptions`, and `extends`. Options on the command line win. A `tslang --new` project gets the section, so `tslang` in its folder runs it. See [Project file](running.md#project-file-tsconfigjson)
- A shared library's declarations live in a file beside it, `<library stem>.d.ts` (`foo.dll` → `foo.d.ts`, `libfoo.so` → `libfoo.d.ts`), written by `--emit=dll` (and beside each object by `--emit=obj`). `import` reads them from there, so a library for any target can be imported on any host: an Android `.so` on Windows, say. They are no longer exported from the library itself (`__decls_*`), so a library built by an earlier tslang has to be built again before it can be imported
- New memory model **`-mm=own`**: single ownership inferred at compile time, like Rust's borrow checker. There is no collector and no reference counting, and a program whose ownership cannot be proven is a compile error. Shared ownership is opt-in with `Shared<T>`. See [Memory models](memory-models.md) and the [`-mm=own` guide](own-memory-model.md)
- **Android** targets (`arm64-v8a`, `armeabi-v7a`, `x86_64`, API 29+): executables and shared libraries linked through the Android NDK. The release zip carries the collector, async runtime and default library for all three ABIs. See [Compile for Android](android.md)
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
