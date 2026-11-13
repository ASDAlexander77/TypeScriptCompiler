# What's new

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
