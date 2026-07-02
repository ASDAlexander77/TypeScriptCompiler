# Custom language with your own extension and compile command in CMake

This registers TypeScript CMake *language* (`TSLANG`) that:

- owns its own source extension (`.ts`),
- is built with **tslang --emit=obj**,
- produces `.obj` files that CMake links automatically alongside regular C++.

CMake then handles dependency tracking and incremental builds for `.ts`
sources the same way it does for `.cpp`.

## Layout

```
custom_lang/
├── CMakeLists.txt
├── cmake/
│   ├── CMakeDetermineTSLANGCompiler.cmake
│   ├── CMakeTSLANGCompiler.cmake.in
│   ├── CMakeTSLANGInformation.cmake
│   └── CMakeTestTSLANGCompiler.cmake
├── main.cpp
└── mycode.ts
```

## How it works

- `enable_language(TSLANG)` runs `CMakeDetermineTSLANGCompiler.cmake`, which finds
  `tslangc`, then loads `CMakeTSLANGInformation.cmake`, which registers the compile
  rule (`CMAKE_TSLANG_COMPILE_OBJECT` — *tslang --emit=obj*).
- Because `CMAKE_TSLANG_SOURCE_FILE_EXTENSIONS` contains `tslang`, any `.ts` source
  is routed to your command and compiled to a `.obj`.
- That `.obj` is added to the target and linked with `main.cpp`'s object using
  the C++ linker (`CMAKE_TSLANG_LINK_EXECUTABLE`).
- Change `mycode.ts` and only it recompiles.
- Compile: `cmake --preset default && cmake --build --preset default`

## Passing flags

```cmake
set(CMAKE_TSLANG_FLAGS "--opt_level=3")                       # global
set_source_files_properties(mycode.ts PROPERTIES
    COMPILE_OPTIONS "--define;TSLANG=1")                      # per-file
```

## Minimal alternative

If you don't need a first-class language, either:

- Mark a file as an already-built object and just link it:
  `set_source_files_properties(mycode.obj PROPERTIES EXTERNAL_OBJECT TRUE GENERATED TRUE)`
  and add it to `add_executable`, producing it with `add_custom_command`; or
- Compile a differently-named file *as C++*:
  `set_source_files_properties(mycode.tslang PROPERTIES LANGUAGE CXX)`.

Use the typescript-language setup below when `.ts` is a source type you
compile often and want CMake to treat as first-class.
