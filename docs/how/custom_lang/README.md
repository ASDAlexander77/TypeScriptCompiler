# Custom language with your own extension and compile command in CMake

This example registers a brand-new CMake *language* (`FOO`) that:

- owns its own source extension (`.foo`),
- is built with **your own compile command**,
- produces `.obj` files that CMake links automatically alongside regular C++.

CMake then handles dependency tracking and incremental builds for `.foo`
sources the same way it does for `.cpp`.

## Layout

```
custom_lang/
├── CMakeLists.txt
├── cmake/
│   ├── CMakeDetermineFOOCompiler.cmake
│   ├── CMakeFOOCompiler.cmake.in
│   ├── CMakeFOOInformation.cmake
│   └── CMakeTestFOOCompiler.cmake
├── main.cpp
└── mycode.foo
```

## How it works

- `enable_language(FOO)` runs `CMakeDetermineFOOCompiler.cmake`, which finds
  `fooc`, then loads `CMakeFOOInformation.cmake`, which registers the compile
  rule (`CMAKE_FOO_COMPILE_OBJECT` — *your command*).
- Because `CMAKE_FOO_SOURCE_FILE_EXTENSIONS` contains `foo`, any `.foo` source
  is routed to your command and compiled to a `.obj`.
- That `.obj` is added to the target and linked with `main.cpp`'s object using
  the C++ linker (`CMAKE_FOO_LINK_EXECUTABLE`).
- Change `mycode.foo` and only it recompiles.

## Passing flags

```cmake
set(CMAKE_FOO_FLAGS "--opt-level=2")                       # global
set_source_files_properties(mycode.foo PROPERTIES
    COMPILE_OPTIONS "--define;FOO=1")                      # per-file
```

## Minimal alternative

If you don't need a first-class language, either:

- Mark a file as an already-built object and just link it:
  `set_source_files_properties(mycode.obj PROPERTIES EXTERNAL_OBJECT TRUE GENERATED TRUE)`
  and add it to `add_executable`, producing it with `add_custom_command`; or
- Compile a differently-named file *as C++*:
  `set_source_files_properties(mycode.foo PROPERTIES LANGUAGE CXX)`.

Use the full custom-language setup below when `.foo` is a source type you
compile often and want CMake to treat as first-class.
