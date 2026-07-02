# Hello World — TSLang + CMake

A minimal CMake project that uses the [`tslang`](../../README.md) native
TypeScript compiler to build a `*.ts` file into a native executable.

```
hello-cmake/
├── CMakeLists.txt          # top-level project, calls add_ts_executable()
├── cmake/TSLang.cmake      # reusable helper: teaches CMake to compile *.ts
└── src/hello.ts            # the program
```

## Prerequisites

You need a **built** TypeScriptCompiler tree (the `__build` folder produced by
`build_tslang_release.bat` / `.sh`). The helper derives all library paths from
its location.

## Configure & build

Point CMake at your `__build` folder via `TSLANG_ROOT`:

```bat
cd examples\hello-cmake
cmake -DTSLANG_ROOT=C:/dev/TypeScriptCompiler/__build -B build
cmake --build build
```

If your layout differs, override paths individually instead of `TSLANG_ROOT`:

```bat
cmake -B build ^
  -DTSLANG_EXECUTABLE=C:/path/to/tslang.exe ^
  -DTSLANG_LIB_PATH=C:/path/to/tslang/lib ^
  -DLLVM_LIB_PATH=C:/path/to/llvm/lib ^
  -DGC_LIB_PATH=C:/path/to/gc
```

## Run

```bat
cmake --build build --target run
```

Expected output:

```text
Hello World!
```

## Using the helper in your own project

```cmake
include(TSLang)

add_ts_executable(myapp
    src/main.ts
    OUTPUT_NAME myapp
    OPTIONS --di --opt_level=0)   # any extra tslang flags
```
