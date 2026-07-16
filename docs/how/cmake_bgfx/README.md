# bgfx + GLFW TypeScript sample

Cross-platform graphics sample for the TypeScript native compiler (`tslang`). Application logic is written in TypeScript; GLFW windowing and bgfx rendering live in a thin C++ bridge exposed through `.d.ts` FFI declarations.

Control flow matches [`cmake_vulkan`](../cmake_vulkan/) and [`cmake_winapp`](../cmake_winapp/): **C++ owns the event loop**, TypeScript creates the window and reacts via an `onMessage` callback.

## What it does

- Opens an 800x600 GLFW window (no client API — bgfx owns rendering)
- Initializes bgfx with the native window handle (Win32, X11, or Cocoa)
- C++ `run_loop()` polls GLFW and dispatches `Messages.Frame` each iteration
- TypeScript `onMessage` calls `run_bgfx_frame()` on Frame (like `run_vulkan()` on Paint)
- Renders an animated clear color and on-screen debug text: "Hello from tslang"
- Quits on Escape or window close

## Prerequisites

1. **Built `tslang`** — follow the main [README](../../../README.md) to build the compiler and install the default library (`tslang --install-default-lib`).
2. **CMake 3.20+** and **Ninja** (required for the custom `TSLANG` CMake language).
3. **C++20** toolchain (GCC, Clang, or MSVC).
4. **Graphics drivers** — bgfx auto-selects an available backend (OpenGL, Vulkan, DirectX on Windows).

### Linux packages (Fedora example)

```bash
sudo dnf install cmake ninja-build gcc-c++ ncurses-devel \
  libX11-devel libXcursor-devel libXi-devel libXrandr-devel \
  libXinerama-devel libXxf86vm-devel mesa-libGL-devel
```

### Windows

- Visual Studio 2022+ with C++ workload, CMake, and Ninja.
- Ensure `tslang.exe` is on `PATH` or pass `-DTSLANG_ROOT=...` pointing at the folder that contains `bin/tslang.exe`.

## Layout

```
cmake_bgfx/
├── CMakeLists.txt          # FetchContent GLFW + bgfx, mixed TS/C++ target
├── CMakePresets.json
├── cmake/                  # TSLANG custom language modules
├── src/
│   ├── main.ts             # entry point (Main)
│   ├── application.ts      # constructs AppWindow only
│   ├── appwindow.ts        # onMessage handler (Frame / KeyDown / Close / Destroy)
│   └── bgfx_glfw.d.ts      # FFI declarations
└── native/
    ├── bgfx_bridge.cpp     # extern "C" GLFW + bgfx glue + run_loop()
    └── main_entry.cpp      # main() -> Main() -> run_loop()
```

## Build and run

### Linux

```bash
cd docs/how/cmake_bgfx

cmake --preset default
# or explicitly:
# cmake --preset default -DTSLANG_ROOT=/path/to/TypeScriptCompiler/__build

cmake --build --preset default
cmake --build --target run
```

Binary: `build/cmake_bgfx`

If CMake reports `tslang compiler not found`, build the compiler from the repo root first:

```bash
./prepare_3rdParty_release.sh
cd tslang && ./config_tslang_release.sh && ./build_tslang_release.sh
./bin/tslang --install-default-lib
```

Then re-run `cmake --preset default` from `docs/how/cmake_bgfx`.

### Windows

```bat
cd docs\how\cmake_bgfx

cmake --preset default -DTSLANG_ROOT=C:\path\to\TypeScriptCompiler\__build
cmake --build --preset default
cmake --build --target run
```

Binary: `build\cmake_bgfx.exe`

### macOS

Same CMake flow as Linux. Requires Xcode command-line tools and a working OpenGL/Metal backend for bgfx.

### Debug build

```bash
cmake --preset debug
cmake --build --preset debug
```

## How it works

1. **`main_entry.cpp`** calls TypeScript `Main()`, then C++ `run_loop()`.
2. **`Main()`** constructs `AppWindow`, which registers `onMessage` and calls `create_bgfx`.
3. **`run_loop()`** polls GLFW, dispatches `Messages.Frame` each tick, then `Messages.Destroy` on exit.
4. **CMake `TSLANG` language** compiles `.ts` sources to object files with `tslang --emit=obj`.
5. **`bgfx_bridge.cpp`** implements the flat `extern "C"` API declared in `bgfx_glfw.d.ts`.
6. **FetchContent** downloads GLFW 3.4 and [bgfx.cmake](https://github.com/bkaradzic/bgfx.cmake) on first configure.

## Platform notes

| Platform | Window backend | Notes |
|----------|----------------|-------|
| Windows  | Win32 (`glfwGetWin32Window`) | bgfx D3D11/D3D12/OpenGL/Vulkan |
| Linux    | X11 (`glfwGetX11Display` / `glfwGetX11Window`) | Wayland disabled (`GLFW_BUILD_WAYLAND=OFF`) |
| macOS    | Cocoa (`glfwGetCocoaWindow`) | OpenGL/Metal via bgfx |

## Related samples

- [`cmake_winapp`](../cmake_winapp/) — Win32 window only (C++ message loop)
- [`cmake_vulkan`](../cmake_vulkan/) — Win32 + Vulkan cube (same onMessage pattern)
- [`cmake_tslang`](../cmake_tslang/) — mixed C++/TypeScript with custom CMake language
- [`c-cpp-header-import.md`](../../c-cpp-header-import.md) — FFI / native binding design
