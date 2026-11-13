# TypeScript Native Compiler

Powered by [![LLVM|MLIR](https://llvm.org/img/LLVM-Logo-Derivative-1.png)](https://llvm.org/)

Developed by [![Claude Code](https://img.shields.io/badge/Claude-Code-D97757?logo=claude&logoColor=white)](https://claude.com/claude-code)

[![Donate](https://img.shields.io/badge/Donate-PayPal-green.svg)](https://www.paypal.com/donate/?hosted_button_id=BBJ4SQYLA6D2L)
[![Test Build (Windows)](https://github.com/ASDAlexander77/TypeScriptCompiler/actions/workflows/cmake-test-release-win.yml/badge.svg)](https://github.com/ASDAlexander77/TypeScriptCompiler/actions/workflows/cmake-test-release-win.yml)
[![Test Build (Linux)](https://github.com/ASDAlexander77/TypeScriptCompiler/actions/workflows/cmake-test-release-linux.yml/badge.svg)](https://github.com/ASDAlexander77/TypeScriptCompiler/actions/workflows/cmake-test-release-linux.yml)

A native ahead-of-time (AOT) and JIT compiler for **TypeScript**, built on **LLVM/MLIR**.
It compiles `.ts` files directly to native executables, shared libraries and WebAssembly, or runs
them on the fly via a built-in JIT — no Node.js or JavaScript runtime required.

**Targets:** Windows, Linux, Android (`arm64-v8a`, `armeabi-v7a`, `x86_64`), WebAssembly (`wasm32`, `wasm64`).

## Contents

- [Quick start](#quick-start)
- [Memory models](#memory-models)
- [Documentation](#documentation)
- [Showcase: native_ts_graphics](#showcase-native_ts_graphics)
- [Community](#community) · [License](#license) · [Contributing](#contributing)

## Quick start

Download `tslang` from [Releases](https://github.com/ASDAlexander77/TypeScriptCompiler/releases/),
or [try it on Compiler Explorer](https://godbolt.org/#z:OYLghAFBqd5TKALEBjA9gEwKYFFMCWALugE4A0BIEAZgQDbYB2AhgLbYgDkAjF%2BTXRMiAZVQtGIHgA4BQogFUAztgAKAD24AGfgCsp5eiyagiATwAO2JalIELRcisaoiBIdWaYAwunoBXNiYQACZyLwAZAiZsADlAgCNsUhAAZh5yC3QlYncmXwCg0Mzs3KEomPi2JJT0p2wXNyERIhZSIgLA4LDnbFc8lraiCrjE5LSMpVb2zqKe6eHo0erx9IBKJ3R/UlROLhYEqdIWVwBqVCMlJVOAEWwLIY5hU4BSAHYAIRetAEFT//OQiO/lcZAgFn8CXoBFQp1YHBApyO0WAa1en3eN2%2BP2xANOFjswli7GwEDWiIAbugCJh0V9fni8QTokQIC8QiE7g92k8iHCSYj2SFXiEPqciEgCEoAHTw7BrF6pel/AGY7G4gEHI4nPnM4QAWWw2DcJjJlOpmEVYoA9NbTmx/FNTklTgQ2BZGLzsLToqccHYKd7zpcVEp1W8sb9sRcWFdTj9UBh/MIUVzHsw%2Bdh1EQvNc0zyM3SNf8MExgaDSGSiwzGf8lP4rJWhQmkymTKdjLSfv5CCbgEKFUrTrbAWWiKQQSRSNdff6CIHaTGrtZ7Y6%2BeJ6PQkQ3kmTi%2BjIzia/89URDcaUWbTlSadWVbXT2yOQAVJDYeOJrZt4C3e7p54cMa1zYCcSCnPqQiYCwZgdnyPBaOw0oDla%2B5qlGx6nMAzDJCwOYAEr3GQRBKFeN60u8yq1iehKskKADi2HHH2HafsmzGkIR7QytKSEcoOlGqhG4aHtiNDJv0Qj2iw0RVhR%2B6MHyODckQvKIvmKkZlaw52ugADW4roOcHG4e%2BLCnBxNDJMwuwGR2TAdoc446uKljYPuSn/nyio3HC2AAO4fq2fbqbye5DiOem2bYIE5nZtKxjkwD2WZTBCAAtFqTlnPWCRLmGGEeQWwjSqexIcGFAn/IVGnFae559hV%2B62tVvLSlhMRMdgBFZFxFXaacySkGQiItYWUpwugfLoDQLlWIFX7BX%2BRWOOcxipXyJy7HG7U4fhnHEcJXAbPQ3AAKz8MEXA6OQ6DcM%2BrliHYDinMSbiBkiWw7O%2B7LpPwRDaEdGy6WkqTSqk4MQ5DkMAGyGNwAAs/BsCAbzw2DqTSJjqTw/DbzQ1oWjw%2BQl3XbdXD8EoIBaOQ/1XUd5BwLAKBZn0/hTpQ1BtMASiqMYDQiEg6B%2BZdfDkBg7oMLheS8zE9AC0LJP8OLFgMOMxGoAA%2BloWsa6kp0a8AqBi%2BgEuMKQZV7ErJsq2bADybPy8LAPhOofQ/KQ3PcPwLOoC0hCXfwgjCGIEicDwRNB4oKgaM7%2BgZEYJggOYVg2E9jjQoc8AbOgDh5JT5M3YGpB2DglOQBsSifbsBhTP7Mv84LTu8PwfnHBY3B8MdZ0Xc7ZO4K7qBs2Qpz3VYj32Hyr3zu%2Bvim8kpxaNKi8EtgeunAAUgAks%2BpwQHR3hohAo/YOPz1T4G5CnPgxDDz9PBrH9ANrBsb4sP61Bd1wiPkMjISndK0NoY8GAdDU6p0QipAAJyQKAcTXuXsnBUxpk/IGaR/4Yz1uAngeNIHYLeG8OGXBUg9zpjdBBtMdAbEZsgNA1tVYUCoBAZW9Ck42B1trPWBsjY4ApDCbAAA1Ag/lbZWBFoHBgOZpzUASM7BI0Q2hmA7vwORrBSBmFtgkXQfRaai3FryW2TB6CKNITgBI/hgDeAkPQfOoscBsGMMASQJiCAcX6IGfO10fZsz2KLFkDRnYZ2OGo3wOBnbjjdEojYNAjDc0EcI0RSi5DBw3GHCO8hlBqE0KQuOhgHFJ1cqnCehgCCZ3LjdXOQIEHoCLiXawWd6iNDyJ4JgPg/BdAMJEJYVQagGCyDkJo%2BQ2lFAyH0soTARjdPGJMBo2iBgLFmMEaZjTmgLAmWMFIkx5lDMWU4VZXT1lSArlXMOn9zpwNIWTEeD1bATxelLd6s8bbz0XsvDia8t47z3gfXe18pwinWI/Omz9yCv3fmScgwN0jShCNIdI4d4bSEgajRFWhZAnS/iQ0mCDKbUwofTahEAUAuI5kwuhZsLZ3WuWnO5b0Z5kueUvEqbzTob23rvfeD9wi9jIDSAwkcQ6SHDkkqOmTY41xmRJYIEAvALI6S0tZKwNklH6XkWVIzSgDIVT0pZsyVlDDVQ03VTBBjtC1VM3Z%2Brtk1z2ZUA598aYcWwLyrQpzMX8DJhECI/D9SnE3nhU4cS/Lz2PqfSe9y6Vz1IAvRlK93lsq%2BYfX5t8IEZFOI8%2Bh/z76Asoag8G6MoYFtSIQ7%2BisyEFxxcgoFn8QhurLdmwG5Ai45A8PDIAA%3D%3D).

File `hello.ts`:

```typescript
function main() {
    print("Hello World!");
}
```

Run it with the JIT:

```bat
tslang hello.ts
```

Or compile it to an executable (on Linux add `--relocation-model=pic`):

```bat
tslang --emit=exe hello.ts
hello.exe
```

```text
Hello World!
```

Start a new project:

```bat
tslang --new Test1
```

creates a Visual Studio Code project, and

```bat
tslang --cmake Test1
```

a CMake project for mixed C++/TypeScript (see [What's new](docs/whats-new.md) for building it).

More in [Running and compiling programs](docs/running.md): a larger example, the JIT cache,
debugging JIT code with GDB, and the environment variables the linker uses.

## Memory models

How heap memory is managed is selected with `-mm=`:

| flag | | |
| --- | --- | --- |
| `-mm=gc` | garbage collection (Boehm) | the default |
| `-mm=rc` | reference counting - freed as soon as the last reference goes, no collector, no `libgc` | **does not collect reference cycles** |
| `-mm=none` | nothing is ever freed | short-lived programs |
| `-mm=own` | single ownership inferred at compile time, like Rust's borrow checker - no collector, no counting | **in development**; ownership that cannot be proven is a compile error |

See [memory-models.md](docs/memory-models.md) and the [`-mm=own` guide](docs/own-memory-model.md).
Under `-mm=own` tslang runs within 1.1-1.4x of idiomatic Rust on an object-heavy ray tracer and on
par with it on array loops — see [Performance](docs/performance.md).

## Documentation

| Topic | Page |
| --- | --- |
| Running with the JIT, compiling executables, debugging JIT code | [docs/running.md](docs/running.md) |
| Android: cross-compiling and building the runtime libraries | [docs/android.md](docs/android.md) |
| WebAssembly | [docs/wasm.md](docs/wasm.md) |
| Memory models (`gc`, `rc`, `none`) | [docs/memory-models.md](docs/memory-models.md) |
| The `-mm=own` single-owner model | [docs/own-memory-model.md](docs/own-memory-model.md) |
| Benchmarks against Rust and C++ | [docs/performance.md](docs/performance.md) |
| C and C++ bindings with `tsbindgen` | [docs/tsbindgen.md](docs/tsbindgen.md) |
| Calling C++ classes (`@cpp`) | [docs/cpp-classes.md](docs/cpp-classes.md) |
| Building the compiler from source | [docs/building.md](docs/building.md) |
| What's new and roadmap | [docs/whats-new.md](docs/whats-new.md) |
| JavaScript built-in objects library | [Default Library repo](https://github.com/ASDAlexander77/TypeScriptCompilerDefaultLib/) |
| How-to guides and more | [Wiki](https://github.com/ASDAlexander77/TypeScriptCompiler/wiki) |

## Showcase: native_ts_graphics

**[native_ts_graphics](https://github.com/ASDAlexander77/native_ts_graphics)** is a set of 25
real-time 3D graphics samples — TypeScript ports of NVIDIA's [Donut](https://github.com/NVIDIA-RTX/Donut)
samples on **Direct3D 12** and **Vulkan**, from `basic_triangle` to ray tracing, meshlets and work
graphs — compiled to native code with tslang. `.ts` files are a first-class CMake language
(`TSLANG`) linked with the C++ framework through a C interop layer, and the samples run on
**Windows** and **Android**.

## Community

- [GitHub Discussions](https://github.com/ASDAlexander77/TypeScriptCompiler/discussions) (preferred)
- [![Join the chat at https://gitter.im/ASDAlexander77/TypeScriptCompiler](https://badges.gitter.im/Join%20Chat.svg)](https://gitter.im/ASDAlexander77/TypeScriptCompiler?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge) (legacy)

## License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

## Contributing

Issues and pull requests are welcome. See the
[Wiki](https://github.com/ASDAlexander77/TypeScriptCompiler/wiki) for documentation,
build notes, and how-to guides.
