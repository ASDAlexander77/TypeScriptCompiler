# Building from source

## Build on Windows

### Windows Requirements

- ``Visual Studio 2026``
- ~50 GB of free disk space (LLVM/MLIR build dominates this)

> The hardcoded `C:\dev\...` paths in the scripts below are examples — adjust them to your checkout location.

First, precompile dependencies

```bat
cd TypeScriptCompiler
prepare_3rdParty.bat
```

To build ``TSLANG`` binaries:

```bat
cd TypeScriptCompiler\tslang
config_tslang_release.bat
build_tslang_release.bat
```

## Build on Linux (Ubuntu 20.04 and 22.04)

### Linux Requirements

- `gcc` or `clang`
- `cmake`
- `ninja-build`
- ~50 GB of free disk space (LLVM/MLIR build dominates this)
- sudo apt-get install ``libtinfo-dev``

First, precompile dependencies

```bash
chmod +x *.sh
cd ~/TypeScriptCompiler
./prepare_3rdParty.sh
```

To build ``TSLANG`` binaries:

```bash
cd ~/TypeScriptCompiler/tslang
chmod +x *.sh
./config_tslang_release.sh
./build_tslang_release.sh
```

## Android

See [Build the Android libraries from source](android.md#build-the-android-libraries-from-source).
