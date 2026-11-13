# Android

## Compile for Android

`tslang` cross-compiles for Android from a Windows host. It targets `arm64-v8a`
(`-mtriple=aarch64-linux-android29`), `armeabi-v7a` (`-mtriple=armv7a-linux-androideabi29`) and
`x86_64` (`-mtriple=x86_64-linux-android29`), and links
through the Android NDK (tested with r29 and r30). The triple has to carry the API level, and 29
is the lowest the default library supports. Point `--android-ndk-path` (or `ANDROID_NDK_HOME`) at
the NDK.

The Windows release zip carries the Android libraries for all three ABIs:

| Library | Option |
| --- | --- |
| `android\<abi>\lib\libgc.a` | `--gc-lib-path=<zip>\android\<abi>\lib` (only under `-mm=gc`) |
| `android\<abi>\lib\libTypeScriptAsyncRuntime.a` | `--tslang-lib-path=<zip>\android\<abi>\lib` |
| `defaultlib\...` (the default library, beside the Windows one) | `--default-lib-path=<zip>` |

To build them from source instead, see [Build for Android](#build-the-android-libraries-from-source).

Build an executable for arm64-v8a:

```bat
tslang --emit=exe -mtriple=aarch64-linux-android29 --opt ^
    --android-ndk-path=C:\Android\android-ndk-r30 ^
    --default-lib-path=<zip> ^
    --gc-lib-path=<zip>\android\arm64-v8a\lib ^
    --tslang-lib-path=<zip>\android\arm64-v8a\lib ^
    hello.ts -o hello
```

`--emit=dll` takes the same options and builds a shared library (`-o libhello.so`) for an app to
load. Either way the output is one self-contained, position-independent binary: the default
library, collector and libc++ are linked in statically, so it needs only Bionic's libc, libm and
libdl. Android has no libcurl, so `fetch()` throws.

Run it on a device or emulator. Run `chmod +x` because a file pushed from Windows is not executable:

```bat
adb push hello /data/local/tmp/
adb shell "chmod +x /data/local/tmp/hello && /data/local/tmp/hello"
```

Result

```text
Hello World!
```

## Build the Android libraries from source

The release zip already carries these libraries. To build them yourself, use a Windows host with
the Android NDK (r29 or r30) in `ANDROID_NDK_HOME` and Ninja on `PATH`. Run these after
`prepare_3rdParty.bat`, which provides the gc sources and the LLVM headers they use. Each script builds both
`arm64-v8a`, `armeabi-v7a` and `x86_64` for API level 29, or only the ABI you pass:

```bat
cd TypeScriptCompiler
scripts\build_gc_release_android.bat
scripts\build_tslang_runtime_release_android.bat
```

| Script | Builds | Pass it as |
| --- | --- | --- |
| `build_gc_release_android.bat` | Boehm GC, static, parallel marking off ([bdwgc#980](https://github.com/bdwgc/bdwgc/issues/980)) | `--gc-lib-path=3rdParty\gc\android\<abi>\release\lib` |
| `build_tslang_runtime_release_android.bat` | `libTypeScriptAsyncRuntime.a` | `--tslang-lib-path=__build\tslang-runtime\release\android\<abi>` |

Build the default library for Android with `scripts\build_android.bat` in the
[Default Library repo](https://github.com/ASDAlexander77/TypeScriptCompilerDefaultLib/). It
builds the release and debug libraries under gc, rc, none and own into the same `__build\defaultlib` tree as
every other target, so `--default-lib-path=<TypeScriptCompilerDefaultLib>\__build` finds them.
