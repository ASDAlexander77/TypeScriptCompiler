@rem TypeScriptAsyncRuntime for Android, linked into programs compiled with
@rem -mtriple=<arch>-linux-android29.
@rem
@rem   scripts\build_tslang_runtime_release_android.bat [arm64-v8a|x86_64]   (no argument: both)
@rem
@rem Needs the Android NDK in ANDROID_NDK_HOME, Ninja, the Android gc
@rem (scripts\build_gc_release_android.bat) for gc.h, and 3rdParty\llvm\x64\release\include for
@rem the one header-only MLIR file (see tslang\runtime\CMakeLists.txt). Installs
@rem libTypeScriptAsyncRuntime.a into __build\tslang-runtime\release\android\<abi>.
@echo off
setlocal

if "%ANDROID_NDK_HOME%"=="" (
    echo ANDROID_NDK_HOME is not set: point it at the Android NDK, e.g. C:\Android\android-ndk-r30
    exit /b 1
)

if "%~1"=="" (
    call "%~f0" arm64-v8a || exit /b 1
    call "%~f0" x86_64 || exit /b 1
    exit /b 0
)

set ABI=%~1
set ROOT=%~dp0..
set BUILD_DIR=%ROOT%\__build\tslang-runtime\ninja\android\%ABI%\release

cmake -S "%ROOT%\tslang\runtime" -B "%BUILD_DIR%" -G Ninja -Wno-dev ^
    -DCMAKE_TOOLCHAIN_FILE="%ANDROID_NDK_HOME%\build\cmake\android.toolchain.cmake" ^
    -DANDROID_ABI=%ABI% -DANDROID_PLATFORM=android-29 ^
    -DCMAKE_BUILD_TYPE=Release ^
    -DTSLANG_GC_INCLUDE="%ROOT%\3rdParty\gc\android\%ABI%\release\include" ^
    -DTSLANG_MLIR_INCLUDE="%ROOT%\3rdParty\llvm\x64\release\include" ^
    -DCMAKE_INSTALL_PREFIX="%ROOT%\__build\tslang-runtime\release\android\%ABI%" || exit /b 1
cmake --build "%BUILD_DIR%" -j 8 || exit /b 1
cmake --install "%BUILD_DIR%" || exit /b 1
