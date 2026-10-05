@rem Android Boehm, static, for programs compiled with -mtriple=<arch>-linux-android29.
@rem
@rem   scripts\build_gc_release_android.bat [arm64-v8a|x86_64]   (no argument: both)
@rem
@rem Needs the Android NDK (tested with r30) in ANDROID_NDK_HOME, and Ninja. Installs into
@rem 3rdParty\gc\android\<abi>\release. API level 29 is the floor the default library needs
@rem (timespec_get).
@rem
@rem Parallel marking is off: with it, several threads allocating at once deadlocked the
@rem collector during marking on an API 29 x86_64 emulator (bdwgc 8.2.12 and master; reported
@rem upstream as https://github.com/bdwgc/bdwgc/issues/980). Marking then runs on the collecting
@rem thread alone.
@echo off
setlocal

if "%ANDROID_NDK_HOME%"=="" (
    echo ANDROID_NDK_HOME is not set: point it at the Android NDK, e.g. C:\Android\android-ndk-r30
    exit /b 1
)
if not exist "%ANDROID_NDK_HOME%\build\cmake\android.toolchain.cmake" (
    echo "%ANDROID_NDK_HOME%" is not an Android NDK: build\cmake\android.toolchain.cmake is missing
    exit /b 1
)

if "%~1"=="" (
    call "%~f0" arm64-v8a || exit /b 1
    call "%~f0" x86_64 || exit /b 1
    exit /b 0
)

set ABI=%~1
set ROOT=%~dp0..
set BUILD_DIR=%ROOT%\__build\gc\android\%ABI%\release

cmake -S "%ROOT%\3rdParty\gc-8.2.12" -B "%BUILD_DIR%" -G Ninja -Wno-dev ^
    -DCMAKE_TOOLCHAIN_FILE="%ANDROID_NDK_HOME%\build\cmake\android.toolchain.cmake" ^
    -DANDROID_ABI=%ABI% -DANDROID_PLATFORM=android-29 ^
    -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=OFF -DCMAKE_POSITION_INDEPENDENT_CODE=ON ^
    -DCMAKE_INSTALL_PREFIX="%ROOT%\3rdParty\gc\android\%ABI%\release" ^
    -Denable_threads=ON -Denable_parallel_mark=OFF -Denable_cplusplus=OFF -Denable_docs=OFF -Dbuild_tests=OFF -Dbuild_cord=OFF || exit /b 1
cmake --build "%BUILD_DIR%" -j 8 || exit /b 1
cmake --install "%BUILD_DIR%" || exit /b 1
