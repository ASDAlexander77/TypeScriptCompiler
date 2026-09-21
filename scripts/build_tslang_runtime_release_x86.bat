@rem TypeScriptAsyncRuntime for x86, linked into 32-bit programs (-mtriple=i686-pc-windows-msvc).
@rem Needs 3rdParty\gc\x86\release (scripts\build_gc_release_vs_x86.bat). Installs into
@rem __build\tslang-runtime\release\x86; pass --tslang-lib-path=__build\tslang-runtime\release
@rem (the parent: for an x86 target the compiler looks in its x86 subdirectory).
pushd %~dp0..
cmake -S tslang\runtime -B __build\tslang-runtime\msbuild\x86\release -G "Visual Studio 18 2026" -A Win32 -Wno-dev -DTSLANG_GC_INCLUDE=%cd%\3rdParty\gc\x86\release\include -DTSLANG_MLIR_INCLUDE=%cd%\3rdParty\llvm\x64\release\include -DCMAKE_INSTALL_PREFIX=%cd%\__build\tslang-runtime\release\x86
cmake --build __build\tslang-runtime\msbuild\x86\release --config Release -j 8
cmake --install __build\tslang-runtime\msbuild\x86\release --config Release
popd
