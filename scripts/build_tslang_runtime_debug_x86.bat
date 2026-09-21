@rem TypeScriptAsyncRuntime for x86, linked into 32-bit programs (-mtriple=i686-pc-windows-msvc).
@rem Needs 3rdParty\gc\x86\debug (scripts\build_gc_debug_vs_x86.bat). Installs into
@rem __build\tslang-runtime\debug\x86; pass --tslang-lib-path=__build\tslang-runtime\debug\x86.
pushd %~dp0..
cmake -S tslang\runtime -B __build\tslang-runtime\msbuild\x86\debug -G "Visual Studio 18 2026" -A Win32 -Wno-dev -DTSLANG_GC_INCLUDE=%cd%\3rdParty\gc\x86\debug\include -DTSLANG_MLIR_INCLUDE=%cd%\3rdParty\llvm\x64\release\include -DCMAKE_INSTALL_PREFIX=%cd%\__build\tslang-runtime\debug\x86
cmake --build __build\tslang-runtime\msbuild\x86\debug --config Debug -j 8
cmake --install __build\tslang-runtime\msbuild\x86\debug --config Debug
popd
