@rem x86 (Win32) Boehm, static, for release (--opt) 32-bit programs (-mtriple=i686-pc-windows-msvc).
@rem
@rem Installs into 3rdParty\gc\x86\release, then copies gc.lib into the x86 subdirectory of the
@rem x64 lib directory, 3rdParty\gc\x64\release\lib\x86. For an x86 target the compiler looks in
@rem the x86 subdirectory of the lib path it is given, so one flag serves both machines:
@rem   --gc-lib-path=3rdParty\gc\x64\release\lib
@rem Only that x86 subdirectory is added; the x64 files are not touched.
pushd
mkdir __build\gc\msbuild\x86\release
cd __build\gc\msbuild\x86\release
cmake ../../../../../3rdParty/gc-8.2.12 -G "Visual Studio 18 2026" -A Win32 %EXTRA_PARAM% -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=OFF -Wno-dev -DCMAKE_INSTALL_PREFIX=../../../../../3rdParty/gc/x86/release -Denable_threads=ON -Denable_cplusplus=OFF -DCMAKE_POLICY_DEFAULT_CMP0091=NEW -DCMAKE_MSVC_RUNTIME_LIBRARY=MultiThreaded
cmake --build . --config Release -j 8
cmake --install . --config Release
popd
if not exist "%~dp0..\3rdParty\gc\x64\release\lib\x86" mkdir "%~dp0..\3rdParty\gc\x64\release\lib\x86"
copy /Y "%~dp0..\3rdParty\gc\x86\release\lib\gc.lib" "%~dp0..\3rdParty\gc\x64\release\lib\x86\gc.lib"
