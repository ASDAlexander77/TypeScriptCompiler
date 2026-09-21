@rem x86 (Win32) Boehm built as a DLL, for release (--opt) 32-bit -shared links and programs that
@rem import a shared library (-mtriple=i686-pc-windows-msvc).
@rem
@rem A program that loads a tslang shared library ends up with TWO collectors when both the
@rem executable and the library link gc.lib statically: each has its own heap and its own idea
@rem of what the roots are. Objects the library allocates are then invisible to the executable's
@rem roots, so the collector frees strings the executable is still holding. See item 5ao in
@rem tslang/docs/reference-counting-evaluation.md.
@rem
@rem Static linking stays the default and is correct on its own - one binary, one collector.
@rem This build exists for the shared case, where one collector has to be shared too.
@rem
@rem Installs into 3rdParty\gcdll\x86\release, then copies the import library gc.lib and gc.dll
@rem into the x86 subdirectory of the x64 lib directory, 3rdParty\gcdll\x64\release\lib\x86. For an
@rem x86 target the compiler looks in the x86 subdirectory of the lib path it is given and finds
@rem gc.dll beside gc.lib there, so one flag serves both machines:
@rem   --gc-shared-lib-path=3rdParty\gcdll\x64\release\lib
@rem Only that x86 subdirectory is added; the x64 files are not touched.
pushd
mkdir __build\gcdll\msbuild\x86\release
cd __build\gcdll\msbuild\x86\release
cmake ../../../../../3rdParty/gc-8.2.12 -G "Visual Studio 18 2026" -A Win32 %EXTRA_PARAM% -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=ON -Wno-dev -DCMAKE_INSTALL_PREFIX=../../../../../3rdParty/gcdll/x86/release -Denable_threads=ON -Denable_cplusplus=OFF -DCMAKE_POLICY_DEFAULT_CMP0091=NEW -DCMAKE_MSVC_RUNTIME_LIBRARY=MultiThreaded
cmake --build . --config Release -j 8
cmake --install . --config Release
popd
if not exist "%~dp0..\3rdParty\gcdll\x64\release\lib\x86" mkdir "%~dp0..\3rdParty\gcdll\x64\release\lib\x86"
copy /Y "%~dp0..\3rdParty\gcdll\x86\release\lib\gc.lib" "%~dp0..\3rdParty\gcdll\x64\release\lib\x86\gc.lib"
copy /Y "%~dp0..\3rdParty\gcdll\x86\release\bin\gc.dll" "%~dp0..\3rdParty\gcdll\x64\release\lib\x86\gc.dll"
