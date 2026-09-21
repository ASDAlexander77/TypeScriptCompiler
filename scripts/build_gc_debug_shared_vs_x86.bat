@rem Debug counterpart of build_gc_release_shared_vs_x86.bat - x86 (Win32) Boehm built as a DLL,
@rem for debug (no --opt) x86 -shared links and programs that import a shared library.
@rem See that script for why the shared collector exists at all (item 5ao: two statically
@rem linked collectors free each other's objects).
@rem
@rem Installs into 3rdParty\gcdll\x86\debug, then copies the import library gc.lib and gc.dll into
@rem the x86 subdirectory of the x64 lib directory, 3rdParty\gcdll\x64\debug\lib\x86, so one flag
@rem serves both machines:
@rem   --gc-shared-lib-path=3rdParty\gcdll\x64\debug\lib
@rem Only that x86 subdirectory is added; the x64 files are not touched.
@rem
@rem MultiThreadedDebug (/MTd) to match the debug linker, which links libcmtd/libvcruntimed/
@rem libucrtd - mixing static and dynamic, or debug and release, CRTs crashes at startup.
pushd
mkdir __build\gcdll\msbuild\x86\debug
cd __build\gcdll\msbuild\x86\debug
cmake ../../../../../3rdParty/gc-8.2.12 -G "Visual Studio 18 2026" -A Win32 %EXTRA_PARAM% -DCMAKE_BUILD_TYPE=Debug -DBUILD_SHARED_LIBS=ON -Wno-dev -DCMAKE_INSTALL_PREFIX=../../../../../3rdParty/gcdll/x86/debug -Denable_threads=ON -Denable_cplusplus=OFF -DCMAKE_POLICY_DEFAULT_CMP0091=NEW -DCMAKE_MSVC_RUNTIME_LIBRARY=MultiThreadedDebug
cmake --build . --config Debug -j 8
cmake --install . --config Debug
popd
if not exist "%~dp0..\3rdParty\gcdll\x64\debug\lib\x86" mkdir "%~dp0..\3rdParty\gcdll\x64\debug\lib\x86"
copy /Y "%~dp0..\3rdParty\gcdll\x86\debug\lib\gc.lib" "%~dp0..\3rdParty\gcdll\x64\debug\lib\x86\gc.lib"
copy /Y "%~dp0..\3rdParty\gcdll\x86\debug\bin\gc.dll" "%~dp0..\3rdParty\gcdll\x64\debug\lib\x86\gc.dll"
