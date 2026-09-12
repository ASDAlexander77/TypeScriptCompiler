@rem Debug counterpart of build_gc_release_shared_vs.bat - Boehm built as a DLL, installed
@rem beside the static debug one.
@rem
@rem See that script for why the shared collector exists at all (item 5ao: two statically
@rem linked collectors free each other's objects). This one exists because the shared-library
@rem tests are registered per configuration and look for gc.lib under
@rem 3rdParty/gcdll/x64/<config>: with only the release build installed, a debug tree has no
@rem shared collector and every -shared test fails to link with "could not open 'gc.lib'".
@rem
@rem MultiThreadedDebug (/MTd) to match the debug test linker, which links libcmtd/libvcruntimed/
@rem libucrtd - mixing static and dynamic, or debug and release, CRTs crashes at startup.
pushd
mkdir __build\gcdll\msbuild\x64\debug
cd __build\gcdll\msbuild\x64\debug
cmake ../../../../../3rdParty/gc-8.2.12 -G "Visual Studio 18 2026" -A x64 %EXTRA_PARAM% -DCMAKE_BUILD_TYPE=Debug -DBUILD_SHARED_LIBS=ON -Wno-dev -DCMAKE_INSTALL_PREFIX=../../../../../3rdParty/gcdll/x64/debug -Denable_threads=ON -Denable_cplusplus=OFF -DCMAKE_POLICY_DEFAULT_CMP0091=NEW -DCMAKE_MSVC_RUNTIME_LIBRARY=MultiThreadedDebug
cmake --build . --config Debug -j 8
cmake --install . --config Debug
popd
