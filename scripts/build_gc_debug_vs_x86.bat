pushd
mkdir __build\gc\msbuild\x86\debug
cd __build\gc\msbuild\x86\debug
cmake ../../../../../3rdParty/gc-8.2.12 -G "Visual Studio 18 2026" -A Win32 %EXTRA_PARAM% -DCMAKE_BUILD_TYPE=Debug -DBUILD_SHARED_LIBS=OFF -Wno-dev -DCMAKE_INSTALL_PREFIX=../../../../../3rdParty/gc/x86/debug -Denable_threads=ON -Denable_cplusplus=OFF -DCMAKE_POLICY_DEFAULT_CMP0091=NEW -DCMAKE_MSVC_RUNTIME_LIBRARY=MultiThreadedDebug
cmake --build . --config Debug -j 20
cmake --install . --config Debug
popd