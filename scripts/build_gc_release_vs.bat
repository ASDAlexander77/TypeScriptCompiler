pushd
mkdir __build\gc\msbuild\x64\release
cd __build\gc\msbuild\x64\release
cmake ../../../../../3rdParty/gc-8.2.8 -G "Visual Studio 18 2026" -A x64 %EXTRA_PARAM% -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=OFF -Wno-dev -DCMAKE_INSTALL_PREFIX=../../../../../3rdParty/gc/x64/release -Denable_threads=ON -Denable_cplusplus=OFF -DCMAKE_POLICY_DEFAULT_CMP0091=NEW -DCMAKE_MSVC_RUNTIME_LIBRARY=MultiThreaded
cmake --build . --config Release -j 8
cmake --install . --config Release
rem the project links gc against the install root (GC_LIB_PATH), but install puts libs in lib\ - mirror them up
copy /Y ..\..\..\..\..\3rdParty\gc\x64\release\lib\*.lib ..\..\..\..\..\3rdParty\gc\x64\release\
popd