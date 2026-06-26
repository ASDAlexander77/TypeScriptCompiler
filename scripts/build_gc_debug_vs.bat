pushd
mkdir __build\gc\msbuild\x64\debug
cd __build\gc\msbuild\x64\debug
cmake ../../../../../3rdParty/gc-8.2.8 -G "Visual Studio 18 2026" -A x64 %EXTRA_PARAM% -DCMAKE_BUILD_TYPE=Debug -DBUILD_SHARED_LIBS=OFF -Wno-dev -DCMAKE_INSTALL_PREFIX=../../../../../3rdParty/gc/x64/debug -Denable_threads=ON -Denable_cplusplus=OFF -DCMAKE_POLICY_DEFAULT_CMP0091=NEW -DCMAKE_MSVC_RUNTIME_LIBRARY=MultiThreadedDebug
cmake --build . --config Debug -j 8
cmake --install . --config Debug
rem the project links gc against the install root (GC_LIB_PATH), but install puts libs in lib\ - mirror them up
copy /Y ..\..\..\..\..\3rdParty\gc\x64\debug\lib\*.lib ..\..\..\..\..\3rdParty\gc\x64\debug\
copy /Y ..\..\..\..\..\3rdParty\gc\x64\debug\lib\*.pdb ..\..\..\..\..\3rdParty\gc\x64\debug\
popd