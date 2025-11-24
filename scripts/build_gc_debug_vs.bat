pushd
mkdir __build\gc\msbuild\x64\debug
cd __build\gc\msbuild\x64\debug
if exist "C:/Program Files/Microsoft Visual Studio/18/Professional" set EXTRA_PARAM=-DCMAKE_GENERATOR_INSTANCE="C:/Program Files/Microsoft Visual Studio/18/Professional"
cmake ../../../../../3rdParty/gc-8.2.8 -G "Visual Studio 18 2026" -A x64 %EXTRA_PARAM% -DCMAKE_BUILD_TYPE=Debug -DBUILD_SHARED_LIBS=OFF -Wno-dev -DCMAKE_INSTALL_PREFIX=../../../../../3rdParty/gc/x64/debug -Denable_threads=ON -Denable_cplusplus=OFF
cmake --build . --config Debug -j 8
popd