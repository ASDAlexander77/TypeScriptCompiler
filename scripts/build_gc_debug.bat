pushd
mkdir __build\gc\msbuild\x64\debug
cd __build\gc\msbuild\x64\debug
if exist "C:/Program Files/Microsoft Visual Studio/2022/Professional" set EXTRA_PARAM=-DCMAKE_GENERATOR_INSTANCE="C:/Program Files/Microsoft Visual Studio/2022/Professional"
cmake ../../../../../3rdParty/gc-8.0.4 -G "Visual Studio 17 2022" -A x64 %EXTRA_PARAM% -DCMAKE_BUILD_TYPE=Debug -Wno-dev -DCMAKE_INSTALL_PREFIX=../../../../../3rdParty/gc/x64/debug -Denable_threads=ON -Denable_cplusplus=OFF
cmake --build . --config Debug -j 8
popd