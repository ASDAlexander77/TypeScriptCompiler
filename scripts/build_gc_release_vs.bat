pushd
mkdir __build\gc\msbuild\x64\release
cd __build\gc\msbuild\x64\release
if exist "C:/Program Files/Microsoft Visual Studio/2022/Professional" set EXTRA_PARAM=-DCMAKE_GENERATOR_INSTANCE="C:/Program Files/Microsoft Visual Studio/2022/Professional"
cmake ../../../../../3rdParty/gc-8.0.4 -G "Visual Studio 17 2022" -A x64 %EXTRA_PARAM% -DCMAKE_BUILD_TYPE=Release -Wno-dev -DCMAKE_INSTALL_PREFIX=../../../../../3rdParty/gc/x64/release -Denable_threads=ON -Denable_cplusplus=OFF
cmake --build . --config Release -j 8
cmake --install . --config Release
popd