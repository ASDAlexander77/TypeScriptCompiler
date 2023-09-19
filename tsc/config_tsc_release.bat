pushd
mkdir "../__build/tsc/msbuild/x64/release"
cd "../__build/tsc/msbuild/x64/release"
if exist "C:/Program Files/Microsoft Visual Studio/2022/Professional" set EXTRA_PARAM=-DCMAKE_GENERATOR_INSTANCE="C:/Program Files/Microsoft Visual Studio/2022/Professional"
cmake ../../../../../tsc -G "Visual Studio 17 2022" -A x64 -DCMAKE_BUILD_TYPE=Release -Wno-dev %EXTRA_PARAM%
popd
