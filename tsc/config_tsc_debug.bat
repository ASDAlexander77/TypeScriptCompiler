pushd
mkdir "../__build/tsc"
cd "../__build/tsc"
if exist "C:/Program Files/Microsoft Visual Studio/2022/Professional" set EXTRA_PARAM=-DCMAKE_GENERATOR_INSTANCE="C:/Program Files/Microsoft Visual Studio/2022/Professional"
cmake ../../tsc -G "Visual Studio 17 2022" -A x64 -DCMAKE_BUILD_TYPE=Debug -Wno-dev %EXTRA_PARAM%
popd
