pushd
mkdir "../__build/tsc/msbuild/x64/debug"
cd "../__build/tsc/msbuild/x64/debug"
if exist "C:/Program Files/Microsoft Visual Studio/2022/Professional" set EXTRA_PARAM=-DCMAKE_GENERATOR_INSTANCE="C:/Program Files/Microsoft Visual Studio/2022/Professional"
cmake ../../../../../tsc -G "Visual Studio 17 2022" -A x64 -DCMAKE_BUILD_TYPE=Debug -DTSC_PACKAGE_VERSION:STRING=1.2.3 -Wno-dev %EXTRA_PARAM%
popd
