pushd
mkdir "../__build/tsc/msbuild/x64/release"
cd "../__build/tsc/msbuild/x64/release"
cmake ../../../../../tsc -G "Visual Studio 18 2026" -A x64 -DCMAKE_BUILD_TYPE=Release -Wno-dev %EXTRA_PARAM%
popd
