pushd
mkdir "../__build/tslang/msbuild/x64/release"
cd "../__build/tslang/msbuild/x64/release"
cmake ../../../../../tslang -G "Visual Studio 18 2026" -A x64 -DCMAKE_BUILD_TYPE=Release -Wno-dev %EXTRA_PARAM%
popd
