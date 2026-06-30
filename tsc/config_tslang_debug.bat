pushd
mkdir "../__build/tslang/msbuild/x64/debug"
cd "../__build/tslang/msbuild/x64/debug"
cmake ../../../../../tslang -G "Visual Studio 18 2026" -A x64 -DCMAKE_BUILD_TYPE=Debug -DTSLANG_PACKAGE_VERSION:STRING=1.2.3 -DTYPESCRIPT_INCLUDE_TESTS=any -Wno-dev %EXTRA_PARAM%
popd
