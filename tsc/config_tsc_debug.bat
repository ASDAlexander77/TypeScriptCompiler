pushd
mkdir "../__build/tsc/msbuild/x64/debug"
cd "../__build/tsc/msbuild/x64/debug"
cmake ../../../../../tsc -G "Visual Studio 18 2026" -A x64 -DCMAKE_BUILD_TYPE=Debug -DTSC_PACKAGE_VERSION:STRING=1.2.3 -DTYPESCRIPT_INCLUDE_TESTS=any -Wno-dev %EXTRA_PARAM%
popd
