pushd
mkdir "../__build/tsc"
cd "../__build/tsc"
cmake ../../tsc -G "Visual Studio 17 2022" -A x64 -DCMAKE_BUILD_TYPE=Debug -Wno-dev
popd
