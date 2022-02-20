pushd
mkdir "../__build/tsc-release"
cd "../__build/tsc-release"
cmake ../../tsc -G "Visual Studio 17 2022" -A x64 -DCMAKE_BUILD_TYPE=Release -Wno-dev
popd
