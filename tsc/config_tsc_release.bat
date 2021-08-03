pushd
mkdir ../__build/tsc-release
cd ../__build/tsc-release
cmake ../../tsc -G "Visual Studio 16 2019" -A x64 -DCMAKE_BUILD_TYPE=Release -Wno-dev
popd
