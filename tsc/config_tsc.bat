pushd
mkdir ..\__build\tsc
cd ..\__build\tsc
cmake ../../tsc -G "Visual Studio 16 2019" -DCMAKE_BUILD_TYPE=Debug -Thost=x64
popd
