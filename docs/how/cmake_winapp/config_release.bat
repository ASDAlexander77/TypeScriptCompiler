pushd
mkdir "__build/release"
cd "__build/release"
cmake ../.. -G "Visual Studio 16 2019" -A x64 -DCMAKE_BUILD_TYPE=Release -Wno-dev
cmake --build . --config Release -j 1 -trace-source=..\..\CMakeLists.txt
popd
