pushd
mkdir "__build/release"
cd "__build/release"
cmake ../.. -G "Visual Studio 18 2026" -A x64 -DCMAKE_BUILD_TYPE=Release -Wno-dev
cmake --build . --config Release -j 1
popd
