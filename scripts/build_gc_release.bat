pushd
mkdir __build\gc-release
cd __build\gc-release
cmake ../../3rdParty/gc-8.0.4 -G "Visual Studio 16 2019" -A x64 -DCMAKE_BUILD_TYPE=Release -Wno-dev -DCMAKE_INSTALL_PREFIX=../../3rdParty/gc/release -Denable_threads=ON -Denable_cplusplus=OFF
cmake --build . --config Release -j 8
cmake --install . --config Release
popd