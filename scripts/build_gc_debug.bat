pushd
mkdir __build\gc
cd __build\gc
cmake ../../3rdParty/gc-8.0.4 -G "Visual Studio 16 2019" -A x64 -DCMAKE_BUILD_TYPE=Debug -Wno-dev -DCMAKE_INSTALL_PREFIX=../../3rdParty/gc/debug -DGC_NAMESPACE=1 -DGC_NOT_DLL=1
cmake --build . --config Debug -j 8
popd