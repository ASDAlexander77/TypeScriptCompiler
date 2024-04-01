pushd
mkdir __build\gc\ninja\debug
cd __build\gc\ninja\debug
cmake ../../../../3rdParty/gc-8.0.4 -G "Ninja" -DCMAKE_BUILD_TYPE=Debug -Wno-dev -DCMAKE_POSITION_INDEPENDENT_CODE=ON -DCMAKE_INSTALL_PREFIX=../../../../3rdParty/gc/debug -Denable_threads=ON -Denable_cplusplus=OFF
cmake --build . --config Debug -j 8
popd