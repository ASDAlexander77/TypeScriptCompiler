pushd
mkdir __build\gc\ninja\release
cd __build\gc\ninja\release
cmake ../../../../3rdParty/gc-8.2.8 -G "Ninja" -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=OFF -Wno-dev -DCMAKE_POSITION_INDEPENDENT_CODE=ON -DCMAKE_INSTALL_PREFIX=../../../../3rdParty/gc/release -Denable_threads=ON -Denable_cplusplus=OFF
cmake --build . --config Release -j 8
cmake --install . --config Release
popd