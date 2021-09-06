#!/bin/sh
mkdir -p __build/gc-ninja-release
mkdir -p 3rdParty/gc/release
cd __build/gc-ninja-release
cmake ../../3rdParty/gc-8.0.4 -G "Ninja" -DCMAKE_BUILD_TYPE=Release -Wno-dev -DCMAKE_POSITION_INDEPENDENT_CODE=ON -DCMAKE_INSTALL_PREFIX=../../3rdParty/gc/release -Denable_threads=ON
cmake --build . --config Release -j 8
cp ./lib* ../../3rdParty/gc/release/