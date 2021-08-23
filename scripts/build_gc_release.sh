#!/bin/sh
mkdir -p __build/gc-ninja/release
cd __build/gc-ninja
cmake ../../3rdParty/gc-8.0.4 -G "Ninja" -DCMAKE_BUILD_TYPE=Release -Wno-dev -DCMAKE_INSTALL_PREFIX=../../3rdParty/gc/release
cmake --build . --config Release --target install -j 8
cmake --install . --config Release
