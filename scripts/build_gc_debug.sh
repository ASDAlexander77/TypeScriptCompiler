#!/bin/sh
mkdir -p __build/gc-ninja/debug
cd __build/gc-ninja
cmake ../../3rdParty/gc-8.0.4 -G "Ninja" -A x64 -DCMAKE_BUILD_TYPE=Debug -Wno-dev -DCMAKE_INSTALL_PREFIX=../../3rdParty/gc/debug -DGC_NAMESPACE=1 -DGC_NOT_DLL=1
cmake --build . --config Debug --target install -j 8
cmake --install . --config Debug
