#!/bin/sh
mkdir -p ../__build/tslang/ninja/release
cd ../__build/tslang/ninja/release
cmake ../../../../tslang -G "Ninja" -DCMAKE_BUILD_TYPE=Release -Wno-dev

