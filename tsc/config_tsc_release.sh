#!/bin/sh
mkdir -p ../__build/tsc/ninja/release
cd ../__build/tsc/ninja/release
cmake ../../../../tsc -G "Ninja" -DCMAKE_BUILD_TYPE=Release -Wno-dev

