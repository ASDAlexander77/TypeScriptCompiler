#!/bin/sh
mkdir -p ../__build/tsc/ninja/debug
cd ../__build/tsc/ninja/debug
cmake ../../../../tsc -G "Ninja" -DCMAKE_BUILD_TYPE=Debug -Wno-dev

