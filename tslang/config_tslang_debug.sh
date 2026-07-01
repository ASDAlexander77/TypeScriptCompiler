#!/bin/sh
mkdir -p ../__build/tslang/ninja/debug
cd ../__build/tslang/ninja/debug
cmake ../../../../tslang -G "Ninja" -DCMAKE_BUILD_TYPE=Debug -Wno-dev

