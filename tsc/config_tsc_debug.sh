#!/bin/sh
mkdir -p ../__build/tsc-ninja
cd ../__build/tsc-ninja
cmake ../../tsc -G "Ninja" -DCMAKE_BUILD_TYPE=Debug -Wno-dev

