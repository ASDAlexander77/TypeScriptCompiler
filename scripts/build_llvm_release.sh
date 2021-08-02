#!/bin/sh
cd __build/llvm-ninja
cmake --build . --config Release --target install -j 8
cmake --install . --config Release
