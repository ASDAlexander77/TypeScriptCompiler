#!/bin/sh
cd __build/llvm/ninja/release
cmake --build . --config Release --target install -j 8
cmake --install . --config Release
