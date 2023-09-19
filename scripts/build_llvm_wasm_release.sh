#!/bin/sh
cd __build/llvm/ninja/wasm/release
cmake --build . --config Release --target install -j 1
cmake --install . --config Release
