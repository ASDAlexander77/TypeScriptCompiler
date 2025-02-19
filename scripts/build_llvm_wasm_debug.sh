#!/bin/sh
cd __build/llvm/ninja/wasm/debug
cmake --build . --config Debug --target install -j 1
cmake --install . --config Debug
