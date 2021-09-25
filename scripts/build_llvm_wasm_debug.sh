#!/bin/sh
cd __build/llvm-wasm
cmake --build . --config Debug --target install -j 8
cmake --install . --config Debug
