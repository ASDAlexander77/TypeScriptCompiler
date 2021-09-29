#!/bin/sh
cd __build/llvm-wasm/debug
cmake --build . --config Debug --target install -j 1
cmake --install . --config Debug
