#!/bin/sh
cd __build/llvm-wasm
emcmake cmake --build . --config Debug --target install -j 8
emcmake cmake --install . --config Debug
