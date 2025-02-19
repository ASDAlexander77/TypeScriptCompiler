#!/bin/sh
cd __build/llvm/ninja/debug
cmake --build . --config Debug --target install -j 2
cmake --install . --config Debug
