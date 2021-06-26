#!/bin/sh
cd __build/llvm-ninja
cmake --build . --config Debug --target install -j 8
cmake --install . --config Debug
