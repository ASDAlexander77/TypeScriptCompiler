#!/bin/sh
cd __build/llvm-ninja
cmake --build . --config Debug --target install -j 2
cmake --install . --config Debug
