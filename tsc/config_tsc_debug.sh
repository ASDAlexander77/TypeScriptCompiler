#!/bin/sh
mkdir -p ../__build/tsc-ninja
cd ../__build/tsc-ninja
cmake ../../tsc -G "Ninja" -DCMAKE_BUILD_TYPE=Debug -Wno-dev -DMLIR_DIR=$PWD/../llvm-ninja/lib/cmake/mlir -DLLVM_EXTERNAL_LIT=$PWD/../llvm-ninja/debug/bin/llvm-lit

