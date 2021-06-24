#!/bin/sh
mkdir -p ../__build/tsc-ninja
cd ../__build/tsc-ninja
cmake ../../tsc -G "Ninja" -DCMAKE_BUILD_TYPE=Debug -Wno-dev -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ -DLLVM_ENABLE_LLD=ON -DMLIR_DIR=$PWD/../llvm-ninja/lib/cmake/mlir -DLLVM_EXTERNAL_LIT=$PWD/../llvm-ninja/debug/bin/llvm-lit

