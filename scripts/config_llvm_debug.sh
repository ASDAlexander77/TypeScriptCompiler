#!/bin/sh
mkdir -p __build/llvm-ninja/debug
cd __build/llvm-ninja
cmake ../../3rdParty/llvm-project/llvm -G "Ninja" -DLLVM_TARGETS_TO_BUILD="X86" -DCMAKE_BUILD_TYPE=Debug -DCMAKE_INSTALL_PREFIX=../../3rdParty/llvm-ninja/debug -DLLVM_INSTALL_UTILS=ON -DLLVM_ENABLE_ASSERTIONS=ON -DLLVM_ENABLE_PROJECTS="lld;mlir" -DLLVM_ENABLE_EH=ON -DLLVM_ENABLE_RTTI=ON -DLLVM_REQUIRES_RTTI=ON -DLLVM_ENABLE_PIC=ON

