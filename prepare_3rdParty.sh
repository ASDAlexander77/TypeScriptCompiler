#!/bin/sh
echo "Downloading LLVM"
git submodule update --init --recursive
echo "Configuring LLVM (Debug)"
sh -f scripts/config_llvm_debug.sh
echo "Building LLVM (Debug)"
sh -f scripts/build_llvm_debug.sh

