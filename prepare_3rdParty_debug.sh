#!/bin/sh
echo "Downloading LLVM"
git submodule update --init --recursive
echo "Configuring LLVM (Debug)"
sh -f scripts/config_llvm_debug.sh
echo "Building LLVM (Debug)"
sh -f scripts/build_llvm_debug.sh
echo "Building GC (Debug)"
curl -o gc-8.0.4.tar.gz https://www.hboehm.info/gc/gc_source/gc-8.0.4.tar.gz
tar -xvzf gc-8.0.4.tar.gz -C ./3rdParty/
sh -f scripts/build_gc_debug.sh


