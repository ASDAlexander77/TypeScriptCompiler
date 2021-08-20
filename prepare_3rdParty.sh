#!/bin/sh
echo "Downloading LLVM"
git submodule update --init --recursive
#echo "Configuring LLVM (Debug)"
#sh -f scripts/config_llvm_debug.sh
#echo "Building LLVM (Debug)"
#sh -f scripts/build_llvm_debug.sh
#echo "Building GC (Release)"
#sh -f scripts/build_gc_debug.sh
echo "Configuring LLVM (Release)"
sh -f scripts/config_llvm_release.sh
echo "Building LLVM (Release)"
sh -f scripts/build_llvm_release.sh
echo "Building GC (Release)"
sh -f scripts/build_gc_release.sh


