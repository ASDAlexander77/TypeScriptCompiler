#!/bin/bash
echo "Downloading LLVM"
#git submodule update --init --recursive
echo "Configuring LLVM (Debug)"
#./scripts/config_llvm_debug.sh
echo "Building LLVM (Debug)"
./scripts/build_llvm_debug.sh
echo "Building GC (Debug)"
curl -o gc-8.0.4.tar.gz https://www.hboehm.info/gc/gc_source/gc-8.0.4.tar.gz
curl -o libatomic_ops-7.6.10.tar.gz https://www.hboehm.info/gc/gc_source/libatomic_ops-7.6.10.tar.gz
tar -xvzf gc-8.0.4.tar.gz -C ./3rdParty/
tar -xvzf libatomic_ops-7.6.10.tar.gz -C ./3rdParty/
cp -a ./3rdParty/libatomic_ops-7.6.10/ ./3rdParty/gc-8.0.4/libatomic_ops/
cp -a ./docs/fix/gc/CMakeLists.txt ./3rdParty/gc-8.0.4/
cp -a ./docs/fix/gc/tests/CMakeLists.txt ./3rdParty/gc-8.0.4/tests/
./scripts/build_gc_debug.sh


