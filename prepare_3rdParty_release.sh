#!/bin/bash
echo "Downloading LLVM"
git submodule update --init --recursive
echo "Configuring LLVM (Release)"
./scripts/config_llvm_release.sh
echo "Building LLVM (Release)"
./scripts/build_llvm_release.sh
echo "Building GC (Release)"
curl -o gc-8.2.12.tar.gz https://github.com/bdwgc/bdwgc/releases/download/v8.2.12/gc-8.2.12.tar.gz
curl -o libatomic_ops-7.10.0.tar.gz https://github.com/bdwgc/libatomic_ops/releases/download/v7.10.0/libatomic_ops-7.10.0.tar.gz
tar -xvzf gc-8.2.12.tar.gz -C ./3rdParty/
tar -xvzf libatomic_ops-7.10.0.tar.gz -C ./3rdParty/
cp -a ./3rdParty/libatomic_ops-7.10.0/ ./3rdParty/gc-8.2.12/libatomic_ops/
./scripts/build_gc_release.sh


