#!/bin/bash
echo "Downloading LLVM"
git submodule update --init --recursive
echo "Configuring LLVM (Release)"
./scripts/config_llvm_release.sh
echo "Building LLVM (Release)"
./scripts/build_llvm_release.sh
echo "Building GC (Release)"
curl -o gc-8.0.4.tar.gz https://www.hboehm.info/gc/gc_source/gc-8.0.4.tar.gz
curl -o libatomic_ops-7.6.10.tar.gz https://www.hboehm.info/gc/gc_source/libatomic_ops-7.6.10.tar.gz
tar -xvzf gc-8.0.4.tar.gz -C ./3rdParty/
tar -xvzf libatomic_ops-7.6.10.tar.gz -C ./3rdParty/
cp -a ./3rdParty/libatomic_ops-7.6.10/ ./3rdParty/gc-8.0.4/libatomic_ops/
cp -ar ./docs/fix/gc/* ./3rdParty/gc-8.0.4/
./scripts/build_gc_release.sh


