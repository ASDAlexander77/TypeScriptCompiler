#!/bin/bash
echo "Configuring LLVM (Debug) (WASM)"
./scripts/config_llvm_wasm_debug.sh
echo "Building LLVM (Debug) (WASM)"
./scripts/build_llvm_wasm_debug.sh
