pushd
mkdir ..\__build\tsc
cd ..\__build\tsc
cmake ../../tsc -G "Visual Studio 16 2019" -DCMAKE_BUILD_TYPE=Debug -Thost=x64 -DMLIR_DIR=../3rdParty/llvm/debug/lib/cmake/mlir -DLLVM_EXTERNAL_LIT=../__build/llvm/debug/bin/llvm-lit.py
popd
