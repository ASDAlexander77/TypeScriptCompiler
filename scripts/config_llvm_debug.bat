pushd
mkdir __build\llvm\debug
cd __build\llvm
cmake ..\..\3rdParty\llvm-project\llvm -G "Visual Studio 16 2019" -A x64 -DLLVM_TARGETS_TO_BUILD="host" -DLLVM_EXPERIMENTAL_TARGETS_TO_BUILD=WebAssembly -DCMAKE_BUILD_TYPE=Debug -Thost=x64 -DCMAKE_INSTALL_PREFIX=../../3rdParty/llvm/debug -DLLVM_INSTALL_UTILS=ON -DLLVM_ENABLE_ASSERTIONS=ON -DLLVM_ENABLE_PROJECTS="clang;compiler-rt;lld;mlir" -DLLVM_ENABLE_EH=ON -DLLVM_ENABLE_RTTI=ON -DLLVM_REQUIRES_RTTI=ON -DLLVM_ENABLE_PIC=ON
popd

