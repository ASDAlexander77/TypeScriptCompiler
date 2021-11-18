pushd
mkdir __build\llvm-release\release
cd __build\llvm-release
cmake ..\..\3rdParty\llvm-project\llvm -G "Visual Studio 16 2019" -A x64 -DLLVM_TARGETS_TO_BUILD="host" -DCMAKE_BUILD_TYPE=Release -Thost=x64 -DCMAKE_INSTALL_PREFIX=../../3rdParty/llvm-release/release -DLLVM_INSTALL_UTILS=ON -DLLVM_ENABLE_ASSERTIONS=OFF -DLLVM_ENABLE_PLUGINS=ON -DLLVM_ENABLE_PROJECTS="clang;compiler-rt;lld;mlir" -DLLVM_ENABLE_EH=ON -DLLVM_ENABLE_RTTI=ON -DLLVM_REQUIRES_RTTI=ON -DLLVM_ENABLE_PIC=ON
popd
