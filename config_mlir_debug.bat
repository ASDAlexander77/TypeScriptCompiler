pushd
mkdir __build\llvm\debug
mkdir __build\mlir
cd __build\mlir
cmake ..\..\3rdParty\llvm-project\llvm -G "Visual Studio 16 2019" -DLLVM_ENABLE_PROJECTS=mlir -DLLVM_TARGETS_TO_BUILD="host" -Thost=x64 -DCMAKE_BUILD_TYPE=Debug -DLLVM_ENABLE_ASSERTIONS=ON -DCMAKE_INSTALL_PREFIX=../../3rdParty/llvm/debug
popd
