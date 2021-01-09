pushd
mkdir __build\mlir\debug
cd __build\mlir
cmake ..\..\3rdParty\llvm-project\llvm -G "Visual Studio 16 2019" -DLLVM_ENABLE_PROJECTS=mlir -DLLVM_BUILD_EXAMPLES=ON -DLLVM_TARGETS_TO_BUILD="host" -Thost=x64  -DCMAKE_INSTALL_PREFIX=../../3rdParty/mlir/debug -DCMAKE_BUILD_TYPE=Debug -DLLVM_ENABLE_ASSERTIONS=ON
popd
