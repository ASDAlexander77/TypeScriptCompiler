pushd
mkdir __build_mlir
cd __build_mlir
cmake ..\3rdParty\llvm-project\llvm -G "Visual Studio 16 2019" -DLLVM_ENABLE_PROJECTS=mlir -DLLVM_TARGETS_TO_BUILD="host" -DCMAKE_BUILD_TYPE=Release -Thost=x64 -DCMAKE_BUILD_TYPE=Release -DLLVM_ENABLE_ASSERTIONS=ON
popd
