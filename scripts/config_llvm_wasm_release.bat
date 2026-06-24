pushd
mkdir __build\llvm\msbuild\wasm\release
cd __build\llvm\msbuild\wasm\release
emcmake cmake ..\..\..\..\..\3rdParty\llvm-project\llvm -G "Visual Studio 18 2026" -A x64 -DLLVM_EXPERIMENTAL_TARGETS_TO_BUILD="WebAssembly" -DCMAKE_BUILD_TYPE=Release -Thost=x64 -DCMAKE_INSTALL_PREFIX=../../../../../3rdParty/llvm/wasm/release -DLLVM_INSTALL_UTILS=ON -DLLVM_ENABLE_ASSERTIONS=ON -DLLVM_ENABLE_PLUGINS=ON -DLLVM_ENABLE_PROJECTS="mlir" -DLLVM_ENABLE_EH=ON -DLLVM_ENABLE_RTTI=ON -DLLVM_REQUIRES_RTTI=ON -DLLVM_ENABLE_PIC=ON
popd

