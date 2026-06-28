pushd
mkdir __build\llvm\msbuild\x64\debug
cd __build\llvm\msbuild\x64\debug
cmake ..\..\..\..\..\3rdParty\llvm-project\llvm -G "Visual Studio 18 2026" -A x64 %EXTRA_PARAM% -DLLVM_TARGETS_TO_BUILD="host;ARM;AArch64" -DLLVM_EXPERIMENTAL_TARGETS_TO_BUILD="WebAssembly" -DCMAKE_BUILD_TYPE=Debug -T="ClangCL" -DCMAKE_INSTALL_PREFIX=../../../../../3rdParty/llvm/x64/debug -DLLVM_INSTALL_UTILS=ON -DLLVM_ENABLE_ASSERTIONS=ON -DLLVM_ENABLE_PLUGINS=ON -DLLVM_ENABLE_PROJECTS="clang;lld;mlir" -DLLVM_ENABLE_EH=ON -DLLVM_ENABLE_RTTI=ON -DLLVM_REQUIRES_RTTI=ON -DLLVM_ENABLE_PIC=ON
popd

