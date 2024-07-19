pushd
mkdir __build\llvm\msbuild\x64\debug
cd __build\llvm\msbuild\x64\debug
if exist "C:/Program Files/Microsoft Visual Studio/2022/Professional" set EXTRA_PARAM=-DCMAKE_GENERATOR_INSTANCE="C:/Program Files/Microsoft Visual Studio/2022/Professional"
cmake ..\..\..\..\..\3rdParty\llvm-project\llvm -G "Visual Studio 17 2022" -A x64 %EXTRA_PARAM% -DLLVM_TARGETS_TO_BUILD="host;ARM;AArch64" -DLLVM_EXPERIMENTAL_TARGETS_TO_BUILD="WebAssembly" -DCMAKE_BUILD_TYPE=Debug -T="v142" -DCMAKE_INSTALL_PREFIX=../../../../../3rdParty/llvm/x64/debug -DLLVM_INSTALL_UTILS=ON -DLLVM_ENABLE_ASSERTIONS=ON -DLLVM_ENABLE_PLUGINS=ON -DLLVM_ENABLE_PROJECTS="clang;lld;mlir" -DLLVM_ENABLE_EH=ON -DLLVM_ENABLE_RTTI=ON -DLLVM_REQUIRES_RTTI=ON -DLLVM_ENABLE_PIC=ON
popd

