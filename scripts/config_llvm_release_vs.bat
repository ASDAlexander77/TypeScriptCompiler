pushd
mkdir __build\llvm\msbuild\x64\release
cd __build\llvm\msbuild\x64\release
cmake ..\..\..\..\..\3rdParty\llvm-project\llvm -G "Visual Studio 18 2026" -A x64 %EXTRA_PARAM% -DLLVM_TARGETS_TO_BUILD="host;ARM;AArch64" -DLLVM_EXPERIMENTAL_TARGETS_TO_BUILD="WebAssembly" -DCMAKE_BUILD_TYPE=Release -DCMAKE_MSVC_RUNTIME_LIBRARY=MultiThreaded -Thost=x64 -DCMAKE_INSTALL_PREFIX=../../../../../3rdParty/llvm/x64/release -DLLVM_INSTALL_UTILS=ON -DLLVM_ENABLE_ASSERTIONS=OFF -DLLVM_ENABLE_PLUGINS=ON -DLLVM_ENABLE_PROJECTS="clang;lld;mlir" -DLLVM_ENABLE_EH=ON -DLLVM_ENABLE_RTTI=ON -DLLVM_REQUIRES_RTTI=ON -DLLVM_ENABLE_PIC=ON
popd
