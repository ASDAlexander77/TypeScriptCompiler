pushd
mkdir __build\llvm\release
cd __build\llvm
cmake ..\..\3rdParty\llvm-project\llvm -G "Visual Studio 16 2019" -DLLVM_TARGETS_TO_BUILD="host" -DCMAKE_BUILD_TYPE=Release -Thost=x64 -DCMAKE_INSTALL_PREFIX=../../3rdParty/llvm/release -DLLVM_INSTALL_UTILS=ON -DLLVM_ENABLE_ASSERTIONS=OFF -DLLVM_ENABLE_PROJECTS=clang;clang-tools-extra;compiler-rt;libc;libclc;libcxx;libcxxabi;libunwind;lld;lldb;mlir;parallel-libs;polly;pstl
popd
