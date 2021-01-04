pushd
mkdir __build
cd __build
cmake ..\3rdParty\llvm-project\llvm -G "Visual Studio 16 2019" -DLLVM_TARGETS_TO_BUILD="host" -DCMAKE_BUILD_TYPE=Debug -Thost=x64 -DCMAKE_INSTALL_PREFIX=../3rdParty/llvm/debug -DLLVM_INSTALL_UTILS=ON -DLLVM_ENABLE_PROJECTS=clang;clang-tools-extra;compiler-rt;libc;libclc;libcxx;libcxxabi;libunwind;lld;lldb;parallel-libs;polly;pstl
popd
