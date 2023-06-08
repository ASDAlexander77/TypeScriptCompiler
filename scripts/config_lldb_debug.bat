pushd
mkdir __build\lldb\ninja\debug
cd __build\lldb\ninja\debug
cmake ..\..\..\..\3rdParty\llvm-project\llvm -G "Ninja" -DCMAKE_BUILD_TYPE=Debug -DCMAKE_INSTALL_PREFIX=../../../../3rdParty/lldb/debug -DLLVM_INSTALL_UTILS=ON -DLLVM_ENABLE_ASSERTIONS=ON -DLLVM_ENABLE_PLUGINS=ON -DLLVM_ENABLE_PROJECTS="lldb" -DLLVM_ENABLE_EH=ON -DLLVM_ENABLE_RTTI=ON -DLLVM_REQUIRES_RTTI=ON -DLLVM_ENABLE_PIC=ON
popd

