pushd
mkdir __build\lldb\ninja\release
cd __build\lldb\ninja\release
cmake ..\..\..\..\3rdParty\llvm-project\llvm -G "Ninja" -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=../../../../3rdParty/lldb/release -DLLVM_INSTALL_UTILS=ON -DLLVM_ENABLE_ASSERTIONS=OFF -DLLVM_ENABLE_PLUGINS=ON -DLLVM_ENABLE_PROJECTS="lldb" -DLLVM_ENABLE_EH=ON -DLLVM_ENABLE_RTTI=ON -DLLVM_REQUIRES_RTTI=ON -DLLVM_ENABLE_PIC=ON
popd
