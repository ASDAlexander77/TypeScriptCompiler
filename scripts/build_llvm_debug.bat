pushd
cd __build\llvm\msbuild\x64\debug
cmake --build . --config Debug --target install -j 8
cmake --install . --config Debug
popd