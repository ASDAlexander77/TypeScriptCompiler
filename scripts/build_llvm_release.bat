pushd
cd __build\llvm\msbuild\x64\release
cmake --build . --config Release --target install -j 8
cmake --install . --config Release
popd