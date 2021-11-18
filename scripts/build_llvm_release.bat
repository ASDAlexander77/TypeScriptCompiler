pushd
cd __build\llvm-release
cmake --build . --config Release --target install -j 8
cmake --install . --config Release
popd