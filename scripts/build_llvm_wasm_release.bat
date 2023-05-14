pushd
cd __build\llvm-wasm
cmake --build . --config Release --target install -j 8
cmake --install . --config Release
popd