pushd
cd __build\mlir
cmake --build . --config Debug --target install -j 8
cmake --install . --config Debug
popd