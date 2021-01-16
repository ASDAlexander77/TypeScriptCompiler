pushd
cd __build\antlr4
cmake --build . --config Release --target install -j 8
cmake --install . --config Release
popd