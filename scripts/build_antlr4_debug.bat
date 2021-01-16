pushd
cd __build\antlr4
cmake --build . --config Debug --target install -j 8
cmake --install . --config Debug
popd