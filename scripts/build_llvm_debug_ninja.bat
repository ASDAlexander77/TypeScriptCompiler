pushd
cd __build\llvm\ninja\debug
cmake --build . --config Debug --target install
cmake --install . --config Debug
popd