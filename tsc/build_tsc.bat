pushd
cd __build
cmake --build . --target tsc --config Debug -j 8
popd