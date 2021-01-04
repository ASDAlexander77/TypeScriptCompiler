pushd
cd __build
cmake --build . --config Release --target install -j 8
cmake --install . --config Release
popd