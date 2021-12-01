pushd
cd __build\lldb
cmake --build . --config Debug --target install -j 8
cmake --install . --config Debug
popd