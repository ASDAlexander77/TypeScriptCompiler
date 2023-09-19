pushd
cd __build\lldb\ninja\debug
cmake --build . --config Debug --target install -j 8
cmake --install . --config Debug
popd