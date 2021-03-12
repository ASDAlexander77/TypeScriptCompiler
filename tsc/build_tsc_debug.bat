pushd
cd %1..\__build\tsc
cmake --build . --config Debug -j 8
popd