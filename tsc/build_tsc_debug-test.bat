pushd
cd ..\__build\tsc
cmake --build . --target test/check-typescript --config Debug -j 8
popd