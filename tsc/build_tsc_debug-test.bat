pushd
cd ..\__build\tsc
cmake --build . --target test/regression/check-typescript --config Debug -j 8
cmake --build . --target RUN_TESTS --config Debug -j 8
popd