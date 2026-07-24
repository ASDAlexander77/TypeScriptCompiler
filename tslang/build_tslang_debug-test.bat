pushd
cd ../__build/tslang/msbuild/x64/debug
cmake --build . --config Debug -j 24
set CTEST_OUTPUT_ON_FAILURE=TRUE
set CTEST_PARALLEL_LEVEL=16
cmake --build . --target RUN_TESTS --config Debug -j 24
popd