pushd
cd ../__build/tsc/msbuild/x64/debug
set CTEST_OUTPUT_ON_FAILURE=TRUE
set CTEST_PARALLEL_LEVEL=16
cmake --build . --target RUN_TESTS --config Debug --parallel
popd