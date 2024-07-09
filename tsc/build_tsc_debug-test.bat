pushd
cd ../__build/tsc/windows-msbuild-debug
set CTEST_OUTPUT_ON_FAILURE=TRUE
cmake --build . --target RUN_TESTS --config Debug -j 16
popd