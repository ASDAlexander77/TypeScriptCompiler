#!/bin/sh
cd ../__build/tsc-ninja-release
cmake --build . --target test/regression/check-typescript --config Debug -j 16
set CTEST_OUTPUT_ON_FAILURE=TRUE
cmake --build . --target RUN_TESTS --config Release -j 16

