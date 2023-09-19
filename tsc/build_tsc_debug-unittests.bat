pushd
cd ../__build/tsc/msbuild/x64/debug
cmake --build . --target MLIRTypeScriptUnitTests --config Debug -j 16
popd