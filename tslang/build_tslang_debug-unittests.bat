pushd
cd ../__build/tslang/msbuild/x64/debug
cmake --build . --target MLIRTypeScriptUnitTests --config Debug -j 16
popd