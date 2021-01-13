pushd
cd ..\__build\tsc
cmake --build . --config Debug -j 8 -t MLIRTypeScript
popd