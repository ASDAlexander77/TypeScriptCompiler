pushd
cd ..\__build\tsc
cmake --build . --target tsc --config Debug -j 8
popd