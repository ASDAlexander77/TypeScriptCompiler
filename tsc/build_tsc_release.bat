pushd
cd ../__build/tsc/msbuild/x64/release
cmake --build . --config Release -j 24
popd