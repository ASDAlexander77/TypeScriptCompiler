pushd
mkdir "__build/debug-ninja"
cd "__build/debug-ninja"
cmake ../.. -G "Ninja" -DCMAKE_BUILD_TYPE=Debug -Wno-dev -DCMAKE_TOOLCHAIN_FILE=../../llvm_tool.cmake
cmake --build . --config Debug -j 1 -trace-source=..\..\CMakeLists.txt
popd
