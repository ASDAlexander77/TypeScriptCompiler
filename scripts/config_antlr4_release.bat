pushd
mkdir __build\antlr4\release
cd __build\antlr4
cmake ..\..\3rdParty\antlr\runtime\Cpp\ -G "Visual Studio 16 2019" -A x64 -DCMAKE_BUILD_TYPE=Release -Thost=x64 -DCMAKE_INSTALL_PREFIX=../../3rdParty/antlr4/release -DANTLR4_INSTALL=ON -DWITH_STATIC_CRT=OFF
popd
