set GC_LIB_PATH=C:\dev\TypeScriptCompiler\__build\gc\msbuild\x64\debug\Debug
set LLVM_LIB_PATH=C:\dev\TypeScriptCompiler\__build\llvm\msbuild\x64\debug\Debug\lib
set TSC_LIB_PATH=C:\dev\TypeScriptCompiler\__build\tsc\windows-msbuild-debug\lib
set EMSDK_SYSROOT_PATH=C:\utils\emsdk\upstream\emscripten\cache\sysroot
C:\dev\TypeScriptCompiler\__build\tsc\windows-msbuild-debug\bin\tsc.exe --emit=exe --nogc -mtriple=wasm32-pc-emscripten C:\temp\1.ts
copy 1.wasm C:\temp\webassembly3\hello.wasm