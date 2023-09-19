set GC_LIB_PATH=C:\dev\TypeScriptCompiler\__build\gc\msbuild\x64\debug\Debug
set LLVM_LIB_PATH=C:\dev\TypeScriptCompiler\__build\llvm\msbuild\x64\debug\Debug\lib
set TSC_LIB_PATH=C:\dev\TypeScriptCompiler\__build\tsc\windows-msbuild-debug\lib
C:\dev\TypeScriptCompiler\__build\tsc\windows-msbuild-debug\bin\tsc.exe --emit=exe --nogc --di -mtriple=wasm32-unknown-unknown C:\temp\1.ts
copy 1.wasm C:\temp\webassembly3\hello.wasm