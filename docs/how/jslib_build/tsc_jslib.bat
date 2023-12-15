call jslib\clean.bat

set GC_LIB_PATH=C:\dev\TypeScriptCompiler\__build\gc\msbuild\x64\debug\Debug
set LLVM_LIB_PATH=C:\dev\TypeScriptCompiler\__build\llvm\msbuild\x64\debug\Debug\lib
set TSC_LIB_PATH=C:\dev\TypeScriptCompiler\__build\tsc\windows-msbuild-debug\lib
C:\dev\TypeScriptCompiler\__build\tsc\windows-msbuild-debug\bin\tsc.exe --emit=dll C:\temp\jslib\lib.ts -o C:\temp\jslib\dll\lib.dll

C:\dev\TypeScriptCompiler\__build\tsc\windows-msbuild-debug\bin\tsc.exe --emit=obj --export=none C:\temp\jslib\lib.ts -o C:\temp\jslib\lib\lib.obj
C:\dev\TypeScriptCompiler\3rdParty\llvm\x64\release\bin\llvm-lib.exe jslib\lib\lib.obj
