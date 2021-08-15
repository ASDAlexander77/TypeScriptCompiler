set FILENAME=%1
del %FILENAME%.exe
del %FILENAME%.ll
set LLVMPATH=C:\TypeScriptCompiler\3rdParty\llvm\release\bin
set TSCPATH=C:\TypeScriptCompiler\__build\tsc\bin
set LIBPATH="C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\VC\Tools\MSVC\14.29.30037\lib\x64"
set SDKPATH="C:\Program Files (x86)\Windows Kits\10\Lib\10.0.18362.0\um\x64"
set UCRTPATH="C:\Program Files (x86)\Windows Kits\10\Lib\10.0.18362.0\ucrt\x64"
set CLANGLIBPATH=C:\TypeScriptCompiler\3rdParty\llvm\release\lib\clang\13.0.0\lib\windows
%TSCPATH%\tsc.exe --emit=llvm %FILENAME%.ts 2>%FILENAME%.ll
%LLVMPATH%\llc.exe --filetype=obj -o=%FILENAME%.o %FILENAME%.ll
%LLVMPATH%\lld.exe -flavor link %FILENAME%.o /libpath:%LIBPATH% /libpath:%SDKPATH% /libpath:%UCRTPATH% /libpath:%CLANGLIBPATH% /defaultlib:libcmt.lib libvcruntime.lib clang_rt.builtins-x86_64.lib
del *.o

echo "RUN:..."
%FILENAME%.exe
