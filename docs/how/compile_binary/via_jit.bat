set FILENAME=1
del %FILENAME%.exe
del %FILENAME%.ll
set LLVMPATH=C:\dev\TypeScriptCompiler\3rdParty\llvm\release\bin
set TSCPATH=C:\dev\TypeScriptCompiler\__build\tsc\bin
set LIBPATH="C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\VC\Tools\MSVC\14.29.30037\lib\x64"
set SDKPATH="C:\Program Files (x86)\Windows Kits\10\Lib\10.0.18362.0\um\x64"
set UCRTPATH="C:\Program Files (x86)\Windows Kits\10\Lib\10.0.18362.0\ucrt\x64"
%TSCPATH%\tsc.exe --emit=jit -dump-object-file -object-filename=%FILENAME%.o C:\temp\%FILENAME%.ts
%LLVMPATH%\lld.exe -flavor link %FILENAME%.o /libpath:%LIBPATH% /libpath:%SDKPATH% /libpath:%UCRTPATH% /defaultlib:libcmt.lib libvcruntime.lib
del %FILENAME%.o

echo "RUN:..."
%FILENAME%.exe
