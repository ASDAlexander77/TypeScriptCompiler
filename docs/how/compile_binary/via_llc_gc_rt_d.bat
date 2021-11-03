set FILENAME=1
del %FILENAME%.exe
rem del %FILENAME%.ll
set LLVMPATH=C:\dev\TypeScriptCompiler\3rdParty\llvm\release\bin
set TSCPATH=C:\dev\TypeScriptCompiler\__build\tsc\bin
set LIBPATH="C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\VC\Tools\MSVC\14.29.30133\lib\x64"
set SDKPATH="C:\Program Files (x86)\Windows Kits\10\Lib\10.0.18362.0\um\x64"
set UCRTPATH="C:\Program Files (x86)\Windows Kits\10\Lib\10.0.18362.0\ucrt\x64"
set LLVMLIBPATH=C:\dev\TypeScriptCompiler\3rdParty\llvm\release\lib
set CLANGLIBPATH=C:\dev\TypeScriptCompiler\3rdParty\llvm\release\lib\clang\13.0.0\lib\windows
set GCLIBPATH=C:\dev\TypeScriptCompiler\3rdParty\gc\Release
%TSCPATH%\tsc.exe --emit=llvm C:\temp\%FILENAME%.ts 2>%FILENAME%.ll
%LLVMPATH%\llc.exe -O0 --debug-entry-values --debugger-tune=lldb --xcoff-traceback-table --debugify-level=location+variables --filetype=obj -o=%FILENAME%.o %FILENAME%.ll
%LLVMPATH%\lld.exe -flavor link /debug %FILENAME%.o /libpath:%LIBPATH% /libpath:%SDKPATH% /libpath:%UCRTPATH% /libpath:%CLANGLIBPATH% /libpath:%GCLIBPATH% msvcrt.lib ucrt.lib kernel32.lib user32.lib gcmt-lib.lib clang_rt.builtins-x86_64.lib
del %FILENAME%.o

echo "RUN:..."
%FILENAME%.exe
