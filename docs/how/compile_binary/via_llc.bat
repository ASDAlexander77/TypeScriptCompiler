del out.exe
del 1.ll
set LLVMPATH=C:\dev\TypeScriptCompiler\3rdParty\llvm\debug\bin
set TSCPATH=C:\dev\TypeScriptCompiler\__build\tsc\bin
set VCFLD=C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\SDK\ScopeCppSDK\vc15
set LIBPATH=%VCFLD%\VC\lib
set SDKPATH=%VCFLD%\SDK\lib
set LLVMLIBPATH=C:\dev\TypeScriptCompiler\3rdParty\llvm\debug\lib
rem set CLANGLIBPATH=C:\dev\TypeScriptCompiler\3rdParty\llvm\debug\lib\clang\13.0.0\lib\windows
%TSCPATH%\tsc.exe --emit=llvm C:\temp\1.ts 2>1.ll
%LLVMPATH%\llc.exe --filetype=obj -o=out.o 1.ll
rem %LLVMPATH%\lld.exe -flavor link out.o /libpath:"%LIBPATH%" /libpath:"%SDKPATH%" /libpath:"%LLVMLIBPATH%" /libpath:"%CLANGLIBPATH%" "%LIBPATH%\libcmt.lib" "%LIBPATH%\libvcruntime.lib" "%SDKPATH%\kernel32.lib" "%SDKPATH%\libucrt.lib" "%SDKPATH%\uuid.lib" "%CLANGLIBPATH%\clang_rt.builtins-x86_64.lib"
%LLVMPATH%\lld.exe -flavor link out.o /libpath:"%LIBPATH%" /libpath:"%SDKPATH%" /libpath:"%LLVMLIBPATH%" /libpath:"%CLANGLIBPATH%" "%LIBPATH%\libcmt.lib" "%LIBPATH%\libvcruntime.lib" "%SDKPATH%\kernel32.lib" "%SDKPATH%\libucrt.lib" "%SDKPATH%\uuid.lib"
del *.o

echo "RUN:..."
out.exe
