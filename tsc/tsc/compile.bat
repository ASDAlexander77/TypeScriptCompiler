set FILENAME=%1
set VCPATH=C:/Program Files (x86)/Microsoft Visual Studio/2019/Community/SDK/ScopeCppSDK/vc15/VC/lib
set SDKPATH=C:/Program Files (x86)/Microsoft Visual Studio/2019/Community/SDK/ScopeCppSDK/vc15/SDK/lib
set EXEPATH=C:/dev/TypeScriptCompiler/3rdParty/llvm/debug/bin
set TSCEXEPATH=C:/dev/TypeScriptCompiler/__build/tsc/bin
%TSCEXEPATH%\tsc.exe --emit=llvm %2 2> %FILENAME%.il
%EXEPATH%\llc.exe --filetype=obj -o=%FILENAME%.o %FILENAME%.il
%EXEPATH%\lld.exe -flavor link %FILENAME%.o "%VCPATH%\libcmt.lib" "%VCPATH%\libvcruntime.lib" "%SDKPATH%\kernel32.lib" "%SDKPATH%\libucrt.lib" "%SDKPATH%\uuid.lib"
del %FILENAME%.il
del %FILENAME%.o
