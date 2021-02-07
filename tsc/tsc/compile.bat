set FILENAME=test_run
set VCPATH=C:/Program Files (x86)/Microsoft Visual Studio/2019/Community/SDK/ScopeCppSDK/vc15/VC/lib
set SDKPATH=C:/Program Files (x86)/Microsoft Visual Studio/2019/Community/SDK/ScopeCppSDK/vc15/SDK/lib
set EXEPATH=C:/dev/TypeScriptCompiler/3rdParty/llvm/debug/bin
set TSCEXEPATH=C:/dev/TypeScriptCompiler/__build/tsc/bin
%TSCEXEPATH%\tsc.exe --emit=mlir-llvm %1 2> %FILENAME%.mlir
%EXEPATH%\mlir-translate.exe --mlir-to-llvmir -o=%FILENAME%.il %FILENAME%.mlir
%EXEPATH%\llc.exe --filetype=obj -o=%FILENAME%.o %FILENAME%.il
%EXEPATH%\lld.exe -flavor link %FILENAME%.o "%VCPATH%\libcmt.lib" "%VCPATH%\libvcruntime.lib" "%SDKPATH%\kernel32.lib" "%SDKPATH%\libucrt.lib" "%SDKPATH%\uuid.lib"
