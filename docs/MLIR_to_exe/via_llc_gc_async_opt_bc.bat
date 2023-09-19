call clean.bat
call config_release.bat
%TSCEXEPATH%\tsc.exe --opt --emit=bc C:\temp\%FILENAME%.ts -o=%FILENAME%.bc
rem %TSCEXEPATH%\tsc.exe --opt --emit=mlir-affine C:\temp\%FILENAME%.ts -o=%FILENAME%.mlir
%LLVMPATH%\llc.exe -O3 --filetype=obj -o=%FILENAME%.o %FILENAME%.bc
%LLVMPATH%\lld.exe -flavor link %FILENAME%.o /libpath:%LIBPATH% /libpath:%SDKPATH% /libpath:%UCRTPATH% /libpath:%LLVMLIBPATH% /libpath:%GCLIBPATH% /libpath:%TSCLIBPATH% msvcrt.lib ucrt.lib kernel32.lib user32.lib gcmt-lib.lib TypeScriptAsyncRuntime.lib LLVMSupport.lib
rem del %FILENAME%.o

echo "RUN:..."
%FILENAME%.exe
