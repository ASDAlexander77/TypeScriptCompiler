set FILENAME=1
call config_release.bat
%LLVMPATH%\mlir-translate.exe --mlir-to-llvmir %FILENAME%.mlir -o=%FILENAME%.ll
%LLVMPATH%\llc.exe -O3 --filetype=obj -o=%FILENAME%.o %FILENAME%.ll
%LLVMPATH%\lld.exe -flavor link %FILENAME%.o /libpath:%LIBPATH% /libpath:%SDKPATH% /libpath:%UCRTPATH% /libpath:%LLVMLIBPATH% /libpath:%GCLIBPATH% /libpath:%TSCLIBPATH% msvcrt.lib ucrt.lib kernel32.lib user32.lib gcmt-lib.lib TypeScriptAsyncRuntime.lib LLVMSupport.lib
del %FILENAME%.o

echo "RUN:..."
%FILENAME%.exe
