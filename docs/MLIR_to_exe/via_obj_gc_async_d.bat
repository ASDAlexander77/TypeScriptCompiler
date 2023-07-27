call clean.bat
call config_debug.bat
%TSCEXEPATH%\tsc.exe --emit=obj -di C:\temp\%FILENAME%.ts -o=%FILENAME%.o
%LLVMPATH%\lld.exe -flavor link %FILENAME%.o /DEBUG:FULL /libpath:%LIBPATH% /libpath:%SDKPATH% /libpath:%UCRTPATH% /libpath:%LLVMLIBPATH% /libpath:%GCLIBPATH% /libpath:%TSCLIBPATH% msvcrtd.lib ucrtd.lib kernel32.lib user32.lib gcmt-lib.lib TypeScriptAsyncRuntime.lib LLVMSupport.lib
del %FILENAME%.o

echo "RUN:..."
%FILENAME%.exe
