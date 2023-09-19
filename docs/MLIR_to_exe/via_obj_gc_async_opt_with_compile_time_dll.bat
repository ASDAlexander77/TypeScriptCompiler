call clean.bat
call config_release.bat
set FILENAME=1
%TSCEXEPATH%\tsc.exe --opt --emit=llvm C:\temp\%FILENAME%.ts -o=%FILENAME%.ll
%TSCEXEPATH%\tsc.exe --opt --emit=obj C:\temp\%FILENAME%.ts -o=%FILENAME%.obj
%LLVMPATH%\lld.exe -flavor link %FILENAME%.obj dll.lib /libpath:%LIBPATH% /libpath:%SDKPATH% /libpath:%UCRTPATH% /libpath:%LLVMLIBPATH% /libpath:%GCLIBPATH% /libpath:%TSCLIBPATH% msvcrt.lib ucrt.lib kernel32.lib user32.lib gcmt-lib.lib TypeScriptAsyncRuntime.lib LLVMSupport.lib 

echo "RUN:..."
%FILENAME%.exe
