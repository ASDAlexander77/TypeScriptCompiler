call clean.bat
call config_release.bat
%TSCEXEPATH%\tsc.exe --opt --emit=obj C:\temp\%FILENAME%.ts -o=%FILENAME%.obj
"%CLANGPATH%\clang-cl.exe" %FILENAME%.obj /link /libpath:%LIBPATH% /libpath:%SDKPATH% /libpath:%UCRTPATH% /libpath:%LLVMLIBPATH% /libpath:%GCLIBPATH% /libpath:%TSCLIBPATH% msvcrt.lib ucrt.lib kernel32.lib user32.lib gcmt-lib.lib TypeScriptAsyncRuntime.lib LLVMSupport.lib

rem del %FILENAME%.o

echo "RUN:..."
%FILENAME%.exe
