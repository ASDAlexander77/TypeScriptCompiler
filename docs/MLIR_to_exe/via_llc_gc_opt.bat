call clean.bat
call config_release.bat
%TSCEXEPATH%\tsc.exe --opt --emit=llvm C:\temp\%FILENAME%.ts -o=%FILENAME%.ll
%LLVMPATH%\llc.exe -O3 --filetype=obj -o=%FILENAME%.o %FILENAME%.ll
%LLVMPATH%\lld.exe -flavor link %FILENAME%.o /libpath:%LIBPATH% /libpath:%SDKPATH% /libpath:%UCRTPATH% /libpath:%GCLIBPATH% msvcrt.lib ucrt.lib kernel32.lib user32.lib gcmt-lib.lib LLVMSupport.lib
del %FILENAME%.o

echo "RUN:..."
%FILENAME%.exe
