call clean.bat
call config_debug.bat
rem %TSCEXEPATH%\tsc.exe --emit=mlir-affine C:\temp\%FILENAME%.ts -o=%FILENAME%.mlir
%TSCEXEPATH%\tsc.exe --emit=llvm -di C:\temp\%FILENAME%.ts -o=%FILENAME%.ll
%LLVMPATH%\llc.exe --filetype=obj -o=%FILENAME%.o %FILENAME%.ll
%LLVMPATH%\lld.exe -flavor link %FILENAME%.o  /DEBUG:FULL /libpath:%LIBPATH% /libpath:%SDKPATH% /libpath:%UCRTPATH% /libpath:%GCLIBPATH% msvcrtd.lib ucrtd.lib kernel32.lib user32.lib gcmt-lib.lib
del %FILENAME%.o

echo "RUN:..."
%FILENAME%.exe
