call clean.bat
call config_release.bat
rem %TSCEXEPATH%\tsc.exe --emit=mlir-affine C:\temp\%FILENAME%.ts -o=%FILENAME%.mlir
%TSCEXEPATH%\tsc.exe --emit=llvm C:\temp\%FILENAME%.ts -o=%FILENAME%.ll
%LLVMPATH%\llc.exe --filetype=obj -o=%FILENAME%.o %FILENAME%.ll
%LLVMPATH%\lld.exe -flavor link %FILENAME%.o /libpath:%LIBPATH% /libpath:%SDKPATH% /libpath:%UCRTPATH% /libpath:%GCLIBPATH% msvcrt.lib ucrt.lib kernel32.lib user32.lib gcmt-lib.lib
del %FILENAME%.o

echo "RUN:..."
%FILENAME%.exe
