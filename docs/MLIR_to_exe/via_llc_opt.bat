call clean.bat
call config_release.bat
%TSCPATH%\tsc.exe -opt --emit=llvm -nogc C:\temp\%FILENAME%.ts 2>%FILENAME%.ll
%LLVMPATH%\llc.exe -O3 --filetype=obj -o=%FILENAME%.o %FILENAME%.ll
%LLVMPATH%\lld.exe -flavor link %FILENAME%.o /libpath:%LIBPATH% /libpath:%SDKPATH% /libpath:%UCRTPATH% /defaultlib:libcmt.lib libvcruntime.lib
del %FILENAME%.o

echo "RUN:..."
%FILENAME%.exe
