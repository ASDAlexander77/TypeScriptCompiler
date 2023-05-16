call clean.bat
call config_debug.bat
%TSCEXEPATH%\tsc.exe --emit=llvm C:\temp\%FILENAME%.ts -o=%FILENAME%.ll
%TSCEXEPATH%\tsc.exe --emit=jit --shared-libs=%TSCEXEPATH%\TypeScriptRuntime.dll -dump-object-file -object-filename=%FILENAME%.o C:\temp\%FILENAME%.ts
%LLVMPATH%\lld.exe -flavor link %FILENAME%.o /libpath:%LIBPATH% /libpath:%SDKPATH% /libpath:%UCRTPATH% /libpath:%GCLIBPATH% msvcrt.lib ucrt.lib kernel32.lib user32.lib gc-lib.lib
del %FILENAME%.o

echo "RUN:..."
%FILENAME%.exe
