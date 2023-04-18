call clean.bat
call config_release.bat
%TSCEXEPATH%\tsc.exe --emit=llvm C:\temp\%FILENAME%.ts 2>%FILENAME%.ll
%TSCEXEPATH%\tsc.exe --emit=jit -nogc --shared-libs=%LLVMPATH%\mlir_async_runtime.dll -dump-object-file -object-filename=%FILENAME%.o C:\temp\%FILENAME%.ts
%TSCEXEPATH%\tsc.exe --emit=jit -nogc -dump-object-file -object-filename=%FILENAME%.o C:\temp\%FILENAME%.ts
%LLVMPATH%\lld.exe -flavor link %FILENAME%.o /libpath:%LIBPATH% /libpath:%SDKPATH% /libpath:%UCRTPATH% /defaultlib:libcmt.lib libvcruntime.lib
del %FILENAME%.o

echo "RUN:..."
%FILENAME%.exe
