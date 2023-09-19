call clean.bat
call config_debug.bat
%TSCEXEPATH%\tsc.exe --emit=llvm -di -nogc C:\temp\%FILENAME%.ts -o=%FILENAME%.ll
%LLVMPATH%\llc.exe -O0 --experimental-debug-variable-locations --debug-entry-values --debugger-tune=lldb --xcoff-traceback-table --debugify-level=location+variables --filetype=obj -o=%FILENAME%.o %FILENAME%.ll
%LLVMPATH%\lld.exe -flavor link /DEBUG:FULL %FILENAME%.o /libpath:%LIBPATH% /libpath:%SDKPATH% /libpath:%UCRTPATH% /defaultlib:libcmt.lib libvcruntime.lib
del %FILENAME%.o

echo "RUN:..."
%FILENAME%.exe
