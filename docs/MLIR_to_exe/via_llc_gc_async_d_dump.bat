call clean.bat
call config_debug.bat
%TSCEXEPATH%\tsc.exe --emit=llvm -debug-only=llvm -mlir-disable-threading C:\temp\%FILENAME%.ts -o=%FILENAME%.ll
