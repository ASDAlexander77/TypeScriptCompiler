call clean.bat
call config_release.bat
%TSCEXEPATH%\tsc.exe -nogc --emit=obj --opt --mtriple=wasm32-unknown-unknown C:\temp\%FILENAME%.ts -o=%FILENAME%.o
%LLVMPATH%\wasm-ld.exe %FILENAME%.o -o %FILENAME%.wasm --no-entry --export-all --allow-undefined
del *.o
