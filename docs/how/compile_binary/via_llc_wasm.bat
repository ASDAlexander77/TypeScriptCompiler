set FILENAME=1
del %FILENAME%.exe
del %FILENAME%.ll
set LLVMPATH=C:\dev\TypeScriptCompiler\3rdParty\llvm\release\bin
set TSCPATH=C:\dev\TypeScriptCompiler\__build\tsc\bin
%TSCPATH%\tsc.exe --emit=llvm C:\temp\%FILENAME%.ts 2>%FILENAME%.ll
%LLVMPATH%\llc.exe -mtriple=wasm32-unknown-unknown -O3 --filetype=obj -o=%FILENAME%.o %FILENAME%.ll
%LLVMPATH%\wasm-ld.exe %FILENAME%.o -o %FILENAME%.wasm --no-entry --export-all --allow-undefined
del *.o
