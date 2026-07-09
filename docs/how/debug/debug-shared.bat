@echo off
rem ============================================================================
rem  Repro + lldb launcher for the shared-component (test-compile-shared-component)
rem  loader crash. Builds shared.dll + use_shared.exe with the STATIC-CRT link
rem  line (matching the CRT fix) and KEEPS them, then opens lldb stopped ready to
rem  hit the access violation.
rem
rem  Usage:   debug-shared.bat            (builds, then launches lldb)
rem           debug-shared.bat build      (build only, no debugger)
rem           debug-shared.bat run        (just run use_shared.exe, show exit code)
rem ============================================================================
setlocal

rem --- Python 3.10 so lldb can load (needs python310.dll + a clean stdlib path) ---
set "PY310=C:\Users\duzha\AppData\Local\Python\pythoncore-3.10-64"
set "PATH=%PY310%;%PATH%"
set "PYTHONHOME=%PY310%"
set "PYTHONPATH="

rem --- toolchain / lib paths (from the test harness compiled.bat) ---
set "WORK=%~dp0"
set "TESTS=I:\TypeScriptCompiler\tslang\test\tester\tests"
set "tslangEXEPATH=I:\TypeScriptCompiler\__build\tslang\msbuild\x64\debug\bin"
set "tslang_LIB_PATH=I:\TypeScriptCompiler\__build\tslang\msbuild\x64\debug\lib"
set "LLVMEXEPATH=I:\TypeScriptCompiler\3rdParty\llvm\x64\debug\bin"
set "LLVM_LIB_PATH=I:\TypeScriptCompiler\3rdParty\llvm\x64\debug\lib"
set "GC_LIB_PATH=I:\TypeScriptCompiler\3rdParty\gc\x64\debug"
set "LIBPATH=C:\Program Files\Microsoft Visual Studio\18\Professional\VC\Tools\MSVC\14.51.36231\lib\x64"
set "SDKPATH=C:\Program Files (x86)\Windows Kits\10\Lib\10.0.26100.0\um\x64"
set "UCRTPATH=C:\Program Files (x86)\Windows Kits\10\Lib\10.0.26100.0\ucrt\x64"

rem --- STATIC CRT link line (matches the test-runner.cpp fix) ---
set "LIBS=libcmtd.lib libvcruntimed.lib libucrtd.lib ntdll.lib TypeScriptAsyncRuntime.lib gc.lib LLVMSupport.lib kernel32.lib user32.lib gdi32.lib winspool.lib shell32.lib ole32.lib oleaut32.lib uuid.lib comdlg32.lib advapi32.lib"
set "LIBPATHS=/libpath:"%GC_LIB_PATH%" /libpath:"%LLVM_LIB_PATH%" /libpath:"%tslang_LIB_PATH%" /libpath:"%LIBPATH%" /libpath:"%SDKPATH%" /libpath:"%UCRTPATH%""

cd /d "%WORK%"

if /i "%~1"=="run" goto run

rem NOTE: WinDbg/cdb need CodeView, NOT DWARF -- do NOT pass --lldb here (that
rem emits DWARF and cdb can't read it). /DEBUG on the link produces the .pdb.
echo === [1/4] compile shared.ts -^> shared.obj ===
"%tslangEXEPATH%\tslang.exe" --emit=obj --di --opt_level=0 "%TESTS%\shared.ts" -o=shared.obj || goto err

echo === [2/4] link shared.dll ===
"%LLVMEXEPATH%\lld.exe" -flavor link /out:shared.dll /DLL /DEBUG shared.obj %LIBS% %LIBPATHS% || goto err

echo === [3/4] compile use_shared.ts -^> use_shared.obj ===
"%tslangEXEPATH%\tslang.exe" --emit=obj --di --opt_level=0 "%TESTS%\use_shared.ts" -o=use_shared.obj || goto err

echo === [4/4] link use_shared.exe ===
"%LLVMEXEPATH%\lld.exe" -flavor link /out:use_shared.exe /DEBUG use_shared.obj %LIBS% %LIBPATHS% || goto err

echo.
echo Built: %WORK%use_shared.exe  (+ shared.dll)
echo.

if /i "%~1"=="build" goto done

:debug
echo === launching cdb (Microsoft symbols; stops on the access violation) ===
echo     it will auto: continue, show regs, stack, then drop to interactive prompt
echo.
set "_NT_SYMBOL_PATH=srv*C:\symbols*https://msdl.microsoft.com/download/symbols"
"C:\Users\duzha\AppData\Local\Microsoft\WindowsApps\cdbX64.exe" -c "g; .echo ===FAULT===; r; .echo ===STACK===; kn 40" "%WORK%use_shared.exe"
goto done

:run
echo === running use_shared.exe ===
"%WORK%use_shared.exe"
echo EXITCODE=%ERRORLEVEL%
goto done

:err
echo *** BUILD FAILED (errorlevel %ERRORLEVEL%) ***
exit /b 1

:done
endlocal
