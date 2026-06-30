pushd
cd ../../TypeScriptCompilerDefaultLib/
call build.bat

xcopy dll\*.* "../TypeScriptCompiler/__build/tslang/windows-msbuild-2026-debug/bin/defaultlib/dll/" /i /y
xcopy lib\*.* "../TypeScriptCompiler/__build/tslang/windows-msbuild-2026-debug/bin/defaultlib/lib/" /i /y
xcopy src\*.d.ts "../TypeScriptCompiler/__build/tslang/windows-msbuild-2026-debug/bin/defaultlib/" /i /y

popd
