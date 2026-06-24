pushd
cd ../../TypeScriptCompilerDefaultLib/
call build.bat

xcopy dll\*.* "../TypeScriptCompiler/__build/tsc/windows-msbuild-2026-release/bin/defaultlib/dll/" /i /y
xcopy lib\*.* "../TypeScriptCompiler/__build/tsc/windows-msbuild-2026-release/bin/defaultlib/lib/" /i /y
xcopy src\*.d.ts "../TypeScriptCompiler/__build/tsc/windows-msbuild-2026-release/bin/defaultlib/" /i /y

popd
