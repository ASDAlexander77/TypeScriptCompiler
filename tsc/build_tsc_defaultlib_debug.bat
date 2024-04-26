pushd
cd ../../TypeScriptCompilerDefaultLib/
call build.bat

xcopy dll\*.* "../TypeScriptCompiler/__build/tsc/windows-msbuild-debug/bin/defaultlib/dll/" /i /y
xcopy lib\*.* "../TypeScriptCompiler/__build/tsc/windows-msbuild-debug/bin/defaultlib/lib/" /i /y
xcopy src\*.d.ts "../TypeScriptCompiler/__build/tsc/windows-msbuild-debug/bin/defaultlib/" /i /y

popd
