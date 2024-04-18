pushd
cd ../../TypeScriptCompilerDefaultLib/
call build.bat

xcopy dll\*.* "../TypeScriptCompiler/__build/tsc/msbuild/x64/debug/defaultlib/dll/" /i /y
xcopy lib\*.* "../TypeScriptCompiler/__build/tsc/msbuild/x64/debug/defaultlib/lib/" /i /y
xcopy src\*.d.ts "../TypeScriptCompiler/__build/tsc/msbuild/x64/debug/defaultlib/" /i /y

popd
