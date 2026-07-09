pushd
cd ../../TypeScriptCompilerDefaultLib/
call build.bat

rem Copy the whole staged defaultlib tree so per-build subfolders are preserved:
rem defaultlib\dll\{debug,release}, defaultlib\lib\{debug,release}, *.d.ts, generics\
xcopy __build\defaultlib\*.* "../TypeScriptCompiler/__build/tslang/windows-msbuild-2026-release/bin/defaultlib/" /i /e /y

popd
