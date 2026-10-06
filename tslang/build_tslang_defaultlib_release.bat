pushd
cd ../../TypeScriptCompilerDefaultLib/
call build.bat

rem Copy the whole staged defaultlib tree so every target's, build's and model's subfolders are
rem preserved: defaultlib\{dll,lib}\<arch>\<vendor>\<os>\<env>\{debug,release}\{gc,rc,none},
rem *.d.ts, generics\
xcopy __build\defaultlib\*.* "../TypeScriptCompiler/__build/tslang/windows-msbuild-2026-release/bin/defaultlib/" /i /e /y

popd
