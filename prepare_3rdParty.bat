@echo off
set BUILD=debug
set TOOL=vs
if not "%1"=="" (
	set BUILD=%1
)

set p=%cd%

IF EXIST ".\3rdParty\llvm\x64\%BUILD%\bin" (
  echo "No need to build LLVM (%BUILD%)"
) ELSE (
  cd %p%
  echo "Downloading LLVM"
  git submodule update --init --recursive --progress
  rem copy /Y .\docs\fix\AddCompilerRT.cmake .\3rdParty\llvm-project\compiler-rt\cmake\Modules\
  echo "Configuring LLVM (%BUILD%)"
  cd %p%
  @call scripts\config_llvm_%BUILD%_%TOOL%.bat
  echo "Building LLVM (%BUILD%)"
  cd %p%
  @call scripts\build_llvm_%BUILD%_%TOOL%.bat
)

IF EXIST ".\3rdParty\gc\x64\%BUILD%\gc-lib.lib" (
  echo "No need to build GC (%BUILD%)"
) ELSE (
  cd %p%
  echo "Downloading BDWGC"
  curl -o gc-8.0.4.tar.gz https://www.hboehm.info/gc/gc_source/gc-8.0.4.tar.gz
  echo "Opening TAR.GZ BDWGC"  
  tar -xvzf gc-8.0.4.tar.gz -C ./3rdParty/
  echo "Downloading Libatomic_ops"
  curl -o libatomic_ops-7.6.10.tar.gz https://www.hboehm.info/gc/gc_source/libatomic_ops-7.6.10.tar.gz
  echo "Opening TAR.GZ Libatomic_ops"  
  tar -xvzf libatomic_ops-7.6.10.tar.gz -C ./3rdParty/
  echo "Copy to gc-<ver>/libatomic_ops"  
  xcopy  /E /H /C /I /Y .\3rdParty\libatomic_ops-7.6.10\ .\3rdParty\gc-8.0.4\libatomic_ops\
  echo "Copy fixes"  
  xcopy  /E /H /C /I /Y .\docs\fix\gc\ .\3rdParty\gc-8.0.4\
  cd %p%
  @call scripts\build_gc_%BUILD%_%TOOL%.bat
  cd %p%
  if "%BUILD%"=="debug" ( xcopy  /E /H /C /I /Y .\__build\gc\msbuild\x64\%BUILD%\Debug\ .\3rdParty\gc\x64\debug\ )
  if "%BUILD%"=="release" ( xcopy  /E /H /C /I /Y .\__build\gc\msbuild\x64\%BUILD%\Release\ .\3rdParty\gc\x64\release\ )
)