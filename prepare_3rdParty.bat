@echo off
set BUILD=debug
if not "%1"=="" (
	set BUILD=%1
)

set p=%cd%

IF EXIST ".\3rdParty\llvm\%BUILD%\bin" (
  echo "No need to build LLVM (%BUILD%)"
) ELSE (
  echo "Downloading LLVM"
  git submodule update --init --recursive
  copy /Y .\docs\fix\AddCompilerRT.cmake .\3rdParty\llvm-project\compiler-rt\cmake\Modules\
  echo "Configuring LLVM (%BUILD%)"
  cd %p%
  @call scripts\config_llvm_%BUILD%.bat
  echo "Building LLVM (%BUILD%)"
  cd %p%
  @call scripts\build_llvm_%BUILD%.bat
)

IF EXIST ".\3rdParty\gc\%BUILD%\gc-lib.lib" (
  echo "No need to build GC (%BUILD%)"
) ELSE (
  echo "Downloading BDWGC"
  curl -o gc-8.0.4.tar.gz https://www.hboehm.info/gc/gc_source/gc-8.0.4.tar.gz
  echo "Opening TAR.GZ BDWGC"  
  tar -xvzf gc-8.0.4.tar.gz -C ./3rdParty/
  echo "Downloading Libatomic_ops"
  curl -o libatomic_ops-7.6.10.tar.gz https://www.hboehm.info/gc/gc_source/libatomic_ops-7.6.10.tar.gz
  echo "Opening TAR.GZ Libatomic_ops"  
  tar -xvzf libatomic_ops-7.6.10.tar.gz -C ./3rdParty/
  echo "Copy to  gc-<ver>/libatomic_ops"  
  xcopy  /E /H /C /I /Y .\3rdParty\libatomic_ops-7.6.10\ .\3rdParty\gc-8.0.4\libatomic_ops\
  cd %p%
  @call scripts\build_gc_%BUILD%.bat
  cd %p%
  if "%BUILD%"=="debug" ( xcopy  /E /H /C /I /Y .\__build\gc\%BUILD%\ .\3rdParty\gc\%BUILD%\ )
  if "%BUILD%"=="release" ( xcopy  /E /H /C /I /Y .\__build\gc-release\%BUILD%\ .\3rdParty\gc\%BUILD%\ )
)