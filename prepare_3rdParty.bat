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
  @call scripts\config_llvm_%BUILD%.bat
  echo "Building LLVM (%BUILD%) 2"
  cd %p%
  @call scripts\build_llvm_%BUILD%.bat
)
