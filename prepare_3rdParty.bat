@echo off
set BUILD=debug
if not "%1"=="" (
	set BUILD=%1
)

set p=%cd%

IF EXIST ".\3rdParty\llvm\%BUILD%\bin_" (
  echo "No need to build LLVM (%BUILD%)"
) ELSE (
  echo "Configuring LLVM (%BUILD%)"
  cd %p%
  @call scripts\config_llvm_%BUILD%.bat
  echo "Building LLVM (%BUILD%)"
  cd %p%
  @call scripts\build_llvm_%BUILD%.bat
)

IF EXIST ".\3rdParty\antlr4\%BUILD%_" (
  echo "No need to build ANTLR (C++) (%BUILD%)"
) ELSE (
  echo "Configuring ANTLR4 (C++) (%BUILD%)"
  cd %p%
  @call scripts\config_antlr4_%BUILD%.bat
  echo "Building ANTLR (C++) (%BUILD%)"
  cd %p%
  @call scripts\build_antlr4_%BUILD%.bat
)
