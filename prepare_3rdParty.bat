@echo off
set BUILD=debug
if not "%1"=="" (
	set BUILD=%1
)

set p=%cd%

IF NOT EXIST ".\3rdParty\llvm\%BUILD%\bin" (
  echo "Configuring LLVM (%BUILD%)"
  cd %p%
  @call scripts\config_llvm_%BUILD%.bat
  echo "Building LLVM (%BUILD%)"
  cd %p%
  @call scripts\build_llvm_%BUILD%.bat
) ELSE (
  echo "No need to build LLVM (%BUILD%)"
)

IF NOT EXIST ".\3rdParty\antlr4\%BUILD%" (
  echo "Configuring ANTLR4 (C++) (%BUILD%)"
  cd %p%
  @call scripts\config_antlr4_%BUILD%.bat
  echo "Building ANTLR (C++) (%BUILD%)"
  cd %p%
  @call scripts\build_antlr4_%BUILD%.bat
) ELSE (
  echo "No need to build ANTLR (C++) (%BUILD%)"
)
