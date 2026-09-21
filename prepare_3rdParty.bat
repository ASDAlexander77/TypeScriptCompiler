@echo off
set BUILD=debug
set TOOL=vs
if not "%1"=="" (
	set BUILD=%1
)

set ARCH=x64
if not "%2"=="" set ARCH=%2

set GC_VER=8.2.12
set LIBATOMIC_OPS_VER=7.10.0

set p=%cd%

IF EXIST ".\3rdParty\llvm\x64\%BUILD%\bin" (
  echo "No need to build LLVM (%BUILD%)"
) ELSE (
  cd %p%
  echo "Downloading LLVM"
  git submodule update --init --recursive
  echo "Configuring LLVM (%BUILD%)"
  cd %p%
  @call scripts\config_llvm_%BUILD%_%TOOL%.bat
  echo "Building LLVM (%BUILD%)"
  cd %p%
  @call scripts\build_llvm_%BUILD%_%TOOL%.bat
)

IF EXIST ".\3rdParty\gc\x64\%BUILD%\lib\gc.lib" (
  echo "No need to build GC (%BUILD%)"
) ELSE (
  cd %p%
  echo "Downloading BDWGC"
  curl -o gc-%GC_VER%.tar.gz -L https://github.com/bdwgc/bdwgc/releases/download/v%GC_VER%/gc-%GC_VER%.tar.gz
  echo "Opening TAR.GZ BDWGC"  
  tar -xvzf gc-%GC_VER%.tar.gz -C ./3rdParty/
  echo "Downloading Libatomic_ops"
  curl -o libatomic_ops-%LIBATOMIC_OPS_VER%.tar.gz -L https://github.com/bdwgc/libatomic_ops/releases/download/v%LIBATOMIC_OPS_VER%/libatomic_ops-%LIBATOMIC_OPS_VER%.tar.gz
  echo "Opening TAR.GZ Libatomic_ops"  
  tar -xvzf libatomic_ops-%LIBATOMIC_OPS_VER%.tar.gz -C ./3rdParty/
  echo "Copy to gc-%GC_VER%/libatomic_ops"  
  xcopy  /E /H /C /I /Y .\3rdParty\libatomic_ops-%LIBATOMIC_OPS_VER%\ .\3rdParty\gc-%GC_VER%\libatomic_ops\
  cd %p%
  @call scripts\build_gc_%BUILD%_%TOOL%.bat
)

rem Boehm as a DLL too: TypeScriptRuntime.dll, the default library's DLL and user shared
rem libraries take the collector from gc.dll so a process has only one.
rem See tslang/docs/single-gc-collector-design.md.
IF EXIST ".\3rdParty\gcdll\x64\%BUILD%\lib\gc.lib" (
  echo "No need to build shared GC (%BUILD%)"
) ELSE (
  cd %p%
  @call scripts\build_gc_%BUILD%_shared_%TOOL%.bat
)

rem x86 (Win32) Boehm, for compiling 32-bit programs (-mtriple=i686-pc-windows-msvc). Opt-in:
rem   prepare_3rdParty.bat release x86
rem Checked where the compiler looks: each x86 script also copies its output into the x86
rem subdirectory of the x64 lib directory (one --gc-lib-path / --gc-shared-lib-path serves both
rem machines). A build from before that copy existed is re-run, which redoes the copy.
IF "%ARCH%"=="x86" (
  IF EXIST ".\3rdParty\gc\x64\%BUILD%\lib\x86\gc.lib" (
    echo "No need to build x86 GC (%BUILD%)"
  ) ELSE (
    cd %p%
    @call scripts\build_gc_%BUILD%_%TOOL%_x86.bat
  )
  IF EXIST ".\3rdParty\gcdll\x64\%BUILD%\lib\x86\gc.dll" (
    echo "No need to build x86 shared GC (%BUILD%)"
  ) ELSE (
    cd %p%
    @call scripts\build_gc_%BUILD%_shared_%TOOL%_x86.bat
  )
)