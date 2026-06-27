@echo off
set BUILD=debug
set TOOL=vs
if not "%1"=="" (
	set BUILD=%1
)

set GC_VER=8.2.8
set LIBATOMIC_OPS_VER=7.8.2

set p=%cd%

IF EXIST ".\3rdParty\llvm\x64\%BUILD%\bin" (
  echo "No need to build LLVM (%BUILD%)"
) ELSE (
  cd %p%
  echo "Downloading LLVM"
  git submodule update --init --recursive
  rem copy /Y .\docs\fix\AddCompilerRT.cmake .\3rdParty\llvm-project\compiler-rt\cmake\Modules\
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
  curl -o gc-%GC_VER%.tar.gz https://www.hboehm.info/gc/gc_source/gc-%GC_VER%.tar.gz
  echo "Opening TAR.GZ BDWGC"  
  tar -xvzf gc-%GC_VER%.tar.gz -C ./3rdParty/
  echo "Downloading Libatomic_ops"
  curl -o libatomic_ops-%LIBATOMIC_OPS_VER%.tar.gz https://www.hboehm.info/gc/gc_source/libatomic_ops-%LIBATOMIC_OPS_VER%.tar.gz
  echo "Opening TAR.GZ Libatomic_ops"  
  tar -xvzf libatomic_ops-%LIBATOMIC_OPS_VER%.tar.gz -C ./3rdParty/
  echo "Copy to gc-%GC_VER%/libatomic_ops"  
  xcopy  /E /H /C /I /Y .\3rdParty\libatomic_ops-%LIBATOMIC_OPS_VER%\ .\3rdParty\gc-%GC_VER%\libatomic_ops\
  cd %p%
  @call scripts\build_gc_%BUILD%_%TOOL%.bat
  rem build_gc_*_vs.bat runs "cmake --install" which installs the libs to
  rem 3rdParty\gc\x64\%BUILD%\lib (where tsc now links from) - no extra copy needed
)