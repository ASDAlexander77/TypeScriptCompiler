# Builds only the library compiled programs link - TypeScriptAsyncRuntime - for any target arch,
# without MLIR or LLVM. The main project cannot do this for x86: it needs the x64-only MLIR and Clang
# packages. The runtime needs neither: it includes one header-only MLIR file for the C ABI it
# implements, and gc.h. Sources are shared with lib/TypeScriptAsyncRuntime, so the two cannot drift.
#
#   cmake -S tslang/runtime -B <dir> -A Win32 -DTSLANG_GC_INCLUDE=<gc>/include -DTSLANG_MLIR_INCLUDE=<llvm>/include
cmake_minimum_required(VERSION 3.20)
project(TypeScriptRuntimeLibs CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
# Static CRT, like everything else linked into a compiled program (see exe.cpp).
set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")

set(TSLANG_GC_INCLUDE "" CACHE PATH "Boehm include directory for the target arch")
set(TSLANG_MLIR_INCLUDE "" CACHE PATH "Directory holding mlir/ExecutionEngine/AsyncRuntime.h")
if (NOT EXISTS "${TSLANG_GC_INCLUDE}/gc.h")
  message(FATAL_ERROR "TSLANG_GC_INCLUDE must point at a directory with gc.h, got '${TSLANG_GC_INCLUDE}'")
endif()
if (NOT EXISTS "${TSLANG_MLIR_INCLUDE}/mlir/ExecutionEngine/AsyncRuntime.h")
  message(FATAL_ERROR "TSLANG_MLIR_INCLUDE must hold mlir/ExecutionEngine/AsyncRuntime.h, got '${TSLANG_MLIR_INCLUDE}'")
endif()

set(SRC "${CMAKE_CURRENT_SOURCE_DIR}/../lib/TypeScriptAsyncRuntime")
add_library(TypeScriptAsyncRuntime STATIC "${SRC}/AsyncRuntime.cpp" "${SRC}/DynamicRuntime.cpp")
target_include_directories(TypeScriptAsyncRuntime PRIVATE
  "${CMAKE_CURRENT_SOURCE_DIR}/../include"
  "${CMAKE_CURRENT_SOURCE_DIR}/../lib"
  "${TSLANG_GC_INCLUDE}"
  "${TSLANG_MLIR_INCLUDE}")
if (MSVC)
  # The same definitions the in-tree add_mlir_library build compiles with.
  target_compile_definitions(TypeScriptAsyncRuntime PRIVATE
    WIN32 _WINDOWS UNICODE _UNICODE
    _CRT_SECURE_NO_DEPRECATE _CRT_SECURE_NO_WARNINGS _CRT_NONSTDC_NO_DEPRECATE _CRT_NONSTDC_NO_WARNINGS
    _SCL_SECURE_NO_DEPRECATE _SCL_SECURE_NO_WARNINGS
    __STDC_CONSTANT_MACROS __STDC_FORMAT_MACROS __STDC_LIMIT_MACROS)
  target_compile_options(TypeScriptAsyncRuntime PRIVATE /GR /EHsc)
endif()

install(TARGETS TypeScriptAsyncRuntime ARCHIVE DESTINATION .)
