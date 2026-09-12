cmake_minimum_required(VERSION 3.17.3)

if(POLICY CMP0068)
  cmake_policy(SET CMP0068 NEW)
  set(CMAKE_BUILD_WITH_INSTALL_NAME_DIR ON)
endif()

if(POLICY CMP0075)
  cmake_policy(SET CMP0075 NEW)
endif()

if(POLICY CMP0077)
  cmake_policy(SET CMP0077 NEW)
endif()

if(POLICY CMP0116)
  cmake_policy(SET CMP0116 NEW)
endif()

project(tslang LANGUAGES CXX C)

find_program(CCACHE_FOUND ccache)
if(CCACHE_FOUND)
    set_property(GLOBAL PROPERTY RULE_LAUNCH_COMPILE ccache)
    set_property(GLOBAL PROPERTY RULE_LAUNCH_LINK ccache) # Less useful to do it for linking, see edit2
endif(CCACHE_FOUND)

# Add path for custom modules
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

include(config_macros)

set(CMAKE_CXX_STANDARD 17 CACHE STRING "C++ standard to conform to")
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
#set(CMAKE_CXX_CLANG_TIDY "clang-tidy;-checks=*")

##### LLVM config ########

# set vars
if ("${CMAKE_BUILD_TYPE}" STREQUAL "")
set(CMAKE_BUILD_TYPE "Debug")
endif()

if ("${MLIR_DIR}" STREQUAL "" OR "${MLIR_DIR}" STREQUAL "MLIR_DIR-NOTFOUND")
  if (MSVC OR WIN32)
    #file(REAL_PATH "${CMAKE_SOURCE_DIR}/../3rdParty/llvm/${CMAKE_BUILD_TYPE}/lib/cmake/mlir" MLIR_DIR)
    get_filename_component(MLIR_DIR "${CMAKE_SOURCE_DIR}/../3rdParty/llvm/x64/${CMAKE_BUILD_TYPE}/lib/cmake/mlir" REALPATH BASE_DIR "${CMAKE_SOURCE_DIR}")
  else()
    #file(REAL_PATH "${CMAKE_SOURCE_DIR}/../3rdParty/llvm/${CMAKE_BUILD_TYPE}/lib/cmake/mlir" MLIR_DIR)
    string(TOLOWER "${CMAKE_BUILD_TYPE}" CMAKE_BUILD_TYPE_LOWERCASE)
    get_filename_component(MLIR_DIR "${CMAKE_SOURCE_DIR}/../3rdParty/llvm/${CMAKE_BUILD_TYPE_LOWERCASE}/lib/cmake/mlir" REALPATH BASE_DIR "${CMAKE_SOURCE_DIR}")
  endif()
endif()
message(STATUS "MLIR_DIR is ${MLIR_DIR}")

if ("${Clang_DIR}" STREQUAL "" OR "${Clang_DIR}" STREQUAL "Clang_DIR-NOTFOUND")
  if (MSVC OR WIN32)
    #file(REAL_PATH "${CMAKE_SOURCE_DIR}/../3rdParty/llvm/${CMAKE_BUILD_TYPE}/lib/cmake/clang" Clang_DIR)
    get_filename_component(Clang_DIR "${CMAKE_SOURCE_DIR}/../3rdParty/llvm/x64/${CMAKE_BUILD_TYPE}/lib/cmake/clang" REALPATH BASE_DIR "${CMAKE_SOURCE_DIR}")
  else()
    #file(REAL_PATH "${CMAKE_SOURCE_DIR}/../3rdParty/llvm/${CMAKE_BUILD_TYPE}/lib/cmake/clang" Clang_DIR)
    string(TOLOWER "${CMAKE_BUILD_TYPE}" CMAKE_BUILD_TYPE_LOWERCASE)
    get_filename_component(Clang_DIR "${CMAKE_SOURCE_DIR}/../3rdParty/llvm/${CMAKE_BUILD_TYPE_LOWERCASE}/lib/cmake/clang" REALPATH BASE_DIR "${CMAKE_SOURCE_DIR}")
  endif()
endif()
message(STATUS "Clang_DIR is ${Clang_DIR}")

if ("${BDWgc_DIR}" STREQUAL "" OR "${BDWgc_DIR}" STREQUAL "BDWgc_DIR-NOTFOUND")
  string(TOLOWER "${CMAKE_BUILD_TYPE}" CMAKE_BUILD_TYPE_LOWERCASE)
  if (MSVC OR WIN32)
    get_filename_component(BDWgc_DIR "${CMAKE_SOURCE_DIR}/../3rdParty/gc/x64/${CMAKE_BUILD_TYPE_LOWERCASE}/lib/cmake/bdwgc" REALPATH BASE_DIR "${CMAKE_SOURCE_DIR}")
  else()
    get_filename_component(BDWgc_DIR "${CMAKE_SOURCE_DIR}/../3rdParty/gc/${CMAKE_BUILD_TYPE_LOWERCASE}/lib/cmake/bdwgc" REALPATH BASE_DIR "${CMAKE_SOURCE_DIR}")
  endif()
endif()
message(STATUS "BDWgc_DIR is ${BDWgc_DIR}")

if ("${PYTHON_EXECUTABLE}" STREQUAL "" OR "${PYTHON_EXECUTABLE}" STREQUAL "PYTHON_EXECUTABLE-NOTFOUND")
set(PYTHON_EXECUTABLE "python.exe")
endif()

message(STATUS "CMAKE_VS_PLATFORM_TOOLSET_HOST_ARCHITECTURE was [${CMAKE_VS_PLATFORM_TOOLSET_HOST_ARCHITECTURE}], set to x64")
set(CMAKE_VS_PLATFORM_TOOLSET_HOST_ARCHITECTURE "x64")

find_package(MLIR REQUIRED CONFIG)
if (NOT MLIR_FOUND)
    message(FATAL_ERROR "Failed to find MLIR")
endif()

find_package(Clang REQUIRED PATHS "${CLANG_DIR_ABSOLUTE}" NO_DEFAULT_PATH)
if (NOT Clang_FOUND)
    message(FATAL_ERROR "Failed to find Clang")
endif()

# bdwgc ships a CMake package (BDWgc::gc / BDWgc::cord imported targets) that
# carries the include dir and the per-config static lib location.
find_package(BDWgc REQUIRED CONFIG)
if (NOT BDWgc_FOUND)
    message(FATAL_ERROR "Failed to find BDWgc (garbage collector)")
endif()

# Boehm built as a DLL (gc.dll + its import library gc.lib). One process must have one
# collector: every Boehm linked statically brings its own heap and roots, and a collector frees
# objects that only another binary's memory still references. So anything that shares a process
# with other gc binaries - TypeScriptRuntime.dll under the JIT, the default library's DLL, user
# shared libraries - takes Boehm from gc.dll. See docs/single-gc-collector-design.md.
# Built by scripts/build_gc_{release,debug}_shared_vs.bat (prepare_3rdParty.bat calls it).
string(TOLOWER "${CMAKE_BUILD_TYPE}" CMAKE_BUILD_TYPE_LOWERCASE)
if ("${TSLANG_GC_SHARED_PREFIX}" STREQUAL "")
  if (MSVC OR WIN32)
    get_filename_component(TSLANG_GC_SHARED_PREFIX "${CMAKE_SOURCE_DIR}/../3rdParty/gcdll/x64/${CMAKE_BUILD_TYPE_LOWERCASE}" REALPATH BASE_DIR "${CMAKE_SOURCE_DIR}")
  else()
    get_filename_component(TSLANG_GC_SHARED_PREFIX "${CMAKE_SOURCE_DIR}/../3rdParty/gcdll/${CMAKE_BUILD_TYPE_LOWERCASE}" REALPATH BASE_DIR "${CMAKE_SOURCE_DIR}")
  endif()
endif()
message(STATUS "TSLANG_GC_SHARED_PREFIX is ${TSLANG_GC_SHARED_PREFIX}")

if (WIN32)
    if (NOT EXISTS "${TSLANG_GC_SHARED_PREFIX}/lib/gc.lib" OR NOT EXISTS "${TSLANG_GC_SHARED_PREFIX}/bin/gc.dll")
        message(FATAL_ERROR "shared Boehm (gc.dll) not found at ${TSLANG_GC_SHARED_PREFIX}. "
                            "TypeScriptRuntime.dll needs it so a JIT process has one collector. "
                            "Build it with scripts/build_gc_${CMAKE_BUILD_TYPE_LOWERCASE}_shared_vs.bat, "
                            "or point TSLANG_GC_SHARED_PREFIX at an existing install.")
    endif()

    # Same headers as the static package; GC_DLL makes them declare the API dllimport.
    get_target_property(TSLANG_GC_INCLUDE_DIRS BDWgc::gc INTERFACE_INCLUDE_DIRECTORIES)
    add_library(tslang_gc_shared SHARED IMPORTED GLOBAL)
    set_target_properties(tslang_gc_shared PROPERTIES
        IMPORTED_IMPLIB "${TSLANG_GC_SHARED_PREFIX}/lib/gc.lib"
        IMPORTED_LOCATION "${TSLANG_GC_SHARED_PREFIX}/bin/gc.dll"
        INTERFACE_INCLUDE_DIRECTORIES "${TSLANG_GC_INCLUDE_DIRS}"
        INTERFACE_COMPILE_DEFINITIONS "GC_DLL")
endif()

message(STATUS "Using MLIRConfig.cmake in: ${MLIR_CMAKE_DIR}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_CMAKE_DIR}")
message(STATUS "Using ClangConfig.cmake in: ${CLANG_CMAKE_DIR}")

set(LLVM_RUNTIME_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/bin)
set(LLVM_LIBRARY_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/lib)
set(MLIR_BINARY_DIR ${CMAKE_BINARY_DIR})

set(LLVM_ENABLE_EH "ON")
set(LLVM_ENABLE_RTTI "ON")
set(LLVM_REQUIRES_RTTI "ON")
set(LLVM_ENABLE_PIC "ON")

# clang fix
set(CMAKE_BUILD_WITH_INSTALL_RPATH "ON")

if(MSVC)
  # FIXME: In fact, sanitizers should support both /MT and /MD, see PR20214.
  # Use the static CRT (/MT, /MTd) to match the prebuilt static LLVM/MLIR libs.
  # Use a generator expression so it is correct for multi-config generators
  # (e.g. Visual Studio), where CMAKE_BUILD_TYPE is not meaningful at configure time.
  set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")

  # HandleLLVMOptions (included below) otherwise forces the dynamic CRT default and
  # would override CMAKE_MSVC_RUNTIME_LIBRARY. These tell it to select the static CRT
  # for every configuration so all targets (including unittests/gtest) match.
  set(LLVM_USE_CRT_DEBUG MTd)
  set(LLVM_USE_CRT_RELEASE MT)
  set(LLVM_USE_CRT_RELWITHDEBINFO MT)
  set(LLVM_USE_CRT_MINSIZEREL MT)

  message(STATUS "CMAKE_MSVC_RUNTIME_LIBRARY is ${CMAKE_MSVC_RUNTIME_LIBRARY}")
endif()

list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${CLANG_CMAKE_DIR}")
include(TableGen)
include(AddLLVM)
include(AddClang)
include(AddMLIR)
include(HandleLLVMOptions)

include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${CLANG_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})
include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${PROJECT_BINARY_DIR}/include)
link_directories(${LLVM_BUILD_LIBRARY_DIR})
# LLVM_DEFINITIONS is a single space-separated string (e.g.
# "-D_GNU_SOURCE -D_GLIBCXX_USE_CXX11_ABI=1 -D__STDC_CONSTANT_MACROS ...").
# Passing it to add_definitions() unsplit makes a single bogus definition where
# _GLIBCXX_USE_CXX11_ABI gets the value "1 -D__STDC_...", which then conflicts
# with the clean _GLIBCXX_USE_CXX11_ABI=1 emitted by HandleLLVMOptions and
# triggers a "redefined" error. Tokenize it into a proper list first.
separate_arguments(LLVM_DEFINITIONS_LIST NATIVE_COMMAND "${LLVM_DEFINITIONS}")
add_definitions(${LLVM_DEFINITIONS_LIST})
add_definitions(-D_SILENCE_NONFLOATING_COMPLEX_DEPRECATION_WARNING)

# enabling testing
enable_testing()

######

add_subdirectory(include)
add_subdirectory(lib)
add_subdirectory(test)
add_subdirectory(tslang)
add_subdirectory(ts-new-parser)

if (NOT DEFINED TYPESCRIPT_IGNORE_UNIT_TESTS)
  # enable unit tests
  set (MLIRTYPESCRIPT_LLVM_THIRD_PARTY_DIR "${CMAKE_SOURCE_DIR}/../3rdParty/llvm-project/third-party")
  message(STATUS "Using MLIRTYPESCRIPT_LLVM_THIRD_PARTY_DIR = ${MLIRTYPESCRIPT_LLVM_THIRD_PARTY_DIR}")
  set(UNITTEST_DIR ${MLIRTYPESCRIPT_LLVM_THIRD_PARTY_DIR}/unittest)
  if(EXISTS ${UNITTEST_DIR}/googletest/include/gtest/gtest.h)
    add_subdirectory(${UNITTEST_DIR} third-party/unittest)
  endif()
  
  add_definitions(-DTYPESCRIPT_INCLUDE_TESTS)
  add_custom_target(MLIRTypeScriptUnitTests ALL)
  if (EXISTS ${MLIRTYPESCRIPT_LLVM_THIRD_PARTY_DIR}/unittest/googletest/include/gtest/gtest.h)
    add_subdirectory(unittests)
  else()
    message(WARNING "!!! gtest not found, unittests will not be available !!!")
  endif()
endif()
