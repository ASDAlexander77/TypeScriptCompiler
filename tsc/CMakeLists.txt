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

project(tsc LANGUAGES CXX C)

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
  if (MSVC)
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
  if (MSVC)
    #file(REAL_PATH "${CMAKE_SOURCE_DIR}/../3rdParty/llvm/${CMAKE_BUILD_TYPE}/lib/cmake/clang" Clang_DIR)
    get_filename_component(Clang_DIR "${CMAKE_SOURCE_DIR}/../3rdParty/llvm/x64/${CMAKE_BUILD_TYPE}/lib/cmake/clang" REALPATH BASE_DIR "${CMAKE_SOURCE_DIR}")
  else()
    #file(REAL_PATH "${CMAKE_SOURCE_DIR}/../3rdParty/llvm/${CMAKE_BUILD_TYPE}/lib/cmake/clang" Clang_DIR)
    string(TOLOWER "${CMAKE_BUILD_TYPE}" CMAKE_BUILD_TYPE_LOWERCASE)
    get_filename_component(Clang_DIR "${CMAKE_SOURCE_DIR}/../3rdParty/llvm/${CMAKE_BUILD_TYPE_LOWERCASE}/lib/cmake/clang" REALPATH BASE_DIR "${CMAKE_SOURCE_DIR}")
  endif()
endif()
message(STATUS "Clang_DIR is ${Clang_DIR}")

if ("${PYTHON_EXECUTABLE}" STREQUAL "" OR "${PYTHON_EXECUTABLE}" STREQUAL "PYTHON_EXECUTABLE-NOTFOUND")
set(PYTHON_EXECUTABLE "python.exe")
endif()

message(STATUS "CMAKE_VS_PLATFORM_TOOLSET_HOST_ARCHITECTURE was ${CMAKE_VS_PLATFORM_TOOLSET_HOST_ARCHITECTURE} and set to x64")
set(CMAKE_VS_PLATFORM_TOOLSET_HOST_ARCHITECTURE "x64")

find_package(MLIR REQUIRED CONFIG)
if (NOT MLIR_FOUND)
    message(FATAL_ERROR "Failed to find MLIR")
endif()

find_package(Clang REQUIRED PATHS "${CLANG_DIR_ABSOLUTE}" NO_DEFAULT_PATH)
if (NOT Clang_FOUND)
    message(FATAL_ERROR "Failed to find Clang")
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
add_definitions(${LLVM_DEFINITIONS})

if (WIN32)
else()
	# linux
	add_definitions(-D_SILENCE_NONFLOATING_COMPLEX_DEPRECATION_WARNING)
endif()

# enabling testing
enable_testing()

######

add_subdirectory(include)
add_subdirectory(lib)
add_subdirectory(test)
add_subdirectory(tsc)
add_subdirectory(tsc-new-parser)

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
