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

project(tsc LANGUAGES CXX C)

# Add path for custom modules
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

include(config_macros)

set(CMAKE_CXX_STANDARD 17 CACHE STRING "C++ standard to conform to")
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)
set(CMAKE_CXX_EXTENSIONS OFF)

##### LLVM config ########

# set vars
if ("${CMAKE_BUILD_TYPE}" STREQUAL "")
set(CMAKE_BUILD_TYPE "Debug")
endif()

if ("${MLIR_DIR}" STREQUAL "" OR "${MLIR_DIR}" STREQUAL "MLIR_DIR-NOTFOUND")
  if (MSVC)
    #file(REAL_PATH "${CMAKE_SOURCE_DIR}/../3rdParty/llvm/${CMAKE_BUILD_TYPE}/lib/cmake/mlir" MLIR_DIR)
    get_filename_component(MLIR_DIR "${CMAKE_SOURCE_DIR}/../3rdParty/llvm/${CMAKE_BUILD_TYPE}/lib/cmake/mlir" REALPATH BASE_DIR "${CMAKE_SOURCE_DIR}")
  else()
    #file(REAL_PATH "${CMAKE_SOURCE_DIR}/../3rdParty/llvm-ninja/${CMAKE_BUILD_TYPE}/lib/cmake/mlir" MLIR_DIR)
    get_filename_component(MLIR_DIR "${CMAKE_SOURCE_DIR}/../3rdParty/llvm-ninja/${CMAKE_BUILD_TYPE}/lib/cmake/mlir" REALPATH BASE_DIR "${CMAKE_SOURCE_DIR}")
  endif()
endif()
message(STATUS "MLIR_DIR is ${MLIR_DIR}")

if ("${PYTHON_EXECUTABLE}" STREQUAL "" OR "${PYTHON_EXECUTABLE}" STREQUAL "PYTHON_EXECUTABLE-NOTFOUND")
set(PYTHON_EXECUTABLE "python.exe")
endif()

message(STATUS "CMAKE_VS_PLATFORM_TOOLSET_HOST_ARCHITECTURE was ${CMAKE_VS_PLATFORM_TOOLSET_HOST_ARCHITECTURE} and set to x64")
set(CMAKE_VS_PLATFORM_TOOLSET_HOST_ARCHITECTURE "x64")

find_package(MLIR REQUIRED CONFIG)

message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")

set(LLVM_RUNTIME_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/bin)
set(LLVM_LIBRARY_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/lib)
set(MLIR_BINARY_DIR ${CMAKE_BINARY_DIR})

set(LLVM_ENABLE_EH "ON")
set(LLVM_ENABLE_RTTI "ON")
set(LLVM_REQUIRES_RTTI "ON")
set(LLVM_ENABLE_PIC "ON")

list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")
include(TableGen)
include(AddLLVM)
include(AddMLIR)
include(HandleLLVMOptions)

include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})
include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${PROJECT_BINARY_DIR}/include)
link_directories(${LLVM_BUILD_LIBRARY_DIR})
add_definitions(${LLVM_DEFINITIONS})

# enabling testing
enable_testing()

######

add_subdirectory(include)
add_subdirectory(lib)
add_subdirectory(test)
add_subdirectory(tsc)
add_subdirectory(tsc-opt)
add_subdirectory(tsc-translate)
add_subdirectory(tsc-new-parser)

