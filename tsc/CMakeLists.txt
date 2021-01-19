cmake_minimum_required(VERSION 3.19.2)

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
set(CMAKE_MODULE_PATH
  ${CMAKE_MODULE_PATH}
  "${CMAKE_CURRENT_SOURCE_DIR}/cmake"
  )

set(CMAKE_CXX_STANDARD 14 CACHE STRING "C++ standard to conform to")

##### LLVM config ########

# set vars
if ("${CMAKE_BUILD_TYPE}" STREQUAL "")
set(CMAKE_BUILD_TYPE "Debug")
endif()

if ("${LLVM_EXTERNAL_LIT}" STREQUAL "" OR "${LLVM_EXTERNAL_LIT}" STREQUAL "LLVM_EXTERNAL_LIT-NOTFOUND")
file(REAL_PATH "${CMAKE_SOURCE_DIR}/../__build/llvm/${CMAKE_BUILD_TYPE}/bin/llvm-lit.py" LLVM_EXTERNAL_LIT)
message(STATUS "LLVM_EXTERNAL_LIT is ${LLVM_EXTERNAL_LIT}")
endif()

if ("${MLIR_DIR}" STREQUAL "" OR "${MLIR_DIR}" STREQUAL "MLIR_DIR-NOTFOUND")
file(REAL_PATH "${CMAKE_SOURCE_DIR}/../3rdParty/llvm/${CMAKE_BUILD_TYPE}/lib/cmake/mlir" MLIR_DIR)
message(STATUS "MLIR_DIR is ${MLIR_DIR}")
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

###### ANTLR4 config ########

# required if linking to static library
add_definitions(-DANTLR4CPP_STATIC)

# using /MD flag for antlr4_runtime (for Visual C++ compilers only)
set(ANTLR4_WITH_STATIC_CRT OFF)

# set variable pointing to the antlr tool that supports C++
# this is not required if the jar file can be found under PATH environment
file(REAL_PATH "${CMAKE_SOURCE_DIR}/../3rdParty/antlr-bin/antlr-4.9.1-complete.jar" ANTLR_EXECUTABLE)
message(STATUS "ANTLR_EXECUTABLE is ${ANTLR_EXECUTABLE}, Java executable is ${Java_JAVA_EXECUTABLE}")

# Set path to generator
set(ANTLR4_JAR_LOCATION ${ANTLR_EXECUTABLE})

file(REAL_PATH "${CMAKE_SOURCE_DIR}/../3rdParty/antlr4/${CMAKE_BUILD_TYPE}/lib/cmake/antlr4-runtime" antlr4-runtime_DIR)
file(REAL_PATH "${CMAKE_SOURCE_DIR}/../3rdParty/antlr4/${CMAKE_BUILD_TYPE}/lib/cmake/antlr4-generator" antlr4-generator_DIR)

# Bring in the required packages
find_package(antlr4-runtime REQUIRED)
find_package(antlr4-generator REQUIRED)

# add antrl4cpp artifacts to project environment
include_directories(${ANTLR4_INCLUDE_DIR})
link_directories(${ANTLR4_LIB_DIR})

include(grammar_generator)

# enabling testing
enable_testing()

######

add_subdirectory(include)
add_subdirectory(lib)
add_subdirectory(test)
add_subdirectory(tsc)
add_subdirectory(tsc-opt)
add_subdirectory(tsc-translate)

