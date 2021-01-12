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
if ("${LLVM_EXTERNAL_LIT}" STREQUAL "" OR "${LLVM_EXTERNAL_LIT}" STREQUAL "LLVM_EXTERNAL_LIT-NOTFOUND")
file(REAL_PATH "${CMAKE_SOURCE_DIR}/../__build/llvm/debug/bin/llvm-lit.py" LLVM_EXTERNAL_LIT)
message(STATUS "LLVM_EXTERNAL_LIT is ${LLVM_EXTERNAL_LIT}")
endif()

if ("${MLIR_DIR}" STREQUAL "" OR "${MLIR_DIR}" STREQUAL "MLIR_DIR-NOTFOUND")
file(REAL_PATH "${CMAKE_SOURCE_DIR}/../3rdParty/llvm/debug/lib/cmake/mlir" MLIR_DIR)
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
# add external build for antlrcpp
include(ExternalAntlr4Cpp)
# add antrl4cpp artifacts to project environment
include_directories(${ANTLR4_INCLUDE_DIRS})

# set variable pointing to the antlr tool that supports C++
# this is not required if the jar file can be found under PATH environment
set(ANTLR_EXECUTABLE ${CMAKE_SOURCE_DIR}/../3rdParty/antlr-bin/antlr-4.9.1-complete.jar)
# add macros to generate ANTLR Cpp code from grammar
find_package(ANTLR REQUIRED)

###### 

add_subdirectory(include)
add_subdirectory(lib)
add_subdirectory(test)
add_subdirectory(tsc-grammar)
add_subdirectory(tsc)
add_subdirectory(tsc-opt)
add_subdirectory(tsc-translate)

