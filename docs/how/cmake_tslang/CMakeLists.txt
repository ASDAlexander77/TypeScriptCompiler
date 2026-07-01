cmake_minimum_required(VERSION 3.20)

# Make CMake find ts-language modules
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

project(Demo CXX)

# Enable TS-language
enable_language(TSLANG)

# Include folders
include_directories(${CMAKE_TSLANG_DIR}/defaultlib)

# Lib folders
link_directories(${CMAKE_TSLANG_DIR} ${CMAKE_TSLANG_DIR}/defaultlib/lib)

# set options
if (CMAKE_BUILD_TYPE STREQUAL "Release")
	set(CMAKE_TSLANG_FLAGS "--opt --opt_level=3") # global
else()
	set(CMAKE_TSLANG_FLAGS "--di --opt_level=0") # global
endif()

if(WIN32)
else()
    list(APPEND CMAKE_TSLANG_FLAGS "-relocation-model=pic") # global    
endif()

# .ts files compile with TSLANG command; .cpp with the C++ compiler.
add_executable(${PROJECT_NAME}
    main.cpp
    mycode.ts
    adder.ts
)

# required libs
set(TSLANG_LINK_LIBS "gc" "LLVMSupport" "TypeScriptAsyncRuntime" "TypeScriptDefaultLib")

# ntdll provides RtlGetLastNtStatus (pulled in by LLVMSupport) on Windows
if(WIN32)
    list(APPEND TSLANG_LINK_LIBS "ntdll")
else()
    list(APPEND TSLANG_LINK_LIBS "LLVMDemangle" "stdc++" "m" "pthread" "tinfo" "dl" "rt")
endif()

target_link_libraries(${PROJECT_NAME} ${TSLANG_LINK_LIBS})
