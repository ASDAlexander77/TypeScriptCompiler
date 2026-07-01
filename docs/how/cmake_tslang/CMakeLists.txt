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

# .ts files compile with TSLANG command; .cpp with the C++ compiler.
add_executable(${PROJECT_NAME}
    main.cpp
    mycode.ts
)

# required libs
target_link_libraries(${PROJECT_NAME} "gc" "LLVMSupport" "TypeScriptAsyncRuntime" "TypeScriptDefaultLib")

# ntdll provides RtlGetLastNtStatus (pulled in by LLVMSupport) on Windows
if(WIN32)
    target_link_libraries(${PROJECT_NAME} "ntdll")
endif()
