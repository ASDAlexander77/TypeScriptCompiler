cmake_minimum_required(VERSION 3.20)

# Make CMake find your custom-language modules
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

project(Demo CXX)

# Enable TS-language
enable_language(TSLANG)

# .ts files compile with TSLANG command; .cpp with the C++ compiler.
add_executable(app
    main.cpp
    mycode.ts
)
