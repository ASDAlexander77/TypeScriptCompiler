cmake_minimum_required(VERSION 3.20)

# Make CMake find your custom-language modules
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

project(Demo CXX)

# Enable your custom language (runs the Determine/Information modules)
enable_language(FOO)

# .foo files compile with YOUR command; .cpp with the C++ compiler.
add_executable(app
    main.cpp
    mycode.foo
)
