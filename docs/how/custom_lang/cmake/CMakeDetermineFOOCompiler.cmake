# Locate your custom compiler
find_program(CMAKE_FOO_COMPILER
    NAMES fooc fooc.exe
    HINTS "${CMAKE_SOURCE_DIR}/tools"
    DOC "Custom FOO compiler")

mark_as_advanced(CMAKE_FOO_COMPILER)

# Which source extensions belong to FOO, and the object suffix
set(CMAKE_FOO_SOURCE_FILE_EXTENSIONS foo)
set(CMAKE_FOO_OUTPUT_EXTENSION .obj)   # .o on Linux
set(CMAKE_FOO_COMPILER_ENV_VAR "FOOC")

# Emit the compiler-id config file CMake expects
configure_file(
    ${CMAKE_CURRENT_LIST_DIR}/CMakeFOOCompiler.cmake.in
    ${CMAKE_PLATFORM_INFO_DIR}/CMakeFOOCompiler.cmake @ONLY)
